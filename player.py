import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    Tiny LM baseline chess player.

    REQUIRED:
        Subclasses chess_tournament.players.Player
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(self, name: str = "TinyLMPlayer"):
           
        super().__init__(name)

        self.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        self.temperature = 0.7
        self.max_new_tokens = 8

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded components
        self.tokenizer = None
        self.model = None

    # -------------------------
    # Lazy loading
    # -------------------------
    def _load_model(self):
        if self.model is None:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        # Limit how many moves we include so the prompt doesn't get huge
        legal_moves = legal_moves[:40]

        return (
            "You are a chess assistant.\n"
            "Choose ONE move from the provided legal moves list.\n"
            f"FEN: {fen}\n"
            f"Legal moves: {' '.join(legal_moves)}\n"
            "Move (UCI only):"
        )

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    def _score_move(self, prompt: str, move_uci: str) -> float:
        """
        Returns a score (log-probability) for choosing `move_uci` as the next text after `prompt`.
        Higher = the model likes that move more.
        """
        # We include a leading space to help many tokenizers separate words
        continuation = " " + move_uci

        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        full_ids = self.tokenizer(prompt + continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        # The move tokens are the extra tokens after the prompt
        move_token_start = prompt_ids.shape[1]
        move_token_ids = full_ids[0, move_token_start:]

        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits  # shape: [1, seq_len, vocab]

        # logits at position t predicts token t+1
        # So to score token at position i, use logits at i-1
        log_probs = F.log_softmax(logits[0], dim=-1)

        score = 0.0
        for i in range(move_token_start, full_ids.shape[1]):
            token_id = full_ids[0, i].item()
            score += log_probs[i - 1, token_id].item()

        return float(score)

    def _candidate_moves(self, board: chess.Board, max_candidates: int = 20) -> list[str]:
        """
        Builds a small list of candidate legal moves.
        Priority: captures, checks, promotions. Then fill randomly up to max_candidates.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        good = []
        for m in legal_moves:
            if board.is_capture(m) or board.gives_check(m) or (m.promotion is not None):
                good.append(m)

        # Remove duplicates while keeping order
        seen = set()
        good_unique = []
        for m in good:
            u = m.uci()
            if u not in seen:
                seen.add(u)
                good_unique.append(u)

        # Fill with random legal moves until we have enough
        all_uci = [m.uci() for m in legal_moves]
        random.shuffle(all_uci)

        candidates = good_unique[:]
        for u in all_uci:
            if len(candidates) >= max_candidates:
                break
            if u not in seen:
                seen.add(u)
                candidates.append(u)

        # If still empty for some reason, just take a few legal moves
        if not candidates:
            candidates = all_uci[:min(max_candidates, len(all_uci))]

        return candidates

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return None

        try:
            self._load_model()
        except Exception:
            return random.choice(legal_moves)

        # Prompt for scoring (short and stable)
        prompt = (
            "You are a strong chess assistant.\n"
            "Given the position, choose the best move.\n"
            f"FEN: {fen}\n"
            "Best move:"
        )

        # Candidate list (small set)
        candidates = self._candidate_moves(board, max_candidates=20)

        # Score each candidate and take the best
        best_move = None
        best_score = -1e30

        try:
            for m in candidates:
                s = self._score_move(prompt, m)
                if s > best_score:
                    best_score = s
                    best_move = m
        except Exception:
            # If anything goes wrong with scoring, fall back safely
            return random.choice(legal_moves)

        # Final safety: ensure legal
        if best_move in set(legal_moves):
            return best_move

        return random.choice(legal_moves)
