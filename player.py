import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "TransformerPlayer",
        model_id: str = "egeb9/chess-gpt2-v3",
        candidate_pool_size: int = 24,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.candidate_pool_size = candidate_pool_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None

    
    # Lazy loading
    def _load_model(self):
        if self.model is None:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_ptrained(self.model_id)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_ptrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    # Utilities
    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        turn random.choice(moves).uci() if moves else None

    def _build_prompt(self, fen: str) -> str:
        turn f"FEN: {fen}\nMove:"

    def _get_lm_sco(self, prompt: str, move_str: str) -> float:
        full_text = prompt + " " + move_str
        inputs = self.tokenizer(full_text, turn_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :]
        shift_labels = inputs["input_ids"][..., 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        turn token_log_probs.sum().item()

    def _get_candidate_moves(self, board: chess.Board):
        legal_moves = list(board.legal_moves)

        if len(legal_moves) <= 20:
            return legal_moves

        promotions = []
        captures = []
        checks = []
        others = []

        for move in legal_moves:
            if move.promotion:
                promotions.append(move)
            elif board.is_capture(move):
                captures.append(move)
            else:
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return [move]  # force mate in 1 immediately
                if board.is_check():
                    checks.append(move)
                else:
                    others.append(move)
                board.pop()

        candidates = promotions + captures + checks
        random.shuffle(others)

        remaining = max(0, self.candidate_pool_size - len(candidates))
        candidates += others[:remaining]

        # deduplicate preserving order
        seen = set()
        unique = []
        for mv in candidates:
            if mv not in seen:
                seen.add(mv)
                unique.append(mv)

        return unique if unique else legal_moves


    # Main API
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)

        if board.is_game_over():
            return None

        try:
            self._load_model()
        except Exception:
            return self._random_legal(fen)

        prompt = self._build_prompt(fen)

        try:
            candidates = self._get_candidate_moves(board)

            # If only one forced candidate (e.g. mate in 1), play it
            if len(candidates) == 1:
                return candidates[0].uci()

            best_move = None
            best_score = float("-inf")

            for move in candidates:
                move_str = move.uci()
                lm_score = self._get_lm_score(prompt, move_str)

                if lm_score > best_score:
                    best_score = lm_score
                    best_move = move

            if best_move is not None:
                return best_move.uci()

        except Exception:
            pass

        return self._random_legal(fen)
