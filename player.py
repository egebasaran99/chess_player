import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    GPT-2 based chess player for the midterm assignment.

    - Inherits from chess_tournament.players.Player
    - Can be initialized with only the player name
    - Uses a public Hugging Face model by default
    - Keeps rule-based logic light: legality + mild candidate prioritization
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "TransformerPlayer",
        model_id: str = "egeb9/chess-gpt2-midterm_new",
        temperature: float = 0.7,
        max_new_tokens: int = 8,
        tactical_weight: float = 0.08,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tactical_weight = tactical_weight

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy loading
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
    # Utilities
    # -------------------------
    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    def _build_prompt(self, fen: str) -> str:
        return f"FEN: {fen}\nMove:"

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    def _get_lm_score(self, prompt: str, move_str: str) -> float:
        """
        Scores a candidate move by computing its conditional log-probability
        under the language model.
        """
        full_text = prompt + " " + move_str
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :]
        shift_labels = inputs["input_ids"][..., 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.sum().item()

    def _score_move_tactical(self, board: chess.Board, move: chess.Move) -> float:
        """
        Very light normalized tactical tie-break score.
        Intended to stay small so the LM remains primary.
        Approximate range: [-1, 1]
        """
        score = 0.0

        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                value_map = {
                    chess.PAWN: 1,
                    chess.KNIGHT: 3,
                    chess.BISHOP: 3,
                    chess.ROOK: 5,
                    chess.QUEEN: 9,
                }
                score += value_map.get(captured_piece.piece_type, 0) / 10.0

        if move.promotion:
            score += 0.3

        board.push(move)
        if board.is_checkmate():
            score += 1.0
        elif board.is_check():
            score += 0.2
        board.pop()

        return max(-1.0, min(1.0, score))

    def _get_candidate_moves(self, board: chess.Board):
        """
        Light candidate filtering:
        - if few legal moves, keep all
        - otherwise prioritize promotions, captures, and checking moves
        - then add a few random remaining legal moves
        """
        legal_moves = list(board.legal_moves)

        if len(legal_moves) <= 8:
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
                if board.is_check():
                    checks.append(move)
                else:
                    others.append(move)
                board.pop()

        candidates = promotions + captures + checks

        remaining = 12 - len(candidates)
        if remaining > 0:
            random.shuffle(others)
            candidates += others[:remaining]

        if not candidates:
            return legal_moves

        return candidates

    # -------------------------
    # Main API
    # -------------------------
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

            best_move = None
            best_score = float("-inf")

            for move in candidates:
                move_str = move.uci()

                lm_score = self._get_lm_score(prompt, move_str)
                tactical_score = self._score_move_tactical(board, move)

                total_score = lm_score + self.tactical_weight * tactical_score

                if total_score > best_score:
                    best_score = total_score
                    best_move = move

            if best_move is not None:
                return best_move.uci()

        except Exception:
            pass

        return self._random_legal(fen)
