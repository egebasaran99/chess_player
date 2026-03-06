import chess
import random
import torch
import torch.nn.functional as F
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)

        self.model_id = "egeb9/chess-gpt2-midterm_new"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    def _candidate_moves(self, board: chess.Board, max_candidates: int = 16) -> list[str]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        priority = []
        for m in legal_moves:
            if board.is_capture(m) or board.gives_check(m) or (m.promotion is not None):
                priority.append(m.uci())

        seen = set(priority)
        others = [m.uci() for m in legal_moves if m.uci() not in seen]
        random.shuffle(others)

        candidates = priority + others
        return candidates[:max_candidates]

    def _score_move(self, prompt: str, move_uci: str) -> float:
        continuation = " " + move_uci

        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        full_ids = self.tokenizer(
            prompt + continuation, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        move_start = prompt_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits

        log_probs = F.log_softmax(logits[0], dim=-1)

        score = 0.0
        for i in range(move_start, full_ids.shape[1]):
            token_id = full_ids[0, i].item()
            score += log_probs[i - 1, token_id].item()

        return float(score)

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        if not legal_moves:
            return None

        try:
            self._load_model()
        except Exception:
            return random.choice(legal_moves)

        prompt = f"FEN: {fen}\nMove:"

        candidates = self._candidate_moves(board, max_candidates=12)

        best_move = None
        best_score = -1e30

        try:
            for move in candidates:
                score = self._score_move(prompt, move)
                if score > best_score:
                    best_score = score
                    best_move = move
        except Exception:
            return random.choice(legal_moves)

        if best_move in legal_moves:
            return best_move

        return random.choice(legal_moves)
