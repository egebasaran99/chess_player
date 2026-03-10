import chess
import random
import torch
import torch.nn.functional as F
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


class TransformerPlayer(Player):
    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)

        self.model_id = "egeb9/chess-gpt2-v3"
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

    def _random_legal(self, board: chess.Board) -> Optional[str]:
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    def _material_score(self, board: chess.Board, color: bool) -> float:
        score = 0.0
        for piece_type, value in PIECE_VALUES.items():
            score += len(board.pieces(piece_type, color)) * value
            score -= len(board.pieces(piece_type, not color)) * value
        return score

    def _score_move_lm(self, prompt: str, move_uci: str) -> float:
        """
        Score how likely the model thinks this move is after the prompt.
        Higher is better.
        """
        continuation = " " + move_uci

        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.device)

        full_ids = self.tokenizer(
            prompt + continuation,
            return_tensors="pt",
            add_special_tokens=False
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

    def _score_move_tactical(self, board: chess.Board, move: chess.Move) -> float:
        """
        Simple chess-based score to make the player less passive.
        """
        color = board.turn
        before_material = self._material_score(board, color)

        new_board = board.copy()
        new_board.push(move)

        # Winning immediately is huge
        if new_board.is_checkmate():
            return 1000.0

        score = 0.0

        # Checks are good
        if new_board.is_check():
            score += 1.5

        # Promotions are very strong
        if move.promotion is not None:
            score += 8.0

        # Reward material improvement
        after_material = self._material_score(new_board, color)
        score += 3.0 * (after_material - before_material)

        # Penalize moves that allow a strong immediate reply
        worst_reply_delta = 0.0

        for opp_move in new_board.legal_moves:
            reply_board = new_board.copy()
            reply_board.push(opp_move)

            our_material_after_reply = self._material_score(reply_board, color)
            delta = our_material_after_reply - after_material

            if delta < worst_reply_delta:
                worst_reply_delta = delta

        score += 2.0 * worst_reply_delta

        return score

    def _candidate_moves(self, board: chess.Board, max_candidates: int = 20) -> list[str]:
        """
        Build a legal candidate set.
        Priority:
        - captures
        - checks
        - promotions
        - castling
        - then other moves
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        priority = []
        others = []

        for move in legal_moves:
            u = move.uci()

            if (
                board.is_capture(move)
                or board.gives_check(move)
                or move.promotion is not None
                or board.is_castling(move)
            ):
                priority.append(u)
            else:
                others.append(u)

        # Small opening bias toward reasonable developing moves
        if board.fullmove_number <= 10:
            opening_like = {
                "e2e4", "d2d4", "c2c4", "g1f3", "b1c3",
                "e7e5", "d7d5", "c7c5", "g8f6", "b8c6",
                "f1c4", "f8c5", "f1b5", "f8b4",
                "e1g1", "e8g8", "e1c1", "e8c8"
            }

            preferred = []
            rest = []
            for u in others:
                if u in opening_like:
                    preferred.append(u)
                else:
                    rest.append(u)

            others = preferred + rest

        random.shuffle(others)

        candidates = []
        seen = set()

        for u in priority + others:
            if u not in seen:
                seen.add(u)
                candidates.append(u)

            if len(candidates) >= max_candidates:
                break

        return candidates

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)

        if board.is_game_over():
            return None

        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return None

        try:
            self._load_model()
        except Exception:
            return self._random_legal(board)

        prompt = f"FEN: {fen}\nMove:"
        candidates = self._candidate_moves(board, max_candidates=20)

        if not candidates:
            return self._random_legal(board)

        best_move = None
        best_score = -1e30

        try:
            for move_uci in candidates:
                move = chess.Move.from_uci(move_uci)

                lm_score = self._score_move_lm(prompt, move_uci)
                tactical_score = self._score_move_tactical(board, move)

                # Tactical score should matter more than LM score
                total_score = tactical_score + 0.5 * lm_score

                if total_score > best_score:
                    best_score = total_score
                    best_move = move_uci

        except Exception:
            return self._random_legal(board)

        if best_move in legal_moves:
            return best_move

        return self._random_legal(board)
