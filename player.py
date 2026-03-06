import chess
import random
import re
import torch
import torch.nn.functional as F
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    UCI_REGEX = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")

    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)

        # CHANGE THIS
        self.model_id = "YOUR_USERNAME/chess-gpt2-midterm_new"

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

    def _extract_first_uci(self, text: str) -> Optional[str]:
        text = text.lower()
        match = self.UCI_REGEX.search(text)
        return match.group(0) if match else None

    def _generate_move(self, fen: str) -> Optional[str]:
        prompt = f"FEN: {fen}\nMove:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_first_uci(text)

    def _score_move(self, prompt: str, move_uci: str) -> float:
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

    def _candidate_moves(self, board: chess.Board, max_candidates: int = 20) -> list[str]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        priority = []
        others = []

        for m in legal_moves:
            u = m.uci()

            # Strong/forcing moves first
            if (
                board.is_capture(m)
                or board.gives_check(m)
                or (m.promotion is not None)
                or board.is_castling(m)
            ):
                priority.append(u)
            else:
                others.append(u)

        # Mild opening bias: prefer center/development if early game
        if board.fullmove_number <= 10:
            preferred = []
            rest = []

            opening_like = {
                "e2e4", "d2d4", "c2c4", "g1f3", "b1c3",
                "e7e5", "d7d5", "c7c5", "g8f6", "b8c6",
                "f1c4", "f8c5", "f1b5", "f8b4",
                "e1g1", "e8g8", "e1c1", "e8c8"
            }

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
        legal_moves = [m.uci() for m in board.legal_moves]

        if not legal_moves:
            return None

        try:
            self._load_model()
        except Exception:
            return random.choice(legal_moves)

        prompt = f"FEN: {fen}\nMove:"

        # Base candidate pool from python-chess
        candidates = self._candidate_moves(board, max_candidates=20)

        # Hybrid trick: include one free-generated move if legal
        try:
            generated = self._generate_move(fen)
            if generated in legal_moves and generated not in candidates:
                candidates.append(generated)
        except Exception:
            pass

        # Final safety
        if not candidates:
            return random.choice(legal_moves)

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
