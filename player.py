import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    UCI_REGEX = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")

    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)

        # CHANGE THIS TO YOUR HF MODEL
        self.model_id = "YOUR_USERNAME/chess-gpt2-midterm"

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

    def _extract_first_uci(self, text: str) -> Optional[str]:
        text = text.lower()
        match = self.UCI_REGEX.search(text)
        return match.group(0) if match else None

    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

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

        try:
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
            move = self._extract_first_uci(text)

            if move in legal_moves:
                return move

        except Exception:
            pass

        return random.choice(legal_moves)
