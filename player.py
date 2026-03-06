import chess
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    UCI_REGEX = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")

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

    def _extract_first_uci(self, text: str) -> Optional[str]:
        text = text.lower()
        match = self.UCI_REGEX.search(text)
        return match.group(0) if match else None

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)

        # If no legal moves exist, game is over
        if board.is_game_over():
            return None

        try:
            self._load_model()
        except Exception:
            return None

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

            # Return whatever the model produced, even if illegal
            return move

        except Exception:
            return None
