import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

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

        self.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
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

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        try:
            self._load_model()
        except Exception:
            return self._random_legal(fen)

        prompt = self._build_prompt(fen)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):]

            move = self._extract_move(decoded)

            if move:
            # Check if the move is actually legal in this position
                board = chess.Board(fen)
                legal = {m.uci() for m in board.legal_moves}
            if move in legal:
                return move
                if move:
                    return move

        except Exception:
            pass

        return self._random_legal(fen)
