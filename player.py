import random
import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TransformerPlayer:
    def __init__(self, model_name="egebasaran99/chess-gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Small tactical weight so LM dominates
        self.tactical_weight = 0.08

    def _get_lm_score(self, prompt, move_str):
        """
        Score move using the language model.
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

    def _score_move_tactical(self, board, move):
        """
        Very light normalized tactical evaluation.
        Range approximately [-1, 1]
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
                    chess.QUEEN: 9
                }
                score += value_map.get(captured_piece.piece_type, 0) / 10.0

        board.push(move)

        if board.is_checkmate():
            score += 1.0
        elif board.is_check():
            score += 0.2

        board.pop()

        return max(-1.0, min(1.0, score))

    def _get_candidate_moves(self, board):
        """
        Light candidate filtering.
        Prioritize tactically relevant moves but do not exclude too much.
        """
        legal_moves = list(board.legal_moves)

        if len(legal_moves) <= 8:
            return legal_moves

        captures = []
        promotions = []
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

    def get_move(self, fen):
        board = chess.Board(fen)

        if board.is_game_over():
            return None

        candidates = self._get_candidate_moves(board)
        prompt = fen + " ->"

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

        if best_move is None:
            return random.choice(list(board.legal_moves)).uci()

        return best_move.uci()
