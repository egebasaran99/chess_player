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
        tactical_weight: float = 0.08,
        candidate_pool_size: int = 24,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.tactical_weight = tactical_weight
        self.candidate_pool_size = candidate_pool_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None

        # Simple anti-repetition memory for positions this player has seen
        self.seen_fens = {}

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

    def _get_lm_score(self, prompt: str, move_str: str) -> float:
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

    def _material_balance(self, board: chess.Board) -> float:
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        score = 0
        for piece_type, value in values.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value

        return score

    def _is_endgame(self, board: chess.Board) -> bool:
        total_nonking = 0
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            total_nonking += len(board.pieces(piece_type, chess.WHITE))
            total_nonking += len(board.pieces(piece_type, chess.BLACK))
        return total_nonking <= 4

    def _king_activity_bonus(self, board: chess.Board, side: bool) -> float:
        king_sq = board.king(side)
        if king_sq is None:
            return 0.0
        file_idx = chess.square_file(king_sq)
        rank_idx = chess.square_rank(king_sq)
        # closer to center = better in endgames
        dist = abs(file_idx - 3.5) + abs(rank_idx - 3.5)
        return (7.0 - dist) / 7.0

    def _score_move_tactical(self, board: chess.Board, move: chess.Move) -> float:
        score = 0.0
        mover = board.turn

        # Capture bonus
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
                score += value_map.get(captured_piece.piece_type, 0) / 8.0

        # Promotion bonus
        if move.promotion:
            score += 0.8

        before_material = self._material_balance(board)

        board.push(move)

        # Immediate tactical outcomes
        if board.is_checkmate():
            board.pop()
            return 1.0

        if board.is_stalemate():
            score -= 0.9

        if board.is_check():
            score += 0.25

        # Avoid giving opponent mate in 1
        opp_has_mate_in_1 = False
        for reply in board.legal_moves:
            board.push(reply)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                opp_has_mate_in_1 = True
                break
        if opp_has_mate_in_1:
            score -= 1.0

        # Material improvement bonus
        after_material = self._material_balance(board)
        if mover == chess.WHITE:
            score += 0.10 * (after_material - before_material)
        else:
            score += 0.10 * (before_material - after_material)

        # Anti-repetition penalty
        fen_key = board.board_fen() + (" w" if board.turn == chess.WHITE else " b")
        repeat_count = self.seen_fens.get(fen_key, 0)
        score -= 0.35 * repeat_count

        # Endgame king activity
        if self._is_endgame(board):
            own_king_bonus = self._king_activity_bonus(board, mover)
            opp_king_bonus = self._king_activity_bonus(board, not mover)
            score += 0.15 * own_king_bonus
            score -= 0.05 * opp_king_bonus

        board.pop()

        return max(-1.0, min(1.0, score))

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

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)

        if board.is_game_over():
            return None

        # record current position
        fen_key = board.board_fen() + (" w" if board.turn == chess.WHITE else " b")
        self.seen_fens[fen_key] = self.seen_fens.get(fen_key, 0) + 1

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
