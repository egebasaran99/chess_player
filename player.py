import chess
import random
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    def __init__(
        self,
        name: str = "TransformerPlayer",
        model_id: str = "egeb9/chess-gpt2-v4",
        candidate_pool_size: int = 24,
        opening_weight: float = 0.18,
        tactical_weight: float = 0.12,
        repetition_penalty: float = 0.20,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.candidate_pool_size = candidate_pool_size
        self.opening_weight = opening_weight
        self.tactical_weight = tactical_weight
        self.repetition_penalty = repetition_penalty

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

        self.seen_fens = {}

    # Lazy loading
    def _load_model(self):
        if self.model is None:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    # Utilities
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

    def _piece_value(self, piece_type: int) -> int:
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100,
        }
        return values.get(piece_type, 0)

    def _material_balance(self, board: chess.Board) -> float:
        score = 0.0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            score += len(board.pieces(piece_type, chess.WHITE)) * self._piece_value(piece_type)
            score -= len(board.pieces(piece_type, chess.BLACK)) * self._piece_value(piece_type)
        return score

    def _is_immediate_stalemate(self, board: chess.Board, move: chess.Move) -> bool:
        board.push(move)
        result = board.is_stalemate()
        board.pop()
        return result

    def _allows_opponent_mate_in_1(self, board: chess.Board, move: chess.Move) -> bool:
        board.push(move)
        for reply in board.legal_moves:
            board.push(reply)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                board.pop()
                return True
        board.pop()
        return False

    def _score_move_opening(self, board: chess.Board, move: chess.Move) -> float:
        if board.fullmove_number > 8:
            return 0.0

        score = 0.0
        mover = board.turn
        piece = board.piece_at(move.from_square)
        if piece is None:
            return 0.0

        uci = move.uci()

        # Very early central pawn moves
        if board.fullmove_number <= 3:
            if mover == chess.WHITE and uci in {"e2e4", "d2d4", "c2c4", "g1f3", "b1c3"}:
                score += 0.45
            if mover == chess.BLACK and uci in {"e7e5", "d7d5", "c7c5", "g8f6", "b8c6"}:
                score += 0.45

        # Minor piece development
        if piece.piece_type == chess.KNIGHT:
            score += 0.18
        elif piece.piece_type == chess.BISHOP:
            score += 0.14

        # Castling bonus
        if board.is_castling(move):
            score += 0.55

        # Slight penalty for early queen moves
        if piece.piece_type == chess.QUEEN and board.fullmove_number <= 6:
            score -= 0.30

        # Slight penalty for rook moves before castling unless tactical
        if piece.piece_type == chess.ROOK and board.fullmove_number <= 6:
            score -= 0.18

        # Slight penalty for edge pawn pushes early
        if piece.piece_type == chess.PAWN:
            from_file = chess.square_file(move.from_square)
            if from_file in {0, 7} and board.fullmove_number <= 5:
                score -= 0.12

        # Small center occupancy / control effect after move
        center = {chess.D4, chess.E4, chess.D5, chess.E5}
        ext_center = {
            chess.C3, chess.D3, chess.E3, chess.F3,
            chess.C4, chess.D4, chess.E4, chess.F4,
            chess.C5, chess.D5, chess.E5, chess.F5,
            chess.C6, chess.D6, chess.E6, chess.F6,
        }

        board.push(move)
        moved_piece = board.piece_at(move.to_square)
        if moved_piece and move.to_square in center:
            score += 0.18
        elif moved_piece and move.to_square in ext_center:
            score += 0.08
        board.pop()

        return max(-1.0, min(1.0, score))

    def _score_move_tactical(self, board: chess.Board, move: chess.Move) -> float:
        score = 0.0
        mover = board.turn

        moved_piece = board.piece_at(move.from_square)
        captured_piece = board.piece_at(move.to_square) if board.is_capture(move) else None

        if captured_piece:
            score += 0.16 * self._piece_value(captured_piece.piece_type)

        if move.promotion:
            score += 1.4

        before_material = self._material_balance(board)

        board.push(move)

        if board.is_checkmate():
            board.pop()
            return 3.0

        if board.is_stalemate():
            score -= 2.0

        if board.is_check():
            score += 0.55

        # Opponent mate in 1 after our move
        opp_mate_in_1 = False
        for reply in board.legal_moves:
            board.push(reply)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                opp_mate_in_1 = True
                break
        if opp_mate_in_1:
            score -= 3.0

        # Moved piece hanging / under-defended
        moved_after = board.piece_at(move.to_square)
        if moved_after is not None:
            opp_attackers = len(board.attackers(not mover, move.to_square))
            own_defenders = len(board.attackers(mover, move.to_square))
            moved_value = self._piece_value(moved_after.piece_type)

            if opp_attackers > 0 and own_defenders == 0:
                score -= 0.28 * moved_value
            elif opp_attackers > own_defenders:
                score -= 0.12 * moved_value

        after_material = self._material_balance(board)
        if mover == chess.WHITE:
            score += 0.10 * (after_material - before_material)
        else:
            score += 0.10 * (before_material - after_material)

        # Repetition penalty
        fen_key = board.board_fen() + (" w" if board.turn == chess.WHITE else " b")
        repeat_count = self.seen_fens.get(fen_key, 0)
        score -= self.repetition_penalty * repeat_count

        board.pop()

        return max(-3.0, min(3.0, score))

    def _get_candidate_moves(self, board: chess.Board):
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return legal_moves

        safe_moves = []
        risky_moves = []

        for move in legal_moves:
            if self._is_immediate_stalemate(board, move):
                risky_moves.append(move)
                continue
            if self._allows_opponent_mate_in_1(board, move):
                risky_moves.append(move)
                continue
            safe_moves.append(move)

        base_moves = safe_moves if safe_moves else legal_moves

        if len(base_moves) <= 20:
            return base_moves

        promotions = []
        captures = []
        checks = []
        others = []

        for move in base_moves:
            if move.promotion:
                promotions.append(move)
            elif board.is_capture(move):
                captures.append(move)
            else:
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return [move]
                if board.is_check():
                    checks.append(move)
                else:
                    others.append(move)
                board.pop()

        candidates = promotions + captures + checks
        random.shuffle(others)

        remaining = max(0, self.candidate_pool_size - len(candidates))
        candidates += others[:remaining]

        seen = set()
        unique = []
        for mv in candidates:
            if mv not in seen:
                seen.add(mv)
                unique.append(mv)

        return unique if unique else base_moves

    # Main API
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)

        if board.is_game_over():
            return None

        fen_key = board.board_fen() + (" w" if board.turn == chess.WHITE else " b")
        self.seen_fens[fen_key] = self.seen_fens.get(fen_key, 0) + 1

        try:
            self._load_model()
        except Exception:
            return self._random_legal(fen)

        prompt = self._build_prompt(fen)

        try:
            candidates = self._get_candidate_moves(board)

            if len(candidates) == 1:
                return candidates[0].uci()

            best_move = None
            best_score = float("-inf")

            for move in candidates:
                move_str = move.uci()
                lm_score = self._get_lm_score(prompt, move_str)
                opening_score = self._score_move_opening(board, move)
                tactical_score = self._score_move_tactical(board, move)

                total_score = (
                    lm_score
                    + self.opening_weight * opening_score
                    + self.tactical_weight * tactical_score
                )

                if total_score > best_score:
                    best_score = total_score
                    best_move = move

            if best_move is not None:
                return best_move.uci()

        except Exception:
            pass

        return self._random_legal(fen)
