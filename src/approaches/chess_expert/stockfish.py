"""
Helpers for integrating Stockfish/NNUE evaluations into the Chess Expert pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

try:
    import chess
    import chess.engine
except ImportError as exc:  # pragma: no cover - fallback for environments without python-chess
    chess = None  # type: ignore
    chess_engine_import_error = exc  # type: ignore
else:
    chess_engine_import_error = None  # type: ignore


__all__ = [
    "score_to_centipawns",
    "normalize_centipawns",
    "StockfishEvaluator",
]


def score_to_centipawns(score: chess.engine.PovScore, mate_score: int = 10_000) -> float:
    """
    Convert a Stockfish PovScore into centipawns from White's perspective.

    When a mate is detected, ``mate_score`` (with sign) is returned to keep values bounded.
    """
    if score.is_mate():
        mate_in = score.mate()
        if mate_in is None:
            return math.copysign(mate_score, score.white().score(mate_score=mate_score))
        return math.copysign(mate_score, mate_in)
    return float(score.white().score(mate_score=mate_score))


def normalize_centipawns(cp: float, clamp: float = 1000.0) -> float:
    """Normalize a centipawn value into [-1, 1] with optional clamping."""
    if clamp <= 0:
        raise ValueError("`clamp` must be positive.")
    cp = max(-clamp, min(clamp, cp))
    return cp / clamp


@dataclass
class StockfishConfig:
    """Configuration for running a Stockfish-compatible engine."""

    path: str
    depth: Optional[int] = 12
    nodes: Optional[int] = None
    movetime: Optional[float] = None
    threads: int = 1
    hash_size: int = 256  # MB
    cp_clamp: float = 1000.0

    def build_limit(self) -> chess.engine.Limit:
        kwargs = {}
        if self.depth is not None:
            kwargs["depth"] = self.depth
        if self.nodes is not None:
            kwargs["nodes"] = self.nodes
        if self.movetime is not None:
            kwargs["time"] = self.movetime
        return chess.engine.Limit(**kwargs)


class StockfishEvaluator:
    """
    Thin wrapper around a UCI engine that produces normalized centipawn evaluations for FEN strings.
    """

    def __init__(self, config: StockfishConfig):
        if chess is None:  # pragma: no cover - safeguards unit tests without python-chess
            raise RuntimeError(
                "python-chess is required to use StockfishEvaluator. "
                f"Import error: {chess_engine_import_error}"
            )
        self.config = config
        self._engine = chess.engine.SimpleEngine.popen_uci(config.path)
        self._engine.configure({"Threads": config.threads, "Hash": config.hash_size})
        self._limit = config.build_limit()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("StockfishEvaluator has been closed.")

    def evaluate(self, fen: str) -> float:
        """Evaluate a single FEN position and return a normalized centipawn score."""
        self._ensure_open()
        board = chess.Board(fen)
        info = self._engine.analyse(board, self._limit)
        score = info.get("score")
        if score is None:
            return 0.0
        cp = score_to_centipawns(score, mate_score=int(self.config.cp_clamp))
        return normalize_centipawns(cp, clamp=self.config.cp_clamp)

    def evaluate_many(self, fens: Sequence[str]) -> List[float]:
        """Evaluate multiple FEN strings sequentially."""
        return [self.evaluate(fen) for fen in fens]

    def __call__(self, fen: str) -> float:
        return self.evaluate(fen)

    def close(self) -> None:
        if not self._closed:
            try:
                self._engine.quit()
            finally:
                self._closed = True

    def __enter__(self) -> "StockfishEvaluator":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
