from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from contextlib import nullcontext
from itertools import islice
from typing import Generator, List, Optional

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from approaches.chess_expert.stockfish import (
        StockfishConfig,
        normalize_centipawns,
        score_to_centipawns,
    )
except Exception:
    # Fallback definitions preserve script usability even if the module cannot be imported.
    from dataclasses import dataclass

    @dataclass
    class StockfishConfig:
        path: str
        depth: Optional[int] = 12
        nodes: Optional[int] = None
        movetime: Optional[float] = None
        threads: int = 1
        hash_size: int = 256
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

    def score_to_centipawns(score: chess.engine.PovScore, mate_score: int = 10_000) -> float:
        if score.is_mate():
            mate_in = score.mate()
            if mate_in is None:
                return math.copysign(mate_score, score.white().score(mate_score=mate_score))
            return math.copysign(mate_score, mate_in)
        return float(score.white().score(mate_score=mate_score))

    def normalize_centipawns(cp: float, clamp: float = 1000.0) -> float:
        if clamp <= 0:
            raise ValueError("`clamp` must be positive.")
        cp = max(-clamp, min(clamp, cp))
        return cp / clamp


DEFAULT_TEMPLATE = (
    "Position FEN: {fen}\n"
    "Side to move: {side_to_move}\n"
    "Moves so far: {move_history}\n"
    "Stockfish evaluation (centipawns): {cp:.1f}\n"
    "Normalized evaluation: {normalized_eval:.4f}"
)


def iter_games(path: pathlib.Path) -> Generator[chess.pgn.Game, None, None]:
    """Yield games from a PGN file."""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            yield game


def describe_position(template: str, fen: str, move_history: List[str], cp: float, normalized: float) -> str:
    side_to_move = "White" if fen.split()[1] == "w" else "Black"
    history = " ".join(move_history) if move_history else "<start position>"
    return template.format(
        fen=fen,
        side_to_move=side_to_move,
        move_history=history,
        cp=cp,
        normalized_eval=normalized,
    )


def generate_samples(
    pgn_path: pathlib.Path,
    tokenizer_name: str,
    output_path: pathlib.Path,
    template: str = DEFAULT_TEMPLATE,
    max_games: Optional[int] = None,
    max_positions: Optional[int] = None,
    max_length: int = 512,
    cp_clamp: float = 1000.0,
    tokenizer_pad_left: bool = True,
    engine_cfg: Optional[StockfishConfig] = None,
) -> None:
    """Main driver that generates the dataset."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if tokenizer_pad_left else "right"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_positions = 0

    engine_options = {}
    engine_cm = nullcontext()
    limit = None
    if engine_cfg is not None:
        engine_options = {"Threads": engine_cfg.threads, "Hash": engine_cfg.hash_size}
        limit = engine_cfg.build_limit()
        engine_cm = chess.engine.SimpleEngine.popen_uci(engine_cfg.path)

    with engine_cm as engine, output_path.open("w", encoding="utf-8") as sink:
        if engine is not None:
            engine.configure(engine_options)
        game_iter = iter_games(pgn_path)
        if max_games is not None:
            game_iter = islice(game_iter, max_games)
        for _, game in enumerate(tqdm(game_iter, desc="Games", unit="game"), start=1):

            board = game.board()
            move_history: List[str] = []

            for move in game.mainline_moves():
                san = board.san(move)
                board.push(move)
                move_history.append(san)

                if engine is not None:
                    try:
                        analysis = engine.analyse(board, limit)
                    except chess.engine.EngineTerminatedError as exc:
                        raise RuntimeError("Chess engine terminated unexpectedly.") from exc
                    except chess.engine.EngineError as exc:
                        raise RuntimeError(f"Engine error while analysing position: {exc}") from exc

                    pov_score = analysis.get("score")
                    if pov_score is None:
                        continue

                    cp = score_to_centipawns(pov_score, mate_score=int(cp_clamp))
                else:
                    cp = random.uniform(-cp_clamp, cp_clamp)
                normalized = normalize_centipawns(cp, clamp=cp_clamp)
                fen = board.fen()
                text = describe_position(template, fen, move_history, cp, normalized)

                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_attention_mask=True,
                )

                seq_len = len(encoding["input_ids"])
                chess_eval = [normalized] * seq_len
                chess_mask = [True] * seq_len

                record = {
                    "text": text,
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                    "chess_eval": chess_eval,
                    "chess_mask": chess_mask,
                    "fen": fen,
                    "cp": cp,
                    "move_index": len(move_history),
                    "san": san,
                }
                sink.write(json.dumps(record) + "\n")

                total_positions += 1
                if max_positions is not None and total_positions >= max_positions:
                    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stockfish-labelled chess dataset for the ChessExpert MoE.")
    parser.add_argument("--pgn", required=True, type=pathlib.Path, help="Path to the input PGN file.")
    parser.add_argument("--engine-path", type=pathlib.Path, default=None, help="Path to the Stockfish (UCI) engine binary.")
    parser.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer name or path.")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Output JSONL file.")

    parser.add_argument("--depth", type=int, default=12, help="Search depth for the engine.")
    parser.add_argument("--nodes", type=int, default=None, help="Node limit for the engine (optional).")
    parser.add_argument("--movetime", type=float, default=None, help="Move time (seconds) for the engine (optional).")
    parser.add_argument("--threads", type=int, default=1, help="Number of engine threads.")
    parser.add_argument("--hash-size", type=int, default=256, help="Hash size in MB for the engine.")

    parser.add_argument("--max-games", type=int, default=None, help="Optional limit on number of PGN games.")
    parser.add_argument("--max-positions", type=int, default=None, help="Optional limit on number of analysed positions.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum tokenizer length.")
    parser.add_argument("--cp-clamp", type=float, default=1000.0, help="Clamp absolute centipawn values before normalisation.")
    parser.add_argument("--template", type=str, default=DEFAULT_TEMPLATE, help="Custom text template for each record.")
    parser.add_argument("--pad-left", action="store_true", help="Pad sequences on the left when padding is required.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine_cfg = None
    if args.engine_path is not None:
        engine_cfg = StockfishConfig(
            path=str(args.engine_path),
            depth=args.depth,
            nodes=args.nodes,
            movetime=args.movetime,
            threads=args.threads,
            hash_size=args.hash_size,
            cp_clamp=args.cp_clamp,
        )

    generate_samples(
        pgn_path=args.pgn,
        tokenizer_name=args.tokenizer,
        output_path=args.output,
        template=args.template,
        max_games=args.max_games,
        max_positions=args.max_positions,
        max_length=args.max_length,
        cp_clamp=args.cp_clamp,
        tokenizer_pad_left=args.pad_left,
        engine_cfg=engine_cfg,
    )


if __name__ == "__main__":
    main()
