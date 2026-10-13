"""Demo solver: random legal one-tile plays. Exists to exercise the plugin interface."""

from __future__ import annotations

import random
import time
from typing import Any

from wwf.core.board import Board, Tile
from wwf.core.config import BoardConfig
from wwf.core.dictionary import Dictionary
from wwf.core.move import Direction, Placement, build_move
from wwf.core.rack import BLANK, Rack
from wwf.core.scoring import score_move, validate_move
from wwf.solvers.base import (
    ScoredMove,
    SolverParam,
    SolverResult,
    leave_after,
    resolve_params,
    sort_key_score_only,
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class DummyRandomSolver:
    id = "dummy_random"
    display_name = "Random legal play (demo)"
    description = (
        "DEMO ONLY. Tries random single-tile plays at anchor squares and keeps the legal ones. "
        "Useful for testing the plugin interface, not for playing."
    )
    params: tuple[SolverParam, ...] = (
        SolverParam("attempts", "Attempts", "int", 200, min=1, max=10_000,
                    help="Random (anchor, direction, tile) triples to try."),
        SolverParam("max_results", "Max results", "int", 25, min=1, max=500,
                    help="How many top moves to return."),
        SolverParam("seed", "Random seed", "int", -1, min=-1,
                    help="-1 for a different result each time."),
    )  # fmt: skip

    def solve(
        self, board: Board, rack: Rack, dictionary: Dictionary, cfg: BoardConfig, **params: Any
    ) -> SolverResult:
        p = resolve_params(self.params, params)
        t0 = time.perf_counter()
        rng = random.Random(None if p["seed"] < 0 else p["seed"])
        anchors = sorted(board.anchors())
        found: dict[frozenset[tuple[int, int, str, bool]], ScoredMove] = {}
        attempts = 0
        if anchors and rack.tiles:
            for _ in range(p["attempts"]):
                attempts += 1
                r, c = rng.choice(anchors)
                direction = rng.choice((Direction.ACROSS, Direction.DOWN))
                t = rng.choice(rack.tiles)
                tile = Tile(rng.choice(LETTERS), is_blank=True) if t == BLANK else Tile(t)
                move = build_move(board, [Placement(r, c, tile)], direction)
                if move.key() in found or validate_move(board, move, dictionary, cfg, rack):
                    continue
                found[move.key()] = ScoredMove(
                    move, score_move(board, move, cfg), leave_after(rack, move)
                )
        moves = sorted(found.values(), key=sort_key_score_only)
        elapsed = int((time.perf_counter() - t0) * 1000)
        return SolverResult(
            tuple(moves[: p["max_results"]]),
            elapsed,
            {"attempts": attempts, "legal_found": len(found)},
        )
