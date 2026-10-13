from __future__ import annotations

import random

from wwf.core.board import Board
from wwf.core.config import BoardConfig
from wwf.core.dictionary import Dictionary
from wwf.core.rack import Rack
from wwf.solvers.anchor_trie import AnchorTrieSolver


def bag_tiles(cfg: BoardConfig, blanks: bool = True) -> list[str]:
    return [t for t, n in sorted(cfg.bag.items()) for _ in range(n) if blanks or t != "?"]


def random_game_board(
    cfg: BoardConfig, dictionary: Dictionary, seed: int, turns: int
) -> tuple[Board, list[str]]:
    """Play `turns` random good moves from a shuffled bag. Returns the board and the rest of the
    bag (without blanks, so test racks stay small enough for the reference generator)."""
    rng = random.Random(seed)
    bag = bag_tiles(cfg)
    rng.shuffle(bag)
    board = Board.empty(cfg)
    solver = AnchorTrieSolver()
    for _ in range(turns):
        rack = Rack(tuple(bag.pop() for _ in range(min(7, len(bag)))))
        result = solver.solve(board, rack, dictionary, cfg, max_results=10)
        if result.moves:
            board = board.with_move(rng.choice(result.moves).move)
    return board, [t for t in bag if t != "?"]
