"""Super Score mystery squares: the reward a play on one earns, and what it does to the score.

The reward is random, so a board alone cannot score a move on a mystery square. Given the reward
that was (or might be) earned, a letter or word reward acts as that bonus on the square and Points
Plus adds a flat 10; Powerful Tile and Fire Tile change the rack, not the score.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import product

from wwf.core.board import Board, Bonus
from wwf.core.config import BoardConfig
from wwf.core.move import Move
from wwf.core.scoring import ScoreBreakdown, score_move

POINTS_PLUS = 10

# code -> label. Codes are the board's bonus characters where there is one.
REWARDS: dict[str, str] = {
    "d": "Double Letter",
    "t": "Triple Letter",
    "q": "Quadruple Letter",
    "D": "Double Word",
    "T": "Triple Word",
    "Q": "Quadruple Word",
    "+": f"Points Plus (+{POINTS_PLUS})",
    "p": "Powerful Tile (rack only)",
    "f": "Fire Tile (rack only)",
}
_AS_BONUS = {
    "d": Bonus.DL, "t": Bonus.TL, "q": Bonus.QL, "D": Bonus.DW, "T": Bonus.TW, "Q": Bonus.QW,
}  # fmt: skip

Cell = tuple[int, int]


def mystery_cells(board: Board, move: Move) -> list[Cell]:
    """The mystery squares this move puts a new tile on, in placement order."""
    return [(p.row, p.col) for p in move.placements if board.bonus(p.row, p.col) is Bonus.MYSTERY]


def score_with_rewards(
    board: Board, move: Move, cfg: BoardConfig, rewards: Mapping[Cell, str]
) -> ScoreBreakdown:
    """The move's score if each mystery square it covers earned `rewards[cell]`. A square with
    no reward (or one that doesn't change the score) counts as a plain square."""
    cells = mystery_cells(board, move)
    changes = {cell: _AS_BONUS.get(rewards.get(cell, ""), Bonus.NONE) for cell in cells}
    s = score_move(board.with_bonuses(changes), move, cfg)
    plus = POINTS_PLUS * sum(rewards.get(cell) == "+" for cell in cells)
    return replace(s, total=s.total + plus, reward_points=plus) if plus else s


def best_case(board: Board, move: Move, cfg: BoardConfig) -> tuple[int, dict[Cell, str]]:
    """The highest score any rewards could give this move, and the rewards that give it."""
    cells = mystery_cells(board, move)
    best_total, best = -1, {}
    for combo in product(REWARDS, repeat=len(cells)):
        rewards = dict(zip(cells, combo, strict=True))
        total = score_with_rewards(board, move, cfg, rewards).total
        if total > best_total:
            best_total, best = total, rewards
    return best_total, best
