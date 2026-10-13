"""Comparing two boards: which tiles are new and which existing tiles changed."""

from __future__ import annotations

from dataclasses import dataclass

from wwf.core.board import Board, Tile
from wwf.core.move import Placement


@dataclass(frozen=True)
class BoardDiff:
    placements: tuple[Placement, ...]  # empty before, a tile now; sorted by (row, col)
    # existing tiles whose letter differs now, or that are gone: (row, col, before, after)
    changed: tuple[tuple[int, int, Tile, Tile | None], ...]


def diff_boards(before: Board, after: Board) -> BoardDiff:
    """What `after` adds to `before`. Tiles never leave a WWF board, so a changed or missing
    tile means an earlier reading or typing mistake, not part of a move."""
    placements: list[Placement] = []
    changed: list[tuple[int, int, Tile, Tile | None]] = []
    for r in range(before.size):
        for c in range(before.size):
            old, new = before.cells[r][c], after.cells[r][c]
            if old is None and new is not None:
                placements.append(Placement(r, c, new))
            elif old is not None and (new is None or new.letter != old.letter):
                changed.append((r, c, old, new))
    return BoardDiff(tuple(placements), tuple(changed))
