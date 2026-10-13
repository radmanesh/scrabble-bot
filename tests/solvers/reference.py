"""Slow, obviously correct reference move generator.

For each direction and each empty starting cell, lay every distinct ordering of 1..n rack tiles
into the successive empty cells (existing tiles are skipped over, becoming part of the word),
expand blanks into all 26 letters, and keep whatever `validate_move` accepts. Every legal move is
a contiguous run of new tiles in one line, so this enumerates all of them. The only pruning is
that a legal move must cover an anchor square, which `validate_move` would enforce anyway.
"""

from __future__ import annotations

from itertools import permutations, product

from wwf.core.board import Board, Tile
from wwf.core.config import BoardConfig
from wwf.core.dictionary import Dictionary
from wwf.core.move import Placement, build_move
from wwf.core.rack import BLANK, Rack
from wwf.core.scoring import score_move, validate_move

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

MoveKey = frozenset[tuple[int, int, str, bool]]


def reference_moves(
    board: Board, rack: Rack, dictionary: Dictionary, cfg: BoardConfig
) -> dict[MoveKey, int]:
    """Every legal move, keyed by its new-tile placements, mapped to its total score."""
    anchors = board.anchors()
    tiles = rack.tiles
    orderings = {perm for n in range(1, len(tiles) + 1) for perm in permutations(tiles, n)}
    out: dict[MoveKey, int] = {}
    size = board.size
    for across in (True, False):
        for line in range(size):
            for start in range(size):
                cells: list[tuple[int, int]] = []
                pos = start
                while pos < size and len(cells) < len(tiles):
                    r, c = (line, pos) if across else (pos, line)
                    if board.get(r, c) is None:
                        cells.append((r, c))
                    elif not cells:
                        break  # start must be an empty cell
                    pos += 1
                for order in orderings:
                    if len(order) > len(cells):
                        continue
                    spots = cells[: len(order)]
                    if not anchors.intersection(spots):
                        continue
                    blank_slots = [i for i, t in enumerate(order) if t == BLANK]
                    for fill in product(LETTERS, repeat=len(blank_slots)):
                        letters = list(order)
                        for i, ch in zip(blank_slots, fill, strict=True):
                            letters[i] = ch
                        placements = [
                            Placement(r, c, Tile(letters[i], is_blank=i in blank_slots))
                            for i, (r, c) in enumerate(spots)
                        ]
                        move = build_move(board, placements)
                        if move.key() in out:
                            continue
                        if validate_move(board, move, dictionary, cfg, rack):
                            continue
                        out[move.key()] = score_move(board, move, cfg).total
    return out
