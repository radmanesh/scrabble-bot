"""Turning a typed word and its start square into the tiles it places."""

from __future__ import annotations

from wwf.core.board import Board, Tile
from wwf.core.move import Direction, Placement


def placements_for_word(
    board: Board, word: str, row: int, col: int, direction: Direction
) -> tuple[list[Placement], list[str]]:
    """The new tiles `word` places when written from (row, col) (zero-based) in `direction`.

    The word includes any tiles already on the board it runs through, which must match.
    Lowercase letters are blanks. Returns the placements and a list of problems:
    `empty_word`, `bad_letter`, `out_of_bounds`, `letter_mismatch:<row>,<col>:<letter>`
    (one-based coordinates), `no_new_tiles`.
    """
    word = word.strip()
    if not word:
        return [], ["empty_word"]
    if not (word.isascii() and word.isalpha()):
        return [], ["bad_letter"]
    dr, dc = direction.step
    placements: list[Placement] = []
    problems: list[str] = []
    for i, ch in enumerate(word):
        r, c = row + i * dr, col + i * dc
        if not board.in_bounds(r, c):
            problems.append("out_of_bounds")
            break
        existing = board.get(r, c)
        if existing is None:
            placements.append(Placement(r, c, Tile(ch.upper(), is_blank=ch.islower())))
        elif existing.letter != ch.upper():
            problems.append(f"letter_mismatch:{r + 1},{c + 1}:{existing.letter}")
    if not placements and not problems:
        problems.append("no_new_tiles")
    return placements, problems
