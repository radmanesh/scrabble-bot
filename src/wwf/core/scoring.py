"""Move scoring and rule validation. The single source of truth for both."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from wwf.core.board import Board, Tile
from wwf.core.config import BoardConfig
from wwf.core.dictionary import Dictionary
from wwf.core.move import Direction, Move, word_extent
from wwf.core.rack import BLANK, Rack


@dataclass(frozen=True)
class WordScore:
    word: str
    start: tuple[int, int]
    direction: Direction
    points: int


@dataclass(frozen=True)
class ScoreBreakdown:
    words: tuple[WordScore, ...]
    bingo_bonus: int
    total: int


@dataclass(frozen=True)
class FormedWord:
    word: str
    start: tuple[int, int]
    direction: Direction
    cells: tuple[tuple[int, int, Tile], ...]


def words_formed(board: Board, move: Move) -> list[FormedWord]:
    """Every word of 2+ letters the move creates: the main word, then each perpendicular word.

    For a one-tile move the "main" direction may form a single letter; it is then dropped and
    only the perpendicular word counts, so nothing is ever counted twice.
    """
    if not move.placements:
        return []
    new = {(p.row, p.col): p.tile for p in move.placements}
    out: list[FormedWord] = []

    def add(r: int, c: int, d: Direction) -> None:
        start, run = word_extent(board, r, c, d, new)
        if len(run) >= 2:
            word = "".join(t.letter for _, _, t in run)
            out.append(FormedWord(word, start, d, tuple(run)))

    first = move.placements[0]
    add(first.row, first.col, move.direction)
    for p in move.placements:
        add(p.row, p.col, move.direction.other)
    return out


def _score_word(
    board: Board, fw: FormedWord, new_cells: set[tuple[int, int]], cfg: BoardConfig
) -> int:
    values = cfg.letter_values
    lm = board.letter_mults
    wm = board.word_mults
    letters = 0
    word_mult = 1
    for r, c, tile in fw.cells:
        value = 0 if tile.is_blank else values[tile.letter]
        if (r, c) in new_cells:
            letters += value * lm[r][c]
            word_mult *= wm[r][c]
        else:
            letters += value
    return letters * word_mult


RawPlacement = tuple[int, int, str, bool]  # (row, col, letter, is_blank)


def score_raw(board: Board, placements: Sequence[RawPlacement], cfg: BoardConfig) -> int:
    """Total score of a move given as raw placements, without building any objects.

    Equal to `score_move(board, build_move(board, ...), cfg).total`; solvers use it to rank
    thousands of candidates and call `score_move` only for the ones they return.
    """
    cells = board.cells
    size = board.size
    lm = board.letter_mults
    wm = board.word_mults
    values = cfg.letter_values
    new = {(r, c): (letter, blank) for r, c, letter, blank in placements}

    def run_score(r: int, c: int, dr: int, dc: int) -> int:
        while True:
            pr, pc = r - dr, c - dc
            if pr < 0 or pc < 0 or (cells[pr][pc] is None and (pr, pc) not in new):
                break
            r, c = pr, pc
        letters = 0
        mult = 1
        length = 0
        while r < size and c < size:
            t = cells[r][c]
            if t is not None:
                letters += 0 if t.is_blank else values[t.letter]
            else:
                n = new.get((r, c))
                if n is None:
                    break
                letters += (0 if n[1] else values[n[0]]) * lm[r][c]
                mult *= wm[r][c]
            length += 1
            r += dr
            c += dc
        return letters * mult if length >= 2 else 0

    r0, c0 = placements[0][0], placements[0][1]
    if len(placements) == 1:
        total = run_score(r0, c0, 0, 1) + run_score(r0, c0, 1, 0)
    else:
        across = all(p[0] == r0 for p in placements)
        dr, dc = (0, 1) if across else (1, 0)
        total = run_score(r0, c0, dr, dc)
        for r, c, _, _ in placements:
            total += run_score(r, c, dc, dr)
    if len(placements) >= cfg.bingo_tiles:
        total += cfg.bingo_bonus
    return total


def score_move(board: Board, move: Move, cfg: BoardConfig) -> ScoreBreakdown:
    new_cells = {(p.row, p.col) for p in move.placements}
    words = tuple(
        WordScore(fw.word, fw.start, fw.direction, _score_word(board, fw, new_cells, cfg))
        for fw in words_formed(board, move)
    )
    bingo = cfg.bingo_bonus if len(move.placements) >= cfg.bingo_tiles else 0
    return ScoreBreakdown(
        words=words, bingo_bonus=bingo, total=sum(w.points for w in words) + bingo
    )


def validate_move(
    board: Board,
    move: Move,
    dictionary: Dictionary,
    cfg: BoardConfig,
    rack: Rack | None = None,
) -> list[str]:
    """Returns a list of rule violations; empty list means legal.

    `rack` is optional; when given, tiles are checked against it (a blank consumes a "?").
    """
    ps = move.placements
    if not ps:
        return ["no_tiles_placed"]
    violations: list[str] = []

    positions = [(p.row, p.col) for p in ps]
    if any(not board.in_bounds(r, c) for r, c in positions):
        return ["out_of_bounds"]
    if len(set(positions)) != len(positions) or any(
        board.get(r, c) is not None for r, c in positions
    ):
        violations.append("cell_occupied")

    rows = {r for r, _ in positions}
    cols = {c for _, c in positions}
    in_line = len(rows) == 1 or len(cols) == 1
    if not in_line:
        violations.append("not_in_line")
    else:
        if len(ps) == 1:
            line_dir = move.direction
        else:
            line_dir = Direction.ACROSS if len(rows) == 1 else Direction.DOWN
            if line_dir is not move.direction:
                violations.append("not_in_line")
        # every cell between the first and last new tile must be filled
        occupied = set(positions)
        if line_dir is Direction.ACROSS:
            r = next(iter(rows))
            span = [(r, c) for c in range(min(cols), max(cols) + 1)]
        else:
            c = next(iter(cols))
            span = [(r, c) for r in range(min(rows), max(rows) + 1)]
        if any(cell not in occupied and board.get(*cell) is None for cell in span):
            violations.append("gap_in_word")

    if board.is_empty():
        if cfg.center not in set(positions):
            violations.append("first_move_not_on_center")
    else:
        touches = any(
            board.get(r + dr, c + dc) is not None
            for r, c in positions
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        if not touches:
            violations.append("no_contact_with_existing_tiles")

    if rack is not None:
        available = rack.counts()
        needed: Counter[str] = Counter(BLANK if p.tile.is_blank else p.tile.letter for p in ps)
        for letter, n in sorted(needed.items()):
            if available[letter] < n:
                violations.append(f"tile_not_in_rack:{letter}")

    structural = {"cell_occupied", "not_in_line", "gap_in_word"}
    if not structural.intersection(violations):
        formed = words_formed(board, move)
        if not formed:
            violations.append(f"word_not_in_dictionary:{ps[0].tile.letter}")
        for fw in formed:
            if not dictionary.contains(fw.word):
                violations.append(f"word_not_in_dictionary:{fw.word}")
    return violations
