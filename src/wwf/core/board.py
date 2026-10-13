"""Immutable board model."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wwf.core.config import BoardConfig
    from wwf.core.move import Move


class Bonus(Enum):
    NONE = ""
    DL = "d"
    TL = "t"
    DW = "D"
    TW = "T"
    CENTER = "*"

    @classmethod
    def from_char(cls, ch: str) -> Bonus:
        return cls.NONE if ch == "." else cls(ch)

    @property
    def letter_multiplier(self) -> int:
        return {Bonus.DL: 2, Bonus.TL: 3}.get(self, 1)

    @property
    def word_multiplier(self) -> int:
        return {Bonus.DW: 2, Bonus.TW: 3}.get(self, 1)


@dataclass(frozen=True)
class Tile:
    letter: str  # "A"-"Z"; for a blank, the letter it represents
    is_blank: bool = False

    @classmethod
    def from_char(cls, ch: str) -> Tile:
        """Uppercase is a normal tile, lowercase is a blank standing for that letter."""
        if len(ch) != 1 or not ch.isalpha() or not ch.isascii():
            raise ValueError(f"invalid tile character {ch!r}")
        return cls(ch.upper(), is_blank=ch.islower())

    def to_char(self) -> str:
        return self.letter.lower() if self.is_blank else self.letter


Cells = tuple[tuple[Tile | None, ...], ...]
Bonuses = tuple[tuple[Bonus, ...], ...]


@dataclass(frozen=True)
class Board:
    size: int
    cells: Cells  # cells[row][col]
    bonuses: Bonuses

    @classmethod
    def empty(cls, config: BoardConfig) -> Board:
        size = config.size
        bonuses = tuple(tuple(Bonus.from_char(ch) for ch in row) for row in config.layout)
        cells: Cells = tuple(tuple(None for _ in range(size)) for _ in range(size))
        return cls(size, cells, bonuses)

    @classmethod
    def from_rows(cls, config: BoardConfig, rows: Sequence[str | Sequence[str | None]]) -> Board:
        """Build from row strings ('.' or ' ' empty, lowercase blank) or lists of str/None."""
        base = cls.empty(config)
        if len(rows) != base.size:
            raise ValueError(f"expected {base.size} rows, got {len(rows)}")
        cells: list[tuple[Tile | None, ...]] = []
        for r, row in enumerate(rows):
            if len(row) != base.size:
                raise ValueError(f"row {r} has {len(row)} cells, expected {base.size}")
            cells.append(
                tuple(
                    None if ch is None or ch in (".", " ", "") else Tile.from_char(ch) for ch in row
                )
            )
        return cls(base.size, tuple(cells), base.bonuses)

    def to_rows(self) -> list[str]:
        return ["".join("." if t is None else t.to_char() for t in row) for row in self.cells]

    def to_grid(self) -> list[list[str | None]]:
        return [[None if t is None else t.to_char() for t in row] for row in self.cells]

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def get(self, r: int, c: int) -> Tile | None:
        if not self.in_bounds(r, c):
            return None
        return self.cells[r][c]

    def bonus(self, r: int, c: int) -> Bonus:
        return self.bonuses[r][c]

    @property
    def center(self) -> tuple[int, int]:
        for r, row in enumerate(self.bonuses):
            for c, b in enumerate(row):
                if b is Bonus.CENTER:
                    return (r, c)
        return (self.size // 2, self.size // 2)

    def with_tiles(self, placements: Iterable[tuple[int, int, Tile]]) -> Board:
        grid = [list(row) for row in self.cells]
        for r, c, tile in placements:
            if grid[r][c] is not None:
                raise ValueError(f"cell ({r},{c}) is occupied")
            grid[r][c] = tile
        return Board(self.size, tuple(tuple(row) for row in grid), self.bonuses)

    def with_move(self, move: Move) -> Board:
        return self.with_tiles((p.row, p.col, p.tile) for p in move.placements)

    def is_empty(self) -> bool:
        return all(t is None for row in self.cells for t in row)

    def tile_count(self) -> int:
        return sum(t is not None for row in self.cells for t in row)

    def anchors(self) -> set[tuple[int, int]]:
        """Empty cells orthogonally adjacent to a tile; the centre only if the board is empty."""
        if self.is_empty():
            return {self.center}
        out: set[tuple[int, int]] = set()
        for r in range(self.size):
            for c in range(self.size):
                if self.cells[r][c] is not None:
                    continue
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if self.get(r + dr, c + dc) is not None:
                        out.add((r, c))
                        break
        return out

    def transpose(self) -> Board:
        cells = tuple(zip(*self.cells, strict=True))
        bonuses = tuple(zip(*self.bonuses, strict=True))
        return Board(self.size, cells, bonuses)
