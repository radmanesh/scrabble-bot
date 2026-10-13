from __future__ import annotations

from pathlib import Path

import pytest

from wwf.core.board import Board, Tile
from wwf.core.config import BoardConfig, assets_dir, load_board_config, project_root
from wwf.core.dictionary import TrieDictionary, load_dictionary
from wwf.core.move import Move, Placement, build_move


@pytest.fixture(scope="session")
def cfg() -> BoardConfig:
    return load_board_config()


@pytest.fixture(scope="session")
def dictionary() -> TrieDictionary:
    return load_dictionary(assets_dir() / "dictionaries" / "enable1.txt", project_root() / ".cache")


@pytest.fixture
def empty(cfg: BoardConfig) -> Board:
    return Board.empty(cfg)


def place(board: Board, row: int, col: int, word: str, across: bool = True) -> Board:
    """Put existing tiles on a board (no legality checks). Lowercase letters are blanks."""
    dr, dc = (0, 1) if across else (1, 0)
    return board.with_tiles(
        (row + i * dr, col + i * dc, Tile.from_char(ch)) for i, ch in enumerate(word)
    )


def move_of(board: Board, cells: list[tuple[int, int, str]]) -> Move:
    """Build a Move from (row, col, char) triples; lowercase chars are blanks."""
    return build_move(board, [Placement(r, c, Tile.from_char(ch)) for r, c, ch in cells])


def word_move(board: Board, row: int, col: int, word: str, across: bool = True) -> Move:
    """Move placing `word` from (row, col), skipping '.' positions that are already occupied."""
    dr, dc = (0, 1) if across else (1, 0)
    cells = [(row + i * dr, col + i * dc, ch) for i, ch in enumerate(word) if ch != "."]
    return move_of(board, cells)


FIXTURES = Path(__file__).parent
