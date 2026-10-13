"""The failure modes seen in real pictures: warm light, vignetting, a button drawn over the
board, a board that runs off the picture, a picture of the wrong game, a rack out of frame."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tests.vision.synthetic import (
    SAMPLE_BOARD,
    photo,
    tinted,
    vignetted,
    with_button,
)
from wwf.core.config import BoardConfig, load_board_config
from wwf.vision.pipeline import read_image
from wwf.vision.reader import ReaderModel
from wwf.vision.synth import available_fonts
from wwf.vision.train import build_model

pytestmark = pytest.mark.skipif(not available_fonts(), reason="no system fonts to render letters")

QUAD = [[420.0, 180.0], [1110.0, 205.0], [1125.0, 905.0], [400.0, 890.0]]
EMPTY = ["." * 15] * 15


@pytest.fixture(scope="module")
def cfg() -> BoardConfig:
    return load_board_config()


@pytest.fixture(scope="module")
def model() -> ReaderModel:
    return build_model([], load_board_config(), synthetic=True)


def rows(board: list[list[str | None]]) -> list[str]:
    return ["".join(c or "." for c in row) for row in board]


def grid(board: list[str]) -> list[list[str | None]]:
    return [[None if ch == "." else ch for ch in row] for row in board]


def tiles(board: list[str]) -> int:
    return sum(ch != "." for row in board for ch in row)


# --- first picture ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gain", [(0.55, 0.8, 1.2), (0.75, 0.9, 1.1), (1.1, 1.0, 0.8)], ids=["very warm", "warm", "cool"]
)
def test_tint_does_not_invent_tiles(cfg: BoardConfig, model: ReaderModel, gain) -> None:  # type: ignore[no-untyped-def]
    empty = read_image(tinted(photo(cfg, EMPTY, QUAD), gain), model, cfg)
    assert empty is not None
    assert rows(empty.board) == EMPTY, "a colour cast turned empty squares into tiles"

    played = read_image(tinted(photo(cfg, SAMPLE_BOARD, QUAD), gain), model, cfg)
    assert played is not None
    assert rows(played.board) == SAMPLE_BOARD


def test_vignetting_does_not_invent_tiles(cfg: BoardConfig, model: ReaderModel) -> None:
    img = vignetted(photo(cfg, SAMPLE_BOARD, QUAD))
    r = read_image(img, model, cfg)
    assert r is not None and rows(r.board) == SAMPLE_BOARD


def test_a_zoomed_board_is_found_at_the_right_scale(cfg: BoardConfig, model: ReaderModel) -> None:
    big = [[120.0, 40.0], [1480.0, 50.0], [1490.0, 1150.0], [110.0, 1140.0]]  # fills the frame
    r = read_image(photo(cfg, SAMPLE_BOARD, big), model, cfg)
    assert r is not None and rows(r.board) == SAMPLE_BOARD


def test_an_empty_board_with_a_button_over_it(cfg: BoardConfig, model: ReaderModel) -> None:
    img = with_button(photo(cfg, EMPTY, QUAD), (1080, 520))  # at the right edge of the board
    r = read_image(img, model, cfg)
    assert r is not None
    assert rows(r.board) == EMPTY


def test_a_board_running_off_the_picture_is_refused(cfg: BoardConfig, model: ReaderModel) -> None:
    off = [[900.0, 180.0], [1900.0, 205.0], [1920.0, 1205.0], [880.0, 1190.0]]  # right/bottom cut
    r = read_image(photo(cfg, SAMPLE_BOARD, off), model, cfg)
    assert r is None or r.rejected or "board_cut_off" in r.warnings


# --- the next turn ---------------------------------------------------------------------------


def previous_without(board: list[str], gone: list[tuple[int, int]]) -> list[list[str | None]]:
    return [
        [None if ch == "." or (r, c) in gone else ch for c, ch in enumerate(row)]
        for r, row in enumerate(board)
    ]


NEW = [(4, 4), (4, 5), (9, 3), (9, 4)]  # tiles that appear this turn


def test_next_turn_reads_only_the_new_tiles(cfg: BoardConfig, model: ReaderModel) -> None:
    previous = previous_without(SAMPLE_BOARD, NEW)
    r = read_image(photo(cfg, SAMPLE_BOARD, QUAD), model, cfg, previous=previous)
    assert r is not None and not r.rejected
    assert rows(r.board) == SAMPLE_BOARD
    assert (r.kept_from_previous, r.new_tiles, r.conflicts) == (tiles(SAMPLE_BOARD) - 4, 4, 0)
    assert r.match is not None and r.match > 0.9


def test_next_turn_under_a_colour_cast_is_unchanged(cfg: BoardConfig, model: ReaderModel) -> None:
    previous = previous_without(SAMPLE_BOARD, NEW)
    img = vignetted(tinted(photo(cfg, SAMPLE_BOARD, QUAD), (0.6, 0.8, 1.2)))
    r = read_image(img, model, cfg, previous=previous)
    assert r is not None and rows(r.board) == SAMPLE_BOARD


def test_no_new_tiles_means_none_are_added(cfg: BoardConfig, model: ReaderModel) -> None:
    previous = grid(SAMPLE_BOARD)
    r = read_image(photo(cfg, SAMPLE_BOARD, QUAD), model, cfg, previous=previous)
    assert r is not None and rows(r.board) == SAMPLE_BOARD and r.new_tiles == 0


def test_a_button_is_not_a_new_tile(cfg: BoardConfig, model: ReaderModel) -> None:
    img = with_button(photo(cfg, SAMPLE_BOARD, QUAD), (1090, 700))
    r = read_image(img, model, cfg, previous=grid(SAMPLE_BOARD))
    assert r is not None and rows(r.board) == SAMPLE_BOARD


def test_a_different_game_is_refused(cfg: BoardConfig, model: ReaderModel) -> None:
    other = [
        "...............",
        "...............",
        ".....CRANE.....",
        ".......A.......",
        "...WORDS.......",
        "...............",
        "...............",
        ".......HELLO...",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
    ]
    r = read_image(photo(cfg, SAMPLE_BOARD, QUAD), model, cfg, previous=grid(other))
    assert r is not None and r.rejected
    assert "picture_does_not_match" in r.warnings
    assert rows(r.board) == other  # nothing was changed


def test_a_removed_tile_stays_but_is_flagged(cfg: BoardConfig, model: ReaderModel) -> None:
    """The board on file has a tile the picture doesn't show (a typing slip): keep it, flag it."""
    wrong = [row[:] for row in grid(SAMPLE_BOARD)]
    wrong[2][2] = "X"  # an extra tile on an empty square, joined to nothing
    wrong[4][3] = "S"  # touches QUIZ
    r = read_image(photo(cfg, SAMPLE_BOARD, QUAD), model, cfg, previous=wrong)
    assert r is not None and not r.rejected
    assert r.board[2][2] == "X" and r.board[4][3] == "S"
    assert r.needs_review[2][2] and r.needs_review[4][3]
    assert r.conflicts == 2


def test_a_rack_out_of_frame_is_not_read(cfg: BoardConfig, model: ReaderModel) -> None:
    low = [[420.0, 560.0], [1110.0, 570.0], [1125.0, 1195.0], [400.0, 1190.0]]  # rack below edge
    r = read_image(photo(cfg, SAMPLE_BOARD, low), model, cfg)
    assert r is not None
    assert r.rack == [] and "rack_cut_off" in r.warnings


def test_alignment_survives_a_slightly_wrong_start(cfg: BoardConfig, model: ReaderModel) -> None:
    from wwf.vision.align import refine_alignment
    from wwf.vision.board import find_board

    img = photo(cfg, SAMPLE_BOARD, QUAD)
    fit = find_board(img, cfg)
    assert fit is not None
    shift = np.eye(3)
    shift[0, 2], shift[1, 2] = 0.3, -0.25  # a third of a square off
    off = type(fit)(**{**fit.__dict__, "homography": fit.homography @ shift})
    fixed = refine_alignment(img, off, cfg, model.occupancy)
    err_before = np.abs(np.array(off.quad) - np.array(fit.quad)).max()
    err_after = np.abs(np.array(fixed.quad) - np.array(fit.quad)).max()
    assert err_after < err_before / 2, (err_before, err_after)
    assert cv2 is not None
