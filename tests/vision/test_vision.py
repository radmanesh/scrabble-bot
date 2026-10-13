from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.vision.synthetic import SAMPLE_BOARD, photo
from wwf.core.config import BoardConfig, assets_dir, load_board_config
from wwf.vision.board import find_board, visible_mask
from wwf.vision.occupancy import OccupancyModel, measure
from wwf.vision.pipeline import Recognizer, decode, read_image
from wwf.vision.reader import (
    CELL,
    ReaderModel,
    board_image,
    centred_crop,
    glyph_features,
    glyph_mask,
    pad_board,
)
from wwf.vision.synth import available_fonts, render_tile
from wwf.vision.train import build_model, evaluate, load_examples

pytestmark = pytest.mark.skipif(not available_fonts(), reason="no system fonts to render letters")

QUAD = [[420.0, 180.0], [1110.0, 205.0], [1125.0, 905.0], [400.0, 890.0]]  # tilted, perspective


@pytest.fixture(scope="module")
def current() -> BoardConfig:
    return load_board_config()  # the current app layout the finder matches against


@pytest.fixture(scope="module")
def synthetic_model(current: BoardConfig) -> ReaderModel:
    return build_model([], current, synthetic=True)  # font-rendered letters only


def test_finds_a_tilted_board(current: BoardConfig) -> None:
    img = photo(current, SAMPLE_BOARD, QUAD)
    fit = find_board(img, current)
    assert fit is not None
    err = np.abs(np.array(fit.quad) - np.array(QUAD)).max()
    assert err < 0.25 * fit.pitch, f"corner error {err:.1f}px"
    assert fit.matches >= 30


def test_no_board_in_a_blank_picture(current: BoardConfig) -> None:
    assert find_board(np.full((800, 600, 3), 128, np.uint8), current) is None


def test_occupancy(current: BoardConfig) -> None:
    img = photo(current, SAMPLE_BOARD, QUAD)
    fit = find_board(img, current)
    assert fit is not None
    m = measure(board_image(img, fit), current, visible_mask(img.shape, fit.homography))
    p = OccupancyModel.default().probability(m)
    for r in range(15):
        for c in range(15):
            assert (p[r, c] >= 0.5) == (SAMPLE_BOARD[r][c] != "."), (r, c, p[r, c])


def test_glyphs_dark_or_white_and_blanks() -> None:
    font, index = available_fonts()[0]
    dark = glyph_mask(render_tile("E", font, index, (190, 200, 120), 70))
    white = glyph_mask(render_tile("E", font, index, (190, 200, 120), 250))
    assert dark is not None and white is not None
    assert np.abs(glyph_features(dark) - glyph_features(white)).mean() < 0.05
    blank = np.full((48, 48, 3), (190, 200, 120), np.uint8)
    assert glyph_mask(blank) is None


def test_a_crop_is_moved_onto_a_letter_the_grid_cut_off() -> None:
    """A grid a quarter of a square too high cuts an L's foot off; re-centring brings it back."""
    font, index = available_fonts()[0]
    tile = render_tile("L", font, index, (190, 200, 120), 70)
    board = np.full((3 * CELL, 3 * CELL, 3), (190, 200, 120), np.uint8)
    board[CELL : 2 * CELL, CELL : 2 * CELL] = tile
    padded = pad_board(board)
    off = int(0.25 * CELL)  # the square the grid thinks is (1, 1) sits this far above the tile
    shifted = np.roll(padded, -off, axis=0)
    lo, hi = CELL // 2 + CELL, CELL // 2 + 2 * CELL  # square (1, 1) in the padded board
    plain = shifted[lo:hi, lo:hi]
    fixed = centred_crop(shifted, 1, 1)
    whole = glyph_mask(tile)
    assert whole is not None and glyph_mask(fixed) is not None and glyph_mask(plain) is not None
    d_fixed = np.abs(glyph_features(glyph_mask(fixed)) - glyph_features(whole)).mean()
    d_plain = np.abs(glyph_features(glyph_mask(plain)) - glyph_features(whole)).mean()
    assert d_fixed < d_plain


def test_a_model_file_with_other_occupancy_features_falls_back(tmp_path: Path) -> None:
    """Weights saved when the occupancy model had fewer features can't be used: use the default."""
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path,
        glyph_x=np.zeros((1, 4), np.float32),
        glyph_y=np.zeros(1, np.int64),
        occ_plain=np.zeros(7, np.float32),
        occ_bonus=np.zeros(7, np.float32),
    )
    model = ReaderModel.load(path)
    assert model.occupancy.w_plain.shape == OccupancyModel.default().w_plain.shape


def test_reads_the_synthetic_board(current: BoardConfig, synthetic_model: ReaderModel) -> None:
    result = read_image(photo(current, SAMPLE_BOARD, QUAD), synthetic_model, current)
    assert result is not None
    got = ["".join(ch or "." for ch in row) for row in result.board]
    assert got == SAMPLE_BOARD
    assert result.geometry is not None and len(result.geometry.cells) == 15


def test_recognize_bytes_and_decode(current: BoardConfig, synthetic_model: ReaderModel,
                                    tmp_path: Path) -> None:  # fmt: skip
    ok, buf = cv2.imencode(
        ".jpg", photo(current, SAMPLE_BOARD, QUAD), [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    assert ok
    path = tmp_path / "reader.npz"
    synthetic_model.save(path)
    recognizer = Recognizer.load(current, path)
    assert recognizer is not None
    result = recognizer(buf.tobytes())
    assert result.warnings == []
    assert result.elapsed_ms < 2000
    assert ["".join(ch or "." for ch in row) for row in result.board] == SAMPLE_BOARD
    assert decode(buf.tobytes()).shape[:2] == (1200, 1600)

    nothing = cv2.imencode(".png", np.full((400, 300, 3), 90, np.uint8))[1].tobytes()
    assert recognizer(nothing).warnings == ["board_not_found"]


# --- real pictures: only when the owner's fixtures are present ---------------------------------

FIXTURES = assets_dir() / "archive"


def _labelled() -> int:
    if not FIXTURES.is_dir():
        return 0
    return len(load_examples(FIXTURES))


needs_pictures = pytest.mark.skipif(
    _labelled() < 5, reason="needs labelled pictures in assets/archive"
)


@pytest.mark.slow
@needs_pictures
def test_held_out_accuracy_on_real_photos(current: BoardConfig,
                                          capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    """Each position is held out of training in turn; the first picture of a game is read
    well, and a simulated next turn (a few tiles removed from the board on file) gains almost
    nothing wrong."""
    import re

    evaluate(load_examples(FIXTURES), current)
    out = capsys.readouterr().out
    print(out.strip().splitlines()[-2:])
    accuracy = float(re.search(r"first picture: ([0-9.]+)%", out).group(1))  # type: ignore[union-attr]
    assert accuracy >= 97.5, out.strip().splitlines()[-2:]
    errors, pictures = (
        int(x)
        for x in re.search(r"next turn:\s+(\d+) errors over (\d+) pictures", out).groups()  # type: ignore[union-attr]
    )
    assert errors <= pictures, f"{errors} next-turn errors over {pictures} pictures"
    assert "0 pictures refused" in out, "a picture of the right game was refused"


@pytest.mark.slow
@needs_pictures
def test_empty_boards_read_empty_in_any_light(current: BoardConfig) -> None:
    """The empty-board photos include a heavily orange-tinted one that once read as 168 tiles."""
    examples = {e.name: e for e in load_examples(FIXTURES)}
    empties = [e for e in examples.values() if all(ch == "." for row in e.board for ch in row)]
    if not empties:
        pytest.skip("no empty-board pictures")
    model = build_model(list(examples.values()), current, synthetic=True)
    from wwf.vision.pipeline import read_image

    for ex in empties:
        r = read_image(ex.image, model, current)
        assert r is not None, ex.name
        assert sum(c is not None for row in r.board for c in row) == 0, ex.name


@pytest.mark.slow
@needs_pictures
def test_a_picture_of_another_game_is_refused(current: BoardConfig) -> None:
    from wwf.vision.pipeline import read_image

    def letters(e: object) -> dict[tuple[int, int], str]:
        return {
            (r, c): ch.upper()
            for r, row in enumerate(e.board)  # type: ignore[attr-defined]
            for c, ch in enumerate(row)
            if ch != "."
        }

    examples = [e for e in load_examples(FIXTURES) if len(letters(e)) > 30]
    pair = None
    for i, a in enumerate(examples):
        for b in examples[i + 1 :]:
            la, lb = letters(a), letters(b)
            shared = la.keys() & lb.keys()
            # the same game keeps its letters from turn to turn; another game does not
            if len(shared) >= 8 and sum(la[k] == lb[k] for k in shared) / len(shared) < 0.4:
                pair = (a, b)
                break
        if pair:
            break
    if pair is None:
        pytest.skip("needs pictures of two different games")
    a, b = pair
    model = build_model(examples, current, synthetic=True)
    previous = [[None if ch == "." else ch.upper() for ch in row] for row in a.board]
    r = read_image(b.image, model, current, previous=previous)
    assert r is not None and r.rejected and "picture_does_not_match" in r.warnings
