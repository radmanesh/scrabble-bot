from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.vision.synthetic import SAMPLE_BOARD, photo
from wwf.core.config import BoardConfig, assets_dir, load_board_config
from wwf.vision.board import find_board
from wwf.vision.pipeline import Recognizer, decode, read_image
from wwf.vision.reader import (
    ReaderModel,
    board_image,
    cell_crops,
    glyph_features,
    glyph_mask,
    tile_probability,
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


def test_occupancy_rule(current: BoardConfig) -> None:
    img = photo(current, SAMPLE_BOARD, QUAD)
    fit = find_board(img, current)
    assert fit is not None
    crops = cell_crops(board_image(img, fit))
    for r in range(15):
        for c in range(15):
            p = tile_probability(crops[r, c], current.layout[r][c])
            assert (p >= 0.5) == (SAMPLE_BOARD[r][c] != "."), (r, c, p)


def test_glyphs_dark_or_white_and_blanks() -> None:
    font, index = available_fonts()[0]
    dark = glyph_mask(render_tile("E", font, index, (190, 200, 120), 70))
    white = glyph_mask(render_tile("E", font, index, (190, 200, 120), 250))
    assert dark is not None and white is not None
    assert np.abs(glyph_features(dark) - glyph_features(white)).mean() < 0.05
    blank = np.full((48, 48, 3), (190, 200, 120), np.uint8)
    assert glyph_mask(blank) is None


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

FIXTURES = assets_dir() / "fixtures"


def _verified() -> int:
    if not FIXTURES.is_dir():
        return 0
    return sum(
        json.loads(p.read_text()).get("status") == "verified" for p in FIXTURES.glob("*.json")
    )


@pytest.mark.slow
@pytest.mark.skipif(_verified() < 2, reason="needs labelled photos in assets/fixtures")
def test_held_out_accuracy_on_real_photos(current: BoardConfig,
                                          capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    evaluate(load_examples(FIXTURES), current)
    summary = capsys.readouterr().out.strip().splitlines()[-1]
    print(summary)
    accuracy = float(summary.split("accuracy ")[1].split("%")[0])
    assert accuracy >= 97.0, summary
