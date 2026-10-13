"""recognize(image_bytes) -> RecognitionResult: find the board, read every square and the rack."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from wwf.core.config import BoardConfig, assets_dir
from wwf.vision.board import find_board
from wwf.vision.reader import (
    LETTERS,
    ReaderModel,
    board_image,
    cell_boxes,
    cell_crops,
    glyph_features,
    glyph_mask,
    rack_crops,
    rack_letter_area,
    slot_boxes,
    slot_has_tile,
    tile_probability,
)

REVIEW_BELOW = 0.85  # spec: squares under this confidence are flagged for review
# The letter matcher's raw margin is rescaled so REVIEW_BELOW falls where, on boards held out of
# training, it flags ~90% of wrong letters while leaving most right ones alone (raw margin 0.5).
LETTER_CONF_SCALE = REVIEW_BELOW / 0.5
MAX_SIDE = 2000
DEFAULT_MODEL = assets_dir() / "models" / "reader.npz"

Box = tuple[int, int, int, int]


@dataclass
class Geometry:
    cells: list[list[Box]]
    rack: list[Box]
    pitch: float
    confidence: float
    quad: list[list[float]]
    image_width: int
    image_height: int


@dataclass
class RecognitionResult:
    board: list[list[str | None]]
    rack: list[str]
    confidence: list[list[float]]
    needs_review: list[list[bool]]
    geometry: Geometry | None = None
    warnings: list[str] = field(default_factory=list)
    input_type: str = "photo"
    elapsed_ms: int = 0


def decode(image_bytes: bytes) -> np.ndarray:
    """BGR array with EXIF orientation applied and the long side at most MAX_SIDE."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        scale = MAX_SIDE / max(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        rgb = np.asarray(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_image(bgr: np.ndarray, model: ReaderModel, cfg: BoardConfig) -> RecognitionResult | None:
    fit = find_board(bgr, cfg)
    if fit is None:
        return None
    crops = cell_crops(board_image(bgr, fit))
    p_tile = np.array([[tile_probability(crops[r, c], cfg.layout[r][c]) for c in range(15)]
                       for r in range(15)])  # fmt: skip

    board: list[list[str | None]] = [[None] * 15 for _ in range(15)]
    conf = np.ones((15, 15), dtype=np.float32)
    glyph_cells, glyph_feats = [], []
    for r in range(15):
        for c in range(15):
            p = float(p_tile[r, c])
            conf[r, c] = max(p, 1 - p)
            if p < 0.5:
                continue
            g = glyph_mask(crops[r, c])
            if g is None:
                board[r][c] = "?"  # looks like a tile but no letter could be found
                conf[r, c] = 0.0
                continue
            glyph_cells.append((r, c))
            glyph_feats.append(glyph_features(g))
    if glyph_feats:
        letters, lconf = model.letters(np.stack(glyph_feats))
        for (r, c), li, lc in zip(glyph_cells, letters, lconf, strict=True):
            board[r][c] = LETTERS[li]
            conf[r, c] = min(conf[r, c], min(1.0, float(lc) * LETTER_CONF_SCALE))

    rack: list[str] = []
    for slot in rack_crops(bgr, fit):
        if not slot_has_tile(slot):
            break  # an empty slot: tiles fill the rack from the left
        g = glyph_mask(rack_letter_area(slot), rack=True)
        if g is None:
            rack.append("?")
            continue
        li, _ = model.letters(glyph_features(g)[None])
        rack.append(LETTERS[int(li[0])])

    h, w = bgr.shape[:2]
    geometry = Geometry(cell_boxes(fit), slot_boxes(fit)[: len(rack)], fit.pitch, fit.confidence,
                        fit.quad, w, h)  # fmt: skip
    warnings = [] if fit.confidence > 0.6 else ["low_board_confidence"]
    return RecognitionResult(
        board=[[None if ch is None else ch for ch in row] for row in board],
        rack=rack,
        confidence=conf.round(3).tolist(),
        needs_review=(conf < REVIEW_BELOW).tolist(),
        geometry=geometry,
        warnings=warnings,
    )


def recognize(image_bytes: bytes, model: ReaderModel, cfg: BoardConfig) -> RecognitionResult:
    start = time.perf_counter()
    bgr = decode(image_bytes)
    result = read_image(bgr, model, cfg)
    if result is None:
        result = RecognitionResult(
            board=[[None] * 15 for _ in range(15)],
            rack=[],
            confidence=[[0.0] * 15 for _ in range(15)],
            needs_review=[[False] * 15 for _ in range(15)],
            warnings=["board_not_found"],
        )
    result.elapsed_ms = int((time.perf_counter() - start) * 1000)
    return result


class Recognizer:
    """A loaded reader model, callable on image bytes."""

    def __init__(self, model: ReaderModel, cfg: BoardConfig) -> None:
        self.model = model
        self.cfg = cfg

    def __call__(self, image_bytes: bytes) -> RecognitionResult:
        return recognize(image_bytes, self.model, self.cfg)

    @classmethod
    def load(cls, cfg: BoardConfig, path: Path = DEFAULT_MODEL) -> Recognizer | None:
        if not path.exists():
            return None
        return cls(ReaderModel.load(path), cfg)
