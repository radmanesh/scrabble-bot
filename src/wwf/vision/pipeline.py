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
from wwf.vision.align import refine_alignment
from wwf.vision.board import find_boards, visible_mask
from wwf.vision.occupancy import agreement, measure
from wwf.vision.reader import (
    LETTERS,
    ReaderModel,
    board_image,
    cell_boxes,
    centred_crop,
    glyph_features,
    glyph_mask,
    pad_board,
    rack_crops,
    rack_letter_area,
    rack_visible,
    slot_boxes,
    slot_has_tile,
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
    rack_needs_review: list[bool] = field(default_factory=list)
    geometry: Geometry | None = None
    warnings: list[str] = field(default_factory=list)
    input_type: str = "photo"
    elapsed_ms: int = 0
    # reading a new turn against the board already on file
    merged: bool = False  # the previous board was applied here (the caller must not repeat it)
    rejected: bool = False  # the picture does not match that board: nothing was changed
    match: float | None = None  # 0.5 chance .. 1.0 perfect agreement with the board on file
    kept_from_previous: int = 0
    new_tiles: int = 0
    conflicts: int = 0


def decode(image_bytes: bytes) -> np.ndarray:
    """BGR array with EXIF orientation applied and the long side at most MAX_SIDE."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        scale = MAX_SIDE / max(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        rgb = np.asarray(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


MATCH_MIN = 0.80  # below this the picture does not agree with the board we already have
MAX_NEW_TILES = 21  # three full racks: more on one turn is a misreading, not a move
CONFLICT_BELOW = 0.12  # a kept tile that now looks this surely empty is flagged for a look
ALIGN = True  # refine the grid from where the letters sit (see wwf.vision.align)


def _is_small_cluster(marks: np.ndarray, most: int = 6, span: int = 3) -> bool:
    """A few marks all within a small patch: a button or reflection, not a move."""
    n = int(marks.sum())
    if n == 0 or n > most:
        return False
    ys, xs = np.nonzero(marks)
    return int(ys.max() - ys.min()) < span and int(xs.max() - xs.min()) < span


def connected_to_game(tiles: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Squares of `tiles` that touch (through other tiles) one of the `anchors` squares.

    Every tile of a real game is joined to the centre by a chain of tiles, so tiles that are not
    are not tiles: a reflection, a button drawn over the board, a misread."""
    _, labels = cv2.connectedComponents(tiles.astype(np.uint8), connectivity=4)
    keep = {int(v) for v in np.unique(labels[anchors & tiles]) if v != 0}
    return np.isin(labels, list(keep)) & tiles if keep else np.zeros_like(tiles)


def _quad_homography(quad: list[list[float]]) -> np.ndarray:
    """The homography (cell coordinates -> picture pixels) for a stored board quad."""
    src = np.float32([[0, 0], [15, 0], [15, 15], [0, 15]])
    return cv2.getPerspectiveTransform(src, np.float32(quad)).astype(np.float64)


def _empty_result(
    previous: list[list[str | None]] | None, warnings: list[str], **extra: object
) -> RecognitionResult:
    board = [row[:] for row in previous] if previous else [[None] * 15 for _ in range(15)]
    return RecognitionResult(
        board=board,
        rack=[],
        confidence=[[1.0 if ch else 0.0 for ch in row] for row in board],
        needs_review=[[False] * 15 for _ in range(15)],
        warnings=warnings,
        **extra,  # type: ignore[arg-type]
    )


def read_image(
    bgr: np.ndarray,
    model: ReaderModel,
    cfg: BoardConfig,
    previous: list[list[str | None]] | None = None,
    hint_quad: list[list[float]] | None = None,
) -> RecognitionResult | None:
    """Read a picture. With `previous` (the board already on file), tiles stay where they are
    and only squares that were empty are read: tiles never leave a WWF board.

    Returns None only when no board could be found at all."""
    hints = (_quad_homography(hint_quad),) if hint_quad else ()
    fits = find_boards(bgr, cfg, hints=hints)
    if not fits:
        return None

    have_previous = previous is not None and any(any(row) for row in previous)
    prev_tiles = np.array(
        [[bool(ch) for ch in row] for row in previous] if previous else np.zeros((15, 15), bool)
    )
    use_previous = have_previous and int(prev_tiles.sum()) >= 5

    # choose among the candidate positions: with a board on file, the one that agrees with it
    best = None
    for fit in fits[:5]:
        crop_board = board_image(bgr, fit)
        inside = visible_mask(bgr.shape, fit.homography)
        m = measure(crop_board, cfg, inside)
        score = (
            agreement(model.occupancy, m, np.ones((15, 15), bool), prev_tiles)
            if use_previous
            else 0.0
        )
        if best is None or score > best[0]:
            best = (score, fit, crop_board, m)
        if not use_previous:
            break
    assert best is not None
    match, fit, crop_board, m = best
    if ALIGN and not (use_previous and match < MATCH_MIN):
        aligned = refine_alignment(bgr, fit, cfg, model.occupancy)
        if aligned is not fit:
            fit = aligned
            crop_board = board_image(bgr, fit)
            m = measure(crop_board, cfg, visible_mask(bgr.shape, fit.homography))
    h, w = bgr.shape[:2]
    geometry = Geometry(cell_boxes(fit), [], fit.pitch, fit.confidence, fit.quad, w, h)

    if fit.cut_off and (fit.confidence <= 0.6 or fit.cream < 0.45):
        # part of the board is outside the picture and what is inside fits poorly: refuse
        r = _empty_result(previous, ["board_cut_off"], merged=have_previous, rejected=True)
        r.geometry = geometry
        r.kept_from_previous = int(prev_tiles.sum())
        return r

    if use_previous and match < MATCH_MIN:
        r = _empty_result(previous, ["picture_does_not_match"], merged=True, rejected=True)
        r.match, r.geometry = match, geometry
        r.kept_from_previous = int(prev_tiles.sum())
        return r

    occ = model.occupancy
    if use_previous:
        # tune to this picture with what the board on file says about each square; the new
        # tiles are labelled "empty" there, so refit once without the squares that disagree
        occ = occ.calibrated(m, np.ones((15, 15), bool), prev_tiles)
        suspect = ~prev_tiles & (occ.probability(m) >= 0.5)
        occ = model.occupancy.calibrated(m, ~suspect, prev_tiles)
    p = occ.probability(m)

    candidates = (p >= 0.5) & ~prev_tiles if use_previous else (p >= 0.5)
    warnings: list[str] = []
    centre = np.zeros((15, 15), bool)
    centre[cfg.center] = True
    stray = np.zeros((15, 15), bool)
    if use_previous:
        # a new tile must join the tiles on the board (through other new tiles)
        joined = connected_to_game(prev_tiles | candidates, prev_tiles | centre)
        stray = candidates & ~joined
        candidates = candidates & joined
    else:
        joined = connected_to_game(candidates, centre)
        stray = candidates & ~joined  # kept, but flagged: nothing links it to the game
        if not candidates[cfg.center] and _is_small_cluster(candidates):
            # an empty centre square means an empty board, whatever stray marks were seen
            # (a button drawn over the board, a reflection)
            candidates = np.zeros((15, 15), bool)
            stray = np.zeros((15, 15), bool)
            warnings.append("ignored_marks")
    if int(stray.sum()):
        warnings.append("stray_tiles")
    if use_previous and int(candidates.sum()) > MAX_NEW_TILES:
        warnings.append("too_many_new_tiles")
        candidates = np.zeros((15, 15), bool)
    if fit.cut_off:
        warnings.append("board_cut_off")
    if fit.confidence <= 0.6:
        warnings.append("low_board_confidence")

    padded = pad_board(crop_board)
    board: list[list[str | None]] = [[None] * 15 for _ in range(15)]
    conf = np.ones((15, 15), dtype=np.float32)
    glyph_cells, glyph_feats = [], []
    for r in range(15):
        for c in range(15):
            if use_previous and prev_tiles[r, c]:
                board[r][c] = previous[r][c]  # type: ignore[index]
                lost = bool(m.inside[r, c]) and p[r, c] < CONFLICT_BELOW
                conf[r, c] = 0.5 if lost else 1.0
                continue
            if not m.inside[r, c]:
                conf[r, c] = 0.0 if not use_previous else 1.0  # outside the picture: unknown
                continue
            if not candidates[r, c]:
                conf[r, c] = max(1.0 - float(p[r, c]), 0.0) if p[r, c] > 0.15 else 1.0
                continue
            g = glyph_mask(centred_crop(padded, r, c))
            if g is None:
                board[r][c] = "?"  # looks like a tile but no letter could be found
                conf[r, c] = 0.0
                continue
            glyph_cells.append((r, c))
            glyph_feats.append(glyph_features(g))
            conf[r, c] = 0.3 if stray[r, c] else float(p[r, c])
    if glyph_feats:
        letters, lconf = model.letters(np.stack(glyph_feats))
        for (r, c), li, lc in zip(glyph_cells, letters, lconf, strict=True):
            board[r][c] = LETTERS[li]
            conf[r, c] = min(conf[r, c], min(1.0, float(lc) * LETTER_CONF_SCALE))

    rack: list[str] = []
    rack_unsure: list[bool] = []
    rack_slots = rack_crops(bgr, fit) if rack_visible(fit, bgr.shape) >= 0.9 else []
    if not rack_slots:
        warnings.append("rack_cut_off")
    for slot in rack_slots:
        if not slot_has_tile(slot):
            break  # an empty slot: tiles fill the rack from the left
        g = glyph_mask(rack_letter_area(slot), rack=True)
        if g is None:
            rack.append("?")  # a tile with no letter: a blank
            rack_unsure.append(False)
            continue
        li, lc = model.letters(glyph_features(g)[None])
        rack.append(LETTERS[int(li[0])])
        rack_unsure.append(min(1.0, float(lc[0]) * LETTER_CONF_SCALE) < REVIEW_BELOW)

    geometry.rack = slot_boxes(fit)[: len(rack)]
    new_tiles = int(candidates.sum())
    kept = int(prev_tiles.sum()) if use_previous else 0
    conflicts = int(sum(1 for r in range(15) for c in range(15)
                        if use_previous and prev_tiles[r, c] and conf[r, c] == 0.5))  # fmt: skip
    return RecognitionResult(
        board=board,
        rack=rack,
        confidence=conf.round(3).tolist(),
        needs_review=(conf < REVIEW_BELOW).tolist(),
        rack_needs_review=rack_unsure,
        geometry=geometry,
        warnings=warnings,
        merged=use_previous,
        kept_from_previous=kept,
        new_tiles=new_tiles if use_previous else 0,
        conflicts=conflicts,
        match=match if use_previous else None,
    )


def recognize(
    image_bytes: bytes,
    model: ReaderModel,
    cfg: BoardConfig,
    previous: list[list[str | None]] | None = None,
    hint_quad: list[list[float]] | None = None,
) -> RecognitionResult:
    start = time.perf_counter()
    bgr = decode(image_bytes)
    result = read_image(bgr, model, cfg, previous, hint_quad)
    if result is None:
        result = _empty_result(previous, ["board_not_found"])
        result.merged = previous is not None and any(any(row) for row in previous)
        result.kept_from_previous = sum(bool(ch) for row in previous or [] for ch in row)
    result.elapsed_ms = int((time.perf_counter() - start) * 1000)
    return result


class Recognizer:
    """A loaded reader model, callable on image bytes."""

    def __init__(self, model: ReaderModel, cfg: BoardConfig) -> None:
        self.model = model
        self.cfg = cfg

    def __call__(
        self,
        image_bytes: bytes,
        previous: list[list[str | None]] | None = None,
        hint_quad: list[list[float]] | None = None,
    ) -> RecognitionResult:
        return recognize(image_bytes, self.model, self.cfg, previous, hint_quad)

    @classmethod
    def load(cls, cfg: BoardConfig, path: Path = DEFAULT_MODEL) -> Recognizer | None:
        if not path.exists():
            return None
        return cls(ReaderModel.load(path), cfg)
