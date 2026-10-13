"""Read squares and rack tiles once the board has been located.

Two small nearest-neighbour models, both built from labelled fixtures (see train.py):

* occupancy (a rule, no training): an empty plain square is cream with no colour, and an empty
  bonus square has its bonus's colour. So a square is a tile if it is colourful where the layout
  says plain, or a different colour from its bonus square, or (a coral tile on a red DW square)
  the right colour but holding a dark letter.
* letters: the glyph is cut out by taking pixels much darker or much lighter than the tile colour
  (last-played tiles have white letters), centred and scaled with its aspect ratio kept, then
  described with HOG features and matched against labelled glyphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from wwf.vision.board import BONUS_HUES, BoardFit

CELL = 48  # canonical pixels per board square
GLYPH = 24  # glyph image side
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Rack slots in board coordinates (one board square = 1), measured on the current WWF app:
# 7 equal slots below the "letters left" line, filled from the left.
RACK_TOP, RACK_BOTTOM = 17.0, 19.35
RACK_LEFT, RACK_RIGHT = -0.85, 15.85
RACK_SLOTS = 7
RACK_PX = 64  # canonical pixels per rack slot side


# --- cropping ----------------------------------------------------------------------------------


def _warp(bgr: np.ndarray, fit: BoardFit, region: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Warp a quad given in board coordinates (TL, TR, BR, BL) to a `size` (w, h) image."""
    src = fit.to_image(region).astype(np.float32)
    w, h = size
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(src, dst), (w, h),
                               flags=cv2.INTER_AREA)  # fmt: skip


def board_image(bgr: np.ndarray, fit: BoardFit) -> np.ndarray:
    region = np.float64([[0, 0], [15, 0], [15, 15], [0, 15]])
    return _warp(bgr, fit, region, (15 * CELL, 15 * CELL))


def cell_crops(board: np.ndarray) -> np.ndarray:
    """(15, 15, CELL, CELL, 3) crops of the straightened board."""
    return board.reshape(15, CELL, 15, CELL, 3).transpose(0, 2, 1, 3, 4)


def rack_crops(bgr: np.ndarray, fit: BoardFit) -> list[np.ndarray]:
    region = np.float64(
        [[RACK_LEFT, RACK_TOP], [RACK_RIGHT, RACK_TOP], [RACK_RIGHT, RACK_BOTTOM],
         [RACK_LEFT, RACK_BOTTOM]]
    )  # fmt: skip
    strip = _warp(bgr, fit, region, (RACK_SLOTS * RACK_PX, RACK_PX))
    return [strip[:, i * RACK_PX : (i + 1) * RACK_PX] for i in range(RACK_SLOTS)]


def rack_letter_area(slot: np.ndarray) -> np.ndarray:
    """The part of a rack slot that holds the letter: below the tile's darker top edge."""
    h, w = slot.shape[:2]
    return slot[int(0.16 * h) : int(0.96 * h), int(0.05 * w) : int(0.95 * w)]


def slot_has_tile(slot: np.ndarray) -> bool:
    """Rack tiles are coloured; an empty slot shows the cream background."""
    hsv = cv2.cvtColor(slot, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 1] > 55) & (hsv[..., 2] > 90)).mean()) > 0.08


def slot_boxes(fit: BoardFit) -> list[tuple[int, int, int, int]]:
    """Image-space bounding boxes (x, y, w, h) of the rack slots."""
    width = (RACK_RIGHT - RACK_LEFT) / RACK_SLOTS
    boxes = []
    for i in range(RACK_SLOTS):
        x0 = RACK_LEFT + i * width
        quad = np.float64([[x0, RACK_TOP], [x0 + width, RACK_TOP], [x0 + width, RACK_BOTTOM],
                           [x0, RACK_BOTTOM]])  # fmt: skip
        boxes.append(_bbox(fit.to_image(quad)))
    return boxes


def cell_boxes(fit: BoardFit) -> list[list[tuple[int, int, int, int]]]:
    out = []
    for r in range(15):
        row = []
        for c in range(15):
            quad = np.float64([[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]])
            row.append(_bbox(fit.to_image(quad)))
        out.append(row)
    return out


def _bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    x, y, w, h = np.rint([x0, y0, x1 - x0, y1 - y0]).astype(int)
    return int(x), int(y), int(w), int(h)


# --- features ----------------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _hue_distance(h: float, lo: int, hi: int) -> float:
    """Distance from hue h to the range [lo, hi] on OpenCV's 180-degree hue circle."""
    if lo <= h <= hi:
        return 0.0
    d = min(abs(h - lo), abs(h - hi))
    return float(min(d, 180 - d))


def tile_probability(crop: np.ndarray, layout_char: str) -> float:
    """Probability that a board square holds a tile, from colour and the expected bonus square."""
    h, w = crop.shape[:2]
    m = int(0.1 * h)
    inner = crop[m : h - m, m : w - m]
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    # cream squares sit near saturation 20; glare can wash tiles down to ~60
    colourful = (hsv[..., 1] > 55) & (hsv[..., 2] > 90)
    frac = float(colourful.mean())
    p_colour = _sigmoid((frac - 0.18) / 0.04)
    hue_range = BONUS_HUES.get(layout_char)
    if hue_range is None or frac < 0.05:
        return p_colour  # a plain square, or no colour at all
    hues = hsv[..., 0][colourful].astype(np.float32)
    # circular median is overkill: red sits at both ends, so fold hues above 170 to below 0
    hues = np.where(hues > 170, hues - 180, hues)
    p_other_colour = _sigmoid((_hue_distance(float(np.median(hues)), *hue_range) - 4.0) / 1.5)
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark = float((gray < np.percentile(gray, 70) - 45).mean())
    p_letter = _sigmoid((dark - 0.025) / 0.008)  # bonus squares carry white text, never dark
    return p_colour * max(p_other_colour, p_letter)


def glyph_mask(crop: np.ndarray, rack: bool = False) -> np.ndarray | None:
    """The letter as a GLYPH x GLYPH float image (letter bright), or None if there is none.

    Letters are only ~50 grey levels from their textured tile, and last-played tiles have white
    letters, so: subtract a local background (median blur), pick the polarity from the tails,
    threshold adaptively (Otsu) to find the letter's box, then take the contrast values (not the
    binary mask) inside that box so thin strokes survive.
    """
    h, w = crop.shape[:2]
    m = int(0.08 * h)
    inner = cv2.cvtColor(crop[m : h - m, m : w - m], cv2.COLOR_BGR2GRAY)
    k = (inner.shape[0] // 2) | 1
    bg = cv2.medianBlur(inner, k).astype(np.float32)
    diff = inner.astype(np.float32) - bg
    lo, hi = np.percentile(diff, 4), np.percentile(diff, 96)
    letter = -diff if -lo >= hi else diff  # letter strokes now positive
    if letter.max() < 18:
        return None  # nothing stands out from the tile: no letter
    scaled = np.clip(letter, 0, None)
    u8 = (scaled / scaled.max() * 255).astype(np.uint8)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    ih, iw = mask.shape
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.15 * areas.max() or area < 5:
            continue
        cx, cy = cents[i]
        if not (0.1 * iw < cx < 0.9 * iw and 0.1 * ih < cy < 0.9 * ih):
            continue  # specks and tile edges
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if bw > 0.55 * iw and bh < 0.2 * ih:
            continue  # a wide flat band is a tile edge or shading, never a letter
        if rack and cx > 0.62 * iw and cy < 0.38 * ih and area < 0.5 * areas.max():
            continue  # the rack tile's point value
        keep.append(i)
    if not keep:
        return None
    sel = np.isin(labels, keep)
    ys, xs = np.nonzero(sel)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if (y1 - y0) < (0.45 if rack else 0.3) * ih:
        return None  # too small to be a letter (a blank's glossy pattern, specks)
    pad = 1
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(ih, y1 + pad), min(iw, x1 + pad)
    patch = scaled[y0:y1, x0:x1] / (np.percentile(scaled[sel], 90) + 1e-6)
    # centre on a square canvas, keeping the aspect ratio (so I stays thin)
    gh, gw = patch.shape
    side = int(max(gh, gw) * 1.1) + 2
    canvas = np.zeros((side, side), np.float32)
    oy, ox = (side - gh) // 2, (side - gw) // 2
    canvas[oy : oy + gh, ox : ox + gw] = np.clip(patch, 0, 1)
    return cv2.resize(canvas, (GLYPH, GLYPH), interpolation=cv2.INTER_AREA)


def _hog(img: np.ndarray, cell: int = 6, bins: int = 9) -> np.ndarray:
    """Histogram of oriented gradients: cell histograms, L2-normalised over 2x2 cell blocks."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins  # unsigned orientation, 0..bins
    idx = np.minimum(ang.astype(np.int64), bins - 1)
    n = img.shape[0] // cell
    hist = np.zeros((n, n, bins), np.float32)
    for i in range(n):
        for j in range(n):
            sl = (slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell))
            hist[i, j] = np.bincount(idx[sl].ravel(), mag[sl].ravel(), minlength=bins)
    blocks = []
    for i in range(n - 1):
        for j in range(n - 1):
            b = hist[i : i + 2, j : j + 2].ravel()
            blocks.append(b / (np.linalg.norm(b) + 1e-6))
    return np.concatenate(blocks)


def glyph_features(glyph: np.ndarray) -> np.ndarray:
    small = cv2.resize(glyph, (12, 12), interpolation=cv2.INTER_AREA).ravel()
    return np.concatenate([_hog(glyph), small * 0.5]).astype(np.float32)


# --- model -------------------------------------------------------------------------------------


@dataclass
class ReaderModel:
    glyph_x: np.ndarray  # (M, G) glyph features
    glyph_y: np.ndarray  # (M,) letter index 0-25

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, glyph_x=self.glyph_x, glyph_y=self.glyph_y)

    @classmethod
    def load(cls, path: Path) -> ReaderModel:
        with np.load(path) as data:
            return cls(glyph_x=data["glyph_x"], glyph_y=data["glyph_y"])

    def letters(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Best letter index and a 0-1 confidence (margin to the best other letter)."""
        x = np.atleast_2d(feats).astype(np.float32)
        # squared distances via |a|^2 + |b|^2 - 2ab: one matrix product, no (N, M, G) array
        sq = (x**2).sum(1)[:, None] + (self.glyph_x**2).sum(1)[None, :] - 2 * x @ self.glyph_x.T
        d = np.sqrt(np.maximum(sq, 0.0))
        per_class = np.full((len(x), 26), np.inf, dtype=np.float32)
        for k in range(26):
            cols = self.glyph_y == k
            if cols.any():
                per_class[:, k] = d[:, cols].min(axis=1)
        order = np.argsort(per_class, axis=1)
        rows = np.arange(len(x))
        d1, d2 = per_class[rows, order[:, 0]], per_class[rows, order[:, 1]]
        conf = np.clip(1.0 - d1 / np.maximum(d2, 1e-6), 0.0, 1.0) * 2.0
        return order[:, 0], np.clip(conf, 0.0, 1.0).astype(np.float32)
