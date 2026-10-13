"""Find the board in a photo or screenshot by matching its coloured bonus squares to the layout.

The bonus squares (DL blue, TL green, DW red, TW orange) sit at known cells. We detect blobs of
those colours, vote for the similarity transform (scale, rotation, translation) that puts the most
layout cells of each colour on a blob of that colour, then refine to a full homography from the
matched pairs. Tiles cover some bonus squares, and tiles of similar colours add stray blobs; the
voting tolerates both.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wwf.core.config import BoardConfig

# OpenCV hue (0-180) ranges for each bonus colour, measured on real photos of the current app.
BONUS_HUES: dict[str, tuple[int, int]] = {
    "d": (96, 110),  # DL blue
    "t": (26, 40),  # TL green
    "D": (2, 9),  # DW red
    "T": (10, 16),  # TW orange
}


@dataclass(frozen=True)
class Blob:
    kind: str  # layout character: d t D T
    x: float
    y: float
    size: float  # sqrt(area)


@dataclass(frozen=True)
class BoardFit:
    homography: np.ndarray  # maps board cell coordinates (col, row; 0..15) to image pixels
    pitch: float  # approximate cell size in pixels
    matches: int  # bonus squares that agree with the fit
    expected: int  # bonus squares in the layout
    confidence: float  # 0-1

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        """Map (N, 2) cell coordinates (col, row) to image pixels."""
        p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), self.homography)
        return p.reshape(-1, 2)

    @property
    def quad(self) -> list[list[float]]:
        """Board corners in image pixels: top-left, top-right, bottom-right, bottom-left."""
        corners = np.float64([[0, 0], [15, 0], [15, 15], [0, 15]])
        return [[round(float(x), 1), round(float(y), 1)] for x, y in self.to_image(corners)]


def detect_bonus_blobs(bgr: np.ndarray) -> list[Blob]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    colourful = (s > 90) & (v > 110)
    blobs: list[Blob] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for kind, (lo, hi) in BONUS_HUES.items():
        mask = (colourful & (h >= lo) & (h <= hi)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill the white label text
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            _, _, w, hh, area = stats[i]
            if area < 60 or w < 6 or hh < 6:
                continue
            if not 0.6 < w / hh < 1.6:
                continue  # words of tiles, long UI bars
            if area < 0.45 * w * hh:
                continue  # not a filled square
            blobs.append(Blob(kind, float(cents[i][0]), float(cents[i][1]), float(np.sqrt(area))))
    return blobs


def _layout_cells(cfg: BoardConfig) -> dict[str, np.ndarray]:
    cells: dict[str, list[tuple[float, float]]] = {k: [] for k in BONUS_HUES}
    for r, row in enumerate(cfg.layout):
        for c, ch in enumerate(row):
            if ch in cells:
                cells[ch].append((c + 0.5, r + 0.5))
    return {k: np.array(v, dtype=np.float64) for k, v in cells.items() if v}


def _vote(
    blobs: list[Blob], cells: dict[str, np.ndarray], pitch: float, theta: float
) -> tuple[int, np.ndarray]:
    """Best translation for a given scale and rotation, and how many blobs it explains."""
    cos, sin = np.cos(theta), np.sin(theta)
    rot = np.array([[cos, -sin], [sin, cos]]) * pitch
    votes: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for i, b in enumerate(blobs):
        layout = cells.get(b.kind)
        if layout is None:
            continue
        t = np.array([b.x, b.y]) - layout @ rot.T
        votes.append(t)
        owners.append(np.full(len(t), i))
    if not votes:
        return 0, np.zeros(2)
    v = np.concatenate(votes)
    o = np.concatenate(owners)
    bin_size = pitch * 0.35
    keys = np.floor(v / bin_size).astype(np.int64)
    key_ids = keys[:, 0] * 1_000_003 + keys[:, 1]
    # distinct blobs per bin: a blob may vote into the same bin through several cells
    pairs = np.unique(np.stack([key_ids, o], axis=1), axis=0)
    uniq, counts = np.unique(pairs[:, 0], return_counts=True)
    best, best_t = 0, np.zeros(2)
    for u in uniq[np.argsort(counts)[-5:]]:
        # re-centre on the bin and count blobs within one bin width, so bin edges don't split
        centre = v[key_ids == u].mean(axis=0)
        near = np.linalg.norm(v - centre, axis=1) < bin_size
        n = len(np.unique(o[near]))
        if n > best:
            best, best_t = n, v[near].mean(axis=0)
    return best, best_t


def find_board(bgr: np.ndarray, cfg: BoardConfig) -> BoardFit | None:
    blobs = detect_bonus_blobs(bgr)
    cells = _layout_cells(cfg)
    expected = sum(len(v) for v in cells.values())
    if len(blobs) < 6:
        return None
    side = float(np.median([b.size for b in blobs]))
    # A bonus square fills roughly 55-90% of its cell's width. Search coarsely, then refine
    # around the best guess.
    best = (0, 0.0, 0.0, np.zeros(2))
    for pitch in side / np.linspace(0.55, 0.9, 8):
        for theta in np.deg2rad(np.arange(-16, 17, 4)):
            n, t = _vote(blobs, cells, float(pitch), float(theta))
            if n > best[0]:
                best = (n, float(pitch), float(theta), t)
    _, p0, th0, _ = best
    for pitch in p0 * np.linspace(0.95, 1.05, 5):
        for theta in th0 + np.deg2rad(np.arange(-3, 4, 1)):
            n, t = _vote(blobs, cells, float(pitch), float(theta))
            if n > best[0]:
                best = (n, float(pitch), float(theta), t)
    n, pitch, theta, t = best
    if n < 6:
        return None
    cos, sin = np.cos(theta), np.sin(theta)
    H = np.array(
        [[pitch * cos, -pitch * sin, t[0]], [pitch * sin, pitch * cos, t[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    # Refine: match each blob to the nearest layout cell of its colour, fit a homography, repeat.
    pts = {k: np.array([[b.x, b.y] for b in blobs if b.kind == k]) for k in cells}
    matches = 0
    for tol in (0.45, 0.35, 0.3):
        src, dst = [], []
        for k, layout in cells.items():
            if len(pts.get(k, [])) == 0:
                continue
            proj = cv2.perspectiveTransform(layout.reshape(-1, 1, 2), H).reshape(-1, 2)
            for p in pts[k]:
                d = np.linalg.norm(proj - p, axis=1)
                j = int(np.argmin(d))
                if d[j] < tol * pitch:
                    src.append(layout[j])
                    dst.append(p)
        if len(src) < 6:
            break
        src_a, dst_a = np.float64(src), np.float64(dst)
        H_new, inliers = cv2.findHomography(src_a, dst_a, cv2.RANSAC, 0.25 * pitch)
        if H_new is None:
            break
        H = H_new
        matches = int(inliers.sum()) if inliers is not None else len(src)
    if matches < 6:
        return None
    confidence = min(1.0, matches / max(10.0, 0.5 * expected))
    return BoardFit(H, pitch, matches, expected, confidence)
