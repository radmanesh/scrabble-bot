"""Find the board in a photo or screenshot by matching its coloured bonus squares to the layout.

The bonus squares (DL blue, TL green, DW red, TW orange) sit at known cells. We detect blobs of
those colours, vote for the similarity transform (scale, rotation, translation) that puts the most
layout cells of each colour on a blob of that colour, then refine to a full homography from the
matched pairs. Tiles cover some bonus squares, and tiles of similar colours add stray blobs; the
voting tolerates both.

The scale is searched over a wide range set by the image size, not guessed from blob sizes (tile
textures and hands fool that), and several candidate fits are compared by how much the *empty
squares* look like empty squares (cream, in this picture's own light). A wrong scale puts the grid
across letters and tile edges and scores near zero on that. A board that runs off the picture is
reported rather than silently misread.
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
    cream: float = 1.0  # fraction of plain squares that look like empty squares in this picture
    visible: float = 1.0  # fraction of the 225 squares whose centre is inside the picture

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        """Map (N, 2) cell coordinates (col, row) to image pixels."""
        p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), self.homography)
        return p.reshape(-1, 2)

    @property
    def cut_off(self) -> bool:
        """Part of the board is outside the picture."""
        return self.visible < 0.97

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
    """Best translation for a given scale and rotation, and how many blobs it explains.

    Only blobs whose size suits this scale vote: a bonus square is about 0.65 of a cell wide."""
    cos, sin = np.cos(theta), np.sin(theta)
    rot = np.array([[cos, -sin], [sin, cos]]) * pitch
    lo, hi = 0.4 * pitch, 0.95 * pitch
    votes: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for i, b in enumerate(blobs):
        layout = cells.get(b.kind)
        if layout is None or not lo <= b.size <= hi:
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
    for u in uniq[np.argsort(counts)[-3:]]:
        # re-centre on the bin and count blobs within one bin width, so bin edges don't split
        centre = v[key_ids == u].mean(axis=0)
        near = np.linalg.norm(v - centre, axis=1) < bin_size
        n = len(np.unique(o[near]))
        if n > best:
            best, best_t = n, v[near].mean(axis=0)
    return best, best_t


# --- checking a fit against the picture --------------------------------------------------------

CHECK_PX = 24  # pixels per cell for the check
CREAM_DE = 14.0  # Lab distance below which a square counts as "the same cream"


def warp_cells(bgr: np.ndarray, homography: np.ndarray, px: int) -> np.ndarray:
    """The board straightened to 15*px square, `px` pixels per cell."""
    scale = np.diag([1.0 / px, 1.0 / px, 1.0])
    return cv2.warpPerspective(
        bgr, homography @ scale, (15 * px, 15 * px), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_AREA
    )


def cell_medians(img: np.ndarray, px: int, margin: float = 0.18) -> np.ndarray:
    """(15, 15, C) median of each cell's inner region."""
    m = int(margin * px)
    c = img.shape[2] if img.ndim == 3 else 1
    cells = img.reshape(15, px, 15, px, c)[:, m : px - m, :, m : px - m]
    return np.median(cells.transpose(0, 2, 1, 3, 4).reshape(15, 15, -1, c), axis=2)


def cream_colour(colours: np.ndarray, mask: np.ndarray, radius: float = 10.0) -> np.ndarray:
    """The typical colour of the squares in `mask`: the biggest cluster of similar colours.

    `colours` is (15, 15, 3) in Lab, `mask` a (15, 15) bool. For the plain squares this is the
    colour of an empty square in this picture, in whatever light; it works as long as most of
    them are empty (they are, even late in a game)."""
    pts = colours[mask]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    densest = int(np.argmax((d < radius).sum(axis=1)))
    return np.median(pts[d[densest] < radius], axis=0)


def plain_mask(cfg: BoardConfig) -> np.ndarray:
    return np.array([[ch == "." for ch in row] for row in cfg.layout])


def visible_mask(shape: tuple[int, ...], homography: np.ndarray) -> np.ndarray:
    """(15, 15) mask of the squares whose centre lies inside the picture."""
    h, w = shape[:2]
    cols, rows = np.meshgrid(np.arange(15) + 0.5, np.arange(15) + 0.5)
    pts = np.stack([cols.ravel(), rows.ravel()], axis=1).reshape(-1, 1, 2)
    img = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
    inside = (img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h)
    return inside.reshape(15, 15)


def cream_fraction(bgr: np.ndarray, homography: np.ndarray, cfg: BoardConfig) -> float:
    """How much of the board's visible plain squares look like empty cream squares."""
    lab = cv2.cvtColor(warp_cells(bgr, homography, CHECK_PX), cv2.COLOR_BGR2LAB).astype(np.float32)
    col = cell_medians(lab, CHECK_PX)
    plain = plain_mask(cfg) & visible_mask(bgr.shape, homography)
    if plain.sum() < 20:
        return 0.0
    cream = cream_colour(col, plain)
    return float((np.linalg.norm(col - cream, axis=2)[plain] < CREAM_DE).mean())


def visible_fraction(shape: tuple[int, ...], homography: np.ndarray) -> float:
    """Fraction of the squares whose centre lies inside the picture."""
    return float(visible_mask(shape, homography).mean())


def _homography(pitch: float, theta: float, t: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array(
        [[pitch * cos, -pitch * sin, t[0]], [pitch * sin, pitch * cos, t[1]], [0, 0, 1]],
        dtype=np.float64,
    )


def _refine(
    h0: np.ndarray, pitch: float, blobs: list[Blob], cells: dict[str, np.ndarray]
) -> tuple[np.ndarray, int]:
    """Fit a full homography to the blobs that sit on layout cells of their colour."""
    pts = {k: np.array([[b.x, b.y] for b in blobs if b.kind == k]) for k in cells}
    H, matches = h0, 0
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
        H_new, inliers = cv2.findHomography(
            np.float64(src), np.float64(dst), cv2.RANSAC, 0.25 * pitch
        )
        if H_new is None:
            break
        H = H_new
        matches = int(inliers.sum()) if inliers is not None else len(src)
    return H, matches


def _pitch_of(h: np.ndarray) -> float:
    return float(np.hypot(h[0, 0], h[1, 0]))


CREAM_BGR = np.array([222.0, 232.0, 240.0])  # an empty square in a neutrally lit picture
SOLID_MATCHES = 12  # fewer bonus squares than this and a white-balanced second look is tried


def white_balance(bgr: np.ndarray, quantile: float = 90.0) -> np.ndarray:
    """Scale the channels so the brightest region (the lit screen) is neutral cream.

    Under a warm lamp the cream squares fall in the orange hue range of the TW squares and the
    bonus colours cannot be told apart; after this they can."""
    v = bgr.max(axis=2)
    bright = bgr[v >= np.percentile(v, quantile)]
    cream = np.median(bright.reshape(-1, 3), axis=0)
    gain = np.clip(CREAM_BGR / np.maximum(cream, 1.0), 0.5, 2.5)
    return np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def find_boards(
    bgr: np.ndarray,
    cfg: BoardConfig,
    hints: tuple[np.ndarray, ...] = (),
    top: int = 5,
) -> list[BoardFit]:
    """Candidate positions of the board, best first. Falls back to a white-balanced copy of the
    picture when the first look finds no solid fit (a tinted picture)."""
    fits = _find_boards(bgr, cfg, hints, top)
    if fits and fits[0].matches >= SOLID_MATCHES:
        return fits
    balanced = _find_boards(white_balance(bgr), cfg, hints, top)
    if balanced and (not fits or balanced[0].matches > fits[0].matches):
        return balanced
    return fits


def _find_boards(
    bgr: np.ndarray,
    cfg: BoardConfig,
    hints: tuple[np.ndarray, ...] = (),
    top: int = 5,
) -> list[BoardFit]:
    """Candidate positions of the board, best first by votes weighted with cream-likeness.

    Candidate scales come from a wide absolute range (a cell is between 1% and 5% of the
    picture's long side). `hints` are homographies worth trying as they are, such as where the
    board was in the previous picture of the same game."""
    blobs = detect_bonus_blobs(bgr)
    cells = _layout_cells(cfg)
    expected = sum(len(v) for v in cells.values())
    long_side = max(bgr.shape[:2])

    seeds: list[tuple[int, float, float, np.ndarray]] = []
    if len(blobs) >= 6:
        coarse: list[tuple[int, float, float, np.ndarray]] = []
        for pitch in np.geomspace(0.011, 0.05, 26) * long_side:
            for theta in np.deg2rad((-8.0, 0.0, 8.0)):
                n, t = _vote(blobs, cells, float(pitch), float(theta))
                if n >= 5:
                    coarse.append((n, float(pitch), float(theta), t))
        coarse.sort(key=lambda c: -c[0])
        for c in coarse:  # distinct scales only
            if all(abs(c[1] - s_[1]) / s_[1] > 0.08 for s_ in seeds):
                seeds.append(c)
            if len(seeds) >= top:
                break

    found: list[tuple[float, BoardFit]] = []

    def consider(h: np.ndarray, pitch: float, matches: int) -> None:
        cream = cream_fraction(bgr, h, cfg)
        confidence = min(1.0, matches / max(10.0, 0.5 * expected))
        if cream < 0.1:
            confidence *= 0.5
        fit = BoardFit(
            h, pitch, matches, expected, confidence, cream, visible_fraction(bgr.shape, h)
        )
        found.append((max(matches, 1) * (0.1 + cream), fit))

    for n0, pitch0, theta0, t0 in seeds:
        local = (n0, pitch0, theta0, t0)
        for f in (0.96, 1.0, 1.04):
            for dth in np.deg2rad((-3.0, 0.0, 3.0)):
                n, t = _vote(blobs, cells, pitch0 * f, theta0 + float(dth))
                if n > local[0]:
                    local = (n, pitch0 * f, theta0 + float(dth), t)
        n, pitch, theta, t = local
        h, matches = _refine(_homography(pitch, theta, t), pitch, blobs, cells)
        if matches >= 6:
            consider(h, pitch, matches)
    for hint in hints:
        pitch = _pitch_of(hint)
        if not 0.008 * long_side < pitch < 0.06 * long_side:
            continue
        h, matches = _refine(hint, pitch, blobs, cells) if len(blobs) >= 6 else (hint, 0)
        consider(h if matches >= 6 else hint, pitch, matches)
    found.sort(key=lambda f: -f[0])
    return [f for _, f in found]


def find_board(bgr: np.ndarray, cfg: BoardConfig) -> BoardFit | None:
    """The board's most likely position in the picture, or None when no layout fits."""
    fits = find_boards(bgr, cfg)
    return fits[0] if fits else None
