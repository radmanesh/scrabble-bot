"""Deciding which squares hold a tile, in any light.

Fixed colour thresholds fail as soon as the light changes: a warm lamp turns every cream square
orange, glare washes tiles out, vignetting darkens the edges. So everything here is measured
*relative to the same picture*:

* the colour of an empty plain square is whatever the biggest cluster of plain squares looks like
  in this picture, and an empty bonus square should look like the other empty squares of its kind;
* a tile is how far a square is from those references, plus how strongly a letter-shaped stroke
  stands out from the square's own background.

A small logistic model turns those measurements into a probability. It is learned from your
labelled pictures. On a new turn the squares the previous board already tells us about calibrate
the model to this very picture (and say whether the picture matches the board at all).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wwf.core.config import BoardConfig
from wwf.vision.board import cream_colour

N_FEATURES = 7

# How an empty bonus square differs from an empty cream square in Lab, measured over the labelled
# pictures (median; the a and b channels vary by only 1-4 between pictures). A reference built
# this way works even late in a game, when most squares of a kind are covered by tiles.
BONUS_OFFSET = {
    "d": (-49.0, -7.0, -28.0),  # DL blue
    "t": (-32.0, -13.0, 27.0),  # TL green
    "D": (-48.0, 24.0, 12.0),  # DW red
    "T": (-38.0, 19.0, 24.0),  # TW orange
    "*": (-60.0, 26.5, -10.0),  # the centre star, purple
}
L_WEIGHT = 0.5  # brightness varies with exposure, hue does not


@dataclass
class Measures:
    """Per-square measurements of a straightened board, all relative to this picture."""

    de_ref: np.ndarray  # (15, 15) distance from the square's reference colour (cream or its kind)
    de_cream: np.ndarray  # (15, 15) distance from the picture's empty-cream colour
    strokes: np.ndarray  # (15, 15) fraction of pixels that stand out from the square's background
    glyph_h: np.ndarray  # (15, 15) height of the biggest letter-like stroke group, 0 if none
    peak: np.ndarray  # (15, 15) strongest stroke contrast
    bonus: np.ndarray  # (15, 15) bool: a bonus square in the layout
    inside: np.ndarray  # (15, 15) bool: the square is inside the picture

    def features(self) -> np.ndarray:
        """(15, 15, N_FEATURES): the same layout of features for plain and bonus squares."""
        b = self.bonus.astype(np.float32)
        return np.stack(
            [
                self.de_ref / 30.0,
                self.strokes / 0.15,
                self.glyph_h,
                self.peak / 60.0,
                self.de_cream / 30.0 * b,
                b,
                np.ones_like(b),
            ],
            axis=-1,
        ).astype(np.float32)


def _glyph_stats(gray_inner: np.ndarray) -> tuple[float, float]:
    """(height fraction of the biggest letter-like blob, peak contrast) for one square."""
    k = (gray_inner.shape[0] // 2) | 1
    bg = cv2.medianBlur(gray_inner, k).astype(np.float32)
    diff = gray_inner.astype(np.float32) - bg
    lo, hi = np.percentile(diff, 4), np.percentile(diff, 96)
    letter = -diff if -lo >= hi else diff  # letter strokes positive, dark or light alike
    peak = float(letter.max())
    if peak < 10:
        return 0.0, peak
    scaled = np.clip(letter, 0, None)
    u8 = (scaled / scaled.max() * 255).astype(np.uint8)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ih, iw = mask.shape
    best = 0.0
    for i in range(1, n):
        _, _, bw, bh, area = stats[i]
        cx, cy = cents[i]
        if area < 5 or not (0.1 * iw < cx < 0.9 * iw and 0.1 * ih < cy < 0.9 * ih):
            continue
        if bw > 0.55 * iw and bh < 0.2 * ih:
            continue  # a wide flat band: a tile edge or shading
        best = max(best, bh / ih)
    return best, peak


def _nearest_cluster(
    colours: np.ndarray, expected: np.ndarray, weights: np.ndarray, radius: float = 14.0,
    accept: float = 45.0,
) -> np.ndarray | None:  # fmt: skip
    """This picture's own look of a bonus kind: the tight group of squares whose colour is
    nearest to `expected`, if one is near enough.

    Using the squares themselves (rather than a fixed offset from cream) matters because
    screenshots render the colours more saturated than a camera does. Tiles that cover most
    squares of a kind form groups too, but far from the bonus colour, so they are not chosen."""
    if len(colours) < 2:
        return None
    w = colours * weights
    d = np.linalg.norm(w[:, None, :] - w[None, :, :], axis=2)
    best: tuple[float, np.ndarray] | None = None
    for i in range(len(colours)):
        members = d[i] < radius
        if members.sum() < 2:
            continue
        centre = np.median(colours[members], axis=0)
        dist = float(np.linalg.norm((centre - expected) * weights))
        if best is None or dist < best[0]:
            best = (dist, centre)
    return best[1] if best is not None and best[0] < accept else None


def measure(board_bgr: np.ndarray, cfg: BoardConfig, inside: np.ndarray) -> Measures:
    """Measure every square of a straightened board (15 x 15 squares of equal size)."""
    px = board_bgr.shape[0] // 15
    lab = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
    m = int(0.12 * px)
    colour = np.zeros((15, 15, 3), np.float32)
    strokes = np.zeros((15, 15), np.float32)
    glyph_h = np.zeros((15, 15), np.float32)
    peak = np.zeros((15, 15), np.float32)
    for r in range(15):
        for c in range(15):
            ys, xs = slice(r * px + m, (r + 1) * px - m), slice(c * px + m, (c + 1) * px - m)
            colour[r, c] = np.median(lab[ys, xs].reshape(-1, 3), axis=0)
            g = gray[ys, xs]
            strokes[r, c] = float((np.abs(g.astype(np.float32) - np.median(g)) > 28).mean())
            gm = int(0.08 * px)
            inner = gray[r * px + gm : (r + 1) * px - gm, c * px + gm : (c + 1) * px - gm]
            glyph_h[r, c], peak[r, c] = _glyph_stats(inner)

    plain = np.array([[ch == "." for ch in row] for row in cfg.layout]) & inside
    bonus = np.array([[ch in "dtDT*" for ch in row] for row in cfg.layout])
    # The empty-cream colour comes from plain squares that show no letter-like stroke.
    blank = (strokes < 0.07) & (glyph_h < 0.4)
    cream = None
    for use in (plain & blank, plain):
        if use.sum() >= 3:
            cream = cream_colour(colour, use, 10.0)
            break
    if cream is None:
        cream = np.median(colour.reshape(-1, 3), axis=0)
    de_cream = np.linalg.norm(colour - cream, axis=2)
    de_ref = de_cream.copy()
    weights = np.array([L_WEIGHT, 1.0, 1.0], np.float32)
    for kind, offset in BONUS_OFFSET.items():  # includes the centre star
        kmask = np.array([[ch == kind for ch in row] for row in cfg.layout])
        expected = cream + np.array(offset, np.float32)
        ref = _nearest_cluster(colour[kmask & inside], expected, weights) if kind != "*" else None
        ref = expected if ref is None else ref
        de_ref[kmask] = np.linalg.norm((colour[kmask] - ref) * weights, axis=1)
    return Measures(de_ref, de_cream, strokes, glyph_h, peak, bonus, inside)


# --- the model ---------------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    prior: np.ndarray | None = None,
    strength: float = 1.0,
    iterations: int = 30,
) -> np.ndarray:
    """Logistic regression by Newton's method, pulled towards `prior` with weight `strength`."""
    d = x.shape[1]
    w0 = np.zeros(d) if prior is None else prior.astype(np.float64)
    w = w0.copy()
    reg = strength * np.eye(d)
    for _ in range(iterations):
        p = _sigmoid(x @ w)
        grad = x.T @ (p - y) + strength * (w - w0)
        hess = (x * (p * (1 - p))[:, None]).T @ x + reg + 1e-6 * np.eye(d)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w.astype(np.float32)


# Weights learned from the labelled pictures (see wwf.vision.train); used when a model file
# has none. Features: distance from the reference colour, strokes, glyph height, peak contrast,
# distance from cream (bonus squares), is-bonus, bias.
DEFAULT_PLAIN = (3.44, 0.18, 3.47, 5.3, 0.0, 0.0, -9.17)
DEFAULT_BONUS = (9.22, 1.39, 0.66, 2.13, -0.33, -3.56, -3.56)


@dataclass
class OccupancyModel:
    w_plain: np.ndarray  # (N_FEATURES,)
    w_bonus: np.ndarray

    @classmethod
    def default(cls) -> OccupancyModel:
        return cls(np.float32(DEFAULT_PLAIN), np.float32(DEFAULT_BONUS))

    def logits(self, m: Measures, w_plain: np.ndarray | None = None,
               w_bonus: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
        f = m.features()
        wp = self.w_plain if w_plain is None else w_plain
        wb = self.w_bonus if w_bonus is None else w_bonus
        return np.where(m.bonus, f @ wb, f @ wp)

    def probability(self, m: Measures) -> np.ndarray:
        """(15, 15) probability that each square holds a tile; 0 outside the picture."""
        return np.where(m.inside, _sigmoid(self.logits(m)), 0.0)

    @classmethod
    def train(cls, rows: list[tuple[Measures, np.ndarray]]) -> OccupancyModel:
        """Fit from (measures, tile-truth (15,15) bool) pairs, one per labelled picture."""
        parts: dict[bool, list[tuple[np.ndarray, np.ndarray]]] = {False: [], True: []}
        for m, truth in rows:
            f = m.features()
            for is_bonus in (False, True):
                sel = (m.bonus == is_bonus) & m.inside
                parts[is_bonus].append((f[sel], truth[sel].astype(np.float64)))
        weights = []
        for is_bonus in (False, True):
            x = np.concatenate([p[0] for p in parts[is_bonus]])
            y = np.concatenate([p[1] for p in parts[is_bonus]])
            weights.append(fit_logistic(x, y, strength=0.5))
        return cls(weights[0], weights[1])

    def calibrated(self, m: Measures, known: np.ndarray, truth: np.ndarray) -> OccupancyModel:
        """A copy tuned to this picture with the squares whose state is already known."""
        f = m.features()
        out = []
        for is_bonus, w0 in ((False, self.w_plain), (True, self.w_bonus)):
            sel = known & m.inside & (m.bonus == is_bonus)
            y = truth[sel].astype(np.float64)
            if sel.sum() < 15 or y.min() == y.max():
                out.append(w0)  # not enough of both kinds to learn from
                continue
            out.append(fit_logistic(f[sel], y, prior=w0, strength=8.0))
        return OccupancyModel(out[0], out[1])


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve: how often a tile scores above an empty square."""
    pos, neg = scores[positive], scores[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def agreement(model: OccupancyModel, m: Measures, known: np.ndarray, truth: np.ndarray) -> float:
    """How well the model, untuned, tells known tiles from known empties in this picture.

    Near 1.0 when the board really is where the fit says and it is the same game; near 0.5
    (chance) for a wrong fit or a different board."""
    sel = known & m.inside
    if sel.sum() < 20:
        return 0.5
    return auc(model.logits(m)[sel], truth[sel])
