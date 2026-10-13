"""Sub-square alignment of a board fit from where the letters sit.

The bonus squares pin the grid down well early in a game, but late in a game most of them are
covered and the fit can drift by a third of a square. Every letter sits at the centre of its
square, so the offsets of the letters from their square centres say how the grid is displaced:
fit a small affine correction to them and apply it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from wwf.core.config import BoardConfig
from wwf.vision.board import BoardFit, visible_mask
from wwf.vision.occupancy import OccupancyModel, measure
from wwf.vision.reader import board_image, cell_crops, glyph_centre

MIN_LETTERS = 14  # fewer than this and the offsets say little
STOP_BELOW = 0.025  # a correction of under 1/40 of a square is not worth applying


def refine_alignment(
    bgr: np.ndarray,
    fit: BoardFit,
    cfg: BoardConfig,
    occupancy: OccupancyModel,
    iterations: int = 2,
) -> BoardFit:
    for _ in range(iterations):
        board = board_image(bgr, fit)
        inside = visible_mask(bgr.shape, fit.homography)
        m = measure(board, cfg, inside)
        p = occupancy.probability(m)
        crops = cell_crops(board)
        rows = []
        for r, c in zip(*np.nonzero((p >= 0.8) & inside), strict=True):
            centre = glyph_centre(crops[r, c])
            if centre is not None:
                rows.append((c + 0.5, r + 0.5, centre[0] - 0.5, centre[1] - 0.5))
        if len(rows) < MIN_LETTERS:
            break
        u, v, dx, dy = (np.array(col) for col in zip(*rows, strict=True))
        design = np.stack([np.ones_like(u), u - 7.5, v - 7.5], axis=1)  # centred on the board
        weights = np.ones_like(u)
        coef_x = coef_y = np.zeros(3)
        for _ in range(3):  # reweighted least squares: letters wider than tall etc. are outliers
            w = np.sqrt(weights)[:, None]
            coef_x = np.linalg.lstsq(design * w, dx * w[:, 0], rcond=None)[0]
            coef_y = np.linalg.lstsq(design * w, dy * w[:, 0], rcond=None)[0]
            resid = np.hypot(dx - design @ coef_x, dy - design @ coef_y)
            weights = 1.0 / np.maximum(resid, 0.05)
        corners = np.array([[0, 0], [15, 0], [15, 15], [0, 15]], np.float64) - 7.5
        shift = np.hypot(
            np.array([1, 0, 0]) @ coef_x + corners @ coef_x[1:],
            np.array([1, 0, 0]) @ coef_y + corners @ coef_y[1:],
        )
        if shift.max() < STOP_BELOW:
            break
        # new cell coordinates (u, v) correspond to old coordinates (u, v) + d(u, v)
        a0, a1, a2 = coef_x
        b0, b1, b2 = coef_y
        t = np.array(
            [
                [1 + a1, a2, a0 - 7.5 * a1 - 7.5 * a2],
                [b1, 1 + b2, b0 - 7.5 * b1 - 7.5 * b2],
                [0.0, 0.0, 1.0],
            ]
        )
        fit = dataclasses.replace(fit, homography=fit.homography @ t)
    return fit
