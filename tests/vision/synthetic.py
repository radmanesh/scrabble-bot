"""Synthetic photos of a board, with known answers, for tests that can't use real fixtures."""

from __future__ import annotations

import cv2
import numpy as np

from wwf.core.config import BoardConfig
from wwf.vision.synth import available_fonts, render_tile

CELL = 40
# BGR colours matching the measured hues of the current app's bonus squares
BONUS_BGR = {
    "d": (220, 160, 60), "t": (70, 200, 150), "D": (80, 90, 230), "T": (60, 140, 240),
    "Q": (143, 54, 176), "q": (134, 143, 31),
}  # fmt: skip
BONUS_LABEL = {"d": "DL", "t": "TL", "D": "DW", "T": "TW", "Q": "QW", "q": "QL"}
CREAM = (225, 235, 242)
OFFSET = {"*": (-60.0, 26.5, -10.0), "?": (-55.0, 35.0, -38.0)}  # the purple squares


def _offset_colour(offset: tuple[float, float, float]) -> tuple[int, int, int]:
    """A square's colour: cream moved by its measured offset in Lab."""
    cream = cv2.cvtColor(np.uint8([[CREAM]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    purple = np.clip(cream + np.array(offset, np.float32), 0, 255).astype(np.uint8)
    b, g, r = cv2.cvtColor(np.uint8([[purple]]), cv2.COLOR_LAB2BGR)[0, 0]
    return int(b), int(g), int(r)


def board_picture(cfg: BoardConfig, board: list[str]) -> np.ndarray:
    """The straightened board: cream squares, coloured bonus squares, teal lettered tiles."""
    font, index = available_fonts()[0]
    img = np.full((15 * CELL, 15 * CELL, 3), (205, 215, 222), np.uint8)
    for r in range(15):
        for c in range(15):
            y, x = r * CELL, c * CELL
            ch = board[r][c]
            if ch != ".":
                tile = render_tile(ch.upper(), font, index, (190, 200, 120), 80, CELL - 2)
                img[y + 1 : y + CELL - 1, x + 1 : x + CELL - 1] = tile
                continue
            cv2.rectangle(img, (x + 2, y + 2), (x + CELL - 3, y + CELL - 3), CREAM, -1)
            kind = cfg.layout[r][c]
            colour = _offset_colour(OFFSET[kind]) if kind in OFFSET else BONUS_BGR.get(kind)
            if colour is not None:
                m = CELL // 7
                cv2.rectangle(img, (x + m, y + m), (x + CELL - m, y + CELL - m), colour, -1)
                if cfg.layout[r][c] == "?":  # a big white question mark in a bright border
                    cv2.rectangle(
                        img, (x + m, y + m), (x + CELL - m, y + CELL - m), (60, 220, 255), 2
                    )
                    cv2.putText(img, "?", (x + CELL // 3, y + CELL - m - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)  # fmt: skip
                else:
                    cv2.putText(img, BONUS_LABEL.get(cfg.layout[r][c], "DL"),
                                (x + m + 2, y + CELL // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, (255, 255, 255), 1)  # fmt: skip
    return img


def photo(cfg: BoardConfig, board: list[str], quad: list[list[float]],
          size: tuple[int, int] = (1600, 1200)) -> np.ndarray:  # fmt: skip
    """The board warped into a dark 'room' at the given corners (TL, TR, BR, BL)."""
    flat = board_picture(cfg, board)
    w, h = size
    src = np.float32([[0, 0], [15 * CELL, 0], [15 * CELL, 15 * CELL], [0, 15 * CELL]])
    M = cv2.getPerspectiveTransform(src, np.float32(quad))
    room = np.full((h, w, 3), (40, 50, 60), np.uint8)
    warped = cv2.warpPerspective(flat, M, (w, h))
    mask = cv2.warpPerspective(np.ones(flat.shape[:2], np.uint8), M, (w, h)) > 0
    room[mask] = warped[mask]
    return room


SAMPLE_BOARD = [
    "...............",
    "...............",
    "...............",
    "...............",
    "....QUIZ.......",
    ".......E.......",
    ".......B.......",
    ".......RAVEN...",
    ".......A.......",
    "...JOLTS.......",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
]


def tinted(img: np.ndarray, gain: tuple[float, float, float]) -> np.ndarray:
    """A colour cast, e.g. a warm lamp: per-channel gain in BGR order."""
    return np.clip(img.astype(np.float32) * np.array(gain, np.float32), 0, 255).astype(np.uint8)


def vignetted(img: np.ndarray, strength: float = 0.45) -> np.ndarray:
    """Darker towards the corners of the picture."""
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot((x - w / 2) / (w / 2), (y - h / 2) / (h / 2))
    return np.clip(img * (1.0 - strength * np.clip(d - 0.3, 0, None))[..., None], 0, 255).astype(
        np.uint8
    )


def with_button(img: np.ndarray, centre: tuple[int, int], size: int = 40) -> np.ndarray:
    """A translucent grey rounded square drawn over the picture, like iOS's assistive touch."""
    out = img.copy()
    x, y = centre
    layer = out.copy()
    cv2.rectangle(layer, (x - size, y - size), (x + size, y + size), (120, 120, 120), -1)
    cv2.circle(layer, (x, y), size // 2, (235, 235, 235), -1)
    return cv2.addWeighted(layer, 0.7, out, 0.3, 0)
