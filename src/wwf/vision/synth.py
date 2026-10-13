"""Synthetic tile images: letters rendered from system fonts on tile colours.

They give the letter matcher examples of every letter, including ones the labelled fixtures don't
have yet. Fonts that aren't installed are skipped, so this is a no-op on machines without them.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wwf.vision.reader import CELL, LETTERS

FONT_CANDIDATES = [
    ("/System/Library/Fonts/SFNSRounded.ttf", 0),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    ("/System/Library/Fonts/Helvetica.ttc", 1),  # bold face
    ("/System/Library/Fonts/Avenir Next.ttc", 2),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
]
# tile backgrounds seen in the app (BGR): teal, coral, orange
TILE_COLOURS = [(190, 200, 120), (120, 130, 235), (90, 175, 240)]
LETTER_GREYS = [70, 95, 120]


def available_fonts() -> list[tuple[str, int]]:
    return [(p, i) for p, i in FONT_CANDIDATES if Path(p).exists()]


def render_tile(letter: str, font_path: str, index: int, bg: tuple[int, int, int],
                grey: int, size: int = CELL) -> np.ndarray:  # fmt: skip
    """A CELL x CELL BGR square with the letter centred at roughly app proportions."""
    font = ImageFont.truetype(font_path, int(size * 0.62), index=index)
    im = Image.new("RGB", (size, size), (bg[2], bg[1], bg[0]))
    draw = ImageDraw.Draw(im)
    x0, y0, x1, y1 = draw.textbbox((0, 0), letter, font=font)
    draw.text(((size - (x1 - x0)) / 2 - x0, (size - (y1 - y0)) / 2 - y0), letter,
              font=font, fill=(grey, grey, grey))  # fmt: skip
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def synthetic_tiles() -> list[tuple[np.ndarray, str]]:
    out = []
    for font_path, index in available_fonts():
        for letter in LETTERS:
            for bg in TILE_COLOURS:
                for grey in LETTER_GREYS:
                    out.append((render_tile(letter, font_path, index, bg, grey), letter))
    return out
