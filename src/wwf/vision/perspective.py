"""Perspective correction: map the board in a photo to a flat, square image.

Milestone 4 adds finding the phone screen and the board automatically; for now the corners come
from a fixture label's "board_quad" (top-left, top-right, bottom-right, bottom-left, in pixels).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

Quad = Sequence[Sequence[float]]


def warp_board(image_bytes: bytes, quad: Quad, size: int = 900) -> np.ndarray:
    """The board straightened into a `size` x `size` BGR image."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    src = np.asarray(quad, dtype=np.float32)
    if src.shape != (4, 2):
        raise ValueError("quad must be four [x, y] points")
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (size, size), flags=cv2.INTER_CUBIC)


def warp_board_png(image_bytes: bytes, quad: Quad, size: int = 900) -> bytes:
    ok, buf = cv2.imencode(".png", warp_board(image_bytes, quad, size))
    if not ok:
        raise ValueError("could not encode image")
    return buf.tobytes()
