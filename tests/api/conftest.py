from __future__ import annotations

import io
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wwf.api.app import create_app
from wwf.api.services import Settings
from wwf.api.session import SessionStore


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def client(clock: FakeClock) -> Iterator[TestClient]:
    settings = Settings()
    app = create_app(settings, sessions=SessionStore(settings.session_ttl_s, clock=clock))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def png_bytes() -> bytes:
    """Stand-in screenshot until real fixtures exist (assets/fixtures/)."""
    buf = io.BytesIO()
    Image.new("RGB", (390, 844), (240, 230, 210)).save(buf, format="PNG")
    return buf.getvalue()


def empty_board() -> list[list[str | None]]:
    return [[None] * 15 for _ in range(15)]


def board_with(row: int, col: int, word: str) -> list[list[str | None]]:
    board = empty_board()
    for i, ch in enumerate(word):
        board[row][col + i] = ch
    return board
