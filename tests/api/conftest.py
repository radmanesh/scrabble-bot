from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.api.helpers import FakeReader
from wwf.api.app import create_app
from wwf.api.games import Game
from wwf.api.services import Services, Settings


@pytest.fixture
def games_dir(tmp_path: Path) -> Path:
    return tmp_path / "games"


@pytest.fixture
def reader() -> FakeReader:
    return FakeReader()


@pytest.fixture
def client(games_dir: Path, reader: FakeReader) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(games_dir=games_dir), recognizer=reader)) as c:
        yield c


@pytest.fixture
def services(client: TestClient) -> Services:
    svc: Services = client.app.state.services  # type: ignore[attr-defined]
    return svc


@pytest.fixture
def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (390, 844), (240, 230, 210)).save(buf, format="PNG")
    return buf.getvalue()


def load(services: Services, game_id: str) -> Game:
    return services.game(game_id)
