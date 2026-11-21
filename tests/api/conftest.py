from __future__ import annotations

import io
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.api.helpers import FakeReader
from wwf.api.app import create_app
from wwf.api.games import Game
from wwf.api.services import Services, Settings
from wwf.core.config import cache_dir, config_dir, dictionaries_dir
from wwf.core.dictionary import load_dictionary_registry


@pytest.fixture
def games_dir(tmp_path: Path) -> Path:
    return tmp_path / "games"


@pytest.fixture
def reader() -> FakeReader:
    return FakeReader()


@pytest.fixture
def client(games_dir: Path, reader: FakeReader, tmp_path: Path) -> Iterator[TestClient]:
    # Word lists are copied so that words learned during a test never reach the real files.
    lists = tmp_path / "dictionaries"
    shutil.copytree(dictionaries_dir(), lists)
    registry = load_dictionary_registry(config_dir() / "dictionaries.json", lists, cache_dir())
    app = create_app(Settings(games_dir=games_dir), recognizer=reader, dictionaries=registry)
    with TestClient(app) as c:
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
