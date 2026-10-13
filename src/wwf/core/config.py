"""Load config/*.json into typed, validated dataclasses."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_BONUS_CHARS = frozenset(".TDtd*")


class ConfigError(ValueError):
    """Raised when a config file is malformed."""


def project_root() -> Path:
    """Repo root holding config/ and assets/. Overridable with WWF_ROOT."""
    env = os.environ.get("WWF_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return project_root() / "config"


def assets_dir() -> Path:
    return project_root() / "assets"


@dataclass(frozen=True)
class BoardConfig:
    name: str
    size: int
    center: tuple[int, int]
    layout: tuple[str, ...]
    letter_values: dict[str, int]
    bag: dict[str, int]
    bingo_bonus: int
    bingo_tiles: int

    def value(self, letter: str, is_blank: bool = False) -> int:
        return 0 if is_blank else self.letter_values[letter]


def board_config_from_dict(data: dict[str, Any]) -> BoardConfig:
    size = int(data["size"])
    layout = tuple(data["layout"])
    if len(layout) != size:
        raise ConfigError(f"layout has {len(layout)} rows, expected {size}")
    for i, row in enumerate(layout):
        if len(row) != size:
            raise ConfigError(f"layout row {i} has {len(row)} chars, expected {size}")
        bad = set(row) - VALID_BONUS_CHARS
        if bad:
            raise ConfigError(f"layout row {i} has invalid chars {sorted(bad)}")
    if layout != layout[::-1]:
        raise ConfigError("layout is not symmetric top-to-bottom")
    if any(row != row[::-1] for row in layout):
        raise ConfigError("layout is not symmetric left-to-right")
    center = (int(data["center"][0]), int(data["center"][1]))
    if not (0 <= center[0] < size and 0 <= center[1] < size):
        raise ConfigError(f"center {center} is off the board")

    letter_values = {str(k): int(v) for k, v in data["letter_values"].items()}
    missing = [chr(c) for c in range(ord("A"), ord("Z") + 1) if chr(c) not in letter_values]
    if missing:
        raise ConfigError(f"letter_values missing {missing}")

    return BoardConfig(
        name=str(data.get("name", "custom")),
        size=size,
        center=center,
        layout=layout,
        letter_values=letter_values,
        bag={str(k): int(v) for k, v in data["bag"].items()},
        bingo_bonus=int(data["bingo_bonus"]),
        bingo_tiles=int(data["bingo_tiles"]),
    )


def load_board_config(path: Path | None = None) -> BoardConfig:
    path = path or config_dir() / "wwf_classic.json"
    with path.open() as f:
        return board_config_from_dict(json.load(f))
