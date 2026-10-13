"""Fixture collection: save pictures taken in the app as test fixtures, then label them.

A fixture is an image plus a JSON label with the true board and rack, side by side in
assets/fixtures/ (the vision tests read them from there):

    photo-20261012-201500.jpg
    photo-20261012-201500.json   {"image": ..., "input_type": "photo", "status": "verified",
                                  "board": [15 row strings], "rack": "AB?", "labelled_at": ...}

Board rows use "." for an empty square, uppercase for a tile and lowercase for a blank.
`status` is "draft" for a guessed label nobody has checked yet and "verified" once a person has
saved it from the review grid. A label may also carry "board_quad": the board's four corners in
image pixels (top-left, top-right, bottom-right, bottom-left), ground truth for grid detection.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from wwf.api.schemas import BOARD_SIZE, BoardGrid

Source = Literal["photo", "screenshot"]
IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$")


class FixtureError(ValueError):
    pass


@dataclass(frozen=True)
class Fixture:
    name: str
    image: Path
    input_type: str
    status: Literal["none", "draft", "verified"]

    @property
    def labelled(self) -> bool:
        return self.status == "verified"

    @property
    def content_type(self) -> str:
        return IMAGE_TYPES.get(self.image.suffix.lower(), "application/octet-stream")


def grid_to_rows(grid: BoardGrid) -> list[str]:
    return ["".join(c or "." for c in row) for row in grid]


def rows_to_grid(rows: list[str]) -> BoardGrid:
    if len(rows) != BOARD_SIZE or any(len(r) != BOARD_SIZE for r in rows):
        raise FixtureError(f"board must be {BOARD_SIZE} rows of {BOARD_SIZE} characters")
    return [[None if ch in ". " else ch for ch in row] for row in rows]


class FixtureStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _check_name(self, name: str) -> str:
        if not NAME_RE.match(name) or ".." in name:
            raise FixtureError(f"invalid fixture name {name!r}")
        return name

    def save_image(self, data: bytes, content_type: str, source: Source) -> str:
        """Store a picture under a new timestamped name, returned without its extension."""
        self.directory.mkdir(parents=True, exist_ok=True)
        ext = EXTENSIONS.get(content_type, ".jpg")
        base = f"{source}-{datetime.now():%Y%m%d-%H%M%S}"
        name, n = base, 2
        while self._image_path(name) is not None or (self.directory / f"{name}.json").exists():
            name, n = f"{base}-{n}", n + 1
        (self.directory / f"{name}{ext}").write_bytes(data)
        return name

    def _image_path(self, name: str) -> Path | None:
        for ext in IMAGE_TYPES:
            p = self.directory / f"{name}{ext}"
            if p.exists():
                return p
        return None

    def get(self, name: str) -> Fixture:
        image = self._image_path(self._check_name(name))
        if image is None:
            raise FixtureError(f"no fixture image named {name!r}")
        label = self.read_label(name)
        if label is None:
            return Fixture(name, image, _guess_type(name), "none")
        status = "draft" if label.get("status") == "draft" else "verified"
        return Fixture(name, image, str(label.get("input_type", _guess_type(name))), status)

    def list(self) -> list[Fixture]:
        if not self.directory.is_dir():
            return []
        names = sorted(
            p.stem for p in self.directory.iterdir()
            if p.suffix.lower() in IMAGE_TYPES and NAME_RE.match(p.stem)
        )  # fmt: skip
        return [self.get(n) for n in dict.fromkeys(names)]

    def read_label(self, name: str) -> dict[str, object] | None:
        path = self.directory / f"{self._check_name(name)}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise FixtureError(f"{path.name} is not a JSON object")
        return data

    def label_grid(self, name: str) -> tuple[BoardGrid, str] | None:
        label = self.read_label(name)
        if label is None:
            return None
        rows = label.get("board")
        if not isinstance(rows, list):
            raise FixtureError(f"{name}.json has no board")
        return rows_to_grid([str(r) for r in rows]), str(label.get("rack", ""))

    def write_label(self, name: str, grid: BoardGrid, rack: str) -> Path:
        """Save a person-checked label. Extra fields already in the file (board_quad) are kept."""
        fixture = self.get(name)
        path = self.directory / f"{name}.json"
        label = dict(self.read_label(name) or {})
        label.pop("notes", None)  # draft notes no longer apply
        label |= {
            "image": fixture.image.name,
            "input_type": fixture.input_type,
            "status": "verified",
            "board": grid_to_rows(grid),
            "rack": rack,
            "labelled_at": datetime.now().isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(label, indent=2) + "\n")
        return path

    def write_draft(
        self, name: str, grid: BoardGrid, rack: str, quad: list[list[float]] | None, notes: str
    ) -> Path:
        """Save a guessed label for a person to check. Never overwrites an existing label."""
        fixture = self.get(name)
        path = self.directory / f"{name}.json"
        if path.exists():
            return path
        label: dict[str, object] = {
            "image": fixture.image.name,
            "input_type": fixture.input_type,
            "status": "draft",
            "board": grid_to_rows(grid),
            "rack": rack,
            "notes": notes,
        }
        if quad:
            label["board_quad"] = quad
        path.write_text(json.dumps(label, indent=2) + "\n")
        return path


def _guess_type(name: str) -> str:
    return "screenshot" if name.startswith("screenshot") else "photo"
