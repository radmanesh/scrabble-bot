"""Shared pieces for the page tests: a scripted board reader and form builders."""

from __future__ import annotations

import io
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from wwf.api.schemas import GridGeometryOut, RecognitionResultOut


def reading(
    rows: dict[int, tuple[int, str]],
    rack: str,
    unsure: frozenset[tuple[int, int]] | set[tuple[int, int]] = frozenset(),
    found: bool = True,
) -> RecognitionResultOut:
    board: list[list[str | None]] = [[None] * 15 for _ in range(15)]
    for r, (c0, word) in rows.items():
        for i, ch in enumerate(word):
            board[r][c0 + i] = ch
    geometry = GridGeometryOut(
        cells=[[(0, 0, 1, 1)] * 15 for _ in range(15)], rack=[], pitch=1.0, confidence=1.0,
        image_width=1, image_height=1,
        quad=[[0.0, 0.0], [15.0, 0.0], [15.0, 15.0], [0.0, 15.0]],
    )  # fmt: skip
    return RecognitionResultOut(
        board=board if found else [[None] * 15 for _ in range(15)],
        rack=list(rack) if found else [],
        confidence=[[0.5 if (r, c) in unsure else 1.0 for c in range(15)] for r in range(15)],
        needs_review=[[(r, c) in unsure for c in range(15)] for r in range(15)],
        geometry=geometry if found else None,
        warnings=[] if found else ["board_not_found"],
    )


class FakeReader:
    """Returns whatever reading the test queues next; remembers what it was told."""

    def __init__(self) -> None:
        self.next: RecognitionResultOut = reading({}, "")
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        data: bytes,
        previous: list[list[str | None]] | None = None,
        hint_quad: list[list[float]] | None = None,
    ) -> RecognitionResultOut:
        self.calls.append({"previous": previous, "hint_quad": hint_quad})
        return self.next


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), (200, 200, 200)).save(buf, format="PNG")
    return buf.getvalue()


def form(rows: dict[int, tuple[int, str]], rack: str, action: str = "save") -> dict[str, str]:
    """What the board form posts: 225 squares, 7 rack tiles and the button pressed."""
    data = {f"c-{r}-{c}": "" for r in range(15) for c in range(15)}
    for r, (c0, word) in rows.items():
        for i, ch in enumerate(word):
            data[f"c-{r}-{c0 + i}"] = ch
    data |= {f"r-{i}": rack[i] if i < len(rack) else "" for i in range(7)}
    return data | {"action": action}


def new_game(client: TestClient) -> str:
    r = client.post("/new", follow_redirects=False)
    assert r.status_code == 303
    return str(r.headers["location"]).rsplit("/", 1)[1]


def started_game(client: TestClient, rack: str = "SATIRE?", phase: str = "start_me") -> str:
    """A game with HELLO across row 8 (one-based) confirmed as the starting position."""
    gid = new_game(client)
    r = client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, rack, phase),
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    return gid


def send_picture(client: TestClient, gid: str, source: str = "photo") -> Any:
    return client.post(
        f"/game/{gid}/image", files={"image": ("b.png", png(), "image/png")},
        data={"source": source}, follow_redirects=False,
    )  # fmt: skip
