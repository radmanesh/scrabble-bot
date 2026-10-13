"""Next turn: a new picture keeps the tiles already on the board and reads only new ones."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from wwf.api.app import create_app
from wwf.api.schemas import GridGeometryOut, RecognitionResultOut
from wwf.api.services import Settings


def _reading(
    rows: dict[int, tuple[int, str]],
    rack: str,
    unsure: frozenset[tuple[int, int]] = frozenset(),
    found: bool = True,
) -> RecognitionResultOut:
    board: list[list[str | None]] = [[None] * 15 for _ in range(15)]
    for r, (c0, word) in rows.items():
        for i, ch in enumerate(word):
            board[r][c0 + i] = ch
    geometry = GridGeometryOut(
        cells=[[(0, 0, 1, 1)] * 15 for _ in range(15)], rack=[], pitch=1.0, confidence=1.0,
        image_width=1, image_height=1,
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
    """Returns whatever reading the test queues next."""

    def __init__(self) -> None:
        self.next: RecognitionResultOut = _reading({}, "")

    def __call__(self, data: bytes) -> RecognitionResultOut:
        return self.next


@pytest.fixture
def reader() -> FakeReader:
    return FakeReader()


@pytest.fixture
def client(reader: FakeReader) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(), recognizer=reader)) as c:
        yield c


def _png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (40, 40), (200, 200, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _state(client: TestClient, sid: str) -> dict:  # type: ignore[type-arg]
    return client.get(f"/api/v1/session/{sid}").json()  # type: ignore[no-any-return]


def _first_turn(client: TestClient, reader: FakeReader) -> str:
    reader.next = _reading({7: (5, "HELLO")}, "ABCDEFG")
    r = client.post(
        "/upload", files={"image": ("a.png", _png(), "image/png")}, follow_redirects=False
    )
    return r.headers["location"].rsplit("/", 1)[1]


def _next_turn(client: TestClient, sid: str, extra: dict[str, str] | None = None) -> str:
    r = client.post(f"/game/{sid}/image", files={"image": ("b.png", _png(), "image/png")},
                    data=extra or {}, follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    return client.get(f"/game/{sid}").text


def test_first_picture_is_read_as_is(client: TestClient, reader: FakeReader) -> None:
    sid = _first_turn(client, reader)
    state = _state(client, sid)
    assert state["board"][7][5:10] == list("HELLO") and state["rack"] == "ABCDEFG"
    assert state["recognition"]["kept_from_previous"] == 0


def test_next_turn_keeps_tiles_and_reads_new_ones(client: TestClient, reader: FakeReader) -> None:
    sid = _first_turn(client, reader)
    # the new picture: HELLO misread as HELLS (confidently), plus a new word down from the H
    reader.next = _reading({7: (5, "HELLS"), 8: (5, "A"), 9: (5, "T")}, "XYZ")
    html = _next_turn(client, sid)
    state = _state(client, sid)
    assert state["board"][7][5:10] == list("HELLO")  # kept, not replaced by the misread S
    assert state["board"][8][5] == "A" and state["board"][9][5] == "T"  # new tiles read
    assert state["rack"] == "XYZ"  # the rack is always read fresh
    rec = state["recognition"]
    assert (rec["kept_from_previous"], rec["new_tiles"], rec["conflicts"]) == (5, 2, 1)
    assert rec["needs_review"][7][9] is True  # the disagreement is flagged for a look
    assert "Kept the 5" in html and "looked different in the new picture" in html


def test_unsaved_grid_edits_are_the_starting_point(client: TestClient, reader: FakeReader) -> None:
    sid = _first_turn(client, reader)
    # the user corrected HELLO to HELLS on the page but never pressed Save
    grid = {f"c-{r}-{c}": "" for r in range(15) for c in range(15)}
    for i, ch in enumerate("HELLS"):
        grid[f"c-7-{5 + i}"] = ch
    reader.next = _reading({7: (5, "HELLS"), 6: (9, "A")}, "Q")
    _next_turn(client, sid, grid)
    state = _state(client, sid)
    assert state["board"][7][5:10] == list("HELLS")
    assert state["board"][6][9] == "A"
    assert state["recognition"]["conflicts"] == 0


def test_board_not_found_keeps_everything(client: TestClient, reader: FakeReader) -> None:
    sid = _first_turn(client, reader)
    reader.next = _reading({}, "", found=False)
    html = _next_turn(client, sid)
    state = _state(client, sid)
    assert state["board"][7][5:10] == list("HELLO")
    assert state["recognition"]["conflicts"] == 0
    assert "find the board in the new picture" in html


def test_unsure_new_tiles_stay_flagged(client: TestClient, reader: FakeReader) -> None:
    sid = _first_turn(client, reader)
    reader.next = _reading({7: (5, "HELLO"), 8: (9, "X")}, "AB", unsure={(8, 9)})
    _next_turn(client, sid)
    rec = _state(client, sid)["recognition"]
    assert rec["needs_review"][8][9] is True and rec["conflicts"] == 0


# --- a reader that is told about the board on file -------------------------------------------


class ContextReader:
    """Records what it is told and returns whatever reading the test queues."""

    def __init__(self) -> None:
        self.next: RecognitionResultOut = _reading({}, "")
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        data: bytes,
        previous: list[list[str | None]] | None = None,
        hint_quad: list[list[float]] | None = None,
    ) -> RecognitionResultOut:
        self.calls.append({"previous": previous, "hint_quad": hint_quad})
        return self.next


@pytest.fixture
def ctx_reader() -> ContextReader:
    return ContextReader()


@pytest.fixture
def ctx_client(ctx_reader: ContextReader) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(), recognizer=ctx_reader)) as c:
        yield c


def test_the_reader_is_told_the_board_on_file(
    ctx_client: TestClient, ctx_reader: ContextReader
) -> None:
    ctx_reader.next = _reading({7: (5, "HELLO")}, "ABCDEFG")
    sid = _first_turn(ctx_client, ctx_reader)  # type: ignore[arg-type]
    assert ctx_reader.calls[0]["previous"] is None  # nothing on file for the first picture

    ctx_reader.next = _reading({7: (5, "HELLO"), 8: (5, "A")}, "XYZ")
    _next_turn(ctx_client, sid)
    told = ctx_reader.calls[1]["previous"]
    assert told is not None and told[7][5:10] == list("HELLO")  # type: ignore[index]
    assert ctx_reader.calls[1]["hint_quad"] is None or isinstance(
        ctx_reader.calls[1]["hint_quad"], list
    )


def test_a_refused_picture_changes_nothing(
    ctx_client: TestClient, ctx_reader: ContextReader
) -> None:
    ctx_reader.next = _reading({7: (5, "HELLO")}, "ABCDEFG")
    sid = _first_turn(ctx_client, ctx_reader)  # type: ignore[arg-type]
    before = _state(ctx_client, sid)

    refused = _reading({}, "")
    refused.rejected = True
    refused.merged = True
    refused.warnings = ["picture_does_not_match"]
    refused.match = 0.55
    ctx_reader.next = refused
    html = _next_turn(ctx_client, sid)
    after = _state(ctx_client, sid)
    assert after["board"] == before["board"] and after["rack"] == before["rack"]
    assert after["has_image"] is True
    assert "doesn't look like this game's board" in html and "55%" in html
    assert "Nothing was changed" in html
    # the notice is shown once
    assert "doesn't look like this game's board" not in ctx_client.get(f"/game/{sid}").text


def test_an_unreadable_rack_leaves_yours_alone(
    ctx_client: TestClient, ctx_reader: ContextReader
) -> None:
    ctx_reader.next = _reading({7: (5, "HELLO")}, "ABCDEFG")
    sid = _first_turn(ctx_client, ctx_reader)  # type: ignore[arg-type]
    cut = _reading({7: (5, "HELLO"), 8: (5, "A")}, "")
    cut.merged = True
    cut.warnings = ["rack_cut_off"]
    ctx_reader.next = cut
    html = _next_turn(ctx_client, sid)
    state = _state(ctx_client, sid)
    assert state["rack"] == "ABCDEFG"
    assert state["board"][8][5] == "A"  # the board part of the reading is still used
    assert "rack is partly outside the picture" in html
