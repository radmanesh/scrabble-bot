"""Fixture collection mode (wwf-serve --collect-fixtures)."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wwf.api.app import create_app
from wwf.api.fixtures import FixtureError, FixtureStore
from wwf.api.services import Settings


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    return tmp_path / "fixtures"


@pytest.fixture
def client(fixtures_dir: Path) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(fixtures_dir=fixtures_dir))) as c:
        yield c


def _image(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300, 400), (90, 120, 150)).save(buf, format=fmt)
    return buf.getvalue()


def _upload(client: TestClient, data: bytes, source: str) -> str:
    """Take a picture in "collect pictures" mode: it is saved and the camera comes back."""
    r = client.post(
        "/upload",
        files={"image": ("x", data, "image/jpeg")},
        data={"source": source, "collect": "1"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    location: str = r.headers["location"]
    assert location.startswith("/?collect=1&saved=")  # straight back to the camera
    return location.split("saved=", 1)[1]


def test_pictures_are_saved_and_listed(client: TestClient, fixtures_dir: Path) -> None:
    photo = _upload(client, _image("JPEG"), "photo")
    shot = _upload(client, _image("PNG"), "auto")
    gallery_photo = _upload(client, _image("JPEG"), "auto")
    assert photo.startswith("photo-") and gallery_photo.startswith("photo-")
    assert shot.startswith("screenshot-")
    assert len({photo, shot, gallery_photo}) == 3  # same-second pictures get distinct names
    assert (fixtures_dir / f"{photo}.jpg").exists() and (fixtures_dir / f"{shot}.png").exists()

    home = client.get(f"/?collect=1&saved={photo}").text
    assert "Collecting test pictures" in home and "3 so far" in home and photo in home
    assert str(fixtures_dir) in home

    page = client.get("/fixtures").text
    assert page.count('class="card fixture') == 3
    assert "0 checked" in page
    assert (
        client.get(f"/fixtures/{photo}/image").content
        == (fixtures_dir / f"{photo}.jpg").read_bytes()
    )


def test_label_round_trip(client: TestClient, fixtures_dir: Path) -> None:
    name = _upload(client, _image("JPEG"), "photo")
    r = client.post(f"/fixtures/{name}/open", follow_redirects=False)
    review = r.headers["location"]
    html = client.get(review).text
    assert "Label a test picture" in html and "Play this position as a game" in html
    assert "Save label" in html and name in html
    assert 'value="save"' not in html  # one save path while labelling: no session-only Save

    form = {f"c-{r}-{c}": "" for r in range(15) for c in range(15)}
    for i, ch in enumerate("HELlO"):
        form[f"c-7-{5 + i}"] = ch
    form |= {f"r-{i}": ch for i, ch in enumerate("AB?")} | {"action": "fixture"}
    r = client.post(f"{review}/board", data=form, follow_redirects=False)
    assert r.headers["location"] == f"/fixtures?labelled={name}"

    label = json.loads((fixtures_dir / f"{name}.json").read_text())
    assert label["image"] == f"{name}.jpg"
    assert label["input_type"] == "photo"
    assert label["status"] == "verified"
    assert label["board"][7] == ".....HELlO....."
    assert label["board"][0] == "." * 15
    assert label["rack"] == "AB?"
    assert "1 checked" in client.get("/fixtures").text

    # reopening loads the saved label back into the grid
    review2 = client.post(f"/fixtures/{name}/open", follow_redirects=False).headers["location"]
    state = client.get("/api/v1/session/" + review2.rsplit("/", 1)[1]).json()
    assert state["board"][7][5:10] == ["H", "E", "L", "l", "O"]
    assert state["rack"] == "AB?"


def test_fixture_mode_off_by_default() -> None:
    with TestClient(create_app(Settings())) as c:
        assert c.get("/fixtures").status_code == 404
        assert "test pictures" not in c.get("/").text


def test_names_cannot_escape_the_folder(client: TestClient, tmp_path: Path) -> None:
    store = FixtureStore(tmp_path)
    for bad in ("../secret", "a/b", ".hidden", ""):
        with pytest.raises(FixtureError):
            store.get(bad)
    assert client.post("/fixtures/..%2Fx/open").status_code == 404


def _form(rows: dict[int, tuple[int, str]], rack: str, action: str) -> dict[str, str]:
    form = {f"c-{r}-{c}": "" for r in range(15) for c in range(15)}
    for r, (c0, word) in rows.items():
        for i, ch in enumerate(word):
            form[f"c-{r}-{c0 + i}"] = ch
    return form | {f"r-{i}": ch for i, ch in enumerate(rack)} | {"action": action}


def test_draft_is_shown_then_verified_keeping_extra_fields(
    client: TestClient, fixtures_dir: Path
) -> None:
    name = _upload(client, _image("JPEG"), "photo")
    draft = {
        "image": f"{name}.jpg", "input_type": "photo", "status": "draft",
        "board": ["." * 15] * 7 + [".....HELLO....."] + ["." * 15] * 7, "rack": "AB",
        "board_quad": [[1, 2], [3, 4], [5, 6], [7, 8]], "notes": "guessed",
    }  # fmt: skip
    (fixtures_dir / f"{name}.json").write_text(json.dumps(draft))
    page = client.get("/fixtures").text
    assert "draft: check it" in page and "Check draft" in page and "1 with a guessed draft" in page

    review = client.post(f"/fixtures/{name}/open", follow_redirects=False).headers["location"]
    html = client.get(review).text
    assert "filled in automatically" in html
    assert 'value="H"' in html  # the draft pre-fills the grid

    client.post(f"{review}/board", data=_form({7: (5, "HELLS")}, "AB", "fixture"))
    label = json.loads((fixtures_dir / f"{name}.json").read_text())
    assert label["status"] == "verified"
    assert label["board"][7] == ".....HELLS....."
    assert label["board_quad"] == [[1, 2], [3, 4], [5, 6], [7, 8]]  # kept
    assert "notes" not in label


def test_save_and_solve_also_writes_the_label(client: TestClient, fixtures_dir: Path) -> None:
    name = _upload(client, _image("JPEG"), "photo")
    review = client.post(f"/fixtures/{name}/open", follow_redirects=False).headers["location"]
    r = client.post(
        f"{review}/board",
        data=_form({7: (5, "HELLO")}, "SATIRE", "solve"),
        follow_redirects=False,
    )
    assert r.headers["location"].startswith("/game/")
    label = json.loads((fixtures_dir / f"{name}.json").read_text())
    assert label["board"][7] == ".....HELLO....." and label["rack"] == "SATIRE"


# --- with the board reader -----------------------------------------------------------------


def _reader_app(tmp_path: Path) -> tuple[TestClient, Path, Path]:
    import pytest as _pytest

    from wwf.core.config import load_board_config
    from wwf.vision.synth import available_fonts
    from wwf.vision.train import build_model

    if not available_fonts():
        _pytest.skip("no system fonts to render letters")
    model_path = tmp_path / "reader.npz"
    build_model([], load_board_config(), synthetic=True).save(model_path)
    fixtures_dir = tmp_path / "fx"
    client = TestClient(create_app(Settings(fixtures_dir=fixtures_dir, model_path=model_path)))
    return client, fixtures_dir, model_path


def _synthetic_jpeg() -> bytes:
    import cv2

    from tests.vision.synthetic import SAMPLE_BOARD, photo
    from wwf.core.config import load_board_config

    quad = [[420.0, 180.0], [1110.0, 205.0], [1125.0, 905.0], [400.0, 890.0]]
    return cv2.imencode(".jpg", photo(load_board_config(), SAMPLE_BOARD, quad))[1].tobytes()


def test_capture_is_read_into_a_draft(tmp_path: Path) -> None:
    from tests.vision.synthetic import SAMPLE_BOARD

    client, fixtures_dir, _ = _reader_app(tmp_path)
    with client:
        assert client.get("/api/v1/health").json()["model_loaded"] is True
        name = _upload(client, _synthetic_jpeg(), "photo")
        label = json.loads((fixtures_dir / f"{name}.json").read_text())
        assert label["status"] == "draft"
        assert label["board"] == SAMPLE_BOARD
        assert len(label["board_quad"]) == 4
        assert "draft: check it" in client.get("/fixtures").text

        review = client.post(f"/fixtures/{name}/open", follow_redirects=False).headers["location"]
        html = client.get(review).text
        assert "filled in automatically" in html and "Board read from the picture" in html


def test_saving_a_label_retrains_the_reader(tmp_path: Path) -> None:
    import time

    client, fixtures_dir, model_path = _reader_app(tmp_path)
    with client:
        name = _upload(client, _synthetic_jpeg(), "photo")
        before = model_path.stat().st_mtime_ns
        review = client.post(f"/fixtures/{name}/open", follow_redirects=False).headers["location"]
        form = _form({4: (4, "QUIZ"), 7: (7, "RAVEN"), 9: (3, "JOLTS")}, "AB", "fixture")
        for r, ch in ((5, "E"), (6, "B"), (8, "A")):
            form[f"c-{r}-7"] = ch
        client.post(f"{review}/board", data=form)
        assert json.loads((fixtures_dir / f"{name}.json").read_text())["status"] == "verified"
        svc = client.app.state.services  # type: ignore[attr-defined]
        deadline = time.time() + 60
        while (svc.retraining or svc.reader_trained_at is None) and time.time() < deadline:
            time.sleep(0.2)
        assert svc.reader_trained_at is not None, "retraining did not finish"
        assert model_path.stat().st_mtime_ns > before


def test_a_normal_upload_while_collecting_is_a_game_and_a_fixture(
    client: TestClient, fixtures_dir: Path
) -> None:
    r = client.post("/upload", files={"image": ("x.jpg", _image("JPEG"), "image/jpeg")},
                    data={"source": "photo"}, follow_redirects=False)  # fmt: skip
    assert r.headers["location"].startswith("/game/")  # you stay in the game
    assert len(list(fixtures_dir.glob("photo-*.jpg"))) == 1  # and the picture is kept as data
    gid = r.headers["location"].rsplit("/", 1)[1]
    page = client.get(f"/game/{gid}").text
    assert "Label a test picture" not in page  # it is a game, not a labelling page
    assert gid in client.get("/").text  # and it is listed


def test_play_from_a_picture_makes_a_saved_game(client: TestClient, fixtures_dir: Path) -> None:
    name = _upload(client, _image("JPEG"), "photo")
    (fixtures_dir / f"{name}.json").write_text(
        json.dumps({
            "image": f"{name}.jpg", "input_type": "photo", "status": "verified",
            "board": ["." * 15] * 7 + [".....HELLO....."] + ["." * 15] * 7, "rack": "SATIRE?",
        })
    )  # fmt: skip
    r = client.post(f"/fixtures/{name}/play", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/game/")
    gid = r.headers["location"].rsplit("/", 1)[1]
    state = client.get(f"/api/v1/games/{gid}").json()
    assert state["board"][7][5:10] == list("HELLO") and state["rack"] == "SATIRE?"
    assert gid in client.get("/").text
    # labelling it is untouched
    assert json.loads((fixtures_dir / f"{name}.json").read_text())["status"] == "verified"


def test_labelling_page_marks_squares_the_picture_contradicts(tmp_path: Path) -> None:
    from tests.vision.synthetic import SAMPLE_BOARD

    client, fixtures_dir, _ = _reader_app(tmp_path)
    with client:
        name = _upload(client, _synthetic_jpeg(), "photo")
        board = [list(row) for row in SAMPLE_BOARD]
        board[1][1] = "X"  # a tile the picture does not show
        board[3][2] = "Y"
        (fixtures_dir / f"{name}.json").write_text(
            json.dumps({
                "image": f"{name}.jpg", "input_type": "photo", "status": "verified",
                "board": ["".join(r) for r in board], "rack": "",
            })
        )  # fmt: skip
        review = client.post(f"/fixtures/{name}/open", follow_redirects=False).headers["location"]
        html = client.get(review).text
        assert html.count("contradiction") >= 2 and "disagree with the" in html
        assert "row 2, column 2" in html
