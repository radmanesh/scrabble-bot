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
    r = client.post(
        "/upload", files={"image": ("x", data, "image/jpeg")}, data={"source": source},
        follow_redirects=False,
    )  # fmt: skip
    assert r.status_code == 303
    location: str = r.headers["location"]
    assert location.startswith("/?saved=")  # straight back to the camera
    return location.split("=", 1)[1]


def test_pictures_are_saved_and_listed(client: TestClient, fixtures_dir: Path) -> None:
    photo = _upload(client, _image("JPEG"), "photo")
    shot = _upload(client, _image("PNG"), "auto")
    gallery_photo = _upload(client, _image("JPEG"), "auto")
    assert photo.startswith("photo-") and gallery_photo.startswith("photo-")
    assert shot.startswith("screenshot-")
    assert len({photo, shot, gallery_photo}) == 3  # same-second pictures get distinct names
    assert (fixtures_dir / f"{photo}.jpg").exists() and (fixtures_dir / f"{shot}.png").exists()

    home = client.get(f"/?saved={photo}").text
    assert "Collecting fixtures" in home and "3 so far" in home and photo in home
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
    assert "Save label" in html and name in html
    assert 'value="save"' not in html  # one save path while labelling: no session-only Save

    form = {f"c-{r}-{c}": "" for r in range(15) for c in range(15)}
    for i, ch in enumerate("HELlO"):
        form[f"c-7-{5 + i}"] = ch
    form |= {f"r-{i}": ch for i, ch in enumerate("AB?")} | {"action": "fixture"}
    r = client.post(review, data=form, follow_redirects=False)
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
    with TestClient(create_app()) as c:
        assert c.get("/fixtures").status_code == 404
        assert "Collecting fixtures" not in c.get("/").text


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
    assert "filled in as a guess" in html
    assert 'value="H"' in html  # the draft pre-fills the grid

    client.post(review, data=_form({7: (5, "HELLS")}, "AB", "fixture"))
    label = json.loads((fixtures_dir / f"{name}.json").read_text())
    assert label["status"] == "verified"
    assert label["board"][7] == ".....HELLS....."
    assert label["board_quad"] == [[1, 2], [3, 4], [5, 6], [7, 8]]  # kept
    assert "notes" not in label


def test_save_and_solve_also_writes_the_label(client: TestClient, fixtures_dir: Path) -> None:
    name = _upload(client, _image("JPEG"), "photo")
    review = client.post(f"/fixtures/{name}/open", follow_redirects=False).headers["location"]
    r = client.post(
        review, data=_form({7: (5, "HELLO")}, "SATIRE", "solve"), follow_redirects=False
    )
    assert r.headers["location"].startswith("/results/")
    label = json.loads((fixtures_dir / f"{name}.json").read_text())
    assert label["board"][7] == ".....HELLO....." and label["rack"] == "SATIRE"
