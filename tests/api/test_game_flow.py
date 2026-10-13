"""The multi-turn loop: solve, play, the opponent's move, undo, persistence."""

from __future__ import annotations

import html
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.api.test_next_turn import FakeReader, _png, _reading
from wwf.api.app import create_app
from wwf.api.services import Settings
from wwf.core.board import Board, Tile
from wwf.core.config import load_board_config
from wwf.core.move import Placement, build_move
from wwf.core.scoring import score_move

API = "/api/v1"


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


def _form(rows: dict[int, tuple[int, str]], rack: str, action: str = "save") -> dict[str, str]:
    form = {f"c-{r}-{c}": "" for r in range(15) for c in range(15)}
    for r, (c0, word) in rows.items():
        for i, ch in enumerate(word):
            form[f"c-{r}-{c0 + i}"] = ch
    form |= {f"r-{i}": rack[i] if i < len(rack) else "" for i in range(7)}
    return form | {"action": action}


def _start(client: TestClient, rack: str = "SATIRE?") -> str:
    """A new game with HELLO across row 8 (one-based), saved with the given rack."""
    r = client.post("/new", follow_redirects=False)
    gid: str = r.headers["location"].rsplit("/", 1)[1]
    client.post(f"/game/{gid}/board", data=_form({7: (5, "HELLO")}, rack))
    return gid


def _text(response: Any) -> str:
    """The page as plain text: Jinja escapes apostrophes as &#39;."""
    return html.unescape(response.text)


def _game(client: TestClient, gid: str) -> dict[str, Any]:
    return client.get(f"{API}/games/{gid}").json()  # type: ignore[no-any-return]


def _expected_score(grid_rows: list[str], placements: list[tuple[int, int, str]]) -> int:
    cfg = load_board_config()
    board = Board.from_rows(cfg, grid_rows)
    move = build_move(board, [Placement(r, c, Tile.from_char(ch)) for r, c, ch in placements])
    return score_move(board, move, cfg).total


HELLO_ROWS = ["." * 15] * 7 + [".....HELLO....."] + ["." * 15] * 7


# --- your move -------------------------------------------------------------------------------


def test_play_a_solver_move(client: TestClient) -> None:
    gid = _start(client)
    client.post(f"/game/{gid}/solve")
    page = client.get(f"/game/{gid}").text
    assert "Best moves" in page and "Play " in page  # the Play button is on the page

    top = client.post(f"{API}/session/{gid}/solve").json()["moves"][0]
    r = client.post(f"/game/{gid}/play/0", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/game/{gid}"

    g = _game(client, gid)
    assert g["phase"] == "opponent_turn"
    assert g["scores"] == {"me": top["score"]["total"], "opponent": 0}
    assert g["rack"] == top["leave"]
    turn = g["turns"][0]
    assert (turn["player"], turn["word"], turn["source"]) == ("me", top["word"], "solver")
    assert g["board"] == top["board_after"]
    assert "Opponent's turn" in client.get(f"/game/{gid}").text


def test_play_needs_a_solve_first(client: TestClient) -> None:
    gid = _start(client)
    r = client.post(f"/game/{gid}/play/0")
    assert r.status_code == 422 and "solve again" in r.text


# --- the opponent's move ---------------------------------------------------------------------


def _my_pass(client: TestClient, gid: str) -> None:
    client.post(f"/game/{gid}/pass", data={"who": "me"})
    assert _game(client, gid)["phase"] == "opponent_turn"


def test_opponent_typed_word(client: TestClient) -> None:
    gid = _start(client)
    _my_pass(client, gid)
    # OX down from the O of HELLO: the O is at one-based row 8, column 10
    r = client.post(
        f"/game/{gid}/opponent",
        data={"word": "OX", "row": "8", "col": "10", "direction": "down"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    g = _game(client, gid)
    assert g["phase"] == "my_turn"
    t = g["turns"][-1]
    assert (t["player"], t["word"], t["direction"], t["start"]) == (
        "opponent",
        "OX",
        "down",
        [7, 9],
    )
    assert t["points"] == _expected_score(HELLO_ROWS, [(8, 9, "X")])
    assert g["scores"]["opponent"] == t["points"] > 0
    assert g["board"][8][9] == "X"


def test_typed_word_problems_are_explained(client: TestClient) -> None:
    gid = _start(client)
    _my_pass(client, gid)
    wrong_letter = client.post(
        f"/game/{gid}/opponent",
        data={"word": "HALLO", "row": "8", "col": "6", "direction": "across"},
    )
    assert wrong_letter.status_code == 422
    assert "already has E at row 8, column 7" in _text(wrong_letter)

    off_board = client.post(
        f"/game/{gid}/opponent",
        data={"word": "QUIZ", "row": "1", "col": "14", "direction": "across"},
    )
    assert "runs off the board" in off_board.text

    no_contact = client.post(
        f"/game/{gid}/opponent", data={"word": "CAT", "row": "1", "col": "1", "direction": "across"}
    )
    assert "doesn't touch any tile" in _text(no_contact)
    assert _game(client, gid)["turns"][-1]["kind"] == "pass"  # nothing was logged


def test_word_not_in_the_list_can_be_forced(client: TestClient) -> None:
    gid = _start(client)
    _my_pass(client, gid)
    data = {"word": "OZX", "row": "8", "col": "10", "direction": "down"}
    r = client.post(f"/game/{gid}/opponent", data=data)
    assert r.status_code == 422
    assert "OZX isn't in" in _text(r) and 'name="force"' in r.text  # offered, not assumed

    r = client.post(f"/game/{gid}/opponent", data=data | {"force": "1"}, follow_redirects=False)
    assert r.status_code == 303
    t = _game(client, gid)["turns"][-1]
    assert (t["word"], t["forced"]) == ("OZX", True)
    assert 'class="badge"' in client.get(f"/game/{gid}").text


def test_opponent_pass(client: TestClient) -> None:
    gid = _start(client)
    _my_pass(client, gid)
    client.post(f"/game/{gid}/pass", data={"who": "opponent"})
    g = _game(client, gid)
    assert g["phase"] == "my_turn" and g["turns"][-1]["kind"] == "pass"
    assert g["scores"] == {"me": 0, "opponent": 0}


# --- the opponent's move from the board ------------------------------------------------------


def test_grid_edit_is_found_as_the_opponents_move(client: TestClient) -> None:
    gid = _start(client)
    _my_pass(client, gid)
    # they played OX: type the X into the grid and save, as after correcting a photo
    client.post(f"/game/{gid}/board", data=_form({7: (5, "HELLO"), 8: (9, "X")}, "SATIRE?"))
    page = client.get(f"/game/{gid}").text
    assert "Opponent's move found" in html.unescape(page)
    assert "<b>OX</b>" in page and "Log opponent" in page

    r = client.post(f"/game/{gid}/pending", data={"action": "confirm_opponent"},
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    g = _game(client, gid)
    assert g["phase"] == "my_turn"
    t = g["turns"][-1]
    assert (t["player"], t["word"], t["source"]) == ("opponent", "OX", "grid")
    assert t["points"] == _expected_score(HELLO_ROWS, [(8, 9, "X")])


def test_unclear_tiles_are_not_logged(client: TestClient) -> None:
    gid = _start(client)
    _my_pass(client, gid)
    # two stray tiles that are not in one line: a misread, not a move
    client.post(
        f"/game/{gid}/board", data=_form({7: (5, "HELLO"), 5: (2, "Q"), 10: (11, "Z")}, "SATIRE?")
    )
    page = _text(client.get(f"/game/{gid}"))
    assert "aren't in one row or one column" in page
    assert 'value="confirm_opponent"' not in page  # nothing to confirm
    r = client.post(f"/game/{gid}/pending", data={"action": "confirm_opponent"})
    assert r.status_code == 422

    client.post(f"/game/{gid}/pending", data={"action": "accept"})  # just take the board
    assert "move found" not in client.get(f"/game/{gid}").text


def test_my_turn_new_tiles_can_be_logged_either_way(client: TestClient) -> None:
    gid = _start(client)
    _my_pass(client, gid)
    client.post(f"/game/{gid}/pass", data={"who": "opponent"})  # back to my turn, with history
    client.post(f"/game/{gid}/board", data=_form({7: (5, "HELLO"), 8: (9, "X")}, "SATIRE?"))
    page = client.get(f"/game/{gid}").text
    assert "New tiles on the board" in page
    assert 'value="confirm_me"' in page and 'value="confirm_opponent"' in page

    client.post(f"/game/{gid}/pending", data={"action": "confirm_me"})
    g = _game(client, gid)
    assert g["turns"][-1]["player"] == "me" and g["phase"] == "opponent_turn"


def test_first_board_is_not_a_move(client: TestClient) -> None:
    """Before any turn is logged, the board you type or photograph is just the position."""
    gid = _start(client)
    assert "New tiles on the board" not in client.get(f"/game/{gid}").text
    client.post(f"/game/{gid}/board", data=_form({7: (5, "HELLO"), 8: (9, "X")}, "SATIRE?"))
    assert "New tiles on the board" not in client.get(f"/game/{gid}").text


def test_photo_in_the_opponents_turn_finds_their_move(
    client: TestClient, reader: FakeReader
) -> None:
    reader.next = _reading({7: (5, "HELLO")}, "ABCDEFG")
    r = client.post("/upload", files={"image": ("a.png", _png(), "image/png")},
                    follow_redirects=False)  # fmt: skip
    gid = r.headers["location"].rsplit("/", 1)[1]
    _my_pass(client, gid)

    reader.next = _reading({7: (5, "HELLO"), 8: (9, "X")}, "ABCDEFG")
    client.post(f"/game/{gid}/image", files={"image": ("b.png", _png(), "image/png")})
    page = client.get(f"/game/{gid}").text
    assert "<b>OX</b>" in page and "Log opponent" in page
    client.post(f"/game/{gid}/pending", data={"action": "confirm_opponent"})
    g = _game(client, gid)
    assert g["turns"][-1]["word"] == "OX" and g["phase"] == "my_turn"
    assert g["turns"][-1]["source"] == "photo"


# --- undo ------------------------------------------------------------------------------------


def test_undo_restores_the_board_rack_and_turn(client: TestClient) -> None:
    gid = _start(client)
    before = _game(client, gid)
    client.post(f"/game/{gid}/solve")
    client.post(f"/game/{gid}/play/0")
    assert _game(client, gid)["turns"]

    client.post(f"/game/{gid}/undo")
    after = _game(client, gid)
    assert after["turns"] == [] and after["phase"] == "my_turn"
    assert after["board"] == before["board"] and after["rack"] == before["rack"]
    assert after["scores"] == {"me": 0, "opponent": 0}


def test_undo_an_opponent_move_from_the_grid(client: TestClient) -> None:
    gid = _start(client)
    _my_pass(client, gid)
    client.post(f"/game/{gid}/board", data=_form({7: (5, "HELLO"), 8: (9, "X")}, "SATIRE?"))
    client.post(f"/game/{gid}/pending", data={"action": "confirm_opponent"})
    client.post(f"/game/{gid}/undo")
    g = _game(client, gid)
    assert g["phase"] == "opponent_turn" and g["board"][8][9] is None
    assert [t["kind"] for t in g["turns"]] == ["pass"]


def test_nothing_to_undo(client: TestClient) -> None:
    gid = _start(client)
    r = client.post(f"/game/{gid}/undo")
    assert r.status_code == 422 and "Nothing to undo" in r.text


# --- drawing tiles and the bag ---------------------------------------------------------------


def test_draw_prompt_and_adding_tiles(client: TestClient) -> None:
    gid = _start(client)
    client.post(f"/game/{gid}/solve")
    client.post(f"/game/{gid}/play/0")
    kept = _game(client, gid)["rack"]
    client.post(f"/game/{gid}/pass", data={"who": "opponent"})  # their turn is over: back to me

    page = client.get(f"/game/{gid}").text
    assert "What did you draw?" in page
    assert len(kept) < 7

    need = 7 - len(kept)
    r = client.post(f"/game/{gid}/draw", data={"letters": "Z" * need, "action": "add"},
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    assert _game(client, gid)["rack"] == kept + "Z" * need
    assert "What did you draw?" not in client.get(f"/game/{gid}").text  # the rack is full

    too_many = client.post(f"/game/{gid}/draw", data={"letters": "AB"})
    assert too_many.status_code == 422 and "most is 7" in too_many.text


def test_add_and_solve(client: TestClient) -> None:
    gid = _start(client, rack="SAT")
    client.post(f"/game/{gid}/pass", data={"who": "me"})
    client.post(f"/game/{gid}/pass", data={"who": "opponent"})
    assert "What did you draw?" in client.get(f"/game/{gid}").text
    client.post(f"/game/{gid}/draw", data={"letters": "IRE?", "action": "solve"})
    g = _game(client, gid)
    assert g["rack"] == "SATIRE?"
    assert client.get(f"/game/{gid}").text.count("<tr") > 1  # a moves table


def test_bag_counts(client: TestClient) -> None:
    gid = _start(client, rack="SATIRE?")
    g = _game(client, gid)
    assert g["in_bag"] == 104 - 5 - 7 - 7
    assert g["unseen"]["E"] == 13 - 1 - 1 and g["unseen"]["?"] == 1


# --- saved games -----------------------------------------------------------------------------


def test_games_are_listed_renamed_and_deleted(client: TestClient, games_dir: Path) -> None:
    gid = _start(client)
    assert (games_dir / f"{gid}.json").exists()
    home = client.get("/").text
    assert gid in home and "Your games" in home and "Your turn" in home
    assert client.get(f"{API}/games").json()["games"][0]["id"] == gid

    client.post(f"/game/{gid}/rename", data={"name": "Kara, round 2"})
    assert _game(client, gid)["name"] == "Kara, round 2"
    assert "Kara, round 2" in client.get("/").text

    r = client.post(f"/game/{gid}/delete", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert not (games_dir / f"{gid}.json").exists()
    assert client.get(f"/game/{gid}").status_code == 404


def test_games_survive_a_restart(games_dir: Path) -> None:
    with TestClient(create_app(Settings(games_dir=games_dir))) as first:
        gid = _start(first)
        first.post(f"/game/{gid}/solve")
        first.post(f"/game/{gid}/play/0")
        first.post(f"/game/{gid}/pass", data={"who": "opponent"})
        saved = _game(first, gid)

    with TestClient(create_app(Settings(games_dir=games_dir))) as second:
        again = _game(second, gid)
        for key in ("name", "phase", "board", "rack", "scores", "turns", "in_bag"):
            assert again[key] == saved[key], key
        assert second.get(f"/game/{gid}").status_code == 200
        # and the JSON session endpoints see it too
        assert second.get(f"{API}/session/{gid}").json()["expires_in_s"] == -1
        second.post(f"/game/{gid}/undo")  # takes back the opponent's pass
        undone = _game(second, gid)
        assert [t["kind"] for t in undone["turns"]] == ["play"] and undone[
            "phase"
        ] == "opponent_turn"


def test_a_picture_is_kept_with_the_game(games_dir: Path, reader: FakeReader) -> None:
    reader.next = _reading({7: (5, "HELLO")}, "ABCDEFG")
    png = _png()
    with TestClient(create_app(Settings(games_dir=games_dir), recognizer=reader)) as first:
        r = first.post("/upload", files={"image": ("a.png", png, "image/png")},
                       follow_redirects=False)  # fmt: skip
        gid = r.headers["location"].rsplit("/", 1)[1]
        assert (games_dir / f"{gid}.png").exists()
    with TestClient(create_app(Settings(games_dir=games_dir), recognizer=reader)) as second:
        assert second.get(f"{API}/session/{gid}/image").content == png


def test_a_played_move_drops_the_stale_picture(client: TestClient, games_dir: Path,
                                               reader: FakeReader) -> None:  # fmt: skip
    reader.next = _reading({7: (5, "HELLO")}, "SATIRE?")
    r = client.post("/upload", files={"image": ("a.png", _png(), "image/png")},
                    follow_redirects=False)  # fmt: skip
    gid = r.headers["location"].rsplit("/", 1)[1]
    assert client.get(f"{API}/session/{gid}").json()["has_image"] is True
    client.post(f"/game/{gid}/solve")
    client.post(f"/game/{gid}/play/0")
    assert client.get(f"{API}/session/{gid}").json()["has_image"] is False
    assert not list(games_dir.glob(f"{gid}.png"))


def test_game_file_is_plain_json(client: TestClient, games_dir: Path) -> None:
    gid = _start(client)
    data = json.loads((games_dir / f"{gid}.json").read_text())
    assert (
        data["version"] == 1 and data["board"][7] == ".....HELLO....." and data["rack"] == "SATIRE?"
    )
    assert len(data["board"]) == 15 and data["phase"] == "my_turn"


def test_unknown_game(client: TestClient) -> None:
    assert client.get(f"{API}/games/nope").status_code == 404
    assert client.get("/game/nope").status_code == 404
