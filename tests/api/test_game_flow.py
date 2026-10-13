"""The turn loop: an editable open turn that is locked when confirmed."""

from __future__ import annotations

import html as htmllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.api.helpers import FakeReader, form, new_game, reading, send_picture, started_game
from wwf.api.games import GameStore
from wwf.api.services import Services


def _text(response: object) -> str:
    return htmllib.unescape(response.text)  # type: ignore[attr-defined]


@pytest.fixture
def retrains(services: Services, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(services, "schedule_retrain", lambda: calls.append(1) or True)
    return calls


def _word(client: TestClient, gid: str, word: str, row: int, col: int, direction: str,
          player: str) -> object:  # fmt: skip
    return client.post(
        f"/game/{gid}/word",
        data={"word": word, "row": row, "col": col, "direction": direction, "player": player},
        follow_redirects=False,
    )


# --- the starting position ------------------------------------------------------------------


def test_starting_position_locks_the_board(client: TestClient, services: Services) -> None:
    gid = started_game(client)
    game = services.game(gid)
    assert game.started and len(game.turns) == 1
    start = game.turns[0]
    assert (start.kind, start.tiles, start.points) == ("start", 5, 0)
    assert game.phase == "my_turn" and game.base_board[7][5:10] == list("HELLO")
    html = client.get(f"/game/{gid}").text
    assert html.count(" readonly") == 5  # the five locked squares


def test_locked_squares_ignore_edits(client: TestClient, services: Services) -> None:
    gid = started_game(client)
    # the page posts HELLS and an extra tile; the locked S-for-O edit is dropped, the new tile kept
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLS"), 8: (5, "A")}, "SATIRE?"))
    game = services.game(gid)
    assert game.board[7][5:10] == list("HELLO") and game.board[8][5] == "A"
    assert game.open_source == "grid"


def test_start_with_opponent_to_move(client: TestClient, services: Services) -> None:
    gid = started_game(client, phase="start_opponent")
    assert services.game(gid).phase == "opponent_turn"
    assert "Turn 2" in client.get(f"/game/{gid}").text


# --- your move: suggestion, then confirm ----------------------------------------------------


def test_play_a_suggestion_is_an_open_turn_until_confirmed(
    client: TestClient, services: Services
) -> None:
    gid = started_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "solve"))
    r = client.post(f"/game/{gid}/play/0", follow_redirects=False)
    assert r.status_code == 303
    game = services.game(gid)
    assert len(game.turns) == 1  # nothing logged yet
    assert game.open_source == "suggestion" and game.open_player == "me"
    assert sum(ch is not None for row in game.board for ch in row) > 5
    assert game.base_board[7][5:10] == list("HELLO")
    assert len(game.rack) < 7  # the leave
    page = client.get(f"/game/{gid}").text
    assert "not confirmed" in page and "Confirm your move" in page

    # typing a word or playing another suggestion needs the open turn dealt with first
    assert _word(client, gid, "AX", 1, 1, "across", "me").status_code == 422  # type: ignore[attr-defined]

    client.post(f"/game/{gid}/board", data=_current_form(services, gid, "confirm_me"))
    game = services.game(gid)
    assert len(game.turns) == 2 and game.turns[1].player == "me" and game.turns[1].points > 0
    assert game.phase == "opponent_turn" and game.open_source == ""
    assert game.scores()["me"] == game.turns[1].points
    assert game.base_board == game.board


def _current_form(services: Services, gid: str, action: str) -> dict[str, str]:
    game = services.game(gid)
    data = {f"c-{r}-{c}": game.board[r][c] or "" for r in range(15) for c in range(15)}
    data |= {f"r-{i}": game.rack[i] if i < len(game.rack) else "" for i in range(7)}
    return data | {"action": action}


def test_cancel_returns_board_and_rack(client: TestClient, services: Services) -> None:
    gid = started_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "solve"))
    client.post(f"/game/{gid}/play/0")
    client.post(f"/game/{gid}/board", data={"action": "cancel"})
    game = services.game(gid)
    assert game.board == game.base_board and game.rack == "SATIRE?"
    assert game.open_source == "" and len(game.turns) == 1


def test_open_turn_tiles_can_be_edited_before_confirming(
    client: TestClient, services: Services
) -> None:
    gid = started_game(client)
    assert _word(client, gid, "HAT", 8, 6, "down", "me").status_code == 303  # type: ignore[attr-defined]
    game = services.game(gid)
    assert game.board[8][5] == "A" and game.board[9][5] == "T"
    assert game.rack == "SIRE?"  # A and T used; the rack left over
    # you mistyped: it is HIT. Edit the A, then confirm.
    data = _current_form(services, gid, "confirm_me")
    data["c-8-5"] = "I"
    client.post(f"/game/{gid}/board", data=data)
    turn = services.game(gid).turns[-1]
    assert turn.word == "HIT" and turn.points > 0


# --- the opponent's move: typed ------------------------------------------------------------


def test_typed_opponent_move_is_confirmed_then_it_is_your_turn(
    client: TestClient, services: Services
) -> None:
    gid = started_game(client, phase="start_opponent")
    assert _word(client, gid, "HAT", 8, 6, "down", "opponent").status_code == 303  # type: ignore[attr-defined]
    game = services.game(gid)
    assert game.open_player == "opponent" and game.rack == "SATIRE?"  # your rack is untouched
    assert "Confirm opponent&#39;s move" in client.get(f"/game/{gid}").text
    client.post(f"/game/{gid}/board", data=_current_form(services, gid, "confirm_opponent"))
    game = services.game(gid)
    assert game.turns[-1].player == "opponent" and game.phase == "my_turn"
    assert game.scores()["opponent"] == game.turns[-1].points > 0


def test_typed_word_problems_are_explained(client: TestClient) -> None:
    gid = started_game(client)
    r = _word(client, gid, "XYZ", 1, 14, "across", "opponent")
    assert r.status_code == 422 and "runs off the board" in _text(r)  # type: ignore[attr-defined]
    r = _word(client, gid, "HAT", 0, 1, "across", "opponent")
    assert r.status_code == 422  # type: ignore[attr-defined]
    r = _word(client, gid, "HAT", 8, 6, "down", "opponent")
    assert r.status_code == 303  # type: ignore[attr-defined]
    r = _word(client, gid, "AX", 1, 1, "across", "opponent")  # one open turn at a time
    assert r.status_code == 422 and "Confirm or cancel" in _text(r)  # type: ignore[attr-defined]


def test_word_not_in_the_list_can_be_forced(client: TestClient, services: Services) -> None:
    gid = started_game(client, phase="start_opponent")
    _word(client, gid, "HQZ", 8, 6, "down", "opponent")
    r = client.post(f"/game/{gid}/board",
                    data=_current_form(services, gid, "confirm_opponent"))  # fmt: skip
    assert r.status_code == 422 and "isn't in" in _text(r)
    data = _current_form(services, gid, "confirm_opponent") | {"force": "1"}
    client.post(f"/game/{gid}/board", data=data)
    turn = services.game(gid).turns[-1]
    assert turn.word == "HQZ" and turn.forced


def test_illegal_placement_cannot_be_confirmed_but_can_be_fixed(
    client: TestClient, services: Services
) -> None:
    gid = started_game(client, phase="start_opponent")
    # two tiles far apart: not one line of contiguous tiles
    data = _current_form(services, gid, "confirm_opponent")
    data["c-8-5"] = "A"
    data["c-3-12"] = "T"
    r = client.post(f"/game/{gid}/board", data=data)
    assert r.status_code == 422 and len(services.game(gid).turns) == 1
    assert services.game(gid).board[3][12] == "T"  # the edit was kept


# --- undo, pass, update ---------------------------------------------------------------------


def test_undo_reopens_the_last_turn(client: TestClient, services: Services) -> None:
    gid = started_game(client, phase="start_opponent")
    _word(client, gid, "HAT", 8, 6, "down", "opponent")
    client.post(f"/game/{gid}/board", data=_current_form(services, gid, "confirm_opponent"))
    assert len(services.game(gid).turns) == 2
    client.post(f"/game/{gid}/undo")
    game = services.game(gid)
    assert len(game.turns) == 1 and game.phase == "opponent_turn"
    assert game.board[8][5] == "A"  # the tiles are back as an editable open turn
    assert game.base_board[8][5] is None
    assert game.open_player == "opponent" and game.scores()["opponent"] == 0


def test_undo_the_start_goes_back_to_setup(client: TestClient, services: Services) -> None:
    gid = started_game(client)
    client.post(f"/game/{gid}/undo")
    game = services.game(gid)
    assert game.started is False and game.turns == []
    assert game.base_board[7][5] is None and game.board[7][5] == "H"
    assert "Starting position" in client.get(f"/game/{gid}").text


def test_pass_and_undo_pass(client: TestClient, services: Services) -> None:
    gid = started_game(client)
    client.post(f"/game/{gid}/pass", data={"who": "me"})
    game = services.game(gid)
    assert game.turns[-1].kind == "pass" and game.phase == "opponent_turn"
    client.post(f"/game/{gid}/undo")
    assert services.game(gid).phase == "my_turn"


def test_just_update_the_board_logs_a_board_update(client: TestClient, services: Services) -> None:
    gid = started_game(client)
    data = _current_form(services, gid, "accept")
    data["c-8-5"], data["c-9-5"] = "A", "T"
    client.post(f"/game/{gid}/board", data=data)
    game = services.game(gid)
    update = game.turns[-1]
    assert (update.kind, update.tiles, update.points) == ("start", 2, 0)
    assert game.phase == "my_turn" and game.base_board[9][5] == "T"
    assert "Board updated" in client.get(f"/game/{gid}").text


def test_drew_prompt_after_the_opponents_move(client: TestClient, services: Services) -> None:
    gid = started_game(client, rack="SIRE", phase="start_opponent")
    _word(client, gid, "HAT", 8, 6, "down", "opponent")
    client.post(f"/game/{gid}/board", data=_current_form(services, gid, "confirm_opponent"))
    html = client.get(f"/game/{gid}").text
    assert "What did you draw?" in html
    client.post(f"/game/{gid}/draw", data={"letters": "XYZ", "action": "add"})
    assert services.game(gid).rack == "SIREXYZ"


# --- pictures -------------------------------------------------------------------------------


def test_picture_turn_labels_the_picture_when_confirmed(
    client: TestClient, services: Services, reader: FakeReader, retrains: list[int]
) -> None:
    gid = started_game(client, phase="start_opponent")
    reader.next = reading({7: (5, "HELLO"), 8: (5, "A"), 9: (5, "T")}, "XYZ")
    assert send_picture(client, gid).status_code == 303
    game = services.game(gid)
    assert len(game.pictures) == 1 and game.open_source == "picture"
    assert game.rack == "XYZ" and len(game.turns) == 1
    assert game.pictures[0].label_board is None  # not a label until the turn is confirmed
    assert retrains == []

    # the reader got a letter wrong; the user fixes it on the open turn, then confirms
    data = _current_form(services, gid, "confirm_opponent")
    data["c-9-5"] = "N"
    client.post(f"/game/{gid}/board", data=data)
    game = services.game(gid)
    pic = game.pictures[0]
    assert game.turns[-1].picture == pic.id and game.turns[-1].word == "HAN"
    assert pic.label_board is not None and pic.label_board[9][5] == "N"
    assert pic.read_board[9][5] == "T"  # what the reader had said is kept for comparison
    assert pic.label_rack == "XYZ" and pic.label_turn == 2
    assert retrains == [1]
    assert f"/game/{gid}/picture/{pic.id}" in client.get(f"/game/{gid}").text


def test_only_the_latest_picture_of_a_turn_is_labelled(
    client: TestClient, services: Services, reader: FakeReader, retrains: list[int]
) -> None:
    """An earlier picture of the turn may predate tiles the confirmed board has."""
    gid = started_game(client, phase="start_opponent")
    reader.next = reading({7: (5, "HELLO")}, "XYZ")
    send_picture(client, gid)
    reader.next = reading({7: (5, "HELLO"), 8: (5, "A"), 9: (5, "T")}, "XYZ")
    send_picture(client, gid)
    client.post(f"/game/{gid}/board", data=_current_form(services, gid, "confirm_opponent"))
    first, second = services.game(gid).pictures
    assert first.label_board is None
    assert second.label_board is not None and second.label_board[9][5] == "T"
    assert services.game(gid).turns[-1].picture == second.id


def test_typed_turns_do_not_label_pictures(
    client: TestClient, services: Services, reader: FakeReader, retrains: list[int]
) -> None:
    gid = started_game(client, phase="start_opponent")
    reader.next = reading({7: (5, "HELLO")}, "ABC")  # a picture with nothing new
    send_picture(client, gid)
    _word(client, gid, "HAT", 8, 6, "down", "opponent")
    client.post(f"/game/{gid}/board", data=_current_form(services, gid, "confirm_opponent"))
    assert services.game(gid).pictures[0].label_board is None and retrains == []


def test_starting_from_a_picture_labels_it(
    client: TestClient, services: Services, reader: FakeReader, retrains: list[int]
) -> None:
    gid = new_game(client)
    reader.next = reading({7: (5, "HELLO")}, "ABCDEFG")
    send_picture(client, gid, source="upload")
    assert services.game(gid).pictures[0].source == "upload"
    client.post(f"/game/{gid}/board", data=_current_form(services, gid, "start_me"))
    pic = services.game(gid).pictures[0]
    assert pic.label_board is not None and pic.label_board[7][5:10] == "HELLO"
    assert retrains == [1]


def test_next_picture_keeps_locked_tiles_and_reads_the_rest(
    client: TestClient, services: Services, reader: FakeReader
) -> None:
    gid = started_game(client)
    # HELLO misread as HELLS (confidently) plus a new word down from the H
    reader.next = reading({7: (5, "HELLS"), 8: (5, "A"), 9: (5, "T")}, "XYZ")
    send_picture(client, gid)
    game = services.game(gid)
    assert game.board[7][5:10] == list("HELLO")  # locked, not replaced by the misread S
    assert game.board[8][5] == "A" and game.board[9][5] == "T"
    rec = game.recognition
    assert rec is not None and (rec.kept_from_previous, rec.new_tiles, rec.conflicts) == (5, 2, 1)
    assert reader.calls[-1]["previous"][7][5:10] == list("HELLO")  # type: ignore[index]
    page = _text(client.get(f"/game/{gid}"))
    assert "looked different in the new picture" in page and "undo" in page


def test_retaking_replaces_the_open_tiles(
    client: TestClient, services: Services, reader: FakeReader
) -> None:
    gid = started_game(client)
    reader.next = reading({7: (5, "HELLO"), 8: (5, "A")}, "XYZ")
    send_picture(client, gid)
    reader.next = reading({7: (5, "HELLO"), 8: (7, "L")}, "XYZ")  # the second picture
    send_picture(client, gid)
    game = services.game(gid)
    assert game.board[8][5] is None and game.board[8][7] == "L"
    assert len(game.pictures) == 2


def test_refused_picture_changes_nothing_and_is_not_kept(
    client: TestClient, services: Services, reader: FakeReader
) -> None:
    gid = started_game(client)
    before = services.game(gid)
    board, rack = [row[:] for row in before.board], before.rack
    refused = reading({}, "")
    refused.rejected = refused.merged = True
    refused.warnings = ["picture_does_not_match"]
    refused.match = 0.55
    reader.next = refused
    send_picture(client, gid)
    after = services.game(gid)
    assert after.board == board and after.rack == rack and after.pictures == []
    page = _text(client.get(f"/game/{gid}"))
    assert "doesn't look like this game's board" in page and "55%" in page
    assert "doesn't look like this game's board" not in _text(client.get(f"/game/{gid}"))


def test_unreadable_rack_leaves_yours_alone(
    client: TestClient, services: Services, reader: FakeReader
) -> None:
    gid = started_game(client)
    cut = reading({7: (5, "HELLO"), 8: (5, "A")}, "")
    cut.merged = True
    cut.warnings = ["rack_cut_off"]
    reader.next = cut
    send_picture(client, gid)
    game = services.game(gid)
    assert game.rack == "SATIRE?" and game.board[8][5] == "A"
    assert "rack is partly outside the picture" in _text(client.get(f"/game/{gid}"))


def test_game_page_shows_only_the_last_few_pictures(
    client: TestClient, services: Services, reader: FakeReader
) -> None:
    gid = started_game(client)
    reader.next = reading({7: (5, "HELLO")}, "ABC")
    for _ in range(6):
        send_picture(client, gid)
    html = client.get(f"/game/{gid}").text
    assert html.count('class="pic"') == 4 and "The last 4 of 6 pictures" in html


# --- persistence ----------------------------------------------------------------------------


def test_game_round_trips_through_disk(
    client: TestClient, services: Services, reader: FakeReader, games_dir: Path
) -> None:
    gid = started_game(client, phase="start_opponent")
    reader.next = reading({7: (5, "HELLO"), 8: (5, "A"), 9: (5, "T")}, "XYZ")
    send_picture(client, gid)
    client.post(f"/game/{gid}/board", data=_current_form(services, gid, "confirm_opponent"))
    client.post(f"/game/{gid}/rename", data={"name": "Round 2"})

    again = GameStore(games_dir).get(gid)
    assert again is not None and again.name == "Round 2"
    live = services.game(gid)
    assert again.board == live.board and again.base_board == live.base_board
    assert [t.word for t in again.turns] == [t.word for t in live.turns]
    assert again.pictures[0].label_board == live.pictures[0].label_board
    assert GameStore(games_dir).picture_bytes(again, again.pictures[0])

    client.post(f"/game/{gid}/delete")
    assert not (games_dir / f"{gid}.json").exists() and not (games_dir / gid).exists()


def test_home_lists_games(client: TestClient) -> None:
    gid = started_game(client)
    html = client.get("/").text
    assert f"/game/{gid}" in html and "Your games" in html


def test_old_games_keep_their_single_picture(games_dir: Path, png_bytes: bytes) -> None:
    import json

    games_dir.mkdir()
    rows = ["." * 15] * 15
    (games_dir / "abc.json").write_text(json.dumps({
        "version": 1, "id": "abc", "name": "Old", "phase": "my_turn", "solver_id": "anchor_trie",
        "params": {}, "dictionary_id": "words", "board": rows, "base_board": rows, "rack": "AB",
        "turns": [], "image_file": "abc.png", "image_type": "image/png",
    }))  # fmt: skip
    (games_dir / "abc.png").write_bytes(png_bytes)
    store = GameStore(games_dir)
    game = store.get("abc")
    assert game is not None and len(game.pictures) == 1 and game.started is False
    assert store.picture_bytes(game, game.pictures[0]) == png_bytes
    assert not (games_dir / "abc.png").exists()
    assert GameStore(games_dir).get("abc").pictures[0].file == "p1.png"  # type: ignore[union-attr]


def test_game_pictures_become_training_examples(
    client: TestClient, services: Services, reader: FakeReader, games_dir: Path
) -> None:
    from wwf.vision.train import load_game_examples

    gid = new_game(client)
    reader.next = reading({7: (5, "HELLO")}, "ABCDEFG")
    send_picture(client, gid)
    assert load_game_examples(games_dir) == []  # not confirmed: no label yet
    client.post(f"/game/{gid}/board", data=_current_form(services, gid, "start_me"))
    (example,) = load_game_examples(games_dir)
    assert example.board[7][5:10] == "HELLO" and example.rack == "ABCDEFG"
