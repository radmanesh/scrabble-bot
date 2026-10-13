"""The flows people actually follow, start to finish, through the pages."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.api.helpers import FakeReader, form, new_game, reading, send_picture, started_game
from wwf.api.services import Services


def _record(client: TestClient, services: Services, gid: str, player: str, word: str,
            row: int, col: int, direction: str) -> int:  # fmt: skip
    return client.post(
        f"/game/{gid}/compose/record",
        data={
            "word": word,
            "row": row,
            "col": col,
            "direction": direction,
            "player": player,
            "turn": services.game(gid).turn_number,
        },
        follow_redirects=False,
    ).status_code


def test_resume_find_a_suggestion_preview_it_and_record_it(
    client: TestClient, services: Services
) -> None:
    gid = started_game(client)
    assert f"/game/{gid}" in client.get("/").text  # resume from the games page
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "solve"))
    before = services.game(gid).board
    page = client.get(f"/game/{gid}?do=find&move=1").text  # choose the second move
    assert "Record this move" in page and "proposed" in page
    assert services.game(gid).board == before  # choosing changed nothing
    assert client.post(f"/game/{gid}/record/1", data={"turn": 2}, follow_redirects=False
                       ).status_code == 303  # fmt: skip
    game = services.game(gid)
    assert len(game.turns) == 2 and game.turns[-1].player == "me" and game.phase == "opponent_turn"


def test_resume_and_record_my_own_word_by_typing(client: TestClient, services: Services) -> None:
    gid = started_game(client)
    assert _record(client, services, gid, "me", "HAT", 8, 6, "down") == 303
    game = services.game(gid)
    assert game.turns[-1].word == "HAT" and game.phase == "opponent_turn"
    follow = client.get(f"/game/{gid}").text
    assert "Turns" in follow and "HAT" in follow  # the move is in the log


def test_record_the_opponents_word_then_find_the_next_move(
    client: TestClient, services: Services
) -> None:
    gid = started_game(client, phase="start_opponent")
    assert _record(client, services, gid, "opponent", "HAT", 8, 6, "down") == 303
    game = services.game(gid)
    assert game.phase == "my_turn" and game.turns[-1].player == "opponent"
    page = client.get(f"/game/{gid}").text
    assert "Find my best move" in page  # the action for the turn that is next
    r = client.post(f"/game/{gid}/solve", follow_redirects=False)
    assert r.status_code == 303 and services.game(gid).last_result is not None


def test_a_photo_is_read_corrected_and_confirmed(
    client: TestClient, services: Services, reader: FakeReader
) -> None:
    gid = started_game(client, phase="start_opponent")
    reader.next = reading({7: (5, "HELLO"), 8: (5, "A"), 9: (5, "T")}, "XYZ", unsure={(9, 5)})
    send_picture(client, gid)
    game = services.game(gid)
    assert len(game.turns) == 1 and game.open_source == "picture"  # read, not yet confirmed
    assert "Squares to check" in client.get(f"/game/{gid}").text
    data = {"action": "confirm_opponent", "rack": game.rack, "turn": str(game.turn_number)}
    for r, row in enumerate(game.board):
        for c, ch in enumerate(row):
            if ch:
                data[f"c-{r}-{c}"] = ch
    data["c-9-5"] = "N"  # the user corrects the unsure letter
    client.post(f"/game/{gid}/board", data=data)
    game = services.game(gid)
    assert game.turns[-1].word == "HAN" and game.pictures[0].label_board is not None


def test_pass_undo_and_rename(client: TestClient, services: Services) -> None:
    gid = started_game(client, phase="start_opponent")
    client.post(f"/game/{gid}/rename", data={"name": "Friday"})
    assert services.game(gid).name == "Friday"
    client.post(f"/game/{gid}/pass", data={"who": "opponent"})
    assert services.game(gid).turns[-1].kind == "pass"
    client.post(f"/game/{gid}/undo")
    game = services.game(gid)
    assert len(game.turns) == 1 and game.phase == "opponent_turn"


def test_a_draft_survives_reload_and_back_navigation(
    client: TestClient, services: Services
) -> None:
    gid = started_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO"), 8: (5, "A")}, "SATIRE?"))
    first = client.get(f"/game/{gid}").text
    again = client.get(f"/game/{gid}").text
    assert "not confirmed" in first and "not confirmed" in again
    assert 'id="composer"' not in first  # the draft is the thing to confirm or cancel
    client.post(f"/game/{gid}/board", data={"action": "cancel"})
    assert services.game(gid).board[8][5] is None and "Record my move" in client.get(
        f"/game/{gid}").text  # fmt: skip


def test_conflicts_invalid_positions_blanks_and_stale_suggestions(
    client: TestClient, services: Services
) -> None:
    gid = started_game(client)
    # an occupied square that doesn't match
    assert _record(client, services, gid, "me", "XAT", 8, 6, "down") == 422
    # off the board, and not touching anything
    assert _record(client, services, gid, "me", "HAT", 8, 14, "across") == 422
    assert _record(client, services, gid, "me", "CAT", 1, 1, "across") == 422
    assert len(services.game(gid).turns) == 1
    # a blank tile scores nothing
    plain = client.get(
        f"/game/{gid}/compose/preview.json",
        params={"word": "HAT", "row": 8, "col": 6, "direction": "down"},
    ).json()
    blank = client.get(
        f"/game/{gid}/compose/preview.json",
        params={"word": "HAt", "row": 8, "col": 6, "direction": "down"},
    ).json()
    assert blank["points"] < plain["points"]  # fmt: skip
    # a rack change makes earlier suggestions stale
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "solve"))
    assert services.game(gid).last_result is not None
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIR", "save"))
    assert services.game(gid).last_result is None


def test_repeated_submissions_record_one_turn(client: TestClient, services: Services) -> None:
    gid = started_game(client, phase="start_opponent")
    codes = [_record(client, services, gid, "opponent", "HAT", 8, 6, "down") for _ in range(1)]
    data = {"word": "HAT", "row": 8, "col": 6, "direction": "down", "player": "opponent", "turn": 2}
    for _ in range(3):
        codes.append(client.post(f"/game/{gid}/compose/record", data=data).status_code)
    assert codes[0] == 303 and set(codes[1:]) == {422}
    assert len(services.game(gid).turns) == 2


def test_a_new_game_from_nothing_can_be_set_up_by_hand(
    client: TestClient, services: Services
) -> None:
    gid = new_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (6, "CAT")}, "SATIRE?", "start_me"))
    game = services.game(gid)
    assert game.started and game.turns[0].kind == "start" and game.phase == "my_turn"
