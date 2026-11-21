"""Super Score games: the mode comes from the picture, drives the board, and is saved."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.api.conftest import load
from tests.api.helpers import FakeReader, form, new_game, reading, send_picture
from wwf.api.games import game_from_dict, game_to_dict
from wwf.api.services import Services


def upload_super(client: TestClient, reader: FakeReader) -> str:
    reader.next = reading({7: (6, "COT")}, "CEUEL", mode="super_score")
    r = send_picture_new(client)
    return str(r.headers["location"]).rsplit("/", 1)[1]


def send_picture_new(client: TestClient):  # type: ignore[no-untyped-def]
    from tests.api.helpers import png

    return client.post(
        "/upload", files={"image": ("b.png", png(), "image/png")},
        data={"source": "upload"}, follow_redirects=False,
    )  # fmt: skip


def test_a_picture_of_a_super_score_board_makes_a_super_score_game(
    client: TestClient, services: Services, reader: FakeReader
) -> None:
    gid = upload_super(client, reader)
    assert load(services, gid).mode == "super_score"
    assert reader.calls[-1]["mode"] is None  # a new game lets the picture say which board it is


def test_a_picture_of_a_normal_board_stays_normal(
    client: TestClient, services: Services, reader: FakeReader
) -> None:
    reader.next = reading({7: (6, "COT")}, "CEUEL", mode="normal")
    r = send_picture_new(client)
    assert load(services, str(r.headers["location"]).rsplit("/", 1)[1]).mode == "normal"


def test_the_page_shows_the_quadruple_squares(client: TestClient, reader: FakeReader) -> None:
    gid = upload_super(client, reader)
    html = client.get(f"/game/{gid}").text
    assert "mode-badge" in html
    assert 'placeholder="QW"' in html and 'placeholder="QL"' in html
    assert "mystery square" in html
    normal = client.get(f"/game/{new_game(client)}").text
    assert 'placeholder="QW"' not in normal and "mode-badge" not in normal


def test_the_mode_can_be_changed_until_the_game_starts(
    client: TestClient, services: Services
) -> None:
    gid = new_game(client)
    assert client.post(f"/game/{gid}/mode", data={"mode": "super_score"}).status_code == 200
    assert load(services, gid).mode == "super_score"
    assert client.post(f"/game/{gid}/mode", data={"mode": "bogus"}).status_code == 422
    r = client.post(f"/game/{gid}/board", data=form({7: (6, "COT")}, "CEUELAB", "start_me"),
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    assert client.post(f"/game/{gid}/mode", data={"mode": "normal"}).status_code == 409
    assert load(services, gid).mode == "super_score"


def test_a_started_game_keeps_its_mode_when_reading_the_next_picture(
    client: TestClient, services: Services, reader: FakeReader
) -> None:
    gid = new_game(client)
    client.post(f"/game/{gid}/mode", data={"mode": "super_score"})
    client.post(f"/game/{gid}/board", data=form({7: (6, "COT")}, "CEUELAB", "start_me"))
    reader.next = reading({7: (6, "COT")}, "CEUELAB", mode="normal")
    send_picture(client, gid)
    assert reader.calls[-1]["mode"] == "super_score"
    assert load(services, gid).mode == "super_score"


def test_moves_are_scored_on_the_games_board(client: TestClient, services: Services) -> None:
    def best(mode: str) -> int:
        gid = new_game(client)
        client.post(f"/game/{gid}/mode", data={"mode": mode})
        client.post(f"/game/{gid}/board", data=form({7: (6, "COT")}, "ZAXQUIT", "start_me"))
        services.solve(load(services, gid))
        return load(services, gid).last_result.moves[0].score.total  # type: ignore[union-attr]

    assert best("super_score") != best("normal")


def test_the_mode_is_saved_with_the_game(client: TestClient, services: Services) -> None:
    gid = new_game(client)
    client.post(f"/game/{gid}/mode", data={"mode": "super_score"})
    g = load(services, gid)
    assert game_to_dict(g)["mode"] == "super_score"
    assert game_from_dict(game_to_dict(g)).mode == "super_score"
    old = game_to_dict(g)
    del old["mode"]  # games saved before modes existed
    assert game_from_dict(old).mode == "normal"


def test_home_marks_super_score_games(client: TestClient, reader: FakeReader) -> None:
    upload_super(client, reader)
    assert "Super Score" in client.get("/").text


# --- mystery square rewards ---------------------------------------------------------------------

ABS_DOWN = {5: (5, "B"), 6: (5, "A"), 7: (5, "S")}  # BAS; S joins COT as SCOT; B is on a "?"
COT = {7: (6, "COT")}


def mystery_game(client: TestClient) -> str:
    """A Super Score game started with COT in the middle, your turn."""
    gid = new_game(client)
    client.post(f"/game/{gid}/mode", data={"mode": "super_score"})
    client.post(f"/game/{gid}/board", data=form(COT, "ABSQUIT", "start_me"))
    return gid


def tiles(*rows: dict[int, tuple[int, str]]) -> dict[int, tuple[int, str]]:
    """Merge single-column plays into one board: row -> (first column, letters)."""
    out: dict[int, tuple[int, str]] = {}
    for rows_ in rows:
        for r, (c0, word) in rows_.items():
            c_old, w_old = out.get(r, (c0, ""))
            lo = min(c0, c_old) if w_old else c0
            cells = {c_old + i: ch for i, ch in enumerate(w_old)} | {
                c0 + i: ch for i, ch in enumerate(word)
            }
            lo, hi = min(cells), max(cells)
            out[r] = (lo, "".join(cells.get(c, ".") for c in range(lo, hi + 1)))
    return out


def place_abs(client: TestClient, gid: str, **extra: str) -> str:
    """Put BAS down column 5 (the B on the mystery square (5, 5)) as the open turn."""
    board = form(tiles(COT, ABS_DOWN), "QUIT", "save") | extra
    for key in [k for k, v in board.items() if v == "."]:
        board[key] = ""
    return client.post(f"/game/{gid}/board", data=board).text


def test_a_move_on_a_mystery_square_asks_which_reward(
    client: TestClient, services: Services
) -> None:
    gid = mystery_game(client)
    page = place_abs(client, gid)
    assert "which reward did you get?" in page and 'name="reward-5-5"' in page
    assert "<b>13 points</b>" in page  # BAS 6 + SCOT 7, the mystery square counting as plain


def test_the_chosen_reward_is_scored_and_saved_with_the_turn(
    client: TestClient, services: Services
) -> None:
    gid = mystery_game(client)
    assert "<b>31 points</b>" in place_abs(client, gid, **{"reward-5-5": "Q"})  # BAS 6*4 + 7
    assert load(services, gid).open_rewards == {"5,5": "Q"}
    board = form(tiles(COT, ABS_DOWN), "QUIT", "confirm_me") | {"reward-5-5": "+"}
    for key in [k for k, v in board.items() if v == "."]:
        board[key] = ""
    assert client.post(f"/game/{gid}/board", data=board, follow_redirects=False).status_code == 303
    turn = load(services, gid).turns[-1]
    assert (turn.points, turn.rewards) == (23, {"5,5": "+"})  # 13 plus Points Plus
    assert load(services, gid).open_rewards == {}


def test_a_reward_on_a_square_the_move_misses_is_ignored(
    client: TestClient, services: Services
) -> None:
    gid = mystery_game(client)
    place_abs(client, gid, **{"reward-9-9": "Q", "reward-5-5": "D"})
    assert load(services, gid).open_rewards.get("5,5") == "D"
    from wwf.api import gameplay

    pend = gameplay.pending_move(services, load(services, gid))
    assert pend is not None and pend.mystery == [(5, 5)]
    assert pend.analysis.score is not None and pend.analysis.score.total == 6 * 2 + 7


def test_suggestions_show_a_best_case_only_for_mystery_squares(
    client: TestClient, services: Services
) -> None:
    gid = mystery_game(client)
    g = load(services, gid)
    g.rack = "ABSQUIT"
    moves = services.solve(g, params={"max_results": 100}).moves
    on_mystery = [m for m in moves if m.mystery]
    assert on_mystery, "expected a suggestion that lands on a mystery square"
    for m in moves:
        assert m.best is None or m.mystery
        if m.best is not None:
            assert m.best > m.score.total


def test_undo_brings_the_reward_back(client: TestClient, services: Services) -> None:
    gid = mystery_game(client)
    board = form(tiles(COT, ABS_DOWN), "QUIT", "confirm_me") | {"reward-5-5": "T"}
    for key in [k for k, v in board.items() if v == "."]:
        board[key] = ""
    client.post(f"/game/{gid}/board", data=board)
    client.post(f"/game/{gid}/undo")
    assert load(services, gid).open_rewards == {"5,5": "T"}
