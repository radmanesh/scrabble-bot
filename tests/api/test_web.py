"""The server-rendered pages, driven as a browser without JavaScript would."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from tests.api.helpers import FakeReader, form, new_game, reading
from wwf.api.services import Services


def test_home_puts_the_camera_first(client: TestClient) -> None:
    html = client.get("/").text
    camera = html.index('capture="environment"')
    upload = html.index('id="file-form"')
    assert camera < upload, "the camera must come before the file upload"
    assert html.count('name="image"') == 2
    file_form = html[upload : html.index("</form>", upload)]
    assert 'accept="image/*"' in file_form and "capture" not in file_form
    assert 'id="viewfinder"' in html and 'action="/new"' in html
    assert "Fixtures" not in html and "collect" not in html.lower()


def test_game_page_offers_camera_then_upload(client: TestClient) -> None:
    gid = new_game(client)
    html = client.get(f"/game/{gid}").text
    assert html.index('capture="environment"') < html.index("Upload instead")
    assert 'id="camera-button"' in html and 'id="viewfinder"' in html
    assert "Starting position" in html  # nothing is locked until it is confirmed


def test_upload_starts_a_game_and_keeps_the_picture(
    client: TestClient, services: Services, reader: FakeReader, png_bytes: bytes
) -> None:
    reader.next = reading({7: (5, "HELLO")}, "ABCDEFG")
    r = client.post("/upload", files={"image": ("s.png", png_bytes, "image/png")},
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    gid = r.headers["location"].rsplit("/", 1)[1]
    game = services.game(gid)
    assert len(game.pictures) == 1 and game.pictures[0].source == "photo"
    assert game.board[7][5:10] == list("HELLO") and game.rack == "ABCDEFG"
    assert game.started is False and game.open_source == "picture"
    html = client.get(r.headers["location"]).text
    assert f"/game/{gid}/picture/p1" in html
    assert client.get(f"/game/{gid}/picture/p1").content == png_bytes


def test_upload_rejects_non_image(client: TestClient) -> None:
    r = client.post("/upload", files={"image": ("s.png", b"nope", "image/png")})
    assert r.status_code == 400
    assert "an image we can read" in r.text


def test_board_form_renders_every_square(client: TestClient) -> None:
    gid = new_game(client)
    html = client.get(f"/game/{gid}").text
    assert html.count('class="cell ') == 225 + 7
    assert 'placeholder="TW"' in html


def test_solve_shows_ranked_moves(client: TestClient) -> None:
    gid = new_game(client)
    r = client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "start_me"),
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    r = client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "solve"),
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303 and r.headers["location"] == f"/game/{gid}"
    html = client.get(f"/game/{gid}").text
    assert "Smart brute force" in html
    assert html.count("<tr") == 26  # header + 25 moves
    assert re.search(r'class="cell[^"]* new', html)
    preview = client.get(f"/game/{gid}/preview/3").text
    assert "<h2>" in preview and "board preview" in preview


def test_board_form_rejects_bad_cells(client: TestClient) -> None:
    gid = new_game(client)
    data = form({7: (5, "HELLO")}, "AB") | {"c-0-0": "1"}
    r = client.post(f"/game/{gid}/board", data=data)
    assert r.status_code == 422 and "Row 1, column 1" in r.text


def test_solve_needs_a_rack(client: TestClient) -> None:
    gid = new_game(client)
    client.post(f"/game/{gid}/board", data=form({}, "", "start_me"))
    r = client.post(f"/game/{gid}/board", data=form({}, "", "solve"))
    assert r.status_code == 422 and "rack" in r.text


def test_lowercase_cell_is_a_blank(client: TestClient, services: Services) -> None:
    gid = new_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELlO")}, "S"))
    assert services.game(gid).board[7][5:10] == ["H", "E", "L", "l", "O"]


def test_options_page_is_generated_from_schema(client: TestClient) -> None:
    html = client.get("/options").text
    assert 'name="p.anchor_trie.max_results"' in html and 'max="500"' in html
    assert 'name="p.dummy_random.seed"' in html
    assert "planned" in html and 'value="gaddag"' in html
    assert 'name="dictionary_id"' in html


def test_options_apply_to_the_game_and_new_games(client: TestClient, services: Services) -> None:
    gid = new_game(client)
    r = client.post(
        "/options",
        data={
            "solver_id": "anchor_trie",
            "game_id": gid,
            "dictionary_id": "words",
            "p.anchor_trie.max_results": "4",
            "p.anchor_trie.min_score": "0",
            "p.anchor_trie.allow_blank_branching": ["false", "true"],
            "p.anchor_trie.tie_break": "score_only",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    game = services.game(gid)
    assert game.params == {"max_results": 4, "min_score": 0, "allow_blank_branching": True,
                           "tie_break": "score_only"}  # fmt: skip
    assert game.dictionary_id == "words"
    later = new_game(client)  # the cookie carries the choice into games started later
    assert services.game(later).params["max_results"] == 4
    assert 'value="4"' in client.get("/options").text


def test_options_reject_out_of_range(client: TestClient) -> None:
    r = client.post("/options", data={"solver_id": "anchor_trie", "p.anchor_trie.max_results": "0"})
    assert r.status_code == 422 and "max_results" in r.text


def test_missing_game_page(client: TestClient) -> None:
    r = client.get("/game/doesnotexist")
    assert r.status_code == 404 and "Game not found" in r.text


def test_removed_pages_are_gone(client: TestClient) -> None:
    assert client.get("/fixtures").status_code == 404
    assert client.post("/api/v1/session").status_code in (404, 405)
    assert client.get("/api/v1/health").json()["status"] == "ok"


# --- home: game cards ------------------------------------------------------------------------


def test_home_shows_each_game_as_a_card_with_its_saved_board(
    client: TestClient, services: Services
) -> None:
    gid = new_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "start_me"))
    html = client.get("/").text
    assert 'class="game-card' in html and f'href="/game/{gid}"' in html
    assert "Continue game" in html and "Your turn" in html
    assert "You <b>0</b>" in html and "Opponent <b>0</b>" in html
    assert 'class="board-preview"' in html and "5 confirmed tiles" in html
    # one image for assistive technology, not a tab stop or element per square
    assert html.count('role="img"') == 1
    assert html.index("New game") < html.index('class="game-cards"')  # beside the title


def test_card_preview_follows_the_saved_board_and_marks_unconfirmed_tiles(
    client: TestClient, services: Services
) -> None:
    gid = new_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "start_me"))
    # a typed word is an open turn: on the board, not confirmed
    client.post(
        f"/game/{gid}/word",
        data={"word": "HAT", "row": 8, "col": 6, "direction": "down", "player": "opponent"},
    )
    html = client.get("/").text
    assert "not confirmed yet" in html
    assert html.count('class="pv-draft"') > 0 and html.count('class="pv-tile"') == 5


def test_a_suggestion_is_never_drawn_on_the_card(client: TestClient, services: Services) -> None:
    gid = new_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "start_me"))
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "solve"))
    assert services.game(gid).last_result is not None  # there are suggestions
    html = client.get("/").text
    assert html.count('class="pv-tile"') == 5 and 'class="pv-draft"' not in html


def test_card_delete_names_the_game_and_rename_is_a_separate_form(client: TestClient) -> None:
    gid = new_game(client)
    client.post(f"/game/{gid}/rename", data={"name": "Sunday game"})
    html = client.get("/").text
    assert "Delete “Sunday game”" in html and 'data-confirm="Delete “Sunday game”' in html
    assert f'action="/game/{gid}/rename"' in html and f'action="/game/{gid}/delete"' in html


def test_home_without_games_still_starts_with_the_camera(client: TestClient) -> None:
    html = client.get("/").text
    assert "Start a game" in html and 'class="game-cards"' not in html
    assert "Take a photo of the board" in html and "Enter board manually" in html
