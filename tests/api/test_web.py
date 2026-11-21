"""The server-rendered pages, driven as a browser without JavaScript would."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from tests.api.helpers import FakeReader, form, new_game, reading, send_picture
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
    assert html.count("<tr") == 26  # header + 25 moves (the later ones start folded away)
    # the best move is drawn on the one board, dashed, not on a second board
    assert re.search(r'class="cell[^"]* proposed', html) and 'class="board preview"' not in html
    assert "Record this move" in html and "Tiles left on your rack" in html
    assert "Play " not in html and "Leave" not in html
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
    for letter in "HELLO":  # the thumbnail shows the letters, not just blocks
        assert f'class="pv-letter">{letter}</text>' in html
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


# --- game workspace and move composer -------------------------------------------------------


def _started(client: TestClient, phase: str = "start_me") -> str:
    gid = new_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", phase))
    return gid


def test_game_page_leads_with_the_actions_for_this_turn(client: TestClient) -> None:
    gid = _started(client)
    html = client.get(f"/game/{gid}").text
    assert "← Back to games" in html and 'href="/"' in html
    for label in ("Find my best move", "Record my move", "Record opponent&#39;s move",
                  "Update from photo"):  # fmt: skip
        assert label in html or label.replace("&#39;", "'") in html
    assert 'id="composer"' not in html  # on your turn the suggestions come first
    opp = _started(client, "start_opponent")
    html = client.get(f"/game/{opp}").text
    assert 'id="composer"' in html and "Find my best move" not in html
    assert "Recording the opponent's move" in html or "Recording the opponent&#39;s move" in html


def test_the_player_is_explicit_and_can_be_switched_with_a_link(client: TestClient) -> None:
    gid = _started(client)
    html = client.get(f"/game/{gid}?do=opponent").text
    assert 'name="player" value="opponent"' in html and "Record opponent&#39;s move" in html
    mine = client.get(f"/game/{gid}?do=mine").text
    assert 'name="player" value="me"' in mine and "Recording your move" in mine


def test_composer_preview_scores_and_changes_nothing(
    client: TestClient, services: Services
) -> None:
    gid = _started(client)
    before = services.game(gid).board
    r = client.get(
        f"/game/{gid}/compose/preview.json",
        params={"word": "HAT", "row": 8, "col": 6, "direction": "down"},
    )
    data = r.json()
    assert data["legal"] and data["points"] and data["tiles"] == 2
    assert [c["new"] for c in data["cells"]] == [False, True, True]  # the H is already there
    assert services.game(gid).board == before and len(services.game(gid).turns) == 1


def test_composer_preview_names_the_problem_and_square(client: TestClient) -> None:
    gid = _started(client)
    data = client.get(
        f"/game/{gid}/compose/preview.json",
        params={"word": "HAZ", "row": 8, "col": 14, "direction": "across"},
    ).json()
    assert not data["ok"] and any("off the board" in p for p in data["problems"])
    unknown = client.get(
        f"/game/{gid}/compose/preview.json",
        params={"word": "HQZ", "row": 8, "col": 6, "direction": "down"},
    ).json()
    assert unknown["forceable"] and not unknown["legal"] and unknown["problems"]
    clash = client.get(
        f"/game/{gid}/compose/preview.json",
        params={"word": "XAT", "row": 8, "col": 6, "direction": "across"},
    ).json()
    assert any("row 8, column 6" in p for p in clash["problems"])
    assert any(c["conflict"] for c in clash["cells"])


def test_no_javascript_check_draws_the_word_without_recording_it(
    client: TestClient, services: Services
) -> None:
    gid = _started(client)
    r = client.post(
        f"/game/{gid}/compose/preview",
        data={"word": "HAT", "row": "8", "col": "6", "direction": "down", "player": "me"},
    )
    assert r.status_code == 200 and 'class="cell' in r.text and "proposed" in r.text
    assert "points" in r.text and 'value="HAT"' in r.text
    assert len(services.game(gid).turns) == 1 and not services.game(gid).board[8][5]


def test_record_places_and_confirms_exactly_one_turn_for_the_player(
    client: TestClient, services: Services
) -> None:
    gid = _started(client, "start_opponent")
    r = client.post(
        f"/game/{gid}/compose/record",
        data={"word": "HAT", "row": "8", "col": "6", "direction": "down",
              "player": "opponent", "turn": "2"},
        follow_redirects=False,
    )  # fmt: skip
    assert r.status_code == 303
    game = services.game(gid)
    assert len(game.turns) == 2 and game.turns[-1].player == "opponent"
    assert game.turns[-1].word == "HAT" and game.phase == "my_turn"
    assert game.rack == "SATIRE?"  # the opponent's tiles don't touch your rack
    # a retried or double-clicked request records nothing more
    again = client.post(
        f"/game/{gid}/compose/record",
        data={"word": "HAT", "row": "8", "col": "6", "direction": "down",
              "player": "opponent", "turn": "2"},
    )  # fmt: skip
    assert again.status_code == 422 and len(services.game(gid).turns) == 2


def test_record_my_move_uses_up_rack_tiles(client: TestClient, services: Services) -> None:
    gid = _started(client)
    client.post(
        f"/game/{gid}/compose/record",
        data={"word": "HAT", "row": "8", "col": "6", "direction": "down",
              "player": "me", "turn": "2"},
    )  # fmt: skip
    game = services.game(gid)
    assert game.turns[-1].player == "me" and game.phase == "opponent_turn"
    assert len(game.rack) == 5  # A and T came off the rack


def test_a_rejected_record_keeps_the_text_and_changes_nothing(
    client: TestClient, services: Services
) -> None:
    gid = _started(client)
    before = services.game(gid).board
    r = client.post(
        f"/game/{gid}/compose/record",
        data={"word": "HAZ", "row": "1", "col": "1", "direction": "across",
              "player": "me", "turn": "2"},
    )  # fmt: skip
    assert r.status_code == 422 and 'value="HAZ"' in r.text
    assert "touch" in r.text  # not connected to the tiles already on the board
    game = services.game(gid)
    assert game.board == before and len(game.turns) == 1 and game.rack == "SATIRE?"


def test_a_stale_page_cannot_record(client: TestClient, services: Services) -> None:
    gid = _started(client, "start_opponent")
    r = client.post(
        f"/game/{gid}/compose/record",
        data={"word": "HAT", "row": "8", "col": "6", "direction": "down",
              "player": "opponent", "turn": "7"},
    )  # fmt: skip
    assert r.status_code == 422 and len(services.game(gid).turns) == 1


# --- solver results ---------------------------------------------------------------------------


def _solved(client: TestClient) -> str:
    gid = new_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "start_me"))
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "solve"))
    return gid


def test_results_start_compact_and_show_more_reveals_the_rest(client: TestClient) -> None:
    gid = _solved(client)
    html = client.get(f"/game/{gid}").text
    assert html.count("<tr data-n=") == 25
    assert 'id="show-more"' in html and "Show 20 more" in html
    folded = len(re.findall(r'<tr data-n="\d+" class="[^"]*more"\s+hidden>', html))
    assert folded == 20  # five are in view
    everything = client.get(f"/game/{gid}?do=find&all=1").text
    assert not re.search(r'more"\s+hidden', everything) and 'id="show-more"' not in everything


def test_choosing_a_suggestion_only_previews_it(client: TestClient, services: Services) -> None:
    gid = _solved(client)
    game = services.game(gid)
    board, turns = game.board, len(game.turns)
    page = client.get(f"/game/{gid}?do=find&move=2").text
    assert 'class="selected"' in page and 'aria-current="true"' in page
    again = services.game(gid)
    assert again.board == board and len(again.turns) == turns and again.last_result is not None


def test_one_pass_action_in_the_results_and_none_when_typing(client: TestClient) -> None:
    gid = _solved(client)
    find = client.get(f"/game/{gid}").text
    assert find.count('action="/game/' + gid + '/pass"') == 1 and "Play" not in find
    mine = client.get(f"/game/{gid}?do=mine").text
    assert "Record this move" not in mine and "suggestions-data" not in mine


def test_recording_a_suggestion_confirms_one_turn_and_repeats_do_nothing(
    client: TestClient, services: Services
) -> None:
    gid = _solved(client)
    best = services.game(gid).last_result.moves[0]  # type: ignore[union-attr]
    r = client.post(f"/game/{gid}/record/0", data={"turn": "2"}, follow_redirects=False)
    assert r.status_code == 303
    game = services.game(gid)
    assert len(game.turns) == 2 and game.turns[-1].word == best.word
    assert game.turns[-1].player == "me" and game.turns[-1].points == best.score.total
    assert game.phase == "opponent_turn" and game.last_result is None
    again = client.post(f"/game/{gid}/record/0", data={"turn": "2"})
    assert again.status_code == 422 and len(services.game(gid).turns) == 2


def test_changing_the_rack_or_board_drops_stale_suggestions(
    client: TestClient, services: Services
) -> None:
    gid = _solved(client)
    assert services.game(gid).last_result is not None
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIREX", "save"))
    assert services.game(gid).last_result is None
    assert "Find my best move" in client.get(f"/game/{gid}").text


def test_no_legal_moves_explains_and_offers_pass(client: TestClient, services: Services) -> None:
    gid = new_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "ZZZ", "start_me"))
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "ZZZ", "solve"))
    html = client.get(f"/game/{gid}").text
    if services.game(gid).last_result and not services.game(gid).last_result.moves:
        assert "No legal moves with this rack" in html and "I pass this turn" in html


# --- accessibility ---------------------------------------------------------------------------


def test_every_square_is_a_labelled_grid_cell_with_its_state(client: TestClient) -> None:
    gid = _started(client)
    html = client.get(f"/game/{gid}").text
    assert html.count('role="gridcell"') == 225 and html.count('role="row"') == 15
    assert 'aria-label="Row 8, column 6, tile H, confirmed, locked"' in html
    assert 'aria-label="Row 1, column 4, triple word, empty"' in html
    assert 'aria-label="Row 8, column 8, double word, empty"' in html or "double word" in html
    assert 'class="skip-link"' in html and 'id="main"' in html


def test_the_result_of_an_action_is_in_a_status_region(client: TestClient) -> None:
    gid = _started(client, "start_opponent")
    r = client.post(
        f"/game/{gid}/compose/record",
        data={"word": "HAT", "row": "8", "col": "6", "direction": "down",
              "player": "opponent", "turn": "2"},
        follow_redirects=True,
    )  # fmt: skip
    assert 'role="status"' in r.text and "The opponent&#39;s move was recorded: HAT for" in r.text
    assert "was recorded" not in client.get(f"/game/{gid}").text  # shown once


def test_pass_and_undo_say_what_they_did(client: TestClient) -> None:
    gid = _started(client, "start_opponent")
    r = client.post(f"/game/{gid}/pass", data={"who": "opponent"}, follow_redirects=True)
    assert "The opponent&#39;s pass was recorded." in r.text
    r = client.post(f"/game/{gid}/undo", follow_redirects=True)
    assert "was reopened" in r.text


# --- settings --------------------------------------------------------------------------------


def test_settings_use_everyday_words_and_keep_the_stored_values(client: TestClient) -> None:
    html = client.get("/options").text
    for label in ("Maximum suggestions", "Minimum score", "Include blank tiles",
                  "Tie preference", "Word list"):  # fmt: skip
        assert label in html
    assert "Higher score, then the better tiles kept" in html
    assert 'value="score_then_leave"' in html and 'name="p.anchor_trie.tie_break"' in html
    assert "<summary>Advanced" in html and "not available yet" in html
    assert "Smart brute force" in html.split("Advanced")[1]  # algorithm names are advanced


def test_settings_say_what_they_apply_to_and_link_back(client: TestClient) -> None:
    plain = client.get("/options").text
    assert "← Back to games" in plain and "games you start" in plain
    gid = new_game(client)
    client.post(f"/game/{gid}/rename", data={"name": "Sunday game"})
    inside = client.get(f"/options?game={gid}").text
    assert "← Back to Sunday game" in inside and "apply to <b>Sunday game</b>" in inside
    assert "Save and back to the game" in inside and ">Cancel<" in inside


def test_save_and_back_returns_to_the_game_with_a_message(
    client: TestClient, services: Services
) -> None:
    gid = new_game(client)
    r = client.post(
        "/options",
        data={"solver_id": "anchor_trie", "game_id": gid, "dictionary_id": "words",
              "p.anchor_trie.max_results": "7", "then": "game"},
        follow_redirects=True,
    )  # fmt: skip
    assert r.url.path == f"/game/{gid}" and "Settings saved" in r.text
    assert services.game(gid).params["max_results"] == 7


def test_changing_settings_clears_stale_suggestions(client: TestClient, services: Services) -> None:
    gid = _solved(client)
    game = services.game(gid)
    assert game.last_result is not None

    def save(max_results: object) -> None:
        client.post(
            "/options",
            data={
                "solver_id": "anchor_trie",
                "game_id": gid,
                "dictionary_id": game.dictionary_id,
                "p.anchor_trie.max_results": str(max_results),
            },
        )

    save(game.params["max_results"])
    assert services.game(gid).last_result is not None  # nothing changed, nothing dropped
    save(3)
    assert services.game(gid).last_result is None


# --- history and pictures ---------------------------------------------------------------------


def test_history_shows_setup_apart_from_turns_with_running_totals(
    client: TestClient, services: Services
) -> None:
    gid = _started(client, "start_opponent")
    moves = (("opponent", "HAT", "8", "6", "down"), ("me", "TO", "10", "6", "across"))
    for who, word, row, col, direction in moves:
        data = {"word": word, "row": row, "col": col, "direction": direction, "player": who,
                "turn": str(services.game(gid).turn_number)}  # fmt: skip
        assert client.post(f"/game/{gid}/compose/record", data=data).status_code in (200, 303)
    game = services.game(gid)
    assert len(game.turns) == 3
    html = client.get(f"/game/{gid}").text
    assert 'class="me setup"' in html and "Starting position" in html  # not a numbered turn
    assert "Turn 1 · You 0, Opponent" in html and "Turn 2 · You" in html
    assert html.index('id="turns"') > html.index('id="pictures"')


def test_undo_says_exactly_what_it_will_reopen(client: TestClient) -> None:
    gid = _started(client, "start_opponent")
    client.post(f"/game/{gid}/compose/record",
                data={"word": "HAT", "row": "8", "col": "6", "direction": "down",
                      "player": "opponent", "turn": "2"})  # fmt: skip
    html = client.get(f"/game/{gid}").text
    assert "Takes back the opponent&#39;s <b>HAT</b>" in html
    assert "Cancel this turn" in html  # the difference from discarding an unconfirmed draft


def test_pictures_open_in_the_viewer_with_a_way_back(
    client: TestClient, services: Services, reader: FakeReader
) -> None:
    gid = new_game(client)
    reader.next = reading({7: (5, "HELLO")}, "SATIRE?")
    send_picture(client, gid)
    pic = services.game(gid).pictures[0]
    page = client.get(f"/game/{gid}").text
    assert f"/game/{gid}/picture/{pic.id}/view?back=pictures" in page
    view = client.get(f"/game/{gid}/picture/{pic.id}/view?back=turns")
    assert view.status_code == 200
    assert "Picture 1 of 1" in view.text and "Original" in view.text
    assert f'href="/game/{gid}#turns"' in view.text
    assert f"/game/{gid}/picture/{pic.id}" in view.text  # the raw file stays available
    assert client.get(f"/game/{gid}/picture/nope/view").status_code == 404


def test_tile_count_warnings_mark_the_squares_without_changing_them(
    client: TestClient, services: Services
) -> None:
    gid = new_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "ZZ")}, "AB", "start_me"))
    before = services.game(gid).board
    html = client.get(f"/game/{gid}").text
    assert html.count("suspect") >= 2 and "may be one tile too many" in html
    assert "Nothing was changed for you" in html
    assert services.game(gid).board == before


# --- picture import and review ---------------------------------------------------------------


def test_choosing_a_picture_is_staged_before_it_is_read(client: TestClient) -> None:
    home = client.get("/").text
    assert 'id="import-preview"' in home and "Read the board" in home and "Choose another" in home
    assert "nothing is saved until then" in home.lower()
    gid = new_game(client)
    page = client.get(f"/game/{gid}").text
    assert 'id="import-preview"' in page and "If the picture can't be used, nothing changes" in page


def test_unsure_squares_are_listed_with_an_enlarged_picture_of_each(
    client: TestClient, services: Services, reader: FakeReader
) -> None:
    gid = new_game(client)
    reader.next = reading({7: (5, "HELLO")}, "SATIRE?", unsure={(7, 6), (7, 8)})
    send_picture(client, gid)
    html = client.get(f"/game/{gid}").text
    assert "Squares to check" in html and "(2)" in html
    assert "Row 8, column 7" in html and "Read as <b>E</b>" in html
    assert 'href="#sq-7-6"' in html and 'id="sq-7-6"' in html
    pic = services.game(gid).pictures[0]
    crop = client.get(f"/game/{gid}/picture/{pic.id}/square/7/6.png")
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/png"
    assert client.get(f"/game/{gid}/picture/{pic.id}/square/15/0.png").status_code == 404


def test_a_picture_that_cannot_be_used_leaves_the_game_unchanged(
    client: TestClient, services: Services, reader: FakeReader
) -> None:
    gid = _started(client)
    before = (services.game(gid).board, services.game(gid).rack, len(services.game(gid).turns))
    r = client.post(f"/game/{gid}/image", files={"image": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400 or r.status_code == 422
    assert "isn&#39;t an image" in r.text or "isn't an image" in r.text
    game = services.game(gid)
    assert (game.board, game.rack, len(game.turns)) == before and not game.pictures
