from __future__ import annotations

from fastapi.testclient import TestClient

from tests.api.conftest import FakeClock, board_with, empty_board
from wwf.core.board import Board
from wwf.core.config import load_board_config
from wwf.core.dictionary import default_dictionary_registry
from wwf.core.rack import Rack
from wwf.solvers.anchor_trie import AnchorTrieSolver

API = "/api/v1"


def _session_with_board(client: TestClient, png_bytes: bytes) -> str:
    r = client.post(f"{API}/recognize", files={"image": ("shot.png", png_bytes, "image/png")})
    assert r.status_code == 200
    sid: str = r.json()["session_id"]
    r = client.put(
        f"{API}/session/{sid}/board", json={"board": board_with(7, 5, "HELLO"), "rack": "SATIRE?"}
    )
    assert r.status_code == 200
    return sid


def test_health(client: TestClient) -> None:
    r = client.get(f"{API}/health")
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "ok"
    assert body["dictionary_words"] > 170_000
    assert body["model_loaded"] is False
    assert body["request_id"] and r.headers["X-Request-ID"] == body["request_id"]


def test_upload_creates_session_with_image(client: TestClient, png_bytes: bytes) -> None:
    r = client.post(f"{API}/recognize", files={"image": ("shot.png", png_bytes, "image/png")})
    body = r.json()
    assert r.status_code == 200
    assert body["warnings"] == ["recognition_unavailable"]
    assert len(body["board"]) == 15 and all(c is None for row in body["board"] for c in row)
    sid = body["session_id"]

    state = client.get(f"{API}/session/{sid}").json()
    assert state["has_image"] is True
    assert state["solver_id"] == "anchor_trie"
    assert state["dictionary_id"] == "words"  # config/dictionaries.json default

    img = client.get(f"{API}/session/{sid}/image")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    assert img.content == png_bytes


def test_bad_uploads(client: TestClient, png_bytes: bytes) -> None:
    r = client.post(f"{API}/recognize", files={"image": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
    assert r.json()["error"] == "undecodable_image"

    big = png_bytes + b"\0" * (10 * 1024 * 1024)
    r = client.post(f"{API}/recognize", files={"image": ("big.png", big, "image/png")})
    assert r.status_code == 413
    assert r.json()["error"] == "image_too_large"


def test_correct_board_and_solve_with_each_solver(client: TestClient, png_bytes: bytes) -> None:
    sid = _session_with_board(client, png_bytes)
    state = client.get(f"{API}/session/{sid}").json()
    assert state["corrected"] is True
    assert state["rack"] == "SATIRE?"
    assert state["board"][7][5:10] == list("HELLO")

    r = client.post(f"{API}/session/{sid}/solve")
    body = r.json()
    assert r.status_code == 200
    assert body["solver_id"] == "anchor_trie"
    assert len(body["moves"]) == 25
    top = body["moves"][0]
    # same answer as calling the solver directly with the app's default layout and dictionary
    cfg = load_board_config()
    board = Board.from_rows(cfg, board_with(7, 5, "HELLO"))
    direct = AnchorTrieSolver().solve(
        board, Rack.from_string("SATIRE?"), default_dictionary_registry().get("words"), cfg
    )
    assert top["score"]["total"] == direct.moves[0].score.total
    assert top["score"]["bingo_bonus"] == 35
    assert len(top["board_after"]) == 15
    for p in top["placements"]:
        cell = top["board_after"][p["row"]][p["col"]]
        assert cell == (p["letter"].lower() if p["is_blank"] else p["letter"])

    r = client.post(
        f"{API}/session/{sid}/solve",
        json={"solver_id": "dummy_random", "params": {"seed": 1, "max_results": 3}},
    )
    body = r.json()
    assert r.status_code == 200
    assert body["solver_id"] == "dummy_random"
    assert 0 < len(body["moves"]) <= 3
    assert body["stats"]["attempts"] == 200


def test_solve_params_override_session_options(client: TestClient, png_bytes: bytes) -> None:
    sid = _session_with_board(client, png_bytes)
    body = client.post(f"{API}/session/{sid}/solve", json={"params": {"max_results": 2}}).json()
    assert len(body["moves"]) == 2
    assert body["params"]["tie_break"] == "score_then_leave"


def test_options(client: TestClient, png_bytes: bytes) -> None:
    sid = _session_with_board(client, png_bytes)
    r = client.put(
        f"{API}/session/{sid}/options",
        json={"solver_id": "anchor_trie", "params": {"max_results": 5}, "dictionary_id": "words"},
    )
    assert r.status_code == 200
    assert r.json()["params"]["max_results"] == 5
    body = client.post(f"{API}/session/{sid}/solve").json()
    assert len(body["moves"]) == 5
    assert body["dictionary_id"] == "words"


def test_invalid_options_are_422(client: TestClient, png_bytes: bytes) -> None:
    sid = _session_with_board(client, png_bytes)
    cases = [
        ({"solver_id": "anchor_trie", "params": {"max_results": 0}}, "invalid_params"),
        ({"solver_id": "anchor_trie", "params": {"bogus": 1}}, "invalid_params"),
        ({"solver_id": "gaddag", "params": {}}, "unknown_solver"),
        ({"solver_id": "anchor_trie", "dictionary_id": "nope"}, "unknown_dictionary"),
    ]
    for body, error in cases:
        r = client.put(f"{API}/session/{sid}/options", json=body)
        assert r.status_code == 422, body
        assert r.json()["error"] == error


def test_invalid_board_is_422(client: TestClient) -> None:
    sid = client.post(f"{API}/session").json()["session_id"]
    bad = empty_board()
    bad[0][0] = "AB"
    for body in ({"board": bad, "rack": ""}, {"board": [[None]], "rack": ""},
                 {"board": empty_board(), "rack": "ABCDEFGH"}):  # fmt: skip
        r = client.put(f"{API}/session/{sid}/board", json=body)
        assert r.status_code == 422
        assert r.json()["error"] == "validation_error"


def test_expired_session_is_404(client: TestClient, clock: FakeClock) -> None:
    sid = client.post(f"{API}/session").json()["session_id"]
    assert client.get(f"{API}/session/{sid}").status_code == 200
    clock.now += 29 * 60
    assert client.get(f"{API}/session/{sid}").status_code == 200  # access refreshes the TTL
    clock.now += 31 * 60
    r = client.get(f"{API}/session/{sid}")
    assert r.status_code == 404
    assert r.json()["error"] == "session_not_found"
    assert client.get(f"{API}/session/unknown").status_code == 404


def test_catalogues(client: TestClient) -> None:
    solvers = client.get(f"{API}/solvers").json()
    by_id = {s["id"]: s for s in solvers["solvers"]}
    assert solvers["default_id"] == "anchor_trie"
    assert by_id["gaddag"]["planned"] and not by_id["gaddag"]["enabled"]
    params = {p["name"]: p for p in by_id["anchor_trie"]["params"]}
    assert params["max_results"]["max"] == 500
    assert params["tie_break"]["choices"] == ["score_then_leave", "score_only"]

    dictionaries = client.get(f"{API}/dictionaries").json()
    assert {d["id"] for d in dictionaries["dictionaries"]} >= {"enable1", "words"}


def test_validate(client: TestClient) -> None:
    board = board_with(7, 5, "HELLO")
    legal = {
        "placements": [{"row": 8, "col": 8, "letter": "A"}, {"row": 8, "col": 9, "letter": "X"}]
    }
    r = client.post(f"{API}/validate", json={"board": board, "rack": "AX", "move": legal})
    assert r.status_code == 200
    assert r.json()["violations"] == []
    assert r.json()["score"]["total"] == 21

    far = {"placements": [{"row": 0, "col": 0, "letter": "q"}, {"row": 0, "col": 1, "letter": "I"}]}
    r = client.post(
        f"{API}/validate",
        json={"board": board, "rack": "?I", "move": far, "dictionary_id": "words"},
    )
    assert r.json()["violations"] == ["no_contact_with_existing_tiles"]
    assert r.json()["score"] is None


def test_openapi_documents_every_route(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    session = "/session/{session_id}"
    for path in (
        "/recognize", "/session", session, f"{session}/board", f"{session}/solve",
        f"{session}/options", f"{session}/image", "/solvers", "/dictionaries", "/validate",
        "/health",
    ):  # fmt: skip
        assert f"{API}{path}" in paths, path
    assert not any(p.startswith("/review") for p in paths)  # pages are not part of the API


def test_solve_timeout_is_504() -> None:
    from wwf.api.app import create_app
    from wwf.api.services import Settings

    with TestClient(create_app(Settings(solve_timeout_s=0.001))) as client:
        sid = client.post(f"{API}/session").json()["session_id"]
        client.put(
            f"{API}/session/{sid}/board",
            json={"board": board_with(7, 5, "HELLO"), "rack": "SATIRE?"},
        )
        r = client.post(f"{API}/session/{sid}/solve")
        assert r.status_code == 504
        assert r.json()["error"] == "solve_timeout"
        assert r.json()["detail"]["partial"] is True
