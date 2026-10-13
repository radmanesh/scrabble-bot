"""The server-rendered pages, driven as a browser without JavaScript would."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient


def _new_review(client: TestClient) -> str:
    r = client.post("/new", follow_redirects=False)
    assert r.status_code == 303
    location: str = r.headers["location"]
    assert location.startswith("/review/")
    return location.rsplit("/", 1)[1]


def _board_form(rows: dict[int, tuple[int, str]], rack: str) -> dict[str, str]:
    form = {f"c-{r}-{c}": "" for r in range(15) for c in range(15)}
    for r, (c0, word) in rows.items():
        for i, ch in enumerate(word):
            form[f"c-{r}-{c0 + i}"] = ch
    for i in range(7):
        form[f"r-{i}"] = rack[i] if i < len(rack) else ""
    return form


def test_upload_page(client: TestClient) -> None:
    html = client.get("/").text
    assert 'accept="image/*"' in html and 'capture="environment"' in html
    assert "/static/htmx.min.js" in html


def test_upload_redirects_to_review(client: TestClient, png_bytes: bytes) -> None:
    r = client.post("/upload", files={"image": ("s.png", png_bytes, "image/png")},
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    html = client.get(r.headers["location"]).text
    assert "/api/v1/session/" in html and "/image" in html
    assert "recognition isn't built yet" in html


def test_upload_rejects_non_image(client: TestClient) -> None:
    r = client.post("/upload", files={"image": ("s.png", b"nope", "image/png")})
    assert r.status_code == 400
    assert "an image we can read" in r.text


def test_three_steps_to_moves(client: TestClient) -> None:
    sid = _new_review(client)
    html = client.get(f"/review/{sid}").text
    assert html.count('class="cell ') == 225 + 7
    assert 'placeholder="TW"' in html

    form = _board_form({7: (5, "HELLO")}, "SATIRE?") | {"action": "solve"}
    r = client.post(f"/review/{sid}", data=form, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/results/{sid}"

    html = client.get(f"/results/{sid}").text
    assert "Smart brute force" in html
    assert html.count("<tr") == 26  # header + 25 moves
    assert "AERIEST" in html
    assert re.search(r'class="cell[^"]* new', html)  # preview highlights the new tiles

    preview = client.get(f"/results/{sid}/preview/3").text
    assert "<h2>" in preview and "board preview" in preview


def test_review_rejects_bad_cells(client: TestClient) -> None:
    sid = _new_review(client)
    form = _board_form({7: (5, "HELLO")}, "AB") | {"c-0-0": "1", "action": "save"}
    r = client.post(f"/review/{sid}", data=form)
    assert r.status_code == 422
    assert "Row 1, column 1" in r.text


def test_solve_needs_a_rack(client: TestClient) -> None:
    sid = _new_review(client)
    r = client.post(f"/review/{sid}", data=_board_form({}, "") | {"action": "solve"})
    assert r.status_code == 422
    assert "rack" in r.text


def test_lowercase_cell_is_a_blank(client: TestClient) -> None:
    sid = _new_review(client)
    form = _board_form({7: (5, "HELlO")}, "S") | {"action": "save"}
    client.post(f"/review/{sid}", data=form)
    state = client.get(f"/api/v1/session/{sid}").json()
    assert state["board"][7][5:10] == ["H", "E", "L", "l", "O"]


def test_options_page_is_generated_from_schema(client: TestClient) -> None:
    html = client.get("/options").text
    assert 'name="p.anchor_trie.max_results"' in html and 'max="500"' in html
    assert 'name="p.dummy_random.seed"' in html
    assert "planned" in html and 'value="gaddag"' in html
    assert 'name="dictionary_id"' in html


def test_options_save_applies_to_session_and_new_sessions(client: TestClient) -> None:
    sid = _new_review(client)
    r = client.post(
        "/options",
        data={
            "solver_id": "anchor_trie",
            "session_id": sid,
            "dictionary_id": "words",
            "p.anchor_trie.max_results": "4",
            "p.anchor_trie.min_score": "0",
            "p.anchor_trie.allow_blank_branching": ["false", "true"],
            "p.anchor_trie.tie_break": "score_only",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    state = client.get(f"/api/v1/session/{sid}").json()
    assert state["params"] == {"max_results": 4, "min_score": 0, "allow_blank_branching": True,
                               "tie_break": "score_only"}  # fmt: skip
    assert state["dictionary_id"] == "words"

    # the cookie carries the choice into sessions started later
    later = _new_review(client)
    assert client.get(f"/api/v1/session/{later}").json()["params"]["max_results"] == 4
    assert 'value="4"' in client.get("/options").text


def test_options_save_rejects_out_of_range(client: TestClient) -> None:
    r = client.post("/options", data={"solver_id": "anchor_trie", "p.anchor_trie.max_results": "0"})
    assert r.status_code == 422
    assert "max_results" in r.text


def test_expired_session_page(client: TestClient) -> None:
    r = client.get("/review/doesnotexist")
    assert r.status_code == 404
    assert "Session expired" in r.text
