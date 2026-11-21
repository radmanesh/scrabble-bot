"""What the interface work must not change: saved games, read-only pages, solver results.

These are characterization tests. They describe behaviour as it is, so a change to how pages are
laid out cannot quietly change what is stored or what the solver says.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.api.helpers import FakeReader, form, new_game, reading, send_picture, started_game
from wwf.api.games import GameStore, game_from_dict, game_to_dict
from wwf.api.services import Services


def _fingerprint(games_dir: Path) -> dict[str, str]:
    """Every file under the games folder with its content hash."""
    return {
        str(p.relative_to(games_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(games_dir.rglob("*"))
        if p.is_file()
    }


def _form_for(services: Services, gid: str, action: str) -> dict[str, str]:
    g = services.game(gid)
    data = {"action": action, "rack": g.rack, "turn": str(g.turn_number)}
    for r, row in enumerate(g.board):
        for c, ch in enumerate(row):
            if ch:
                data[f"c-{r}-{c}"] = ch
    return data


def _representative_games(
    client: TestClient, services: Services, reader: FakeReader
) -> dict[str, str]:
    """A game in each state the pages have to cope with."""
    games: dict[str, str] = {}
    games["empty"] = new_game(client)
    games["own_turn"] = started_game(client)
    games["opponent_turn"] = started_game(client, phase="start_opponent")

    reader.next = reading({7: (5, "HELLO"), 8: (5, "A")}, "XYZ")
    games["from_photo"] = new_game(client)
    send_picture(client, games["from_photo"])

    games["open_draft"] = started_game(client)
    client.post(
        f"/game/{games['open_draft']}/board",
        data=form({7: (5, "HELLO"), 8: (5, "A"), 9: (5, "T")}, "SATIRE?"),
    )

    games["blank"] = started_game(client)
    client.post(
        f"/game/{games['blank']}/board",
        data=form({7: (5, "HELLO"), 8: (5, "a")}, "SATIRE?"),
    )
    return games


def test_every_kind_of_game_round_trips_through_the_file_format(
    client: TestClient, services: Services, reader: FakeReader, games_dir: Path
) -> None:
    games = _representative_games(client, services, reader)
    assert len(games) == 6
    for name, gid in games.items():
        stored = json.loads((games_dir / f"{gid}.json").read_text())
        again = json.loads(json.dumps(game_to_dict(game_from_dict(stored))))
        assert again == stored, name


def test_reopening_the_store_keeps_ids_times_and_pictures(
    client: TestClient, services: Services, reader: FakeReader, games_dir: Path
) -> None:
    games = _representative_games(client, services, reader)
    fresh = GameStore(games_dir)
    for name, gid in games.items():
        live, again = services.game(gid), fresh.get(gid)
        assert again is not None, name
        assert (again.id, again.created_at, again.updated_at) == (
            live.id, live.created_at, live.updated_at,
        ), name  # fmt: skip
        assert [p.id for p in again.pictures] == [p.id for p in live.pictures], name
        assert again.board == live.board and again.rack == live.rack, name


def test_reading_pages_changes_nothing_on_disk(
    client: TestClient, services: Services, reader: FakeReader, games_dir: Path
) -> None:
    games = _representative_games(client, services, reader)
    before = _fingerprint(games_dir)
    client.get("/")
    client.get("/options")
    for gid in games.values():
        assert client.get(f"/game/{gid}").status_code == 200
        client.get(f"/options?game={gid}")
    assert _fingerprint(games_dir) == before


def test_previewing_a_suggestion_changes_nothing_on_disk(
    client: TestClient, games_dir: Path
) -> None:
    gid = started_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "solve"))
    before = _fingerprint(games_dir)
    assert client.get(f"/game/{gid}/preview/0").status_code == 200
    assert client.get(f"/game/{gid}/preview/1").status_code == 200
    assert _fingerprint(games_dir) == before


def test_the_solver_ranking_for_a_position_is_stable(
    client: TestClient, services: Services
) -> None:
    """Same board, rack and settings: same moves, scores and order, every time."""
    gid = started_game(client)
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO")}, "SATIRE?", "solve"))
    first = services.game(gid).last_result
    assert first is not None and first.moves
    summary = [(m.word, m.score.total, m.start, m.direction) for m in first.moves[:10]]
    client.post(f"/game/{gid}/solve")
    second = services.game(gid).last_result
    assert second is not None
    assert [(m.word, m.score.total, m.start, m.direction) for m in second.moves[:10]] == summary
    scores = [m.score.total for m in first.moves]
    assert scores == sorted(scores, reverse=True)


def test_repeating_a_confirmation_does_not_record_a_second_turn(
    client: TestClient, services: Services
) -> None:
    gid = started_game(client, phase="start_opponent")
    client.post(f"/game/{gid}/board", data=form({7: (5, "HELLO"), 8: (5, "A"), 9: (5, "T")}, "ABC"))
    data = _form_for(services, gid, "confirm_opponent")
    client.post(f"/game/{gid}/board", data=data)
    turns = len(services.game(gid).turns)
    client.post(f"/game/{gid}/board", data=data)  # a double click, or a retried request
    assert len(services.game(gid).turns) == turns
