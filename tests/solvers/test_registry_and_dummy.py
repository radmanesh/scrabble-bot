from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import place
from wwf.core.board import Board
from wwf.core.config import BoardConfig
from wwf.core.dictionary import TrieDictionary
from wwf.core.rack import Rack
from wwf.core.scoring import validate_move
from wwf.solvers.base import ParamError, SolverParam, resolve_params
from wwf.solvers.dummy_random import DummyRandomSolver
from wwf.solvers.registry import RegistryError, discover_solvers, load_registry


def _write(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    path = tmp_path / "solvers.json"
    path.write_text(json.dumps({"solvers": entries}))
    return path


def test_registry_loads_both_solvers() -> None:
    reg = load_registry()
    assert reg.default_id == "anchor_trie"
    assert reg.get("anchor_trie").display_name == "Smart brute force"
    assert reg.get("dummy_random").display_name == "Random legal play (demo)"
    planned = {i.id for i in reg.infos() if i.planned}
    assert planned == {"gaddag", "leave_aware", "monte_carlo"}
    with pytest.raises(KeyError):
        reg.get("gaddag")


def test_registry_rejects_unknown_id(tmp_path: Path) -> None:
    path = _write(
        tmp_path, [{"id": "anchor_trie", "enabled": True}, {"id": "nope", "enabled": True}]
    )
    with pytest.raises(RegistryError, match="nope"):
        load_registry(path)


def test_registry_rejects_two_defaults(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [{"id": "anchor_trie", "default": True}, {"id": "dummy_random", "default": True}],
    )
    with pytest.raises(RegistryError, match="default"):
        load_registry(path)


def test_registry_disabled_implementation_is_listed_but_not_runnable(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"id": "anchor_trie"}, {"id": "dummy_random", "enabled": False}])
    reg = load_registry(path)
    assert reg.info("dummy_random").planned
    with pytest.raises(KeyError):
        reg.get("dummy_random")


def test_discovery_finds_solver_classes() -> None:
    found = discover_solvers()
    assert set(found) == {"anchor_trie", "dummy_random"}


def test_dummy_honours_max_results(cfg: BoardConfig, dictionary: TrieDictionary) -> None:
    board = place(Board.empty(cfg), 7, 5, "HELLO")
    rack = Rack.from_string("SAEIOTX")
    solver = DummyRandomSolver()
    result = solver.solve(board, rack, dictionary, cfg, attempts=2000, max_results=3, seed=1)
    assert result.stats["attempts"] == 2000
    assert result.stats["legal_found"] > 3
    assert len(result.moves) == 3
    for sm in result.moves:
        assert len(sm.move.placements) == 1
        assert validate_move(board, sm.move, dictionary, cfg, rack) == []
    totals = [m.score.total for m in result.moves]
    assert totals == sorted(totals, reverse=True)


def test_dummy_is_deterministic_with_seed(cfg: BoardConfig, dictionary: TrieDictionary) -> None:
    board = place(Board.empty(cfg), 7, 5, "HELLO")
    rack = Rack.from_string("SAEIOT?")
    solver = DummyRandomSolver()
    a = solver.solve(board, rack, dictionary, cfg, seed=7)
    b = solver.solve(board, rack, dictionary, cfg, seed=7)
    assert [m.move for m in a.moves] == [m.move for m in b.moves]


def test_dummy_on_empty_board_finds_nothing(cfg: BoardConfig, dictionary: TrieDictionary) -> None:
    result = DummyRandomSolver().solve(Board.empty(cfg), Rack.from_string("AB"), dictionary, cfg)
    assert result.moves == ()


def test_resolve_params() -> None:
    decl = (
        SolverParam("n", "N", "int", 5, min=1, max=10),
        SolverParam("on", "On", "bool", True),
        SolverParam("mode", "Mode", "choice", "a", choices=("a", "b")),
        SolverParam("w", "W", "float", 0.5, min=0.0, max=1.0),
    )
    assert resolve_params(decl, {}) == {"n": 5, "on": True, "mode": "a", "w": 0.5}
    assert resolve_params(decl, {"n": "7", "on": "false", "w": 1}) == {
        "n": 7, "on": False, "mode": "a", "w": 1.0,
    }  # fmt: skip
    with pytest.raises(ParamError) as e:
        resolve_params(decl, {"n": 11, "mode": "c", "w": -1, "x": 1})
    assert len(e.value.errors) == 4
