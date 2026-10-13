from __future__ import annotations

import json
from pathlib import Path

import pytest

from wwf.core.board import Board
from wwf.core.config import BoardConfig
from wwf.core.dictionary import (
    DictionaryError,
    default_dictionary_registry,
    load_dictionary_registry,
)
from wwf.core.rack import Rack
from wwf.solvers.anchor_trie import AnchorTrieSolver


def _setup(tmp_path: Path, files: dict[str, str], config: dict[str, object] | None) -> Path:
    words = tmp_path / "dictionaries"
    words.mkdir()
    for name, content in files.items():
        (words / name).write_text(content)
    cfg_path = tmp_path / "dictionaries.json"
    if config is not None:
        cfg_path.write_text(json.dumps(config))
    return cfg_path


def test_project_registry_offers_both_lists() -> None:
    reg = default_dictionary_registry()
    assert {i.id for i in reg.infos()} >= {"enable1", "words"}
    assert reg.default_id == "enable1"
    assert not reg.get("enable1").contains("QI")
    assert reg.get("words").contains("QI")
    assert reg.get() is reg.get("enable1")  # loaded once, then reused
    with pytest.raises(KeyError):
        reg.get("nope")


def test_unlisted_file_is_discovered(tmp_path: Path) -> None:
    cfg = _setup(
        tmp_path,
        {"a.txt": "cat\ndog\n", "extra.txt": "qi\nza\n"},
        {"default": "a", "dictionaries": [{"id": "a", "file": "a.txt", "label": "A"}]},
    )
    reg = load_dictionary_registry(cfg, tmp_path / "dictionaries")
    assert [i.id for i in reg.infos()] == ["a", "extra"]
    assert not reg.is_loaded("extra")
    assert reg.get("extra").contains("ZA")
    assert reg.is_loaded("extra")


def test_works_without_config_file(tmp_path: Path) -> None:
    cfg = _setup(tmp_path, {"only.txt": "cat\n"}, None)
    reg = load_dictionary_registry(cfg, tmp_path / "dictionaries")
    assert reg.default_id == "only"


def test_env_overrides_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _setup(tmp_path, {"a.txt": "cat\n", "b.txt": "dog\n"}, {"default": "a"})
    monkeypatch.setenv("WWF_DICTIONARY", "b")
    assert load_dictionary_registry(cfg, tmp_path / "dictionaries").default_id == "b"
    monkeypatch.setenv("WWF_DICTIONARY", "missing")
    with pytest.raises(DictionaryError, match="missing"):
        load_dictionary_registry(cfg, tmp_path / "dictionaries")


def test_missing_file_rejected(tmp_path: Path) -> None:
    cfg = _setup(tmp_path, {}, {"dictionaries": [{"id": "x", "file": "x.txt"}]})
    with pytest.raises(DictionaryError, match="not found"):
        load_dictionary_registry(cfg, tmp_path / "dictionaries")


def test_solver_uses_chosen_dictionary(cfg: BoardConfig) -> None:
    reg = default_dictionary_registry()
    board, rack, solver = Board.empty(cfg), Rack.from_string("QI"), AnchorTrieSolver()
    assert solver.solve(board, rack, reg.get("enable1"), cfg).moves == ()
    words = {m.move.main_word for m in solver.solve(board, rack, reg.get("words"), cfg).moves}
    assert words == {"QI"}
