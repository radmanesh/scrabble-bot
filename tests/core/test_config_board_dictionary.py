from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from wwf.core.board import Board, Bonus, Tile
from wwf.core.config import BoardConfig, ConfigError, board_config_from_dict, config_dir
from wwf.core.dictionary import TrieDictionary, load_dictionary
from wwf.core.rack import Rack


def _raw() -> dict[str, object]:
    with (config_dir() / "wwf_classic.json").open() as f:
        data: dict[str, object] = json.load(f)
    return data


def test_config_loads(cfg: BoardConfig) -> None:
    assert cfg.size == 15
    assert cfg.center == (7, 7)
    assert cfg.bingo_bonus == 35
    assert sum(cfg.bag.values()) == 104
    assert cfg.letter_values["Q"] == 10 and cfg.letter_values["?"] == 0


def test_config_rejects_wrong_row_count() -> None:
    data = _raw()
    data["layout"] = list(data["layout"])[:14]  # type: ignore[call-overload]
    with pytest.raises(ConfigError, match="rows"):
        board_config_from_dict(data)


def test_config_rejects_asymmetric_layout() -> None:
    data = copy.deepcopy(_raw())
    layout = list(data["layout"])  # type: ignore[call-overload]
    layout[0] = "T" + layout[0][1:]
    data["layout"] = layout
    with pytest.raises(ConfigError, match="symmetric"):
        board_config_from_dict(data)


def test_config_rejects_wrong_row_width() -> None:
    data = copy.deepcopy(_raw())
    layout = list(data["layout"])  # type: ignore[call-overload]
    layout[7] = layout[7] + "."
    data["layout"] = layout
    with pytest.raises(ConfigError, match="chars"):
        board_config_from_dict(data)


def test_board_bonuses(empty: Board) -> None:
    assert empty.bonus(0, 3) is Bonus.TW
    assert empty.bonus(7, 7) is Bonus.CENTER
    assert empty.bonus(14, 11) is Bonus.TW
    assert empty.bonus(1, 5) is Bonus.DW
    assert empty.center == (7, 7)


def test_anchors(empty: Board) -> None:
    assert empty.anchors() == {(7, 7)}
    b = empty.with_tiles([(7, 7, Tile("A"))])
    assert b.anchors() == {(6, 7), (8, 7), (7, 6), (7, 8)}
    corner = empty.with_tiles([(0, 0, Tile("A"))])
    assert corner.anchors() == {(0, 1), (1, 0)}


def test_board_rows_roundtrip(cfg: BoardConfig, empty: Board) -> None:
    b = empty.with_tiles([(7, 7, Tile("A")), (7, 8, Tile("T", is_blank=True))])
    rows = b.to_rows()
    assert rows[7] == ".......At......"
    assert Board.from_rows(cfg, rows) == b
    assert b.transpose().transpose() == b
    assert b.transpose().get(8, 7) == Tile("T", is_blank=True)


def test_with_move_rejects_occupied(empty: Board) -> None:
    b = empty.with_tiles([(7, 7, Tile("A"))])
    with pytest.raises(ValueError):
        b.with_tiles([(7, 7, Tile("B"))])


def test_rack() -> None:
    r = Rack.from_string("ab?c")
    assert r.tiles == ("A", "B", "?", "C")
    assert r.counts()["?"] == 1
    with pytest.raises(ValueError):
        Rack.from_string("ABCDEFGH")


def test_dictionary_basics(dictionary: TrieDictionary) -> None:
    assert len(dictionary) > 170_000
    assert dictionary.contains("XI") and dictionary.contains("hello")
    assert not dictionary.contains("A")  # single letters are never words
    node = dictionary.root()
    for ch in "CAT":
        node = node.children[ch]
    assert node.terminal


def test_dictionary_cache_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "words.txt"
    src.write_text("cat\ncats\na\nDOG\n\n")
    cache = tmp_path / "cache"
    first = load_dictionary(src, cache)
    assert len(list(cache.iterdir())) == 1
    second = load_dictionary(src, cache)
    assert isinstance(second, TrieDictionary)
    assert len(first) == len(second) == 3
    src.write_text("cat\n")  # new content -> new hash -> rebuilt
    assert len(load_dictionary(src, cache)) == 1


def test_current_layout_is_the_default() -> None:
    from wwf.core.config import load_board_config

    current = load_board_config()
    assert current.name == "wwf_current"
    assert current.layout[2] == ".d..d.....d..d."  # read from the owner's photos
    assert current.layout[7] == "...D...*...D..."
    assert current.letter_values["Z"] == 10 and current.bingo_bonus == 35
