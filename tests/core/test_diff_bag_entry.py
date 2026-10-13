from __future__ import annotations

from tests.conftest import place
from wwf.core.bag import bag_report
from wwf.core.board import Board, Tile
from wwf.core.config import BoardConfig
from wwf.core.diff import diff_boards
from wwf.core.entry import placements_for_word
from wwf.core.move import Direction
from wwf.core.rack import Rack


def test_diff_new_and_changed(cfg: BoardConfig, empty: Board) -> None:
    before = place(empty, 7, 5, "HELLO")
    after = place(before, 8, 5, "AT", across=False)  # new tiles under the H
    after = Board(
        after.size,
        tuple(
            tuple(Tile("S") if (r, c) == (7, 9) else t for c, t in enumerate(row))
            for r, row in enumerate(after.cells)
        ),
        after.bonuses,
    )  # and the O misread as S
    d = diff_boards(before, after)
    assert [(p.row, p.col, p.tile.letter) for p in d.placements] == [(8, 5, "A"), (9, 5, "T")]
    assert [(r, c, b.letter, a.letter if a else None) for r, c, b, a in d.changed] == [
        (7, 9, "O", "S")
    ]
    assert diff_boards(before, before).placements == ()


def test_diff_ignores_blank_flag_but_sees_removed(cfg: BoardConfig, empty: Board) -> None:
    before = place(empty, 7, 5, "HEL")
    marked = place(empty, 7, 5, "HeL")  # the E re-marked as a blank: a correction, not a change
    assert diff_boards(before, marked).changed == ()
    gone = place(empty, 7, 5, "HE")
    assert [(r, c) for r, c, _, a in diff_boards(before, gone).changed] == [(7, 7)]


def test_bag_report(cfg: BoardConfig, empty: Board) -> None:
    board = place(empty, 7, 5, "HELLO")
    r = bag_report(cfg, board, Rack.from_string("SATIRE?"))
    assert r.on_board == 5 and r.unseen_total == 104 - 5 - 7 and r.in_bag == r.unseen_total - 7
    assert r.unseen["E"] == 13 - 1 - 1  # one on the board, one on the rack
    assert r.unseen["?"] == 1
    assert r.over == {}


def test_bag_report_flags_hidden_blank(cfg: BoardConfig, empty: Board) -> None:
    board = place(empty, 7, 0, "QQ")  # the bag has one Q, so one of these must be a blank
    r = bag_report(cfg, board, Rack(()))
    assert r.over == {"Q": 1} and "Q" not in r.unseen
    assert r.unseen_total == 104 - 2


def test_bag_never_negative(cfg: BoardConfig, empty: Board) -> None:
    r = bag_report(cfg, empty, Rack(()), opponent_rack=7)
    assert r.in_bag == 104 - 7
    assert bag_report(cfg, empty, Rack(()), opponent_rack=500).in_bag == 0


def test_placements_for_word(cfg: BoardConfig, empty: Board) -> None:
    board = place(empty, 7, 5, "HELLO")
    ps, problems = placements_for_word(board, "HELLOS", 7, 5, Direction.ACROSS)
    assert problems == [] and [(p.row, p.col, p.tile.letter) for p in ps] == [(7, 10, "S")]

    ps, problems = placements_for_word(board, "OATS", 7, 9, Direction.DOWN)  # through the O
    assert problems == []
    assert [(p.row, p.col, p.tile.letter, p.tile.is_blank) for p in ps] == [
        (8, 9, "A", False), (9, 9, "T", False), (10, 9, "S", False),
    ]  # fmt: skip
    # lowercase new letters are blanks; a lowercase letter on an existing tile still matches
    ps, _ = placements_for_word(board, "oaT", 7, 9, Direction.DOWN)
    assert [(p.tile.letter, p.tile.is_blank) for p in ps] == [("A", True), ("T", False)]


def test_placements_for_word_problems(cfg: BoardConfig, empty: Board) -> None:
    board = place(empty, 7, 5, "HELLO")
    assert placements_for_word(board, "", 0, 0, Direction.ACROSS)[1] == ["empty_word"]
    assert placements_for_word(board, "A1", 0, 0, Direction.ACROSS)[1] == ["bad_letter"]
    assert placements_for_word(board, "HELLO", 7, 5, Direction.ACROSS)[1] == ["no_new_tiles"]
    assert placements_for_word(board, "HELPO", 7, 5, Direction.ACROSS)[1] == [
        "letter_mismatch:8,9:L"
    ]
    assert placements_for_word(board, "ABC", 0, 13, Direction.ACROSS)[1] == ["out_of_bounds"]
