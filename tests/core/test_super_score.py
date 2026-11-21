"""Super Score mode: its board, the quadruple squares, and that the solver agrees with scoring."""

from __future__ import annotations

import random

import pytest

from tests.conftest import place, word_move
from tests.solvers.conftest import random_game_board
from tests.solvers.reference import reference_moves
from wwf.core.board import Board, Bonus
from wwf.core.config import MODES, BoardConfig, ConfigError, load_board_config
from wwf.core.dictionary import TrieDictionary
from wwf.core.rack import Rack
from wwf.core.scoring import score_move, score_raw
from wwf.solvers.anchor_trie import AnchorTrieSolver, generate_raw_moves


@pytest.fixture(scope="module")
def sup() -> BoardConfig:
    return load_board_config(mode="super_score")


def test_every_mode_has_a_config() -> None:
    for mode in MODES:
        assert load_board_config(mode=mode).mode == mode
    assert load_board_config().mode == "normal"
    with pytest.raises(ConfigError):
        load_board_config(mode="nope")


def test_super_score_board_layout(sup: BoardConfig) -> None:
    board = Board.empty(sup)
    assert board.bonus(0, 2) is Bonus.QW and board.bonus(14, 12) is Bonus.QW
    assert board.bonus(0, 6) is Bonus.QL and board.bonus(8, 14) is Bonus.QL
    assert board.bonus(1, 1) is Bonus.MYSTERY and board.bonus(11, 11) is Bonus.MYSTERY
    assert board.bonus(1, 5) is Bonus.TW and board.bonus(3, 7) is Bonus.DW
    assert board.center == (7, 7)
    counts = {b: sum(row.count(b) for row in board.bonuses) for b in Bonus}
    assert counts[Bonus.QW] == 8 and counts[Bonus.QL] == 8 and counts[Bonus.MYSTERY] == 12


def test_quadruple_word(sup: BoardConfig) -> None:
    board = Board.empty(sup)
    s = score_move(board, word_move(board, 0, 2, "CAT"), sup)  # QW at (0,2)
    assert s.total == (4 + 1 + 1) * 4


def test_quadruple_letter(sup: BoardConfig) -> None:
    board = Board.empty(sup)
    s = score_move(board, word_move(board, 0, 6, "ZAX"), sup)  # QL at (0,6) and (0,8)
    assert s.total == 10 * 4 + 1 + 8 * 4


def test_mystery_square_scores_as_plain(sup: BoardConfig) -> None:
    board = Board.empty(sup)
    assert score_move(board, word_move(board, 1, 0, "CAT"), sup).total == 4 + 1 + 1  # (1,1) is "?"


def test_quadruple_squares_only_count_for_new_tiles(sup: BoardConfig) -> None:
    board = place(Board.empty(sup), 0, 2, "CAT")
    s = score_move(board, word_move(board, 0, 2, "...S"), sup)
    assert s.total == 4 + 1 + 1 + 1  # CAT's quadruple word is spent; S lands on a plain square


def test_score_raw_agrees_with_score_move(sup: BoardConfig, dictionary: TrieDictionary) -> None:
    board, bag = random_game_board(sup, dictionary, seed=3, turns=3)
    rack = Rack(tuple(random.Random(3).sample(bag, 6)))
    solved = AnchorTrieSolver().solve(board, rack, dictionary, sup, max_results=20)
    assert solved.moves
    for sm in solved.moves:
        raw = tuple((p.row, p.col, p.tile.letter, p.tile.is_blank) for p in sm.move.placements)
        assert score_raw(board, raw, sup) == score_move(board, sm.move, sup).total == sm.score.total


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_solver_matches_reference_on_super_score(
    sup: BoardConfig, dictionary: TrieDictionary, seed: int
) -> None:
    board, bag = random_game_board(sup, dictionary, seed, seed + 1)
    rack = Rack(tuple(random.Random(seed).sample(bag, 5)))
    expected = reference_moves(board, rack, dictionary, sup)
    raw_moves, _ = generate_raw_moves(board, rack, dictionary)
    got = {frozenset(raw): score_raw(board, raw, sup) for raw in raw_moves}
    assert got == expected


def test_modes_differ_on_the_same_play(sup: BoardConfig) -> None:
    normal = load_board_config()
    a = score_move(Board.empty(sup), word_move(Board.empty(sup), 0, 2, "CAT"), sup).total
    b = score_move(Board.empty(normal), word_move(Board.empty(normal), 0, 2, "CAT"), normal).total
    assert a != b
