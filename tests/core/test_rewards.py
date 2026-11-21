"""Mystery square rewards: what each one does to a move's score."""

from __future__ import annotations

import pytest

from tests.conftest import word_move
from wwf.core.board import Board
from wwf.core.config import BoardConfig, load_board_config
from wwf.core.rewards import REWARDS, best_case, mystery_cells, score_with_rewards


@pytest.fixture(scope="module")
def sup() -> BoardConfig:
    return load_board_config(mode="super_score")


def play(sup: BoardConfig):  # type: ignore[no-untyped-def]
    board = Board.empty(sup)
    return board, word_move(board, 1, 0, "CAT")  # C(1,0) A(1,1) on "?" T(1,2): 4+1+1 = 6 plain


def test_finds_the_mystery_squares_a_move_covers(sup: BoardConfig) -> None:
    board, move = play(sup)
    assert mystery_cells(board, move) == [(1, 1)]


@pytest.mark.parametrize(
    ("reward", "total"),
    [
        ("", 6), ("p", 6), ("f", 6),  # no change to the move
        ("d", 7), ("t", 8), ("q", 9),  # the A (1 point) on the square
        ("D", 12), ("T", 18), ("Q", 24),  # the whole word
        ("+", 16),
    ],
)  # fmt: skip
def test_each_reward(sup: BoardConfig, reward: str, total: int) -> None:
    board, move = play(sup)
    s = score_with_rewards(board, move, sup, {(1, 1): reward} if reward else {})
    assert s.total == total
    assert s.reward_points == (10 if reward == "+" else 0)


def test_best_case(sup: BoardConfig) -> None:
    board, move = play(sup)
    total, chosen = best_case(board, move, sup)
    assert total == 24 and chosen == {(1, 1): "Q"}
    assert set(REWARDS) >= set(chosen.values())


def test_a_move_off_the_mystery_squares_has_no_rewards(sup: BoardConfig) -> None:
    board = Board.empty(sup)
    move = word_move(board, 0, 2, "CAT")
    assert mystery_cells(board, move) == []
    assert best_case(board, move, sup)[0] == score_with_rewards(board, move, sup, {}).total
