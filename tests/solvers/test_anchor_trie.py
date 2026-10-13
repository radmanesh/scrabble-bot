from __future__ import annotations

import random
import time

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.conftest import place
from tests.solvers.conftest import random_game_board
from tests.solvers.reference import reference_moves
from wwf.core.board import Board, Tile
from wwf.core.config import BoardConfig
from wwf.core.dictionary import TrieDictionary
from wwf.core.move import Placement, build_move
from wwf.core.rack import Rack
from wwf.core.scoring import score_move, score_raw, validate_move
from wwf.solvers.anchor_trie import AnchorTrieSolver, cross_checks, generate_raw_moves
from wwf.solvers.base import ParamError

SOLVER = AnchorTrieSolver()


def _cases() -> list[tuple[int, int, int, bool]]:
    """(seed, turns, rack size, with blank) for the 20 randomised equivalence cases."""
    cases = [(seed, seed % 7, 5, False) for seed in range(16)]
    cases += [(100 + seed, 1 + seed, 3, True) for seed in range(4)]
    return cases


@pytest.mark.parametrize(("seed", "turns", "rack_size", "with_blank"), _cases())
def test_matches_reference_generator(
    cfg: BoardConfig,
    dictionary: TrieDictionary,
    seed: int,
    turns: int,
    rack_size: int,
    with_blank: bool,
) -> None:
    board, bag = random_game_board(cfg, dictionary, seed, turns)
    rng = random.Random(seed)
    tiles = rng.sample(bag, rack_size - (1 if with_blank else 0))
    rack = Rack(tuple(tiles) + (("?",) if with_blank else ()))

    expected = reference_moves(board, rack, dictionary, cfg)
    raw_moves, _ = generate_raw_moves(board, rack, dictionary)
    got = {frozenset(raw): score_raw(board, raw, cfg) for raw in raw_moves}

    assert len(got) == len(raw_moves), "generator emitted a move twice"
    missing = expected.keys() - got.keys()
    extra = got.keys() - expected.keys()
    assert not missing, (
        f"solver missed {len(missing)} legal moves, e.g. {sorted(next(iter(missing)))}"
    )
    assert not extra, (
        f"solver produced {len(extra)} illegal moves, e.g. {sorted(next(iter(extra)))}"
    )
    assert got == expected, "score_raw disagrees with score_move"

    result = SOLVER.solve(board, rack, dictionary, cfg, max_results=500)
    if expected:
        assert result.moves[0].score.total == max(expected.values())
    else:
        assert result.moves == ()


def test_returned_moves_validate(cfg: BoardConfig, dictionary: TrieDictionary) -> None:
    for seed in range(5):
        board, _ = random_game_board(cfg, dictionary, seed, 6)
        rack = Rack.from_string("RETAIN?")
        result = SOLVER.solve(board, rack, dictionary, cfg, max_results=500)
        assert result.moves
        for sm in result.moves:
            assert validate_move(board, sm.move, dictionary, cfg, rack) == []
            assert sm.score == score_move(board, sm.move, cfg)


def test_empty_board_moves_cover_center(cfg: BoardConfig, dictionary: TrieDictionary) -> None:
    board = Board.empty(cfg)
    result = SOLVER.solve(board, Rack.from_string("HELLOAB"), dictionary, cfg, max_results=500)
    assert result.stats["anchors"] == 1
    assert result.moves
    for sm in result.moves:
        assert any((p.row, p.col) == (7, 7) for p in sm.move.placements)


def test_sorted_and_limited(cfg: BoardConfig, dictionary: TrieDictionary) -> None:
    board, _ = random_game_board(cfg, dictionary, 3, 5)
    rack = Rack.from_string("SATIRE")
    result = SOLVER.solve(board, rack, dictionary, cfg, max_results=7, tie_break="score_only")
    assert len(result.moves) == 7
    keys = [(-m.score.total, m.move.main_word) for m in result.moves]
    assert keys == sorted(keys)
    floor = SOLVER.solve(board, rack, dictionary, cfg, min_score=30, max_results=500)
    assert all(m.score.total >= 30 for m in floor.moves)
    assert set(result.stats) >= {"anchors", "cross_checks", "candidates", "moves"}


def test_ties_at_cutoff_are_resolved_by_leave(cfg: BoardConfig, dictionary: TrieDictionary) -> None:
    board, _ = random_game_board(cfg, dictionary, 4, 3)
    rack = Rack.from_string("QAEIRST")
    full = SOLVER.solve(board, rack, dictionary, cfg, max_results=500)
    top = SOLVER.solve(board, rack, dictionary, cfg, max_results=5)
    assert [m.move for m in top.moves] == [m.move for m in full.moves[:5]]


def test_blank_branching_off_ignores_blanks(cfg: BoardConfig, dictionary: TrieDictionary) -> None:
    board = Board.empty(cfg)
    result = SOLVER.solve(
        board, Rack.from_string("CAT?"), dictionary, cfg, allow_blank_branching=False
    )
    assert all(not p.tile.is_blank for m in result.moves for p in m.move.placements)
    assert {m.move.main_word for m in result.moves} == {"CAT", "ACT", "TA", "AT"}


def test_blank_scores_zero(cfg: BoardConfig, dictionary: TrieDictionary) -> None:
    result = SOLVER.solve(Board.empty(cfg), Rack.from_string("?"), dictionary, cfg)
    assert result.moves == ()  # one tile on an empty board can never form a word
    board = place(Board.empty(cfg), 7, 6, "X")
    result = SOLVER.solve(board, Rack.from_string("?"), dictionary, cfg, max_results=500)
    words = {m.move.main_word for m in result.moves}
    assert {"AX", "EX", "OX", "XI", "XU"} <= words
    assert all(m.score.total == 8 for m in result.moves)  # only the existing X scores


def test_cross_checks(cfg: BoardConfig, dictionary: TrieDictionary) -> None:
    board = place(Board.empty(cfg), 6, 7, "A")
    board = board.with_tiles(
        [(8, 7, Tile("E"))]
    )  # A_E: ACE, AGE, ALE, APE, ARE, ATE, AWE, AXE, AYE...
    grid = [[t.letter if t else None for t in row] for row in board.cells]
    checks = cross_checks(grid, dictionary.root(), {(7, 7), (5, 7)})
    assert {"C", "G", "L", "P", "R", "T", "W", "X", "Y"} <= checks[(7, 7)]
    assert "Q" not in checks[(7, 7)]
    assert checks[(5, 7)] == frozenset(  # _A forms a two-letter word ending in A
        ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if dictionary.contains(ch + "A")
    )


def test_bad_params_rejected(cfg: BoardConfig, dictionary: TrieDictionary) -> None:
    board, rack = Board.empty(cfg), Rack.from_string("CAT")
    for bad in ({"max_results": 0}, {"tie_break": "nope"}, {"bogus": 1}, {"max_results": 1.5}):
        with pytest.raises(ParamError):
            SOLVER.solve(board, rack, dictionary, cfg, **bad)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), turns=st.integers(0, 6), pick=st.integers(0, 10_000))
def test_legal_move_scores_are_transpose_invariant(
    cfg: BoardConfig, dictionary: TrieDictionary, seed: int, turns: int, pick: int
) -> None:
    board, bag = random_game_board(cfg, dictionary, seed, turns)
    rack = Rack(tuple(random.Random(seed).sample(bag, 6)))
    raw_moves, _ = generate_raw_moves(board, rack, dictionary)
    if not raw_moves:
        return
    raw = raw_moves[pick % len(raw_moves)]
    move = build_move(board, [Placement(r, c, Tile(ch, b)) for r, c, ch, b in raw])
    assert validate_move(board, move, dictionary, cfg, rack) == []
    assert (
        score_move(board, move, cfg).total
        == score_move(board.transpose(), move.transpose(), cfg).total
    )


@pytest.mark.slow
def test_performance_full_board_with_blank(cfg: BoardConfig, dictionary: TrieDictionary) -> None:
    """Spec bar: a full mid-game board, 7-tile rack with one blank, under 500 ms on one core.
    Measured as CPU time so a busy CI machine does not make it flaky."""
    board, _ = random_game_board(cfg, dictionary, 1, 14)
    assert board.tile_count() >= 50
    rack = Rack.from_string("AEIRST?")
    SOLVER.solve(board, rack, dictionary, cfg)  # warm up
    best = min(_cpu_ms(lambda: SOLVER.solve(board, rack, dictionary, cfg)) for _ in range(3))
    assert best < 500, f"{best:.0f} ms"


def _cpu_ms(fn: object) -> float:
    t = time.process_time()
    fn()  # type: ignore[operator]
    return (time.process_time() - t) * 1000
