from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import move_of, place, word_move
from wwf.core.board import Board, Tile
from wwf.core.config import BoardConfig
from wwf.core.move import Direction, Placement, build_move
from wwf.core.scoring import WordScore, score_move, score_raw


def test_first_move_hello(cfg: BoardConfig, empty: Board) -> None:
    move = word_move(empty, 7, 5, "HELLO")
    assert move.direction is Direction.ACROSS
    assert move.main_word == "HELLO"
    assert move.start == (7, 5)
    s = score_move(empty, move, cfg)
    # H3 E1 L2 L2 O1; (7,7) is the centre, which carries no bonus in WWF
    assert s.words == (WordScore("HELLO", (7, 5), Direction.ACROSS, 9),)
    assert s.bingo_bonus == 0
    assert s.total == 9


def test_blank_on_triple_word(cfg: BoardConfig, empty: Board) -> None:
    board = place(empty, 0, 4, "AT")
    # blank C on TW (0,3), S on TL (0,6): (0 + 1 + 1 + 1*3) * 3
    move = move_of(board, [(0, 3, "c"), (0, 6, "S")])
    assert move.main_word == "CATS"
    s = score_move(board, move, cfg)
    assert s.words == (WordScore("CATS", (0, 3), Direction.ACROSS, 15),)
    assert s.total == 15


def test_blank_scores_zero_even_on_letter_bonus(cfg: BoardConfig, empty: Board) -> None:
    board = place(empty, 0, 4, "AT")
    blank = score_move(board, move_of(board, [(0, 3, "c"), (0, 6, "s")]), cfg)
    assert blank.total == (0 + 1 + 1 + 0) * 3


def test_bingo(cfg: BoardConfig, empty: Board) -> None:
    move = word_move(empty, 7, 1, "RETAINS")
    s = score_move(empty, move, cfg)
    # R1 E1 T1x2(DL at 7,3) A1 I1 N2 S1 = 9, plus the 35-point WWF bingo
    assert s.words == (WordScore("RETAINS", (7, 1), Direction.ACROSS, 9),)
    assert s.bingo_bonus == 35
    assert s.total == 44


def test_parallel_play_scores_every_word(cfg: BoardConfig, empty: Board) -> None:
    board = place(empty, 7, 5, "HELLO")
    s = score_move(board, move_of(board, [(8, 8, "A"), (8, 9, "X")]), cfg)
    assert {(w.word, w.points) for w in s.words} == {("AX", 9), ("LA", 3), ("OX", 9)}
    assert s.total == 21


def test_existing_tiles_score_face_value(cfg: BoardConfig, empty: Board) -> None:
    # existing tile on a DL square earns no bonus; the new S does not land on one
    board = place(empty, 7, 1, "RETAIN")
    s = score_move(board, move_of(board, [(7, 7, "S")]), cfg)
    assert s.total == 1 + 1 + 1 + 1 + 1 + 2 + 1


def test_word_bonuses_stack(cfg: BoardConfig, empty: Board) -> None:
    # (1,5) and (1,9) are both DW: word score x4
    board = place(empty, 1, 6, "ERM")
    s = score_move(board, move_of(board, [(1, 5, "T"), (1, 9, "S")]), cfg)
    assert s.words[0].word == "TERMS"
    assert s.total == (1 + 1 + 1 + 4 + 1) * 4


def test_single_tile_hook_counts_only_real_words(cfg: BoardConfig, empty: Board) -> None:
    board = place(empty, 7, 5, "HELLO")
    hook = move_of(board, [(8, 9, "X")])
    assert hook.direction is Direction.DOWN
    assert hook.main_word == "OX"
    s = score_move(board, hook, cfg)
    assert [w.word for w in s.words] == ["OX"]
    assert s.total == 9


# --- transpose property -------------------------------------------------------------------


@st.composite
def board_and_move(draw: st.DrawFn, cfg: BoardConfig) -> tuple[Board, list[Placement]]:
    size = cfg.size
    letters = st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    occupied = draw(
        st.dictionaries(
            st.tuples(st.integers(0, size - 1), st.integers(0, size - 1)),
            st.tuples(letters, st.booleans()),
            max_size=40,
        )
    )
    board = Board.empty(cfg).with_tiles(
        (r, c, Tile(ch, blank)) for (r, c), (ch, blank) in occupied.items()
    )
    across = draw(st.booleans())
    line = draw(st.integers(0, size - 1))
    start = draw(st.integers(0, size - 1))
    n = draw(st.integers(1, 7))
    cells: list[tuple[int, int]] = []
    pos = start
    while len(cells) < n and pos < size:
        r, c = (line, pos) if across else (pos, line)
        if board.get(r, c) is None:
            cells.append((r, c))
        pos += 1
    if not cells:
        r, c = (line, start) if across else (start, line)
        board = Board(
            board.size,
            tuple(
                tuple(None if (rr, cc) == (r, c) else t for cc, t in enumerate(row))
                for rr, row in enumerate(board.cells)
            ),
            board.bonuses,
        )
        cells = [(r, c)]
    placements = [Placement(r, c, Tile(draw(letters), draw(st.booleans()))) for r, c in cells]
    return board, placements


def _canonical(breakdown_words: tuple[WordScore, ...]) -> list[tuple[str, int]]:
    return sorted((w.word, w.points) for w in breakdown_words)


@settings(max_examples=300, deadline=None)
@given(data=st.data())
def test_score_is_transpose_invariant(cfg: BoardConfig, data: st.DataObject) -> None:
    board, placements = data.draw(board_and_move(cfg))
    move = build_move(board, placements)
    t_board = board.transpose()
    t_move = move.transpose()

    a = score_move(board, move, cfg)
    b = score_move(t_board, t_move, cfg)
    assert a.total == b.total
    assert _canonical(a.words) == _canonical(b.words)

    # the allocation-free scorer the solver ranks with agrees with the full breakdown
    raw = [(p.row, p.col, p.tile.letter, p.tile.is_blank) for p in move.placements]
    assert score_raw(board, raw, cfg) == a.total

    # rebuilding the move from scratch on the transposed board agrees too
    rebuilt = build_move(t_board, t_move.placements)
    assert score_move(t_board, rebuilt, cfg).total == a.total
