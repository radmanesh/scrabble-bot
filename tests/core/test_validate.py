from __future__ import annotations

from tests.conftest import move_of, place, word_move
from wwf.core.board import Board, Tile
from wwf.core.config import BoardConfig
from wwf.core.dictionary import TrieDictionary
from wwf.core.move import Direction, Move, Placement
from wwf.core.rack import Rack
from wwf.core.scoring import validate_move


def _hello(empty: Board) -> Board:
    return place(empty, 7, 5, "HELLO")


def test_legal_first_move(cfg: BoardConfig, empty: Board, dictionary: TrieDictionary) -> None:
    move = word_move(empty, 7, 5, "HELLO")
    assert validate_move(empty, move, dictionary, cfg, Rack.from_string("HELLOXY")) == []


def test_not_in_line(cfg: BoardConfig, empty: Board, dictionary: TrieDictionary) -> None:
    board = _hello(empty)
    move = move_of(board, [(6, 5, "A"), (8, 6, "A")])
    assert "not_in_line" in validate_move(board, move, dictionary, cfg)


def test_gap_in_word(cfg: BoardConfig, empty: Board, dictionary: TrieDictionary) -> None:
    board = _hello(empty)
    # S hooks HELLO at (7,10); (7,11) is left empty before (7,12)
    move = move_of(board, [(7, 10, "S"), (7, 12, "O")])
    assert "gap_in_word" in validate_move(board, move, dictionary, cfg)
    across_gap = move_of(empty, [(7, 6, "A"), (7, 8, "T")])
    assert "gap_in_word" in validate_move(empty, across_gap, dictionary, cfg)


def test_first_move_not_on_center(
    cfg: BoardConfig, empty: Board, dictionary: TrieDictionary
) -> None:
    move = word_move(empty, 0, 0, "HELLO")
    assert validate_move(empty, move, dictionary, cfg) == ["first_move_not_on_center"]


def test_no_contact(cfg: BoardConfig, empty: Board, dictionary: TrieDictionary) -> None:
    board = _hello(empty)
    move = word_move(board, 0, 0, "CAT")
    assert validate_move(board, move, dictionary, cfg) == ["no_contact_with_existing_tiles"]


def test_cell_occupied(cfg: BoardConfig, empty: Board, dictionary: TrieDictionary) -> None:
    board = _hello(empty)
    move = Move(
        (Placement(7, 5, Tile("J")), Placement(7, 6, Tile("O"))),
        Direction.ACROSS,
        "JO",
        (7, 5),
    )
    assert "cell_occupied" in validate_move(board, move, dictionary, cfg)


def test_word_not_in_dictionary(cfg: BoardConfig, empty: Board, dictionary: TrieDictionary) -> None:
    move = word_move(empty, 7, 6, "QZX")
    assert validate_move(empty, move, dictionary, cfg) == ["word_not_in_dictionary:QZX"]


def test_cross_word_not_in_dictionary(
    cfg: BoardConfig, empty: Board, dictionary: TrieDictionary
) -> None:
    board = _hello(empty)
    # XI under H, E forms HX and EI down: both invalid, while XI itself is a word
    move = move_of(board, [(8, 5, "X"), (8, 6, "I")])
    v = validate_move(board, move, dictionary, cfg)
    assert sorted(v) == ["word_not_in_dictionary:EI", "word_not_in_dictionary:HX"]


def test_single_letter_is_not_a_word(
    cfg: BoardConfig, empty: Board, dictionary: TrieDictionary
) -> None:
    move = move_of(empty, [(7, 7, "A")])
    assert validate_move(empty, move, dictionary, cfg) == ["word_not_in_dictionary:A"]


def test_tile_not_in_rack(cfg: BoardConfig, empty: Board, dictionary: TrieDictionary) -> None:
    move = word_move(empty, 7, 5, "HELLO")
    v = validate_move(empty, move, dictionary, cfg, Rack.from_string("HELOABC"))
    assert v == ["tile_not_in_rack:L"]


def test_blank_consumes_question_mark(
    cfg: BoardConfig, empty: Board, dictionary: TrieDictionary
) -> None:
    move = word_move(empty, 7, 5, "HELlO")
    assert validate_move(empty, move, dictionary, cfg, Rack.from_string("HEL?O")) == []
    v = validate_move(empty, move, dictionary, cfg, Rack.from_string("HELLO"))
    assert v == ["tile_not_in_rack:?"]


def test_through_existing_tiles_is_legal(
    cfg: BoardConfig, empty: Board, dictionary: TrieDictionary
) -> None:
    board = _hello(empty)
    move = word_move(board, 7, 5, ".....S")
    assert move.main_word == "HELLOS"
    assert validate_move(board, move, dictionary, cfg) == []
    down = word_move(board, 5, 9, "BO.K", across=False)  # BOOK through the O of HELLO
    assert down.main_word == "BOOK"
    assert validate_move(board, down, dictionary, cfg) == []
