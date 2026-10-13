"""Turn-by-turn play: logging both players' moves, spotting the opponent's, and undoing.

The model is simple. A game has a `base_board` (the board when the pending turn began) and a
`board` (what is on the page now). Tiles on `board` that are not on `base_board` are the pending
move: an opponent's move read from a photo, typed into the grid, or entered as a word. Logging
a turn writes it to the history with a snapshot of the state before it, which is what undo
restores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wwf.api.fixtures import grid_to_rows, rows_to_grid
from wwf.api.games import Game, Player, Turn
from wwf.api.schemas import MAX_RACK, check_rack
from wwf.api.services import ApiError, Services
from wwf.core.bag import BagReport, bag_report
from wwf.core.board import Board
from wwf.core.diff import diff_boards
from wwf.core.entry import placements_for_word
from wwf.core.move import Direction, Move, Placement, build_move
from wwf.core.rack import Rack
from wwf.core.scoring import ScoreBreakdown, score_move, validate_move

DICTIONARY_PREFIX = "word_not_in_dictionary:"


class MoveRejected(Exception):
    """A move cannot be logged. `forceable` is true when the only problem is the word list."""

    def __init__(self, problems: list[str], forceable: bool = False) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems
        self.forceable = forceable


@dataclass
class Analysis:
    move: Move
    violations: list[str]
    score: ScoreBreakdown | None  # None when the tiles do not even form a line

    @property
    def legal(self) -> bool:
        return not self.violations

    @property
    def forceable(self) -> bool:
        """Only the word list disagrees: the game itself may well have accepted the move."""
        return bool(self.violations) and all(
            v.startswith(DICTIONARY_PREFIX) for v in self.violations
        )


@dataclass
class Pending:
    """New tiles since the turn began, understood as a move."""

    analysis: Analysis
    placements: list[Placement]
    changed: int = 0  # existing tiles whose letter differs from the base board
    positions: set[tuple[int, int]] = field(default_factory=set)


def safe_board(svc: Services, grid: list[list[str | None]]) -> Board | None:
    """The grid as a Board, or None while it holds something unreadable (a "?" tile)."""
    try:
        return svc.board(grid)
    except ValueError:
        return None


def analyse(
    svc: Services, board: Board, placements: list[Placement], dictionary_id: str
) -> Analysis:
    dictionary = svc.dictionaries.get(dictionary_id)
    move = build_move(board, placements)
    violations = validate_move(board, move, dictionary, svc.cfg)
    structural = [v for v in violations if not v.startswith(DICTIONARY_PREFIX)]
    score = None if structural else score_move(board, move, svc.cfg)
    return Analysis(move, violations, score)


def pending_move(svc: Services, game: Game) -> Pending | None:
    """The move implied by the tiles added to the base board, if there are any."""
    base, now = safe_board(svc, game.base_board), safe_board(svc, game.board)
    if base is None or now is None:
        return None
    diff = diff_boards(base, now)
    if not diff.placements:
        return None
    placements = list(diff.placements)
    return Pending(
        analysis=analyse(svc, base, placements, game.dictionary_id),
        placements=placements,
        changed=len(diff.changed),
        positions={(p.row, p.col) for p in placements},
    )


def bag_for(svc: Services, game: Game) -> BagReport | None:
    board = safe_board(svc, game.board)
    if board is None:
        return None
    try:
        rack = Rack.from_string(game.rack)
    except ValueError:
        return None
    return bag_report(svc.cfg, board, rack)


def drew_prompt(svc: Services, game: Game) -> str | None:
    """The tiles kept from the last turn, when it is your turn and the rack is not full."""
    if game.phase != "my_turn" or not game.turns or len(game.rack) >= MAX_RACK:
        return None
    bag = bag_for(svc, game)
    if bag is not None and bag.in_bag == 0:
        return None  # nothing left to draw
    return game.rack


# --- logging turns ----------------------------------------------------------------------------


def _snapshot(game: Game, *, board: list[list[str | None]] | None = None) -> dict[str, Any]:
    return {
        "before_board": grid_to_rows(board if board is not None else game.board),
        "before_rack": game.rack,
        "before_base": grid_to_rows(game.base_board),
        "before_phase": game.phase,
    }


def _finish(game: Game, turn: Turn, player: Player) -> Turn:
    game.turns.append(turn)
    game.phase = "opponent_turn" if player == "me" else "my_turn"
    game.base_board = [row[:] for row in game.board]
    game.last_result = None
    return turn


def _play_turn(
    game: Game, player: Player, analysis: Analysis, source: str, forced: bool, snap: dict[str, Any]
) -> Turn:
    move, score = analysis.move, analysis.score
    assert score is not None
    return Turn(
        n=len(game.turns) + 1,
        player=player,
        kind="play",
        word=move.main_word,
        direction=move.direction.value,
        start=move.start,
        placements=[
            {"row": p.row, "col": p.col, "letter": p.tile.letter, "is_blank": p.tile.is_blank}
            for p in move.placements
        ],
        words=[{"word": w.word, "points": w.points} for w in score.words],
        points=score.total,
        bingo=score.bingo_bonus,
        source=source,
        forced=forced,
        **snap,
    )


def play_solver_move(game: Game, n: int) -> Turn:
    """Make move `n` of the last solve: the board, your rack and the turn all move on."""
    result = game.last_result
    if result is None or not 0 <= n < len(result.moves):
        raise ApiError(422, "no_such_move", {"n": n})
    mv = result.moves[n]
    turn = Turn(
        n=len(game.turns) + 1,
        player="me",
        kind="play",
        word=mv.word,
        direction=mv.direction,
        start=mv.start,
        placements=[p.model_dump() for p in mv.placements],
        words=[w.model_dump() for w in mv.score.words],
        points=mv.score.total,
        bingo=mv.score.bingo_bonus,
        source="solver",
        **_snapshot(game),
    )
    game.board = [row[:] for row in mv.board_after]
    game.rack = mv.leave
    game.clear_picture()
    return _finish(game, turn, "me")


def play_typed_move(
    svc: Services,
    game: Game,
    player: Player,
    word: str,
    row: int,
    col: int,
    direction: str,
    force: bool = False,
) -> Turn:
    """Log a move given as a word and where it starts (one-based row and column)."""
    board = safe_board(svc, game.board)
    if board is None:
        raise MoveRejected(["unreadable_board"])
    placements, problems = placements_for_word(board, word, row - 1, col - 1, Direction(direction))
    if problems:
        raise MoveRejected(problems)
    analysis = analyse(svc, board, placements, game.dictionary_id)
    if analysis.score is None or (analysis.violations and not (force and analysis.forceable)):
        raise MoveRejected(analysis.violations, forceable=analysis.forceable)
    snap = _snapshot(game)
    game.board = board.with_move(analysis.move).to_grid()
    game.clear_picture()
    turn = _play_turn(game, player, analysis, "typed", forced=bool(analysis.violations), snap=snap)
    return _finish(game, turn, player)


def confirm_pending(svc: Services, game: Game, player: Player, force: bool = False) -> Turn:
    """Log the tiles added since the turn began as `player`'s move."""
    pending = pending_move(svc, game)
    if pending is None:
        raise MoveRejected(["no_new_tiles"])
    analysis = pending.analysis
    if analysis.score is None or (analysis.violations and not (force and analysis.forceable)):
        raise MoveRejected(analysis.violations, forceable=analysis.forceable)
    # the board already holds the tiles, so undo returns to the base board
    snap = _snapshot(game, board=game.base_board)
    turn = _play_turn(
        game, player, analysis, "photo" if game.image else "grid", bool(analysis.violations), snap
    )
    return _finish(game, turn, player)


def accept_board(game: Game) -> None:
    """Take the board as it is, without logging a move: it becomes the new base."""
    game.base_board = [row[:] for row in game.board]


def pass_turn(game: Game, player: Player) -> Turn:
    turn = Turn(
        n=len(game.turns) + 1, player=player, kind="pass", source="typed", **_snapshot(game)
    )
    return _finish(game, turn, player)


def undo(game: Game) -> Turn:
    if not game.turns:
        raise ApiError(422, "nothing_to_undo")
    turn = game.turns.pop()
    game.board = rows_to_grid(turn.before_board)
    game.rack = turn.before_rack
    game.base_board = rows_to_grid(turn.before_base)
    game.phase = "opponent_turn" if turn.before_phase == "opponent_turn" else "my_turn"
    game.last_result = None
    game.clear_picture()
    return turn


def add_drawn_tiles(game: Game, letters: str) -> str:
    """Add the tiles you drew to the rack. Raises ValueError for a bad letter or a full rack."""
    drawn = check_rack(letters)
    if not drawn:
        raise ValueError("Type the letters you drew.")
    combined = game.rack + drawn
    if len(combined) > MAX_RACK:
        raise ValueError(f"Your rack would have {len(combined)} tiles; the most is {MAX_RACK}.")
    game.rack = combined
    game.last_result = None
    return combined


# --- plain-language messages -----------------------------------------------------------------


def explain(problem: str, dictionary_label: str = "the word list") -> str:
    """A violation or entry problem as a sentence."""
    name, _, detail = problem.partition(":")
    if name == "word_not_in_dictionary":
        return f"{detail} isn't in {dictionary_label}."
    if name == "letter_mismatch":
        where, _, letter = detail.rpartition(":")
        row, _, col = where.partition(",")
        return f"The board already has {letter} at row {row}, column {col}."
    return {
        "not_in_line": "The new tiles aren't in one row or one column.",
        "gap_in_word": "There's a gap between the new tiles.",
        "first_move_not_on_center": "The first word must cover the centre square.",
        "no_contact_with_existing_tiles": "The word doesn't touch any tile already on the board.",
        "cell_occupied": "A tile is already on one of those squares.",
        "out_of_bounds": "The word runs off the board.",
        "empty_word": "Type the word first.",
        "bad_letter": "Use letters only (lowercase marks a blank).",
        "no_new_tiles": "That word is already on the board: no new tiles.",
        "unreadable_board": "Fix the unreadable tiles ('?') on the board first.",
        "no_tiles_placed": "No new tiles.",
    }.get(name, name)
