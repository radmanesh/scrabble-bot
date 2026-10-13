"""Turn-by-turn play: one open turn at a time, locked once confirmed.

A game has a `base_board` (everything confirmed so far) and a `board` (what is on the page now).
The tiles on `board` that are not on `base_board` are the *open turn*, whichever way they got
there: a suggestion you played, a word you typed for either player, or a picture. The open turn
is editable until `confirm_turn` writes it to the history, which locks it. Undo reopens the last
confirmed turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wwf.api.games import Game, Picture, Player, Turn, grid_to_rows, rows_to_grid
from wwf.api.schemas import MAX_RACK, BoardGrid, check_rack
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
    """The open turn's tiles, understood as a move."""

    analysis: Analysis
    placements: list[Placement]
    positions: set[tuple[int, int]] = field(default_factory=set)


def safe_board(svc: Services, grid: BoardGrid) -> Board | None:
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


def open_positions(game: Game) -> set[tuple[int, int]]:
    """Squares that hold a tile now and are not locked: the open turn."""
    return {
        (r, c)
        for r, row in enumerate(game.board)
        for c, ch in enumerate(row)
        if ch and not game.base_board[r][c]
    }


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
    last = game.last_turn()
    if (
        game.phase != "my_turn"
        or last is None
        or last.kind == "start"
        or len(game.rack) >= MAX_RACK
        or open_positions(game)
    ):
        return None
    bag = bag_for(svc, game)
    if bag is not None and bag.in_bag == 0:
        return None  # nothing left to draw
    return game.rack


# --- editing the open turn --------------------------------------------------------------------


def apply_edit(game: Game, board: BoardGrid, rack: str) -> int:
    """Take the board and rack from the page. Squares of confirmed turns are read-only: whatever
    the page says for them is ignored. Returns how many such squares differed."""
    ignored = 0
    merged: BoardGrid = []
    for r, row in enumerate(board):
        out: list[str | None] = []
        for c, ch in enumerate(row):
            locked = game.base_board[r][c]
            if locked:
                ignored += ch != locked
                out.append(locked)
            else:
                out.append(ch)
        merged.append(out)
    if merged != game.board:
        game.last_result = None
    game.board = merged
    game.rack = rack
    _track_source(game, "grid")
    return ignored


def _track_source(game: Game, source: str) -> None:
    """Keep `open_source` in step with whether there is an open turn."""
    if not open_positions(game):
        game.open_source = ""
        game.open_player = None
    elif not game.open_source:
        game.open_source = source


def _rack_after(rack: str, placements: list[Placement]) -> str:
    left = list(rack)
    for p in placements:
        tile = "?" if p.tile.is_blank else p.tile.letter.upper()
        if tile in left:
            left.remove(tile)
    return "".join(left)


def _require_closed(game: Game) -> None:
    if open_positions(game):
        raise MoveRejected(["turn_open"])


def apply_solver_move(game: Game, n: int) -> None:
    """Put move `n` of the last solve on the board as the open turn; your rack becomes the leave."""
    result = game.last_result
    if result is None or not 0 <= n < len(result.moves):
        raise ApiError(422, "no_such_move", {"n": n})
    _require_closed(game)
    mv = result.moves[n]
    game.board = [row[:] for row in mv.board_after]
    game.rack = mv.leave
    game.open_source = "suggestion"
    game.open_player = "me"
    game.last_result = None
    game.recognition = None


def apply_typed_move(
    svc: Services, game: Game, player: Player, word: str, row: int, col: int, direction: str
) -> None:
    """Put a word (one-based row and column of its first letter) on the board as the open turn."""
    _require_closed(game)
    board = safe_board(svc, game.board)
    if board is None:
        raise MoveRejected(["unreadable_board"])
    placements, problems = placements_for_word(board, word, row - 1, col - 1, Direction(direction))
    if problems:
        raise MoveRejected(problems)
    game.board = board.with_move(build_move(board, placements)).to_grid()
    if player == "me":
        game.rack = _rack_after(game.rack, placements)
    game.open_source = "typed"
    game.open_player = player
    game.last_result = None
    game.recognition = None


def cancel_turn(game: Game) -> None:
    """Throw away the open turn: the board and rack go back to how the turn began."""
    game.board = [row[:] for row in game.base_board]
    game.rack = game.base_rack
    game.open_source = ""
    game.open_player = None
    game.last_result = None
    game.recognition = None


# --- confirming turns -------------------------------------------------------------------------


def _lock(game: Game, turn: Turn) -> Turn:
    """Close the open turn: it joins the history, and the picture it was read from gets the
    confirmed board as its label.

    Only the latest picture is labelled. The confirmed board is what the user checked against that
    one; an earlier picture of the same turn may well show the board before some of those tiles
    were played, and labelling it with tiles it doesn't show teaches the reader that empty squares
    are tiles."""
    open_pics = game.open_pictures()
    if game.open_source == "picture" and open_pics:
        pic = open_pics[-1]
        turn.picture = pic.id
        pic.label_board = grid_to_rows(game.board)
        pic.label_rack = game.rack
        pic.label_turn = turn.n
    game.turns.append(turn)
    game.base_board = [row[:] for row in game.board]
    game.base_rack = game.rack
    game.open_source = ""
    game.open_player = None
    game.last_result = None
    game.recognition = None
    return turn


def _snapshot(game: Game) -> dict[str, Any]:
    return {
        "before_base": grid_to_rows(game.base_board),
        "before_base_rack": game.base_rack,
        "before_phase": game.phase,
    }


def start_game(game: Game, phase: str) -> Turn:
    """Confirm the position on the board as where tracking begins. No legality check: it is
    whatever the phone shows."""
    if game.started:
        raise ApiError(422, "already_started")
    turn = Turn(
        n=1,
        player="me",
        kind="start",
        tiles=sum(1 for row in game.board for ch in row if ch),
        source=game.open_source or "grid",
        **_snapshot(game),
    )
    game.started = True
    game.phase = "opponent_turn" if phase == "opponent_turn" else "my_turn"
    game.turns.clear()
    return _lock(game, turn)


def confirm_turn(svc: Services, game: Game, player: Player, force: bool = False) -> Turn:
    """Log the open turn as `player`'s move and lock it."""
    pending = pending_move(svc, game)
    if pending is None:
        raise MoveRejected(["no_new_tiles"])
    analysis = pending.analysis
    if analysis.score is None or (analysis.violations and not (force and analysis.forceable)):
        raise MoveRejected(analysis.violations, forceable=analysis.forceable)
    move, score = analysis.move, analysis.score
    turn = Turn(
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
        source=game.open_source or "grid",
        forced=bool(analysis.violations),
        **_snapshot(game),
    )
    game.phase = "opponent_turn" if player == "me" else "my_turn"
    return _lock(game, turn)


def accept_board(game: Game) -> Turn:
    """Take the open tiles as they are without scoring them (a picture that caught up several
    turns, say). The history shows it as a board update; the phase stays."""
    tiles = len(open_positions(game))
    if not tiles:
        raise MoveRejected(["no_new_tiles"])
    turn = Turn(
        n=len(game.turns) + 1,
        player=game.player_to_move,
        kind="start",
        tiles=tiles,
        source=game.open_source or "grid",
        **_snapshot(game),
    )
    return _lock(game, turn)


def pass_turn(game: Game, player: Player) -> Turn:
    if open_positions(game):
        raise MoveRejected(["turn_open"])
    turn = Turn(
        n=len(game.turns) + 1, player=player, kind="pass", source="typed", **_snapshot(game)
    )
    game.phase = "opponent_turn" if player == "me" else "my_turn"
    return _lock(game, turn)


def undo(game: Game) -> Turn:
    """Reopen the last confirmed turn: its tiles become the open turn again, editable."""
    if not game.turns:
        raise ApiError(422, "nothing_to_undo")
    turn = game.turns.pop()
    game.base_board = rows_to_grid(turn.before_base)
    game.base_rack = turn.before_base_rack
    game.phase = "opponent_turn" if turn.before_phase == "opponent_turn" else "my_turn"
    game.board = game.board if turn.kind != "pass" else [row[:] for row in game.base_board]
    game.open_source = turn.source if open_positions(game) else ""
    game.open_player = turn.player if turn.kind == "play" and game.open_source else None
    for pic in game.pictures:  # the picture's label belonged to that turn; it can change now
        if pic.label_turn == turn.n:
            pic.label_board = pic.label_rack = pic.label_turn = None
    if not game.turns:
        game.started = False
    game.last_result = None
    game.recognition = None
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
    game.base_rack = combined if not open_positions(game) else game.base_rack
    game.last_result = None
    return combined


def labelled_pictures(game: Game, turn: Turn) -> list[Picture]:
    """The pictures that this confirmed turn gave a label to."""
    return [p for p in game.pictures if p.label_turn == turn.n]


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
        "no_new_tiles": "No new tiles on the board.",
        "turn_open": "Confirm or cancel the open turn first.",
        "unreadable_board": "Fix the unreadable tiles ('?') on the board first.",
        "no_tiles_placed": "No new tiles.",
    }.get(name, name)
