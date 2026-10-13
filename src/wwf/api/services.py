"""Application services shared by the JSON API and the server-rendered pages.

Everything here is synchronous; routes run the slow parts (recognition, solving) in a thread pool.
"""

from __future__ import annotations

import inspect
import io
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from wwf.api.fixtures import FixtureStore
from wwf.api.games import Game, GameStore
from wwf.api.schemas import (
    BOARD_SIZE,
    BoardGrid,
    DictionariesResponse,
    DictionaryInfoOut,
    MoveIn,
    MoveOut,
    PlacementOut,
    RecognitionResultOut,
    ScoreOut,
    SessionState,
    SolveResponse,
    SolverInfoOut,
    SolverParamOut,
    SolversResponse,
    ValidateResponse,
    WordOut,
)
from wwf.api.session import Session, SessionStore
from wwf.core.board import Board, Tile
from wwf.core.config import BoardConfig
from wwf.core.dictionary import DictionaryRegistry
from wwf.core.move import Direction, Placement, build_move
from wwf.core.rack import Rack
from wwf.core.scoring import ScoreBreakdown, score_move, validate_move
from wwf.solvers.base import ParamError, ScoredMove, resolve_params
from wwf.solvers.registry import SolverRegistry

# A recognizer turns image bytes into a recognition result. The vision package will provide one
# (milestones 3-4); until then the app runs without and the user types the board in.
Recognizer = Callable[[bytes], RecognitionResultOut]

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DRAFT_NOTE = "Read automatically by the board reader."


log = logging.getLogger("wwf.api")


class ApiError(Exception):
    """An error with an HTTP status, rendered as {error, detail, request_id}."""

    def __init__(self, status: int, error: str, detail: Any = None) -> None:
        super().__init__(error)
        self.status = status
        self.error = error
        self.detail = detail


@dataclass
class Settings:
    session_ttl_s: float = 30 * 60
    solve_timeout_s: float = 10.0
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    worker_threads: int = 4
    fixtures_dir: Path | None = None  # set: fixture collection mode (wwf-serve --collect-fixtures)
    model_path: Path | None = None  # board reader; None disables reading pictures and retraining
    games_dir: Path | None = None  # where saved games live; None keeps games in memory only


def unavailable_recognition(warning: str = "recognition_unavailable") -> RecognitionResultOut:
    return RecognitionResultOut(
        board=[[None] * BOARD_SIZE for _ in range(BOARD_SIZE)],
        rack=[],
        confidence=[[0.0] * BOARD_SIZE for _ in range(BOARD_SIZE)],
        needs_review=[[False] * BOARD_SIZE for _ in range(BOARD_SIZE)],
        warnings=[warning],
    )


class Services:
    def __init__(
        self,
        cfg: BoardConfig,
        solvers: SolverRegistry,
        dictionaries: DictionaryRegistry,
        sessions: SessionStore,
        settings: Settings,
        recognizer: Recognizer | None = None,
        games: GameStore | None = None,
    ) -> None:
        self.cfg = cfg
        self.solvers = solvers
        self.dictionaries = dictionaries
        self.sessions = sessions
        self.games = (
            games if games is not None else GameStore(settings.games_dir, settings.session_ttl_s)
        )
        self.settings = settings
        self.recognizer = recognizer
        self.fixtures = FixtureStore(settings.fixtures_dir) if settings.fixtures_dir else None
        self._retrain_lock = threading.Lock()
        self._retraining = False
        self._retrain_again = False
        self.reader_trained_at: float | None = None
        self.label_problems: dict[str, int] = {}  # picture -> squares its picture contradicts

    # --- options ----------------------------------------------------------------------------

    def default_options(self) -> tuple[str, dict[str, Any], str]:
        sid = self.solvers.default_id
        return sid, resolve_params(self.solvers.get(sid).params, {}), self.dictionaries.default_id

    def check_options(
        self, solver_id: str, params: dict[str, Any], dictionary_id: str | None
    ) -> tuple[str, dict[str, Any], str]:
        """Validated (solver_id, full params, dictionary_id). Raises ApiError(422)."""
        try:
            solver = self.solvers.get(solver_id)
        except KeyError:
            known = [i.id for i in self.solvers.infos() if i.enabled]
            raise ApiError(
                422, "unknown_solver", {"solver_id": solver_id, "known": known}
            ) from None
        try:
            full = resolve_params(solver.params, params)
        except ParamError as e:
            raise ApiError(422, "invalid_params", e.errors) from None
        did = dictionary_id or self.dictionaries.default_id
        try:
            self.dictionaries.info(did)
        except KeyError:
            known = [i.id for i in self.dictionaries.infos()]
            raise ApiError(
                422, "unknown_dictionary", {"dictionary_id": did, "known": known}
            ) from None
        return solver_id, full, did

    # --- sessions ---------------------------------------------------------------------------

    def _resolve_options(
        self, options: tuple[str, dict[str, Any], str | None] | None
    ) -> tuple[str, dict[str, Any], str]:
        if options is not None:
            try:
                return self.check_options(*options)
            except ApiError:
                pass  # a stale cookie must not break starting a game
        return self.default_options()

    def new_session(self, options: tuple[str, dict[str, Any], str | None] | None = None) -> Session:
        sid, params, did = self._resolve_options(options)
        return self.sessions.create(sid, params, did)

    def new_game(
        self,
        options: tuple[str, dict[str, Any], str | None] | None = None,
        *,
        ephemeral: bool = False,
        name: str = "",
    ) -> Game:
        sid, params, did = self._resolve_options(options)
        return self.games.create(sid, params, did, name=name, ephemeral=ephemeral)

    def session(self, session_id: str) -> Session:
        """A saved game or a session: both carry a board, a rack and options."""
        game = self.games.find(session_id)
        if game is not None:
            return game
        try:
            return self.sessions.get(session_id)
        except KeyError:
            raise ApiError(404, "session_not_found", {"session_id": session_id}) from None

    def game(self, game_id: str) -> Game:
        game = self.games.find(game_id)
        if game is None:
            raise ApiError(404, "game_not_found", {"game_id": game_id})
        return game

    def state(self, s: Session, request_id: str = "") -> SessionState:
        return SessionState(
            request_id=request_id,
            session_id=s.id,
            board=s.board,
            rack=s.rack,
            has_image=s.image is not None,
            recognition=s.recognition,
            corrected=s.corrected,
            solver_id=s.solver_id,
            params=s.params,
            dictionary_id=s.dictionary_id,
            has_result=s.last_result is not None,
            expires_in_s=-1 if isinstance(s, Game) else self.sessions.expires_in(s),
        )

    def set_board(self, s: Session, board: BoardGrid, rack: str) -> None:
        s.board = board
        s.rack = rack
        s.corrected = True
        s.last_result = None
        if isinstance(s, Game):
            s.follow_board()

    def set_options(
        self, s: Session, solver_id: str, params: dict[str, Any], did: str | None
    ) -> None:
        s.solver_id, s.params, s.dictionary_id = self.check_options(solver_id, params, did)

    # --- recognition ------------------------------------------------------------------------

    def check_image(self, data: bytes) -> str:
        """Content type of a decodable image. Raises ApiError 413/400."""
        if len(data) > self.settings.max_upload_bytes:
            raise ApiError(413, "image_too_large", {"max_bytes": self.settings.max_upload_bytes})
        if not data:
            raise ApiError(400, "undecodable_image", "empty upload")
        try:
            with Image.open(io.BytesIO(data)) as im:
                fmt = im.format or "octet-stream"
                im.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
            raise ApiError(400, "undecodable_image", str(e)) from None
        return Image.MIME.get(fmt, "application/octet-stream")

    def recognize(
        self,
        data: bytes,
        previous: BoardGrid | None = None,
        hint_quad: list[list[float]] | None = None,
    ) -> RecognitionResultOut:
        if self.recognizer is None:
            return unavailable_recognition()
        try:
            takes_context = "previous" in inspect.signature(self.recognizer).parameters
            if takes_context:
                return self.recognizer(data, previous=previous, hint_quad=hint_quad)
            return self.recognizer(data)
        except Exception as e:  # recognition failure is a warning, never a crash
            log.exception("reading a picture failed")
            return unavailable_recognition(f"recognition_failed:{type(e).__name__}")

    def attach_image(
        self, s: Session, data: bytes, content_type: str, previous: BoardGrid | None = None
    ) -> RecognitionResultOut:
        """Read a picture into the session: board, rack and the picture itself.

        With `previous` (the board on file), tiles already on it are kept and only its empty
        squares are read from the new picture: tiles never leave a WWF board. A picture that
        does not belong to that board is refused and changes nothing."""
        have_previous = previous is not None and any(any(row) for row in previous)
        hint = s.recognition.geometry.quad if s.recognition and s.recognition.geometry else None
        result = self.recognize(data, previous if have_previous else None, hint)
        if have_previous and not result.merged and previous is not None:
            result = merge_with_previous(result, previous)  # a reader that does not merge itself
        if result.rejected:
            s.notice = {"kind": "refused", "warnings": result.warnings, "match": result.match}
            return result
        s.image = data
        s.image_type = content_type
        s.recognition = result
        s.board = [row[:] for row in result.board]
        if "rack_cut_off" not in result.warnings:
            s.rack = "".join(result.rack)  # an unreadable rack leaves yours alone
        s.corrected = False
        s.last_result = None
        if isinstance(s, Game):
            s.follow_board()
        return result

    # --- solving ----------------------------------------------------------------------------

    def board(self, grid: BoardGrid) -> Board:
        return Board.from_rows(self.cfg, grid)

    def solve(
        self,
        s: Session,
        solver_id: str | None = None,
        params: dict[str, Any] | None = None,
        dictionary_id: str | None = None,
    ) -> SolveResponse:
        if solver_id is None or solver_id == s.solver_id:
            base_params = s.params if params is None else {**s.params, **params}
            sid = s.solver_id
        else:
            base_params, sid = params or {}, solver_id
        sid, full, did = self.check_options(sid, base_params, dictionary_id or s.dictionary_id)
        board = self.board(s.board)
        rack = Rack.from_string(s.rack)
        result = self.solvers.get(sid).solve(
            board, rack, self.dictionaries.get(did), self.cfg, **full
        )
        response = SolveResponse(
            solver_id=sid,
            params=full,
            dictionary_id=did,
            elapsed_ms=result.elapsed_ms,
            stats=result.stats,
            moves=[self.move_out(board, m) for m in result.moves],
        )
        s.last_result = response
        return response

    def move_out(self, board: Board, sm: ScoredMove) -> MoveOut:
        m = sm.move
        return MoveOut(
            word=m.main_word,
            direction=m.direction.value,
            start=m.start,
            placements=[
                PlacementOut(row=p.row, col=p.col, letter=p.tile.letter, is_blank=p.tile.is_blank)
                for p in m.placements
            ],
            score=score_out(sm.score),
            leave=sm.leave,
            board_after=board.with_move(m).to_grid(),
        )

    def validate(
        self, grid: BoardGrid, rack: str, move_in: MoveIn, dictionary_id: str | None
    ) -> ValidateResponse:
        did = dictionary_id or self.dictionaries.default_id
        try:
            dictionary = self.dictionaries.get(did)
        except KeyError:
            raise ApiError(422, "unknown_dictionary", {"dictionary_id": did}) from None
        board = self.board(grid)
        placements = [
            Placement(
                p.row,
                p.col,
                Tile(p.letter.upper(), p.letter.islower() if p.is_blank is None else p.is_blank),
            )
            for p in move_in.placements
        ]
        direction = Direction(move_in.direction) if move_in.direction else None
        move = build_move(board, placements, direction)
        violations = validate_move(board, move, dictionary, self.cfg, Rack.from_string(rack))
        score = None if violations else score_out(score_move(board, move, self.cfg))
        return ValidateResponse(violations=violations, score=score)

    # --- learning from labels (fixture collection mode) ------------------------------------

    def schedule_retrain(self) -> bool:
        """Rebuild the board reader from the labelled fixtures in the background.

        Runs at most one retrain at a time; labels saved meanwhile trigger one more pass after
        it. Returns False when retraining is off (no fixtures folder or no model path).
        """
        if self.fixtures is None or self.settings.model_path is None:
            return False
        with self._retrain_lock:
            if self._retraining:
                self._retrain_again = True
                return True
            self._retraining = True
        threading.Thread(target=self._retrain_loop, name="wwf-retrain", daemon=True).start()
        return True

    @property
    def retraining(self) -> bool:
        return self._retraining

    def _retrain_loop(self) -> None:
        from wwf.api.vision_adapter import load_recognizer, retrain

        assert self.fixtures is not None and self.settings.model_path is not None
        while True:
            start = time.perf_counter()
            try:
                issues = retrain(self.fixtures.directory, self.cfg, self.settings.model_path)
                if issues is not None:
                    self._note_label_problems(issues)
                    recognizer = load_recognizer(self.cfg, self.settings.model_path)
                    if recognizer is not None:
                        self.recognizer = recognizer
                        self.reader_trained_at = time.time()
                    log.info("retrained board reader in %.1fs", time.perf_counter() - start)
            except Exception:
                log.exception("retraining the board reader failed")
            with self._retrain_lock:
                if self._retrain_again:
                    self._retrain_again = False
                    continue
                self._retraining = False
                return

    def _note_label_problems(self, issues: list[tuple[str, int, int]]) -> None:
        counts: dict[str, int] = {}
        for name, _, _ in issues:
            counts[name] = counts.get(name, 0) + 1
        self.label_problems = counts

    def refresh_label_problems(self) -> None:
        """Find labels that contradict their pictures (run in the background at startup)."""
        if self.fixtures is None or self.recognizer is None:
            return
        try:
            from wwf.vision.train import label_issues

            found = label_issues(self.fixtures.directory, self.cfg)
            self._note_label_problems([(i.name, i.row, i.col) for i in found])
        except Exception:
            log.exception("checking labels against their pictures failed")

    def picture_problems(self, image: bytes, board: BoardGrid) -> list[tuple[int, int, str]]:
        """Squares of `board` that `image` contradicts, using the loaded reader."""
        if self.recognizer is None:
            return []
        from wwf.api.vision_adapter import check_picture

        try:
            return check_picture(self.recognizer, image, board, self.cfg)
        except Exception:
            log.exception("checking a label against its picture failed")
            return []

    def draft_unlabelled_fixtures(self) -> int:
        """Read every fixture picture that has no label yet into a draft. Returns how many."""
        if self.fixtures is None or self.recognizer is None:
            return 0
        drafted = 0
        for fixture in self.fixtures.list():
            if fixture.status != "none":
                continue
            try:
                result = self.recognizer(fixture.image.read_bytes())
            except Exception:
                log.exception("could not read fixture %s", fixture.name)
                continue
            if result.geometry is None:
                continue
            self.fixtures.write_draft(fixture.name, result.board, "".join(result.rack),
                                      result.geometry.quad, DRAFT_NOTE)  # fmt: skip
            drafted += 1
        if drafted:
            log.info("drafted labels for %d unlabelled fixtures", drafted)
        return drafted

    # --- catalogues -------------------------------------------------------------------------

    def solvers_response(self) -> SolversResponse:
        return SolversResponse(
            default_id=self.solvers.default_id,
            solvers=[
                SolverInfoOut(
                    id=i.id,
                    display_name=i.display_name,
                    description=i.description,
                    enabled=i.enabled,
                    default=i.default,
                    planned=i.planned,
                    params=[
                        SolverParamOut(
                            name=p.name,
                            label=p.label,
                            type=p.type,
                            default=p.default,
                            min=p.min,
                            max=p.max,
                            choices=list(p.choices),
                            help=p.help,
                        )
                        for p in i.params
                    ],
                )
                for i in self.solvers.infos()
            ],
        )

    def dictionaries_response(self) -> DictionariesResponse:
        return DictionariesResponse(
            default_id=self.dictionaries.default_id,
            dictionaries=[
                DictionaryInfoOut(
                    id=i.id,
                    label=i.label,
                    description=i.description,
                    default=i.id == self.dictionaries.default_id,
                    loaded=self.dictionaries.is_loaded(i.id),
                )
                for i in self.dictionaries.infos()
            ],
        )


REVIEW_BELOW = 0.85


def merge_with_previous(result: RecognitionResultOut, previous: BoardGrid) -> RecognitionResultOut:
    """The next turn's reading: previous tiles kept, empty squares taken from the new picture.

    A kept tile is flagged for review only when the new picture confidently shows something
    else there (a different letter, or nothing), which usually means the earlier board was wrong.
    """
    board = [row[:] for row in result.board]
    confidence = [row[:] for row in result.confidence]
    review = [row[:] for row in result.needs_review]
    read_board = result.geometry is not None  # the board was found in the new picture
    kept = new = conflicts = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            before = previous[r][c]
            if before is None:
                new += board[r][c] is not None
                continue
            kept += 1
            seen = result.board[r][c]
            sure = result.confidence[r][c] >= REVIEW_BELOW
            disagrees = read_board and sure and (seen is None or seen.upper() != before.upper())
            board[r][c] = before
            confidence[r][c] = 0.5 if disagrees else 1.0
            review[r][c] = disagrees
            conflicts += disagrees
    return result.model_copy(
        update={
            "board": board,
            "confidence": confidence,
            "needs_review": review,
            "kept_from_previous": kept,
            "new_tiles": new,
            "conflicts": conflicts,
        }
    )


def score_out(s: ScoreBreakdown) -> ScoreOut:
    return ScoreOut(
        total=s.total,
        bingo_bonus=s.bingo_bonus,
        words=[WordOut(word=w.word, points=w.points) for w in s.words],
    )
