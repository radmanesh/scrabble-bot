"""Application services shared by the JSON API and the server-rendered pages.

Everything here is synchronous; routes run the slow parts (recognition, solving) in a thread pool.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image, UnidentifiedImageError

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
    ) -> None:
        self.cfg = cfg
        self.solvers = solvers
        self.dictionaries = dictionaries
        self.sessions = sessions
        self.settings = settings
        self.recognizer = recognizer

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

    def new_session(self, options: tuple[str, dict[str, Any], str | None] | None = None) -> Session:
        if options is not None:
            try:
                sid, params, did = self.check_options(*options)
            except ApiError:
                sid, params, did = self.default_options()
        else:
            sid, params, did = self.default_options()
        return self.sessions.create(sid, params, did)

    def session(self, session_id: str) -> Session:
        try:
            return self.sessions.get(session_id)
        except KeyError:
            raise ApiError(404, "session_not_found", {"session_id": session_id}) from None

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
            expires_in_s=self.sessions.expires_in(s),
        )

    def set_board(self, s: Session, board: BoardGrid, rack: str) -> None:
        s.board = board
        s.rack = rack
        s.corrected = True
        s.last_result = None

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

    def recognize(self, data: bytes) -> RecognitionResultOut:
        if self.recognizer is None:
            return unavailable_recognition()
        try:
            return self.recognizer(data)
        except Exception as e:  # recognition failure is a warning, never a crash
            return unavailable_recognition(f"recognition_failed:{type(e).__name__}")

    def attach_image(self, s: Session, data: bytes, content_type: str) -> RecognitionResultOut:
        """Store the image on the session and replace its board and rack with what was read."""
        result = self.recognize(data)
        s.image = data
        s.image_type = content_type
        s.recognition = result
        s.board = [row[:] for row in result.board]
        s.rack = "".join(result.rack)
        s.corrected = False
        s.last_result = None
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


def score_out(s: ScoreBreakdown) -> ScoreOut:
    return ScoreOut(
        total=s.total,
        bingo_bonus=s.bingo_bonus,
        words=[WordOut(word=w.word, points=w.points) for w in s.words],
    )
