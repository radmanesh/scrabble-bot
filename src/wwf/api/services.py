"""Application services behind the server-rendered pages.

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

from wwf.api.games import Game, GameStore, Picture, grid_to_rows
from wwf.api.schemas import (
    BOARD_SIZE,
    BoardGrid,
    DictionariesResponse,
    DictionaryInfoOut,
    MoveOut,
    PlacementOut,
    RecognitionResultOut,
    ScoreOut,
    SolveResponse,
    SolverInfoOut,
    SolverParamOut,
    SolversResponse,
    WordOut,
)
from wwf.core.board import Board
from wwf.core.config import BoardConfig
from wwf.core.dictionary import DictionaryRegistry
from wwf.core.rack import Rack
from wwf.core.scoring import ScoreBreakdown
from wwf.solvers.base import ParamError, ScoredMove, resolve_params
from wwf.solvers.registry import SolverRegistry

# A recognizer turns image bytes into a recognition result. Without one (no trained reader yet)
# the app still works: the board is typed in.
Recognizer = Callable[..., RecognitionResultOut]

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


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
    solve_timeout_s: float = 10.0
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    worker_threads: int = 4
    model_path: Path | None = None  # board reader; None disables reading pictures and retraining
    games_dir: Path | None = None  # where saved games live; None keeps games in memory only
    archive_dir: Path | None = None  # older labelled pictures, used for training only


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
        settings: Settings,
        recognizer: Recognizer | None = None,
        games: GameStore | None = None,
    ) -> None:
        self.cfg = cfg
        self.solvers = solvers
        self.dictionaries = dictionaries
        self.games = games if games is not None else GameStore(settings.games_dir)
        self.settings = settings
        self.recognizer = recognizer
        self._retrain_lock = threading.Lock()
        self._retraining = False
        self._retrain_again = False
        self.reader_trained_at: float | None = None

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

    # --- games ------------------------------------------------------------------------------

    def _resolve_options(
        self, options: tuple[str, dict[str, Any], str | None] | None
    ) -> tuple[str, dict[str, Any], str]:
        if options is not None:
            try:
                return self.check_options(*options)
            except ApiError:
                pass  # a stale cookie must not break starting a game
        return self.default_options()

    def new_game(
        self, options: tuple[str, dict[str, Any], str | None] | None = None, *, name: str = ""
    ) -> Game:
        sid, params, did = self._resolve_options(options)
        return self.games.create(sid, params, did, name=name)

    def game(self, game_id: str) -> Game:
        game = self.games.get(game_id)
        if game is None:
            raise ApiError(404, "game_not_found", {"game_id": game_id})
        return game

    def set_options(self, g: Game, solver_id: str, params: dict[str, Any], did: str | None) -> None:
        before = (g.solver_id, g.params, g.dictionary_id)
        g.solver_id, g.params, g.dictionary_id = self.check_options(solver_id, params, did)
        if (g.solver_id, g.params, g.dictionary_id) != before:
            g.last_result = None  # suggestions were made with the old settings

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

    def attach_picture(
        self, g: Game, data: bytes, content_type: str, source: str
    ) -> RecognitionResultOut:
        """Read a picture into the game's open turn and keep the picture with the game.

        Once the game has started, tiles already confirmed are kept and only the other squares
        are read from the picture: tiles never leave a WWF board. A picture that does not
        belong to the game is refused and changes nothing (and is not kept)."""
        previous = g.base_board if g.started and any(any(r) for r in g.base_board) else None
        hint = g.recognition.geometry.quad if g.recognition and g.recognition.geometry else None
        result = self.recognize(data, previous, hint)
        if previous is not None and not result.merged:
            result = merge_with_previous(result, previous)  # a reader that does not merge itself
        if result.rejected:
            g.notice = {"kind": "refused", "warnings": result.warnings, "match": result.match}
            return result
        pic = self.games.add_picture(g, data, content_type, source)
        pic.read_board = grid_to_rows(result.board)
        pic.read_rack = "".join(result.rack)
        if result.geometry is not None:
            pic.quad = [list(p) for p in result.geometry.quad]
        g.recognition = result
        g.board = [row[:] for row in result.board]
        if "rack_cut_off" not in result.warnings:
            g.rack = "".join(result.rack)  # an unreadable rack leaves yours alone
        g.last_result = None
        # whatever is new on the board now is the open turn, read from this picture
        if any(
            ch and not g.base_board[r][c]
            for r, row in enumerate(g.board)
            for c, ch in enumerate(row)
        ):
            g.open_source = "picture"
            g.open_player = None
        else:
            g.open_source = ""
        return result

    def picture(self, g: Game, picture_id: str) -> tuple[Picture, bytes] | None:
        pic = g.picture(picture_id)
        data = self.games.picture_bytes(g, pic) if pic else None
        return (pic, data) if pic is not None and data is not None else None

    # --- solving ----------------------------------------------------------------------------

    def board(self, grid: BoardGrid) -> Board:
        return Board.from_rows(self.cfg, grid)

    def solve(
        self,
        s: Game,
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

    # --- learning from confirmed pictures --------------------------------------------------

    def schedule_retrain(self) -> bool:
        """Rebuild the board reader from the labelled pictures in the background.

        Runs at most one retrain at a time; turns confirmed meanwhile trigger one more pass
        after it. Returns False when retraining is off (no model path)."""
        if self.settings.model_path is None:
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

        assert self.settings.model_path is not None
        while True:
            start = time.perf_counter()
            try:
                done = retrain(
                    self.settings.archive_dir,
                    self.settings.games_dir,
                    self.cfg,
                    self.settings.model_path,
                )
                if done:
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
