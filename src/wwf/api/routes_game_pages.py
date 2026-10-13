"""The game page and its actions: solve, play, enter the opponent's move, undo, next picture.

One page per game (`/game/{id}`) shows whatever the current turn needs. Every action is a plain
POST that redirects back, so it all works without JavaScript.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from wwf.api import gameplay
from wwf.api.app import services as get_services
from wwf.api.gameplay import MoveRejected, Pending
from wwf.api.games import Game
from wwf.api.routes_recognize import read_upload
from wwf.api.routes_solve import solve_session
from wwf.api.routes_web import (
    _board_image_url,
    _parse_board_form,
    _redirect,
    _render,
    _take_picture,
    _upload_error,
)
from wwf.api.schemas import BOARD_SIZE, MAX_RACK
from wwf.api.services import ApiError, Services

router = APIRouter(include_in_schema=False)


# --- helpers ---------------------------------------------------------------------------------


def _game_or_page(request: Request, game_id: str) -> Game | HTMLResponse:
    try:
        return get_services(request).game(game_id)
    except ApiError:
        return _render(
            request, "error.html", 404, title="Game not found",
            message="That game doesn't exist (it may have been deleted).",
        )  # fmt: skip


def _save(request: Request, game: Game) -> None:
    get_services(request).games.save(game)


def _selected(n: int, count: int) -> int:
    return min(max(n, 0), max(count - 1, 0))


def _solve_error(e: ApiError) -> str:
    if e.error == "solve_timeout":
        return "The solver took longer than 10 seconds. Try fewer blanks or another solver."
    return f"Could not solve: {e.error} {e.detail or ''}".strip()


def _pending_view(svc: Services, game: Game, pending: Pending | None) -> dict[str, Any] | None:
    if pending is None:
        return None
    a = pending.analysis
    label = svc.dictionaries.info(game.dictionary_id).label
    score = a.score
    return {
        "word": a.move.main_word,
        "direction": a.move.direction.value,
        "start": (a.move.start[0] + 1, a.move.start[1] + 1),
        "tiles": len(pending.placements),
        "changed": pending.changed,
        "points": score.total if score else None,
        "words": [(w.word, w.points) for w in score.words] if score else [],
        "bingo": score.bingo_bonus if score else 0,
        "legal": a.legal,
        "forceable": a.forceable,
        "problems": [gameplay.explain(v, label) for v in a.violations],
    }


def _page(request: Request, game: Game, status: int = 200, **ctx: Any) -> HTMLResponse:
    svc = get_services(request)
    pending = gameplay.pending_move(svc, game)
    bag = gameplay.bag_for(svc, game)
    rec = game.recognition
    last = game.last_turn()
    result = game.last_result
    selected = (
        _selected(int(request.query_params.get("move", 0) or 0), len(result.moves)) if result else 0
    )
    rack = list(game.rack) + [""] * (MAX_RACK - len(game.rack))
    suspects = ""
    if bag is not None and bag.over:
        suspects = ", ".join(
            f"{letter} ({n} more than the bag holds)" for letter, n in sorted(bag.over.items())
        )
    unsure_rack = list(rec.rack_needs_review) if rec else []
    ctx.setdefault("rack_review", unsure_rack + [False] * (MAX_RACK - len(unsure_rack)))
    ctx.setdefault("notice", game.notice)
    game.notice = None  # shown once
    ctx.setdefault("board", game.board)
    ctx.setdefault("rack", rack)
    ctx.setdefault("errors", [])
    ctx.setdefault("form_values", {"word": "", "row": "", "col": "", "direction": "across"})
    ctx.setdefault("force_offer", False)
    return _render(
        request,
        "game.html",
        status,
        game=game,
        session=game,  # the shared top bar links Options to the current game
        recognition=rec,
        needs_review=rec.needs_review if rec else [[False] * BOARD_SIZE] * BOARD_SIZE,
        pending=_pending_view(svc, game, pending),
        pending_positions=pending.positions if pending else set(),
        scores=game.scores(),
        bag=bag,
        suspects=suspects,
        recent={(p["row"], p["col"]) for p in last.placements} if last else set(),
        drew=gameplay.drew_prompt(svc, game),
        result=result,
        solver_name=svc.solvers.info(result.solver_id).display_name if result else "",
        selected=selected,
        step=3 if game.phase == "opponent_turn" else (2 if result and result.moves else 1),
        labelling=svc.fixtures is not None and game.ephemeral and bool(game.fixture_name),
        problem_squares={(r, c): says for r, c, says in game.label_problems},
        board_image_url=_board_image_url(svc, game),
        turns=list(reversed(game.turns)),
        **ctx,
    )


# --- the page --------------------------------------------------------------------------------


@router.get("/game/{game_id}", response_class=HTMLResponse)
def game_page(request: Request, game_id: str, saved: int = 0) -> HTMLResponse:
    game = _game_or_page(request, game_id)
    return game if isinstance(game, HTMLResponse) else _page(request, game, saved=bool(saved))


@router.get("/review/{game_id}")
@router.get("/results/{game_id}")
def old_links(game_id: str) -> RedirectResponse:
    """The earlier Review and Results pages are now one game page."""
    return _redirect(f"/game/{game_id}")


@router.get("/game/{game_id}/preview/{n}", response_class=HTMLResponse)
def preview(request: Request, game_id: str, n: int) -> Response:
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    if game.last_result is None or not game.last_result.moves:
        return HTMLResponse("")
    result = game.last_result
    return _render(
        request, "_preview.html", game=game, result=result,
        selected=_selected(n, len(result.moves)), oob=True,
    )  # fmt: skip


# --- the board and solving -------------------------------------------------------------------


@router.post("/game/{game_id}/board")
async def board_submit(request: Request, game_id: str) -> Response:
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    svc = get_services(request)
    form = await request.form()
    board, rack_cells, rack, errors = _parse_board_form(form)
    if errors:
        return _page(request, game, 422, board=board, rack=rack_cells, errors=errors)
    svc.set_board(game, board, rack)
    action = form.get("action")
    if svc.fixtures is not None and game.ephemeral and game.fixture_name:
        # labelling a fixture: every save writes the label file, so nothing typed is lost
        svc.fixtures.write_label(game.fixture_name, board, rack)
        svc.schedule_retrain()  # learn from the corrected label
        if action != "solve":
            return _redirect(f"/fixtures?labelled={game.fixture_name}")
    if action == "solve":
        if not rack:
            _save(request, game)
            return _page(request, game, 422, errors=["Enter your rack tiles before solving."])
        try:
            await solve_session(request, svc, game)
        except ApiError as e:
            _save(request, game)
            return _page(request, game, e.status, errors=[_solve_error(e)])
    _save(request, game)
    return _redirect(f"/game/{game.id}" + ("" if action == "solve" else "?saved=1"))


@router.post("/game/{game_id}/solve")
async def solve_again(request: Request, game_id: str) -> Response:
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    if not game.rack:
        return _page(request, game, 422, errors=["Enter your rack tiles before solving."])
    try:
        await solve_session(request, get_services(request), game)
    except ApiError as e:
        return _page(request, game, e.status, errors=[_solve_error(e)])
    _save(request, game)
    return _redirect(f"/game/{game.id}")


@router.post("/game/{game_id}/draw")
async def draw(
    request: Request, game_id: str, letters: str = Form(""), action: str = Form("add")
) -> Response:
    """Add the tiles you drew to the rack, optionally going straight on to solving."""
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    try:
        gameplay.add_drawn_tiles(game, letters)
    except ValueError as e:
        return _page(request, game, 422, errors=[str(e)])
    if action == "solve":
        try:
            await solve_session(request, get_services(request), game)
        except ApiError as e:
            _save(request, game)
            return _page(request, game, e.status, errors=[_solve_error(e)])
    _save(request, game)
    return _redirect(f"/game/{game.id}")


# --- turns -----------------------------------------------------------------------------------


@router.post("/game/{game_id}/play/{n}")
def play(request: Request, game_id: str, n: int) -> Response:
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    try:
        gameplay.play_solver_move(game, n)
    except ApiError:
        return _page(
            request, game, 422, errors=["That move isn't available any more: solve again."]
        )
    _save(request, game)
    return _redirect(f"/game/{game.id}")


@router.post("/game/{game_id}/opponent")
def opponent_move(
    request: Request,
    game_id: str,
    word: str = Form(""),
    row: str = Form(""),
    col: str = Form(""),
    direction: str = Form("across"),
    force: str = Form(""),
) -> Response:
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    svc = get_services(request)
    values = {"word": word, "row": row, "col": col, "direction": direction}
    try:
        r, c = int(row), int(col)
        if direction not in ("across", "down") or not (
            1 <= r <= BOARD_SIZE and 1 <= c <= BOARD_SIZE
        ):
            raise ValueError
    except ValueError:
        msg = f"Row and column must be numbers from 1 to {BOARD_SIZE}."
        return _page(request, game, 422, form_values=values, errors=[msg])
    label = svc.dictionaries.info(game.dictionary_id).label
    try:
        gameplay.play_typed_move(svc, game, "opponent", word, r, c, direction, force=bool(force))
    except MoveRejected as e:
        return _page(request, game, 422, form_values=values, force_offer=e.forceable,
                     errors=[gameplay.explain(p, label) for p in e.problems])  # fmt: skip
    _save(request, game)
    return _redirect(f"/game/{game.id}")


@router.post("/game/{game_id}/pending")
def pending_action(
    request: Request, game_id: str, action: str = Form(...), force: str = Form("")
) -> Response:
    """Do something with the new tiles found on the board: log them, or just accept them."""
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    svc = get_services(request)
    if action == "accept":
        gameplay.accept_board(game)
    elif action in ("confirm_opponent", "confirm_me"):
        player = "opponent" if action == "confirm_opponent" else "me"
        label = svc.dictionaries.info(game.dictionary_id).label
        try:
            gameplay.confirm_pending(svc, game, player, force=bool(force))
        except MoveRejected as e:
            return _page(
                request, game, 422, errors=[gameplay.explain(p, label) for p in e.problems]
            )
    else:
        return _page(request, game, 422, errors=["Unknown action."])
    _save(request, game)
    return _redirect(f"/game/{game.id}")


@router.post("/game/{game_id}/pass")
def pass_turn(request: Request, game_id: str, who: str = Form("opponent")) -> Response:
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    gameplay.pass_turn(game, "me" if who == "me" else "opponent")
    _save(request, game)
    return _redirect(f"/game/{game.id}")


@router.post("/game/{game_id}/undo")
def undo(request: Request, game_id: str) -> Response:
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    try:
        gameplay.undo(game)
    except ApiError:
        return _page(request, game, 422, errors=["Nothing to undo."])
    _save(request, game)
    return _redirect(f"/game/{game.id}")


# --- the next picture ------------------------------------------------------------------------


@router.post("/game/{game_id}/image")
async def next_picture(
    request: Request, game_id: str, image: UploadFile = File(...), source: str = Form("photo")
) -> Response:
    """Read a new picture. Tiles already on the board are kept; only new ones are read."""
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    svc = get_services(request)
    # start from the board as it is on the page now (the browser sends the grid along, so
    # unsaved corrections count), else as last saved
    previous = game.board
    form = await request.form()
    if "c-0-0" in form:
        grid, _, _, errors = _parse_board_form(form)
        if not errors:
            previous = grid
    try:
        data = await read_upload(svc, image)
        await _take_picture(request, svc, game, data, source, previous)
    except ApiError as e:
        return _page(request, game, e.status, errors=[_upload_error(e)])
    _save(request, game)
    return _redirect(f"/game/{game.id}")


# --- housekeeping ----------------------------------------------------------------------------


@router.post("/game/{game_id}/rename")
def rename(request: Request, game_id: str, name: str = Form("")) -> Response:
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    game.name = name.strip()[:60] or game.name
    _save(request, game)
    return _redirect(f"/game/{game.id}")


@router.post("/game/{game_id}/delete")
def delete(request: Request, game_id: str) -> Response:
    get_services(request).games.delete(game_id)
    return _redirect("/")
