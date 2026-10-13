"""The game page and its actions.

One page per game (`/game/{id}`) shows whatever the game needs next: set up the starting
position, then each turn's open tiles (editable until confirmed), the suggestions, the word
entry and the pictures. Every action is a plain POST that redirects back, so it all works
without JavaScript.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from wwf.api import gameplay
from wwf.api.app import run_blocking
from wwf.api.app import services as get_services
from wwf.api.gameplay import MoveRejected, Pending
from wwf.api.games import Game
from wwf.api.routes_web import (
    _parse_board_form,
    _redirect,
    _render,
    picture_source,
    read_upload,
    upload_error,
)
from wwf.api.schemas import BOARD_SIZE, MAX_RACK
from wwf.api.services import ApiError, Services
from wwf.vision.perspective import warp_board_png

router = APIRouter(include_in_schema=False)
RECENT_PICTURES = 4


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


ACTIONS = ("find", "mine", "opponent")


def _action_for(request: Request, game: Game) -> str:
    """Which of the workspace's actions is showing: the one asked for with `?do=`, otherwise
    the one the turn state points at. Only your own turn can find a best move."""
    asked = request.query_params.get("do", "")
    if asked == "find" and game.phase != "my_turn":
        asked = ""
    if asked in ACTIONS:
        return asked
    return "find" if game.phase == "my_turn" else "opponent"


def _parse_word_form(
    word: str, row: str, col: str, direction: str
) -> tuple[tuple[int, int] | None, str | None]:
    """The one-based start as numbers, or a sentence saying what is wrong with it."""
    try:
        r, c = int(row), int(col)
        if direction not in ("across", "down") or not (
            1 <= r <= BOARD_SIZE and 1 <= c <= BOARD_SIZE
        ):
            raise ValueError
    except ValueError:
        return None, f"Row and column must be numbers from 1 to {BOARD_SIZE}."
    return (r, c), None


def _preview_payload(
    svc: Services, game: Game, word: str, start: tuple[int, int], direction: str
) -> dict[str, Any]:
    """A typed word as the composer shows it. Reads only: the game is not changed."""
    label = svc.dictionaries.info(game.dictionary_id).label
    try:
        p = gameplay.preview_typed_move(svc, game, word, start[0], start[1], direction)
    except MoveRejected as e:
        problems = [gameplay.explain(x, label) for x in e.problems]
        return {
            "ok": False, "cells": [], "problems": problems, "legal": False, "forceable": False,
            "points": None, "words": [], "bingo": 0, "tiles": 0, "start": [start[0], start[1]],
        }  # fmt: skip
    codes = p.entry_problems or p.violations
    return {
        "ok": p.legal or p.forceable,
        "cells": p.cells,
        "problems": [gameplay.explain(x, label) for x in codes],
        "legal": p.legal,
        "forceable": p.forceable,
        "points": p.score.total if p.score else None,
        "words": [{"word": w.word, "points": w.points} for w in p.score.words] if p.score else [],
        "bingo": p.score.bingo_bonus if p.score else 0,
        "tiles": p.new_tiles,
        "start": [start[0], start[1]],
    }


SHOWN_AT_FIRST = 5


def _suggestions_view(game: Game, selected: int) -> list[dict[str, Any]]:
    """The solver's moves as the page needs them: the tiles each one would put on the board, so
    choosing one can draw it without another request."""
    result = game.last_result
    if result is None:
        return []
    out = []
    for n, m in enumerate(result.moves):
        dr, dc = (0, 1) if m.direction == "across" else (1, 0)
        new = {(p.row, p.col) for p in m.placements}
        cells = []
        for i in range(len(m.word)):
            r, c = m.start[0] + i * dr, m.start[1] + i * dc
            v = m.board_after[r][c] or ""
            cells.append({"r": r, "c": c, "letter": v.upper(), "blank": v.islower(),
                          "new": (r, c) in new, "conflict": False})  # fmt: skip
        out.append({
            "n": n, "word": m.word, "score": m.score.total, "direction": m.direction,
            "start": [m.start[0] + 1, m.start[1] + 1], "leave": m.leave,
            "words": [{"word": w.word, "points": w.points} for w in m.score.words],
            "bingo": m.score.bingo_bonus, "tiles": len(m.placements), "cells": cells,
        })  # fmt: skip
    return out


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
        "points": score.total if score else None,
        "words": [(w.word, w.points) for w in score.words] if score else [],
        "bingo": score.bingo_bonus if score else 0,
        "legal": a.legal,
        "forceable": a.forceable,
        "problems": [gameplay.explain(v, label) for v in a.violations],
    }


def _page(request: Request, game: Game, status: int = 200, **ctx: Any) -> HTMLResponse:
    svc = get_services(request)
    pending = gameplay.pending_move(svc, game) if game.started else None
    open_tiles = gameplay.open_positions(game)
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
    ctx.setdefault("composer_preview", None)
    ctx.setdefault("do_action", _action_for(request, game))
    suggestions = _suggestions_view(game, selected) if result else []
    ctx.setdefault("suggestions", suggestions)
    ctx.setdefault("show_all", request.query_params.get("all") == "1")
    ctx.setdefault("first_n", SHOWN_AT_FIRST)
    if ctx["do_action"] == "find" and suggestions and ctx["composer_preview"] is None:
        ctx["composer_preview"] = {"cells": suggestions[selected]["cells"]}  # drawn on the board
    latest = game.latest_picture
    pictures = list(reversed(game.pictures[-RECENT_PICTURES:]))
    return _render(
        request,
        "game.html",
        status,
        game=game,
        recognition=rec,
        needs_review=rec.needs_review if rec else [[False] * BOARD_SIZE] * BOARD_SIZE,
        pending=_pending_view(svc, game, pending),
        open_tiles=open_tiles,
        locked={(r, c) for r, row in enumerate(game.base_board) for c, ch in enumerate(row) if ch},
        scores=game.scores(),
        bag=bag,
        suspects=suspects,
        recent={(p["row"], p["col"]) for p in last.placements} if last else set(),
        drew=gameplay.drew_prompt(svc, game),
        result=result,
        solver_name=svc.solvers.info(result.solver_id).display_name if result else "",
        selected=selected,
        latest_picture=latest,
        pictures=pictures,
        picture_count=len(game.pictures),
        has_quad=bool(latest and latest.quad),
        turns=list(reversed(game.turns)),
        open_player=game.open_player or game.player_to_move,
        **ctx,
    )


def _finish(request: Request, game: Game, labelled: bool = False) -> Response:
    _save(request, game)
    if labelled:
        get_services(request).schedule_retrain()  # learn from the confirmed picture
    return _redirect(f"/game/{game.id}")


# --- the page --------------------------------------------------------------------------------


@router.get("/game/{game_id}", response_class=HTMLResponse)
def game_page(request: Request, game_id: str, saved: int = 0) -> HTMLResponse:
    game = _game_or_page(request, game_id)
    return game if isinstance(game, HTMLResponse) else _page(request, game, saved=bool(saved))


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


# --- the board form: save, solve, confirm ------------------------------------------------------


async def solve_game(request: Request, svc: Services, game: Game) -> None:
    timeout = svc.settings.solve_timeout_s
    try:
        await run_blocking(request, svc.solve, game, timeout=timeout)
    except TimeoutError:
        # Solvers are not interruptible, so there is nothing partial to return yet.
        raise ApiError(504, "solve_timeout", {"timeout_s": timeout}) from None


@router.post("/game/{game_id}/board")
async def board_submit(request: Request, game_id: str) -> Response:
    """Everything the board form can do: the squares and rack on the page are taken first
    (squares of confirmed turns are ignored), then the action runs."""
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    svc = get_services(request)
    form = await request.form()
    action = str(form.get("action", "save"))
    if action == "cancel":
        gameplay.cancel_turn(game)
        return _finish(request, game)
    board, rack_cells, rack, errors = _parse_board_form(form)
    if errors:
        return _page(request, game, 422, board=board, rack=rack_cells, errors=errors)
    gameplay.apply_edit(game, board, rack)
    label = svc.dictionaries.info(game.dictionary_id).label
    labelled = False
    try:
        if action == "solve":
            if not rack:
                raise MoveRejected(["empty_rack"])
            await solve_game(request, svc, game)
        elif action in ("start_me", "start_opponent"):
            turn = gameplay.start_game(game, "my_turn" if action == "start_me" else "opponent_turn")
            labelled = bool(gameplay.labelled_pictures(game, turn))
        elif action in ("confirm_me", "confirm_opponent"):
            who = "me" if action == "confirm_me" else "opponent"
            turn = gameplay.confirm_turn(svc, game, who, force=bool(form.get("force")))
            labelled = bool(gameplay.labelled_pictures(game, turn))
        elif action == "accept":
            turn = gameplay.accept_board(game)
            labelled = bool(gameplay.labelled_pictures(game, turn))
    except MoveRejected as e:
        _save(request, game)
        problems = [
            "Enter your rack tiles before solving." if p == "empty_rack"
            else gameplay.explain(p, label) for p in e.problems
        ]  # fmt: skip
        return _page(request, game, 422, errors=problems)
    except ApiError as e:
        _save(request, game)
        return _page(request, game, e.status, errors=[_solve_error(e)])
    _save(request, game)
    if labelled:
        svc.schedule_retrain()
    return _redirect(f"/game/{game.id}" + ("?saved=1" if action == "save" else ""))


@router.post("/game/{game_id}/solve")
async def solve_again(request: Request, game_id: str) -> Response:
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    if not game.rack:
        return _page(request, game, 422, errors=["Enter your rack tiles before solving."])
    try:
        await solve_game(request, get_services(request), game)
    except ApiError as e:
        return _page(request, game, e.status, errors=[_solve_error(e)])
    return _finish(request, game)


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
            await solve_game(request, get_services(request), game)
        except ApiError as e:
            _save(request, game)
            return _page(request, game, e.status, errors=[_solve_error(e)])
    return _finish(request, game)


# --- putting a move on the board -------------------------------------------------------------


@router.post("/game/{game_id}/play/{n}")
def play(request: Request, game_id: str, n: int) -> Response:
    """Put a suggestion on the board as the open turn. It is confirmed separately."""
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    try:
        gameplay.apply_solver_move(game, n)
    except MoveRejected as e:
        return _page(request, game, 422, errors=[gameplay.explain(p) for p in e.problems])
    except ApiError:
        return _page(
            request, game, 422, errors=["That move isn't available any more: solve again."]
        )
    return _finish(request, game)


@router.post("/game/{game_id}/record/{n}")
def record_suggestion(request: Request, game_id: str, n: int, turn: str = Form("")) -> Response:
    """Record a suggestion as your move: put it on the board and confirm it, as one step."""
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    svc = get_services(request)
    try:
        gameplay.record_solver_move(
            svc, game, n, expected_turn=int(turn) if turn.isdigit() else None
        )
    except MoveRejected as e:
        return _page(request, game, 422, errors=[gameplay.explain(p) for p in e.problems])
    except ApiError:
        return _page(
            request, game, 422, errors=["That move isn't available any more: find moves again."]
        )
    return _finish(request, game)


@router.post("/game/{game_id}/word")
def typed_word(
    request: Request,
    game_id: str,
    word: str = Form(""),
    row: str = Form(""),
    col: str = Form(""),
    direction: str = Form("across"),
    player: str = Form("opponent"),
) -> Response:
    """Put a typed word on the board as the open turn, for you or the opponent."""
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
        gameplay.apply_typed_move(
            svc, game, "me" if player == "me" else "opponent", word, r, c, direction
        )
    except MoveRejected as e:
        return _page(request, game, 422, form_values=values,
                     errors=[gameplay.explain(p, label) for p in e.problems])  # fmt: skip
    return _finish(request, game)


@router.get("/game/{game_id}/compose/preview.json")
def compose_preview_json(
    request: Request,
    game_id: str,
    word: str = "",
    row: str = "",
    col: str = "",
    direction: str = "across",
) -> Response:
    """What a typed word would do, for the composer's live preview. Changes nothing."""
    game = get_services(request).games.get(game_id)
    if game is None:
        return JSONResponse({"ok": False, "problems": ["That game doesn't exist."]}, 404)
    start, problem = _parse_word_form(word, row, col, direction)
    if not word.strip() or start is None:
        problems = [problem] if problem and (row or col) else []
        return JSONResponse({"ok": False, "incomplete": True, "cells": [], "problems": problems})
    return JSONResponse(_preview_payload(get_services(request), game, word, start, direction))


@router.post("/game/{game_id}/compose/preview")
def compose_preview(
    request: Request,
    game_id: str,
    word: str = Form(""),
    row: str = Form(""),
    col: str = Form(""),
    direction: str = Form("across"),
    player: str = Form("opponent"),
) -> Response:
    """The same preview without JavaScript: the page comes back with the word drawn on the
    board. Still nothing is changed."""
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    values = {"word": word, "row": row, "col": col, "direction": direction}
    do = "mine" if player == "me" else "opponent"
    start, problem = _parse_word_form(word, row, col, direction)
    if start is None or not word.strip():
        errors = [problem or "Type the word first."]
        return _page(request, game, 422, form_values=values, errors=errors, do_action=do)
    payload = _preview_payload(get_services(request), game, word, start, direction)
    return _page(request, game, form_values=values, composer_preview=payload, do_action=do)


@router.post("/game/{game_id}/compose/record")
def compose_record(
    request: Request,
    game_id: str,
    word: str = Form(""),
    row: str = Form(""),
    col: str = Form(""),
    direction: str = Form("across"),
    player: str = Form("opponent"),
    force: str = Form(""),
    turn: str = Form(""),
) -> Response:
    """Place a typed word and confirm it as one move for the player shown. All or nothing."""
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    svc = get_services(request)
    who = "me" if player == "me" else "opponent"
    values = {"word": word, "row": row, "col": col, "direction": direction}
    do = "mine" if who == "me" else "opponent"
    start, problem = _parse_word_form(word, row, col, direction)
    if start is None:
        return _page(request, game, 422, form_values=values, errors=[problem or ""], do_action=do)
    label = svc.dictionaries.info(game.dictionary_id).label
    try:
        gameplay.record_typed_move(
            svc, game, who, word, start[0], start[1], direction,
            force=bool(force), expected_turn=int(turn) if turn.isdigit() else None,
        )  # fmt: skip
    except MoveRejected as e:
        preview = _preview_payload(svc, game, word, start, direction)
        errors = [gameplay.explain(p, label) for p in e.problems]
        return _page(request, game, 422, form_values=values, composer_preview=preview,
                     errors=errors, do_action=do)  # fmt: skip
    return _finish(request, game)


@router.post("/game/{game_id}/pass")
def pass_turn(request: Request, game_id: str, who: str = Form("opponent")) -> Response:
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    try:
        gameplay.pass_turn(game, "me" if who == "me" else "opponent")
    except MoveRejected as e:
        return _page(request, game, 422, errors=[gameplay.explain(p) for p in e.problems])
    return _finish(request, game)


@router.post("/game/{game_id}/undo")
def undo(request: Request, game_id: str) -> Response:
    """Reopen the last confirmed turn."""
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    try:
        gameplay.undo(game)
    except ApiError:
        return _page(request, game, 422, errors=["Nothing to undo."])
    return _finish(request, game)


# --- pictures --------------------------------------------------------------------------------


@router.post("/game/{game_id}/image")
async def next_picture(
    request: Request, game_id: str, image: UploadFile = File(...), source: str = Form("photo")
) -> Response:
    """Read a picture into the open turn. Confirmed tiles are kept; the rest is read."""
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    svc = get_services(request)
    try:
        data = await read_upload(svc, image)
        content_type = svc.check_image(data)
        await run_blocking(
            request, svc.attach_picture, game, data, content_type,
            picture_source(source, content_type),
        )  # fmt: skip
    except ApiError as e:
        return _page(request, game, e.status, errors=[upload_error(e)])
    return _finish(request, game)


@router.get("/game/{game_id}/picture/{picture_id}", response_class=Response)
def picture_file(request: Request, game_id: str, picture_id: str) -> Response:
    svc = get_services(request)
    game = svc.games.get(game_id)
    found = svc.picture(game, picture_id) if game is not None else None
    if found is None:
        raise ApiError(404, "picture_not_found", {"picture": picture_id})
    pic, data = found
    return Response(data, media_type=pic.content_type, headers={"Cache-Control": "private"})


@router.get("/game/{game_id}/picture/{picture_id}/board.png", response_class=Response)
def picture_board(request: Request, game_id: str, picture_id: str) -> Response:
    """The board in the picture, straightened with the corners the reader found."""
    svc = get_services(request)
    game = svc.games.get(game_id)
    found = svc.picture(game, picture_id) if game is not None else None
    if found is None or not found[0].quad:
        raise ApiError(404, "picture_not_found", {"picture": picture_id})
    png = warp_board_png(found[1], found[0].quad)
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=60"})


# --- housekeeping ----------------------------------------------------------------------------


@router.post("/game/{game_id}/rename")
def rename(request: Request, game_id: str, name: str = Form("")) -> Response:
    game = _game_or_page(request, game_id)
    if isinstance(game, HTMLResponse):
        return game
    game.name = name.strip()[:60] or game.name
    return _finish(request, game)


@router.post("/game/{game_id}/delete")
def delete(request: Request, game_id: str) -> Response:
    get_services(request).games.delete(game_id)
    return _redirect("/")
