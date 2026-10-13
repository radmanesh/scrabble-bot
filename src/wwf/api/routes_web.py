"""Server-rendered pages: Home and Options. The game page is in routes_game_pages.py.

Every form is a plain POST that redirects, so the pages work without JavaScript; app.js adds
the grid's keyboard flow and HTMX swaps the move preview in place.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.datastructures import FormData

from wwf.api.app import WEB_DIR, run_blocking
from wwf.api.app import services as get_services
from wwf.api.games import Game
from wwf.api.schemas import BOARD_SIZE, MAX_RACK, BoardGrid, check_rack
from wwf.api.services import ApiError, Services

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=WEB_DIR / "templates")
templates.env.filters["truthy"] = lambda v: v is True or str(v).lower() in ("true", "1", "on")

OPTIONS_COOKIE = "wwf_options"
BONUS_LABELS = {"T": "TW", "D": "DW", "t": "TL", "d": "DL", "*": "★", ".": ""}
BONUS_CLASSES = {"T": "tw", "D": "dw", "t": "tl", "d": "dl", "*": "ctr", ".": ""}


# --- helpers ---------------------------------------------------------------------------------


def _render(request: Request, name: str, status: int = 200, **ctx: Any) -> HTMLResponse:
    svc = get_services(request)
    layout = svc.cfg.layout
    ctx.setdefault("bonus", [[BONUS_CLASSES[ch] for ch in row] for row in layout])
    ctx.setdefault("bonus_label", [[BONUS_LABELS[ch] for ch in row] for row in layout])
    ctx.setdefault("size", BOARD_SIZE)
    return templates.TemplateResponse(request, name, ctx, status_code=status)


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def _read_options_cookie(request: Request) -> dict[str, Any]:
    raw = request.cookies.get(OPTIONS_COOKIE)
    if not raw:
        return {}
    try:
        data = json.loads(base64.urlsafe_b64decode(raw.encode()))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _cookie_options(request: Request) -> tuple[str, dict[str, Any], str | None] | None:
    saved = _read_options_cookie(request)
    sid = saved.get("solver_id")
    if not isinstance(sid, str):
        return None
    params = saved.get("params", {}).get(sid, {})
    return sid, params if isinstance(params, dict) else {}, saved.get("dictionary_id")


def _new_game(request: Request, *, name: str = "") -> Game:
    """A new game with the options remembered in the cookie."""
    return get_services(request).new_game(_cookie_options(request), name=name)


async def read_upload(svc: Services, image: UploadFile) -> bytes:
    """The upload's bytes, refusing anything over the size limit without reading it all."""
    limit = svc.settings.max_upload_bytes
    data = await image.read(limit + 1)
    if len(data) > limit:
        raise ApiError(413, "image_too_large", {"max_bytes": limit})
    return data


def upload_error(e: ApiError) -> str:
    if e.error == "image_too_large":
        return "That image is over 10 MB. Try a screenshot instead of a photo, or resize it."
    return "That file isn't an image we can read. Upload a PNG or JPEG screenshot."


def picture_source(value: str, content_type: str) -> str:
    """photo or upload: where a picture came from."""
    if value == "auto":
        return "upload"
    return "upload" if value == "upload" else "photo"


# --- home ------------------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    svc = get_services(request)
    return _render(
        request,
        "upload.html",
        model_loaded=svc.recognizer is not None,
        games=[_game_summary(g) for g in svc.games.saved()],
    )


def _preview_rows(g: Game) -> tuple[list[str], int, int]:
    """The saved board for a thumbnail: 15 strings of "." empty, "c" confirmed tile, "d" tile of
    the open turn that isn't confirmed yet. It is the board, not the latest picture, and never a
    suggestion: a suggestion is only ever drawn when you choose it on the game page."""
    rows, confirmed, draft = [], 0, 0
    for r in range(BOARD_SIZE):
        row = []
        for c in range(BOARD_SIZE):
            if g.board[r][c] is None:
                row.append(".")
            elif g.base_board[r][c] is not None:
                row.append("c")
                confirmed += 1
            else:
                row.append("d")
                draft += 1
        rows.append("".join(row))
    return rows, confirmed, draft


def _game_summary(g: Game) -> dict[str, Any]:
    scores = g.scores()
    rows, confirmed, draft = _preview_rows(g)
    if not g.started:
        status = "Setting up"
    else:
        status = "Your turn" if g.phase == "my_turn" else "Opponent's turn"
    tiles = f"{confirmed} confirmed tile{'s' if confirmed != 1 else ''}"
    if draft:
        tiles += f", {draft} not confirmed yet"
    return {
        "id": g.id,
        "name": g.name,
        "started": g.started,
        "phase": g.phase,
        "status": status,
        "me": scores["me"],
        "opponent": scores["opponent"],
        "turns": sum(1 for t in g.turns if t.kind != "start"),
        "pictures": len(g.pictures),
        "updated": g.updated_at.replace("T", " ")[:16],
        "preview": rows,
        "summary": f"Board with {tiles}." if confirmed + draft else "Empty board.",
    }


@router.post("/upload")
async def upload(
    request: Request, image: UploadFile = File(...), source: str = Form("photo")
) -> Response:
    """Start a game from a picture of it."""
    svc = get_services(request)
    try:
        data = await read_upload(svc, image)
        content_type = svc.check_image(data)
    except ApiError as e:
        return _render(request, "upload.html", e.status, error=upload_error(e),
                       model_loaded=svc.recognizer is not None,
                       games=[_game_summary(g) for g in svc.games.saved()])  # fmt: skip
    game = _new_game(request)
    await run_blocking(
        request, svc.attach_picture, game, data, content_type, picture_source(source, content_type)
    )
    svc.games.save(game)
    return _redirect(f"/game/{game.id}")


@router.post("/new")
def new_board(request: Request) -> Response:
    game = _new_game(request)
    return _redirect(f"/game/{game.id}")


# --- the board form --------------------------------------------------------------------------


def _parse_board_form(form: FormData) -> tuple[BoardGrid, list[str], str, list[str]]:
    """Grid, rack cells, rack string and error messages from the game page's form."""
    errors: list[str] = []
    board: BoardGrid = []
    for r in range(BOARD_SIZE):
        row: list[str | None] = []
        for c in range(BOARD_SIZE):
            v = str(form.get(f"c-{r}-{c}", "")).strip()
            if not v:
                row.append(None)
            elif len(v) == 1 and v.isascii() and v.isalpha():
                row.append(v)
            else:
                row.append(v[:1] or None)
                errors.append(f"Row {r + 1}, column {c + 1}: {v!r} is not a single letter.")
        board.append(row)
    rack_cells = [str(form.get(f"r-{i}", "")).strip()[:1] for i in range(MAX_RACK)]
    try:
        rack = check_rack("".join(rack_cells))
    except ValueError as e:
        rack = ""
        errors.append(f"Rack: {e}.")
    return board, rack_cells, rack, errors


# --- options ---------------------------------------------------------------------------------


def _options_context(svc: Services, saved: dict[str, Any], game: Game | None) -> dict[str, Any]:
    """Current choices: the game's if there is one, else the cookie's, else defaults."""
    solvers = svc.solvers_response().solvers
    values: dict[str, dict[str, Any]] = {
        i.id: {p.name: p.default for p in i.params} for i in solvers
    }
    for sid, params in (saved.get("params") or {}).items():
        if sid in values and isinstance(params, dict):
            values[sid].update({k: v for k, v in params.items() if k in values[sid]})
    chosen = saved.get("solver_id") or svc.solvers.default_id
    dictionary = saved.get("dictionary_id") or svc.dictionaries.default_id
    if game is not None:
        chosen, dictionary = game.solver_id, game.dictionary_id
        values.setdefault(chosen, {}).update(game.params)
    return {
        "solvers": solvers,
        "values": values,
        "chosen": chosen,
        "dictionaries": svc.dictionaries_response().dictionaries,
        "dictionary_id": dictionary,
    }


def _optional_game(svc: Services, game_id: str | None) -> Game | None:
    return svc.games.get(game_id) if game_id else None


@router.get("/options", response_class=HTMLResponse)
def options_page(request: Request, game: str | None = None, saved: int = 0) -> HTMLResponse:
    svc = get_services(request)
    g = _optional_game(svc, game)
    ctx = _options_context(svc, _read_options_cookie(request), g)
    return _render(request, "options.html", game=g, saved=bool(saved), errors=[], **ctx)


@router.post("/options")
async def options_submit(
    request: Request, solver_id: str = Form(...), dictionary_id: str = Form(""),
    game_id: str = Form(""),
) -> Response:  # fmt: skip
    svc = get_services(request)
    form = await request.form()
    g = _optional_game(svc, game_id)
    saved = _read_options_cookie(request)
    all_params: dict[str, dict[str, Any]] = dict(saved.get("params") or {})
    for key in form:
        if key.startswith("p."):
            _, sid, name = key.split(".", 2)
            # checkboxes send a hidden "false" then "true" when ticked: the last value wins
            all_params.setdefault(sid, {})[name] = form.getlist(key)[-1]
    try:
        sid, full, did = svc.check_options(
            solver_id, all_params.get(solver_id, {}), dictionary_id or None
        )
    except ApiError as e:
        ctx = _options_context(svc, {**saved, "solver_id": solver_id, "params": all_params,
                                     "dictionary_id": dictionary_id}, None)  # fmt: skip
        errors = e.detail if isinstance(e.detail, list) else [f"{e.error}: {e.detail}"]
        return _render(request, "options.html", 422, game=g, saved=False,
                       errors=errors, **ctx)  # fmt: skip
    all_params[sid] = full
    if g is not None:
        svc.set_options(g, sid, full, did)
        svc.games.save(g)
    cookie = {"solver_id": sid, "params": all_params, "dictionary_id": did}
    target = f"/options?saved=1&game={g.id}" if g is not None else "/options?saved=1"
    response = _redirect(target)
    response.set_cookie(
        OPTIONS_COOKIE,
        base64.urlsafe_b64encode(json.dumps(cookie).encode()).decode(),
        max_age=365 * 24 * 3600,
        samesite="lax",
    )
    return response
