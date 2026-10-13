"""Server-rendered pages: Upload, Review, Results, Options.

Every form is a plain POST that redirects, so the pages work without JavaScript; app.js adds the
review grid's keyboard flow and HTMX swaps the results preview in place.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.datastructures import FormData

from wwf.api.app import WEB_DIR, run_blocking
from wwf.api.app import services as get_services
from wwf.api.fixtures import FixtureError, FixtureStore, Source
from wwf.api.routes_recognize import read_upload
from wwf.api.routes_solve import solve_session
from wwf.api.schemas import BOARD_SIZE, MAX_RACK, BoardGrid, check_rack
from wwf.api.services import ApiError, Services
from wwf.api.session import Session
from wwf.core.config import project_root

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
    ctx.setdefault("fixtures_mode", svc.fixtures is not None)
    return templates.TemplateResponse(request, name, ctx, status_code=status)


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def _session_or_page(request: Request, session_id: str) -> Session | HTMLResponse:
    try:
        return get_services(request).session(session_id)
    except ApiError:
        message = "This session no longer exists. Sessions last 30 minutes."
        return _render(request, "error.html", 404, title="Session expired", message=message)


def _read_options_cookie(request: Request) -> dict[str, Any]:
    raw = request.cookies.get(OPTIONS_COOKIE)
    if not raw:
        return {}
    try:
        data = json.loads(base64.urlsafe_b64decode(raw.encode()))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _cookie_session_options(request: Request) -> tuple[str, dict[str, Any], str | None] | None:
    saved = _read_options_cookie(request)
    sid = saved.get("solver_id")
    if not isinstance(sid, str):
        return None
    params = saved.get("params", {}).get(sid, {})
    return sid, params if isinstance(params, dict) else {}, saved.get("dictionary_id")


def _new_session(request: Request) -> Session:
    return get_services(request).new_session(_cookie_session_options(request))


# --- upload ----------------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def upload_page(request: Request, saved: str | None = None) -> HTMLResponse:
    svc = get_services(request)
    ctx: dict[str, Any] = {"model_loaded": svc.recognizer is not None}
    if svc.fixtures is not None:
        ctx["fixture_count"] = len(svc.fixtures.list())
        ctx["fixture_dir"] = _display_path(svc.fixtures.directory)
        ctx["saved"] = saved
    return _render(request, "upload.html", **ctx)


def _source(value: str, content_type: str) -> Source:
    """photo or screenshot. Uploaded files ("auto") count as screenshots when they are PNG."""
    if value == "auto":
        return "screenshot" if content_type == "image/png" else "photo"
    return "screenshot" if value == "screenshot" else "photo"


async def _take_picture(
    request: Request, svc: Services, s: Session, data: bytes, source: str
) -> None:
    """Recognize the picture into the session; in fixture mode, also save it as a fixture."""
    content_type = svc.check_image(data)
    if svc.fixtures is not None:
        s.fixture_name = svc.fixtures.save_image(data, content_type, _source(source, content_type))
    await run_blocking(request, svc.attach_image, s, data, content_type)


@router.post("/upload")
async def upload(
    request: Request, image: UploadFile = File(...), source: str = Form("photo")
) -> Response:
    svc = get_services(request)
    try:
        data = await read_upload(svc, image)
        svc.check_image(data)
    except ApiError as e:
        return _render(request, "upload.html", e.status, error=_upload_error(e),
                       model_loaded=svc.recognizer is not None)  # fmt: skip
    s = _new_session(request)
    await _take_picture(request, svc, s, data, source)
    if svc.fixtures is not None:
        # collecting: go straight back to the camera for the next picture
        return _redirect(f"/?saved={s.fixture_name}")
    return _redirect(f"/review/{s.id}")


def _upload_error(e: ApiError) -> str:
    if e.error == "image_too_large":
        return "That image is over 10 MB. Try a screenshot instead of a photo, or resize it."
    return "That file isn't an image we can read. Upload a PNG or JPEG screenshot."


@router.post("/new")
def new_board(request: Request) -> Response:
    return _redirect(f"/review/{_new_session(request).id}")


# --- review ----------------------------------------------------------------------------------


def _review(request: Request, s: Session, status: int = 200, **ctx: Any) -> HTMLResponse:
    rec = s.recognition
    needs_review = rec.needs_review if rec else [[False] * BOARD_SIZE] * BOARD_SIZE
    rack = list(s.rack) + [""] * (MAX_RACK - len(s.rack))
    ctx.setdefault("board", s.board)
    ctx.setdefault("rack", rack)
    return _render(request, "review.html", status, session=s, recognition=rec,
                   needs_review=needs_review, **ctx)  # fmt: skip


@router.get("/review/{session_id}", response_class=HTMLResponse)
def review_page(request: Request, session_id: str) -> HTMLResponse:
    s = _session_or_page(request, session_id)
    return s if isinstance(s, HTMLResponse) else _review(request, s)


def _parse_board_form(form: FormData) -> tuple[BoardGrid, list[str], str, list[str]]:
    """Grid, rack cells, rack string and error messages from the review form."""
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


@router.post("/review/{session_id}")
async def review_submit(request: Request, session_id: str) -> Response:
    s = _session_or_page(request, session_id)
    if isinstance(s, HTMLResponse):
        return s
    svc = get_services(request)
    form = await request.form()
    board, rack_cells, rack, errors = _parse_board_form(form)
    if errors:
        return _review(request, s, 422, board=board, rack=rack_cells, errors=errors)
    svc.set_board(s, board, rack)
    if form.get("action") == "fixture" and svc.fixtures is not None and s.fixture_name:
        svc.fixtures.write_label(s.fixture_name, board, rack)
        return _redirect(f"/fixtures?labelled={s.fixture_name}")
    if form.get("action") == "save":
        return _redirect(f"/review/{s.id}?saved=1")
    if not rack:
        return _review(request, s, 422, errors=["Enter your rack tiles before solving."])
    try:
        await solve_session(request, svc, s)
    except ApiError as e:
        return _review(request, s, e.status, errors=[_solve_error(e)])
    return _redirect(f"/results/{s.id}")


def _solve_error(e: ApiError) -> str:
    if e.error == "solve_timeout":
        return "The solver took longer than 10 seconds. Try fewer blanks or another solver."
    return f"Could not solve: {e.error} {e.detail or ''}".strip()


@router.post("/review/{session_id}/image")
async def review_reupload(
    request: Request, session_id: str, image: UploadFile = File(...), source: str = Form("photo")
) -> Response:
    s = _session_or_page(request, session_id)
    if isinstance(s, HTMLResponse):
        return s
    svc = get_services(request)
    try:
        data = await read_upload(svc, image)
        await _take_picture(request, svc, s, data, source)
    except ApiError as e:
        return _review(request, s, e.status, errors=[_upload_error(e)])
    return _redirect(f"/review/{s.id}")


# --- results ---------------------------------------------------------------------------------


def _selected(n: int, count: int) -> int:
    return min(max(n, 0), max(count - 1, 0))


@router.get("/results/{session_id}", response_class=HTMLResponse)
def results_page(request: Request, session_id: str, move: int = 0) -> Response:
    s = _session_or_page(request, session_id)
    if isinstance(s, HTMLResponse):
        return s
    if s.last_result is None:
        return _redirect(f"/review/{s.id}")
    svc = get_services(request)
    result = s.last_result
    solver_name = svc.solvers.info(result.solver_id).display_name
    return _render(request, "results.html", session=s, result=result, solver_name=solver_name,
                   selected=_selected(move, len(result.moves)))  # fmt: skip


@router.get("/results/{session_id}/preview/{n}", response_class=HTMLResponse)
def results_preview(request: Request, session_id: str, n: int) -> Response:
    s = _session_or_page(request, session_id)
    if isinstance(s, HTMLResponse):
        return s
    if s.last_result is None or not s.last_result.moves:
        return HTMLResponse("")
    return _render(request, "_preview.html", session=s, result=s.last_result,
                   selected=_selected(n, len(s.last_result.moves)))  # fmt: skip


@router.post("/results/{session_id}/solve")
async def results_resolve(request: Request, session_id: str) -> Response:
    s = _session_or_page(request, session_id)
    if isinstance(s, HTMLResponse):
        return s
    try:
        await solve_session(request, get_services(request), s)
    except ApiError as e:
        return _review(request, s, e.status, errors=[_solve_error(e)])
    return _redirect(f"/results/{s.id}")


# --- options ---------------------------------------------------------------------------------


def _options_context(
    svc: Services, saved: dict[str, Any], session: Session | None
) -> dict[str, Any]:
    """Current choices: the session's if there is one, else the cookie's, else defaults."""
    solvers = svc.solvers_response().solvers
    values: dict[str, dict[str, Any]] = {
        i.id: {p.name: p.default for p in i.params} for i in solvers
    }
    for sid, params in (saved.get("params") or {}).items():
        if sid in values and isinstance(params, dict):
            values[sid].update({k: v for k, v in params.items() if k in values[sid]})
    chosen = saved.get("solver_id") or svc.solvers.default_id
    dictionary = saved.get("dictionary_id") or svc.dictionaries.default_id
    if session is not None:
        chosen, dictionary = session.solver_id, session.dictionary_id
        values.setdefault(chosen, {}).update(session.params)
    return {
        "solvers": solvers,
        "values": values,
        "chosen": chosen,
        "dictionaries": svc.dictionaries_response().dictionaries,
        "dictionary_id": dictionary,
    }


def _optional_session(svc: Services, session_id: str | None) -> Session | None:
    if not session_id:
        return None
    try:
        return svc.session(session_id)
    except ApiError:
        return None


@router.get("/options", response_class=HTMLResponse)
def options_page(request: Request, session: str | None = None, saved: int = 0) -> HTMLResponse:
    svc = get_services(request)
    s = _optional_session(svc, session)
    ctx = _options_context(svc, _read_options_cookie(request), s)
    return _render(request, "options.html", session=s, saved=bool(saved), errors=[], **ctx)


@router.post("/options")
async def options_submit(
    request: Request, solver_id: str = Form(...), dictionary_id: str = Form(""),
    session_id: str = Form(""),
) -> Response:  # fmt: skip
    svc = get_services(request)
    form = await request.form()
    s = _optional_session(svc, session_id)
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
        return _render(request, "options.html", 422, session=s, saved=False,
                       errors=errors, **ctx)  # fmt: skip
    all_params[sid] = full
    if s is not None:
        svc.set_options(s, sid, full, did)
    cookie = {"solver_id": sid, "params": all_params, "dictionary_id": did}
    target = f"/options?saved=1&session={s.id}" if s is not None else "/options?saved=1"
    response = _redirect(target)
    response.set_cookie(
        OPTIONS_COOKIE,
        base64.urlsafe_b64encode(json.dumps(cookie).encode()).decode(),
        max_age=365 * 24 * 3600,
        samesite="lax",
    )
    return response


# --- fixture collection (wwf-serve --collect-fixtures) -------------------------------------


def _display_path(path: Path) -> str:
    """The folder relative to the project when it is inside it, else absolute."""
    try:
        return str(path.resolve().relative_to(project_root().resolve()))
    except ValueError:
        return str(path)


def _fixtures(request: Request) -> FixtureStore:
    store = get_services(request).fixtures
    if store is None:
        raise ApiError(404, "fixture_mode_off", "start the server with --collect-fixtures")
    return store


@router.get("/fixtures", response_class=HTMLResponse)
def fixtures_page(request: Request, labelled: str | None = None) -> HTMLResponse:
    store = _fixtures(request)
    items = store.list()
    return _render(
        request,
        "fixtures.html",
        fixtures=items,
        directory=_display_path(store.directory),
        labelled=labelled,
        done=sum(f.labelled for f in items),
    )


@router.get("/fixtures/{name}/image", response_class=Response)
def fixture_image(request: Request, name: str) -> Response:
    try:
        fixture = _fixtures(request).get(name)
    except FixtureError as e:
        raise ApiError(404, "fixture_not_found", str(e)) from None
    return Response(fixture.image.read_bytes(), media_type=fixture.content_type)


@router.post("/fixtures/{name}/open")
def fixture_open(request: Request, name: str) -> Response:
    """Open a fixture in the review grid, with its existing label if it has one."""
    svc = get_services(request)
    store = _fixtures(request)
    try:
        fixture = store.get(name)
        label = store.label_grid(name)
    except FixtureError as e:
        raise ApiError(404, "fixture_not_found", str(e)) from None
    s = _new_session(request)
    data = fixture.image.read_bytes()
    svc.attach_image(s, data, fixture.content_type)
    s.fixture_name = name
    if label is not None:
        s.board, s.rack = label
    return _redirect(f"/review/{s.id}")
