"""FastAPI application factory and the `wwf-serve` entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from wwf.api.schemas import HealthResponse
from wwf.api.services import ApiError, Recognizer, Services, Settings
from wwf.api.session import SessionStore
from wwf.core.config import load_board_config
from wwf.core.dictionary import DictionaryRegistry, default_dictionary_registry
from wwf.solvers.registry import SolverRegistry, load_registry

log = logging.getLogger("wwf.api")
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
API_PREFIX = "/api/v1"


def services(request: Request) -> Services:
    svc: Services = request.app.state.services
    return svc


def request_id(request: Request) -> str:
    rid: str = getattr(request.state, "request_id", "")
    return rid


async def run_blocking[T](
    request: Request, fn: Callable[..., T], *args: Any, timeout: float | None = None
) -> T:
    """Run `fn` in the app's worker pool so the event loop stays free."""
    loop = asyncio.get_running_loop()
    executor = getattr(request.app.state, "executor", None)  # None: the loop's default pool
    future = loop.run_in_executor(executor, lambda: fn(*args))
    return await asyncio.wait_for(future, timeout)


def _error(status: int, error: str, detail: Any, rid: str) -> JSONResponse:
    return JSONResponse(
        {"error": error, "detail": detail, "request_id": rid},
        status_code=status,
        headers={"X-Request-ID": rid},
    )


def create_app(
    settings: Settings | None = None,
    *,
    solvers: SolverRegistry | None = None,
    dictionaries: DictionaryRegistry | None = None,
    sessions: SessionStore | None = None,
    recognizer: Recognizer | None = None,
) -> FastAPI:
    settings = settings or Settings()
    svc = Services(
        cfg=load_board_config(),
        solvers=load_registry() if solvers is None else solvers,
        dictionaries=default_dictionary_registry() if dictionaries is None else dictionaries,
        sessions=SessionStore(settings.session_ttl_s) if sessions is None else sessions,
        settings=settings,
        recognizer=recognizer,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc.dictionaries.get()  # load (or read the cache of) the default dictionary up front
        executor = ThreadPoolExecutor(settings.worker_threads, thread_name_prefix="wwf")
        app.state.executor = executor

        async def sweep() -> None:
            while True:
                await asyncio.sleep(60)
                removed = svc.sessions.sweep()
                if removed:
                    log.info("swept %d expired sessions", removed)

        task = asyncio.create_task(sweep())
        try:
            yield
        finally:
            task.cancel()
            app.state.executor = None
            executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="WWF Solver",
        version="0.1.0",
        description="Reconstructs a Words With Friends board from a screenshot and ranks moves.",
        lifespan=lifespan,
    )
    app.state.services = svc
    app.state.executor = None

    @app.middleware("http")
    async def add_request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log.info(
            "%s %s %s %d %.0fms",
            rid,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        return response

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.error, exc.detail, request_id(request))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", "")} for e in exc.errors()]
        return _error(422, "validation_error", detail, request_id(request))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        error = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, error, exc.detail, request_id(request))

    @app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["meta"])
    def health(request: Request) -> HealthResponse:
        did = svc.dictionaries.default_id
        return HealthResponse(
            request_id=request_id(request),
            status="ok",
            dictionary_id=did,
            dictionary_words=len(svc.dictionaries.get(did)),
            model_loaded=svc.recognizer is not None,
            sessions=len(svc.sessions),
        )

    from wwf.api import routes_options, routes_recognize, routes_solve, routes_web

    app.include_router(routes_recognize.router, prefix=API_PREFIX)
    app.include_router(routes_solve.router, prefix=API_PREFIX)
    app.include_router(routes_options.router, prefix=API_PREFIX)
    app.include_router(routes_web.router)
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the WWF Solver web app.")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from a phone")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(
        "wwf.api.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
