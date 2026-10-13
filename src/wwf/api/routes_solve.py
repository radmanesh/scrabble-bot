"""Session state, board corrections, solving and move validation."""

from __future__ import annotations

from fastapi import APIRouter, Request

from wwf.api.app import request_id, run_blocking, services
from wwf.api.schemas import (
    BoardIn,
    SessionState,
    SolveIn,
    SolveResponse,
    ValidateIn,
    ValidateResponse,
)
from wwf.api.services import ApiError, Services
from wwf.api.session import Session

router = APIRouter(tags=["solving"])


@router.get("/session/{session_id}", response_model=SessionState)
def get_session(request: Request, session_id: str) -> SessionState:
    svc = services(request)
    return svc.state(svc.session(session_id), request_id(request))


@router.put("/session/{session_id}/board", response_model=SessionState)
def put_board(request: Request, session_id: str, body: BoardIn) -> SessionState:
    """Replace the board and rack wholesale with the user's corrections."""
    svc = services(request)
    s = svc.session(session_id)
    svc.set_board(s, body.board, body.rack)
    return svc.state(s, request_id(request))


async def solve_session(
    request: Request, svc: Services, s: Session, body: SolveIn | None = None
) -> SolveResponse:
    body = body or SolveIn()
    timeout = svc.settings.solve_timeout_s
    try:
        return await run_blocking(
            request, svc.solve, s, body.solver_id, body.params, body.dictionary_id, timeout=timeout
        )
    except TimeoutError:
        # Solvers are not interruptible, so there is nothing partial to return yet.
        raise ApiError(
            504, "solve_timeout", {"partial": True, "moves": [], "timeout_s": timeout}
        ) from None


@router.post("/session/{session_id}/solve", response_model=SolveResponse)
async def solve(request: Request, session_id: str, body: SolveIn | None = None) -> SolveResponse:
    """Solve the session's board. Omitted solver, params or dictionary fall back to the
    session's options; params given here override the session's for this call only."""
    svc = services(request)
    response = await solve_session(request, svc, svc.session(session_id), body)
    return response.model_copy(update={"request_id": request_id(request)})


@router.post("/validate", response_model=ValidateResponse)
def validate(request: Request, body: ValidateIn) -> ValidateResponse:
    """Check a hand-entered move. Returns the named rule violations and, if legal, its score."""
    result = services(request).validate(body.board, body.rack, body.move, body.dictionary_id)
    return result.model_copy(update={"request_id": request_id(request)})
