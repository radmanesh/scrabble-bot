"""Solver and dictionary catalogues, and per-session options."""

from __future__ import annotations

from fastapi import APIRouter, Request

from wwf.api.app import request_id, services
from wwf.api.schemas import DictionariesResponse, OptionsIn, SessionState, SolversResponse

router = APIRouter(tags=["options"])


@router.get("/solvers", response_model=SolversResponse)
def list_solvers(request: Request) -> SolversResponse:
    """Every solver in config/solvers.json with its parameter schema; planned ones are disabled."""
    return (
        services(request).solvers_response().model_copy(update={"request_id": request_id(request)})
    )


@router.get("/dictionaries", response_model=DictionariesResponse)
def list_dictionaries(request: Request) -> DictionariesResponse:
    return (
        services(request)
        .dictionaries_response()
        .model_copy(update={"request_id": request_id(request)})
    )


@router.put("/session/{session_id}/options", response_model=SessionState)
def put_options(request: Request, session_id: str, body: OptionsIn) -> SessionState:
    """Choose the session's solver, its parameters and the dictionary. 422 on invalid values."""
    svc = services(request)
    s = svc.session(session_id)
    svc.set_options(s, body.solver_id, body.params, body.dictionary_id)
    return svc.state(s, request_id(request))
