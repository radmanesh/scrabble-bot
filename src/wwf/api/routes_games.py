"""Saved games over JSON: the list, and one game with its turn log."""

from __future__ import annotations

from fastapi import APIRouter, Request

from wwf.api import gameplay
from wwf.api.app import request_id, services
from wwf.api.games import Game, Turn
from wwf.api.schemas import (
    GameOut,
    GamesResponse,
    GameSummary,
    PlacementOut,
    TurnOut,
    WordOut,
)

router = APIRouter(tags=["games"])


def _turn_out(t: Turn) -> TurnOut:
    return TurnOut(
        n=t.n,
        player=t.player,
        kind=t.kind,
        word=t.word,
        direction=t.direction,
        start=t.start,
        points=t.points,
        bingo=t.bingo,
        words=[WordOut(**w) for w in t.words],
        placements=[PlacementOut(**p) for p in t.placements],
        source=t.source,
        forced=t.forced,
        at=t.at,
    )


def _summary(g: Game) -> GameSummary:
    return GameSummary(
        id=g.id,
        name=g.name,
        phase=g.phase,
        scores=g.scores(),
        turns=len(g.turns),
        updated_at=g.updated_at,
    )


@router.get("/games", response_model=GamesResponse)
def list_games(request: Request) -> GamesResponse:
    """Saved games, most recently played first."""
    svc = services(request)
    return GamesResponse(
        request_id=request_id(request), games=[_summary(g) for g in svc.games.saved()]
    )


@router.get("/games/{game_id}", response_model=GameOut)
def get_game(request: Request, game_id: str) -> GameOut:
    """One game: the board, rack, scores, unseen tiles and every logged turn."""
    svc = services(request)
    g = svc.game(game_id)
    bag = gameplay.bag_for(svc, g)
    return GameOut(
        request_id=request_id(request),
        id=g.id,
        name=g.name,
        phase=g.phase,
        board=g.board,
        rack=g.rack,
        scores=g.scores(),
        in_bag=bag.in_bag if bag else 0,
        unseen=bag.unseen if bag else {},
        turns=[_turn_out(t) for t in g.turns],
        solver_id=g.solver_id,
        dictionary_id=g.dictionary_id,
        created_at=g.created_at,
        updated_at=g.updated_at,
    )
