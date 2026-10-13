"""Bridges the vision package to the API: loads the reader and converts its results."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from wwf.api.schemas import GridGeometryOut, RecognitionResultOut
from wwf.core.config import BoardConfig
from wwf.vision.pipeline import RecognitionResult, Recognizer

log = logging.getLogger("wwf.api")


def to_out(r: RecognitionResult) -> RecognitionResultOut:
    g = r.geometry
    geometry = None
    if g is not None:
        geometry = GridGeometryOut(
            cells=[[tuple(b) for b in row] for row in g.cells],
            rack=[tuple(b) for b in g.rack],
            pitch=g.pitch,
            confidence=g.confidence,
            image_width=g.image_width,
            image_height=g.image_height,
            quad=g.quad,
        )
    return RecognitionResultOut(
        board=r.board,
        rack=r.rack,
        confidence=r.confidence,
        needs_review=r.needs_review,
        rack_needs_review=r.rack_needs_review,
        geometry=geometry,
        warnings=r.warnings,
        input_type="photo" if r.input_type == "photo" else "screenshot",
        elapsed_ms=r.elapsed_ms,
        kept_from_previous=r.kept_from_previous,
        new_tiles=r.new_tiles,
        conflicts=r.conflicts,
        merged=r.merged,
        rejected=r.rejected,
        match=r.match,
    )


def load_recognizer(cfg: BoardConfig, path: Path) -> Callable[..., RecognitionResultOut] | None:
    """The board reader as an API recognizer, or None when no model has been trained yet."""
    try:
        recognizer = Recognizer.load(cfg, path)
    except Exception:
        log.exception("could not load the board reader from %s", path)
        return None
    if recognizer is None:
        log.info("no board reader at %s: pictures will not be read automatically", path)
        return None
    log.info("loaded board reader from %s", path)

    def read(
        data: bytes,
        previous: list[list[str | None]] | None = None,
        hint_quad: list[list[float]] | None = None,
    ) -> RecognitionResultOut:
        return to_out(recognizer(data, previous, hint_quad))

    read.model = recognizer.model  # type: ignore[attr-defined]
    return read


def retrain(archive: Path | None, games_dir: Path | None, cfg: BoardConfig, path: Path) -> bool:
    """Rebuild the reader from the archive and the games' labelled pictures. False when there
    was nothing to learn from."""
    from wwf.vision.train import train_from_sources

    return train_from_sources(archive, games_dir, cfg, path) is not None
