"""Image upload and recognition."""

from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import Response

from wwf.api.app import request_id, run_blocking, services
from wwf.api.schemas import RecognizeResponse, SessionState
from wwf.api.services import ApiError, Services
from wwf.api.session import Session

router = APIRouter(tags=["recognition"])


async def read_upload(svc: Services, image: UploadFile) -> bytes:
    """The upload's bytes, refusing anything over the size limit without reading it all."""
    limit = svc.settings.max_upload_bytes
    data = await image.read(limit + 1)
    if len(data) > limit:
        raise ApiError(413, "image_too_large", {"max_bytes": limit})
    return data


async def recognize_into(request: Request, svc: Services, s: Session, data: bytes) -> None:
    content_type = svc.check_image(data)
    await run_blocking(request, svc.attach_image, s, data, content_type)


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(request: Request, image: UploadFile = File(...)) -> RecognizeResponse:
    """Run recognition on an uploaded screenshot or photo and start a session with the result.

    Until the vision pipeline is built, the board comes back empty with the warning
    `recognition_unavailable`; the session still keeps the image for the review page.
    """
    svc = services(request)
    data = await read_upload(svc, image)
    s = svc.new_session()
    await recognize_into(request, svc, s, data)
    assert s.recognition is not None
    return RecognizeResponse(
        **s.recognition.model_dump(), session_id=s.id, request_id=request_id(request)
    )


@router.post("/session", response_model=SessionState, status_code=201)
def create_session(request: Request) -> SessionState:
    """Start an empty session, for typing a board in by hand."""
    svc = services(request)
    return svc.state(svc.new_session(), request_id(request))


@router.get("/session/{session_id}/image", response_class=Response)
def session_image(request: Request, session_id: str) -> Response:
    """The image uploaded for this session."""
    s = services(request).session(session_id)
    if s.image is None:
        raise ApiError(404, "no_image", {"session_id": session_id})
    return Response(s.image, media_type=s.image_type, headers={"Cache-Control": "private"})
