"""aiohttp routes for the download manager.

Endpoint surface (all under ``/api/download``), mirroring the response
envelope used by ``app/assets/api/routes.py``:

  POST   /api/download/enqueue
  GET    /api/download
  POST   /api/download/availability
  POST   /api/download/credentials
  GET    /api/download/credentials
  GET    /api/download/credentials/{id}
  DELETE /api/download/credentials/{id}
  GET    /api/download/{id}
  POST   /api/download/{id}/pause
  POST   /api/download/{id}/resume
  POST   /api/download/{id}/cancel
  POST   /api/download/{id}/priority

Note on ordering: the static ``credentials`` routes are registered before the
dynamic ``/api/download/{id}`` route so a request to ``.../credentials`` is not
captured as ``id == "credentials"``.
"""

from __future__ import annotations

import json

from aiohttp import web
from pydantic import BaseModel, ValidationError

from app.model_downloader.api import schemas_in, schemas_out
from app.model_downloader.credentials.store import (
    CREDENTIAL_STORE,
    CredentialValidationError,
)
from app.model_downloader.manager import DOWNLOAD_MANAGER, DownloadError

ROUTES = web.RouteTableDef()


def register_routes(app: web.Application) -> None:
    """Wire the download-manager routes into the running aiohttp app."""
    app.add_routes(ROUTES)


# ----- envelope helpers (same shape as app/assets/api/routes.py) -----


def _error(status: int, code: str, message: str, details: dict | None = None) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message, "details": details or {}}},
        status=status,
    )


def _ok(payload, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status)


async def _parse(request: web.Request, model: type[BaseModel]):
    try:
        raw = await request.json()
    except json.JSONDecodeError:
        return _error(400, "INVALID_JSON", "Request body must be valid JSON.")
    try:
        return model.model_validate(raw)
    except ValidationError as ve:
        return _error(400, "INVALID_BODY", "Validation failed.", {"errors": json.loads(ve.json())})


def _from_download_error(e: DownloadError) -> web.Response:
    return _error(e.http_status, e.code, e.message)


# ----- downloads: collection + enqueue + availability -----


@ROUTES.post("/api/download/enqueue")
async def enqueue(request: web.Request) -> web.Response:
    parsed = await _parse(request, schemas_in.EnqueueRequest)
    if isinstance(parsed, web.Response):
        return parsed
    try:
        download_id = await DOWNLOAD_MANAGER.enqueue(
            parsed.url,
            parsed.model_id,
            priority=parsed.priority,
            expected_sha256=parsed.expected_sha256,
            allow_any_extension=parsed.allow_any_extension,
            credential_id=parsed.credential_id,
        )
    except DownloadError as e:
        return _from_download_error(e)
    return _ok({"download_id": download_id, "accepted": True}, status=202)


@ROUTES.get("/api/download")
async def list_downloads(request: web.Request) -> web.Response:
    return _ok({"downloads": await DOWNLOAD_MANAGER.list()})


@ROUTES.post("/api/download/availability")
async def availability(request: web.Request) -> web.Response:
    parsed = await _parse(request, schemas_in.AvailabilityRequest)
    if isinstance(parsed, web.Response):
        return parsed
    return _ok({"models": await DOWNLOAD_MANAGER.availability(parsed.models)})


# ----- credentials (secrets are write-only) — must precede /{id} -----


@ROUTES.post("/api/download/credentials")
async def upsert_credential(request: web.Request) -> web.Response:
    parsed = await _parse(request, schemas_in.CredentialUpsertRequest)
    if isinstance(parsed, web.Response):
        return parsed
    try:
        view = await CREDENTIAL_STORE.upsert(
            parsed.host,
            parsed.secret,
            auth_scheme=parsed.auth_scheme,
            header_name=parsed.header_name,
            query_param=parsed.query_param,
            label=parsed.label,
            match_subdomains=parsed.match_subdomains,
            enabled=parsed.enabled,
        )
    except CredentialValidationError as e:
        return _error(400, "INVALID_CREDENTIAL", str(e))
    return _ok(schemas_out.credential_to_dict(view), status=201)


@ROUTES.get("/api/download/credentials")
async def list_credentials(request: web.Request) -> web.Response:
    views = await CREDENTIAL_STORE.list()
    return _ok({"credentials": [schemas_out.credential_to_dict(v) for v in views]})


@ROUTES.get("/api/download/credentials/{id}")
async def get_credential(request: web.Request) -> web.Response:
    view = await CREDENTIAL_STORE.get(request.match_info["id"])
    if view is None:
        return _error(404, "NOT_FOUND", "No such credential.")
    return _ok(schemas_out.credential_to_dict(view))


@ROUTES.delete("/api/download/credentials/{id}")
async def delete_credential(request: web.Request) -> web.Response:
    deleted = await CREDENTIAL_STORE.delete(request.match_info["id"])
    if not deleted:
        return _error(404, "NOT_FOUND", "No such credential.")
    return _ok({"deleted": True})


# ----- single download by id (dynamic; registered last) -----


@ROUTES.get("/api/download/{id}")
async def get_download(request: web.Request) -> web.Response:
    view = await DOWNLOAD_MANAGER.status(request.match_info["id"])
    if view is None:
        return _error(404, "NOT_FOUND", "No such download.")
    return _ok(view)


@ROUTES.post("/api/download/{id}/pause")
async def pause(request: web.Request) -> web.Response:
    try:
        await DOWNLOAD_MANAGER.pause(request.match_info["id"])
    except DownloadError as e:
        return _from_download_error(e)
    return _ok({"ok": True})


@ROUTES.post("/api/download/{id}/resume")
async def resume(request: web.Request) -> web.Response:
    try:
        await DOWNLOAD_MANAGER.resume(request.match_info["id"])
    except DownloadError as e:
        return _from_download_error(e)
    return _ok({"ok": True})


@ROUTES.post("/api/download/{id}/cancel")
async def cancel(request: web.Request) -> web.Response:
    try:
        await DOWNLOAD_MANAGER.cancel(request.match_info["id"])
    except DownloadError as e:
        return _from_download_error(e)
    return _ok({"ok": True})


@ROUTES.post("/api/download/{id}/priority")
async def set_priority(request: web.Request) -> web.Response:
    parsed = await _parse(request, schemas_in.PriorityRequest)
    if isinstance(parsed, web.Response):
        return parsed
    try:
        await DOWNLOAD_MANAGER.set_priority(request.match_info["id"], parsed.priority)
    except DownloadError as e:
        return _from_download_error(e)
    return _ok({"ok": True})
