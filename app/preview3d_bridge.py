"""Round-trip render bridge for Preview3D / SaveGLB IMAGE+MASK outputs.

Flow: a node's execute() calls bridge.request_render(), which sends a
websocket "preview3d.render_request" event to the client and awaits a
Future. The client renders the file with a fixed camera at fixed
resolution, uploads the PNGs to /temp/, and POSTs {render_id, image,
mask} to /3d/render_response, which resolves the Future. The node then
loads image/mask via the standard LoadImage path.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import uuid
from typing import Any

from aiohttp import web


_RENDER_TIMEOUT_SEC = 60


class Preview3DBridge:
    """Server-side half of the Preview3D / SaveGLB render round-trip.

    Owns the pending Future registry and the POST /3d/render_response
    route. Instantiated once on PromptServer and surfaced as
    PromptServer.instance.preview3d_bridge.
    """

    def __init__(self) -> None:
        self._pending: dict[str, concurrent.futures.Future] = {}

    def add_routes(self, routes: web.RouteTableDef) -> None:
        @routes.post("/3d/render_response")
        async def render_response(request: web.Request) -> web.Response:
            try:
                data = await request.json()
            except Exception:
                return web.Response(status=400, text="invalid json")
            render_id = data.get("render_id")
            if not render_id or render_id not in self._pending:
                return web.Response(status=404, text="unknown render_id")
            future = self._pending.pop(render_id)
            if future.done():
                return web.Response(status=409, text="already resolved")
            if "error" in data:
                future.set_exception(RuntimeError(str(data["error"])))
            else:
                future.set_result({
                    "image": data.get("image"),
                    "mask": data.get("mask"),
                })
            return web.json_response({"ok": True})

    async def request_render(
        self,
        node_id: str,
        file_path: str,
        file_type: str = "output",
        camera_info: dict | None = None,
        timeout: float = _RENDER_TIMEOUT_SEC,
    ) -> dict[str, Any]:
        """Send a render request and await the client's response.

        node_id should be the executor-provided cls.hidden.unique_id of
        the requesting Preview3D / SaveGLB node — the client uses it to
        find the matching viewer instance.
        """
        from server import PromptServer
        server = PromptServer.instance
        if server is None:
            raise RuntimeError("PromptServer is not initialized")

        render_id = uuid.uuid4().hex
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._pending[render_id] = future

        payload = {
            "render_id": render_id,
            "node_id": node_id,
            "file_path": file_path,
            "type": file_type,
            "camera_info": camera_info,
        }
        server.send_sync("preview3d.render_request", payload, server.client_id)

        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(future), timeout=timeout
            )
        except asyncio.TimeoutError:
            self._pending.pop(render_id, None)
            raise RuntimeError(
                f"Preview3D render bridge: client did not respond in {timeout}s "
                f"(node_id={node_id}, file={file_path}). Is the workflow open "
                "in a browser tab?"
            )
        except Exception:
            self._pending.pop(render_id, None)
            raise


def load_image_and_mask(image_ref: str, mask_ref: str):
    import nodes
    loader = nodes.LoadImage()
    image_tensor, _ = loader.load_image(image=image_ref)
    _, mask_tensor = loader.load_image(image=mask_ref)
    return image_tensor, mask_tensor


__all__ = [
    "Preview3DBridge",
    "load_image_and_mask",
]
