"""Static player + run files over HTTP, and a WebSocket relay for the audio stream.

/ws/feed  - producers (toonify.stream_client) send the handshake, PCM chunks and `end`.
/ws/play  - players receive exactly those messages. The last handshake and the last emotion are replayed to late
           joiners.
"""
from __future__ import annotations

import json

from aiohttp import WSMsgType, web

from .config import PLAYER_DIR, RUNS_DIR


def make_app() -> web.Application:
    players: set[web.WebSocketResponse] = set()
    state = {"start": None, "emotion": None}

    async def broadcast(kind, data):
        for ws in list(players):
            try:
                await (ws.send_bytes(data) if kind == "bytes" else ws.send_str(data))
            except ConnectionError:
                players.discard(ws)

    async def play(request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        players.add(ws)
        for kind in ("start", "emotion"):
            if state[kind]:
                await ws.send_str(state[kind])
        try:
            async for _ in ws:   # players only listen
                pass
        finally:
            players.discard(ws)
        return ws

    async def feed(request):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=16 * 2**20)
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                await broadcast("bytes", msg.data)
            elif msg.type == WSMsgType.TEXT:
                try:
                    kind = json.loads(msg.data).get("type")
                except (ValueError, AttributeError):
                    continue
                if kind in state:
                    state[kind] = msg.data
                await broadcast("str", msg.data)
        return ws

    async def index(request):
        raise web.HTTPFound("/player/index.html" + (f"?{request.query_string}" if request.query_string else ""))

    @web.middleware
    async def no_cache(request, handler):
        # Re-running a stage rewrites files in place; make browsers revalidate instead of using stale copies.
        resp = await handler(request)
        if request.path.startswith(("/player/", "/runs/")):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    app = web.Application(middlewares=[no_cache])
    app.router.add_get("/", index)
    app.router.add_get("/ws/play", play)
    app.router.add_get("/ws/feed", feed)
    app.router.add_static("/player/", PLAYER_DIR)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    app.router.add_static("/runs/", RUNS_DIR)
    return app


def serve(host: str = "0.0.0.0", port: int = 8780, ssl_cert: str | None = None, ssl_key: str | None = None) -> None:
    """Browsers only allow AudioWorklet in a secure context: open the player via localhost (e.g. an SSH
    tunnel) or serve HTTPS with a certificate (e.g. from mkcert, as miep does)."""
    ctx = None
    if ssl_cert:
        import ssl
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(ssl_cert, ssl_key)
    http, ws = ("https", "wss") if ctx else ("http", "ws")
    print(f"[serve] player on {http}://{host}:{port}/?run=<run id>   audio feed {ws}://{host}:{port}/ws/feed")
    web.run_app(make_app(), host=host, port=port, ssl_context=ctx, print=None)
