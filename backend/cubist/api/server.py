"""FastAPI app + WebSocket bus. Person E owns.

Routes:
  GET  /api/health            liveness
  GET  /api/engines           list all engines
  GET  /api/generations       list all generations
  GET  /api/games?gen=N       games in a generation
  POST /api/generations/run   trigger a new generation
  WS   /ws                    live event stream
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from cubist.api.routes import router
from cubist.api.websocket import bus

app = FastAPI(title="Cubist")
app.include_router(router, prefix="/api")


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    q = bus.subscribe()
    try:
        while True:
            envelope = await q.get()
            await websocket.send_json(envelope)
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(q)
