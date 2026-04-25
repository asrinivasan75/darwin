"""FROZEN CONTRACT — do not change without team sync.

WebSocket event payloads. Backend (Person E) emits these; frontend
(Person D) consumes them. Mirror in `frontend/src/api/events.ts` MUST
stay in sync.
"""

from __future__ import annotations

import asyncio
from typing import Literal, Union

from pydantic import BaseModel, Field


class GenerationStarted(BaseModel):
    type: Literal["generation.started"] = "generation.started"
    number: int
    champion: str


class StrategistQuestion(BaseModel):
    type: Literal["strategist.question"] = "strategist.question"
    index: int  # 0..4
    category: str  # "prompt" | "search" | "book" | "evaluation" | "sampling"
    text: str


class BuilderCompleted(BaseModel):
    type: Literal["builder.completed"] = "builder.completed"
    question_index: int
    engine_name: str
    ok: bool
    error: str | None = None


class GameMove(BaseModel):
    type: Literal["game.move"] = "game.move"
    game_id: int
    fen: str
    san: str
    white: str
    black: str
    ply: int


class GameFinished(BaseModel):
    type: Literal["game.finished"] = "game.finished"
    game_id: int
    result: str  # "1-0" | "0-1" | "1/2-1/2"
    termination: str
    pgn: str
    white: str
    black: str


class GenerationFinished(BaseModel):
    type: Literal["generation.finished"] = "generation.finished"
    number: int
    new_champion: str
    elo_delta: float
    promoted: bool  # True if a new champion was crowned


Event = Union[
    GenerationStarted,
    StrategistQuestion,
    BuilderCompleted,
    GameMove,
    GameFinished,
    GenerationFinished,
]


class Envelope(BaseModel):
    """All WS messages are wrapped in this envelope."""

    event: Event = Field(discriminator="type")


class EventBus:
    """In-process pub/sub fanout for WS subscribers.

    Each `/ws` connection calls `subscribe()` to get its own bounded queue;
    orchestration and tournament code calls `emit()` with an event dict
    matching one of the frozen `Event` types above. The envelope wrapping
    matches the `Envelope` contract so the frontend decoder in
    `frontend/src/api/events.ts` can discriminate on `event.type`.

    Backpressure policy: each subscriber queue is bounded at 1000. If a
    subscriber stops draining (slow/stalled browser), `put_nowait` raises
    `QueueFull` and we drop the event for *that subscriber only*. This is
    intentional — we'd rather show a partial stream than block the entire
    orchestrator on one stuck client.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        """Register a new subscriber and return its dedicated queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Drop a subscriber. Safe to call multiple times."""
        self._subscribers.discard(q)

    async def emit(self, event_payload: dict) -> None:
        """Broadcast an event to every live subscriber.

        `event_payload` should be a dict whose `type` field matches one of
        the `Event` discriminated-union members. It is wrapped in the
        `Envelope` shape (`{"event": event_payload}`) before fanout.
        """
        envelope = {"event": event_payload}
        for q in list(self._subscribers):
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                pass


bus = EventBus()
