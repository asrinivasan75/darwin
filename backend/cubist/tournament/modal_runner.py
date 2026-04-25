"""Remote tournament backend: dispatch each game to a Modal container.

Each ``play_game_remote.starmap`` invocation fans out one Modal container
per game in the round-robin. Containers run in real OS-level parallel —
no GIL ceiling — so a 20-game tournament that takes ~130 s locally
finishes in ~10–20 s wall-clock.

The remote function reuses ``cubist.tournament.referee.play_game`` for
the actual chess loop so the local-vs-remote behavior is identical
(same termination rules, same time budget, same fallback handling). It
captures events into a list and returns them; the local round_robin
re-emits them through the bus when the result lands. This means the
LiveBoards panel will update once per *game* completion rather than
per-move during the tournament — a small UX trade for the wall-clock
speedup. (Per-move streaming via ``modal.Queue`` is a follow-up.)

The Modal app is deployed once via:
    modal deploy backend/cubist/tournament/modal_runner.py
And re-deployed whenever the cubist package changes.
"""

from __future__ import annotations

import modal

# Image: Debian slim + python-chess + the local cubist package. The
# `cubist` source is baked into the image at deploy time so the remote
# function can ``from cubist.tournament.referee import play_game``
# without divergence from the local implementation.
#
# Stripped image: pure-code engines don't need the LLM SDKs
# (google-genai, anthropic). Saves ~100 MB image weight and ~2 s
# cold-start. cubist.config doesn't *import* those libs — only
# cubist.llm does, and the tournament path doesn't touch llm.py.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "chess",
        "sqlmodel",
        "pydantic",
        "pydantic-settings",
    )
    .add_local_python_source("cubist", copy=True)
)

app = modal.App("cubist-tournament", image=image)

# Shared event queue. Containers push every game.move / game.finished
# event into this queue; the local round_robin process subscribes and
# re-emits to the bus in real time so the dashboard sees moves as they
# happen instead of batched at game-end. Named (not ephemeral) so the
# local process can attach to the same queue without being inside an
# ``app.run()`` context.
events_queue = modal.Queue.from_name("cubist-events", create_if_missing=True)


@app.function(
    cpu=1,
    # Hard ceiling per game: 60s. With a 5s per-move budget and a 60-
    # move expected game length, a healthy game lands well under this.
    # Pathologically slow engines (e.g., depth-unbounded quiescence
    # in capture-dense positions) get killed here so they can't gate
    # the whole tournament's wall-clock.
    timeout=60,
    # Cap concurrency to 40 — handles 4 candidates + 1 incumbent =
    # 5 engines × 4 ordered pairs × games_per_pairing=2 = 40-game
    # worst-case round-robins without queuing.
    max_containers=40,
    # No idle containers by default — zero baseline cost. Toggle warm
    # pool on demand via:
    #   modal app keep-warm cubist-tournament play_game_remote N
    # or Modal dashboard, or the (planned) /api/modal/warm-up endpoint.
    min_containers=0,
    # Memory snapshots checkpoint the container *after* `from cubist...`
    # imports complete, so cold starts skip Python init and import
    # resolution — typically 3-5s saved per container. This makes the
    # min_containers=0 default tolerable: first-game cold-start drops
    # from ~10s (full init) to ~1-2s (just executor startup).
    enable_memory_snapshot=True,
)
async def play_game_remote(
    white_src: str,
    white_name: str,
    black_src: str,
    black_name: str,
    time_per_move_ms: int,
    game_id: int,
) -> dict:
    """Run one chess game inside a Modal container.

    Args:
        white_src / black_src: full Python source of each engine module
            (the contents of ``engines/generated/<name>.py`` or, for the
            baseline, ``inspect.getsourcefile(type(baseline))`` read whole).
        white_name / black_name: ``engine.name`` of each side. Used for
            keying the synthesized module namespaces; the engine
            self-names in its ``__init__``.
        time_per_move_ms: forwarded to ``referee.play_game``.
        game_id: forwarded to ``referee.play_game`` so emitted events
            tag back to the same board on the dashboard.

    Returns:
        A dict with the GameResult fields (``white, black, result,
        termination, pgn``). Live events stream out via the
        ``events_queue`` modal.Queue while the game is in progress.
    """
    import sys
    import types

    from cubist.tournament.referee import play_game

    def _load(src: str, name: str):
        # Unique module name per game so reimports across concurrent
        # games inside the same warmed container don't clash.
        mod_name = (
            f"cubist_remote_engine_{name.replace('-', '_')}_{game_id}"
        )
        mod = types.ModuleType(mod_name)
        sys.modules[mod_name] = mod
        exec(
            compile(src, f"<remote:{name}>", "exec"),
            mod.__dict__,
        )
        return mod.engine

    white = _load(white_src, white_name)
    black = _load(black_src, black_name)

    # Buffer events locally and flush in batches. Each ``put.aio`` is a
    # control-plane RPC (~50-100ms); doing one per move is too expensive
    # — 50 events/game × 100ms = 5s overhead PER GAME. Batching to 10
    # events per RPC drops that to ~0.5s/game while the dashboard still
    # ticks ~6 times per typical game (close-enough to live).
    pending: list[dict] = []
    BATCH = 10

    async def flush() -> None:
        if pending:
            await events_queue.put_many.aio(pending)
            pending.clear()

    async def emit(event: dict) -> None:
        pending.append(event)
        if len(pending) >= BATCH:
            await flush()

    result = await play_game(
        white=white,
        black=black,
        time_per_move_ms=time_per_move_ms,
        on_event=emit,
        game_id=game_id,
    )
    # Flush the tail (game.finished + any remaining moves under BATCH).
    await flush()

    return {
        "white": result.white,
        "black": result.black,
        "result": result.result,
        "termination": result.termination,
        "pgn": result.pgn,
    }
