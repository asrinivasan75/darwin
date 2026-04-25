/**
 * useEventStream.ts — React hook that provides a unified CubistEvent log.
 *
 * Transparently switches between two event sources:
 *   - **Mock mode** (`?mock` in the URL): drives the UI from {@link startMockStream}
 *     so the dashboard can be developed and demoed without a live backend.
 *   - **Live mode** (default): opens a WebSocket via {@link connectEvents} and
 *     forwards every envelope's inner event into the accumulated state array.
 *
 * The returned array grows monotonically — events are never removed. All
 * components filter this array client-side, which keeps state management
 * trivially simple and avoids prop-drilling a complex store.
 *
 * @module useEventStream
 */

import { useEffect, useState } from "react";
import { connectEvents } from "../api/client";
import { startMockStream } from "./mockEvents";
import type { CubistEvent } from "../api/events";

/**
 * Accumulates all {@link CubistEvent}s received since the component mounted.
 *
 * Activates mock mode when the URL contains `?mock` (any value); otherwise
 * connects to the live WebSocket endpoint proxied through Vite at `/ws`.
 *
 * @returns read-only, ever-growing array of events in arrival order
 *
 * @example
 * ```tsx
 * const events = useEventStream();
 * const questions = events.filter(e => e.type === "strategist.question");
 * ```
 */
export function useEventStream(): CubistEvent[] {
  const [events, setEvents] = useState<CubistEvent[]>([]);

  useEffect(() => {
    const useMock = new URLSearchParams(location.search).has("mock");

    /** Appends one event to the accumulated log without mutating prior state. */
    const push = (e: CubistEvent) => setEvents((prev) => [...prev, e]);

    if (useMock) {
      // startMockStream returns a cleanup fn that cancels all pending timeouts
      return startMockStream(push);
    }

    const ws = connectEvents(push);
    return () => ws.close();
  }, []); // intentionally runs once per mount — source does not change

  return events;
}
