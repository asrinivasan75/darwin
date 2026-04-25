/**
 * App.tsx — top-level layout shell for the Cubist dashboard.
 *
 * Mounts the {@link useEventStream} hook and fans the accumulated event log
 * out to all five dashboard components. No per-component state management is
 * needed — each component filters the flat event array client-side.
 *
 * Layout (Tailwind grid):
 *   Row 1 (3 columns): LiveBoard | StrategistFeed | EloChart
 *   Row 2 (2 columns): Bracket   | GenerationTimeline
 *
 * The "Run Generation" button in the header fires `POST /api/generations/run`
 * which triggers the backend orchestration loop. The button is fire-and-forget;
 * progress is reflected through the WebSocket event stream.
 *
 * @module App
 */

import { useEventStream } from "./hooks/useEventStream";
import LiveBoard from "./components/LiveBoard";
import EloChart from "./components/EloChart";
import StrategistFeed from "./components/StrategistFeed";
import Bracket from "./components/Bracket";
import GenerationTimeline from "./components/GenerationTimeline";

/**
 * App — root component that assembles the full Cubist dashboard.
 *
 * Uses {@link useEventStream} to obtain the live (or mock) event log, then
 * passes it to every panel. Switching from mock to live requires only removing
 * `?mock` from the URL — no code changes needed.
 *
 * @returns the full page layout with header and all five dashboard panels
 */
export default function App() {
  // Single source of truth for all WebSocket events — shared read-only across
  // every panel so each can derive its own view without extra state management.
  const events = useEventStream();

  /** Fires the backend orchestration endpoint to kick off a new generation. */
  function runGeneration() {
    fetch("/api/generations/run", { method: "POST" }).catch(() => {
      // Backend may not be running in offline/mock development — ignore silently
    });
  }

  return (
    <div className="min-h-screen p-6">
      {/* ── Header ───────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-widest text-gray-100">
            CUBIST
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Self-improving chess engine
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Event counter badge — useful for debugging during demo setup */}
          <span className="text-xs text-gray-500 font-mono">
            {events.length} events
          </span>

          <button
            onClick={runGeneration}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white text-sm font-semibold rounded transition-colors"
          >
            Run Generation
          </button>
        </div>
      </header>

      {/* ── Row 1: Live board + Strategist feed + Elo chart ──────────────── */}
      <div className="grid grid-cols-3 gap-6 mb-6">
        <LiveBoard events={events} />
        <StrategistFeed events={events} />
        <EloChart events={events} />
      </div>

      {/* ── Row 2: Tournament bracket + Generation history ────────────────── */}
      <div className="grid grid-cols-2 gap-6">
        <Bracket events={events} />
        <GenerationTimeline events={events} />
      </div>
    </div>
  );
}
