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

import { useEffect } from "react";
import { useEventStream } from "./hooks/useEventStream";
import LiveBoards from "./components/LiveBoards";
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

  // Whether a generation is currently running, derived from the event log.
  // ``generation.started`` flips this on; any of ``finished`` / ``cancelled``
  // flips it off. We walk from the end of the array because the latest
  // boundary event wins.
  const isRunning = (() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const t = events[i].type;
      if (t === "generation.started") return true;
      if (t === "generation.finished" || t === "generation.cancelled") {
        return false;
      }
    }
    return false;
  })();

  /** Cancel any running generation and start a new one. */
  function runGeneration() {
    fetch("/api/generations/run", { method: "POST" }).catch(() => {
      // Backend may not be running in offline/mock development — ignore silently
    });
  }

  /** Cancel the running generation, leaving the dashboard idle. */
  function stopGeneration() {
    fetch("/api/generations/stop", { method: "POST" }).catch(() => {
      // Same offline-tolerance as runGeneration
    });
  }

  // Stronger than Stop: also wipes engines/games/strategist questions on
  // the server and broadcasts ``state.cleared`` so every connected
  // dashboard zeroes its event log. We confirm because this is destructive
  // — once cleared, history can't be recovered without replaying the
  // generations from scratch.
  function clearAll() {
    if (
      !window.confirm(
        "Clear all engines, games, and strategist history? This cannot be undone."
      )
    ) {
      return;
    }
    fetch("/api/state/clear", { method: "POST" }).catch(() => {
      // Backend may be offline (mock dev mode) — frontend won't see a
      // state.cleared event, but there's nothing to clear in that case.
    });
  }

  // Cancel the in-flight generation when the user closes/reloads the tab.
  // sendBeacon is the only network call the browser guarantees to flush
  // during pagehide — fetch() can race the document teardown and get
  // dropped, leaving a generation churning the LLM with nobody watching.
  useEffect(() => {
    const onPageHide = () => {
      try {
        navigator.sendBeacon("/api/generations/stop");
      } catch {
        // sendBeacon throws on some browsers if the URL is rejected.
        // Best-effort only — there's nothing useful we can do here.
      }
    };
    window.addEventListener("pagehide", onPageHide);
    return () => window.removeEventListener("pagehide", onPageHide);
  }, []);

  return (
    <div className="min-h-screen p-6">
      {/* ── Header ───────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-widest text-gray-100">
            SELF-IMPROVING CHESS ENGINE
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Agentic tournament selection, one generation at a time
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Event counter badge — useful for debugging during demo setup */}
          <span className="text-xs text-gray-500 font-mono">
            {events.length} events
          </span>

          <button
            onClick={stopGeneration}
            disabled={!isRunning}
            className="px-3 py-2 bg-gray-700 hover:bg-gray-600 active:bg-gray-800 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed text-white text-sm font-semibold rounded transition-colors"
          >
            ■ Stop
          </button>

          <button
            onClick={clearAll}
            disabled={events.length === 0 && !isRunning}
            className="px-3 py-2 bg-red-700 hover:bg-red-600 active:bg-red-800 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed text-white text-sm font-semibold rounded transition-colors"
          >
            Clear
          </button>

          <button
            onClick={runGeneration}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white text-sm font-semibold rounded transition-colors"
          >
            {isRunning ? "Restart Generation" : "Run Generation"}
          </button>
        </div>
      </header>

      {/* ── Row 1: Live boards (full width) ──────────────────────────────── */}
      <div className="mb-6">
        <LiveBoards events={events} />
      </div>

      {/* ── Row 2: Strategist feed + Elo chart ───────────────────────────── */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        <StrategistFeed events={events} />
        <EloChart events={events} />
      </div>

      {/* ── Row 3: Tournament bracket + Generation history ────────────────── */}
      <div className="grid grid-cols-2 gap-6">
        <Bracket events={events} />
        <GenerationTimeline events={events} />
      </div>
    </div>
  );
}
