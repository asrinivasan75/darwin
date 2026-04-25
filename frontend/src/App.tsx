/**
 * App.tsx — top-level layout shell for the Darwin dashboard.
 *
 * Mounts the {@link useEventStream} hook and fans the accumulated event log
 * out to all five dashboard components. No per-component state management is
 * needed — each component filters the flat event array client-side.
 *
 * @module App
 */

import { useEffect } from "react";
import { useEventStream } from "./hooks/useEventStream";
import LiveBoards from "./components/LiveBoards";
import EloChart from "./components/EloChart";
import EnginesEloChart from "./components/EnginesEloChart";
import StrategistFeed from "./components/StrategistFeed";
import Bracket from "./components/Bracket";
import GenerationTimeline from "./components/GenerationTimeline";

export default function App() {
  const events = useEventStream();

  const { isRunning, currentGen, currentChampion, finishedCount } = (() => {
    let running = false;
    let gen: number | null = null;
    let champion: string | null = null;
    let finished = 0;
    for (const e of events) {
      if (e.type === "generation.started") {
        running = true;
        gen = e.number;
        if (champion === null) champion = e.champion;
      } else if (e.type === "generation.finished") {
        running = false;
        gen = e.number;
        champion = e.new_champion;
        finished += 1;
      } else if (e.type === "generation.cancelled") {
        running = false;
      }
    }
    return {
      isRunning: running,
      currentGen: gen,
      currentChampion: champion,
      finishedCount: finished,
    };
  })();

  function runGeneration() {
    fetch("/api/generations/run", { method: "POST" }).catch(() => {});
  }

  function stopGeneration() {
    fetch("/api/generations/stop", { method: "POST" }).catch(() => {});
  }

  function clearAll() {
    if (
      !window.confirm(
        "Clear all engines, games, and strategist history? This cannot be undone.",
      )
    ) {
      return;
    }
    fetch("/api/state/clear", { method: "POST" }).catch(() => {});
  }

  useEffect(() => {
    const onPageHide = () => {
      try {
        navigator.sendBeacon("/api/generations/stop");
      } catch {
        /* sendBeacon throws on some browsers — best-effort only. */
      }
    };
    window.addEventListener("pagehide", onPageHide);
    return () => window.removeEventListener("pagehide", onPageHide);
  }, []);

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      {/* Top hairline — a single warm thread suggesting "this room has a
          ceiling." Pure decoration. */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-x-0 top-0 h-px"
        style={{
          background:
            "linear-gradient(90deg, transparent, rgba(201,168,118,0.35) 30%, rgba(201,168,118,0.35) 70%, transparent)",
        }}
      />

      <div className="relative mx-auto w-full max-w-[1480px] px-8 pb-20 pt-10">
        <Header
          isRunning={isRunning}
          currentGen={currentGen}
          currentChampion={currentChampion}
          finishedCount={finishedCount}
          eventCount={events.length}
          onRun={runGeneration}
          onStop={stopGeneration}
          onClear={clearAll}
        />

        {/* Row 1 — live boards (what is it doing right now?) */}
        <section className="rise" style={{ animationDelay: "240ms" }}>
          <LiveBoards events={events} />
        </section>

        {/* Row 2 — this generation: cause (strategist) + effect (bracket) */}
        <section
          className="mt-6 grid grid-cols-1 gap-6 rise lg:grid-cols-2"
          style={{ animationDelay: "360ms" }}
        >
          <StrategistFeed events={events} />
          <Bracket events={events} />
        </section>

        {/* Row 3 — across generations: champion line + every engine */}
        <section
          className="mt-6 grid grid-cols-1 gap-6 rise lg:grid-cols-2"
          style={{ animationDelay: "480ms" }}
        >
          <EloChart events={events} />
          <EnginesEloChart events={events} />
        </section>

        {/* Row 4 — full history list */}
        <section
          className="mt-6 rise"
          style={{ animationDelay: "600ms" }}
        >
          <GenerationTimeline events={events} />
        </section>

        <Footer />
      </div>
    </div>
  );
}

// ── Header ─────────────────────────────────────────────────────────────────

interface HeaderProps {
  isRunning: boolean;
  currentGen: number | null;
  currentChampion: string | null;
  finishedCount: number;
  eventCount: number;
  onRun: () => void;
  onStop: () => void;
  onClear: () => void;
}

function Header(props: HeaderProps) {
  const {
    isRunning,
    currentGen,
    currentChampion,
    finishedCount,
    eventCount,
    onRun,
    onStop,
    onClear,
  } = props;

  return (
    <header
      className="rise mb-10 flex flex-col gap-8 lg:mb-12 lg:flex-row lg:items-end lg:justify-between"
      style={{ animationDelay: "60ms" }}
    >
      <div className="relative">
        {currentGen !== null ? (
          <div className="eyebrow mb-3">
            <span>GENERATION {currentGen}</span>
          </div>
        ) : null}

        <h1
          className="font-display leading-[0.92]"
          style={{
            fontSize: "clamp(56px, 8.4vw, 124px)",
            color: "var(--ink)",
            fontVariationSettings:
              '"opsz" 144, "SOFT" 100, "wght" 360',
            letterSpacing: "-0.025em",
          }}
        >
          darwin
        </h1>

        <p
          className="mt-4 max-w-xl text-[14.5px] leading-relaxed"
          style={{ color: "var(--ink-soft)" }}
        >
          An evolutionary loop for chess engines.
          <span style={{ color: "var(--ink-faint)" }}>
            {" "}
            The LLM proposes, two more build, the board decides who survives.
          </span>
        </p>
      </div>

      <div className="flex flex-col items-stretch gap-4 lg:items-end">
        <ChampionBadge
          currentGen={currentGen}
          currentChampion={currentChampion}
          isRunning={isRunning}
        />

        <div className="flex flex-wrap items-center justify-end gap-2">
          <button
            onClick={onStop}
            disabled={!isRunning}
            className="btn btn-ghost"
            title="Cancel the running generation"
          >
            ◼ stop
          </button>
          <button
            onClick={onClear}
            disabled={eventCount === 0 && !isRunning}
            className="btn btn-danger"
            title="Wipe local + server state"
          >
            clear
          </button>
          <button onClick={onRun} className="btn btn-primary">
            {isRunning
              ? "restart generation"
              : finishedCount > 0
                ? "next generation"
                : "run generation"}
          </button>
        </div>
      </div>
    </header>
  );
}

interface ChampionBadgeProps {
  currentGen: number | null;
  currentChampion: string | null;
  isRunning: boolean;
}

/**
 * Headline lockup for the reigning champion.
 *
 * Reads as a small museum placard: a label, a numeric generation in
 * Fraunces, and the champion's engine-name in a typewriter-mono so the
 * hash suffixes don't fight the display face. A firefly pulse under the
 * label tells you whether the dashboard is mid-generation.
 */
function ChampionBadge({
  currentGen,
  currentChampion,
  isRunning,
}: ChampionBadgeProps) {
  return (
    <div
      className="panel relative overflow-hidden px-5 py-3.5"
      style={{ minWidth: 320 }}
      title={currentChampion ?? "no generation yet"}
    >
      <div className="relative z-10 flex items-center gap-4">
        <div className="flex flex-col">
          <span className="eyebrow" style={{ marginBottom: 2 }}>
            {isRunning ? "running" : "champion"}
          </span>
          <div className="flex items-baseline gap-2">
            <span
              className="font-display-tight leading-none"
              style={{ fontSize: 28, color: "var(--ink)" }}
            >
              {currentGen ?? "—"}
            </span>
            <span
              className="text-[10px] uppercase tracking-woodland"
              style={{ color: "var(--ink-faint)" }}
            >
              gen
            </span>
          </div>
        </div>

        <div
          className="h-9 w-px"
          style={{ background: "var(--line-strong)" }}
        />

        <div className="flex min-w-0 flex-1 flex-col">
          <span
            className="font-mono-tab truncate text-[12.5px]"
            style={{ color: "var(--bronze-300)" }}
          >
            {currentChampion ?? "no champion"}
          </span>
          <span
            className="mt-0.5 flex items-center gap-1.5 text-[10px]"
            style={{ color: "var(--ink-faint)" }}
          >
            {isRunning ? (
              <>
                <span className="firefly" />
                <span className="uppercase tracking-woodland">running</span>
              </>
            ) : (
              <span className="uppercase tracking-woodland">idle</span>
            )}
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Footer ─────────────────────────────────────────────────────────────────

function Footer() {
  return (
    <footer
      className="mt-16 flex items-center justify-between border-t pt-5 text-[10.5px] tracking-woodland uppercase"
      style={{
        borderColor: "var(--line)",
        color: "var(--ink-faint)",
      }}
    >
      <span>Built at the Point72 Hackathon</span>
      <span className="font-mono-tab normal-case tracking-normal">
        ws · /api · sqlite
      </span>
    </footer>
  );
}
