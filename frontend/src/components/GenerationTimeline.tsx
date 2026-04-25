/**
 * GenerationTimeline.tsx — historical record of every completed generation.
 *
 * One row per started generation (in arrival order). Each row reads as a
 * field-journal entry: gen number, "from" champion, two strategist
 * questions (truncated), the new champion, Elo delta, and a verdict
 * (PROMOTED / KEPT / running). The baseline source is downloadable from
 * the panel head; each completed champion's source is downloadable from
 * its row.
 *
 * @module GenerationTimeline
 */

import type {
  DarwinEvent,
  GenerationStarted,
  StrategistQuestion,
  GenerationFinished,
} from "../api/events";
import { PanelHead, EmptyPlot } from "./LiveBoards";

interface GenerationTimelineProps {
  events: DarwinEvent[];
}

interface GenRow {
  number: number;
  championBefore: string;
  questions: string[];
  newChampion: string | undefined;
  eloDelta: number | undefined;
  promoted: boolean | undefined;
}

const Q_MAX_LEN = 56;

export default function GenerationTimeline({ events }: GenerationTimelineProps) {
  const rows = buildRows(events);

  return (
    <div className="panel relative flex flex-col p-6">
      <a
        href="/api/engines/baseline-v0/code"
        download="baseline-v0.py"
        className="font-mono-tab absolute right-6 top-6 inline-flex items-center gap-1.5 text-[11px] transition-colors"
        style={{ color: "var(--bronze-300)" }}
        title="Download baseline-v0.py — the seed engine"
      >
        ↓ baseline-v0.py
      </a>
      <PanelHead title="Generations" />

      {rows.length === 0 ? (
        <EmptyPlot
          message="No generations yet."
          hint="Each completed run is recorded here."
        />
      ) : (
        <ol className="mt-5 flex flex-col">
          {rows.map((row, i) => (
            <GenerationEntry
              key={row.number}
              row={row}
              isLast={i === rows.length - 1}
            />
          ))}
        </ol>
      )}
    </div>
  );
}

interface GenerationEntryProps {
  row: GenRow;
  isLast: boolean;
}

/**
 * One entry in the timeline list. Layout:
 *   gen number (display face) │ field-journal block
 * The vertical rule on the left ties consecutive entries together.
 */
function GenerationEntry({ row, isLast }: GenerationEntryProps) {
  const inProgress = row.newChampion === undefined;
  const promoted = row.promoted === true;
  const delta = row.eloDelta;

  const verdictLabel = inProgress
    ? "running"
    : promoted
      ? "promoted"
      : "kept";

  const verdictColor = inProgress
    ? "var(--ink-faint)"
    : promoted
      ? "var(--moss-300)"
      : "var(--bronze-300)";

  return (
    <li className="relative flex gap-4 pb-5">
      {/* Spine */}
      <div className="relative flex w-12 shrink-0 flex-col items-center">
        <span
          className="font-display italic leading-none"
          style={{
            fontSize: 26,
            color: inProgress ? "var(--ink-faint)" : "var(--ink-soft)",
            fontVariationSettings:
              '"opsz" 96, "SOFT" 60, "wght" 360',
          }}
        >
          {row.number}
        </span>
        {!isLast && (
          <span
            aria-hidden
            className="absolute top-7 bottom-0 w-px"
            style={{ background: "var(--line-strong)" }}
          />
        )}
      </div>

      {/* Body */}
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span
            className="font-mono-tab text-[11.5px] truncate"
            style={{ color: "var(--ink-muted)", maxWidth: 200 }}
            title={row.championBefore}
          >
            from {shortName(row.championBefore)}
          </span>

          {!inProgress && (
            <>
              <span style={{ color: "var(--ink-faint)" }}>→</span>
              <span
                className="font-mono-tab inline-flex items-center gap-1.5 text-[11.5px]"
                style={{ color: "var(--bronze-300)" }}
                title={row.newChampion}
              >
                {shortName(row.newChampion!)}
                <a
                  href={`/api/engines/${encodeURIComponent(
                    row.newChampion!,
                  )}/code`}
                  download={`${row.newChampion}.py`}
                  className="text-[10px]"
                  style={{ color: "var(--ink-faint)" }}
                  title={`Download ${row.newChampion}.py`}
                >
                  .py↓
                </a>
              </span>
            </>
          )}

          <span
            className="ml-auto badge"
            style={{
              color: verdictColor,
              borderColor:
                verdictLabel === "promoted"
                  ? "rgba(170,189,149,0.45)"
                  : verdictLabel === "running"
                    ? "var(--line-strong)"
                    : "rgba(220,194,148,0.35)",
              background:
                verdictLabel === "promoted"
                  ? "rgba(63,87,57,0.18)"
                  : verdictLabel === "running"
                    ? "transparent"
                    : "rgba(122,90,55,0.18)",
            }}
          >
            {verdictLabel === "running" && (
              <span className="firefly" aria-hidden />
            )}
            {verdictLabel}
          </span>

          {!inProgress && delta !== undefined && (
            <span
              className="font-mono-tab text-[11.5px]"
              style={{ color: eloDeltaColor(delta) }}
            >
              {formatDelta(delta)}
            </span>
          )}
        </div>

        {/* Strategist's two leading questions, in italics */}
        <div className="mt-2 flex flex-col gap-1">
          {[0, 1].map((i) => (
            <p
              key={i}
              className="font-display text-[12.5px] italic leading-snug"
              style={{
                color: row.questions[i]
                  ? "var(--ink-soft)"
                  : "var(--ink-faint)",
                fontVariationSettings:
                  '"opsz" 24, "SOFT" 60, "wght" 380',
              }}
              title={row.questions[i]}
            >
              {row.questions[i] ? (
                <>
                  <span
                    className="mr-2"
                    style={{ color: "var(--bronze-400)" }}
                  >
                    ❝
                  </span>
                  {truncate(row.questions[i], Q_MAX_LEN)}
                </>
              ) : (
                "—"
              )}
            </p>
          ))}
        </div>
      </div>
    </li>
  );
}

// ── Data assembly (unchanged from the original; kept here for clarity) ─────

function buildRows(events: DarwinEvent[]): GenRow[] {
  const starts = events.filter(
    (e): e is GenerationStarted => e.type === "generation.started",
  );
  const questions = events.filter(
    (e): e is StrategistQuestion => e.type === "strategist.question",
  );
  const finishes = events.filter(
    (e): e is GenerationFinished => e.type === "generation.finished",
  );

  return starts.map((start) => {
    const finish = finishes.find((f) => f.number === start.number);

    const startIdx = events.indexOf(start);
    const nextStart = starts.find((s) => s.number === start.number + 1);
    const endIdx = nextStart ? events.indexOf(nextStart) : events.length;

    const genQuestions = questions
      .filter((q) => {
        const qi = events.indexOf(q);
        return qi > startIdx && qi < endIdx;
      })
      .sort((a, b) => a.index - b.index)
      .map((q) => q.text);

    return {
      number: start.number,
      championBefore: start.champion,
      questions: genQuestions,
      newChampion: finish?.new_champion,
      eloDelta: finish?.elo_delta,
      promoted: finish?.promoted,
    };
  });
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}

function shortName(name: string): string {
  return name.replace(/-[a-z0-9]{3}$/, "").slice(0, 14);
}

function eloDeltaColor(delta: number): string {
  if (delta > 0) return "var(--moss-300)";
  if (delta < 0) return "var(--ember-500)";
  return "var(--ink-muted)";
}

function formatDelta(delta: number): string {
  const sign = delta >= 0 ? "+" : "";
  return `${sign}${delta.toFixed(1)}`;
}
