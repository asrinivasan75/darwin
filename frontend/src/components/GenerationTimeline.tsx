/**
 * GenerationTimeline.tsx — historical record of every completed generation.
 *
 * Groups events by generation number and renders one row per generation showing:
 *   - Generation number
 *   - Champion before the generation started
 *   - Up to 2 strategist questions (abbreviated for readability)
 *   - The new champion after the tournament
 *   - Elo delta (positive = improvement)
 *   - A PROMOTED / KEPT / in-progress badge
 *
 * The in-progress generation (the one started but not yet finished) appears as
 * the last row with "…" placeholders so judges can see the pipeline is live.
 *
 * @listens {GenerationStarted}  - opens a new row in the timeline
 * @listens {StrategistQuestion} - populates the question columns per generation
 * @listens {GenerationFinished} - closes the row with result, elo delta, badge
 *
 * @module GenerationTimeline
 */

import type {
  CubistEvent,
  GenerationStarted,
  StrategistQuestion,
  GenerationFinished,
} from "../api/events";

/** Props accepted by {@link GenerationTimeline}. */
interface GenerationTimelineProps {
  /** Full accumulated event log from {@link useEventStream}. */
  events: CubistEvent[];
}

/** All data we need to render one timeline row, assembled from multiple events. */
interface GenRow {
  number: number;
  championBefore: string;
  /** Up to 2 strategist questions for this generation, may be shorter if still arriving. */
  questions: string[];
  /** Set once GenerationFinished arrives; undefined while in progress. */
  newChampion: string | undefined;
  eloDelta: number | undefined;
  promoted: boolean | undefined;
}

/** Max chars to show per question cell before truncating. */
const Q_MAX_LEN = 38;

/**
 * GenerationTimeline — scrollable table showing every generation that has
 * started, ordered newest-last, updating live as events arrive.
 *
 * @param props.events - the full accumulated event log from useEventStream()
 * @returns a dark card with one generation per row
 */
export default function GenerationTimeline({ events }: GenerationTimelineProps) {
  const rows = buildRows(events);

  if (rows.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <TimelineHeader />
        <p className="text-gray-500 text-sm italic">
          Waiting for first generation…
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 overflow-x-auto">
      <TimelineHeader />

      <table className="text-xs border-collapse w-full">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="p-1.5 text-left text-gray-500 font-normal w-8">Gen</th>
            <th className="p-1.5 text-left text-gray-500 font-normal">Champion Before</th>
            <th className="p-1.5 text-left text-gray-500 font-normal" colSpan={2}>
              Questions (2)
            </th>
            <th className="p-1.5 text-left text-gray-500 font-normal">Winner</th>
            <th className="p-1.5 text-right text-gray-500 font-normal">Elo Δ</th>
            <th className="p-1.5 text-center text-gray-500 font-normal">Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <GenerationRow key={row.number} row={row} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Internal sub-components ──────────────────────────────────────────────────

/**
 * Header bar for the timeline panel: title plus a download link for the
 * baseline engine source. Baseline is the seed file every strategist
 * prompt builds off, so judges/operators reading the timeline often want
 * to inspect or save it. Surfacing it here keeps it reachable even when
 * the table is empty (no generations run yet).
 */
function TimelineHeader() {
  return (
    <div className="flex items-center justify-between mb-3">
      <h2 className="text-xs font-semibold tracking-wider text-gray-400 uppercase">
        Generation Timeline
      </h2>
      <a
        href="/api/engines/baseline-v0/code"
        download="baseline-v0.py"
        className="text-blue-400 hover:text-blue-300 text-[10px] font-mono"
        title="Download baseline-v0.py — the seed engine the strategist prompts off"
      >
        ↓ baseline-v0.py
      </a>
    </div>
  );
}

/**
 * Renders one generation row in the timeline table.
 *
 * @param props.row - assembled generation data (may be partially filled if in progress)
 */
function GenerationRow({ row }: { row: GenRow }) {
  const inProgress = row.newChampion === undefined;

  return (
    <tr className="border-b border-gray-700/50 hover:bg-gray-700/20 transition-colors">
      {/* Generation number */}
      <td className="p-1.5 text-gray-400 font-mono">{row.number}</td>

      {/* Champion before this generation */}
      <td
        className="p-1.5 text-gray-300 font-mono truncate max-w-[100px]"
        title={row.championBefore}
      >
        {shortName(row.championBefore)}
      </td>

      {/* Up to 2 question cells, padded with "—" if still arriving */}
      {Array.from({ length: 2 }).map((_, i) => (
        <td
          key={i}
          className="p-1.5 text-gray-400 max-w-[90px] truncate"
          title={row.questions[i]}
        >
          {row.questions[i]
            ? truncate(row.questions[i], Q_MAX_LEN)
            : <span className="text-gray-600">—</span>}
        </td>
      ))}

      {/* New champion after the generation, with a download link to its
          .py source. Hidden while in-progress since no champion is set
          yet. */}
      <td
        className="p-1.5 font-mono truncate max-w-[140px] text-gray-300"
        title={row.newChampion}
      >
        {inProgress ? (
          <span className="text-gray-500">…</span>
        ) : (
          <span className="inline-flex items-center gap-1.5">
            <span className="truncate">{shortName(row.newChampion!)}</span>
            <a
              href={`/api/engines/${encodeURIComponent(row.newChampion!)}/code`}
              download={`${row.newChampion}.py`}
              className="text-blue-400 hover:text-blue-300 text-[10px] font-mono shrink-0"
              title={`Download ${row.newChampion}.py`}
            >
              .py↓
            </a>
          </span>
        )}
      </td>

      {/* Elo delta, colour-coded: positive = green, negative = red */}
      <td className={`p-1.5 text-right font-mono ${eloDeltaClass(row.eloDelta)}`}>
        {inProgress
          ? <span className="text-gray-500">…</span>
          : formatDelta(row.eloDelta!)}
      </td>

      {/* Status badge */}
      <td className="p-1.5 text-center">
        {inProgress ? (
          <span className="text-gray-500 text-xs italic">running</span>
        ) : row.promoted ? (
          <span className="bg-green-700 text-white text-xs px-1.5 py-0.5 rounded font-semibold">
            ↑ PROMOTED
          </span>
        ) : (
          <span className="bg-gray-600 text-gray-300 text-xs px-1.5 py-0.5 rounded">
            = KEPT
          </span>
        )}
      </td>
    </tr>
  );
}

// ── Data assembly ────────────────────────────────────────────────────────────

/**
 * Assembles one {@link GenRow} per started generation from the flat event log.
 *
 * Groups by generation number by scanning for GenerationStarted events, then
 * attaches questions and the optional GenerationFinished result.
 */
function buildRows(events: CubistEvent[]): GenRow[] {
  const starts = events.filter(
    (e): e is GenerationStarted => e.type === "generation.started"
  );
  const questions = events.filter(
    (e): e is StrategistQuestion => e.type === "strategist.question"
  );
  const finishes = events.filter(
    (e): e is GenerationFinished => e.type === "generation.finished"
  );

  return starts.map((start) => {
    const finish = finishes.find((f) => f.number === start.number);

    // Questions are emitted after generation.started with no explicit gen tag,
    // so we bucket them by arrival position: questions that arrived after the
    // start of this generation and before the start of the next one.
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

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Truncates a string to `max` chars, appending "…" if needed. */
function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}

/** Shortens an engine name for narrow columns. */
function shortName(name: string): string {
  return name.replace(/-[a-z0-9]{3}$/, "").slice(0, 12);
}

/** Returns the Tailwind text colour for an Elo delta value. */
function eloDeltaClass(delta: number | undefined): string {
  if (delta === undefined) return "text-gray-500";
  if (delta > 0) return "text-green-400";
  if (delta < 0) return "text-red-400";
  return "text-gray-400";
}

/** Formats an Elo delta with a leading sign. */
function formatDelta(delta: number): string {
  const sign = delta >= 0 ? "+" : "";
  return `${sign}${delta.toFixed(1)}`;
}
