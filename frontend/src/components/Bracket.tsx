/**
 * Bracket.tsx — round-robin tournament result matrix for the current generation.
 *
 * Shows all pairings as a 2D table where rows are the "white" engine and
 * columns are the "black" engine. Each cell fills in with W / L / D as
 * {@link GameFinished} events arrive. Diagonal cells (same engine vs itself)
 * are grayed out. The total-points column (wins×1 + draws×0.5) is always
 * shown and updates live.
 *
 * The current champion's row and column are highlighted in blue so judges can
 * immediately see how the baseline performs against challengers.
 *
 * @listens {GenerationStarted}  - identifies the current generation and champion
 * @listens {BuilderCompleted}   - populates the engine roster (ok=true only)
 * @listens {GameFinished}       - fills in result cells
 *
 * @module Bracket
 */

import type {
  CubistEvent,
  GenerationStarted,
  BuilderCompleted,
  GameFinished,
} from "../api/events";

/** Props accepted by {@link Bracket}. */
interface BracketProps {
  /** Full accumulated event log from {@link useEventStream}. */
  events: CubistEvent[];
}

/**
 * Bracket — renders the round-robin result table for the most recent
 * generation, updating cells in real time as GameFinished events arrive.
 *
 * @param props.events - the full accumulated event log from useEventStream()
 * @returns a dark card with an engine-vs-engine result grid
 */
export default function Bracket({ events }: BracketProps) {
  // Find the LATEST generation.started boundary so we can scope every
  // downstream lookup (builders, finished games) to just the current
  // generation. Without this, gen 1's accepted engines and finished
  // games leak into gen 2's bracket — names overlap and points read
  // as 0/garbage.
  let lastBoundary = -1;
  for (let i = 0; i < events.length; i++) {
    const t = events[i].type;
    if (t === "generation.started" || t === "generation.cancelled") {
      lastBoundary = i;
    }
  }
  const currentEvents = events.slice(lastBoundary + 1);

  const genStarts = events.filter(
    (e): e is GenerationStarted => e.type === "generation.started",
  );
  const currentGen = genStarts[genStarts.length - 1] as
    | GenerationStarted
    | undefined;

  if (!currentGen) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h2 className="text-xs font-semibold tracking-wider text-gray-400 uppercase mb-3">
          Tournament Bracket
        </h2>
        <p className="text-gray-500 text-sm italic">
          Waiting for generation to start…
        </p>
      </div>
    );
  }

  // Collect all successfully built engines for the current generation.
  // Scoped to currentEvents (post-boundary) so gen 1's accepted
  // candidates don't appear in gen 2's roster.
  const builderEvents = currentEvents.filter(
    (e): e is BuilderCompleted =>
      e.type === "builder.completed" && e.ok,
  );

  // The roster: champion first, then all ok candidates in question_index order
  const engines: string[] = [
    currentGen.champion,
    ...builderEvents
      .sort((a, b) => a.question_index - b.question_index)
      .map((b) => b.engine_name),
  ];

  // Finished games for this generation — scoped to currentEvents so a
  // finished gen-1 game doesn't fill in a gen-2 cell when the engine
  // names happen to share a prefix.
  const finishedGames = currentEvents.filter(
    (e): e is GameFinished => e.type === "game.finished",
  );

  /**
   * Aggregate the head-to-head matchup between rowEngine and colEngine
   * across both color games. Returns the row engine's score, total
   * games played in this pairing, and a label like "1.5/2".
   *
   * Previous version showed per-color cells (W/L/D for one specific
   * color assignment). That was confusing because the same pair could
   * show "W" in one cell and "D" in the symmetric cell — both are
   * correct (different games, white has advantage), but visually it
   * read as inconsistent. Aggregate is symmetric and matches how
   * round-robin standings are usually presented.
   */
  function pairScore(
    rowEngine: string,
    colEngine: string,
  ): { score: number; played: number } {
    let score = 0;
    let played = 0;
    for (const g of finishedGames) {
      if (g.white === rowEngine && g.black === colEngine) {
        played += 1;
        if (g.result === "1-0") score += 1;
        else if (g.result === "1/2-1/2") score += 0.5;
      } else if (g.white === colEngine && g.black === rowEngine) {
        played += 1;
        if (g.result === "0-1") score += 1;
        else if (g.result === "1/2-1/2") score += 0.5;
      }
    }
    return { score, played };
  }

  /**
   * Format the aggregate score as `n.n/m`. "1.5/2" reads cleanly and
   * shows partial progress when the matchup isn't finished yet.
   */
  function formatPairCell(
    score: number,
    played: number,
  ): string | null {
    if (played === 0) return null;
    // Trim trailing ".0" — "1/2" reads better than "1.0/2"
    const s = score === Math.trunc(score) ? `${score}` : `${score.toFixed(1)}`;
    return `${s}/${played}`;
  }

  /**
   * Computes total tournament points for an engine — sum of all its
   * pair-aggregate scores across the cohort.
   */
  function totalPoints(engine: string): number {
    let points = 0;
    for (const opp of engines) {
      if (opp === engine) continue;
      points += pairScore(engine, opp).score;
    }
    return points;
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 overflow-x-auto">
      <h2 className="text-xs font-semibold tracking-wider text-gray-400 uppercase mb-3">
        Tournament Bracket — Gen {currentGen.number}
      </h2>

      <table className="text-xs border-collapse w-full">
        <thead>
          <tr>
            {/* Empty corner cell */}
            <th className="p-1 text-gray-500 text-left font-normal w-32">
              vs →
            </th>
            {engines.map((eng) => (
              <th
                key={eng}
                className={`p-1 text-center font-mono font-normal max-w-[60px] ${
                  eng === currentGen.champion
                    ? "text-blue-400"
                    : "text-gray-400"
                }`}
                title={eng}
              >
                {shortName(eng)}
              </th>
            ))}
            <th className="p-1 text-center text-gray-400 font-semibold">Pts</th>
          </tr>
        </thead>
        <tbody>
          {engines.map((rowEng) => (
            <tr
              key={rowEng}
              className={
                rowEng === currentGen.champion ? "bg-blue-900/20" : ""
              }
            >
              {/* Row engine label */}
              <td
                className={`p-1 font-mono truncate max-w-[120px] ${
                  rowEng === currentGen.champion
                    ? "text-blue-400"
                    : "text-gray-300"
                }`}
                title={rowEng}
              >
                {shortName(rowEng)}
              </td>

              {/* One cell per column engine */}
              {engines.map((colEng) => {
                if (rowEng === colEng) {
                  // Diagonal — no self-play
                  return (
                    <td
                      key={colEng}
                      className="p-1 text-center bg-gray-700 text-gray-600"
                    >
                      —
                    </td>
                  );
                }
                const { score, played } = pairScore(rowEng, colEng);
                const cellLabel = formatPairCell(score, played);
                // Color: green if won the matchup outright (>50%),
                // red if lost (<50%), yellow if exactly tied (50%),
                // dim gray if no games played yet.
                let colorClass = "text-gray-600";
                if (played > 0) {
                  if (score > played / 2) colorClass = "text-green-400";
                  else if (score < played / 2) colorClass = "text-red-400";
                  else colorClass = "text-yellow-400";
                }
                return (
                  <td
                    key={colEng}
                    className={`p-1 text-center font-semibold ${colorClass}`}
                  >
                    {cellLabel ?? <span className="text-gray-600">·</span>}
                  </td>
                );
              })}

              {/* Total points */}
              <td className="p-1 text-center font-bold text-gray-200">
                {totalPoints(rowEng)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Abbreviates an engine name to fit in a narrow table column.
 * "gen1-book-a3f" → "g1-book", "baseline-v0" → "base-v0"
 */
function shortName(name: string): string {
  // Trim trailing hash-like suffix (e.g. "-a3f") to keep names readable
  return name.replace(/-[a-z0-9]{3}$/, "").slice(0, 10);
}

// resultClass removed — replaced by inline color logic in the cell
// renderer when bracket switched from per-color W/L/D cells to
// pair-aggregate score cells.
