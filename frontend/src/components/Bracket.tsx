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
  // Find the most recently started generation
  const genStarts = events.filter(
    (e): e is GenerationStarted => e.type === "generation.started"
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
  // BuilderCompleted.engine_name gives us the candidate names.
  const builderEvents = events.filter(
    (e): e is BuilderCompleted =>
      e.type === "builder.completed" && e.ok
  );

  // The roster: champion first, then all ok candidates in question_index order
  const engines: string[] = [
    currentGen.champion,
    ...builderEvents
      .sort((a, b) => a.question_index - b.question_index)
      .map((b) => b.engine_name),
  ];

  // Finished games for this generation — we don't have a generation tag on
  // GameFinished, so we include all finished games and let the matrix filter
  // by name. This is safe because engine names embed the generation number.
  const finishedGames = events.filter(
    (e): e is GameFinished => e.type === "game.finished"
  );

  /**
   * Returns the result symbol for the pairing (rowEngine as White, colEngine
   * as Black). Returns undefined if the game hasn't finished yet.
   *
   * We check both orderings because the round-robin may flip colours.
   */
  function cellResult(
    rowEngine: string,
    colEngine: string
  ): string | undefined {
    // White = row, Black = col
    const asWhite = finishedGames.find(
      (g) => g.white === rowEngine && g.black === colEngine
    );
    if (asWhite) {
      if (asWhite.result === "1-0") return "W";
      if (asWhite.result === "0-1") return "L";
      return "D";
    }
    // White = col, Black = row (same pairing, colours swapped)
    const asBlack = finishedGames.find(
      (g) => g.white === colEngine && g.black === rowEngine
    );
    if (asBlack) {
      if (asBlack.result === "1-0") return "L";
      if (asBlack.result === "0-1") return "W";
      return "D";
    }
    return undefined;
  }

  /**
   * Computes total tournament points for an engine.
   * Scoring: wins×1, draws×0.5, losses×0
   */
  function totalPoints(engine: string): number {
    let points = 0;
    for (const opp of engines) {
      if (opp === engine) continue;
      const r = cellResult(engine, opp);
      if (r === "W") points += 1;
      if (r === "D") points += 0.5;
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
                const result = cellResult(rowEng, colEng);
                return (
                  <td
                    key={colEng}
                    className={`p-1 text-center font-semibold ${resultClass(result)}`}
                  >
                    {result ?? (
                      <span className="text-gray-600">·</span>
                    )}
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

/** Returns the Tailwind text colour class for W / L / D / undefined. */
function resultClass(result: string | undefined): string {
  if (result === "W") return "text-green-400";
  if (result === "L") return "text-red-400";
  if (result === "D") return "text-yellow-400";
  return "text-gray-600";
}
