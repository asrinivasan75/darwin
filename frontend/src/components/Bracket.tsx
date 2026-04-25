/**
 * Bracket.tsx — round-robin tournament result matrix for the current generation.
 *
 * Aggregates head-to-head pair scores (white-vs-black + black-vs-white) into
 * a single cell so the matrix is symmetric. The reigning champion's row and
 * column are tinted bronze.
 *
 * @module Bracket
 */

import type {
  DarwinEvent,
  GenerationStarted,
  BuilderCompleted,
  GameFinished,
} from "../api/events";
import { PanelHead, EmptyPlot } from "./LiveBoards";

interface BracketProps {
  events: DarwinEvent[];
}

export default function Bracket({ events }: BracketProps) {
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
      <div className="panel flex flex-col p-6">
        <PanelHead title="Tournament" />
        <EmptyPlot
          message="No tournament running."
          hint="Pairings appear once the strategist's questions become engines."
        />
      </div>
    );
  }

  const builderEvents = currentEvents.filter(
    (e): e is BuilderCompleted => e.type === "builder.completed" && e.ok,
  );

  const engines: string[] = [
    currentGen.champion,
    ...builderEvents
      .sort((a, b) => a.question_index - b.question_index)
      .map((b) => b.engine_name),
  ];

  const finishedGames = currentEvents.filter(
    (e): e is GameFinished => e.type === "game.finished",
  );

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

  function formatPairCell(score: number, played: number): string | null {
    if (played === 0) return null;
    const s = score === Math.trunc(score) ? `${score}` : `${score.toFixed(1)}`;
    return `${s}/${played}`;
  }

  function totalPoints(engine: string): number {
    let points = 0;
    for (const opp of engines) {
      if (opp === engine) continue;
      points += pairScore(engine, opp).score;
    }
    return points;
  }

  // For total-points "tally" bars
  const maxTotal = Math.max(
    1,
    ...engines.map((e) => totalPoints(e)),
  );

  return (
    <div className="panel flex flex-col overflow-hidden p-6">
      <PanelHead
        title={`Tournament · gen ${currentGen.number}`}
        meta={`${engines.length} engines`}
      />

      <div className="mt-5 overflow-x-auto">
        <table className="w-full border-separate border-spacing-y-1 text-[11.5px]">
          <thead>
            <tr>
              <th
                className="px-2 pb-3 pt-1 text-left text-[10px] font-normal uppercase tracking-woodland"
                style={{ color: "var(--ink-faint)", width: 150 }}
              >
                vs ↓
              </th>
              {engines.map((eng) => (
                <th
                  key={eng}
                  className="px-1 pb-3 pt-1 text-center font-normal"
                  style={{
                    color:
                      eng === currentGen.champion
                        ? "var(--bronze-300)"
                        : "var(--ink-muted)",
                  }}
                  title={eng}
                >
                  <span
                    className="font-mono-tab text-[10.5px] uppercase tracking-[0.06em]"
                    style={{
                      writingMode: "vertical-rl",
                      transform: "rotate(180deg)",
                      display: "inline-block",
                      maxHeight: 70,
                      whiteSpace: "nowrap",
                    }}
                  >
                    {shortName(eng)}
                  </span>
                </th>
              ))}
              <th
                className="px-2 pb-3 pt-1 text-right text-[10px] font-normal uppercase tracking-woodland"
                style={{ color: "var(--ink-faint)" }}
              >
                pts
              </th>
            </tr>
          </thead>
          <tbody>
            {engines.map((rowEng) => {
              const isChamp = rowEng === currentGen.champion;
              const total = totalPoints(rowEng);
              return (
                <tr
                  key={rowEng}
                  style={{
                    background: isChamp
                      ? "linear-gradient(90deg, rgba(201,168,118,0.06), transparent 70%)"
                      : "transparent",
                  }}
                >
                  <td
                    className="font-mono-tab truncate rounded-l px-2 py-1.5 text-[11.5px]"
                    style={{
                      color: isChamp
                        ? "var(--bronze-300)"
                        : "var(--ink-soft)",
                      maxWidth: 150,
                    }}
                    title={rowEng}
                  >
                    {isChamp && (
                      <span
                        aria-hidden
                        className="mr-1.5 inline-block"
                        style={{ color: "var(--bronze-400)" }}
                      >
                        ✦
                      </span>
                    )}
                    {shortName(rowEng)}
                  </td>

                  {engines.map((colEng) => {
                    if (rowEng === colEng) {
                      return (
                        <td
                          key={colEng}
                          className="px-1 py-1.5 text-center font-mono-tab"
                          style={{
                            background: "rgba(232,226,211,0.025)",
                            color: "var(--ink-faint)",
                          }}
                        >
                          ·
                        </td>
                      );
                    }
                    const { score, played } = pairScore(rowEng, colEng);
                    const cellLabel = formatPairCell(score, played);

                    let color = "var(--ink-faint)";
                    if (played > 0) {
                      if (score > played / 2) color = "var(--moss-300)";
                      else if (score < played / 2) color = "var(--ember-500)";
                      else color = "var(--bronze-300)";
                    }

                    return (
                      <td
                        key={colEng}
                        className="font-mono-tab px-1 py-1.5 text-center"
                        style={{ color }}
                      >
                        {cellLabel ?? (
                          <span style={{ color: "var(--ink-faint)" }}>·</span>
                        )}
                      </td>
                    );
                  })}

                  <td
                    className="rounded-r px-2 py-1.5 text-right"
                    style={{ minWidth: 80 }}
                  >
                    <div className="flex items-center justify-end gap-2">
                      <div
                        className="h-1 w-12 overflow-hidden rounded-full"
                        style={{ background: "rgba(232,226,211,0.06)" }}
                        aria-hidden
                      >
                        <div
                          className="h-full origin-left"
                          style={{
                            width: `${(total / maxTotal) * 100}%`,
                            background: isChamp
                              ? "linear-gradient(90deg, var(--bronze-500), var(--bronze-300))"
                              : "linear-gradient(90deg, var(--moss-600), var(--moss-400))",
                            transform: "scaleX(1)",
                            animation: "tally 0.7s ease-out",
                          }}
                        />
                      </div>
                      <span
                        className="font-display-tight"
                        style={{
                          fontSize: 15,
                          color: isChamp
                            ? "var(--bronze-300)"
                            : "var(--ink)",
                        }}
                      >
                        {total}
                      </span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function shortName(name: string): string {
  return name.replace(/-[a-z0-9]{3}$/, "").slice(0, 12);
}
