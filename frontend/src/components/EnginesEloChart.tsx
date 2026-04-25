/**
 * EnginesEloChart.tsx — Elo trajectory of every engine that's played.
 *
 * Top-N engines (by peak Elo) get a line; baseline-v0 always claims the
 * primary slot.  See the original comments for forward-fill semantics.
 *
 * @module EnginesEloChart
 */

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { DarwinEvent, GenerationFinished } from "../api/events";
import { PanelHead, EmptyPlot } from "./LiveBoards";

interface EnginesEloChartProps {
  events: DarwinEvent[];
}

type Row = { gen: number } & Record<string, number | undefined>;

/**
 * An earthy palette: baseline-v0 owns the warm bronze leadership colour,
 * everything else takes a moss/lichen/sandstone shade so no engine
 * "outshouts" the others. Saturation is deliberately restrained.
 */
const COLORS = [
  "#c9a876", // baseline — warm bronze, leads
  "#88a474", // moss
  "#dcc294", // sandstone
  "#7d8aa0", // slate
  "#aabd95", // lichen
  "#b58957", // amber bronze
  "#6b8a5c", // forest
  "#c5705f", // rust
  "#9c7647", // walnut
  "#54704a", // deep moss
];

const TOP_N = 8;

export default function EnginesEloChart({ events }: EnginesEloChartProps) {
  const finished = events.filter(
    (e): e is GenerationFinished => e.type === "generation.finished",
  );

  const series: Record<string, Array<{ gen: number; elo: number }>> = {};
  series["baseline-v0"] = [{ gen: 0, elo: 1500 }];

  for (const ev of finished) {
    if (!ev.ratings) continue;
    for (const [name, elo] of Object.entries(ev.ratings)) {
      if (!series[name]) series[name] = [];
      series[name].push({ gen: ev.number, elo });
    }
  }

  const peakElo: Array<[string, number]> = Object.entries(series).map(
    ([name, points]) => [name, Math.max(...points.map((p) => p.elo))],
  );
  peakElo.sort((a, b) => b[1] - a[1]);
  const topEngines = new Set(peakElo.slice(0, TOP_N).map(([name]) => name));

  const allGens = new Set<number>();
  for (const name of topEngines) {
    for (const p of series[name]) allGens.add(p.gen);
  }
  const sortedGens = Array.from(allGens).sort((a, b) => a - b);

  const data: Row[] = sortedGens.map((gen) => {
    const row: Row = { gen };
    for (const name of topEngines) {
      const point = series[name].find((p) => p.gen === gen);
      if (point) row[name] = point.elo;
    }
    return row;
  });

  const peakLookup = new Map(peakElo);
  const engines = Array.from(topEngines).sort((a, b) => {
    if (a === "baseline-v0") return -1;
    if (b === "baseline-v0") return 1;
    return (peakLookup.get(b) ?? 0) - (peakLookup.get(a) ?? 0);
  });

  const eloValues: number[] = [];
  for (const row of data) {
    for (const [k, v] of Object.entries(row)) {
      if (k === "gen") continue;
      if (typeof v === "number") eloValues.push(v);
    }
  }
  const minElo =
    eloValues.length > 0
      ? Math.floor((Math.min(...eloValues) - 30) / 50) * 50
      : 1450;
  const maxElo =
    eloValues.length > 0
      ? Math.ceil((Math.max(...eloValues) + 30) / 50) * 50
      : 1550;

  const shortName = (name: string) =>
    name.length > 20 ? name.slice(0, 19) + "…" : name;

  return (
    <div className="panel flex flex-col p-6">
      <PanelHead
        title="All engines"
        meta={
          finished.length === 0
            ? "no cohorts complete"
            : `top ${Math.min(TOP_N, engines.length)} by peak`
        }
      />

      {finished.length === 0 ? (
        <EmptyPlot
          message="No generations finished yet."
          hint="Each finished generation adds its cohort."
        />
      ) : (
        <>
          {/* Custom legend — typographic, not a recharts dump.
              Each row pairs an engine name with its peak Elo. */}
          <ul className="mt-5 grid grid-cols-2 gap-x-6 gap-y-1.5 sm:grid-cols-3 xl:grid-cols-4">
            {engines.map((name, i) => {
              const peak = peakLookup.get(name) ?? 0;
              return (
                <li
                  key={name}
                  className="flex items-center gap-2.5 text-[11.5px]"
                  title={`${name} · peak ${peak.toFixed(0)}`}
                >
                  <span
                    aria-hidden
                    className="inline-block h-[2px] w-5 shrink-0 rounded"
                    style={{
                      background: COLORS[i % COLORS.length],
                      boxShadow: `0 0 0 1px ${COLORS[i % COLORS.length]}33`,
                    }}
                  />
                  <span
                    className="font-mono-tab truncate"
                    style={{ color: "var(--ink-soft)" }}
                  >
                    {shortName(name)}
                  </span>
                  <span
                    className="font-mono-tab ml-auto"
                    style={{ color: "var(--ink-faint)" }}
                  >
                    {peak.toFixed(0)}
                  </span>
                </li>
              );
            })}
          </ul>

          <div className="mt-4 -mx-2">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart
                data={data}
                margin={{ top: 8, right: 14, left: -8, bottom: 0 }}
              >
                <CartesianGrid
                  strokeDasharray="2 5"
                  stroke="rgba(232,226,211,0.06)"
                  vertical={false}
                />
                <XAxis
                  dataKey="gen"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  allowDecimals={false}
                  tick={{
                    fill: "var(--ink-faint)",
                    fontSize: 10.5,
                    fontFamily: "IBM Plex Mono",
                  }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(232,226,211,0.1)" }}
                />
                <YAxis
                  domain={[minElo, maxElo]}
                  tick={{
                    fill: "var(--ink-faint)",
                    fontSize: 10.5,
                    fontFamily: "IBM Plex Mono",
                  }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  cursor={{
                    stroke: "rgba(201,168,118,0.3)",
                    strokeDasharray: "3 3",
                  }}
                  contentStyle={{
                    background:
                      "linear-gradient(180deg, rgba(40,48,42,0.97), rgba(28,34,30,0.97))",
                    border: "1px solid rgba(232,226,211,0.13)",
                    borderRadius: 8,
                    color: "var(--ink)",
                    fontSize: 11.5,
                    fontFamily: "Instrument Sans",
                    boxShadow: "0 12px 28px -16px rgba(0,0,0,0.6)",
                  }}
                  labelFormatter={(label) => `Generation ${label}`}
                />
                {engines.map((name, i) => (
                  <Line
                    key={name}
                    type="linear"
                    dataKey={name}
                    name={name}
                    stroke={COLORS[i % COLORS.length]}
                    strokeWidth={name === "baseline-v0" ? 2.6 : 1.5}
                    strokeOpacity={name === "baseline-v0" ? 1 : 0.85}
                    connectNulls={true}
                    dot={{ r: 2.5, strokeWidth: 0 }}
                    activeDot={{
                      r: 5,
                      stroke: "var(--bark-850)",
                      strokeWidth: 2,
                    }}
                    isAnimationActive={true}
                    animationDuration={500}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
