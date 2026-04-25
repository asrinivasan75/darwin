/**
 * EnginesEloChart.tsx — Elo trajectory of every engine that's played.
 *
 * Walks every `generation.finished` event, pulls its `ratings` dict
 * (post-tournament Elo for each engine in the cohort), and renders one
 * Recharts line per engine. An engine that only played one generation
 * shows a single dot; one that's stuck around (baseline-v0, repeat
 * incumbents) gets a multi-point line. Names that are unique to a
 * single gen don't pollute the legend with permanently-flat lines —
 * we only colour them on the gens where they actually played.
 *
 * The single-line champion-Elo view in {@link EloChart} is the
 * "headline" — this chart is the "every contender at once" detail view
 * that lets you see which candidates traded blows.
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
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { CubistEvent, GenerationFinished } from "../api/events";

interface EnginesEloChartProps {
  events: CubistEvent[];
}

/** One row of the data array, with a `gen` index plus an Elo per engine. */
type Row = { gen: number } & Record<string, number | undefined>;

// A small palette of distinct colors. Recharts will recycle if there
// are more engines than colors — fine for a hackathon dashboard.
const COLORS = [
  "#60a5fa", // baseline (blue) — first slot, baseline-v0 always lands here
  "#f97316", // orange
  "#10b981", // emerald
  "#a855f7", // purple
  "#ec4899", // pink
  "#eab308", // yellow
  "#06b6d4", // cyan
  "#f43f5e", // rose
  "#84cc16", // lime
  "#8b5cf6", // violet
];

export default function EnginesEloChart({ events }: EnginesEloChartProps) {
  const finished = events.filter(
    (e): e is GenerationFinished => e.type === "generation.finished",
  );

  // Discover every engine that ever has a rating. Stable sort with
  // baseline-v0 first so its color slot is always blue.
  const allEngines = new Set<string>();
  for (const ev of finished) {
    if (!ev.ratings) continue;
    for (const name of Object.keys(ev.ratings)) {
      allEngines.add(name);
    }
  }
  const engines = Array.from(allEngines).sort((a, b) => {
    if (a === "baseline-v0") return -1;
    if (b === "baseline-v0") return 1;
    return a.localeCompare(b);
  });

  // Seed gen 0 with baseline at 1500 (matching seed_baseline.py + the
  // single-line EloChart's origin point) so the chart has an anchor.
  const data: Row[] = [];
  if (engines.length > 0) {
    const seed: Row = { gen: 0 };
    if (allEngines.has("baseline-v0")) seed["baseline-v0"] = 1500;
    data.push(seed);
  }

  for (const ev of finished) {
    const row: Row = { gen: ev.number };
    if (ev.ratings) {
      for (const [name, elo] of Object.entries(ev.ratings)) {
        row[name] = elo;
      }
    }
    data.push(row);
  }

  // Y-axis: ±30 points around the min/max actually present.
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

  // Truncate long candidate names in the legend so it wraps nicely.
  const shortName = (name: string) =>
    name.length > 18 ? name.slice(0, 17) + "…" : name;

  return (
    <div className="bg-gray-800 rounded-lg p-4 flex flex-col">
      <h2 className="text-xs font-semibold tracking-wider text-gray-400 uppercase mb-3">
        All Engines Elo (per cohort)
      </h2>

      {finished.length === 0 ? (
        <p className="text-gray-500 text-sm italic mt-2">
          Waiting for first generation to finish…
        </p>
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <LineChart
            data={data}
            margin={{ top: 4, right: 8, left: -10, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="gen"
              tick={{ fill: "#9ca3af", fontSize: 11 }}
              label={{
                value: "Generation",
                position: "insideBottomRight",
                offset: -4,
                fill: "#6b7280",
                fontSize: 10,
              }}
            />
            <YAxis
              domain={[minElo, maxElo]}
              tick={{ fill: "#9ca3af", fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1f2937",
                border: "1px solid #374151",
                borderRadius: 6,
                color: "#e5e7eb",
                fontSize: 11,
              }}
              labelFormatter={(label) => `Gen ${label}`}
            />
            <Legend
              wrapperStyle={{ fontSize: 10, color: "#9ca3af" }}
              formatter={(value: string) => shortName(value)}
            />
            {engines.map((name, i) => (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                name={name}
                stroke={COLORS[i % COLORS.length]}
                strokeWidth={name === "baseline-v0" ? 2.5 : 1.5}
                // connectNulls=false so an engine that only plays one
                // gen doesn't draw a flat line across the whole chart.
                connectNulls={false}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
                isAnimationActive={true}
                animationDuration={500}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
