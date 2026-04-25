/**
 * EloChart.tsx — champion Elo rating across generations.
 *
 * Standard chess Elo, K=32. Baseline-v0 starts at 1500 (the chess
 * midpoint). Each `generation.finished` event carries an `elo_delta`
 * — the difference between the new champion's post-tournament rating
 * and its pre-tournament rating — and we accumulate that into a line
 * the dashboard plots gen-over-gen.
 *
 * Reading the chart:
 *   - line climbs    new champions are scoring better than expected
 *                     against the existing field
 *   - line stalls    cohort is plateaued, candidates not exceeding
 *                     incumbent's Elo
 *   - line dips       weaker engine took the title via random
 *                     tiebreak / variance — should self-correct in 1-2
 *                     gens as the over-rated champion loses against
 *                     better candidates
 *
 * @listens {GenerationFinished}  - each event appends one data point
 *
 * @module EloChart
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
import type { CubistEvent, GenerationFinished } from "../api/events";

interface EloChartProps {
  events: CubistEvent[];
}

interface EloPoint {
  gen: number;
  elo: number;
  champion: string;
  promoted: boolean;
}

export default function EloChart({ events }: EloChartProps) {
  const finishedEvents = events.filter(
    (e): e is GenerationFinished => e.type === "generation.finished",
  );

  // Gen 0 = the seeded baseline at 1500 — the chess Elo midpoint and
  // the value `seed_baseline.py` writes into EngineRow.
  const data: EloPoint[] = [
    { gen: 0, elo: 1500, champion: "baseline-v0", promoted: false },
  ];

  // Plot each generation's champion's actual post-tournament Elo —
  // *not* a cumulative sum of elo_deltas. The deltas belong to
  // *different* engines from gen to gen (whoever wins that gen), so
  // adding them isn't meaningful. The ratings dict (added in this
  // commit) gives us the exact value to plot directly.
  //
  // Falls back to (prev + elo_delta) for older payloads that pre-date
  // the `ratings` field, so historical event logs still render.
  for (const ev of finishedEvents) {
    let elo: number;
    if (ev.ratings && ev.ratings[ev.new_champion] !== undefined) {
      elo = ev.ratings[ev.new_champion];
    } else {
      // Legacy fallback — assume continuous (incorrect but better than nothing).
      const prev = data[data.length - 1].elo;
      elo = prev + ev.elo_delta;
    }
    data.push({
      gen: ev.number,
      elo: Math.round(elo * 10) / 10,
      champion: ev.new_champion,
      promoted: ev.promoted,
    });
  }

  // Y axis padding: ±30 points around min/max, snapped to nearest 50.
  const eloValues = data.map((d) => d.elo);
  const minElo = Math.floor((Math.min(...eloValues) - 30) / 50) * 50;
  const maxElo = Math.ceil((Math.max(...eloValues) + 30) / 50) * 50;

  return (
    <div className="bg-gray-800 rounded-lg p-4 flex flex-col">
      <h2 className="text-xs font-semibold tracking-wider text-gray-400 uppercase mb-3">
        Champion Elo (cumulative, K=32)
      </h2>

      {finishedEvents.length === 0 ? (
        <p className="text-gray-500 text-sm italic mt-2">
          Waiting for first generation to finish…
        </p>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
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
                fontSize: 12,
              }}
              formatter={(value: number, _name, item) => {
                const p = item.payload as EloPoint;
                return [
                  `${value}`,
                  p.promoted ? `${p.champion} (promoted)` : p.champion,
                ];
              }}
              labelFormatter={(label) => `Gen ${label}`}
            />
            <Line
              type="monotone"
              dataKey="elo"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ fill: "#3b82f6", r: 4 }}
              activeDot={{ r: 6, fill: "#60a5fa" }}
              isAnimationActive={true}
              animationDuration={800}
              animationEasing="ease-out"
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
