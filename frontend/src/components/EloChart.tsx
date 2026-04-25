/**
 * EloChart.tsx — animated Elo rating history line chart.
 *
 * Builds a cumulative Elo series from {@link GenerationFinished} events and
 * renders it with Recharts. Each new data point triggers Recharts' built-in
 * animation so the line visibly "climbs" on screen — this is the dashboard's
 * hero shot for judges.
 *
 * The baseline Elo is 1500 (generation 0, the random/baseline engine).
 * Each subsequent generation's Elo is computed as:
 *   elo[n] = elo[n-1] + generation_finished.elo_delta
 *
 * Non-promoted generations (elo_delta ≤ 0) are still plotted so the chart
 * shows stagnation/regression honestly.
 *
 * @listens {GenerationFinished} - each event appends one data point
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

/** Props accepted by {@link EloChart}. */
interface EloChartProps {
  /** Full accumulated event log from {@link useEventStream}. */
  events: CubistEvent[];
}

/** One data point on the Elo chart: which generation and what Elo rating. */
interface EloPoint {
  gen: number;
  elo: number;
}

/**
 * EloChart — renders an animated line chart of the champion's Elo rating
 * across generations, updating in real time as GenerationFinished events arrive.
 *
 * @param props.events - the full accumulated event log from useEventStream()
 * @returns a Recharts LineChart inside a dark card, or a placeholder if no
 *          generation has finished yet
 */
export default function EloChart({ events }: EloChartProps) {
  const finishedEvents = events.filter(
    (e): e is GenerationFinished => e.type === "generation.finished"
  );

  // Accumulate Elo cumulatively from the baseline of 1500.
  // The baseline (gen 0) is always the first point so the chart has an origin.
  const data: EloPoint[] = [{ gen: 0, elo: 1500 }];

  for (const ev of finishedEvents) {
    const prev = data[data.length - 1].elo;
    data.push({ gen: ev.number, elo: Math.round((prev + ev.elo_delta) * 10) / 10 });
  }

  // Compute a reasonable Y-axis domain so the line uses vertical space well.
  // Add 30-point padding on each side; floor to the nearest 50 below minimum.
  const eloValues = data.map((d) => d.elo);
  const minElo = Math.floor((Math.min(...eloValues) - 30) / 50) * 50;
  const maxElo = Math.ceil((Math.max(...eloValues) + 30) / 50) * 50;

  return (
    <div className="bg-gray-800 rounded-lg p-4 flex flex-col">
      <h2 className="text-xs font-semibold tracking-wider text-gray-400 uppercase mb-3">
        Elo History
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
              formatter={(value: number) => [`${value}`, "Elo"]}
              labelFormatter={(label) => `Gen ${label}`}
            />
            {/*
             * isAnimationActive + animationDuration ensure the line visibly
             * animates each time a new GenerationFinished event arrives.
             */}
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
