/**
 * EloChart.tsx — champion Elo rating across generations.
 *
 * Standard chess Elo, K=32. Baseline-v0 starts at 1500 (chess midpoint).
 * Plots each generation's actual post-tournament Elo, falling back to
 * cumulative-delta for legacy payloads.
 *
 * @module EloChart
 */

import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from "recharts";
import type { DarwinEvent, GenerationFinished } from "../api/events";
import { PanelHead, EmptyPlot } from "./LiveBoards";

interface EloChartProps {
  events: DarwinEvent[];
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

  const data: EloPoint[] = [
    { gen: 0, elo: 1500, champion: "baseline-v0", promoted: false },
  ];

  for (const ev of finishedEvents) {
    let elo: number;
    if (ev.ratings && ev.ratings[ev.new_champion] !== undefined) {
      elo = ev.ratings[ev.new_champion];
    } else {
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

  const eloValues = data.map((d) => d.elo);
  const minElo = Math.floor((Math.min(...eloValues) - 30) / 50) * 50;
  const maxElo = Math.ceil((Math.max(...eloValues) + 30) / 50) * 50;
  const peak = Math.max(...eloValues);
  const trough = Math.min(...eloValues);
  const current = data[data.length - 1].elo;

  return (
    <div className="panel flex flex-col p-6">
      <PanelHead title="Champion Elo" meta="K = 32" />

      {finishedEvents.length === 0 ? (
        <EmptyPlot
          message="No generations finished yet."
          hint="The line begins once a generation completes."
        />
      ) : (
        <>
          {/* A small typographic readout alongside the chart */}
          <div className="mt-5 flex items-end gap-6">
            <div>
              <div
                className="text-[9.5px] uppercase tracking-woodland"
                style={{ color: "var(--ink-faint)" }}
              >
                now
              </div>
              <div
                className="font-display-tight leading-none"
                style={{ fontSize: 38, color: "var(--ink)" }}
              >
                {current.toFixed(0)}
              </div>
            </div>
            <Stat label="peak" value={peak.toFixed(0)} />
            <Stat label="trough" value={trough.toFixed(0)} />
            <Stat
              label="generations"
              value={String(finishedEvents.length)}
            />
          </div>

          <div className="mt-4 -mx-2">
            <ResponsiveContainer width="100%" height={210}>
              <ComposedChart
                data={data}
                margin={{ top: 8, right: 12, left: -8, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="elo-area" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="0%"
                      stopColor="var(--moss-500)"
                      stopOpacity={0.45}
                    />
                    <stop
                      offset="100%"
                      stopColor="var(--moss-700)"
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="2 5"
                  stroke="rgba(232,226,211,0.07)"
                  vertical={false}
                />
                <XAxis
                  dataKey="gen"
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
                  formatter={(value: number, _name, item) => {
                    const p = item.payload as EloPoint;
                    return [
                      `${value}`,
                      p.promoted ? `${p.champion} (promoted)` : p.champion,
                    ];
                  }}
                  labelFormatter={(label) => `Generation ${label}`}
                />
                <Area
                  type="monotone"
                  dataKey="elo"
                  stroke="none"
                  fill="url(#elo-area)"
                  isAnimationActive
                  animationDuration={800}
                />
                <Line
                  type="monotone"
                  dataKey="elo"
                  stroke="var(--moss-400)"
                  strokeWidth={2}
                  dot={{
                    fill: "var(--bronze-400)",
                    stroke: "var(--bark-850)",
                    strokeWidth: 2,
                    r: 4,
                  }}
                  activeDot={{
                    r: 6,
                    fill: "var(--bronze-300)",
                    stroke: "var(--bark-850)",
                    strokeWidth: 2,
                  }}
                  isAnimationActive={true}
                  animationDuration={800}
                  animationEasing="ease-out"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div
        className="text-[9.5px] uppercase tracking-woodland"
        style={{ color: "var(--ink-faint)" }}
      >
        {label}
      </div>
      <div
        className="font-mono-tab leading-none"
        style={{ fontSize: 18, color: "var(--ink-soft)", marginTop: 4 }}
      >
        {value}
      </div>
    </div>
  );
}
