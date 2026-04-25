/**
 * EngineAnalytics.tsx - persisted analytics across Darwin generations.
 *
 * This page intentionally derives every metric from REST-backed persisted
 * state, not local WebSocket history, so it survives reloads and browsers.
 */

import { useEffect, useMemo, useState } from "react";
import {
  fetchEngineCode,
  fetchEngines,
  fetchGames,
  fetchGenerations,
} from "../api/client";
import { EmptyPlot, PanelHead } from "./LiveBoards";

type LoadState = "loading" | "ready" | "error";
type Category = "prompt" | "search" | "book" | "evaluation" | "sampling" | "unknown";

interface EngineRow {
  name: string;
  generation: number;
  parent_name: string | null;
  code_path: string;
  elo: number;
  created_at: string;
}

interface GameRow {
  generation: number;
  white_name: string;
  black_name: string;
  result: "1-0" | "0-1" | "1/2-1/2" | "*";
  termination: string;
}

interface GenerationRow {
  number: number;
  champion_before: string;
  champion_after: string;
  strategist_questions_json: string;
  finished_at: string | null;
}

interface Question {
  category: Category;
  text: string;
}

interface ScoreLine {
  played: number;
  wins: number;
  draws: number;
  losses: number;
  score: number;
}

interface EngineMetric extends ScoreLine {
  name: string;
  generation: number;
  category: Category;
  elo: number;
  winRate: number;
  isChampion: boolean;
}

interface CategoryMetric {
  category: Category;
  tried: number;
  accepted: number;
  champions: number;
  avgElo: number;
  avgWinRate: number;
  promotionRate: number;
  effectiveness: number;
}

interface PromptMetric {
  generation: number;
  category: Category;
  text: string;
  accepted: number;
  bestEngine: string;
  winRate: number;
  elo: number;
  promoted: boolean;
  effectiveness: number;
}

interface GenerationMetric {
  number: number;
  before: string;
  after: string;
  promoted: boolean;
  winningCategory: Category;
  candidates: number;
  ratingGap: number;
}

interface FeatureMetric {
  key: string;
  label: string;
  engines: number;
  champions: number;
  avgWinRate: number;
  avgElo: number;
}

interface Overview {
  totalGenerations: number;
  totalEngines: number;
  totalGames: number;
  promotionRate: number;
  bestChampionElo: number;
  currentChampion: string;
}

interface Analytics {
  overview: Overview;
  categories: CategoryMetric[];
  prompts: PromptMetric[];
  generations: GenerationMetric[];
  engines: EngineMetric[];
  features: FeatureMetric[];
  insights: Array<{ label: string; value: string; detail: string }>;
}

const CATEGORIES: Category[] = ["prompt", "search", "book", "evaluation", "sampling"];

const FEATURE_RULES = [
  { key: "alpha-beta", label: "Alpha-beta / minimax", re: /\balpha|beta|minimax|negamax/i },
  { key: "transposition", label: "Transposition / cache", re: /transposition|zobrist|memo|cache|tt\b/i },
  { key: "opening-book", label: "Opening book", re: /opening|book|fen_prefix|book_move/i },
  { key: "piece-square", label: "Piece-square tables", re: /piece.?square|pst|square_table|piece_square/i },
  { key: "mobility", label: "Mobility scoring", re: /mobility|legal_moves|move_count/i },
  { key: "king-safety", label: "King safety", re: /king.?safety|king_zone|attackers|chebyshev/i },
  { key: "pawn-structure", label: "Pawn structure", re: /pawn.?structure|doubled|isolated|passed pawn/i },
  { key: "sampling", label: "Sampling / MCTS", re: /monte carlo|mcts|rollout|ucb|sampling|stochastic/i },
  { key: "runtime-llm", label: "Runtime LLM usage", re: /complete_text|complete\(|darwin\.llm|anthropic|google/i },
];

export default function EngineAnalytics() {
  const [state, setState] = useState<LoadState>("loading");
  const [engines, setEngines] = useState<EngineRow[]>([]);
  const [generations, setGenerations] = useState<GenerationRow[]>([]);
  const [games, setGames] = useState<GameRow[]>([]);
  const [sources, setSources] = useState<Record<string, string | null>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setState("loading");
      setError(null);
      try {
        const [engineRaw, generationRaw, gameRaw] = await Promise.all([
          fetchEngines(),
          fetchGenerations(),
          fetchGames(),
        ]);

        const engineRows = parseEngines(engineRaw);
        const generationRows = parseGenerations(generationRaw);
        const gameRows = parseGames(gameRaw);

        const sourceEntries = await Promise.all(
          engineRows.map(async (engine) => {
            try {
              return [engine.name, await fetchEngineCode(engine.name)] as const;
            } catch {
              return [engine.name, null] as const;
            }
          }),
        );

        if (cancelled) return;
        setEngines(engineRows);
        setGenerations(generationRows);
        setGames(gameRows);
        setSources(Object.fromEntries(sourceEntries));
        setState("ready");
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Could not load analytics data.");
        setState("error");
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const analytics = useMemo(
    () => buildAnalytics(engines, generations, games, sources),
    [engines, generations, games, sources],
  );

  if (state === "loading") {
    return (
      <div className="rise" style={{ animationDelay: "240ms" }}>
        <div className="panel p-6">
          <PanelHead title="Analytics" meta="loading" />
          <EmptyPlot message="Loading analytics." hint="Reading generations, games, engines, and source." />
        </div>
      </div>
    );
  }

  if (state === "error") {
    return (
      <div className="rise" style={{ animationDelay: "240ms" }}>
        <div className="panel p-6">
          <PanelHead title="Analytics" meta="error" />
          <EmptyPlot message="Could not load analytics." hint={error ?? undefined} />
        </div>
      </div>
    );
  }

  if (analytics.overview.totalGenerations === 0) {
    return (
      <div className="rise" style={{ animationDelay: "240ms" }}>
        <div className="panel p-6">
          <PanelHead title="Analytics" meta="persisted DB" />
          <EmptyPlot
            message="No completed generations yet."
            hint="Run a generation to analyze prompts, categories, and engine changes."
          />
        </div>
      </div>
    );
  }

  return (
    <div className="rise flex flex-col gap-6" style={{ animationDelay: "240ms" }}>
      <OverviewGrid overview={analytics.overview} />
      <InsightPanel insights={analytics.insights} />

      <section className="grid grid-cols-1 gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <CategoryPanel categories={analytics.categories} />
        <FeaturePanel features={analytics.features} />
      </section>

      <section className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <PromptPanel prompts={analytics.prompts} />
        <GenerationPanel generations={analytics.generations} />
      </section>

      <EngineLeaderboard engines={analytics.engines} />
    </div>
  );
}

function OverviewGrid({ overview }: { overview: Overview }) {
  return (
    <section className="grid grid-cols-2 gap-4 lg:grid-cols-6">
      <StatTile label="generations" value={String(overview.totalGenerations)} />
      <StatTile label="engines" value={String(overview.totalEngines)} />
      <StatTile label="games" value={String(overview.totalGames)} />
      <StatTile label="promotion" value={`${Math.round(overview.promotionRate * 100)}%`} />
      <StatTile label="best champ elo" value={overview.bestChampionElo.toFixed(0)} />
      <StatTile label="current" value={shortName(overview.currentChampion, 16)} title={overview.currentChampion} />
    </section>
  );
}

function StatTile({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="panel px-4 py-3" title={title}>
      <div
        className="text-[9.5px] uppercase tracking-woodland"
        style={{ color: "var(--ink-faint)" }}
      >
        {label}
      </div>
      <div
        className="font-display-tight mt-1 truncate leading-none"
        style={{ color: "var(--ink)", fontSize: 25 }}
      >
        {value}
      </div>
    </div>
  );
}

function InsightPanel({ insights }: { insights: Analytics["insights"] }) {
  return (
    <section className="panel p-6">
      <PanelHead title="Insight callouts" meta="inferred from persisted data" />
      <div className="mt-5 grid grid-cols-1 gap-3 lg:grid-cols-4">
        {insights.map((insight) => (
          <div
            key={insight.label}
            className="rounded-md p-4"
            style={{
              border: "1px solid var(--line)",
              background: "rgba(15,19,17,0.34)",
            }}
          >
            <div
              className="text-[9.5px] uppercase tracking-woodland"
              style={{ color: "var(--ink-faint)" }}
            >
              {insight.label}
            </div>
            <div
              className="font-display-tight mt-1 truncate"
              style={{ color: "var(--bronze-300)", fontSize: 22 }}
              title={insight.value}
            >
              {insight.value}
            </div>
            <p className="mt-2 text-[12px] leading-snug" style={{ color: "var(--ink-muted)" }}>
              {insight.detail}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}

function CategoryPanel({ categories }: { categories: CategoryMetric[] }) {
  const maxEffectiveness = Math.max(1, ...categories.map((c) => c.effectiveness));
  return (
    <section className="panel p-6">
      <PanelHead title="Category effectiveness" meta="prompts to tournament outcomes" />
      <div className="mt-5 overflow-x-auto">
        <table className="w-full border-separate border-spacing-y-1 text-[11.5px]">
          <thead>
            <tr style={{ color: "var(--ink-faint)" }}>
              <HeaderCell label="category" />
              <HeaderCell label="tried" align="right" />
              <HeaderCell label="accepted" align="right" />
              <HeaderCell label="champs" align="right" />
              <HeaderCell label="win rate" align="right" />
              <HeaderCell label="elo" align="right" />
              <HeaderCell label="impact" />
            </tr>
          </thead>
          <tbody>
            {categories.map((row) => (
              <tr key={row.category}>
                <td className="font-mono-tab rounded-l px-2 py-2" style={{ color: categoryColor(row.category) }}>
                  {row.category}
                </td>
                <Cell value={String(row.tried)} align="right" />
                <Cell value={String(row.accepted)} align="right" />
                <Cell value={String(row.champions)} align="right" />
                <Cell value={`${Math.round(row.avgWinRate * 100)}%`} align="right" />
                <Cell value={row.avgElo.toFixed(0)} align="right" />
                <td className="rounded-r px-2 py-2">
                  <Bar value={row.effectiveness / maxEffectiveness} color={categoryColor(row.category)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[11.5px]" style={{ color: "var(--ink-faint)" }}>
        Effectiveness combines champion promotions, tournament win rate, and final Elo.
      </p>
    </section>
  );
}

function FeaturePanel({ features }: { features: FeatureMetric[] }) {
  const maxEngines = Math.max(1, ...features.map((f) => f.engines));
  return (
    <section className="panel p-6">
      <PanelHead title="Code feature analytics" meta="static source scan" />
      <div className="mt-5 flex flex-col gap-3">
        {features.map((feature) => (
          <div key={feature.key}>
            <div className="mb-1 flex items-baseline gap-3">
              <span className="text-[12px]" style={{ color: "var(--ink-soft)" }}>
                {feature.label}
              </span>
              <span className="font-mono-tab ml-auto text-[11px]" style={{ color: "var(--ink-faint)" }}>
                {feature.engines} engines / {feature.champions} champs / {Math.round(feature.avgWinRate * 100)}%
              </span>
            </div>
            <Bar value={feature.engines / maxEngines} color="var(--moss-400)" />
          </div>
        ))}
      </div>
    </section>
  );
}

function PromptPanel({ prompts }: { prompts: PromptMetric[] }) {
  return (
    <section className="panel p-6">
      <PanelHead title="Prompt leaderboard" meta="category-to-engine attribution is inferred" />
      <div className="mt-5 flex flex-col gap-3">
        {prompts.slice(0, 8).map((prompt, index) => (
          <article
            key={`${prompt.generation}-${prompt.category}-${index}`}
            className="rounded-md p-4"
            style={{
              border: "1px solid var(--line)",
              background: prompt.promoted
                ? "linear-gradient(90deg, rgba(84,112,74,0.16), rgba(15,19,17,0.35))"
                : "rgba(15,19,17,0.34)",
            }}
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="badge" style={{ color: categoryColor(prompt.category) }}>
                gen {prompt.generation} / {prompt.category}
              </span>
              {prompt.promoted ? (
                <span className="badge" style={{ color: "var(--moss-300)" }}>
                  promoted
                </span>
              ) : null}
              <span className="font-mono-tab ml-auto text-[11px]" style={{ color: "var(--ink-faint)" }}>
                {Math.round(prompt.winRate * 100)}% / {prompt.elo.toFixed(0)} Elo
              </span>
            </div>
            <p className="mt-3 text-[13px] leading-snug" style={{ color: "var(--ink-soft)" }}>
              {prompt.text}
            </p>
            <p className="mt-2 font-mono-tab text-[11px]" style={{ color: "var(--bronze-300)" }}>
              best: {prompt.bestEngine === "-" ? "no accepted engine" : prompt.bestEngine}
            </p>
          </article>
        ))}
      </div>
    </section>
  );
}

function GenerationPanel({ generations }: { generations: GenerationMetric[] }) {
  return (
    <section className="panel p-6">
      <PanelHead title="Generation impact" meta="champion lineage" />
      <div className="mt-5 flex flex-col">
        {generations.slice(-10).reverse().map((gen) => (
          <div
            key={gen.number}
            className="grid grid-cols-[3.25rem_1fr_auto] items-center gap-3 border-b py-3 last:border-b-0"
            style={{ borderColor: "var(--line)" }}
          >
            <span className="font-display-tight leading-none" style={{ color: "var(--ink)", fontSize: 24 }}>
              {gen.number}
            </span>
            <div className="min-w-0">
              <div className="flex min-w-0 flex-wrap items-center gap-2">
                <span className="font-mono-tab truncate text-[11.5px]" style={{ color: "var(--ink-muted)" }}>
                  {shortName(gen.before, 18)}
                </span>
                <span style={{ color: "var(--ink-faint)" }}>to</span>
                <span className="font-mono-tab truncate text-[11.5px]" style={{ color: "var(--bronze-300)" }}>
                  {shortName(gen.after, 18)}
                </span>
              </div>
              <div className="mt-1 flex flex-wrap gap-2 text-[11px]" style={{ color: "var(--ink-faint)" }}>
                <span>{gen.candidates} candidates</span>
                <span>/</span>
                <span>{gen.promoted ? gen.winningCategory : "kept incumbent"}</span>
                <span>/</span>
                <span>{formatSigned(gen.ratingGap)} rating gap</span>
              </div>
            </div>
            <span
              className="badge"
              style={{ color: gen.promoted ? "var(--moss-300)" : "var(--bronze-300)" }}
            >
              {gen.promoted ? "promoted" : "kept"}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function EngineLeaderboard({ engines }: { engines: EngineMetric[] }) {
  return (
    <section className="panel p-6">
      <PanelHead title="Engine leaderboard" meta="ranked by tournament score" />
      <div className="mt-5 overflow-x-auto">
        <table className="w-full border-separate border-spacing-y-1 text-[11.5px]">
          <thead>
            <tr style={{ color: "var(--ink-faint)" }}>
              <HeaderCell label="engine" />
              <HeaderCell label="category" />
              <HeaderCell label="gen" align="right" />
              <HeaderCell label="games" align="right" />
              <HeaderCell label="score" align="right" />
              <HeaderCell label="win rate" align="right" />
              <HeaderCell label="elo" align="right" />
            </tr>
          </thead>
          <tbody>
            {engines.slice(0, 14).map((engine) => (
              <tr key={engine.name}>
                <td className="font-mono-tab rounded-l px-2 py-2" style={{ color: engine.isChampion ? "var(--bronze-300)" : "var(--ink-soft)" }} title={engine.name}>
                  {shortName(engine.name, 28)}
                </td>
                <td className="px-2 py-2" style={{ color: categoryColor(engine.category) }}>
                  {engine.category}
                </td>
                <Cell value={String(engine.generation)} align="right" />
                <Cell value={String(engine.played)} align="right" />
                <Cell value={engine.score.toFixed(1)} align="right" />
                <Cell value={`${Math.round(engine.winRate * 100)}%`} align="right" />
                <Cell value={engine.elo.toFixed(0)} align="right" />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function HeaderCell({ label, align = "left" }: { label: string; align?: "left" | "right" }) {
  return (
    <th
      className="px-2 pb-2 text-[10px] font-normal uppercase tracking-woodland"
      style={{ textAlign: align }}
    >
      {label}
    </th>
  );
}

function Cell({ value, align = "left" }: { value: string; align?: "left" | "right" }) {
  return (
    <td className="px-2 py-2 font-mono-tab" style={{ color: "var(--ink-soft)", textAlign: align }}>
      {value}
    </td>
  );
}

function Bar({ value, color }: { value: number; color: string }) {
  const width = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="h-2 overflow-hidden rounded-full" style={{ background: "rgba(232,226,211,0.06)" }}>
      <div className="h-full rounded-full" style={{ width: `${width}%`, background: color }} />
    </div>
  );
}

function buildAnalytics(
  engineRows: EngineRow[],
  generationRows: GenerationRow[],
  gameRows: GameRow[],
  sources: Record<string, string | null>,
): Analytics {
  const completed = generationRows
    .filter((g) => Boolean(g.finished_at))
    .sort((a, b) => a.number - b.number);
  const championNames = new Set(completed.map((g) => g.champion_after));
  const engineByName = new Map(engineRows.map((engine) => [engine.name, engine]));
  const stats = buildScoreLines(engineRows, gameRows);

  const engineMetrics = Array.from(stats.entries()).map(([name, score]) => {
    const row = engineByName.get(name);
    const category = parseCategory(name);
    return {
      name,
      generation: row?.generation ?? parseGenerationNumber(name) ?? 0,
      category,
      elo: row?.elo ?? 1500,
      isChampion: championNames.has(name),
      ...score,
      winRate: score.played > 0 ? score.score / score.played : 0,
    };
  });

  for (const row of engineRows) {
    if (!stats.has(row.name)) {
      engineMetrics.push({
        name: row.name,
        generation: row.generation,
        category: parseCategory(row.name),
        elo: row.elo,
        isChampion: championNames.has(row.name),
        played: 0,
        wins: 0,
        draws: 0,
        losses: 0,
        score: 0,
        winRate: 0,
      });
    }
  }

  const sortedEngines = engineMetrics.sort(
    (a, b) => b.winRate - a.winRate || b.elo - a.elo || b.played - a.played,
  );

  const questionsByGeneration = new Map(
    completed.map((generation) => [
      generation.number,
      parseQuestions(generation.strategist_questions_json),
    ]),
  );

  const categories = buildCategoryMetrics(completed, sortedEngines, questionsByGeneration);
  const prompts = buildPromptMetrics(completed, sortedEngines, questionsByGeneration);
  const generations = buildGenerationMetrics(completed, engineRows, engineByName);
  const features = buildFeatureMetrics(sortedEngines, sources, championNames);
  const currentChampion = completed[completed.length - 1]?.champion_after ?? "baseline-v0";
  const bestChampionElo = Math.max(
    1500,
    ...Array.from(championNames).map((name) => engineByName.get(name)?.elo ?? 1500),
  );

  return {
    overview: {
      totalGenerations: completed.length,
      totalEngines: engineRows.length,
      totalGames: gameRows.length,
      promotionRate:
        completed.length > 0
          ? completed.filter((g) => g.champion_after !== g.champion_before).length / completed.length
          : 0,
      bestChampionElo,
      currentChampion,
    },
    categories,
    prompts,
    generations,
    engines: sortedEngines,
    features,
    insights: buildInsights(categories, prompts, sortedEngines, features),
  };
}

function buildScoreLines(engineRows: EngineRow[], games: GameRow[]): Map<string, ScoreLine> {
  const out = new Map<string, ScoreLine>();
  const ensure = (name: string) => {
    if (!out.has(name)) {
      out.set(name, { played: 0, wins: 0, draws: 0, losses: 0, score: 0 });
    }
    return out.get(name)!;
  };

  for (const engine of engineRows) ensure(engine.name);

  for (const game of games) {
    const white = ensure(game.white_name);
    const black = ensure(game.black_name);
    white.played += 1;
    black.played += 1;

    if (game.result === "1-0") {
      white.wins += 1;
      white.score += 1;
      black.losses += 1;
    } else if (game.result === "0-1") {
      black.wins += 1;
      black.score += 1;
      white.losses += 1;
    } else if (game.result === "1/2-1/2") {
      white.draws += 1;
      black.draws += 1;
      white.score += 0.5;
      black.score += 0.5;
    }
  }

  return out;
}

function buildCategoryMetrics(
  generations: GenerationRow[],
  engines: EngineMetric[],
  questionsByGeneration: Map<number, Question[]>,
): CategoryMetric[] {
  return CATEGORIES.map((category) => {
    const tried = Array.from(questionsByGeneration.values()).flat().filter((q) => q.category === category).length;
    const categoryEngines = engines.filter((engine) => engine.category === category);
    const champions = generations.filter(
      (gen) => gen.champion_after !== gen.champion_before && parseCategory(gen.champion_after) === category,
    ).length;
    const avgElo = average(categoryEngines.map((engine) => engine.elo), 1500);
    const avgWinRate = average(categoryEngines.map((engine) => engine.winRate), 0);
    const promotionRate = tried > 0 ? champions / tried : 0;
    return {
      category,
      tried,
      accepted: categoryEngines.length,
      champions,
      avgElo,
      avgWinRate,
      promotionRate,
      effectiveness: champions * 100 + avgWinRate * 50 + Math.max(0, avgElo - 1500) / 5,
    };
  }).sort((a, b) => b.effectiveness - a.effectiveness);
}

function buildPromptMetrics(
  generations: GenerationRow[],
  engines: EngineMetric[],
  questionsByGeneration: Map<number, Question[]>,
): PromptMetric[] {
  const rows: PromptMetric[] = [];

  for (const generation of generations) {
    const questions = questionsByGeneration.get(generation.number) ?? [];
    for (const question of questions) {
      const candidates = engines.filter(
        (engine) => engine.generation === generation.number && engine.category === question.category,
      );
      const best = candidates[0];
      const promoted =
        generation.champion_after !== generation.champion_before &&
        parseCategory(generation.champion_after) === question.category;
      const effectiveness = (promoted ? 100 : 0) + (best?.winRate ?? 0) * 50 + Math.max(0, (best?.elo ?? 1500) - 1500) / 5;

      rows.push({
        generation: generation.number,
        category: question.category,
        text: question.text,
        accepted: candidates.length,
        bestEngine: best?.name ?? "-",
        winRate: best?.winRate ?? 0,
        elo: best?.elo ?? 1500,
        promoted,
        effectiveness,
      });
    }
  }

  return rows.sort((a, b) => b.effectiveness - a.effectiveness);
}

function buildGenerationMetrics(
  generations: GenerationRow[],
  engines: EngineRow[],
  engineByName: Map<string, EngineRow>,
): GenerationMetric[] {
  return generations.map((generation) => {
    const beforeElo = engineByName.get(generation.champion_before)?.elo ?? 1500;
    const afterElo = engineByName.get(generation.champion_after)?.elo ?? 1500;
    const promoted = generation.champion_after !== generation.champion_before;
    return {
      number: generation.number,
      before: generation.champion_before,
      after: generation.champion_after,
      promoted,
      winningCategory: promoted ? parseCategory(generation.champion_after) : "unknown",
      candidates: engines.filter((engine) => engine.generation === generation.number).length,
      ratingGap: afterElo - beforeElo,
    };
  });
}

function buildFeatureMetrics(
  engines: EngineMetric[],
  sources: Record<string, string | null>,
  championNames: Set<string>,
): FeatureMetric[] {
  return FEATURE_RULES.map((rule) => {
    const matching = engines.filter((engine) => {
      const source = sources[engine.name];
      return typeof source === "string" && rule.re.test(source);
    });
    return {
      key: rule.key,
      label: rule.label,
      engines: matching.length,
      champions: matching.filter((engine) => championNames.has(engine.name)).length,
      avgWinRate: average(matching.map((engine) => engine.winRate), 0),
      avgElo: average(matching.map((engine) => engine.elo), 1500),
    };
  }).sort((a, b) => b.champions - a.champions || b.engines - a.engines);
}

function buildInsights(
  categories: CategoryMetric[],
  prompts: PromptMetric[],
  engines: EngineMetric[],
  features: FeatureMetric[],
): Analytics["insights"] {
  const topCategory = categories[0];
  const topPrompt = prompts[0];
  const topEngine = engines[0];
  const topFeature = features.find((feature) => feature.champions > 0) ?? features[0];

  return [
    {
      label: "best category",
      value: topCategory ? topCategory.category : "-",
      detail: topCategory
        ? `${topCategory.champions} champion promotions, ${Math.round(topCategory.avgWinRate * 100)}% avg win rate.`
        : "No category data yet.",
    },
    {
      label: "best prompt",
      value: topPrompt ? `gen ${topPrompt.generation} ${topPrompt.category}` : "-",
      detail: topPrompt
        ? `${topPrompt.promoted ? "Promoted a champion" : "Best candidate"} with ${Math.round(topPrompt.winRate * 100)}% win rate.`
        : "No strategist prompts yet.",
    },
    {
      label: "best engine",
      value: topEngine ? shortName(topEngine.name, 20) : "-",
      detail: topEngine
        ? `${topEngine.score.toFixed(1)}/${topEngine.played} score, ${topEngine.elo.toFixed(0)} Elo.`
        : "No engine results yet.",
    },
    {
      label: "champion feature",
      value: topFeature ? topFeature.label : "-",
      detail: topFeature
        ? `Appears in ${topFeature.champions} promoted champion source file${topFeature.champions === 1 ? "" : "s"}.`
        : "No source feature matches yet.",
    },
  ];
}

function parseEngines(raw: unknown): EngineRow[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((row): row is EngineRow => {
      const item = row as Partial<EngineRow>;
      return (
        typeof item.name === "string" &&
        typeof item.generation === "number" &&
        typeof item.code_path === "string" &&
        typeof item.elo === "number"
      );
    })
    .sort((a, b) => a.generation - b.generation || a.name.localeCompare(b.name));
}

function parseGenerations(raw: unknown): GenerationRow[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((row): row is GenerationRow => {
      const item = row as Partial<GenerationRow>;
      return (
        typeof item.number === "number" &&
        typeof item.champion_before === "string" &&
        typeof item.champion_after === "string" &&
        typeof item.strategist_questions_json === "string"
      );
    })
    .sort((a, b) => a.number - b.number);
}

function parseGames(raw: unknown): GameRow[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter((row): row is GameRow => {
    const item = row as Partial<GameRow>;
    return (
      typeof item.generation === "number" &&
      typeof item.white_name === "string" &&
      typeof item.black_name === "string" &&
      typeof item.result === "string" &&
      typeof item.termination === "string"
    );
  });
}

function parseQuestions(raw: string): Question[] {
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((item) => {
        const row = item as Partial<Question>;
        return {
          category: normalizeCategory(row.category),
          text: typeof row.text === "string" ? row.text : "",
        };
      })
      .filter((q) => q.text.length > 0);
  } catch {
    return [];
  }
}

function normalizeCategory(value: unknown): Category {
  return typeof value === "string" && CATEGORIES.includes(value as Category)
    ? (value as Category)
    : "unknown";
}

function parseCategory(name: string): Category {
  const match = /^gen\d+-([a-z]+)-/.exec(name);
  return normalizeCategory(match?.[1]);
}

function parseGenerationNumber(name: string): number | null {
  const match = /^gen(\d+)-/.exec(name);
  return match ? Number(match[1]) : null;
}

function categoryColor(category: Category): string {
  switch (category) {
    case "prompt":
      return "var(--bronze-500)";
    case "search":
      return "var(--moss-300)";
    case "book":
      return "var(--moss-500)";
    case "evaluation":
      return "var(--bronze-300)";
    case "sampling":
      return "#7d8aa0";
    default:
      return "var(--ink-faint)";
  }
}

function average(values: number[], fallback: number): number {
  const clean = values.filter((value) => Number.isFinite(value));
  if (clean.length === 0) return fallback;
  return clean.reduce((sum, value) => sum + value, 0) / clean.length;
}

function shortName(name: string, max = 14): string {
  return name.length > max ? `${name.slice(0, max - 1)}...` : name;
}

function formatSigned(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(0)}`;
}
