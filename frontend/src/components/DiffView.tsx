/**
 * DiffView.tsx - compare champion engine source across generations.
 *
 * Uses the existing generations REST endpoint to identify each generation's
 * champion, then loads source through `/api/engines/{name}/code`.
 */

import { useEffect, useMemo, useState } from "react";
import { fetchEngineCode, fetchGenerations } from "../api/client";
import { EmptyPlot, PanelHead } from "./LiveBoards";

type LoadState = "idle" | "loading" | "ready" | "error";

interface GenerationRecord {
  number: number;
  champion_before: string;
  champion_after: string;
  finished_at: string | null;
}

interface CompareRow {
  number: number;
  from: string;
  to: string;
}

interface DiffLine {
  kind: "context" | "added" | "removed";
  text: string;
}

interface CodeState {
  fromCode: string;
  toCode: string;
}

export default function DiffView() {
  const [state, setState] = useState<LoadState>("loading");
  const [generations, setGenerations] = useState<GenerationRecord[]>([]);
  const [selectedGen, setSelectedGen] = useState<number | null>(null);
  const [codeState, setCodeState] = useState<CodeState | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadGenerations() {
      setState("loading");
      setError(null);
      try {
        const raw = await fetchGenerations();
        const records = parseGenerations(raw);
        if (cancelled) return;
        setGenerations(records);
        setSelectedGen(records[records.length - 1]?.number ?? null);
        setState("idle");
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Could not load generations.");
        setState("error");
      }
    }

    loadGenerations();
    return () => {
      cancelled = true;
    };
  }, []);

  const comparisons = useMemo(() => buildComparisons(generations), [generations]);
  const selectedComparison =
    comparisons.find((row) => row.number === selectedGen) ?? comparisons[0];

  useEffect(() => {
    if (!selectedComparison) {
      setCodeState(null);
      return;
    }

    let cancelled = false;

    async function loadCode() {
      setState("loading");
      setError(null);
      setCodeState(null);
      try {
        const [fromCode, toCode] = await Promise.all([
          fetchEngineCode(selectedComparison.from),
          fetchEngineCode(selectedComparison.to),
        ]);
        if (cancelled) return;
        setCodeState({ fromCode, toCode });
        setState("ready");
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Could not load engine source.");
        setState("error");
      }
    }

    loadCode();
    return () => {
      cancelled = true;
    };
  }, [selectedComparison]);

  const diff = useMemo(() => {
    if (!codeState) return [];
    return buildLineDiff(codeState.fromCode, codeState.toCode);
  }, [codeState]);

  const summary = useMemo(() => summarizeDiff(diff), [diff]);
  const hasNoGenerations = state !== "loading" && comparisons.length === 0;
  const unchanged =
    state === "ready" &&
    codeState !== null &&
    selectedComparison !== undefined &&
    (selectedComparison.from === selectedComparison.to ||
      codeState.fromCode === codeState.toCode);

  return (
    <div className="rise" style={{ animationDelay: "240ms" }}>
      <div className="panel flex flex-col p-6">
        <PanelHead
          title="Diff View"
          meta={
            selectedComparison
              ? `gen ${selectedComparison.number}`
              : "champion history"
          }
        />

        {hasNoGenerations ? (
          <EmptyPlot
            message="No completed generations yet."
            hint="Run a generation to compare champion source changes."
          />
        ) : (
          <>
            <div className="mt-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
              <div className="flex min-w-0 flex-col gap-2">
                <label
                  htmlFor="diff-generation"
                  className="eyebrow"
                  style={{ width: "fit-content" }}
                >
                  Compare generation
                </label>
                <select
                  id="diff-generation"
                  value={selectedComparison?.number ?? ""}
                  onChange={(e) => setSelectedGen(Number(e.target.value))}
                  className="font-mono-tab rounded-md px-3 py-2 text-[12px] outline-none"
                  style={{
                    color: "var(--ink)",
                    background: "rgba(15,19,17,0.88)",
                    border: "1px solid var(--line-strong)",
                    minWidth: 220,
                  }}
                  disabled={state === "loading"}
                >
                  {comparisons.map((row) => (
                    <option key={row.number} value={row.number}>
                      gen {row.number}
                    </option>
                  ))}
                </select>
              </div>

              {selectedComparison ? (
                <div className="flex min-w-0 flex-1 flex-col gap-2 lg:items-end">
                  <div className="flex max-w-full flex-wrap items-center gap-2 text-[11.5px]">
                    <EngineLink name={selectedComparison.from} />
                    <span style={{ color: "var(--ink-faint)" }}>to</span>
                    <EngineLink name={selectedComparison.to} />
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <SummaryBadge label={`+${summary.added}`} tone="added" />
                    <SummaryBadge label={`-${summary.removed}`} tone="removed" />
                    <SummaryBadge label={`${summary.context} context`} tone="context" />
                  </div>
                </div>
              ) : null}
            </div>

            {state === "loading" ? (
              <EmptyPlot message="Loading engine source." hint="Fetching both champion files." />
            ) : state === "error" ? (
              <EmptyPlot
                message="Could not render diff."
                hint={error ?? "The selected engine source is unavailable."}
              />
            ) : unchanged ? (
              <EmptyPlot
                message="No source changes."
                hint="The selected generation kept the same champion or identical source."
              />
            ) : (
              <DiffCode lines={diff} />
            )}
          </>
        )}
      </div>
    </div>
  );
}

function EngineLink({ name }: { name: string }) {
  return (
    <a
      href={`/api/engines/${encodeURIComponent(name)}/code`}
      download={`${name}.py`}
      className="font-mono-tab max-w-[260px] truncate transition-colors"
      style={{ color: "var(--bronze-300)" }}
      title={`Download ${name}.py`}
    >
      {name}.py
    </a>
  );
}

function SummaryBadge({
  label,
  tone,
}: {
  label: string;
  tone: "added" | "removed" | "context";
}) {
  const color =
    tone === "added"
      ? "var(--moss-300)"
      : tone === "removed"
        ? "var(--ember-500)"
        : "var(--ink-muted)";

  return (
    <span className="badge" style={{ color }}>
      {label}
    </span>
  );
}

function DiffCode({ lines }: { lines: DiffLine[] }) {
  return (
    <div
      className="mt-5 overflow-hidden rounded-md"
      style={{
        border: "1px solid var(--line)",
        background: "rgba(10,14,12,0.58)",
      }}
    >
      <div
        className="max-h-[66vh] overflow-auto py-3 font-mono-tab text-[11.5px] leading-5"
        style={{ color: "var(--ink-soft)" }}
      >
        {lines.map((line, index) => (
          <div
            key={`${index}-${line.kind}-${line.text}`}
            className="grid min-w-max grid-cols-[3.5rem_1fr] gap-3 px-4"
            style={{
              background:
                line.kind === "added"
                  ? "rgba(84,112,74,0.18)"
                  : line.kind === "removed"
                    ? "rgba(168,80,64,0.14)"
                    : "transparent",
              color:
                line.kind === "added"
                  ? "var(--moss-300)"
                  : line.kind === "removed"
                    ? "var(--ember-500)"
                    : "var(--ink-soft)",
            }}
          >
            <span
              className="select-none text-right"
              style={{ color: "var(--ink-faint)" }}
            >
              {index + 1}
            </span>
            <span className="whitespace-pre">
              {prefixFor(line.kind)}
              {line.text || " "}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function prefixFor(kind: DiffLine["kind"]): string {
  if (kind === "added") return "+ ";
  if (kind === "removed") return "- ";
  return "  ";
}

function parseGenerations(raw: unknown): GenerationRecord[] {
  if (!Array.isArray(raw)) return [];

  return raw
    .filter((item): item is GenerationRecord => {
      if (typeof item !== "object" || item === null) return false;
      const row = item as Partial<GenerationRecord>;
      return (
        typeof row.number === "number" &&
        typeof row.champion_before === "string" &&
        typeof row.champion_after === "string" &&
        typeof row.finished_at === "string"
      );
    })
    .sort((a, b) => a.number - b.number);
}

function buildComparisons(generations: GenerationRecord[]): CompareRow[] {
  return generations.map((generation, index) => ({
    number: generation.number,
    from: index === 0 ? generation.champion_before : generations[index - 1].champion_after,
    to: generation.champion_after,
  }));
}

function buildLineDiff(fromCode: string, toCode: string): DiffLine[] {
  const before = splitLines(fromCode);
  const after = splitLines(toCode);
  const table = buildLcsTable(before, after);
  const lines: DiffLine[] = [];

  let i = 0;
  let j = 0;
  while (i < before.length && j < after.length) {
    if (before[i] === after[j]) {
      lines.push({ kind: "context", text: before[i] });
      i += 1;
      j += 1;
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      lines.push({ kind: "removed", text: before[i] });
      i += 1;
    } else {
      lines.push({ kind: "added", text: after[j] });
      j += 1;
    }
  }

  while (i < before.length) {
    lines.push({ kind: "removed", text: before[i] });
    i += 1;
  }

  while (j < after.length) {
    lines.push({ kind: "added", text: after[j] });
    j += 1;
  }

  return lines;
}

function splitLines(source: string): string[] {
  return source.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
}

function buildLcsTable(before: string[], after: string[]): number[][] {
  const table = Array.from({ length: before.length + 1 }, () =>
    Array(after.length + 1).fill(0) as number[],
  );

  for (let i = before.length - 1; i >= 0; i -= 1) {
    for (let j = after.length - 1; j >= 0; j -= 1) {
      table[i][j] =
        before[i] === after[j]
          ? table[i + 1][j + 1] + 1
          : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }

  return table;
}

function summarizeDiff(lines: DiffLine[]) {
  return lines.reduce(
    (acc, line) => {
      if (line.kind === "added") acc.added += 1;
      else if (line.kind === "removed") acc.removed += 1;
      else acc.context += 1;
      return acc;
    },
    { added: 0, removed: 0, context: 0 },
  );
}
