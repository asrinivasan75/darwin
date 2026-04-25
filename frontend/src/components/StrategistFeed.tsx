/**
 * StrategistFeed.tsx — live feed of LLM strategist questions and builder outcomes.
 *
 * @module StrategistFeed
 */

import type {
  DarwinEvent,
  StrategistQuestion,
  BuilderCompleted,
} from "../api/events";
import { PanelHead, EmptyPlot } from "./LiveBoards";

interface StrategistFeedProps {
  events: DarwinEvent[];
}

/**
 * Per-category hue used for the left margin rule. One earthy color per
 * category — the rule is the only category indicator now that the badge
 * is gone.
 */
const CATEGORY_DOT: Record<StrategistQuestion["category"], string> = {
  book: "var(--moss-500)",
  prompt: "var(--bronze-500)",
  search: "#7d8aa0",
  evaluation: "var(--bronze-300)",
  sampling: "var(--moss-300)",
};

export default function StrategistFeed({ events }: StrategistFeedProps) {
  let lastBoundary = -1;
  for (let i = 0; i < events.length; i++) {
    const t = events[i].type;
    if (t === "generation.started" || t === "generation.cancelled") {
      lastBoundary = i;
    }
  }
  const currentEvents = events.slice(lastBoundary + 1);

  const questions = currentEvents.filter(
    (e): e is StrategistQuestion => e.type === "strategist.question",
  );
  const builders = currentEvents.filter(
    (e): e is BuilderCompleted => e.type === "builder.completed",
  );

  const builderFor = (index: number): BuilderCompleted | undefined =>
    builders.find((b) => b.question_index === index);

  return (
    <div className="panel flex flex-col p-6">
      <PanelHead
        title="Strategist"
        meta={
          questions.length === 0
            ? "no questions yet"
            : `${questions.length} question${questions.length === 1 ? "" : "s"}`
        }
      />

      {questions.length === 0 ? (
        <EmptyPlot
          message="No strategist questions yet."
          hint="Each generation, the LLM proposes up to five distinct improvement directions."
        />
      ) : (
        <ol className="mt-5 flex flex-col gap-3">
          {questions.map((q, i) => (
            <QuestionCard
              key={q.index}
              question={q}
              builder={builderFor(q.index)}
              ordinal={i + 1}
            />
          ))}
        </ol>
      )}
    </div>
  );
}

interface QuestionCardProps {
  question: StrategistQuestion;
  builder: BuilderCompleted | undefined;
  ordinal: number;
}

function QuestionCard({ question, builder, ordinal }: QuestionCardProps) {
  const dot = CATEGORY_DOT[question.category];
  const settled = builder !== undefined;

  return (
    <li
      className="bloom relative overflow-hidden rounded-lg p-4"
      style={{
        animationDelay: `${ordinal * 70}ms`,
        background:
          "linear-gradient(180deg, rgba(34,41,35,0.7), rgba(22,27,24,0.9))",
        border: "1px solid var(--line)",
      }}
    >
      {/* Left column rule, tinted by category — reads like a margin annotation */}
      <span
        aria-hidden
        className="absolute left-0 top-3 bottom-3 w-[3px] rounded-r"
        style={{ background: dot, opacity: 0.7 }}
      />

      <div className="flex items-start gap-4 pl-3">
        <div className="flex w-10 shrink-0 flex-col items-start pt-0.5">
          <span
            className="font-display italic leading-none"
            style={{ fontSize: 22, color: "var(--ink-faint)" }}
          >
            {String(ordinal).padStart(2, "0")}
          </span>
        </div>

        <div className="min-w-0 flex-1">
          <p
            className="font-display text-[15.5px] leading-snug"
            style={{
              color: "var(--ink)",
              fontVariationSettings:
                '"opsz" 24, "SOFT" 50, "wght" 380',
            }}
          >
            {question.text}
          </p>

          {builder?.ok && (
            <p
              className="font-mono-tab mt-2 truncate text-[11.5px]"
              style={{ color: "var(--bronze-300)" }}
              title={builder.engine_name}
            >
              → <span className="italic">{builder.engine_name}</span>
            </p>
          )}
          {builder && !builder.ok && (
            <p
              className="mt-2 truncate text-[11.5px] italic"
              style={{ color: "var(--ember-500)" }}
              title={builder.error ?? undefined}
            >
              rejected — {builder.error}
            </p>
          )}
          {!builder && (
            <p
              className="mt-2 text-[11px] uppercase tracking-woodland"
              style={{ color: "var(--ink-faint)" }}
            >
              <span className="firefly mr-1.5 align-middle" />
              building
            </p>
          )}
        </div>

        <StatusGlyph builder={builder} settled={settled} />
      </div>
    </li>
  );
}

function StatusGlyph({
  builder,
  settled,
}: {
  builder: BuilderCompleted | undefined;
  settled: boolean;
}) {
  if (!settled) {
    return (
      <span
        aria-label="pending"
        className="mt-1 inline-block h-3 w-3 shrink-0 rounded-full"
        style={{
          background:
            "radial-gradient(circle at 30% 30%, rgba(232,226,211,0.5), rgba(232,226,211,0.05))",
          border: "1px solid var(--line-strong)",
        }}
      />
    );
  }
  if (builder!.ok) {
    return (
      <svg
        aria-label="accepted"
        className="mt-1 shrink-0"
        width={18}
        height={18}
        viewBox="0 0 18 18"
      >
        <circle
          cx={9}
          cy={9}
          r={8}
          fill="rgba(63,87,57,0.25)"
          stroke="var(--moss-500)"
        />
        <path
          d="M5 9.6 L8 12 L13 6.5"
          fill="none"
          stroke="var(--moss-300)"
          strokeWidth={1.6}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }
  return (
    <svg
      aria-label="rejected"
      className="mt-1 shrink-0"
      width={18}
      height={18}
      viewBox="0 0 18 18"
    >
      <circle
        cx={9}
        cy={9}
        r={8}
        fill="rgba(168,80,64,0.18)"
        stroke="var(--ember-600)"
      />
      <path
        d="M6 6 L12 12 M12 6 L6 12"
        fill="none"
        stroke="var(--ember-500)"
        strokeWidth={1.6}
        strokeLinecap="round"
      />
    </svg>
  );
}
