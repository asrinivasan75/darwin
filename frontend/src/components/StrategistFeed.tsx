/**
 * StrategistFeed.tsx — live feed of LLM strategist questions and builder outcomes.
 *
 * Renders one card per {@link StrategistQuestion} event. Each card shows:
 *   - A colour-coded category badge (book / prompt / search / evaluation / sampling)
 *   - The question text proposed by the strategist LLM
 *   - A status indicator that updates when the matching {@link BuilderCompleted}
 *     event arrives (pending → success or failure)
 *   - The generated engine name on success
 *
 * This panel has the highest demo value: judges watch questions appear one by
 * one and builders flip to green as candidate engines are generated.
 *
 * @listens {StrategistQuestion} - one card per question (index 0–4)
 * @listens {BuilderCompleted}   - updates the status indicator on the matching card
 *
 * @module StrategistFeed
 */

import type {
  CubistEvent,
  StrategistQuestion,
  BuilderCompleted,
} from "../api/events";

/** Props accepted by {@link StrategistFeed}. */
interface StrategistFeedProps {
  /** Full accumulated event log from {@link useEventStream}. */
  events: CubistEvent[];
}

/**
 * Maps each question category to a Tailwind background colour class so cards
 * are visually distinct at a glance on demo day.
 */
const CATEGORY_COLORS: Record<StrategistQuestion["category"], string> = {
  book:       "bg-green-700",
  prompt:     "bg-purple-700",
  search:     "bg-blue-700",
  evaluation: "bg-orange-700",
  sampling:   "bg-pink-700",
};

/**
 * StrategistFeed — displays the LLM strategist's improvement questions and
 * their corresponding builder outcomes for the current generation.
 *
 * @param props.events - the full accumulated event log from useEventStream()
 * @returns a scrollable column of question cards, newest at the bottom
 */
export default function StrategistFeed({ events }: StrategistFeedProps) {
  // Extract questions and builder results from the flat event log
  const questions = events.filter(
    (e): e is StrategistQuestion => e.type === "strategist.question"
  );
  const builders = events.filter(
    (e): e is BuilderCompleted => e.type === "builder.completed"
  );

  /** Look up the builder result for a given question index, if it has arrived. */
  const builderFor = (index: number): BuilderCompleted | undefined =>
    builders.find((b) => b.question_index === index);

  if (questions.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 flex flex-col">
        <h2 className="text-xs font-semibold tracking-wider text-gray-400 uppercase mb-3">
          Strategist Feed
        </h2>
        <p className="text-gray-500 text-sm italic mt-2">
          Waiting for strategist…
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 flex flex-col">
      <h2 className="text-xs font-semibold tracking-wider text-gray-400 uppercase mb-3">
        Strategist Feed
      </h2>

      <div className="flex flex-col gap-3 overflow-y-auto">
        {questions.map((q) => {
          const builder = builderFor(q.index);
          return (
            <QuestionCard key={q.index} question={q} builder={builder} />
          );
        })}
      </div>
    </div>
  );
}

// ── Internal sub-components ──────────────────────────────────────────────────

/** Props for the individual question card. */
interface QuestionCardProps {
  question: StrategistQuestion;
  /** Undefined while the builder is still running. */
  builder: BuilderCompleted | undefined;
}

/**
 * Renders a single strategist question as a dark card with a category badge
 * and a status indicator on the right.
 *
 * @param props.question - the strategist question event
 * @param props.builder  - the corresponding builder.completed event (may be undefined)
 */
function QuestionCard({ question, builder }: QuestionCardProps) {
  const badgeColor = CATEGORY_COLORS[question.category];

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-md p-3 flex items-start gap-3">
      {/* Category badge */}
      <span
        className={`${badgeColor} text-white text-xs font-semibold px-2 py-0.5 rounded shrink-0 mt-0.5 uppercase tracking-wide`}
      >
        {question.category}
      </span>

      {/* Question text and engine name */}
      <div className="flex-1 min-w-0">
        <p className="text-gray-200 text-sm leading-snug">{question.text}</p>
        {builder?.ok && (
          <p className="text-gray-400 text-xs mt-1 truncate">
            → {builder.engine_name}
          </p>
        )}
        {builder && !builder.ok && (
          <p className="text-red-400 text-xs mt-1 truncate">
            ✗ {builder.error}
          </p>
        )}
      </div>

      {/* Status indicator */}
      <StatusIcon builder={builder} />
    </div>
  );
}

/**
 * Small icon showing whether the builder is pending, succeeded, or failed.
 *
 * @param props.builder - the builder.completed event, or undefined if still running
 */
function StatusIcon({ builder }: { builder: BuilderCompleted | undefined }) {
  if (!builder) {
    // Pulsing dot: builder is still generating code
    return (
      <span className="shrink-0 mt-1 w-2.5 h-2.5 rounded-full bg-gray-500 animate-pulse" />
    );
  }
  if (builder.ok) {
    return <span className="shrink-0 mt-0.5 text-green-400 text-base">✓</span>;
  }
  return <span className="shrink-0 mt-0.5 text-red-400 text-base">✗</span>;
}
