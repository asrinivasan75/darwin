/**
 * LiveBoards.tsx — grid of mini chess boards, one per active game.
 *
 * Listens to GameMove / GameFinished and folds them into per-game state
 * scoped to the latest generation boundary. See the original module
 * doc for the rationale around showing every board at once.
 *
 * @module LiveBoards
 */

import { Component, memo, useMemo } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { Chessboard } from "react-chessboard";
import type { DarwinEvent, GameMove, GameFinished } from "../api/events";

interface LiveBoardsProps {
  events: DarwinEvent[];
}

// Show only 2 boards live. The full tournament can have 90-800 games
// concurrent; rendering all of them is unreadable and CPU-heavy. Two
// boards is enough for the demo to feel "alive" (latest moves come in,
// games rotate as they finish), and the bracket panel separately shows
// every game's result. Sorted by most-recent activity (see below) so
// the boards always show whatever's actively making moves.
const MAX_BOARDS = 2;

interface GameState {
  game_id: number;
  fen: string;
  san_history: string[];
  white: string;
  black: string;
  ply: number;
  finished: boolean;
  result: string | null;
  termination: string | null;
  last_event_idx: number;
}

const PIECE_NAMES: Record<string, string> = {
  K: "king",
  Q: "queen",
  R: "rook",
  B: "bishop",
  N: "knight",
};

function verboseMove(san: string, color: "white" | "black"): string {
  if (!san) return "";
  let s = san.replace(/[+#]$/, "");
  if (s === "O-O" || s === "0-0") return `${color} castles kingside`;
  if (s === "O-O-O" || s === "0-0-0") return `${color} castles queenside`;

  let promotion = "";
  const promMatch = s.match(/=([QRBN])$/);
  if (promMatch) {
    promotion = ` and promotes to ${PIECE_NAMES[promMatch[1]]}`;
    s = s.replace(/=([QRBN])$/, "");
  }

  const first = s[0];
  let piece: string;
  if (first && first in PIECE_NAMES) {
    piece = PIECE_NAMES[first];
    s = s.slice(1);
  } else {
    if (s.length > 2 && /^[a-h]/.test(s) && s[1] === "x") {
      piece = `${s[0]}-pawn`;
      s = s.slice(1);
    } else {
      piece = "pawn";
    }
  }

  const captures = s.includes("x");
  s = s.replace("x", "");
  const dest = s.slice(-2);
  if (!/^[a-h][1-8]$/.test(dest)) return san;

  const verb = captures ? "takes" : "to";
  return `${color} ${piece} ${verb} ${dest}${promotion}`;
}

export default function LiveBoards({ events }: LiveBoardsProps) {
  const { games, runningCount, doneCount } = useMemo(() => {
    let lastBoundary = -1;
    for (let i = 0; i < events.length; i++) {
      const t = events[i].type;
      if (t === "generation.started" || t === "generation.cancelled") {
        lastBoundary = i;
      }
    }

    const map = new Map<number, GameState>();
    for (let i = lastBoundary + 1; i < events.length; i++) {
      const e = events[i];
      if (e.type === "game.move") {
        const m = e as GameMove;
        const prev = map.get(m.game_id);
        map.set(m.game_id, {
          game_id: m.game_id,
          fen: m.fen,
          san_history: prev ? [...prev.san_history, m.san] : [m.san],
          white: m.white,
          black: m.black,
          ply: m.ply,
          finished: prev?.finished ?? false,
          result: prev?.result ?? null,
          termination: prev?.termination ?? null,
          last_event_idx: i,
        });
      } else if (e.type === "game.finished") {
        const f = e as GameFinished;
        const prev = map.get(f.game_id);
        if (prev) {
          map.set(f.game_id, {
            ...prev,
            finished: true,
            result: f.result,
            termination: f.termination,
            last_event_idx: i,
          });
        } else {
          map.set(f.game_id, {
            game_id: f.game_id,
            fen: "start",
            san_history: [],
            white: f.white,
            black: f.black,
            ply: 0,
            finished: true,
            result: f.result,
            termination: f.termination,
            last_event_idx: i,
          });
        }
      }
    }

    // Tally totals BEFORE the slice. The header shows aggregate
    // counts ("47 running · 43 done · 90 total") so the user can see
    // tournament progress even though the grid only renders 2 boards.
    let running = 0;
    let done = 0;
    for (const g of map.values()) {
      if (g.finished) done += 1;
      else running += 1;
    }

    // Sort by most-recent-activity descending so the visible boards
    // are always whatever's actively making moves. With MAX_BOARDS=2,
    // this means: as games finish or stall, fresher games slide in.
    // Within the same recency bucket, prefer in-progress games over
    // finished ones so a board that just hit checkmate doesn't hog
    // the slot while other games are still being played.
    const visibleGames = Array.from(map.values())
      .sort((a, b) => {
        if (a.finished !== b.finished) return a.finished ? 1 : -1;
        return b.last_event_idx - a.last_event_idx;
      })
      .slice(0, MAX_BOARDS);

    return {
      games: visibleGames,
      runningCount: running,
      doneCount: done,
    };
  }, [events]);

  // Header meta — counts accurate to the full game set, color-coded so
  // status reads at a glance: amber/pulsing for running, green for done,
  // dim for the "/total" denominator. Done count includes synthesized
  // forfeits (termination=error), so it grows steadily even if some
  // candidates time out.
  const totalGames = runningCount + doneCount;
  let meta: ReactNode;
  if (totalGames === 0) {
    meta = (
      <span style={{ color: "var(--ink-faint)" }}>no games yet</span>
    );
  } else {
    meta = (
      <span className="inline-flex items-center gap-2 normal-case tracking-normal">
        {runningCount > 0 && (
          <span className="inline-flex items-center gap-1.5">
            {/* Pulsing dot — visual heartbeat that the tournament is alive */}
            <span
              className="inline-block w-1.5 h-1.5 rounded-full animate-pulse"
              style={{ backgroundColor: "#fbbf24" }}
            />
            <span style={{ color: "#fbbf24" }} className="font-semibold">
              {runningCount}
            </span>
            <span style={{ color: "var(--ink-faint)" }}>running</span>
          </span>
        )}
        {runningCount > 0 && doneCount > 0 && (
          <span style={{ color: "var(--ink-faint)" }}>·</span>
        )}
        {doneCount > 0 && (
          <span className="inline-flex items-center gap-1.5">
            <span style={{ color: "#10b981" }} className="font-semibold">
              {doneCount}
            </span>
            <span style={{ color: "var(--ink-faint)" }}>done</span>
          </span>
        )}
        <span style={{ color: "var(--ink-faint)" }}>
          / {totalGames}
        </span>
      </span>
    );
  }

  return (
    <div className="panel p-6">
      <PanelHead title="Live boards" meta={meta} />

      {games.length === 0 ? (
        <EmptyPlot
          message="No game in progress."
          hint="Start a generation to begin the tournament."
        />
      ) : (
        <div
          className="mt-5 gap-4"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          }}
        >
          {games.map((g, i) => (
            <BoardCard key={g.game_id} game={g} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Panel-head + empty-plot, shared utilities ─────────────────────────────

export function PanelHead({
  eyebrow,
  title,
  meta,
}: {
  /** Optional small-caps label above the title. Omit for a tighter head. */
  eyebrow?: string;
  title: string;
  /** Right-aligned meta. ReactNode so panels can render colored counters,
   *  pulsing dots, etc. — strings still work as before. */
  meta?: ReactNode;
}) {
  return (
    <div>
      <div className="flex items-end justify-between gap-4">
        <div>
          {eyebrow ? <span className="eyebrow">{eyebrow}</span> : null}
          <h2 className={`panel-title ${eyebrow ? "mt-1.5" : ""}`}>{title}</h2>
        </div>
        {meta ? (
          <span
            className="text-[11px] tracking-woodland uppercase"
            style={{ color: "var(--ink-faint)" }}
          >
            {meta}
          </span>
        ) : null}
      </div>
      <div className="ornament mt-3" />
    </div>
  );
}

export function EmptyPlot({
  message,
  hint,
}: {
  message: string;
  hint?: string;
}) {
  return (
    <div
      className="mt-6 flex flex-col items-center justify-center gap-2 rounded-md py-12 text-center"
      style={{
        border: "1px dashed var(--line-strong)",
        background:
          "radial-gradient(380px 160px at 50% 0%, rgba(63,87,57,0.18), transparent 70%)",
      }}
    >
      <span
        className="font-display italic"
        style={{ fontSize: 19, color: "var(--bronze-300)" }}
      >
        {message}
      </span>
      {hint ? (
        <span
          className="text-[12.5px]"
          style={{ color: "var(--ink-faint)" }}
        >
          {hint}
        </span>
      ) : null}
    </div>
  );
}

// ── Board card ────────────────────────────────────────────────────────────

/**
 * Catches render errors from react-chessboard so a malformed FEN (or
 * an internal lib bug) renders a small placeholder square instead of
 * collapsing the parent flex column to zero height — which is what
 * caused the "boards missing entirely" symptom on the dashboard.
 */
class BoardErrorBoundary extends Component<
  { children: ReactNode; fen: string },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.warn("[LiveBoards] Chessboard render failed:", error, info);
  }

  componentDidUpdate(prevProps: { fen: string }) {
    // Reset error state when the FEN changes — the next move might
    // be valid even if the previous one wasn't.
    if (prevProps.fen !== this.props.fen && this.state.hasError) {
      this.setState({ hasError: false });
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          className="flex items-center justify-center text-[10px] italic"
          style={{
            width: 168,
            height: 168,
            color: "var(--ink-faint)",
            background: "rgba(34,41,35,0.5)",
          }}
        >
          board unavailable
        </div>
      );
    }
    return this.props.children;
  }
}

interface BoardCardProps {
  game: GameState;
  index: number;
}

const BoardCard = memo(BoardCardImpl, (prev, next) => {
  // Only re-render when this specific game's data actually changed.
  // The parent passes a fresh `games` array on every event, but we
  // don't care if some other game made a move — only ours.
  return (
    prev.index === next.index &&
    prev.game.game_id === next.game.game_id &&
    prev.game.fen === next.game.fen &&
    prev.game.ply === next.game.ply &&
    prev.game.finished === next.game.finished &&
    prev.game.result === next.game.result &&
    prev.game.termination === next.game.termination &&
    prev.game.san_history.length === next.game.san_history.length
  );
});

function BoardCardImpl({ game, index }: BoardCardProps) {
  const movePairs: { fullMove: number; text: string }[] = [];
  for (let i = 0; i < game.san_history.length; i += 2) {
    const fullMove = Math.floor(i / 2) + 1;
    const white = game.san_history[i];
    const black = game.san_history[i + 1];
    movePairs.push({
      fullMove,
      text: black ? `${fullMove}. ${white} ${black}` : `${fullMove}. ${white}`,
    });
  }
  movePairs.reverse();

  const isHallucination = game.termination === "illegal_move";
  const isCheckmate = game.termination === "checkmate";
  const isDraw =
    game.result === "1/2-1/2" ||
    game.termination === "stalemate" ||
    game.termination === "max_moves" ||
    game.termination === "draw";
  const drawLabel =
    game.termination === "stalemate"
      ? "drawn — stalemate"
      : game.termination === "max_moves"
        ? "drawn — move limit"
        : "drawn";
  const verdictLabel = isHallucination
    ? "hallucination"
    : isCheckmate
      ? "checkmate"
      : isDraw
        ? drawLabel
        : game.termination
          ? game.termination.replace(/_/g, " ")
          : null;

  const verdictColor = isHallucination
    ? "var(--ember-500)"
    : isCheckmate
      ? "var(--bronze-400)"
      : isDraw
        ? "var(--moss-400)"
        : "var(--ink-muted)";

  // "live" cards drift very softly to break the static-grid feeling.
  const animStyle = !game.finished
    ? { animation: `drift 6.${(index % 5) + 1}s ease-in-out infinite` }
    : {};

  return (
    <div
      className={`relative rounded-lg p-3 transition-opacity ${
        game.finished ? "opacity-80" : ""
      }`}
      style={{
        background:
          "linear-gradient(180deg, rgba(34,41,35,0.85), rgba(22,27,24,0.92))",
        border: "1px solid var(--line)",
        boxShadow:
          "inset 0 1px 0 rgba(232,226,211,0.04), 0 12px 26px -22px rgba(0,0,0,0.7)",
        ...animStyle,
      }}
    >
      {/* Card header — a small placard with game id and ply */}
      <div className="mb-2 flex items-center justify-between">
        <span
          className="font-mono-tab text-[10.5px] tracking-woodland uppercase"
          style={{ color: "var(--ink-faint)" }}
        >
          board · {String(game.game_id).padStart(2, "0")}
        </span>
        <span
          className="font-mono-tab text-[10.5px]"
          style={{ color: "var(--ink-faint)" }}
        >
          ply {game.ply}
        </span>
      </div>

      {/* Two-column body: warm wooden board + scoresheet */}
      <div className="flex gap-3">
        <div className="flex shrink-0 flex-col" style={{ width: 168 }}>
          <PlayerLine
            color="black"
            name={game.black}
            highlighted={
              game.finished && game.result === "0-1"
            }
          />
          <div
            className="my-1.5 overflow-hidden rounded"
            style={{
              boxShadow:
                "0 0 0 1px rgba(0,0,0,0.45), 0 6px 18px -10px rgba(0,0,0,0.6), inset 0 0 0 2px rgba(122,90,55,0.55)",
              minHeight: 168,
            }}
          >
            <BoardErrorBoundary fen={game.fen}>
              <Chessboard
                // Defensive: empty / undefined fen happens when a game
                // was synthesized as an error-forfeit before any moves
                // landed (Modal timeout fallback). react-chessboard
                // silently renders nothing for empty position; force
                // the starting position so the slot has a board.
                position={
                  !game.fen || game.fen === "start"
                    ? "start"
                    : game.fen
                }
                arePiecesDraggable={false}
                customDarkSquareStyle={{
                  backgroundColor: "var(--board-dark)",
                }}
                customLightSquareStyle={{
                  backgroundColor: "var(--board-light)",
                }}
                boardWidth={168}
              />
            </BoardErrorBoundary>
          </div>
          <PlayerLine
            color="white"
            name={game.white}
            highlighted={
              game.finished && game.result === "1-0"
            }
          />
        </div>

        <div className="flex min-w-0 flex-1 flex-col text-[10.5px]">
          {game.finished && (
            <div className="mb-1.5 flex items-baseline gap-2">
              {verdictLabel && (
                <span
                  className="font-display italic leading-none"
                  style={{ fontSize: 12.5, color: verdictColor }}
                >
                  {verdictLabel}
                </span>
              )}
              <span
                className="font-mono-tab"
                style={{ color: "var(--bronze-300)" }}
              >
                {game.result}
              </span>
            </div>
          )}

          <div
            className="font-mono-tab leading-tight"
            style={{
              color: "var(--ink-soft)",
              maxHeight: 196,
              overflowY: "auto",
            }}
          >
            {movePairs.length === 0 ? (
              <span
                className="italic"
                style={{ color: "var(--ink-faint)" }}
              >
                no moves yet
              </span>
            ) : (
              movePairs.map(({ fullMove, text }) => (
                <div
                  key={fullMove}
                  className="truncate"
                  title={text}
                  style={{ color: "var(--ink-soft)" }}
                >
                  {text}
                </div>
              ))
            )}
          </div>

          {/* Latest move, verbose — anchored to the bottom of the card */}
          {game.san_history.length > 0 && !game.finished && (
            <div
              className="mt-2 border-t pt-2 text-[10.5px] italic"
              style={{
                borderColor: "var(--line)",
                color: "var(--ink-faint)",
              }}
            >
              {verboseMove(
                game.san_history[game.san_history.length - 1],
                game.san_history.length % 2 === 1 ? "white" : "black",
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PlayerLine({
  color,
  name,
  highlighted,
}: {
  color: "white" | "black";
  name: string;
  highlighted: boolean;
}) {
  return (
    <div className="flex min-w-0 items-center gap-2">
      <span
        aria-hidden
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{
          background: color === "white" ? "var(--bronze-200)" : "#0d100d",
          border: "1px solid rgba(232,226,211,0.25)",
        }}
      />
      <span
        className="font-mono-tab truncate text-[11px]"
        style={{
          color: highlighted ? "var(--bronze-300)" : "var(--ink-soft)",
        }}
        title={name}
      >
        {name}
      </span>
    </div>
  );
}
