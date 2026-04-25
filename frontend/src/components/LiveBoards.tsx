/**
 * LiveBoards.tsx — grid of mini chess boards, one per active game.
 *
 * Listens to GameMove / GameFinished and folds them into per-game state
 * scoped to the latest generation boundary. See the original module
 * doc for the rationale around showing every board at once.
 *
 * @module LiveBoards
 */

import { useMemo } from "react";
import { Chessboard } from "react-chessboard";
import type { DarwinEvent, GameMove, GameFinished } from "../api/events";

interface LiveBoardsProps {
  events: DarwinEvent[];
}

const MAX_BOARDS = 16;

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
  const games = useMemo<GameState[]>(() => {
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

    return Array.from(map.values())
      .sort((a, b) => a.game_id - b.game_id)
      .slice(0, MAX_BOARDS);
  }, [events]);

  return (
    <div className="panel p-6">
      <PanelHead
        title="Live boards"
        meta={
          games.length === 0
            ? "no games yet"
            : `${games.length} game${games.length === 1 ? "" : "s"}`
        }
      />

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
  meta?: string;
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

interface BoardCardProps {
  game: GameState;
  index: number;
}

function BoardCard({ game, index }: BoardCardProps) {
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
            }}
          >
            <Chessboard
              position={game.fen === "start" ? "start" : game.fen}
              arePiecesDraggable={false}
              customDarkSquareStyle={{
                backgroundColor: "var(--board-dark)",
              }}
              customLightSquareStyle={{
                backgroundColor: "var(--board-light)",
              }}
              boardWidth={168}
            />
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
