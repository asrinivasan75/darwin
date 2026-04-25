/**
 * LiveBoards.tsx — grid of mini chess boards, one per active game.
 *
 * Replaces the single-board {@link LiveBoard} with a grid of up to
 * ``MAX_BOARDS`` boards. Each board tracks the latest position of one
 * ``game_id`` derived from the event log. Games marked finished by a
 * ``game.finished`` event remain visible (with their final FEN and a
 * result chip) until pushed off the grid by newer games — judges can
 * still see the just-completed game while the next pair gets going.
 *
 * Rationale: a tournament can have ~12+ games per generation playing
 * concurrently. A single "hot" board (LiveBoard) buries that parallelism.
 * Showing N boards at once makes the round-robin visible.
 *
 * @listens {GameMove}        — updates a per-game FEN
 * @listens {GameFinished}    — marks a game as final
 * @listens {GenerationStarted | GenerationCancelled} — clears the grid
 *
 * @module LiveBoards
 */

import { useMemo } from "react";
import { Chessboard } from "react-chessboard";
import type { CubistEvent, GameMove, GameFinished } from "../api/events";

interface LiveBoardsProps {
  events: CubistEvent[];
}

// Max number of boards rendered simultaneously. With games_per_pairing=2
// and 3 engines (baseline + 2 candidates), the round-robin schedules
// 3*2*2 = 12 concurrent games. We render all of them so a judge can see
// the full tournament at once. Cap is a safety net in case a future
// config bumps the cohort to 4 candidates (4*3*2 = 24 games — at that
// point we'd want a different layout anyway).
const MAX_BOARDS = 12;

/** Per-board state derived from the event log. */
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
  /** Index in the events array of the last update — used to sort recency. */
  last_event_idx: number;
}

// Major/minor pieces. Pawn isn't here on purpose: in SAN, pawn moves
// have no leading piece letter (`e4`, `exd5`, `e8=Q`), so the pawn case
// is detected by absence — see `verboseMove` below.
const PIECE_NAMES: Record<string, string> = {
  K: "king",
  Q: "queen",
  R: "rook",
  B: "bishop",
  N: "knight",
};

/**
 * Convert a SAN move into a human-readable phrase.
 *
 * Examples:
 *   "Rxb3+"  → "black rook takes b3"
 *   "O-O"    → "white castles kingside"
 *   "e4"     → "white pawn to e4"
 *   "exd5"   → "white e-pawn takes d5"   // file disambiguates the pawn
 *   "e8=Q"   → "white pawn to e8 and promotes to queen"
 *
 * Best-effort — unparseable inputs fall back to the raw SAN.
 */
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
    // Pawn move. Pawn captures always lead with the source file
    // (e.g. "exd5") — keep it as part of the piece name so it isn't
    // swallowed by the disambiguation strip below.
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

/**
 * LiveBoards — fold the event log into a per-game-id state map and render
 * up to ``MAX_BOARDS`` of the most recently active boards.
 *
 * On ``generation.started`` or ``generation.cancelled`` we drop everything
 * — those terminal events delimit the boundary of one tournament.
 */
export default function LiveBoards({ events }: LiveBoardsProps) {
  const games = useMemo<GameState[]>(() => {
    // Find the latest "boundary" event — anything before it is from a
    // previous (or cancelled) generation and not worth showing.
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
          // We received finished without any moves — render an empty board
          // anyway so the result is at least visible.
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

    // Stable layout: sort by game_id ascending so each game keeps its
    // grid slot from start to finish. A game that lands in the top-left
    // when the tournament starts stays there until it ends — matches
    // how a human would expect to follow a row of games on a chess
    // tournament hall display.
    return Array.from(map.values())
      .sort((a, b) => a.game_id - b.game_id)
      .slice(0, MAX_BOARDS);
  }, [events]);

  return (
    <div className="bg-gray-800 rounded-lg p-4 col-span-full">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs font-semibold tracking-wider text-gray-400 uppercase">
          Live Boards
        </h2>
        <span className="text-xs text-gray-500">
          {games.length === 0 ? "no active games" : `${games.length} active`}
        </span>
      </div>

      {games.length === 0 ? (
        <p className="text-gray-500 text-xs italic text-center py-12">
          Waiting for first game…
        </p>
      ) : (
        <div className="flex flex-wrap gap-4">
          {games.map((g) => (
            <BoardCard key={g.game_id} game={g} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Internal sub-components ────────────────────────────────────────────────

interface BoardCardProps {
  game: GameState;
}

function BoardCard({ game }: BoardCardProps) {
  // SAN history is in chronological order; even indices are white moves,
  // odd are black. Reverse for display so the most recent move sits on top
  // — keeps the latest action visible without scrolling.
  const verboseHistory = game.san_history
    .map((san, idx) => ({
      idx,
      text: verboseMove(san, idx % 2 === 0 ? "white" : "black"),
    }))
    .reverse();

  // Bold the verdict for the outcomes operators care about most when
  // judging a candidate engine: hallucinated/illegal moves, real
  // checkmates, and draws (stalemate / max-moves / threefold). Other
  // terminations (time, error) get plain styling.
  const isHallucination = game.termination === "illegal_move";
  const isCheckmate = game.termination === "checkmate";
  const isDraw =
    game.result === "1/2-1/2" ||
    game.termination === "stalemate" ||
    game.termination === "max_moves" ||
    game.termination === "draw";
  const drawLabel =
    game.termination === "stalemate"
      ? "Draw — stalemate"
      : game.termination === "max_moves"
        ? "Draw — move limit"
        : "Draw";
  const verdictLabel = isHallucination
    ? "Hallucination"
    : isCheckmate
      ? "Checkmate"
      : isDraw
        ? drawLabel
        : game.termination
          ? game.termination.replace(/_/g, " ")
          : null;

  return (
    <div
      className={`bg-gray-900 rounded p-2 flex flex-col text-xs ${
        game.finished ? "opacity-70" : ""
      }`}
      style={{ width: 360 }}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="text-gray-400 font-mono">#{game.game_id}</span>
        <span className="text-gray-500 font-mono">ply {game.ply}</span>
      </div>

      {/* Two-column body: board + player labels on the left, scrollable
          move log on the right. The right column flexes to fill the
          remaining card width. The result chip lives at the top of the
          right column so it sits directly above the most recent move. */}
      <div className="flex gap-2">
        <div className="flex flex-col shrink-0" style={{ width: 170 }}>
          <div className="flex items-center gap-1 truncate">
            <span className="inline-block w-2 h-2 rounded-sm bg-gray-900 border border-gray-500 shrink-0" />
            <span
              className="text-gray-300 font-mono truncate"
              title={game.black}
            >
              {game.black}
            </span>
          </div>

          <div className="my-1">
            <Chessboard
              position={game.fen === "start" ? "start" : game.fen}
              arePiecesDraggable={false}
              customDarkSquareStyle={{ backgroundColor: "#374151" }}
              customLightSquareStyle={{ backgroundColor: "#9ca3af" }}
              boardWidth={170}
            />
          </div>

          <div className="flex items-center gap-1 truncate">
            <span className="inline-block w-2 h-2 rounded-sm bg-gray-100 border border-gray-500 shrink-0" />
            <span
              className="text-gray-300 font-mono truncate"
              title={game.white}
            >
              {game.white}
            </span>
          </div>
        </div>

        <div className="flex-1 min-w-0 overflow-y-auto text-[10px] leading-tight text-gray-400 font-mono max-h-[200px]">
          {game.finished && (
            <div className="mb-0.5 flex items-baseline gap-2">
              {verdictLabel && (
                <span
                  className={`font-mono ${
                    isHallucination
                      ? "font-bold text-red-400"
                      : isCheckmate
                        ? "font-bold text-yellow-300"
                        : isDraw
                          ? "font-bold text-sky-300"
                          : "text-gray-400"
                  }`}
                >
                  {verdictLabel}
                </span>
              )}
              <span className="text-yellow-400 font-mono">{game.result}</span>
            </div>
          )}
          {verboseHistory.length === 0 ? (
            <span className="italic text-gray-600">no moves yet</span>
          ) : (
            verboseHistory.map(({ idx, text }) => (
              <div key={idx} className="truncate" title={text}>
                {idx + 1}. {text}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
