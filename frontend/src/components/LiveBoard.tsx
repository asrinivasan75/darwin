/**
 * LiveBoard.tsx — real-time chess board showing the most recently active game.
 *
 * Tracks the last {@link GameMove} event across all games and renders the
 * resulting FEN with react-chessboard. Because a real generation can emit
 * ~2400 game.move events, this component intentionally only cares about the
 * single most recent move — earlier moves are silently discarded.
 *
 * The "thinking…" indicator activates after 2 seconds of no new move event for
 * the active game, signalling that the engine is deliberating. It resets as
 * soon as the next move arrives.
 *
 * @listens {GameMove}     - updates the board position and resets the thinking timer
 * @listens {GameFinished} - not directly consumed; board remains on last position
 *
 * @module LiveBoard
 */

import { useEffect, useRef, useState } from "react";
import { Chessboard } from "react-chessboard";
import type { CubistEvent, GameMove } from "../api/events";

/** Props accepted by {@link LiveBoard}. */
interface LiveBoardProps {
  /** Full accumulated event log from {@link useEventStream}. */
  events: CubistEvent[];
}

/** How long (ms) to wait after the last move before showing "thinking…". */
const THINKING_DELAY_MS = 2000;

/**
 * LiveBoard — renders the chess board at the position from the most recent
 * game.move event, and shows a "thinking…" badge when the engine is slow.
 *
 * @param props.events - the full accumulated event log from useEventStream()
 * @returns a dark card containing the chess board, player names, and status
 */
export default function LiveBoard({ events }: LiveBoardProps) {
  const [thinking, setThinking] = useState(false);

  // Only consider game.move events
  const moveEvents = events.filter(
    (e): e is GameMove => e.type === "game.move"
  );

  // The most recent move across all games is the one to display.
  // This intentionally renders only one board — the "hot" game judges can watch.
  const lastMove = moveEvents[moveEvents.length - 1] as GameMove | undefined;

  // Unique key for the current board state: changes on every new move.
  // Using game_id + ply avoids treating a new game's ply-1 as "no change".
  const moveKey = lastMove ? `${lastMove.game_id}-${lastMove.ply}` : "";

  // Ref tracks the key we last set the timer for, preventing the effect from
  // running again if the events array reference changes but no new move arrived.
  const prevKeyRef = useRef<string>("");

  useEffect(() => {
    if (moveKey === prevKeyRef.current) return;
    prevKeyRef.current = moveKey;

    // A new move arrived — engine is clearly not thinking right now
    setThinking(false);

    const timer = setTimeout(() => setThinking(true), THINKING_DELAY_MS);
    return () => clearTimeout(timer);
  }, [moveKey]);

  const fen = lastMove?.fen ?? "start";
  const white = lastMove?.white ?? "White";
  const black = lastMove?.black ?? "Black";
  const ply = lastMove?.ply ?? 0;

  return (
    <div className="bg-gray-800 rounded-lg p-4 flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs font-semibold tracking-wider text-gray-400 uppercase">
          Live Board
        </h2>
        {/* Thinking indicator: visible only when the engine is deliberating */}
        {lastMove && thinking && (
          <span className="text-xs text-yellow-400 animate-pulse font-medium">
            thinking…
          </span>
        )}
        {lastMove && !thinking && (
          <span className="text-xs text-gray-500">
            ply {ply}
          </span>
        )}
      </div>

      {/* Black player name (shown above the board as per chess convention) */}
      <PlayerLabel name={black} side="black" />

      <div className="my-2">
        <Chessboard
          position={fen}
          arePiecesDraggable={false}
          customDarkSquareStyle={{ backgroundColor: "#374151" }}
          customLightSquareStyle={{ backgroundColor: "#9ca3af" }}
          boardWidth={260}
        />
      </div>

      {/* White player name (shown below the board) */}
      <PlayerLabel name={white} side="white" />

      {!lastMove && (
        <p className="text-gray-500 text-xs text-center mt-2 italic">
          Waiting for first game…
        </p>
      )}
    </div>
  );
}

// ── Internal sub-components ──────────────────────────────────────────────────

/** Props for the player name label. */
interface PlayerLabelProps {
  name: string;
  side: "white" | "black";
}

/**
 * Renders a player name label with a small colour swatch indicating the side.
 *
 * @param props.name - engine name to display
 * @param props.side - "white" or "black", used to pick the swatch colour
 */
function PlayerLabel({ name, side }: PlayerLabelProps) {
  const swatchClass =
    side === "white"
      ? "bg-gray-100 border border-gray-500"
      : "bg-gray-900 border border-gray-500";

  return (
    <div className="flex items-center gap-2">
      <span className={`inline-block w-3 h-3 rounded-sm shrink-0 ${swatchClass}`} />
      <span className="text-gray-300 text-xs font-mono truncate" title={name}>
        {name}
      </span>
    </div>
  );
}
