/**
 * mockEvents.ts — offline development fixture for the Cubist dashboard.
 *
 * Simulates a full two-generation lifecycle without a live backend. Each event
 * is delivered on a real timer so animations and state transitions behave
 * exactly as they will in production. Pass `?mock` (or `?mock=1`) in the URL
 * to activate this stream via {@link useEventStream}.
 *
 * The sequence covers:
 *   Gen 1: strategist questions → builder completions → games → generation finish
 *   Gen 2: strategist questions → a handful of game moves (in-progress state)
 *
 * @module mockEvents
 */

import type { CubistEvent } from "../api/events";

/**
 * Starts a mock WebSocket-like event stream that fires {@link CubistEvent}s
 * on real wall-clock timers, replicating the pacing of a live generation run.
 *
 * @param onEvent - callback invoked for each event as it "arrives"
 * @returns cleanup function that cancels all pending timeouts
 */
export function startMockStream(onEvent: (e: CubistEvent) => void): () => void {
  /** Each entry defines how long after the *previous* event this one fires. */
  const seq: { delay: number; event: CubistEvent }[] = [
    // ── Generation 1 startup ──────────────────────────────────────────────
    {
      delay: 0,
      event: { type: "generation.started", number: 1, champion: "baseline-v0" },
    },

    // ── Strategist phase: 5 improvement hypotheses proposed by the LLM ───
    {
      delay: 900,
      event: {
        type: "strategist.question",
        index: 0,
        category: "book",
        text: "Would adding a 6-move opening book reduce early blunders?",
      },
    },
    {
      delay: 700,
      event: {
        type: "strategist.question",
        index: 1,
        category: "prompt",
        text: "Would prompting for threat detection first improve tactics?",
      },
    },
    {
      delay: 700,
      event: {
        type: "strategist.question",
        index: 2,
        category: "search",
        text: "Would 1-ply lookahead with LLM eval beat raw move selection?",
      },
    },
    {
      delay: 700,
      event: {
        type: "strategist.question",
        index: 3,
        category: "evaluation",
        text: "Would scoring positions before moving improve endgames?",
      },
    },
    {
      delay: 700,
      event: {
        type: "strategist.question",
        index: 4,
        category: "sampling",
        text: "Would majority vote across 3 LLM samples reduce move variance?",
      },
    },

    // ── Builder phase: candidate engines generated (one per question) ─────
    // question_index 2 (search) fails due to a smoke-game timeout
    {
      delay: 2200,
      event: {
        type: "builder.completed",
        question_index: 0,
        engine_name: "gen1-book-a3f",
        ok: true,
        error: null,
      },
    },
    {
      delay: 600,
      event: {
        type: "builder.completed",
        question_index: 1,
        engine_name: "gen1-prompt-b1d",
        ok: true,
        error: null,
      },
    },
    {
      delay: 600,
      event: {
        type: "builder.completed",
        question_index: 2,
        engine_name: "gen1-search-c2e",
        ok: false,
        error: "smoke game timeout after 20 s",
      },
    },
    {
      delay: 600,
      event: {
        type: "builder.completed",
        question_index: 3,
        engine_name: "gen1-eval-d7g",
        ok: true,
        error: null,
      },
    },
    {
      delay: 600,
      event: {
        type: "builder.completed",
        question_index: 4,
        engine_name: "gen1-sample-e5h",
        ok: true,
        error: null,
      },
    },

    // ── Game phase: round-robin tournament moves (game 1) ─────────────────
    // Each game.move event carries the board FEN so LiveBoard can render it.
    {
      delay: 1200,
      event: {
        type: "game.move",
        game_id: 1,
        fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        san: "e4",
        white: "baseline-v0",
        black: "gen1-book-a3f",
        ply: 1,
      },
    },
    {
      delay: 1500,
      event: {
        type: "game.move",
        game_id: 1,
        fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        san: "e5",
        white: "baseline-v0",
        black: "gen1-book-a3f",
        ply: 2,
      },
    },
    {
      delay: 1500,
      event: {
        type: "game.move",
        game_id: 1,
        fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        san: "Nf3",
        white: "baseline-v0",
        black: "gen1-book-a3f",
        ply: 3,
      },
    },
    {
      delay: 1500,
      event: {
        type: "game.move",
        game_id: 1,
        fen: "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        san: "Nc6",
        white: "baseline-v0",
        black: "gen1-book-a3f",
        ply: 4,
      },
    },
    {
      delay: 1500,
      event: {
        type: "game.move",
        game_id: 1,
        fen: "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        san: "Bb5",
        white: "baseline-v0",
        black: "gen1-book-a3f",
        ply: 5,
      },
    },
    // Game 1 finishes — gen1-book-a3f wins
    {
      delay: 3000,
      event: {
        type: "game.finished",
        game_id: 1,
        result: "0-1",
        termination: "checkmate",
        pgn: "1. e4 e5 2. Nf3 Nc6 3. Bb5",
        white: "baseline-v0",
        black: "gen1-book-a3f",
      },
    },

    // ── Game 2: different pairing ─────────────────────────────────────────
    {
      delay: 800,
      event: {
        type: "game.move",
        game_id: 2,
        fen: "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
        san: "d4",
        white: "gen1-prompt-b1d",
        black: "gen1-eval-d7g",
        ply: 1,
      },
    },
    {
      delay: 1500,
      event: {
        type: "game.move",
        game_id: 2,
        fen: "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
        san: "d5",
        white: "gen1-prompt-b1d",
        black: "gen1-eval-d7g",
        ply: 2,
      },
    },
    // Game 2 finishes — draw
    {
      delay: 2500,
      event: {
        type: "game.finished",
        game_id: 2,
        result: "1/2-1/2",
        termination: "stalemate",
        pgn: "1. d4 d5",
        white: "gen1-prompt-b1d",
        black: "gen1-eval-d7g",
      },
    },

    // Game 3 result (no moves shown — simulates a game that ran in the background)
    {
      delay: 1000,
      event: {
        type: "game.finished",
        game_id: 3,
        result: "1-0",
        termination: "checkmate",
        pgn: "1. e4 c5",
        white: "gen1-book-a3f",
        black: "gen1-sample-e5h",
      },
    },
    {
      delay: 800,
      event: {
        type: "game.finished",
        game_id: 4,
        result: "1-0",
        termination: "resignation",
        pgn: "1. d4 Nf6",
        white: "gen1-eval-d7g",
        black: "baseline-v0",
      },
    },

    // ── Generation 1 finishes — gen1-book-a3f promoted as new champion ───
    {
      delay: 1500,
      event: {
        type: "generation.finished",
        number: 1,
        new_champion: "gen1-book-a3f",
        elo_delta: 28.5,
        promoted: true,
      },
    },

    // ── Generation 2 startup ──────────────────────────────────────────────
    {
      delay: 2000,
      event: {
        type: "generation.started",
        number: 2,
        champion: "gen1-book-a3f",
      },
    },

    // ── Gen 2 strategist questions ────────────────────────────────────────
    {
      delay: 900,
      event: {
        type: "strategist.question",
        index: 0,
        category: "search",
        text: "Would 2-ply alpha-beta pruning outperform single-ply LLM?",
      },
    },
    {
      delay: 700,
      event: {
        type: "strategist.question",
        index: 1,
        category: "evaluation",
        text: "Would piece-square tables improve positional scoring?",
      },
    },
    {
      delay: 700,
      event: {
        type: "strategist.question",
        index: 2,
        category: "prompt",
        text: "Would a chain-of-thought reasoning prompt reduce blunders?",
      },
    },
    {
      delay: 700,
      event: {
        type: "strategist.question",
        index: 3,
        category: "book",
        text: "Would a 10-move opening book cover more common lines?",
      },
    },
    {
      delay: 700,
      event: {
        type: "strategist.question",
        index: 4,
        category: "sampling",
        text: "Would beam search over top-5 moves beat greedy selection?",
      },
    },

    // ── Gen 2 builders completing ─────────────────────────────────────────
    {
      delay: 2200,
      event: {
        type: "builder.completed",
        question_index: 0,
        engine_name: "gen2-search-f4a",
        ok: true,
        error: null,
      },
    },
    {
      delay: 600,
      event: {
        type: "builder.completed",
        question_index: 1,
        engine_name: "gen2-eval-g9b",
        ok: true,
        error: null,
      },
    },

    // ── Gen 2 in-progress game moves (dashboard shows live board) ─────────
    {
      delay: 1200,
      event: {
        type: "game.move",
        game_id: 5,
        fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        san: "e4",
        white: "gen1-book-a3f",
        black: "gen2-search-f4a",
        ply: 1,
      },
    },
    {
      delay: 1500,
      event: {
        type: "game.move",
        game_id: 5,
        fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        san: "e5",
        white: "gen1-book-a3f",
        black: "gen2-search-f4a",
        ply: 2,
      },
    },
    {
      delay: 1800,
      event: {
        type: "game.move",
        game_id: 5,
        fen: "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq f3 0 2",
        san: "f4",
        white: "gen1-book-a3f",
        black: "gen2-search-f4a",
        ply: 3,
      },
    },
  ];

  const timers: ReturnType<typeof window.setTimeout>[] = [];
  let accumulated = 0;

  for (const { delay, event } of seq) {
    accumulated += delay;
    timers.push(window.setTimeout(() => onEvent(event), accumulated));
  }

  // Return cleanup so useEventStream can cancel pending timeouts on unmount
  return () => timers.forEach((id) => clearTimeout(id));
}
