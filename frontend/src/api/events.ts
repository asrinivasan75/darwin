// FROZEN CONTRACT — mirror of backend/cubist/api/websocket.py.
// Do not change without team sync; both files must stay aligned.

export type GenerationStarted = {
  type: "generation.started";
  number: number;
  champion: string;
};

export type StrategistQuestion = {
  type: "strategist.question";
  index: number;
  category: "prompt" | "search" | "book" | "evaluation" | "sampling";
  text: string;
};

export type BuilderCompleted = {
  type: "builder.completed";
  question_index: number;
  engine_name: string;
  ok: boolean;
  error: string | null;
};

export type GameMove = {
  type: "game.move";
  game_id: number;
  fen: string;
  san: string;
  white: string;
  black: string;
  ply: number;
};

export type GameFinished = {
  type: "game.finished";
  game_id: number;
  result: "1-0" | "0-1" | "1/2-1/2";
  termination: string;
  pgn: string;
  white: string;
  black: string;
};

export type GenerationFinished = {
  type: "generation.finished";
  number: number;
  new_champion: string;
  elo_delta: number;
  promoted: boolean;
};

// Emitted by run_generation_task when an asyncio.CancelledError fires —
// either because the user clicked Stop, opened a new generation while one
// was still running, or sent the beforeunload sendBeacon on page reload.
// Frontend uses it to clear in-progress dashboard panels.
export type GenerationCancelled = {
  type: "generation.cancelled";
  number: number;
};

// Emitted by POST /api/state/clear after the operator wipes the database
// and on-disk generated engines. The frontend handles this in
// useEventStream by resetting the accumulated event log to []; every
// panel then renders its empty state. Not stored in the log itself.
export type StateCleared = {
  type: "state.cleared";
};

export type CubistEvent =
  | GenerationStarted
  | StrategistQuestion
  | BuilderCompleted
  | GameMove
  | GameFinished
  | GenerationFinished
  | GenerationCancelled
  | StateCleared;

export type Envelope = { event: CubistEvent };
