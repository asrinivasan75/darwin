/**
 * client.ts — REST and WebSocket client for the Darwin backend API.
 *
 * All requests are relative so Vite's dev-server proxy (vite.config.ts)
 * transparently forwards them to `http://127.0.0.1:8000` in development
 * and the same-origin server in production.
 *
 * WebSocket path: `/ws`  → proxied to `ws://127.0.0.1:8000/ws`
 * REST base:      `/api` → proxied to `http://127.0.0.1:8000/api`
 *
 * Event shapes are defined in {@link ./events} (frozen contract — do not
 * redefine them here).
 *
 * @module client
 */

import type { Envelope } from "./events";

/**
 * Fetches the full list of engines from the REST API.
 *
 * @returns raw JSON response from `GET /api/engines`
 */
export async function fetchEngines(): Promise<unknown> {
  const r = await fetch("/api/engines");
  return r.json();
}

/**
 * Fetches the list of all generations from the REST API.
 *
 * @returns raw JSON response from `GET /api/generations`
 */
export async function fetchGenerations(): Promise<unknown> {
  const r = await fetch("/api/generations");
  return r.json();
}

/**
 * Fetches the Python source for an engine by name.
 *
 * @param name - engine name from the generations/engines API
 * @returns raw Python source from `GET /api/engines/{name}/code`
 */
export async function fetchEngineCode(name: string): Promise<string> {
  const r = await fetch(`/api/engines/${encodeURIComponent(name)}/code`);
  if (!r.ok) {
    throw new Error(`Could not load source for ${name} (${r.status})`);
  }
  return r.text();
}

/**
 * Fetches games for a specific generation.
 *
 * @param gen - generation number to filter by
 * @returns raw JSON response from `GET /api/games?gen={gen}`
 */
export async function fetchGames(gen: number): Promise<unknown> {
  const r = await fetch(`/api/games?gen=${gen}`);
  return r.json();
}

/**
 * Opens a WebSocket connection to the backend event bus and forwards every
 * incoming {@link DarwinEvent} to the provided callback.
 *
 * The caller is responsible for closing the returned WebSocket on cleanup
 * (e.g. from a React useEffect return value).
 *
 * Messages are expected to arrive as JSON-encoded {@link Envelope} objects:
 * `{ "event": { "type": "...", ... } }`
 *
 * @param onEvent - invoked once per arriving event, synchronously after JSON parse
 * @returns the raw WebSocket instance; call `.close()` to disconnect
 *
 * @example
 * ```ts
 * const ws = connectEvents((e) => console.log(e.type));
 * // later:
 * ws.close();
 * ```
 */
export function connectEvents(onEvent: (e: Envelope["event"]) => void): WebSocket {
  const ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onmessage = (msg) => {
    const env = JSON.parse(msg.data) as Envelope;
    onEvent(env.event);
  };

  ws.onerror = (err) => {
    console.error("[darwin] WebSocket error", err);
  };

  return ws;
}
