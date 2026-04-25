# Shortcomings

Known limitations. Deliberate punts given the timeline, not surprises.

- **Selection is by tournament score, not Elo.** Highest cohort score wins; ties are broken randomly. Elo is persisted and shown but does not gate promotion. Over many generations this can let a slightly-noisier engine displace a slightly-stronger one.
- **No time-decay on Elo.** An engine that only ever played in generation 1 holds its gen-1 Elo forever (forward-fill in the chart). Useful for legibility, misleading for absolute strength comparisons.
- **Strategist question pool is small.** On the deterministic-strategist branch the rotation cycles after ~5 generations per category. On the LLM-strategist branch novelty depends on whatever the model is willing to suggest given recent history — empirically that also plateaus.
- **Builder failures are common at first.** ~30–50% of generated engines hit a static gate on the first turn. Logging tells you which gate killed which candidate, but the failure rate eats into the per-gen cohort size.
- **Mid-tournament dashboard state is stale.** The bracket's blue-highlighted "incumbent" row tracks the champion *coming into* the gen and only flips when `generation.finished` fires. Screenshotting mid-tournament shows stale state.
- **Single-machine SQLite.** Persistence is a SQLite file (`backend/darwin.db`). Fine for a hackathon and for `make replay`; not suitable for a long-running, multi-tenant deployment.
- **No resume.** If a generation crashes or the backend restarts mid-tournament, the partial state is dropped. The orchestrator restarts from the last persisted champion.
- **Modal backend is opt-in.** It works and is faster, but it's not the default — there are still some sharp edges around stale-event drainage when a generation is cancelled.
- **Pure-code branch is local-only.** It does not ship to `main`. Merging would require also bringing across the Modal deployment, the model env vars, and an API contract change for `ratings` on `generation.finished`.
