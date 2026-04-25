# Development Process

Darwin was built at a weekend hackathon by **five engineers working in parallel**, each driving their own AI coding agent against a set of frozen contracts. The product has two parallelisms: the engine runs LLM strategist + builders concurrently per generation, *and* the build itself ran five concurrent humans + agents over the weekend. This doc covers the build-time process. For the runtime engine pipeline, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Workstream split

Each engineer owned exactly one branch and one plan. Plans live in [plans/](../plans/).

| Branch | Owner | Scope | Plan |
|---|---|---|---|
| `feat/engine-core` | Person A | Engine Protocol, baseline, registry, generated-engine loader. | [person-a-engine-core.md](../plans/person-a-engine-core.md) |
| `feat/tournament` | Person B | Referee, parallel round-robin, Elo, selection gate, eval CLI. | [plans/person-b-tournament.md](../plans/person-b-tournament.md) |
| `feat/agents` | Person C | Strategist, builder, validator, prompts. | [plans/person-c-agents.md](../plans/person-c-agents.md) |
| `feat/frontend` | Person D | Dashboard: live board, Elo chart, strategist feed, bracket. | [plans/person-d-frontend.md](../plans/person-d-frontend.md) |
| `feat/infra` | Person E | API, WebSocket bus, DB, orchestration loop, demo scripts. | [plans/person-e-infra.md](../plans/person-e-infra.md) |

## Frozen contracts as parallelization seams

Five engineers can only build in parallel if they agree on interfaces *up front* and don't touch them. Three contracts (four files) carried that load:

- **Engine Protocol** — what an engine looks like to the tournament.
- **DB schema** — what each workstream persists.
- **WebSocket events** — what the dashboard consumes.

See [ARCHITECTURE.md](ARCHITECTURE.md#frozen-contracts) for the file paths. Every plan declares which frozen contracts it touches; almost always the answer is "consumes, never modifies." When a contract genuinely needed to change (e.g. adding `ratings` to `generation.finished`), it required cross-workstream sign-off and a coordinated frontend update.

## Per-engineer agent workflow

Each engineer ran a coding agent against their own plan in their own worktree. The plans are written for the agent — concrete files to create, exact stubs, test invocations, definition-of-done checklists. The human reviews, integration-tests, and merges.

Two patterns made parallel agent work tractable:

1. **Stub-first integration.** Person C shipped stub `propose_questions` / `build_engine` to a branch on day one, so Person E could wire orchestration end-to-end before the real agents existed. Person D used a `?mock=1` event stream to develop the dashboard without the backend.
2. **`RandomEngine` as universal sparring partner.** B's tournament tests, C's validator smoke games, and A's registry tests all use `RandomEngine` so nothing in CI burns API budget.

## Merge order

Sequenced so each merge unblocks the next person:

1. **A merges `feat/engine-core`.** Engine Protocol + baseline land. Unblocks B and C.
2. **B merges `feat/tournament`.** Round-robin + Elo land. Unblocks E's full orchestration.
3. **C merges `feat/agents`.** Real strategist + builder replace stubs. Unblocks E's real generation runs.
4. **E merges `feat/infra`.** API + WS + orchestration land. Unblocks D's switch from mocks to live.
5. **D merges `feat/frontend`.** Demo-ready.

After D: eval, polish, demo rehearsal. Follow-ups (see [plans/](../plans/)) are post-hackathon polish on the same model — one branch, one owner, frozen contracts respected.

## What worked

- **Frozen contracts up front.** Every workstream had a concrete shape to mock against on day zero. Zero merge conflicts on the four contract files.
- **Stub-first.** Stubs unblocked downstream consumers within hours, not days.
- **Plans written for agents.** Concrete file paths and exact test commands meant the agent could iterate without cross-workstream context.
- **Top-2 lineage.** Carrying the runner-up forward stopped the population from collapsing onto a single line of descent.

## What broke

- **Builder hallucinated `from darwin import config as settings`** on every generated engine — silently fell back to `next(iter(legal_moves))`. Caught and fixed in [followup-2-builder-quality.md](../plans/followup-2-builder-quality.md).
- **Round-robin fired all pairings in parallel** without a semaphore. Under Gemini rate limits every game timed out at once. Fix in [followup-1-tournament-concurrency.md](../plans/followup-1-tournament-concurrency.md).
- **API path didn't resume from the DB champion** — every API-triggered generation started from baseline. Fix in [followup-3-champion-resume.md](../plans/followup-3-champion-resume.md).
- **LiveBoard wasn't rendering moves**, and termination reasons were invisible on the dashboard. Fix in [followup-4-frontend-polish.md](../plans/followup-4-frontend-polish.md).

Each break was a contract *technically* respected but *semantically* wrong (a fallback swallowing errors, an unbounded `gather`, a hardcoded champion). Frozen contracts gave merge-time safety; the follow-ups are runtime fixes the contracts couldn't catch.
