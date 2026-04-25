# Build Plans

The hackathon split: five engineers, five branches, five AI agents, one weekend. The canonical workflow doc is [docs/PROCESS.md](../docs/PROCESS.md); the runtime architecture is [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md). Each plan below conforms to the standard 7-heading template (Goal / Scope / Frozen contracts touched / Deliverables / Acceptance criteria / Risks & mitigations / Status).

## Per-engineer plans

| Plan | Branch | Owner | Status |
|---|---|---|---|
| [Engine core & baseline](./person-a-engine-core.md) | `feat/engine-core` | A | Merged |
| [Tournament & referee](./person-b-tournament.md) | `feat/tournament` | B | Merged |
| [Agents (strategist + builder)](./person-c-agents.md) | `feat/agents` | C | Merged |
| [Frontend dashboard](./person-d-frontend.md) | `feat/frontend` | D | Merged |
| [Infra, API, orchestration, demo](./person-e-infra.md) | `feat/infra` | E | Merged |

## Follow-ups (post-hackathon polish)

| Plan | Branch | Status |
|---|---|---|
| [Tournament concurrency + referee observability](./followup-1-tournament-concurrency.md) | `followup/tournament-concurrency` | Merged |
| [Builder quality gate + prompt fix](./followup-2-builder-quality.md) | `followup/builder-quality` | Merged |
| [Champion resume in `run_generation_task`](./followup-3-champion-resume.md) | `followup/champion-resume` | Merged |
| [Frontend: live-board streaming + termination badges](./followup-4-frontend-polish.md) | `followup/frontend-polish` | Merged |

## Frozen contracts

These define how the workstreams integrate. **Do not change them without paging the team.** See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#frozen-contracts) for full links.

- [backend/darwin/engines/base.py](../backend/darwin/engines/base.py) — Engine Protocol
- [backend/darwin/storage/models.py](../backend/darwin/storage/models.py) — SQLite schema
- [backend/darwin/api/websocket.py](../backend/darwin/api/websocket.py) + [frontend/src/api/events.ts](../frontend/src/api/events.ts) — WS event payloads

## Merge order

1. Person A merges `feat/engine-core`. Unblocks B and C.
2. Person B merges `feat/tournament`. Unblocks E's full orchestration.
3. Person C merges `feat/agents`. Unblocks E's real generation runs.
4. Person E merges `feat/infra`. Unblocks D's switch from mocks to live.
5. Person D merges `feat/frontend`. Demo-ready.

After D merges: eval, polish, demo rehearsal. Follow-ups land on the same one-branch / one-owner discipline.
