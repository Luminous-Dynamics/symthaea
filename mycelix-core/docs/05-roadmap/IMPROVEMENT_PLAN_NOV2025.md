# Zero-TrustML + Mycelix Improvement Plan (November 2025)

> Goal: align the codebase, documentation, and automation with the public claims (100 % Byzantine detection, 45 % BFT tolerance, production readiness) so that every assertion is verifiable and every subsystem has a clear owner + timeline.

---

## 1. Verifiable Performance Evidence

**Objective:** turn README claims (100 % detection, 0.7 ms latency, 147/147 tests) into reproducible artefacts published with each release.

| Deliverable | Description | Target | Owner |
|-------------|-------------|--------|-------|
| Benchmark harness automation | Wrap `scripts/run_attack_matrix.py`, validation sweeps, and latency microbenches in a single `nix develop -c just benchmarks` entry point that emits JSON + plots to `artifacts/benchmarks/<date>/`. | Week 1 | ML + Infra |
| CI artefact upload | Extend GitHub Actions to upload the benchmark bundle + `results/*.json` on every `main` push and link to it from README badges. | Week 2 | DevEx |
| Evidence links in docs | Update README + 0TML README to reference the latest benchmark artefact (auto-updated badge) so visitors can click through to raw data. | Week 3 | Docs |

**Success criteria:** every release tag includes a pointer to measurement artefacts; README claims reference dated evidence.

---

## 2. Production-Grade Persistence

**Objective:** finish the hybrid audit trail (PostgreSQL + Holochain) so gradients, PoGQ proofs, and committee votes survive process restarts.

| Deliverable | Description | Target | Owner |
|-------------|-------------|--------|-------|
| Async PostgreSQL backend | Implement asyncpg CRUD with schema migrations, checksum validation, and integration tests under `tests/integration/test_postgres_backend.py`. | Week 2 | Backend |
| Holochain round-trip tests | Pair the WebSocket backend with conductor fixtures to prove store/retrieve/list flows; capture fixtures under `tests/holochain/`. | Week 3 | Holochain |
| Storage strategy validation | Add test matrix ensuring `Phase10Coordinator` writes to all configured backends (all/primary/quorum) with failure injection. | Week 4 | FL Core |

**Success criteria:** test suite covers full persistence flows; documentation no longer calls out “mock implementation”.

---

## 3. Unified CI / CD Pipeline

**Objective:** collapse `.github/workflows/ci.yml` and `.github/workflows/ci.yaml` into one matrix that runs Rust, Python, docs, and benches with shared caching.

| Deliverable | Description | Target | Owner |
|-------------|-------------|--------|-------|
| Workflow consolidation | Create `ci.yaml` with matrix axes `{lang: rust/python/docs}`, re-use nix shells where required, delete duplicate workflow. | Week 1 | DevEx |
| Artefact + caching strategy | Share Cargo, Poetry, and MkDocs caches using `actions/cache` to keep runtimes under 10 min. | Week 2 | DevEx |
| Required status update | Update branch protections + CONTRIBUTING to reference the new workflow names. | Week 2 | DevEx |

**Success criteria:** single CI badge, reduced maintenance, consistent gating across repos.

---

## 4. Product Focus & Repo Hygiene

**Objective:** clarify maturity of adjacent products (marketplace, desktop) and ensure Zero-TrustML remains the primary deliverable in this repo.

| Deliverable | Description | Target | Owner |
|-------------|-------------|--------|-------|
| Component maturity table | Add CONTRIBUTING section that labels each sub-project (core, marketplace, desktop, websites) as `production`, `beta`, or `experimental`. | Week 1 | Docs |
| Repo split decision | Decide whether marketplace/desktop move to their own repos; document migration plan if yes. | Week 3 | Product |
| Subtree CI gating | Optional: introduce lightweight lint/test workflows for marketplace/desktop if they remain in monorepo. | Week 4 | Frontend |

**Success criteria:** contributors know where to focus; CI failures in experimental areas cannot block core releases.

---

## 5. Documentation Experience

**Objective:** use the live MkDocs site (mycelix.net) as the canonical entry point and reduce duplication across README/CLAUDE/root notes.

| Deliverable | Description | Target | Owner |
|-------------|-------------|--------|-------|
| Canonical doc badge | Add “Docs @ mycelix.net” badge to README + CLAUDE + 0TML README pointing to the MkDocs site. | Week 1 | Docs |
| Page freshness tags | Introduce admonitions (`!!! info "Status: Current (Nov 2025)"`) across key docs to signal whether content is canonical, archived, or roadmap. | Week 2 | Docs |
| Root-notes triage | Move historical session logs to `/docs/archive/` and link them via MkDocs `tags` so newcomers land on current guidance first. | Week 4 | Docs |

**Success criteria:** single navigation path via MkDocs; README stays concise with links to living docs; contributors report lower onboarding friction.

---

**Tracking & Communication**

- Progress reviewed in weekly stand-ups; update this plan as milestones complete.
- Add checklist issues in GitHub Projects for each deliverable to keep ownership visible.
- Publish monthly status summaries under `docs/root-notes/` referencing this plan.

Let’s make the claims provable, the storage durable, CI reliable, scope intentional, and documentation delightful. 💫
