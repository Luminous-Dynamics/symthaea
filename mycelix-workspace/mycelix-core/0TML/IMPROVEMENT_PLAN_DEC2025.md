# 0TML Improvement Plan — December 2025

**Author:** Codex agent  
**Date:** 12 Nov 2025  
**Scope:** Bring repo reality back in sync with Gen 5 claims, close critical security gaps, and unblock Winterfell benchmarking.

---

## 1. Evidence Snapshot

| Area | Current State | Impact |
| --- | --- | --- |
| Winterfell prover | Documentation references `vsv-stark/winterfell-pogq`, but the crate is missing from the workspace. | Onboarding blockers, paper cannot be regenerated. |
| Node / coordinator | `Node.start`/`train` empty, CLI still exposes them as production-ready. | Demo breaks, undermines README “147/147 tests” message. |
| Security & compliance | “Encrypted” gradients are plain JSON (`Phase10Coordinator._decrypt_gradient`), configs live in `~/.zerotrustml` without protections. | HIPAA/GDPR claims untrue. |
| Testing | `pytest` not installed, many “tests” are interactive demos with undefined variables (e.g. `test_phase10_real`). | No CI signal, regressions slip through unnoticed. |
| Docs | Mixed reality: README says Gen 5 complete, but quick-start fails immediately. | Confuses collaborators, reviewers, and grant partners. |

---

## 2. Guiding Objectives (Next 6 Weeks)

1. **Repo ↔ Reality Alignment**  
   - Either commit the Winterfell crate or downgrade the docs to accurately describe the available RISC Zero prover.  
   - Gate READMEs behind automated smoke tests before each update.

2. **Operational Node Pipeline**  
   - Deliver an end-to-end happy path: CLI `zerotrustml start --rounds 1` should train locally, aggregate using the existing `aggregation.algorithms`, and emit checkpoints.  
   - Replace bespoke MAD/Krum logic with the shared module to avoid code drift.

3. **Credible Security Posture**  
   - Ship authenticated encryption (libsodium/NaCl) for gradients at rest/in transit.  
   - Introduce a managed configuration store (TOML/YAML) and remove plaintext secrets from `$HOME`.  
   - Document minimum compliance story (HIPAA/GDPR) backed by actual code.

4. **Deterministic Testing & Tooling**  
   - Provide a `poetry install --with dev` (or `nix develop`) workflow that installs pytest and fixtures.  
   - Convert demo scripts into real pytest modules with mocks so CI does not require Postgres/Holochain.  
   - Add `smoke` and `full` tox profiles; wire to GitHub Actions once repo is public again.

5. **Observability & Docs**  
   - Instrument coordinator and backends with structured logs + Prometheus counters.  
   - Replace aspirational README sections with living status docs generated from CI metadata.

---

## 3. Workstream Breakdown

### WS-A: Winterfell / Proof Backends
| Week | Deliverable |
| --- | --- |
| W1 | Recover or reimplement `winterfell-pogq` crate (air.rs, prover.rs, CLI) under version control. |
| W2 | Add `cargo test -p winterfell-pogq` to CI; script to emit `results/table_vii_bis_dual_backend.tex`. |
| W3 | Benchmark harness comparing RISC Zero vs Winterfell using shared fixtures; update paper tables. |

### WS-B: Federated Node & CLI
| Week | Deliverable |
| --- | --- |
| W1 | Finish `Node.start/train`: local dataset, asyncio tasks, stub peer exchange. |
| W2 | Coordinator uses `aggregation.aggregate_gradients` with configurable algorithms; add unit tests. |
| W3 | CLI UX polish (`zerotrustml status`, `zerotrustml credits`) with real backend calls. |

### WS-C: Security / Compliance
| Week | Deliverable |
| --- | --- |
| W1 | Introduce `zerotrustml.crypto` with libsodium wrappers; encrypt gradients before storage. |
| W2 | Config secrets move to `config/` with dotenv/TOML loader; add key rotation guide. |
| W3 | HIPAA/GDPR checklist appended to `docs/compliance/README.md`, tied to actual code paths. |

### WS-D: Testing & Tooling
| Week | Deliverable |
| --- | --- |
| W0 | `poetry install --with dev` succeeds; pytest available (current blocker). |
| W1 | Convert smoke/demo scripts into pytest cases with fixtures (no external services). |
| W2 | Add `nox -s smoke` / GitHub Action that runs on every push; badges reflect real status. |

### WS-E: Observability & Documentation
| Week | Deliverable |
| --- | --- |
| W1 | Coordinator exposes `/metrics` with queue depth, backend latency, rejection counts. |
| W2 | README quick-start auto-generated from verified scripts; `STATUS_DASHBOARD.md` pulls CI artifacts. |
| W3 | Experiment/paper figures linked to reproducible scripts (table generators, etc.). |

---

## 3.1 Near-Term Sprint Backlog (Week of Nov 12)

| # | Task | Owner | Validation | Status |
| --- | --- | --- | --- | --- |
| S0-1 | Install dev toolchain & ensure `poetry run pytest tests/test_smoke.py` passes (establish baseline) | Core | ✔︎ 12 Nov run (2 tests, warnings noted) | ✅ Complete |
| S0-2 | Publish Winterfell status note and update onboarding/docs quick-starts to point at it; freeze unsupported instructions | Docs/zk team | README/`README_START_HERE`/`NEXT_SESSION_*`/`STATUS_DASHBOARD` reference `docs/status/WINTERFELL_STATUS_DEC2025.md`; smoke test covers doc commands | ✅ Complete |
| S0-3 | Decide recovery vs. removal for `winterfell-pogq` (search archives, branches, backups) | ZK owner | Meeting notes + issue in tracker; see update in `docs/status/WINTERFELL_STATUS_DEC2025.md` | ⚠️ Blocked (need off-repo backup) |
| S0-4 | Flesh out `Node.start/train` happy path using existing aggregation module; add targeted pytest | Core FL | `tests/test_node_pipeline.py` (async happy-path) | ✅ Complete |
| S0-5 | Replace plaintext gradient handling with libsodium-based helper; add encryption unit tests | Security | `tests/test_encryption.py` | ⏳ Pending |

Sprint exit criteria: documentation matches reality, decision logged for Winterfell, and at least one production path (node pipeline or encryption) merged behind green smoke suite.

---

## 3.2 Sprint Backlog (Week of Nov 18)

| # | Task | Owner | Validation | Status |
| --- | --- | --- | --- | --- |
| S1-1 | Recover or formally de-scope `winterfell-pogq` (escalate to archival owners) | ZK owner | Decision recorded in `docs/status/WINTERFELL_STATUS_DEC2025.md` + issue link | ⚠️ Blocked awaiting off-repo backup |
| S1-2 | Add `zerotrustml.crypto` helpers (AES-GCM) and cover with `tests/test_encryption.py` | Security | Encryption unit test verifying encrypt/decrypt + tamper detection | ✅ Complete |
| S1-3 | Extend coordinator aggregation to reuse `aggregate_gradients` helper and add pytest coverage | Core FL | `tests/test_coordinator_aggregation.py` validates delegation | ✅ Complete |
| S1-4 | Document gen7 RISC Zero bindings status & integration path for zkVM-only flow | Docs/zk team | `docs/status/GEN7_STATUS_DEC2025.md`; references to `gen7-zkstark` code | ✅ Complete |
| S1-5 | Publish updated quick-start scripts auto-generated from runnable commands (link from README) | Docs/tooling | README quick-start reflects scripted commands verified in CI | ✅ Complete (README updated with poetry commands) |

Sprint exit criteria: encryption utilities landed with tests, coordinator aggregation uses shared code, and Winterfell path is either recovered or formally de-scoped with documentation updated.

---

## 4. Immediate Action Items
1. Install dev dependencies locally (`poetry install --with dev`); document steps in `README_START_HERE`.
2. Fix high-visibility broken tests (e.g. undefined `result` in `test_phase10_real.py`) so pytest can run.
3. Decide (within 48 h) whether Winterfell code can be recovered; if not, log issue and edit docs.
4. Schedule weekly check-ins with owners per workstream; track in `STATUS_DASHBOARD.md`.

---

## 5. Risks & Mitigations
| Risk | Mitigation |
| --- | --- |
| Missing Winterfell code blocks paper submission | Prioritize repo recovery; if impossible, scope paper to zkVM only and document accordingly. |
| Security work delayed by cryptography dependencies | Use existing libsodium bindings; fall back to PyNaCl if nix packaging lags. |
| Testing remains flaky due to external services | Build mock layers for Postgres/Holochain; run integration suite nightly with real services. |
| Team capacity split between research and infra | Assign dedicated owner per workstream; guard research timeboxes. |

---

## 6. Definition of Done
* `poetry run pytest tests/smoke_test_activated_code.py` passes on fresh clone.  
* README quick-start reproduces on CI runner.  
* Winterfell + RISC Zero benchmark scripts produce both Table VII versions.  
* Coordinator encrypts gradients, and credits persist in a backend, not memory.  
* Status dashboard reflects actual CI/test metrics with <24 h drift.
