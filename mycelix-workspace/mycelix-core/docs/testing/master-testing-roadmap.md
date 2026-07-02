# Master Testing Roadmap (Phases 0–1)

_Last updated: 2025-10-21_

This living document ties together every testing phase and links to the canonical specs/results.

## Phase 0 – Foundational Validation (2024 – Q1 2025)
- **Simulation Benchmarks:** Early PoGQ simulation work (see `SPIKE_TESTS_SUMMARY.md`, `ACCURACY_IMPROVEMENT_STRATEGIES.md`).
- **Rust/Holochain Prototyping:** Baseline integration (`CONDUCTOR_WRAPPER_INTEGRATION_GUIDE.md`).
- **Outcome:** Confidence in individual components but not yet joined during live execution.

## Phase 1a – Hybrid Trust Layer (Phase 3, 2025 Q2)
- Summary docs: `0TML/PHASE_3_COMPLETE.md`, `0TML/PHASE_3.2_TRUST_COMPLETE.md`.
- Achieved 100% detection in controlled cold-start tests (see `0TML/docs/SESSION_1_INTEGRATION_COMPLETE.md`).
- Key artifacts: `hybrid_zerotrustml_complete.py`, reputation gap analysis data.

## Phase 1b – Holochain Integration (Phase 7)
- Primary docs: `0TML/PHASE7_ADMINWEBSOCKET_COMPLETE.md`, `0TML/PHASE_7_FINAL_STATUS.md`.
- Validated WebSocket auth, DHT checkpointing, reconnection behavior.
- Outstanding risks at that time: infrastructure flakiness (now resolved via new flake/dev shell).

## Phase 2 – Production-Grade Multi-Backend (Phase 10)
- Canonical summary: `0TML/PHASE10_IMPLEMENTATION_STATUS.md`, `0TML/docs/COMPLETE_SESSION_SUMMARY.md`.
- Holochain + PostgreSQL + optional LocalFile backends via `zerotrustml.backends`.
- Regression suite: `tests/test_zerotrustml_credits_integration.py` (now part of CI shell).

## Current Submission Sprint (Oct 20 – Nov 16, 2025)
- Detailed plan: `0TML/docs/06-architecture/0TML Testing Status & Completion Roadmap.md` (v2.0).
- Working log: [`week-2025-10-20.md`](week-2025-10-20.md).
- Tooling: `flake.nix`, `Justfile`, Poetry-managed deps (`0TML/pyproject.toml`).

### Critical Checks Before Submission
1. CIFAR-10 BFT 40%/50% trials (10 seeds each).
2. Sleeper agent resilience (stateful vs stateless).
3. Statistical appendix (mean ± std, p-values).
4. 300 DPI figures with error bands.
5. Abstract/tables updated with empirical data.

## Phase 1 (Post-Submission, Nov 2025 – Apr 2026)
- **ByzFL integration:** bring Zero-TrustML defense into ByzFL baseline (Month 2).
- **FedGuard hybridization:** integrate PoGQ + FedGuard membership scoring (Month 2).
- **Multi-dataset evaluation:** Fashion-MNIST, CIFAR-100, MedMNIST (Months 3–4).
- **Adaptive attacks:** multi-phase, sequential adversaries (Month 4).
- **Operational stress:** Five Eyes coalition sim, DDIL network resilience (Months 5–6).
- **Red team audit:** external penetration test (Month 6).

Each backlog item will graduate from this list into a dated sprint log when scheduled.

## Documentation Signals
- Sprint logs live in `docs/testing/week-YYYY-MM-DD.md`.
- Final PDFs/figures stored under `results/` with config hashes.
- Keep this roadmap in sync whenever a phase completes or new backlog items emerge.
