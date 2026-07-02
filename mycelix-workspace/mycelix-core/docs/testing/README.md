# Testing Integration Hub

This folder unifies status and plans across **all testing phases**.

## Key References
- [`master-testing-roadmap.md`](master-testing-roadmap.md) – snapshot across Phases 0–1.
- [`week-2025-10-20.md`](week-2025-10-20.md) – active sprint log (Oct 20 – Nov 16).
- `../0TML/docs/06-architecture/0TML Testing Status & Completion Roadmap.md` – detailed Zero-TrustML roadmap v2.0.
- `../VALIDATED_CLAIMS.md` – mapping from public claims to concrete tests/benchmarks.

## How to Contribute Results
1. Run experiments from within the reproducible shell (`nix develop`).
2. Save outputs (metrics, figures) under `results/` with seed/config metadata.
3. Update the sprint log (`week-YYYY-MM-DD.md`) and the master roadmap with status/links.
4. Regenerate figures with `scripts/plots/*.py` (ensure 300 DPI PDFs).

## Backlog Tracking
Deferred items (ByzFL integration, FedGuard baselines, multi-dataset expansion, etc.) are parked in the Phase 1 section of the master roadmap and will be promoted into sprint logs when scheduled.

---

## Test Layers Overview (Where Things Live)

To keep the testing story coherent across languages and components, use this rough hierarchy:

- **Rust unit / integration tests**
  - Location: `libs/fl-aggregator/tests/**`, `mycelix-workspace/sdk/src/**` + inline `#[cfg(test)]` modules.
  - Purpose: algorithm-level guarantees (Krum/Median/TrimmedMean invariants, clipping, MATL primitives).
  - When to add: any time you change core aggregation, detection, or MATL math.

- **Python / 0TML tests**
  - Location: `0TML/tests/**`, plus benchmark/experiment scripts under `0TML/benchmarks` and `0TML/experiments`.
  - Purpose: end-to-end FL scenarios, BFT stress tests (30–45%), label-skew/non-IID behavior.
  - These are currently the strongest evidence for 45% BFT and long-run stability.

- **Holochain / hApp tests (Tryorama)**
  - Location: `0TML/holochain/tests/**`, `0TML/mycelix_fl/holochain/tests/**`, `mycelix-workspace/tests/ecosystem/tests/**`.
  - Purpose: cross-agent, cross-hApp behavior: bridge protocol, MATL reputation flows, FL zome externs.
  - `fl-bft.test.ts` (design-phase) encodes Holochain-level BFT acceptance criteria for 33–45% adversary scenarios.

- **SDK tests (TS + Rust)**
  - TS: `mycelix-workspace/sdk-ts/tests/**` (MATL, Epistemic, Bridge, FL, integrations).
  - Rust: `mycelix-workspace/sdk/src/**` + `sdk/tests/**` (MATL / epistemic / bridge invariants).
  - Purpose: correctness and ergonomics of the public APIs used by hApps and external apps.

When adding new tests or interpreting failures, start at the most local layer (unit) and move outward (0TML, Holochain, ecosystem) only when necessary. This keeps debugging tractable and ensures we can always point from a public claim to a specific layer of evidence in `docs/VALIDATED_CLAIMS.md`.
