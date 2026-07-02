# Shared Test Fixtures (Design Skeleton)

**Purpose**: Provide a common place to store cross-language test fixtures (Python 0TML, Rust, TypeScript) so that core algorithms produce the *same* results across runtimes.

This is especially important for:
- PoGQ / MATL composite scores
- Byzantine classification thresholds (Honest vs Byzantine)
- Label-skew / non-IID scenarios

---

## Directory Structure (Proposed)

This folder is intentionally light right now. A future structure could look like:

```
Mycelix-Core/tests/shared-fixtures/
├── pogq/
│   ├── simple_cases.json      # Small synthetic gradients + expected scores
│   └── edge_cases.json        # Extremes, NaN/Inf, boundary conditions
├── matl/
│   └── composite_scores.json  # PoGQ + reputation → composite score cases
└── label_skew/
    └── scenarios.json         # Label-skew regimes with expected FP/FN ranges
```

Each JSON file would capture:
- A small set of input gradients / scores / parameters.
- Expected outputs (scores, classifications, thresholds) computed *once* in Python/0TML.

Rust (`libs/fl-aggregator`, `mycelix-workspace/sdk`) and TS (`sdk-ts`) tests can then:
- Parse these fixtures.
- Assert they reproduce the same results using their implementations.

---

## Current Status

- This directory currently serves as a placeholder and design note.
- No fixtures are used in CI yet; introducing them should be done deliberately and tied to:
  - Updates to PoGQ/MATL implementations, and
  - Entries in `docs/VALIDATED_CLAIMS.md` that reference cross-runtime consistency.

When you are ready to lock cross-language behavior, start by:
1. Creating a small `pogq/simple_cases.json` derived from 0TML output.
2. Adding a Rust test in `fl-aggregator` or `mycelix-workspace/sdk` that reads it.
3. Mirroring the same expectations in `sdk-ts/tests`.

