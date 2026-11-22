# Temporal Joint Oracle (evaluator side)

**Package:** `symthaea-temporal-joint-oracle`  
**Status:** evaluator-only exact reference; Rust tests authored but not compiled or executed in the current authoring environment  
**Tracking:** NEURO-SILICON-009 / #9666

## Security boundary

This package accepts evaluator-only future-stream commitments, realized temporal exposure and measured resource costs. It must not be linked into, mounted inside, or invoked by candidate execution.

The candidate-side `symthaea-temporal-joint-control` package does not depend on this oracle. This evaluator package depends on the candidate package only for the public plan/action wire types; it never calls the candidate selector or trusts the candidate's estimated benefit as truth.

Package separation is not an operating-system security boundary. Confirmatory use still depends on #9437/#9630 process isolation, protected oracle inputs, exact artifact intake and signed receipts.

## Exact enumeration

The evaluator declares a sorted registry of optional binary action factors and requires every subset to be present exactly once. Four factors can represent sensing, memory, compute and verification, producing all 16 joint assignments. The hard limit is 16 factors (65,536 alternatives); larger problems must use preregistered factorial/covering designs and report unresolved interaction uncertainty rather than calling the result exact.

Every alternative must share the same episode fingerprint: task, hidden future stream, model, initial state, candidate feature snapshot, evaluator profile, exposure horizon and resource budget.

The oracle independently checks:

- complete subset coverage with no duplicates or omissions;
- stable factor IDs and action kinds;
- unique run and plan IDs;
- exact paired-context equality;
- realized costs against every hard resource ceiling;
- observed maximum gap against the hard gap constraint;
- the sum of wrong-belief, abstention and missing-output exposure against episode duration.

It reports a separate optimum and candidate regret for each exposure axis and the full non-dominated frontier over realized exposure, measured resource cost and observation gap. It emits no scalarized hidden leaderboard.

## Exact interaction decomposition

The evaluator also exposes `decompose_exact_joint_interactions()`. It uses the admitted complete 2^n binary factorial and computes exact subset Möbius/Harsanyi dividends from measured outcomes:

- wrong-belief, abstention and missing-output benefit dividends;
- twelve measured resource-cost dividends in `ResourceDimension::ALL` order;
- every non-empty subset term and an aggregate by interaction order;
- an exact reconstruction check against the full-joint contrast.

Positive exposure dividends represent synergy; negative exposure dividends represent antagonism. Resource dividends use the opposite interpretation: positive means added cost, negative means savings. The transform does not establish causal effects unless the intervention design identifies them. It is limited to the declared finite binary factors and shares the oracle's 16-factor maximum.

Fifteen Rust test functions are authored in the evaluator package, including hand-calculated pairwise, three-way and four-way controls plus a direct inclusion–exclusion reference implementation checked against the production transform. **None has been executed** in the current authoring environment.

## Local validation

From the repository root:

```sh
cargo test --manifest-path research/temporal-joint-oracle/Cargo.toml
```

The package has a path dependency on the candidate plan contract, but not vice versa. This research workspace does not yet contain a committed Cargo.lock; before confirmatory use, generate and commit the lockfile, then run with --locked. A successful test run is still required before any test is described as passing.
