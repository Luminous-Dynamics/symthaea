# Symthaea Temporal Evaluator

This is a separate, evaluator-side Rust package for paired retain/evict effects and exhaustive small-subset upper bounds. It is not a candidate runtime dependency.

The current source provides:

- paired intervention checks that require identical episode/task/future-stream/model/initial-state/candidate-feature digests and identical byte/compute/refresh/observation budgets;
- componentwise wrong-belief, abstention and missing-output exposure differences;
- group interventions to expose non-additive memory interactions;
- exact subset enumeration up to 16 candidate items;
- separate best-known subsets for each exposure component and a Pareto frontier, with no weighted scalar objective;
- hard checks for duplicate IDs, unordered selections, incorrect bytes, budget overruns, incomplete subset enumeration and exposure totals exceeding episode duration.

The exact subset result is an evaluator-only clairvoyant upper bound, not a deployable candidate policy. It must never be exposed to the candidate or used to train against confirmatory outcomes.

Package separation is an architectural boundary only. It does not provide OS-level confidentiality. Before confirmatory use, complete #9437: separate OS identities and protected oracle storage, candidate sandboxing, evaluator-owned signing keys, signed receipts, replay protection and adversarial filesystem/descriptor tests.

No Cargo build, format, Clippy or test execution is claimed until a Rust toolchain is available.