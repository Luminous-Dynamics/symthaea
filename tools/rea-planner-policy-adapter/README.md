# REA planner policy adapter v0

This standalone Rust CLI calls the existing `symthaea-rea-investigator` APIs directly over a frozen six-case synthetic manifest.

It invokes:
- point EIG: `plan_next_experiment`;
- robust EIG: `plan_next_experiment_robust`;
- decision-aware EVSI/minimax: `plan_next_experiment_decision_aware`.

It adds transparent baselines:
- deterministic case-ID seeded eligible choice (a repeatable random-style baseline, not a statistical estimate);
- cost-only over the same hard eligibility gates;
- greedy expected mandatory-obligation closure per unit cost;
- an obligation-gated decision-aware planner that filters to candidates with a possible admissible transition to satisfy at least one currently unmet mandatory obligation.

The adapter does not implement the planner algorithms. It never reads the grader-only DGP file, does not execute candidate experiments, and does not mutate the target. It emits explicit claim ceilings.

## Run

```bash
cargo test --manifest-path tools/rea-planner-policy-adapter/Cargo.toml --locked
cargo run --manifest-path tools/rea-planner-policy-adapter/Cargo.toml --bin rea-planner-policy-adapter --locked
```

The manifest and source case projection are public, synthetic development data. The Evidence origin tags are caller-supplied fixtures, not authenticated observation provenance. Likelihoods and utilities are not empirical or human-calibrated. Results qualify API composition and deterministic harness semantics only; they are not capability or SOTA results.

The adapter branch is stacked on the exact planner implementation branch ending at `2cf30ebc544a336d369246613895555a85696fe5`. Its input snapshot is copied from PR #11151. After the planner and manifest branches are reviewed/merged, retarget and remove the frozen duplicates under the agreed ownership model.

## Output

The JSON records each case and each policy's selected candidate, abstention, ranked candidate metrics, rejections and claim ceilings. Obligation-gated selection is an admissibility filter only: it cannot claim readiness based on predicted outcomes.
