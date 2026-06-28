# Alpha.6 Upgrade Notes

Alpha.6 is a research-hardening release.

It adds three practical capabilities:

1. Experiment matrices for repeated dimension-by-noise comparisons.
2. Paired comparison/sign-test helpers for lightweight statistical sanity checks.
3. Local research artifact receipts shaped for later Mycelix source-chain integration.

The crate still does not claim quantum consciousness, quantum advantage, or hardware execution.

## Suggested next step

Run `./scripts/verify-local.sh`, then run:

- `cargo run --example experiment_matrix`
- `cargo run --example significance_probe`
- `cargo run --example research_receipt`

Save the output in a lab-notes directory and attach a claim boundary before interpreting results.
