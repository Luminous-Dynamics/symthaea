# PITB exact support-antichain analyzer (Rust)

This standalone Rust crate computes the **finite minimal-sufficient-support antichain** from exhaustive, exact-target premise-ablation outcomes. It is not a theorem-level logical-independence prover, and it does not infer necessity from one observed proof term.

## The distinction

For one frozen target, fixed context, Lean environment/toolchain, solver policy and resource budget:

- **Observed proof-term support**: declarations referenced by one checked proof construction. This is evidence about that proof, not necessity.
- **Sufficient route**: a set of available declarations under which the exact target is solved and both the kernel and independent semantic gate pass.
- **Minimal sufficient route**: a sufficient set for which every proper subset is a non-solve under the same frozen experiment.
- **Antichain**: all inclusion-minimal sufficient routes. Multiple alternatives must remain separate. For example, `[{h}, {hEq}]` has an empty intersection: neither declaration is common to every minimal route.
- **Necessary across all observed minimal routes**: the intersection of the finite antichain. This remains policy-, budget-, environment- and context-specific; it is not a general logical-necessity theorem.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-support-antichain/Cargo.toml
cargo test --locked --manifest-path tools/pitb-support-antichain/Cargo.toml
cargo run --locked --manifest-path tools/pitb-support-antichain/Cargo.toml -- --self-test
cargo run --locked --manifest-path tools/pitb-support-antichain/Cargo.toml -- --self-test-outcomes /tmp/support-outcomes.synthetic.jsonl
cargo run --locked --manifest-path tools/pitb-support-antichain/Cargo.toml -- \
  --outcomes path/to/outcomes.jsonl \
  --predictions path/to/predictions.jsonl \
  --report /tmp/support-antichain-report.json \
  --max-variables 12
```

Use `--predictions` only when every analyzed group has exactly one prediction from the same model ID/version. Abstentions count against prediction coverage; a prediction cannot claim a route or probability while abstaining.

## Outcome contract

Every JSONL outcome row binds:
- study, component and target IDs;
- exact target statement digest;
- environment, toolchain, fixed-context, budget and solver-policy digests;
- the fixed mutable declaration universe and one available subset;
- ordered local-context digest;
- terminal outcome and both proof gates;
- state, run, checker and semantic-gate receipt digests.

The analyzer requires exactly one row for every subset in the registered powerset `2^D`. The full-context baseline must pass. Rows cannot mix identities or budgets, repeat local-context digests across distinct subsets, or classify an unresolved result as a non-solve. A `verified_success` must have both `kernel_passed=true` and `semantic_gate_passed=true`, `attempt_finished=true`, and `terminal_reason=verified_success`.

A `non_solve` is admitted only when the attempt finished and the terminal reason is `budget_exhausted` or `search_exhausted`. Interrupted attempts, infrastructure failures, invalid targets, checker failures, unresolved results, and kernel-valid proofs rejected by the semantic gate are **not** evidence that a premise subset is insufficient; they invalidate that ablation row and must be excluded/rerun. Each row carries a digest for the terminal-reason evidence. This prevents runner crashes, task corruption, or semantically defective proof candidates from manufacturing false members of the support antichain.

The default exhaustive cap is 12 mutable declarations (4,096 subsets); a hard maximum of 16 (65,536 subsets) is enforced. Larger contexts require a separate approximate/design-based protocol and must not be labeled exhaustive.

## Prediction metrics

When a prediction ledger is provided, it scores:
- necessary-declaration precision/recall/F1 against the intersection of observed minimal routes;
- proposed-route sufficiency (the proposal contains at least one complete minimal route);
- exact minimal-route recovery;
- prediction coverage and selective route risk;
- Brier score for supplied route-sufficiency probabilities.

Each group appears once in the prediction ledger. Different model IDs or versions must be scored separately. The prediction's run receipt is digest-bound in the row, but this tool does not open/authenticate that receipt's bytes.

## Trust boundary

This analyzer validates row structure, identity consistency, exact subset coverage and digest syntax. It does **not** authenticate referenced receipt bytes, independently rerun Lean, validate the target-to-source correspondence, prove dependency-inventory completeness, or establish the semantic-gate implementation. Those receipts must be independently opened and verified by the surrounding benchmark workflow before results are admitted.

A success/non-solve lattice is an experimental observation under the frozen prover, checker, context and budget. Non-monotonic observations are reported; they are not silently normalized away. If a solver behaves inconsistently across supersets, report that rather than inventing a monotone theorem.

## Synthetic controls

`--self-test` covers:
- alternative sufficient routes and their empty intersection;
- conjunctive routes and common necessary declarations;
- incomplete powerset rejection;
- mixed budget identity rejection;
- duplicate local-context digest rejection;
- success without both proof gates rejection;
- interrupted and infrastructure-failed and kernel-valid/semantic-rejected runs cannot count as non-solves;
- a non-solve must have a completed budget/search exhaustion reason;
- exhaustive cap enforcement;
- prediction coverage, sufficiency, exact route and Brier score.

The synthetic controls are contract tests only. They are not measured Lean tasks or benchmark performance.

## Research relationship

- DAPS and LeanGraphSearch improve candidate retrieval and dependency exploration.
- TheoremGraph gives a large formal/informal knowledge graph but keeps candidate informal edges provenance-labeled.
- CoCo-Prover uses intra-declaration AND/OR proof graphs plus inter-declaration dependency DAGs for cost-aware scheduling.
- PITB adds counterfactual sufficient-route measurement: whether a bundle succeeds, whether smaller subsets fail, and which declarations are common across all minimal routes.
- The DAPS importer in PITB keeps external graph edges candidate-only until an independent Lean extraction reproduces their exact semantic roles.

Next: run matched DAPS / LeanGraphSearch / BM25 / no-retrieval arms on the same frozen context lattice; then measure whether typed bundle retrieval increases exact-target verified completion and cost-efficiency without degrading abstention calibration.
