# Proof Insight Transfer v0: executable analysis contract

**Status:** implementation contract only. No run configuration, validation baseline receipt, scored result rows, or mathematical transfer result is being asserted by this file.

The preregistered protocol must be executable before held-out outcomes are examined. The Rust binary `analyze_pitb_results` implements its primary paired analysis and rejects common sources of post-hoc advantage: a changed protocol/split, a baseline not selected on validation, known-exposed family 089 entering the eligible denominator, unequal arm budgets, per-target configuration drift, duplicate exact statement digests, incomplete eight-arm comparisons, missing audit receipts, and inconsistent correctness flags.

## Inputs and invocation

Run from the staged `symthaea-proof-insights` crate directory after the source indexer and independent semantic/proof auditors are available:

```bash
cargo run --locked --bin analyze_pitb_results -- --config /path/to/frozen-run-config.json --results /path/to/target-outcomes.jsonl --protocol /path/to/evaluation-protocol.md --splits /path/to/splits.json --baseline-receipt /path/to/baseline-selection-receipt.json --out /path/to/analysis-report.json
```

Inputs are fail-closed. `--protocol` is hashed as raw bytes and must match the digest in the frozen run config. The exact `splits.json` bytes and the split artifact's `split_plan.partition_digest_sha256` must match too. The baseline-selection receipt is likewise content-hashed. Keep all five input artifacts, the output report, and the per-target receipt bundle together.

## Frozen run config

The run config uses `schema_version = "pitb.run-config.v0"` and contains:

- `protocol_id`: `proof-insight-transfer-v0`.
- `protocol_sha256`: SHA-256 of the exact preregistered protocol file.
- `split_manifest_sha256`: SHA-256 of the exact frozen `splits.json` file.
- `partition_digest_sha256`: `e7ac417556cbd47cee1ddd51b98f9684fadfc8cd097587ae3beacc4dfb59f970` for the current frozen partition.
- `baseline_selection_receipt_sha256`: SHA-256 of the validation-only selection receipt.
- `known_exposed_family_ids`: must include `089`; every family in this list is barred from the confirmatory eligible denominator.
- `bootstrap_replicates`: exactly `10000`, per the protocol.
- `bootstrap_seed`: fixed integer seed, frozen before held-out outcomes are inspected.
- `model_version` and `checker_version`: immutable identifiers, not labels such as `latest`.
- `budget`: shared task-level limits for `max_model_calls`, `max_tokens`, `max_wall_ms`, and `max_cost_microusd`.

The budget values are experiment choices, not defaults supplied by this implementation. Choose them before the test run, and encode identical limits for all arms. The checker identifier must distinguish relevant Lean/Mathlib and audit-policy revisions.

The analyzer also verifies that the split document identifies upstream commit `1581d4cbdd69a0faf34b2089614fe2a814316a6a`, has 372 unique families and exact 260/56/56 train/validation/test family counts, and places every known-exposed family in the frozen test partition.

## Validation-only baseline receipt

Use `schema_version = "pitb.baseline-selection.v0"` with:

- `selected_split = "validation"`;
- `metric = "semantic_gate_success_rate"`;
- `arm_scores` containing **exactly** A, B, C, D, and E;
- `selected_arm` equal to the highest score, with exact ties broken by alphabetical arm ID;
- `tie_break_rule = "highest score; alphabetical arm ID on exact tie"`;
- `frozen_before_test = true`;
- `test_outcomes_seen_before_selection = false`.

The receipt is a provenance assertion whose exact bytes are hashed into the frozen config. Keep the underlying validation outcomes and selection code beside it so a reviewer can recompute the scores. The test analyzer verifies the declared score/selection consistency, but cannot infer from a boolean that a hidden test partition was never inspected.

## Target outcome JSONL

Each non-empty line is one JSON object with `schema_version = "pitb.target-outcome.v0"`. Keep excluded and unknown targets as rows with `arms: []`, `eligibility` set to `excluded` or `unknown`, and a specific `exclusion_reason`; they remain in the reported accounting but are not scored. Every row must identify its family in the frozen split and declare `split = "test"`.

An `eligible` target must include:

- unique `target_id` and `family_id`;
- `source_commit`, exactly pinned to the frozen upstream commit;
- exact Lean `theorem_id`, repository-relative `artifact_path`, `artifact_sha256`, `statement_sha256`, `lean_toolchain`, and `mathlib_revision`;
- SHA-256 references to eligibility, statement-correspondence, proof/axiom audit, successor-test, and mutation-control receipts;
- exactly one result for each arm A-H.

Every arm object contains the `arm`, `status`, `checker_passed`, `semantic_gate_passed`, matching model/checker versions and `budget`, actual `model_calls_used`, `tokens_used`, `wall_ms`, and `cost_microusd`, plus SHA-256 references for fixed arm configuration, exact input, retrieval log, checker receipt, and semantic-gate receipt. Preserve the referenced artifacts. Receipt digests in this JSONL bind the audit trail but do not substitute for the receipt contents or an independent reviewer.

Allowed status values are `verified_success`, `proof_only`, `failed`, `timeout`, `not_attempted`, `tool_unavailable`, and `out_of_scope`.

- `verified_success` requires both checker and semantic gates to pass.
- `proof_only` means the checker passed but semantic validation did not; this is not a primary-endpoint success.
- all other statuses require both correctness flags to be false.
- every arm must stay within every declared budget limit.
- each arm's `config_sha256` must stay constant across eligible held-out targets; per-target prompt/retrieval context belongs in its separate `input_sha256` and retrieval-log digest.

The analyzer rejects duplicate eligible statement digests instead of silently treating repeated exact targets as independent wins. It also rejects missing arm rows, changed model/checker versions, changed budgets, invalid paths, malformed receipt digests, duplicate target IDs, and eligible rows from known-exposed family 089.

## Report and decision rule

The JSON report records hashes of the run config and raw JSONL, input protocol/split/selection-receipt digests, upstream commit, denominator accounting, per-arm raw checker and semantic-gate success rates, total/mean use and cost-per-verified-solve, the selected primary comparison, and the explicit external-replication status `not_evaluated`.

The primary effect is the signed percentage-point difference:

```text
semantic-gate success rate (F: typed insight)
  minus
semantic-gate success rate (validation-selected baseline in A-E)
```

The 95% percentile interval uses 10,000 deterministic paired bootstrap replicates, resampling **families** with replacement and retaining all target pairs within each sampled family. The preregistered utility gate passes only when the point estimate is at least +10 percentage points, the lower confidence bound is above zero, and at least two family clusters are present. A missing gate is never fixed by post-hoc baseline substitution or a relaxed interval.

A passing result supports only a within-benchmark utility claim on the supplied eligible targets. It does not establish external validity, absence of pretraining contamination, or superiority over the entire field. A separate exposure-resistant external challenge set and independent replication remain required.

## Important limitation

This binary validates data structure, protocol/split hashes, the validation-selected baseline receipt, matched budgets/configs, evidence-digest presence, status consistency, and the preregistered statistical calculation. It **does not run Lean**, inspect the contents of checker/semantic receipts, determine whether a formal theorem matches the intended informal claim, or prove mathematical correctness. Those are separate upstream gates. The test rows must not set `semantic_gate_passed = true` until those gates have actually passed and their receipts have been retained.

Synthetic Rust unit tests exercise deterministic sampling, confidence-interval quantiles, validation tie-breaking, SHA-256 shape, and repository-path containment. These test fixtures validate the analysis implementation only; they are not theorem-proving benchmark results.
