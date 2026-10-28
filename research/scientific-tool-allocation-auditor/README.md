# Scientific Tool Allocation Auditor v0.1

A standalone Rust CLI for validating and summarizing paired tool-allocation trial receipts in the Symthaea scientific-discovery programme.

This is the first executable evidence/audit seam for [SCI-TOOL-ALLOC-001](https://github.com/Luminous-Dynamics/symthaea/issues/8721). It does **not** run NewtonBench, invoke an LLM, choose tools, or claim a model comparison.

## Run

From the repository root:

```sh
cargo test --locked --manifest-path research/scientific-tool-allocation-auditor/Cargo.toml
cargo run --locked --manifest-path research/scientific-tool-allocation-auditor/Cargo.toml -- MANIFEST.json TRIALS.jsonl
```

For a no-run smoke check using the checked-in empty-trial fixture:

```sh
cargo run --locked --manifest-path research/scientific-tool-allocation-auditor/Cargo.toml -- \
  research/scientific-tool-allocation-auditor/fixtures/manifest.json \
  research/scientific-tool-allocation-auditor/fixtures/trials.jsonl
```

This fixture intentionally contains zero trial rows. The expected report has two assigned cells, two missing cells, zero completed runs per arm, and no completed-conditional recovery rate. It is an input/denominator test, not an experimental result.

The CLI emits a deterministic JSON report with:
- every manifest-assigned task/seed/treatment cell and missing cells;
- terminal outcome counts, retaining infrastructure failures, budget violations, invalid and not-evaluable rows;
- all-assigned and completed-conditional recovery rates with explicit denominators;
- false-confident-singleton and premature-commitment denominators;
- paired treatment-minus-no-tool deltas on complete pairs only, plus incomplete-pair counts;
- SHA-256 commitments to the exact raw manifest bytes, raw JSONL bytes and canonical report body.

## Input contract

The manifest freezes protocol ID, model/revision, prompt digest, task-manifest digest, tool-registry digest, expected treatment arms, task/seed identities, public strata, and a shared resource budget. Each JSONL trial commits to the exact manifest digest and both the policy-output and independent-evaluator-score artifact digests.

Calculate the manifest binding before emitting trial records:

```sh
sha256sum MANIFEST.json
```

Each trial's `manifest_sha256` must equal the SHA-256 of the exact manifest bytes, including any final newline. The CLI rejects duplicate cells, unknown task/seed pairs, undeclared arms, malformed digests, inconsistent terminal/outcome combinations, no-tool invocations, tool calls when tools are unavailable, and a completed always-tool arm that used no tool when a tool was available.

The manifest sets a single per-trial cap for all arms via `resource_budget.max_tool_calls`, `max_experiment_queries`, `max_tokens`, and `max_elapsed_ms`. A trial that exceeds any cap must be marked `budget_violation` and cannot carry scored outcomes. An overrun marked as completed or as an unrelated failure is rejected; a budget-violation disposition without an actual cap breach is also rejected. This prevents methods from being silently compared under different allowed budgets.

Completed rows must provide both a policy-output digest and a separate independent-evaluator-score digest. The auditor checks the evaluator digest's syntax and presence but does not itself execute or authenticate the evaluator. The evaluator-score artifact must be independently generated and must not expose held-out ground truth to the policy.

Missing rows are reported from the manifest's expected denominator; they are not silently dropped. Paired deltas use only complete runs of both arms and report how many expected pairs were incomplete. Failed or over-budget cells remain in the all-assigned denominator and are separately counted.

The manifest JSON fields are `schema`, `protocol_id`, `model_id`, `model_revision`, `prompt_sha256`, `task_manifest_sha256`, `tool_registry_sha256`, `treatments`, `resource_budget`, and `task_seeds`. Each task-seed record contains `task_id`, `domain`, `law_complexity`, `system_complexity`, `noise_level`, `seed`, and `tool_available`. The resource-budget object contains the four positive integer `max_*` caps above.

Each trial JSONL record contains `schema`, `protocol_id`, `manifest_sha256`, `task_id`, `seed`, `treatment`, `terminal_status`, `law_recovered`, `heldout_prediction_nrmse`, `false_confident_singleton`, `premature_commitment`, `tool_calls`, `experiment_queries`, `tokens`, `elapsed_ms`, `policy_output_sha256`, and `evaluator_score_sha256`. Both digests are required for completed trials. A non-completed trial may still include a policy-output digest if an artifact exists, but it may not claim an independent evaluator score. Treatment values are `no_tool`, `always_tool`, `static_threshold`, `heuristic_voc`, and `calibrated_voc`. Terminal values are `completed`, `infrastructure_failure`, `invalid`, `not_evaluable`, and `budget_violation`.

## Evidence boundary

Content hashes bind exact bytes and report content. They do not authenticate the process that produced a trial, prove the scorer's ground truth, or establish scientific validity. A downstream NewtonBench adapter must preserve upstream task IDs, environment/model configuration, tool availability, terminal outcomes and independent evaluator scores; it must also avoid exposing held-out ground truth to the evaluated policy.

No empirical, NewtonBench-reproduction, or SOTA result is claimed.
