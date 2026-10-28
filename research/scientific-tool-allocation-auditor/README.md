# Scientific Tool Allocation Auditor v0.1

A standalone Rust CLI for validating and summarizing paired tool-allocation trial receipts in the Symthaea scientific-discovery programme.

This is the first executable evidence/audit seam for [SCI-TOOL-ALLOC-001](https://github.com/Luminous-Dynamics/symthaea/issues/8721). It does **not** run NewtonBench, invoke an LLM, choose tools, or claim a model comparison.

## Run

From the repository root:

```sh
cargo test --locked --manifest-path research/scientific-tool-allocation-auditor/Cargo.toml
cargo run --locked --manifest-path research/scientific-tool-allocation-auditor/Cargo.toml -- MANIFEST.json TRIALS.jsonl
```

The CLI emits a deterministic JSON report with:
- every manifest-assigned task/seed/treatment cell and missing cells;
- terminal outcome counts, retaining infrastructure failures, invalid and not-evaluable rows;
- all-assigned and completed-conditional recovery rates with explicit denominators;
- false-confident-singleton and premature-commitment denominators;
- paired treatment-minus-no-tool deltas on complete pairs only, plus incomplete-pair counts;
- SHA-256 commitments to the exact raw manifest bytes, raw JSONL bytes and canonical report body.

## Input contract

The manifest freezes protocol ID, model/revision, prompt digest, task-manifest digest, tool-registry digest, expected treatment arms, task/seed identities and public strata. Each JSONL trial commits to the exact manifest digest and its policy output digest.

Calculate the manifest binding before emitting trial records:

```sh
sha256sum MANIFEST.json
```

Each trial's `manifest_sha256` must equal the SHA-256 of the exact manifest bytes, including any final newline. The CLI rejects duplicate cells, unknown task/seed pairs, undeclared arms, malformed digests, inconsistent terminal/outcome combinations, no-tool invocations, tool calls when tools are unavailable, and a completed always-tool arm that used no tool when a tool was available.

Missing rows are reported from the manifest's expected denominator; they are not silently dropped. Paired deltas use only complete runs of both arms and report how many expected pairs were incomplete.

The manifest also fixes one shared per-trial resource cap across every arm: `max_tool_calls`, `max_experiment_queries`, `max_tokens`, and `max_elapsed_ms`. Every consumed resource must stay within these caps for a trial to be counted as completed. An over-cap trial must be marked `budget_violation` and may not carry scored outcomes; the report retains its resource counts and includes it in the all-assigned denominator. The auditor rejects both an unlabelled overrun and a budget-violation label without an evidenced cap breach.

The manifest JSON fields are:
`schema`, `protocol_id`, `model_id`, `model_revision`, `prompt_sha256`, `task_manifest_sha256`, `tool_registry_sha256`, `treatments`, and `task_seeds`. Each task-seed record contains `task_id`, `domain`, `law_complexity`, `system_complexity`, `noise_level`, `seed`, and `tool_available`.

Each trial JSONL record contains `schema`, `protocol_id`, `manifest_sha256`, `task_id`, `seed`, `treatment`, `terminal_status`, `law_recovered`, `heldout_prediction_nrmse`, `false_confident_singleton`, `premature_commitment`, `tool_calls`, `experiment_queries`, `tokens`, `elapsed_ms`, and optional `policy_output_sha256`. Treatment values are `no_tool`, `always_tool`, `static_threshold`, `heuristic_voc`, and `calibrated_voc`. Terminal values are `completed`, `infrastructure_failure`, `invalid`, and `not_evaluable`.

Completed rows require all four scored outcomes and a valid policy-output digest. Non-completed rows must not carry scored outcomes. A no-tool row must have zero tool calls, and an always-tool row must record a call when a tool was available and execution completed.

## Evidence boundary

Content hashes bind exact bytes and report content. They do not authenticate the process that produced the trial, prove the scorer's ground truth, or establish scientific validity. A downstream NewtonBench adapter must preserve upstream task IDs, environment/model configuration, tool availability, terminal outcomes and independent evaluator scores; it must also avoid exposing held-out ground truth to the evaluated policy.

No empirical, NewtonBench-reproduction, or SOTA result is claimed.
