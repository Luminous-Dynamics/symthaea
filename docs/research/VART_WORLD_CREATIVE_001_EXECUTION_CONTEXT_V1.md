# VART-WORLD-CREATIVE-001 — Prospective Execution Context v1

Status: confirmatory qualification contract. It does not authorize execution by itself.

## Purpose

Raw evidence can be internally hash-consistent while still having been produced by the wrong policy implementation, fixture, source tree, environment, candidate generator, or physical-admission policy. VART therefore treats the complete execution context as a preregistered input, not a retrospective runtime label.

For every confirmatory trial, the campaign planner creates `execution_context.json` **before any trial outcome exists**. Its exact raw-byte SHA-256 is stored in the frozen `trial_inventory.json` entry for that `trial_id`. The externally anchored `confirmatory_freeze.json` binds the raw-byte SHA-256 of that inventory.

A runtime-produced manifest may only reference the prospectively frozen context digest. It may not choose or rewrite the context after execution.

## Canonical execution-context object

The v1 object contains exactly the scientific inputs needed to identify the mechanism being tested:

- `schema = "symthaea.vart-world-creative-001.execution-context.v1"`
- `experiment_id`
- `campaign`
- `trial_id`
- `paired_block_id`
- `policy`
- `policy_sha256`
- `world_fixture_sha256`
- `seed`
- `revision_index`
- `source_head`
- `source_tree`
- `environment_digest`
- `candidate_generator_sha256`
- `physical_admission_policy_sha256`
- `metric_definition_set_sha256`
- `analysis_contract_sha256`
- `trial_manifest_schema_sha256`

Canonical bytes are UTF-8 JSON with keys sorted, separators `(',', ':')`, no insignificant whitespace, and a final newline omitted. Any change to bytes changes the digest.

## Frozen trial inventory

Each confirmatory inventory contains both the ordinary ordered `trial_ids` and a `trial_contexts` map:

`trial_contexts[trial_id] = execution_context_sha256`

The map must contain exactly the preregistered trial IDs. No trial may share an execution-context digest with a different trial identity unless the canonical context bytes are themselves identical, which ordinary VART trial IDs prevent because `trial_id` is part of the object.

## Freeze bindings

The externally anchored confirmatory freeze supplies:

- `source.head`
- `source.tree`
- `environment_digest`
- `candidate_generator_sha256`
- `physical_admission_policy_sha256`
- `policy_digests`
- `ablation_policy_digests`
- `metric_definition_set_sha256`
- `analysis_contract_sha256`
- `trial_manifest_schema_sha256`
- `trial_inventory_sha256`

The qualified verifier reconstructs every execution context and requires equality against both the trial manifest and those frozen campaign-level values.

## Evidence-package binding

Each trial manifest contains `execution_context_sha256`.

Each trial `evidence_index.json` maps logical name `execution_context` to the exact context artifact. The artifact digest must equal:

1. `manifest.execution_context_sha256`; and
2. `trial_inventory.trial_contexts[trial_id]`.

For complete trials, the typed application receipt should repeat `execution_context_sha256`, so the committed mutation is tied to the same preregistered mechanism context.

## Policy binding

For ordinary policies, `execution_context.policy_sha256` must equal `confirmatory_freeze.policy_digests[policy]`.

For ablations, it must equal `confirmatory_freeze.ablation_policy_digests[policy]`.

A manifest cannot substitute another digest while retaining the same policy label.

## Fixture binding

`execution_context.world_fixture_sha256` must equal the manifest fixture digest and must be a member of the prospectively frozen fixture inventory/set identified by the confirmatory freeze.

Generalization fixtures remain subject to their separate reveal/leakage rules; membership proof does not authorize early disclosure to the policy.

## Source and environment binding

`source_head`, `source_tree`, and `environment_digest` are fixed campaign inputs. A source or environment change after freeze creates a new confirmatory lineage and a fresh evidence root.

The execution context is not a substitute for independently making the frozen source tree fetchable and reproducible. It is the cryptographic link between that qualified source/environment and each individual trial.

## Candidate-generation and admission binding

The context prospectively freezes the candidate-generator and physical-admission-policy digests. This prevents a producer from changing either mechanism after seeing results while leaving the policy label untouched.

A future independent simulator replay may provide stronger semantic validation of the resulting candidate/admission evidence; v1 establishes mechanism identity and prospective closure.

## Required rejection classes

The execution-context qualification layer adds deterministic rejection classes:

- `EXECUTION_CONTEXT_DIGEST_MISMATCH`
- `EXECUTION_CONTEXT_INVENTORY_MISMATCH`
- `FROZEN_POLICY_IMPLEMENTATION_MISMATCH`
- `FROZEN_FIXTURE_MISMATCH`
- `FROZEN_SOURCE_MISMATCH`
- `FROZEN_ENVIRONMENT_MISMATCH`
- `FROZEN_CANDIDATE_GENERATOR_MISMATCH`
- `FROZEN_ADMISSION_POLICY_MISMATCH`
- `FROZEN_SCHEMA_MISMATCH`

These are integrity/qualification failures, not poor scientific outcomes.
