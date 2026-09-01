# VART-WORLD-CREATIVE-001 — Pilot Runtime Adapter v1

Status: **pilot-only adapter contract**. This document does not authorize confirmatory execution or scientific claims.

The pilot orchestrator intentionally does not guess the local Rust binary or CLI surface. The qualified runtime tree remains the authority for how a trial is executed; this contract defines only the inputs it must accept and the evidence it must emit.

## 1. Source binding

The adapter may execute only from the clean qualified VART runtime:

- HEAD `844d10609a9f03e26a06f22778db4b8cdfb6a3ef`
- TREE `38e5506c8f7f88d58e1ff03a77585091d9263a98`
- qualified v0.5-A parent HEAD `33820b3d9e904280e6264719fe7717cb2e5dd5bb`
- qualified v0.5-A parent TREE `e93c6dbfa05b602100ff924efaa5d95f92ef5a65`

The orchestrator independently checks HEAD, TREE, and a clean working tree before any pilot evidence root is created.

## 2. Normative environment inputs

For every pilot cell the orchestrator exports:

- `VART_EXPERIMENT_ID=VART-WORLD-CREATIVE-001`
- `VART_CAMPAIGN=pilot`
- `VART_NONCANONICAL=1`
- `VART_CONFIRMATORY_EXECUTION_AUTHORIZED=0`
- `VART_CLAIM_AUTHORIZED=0`
- `VART_CELL_ID`
- `VART_TRIAL_ID`
- `VART_POLICY`
- `VART_FIXTURE`
- `VART_SEED`
- `VART_REVISION_INDEX`
- `VART_PAIRED_BLOCK_ID`
- `VART_OUTPUT_ROOT`
- `VART_ANALYSIS_CONTRACT_SHA256`
- `VART_METRIC_DEFINITION_SET_SHA256`

The runtime entrypoint may additionally accept equivalent CLI arguments, but it must fail if CLI and environment values disagree.

## 3. Runtime command

`VART_WORLD_CREATIVE_001_PILOT_RUN.template.json` deliberately contains an unresolved `runtime_argv`. Bind it to the actual local VART trial entrypoint only after inspecting the qualified runtime tree.

The command runs once per pilot cell and must be deterministic with respect to the frozen inputs and declared runtime state, except for explicitly measured simulator/renderer behavior whose environment identity is retained in evidence.

## 4. Files owned by the orchestrator

The runtime MUST NOT modify these top-level pilot-root files:

- `analysis_contract.json`
- `metric_definitions.json`
- `trial_inventory.json`
- `primary_results.json`
- `confirmatory_freeze.json`
- `_orchestrator/**`

The orchestrator creates and hashes them before trial execution.

## 5. Files owned by one trial

A runtime invocation may write only below:

`$VART_OUTPUT_ROOT/trials/$VART_TRIAL_ID/`

At minimum it must emit the Evidence Package v1 artifacts required by `scripts/verify_vart_world_creative_001.py`, including:

- `manifest.json`
- `evidence_index.json`
- experience evidence
- `RevisionHypothesis`
- canonical physically admitted candidate-set artifact
- selected proposal
- typed application receipt for completed trials
- revisit observation for completed trials
- `RevisionOutcome` for completed trials
- RANDOM_VALID draw receipt when applicable
- ablation receipt when applicable

Every manifest must bind the exact pilot analysis-contract and metric-definition digests supplied by the orchestrator.

## 6. Ablation receipts

Every ablation policy emits an `ablation_receipt` logical artifact whose SHA-256 is bound by `manifest.ablation_receipt_sha256`.

Receipt schema:

`"symthaea.vart-world-creative-001.ablation-receipt.v1"`

Required identity fields:

- `experiment_id`
- `trial_id`
- `policy`
- `removed_channels`
- `preregistered_ablation = true`
- `assertions`

Required pilot semantics:

### `no_embodied_experience`

- `removed_channels` contains `embodied_experience`.
- `assertions.experience_episode_available = false`.
- The logical `experience_episode` evidence file is an explicit sentinel, not missing data.

Sentinel schema:

`"symthaea.vart-world-creative-001.ablation-sentinel.v1"`

It binds:

- `experiment_id`
- `trial_id`
- `policy = no_embodied_experience`
- `channel = ExperienceEpisode`
- `available = false`

The sentinel's raw-byte digest occupies `manifest.experience_episode_sha256`. This preserves the fixed trial-manifest closure while making the absence intentional, typed, and independently verifiable.

### `no_counterfactual_evaluation`

- `removed_channels` contains `counterfactual_evaluation`.
- `assertions.counterfactual_evaluation_performed = false`.
- The candidate-set evidence must contain no counterfactual observation/render/score fields.
- `evidence_index.json` must contain no logical file whose name begins `counterfactual_`.

## 7. RANDOM_VALID

RANDOM_VALID must implement `sha256-counter-v1` exactly as frozen in `VART_WORLD_CREATIVE_001_RANDOM_VALID_V1.md` and pass the frozen cross-language vectors before pilot execution.

It must never use OS entropy, Rust `rand`, Python RNG state, candidate outcome values, FULL/HEURISTIC choices, or later observations.

## 8. Exit semantics

Exit status distinguishes **infrastructure failure** from a **scientifically valid non-successful trial**.

- Exit `0`: the trial reached a scientifically accounted terminal state and emitted a complete evidence package. `trial_state` may be `complete`, `aborted`, or another preregistered scientifically accounted state as permitted by the manifest contract.
- Non-zero exit: the adapter/runtime itself failed to produce trustworthy trial evidence. The orchestrator stops immediately and the pilot does not pass.

A poor FULL outcome is not an infrastructure failure. A broken evidence chain is not an ordinary poor outcome.

## 9. Write ordering

For a completed trial the evidence must establish:

`RevisionHypothesis closed -> selection closed -> typed mutation receipt -> revisit closed -> RevisionOutcome closed`

The manifest and `evidence_index.json` are written last, after the referenced artifacts are durable. The evidence index binds exact raw-byte SHA-256 values.

## 10. Pilot boundary

Every pilot manifest must contain:

- `campaign = pilot`
- `included_in_confirmatory_analysis = false`

No pilot tool may set confirmatory or claim authorization true.

A passing pilot means only that the measurement and verification machinery works. It cannot establish efficacy, superiority, calibration improvement, transfer, general creativity, consciousness, or physical-world competence.
