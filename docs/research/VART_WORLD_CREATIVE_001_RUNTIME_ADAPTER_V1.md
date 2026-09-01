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

## 2. Policy-output isolation

Each pilot policy executes in a fresh private staging directory owned only by that invocation. The shared pilot evidence root is **not** passed to the runtime process.

The runtime receives only its per-cell `VART_OUTPUT_ROOT`; after the process exits successfully and its trial package is validated, the orchestrator moves the sealed `trials/<trial-id>/` directory into the shared evidence package.

This makes cross-policy output inspection structurally unavailable through the experiment output surface rather than relying only on a self-reported flag.

A runtime invocation must write exactly one subtree:

`$VART_OUTPUT_ROOT/trials/$VART_TRIAL_ID/`

Writing any sibling/top-level artifact causes pilot rejection.

## 3. Normative environment inputs

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
- `VART_OUTPUT_ROOT` — private per-cell staging root
- `VART_ANALYSIS_CONTRACT_SHA256`
- `VART_METRIC_DEFINITION_SET_SHA256`

The runtime entrypoint may additionally accept equivalent CLI arguments, but it must fail if CLI and environment values disagree.

## 4. Runtime command

`VART_WORLD_CREATIVE_001_PILOT_RUN.template.json` deliberately contains an unresolved `runtime_argv`. Bind it to the actual local VART trial entrypoint only after inspecting the qualified runtime tree.

The template exposes `{output_root}`, not the shared pilot root. The command runs once per pilot cell and must be deterministic with respect to the frozen inputs and declared runtime state, except for explicitly measured simulator/renderer behavior whose environment identity is retained in evidence.

## 5. Required decision-input artifact

Every trial exports a canonical logical artifact `decision_input` and binds its raw-byte SHA-256 as `manifest.decision_input_sha256`.

It contains at minimum:

- `experiment_id`
- `paired_block_id`
- `seed`
- `revision_index`
- pre-decision world/observation identities required by the policy

For FULL / RANDOM_VALID / HEURISTIC paired comparisons, the decision-input artifact is byte-identical wherever the policy definitions specify the same evidence surface.

It must not contain another policy's choice/outcome, later revisit/outcome data, human labels, or unrevealed generalization-fixture targets.

The pilot verifier independently checks paired `decision_input_sha256` equality in addition to candidate-set equality.

## 6. Candidate-set retention

Every manifest binds both:

- `generated_candidate_count`
- `admissible_candidate_count`

The canonical candidate-set artifact retains **all generated candidates**, including physically rejected candidates, in frozen order. Every entry explicitly records `physically_admitted: true|false`.

The candidate array length equals `generated_candidate_count`; the number of admitted entries equals `admissible_candidate_count`.

This makes rejected-candidate truncation independently detectable.

## 7. Required evidence package

At minimum each trial emits:

- `manifest.json`
- `evidence_index.json`
- `decision_input`
- experience evidence or explicit ablation sentinel
- `RevisionHypothesis`
- canonical generated/admitted candidate-set artifact
- selected proposal
- typed application receipt for completed trials
- revisit observation for completed trials
- `RevisionOutcome` for completed trials
- RANDOM_VALID draw receipt when applicable
- ablation receipt when applicable

Every manifest binds the exact pilot analysis-contract and metric-definition digests supplied by the orchestrator.

For a completed trial, the typed applied receipt additionally binds:

- `decision_input_sha256`
- `revision_hypothesis_sha256`
- `candidate_set_sha256`
- `selected_proposal_sha256`
- `world_version_before`
- `world_version_after`

This prevents replacement of the hypothesis, candidate surface, or decision input after the edit without breaking receipt closure.

## 8. Ablation receipts

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

The sentinel's raw-byte digest occupies `manifest.experience_episode_sha256`. This preserves fixed manifest closure while making the absence intentional, typed, and independently verifiable.

### `no_counterfactual_evaluation`

- `removed_channels` contains `counterfactual_evaluation`.
- `assertions.counterfactual_evaluation_performed = false`.
- Candidate-set evidence contains no counterfactual observation/render/score fields.
- `evidence_index.json` contains no logical file whose name begins `counterfactual_`.

## 9. RANDOM_VALID

RANDOM_VALID implements `sha256-counter-v1` exactly as frozen in `VART_WORLD_CREATIVE_001_RANDOM_VALID_V1.md` and passes the frozen cross-language vectors before pilot execution.

It must never use OS entropy, Rust `rand`, Python RNG state, candidate outcome values, FULL/HEURISTIC choices, or later observations.

## 10. Exit semantics

Exit status distinguishes **infrastructure failure** from a **scientifically valid non-successful trial**.

- Exit `0`: the trial reached a scientifically accounted terminal state and emitted a complete evidence package.
- Non-zero exit: the adapter/runtime failed to produce trustworthy trial evidence. The orchestrator stops immediately and the pilot does not pass.

A poor FULL outcome is not an infrastructure failure. A broken evidence chain is not an ordinary poor outcome.

## 11. Write ordering

For a completed trial the evidence establishes:

`decision input closed -> RevisionHypothesis closed -> selection closed -> typed mutation receipt -> revisit closed -> RevisionOutcome closed`

The manifest and `evidence_index.json` are written last, after referenced artifacts are durable.

## 12. Pilot vs confirmatory verifier

The pilot runner invokes `scripts/verify_vart_world_creative_001_pilot.py`.

A confirmatory claim MUST NOT use that wrapper. Confirmatory closeout uses `scripts/verify_vart_world_creative_001_qualified.py` with an **externally anchored SHA-256 of `confirmatory_freeze.json`** committed before confirmatory outcomes are observed, and it must pass the N1–N20 verifier qualification suite.

## 13. Pilot boundary

Every pilot manifest contains:

- `campaign = pilot`
- `included_in_confirmatory_analysis = false`

No pilot tool may set confirmatory or claim authorization true.

A passing pilot means only that the measurement and verification machinery works. It cannot establish efficacy, superiority, calibration improvement, transfer, general creativity, consciousness, or physical-world competence.
