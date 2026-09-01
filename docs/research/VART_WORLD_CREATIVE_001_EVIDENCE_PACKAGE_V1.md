# VART-WORLD-CREATIVE-001 — Evidence Package v1

Status: evidence-format contract. This document defines the byte-level producer/verifier boundary for pilot and confirmatory VART evidence. It does not authorize execution or claims.

## 1. Hashing rule

Unless a field explicitly says otherwise, every `*_sha256` value is the lowercase SHA-256 of the **exact raw bytes** of the referenced durable artifact.

Do not reconstruct hashes from parsed JSON objects. Do not normalize whitespace after export. The verifier hashes the bytes on disk.

## 2. Evidence-root structure

A complete evidence root contains:

- `analysis_contract.json`
- `metric_definitions.json`
- `trial_inventory.json`
- `primary_results.json`
- `confirmatory_freeze.json`
- `trials/<trial_id>/...`

Pilot roots may additionally contain `_orchestrator/**`. Confirmatory roots must be fresh and must not contain pilot trials.

For confirmatory closeout, the raw-byte SHA-256 of `confirmatory_freeze.json` is recorded **outside the evidence root before confirmatory execution** and supplied to `scripts/verify_vart_world_creative_001_qualified.py --expected-freeze-sha256 ...`.

Self-consistency inside the evidence package is not sufficient to prove a contract was frozen prospectively.

## 3. Trial manifest

Each trial directory contains `manifest.json` conforming to `VART_WORLD_CREATIVE_001_TRIAL_MANIFEST.schema.json`.

Important bindings include:

- experiment/campaign/trial identity
- paired block identity
- policy and policy digest
- world fixture, seed, revision index
- pre-revision world version
- `decision_input_sha256`
- `experience_episode_sha256`
- `revision_hypothesis_sha256`
- `candidate_set_sha256`
- `generated_candidate_count`
- `admissible_candidate_count`
- selected proposal/index
- RANDOM_VALID receipt when applicable
- ablation receipt when applicable
- applied receipt/world version after
- revisit/outcome
- confirmatory inclusion state
- integrity state
- metric-definition and analysis-contract digests
- evidence-index digest

`generated_candidate_count` and `admissible_candidate_count` are distinct. Rejected candidates remain evidence.

## 4. Decision input

Every trial exports a logical `decision_input` artifact through `evidence_index.json`; its raw-byte digest equals `manifest.decision_input_sha256`.

The decision input is the exact policy-visible evidence surface at decision time. For paired FULL/RANDOM_VALID/HEURISTIC comparisons it must be byte-identical wherever the preregistered policy definitions prescribe the same inputs.

It must not contain:

- another policy's selected action or outcome;
- later revisit/outcome evidence;
- human evaluation labels unavailable at decision time;
- unrevealed generalization target/solution/trap annotations.

## 5. Prospective hypothesis

The `RevisionHypothesis` artifact is created and closed before mutation. The typed application receipt for a completed trial binds its exact `revision_hypothesis_sha256` in addition to the manifest.

This dual binding is necessary: changing both the hypothesis file and manifest after the fact must still conflict with the earlier application receipt.

## 6. Candidate set

The canonical candidate-set artifact contains the full frozen candidate sequence generated for the trial, including rejected candidates.

Every candidate records at least:

- proposal identity/digest;
- `physically_admitted: true|false`;
- admission evidence references required by the World Forge boundary.

The array length equals `generated_candidate_count`; the number of admitted entries equals `admissible_candidate_count`.

The ordering is meaningful and frozen. RANDOM_VALID chooses among the admitted subsequence using `sha256-counter-v1`.

Paired primary policies must use the same `candidate_set_sha256` wherever the policy definition allows a shared candidate surface.

## 7. Selected proposal and typed application receipt

For a completed trial, the selected proposal is a physically admitted candidate and the typed application receipt binds:

- `decision_input_sha256`
- `revision_hypothesis_sha256`
- `candidate_set_sha256`
- `selected_proposal_sha256`
- `world_version_before`
- `world_version_after`

The verifier reconstructs these links from raw evidence instead of trusting a runtime PASS label.

## 8. Revisit and outcome

A completed `RevisionOutcome` is admissible only after a durable revisit observation of `world_version_after`.

The revisit retains a committed/grounded provenance domain. A structurally equal counterfactual observation cannot substitute for a committed historical revisit.

## 9. `evidence_index.json`

The index belongs to one trial and maps logical evidence names to relative paths. Relative paths must remain within the trial directory.

It also carries the temporal/order markers required to prove:

`decision/hypothesis closure -> selection -> application -> revisit -> outcome`

The manifest's `evidence_bundle_sha256` is the raw-byte SHA-256 of `evidence_index.json`.

## 10. RANDOM_VALID receipt

RANDOM_VALID emits a receipt bound to:

- algorithm `sha256-counter-v1`
- unsigned 64-bit seed
- paired block ID
- candidate-set digest
- admitted candidate count
- accepted counter
- accepted SHA-256 digest
- selected admitted-candidate index

The independent verifier recomputes the draw from scratch.

## 11. Ablation evidence

A preregistered ablation must be explicit evidence, not ambiguous absence.

Every ablation emits an `ablation_receipt` bound by `manifest.ablation_receipt_sha256`.

For `no_embodied_experience`, the fixed experience slot contains a typed ablation sentinel whose hash occupies `experience_episode_sha256`; this distinguishes intentional channel removal from missing evidence.

For `no_counterfactual_evaluation`, the receipt asserts that counterfactual evaluation did not occur and the exported candidate/evidence surface contains no counterfactual observation/render/score artifacts.

## 12. Policy-output isolation in the pilot

The pilot orchestrator executes each cell with a fresh private staging root and never gives the runtime the shared pilot evidence root. Only after the process exits and the one-trial layout is validated is that sealed trial directory moved into the shared package.

This reduces cross-policy information leakage structurally. The verifier still checks decision-input content and explicit leak markers as defense in depth.

## 13. Scientific failure vs integrity failure

A complete, provenance-valid trial with a poor outcome remains scientific evidence.

A trial with broken hash/provenance/temporal closure is an integrity failure and cannot be silently interpreted as merely poor performance.

Conversely, a valid poor-performing trial cannot be relabeled integrity-invalid after outcomes are known in order to exclude it.

## 14. Trial inventory

`trial_inventory.json` prospectively enumerates the complete trial identities and expected count. The confirmatory freeze binds its raw-byte digest at top-level `trial_inventory_sha256`.

Missing interior trials are selective omission. A truncated favorable prefix without a prospectively frozen stopping rule is unauthorized early stopping. Duplicate identities and unregistered trials are rejected.

## 15. Primary results surface

`primary_results.json` may report the preregistered outcome channels but must never add a forbidden synthetic aggregate such as `world_quality`, `creative_score`, `beauty_score`, `cinematic_quality`, or `intelligence_score`.

World improvement and causal prediction calibration remain separate claims.

## 16. Qualification boundary

Pilot evidence is checked by `verify_vart_world_creative_001_pilot.py` and can establish only plumbing/integrity readiness.

Confirmatory claim admission requires:

1. the externally anchored freeze SHA-256;
2. `verify_vart_world_creative_001_qualified.py`;
3. canonical valid-bundle acceptance;
4. deterministic rejection of N1–N20 with the expected reason classes;
5. complete trial accounting;
6. zero unresolved integrity-invalid confirmatory trials;
7. analysis under the frozen cluster-aware contract.

The maximum claim remains:

`EvidenceBoundExperienceConditionedWorldImprovementQualified`

Nothing in this format establishes general creativity, general intelligence, consciousness, universal aesthetic competence, or physical-world transfer.
