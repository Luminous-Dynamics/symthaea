# VART-WORLD-CREATIVE-001 — Evidence Package v1

Status: pilot-format contract. It defines the exporter/verifier boundary and does not authorize confirmatory execution.

## Goal

The runtime producer and independent verifier must exchange evidence through ordinary files and raw-byte SHA-256 digests. The verifier must not import the World Forge decision implementation or trust runtime PASS labels.

All `*_sha256` values in this package refer to the SHA-256 of the exact raw artifact bytes at the referenced path unless a field explicitly states otherwise.

## Root layout

A verifier input root contains:

- `confirmatory_freeze.json`
- `analysis_contract.json`
- `metric_definitions.json`
- `trial_inventory.json`
- `primary_results.json`
- `trials/<trial-id>/manifest.json`
- `trials/<trial-id>/evidence_index.json`
- the per-trial evidence files named by that evidence index

Pilot roots may use the same layout, but every pilot `TrialManifest` must set `campaign="pilot"` and `included_in_confirmatory_analysis=false`. Pilot and confirmatory roots must never be the same directory or digest lineage.

## `evidence_index.json`

Each trial directory contains an evidence index with at least:

- `trial_id`
- `files`, a map from logical names to safe relative paths
- `timestamps_ns`
- `cross_policy_outcome_observed_before_selection`
- `prospective_exclusion_reason_classes`

Required logical files for a complete ordinary trial:

- `experience_episode`
- `revision_hypothesis`
- `candidate_set`
- `selected_proposal`
- `applied_receipt`
- `revisit_observation`
- `revision_outcome`

`random_valid` additionally requires `random_draw_receipt`.

The SHA-256 of the raw `evidence_index.json` bytes is recorded as `evidence_bundle_sha256` in the trial manifest. The index is therefore a closure map, not an unbound convenience file.

Paths must be relative, must not contain `..`, and must remain inside the trial directory.

## Temporal closure

A complete trial exports integer monotonic evidence timestamps:

- `hypothesis_closed`
- `selection_closed`
- `applied_receipt`
- `revisit_closed`
- `outcome_closed`

The required ordering is:

`hypothesis_closed <= selection_closed < applied_receipt <= revisit_closed <= outcome_closed`

The timestamp source and clock semantics must be frozen before confirmatory execution. These timestamps establish ordering, not cross-host wall-clock truth.

## Candidate-set artifact

`candidate_set` is the exact ordered candidate surface presented to the policy after physical admission work. It contains a `candidates` array. Each candidate contains at minimum:

- `proposal_sha256`: raw-byte SHA-256 of that proposal artifact;
- `physically_admitted`: boolean.

The order of physically admitted candidates is part of the scientific contract. It must not be re-sorted by policy. FULL, RANDOM_VALID and HEURISTIC paired trials use the exact same candidate-set artifact/digest wherever the preregistered policy definition permits that comparison.

`selection_index` is the index within the ordered subsequence of candidates whose `physically_admitted` value is true.

## Proposal and application closure

For a complete trial:

- the selected admitted candidate's `proposal_sha256` equals `TrialManifest.selected_proposal_sha256`;
- the `selected_proposal` file bytes hash to that same digest;
- the applied receipt records the same selected proposal digest;
- the applied receipt records `world_version_before` and `world_version_after` matching the trial manifest;
- the revisit observation records the resulting `world_version_after`.

A structurally equal counterfactual cannot substitute for a committed observation. Revisit provenance must remain in an admitted committed/grounded provenance domain.

## RANDOM_VALID receipt

RANDOM_VALID uses `sha256-counter-v1` from `VART_WORLD_CREATIVE_001_RANDOM_VALID_V1.md`.

The receipt contains:

- `algorithm`
- `seed`
- `paired_block_id`
- `candidate_set_sha256`
- `admissible_candidate_count`
- `counter`
- `accepted_digest_sha256`
- `selected_index`

The independent verifier recomputes all of these values.

## Freeze bindings

`confirmatory_freeze.json` binds at minimum the raw-byte SHA-256 of:

- `analysis_contract.json`
- `metric_definitions.json`
- `trial_inventory.json`

Every trial manifest independently repeats the analysis-contract and metric-definition digests. This prevents a trial from being interpreted under a different metric direction or statistical contract after execution.

## Trial inventory

`trial_inventory.json` prospectively enumerates every confirmatory `trial_id` and an `expected_trial_count` equal to the number of enumerated IDs unless a different preregistered stopping design is introduced in a new contract version.

Missing preregistered trials, duplicate identities, unregistered trials, or selective removal of failures are verifier failures. Aborted, integrity-invalid and missing-evidence trials remain represented rather than disappearing.

## Scientific failure versus integrity failure

A complete trial with valid evidence closure may produce a bad outcome. That is a scientific negative result and remains analyzable.

A trial with broken hash/provenance/ordering closure is `invalid_integrity`; it cannot be silently treated as merely poor performance.

Conversely, a complete integrity-valid poor trial cannot be relabeled invalid to remove it from analysis. Any allowed confirmatory exclusion class must have been frozen prospectively.

## Primary results

`primary_results.json` reports preregistered channels separately. Forbidden primary aggregate keys include at least:

- `world_quality`
- `creative_score`
- `beauty_score`
- `cinematic_quality`
- `intelligence_score`

Additional forbidden aggregate names may be frozen prospectively.

## Independent verifier

`scripts/verify_vart_world_creative_001.py` is the initial standalone verifier implementation for this contract. It checks raw-byte digest closure, paired candidate-set equality, reproducible RANDOM_VALID selection, world-version continuity, provenance boundaries, inventory completeness, pilot contamination, aggregate-score prohibition, analysis/freeze bindings, longitudinal world chains, and information-leak markers.

The verifier itself is not qualified until the full N1–N20 negative suite is implemented and deterministically rejected for the expected reason classes. The synthetic script test is only an implementation smoke gate, not verifier qualification.
