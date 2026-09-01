# VART-WORLD-CREATIVE-001 — Post-Pilot Disposition v1

Status: transition/qualification contract. It does **not** authorize confirmatory execution or scientific claims.

## Purpose

The noncanonical pilot exists to expose instrumentation, serialization, isolation, pairing, evidence-closure, and verifier defects before confirmatory outcomes exist. A `PILOT_PLUMBING_PASS` is therefore necessary but not sufficient for confirmatory freeze.

Before any confirmatory freeze is created, the exact sealed pilot root is independently audited with `scripts/audit_vart_world_creative_001_pilot.py`. The resulting audit and every observed pilot defect must be dispositioned explicitly.

## Required inputs

A post-pilot disposition binds:

- `pilot_receipt_sha256`
- `pilot_evidence_closure_sha256`
- `pilot_design_sha256`
- `pilot_source_head`
- `pilot_source_tree`
- independent audit verdict
- ordered defect records
- exact resolution classification for every defect
- whether a new noncanonical pilot is required
- whether a new preregistration lineage is required

## Paired-design coherence

For every `paired_block_id`, all policies in that block must share the exact same:

- world fixture
- seed
- revision index

A block may not contain the same policy twice.

This rule is checked against `_orchestrator/resolved_plan.json`, which records what was actually executed, rather than trusting a summary table or intended template.

A mismatch is `PAIRED_BLOCK_WORLD_INPUT_MISMATCH` and makes the pilot ineligible to close the pre-confirmatory transition. Because the pilot is noncanonical, this is an instrumentation/protocol finding rather than a failed efficacy result.

## Defect classes

Every defect is assigned exactly one class:

### `instrumentation_plumbing`

Examples:

- serializer/exporter bug
- evidence index omission
- staging/isolation bug
- incorrect pilot seed wiring
- incorrect paired-block wiring
- verifier plumbing defect
- missing run receipt

These defects may be repaired without changing the preregistered scientific hypothesis **only if** the repair does not change the scientific mechanism, metric definition/direction, threshold, candidate budget, stopping rule, multiplicity treatment, or claim criterion.

Any affected pilot must be rerun from a fresh noncanonical evidence root after the repair.

### `scientific_mechanism`

Examples:

- changing FULL policy behavior
- changing candidate generation semantics
- changing physical-admission semantics
- changing the evidence surface available to a policy
- changing an ablation definition
- changing persistent-world continuation semantics

After pilot outcomes have been inspected, such a change requires a **new preregistration lineage**, a fresh pilot, and a fresh future confirmatory evidence root.

### `scientific_contract`

Examples:

- changing hypotheses after viewing outcomes
- changing metric definitions or directions
- changing superiority/noninferiority thresholds
- changing candidate/revision budgets based on pilot outcome magnitudes
- changing stopping rules
- changing multiplicity treatment
- changing missingness/outlier rules to favor observed outcomes

These always require a **new preregistration lineage**.

## Outcome-blind repair rule

Pilot artifacts may be inspected for whether the pipeline behaved correctly. Outcome magnitudes may not be used to tune the confirmatory scientific contract.

A repair record must state which files/fields were inspected and whether policy outcomes, effect magnitudes, human preference values, or comparative rankings were viewed. If those values were used to choose a scientific mechanism or contract change, the old preregistration lineage is closed and cannot support confirmatory execution.

## Confirmatory-freeze eligibility

`confirmatory_freeze_eligible` may be `true` only when all of the following are true:

1. the latest pilot audit verdict is `PILOT_AUDIT_PASS`;
2. the sealed evidence closure matches the pilot receipt;
3. paired-block semantics pass;
4. every defect has an explicit disposition;
5. no unresolved defect remains;
6. no required pilot rerun remains outstanding;
7. no scientific-mechanism or scientific-contract change remains in the current preregistration lineage;
8. the exact source lineage intended for confirmatory execution is durably fetchable and reproducible;
9. confirmatory execution and claim authorization remain false until the separate confirmatory freeze is externally anchored.

The disposition receipt itself never sets either authorization to true.

## Current reported pilot finding

The 2026-09-01 execution report supplied for the live pilot records `PILOT_PLUMBING_PASS`, but its summary matrix assigns different seeds to policies labeled as belonging to the same paired blocks. The remote research template instead uses a shared seed within those blocks.

Until the sealed pilot root is audited, treat this as a reported configuration-drift finding. If `_orchestrator/resolved_plan.json` confirms the differing seeds, classify it as `instrumentation_plumbing`, repair the seed/block wiring, and rerun the noncanonical pilot from a fresh evidence root. Do not tune scientific hypotheses or thresholds from the first pilot's outcome values.

## Claim ceiling

Passing post-pilot disposition establishes only that the experiment is ready to be *frozen prospectively*. It does not establish that FULL Symthaea improves worlds, outperforms baselines, generalizes, becomes better calibrated, or warrants any efficacy claim.
