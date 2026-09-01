# VART-WORLD-CREATIVE-001 — Execution Sequence

Status: execution planning only. This document does not authorize confirmatory execution or scientific claims.

## Authoritative reported runtime parent

The live integration report supplied on 2026-09-01 records WORLD-FORGE v0.5-A as:

- HEAD: `33820b3d9e904280e6264719fe7717cb2e5dd5bb`
- TREE: `e93c6dbfa05b602100ff924efaa5d95f92ef5a65`

and reports the VART-WORLD-CREATIVE-001 integration descendant as commit beginning `844d10609a9f03e26a06...`, tree beginning `38e5506c8f7f88d58e1f...`.

These runtime commits are not yet reachable through the connected GitHub repository, so this sequence MUST NOT be treated as source qualification until the exact commits or an exact tree-equivalent lineage are pushed and independently fetched.

## Phase P0 — Source closure

1. Push the exact qualified v0.5-A runtime parent and the VART integration descendant.
2. Fetch both commits from GitHub and verify exact HEAD/TREE identity.
3. Bind parent qualification receipts and the VART integration receipt.
4. Verify no changes to protected authority boundaries.
5. Keep `execution_authorized=false`, `confirmatory_authorized=false`, and `claim_authorized=false`.

Exit: exact source lineage is remotely reproducible.

## Phase P1 — Noncanonical plumbing pilot

Purpose: discover instrumentation or harness defects only. Pilot results are never admissible for the confirmatory claim.

Run a deliberately small campaign across:

- policies: FULL, RANDOM_VALID, HEURISTIC;
- fixtures: one ordinary fixture plus PrettyTrap and MemoryTrap;
- seeds: a small explicitly noncanonical set;
- cycles: enough to exercise hypothesis -> proposal -> apply -> revisit -> outcome -> calibration closure.

The pilot must verify:

- every `RevisionHypothesis` predates the corresponding world mutation;
- every `RevisionOutcome` is generated only after a revisit observation;
- `CreativeTrial` closes all hashes and world-version transitions;
- rejected candidate evidence is retained;
- random-valid draws only from physically admitted candidates;
- baseline policies use the same candidate set and physical authority gates;
- no aggregate `world_quality` or equivalent scalar is emitted;
- calibration records cannot be rewritten after outcome observation;
- Reality Ledger provenance distinguishes committed, counterfactual, replay and historical observations;
- aborted/incomplete trials cannot enter the confirmatory evidence root.

If the pilot exposes a production mechanism defect, create a new source lineage. Do not patch the mechanism under confirmatory qualification.

## Phase P2 — Freeze the confirmatory protocol

Before inspecting confirmatory outcomes, freeze:

- exact source HEAD/TREE;
- environment/toolchain/GPU/renderer identities where causally relevant;
- fixture corpus and fixture digests;
- seed list;
- policy implementations and policy digests;
- ablation matrix;
- candidate-generation budget;
- revision budget and stopping rule;
- prediction channels and confidence-bound semantics;
- outcome-vector definitions;
- protected physical/safety invariants;
- calibration metrics;
- missing-data/drop policy;
- human-evaluation sampling/blinding protocol;
- success/failure thresholds;
- evidence root;
- independent verifier identity.

Only then may `confirmatory_authorized=true` be considered.

## Phase P3 — Confirmatory campaign

Run paired trials so FULL, RANDOM_VALID and HEURISTIC see equivalent world/seed conditions and the same admissible candidate set wherever the policy definition permits it.

Minimum scientific comparisons:

1. FULL vs RANDOM_VALID — isolates selection/judgment value.
2. FULL vs HEURISTIC — tests whether the cognitive architecture beats an explicit deterministic rule.
3. FULL vs NO_EXPERIENCE — tests the value of embodied journey evidence.
4. FULL vs NO_COUNTERFACTUALS — tests counterfactual evaluation.
5. FULL vs NO_LEDGER — tests provenance/continuity contributions without allowing provenance violations to be silently reclassified.
6. FULL vs NO_DEPTH — tests depth evidence contribution.

Run both:

- longitudinal worlds: repeated revisions in the same persistent world;
- generalization worlds: fewer revisions across unseen worlds.

Adversarial fixtures must include PrettyTrap, LocalOptimum, HiddenDependency, DelayedConsequence, CounterfactualDecoy and MemoryTrap.

## Phase P4 — Analysis without scalar collapse

Report each outcome channel separately. At minimum:

- physical validity;
- declared-goal consequence;
- perceptual consequence;
- side effects;
- counterfactual quality;
- optional blinded human preference;
- prediction error and confidence calibration;
- provenance/integrity violations;
- abort/drop/missing-evidence counts.

Do not publish a single aggregate creative/world-quality score.

For prediction calibration, compare predicted effects against observed effects per channel and over time. Improvement in outcome quality and improvement in causal calibration are separate claims.

## Phase P5 — Independent closeout

An independent verifier must reconstruct trial closure from exported evidence rather than trusting runtime PASS labels.

It must reject at least:

- post-hoc hypothesis mutation;
- candidate-set substitution;
- random-valid selection from an inadmissible candidate;
- world-version mismatch;
- missing revisit observation;
- proposal/receipt mismatch;
- counterfactual/committed provenance substitution;
- dropped or missing evidence hidden as continuity;
- aggregate-score insertion into the qualified result surface;
- duplicate or selectively omitted trials;
- results from the noncanonical pilot entering confirmatory evidence.

Only after this closeout may the bounded claim be admitted:

`EvidenceBoundExperienceConditionedWorldImprovementQualified`

This claim does not establish general creativity, general intelligence, consciousness, universal aesthetic competence, or physical-world transfer.
