# VART-WORLD-CREATIVE-001 — Execution Sequence

Status: execution planning only. This document does not authorize confirmatory execution or scientific claims.

## Authoritative reported runtime parent

The live integration report supplied on 2026-09-01 records WORLD-FORGE v0.5-A as:

- HEAD: `33820b3d9e904280e6264719fe7717cb2e5dd5bb`
- TREE: `e93c6dbfa05b602100ff924efaa5d95f92ef5a65`

and the fully qualified local VART-WORLD-CREATIVE-001 integration as:

- HEAD: `844d10609a9f03e26a06f22778db4b8cdfb6a3ef`
- TREE: `38e5506c8f7f88d58e1ff03a77585091d9263a98`

The runtime commit remains local-only from the connected GitHub view. Do not replace it with an approximate remote tree.

## Phase P0 — Source and instrument closure

1. Bind the exact qualified VART runtime HEAD/TREE above.
2. Require a clean runtime working tree.
3. Preserve the v0.5-A and VART integration qualification receipts.
4. Bind the research-contract branch and evidence-format versions used by the pilot.
5. Keep confirmatory execution and claim authorization false.

Exit: the runtime and research instrument are unambiguously identified.

## Phase P1 — Noncanonical plumbing pilot

Purpose: discover instrumentation, serialization, pairing, adapter, or verifier defects only. Pilot results are never admissible for the confirmatory efficacy claim.

Run exactly the frozen eight cells in `VART_WORLD_CREATIVE_001_PILOT_RUN.template.json` after binding its unresolved `runtime_argv` to the real local VART trial entrypoint.

The orchestrator must:

- verify exact clean runtime HEAD/TREE before creating evidence;
- use a fresh pilot root;
- execute each policy in a fresh private staging root so other policy outputs are not visible through the experiment output surface;
- bind pilot-only analysis and metric-definition digests;
- retain all generated candidates, including rejected candidates;
- require byte-identical decision-input and candidate-set surfaces for paired primary policies where preregistered;
- verify RANDOM_VALID through `sha256-counter-v1`;
- require explicit ablation receipts/sentinels;
- invoke `verify_vart_world_creative_001_pilot.py` only after all eight trial directories are sealed.

The pilot must verify:

- every `RevisionHypothesis` predates the corresponding world mutation;
- every `RevisionOutcome` is generated only after a revisit observation;
- applied receipts bind the decision input, hypothesis, candidate set, selected proposal, and world transition;
- `CreativeTrial` closes all hashes and world-version transitions;
- rejected candidates remain evidence;
- random-valid draws only from admitted candidates and are independently reproducible;
- no aggregate `world_quality` or equivalent scalar is emitted;
- calibration records cannot be rewritten after outcome observation;
- Reality Ledger provenance distinguishes committed, counterfactual, replay and historical observations;
- aborted/incomplete/integrity-invalid trials cannot disappear from accounting;
- pilot trials cannot enter confirmatory analysis.

If the pilot exposes a production mechanism defect, create a new runtime source lineage. If it exposes a scientific-contract defect, create a new preregistration lineage. Do not silently repair either lineage in place after inspecting outcome values.

Pilot exit claim: at most `PILOT_PLUMBING_PASS`.

## Phase P2 — Freeze the confirmatory protocol

Before inspecting confirmatory outcomes, freeze:

- exact source HEAD/TREE and qualification receipt;
- environment/toolchain/GPU/renderer identities where causally relevant;
- fixture corpus and fixture digests;
- hidden generalization fixture corpus and reveal protocol;
- seed list;
- policy implementations and policy digests;
- ablation implementations and digests;
- candidate generator and physical-admission policy;
- candidate/revision budgets and stopping rule;
- prediction channels and confidence semantics;
- metric definitions and directions;
- analysis contract, cluster unit, uncertainty method and multiplicity policy;
- missingness/abort/exclusion policy;
- human-evaluation sampling/blinding protocol;
- complete trial inventory;
- evidence root;
- independent verifier source;
- qualified verifier wrapper;
- N1–N20 verifier qualification suite.

Set `frozen=true` only when all required fields are complete.

Then compute SHA-256 over the exact raw bytes of `confirmatory_freeze.json` and record that digest **outside the confirmatory evidence root** before execution (for example in a dedicated signed/tagged preregistration commit or immutable receipt). The freeze file must not try to contain its own hash.

Confirmatory closeout must receive that externally anchored digest through:

`verify_vart_world_creative_001_qualified.py --expected-freeze-sha256 <digest>`

Only after that anchor exists may confirmatory execution authorization be considered.

## Phase P3 — Verifier qualification before scientific interpretation

Before confirmatory results may be interpreted for the bounded claim:

1. a canonical valid evidence bundle is accepted;
2. all N1–N20 attacks are deterministically rejected for their preregistered reason classes;
3. the verifier source/executable digest is frozen;
4. the qualified verifier requires the external freeze anchor;
5. verifier changes after outcome inspection create a new verifier lineage and are disclosed.

CI may execute the synthetic N1–N20 suite continuously, but CI success alone does not authorize a scientific claim; the actual frozen verifier digest must be bound into the confirmatory freeze.

## Phase P4 — Confirmatory campaign

Run paired trials so FULL, RANDOM_VALID and HEURISTIC see equivalent world/seed conditions and the same decision/candidate surfaces wherever the policy definition permits it.

Minimum scientific comparisons:

1. FULL vs RANDOM_VALID — isolates selection/judgment value.
2. FULL vs HEURISTIC — tests whether the cognitive architecture beats an explicit deterministic rule.
3. FULL vs NO_EXPERIENCE — tests the value of embodied journey evidence.
4. FULL vs NO_COUNTERFACTUALS — tests counterfactual evaluation.
5. FULL vs NO_LEDGER — tests provenance/continuity contributions without allowing provenance violations to be silently reclassified.
6. FULL vs NO_DEPTH — tests depth evidence contribution.

Run both:

- longitudinal worlds: repeated revisions in the same persistent world, analyzed with world identity as the cluster unit;
- generalization worlds: fewer revisions across genuinely unseen worlds.

Adversarial fixtures include PrettyTrap, LocalOptimum, HiddenDependency, DelayedConsequence, CounterfactualDecoy and MemoryTrap.

## Phase P5 — Analysis without scalar collapse

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

Improvement in outcome quality and improvement in causal calibration remain separate hypotheses and separate claims.

Repeated revisions from one world are not independent replicates.

## Phase P6 — Independent closeout

The qualified verifier reconstructs trial closure from exported evidence rather than trusting runtime PASS labels.

It rejects at least:

- post-hoc hypothesis mutation;
- candidate-set or decision-input substitution;
- random-valid selection manipulation;
- inadmissible selection;
- world-version mismatch;
- missing revisit;
- proposal/receipt mismatch;
- counterfactual/committed provenance substitution;
- pilot contamination;
- selective omission and unauthorized early stopping;
- duplicate trial identities;
- forbidden aggregate scores;
- metric/threshold/stopping-rule mutation;
- evidence truncation;
- cross-policy information leakage;
- hidden generalization-fixture leakage;
- scientific/integrity failure reclassification.

Only after this closeout may the bounded claim be admitted:

`EvidenceBoundExperienceConditionedWorldImprovementQualified`

This claim does not establish general creativity, general intelligence, consciousness, universal aesthetic competence, or physical-world transfer.
