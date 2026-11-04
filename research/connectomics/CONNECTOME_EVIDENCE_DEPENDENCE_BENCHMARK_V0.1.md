# Connectome Evidence Dependence and False-Corroboration Benchmark V0.1

**Status:** research protocol and synthetic benchmark design  
**Issue:** [#10404 — SCI-CONNECTOMICS-070](https://github.com/Luminous-Dynamics/symthaea/issues/10404)  
**Date:** 2026-11-04  
**Scope:** provenance dependence, claim-fusion eligibility, and conservative abstention  
**Claim ceiling:** synthetic contract validation only; no biological independence or truth claim

## 1. Objective

Connectome evidence systems frequently collect multiple outputs that appear to corroborate one another: specialist models, vision-language models, morphology rules, graph heuristics, reviewers, adjudicators, acquisition modalities, and statistical analyses.

Those outputs may share upstream measurements, segmentation, annotation, training data, preprocessing, registration, selection, or human exposure. Different artifacts and producer names do not establish independent information.

This benchmark tests whether a system can distinguish:

- multiple records from multiple independent information pathways;
- artifact integrity from provenance completeness;
- provenance completeness from actual independence;
- claim support from claim-fusion eligibility;
- known dependence from partially identified or unknown dependence.

The objective is not to estimate one universal correlation coefficient. It is to prevent unsupported confidence promotion and to expose the assumptions required by any fusion procedure.

## 2. Relation to existing programme

This protocol composes with, but does not replace:

- [#10349](https://github.com/Luminous-Dynamics/symthaea/issues/10349): typed multiplex claims across anatomy, correspondence, statistics, function, and causality;
- [#9255](https://github.com/Luminous-Dynamics/symthaea/issues/9255): query-scoped proof-carrying reliability certificates;
- [#9344](https://github.com/Luminous-Dynamics/symthaea/issues/9344): query evaluation across plausible reconstruction worlds;
- [#9186](https://github.com/Luminous-Dynamics/symthaea/issues/9186): multi-rater reliability and reviewer calibration;
- [#9403](https://github.com/Luminous-Dynamics/symthaea/issues/9403): domain shift, calibration, and safe abstention;
- [#10397](https://github.com/Luminous-Dynamics/symthaea/issues/10397): statistical hypothesis-family closure and multiplicity;
- [#5771](https://github.com/Luminous-Dynamics/symthaea/issues/5771): evidence-bound claim graphs and non-laundering derivation.

A provenance graph can be structurally valid while omitting an unrecorded common cause. The benchmark therefore distinguishes verified declared relationships from unknown ancestry. It must not claim that a finite manifest proves the absence of all hidden dependence.

## 3. Core model

Represent evidence as a directed, typed provenance graph:

```
source image / acquisition
        |
        +---- preprocessing / registration
        |              |
        |              +---- specialist model output
        |              +---- VLM output
        |
        +---- segmentation revision
                       |
                       +---- morphology features
                       +---- synapse candidate
                       +---- graph heuristic
```

A separate reviewer-exposure graph records whether human decisions were made independently, blinded, or after exposure to model suggestions or earlier votes.

Every evidence record should bind:

- stable evidence ID and evidence-role type;
- exact artifact digest and immutable source/revision identity;
- producer/model/version digest, where applicable;
- training and fine-tuning data commitments, where known;
- source image, segmentation, annotation and preprocessing ancestry;
- acquisition session, volume, region and coordinate-transform ancestry;
- reviewer/adjudicator pseudonymous ID and exposure protocol;
- declared dependence edges and their provenance;
- explicit unknown or unavailable ancestry fields.

A digest proves byte identity. A signature authenticates a signer under a trust policy. Neither proves independence.

## 4. Dependence relation vocabulary

V0.1 uses a small closed vocabulary:

| Relation | Meaning |
|---|---|
| `DERIVED_FROM` | Artifact was computed from another artifact |
| `SHARES_SOURCE_MEASUREMENT` | Records depend on the same underlying measurement |
| `SHARES_TRAINING_DATA` | Models share known training or fine-tuning data |
| `SHARES_ANNOTATION_LINEAGE` | Labels or annotations share an upstream source |
| `SHARED_REVIEW_EXPOSURE` | Reviewers saw the same suggestion, cue, or earlier vote |
| `SHARED_SELECTION_MECHANISM` | Evidence entered the sample through a shared selection process |
| `INDEPENDENT_UNDER_DECLARED_DESIGN` | A specific preregistered design supports a bounded independence claim |

The last relation is not a free producer assertion. It must reference a pinned protocol that specifies the independence unit, blinding, sampling, source separation, and any remaining shared causes.

Unknown ancestry is a first-class state. It must never be silently rewritten as independence.

## 5. Fusion eligibility states

The protocol separates provenance from inference:

- `PROVENANCE_COMPLETE_WITHIN_DECLARED_SCOPE`: all required fields for the declared scope are present and internally consistent.
- `KNOWN_DEPENDENCE`: at least one relevant shared ancestry or exposure relation is present.
- `PARTIALLY_IDENTIFIED_DEPENDENCE`: some dependence is known, while other relevant paths remain unresolved.
- `DEPENDENCE_UNKNOWN`: required ancestry is absent or cannot be verified.
- `INDEPENDENCE_SUPPORTED_WITHIN_DECLARED_DESIGN`: a pinned design supports the specified independence claim within its declared scope.
- `FUSION_NOT_ELIGIBLE`: the requested fusion rule assumes more independence than the evidence supports.
- `SENSITIVITY_ONLY`: results are reported over a declared family of admissible dependence structures.
- `NOT_IDENTIFIABLE` / `NOT_ESTIMABLE`: the available evidence cannot support the requested fusion estimand.

These are not biological truth labels. The validator may establish that a producer's declaration is internally consistent with the supplied graph; it cannot prove that the graph captures every real-world common cause.

## 6. Synthetic known-answer corpus

The initial corpus must include at least the following cases:

1. **Shared image and segmentation:** two models have different weights and outputs but consume the same EM crops and segmentation revision. They are not independent confirmations of segmentation truth.
2. **Partial training overlap:** a specialist model and VLM have partially shared training data. Dependence remains partial or unknown; do not treat model names as independent evidence.
3. **Blinded independent adjudication:** two reviewers receive the same candidate but are blinded to model identity and each other's decisions. The independence claim is limited to what the protocol actually establishes.
4. **Shared suggestion exposure:** reviewers see the same model suggestion before voting. Mark shared exposure; reject naïve independent-vote multiplication.
5. **Shared acquisition transform:** distinct modalities were acquired in one session and registered through the same transform. Preserve the common transform ancestry.
6. **Inherited segmentation error:** a synapse classifier and graph heuristic both consume the same flawed segmentation. Their agreement does not independently establish synapse truth.
7. **Unknown training ancestry:** training-data lineage is missing. Unknown is not independent.
8. **False independence assertion:** a producer declares independence despite a known shared ancestor. Reject the declaration.
9. **Stale dependency graph:** an ancestor digest changes or a dependency edge is removed after fusion. Invalidate every dependent fusion result.
10. **Contradictory or cyclic provenance:** reject invalid graph structures or return a specific non-estimable status.
11. **Independent negative control:** an independently acquired control disagrees with several correlated positive outputs. Preserve the conflict; do not majority-vote away the independent evidence.
12. **Partial identification:** several dependence structures remain admissible. Report a sensitivity envelope or abstain; do not invent a single posterior.

Every fixture includes evaluator-only expected disposition, a frozen input digest, and at least one single-field mutation. Evaluator-only truth is unavailable to candidate ranking or runtime fusion.

## 7. Comparator methods

Evaluate the same frozen evidence bundles with:

1. Naïve record-count majority vote.
2. Naïve independent likelihood multiplication.
3. Model-name or modality-count heuristic (negative control).
4. Provenance-aware dependence grouping with conservative fusion.
5. Sensitivity analysis over admissible dependence structures.
6. Abstention/escalation when dependence cannot be bounded.
7. Independent expert adjudication under a preregistered blinded protocol.

The first three are intentionally unsafe comparators, not recommended production methods.

## 8. Metrics

Report separately:

- false independent-corroboration rate;
- false claim-promotion rate under shared-ancestry attacks;
- calibration and proper scoring rules under known dependence structures;
- coverage and abstention under unknown/partially identified dependence;
- sensitivity to omitted/added ancestry edges and stale source revisions;
- provenance graph completeness and contradiction-detection recall;
- number and type of unsupported independence declarations;
- review minutes and adjudication cost needed to resolve dependence ambiguity;
- downstream query-result changes caused by corrected dependence assumptions.

Do not compress these metrics into one score. A low false-promotion rate achieved by abstaining on every case is not a useful system; report coverage and cost alongside safety.

## 9. Proposed promotion gates

These are design goals, not results:

- Zero false independence promotions on exhaustive synthetic negative controls.
- 100% detection of enumerated shared-ancestor and shared-exposure attacks.
- Unknown ancestry never defaults to independent.
- No probabilistic fusion unless the dependence model, assumptions, and calibration domain are explicit.
- Deterministic independent replay for identical pinned inputs.
- Any changed dependency ancestor invalidates every derived fusion result in its transitive closure.
- No unrelated result is invalidated unless a declared dependency path justifies it.
- Real-data reports expose dependence assumptions and sensitivity bounds; synthetic validation is never described as biological independence.

## 10. V0.1 implementation boundary

The first implementation should be a standalone Rust validator over a strict versioned JSON bundle:

1. Validate schema and unique evidence IDs.
2. Recompute exact artifact and bundle digests from supplied bytes.
3. Validate relation endpoints, relation types, and acyclicity where required.
4. Check that every claimed independence relation references an allowed, pinned protocol.
5. Propagate unknown ancestry as unknown; never infer independence from missing edges.
6. Recompute dependence/fusion eligibility counters.
7. Invalidate derived receipts when any transitive ancestor changes.
8. Emit deterministic JSON with separate structural, provenance, dependence and fusion dispositions.

Do not implement generic Bayesian fusion in V0.1. A later statistical layer must define the estimand, dependence model, identification assumptions, and calibration design before any posterior or combined probability is emitted.

## 11. Claim ceiling

A passing validator establishes only that the supplied provenance graph and declared fusion contract satisfy the implemented structural rules. It does not establish:

- that every real common cause was recorded;
- that two measurement processes are biologically independent;
- that reviewer agreement is biological truth;
- that a model is accurate or calibrated on a new domain;
- that an anatomical, functional, statistical or causal claim is true.

The appropriate result when dependence cannot be bounded is an explicit limitation, not a more confident answer.
