# PSAFE: Psychoactive Safety and Evidence Benchmark

**Status:** research protocol proposal v0.1 — design specification only; no benchmark result is claimed  
**Owner:** Symthaea research  
**Target home:** `crates/domains/symthaea-psych-bench` integration after review  
**Date:** 2026-10-20

## Executive thesis

Symthaea should not try to win by generating the most novel psychoactive molecule or optimizing a single subjective effect. It should win by helping researchers make **more reliable, safer, reproducible decisions from incomplete pharmacology evidence**.

The repository already contains a broad psychological benchmark suite and a neuromodulator-dynamics component. Those are useful starting points, not validation that the system can predict human drug effects. PSAFE proposes a separate, evidence-first evaluation lane linking known-compound observations to pharmacology, toxicology, clinical evidence, and cautiously interpreted reports of experience.

The initial deliverable is a benchmark protocol and data contract. It is not a validated model, clinical tool, drug-discovery result, or safety certification.

## Scope and safety boundary

### In scope

- Evidence retrieval, citation tracing, and source-quality assessment for **known, characterized compounds**.
- Assay and unit harmonization that preserves the original measurement and experimental context.
- Evaluation of established, generic ADMET and toxicology tasks using public benchmark data.
- Safety-oriented synthesis of known interaction warnings, adverse-event signals, clinical trial records, and uncertainty.
- Research into how existing, documented pharmacological observations relate to cognition and reported experience, without treating subjective reports as a direct measurement of mechanism.
- Triage of evidence gaps and prioritization of follow-up **safety** measurements under qualified human review.

### Explicitly out of scope for this first benchmark

- De novo psychoactive molecule generation or synthesis instructions.
- Ranking novel analogues for greater potency, stronger intoxication, dependence liability, or a target subjective experience.
- Recommending doses, combinations, routes of administration, or ways to evade safeguards.
- Treating inferred receptor activity, simulation outputs, or anecdotal reports as proof of human effects.
- Automated decisions to administer a compound or substitute for toxicology review, clinical judgment, ethics review, or regulatory authorization.

These boundaries preserve a legitimate research program across recreational-substance characterization, entheogenic research, and therapeutic development while keeping the benchmark centered on evidence quality, safety, and accountable human decisions.

## The paradigm shifts

### 1. From a compound score to an evidence graph

A single scalar such as an aggregate "quality" or "safety" score hides important conflicts and trade-offs. PSAFE should represent each claim as a traceable evidence item and retain conflicting observations rather than averaging them away.

An evidence item should preserve, when available:

- Stable compound identity and structure record; salt, stereochemistry, and formulation distinctions must not be silently collapsed.
- Source record, publication or registry identifier, publication date, retrieval date, checksum, and license/redistribution status.
- Assay identifier, target or endpoint, original measured value and unit, standardized value and unit, assay type, species, tissue/cell system, and experimental conditions.
- Study design, sample size where applicable, uncertainty or variability reported by the source, and known limitations.
- Extraction method/version, normalization steps, reviewer or adjudication state, and links to contradictory or corroborating records.

An unknown field remains unknown. Missing context must not be fabricated or filled from another study without an explicit, traceable join.

### 2. From confident prediction to calibrated uncertainty and abstention

For each prediction, return (a) a result, (b) uncertainty or calibration information, (c) applicability-domain status, (d) evidence provenance, and (e) a reason to abstain when support is insufficient.

An out-of-domain compound, incompatible assay, weak label, or unresolved identity should produce an explicit abstention/review result rather than a confident guess. Confidence is a quantity to validate, not a synonym for model fluency.

### 3. From benchmark performance to safety-weighted utility

Ordinary average accuracy can hide rare but consequential false negatives. Report endpoint-specific sensitivity, specificity, precision-recall, false-negative counts, confidence intervals, calibration, and workload imposed on human reviewers. Define severe-hazard categories and decision thresholds before opening the locked test set.

No single model score may waive a critical safety review. A "safe" prediction is never proof of safety.

### 4. From random splits to realistic tests of generalization

Use at least three distinct evaluation regimes:

1. **Standard benchmark split:** reproduce the published split and metric for comparability.
2. **Chemical-structure holdout:** scaffold/group split, deduplicated by exact structure and relevant parent/assay families.
3. **Temporal/prospective holdout:** train only on evidence available before a frozen date; test on later publications or independently collected records.

Also audit duplicates, salts/tautomers, shared papers, assay-series leakage, and near-identical records crossing partitions. Keep an untouched lockbox for final evaluation. Report the exact data snapshot, split digest, model configuration, seeds, and environment.

### 5. From leaderboard chasing to reproducible science

A result counts only when another researcher can reproduce it from available data, code, and a documented environment. This is a differentiator: a 2026 audit of top-ranked TDC ADMET models reported that only 3 of 10 assessed models passed all of its reproducibility and methodological checks. PSAFE should treat reproducibility and leakage audits as core benchmark dimensions, not optional documentation.

### 6. From maximizing information to maximizing decision value

Once a reliable baseline exists, an active-learning component can propose which missing **safety-relevant measurement** would most reduce uncertainty or change a predefined safety decision. Measure information gain and human-review cost. Do not let the system translate this into autonomous synthesis or potency optimization.

### 7. From context-free reports to culturally and epistemically responsible evidence

Human-reported experience is important but heterogeneous. Keep direct measurement, clinician-reported outcome, participant report, historical account, and culturally governed traditional knowledge as distinct evidence types. Do not flatten them into a universal score. Any future collection involving people or Indigenous knowledge requires appropriate consent, ethics review, data governance, attribution, and benefit-sharing.

## Benchmark tracks

### Track A — Evidence retrieval and claim support

**Question:** Can the system find the right source and distinguish what it says from what it does not say?

- Test against a human-adjudicated set of questions about known-compound safety and established pharmacology.
- Measure retrieval Recall@k, MRR/nDCG, exact record/identifier accuracy, citation precision, citation completeness, and unsupported-claim rate.
- Include adversarial negatives: similar compound names, different stereoisomers/salts, conflicting papers, incorrect units, retracted or superseded findings, and sources that do not support the stated conclusion.
- Require source-level support for every factual high-risk claim. A citation that merely mentions a compound does not count as supporting a claim.

### Track B — Measurement and provenance integrity

**Question:** Can the data pipeline preserve what was actually measured?

- Test canonicalization and identity resolution, unit conversion, censoring/inequality handling (for example, values reported as less than a threshold), repeated measurements, and assay context.
- Keep raw and transformed values side by side.
- Required negative controls: malformed structures, unsupported units, ambiguous compound identities, missing assay context, duplicate papers, and contradictory measurements.
- Report exact-match correctness, unit-conversion error rate, duplicate leakage, provenance completeness, and fail-closed behavior.

### Track C — Generic ADMET/toxicology baselines

**Question:** Does Symthaea improve on transparent, reproducible baselines for established endpoints?

- Start with public Therapeutics Data Commons ADMET tasks and use the versioned official splits for a comparable evaluation.
- Report each endpoint separately; do not hide a weak toxicity endpoint inside an average across easier tasks.
- Baselines: majority/mean predictor, regularized linear model, molecular-fingerprint plus random-forest/gradient-boosting baseline, and the strongest public model that can be independently executed under the same split.
- Metrics: AUROC where appropriate, AUPRC for rare positives, sensitivity/specificity at prespecified thresholds, MAE or Spearman correlation for continuous endpoints, Brier score and calibration plots, plus confidence intervals.
- This track tests generic safety-prediction capability. It does **not** establish predictive validity for psychedelic subjective effects or any specific clinical indication.

### Track D — Safety evidence synthesis

**Question:** Can it synthesize different kinds of evidence without overstating what they establish?

- Separate controlled study evidence, official prescribing/regulatory material, trial registry entries, assay measurements, pharmacovigilance reports, and anecdotal/self-reported data.
- Distinguish hazard identification, association, causality, and incidence. A spontaneous adverse-event report alone cannot establish that a compound caused an event or estimate event incidence.
- Evaluate correct contradiction detection, correct evidence-grade assignment, interaction-warning citation accuracy, source freshness, and appropriate abstention when the evidence cannot answer the question.

### Track E — Experience and cognition evidence (later phase)

**Question:** Can the system organize existing, ethically sourced observations without confusing subjective reports with objective mechanism?

- Start with publicly documented, already characterized compounds and existing, consented research datasets where reuse is permitted.
- Use explicit outcome definitions, study-level splits, cultural/context metadata where ethically appropriate, and preregistered hypotheses.
- Compare Symthaea with simple statistical baselines and expert-curated evidence summaries.
- Do not use this track to rank novel compounds for intensity, desirability, or other subjective-effect optimization.

## Proposed targets and release gates

The values below are **engineering targets for the pilot**, not observed results or claimed state-of-the-art performance. Confirm that the dataset is large and sufficiently representative before interpreting a threshold.

| Gate | Initial target | Failure response |
|---|---|---|
| Data identity and transformation | 100% of frozen unit/identity negative-control tests pass | Block release; do not silently coerce ambiguous records |
| Leakage | Zero known exact-identity, duplicate-record, or source-group leakage across the locked partitions; disclose residual scaffold similarity | Rebuild splits and rerun the full evaluation |
| Provenance | 100% of displayed high-risk claims have source-linked evidence; target ≥98% citation correctness on the adjudicated set | Block high-risk output or require human review |
| Severe-risk detection | Pilot aim: ≥95% sensitivity for prespecified severe-hazard labels on a sufficiently powered external set, with confidence intervals and false-positive workload reported | Do not claim the gate is met if sample size or label quality is insufficient |
| Calibration | Publish Brier score, reliability diagram, calibration error, and selective risk/coverage; target calibrated probabilities before using decision thresholds | Use abstention; recalibrate on validation only, never on the final test set |
| Generalization | Report standard, scaffold/group, and temporal splits separately; no claim based only on random split performance | Restrict claims to the validated domain |
| Reproducibility | One documented command or workflow, pinned data manifest, environment/config digest, repeated seeds, and independently repeatable metrics | Results remain provisional |
| Model improvement | Beat the strongest reproducible matched baseline on a preregistered primary metric with uncertainty intervals, on more than one independent holdout where the data support it; no regression in severe-risk sensitivity | Report no improvement if evidence does not support it |

A 95% sensitivity target is not meaningful if the external set contains too few independent severe-hazard cases to estimate it reliably. Report counts, denominators, intervals, and label uncertainty; never turn an underpowered result into a pass. Thresholds must be frozen before looking at final test outcomes.

For robust comparisons, use at least five independent seeds when training is stochastic, report mean and dispersion, and use bootstrap confidence intervals on the same locked test examples. Any claimed win over a published model must name its exact version and reproduce its baseline under the same split and compute constraints. Published leaderboard scores alone are not a fair comparator.

## Competitive positioning

| Existing ecosystem | What it gives us | PSAFE's proposed contribution |
|---|---|---|
| Therapeutics Data Commons (TDC) ADMET | Public tasks, standardized splits/metrics, leaderboard comparability | Reproduce relevant baseline tasks, then add safety-focused provenance, calibration, leakage and domain audits |
| ChEMBL and BindingDB | Curated experimental activity and binding records | Retain assay context and conflicting measurements; avoid treating heterogeneous values as interchangeable |
| Published ADMET predictors | Strong model baselines on generic endpoints | Compare only models we can actually run and reproduce; report failures honestly |
| Regulatory material and clinical-trial registries | Clinical-development and risk-management context | Trace claims to primary sources and keep computational evidence distinct from human evidence |
| Symthaea psych-bench and neuromodulator components | Cognitive benchmark infrastructure and candidate mechanistic hypotheses | Test connections only when external evidence supports them; no inference that internal simulation equals human pharmacology |

### What would genuinely count as better than the state of the art?

Not an attractive demo or a higher number on one random split. A defensible claim would require:

1. A frozen, auditable dataset and leakage-resistant evaluation.
2. Reproduced strong baselines under the same splits.
3. Statistically supported improvement on a preregistered endpoint or a clearly demonstrated improvement in calibration/abstention at matched sensitivity.
4. Better evidence traceability and fewer unsupported claims, measured by independent adjudication.
5. Prospective or temporally held-out validation, followed by external replication before making broad claims.

The potentially distinctive product is the combination of **predictive performance, explicit uncertainty, evidence traceability, and reproducible failure detection**. The platform must earn each part independently.

## Data sources and operating rules

- **Therapeutics Data Commons (TDC):** start with its ADMET benchmark group and official split/evaluation protocol: https://tdcommons.ai/benchmark/admet_group/overview/
- **ChEMBL:** experimental binding, functional, ADME, toxicity, and physicochemical assay data; preserve assay type, standard values/units, and source identifiers: https://chembl.gitbook.io/chembl-interface-documentation/
- **BindingDB:** experimentally measured protein–small-molecule binding data with publication-linked records: https://www.bindingdb.org/rwd/bind/info.jsp
- **OECD QSAR guidance:** validation principles and a defined applicability domain are prerequisites for responsible interpretation: https://www.oecd.org/en/publications/guidance-document-on-the-validation-of-quantitative-structure-activity-relationship-q-sar-models_9789264085442-en.html
- **FDA psychedelic-drug guidance:** use for clinical-development context and human-study design boundaries, not as a source of molecular potency labels: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/psychedelic-drugs-considerations-clinical-investigations
- **TDC reproducibility audit (2026):** an external cautionary study on reproducibility, data leakage, and overfitting: https://pubmed.ncbi.nlm.nih.gov/42392971/

Before redistribution, verify each dataset's current license, allowed uses, version, and attribution requirements. Store acquisition timestamps, checksums, source release identifiers, and transformations. Keep personal data out of the first release; any later human-subject dataset requires separate privacy, consent, ethics, and governance review.

## Implementation sequence

1. **Protocol and source audit (this proposal):** agree the scope, task definitions, claim ceiling, source/license rules, and release gates before collecting data.
2. **Small frozen pilot:** choose a manageable subset of generic ADMET endpoints and evidence-retrieval questions; produce a data manifest and leakage report. Do not start with de novo molecule generation.
3. **Reproducible baselines:** execute trivial and fingerprint-based baselines first; record exact commands, seeds, versions, time, and memory.
4. **Evidence and prediction contracts (implemented in this draft; tests not yet run):** `pharmacology_evidence.rs` captures source provenance, reuse status, raw and normalized measurements, assay context, review state, limitations, conflict links, structural validation, and a deterministic BLAKE3 content digest. `pharmacology_prediction.rs` defines safety-only prediction outputs, model/training/split provenance, calibration evidence, applicability-domain state, and explicit abstention. Out-of-domain records must abstain, and a calibrated-status claim requires a validation method and manifest digest.
5. **Safety metric runner (implemented in this draft; tests not yet run):** `pharmacology_metrics.rs` reports sensitivity, specificity, precision, negative predictive value, balanced accuracy, AUROC, non-interpolated average precision, Brier score, equal-width-bin expected calibration error, and coverage/abstention counts. Metrics that are undefined on a degenerate set remain `None`; all probability metrics exclude abstentions and report coverage separately so the system cannot make itself look accurate by refusing difficult cases.
6. **Local validation and negative controls:** run `cargo test -p symthaea-psych-bench` in the pinned development environment, then add tests for absent evidence, unsupported endpoints, corrupted manifests, split leakage, and non-reproducible configuration. Until the tests run, the new Rust code is unverified.
7. **Frozen data manifest and baseline:** select a small public ADMET subset, record the exact source versions/licenses/digests, and reproduce simple baselines before introducing Symthaea-specific reasoning.
8. **Symthaea adapter:** integrate cognitive/evidence reasoning only after standalone baselines work, then run ablations to establish whether Symthaea contributes beyond the simpler model.
6. **Independent review:** ask pharmacology/toxicology experts to adjudicate labels and high-risk cases. Keep failed results and negative controls in the report.
7. **Prospective extension:** add newly published, time-held-out evidence and then ethically approved experimental collaboration where justified.

This is designed to be executable without waiting on existing CI runners: data inventory, source/license review, schema design, split audits, deterministic baseline fixtures, and a preregistration-style evaluation plan can all start locally. No external data have been acquired and no benchmark has been run as part of this proposal.

## Claim ceiling

Passing synthetic or retrospective benchmarks establishes only performance against the tested data, labels, splits, and assumptions. It does not establish that a compound is safe, that a human experience can be predicted, that a therapeutic outcome will occur, or that a model is suitable for clinical use. Those claims require independent domain expertise and appropriate experimental, ethics, and regulatory processes.
