# CETB claim-specific evidence qualification ladder V0

**Status:** Proposed internal qualification framework; not an existing regulatory standard.  
**Date:** 2026-11-21  
**Scope:** Each empirical claim, endpoint, modality and deployment context is qualified separately.

## 1. Purpose

CETB needs a way to prevent evidence from being transferred across unlike claims. A good synthetic mechanism result does not establish an empirical biomarker. A published-analysis reproduction does not establish unseen-participant generalization. A subject-held-out model does not establish site transportability. A prospective research result does not establish clinical utility.

The evidence ladder assigns a qualification tier to a specific claim, not to a model or repository globally.

A qualification record must state:
- the exact claim and operational endpoint;
- modality, task, population, device and setting;
- the dataset(s), cohort(s), release versions and participant flow;
- whether the outcome was visible during development;
- the split and independent unit;
- reference implementation and source revisions;
- statistical analysis and uncertainty;
- negative controls, failure cases and abstentions;
- provenance and independent replay evidence;
- the highest tier passed and the unresolved gates.

Do not average tiers into a scalar grade. The ladder is conjunctive: a claim advances only when every required gate for that tier is met.

## 2. Qualification tiers

### Q0 — Synthetic implementation evidence

**Question:** Does the declared implementation behave as specified on known-answer and adversarial synthetic cases?

Requirements:
- deterministic known-answer fixtures;
- invalid-input and negative-control tests;
- explicit synthetic-only status;
- no empirical or clinical inference from the result.

Permitted claim: the implementation passes specified synthetic tests.

Not permitted: biological validity, benchmark performance, consciousness detection or clinical readiness.

### Q1 — Published-method reproduction

**Question:** Can an independent run replay a published method on the exact source data and analysis protocol?

Requirements:
- exact dataset DOI/version and authorized access terms;
- frozen file manifest and checksums;
- participant flow reconciled to publication/supplement;
- exact preprocessing, estimator, exclusions, split and statistical method;
- reference-code revision and environment pinned;
- numerical tolerance declared before replay;
- complete failures and exclusions reported;
- independent clean-environment replay.

Permitted claim: the declared published analysis was reproduced within the stated tolerance on the named dataset.

Not permitted: independent replication, new generalization or clinical validity.

### Q2 — Held-out participant evaluation

**Question:** Does a frozen prediction procedure generalize to participants not used for feature selection or tuning?

Requirements:
- participant is the outer independent unit;
- all sessions, runs, trials and segments from a participant remain in one fold;
- all preprocessing, feature selection, estimator selection, calibration and tuning occur inside training folds;
- split assignments are immutable and machine-checked;
- outer-fold predictions are retained;
- participant-clustered uncertainty, calibration and negative controls are reported;
- model-selection history and candidate multiplicity are disclosed.

Permitted claim: held-out participant performance within the named dataset, task and modality.

Not permitted: cross-site, cross-device or independent-cohort generalization unless those are separately tested.

### Q3 — Independent cohort / site / device replication

**Question:** Does the claim survive a genuinely independent source of participants or acquisition?

Requirements:
- a different cohort and preferably a different site/device or independent recruiting programme;
- source-data lineage shown to be independent rather than a mirror, derivative or secondary analysis of the original sample;
- no adaptation to the confirmation outcomes;
- frozen model, features, preprocessing and score convention;
- independent analysts and clean replay;
- same operational endpoint and matched cohort/eligibility definitions;
- uncertainty and subgroup/site performance reported.

A second paper on the same participants, a different derivative of the same raw data, or a new split of the same participants does not satisfy Q3.

Permitted claim: independently replicated performance for the named endpoint and acquisition domain.

Not permitted: unrestricted transport to other populations, devices or clinical uses.

### Q4 — Prospective preregistered multi-site evaluation

**Question:** Does the frozen method perform as expected on future data collected after the protocol and model have been locked?

Requirements:
- prospective protocol and prediction cards registered before confirmatory outcomes are collected/accessed;
- independent site participation and governance;
- fixed eligibility, endpoint, stopping, missingness and exclusion rules;
- predeclared statistical and multiplicity plan;
- continuous logging of failures, abstentions and protocol deviations;
- external adjudication or evaluator independence;
- calibrated uncertainty and subgroup/domain performance.

Permitted claim: prospective performance for the protocol, population and sites studied.

Not permitted: clinical utility or deployment readiness without the additional intended-use evidence.

### Q5 — Clinical utility / intended-use qualification

**Question:** Does the method improve decisions and outcomes in the actual clinical context for which it is intended?

Requirements:
- clinically meaningful intended use and comparator defined;
- prospective multicentre evaluation in the target population;
- decision-utility and patient-relevant outcomes, not accuracy alone;
- harms from false positives, false negatives and abstention quantified;
- workflow, operator burden, calibration, drift and failure recovery tested;
- ethics, regulatory and quality-system requirements addressed for the jurisdiction and intended use;
- independent replication and post-deployment monitoring plan.

Permitted claim: only the specific clinical utility and intended-use performance supported by the evidence and applicable authorization.

This tier is not a general claim that the system detects consciousness. A clinical device claim requires an explicitly defined endpoint and regulatory path.

## 3. Cross-cutting gates

These apply at every tier:

1. **Construct integrity:** report, memory, task performance, responsiveness, physiology and latent phenomenal experience are not interchangeable.
2. **Data integrity:** dataset version, release state, rights, file digests and participant flow are explicit.
3. **Statistical-unit integrity:** repeated measurements do not become independent participants.
4. **Selection integrity:** feature/model/estimator selection is nested or otherwise separated from confirmation.
5. **Measurement integrity:** within-condition repeatability and data-quality dependence are quantified where relevant.
6. **Negative controls:** nuisance-only, shuffled-label, timing/artifact and other applicable controls are reported.
7. **Claim ceilings:** absent, ambiguous, failed or unsupported evidence cannot be silently promoted to a negative or positive consciousness label.
8. **Replay integrity:** exact command, code revision, environment, configuration and output digests accompany confirmatory results.

## 4. CETB's own initial status

For the current research branch:
- RCAF kernel implementation and synthetic test contracts: proposed Q0.
- Published ds006623 and COGITATE work: Q1 targets only; no published analysis has yet been reproduced by this branch.
- COGITATE fMRI/MEG cohort analyses: Q2/Q3 protocols are drafted; no signal data have been downloaded or analysed here.
- Independent multi-site prospective validation: not started.
- Clinical intended-use evaluation: not started.

A protocol or test suite does not earn the tier of the future experiment it describes. The highest achieved tier remains the highest tier with actual evidence.

## 5. Relation to SOTA and industry

This is a benchmark-governance differentiator, not an accuracy metric. It should interoperate with BIDS, OpenNeuro/NEMAR, fMRIPrep/XCP-D and established EEG/MEG tools. It borrows the adversarial prediction discipline demonstrated by COGITATE and the repeatability / quality concerns raised by recent precision-fMRI and connectomics work.

A better-than-SOTA claim must be tied to a specific tier and matched endpoint. A higher Q-tier claim may be more valuable than a small within-cohort accuracy increase, but it is not a substitute for matched predictive performance.

## 6. Sources and status

This proposed ladder is informed by:
- Melloni et al. (2025), COGITATE, Nature, DOI 10.1038/s41586-025-08888-1.
- Bryant & Whyte (2026), DOI 10.1093/nc/niag029; correction DOI 10.1093/nc/niag071.
- Lynch et al. (2026), precision-fMRI Network Similarity Index, DOI 10.1016/j.neuron.2026.05.020.
- Ziemann et al. (2026), clinical translation of TMS–EEG PCI, DOI 10.1016/j.clinph.2025.2111487.

The tier names and gates are proposed CETB policy, not externally ratified standards or medical-device certifications.
