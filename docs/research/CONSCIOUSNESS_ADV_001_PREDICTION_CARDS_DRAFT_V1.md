# CONSCIOUSNESS-ADV-001 — published-prediction card draft v1

**Status:** DRAFT; not preregistered, not peer-reviewed, and not a result.  
**Evidence anchor:** Cogitate et al., Nature (2025), DOI 10.1038/s41586-025-08888-1.  
**OSF preregistration:** https://osf.io/92tbg/  
**Companion data audit:** [CONSCIOUSNESS_ADV_001_PUBLIC_DATA_AUDIT_2026-10-29.md](CONSCIOUSNESS_ADV_001_PUBLIC_DATA_AUDIT_2026-10-29.md)

## Purpose and status boundary

This document converts predictions described in the published Nature article into explicit candidate card templates for an eventual reproducibility benchmark. The exact OSF preregistration must be retrieved and version-pinned before a card is marked frozen. The preregistration page could not be read in this audit environment (the open request returned HTTP 403), so the summaries below are grounded in the article's publicly accessible description, not a claim that we independently inspected all preregistered criteria.

The original study already reports results on these data. Reanalysis of the same data is a reproduction exercise, not new confirmation. Any novel theory-comparison claim requires a prospective prediction and held-out or independent data.

## Card family 1 — content decoding and task independence

**Divergence described in the article**

- IIT: conscious content should be represented most strongly in posterior cortex; PFC is not necessary, and adding PFC information should not improve decoding over posterior-only data.
- GNWT: conscious content should be available in the prefrontal/global-workspace system, including in theory-relevant conditions where task demands are carefully controlled.

**Candidate observables**

- Separately decode face-versus-object category and letter-versus-false-font category using cross-task generalization.
- Separately decode orientation (front/left/right, with the exact analysis contrast copied from the registered plan).
- Estimate posterior-only versus posterior-plus-PFC decoding performance, region by region and modality by modality.

**Minimum controls**

- Chance/unconditional baseline and a simple time-/feature-based classifier.
- Held-out subjects/sessions where feasible; do not treat correlated trials or electrodes as independent subjects.
- Separate task-relevant and task-irrelevant contrasts; retain each modality's own acquisition and preprocessing.
- Freeze ROI definitions, windows, category pairings, cross-validation folds and multiplicity correction before confirmatory outputs are inspected.

**Metrics**

Balanced accuracy or preregistered decoding score with subject-level uncertainty; difference between posterior-only and posterior-plus-PFC decoders; cross-task generalization gap; calibration if probabilistic outputs are used; fraction of missing/inconclusive cases.

## Card family 2 — maintenance of conscious content over time

**Divergence described in the article**

- IIT: posterior content-specific activity/information is sustained across the duration of the percept, with activity tracking the 0.5 s / 1.0 s / 1.5 s stimulus-duration manipulation.
- GNWT: PFC shows a brief content-specific ignition near stimulus onset (the article describes approximately 0.3–0.5 s after onset) and another event around offset, with information maintained in a latent/activity-silent state between workspace updates.

**Candidate observables**

- Duration-dependent response profiles in theory-defined posterior and PFC ROIs.
- Time-resolved decoding and cross-temporal representational similarity, with modality-specific analysis in iEEG and M-EEG.

**Minimum controls**

- Exact preregistered windowing and onset/offset alignment; include sensitivity to plausible latency shifts as separately labeled analyses.
- Match trial/subject inclusion to the source analysis; report whether a null is decisive or the data/measurement are insufficient.
- For a replication, compare against the original result and preserve deviations. For a new test, use a previously unexamined cohort/task and freeze the predictions first.

**Metrics**

Preregistered model comparison or effect estimate with uncertainty; onset/offset timing error; sustained-representation duration; held-out subject prediction score; sensitivity to latency/window choices; missingness and inconclusive-rate.

## Card family 3 — interareal connectivity

**Divergence described in the article**

- IIT: sustained short-range connectivity within posterior cortex linking low-level sensory and higher-level category-selective regions.
- GNWT: long-range connectivity between high-level category-selective regions and PFC/global-workspace areas.

**Candidate observables**

- The registered connectivity metric and exact regions, contrast, time/frequency window and task condition, copied from the OSF protocol before running.
- Compare each candidate pattern against a topology-appropriate null and the official consortium pipeline, separately by modality.

**Minimum controls**

- Do not substitute structural proximity for functional connectivity or substitute spectral connectivity for IIT Phi.
- Report spatial coverage and missing electrodes/regions. The article itself notes that sparse V1/V2 intracranial sampling limits some inferences.
- Preserve negative results and modality-specific inconclusiveness rather than merging all modalities into a winner score.

**Metrics**

Registered effect and uncertainty; model comparison/calibrated predictive score; replication deviation; robustness to ROI/threshold choices; modality-specific missing-data coverage.

## Preregistration freeze checklist

Do not label any card FROZEN until all boxes are checked:

- [ ] Retrieve and archive the exact OSF preregistration/version referenced by the paper; calculate a source digest.
- [ ] Copy the formal wording, primary and secondary outcomes, analysis windows/ROIs, exclusion and missing-data rules, and pass/fail interpretation from the preregistration.
- [ ] Reconcile the article's modality-specific analysis counts with the data-release subject/session manifests and quality-control exclusions.
- [ ] Pin the official analysis-code commit and each modality-specific environment.
- [ ] Declare the planned baseline, statistical model, multiplicity policy, uncertainty interval and stopping rule.
- [ ] Name a reviewer who did not implement the analysis.
- [ ] Mark any post-hoc analysis as exploratory and retain its separate profile/results.

## Benchmark progression

1. **Source reconstruction:** produce an auditable receipt for the exact registration, code, environment and subject/session manifest.
2. **Reproduction:** run the official analysis and compare outputs to the published result within an explicitly defined tolerance; do not report this as a new theory victory.
3. **Stress test:** perturb preprocessing and model assumptions only in a separate exploratory branch, reporting how conclusions vary.
4. **Prospective transfer:** freeze theory cards before evaluating an independent or held-out cohort/task. Cross-paradigm transfer is a separate test, not a substitute for replication.
5. **Adversarial scoring:** compare each theory card against chance/unconditional, simple neural features, the official pipeline and the competing theory's own prediction. Report calibrated predictive scores, effect uncertainty, cross-subject generalization, runtime, missingness and reproducibility separately.

## Claim ceiling

These card drafts specify candidate empirical questions. They do not establish that IIT, GNWT or another theory is wholly true or false; do not identify a scalar as a consciousness detector; and do not infer that Symthaea or any AI is phenomenally conscious from task performance or integration metrics.