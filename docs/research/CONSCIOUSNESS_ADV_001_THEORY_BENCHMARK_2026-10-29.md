# CONSCIOUSNESS-ADV-001 — Model-neutral theory prediction benchmark

**Date:** 2026-10-29  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8971  
**Related existing work:** https://github.com/Luminous-Dynamics/symthaea/issues/8238  
**Status:** protocol proposal; no neural dataset analyzed and no AI-consciousness verdict attempted.

## Executive idea

The field advances when competing theories make predictions that can fail, not when a system receives a high score from a metric that was never validated as a consciousness measure.

This proposal benchmarks theory-specific predictions about conscious perception and access using synthetic mechanisms and eligible public neural data. It treats first-person experience as the target question, but does not pretend that task accuracy, self-report, a model's architecture, or a single integrated-information measure directly reveals subjective experience.

## Empirical anchor: a model-neutral adversarial collaboration

The 2025 Nature study “Adversarial testing of global neuronal workspace and integrated information theories of consciousness” preregistered predictions from IIT and GNWT and studied 256 human participants using fMRI, MEG and intracranial EEG.

The results aligned with some predictions but challenged important claims of both theories. Posterior areas carried sustained content information, yet the predicted sustained posterior synchronization central to one IIT claim was not observed; the predicted prefrontal onset-and-offset ignition pattern from GNWT was generally absent, and findings for some PFC content predictions were limited or inconclusive. The authors emphasize the value of a theory-neutral adversarial process.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12137136/  
DOI: https://doi.org/10.1038/s41586-025-08888-1

This does not “disprove consciousness theories” wholesale. It demonstrates that their operational predictions can be challenged, and that a careful result can contain partial support, contradiction and inconclusiveness at the same time.

## Define the constructs separately

Do not collapse these into one value:

- **Phenomenal consciousness:** whether there is something it is like to experience a state; this is not directly measured by one behavioural or computational metric.
- **Conscious content:** which percept, if any, is present.
- **Access/reportability:** whether information is globally available for deliberate report or decision.
- **Attention:** selection and weighting of information.
- **Working memory:** retention and use of information over time.
- **Metacognition:** confidence or knowledge about one's own cognitive performance.
- **Arousal/wakefulness:** state-level availability of processing.
- **Computational properties:** recurrence, global broadcasting, integration, prediction error, internal state retention and perturbation response.

These constructs interact, but are not interchangeable. The benchmark should report which construct each prediction concerns.

## Initial theory families

The v1 theory-card registry should cover four families, each versioned and made precise enough to risk a failure:

1. **GNWT — global neuronal workspace.** Define a task/stimulus-specific global availability and ignition/update prediction, including neural area, time window and condition.
2. **IIT — integrated information theory.** Pin the exact mathematical formalism and version (including PyPhi config). If an exact calculation is intractable, record an Unsupported/Inconclusive outcome; do not replace Phi with spectral connectivity, a graph statistic, or a hand-written estimate and label it IIT Phi.
3. **Recurrent Processing Theory.** Define a local recurrent-feedback prediction for conscious visual content and distinguish it from later access and report.
4. **Predictive processing / active inference.** State a concrete precision-weighted inference/update mechanism and its predictions under controlled ambiguity, uncertainty or expectation manipulation.

Do not use the theory name as the model. Each registry entry binds a specific theory version, model implementation, measurement, contrast, expected outcome and failure condition.

## Theory-card schema

Each frozen theory card records:

- stable card ID and theory/version;
- target construct and operational measure;
- task, stimulus, experimental condition and control;
- exact time window / anatomical or network region / state variables;
- prediction distribution, expected direction and uncertainty;
- competing theories' predictions for the same contrast;
- primary outcome, threshold, multiplicity strategy and missing-data rule;
- support / contradiction / inconclusive regions;
- assumptions, identifiability limits and degrees of freedom;
- source/data hashes, transformation/code digest and exact analysis profile;
- independent reviewer and preregistration reference.

If the theory changes after seeing data, create a new version. Never rewrite the old theory card to make the original prediction pass.

## Synthetic mechanism suite

Start with small, interpretable dynamical networks where the generating causal structure is fully known:

- feed-forward-only sensory network;
- local recurrent visual circuit;
- global broadcast network with thresholded ignition/update;
- recurrent-plus-broadcast hybrid;
- delayed response with a specified active or activity-silent retention mechanism;
- matched models engineered to achieve similar task accuracy while differing in internal mechanism;
- lesions/perturbations selectively affecting recurrence, broadcast links, or candidate integrative interactions.

The model evaluator must predict the effect of perturbations before the fixture executes. It should not simply read the generator label. Synthetic results qualify the mechanism test and score pipeline only; they do not establish which mechanism exists in the human brain.

## Published prediction-card drafts

Candidate operationalizations of the three theory divergences described in the Nature article are recorded in [CONSCIOUSNESS_ADV_001_PREDICTION_CARDS_DRAFT_V1.md](CONSCIOUSNESS_ADV_001_PREDICTION_CARDS_DRAFT_V1.md). They remain drafts, not frozen preregistrations: the exact OSF registration could not be opened by the research-browser interface during the source audit. No card may be treated as confirmatory until the exact registration version, registered windows/regions, outcome criteria, exclusions and deviations are directly captured and hashed.

## Public neural-data replication gate

The source-discovery gate has now been audited in [CONSCIOUSNESS_ADV_001_PUBLIC_DATA_AUDIT_2026-10-29.md](CONSCIOUSNESS_ADV_001_PUBLIC_DATA_AUDIT_2026-10-29.md), with a machine-readable companion manifest in [CONSCIOUSNESS_ADV_001_DATASET_MANIFEST_V1.json](CONSCIOUSNESS_ADV_001_DATASET_MANIFEST_V1.json).

The primary article links the OSF preregistration, public raw/BIDS data bundles and XNAT, plus public modality-separated analysis and task code. The article and bundle portal state CC BY 4.0, while the portal also requires accepting its Terms of Use and GDPR requirements. This clears source discovery only: no bundle was downloaded or hashed, and no neural-data analysis has run.

A sample-accounting issue remains unresolved: the paper's analyzed per-modality counts differ from the counts in the data-release overview, despite both reporting 256 overall. Reconcile subject/session manifests, exclusions and analysis inputs before computing inferential statistics. The first resource-bounded step is to use the 9 GB fMRI BIDS sample bundle for a plumbing smoke test only; the full fMRI BIDS bundle is listed as 261 GB. A sample-bundle run is not confirmatory evidence.

For the first confirmatory run, copy one exact theory-specific prediction, contrast, observable, ROI/window, exclusion rule, multiplicity policy and support/contradiction criteria from the preregistration before opening confirmatory outputs. Prefer the consortium's modality-specific analysis code, pin its commit and environment, preserve the official baseline, and report any deviation as a separate exploratory profile. Expand to M-EEG/iEEG only after the first path produces an auditable receipt.

## DMT and visual-pattern phenomena: separate adjacent benchmark

A 2024 within-subject MRI study reports changes in peripheral V1 population receptive-field size during DMT-associated visual effects. This may support a separate research line for altered visual geometry, but it is not a test of the “Code of Reality” external-code interpretation or an accepted marker of consciousness.

Related safe evidence audit: https://github.com/Luminous-Dynamics/symthaea/issues/8970

Keep psychedelic phenomenology in its own evidence stratum. Do not pool it with ordinary conscious-perception trials unless a theory card explicitly predicts a common mechanism and the data are ethically and lawfully obtained. This repository work does not prescribe drug use or an optical exposure procedure.

## Baselines and metrics

Required baselines:
- unconditional/chance predictor;
- task-only behavioural predictor;
- simple temporal/neural feature baseline;
- each candidate theory's own frozen predictions;
- source publication's original preregistered analysis, where available.

Report these dimensions separately:
- prediction calibration and held-out log/Brier score for probabilistic predictions;
- effect estimate and uncertainty for the primary preregistered contrast;
- intervention/lesion outcome accuracy;
- cross-task and cross-dataset transfer;
- sensitivity to regions, windows, preprocessing and analysis flexibility;
- identifiability and fraction of outcomes marked inconclusive;
- model complexity, runtime and reproduction cost;
- negative-control and integrity failures.

No overall weighted “consciousness score” is permitted to hide a failed core prediction or missing evidence. A model can be strong at predicting reports while remaining silent on phenomenal experience.

## Fail-closed IIT/PyPhi dependency

Existing issue #8238 identifies that the current cross-validation scaffold cannot substantiate an IIT validation claim because the Rust reference invocation is missing, failure paths can fall back to hard-coded estimates, and formalism selection is not explicit. The first implementation milestone is to resolve or keep blocked on those issues.

Mandatory outcomes:
- missing executable, timeout, nonzero exit, malformed JSON, unsupported formalism or all-invalid runs return typed missing/error outcomes;
- no numeric fallback values;
- pin exact PyPhi version, formalism preset, causal transition model and model digest;
- label spectral connectivity as connectivity only, never as IIT Phi;
- regression tests verify failure behaviour before any theory comparison is admitted.

## Research success criteria

- At least two genuinely divergent, preregisterable predictions are specified for the initial theory pair.
- Synthetic controls correctly distinguish mechanism cases with uncertainty and negative controls reported.
- One public-data preregistered result is reproduced with exact-source provenance, or a concrete data-access/metadata blocker is recorded.
- Historical theory cards and all contradictory results remain immutable and visible.
- A held-out subject/session or independent dataset is used before any transfer claim.
- No model/architecture score is presented as an AI-consciousness certification.

## Integration boundaries

- Symthaea owns versioned theory cards, hypotheses, disconfirmation candidates and evidence ledgers.
- Symtropy owns deterministic dynamics simulation, intervention scheduling and replay receipts for synthetic models.
- Existing PyPhi issue #8238 owns IIT/PyPhi formalism correctness; do not create a competing adapter.
- Existing shared-grammar work remains about model structure, not a proxy for consciousness.
- Any bridge from computational behaviours to claims about subjective experience remains explicitly interpretive unless supported by additional theory and evidence.

## Claim ceiling

This benchmark can improve mechanistic explanations and compare theories' empirical predictions. It cannot by itself establish or rule out subjective experience in an AI. No dataset has been reanalyzed and no theory has passed or failed because of this document.

## References

- Cogitate Consortium (2025), Nature, “Adversarial testing of global neuronal workspace and integrated information theories of consciousness.” https://doi.org/10.1038/s41586-025-08888-1
- Seth & Bayne (2022), Nature Reviews Neuroscience, “Theories of consciousness.” https://doi.org/10.1038/s41583-022-00587-4
- Branco et al. (2024), NeuroImage, “Rapid effects of tryptamine psychedelics on perceptual distortions and early visual cortical population receptive fields.” https://doi.org/10.1016/j.neuroimage.2024.120718
