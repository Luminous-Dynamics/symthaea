# Neuroscience Research and Validation

**Status:** experimental research code and qualification work. This directory is not a clinically validated system and does not establish a diagnostic, prognostic, treatment-selection or consciousness biomarker.

## Evidence boundary

The repository contains several different kinds of neuroscience-related work. They must be reported separately.

| Track | Current implementation | What it establishes | What it does not establish |
|---|---|---|---|
| C. elegans topology | Embedded representative 279-node topology with deterministic pseudo-random connectivity | Reproducible software fixture for graph and HDC experiments | Empirical C. elegans connectome validation; the edges are not loaded from the published synapse table |
| Sleep EEG | Sleep-EDF example and signal-processing components | A code path and candidate analysis on EEG/EOG/EMG data | Generalizable sleep staging, diagnostic performance or clinical utility without pinned cohort, labels, splits and independent metrics |
| Meditation / task EEG | Experimental EEG feature and state-analysis code | Candidate feature computations for research | A validated measure of meditation, flow, “superconsciousness”, or any person's mental state |
| EM connectomics | Connectomics Autopilot research programme | Reconstruction, proofreading and query-assurance research for microscopy connectomes | Human in-vivo functional connectivity or clinical prediction |
| Human clinical connectomics | NeuroTwin translation plan | Proposed evaluation design and qualification gates | A currently validated patient-specific brain twin or treatment-response model |

A test that runs a function is not proof that its scientific hypothesis is true. A synthetic fixture must be reported as synthetic. A dataset path in source code does not prove the expected data files are present, licensed, correctly labelled or independently validated.

## Current research programme

### 1. Connectomics Autopilot

See [CONNECTOME_RESEARCH_PROGRAM_V2.md](./CONNECTOME_RESEARCH_PROGRAM_V2.md).

This programme addresses microscopy / EM connectomics: source provenance, neuron and synapse reconstruction uncertainty, candidate correction, query robustness across plausible reconstructions, pipeline sensitivity, scientific value of information and evidence invalidation. Results on this track must not be compared directly with clinical fMRI or diffusion-MRI prediction metrics.

### 2. C. elegans graph fixture

See [CELEGANS_VALIDATION.md](./CELEGANS_VALIDATION.md).

The current embedded Rust structure is a synthetic, deterministic topology fixture. It can be useful for testing code paths, randomization controls and graph algorithms, but not as an empirical biological connectome until a versioned observed edge table and source-data manifest are imported and independently checked.

### 3. EEG signal research

Sleep-EDF and OpenNeuro are potential data sources. Every reproduction must pin exact dataset release, subject inclusion, file digests, channel definitions, reference montage, sampling rate, scoring standard, exclusions and split assignments. Splits must be participant-level; repeated epochs from one person must not leak across train and test.

Report macro-F1, per-stage sensitivity/specificity, balanced accuracy, Cohen's kappa, confusion matrices, calibration where relevant and participant-level confidence intervals. A single overall percentage is insufficient, especially for imbalanced sleep stages. Compare against established sleep-scoring baselines under the same preprocessing and cohort.

Meditation and related EEG analyses are exploratory. Avoid assigning subjective or clinical states solely from band power or connectivity features. Any state labels require a prespecified operational definition, appropriate ground truth and external validation.

## Clinical translation: NeuroTwin

See [NEUROTWIN_CLINICAL_TRANSLATION_V0_1.md](./NEUROTWIN_CLINICAL_TRANSLATION_V0_1.md).

The proposed clinical track is human multimodal neuroimaging and electrophysiology: diffusion MRI, resting/task fMRI, EEG/MEG, longitudinal symptoms, treatment exposure and patient-relevant outcomes. Its first research question is whether such data add robust, calibrated value to clinical-only baselines for depression/anxiety trajectories and treatment-response research.

The plan distinguishes prognosis from treatment-effect estimation, requires external-site validation, prohibits leakage across participants and visits, and keeps Φ / IIT-inspired indicators outside primary clinical endpoints until independently validated for a defined construct.

## Running code checks

Commands below run repository tests; they do not automatically validate scientific hypotheses or clinical claims.

- C. elegans fixture tests: cargo test celegans
- EEG feature tests: cargo test meditation
- Clinical examples require the exact licensed/approved dataset files and their data dictionaries.

## Evidence reporting vocabulary

Use these labels in reports and benchmark output:

- **Synthetic fixture:** generated or hand-authored inputs.
- **Internal code test:** checks implementation behavior only.
- **External input, unverified:** a local file is parsed, but provenance/authenticity has not been independently established.
- **Empirical reproduction:** pinned source data and protocol reproduce a published or registered result.
- **External validation:** a locked cohort/site not used for model selection validates the result.
- **Prospective clinical validation:** a protocol-approved study evaluates patient-relevant outcomes in intended workflow.

Never silently upgrade a result from one category to another.
