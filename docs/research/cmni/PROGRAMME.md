# CMNI: Condensed-Matter Nuclear Intelligence

**Status:** research programme proposal, v0  
**Owner:** Symthaea research / evidence qualification  
**Tracking:** https://github.com/Luminous-Dynamics/symthaea/issues/13023  
**Companion:** https://github.com/Luminous-Dynamics/symtropy/issues/6453

## 1. Mission

Build an evidence-gated scientific workflow for investigating how condensed-matter environments influence low-energy nuclear reaction (LENR) probabilities. The target is a reproducible chain:

`claim → source → hypothesis → quantitative prediction → discriminating experiment → observation → independent qualification`

The programme investigates the physics. It does **not** assume that room-temperature cold fusion, anomalous nuclear reactions, excess heat, or useful net energy exist.

## 2. Separate two research questions

These tracks must never be conflated.

### Track A — accelerator-driven D–D fusion in solids

This is established nuclear fusion under externally supplied deuteron-beam excitation, with metal targets/hydrides affecting measured yields.

Anchor observations:

- Chen et al. reported a 15(2)% D–D fusion-rate increase when palladium was electrochemically deuterium-loaded during deuteron bombardment (*Nature*, 2025): https://doi.org/10.1038/s41586-025-09042-7
- Karahadian et al. reported a low-energy yield plateau in Pd/Ti targets over 0.25–6.5 keV, with a departure from a constant-screening model below about 2 keV (*Nature Communications*, 2026): https://doi.org/10.1038/s41467-026-74421-1
- The 2026 paper explicitly states that current condensed-matter fusion rates are not relevant to energy production. Its heterogeneous interpretation is phenomenological, not a uniquely established microscopic mechanism.

Critical data-access state, checked 2026-11-20:

- Dataset record: https://zenodo.org/records/20435774
- DOI: `10.5281/zenodo.20435774`
- The record marks the files embargoed until **2026-12-01**. Until the release, do not claim raw-data reconstruction or independent reproduction. Use the published methods, reported tables/figures, and clearly labelled synthetic fixtures only.
- In the 2026 paper, proton data cover the lower-energy range; neutron measurements cover the higher-energy range. The low-energy plateau must not be described as simultaneously validated by neutron and proton channels at every point.

### Track B — room-temperature electrochemical LENR / excess-heat claims

This is a separate, unverified claim class. The 2019 multi-institution reassessment reported no evidence for cold fusion while identifying useful materials and low-energy nuclear science: https://doi.org/10.1038/s41586-019-1256-6

A heat anomaly alone is not evidence of a nuclear process. Any future claim must join traceable calorimetry, full energy accounting, independently measured nuclear products/isotopes where predicted, and blinded replication. Track B receives no evidential credit from Track A's beam-driven fusion observations.

## 3. Claim taxonomy and competing models

Every statement in the evidence ledger must be typed as one of:

- **Measured:** directly supported by a source measurement with documented instrument and uncertainty.
- **Reproduced:** independently regenerated from source data and a versioned procedure.
- **Interpreted:** a source author's explanation, not itself a direct observation.
- **Hypothesis:** a falsifiable proposed mechanism.
- **Externally replicated:** independently reproduced by another group using a materially independent setup.
- **Retracted / superseded:** retained in history, excluded from current evidence.

The first model ladder is:

1. **Bare-nucleus baseline:** standard D–D cross-section reference (e.g. Bosch–Hale parameterization where applicable), with source, units, and validity domain pinned.
2. **Homogeneous screening:** constant effective screening potential in an ALR-style screened-tunnelling model.
3. **Heterogeneous screening:** depth-/state-weighted distribution of local effective environments. A fitted effective potential is a descriptive parameter, not proof of electron-screening causality.
4. **Defect/trap and transport model:** local deuterium occupancy, defect/trap population, diffusion, implantation, replenishment, and desorption influence the target density/environment.
5. **Nuisance/instrument model:** energy spread, stopping-power uncertainty, charge normalization, detector efficiency/resolution, dead time, threshold/PSD selection, background, calibration drift, and sample-history confounds.
6. **Speculative mechanisms:** resonant or otherwise non-standard mechanisms remain dormant until they make a quantitatively discriminating prediction that survives the conventional nuisance and materials models.

No model is promoted because it fits the training curve. Models must be identifiable, predict held-out conditions, and survive ablation and sensitivity tests.

## 4. Data and provenance contract

Preserve raw source assets unchanged. Store extracted/cleaned/derived data as separate artifacts with a transformation receipt.

Minimum run-level fields:

- Stable experiment, campaign, sample, and run identifiers.
- Publication/dataset DOI, version, access date, licence/access state, file URI, and SHA-256 digest.
- Material identity, purity, geometry, thickness, phase, surface treatment, vendor/batch, preparation/anneal history, and pre/post characterization.
- Isotope, loading state, composition/occupancy measurement, loading protocol metadata, and time history.
- Beam species, nominal and measured energy distribution, center-of-mass conversion, current, integrated charge, spot/geometry, and timing.
- Detector type/channel, geometry, calibration, efficiency, energy resolution, thresholds, selection/PSD procedure, live time/dead time, and coverage by energy.
- Raw spectra/counts, background intervals, background model/subtraction, and the exact analysis pipeline.
- Statistical uncertainty, correlated systematic uncertainty, covariance information, and missing/unknown fields.
- Code revision, parameter set, solver/tool versions, random seeds where relevant, and replay receipt.

Missing metadata is represented as unknown, never filled by inference. Synthetic or plot-digitized observations must carry explicit provenance labels and must never be silently mixed with source measurements.

## 5. Evaluation design

### Reproduction before discovery

First reproduce the source's definitions and reported results. For the 2026 reference, record the relative thick-target yield normalization, reference energy, detector-channel coverage, fitting region, parameter constraints, energy spread, efficiency corrections, and uncertainty assumptions.

Do not substitute an easier quantity (e.g. raw counts, fitted (U_e), or a synthetic curve) for the reported measurement.

### Predictive holdouts

- Hold out entire campaigns, samples, materials, and/or laboratories.
- Hold out contiguous energy windows when the intended claim is interpolation/extrapolation across energy.
- Never randomly split adjacent points from a single run across train and test.
- Freeze preprocessing, model classes, fitting code, hyperparameters, thresholds, and scoring before revealing holdouts.
- Report both log-space and original-scale residuals, calibration, prediction intervals, and residual structure.

### Required scorecard

- Reproduction error against reported values and source uncertainty.
- Held-out negative log predictive density / proper predictive score.
- Prediction-interval coverage (50%, 90%, 95%) and sharpness.
- Leave-one-material/campaign/lab-out degradation.
- Parameter identifiability and correlation; confidence/profile likelihood or posterior diagnostics.
- Sensitivity to stopping power, beam energy spread, and nuisance assumptions.
- False-positive and missed-detection rates on blinded artefact controls.
- Runtime, memory, and experimental cost per unit information gain.
- End-to-end replay success from immutable inputs and pinned code.

### Blinded artifact challenge

Construct synthetic datasets with known generating truth and injected faults:

- Background shifts and time-correlated drift.
- Calibration offset/nonlinearity and detector resolution changes.
- Beam-current/charge normalization errors.
- Dead-time and saturation errors.
- Spectral overlap, threshold changes, selection/PSD leakage.
- Missing runs, duplicated records, timestamp skew, and sample swaps.
- Realistic low-count Poisson observations and correlated systematic errors.

Report false-positive rate at fixed sensitivity; publish confusion matrices and uncertainty. A proposed initial target is ≤1% false discovery on the preregistered blinded artifact suite, provided the sensitivity and confidence interval are also reported. This is a programme target, not a claim about current performance.

## 6. Paradigm shifts

1. **From a single screening scalar to a spatially heterogeneous state:** link reaction-yield variation to measurable depth, defects, local occupancy, transport, phase, and surface state.
2. **From retrospective curve fitting to prospective prediction:** a mechanism earns confidence by predicting a new material/condition before the data are revealed.
3. **From positive-result hunting to information gain:** select the experiment that best separates competing hypotheses, including experiments expected to produce null results.
4. **From one instrument to an observation graph:** link calorimetry, particle spectra, isotope analysis, structural characterization, beam diagnostics, calibration and background records.
5. **From papers-as-authority to evidence-as-data:** machine-readable claims, provenance, uncertainty, negative results, and independent replay.
6. **From opaque AI explanations to falsifiable scientific artifacts:** every Symthaea proposal must include equations/units, assumptions, predicted observables, alternative explanations, falsifiers, and the minimum discriminating dataset.

## 7. Work sequence

### CMNI-0 — evidence foundation (start immediately)

- Build the bibliography/claim ledger for 2019, 2025, and 2026 anchor studies.
- Record the 2026 Zenodo embargo and reopen the data-ingestion task on/after 2026-12-01.
- Extract the published model equations, fitting ranges, detector coverage, uncertainty definitions, and material metadata.
- Write unit-checked reference calculations and synthetic fixtures before touching embargoed data.

### CMNI-1 — baseline reconstruction

- When the source files are public, freeze their hashes and metadata.
- Regenerate published processed curves independently.
- Reconcile all deviations before model extension.
- Do not claim source reproduction until every included artifact is accessible and the output comparison is recorded.

### CMNI-2 — model comparison and prospective predictions

- Compare bare, constant-screening, heterogeneous, defect/transport, and nuisance-aware models.
- Run material/campaign holdouts and ablations.
- Produce preregistered predictions for a future independent dataset.

### CMNI-3 — experiment design and independent validation

- Recommend high-information experiments with explicit predictions, controls, and falsifiers.
- Require qualified laboratory partners for ion-beam, radiation, electrochemical, and calorimetric work.
- Keep Symthaea advisory: it may design and audit plans, but it does not directly actuate physical equipment.
- Qualify any room-temperature LENR/excess-heat claim only under a separate protocol and independent replication.

## 8. Symthaea implementation boundary

Symthaea owns literature intelligence, hypothesis graphs, causal/model comparison, active-learning proposals, data lineage, and evidence receipts. It must call specialist physics/materials solvers through versioned interfaces rather than claim that HDC/CfC itself computes quantum nuclear tunnelling or first-principles materials properties.

The independent qualification path must be able to reject Symthaea's proposal. A model's confidence, internal coherence, Phi, or reward signal is never physical evidence.

## 9. Anchor references

- Berlinguette et al., “Revisiting the cold case of cold fusion,” *Nature* (2019): https://doi.org/10.1038/s41586-019-1256-6
- Chen et al., “Electrochemical loading enhances deuterium fusion rates in a metal target,” *Nature* (2025): https://doi.org/10.1038/s41586-025-09042-7
- Karahadian et al., “Enhanced nuclear fusion in the sub-keV energy regime,” *Nature Communications* (2026): https://doi.org/10.1038/s41467-026-74421-1
- Processed data record (embargoed until 2026-12-01): https://zenodo.org/records/20435774
- ARPA-E LENR exploratory project descriptions: https://arpa-e.energy.gov/sites/default/files/2025-01/Project%20Descriptions_LENR.pdf

---

**Status discipline:** proposed ≠ implemented; synthetic validation ≠ source-data reproduction; source reproduction ≠ independent replication; accelerator-driven D–D fusion ≠ room-temperature cold fusion; excess heat ≠ nuclear origin; measured heat ≠ net energy.
