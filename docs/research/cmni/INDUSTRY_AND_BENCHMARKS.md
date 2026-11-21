# CMNI industry landscape, SOTA comparison, and benchmark charter

**Snapshot date:** 2026-11-20  
**Programme issue:** https://github.com/Luminous-Dynamics/symthaea/issues/13023  
**Symthaea draft:** https://github.com/Luminous-Dynamics/symthaea/pull/13024  
**Symtropy companion:** https://github.com/Luminous-Dynamics/symtropy/pull/6455

## Executive position

The strongest near-term scientific opportunity is not to claim a cold-fusion energy system. It is to build a better evidence and prediction infrastructure for condensed-matter effects on low-energy nuclear reactions.

Two problems must be kept separate:

- **Accelerator-driven solid-state D–D fusion:** incoming deuterons are supplied by an ion beam. Metal hydrides can alter measured reaction yields. This is a legitimate nuclear/materials research topic.
- **Room-temperature electrochemical LENR / excess heat:** claims of spontaneous or triggered nuclear reactions without an accelerator remain disputed and have not been established as a repeatable, independently accepted energy technology.

The 2025 Pd study and 2026 Pd/Ti hydride study provide a new quantitative benchmark for the first track; they do not establish the second track or net energy.

## 1. External state of the art

### Published experimental anchors

| Work | What it actually demonstrates | What it does not demonstrate |
|---|---|---|
| Chen et al., *Nature* (2025), https://doi.org/10.1038/s41586-025-09042-7 | 15(2)% increase in beam-driven D–D fusion rate in electrochemically deuterium-loaded palladium relative to its comparison condition | No accelerator-free cold fusion, no net energy, no power-generation demonstration |
| Karahadian et al., *Nature Communications* (2026), https://doi.org/10.1038/s41467-026-74421-1 | Reported low-energy D–D yield plateau in Pd/Ti hydrides from 0.25–6.5 keV, with deviation from a homogeneous screening model below about 2 keV | No self-sustaining reaction or useful power; the heterogeneous screening fit is phenomenological, not a unique microscopic mechanism |
| Berlinguette et al., *Nature* (2019), https://doi.org/10.1038/s41586-019-1256-6 | Multi-institution reassessment found no evidence for cold fusion, while developing useful materials-science and low-energy nuclear diagnostic approaches | Does not rule out all future condensed-matter nuclear effects; it does set a high bar for new claims |

### ARPA-E's benchmark for LENR evidence

The ARPA-E LENR Exploratory Topic is a useful external rubric because it specifies the quality of evidence expected before a nuclear-reaction claim is taken seriously. Its criteria include:

- Nuclear-product diagnostics statistically above backgrounds and relevant controls at **greater than 99.7% (3σ) confidence**.
- Multiple detectors/positions and/or pre-/post-experiment elemental or isotopic analysis, with adequate time and energy resolution.
- Explicit background and contaminant characterization.
- One comprehensive statistical framework for all measurements, including correlated uncertainties.
- Calorimetry cannot be the sole or primary diagnostic; it may be one member of a diagnostic suite.
- A complete energy-balance model covering possible sources and sinks, with calorimetric thresholds and resolution derived from the uncertainty analysis.
- Clear independent/dependent variables, controlled conditions, and hypothesis-specific experiments.
- Peer review, publication, and safety analysis.

These requirements are from ARPA-E's Exploratory Topic FOA, not a claim that any particular LENR experiment currently satisfies them.

Official sources:
- FOA technical discussion: https://arpa-e-foa.energy.gov/FileContent.aspx?FileID=68c066ad-2aa8-4bbe-bfbf-fcb73762fdda
- ARPA-E LENR workshop: https://arpa-e.energy.gov/news-and-events/events/low-energy-nuclear-reactions-workshop

## 2. Publicly visible programme landscape

The public ARPA-E portfolio shows that current work is organized around complementary capabilities, not one group claiming to have solved the field.

| Organization / project | Publicly stated capability | CMNI implication |
|---|---|---|
| University of Michigan — Ionizing Radiation Detection for Exploratory LENR Experiments | Active through 2027-01-06; neutron, gamma and ion detection, DAQ, background analysis | We should interoperate with specialist detector evidence rather than build a substitute detector |
| MIT — Neutron Emission from Laser-Stimulated Metal Hydrides | Project ran 2023-06 to 2026-07; hypothesis-driven campaign using nuclear/material diagnostics, neutrons and isotopic “ash” | Strong model for multi-messenger and pre/post material analysis |
| Texas Tech — Advanced Materials Characterization and Nuclear Product Detection | Project ran 2023-06 to 2026-07; materials characterization and nuclear-product detection capability | Materials provenance and characterization must be first-class data, not optional metadata |
| Energetics Technology Center — CATHODE | Private-company detector/electrochemistry effort, with U.S. Naval Research Laboratory, Army Cold Regions Research Lab and Naval Surface Warfare Center partners | Instrument geometry, close-proximity detection and experiment-specific detector integration are active engineering directions |

Sources:
- University of Michigan: https://arpa-e.energy.gov/programs-and-initiatives/search-all-projects/ionizing-radiation-detection-exploratory-experiments-low-energy-nuclear-reactions
- MIT: https://arpa-e.energy.gov/programs-and-initiatives/search-all-projects/neutron-emission-laser-stimulated-metal-hydrides
- Texas Tech: https://arpa-e.energy.gov/programs-and-initiatives/search-all-projects/advanced-materials-characterization-and-nuclear-product-detection-lenr
- Energetics Technology Center: https://arpa-e.energy.gov/programs-and-initiatives/search-all-projects/cathode-cathode-scintillator-detector-electrochemistry

Mainline fusion companies developing tokamak, stellarator, inertial, or magnetized-target systems are adjacent but not direct LENR competitors. Their plasma confinement, driver, fuel-cycle, and engineering-readiness metrics do not measure the same scientific claim. We should not compare a metal-hydride research programme to fusion-device net-electricity milestones as though they were the same technology category.

## 3. Where Symthaea/Symtropy can outperform the field

We do not have a basis to claim better nuclear physics, better detectors, or better experimental facilities than specialist laboratories. Our differentiation should be the evidence-computation layer that connects these capabilities.

### Proposed differentiators

1. **Evidence graph instead of paper collection.** Each claim links to the exact source, sample, run, calibration, instrument, raw artifact, analysis, uncertainty, code revision, and independent review.
2. **Counterfactual material twins.** Compare targets with controlled changes to depth profile, D occupancy, defects, strain, surface chemistry, phase, and beam-energy distribution. Identify which changes are actually distinguishable from available measurements.
3. **Spatially heterogeneous models.** Replace a single global effective screening parameter with a distribution of local environments tied to measured depth/material state. Preserve the homogeneous model as a baseline and treat fitted distributions as phenomenology until independently identified.
4. **Prospective prediction.** Freeze a model and its predictions before a new material, campaign, or energy band is revealed. Score the predictions on whole-experiment holdouts.
5. **Multi-messenger inference.** Jointly analyze proton, neutron, gamma, isotope, material, calorimetry, beam-current, and background records where measured. Model channel-specific coverage and correlated systematics; never impute a missing detector channel as an observation.
6. **Active learning for falsification.** Select the next experiment by expected separation among competing hypotheses per unit cost/time, including high-value null experiments—not by the chance of seeing a positive anomaly.
7. **Negative-result infrastructure.** Publish quantitative upper limits, detection efficiency, acceptance, background, uncertainty, and null-result scope. A well-powered null is an output, not a failed run.
8. **Reproducibility by construction.** Make exact replay from immutable inputs, hashes, pinned solvers, parameters, calibrations, and scripts a release requirement.

### Proper division of labour

- Symthaea: claim extraction, literature map, model comparison, causal hypotheses, active learning, uncertainty-aware predictions, anomaly challenge analysis.
- Symtropy: physical state coupling, solver orchestration, conservation checks, detector forward models, uncertainty propagation, and deterministic replay.
- Specialist tools: evaluated nuclear cross-section libraries, stopping/implantation solvers, DFT/MD/phase-field/materials tools, detector Monte Carlo, and accredited isotope/radiation analysis.
- Independent reviewers/laboratories: calibration, blinded controls, physical execution, replication, and evidence qualification.

Symtropy's existing rigid-body physics engine must not be represented as a first-principles nuclear or electronic-structure solver. It is the integration, state, conservation, and replay framework.

## 4. Benchmark charter

The benchmark must distinguish **reproduction**, **prediction**, **diagnostic integrity**, **physical validation**, and **operational efficiency**.

### B0 — source fidelity

- Recover source assets with legal access; preserve raw bytes and SHA-256.
- Extract the paper's stated equations, run definitions, normalization, and uncertainty.
- Match reported figures/tables within reported uncertainty before extending the model.
- Separate measured data, derived data, digitized plot estimates, and synthetic fixtures.

The 2026 Zenodo record https://zenodo.org/records/20435774 is embargoed until **2026-12-01** (state checked 2026-11-20). Until release, no project report may claim raw-data reconstruction from that record.

### B1 — physical baseline fidelity

- Implement the bare-nucleus reference and the source paper's homogeneous ALR-style screened baseline.
- Recover the homogeneous limit exactly from any heterogeneous implementation.
- Reproduce the published relative thick-target yield definition and reference-energy normalization.
- Pin units, stopping-power source, S-factor source, energy spread, target density, and uncertainty treatment.
- Do not fit or score against hand-digitized graph points as though they were original raw measurements.

### B2 — prospective generalization

Train on source-defined campaigns/materials and reserve whole campaigns, materials, and contiguous energy ranges as holdouts.

Required metrics:
- Negative log predictive density or another declared proper probabilistic score.
- Log-scale and original-scale residuals.
- 50%, 90%, and 95% prediction-interval coverage and sharpness.
- Leave-one-material-out, leave-one-campaign-out, and eventually leave-one-laboratory-out performance.
- Parameter identifiability, posterior/profile-likelihood correlations, and sensitivity to priors/nuisance assumptions.

A new heterogeneous model only earns an improvement if it beats the homogeneous baseline on preregistered held-out predictions without damaging the high-energy reference behavior or producing badly miscalibrated uncertainty.

### B3 — diagnostic and analysis integrity

- Blind synthetic injections of real-like low-count signals and artifacts: energy-calibration drift, detector efficiency changes, background excursions, beam-current errors, dead-time, saturation, threshold/PSD leakage, spectral overlap, sample swaps, clock drift and duplicated records.
- Publish sensitivity, specificity, confusion matrices, false-discovery rates, and uncertainty.
- Internal stretch target: ≤1% false discoveries on a preregistered artifact challenge at a declared sensitivity. This is our engineering target, **not** the ARPA-E nuclear-evidence criterion and not evidence of physical detection.
- For a physical nuclear-reaction claim, separately meet the ARPA-E-style >99.7% statistical confidence above backgrounds and controls, with the full diagnostic suite and stated assumptions.

### B4 — calorimetry and energy accounting

- Calibrate using traceable electrical reference heaters and blank/control conditions.
- Characterize response time, thermal gradients, drift, recombination/chemical contributions, gas/liquid flows, and all external inputs.
- Publish a complete uncertainty budget, correlated systematic terms, and detection threshold derived from uncertainty.
- Treat calorimetry as corroborating evidence, not the sole/primary evidence for a nuclear origin.
- Require predicted nuclear products and energy partition to be quantitatively consistent with any claimed nuclear heat source.

### B5 — replication and scientific transfer

- One preregistered discovery run is not a replication.
- Require independent sample preparation and a materially independent setup.
- Blind the key condition assignment and analysis where practicable.
- Release raw data, calibration, sample history, code, and analysis receipts subject to lawful restrictions.
- A null replication must be published with power/sensitivity and an upper bound.

### B6 — experiment-design efficiency

Compare Symthaea's active-learning policy against:
- uniform parameter sweeps;
- conventional one-factor-at-a-time design;
- expert-designed control matrices;
- random design under the same cost constraints.

Metric: reduction in uncertainty/model ambiguity per unit of beam time, instrument time, sample cost, and analysis compute. A policy that merely finds more positive-looking excursions is a failure.

### B7 — compute and replay

Report wall time, peak memory, CPU/GPU budget, solver call count, convergence, numerical error, and replay fidelity. Compare like-for-like against specialist baselines. Do not claim Symtropy beats DFT, MD, SRIM/TRIM, or detector Monte Carlo on physics accuracy; it must prove value as the orchestration, evidence, uncertainty and replay layer.

## 5. Proposed release gates

| Gate | Exit criteria | Status now |
|---|---|---|
| CMNI-0: programme foundation | Claim registry, schema, evidence boundaries, model/benchmark specs | Draft documentation created |
| CMNI-1: source reconstruction | Raw dataset legally available; file hashes; independent reconstruction within source uncertainty | Blocked on 2026-12-01 release for the 2026 Zenodo record |
| CMNI-2: baseline solver | Independently reviewed numerical implementation; homogeneous limit and convergence validated | Research-only Rust prototype created; Rust tests not run in current environment |
| CMNI-3: model competition | Preregistered holdouts, calibrated predictions, identifiability, artifact challenge | Specified, not implemented |
| CMNI-4: external experiment | Institutional partner, approved controls/safety, calibrated instruments, preregistration | Not started |
| CMNI-5: replication | Independent laboratory replication and open evidence bundle | Not started |
| CMNI-6: energy relevance | Reproducible net energy accounting and scaling evidence | Not applicable unless prior gates pass |

## 6. What “better than SOTA” means here

A credible outperformance claim is one of these, with an independently inspectable benchmark:

- Better held-out reaction-yield prediction than the homogeneous ALR baseline, with calibrated uncertainty and transfer to a withheld material/campaign.
- Lower false-positive rates than the baseline analysis pipeline at matched sensitivity on blinded artifact data.
- More informative experiment selection than uniform, random, one-factor-at-a-time, or expert design at matched cost.
- More complete and faster evidence replay/qualification than paper-by-paper manual workflows, without losing raw provenance or uncertainty.
- Better reproducibility across laboratories, including successful reproduction of null results.

A lower training error, prettier heatmap, more complex model, more AI-generated hypotheses, or a positive calorimetric anomaly is not a SOTA result.

## 7. Immediate next work

1. Finish scientific review of the two draft PRs; execute the Rust tests and independent numerical cross-check in a suitable Rust environment.
2. Add source-specific benchmark fixtures only from openly accessible data, clearly labelling digitized points and reported values.
3. Prepare a retrieval checklist for the 2026 Zenodo files; ingest after 2026-12-01 if public, then freeze checksums.
4. Add the 2025 paper's available supporting data as a separate source and reproduce its stated comparison first.
5. Implement the first blinded diagnostic artifact dataset before training Symthaea to identify anomalies.
6. Seek collaboration with materials-science, low-energy nuclear physics, detector, calorimetry, and isotope-analysis teams. Physical experiments require qualified institutional facilities and safety oversight.

## References

- Chen et al. (2025), https://doi.org/10.1038/s41586-025-09042-7
- Karahadian et al. (2026), https://doi.org/10.1038/s41467-026-74421-1
- Berlinguette et al. (2019), https://doi.org/10.1038/s41586-019-1256-6
- ARPA-E LENR Exploratory Topic FOA, https://arpa-e-foa.energy.gov/FileContent.aspx?FileID=68c066ad-2aa8-4bbe-bfbf-fcb73762fdda
- ARPA-E 2021 LENR workshop, https://arpa-e.energy.gov/news-and-events/events/low-energy-nuclear-reactions-workshop
- ARPA-E University of Michigan detector project, https://arpa-e.energy.gov/programs-and-initiatives/search-all-projects/ionizing-radiation-detection-exploratory-experiments-low-energy-nuclear-reactions
- ARPA-E MIT project, https://arpa-e.energy.gov/programs-and-initiatives/search-all-projects/neutron-emission-laser-stimulated-metal-hydrides
- ARPA-E Texas Tech project, https://arpa-e.energy.gov/programs-and-initiatives/search-all-projects/advanced-materials-characterization-and-nuclear-product-detection-lenr
- ARPA-E Energetics Technology Center CATHODE project, https://arpa-e.energy.gov/programs-and-initiatives/search-all-projects/cathode-cathode-scintillator-detector-electrochemistry
- 2026 dataset record (embargoed until 2026-12-01), https://zenodo.org/records/20435774


## 8. Source-specific 2025 detector benchmark and external comparison

The 2025 Thunderbird publication provides a substantial source-grounded detector reference:
- One EJ-309 liquid-scintillator detector with a PMT and 14-bit / 500 MHz digitizer.
- Tail/total PSD with per-light-output gamma fits, a `mu_gamma + 5*sigma_gamma` lower boundary, 0.2 PSD-unit upper extension and 50-keVee threshold.
- MCNP6.2 + DRiFT and Geant4 response modelling; GRAVEL unfolding against a Geant4 response matrix.
- The article reports >99.9999% gamma rejection for its stated cut and background of 0.21(9) n/s.
- The SI reports 7-target beam-loading repeatability at about 69.3(2) n/s after 120 min at 0.2 mA; the 0.5 mA three-campaign electrochemical comparisons are separate.
- Controls include sheath-voltage OFF returning the signal to background and H2O-for-D2O yielding an 88.3(3)% reduction.
- D/Pd during the actual fusion experiments was not directly measured; ~0.7 was inferred from independent XRD characterization.

This is a valuable reproducible benchmark, but it is not multi-detector independent replication. The highest-value CMNI engineering opportunity is to connect detector response, sample history, beam composition, fuel-state measurements and source time-series into one uncertainty-aware evidence graph.

### How we should exceed the published workflow

Do not claim a higher gamma-rejection figure based on a simulated gate. Instead, target:
1. Lower blinded false-neutron rates on gamma-only controls at matched neutron detection sensitivity.
2. Better calibrated count-rate and spectrum uncertainty under threshold/gain/PSD drift.
3. Cross-validation between independently calibrated detector technologies or geometries.
4. Source-to-result replay from raw waveform (when available), calibration, background, response matrices, unfolding code and versioned parameters.
5. Whole-campaign prediction of the seven-target beam-repeatability and H2O/D2O controls, with no mixing of their distinct operating conditions.
6. Joint inference of target state and nuclear yield that honestly reports the current D/Pd measurement gap.

The source archive is a 14.7 GB file whose internal manifest remains uninspected, and the article's DOI citation differs from the title-matched visible Zenodo v2 record. No raw-waveform reproduction or exact detector-chain replay should be claimed until that discrepancy is resolved and source artifacts are ingested.



## 9. New statistical and spatiotemporal SOTA benchmarks

Add two benchmark families before claiming a new model is better than the published homogeneous/phenomenological fits.

### Shared-reference and correlated uncertainty benchmark
- Generate counts with one shared 6.5-keV reference denominator and correlated beam-charge/loading/systematic perturbations.
- Compare diagonal weighted log-space fitting against generalized least squares with the full covariance.
- Measure parameter bias, confidence-interval coverage, false model preference and residual calibration.
- The diagonal implementation must be retained as a source-faithful baseline; covariance-aware inference is an additional benchmark until the original analysis treatment is inspected.
- Target: nominal 90% and 95% intervals should achieve their nominal coverage within Monte Carlo uncertainty on blinded synthetic data.

### Spatial-beam/material-state benchmark
- Generate counterfactual cases with identical integrated charge but different 2-D spot profiles, same beam spot but different depth profiles, and different D+/D2+ spectra.
- Compare bulk-average, 1-D depth-only and space-time target-state models at equal input information/compute budget.
- Hold out complete spot profiles, irradiation campaigns and material states.
- Metrics: held-out log predictive score, calibration, yield residual structure, mass-balance residual, spatial profile reconstruction, and runtime.
- Target: the richer model earns credit only when its additional measured state variables improve held-out prediction; complexity alone is not a win.

These benchmarks are proposed research targets, not demonstrated SOTA advantages.


## 10. Multi-modal measurement observability benchmark

The next SOTA comparison should score not just the latent-profile fit, but whether the available instruments can actually resolve the proposed feature.

Generate blinded latent profiles with near-surface layers of 4, 10, 30, 70, 150 and 500 nm; apply source-grounded or explicitly synthetic instrument-response kernels for:
- SIMS, with sputter mixing, matrix and isotope cross-talk;
- D-specific NRA and separate ¹H-specific NRA;
- ERDA;
- variable-energy slow-PAS using energy-dependent Makhov implantation kernels;
- ZEM effective defect-sensitive response;
- lateral beam/detector point-spread as a separate axis.

Compare:
1. naive pointwise interpretation;
2. one-modality response-aware inversion;
3. joint multi-modal inversion with explicit covariance and modality-specific latent observables.

Score feature detectability, layer thickness/amplitude bias, false-feature rate, credible-interval coverage, response-kernel mismatch sensitivity, effective rank/condition number and leave-one-modality-out reconstruction. The method must abstain when a feature is not identifiable at the supplied resolution.

A richer model earns credit only if it improves held-out profile recovery and uncertainty calibration at matched measured information. It must not claim nanometre-scale resolution from a modality whose response kernel does not support it.

Relevant capability sources:
- SIMS depth profiling: https://www.nlr.gov/materials-science/dynamic-secondary-ion
- ¹H NRA and depth response: https://pmc.ncbi.nlm.nih.gov/articles/PMC4841331/
- D(³He,p)α NRA / overlayer broadening: https://www.sciencedirect.com/science/article/pii/S0168583X13002231
- Slow-positron Makhov depth response: https://pmc.ncbi.nlm.nih.gov/articles/PMC9417578/
- Pd ZEM effective defect profiling: https://doi.org/10.1021/acsnano.4c16841
