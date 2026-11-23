# SEMI-BENCH-001 — 2026 inspection, metrology and packaging frontier

**Review date:** 2026-11-21  
**New workstreams:** [SEMI-INSPECT-001 #13083](https://github.com/Luminous-Dynamics/symthaea/issues/13083), [SEMI-MET-002 #13084](https://github.com/Luminous-Dynamics/symthaea/issues/13084), [SEMI-PKG-TEST-001 #13085](https://github.com/Luminous-Dynamics/symthaea/issues/13085)

This review extends the benchmark programme beyond classification to three higher-value decisions: what to inspect next, when an AI measurement is trustworthy, and whether complex chiplet packages remain testable after integration.

## 1. Adaptive inspection: optimize the inspection policy, not only the classifier

### Recent evidence: noisy-label active learning for wafer bin maps

Lim, Kim and Lim, *A deep active learning framework for defect classification of wafer bin maps under noisy labels*, Advanced Engineering Informatics, September 2026:
- DOI: https://doi.org/10.1016/j.aei.2026.104621
- Open access abstract/results: https://www.sciencedirect.com/science/article/pii/S1474034626003137

The proposed framework combines:
- coverage-oriented diversity sampling in a learned embedding;
- Bayesian epistemic uncertainty approximated through Monte Carlo dropout;
- loss-based Gaussian mixture filtering to identify likely noisy labels;
- adaptive feedback that marks ambiguous/noisy neighborhoods for later re-examination.

Reported results under a 20% noisy-label scenario include accuracy 0.8061 ± 0.0088 and F1 0.7991 ± 0.0094. Removing uncertainty, diversity or label filtering worsens the reported ablation results. These are paper-reported results under its dataset and protocol, not a Luminous reproduction or a live-fab result.

The important research question is not merely whether uncertainty sampling improves classification. It is whether the policy obtains more useful information per inspection, while avoiding confirmation bias from noisy and correlated labels.

### Recent evidence: latent risk in uninspected wafers

Kim and Lee, *Volume-adjusted prioritization of suspected equipment by forecasting latent defect risk in uninspected wafers under sparse and irregular semiconductor defect monitoring*, Computers & Industrial Engineering, October 2026:
- DOI: https://doi.org/10.1016/j.cie.2026.112291

The paper highlights an important operational asymmetry: many processed wafers remain unverified during the process-to-inspection interval. It models uninspected wafers as latent risks and accounts for inspection lag and throughput.

This motivates a benchmark that includes the uninspected population—not only the subset that has already received labels.

### New paradigm: value-of-information scheduling

Inspection should be a sequential constrained decision:
1. Maintain a probabilistic belief over inspected and uninspected population state.
2. Record process/inspection time, throughput, tool state and observation delay.
3. Estimate the expected impact of an unresolved defect, including the number and age of exposed uninspected wafers.
4. Select the next inspection or measurement using expected information value adjusted for cost and delay.
5. Update the belief when a label arrives, preserving uncertainty and the inspection-selection policy that created the observed data.
6. Escalate or abstain when the evidence is not sufficient.

The benchmark must compare random, systematic, risk-only, uncertainty-only, diversity-only, volume-adjusted and expected-value-of-information policies at matched inspection budgets.

Metrics should include defects found per 100 inspections, cost-weighted missed defects, time-to-detection, residual latent risk, turnaround, coverage, queue age, operator burden, and policy regret against a clairvoyant oracle on synthetic fixtures.

Crucially, inspected wafers may be a missing-not-at-random sample. A model trained naively on inspected-only data can learn inspection policy rather than process risk. Use logged selection propensity, inverse-propensity/doubly robust methods only when their assumptions hold, or explicitly report the causal/decision estimate unresolved.

## 2. Physics-bounded AI metrology: publish the model's detection limits

### NIST reference dataset and benchmark

NIST's public page, *Detection Limits of AI Segmentation Models Applied to Scanning Electron Microscopy (SEM) Image-Based Detection*:
https://pages.nist.gov/detection_limits/web/index.html

It describes:
- six simulated SEM image datasets;
- 567 images per dataset, each 512×512 pixels, with varied noise and contrast;
- three trained U-Net models of approximately 7.9 million parameters;
- 540 visualization graphs spanning image data quality, AI model performance, detection limits and mappings between AI metrics and signal-to-noise metrics;
- human and numerical observer comparisons as noise and contrast change.

The related publication is Bajcsy et al., *Detection limits of AI-based SEM dimensional metrology*, SPIE Advanced Lithography + Patterning, February 2025; NIST publication page dated March 14, 2025:
https://www.nist.gov/publications/detection-limits-ai-based-sem-dimensional-metrology

NIST's continuing project *Overlay Metrology Using Physics and AI-Based Scanning Electron Microscopy* describes physics-based Monte Carlo image simulation, comparison with traditional measurement, sparse scanning, AI-based acquisition/denoising/analysis and explicit characterization of AI detection limits:
https://www.nist.gov/programs-projects/overlay-metrology-using-physics-and-ai-based-scanning-electron-microscopy

NIST links public training and inference code:
- https://github.com/usnistgov/WIPP-unet-train-plugin
- https://github.com/usnistgov/WIPP-unet-inference-plugin

Both plugin repositories were archived on June 12, 2026. Treat as reproducibility references, not actively maintained dependencies. Public web visualizations and metadata do not automatically grant raw image/model files for redistribution. Exact artifact access, rights and digests remain an acquisition gate.

### New paradigm: detection-limit-aware engineering

A segmentation score is not a dimensional measurement. A denoised image can look better while its edge moves. A high-confidence model can produce a plausible but false feature.

The benchmark must report detection probability and false-alarm probability as functions of feature size, contrast and SNR, and establish an abstention region beyond the model's supported detection limit. Physics-based and AI observers should be compared against common reference geometry and uncertainty definitions.

Add a sparse-scan track: compare full scans with fixed and adaptive sparse acquisition, and measure total measurement time, reconstruction error, missed-feature probability and dimensional uncertainty. This aligns with NIST's current emphasis on integrating physics simulation, AI and sparse scanning for SEM overlay/CD metrology.

## 3. Advanced packaging: design for defect observability over the lifecycle

IEEE Design & Test, *Testing of Chiplet-Based Packages Based on Advanced Heterogeneous Integration*, Volume 43, Issue 5 (October 2026), DOI 10.1109/MDAT.2026.3706448:
https://doi.org/10.1109/MDAT.2026.3706448

The survey reviews test challenges in many-chiplet packages, including die shift, warpage, misalignment, residue-induced shorts/opens, coupling in dense redistribution layers and defects that span multiple chiplets. It covers IEEE 1838, UCIe-oriented test support, external test, built-in self-test, physically aware test-pattern generation, diagnosis and field monitoring.

### New paradigm: lifecycle testability graph

Co-design the package with its ability to expose defects before and after interfaces become buried. Represent each fault mechanism with:
- excitation path and test access;
- observation point and observable signature;
- minimum test time and power/thermal constraints;
- localization precision and residual ambiguity;
- pre-bond/post-bond/in-field detectability;
- containment, repair and telemetry options.

Report per-fault-family coverage and escaped-defect probability, not one aggregate coverage number. Preserve known-good-die semantics as relative to a precise test profile; passing a small scan or continuity test does not establish every function as good.

## 4. What this changes in the Luminous strategy

| Previously emphasized | New emphasis |
|---|---|
| Classify inspected wafers | Choose which wafer/site/tool to inspect next |
| Report model confidence | Establish calibrated detection limits and abstention |
| Optimize package PPA | Optimize PPA plus testability, diagnosability and lifecycle observability |
| Treat a label as ground truth | Model noisy, delayed and policy-selected labels |
| Evaluate static model accuracy | Evaluate sequential decision value under fixed resource budgets |
| Report average test coverage | Report per-defect-family escaped risk and localization ability |

These are complementary to existing canonical owners:
- Symthaea reasons, scores uncertainty and proposes decisions.
- Symtropy models acquisition/inspection scenarios, physical image formation, timing and causal counterfactuals.
- Mycelix preserves sample/wafer/lot/tool/measurement provenance and inspection-selection records.
- Sol-Atlas tracks the paper, data snapshot, standard, model and decision-source lineage.
- Human/site systems retain execution, hold and equipment authority.

## 5. Required claim boundaries

- Uninspected wafer = unknown, not pass.
- Predicted latent risk = belief, not measured defect.
- AI segmentation = image-derived output, not calibrated dimensional truth.
- A detection-limit curve is only valid for its exact model, image formation, noise/contrast regime and reference profile.
- Synthetic package-fault coverage is not real package qualification.
- Public benchmark classification gains do not establish fab yield, cost or energy improvement.

All three new workstreams begin with synthetic/public reference evidence and progress to physical measurements only after exact data rights, calibration, test authority and independent verification are established.
