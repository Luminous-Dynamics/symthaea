# Fusion Equilibrium Challenge — Zero-Shot Transfer Research Programme

**Date:** 2026-11-14  
**Status:** Benchmark programme; no trained result or tokamak-control capability claimed  
**Owner:** Symthaea research / physics inference

## Executive direction

The Fusion Equilibrium Challenge (NeurIPS 2026) published final results on 14 November 2026. It provides a public, harmonized benchmark for reconstructing a tokamak's 2-D poloidal flux map from coil currents, plasma current and Thomson-scattering profiles, without magnetic-diagnostic measurements as model inputs.

This is a strong test for Symthaea's spatiotemporal representations, scientific model selection and cross-domain generalization. The most interesting question is not whether a model can interpolate familiar DIII-D shots. It is whether a model trained on DIII-D can remain accurate, calibrated and physically coherent on the unseen MAST machine.

This programme does not assume HDC/CfC is superior. Those representations must compete against the official starter baseline and conventional spatial/temporal models under identical input, split, preprocessing and scoring contracts.

## Official benchmark and current SOTA

The official data release contains 9,121 shots and approximately 98 GB:

| Split | Shots | Role |
|---|---:|---|
| DIII-D train | 7,041 | Supervised training |
| DIII-D public test | 874 | Public test inputs; targets withheld |
| MAST public test | 1,206 | Zero-shot cross-machine transfer; no MAST training split |

Each shot contains a 65×65 EFIT poloidal-flux map over time, plus q95 and betaN targets. Inputs include coil currents, plasma current, Thomson-scattering electron temperature/density profiles, geometry and timestamps. The dataset is released under CC BY 4.0.

Official final results published 14 November 2026:

| Result | Score |
|---|---:|
| Best DIII-D composite S | 0.997330 |
| Best DIII-D flux-map R²ψ | 0.999949 |
| Best reported cross-machine MAST composite S | 0.876103 |
| Corresponding DIII-D S for that transfer entry | 0.922797 |
| Best MAST flux-map R²ψ honorable mention | 0.989460 |
| Best MAST q95/betaN scalar R² honorable mention | 0.620098 |

The official cross-machine ratio is G = S(MAST)/S(DIII-D). The organizers explicitly note that this ratio can increase when DIII-D performance falls, and therefore report a stricter DIII-D floor alongside the ranking. Symthaea must report absolute scores on both machines and must not optimize G alone.

The official composite is:

S = 0.55·R²ψ + 0.15·R²(q95,betaN) + 0.10·(1 − D_LCFS) + 0.20·Consistency

The consistency term measures agreement of seven flux-derived quantities: magnetic-axis R/Z, elongation, upper/lower triangularity, plasma volume and internal inductance. Do not optimize pixelwise R² alone.

## Primary sources

- Challenge data, splits, signal dictionary, data errata and license: https://fusion-equilibrium-challenge.sophelio.io/data/
- Official final results (published 14 November 2026): https://fusion-equilibrium-challenge.sophelio.io/results/
- Starter kit, baseline, official scorer copy, local scoring harness and data fixes: https://github.com/Sophelio/fusion-equilibrium-challenge-starter
- Dataset: https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge

## Mandatory data-integrity rules

The starter-kit maintainers document two errors in the released input data. A model result is inadmissible unless these fixes are applied and their exact code revision is recorded.

### DIII-D plasma-current time axis

The supplied plasma-current timestamp array is wrong on approximately 69% of DIII-D shots:
- 4,930 of 7,041 training shots.
- 586 of 874 public-test shots.

The current values are correct; only the associated time axis needs a per-shot correction. Approximately 31% of shots already have the correct axis. Do not blanket-shift all records. Use the challenge's per-shot fix, which is reported validated against all 9,113 DIII-D source shots.

### MAST Thomson core-channel alignment

MAST core-radius metadata has 130 entries while electron temperature and density have 131 channels. Channel zero has no radius calibration and is NaN. The starter kit provides the alignment helper. Do not silently shift or truncate channels without recording this correction.

### Temporal and split semantics

- The benchmark contains diverted plasma frames only.
- Filtering makes EFIT timestamps non-contiguous; never assume a constant timestep.
- Resample input diagnostics onto EFIT timestamps. Never interpolate target flux maps.
- Split by whole shot, never by timestep. Samples within one discharge are strongly correlated.
- Do not use withheld EFIT targets in preprocessing, normalization, cached features, feature selection or model-state construction.
- DIII-D and MAST dsep fields are not semantically equivalent and must not be treated as a shared cross-machine feature or metric.
- Preserve shot order and every requested frame when using the official submission interface.

## Research hypothesis portfolio

### H1 — Low-dimensional spatial representation

Compare the official PCA-on-flux baseline with alternative latent representations. Evaluate how many modes are needed for high-fidelity reconstruction and how latent truncation affects LCFS geometry and derived quantities.

### H2 — Irregular-time dynamics

Test whether a temporal model improves the reconstructed equilibrium sequence and cross-machine transfer. A CfC/LTC sequence model is a candidate because EFIT time spacing is irregular; do not resample targets onto a fabricated uniform grid to make the model convenient.

### H3 — Machine-conditioned but transfer-safe representation

Represent device geometry, coil layout, diagnostic chord locations and machine identity explicitly. Compare:
- no machine identifier;
- explicit geometric conditioning;
- learned machine embedding;
- geometry-conditioned shared model.

The MAST test must remain zero-shot. No fitting, normalization or hyperparameter tuning on hidden MAST targets.

### H4 — Physics and geometry consistency

Measure how well flux maps preserve the official derived quantities and LCFS geometry. Consider additional physics-residual terms only when the governing assumptions and available profiles make them identifiable. A loss term is not physics-informed merely because it contains a physics variable name.

### H5 — Uncertainty and open-world model adequacy

Estimate prediction uncertainty per shot and timestep. Test whether confidence degrades appropriately on MAST and deliberately shifted diagnostics/geometry. Use risk-coverage curves and OOD detection; high-confidence wrong predictions are failures.

### H6 — Symthaea representation ablation

Compare an HDC encoder, a CfC/LTC encoder, and a conventional neural or PCA-based representation. Keep data, splits, capacity, training budget and evaluation fixed. HDC/CfC only wins if it improves the official metrics, cross-machine transfer, calibration or computational efficiency with reproducible evidence.

## Baseline ladder

1. Exact official starter-kit baseline, pinned to a full commit.
2. PCA-on-ψ plus conventional regression.
3. Conventional 2-D CNN or encoder-decoder.
4. Conventional temporal model with irregular-time features.
5. Geometry-conditioned spatial/temporal model.
6. Symthaea HDC spatial representation.
7. Symthaea HDC + CfC/LTC sequence candidate.
8. Physics/shape-regularized variants, evaluated only as explicit ablations.
9. Reproduced leading public challenge entries where their reproduction packages are published and license-compatible.

Use the official scorer and local scoring harness, not a home-grown approximation of the composite score.

## Metrics and reporting

### Official metrics — always report separately

- Composite S.
- R²ψ.
- R²(q95,betaN).
- LCFS Hausdorff distance D_LCFS.
- Aggregate consistency and each of the seven derived quantities.
- Absolute DIII-D score.
- Absolute MAST score.
- G only as a secondary ratio, never alone.

### Additional research metrics

- Calibration error and coverage at nominal 50%, 80%, 90% and 95% intervals.
- Risk-coverage curves and OOD AUROC/AUPRC.
- Per-shot, per-machine and per-time-window error distributions.
- Irregular-time sequence continuity and error versus timestamp gap.
- Sensitivity to missing/noisy diagnostics and geometry perturbations.
- Runtime, memory, parameter count, throughput and energy/computational cost.
- Failure rate and non-finite prediction counts.

## Proposed targets — aspirational, not achieved

| Metric | Initial target |
|---|---:|
| DIII-D held-out composite S | ≥ 0.9975 |
| DIII-D flux-map R²ψ | ≥ 0.99995 |
| MAST absolute composite S | ≥ 0.90 |
| DIII-D absolute S retained during MAST transfer | ≥ 0.92 |
| 90% predictive-interval empirical coverage | 85–95% |
| Target leakage | Zero |
| Exact-run reproducibility | Full data/config/source identity and deterministic replay where the toolchain permits |

A target failure is still informative when the failure mechanism and applicability boundary are documented. Do not tune against hidden test targets or change the scorer after observing results.

## Execution plan

### P0 — Freeze and reproduce

- Pin challenge starter-kit commit, official scorer, data version, data-fix code and dataset license.
- Produce a manifest with hashes, source split, shot identity and preprocessing version.
- Reproduce the official starter on a shot-held-out DIII-D split.
- Verify the scorer's perfect and zero known-answer fixtures.

### P1 — Baseline and leakage audit

- Re-run PCA and conventional baselines.
- Confirm that every input feature is permitted.
- Add a negative control that deliberately leaks withheld EFIT targets and show the audit rejects it.
- Verify shot-level split integrity and per-shot timestamp correction.

### P2 — Symthaea candidates

- Build a shot-level representation with explicit diagnostic/geometry metadata.
- Compare conventional, HDC and CfC/LTC candidates.
- Add uncertainty and OOD heads as independently scored ablations.
- Do not assume that a higher latent dimension or a more complex model is better.

### P3 — Zero-shot MAST and failure analysis

- Evaluate the 1,206 MAST inputs without target-informed tuning.
- Report every official metric and confidence calibration.
- Stratify by machine geometry, profile availability, temporal gaps and operating-regime proxies available in the allowed inputs.
- Preserve hard failures and out-of-domain cases.

### P4 — Evidence integration

- Emit a versioned benchmark report with exact source revisions, data manifest, model configuration, score breakdowns, uncertainty, residuals and failure cases.
- After reproducibility is established, hand model/run artifacts to Symtropy's existing solver-neutral scenario receipt work (#1685).
- A scenario/replay receipt proves artifact identity and replay semantics only; it does not certify an EFIT model, a tokamak equilibrium or a physical controller.

## System ownership

- **Symthaea:** representation, temporal inference, model comparison, uncertainty, OOD and scientific reasoning.
- **Symtropy:** immutable scenario identity, deterministic replay, solver/run receipts and projection/claim boundary.
- **Official challenge scorer and EFIT:** reference evaluator and target provenance.
- **Spark:** separate materials-mediated nuclear-reaction benchmark. Do not conflate tokamak equilibrium inference with lattice reaction physics.

## Safety and claim boundary

All work is offline analysis of an openly published benchmark. No live tokamak connection, actuator commands, pulse planning, interlock bypass, operational recommendations or control authority is part of this programme. A benchmark score does not qualify a plasma controller, a plasma solver, a reactor design or a physical system.
