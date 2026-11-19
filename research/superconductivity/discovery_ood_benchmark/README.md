# SC-DISCOVERY-OOD-BENCHMARK-001

## Purpose

The objective is to move superconductivity ML from high-quality interpolation to trustworthy discovery decisions for material families and conditions outside the training distribution.

The November 2026 AutoSuperCon preprint reports 20,354 experimental-property records across 6,793 superconductors and eight properties. Its composition predictors report strong within-population metrics (R²_log = 0.83 for Tc and 0.69 for Hc2), yet the leave-one-family-out evaluation fails badly for unseen families; the cuprate Tc score is reported as R²_log = -10.2. This is the failure mode this contract is designed to make visible rather than average away. Source: https://arxiv.org/abs/2611.15627.

BEE-NET demonstrates a complementary route for electron-phonon materials: learn the Eliashberg spectral function α²F(ω), then connect ML, DFT, structural screening and synthesis. Its reported 0.87 K MAE is against DFT/Allen-Dynes calculated Tc labels, not experimentally measured Tc. The study also reports computational screening and synthesized follow-up candidates. Source: https://www.nature.com/articles/s41524-026-01964-8.

A separate 2026 critical-field workflow expands the property space beyond Tc, calculating critical-field quantities for approximately 7,300 electron-phonon superconductors and deriving coherence length, penetration depth and GL quantities. Source: https://www.nature.com/articles/s41524-026-02266-9.

## Benchmark tracks

1. **Within-family interpolation.** Grouped material-level holdouts. Useful for interpolation, but it must not be described as new-family discovery.
2. **Leave-family-out.** Exclude all records from one superconducting family from training and calibration.
3. **Prototype-out.** Hold out a crystal prototype or structural family.
4. **Source-out.** Hold out complete papers or data sources to expose source-specific leakage.
5. **Temporal-out.** Train on records published before a frozen cutoff and evaluate only on later publications.
6. **Condition shift.** v0.1 adds categorical holdouts for field orientation, sample form and transition criterion. Quantitative pressure-regime, doping and strain bins remain planned extensions; they require pre-registered binning rules and enough independent specimens in each condition.

The Rust v0.1 auditor implements group holdout by family/prototype/source and the three categorical condition axes above, plus temporal holdout, evidence-class separation, prediction coverage, selective abstention and error/interval summaries.

## Evidence strata and state identity

Every observation must retain:
- material identity and composition;
- superconducting family and crystal prototype;
- publication/source identity and date;
- specimen or batch identity where available;
- evidence class: measured, paper-reported, computed or synthetic fixture;
- property, value and unit;
- pressure, measurement temperature, applied field and orientation;
- sample form, measurement method and transition criterion (onset, midpoint, zero resistance, bulk/thermodynamic, etc.).

A computed Allen-Dynes or SCDFT Tc is not silently converted into a measured Tc. Paper-reported and independently reproduced values remain distinguishable. Sample criteria and field orientations are not interchangeable labels. The same restrictions apply to Jc, Hc2, λ, ξ and gap: report measurement conditions and method alongside the number.

## Metrics

Report metrics by property, evidence stratum and split:
- MAE and median absolute relative error in physical units;
- mean absolute log10 error and R²_log for strictly positive targets;
- prediction interval coverage **and** interval width;
- calibration error and selective risk as a function of prediction coverage;
- precision@K, recall@K and enrichment factor@K for pre-registered candidate thresholds;
- false-negative rate at relevant screening thresholds;
- wall-clock/compute/DFT/synthesis/measurement cost per independently confirmed candidate.

Do not use row bootstrap intervals when many rows are repeated measurements from the same family, specimen, paper or material. Use family- or material-system-clustered resampling and paired comparisons on exactly the same test units. Report the number of independent families and systems, not just the record count.

## Baselines and fair industry comparison

Evaluate on identical frozen data and split manifests:
- training-set mean and family-mean baselines;
- composition-only descriptors (e.g. Magpie-type) with random forest / gradient boosting;
- reproduce the published AutoSuperCon/tabular baseline where exact data, transforms, split and model implementation are available;
- structure-aware graph / equivariant models on the subset with trustworthy crystal structures;
- BEE-NET-style α²F(ω) prediction only in the electron-phonon domain where the required inputs and labels exist;
- Allen-Dynes, Eliashberg and SCDFT physics calculations only on domains where their assumptions and inputs apply;
- a hierarchical, mechanism-aware hybrid model as the Luminous research hypothesis.

Never compare a model trained against calculated Tc with one evaluated against measured Tc as though they shared the target. Do not let one model use extra test-time inputs unavailable to another; show both matched-input and privileged-input tracks.

## Paradigm shifts to pursue

### 1. Predict response surfaces, not one Tc number

Predict Tc, Hc2(T, orientation), Jc(B,T,angle), Hc1, λ tensor, ξ, gap, superfluid stiffness and depairing current as functions of material state. Carry the measurement protocol and uncertainties through the surface.

### 2. Learn mechanisms rather than composition correlations alone

Represent α²F(ω), Fermi-surface geometry, multiband coupling, spin/orbital fluctuations where relevant, superfluid stiffness, anisotropy and pair-breaking scales. Use separate experts for conventional electron-phonon, multiband, heavy-fermion, iron-based and cuprate regimes rather than forcing one universal mechanism into one scalar regressor.

### 3. Make OOD detection and abstention first-class outputs

A predictor should be allowed to say "not qualified for this family / pressure / prototype". Report its coverage-risk curve. A small MAE on the easy 30% of cases is not a win if 70% of the candidate space is silently excluded.

### 4. Close the computation-to-experiment loop

Rank experiments by expected reduction in decision uncertainty per unit cost, including synthesis feasibility and measurement capability. Preserve pre-registration, specimen chain-of-custody, failed synthesis attempts and null observations; failed experiments are useful evidence, not records to delete.

### 5. Make claims evidence-carrying

Every prediction must be traceable to source records, sample state, calculation code and version, model digest, frozen split manifest, uncertainty method, abstention decision and an independent verifier.

## Prospective targets (not achieved results)

These are proposed acceptance goals, not current results or SOTA claims:

- Reproduce published baselines under their original target definitions and splits before making any superiority claim.
- On strict family/prototype/source/temporal holdouts, aim for at least 20% lower mean absolute log10 error than the strongest matched baseline, with a paired family-bootstrap 95% interval excluding zero improvement.
- For nominal 90% prediction intervals, aim for 85–95% empirical coverage on adequately populated holdouts, while reporting interval width and subgroup coverage.
- Aim for at least 1.5× precision/enrichment at K ≤ 50 versus the strongest matched baseline at an equal compute budget.
- Aim for at least 2× fewer expensive DFT/synthesis/measurement follow-ups per independently confirmed useful candidate.
- Report the fraction abstained and the risk at each coverage point. An improvement obtained by abstaining on most candidates is not acceptable.
- For experimental Jc claims, align to the relevant measurement method and standard; Jc varies with field, temperature, orientation and sample quality.

## What is and is not implemented here

Implemented in v0.1: schema, synthetic-only fixture, family/prototype/source and temporal split checks, evidence-tier guardrails, prediction coverage/abstention checks, basic MAE/log error/interval coverage, and negative-control tests.

Not implemented yet: actual AutoSuperCon ingestion, paper/source reconciliation, measured specimen-level corpus, external baseline reproduction, calibrated confidence intervals at scale, grouped bootstrap, condition-shift holdouts, or any physical materials discovery. The fixture and its metrics are only software validation.

## Run

```sh
cargo fmt --manifest-path tools/sc-discovery-ood-benchmark-auditor/Cargo.toml --all -- --check
cargo test --manifest-path tools/sc-discovery-ood-benchmark-auditor/Cargo.toml --bin sc_discovery_ood_audit
cargo run --manifest-path tools/sc-discovery-ood-benchmark-auditor/Cargo.toml -- \
  --schema data/schemas/sc_discovery_benchmark_v0_1.schema.json \
  --fixture data/research/sc_discovery_benchmark_fixture_v0_1.json
```

## Integration with the evidence-aware discovery benchmark

This OOD contract has a deliberately narrow role: it owns the split/leakage auditor and evaluation of predictions under family, prototype, source, temporal and condition shifts. The complementary [evidence-aware discovery benchmark PR #12860](https://github.com/Luminous-Dynamics/symthaea/pull/12860) owns the broader observation/evidence contract and the canonical published-baseline registry. Reuse that registry rather than independently curating a second, potentially divergent baseline list.

The canonical registry currently includes AutoSuperCon, BEE-NET, NEMAD, Geisler et al.'s calculated high-field database, HTSC-2025, the 2026 feasibility-first hierarchical inference paper, and DataG Non-Sc. All remain `not_reproduced`; the registry explicitly states that no external data were downloaded and no model training was performed.

## Future v0.2: feasibility-first, then conditional property prediction

The August 2026 hierarchical-inference paper provides a useful architecture comparator: first estimate whether superconductivity is feasible, then estimate (T_c) conditionally. DataG Non-Sc supplies a large reported negative-class track, while AutoSuperCon supplies extracted experimental property observations. These sources must not be naively merged: each negative label needs a source, a measurement/search scope, conditions and a detection/observability criterion.

The current v0.1 OOD schema/auditor does **not** yet encode a calibrated superconducting-feasibility classifier or conditionally gated (T_c) predictions. Treat this as a separately versioned extension, not as an implemented v0.1 capability.

Required rules for that extension:

1. Labels must be tri-state at minimum: `observed_superconducting`, `observed_non_superconducting_under_conditions`, and `unknown_or_not_reported`. Absence from a superconductivity database is not a verified negative.
2. Stage one emits calibrated (P(SC\mid x, c)) and an abstention state, where (c) carries sample form, pressure, temperature range, measurement method and search coverage. A material may be superconducting under one condition and not another.
3. Stage two predicts (T_c) and other properties only conditionally on a qualified superconducting feasibility state; preserve probability of feasibility separately. End-to-end candidate ranking must account for false positives, false negatives and abstention rather than silently setting (T_c=0) for unknown labels.
4. Hold out source/lab, chemistry family, prototype and time in addition to ordinary material groups. Keep all data-derived thresholds, class weighting, calibration and feature selection within training folds.
5. Report class prevalence and independent material/family counts, macro PR-AUC, ROC-AUC, Brier score, log loss, calibration slope/intercept, recall at a fixed false-discovery rate, selective risk vs coverage, and conditional (T_c) MAE/interval coverage on confirmed positive cases.
6. Report a separate end-to-end decision metric: validated candidates per fixed compute/experiment budget, false-positive experimental burden, and cost-to-qualified-answer. Do not compare conditional (T_c) MAE to an all-material classifier score.
7. Include negative controls for database non-report treated as negative, condition-mismatched labels, family/source leakage, and selective abstention on the hardest class.

The research target is not simply a better (T_c) MAE. It is calibrated detection of *where superconductivity is evidenced*, followed by conditional property prediction, OOD abstention, and prospective confirmation. This target needs source-qualified labels before model training or SOTA comparison.
