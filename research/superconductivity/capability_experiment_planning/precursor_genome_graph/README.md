# SC-PRECURSOR-GENOME-GRAPH-0.1

**Status:** pinned public-dataset audit and cohort-reconciliation benchmark; not a new onset predictor.  
**Research issue:** [#13025](https://github.com/Luminous-Dynamics/symthaea/issues/13025)  
**Onset likelihood evaluator:** [PR #13013](https://github.com/Luminous-Dynamics/symthaea/pull/13013)

## Pinned source and attribution

This benchmark reads the public [Precursor Genome](https://github.com/lauren-walters/precursor-genome) ledger without copying its 31 MB JSON into this repository.

- Source commit: `78cb026aa6c9b53aa0ec26ea5611d523bc3c320a`
- Ledger Git blob SHA-1: `a8ef03cee29670b22ed04b9cebb1686fe26e1acd`
- Dataset DOI: [10.5281/zenodo.21285545](https://doi.org/10.5281/zenodo.21285545)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Attribution: Lauren Walters, Matthew J. McDermott, Bernardus Rendy, Yuxing Fei, and Gerbrand Ceder, *Precursor Genome* (2026), Lawrence Berkeley National Laboratory. Changes made here: a Rust audit, graph statistics and evidence-cohort reconciliation. This project does not imply endorsement by the dataset authors.

## Why the graph matters

The dataset contains 1,035 unique unordered pairs among 46 precursors. Since `46 × 45 / 2 = 1,035`, the experiment design is the complete undirected graph (K_{46}): every precursor has exactly 45 incident pair experiments, and there are no repeated unordered pairs in this cohort.

This is not a random sample of isolated reactions. It permits pair-interaction and per-precursor analyses, but it creates an important validation hazard: a random pair split leaks both precursor identities across train and test. More credible generalization tests should include:

- 46-fold leave-one-precursor-out: hold out all 45 edges incident to one precursor.
- Chemistry-family-held-out splits, with families declared independently of observed labels.
- Double-node-held-out tests for truly unseen precursor combinations.
- Random pair splits only as an easier diagnostic, not the main generalization claim.

## Exact cohort reconciliation

The raw ledger categories count:

| Raw ledger category | Records |
|---|---:|
| Transformed | 384 |
| Unreacted | 369 |
| Completely reacted | 143 |
| Partially reacted | 113 |
| Unclassified / no reaction category | 26 |
| **Total** | **1,035** |

The paper reports 1,000 analyzable powder/XRD outcomes. The reconciliation is:

1. **26 physical handling failures:** samples not recovered as a characterizable powder. These are unknown outcomes, not negative reactions.
2. **9 poor XRD refinements:** active refinement human quality score 3. These are excluded from the evaluation cohort, not negative reactions.
3. **1,000 analyzable samples:** the remaining categorized outcomes.

After those exclusions, the benchmark cohort is:

| Analyzable category | Count | Cross-precursor-product endpoint |
|---|---:|---|
| Transformed | 380 | Negative subtype: precursor-only oxidation/reduction/decomposition |
| Unreacted | 367 | Negative subtype: original precursors recovered |
| Completely reacted | 143 | Positive: product includes ions from both precursors |
| Partially reacted | 110 | Positive: at least one product includes ions from both precursors |
| **Total** | **1,000** | |

The binary endpoint is therefore 253 positive (25.3%) and 747 negative (74.7%). The label mapping follows Walters et al.: complete and partial are reactive; unreacted and transformed are non-reactive *for the cross-precursor-product endpoint*. Transformed does not mean chemically inert.

The auditor applies exclusions in this order: physical failure first, then quality score 3, then a known category. It fails closed if residual unknown outcomes remain or if the counts diverge from the pinned paper cohort.

## Thermal metadata: do not substitute melting point for the experiment

The paper defines the reference temperature as the first thermally accessible transition for each precursor. Depending on chemistry this may be melting, decomposition, reduction, sublimation or a phase transformation. It also defines a decomposition-product proxy melting point for certain model features. These are not interchangeable with a raw `melting_point_c` field.

The reported dwell setpoint is calculated by rounding the Tammann estimate down to 100 °C increments and clipping it to 200–1,100 °C. The audit preserves `synthesis.heating_temperature` as recorded, reports its missingness and histogram, and never recomputes a missing value from `melting_point_c`. It does not claim that the recorded setpoint alone is the specimen's complete thermal history.

## Competitive benchmark ladder

### Reproduction before beating SOTA

Walters et al. (arXiv:2611.23527, posted 2026-11-17) report a three-feature log-normal AFT model with 88 °C MAE on 26 held-out in-situ onset measurements, against 131 °C for Tammann's rule. The 32-feature AFT reports 84 °C MAE but a larger RMSE (122 °C vs 101 °C) and no demonstrated paired improvement over the three-feature model. The models are trained on 1,000 Precursor Genome plus literature data for 1,866 experiments, not on the 1,000 in-house records alone.

The next valid comparison is an exact reproduction of the same PG + PG-LIT cohort, preprocessing, 26-pair holdout and score definitions. The source ledger alone is not sufficient to reproduce the published AFT model: the literature observations, decomposition-product proxy features and exact model implementation must also be pinned and audited.

### New tests that could provide a real advantage

1. **Graph-aware reactivity:** compare a dyadic model with precursor effects and sparse pair interactions against simple prevalence, per-precursor degree, Tammann and thermodynamic-only baselines.
2. **Cold-start chemistry:** evaluate leave-one-precursor-out, chemistry-family-held-out and double-node-held-out performance; report per-fold uncertainty, not only pooled accuracy.
3. **Calibrated outcome prediction:** ROC-AUC and PR-AUC, Brier score, log loss, balanced accuracy, calibration slope/intercept, expected calibration error and per-precursor error slices. Accuracy alone is misleading at 25.3% positive prevalence.
4. **Censored onset prediction:** keep left-, right- and interval-censored likelihoods; exact °C MAE/RMSE must use independent measured-onset records only.
5. **Prospective experiment selection:** equalize furnace, material, time and characterization budgets; measure onset-interval contraction, information gain, cost per resolved pair and decision regret.

## Execution

```bash
cargo test --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_precursor_genome_graph_audit

cargo run --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_precursor_genome_graph_audit -- \
  --manifest data/research/precursor_genome_graph_benchmark_v0_1.json \
  --manifest-schema data/schemas/precursor_genome_graph_benchmark_v0_1.schema.json \
  --ledger /path/to/ledger_precursor_genome.json \
  --out precursor-genome-audit-receipt.json
```

The exact-head GitHub workflow fetches the pinned upstream ledger, verifies its Git blob object ID, runs the Rust tests and emits the compact audit receipt.

## Claim ceiling

This benchmark establishes only source identity, complete-pair-graph structure, label semantics and agreement with the published 1,000-sample cohort. It does not reproduce the AFT model, predict any new onset temperature, demonstrate predictive superiority or establish a new superconducting material.
