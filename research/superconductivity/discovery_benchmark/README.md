# Superconductivity discovery benchmark contract

This workstream implements the benchmark contract and leakage auditor, not a predictor or a materials result.

## Why random-split performance is not enough

The November 2026 AutoSuperCon study reports 20,354 experimental parameter entries across 6,793 superconductors and eight properties. Its composition-based models report log-space R-squared of 0.83 for Tc, with median absolute error 1.6 K across 5,843 materials, and 0.69 for orientation-averaged Hc2 across 1,204 materials. In leave-one-family-out evaluation, the cuprate Tc score falls to -10.2, compared with +0.78 when cuprates are represented in training; the paper also reports severe failure for iron-based materials. The withheld-family cuprate Hc2 score is -9.0, improving to +0.4 when measured Tc is provided.

That is the central benchmark gap: excellent interpolation inside known families is not evidence of reliable extrapolation to a new superconducting family.

BEE-NET is a different and complementary baseline. Its reported 0.87 K MAE is against DFT-derived Allen-Dynes Tc labels, not measured experimental Tc. It predicts the Eliashberg spectral function and integrates ML screening with first-principles calculations, stability filtering and experiments. The 2026 report describes 741 computationally stable candidates and synthesis/experimental confirmation of two previously unreported compounds. This is a genuine workflow milestone, but it must remain a separate computational-label and experimental-confirmation track.

## Split contract

- Family holdout: no family may appear in both train and test.
- Source-paper holdout: all observations from a paper move together.
- Pressure-regime holdout: pressure bounds are explicit; the gap between train and test is excluded.
- Time-forward holdout: the training cutoff and test start dates are frozen.
- Prototype holdout: prototype identity is versioned and held out as a group.

Each split must group by both material_state_id and source_id. Additional grouping keys are declared per split. Formula grouping can be required for strict composition generalization. Pressure extrapolation may intentionally include the same composition at a different explicitly identified state.

## Evidence contract

The target authority is explicit and must match every observation in a task. Retain normalized formula, family/prototype taxonomy, material-state identity, pressure, temperature when reported, field orientation, sample form, measurement method and criterion, source paper and date, source locator, extraction method, reported-versus-derived status, and uncertainty.

Do not merge resistive onset, zero resistance, diamagnetic onset, bulk thermodynamic transition, or calculated Tc into one label. In particular, a calculated DFT/SCDFT/Allen-Dynes Tc is not experimentally measured, and an unspecified measurement condition is not known.

## Metrics for a real benchmark run

Report by property, measurement class, label authority, family and split regime: MAE/RMSE in physical units and log-space; median absolute error; paired confidence intervals; prediction-interval coverage and width; calibration; risk-versus-coverage for abstaining models; precision/recall and top-k enrichment at fixed candidate budgets; and cost per independently confirmed candidate. Bootstrap by material/source groups, not individual rows.

A high global score must never conceal a catastrophic family-specific failure.

## Current implementation boundary

The version 0.1 auditor validates a seven-row synthetic adversarial fixture. It proves only that the contract and split guardrails behave as coded. The fixture is fabricated, is not an experimental or computational dataset, must not train a model, and has no benchmark score. The Rust auditor deliberately rejects external datasets until a separate importer pins source version, licensing/terms, preprocessing and immutable digests.

## Research references

- Manium et al., AutoSuperCon, arXiv:2611.15627 (11 Nov 2026): https://arxiv.org/abs/2611.15627
- Gibson et al., BEE-NET, npj Computational Materials 12, 95 (27 Jan 2026): https://doi.org/10.1038/s41524-026-01964-8
- HTSC-2025 family/system-aware dataset: https://doi.org/10.1088/1674-1056/adf042
- Finite-momentum SCDFT framework, PRL 137, 196001 (3 Nov 2026): https://doi.org/10.1103/lrhl-j9bt
