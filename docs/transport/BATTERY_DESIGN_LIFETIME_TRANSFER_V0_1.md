# Battery design-level cycle-life transfer benchmark v0.1

Status: benchmark protocol, synthetic fixture and audit scaffold. No battery performance, life, safety or production claim.

## Mission

Test whether cycle-life estimates can transfer to **unseen material/design combinations** from limited early-cycle observations, without access to target-design lifetime labels. This is a design-discovery task, not ordinary SOH/RUL prediction for a known battery.

Compose with Symthaea BATTERY-INVERSE-DESIGN-001 #11671 and EV-SOH-TRANSFER-BENCH-001 #11252. #11252 owns existing-cell health/RUL datasets, source manifests and prognostics evaluation; this benchmark owns the distinct design-level transfer question and prospective evaluation of unseen combinations.

## Research anchor

Zhang et al., *Discovery Learning predicts battery cycle life from minimal experiments*, Nature 650, 110–115 (2026), DOI 10.1038/s41586-025-09951-7. The paper reports 123 industrial-grade large-format lithium-ion pouch cells, diverse material/design combinations and protocols; 7.2% cycle-life test error on its stated evaluation using physical features from the first 50 cycles for 51% of prototypes; and estimated savings of 98% in time and 95% in energy under conservative assumptions. These are paper-reported results and are not directly comparable unless source dataset, target, cutoff, split and error formula match exactly.

Source: https://www.nature.com/articles/s41586-025-09951-7

## Core paradigm shift

Move from asking 'how healthy is this cell?' to asking 'will this new design combination satisfy the required lifetime, and how much early evidence is enough to reject or advance it?'

```text
material identity != design combination identity
early-cycle features != complete lifetime evidence
cycle-life prediction != observed end-of-life
one cell's result != replicated design-level performance
```

## Benchmark dimensions

### Prediction budgets

Freeze cycle cutoffs at 1, 5, 10, 20 and 50 cycles where source coverage permits. Every feature must bind its last physical observation and availability time. The model must not use later cycles, end-of-life labels, post-test impedance, or target-design statistics unavailable at the decision time.

Report the share of cells eligible at each cutoff. Do not present a 50-cycle result as applying to cells with fewer than 50 observed cycles.

### Holdouts

- Exact published-paper split and metric for reproduction.
- Leave-one-cell-out (cell unseen, design may be represented).
- Leave-one-material/design-combination-out (all cells for the held-out design excluded from fitting, preprocessing, selection and calibration).
- Protocol-family holdout where meaningful.
- Prospective preregistered top-K vs random vs uncertainty/diversity controls; freeze predictions before physical lifetime outcomes are revealed.

Random row/cycle splits are diagnostic anti-baselines only, never the headline transfer score.

### Baseline tournament

1. Physical/throughput and empirical trend baselines.
2. Ridge/linear and SVR.
3. Random forest and gradient-boosted trees/XGBoost.
4. TCN/LSTM/Transformer only with matched information and compute budgets.
5. Physics-guided feature models.
6. Reproduction of Discovery Learning only if exact code/data/split/rights are available. Otherwise label its published result as an external comparator, not a local reproduction.

### Metrics

- Reproduce the source paper's exact cycle-life error first.
- MAE/RMSE in cycles, normalized/percentage error with exact denominator, signed bias, P50/P90/worst-design error.
- Per-design-combination macro error as well as pooled per-cell error; a design with more specimens must not dominate the macro result.
- Prediction interval coverage, width and interval score, calibrated only on training/validation designs.
- Ranking: top-K enrichment, pairwise ordering, Spearman/Kendall and simple/Pareto regret where enough designs exist.
- Efficiency: observed cycles, wall time, equipment time, measured energy and material consumed per candidate decision.
- OOD/abstention quality by chemistry, design family, cell format, protocol and lab.

### Censoring and competing events

An observed end-of-life event is different from a right-censored observation, a competing failure, and an unknown endpoint. Only observed events receive ordinary point-error labels. Censored cells must be retained for separately qualified survival metrics; they are never zero-filled or assigned the last observed cycle as their lifetime.

## Proposed research targets

- Reproduce 7.2% only on the exact published split/metric before claiming parity.
- Then target <=7.2% on matched unseen-design splits with no worse tail error or interval coverage; <=5% is a stretch target, not a forecast.
- Demonstrate top-K enrichment over random and strong tree/physical baselines on prospectively frozen design groups.
- Validate time/energy savings with actual, boundary-matched measurements. Do not inherit the published 98%/95% estimates as guaranteed savings.
- Unsupported chemistry/design/protocol domains must produce ABSTAIN rather than confident extrapolation.

## Software audit scope

The Rust reference audit checks physical-cell identity, design-group holdout separation, observation cutoffs, prediction-time availability, unique predictions, and explicit event/censor/competing-event semantics. Its point-error evaluator scores event-observed cells only, reports both pooled-cell and design-group macro metrics, and keeps censored/competing outcomes out of point RMSE.

It does not implement survival analysis, train a predictor, authenticate source bytes, estimate causal design effects, authorize cell testing or qualify a battery. Those remain separate research and physical-evidence tasks.

## Next increments

1. License-aware reconstruction of the 123-cell industrial pouch-cell source and supplements.
2. Exact reproduction of the paper split/metric, with code and feature parity.
3. Cross-design and protocol-group split manifests, adversarial leakage controls and calibration.
4. Matched baseline tournament and data-efficiency curves.
5. Blind prospective top-K/random/diversity evaluation on independently built design combinations.