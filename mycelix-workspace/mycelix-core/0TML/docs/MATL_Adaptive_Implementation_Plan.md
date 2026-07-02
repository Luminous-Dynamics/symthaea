# MATL Adaptive Implementation Plan

**Date:** 2025-10-27  
**Owner:** Detector/RB-BFT Engineering  
**Status:** Draft (ready for execution)

---

## 1. Goals

1. Close the remaining detection gap for the 50 % sign-flip scenario without introducing regressions.
2. Expand the automated harness to cover non-IID label distributions and advanced Sybil behaviours.
3. Operationalise MATL adaptiveness with online retraining and continuous monitoring hooks.

---

## 2. Work Breakdown

### 2.1 Sign-Flip Symmetry Breaker (Highest Priority)

| Task | Description | Artefacts | Owner |
|------|-------------|-----------|-------|
| S1 | **Consensus Split Detector** – add a detector inside `RBBFTAggregator` to identify when gradients fall into two near-opposite clusters (high-magnitude cosine ≈ -1 across two cohorts). Trigger when ≥50 % of nodes fall into two clusters with opposing mean vectors and similar reputations. | `test_30_bft_validation.py`, new helper in `zerotrustml/ml` | Detector team |
| S2 | **Heuristic Response** – when consensus split is triggered: (a) suppress blanket committee rejections, (b) tighten PoGQ thresholds for both cohorts, (c) emit an audit alert. | Aggregator logic, logging hook (`Audit Guild` channel) | Detector + Ops |
| S3 | **Temporal Features Upgrade** – extend feature logging to capture per-node reputation delta and consensus split flags; feed into MATL training. | `ml_training_data.jsonl`, `train_matl_detector.py` | Detector team |
| S4 | **Targeted Simulation Suite** – add a focused sign-flip 50 % regression test (`tests/regression/test_signflip_symmetry.py`) exercised in CI. | New regression test, CI job | QA |
| Milestone | Sign-flip @50 % returns to ≤5 % FP with ≥95 % detection in harness. |

### 2.2 Harness Extensions (after S1 milestones)

| Task | Description | Artefacts | Owner |
|------|-------------|-----------|-------|
| H1 | **Label-Skew Support** – parameterise `run_attack_matrix.py` with `DISTRIBUTION=label_skew` and `alpha` sweep; update dataset profile to expose non-IID splits. | `scripts/run_attack_matrix.py`, dataset configs | Harness |
| H2 | **Advanced Sybil Behaviours** – implement temporal-drift and entropy-smoothing attack generators (`tests/adversarial_attacks/advanced_sybil.py`); hook into attack matrix. | New attack module, harness wiring | Harness |
| H3 | **Artefact Reporting** – extend matrix output to include distribution metadata and anomaly flags. | `bft_attack_matrix.json`, documentation | Harness |

### 2.3 Adaptive Ops & CI

| Task | Description | Artefacts | Owner |
|------|-------------|-----------|-------|
| C1 | **Online Retraining Hook** – script to retrain `models/byzantine_detector_latest` daily from aggregated logs; deploy as cron/Nix job. | `scripts/retrain_matl_detector.sh`, pipeline config | Ops |
| C2 | **Ensemble Integration** – add an Isolation Forest (or autoencoder) to run alongside MATL; expose combined score in aggregator. | `zerotrustml/ml/ensemble.py`, `test_30_bft_validation.py` | Detector team |
| C3 | **CI Automation** – create GitHub Actions workflow (nightly) running `run_attack_matrix.sh`, archiving `bft_attack_matrix.json` + plots, and failing on regressions. **Status:** Implemented (`matl-regression.yml` nightly job with label-skew summary). | `.github/workflows/matl-regression.yml`, Grafana datasource updates | Ops/QA |
| C4 | **Monitoring & Alerts** – push key metrics (per-attack detection/FP rates, consensus split alerts) to Prometheus/Grafana and set alert thresholds. **Status:** Prometheus artefact export + Grafana playbook documented (`docs/CI_and_Monitoring.md`). | Grafana dashboard, alert rules | Ops |
| C5 | **Audit Loop** – queue uncertain anomalies to Audit Guild via existing notification channel; track human-labelled outcomes for inclusion in retraining datasets. **Status:** Export tooling + review workflow documented (`scripts/export_audit_review.py`, `docs/Audit_Review_Workflow.md`). | Alert service integration, documentation | Governance liaison |

---

## 3. Risk & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sign-flip heuristic misclassifies benign oscillations in non-IID settings | False positives under real skew | Gate by reputation delta + cluster persistence (require multiple consecutive rounds) |
| Online retraining introduces drift from noisy labels | Detector instability | Use rolling validation set + hold-out synthetic baseline before promoting new model |
| Ensemble increases runtime | Exceeds round deadlines | Profile IsolationForest (or autoencoder) and offload heavy models to background threads; cache features |
| CI job runtime too long | Slows nightly build | Allow matrix caching per attack; parallelise by attack type |

---

## 4. Deliverables & Timeline (High-level)

1. **Week 1:** Complete S1 (symmetry breaker + regression test).  
2. **Week 2:** Implement H1/H2 and update documentation.  
3. **Week 3:** Deploy C1–C3 (online retraining + nightly CI).  
4. **Week 4+:** Roll out ensemble (C2), monitoring/alerts (C4), and audit loop (C5); run extended sign-off involving Audit Guild.

---

## 5. Acceptance Criteria

- MATL detector maintains ≤5 % FP and ≥95 % detection for all attacks at 50 % Byzantine in harness.  
- Non-IID and advanced Sybil scenarios covered in automated matrix with artefacts published nightly.  
- Online retraining pipeline produces `models/byzantine_detector_latest` snapshots with validation metrics meeting current thresholds before promotion.  
- Monitoring dashboards show per-attack detection/FP trendlines with alerts on regression.
