# PHM 2016 CMP source audit and benchmark integrity manifest v1

**Workstream:** #13334 — CMP virtual metrology and constrained recipe screening  
**Scorecard:** #13273  
**Luminous base revision:** `e36ef62abb9aa83db69f7e771760c18d42766597`  
**Status:** source/tree audit only. No PHM data files were downloaded or executed.

## 1. Public mirror identity

Repository: https://github.com/akangel0307/PHM-Data-Challenge  
Pinned branch commit: `00d443e3f57379e3ad3d12c15a0738282b8210e5`  
Pinned commit tree: `e5e6d95906a00bee908b96e4d9d124792bc895dc`  
Commit date: 2020-09-18T07:43:53Z  
Default branch at inspection: `master`  
GitHub repository metadata: public; GitHub reports MIT for the repository.  
Repository README blob: `920e354776c01ffaa9cb27e61b4dcd8a89035e32` (2,000 bytes)  
LICENSE blob: `fc12fb0d898731b215bc0b95e73d2d30b083d496` (1,064 bytes)

The MIT licence is a licence assertion for the repository's code. It is **not** evidence that the original PHM Society data have been relicensed under MIT. Data acquisition, redistribution and derived-data publication still require checking the PHM Society challenge terms and any separate data terms.

The pinned Git tree has 632 blobs and is not truncated. Main inventory counts:
- `data/`: 563 blobs;
- `two mode division experiment/`: 25 blobs;
- `three mode division experiment/`: 40 blobs;
- repository root and metadata account for the remaining blobs.

These identifiers are GitHub Git object IDs (SHA-1-style Git blob/tree identifiers), **not SHA-256 hashes computed from downloaded bytes**. File byte lengths below are the repository API's declared blob sizes. This manifest is a source pin; it is not a local content-hash verification or a dataset execution receipt.

## 2. Sensor and label bundle inventory

The pinned tree exposes three raw wafer-sensor cohorts. Each contains 185 CSV files:

| Cohort | CSV count | Total bytes |
|---|---:|---:|
| CMP training | 185 | 135,447,907 |
| CMP validation | 185 | 29,153,312 |
| CMP test | 185 | 31,032,148 |
| **Total** | **555** | **195,633,367** |

Label files:
- `CMP-training-removalrate.csv`: 43,095 bytes, blob `fa53a6257a333433c4c48a493c8e42b8f0d2a5f9`;
- `CMP-validation-removalrate.csv`: 9,227 bytes, blob `e356ac46271833456eb43374b5c04ed817e21740`;
- `CMP-test-removalrate.csv`: 9,215 bytes, blob `98ee87f5fe2790932ff5020f5de7f60bdaef287a`.

The tree also contains a nested `PHM16TestValidationAnswers` directory with copies of test and validation answer CSVs, plus macOS metadata sidecars. These should be deduplicated by content during a future ingestion, not counted as extra wafers.

Three-mode derived-array files:
| File | Bytes | Git blob ID |
|---|---:|---|
| `train_data.npy` | 1,801,788 | `a481440580d0e0f38ffed857cec68dc9f537aaab` |
| `validation_data.npy` | 386,689 | `058823691a3c98327217116444519d1f1d97cbd9` |
| `test_data.npy` | 386,689 | `ec60eb5353f0c5bb9c52c4dab768dfffd7344eb4` |

The two-mode derived table `data_final.csv` is 518,566 bytes, blob `b5241d836d8915a259cd539d04c89ad4a0ae483b`.

## 3. Pinned implementation files

| Path | Bytes | Git blob ID |
|---|---:|---|
| `two mode division experiment/data_utils.py` | 5,351 | `9b510d6e8af5a7eb183efabfc0960121e3815012` |
| `two mode division experiment/regression.py` | 4,547 | `c02dce35990aa737f6239dfbff996a2971054563` |
| `three mode division experiment/data_utils.py` | 5,626 | `411e92efcb7d726cc2b0c7637995ef73b43304d5` |
| `three mode division experiment/regression.py` | 4,432 | `22fcc1866435fd21b26133197d5a75db2173308c` |

The mirror README documents Python 3 / Anaconda / XGBoost and several classical regressors. The scripts are old (the pinned commit is from 2020) and should be treated as a historical reproduction target, not a maintained production pipeline.

## 4. Benchmark-integrity findings

### 4.1 Target-conditioned mode selection is not deployable routing

The two-mode `split_data` divides rows using the target itself:
- mode 0: `AVG_REMOVAL_RATE <= 120`;
- mode 1: `AVG_REMOVAL_RATE > 120`.

The corresponding regression script applies those same target-defined splits to both train and test data, then fits a separate regressor in each mode.

The three-mode experiment similarly uses:
- target (>120);
- target (leq120) and Stage A;
- target (leq120) and Stage B.

Stage may be observable before a prediction; the removal-rate target is not. Therefore:
- the target-defined partition may be reproduced as **retrospective oracle stratification**;
- it must not be called a deployable online regime classifier or mixture-of-experts router;
- any operational router must be trained and evaluated using features/context available before physical MRR is measured;
- the gap between oracle strata and observable-context routing must be reported.

This is a test-protocol issue, not evidence of malicious or intentional leakage.

### 4.2 Sensor aggregation drops timing detail

The pinned preprocessing:
- loops over sensor CSV files with `os.listdir`;
- drops the `TIMESTAMP` column before aggregation;
- groups by `WAFER_ID` and `STAGE`;
- computes mean, standard deviation, minimum, median and maximum;
- merges the resulting feature table with the removal-rate labels.

This creates useful wafer-level summaries but removes the raw temporal trajectory needed to test exact observation/label availability, late sensor arrival, within-wafer change points and causal ordering. The filesystem iteration order is not explicitly sorted. A future reproduction must freeze a deterministic source-file order and retain timestamps in a separate provenance channel even if the model feature pipeline excludes them.

### 4.3 Outlier filtering is not equivalent to the late-2026 CMP paper

The mirror's `load_dataframe` drops target rows with `AVG_REMOVAL_RATE > 1000`. The November 2026 CMP paper discusses four extreme MRR observations around 4,129–4,326 and applies a narrower declared inclusion rule (MRR in ((0,500])) for its reported cleaned analysis, alongside explicit sensitivity discussion.

Those protocols are not interchangeable:
- preserve a raw-label ledger;
- report the original four observations separately;
- reproduce the mirror's `<=1000` exclusion as a faithful legacy track;
- reproduce the paper's ((0,500]) rule as a separate paper-faithful track;
- add robust/no-exclusion sensitivity tracks;
- never silently discard anomalous physical measurements as “bad data”.

The four extremes are disputed/anomalous observations, not proven instrument faults. Do not infer their physical cause from magnitude alone.

### 4.4 The mirror's partitions are not the paper's chronological time-out split

The older mirror provides training, validation and test arrays, but its experiment script does not reconstruct the November 2026 paper's forward-time design. Its target-conditioned regime split is not a chronological validation protocol.

The late-2026 paper reports a different CMP evaluation design:
- 1,581 observations in the development history;
- 1,451 proper-training observations;
- 130 disjoint calibration observations;
- the final 396 observations as a chronological time-out test.

It reports a point-model (R^2=0.9543), RMSE (=5.9743), MAE (=4.7199), and 95.96% empirical coverage for nominal 95% intervals. Its 20 model-in-the-loop contexts report 95% target-window hit and 100% feasibility. Crucially, matched constrained baselines reportedly reached the same hit and feasibility rates. The reproduction must therefore ablate the predictive model, residual-GP uncertainty, interval guard, hard constraints and candidate-search method separately.

The paper also points to a Figshare supplementary archive, DOI `10.6084/m9.figshare.33184149`. Archive metadata, files, licence and content hashes were not independently retrieved during this audit; treat them as **unverified** until fetched and verified.

## 5. Required fair reproduction tracks

1. **Legacy mirror-faithful:** pinned preprocessing, target filter (>1000), target-conditioned two-/three-mode strata, original regressors and original test procedure. Label this retrospective / non-deployable for target-based routing.
2. **Observable-context routing:** same candidate model families, but route only on Stage, chamber/tool/consumable identifiers and process features available at prediction time. Include an UNKNOWN/pooled fallback.
3. **Pooled no-router baseline:** same time splits and model budget without mode routing.
4. **November 2026 paper-faithful:** 1,451 train / 130 calibration / 396 time-out; reproduce the published feature selection, residual-GP interval and constraint-search pipeline after retrieving its supplementary archive.
5. **Robust outlier sensitivity:** compare raw, mirror-filtered, paper-filtered and robust-loss treatment; retain all original observations with reason-coded exclusions.
6. **Evidence-gated operational track:** physical labels must pass numeric/revision/calibration qualification, with delayed availability represented explicitly. This is not the paper's immediate/offline regime and must be reported separately.

All tracks must keep MRR units, chronological ordering, wafer/stage identities and all preprocessing fit boundaries explicit. Scaling/feature selection must be fit on training data only. No future-label-derived mode, normalization, calibration or threshold may enter the prediction path.

## 6. What is not claimed

- No dataset bytes have been downloaded or executed.
- No SHA-256 of source data bytes has been calculated; listed Git object IDs are not raw SHA-256 values.
- Repository MIT metadata does not prove the PHM data licence.
- The mirror's target-conditioned regime models are not a deployable gate.
- The public mirror is not a full historian or a traceable physical-metrology chain.
- The November 2026 paper's supplementary Figshare file inventory remains unverified.
- No Luminous prediction, calibration, feasibility, success@K or manufacturing savings result is reported here.
