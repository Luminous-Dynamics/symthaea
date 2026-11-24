# Mycelial Reconstruction Calibration and Selective Risk

**Date:** 2026-11-22  
**Status:** First evidence-retaining calibration and risk–coverage implementation committed; exact-head validation pending.  
**Code:** `crates/domains/symthaea-canvas/src/mycelial_reconstruction_calibration.rs`

## 1. Paradigm shift: from confidence scores to risk-controlled reconstruction

A reconstruction system should not merely output a junction class and confidence. It should be able to answer:

- When it says "anastomosis" with 0.99 confidence, how often is that prediction actually a fusion?
- Does 0.9 confidence correspond to approximately 90% correctness on held-out organisms?
- How much graph-construction risk remains when the system accepts only the most confident predictions?
- How much improvement is possible if uncertain junctions are sent to annotation/review?
- Does the confidence estimate remain useful across species, time points, image quality and overlap density?

The objective is not to maximize automation coverage at any cost. It is to maximize useful automation **subject to a measurable risk budget**, especially for false fusion because a single false connection can create cycles, shorten graph paths, and distort downstream resilience estimates.

Confidence is an evaluation signal only. It never authorizes anastomosis or changes accepted topology.

## 2. Implemented probability contract

Each predicted junction receives seven probabilities, in a fixed schema order:

1. Origin
2. Branch
3. Tip
4. Anastomosis
5. Projected crossing
6. Uncertain
7. NoObject

The vector must be finite, lie in [0,1], and sum to one within a configured numeric tolerance.

Targets are created from the reconstruction score's explicit one-to-one junction matching:
- matched predicted junction → class of its matched ground-truth junction;
- unmatched predicted junction → NoObject.

Missed ground-truth junctions are not fabricated as prediction records. They remain false negatives in the detection/recall scorecard. Calibration and detection are therefore separate, complementary measurements.

The report retains a complete sorted per-junction observation inventory, including the exact probability-vector bit patterns and target class. It binds that inventory with a digest and independently recomputes all aggregate results during report validation. The output is not just an aggregate score with an opaque input digest.

## 3. Metrics implemented

### Classwise calibration

For every class, the report computes one-vs-rest:
- Brier score;
- expected calibration error (ECE);
- reliability bins with sample count, mean confidence, empirical frequency and absolute gap.

Classwise calibration matters because a model may be well calibrated for tips and branches but badly calibrated for anastomoses. A single top-label ECE would hide this.

### Top-label calibration

The report also computes:
- top-label Brier score;
- top-label ECE;
- negative log-likelihood (NLL);
- top-label accuracy.

These answer different questions. Brier and NLL reward proper probabilistic predictions; ECE compares confidence with empirical correctness. ECE is bin-dependent, so it must not be used alone.

### Selective risk and abstention

Predictions are sorted from highest to lowest top-label confidence. Every prediction with an identical confidence enters the retained set together; ties are never split.

For each threshold, the report records:
- retained prediction count and coverage;
- classification error count and selective risk;
- predicted anastomosis count;
- false anastomosis count and rate.

It also computes a piecewise-linear area under the risk–coverage curve (AURC), an oracle AURC that ranks all correct predictions before all errors, and their non-negative excess. The oracle is an offline lower bound that uses ground truth; it is not an inference policy.

The intended operational interpretation is: retain high-confidence predictions automatically, route lower-confidence ones to review, and measure the remaining error risk at each automation coverage.

### High-consequence false-fusion metric

False anastomosis predictions are explicitly counted and their rate among predicted anastomoses is reported at every selective threshold. This is a separate metric because a model can achieve strong aggregate accuracy while making a small number of costly false connections.

## 4. Reproducibility and validation

The calibration receipt binds:
- source observation digest;
- ground-truth and prediction graph digests;
- reconstruction score digest;
- matching policy and calibration policy IDs;
- bin count and probability numeric tolerances;
- sorted per-junction probability vectors and target classes;
- all per-class and top-label metrics;
- selective-risk curve and AURC/oracle-AURC summary.

The report validator reconstructs samples from the retained observations and recomputes:
- prediction and false-anastomosis counts;
- per-class prediction counts;
- classwise Brier/ECE reliability bins;
- top-label Brier/ECE/NLL/accuracy;
- full selective-risk curve;
- AURC, oracle AURC and excess area.

It rejects a receipt whose aggregate metrics cannot be reproduced from its evidence inventory. This gives consumers a way to verify the arithmetic without trusting the producer's aggregate values.

The probability class count is bounded to seven and the prediction inventory is bounded to 8,192 junctions. Calibration bins are bounded to 100.

## 5. Relevant SOTA and industry comparisons

### 3D filament reconstruction

Sten et al., *Three-dimensional graph reconstruction of filamentous structures from z-stack images*, Ecological Informatics 90, 103317 (December 2025), DOI https://doi.org/10.1016/j.ecoinf.2025.103317, report overlap-detection F1 around 0.91–0.92 on annotated fungal images. It is a reconstruction benchmark, not a direct probability-calibration baseline. The first fair comparison is to reproduce its detection task and then evaluate confidence on the same held-out cases, if licensing and annotation policies permit.

### 2D ridge/overlap detection

A 2024 *Computers and Electronics in Agriculture* paper, *A ridge-based detection algorithm with filament overlap identification for 2D mycelium network analysis*, reports detection F1 in the 0.88–0.94 range, crossing F1 around 0.89, and strong length agreement. This supports separate crossing/fusion and segment-length metrics rather than one graph score.

### SkelPy and temporal analysis

The 2026 SkelPy study reports a public workflow over 180 time-series images from three Botrytis isolates, sampled every four hours across 72 hours. It is primarily a 2D focal-plane time-series dataset, not equivalent to a 3D z-stack reconstruction task. The article is CC BY-NC 4.0; it is a non-commercial research comparator unless separate rights are obtained. Its existence is useful for designing time/replicate splits and longitudinal metrics, but it is not automatically a redistributable commercial benchmark.

### Calibration and uncertainty literature

Recent 2026 medical-image segmentation work is moving from raw confidence to error-aligned uncertainty, calibration and selective referral under a fixed human-review budget. A 2026 review recommends multiple calibration/uncertainty measures because ECE depends on binning and does not capture all kinds of uncertainty. A 2026 risk-aligned segmentation paper illustrates the value of measuring uncertainty against downstream error geometry rather than relying on confidence alone.

These are adjacent-domain methodological comparators. Symthaea must use a matched mycelial junction/edge task before making any comparative performance claim.

## 6. Proposed benchmark protocol

Targets below are proposals, not achieved results.

### Calibration

| Metric | Proposed target |
|---|---:|
| Top-label ECE | ≤0.05 |
| Per-class ECE for anastomosis, crossing and branch | ≤0.05 |
| Top-label Brier | Better than uncalibrated and temperature-scaled baselines |
| NLL | Better than uncalibrated baseline without class-specific regression |
| Reliability-bin count | Publish sensitivity at 5, 10, 15 and 20 bins |

Do not claim success on a single pooled ECE. Report calibration by class and organism/replicate, alongside Brier, NLL, reliability diagrams and confidence intervals.

### Selective prediction

Compare:
- raw maximum softmax probability;
- temperature scaling;
- per-class temperature scaling where justified;
- isotonic/Platt calibration on a separate calibration split;
- MC dropout / test-time augmentation;
- deep ensembles if compute budget permits;
- selective-set or conformal prediction with explicit coverage guarantees under its exchangeability assumptions.

At equal automation coverage (e.g. 50%, 70%, 80%, 90%, 95%), compare:
- selective risk;
- false-anastomosis risk;
- AURC and excess AURC against the oracle ranking;
- reduction in errors at a fixed annotation/review budget.

Also compare at equal target risk: how much of the graph can the system safely automate while keeping false-fusion risk below the declared limit?

### False-fusion safety

Proposed targets on a held-out crossing-only negative set:
- zero false accepted anastomoses in deterministic conformance controls;
- false-anastomosis rate below 0.5% overall on a matched held-out benchmark;
- no high-confidence (≥0.95) false fusion without a corresponding risk/abstention failure being visible in the scorecard.

These are ambitious targets, not current results. Publish denominators and confidence intervals; a zero count on a tiny test set is not evidence of a low population rate.

### Data splitting and distribution shift

- Split by organism and replicate, not by adjacent frames.
- Keep all frames from one biological replicate in one split.
- Hold out at least one species/isolate if a multi-species dataset is available.
- Stratify by overlap density, image contrast, z-stack spacing, occlusion, filament thickness and acquisition device.
- Calibrate on a separate validation set; never tune thresholds on the final test set.
- Report in-domain and cross-domain risk/coverage separately.
- Evaluate confidence under plausible image perturbations and segmentation uncertainty.

### Asymmetric costs and review policy

A false fusion can create a false cycle and corrupt downstream network metrics. A missed fusion may fragment an otherwise connected graph. Publish a cost matrix for these outcomes; do not collapse them into unweighted accuracy.

Human review should be budgeted by *junction/edge decisions* and by actual annotation time. A review policy must not assume every abstention costs the same amount.

## 7. Next paradigm-shift extensions

1. **Risk-constrained graph reconstruction:** choose thresholds to maximize accepted graph coverage subject to a predeclared false-fusion risk budget.
2. **Set-valued topology:** return {crossing, anastomosis} when evidence does not justify one class, rather than forcing a single label.
3. **Conformal selective reconstruction:** calibrate prediction sets on a held-out exchangeable calibration sample; explicitly test shift because nominal coverage is not guaranteed under arbitrary species/device shift.
4. **Topology-impact-aware review:** prioritize uncertain junctions by the potential change in connected components, cycle rank, root reachability and downstream failure-risk estimates—not confidence alone.
5. **Active annotation:** choose the next junction to label by expected reduction in held-out topology uncertainty per unit expert review time.
6. **Decision-quality benchmark:** compare accepted graphs after review at fixed human budgets, including graph error and downstream task performance.

The central ambition is to produce a system that knows which connections it has evidence to assert, knows which ones require review, and can quantify the cost of abstaining. A confidence score is not a biological fact and does not authorize a connection.

## 8. Current status boundary

Implemented in source:
- seven-class probability contract including NoObject;
- classwise Brier/ECE;
- top-label Brier/ECE/NLL/accuracy;
- confidence-threshold risk–coverage curve with tie-preserving threshold groups;
- false-anastomosis count/rate;
- piecewise-linear AURC, oracle AURC and excess area;
- retained per-junction probability/target evidence and arithmetic recomputation validation;
- synthetic false-fusion negative-control example.

Not yet achieved:
- no actual model has been calibrated using real microscopy data;
- no real 2D/3D fungal benchmark has been ingested;
- no industry/SOTA comparison has been run;
- no human-review study or actual annotation-time measurement has been run;
- no conformal coverage guarantee under shift has been established;
- exact-head Rust/Clippy validation is pending.

The next evidence-bearing milestone is a matched, licensed, organism-held-out annotated dataset with frozen calibration splits and a pre-registered human-review budget.
