# SEMI-BENCH-001 — 2026 literature and baseline review

**Review date:** 2026-11-21  
**Parent:** [SEMI-BENCH-001 #13058](https://github.com/Luminous-Dynamics/symthaea/issues/13058)  
**Execution follow-up:** [SEMI-BENCH-002 #13067](https://github.com/Luminous-Dynamics/symthaea/issues/13067)

This note records two recent reference studies discovered during the benchmark design. They are comparison anchors, not a claim that the field has one definitive state-of-the-art result.

## 1. SECOM: supplementary 2026 baseline under rare failures and missingness

Park and Baek, *Data-Driven Defect Prediction for Manufacturing Quality Monitoring Under Class Imbalance and Missing Data: A Performance–Efficiency Trade-Off Analysis*, published 24 June 2026:

https://www.mdpi.com/2075-1702/14/7/716  
DOI: https://doi.org/10.3390/machines14070716

The paper's main experiment is on Bosch Production Line Performance; SECOM is a supplementary semiconductor-dataset check. For SECOM, the authors report 1,567 observations, 104 defective observations (6.64%), and 474 process variables after removing time and constant/single-value features. They use a stratified 60/20/20 split (939/314/314), leaving only 21 defective examples in each validation and test subset.

Reported best PR-AUC configurations in the supplementary SECOM table:

| Imputation | Feature set | Model | Imbalance handling | PR-AUC |
|---|---|---|---|---:|
| Zero | A | LightGBM | None | 0.2207 |
| Zero | B | Random Forest | Balanced | 0.2033 |
| Median | A | Random Forest | Balanced | 0.1947 |
| Median | B | Random Forest | Balanced | 0.1955 |
| KNN | A | LightGBM | None | 0.1997 |
| KNN | B | Random Forest | Balanced | 0.2168 |

Feature set B adds per-feature missingness indicators and a per-sample missing ratio. The authors report that a validation-selected threshold targeting at least 0.90 precision resulted in zero held-out test recall in this small SECOM evaluation. They explicitly caution that the small validation/test counts make the operating-point result unstable.

**Benchmark implication:** reproduce PR-AUC and the validation-selected threshold protocol separately. Do not infer a stable production alarm operating point from 21 positive validation and 21 positive test cases. Report uncertainty, the number of positive cases and how often threshold selection has no feasible operating point.

Official dataset reference:
https://archive.ics.uci.edu/dataset/179/secom  
DOI: https://doi.org/10.24432/C54305

UCI reports 1,567 instances, 591 features, 104 failures, missing values, two source data files plus a names file, and CC BY 4.0 licensing. Preserve the exact file layout and original label/timestamp semantics in the data-admission step. Do not normalize away the distinction between official UCI metadata and the common convention of referring to the sensor matrix separately from label/time fields.

## 2. WM-811K: 2026 PSS-HNet reference

Song et al., *Integrating Local and Global Features for Wafer Defect Pattern Classification via Sequential Hybrid Architecture*, published 31 March 2026:

https://www.mdpi.com/2227-9717/14/7/1134  
DOI: https://doi.org/10.3390/pr14071134

Reported results:

| Model | Accuracy | Macro-recall | Macro-F1 | Parameters | FLOPs | Latency |
|---|---:|---:|---:|---:|---:|---:|
| iFormer | 0.9782 ± 0.0006 | 0.8852 ± 0.0022 | 0.9044 ± 0.0010 | 6.245M | 1.103G | 8.666 ms |
| PSS-HNet | 0.9780 ± 0.0002 | 0.8954 ± 0.0034 | 0.9098 ± 0.0008 | 4.381M | 0.754G | 7.682 ms |

The authors trained on an H100, measured latency on an RTX 3060 Ti, reported three independent seeds and used the 172,950 labelled WM-811K maps in a stratified 80/20 sample-level train/test split across nine classes. Their article states that cross-dataset and different-process-condition generalization remains to be validated, minority class supports are limited, and batch/latency variability needs further assessment. Their experiment focuses on controlled variants within an iFormer-based architecture, not a broad comparison with every CNN and transformer family.

**Benchmark implication:** use PSS-HNet/iFormer metrics as a reproduction target only under a matched protocol. Add a separate wafer/lot-group holdout if exact lot identifiers are available and the source structure supports it. Do not compare an unseen-lot result directly to a random-split result as if the split difficulty were equal.

WM-811K source lineage must be recorded with a specific distributor snapshot. Verify dataset reuse/redistribution terms before checking in any content; this repository stores no raw dataset.

## 3. Competitive protocol corrections

These findings change the recommended evaluation sequence:

1. **Reproduction track:** match published sample filters, labels, preprocessing, split, metric conventions, seeds and hardware as closely as possible.
2. **Leakage-audit track:** add duplicate/identity/group/temporal leakage controls without changing the reproduced protocol.
3. **Generalization track:** evaluate forward-in-time and/or group-held-out performance only where exact timestamps or lot identities support the split.
4. **Decision track:** evaluate calibration, risk-versus-coverage and precision-constrained recall using validation-selected thresholds applied once to held-out test data.
5. **Efficiency track:** report per-sample latency, batch throughput, p95/p99 latency, memory and power only when properly instrumented.
6. **Physical-impact track:** keep separate; classification or process prediction alone does not prove yield, scrap, energy or cost improvement.

A model that wins on random splits but fails on unseen lots is not a successful fab-transfer model. A model with high PR-AUC but zero recall at a conservative production threshold is not an effective alarm system. A model that improves macro-F1 but doubles inference time may or may not be better, depending on inspection throughput and decision costs.

## 4. Research opportunity beyond these papers

The highest-value next experiment is not to invent another image backbone immediately. First establish a faithful, independently reproducible reference and identify failure modes that these papers leave unresolved:

- unseen-lot and unseen-process-condition generalization;
- class support and rare-pattern recall stability;
- measurement corruption, missingness and calibration shift;
- confidence calibration and abstention;
- throughput variability and edge-device resource limits;
- selective inspection: when should the system classify, request another measurement, route to human review or abstain?

A later Symthaea HDC/CfC model should be evaluated as a challenger against PSS-HNet/iFormer and compact CNN baselines under identical input representation, split, compute budget and metric definitions. Its architectural novelty is not evidence of superiority.

## Claim ceiling

This review establishes current literature anchors and their reported protocols. It does not reproduce either paper, establish a new SOTA result, validate transfer to a live fab or demonstrate physical manufacturing improvement.


## EDA placement: proxy objectives versus final routed design

ChiPBench (NeurIPS 2025 Datasets & Benchmarks Track) evaluates six AI placers on 20 CPU/GPU/NPU circuits through the full OpenROAD flow and reports degradation versus hierarchical baseline for the evaluated AI methods: https://proceedings.neurips.cc/paper_files/paper/2025/hash/1cba8502063fab9df252a63968691768-Abstract-Datasets_and_Benchmarks_Track.html.

PPAPlace (ICCAD 2026) reports approximately -0.08 Pearson correlation between macro HPWL and WNS, finds post-global-routing labels more faithful to final timing than HPWL/pre-route timing, and reports 22% WNS / 51% TNS improvement on five held-out ChiPBench circuits versus the hierarchical baseline while preserving power/routability: https://arxiv.org/abs/2608.13790. Treat as paper-reported, not reproduced Luminous results.

A separate end-to-end evaluation workstream is [SEMI-EDA-001 #13069](https://github.com/Luminous-Dynamics/symthaea/issues/13069). Its central rule is that predicted or intermediate metrics cannot stand in for final routed outcomes.

## Manufacturing intelligence and semiconductor digital-twin standards

NIST's 2026 Super Intelligence for Manufacturing programme describes TIDAL (Teamed, Integrated, DT-assured and Layered Intelligence), spanning asset, site/area and enterprise levels, and includes a shared operation test bench combining real equipment and simulators: https://www.nist.gov/programs-projects/super-intelligence-si-manufacturing.

SEMI SNARF #7465 proposes a Specification for Digital Twins in Semiconductor Manufacturing covering framework definitions, application classes, external/security interfaces, lifecycle and prediction-quality metrics: https://downloads.semi.org/web/wstdsbal.nsf/0e0afa4c4969bea688256efd0062a27c/61c2bf6034c0eed988258dcd00669238%21OpenDocument. This is a standard-development activity proposal, not a final published standard or a Luminous conformance result.


## MixedWM38 and compositional zero-shot recognition

The upstream MixedWM38 README explicitly states that GAN-generated wafer maps were added to balance class counts: https://github.com/Junliangwangdhu/WaferMap. A public multi-label reproduction reports macro-F1 0.9846 and exact-match 0.9696 but explicitly cautions that MixedWM38 is largely GAN-synthesized and saturates quickly: https://github.com/ALEX8642/wafer-mixed. These are useful pipeline-comparison results, not evidence of real-fab generalization.

Yang, Jang and Kim (Expert Systems with Applications 299, 2026, article 130127) propose generalized zero-shot classification of mixed wafer patterns from imbalanced single-defect training using CMDM and PASS, reporting 74.3% harmonic-mean accuracy in the most imbalanced setting and 3.92 percentage-point improvement over a compared diffusion classifier: https://doi.org/10.1016/j.eswa.2025.130127. This supports a separate leave-combination-out track rather than only random multi-label interpolation.
