# Real Irregular Time-Series Validation Matrix (v0.1)

**Tracking:** NEURO-SILICON-011 / #9700  
**Parent research:** joint temporal resource control #9666 / PR #9667; replay baselines #9677 / PR #9680  
**Status:** external benchmark mapping and adapter contract drafted; no external model reproduction or dataset run

## 1. Purpose

Synthetic temporal fixtures are required for exact causal controls, hidden regime transitions, temporal aliasing and counterfactual oracle scoring. They do not establish real-world forecasting superiority.

This track adds a separate external-validation lane for irregular, missing, delayed, asynchronous and multimodal time series. The key principle is to compare on the benchmark's official task and split, preserving its native sampling law and data availability semantics.

## 2. Relevant public benchmark assets

### 2.1 Time-IMM / IMM-TSF (NeurIPS 2025 Datasets & Benchmarks)

Time-IMM is the closest direct match to the irregular temporal core. The benchmark describes nine irregularity causes spanning trigger-based, constraint-based and artifact-based patterns, and multimodal/asynchronous observations. Its paper evaluates nine datasets with entity-specific forecasting windows and chronological 60/20/20 train/validation/test splits. Its baselines include regular forecasting models, large time-series models and irregular-specific models.

- Paper: https://proceedings.neurips.cc/paper_files/paper/2025/hash/4199594d3c15736df2bf5274fa3155f4-Abstract-Datasets_and_Benchmarks_Track.html
- Dataset: https://github.com/blacksnail789521/Time-IMM
- Benchmark library: https://github.com/blacksnail789521/IMM-TSF

Repository audit at Time-IMM data commit `593a1ef31c9a5354f7b672fb2e8711b0aa8e2a24`: the release describes nine datasets, but only eight are redistributable because MIMIC-IV is restricted; authorized users must generate its processed data locally. The documented structure stores `time_series.csv` and `text.csv` per entity, each with a `date_time` field. Verify whether that timestamp means event time or actual availability time for each modality; do not infer arrival time from event time. IMM-TSF code is pinned at `99be43c391e21cef8247a138fa0b479d041a2b22`. Its README notes 24GB total GPU RAM for running LLaMA 3.1/DeepSeek text-embedding paths; structured-only runs or smaller documented text encoders should be treated as separate arms, not assumed to require those large models. Use official split and entity-window rules first. Retain original event times, modality sampling, missingness and cause metadata where license/task rules permit. Do not regularize away native irregularity.

### 2.2 APN / Time-Aware Patch Aggregation (AAAI 2026)

APN is a strong efficiency-oriented irregular multivariate forecasting comparator. Its TAPA module learns adaptive patch boundaries and uses time-aware weighted aggregation to regularize irregular inputs; its query/MLP head is intentionally shallow.

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/39563
- Official code: https://github.com/decisionintelligence/APN

Repository audit at APN commit `cd952af7fe60448d9bd826423d0b57944d178060`: its README states Python 3.11.13 and PyTorch 2.6.0+cu124 as the tested environment, and lists PhysioNet, MIMIC, HumanActivity and USHCN. Crucially, the repository README explicitly notes that although the paper states AdamW, experiments were conducted using Adam. The primary reproduction must preserve the released Adam behavior; an AdamW-corrected run is a separate ablation, never silently substituted. Reproduce the official training/evaluation protocol and compare on identical input timestamps and masks. The authors report competitive accuracy and efficiency; treat that as a literature claim until independently reproduced.

### 2.3 KAFNet / Canonical Pre-Alignment (AAAI 2026)

KAFNet is an important counterpoint to methods that avoid canonical pre-alignment. It uses pre-convolution, temporal kernel aggregation and frequency linear attention to control the compute cost of asynchronous multivariate series while retaining cross-variate structure.

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/40149
- Official code: https://github.com/zhouziyu02/KAFNet

Paper/code audit: KAFNet code is pinned here at `1cb7280b05dd851dbd30c17f7a46fa7f6c5318e6`. Its README documents `bash ./test.sh` for reproducing Human Activity; the AAAI paper reports experiments on four public irregular time-series datasets. Before claiming a full reproduction, inspect and execute all released scripts for every paper dataset and compare them against the paper tables. The authors report state-of-the-art results with 7.2x fewer parameters and 8.4x training/inference acceleration; these remain literature claims, not reproduced measurements here. Pin dependencies and compare preprocessing, parameter count, accelerator and measurement boundary.

### 2.4 GIFT-Eval (broad zero-shot time-series forecasting)

GIFT-Eval is a broad forecasting test bed, not an irregular-only or continual-learning benchmark. It contains 23 datasets, about 144,000 series and 177 million points across seven domains and ten frequencies, and supports zero-shot evaluation of general time-series forecasting models.

- Paper: https://arxiv.org/abs/2410.10393
- Official code: https://github.com/SalesforceAIResearch/gift-eval
- Dataset card: https://huggingface.co/datasets/Salesforce/GiftEval

Repository audit at GIFT-Eval commit `8895d8600d33983949916694047b372f6731a64c`: the project maintains explicit `TestData Leakage` and `Zero-shot` metadata. Its README defines zero-shot as neither using the GIFT-Eval training split nor leaking test data. Respect those columns and the official evaluation protocol. Use the official task contracts for broad generalization; do not combine its aggregate with Time-IMM's irregular forecasting score.

### 2.5 GIFT-Eval-CTX (contextual forecasting; Oct 2026 preprint)

GIFT-Eval-CTX studies structured forward-looking context with 10,000 instances across 20 datasets. Its structured macro/micro effects are derived from realized future trajectories. This makes it useful for studying the value of *legitimately available* external context, but also a serious leakage trap.

- Paper: https://arxiv.org/abs/2610.17794

Never provide realized-future-derived effect annotations to a forecast-time candidate. Treat them as evaluator-side labels or oracle analyses unless the equivalent information is independently available in the real task before the forecast origin. Report history-only and legitimate-context conditions separately.

### 2.6 NeuroBench 1.0

NeuroBench has relevant v1.0 algorithm tasks including few-shot class-incremental keyword learning and chaotic-function prediction, alongside event-camera and neural-decoding tasks. It provides an algorithm track for correctness/complexity and a system track for deployment-aware timing and efficiency.

- Framework: https://neurobench.ai/
- Code and leaderboards: https://github.com/NeuroBench/neurobench
- Methods: https://www.nature.com/articles/s41467-025-56739-4

Use chaotic-function prediction as a useful dynamical-system cross-check and class-incremental keyword learning as a continual adaptation cross-check if the Symthaea adapter meets the benchmark contract. The algorithm track's v1.0 scope does not include all preprocessing/postprocessing cost; system claims need end-to-end measurement and a complete workflow report.

### 2.7 Continual-learning baselines (Avalanche / ContinualAI)

For task-, domain- and class-incremental settings, include well-defined replay and stability baselines where assumptions match:
- online experience replay / reservoir replay;
- task-balanced replay / iCaRL-style exemplars;
- ER-ACE / ER-AML / supervised-contrastive replay;
- GEM / A-GEM;
- EWC / Synaptic Intelligence / LwF;
- GDumb and generative replay.

Reference: https://github.com/ContinualAI/continual-learning-baselines

Do not transfer image classification scores to continuous forecasting. Use these methods as algorithmic comparators only on compatible label/task-incremental adapters with a clear task oracle setting.

### 2.8 CL-Bench and MemoryBench

CL-Bench 1.0 measures LLM agents learning from prior task experience; its current leaderboard shows simple in-context learning ahead of some dedicated memory configurations. MemoryBench measures LLM-system learning from accumulated user feedback. Both are valuable stateful-agent evaluation contexts, not direct HDC/LTC time-series leaderboards.

- CL-Bench: https://continual-learning-bench.com/
- MemoryBench: https://proceedings.mlr.press/v306/ai26e.html

Keep model architecture, prompt/context access, memory budget and evaluator information exposure explicit in any adapter comparison.

## 3. Comparator tiers

| Tier | Required comparator | Why it belongs | What it can establish |
|---|---|---|---|
| Irregular forecasting | APN/TAPA | Adaptive temporal patching | Accuracy/efficiency under irregular sampling |
| Irregular forecasting | KAFNet/CPA | Asynchrony and cross-channel interaction | Cost of alignment vs learned compression |
| Real irregular data | Time-IMM/IMM-TSF | Nine cause-driven irregularity types | Generalization across irregularity mechanisms |
| Broad forecasting | GIFT-Eval models | Multi-domain, multi-frequency zero-shot forecasting | Broad transfer, not irregular-only superiority |
| Contextual forecasting | GIFT-Eval-CTX | Verified context value | Value of forecast-time context without leakage |
| Dynamical systems | NeuroBench chaotic prediction | Known dynamical prediction task | Stability, numerical prediction and complexity |
| Continual learning | FIFO/reservoir/ER/GEM/A-GEM/EWC/SI | Stability-plasticity reference methods | Forgetting and adaptation under matched replay/compute |
| Stateful agents | CL-Bench / MemoryBench | Experience and feedback reuse | Agent-level learning, not model-level forecasting |

## 4. Input contract: preserve causality and irregularity

Every event must preserve:
- entity and series identity;
- modality / channel identity;
- event timestamp (when the phenomenon occurred);
- availability timestamp (when the candidate could first observe it);
- source-local sequence / tie-break key;
- observation status: observed, missing, censored, delayed, invalid or not scheduled;
- value and unit/scale metadata;
- immutable sample/source digest.

At forecast origin t, candidate inputs must satisfy availability_time <= t. An event with event_time < t but availability_time > t is late-arriving information and must remain unavailable until its arrival. Never reorder by event-time and thereby leak delayed observations backward. Preserve simultaneous events and ordering ties explicitly.

Retain native missingness masks and observation cadence. If a method requires resampling, include it as a separately named preprocessor and charge its CPU, memory, latency and energy. No implicit forward fill, interpolation, zero imputation or regularization.

## 5. Splits, training and information parity

- Prefer each benchmark's official chronological/entity-aware split.
- Pin code commit, dataset release, license, preprocessing, scaler fit window and all train/validation/test boundaries.
- Fit transformations on train only; never normalize with future test data.
- Keep entity identity leakage controlled.
- Separate observed covariates, known-future covariates, unknown-future targets and evaluator-only annotations.
- Use identical information availability for each baseline.
- Freeze validation-based hyperparameter selection and tuning budgets; test data cannot be used for early stopping or checkpoint selection.
- Preserve official task metrics and report supplementary Symthaea-specific temporal exposure metrics separately.

## 6. Metrics and evidence

Forecast quality:
- MAE/RMSE for point forecasts;
- MASE or scaled errors where the official task specifies them;
- CRPS/quantile loss and coverage for probabilistic predictions;
- metrics per entity, horizon, modality and irregularity cause.

Temporal competence:
- wrong-belief, abstention and missing-output exposure separately;
- event detection delay and regime recovery time;
- calibration and selective risk;
- performance under delayed/missing/censored observation.

Efficiency:
- parameter/model footprint and peak memory;
- preprocessing + model + postprocessing + transfer time;
- p50/p95/p99 end-to-end latency and deadline misses;
- CPU/GPU compute and utilization;
- physical energy at a declared calibrated measurement boundary;
- retained byte-time, observation count and replay I/O/work where applicable.

Never report only a normalized aggregate or average across datasets. Include worst-dataset, worst-regime and per-horizon breakdowns and uncertainty intervals.

## 7. Claim gates (targets, not results)

- Reproduce official split and metric calculations before comparing.
- Zero future-context or future-label leakage.
- Pin exact source/data/preprocessor commits and record artifact hashes.
- At matched data, information, tuning budget and resource boundary, target at least 10% improvement on a preregistered primary metric with paired uncertainty intervals excluding zero and no material regression on other quality dimensions; or match quality while reducing a measured resource by 20% without exceeding another hard budget.
- Any SOTA claim requires official public-split head-to-head results against the strongest relevant baselines and independent reproduction.
- Synthetic results can establish mechanism validity but cannot establish real-world SOTA.

## 8. Immediate implementation order

1. Freeze the event/availability-time schema and comparator registry.
2. Reproduce Time-IMM official splits and baseline run manifests.
3. Port a transparent interpolation/linear and GRU-D or CRU-style baseline.
4. Reproduce APN, then KAFNet, pinning official commit and dependency stack.
5. Add Symthaea with no hidden future labels and paired runs.
6. Extend to GIFT-Eval broad zero-shot only after task and input parity are clear.
7. Add NeuroBench chaotic prediction and class-incremental tasks as separate tracks.
8. Measure end-to-end resources and publish per-dataset raw results.

No external model has been reproduced and no benchmark score is claimed by this plan.
