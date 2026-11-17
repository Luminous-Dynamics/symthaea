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

Repository audit at Time-IMM commit `e934cef23d526817dcf0d1a66c6f82274c96c04b` (the previously noted `593a1ef...` is the README blob SHA, not a commit): the release describes nine datasets, but only eight are redistributable because MIMIC-IV is restricted; authorized users must generate its processed data locally. The documented structure stores `time_series.csv` and `text.csv` per entity, each with a `date_time` field. The event-contract crate now supports content-addressed opaque payloads so text/embedding content can be kept separate from timing metadata, and a strict CSV adapter handles date_time/record_id numeric and text tables. Verify whether each `date_time` means event time or actual availability time for that modality; do not infer arrival time from event time. IMM-TSF commit is `b9e245f2238a9c35d241dbc5bd943c28193a5b90` (README blob SHA `99be43c...`). Its README notes 24GB total GPU RAM for running LLaMA 3.1/DeepSeek text-embedding paths; structured-only runs or smaller documented text encoders should be treated as separate arms, not assumed to require those large models. Use the new strict CSV adapter only after verifying per-modality timestamp meaning and choosing an explicit availability policy; its EventTimeAssumedImmediate mode is an approximation, not evidence of actual arrival. Use official split and entity-window rules first. Retain original event times, modality sampling, missingness and cause metadata where license/task rules permit. Do not regularize away native irregularity.

### 2.2 APN / Time-Aware Patch Aggregation (AAAI 2026)

APN is a strong efficiency-oriented irregular multivariate forecasting comparator. Its TAPA module learns adaptive patch boundaries and uses time-aware weighted aggregation to regularize irregular inputs; its query/MLP head is intentionally shallow.

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/39563
- Official code: https://github.com/decisionintelligence/APN

Repository audit at APN commit `f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4` (README blob SHA `cd952af...`): its README states Python 3.11.13 and PyTorch 2.6.0+cu124 as the tested environment, and lists PhysioNet, MIMIC, HumanActivity and USHCN. Crucially, the repository README explicitly notes that although the paper states AdamW, experiments were conducted using Adam. The primary reproduction must preserve the released Adam behavior; an AdamW-corrected run is a separate ablation, never silently substituted. Code audit: `utils/configs.py` defaults `train_val_loader_shuffle` and `train_val_loader_drop_last` to `None`; `data/data_provider/data_factory.py` applies `configured_value or True` to all non-test loaders. Thus train and validation are both shuffled and drop-last by default, and explicitly passing zero cannot disable either setting. Validation therefore sees a randomly truncated subset per epoch, which can perturb early stopping. The P12 script requests five iterations (main seeds 2024–2028), 200 epochs and patience 10. The clean comparison must set validation shuffle=false and drop_last=false and evaluate the full validation set; preserve the released loader path separately for official-code reproduction. Reproduce the official training/evaluation protocol and compare on identical input timestamps and masks. The authors report competitive accuracy and efficiency; treat that as a literature claim until independently reproduced.

### 2.3 KAFNet / Canonical Pre-Alignment (AAAI 2026)

KAFNet is an important counterpoint to methods that avoid canonical pre-alignment. It uses pre-convolution, temporal kernel aggregation and frequency linear attention to control the compute cost of asynchronous multivariate series while retaining cross-variate structure.

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/40149
- Official code: https://github.com/zhouziyu02/KAFNet

Paper/code audit: KAFNet repository commit is `1532833bca60a114476affff0f7360f955f250ac` (README blob SHA `1cb7280...`). Its README documents `bash ./test.sh` for reproducing Human Activity; the AAAI paper reports experiments on four public irregular time-series datasets. Before claiming a full reproduction, inspect and execute all released scripts for every paper dataset and compare them against the paper tables. The authors report state-of-the-art results with 7.2x fewer parameters and 8.4x training/inference acceleration; these remain literature claims, not reproduced measurements here. Pin dependencies and compare preprocessing, parameter count, accelerator and measurement boundary.

### 2.4 GIFT-Eval (broad zero-shot time-series forecasting)

GIFT-Eval is a broad forecasting test bed, not an irregular-only or continual-learning benchmark. It contains 23 datasets, about 144,000 series and 177 million points across seven domains and ten frequencies, and supports zero-shot evaluation of general time-series forecasting models.

- Paper: https://arxiv.org/abs/2410.10393
- Official code: https://github.com/SalesforceAIResearch/gift-eval
- Dataset card: https://huggingface.co/datasets/Salesforce/GiftEval

Repository audit at GIFT-Eval commit `82f89c3a2de1e7cd98e35dbcb6917cb4ebb6fd18` (README blob SHA `8895d860...`): the project maintains explicit `TestData Leakage` and `Zero-shot` metadata. Its README defines zero-shot as neither using the GIFT-Eval training split nor leaking test data. Respect those columns and the official evaluation protocol. Use the official task contracts for broad generalization; do not combine its aggregate with Time-IMM's irregular forecasting score.

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

## 2.8 Current time-series foundation-model scorecard (checked 17 Nov 2026)

The standard zero-shot forecasting market has moved quickly. The current comparator set should explicitly include **TimeH2O-v1**, **Timestral-209M**, **ci-proto**, **TimeRSI-1.5**, **TimeRSI**, and **t0-gamma**, alongside Chronos-2, TimesFM-3 and Toto-2.0. Do not use one source or metric as a universal rank.

### FEV-Bench snapshot and source discrepancy

H2O's public surfaces now expose three dated win-rate observations, which should remain separate:

| Public surface / snapshot | TimeH2O-v1 | Timestral-209M | t0-gamma | TimesFM-3 | Chronos-2 | Toto-2.0-2.5B |
|---|---:|---:|---:|---:|---:|---:|
| H2O live benchmark page (retrieved 17 Nov; no explicit page snapshot date) | 91.50% | 89.00% | 87.40% | 82.00% | 75.80% | 72.20% |
| H2O GitHub benchmark README (benchmark data read 13 Nov; fetched 17 Nov) | 89.60% | 87.10% | 84.70% | 79.30% | 72.20% | 69.40% |
| H2O launch press release (benchmark view dated 24 Oct; release 26 Oct) | 91.90% | 89.60% | 88.10% | not stated here | not stated here | not stated here |

- H2O benchmark page: https://h2oai.github.io/benchmarks/
- H2O model card / repository: https://huggingface.co/h2oai/timeh2o-v1 and https://github.com/h2oai/benchmarks
- Mistral Timestral release: https://mistral.ai/news/timestral/

These public surfaces are inconsistent. Preserve every source URL, retrieval date, benchmark data date, raw snapshot and aggregation code; do not average the observations or select the most favorable number. The current TSFM.ai mirror also reports 170 GIFT-Eval models, with Taichu-TimeSeries-Agent-v2 at average MASE rank 18.02 and Timestral-209M at 32.52. FEV-Bench's MASE-skill view separately lists TimeRSI-1.5 at 41.44, ci-proto at 40.80 and TimeRSI at 40.58. These are metric-specific rankings, not the same target as GIFT-Eval's grouped-slice average rank. The H2O model card separately reports TimeH2O-v1 FEV-Bench SQL skill 50.83 and MASE skill 39.95, as well as GIFT-Eval CRPS 0.441 / MASE 0.654. Its current model card lists a non-commercial licence: https://huggingface.co/h2oai/timeh2o-v1 . Accuracy references do not imply commercial self-hosting rights.

### Cross-benchmark rankings are different questions

The TSFM.ai mirror reports that its FEV-Bench point-accuracy MASE-skill view (data loaded 15 Nov 2026) has TimeRSI-1.5 at 41.44, ci-proto at 40.80 and TimeRSI at 40.58. Its GIFT-Eval page ranks grouped probabilistic slices by average MASE rank and is led by Taichu-TimeSeries-Agent-v2, STRIDE with Synapse and RegLab-AH-2. Its BOOM telemetry view is led by Toto 2.0 models on CRPS. Those are different datasets, metrics and aggregation procedures; they should not be mixed into a single table as though ranks were interchangeable: https://tsfm.ai/benchmarks .

### 17 Nov 2026 FEV-Bench live-view update

Two live FEV-Bench surfaces were rechecked on 17 November. They rank different metrics and their observations must not be reconciled into a single leaderboard:

**Point-accuracy MASE skill** — TSFM.ai's current 52-entry view lists TimeRSI-1.5 at 41.44, ci-proto 40.80, TimeRSI 40.58, TimeH2O-v1 39.95, Timestral-209M 39.49, Simons-297M 37.85 and TimesFM-3 37.82. This is MASE skill, not SQL win rate: https://tsfm.ai/benchmarks/fev-bench

**SQL win rate** — The Aggregate's daily-refreshed 16 Nov snapshot lists ci-proto 90.00%, TimeH2O-v1 89.65%, Timestral-209M 87.06%, t0-gamma 84.68%, Simons-297M 81.97%, TimesFM-3 79.26%, Chronos-2 72.24%, and Toto-2.0-2.5B 69.44%: https://theaggregate.ai/benchmark?slug=fev-bench-sql

**Conflicting live vendor dashboard** — H2O's current live dashboard retrieved on 17 Nov displays TimeH2O-v1 91.50%, Timestral-209M 89.00%, t0-gamma 87.40%, TimesFM-3 82.00%, Chronos-2 75.80% and Toto-2.0-2.5B 72.20%: https://h2oai.github.io/benchmarks/ . This conflicts with its GitHub README data read 13 Nov and with the Aggregate snapshot. Keep it as a separate live-surface observation; do not average it with the dated dataset or choose the most favorable number.

All three observations are pinned in the machine-readable comparator snapshot with source type, metric, date and qualification caveat. This matters for Symthaea's benchmark design: report the official FEV task-level skill and win-rate metrics separately, then separately report irregularity, calibration, decision regret, completeness safety and system cost. A competitor's lead on one FEV metric is not a universal win across every forecasting or decision task.

### Machine-readable comparator snapshot

The exact score observations above are now pinned separately in
`crates/domains/symthaea-capability-closure/fixtures/tsfm_comparator_snapshot_v0.1.json`
and linked from `real_irregular_benchmark_registry_v0.1.json`. It records
TimeH2O-v1 (108.8M), Timestral-209M, ci-proto, TimeRSI-1.5, Toto-2.0-2.5B,
Chronos-Bolt and TimesFM-3, including source URLs, retrieval dates, claim type,
metric and task-specific caveats. The two conflicting H2O FEV-Bench surfaces
remain separate observations. No external checkpoint is claimed to have been
reproduced. The registry and comparator fixture are hashed in the exact-head
validation receipt.

### Required matched comparison set

1. **FEV-Bench:** use the official 100-task suite and record SQL / WQL / MASE / WAPE scores separately; retain raw per-task results, win-rate estimator and bootstrap uncertainty.
2. **GIFT-Eval:** probabilistic CRPS, MASE, quantile coverage and slice-weighted rank; separate official zero-shot results from any fine-tuning.
3. **TIME:** task-centric forecast accuracy over its official 50-dataset / 98-task surface; pin official task definitions and checkpoint.
4. **BOOM:** observability telemetry CRPS and anomaly/forecasting behavior; do not generalize a telemetry advantage to arbitrary irregular multimodal series.
5. **BITS and Time-IMM:** retain the irregular sampling, missingness, trigger/constraint/artifact causes, source cadence and event-vs-arrival semantics. These remain the closer target for the proposed Symthaea event contract than a broad regular-grid forecast score.
6. **Cost Pareto:** record model size, cold/warm p50/p95/p99 latency, throughput at matched context/horizon/batch, peak memory and measured joules at the same hardware boundary. Separate inference cost from the full event-validation, preprocessing, update and memory pipeline.

For each public baseline, freeze checkpoint hash/revision, licence, model card revision, tokenizer/normalizer, official harness commit, dataset release, split, context, horizon, covariate availability, quantile levels, tuning budget, metric implementation and aggregation. A vendor-reported lead is a comparator to reproduce, not proof that Symthaea has lost or won. Do not compare TimesFM 3.0 self-hosted weights with a hosted API result without recording the separate licence, service terms and system boundary.

## 2.9 Simons-297M and the research-loop efficiency frontier (11 Nov 2026)

**Simons** is now a mandatory open-weight comparator, not just another checkpoint. Abel AI Lab reports a 25M/60M/178M/297M family, with Simons-297M fourth among the open-weight models shown for GIFT-Eval, FEV-Bench and TIME; its release table also reports first place on M6 (168 entries). It supports nine quantiles and known-future covariates. The vendor reports full GIFT-Eval in under 15 minutes on one A100 and about 20x the speed of leading similarly sized models under their respective leaderboard settings. These are vendor-reported results; pin checkpoint, evaluation settings and hardware before reproduction.

- Release and leaderboard table: https://abel.ai/research/introducing-simons/
- Code, training recipes and synthetic-data generator: https://github.com/Abel-ai-lab/Simons
- Weights: Simons-25M/60M/178M/297M; OpenMDW-1.1
- Training/inference code: Apache-2.0
- Public pretraining collection: OpenMDW-1.1, while each source dataset retains its own licence

The same release is a methodological challenge to architecture-first research. Abel reports a two-week campaign costing less than $8,000 GPU time, more than 250 research directions and nearly 2,000 experiments. Its own attribution says roughly three quarters of first-campaign gains came from data, synthetic-data mixture and input transformations; around one fifth came from optimizer/training schedule; backbone changes contributed comparatively little. It also reports validation-only selection followed by fixed-code retraining on reserved seeds, with agents barred from modifying evaluator code or seeing benchmark labels. These are Abel's own development claims, not independently audited Symthaea results.

### Add a research-loop benchmark, not another model leaderboard

A research system should be scored on both **what it discovers** and **how much verified cost it takes to discover it**. Add a separate meta-benchmark lane with these outputs:

1. **Quality–compute Pareto curve:** official validation skill / CRPS / decision regret against cumulative GPU-hours, CPU-hours, wall-clock and USD. Report the best verified model reachable at each preregistered budget.
2. **Cost per confirmed improvement:** marginal compute and cost for an improvement that survives the reserved-seed confirmation protocol and has no material regression on guard metrics.
3. **Confirmation rate / false-discovery rate:** fraction of validation-selected candidates whose gains replicate on previously reserved seeds; report uncertainty, not only the best lucky run.
4. **Research throughput:** hypotheses, completed experiments, failed experiments, useful artifacts, and median time from hypothesis to independently confirmed result.
5. **Attribution quality:** matched ablations for data quality/mix, missingness/time transforms, optimizer/schedule, model architecture and inference-time policy. Do not attribute a gain to the backbone when it came from a preprocessing change.
6. **Evaluator integrity:** freeze official splits, metric code and label access; the proposer/researcher cannot mutate or inspect evaluator code or holdout labels; selection uses validation only; confirmation uses reserved seeds; final test is run once after protocol lock.

For Symthaea, the central target should be a higher area under the **verified improvement versus cumulative compute** curve, while preserving the ordinary forecasting quality gates. A proposed search target is at least 50% lower compute-to-confirmation than a carefully budget-matched manual research baseline at equal verified improvement, or at least 20% higher confirmed improvement under the same compute budget. These are research targets, not results. The research loop must not tune against test labels or weaken guards to increase its confirmation rate.

The machine-readable comparator snapshot records Simons' reported ranks, licences, run-time and development-process claims separately from TimeH2O, Timestral, TimesFM, Chronos and Toto. Do not merge their scores across FEV-Bench win rate, MASE skill, GIFT-Eval MASE/CRPS, TIME, M6, BOOM and inference-time measurements.

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
- performance under delayed/missing/censored observation;
- **false-complete rate** (primary completeness safety metric): an interval declared complete despite a later source event within it;
- false-incomplete rate and completeness lag;
- late-arrival breach detection recall and detection latency;
- source-specific watermark skew and asynchronous modality delay;
- fraction of absent/unknown facts retained as Unknown until all required sources close the interval.

Completeness is evaluated against hidden source schedules. A source watermark is a policy-bound progress claim, not proof of physical truth; measure false-complete and false-incomplete behavior separately.

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


## 9. Bitemporal source history

The candidate event adapter now includes a microsecond append-only event
revision ledger. It distinguishes event/valid time from transaction/knowledge
time and supports assertion, supersession and retraction without overwriting
prior history. This supplements the existing day-granularity Broca bitemporal
ledger and its alternative-proof runtime (PRs #9109 and #9292); it does not
replace or yet compose with them. This distinction is important for irregular
datasets where corrections and late arrivals may occur within the same day.


## 10. Time-IMM/IMM-TSF loader leakage audit

Pinned IMM-TSF parser commit:
`2642b920fb231ef3df43ed85adc6bd003e7664d8`.

Source review of `lib/parse_datasets.py` found two benchmark-harness conditions
that must be reported explicitly:

1. In `ChunkedTimeSeriesDataset.__init__`, each record's feature columns are
   normalized with that entire record's mean and standard deviation before
   chunk creation and before train/validation/test splitting (lines 95–111).
   The public `load_data` path sets `normalize=True` unconditionally (lines
   660–675). Under the default `split_method=sample`, this makes validation
   and test distribution statistics available to training features. It is
   unsupervised future-distribution leakage, not target-label leakage, but it
   changes the information contract and must be disclosed.
2. The sample split partitions chunk indices 60/20/20 after chunk generation
   (lines 715–730). Each chunk spans history + prediction window and starts at
   a stride; no purge/embargo is applied to ensure a training target interval
   ends before the first validation forecast origin or a validation target
   ends before the first test origin. With overlapping chunks, target intervals
   can therefore overlap later partitions.

The code also removes chunks without history-window text even when
`enable_text=False` (lines 207–226). This affects sample selection and should
be disclosed when comparing numeric-only and multimodal modes. The text loader
sorts text.csv by `date_time` and filters text to the history window, but the
CSV does not expose a separate availability/arrival timestamp in the documented
schema; do not equate that field with actual system availability without
dataset-level evidence.

### Required dual reporting

- **Official reproduction:** reproduce the pinned harness and split/normalizer
  as released; label this result as the official-code path and record its
  full-record normalization/overlap behavior.
- **Leakage-controlled comparison:** fit all normalization statistics on the
  training portion only, apply the fitted transform unchanged to validation
  and test, and use the new purged chronological window splitter with
  train-target and validation-target boundary checks. Apply the same clean
  information contract to Symthaea and all comparators.

Do not silently replace the official protocol or present clean-split numbers
as official leaderboard results. The new splitter does not itself fix the
full-record normalizer; train-only scaler fitting remains an adapter task.


## 11. Clean-split scaler implementation

The event-contract crate now has a `TrainOnlyStandardizer` bound to the
purged split plan. It fits deterministic per-entity/per-feature moments only
from unique raw observation IDs represented in training windows. Observations
reused across overlapping training histories are counted once; conflicting
values for an identical raw-observation identity fail closed. Validation, test
and purged-window values are not read. It reports counts, means and sample
standard deviations (ddof=1), uses Welford accumulation, and refuses a forged
split plan whose per-entity boundary/purge assignment is inconsistent.

This is the clean-comparison preprocessing path. The official IMM-TSF path
still needs to be reproduced as released, including its full-record scaler,
then reported separately. The adapter must omit missing values from numeric
moments while retaining their explicit missingness mask. No imputation is
performed by this scaler.

Nine source-authored scaler tests cover validation/test isolation, overlap
deduplication, conflicting duplicate rejection, entity/feature separation,
constant/singleton handling, input-order determinism, split plan validation,
and non-finite values.


## 12. Concrete Time-IMM CSV adapter

The current Time-IMM repository (tree commit
`e934cef23d526817dcf0d1a66c6f82274c96c04b`) contains the EPA-Air/Bexar
per-entity files:
- `time_series.csv` blob `c12a34fa63f6fdab7b9608a3c581b0d929a2626e`,
  4,370 data rows, columns
  `date_time,record_id,temp,pm2_5,aqi,ozone`;
- `text.csv` blob `01840c5b62c699e2a3a9b5d6198067c383df8485`,
  18 data rows, columns `date_time,record_id,text`.

The visible date_time values are naive local wall-time strings such as
`2024-01-01 00:00:00`; no timezone or separate arrival timestamp is present
in these CSV rows. The new `research/irregular-event-contract/src/time_imm.rs`
adapter parses both schemas but requires the caller to specify an IANA/fixed
timezone and an explicit availability policy. It rejects ambiguous/nonexistent
DST times, records timezone and availability-policy digests, hashes exact
source files and canonical rows, preserves text payload whitespace, and returns
a quantization-error bound for numeric fixed-point conversion.

This makes dataset ingestion executable without falsely inferring physical
arrival time. The illustrative America/Chicago parser test is a timezone
mechanics test, not yet a certified production timezone declaration for the
dataset. The dataset-specific timezone must be verified before any real-data
claim. No Time-IMM dataset benchmark has been executed.


The canonical adapter-configuration digest is feature-order independent and
binds dataset ID, entity record ID, timezone rule, availability policy, and
every numeric column's modality ID, unit code and decimal scale. Both numeric
and text parsed-table results return this digest, the dataset/entity IDs, and
the source-file digest.
