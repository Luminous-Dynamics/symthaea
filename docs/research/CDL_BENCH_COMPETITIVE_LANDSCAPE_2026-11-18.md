# CDL-Bench Competitive Landscape and Benchmark Amendment

**Date:** 2026-11-18  
**Status:** Research addendum; no benchmark result is claimed  
**Applies to:** CDL-Bench v1 protocol and standalone codec prototype

## Executive correction

CDL-Bench must not treat general-purpose codecs as its only serious competition. By November 2026, there is an established and actively evolving field of lossless numerical/time-series compression, neural time-series compression, and non-stationary stream adaptation.

The 5% total-size improvement in the initial CDL-Bench protocol is retained as a **pilot/entry gate**, not a final SOTA ambition. It is already comparable to a reported average 5.1% compression-ratio gain from SEP, and SEP reports 7.6% with pretraining on its evaluated datasets. Before any confirmatory experiment, the team should update the final superiority target and matched baseline matrix as specified below.

This addendum does not alter or retract the existing Symthaea C1/C2/C3 results. It updates the external competitive frame for the new benchmark.

## 1. Relevant external work

### 1.1 TSCom-Bench — lossless compression as time-series model evaluation

Wan et al., *Lossless Compression: A New Benchmark for Time Series Model Evaluation* (2025), introduces TSCom-Bench and frames entropy-coded lossless compression as a measure of full distributional modelling, complementing forecasting, anomaly detection, imputation, and classification.

The paper evaluates time-series backbones including TimeXer, iTransformer, PatchTST, Autoformer, DLinear, LightTS, SCINet and Informer. Its listed datasets include PEMS08, Traffic, Electricity, Weather, ETTh2 and Solar, alongside standard Enwik9, image, audio, float, Silesia and backup corpora. It describes a shared arithmetic-coding path and reports bits per byte, compression ratio and compression throughput. Its protocol uses designated train/test splits and explicitly disallows adaptive coding across train and test.

References:
- https://arxiv.org/abs/2509.21002
- https://anonymous.4open.science/r/TSCom-Bench-8262

**CDL-Bench consequence:** add an offline held-out TSCom-style track alongside the online/prequential track. Do not compare their numbers as if the protocols were identical.

### 1.2 SEP — neural lossless compression with semantic enhancement and multi-stream execution

Wan et al., *SEP: A General Lossless Compression Framework with Semantics Enhancement and Multi-Stream Pipelines* (IJCAI 2025), reports evaluations on seven real-world datasets. The paper reports an average compression-ratio gain of 5.1% over its evaluated SOTA compressors, rising to 7.6% with pretrained models; it also reports an average speed improvement of 30.0% and an average GPU-memory-footprint reduction of 36.2% (maximum 63.1%).

SEP targets correlations in binary byte-stream patches and uses GPU/multi-core streams plus shared memory pooling. Its reported results include Enwik9, book, sound, image, float, Silesia and backup datasets.

Reference:
- https://ijcai.org/proceedings/2025/370
- https://github.com/damonwan1/SEP

**CDL-Bench consequence:** SEP must be in the comparison matrix when its implementation and evaluation setup can be reproduced. Match hardware and include model/checkpoint, training, decoder and side-information accounting. A small Rust codec beating Zstd alone is not evidence of superiority over neural lossless pipelines.

### 1.3 Pco — specialized lossless numerical compression

Pco is a maintained Rust codec for numerical sequences, including integer and floating-point types, with use cases in columnar storage, long-term time series, tensors and edge delivery. Its pipeline:
1. detects numerical modes and splits values into latent variables;
2. chooses delta, delta-of-delta, lookback or linear-prediction residual transforms where useful;
3. entropy-codes approximate bins together with exact offsets.

The Pco project warns that semantically different sequences should not be mixed into one chunk and notes that larger chunks (over roughly 10k values) generally compress better. Its published benchmark page reports, for example, a NYC taxi numerical dataset, Pco ratios of 6.89 at level 8 and 6.98 at level 12, compared with 5.32 for Parquet+Zstd at Zstd level 22 and 2.85 for Blosc+Zstd at its stated max level. These figures use the project’s specified dataset preparation and a single performance core on a MacBook Pro M3 Max; they are not universal ratios.

References:
- https://github.com/pcodec/pcodec
- https://github.com/pcodec/pcodec/blob/main/docs/benchmark_results.md
- https://arxiv.org/abs/2502.06112

**CDL-Bench consequence:** Pco is a mandatory numerical-data baseline for typed numeric streams. Preserve per-channel separation and test wrappers for timestamps, missingness, units and random access. Do not flatten a multivariate stream in a way that destroys the per-column structure Pco expects.

### 1.4 Wave Zero Streaming — non-stationary streaming adaptation

Wang et al., *Waved Zero Streaming: A Time Series Compression Framework with Adaptive Segmentation Optimization* (Information Processing & Management, September 2026), describes a lossless streaming framework for non-stationary time series. Its indexed abstract and methodology describe:
- adaptive segmentation based on local fluctuation / leading-zero-variance behaviour;
- local parameter optimization;
- dynamic selection among compression algorithms for streams;
- evaluation across 43 heterogeneous datasets.

The source currently available for this research pass exposes the abstract and section summaries, not a fully inspectable result table. No numerical performance claim is adopted here.

References:
- https://www.sciencedirect.com/science/article/pii/S0306457326000889
- https://eurekamag.com/research/104/828/104828221.php

**CDL-Bench consequence:** add non-stationary regime-change and streaming tests. WZS should be a reproduction target when its implementation and exact configuration are available.

### 1.5 Sprintz, Gorilla and Chimp — edge and time-series baselines

Sprintz was designed for IoT sensor streams with low memory and latency; its paper reports less than 1 KB of memory and virtually no added latency, with a high-throughput decompression path. It is an important edge baseline even though it predates the recent neural-compression work.

Gorilla and Chimp are lossless floating-point time-series codecs based on XOR differences and leading/trailing-zero handling. They are relevant for short slices and fast decoding.

References:
- Sprintz: https://arxiv.org/abs/1808.02515
- Time-series compression survey: https://doi.org/10.1145/3560814
- Gorilla / Chimp discussion: https://doi.org/10.1145/3767158

**CDL-Bench consequence:** evaluate separate edge operating points with explicit memory ceilings, and include Gorilla/Chimp for timestamped floating-point traces.

### 1.6 Lossy compressors are a separate track

Machete (ACM TODS 2025) reports strong results for time-series databases under bounded-error lossy compression. That is useful for a future approximate-reconstruction programme, but it is not a direct competitor to CDL-Bench's exact byte-for-byte contract.

Reference:
- https://doi.org/10.1145/3767158

Do not combine lossless and lossy scores in one leaderboard. If an approximate track is added, report error bounds, task utility, pointwise worst-case error, and reconstruction metrics independently.

### 1.7 Online expert mixing and context adaptation are established directions

Two recent research directions make it important not to claim novelty for "adaptive expert selection" or "change-aware context" in isolation.

Huang, Ma and Michailidis, *Model-Agnostic Online Certificate-Driven Calibration for Time Series Forecasting Under Distribution Shift* (UAI 2026 / PMLR 337), develops an online martingale PAC-Bayesian certificate for temporally dependent, non-stationary forecasting. Its predictor uses a gated residual Bayesian head over a fixed backbone, follows predict-then-update ordering, and falls back to the backbone when the gate is closed. The key lesson for CDL-Bench is that online adaptation should expose uncertainty and abstention/fallback behaviour, and should be evaluated under temporal dependence rather than relying on IID assumptions.

Reference:
- https://proceedings.mlr.press/v337/huang26b.html

Du and Tang, *Breaking Context Inertia: Adaptive Context Management for Time Series Foundation Models in Non-Stationary Environments* (Neurocomputing, 1 October 2026), addresses stale historical context after distribution shifts in time-series foundation models. Its stated motivation is that old observations can contaminate predictions after a regime change. This overlaps directly with CDL-Bench's proposed fast/slow context memory and change-aware reset policy.

Reference:
- https://www.sciencedirect.com/science/article/pii/S0925231226015274

WZS (Section 1.4) already describes adaptive segmentation and online compression-algorithm selection in non-stationary streams. Taken together, these works mean:
- fixed-share Bayesian/online expert mixtures are a baseline family, not a novelty claim;
- change-point detection, context trimming, recent/global buffers and gated fallbacks must be compared as serious alternatives;
- the potential contribution has to be a *compression-specific* result: deterministic decoder-synchronous expert updates, exact coded bits under a shared coder, no transmitted expert-choice side channel, full model/state accounting, and controlled interventional evaluation.

A further compression-adaptation point is that **segmentation itself is an industry baseline**, not merely a side feature. The WZS paper's abstract reports results on 43 datasets and author-reported mean compression-ratio improvements of 36.41% over its compared lossless time-series algorithms, 13.28% over Camel, and 15.07% over general-purpose compressors, with separate latency claims. These figures are the paper's own results and must not be treated as independently reproduced here. They raise the bar materially: CDL-Bench must compare adaptive segmentation, algorithm selection and per-segment restart costs, not only a monolithic adaptive predictor.

The research branch now also includes `ScmChangePointMixturePredictor` (model 8): an integer Q8 surprisal CUSUM, selective reset of fast conditional contexts, retained slow base histograms, and deterministic re-priming toward the temporal expert. It adds no transmitted reset markers. The controlled C++ detector check has passed selected stationary / step-shift / gradual-drift scenarios; SCM stream coding, Rust tests and bit savings remain unverified. See:
- `docs/research/CDL_CHANGEPOINT_CUSUM_V1_2026-11-19.md`
- `research/cdl-bench/reference/cusum_detector_reference.cpp`
- issue [CDL-Bench-006 #12873](https://github.com/Luminous-Dynamics/symthaea/issues/12873)

Further relevant work:
- `A Closer Look at Adaptive Regret` (JMLR 2016) analyses fixed-share and its adaptive-regret properties; fixed-share is an established tracking baseline, not a new invention. https://www.jmlr.org/papers/volume17/13-533.html
- `Strongly Adaptive Online Learning` develops regret guarantees over sub-intervals, motivating per-regime rather than only whole-stream performance. https://proceedings.mlr.press/v37/daniely15.pdf
- Huang et al. (UAI/PMLR 2026) use an online martingale certificate to gate a residual Bayesian correction and fall back to a fixed forecaster when closed. https://proceedings.mlr.press/v337/huang26b.html
- Wang et al.'s WZS (September 2026) integrates adaptive segmentation, parameter optimization and stream-aware algorithm selection across 43 datasets. https://www.sciencedirect.com/science/article/pii/S0306457326000889
- The recent hierarchical model-level adaptation work describes frozen slow priors, fixed intermediate blending and a fast recent-buffer correction. https://www.sciencedirect.com/science/article/pii/S0893608026012578

These works make the next comparison matrix explicit:
1. static per-stream models;
2. fixed-share mixture (model 7);
3. surprisal-triggered fast-context reset (model 8);
4. strongly adaptive interval/specialist experts;
5. explicit segmentation with change-point markers included in the stream;
6. WZS-style selector/segment strategy where reproducible;
7. best numerical and general-purpose codecs.

For the coding-specific comparison, all arms must use one shared entropy coder and the exact same source bytes. If reset boundaries are inferred from decoded history, they may require no side-channel bits but still incur compute and state costs. If a baseline transmits segment markers, those markers must be counted. Report realized coded bits, not only predictive log loss.

The CDL-Bench research branch now includes `ScmExpertMixturePredictor` (fixed-share mixture of typed-field, lag and known-parent experts) and a separate reference report:
- `docs/research/CDL_EXPERT_MIXTURE_FIXED_SHARE_V1_2026-11-18.md`

Its 40-stream public-development reference simulation shows a very small overall ideal-code-cost improvement versus lag, but a small regression on intervention streams. We therefore treat expert mixing as an adaptation baseline and falsification tool, not as a claimed causal-compression breakthrough. The report clearly separates ideal prequential bits from actual arithmetic-coded frame measurements.

## 2. Revised benchmark matrix

CDL-Bench should publish three compatible but separate tracks.

### Track A — Offline held-out time-series model compression

- Follow TSCom-Bench's train/test separation and entropy-coder interface.
- Compare TimeXer, iTransformer, PatchTST and relevant baselines where reproducible.
- Report bits/byte, compression ratio and throughput.
- Count the model checkpoint and decoder in cold-start accounting; show amortized deployment cost separately.

### Track B — Online / prequential adaptation

- Predict and code before revealing each new observation.
- Adapt only after decoding/observing the symbol.
- Test change points, schedule shifts, varying sensor noise and interventions.
- Compare fixed models, adaptive byte models, Sprintz-like online predictors and the Symthaea arms.
- Count every update, side channel, state reset and adaptation payload.

### Track C — Causal / interventional mechanism transfer

- Use the E1 finite-state and E2 structural causal generators.
- Train on base mechanisms; hold out mechanism recombinations and intervention values.
- Compare no-recall, nearest-neighbour recall, temporal-state, and typed mechanism-library arms.
- Measure both predictive code length and adaptation observations/bits to a preregistered quality threshold.
- Require paired counterfactuals and multiple schedules; no within-arm correlation as causal evidence.

## 3. Revised architecture: a reversible transform and mechanism pipeline

The standalone adaptive byte arithmetic coder is a correctness baseline only. It is not expected to compete with mature numerical codecs by itself.

The next compression layer should be:

1. **Typed stream schema:** channel identity, integer/float type, units, timestamps, missingness and sample cadence.
2. **Reversible transform selection:** raw representation, delta, delta-of-delta, XOR, bitshuffle, frame-of-reference or linear-prediction residuals where mathematically exact.
3. **Adaptive segmentation:** isolate stationary and volatile regimes; reset predictor state only at explicit, encoded boundaries.
4. **Causal cross-channel prediction:** use validated parent mechanisms to predict downstream channels; encode exact residuals and all mechanism identifiers/parameters.
5. **Shared entropy coding:** one verified coder interface consumes integer frequencies or a stable discretized distribution.
6. **Framing and integrity:** random-access chunk boundaries, checksums, schema/version, model digest, and deterministic decoder replay.

For every candidate transform/model, compare the extra metadata/model bits against the residual-bit savings. A more sophisticated causal model is rejected if it increases total code length or adaptation cost without improving held-out interventional quality.

## 4. Baseline and dataset plan

### Mandatory general-purpose codecs
- raw;
- Zstandard (multiple documented levels);
- Brotli;
- XZ/LZMA2;
- gzip as a lightweight reference.

### Mandatory numeric/time-series codecs
- Pco;
- Sprintz;
- Gorilla;
- Chimp;
- delta / delta-of-delta / XOR / bitshuffle pipelines;
- TSCom-Bench reference arithmetic-coding models and available pretrained backbones;
- SEP where an exact reproducible implementation and model configuration are available;
- WZS when the implementation and parameters can be reproduced.

### Dataset families
1. TSCom-Bench datasets: PEMS08, Traffic, Electricity, Weather, ETTh2, Solar.
2. Pco numerical tables: air quality, NYC taxi and r/place numeric columns, under their published preparation rules.
3. Silesia / Enwik9 / sound / image / float / backup for general-domain controls.
4. Symtropy and robotics sensor traces with provenance: position, velocity, force, current, torque, temperature, vibration and intervention events.
5. E1/E2 synthetic environments with hidden states and intervention manifests.

Keep splits by environment, generator seed, mechanism family and intervention, not random neighbouring timesteps. Preserve source hashes, data licenses, channel order and exact preprocessing manifests.

## 5. Revised success ambition

The initial 5% improvement gate in CDL-Bench v1 remains a **pilot signal**, not an adequate final SOTA claim in time-series compression.

Before confirmatory experiments, preregister a stronger, domain-specific target such as:
- at least 8% lower **total** encoded bits than the strongest reproduced matched baseline on at least two declared sensor/time-series families, or at least 12% on one narrow, clearly scoped family;
- paired 95% confidence interval excluding zero;
- exact reconstruction;
- model, decoder, transform metadata, intervention metadata, initial state and adaptation payload all counted;
- no more than a 2% regression in total size on the other designated target families, unless the result is explicitly presented as a Pareto trade-off;
- throughput and memory reported on identical hardware; publish the Pareto frontier rather than hiding a slower or larger model behind one ratio.

For the causal track, add an independent target:
- at least 20% fewer adaptation observations or adaptation bits to reach the same preregistered prediction-quality threshold on held-out interventions, compared with the strongest matched non-causal baseline;
- 20 or more independent generator seeds, multiple schedule interleavings, paired interventions and raw per-seed results.

These are research targets, not current results. They should be finalized before the first confirmatory run and should not be retrofitted to early smoke numbers.

## 6. Immediate engineering implications

1. Keep the current order-0/order-1 arithmetic coder as the codec integrity reference; do not optimize it prematurely.
2. Add typed numerical stream records and reversible transforms before any Symthaea predictor integration.
3. Add Pco CLI/API and TSCom-Bench reference comparisons as reproducible optional baselines; record exact versions and preparation commands.
4. Add an adaptive segmentation arm, beginning with a simple change-point detector before genetic or expensive search.
5. Preserve E1/E2 generators, and add multiple SCM parameterizations and graph topologies before making causal generalization claims.
6. Implement a typed causal mechanism library only after the non-causal numeric baselines and bit accounting are solid.
7. Retain all original C1/C2/C3 findings and their caveats. The new benchmark must be judged on actual bitstreams and held-out causal transfer, not on earlier normalized cosine diagnostics.

**Evidence boundary:** this document is a literature-grounded competitive amendment. It contains no new compression run, no reproduced industry result, and no claim that CDL-Bench currently beats any baseline.


## 7. New strategic result: lossless and bounded-error prediction have different economics

Cadence (Tacconelli, 2026-09-05), *Error-Bounded Lossy Compression of Demand Time Series with a Time-Series Foundation Model*, should materially change how CDL-Bench allocates effort.

Its central quantitative observation is that when a well-matched residual model's code length scales with residual magnitude, improving a predictor transfers only logarithmically into lossless bit savings:

[
\Delta b \approx \log_2\left(\frac{\operatorname{MAE}_{old}}{\operatorname{MAE}_{new}}\right).
]

The paper reports a TimesFM-3 versus linear-predictor accuracy ratio of about 1.51x but only 0.60 bits out of 20.28, approximately 2.9% on that example. It argues that a 2x reduction in a 20-bit-per-value lossless code would require an implausibly large predictor improvement. This is a warning against assuming that a better forecast automatically creates a large lossless compression win.

Cadence instead uses an error-bounded residual quantizer. When the forecast is within a declared tolerance band, the residual index is zero, changing the coding economics discontinuously. The paper reports:
- 13.3% median gains over its best classical error-bounded predictor on 49 EIA-930 grid-load series at tolerance 0.05;
- 28.3% on 50 MTA subway ridership series at the corresponding reported tolerance;
- 21.4% median across 297 demand-series/tolerance pairs in its final abstract;
- a negative result of -0.8% median and 0/27 gains on SDRBench scientific-field slices.

The last result is as important as the gains: the advantage is localized to the tested aggregate human-demand domain, not evidence that the approach improves generic numerical data.

Cadence further reports an encoder/decoder determinism issue: neural predictions varied with batch size and execution configuration, creating a desynchronization risk. Its container therefore records group size and device/execution configuration. CDL-Bench should adopt the general lesson: **the decoder's entire prediction-state evolution must be reproducible from the container and decoded prefix, not merely statistically similar.**

Primary sources:
- Paper: https://arxiv.org/abs/2609.06008
- Code and results: https://github.com/robtacconelli/Cadence
- Experiments log: https://github.com/robtacconelli/Cadence/blob/main/EXPERIMENTS.md

### 7.1 Strategic change

Keep the exact-lossless track, but do not expect a large foundation model's modest forecasting gain to yield a proportionate compression gain. On structured numeric streams, compare small residual models carefully and require a real, shared entropy coder.

Add a separate **bounded-error numeric track** with an explicit per-sample guarantee:

[
|x_t-\hat{x}_t|\leq\tau
\quad\text{for every sample }t.
]

This track must not share a leaderboard or headline score with exact lossless compression. It must report the exact tolerance, maximum observed error, error distribution, residual-code bits, model/bootstrap bytes, decoder configuration, and end-to-end size.

### 7.2 What CDL-Bench can test beyond Cadence

Cadence has already demonstrated that error-bounded predictive coding can be effective in a domain-localized demand regime. We should not claim novelty for “forecast plus tolerance quantization” alone.

Our candidate research contribution is to test:
1. **Mechanism-conditioned bounded-error coding:** causal multi-channel mechanisms predict the current series; encode quantized residuals only when required.
2. **Cross-channel interventions:** change one upstream variable and test whether the model preserves unrelated channels while adapting descendants.
3. **Per-channel error budgets:** allocate a global application error budget according to measurement units, risk, and downstream utility, while preserving hard per-sample limits.
4. **Regime-adaptive, decoder-synchronous coding:** segment non-stationary streams only when the segmentation decision can be reproduced exactly by the decoder or is explicitly coded.
5. **Active tolerance selection:** choose a tolerance from an application-specified error budget, never silently increase it to improve compression.
6. **Domain-localization and falsification:** test demand series, telemetry, sensor streams, and scientific fields separately; expect domain failures and retain them.

### 7.3 Revised SOTA comparisons

| Track | Required competitive baselines | What must be matched |
|---|---|---|
| Exact integer lossless | Pco, Sprintz, Gorilla/Chimp where applicable, Zstd, TSCom-Bench/SEP for appropriate corpora | Same input values, column layout, units/metadata, train/test rules, hardware |
| Exact float lossless | Gorilla, Chimp, Elf, Pco where supported, Zstd; SEP for general byte streams | Exact IEEE bit patterns, NaN payload policy, signed zero, timestamp layout |
| Error-bounded numeric | Cadence, SZ3/Lorenzo, interpolation/LPC predictors, Machete only under its published error semantics, plus domain baselines | Same pointwise tolerance, data split, predictor budget, entropy coder and context bootstrap |
| Non-stationary streaming | WZS, adaptive segmentation, Zstd stream mode, Sprintz/edge baselines | Same change points, latency budget, chunk reset semantics and memory cap |
| Causal transfer | CDL E1/E2 paired mechanism interventions, plus CausalWorld for embodied external evaluation | Same observations, interventions, online information, compute and adaptation budget |

### 7.4 Revised success targets

For **exact lossless** coding:
- retain 100% byte/value-exact round-trip as a hard requirement;
- a small predictor-only improvement is not success unless total coded size, after all metadata, is materially better;
- use a domain-targeted total-size target and report the Pareto frontier instead of optimizing a generic average.

For **bounded-error numeric coding**:
- target at least 10% lower end-to-end bytes than the strongest reproduced classical error-bounded predictor on a preregistered domain family, with paired confidence intervals excluding zero;
- enforce the exact per-sample tolerance for 100% of reconstructed samples;
- include all model bootstrap/context, metadata and per-block state costs;
- require a held-out domain and an explicit negative-control domain;
- report results separately by tolerance level and stream length.

For **causal multi-channel coding**:
- target a measurable improvement in both total residual bits and adaptation cost after interventions;
- require held-out structural mechanisms and combinations, not merely a new random seed;
- never attribute a gain to causal reasoning unless the mechanism arm beats matched non-causal predictors under the same entropy coder and information budget.

These are targets for future registered experiments, not current results.

## 8. SEP, WZS and the numerical baseline implications

SEP (IJCAI 2025) reports 5.1% average compression-ratio improvement, 7.6% with pretraining, 30.0% average speed improvement, and 36.2% average GPU memory-footprint reduction on its seven tested corpora under its A800-based evaluation. It is a strong general lossless neural compression comparison, although its framework and compute scale differ substantially from this small Rust prototype.

WZS (Information Processing & Management, September 2026) reports evaluation over 43 datasets and claims a 36.41% average compression-ratio improvement, 13.28% over Camel, and 15.07% over general compression methods in its experimental setup. It is especially relevant to non-stationary streams and adaptive segmentation. These are author-reported results and should be independently reproduced before being adopted as expected performance.

Pco's 2025 paper describes a dedicated numerical format using mode decomposition and delta encoding before a binning-based entropy model; it reports 29–94% higher compression ratio than compared approaches on six real-world datasets. Pco's documentation warns against mixing semantically different sequences into a single chunk and recommends sufficiently large chunks for good compression.

Primary sources:
- SEP: https://ijcai.org/proceedings/2025/370
- WZS: https://www.sciencedirect.com/science/article/pii/S0306457326000889
- Pco paper: https://arxiv.org/abs/2502.06112
- Pco implementation and format: https://github.com/pcodec/pcodec

### 8.1 Engineering consequence

Do not replace the current small arithmetic coder. Retain it as a transparent reference backend. Build and compare typed transforms and numeric codecs as separate baselines. A serious competitive result must include Pco and the relevant domain-specialist baselines, not only raw bytes and Zstandard.


## 9. Additional 2026 neural lossless baselines: Nacrith and BOA Constrictor

The benchmark matrix should include two more strong architecture-level comparisons.

### 9.1 Nacrith — high-precision transformer/context ensemble

Tacconelli's *Nacrith: Neural Lossless Compression via Ensemble Context Modeling and High-Precision CDF Coding* (February 2026) combines:
- a 135M-parameter SmolLM2 transformer;
- an online token-level N-gram predictor;
- adaptive context mixing and a document-level log-space bias head;
- a 24-bit CDF precision upgrade;
- a 32-bit arithmetic coder;
- a hybrid binary container and optimized inference backend.

The paper reports 0.9389 bits/byte on enwik8 and 0.918 bits/byte on alice29.txt, and 0.723 bits/byte on a post-training-cutoff document. These are author-reported natural-language results, not time-series or causal-transfer results. Its core lesson for CDL-Bench is that the prediction model is only one component: probability precision, context mixing, online adaptation, chunk format and inference/decoder determinism materially affect the end result.

References:
- https://arxiv.org/abs/2602.19626
- https://github.com/robtacconelli/Nacrith-GPU

### 9.2 BOA Constrictor — state-space models for scientific compression

Gupta, Doglioni and Elliott's *BOA Constrictor: A Mamba-based lossless compressor for scientific data* (published May 12, 2026) is a particularly relevant comparison for Symthaea's CfC/LTC direction. It uses bytewise autoregressive Mamba prediction with a streaming range coder.

The 2026 paper reports a 2.21 MB model and model-inclusive effective compression ratios of 7.23x on ATLAS Open Data (HDF5) and 9.13x on simulated HepMC v3 particle-collision records, exceeding the next-best traditional algorithms it compared against on those datasets. The authors also discuss data-format dependence and throughput trade-offs.

These results make BOA a direct baseline for any claim that a continuous-time / liquid state predictor offers a useful alternative to modern state-space sequence models on structured scientific streams.

References:
- https://doi.org/10.1088/2632-2153/ae64a9
- https://github.com/boa-collaboration/boa-constrictor

### 9.3 Updated architecture comparison

| Architecture | Relevant strength | What Symthaea must demonstrate beyond it |
|---|---|---|
| Nacrith / transformer + online context mixer | Strong textual next-symbol modelling, high-precision CDF and online adaptation | Causal mechanism reuse, cross-channel transfer, lower resource cost on the chosen sensor domain |
| BOA / Mamba + range coder | State-space sequence model for structured scientific bytes | Measurable benefit from HDC/CfC dynamics or factorized causal mechanisms on held-out interventions |
| SEP / patch semantics + neural backbones + multi-stream execution | Semantic patching and hardware throughput/memory optimization | Equivalent throughput/resource discipline and a stronger causal/typed-stream advantage |
| WZS / adaptive segmentation + local method selection | Non-stationary stream handling | Reproducible regime detection and decoder-synchronous method switching |
| Cadence / forecast plus bounded-error residual coding | Strong domain-localized pointwise-error compression | A distinct causal multi-channel or application-budget advantage under the same tolerances |

### 9.4 Revised experimental design

The CDL-Bench model matrix must now add:
- a compact Mamba/SSM predictor when a reproducible model and decoder are available;
- a transformer + online context-mixing predictor for text/byte-stream control;
- a context-only / N-gram / PAQ-like statistical control;
- matched adaptive CDF precision and entropy-coder ablations;
- a decoder-determinism test across chunk sizes, thread counts, devices and floating-point configurations.

The comparison must not claim that Symthaea beats Mamba because a hand-built synthetic SCM is easier for its generator, or that it beats a 135M transformer because its checkpoint is not charged. Every model pays for its checkpoint, decoder, context bootstrap, side information and resource footprint in the declared accounting.

A credible near-term target is not universal byte-compression superiority. It is a repeatable Pareto advantage on a tightly specified family of interventional multi-sensor streams: total bits, adaptation observations/bits, memory, latency and exact or bounded-error fidelity all reported separately.

**Evidence boundary:** the figures above are the cited authors' reported results and have not been independently reproduced by CDL-Bench.


## 10. Adjacent causal-compression and causal-representation work

The novelty claim must be narrower than “causal compression” or “causal time-series learning.” Both areas have active 2026 work.

### 10.1 CSCProv — causal-semantic provenance graph reduction for security

CSCProv (published 17 September 2026) compresses provenance graphs by aggregating nodes only when structural dependency patterns and behavioural semantics remain consistent. It introduces a Causal-Semantic Consistency metric and reports up to 30× graph reduction on DARPA Transparent Computing Engagement 3 while preserving attack-critical dependencies; it reports downstream detection-time reductions of 32.7–48.3%.

This is graph summarization for security analytics, not a lossless byte-stream codec. It nevertheless establishes an important design principle: compressibility must be constrained by preservation of task-critical causal paths.

CDL-Bench security extension should therefore preserve attack-propagation dependencies, not merely byte equality or generic graph similarity, when evaluating compressed audit/provenance views.

Reference:
- https://link.springer.com/article/10.1186/s42400-026-00648-6

### 10.2 CDTS² — causally disentangled time-series representations

CDTS² (Applied Sciences, 10 September 2026) separates trend and seasonality representations, uses HSIC regularization and a causal-discovery objective, and evaluates counterfactual prediction under time-varying confounding. The paper reports RMSE improvements of 11.6% on a tumor-growth dataset and 17.1% on CityLearn under severe confounding, plus improved interventional-robustness and disentanglement metrics.

This is causal representation learning and counterfactual forecasting, not a byte-level compression system. It is a useful representation/model baseline for CDL's causal-transfer track. It also suggests that simply splitting latent vectors into trend and seasonality is not sufficient; we should test whether interventions on one factor leave unrelated factor representations invariant.

Reference:
- https://www.mdpi.com/2076-3417/16/18/8993

### 10.3 MDL-guided regime-specific causal discovery

A 2026 industrial time-series diagnostic protocol for non-stationary processes uses an MDL objective to jointly identify change points and regime-specific graph structure, then estimates interventional effects with ancestral support gating. Its evaluation uses the Tennessee Eastman process benchmark.

This is directly relevant to CDL's regime segmentation and causal mechanism validity domains. A mechanism should not be treated as globally stable if the generator or real process has changed regime.

Reference:
- https://onlinelibrary.wiley.com/doi/10.1155/int/3436744

### 10.4 Revised novelty boundary

The proposed CDL contribution should be described as a research hypothesis with these combined requirements:

1. Actual data coding: emit a deterministic bitstream and reconstruct the exact source or satisfy a separately declared pointwise error bound.
2. Typed multichannel semantics: maintain units, channel identity, timing, missingness and provenance.
3. Causal mechanism reuse: factorize mechanisms and test unseen combinations rather than only new seeds.
4. Intervention-aware residual coding: after an intervention do(X=x), update affected descendants while preserving independent mechanisms; encode the remaining residual.
5. Decoder synchronization: every predictor/model update must be reproducible from the frame and decoded prefix, including numerical execution configuration when necessary.
6. Application-critical invariants: in a security graph, preserve attack paths; in a physical system, preserve safety-critical event and extrema constraints; in energy telemetry, preserve declared peak/error requirements.

The combination may be differentiated, but it is not proven novel or superior merely because the pieces are assembled. A formal related-work review and matched implementation comparisons are required before making a publication or SOTA claim.
