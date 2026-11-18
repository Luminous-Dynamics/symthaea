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
