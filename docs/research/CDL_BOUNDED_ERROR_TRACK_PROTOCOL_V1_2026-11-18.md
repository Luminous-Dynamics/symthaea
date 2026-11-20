# CDL-Bench Bounded-Error Numeric Track v1

**Registered:** 2026-11-18  
**Status:** protocol before confirmatory experiments; Rust reference implementation authored, uncompiled  
**Fidelity class:** lossy, with a strict per-sample absolute-error bound  
**No claim:** this is not an exact-lossless benchmark and does not constitute a current SOTA result.

## 1. Research question

Can a decoder-synchronous predictor, possibly conditioned on validated causal relationships across sensor channels, reduce the total bits required to represent numeric streams when the application supplies a fixed pointwise error tolerance?

For every sample \(x_t\), the decoded value \(\hat{x}_t\) must satisfy:

\[
|x_t-\hat{x}_t| \leq \tau_t
\]

The first reference implementation uses one constant integer tolerance \(\tau\) for an i16 stream. Per-channel and time-varying budgets are future extensions and must be explicitly serialized.

This track is separate from exact lossless compression. Results must never be combined into a single compression leaderboard.

## 2. Why this track exists

Cadence (Tacconelli, September 2026) demonstrates a meaningful difference between exact lossless and error-bounded predictive coding.

For lossless coding, better prediction accuracy usually changes residual code length logarithmically. A substantial forecasting improvement may therefore produce only a small bit reduction.

For bounded-error coding, when the forecast lies within the declared tolerance band, the residual index can be zero. That creates a threshold effect that can materially reduce entropy-coded residuals.

Cadence reports domain-localized gains on 2026 grid-demand and subway-ridership series but a negative result on SDRBench scientific data. This motivates explicit domain tests and negative controls, not a universal claim about numeric data.

Primary source:
- https://arxiv.org/abs/2609.06008
- https://github.com/robtacconelli/Cadence

## 3. Current reference implementation

Location: research/cdl-bench/src/bounded_error.rs

The prototype currently includes:
- signed i16 source values;
- a previous-reconstructed-value predictor;
- a linear predictor 2 * previous - previous_previous;
- quantized integer residual indices;
- order-0 and order-1 adaptive byte models;
- a separate 36-byte bounded-error frame;
- serialized tolerance, predictor ID, entropy-model ID, sample count, inner payload length, reconstructed-data checksum and inner-frame checksum;
- a deterministic decoder that consumes only reconstructed prior samples;
- a strict output-length limit and canonical u64 varint decoder.

### 3.1 Quantization rule

Set:

\[
q = 2\tau+1.
\]

For predictor value \(p_t\), choose the nearest integer index:

\[
k_t = \operatorname{round}\left(\frac{x_t-p_t}{q}\right)
\]

and reconstruct:

\[
\hat{x}_t =
\operatorname{clip}_{i16}(p_t+k_tq).
\]

Since the integer grid spacing is odd, nearest-grid reconstruction has error at most:

\[
\left\lfloor \frac{q}{2}\right\rfloor = \tau.
\]

Clipping to the i16 representable interval cannot increase the absolute error when the source is itself an i16 value.

**Encoder synchrony rule:** the encoder must advance the predictor state using reconstructed x-hat, not the original x. The decoder performs the same update from decoded values. Otherwise a small divergence can cause all subsequent predictions to desynchronize.

The encoder also checks the per-sample bound before emitting a frame. The decoder checksum protects deterministic reconstruction, but CRC-32 is accidental-corruption detection, not cryptographic authenticity.

## 4. Mandatory metrics

Every result must report:

1. **Exact tolerance:** integer units, physical units, and source scale.
2. **Maximum absolute error:** hard qualification metric; must be <= tau for every sample.
3. **MAE and RMSE:** diagnostic metrics only; neither substitutes for the maximum-error bound.
4. **Zero-index rate:** percentage of samples whose quantized residual index is zero.
5. **Residual-index distribution:** entropy, tails, and run statistics.
6. **End-to-end bits/value:** includes all frame metadata, bootstrap/context and adaptation bytes.
7. **Compression ratio:** always state whether denominator is raw typed values or another baseline.
8. **Encode/decode throughput and latency:** p50/p95/p99 at declared chunk sizes.
9. **Peak memory and model size:** measure rather than infer.
10. **Determinism:** exact repeated decode on the declared runtime/hardware; include any device, batch or precision details needed for decoder agreement.
11. **Abstentions/failures:** report every invalid, out-of-domain, or rejected stream.

The same entropy coder must be used for the predictor-comparison experiment. A second experiment may compare alternative coders, but must do so for every predictor arm.

## 5. Baselines

### Classical prediction + identical entropy coder
- persistence / previous value;
- linear prediction and local polynomial predictors;
- 32-tap linear prediction where appropriate;
- Lorenzo predictors for multidimensional scientific fields;
- interpolation and spline predictors;
- domain-specific predictors under identical tolerance and split.

### Published systems
- Cadence on its declared demand-series datasets and tolerances, where reproducible;
- SZ3 / Lorenzo-family predictors for scientific data;
- Pco, Sprintz, Gorilla and Chimp for their supported exact-lossless cases, reported as distinct fidelity baselines rather than as bounded-error competitors;
- Machete only under the exact published error semantics and metrics;
- WZS for non-stationary time-series streams under a matched lossless setup.

### Symthaea experimental arms
- previous-value baseline;
- linear baseline;
- HDC-only residual predictor;
- HDC + CfC/LTC predictor;
- causal cross-channel mechanism predictor;
- active tolerance-budget allocation (future).

All predictor arms use the same entropy coder, quantizer rule, test split, and declared observation history. Any pretrained model's context bootstrap, checkpoint bytes and execution configuration must be counted.

## 6. Dataset plan

The initial suite should include several distinct regimes, not only the domain where a model is expected to win.

| Domain | Candidate corpus | Purpose |
|---|---|---|
| Grid demand | EIA-930 2026 series | Test demand-localized forecasting and tolerance coding |
| Transit demand | MTA station/complex 2026 series | Test periodic human-mobility demand |
| Operational telemetry | NAB and frozen industrial traces | Test operational regime changes |
| Scientific fields | SDRBench | Negative-control domain for domain-localized demand models |
| Synthetic | controlled i16 sequences and E2 SCM outputs | Test mechanisms/interventions with known ground truth |
| Physical sensors | provenance-backed Symtropy/robotics traces | Test eventual transfer beyond public demand datasets |

For every corpus, record license, raw hash, sample cadence, unit, preprocessing, missingness policy, normalization, split dates, and model-training contamination review.

Split by complete series, site, temporal period, and intervention regime. Do not randomly divide adjacent time points from a single trajectory.

## 7. Causal extension hypothesis

A causal predictor may condition one channel's reconstruction on other channels through a typed mechanism library.

Example: motor torque, acceleration and current may jointly predict a downstream vibration measurement. A validated mechanism may reduce the residual index for vibration without forcing the system to learn every sensor channel independently.

The causal arm must be tested with:
- baseline observations;
- do-interventions on an upstream channel;
- interventions on a downstream channel;
- paired no-intervention controls;
- held-out mechanism parameters;
- held-out channel combinations;
- multiple stream schedules and regime changes.

The evaluator should check whether the model correctly changes predictions for descendants while preserving unaffected channels. A within-arm correlation or improvement on one fixed schedule is not evidence of causal transfer.

## 8. Preregistered targets

All are research targets, not present results.

### Hard integrity gate
- 100% of decoded samples satisfy the declared pointwise error bound.
- 100% exact deterministic reconstruction of the encoded approximation from the same frame/configuration.
- malformed/truncated frames reject.
- no silent output-length or model-version mismatch.

### Compression gate
Target at least **10% lower end-to-end bytes** than the strongest reproduced classical error-bounded predictor on at least one preregistered domain family, with paired 95% confidence interval excluding zero.

A stronger result would reproduce across two distinct domain families. The team must also report at least one negative-control domain, and retain negative outcomes.

### Resource gate
Publish a Pareto frontier for end-to-end bytes, encode/decode throughput, peak RAM, model/bootstrap size, and adaptation cost. No universal win is claimed if the method only compresses better by becoming prohibitively slower or larger.

### Causal-transfer gate
Target at least 20% fewer adaptation observations or adaptation bits to recover a preregistered predictive-quality threshold after held-out interventions, compared with the strongest matched non-causal predictor. Use at least 20 independent generator seeds, multiple intervention patterns, and paired controls.

## 9. Immediate next work

1. Run unit tests for the reference Rust crate in an equipped Rust toolchain; source presence is not passing-test evidence.
2. Cross-check the quantizer with an independent implementation, including both i16 extremes, zero tolerance and maximum tolerance.
3. Add a typed floating-point wrapper only after i16 behavior is validated; preserve exact IEEE metadata and specify NaN, infinity, signed-zero and timestamp semantics.
4. Reproduce Cadence's baseline protocol on its declared data/tolerance before making a comparison claim.
5. Add Pco and TSCom-Bench baselines to the exact-lossless track.
6. Add a per-channel causal predictor only after the single-channel bounded-error baseline is stable.
7. Keep all approximate coding isolated from live Symthaea memory, safety and evidence gates.

## 10. Claims we will not make

- Bounded-error compression is not lossless compression.
- A forecast with lower MAE does not automatically yield materially fewer exact-lossless bits.
- Cadence's demand-series gains do not imply gains on scientific or arbitrary sensor streams.
- A synthetic demand-like generator is not an EIA, MTA, or real industrial dataset.
- A source-level prototype is not a benchmark result.
- A successful error bound is not proof that the output is safe for every downstream engineering use; application-specific error budgets and validation are required.
