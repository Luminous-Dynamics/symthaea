# CDL-Bench decoder-synchronous change-point adaptation v1

**Date:** 2026-11-19  
**Status:** Rust predictor and tests authored; independent C++ detector micro-reference compiled and executed; stream-level Rust validation and actual coded-size measurements pending.  
**Scope:** SCM2 public development fixtures and controlled surprisal sequences. No blind evaluation result is claimed.

## Research question

Can a lossless stream codec detect that its current predictive context has become stale, reset only the fast part of its model, and recover without transmitting a reset marker or receiving intervention labels?

The existing fixed-share expert mixture (model 7) already shifts probability toward its noncausal lag expert when the parent expert stops predicting well. This follow-on asks whether explicit **change-point-triggered fast-context reset** improves adaptation after a sustained predictive surprise increase, while preserving the slow global statistics.

## Implemented model 8

`ScmChangePointMixturePredictor` combines the same three specialists as model 7:
- typed field/lane base distribution;
- same-channel temporal lag context;
- known-parent context (oracle topology; graph is supplied, not learned).

The encoder and decoder run exactly the same operations:
1. Build the current mixture CDF from the decoded prefix.
2. Decode the next byte.
3. Reconstruct its prequential surprisal from the CDF interval width.
4. Update the CUSUM detector from that surprisal.
5. If a change alarm fires, reset only conditional context histograms and re-prime expert weights toward lag.
6. Update shared statistics with the decoded byte.

No intervention label, boundary, reset bit, or expert-selection bit is transmitted. The detector state is reproducible from the decoded prefix.

### Fast / slow separation

On alarm:
- Retain the header and field/lane base histograms as the slow global memory.
- Reset lag and parent conditional histograms to their deterministic initial counts.
- Re-prime expert weights to 25% field / 50% lag / 25% parent.
- Preserve the last completed record for temporal conditioning.
- Start a 256-byte cooldown to avoid an immediate sequence of duplicate resets.

This is a constrained fast/slow memory architecture. It does not discard all learned statistics and does not retrain the model.

### Fixed integer detector defaults

| Parameter | Value | Purpose |
|---|---:|---|
| Surprisal representation | Q8 bits | Integer-only detector arithmetic |
| Slow baseline EMA | 1/64 | Tracks long-run prequential surprise |
| Fast diagnostic EMA | 1/4 | Supplies the post-alarm baseline |
| Positive drift allowance | 0.25 bits / byte | Suppresses small persistent fluctuations |
| CUSUM threshold | 8 cumulative bits | Alarm boundary |
| Warmup | 64 bytes | Avoids startup alarms |
| Cooldown | 256 bytes | Limits repeated reset churn |

The Q8 log2-ratio helper uses eight rounds of fixed-point squaring, not floating point or `libm`. Exact powers of two are preserved; other ratios are rounded down to Q8 resolution.

The constants are **fixed v1 defaults**, not tuned or established optimal values. The confirmatory experiment must sweep a preregistered threshold/drift grid on development streams and freeze one policy before blind evaluation.

## Controlled detector reference check

`research/cdl-bench/reference/cusum_detector_reference.cpp` independently implements the fixed-point log2-ratio and CUSUM state machine.

It was compiled locally with:

```sh
g++ -O2 -std=c++20 -Wall -Wextra -Wconversion -Werror \
  research/cdl-bench/reference/cusum_detector_reference.cpp \
  -o /tmp/cdl_cusum_reference
/tmp/cdl_cusum_reference
```

Observed output:

| Check | Result |
|---|---:|
| `log2_q8(1)` | 0 |
| `log2_q8(2)` | 256 |
| `log2_q8(4)` | 512 |
| `log2_q8(8)` | 768 |
| `log2_q8(16)` | 1,024 |
| `log2_q8(100)` | 1,700 |
| Stationary 4-bit surprise, 100,000 samples | 0 alarms |
| Step from 4-bit to 7-bit surprise after warmup | detected in 3 observations |
| Gradual +0.5-bit drift over 512 observations | 0 alarms |

The C++ micro-reference compiled with `-Werror` and exited successfully. It validates the integer helper and selected controlled detector scenarios only. It does **not** replay an SCM2 trace, compare predictor likelihoods, validate the Rust source, or establish empirical false-alarm rates on real streams.

## Implementation and tests

- `research/cdl-bench/src/scm_predictor.rs`: `ScmChangePointMixturePredictor`, model ID 8; fixed-point log2 ratio; CUSUM; context reset.
- `research/cdl-bench/src/main.rs`: `change_point_frame`, final expert weights, alarm counts, exact round-trip, and encoder/decoder state equality.
- `research/cdl-bench/src/lib.rs`: public predictor export.
- `research/cdl-bench/reference/cusum_detector_reference.cpp`: independent controlled reference.
- `research/cdl-bench/README.md`: run command and interpretation boundary.

Authored Rust tests cover:
- monotone/exact-power log2 ratio cases;
- stationary no-alarm and sustained-step detection;
- exact observational/interventional stream round trips;
- equality of encoder and decoder expert weights/alarm counts;
- stable unique model ID and larger state footprint.

The Rust tests have not run. No Rust compile, `cargo test`, `rustfmt`, actual arithmetic-coded frame sizes, throughput or peak memory are claimed.

## Scientific and industry positioning

Adaptive segmentation is already a serious lossless time-series compression direction. WZS (September 2026) reports an adaptive-segmentation and algorithm-selection framework evaluated on 43 datasets, with author-reported improvements over Camel/Chimp/ACTF and general-purpose methods. Its figures are paper-reported and have not been independently reproduced here. CDL-Bench must compare against WZS when its implementation/configuration can be reproduced, as well as Pco, Sprintz, Gorilla, Chimp, ACTF and appropriate generic codecs.

Online gated adaptation and fast/slow context methods are also established directions:
- The 2026 online certificate-driven calibration work uses predict-then-update, a gated residual correction and fallback to a fixed backbone when the gate is closed.
- Recent model-level adaptation work describes frozen slow priors, fixed intermediate blending and a fast recent-buffer correction.
- Strongly adaptive online learning and fixed-share literature already studies regret on local intervals and switching experts.

Therefore neither CUSUM, fast/slow memory, gating, nor expert switching is claimed as novel by itself. The narrower testable contribution is **decoder-synchronous change detection and selective context reset inside an exact entropy-coding pipeline, with no transmitted reset side channel and with total model/state/code costs reported**.

Sources:
- WZS: https://www.sciencedirect.com/science/article/pii/S0306457326000889
- Online certificate-driven calibration: https://proceedings.mlr.press/v337/huang26b.html
- Beyond temporal encoding: https://www.sciencedirect.com/science/article/pii/S0893608026012578
- A Closer Look at Adaptive Regret: https://www.jmlr.org/papers/volume17/13-533.html
- Strongly Adaptive Online Learning: https://proceedings.mlr.press/v37/daniely15.pdf

## Confirmatory experiment

The current synthetic intervention case is not sufficient. Test at least these regimes with paired seed and exogenous-noise controls:
1. stationary observational stream;
2. abrupt upstream intervention and restoration;
3. intervention recurrence with a different magnitude;
4. gradual drift in one mechanism;
5. variance/noise scale shift;
6. unrelated common-cause shock;
7. multiple changes at different intervals;
8. high-entropy/noisy controls where surprise rises without a structural intervention.

Compare:
- fixed lag specialist;
- parent oracle;
- fixed-share mixture (model 7);
- CUSUM-reset mixture (model 8);
- no-reset adaptive CUSUM diagnostic;
- best matched Pco/Sprintz/Gorilla/Chimp/ACTF/WZS baseline as appropriate;
- general-purpose Zstd/Brotli/XZ under identical source bytes.

Report at least:
- actual arithmetic-coded bits and ideal prequential bits separately;
- total-size accounting including decoder/model code, metadata, state and resets;
- false alarms per 10,000 samples on stationary/control regimes;
- detection delay in samples and wall-clock time;
- post-change recovery time and cumulative post-shift regret;
- number of resets, context bytes cleared and adaptation observations;
- stationary-regime compression penalty;
- throughput, peak memory, update latency and energy on identical hardware.

## Target gates (research goals, not results)

- Zero false alarms in 100,000 stationary reference samples during the controlled logic check; then target no more than 1 false alarm per 10,000 samples on blind stationary sensor streams.
- Detect a severe held-out shift (at least 3 bits/sample excess surprisal under the pre-shift model) within 8 observations after warmup in at least 95% of paired trials.
- Limit stationary total-bit regression against model 7 to at most 1%.
- Reduce post-shift cumulative ideal code regret by at least 10% against model 7 on declared shift families, with paired 95% confidence intervals.
- For a compression-superiority claim, require at least 5% lower **end-to-end actual coded bits** than the strongest reproduced noncausal and adaptive-segmentation baselines on held-out intervention streams, with exact reconstruction and paired 95% confidence interval excluding zero.
- Publish the full Pareto frontier for total bits, throughput, memory, detector delay and energy; do not hide harmful regimes.

The 5% superiority gate is a future target. The 0.25-bit drift allowance, 8-bit threshold, 64-byte warmup and 256-byte cooldown must not be tuned against the blind test.

## Main risk

A CUSUM can react to an unusually surprising sample, not just a regime change. The detector may reset useful contexts after isolated outliers, or miss shifts that do not increase mixture surprisal. Model 8 may therefore lose to fixed-share model 7 on noisy or gradual regimes. The benchmark must preserve those negative results and compare alternative change features (per-channel residuals, model disagreement, and calibrated predictive risk) under the same predict-then-update constraint.

No causal graph discovery, intervention identification, or SOTA advantage is claimed by this implementation.
