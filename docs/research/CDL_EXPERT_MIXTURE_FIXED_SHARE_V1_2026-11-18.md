# CDL-Bench fixed-share expert-mixture adaptation v1

**Date:** 2026-11-18  
**Status:** Rust implementation and tests authored on the research branch; independent C++ reference simulation executed; Rust compilation and actual arithmetic-coded frame measurements remain pending.  
**Scope:** public E3 SCM2 development fixtures only. These are not blind evaluation data.

## Research question

Can a compressor avoid committing to one stationary predictor by mixing several decoder-synchronous specialist distributions, and can it shift probability mass after an unannounced intervention without transmitting an expert-selection label?

This tests an adaptive **coding policy**, not learned causal discovery. The parent expert receives the known SCM topology; the temporal-lag expert is noncausal and uses only the same variable's previous reconstructed value. The mixture should select whichever expert best predicts the next decoded byte under the current regime.

## Implemented model 7

`ScmExpertMixturePredictor` combines three specialist CDFs:

| Expert | Source | Model |
|---|---|---|
| Field | Typed variable/lane base histograms | Stateless with respect to time |
| Lag | Base plus previous-timestep same-variable context | Noncausal temporal |
| Parent | Base plus same-timestep parent i16 context from the known SCM graph | Oracle-topology control |

Before each byte, the predictor forms the weighted mixture of the experts' **original probabilities**. The Rust implementation uses a common u128 rational denominator across the three CDF totals, so it does not first round each expert to a coarse common probability table. It then converts the final mixture to a positive 32,768-total integer CDF by deterministic largest-remainder allocation with symbol-ID tie-breaking.

After the byte is decoded, posterior expert weights update from each expert's likelihood. A fixed-share rule redistributes **1/256 of the total expert-weight mass per byte** equally across the three experts. This prevents an expert from becoming permanently extinct and permits recovery after a regime transition. All arithmetic and tie-breaking are deterministic; there is no hidden intervention label, expert-selection symbol, or out-of-band message.

The specialist state is shared efficiently:
- one SCM2 header histogram;
- one set of typed field/lane base histograms;
- one lag-context table (four contexts per field/lane);
- one parent-context table (16 contexts per field/lane);
- current and previous reconstructed records.

This avoids keeping three duplicate copies of the shared header and field statistics. The source-level histogram payload is approximately 108 KB before alignment/padding and the mixture-weight fields; the Rust CLI reports the compiler's actual `size_of` value when run. Memory is therefore a real trade-off even though per-symbol expert selection costs no transmitted bits.

## Independent reference experiment

A separate C++20 reference simulator using `unsigned __int128` arithmetic was compiled and run. It independently mirrors:
- the SCM2 16-byte learner-visible header and five i16 records;
- the SplitMix64 generator and E3 held-out mechanism configurations;
- decoder-synchronous base, lag and parent histogram updates;
- exact rational mixture probabilities, posterior weights, fixed-share update, and integer final-CDF normalization.

It evaluated 40 streams:
- two public E3 held-out development configurations;
- ten public seeds per configuration (`0xCA55A1 + seed_index + 200`);
- 1,024 timesteps per stream (10,256 learner-visible bytes);
- one observational and one paired `do(x1=20000)` trace per configuration/seed, with the same seed and exogenous noise;
- intervention active on `[256, 768)`.

The score is **ideal prequential code length**, not actual arithmetic-coded frame bytes. Standalone expert scores use their own raw integer CDFs. The mixture score uses its 32,768-total mixture CDF.

### Mean ideal code length

Lower is better; values are bits per source byte.

| Model | Observational | Interventional | All streams |
|---|---:|---:|---:|
| Typed field | 7.393028 | 7.067700 | 7.230364 |
| Temporal lag | 7.386392 | **6.983605** | 7.184999 |
| Parent oracle | 7.382782 | 7.029233 | 7.206007 |
| Fixed-share mixture | **7.380248** | 6.985519 | **7.182884** |

The mixture's paired relative ideal-cost differences versus lag (20 clusters, each retaining its observational/interventional pair) were:

| Evaluation slice | Mixture vs lag | Paired cluster-bootstrap 95% interval |
|---|---:|---:|
| All streams | -0.02943% | -0.04900% to -0.00969% |
| Observational | -0.08321% | -0.11545% to -0.05089% |
| Interventional | +0.02742% | +0.01970% to +0.03488% |

The mixture had lower ideal code cost than lag on 17/20 observational streams but only 2/20 intervention streams. Overall average improvement is very small and is driven by observational traces. On intervention traces it is slightly worse than simply using lag. This is not an intervention-robust compression win.

### Expert switching diagnostic

Across all streams, mean final expert weights were approximately 6.31% field, 63.35% lag and 30.34% parent. The regime-specific means show adaptation:

| Stream regime | Field weight | Lag weight | Parent weight |
|---|---:|---:|---:|
| Observational | 10.50% | 39.31% | 50.19% |
| Interventional | 2.13% | 87.38% | 10.49% |

In one representative held-out trace, weights after 256 observations (before intervention), 512 observations (mid-intervention), and 896 observations (128 post-intervention observations) were:

| Checkpoint | Field | Lag | Parent |
|---|---:|---:|---:|
| Pre-intervention | 10.88% | 28.16% | 60.96% |
| Mid-intervention | 1.29% | 97.41% | 1.30% |
| Post-intervention window | 1.39% | 87.67% | 10.94% |

No intervention target, boundary, or forced value is visible in the SCM2 observation stream. This is evidence that the mixture can shift from graph-aligned predictions toward a noncausal temporal fallback using decoded observations alone. It is a single synthetic regime example, not a general guarantee.

## Interpretation

1. **Online model selection is useful, but not sufficient.** The mixture tracks which specialist fits the current regime, but its overall improvement is only about three hundredths of one percent in ideal code cost.
2. **Interventions still expose the limit of the parent expert.** The model does the right operational thing—its weight moves toward lag—but it does not outperform the lag expert during intervention.
3. **Fixed-share is a concrete recovery mechanism.** Unlike a posterior with weights allowed to collapse to zero, the fixed-share floor ensures all experts remain available. Its hazard rate is a benchmark parameter and must be preregistered, not tuned on confirmatory streams.
4. **The current parent expert is an oracle.** Since its topology is supplied, even a successful mixture result would not prove causal graph learning.
5. **Cold-start and memory costs matter.** A small per-stream saving may not compensate for predictor code, state memory or throughput costs. The CLI reports inline state size; actual end-to-end measurements still need the Rust runner.

## Next paradigm-shift experiment

Move from expert selection to **intervention-aware switching with a certified fallback**:

- Retain the fixed-share mixture as a universal baseline.
- Add a dedicated change-point expert using only decoder-visible prequential surprisal and residual innovations.
- Separate slow mechanism memory from fast regime memory; do not erase stable mechanisms after a temporary intervention.
- Add a fast recent buffer and a slower global baseline, but charge each buffer's memory and code cost.
- Permit a mechanism update only after evidence exceeds a preregistered threshold; otherwise fall back to lag/field models.
- Record time-to-detection, false alarms per 10,000 samples, recovery steps, post-shift regret, and cumulative bits.
- Test intervention onset, intervention removal, recurring interventions, unseen intervention magnitudes, gradual drift, and unrelated common-cause shocks.
- Require that intervention labels and hidden generator parameters remain evaluator-only.

A future result should be judged against tuned lag/EMA, the current fixed-share mixture, parent oracle, shuffled-parent control, and matched SSM/Transformer forecasts converted through the same calibrated byte CDF.

## Confirmatory gates

Before a causal/adaptive compression claim:
- at least 20 blind seeds per held-out mechanism tuple and three preregistered schedules;
- at least five distinct change types (abrupt intervention, removal, recurrence, gradual drift, common-cause shift);
- exact byte-for-byte reconstruction;
- actual arithmetic-coded bits, not only ideal NLL;
- at least 5% lower end-to-end total bits than the strongest tuned noncausal baseline on held-out intervention streams, with paired 95% CI excluding zero;
- time-to-detect and false-alarm rates reported with confidence intervals;
- predictor code, CDF/calibration data, state memory, decoder and adaptation costs included;
- throughput and peak memory measured on identical hardware;
- null/harmful regimes reported, not averaged away.

The 5% threshold is a future gate, not an achieved result.

## Evidence boundary and files

- Implementation: `research/cdl-bench/src/scm_predictor.rs` (`ScmExpertMixturePredictor`, model ID 7).
- CLI: `research/cdl-bench/src/main.rs` (`online_mixture_frame` and final expert weights).
- API export: `research/cdl-bench/src/lib.rs`.
- Tests authored: exact round trips; deterministic positive normalized weights; weight switching during intervention and partial recovery after its removal; shared-state footprint guard.
- The independent C++ reference simulation was compiled and run. It does not compile or execute the Rust implementation.
- Rust compilation, `cargo test`, `rustfmt`, actual arithmetic-coded sizes, throughput and peak memory remain unverified. The GitHub workflows observed on earlier branch heads were skipped, not passed.
- The E3 configurations and seeds are public development fixtures, not a blind test set.
