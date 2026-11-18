# CDL-Bench SCM context ablations v1.2

**Date:** 2026-11-18  
**Status:** implementation added to the research branch; Rust compiler execution and arithmetic-coded size measurements pending  
**Scope:** learner-visible SCM2 five-channel, fixed-point i16 traces only  
**Revision note:** this version supersedes the prior SCM1/SCM2 draft calculations. SCM2 removes intervention metadata from learner-visible bytes; the implementation and exploratory Python model use additive field-plus-context frequency interpolation. Numbers below use the 16-byte label-free SCM2 header.

## Research question

Does preserving typed variable structure improve byte coding, and does conditioning on known causal parents add value beyond an equally available temporal baseline—especially under intervention?

The benchmark must distinguish:
1. knowing which field a byte belongs to;
2. exploiting that field's own temporal persistence;
3. exploiting cross-variable parent relationships;
4. learning reusable mechanisms and transferring them to held-out interventions.

The first three are implemented as separate predictor arms. The fourth remains future work.

## Implemented predictor arms

All custom predictors implement the shared `BytePredictor` interface and use the same arithmetic coder, frame format, checksums, and decoder-synchronous symbol updates. The learner-visible SCM2 header is 16 bytes: magic, version, zero flags, fixed variable count, and sequence length. It contains no intervention target, boundary, forced value or mechanism parameters. Records follow as five signed i16 values in topological order, each little-endian. The intervention manifest and mechanism configuration remain in crate-private `ScmTrace` fields and are never serialized. The fixed E3 mechanism combinations are public development fixtures visible in source, not a blind test split; confirmatory claims require an externally controlled parameter/seed manifest.

| Frame arm | Model ID | Information used |
|---|---:|---|
| `field_frame` | 2 | Variable identity and byte lane; adaptive base histogram per field/lane |
| `parent_oracle_frame` | 3 | Field/lane plus same-timestep parent i16 values, quantized into four bins |
| `lag_frame` | 4 | Field/lane plus that variable's previous-timestep i16 value, quantized into four bins |
| `hybrid_frame` | 5 | Parent-sign context plus previous-timestep four-bin same-variable context |
| `shuffled_parent_frame` | 6 | Same context arity as parent-oracle, but deliberately misassigned downstream parent identities |

For parent, lag, hybrid and shuffled-parent arms, each CDF adds the field/lane base frequencies to the selected conditional-context frequencies. Both histograms update after each decoded symbol. This additive count model supplies field-level backoff evidence when a conditional context is sparse. The typed-field arm uses the base histogram alone. Counts are maintained with deterministic integer rescaling, so the encoder and decoder can reproduce the same CDF without transmitted per-symbol probabilities.

The parent and hybrid models are **oracle-topology controls**: their code is supplied the SCM graph. They do not discover the graph, infer causal structure from data, estimate structural equations, or demonstrate transfer. The lag model is a noncausal temporal comparator. The shuffled-parent arm uses the same number of parent slots but intentionally changes parent identities; it tests whether a score depends on correct topology rather than only context capacity.

Each predictor starts from deterministic state for each stream. Predictor state is decoder-synchronous; no separate trained checkpoint is currently needed. The frame does not embed predictor implementation code. State bytes, executable/code distribution, and any learned parameters must be included in cold-start and amortized accounting.

## Preliminary independent diagnostic

An independent Python translation of the deterministic SCM generator and the **revised additive base-plus-context frequency updates** was evaluated on the two E3 held-out mechanism combinations, 10 explicit seeds per combination, and paired observational / `do(x1=20000)` traces using the same seed and exogenous noise: 40 streams total, 4,096 steps each, and 40,976 bytes per SCM2 stream. The intervention target/window/value are not serialized into the stream; paired treatment labels are held only in the evaluator's scoring manifest. The seed set was `0xCA55_A1 + seed_index + 200` for `seed_index = 0..9`. These are explicit public development seeds, not concealed challenge seeds.

The diagnostic is cumulative prequential ideal code length:
`sum_t log2(total_frequency_t / observed_symbol_frequency_t)`,
including the 16-byte SCM2 schema/length header. It measures probabilities emitted by the Python translation. It is **not** the output size of the Rust arithmetic coder.

| Model | Mean ideal bits / input byte |
|---|---:|
| Order-0 | 7.53074 |
| Order-1 | 7.70285 |
| Typed field/lane | 7.02518 |
| Temporal lag | 6.95134 |
| Parent oracle | 6.97503 |
| Parent + lag hybrid | 6.96331 |
| Shuffled-parent negative control | 6.98965 |

The temporal-lag model is the strongest of these current context arms overall. Relative to lag, mean paired ideal-code-cost ratios across the 20 (mechanism, seed) groups were:

| Context arm vs lag | Mean relative cost | Paired cluster-bootstrap 95% interval |
|---|---:|---:|
| Parent oracle | +0.3408% | +0.3233% to +0.3586% |
| Parent + lag hybrid | +0.1722% | +0.1570% to +0.1875% |
| Shuffled-parent | +0.5511% | +0.5278% to +0.5743% |

The intervention split is the critical result. SCM2 does not expose the intervention descriptor or boundary through the learner-visible header:

| Context arm vs lag | Observational relative cost | Interventional relative cost |
|---|---:|---:|
| Parent oracle | -0.0672% | +0.7732% |
| Parent + lag hybrid | -0.0797% | +0.4391% |
| Shuffled-parent | +0.1443% | +0.9822% |

Across the paired group bootstrap, the observational intervals were [-0.1284%, -0.0051%] for parent-oracle, [-0.1126%, -0.0463%] for hybrid, and [+0.0748%, +0.2141%] for shuffled-parent. Interventional intervals were [+0.7400%, +0.8065%], [+0.4329%, +0.4453%], and [+0.9519%, +1.0122%], respectively.

Interpretation:
- Correct parent topology provides a small observational advantage over the matched lag arm in this synthetic SCM.
- That advantage reverses under the held-out intervention; parent and hybrid become worse than lag.
- The shuffled-parent negative control is worse than the correct-parent model in both regimes, suggesting the observational difference is not simply extra context capacity.
- The simple noncausal lag model is still more robust across the paired observational/interventional mix.

This is a useful falsification result, not a causal-compression win. It suggests the right next question is whether a **learned, factorized mechanism model can detect and adapt to intervention-induced changes** without receiving hidden mechanism labels or the evaluator's graph.

### Evidence boundary

The independent Python calculation mirrors the deterministic generator, SCM2 record layout and histogram updates but is not compiler-backed execution of the Rust implementation. Rust compilation, unit tests, rustfmt, actual arithmetic-coded frame lengths, throughput, memory, energy and external-codec comparisons remain unverified. No SOTA result or deployed-size reduction is claimed. The earlier unsmoothed-context percentages are superseded and must not be combined with this table.

## Hypotheses and falsification

- **H1 — typed fields help:** field/lane coding should beat flattened order-0/order-1 controls on held-out SCM streams and then be compared with tuned multichannel codecs.
- **H2 — temporal persistence is strong:** lag context is an essential baseline; it should be tuned as hard as any causal model.
- **H3 — causal context adds intervention value:** a learned mechanism model must beat the best noncausal temporal predictor on held-out interventions, not only in observational data.
- **H4 — compositional transfer matters:** a mechanism library must reuse separately learned mechanisms across unseen combinations. A lookup table or supplied graph does not count.
- **H5 — topology matters:** the correct-parent arm must outperform the shuffled-parent negative control under paired seeds; otherwise the graph alignment claim is unsupported.

Null results are valid. Do not alter seeds, context quantization, schedule, or thresholds after inspecting confirmatory outcomes. Any change to model or threshold must be registered as a new version and tested on untouched seeds.

## Next executable milestone

Build a learned, factorized mechanism predictor with:
1. explicit train/validation/test split by full mechanism tuple;
2. parameter estimation from training observations only;
3. no evaluator-side graph, parameter, latent state or intervention-label leakage into learner-visible input;
4. deterministic fixed-point inference and decoder-synchronous updates;
5. paired observational / intervention traces with identical seed and exogenous noise;
6. direct comparisons against order-0, order-1, typed-field, tuned EMA/lag, parent-oracle, hybrid and shuffled-parent arms;
7. exact byte-for-byte reconstruction and checksummed frames;
8. cold-start/amortized accounting for predictor code, parameters, adaptation state and headers.

A useful next paradigm shift is **mechanism-conditioned coding with online change detection**: separate stable mechanism parameters from exogenous innovations, but allow the encoder/decoder to synchronously detect when an intervention changes the active mechanism. Charge every bit for change points, mechanism switches and adaptation. The model must fail closed to the lag/statistical fallback when posterior evidence is insufficient. This is a research hypothesis, not an implemented capability.

## Confirmatory success gates

Before describing a causal coding advantage:
- use at least 20 held-out seeds, three preregistered temporal schedules, and multiple unseen mechanism tuples;
- require 100% exact reconstruction for the lossless track;
- require at least 5% lower **end-to-end total bits** than the strongest tuned noncausal temporal baseline on held-out intervention traces;
- require a paired 95% confidence interval excluding zero;
- report observational and interventional performance separately;
- publish throughput, peak resident memory, predictor state, code size, training/adaptation cost and size/latency Pareto frontier;
- include negative-control mechanisms and the shuffled-parent arm;
- report cold-start and amortized model-code cost at realistic stream lengths.

The 5% threshold is a future success gate, not a result. If a learned mechanism model does not clear it, prefer the simpler noncausal model for this workload.

## Files

- `research/cdl-bench/src/scm_predictor.rs`: field, parent-oracle, lag, hybrid and shuffled-parent predictors over the label-free SCM2 record layout.
- `research/cdl-bench/src/main.rs`: causal CLI round-trip and frame-size comparisons.
- `research/cdl-bench/src/lib.rs`: module and public predictor exports.
- `research/cdl-bench/README.md`: usage, accounting and interpretation boundaries.
