# CDL-Bench SCM context ablations v1

**Date:** 2026-11-18  
**Status:** implementation added to the research branch; Rust compiler execution and codec-size measurements pending  
**Scope:** SCM1 five-channel, fixed-point i16 traces only

## Research question

Does preserving the typed variable structure of a multichannel stream improve coding, and does conditioning on known causal parents add value beyond an equally available temporal baseline?

This is a deliberately staged question. The benchmark must not conflate:
1. knowing which field a byte belongs to;
2. exploiting that field's own temporal persistence;
3. exploiting cross-variable causal parents;
4. learning reusable mechanisms and transferring them to held-out interventions.

The first three are now separate executable predictor arms. The fourth remains future work.

## Implemented predictor arms

All custom predictors implement the shared `BytePredictor` interface and use the same arithmetic coder, frame format, checksums, and decoder-synchronous symbol updates. The SCM1 header is 33 bytes; the data records follow as five signed i16 values in topological order, each little-endian.

| Frame arm | Model ID | Information used |
|---|---:|---|
| `field_frame` | 2 | Variable identity and byte lane; one adaptive histogram per field/lane |
| `parent_oracle_frame` | 3 | Field/lane plus same-timestep parent variables, quantized from their reconstructed signed i16 values into four bins |
| `lag_frame` | 4 | Field/lane plus that same variable's previous-timestep reconstructed i16 value, quantized into four bins |
| `hybrid_frame` | 5 | Same-timestep parent topology, with one sign bit per parent, plus the previous-timestep four-bin same-variable context |

The parent and hybrid models are **oracle-topology controls**. Their code is supplied the SCM graph. They do not discover the graph, infer causal structure from data, estimate the structural equations, or demonstrate transfer. The lag model is a noncausal temporal comparator. This architecture makes it possible to ask whether known cross-variable structure actually helps beyond persistence.

Each model starts from a deterministic empty state for each stream. Its per-stream state is decoder-synchronous and requires no separately trained checkpoint. The frame does not embed the predictor implementation. State bytes, executable bytes, and any distribution of model code must remain visible in total-system accounting.

## Preliminary independent diagnostic

An independent Python translation of the deterministic SCM generator and the predictor count updates was evaluated on the two E3 held-out mechanism combinations, 10 seeds per combination, with an observational and paired `do(x1=20000)` trace for each seed (40 streams, 4,096 steps each).

The diagnostic is cumulative prequential ideal code length:
`sum_t log2(total_frequency_t / observed_symbol_frequency_t)`,
including the 33-byte SCM header. It measures the probabilities emitted by the translated predictor. It is **not** the output size of the Rust arithmetic coder.

| Model | Mean ideal bits / input byte |
|---|---:|
| Order-0 | 7.52945 |
| Order-1 | 7.70197 |
| Typed field/lane | 7.02431 |
| Temporal lag | 6.87288 |
| Parent oracle | 6.93175 |
| Parent + lag hybrid | 6.92072 |

Relative to the temporal lag model, parent-oracle had about **0.90% higher** ideal code cost overall (it won on 10/40 streams); hybrid had about **0.70% higher** cost (it lost on all 40 streams).

The parent-vs-lag result is regime-dependent:
- Observational traces: parent-oracle was about 0.095% lower in mean ideal cost.
- Interventional traces: parent-oracle was about 1.894% higher in mean ideal cost and lost on all 20 paired intervention streams.

The paired cluster bootstrap sampled the 20 (held-out mechanism, seed) groups while retaining each group's observational/interventional pair. Its 95% intervals for relative ideal-code-cost increase versus lag were approximately +0.875% to +0.926% for parent-oracle and +0.675% to +0.732% for hybrid.

These figures are exploratory logic diagnostics only. The source has not been compiled or run with Rust in the present environment; no arithmetic-coded frame lengths, throughput, memory, energy, or external-codec comparison are claimed. The results are not evidence of SOTA or of causal compression. They indicate that the graph-conditioned histograms, as currently specified, do not yet beat a simple temporal baseline—especially after intervention.

## Hypotheses and falsification

- **H1 — typed fields help:** field/lane conditioning should reduce actual coded size against order-0/order-1 on held-out SCM streams. It must also be compared with a tuned multichannel baseline, not just flattened bytes.
- **H2 — temporal persistence is strong:** the lag arm should be competitive on both observational and interventional traces. Its performance is a control, not a claim of causal reasoning.
- **H3 — causal context adds intervention value:** a learned mechanism model must beat the best noncausal temporal predictor on held-out interventions, not merely beat a weak byte model or the parent-oracle diagnostic.
- **H4 — compositional transfer matters:** an actual mechanism library should reuse independently learned mechanisms across unseen combinations. A tuple lookup or a supplied graph does not count.

Null results are valid. Do not change context quantization, seeds, schedule, or the success threshold after seeing the confirmatory result. Any improved model or threshold must be registered as a new version and evaluated on untouched seeds.

## Next executable milestone

Build a learned, factorized mechanism predictor with:
1. explicit train/validation/test split by full mechanism tuple;
2. parameter estimation from training observations only;
3. no evaluator-side graph, parameter, latent-state, or intervention-label leakage into the stream;
4. deterministic fixed-point inference and decoder-synchronous updates;
5. paired observational / intervention traces with identical seed and exogenous noise;
6. direct comparison against order-0, order-1, typed-field, tuned lag/EMA, parent-oracle and hybrid arms;
7. exact byte-for-byte reconstruction and checksummed frames;
8. cold-start and amortized costs for predictor code, parameters, adaptation state, and all headers.

## Confirmatory success gates

Before describing a causal coding advantage:
- use at least 20 held-out seeds, three predeclared temporal schedules, and multiple unseen mechanism tuples;
- require 100% exact reconstruction for the lossless track;
- require at least 5% lower **end-to-end total bits** than the strongest tuned noncausal temporal baseline on the held-out intervention set;
- require a paired 95% confidence interval excluding zero;
- report observational and intervention results separately;
- publish throughput, peak resident memory, predictor state, code size, training/adaptation cost, and a size/latency Pareto frontier;
- include negative-control mechanisms and a shuffled-parent control to detect graph leakage or context overfitting.

The target is a research gate, not an achieved result. If the mechanism model does not clear it, prefer the simpler noncausal model for this workload.

## Files

- `research/cdl-bench/src/scm_predictor.rs`: field, parent-oracle, lag, and hybrid predictors.
- `research/cdl-bench/src/main.rs`: causal CLI round-trip and frame-size comparisons.
- `research/cdl-bench/src/lib.rs`: module and public predictor exports.
- `research/cdl-bench/README.md`: usage, accounting, and interpretation boundaries.
