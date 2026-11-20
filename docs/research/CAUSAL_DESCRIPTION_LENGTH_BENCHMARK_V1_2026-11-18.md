# Causal Description Length Benchmark (CDL-Bench) v1

**Registered:** 2026-11-18  
**Status:** Research protocol / preregistration; no new benchmark run is claimed  
**Scope:** Standalone benchmark design and Rust reference-codec workstream. This document does not change the live cognitive loop or promote any experimental feature to a production default.

## Executive proposal

Build an evaluation in which a system must do two things at once:

1. Encode an observation stream into a real, decodable, lossless bitstream.
2. Learn reusable causal mechanisms that reduce the number of bits needed for future streams, including streams generated after interventions and mechanism recombinations.

The primary question is:

> After receiving the same observational and interventional evidence, can a causal mechanism learner encode held-out trajectories in fewer **total** bits than strong non-causal predictors and general-purpose codecs, at a matched resource budget—and retain that advantage when the environment is intervened upon?

This is deliberately narrower than “compression is intelligence.” It is a falsifiable engineering question about prediction, mechanism reuse, transfer, and resource efficiency.

The proposed differentiator is **causal program compression**: represent stable, reusable mechanisms once; encode which mechanisms apply and their parameters; then encode the irreducible residuals. A change to one causal variable should not force the system to relearn every unrelated mechanism.

## 1. Why a separate track is necessary

The existing Predictive Compression Program has already produced useful results. CDL-Bench must build on those results without repeating their known measurement failures.

### 1.1 C1: learning and temporal state are different claims

The July 2026 C1 post-annihilation-fix run reports:
- P2 confirmed: live training beat frozen weights on learned-script prediction growth in 10/10 seeds.
- P3 confirmed: the registered repetitive-versus-empty structure comparison passed in 10/10 seeds.
- P1 not supported: live temporal state beat the memoryless control on varied-regime Δcos in only 4/10 seeds.

Interpretation: the tested predictor learned a useful one-step mapping; the experiment did not establish that carried liquid state adds a robust sequence-modeling advantage over per-cycle training.

**Measurement boundary:** C1's bits_saved_persist is a relative predictive-NLL diagnostic in a normalized 256-dimensional representation under a shared residual model. It is not an observed compressed byte stream, not a file-size result, and not yet a codec comparison.

### 1.2 C2: one-step update loss was an optimizer-step proxy

The C2 Chronicle experiment found bits_saved_by_update nearly constant across episodes and schedules (approximately 0.029 bits-like units). The value matched the relative reduction expected from the fixed learning rate, rather than discriminating episode content. The downstream selection overlap varied strongly with schedule and was not evidence of a useful compression gate.

**Design consequence:** do not use a single gradient-step loss delta as a content-quality or memory-priority signal. A candidate signal must predict held-out future coding utility under a true counterfactual replay test.

### 1.3 C3: nearest-neighbour recall is not a portable causal model

The C3 recall arc exposed:
- an initially saturated recall hit-rate, making a binary on/off comparison unable to assess graded matches;
- a similarity dose-response confounded by repetition/predictability;
- paraphrase harm that persisted after frequency equalization;
- schedule-dependent effects: the same sentence sets changed direction when the number and interleaving of tiers changed.

C3f's central methodological result is that recall effects from a fixed stream schedule cannot be promoted into context-free content properties. The mechanism must be tested across multiple schedules and environments.

**Design consequence:** this benchmark uses held-out generators, varied interleavings, paired on/off interventions, and cross-environment transfer. It does not use within-arm correlation as evidence of causation. It also treats nearest-neighbour recall, causal mechanism retrieval, and no retrieval as distinct arms.

### 1.4 Research question not yet answered

The existing programme does not yet show that Symthaea:
- emits an actual lossless bitstream using its own probability forecasts;
- beats Zstandard, Brotli, LZMA2, or strong context-mixing/neural compression under matched evaluation conditions;
- learns causal mechanisms that transfer across interventions;
- reduces the number of new observations needed to adapt to a changed mechanism;
- benefits from a compact causal mechanism library more than from a large episodic store.

CDL-Bench addresses these gaps. It is additive to C1–C4, not a rewrite of them.

## 2. Core hypotheses

Each hypothesis is independently falsifiable.

| ID | Hypothesis | What would support it | What would refute it |
|---|---|---|---|
| H1 | A causal mechanism library reduces held-out total description length | Lower paired total bits than matched non-causal models on held-out causal systems | No gain after model, decoder, state, and side-information costs are counted |
| H2 | Mechanism reuse improves interventional transfer | Fewer adaptation observations/bits after a new intervention at matched predictive quality | No advantage over a strong context model or retrained baseline |
| H3 | Factorized mechanisms generalize to unseen combinations | Better coding of held-out recombinations of known mechanisms | Advantage disappears when combinations, not just seeds, are held out |
| H4 | Active experiment selection reduces future coding cost | More reduction in held-out future bits per acquisition cost than random/uncertainty-only querying | No benefit after including query cost and all observed data |
| H5 | HDC/CfC contributes a distinct efficiency trade-off | A measured Pareto improvement in total bits, memory, latency, or adaptation cost over ablations | Any claimed advantage vanishes under matched parameter/resource budgets |

No hypothesis predicts or implies consciousness, general intelligence, or universal compression superiority.

## 3. Operational definitions and accounting

### 3.1 Actual code length is the primary endpoint

Every submitted predictor must expose a deterministic interface that provides a categorical probability distribution (or integer frequency table) for the next symbol/event before the actual symbol is revealed. That distribution is consumed by one shared, independently tested entropy coder (range coder or rANS).

For an evaluated stream, report:

\[
L_{total} =
L_{decoder} + L_{model} + L_{initial\ state} +
L_{side\ information} + L_{adaptation} + L_{payload}
\]

Where applicable, also report the amortized deployment form:

\[
L_{amortized}(N) =
\frac{L_{shared\ system}}{N} + L_{per\ stream}
\]

The accounting contract:
- Include model/checkpoint bytes, codebook/dictionary bytes, tokenizer or feature-map description, decoder executable or reproducible decoder package, initial recurrent state, per-stream metadata, and adaptation updates whenever the decoder requires them.
- Count all side information needed to reproduce the exact source stream.
- Disclose the training corpus, model provenance, training compute, and whether a baseline receives a pre-trained model. Training corpus bytes are not silently treated as per-stream payload; model artifacts that embody the corpus must be counted in the cold-start system total.
- Publish both **cold-start** (system cost charged to the evaluated task) and **amortized** (shared model cost spread across N declared streams) results. Always publish break-even N.
- Never compare a neural predictor's payload-only bits with a conventional codec's complete file size.
- Never infer realized bits from cosine similarity, MSE, PE, or an uncalibrated continuous-space loss.

### 3.2 Prequential scoring

For online/adaptive arms, the order is strict:

1. Freeze the current predictor state.
2. Predict the next symbol/event and emit its code.
3. Reveal the observation.
4. Update the model using that observation.
5. Continue.

The decoder performs the same state transitions from the decoded observations. Any asynchronous or nondeterministic update is out of scope until deterministic replay is demonstrated.

### 3.3 Losslessness and corruption handling

The codec layer is independent of the cognitive predictor:
- exact byte-for-byte round-trip is mandatory;
- the decoder must reject truncated, malformed, version-mismatched, or checksum-invalid input;
- frequency quantization and probability clipping must be deterministic and shared by encoder/decoder;
- no hidden labels, test data, or non-declared external lookup may be used during decoding;
- model reset and stream-boundary behavior must be explicit.

The codec's correctness test suite must include empty input, one-byte input, all 256 byte values, repeated bytes, long runs, random bytes, adversarial frequencies, maximum-size inputs, truncation at every framing boundary, and bit-flip/checksum mutations.

### 3.4 Predictive bits and causal transfer are separate scorecards

Report:
- actual payload bits and total system bits;
- prequential log-loss / bits per symbol;
- interventional prediction quality;
- causal mechanism recovery where the generator exposes ground truth;
- calibration and abstention;
- adaptation sample/bits needed to restore a target quality;
- CPU time, wall time, peak RSS, model size, decode throughput, and energy only when measured with appropriate instrumentation.

No single scalar may hide a failure in another dimension.

## 4. Benchmark environments

Use a deterministic, versioned generator so every task has ground truth and can be independently reproduced. Do not rely solely on natural-language prompts or fixed repeated sentence schedules.

### E0 — Codec and statistical controls

Purpose: validate the bitstream layer and make sure a predictor beats trivial structure only when structure exists.

- IID byte sources with known distributions.
- First- and second-order Markov byte streams.
- Periodic, bursty, and long-run streams.
- Adversarial/random incompressible streams.
- Distribution shifts with a known change point.

These are codec sanity checks, not evidence of causal reasoning.

### E1 — Latent finite-state systems

Generate streams from hidden finite-state machines with:
- held-out transition combinations;
- controllable noise;
- varying observation aliases;
- hidden states with different dwell-time distributions;
- reset and continuation conditions.

Purpose: distinguish one-step transition fitting from genuine use of history.

### E2 — Structural causal models (SCMs)

Generate small causal worlds (initial target: 4–12 variables) using versioned structural equations. Include:
- linear and nonlinear mechanisms;
- discrete and continuous variables;
- sparse and dense causal graphs;
- confounding and independent exogenous noise;
- stable mechanisms shared across worlds;
- interventions that replace one variable's generating mechanism;
- missing observations and measurement noise.

Expose interventions (for example, do(X=x)) to the learner in the interaction stream. Keep the true graph/equations sealed from the evaluated system except in the oracle diagnostic arm.

### E3 — Mechanism recombination and transfer

Train on environments containing mechanisms A, B, and C separately or in some combinations. Test on new combinations such as A+C and B+C, plus a held-out mechanism parameter range.

This tests whether the system learned reusable mechanisms or merely memorized whole environments.

### E4 — Physical/sensor stream pilot

After E0–E3 are stable, add frozen sensor traces from Symtropy/robotics digital twins: e.g. position, velocity, motor current, force, temperature, vibration, and control events. Preserve native units, timestamps, missingness, and sensor metadata. Evaluate the encoded stream against both a generic codec and appropriate domain baselines.

Real-world traces must have provenance, split boundaries, license, and leakage checks. Do not claim a physical-world result from a synthetic simulation.

## 5. Evaluation protocol

### 5.1 Splits

Split by **generator seed, mechanism family, intervention pattern, and mechanism combination**, not by randomly splitting neighbouring timesteps from the same trajectory.

- Train: public generator families and designated training seeds.
- Validation: disjoint seeds and combinations, available for development.
- Test: hidden seeds, held-out combinations, and held-out intervention values.
- Stress: new graph sizes, noise regimes, variable reorderings, and schedule/interleaving changes.

Pin the generator revision, parameters, and exact task manifest before runs. Test manifests must be committed and hashed before evaluation. Do not tune thresholds or hyperparameters on test results.

### 5.2 Intervention battery

For every base environment, include:
1. observational continuation;
2. intervention on one node;
3. intervention on two coupled nodes;
4. intervention on an upstream cause;
5. intervention on a downstream variable;
6. intervention followed by restoration;
7. two interventions in a new order;
8. held-out parameter value;
9. mechanism replacement;
10. composition of known mechanisms in an unseen arrangement.

The causal learner must identify what changed and preserve predictions for unaffected mechanisms. Measure these separately.

### 5.3 Schedule robustness

Every finding about retrieval, online learning, or causal reuse must be replicated under at least three deterministic schedules:
- balanced round-robin;
- skewed frequency (1:3);
- blocked/interleaved schedule with the same per-tier counts but a different order.

Add random schedule permutations as a separate robustness condition. A result that only appears under one ordering is schedule-specific until replicated.

### 5.4 Repetition and statistics

- Confirmatory phase: at least 20 independent generator seeds per environment family.
- Use paired comparisons on identical streams and interventions.
- Report per-seed results, median, mean, dispersion, 95% bootstrap confidence intervals, and all failed/abstained cases.
- Pre-register primary endpoints, exclusions, and stopping rules before the first confirmatory run.
- A quick smoke run may diagnose plumbing but cannot clear confirmatory gates.
- Do not convert a 7/10 sign tally into a headline capability claim without effect-size and uncertainty reporting.

## 6. Systems and baselines

All models see identical observations, intervention controls, and evaluation windows. Match either parameter count or measured compute/memory budgets; when exact matching is impossible, report the Pareto frontier.

### General-purpose lossless baselines
- raw bytes / no compression;
- Zstandard at documented levels, with and without a trained dictionary where applicable;
- Brotli;
- XZ/LZMA2;
- an established context-model/PPM/CMIX-class compressor for the corpus where practical.

Record exact version, build flags, dictionary training split, thread count, memory limit, CPU/GPU, and commands.

### Predictive baselines
- empirical order-0 and order-1 models;
- a persistence/last-value predictor;
- a matched finite-context or context-tree predictor;
- a byte-level neural next-symbol predictor when a reproducible implementation and inference budget are available.

### Symthaea arms
- HDC encoder + stateless predictor;
- HDC + CfC/LTC one-step predictor;
- HDC + CfC/LTC with recurrent state;
- episodic nearest-neighbour retrieval (diagnostic only; no assumption of benefit);
- causal mechanism library with typed structural equations;
- causal mechanism library + HDC/CfC residual predictor;
- active experiment-selection arm.

Ablate one capability at a time. Do not give the causal arm extra training data, extra feedback, or a larger search budget without counting it.

### Oracle diagnostic

A model with access to the generator's true mechanisms may be used to estimate an attainable lower bound on code length. It is not a competitor, not part of the leaderboard, and must be clearly labelled as an oracle.

## 7. New architecture hypothesis: causal program compression

The proposed mechanism is not “retrieve the nearest old episode and blend its output.” It is a typed library of reusable, executable causal mechanisms.

A mechanism record should carry:
- variable signature and units;
- structural equation or executable predictor;
- parent/dependency set;
- parameter distribution and uncertainty;
- validity domain and known invariances;
- provenance and evidence IDs;
- interventions under which it has been tested;
- counterexamples/falsification set;
- version and deterministic serialization.

At inference time:
1. identify the mechanisms relevant to the current variables;
2. compose the smallest compatible subgraph;
3. produce a predictive distribution for each next variable/event;
4. encode the observation with a shared entropy coder;
5. update only mechanisms whose evidence warrants updating;
6. retain residuals for unexplained behavior instead of forcing it into a known mechanism.

This yields a concrete MDL trade-off:

\[
L(\text{mechanism library}) + L(\text{composition}) +
L(\text{parameters}) + L(\text{residual stream})
\]

The mechanism library is worthwhile only when its reusable structure saves more future bits than it costs to represent and maintain.

### Active information acquisition

For an experiment/query action a, estimate:

\[
VOI(a) =
\frac{
\mathbb{E}[L_{future}\mid D] -
\mathbb{E}[L_{future}\mid D,\operatorname{observe}(a)]
}{
C(a)
}
\]

where C(a) includes acquisition cost, latency, and any resource budget that can be measured. Compare against random, uncertainty-only, and fixed-schedule query policies.

The agent should prefer experiments that separate competing causal explanations, not merely observations that are easy to predict.

## 8. Primary metrics and success criteria

### Gate G0 — codec integrity (hard requirement)

- 100% exact round-trip over all deterministic positive fixtures.
- 100% rejection of the frozen malformed/truncated/corrupted fixture set.
- zero silent data corruption.
- deterministic encoded output for pinned model state, version, and input.
- no unaccounted side channels.

Any failure blocks all compression claims.

### Gate G1 — meaningful predictive coding gain

On held-out streams, the predictor must beat:
- raw bytes;
- persistence;
- empirical order-0 and order-1;
- the strongest matched predictive baseline.

The gain must be present in **real coded bits**, not only continuous-space loss.

### Gate G2 — causal transfer gain

After a held-out intervention or mechanism recombination, the causal arm must recover target predictive quality with fewer additional observations and fewer adaptation bits than matched non-causal baselines. Report both the quality target and the adaptation cost; do not call a faster but inaccurate system a win.

### Gate G3 — targeted codec competitiveness

Research target (not a present result): at least **5% lower total encoded size** than the strongest matched baseline on at least one declared structured sensor/causal stream family, with the paired 95% confidence interval excluding zero, while:
- preserving exact reconstruction;
- staying inside the pre-registered memory/latency budget;
- not losing more than 5% on the other designated stream families;
- counting model, decoder, state, and side information.

The target is domain-specific. Beating Zstd on all files or the Hutter Prize is not a near-term claim.

### Gate G4 — resource efficiency

Publish the full Pareto frontier for:
- total bits;
- encode/decode throughput;
- peak RSS/model size;
- adaptation data and wall time;
- energy only when independently measured.

Set hardware-specific operating points after collecting a baseline on the target machine. Do not infer energy from runtime or report uninstrumented power estimates.

### Gate G5 — reproducibility and external qualification

- Public scripts, immutable input manifests, checksums, versioned model artifacts, and raw per-seed output.
- One independent reproduction on a second machine.
- One held-out external dataset with license/provenance and train/test separation.
- No claim of SOTA superiority until the complete matched protocol is independently reproduced.

## 9. Comparison with the field

These are different tracks and must not be collapsed into a single ranking.

### Hutter Prize / deep universal text compression

The official Hutter Prize page currently lists Vladimir Ivanov's fx2-cmix-T result at **100,424,672 bytes** for enwik9, dated 24 July 2026, with roughly 10 GB RAM and approximately 50 hours reported. The page's next reward threshold is below 99,420,425 bytes. The benchmark includes the decompressor and compressed archive in its size accounting.

This is a serious high-resource universal-text-compression target. CDL-Bench should not pretend that a 256-D relative predictive metric is comparable to it. A later enwik9 entry would be a separate research effort with its own CPU, memory, time, and decompressor accounting.

Reference: https://prize.hutter1.net/

### General neural prediction as compression

Delétang et al., *Language Modeling Is Compression* (ICLR 2024), show that large language models can be turned into general-purpose compressors. Their reported Chinchilla 70B experiments compressed ImageNet patches to 43.4% of raw size and LibriSpeech samples to 16.4%, compared with reported PNG and FLAC ratios of 58.5% and 30.3%, respectively.

This establishes a strong neural-prediction baseline and a reason to report both raw predictive cross-entropy and realized bitstreams. The model-size accounting must be disclosed separately in cold-start and amortized comparisons.

Reference: https://proceedings.iclr.cc/paper_files/paper/2024/hash/3cbf627fa24fb6cb576e04e689b9428b-Abstract-Conference.html

### Practical general-purpose codecs

The Zstandard project reports a benchmark on the Silesia corpus using an Intel Core i7-9700K and lzbench: Zstd 1.5.7 level 1 at ratio 2.896, 510 MB/s compression, and 1,550 MB/s decompression; Brotli 1.1.0 level 1 at ratio 2.883, 290 MB/s compression, and 425 MB/s decompression. These are the project's stated reference conditions, not universal performance guarantees.

Reference: https://github.com/facebook/zstd

### Causal transfer and counterfactual evaluation

CausalWorld provides an intervention-capable robotics benchmark designed around causal structure and transfer between related environments: https://arxiv.org/abs/2010.04296

WhatIfBench (Findings of EMNLP 2026) evaluates open-form, long-horizon counterfactual reasoning with 220 STEM, humanities/social-science, and hybrid scenarios. It reports that the strongest evaluated model reached 64.62% under its proposed evaluation. It is a useful external reasoning comparison, not a compression benchmark: https://aclanthology.org/2026.findings-emnlp.807/

ARC-AGI-2 explicitly adds efficiency measurement to its abstract reasoning evaluation. It is useful as a reminder to evaluate capability and resource cost together, but it does not measure lossless coding or causal mechanism recovery: https://arcprize.org/arc-agi/2

## 10. How this could change the field

1. **From prediction scores to executable evidence.** A system must emit an actual bitstream whose decoder reconstructs the input exactly.
2. **From episodic recall to mechanism memory.** Store equations, causal dependencies, validity domains, uncertainty, and counterexamples—not only nearby vectors and a stored next output.
3. **From passive compression to experiment-directed compression.** The system chooses observations that reduce future description length, with query cost counted.
4. **From one environment to mechanism recombination.** Generalization is measured on new combinations of known causal modules and interventions, not only new random seeds.
5. **From one leaderboard scalar to Pareto-qualified capability.** Bits, transfer, calibration, memory, latency, energy, and integrity remain visible separately.
6. **From architecture branding to pre-registered falsification.** Every new mechanism gets an explicit control, a manipulation check, schedule variation, a kill condition, and a place to publish a null or harmful result.

## 11. Execution plan

### Workstream A — independent Rust codec laboratory

Create a small standalone Rust workspace, outside Symthaea's live cognitive-loop dependencies, with:
- a public `BytePredictor` trait / `ByteCdf` interface that returns an integer cumulative-frequency table for the next byte and updates only from observed/decoded symbols;
- a shared entropy coder that can accept built-in order-0/order-1 models or caller-supplied predictors through the same interface;
- a deterministic range coder or rANS implementation;
- adaptive order-0 and order-1 reference predictors;
- a framed format with version, source length, and checksum;
- encoder/decoder round-trip tests and adversarial corruption tests;
- JSONL/CSV result output with model and payload byte counts.

The first milestone is correctness, not a compression win. Keep the probability interface stable so Symthaea predictors can plug in later.

### Workstream B — deterministic causal stream generator

Implement E0–E3 generators with pinned seeds and hidden-test manifests. Store generator metadata and raw streams separately from model output. Build the task manifest before any confirmatory run.

### Workstream C — model adapters and baselines

Add Symthaea predictor adapters only after the generic codec is independently tested. Begin with stateless and one-step predictors; add recurrent state and causal mechanism composition as separate arms.

### Workstream D — resource-qualified experiments

Run quick smoke tests for plumbing, then preregister the N≥20 confirmatory suite. Record raw data and independent checksums. Replicate at least one run on another machine.

### Workstream E — physical transfer

Only after synthetic causal transfer works, connect real Symtropy/robotics sensor traces and evaluate against Zstd plus domain-specific baselines.

## 12. Stop conditions and integrity rules

- If the exact coder/decoder round-trip fails, stop.
- If all apparent gains disappear after including model/decoder/side information, report no compression advantage.
- If gains exist only on seen schedules or repeated strings, classify them as schedule-specific.
- If a causal arm cannot beat matched non-causal baselines on held-out interventions, do not promote causal compression as an achieved capability.
- If a mechanism library increases total bits or adaptation cost without improving held-out quality, retain the simpler baseline.
- Do not feed this metric into live memory admission, attention, or safety gates until an independent causal utility result exists.
- Preserve all negative results and pre-registration amendments in place; do not overwrite earlier measurements.

## 13. Initial deliverables

- [ ] Protocol reviewed and frozen.
- [ ] Standalone Rust codec crate with exact round-trip and corruption tests.
- [ ] E0 statistical source generator.
- [ ] E1 finite-state source generator.
- [ ] E2 SCM/intervention generator.
- [ ] Paired baseline runner and machine-readable output.
- [ ] Symthaea stateless predictor adapter.
- [ ] Symthaea CfC/LTC adapter.
- [ ] Causal mechanism library arm.
- [ ] Independent replication.

**Current evidence status:** this document is a research protocol. No new codec, benchmark run, compression ratio, causal-transfer result, or SOTA improvement is claimed by this change.
