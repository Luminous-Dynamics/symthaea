# Cognitive-loop prediction-evidence benchmark (PEV-1)

**Status:** proposed benchmark contract; typed-boundary tests are authored but have not been run end-to-end. The full 15-case consumer-level protocol remains outstanding.

## 1. Research claim

A prediction-error (PE) number is not evidence by itself. It must carry provenance stating whether the current predicted and observed states form a comparable, finite measurement. A compatibility sentinel such as `1.0` may remain useful for diagnostics, but it must not silently become surprise, reward, confidence, urgency, replay importance, a learning target, or a physical-cost estimate.

The core invariant is:

> **Invalid PE has zero influence on PE-dependent state transitions, while independent evidence channels continue to function.**

This is narrower and more testable than a claim about consciousness, general intelligence, or state-of-the-art performance.

## 2. Paradigm shift: evidence objects, not sentinel-valued floats

The long-term architecture should make invalid numeric states unrepresentable at PE-consuming APIs. This PR uses a compatibility float plus a validity flag to contain risk without immediately breaking every call site. That is an incremental migration boundary, not the final abstraction.

Target API:

```rust
enum PredictionEvidenceStatus {
    Valid,
    NoPriorPrediction,
    ZeroNormPrediction,
    EmptyObservation,
    DimensionMismatch,
    NonFiniteInput,
    StaleOrOutOfOrder,
    SourceUnavailable,
    NotEvaluated,
}

struct PredictionErrorEvidence {
    status: PredictionEvidenceStatus,
    value: Option<f32>,
    prediction_cycle: Option<u64>,
    observation_cycle: u64,
}
```

The typed migration layer exists in `src/cognitive_loop/types/prediction_evidence.rs` as `PredictionErrorEvidence` and `PredictionEvidenceStatus`. The value and status are private; valid construction rejects non-finite values, invalid statuses carry no numeric value, and deserialization rejects inconsistent records. `CycleResult::prediction_evidence_at` and `PsiAttestationRecord::prediction_evidence` adapt legacy float-plus-flag fields at API boundaries. Both the perception-phase urgency result and shared `CycleState` now carry typed PE evidence only; their duplicate numeric/boolean PE fields have been removed. Urgency, surprise exploration, episodic replay, dream consolidation, subsystem feedback, consciousness feedback, core training, safety enforcement, civic-crisis detection, scientific-method updates, and output/voice gates derive usability from evidence bound to the current observation cycle. The raw encoder number remains only as a compatibility/diagnostic payload. The core HDC encoder now records prediction source cycles, the main text path feeds the CfC prediction back at output time, and `encode_at` rejects predictions not produced by the immediately preceding cycle before PE can train attention/history. The embedding/HV path similarly tracks the producer cycle and advances its state window after rejection. Main text, embedding, pre-encoded-HV, and safety-precheck result paths expose `CycleMetadata.prediction_evidence` with status and cycle provenance, without raw input vectors. `NotEvaluated` distinguishes a cycle intentionally gated before comparison from a source that failed to provide evidence. Some interfaces outside this shared phase boundary still use compatibility fields and remain migration targets. Remaining enforcement goals are:

- Only `Valid` may expose a PE value to learning, confidence, surprise, safety assessment, replay, or thermodynamic PE consumers.
- Missing PE never means zero error, maximum error, or “repeat the last value as a fresh sample.”
- Every consumer records the evidence status and a decision digest; telemetry may expose a compatibility number only alongside explicit validity/status.
- Non-PE evidence stays live. A missing PE must not suppress an independent low-coherence safety event, external reward, valid sensory novelty, or measured energy-budget signal.
- Compatibility conversions are one-way at boundaries: legacy float APIs can emit a diagnostic sentinel, but downstream code must not reconstruct validity from that number.

A later hardening step should add source and observation identifiers plus a digest of the normalized prediction/observation metadata, so duplicate or delayed outcomes are detected rather than counted twice. Hash metadata with domain separation; do not hash or log raw user text by default.

This design separates three questions currently too easy to conflate: **Was a value computed? Was it comparable? Is it permitted to influence this consumer?** A finite float alone answers none of them.

## 3. Counterfactual test matrix

Each scenario should be replayed against the same initial state and deterministic seed. Compare a run with invalid PE set to different numeric payloads (`0.0`, `1.0`, `NaN`, and infinity where the type permits it) against a matched run where PE is not read by any PE-dependent consumer.

| Case | Input condition | Required behavior |
|---|---|---|
| PEV-01 | Cold start, no prior prediction | Report compatibility diagnostic if required; mark evidence absent; admit no PE-driven update or replay sample |
| PEV-02 | No prediction object | Same as PEV-01; do not train encoder attention or PE EMA from a sentinel |
| PEV-03 | Zero-norm prediction | Mark degeneracy explicitly; do not treat the fallback number as an error observation |
| PEV-04 | Embeddings empty, unequal dimensionality, or non-finite | Produce no comparable MSE; do not substitute zero error; advance the state/prediction window |
| PEV-05 | Repeated invalid observations | Do not increment or reset PE-derived streaks, grow PE histories, repeatedly replay a held last-known value, or charge it again as fresh memory pressure |
| PEV-06 | Invalid then valid high PE | Recover immediately; admit the new valid value once and preserve the previous invalid gap as missingness |
| PEV-07 | Invalid then valid low PE | Low-error streak advances only on valid comparisons; invalid cycles cannot manufacture a “stable/predictable” streak |
| PEV-08 | Same numeric value, validity toggled | Invalid `1.0` and valid `1.0` must take different evidence paths; numeric equality must not bypass the validity contract |
| PEV-09 | Invalid PE, high independent coherence/sensory variance/energy signal | Independent subsystems still respond according to their own evidence; invalid PE must not disable the entire cognitive cycle |
| PEV-10 | Invalid PE after a queued plan/action | Do not evaluate its PE-based outcome yet; do not lose a pending outcome solely because the current observation is missing |
| PEV-11 | Snapshot, serialization, or old record with missing validity field | Default conservatively to invalid; preserve ABI layout where promised and verify old-data migration |
| PEV-12 | Long stream with task shifts and distractors | No false PE-driven reinforcement, forgetting, confidence inflation, or reactivation caused by an invalid sentinel |
| PEV-13 | Cycle preempted before a prediction comparison (e.g., safety gate) | Report `NotEvaluated`; do not fabricate zero/high PE; preserve the independent preemption/safety decision |
| PEV-14 | Prediction reused after a skipped/out-of-order cycle | Classify as `StaleOrOutOfOrder`; do not update encoder attention/PE history; preserve source and observation cycle IDs; recover on the next valid fresh prediction |
| PEV-15 | Explicit reset or expectation reset | Clear both encoder-HDC and embedding/HV prediction windows and provenance; first post-reset observation is `NoPriorPrediction`; no pre-reset sample may update PE-dependent state |

Where a subsystem is independent of PE, the expected result is not “no change to the entire system.” The invariant applies only to the PE-derived component; coherence, energy, explicit reward, sensory variance, and other valid signals remain eligible to change state.

The typed-value contract has additional unit fixtures: a value is available only for its recorded observation cycle; non-finite values cannot be constructed as valid evidence; invalid statuses cannot expose a value; and deserialization rejects a missing valid value or a numeric value attached to an invalid status. These fixtures are not evidence of a passing build until executed at the exact commit under test.

## 4. Metrics

Report raw counts as well as rates. Do not collapse all dimensions into one score.

### Three-tier scoreboard (proposed gates, not measured results)

1. **Evidence integrity:** zero invalid-evidence leakage, false replay admission, false PE-only surprise, stale-sample reuse, or duplicate PE admission. Run deterministic counterfactuals plus at least 10,000 generated/interleaved invalid cases. Any nonzero result blocks a correctness claim.
2. **Predictive utility:** compare normalized prediction loss and calibration against persistence and zero-prediction baselines. Stretch target: beat persistence on aggregate held-out streams with a paired 95% confidence interval excluding zero, improve at least four of six heterogeneous task families, and report every task rather than hiding regressions in an average. Track valid-sample coverage and missingness separately so a system cannot “improve” by refusing hard cases.
3. **Continual-learning utility:** on a public sequential-task suite such as CL-Bench, report absolute reward, gain over the same system's stateless baseline, per-task gains, forgetting/transfer, latency and cost. Stretch target: positive aggregate gain with a paired confidence interval above zero against the matched stateless baseline, then a better reward/cost frontier than the strongest comparable baseline. Use at least five repeated runs per task (ten for the internal PEV suite); publish variance and traces. Only claim a leaderboard win after running the full matching benchmark protocol.

Keep evidence integrity as a hard gate rather than a weighted metric: strong task accuracy cannot compensate for invalid evidence changing safety or learning state. Resource overhead is an explicit engineering objective, but no p95 overhead limit should be claimed until measured on a pinned machine and workload.

- **Invalid-evidence leakage rate:** PE-dependent state transitions caused by invalid observations / invalid observations. Gate: **0 in deterministic tests** and **0 over at least 10,000 generated invalid/interleaved cases**.
- **False replay admission:** invalid transitions admitted to episodic replay. Gate: **0**.
- **False-surprise rate:** invalid PE causing a PE-only surprise, vigilance, exploration, or PROBE decision. Gate: **0**.
- **Valid-evidence preservation:** proportion of finite, comparable observations that reach each intended PE consumer exactly once. Gate: **100% on deterministic fixtures**.
- **Recovery latency:** number of valid observations required before the system resumes PE-driven updates after missingness. Target: **one valid observation**, without blending it with an invalid sentinel.
- **Independent-channel preservation:** matched cases where coherence, sensory variance, explicit reward, or physical-energy evidence should act. Gate: intended non-PE transition remains present.
- **Stale-evidence reuse:** count of previously valid PE samples applied as a fresh observation in later invalid cycles. Gate: **0**.
- **Resource cost:** cycle latency (p50/p95/p99), CPU time, peak memory, replay-buffer size, and incremental serialized bytes. Report measured baselines and deltas; do not set a performance claim before measurement.

For stream-learning experiments also report per-step task performance, forward/backward transfer, forgetting, calibration (Brier score/ECE on resolved outcomes), number of admitted learning samples, time-to-target quality, and compute cost. Report worst-task and dispersion alongside aggregate scores.

### ABI snapshot boundary

`CycleSnapshot` remains `repr(C)` with its legacy numeric PE and one validity byte to preserve the existing WASM/FFI layout. The new `prediction_error_sample()` accessor is the supported consumer boundary: it returns `Some(value)` only when the validity byte is exactly 1 and the value is finite; all other combinations return `None`. Memory pressure, drive/history, learning, perception vigilance, vision surprise, neuroevolution, thermodynamic integration, Muse sonification, therapeutic shadow telemetry, and the mock subsystem now use this accessor rather than interpreting the raw fields independently. The raw fields remain public for ABI compatibility, so continued auditing is still required for any new direct readers.

The accessor's authored fixtures cover invalid flags with `0`, `1`, NaN, and infinities; non-canonical validity bytes (`2`, `255`); a valid finite sample; and NaN even when the byte says valid. This is test coverage, not an executed pass.

### Exactly-once evidence admission

A validity flag prevents malformed PE from being used; it does not prevent one valid event from being applied twice. The typed module now includes `PredictionEvidenceCursor`, a per-consumer monotonic cursor that admits a finite PE sample only once per observation cycle and rejects duplicates or out-of-order cycles. Missing evidence does not advance the cursor, so a same-cycle valid retry remains admissible.

The first consumer integrations are the memory-manager consolidation-pressure input and neuroevolution's accumulated PE fitness stream. Their counterfactual fixtures compare a repeated valid cycle against a matched valid-then-invalid cycle, and verify that a later cycle is admitted normally. This is an in-process, per-consumer guarantee; cursor persistence across checkpoint/restore and adoption by every other stateful PE consumer remain follow-up work.

### Canvas / creative consumer boundary

The separate `symthaea_canvas::CognitiveSnapshot` now carries a serde-defaulted `prediction_error_valid` field and a finite-only accessor. Old serialized snapshots that lack that field default to invalid. The metadata-only adapter fails closed because an FEP-surprise scalar alone does not certify that a predicted state and observed state were comparable. The cycle-snapshot adapter copies PE only through the ABI accessor.

Atelier's hybrid, neural-canvas, and pixel-canvas HDC encoders now add a PE channel only when the accessor returns `Some`; missingness removes that channel rather than inserting a fabricated zero/high sentinel. The aesthetic turbulence channel and musical-state bridge likewise avoid consuming an invalid PE. Timeline interpolation preserves PE validity only when both endpoints have valid evidence; one missing endpoint keeps the interpolated PE missing. New fixtures assert that multiple invalid numeric payloads yield identical HDC input encodings, legacy JSON remains invalid, and invalid endpoints do not create an interpolated measurement. These assertions are authored, not run.

### Typed-contract regression checks

The typed layer includes unit tests for invalid numeric payloads (`0`, `1`, NaN, and positive/negative infinity), non-finite values incorrectly flagged valid, cycle mismatch, stale-status classification, constructor invariants, valid serialization round-trip, malformed serialized records, and a deterministic 10,000-payload boundary sweep. New Canvas and Atelier tests cover legacy snapshot deserialization, sanitization fail-closed behavior, cycle-snapshot provenance, invalid-input HDC invariance, and validity-aware interpolation. The shared admission cursor has tests for duplicate/out-of-order rejection, missing-evidence non-advancement, and explicit reset; memory pressure and neuroevolution have consumer-level duplicate-cycle regressions. Consumer-focused tests cover stale and missing evidence in urgency and surprise paths, plus the encoder's consecutive-cycle acceptance, skipped-cycle rejection, non-finite prediction rejection before PE/attention statistics, explicit service reset of prediction provenance, and cold-start metadata on the pre-encoded-HV entry path. The main text-loop test verifies source-cycle handoff, and output tests check that public telemetry agrees with the compatibility validity bit and records the observation cycle. Keep these separate from the 15 core PEV scenarios so passing type-contract tests cannot be mistaken for coverage of all downstream consumers. These are authored checks, not a pass claim until run at the exact head.

## 5. Experimental protocol

1. Freeze the code SHA, toolchain, feature flags, configuration, seed set, input stream, and expected result digest before measuring.
2. Run deterministic unit/counterfactual cases before long sequences. A skipped or queued workflow is not a pass.
3. Use at least 10 fixed seeds for the first aggregate experiment. Include cold starts, repeated missingness, valid/invalid alternation, low-to-high and high-to-low shifts, and distractor tasks.
4. Compare these ablations on identical input streams: (a) current legacy numeric-sentinel behavior, (b) explicit validity gates, (c) skip all learning on invalid cycles, and (d) use only independently measured signals. Keep non-PE subsystems and compute budgets identical.
5. Publish JSON/CSV raw records with seed, stream index, validity reason, encoder degeneracy, admitted-update counters, per-consumer before/after state digests, runtime, and memory. Include negative controls and the verifier's exit status.
6. Use paired per-seed differences and confidence intervals for performance/cost; avoid declaring improvement from one seed or mean score alone. Re-run any failure from its exact input and preserve the failing trace.
7. Only compare against an external benchmark when the task, information available, reset policy, memory budget, and cost accounting match. Do not compare this evidence-integrity patch directly to foundation-model task accuracy or call correctness parity “SOTA.”

## 6. Baselines and external context

- **Stateless baseline:** required for claims that the system learned from earlier instances rather than benefiting from its underlying capabilities.
- **No-gating ablation:** captures the failure mode where fallback numerics contaminate online updates.
- **Replay-only and fixed-window baselines:** help separate the value of evidence provenance from memory policy or more elaborate adaptation.
- **Resource-normalized comparison:** report quality at a fixed wall-clock/compute budget and time/cost to a fixed quality target.

Continual Learning Bench evaluates sequential related tasks and reports both task reward and *gain over the same system's stateless baseline*, alongside cost. That is the right style of evidence for claims that Symthaea improves from experience: [CL-Bench methodology](https://continual-learning-bench.com/docs/) and [2026 benchmark paper](https://arxiv.org/abs/2606.05661). For realistic memory utility, add Microsoft's [STATE-Bench](https://opensource.microsoft.com/blog/2026/05/19/introducing-state-bench-a-benchmark-for-ai-agent-memory/), which evaluates stateful task success, pass^5 reliability, operational efficiency, and user experience—not merely fact retrieval. Add [MemoryBench (ICML 2026)](https://proceedings.mlr.press/v306/ai26e.html) to test learning from accumulated feedback during service-time interaction. Use those as separate external tracks rather than blending their incomparable scores into PEV-1. The [public leaderboard](https://continual-learning-bench.com/leaderboard.html) is also a useful external reference, but it must not be treated as a direct comparison until Symthaea runs the same task sequences with the same reset, memory, and cost accounting. Current published results show that simple in-context-learning baselines can be highly competitive; greater architectural complexity is not itself evidence of improvement.

**Pinned external comparator (leaderboard snapshot updated 2026-07-18; refresh at evaluation time):** CL-Bench listed ICL with Claude Sonnet 4.6 at aggregate reward `+0.196`, aggregate gain `+0.241`, and cost `$30.43`; ICL with GPT-5.4 at `+0.189`, `+0.189`, and `$18.39`; Claude Code with Sonnet 4.6 at `+0.185`, `+0.241`, and `$38.60`; and Mem0 with GPT-5.4 at `+0.148`, `+0.224`, and `$18.34`. These are leaderboard aggregate metrics, not predictions about Symthaea's score. Use them as a pinned research reference only; any claim of superiority requires full task coverage and matched evaluation conditions.


A 2026 ICLR study, *Forget Forgetting*, also reports that in a regime with abundant replay memory but limited GPU time, the central tension may shift from memory efficiency to plasticity; simple replay can beat more complicated approaches at lower GPU cost. A complementary ICLR 2026 result, [IDER: Idempotent Experience Replay](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e34f98c14b861d6f24879e861fcbce1a-Abstract-Conference.html), makes idempotence and prediction reliability explicit objectives for replay. This suggests a new Symthaea experiment: replay the same evidence event twice and require no second PE-driven state transition, reward, replay-admission count, or thermodynamic charge. That cautions against assuming that more subsystems or more sophisticated memory automatically outperform a simple baseline: [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17fc58b1ba92abbb72c5f457e140d851-Abstract-Conference.html).

For systems performance, MLPerf Training measures time to a defined quality target and documents run-to-run variation, rather than treating an isolated throughput number as a complete comparison: [MLPerf Training](https://mlcommons.org/benchmarks/training/). The September 2026 v6.1 LLM post-training benchmark applies this to an agentic software-engineering workload, reporting wall-clock time to a fixed pass@4 target; it is a model for time-to-quality accounting, not a direct target for this evidence-integrity patch: [MLPerf Training v6.1 post-training announcement](https://mlcommons.org/2026/09/mlperf-training-llm-post-training/).

## 7. Acceptance policy

- Unit and property tests fail closed if an invalid sample mutates a PE-dependent history, learner, replay buffer, confidence value, urgency history, Experience Bus decision, or PE-derived thermodynamic cost.
- Exact-head formatting, build, unit tests, relevant feature combinations, and repository governance checks must pass before the draft is made ready.
- External validation must use a pre-registered evaluation protocol, repeated runs, a stateless baseline, and auditable artifacts.
- Any failure is retained as evidence; no test is weakened merely to regain a green status.
- No claim of beating the state of the art is justified until a comparable external benchmark has been run and results have been independently reproduced.

## 8. Current validation boundary

This document defines acceptance criteria; it is not evidence that the gates have passed. Regression assertions have been authored in the current branch, but the authoring environment has not executed `cargo test` or a release benchmark. Record the exact final PR head and actual test output when a supported build environment is available.


## 9. Typed subsystem-boundary admission

The compatibility `repr(C)` `CycleSnapshot` remains fixed-layout, but PE-dependent consumers should not treat its float and validity byte as separate inputs. The boundary adapter `prediction_error_evidence()` must construct a `PredictionErrorEvidence` with the snapshot's observation cycle and a non-usable status for non-canonical validity bytes or non-finite values. Since the ABI has no prediction-source-cycle field, this adapter must leave that provenance unknown rather than fabricate it.

The preferred cursor API, `admit_evidence(evidence, requested_observation_cycle)`, first records the requested cycle in an observed high-water mark when it is not stale, then checks evidence usability before admission. The observed and admitted watermarks are intentionally separate: invalid evidence remains unadmitted but prevents a delayed older sample from entering after a newer invalid observation; a valid retry at the same cycle remains possible. The legacy `admit(cycle, Option<f32>)` follows the same cursor policy but remains for compatibility, while new stateful PE consumers should use the typed method.

MemoryManager checkpoints use PEVC v2 to persist both watermarks. PEVC v1 checkpoints remain readable by using the previous admitted cycle as the observed floor; legacy nine-byte checkpoints have no cursor state and therefore restore without a cursor. Because v1 did not record invalid-only observations, it cannot reconstruct a newer watermark that was never serialized.

The follow-on LearningManager admission lane uses the same typed cursor for surprise-history updates: duplicate or out-of-order PE must not add history entries or change PE-derived error trends/plasticity, while non-PE signals such as arousal and dissipative health continue to operate. LearningManager retains its legacy 13-byte checkpoint prefix and appends a PEVC v2 extension; legacy 13-byte checkpoints restore without cursor provenance. This consumer-specific gate is in addition to the MemoryManager and NeuroevolutionManager boundaries, not a claim that every PE consumer has already migrated. The DriveManager follow-on applies the same admission contract to the rolling PE window, low-error boredom streak, flow streak/intensity, and adaptive exploration threshold. Its existing 25-byte checkpoint prefix remains stable; new checkpoints append PEVC v2 watermarks, while old 25-byte checkpoints restore with an empty cursor. Repeated-cycle fixtures in the drive tests must use distinct cycle numbers so test sequences model a real monotonic stream. The VisionManager follow-on gates only the PE contribution to visual surprise through the typed cursor. Invalid or repeated PE falls back to measured compressed-state variance and leaves that independent sensory channel live; its legacy 12-byte checkpoint prefix is retained with a PEVC v2 extension for new checkpoints. The PerceptionManager follow-on gates vigilance entry/exit decisions with the same cursor so a contradictory duplicate cycle cannot flip attention state twice. Budget pressure, coherence, and arousal corrections remain independent. Its legacy 13-byte checkpoint prefix is retained with a PEVC v2 extension. PEVC v2 serialization and validation are now centralized in `PredictionEvidenceCursor`; MemoryManager keeps its explicit PEVC v1 migration branch, while all v2 consumers share the same 23-byte encoder/decoder. This turns the extension's magic, version, presence flags, absent-value encoding, and admitted≤observed rule into one auditable contract.

### Additional deterministic gates

- Snapshot validity bytes `0`, `2`, and `255` never yield usable evidence, regardless of the float payload.
- A NaN or infinity with validity byte `1` is classified as non-finite and never yields a value.
- An evidence object requested for another observation cycle is rejected without advancing the admitted watermark; the requested cycle still advances the observed watermark if it is not older than the existing high-water mark.
- A newer invalid cycle followed by a delayed valid older cycle must reject the delayed sample, while a valid retry for the newer invalid cycle remains admissible once.
- The stateful LearningManager consumer admits valid PE once per observation cycle, rejects delayed older PE after a newer invalid observation, permits a valid same-cycle retry, and persists both watermarks across its checkpoint extension.
- The stateful DriveManager consumer admits each PE observation at most once for the error window, boredom/flow streaks, and exploration-threshold updates; checkpoint round-trip preserves both watermarks.
- VisionManager uses PE only when the evidence is valid for the current cycle and has not already been admitted; invalid/replayed PE cannot be injected into the visual-surprise fusion, while compressed-state variance remains active.
- PerceptionManager only allows an admitted, current-cycle PE sample to enter or exit vigilance; duplicate/stale PE cannot toggle attention, and independent budget/coherence/arousal signals remain active.
- All PEVC v2 checkpoint extensions use the same encoder/decoder and validation rules; malformed flags, absent-but-nonzero cycles, invalid magic/version, wrong length, and admitted cycles beyond the observed high-water mark are rejected consistently.
- Checkpoint/restore must preserve observed and admitted watermarks independently; legacy PEVC v1 migration must retain its known admitted-cycle floor.
- A fixed-seed 10,000-entry mixed trace interleaves valid samples, unavailable evidence, non-canonical flags, duplicate and out-of-order cycle numbers, and requested-cycle mismatches. Each admission decision and the cursor state must match an independent reference rule.
- This trace is a cursor/adapter contract test, not a substitute for the 15 consumer-level counterfactual cases. End-to-end state-digest comparisons across downstream learners, replay stores, safety, and checkpoint restoration remain required.

The above are acceptance tests, not measured results until executed against the exact tested commit with a supported Rust toolchain.


## 10. Research experiment: reliability-adjusted replay after hard evidence admission

**Hypothesis.** Once invalid evidence is excluded by the typed contract, valid samples can still differ in predictive reliability. A replay policy that separates evidence eligibility from reliability may improve learning per unit cost over uniform replay and conventional priority-by-error.

This is an experiment proposal, not a production policy or a claim that Symthaea beats an external method. It is motivated by ICLR 2026 work on [Reliability-Adjusted Prioritized Experience Replay (ReaPER)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/60e3d7caf8cfdaf6598467f9b0ebf36a-Abstract-Conference.html), which studies replay prioritization using temporal-difference error reliability, and by [Forget Forgetting](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17fc58b1ba92abbb72c5f457e140d851-Abstract-Conference.html), which reports that simple replay can be highly competitive when memory is abundant but compute is constrained. [IDER](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e34f98c14b861d6f24879e861fcbce1a-Abstract-Conference.html) further motivates testing repeated-event idempotence and prediction reliability independently.

### Candidate ablations

1. **Uniform replay:** sample from the eligible buffer uniformly.
2. **Error-prioritized replay:** conventional priority proportional to a positive offset plus absolute prediction error, raised to exponent \(\alpha\).
3. **Reliability-adjusted replay:** scale error priority by an independently calibrated per-sample reliability estimate, while retaining an explicit novelty/shift channel so out-of-distribution observations are not automatically suppressed.
4. **Matched no-replay/stateless baseline:** estimate the contribution of accumulated experience itself.
5. **Invalid-event negative control:** deliberately inject invalid evidence into the harness, but require all policies to exclude it before sampling. If any arm admits an invalid event, the entire run is disqualified.

One candidate sampling mixture for eligible samples is:

\[
P(i)=(1-\lambda)\,\operatorname{Normalize}((\epsilon+|\delta_i|)^\alpha r_i^\beta)
+\lambda\,\operatorname{Normalize}(n_i)
\]

where \(r_i\) is the calibrated reliability of a valid sample, \(n_i\) is an independently defined novelty/shift score, and \(\lambda\) is fixed before evaluation. Invalid evidence is excluded by eligibility before this equation; reliability is not a substitute for validity. The formula is a test candidate, not an approved production algorithm.

### Acceptance and reporting

- **Hard gate:** zero invalid replay admissions and zero duplicate state transitions across all arms, including replay after checkpoint/restore.
- **Task utility:** positive paired gain over matched stateless and uniform-replay baselines, with per-task results and confidence intervals.
- **Calibration:** report Brier score or another appropriate proper scoring rule, calibration error, and coverage; reliability estimates must be trained/calibrated on held-out outcomes, not on the test stream.
- **Shift robustness:** measure adaptation speed after task/domain shifts and report whether reliability weighting suppresses genuinely novel but initially uncertain evidence.
- **Retention and cost:** report forward/backward transfer, forgetting, valid samples admitted, updates per task, wall-clock time to a fixed quality target, CPU/GPU time, and peak memory.
- **Protocol:** same stream order, task resets, buffer capacity, optimizer/training budget, and seeds for all methods. Use at least 10 paired seeds for the internal study and preserve every failure trace. Report mean and dispersion as well as worst-task outcomes.

This experiment can establish whether reliability-aware replay helps Symthaea's chosen workloads. It cannot establish a general SOTA win without running the external benchmark's complete protocol under comparable conditions.
