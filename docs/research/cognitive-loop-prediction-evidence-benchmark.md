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

### Typed-contract regression checks

The typed layer includes unit tests for invalid numeric payloads (`0`, `1`, NaN, and positive/negative infinity), non-finite values incorrectly flagged valid, cycle mismatch, stale-status classification, constructor invariants, valid serialization round-trip, malformed serialized records, and a deterministic 10,000-payload boundary sweep. Consumer-focused tests cover stale and missing evidence in urgency and surprise paths, plus the encoder's consecutive-cycle acceptance, skipped-cycle rejection, non-finite prediction rejection before PE/attention statistics, explicit service reset of prediction provenance, and cold-start metadata on the pre-encoded-HV entry path. The main text-loop test verifies source-cycle handoff, and output tests check that public telemetry agrees with the compatibility validity bit and records the observation cycle. Keep these separate from the 15 core PEV scenarios so passing type-contract tests cannot be mistaken for coverage of all downstream consumers. These are authored checks, not a pass claim until run at the exact head.

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

Continual Learning Bench evaluates sequential related tasks and reports both task reward and *gain over the same system's stateless baseline*, alongside cost. That is the right style of evidence for claims that Symthaea improves from experience: [CL-Bench methodology](https://continual-learning-bench.com/docs/) and [2026 benchmark paper](https://arxiv.org/abs/2606.05661). The [public leaderboard](https://continual-learning-bench.com/leaderboard.html) is also a useful external reference, but it must not be treated as a direct comparison until Symthaea runs the same task sequences with the same reset, memory, and cost accounting. Current published results show that simple in-context-learning baselines can be highly competitive; greater architectural complexity is not itself evidence of improvement.

**Pinned external comparator (leaderboard snapshot updated 2026-07-18; refresh at evaluation time):** CL-Bench listed ICL with Claude Sonnet 4.6 at aggregate reward `+0.196`, aggregate gain `+0.241`, and cost `$30.43`; ICL with GPT-5.4 at `+0.189`, `+0.189`, and `$18.39`; Claude Code with Sonnet 4.6 at `+0.185`, `+0.241`, and `$38.60`; and Mem0 with GPT-5.4 at `+0.148`, `+0.224`, and `$18.34`. These are leaderboard aggregate metrics, not predictions about Symthaea's score. Use them as a pinned research reference only; any claim of superiority requires full task coverage and matched evaluation conditions.


A 2026 ICLR study, *Forget Forgetting*, also reports that in a regime with abundant replay memory but limited GPU time, the central tension may shift from memory efficiency to plasticity; simple replay can beat more complicated approaches at lower GPU cost. That cautions against assuming that more subsystems or more sophisticated memory automatically outperform a simple baseline: [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17fc58b1ba92abbb72c5f457e140d851-Abstract-Conference.html).

For systems performance, MLPerf Training measures time to a defined quality target and documents run-to-run variation, rather than treating an isolated throughput number as a complete comparison: [MLPerf Training](https://mlcommons.org/benchmarks/training/). The September 2026 v6.1 LLM post-training benchmark applies this to an agentic software-engineering workload, reporting wall-clock time to a fixed pass@4 target; it is a model for time-to-quality accounting, not a direct target for this evidence-integrity patch: [MLPerf Training v6.1 post-training announcement](https://mlcommons.org/2026/09/mlperf-training-llm-post-training/).

## 7. Acceptance policy

- Unit and property tests fail closed if an invalid sample mutates a PE-dependent history, learner, replay buffer, confidence value, urgency history, Experience Bus decision, or PE-derived thermodynamic cost.
- Exact-head formatting, build, unit tests, relevant feature combinations, and repository governance checks must pass before the draft is made ready.
- External validation must use a pre-registered evaluation protocol, repeated runs, a stateless baseline, and auditable artifacts.
- Any failure is retained as evidence; no test is weakened merely to regain a green status.
- No claim of beating the state of the art is justified until a comparable external benchmark has been run and results have been independently reproduced.

## 8. Current validation boundary

This document defines acceptance criteria; it is not evidence that the gates have passed. Regression assertions have been authored in the current branch, but the authoring environment has not executed `cargo test` or a release benchmark. Record the exact final PR head and actual test output when a supported build environment is available.
