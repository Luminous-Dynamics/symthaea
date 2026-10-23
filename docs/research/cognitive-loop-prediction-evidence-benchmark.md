# Cognitive-loop prediction-evidence benchmark (PEV-1)

**Status:** proposed benchmark contract; implementation tests are being added, but this specification has not yet been run end-to-end.

## 1. Research claim

A prediction-error (PE) number is not evidence by itself. It must carry provenance stating whether the current predicted and observed states form a comparable, finite measurement. A compatibility sentinel such as `1.0` may remain useful for diagnostics, but it must not silently become surprise, reward, confidence, urgency, replay importance, a learning target, or a physical-cost estimate.

The core invariant is:

> **Invalid PE has zero influence on PE-dependent state transitions, while independent evidence channels continue to function.**

This is narrower and more testable than a claim about consciousness, general intelligence, or state-of-the-art performance.

## 2. Counterfactual test matrix

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

Where a subsystem is independent of PE, the expected result is not “no change to the entire system.” The invariant applies only to the PE-derived component; coherence, energy, explicit reward, sensory variance, and other valid signals remain eligible to change state.

## 3. Metrics

Report raw counts as well as rates. Do not collapse all dimensions into one score.

- **Invalid-evidence leakage rate:** PE-dependent state transitions caused by invalid observations / invalid observations. Gate: **0 in deterministic tests** and **0 over at least 10,000 generated invalid/interleaved cases**.
- **False replay admission:** invalid transitions admitted to episodic replay. Gate: **0**.
- **False-surprise rate:** invalid PE causing a PE-only surprise, vigilance, exploration, or PROBE decision. Gate: **0**.
- **Valid-evidence preservation:** proportion of finite, comparable observations that reach each intended PE consumer exactly once. Gate: **100% on deterministic fixtures**.
- **Recovery latency:** number of valid observations required before the system resumes PE-driven updates after missingness. Target: **one valid observation**, without blending it with an invalid sentinel.
- **Independent-channel preservation:** matched cases where coherence, sensory variance, explicit reward, or physical-energy evidence should act. Gate: intended non-PE transition remains present.
- **Stale-evidence reuse:** count of previously valid PE samples applied as a fresh observation in later invalid cycles. Gate: **0**.
- **Resource cost:** cycle latency (p50/p95/p99), CPU time, peak memory, replay-buffer size, and incremental serialized bytes. Report measured baselines and deltas; do not set a performance claim before measurement.

For stream-learning experiments also report per-step task performance, forward/backward transfer, forgetting, calibration (Brier score/ECE on resolved outcomes), number of admitted learning samples, time-to-target quality, and compute cost. Report worst-task and dispersion alongside aggregate scores.

## 4. Experimental protocol

1. Freeze the code SHA, toolchain, feature flags, configuration, seed set, input stream, and expected result digest before measuring.
2. Run deterministic unit/counterfactual cases before long sequences. A skipped or queued workflow is not a pass.
3. Use at least 10 fixed seeds for the first aggregate experiment. Include cold starts, repeated missingness, valid/invalid alternation, low-to-high and high-to-low shifts, and distractor tasks.
4. Compare these ablations on identical input streams: (a) current legacy numeric-sentinel behavior, (b) explicit validity gates, (c) skip all learning on invalid cycles, and (d) use only independently measured signals. Keep non-PE subsystems and compute budgets identical.
5. Publish JSON/CSV raw records with seed, stream index, validity reason, encoder degeneracy, admitted-update counters, per-consumer before/after state digests, runtime, and memory. Include negative controls and the verifier's exit status.
6. Use paired per-seed differences and confidence intervals for performance/cost; avoid declaring improvement from one seed or mean score alone. Re-run any failure from its exact input and preserve the failing trace.
7. Only compare against an external benchmark when the task, information available, reset policy, memory budget, and cost accounting match. Do not compare this evidence-integrity patch directly to foundation-model task accuracy or call correctness parity “SOTA.”

## 5. Baselines and external context

- **Stateless baseline:** required for claims that the system learned from earlier instances rather than benefiting from its underlying capabilities.
- **No-gating ablation:** captures the failure mode where fallback numerics contaminate online updates.
- **Replay-only and fixed-window baselines:** help separate the value of evidence provenance from memory policy or more elaborate adaptation.
- **Resource-normalized comparison:** report quality at a fixed wall-clock/compute budget and time/cost to a fixed quality target.

Continual Learning Bench evaluates sequential related tasks and reports both task reward and *gain over the same system's stateless baseline*, alongside cost. That is the right style of evidence for claims that Symthaea improves from experience: [CL-Bench methodology](https://continual-learning-bench.com/docs/) and [2026 benchmark paper](https://arxiv.org/abs/2606.05661).

A 2026 ICLR study, *Forget Forgetting*, also reports that in a regime with abundant replay memory but limited GPU time, the central tension may shift from memory efficiency to plasticity; simple replay can beat more complicated approaches at lower GPU cost. That cautions against assuming that more subsystems or more sophisticated memory automatically outperform a simple baseline: [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17fc58b1ba92abbb72c5f457e140d851-Abstract-Conference.html).

For systems performance, MLPerf Training measures time to a defined quality target and documents run-to-run variation, rather than treating an isolated throughput number as a complete comparison: [MLPerf Training](https://mlcommons.org/benchmarks/training/).

## 6. Acceptance policy

- Unit and property tests fail closed if an invalid sample mutates a PE-dependent history, learner, replay buffer, confidence value, urgency history, Experience Bus decision, or PE-derived thermodynamic cost.
- Exact-head formatting, build, unit tests, relevant feature combinations, and repository governance checks must pass before the draft is made ready.
- External validation must use a pre-registered evaluation protocol, repeated runs, a stateless baseline, and auditable artifacts.
- Any failure is retained as evidence; no test is weakened merely to regain a green status.
- No claim of beating the state of the art is justified until a comparable external benchmark has been run and results have been independently reproduced.

## 7. Current validation boundary

This document defines acceptance criteria; it is not evidence that the gates have passed. Regression assertions have been authored in the current branch, but the authoring environment has not executed `cargo test` or a release benchmark. Record the exact final PR head and actual test output when a supported build environment is available.
