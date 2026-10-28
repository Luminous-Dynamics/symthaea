# PEV-2 — Decision-time evidence binding for prediction-error learning

**Status:** research contract / follow-on specification. This is not yet a production implementation or a measured Symthaea result.

**Tracks:** [Issue #8714](https://github.com/Luminous-Dynamics/symthaea/issues/8714), [CodingAgent PE/confidence separation PR #8700](https://github.com/Luminous-Dynamics/symthaea/pull/8700), [cognitive-loop evidence benchmark PR #8495](https://github.com/Luminous-Dynamics/symthaea/pull/8495).

## 1. Paradigm shift

Move from **“a decision has a PE number”** to **“a decision references the exact evidence snapshot that was available when it was made.”**

An optional scalar preserves missingness, but it does not prove which prediction/observation pair produced it, whether the sample was stale or replayed, or whether an outcome accidentally attached a later value. Decision-time evidence must be immutable and provenance-bearing.

> A posthoc outcome may resolve later, but it may not rewrite what evidence the system possessed at decision time.

Thus: **missing at decision time is not zero PE, maximum PE, or later PE.**

A decision outcome remains recordable when PE is unavailable. Missing PE must not disable independent reward, coherence, uncertainty, or gate-result calibration.

## 2. Evidence contract

Each decision freezes a bounded metadata snapshot, conceptually equivalent to:

    struct DecisionEvidenceSnapshotV1 {
        decision_id: DecisionId,
        decision_tick: u64,
        observation_cycle: u64,
        prediction_source_cycle: Option<u64>,
        status: PredictionEvidenceStatus,
        prediction_error: Option<f64>,
        evidence_event_id: Option<EvidenceEventId>,
        evidence_digest: Option<[u8; 32]>,
    }

This is a proposed contract, not a claim that these exact types already exist. Construction must reject contradictions: Valid requires finite PE and a cycle-consistent source; non-valid statuses cannot expose a usable PE. A legacy scalar API may remain as an explicit adapter, but it must not fabricate source cycles or a digest.

The digest should commit to canonical, domain-separated metadata (stable IDs, status, cycles and value representation as appropriate). It should not hash or log raw user text by default. A digest binds a record to metadata; it does not prove the honesty of the process that generated the record.

PendingReasoningOutcome freezes the snapshot when the decision is created. PosthocOutcome retains the same snapshot reference when the outcome arrives. The outcome timestamp may be later; the evidence reference must not be.

## 3. Required invariants

| ID | Invariant | Required result |
|---|---|---|
| DTA-01 | Decision-time freeze | Later PE cannot replace or fill an earlier decision's PE snapshot. |
| DTA-02 | Missingness | Invalid PE remains unavailable; compatibility sentinels cannot become evidence. |
| DTA-03 | Source/observation ordering | Stale, future, out-of-order, or cross-reset samples are rejected or explicitly classified. |
| DTA-04 | Exactly-once | Replayed evidence IDs produce at most one PE-driven admission. |
| DTA-05 | Counterfactual invariance | Changing only an invalid compatibility payload does not change PE-dependent outputs. |
| DTA-06 | Independent channels | Missing PE does not suppress coherence, uncertainty, explicit reward, or gate/outcome results. |
| DTA-07 | Positive control | Fresh valid PE reaches its intended PE-sensitive consumer once; a no-op system fails. |
| DTA-08 | Recovery | After a missing, stale, or invalid event, a later fresh valid event is admitted without restart. |
| DTA-09 | Resume equivalence | Checkpoint/restore preserves admission identity and matches uninterrupted execution. |
| DTA-10 | Trace closure | Every PE-driven transition references one admitted event; invalid, duplicate, stale, or future events have no downstream influence. |

## 4. Adversarial benchmark panel

Use paired deterministic replays with identical initial state, seed, exogenous observations, actions, and non-PE channels. Change one evidence condition at a time.

1. **Invalid-payload metamorphism:** repeat an invalid record with 0, 1, NaN, positive/negative infinity, and absence. Compare PE-sensitive histories, learning value, action/readiness signals, and outcome records with a no-PE control.
2. **Future-PE substitution:** make a decision without PE, then deliver high valid PE in the next cycle before outcome resolution. The original decision remains PE-missing; the later PE stays attached to its own event.
3. **Valid-value freeze:** decide with valid PE p; later deliver q; posthoc resolution must still reference p and its event ID.
4. **Replay burst:** replay a valid event 2–6 times, including across snapshot/restore; PE-driven admission count remains one.
5. **Temporal/reset faults:** inject stale, future, out-of-order, wrong-epoch, and pre-reset evidence; prove fail-closed handling and subsequent recovery.
6. **Independent-channel negative control:** remove PE while retaining coherence, uncertainty, and explicit reward. Independent outputs must match the corresponding valid-PE-free control.
7. **Positive control:** inject valid PE in a scenario where the consuming policy should respond. Verify a traceable response, ruling out “pass” by disabling all learning.
8. **Trace-level audit:** independently reconstruct evidence references from raw events; reject duplicate IDs, timestamp inversions, unresolved source references, and decision snapshots changed after creation.

## 5. Acceptance gates and reported metrics

These are **proposed targets**, not measured Rust performance or published SOTA results.

### Hard correctness gates

- Zero invalid-sentinel leakage, duplicate admissions, stale/future leakage, and orphan PE-driven transitions in the fixed panel and at least 100,000 generated event cases.
- 100% detection of five injected implementation defects: sentinel promotion, duplicate replay admission, stale-cycle admission, future-PE substitution, and suppression of an independent channel.
- Exact equality of PE-dependent outputs for invalid-payload counterfactuals; exact decision-time evidence linkage for posthoc records.
- Positive-control sensitivity and independent-channel preservation both pass.
- Rejected cases report a classified reason; no missing case is silently counted as valid zero/high error.

### Utility and cost — only after hard gates pass

Report task loss, adaptation gain, time-to-recovery, forgetting/retention, false PE-driven updates, decision/gate outcome calibration, wall-clock latency, and memory overhead. Use paired seeds and identical evidence-access tracks, held-out tasks, dataset-macro effects and confidence intervals. Keep raw counts and failure traces; do not collapse everything into one weighted score.

Proposed initial engineering budget: at most 5% p95 latency overhead for provenance instrumentation, measured against the same build and workload. If it is exceeded, report the trade-off rather than weakening provenance or correctness gates.

## 6. Relation to state of the art and industry

This audit is **orthogonal** to the drift-detector benchmark in #8495. KDD 2026's drift-evaluation work compares 14 methods across seven real-world datasets, four drift types, abrupt/gradual transitions, and advocates leave-one-dataset-out tuning ([paper DOI](https://doi.org/10.1145/3770855.3819070)). That is the appropriate rigor for detector-performance claims. PEV-2 asks whether the evidence and outcome attribution are trustworthy before comparing the detector's utility.

Comparison sequence:

1. Establish evidence correctness and mutation sensitivity.
2. Run the same pinned streams and information-access track for candidate and baselines.
3. Compare utility, false alarms, recovery, forgetting, and cost with paired uncertainty intervals.
4. Publish exact configs, data digests, seeds, raw traces, and negative controls.

A typed contract or clean synthetic fixture does **not** prove better adaptation, general intelligence, consciousness, or SOTA performance. Beating an industry baseline requires a matched full protocol and measured gains, not more internal checks alone.

## 7. Reference-model smoke experiment (not product verification)

The authoring-side Python reference model checks the contract against generated/adversarial traces. Its output is recorded separately so model-only checks are not confused with Rust tests:

- 25,000 invalid-payload counterfactuals: 25,000 rejected without PE history; exact invalid-payload invariance.
- 10,000 events over 39,838 admission attempts: exactly 10,000 unique admissions; 29,838 replays rejected.
- 5,000 checkpoint round trips: resumed state equals uninterrupted state exactly.
- Five deliberately injected faults: all five detected.
- Stale and pre-reset evidence rejected; fresh evidence after reset admitted.
- Later PE did not backfill a prior decision that lacked PE; outcome calibration was preserved.

This is a reference-model smoke test only. It does not compile or run Symthaea, prove the Rust call graph, establish real runner honesty, or measure predictive utility. Production acceptance gates remain pending until Rust implementation and exact-head workflows execute.

## 8. Implementation path

1. Extend CodingAgent pending outcomes to store the frozen decision-time evidence snapshot/reference; do not synthesize provenance for legacy scalar callers.
2. Propagate the immutable reference through posthoc resolution and telemetry.
3. Add focused Rust unit and integration tests for DTA-01 through DTA-10, including the positive control.
4. Add an independent raw-trace auditor and deterministic generated-case test in Rust; keep its scoring separate from model utility.
5. Validate at the exact commit with formatting, compile, focused tests, relevant full suite, and workflow evidence. Queued, skipped, or stale-head runs are not passes.

## 9. External benchmark frontier: correctness is not utility

Current research suggests keeping the evidence-integrity question separate from the learning-algorithm question:

- **Concept-drift detection:** Cerqueira et al., *A Framework for Evaluating and Benchmarking Concept Drift Detection Methods* (KDD 2026) evaluates 14 detector families across seven real-world datasets, four drift types, and abrupt/gradual transitions. It highlights timing-aware metrics and leave-one-dataset-out tuning. Use its stream/dataset protocol for detector claims instead of inventing a separate drift score: [paper](https://doi.org/10.1145/3770855.3819070), [arXiv version](https://arxiv.org/abs/2606.07789).
- **Reliability-aware reinforcement-learning replay:** ReaPER reports improved performance over uniform replay and prioritized experience replay in its evaluated classic-control and Atari-10 settings. It is a comparator only for a matched RL/replay experiment, not for proving provenance correctness: [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/60e3d7caf8cfdaf6598467f9b0ebf36a-Abstract-Conference.html).
- **Continual-learning reliability:** IDER uses idempotence-oriented training and distillation and reports improvements in prediction reliability, accuracy, and forgetting on its evaluated continual-learning benchmarks. Compare it only in a compatible continual-learning setting and reproduce the relevant baseline configuration: [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e34f98c14b861d6f24879e861fcbce1a-Abstract-Conference.html).
- **Memory-rich regimes:** *Forget Forgetting: Continual Learning in a World of Abundant Memory* argues that when storage is abundant but GPU time is scarce, strong simple replay can outperform more elaborate methods at lower compute cost. This motivates reporting compute and wall-clock efficiency, not only average accuracy: [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17fc58b1ba92abbb72c5f457e140d851-Abstract-Conference.html).

These results are from different task families and are **not directly rankable on a single leaderboard**. The proposed evaluation therefore has three separate scorecards:

1. **Integrity:** exact counterfactual invariance, zero invalid/stale/future/duplicate PE influence, mutation sensitivity, positive controls, and decision-time trace closure.
2. **Detection:** timing-aware drift F1, false alarms per 1,000 valid observations, missed-drift rate and normalized detection delay under matched information access.
3. **Learning utility and cost:** held-out task score, calibration, adaptation/regret, retention/forgetting, recovery time, optimizer/update cost, p95 latency, memory and energy.

The first scorecard is a hard admission gate for claiming evidence correctness. It must not be added to the utility score to compensate for task failure, and passing it does not make a SOTA claim. The second and third scorecards require real comparator implementations, pinned datasets/configurations, paired seeds and uncertainty intervals; the currently available synthetic reference results do not fill those cells.

### Proposed falsifiable superiority criterion

After correctness gates pass, pre-register a practical superiority target: improve the **dataset-macro primary utility metric by at least 5% relative to the strongest matched baseline**, with a paired 95% confidence interval excluding zero, while meeting the same access track and compute/memory budget. Report all datasets, worst-task results and negative controls. If this threshold is not met, publish the result as non-superior or inconclusive; do not change the threshold after seeing held-out results. The 5% target is a proposed engineering criterion, not a literature-derived universal standard.
