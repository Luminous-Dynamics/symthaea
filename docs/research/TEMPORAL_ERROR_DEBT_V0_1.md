# Temporal Error Debt (TED) evaluator contract v0.2

**Status:** repeated-checkpoint task generator and evaluator scorer authored; Rust tests not run  
**Tracking:** NEURO-SILICON-008 / [issue #9492](https://github.com/Luminous-Dynamics/symthaea/issues/9492)  
**Related:** T10 bitemporal knowledge stream and NEURO-SILICON-007 / #9394  
**Code:** `research/temporal-bench/src/knowledge_trajectory.rs`, `temporal_error_debt.rs`

## 1. Why another metric

Aggregate accuracy and final-state correctness do not measure how long a system remains wrong after evidence changes. Two systems can reach the same final answer but differ materially in the duration for which they emit a stale answer.

Temporal Error Debt is a time-integrated answer-state metric for continuous knowledge streams. It supplements—not replaces—exact accuracy, correction latency, abstention, missing-output, leakage, and energy reporting.

## 2. Two-clock truth

For a query with fixed valid time (v), the evaluator's correct answer at knowledge time (t) is the newest revision satisfying both:

[
valid_from(r) le v
quadlandquad
observed_at(r) le t.
]

A late correction may already be valid in the modeled world but not yet known to the candidate. A retraction may be valid from an earlier world time but must not affect the answer until its transaction/observation time is reached.

The scorer independently resolves the target assertion, correction and retraction from evaluator-owned oracle metadata; it does not copy the five T10 anchor labels for each interval.

## 3. Repeated-checkpoint T10 trajectory

The new `knowledge_trajectory.rs` generator preserves the original five T10 anchor queries and emits a separate repeated query stream over the same append-only fact history.

It creates two stable candidate-visible query lanes at each checkpoint:

1. **Current valid-time lane:** `valid_at` is fixed just after the retraction's valid-from boundary. Its correct answer transitions from initial value to corrected value to `NoFact` as updates become known.
2. **Historical-as-known lane:** `valid_at` is fixed before the correction's valid-from boundary. Its correct answer remains the initial value even after the later correction and retraction are known.

For each checkpoint, the stream emits both queries with distinct event IDs and stable event ordering. The evaluator publishes a checkpoint binding with the candidate-visible event IDs, fixed valid-time coordinates and knowledge-time checkpoint. It also derives one exposure window per lane.

### Checkpoint cadence

The checkpoint stride is selected as an exact divisor of the source fact-update interval, aiming for approximately four observations per update interval:

[
stride = gcd(interval,max(1,lfloor interval/4 floor)).
]

For the reference fixture with 16-step update intervals, the stride is four steps. Checkpoints run from knowledge time zero through the final anchor query horizon. Each answer is held for one checkpoint stride, including the final checkpoint; that terminal hold is explicit in the generated exposure windows.

This finer schedule is important: an update delayed by one checkpoint should incur a measurable four-step debt rather than being hidden between two coarse update-time snapshots.

The trajectory verifier checks the outer public-task digest, inner stream digest, strict event ordering, every query ID/ordinal/time/valid-time binding, and then reconstructs the original five-query T10 anchor stream and passes it through the original independent bitemporal verifier. The seeded verifier additionally regenerates the complete task and oracle using evaluator-held configuration.

## 4. Exposure-window contract

Each evaluator window contains:

- `query_id`: stable question/fact trajectory identity (current or historical lane);
- `checkpoint_index`: zero-based contiguous index within that query trajectory;
- `valid_at`: fixed valid-time coordinate for the entire trajectory;
- `known_at`: knowledge/transaction-time checkpoint;
- `duration`: interval length in an explicitly declared integer unit.

For one query, checkpoint indices start at zero and advance by one. Every subsequent `known_at` equals the previous `known_at + duration`. Durations are positive and interval ends must not overflow. A trajectory cannot change `valid_at) halfway through. Different query lanes are scored independently; summing their exposure discloses the number of lanes/queries represented.

Each candidate prediction has `(query_id, checkpoint_index, answer)`. Candidate answer states are:

- `Value(x)`: explicit assertion of a value;
- `NoFact`: explicit assertion that no fact applies;
- `Abstain`: explicit refusal to provide a substantive answer;
- absent prediction record: missing output.

The prediction at a checkpoint is treated as constant over that exposure window. The evaluator creates windows from generated checkpoints; it does not infer wall-clock time. When real durations are absent, report chunk-time units, not seconds.

## 5. Exposure definitions

Let expected(v, t_i) be the evaluator-resolved answer and prediction_i be the candidate state for interval i with duration delta_t_i.

[
TED_{wrong} = sum_i Delta t_i cdot
mathbf{1}[prediction_i 
e expected(v,t_i)]
]

where:
- Value(x) is correct iff expected(v, t_i) equals x;
- NoFact is correct iff expected(v, t_i) is empty;
- Abstain is never counted as a correct substantive answer;
- missing is represented by no prediction record.

Three top-level components remain separate:

1. **Wrong-belief exposure:** duration of explicit but incorrect Value/NoFact states.
2. **Abstention exposure:** duration of explicit Abstain states.
3. **Missing-output exposure:** duration for which no prediction was received.

The scorer also reports correct exposure and wrong-belief subsets:
- future-knowledge leakage exposure, including a future correction answer or premature NoFact from a not-yet-known retraction;
- missed-correction exposure, including the old value or NoFact after the correction is known but before retraction is known;
- stale-after-retraction exposure, meaning an explicit value after the retraction is both valid and known.

Correction latency is the knowledge-time lag from correction arrival to the first checkpoint that gives the corrected value for a valid-time coordinate to which the correction applies. Retraction latency is the lag to the first correct NoFact state.

**Answer oscillations** count candidate state changes between adjacent checkpoints when the evaluator's expected state for that fixed valid-time query has not changed. Missing and abstention are explicit distinct states for this check.

Do not collapse any exposure component, latency, leakage count or oscillation count into a weighted aggregate.

## 6. Worked synthetic example

With four distractors and source update interval 16:
- correction observed at knowledge time 80;
- correction valid from time 32;
- retraction observed at knowledge time 96;
- retraction valid from time 48;
- repeated checkpoint stride is four time units.

For a current-lane query at valid time 49, the checkpoints surrounding the updates include [76,80), [80,84), [84,88), ..., [96,100).

An event-aligned candidate emits:
- initial value before knowledge time 80;
- corrected value beginning at knowledge time 80;
- NoFact beginning at knowledge time 96.

It accumulates zero wrong-belief, abstention, missing, future-leak, missed-correction and stale-after-retraction exposure. Correction and retraction latency are both zero.

A candidate that keeps the initial value for one checkpoint after correction accumulates four units of wrong-belief and missed-correction exposure and has correction latency four. A candidate that keeps the corrected value for one checkpoint after retraction accumulates four units of stale-after-retraction exposure and has retraction latency four. Both can eventually reach the same final answer as the event-aligned candidate.

A model that emits the corrected value before knowledge time 80 accumulates future-knowledge leakage exposure. A model that emits NoFact after correction but before the retraction is known also incurs wrong-belief and premature-future-retraction exposure.

## 7. Implemented Rust API

The standalone modules expose:
- `KnowledgeTrajectoryTask` and `KnowledgeCheckpointBinding`;
- `generate_knowledge_trajectory`;
- `verify_knowledge_trajectory` and `verify_knowledge_trajectory_seeded`;
- `knowledge_trajectory_windows`;
- `KnowledgeExposureWindow` and `KnowledgeTimedPrediction`;
- `TemporalErrorDebtScore` and `score_temporal_error_debt`;
- `resolve_expected_value`.

The scorer rejects empty schedules, zero-duration windows, time overflow, non-contiguous intervals, changing valid-time coordinates, duplicate predictions and unknown prediction keys. Missing predictions are inferred from the full generated window schedule. It reports full-window coverage, exposure components, correction/retraction latency, future-knowledge leakage, missed-correction and stale-after-retraction exposure, plus answer oscillations.

The trajectory generator emits both current and historical lanes at every checkpoint. Its semantic verifier reconstructs and verifies the original T10 five-query anchor stream before any score is accepted. The seeded verifier binds the repeated task/oracle back to evaluator-held seed/configuration.

## 8. Test and validation status

Thirteen TED scorer unit tests and eight repeated-trajectory unit tests are authored, including:
- event-aligned zero debt;
- one-checkpoint correction and retraction delays;
- equal final answer with unequal time-integrated debt;
- future correction and future retraction leakage;
- abstention versus missing output;
- historical-as-known preservation;
- answer oscillations without new evidence;
- query schedule and event-order mutation controls;
- seed-replay rejection after a recommitted source-history mutation.

These tests have **not** been compiled or run in this environment; no Rust test pass is claimed.

## 9. What remains

- Integrate the repeated task, scorer and evaluator-only oracle into the isolated evaluator process in #9437.
- Bind candidate output deadlines, stream hashes, model/runtime digests and TED results into evaluator-signed receipts.
- Add calibrated hardware energy per stream and per unit of wrong-belief exposure avoided.
- Run the Rust tests on a pinned Rust toolchain, then compare HDC-LTC against matched recurrent, SNN and continual-memory baselines.
- For OAKS, use official per-chunk scoring plus a separate TED trajectory only after resolving the official data manifest and noncommercial-use license. Never turn chunk indices into wall-clock seconds.

## 10. Claim boundary

TED measures time-integrated answer-state correctness for the declared query trajectory, update semantics and duration units. It does not measure real-world harm without independently validated consequence weights. It is not an official OAKS score and does not establish general continual-learning superiority.
