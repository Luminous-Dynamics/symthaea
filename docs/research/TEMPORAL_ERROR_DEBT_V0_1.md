# Temporal Error Debt (TED) evaluator contract (v0.1)

**Status:** evaluator scorer implemented; repeated-checkpoint task generation and runner integration not implemented  
**Tracking:** NEURO-SILICON-008 / issue #9492  
**Related:** T10 bitemporal knowledge stream in #9394 / #9395  
**Code:** `research/temporal-bench/src/temporal_error_debt.rs`

## 1. Why another metric

Aggregate accuracy and final-state correctness do not measure how long a system remains wrong after evidence changes. Two systems can reach the same final answer but differ materially in the duration for which they emit a stale answer.

Temporal Error Debt is a time-integrated answer-state metric for continuous knowledge streams. It supplements—not replaces—exact accuracy, correction latency, abstention, missing-output, leakage, and energy reporting.

## 2. Two-clock truth

For a query with fixed valid time (v), the evaluator's correct answer at knowledge time (t) is the newest revision satisfying both:

[
valid\_from(r) \le v
\quad\land\quad
observed\_at(r) \le t.
]

A late correction may already be valid in the modeled world but not yet known to the candidate. A retraction may be valid from an earlier world time but must not affect the answer until its transaction/observation time is reached.

The implementation independently resolves the target assertion, correction and retraction from the evaluator-owned oracle metadata; it does not copy the five T10 anchor labels for each interval.

## 3. Exposure-window contract

An evaluator supplies repeated windows for each query:

- `query_id`: stable question/fact query identity;
- `checkpoint_index`: zero-based contiguous index within that query trajectory;
- `valid_at`: fixed valid-time coordinate for the entire trajectory;
- `known_at`: knowledge/transaction-time checkpoint;
- `duration`: interval length in an explicitly declared integer unit.

For one query, checkpoint indices must start at zero and advance by one. The first window defines the start of measurement. Every subsequent `known_at` must equal the previous `known_at + duration`. Durations must be positive and interval ends must not overflow. A trajectory cannot change `valid_at` halfway through. Different queries are scored independently and their exposures may be summed only with the query weighting/count disclosed.

Each candidate prediction has `(query_id, checkpoint_index, answer)`. Candidate answer states are:

- `Value(x)`: explicit assertion of a value;
- `NoFact`: explicit assertion that no fact applies;
- `Abstain`: explicit refusal to provide a substantive answer;
- absent prediction record: missing output.

The prediction at a checkpoint is treated as constant over that exposure window. The evaluator must create windows from source timestamps or explicit event intervals. Do not convert OAKS chunk indices into seconds; when real durations are absent, report the result in chunk-time units.

## 4. Exposure definitions

Let (y^*(v,t)) be the evaluator-resolved answer and (\hat{y}_i) the candidate state for interval (i) of duration (\Delta t_i).

[
TED_{wrong} = \sum_i \Delta t_i\;\mathbf{1}[\hat{y}_i \not\equiv y^*(v,t_i)]
]

where the equivalence relation is:
- `Value(x)` is correct iff (y^*=x);
- `NoFact` is correct iff (y^*=\varnothing);
- `Abstain` is never counted as a correct substantive answer;
- missing is represented by no prediction record.

Three top-level components must remain separate:

1. **Wrong-belief exposure:** duration of explicit but incorrect Value/NoFact states.
2. **Abstention exposure:** duration of explicit Abstain states.
3. **Missing-output exposure:** duration for which no prediction was received.

The implementation also reports correct exposure and the following wrong-belief subsets:
- future-knowledge leakage exposure (including a future correction answer or premature NoFact from a not-yet-known retraction);
- missed-correction exposure (old value or NoFact after the correction is known but before the retraction is known);
- stale-after-retraction exposure (explicit value after the retraction is both valid and known).

Correction latency is measured from correction arrival to the first checkpoint with the corrected value for a valid-time coordinate to which that correction applies. Retraction latency is measured from retraction arrival to the first checkpoint with NoFact for a valid-time coordinate to which the retraction applies.

Do not collapse any of these components into a single weighted score.

## 5. Worked synthetic example

With four distractors and interval step 16, the T10 generator yields:
- correction observed at knowledge time 80;
- correction valid from time 32;
- retraction observed at knowledge time 96;
- retraction valid from time 48.

For a query at valid time 49, use contiguous windows ([0,80), [80,88), [88,96), [96,106)).

An event-aligned candidate emits:
- initial value during ([0,80));
- corrected value during ([80,88)) and ([88,96));
- NoFact during ([96,106)).

Its wrong-belief, abstention, missing, future-leak, missed-correction and stale-after-retraction exposure are all zero. Correction and retraction latency are both zero time units.

A delayed candidate emits the initial value during ([80,88)), then the corrected value during ([88,96)), and NoFact after retraction. It reaches the same final state, but accumulates eight units of wrong-belief and missed-correction exposure and has correction latency eight.

A model that emits the corrected value before knowledge time 80 accumulates future-knowledge leakage exposure. A model that emits NoFact after correction but before the retraction is known also incurs wrong-belief and premature-future-retraction exposure. Abstention and missing output contribute only to their own exposure components.

## 6. Implemented scorer API

The standalone Rust module exposes:
- `KnowledgeExposureWindow`;
- `KnowledgeTimedPrediction`;
- `TemporalErrorDebtScore`;
- `score_temporal_error_debt`;
- `resolve_expected_value`.

The scorer rejects empty schedules, zero-duration windows, time overflow, non-contiguous intervals, changing valid_at coordinates, duplicate predictions and unknown prediction keys. Missing predictions are inferred from the expected window schedule and counted in both the missing count and missing exposure. Full-stream coverage uses all scheduled windows as the denominator.

Twelve unit tests are authored for event-aligned correctness, delayed correction, premature NoFact, stale state after retraction, abstention vs missing, future correction leakage, future retraction leakage, contiguous windows, fixed valid time, zero durations, duplicate/unknown predictions, and equal final answer with different time-integrated debt.

## 7. What remains

- Generate repeated query checkpoint events directly from T10's task definition.
- Bind query windows and candidate output deadlines to an exact source timestamp/event schedule.
- Integrate the scorer into the isolated evaluator process specified by #9437.
- Keep oracle seeds/configuration protected and bind exposure outputs into evaluator receipts.
- Run all authored Rust tests on a pinned Rust toolchain, then compare HDC-LTC and matched baselines.
- Add calibrated hardware energy accounting per stream and per unit of wrong-belief exposure avoided.

## 8. Claim boundary

TED measures time-integrated answer-state correctness for the declared query trajectory, update semantics and duration units. It does not measure real-world harm without independently validated consequence weights. It is not an official OAKS score and does not establish general continual-learning superiority.
