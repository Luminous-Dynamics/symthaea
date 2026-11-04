# MEL-ABSTAIN-001 — Selective musical participation v1

**Status:** Rust evaluator, four paired symbolic families, three frozen baseline policies, strict set/report/baseline schemas, runnable example and nine authored unit tests are committed. Rust compilation, tests and audio/performer qualification are not yet executed.

## Research question

Can Melothaea distinguish when a musical contribution is authorized from when it should preserve space, ask for clarification or abstain because the evidence is insufficient?

A competent musical collaborator must not simply maximize action accuracy. It must avoid false entries without learning to remain silent on every task.

## Decision semantics

V1 deliberately distinguishes:

- **WAIT:** intentional musical silence is the appropriate action.
- **ABSTAIN:** the system lacks sufficient evidence to justify a contribution.
- **ASK:** a bounded clarification request can resolve uncertainty before an effect boundary.
- **RESPOND / CONTINUE / YIELD:** an active musical intervention selected from the authorized valid-action set.

These dispositions are not interchangeable. WAIT is a committed policy decision and remains in risk/coverage scoring. ASK and ABSTAIN are explicit non-commitment dispositions and are excluded from committed-decision coverage. All are reported separately.

The v1 paired fixture uses an `act` versus `withhold` requirement. It does not claim that every musical YIELD is an audible entry; in this first contract, YIELD is treated as an active, authorized interaction intervention. Future versions should split floor relinquishment from audible sound contribution if musician annotation shows those need separate action-cost categories.

## Paired counterfactual design

Each family has exactly two variants. The local-cue signature remains fixed while one declared contextual factor changes:
- collaborator intent;
- performer role;
- stream integrity/availability;
- collaborator veto;
- preceding context (supported by the schema and validator for future fixtures).

The evaluator checks that the selected factor changes, all other declared factors remain fixed, and the `preserved_factors` declaration exactly names the invariant fields. It preserves set-valued musician annotations and rejects decisions outside the member's valid-action set.

The current checked-in symbolic corpus contains four families / eight members:
1. A two-second cadence pause that invites a response versus retains the floor.
2. A lead role versus accompaniment role on the same rest.
3. A valid-stream invitation versus an unavailable/dropout stream.
4. An authorized entry versus an explicit collaborator veto.

The fixture labels are controlled symbolic signatures. They do not prove that two rendered waveform crops are identical. Waveform-grounded stimuli belong to #10393 and must be bound to exact audio digests, renderer identity, sample-frame intervals and permissions.

## Metrics

The Rust scorer reports, independently:
- member-level valid-decision support;
- paired act/withhold accuracy (both variants must be supported);
- false-entry rate on withhold-required members (RESPOND/CONTINUE only);
- unauthorized YIELD count on withhold-required members;
- total withhold-policy violation count/rate, including false entries and unauthorized YIELD;
- missed-invitation rate on act-required members;
- WAIT, ABSTAIN and ASK counts;
- Brier score for the declared probability of an act being appropriate;
- selective risk–coverage points using decision confidence.

ASK/ABSTAIN are not silently counted as correct WAIT. The evaluator also prevents a vetoed or unavailable-stream member from being annotated as `act`.

The act-probability Brier score uses a binary act/withhold requirement in this synthetic v1. It is not a probability that a particular action label is correct. Decision-confidence is a separate declared probability that the selected action is in the musician-annotated valid-decision set.

The risk–coverage curve excludes ASK/ABSTAIN from committed coverage, but keeps WAIT because intentional silence is a committed policy decision. Do not infer calibration from these synthetic fixture results.

## Frozen baselines

The report keeps false audible entries (RESPOND/CONTINUE), unauthorized YIELD, and total withhold-policy violations separate. The checked-in baseline file freezes all three metrics.

The checked-in baseline file contains:
- ALWAYS-WAIT: 4/8 member support, 0/4 fully correct pairs, four missed invitations, zero false entries, zero unauthorized YIELD and zero withhold violations.
- ALWAYS-RESPOND: 4/8 member support, 0/4 fully correct pairs, zero missed invitations, four false entries, zero unauthorized YIELD and four total withhold violations.
- CONTEXT-RULE-FIXTURE-SANITY: 8/8 member support, 4/4 fully correct pairs, zero missed invitations, false entries, unauthorized YIELD and withhold violations; the decisions include two WAITs, one ABSTAIN and one ASK.

These figures are expected results of the controlled fixture evaluator, not model performance or human-level capability. The context-rule baseline is hand-written against the fixtures and is not a learned policy.

## External research motivation

AgentAbstain (NeurIPS 2026 Oral) uses 263 paired near-identical agent tasks across 42 executable environments and eight abstention scenarios. It is designed so always-act and always-refuse policies cannot score well and distinguishes successful abstention from silent inaction or post-hoc questioning. The best reported paired accuracy across 17 frontier models is 59.5%. This is a general tool-agent result, not a music benchmark score. https://agentabstain.github.io/

The music-specific research question is different: can an agent distinguish an intentional musical pause from insufficient evidence, honor a veto, and avoid an entry that would disrupt a performer? The paired evaluation principle is reusable; the task semantics and musician-defined cost matrix are not transferable without study.

## What is not yet measured

This v1 does not include:
- raw-waveform invitation detection;
- musician-calibrated cost weights;
- human annotation disagreement or inter-annotator agreement;
- held-out performer/ensemble/renderer/cultural cohorts;
- conformal prediction or distribution-free guarantees;
- resolution utility or interruption cost for ASK;
- device output and action-effect evidence;
- musician preference, authorship or agency outcomes.

A pilot must establish task-specific false-entry and missed-invitation costs, clarification burden, sample size and meaningful non-inferiority margins before confirmatory targets are frozen. Selective prediction or conformal coverage claims must state their assumptions; exchangeability-based marginal coverage is not automatically a per-performer or out-of-distribution guarantee.

## Reproduction

Run:

`cargo run -p symthaea-muse --example mel_selective_participation`

The example consumes the same frozen prediction and expected-metric artifact used by the Rust unit test. The negative controls must not be presented as learned-model results.

## Claim boundary

This benchmark measures selective participation on explicit, controlled musician-annotation contracts. It does not establish subjective understanding, consciousness, universal musical competence or artistic quality. A high abstention rate alone is not success. Improvement requires a better paired act/withhold trade-off under musician-defined costs and held-out real-performance conditions.
