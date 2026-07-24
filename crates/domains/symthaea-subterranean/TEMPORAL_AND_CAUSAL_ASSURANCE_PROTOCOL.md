# Temporal and Causal Assurance Protocol

**Campaign:** XVIII  
**Date:** 2026-07-20  
**Scope:** `symthaea-subterranean`

## Purpose

Subterranean autonomy cannot safely treat every received fact as current, every timestamp as trustworthy, or every observed response as proof that the immediately preceding command caused it. This protocol defines bounded runtime rules for clock discipline, delayed observations, causal event ordering, command-response attribution, plan freshness, restart continuity, and release evidence.

The protocol is safety-monotonic. Temporal and causal assurance may reduce productive or movement authority. It cannot restore authority removed by physical hazards, protected return reserve, operator constraints, actuator isolation, lifecycle restrictions, stewardship, formal assurance, or final-command invariants.

## Authority order

The deployed control path uses temporal evidence before final actuation. Relevant ordering is:

1. Physical state and hazard assessment.
2. Return-path, mission, team, lifecycle, stewardship, and epistemic constraints.
3. Temporal and causal assessment.
4. Operator, recovery, power, maintenance, and actuator constraints.
5. Formal transition assurance.
6. Final-command invariant enforcement.
7. Plant actuation and evidence retention.

A later authority may remove more command authority. It may not recreate authority removed earlier.

## Clock discipline

Each registered clock sample carries:

- source identity;
- clock domain;
- boot epoch;
- monotonic sequence;
- event time;
- uncertainty bound;
- receipt step.

The clock supervisor rejects or restricts authority for sequence replay, epoch regression, time regression, excessive future time, stale evidence, and excessive uncertainty. A new boot epoch may restart a source sequence but cannot regress the accepted epoch.

Clock identity and source independence are deployment responsibilities. The crate validates declared identities and ordering; it does not prove that two hardware clocks are physically independent.

## Delayed observations

Every timed observation declares its purpose:

- immediate control;
- hazard confirmation;
- mapping;
- forensic history.

Age and uncertainty are evaluated against purpose-specific freshness limits. Historical observations may remain useful for maps, audit, and later analysis while being denied current actuator authority. A delayed measurement is never silently promoted to a current control fact.

## Causal event ledger

Causal events use stable source, boot-epoch, and sequence identities. Explicit dependencies must:

- already exist;
- precede or overlap the dependent event consistently with uncertainty intervals;
- use non-replayed source sequences;
- preserve bounded ledger continuity.

Independent late events are retained as late evidence rather than rewriting previously accepted causal order.

## Command-response attribution

Expected responses specify:

- originating command cause;
- earliest and latest plausible effect steps;
- observed channel;
- expected response sign;
- minimum effect magnitude;
- command identity.

Attribution is conservative:

- `Supported` means one admissible cause fits the observation.
- `Ambiguous` means multiple admissible causes overlap.
- `Contradicted` means the response materially opposes the registered expectation.
- `Unattributed` means no registered cause supports the observation.

The crate does not claim causal identification from correlation alone. Confounded or overlapping effects remove productive authority rather than being declared uniquely caused.

## Plan freshness

A plan is bound to:

- creation and expiry steps;
- state revision;
- hazard revision;
- topology revision;
- calibration revision;
- mission revision;
- whether it authorizes productive work.

Age expiry or any relevant revision mismatch invalidates the plan. An invalid plan cannot continue productive work simply because its original route or objective was once acceptable.

## Temporal authority states

The supervisor emits one of four monotonic authority states:

- **Nominal** — current evidence supports ordinary operation.
- **ProbeOnly** — uncertainty or ambiguity permits bounded information gathering but not ordinary productive work.
- **ReturnOnly** — stale control evidence or invalidated plans remove productive work while preserving protected withdrawal and emergency recovery.
- **HoldForReview** — clock replay, impossible causal ordering, or serious temporal contradiction removes movement until a clean service-location dwell completes.

Cooling, dewatering, sealing, relay deployment, and roof support may remain available where required by physical safety.

## Restart continuity

Operational checkpoint schema version 11 persists:

- temporal supervisor state;
- accepted clock and event ordering;
- plan and review latch state;
- the causal runtime step.

Persisting the runtime step is essential. Without it, a restored formal replay ledger could correctly interpret the first post-restart frame as a repeated sequence. Older checkpoints default the runtime step to zero and remain subject to normal validation and authority restriction.

## Evidence and explanations

Each final command records:

- temporal authority;
- clock faults and degraded clocks;
- stale or historical observations;
- causal contradictions and ambiguities;
- plan invalidation reasons;
- review-latch state;
- the post-temporal command transformation.

Counterfactual explanations may identify temporal or causal uncertainty as a blocker. Explanation is observational and cannot alter control authority.

## Release requirements

The canonical registry includes:

- `SUB-TMP-001` — clock and timestamp integrity;
- `SUB-TMP-002` — delayed-observation authority;
- `SUB-TMP-003` — causal-ordering consistency;
- `SUB-TMP-004` — plan-freshness invalidation;
- `SUB-TMP-005` — temporal checkpoint continuity.

Eight deterministic contracts exercise clock replay rejection, stale-observation restriction, impossible dependencies, plan revision invalidation, ambiguous attribution, same-frame command restriction, clean-dwell recovery, and supervisor validity.

A temporal assurance evidence bundle binds build identity, current supervisor state, recomputed validation results, and distinct authenticated Safety Reviewer and Temporal/Causality Reviewer attestations. The built-in deterministic digest supports reproducible tests only and is not cryptographic authentication.

## Explicit non-claims

This protocol does not claim:

- synchronized physical clocks without calibrated hardware;
- cryptographic authenticity of timestamps or event sources;
- complete causal discovery;
- proof that all confounders have been modeled;
- correctness of external latency bounds;
- real-time operating-system guarantees;
- qualification across clock rollover, leap handling, power loss, or transport partitions without physical testing.

Production release requires calibrated clocks, authenticated transport metadata, hardware-in-the-loop delay and reordering campaigns, power-loss testing, controlled rollover tests, real workspace validation, and independent review.
