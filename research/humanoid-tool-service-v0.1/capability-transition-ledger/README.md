# Capability-transition ledger

**Status:** dependency-free Rust evaluator; synthetic tests authored, not yet executed here.  
**Parent:** HUM-CAPABILITY-TRANSITION-001 #12741.  
**Benchmark anchor:** HumanoidToolBench's five recorded events.

## Why this exists

A terminal success bit cannot locate the capability bottleneck. A policy may touch something, select and lift the correct tool, reach the target, and still fail the physical task. A different policy may fail at the first contact. Those are different research and engineering problems.

The event set is:
1. candidate contact;
2. correct first contact;
3. correct-tool lift;
4. tool-target contact;
5. verified task success.

## Evidence semantics

Each event is one of:
- NotInstrumented: no channel was available; it is excluded from the denominator.
- InstrumentedAbsent: the channel was active and the event did not occur.
- OccurredAtNs(timestamp): the event occurred at a timestamp in the episode's declared clock domain.

When task-success evidence is instrumented, the episode must also carry an independent_verifier_id distinct from both policy_id and source_id. This establishes a separately identified verifier source, not proof of organizational independence; the evaluation protocol must still document how the verifier is isolated from the policy and event recorder.

This avoids turning missing telemetry into a negative observation.

## Statistics

The evaluator keeps policy, split, scenario, execution level and tool-set mode separate. For each event it reports occurrence / instrumented episodes. For each adjacent pair it reports:
- episodes where both channels were instrumented;
- source-event occurrences in that jointly instrumented subset;
- co-occurrence irrespective of temporal order;
- temporally ordered co-occurrence (target timestamp >= source timestamp);
- mean latency for ordered co-occurrences;
- co-occurrence and ordered conversion rates conditional on source-event occurrence.

Events are not forced into a funnel. Regrasping and recovery can create out-of-order timestamps; those events remain in the data and reduce the ordered-transition metric without being discarded.

## Tests

Nine tests cover missing instrumentation vs event absence, out-of-order events, conditional co-occurrence vs temporal order, task-cell isolation, duplicate episode IDs, no-instrumentation episodes, missing general provenance and task-success verifier identity. The library uses only the Rust standard library.

## Integration boundary

This crate currently exposes typed Rust structs and summary functions; it does not yet deserialize the JSON receipt schema or read live HumanoidToolBench logs. A caller must parse a receipt, preserve its source clock and verifier identity, then construct an EpisodeEventLedger. The JSON Schema and six fixtures define the wire format for that future ingest adapter.
