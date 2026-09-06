# Symthaea RCA Shadow Observer

RCA-002 establishes a causally inert observation boundary before any epistemic shadow policy is allowed to evaluate live cognitive output.

```text
completed CycleResult
        |
        | future adapter: values only
        v
FrozenCycleObservationV1
        |
        v
symthaea-rca-shadow::observe
        |
        v
ShadowObservationReceiptV1
        X
        | no control return path
        v
live cognition
```

## Invariants

- This crate does not depend on the root `symthaea` crate.
- It does not depend on GWT, MetaRouter, memory, learning, runtime, networking, tools, action, or recursive-improvement crates.
- It receives owned detached values, not `CycleResult`, `CognitiveLoopService`, callbacks, channels, locks, or mutable handles.
- Large cognitive artifacts cross the boundary only as cryptographic commitments.
- Observation validation fails closed on malformed commitments, out-of-range fixed-point metrics, and incomplete language provenance.
- `observe()` is deterministic and side-effect free.
- Its receipt is observational bookkeeping only. It is not admitted evidence, canonical epistemic state, action authority, or self-improvement promotion authority.

## Deliberate non-scope

RCA-002 does not yet adapt the production cognitive loop, evaluate evidence lineage/currentness, admit epistemic claims, modify GWT or MetaRouter behavior, or influence the next cognitive cycle. RCA-003 may consume these frozen observations in a shadow epistemic evaluator only after this one-way boundary qualifies.
