# PHOT-EVAL-005A: verified anytime-quality trace for maximum independent set

**Status:** protocol/schema draft; not an optimization result.  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8796  
**Companion protocol:** [PHOT-CTRL-003A held-out ablation protocol](PHOT-CTRL-003A-HELDOUT-ABLATION-PROTOCOL.md)  
**Related evaluator drafts:** [#8774 byte-level verifier](https://github.com/Luminous-Dynamics/symthaea/pull/8774), [#8779 failure-inclusive time-to-target](https://github.com/Luminous-Dynamics/symthaea/pull/8779), [#8784 trial-bound receipts](https://github.com/Luminous-Dynamics/symthaea/pull/8784).

## Purpose and boundary

The primary confirmatory endpoint remains failure-inclusive restricted-mean deployment time to an independently verified target. This contract defines a separate secondary diagnostic: the quality of the best independently verified feasible solution available as elapsed deployment time passes.

This matters because an optimizer can emit a claimed incumbent before the run finishes, and QOBLIB accepts optional objective-time-series data for convergence analysis. A reported objective sequence is not, by itself, proof that each checkpoint is feasible or that a candidate had passed the verifier by its recorded timestamp. This schema therefore records candidate identities, verifier dispositions, receipt artifacts, and a clock boundary. It does **not** authenticate the process that supplied a timestamp. Trusted-runner signatures or attested execution, if required by a threat model, are a separate layer.

The quality curve is specific to unweighted maximum independent set. Do not reuse it for weighted MIS or a different objective without a separately versioned metric definition.

## Trace contract

Each run has one globally unique `trial_id`; paired baseline/candidate runs share a `pair_id`, but never a `trial_id`. The root record freezes the protocol, canonical graph, verifier, solver source/configuration, environment, graph size, target, and horizon. Those identities are part of the experiment, not descriptive labels.

Each candidate event records:
- elapsed deployment time from request/job entry, on a monotonic clock, through the persistence boundary for the candidate and its verification evidence;
- the exact candidate bytes' SHA-256 and a relative artifact path;
- a verifier disposition and, optionally, a caller-reported cardinality for diagnostics only;
- for a claimed verified event, the receipt bytes' relative path and SHA-256.

The verified cardinality is recomputed from the candidate bytes by the independent verifier. It is not a trusted input field in the trace. The consumer must read the referenced bytes and recompute their SHA-256 values. It must independently verify graph feasibility and check that receipt context matches the graph, verifier, protocol, trial, pair, seed, candidate digest, and verdict. Strings shaped like hashes, schema validity, and caller-supplied status or cardinality fields are not evidence by themselves.

All candidate attempts remain in the trace, including verifier errors, infeasible outputs, missing receipts, context mismatches, and untrusted artifacts. A non-verified candidate may include a reported cardinality for diagnostics, but it has no verified cardinality and contributes nothing to the quality curve. A purported verified checkpoint whose receipt bytes are absent, malformed, digest-mismatched, context-mismatched, or inconsistent with an independent re-verification invalidates the trace; do not silently downgrade it and continue reporting a score.

## Curve definition

Let n = |V| be the frozen vertex count and H > 0 the frozen horizon in deployment milliseconds. Let C(t) be the largest cardinality among independently verified feasible candidates whose persisted verification event time is at or before t, or zero if none exists. Define q(t) = C(t) / n and the normalized area under the quality curve as:

A_Q(H) = (1 / H) ∫₀ᴴ q(t) dt

Treat q(t) as a right-continuous step function: a verified candidate starts contributing at its recorded verification/persistence time. Do not linearly interpolate between checkpoints, backdate a solution to solver discovery time, or infer an incumbent from an unverified objective log. The quality ratio is bounded in [0, 1] for an unweighted independent set.

For event times sorted as 0 < t₁ ≤ ... ≤ tₘ, include endpoints 0 and H, update the incumbent only from valid events at or before the current time, and integrate each incumbent value over the following interval. Multiple events at the same timestamp use the maximum verified cardinality at that time; their file order cannot change the score. Events after H remain in the artifact and in separate target/TTS reporting, but do not change A_Q(H).

The example trace has n = 10, H = 1000 ms, a purported cardinality-8 event at 20 ms without a verifier receipt, a verified cardinality-2 candidate at 100 ms, and a verified cardinality-4 candidate at 350 ms. The unverified claim is ignored. Thus the normalized quality is 0 for the first 100 ms, 0.2 for the next 250 ms, and 0.4 for the final 650 ms:

A_Q(1000) = (0 × 100 + 0.2 × 250 + 0.4 × 650) / 1000 = 0.31.

This fixture illustrates the semantics only; it is not a solver measurement.

## Reporting and aggregation

Report the quality curve and (A_Q(H)) alongside, not instead of:
- target-hit probability under a frozen common budget;
- failure-inclusive restricted-mean verified time-to-target and the chosen horizon;
- best final verified cardinality and feasibility/verifier-failure counts;
- standardized algorithm runtime and deployment all-in latency as separately defined;
- per-instance results and family-stratified effects.

Freeze (H), target, cost boundary, baseline, and graph/family weights before confirmation runs. Compute one trace score per run, aggregate repetitions within each graph using a predeclared rule, then aggregate graphs/families using declared weights. Do not let the largest graphs dominate solely because they have more vertices or more events; this curve normalizes by vertex count, and equal-graph or equal-family weighting must be explicit. Keep repeated seeds nested within their graph/family during uncertainty analysis. Do not promote (A_Q) to a universal scalar leaderboard or claim superiority from the secondary endpoint alone.

## Fail-closed invariants

1. Event timestamps are finite, non-negative, in nondecreasing serialized order, and share the exact root clock origin and deployment boundary. No stage-duration sum substitutes for a direct clock.
2. Event IDs are unique within a trace. The trial ID is globally unique across compared arms; pair ID is the pairing key.
3. Candidate and receipt artifact paths are relative to the declared evidence bundle. Reject absolute paths, traversal, symlinks escaping the bundle, missing files, duplicate event IDs, digest mismatches, and any single path that is assigned different bytes/digests within the same trace.
4. The independently checked candidate cardinality lies between zero and (n), and every selected vertex is unique/in-range with no selected edge conflict.
5. A verified feasible event must have a validated receipt and a validated candidate; the receipt verdict and cardinality must match recomputation. Caller-provided verified fields alone never raise (C(t)).
6. A non-verified event cannot carry a verified cardinality. Its reported cardinality is descriptive only.
7. An event after (H) is retained but does not affect the integral inside ([0,H]).
8. Missing outputs, crashes, timeouts, and aborts remain represented by terminal outcome even when the trace has no candidate events.
9. Graph/solver/protocol/verifier/environment identity must not change within one trial.
10. The trace and its artifact bundle are content-addressed and kept immutable. Hashes prove correspondence to the bytes under examination only; unsigned content hashes do not authenticate who ran the solver or when the event occurred.

## Acceptance tests for the follow-up evaluator

The next implementation should have deterministic tests for:
- no candidate events: (A_Q=0);
- one valid solution at (t): exact step-function integral before/after (t);
- several verified incumbents: only a strictly better verified cardinality changes the curve;
- unverified cardinality-8 claim followed by verified cardinality-2: the score credits only cardinality 2;
- missing receipt, receipt tampering, context mismatch, candidate tampering, infeasible candidate, and cardinality over (n): no score-bearing target claim;
- same-time events in every input ordering: identical (A_Q);
- an event exactly at (H) contributes no positive-duration area within that horizon;
- an event after (H) is retained but has no effect on (A_Q(H));
- duplicate event IDs, decreasing timestamps, changed graph/verifier/protocol/trial identity, path traversal, or absent artifacts: fail closed;
- an independent hand-calculated fixture agrees exactly with the evaluator;
- replay of a frozen trace yields bit-identical output.

No runtime, energy, speedup, SOTA, quantum advantage, or process-authenticity claim follows from a trace passing structural validation. Solver-performance claims require the frozen held-out experiment and exact-head validation separately.

## References

- QOBLIB contribution guide, including optional objective-time-series artifacts and separate runtime fields: https://github.com/ZIB-AOPT/QOBLIB/blob/main/CONTRIBUTING.md
- QOBLIB MIS benchmark: https://zib-aopt.github.io/QOBLIB/problem/07/
- Symthaea byte-level verifier draft: https://github.com/Luminous-Dynamics/symthaea/pull/8774
- Symthaea direct-clock restricted-mean TTS draft: https://github.com/Luminous-Dynamics/symthaea/pull/8779
