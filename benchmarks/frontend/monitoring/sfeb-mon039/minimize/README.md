# MON-039 — Counterfactual Fault-Set Reduction

MON-039 reduces a deterministic injected fault set while preserving the exact failure signature emitted by an independent recovery verifier. It is a replay-table reducer, not a causal oracle or a chaos injector.

## Offline example

    cd benchmarks/frontend/monitoring/sfeb-mon039/minimize
    go run ./cmd/minimize --input INPUT-example.json
    bash CHECKS.sh

The fixture binds an eight-row replay table to committed scenario and recovery-boundary manifest files. The initial set contains three hypotheses. The same blocked signature is preserved by fault-a plus fault-c, while fault-b is incidental. Because every candidate subset up to the winning cardinality is observed, the example can report a global candidate minimum within this three-fault universe.

## Two minimality levels

- For up to 12 candidate faults, the minimizer enumerates subsets by increasing cardinality and requires complete replay evidence for all smaller candidate sets before claiming global_candidate_minimum.
- Above 12 faults, it uses deterministic single-fault deletion. It may claim one_minimal only after every single deletion from the final set has been evaluated and none preserves the same verdict and signature. It never calls this globally minimal.

The algorithm makes no monotonicity assumption. It compares the exact pair (verdict, failure_signature_sha256), not just "blocked". A different blocked reason is a different outcome. Only a blocked full-set baseline is minimized; an indeterminate baseline is not treated as a reproduced fault.

## Evidence binding

Every observation is pinned to the same trial ID, scenario digest, recovery-boundary digest, exact 40/64-character lowercase code revision, time and canonical fault-set digest. The outcome receipt SHA-256 is recomputed over the complete observation record with its own receipt field blank. The fixture's scenario and recovery-boundary SHA-256 values are verified against committed manifest bytes.

Duplicate, stale, future-dated, incomplete or malformed observations block. A missing subset replay makes minimality not_established; it is never treated as a passing control.

The receipt digest is byte binding, not authenticity. MON-026 or another independent custody system must authenticate the receipt and its provenance before runtime conclusions are admissible.

## Symthaea's role

Symthaea may propose fault hypotheses, interactions and a test order. It cannot decide whether a minimized set reproduces the failure. The independent MON-037/MON-038 verifier outcome and its exact signature are the comparison oracle. That separation allows the model to be useful without grading its own explanation.

Both offline_fixture and runtime_candidate outputs remain campaign-indeterminate, and production_action_permitted is always false. The module does not inject faults, mutate infrastructure or promote traffic.

## Multi-order reduction tournament

The additive MON-039 tournament evaluates the same input observation table under up to eight unique deletion permutations. For fault sets above 12, it reports the smallest complete one-minimal set observed across tested orders, with deterministic tie-breaking. It does not claim a global minimum or unique root cause. Missing observations in any declared order keep the tournament incomplete/indeterminate. Run `go run ./cmd/tournament --input TOURNAMENT-INPUT-example.json` to exercise the wrapper CLI.
