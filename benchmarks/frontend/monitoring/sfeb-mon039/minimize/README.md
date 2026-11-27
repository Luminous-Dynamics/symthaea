# MON-039 — Counterfactual Fault-Set Reduction

MON-039 reduces a deterministic injected fault set while preserving the exact failure signature emitted by an independent recovery verifier. It is a replay-table reducer, not a causal oracle or a chaos injector.

## Offline example

    cd benchmarks/frontend/monitoring/sfeb-mon039/minimize
    go run ./cmd/minimize --input INPUT-example.json
    bash CHECKS.sh

The public fixture starts with three hypotheses. The replay table contains all eight subsets. The same blocked signature is preserved by fault-a plus fault-c, while fault-b is incidental. Since all smaller subsets have observations, the fixture can report a global candidate minimum within the submitted three-fault set.

## Two minimality levels

- For up to 12 candidate faults, the minimizer enumerates subsets by increasing cardinality and requires complete replay evidence for all smaller candidate sets before claiming global_candidate_minimum.
- Above 12 faults, it uses deterministic single-fault deletion. It may claim one_minimal only after every single deletion from the final set has been evaluated and none preserves the same verdict and signature. It never calls this globally minimal.

The algorithm makes no monotonicity assumption. It compares the exact pair (verdict, failure_signature_sha256), not just "blocked". A different blocked reason is a different outcome.

## Evidence binding

Every observation is pinned to the same scenario digest, recovery-boundary digest, exact 40/64-character code revision, time window and canonical fault-set digest. Duplicate, stale, future-dated or malformed observations block. A missing or incomplete subset replay makes minimality not_established; it is never treated as a passing control.

The outcome receipt digest is byte binding only. MON-026 or another independent custody system must authenticate the receipt and its provenance before runtime conclusions are admissible.

## Symthaea's role

Symthaea may propose fault hypotheses, interactions and a test order. It cannot decide whether a minimized set reproduces the failure. The independent MON-037/MON-038 verifier outcome and its exact signature are the comparison oracle. That separation allows the model to be useful without grading its own explanation.

Both offline_fixture and runtime_candidate outputs remain campaign-indeterminate, and production_action_permitted is always false. The module does not inject faults, mutate infrastructure or promote traffic.
