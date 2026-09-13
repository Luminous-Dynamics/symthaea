# symthaea-energy-benchmark-zero-exact

Exact-candidate-universe wrapper for Energy Discovery Benchmark Zero.

The generic Benchmark Zero evaluator is intentionally reusable: every ranked candidate must have truth, but the truth set may contain additional candidates. That generality is unsafe for leakage-controlled Matbench evaluation after exposed-training compositions have been removed from the admissible screening universe.

If excluded truth rows remain present, they can still affect:

- `qualifying_truth_count`;
- top-k recall denominator;
- `global_best_target_error_ev`;
- target regret.

This adapter therefore establishes a stronger measurement theorem before delegating to the existing metric implementation:

```text
ranked candidate ids == truth candidate ids
```

The equality is exact and set-based. Ranking order remains independently committed by Benchmark Zero's existing ordered ranking digest.

## Receipt

`ExactUniverseBenchmarkReceipt` records:

- fixed v1 schema/capability classification;
- canonical SHA-256 of the sorted candidate-id universe;
- exact candidate count;
- the underlying generic `BenchmarkReceipt`.

Structural validation requires both `ranked_candidate_count` and `truth_candidate_count` in the underlying measurement to equal the exact candidate count.

`verify_exact_universe_receipt` recomputes the candidate-set equality, digest and Benchmark Zero metrics and requires exact receipt equality.

## Authority boundary

This adapter does not establish that the candidate universe is globally leakage-free. It only establishes that the truth population used for measurement exactly matches the candidate population that the supplied screening run was allowed to rank.

For the Matbench path, the intended sequence is:

```text
official pinned artifact
    -> exposure/exclusion plan
    -> retained truth-free universe
    -> freeze rankings
    -> construct truth restricted to that exact retained universe
    -> exact-universe Benchmark Zero measurement
```

The wrapper performs measurement only. It does not train a model, construct a screening universe, rank candidates, authorize experiments, certify materials, or declare a policy superior.
