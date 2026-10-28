# Science Benchmark Refresh fixture

This directory is a protocol-first starting point for SCI-BENCH-REFRESH-001 (Issue #8771).

`fixtures/generation-manifest.example.json` is a **format fixture only**. Its revision marker says `NOT_IMPLEMENTED_YET`; SHA-256 fields contain all-zero placeholders; the evaluator is explicitly not qualified. These values must never be interpreted as actual commitments or used to score a policy.

The companion JSON Schema defines the v1 field shape and basic value constraints. It does not by itself enforce cross-field invariants such as:

- task-family weights summing to 1;
- sum of planned task counts equalling the expected denominator;
- canonicalization and domain separation used when computing commitments;
- correctness of the generator or evaluator;
- proof that oracle-only fields were inaccessible to the evaluated policy;
- task-level and family-level duplicate/leakage checks.

Those require a separate Rust validator and hostile fixtures. Do not treat schema validation as benchmark qualification.

Before any actual run, replace the placeholders with real immutable revisions and commitments; independently test the evaluator controls; and verify the runner can enforce the declared policy/oracle boundary. If secrecy cannot be maintained, change the lane to `reproducible_open` and the reveal policy to `open_from_start_not_sealed`.

The current deliverable contains a protocol, schema, and placeholder fixture only. It does not yet generate tasks, hold a secret seed, run NewtonBench, or establish any Symthaea performance result.