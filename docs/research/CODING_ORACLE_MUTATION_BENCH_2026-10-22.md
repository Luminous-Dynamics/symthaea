# Coding Oracle Mutation Benchmark (2026-10-22)

## Status and claim boundary

This is a **development qualification tool**, not a coding-agent leaderboard and not evidence of improved generation ability. Static source review only has been performed in the current authoring environment: Rust/Cargo are unavailable here, so compilation, formatting, runtime execution, schema validation by an external validator, and mutation outcomes have not yet been observed. The benchmark must be run on the exact branch/head before anyone claims it qualifies.

## Paradigm shift: evaluate the evaluator

A test suite's existence does not prove that it checks the right behavior. The harness uses a two-sided control:

1. A hand-authored reference implementation must compile and pass the declared oracle.
2. A behaviorally incorrect but type-correct mutant must compile and fail the exact same oracle.
3. Simulated execution, compile failure in either control, zero executed tests, incomplete test counts, or a passing mutant are qualification failures—not mutation kills.

This design borrows the useful idea of a two-sided reference/empty-patch gate described in [SWE-Bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2), while keeping this small Rust suite self-contained.

## Current fixture tranche

The executable includes 16 deterministic reference/mutant pairs (native Rust functions/data structures), with 47 assertion tests in the shared fixtures.

| Task family | Defect the oracle must detect |
|---|---|
| Signed arithmetic | subtraction substituted for addition |
| Absolute value | negative numbers returned unchanged |
| Greatest common divisor | minimum substituted for Euclidean GCD |
| Clamping | upper-bound check omitted |
| Parity | negative odd integers mishandled |
| Text statistics | uppercase vowels omitted |
| Reversal | input returned without reversing |
| Palindromes | all strings longer than one character rejected |
| Collection normalization | duplicates sorted but not removed |
| Search/indexing | first zero found instead of first negative |
| Selection | maximum returned instead of second-largest distinct value |
| Numeric edge cases | zero treated as having no digits |
| Ordering | strict increase substituted for nondecreasing order |
| Binary-search boundary | lower bound substituted for upper bound |
| Sorted merge | second input slice ignored |
| Overflow | unchecked factorial multiplication substituted for checked arithmetic |

The receipt includes task IDs, complete reference/mutant/test source, per-source BLAKE3 digests, oracle result counts, status classifications, aggregate fixture digest, elapsed times, Rust version where obtainable, target OS/architecture, and an optional exact source revision.

## Run and verify

From the Symthaea repository root:

```bash
GIT_COMMIT_SHA="$(git rev-parse HEAD)" cargo run --release \
  --example coding_oracle_mutation_bench --features code_generation \
  -- --json-out /tmp/symthaea-coding-oracle-mutation-v1.json

cargo run --example coding_oracle_mutation_verify --features code_generation -- \
  /tmp/symthaea-coding-oracle-mutation-v1.json
```

The mutation runner exits nonzero if any reference fails to pass all its tests, or if any mutant is not a complete, compiled, behavioral test failure. It will not count a mutant compile failure as a successful behavioral kill.

The separate receipt verifier checks:
- each task's declared test count against the self-contained test fixture;
- every source/fixture digest and the aggregate fixture digest;
- the status implied by compilation, simulation flag, and test counters;
- whether reference-pass and mutation-kill flags agree with per-variant evidence;
- summary totals, kill rate, and qualification flag.

The JSON receipt schema is in [`schemas/symthaea-coding-oracle-mutation-v1.schema.json`](schemas/symthaea-coding-oracle-mutation-v1.schema.json).

**Verifier limitation:** it checks internal consistency and recomputes digests; it does not rerun `rustc`, authenticate who created the receipt, or make an unsigned receipt tamper-proof. Re-run the mutation harness in a trusted environment to reproduce the outcomes.

## Validation requirements

- Run formatting on the two new Rust examples, then compile/test the benchmark and verifier using the repository's pinned Nix/Rust environment.
- Require 16/16 reference controls to pass and 16/16 mutants to be killed by the behavioral oracle. The current expected outcome is a target, not a measured result.
- Deliberately corrupt a copied receipt (change a source string, a test count, a status, a task ID, or the aggregate digest) and confirm the receipt verifier rejects it.
- Record exact-head CI results separately from local runs; queued or absent checks are not passes.
- Extend mutation families over time: boundary, error handling, input permutation, idempotence, overflow, state invariants, and test runner failure. Review surviving mutants rather than quietly changing the oracle to kill them.

## Known limitations and next tranche

- This is a compact function-level control suite; it is not a substitute for held-out multi-crate Rust issue-fixing tasks or end-to-end repository agents.
- Handwritten mutants validate only the behaviors they challenge. They do not establish universal oracle completeness.
- Most controls are deterministic examples, not property-based proofs. The next increment should add property/metamorphic tests and mutation operators for real production fixes, preserving independent expected behavior.
- Receipts are reproducible manifests, not cryptographic attestations of execution.
- Public task sets have demonstrated contamination and test-validity hazards. OpenAI's 2026-07-08 audit of the 731-task SWE-Bench Pro split flagged 27.4% with an automated pipeline and found 34.1% problematic through a five-engineer annotation campaign; that finding does not prove the same rates for later Pro V2, but it does justify auditing the exact task/version. See [OpenAI's audit](https://openai.com/index/separating-signal-from-noise-coding-evaluations/).
- [Terminal-Bench](https://github.com/harbor-framework/terminal-bench) recommends repeatedly running oracle solutions in the target sandbox; Symthaea should apply the same discipline before using a task family for capability claims.

The right near-term success criterion is not “higher coding score”; it is: **the measurement correctly distinguishes known-good behavior from deliberately wrong behavior, produces a verifiable receipt, and fails closed when the experiment did not actually run.** Only after that should we compare Symthaea's candidate coding policies on held-out repository tasks.
