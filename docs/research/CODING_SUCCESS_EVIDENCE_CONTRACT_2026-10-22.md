# Coding Success Requires Behavioral Evidence (2026-10-22)

## Problem

Two result paths conflated compilation with behavioral correctness. `ExecutionResult::is_success()` returned true when the program compiled and `tests_failed == 0`, even if `tests_passed == 0`. The school/learning path had the same issue in `LessonOutcome::is_success()`. Worse, a compiled lesson with an absent, skipped, or incomplete oracle could receive `actual_quality = 1.0` and become eligible for Broca SSM distillation.

That means a zero-test result could be promoted as a successful example despite providing no behavioral evidence.

## Contract after this change

- `ExecutionResult::is_compilation_success()` means compilation succeeded in a real execution context. This is available for compile-only workflows and is explicitly not a behavioral correctness claim.
- `ExecutionResult::is_test_verified_success()` requires positive observed test evidence, no failed tests, real execution, and no runtime/test-runner error. Because the executor result itself does not know the expected oracle size, callers that do know it must also check completeness.
- `LessonOutcome` carries the syntax-level `tests_expected` count and `test_run_complete` flag. A lesson counts as passed only when a nonempty declared oracle completed and every expected test passed.
- An absent oracle may receive explicit partial compile-only credit (0.5) but cannot count as a passed lesson or qualify for distillation.
- A declared oracle that is missing tests, blocked, incomplete, or accompanied by a runtime error cannot receive a perfect quality score or enter distillation.
- `ExecutionResult::to_surprise()` no longer treats a zero-test run as perfectly unsurprising; zero observed tests and runtime/test-runner errors receive nonzero surprise.

## Scope and limits

The expected test count is computed by counting external Rust `#[test]` declarations in the lesson fixture. This is a simple source-level check, not a Rust parser. The fixture must be well-formed and test-runner errors remain fail-closed. Auto-generated remediation drills now wrap their assertion snippets in a named `#[test]` function so that those fixtures have an explicit test entry point. Higher-level benchmark receipts should continue to use their own explicit expected-vs-observed counts and two-sided reference/mutant controls.

## Regressions added

- Compile-only `ExecutionResult` is a compilation success, but not a behavioral success, and carries nonzero surprise.
- A runtime/test-runner error prevents `is_test_verified_success()` despite optimistic counters.
- A lesson with no tests is not a pass; it keeps only partial mastery credit.
- Partial/incomplete test execution cannot count as a pass, receives discounted mastery, and does not qualify for distillation.
- The expected-test helper is checked against a two-test fixture and missing/empty fixtures.

## Validation

The authoring environment has no `rustc` or `cargo`, so formatting, compilation, unit tests, and end-to-end execution have not been performed. Suggested checks from the repository's pinned Nix shell:

```bash
cargo test --features code_generation --lib language::code_executor
cargo test --features 'code_generation school_learning' --lib school::code_learning
cargo test --features 'code_generation school_learning' --test coding_honesty
```

Confirm the crate's exact feature aliases and canonical test commands before running. Tests and source changes must be reviewed on the exact PR head; queued or absent status checks are not passes.

## Related

- [Benchmark oracle integrity](https://github.com/Luminous-Dynamics/symthaea/pull/8310)
- [Mutation-based oracle controls](https://github.com/Luminous-Dynamics/symthaea/pull/8313)
- [GCS topology advisory-only contract](https://github.com/Luminous-Dynamics/symthaea/pull/8322)
