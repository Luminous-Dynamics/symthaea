# VLEB two-sided oracle verifier self-mutation campaign — 2026-10-29

## Why test the evaluator itself?

A benchmark may pin its task, patch, runner and result artifacts while still having a verifier that silently stops checking one of those pins. A successful nominal test suite does not prove that the tests protect the checks that matter. The new tools/vleb/run_oracle_control_mutation_campaign.py therefore tests whether the two-sided oracle-control adversarial suite detects deliberate, reviewed bypasses in its own verifier.

This complements the replay gate documented in VLEB_TWO_SIDED_ORACLE_REPLAY_2026-10-29.md. It does not replace independent human oracle review, protected runner evidence, or the two-sided replay itself.

## Frozen mutation families

The first catalog contains 15 reviewed guard-bypass sentinels across seven families:

- trusted_input_pins: manifest, reference patch, test-bundle and verifier-source digest checks;
- source_isolation: clean source-worktree enforcement;
- execution_identity: command environment isolation and rejection of secret-bearing environment variable names;
- outcome_classification: requiring an actual recognized test-start marker before classifying a fail;
- two_sided_gate: requiring a passing reference arm and a failing empty baseline;
- post_run_integrity: test-bundle byte integrity, tracked reference-source integrity, and undeclared untracked-file inventory;
- archive_safety: symlinked directory traversal and test-bundle attempts to overwrite source files.

The complete mutation IDs, exact source anchors and one designated failure-sensitive unit test per mutation live in the runner. Each anchor must occur exactly once. If the code shape drifts, the mutation becomes invalid instead of being silently skipped.

## What qualifies as killed?

The campaign first runs the original test_verify_oracle_controls.py suite. If the unmodified suite does not pass with a demonstrated nonzero test count, the campaign stops with no mutation results.

For every sentinel it then:
1. produces the mutation in a temporary copy of the verifier;
2. compiles the modified Python module;
3. runs the full adversarial test suite in a temporary copy;
4. looks for the specifically designated test to fail.

A mutant is **killed** only if compilation succeeds, the test runner demonstrably starts, and the named test that protects the changed guard fails. A surviving mutation, syntax/import/startup failure, timeout, unmatched source anchor or unrelated test failure is not counted as killed. A campaign is complete only if all 15 sentinels are killed and every family has zero invalid, surviving or not-run cases.

The workflow writes a JSON campaign report as a retained artifact. Output always has analysis_only=true and score_eligible=false.

## Targets and interpretation

The initial target is 100% killed for the frozen sentinel catalog and 100% within each family. This is a target for test sensitivity against a finite set of deliberate implementation mutations—not a proof that the verifier is correct, not an exhaustive analysis of every possible bypass, and not a candidate-agent success score.

A passing campaign supports a narrow claim: these named tests detect these specific mutations at the exact tested verifier head. New independent mutation families should be added over time, especially around command execution lifecycle, artifact authenticity, and the interface with protected runner attestations.

## Research connection

SWE-Bench Pro V2's 2026 refresh emphasizes benchmark validity by combining a two-sided empty-baseline/reference-patch gate with sanitized source images and pristine-image regrading. AgentSuite's ICML 2026 COBA work treats benchmark flaws as interactions among user instructions, environment, ground truth and evaluation. The mutation campaign adds a further layer: testing whether VLEB's own evaluator guards are actually protected by its tests. See the SWE-Bench Pro V2 release (https://labs.scale.com/leaderboard/swe_bench_pro_public_v2) and AgentSuite / COBA (https://proceedings.mlr.press/v306/suh26a.html).

## Validation status

The campaign and its tests are draft code wired into the VLEB workflow. They have not yet produced a passing exact-head run in this environment; the metric stays null and ineligible until an exact-head campaign report is retained and reviewed.
