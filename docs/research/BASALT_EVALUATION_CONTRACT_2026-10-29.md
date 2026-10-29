# BASALT Evaluation Contract
## Version 1 — task-level evidence, pairing, and decision rules

**Status:** Proposed protocol; initial offline Rust validator source is staged, but compilation and tests have not yet been executed  
**Date:** 2026-10-29  
**Scope:** Offline evaluation record and analysis rules for repository-level coding agents

This contract turns the BASALT roadmap into a concrete interchange format. It does **not** claim that a BASALT experiment has run, that this schema is already enforced in CI, or that Symthaea outperforms another system.

## Files

- [Run-record JSON Schema](../../benchmarks/basalt/schema/run-record.schema.json)
- [Frozen suite-manifest JSON Schema](../../benchmarks/basalt/schema/suite-manifest.schema.json)
- [Synthetic run record](../../benchmarks/basalt/examples/run-record.example.json)
- [Synthetic frozen suite manifest](../../benchmarks/basalt/examples/suite-manifest.example.json)
- Rust validation command in `xtask`: `cargo run -p xtask -- basalt-validate --input PATH --manifest PATH`
- Validator implementation and adversarial tests: [`xtask/src/basalt_validate.rs`](../../xtask/src/basalt_validate.rs)

The example record and suite manifest are deliberately synthetic. Their revisions, digests, task and outcomes are fabricated fixtures and must never be used as benchmark results. The validator checks exact manifest bytes against `source.manifest_digest` and compares every task specification and quality-audit disposition against the frozen manifest.

## Primary estimand

The primary unit is a **task × system × predeclared attempt/seed**, not a log line, patch, test case, or aggregate summary.

For an eligible task trial, strict verified success is true only when all are true:

1. The task-quality audit status is `accepted`, the oracle is independent of the generator, and material ambiguity was resolved before execution.
2. The recorded outcome is `pass`.
3. The independently versioned verifier reports `pass`, ran isolated from the agent, and evaluated the recorded final-tree digest.
4. Every blocking verifier check passes, including build/type-check/test requirements and change-scope checks required by the task.
5. Integrity is `clean`: no detected solution-material access, test/verifier tampering, forbidden scope change, or stale evidence.

A model-authored test passing is not a substitute for an independent oracle. A candidate's self-reported confidence or `FiberPoint.quality` is not verifier evidence.

## Outcome and denominator rules

Never silently drop difficult rows:

- **Accepted task + pass:** strict success only when every primary condition above is satisfied.
- **Accepted task + fail:** failure.
- **Unresolved, infrastructure error, or not run:** retain as a separate outcome and report its rate. Do not silently convert it to a success or delete it; report both the prespecified strict intent-to-evaluate view and the conditional valid-trial view.
- **Quarantined or invalid task:** exclude from the eligible-task accuracy denominator only under a predeclared, auditable task-quality rule. Keep it in the released record and report its count and reason. Reclassification after seeing system outcomes requires independent review and a sensitivity analysis.
- **Integrity violation:** the attempt cannot count as a success even if ordinary task tests pass. Publish the violation rate separately.

The task-quality auditor must not use which system won as a reason to retain or reject the task. Keep task specification, hidden oracle, verifier bundle, repository state and scoring rubric frozen and content-addressed before the run.

## Pairing and record invariants

The JSON Schemas validate shape. The initial Rust validator in `xtask/src/basalt_validate.rs` adds cross-record invariants and requires the separately supplied frozen suite manifest. Run it as `cargo run -p xtask -- basalt-validate --input RUN.json --manifest SUITE-MANIFEST.json`. It hashes the exact manifest bytes, verifies the pinned digest, checks suite/task identity and quality-audit dispositions, and validates paired trial coverage. Ordinary JSON Schema cannot establish all of these properties:

1. `task_id` is unique in `task_records`; `system_id` is unique in `systems`; and every trial references a declared system.
2. For each eligible `task_id × attempt_index`, every compared system has exactly one row. A declared `paired_group_id`, seed, task spec, base-tree digest, environment, verifier version, and budget policy must match across the comparison arms.
3. Duplicate cells, omitted cells, inconsistent pair keys, or changed task manifests invalidate the comparison. Do not repair them by silently taking the first or best row.
4. The verifier's `bundle_digest` matches the run's pinned verifier bundle, and `evaluated_tree_digest` equals the trial's `final_tree_digest`.
5. A `pass` outcome requires verifier `pass`, all blocking checks `pass`, and integrity `clean`. Contradictions are invalid records and must fail closed.
6. The same immutable environment, isolation policy, task budgets and retry policy apply across paired systems, or each difference must be explicitly declared as a separate experimental factor.
7. Token or cost fields unavailable from a provider remain null/omitted where permitted; do not encode unavailable telemetry as zero.

The Rust module now contains 26 unit-test cases, including paired-arm omissions, duplicate cells, manifest-bound reclassification, digest mismatches, budget violations, unknown fields and verifier/integrity contradictions. They are authored but **not executed** in this environment because no Rust toolchain or mounted checkout was available and the draft PR workflows were skipped. Do not treat these tests as passing evidence.

The manifest digest establishes byte identity, not authenticity. An actor who can rewrite both the manifest and run record can recompute the digest. Signed, independently pinned pre-run and post-run receipts are tracked in [#9034](https://github.com/Luminous-Dynamics/symthaea/issues/9034); confirmatory scoring must remain disabled until that trust boundary is qualified.

## Analysis plan

Pre-register the primary endpoint, comparison arms, task families, exclusions, attempt policy, cost accounting, and stopping rule before opening held-out outcomes.

Report at minimum:

- Strict verified success rate, with the raw numerator and denominator.
- Paired candidate-minus-baseline difference with a 95% interval clustered/resampled at the task level; retain family strata and seed pairing. Avoid treating multiple attempts on one task as independent tasks.
- Success by task family and language; original-versus-transformed results for metamorphic tasks.
- Task-quality quarantine/invalid rate, unresolved rate, infrastructure error rate, and integrity-violation rate.
- Cost per strict verified success, wall-clock p50/p90, tool calls and token usage where measured.
- Scope violations, regressions, verifier contradictions, stale-index/evidence rejections, and reward-hacking incidents.
- Full system and runner settings, exact revisions, dataset split, digests and per-task result records.

For small pilot cohorts, label results exploratory. Thirty base tasks can be a useful coverage floor, but it is **not** a universal statistical power guarantee. Do not claim superiority from a point estimate or from a single lucky seed. Predeclare the minimum practically meaningful improvement and non-inferiority margins before confirmatory evaluation; adjust for multiplicity when making many family-level claims.

## Proposed rollout gates

These are research gates to test—not reported achievements:

**Gate 0 — record validity:** schema parses; the Rust validator builds and all cross-record invariants pass; the exact suite manifest is pinned; all rows have task/environment/verifier identities; negative controls fail closed. Source currently staged is not a passed gate.

**Gate 1 — verifier/task validity:** independent audit accepts the scored tasks and oracles; hidden reference material is inaccessible during agent execution; verifier and task mutation controls are detected.

**Gate 2 — baseline reproducibility:** pinned simple baseline reruns with exact task/seed pairing, and variance is reported. Re-run failures are retained, not cherry-picked.

**Gate 3 — ablation validity:** compare lexical/BM25, syntax/symbol/dependency graph, existing HDC/ProgramManifold, geodesic rejection, and any fusion under identical budgets. Ablations must remove one component at a time and preserve all other settings.

**Gate 4 — promotion to hard rejection:** only consider enabling geodesic hard rejection after a predeclared paired analysis shows the required quality/non-inferiority bound, no unacceptable family-level regression, no false rejection on the qualified control set, and independent replication. Until then, keep it in shadow mode.

**Gate 5 — external comparison:** use a pinned public benchmark version and matching agent/harness/environment/budget. Compare BASALT-specific robustness and assurance dimensions separately. Never present a BASALT percentage as directly comparable to another benchmark's percentage.

## Research position relative to current leaderboards

Public leaderboards measure different mixtures of task resolution, harness, tools, budget and scoring. As of 2026-10-29, published Terminal-Bench 4.0 results are around 64–65% for leading configurations, while the Artificial Analysis Coding Agent Index v2.0 combines 290 tasks across FrontierCode, Terminal-Bench and repository Q&A and reports cost, tokens, time, and reward-hacking checks. These are version-specific references, not a universal SOTA number.

BASALT's differentiated aim should be **verified change per unit cost under controlled shift**, with explicit evidence integrity—not a larger composite score obtained by blending incomparable metrics. Compare with public scores only after pinning the benchmark version and matching the executable setup.

## References

- [Artificial Analysis Coding Agent Index v2.0 methodology](https://artificialanalysis.ai/methodology/coding-agents-benchmarking)
- [Terminal-Bench 4.0 leaderboard](https://www.tbench.ai/?version=4.0)
- [Vals AI Terminal-Bench 4.0 leaderboard and methodology](https://www.vals-ai.com/benchmarks/terminal-bench-4)
- [OpenAI: Separating signal from noise in coding evaluations (July 2026)](https://openai.com/index/separating-signal-from-noise-coding-evaluations/)
- [Epoch AI: Terminal-Bench 4.0 review (September 2026)](https://epoch.ai/benchmarks/terminal-bench-4/review)
- [in-toto Attestation Framework v1.2](https://github.com/in-toto/attestation/blob/main/spec/README.md)
- [SLSA v1.2 specification](https://slsa.dev/spec/v1.2/)
- [GitHub artifact attestations](https://docs.github.com/en/actions/concepts/security/artifact-attestations)
