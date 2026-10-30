# BASALT Evaluation Contract
## Version 1 — task-level evidence, pairing, and decision rules

**Status:** Proposed protocol; the run-record schema remains version 1, while the frozen suite-manifest schema is version 3 and freezes task requirements, verifier coverage, and patch-quality rubrics. The Rust validator and its authored tests have not yet been compiled or executed  
**Date:** 2026-10-29  
**Scope:** Offline evaluation record and analysis rules for repository-level coding agents

This contract turns the BASALT roadmap into a concrete interchange format. It does **not** claim that a BASALT experiment has run, that this schema is already enforced in CI, or that Symthaea outperforms another system.

## Files

- [Run-record JSON Schema](../../benchmarks/basalt/schema/run-record.schema.json)
- [Frozen suite-manifest JSON Schema (version 3)](../../benchmarks/basalt/schema/suite-manifest.schema.json)
- [Synthetic run record](../../benchmarks/basalt/examples/run-record.example.json)
- [Synthetic frozen suite manifest](../../benchmarks/basalt/examples/suite-manifest.example.json)
- Rust validation command in `xtask`: `cargo run -p xtask -- basalt-validate --input PATH --manifest PATH --spec-root DIR`
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

The task-quality auditor must not use which system won as a reason to retain or reject the task. Keep task specification, hidden oracle, verifier bundle, repository state and scoring rubric frozen and content-addressed before the run. The frozen suite manifest also declares each task's required verifier obligations by stable check ID and kind; a trial cannot claim success merely because an arbitrary, easier check passed. Every required obligation must be represented in the verifier output as a blocking check, and every required check must pass for an outcome to count as a candidate success. The Rust deserializer also rejects explicit `null` for optional-but-non-null fields while preserving the record's required-but-nullable digest fields.

## Pairing and record invariants

The JSON Schemas validate shape. The Rust validator in `xtask/src/basalt_validate.rs` adds cross-record invariants and requires the separately supplied frozen suite manifest plus the exact offline task-specification artifacts. Run it as `cargo run -p xtask -- basalt-validate --input RUN.json --manifest SUITE-MANIFEST.json --spec-root TASK-SPEC-DIRECTORY`. Task specifications are loaded offline by their content digest and are never fetched over the network. It hashes the exact manifest bytes, verifies the pinned digest, checks suite/task identity and quality-audit dispositions, and validates paired trial coverage. Ordinary JSON Schema cannot establish all of these properties:

1. `task_id` is unique in `task_records`; `system_id` is unique in `systems`; and every trial references a declared system.
2. For each eligible `task_id × attempt_index`, every compared system has exactly one row. A declared `paired_group_id`, seed, task spec, base-tree digest, environment, verifier version, and budget policy must match across the comparison arms.
3. Duplicate cells, omitted cells, inconsistent pair keys, or changed task manifests invalidate the comparison. Do not repair them by silently taking the first or best row.

Seed-policy semantics are explicit and predeclared in the frozen manifest: `fixed_paired` and `predeclared_random` use the exact per-attempt seed committed before execution and shared across every system arm; `unseeded` uses the literal marker `unseeded`; and `not_applicable` uses `not_applicable`. Every trial must match the manifest's attempt-indexed seed. Never invent or select seeds after seeing system outcomes.
4. The verifier's `bundle_digest` matches the run's pinned verifier bundle, and `evaluated_tree_digest` equals the trial's `final_tree_digest`.
5. A `pass` outcome requires verifier `pass`, all blocking checks `pass`, and integrity `clean`. Contradictions are invalid records and must fail closed.
6. The same immutable environment, isolation policy, task budgets and retry policy apply across paired systems, or each difference must be explicitly declared as a separate experimental factor.
7. Token or cost fields unavailable from a provider remain null/omitted where permitted; do not encode unavailable telemetry as zero.

The Rust module now contains **66 authored unit-test cases**, including run-record invariants, schema-v3 requirement/rubric checks, exact source-span verification, and filesystem-loader controls. The loader tests cover digest-addressed resolution, deduplication, non-regular files, symlink rejection on Unix, per-artifact limits, aggregate-limit boundaries, and arithmetic overflow. They are authored but **not executed**: the local environment has no Rust toolchain/checkout, and the current workspace metadata failure prevents CI jobs from reaching the BASALT crate. Do not treat these tests as passing evidence.

The manifest digest establishes byte identity, not authenticity. An actor who can rewrite both the manifest and run record can recompute the digest. Signed, independently pinned pre-run and post-run receipts are tracked in [#9034](https://github.com/Luminous-Dynamics/symthaea/issues/9034); confirmatory scoring must remain disabled until that trust boundary is qualified.

## Exact specification artifact verification

The validator requires an offline `--spec-root` directory. For a task with specification digest `sha256:<hex>`, the UTF-8 specification artifact must be present as `<spec-root>/<hex>.txt`. The validator reads it with a 4 MiB per-artifact bound, recomputes the full SHA-256 and byte length, checks UTF-8 validity, resolves every requirement's half-open byte range, and recomputes `statement_digest` from the exact selected bytes. A missing artifact, digest/length mismatch, invalid UTF-8, out-of-range span, invalid UTF-8 boundary or statement-digest mismatch fails closed.

This makes source spans executable evidence rather than declarations alone. The expectation digest remains a content commitment, not proof of the expectation's semantic validity; independent requirement adjudication and oracle qualification remain separate gates. Confirmatory holdout specifications must remain outside the agent's policy-visible workspace and be mounted only into the isolated evaluator boundary.

The filesystem loader canonicalizes the supplied specification root, rejects symbolic-link and non-regular artifact entries, checks that each resolved file remains beneath the root, deduplicates artifacts by SHA-256, and enforces both a 4 MiB per-artifact bound and a 128 MiB aggregate bound. These are local artifact-loading and resource-safety guarantees; they do not authenticate the manifest issuer, prevent a concurrently malicious filesystem actor from racing file replacement, or establish semantic adequacy of the requirements. The signed pre-run receipt and isolated evaluator controls remain mandatory for confirmatory use.

The synthetic specification is public only because it is a deliberately fabricated development fixture. The same file layout does not imply that holdout specifications should be committed to a public repository.

## Event-sourced evidence freshness audit

The separate `xtask basalt-state-audit --input STATE-TIMELINE.json` command audits ordered resource changes and evidence records. It starts resources at generation zero, requires every evidence record to pin the exact current resource digest and generation, and uses prior evidence IDs as explicit dependency edges. A resource change increments generation and invalidates direct evidence dependents plus their transitive descendants. Unrelated evidence is retained; digest reversion does not resurrect old evidence. The report distinguishes fresh, directly stale, and transitively stale evidence, and emits a topological re-verification queue. Each evidence record must declare `dependency_coverage=complete|incomplete` and name any unresolved inputs. Incomplete evidence is never eligible under the declared coverage, is queued even before a change, and is conservatively invalidated by every later resource change. Descendants are invalidated transitively. The output distinguishes `structurally_valid` from `confirmatory_eligible`; the latter is always false because a producer's completeness assertion and unsigned timeline do not prove complete observation or issuer identity.

See [the evidence freshness protocol](BASALT_EVIDENCE_FRESHNESS_2026-10-30.md), the [state timeline schema](../../benchmarks/basalt/schema/state-timeline.schema.json), and the [synthetic timeline](../../benchmarks/basalt/examples/state-timeline.example.json).

This auditor establishes consistency relative to the supplied timeline; it does not authenticate event issuers, prove that a runner observed every real change, or establish that recorded checks executed honestly. Event/evidence receipt verification and isolated runner delivery remain mandatory before confirmatory scoring.

## Frozen requirement-to-verifier coverage ledger

Suite-manifest schema version 3 adds a per-task requirement ledger. Each atomic requirement has a unique ID, a content-addressed source span into the pinned task specification (UTF-8 byte offsets, half-open range bounded by the frozen specification byte length), a statement digest, a pre-run expectation digest, explicit/inferred origin, ambiguity disposition, and one or more mapped blocking check IDs. Inferred requirements additionally require a content-addressed rationale; explicit requirements must not carry an inference rationale.

The validator enforces that source-span digests equal the frozen task `spec_digest`, spans are non-empty, requirement IDs are unique, accepted tasks contain no materially unresolved requirement, and every requirement maps to a check declared as required and blocking in that task. The run validator still does not possess the hidden task-spec bytes or oracle semantics, so it cannot prove the offsets resolve to the stated requirement, that the interpretation is correct, or that the mapped test actually covers the behavior. Those are independent reference-assurance and reviewer duties tracked in [#9115](https://github.com/Luminous-Dynamics/symthaea/issues/9115) and [#9081](https://github.com/Luminous-Dynamics/symthaea/issues/9081).

The synthetic fixture's two requirements and digest values are illustrative placeholders only. They are not real prompt quotations, executed tests, or evidence of oracle coverage.

## Frozen task-specific patch-quality rubric

Suite-manifest schema version 3 freezes a task-specific requirement ledger and task-specific rubric before any system outcome is observed. Each task declares a rubric ID/version, a 10000-basis-point total, unique criteria, an explicit applicability decision and rationale, admissible evidence sources, reviewer guidance, and five ordered anchors (scores 0 through 4) for every criterion.

Only criteria marked `scored` carry positive weight; their weights must sum exactly to 10000 basis points. A `not_applicable` criterion must carry zero weight and a non-empty pre-run rationale. Criterion IDs and evidence sources are unique. The validator rejects duplicate criteria, missing evidence, missing guidance, unordered anchors, invalid applicability weights, and weight totals that drift from the declared total. Changing any rubric byte changes the manifest digest and therefore invalidates a record pinned to the previous manifest.

The rubric definition is a frozen measurement instrument, not a score. This validator does not yet accept reviewer ratings or calculate a quality composite. A later scoring/receipt contract must preserve blinded first-pass ratings, reviewer role, evidence digests, adjudication, agreement, and judge-to-human calibration. Quality is reported only conditional on strict verified success; it can never rescue a failed correctness, security, scope, or integrity blocker. LLM ratings remain advisory until independently calibrated.

The synthetic example rubric is illustrative only. Its criterion weights and anchors are not empirical or validated rating standards.

## Oracle strength and evaluator evolution

A fixed verifier is necessary for reproducibility but may not cover every behavior required by the task. TestJack (7 October 2026) demonstrates a concrete evaluator-evolution approach: decompose the prompt into testable requirements, propose tests for uncovered behavior, validate each test against the ground-truth patch, and use an independent reviewer to reject tests that encode incidental reference-implementation details. Across the paper's five evaluated coding benchmarks, the authors reported that 1,545 of 4,487 trials previously marked as passing (34.4%) violated task requirements; the reported aggregate resolution rate fell from 50.6% to 33.2%. These findings are specific to that study, not a universal rate and not a BASALT result.

BASALT tracks an implementation protocol in [#9115](https://github.com/Luminous-Dynamics/symthaea/issues/9115). The task specification and independently reviewed requirement ledger—not the reference patch—define expected behavior. The reference patch is an executable test oracle only. A candidate-conditioned witness test is accepted only when it is grounded in a frozen requirement, passes on the pinned reference, and is reviewed against incidental-reference overfit. Ambiguous or unsupported requirements become task defects, not post-hoc candidate failures.

Accepted witness tests are task-scoped and replayed against every system/attempt arm for that task before publishing an oracle-hardened comparison. Preserve the original fixed-suite result separately from the hardened result. The report must distinguish deeply audited trials, trials screened by shared witnesses, and static-suite-only trials. Deep-audit sampling probability, witness yield, false-reference-rejection rate, replay coverage, evaluator cost, and per-task paired outcomes remain visible. Test-generation and replay costs are not silently charged to only one agent arm.

This does not prove correctness beyond the tested requirement surface. LLM-generated tests and reviewers require separate qualification; the reference patch cannot resolve ambiguous task intent; and no candidate-specific test may be used to rank systems unless the same accepted witness pool is replayed across all arms. See also evaluator reliability and reference assurance in [#9059](https://github.com/Luminous-Dynamics/symthaea/issues/9059) and [#9081](https://github.com/Luminous-Dynamics/symthaea/issues/9081).

## Analysis plan
Patch-quality ratings are a separate secondary endpoint tracked in [#9084](https://github.com/Luminous-Dynamics/symthaea/issues/9084). Report them only conditional on strict verified success; no maintainability/readability score may compensate for a failed blocking obligation. Preserve raw criterion ratings, evidence, adjudication, and reviewer agreement rather than publishing only a composite.


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

Public leaderboards measure different mixtures of task resolution, harness, tools, budget and scoring. As of 2026-10-30, Terminal-Bench 4.0's official leaderboard lists Opus 5.5 + Claude Code at 64.8% ± 3.1%. Artificial Analysis Coding Agent Index v2.0 equal-weights FrontierCode v1.1 (100 repository-engineering tasks), Terminal-Bench 4.0 (66 terminal tasks), and SWE-Atlas-QnA-AA (124 repository-Q&A tasks); it runs three attempts per task and reports cost, tokens, time, and reward-hacking checks. Treat these as version-specific reference points, not a universal SOTA number.

BASALT's differentiated aim should be **verified change per unit cost under controlled shift**, with explicit evidence integrity—not a larger composite score obtained by blending incomparable metrics. SWE-Shift (23 October 2026) provides a relevant precedent: 120 scenarios, 12 change families and nine intervention positions, with lower shifted than clean success for all 17 evaluated systems. BASALT should test this with matched continuations, preservation obligations and evidence-freshness checks. Compare with public scores only after pinning the benchmark version and matching the executable setup.

## References

- [Artificial Analysis Coding Agent Index v2.0 methodology](https://artificialanalysis.ai/methodology/coding-agents-benchmarking)
- [Terminal-Bench 4.0 leaderboard](https://www.tbench.ai/?version=4.0)
- [Vals AI Terminal-Bench 4.0 leaderboard and methodology](https://www.vals-ai.com/benchmarks/terminal-bench-4)
- [OpenAI: Separating signal from noise in coding evaluations (July 2026)](https://openai.com/index/separating-signal-from-noise-coding-evaluations/)
- [Epoch AI: Terminal-Bench 4.0 review (September 2026)](https://epoch.ai/benchmarks/terminal-bench-4/review)
- [TestJack: evaluator evolution with requirement-grounded witness tests (October 2026)](https://arxiv.org/abs/2610.10619)
- [in-toto Attestation Framework v1.2](https://github.com/in-toto/attestation/blob/main/spec/README.md)
- [SLSA v1.2 specification](https://slsa.dev/spec/v1.2/)
- [GitHub artifact attestations](https://docs.github.com/en/actions/concepts/security/artifact-attestations)


For current task-quality context, Scale AI's SWE-Bench Pro V2 update on 22 September 2026 reduced its public split to 642 tasks across 11 repositories after removing 89 invalid tasks and correcting 69 instruction/test contradictions. It disabled agent web tools and re-graded every candidate diff in a pristine image, detecting Go module checksum and cache tampering. BASALT should adopt the same separation between agent-visible workspace and pristine verifier, while explicitly testing patch-controlled verifier execution and environment integrity. Artificial Analysis Coding Agent Index v2.0 now combines 100 private FrontierCode v1.1 maintainer-merge tasks, 66 Terminal-Bench 4.0 tasks, and 124 SWE-Atlas-QnA-AA tasks. Reproduce public lanes with matched harness/budgets and use private maintainer-merge results as methodological context only; do not compare their composite directly with BASALT.