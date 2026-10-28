# VLEB admission-cohort rollup — 2026-10-29

## Purpose

Per-task assurance reports need a cohort roll-up that preserves the locked task denominator and makes every numerator/denominator auditable. A cohort summary must not take an unweighted average of task percentages, omit tasks that fail, or turn analysis into score eligibility.

## Contract

An evaluator-authored `vleb-admission-cohort-v1` manifest freezes:
- a unique cohort ID and the run kind;
- the exact task-ID inventory in order;
- the protocol digest;
- one audit-report path and SHA-256 per assigned task;
- expected report receipt, task-contract, protocol, candidate-patch and auditor digests.

The cohort manifest's SHA-256, schema digest, protocol digest, original auditor digest, and this summarizer's source digest are supplied out of band. Each report is loaded with strict JSON parsing, re-hashed from disk, matched to its task and evaluator-pinned digests, and checked for internally consistent summary counts. Traversal, symlinks, duplicate keys, NaN/Infinity, unknown tasks, duplicate rows, and missing results are rejected.

## Metrics

The rollup pools counts across tasks:
- source-item coverage = traced source items / declared source items;
- required-test trace coverage = traced required test IDs / required test IDs;
- per-test baseline alignment = matching baseline statuses / all mapped concrete executions;
- audit eligibility = audit-eligible tasks / all assigned task IDs;
- full admission pass rate = tasks whose audit, requirement traceability, and per-test baseline alignment all pass / all assigned task IDs.

This tool does **not** average per-task percentages. It reports pooled numerator and denominator counts beside every percentage. Missing or invalid reports remain in the locked task denominator for task-coverage and admission-rate metrics; the cohort is marked incomplete and integrity-invalid rather than silently shrinking. If some report counts cannot be verified, only the diagnostics from individually verified reports are aggregated and the overall cohort remains invalid. The output labels this explicitly with `pooled_metrics_population="verified_reports_only"` and `pooled_metrics_complete=false`; it switches to `all_locked_tasks` only when every expected report is present, valid, and correctly pinned. Task-level admission rates always use the full preregistered task count.

## Why this is a paradigm shift

Treat evidence quality as a measurable property of the evaluation rather than a preamble to model scoring. This lets VLEB ask two separate questions: “Did the agent solve the task?” and “How much of the task was actually eligible for trustworthy scoring?” The latter is reported as a separate admission audit and never folded into a weighted model score.

It also prevents aggregation laundering: a 100% result from a tiny task does not offset missing results from a large task cohort, and a per-task percentage is never treated as an independent observation when its denominator differs from other tasks.

## Usage

Run from a trusted checkout with evaluator-held pins:

```sh
python tools/vleb/summarize_admission_cohort.py \
  --manifest /evaluator/cohort.json \
  --artifact-root /evaluator/audit-reports \
  --expected-cohort-sha256 "$TRUSTED_COHORT_SHA256" \
  --expected-schema-sha256 "$TRUSTED_COHORT_SCHEMA_SHA256" \
  --expected-protocol-sha256 "$TRUSTED_VLEB_PROTOCOL_SHA256" \
  --expected-auditor-sha256 "$TRUSTED_RECEIPT_AUDITOR_SHA256" \
  --expected-summarizer-sha256 "$TRUSTED_COHORT_SUMMARIZER_SHA256"
```

The output always states `analysis_only=true` and `score_eligible=false`. It is a cohort diagnostic; protected provenance, receipt validation, oracle review, and task-level eligibility remain separate requirements.

## Industry baseline and what “better than SOTA” should mean

The current public leaderboard is not a safe single target. Scale's September 22, 2026 SWE-Bench Pro V2 release reports a top public score of 99.4% on 642 tasks, but the refresh itself removed 89 invalid tasks and corrected 69 instruction/test conflicts; its protocol includes restricted network access, pristine-image regrading, and the two-sided empty-patch-fails/reference-patch-passes gate ([Scale leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full), [release explanation](https://labs.scale.com/blog/swe-bench-pro-v2)). A July 2026 OpenAI audit of the *older* SWE-Bench Pro version separately estimated around 30% of tasks had grading defects ([OpenAI audit](https://openai.com/index/separating-signal-from-noise-coding-evaluations/)). These are different benchmark revisions, not contradictory estimates of one frozen dataset; the lesson is that leaderboard scores need versioned validity evidence.

For legacy-code comprehension, LegacyCodeBench (January 2026) describes 200 COBOL programs across four tiers and reports 88–92% for specialized systems and 86–90% for general models using deterministic claim-based behavioral verification ([benchmark paper](https://kalmantic.github.io/legacycodebench/)). Separately, Agent Security League reports a top listed combination at 87.2% functional correctness but 37.4% security correctness ([benchmark dashboard](https://www.endorlabs.com/research/ai-code-security-benchmark)). Those scores use different tasks and protocols, so VLEB must not compare their raw percentages directly. They do motivate a dual-axis goal: verified functional behavior and secure behavior, with independent evidence quality as a third axis.

### Recommended targets

- **100% task admission integrity:** every scored task has source-to-requirement-to-test traceability, reviewed baseline expectations, baseline-fails/reference-passes controls, verified test execution, and immutable task/evidence digests.
- **100% per-test baseline alignment:** each mapped concrete test has one frozen expected status, the baseline log matches it, and reference execution passes every required test.
- **0 silent denominator loss:** every preregistered task is present in the cohort inventory; unavailable/invalid audit evidence invalidates cohort completeness instead of quietly deleting a task.
- **100% report provenance coverage:** every report is digest-pinned and tied to the expected task, protocol, source/auditor identity, and receipt.
- **Separate quality dimensions:** publish task-level resolved rate, source-item coverage, test-ID trace coverage, per-test baseline alignment, audit-eligible task rate, mutation adequacy, security correctness, cost per independently verified fix, and failure/abstention rates. Do not collapse these into a single weighted “trust score.”
- **Statistical comparison:** compare systems on the same locked tasks and budgets, pair results at task level, bootstrap by repository/codebase cluster, disclose model/scaffold/tool/network settings, and report unresolved outcomes rather than dropping them.
- **Performance claim bar:** do not claim VLEB is above SOTA until the cohort gate, exact-head CI, independent rerun, security lane, and preregistered uncertainty analysis have passed. Public benchmark scores are comparators, not a substitute for a measured VLEB result.

This roll-up provides auditable denominators for the admission dimensions. It is not itself a score-admission gate and always emits `score_eligible=false`.


## Exact execution binding (follow-up)

Each cohort row now pins the concrete receipt run ID and attempt number, expected checkout revision, immutable runtime image ID, and expected network mode in addition to the audit-report digest and protocol/auditor pins. The independent auditor emits those fields only after checking its receipt and execution environment; the rollup compares them with the evaluator-pinned row. This makes replaying an otherwise valid report from a different run attempt, checkout revision, or runtime image a cohort-integrity error.

This still depends on the protected evaluator being the authority for the cohort-manifest digest and pinned report digest. A JSON report containing identity fields is not by itself a cryptographic attestation; the live confirmatory pipeline must continue to verify the actual GitHub/Sigstore evidence bundle separately.

## Meta-evaluation: mutate the evaluator's own guardrails

The workflow now also runs `tools/vleb/run_auditor_mutation_campaign.py`. It first executes the unmodified cohort rollup test suite and requires it to pass. It then produces eleven reviewed, syntactically valid guard-bypass mutations: bypass cohort/report digests, weaken locked task inventory, accept an unexpected task, drop run/attempt or revision/image/network binding, restore permissive JSON parsing, ignore eligible-report/admission consistency, and leak score eligibility.

A mutation is counted as killed only when the test suite actually starts and fails on the mutated implementation. Syntax errors, import/startup failures, and timeouts count as invalid, not killed. The campaign fails if any sentinel survives or is invalid. Target is 100% kill of this frozen sentinel catalog; the metric is not a candidate success score and the tool emits `score_eligible=false`.

This is a first-step meta-evaluation rather than a comprehensive proof that the auditor is correct. The sentinel catalog is deliberately reviewable; expanding it with independently designed mutants and mutation families is part of the next research phase.
