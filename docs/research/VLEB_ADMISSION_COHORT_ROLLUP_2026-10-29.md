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

This tool does **not** average per-task percentages. It reports pooled numerator and denominator counts beside every percentage. Missing or invalid reports remain in the locked task denominator for task-coverage and admission-rate metrics; the cohort is marked incomplete and integrity-invalid rather than silently shrinking. If some report counts cannot be verified, only the diagnostics from individually verified reports are aggregated and the overall cohort remains invalid.

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
