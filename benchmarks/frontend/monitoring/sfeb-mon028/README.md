# MON-028 — Falsification-First Assurance Benchmark

**Status:** benchmark contract, independent ground-truth suite model, standard-library scorer, Wilson confidence bounds, matched comparison rules, CLI, tests and scenario corpus added; exact-head Go execution and cross-system reproduction pending.

MON-028 addresses a core weakness in many reliability and chaos benchmarks: the system being tested, its orchestrator, or its dashboard should not control the expected answer or decide which failed experiments count.

## Paradigm

The benchmark authority publishes a versioned, preferably MON-026/027-signed suite containing the exact expected outcome for every case and replicate. The system under test returns only:
- case/replicate identity;
- execution state (executed, skipped, timed out, infrastructure error);
- observed verdict;
- elapsed time and peak memory;
- evidence/output digests.

The scorer joins the result file to the separate suite. It does not trust an expected verdict supplied by the candidate. Missing, duplicate, skipped and timed-out observations remain visible.

## Metrics are not blended

The scorer reports:
- **false-qualified:** expected blocked/indeterminate, observed qualified;
- **false-rejected:** expected qualified, observed blocked/indeterminate;
- exact classification matches and safe-but-different outcomes;
- incomplete/skip/timeout/infrastructure counts;
- one-sided 95% Wilson upper confidence bound for false qualification;
- p50/p95/p99 latency over executed observations;
- completeness and safety gates separately.

A safety gate requires a complete campaign, zero false-qualified outcomes and 100% acceptance of positive controls. It does not average unsafe acceptance away with throughput or latency.

## Matched comparisons

Two systems may be compared only when they use the same campaign, ground-truth suite, fault matrix, workload, environment and resource-budget digests, and have the same case/replicate pairing. A result cannot claim superiority by running fewer faults, consuming more hardware, or using an easier workload.

## Industry baseline

MON-028 complements, rather than replaces, fault-injection platforms:
- Jepsen combines controlled clients, a nemesis and history checkers to evaluate whether observed operations match a consistency model.
- Chaos Mesh provides serial/parallel workflow orchestration and a wide fault surface.
- AWS Fault Injection Service provides experiment templates, steady-state hypotheses and CloudWatch stop conditions.

MON-028's proposed additional contribution is independent known-answer scoring for evidence qualification and recovery claims, including strict accounting for every assigned case. Those platforms' safety controls and orchestration are useful integration targets, not competitors to imitate superficially.

## Use

See `benchmark/` for:
- independent `Suite` and candidate `CampaignResult` types;
- deterministic validation and scoring;
- matched comparison;
- JSON CLI;
- tests, fuzz target and microbenchmark.

Start with `SUITE-example.json`. Run `bash CHECKS.sh` from this directory on a machine with Go 1.23+ to execute formatting, vet, unit, race, fuzz and benchmark checks.

The benchmark suite itself should be signed/custodied with MON-026/027. The scorer's output should also be exported as a MON-026 artifact, with benchmark matrix digest, oracle digest, system implementation digest, environment and resource envelope.

## Status and limits

- JSON profiles and seed data are structurally authored; they have not yet been validated by the Go CLI on this exact head.
- No MON-025 live experiment output exists from this authoring environment.
- No systems have been measured against the suite; there is no SOTA or vendor-superiority claim.
- Exact Go test/race/vet/fuzz/benchmark execution remains pending.
