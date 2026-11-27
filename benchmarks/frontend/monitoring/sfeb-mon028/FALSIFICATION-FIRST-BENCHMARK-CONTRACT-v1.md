# MON-028 — Falsification-First Assurance Benchmark Contract v1

**Profile:** `mon028-falsification-first-benchmark-v1`

## 1. Governing principle

The benchmark oracle is independent of the system under test (SUT). The benchmark authority owns case IDs, replicate counts, expected outcomes, fault schedule and test data. The SUT cannot submit or change expected answers.

Every assigned case/replicate must be represented exactly once in the result file. A skipped, timed-out, missing, duplicated or infrastructure-failed observation is never silently excluded.

## 2. Three outcome classes

- `qualified`: the evidence supports the benchmarked claim.
- `blocked`: evidence positively contradicts the claim or detects a known safety violation.
- `indeterminate`: evidence is missing, untrusted, incomplete or not sufficiently bounded.

These are ordered neither as a scalar quality score nor as interchangeable safe outcomes. Both blocked and indeterminate must prevent an unsafe qualification.

## 3. Independent suite

The signed suite uses the MON-028 domain-separated, length-prefixed binary suite root (fixed field order; cases sorted by CaseID) and binds:
- campaign and benchmark IDs;
- exact fault-matrix digest;
- ground-truth oracle digest;
- workload digest;
- environment profile digest;
- resource-budget digest;
- test cases, family, expected verdict and exact replicate count.

The suite file is created by the benchmark authority and must be signed/custodied separately. The verifier requires the exact expected custody run ID, sequence and previous root (when applicable), as well as an out-of-band trusted public key. The result bundle must use a different trusted key ID and public key controlled by an independent benchmark observer. The SUT receives the case schedule and inputs, not an editable expected-outcome field in its result schema.

## 4. Candidate observations

For each case/replicate the SUT records:
- case ID and replicate number;
- execution state: `executed`, `skipped`, `timed_out` or `infrastructure_error`;
- observed verdict only when executed;
- elapsed nanoseconds and peak memory;
- output digest and evidence-bundle digest.

The scorer validates exact coverage of the suite. An observation for an unknown case, duplicate replicate, invalid execution state, executed observation without a verdict, or skipped observation with an asserted verdict is malformed.

## 5. Metrics

### Safety

`false_qualified` counts cases whose oracle outcome is blocked/indeterminate but the SUT reports qualified. This is the primary safety error.

Report:
- false-qualified count;
- completed unsafe-case denominator;
- planned unsafe-case denominator;
- observed false-qualified rate;
- one-sided 95% Wilson score upper bound;
- false-qualified count and confidence bound by case family and system version.

A campaign with any false-qualified result fails the safety gate. Zero false qualifications is not proof of zero risk; the confidence bound must be reported.

### Usefulness

For oracle-positive cases, report:
- accepted as qualified;
- false-rejected as blocked;
- unresolved as indeterminate;
- positive-control acceptance rate.

For all executed observations, report exact verdict agreement and the full confusion matrix. Safe-but-different (blocked vs indeterminate) outcomes must remain visible.

### Operations and performance

Report all assigned, executed, skipped, timed-out and infrastructure-error cases, plus:
- p50/p95/p99 scorer-observed latency;
- throughput and recovery RPO/RTO from domain artifacts;
- peak memory and resource usage;
- operator intervention counts;
- evidence retention and verification overhead.

Latency is descriptive unless the benchmark specifies a required operational bound. A fast false qualification is not a performance win.

## 6. Confidence bound

The scorer reports a one-sided 95% Wilson score upper confidence bound for the false-qualified proportion. For zero observed errors among (n) completed unsafe cases, the upper bound remains nonzero. With 10,000 independent unsafe trials, it is approximately (2.7\times10^{-4}) (0.027%).

Replicates sharing a reset, host, fault injector or common-cause event may not be statistically independent. Confirmatory analysis must report clustering and use an appropriate cluster-aware interval or resampling method. The built-in Wilson interval is a screening statistic, not a substitute for the confirmatory design. Suite validation caps a single suite at 1,000,000 planned observations and 100,000 case definitions to bound scorer resource use.

## 7. Campaign completeness and gates

A campaign is complete only when:
- suite and campaign profiles/digests match;
- all assigned case/replicate pairs exist exactly once;
- every case finished as executed;
- all required result metadata is present;
- no expected answer is sourced from the SUT;
- all artifacts are hash-addressed.

Safety gate:
- campaign complete;
- zero false-qualified;
- every positive control qualified.

Usefulness gate:
- safety gate passes;
- all required positive cases accepted;
- no hidden execution failures.

Exact classification agreement is reported separately. The benchmark does not replace domain-specific qualification.

## 8. Matched system comparisons

The suite and result bundles must be independently verified before scoring. Candidate and baseline must share:
- same campaign and suite digest;
- same fault matrix, oracle and workload digest;
- same hardware/environment digest;
- same resource-budget digest;
- same replicate IDs and execution policy.

Implementation digest, system identity and version are expected to differ. Runs with mismatched shared digests are not comparable.

Compare paired case-level outcomes. Report candidate-minus-baseline deltas for false qualifications, false rejections, exact matches, p95 latency, resource consumption and interventions. Do not calculate a single weighted score that permits safety loss to be compensated by speed.

## 9. Prospective acceptance targets (not measured)

- 0 false-qualified in at least 10,000 independent adversarial unsafe trials, with a reported confidence bound.
- 100% positive-control acceptance.
- 100% accounting of assigned replicates.
- 0 silently dropped timeout/skipped/infrastructure observations.
- 100% repeatability of suite digest and expected answers across independent benchmark operators.
- 100% source/restore known-answer comparison under the declared scenario set.
- p95 scoring overhead below 5% of end-to-end experiment duration, measured separately from system RTO.

## 10. Independent reproduction

At least two independent operators should run the same signed suite, pinned SUT version/config, identical workload, resource budget and fault schedule on independently provisioned environments. Preserve raw histories and artifacts. Report exact matching and explain any environmental deviations; never pool unlike profiles.

## 11. Limitations

The scorer can validate result accounting and independently authored expectations; separate custody identities reduce self-certification, but do not prove that the oracle is correct, that a fault was injected as claimed, or that the independent observer did not collude with the SUT. MON-025 receipts, MON-026 custody, MON-027 external signing, independent operators and domain-specific checkers remain necessary.
