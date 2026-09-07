# Native Interoception v0.1 — Study Evidence Policy

This document defines the inference boundary layered above the mechanical native-interoception protocol. It does not change the scientific semantics of the interoceptive state model. Its purpose is to prevent exploratory data, excluded runs, indeterminate runs, or analyst-created metric artifacts from being silently promoted into confirmatory evidence.

## Study identity

`StudyPreregistration` wraps the lower-level `ExperimentPreregistration` with an explicit `EvidenceRunClass`:

- `Exploratory`
- `Confirmatory`

The run class is part of the canonical study JSON and therefore part of the study SHA-256. The same mechanical protocol labeled exploratory and confirmatory has two different study identities.

A confirmatory study additionally requires semantic arm identity to remain blinded during primary analysis. An exploratory study may still use blinding, but exploratory status cannot later be changed to confirmatory without creating a different prospective study identity.

## Execution identity

`execute_study` produces a `StudyExecutionTrace` that binds:

- the study run class;
- study-preregistration SHA-256;
- the complete replayable lower-level `ExecutionTrace`.

The study trace validates by re-executing the locked lower-level protocol under the declared execution limits. Exact divergence is an error.

## Exclusion decisions

Every exclusion criterion declared in the prospective protocol must have exactly one `ExclusionCriterionDecision` in an `ExclusionDecisionReceipt`.

Each decision records:

- criterion identifier;
- status: `NotTriggered`, `Triggered`, or `Indeterminate`;
- SHA-256 of the evidence used to make that decision.

An evidence digest is required even for `NotTriggered`. Inclusion therefore leaves a positive audit trail rather than being represented by absence of a record.

Unknown criteria, duplicate decisions, missing decisions, malformed evidence digests, run-class mismatches, study-digest mismatches, and execution-digest mismatches invalidate the receipt.

Disposition is fail-closed:

- any `Triggered` criterion => `Exclude`;
- otherwise any `Indeterminate` criterion => `Indeterminate`;
- only all `NotTriggered` decisions => `Include`.

Excluded and indeterminate traces should be preserved. They are not deleted merely because they cannot support the confirmatory inference set.

## Blinded metric boundary

`extract_study_blinded_metrics` binds the lower-level blinded metric report to:

- study run class;
- study-preregistration digest;
- exclusion-decision digest;
- computed run disposition.

Metric extraction remains possible for excluded or indeterminate studies so the underlying data can be retained and inspected. `confirmatory_eligible()` is true only for a `Confirmatory` study whose disposition is `Include`.

## Qualified confirmatory path

The public confirmatory path is `evaluate_confirmatory_study_bound`.

It requires all of the following artifacts at once:

1. locked `StudyPreregistration`;
2. exact replayable `StudyExecutionTrace`;
3. complete `ExclusionDecisionReceipt`;
4. previously produced `StudyBlindedMetricReport`;
5. declared execution limits.

Before semantic-arm unblinding, the function:

- revalidates the study;
- replays the execution;
- revalidates every exclusion decision;
- verifies run class and all study/exclusion digests;
- recomputes the exclusion disposition;
- recomputes the blinded registered metrics from the exact execution;
- requires exact equality with the supplied blinded report.

Only after all of those checks pass can the lower-level hypothesis evaluator be reached.

The lower-level study evaluator is crate-internal and is not exported as a public confirmatory API. This prevents a caller from presenting a fabricated study-level blinded metric artifact as qualified confirmatory evidence merely because its arm and metric identifiers are structurally valid.

## Confirmatory non-promotion rules

A study cannot produce qualified confirmatory evidence through the study API when any of the following is true:

- the prospective run class is `Exploratory`;
- primary semantic arm identity was not blinded for a confirmatory study;
- any preregistered exclusion criterion is triggered;
- any exclusion decision is indeterminate;
- any exclusion criterion lacks a decision record;
- exclusion evidence is malformed or does not bind the exact study execution;
- the blinded metric report does not exactly reproduce from the execution trace;
- the study, execution, exclusion, or blinded artifact identities do not match.

A later desire to run a confirmatory experiment after an exploratory result requires a newly locked confirmatory `StudyPreregistration` and new confirmatory data. The exploratory observations remain exploratory evidence.

## Null and negative results

A confirmatory hypothesis that is not satisfied remains part of the evidence lineage. The qualified path does not discard failed hypotheses, excluded runs, or indeterminate runs. Their distinct statuses are preserved so that a later report can distinguish:

- hypothesis not supported on an included confirmatory run;
- run excluded by a preregistered criterion;
- run indeterminate because an exclusion decision could not be resolved;
- exploratory result not eligible for confirmatory inference.

This distinction is required for the Affective Emergence program because negative evidence is scientifically informative and should not be collapsed into missing data.

## v0.2 gate

Before the first observational regulatory-affect experiment is considered confirmatory:

- Native Interoception v0.1 must qualify on one exact source head;
- the study must be prospectively labeled `Confirmatory`;
- the study SHA-256 must be frozen before the primary run;
- exclusion criteria must be declared prospectively;
- every criterion must receive an evidence-bearing decision receipt;
- the execution trace must replay exactly;
- blinded metrics must reproduce exactly from that trace;
- the blinded metric artifact should be frozen before semantic-arm unblinding;
- only `evaluate_confirmatory_study_bound` output should be treated as qualified confirmatory hypothesis evidence.

No output from this study layer establishes emotion, feeling, attachment, sentience, or consciousness. It controls the evidential status of later experiments; it does not determine their interpretation.