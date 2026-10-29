# VLEB shift-event auditor self-mutation protocol — 2026-10-29

## Goal

A test suite that passes ordinary positive and negative fixtures may still fail to detect a guard bypass. This campaign makes the shift-event evidence auditor an explicit adversarial target.

## Frozen sentinel set

The current catalog contains 13 reviewed bypass mutations across five guardrail families:

- trusted input pins: protocol, raw manifest and verifier-source pins;
- scenario identity: scenario digest and condition-specific task-contract bindings;
- starting-state identity: initial workspace/checkpoint/revision/image pins;
- shift-event integrity: event payload, change-family classification, and runner-origin/applied declaration;
- evidence freshness: complete stale-artifact disposition, final-workspace test binding, and outcome-vs-check consistency;
- attempt integrity: retry substitution.

The campaign requires the unmodified regression suite to pass first. Each mutation anchor must occur exactly once. Mutated code must compile, the suite must demonstrably start, and the named regression test must fail to count as a kill. Compile errors, timeouts, suite-start failures, unrelated failures and surviving mutants are not kills. Every sentinel and family must be accounted for. The report is analysis-only and never changes benchmark score eligibility.

The sentinels are frozen reviewed guard bypasses, not an exhaustive proof of correctness. A 100% kill rate would only show that the designated tests detect this finite mutation catalog.

## Proposed target

Require 100% kill of the frozen set with zero invalid, surviving or unrun sentinels; review mutation equivalence and maintain coverage across every guardrail family. Add new sentinel families only with independently reviewed regression tests and versioned catalog changes. The broad robustness score remains distinct from this evaluator self-test.

## Trust boundary

The runner still owns network/resource isolation and signed evidence provenance. This campaign tests the sensitivity of the local evidence-consistency checks; it cannot prove that a real shift was applied or that runner logs are authentic.
