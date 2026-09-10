# Distributed-health V2 — V1 compatibility invariant

Status: **design/qualification requirement; not runtime evidence**.

V2 extracts the closed-world distributed-health evaluator so V1 and V2 cannot silently drift into different participant, failure-domain, freshness, or recovery semantics.

The compatibility invariant is:

```text
same V1 local-health input
+ same distributed context
+ same currentness/failure-domain policies
+ same authenticated evidence set
+ same evaluation time
    -> same V1 accept/deny semantics
    -> same V1 current-state digest
    -> same V1 qualified identity
```

The shared evaluator is protocol-neutral. V1 maps generic evaluator errors back into its historical `PostTransitionDistributedHealthError` vocabulary through an isolated compatibility bridge. V2 exposes generic distributed-health errors directly.

V1 domain separators, field ordering, sorting rules, recovery-class encoding, and `QualifiedPostTransitionDistributedHealthId` hash contract must not change as a side effect of extraction.

Qualification must include regression vectors that compare the pre-extraction V1 subject with this V2 branch across successful worlds and representative fail-closed worlds, including duplicate evidence, unknown participants, budget violations, failure-domain violations, recovery-path failures, stale/future evidence, and cross-evidence skew.

If any V1 identity or verdict changes for the same canonical input world, this subject fails qualification and must not be promoted as a semantics-preserving refactor.
