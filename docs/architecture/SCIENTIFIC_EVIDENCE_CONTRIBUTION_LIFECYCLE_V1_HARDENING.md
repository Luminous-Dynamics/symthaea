# Scientific Evidence Contribution Lifecycle v1 — adjudication separation hardening

**Status:** semantic correction companion to `SCIENTIFIC_EVIDENCE_CONTRIBUTION_LIFECYCLE_V1.md`.

This note corrects one over-broad illustrative list in the parent lifecycle document before any implementation exists.

## 1. Core correction

The source/publisher lifecycle stream and the external scientific argument/adjudication graph are separate structures.

The parent document's illustrative lifecycle list included `ExternalInvalidationDeclared`, `LifecycleChallengeDeclared`, and `LifecycleResolutionDeclared` beside source lifecycle actions. That grouping is too broad.

The stronger theorem is:

```text
source lifecycle event
    != external adjudication object
    != argument / defeater relation
    != current eligibility projection
```

## 2. Source lifecycle stream

The generic source lifecycle family should contain only events whose meaning is that the contribution's own authorized source/publisher lineage declared a lifecycle transition.

Illustrative source events are:

```text
CorrectionDeclared
SupersessionDeclared
RetractionDeclared
WithdrawalDeclared
AddendumDeclared?   // only in domains with a precise addendum contract
```

Each source event requires exact source/issuer authority for the exact contribution lineage.

The shared kernel must not accept an external evaluator's finding through this namespace merely because the evaluator can name the contribution ID.

## 3. External adjudication belongs to the argument layer

Objects such as:

```text
ExternalInvalidationFinding
ProvenanceFailureFinding
ExecutionVerificationFailure
MeasurementAdmissionFailure
IntegrityFinding
LifecycleAuthorityChallenge
LifecycleQualificationChallenge
```

belong to the scientific argument/adjudication graph defined conceptually by #701.

They may attack:

```text
a contribution's current-use eligibility
an inference from the contribution to a proposition
a source lifecycle event's qualification/authority
a successor relation
a chronology/currentness claim
```

but they do not become source lifecycle events.

This preserves the historical distinction:

```text
publisher retracted its contribution
```

versus:

```text
an independent evaluator concluded the contribution is inadmissible for use U
```

## 4. Current eligibility is the join

A future `ContributionEligibilityAssessment` should be derived from both structures:

```text
qualified source lifecycle history
        +
qualified external argument/adjudication state
        +
requested scientific use
        +
currentness / historical cutoff
        +
exact admission policy
            ->
ContributionEligibilityAssessment
```

Neither input graph alone defines universal validity.

## 5. Challenges do not mutate source history

If an external review later proves that a purported retraction was unauthorized, the source-lifecycle record is not deleted.

The Atlas retains:

```text
R1 = retraction-shaped declaration was observed
Q1 = qualification initially/externally associated with R1
C1 = later challenge to Q1 or R1 authority
```

and derives the current eligibility view from the exact qualified graph generation.

If R1 never had valid source authority, the current view may refuse to treat it as a qualified source retraction while preserving the fact that the declaration existed.

## 6. External invalidation does not mutate contribution lifecycle

Likewise, an external integrity failure does not rewrite the source lifecycle into `Retracted`.

The contribution remains historically issued under its actual source lifecycle, while the external finding may block current scientific use through an undercutting defeater or admission failure.

Therefore:

```text
source lifecycle state
    != current admissibility
```

is mandatory.

## 7. Separate generation identities

A future implementation should avoid one mixed `lifecycle_generation_id` if that would hide which structure changed.

Prefer conceptually distinct identities such as:

```text
source_lifecycle_generation_id
argument_adjudication_generation_id
eligibility_policy_id
```

with the final eligibility assessment binding all of them.

That permits a new external invalidation to change current eligibility without pretending the publisher lifecycle changed, and a publisher correction to change source lineage without silently rewriting independent defeaters.

## 8. Historical-cutoff semantics remain independent

Both structures also have their own information-availability boundary.

An Atlas view at `t0` may use only source lifecycle events and external adjudication objects admissibly known by `t0` under the selected historical-information policy.

Today's external invalidation must not leak backward any more than today's publisher retraction may leak backward.

## 9. Implementation consequence

Do not implement one enum such as:

```text
EvidenceStatus {
    Active,
    Corrected,
    Retracted,
    Invalidated,
    Challenged,
}
```

because it collapses source history, external argument, and derived eligibility into one mutable label.

The minimum architecture is instead:

```text
immutable EvidenceContribution
        |
        +--> source lifecycle graph
        |
        +--> scientific argument/adjudication graph
                        |
                        v
               use-specific eligibility projection
```

## 10. Qualification cases

A future implementation should prove at least:

1. an external evaluator cannot emit a source `RetractionDeclared` without exact source authority;
2. a publisher retraction and an external invalidation remain distinguishable even when both make current use ineligible;
3. challenging lifecycle authority changes the derived view without deleting the original declaration;
4. an external invalidation changes argument/adjudication generation without falsely changing source lifecycle generation;
5. a source correction changes source lifecycle generation without erasing independent argument objects;
6. historical views apply separate availability cutoffs to both graphs;
7. no single mixed status enum gains truth/disposition/action authority.

## 11. Non-claims

This note does not define universal publisher authority, external review authority, legal retraction semantics, an argument engine, or an eligibility algorithm.

It only ensures that the shared scientific kernel does not conflate **what the source did** with **what external science concluded about the contribution**.
