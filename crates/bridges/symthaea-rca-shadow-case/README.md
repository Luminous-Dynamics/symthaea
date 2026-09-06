# Symthaea RCA Shadow Evidence Case

RCA-003a assembles a **causally inert evidence case**. It does not decide truth or create a belief.

The input boundary is:

```text
one exact proposition digest
        +
validated runtime evidence candidates
        +
one validated current-runtime relevance context
        +
validated declared proposition relations
        ↓
ShadowEvidenceCaseV1
```

## Why case assembly is separate from disposition

Before Symthaea can say that a proposition is supported, contested, defeated, or underdetermined, it must first prove that the inputs being compared actually belong together.

RCA-003a therefore answers only structural/diagnostic questions:

- Does every relation reference exactly one candidate in the case?
- Does every relation target the exact case proposition?
- Does every candidate reconstruct to its own committed observation root?
- Do multiple candidate claims share the same observation root or have disjoint roots?
- Which candidates remain relevant under one exact runtime-relevance context?
- What is the shape of the caller-declared support/opposition relations?
- Is there a currently relevant declared defeater?

It does **not** answer:

```text
Is the proposition true?
Is it supported strongly enough to believe?
What posterior probability should it receive?
Should it enter canonical epistemic state?
Should it enter the workspace?
Should Symthaea act?
Should an architecture be promoted?
```

## One proposition + one relevance context

RCA-003a deliberately does not accept precomputed relevance reports.

Instead it accepts one `ValidatedCurrentRuntimeRelevanceContextV1` and recomputes relevance internally for every candidate.

This prevents a case from mixing:

```text
candidate A evaluated at cycle 100 with lag 0
candidate B evaluated at cycle 120 with lag 20
candidate C evaluated under another execution lineage
```

while presenting the resulting topology as though all items were current under one policy.

The case binds:

```text
case_scope_digest = H(
    RCA-003a contract,
    exact proposition digest,
    exact relevance-context commitment
)
```

Changing either the proposition or relevance context therefore changes case scope identity.

## Lineage is reconstructed, not supplied

Callers do not provide an evidence-lineage graph.

For every candidate, RCA-003a calls `lineage_fragment()` and reconstructs a closed graph itself:

```text
ObservationEventRoot
        ↓
FieldCandidate
```

Shared observation roots are deduplicated. The graph is validated before pairwise independence is assessed.

This preserves the RCA-001.1 theorem:

```text
multiple candidate objects != multiple independent observations
```

Two fields from the same frozen cycle resolve to `SameRoot`, not `Independent`.

## Declared relations are not verified truth

`EvidenceRelationV1` relation labels are caller-supplied, structurally validated declarations. RCA-003a therefore exposes `declared_relation` and `declared_relation_topology`, not an epistemic truth disposition.

`Corroborates` still does not imply independence. Independence comes only from the reconstructed lineage graph.

Relation strength is preserved verbatim for audit. It is never summed, averaged, normalized, or converted to a probability in RCA-003a.

## Issued report boundary

`ShadowEvidenceCaseV1` has private fields and no `Deserialize` implementation.

The report may serialize for audit, but persisted bytes do not recreate a trusted case. A consumer must reload/revalidate the candidates, relation declarations, and relevance context and rerun case assembly.

## Authority boundary

```text
case assembly
    != evidence admission
    != truth disposition
    != canonical epistemic state
    != workspace/GWT admission
    != action authority
    != self-improvement promotion
```

A later RCA-003b may define an experimental **shadow disposition policy** over a qualified case, but only after its aggregation/falsification rules are preregistered and separately qualified.