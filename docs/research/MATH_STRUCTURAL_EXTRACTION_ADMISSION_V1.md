# MATH-RET-CONVERGENCE-001D — Extraction Admission Sidecar v1

Status: frozen admission contract  
Authority: `MeasurementOnly`

## Purpose

The relocation receipt frozen by #4569 predates the byte-level wire contract in
#4570 and the independent Rust canary in #4571. Rather than rewrite that frozen
receipt contract, this tranche adds a companion admission sidecar:

`math-structural-extraction-admission-v1`

A future extraction is admissible only when it has both:

1. a valid `math-structural-extraction-receipt-v1`; and
2. a valid admission sidecar binding that receipt to the exact qualified
   predecessor and compatibility contracts.

This closes the composition gap while preserving all prior exact heads.

## Exact prerequisite gates

The sidecar binds four exact subjects and requires `conclusion == success` for
each:

| Gate | Subject | Workflow run |
| --- | --- | ---: |
| frozen representation execution | `694ef48e53296c2c6fe39a9c07927085b2f2754f` | `35459503395` |
| extraction receipt contract | `9b8f6d911a693090db4c1079f97d45f5fda57dcf` | `35463362013` |
| compatibility wire | `bdcf7203f4d077eb8fc52f5a67b8d9aefb49fd77` | `35463731810` |
| independent Rust wire canary | `556b8e71c4c596484b6cf0d4b0b26f6f03a0291d` | `35463948360` |

It also binds the exact workflow Git blobs. A run with the same display name but
different workflow bytes does not satisfy the contract.

## Exact compatibility artifacts

The admission sidecar fixes:

```text
receipt version
math-structural-extraction-receipt-v1

wire profile
math-structural-compat-wire-v1

#4570 reference vectors blob
b44eef6240e977e94744ee94e2e538576a656881

#4570 Python oracle blob
cb601ce5ec1ad0ea9fbd879a67bd1c1c9b95bdb7

#4571 Rust byte-generator blob
a9465e952eb537c26d7c352a8f0c1914a3ddae81
```

The actual extraction receipt is referenced by SHA-256. Thus the admission
sidecar cannot be detached from the exact receipt it authorizes.

## Why this is a sidecar instead of receipt v2

Changing #4569 after freezing it would make the convergence history harder to
reason about. A sidecar preserves monotonicity:

```text
receipt v1
    proves relocation equality
        +
admission v1
    proves the receipt used the qualified prerequisite subjects and frozen wire
```

Neither artifact subsumes theorem/proof authority.

## Reconstruction law

Admission requires a fresh `target_main_commit` and two explicit checks on that
then-current main:

- frozen source objects were reverified;
- the relevant environment/dependency surface was rechecked.

It also requires:

`merge_frozen_draft_branches == false`

The extraction must reconstruct the qualified artifacts onto current main. It
must not merge the historical draft stack wholesale and accidentally import
unrelated experimental history.

If current-main drift changes representation behavior, the relocation receipt
fails. The response is to diagnose the drift or open a new representation
lineage, not to move the expected values.

## Holdout firewall

Admission requires all three values to remain false:

```text
ranking_evaluated
scores_emitted
labels_exposed_to_extraction
```

The blind representation-compatibility aggregate from #4570 remains the only
holdout-derived object allowed during relocation.

## Live run evidence

This contract tranche intentionally does **not** query GitHub and does not claim
that the four prerequisite runs have succeeded. At creation time some are still
queued.

The future extraction qualification workflow must collect the actual GitHub
Actions results for the exact run IDs and construct an admission sidecar from
those observations. The semantic validator rejects `queued`, `in_progress`,
`skipped`, altered run IDs, altered subject commits, or altered workflow blobs.

Evidence references must include at least the four exact workflow observations
plus the exact extraction receipt artifact.

## Semantic negative controls

The validator rejects:

- authority escalation;
- extraction receipts not bound to `math-structural-compat-wire-v1`;
- queued/non-success prerequisite gates;
- workflow-run substitution;
- compatibility-vector/oracle substitution;
- Rust canary substitution;
- failure to reverify source objects on target main;
- failure to recheck the target environment;
- wholesale merging of frozen draft branches;
- holdout label/ranking/score leakage;
- theorem/truth authority-smuggling fields.

## Required extraction sequence

```text
all four exact prerequisite runs succeed
        ↓
select then-current main
        ↓
reverify frozen source objects + environment
        ↓
reconstruct/extract representations once
        ↓
produce receipt v1 using compat-wire-v1
        ↓
validate relocation receipt
        ↓
collect exact prerequisite run evidence
        ↓
produce + validate admission sidecar v1
        ↓
only then connect real canonical sparse S runtime backend
```

Real H remains a one-variable successor after real S qualifies. Holdout
scientific evaluation remains a later explicit evidence event.

## Nonclaims

A valid admission sidecar says only that the extraction passed the frozen
convergence prerequisites. It does not establish HDC advantage, generalization,
proof success, theorem truth, formal authority, production readiness, or
novelty.
