# Proof Observability V1

Status: architecture/review metadata only  
Tracking issue: #5778  
Parent review contract: Proof Review Capsule V1 / #5773

## Purpose

A proof can check and still be a poor engineering argument. Proof Observability V1 records whether the accepted result materially depends on the premises, implementation regions, dependency closure, and coverage assumptions that reviewers expect.

This layer does not create new theorem truth. It makes proof structure visible and makes semantic weakening harder to hide.

## Governing rule

```text
proof accepted
!= proof non-vacuous
!= intended premises used
!= implementation fully constrained
!= exact coverage
```

The default review surface should answer:

```text
proof checked       yes/no
vacuity              Clear/Detected/Unknown
coverage             Exact/Partial/Bounded/Unknown
used assumptions     N
unused critical      N
unconstrained code   N regions
semantic delta       none/review-required
```

## Common observability report

Every report binds:

- stable report ID;
- parent proof-capsule ID;
- exact proof-goal identity;
- exact subject identity;
- checker/result identity;
- assumptions in scope;
- assumptions actually used when the tool can establish this;
- critical premises unexpectedly unused;
- axiom/trust-root dependency closure when available;
- theorem/lemma dependency closure;
- represented implementation/model regions;
- unconstrained or excluded regions;
- vacuity status and retained evidence;
- coverage class;
- any explicit bound for bounded evidence;
- previous observability identity and semantic delta;
- whether semantic review is required.

Missing observability is represented as `Unknown`; it must never be silently upgraded to `Clear` or `Exact`.

## Coverage classes

### Exact

The retained proof/refinement subject covers the exact declared subject under the stated model and assumptions. `Exact` is invalid when known required implementation regions are unconstrained or excluded.

### Partial

The proof is valid for a deliberate strict subset or projection of the intended implementation/model. The uncovered boundary must be named.

### Bounded

The result is limited by an explicit finite/state/trace/configuration bound. The bound is part of the report identity.

### Unknown

Coverage has not yet been characterized strongly enough to classify as Exact, Partial, or Bounded.

```text
Unknown != failure
Unknown != Exact
```

## Vacuity

Vacuity is separate from theorem acceptance.

A report uses:

- `Clear` only when the admitted analysis has checked the retained contradiction/vacuity obligations;
- `Detected` when contradictory assumptions or another accepted vacuity witness is found;
- `Unknown` when the lane does not yet provide a sufficient analysis.

A `Detected` result cannot be `reviewed-current` for the original unconditional human claim.

## Critical-premise use

Security/safety assumptions may be tagged critical. If a critical premise was previously used and becomes unused, that is a semantic review event even if the theorem still checks.

This catches changes such as:

```text
old proof depends on anti-replay invariant
new proof no longer depends on anti-replay invariant
proof still passes
```

The correct response is not automatic rejection; it is explicit review of whether the premise became redundant for a good reason or the theorem weakened.

## Implementation coverage

For refinement and implementation-proof evidence, reports should distinguish:

```text
represented regions
excluded/unsupported regions
unconstrained regions
```

A source function disappearing from the represented subject must invalidate `Exact` coverage until reviewed.

## Tool adapters

Adapters may provide different observability strengths.

### Lean

Use kernel-checked theorem identity plus available axiom/declaration dependency information. `#print axioms` remains useful but does not by itself establish complete semantic dependency coverage.

### Aeneas / Rust-to-Lean

Bind the exact translated Rust subject. Unsupported/excluded constructs are first-class coverage gaps, not footnotes.

### Verus / SMT-backed deductive lanes

Retain available dependency, assumption, warning, and certificate information. When the producer cannot establish used-vs-in-scope premises, report `Unknown` rather than fabricate precision.

### TLA+ / Alloy / bounded model checking

Bind exact properties, invariants, configuration, fairness and bounds. Constraints that define the model are assumptions; checked properties are not to be relabeled as assumptions after the fact.

## Semantic-delta rule

Human re-review is required when any of these changes materially:

- proof goal;
- critical premise use;
- axiom/trust-root closure;
- proof/theorem dependency closure;
- represented implementation/model regions;
- unconstrained regions;
- vacuity state;
- coverage class or bound.

The machine may re-check far more than the human rereads.

## Negative controls

The validator must reject at least:

1. `vacuity=Detected` while the report remains current for the unconditional claim;
2. `coverage=Exact` with a known unconstrained required region;
3. an unused critical premise without semantic review;
4. `Exact -> Partial` without semantic review;
5. `Bounded` without an explicit bound;
6. observability dependency identity changes while the report remains current;
7. missing data relabeled from `Unknown` to a stronger status without evidence.

## Nonclaims

```text
proof observability != theorem truth
no detected vacuity != complete specification
used-by-proof != semantically intended
coverage report != source refinement
Exact coverage != compiler/native-binary correctness
Unknown observability != evidence of defect
```

## V1 implementation scope

V1 defines the common contract and validates synthetic reports. It does not yet claim that every formal tool has a production adapter. Tool adapters become qualified independently and must identify unsupported observability fields explicitly.
