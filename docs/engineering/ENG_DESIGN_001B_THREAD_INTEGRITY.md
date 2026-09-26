# ENG-DESIGN-001B — Design-Thread Integrity and Change-Impact Semantics

Status: Source contract
Issue: #6026
Parent: ENG-DESIGN-001A / PR #6006 / `0cc3a61871192e011299290b64514b4b341e5f1f`
Authority: repository process integrity only

## Purpose

This child makes ENG-DESIGN traceability machine-checkable without creating a second requirements database, configuration engine, evidence store, PLM/ERP, observation store, or qualification engine.

Canonical external owners retain their facts. ENG-DESIGN stores typed references and integrity relationships only.

## Reference graph

```text
NEED -> REQ / CON -> ASM / IFC -> DEC / RISK -> VER / VAL -> CFG / EVD -> CHG
```

The graph represents dependency and traceability relationships, not a universal maturity ordering.

## Identity and resolution

Claim-bearing references use the parent vocabulary exactly: `NEED`, `REQ`, `CON`, `ASM`, `IFC`, `DEC`, `RISK`, `VER`, `VAL`, `CFG`, `EVD`, `CHG`.

Rules:

1. Every identifier resolves exactly once in its applicable reference set.
2. Duplicate identifiers are invalid even when their payloads are identical.
3. Unknown identifier kinds do not silently become generic references.
4. An unresolved reference blocks every dependent claim.
5. Friendly names and prose titles are not canonical identity.
6. External-owner identities are referenced rather than copied and re-owned.

## Traceability

Every claim-bearing `REQ-*` must trace to at least one `NEED-*` or an explicit `CON-*` that independently justifies the requirement.

Every claim-bearing requirement must identify at least one `VER-*` route or an explicit externally-owned verification route.

`VAL-*` traces to intended-use / `NEED-*`; a validation result linked only to a technical requirement does not establish intended-use satisfaction.

Prerequisite/currentness dependency edges must be acyclic. Self-reference and dependency cycles are rejected. Historical and supersession links are not prerequisite edges.

## Currentness propagation

This profile does not create a new evidence-currentness theory. It consumes the semantics being formalized under `SYM-FV-007A/B` and applies them to engineering references.

```text
dependent claim current
=> relevant assumptions current
and relevant interfaces current
and bound configuration current for that claim
and required evidence current/admissible under its canonical owner
```

Stale assumptions block dependent claims as `BlockedByAssumptionDrift`.

Changed claim-relevant interfaces block dependent integration evidence as `BlockedByInterfaceDrift` until applicability or requalification is established.

Changed claim-relevant configuration generations leave prior evidence historically valid for its original subject but not silently current for the new generation.

## Evidence-plane preservation

```text
MODEL evidence != FIELD evidence
one source observation + many derived artifacts != many source observations
operational fact != engineering qualification
engineering qualification != operational fact
```

A `FIELD`-required verification route cannot be satisfied by `MODEL` evidence.

Derived artifacts retain source-witness identity and cannot inflate independent witness count.

## Change impact

A `CHG-*` record identifies the changed subject/configuration, affected typed references, affected verification/validation cases, potentially affected evidence, applicability reasoning, and required requalification or review.

Default rule:

```text
claim-relevant impact uncertain
-> retain prior evidence historically
-> current applicability unresolved
-> explicit review/requalification decision required
```

Non-ranked dispositions:

- `UnaffectedCurrent`
- `HistoricallyValidNotCurrent`
- `RequalificationRequired`
- `BlockedByAssumptionDrift`
- `BlockedByInterfaceDrift`
- `BlockedByConfigurationDrift`
- `ApplicabilityReviewRequired`

These are not maturity levels and must not become a universal readiness score.

A bounded change may preserve currentness only when its dependency analysis shows the changed element is outside the dependency cone of the specific claim.

A repair that changes a claim dependency requires requalification. A substitution with unresolved equivalence requires applicability review.

## Append-only history

Negative, stale, blocked and superseded evidence remains historically retained. A later successful occurrence adds new evidence; it does not rewrite earlier history.

This preserves the distinction between `failed then later passed` and `always passed`.

## Synthetic corpus

The companion `eng-design-001b-thread-integrity-reference-v1.json` freezes 18 known-answer cases covering complete traceability, orphan requirements, missing verification, detached validation, duplicate identity, unresolved references, dependency cycles, assumption/interface/configuration drift, bounded unaffected change, repair, substitution, evidence-plane mismatch, witness inflation, history deletion, cross-owner laundering, and authority-boundary rejection.

Canonical compact sorted-key JSON + final newline SHA-256:

`9d2aaa3637dcfef8ecf8c0b50bc202a0876b786661a84d713776f0e5dfd157c1`

## Production gate

No automated consumer should rely on these semantics until the parent exact qualifier passes, this corpus receives an independent validator, that validator receives a dedicated hosted exact-head PASS, and refinement to canonical evidence/currentness/configuration owners is explicit.

## Claim ceiling

This source contract establishes only synthetic repository design-thread integrity semantics. It establishes no external certification, real-world capability, service sufficiency, economics, procurement/resource allocation, or execution authority.
