# PIE-002J Strict Evidence-Bound Fact Applicability Oracle

Status: independent reference semantics; synthetic fixtures only.

## Purpose

PIE-002F proves that a utility-context fact carries explicit provenance. PIE-002I proves whether each individual evidence item is applicable to one exact study scope under one explicit profile. PIE-002J freezes the next proposition: when may the **fact itself** be called applicable without laundering inapplicable or unresolved attached provenance?

V1 intentionally chooses the conservative rule.

```text
all attached evidence Applicable
    -> FactApplicable

any attached evidence Inapplicable
    -> FactInapplicable

otherwise any attached evidence Indeterminate
    -> FactIndeterminate
```

There is no `any-applicable` success path.

## Reuse of PIE-002I

The oracle does not copy the applicability algorithm. It loads and executes the qualified parent `scripts/pie-evidence-applicability-oracle.py` directly, so PIE-002J adds only closed-set subject binding and fact-level aggregation.

## Preferred boundary

The preferred function accepts:

- one exact evidence-bound numerical fact;
- exactly one `EvidenceScope` binding for every attached evidence ID;
- one exact `UtilityStudyScope`;
- one exact `ApplicabilityProfile`.

It does **not** accept caller-supplied detached applicability receipts.

For every attached evidence item, the function constructs the PIE-002I `ScopedEvidence` inside the call and recomputes applicability against the exact supplied study/profile.

## Closed evidence set

The attached evidence IDs and evidence-scope binding IDs must be exactly equal.

- missing scope binding -> error;
- extra scope binding -> error;
- duplicate attached evidence ID -> error;
- duplicate scope binding -> error.

An explicitly unknown scope dimension remains expressible through PIE-002I and normally produces `Indeterminate` when material. Omission of an entire evidence-scope binding is not an unknown claim; it is a malformed closed set.

## Canonical set-like ordering

Study dimensions, profile dimensions, evidence-scope claims, evidence items, and evidence-scope bindings are canonicalized before they are preserved in the receipt. Permuting these set-like inputs therefore cannot change the semantic receipt.

This is a semantic canonicalization rule only. This oracle does not yet define a durable byte encoder or cryptographic commitment format.

## Exact fact binding

The receipt preserves:

- fact key;
- exact numerical minimum and maximum;
- attached evidence identities and evidence classes;
- canonical evidence scopes;
- canonical study scope;
- canonical applicability profile;
- every PIE-002I per-evidence receipt;
- final fact applicability disposition.

Changing the numerical fact and recomputing necessarily produces a different receipt. A previously computed receipt cannot be supplied back into the preferred API as authority.

## Conservative provenance policy

PIE-002F evidence tuples are provenance, not a formal derivation graph. Therefore PIE-002J V1 refuses to guess which sources mathematically contributed to which bound of a range.

All attached provenance must be applicable for the fact to be `FactApplicable`.

A future derivation-manifest theorem may permit a narrower contributor set only if the numerical value-construction path itself emits and binds that manifest. A free-form caller-supplied contributor list is insufficient.

## Adversarial coverage

The self-test covers:

1. one applicable source -> FactApplicable;
2. several applicable sources -> FactApplicable;
3. applicable + indeterminate -> FactIndeterminate;
4. applicable + inapplicable -> FactInapplicable;
5. inapplicable dominates indeterminate;
6. missing scope binding fails;
7. extra scope binding fails;
8. duplicate attached evidence fails;
9. duplicate scope binding fails;
10. permutation of set-like inputs yields the same canonical receipt;
11. changing study scope forces recomputation and can change the verdict;
12. changing/weakening the profile forces recomputation and can change the verdict;
13. evidence class is preserved and cannot override scope mismatch;
14. Hypothesis remains Hypothesis even when applicable;
15. changing the numerical range changes the receipt;
16. preferred function signature contains no detached applicability receipt parameter.

## Boundaries

PIE-002J does not establish:

- truth;
- source authenticity;
- source independence;
- derivation correctness;
- uncertainty adequacy;
- trusted freshness/currentness;
- numerical feasibility;
- storage dispatch;
- thermodynamic closure;
- economics;
- execution authority.

It proves only the strict V1 aggregation of all attached provenance after fresh in-call PIE-002I resolution.

## Intended stack

```text
PIE-002F evidence-bearing fact
+ PIE-002I qualified per-evidence resolver
+ exact evidence scopes / study / profile
    -> PIE-002J strict fact applicability receipt
    -> PIE-002D numerical binding
    -> PIE-002G opaque subject-bound witness
```

Tracks #3114, #2935, #2785, #2826, #2764, #2867, #2870, #2990, #1610, #1647, and master #1604.
