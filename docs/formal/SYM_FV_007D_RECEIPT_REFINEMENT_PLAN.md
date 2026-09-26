# SYM-FV-007D Receipt Refinement Plan

Tracking issue: #5992.

This branch is intentionally plan-only while SYM-FV-007B remains unqualified. It must not be cited as formal evidence.

## Problem

`SYM-FV-007B` proves local freshness/admission over an abstract `EvidenceRef` carrying a receipt, observed generation, live generation, and canonical qualification result. It does not prove that a serialized receipt/validator implementation constructs those abstract fields correctly.

## Required refinement

A later implementation must establish an explicit relation between the machine-readable receipt representation and the Lean model, including:

- canonical qualification result mapping (`Pass | Fail | Blocked | EnvironmentFailure`);
- exact subject/receipt identity;
- generation identity and invalidation semantics;
- no tool-specific checker outcome substituting for canonical result;
- malformed/missing fields fail closed;
- stale generation cannot be silently refreshed;
- unknown future result values cannot default to `Pass`;
- receipt decoding cannot widen evidence class, subject, scope, assumptions, or authority.

## Staging

- **007D-A:** serialization/schema relation and hostile malformed-value corpus.
- **007D-B:** pure validator/admission kernel refinement to the Lean model.
- **007D-C:** cross-repository receipt consumption with exact subject/export binding.

## Nonclaims

This plan is not a proof, not validator refinement, not runtime authorization, and not evidence that SYM-FV-007B has qualified.
