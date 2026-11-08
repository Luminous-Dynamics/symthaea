# CEB Review Claim-Support Closure Challenge Key V0 — 2026-11-08

**Status:** Public synthetic development answer key; not hidden ground truth, secure blinding or a sealed holdout.
**Challenge:** [Claim-Support Closure Challenge V0](CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_CHALLENGE_V0_2026-11-08.md).
**Protocol:** [Claim-Support Closure and Selective Invalidation Protocol V0](CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_AND_SELECTIVE_INVALIDATION_PROTOCOL_V0_2026-11-08.md).

## 1. Adjudication rules

Accept different graph decompositions when they preserve:
- every material dependency required by the claim;
- the event-to-dependency impact path;
- the distinction between engineering-claim validity and benchmark rerun/regrade/reuse;
- the distinction between historical truth and current decision-use admissibility;
- the open-world remainder.

A file-path match alone does not prove semantic impact. A missing path match does not prove non-impact. A retain decision needs an unaffectedness argument, not merely an absent diff.

## 2. CCL-S01 — direct source-anchor edit

**C1 — implementation behavior:** expected disposition is **revalidate**. The source anchor for parse_window changed, so the old evidence cannot be reused to claim current behavior. The event packet says the predicate changed but does not state its new semantics; do not assert that the claim is already contradicted. The minimum revalidation is an exact-source inspection and a boundary test for end < start under the existing contract, after the relevant execution gate is qualified.

**C2 — documented option contract:** expected disposition is **retain** for the narrow proposition that the unchanged docs/CLI help still describes the option as inclusive. Its closure contains the unchanged documentation and help anchors, not the changed parse_window hunk. However, any derived claim that runtime behavior conforms to the inclusive documentation must be **revalidated** because it depends on C1 and the implementation/documentation relationship.

**Key lesson:** split the compound statement. “The docs say inclusive” can remain true while “the implementation conforms to the docs” becomes stale.

## 3. CCL-S02 — shared upstream dependency

**C1 — HTTP authentication:** expected disposition is **revalidate**. The changed key provider is a direct dependency of the HTTP claim. Inspect the exact middleware/provider composition and qualify whether bootstrap-key fallback can occur for a tenant-scoped request. The change event is a candidate impact; it does not by itself prove every request is vulnerable.

**C2 — export authentication:** expected disposition is **unresolved**. The closure shares the changed provider and policy nodes, but the complete export-worker caller path is omitted. The supplied policy statement is not sufficient to prove that the caller enforces it. Do not retain merely because the export worker is in a different file or appears to deny.

**Key lesson:** shared dependency impact propagates to every dependent claim; the final disposition can differ because the caller-specific evidence and claim scope differ.

## 4. CCL-S03 — build/feature transition

**C1 — default Linux build selection:** expected disposition is **invalidate for current-tree decision use**. The manifest event changes the feature-default premise, so the old build-selection claim cannot be reused for the new tree. A distinct new claim about the current default build remains **unresolved** until a fresh feature-resolution/build-provenance record establishes which implementation is selected. Do not infer the exact active implementation from a manifest edit alone.

**C2 — both implementations exist in source:** expected disposition is **retain** within the declared source-tree scope, because both cfg-gated definitions remain present and the event changes only feature defaults. This does not establish that both are compiled or selectable in one artifact.

**Key lesson:** build selection and source structural possibility are different claims with different closures.

## 5. CCL-S04 — verifier-only update

**C1 — pass under verifier V1:** expected disposition is **revalidate**. The old pass remains a historical record of what V1 reported, but V1's false-negative defect removes its current decision-use authority for this criterion. The required revalidation action is to regrade the retained exact artifact A and identical inputs with V2. Do not rerun candidate code by default when source, artifact and environment are byte-identical and the canonical release policy permits regrade.

**C2 — source contains the authorization guard:** expected disposition is **retain** for the source-level proposition. It does not depend on V1's verdict. If the claim were that the guard passed a particular test, that would be a different claim and would require the verifier result.

**Key lesson:** evaluator invalidation does not automatically invalidate source facts or artifact lineage.

## 6. CCL-S05 — authority-root revocation

**C1 — signer approval:** split the proposition into two:
- “The historical signed approval bytes were recorded on date D” may be **retained** as a historical record if the bytes and timestamp are authentic records.
- The **current authorized/admissible** proposition (“the approval is currently authorized/admissible under trust root R”) is **invalidated** by the root revocation until a new trust evaluation establishes an alternative valid authority chain.

Do not claim the signature never existed. Do not continue treating it as current approval merely because the signed bytes are unchanged.

**C2 — artifact provenance:** expected disposition is **retain** if the provenance statement linking A to T and B has an independently valid chain and does not depend on signer S or root R. If the same revoked root actually signs the provenance statement, that dependency must be added to the closure and the answer changes.

**Key lesson:** authenticity, historical existence, provenance lineage and current authorization are distinct propositions.

## 7. CCL-S06 — metadata-only event and incomplete closure

**C1 — compatibility-suite receipt:** expected disposition is **retain** for the historical proposition that the exact recorded suite passed for the exact release R, provided the receipt and all relevant input identities are byte-identical. A display title correction does not require rerun or regrade.

**C2 — compatibility for all supported callers:** expected disposition is **unresolved** before and after the metadata event. The complete caller inventory is absent, so the support closure was already incomplete. The harmless edit neither invalidates nor completes it.

**Key lesson:** metadata-only reuse does not authorize a claim whose original support closure was never complete.

## 8. Cross-case errors

- Reusing implementation-conformance evidence after the directly referenced source anchor changes.
- Treating a source/doc statement as proof of runtime conformance.
- Missing a shared transitive dependency because downstream claims live in different files.
- Treating a manifest/feature event as a source-code change, or vice versa.
- Rerunning an artifact when only a verifier changed, where canonical release rules permit regrade.
- Preserving current approval merely because signed bytes still exist after trust-root revocation.
- Treating revoked authority as proof that historical source/artifact provenance never existed.
- Retaining an all-caller claim when the caller inventory is incomplete.
- Equating “no matching path in the diff” with proven non-impact.

## 9. Adjudicator worksheet

For each claim/event pair, record:
1. original claim and time/scope;
2. complete support-closure nodes and typed edges;
3. changed node(s) and event identity;
4. impact path or unaffectedness proof;
5. #10630 disposition;
6. #10610 run/regrade/reuse treatment;
7. historical truth versus current decision-use status;
8. unresolved remainder and claim ceiling.

If reviewers disagree on whether the closure is complete, preserve the disagreement and use unresolved until an independent adjudicator can ground the missing dependency.

## 10. Claim boundary

This key is a synthetic expected-answer guide. It reports no model score, no human reliability estimate and no real repository invalidation outcome. The challenge is public development material and permanently ineligible as a sealed holdout.
