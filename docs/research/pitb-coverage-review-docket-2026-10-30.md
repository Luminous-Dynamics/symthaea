# PITB coverage-review docket: make the holdout audit executable

**Date:** 2026-10-30  
**Status:** tooling addition; all real review obligations remain pending.

## Why this is the next bottleneck

PITB's 372-family split is reproducible, but a deterministic family split is not evidence that the holdout is independent. The current relationship inventory is absent, the signed coverage receipt is absent, and the synthetic edge fixtures cannot certify real relationships. The benchmark must not proceed to confirmatory scoring until an independent, false-negative-oriented relationship review covers the full family universe.

Adding another synthetic relationship graph or another statistical estimator would not resolve this blocker. The next useful deliverable is a review instrument that makes incomplete coverage visible and makes it difficult to overlook apparent singleton families.

## What the new tool does

`scripts/research/pitb_build_coverage_review_docket_v0.py` reads the exact frozen `splits.json` and:

1. Recomputes every assignment from the registered SHA-256 split rule.
2. Recomputes the partition digest and checks the 372-family / 260-56-56 counts.
3. Verifies that each family has a title and safe repository-relative linked artifact paths.
4. Emits two distinct, pending review obligations for each family:
   - `independent_candidate_derivation`: independently derive candidate relationships from the pinned catalogue and source artifacts.
   - `false_negative_challenge`: separately challenge for missed links, including for families with no discovered candidate.
5. Binds the docket to the exact bytes of `splits.json` and its partition digest.

Run it from the repository root:

```sh
python3 scripts/research/pitb_build_coverage_review_docket_v0.py --self-test
python3 scripts/research/pitb_build_coverage_review_docket_v0.py \
  data/benchmarks/proof-insight-transfer-v0/splits.json \
  /tmp/pitb-coverage-review-docket.json
```

For the frozen cohort, the expected docket is **372 families × 2 independent passes = 744 family-review obligations**. Each row starts with a pending status, null reviewer identity, null organization, and no claimed review-log digest. The generator does not create relationship edges, adjudicate candidates, issue signatures, or change the split.

## What constitutes a real completed review

The docket is an assignment and completeness instrument, not an audit result. Reviewers must retain independently generated candidate lists, inspected source paths, search procedures and versions, evidence locators/digests, dismissed candidates and reasons, unresolved candidates, and limitations. Hard-link candidates include exact duplicates, shared source artifacts, formal dependencies, reused lemmas, and equivalent statements. Unresolved hard links remain in the conservative component graph.

The two passes must be genuinely independent. At least one reviewer must derive candidates rather than merely review the other pass's list. The second pass must actively search for false negatives over all 372 families. Reviewer identities and organizations must be established by a trust root provisioned outside the experiment author's sole control. The signed receipt must bind the exact edge payload, frozen split, methodology bytes, and trust-store bytes.

The eventual inventory must account for all families, all candidate decisions, and all unknowns. A list of zero edges is not a completeness proof. A correctly signed statement authenticates who endorsed the inventory under the pinned trust root; it does not prove competence, exhaustiveness, or truth. Preserve the independent review logs for external inspection.

## Fail-closed exit conditions

Do not create or mark the real relationship inventory complete until both full-universe passes and external identity checks are complete. Do not run confirmatory model comparisons until:

- every family is covered by both review passes;
- every candidate has a disposition with evidence and rationale, or remains unresolved;
- all hard cross-split connected components are quarantined under the frozen rule;
- the coverage receipt verifies against an externally pinned trust store and methodology digest;
- the component builder and outcome analyzer independently reconstruct the same graph;
- the remaining eligible independent-component count and validation-derived power gate qualify.

If quarantine leaves too few independent components, the correct result is underpowered/exploratory—not a reshuffled split or a relaxed threshold.

## Validation boundary

The CI smoke test runs the generator twice, byte-compares outputs, validates every frozen family and both pending review rows, and rejects a changed split or malformed family path. That verifies deterministic docket construction against the checked-in manifest. It does **not** complete the review, verify a trust root, qualify the holdout, or show proof-transfer utility. No CI pass is implied until an exact-head run is observed.
