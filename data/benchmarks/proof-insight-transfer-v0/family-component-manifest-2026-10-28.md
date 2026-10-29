# Complete PITB component manifest

Status update: 2026-10-29. The component builder, signed-receipt verifier, and analyzer-side reconciliation are implemented on the research branch. This is infrastructure qualification—not evidence that the actual 372-family inventory has been independently reviewed or that any model-transfer effect exists.

## Current data path

The deterministic builder assigns every frozen family—including singletons and same-split groups—to a content-addressed component. Confirmed and unresolved hard relationships join components; dismissed hard edges and soft relations do not. The frozen split, family assignments, relationship inventory, receipt, trust-store, and methodology are digest-bound.

The builder can consume the original inventory, signed receipt, trust-store bytes, methodology bytes, and caller-pinned trust/methodology digests. It verifies the Ed25519 receipt before marking the coverage state verified. The analyzer then independently re-verifies the same raw inputs and reconstructs the expected connected components from the signed inventory; it does not trust a self-consistent manifest alone.

At the receipt boundary, edge-level reviewer IDs must refer to reviewers in the pinned trust store and in the verified receipt signer set. Every resolved edge must be attributed to reviewers from at least two registered organizations. These runtime checks supplement the strict schemas; schemas alone cannot authenticate reviewer identities.

## Qualification states

- **Incomplete inventory:** remains unqualified and cannot carry a coverage receipt.
- **Cross-split non-dismissed hard component:** remains unqualified; unresolved hard links are conservatively included.
- **Complete inventory with no cross-split hard component but no verified receipt:** `structurally_eligible_receipt_unverified`; diagnostic only.
- **Complete inventory with verified receipt and no cross-split hard component:** the manifest may be marked `qualified_coverage_verified_no_cross_split_hard_relationships`. This qualifies the coverage graph only; it does not, by itself, authorize a benchmark result.

The run config pins the exact component-manifest and coverage-artifact digests. Power planning and test analysis use the same component identifiers and component-level resampling units. Confirmatory use still requires the registered independent-component floor, nondegenerate/informative uncertainty, the weak/strict-null false-claim gate, adequate simulated power, the exact-target semantic gate, and complete target/arm outcome accounting.

## Remaining blockers

1. Provision a trust root outside the experiment author's sole control and obtain endorsements from genuinely independent reviewers in distinct organizations.
2. Complete and independently audit the relationship inventory and retain the review logs, searches, candidate pairs, dismissed candidates, and unresolved relationships. A valid signature authenticates the attestation; it cannot prove exhaustive discovery or reviewer competence.
3. Run Rust tests and the exact-head workflow. The latest head has not reported a status check or PR-triggered workflow run; static inspection is not a pass.
4. Run matched-budget baseline/treatment comparisons only after semantic targets and eligibility are independently qualified. There are currently no measured PITB transfer outcomes, no measured utility gain, and no SOTA win claim.
