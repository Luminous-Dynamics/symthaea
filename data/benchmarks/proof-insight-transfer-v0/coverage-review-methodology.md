# PITB component-coverage review methodology v1

**Status:** review procedure specification. This document is not evidence that a completed review has occurred.

## Purpose and scope

The signed coverage receipt is an attestation that the relationship inventory for the exact frozen PITB family partition was independently derived and reviewed using this procedure. It is not a claim that every proposed relationship is true, and it does not establish that a benchmark row is semantically valid. Those require separate evidence and gates.

The attestation is bound to the exact frozen source commit, the 372-family split partition, the normalized relationship-edge payload, this document's exact UTF-8 bytes, and an exact reviewer trust-store digest.

## Required independent passes

Two reviewers from distinct, pre-registered organizations must sign the same payload. At least one reviewer must independently derive candidate relationships rather than only read the first reviewer's list. At least one reviewer must perform a false-negative-oriented challenge review over the full family set. They must not share a single upstream inventory-generation result as their only source of evidence.

Each review pass must cover all 372 frozen families, including singleton families and families with no known edges. The coverage review must include at minimum:

1. Exact duplicate and equivalent formal statements, after both byte-level and normalized-AST comparison where available.
2. Shared source artifacts, generated declarations, imports, and declaration provenance.
3. Formal dependency links, including transitive dependencies relevant to proof reuse.
4. Reused lemmas, copied proof fragments, shared helper definitions, and theorem aliases.
5. Known source/catalogue cross-links and paths that can connect ostensibly separate families.
6. Candidate pairs that were inspected and dismissed, with evidence and a reason; dismissed edges must not erase an independent hard-link warning without recorded review evidence.
7. Unresolved candidates. An unresolved hard relationship remains included for conservative component construction.

The review log must list each family, source artifacts examined, search procedure/version, discovered candidate edges, dismissed candidates and reasons, unresolved candidates, and limitations. The logged output and search artifacts should be retained independently of the inventory.

## Edge contract

Every inventory edge uses a registered relation and a status. Evidence is content-addressed. A resolved assessment requires at least two reviewer IDs. The receipt verifier independently checks schema-level semantics, family membership, relationship-pair uniqueness, and the normalized edge digest.

Relationship kinds \`exact_duplicate\`, \`shared_source_artifact\`, \`formal_dependency\`, \`reused_lemma\`, and \`equivalent_statement\` are hard links. \`conceptual_similarity\` and \`citation_only\` are not hard links by themselves. Unresolved hard links conservatively join components. Dismissed hard links do not join components, but their evidence and reviewer decision remain in the inventory.

The canonical signed edge digest is SHA-256 over compact JSON serialization of edges sorted by \`edge_id\`, with each edge's \`reviewer_ids\` sorted lexicographically. This makes formatting, input row ordering, and reviewer-list ordering immaterial while preserving every signed semantic field. A change to any edge field changes this digest.

## Trust root and signature message

The reviewer trust store must be provisioned and versioned outside the outcome-producing experiment. Its raw-byte SHA-256 is pinned in the preregistered run config and signed into the receipt. Do not treat a trust-store file created and modified by the same unreviewed PR author as independent identity verification.

Each reviewer signs the following exact UTF-8 message, with a newline after every field:

\`PITB_COMPONENT_COVERAGE_RECEIPT_V1\`
\`pitb.component-coverage-receipt.v1\`
\`1581d4cbdd69a0faf34b2089614fe2a814316a6a\`
\`{source_partition_digest_sha256}\`
\`{relationship_inventory_edges_sha256}\`
\`{review_methodology_sha256}\`
\`{trust_store_sha256}\`
\`complete_relationship_inventory\`

The trust-store digest is the SHA-256 of its exact raw bytes. The review-methodology digest is the SHA-256 of this exact file's raw bytes. The receipt's own SHA-256 is stored in the inventory; the signed message does **not** include the receipt's own digest, avoiding a self-referential hash cycle.

## Fail-closed conditions

Reject the receipt on any signature or digest mismatch; changed methodology or trust-store bytes; an unknown or mismatched reviewer/key; duplicate signer IDs or key IDs; fewer than two valid signers; signers from fewer than two registered organizations; an incomplete inventory; a wrong upstream commit or split partition; malformed edges; or a family reference outside the frozen partition.

A valid signature set authenticates endorsements under the pinned trust store; it cannot prove reviewer competence, honesty, or actual independence. Those are governance and audit questions. Retain the review logs and have an external auditor inspect them. The receipt verifier's output is necessary but not sufficient for confirmatory analysis: the component manifest and analyzer must separately validate component membership, cross-split hard links, statistical qualification, and every other preregistered gate.
