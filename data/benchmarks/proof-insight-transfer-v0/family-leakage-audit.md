# Cross-family leakage audit for PITB v0

## Why this exists

The frozen split manifest is deterministic at the related-result-family level, but the benchmark documentation explicitly warns that this does not remove cross-family mathematical dependencies or pretrained-model exposure. Randomly separating family IDs is therefore necessary but not sufficient for a credible transfer claim.

This secondary audit **never changes or regenerates the frozen split**. It records reviewed relationships, forms connected components using potentially contaminating relations, and quarantines a component if its members span train, validation, or test. Unknown relations are treated conservatively; they are not voted away. Every report retains original and known-quarantined counts.

## Relationship semantics

Hard-leakage kinds are **exact_duplicate**, **shared_source_artifact**, **formal_dependency**, **reused_lemma**, and **equivalent_statement**. A confirmed or unresolved edge of any of these kinds connects its endpoints for quarantine. A dismissed edge does not. **conceptual_similarity** and **citation_only** are recorded but do not by themselves invalidate a holdout: a benchmark about strategy transfer must allow mathematical ideas to be related without treating all similarity as data leakage.

Resolved relationships require at least two distinct reviewer IDs. This is a consistency requirement, not identity authentication. Every edge includes an evidence digest, but the analyzer does not retrieve or authenticate the corresponding bytes.

## Fail-closed reporting

The analyzer re-derives every family's assignment from the registered seed and algorithm, recomputes the partition digest, checks declared counts, and binds the edge inventory to that exact digest. An assignment mutation or a stale edge inventory is rejected.

The report has three possible statuses:

- **not_certified_incomplete_inventory**: no clean-holdout claim is possible, even when no known edges exist.
- **cross_split_hard_relationships_require_quarantine**: one or more active hard-leakage components span splits; all members of each component are listed for known exclusions.
- **no_cross_split_hard_relationships_in_declared_inventory**: no active hard edge spans splits in the inventory declared complete. This is not proof that the relationship search was exhaustive; reviewers must separately inspect the coverage receipt and evidence bytes.

Known-quarantine counts are *not* a corrected score denominator until the inventory is independently complete. Do not modify the split manifest or drop families silently after seeing outcomes. Preserve both the original and adjusted counts and pre-register any confirmatory exclusion rule.

## Run

From the Symthaea repository root:

    cargo run --locked --bin analyze_family_leakage --       data/benchmarks/proof-insight-transfer-v0/splits.json       data/benchmarks/proof-insight-transfer-v0/family-leakage-edges.json       /tmp/family-leakage-report.json

The checked-in synthetic edge fixture is deliberately invented CI test data. Its labels are fabricated to exercise confirmed, unresolved, and non-blocking control paths; they are **not** factual assertions about the named result families. The real inventory **family-leakage-edges.json** must be created only from independently reviewed evidence. It does not exist yet, so PITB currently has no cross-family-complete leakage audit and must not report a clean holdout.

## Stop conditions and next benchmark gates

Before confirmatory use, require a complete reviewed edge inventory, independent byte-level verification of every cited receipt, split-isolation checks for source/target insight records, and a preserved exposure/quarantine ledger. If too few independent components remain after known exclusions, report the study as underpowered rather than reshuffling families or relaxing the gate.

## Automated candidate triage (not adjudication)

Run:

    cargo run --locked --bin discover_family_leakage_candidates -- \
      data/benchmarks/proof-insight-transfer-v0/splits.json \
      /tmp/family-leakage-candidate-report.json

This generator reads only family titles in the frozen manifest. It strips HTML markup/entities, tokenizes Unicode words, removes a fixed stopword list, computes title-token Jaccard overlap plus rarity-weighted overlap, and records candidate-level shared tokens and a digest of the compared title evidence. The threshold configuration is part of the emitted report, so the screen can be reproduced.

All rows are labelled `title_similarity_only` and `unreviewed`, regardless of score. It is a review-queue heuristic, not a semantic dependency detector, leakage finding, or benchmark result. Do not promote candidates to hard edges without inspecting source artifacts and recording independent adjudication. The generator deliberately does not emit the relationship inventory consumed by the quarantine analyzer; that conversion is a separate, reviewed decision.
## Artifact-path candidate triage

Run:

    cargo run --locked --bin discover_artifact_path_candidates -- \
      data/benchmarks/proof-insight-transfer-v0/splits.json \
      /tmp/family-artifact-path-candidates.json

A second generator, `discover_artifact_path_candidates`, screens the family manifest's `linked_paths` separately from the title-only candidate generator. Its report records the paths whose token sets contain rare shared terms, the split labels, overlap metrics and an evidence digest. The report schema is `family-artifact-path-candidate-discovery.schema.json`.

The artifact-path channel is deliberately independent: a family's linked manuscripts are source/provenance context and may share broad mathematical vocabulary without establishing any dependency. Each result is labelled `artifact_path_similarity_only` and `unreviewed`. This channel may improve recall but can sharply increase review load, so measure reviewed precision/recall before merging it with title-based rankings. The frozen split and hard-edge inventory remain untouched.