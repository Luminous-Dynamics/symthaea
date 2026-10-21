# Knowledge source catalog v0

**Status:** source metadata inventory only. It is not a corpus manifest, license grant, expert-reviewed bibliography, or held-out benchmark split.

This directory provides a stable identity for each source pointer and groups related pages/documents under a `source_family_id`. The family—not an individual question or URL—is the minimum unit for preventing source leakage in train/validation/test construction. For example, the four Digital Egypt religious-text URLs are deliberately assigned to one family.

## Current boundary

- All 13 entries are marked `split: development`.
- All rights are `unknown`, `content_access: metadata_only`, and `rights_review: pending`.
- All entries are `annotation_status: not_reviewed`.
- No full source text is copied into this catalog or implied to be approved for ingestion.
- A source being publicly discoverable does not imply that its text may be copied, indexed, redistributed, or used for model training.
- No held-out or cross-tradition generalization claim is supported by this metadata seed alone.

## Split construction

Split by `source_family_id`, not by `source_id`, URI, question ID, passage, or paraphrase. A single family must not span development, validation, test, and holdout in one evaluation version. If a source family appears under multiple IDs, the catalog must unify it before building splits.

The annotation workflow should record who reviewed the bibliographic identity, exact edition/witness, source locators, epistemic role, answerability, access rights and community authorization where relevant. Keep private/restricted source material out of public artifacts. Content extraction should be a separate permissions-gated operation; catalog validation must never crawl URLs.

## Fields

- `source_id`: stable identifier used by case records.
- `source_family_id`: leakage-control group across editions, URLs or closely related pages.
- `tradition_scope`: source grouping for evaluation, not a claim that a tradition is uniform.
- `rights_status`, `content_access`, `rights_review`: distinct fields; none should be inferred from another.
- `split`: currently development for every record.
- `annotation_status`: current human-review state, not a quality grade.

Validate using `source.schema.json` and the focused Rust harness (`--kind source`). The validator rejects unknown-rights/full-text access combinations, source-family split leakage, unsupported `verified_open_content` claims, and limited-use approval while rights remain unknown or restricted. A separate integration test requires every case source ID and URI to resolve to the catalog and the case split to match the catalog split. Schema validity is not proof that source identification, scholarly interpretation, rights, or annotations are correct.
