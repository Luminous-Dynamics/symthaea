# Proof Insight Transfer Benchmark v0

This directory defines a proposed benchmark format. **It currently contains no dataset and no benchmark results.**

## Files

- `insight-record.schema.json` — JSON Schema Draft 2020-12 for one provenance-linked insight record, including a separate semantic-validation section for informal-statement review, dependent-successor outcomes, and mutation controls.
- Future `manifest.json` — pinned source snapshot and inclusion/exclusion accounting.
- Future `records.jsonl` — one schema-conformant JSON object per line, only after source evidence has been inspected and rights reviewed.
- The source indexer's `family_split_plan` and each family's `split` field — deterministic pre-screen train/validation/test assignment (70/15/15). Future `splits.json` should be materialized from this assignment and hashed before suitability screening or model evaluation.

## Required discipline

1. Pin the upstream repository commit and each source file digest. A branch name like `main` is not a reproducible source identity.
2. Distinguish headline theorem, formalized supporting result, informal manuscript claim, and reasoning summary.
3. Cite a specific source path and locator for each strategy claim. Keep excerpts short; prefer locators and authored paraphrases.
4. A strategy record must state preconditions and at least one failure boundary or explicitly state that the boundary is unknown.
5. Keep candidate transfer separate from verified transfer. A transfer is verified only when the exact target statement is checked by the declared proof checker under a recorded axiom/dependency policy.
6. Use the indexer's fixed family split; never reshuffle after inspecting difficulty or results. Keep companion results or alternate proofs in their original family. Audit cross-family dependencies and pretraining exposure separately; a family split is necessary but not sufficient.
7. Add semantic statement checks: confirm the formal statement faithfully encodes the intended claim, then use dependent-successor tests where meaningful downstream declarations exist. Include mutation controls so tests are shown to detect deliberately incorrect target variants.
8. Keep failed, timed-out, out-of-scope, and not-attempted states distinct. No silent drops.
9. No HDC/embedding field is authoritative for truth. Vector indexes, if added, are derived cache data and must be ablatable.

## No claims yet

Creating this schema does not establish that the records are complete, that the schema has been validated by an independent JSON Schema implementation, that proof insights transfer, or that Symthaea outperforms any baseline. Those are explicit future acceptance tests in [the research design](../../../docs/research/proof-insight-transfer-2026.md).
