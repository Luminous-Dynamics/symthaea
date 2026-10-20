# Proof Insight Transfer Benchmark v0

This directory defines a proposed benchmark format. **It currently contains no dataset and no benchmark results.**

## Files

- `insight-record.schema.json` — JSON Schema Draft 2020-12 for one provenance-linked insight record, including semantic-validation evidence and a constructive-solving track for unknown witnesses, proof obligations, and answer equivalence (RPE or formal/expert checking).
- `splits.json` — frozen 372-family pre-screen assignment, pinned to OpenAI math commit `1581d4cbdd69a0faf34b2089614fe2a814316a6a`, with source fingerprints and a partition digest.
- `evaluation-protocol.md` — pre-registered baseline arms, primary endpoint, equal-budget rules, statistics, leakage controls, mutation gates, run-manifest contract, and go/no-go decisions.
- [`../../docs/research/proof-insight-transfer-sota-gap-analysis-2026-10-20.md`](../../../docs/research/proof-insight-transfer-sota-gap-analysis-2026-10-20.md) — 2026 comparison with LeanSearch v2, T², TheoremBench, VeriSoftBench and First Proof; proposes benchmark targets and the next experiments without claiming results.
- `exposure-audit.md` — records a known exploratory inspection of test family `089`; that family remains in the immutable split manifest but must be excluded from confirmatory test denominators.
- `transfer-adjudications.md` — records two static cross-family negative controls and the typed compatibility/inequality-direction gate. They are curated annotations, not measured model failures.
- `candidates/156-borsuk-nine-finite-obstruction.json` — training-family strategy annotation plus a curator-rejected transfer to family 157, with machine-readable compatibility codes; not a scored record.
- `candidates/157-hadwiger-invariant-chain.json` — unreviewed strategy annotation for separating witness construction from an invariant-comparison bridge; not a scored record.
- `pilot-report-2026-10-20.md` — pilot source audit, direct-transfer rejection, provenance notes, and next acceptance gates.
- `crates/bridges/symthaea-proof-insights/src/bin/validate_transfer_contract.rs` — fail-closed consistency checker for typed source/target obligations, inequality direction, domain/object-type compatibility, and target-side proof receipts; its rules do not establish mathematical truth.
- Future `manifest.json` — indexed source artifacts and inclusion/exclusion/unknown accounting produced by the Rust indexer.
- Future `records.jsonl` — one schema-conformant JSON object per line, only after source evidence has been inspected and rights reviewed.
- The indexer emits its own `family_split_plan` and each family's `split`; compare them to `splits.json` with `--verify-splits` before creating any benchmark examples.

## Required discipline

1. Pin the upstream repository commit and each source file digest. A branch name like `main` is not a reproducible source identity.
2. Distinguish headline theorem, formalized supporting result, informal manuscript claim, and reasoning summary.
3. Cite a specific source path and locator for each strategy claim. Keep excerpts short; prefer locators and authored paraphrases.
4. A strategy record must state preconditions and at least one failure boundary or explicitly state that the boundary is unknown.
5. Keep candidate transfer separate from verified transfer. A transfer is verified only when the exact target statement is checked by the declared proof checker under a recorded axiom/dependency policy.
6. Use the indexer's fixed family split; never reshuffle after inspecting difficulty or results. Keep companion results or alternate proofs in their original family. Audit cross-family dependencies and pretraining exposure separately; a family split is necessary but not sufficient.
7. Add semantic statement checks: confirm the formal statement faithfully encodes the intended claim, then use dependent-successor tests where meaningful downstream declarations exist. Include mutation controls so tests are shown to detect deliberately incorrect target variants.
8. Separate theorem proving from constructive problem-solving: on the latter track, the answer/witness must be discovered and all associated conclusions discharged. Track typechecked witness, proved obligations, and answer equivalence separately; compiling a plausible candidate does not count as solved.
9. Keep failed, timed-out, out-of-scope, and not-attempted states distinct. No silent drops.
10. No HDC/embedding field is authoritative for truth. Vector indexes, if added, are derived cache data and must be ablatable.
11. The frozen split manifest was generated from the pinned catalog with an independent implementation of SHA-256 ranking. It has not yet been matched against a compiled Rust indexer output; run the crate and `--verify-splits` before relying on it as an engineering-verified artifact.
12. Follow `evaluation-protocol.md`: select settings and the strongest baseline among A–E and I on validation families only, freeze the primary contrast and compute budget before the test run, preserve all exclusions, and report paired family-level outcomes with uncertainty. Do not make SOTA claims from this pre-screen split alone.

## No claims yet

Creating this schema or protocol does not establish that the records are complete, that the schema has been validated by an independent JSON Schema implementation, that proof insights transfer, or that Symthaea outperforms any baseline. Those require the stated validation gates and actual measured experiments; see also [the research design](../../../docs/research/proof-insight-transfer-2026.md).
