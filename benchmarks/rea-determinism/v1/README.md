# ENG-REA-005: deterministic semantic projection for firmware extraction reports

**Status:** public development known-answer contract.
**Issue:** [ENG-REA-005 #8117](https://github.com/Luminous-Dynamics/symthaea/issues/8117).
**Follows:** [ENG-REA-003 coverage integrity](https://github.com/Luminous-Dynamics/symthaea/pull/8070) and [ENG-REA-004 truthful provenance lineage](https://github.com/Luminous-Dynamics/symthaea/pull/8101).
**Research sources:** [Unblob report guide](https://unblob.org/guide/), [Unblob report model](https://github.com/onekey-sec/unblob/blob/main/python/unblob/report.py), [Unblob processing models](https://github.com/onekey-sec/unblob/blob/main/python/unblob/models.py).

## Paradigm shift

A reproducible analysis needs two different records, not one overloaded digest:

1. **Raw-observation identity:** SHA-256 of the exact producer report bytes, retained unchanged alongside the input digest, provider/version and executable identity, effective configuration, and run receipt.
2. **Semantic projection:** a deterministic representation of the report's meaningful extraction graph, suitable for comparing two runs that processed the same evidence under different temporary directories, worker/PID-derived IDs, or record ordering.

The semantic projection must not erase meaningful differences merely to make outputs match. It preserves file hashes, handlers, chunk intervals, report values, relative extracted paths, errors, and parent-child edges. It removes only explicitly recognized provider-generated IDs from ChunkReport, UnknownChunkReport, and MultiFileReport, and strips the caller-declared disposable extraction root from path fields. The raw report is not rewritten or discarded.

Unblob 26.6.4's actual serialized Task fields are `path`, `depth`, `blob_id`, and `is_multi_file`; each TaskResult contains `task`, `reports`, and `subtasks`. A descendant Task's `blob_id` normally references a producer ID (`ChunkReport` or `MultiFileReport`) on its parent task. Directory expansion also produces same-depth child tasks that inherit the directory task's `blob_id`; the projection represents these as a distinct `directory-expansion` edge only when the parent has a directory StatReport. This matters for multi-file output: two children can legitimately inherit one MultiFileReport ID through an intermediate extraction-directory Task. This prototype uses the real Task field names and models that directory-expansion case instead of flattening it into a fictional direct edge. Producer IDs are generated from worker PID plus a counter in the pinned upstream identifier implementation; path/order and known ID variation are therefore explicit test dimensions. The reports' parent references must be remapped to their semantic producer records, and unknown report types must not disappear.

## Invariants

- Exactly one depth-zero root task; task paths are unique after removing the configured root prefix.
- The root Task has empty `blob_id` at depth 0. Every non-root Task is named exactly once by a parent subtask entry, with exact path, `blob_id`, `depth`, and `is_multi_file` agreement. A depth+1 child must refer to exactly one local parent `ChunkReport` or `MultiFileReport`. A same-depth child is accepted only as directory expansion, where its `blob_id` is inherited and the parent has a directory StatReport.
- Chunk intervals use nonnegative integer, half-open ranges; size = end_offset - start_offset; when StatReport.size is present, the chunk must be in bounds.
- Only the declared temporary root is stripped. Absolute paths outside that root and relative parent traversal are rejected rather than silently rewritten.
- Report/task/subtask ordering is canonicalized, while semantic arrays such as MultiFileReport.paths remain in source order conservatively.
- Provider-generated IDs can change without changing semantic projection only when the ID-to-producer relationship is internally consistent.
- File digests, handlers, ranges, child paths, error reports, unknown/partial indications and unrecognized fields are not ignored.
- Unknown report types and malformed or ambiguous graph references fail closed.
- Semantic digest is not the raw-report digest or complete provenance receipt. Keep all three concepts distinct.

## Benchmark corpus

cases.json contains four synthetic TaskResult fixtures: a nested compressed child with an explicit unknown byte region; a relocated/permuted equivalent; a MultiFileReport with an intermediate extraction-directory task and two same-depth file tasks; and a relocated/permuted equivalent. The field shape follows Unblob 26.6.4's model, but fixtures are handcrafted and not captured provider output. Paired fixtures vary extraction root, provider IDs, task order, report order, and subtask order while preserving the semantic graph.

oracle.json is separate and lists 24 expected relations:

- **5 equivalent:** same semantics under path / generated-ID / report-order variation.
- **6 changed:** source or child hash, handler, chunk range, child path, or a removed hash report changes the semantic projection.
- **13 reject:** dangling/mismatched edges, duplicate IDs/paths/edges, unknown report type, absolute path escape, malformed range, incorrect depth, missing root, parent traversal, and missing report type.

The numbers are fixed-corpus results, not claims of real-provider performance or generalization.

## Run

Requires jq, sha256sum, and sed. The production adapter is not implemented here; the jq projection is a contract/prototype.

    sh benchmarks/rea-determinism/v1/verify.sh
    sh benchmarks/rea-determinism/v1/mutation-controls.sh

The mutation controls intentionally weaken the projection in four ways—drop SHA-256, retain provider-generated IDs, retain absolute extraction paths, or erase handler semantics. The correct suite must detect each mutant.

## Targets

| Metric | Target on the declared corpus |
|---|---:|
| Semantic invariance under incidental path/ID/order changes | 100% |
| Sensitivity to seeded semantic changes | 100% |
| Rejection of malformed/ambiguous lineage or unsupported report schema | 100% |
| Undetected weakened-normalizer mutation controls | 0 |
| Raw producer report altered by the normalizer | 0 |

For a real-provider lane, retain separate measurements for (a) input-byte identity, (b) extraction/coverage integrity, (c) lineage correctness, (d) semantic projection stability, and (e) runtime/emulator/device evidence. Do not merge them into a single score.

## Claims ceiling and next evidence gate

This is a **synthetic report contract using Unblob 26.6.4's Task and TaskResult field schema**. It does not execute Unblob or REA or parse a report emitted by a real upstream process. The schema was checked against upstream commit `1fcc7a0`; fixtures remain handcrafted. No recovered code, emulator, or hardware is used.

Next decisive gate: run a pinned Unblob release against an owned synthetic archive/compression fixture, capture the exact raw --report JSON and its digest, compare the real report against this projection, and separately verify CLI/MCP normalization once REA provides the same report through both surfaces. Only after that should a Rust-native Symthaea importer consume the contract. A semantic projection match is reproducibility evidence, not proof that extracted files are complete, safe, equivalent, or behaviorally correct.
