# ENG-REA-004: provenance lineage without invented byte mappings

**Status:** public development known-answer contract.
**Parent:** [ENG-REA-003 coverage integrity](https://github.com/Luminous-Dynamics/symthaea/pull/8070)
**Issue:** [ENG-REA-004 #8100](https://github.com/Luminous-Dynamics/symthaea/issues/8100)
**Upstream semantics:** [REA firmware-analysis documentation](https://github.com/morluto/rea/blob/main/docs/firmware-analysis.md).

## Problem

Byte coverage and byte provenance are separate claims. Even if every parent-input byte is accounted for, an extracted child has its own byte coordinate system. Compression, unpacking and multi-file extraction do not imply a linear child-offset-to-original-offset map.

REA documents that decompressed children have unknown original byte ranges and that task lineage does not establish a byte-for-byte mapping through decompression. Its normalized report separately carries derivations[].parent_file_range and chunks[].root_file_range. ENG-REA-004 turns this consumer-side distinction into a standalone, adversarial contract.

## Invariants

- $input is the only root task. Every other task has one incoming derivation edge, every edge endpoint exists, and the graph has no disconnected component or cycle.
- provider_blob_id agrees with the child task identity.
- A non-null parent range is an integer, positive-length half-open interval within the immediate parent's byte length and matches the corresponding parent chunk. Multi-file stages retain a null parent range.
- A direct selected-input chunk's root range applies the selection's original-file offset exactly.
- Child-task chunks retain task-local ranges; their root_file_range is null. Do not recursively add parent offsets through transforms.
- A null range remains unknown; it is never normalized to offset zero.
- Task, edge and chunk record order must not affect the semantic disposition. Required stage identity and nonempty provider IDs must be present; explicit unknown differs from a missing field.

## Corpus and gates

cases.json and oracle.json contain 20 known-answer cases: nested valid extraction, fabricated descendant root mappings, mismatched blob identity, missing endpoints, disconnected tasks, cycles/self-edges, duplicate task paths, malformed/out-of-bounds ranges, multi-file null ranges and selected-range offset errors.

| Benchmark | Target |
|---|---|
| Graph integrity | 100% detection of malformed/disconnected graphs in the declared corpus |
| Edge/task identity | 100% agreement of edge blob identity with child task; no missing stage or provider IDs |
| Parent range | 100% correct bounds; non-null chunk-backed ranges tie to the parent chunk |
| Transformed children | 0 fabricated root ranges where the evidence provides only task-local offsets |
| Unknown range handling | 100% null preservation; never silently default unknown to zero |
| Determinism | Same graph yields the same disposition under record permutation |

The public oracle is a regression suite, not held-out generalization evidence.

## Run

Requires jq.

    sh benchmarks/rea-lineage/v1/verify.sh
    sh benchmarks/rea-lineage/v1/mutation-controls.sh

The verifier checks the fixed oracle and repeats each assessment with task, edge and chunk arrays reversed. Mutation controls alter accepted graphs in five ways (invented root mapping, dropped lineage edge, null-to-zero, missing stage identity and empty provider identity) and require rejection.

## Claims ceiling

This benchmark checks a synthetic normalized-lineage contract only. It does not execute REA, Unblob, Binwalk, extracted firmware or hardware. Correct lineage does not prove child contents are semantically equivalent to the source, prove firmware runtime behavior, or establish a causal failure explanation.

## Next evidence gate

Connect this contract to the pinned upstream REA fixture/oracle: verify that actual provider reports preserve direct root ranges only for $input chunks, preserve null ranges for decompressed descendants, and retain task/blob identities under report-order changes. Compare normalized reports across CLI and MCP surfaces. Only after this provider lane passes should a Rust-native Symthaea importer consume the output.
