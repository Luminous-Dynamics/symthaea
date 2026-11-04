# Connectome world correspondence verifier

Rust-only semantic checker for the V0.1 evidence-qualified correspondence receipt.

## Inputs

```sh
cargo test --locked
cargo run --locked -- receipt.json source-world.json target-world.json
```

The inputs are:
1. the correspondence receipt;
2. the exact source-world artifact manifest;
3. the exact target-world artifact manifest.

Candidate and target inventories are pinned separately. The receipt must bind each sidecar's exact raw bytes by SHA-256. The sidecars define the object inventory in scope for the correspondence exercise.

## What V0.1 independently checks

- exact JSON object keys at the report, world, manifest, mapping, evidence, query and registration levels;
- pinned receipt-schema, world-manifest-schema, correspondence-protocol and evidence-policy digests;
- exact source/target manifest raw-byte hashes and embedded world identity;
- unique object IDs in each sidecar inventory;
- relation-specific object-type constraints (artifact, segment, neuron, synapse);
- source object coverage in the receipt and target membership for every mapped target;
- cardinality recomputed from source/target list sizes, including the zero-target UNMAPPED case;
- explicit alternative-set grouping for ambiguous correspondences;
- no source object duplicated across rows unless the rows form one explicit alternative set;
- disjoint target object sets among mutually exclusive alternatives;
- unique mapping and evidence IDs;
- evidence field shape and digest syntax;
- affine 4×4 transform shape, column-vector convention's final row, and non-singular 3×3 linear component;
- query output digest consistency for SAME / DIFFERENT / NOT_ESTIMABLE declarations;
- all mapping, cardinality, query and unmapped-source summary counters.

## Trust boundary

The output is `COUNTERS_RECOMPUTED`, not a biological correspondence certificate. It returns:
- `claimed_disposition`;
- `recomputed_disposition = NOT_ESTIMABLE`;
- `support_status_claim_counts`;
- `trusted_promotion = false`.

V0.1 does **not** authenticate evidence artifact bytes referenced by mapping rows, edit-transaction history, lineage receipts, expert adjudication, physical registration anchors/residuals, signer identity, trust roots or revocation. A field containing `CLAIMED_VERIFIED`, `EXPERT_ADJUDICATED` or `INDEPENDENTLY_VERIFIED` is only a producer claim until these external trust adapters exist. A structurally valid matrix is not a verified registration.

The affine transform is interpreted using column vectors in nanometres: `p_target = T_source_to_target × p_source`. The bottom row must be `[0, 0, 0, 1]` within 1e-9, and the upper-left 3×3 determinant must be nonzero within 1e-12. Actual physical residuals are not recalculated because V0.1 does not yet consume anchor coordinate artifacts.

## Exit status

- `0`: structural checks and counter reconciliation completed;
- `2`: the receipt or sidecars were rejected;
- `64`: CLI invocation error;
- `66`: input file could not be read.

## Authored tests and execution status

Seventeen Rust unit tests are authored. They cover valid split/merge/ambiguity, non-promotion of producer status claims, manifest byte mutation, cardinality mismatch, unknown objects, query-output contradictions, non-affine and singular transforms, duplicate mappings, omitted source inventory objects, alternative-set consistency/overlap, relation/object-type mismatch, and invalid NOT_ESTIMABLE query outputs.

They have **not** been compiled or executed in this authoring environment because `cargo`, `rustc` and `rustfmt` are unavailable. No compile, test, formatting, Clippy or CI pass is claimed.
