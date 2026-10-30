# HARDSEC-009 — Source-family split and leakage audit

**Status:** research prototype; 21 Rust unit tests authored. The focused GitHub Actions run is the acceptance oracle; no benchmark corpus or detector has been evaluated by this tool.

## Purpose

A reported detector score is not meaningful if sibling netlists, identical normalized graph structures, host families or insertion sources cross the train/validation/test boundary. HARDSEC-009 defines the provenance-bearing split manifest first, then validates that manifest before model training, threshold selection or test-set access.

The implementation is `examples/hardsec_family_split_audit.rs`, registered behind Cargo feature `hardsec-family-audit`.

## Manifest contract

The JSON manifest schema is `hardsec-family-split-v1`. Required top-level fields:

- `manifest_id`, `dataset_id`, and `split_protocol`;
- `normalization.normalizer_id` and `normalizer_version`;
- `normalization.cell_library_id` and `synthesis_toolchain_id`;
- `records[]`, one entry per evaluated design sample.

Each record must bind:
- unique `sample_id` and `design_id`;
- `host_family_id` and `insertion_source_id` (clean records use the explicit sentinel `none`);
- immutable `source_uri` and `source_revision`;
- `source_sha256`, `netlist_sha256`, `normalized_structure_sha256`;
- binary `label` (`clean` or `trojan`) and `split` (`train`, `validation`, `test`).

Unknown labels, empty lineage fields, unsupported split names, malformed SHA-256 values, duplicate sample IDs or duplicate design IDs fail closed.

## Confirmatory protocols

### `leave_one_base_family_out`

Every `host_family_id` must be assigned to exactly one of train, validation or test. Sibling host designs/variants cannot appear in more than one split. All three splits must be populated and contain both clean and Trojan records.

### `leave_one_insertion_source_out`

Every Trojan `insertion_source_id` must belong to exactly one split. Clean controls use `insertion_source_id = "none"` and do not create an artificial shared Trojan insertion source.

### `diagnostic_random`

Permits host-family and insertion-source overlap, but exact source-artifact, netlist and normalized-structure digests must still remain split-local. The receipt marks this protocol `confirmatory_generalization_protocol = false`; it must not be reported as evidence of family-level transfer.

## Leakage rules

All protocols reject:
- source-artifact SHA-256 reuse across splits;
- exact netlist SHA-256 reuse across splits;
- normalized-structure SHA-256 reuse across splits;
- content aliases with conflicting labels, even within one split.

Same-split aliases with identical labels are counted, not silently discarded. This lets the benchmark owner see duplicated weight rather than inflate the apparent sample count invisibly.

The normalized-structure digest is meaningful only under the exact recorded normalizer, version, cell library and synthesis toolchain. Different canonicalization or mapping settings require a new identity and cannot be treated as the same feature space.

## Receipt and claim boundary

The tool hashes the raw manifest bytes and emits a deterministic, unsigned JSON receipt outside the manifest directory. It records split counts, family/insertion-source assignments, provenance records and duplicate same-split aliases.

Its claim scope is strictly `manifest_lineage_and_split_group_integrity_only`. It does **not**:
- inspect netlist contents or prove the normalizer is correct;
- verify that labels are true or that a Trojan was inserted;
- establish trigger reachability, payload activation or functional behavior;
- authenticate source URIs/revisions or verify the referenced file bytes;
- train or score a detector, or certify a design.

A later stage should add a byte-root verifier for source and netlist URIs and independently validate normalizer output. The split auditor must not pretend those operations are already done.

## Run

From a complete root workspace:

```bash
cargo test --example hardsec_family_split_audit --features hardsec-family-audit
cargo run --example hardsec_family_split_audit --features hardsec-family-audit -- \
  --manifest tests/fixtures/hardsec/hardsec-family-split-valid.json \
  --receipt /path/outside/fixture/split-audit.json
```

The current root Cargo workspace references a missing globbed member manifest, so the focused CI workflow uses a nested workspace at `tools/hardsec-family-audit-harness/Cargo.toml` to compile the exact same checked-in example source without modifying or weakening production workspace membership.

## Initial known-answer fixture

`tests/fixtures/hardsec/hardsec-family-split-valid.json` contains six fixture records:
- three host families;
- one Trojan and one clean control per family;
- train, validation and test assigned one family each;
- three distinct insertion-source IDs.

It is a synthetic contract fixture, not research data and not a detector benchmark.

## Why the confirmatory split matters

Xiao et al. (2026) report Random-Forest gate F1 of 0.914 on pooled splits and 0.460 on leave-one-base-family-out evaluation; average precision falls from 0.978 to 0.577. Their study covers 49,124 gates, 16 netlists and five host families, and cautions that the family count is too small for population-level claims. These are not directly comparable to a new result, but they establish why random splitting must remain diagnostic and why per-family/worst-family uncertainty must be reported.

Reference: https://arxiv.org/abs/2609.07199

## Proposed next benchmark gate

Before training:
1. freeze the source inventory and lineage graph;
2. normalize each source under a recorded normalizer/library/toolchain identity;
3. run this audit and publish the immutable receipt;
4. preregister the LOFO/LOISO split manifest and tuning budget;
5. compare Random Forest, XGBoost, simple SCOAP/rarity heuristics and graph baselines on identical splits;
6. only then evaluate Symthaea candidates.

A candidate should target at least +5 absolute points of LOFO macro F1 over the strongest matched baseline at the same validation-selected false-positive rate, while preserving worst-family performance and reporting family-level uncertainty. This is a proposed gate, not an achieved result or certification standard.
