# Exact ConnectomeBench2 V2 split replay audit V0.1

## Purpose

Reproduce the source partition assignment—not merely demonstrate that no operation or spatial cube appears in more than one partition.

The official upstream generator creates per-species split JSON files for three banks:
- `operation_bank`: `operation_id`
- `operation_bank_junction_controls`: `source_operation_id`
- `operation_bank_synapse_controls`: `(pre_pt_root_id, post_pt_root_id, synapse_ctr_pt_nm)`

The downstream Parquet postprocessor reconstructs these keys and looks them up in `train.json`, `val.json`, and `test.json`. It loads the files into a Python dictionary; duplicate keys across split files would silently overwrite an earlier assignment. This verifier makes that failure explicit.

## Run

```sh
cargo test --manifest-path tools/connectomics-intake-audit/Cargo.toml
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin connectome_split_replay_audit -- \
  --sidecar-jsonl metadata-sidecars.jsonl \
  --split-root /path/to/ConnectomeEnv/configs/cube_splits \
  --dataset-revision-commit-sha <exact-40-character-HF-v2-commit> \
  --output split-replay-report.json
```

Each species directory must contain `manifest.json`, `train.json`, `val.json`, and `test.json`. The tool requires upstream generator commit `5679b39aa2641d8d0afeed88b441e696f87a66df` and the dataset's exact Hub commit SHA resolved from `revision="v2"`.

## Fail-closed checks

- recompute the upstream combined sample MD5 row key (identity only, never provenance);
- reconstruct the bank-specific key using Python-compatible integer semantics and exact finite coordinate values;
- verify every observed sidecar row maps to the exact split declared by the corresponding upstream JSON;
- require every split key to occur exactly once across all three split files, rejecting both within-split duplicates and cross-split dictionary overwrites;
- compare every bank/split key count to the upstream `manifest.json` `op_counts`;
- hash the bytes of every source split file and species manifest into a deterministic report.

The exact sidecar-to-split replay is complementary to the existing dataset audit: the latter checks duplicate row IDs, operation-group split isolation, 50 µm cube split isolation, labels, routes, masks and modality cohorts. This tool checks that the row's assigned partition equals the actual upstream split key lookup.

## Trust and limitations

A PASS is conditional on the supplied files. The report hashes the split files and requires an exact upstream code commit and operator-supplied Hub V2 commit SHA, but it cannot authenticate that files copied to `--split-root` are the ones used to produce the published V2 data unless their provenance is separately authenticated and pinned. It also does not prove biological independence, label truth, population representativeness or model performance.

The upstream sample key is MD5 of `source_archive|source_archive_sample_hash`; it is only an identity convention. All artifact integrity in the audit report uses SHA-256.
