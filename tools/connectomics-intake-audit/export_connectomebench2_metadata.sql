-- Run from the directory containing the ConnectomeBench2 V2 dataset root.
-- The output is metadata-only JSONL; image byte columns are deliberately excluded.
COPY (
    SELECT
        combined_sample_hash,
        source_archive_sample_hash,
        source_archive,
        sample_type,
        same_neuron,
        has_single_mask,
        has_dual_mask,
        task_routing,
        false_split_correction_label,
        false_merge_identification_label,
        split,
        species,
        has_em,
        present_slots,
        metadata
    FROM read_parquet([
        'metadata/train.parquet',
        'metadata/val.parquet',
        'metadata/test.parquet'
    ])
) TO 'connectomebench2_metadata.jsonl' (FORMAT JSON, ARRAY false);
