# MEL XACLE Audit

A Rust-first, model-independent audit tool for XACLE-style audio-text alignment prediction files. It complements the official XACLE baseline; it does not replace the challenge's official evaluator.

## Commands

Score one prediction file against labelled metadata:

    cargo run --manifest-path tools/mel-xacle-audit/Cargo.toml -- score METADATA.csv PREDICTIONS.csv REPORT.json

Compare a candidate with a baseline on the exact same items:

    cargo run --manifest-path tools/mel-xacle-audit/Cargo.toml -- compare METADATA.csv BASELINE.csv CANDIDATE.csv REPORT.json

Validate a blind submission against the unlabelled evaluation manifest:

    cargo run --manifest-path tools/mel-xacle-audit/Cargo.toml -- validate-submission EVALUATION.csv PREDICTIONS.csv REPORT.json

An optional bootstrap count may be supplied after the report path for score and compare modes. The default is 1,000 paired item-level bootstrap replicates.

## Integrity guarantees

- Joins by sample identifier, never row position.
- Supports the organizer's published metadata fields `wave_file_name` and `average_semantic_alignement_score`, and the baseline's preprocessed metadata dialect `wav_file_name` and `average_score`.
- Supports the baseline inference output field `wav_file_name` and the alternative `wave_file_name` prediction field.
- Rejects files containing both aliases for the same logical field, as well as duplicate, missing or extra sample IDs; no silent inner-join loss.
- Reports the exact identifier and score columns selected for each input.
- Requires finite scores in the official 0–10 scale.
- Requires exact one-to-one prediction coverage of the manifest.
- Reports SHA-256 of the exact metadata and prediction CSV bytes.
- Reports SRCC (average-rank ties), LCC, Kendall tau-b and MSE.
- Reports model_id/model_name group slices and worst-group SRCC where groups have at least 20 items.
- Compares systems using identical item-resampled bootstrap draws and reports a paired 95% SRCC-difference interval.
- Blind submission mode validates output completeness/schema/range without accessing or inferring hidden labels.

## XACLE 2027 context

The official XACLE 2027 task changes the generalization setting from seen generation systems to unseen generation systems. The organizer's public development README describes 7,500 training pairs, 3,000 validation pairs and a 3,000-pair labelled test split, with listener identities and generator model IDs/names. The separate blind evaluation dataset contains 4,000 audio-text pairs and is distributed by organizer email request.

The official baseline uses frozen M2D-CLAP audio/text encoders and a trainable projector/regressor. Its README reports SRCC 0.5648, LCC 0.6142, Kendall tau-b 0.3997 and MSE 3.1847 on its inherited labelled test split. These are not results on the blind XACLE 2027 evaluation dataset.

The organizer's 2027 development-dataset README names the metadata columns `wave_file_name` and `average_semantic_alignement_score` (including that spelling), while the official baseline's dataset loader/evaluator expects a preprocessed `wav_file_name` and `average_score` dialect. The baseline inference script emits `wav_file_name,pred_score`. The audit accepts each documented dialect explicitly, records which columns were selected, and rejects ambiguous files that contain both aliases. This avoids conflating the metadata schema with the baseline prediction-output schema.

This tool closes a specific evaluation-integrity gap in the baseline's public evaluate.py: it performs an inner merge by sample name without explicitly rejecting duplicate IDs or reporting missing/extra coverage. It also does not calculate generator-stratified performance.

## Limits

This is an evaluation and submission-integrity tool, not an audio-text model. It does not download challenge data, access organizer-only labels, authenticate dataset licences, establish an official score, or guarantee unseen-generator generalization. Generator-group metrics are only as meaningful as supplied model_id/model_name metadata and the split lineage. The default group minimum of 20 avoids promoting tiny groups as stable worst-group estimates; all group counts are still reported.

## Research protocol

1. Reproduce the official baseline on the same labelled split and settings.
2. Freeze generator IDs into grouped folds; use leave-one-generator-out validation before tuning.
3. Preserve raw per-listener ratings. Compare MOS regression with rater-aware ordinal and within-listener pairwise learning.
4. Evaluate encoder ablations (M2D-CLAP, CLAP, MERT and compatible combinations) with license/compute constraints recorded.
5. Add uncertainty and out-of-distribution checks; do not calibrate on blind evaluation data.
6. Compare candidates on identical item IDs with paired item-cluster bootstrap intervals.
7. Freeze the submission CSV, model/checkpoint digest, inference config and technical report before deadline.

## Reproducibility

    cargo +1.96.0 fmt --check --manifest-path tools/mel-xacle-audit/Cargo.toml
    cargo +1.96.0 test --locked --manifest-path tools/mel-xacle-audit/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-xacle-audit/Cargo.toml --all-targets -- -D warnings
