// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Network-free audit CLI for `symthaea-matbench-folds`.
//!
//! Usage:
//! `matbench-fold-audit <matbench_expt_gap.json.gz> [fold]`
//!
//! When `fold` is omitted, all five published-procedure folds are audited.

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::process::ExitCode;
use symthaea_matbench_folds::{
    build_leakage_clean_test_fold_from_official_bytes, FoldIndex,
    MATBENCH_EXPT_GAP_FOLD_MANIFEST_SHA256, MATBENCH_V01_VALIDATION_BLOB_SHA1,
    MATBENCH_V01_VALIDATION_COMMIT,
};

fn usage(program: &str) -> String {
    format!("usage: {program} <matbench_expt_gap.json.gz> [fold: 0..4]")
}

fn audit_fold(bytes: &[u8], fold_value: u8) -> Result<Value, String> {
    let fold = FoldIndex::new(fold_value).map_err(|error| error.to_string())?;
    let audit = build_leakage_clean_test_fold_from_official_bytes(bytes, fold)
        .map_err(|error| error.to_string())?;

    let exclusions: Vec<Value> = audit
        .excluded_training_overlap
        .iter()
        .map(|exclusion| {
            json!({
                "row_position": exclusion.row_position,
                "candidate_id": exclusion.candidate_id,
                "symthaea_training_labels": exclusion.symthaea_training_labels,
            })
        })
        .collect();

    Ok(json!({
        "capability_classification": "MEASUREMENT-ONLY LEAKAGE QUALIFICATION -- not an official Matbench leaderboard score, material certification, novelty claim, experiment authorization, or deployment decision.",
        "fold": audit.fold.value(),
        "canonical_test_count": audit.canonical_test_count,
        "retained_test_count": audit.retained_test_count,
        "excluded_training_overlap_count": exclusions.len(),
        "excluded_training_overlap": exclusions,
        "source_artifact_sha256": audit.source_artifact_sha256,
        "upstream_validation_commit": MATBENCH_V01_VALIDATION_COMMIT,
        "upstream_validation_blob_sha1": MATBENCH_V01_VALIDATION_BLOB_SHA1,
        "fold_manifest_sha256": MATBENCH_EXPT_GAP_FOLD_MANIFEST_SHA256,
        "symthaea_training_table_sha256": audit.symthaea_training_table_sha256,
        "symthaea_training_entry_count": audit.symthaea_training_entry_count,
        "overlap_mask_sha256": audit.overlap_mask_sha256,
        "truth_slice_sha256": audit.truth_slice_sha256,
        "qualification_sha256": audit.qualification_sha256,
        "truth_slice": {
            "dataset_id": audit.truth.provenance.slice.dataset_id,
            "split_id": audit.truth.provenance.slice.split_id,
            "source_uri": audit.truth.provenance.source_uri,
            "content_digest": audit.truth.provenance.content_digest,
            "license": audit.truth.provenance.license,
        },
        "disclosures": {
            "fold_derivation": audit.fold_derivation_disclosure,
            "index_identity": audit.index_identity_disclosure,
            "residual_leakage": audit.residual_leakage_disclosure,
        },
    }))
}

fn run() -> Result<(), String> {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "matbench-fold-audit".to_owned());
    let artifact_path = args.next().ok_or_else(|| usage(&program))?;
    let requested_fold = match args.next() {
        Some(raw) => Some(
            raw.parse::<u8>()
                .map_err(|_| format!("invalid fold {raw:?}; expected integer 0..4"))?,
        ),
        None => None,
    };
    if args.next().is_some() {
        return Err(usage(&program));
    }

    let bytes = fs::read(&artifact_path)
        .map_err(|error| format!("failed to read {artifact_path:?}: {error}"))?;

    let fold_values: Vec<u8> = match requested_fold {
        Some(value) => vec![value],
        None => (0..5).collect(),
    };

    let mut receipts = Vec::with_capacity(fold_values.len());
    for fold in fold_values {
        receipts.push(audit_fold(&bytes, fold)?);
    }

    let output = json!({
        "schema": "symthaea.matbench-fold-audit.v0",
        "artifact_path": artifact_path,
        "fold_receipts": receipts,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| format!("failed to encode audit receipt: {error}"))?
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("matbench-fold-audit: {error}");
            ExitCode::FAILURE
        }
    }
}
