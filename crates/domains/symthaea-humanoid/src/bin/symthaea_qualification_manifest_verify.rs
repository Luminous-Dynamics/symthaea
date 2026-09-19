// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonicalize and structurally verify externally collected qualification records.
//!
//! This CLI performs no network access and does not establish runtime plant
//! authority. It accepts records already collected from an external evidence
//! source, orders them canonically, constructs the QUAL-PROMO-001 manifest, and
//! runs the pure structural verifier.

use std::{env, ffi::OsString, fs, path::PathBuf};

use anyhow::{Context, Result, ensure};
use symthaea_humanoid::{
    AuthorityPromotionPrerequisiteManifestV1, QualificationExecutionRecordV1,
    verify_authority_promotion_prerequisites_v1,
};

fn main() -> Result<()> {
    let (input_path, output_path) = parse_args()?;

    let bytes = fs::read(&input_path)
        .with_context(|| format!("failed to read qualification records from {}", input_path.display()))?;
    let mut records: Vec<QualificationExecutionRecordV1> = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse qualification records from {}", input_path.display()))?;

    records.sort_by_key(|record| record.role);
    let manifest = AuthorityPromotionPrerequisiteManifestV1::from_records(records);
    let verification = verify_authority_promotion_prerequisites_v1(&manifest)
        .map_err(|error| anyhow::anyhow!("qualification prerequisite verification failed: {error:?}"))?;

    ensure!(
        verification.structurally_satisfied(),
        "qualification prerequisite verifier returned a non-satisfied report"
    );
    ensure!(
        !verification.verifies_external_execution_truth(),
        "pure structural verifier must not claim external execution attestation"
    );
    ensure!(
        !verification.establishes_runtime_authority(),
        "qualification manifest must not establish runtime plant authority"
    );

    let encoded = serde_json::to_vec_pretty(&manifest).context("failed to serialize canonical manifest")?;
    fs::write(&output_path, encoded)
        .with_context(|| format!("failed to write canonical manifest to {}", output_path.display()))?;

    eprintln!(
        "QUAL-PROMO-002 structural verification PASS: records={} external_execution_truth_attested=false runtime_authority=false lineage={}",
        verification.checked_record_count,
        verification.manifest_lineage_id,
    );
    Ok(())
}

fn parse_args() -> Result<(PathBuf, PathBuf)> {
    let mut args = env::args_os();
    let program = args
        .next()
        .unwrap_or_else(|| OsString::from("symthaea_qualification_manifest_verify"));
    let input = args.next().with_context(|| {
        format!(
            "usage: {} <records.json> <manifest.json>",
            PathBuf::from(&program).display()
        )
    })?;
    let output = args.next().with_context(|| {
        format!(
            "usage: {} <records.json> <manifest.json>",
            PathBuf::from(&program).display()
        )
    })?;
    ensure!(
        args.next().is_none(),
        "usage: {} <records.json> <manifest.json>",
        PathBuf::from(program).display()
    );

    Ok((PathBuf::from(input), PathBuf::from(output)))
}
