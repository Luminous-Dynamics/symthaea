// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproducibility audit for the frozen UniMorph English 4.0 compiler input.
//!
//! This consumes the checked-in human-readable snapshot/selection manifests, retrieves the
//! immutable raw artifact, verifies byte identity and exact selected records, then invokes the
//! production UniMorph compiler and current replay contract. It is evidence for provenance only;
//! it does not claim linguistic correctness or corpus completeness.

use std::{
    io::Read,
    process::Command,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use symthaea_broca::{
    MorphophonologicalResourceEvidence, MorphophonologicalRuleSet,
    MorphophonologicalSourceSlice,
};

const SNAPSHOT_MANIFEST: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../docs/broca/unimorph_eng_4_snapshot_manifest.md"));
const SELECTION_MANIFEST: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../docs/broca/unimorph_eng_4_selection_manifest.md"));

const EXPECTED_RAW_URI: &str =
    "https://raw.githubusercontent.com/unimorph/eng/66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b/eng";
const EXPECTED_COMMIT: &str = "66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b";
const EXPECTED_BLOB_SHA: &str = "8eae5ed242e87e50f6bd182133277f50fe93cef3";
const EXPECTED_README_BLOB_SHA: &str = "197564dd6bb45b2bcdad08428446ad2a5db6138d";
const EXPECTED_LICENSE: &str = "CC BY-SA 3.0";
const EXPECTED_SOURCE: &str = "Wikipedia";
const EXPECTED_ARTIFACT_BLAKE3: &str =
    "c4a677818237fb1060d2541272e2da1d5b6bfd2ae40df00b9187d1ae8566426f";
const EXPECTED_ARTIFACT_BYTES: usize = 18_022_905;
const EXPECTED_SELECTION_BLAKE3: &str =
    "74585e2733a2d036cc396ebb5be77796d10b0de78814ad798731b2b96efbd726";
const EXPECTED_SELECTION_COUNT: usize = 7;

const EXPECTED_SLICES: [(&str, usize, usize, &str); 7] = [
    (
        "eng4:line:1",
        0,
        26,
        "d9374b6a84663b09917a521a865d73749386c059281bbd9ba3cbc92435f049f3",
    ),
    (
        "eng4:line:2",
        26,
        32,
        "907603894b6fe571ab7698f4e54e93dc2f0b707e210df91c9ae870b2c6449e67",
    ),
    (
        "eng4:line:3",
        58,
        35,
        "333a1e0f859e7be770e54645652a9c0da696a4ff3ae8ae50f8c113f93cbc2273",
    ),
    (
        "eng4:line:4",
        93,
        27,
        "3038dfd4dc2907b1d43cbd38f907673bec2ffc037903a65218e2642b0d5d0193",
    ),
    (
        "eng4:line:5",
        120,
        34,
        "fe340586f544413839e9a00c7993f09f51f26cf92b609468fff47a311f8d2f3f",
    ),
    (
        "eng4:line:6",
        154,
        20,
        "9db132698ce145c60901a0f287a93b531c9986efccefe4ed8ff94c0a4d272a03",
    ),
    (
        "eng4:line:7",
        174,
        24,
        "68a55886f3a052a603a873234c9a60a2cfcfbfff7b70eb6c25c04ce5c73fcf28",
    ),
];

const EXPECTED_ROWS: [&str; 7] = [
    "microtome\tmicrotomes\tN;PL\n",
    "microtome\tmicrotomes\tV;PRS;3;SG\n",
    "microtome\tmicrotoming\tV;V.PTCP;PRS\n",
    "microtome\tmicrotomed\tV;PST\n",
    "microtome\tmicrotomed\tV;V.PTCP;PST\n",
    "eat\teats\tV;PRS;3;SG\n",
    "eat\teating\tV;V.PTCP;PRS\n",
];

fn main() -> Result<()> {
    verify_checked_in_manifests()?;

    let uri = parse_required_line(
        SNAPSHOT_MANIFEST,
        "- Immutable raw URI: ",
    )?;
    let commit = parse_required_line(
        SNAPSHOT_MANIFEST,
        "- Immutable commit: ",
    )?;
    let blob = parse_required_line(
        SNAPSHOT_MANIFEST,
        "- Git blob SHA: ",
    )?;
    let artifact_len: usize = parse_required_line(
        SNAPSHOT_MANIFEST,
        "- UTF-8 byte length: ",
    )?
    .parse()
    .context("snapshot manifest byte length is not an integer")?;
    let artifact_blake3 = parse_required_line(
        SNAPSHOT_MANIFEST,
        "- BLAKE3-256: ",
    )?;
    let readme_blob = parse_required_line(
        SNAPSHOT_MANIFEST,
        "README blob SHA at the same commit: ",
    )?;

    if uri != EXPECTED_RAW_URI
        || commit != EXPECTED_COMMIT
        || blob != EXPECTED_BLOB_SHA
        || artifact_len != EXPECTED_ARTIFACT_BYTES
        || artifact_blake3 != EXPECTED_ARTIFACT_BLAKE3
        || readme_blob != EXPECTED_README_BLOB_SHA
        || !SNAPSHOT_MANIFEST.contains(&format!("- Source: {EXPECTED_SOURCE}"))
        || !SNAPSHOT_MANIFEST.contains(&format!("- License: {EXPECTED_LICENSE}"))
    {
        bail!("checked-in UniMorph snapshot manifest disagrees with its frozen identity constants");
    }

    let artifact = fetch_artifact(uri)?;
    if artifact.len() != EXPECTED_ARTIFACT_BYTES {
        bail!(
            "frozen UniMorph artifact byte length mismatch: expected {}, got {}",
            EXPECTED_ARTIFACT_BYTES,
            artifact.len()
        );
    }

    let actual_b3 = blake3::hash(&artifact).to_hex().to_string();
    if actual_b3 != EXPECTED_ARTIFACT_BLAKE3 {
        bail!(
            "frozen UniMorph artifact BLAKE3 mismatch: expected {}, got {}",
            EXPECTED_ARTIFACT_BLAKE3,
            actual_b3
        );
    }

    let actual_blob = git_blob_sha1(&artifact)?;
    if actual_blob != EXPECTED_BLOB_SHA {
        bail!(
            "frozen UniMorph Git blob mismatch: expected {}, got {}",
            EXPECTED_BLOB_SHA,
            actual_blob
        );
    }

    let slices = expected_slices()?;
    if slices.len() != EXPECTED_SELECTION_COUNT {
        bail!("expected {} frozen source slices, got {}", EXPECTED_SELECTION_COUNT, slices.len());
    }

    let mut selected_bytes = Vec::with_capacity(artifact.len().min(256));
    for ((record_id, offset, length, digest), expected_row) in EXPECTED_SLICES
        .iter()
        .zip(EXPECTED_ROWS.iter())
    {
        let slice = artifact
            .get(*offset..offset.saturating_add(*length))
            .with_context(|| format!("frozen slice {record_id} is out of bounds"))?;
        if slice != expected_row.as_bytes() {
            bail!("frozen slice {record_id} does not equal its checked-in manifest row");
        }
        let actual = blake3::hash(slice).to_hex().to_string();
        if actual != *digest {
            bail!("frozen slice {record_id} BLAKE3 mismatch");
        }
        selected_bytes.extend_from_slice(slice);
    }

    let mut witness_selection = slices.clone();
    let (rule_set, witness) = MorphophonologicalRuleSet::compile_unimorph_tsv_source(
        "en",
        "en-unspecified",
        MorphophonologicalResourceEvidence::external(
            "unimorph/eng",
            uri,
            EXPECTED_COMMIT,
            EXPECTED_LICENSE,
            EXPECTED_ARTIFACT_BLAKE3,
        )?,
        "unimorph-eng-4",
        "unimorph-eng-4-selection",
        &artifact,
        std::mem::take(&mut witness_selection),
    )?;

    if witness.source_selection_blake3 != EXPECTED_SELECTION_BLAKE3 {
        bail!(
            "compiler selection digest mismatch: expected {}, got {}",
            EXPECTED_SELECTION_BLAKE3,
            witness.source_selection_blake3
        );
    }

    witness
        .validate_against_source_artifact_and_rule_set(&artifact, &rule_set)
        .context("structural frozen UniMorph witness validation failed")?;
    witness
        .replay_unimorph_tsv_compilation(&artifact, &rule_set)
        .context("current frozen UniMorph compiler replay failed")?;

    // Ensure the manifest's aggregate commitment covers the same seven persisted slice objects,
    // not merely the same concatenated payload.
    let recomputed = recompute_selection_digest(&slices);
    if recomputed != EXPECTED_SELECTION_BLAKE3 {
        bail!(
            "selection manifest aggregate digest mismatch: expected {}, got {}",
            EXPECTED_SELECTION_BLAKE3,
            recomputed
        );
    }

    println!(
        "BROCA_UNIMORPH_FREEZE_AUDIT PASS artifact_bytes={} artifact_blake3={} git_blob={} selection_blake3={} records={} rows_bytes={} compiler_revision={} parser_revision={} build_context_revision={}",
        artifact.len(),
        actual_b3,
        actual_blob,
        witness.source_selection_blake3,
        slices.len(),
        selected_bytes.len(),
        symthaea_broca::UNIMORPH_TSV_COMPILER_IMPLEMENTATION_REVISION,
        symthaea_broca::UNIMORPH_TSV_SOURCE_PARSER_REVISION,
        symthaea_broca::UNIMORPH_TSV_COMPILER_BUILD_CONTEXT_REVISION,
    );
    Ok(())
}

fn verify_checked_in_manifests() -> Result<()> {
    for (name, text) in [
        ("snapshot manifest", SNAPSHOT_MANIFEST),
        ("selection manifest", SELECTION_MANIFEST),
    ] {
        if text.trim().is_empty() {
            bail!("{name} is empty");
        }
    }

    if !SNAPSHOT_MANIFEST.contains(EXPECTED_RAW_URI) {
        bail!("snapshot manifest does not contain the immutable raw URI");
    }

    for required in [
        EXPECTED_COMMIT,
        EXPECTED_BLOB_SHA,
        EXPECTED_ARTIFACT_BLAKE3,
        EXPECTED_SELECTION_BLAKE3,
    ] {
        if !SNAPSHOT_MANIFEST.contains(required) && !SELECTION_MANIFEST.contains(required) {
            bail!("checked-in manifests are missing expected identity {required}");
        }
    }

    if SELECTION_MANIFEST.matches("eng4:line:").count() != EXPECTED_SELECTION_COUNT {
        bail!("selection manifest does not enumerate exactly seven frozen records");
    }

    for ((record_id, offset, length, digest), expected_row) in
        EXPECTED_SLICES.iter().zip(EXPECTED_ROWS.iter())
    {
        let table_row =
            format!("| {record_id} | {line} | {offset} | {length} | {digest} |",
                line = EXPECTED_SLICES
                    .iter()
                    .position(|entry| entry.0 == *record_id)
                    .map(|index| index + 1)
                    .unwrap_or_default());
        if !SELECTION_MANIFEST.contains(&table_row) {
            bail!("selection manifest is missing exact table entry for {record_id}");
        }
        if !SELECTION_MANIFEST.contains(expected_row.trim_end_matches('\n')) {
            bail!("selection manifest is missing exact selected record text for {record_id}");
        }
    }

    Ok(())
}

fn parse_required_line<'a>(text: &'a str, prefix: &str) -> Result<&'a str> {
    text.lines()
        .find_map(|line| line.strip_prefix(prefix))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .with_context(|| format!("missing non-empty manifest field {prefix:?}"))
}

fn fetch_artifact(uri: &str) -> Result<Vec<u8>> {
    let response = ureq::get(uri)
        .timeout(Duration::from_secs(120))
        .call()
        .with_context(|| format!("failed to download frozen UniMorph artifact {uri}"))?;
    let mut bytes = Vec::with_capacity(EXPECTED_ARTIFACT_BYTES);
    response
        .into_reader()
        .take((EXPECTED_ARTIFACT_BYTES as u64) + 1)
        .read_to_end(&mut bytes)
        .context("failed to read frozen UniMorph artifact response")?;
    Ok(bytes)
}

fn git_blob_sha1(bytes: &[u8]) -> Result<String> {
    let mut child = Command::new("git")
        .args(["hash-object", "--stdin"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .context("failed to start git hash-object")?;
    {
        let stdin = child.stdin.as_mut().context("git hash-object stdin unavailable")?;
        std::io::Write::write_all(stdin, bytes)
            .context("failed to stream frozen artifact to git hash-object")?;
    }
    let output = child
        .wait_with_output()
        .context("failed to finish git hash-object")?;
    if !output.status.success() {
        bail!("git hash-object exited unsuccessfully: {}", output.status);
    }
    let digest = String::from_utf8(output.stdout)
        .context("git hash-object emitted non-UTF-8 output")?;
    Ok(digest.trim().to_owned())
}

fn expected_slices() -> Result<Vec<MorphophonologicalSourceSlice>> {
    EXPECTED_SLICES
        .iter()
        .map(|(record_id, byte_offset, byte_length, record_blake3)| {
            if *byte_length == 0 {
                bail!("frozen record {record_id} has zero length");
            }
            Ok(MorphophonologicalSourceSlice {
                record_id: (*record_id).to_owned(),
                byte_offset: *byte_offset,
                byte_length: *byte_length,
                record_blake3: (*record_blake3).to_owned(),
            })
        })
        .collect()
}

fn recompute_selection_digest(slices: &[MorphophonologicalSourceSlice]) -> String {
    let serialized = serde_json::to_vec(slices).expect("source slices are serializable");
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-morphophonological-source-selection-v1\0");
    hasher.update(&serialized);
    hasher.finalize().to_hex().to_string()
}
