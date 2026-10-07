// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproducibility audit for the frozen UniMorph English 4.0 compiler input.
//!
//! This consumes the checked-in snapshot manifests, retrieves the immutable raw artifact, verifies
//! byte identity and exact selected records from the machine-readable selection manifest, then
//! invokes the production UniMorph compiler and current replay contract. It is evidence for provenance only;
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
const SELECTION_MANIFEST_JSON: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../docs/broca/unimorph_eng_4_selection_manifest.json"));

const EXPECTED_RAW_URI: &str =
    "https://raw.githubusercontent.com/unimorph/eng/66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b/eng";
const EXPECTED_COMMIT: &str = "66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b";
const EXPECTED_BLOB_SHA: &str = "8eae5ed242e87e50f6bd182133277f50fe93cef3";
const EXPECTED_README_BLOB_SHA: &str = "197564dd6bb45b2bcdad08446ad2a5db6138d";
const EXPECTED_README_URI: &str =
    "https://raw.githubusercontent.com/unimorph/eng/66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b/README.md";
const EXPECTED_LICENSE: &str = "CC BY-SA 3.0";
const EXPECTED_SOURCE: &str = "Wikipedia";
const EXPECTED_ARTIFACT_BLAKE3: &str =
    "c4a677818237fb1060d2541272e2da1d5b6bfd2ae40df00b9187d1ae8566426f";
const EXPECTED_ARTIFACT_BYTES: usize = 18_022_905;
const EXPECTED_SELECTION_BLAKE3: &str =
    "74585e2733a2d036cc396ebb5be77796d10b0de78814ad798731b2b96efbd726";
const EXPECTED_SELECTION_COUNT: usize = 7;

const EXPECTED_SELECTION_SCHEMA: &str = "broca-unimorph-selection-manifest-v1";

#[derive(Debug, serde::Deserialize)]
struct SelectionManifest {
    schema_version: String,
    upstream_repository: String,
    immutable_commit: String,
    git_blob_sha: String,
    artifact_blake3: String,
    artifact_byte_length: usize,
    aggregate_source_selection_blake3: String,
    records: Vec<SelectionRecord>,
}

#[derive(Debug, serde::Deserialize)]
struct SelectionRecord {
    record_id: String,
    source_line: usize,
    byte_offset: usize,
    byte_length: usize,
    record_blake3: String,
    row: String,
}


fn main() -> Result<()> {
    let selection = parse_selection_manifest()?;
    verify_checked_in_manifests(&selection)?;

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
        || selection.upstream_repository != "unimorph/eng"
        || selection.immutable_commit != EXPECTED_COMMIT
        || selection.git_blob_sha != EXPECTED_BLOB_SHA
        || selection.artifact_blake3 != EXPECTED_ARTIFACT_BLAKE3
        || selection.artifact_byte_length != EXPECTED_ARTIFACT_BYTES
        || selection.aggregate_source_selection_blake3 != EXPECTED_SELECTION_BLAKE3
        || selection.records.len() != EXPECTED_SELECTION_COUNT
    {
        bail!("checked-in UniMorph snapshot manifest disagrees with its frozen identity constants");
    }

    let readme = fetch_artifact(EXPECTED_README_URI)
        .context("failed to download immutable UniMorph README for attribution verification")?;
    let actual_readme_blob = git_blob_sha1(&readme)?;
    if actual_readme_blob != EXPECTED_README_BLOB_SHA {
        bail!(
            "frozen UniMorph README Git blob mismatch: expected {}, got {}",
            EXPECTED_README_BLOB_SHA,
            actual_readme_blob
        );
    }
    let readme_text = std::str::from_utf8(&readme)
        .context("frozen UniMorph README is not UTF-8")?;
    if !readme_text.contains(EXPECTED_SOURCE) || !readme_text.contains(EXPECTED_LICENSE) {
        bail!("immutable UniMorph README does not contain the expected source/license attribution");
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

    let slices = selection.records.iter().map(to_source_slice).collect::<Result<Vec<_>>>()?;
    let mut selected_bytes = Vec::with_capacity(artifact.len().min(256));
    for record in &selection.records {
        let end = record
            .byte_offset
            .checked_add(record.byte_length)
            .with_context(|| format!("frozen slice {} range overflow", record.record_id))?;
        let slice = artifact
            .get(record.byte_offset..end)
            .with_context(|| format!("frozen slice {} is out of bounds", record.record_id))?;
        if slice != record.row.as_bytes() {
            bail!("frozen slice {} does not equal its machine-readable manifest row", record.record_id);
        }
        let actual = blake3::hash(slice).to_hex().to_string();
        if actual != record.record_blake3 {
            bail!("frozen slice {} BLAKE3 mismatch", record.record_id);
        }
        if record.source_line == 0 {
            bail!("frozen slice {} has invalid source line 0", record.record_id);
        }
        let actual_source_line = artifact[..record.byte_offset]
            .iter()
            .filter(|byte| **byte == b'\n')
            .count()
            + 1;
        if actual_source_line != record.source_line {
            bail!(
                "frozen slice {} declares source line {}, but its byte offset is on line {}",
                record.record_id,
                record.source_line,
                actual_source_line
            );
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

    let ci_provenance = collect_ci_provenance();
    println!(
        "BROCA_UNIMORPH_FREEZE_AUDIT PASS artifact_bytes={} artifact_blake3={} git_blob={} selection_blake3={} records={} rows_bytes={} compiler_revision={} parser_revision={} build_context_revision={} ci_provenance={}",
        artifact.len(),
        actual_b3,
        actual_blob,
        witness.source_selection_blake3,
        slices.len(),
        selected_bytes.len(),
        symthaea_broca::UNIMORPH_TSV_COMPILER_IMPLEMENTATION_REVISION,
        symthaea_broca::UNIMORPH_TSV_SOURCE_PARSER_REVISION,
        symthaea_broca::UNIMORPH_TSV_COMPILER_BUILD_CONTEXT_REVISION,
        ci_provenance,
    );
    Ok(())
}

fn parse_selection_manifest() -> Result<SelectionManifest> {
    let manifest: SelectionManifest =
        serde_json::from_str(SELECTION_MANIFEST_JSON).context("invalid machine-readable UniMorph selection manifest")?;
    if manifest.schema_version != EXPECTED_SELECTION_SCHEMA {
        bail!(
            "unsupported UniMorph selection manifest schema: expected {}, got {}",
            EXPECTED_SELECTION_SCHEMA,
            manifest.schema_version
        );
    }
    Ok(manifest)
}

fn to_source_slice(record: &SelectionRecord) -> Result<MorphophonologicalSourceSlice> {
    if record.record_id.trim().is_empty()
        || record.record_blake3.len() != 64
        || !record
            .record_blake3
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("selection manifest record {} has malformed identity", record.record_id);
    }
    if record.byte_length == 0 {
        bail!("selection manifest record {} has zero length", record.record_id);
    }
    if record.source_line == 0 {
        bail!("selection manifest record {} has invalid source line 0", record.record_id);
    }
    Ok(MorphophonologicalSourceSlice {
        record_id: record.record_id.clone(),
        byte_offset: record.byte_offset,
        byte_length: record.byte_length,
        record_blake3: record.record_blake3.clone(),
    })
}

fn verify_checked_in_manifests(selection: &SelectionManifest) -> Result<()> {
    for (name, text) in [
        ("snapshot manifest", SNAPSHOT_MANIFEST),
        ("selection manifest", SELECTION_MANIFEST),
        ("selection manifest JSON", SELECTION_MANIFEST_JSON),
    ] {
        if text.trim().is_empty() {
            bail!("{name} is empty");
        }
    }

    if !SNAPSHOT_MANIFEST.contains(EXPECTED_RAW_URI) {
        bail!("snapshot manifest does not contain the immutable raw artifact URI");
    }
    if !SNAPSHOT_MANIFEST.contains(EXPECTED_README_BLOB_SHA) {
        bail!("snapshot manifest does not contain the expected immutable README blob");
    }

    if !SELECTION_MANIFEST.contains(EXPECTED_SELECTION_BLAKE3)
        || !SELECTION_MANIFEST.contains(EXPECTED_BLOB_SHA)
        || !SELECTION_MANIFEST.contains(EXPECTED_ARTIFACT_BLAKE3)
    {
        bail!("human-readable selection manifest is missing required frozen identities");
    }

    if selection.records.len() != EXPECTED_SELECTION_COUNT {
        bail!(
            "selection manifest does not enumerate exactly {} frozen records",
            EXPECTED_SELECTION_COUNT
        );
    }

    let mut source_lines = Vec::with_capacity(selection.records.len());
    let slices = selection.records.iter().map(to_source_slice).collect::<Result<Vec<_>>>()?;
    for (expected_line, record) in selection.records.iter().enumerate() {
        if record.source_line != expected_line + 1 {
            bail!(
                "selection manifest source line sequence is not contiguous at {}",
                record.record_id
            );
        }
        source_lines.push(record.source_line);
        let table_row = format!(
            "| {} | {} | {} | {} | {} |",
            record.record_id,
            record.source_line,
            record.byte_offset,
            record.byte_length,
            record.record_blake3
        );
        if !SELECTION_MANIFEST.contains(&table_row) {
            bail!("human-readable selection manifest is missing exact table entry for {}", record.record_id);
        }
        if !SELECTION_MANIFEST.contains(record.row.trim_end_matches('\n')) {
            bail!("human-readable selection manifest is missing exact selected record text for {}", record.record_id);
        }
    }

    let recomputed = recompute_selection_digest(&slices);
    if recomputed != selection.aggregate_source_selection_blake3 {
        bail!(
            "machine-readable selection aggregate mismatch: expected {}, got {}",
            selection.aggregate_source_selection_blake3,
            recomputed
        );
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

fn collect_ci_provenance() -> String {
    let fields = [
        ("actions", std::env::var("GITHUB_ACTIONS").unwrap_or_else(|_| "unknown".into())),
        ("run_id", std::env::var("GITHUB_RUN_ID").unwrap_or_else(|_| "unknown".into())),
        ("run_attempt", std::env::var("GITHUB_RUN_ATTEMPT").unwrap_or_else(|_| "unknown".into())),
        ("workflow", std::env::var("GITHUB_WORKFLOW").unwrap_or_else(|_| "unknown".into())),
        ("workflow_ref", std::env::var("GITHUB_WORKFLOW_REF").unwrap_or_else(|_| "unknown".into())),
        ("workflow_sha", std::env::var("GITHUB_WORKFLOW_SHA").unwrap_or_else(|_| "unknown".into())),
        ("sha", std::env::var("GITHUB_SHA").unwrap_or_else(|_| "unknown".into())),
        ("ref", std::env::var("GITHUB_REF").unwrap_or_else(|_| "unknown".into())),
        ("runner_os", std::env::var("RUNNER_OS").unwrap_or_else(|_| "unknown".into())),
        ("runner_arch", std::env::var("RUNNER_ARCH").unwrap_or_else(|_| "unknown".into())),
        ("image_os", std::env::var("ImageOS").unwrap_or_else(|_| "unknown".into())),
        ("image_version", std::env::var("ImageVersion").unwrap_or_else(|_| "unknown".into())),
    ];
    fields
        .into_iter()
        .map(|(key, value)| format!("{key}={}", value.replace(' ', "_")))
        .collect::<Vec<_>>()
        .join(",")
}

fn recompute_selection_digest(slices: &[MorphophonologicalSourceSlice]) -> String {
    let serialized = serde_json::to_vec(slices).expect("source slices are serializable");
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-morphophonological-source-selection-v1\0");
    hasher.update(&serialized);
    hasher.finalize().to_hex().to_string()
}
