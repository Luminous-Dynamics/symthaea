// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strict importer for the frozen January 2026 SorryDB evaluation cohort.
//!
//! This validates exact upstream bytes and emits a deterministic task manifest.
//! It does not rebuild Lean projects or evaluate proof candidates.
//!
//! Usage:
//!   cargo run -p symthaea-proof-insights --bin import_sorrydb_snapshot -- --self-test
//!   cargo run -p symthaea-proof-insights --bin import_sorrydb_snapshot -- \
//!     --input /path/to/SorryDB_2601_1000_evaluation_split.json \
//!     --output /path/to/pitb-sorrydb-task-manifest.json

use std::{
    collections::BTreeSet,
    env,
    fs,
    path::Path,
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SOURCE_REPOSITORY: &str = "SorryDB/SorryDB";
const SOURCE_COMMIT: &str = "74b3b63f8ba890d675f362924b21ac55db410383";
const SOURCE_PATH: &str =
    "data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json";
const SOURCE_GIT_BLOB_SHA1: &str = "af3b663429dc4689ee728d2fb5c7d8d5034476f7";
const SOURCE_SHA256: &str =
    "26bfd11099f6b8db7f80d49bdea5d9f8ea0d23ba65d44d93f67ed42ba1530062";
const SOURCE_BYTES: usize = 1_599_271;
const EXPECTED_TASK_COUNT: usize = 1_000;
const EXPECTED_PROJECT_COUNT: usize = 78;
const EXPECTED_DOCUMENTATION: &str = concat!(
    "deduplicated list of sorries, for each unique goal string ",
    "the most recent inclusion date is chosen"
);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceSnapshot {
    documentation: String,
    sorries: Vec<SourceSorry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceSorry {
    id: String,
    repo: SourceRepository,
    location: SourceLocation,
    debug_info: SourceDebugInfo,
    metadata: SourceMetadata,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceRepository {
    remote: String,
    branch: String,
    commit: String,
    lean_version: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceLocation {
    path: String,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceDebugInfo {
    goal: String,
    url: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceMetadata {
    blame_email_hash: String,
    blame_date: String,
    inclusion_date: String,
}

#[derive(Debug, Serialize)]
struct SourcePin {
    repository: &'static str,
    commit: &'static str,
    path: &'static str,
    git_blob_sha1: &'static str,
    source_bytes: usize,
    source_sha256: &'static str,
}

#[derive(Debug, Serialize)]
struct ManifestTask {
    ordinal: usize,
    sorry_id: String,
    repository: SourceRepository,
    location: SourceLocation,
    blame_email_hash: String,
    blame_date: String,
    inclusion_date: String,
    debug_goal_sha256: String,
    debug_url: String,
}

#[derive(Debug, Serialize)]
struct ManifestPayload {
    schema_version: &'static str,
    snapshot_id: &'static str,
    source: SourcePin,
    task_count: usize,
    project_count: usize,
    tasks: Vec<ManifestTask>,
}

#[derive(Debug, Serialize)]
struct OutputManifest {
    #[serde(flatten)]
    payload: ManifestPayload,
    manifest_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn sha1_hex(input: &[u8]) -> String {
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut h0 = 0x6745_2301_u32;
    let mut h1 = 0xEFCD_AB89_u32;
    let mut h2 = 0x98BA_DCFE_u32;
    let mut h3 = 0x1032_5476_u32;
    let mut h4 = 0xC3D2_E1F0_u32;

    for chunk in padded.chunks_exact(64) {
        let mut words = [0_u32; 80];
        for (index, word) in words.iter_mut().take(16).enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for index in 16..80 {
            words[index] = (words[index - 3]
                ^ words[index - 8]
                ^ words[index - 14]
                ^ words[index - 16])
                .rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);
        for (index, word) in words.iter().enumerate() {
            let (function, constant) = match index {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999_u32),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1_u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC_u32),
                _ => (b ^ c ^ d, 0xCA62_C1D6_u32),
            };
            let temporary = a
                .rotate_left(5)
                .wrapping_add(function)
                .wrapping_add(e)
                .wrapping_add(constant)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temporary;
        }
        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    format!("{h0:08x}{h1:08x}{h2:08x}{h3:08x}{h4:08x}")
}

fn git_blob_sha1(raw: &[u8]) -> String {
    let header = format!("blob {}\0", raw.len());
    let mut object = Vec::with_capacity(header.len() + raw.len());
    object.extend_from_slice(header.as_bytes());
    object.extend_from_slice(raw);
    sha1_hex(&object)
}

fn is_lower_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn contains_forbidden(value: &str, forbidden: &[char]) -> bool {
    value.chars().any(|character| forbidden.contains(&character))
}

fn valid_github_remote(remote: &str) -> bool {
    let Some(rest) = remote.strip_prefix("https://github.com/") else {
        return false;
    };
    if contains_forbidden(rest, &['?', '#', '\\', ' ', '\t', '\n', '\r']) {
        return false;
    }
    let rest = rest.trim_end_matches('/').trim_end_matches(".git");
    let mut parts = rest.split('/');
    let (Some(owner), Some(repository), None) = (parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    !owner.is_empty()
        && !repository.is_empty()
        && owner != "."
        && owner != ".."
        && repository != "."
        && repository != ".."
}

fn valid_branch(branch: &str) -> bool {
    !branch.is_empty()
        && !branch.starts_with('/')
        && !branch.ends_with('/')
        && !branch.contains("..")
        && !contains_forbidden(branch, &['\\', ' ', '\t', '\n', '\r', '\0'])
}

fn valid_lean_version(version: &str) -> bool {
    version.starts_with('v')
        && version.len() > 1
        && version
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'+'))
}

fn validate_relative_source_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.starts_with('/')
        || contains_forbidden(path, &['\\', '\0', ':'])
    {
        bail!("source path is empty, absolute, or contains a forbidden separator");
    }
    for component in path.split('/') {
        if component.is_empty() || component == "." || component == ".." {
            bail!(
                "source path contains an empty, current-directory, or parent-directory component"
            );
        }
    }
    Ok(())
}

fn validate_location(location: &SourceLocation) -> Result<()> {
    validate_relative_source_path(&location.path)?;
    if location.start_line == 0 || location.end_line == 0 {
        bail!("source line numbers must be one-based");
    }
    // SorryDB uses zero-based columns for some source coordinates.
    if (location.end_line, location.end_column)
        <= (location.start_line, location.start_column)
    {
        bail!("source span must have positive length and nondecreasing coordinates");
    }
    Ok(())
}

fn validate_record(record: &SourceSorry, index: usize) -> Result<()> {
    if !is_lower_hex(&record.id, 64) {
        bail!("record {index}: sorry id must be 64 lowercase hexadecimal characters");
    }
    if !valid_github_remote(&record.repo.remote) {
        bail!("record {index}: repository remote is not a canonical GitHub HTTPS URL");
    }
    if !valid_branch(&record.repo.branch) {
        bail!("record {index}: repository branch is empty or malformed");
    }
    if !is_lower_hex(&record.repo.commit, 40) {
        bail!("record {index}: repository commit must be a 40-character lowercase SHA-1");
    }
    if !valid_lean_version(&record.repo.lean_version) {
        bail!("record {index}: Lean toolchain tag is empty or malformed");
    }
    validate_location(&record.location)
        .with_context(|| format!("record {index}: invalid source location"))?;
    if record.debug_info.goal.trim().is_empty() {
        bail!("record {index}: debug goal is empty");
    }
    if !record.debug_info.url.starts_with("https://github.com/") {
        bail!("record {index}: debug URL is not a GitHub HTTPS URL");
    }
    if record.metadata.blame_email_hash.is_empty()
        || record.metadata.blame_date.is_empty()
        || record.metadata.inclusion_date.is_empty()
    {
        bail!("record {index}: required metadata field is empty");
    }
    Ok(())
}

fn validate_snapshot(raw: &[u8]) -> Result<SourceSnapshot> {
    if raw.len() != SOURCE_BYTES {
        bail!(
            "source byte count mismatch: expected {SOURCE_BYTES}, received {}",
            raw.len()
        );
    }
    let actual_sha256 = sha256_hex(raw);
    if actual_sha256 != SOURCE_SHA256 {
        bail!(
            "SorryDB source SHA-256 mismatch: expected {SOURCE_SHA256}, received {actual_sha256}"
        );
    }
    let actual_git_blob_sha1 = git_blob_sha1(raw);
    if actual_git_blob_sha1 != SOURCE_GIT_BLOB_SHA1 {
        bail!(
            "SorryDB Git blob SHA-1 mismatch: expected {SOURCE_GIT_BLOB_SHA1}, received {actual_git_blob_sha1}"
        );
    }

    let snapshot: SourceSnapshot =
        serde_json::from_slice(raw).context("parse pinned SorryDB JSON snapshot")?;
    if snapshot.documentation != EXPECTED_DOCUMENTATION {
        bail!("SorryDB documentation marker differs from the pinned snapshot");
    }
    if snapshot.sorries.len() != EXPECTED_TASK_COUNT {
        bail!(
            "task count mismatch: expected {EXPECTED_TASK_COUNT}, received {}",
            snapshot.sorries.len()
        );
    }

    let mut ids = BTreeSet::new();
    let mut repositories = BTreeSet::new();
    for (index, record) in snapshot.sorries.iter().enumerate() {
        validate_record(record, index)?;
        if !ids.insert(record.id.as_str()) {
            bail!("duplicate sorry id at record {index}: {}", record.id);
        }
        repositories.insert(record.repo.remote.as_str());
    }
    if repositories.len() != EXPECTED_PROJECT_COUNT {
        bail!(
            "project count mismatch: expected {EXPECTED_PROJECT_COUNT}, received {}",
            repositories.len()
        );
    }
    Ok(snapshot)
}

fn build_manifest(raw: &[u8]) -> Result<OutputManifest> {
    let snapshot = validate_snapshot(raw)?;
    let mut tasks = Vec::with_capacity(snapshot.sorries.len());

    for (ordinal, record) in snapshot.sorries.into_iter().enumerate() {
        tasks.push(ManifestTask {
            ordinal,
            sorry_id: record.id,
            repository: record.repo,
            location: record.location,
            blame_email_hash: record.metadata.blame_email_hash,
            blame_date: record.metadata.blame_date,
            inclusion_date: record.metadata.inclusion_date,
            debug_goal_sha256: sha256_hex(record.debug_info.goal.as_bytes()),
            debug_url: record.debug_info.url,
        });
    }

    let payload = ManifestPayload {
        schema_version: "pitb.sorrydb.task-manifest.v0",
        snapshot_id: "SorryDB_2601",
        source: SourcePin {
            repository: SOURCE_REPOSITORY,
            commit: SOURCE_COMMIT,
            path: SOURCE_PATH,
            git_blob_sha1: SOURCE_GIT_BLOB_SHA1,
            source_bytes: SOURCE_BYTES,
            source_sha256: SOURCE_SHA256,
        },
        task_count: tasks.len(),
        project_count: EXPECTED_PROJECT_COUNT,
        tasks,
    };

    let canonical_payload = serde_json::to_vec(&payload)?;
    let manifest_sha256 = sha256_hex(&canonical_payload);
    Ok(OutputManifest {
        payload,
        manifest_sha256,
    })
}

fn run_self_test() -> Result<()> {
    if validate_relative_source_path("Mathlib/Algebra/Example.lean").is_err() {
        bail!("self-test: ordinary repository-relative path was rejected");
    }
    for path in [
        "",
        "/etc/passwd",
        "../outside.lean",
        "Mathlib/../outside.lean",
        r"Mathlib\Outside.lean",
        "C:/outside.lean",
        "Mathlib//Example.lean",
    ] {
        if validate_relative_source_path(path).is_ok() {
            bail!("self-test: unsafe path was accepted: {path:?}");
        }
    }
    if !is_lower_hex(&"a".repeat(64), 64)
        || is_lower_hex(&"A".repeat(64), 64)
        || is_lower_hex(&"g".repeat(64), 64)
    {
        bail!("self-test: hexadecimal digest validation is incorrect");
    }
    if !valid_github_remote("https://github.com/leanprover-community/mathlib4")
        || valid_github_remote("https://github.com/owner/repo/extra")
        || valid_github_remote("http://github.com/owner/repo")
    {
        bail!("self-test: repository remote validation is incorrect");
    }
    if validate_location(&SourceLocation {
        path: "A.lean".to_owned(),
        start_line: 10,
        start_column: 0,
        end_line: 10,
        end_column: 5,
    })
    .is_err()
    {
        bail!("self-test: valid zero-based column coordinate was rejected");
    }
    if validate_location(&SourceLocation {
        path: "A.lean".to_owned(),
        start_line: 10,
        start_column: 8,
        end_line: 10,
        end_column: 5,
    })
    .is_ok()
    {
        bail!("self-test: reversed source span was accepted");
    }
    println!(
        "{}",
        concat!(
            "PITB_SORRYDB_IMPORTER_SELF_TEST_PASS: pinned cohort constants, path safety, ",
            "remote/commit/Lean tags, zero-based columns, and source-span controls"
        )
    );
    Ok(())
}

enum Mode {
    Help,
    SelfTest,
    Import { input: String, output: String },
}

fn parse_args() -> Result<Mode> {
    let mut args = env::args().skip(1);
    let mut input = None;
    let mut output = None;
    let mut self_test = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next(),
            "--output" => output = args.next(),
            "--self-test" => self_test = true,
            "--help" | "-h" => return Ok(Mode::Help),
            _ => bail!("unknown argument: {arg}"),
        }
    }

    if self_test {
        if input.is_some() || output.is_some() {
            bail!("--self-test cannot be combined with --input or --output");
        }
        return Ok(Mode::SelfTest);
    }
    match (input, output) {
        (Some(input), Some(output)) => Ok(Mode::Import { input, output }),
        _ => bail!("both --input and --output are required"),
    }
}

fn run() -> Result<()> {
    match parse_args()? {
        Mode::Help => {
            println!(
                "Usage: import_sorrydb_snapshot --self-test | --input FILE --output FILE"
            );
            Ok(())
        }
        Mode::SelfTest => run_self_test(),
        Mode::Import { input, output } => {
            let raw = fs::read(&input)
                .with_context(|| format!("read SorryDB input {input}"))?;
            let manifest = build_manifest(&raw)?;
            let encoded = serde_json::to_vec_pretty(&manifest)?;
            let out_path = Path::new(&output);
            if let Some(parent) = out_path.parent().filter(|path| !path.as_os_str().is_empty()) {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create output directory {}", parent.display()))?;
            }
            fs::write(out_path, &encoded)
                .with_context(|| format!("write PITB task manifest {}", out_path.display()))?;
            println!(
                concat!(
                    "PITB_SORRYDB_IMPORT_PASS: snapshot=SorryDB_2601 ",
                    "tasks={} repositories={} source_sha256={} manifest_sha256={}"
                ),
                manifest.payload.task_count,
                manifest.payload.project_count,
                SOURCE_SHA256,
                manifest.manifest_sha256
            );
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB_SORRYDB_IMPORT_FAIL: {error:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_path_controls_reject_traversal_and_windows_paths() {
        assert!(validate_relative_source_path("Mathlib/Algebra/Example.lean").is_ok());
        assert!(validate_relative_source_path("../outside.lean").is_err());
        assert!(validate_relative_source_path(r"Mathlib\Outside.lean").is_err());
        assert!(validate_relative_source_path("C:/outside.lean").is_err());
    }

    #[test]
    fn source_coordinates_allow_zero_based_columns_but_reject_reversed_spans() {
        assert!(validate_location(&SourceLocation {
            path: "A.lean".to_owned(),
            start_line: 10,
            start_column: 0,
            end_line: 10,
            end_column: 5,
        })
        .is_ok());
        assert!(validate_location(&SourceLocation {
            path: "A.lean".to_owned(),
            start_line: 10,
            start_column: 8,
            end_line: 10,
            end_column: 5,
        })
        .is_err());
    }

    #[test]
    fn sha1_matches_known_digest_and_git_blob_object() {
        assert_eq!(
            sha1_hex(b"abc"),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        assert_eq!(
            git_blob_sha1(b"abc"),
            "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"
        );
    }

    #[test]
    fn upstream_identity_constants_are_well_formed() {
        assert!(is_lower_hex(SOURCE_COMMIT, 40));
        assert!(is_lower_hex(SOURCE_GIT_BLOB_SHA1, 40));
        assert!(is_lower_hex(SOURCE_SHA256, 64));
    }
}
