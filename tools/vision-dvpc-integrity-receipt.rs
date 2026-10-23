// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-012: content-addressed paired-trace evidence verifier.
//!
//! This tool checks byte-level linkage and a precommitted bundle lock. It does
//! not download or redistribute datasets, adjudicate licenses, score a model,
//! establish statistical independence, or make a performance/SOTA claim.
//!
//! Usage:
//!   rustc --edition=2024 --test tools/vision-dvpc-integrity-receipt.rs -o /tmp/vis-dvpc-012-tests
//!   /tmp/vis-dvpc-012-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-integrity-receipt.rs -o /tmp/vis-dvpc-012
//!   /tmp/vis-dvpc-012 DATASET_ROOT OBS.tsv EVAL.tsv SOURCE_MANIFEST.tsv SPLIT_MANIFEST.tsv SNAPSHOT.tsv UTILITY_SPEC.txt VERIFIER_SOURCE.rs BUNDLE_LOCK.tsv

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

const FORMAT_VERSION: &str = "VIS-DVPC-TRACE-001-v1";
const RECEIPT_VERSION: &str = "VIS-DVPC-INTEGRITY-RECEIPT-001-v1";
const OBS_HEADER: &[&str] = &[
    "OBS_HEADER",
    "trace_id",
    "dataset_id",
    "dataset_revision",
    "split",
    "group_id",
    "step_index",
    "sample_sha256",
    "cheap_output_sha256",
    "pre_features_sha256",
    "utility_spec_sha256",
    "declared_extra_latency_us",
    "declared_extra_energy_uj",
    "estimated_dvpc",
    "reported_uncertainty",
    "shift_signal_sha256",
];
const EVAL_HEADER: &[&str] = &[
    "EVAL_HEADER",
    "trace_id",
    "sample_sha256",
    "cheap_utility_01",
    "full_utility_01",
    "cheap_safety_violations",
    "full_safety_violations",
    "actual_extra_latency_us",
    "actual_extra_energy_uj",
    "shift_label",
    "shift_mechanism",
];
const META_KEYS: &[&str] = &[
    "format",
    "dataset_id",
    "dataset_revision",
    "dataset_snapshot_sha256",
    "source_manifest_sha256",
    "split_manifest_sha256",
    "utility_spec_sha256",
    "license_id",
    "license_scope",
    "higher_is_better",
];
const LOCK_KEYS: &[&str] = &[
    "observations_sha256",
    "evaluations_sha256",
    "source_manifest_sha256",
    "split_manifest_sha256",
    "snapshot_sha256",
    "utility_spec_sha256",
    "verifier_source_sha256",
];

#[derive(Debug, Clone)]
struct Table {
    meta: BTreeMap<String, String>,
    rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
struct Verified {
    receipt: String,
    observation_rows: usize,
    evaluation_rows: usize,
    groups: usize,
    source_entries: usize,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

// Compact, dependency-free SHA-256 implementation. The known-vector tests
// below guard this implementation independently of the admission fixtures.
struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    total_len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667,
                0xbb67ae85,
                0x3c6ef372,
                0xa54ff53a,
                0x510e527f,
                0x9b05688c,
                0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0; 64],
            buffered: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.total_len = self.total_len.wrapping_add(input.len() as u64);
        while !input.is_empty() {
            let take = (64 - self.buffered).min(input.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&input[..take]);
            self.buffered += take;
            input = &input[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
            0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];
        let mut schedule = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate() {
            schedule[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let x = schedule[i - 15];
            let y = schedule[i - 2];
            let s0 = x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
            let s1 = y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10);
            schedule[i] = schedule[i - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let big1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(big1)
                .wrapping_add(choose)
                .wrapping_add(K[i])
                .wrapping_add(schedule[i]);
            let big0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = big0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        let working = [a, b, c, d, e, f, g, h];
        for (state, value) in self.state.iter_mut().zip(working) {
            *state = state.wrapping_add(value);
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        let mut padding = [0u8; 128];
        padding[0] = 0x80;
        let pad_len = if self.buffered < 56 {
            56 - self.buffered
        } else {
            120 - self.buffered
        };
        self.update(&padding[..pad_len]);
        self.update(&bit_len.to_be_bytes());
        let mut digest = [0u8; 32];
        for (chunk, word) in digest.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(bytes);
    hex(&hash.finalize())
}

fn sha256_reader(mut reader: impl Read) -> Result<(String, u64), String> {
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut count = 0u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| invalid(format!("cannot read manifested file: {error}")))?;
        if read == 0 {
            break;
        }
        count = count
            .checked_add(read as u64)
            .ok_or_else(|| invalid("file byte count overflow"))?;
        hash.update(&buffer[..read]);
    }
    Ok((hex(&hash.finalize()), count))
}

fn digest_file(path: &Path) -> Result<(String, u64), String> {
    let file = File::open(path)
        .map_err(|error| invalid(format!("cannot open {}: {error}", path.display())))?;
    sha256_reader(file)
}

fn validate_sha256(value: &str, field: &str) -> Result<String, String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(format!(
            "{field} must be a 64-character hexadecimal SHA-256 digest"
        )));
    }
    Ok(value.to_ascii_lowercase())
}

fn validate_token(value: &str, field: &str) -> Result<(), String> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(invalid(format!(
            "{field} must be a non-empty identifier token without whitespace or paths"
        )));
    }
    Ok(())
}

fn parse_table(input: &str, header: &[&str], tag: &str) -> Result<Table, String> {
    let mut meta = BTreeMap::new();
    let mut rows = Vec::new();
    let mut found_header = false;

    for (index, line) in input.lines().enumerate() {
        let line_number = index + 1;
        if line.is_empty() {
            return Err(invalid(format!("line {line_number}: blank lines are forbidden")));
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.first().copied() == Some("META") {
            if found_header {
                return Err(invalid(format!(
                    "line {line_number}: META records must precede the data header"
                )));
            }
            if fields.len() != 3 || fields[1].is_empty() || fields[2].is_empty() {
                return Err(invalid(format!(
                    "line {line_number}: META requires exactly key and value"
                )));
            }
            if meta.insert(fields[1].to_owned(), fields[2].to_owned()).is_some() {
                return Err(invalid(format!(
                    "line {line_number}: duplicate metadata key {}",
                    fields[1]
                )));
            }
            continue;
        }
        if !found_header {
            if fields != header {
                return Err(invalid(format!(
                    "line {line_number}: exact schema mismatch; reordered or unknown columns are forbidden"
                )));
            }
            found_header = true;
            continue;
        }
        if fields.first().copied() != Some(tag) || fields.len() != header.len() {
            return Err(invalid(format!(
                "line {line_number}: expected {tag} row with {} fields",
                header.len()
            )));
        }
        if fields.iter().any(|field| field.is_empty()) {
            return Err(invalid(format!(
                "line {line_number}: empty fields are forbidden; use '-' for an absent optional digest"
            )));
        }
        rows.push(fields.iter().map(|field| (*field).to_owned()).collect());
    }

    if !found_header || rows.is_empty() {
        return Err(invalid("missing header or data rows"));
    }
    let expected: BTreeSet<&str> = META_KEYS.iter().copied().collect();
    let actual: BTreeSet<&str> = meta.keys().map(String::as_str).collect();
    if actual != expected {
        return Err(invalid("META key set mismatch: missing or unknown fields"));
    }
    if meta.get("format").map(String::as_str) != Some(FORMAT_VERSION) {
        return Err(invalid("unsupported trace format"));
    }
    if meta.get("higher_is_better").map(String::as_str) != Some("true") {
        return Err(invalid("higher_is_better must be the literal true"));
    }
    for key in ["dataset_id", "dataset_revision", "license_id"] {
        validate_token(meta.get(key).map(String::as_str).unwrap_or(""), key)?;
    }
    if !matches!(
        meta.get("license_scope").map(String::as_str),
        Some("synthetic_public_fixture" | "noncommercial_research_only" | "commercial_use_authorized")
    ) {
        return Err(invalid("license_scope must use an explicit recognized declaration"));
    }
    for key in [
        "dataset_snapshot_sha256",
        "source_manifest_sha256",
        "split_manifest_sha256",
        "utility_spec_sha256",
    ] {
        let value = meta.get(key).map(String::as_str).unwrap_or("");
        validate_sha256(value, key)?;
    }
    Ok(Table { meta, rows })
}

fn safe_relative_path(value: &str) -> Result<PathBuf, String> {
    if value.is_empty() || value.contains('\\') || value.contains('\0') {
        return Err(invalid("manifest path is empty or contains a forbidden separator"));
    }
    let path = Path::new(value);
    if path.is_absolute() {
        return Err(invalid("absolute source-manifest paths are forbidden"));
    }
    let mut components = 0usize;
    for component in path.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(invalid("source-manifest paths must contain only normal relative components"));
        }
        components += 1;
    }
    if components == 0 {
        return Err(invalid("empty source-manifest path"));
    }
    Ok(path.to_path_buf())
}

fn checked_source_file(root: &Path, relative: &Path) -> Result<PathBuf, String> {
    let mut current = root.to_path_buf();
    let components: Vec<_> = relative.components().collect();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(part) = component else {
            return Err(invalid("non-normal path component rejected"));
        };
        current.push(part);
        let metadata = fs::symlink_metadata(&current)
            .map_err(|error| invalid(format!("missing manifested path {}: {error}", current.display())))?;
        if metadata.file_type().is_symlink() {
            return Err(invalid(format!(
                "symlink in manifested source path rejected: {}",
                current.display()
            )));
        }
        if index + 1 < components.len() && !metadata.is_dir() {
            return Err(invalid("non-directory encountered before final source path component"));
        }
        if index + 1 == components.len() && !metadata.is_file() {
            return Err(invalid("manifested source path must be a regular file"));
        }
    }
    Ok(current)
}

fn parse_source_manifest(
    root: &Path,
    input: &str,
) -> Result<(BTreeSet<(String, String)>, usize), String> {
    let mut lines = input.lines();
    if lines.next() != Some("SOURCE_MANIFEST_V1\trole\tpath\tbyte_len\tsha256") {
        return Err(invalid("source-manifest header mismatch"));
    }
    let mut seen_paths = BTreeSet::new();
    let mut role_digests = BTreeSet::new();
    let mut count = 0usize;

    for (index, line) in lines.enumerate() {
        let line_number = index + 2;
        if line.is_empty() {
            return Err(invalid(format!("source manifest line {line_number} is blank")));
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 5 || fields[0] != "SOURCE" {
            return Err(invalid(format!("source manifest line {line_number} has an invalid schema")));
        }
        let role = fields[1];
        if !matches!(role, "sample" | "cheap_output" | "pre_features" | "shift_signal") {
            return Err(invalid(format!("unknown source-manifest role {role}")));
        }
        let relative = safe_relative_path(fields[2])?;
        if !seen_paths.insert(fields[2].to_owned()) {
            return Err(invalid(format!("duplicate source-manifest path {}", fields[2])));
        }
        let expected_len = fields[3]
            .parse::<u64>()
            .map_err(|_| invalid(format!("source manifest line {line_number}: invalid byte_len")))?;
        let expected_sha = validate_sha256(fields[4], "source sha256")?;
        let path = checked_source_file(root, &relative)?;
        let (actual_sha, actual_len) = digest_file(&path)?;
        if actual_len != expected_len {
            return Err(invalid(format!(
                "byte length mismatch for {}: expected {expected_len}, got {actual_len}",
                fields[2]
            )));
        }
        if actual_sha != expected_sha {
            return Err(invalid(format!("content digest mismatch for {}", fields[2])));
        }
        role_digests.insert((role.to_owned(), expected_sha));
        count += 1;
    }
    if count == 0 {
        return Err(invalid("source manifest must contain at least one asset"));
    }
    Ok((role_digests, count))
}

fn parse_split_manifest(input: &str) -> Result<BTreeMap<String, String>, String> {
    let mut lines = input.lines();
    if lines.next() != Some("SPLIT_MANIFEST_V1\tgroup_id\tsplit") {
        return Err(invalid("split-manifest header mismatch"));
    }
    let mut assignments = BTreeMap::new();
    for (index, line) in lines.enumerate() {
        let line_number = index + 2;
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 || fields[0] != "SPLIT" {
            return Err(invalid(format!("split manifest line {line_number} has an invalid schema")));
        }
        validate_token(fields[1], "split group_id")?;
        if !matches!(fields[2], "train" | "validation" | "test" | "private_test") {
            return Err(invalid(format!("split manifest line {line_number} has an invalid split")));
        }
        if assignments.insert(fields[1].to_owned(), fields[2].to_owned()).is_some() {
            return Err(invalid(format!("duplicate split-manifest group {}", fields[1])));
        }
    }
    if assignments.is_empty() {
        return Err(invalid("split manifest must contain at least one group"));
    }
    Ok(assignments)
}

fn validate_snapshot(input: &str, meta: &BTreeMap<String, String>) -> Result<(), String> {
    let mut lines = input.lines();
    if lines.next() != Some("SNAPSHOT_HEADER\tkey\tvalue") {
        return Err(invalid("dataset snapshot descriptor header mismatch"));
    }
    let expected: BTreeSet<&str> = ["dataset_id", "dataset_revision", "immutable_ref"]
        .into_iter()
        .collect();
    let mut fields = BTreeMap::new();
    for (index, line) in lines.enumerate() {
        let line_number = index + 2;
        let row: Vec<&str> = line.split('\t').collect();
        if row.len() != 3 || row[0] != "SNAPSHOT" || row[1].is_empty() || row[2].is_empty() {
            return Err(invalid(format!("snapshot line {line_number} has an invalid schema")));
        }
        if fields.insert(row[1].to_owned(), row[2].to_owned()).is_some() {
            return Err(invalid(format!("duplicate snapshot key {}", row[1])));
        }
    }
    let actual: BTreeSet<&str> = fields.keys().map(String::as_str).collect();
    if actual != expected {
        return Err(invalid("dataset snapshot keys must be exactly dataset_id, dataset_revision and immutable_ref"));
    }
    for key in ["dataset_id", "dataset_revision"] {
        if fields.get(key) != meta.get(key) {
            return Err(invalid(format!("dataset snapshot {key} differs from trace metadata")));
        }
    }
    validate_token(fields.get("immutable_ref").map(String::as_str).unwrap_or(""), "immutable_ref")?;
    Ok(())
}

fn parse_lock(input: &str) -> Result<BTreeMap<String, String>, String> {
    let mut lines = input.lines();
    if lines.next() != Some("DVPC_BUNDLE_LOCK_V1\tkey\tsha256") {
        return Err(invalid("bundle-lock header mismatch"));
    }
    let mut lock = BTreeMap::new();
    for (index, line) in lines.enumerate() {
        let line_number = index + 2;
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 || fields[0] != "LOCK" {
            return Err(invalid(format!("bundle lock line {line_number} has an invalid schema")));
        }
        let normalized = validate_sha256(fields[2], "bundle lock digest")?;
        if lock.insert(fields[1].to_owned(), normalized).is_some() {
            return Err(invalid(format!("duplicate bundle-lock field {}", fields[1])));
        }
    }
    let expected: BTreeSet<&str> = LOCK_KEYS.iter().copied().collect();
    let actual: BTreeSet<&str> = lock.keys().map(String::as_str).collect();
    if actual != expected {
        return Err(invalid("bundle lock must contain the exact required digest key set"));
    }
    Ok(lock)
}

fn check_locked_digest(
    lock: &BTreeMap<String, String>,
    key: &str,
    bytes: &[u8],
) -> Result<String, String> {
    let actual = sha256_hex(bytes);
    if lock.get(key).map(String::as_str) != Some(actual.as_str()) {
        return Err(invalid(format!("bundle lock mismatch for {key}")));
    }
    Ok(actual)
}

fn verify_bundle(
    dataset_root: &Path,
    observations: &[u8],
    evaluations: &[u8],
    source_manifest: &[u8],
    split_manifest: &[u8],
    snapshot: &[u8],
    utility_spec: &[u8],
    verifier_source: &[u8],
    bundle_lock: &[u8],
) -> Result<Verified, String> {
    let root = dataset_root
        .canonicalize()
        .map_err(|error| invalid(format!("cannot resolve dataset root: {error}")))?;
    if !root.is_dir() {
        return Err(invalid("dataset root must be a directory"));
    }

    let observation_text = std::str::from_utf8(observations)
        .map_err(|_| invalid("observation file must be UTF-8"))?;
    let evaluation_text = std::str::from_utf8(evaluations)
        .map_err(|_| invalid("evaluation file must be UTF-8"))?;
    let source_manifest_text = std::str::from_utf8(source_manifest)
        .map_err(|_| invalid("source manifest must be UTF-8"))?;
    let split_manifest_text = std::str::from_utf8(split_manifest)
        .map_err(|_| invalid("split manifest must be UTF-8"))?;
    let snapshot_text = std::str::from_utf8(snapshot)
        .map_err(|_| invalid("snapshot descriptor must be UTF-8"))?;
    let utility_text = std::str::from_utf8(utility_spec)
        .map_err(|_| invalid("utility specification must be UTF-8"))?;
    let lock_text = std::str::from_utf8(bundle_lock)
        .map_err(|_| invalid("bundle lock must be UTF-8"))?;

    let lock = parse_lock(lock_text)?;
    let observation_sha = check_locked_digest(&lock, "observations_sha256", observations)?;
    let evaluation_sha = check_locked_digest(&lock, "evaluations_sha256", evaluations)?;
    let source_manifest_sha =
        check_locked_digest(&lock, "source_manifest_sha256", source_manifest)?;
    let split_manifest_sha = check_locked_digest(&lock, "split_manifest_sha256", split_manifest)?;
    let snapshot_sha = check_locked_digest(&lock, "snapshot_sha256", snapshot)?;
    let utility_sha = check_locked_digest(&lock, "utility_spec_sha256", utility_spec)?;
    let verifier_sha = check_locked_digest(&lock, "verifier_source_sha256", verifier_source)?;
    let lock_sha = sha256_hex(bundle_lock);

    let obs = parse_table(observation_text, OBS_HEADER, "OBS")?;
    let eval = parse_table(evaluation_text, EVAL_HEADER, "EVAL")?;
    if obs.meta != eval.meta {
        return Err(invalid(
            "observation/evaluation metadata mismatch; paired files must bind one exact bundle",
        ));
    }

    let linked_digests = [
        ("dataset_snapshot_sha256", snapshot_sha.as_str()),
        ("source_manifest_sha256", source_manifest_sha.as_str()),
        ("split_manifest_sha256", split_manifest_sha.as_str()),
        ("utility_spec_sha256", utility_sha.as_str()),
    ];
    for (field, actual) in linked_digests {
        if obs.meta.get(field).map(String::as_str) != Some(actual) {
            return Err(invalid(format!(
                "trace META {field} does not match the exact locked file bytes"
            )));
        }
    }
    if !utility_text.lines().next().is_some_and(|line| line == "UTILITY_SPEC_V1") {
        return Err(invalid("utility specification must start with UTILITY_SPEC_V1"));
    }
    validate_snapshot(snapshot_text, &obs.meta)?;

    let (role_digests, source_entries) = parse_source_manifest(&root, source_manifest_text)?;
    let split_assignments = parse_split_manifest(split_manifest_text)?;

    let mut observation_by_trace = BTreeMap::new();
    let mut group_assignments = BTreeMap::new();
    let mut group_steps: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();

    for (index, row) in obs.rows.iter().enumerate() {
        let row_number = index + 1;
        validate_token(&row[1], "trace_id")?;
        validate_token(&row[2], "dataset_id")?;
        validate_token(&row[3], "dataset_revision")?;
        validate_token(&row[5], "group_id")?;
        if row[2] != obs.meta["dataset_id"] || row[3] != obs.meta["dataset_revision"] {
            return Err(invalid(format!(
                "observation row {row_number}: dataset identity differs from metadata"
            )));
        }
        if !matches!(row[4].as_str(), "train" | "validation" | "test" | "private_test") {
            return Err(invalid(format!("observation row {row_number}: invalid split")));
        }
        let sample_sha = validate_sha256(&row[7], "sample_sha256")?;
        let cheap_sha = validate_sha256(&row[8], "cheap_output_sha256")?;
        let features_sha = validate_sha256(&row[9], "pre_features_sha256")?;
        let utility_row_sha = validate_sha256(&row[10], "utility_spec_sha256")?;
        if utility_row_sha != utility_sha {
            return Err(invalid(format!(
                "observation row {row_number}: utility digest mismatch"
            )));
        }
        for (role, digest) in [
            ("sample", sample_sha.clone()),
            ("cheap_output", cheap_sha),
            ("pre_features", features_sha),
        ] {
            if !role_digests.contains(&(role.to_owned(), digest)) {
                return Err(invalid(format!(
                    "observation row {row_number}: {role} digest does not resolve in verified source manifest"
                )));
            }
        }
        if row[15] != "-" {
            let shift_sha = validate_sha256(&row[15], "shift_signal_sha256")?;
            if !role_digests.contains(&("shift_signal".to_owned(), shift_sha)) {
                return Err(invalid(format!(
                    "observation row {row_number}: shift-signal digest does not resolve in verified source manifest"
                )));
            }
        }

        let trace_id = row[1].clone();
        if observation_by_trace.insert(trace_id.clone(), sample_sha).is_some() {
            return Err(invalid(format!("duplicate observation trace_id {trace_id}")));
        }
        let group = row[5].clone();
        match group_assignments.get(&group) {
            Some(previous) if previous != &row[4] => {
                return Err(invalid(format!("source group {group} crosses splits")));
            }
            None => {
                group_assignments.insert(group.clone(), row[4].clone());
            }
            _ => {}
        }
        let step = row[6]
            .parse::<u64>()
            .map_err(|_| invalid(format!("observation row {row_number}: invalid step_index")))?;
        if !group_steps.entry(group.clone()).or_default().insert(step) {
            return Err(invalid(format!("duplicate step_index {step} in group {group}")));
        }
    }

    for (group, steps) in &group_steps {
        for (expected, observed) in steps.iter().enumerate() {
            if *observed != expected as u64 {
                return Err(invalid(format!(
                    "group {group} has non-contiguous step index: expected {expected}, got {observed}"
                )));
            }
        }
    }
    if split_assignments != group_assignments {
        return Err(invalid(
            "split manifest must exactly equal observed group-to-split assignments; missing, extra or conflicting groups fail closed",
        ));
    }

    let mut evaluation_seen = BTreeSet::new();
    for (index, row) in eval.rows.iter().enumerate() {
        validate_token(&row[1], "evaluation trace_id")?;
        let digest = validate_sha256(&row[2], "evaluation sample_sha256")?;
        if !evaluation_seen.insert(row[1].clone()) {
            return Err(invalid(format!("duplicate evaluation trace_id {}", row[1])));
        }
        match observation_by_trace.get(&row[1]) {
            Some(expected) if expected == &digest => {}
            Some(_) => {
                return Err(invalid(format!(
                    "evaluation row {} sample digest differs from its observation",
                    index + 1
                )));
            }
            None => {
                return Err(invalid(format!(
                    "evaluation has unmatched trace_id {}",
                    row[1]
                )));
            }
        }
    }
    if evaluation_seen.len() != observation_by_trace.len() {
        return Err(invalid("observation/evaluation rows are not a one-to-one paired set"));
    }

    let mut receipt = String::new();
    let _ = writeln!(receipt, "RECEIPT\tformat\t{RECEIPT_VERSION}");
    let _ = writeln!(receipt, "RECEIPT\tstatus\tPASS_BYTES_AND_MANIFESTS");
    let _ = writeln!(receipt, "RECEIPT\tobservations_sha256\t{observation_sha}");
    let _ = writeln!(receipt, "RECEIPT\tevaluations_sha256\t{evaluation_sha}");
    let _ = writeln!(receipt, "RECEIPT\tsource_manifest_sha256\t{source_manifest_sha}");
    let _ = writeln!(receipt, "RECEIPT\tsplit_manifest_sha256\t{split_manifest_sha}");
    let _ = writeln!(receipt, "RECEIPT\tdataset_snapshot_sha256\t{snapshot_sha}");
    let _ = writeln!(receipt, "RECEIPT\tutility_spec_sha256\t{utility_sha}");
    let _ = writeln!(receipt, "RECEIPT\tverifier_source_sha256\t{verifier_sha}");
    let _ = writeln!(receipt, "RECEIPT\tbundle_lock_sha256\t{lock_sha}");
    let _ = writeln!(receipt, "RECEIPT\tobservation_rows\t{}", obs.rows.len());
    let _ = writeln!(receipt, "RECEIPT\tevaluation_rows\t{}", eval.rows.len());
    let _ = writeln!(receipt, "RECEIPT\tsource_entries_verified\t{source_entries}");
    let _ = writeln!(receipt, "RECEIPT\tgroups_verified\t{}", group_assignments.len());
    let _ = writeln!(
        receipt,
        "RECEIPT\tlicense_scope_declaration\t{}",
        obs.meta["license_scope"]
    );
    let _ = writeln!(receipt, "RECEIPT\tlegal_rights_verdict\tNOT_ASSESSED");
    let _ = writeln!(receipt, "RECEIPT\tstatistical_independence_claim\tNONE");
    let _ = writeln!(receipt, "RECEIPT\tperformance_claim\tNONE");

    Ok(Verified {
        receipt,
        observation_rows: obs.rows.len(),
        evaluation_rows: eval.rows.len(),
        groups: group_assignments.len(),
        source_entries,
    })
}

fn read_bytes(path: &str, description: &str) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| invalid(format!("cannot read {description} {path}: {error}")))
}

fn run_cli(args: &[String]) -> Result<String, String> {
    if args.len() != 10 {
        return Err(
            "usage: vision-dvpc-integrity-receipt DATASET_ROOT OBS.tsv EVAL.tsv SOURCE_MANIFEST.tsv SPLIT_MANIFEST.tsv SNAPSHOT.tsv UTILITY_SPEC.txt VERIFIER_SOURCE.rs BUNDLE_LOCK.tsv".to_owned(),
        );
    }
    let observations = read_bytes(&args[2], "observations")?;
    let evaluations = read_bytes(&args[3], "evaluations")?;
    let source_manifest = read_bytes(&args[4], "source manifest")?;
    let split_manifest = read_bytes(&args[5], "split manifest")?;
    let snapshot = read_bytes(&args[6], "snapshot descriptor")?;
    let utility_spec = read_bytes(&args[7], "utility specification")?;
    let verifier_source = read_bytes(&args[8], "verifier source")?;
    let bundle_lock = read_bytes(&args[9], "bundle lock")?;
    let verified = verify_bundle(
        Path::new(&args[1]),
        &observations,
        &evaluations,
        &source_manifest,
        &split_manifest,
        &snapshot,
        &utility_spec,
        &verifier_source,
        &bundle_lock,
    )?;
    debug_assert_eq!(verified.observation_rows, verified.evaluation_rows);
    debug_assert!(verified.groups > 0);
    debug_assert!(verified.source_entries > 0);
    Ok(verified.receipt)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run_cli(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("REJECT\treason\t{}", error.replace('\t', " ").replace('\n', " "));
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
        observations: Vec<u8>,
        evaluations: Vec<u8>,
        source_manifest: Vec<u8>,
        split_manifest: Vec<u8>,
        snapshot: Vec<u8>,
        utility_spec: Vec<u8>,
        verifier_source: Vec<u8>,
        bundle_lock: Vec<u8>,
    }

    impl Fixture {
        fn new() -> Self {
            let root = env::temp_dir().join(format!(
                "vis-dvpc-012-{}-{}",
                std::process::id(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).expect("temporary fixture root");
            let mut fixture = Self {
                root,
                observations: Vec::new(),
                evaluations: Vec::new(),
                source_manifest: Vec::new(),
                split_manifest: Vec::new(),
                snapshot: b"SNAPSHOT_HEADER\tkey\tvalue\nSNAPSHOT\tdataset_id\tsynthetic-public-fixture\nSNAPSHOT\tdataset_revision\tr1\nSNAPSHOT\timmutable_ref\tfixture-v1\n".to_vec(),
                utility_spec: b"UTILITY_SPEC_V1\nname=fixture-utility\nrange=0..1\nhigher_is_better=true\n".to_vec(),
                verifier_source: b"// exact verifier source bytes for synthetic test fixture\n".to_vec(),
                bundle_lock: Vec::new(),
            };

            let assets = [
                ("sample", "sample-a.bin", b"raw sample a\n".as_slice()),
                ("cheap_output", "cheap-a.bin", b"cheap output a\n".as_slice()),
                ("pre_features", "features-a.bin", b"pre-features a\n".as_slice()),
                ("sample", "sample-b.bin", b"raw sample b\n".as_slice()),
                ("cheap_output", "cheap-b.bin", b"cheap output b\n".as_slice()),
                ("pre_features", "features-b.bin", b"pre-features b\n".as_slice()),
            ];
            let mut manifest = String::from("SOURCE_MANIFEST_V1\trole\tpath\tbyte_len\tsha256\n");
            for (role, path, bytes) in assets {
                fs::write(fixture.root.join(path), bytes).expect("write synthetic source asset");
                let _ = writeln!(
                    manifest,
                    "SOURCE\t{role}\t{path}\t{}\t{}",
                    bytes.len(),
                    sha256_hex(bytes)
                );
            }
            fixture.source_manifest = manifest.into_bytes();
            fixture.split_manifest =
                b"SPLIT_MANIFEST_V1\tgroup_id\tsplit\nSPLIT\tseq-a\ttrain\nSPLIT\tseq-b\ttest\n"
                    .to_vec();

            let snapshot_sha = sha256_hex(&fixture.snapshot);
            let source_manifest_sha = sha256_hex(&fixture.source_manifest);
            let split_manifest_sha = sha256_hex(&fixture.split_manifest);
            let utility_sha = sha256_hex(&fixture.utility_spec);
            let sample_a = sha256_hex(b"raw sample a\n");
            let cheap_a = sha256_hex(b"cheap output a\n");
            let features_a = sha256_hex(b"pre-features a\n");
            let sample_b = sha256_hex(b"raw sample b\n");
            let cheap_b = sha256_hex(b"cheap output b\n");
            let features_b = sha256_hex(b"pre-features b\n");

            let metadata = format!(
                "META\tformat\t{FORMAT_VERSION}\nMETA\tdataset_id\tsynthetic-public-fixture\nMETA\tdataset_revision\tr1\nMETA\tdataset_snapshot_sha256\t{snapshot_sha}\nMETA\tsource_manifest_sha256\t{source_manifest_sha}\nMETA\tsplit_manifest_sha256\t{split_manifest_sha}\nMETA\tutility_spec_sha256\t{utility_sha}\nMETA\tlicense_id\tCC0-1.0\nMETA\tlicense_scope\tsynthetic_public_fixture\nMETA\thigher_is_better\ttrue\n"
            );
            let obs_header = format!("{}\n", OBS_HEADER.join("\t"));
            let obs_rows = format!(
                "OBS\ttrace-a\tsynthetic-public-fixture\tr1\ttrain\tseq-a\t0\t{sample_a}\t{cheap_a}\t{features_a}\t{utility_sha}\t1000\t100\t0.1\t0.2\t-\nOBS\ttrace-b\tsynthetic-public-fixture\tr1\ttest\tseq-b\t0\t{sample_b}\t{cheap_b}\t{features_b}\t{utility_sha}\t1000\t100\t0.2\t0.1\t-\n"
            );
            fixture.observations = format!("{metadata}{obs_header}{obs_rows}").into_bytes();

            let eval_header = format!("{}\n", EVAL_HEADER.join("\t"));
            let eval_rows = format!(
                "EVAL\ttrace-a\t{sample_a}\t0.5\t0.6\t0\t0\t900\t90\tin_domain\tnone\nEVAL\ttrace-b\t{sample_b}\t0.4\t0.7\t0\t0\t950\t95\tin_domain\tnone\n"
            );
            fixture.evaluations = format!("{metadata}{eval_header}{eval_rows}").into_bytes();
            fixture.refresh_lock();
            fixture.sync_inputs();
            fixture
        }

        fn lock_bytes(&self) -> Vec<u8> {
            let entries = [
                ("observations_sha256", sha256_hex(&self.observations)),
                ("evaluations_sha256", sha256_hex(&self.evaluations)),
                ("source_manifest_sha256", sha256_hex(&self.source_manifest)),
                ("split_manifest_sha256", sha256_hex(&self.split_manifest)),
                ("snapshot_sha256", sha256_hex(&self.snapshot)),
                ("utility_spec_sha256", sha256_hex(&self.utility_spec)),
                ("verifier_source_sha256", sha256_hex(&self.verifier_source)),
            ];
            let mut output = String::from("DVPC_BUNDLE_LOCK_V1\tkey\tsha256\n");
            for (key, digest) in entries {
                let _ = writeln!(output, "LOCK\t{key}\t{digest}");
            }
            output.into_bytes()
        }

        fn refresh_lock(&mut self) {
            self.bundle_lock = self.lock_bytes();
        }

        fn sync_inputs(&self) {
            fs::write(self.root.join("obs.tsv"), &self.observations).expect("write observations");
            fs::write(self.root.join("eval.tsv"), &self.evaluations).expect("write evaluations");
            fs::write(self.root.join("source-manifest.tsv"), &self.source_manifest)
                .expect("write source manifest");
            fs::write(self.root.join("split-manifest.tsv"), &self.split_manifest)
                .expect("write split manifest");
            fs::write(self.root.join("snapshot.tsv"), &self.snapshot).expect("write snapshot");
            fs::write(self.root.join("utility-spec.txt"), &self.utility_spec)
                .expect("write utility spec");
            fs::write(self.root.join("verifier-source.rs"), &self.verifier_source)
                .expect("write verifier source");
            fs::write(self.root.join("bundle-lock.tsv"), &self.bundle_lock).expect("write bundle lock");
        }

        fn set_meta(&mut self, key: &str, value: &str) {
            self.observations = replace_meta(&self.observations, key, value);
            self.evaluations = replace_meta(&self.evaluations, key, value);
        }

        fn resign_split_manifest(&mut self, text: &str) {
            self.split_manifest = text.as_bytes().to_vec();
            let digest = sha256_hex(&self.split_manifest);
            self.set_meta("split_manifest_sha256", &digest);
            self.refresh_lock();
            self.sync_inputs();
        }

        fn resign_source_manifest(&mut self, text: &str) {
            self.source_manifest = text.as_bytes().to_vec();
            let digest = sha256_hex(&self.source_manifest);
            self.set_meta("source_manifest_sha256", &digest);
            self.refresh_lock();
            self.sync_inputs();
        }

        fn resign_snapshot(&mut self, bytes: &[u8]) {
            self.snapshot = bytes.to_vec();
            self.refresh_lock();
            self.sync_inputs();
        }

        fn verify(&self) -> Result<String, String> {
            let args = vec![
                "vision-dvpc-integrity-receipt".to_owned(),
                self.root.display().to_string(),
                self.root.join("obs.tsv").display().to_string(),
                self.root.join("eval.tsv").display().to_string(),
                self.root.join("source-manifest.tsv").display().to_string(),
                self.root.join("split-manifest.tsv").display().to_string(),
                self.root.join("snapshot.tsv").display().to_string(),
                self.root.join("utility-spec.txt").display().to_string(),
                self.root.join("verifier-source.rs").display().to_string(),
                self.root.join("bundle-lock.tsv").display().to_string(),
            ];
            run_cli(&args)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn replace_meta(input: &[u8], key: &str, value: &str) -> Vec<u8> {
        let text = std::str::from_utf8(input).expect("fixture is UTF-8");
        let prefix = format!("META\t{key}\t");
        let lines: Vec<String> = text
            .lines()
            .map(|line| {
                if line.starts_with(&prefix) {
                    format!("{prefix}{value}")
                } else {
                    line.to_owned()
                }
            })
            .collect();
        format!("{}\n", lines.join("\n")).into_bytes()
    }

    #[test]
    fn sha256_matches_standard_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn valid_content_addressed_bundle_passes_and_receipt_is_explicit() {
        let fixture = Fixture::new();
        let output = fixture.verify().expect("valid synthetic bundle");
        assert!(output.contains("RECEIPT\tstatus\tPASS_BYTES_AND_MANIFESTS"));
        assert!(output.contains("RECEIPT\tobservations_sha256\t"));
        assert!(output.contains("RECEIPT\tevaluations_sha256\t"));
        assert!(output.contains("RECEIPT\tsource_entries_verified\t6"));
        assert!(output.contains("RECEIPT\tgroups_verified\t2"));
        assert!(output.contains("RECEIPT\tlegal_rights_verdict\tNOT_ASSESSED"));
        assert!(output.contains("RECEIPT\tperformance_claim\tNONE"));
        assert_eq!(output, fixture.verify().expect("deterministic replay"));
    }

    #[test]
    fn locked_observation_tampering_fails() {
        let mut fixture = Fixture::new();
        let changed = String::from_utf8(fixture.observations.clone())
            .expect("fixture UTF-8")
            .replace("\t0.1\t0.2\t-\n", "\t0.9\t0.2\t-\n");
        fixture.observations = changed.into_bytes();
        fixture.sync_inputs();
        assert!(fixture.verify().unwrap_err().contains("bundle lock mismatch for observations_sha256"));
    }

    #[test]
    fn locked_evaluation_tampering_fails() {
        let mut fixture = Fixture::new();
        fixture.evaluations = String::from_utf8(fixture.evaluations.clone())
            .expect("fixture UTF-8")
            .replace("\t0.5\t0.6\t", "\t0.9\t0.6\t")
            .into_bytes();
        fixture.sync_inputs();
        assert!(fixture.verify().unwrap_err().contains("bundle lock mismatch for evaluations_sha256"));
    }

    #[test]
    fn changed_source_bytes_fail_even_when_manifest_is_unchanged() {
        let fixture = Fixture::new();
        fs::write(fixture.root.join("sample-a.bin"), b"tampered sample\n")
            .expect("tamper source asset");
        assert!(fixture.verify().unwrap_err().contains("byte length mismatch") || fixture.verify().unwrap_err().contains("content digest mismatch"));
    }

    #[test]
    fn byte_length_mismatch_fails_after_bundle_recommit() {
        let mut fixture = Fixture::new();
        let text = String::from_utf8(fixture.source_manifest.clone())
            .expect("fixture UTF-8")
            .replace("13\t", "14\t");
        fixture.resign_source_manifest(&text);
        assert!(fixture.verify().unwrap_err().contains("byte length mismatch"));
    }

    #[test]
    fn path_traversal_is_rejected_even_when_lock_and_metadata_match() {
        let mut fixture = Fixture::new();
        let text = String::from_utf8(fixture.source_manifest.clone())
            .expect("fixture UTF-8")
            .replace("sample-a.bin", "../sample-a.bin");
        fixture.resign_source_manifest(&text);
        assert!(fixture.verify().unwrap_err().contains("normal relative components"));
    }

    #[test]
    fn duplicate_source_manifest_paths_fail_closed() {
        let mut fixture = Fixture::new();
        let text = String::from_utf8(fixture.source_manifest.clone())
            .expect("fixture UTF-8")
            .replace("cheap-a.bin", "sample-a.bin");
        fixture.resign_source_manifest(&text);
        assert!(fixture.verify().unwrap_err().contains("duplicate source-manifest path"));
    }

    #[test]
    fn wrong_role_does_not_satisfy_observation_digest() {
        let mut fixture = Fixture::new();
        let text = String::from_utf8(fixture.source_manifest.clone())
            .expect("fixture UTF-8")
            .replace("SOURCE\tsample\tsample-a.bin", "SOURCE\tcheap_output\tsample-a.bin");
        fixture.resign_source_manifest(&text);
        assert!(fixture.verify().unwrap_err().contains("sample digest does not resolve"));
    }

    #[test]
    fn source_group_split_mismatch_fails_when_every_file_is_recommitted() {
        let mut fixture = Fixture::new();
        fixture.resign_split_manifest(
            "SPLIT_MANIFEST_V1\tgroup_id\tsplit\nSPLIT\tseq-a\ttest\nSPLIT\tseq-b\ttest\n",
        );
        assert!(fixture.verify().unwrap_err().contains("split manifest must exactly equal"));
    }

    #[test]
    fn split_manifest_extra_group_fails_closed() {
        let mut fixture = Fixture::new();
        fixture.resign_split_manifest(
            "SPLIT_MANIFEST_V1\tgroup_id\tsplit\nSPLIT\tseq-a\ttrain\nSPLIT\tseq-b\ttest\nSPLIT\tseq-c\tvalidation\n",
        );
        assert!(fixture.verify().unwrap_err().contains("split manifest must exactly equal"));
    }

    #[test]
    fn split_manifest_duplicate_group_fails_closed() {
        let mut fixture = Fixture::new();
        fixture.resign_split_manifest(
            "SPLIT_MANIFEST_V1\tgroup_id\tsplit\nSPLIT\tseq-a\ttrain\nSPLIT\tseq-a\ttrain\nSPLIT\tseq-b\ttest\n",
        );
        assert!(fixture.verify().unwrap_err().contains("duplicate split-manifest group"));
    }

    #[test]
    fn tampered_snapshot_is_rejected_after_lock_is_recommitted() {
        let mut fixture = Fixture::new();
        fixture.resign_snapshot(
            b"SNAPSHOT_HEADER\tkey\tvalue\nSNAPSHOT\tdataset_id\tsynthetic-public-fixture\nSNAPSHOT\tdataset_revision\tr2\nSNAPSHOT\timmutable_ref\tfixture-v1\n",
        );
        assert!(fixture.verify().unwrap_err().contains("trace META dataset_snapshot_sha256 does not match"));
    }

    #[test]
    fn mismatched_sample_pair_fails_after_evaluation_recommit() {
        let mut fixture = Fixture::new();
        let sample_b = sha256_hex(b"raw sample b\n");
        fixture.evaluations = String::from_utf8(fixture.evaluations.clone())
            .expect("fixture UTF-8")
            .replace(&sample_b, &sha256_hex(b"wrong sample\n"))
            .into_bytes();
        fixture.refresh_lock();
        fixture.sync_inputs();
        assert!(fixture.verify().unwrap_err().contains("sample digest differs"));
    }

    #[test]
    fn symlinked_source_asset_is_rejected() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let fixture = Fixture::new();
            fs::remove_file(fixture.root.join("sample-a.bin")).expect("remove fixture asset");
            symlink(fixture.root.join("sample-b.bin"), fixture.root.join("sample-a.bin"))
                .expect("create symlink");
            assert!(fixture.verify().unwrap_err().contains("symlink in manifested source path"));
        }
    }

    #[test]
    fn malformed_lock_and_missing_digests_fail_closed() {
        let mut fixture = Fixture::new();
        fixture.bundle_lock = b"DVPC_BUNDLE_LOCK_V1\tkey\tsha256\nLOCK\tobservations_sha256\tbad\n".to_vec();
        fixture.sync_inputs();
        assert!(fixture.verify().unwrap_err().contains("bundle lock must contain the exact required"));
    }

    #[test]
    fn absent_shift_signal_is_allowed_but_declared_digest_must_resolve() {
        let mut fixture = Fixture::new();
        let shift = sha256_hex(b"missing shift signal\n");
        let changed = String::from_utf8(fixture.observations.clone())
            .expect("fixture UTF-8")
            .replace("\t-\n", &format!("\t{shift}\n"));
        fixture.observations = changed.into_bytes();
        fixture.refresh_lock();
        fixture.sync_inputs();
        assert!(fixture.verify().unwrap_err().contains("shift-signal digest does not resolve"));
    }

    #[test]
    fn bundle_lock_rejects_wrong_verifier_source_bytes() {
        let mut fixture = Fixture::new();
        fixture.verifier_source.extend_from_slice(b"changed");
        fixture.sync_inputs();
        assert!(fixture.verify().unwrap_err().contains("bundle lock mismatch for verifier_source_sha256"));
    }
}
