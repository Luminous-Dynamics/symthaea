//! Offline verification of REA provider-smoke evidence bundles.
//!
//! This verifies artifact bytes against their receipt. It does not verify
//! firmware behavior or prove that REA CLI/MCP parity was run.
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

const BENCHMARK: &str = "ENG-REA-005-REAL-PROVIDER";
const PROVIDER_VERSION: &str = "26.6.4";
const RECEIPT_SCOPE: &str =
    "Pinned Unblob CLI on owned inert nested gzip/tar/text fixtures only; no REA, recovered firmware execution, emulator, or hardware qualification.";
const RECEIPT_SCHEMA: &str = "symthaea.rea-provider-smoke-receipt.v1";
const BUNDLE_MANIFEST_SCHEMA: &str = "symthaea.rea-evidence-bundle-manifest.v1";
const PROVIDER_META_SCHEMA: &str = "symthaea.rea-provider-metamorphic-receipt.v1";

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct VerifiedEvidence {
    pub status: &'static str,
    pub evidence_tier: &'static str,
    pub receipt_sha256: String,
    pub verification_mode: &'static str,
    pub receipt_digest_anchor_matched: bool,
    pub source_head_sha: String,
    pub checkout_commit_sha: String,
    pub provider_version: String,
    pub raw_reports_verified: usize,
    pub equivalent_provider_runs: usize,
    pub semantic_mutations_verified: usize,
    pub provider_metamorphic_variants_verified: u64,
    pub rea_cli_mcp_parity: bool,
    pub firmware_execution: bool,
    pub emulation: bool,
    pub device_qualification: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct VerifyError(pub String);

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for VerifyError {}

type VResult<T> = Result<T, VerifyError>;

fn err(message: impl Into<String>) -> VerifyError {
    VerifyError(message.into())
}

fn required<'a>(value: &'a Value, path: &str) -> VResult<&'a Value> {
    let mut current = value;
    for part in path.split('.') {
        current = current
            .get(part)
            .ok_or_else(|| err(format!("receipt field missing: {path}")))?;
    }
    Ok(current)
}

fn text_field(value: &Value, path: &str) -> VResult<String> {
    required(value, path)?
        .as_str()
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| err(format!("receipt field must be a non-empty string: {path}")))
}

fn digest_field(value: &Value, path: &str) -> VResult<String> {
    let digest = text_field(value, path)?;
    if !is_digest(&digest) {
        return Err(err(format!("receipt field is not lowercase SHA-256: {path}")));
    }
    Ok(digest)
}

fn is_digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_git_sha(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn ensure_regular_non_symlink(path: &Path, label: &str) -> VResult<()> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| err(format!("cannot inspect {label} {}: {e}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(err(format!("{label} must be a regular non-symlink file: {}", path.display())));
    }
    Ok(())
}

fn read_bounded(path: &Path, max_bytes: u64, label: &str) -> VResult<Vec<u8>> {
    ensure_regular_non_symlink(path, label)?;
    let metadata = fs::metadata(path)
        .map_err(|e| err(format!("cannot stat {label} {}: {e}", path.display())))?;
    if metadata.len() > max_bytes {
        return Err(err(format!("{label} exceeds the {max_bytes}-byte verification limit")));
    }
    let mut file = File::open(path)
        .map_err(|e| err(format!("cannot open {label} {}: {e}", path.display())))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| err(format!("cannot read {label} {}: {e}", path.display())))?;
    if bytes.len() as u64 > max_bytes {
        return Err(err(format!("{label} grew beyond the {max_bytes}-byte verification limit")));
    }
    Ok(bytes)
}

fn hash_file(path: &Path) -> VResult<String> {
    ensure_regular_non_symlink(path, "evidence file")?;
    let mut file = File::open(path)
        .map_err(|e| err(format!("cannot open evidence file {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| err(format!("cannot read evidence file {}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn verify_digest_at(path: &Path, expected: &str, label: &str) -> VResult<()> {
    if !is_digest(expected) {
        return Err(err(format!("{label}: invalid SHA-256 string")));
    }
    let actual = hash_file(path)?;
    if actual != expected {
        return Err(err(format!(
            "{label}: digest mismatch; receipt={expected}, actual={actual}"
        )));
    }
    Ok(())
}

fn read_digest_file(path: &Path, label: &str) -> VResult<String> {
    let bytes = read_bounded(path, 1024, label)?;
    let raw = std::str::from_utf8(&bytes)
        .map_err(|e| err(format!("{label} is not UTF-8: {e}")))?;
    let digest = raw.trim();
    if !is_digest(digest) {
        return Err(err(format!("{label}: expected one lowercase SHA-256 line")));
    }
    Ok(digest.to_owned())
}

fn verify_digest_file(path: &Path, expected: &str, label: &str) -> VResult<()> {
    let actual = read_digest_file(path, label)?;
    if actual != expected {
        return Err(err(format!("{label}: recorded digest mismatch")));
    }
    Ok(())
}

fn find_files_named(root: &Path, name: &str, output: &mut Vec<PathBuf>) -> VResult<()> {
    let root_metadata = fs::symlink_metadata(root)
        .map_err(|e| err(format!("cannot inspect extraction directory {}: {e}", root.display())))?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err(err(format!("extraction search root must be a real directory: {}", root.display())));
    }
    let entries = fs::read_dir(root)
        .map_err(|e| err(format!("cannot enumerate evidence directory {}: {e}", root.display())))?;
    for item in entries {
        let item = item.map_err(|e| err(format!("cannot read directory entry: {e}")))?;
        let kind = item
            .file_type()
            .map_err(|e| err(format!("cannot inspect entry {}: {e}", item.path().display())))?;
        // Deliberately do not follow symlinks while reading untrusted artifacts.
        if kind.is_symlink() {
            continue;
        }
        if kind.is_dir() {
            find_files_named(&item.path(), name, output)?;
        } else if kind.is_file() && item.file_name() == name {
            output.push(item.path());
        }
    }
    Ok(())
}

fn collect_bundle_paths(root: &Path, current: &Path, output: &mut Vec<String>) -> VResult<()> {
    let entries = fs::read_dir(current)
        .map_err(|e| err(format!("cannot enumerate bundle directory {}: {e}", current.display())))?;
    for item in entries {
        let item = item.map_err(|e| err(format!("cannot read bundle directory entry: {e}")))?;
        let kind = item.file_type()
            .map_err(|e| err(format!("cannot inspect bundle entry {}: {e}", item.path().display())))?;
        if kind.is_symlink() {
            return Err(err(format!("bundle contains a symlink: {}", item.path().display())));
        }
        if kind.is_dir() {
            collect_bundle_paths(root, &item.path(), output)?;
        } else if kind.is_file() {
            let relative = item.path().strip_prefix(root)
                .map_err(|e| err(format!("bundle path escaped root: {e}")))?
                .to_string_lossy()
                .replace('\\', "/");
            if relative.is_empty() || relative.starts_with('/') || relative.split('/').any(|part| part == "." || part == ".." || part.is_empty()) {
                return Err(err(format!("bundle contains a non-normal relative path: {relative}")));
            }
            output.push(relative);
        } else {
            return Err(err(format!("bundle contains a non-regular entry: {}", item.path().display())));
        }
    }
    Ok(())
}

fn verify_bundle_manifest(root: &Path, expected_digest: &str) -> VResult<()> {
    let manifest_path = root.join("bundle-manifest.json");
    verify_digest_at(&manifest_path, expected_digest, "bundle manifest")?;
    let raw = read_bounded(&manifest_path, 4 * 1024 * 1024, "bundle manifest")?;
    let manifest: Value = serde_json::from_slice(&raw)
        .map_err(|e| err(format!("bundle manifest is not valid JSON: {e}")))?;
    if text_field(&manifest, "schema_version")? != BUNDLE_MANIFEST_SCHEMA {
        return Err(err("bundle manifest schema version mismatch"));
    }
    let files = required(&manifest, "files")?
        .as_array()
        .ok_or_else(|| err("bundle manifest files must be an array"))?;
    let mut manifest_paths = Vec::with_capacity(files.len());
    for entry in files {
        let relative = text_field(entry, "path")?;
        let digest = digest_field(entry, "sha256")?;
        let parsed = Path::new(&relative);
        if parsed.is_absolute() || parsed.components().any(|component| !matches!(component, std::path::Component::Normal(_))) {
            return Err(err(format!("bundle manifest path is not normalized and relative: {relative}")));
        }
        if relative == "receipt.json" || relative == "bundle-manifest.json" {
            return Err(err(format!("bundle manifest must not list its own receipt or manifest: {relative}")));
        }
        if manifest_paths.last().is_some_and(|last: &String| last >= &relative) {
            return Err(err("bundle manifest paths must be unique and sorted"));
        }
        verify_digest_at(&root.join(&relative), &digest, &format!("manifest file {relative}"))?;
        manifest_paths.push(relative);
    }
    let mut actual_paths = Vec::new();
    collect_bundle_paths(root, root, &mut actual_paths)?;
    actual_paths.retain(|p| p != "receipt.json" && p != "bundle-manifest.json");
    actual_paths.sort();
    if actual_paths != manifest_paths {
        return Err(err("bundle manifest file set does not exactly match the artifact bundle"));
    }
    Ok(())
}

fn verify_payload(root: &Path, lane: &str, expected: &str) -> VResult<()> {
    let extracted = root.join(lane).join("extracted");
    let mut matches = Vec::new();
    find_files_named(&extracted, "payload.txt", &mut matches)?;
    if matches.len() != 1 {
        return Err(err(format!(
            "{lane}: expected exactly one extracted payload.txt, found {}",
            matches.len()
        )));
    }
    verify_digest_at(&matches[0], expected, &format!("{lane} extracted payload"))?;
    Ok(())
}

/// Verify a downloaded provider-smoke bundle against checked-out source files.
///
/// expected_head must come from an independently trusted source (such as the
/// PR or release revision that produced the artifact), not the receipt itself.
/// source_root points at the benchmarks/rea-determinism/v1 directory.
pub fn verify_bundle(
    artifact_root: &Path,
    source_root: &Path,
    expected_head: &str,
    expected_receipt_sha256: Option<&str>,
) -> VResult<VerifiedEvidence> {
    if !is_git_sha(expected_head) {
        return Err(err("expected source head must be a lowercase 40-character Git SHA"));
    }

    let receipt_path = artifact_root.join("receipt.json");
    let raw_receipt = read_bounded(&receipt_path, 1_048_576, "provider receipt")?;
    let receipt_sha256 = hash_bytes(&raw_receipt);
    let receipt_digest_anchor_matched = match expected_receipt_sha256 {
        Some(expected) if !is_digest(expected) => return Err(err("expected receipt digest must be a lowercase SHA-256")),
        Some(expected) if expected != receipt_sha256 => return Err(err("provider receipt does not match the independently supplied receipt digest")),
        Some(_) => true,
        None => false,
    };
    let receipt: Value = serde_json::from_slice(&raw_receipt)
        .map_err(|e| err(format!("receipt is not valid JSON: {e}")))?;

    if text_field(&receipt, "schema_version")? != RECEIPT_SCHEMA {
        return Err(err("provider-smoke receipt schema version mismatch"));
    }
    if text_field(&receipt, "benchmark")? != BENCHMARK {
        return Err(err("receipt benchmark is not the supported Unblob provider-smoke contract"));
    }
    if text_field(&receipt, "provider_version")? != PROVIDER_VERSION {
        return Err(err("provider version is not the pinned Unblob 26.6.4 contract"));
    }
    if text_field(&receipt, "claims_ceiling")? != RECEIPT_SCOPE {
        return Err(err("receipt claim scope is missing, broader than, or different from provider-smoke-only"));
    }
    let bundle_manifest_sha = digest_field(&receipt, "bundle_manifest_sha256")?;
    verify_bundle_manifest(artifact_root, &bundle_manifest_sha)?;

    if text_field(&receipt, "qualification.tier")? != "unblob-provider-smoke"
        || required(&receipt, "qualification.rea_cli_mcp_parity")?.as_bool() != Some(false)
        || required(&receipt, "qualification.firmware_execution")?.as_bool() != Some(false)
        || required(&receipt, "qualification.emulation")?.as_bool() != Some(false)
        || required(&receipt, "qualification.device_qualification")?.as_bool() != Some(false)
    {
        return Err(err("provider-smoke qualification scope is missing or escalated"));
    }

    let source_head_sha = text_field(&receipt, "source.head_sha")?;
    let checkout_commit_sha = text_field(&receipt, "source.checkout_commit_sha")?;
    let workflow_event_sha = text_field(&receipt, "source.workflow_event_sha")?;
    if !is_git_sha(&source_head_sha) || source_head_sha != expected_head {
        return Err(err("receipt source head does not match independently supplied expected head"));
    }
    if !is_git_sha(&checkout_commit_sha) {
        return Err(err("receipt checkout commit is not a lowercase 40-character Git SHA"));
    }
    if !is_git_sha(&workflow_event_sha) {
        return Err(err("receipt workflow event SHA is not a lowercase 40-character Git SHA"));
    }

    let executable_path = text_field(&receipt, "executable_path")?;
    if !Path::new(&executable_path).is_absolute() {
        return Err(err("provider executable path must be absolute"));
    }
    let _python_version = text_field(&receipt, "python_version")?;
    let _executable_sha = digest_field(&receipt, "executable_sha256")?;
    let _pip_freeze_sha = digest_field(&receipt, "pip_freeze_sha256")?;

    let normalizer_sha = digest_field(&receipt, "source.normalizer_sha256")?;
    let cases_sha = digest_field(&receipt, "source.cases_manifest_sha256")?;
    let oracle_sha = digest_field(&receipt, "source.oracle_sha256")?;
    let provider_smoke_sha = digest_field(&receipt, "source.provider_smoke_script_sha256")?;
    let provider_meta_script_sha = digest_field(&receipt, "source.provider_metamorphic_script_sha256")?;
    verify_digest_at(&source_root.join("normalize.jq"), &normalizer_sha, "normalizer")?;
    verify_digest_at(&source_root.join("cases.json"), &cases_sha, "cases manifest")?;
    verify_digest_at(&source_root.join("oracle.json"), &oracle_sha, "oracle")?;
    verify_digest_at(&source_root.join("provider-smoke.sh"), &provider_smoke_sha, "provider smoke script")?;
    verify_digest_at(&source_root.join("provider-metamorphic-sweep.sh"), &provider_meta_script_sha, "provider metamorphic script")?;

    let input_sha = digest_field(&receipt, "fixture_input_sha256")?;
    let mutated_input_sha = digest_field(&receipt, "mutated_fixture_input_sha256")?;
    let expected_payload_sha = digest_field(&receipt, "expected_payload_sha256")?;
    let mutated_payload_sha = digest_field(&receipt, "mutated_expected_payload_sha256")?;
    if input_sha == mutated_input_sha || expected_payload_sha == mutated_payload_sha {
        return Err(err("control fixture must change input and payload identity"));
    }

    verify_digest_file(&artifact_root.join("fixture-input-sha256.txt"), &input_sha, "baseline input manifest")?;
    verify_digest_file(&artifact_root.join("mutated-fixture-input-sha256.txt"), &mutated_input_sha, "mutated input manifest")?;
    verify_digest_file(&artifact_root.join("expected-payload-sha256.txt"), &expected_payload_sha, "baseline payload manifest")?;
    verify_digest_file(&artifact_root.join("mutated-expected-payload-sha256.txt"), &mutated_payload_sha, "mutated payload manifest")?;
    verify_digest_at(&artifact_root.join("fixture-source/payload.txt"), &expected_payload_sha, "baseline source payload")?;
    verify_digest_at(&artifact_root.join("fixture-source-mutated/payload.txt"), &mutated_payload_sha, "mutated source payload")?;

    let lanes = [
        ("run-a", "run_a"),
        ("run-b", "run_b"),
        ("run-mutated", "run_mutated"),
    ];
    let mut raw_digests = Vec::new();
    let mut semantic_digests = Vec::new();
    for (lane, receipt_key) in lanes {
        let input_path = artifact_root.join(lane).join("input.bin");
        let raw_report_path = artifact_root.join(lane).join("raw-report.json");
        let projection_path = artifact_root.join(lane).join("semantic-projection.json");
        let expected_input = if lane == "run-mutated" { &mutated_input_sha } else { &input_sha };
        verify_digest_at(&input_path, expected_input, &format!("{lane} input"))?;
        let expected_raw = digest_field(&receipt, &format!("raw_report_sha256.{receipt_key}"))?;
        let expected_semantic = if lane == "run-mutated" {
            digest_field(&receipt, "semantic_projection_sha256.semantic_mutation")?
        } else {
            digest_field(&receipt, "semantic_projection_sha256.equivalent_runs")?
        };
        verify_digest_at(&raw_report_path, &expected_raw, &format!("{lane} raw report"))?;
        verify_digest_at(&projection_path, &expected_semantic, &format!("{lane} semantic projection"))?;
        raw_digests.push(expected_raw);
        semantic_digests.push(expected_semantic);
    }

    if semantic_digests[0] != semantic_digests[1] {
        return Err(err("equivalent provider runs do not share a semantic projection digest"));
    }
    if semantic_digests[0] == semantic_digests[2] {
        return Err(err("controlled semantic mutation did not change the projection digest"));
    }
    if raw_digests[0] == raw_digests[1] {
        return Err(err("the raw report identities do not reflect the recorded run variation"));
    }
    if raw_digests[0] == raw_digests[2] {
        return Err(err("mutated provider run reused the baseline raw report digest"));
    }

    verify_payload(artifact_root, "run-a", &expected_payload_sha)?;
    verify_payload(artifact_root, "run-b", &expected_payload_sha)?;
    verify_payload(artifact_root, "run-mutated", &mutated_payload_sha)?;

    let meta_path = artifact_root.join("provider-metamorphic-receipt.json");
    let meta_sha = digest_field(&receipt, "provider_metamorphic.receipt_sha256")?;
    verify_digest_at(&meta_path, &meta_sha, "provider metamorphic receipt")?;
    let meta_bytes = read_bounded(&meta_path, 1_048_576, "provider metamorphic receipt")?;
    let meta: Value = serde_json::from_slice(&meta_bytes)
        .map_err(|e| err(format!("provider metamorphic receipt is not valid JSON: {e}")))?;
    if text_field(&meta, "schema_version")? != PROVIDER_META_SCHEMA {
        return Err(err("provider metamorphic receipt schema version mismatch"));
    }
    if digest_field(&meta, "raw_report_sha256")? != raw_digests[0] {
        return Err(err("provider metamorphic receipt is bound to a different raw report"));
    }
    if digest_field(&meta, "normalizer_sha256")? != normalizer_sha {
        return Err(err("provider metamorphic receipt is bound to a different normalizer"));
    }
    if digest_field(&meta, "semantic_projection_sha256")? != semantic_digests[0] {
        return Err(err("provider metamorphic receipt is bound to a different projection"));
    }
    let variants = required(&meta, "equivalent_variants")?
        .as_u64()
        .ok_or_else(|| err("provider metamorphic variant count must be an unsigned integer"))?;
    if variants != 64 || required(&meta, "all_variants_preserved_semantic_projection")?.as_bool() != Some(true) {
        return Err(err("provider metamorphic receipt does not confirm all 64 equivalent variants"));
    }
    if required(&meta, "raw_report_unchanged")?.as_bool() != Some(true) {
        return Err(err("provider metamorphic receipt does not confirm raw-report immutability"));
    }
    if required(&receipt, "provider_metamorphic.equivalent_variants")?.as_u64() != Some(64) {
        return Err(err("outer receipt variant count disagrees with nested provider receipt"));
    }
    if digest_field(&receipt, "provider_metamorphic.semantic_projection_sha256")? != semantic_digests[0] {
        return Err(err("outer receipt provider-metamorphic digest disagrees with verified projection"));
    }

    Ok(VerifiedEvidence {
        status: "verified",
        evidence_tier: "unblob-provider-smoke",
        receipt_sha256,
        verification_mode: if receipt_digest_anchor_matched { "receipt-digest-anchored" } else { "integrity-consistency-only" },
        receipt_digest_anchor_matched,
        source_head_sha,
        checkout_commit_sha,
        provider_version: PROVIDER_VERSION.to_owned(),
        raw_reports_verified: 3,
        equivalent_provider_runs: 2,
        semantic_mutations_verified: 1,
        provider_metamorphic_variants_verified: variants,
        // Provider smoke must never be promoted to stronger qualification tiers.
        rea_cli_mcp_parity: false,
        firmware_execution: false,
        emulation: false,
        device_qualification: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
        source: PathBuf,
        artifact: PathBuf,
        expected_head: String,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn write(path: &Path, bytes: &[u8]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut f = File::create(path).unwrap();
        f.write_all(bytes).unwrap();
    }
    fn sha(bytes: &[u8]) -> String {
        hash_bytes(bytes)
    }
    fn put_digest(root: &Path, path: &str, digest: &str) {
        write(&root.join(path), format!("{digest}\n").as_bytes());
    }
    fn create_fixture() -> Fixture {
        let suffix = NEXT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("rea-evidence-verify-{}-{suffix}", std::process::id()));
        let source = root.join("source");
        let artifact = root.join("artifact");
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(&artifact).unwrap();

        let normalizer = b"normalize fixture v1\n";
        let cases = b"{\"schema_version\":\"cases\"}\n";
        let oracle = b"{\"schema_version\":\"oracle\"}\n";
        write(&source.join("normalize.jq"), normalizer);
        write(&source.join("cases.json"), cases);
        write(&source.join("oracle.json"), oracle);
        write(&source.join("provider-smoke.sh"), b"provider smoke script v1\n");
        write(&source.join("provider-metamorphic-sweep.sh"), b"provider metamorphic script v1\n");

        let head = "1234567890abcdef1234567890abcdef12345678".to_owned();
        let checkout = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned();
        let event = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned();
        let input = b"gzip-input-v1";
        let input_mutated = b"gzip-input-v2";
        let payload = b"inert payload v1\n";
        let payload_mutated = b"inert payload v2\n";
        let raw_a = b"{\"provider_report\":\"a\"}\n";
        let raw_b = b"{\"provider_report\":\"b\"}\n";
        let raw_mut = b"{\"provider_report\":\"mutated\"}\n";
        let projection_a = b"{\"projection\":\"same\"}\n";
        let projection_mut = b"{\"projection\":\"changed\"}\n";
        let input_sha = sha(input);
        let mutated_input_sha = sha(input_mutated);
        let payload_sha = sha(payload);
        let mutated_payload_sha = sha(payload_mutated);
        let raw_a_sha = sha(raw_a);
        let raw_b_sha = sha(raw_b);
        let raw_mut_sha = sha(raw_mut);
        let proj_sha = sha(projection_a);
        let proj_mut_sha = sha(projection_mut);
        let normalizer_sha = sha(normalizer);
        let cases_sha = sha(cases);
        let oracle_sha = sha(oracle);

        write(&artifact.join("fixture-source/payload.txt"), payload);
        write(&artifact.join("fixture-source-mutated/payload.txt"), payload_mutated);
        for (lane, input_bytes, raw, projection, extracted) in [
            ("run-a", input.as_slice(), raw_a.as_slice(), projection_a.as_slice(), payload.as_slice()),
            ("run-b", input.as_slice(), raw_b.as_slice(), projection_a.as_slice(), payload.as_slice()),
            ("run-mutated", input_mutated.as_slice(), raw_mut.as_slice(), projection_mut.as_slice(), payload_mutated.as_slice()),
        ] {
            write(&artifact.join(lane).join("input.bin"), input_bytes);
            write(&artifact.join(lane).join("raw-report.json"), raw);
            write(&artifact.join(lane).join("semantic-projection.json"), projection);
            write(&artifact.join(lane).join("extracted/payload.txt"), extracted);
        }
        put_digest(&artifact, "fixture-input-sha256.txt", &input_sha);
        put_digest(&artifact, "mutated-fixture-input-sha256.txt", &mutated_input_sha);
        put_digest(&artifact, "expected-payload-sha256.txt", &payload_sha);
        put_digest(&artifact, "mutated-expected-payload-sha256.txt", &mutated_payload_sha);

        let meta = json!({
            "schema_version": PROVIDER_META_SCHEMA,
            "raw_report_sha256": raw_a_sha,
            "normalizer_sha256": normalizer_sha,
            "semantic_projection_sha256": proj_sha,
            "source_root_for_audit": "/tmp/provider-run-a",
            "seed_start": 1,
            "seed_end": 64,
            "equivalent_variants": 64,
            "all_variants_preserved_semantic_projection": true,
            "raw_report_unchanged": true,
            "transformations": ["scratch-root rebase", "coherent ID remap", "record order"],
            "claims_ceiling": "synthetic",
        });
        let meta_bytes = serde_json::to_vec_pretty(&meta).unwrap();
        let meta_sha = sha(&meta_bytes);
        write(&artifact.join("provider-metamorphic-receipt.json"), &meta_bytes);

        let mut bundle_paths = Vec::new();
        collect_bundle_paths(&artifact, &artifact, &mut bundle_paths).unwrap();
        bundle_paths.sort();
        let bundle_entries: Vec<Value> = bundle_paths
            .iter()
            .map(|path| json!({"path": path, "sha256": hash_file(&artifact.join(path)).unwrap()}))
            .collect();
        let bundle_manifest = json!({
            "schema_version": BUNDLE_MANIFEST_SCHEMA,
            "files": bundle_entries,
        });
        let bundle_manifest_bytes = serde_json::to_vec_pretty(&bundle_manifest).unwrap();
        let bundle_manifest_sha = sha(&bundle_manifest_bytes);
        write(&artifact.join("bundle-manifest.json"), &bundle_manifest_bytes);

        let receipt = json!({
            "schema_version": RECEIPT_SCHEMA,
            "benchmark": BENCHMARK,
            "qualification": {
                "tier": "unblob-provider-smoke",
                "rea_cli_mcp_parity": false,
                "firmware_execution": false,
                "emulation": false,
                "device_qualification": false,
            },
            "source": {
                "head_sha": head,
                "checkout_commit_sha": checkout,
                "workflow_event_sha": event,
                "normalizer_sha256": normalizer_sha,
                "cases_manifest_sha256": cases_sha,
                "oracle_sha256": oracle_sha,
                "provider_smoke_script_sha256": sha(b"provider smoke script v1\n"),
                "provider_metamorphic_script_sha256": sha(b"provider metamorphic script v1\n"),
            },
            "provider_version": PROVIDER_VERSION,
            "executable_path": "/usr/local/bin/unblob",
            "executable_sha256": "c".repeat(64),
            "python_version": "Python 3.12.0",
            "pip_freeze_sha256": "d".repeat(64),
            "fixture_input_sha256": input_sha,
            "mutated_fixture_input_sha256": mutated_input_sha,
            "expected_payload_sha256": payload_sha,
            "mutated_expected_payload_sha256": mutated_payload_sha,
            "raw_report_sha256": {"run_a":raw_a_sha,"run_b":raw_b_sha,"run_mutated":raw_mut_sha},
            "semantic_projection_sha256": {"equivalent_runs":proj_sha,"semantic_mutation":proj_mut_sha},
            "provider_metamorphic": {"receipt_sha256":meta_sha,"equivalent_variants":64,"semantic_projection_sha256":proj_sha},
            "bundle_manifest_sha256": bundle_manifest_sha,
            "claims_ceiling": RECEIPT_SCOPE,
        });
        write(&artifact.join("receipt.json"), serde_json::to_vec_pretty(&receipt).unwrap().as_slice());
        Fixture { root, source, artifact, expected_head: head }
    }

    #[test]
    fn accepts_consistent_provider_receipt_and_keeps_claim_scope_narrow() {
        let f = create_fixture();
        let result = verify_bundle(&f.artifact, &f.source, &f.expected_head, None).unwrap();
        assert_eq!(result.status, "verified");
        assert_eq!(result.evidence_tier, "unblob-provider-smoke");
        assert_eq!(result.provider_metamorphic_variants_verified, 64);
        assert!(!result.rea_cli_mcp_parity);
        assert!(!result.firmware_execution);
        assert!(!result.emulation);
        assert!(!result.device_qualification);
    }

    #[test]
    fn reports_digest_anchor_only_when_independently_supplied() {
        let f = create_fixture();
        let receipt_sha = hash_file(&f.artifact.join("receipt.json")).unwrap();
        let anchored = verify_bundle(&f.artifact, &f.source, &f.expected_head, Some(&receipt_sha)).unwrap();
        assert_eq!(anchored.verification_mode, "receipt-digest-anchored");
        assert!(anchored.receipt_digest_anchor_matched);
        let unanchored = verify_bundle(&f.artifact, &f.source, &f.expected_head, None).unwrap();
        assert_eq!(unanchored.verification_mode, "integrity-consistency-only");
        assert!(!unanchored.receipt_digest_anchor_matched);
    }

    #[test]
    fn rejects_wrong_independent_receipt_digest() {
        let f = create_fixture();
        let incorrect = "0".repeat(64);
        assert!(verify_bundle(&f.artifact, &f.source, &f.expected_head, Some(&incorrect))
            .unwrap_err().to_string().contains("independently supplied receipt digest"));
    }

    #[test]
    fn rejects_wrong_expected_source_head() {
        let f = create_fixture();
        let other = "9999999999999999999999999999999999999999";
        assert!(verify_bundle(&f.artifact, &f.source, other, None).unwrap_err().to_string().contains("expected head"));
    }

    #[test]
    fn rejects_changed_raw_report_bytes() {
        let f = create_fixture();
        write(&f.artifact.join("run-a/raw-report.json"), b"{\"tampered\":true}\n");
        assert!(verify_bundle(&f.artifact, &f.source, &f.expected_head, None).unwrap_err().to_string().contains("digest mismatch"));
    }

    #[test]
    fn rejects_changed_normalizer_source() {
        let f = create_fixture();
        write(&f.source.join("normalize.jq"), b"modified normalizer\n");
        assert!(verify_bundle(&f.artifact, &f.source, &f.expected_head, None).unwrap_err().to_string().contains("normalizer"));
    }

    #[test]
    fn rejects_semantic_mutation_that_claims_same_projection() {
        let f = create_fixture();
        let raw: Value = serde_json::from_slice(&fs::read(f.artifact.join("receipt.json")).unwrap()).unwrap();
        let mut changed = raw;
        let semantic_mutation = changed["semantic_projection_sha256"]["semantic_mutation"].clone();
        let equivalent_digest = changed["semantic_projection_sha256"]["equivalent_runs"].clone();
        changed["semantic_projection_sha256"]["semantic_mutation"] = equivalent_digest;
        assert_ne!(semantic_mutation, changed["semantic_projection_sha256"]["semantic_mutation"]);
        write(
            &f.artifact.join("run-mutated/semantic-projection.json"),
            b"{\"projection\":\"same\"}\n",
        );
        write(&f.artifact.join("receipt.json"), serde_json::to_vec_pretty(&changed).unwrap().as_slice());
        assert!(verify_bundle(&f.artifact, &f.source, &f.expected_head, None).unwrap_err().to_string().contains("did not change"));
    }

    #[test]
    fn rejects_scope_escalation_to_rea_parity_or_device_qualification() {
        let f = create_fixture();
        let raw: Value = serde_json::from_slice(&fs::read(f.artifact.join("receipt.json")).unwrap()).unwrap();
        let mut changed = raw;
        changed["claims_ceiling"] = json!("Validated REA CLI/MCP parity and hardware behavior.");
        write(&f.artifact.join("receipt.json"), serde_json::to_vec_pretty(&changed).unwrap().as_slice());
        assert!(verify_bundle(&f.artifact, &f.source, &f.expected_head, None).unwrap_err().to_string().contains("claim scope"));
    }

    #[test]
    fn rejects_provider_metamorphic_receipt_digest_mismatch() {
        let f = create_fixture();
        write(&f.artifact.join("provider-metamorphic-receipt.json"), b"{\"tampered\":true}");
        assert!(verify_bundle(&f.artifact, &f.source, &f.expected_head, None).unwrap_err().to_string().contains("digest mismatch"));
    }

    #[test]
    fn rejects_missing_and_malformed_sha256_fields() {
        let f = create_fixture();
        let raw: Value = serde_json::from_slice(&fs::read(f.artifact.join("receipt.json")).unwrap()).unwrap();
        let mut changed = raw;
        changed["source"]["oracle_sha256"] = json!("1234");
        write(&f.artifact.join("receipt.json"), serde_json::to_vec_pretty(&changed).unwrap().as_slice());
        assert!(verify_bundle(&f.artifact, &f.source, &f.expected_head, None).unwrap_err().to_string().contains("lowercase SHA-256"));
    }

    #[test]
    fn rejects_provider_script_identity_drift() {
        let f = create_fixture();
        write(&f.source.join("provider-smoke.sh"), b"changed provider smoke script\\n");
        assert!(verify_bundle(&f.artifact, &f.source, &f.expected_head, None)
            .unwrap_err().to_string().contains("provider smoke script"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_raw_report() {
        use std::os::unix::fs::symlink;
        let f = create_fixture();
        let report = f.artifact.join("run-a/raw-report.json");
        fs::remove_file(&report).unwrap();
        symlink(f.artifact.join("run-b/raw-report.json"), &report).unwrap();
        assert!(verify_bundle(&f.artifact, &f.source, &f.expected_head, None)
            .unwrap_err().to_string().contains("non-symlink"));
    }

    #[test]
    fn rejects_wrong_extracted_payload_even_when_receipt_is_unchanged() {
        let f = create_fixture();
        write(&f.artifact.join("run-b/extracted/payload.txt"), b"not the expected payload\n");
        assert!(verify_bundle(&f.artifact, &f.source, &f.expected_head, None).unwrap_err().to_string().contains("extracted payload"));
    }
}
