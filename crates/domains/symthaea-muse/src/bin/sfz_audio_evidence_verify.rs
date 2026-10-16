// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verify provenance and content identity for a Muse SFZ audio comparison bundle.
//!
//! Usage:
//!   cargo run --manifest-path crates/domains/symthaea-muse/Cargo.toml --bin sfz_audio_evidence_verify -- --manifest bundle.json
//!   cargo run --manifest-path crates/domains/symthaea-muse/Cargo.toml --bin sfz_audio_evidence_verify -- --manifest bundle.json --output verification.json
//!
//! This verifies that declared source assets, renderer build inputs, WAV captures,
//! and a differential-comparison receipt are present, hash-consistent, and linked
//! to the same two WAVs. It does not authenticate who built an engine, judge audio
//! quality, establish SFZ-standard compliance, or make a perceptual-equivalence claim.
//!
//! All artifact paths are relative to the manifest's directory. Absolute paths,
//! parent-directory components, symlinks, non-regular files and files larger than
//! 2 GiB are rejected. The verifier never executes the declared render command.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    env,
    error::Error,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    process::ExitCode,
};

const MAX_ARTIFACT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const EXPECTED_MANIFEST_SCHEMA: &str = "muse-sfz-audio-evidence-bundle-v1";
const EXPECTED_COMPARISON_SCHEMA: &str = "muse-sfz-audio-differential-receipt-v1";

#[derive(Debug)]
struct Arguments {
    manifest: PathBuf,
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct EvidenceBundleV1 {
    schema: String,
    bundle_id: String,
    fixture_id: String,
    fixture_pack_digest: String,
    fixture_pack_receipt: FileReferenceV1,
    fixture_manifest: FileReferenceV1,
    sfz: FileReferenceV1,
    midi: FileReferenceV1,
    sample_assets: Vec<FileReferenceV1>,
    candidate: RendererCaptureV1,
    reference: RendererCaptureV1,
    comparison_receipt: FileReferenceV1,
}

#[derive(Debug, Clone, Deserialize)]
struct FileReferenceV1 {
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct RendererCaptureV1 {
    engine_name: String,
    engine_version: String,
    source_revision: String,
    executable: FileReferenceV1,
    build_configuration: FileReferenceV1,
    render_command: String,
    output_wav: FileReferenceV1,
}

#[derive(Debug, Serialize)]
struct ArtifactCheckV1 {
    role: String,
    path: String,
    expected_sha256: String,
    actual_sha256: Option<String>,
    byte_len: Option<u64>,
    passed: bool,
    failure: Option<String>,
}

#[derive(Debug, Serialize)]
struct RendererReceiptV1 {
    engine_name: String,
    engine_version: String,
    source_revision: String,
    render_command: String,
}

#[derive(Debug, Serialize)]
struct ComparisonBindingV1 {
    comparison_schema_verified: bool,
    candidate_output_hash_matches: bool,
    reference_output_hash_matches: bool,
    comparison_status: Option<String>,
    metrics_present: bool,
    quality_qualified: bool,
    failures: Vec<String>,
}

#[derive(Debug, Serialize)]
struct VerificationReceiptV1 {
    schema: &'static str,
    verifier_version: &'static str,
    manifest_sha256: String,
    bundle_id: String,
    fixture_id: String,
    fixture_pack_digest: String,
    fixture_pack_digest_content_matches: bool,
    fixture_manifest_identity_matches: bool,
    candidate_renderer: RendererReceiptV1,
    reference_renderer: RendererReceiptV1,
    artifact_checks: Vec<ArtifactCheckV1>,
    comparison_binding: ComparisonBindingV1,
    integrity_passed: bool,
    quality_qualified: bool,
    failures: Vec<String>,
    scope_boundary: &'static str,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("sfz_audio_evidence_verify: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, Box<dyn Error>> {
    let Some(arguments) = parse_arguments()? else {
        print_usage();
        return Ok(ExitCode::SUCCESS);
    };

    let manifest_metadata = fs::symlink_metadata(&arguments.manifest)?;
    if manifest_metadata.file_type().is_symlink() || !manifest_metadata.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "manifest must be a regular non-symlink file",
        )
        .into());
    }
    if manifest_metadata.len() > 8 * 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "manifest exceeds 8 MiB size limit",
        ).into());
    }
    let canonical_manifest_path = fs::canonicalize(&arguments.manifest)?;
    let manifest_bytes = fs::read(&canonical_manifest_path)?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let manifest: EvidenceBundleV1 = serde_json::from_slice(&manifest_bytes)?;
    validate_manifest_contract(&manifest)?;

    let root = canonical_manifest_path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "manifest has no parent directory"))?;

    let mut artifact_checks = Vec::new();
    artifact_checks.push(check_artifact(
        root,
        "fixture_pack_receipt",
        &manifest.fixture_pack_receipt,
    ));
    artifact_checks.push(check_artifact(root, "fixture_manifest", &manifest.fixture_manifest));
    artifact_checks.push(check_artifact(root, "sfz_source", &manifest.sfz));
    artifact_checks.push(check_artifact(root, "midi_source", &manifest.midi));
    for (index, asset) in manifest.sample_assets.iter().enumerate() {
        artifact_checks.push(check_artifact(
            root,
            &format!("sample_asset_{index:04}"),
            asset,
        ));
    }

    artifact_checks.extend(check_renderer_artifacts(root, "candidate", &manifest.candidate));
    artifact_checks.extend(check_renderer_artifacts(root, "reference", &manifest.reference));

    let fixture_manifest_check = artifact_checks
        .iter()
        .find(|check| check.role == "fixture_manifest");
    let (fixture_manifest_identity_matches, fixture_manifest_failure) =
        if fixture_manifest_check.is_some_and(|check| check.passed) {
            let bytes = fs::read(resolve_artifact(root, &manifest.fixture_manifest.path)?)?;
            verify_fixture_manifest_identity(
                &bytes,
                &manifest.fixture_manifest.path,
                &manifest.fixture_id,
                &manifest.sfz,
                &manifest.sample_assets,
                root,
            )
        } else {
            (false, Some("fixture manifest is unavailable or failed hash verification".to_owned()))
        };

    let pack_receipt_check = artifact_checks
        .iter()
        .find(|check| check.role == "fixture_pack_receipt");
    let fixture_pack_digest_content_matches = if pack_receipt_check.is_some_and(|check| check.passed) {
        fs::read(resolve_artifact(root, &manifest.fixture_pack_receipt.path)?)?
            .split(|byte| byte.is_ascii_whitespace())
            .find(|part| !part.is_empty())
            .is_some_and(|part| part.eq_ignore_ascii_case(manifest.fixture_pack_digest.as_bytes()))
    } else {
        false
    };

    let comparison_check = check_artifact(
        root,
        "comparison_receipt",
        &manifest.comparison_receipt,
    );
    let comparison_bytes = if comparison_check.passed {
        fs::read(resolve_artifact(root, &manifest.comparison_receipt.path)?)?
    } else {
        Vec::new()
    };
    artifact_checks.push(comparison_check);

    let comparison_binding = if comparison_bytes.is_empty() {
        ComparisonBindingV1 {
            comparison_schema_verified: false,
            candidate_output_hash_matches: false,
            reference_output_hash_matches: false,
            comparison_status: None,
            metrics_present: false,
            quality_qualified: false,
            failures: vec!["comparison receipt is unavailable or empty".to_owned()],
        }
    } else {
        validate_comparison_binding(
            &comparison_bytes,
            &manifest.candidate.output_wav.sha256,
            &manifest.reference.output_wav.sha256,
        )
    };

    let mut failures = Vec::new();
    if manifest.schema != EXPECTED_MANIFEST_SCHEMA {
        failures.push(format!(
            "manifest schema mismatch: expected {EXPECTED_MANIFEST_SCHEMA:?}, got {:?}",
            manifest.schema
        ));
    }
    for check in &artifact_checks {
        if !check.passed {
            failures.push(format!(
                "artifact {} ({}) failed: {}",
                check.role,
                check.path,
                check.failure.as_deref().unwrap_or("hash mismatch")
            ));
        }
    }
    if !fixture_pack_digest_content_matches {
        failures.push("fixture-pack receipt content does not match declared tree digest".to_owned());
    }
    if let Some(failure) = fixture_manifest_failure {
        failures.push(failure);
    }
    failures.extend(comparison_binding.failures.iter().cloned());

    let integrity_passed = failures.is_empty()
        && artifact_checks.iter().all(|check| check.passed)
        && comparison_binding.comparison_schema_verified
        && comparison_binding.candidate_output_hash_matches
        && comparison_binding.reference_output_hash_matches
        && comparison_binding.metrics_present;

    let receipt = VerificationReceiptV1 {
        schema: "muse-sfz-audio-evidence-verification-v1",
        verifier_version: "sfz-audio-evidence-verifier-v1",
        manifest_sha256,
        bundle_id: manifest.bundle_id,
        fixture_id: manifest.fixture_id,
        fixture_pack_digest: manifest.fixture_pack_digest,
        fixture_pack_digest_content_matches,
        fixture_manifest_identity_matches,
        candidate_renderer: renderer_receipt(&manifest.candidate),
        reference_renderer: renderer_receipt(&manifest.reference),
        artifact_checks,
        comparison_binding,
        integrity_passed,
        // Integrity verification is not a listening/quality assessment.
        quality_qualified: false,
        failures,
        scope_boundary: "integrity and provenance-link verification only; renderer identity is declared by the manifest and hashes are not a signed attestation; no audio-quality or SFZ compatibility pass is implied",
    };

    let json = serde_json::to_string_pretty(&receipt)?;
    if let Some(output) = arguments.output {
        // Refuse to overwrite any existing artifact, including the manifest itself.
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        let mut file = options.open(&output)?;
        file.write_all(json.as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    println!("{json}");

    Ok(if receipt.integrity_passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

fn parse_arguments() -> Result<Option<Arguments>, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let mut manifest = None;
    let mut output = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => return Ok(None),
            "--manifest" => {
                manifest = Some(PathBuf::from(args.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--manifest requires a path")
                })?));
            }
            "--output" => {
                output = Some(PathBuf::from(args.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--output requires a path")
                })?));
            }
            unknown => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument {unknown:?}; use --help"),
                )
                .into());
            }
        }
    }

    let manifest = manifest.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "missing required --manifest PATH",
        )
    })?;
    Ok(Some(Arguments { manifest, output }))
}

fn print_usage() {
    println!(
        "Usage: sfz_audio_evidence_verify --manifest BUNDLE.json [--output VERIFICATION.json]\n\
         Verifies artifact hashes and ties renderer captures to a differential receipt.\n\
         Exit 0 = integrity verified (not quality-qualified); exit 1 = evidence mismatch; exit 2 = setup error."
    );
}

fn validate_manifest_contract(manifest: &EvidenceBundleV1) -> Result<(), Box<dyn Error>> {
    if manifest.schema != EXPECTED_MANIFEST_SCHEMA {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported manifest schema {:?}; expected {EXPECTED_MANIFEST_SCHEMA:?}",
                manifest.schema
            ),
        )
        .into());
    }
    for (field, value) in [
        ("bundle_id", manifest.bundle_id.as_str()),
        ("fixture_id", manifest.fixture_id.as_str()),
        ("fixture_pack_digest", manifest.fixture_pack_digest.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{field} must not be empty"),
            )
            .into());
        }
    }
    validate_sha256("fixture_pack_digest", &manifest.fixture_pack_digest)?;
    validate_file_reference("fixture_pack_receipt", &manifest.fixture_pack_receipt)?;
    validate_file_reference("fixture_manifest", &manifest.fixture_manifest)?;
    validate_file_reference("sfz", &manifest.sfz)?;
    validate_file_reference("midi", &manifest.midi)?;
    validate_file_reference("comparison_receipt", &manifest.comparison_receipt)?;

    let mut sample_paths = std::collections::BTreeSet::new();
    for asset in &manifest.sample_assets {
        validate_file_reference("sample asset", asset)?;
        if !sample_paths.insert(asset.path.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("duplicate sample asset path {:?}", asset.path),
            )
            .into());
        }
    }
    validate_renderer_contract("candidate", &manifest.candidate)?;
    validate_renderer_contract("reference", &manifest.reference)?;
    Ok(())
}

fn validate_renderer_contract(
    role: &str,
    renderer: &RendererCaptureV1,
) -> Result<(), Box<dyn Error>> {
    for (field, value) in [
        ("engine_name", renderer.engine_name.as_str()),
        ("engine_version", renderer.engine_version.as_str()),
        ("source_revision", renderer.source_revision.as_str()),
        ("render_command", renderer.render_command.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{role}.{field} must not be empty"),
            )
            .into());
        }
    }
    for (name, reference) in [
        ("executable", &renderer.executable),
        ("build_configuration", &renderer.build_configuration),
        ("output_wav", &renderer.output_wav),
    ] {
        validate_file_reference(&format!("{role}.{name}"), reference)?;
    }
    Ok(())
}

fn validate_file_reference(
    label: &str,
    reference: &FileReferenceV1,
) -> Result<(), Box<dyn Error>> {
    let _ = resolve_artifact(Path::new("."), &reference.path)?;
    validate_sha256(&format!("{label}.sha256"), &reference.sha256)
}

fn validate_sha256(label: &str, digest: &str) -> Result<(), Box<dyn Error>> {
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} must be a 64-character hexadecimal SHA-256 digest"),
        )
        .into());
    }
    Ok(())
}

fn check_renderer_artifacts(
    root: &Path,
    role: &str,
    renderer: &RendererCaptureV1,
) -> Vec<ArtifactCheckV1> {
    vec![
        check_artifact(
            root,
            &format!("{role}_renderer_executable"),
            &renderer.executable,
        ),
        check_artifact(
            root,
            &format!("{role}_build_configuration"),
            &renderer.build_configuration,
        ),
        check_artifact(
            root,
            &format!("{role}_output_wav"),
            &renderer.output_wav,
        ),
    ]
}

fn check_artifact(root: &Path, role: &str, reference: &FileReferenceV1) -> ArtifactCheckV1 {
    let fail = |message: String, actual_sha256: Option<String>, byte_len: Option<u64>| ArtifactCheckV1 {
        role: role.to_owned(),
        path: reference.path.clone(),
        expected_sha256: reference.sha256.to_ascii_lowercase(),
        actual_sha256,
        byte_len,
        passed: false,
        failure: Some(message),
    };

    let path = match resolve_artifact(root, &reference.path) {
        Ok(path) => path,
        Err(error) => return fail(error.to_string(), None, None),
    };
    if let Err(error) = ensure_no_symlink_components(root, &reference.path) {
        return fail(error.to_string(), None, None);
    }
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) => return fail(error.to_string(), None, None),
    };
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return fail("artifact is not a regular non-symlink file".to_owned(), None, None);
    }
    if metadata.len() > MAX_ARTIFACT_BYTES {
        return fail(
            format!("artifact exceeds {} MiB size limit", MAX_ARTIFACT_BYTES / (1024 * 1024)),
            None,
            Some(metadata.len()),
        );
    }
    let actual = match sha256_file(&path) {
        Ok(actual) => actual,
        Err(error) => return fail(error.to_string(), None, Some(metadata.len())),
    };
    let matches = actual.eq_ignore_ascii_case(&reference.sha256);
    ArtifactCheckV1 {
        role: role.to_owned(),
        path: reference.path.clone(),
        expected_sha256: reference.sha256.to_ascii_lowercase(),
        actual_sha256: Some(actual),
        byte_len: Some(metadata.len()),
        passed: matches,
        failure: if matches {
            None
        } else {
            Some("SHA-256 mismatch".to_owned())
        },
    }
}

fn verify_fixture_manifest_identity(
    bytes: &[u8],
    fixture_manifest_path: &str,
    fixture_id: &str,
    sfz: &FileReferenceV1,
    sample_assets: &[FileReferenceV1],
    root: &Path,
) -> (bool, Option<String>) {
    let fail = |message: String| (false, Some(message));
    let value: Value = match serde_json::from_slice(bytes) {
        Ok(value) => value,
        Err(error) => return fail(format!("fixture manifest is invalid JSON: {error}")),
    };
    if value.get("schema").and_then(Value::as_str) != Some("muse-sfz-compat-fixture-pack-v1") {
        return fail("fixture manifest has an unknown schema".to_owned());
    }
    let Some(fixtures) = value.get("fixtures").and_then(Value::as_array) else {
        return fail("fixture manifest has no fixtures array".to_owned());
    };
    let Some(fixture) = fixtures.iter().find(|fixture| {
        fixture.get("fixture_id").and_then(Value::as_str) == Some(fixture_id)
    }) else {
        return fail(format!("fixture ID {fixture_id:?} is absent from fixture manifest"));
    };
    let Some(expected_sfz_hash) = fixture.get("sfz_sha256").and_then(Value::as_str) else {
        return fail("selected fixture has no sfz_sha256".to_owned());
    };
    if !expected_sfz_hash.eq_ignore_ascii_case(&sfz.sha256) {
        return fail("fixture manifest SFZ hash does not match declared SFZ".to_owned());
    }
    let Some(expected_sfz_len) = fixture.get("sfz_byte_len").and_then(Value::as_u64) else {
        return fail("selected fixture has no valid sfz_byte_len".to_owned());
    };
    let actual_sfz_len = match fs::metadata(resolve_artifact(root, &sfz.path).unwrap_or_default()) {
        Ok(metadata) => metadata.len(),
        Err(error) => return fail(format!("cannot inspect selected SFZ length: {error}")),
    };
    if expected_sfz_len != actual_sfz_len {
        return fail("fixture manifest SFZ length does not match declared SFZ".to_owned());
    }

    let Some(fixture_sfz_file) = fixture.get("sfz_file").and_then(Value::as_str) else {
        return fail("selected fixture has no sfz_file".to_owned());
    };
    let pack_root = Path::new(fixture_manifest_path).parent().unwrap_or_else(|| Path::new(""));
    let expected_bundle_sfz = pack_root.join(fixture_sfz_file).to_string_lossy().replace('\\\\', "/");
    if expected_bundle_sfz != sfz.path {
        return fail(format!(
            "fixture manifest SFZ path {:?} does not resolve to bundle path {:?}",
            fixture_sfz_file, sfz.path
        ));
    }

    let Some(expected_assets) = fixture.get("sample_assets").and_then(Value::as_array) else {
        return fail("selected fixture has no sample_assets array".to_owned());
    };
    let mut expected_hashes: Vec<String> = expected_assets
        .iter()
        .filter_map(|asset| asset.get("sha256").and_then(Value::as_str).map(str::to_ascii_lowercase))
        .collect();
    if expected_hashes.len() != expected_assets.len() {
        return fail("fixture manifest contains a sample asset without a SHA-256 hash".to_owned());
    }
    let mut declared_hashes: Vec<String> = sample_assets
        .iter()
        .map(|asset| asset.sha256.to_ascii_lowercase())
        .collect();
    expected_hashes.sort();
    declared_hashes.sort();
    if expected_hashes != declared_hashes {
        return fail("declared sample-asset hash multiset does not match selected fixture".to_owned());
    }

    (true, None)
}

fn validate_comparison_binding(
    bytes: &[u8],
    expected_candidate_hash: &str,
    expected_reference_hash: &str,
) -> ComparisonBindingV1 {
    let value: Value = match serde_json::from_slice(bytes) {
        Ok(value) => value,
        Err(error) => {
            return ComparisonBindingV1 {
                comparison_schema_verified: false,
                candidate_output_hash_matches: false,
                reference_output_hash_matches: false,
                comparison_status: None,
                metrics_present: false,
                quality_qualified: false,
                failures: vec![format!("comparison receipt is invalid JSON: {error}")],
            };
        }
    };
    let schema_ok = value.get("schema").and_then(Value::as_str)
        == Some(EXPECTED_COMPARISON_SCHEMA);
    let candidate_hash = value.pointer("/candidate/sha256").and_then(Value::as_str);
    let reference_hash = value.pointer("/reference/sha256").and_then(Value::as_str);
    let candidate_ok = candidate_hash
        .is_some_and(|actual| actual.eq_ignore_ascii_case(expected_candidate_hash));
    let reference_ok = reference_hash
        .is_some_and(|actual| actual.eq_ignore_ascii_case(expected_reference_hash));
    let status = value
        .pointer("/comparison/status")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let metrics_present = value
        .pointer("/comparison/metrics")
        .is_some_and(|metrics| !metrics.is_null() && metrics.is_object());

    let mut failures = Vec::new();
    if !schema_ok {
        failures.push(format!(
            "comparison schema does not equal {EXPECTED_COMPARISON_SCHEMA:?}"
        ));
    }
    if !candidate_ok {
        failures.push("comparison candidate WAV hash does not match declared capture".to_owned());
    }
    if !reference_ok {
        failures.push("comparison reference WAV hash does not match declared capture".to_owned());
    }
    if status.as_deref() != Some("measured_not_quality_qualified") {
        failures.push(format!(
            "comparison status is not a completed measurement: {:?}",
            status
        ));
    }
    if !metrics_present {
        failures.push("comparison receipt does not contain a metrics object".to_owned());
    }

    ComparisonBindingV1 {
        comparison_schema_verified: schema_ok,
        candidate_output_hash_matches: candidate_ok,
        reference_output_hash_matches: reference_ok,
        comparison_status: status,
        metrics_present,
        // This verifier never upgrades measurements into quality qualification.
        quality_qualified: false,
        failures,
    }
}

fn renderer_receipt(renderer: &RendererCaptureV1) -> RendererReceiptV1 {
    RendererReceiptV1 {
        engine_name: renderer.engine_name.clone(),
        engine_version: renderer.engine_version.clone(),
        source_revision: renderer.source_revision.clone(),
        render_command: renderer.render_command.clone(),
    }
}

/// Join only normalized relative paths. Symlinks are checked against the chosen root separately.
fn resolve_artifact(root: &Path, relative: &str) -> Result<PathBuf, Box<dyn Error>> {
    let candidate = Path::new(relative);
    let bytes = relative.as_bytes();
    let has_windows_drive_prefix =
        bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
    if relative.is_empty()
        || relative.contains('\\')
        || has_windows_drive_prefix
        || candidate.is_absolute()
        || candidate.components().any(|component| {
            !matches!(component, Component::Normal(_))
        })
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsafe relative artifact path {relative:?}"),
        )
        .into());
    }

    let mut current = root.to_path_buf();
    for component in candidate.components() {
        match component {
            Component::Normal(name) => current.push(name),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "artifact path contains an unsafe component",
                ).into());
            }
        }
    }
    Ok(current)
}

/// Reject symlinks in the artifact path from the manifest root to the target.
/// Missing components are left for the final artifact check to report as evidence failure.
fn ensure_no_symlink_components(root: &Path, relative: &str) -> Result<(), Box<dyn Error>> {
    let candidate = Path::new(relative);
    let mut current = root.to_path_buf();
    for component in candidate.components() {
        let Component::Normal(name) = component else {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "unsafe path component").into());
        };
        current.push(name);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("artifact path traverses symlink {}", current.display()),
                ).into());
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("not a regular non-symlink file: {}", path.display()),
        )
        .into());
    }
    if metadata.len() > MAX_ARTIFACT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "artifact exceeds maximum file size",
        )
        .into());
    }

    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn sha256_known_vector_is_correct() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn path_resolver_rejects_escape_and_absolute_paths() {
        assert!(resolve_artifact(Path::new("/tmp/bundle"), "../escape.wav").is_err());
        assert!(resolve_artifact(Path::new("/tmp/bundle"), "/tmp/audio.wav").is_err());
        assert!(resolve_artifact(Path::new("/tmp/bundle"), "").is_err());
        assert!(resolve_artifact(Path::new("/tmp/bundle"), "C:\\outside.wav").is_err());
        assert!(resolve_artifact(Path::new("/tmp/bundle"), "folder\\capture.wav").is_err());
        assert!(resolve_artifact(Path::new("/tmp/bundle"), "artifacts/capture.wav").is_ok());
    }

    #[test]
    fn duplicate_sample_paths_are_detectable() {
        let mut paths = BTreeSet::new();
        assert!(paths.insert("samples/a.wav"));
        assert!(!paths.insert("samples/a.wav"));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_parent_directory_is_rejected() {
        use std::os::unix::fs::symlink;
        let root = std::env::temp_dir().join(format!("muse-evidence-root-{}", std::process::id()));
        let outside = root.with_extension("outside");
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&outside);
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("audio.wav"), b"test").unwrap();
        fs::create_dir_all(&root).unwrap();
        symlink(&outside, root.join("linked")).unwrap();
        assert!(ensure_no_symlink_components(&root, "linked/audio.wav").is_err());
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn comparison_binding_rejects_wrong_wav_hashes_and_quality_claims() {
        let json = br#"{
            "schema": "muse-sfz-audio-differential-receipt-v1",
            "candidate": {"sha256": "aaaaaaaa"},
            "reference": {"sha256": "bbbbbbbb"},
            "comparison": {
                "status": "measured_not_quality_qualified",
                "metrics": {"waveform_rmse": 0.0}
            }
        }"#;
        let binding = validate_comparison_binding(json, &"c".repeat(64), &"d".repeat(64));
        assert!(!binding.candidate_output_hash_matches);
        assert!(!binding.reference_output_hash_matches);
        assert!(!binding.quality_qualified);
        assert!(binding.failures.len() >= 2);
    }

    #[test]
    fn comparison_binding_requires_the_known_schema_and_measurement_status() {
        let json = br#"{
            "schema": "unknown-v1",
            "candidate": {"sha256": "aaaaaaaa"},
            "reference": {"sha256": "bbbbbbbb"},
            "comparison": {"status": "not_comparable", "metrics": null}
        }"#;
        let binding = validate_comparison_binding(json, &"a".repeat(64), &"b".repeat(64));
        assert!(!binding.comparison_schema_verified);
        assert!(!binding.metrics_present);
        assert!(!binding.quality_qualified);
        assert_eq!(binding.failures.len(), 4);
    }
}
