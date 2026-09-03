use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    interoception_qualification::VERIFIER_POLICY_VERSION,
    symthaea_interoception::QualificationEvidenceBundle,
};

pub const EVIDENCE_CAPSULE_ARCHIVE_SCHEMA_VERSION: u16 = 1;
pub const EVIDENCE_CAPSULE_ARCHIVE_MANIFEST: &str = "capsule-archive.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCapsuleLogicalPaths {
    pub preregistration: String,
    pub experiment_config: String,
    pub input_sequence: String,
    pub evidence_plane: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCapsuleArchiveManifest {
    pub schema_version: u16,
    pub verifier_policy_version: String,
    pub source_commit: String,
    pub qualification_bundle_sha256: String,
    pub logical_paths: EvidenceCapsuleLogicalPaths,
    pub artifact_name_semantics: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedEvidenceCapsuleArchive {
    pub source_commit: String,
    pub qualification_bundle_sha256: String,
    pub archive_manifest_sha256: String,
    pub verified_logical_object_count: usize,
    pub verified_raw_artifact_count: usize,
    pub evidence_plane_is_raw_artifact: bool,
}

pub fn build_capsule_archive_manifest(
    bundle_path: &Path,
    evidence_root: &Path,
    logical_paths: EvidenceCapsuleLogicalPaths,
    repo_root: Option<&Path>,
) -> Result<VerifiedEvidenceCapsuleArchive> {
    let bundle = read_bundle(bundle_path)?;
    validate_bundle(&bundle)?;
    validate_evidence_root(evidence_root)?;
    for (name, relative) in [
        ("preregistration", logical_paths.preregistration.as_str()),
        ("experiment_config", logical_paths.experiment_config.as_str()),
        ("input_sequence", logical_paths.input_sequence.as_str()),
        ("evidence_plane", logical_paths.evidence_plane.as_str()),
    ] {
        validate_relative_path(name, relative)?;
    }
    validate_artifact_paths(&bundle)?;

    let manifest_path = evidence_root.join(EVIDENCE_CAPSULE_ARCHIVE_MANIFEST);
    if manifest_path.exists() {
        bail!(
            "refusing to overwrite existing evidence capsule archive manifest {}",
            manifest_path.display()
        );
    }

    let manifest = EvidenceCapsuleArchiveManifest {
        schema_version: EVIDENCE_CAPSULE_ARCHIVE_SCHEMA_VERSION,
        verifier_policy_version: VERIFIER_POLICY_VERSION.into(),
        source_commit: bundle.source_commit.clone(),
        qualification_bundle_sha256: bundle_sha256(&bundle)?,
        logical_paths,
        artifact_name_semantics:
            "v1: each EvidenceCapsuleManifest.artifacts[].name is a safe relative path beneath evidence_root"
                .into(),
    };
    let bytes = serde_json::to_vec(&manifest).context("serialize capsule archive manifest")?;
    fs::write(&manifest_path, bytes)
        .with_context(|| format!("write capsule archive manifest {}", manifest_path.display()))?;

    verify_capsule_archive(bundle_path, evidence_root, repo_root)
}

pub fn verify_capsule_archive(
    bundle_path: &Path,
    evidence_root: &Path,
    repo_root: Option<&Path>,
) -> Result<VerifiedEvidenceCapsuleArchive> {
    let bundle = read_bundle(bundle_path)?;
    validate_bundle(&bundle)?;
    validate_evidence_root(evidence_root)?;
    validate_artifact_paths(&bundle)?;

    let manifest_path = evidence_root.join(EVIDENCE_CAPSULE_ARCHIVE_MANIFEST);
    let manifest_bytes = fs::read(&manifest_path)
        .with_context(|| format!("read capsule archive manifest {}", manifest_path.display()))?;
    let manifest: EvidenceCapsuleArchiveManifest =
        serde_json::from_slice(&manifest_bytes).context("parse capsule archive manifest")?;

    if manifest.schema_version != EVIDENCE_CAPSULE_ARCHIVE_SCHEMA_VERSION {
        bail!(
            "unsupported evidence capsule archive schema version {}",
            manifest.schema_version
        );
    }
    if manifest.verifier_policy_version != VERIFIER_POLICY_VERSION {
        bail!("unexpected verifier policy version {}", manifest.verifier_policy_version);
    }
    if manifest.source_commit != bundle.source_commit {
        bail!("capsule archive source commit does not match qualification bundle");
    }
    let expected_bundle_sha = bundle_sha256(&bundle)?;
    if manifest.qualification_bundle_sha256 != expected_bundle_sha {
        bail!("capsule archive qualification bundle SHA-256 mismatch");
    }
    if manifest.artifact_name_semantics
        != "v1: each EvidenceCapsuleManifest.artifacts[].name is a safe relative path beneath evidence_root"
    {
        bail!("unknown artifact-name path semantics in capsule archive");
    }

    let logical = [
        (
            "preregistration",
            manifest.logical_paths.preregistration.as_str(),
            bundle.evidence.preregistration_sha256.as_str(),
        ),
        (
            "experiment_config",
            manifest.logical_paths.experiment_config.as_str(),
            bundle.evidence.experiment_config_sha256.as_str(),
        ),
        (
            "input_sequence",
            manifest.logical_paths.input_sequence.as_str(),
            bundle.evidence.input_sequence_sha256.as_str(),
        ),
        (
            "evidence_plane",
            manifest.logical_paths.evidence_plane.as_str(),
            bundle.evidence.evidence_plane_sha256.as_str(),
        ),
    ];
    for (name, relative, expected_digest) in logical {
        validate_relative_path(name, relative)?;
        verify_relative_object(evidence_root, relative, expected_digest)
            .with_context(|| format!("verify evidence capsule logical object {name}"))?;
    }

    for artifact in &bundle.evidence.artifacts {
        verify_relative_object(evidence_root, &artifact.name, &artifact.sha256)
            .with_context(|| format!("verify raw evidence artifact {}", artifact.name))?;
    }

    if let Some(root) = repo_root {
        let head = command_text(root, "git", &["rev-parse", "HEAD"])?;
        if head.trim() != bundle.source_commit {
            bail!("verification checkout HEAD differs from evidence capsule source commit");
        }
        if sha256_file(&root.join("Cargo.lock"))? != bundle.evidence.cargo_lock_sha256 {
            bail!("evidence capsule Cargo.lock SHA-256 differs from exact source checkout");
        }
        if sha256_optional(&root.join("flake.lock"))? != bundle.evidence.flake_lock_sha256 {
            bail!("evidence capsule flake.lock SHA-256 differs from exact source checkout");
        }
        if sha256_optional(&root.join("rust-toolchain.toml"))?
            != bundle.evidence.rust_toolchain_sha256
        {
            bail!(
                "evidence capsule rust-toolchain.toml SHA-256 differs from exact source checkout"
            );
        }
    }

    let evidence_plane_is_raw_artifact = bundle
        .evidence
        .artifacts
        .iter()
        .any(|artifact| artifact.sha256 == bundle.evidence.evidence_plane_sha256);
    if !evidence_plane_is_raw_artifact {
        bail!(
            "evidence_plane_sha256 is not represented in the capsule raw-artifact digest set"
        );
    }

    Ok(VerifiedEvidenceCapsuleArchive {
        source_commit: bundle.source_commit,
        qualification_bundle_sha256: expected_bundle_sha,
        archive_manifest_sha256: sha256_bytes(&manifest_bytes),
        verified_logical_object_count: 4,
        verified_raw_artifact_count: bundle.evidence.artifacts.len(),
        evidence_plane_is_raw_artifact,
    })
}

fn read_bundle(path: &Path) -> Result<QualificationEvidenceBundle> {
    let bytes = fs::read(path).with_context(|| format!("read qualification bundle {}", path.display()))?;
    serde_json::from_slice(&bytes).context("parse qualification evidence bundle")
}

fn validate_bundle(bundle: &QualificationEvidenceBundle) -> Result<()> {
    bundle
        .validate()
        .map_err(|errors| anyhow::anyhow!("qualification bundle validation failed: {}", errors.join("; ")))
}

fn bundle_sha256(bundle: &QualificationEvidenceBundle) -> Result<String> {
    bundle
        .sha256()
        .map_err(|errors| anyhow::anyhow!("qualification bundle digest failed: {}", errors.join("; ")))
}

fn validate_evidence_root(root: &Path) -> Result<()> {
    if !root.is_dir() {
        bail!("evidence root is not a directory: {}", root.display());
    }
    Ok(())
}

fn validate_artifact_paths(bundle: &QualificationEvidenceBundle) -> Result<()> {
    let mut seen = BTreeSet::new();
    for artifact in &bundle.evidence.artifacts {
        validate_relative_path("artifact name", &artifact.name)?;
        if !seen.insert(artifact.name.as_str()) {
            bail!("duplicate artifact path {}", artifact.name);
        }
    }
    Ok(())
}

fn validate_relative_path(name: &str, value: &str) -> Result<()> {
    let path = Path::new(value);
    if value.trim().is_empty() || path.is_absolute() {
        bail!("{name} must be a non-empty relative path");
    }
    for component in path.components() {
        if !matches!(component, Component::Normal(_)) {
            bail!("{name} contains a disallowed path component: {value}");
        }
    }
    Ok(())
}

fn verify_relative_object(root: &Path, relative: &str, expected_digest: &str) -> Result<()> {
    validate_relative_path("evidence object path", relative)?;
    let path = root.join(relative);
    if !path.is_file() {
        bail!("evidence object is missing or not a regular file: {}", path.display());
    }
    let observed = sha256_file(&path)?;
    if observed != expected_digest {
        bail!(
            "evidence object SHA-256 mismatch for {}: expected {}, observed {}",
            relative,
            expected_digest,
            observed
        );
    }
    Ok(())
}

fn sha256_optional(path: &Path) -> Result<Option<String>> {
    if path.exists() {
        Ok(Some(sha256_file(path)?))
    } else {
        Ok(None)
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hex_digest(hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex_digest(Sha256::digest(bytes))
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    use std::fmt::Write as _;
    let bytes = bytes.as_ref();
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn command_text(repo_root: &Path, program: &str, args: &[&str]) -> Result<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("run {program} {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "command failed: {program} {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout).context("identity command emitted non-UTF-8 output")
}

pub fn manifest_path(evidence_root: &Path) -> PathBuf {
    evidence_root.join(EVIDENCE_CAPSULE_ARCHIVE_MANIFEST)
}
