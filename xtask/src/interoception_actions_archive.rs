use std::{fs, io::Read, path::Path};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::interoception_qualification::{
    verify_actions_archive, ActionsArchiveManifest, VerifiedActionsGate,
    ACTIONS_ARCHIVE_SCHEMA_VERSION, VERIFIER_POLICY_VERSION,
};

#[derive(Debug, Deserialize)]
struct RunIdentity {
    id: u64,
    name: String,
    head_sha: String,
    path: String,
    run_attempt: u32,
}

/// Build `manifest.json` from immutable raw GitHub API responses already saved as
/// `run.json`, `jobs.json`, plus the exact workflow source saved as `workflow.yml`.
///
/// The builder never fetches a "latest" attempt.  The supplied `run.json` itself
/// carries the exact run ID and attempt that become part of the archive identity.
pub fn build_actions_archive(
    archive_dir: &Path,
    gate_id: &str,
    repository: &str,
    repo_root: Option<&Path>,
) -> Result<VerifiedActionsGate> {
    if repository != "Luminous-Dynamics/symthaea" {
        bail!("unexpected repository identity {repository}");
    }

    let manifest_path = archive_dir.join("manifest.json");
    if manifest_path.exists() {
        bail!(
            "refusing to overwrite existing Actions archive manifest {}",
            manifest_path.display()
        );
    }

    let run_path = archive_dir.join("run.json");
    let jobs_path = archive_dir.join("jobs.json");
    let workflow_file = archive_dir.join("workflow.yml");
    for path in [&run_path, &jobs_path, &workflow_file] {
        if !path.is_file() {
            bail!("required Actions archive object is missing: {}", path.display());
        }
    }

    let run_bytes = fs::read(&run_path)
        .with_context(|| format!("read archived run JSON {}", run_path.display()))?;
    let run: RunIdentity = serde_json::from_slice(&run_bytes).context("parse archived run identity")?;
    let (workflow, workflow_path) = required_actions_identity(gate_id)?;
    if run.name != workflow || run.path != workflow_path {
        bail!(
            "run identity does not match gate {gate_id}: workflow={:?}, path={:?}",
            run.name,
            run.path
        );
    }
    validate_sha1("run.head_sha", &run.head_sha)?;
    if run.id == 0 || run.run_attempt == 0 {
        bail!("run ID and run attempt must both be positive");
    }

    let manifest = ActionsArchiveManifest {
        schema_version: ACTIONS_ARCHIVE_SCHEMA_VERSION,
        verifier_policy_version: VERIFIER_POLICY_VERSION.into(),
        gate_id: gate_id.into(),
        repository: repository.into(),
        workflow: workflow.into(),
        workflow_path: workflow_path.into(),
        subject_commit: run.head_sha,
        run_id: run.id,
        run_attempt: run.run_attempt,
        run_json_sha256: sha256_bytes(&run_bytes),
        jobs_json_sha256: sha256_file(&jobs_path)?,
        workflow_file_sha256: sha256_file(&workflow_file)?,
    };

    let manifest_bytes = serde_json::to_vec(&manifest).context("serialize Actions archive manifest")?;
    fs::write(&manifest_path, manifest_bytes)
        .with_context(|| format!("write Actions archive manifest {}", manifest_path.display()))?;

    // Building the manifest is not enough: immediately execute the same fail-closed
    // verifier used by promotion authorization so an incomplete jobs archive, wrong
    // conclusion, wrong attempt, or workflow/source mismatch cannot be mistaken for a
    // completed archive.
    verify_actions_archive(archive_dir, repo_root)
}

fn required_actions_identity(gate_id: &str) -> Result<(&'static str, &'static str)> {
    match gate_id {
        "workspace_ci" => Ok(("CI", ".github/workflows/ci.yml")),
        "showroom_integrity" => Ok(("Showroom Integrity", ".github/workflows/showroom-integrity.yml")),
        other => bail!("unsupported Actions qualification gate {other}"),
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
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

fn validate_sha1(name: &str, value: &str) -> Result<()> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{name} must be 40 lowercase hexadecimal characters");
    }
    Ok(())
}
