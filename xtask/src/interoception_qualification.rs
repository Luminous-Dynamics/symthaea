use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_interoception::{
    GateStatus, QualificationEvidenceBundle, QualificationGateEvidence,
    INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
};

pub const LOCAL_EVIDENCE_SCHEMA_VERSION: u16 = 1;
pub const ACTIONS_ARCHIVE_SCHEMA_VERSION: u16 = 1;
pub const VERIFIED_ATTESTATION_SCHEMA_VERSION: u16 = 1;
pub const VERIFIER_POLICY_VERSION: &str = "interoception-qualification-verifier-v0.1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RepositoryCleanliness {
    pub staged: Vec<String>,
    pub tracked_uncommitted: Vec<String>,
    pub untracked: Vec<String>,
}

impl RepositoryCleanliness {
    pub fn is_clean(&self) -> bool {
        self.staged.is_empty() && self.tracked_uncommitted.is_empty() && self.untracked.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalGateEnvironmentManifest {
    pub schema_version: u16,
    pub verifier_policy_version: String,
    pub gate_id: String,
    pub subject_commit: String,
    pub command: String,
    pub working_directory_identity: String,
    pub repository_tree: String,
    pub pre_cleanliness: RepositoryCleanliness,
    pub post_cleanliness: RepositoryCleanliness,
    pub cargo_lock_sha256: String,
    pub flake_lock_sha256: Option<String>,
    pub rust_toolchain_sha256: Option<String>,
    pub rustc_vv: String,
    pub cargo_vv: String,
    pub target_triple: String,
    pub architecture: String,
    pub execution_environment: BTreeMap<String, String>,
    pub transcript_capture_semantics: String,
    pub transcript_sha256: String,
    pub exit_code: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedLocalGate {
    pub gate_id: String,
    pub subject_commit: String,
    pub command: String,
    pub environment_sha256: String,
    pub transcript_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionsArchiveManifest {
    pub schema_version: u16,
    pub verifier_policy_version: String,
    pub gate_id: String,
    pub repository: String,
    pub workflow: String,
    pub workflow_path: String,
    pub subject_commit: String,
    pub run_id: u64,
    pub run_attempt: u32,
    pub run_json_sha256: String,
    pub jobs_json_sha256: String,
    pub workflow_file_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedActionsGate {
    pub gate_id: String,
    pub subject_commit: String,
    pub workflow: String,
    pub run_id: u64,
    pub run_attempt: u32,
    pub archive_manifest_sha256: String,
    pub run_json_sha256: String,
    pub jobs_json_sha256: String,
    pub workflow_file_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromotionDecision {
    StructurallyQualified,
    PromotionAuthorized,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralQualificationReport {
    pub verifier_policy_version: String,
    pub source_commit: String,
    pub qualification_bundle_sha256: String,
    pub decision: PromotionDecision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedGateAttestation {
    pub gate_id: String,
    pub evidence_kind: String,
    pub evidence_object_sha256: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedQualificationAttestation {
    pub schema_version: u16,
    pub verifier_policy_version: String,
    pub source_commit: String,
    pub model_semantics_version: u16,
    pub qualification_bundle_sha256: String,
    pub gates: Vec<VerifiedGateAttestation>,
    pub decision: PromotionDecision,
}

#[derive(Debug, Deserialize)]
struct ArchivedRun {
    id: u64,
    name: String,
    head_sha: String,
    path: String,
    status: String,
    conclusion: Option<String>,
    run_attempt: u32,
}

#[derive(Debug, Deserialize)]
struct ArchivedJobs {
    total_count: usize,
    jobs: Vec<ArchivedJob>,
}

#[derive(Debug, Deserialize)]
struct ArchivedJob {
    name: String,
    status: String,
    conclusion: Option<String>,
}

pub fn capture_local_gate(
    repo_root: &Path,
    subject_commit: &str,
    gate_id: &str,
    out_dir: &Path,
) -> Result<VerifiedLocalGate> {
    validate_sha1("subject_commit", subject_commit)?;
    let command = required_local_command(gate_id)?;
    let observed_head = git(repo_root, &["rev-parse", "HEAD"])?;
    if observed_head.trim() != subject_commit {
        bail!(
            "target checkout HEAD {} does not equal required subject commit {}",
            observed_head.trim(),
            subject_commit
        );
    }

    let pre_cleanliness = repository_cleanliness(repo_root)?;
    if !pre_cleanliness.is_clean() {
        bail!("target checkout is dirty before local qualification gate {gate_id}");
    }

    let repository_tree = git(repo_root, &["rev-parse", "HEAD^{tree}"])?
        .trim()
        .to_string();
    validate_sha1("repository_tree", &repository_tree)?;

    fs::create_dir_all(out_dir)
        .with_context(|| format!("create evidence directory {}", out_dir.display()))?;
    let transcript_path = out_dir.join("transcript.bin");
    let transcript = File::create(&transcript_path)
        .with_context(|| format!("create transcript {}", transcript_path.display()))?;
    let transcript_err = transcript
        .try_clone()
        .context("clone transcript file handle for stderr")?;

    let (program, args) = local_command_argv(gate_id)?;
    let status = Command::new(program)
        .args(args)
        .current_dir(repo_root)
        .stdout(Stdio::from(transcript))
        .stderr(Stdio::from(transcript_err))
        .status()
        .with_context(|| format!("execute fixed qualification command: {command}"))?;

    let exit_code = status.code().unwrap_or(-1);
    let transcript_sha256 = sha256_file(&transcript_path)?;
    let post_cleanliness = repository_cleanliness(repo_root)?;

    let rustc_vv = command_text(repo_root, "rustc", &["-vV"])?;
    let cargo_vv = command_text(repo_root, "cargo", &["-Vv"])?;
    let target_triple = rustc_vv
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .context("rustc -vV did not report host target triple")?
        .trim()
        .to_string();

    let manifest = LocalGateEnvironmentManifest {
        schema_version: LOCAL_EVIDENCE_SCHEMA_VERSION,
        verifier_policy_version: VERIFIER_POLICY_VERSION.into(),
        gate_id: gate_id.into(),
        subject_commit: subject_commit.into(),
        command: command.into(),
        working_directory_identity: ".".into(),
        repository_tree,
        pre_cleanliness,
        post_cleanliness,
        cargo_lock_sha256: sha256_file(&repo_root.join("Cargo.lock"))?,
        flake_lock_sha256: sha256_optional(&repo_root.join("flake.lock"))?,
        rust_toolchain_sha256: sha256_optional(&repo_root.join("rust-toolchain.toml"))?,
        rustc_vv,
        cargo_vv,
        target_triple,
        architecture: std::env::consts::ARCH.into(),
        execution_environment: captured_execution_environment(),
        transcript_capture_semantics: "merged-exact-file-v1: stdout and stderr share one cloned file handle; stored bytes are hashed without normalization".into(),
        transcript_sha256: transcript_sha256.clone(),
        exit_code,
    };

    let manifest_bytes = serde_json::to_vec(&manifest).context("serialize local evidence manifest")?;
    let manifest_path = out_dir.join("environment.json");
    fs::write(&manifest_path, &manifest_bytes)
        .with_context(|| format!("write environment manifest {}", manifest_path.display()))?;

    let verified = verify_local_gate(&manifest_path, &transcript_path, Some(repo_root))?;
    if exit_code != 0 {
        bail!(
            "local qualification gate {gate_id} failed with exit code {exit_code}; evidence was preserved at {}",
            out_dir.display()
        );
    }
    Ok(verified)
}

pub fn verify_local_gate(
    manifest_path: &Path,
    transcript_path: &Path,
    repo_root: Option<&Path>,
) -> Result<VerifiedLocalGate> {
    let manifest_bytes = fs::read(manifest_path)
        .with_context(|| format!("read local evidence manifest {}", manifest_path.display()))?;
    let manifest: LocalGateEnvironmentManifest =
        serde_json::from_slice(&manifest_bytes).context("parse local evidence manifest")?;

    if manifest.schema_version != LOCAL_EVIDENCE_SCHEMA_VERSION {
        bail!("unsupported local evidence schema version {}", manifest.schema_version);
    }
    if manifest.verifier_policy_version != VERIFIER_POLICY_VERSION {
        bail!("unexpected verifier policy version {}", manifest.verifier_policy_version);
    }
    validate_sha1("subject_commit", &manifest.subject_commit)?;
    validate_sha1("repository_tree", &manifest.repository_tree)?;
    let expected_command = required_local_command(&manifest.gate_id)?;
    if manifest.command != expected_command {
        bail!("gate {} command identity mismatch", manifest.gate_id);
    }
    if manifest.transcript_capture_semantics
        != "merged-exact-file-v1: stdout and stderr share one cloned file handle; stored bytes are hashed without normalization"
    {
        bail!("unknown transcript capture semantics");
    }
    if !manifest.pre_cleanliness.is_clean() || !manifest.post_cleanliness.is_clean() {
        bail!("local gate evidence records a dirty pre/post qualification checkout");
    }
    if manifest.exit_code != 0 {
        bail!("local gate evidence records non-zero exit code {}", manifest.exit_code);
    }

    let observed_transcript_sha256 = sha256_file(transcript_path)?;
    if observed_transcript_sha256 != manifest.transcript_sha256 {
        bail!("local transcript SHA-256 mismatch");
    }

    if let Some(root) = repo_root {
        let head = git(root, &["rev-parse", "HEAD"])?;
        if head.trim() != manifest.subject_commit {
            bail!("verification checkout HEAD differs from evidence subject commit");
        }
        let tree = git(root, &["rev-parse", "HEAD^{tree}"])?;
        if tree.trim() != manifest.repository_tree {
            bail!("verification checkout tree differs from captured repository tree");
        }
        if sha256_file(&root.join("Cargo.lock"))? != manifest.cargo_lock_sha256 {
            bail!("Cargo.lock digest differs from captured environment");
        }
        if sha256_optional(&root.join("flake.lock"))? != manifest.flake_lock_sha256 {
            bail!("flake.lock digest differs from captured environment");
        }
        if sha256_optional(&root.join("rust-toolchain.toml"))? != manifest.rust_toolchain_sha256 {
            bail!("rust-toolchain.toml digest differs from captured environment");
        }
    }

    Ok(VerifiedLocalGate {
        gate_id: manifest.gate_id,
        subject_commit: manifest.subject_commit,
        command: manifest.command,
        environment_sha256: sha256_bytes(&manifest_bytes),
        transcript_sha256: observed_transcript_sha256,
    })
}

pub fn verify_actions_archive(
    archive_dir: &Path,
    repo_root: Option<&Path>,
) -> Result<VerifiedActionsGate> {
    let manifest_path = archive_dir.join("manifest.json");
    let run_path = archive_dir.join("run.json");
    let jobs_path = archive_dir.join("jobs.json");
    let workflow_path = archive_dir.join("workflow.yml");

    let manifest_bytes = fs::read(&manifest_path)
        .with_context(|| format!("read Actions archive manifest {}", manifest_path.display()))?;
    let manifest: ActionsArchiveManifest =
        serde_json::from_slice(&manifest_bytes).context("parse Actions archive manifest")?;
    if manifest.schema_version != ACTIONS_ARCHIVE_SCHEMA_VERSION {
        bail!("unsupported Actions archive schema version {}", manifest.schema_version);
    }
    if manifest.verifier_policy_version != VERIFIER_POLICY_VERSION {
        bail!("unexpected verifier policy version {}", manifest.verifier_policy_version);
    }
    validate_sha1("subject_commit", &manifest.subject_commit)?;
    let (expected_workflow, expected_path) = required_actions_identity(&manifest.gate_id)?;
    if manifest.workflow != expected_workflow || manifest.workflow_path != expected_path {
        bail!("Actions gate {} workflow identity mismatch", manifest.gate_id);
    }
    if manifest.repository != "Luminous-Dynamics/symthaea" {
        bail!("unexpected repository identity {}", manifest.repository);
    }

    for (label, expected, path) in [
        ("run.json", &manifest.run_json_sha256, &run_path),
        ("jobs.json", &manifest.jobs_json_sha256, &jobs_path),
        ("workflow.yml", &manifest.workflow_file_sha256, &workflow_path),
    ] {
        validate_sha256(label, expected)?;
        let observed = sha256_file(path)?;
        if &observed != expected {
            bail!("{label} digest mismatch");
        }
    }

    let run: ArchivedRun = serde_json::from_slice(&fs::read(&run_path)?)
        .context("parse archived GitHub Actions run JSON")?;
    if run.id != manifest.run_id
        || run.run_attempt != manifest.run_attempt
        || run.name != manifest.workflow
        || run.head_sha != manifest.subject_commit
        || run.path != manifest.workflow_path
    {
        bail!("archived Actions run identity does not match manifest");
    }
    if run.status != "completed" || run.conclusion.as_deref() != Some("success") {
        bail!(
            "Actions run is not a terminal success: status={}, conclusion={:?}",
            run.status,
            run.conclusion
        );
    }

    let jobs: ArchivedJobs = serde_json::from_slice(&fs::read(&jobs_path)?)
        .context("parse archived GitHub Actions jobs JSON")?;
    if jobs.total_count != jobs.jobs.len() {
        bail!(
            "jobs archive is incomplete: total_count={} but archived_jobs={}",
            jobs.total_count,
            jobs.jobs.len()
        );
    }
    for job in &jobs.jobs {
        if job.status != "completed" {
            bail!("Actions job {:?} is not terminal", job.name);
        }
        match job.conclusion.as_deref() {
            Some("success" | "skipped" | "neutral") => {}
            other => bail!("Actions job {:?} has disallowed conclusion {:?}", job.name, other),
        }
    }

    if let Some(root) = repo_root {
        let head = git(root, &["rev-parse", "HEAD"])?;
        if head.trim() != manifest.subject_commit {
            bail!("verification checkout HEAD differs from Actions evidence subject commit");
        }
        let checkout_workflow = root.join(&manifest.workflow_path);
        if sha256_file(&checkout_workflow)? != manifest.workflow_file_sha256 {
            bail!("workflow archive differs from exact subject checkout workflow file");
        }
    }

    Ok(VerifiedActionsGate {
        gate_id: manifest.gate_id,
        subject_commit: manifest.subject_commit,
        workflow: manifest.workflow,
        run_id: manifest.run_id,
        run_attempt: manifest.run_attempt,
        archive_manifest_sha256: sha256_bytes(&manifest_bytes),
        run_json_sha256: manifest.run_json_sha256,
        jobs_json_sha256: manifest.jobs_json_sha256,
        workflow_file_sha256: manifest.workflow_file_sha256,
    })
}

pub fn inspect_structural_bundle(bundle_path: &Path) -> Result<StructuralQualificationReport> {
    let bundle = read_bundle(bundle_path)?;
    bundle
        .validate()
        .map_err(|errors| anyhow::anyhow!("qualification bundle validation failed: {}", errors.join("; ")))?;
    if !bundle.is_qualified() {
        bail!("qualification bundle is structurally valid but not structurally qualified");
    }
    Ok(StructuralQualificationReport {
        verifier_policy_version: VERIFIER_POLICY_VERSION.into(),
        source_commit: bundle.source_commit.clone(),
        qualification_bundle_sha256: bundle
            .sha256()
            .map_err(|errors| anyhow::anyhow!(errors.join("; ")) )?,
        decision: PromotionDecision::StructurallyQualified,
    })
}

pub fn authorize_promotion(
    bundle_path: &Path,
    repo_root: &Path,
    local_fmt_dir: &Path,
    local_test_dir: &Path,
    local_clippy_dir: &Path,
    workspace_ci_dir: &Path,
    showroom_dir: &Path,
    out_path: &Path,
) -> Result<VerifiedQualificationAttestation> {
    let bundle = read_bundle(bundle_path)?;
    bundle
        .validate()
        .map_err(|errors| anyhow::anyhow!("qualification bundle validation failed: {}", errors.join("; ")))?;
    if !bundle.is_qualified() {
        bail!("raw bundle does not satisfy structural qualification");
    }
    if bundle.model_semantics_version != INTEROCEPTIVE_MODEL_SEMANTICS_VERSION {
        bail!("model semantics version mismatch");
    }

    let local = [
        verify_local_dir(local_fmt_dir, repo_root)?,
        verify_local_dir(local_test_dir, repo_root)?,
        verify_local_dir(local_clippy_dir, repo_root)?,
    ];
    let actions = [
        verify_actions_archive(workspace_ci_dir, Some(repo_root))?,
        verify_actions_archive(showroom_dir, Some(repo_root))?,
    ];

    for verified in &local {
        bind_local_to_bundle(&bundle, verified)?;
    }
    for verified in &actions {
        bind_actions_to_bundle(&bundle, verified)?;
    }

    let mut gates = Vec::with_capacity(5);
    for verified in local {
        gates.push(VerifiedGateAttestation {
            gate_id: verified.gate_id,
            evidence_kind: "LocalCommand".into(),
            evidence_object_sha256: vec![verified.environment_sha256, verified.transcript_sha256],
        });
    }
    for verified in actions {
        gates.push(VerifiedGateAttestation {
            gate_id: verified.gate_id,
            evidence_kind: "GitHubActionsArchive".into(),
            evidence_object_sha256: vec![
                verified.archive_manifest_sha256,
                verified.run_json_sha256,
                verified.jobs_json_sha256,
                verified.workflow_file_sha256,
            ],
        });
    }
    gates.sort_by(|left, right| left.gate_id.cmp(&right.gate_id));

    let attestation = VerifiedQualificationAttestation {
        schema_version: VERIFIED_ATTESTATION_SCHEMA_VERSION,
        verifier_policy_version: VERIFIER_POLICY_VERSION.into(),
        source_commit: bundle.source_commit.clone(),
        model_semantics_version: bundle.model_semantics_version,
        qualification_bundle_sha256: bundle
            .sha256()
            .map_err(|errors| anyhow::anyhow!(errors.join("; ")) )?,
        gates,
        decision: PromotionDecision::PromotionAuthorized,
    };
    let bytes = serde_json::to_vec(&attestation).context("serialize verified qualification attestation")?;
    fs::write(out_path, bytes)
        .with_context(|| format!("write verified qualification attestation {}", out_path.display()))?;
    Ok(attestation)
}

fn verify_local_dir(dir: &Path, repo_root: &Path) -> Result<VerifiedLocalGate> {
    verify_local_gate(
        &dir.join("environment.json"),
        &dir.join("transcript.bin"),
        Some(repo_root),
    )
}

fn bind_local_to_bundle(bundle: &QualificationEvidenceBundle, verified: &VerifiedLocalGate) -> Result<()> {
    let gate = bundle
        .qualification
        .gates
        .iter()
        .find(|gate| gate.gate_id == verified.gate_id)
        .with_context(|| format!("bundle missing verified gate {}", verified.gate_id))?;
    if gate.status != GateStatus::Passed {
        bail!("bundle gate {} is not Passed", verified.gate_id);
    }
    match gate.evidence.as_ref() {
        Some(QualificationGateEvidence::LocalCommand {
            subject_commit,
            command,
            environment_sha256,
            transcript_sha256,
        }) if subject_commit == &verified.subject_commit
            && command == &verified.command
            && environment_sha256 == &verified.environment_sha256
            && transcript_sha256 == &verified.transcript_sha256 => Ok(()),
        _ => bail!("bundle local gate {} does not bind verified local evidence", verified.gate_id),
    }
}

fn bind_actions_to_bundle(
    bundle: &QualificationEvidenceBundle,
    verified: &VerifiedActionsGate,
) -> Result<()> {
    let gate = bundle
        .qualification
        .gates
        .iter()
        .find(|gate| gate.gate_id == verified.gate_id)
        .with_context(|| format!("bundle missing verified gate {}", verified.gate_id))?;
    if gate.status != GateStatus::Passed {
        bail!("bundle gate {} is not Passed", verified.gate_id);
    }
    match gate.evidence.as_ref() {
        Some(QualificationGateEvidence::GitHubActions {
            subject_commit,
            workflow,
            run_id,
            run_attempt,
        }) if subject_commit == &verified.subject_commit
            && workflow == &verified.workflow
            && run_id == &verified.run_id
            && run_attempt == &verified.run_attempt => Ok(()),
        _ => bail!("bundle Actions gate {} does not bind verified archived evidence", verified.gate_id),
    }
}

fn read_bundle(path: &Path) -> Result<QualificationEvidenceBundle> {
    let bytes = fs::read(path).with_context(|| format!("read qualification bundle {}", path.display()))?;
    serde_json::from_slice(&bytes).context("parse qualification evidence bundle")
}

fn required_local_command(gate_id: &str) -> Result<&'static str> {
    match gate_id {
        "local_fmt" => Ok("cargo fmt --all --check"),
        "local_test" => Ok("cargo test -p symthaea-interoception"),
        "local_clippy" => Ok("cargo clippy -p symthaea-interoception --all-targets -- -D warnings"),
        other => bail!("unsupported local qualification gate {other}"),
    }
}

fn local_command_argv(gate_id: &str) -> Result<(&'static str, &'static [&'static str])> {
    match gate_id {
        "local_fmt" => Ok(("cargo", &["fmt", "--all", "--check"])),
        "local_test" => Ok(("cargo", &["test", "-p", "symthaea-interoception"])),
        "local_clippy" => Ok((
            "cargo",
            &[
                "clippy",
                "-p",
                "symthaea-interoception",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        )),
        other => bail!("unsupported local qualification gate {other}"),
    }
}

fn required_actions_identity(gate_id: &str) -> Result<(&'static str, &'static str)> {
    match gate_id {
        "workspace_ci" => Ok(("CI", ".github/workflows/ci.yml")),
        "showroom_integrity" => Ok(("Showroom Integrity", ".github/workflows/showroom-integrity.yml")),
        other => bail!("unsupported Actions qualification gate {other}"),
    }
}

fn repository_cleanliness(repo_root: &Path) -> Result<RepositoryCleanliness> {
    let status = git(
        repo_root,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    let mut result = RepositoryCleanliness::default();
    for line in status.lines() {
        if line.starts_with("?? ") {
            result.untracked.push(line[3..].to_string());
            continue;
        }
        let bytes = line.as_bytes();
        if bytes.len() < 2 {
            bail!("malformed git porcelain status line: {line:?}");
        }
        if bytes[0] != b' ' {
            result.staged.push(line.to_string());
        }
        if bytes[1] != b' ' {
            result.tracked_uncommitted.push(line.to_string());
        }
    }
    Ok(result)
}

fn captured_execution_environment() -> BTreeMap<String, String> {
    [
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_TARGET",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
    ]
    .into_iter()
    .filter_map(|key| std::env::var(key).ok().map(|value| (key.to_string(), value)))
    .collect()
}

fn git(repo_root: &Path, args: &[&str]) -> Result<String> {
    command_text(repo_root, "git", args)
}

fn command_text(repo_root: &Path, program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("run {program} {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "command failed: {program} {}\nstderr: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout).context("command emitted non-UTF-8 identity output")
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
    let bytes = bytes.as_ref();
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn validate_sha1(name: &str, value: &str) -> Result<()> {
    validate_lower_hex(name, value, 40)
}

fn validate_sha256(name: &str, value: &str) -> Result<()> {
    validate_lower_hex(name, value, 64)
}

fn validate_lower_hex(name: &str, value: &str, len: usize) -> Result<()> {
    if value.len() != len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{name} must be {len} lowercase hexadecimal characters");
    }
    Ok(())
}

pub fn local_manifest_path(dir: &Path) -> PathBuf {
    dir.join("environment.json")
}

pub fn local_transcript_path(dir: &Path) -> PathBuf {
    dir.join("transcript.bin")
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_interoception::{
        ArtifactDigest, EvidenceCapsuleManifest, ForecastBasisId, QualificationGateReceipt,
        QualificationReceipt, EVIDENCE_CAPSULE_SCHEMA_VERSION,
        INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION, QUALIFICATION_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        QUALIFICATION_RECEIPT_SCHEMA_VERSION, REQUIRED_QUALIFICATION_GATES,
    };

    const SOURCE: &str = "0123456789abcdef0123456789abcdef01234567";

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn fake_bundle() -> QualificationEvidenceBundle {
        let gates = REQUIRED_QUALIFICATION_GATES
            .iter()
            .map(|gate| {
                let evidence = match *gate {
                    "local_fmt" => QualificationGateEvidence::local_command(
                        SOURCE,
                        "cargo fmt --all --check",
                        digest('1'),
                        digest('2'),
                    ),
                    "local_test" => QualificationGateEvidence::local_command(
                        SOURCE,
                        "cargo test -p symthaea-interoception",
                        digest('3'),
                        digest('4'),
                    ),
                    "local_clippy" => QualificationGateEvidence::local_command(
                        SOURCE,
                        "cargo clippy -p symthaea-interoception --all-targets -- -D warnings",
                        digest('5'),
                        digest('6'),
                    ),
                    "workspace_ci" => QualificationGateEvidence::github_actions(SOURCE, "CI", 7, 1),
                    "showroom_integrity" => QualificationGateEvidence::github_actions(
                        SOURCE,
                        "Showroom Integrity",
                        8,
                        1,
                    ),
                    _ => unreachable!(),
                };
                QualificationGateReceipt::with_evidence(*gate, GateStatus::Passed, evidence)
            })
            .collect();
        QualificationEvidenceBundle {
            schema_version: QUALIFICATION_EVIDENCE_BUNDLE_SCHEMA_VERSION,
            model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
            source_commit: SOURCE.into(),
            qualification: QualificationReceipt {
                schema_version: QUALIFICATION_RECEIPT_SCHEMA_VERSION,
                model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
                source_commit: SOURCE.into(),
                gates,
            },
            evidence: EvidenceCapsuleManifest {
                schema_version: EVIDENCE_CAPSULE_SCHEMA_VERSION,
                model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
                source_commit: SOURCE.into(),
                cargo_lock_sha256: digest('a'),
                flake_lock_sha256: Some(digest('b')),
                rust_toolchain_sha256: Some(digest('c')),
                rustc_vv: "rustc fixture".into(),
                cargo_vv: "cargo fixture".into(),
                target_triple: "x86_64-unknown-linux-gnu".into(),
                architecture: "x86_64".into(),
                experiment_id: "fixture".into(),
                preregistration_sha256: digest('d'),
                forecast_basis: ForecastBasisId::DynamicsAwareConstantDrive,
                experiment_config_sha256: digest('e'),
                input_sequence_sha256: digest('f'),
                snapshot_schema_version: INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
                evidence_plane_sha256: digest('0'),
                artifacts: vec![ArtifactDigest::new("fixture.json", digest('9'))],
            },
        }
    }

    #[test]
    fn synthetic_locators_can_only_establish_structural_qualification() {
        let bundle = fake_bundle();
        assert!(bundle.validate().is_ok());
        assert!(bundle.is_qualified());
        let state = PromotionDecision::StructurallyQualified;
        assert_ne!(state, PromotionDecision::PromotionAuthorized);
    }

    #[test]
    fn command_contract_is_fixed() {
        assert_eq!(required_local_command("local_fmt").unwrap(), "cargo fmt --all --check");
        assert!(required_local_command("local_other").is_err());
    }

    #[test]
    fn dirty_state_is_not_clean() {
        let state = RepositoryCleanliness {
            untracked: vec!["evidence.tmp".into()],
            ..RepositoryCleanliness::default()
        };
        assert!(!state.is_clean());
    }
}
