use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    process::Command,
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::interoception_qualification::{
    self, ActionsArchiveManifest, VerifiedQualificationAttestation, VERIFIER_POLICY_VERSION,
};

pub const PROMOTION_AUTHORIZATION_ENVELOPE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveActionsVerification {
    pub gate_id: String,
    pub repository: String,
    pub workflow: String,
    pub subject_commit: String,
    pub run_id: u64,
    pub run_attempt: u32,
    pub verified_job_count: usize,
    pub workflow_file_sha256: String,
    pub verification_transport: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotionAuthorizationEnvelope {
    pub schema_version: u16,
    pub verifier_policy_version: String,
    pub verification_mode: String,
    pub structural_and_archive_attestation: VerifiedQualificationAttestation,
    pub live_actions_verification: Vec<LiveActionsVerification>,
    pub decision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct RunIdentity {
    id: u64,
    name: String,
    head_sha: String,
    path: String,
    workflow_id: u64,
    event: String,
    status: String,
    conclusion: Option<String>,
    run_attempt: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct JobsPage {
    total_count: usize,
    jobs: Vec<JobIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct JobIdentity {
    id: u64,
    name: String,
    status: String,
    conclusion: Option<String>,
}

pub fn authorize_promotion_live(
    bundle_path: &Path,
    repo_root: &Path,
    local_fmt_dir: &Path,
    local_test_dir: &Path,
    local_clippy_dir: &Path,
    workspace_ci_dir: &Path,
    showroom_dir: &Path,
    out_path: &Path,
) -> Result<PromotionAuthorizationEnvelope> {
    if out_path.exists() {
        bail!(
            "refusing to overwrite an existing promotion authorization envelope: {}",
            out_path.display()
        );
    }

    let mut live_actions = vec![
        verify_actions_live(workspace_ci_dir, Some(repo_root))?,
        verify_actions_live(showroom_dir, Some(repo_root))?,
    ];
    live_actions.sort_by(|left, right| left.gate_id.cmp(&right.gate_id));

    // The offline authorizer performs the complete bundle/local/archive binding after
    // the live GitHub checks succeed.  Its temporary output never becomes the final
    // promotion artifact; the final envelope below records the live verification mode.
    let temp_path = out_path.with_extension("preauth.tmp.json");
    if temp_path.exists() {
        bail!(
            "temporary promotion artifact already exists; refusing ambiguous overwrite: {}",
            temp_path.display()
        );
    }
    let attestation_result = interoception_qualification::authorize_promotion(
        bundle_path,
        repo_root,
        local_fmt_dir,
        local_test_dir,
        local_clippy_dir,
        workspace_ci_dir,
        showroom_dir,
        &temp_path,
    );
    let _ = fs::remove_file(&temp_path);
    let attestation = attestation_result?;

    let envelope = PromotionAuthorizationEnvelope {
        schema_version: PROMOTION_AUTHORIZATION_ENVELOPE_SCHEMA_VERSION,
        verifier_policy_version: VERIFIER_POLICY_VERSION.into(),
        verification_mode: "live-github-exact-attempt-plus-content-addressed-archive-v1".into(),
        structural_and_archive_attestation: attestation,
        live_actions_verification: live_actions,
        decision: "PromotionAuthorized".into(),
    };
    let bytes = serde_json::to_vec(&envelope).context("serialize promotion authorization envelope")?;
    fs::write(out_path, bytes)
        .with_context(|| format!("write promotion authorization envelope {}", out_path.display()))?;
    Ok(envelope)
}

pub fn verify_actions_live(
    archive_dir: &Path,
    repo_root: Option<&Path>,
) -> Result<LiveActionsVerification> {
    // First verify the durable archive itself.  This rejects stale/missing/tampered
    // objects before any network result is considered.
    let archived_verified = interoception_qualification::verify_actions_archive(archive_dir, repo_root)?;
    let manifest_bytes = fs::read(archive_dir.join("manifest.json"))?;
    let manifest: ActionsArchiveManifest =
        serde_json::from_slice(&manifest_bytes).context("parse Actions archive manifest")?;

    let archived_run: RunIdentity = serde_json::from_slice(&fs::read(archive_dir.join("run.json"))?)
        .context("parse archived run identity for live comparison")?;
    let archived_jobs: JobsPage = serde_json::from_slice(&fs::read(archive_dir.join("jobs.json"))?)
        .context("parse archived jobs identity for live comparison")?;
    if archived_jobs.total_count != archived_jobs.jobs.len() {
        bail!("archived jobs object is incomplete before live verification");
    }

    let live_run_endpoint = format!(
        "repos/{}/actions/runs/{}/attempts/{}",
        manifest.repository, manifest.run_id, manifest.run_attempt
    );
    let live_run_bytes = gh_api(&[&live_run_endpoint])?;
    let live_run: RunIdentity =
        serde_json::from_slice(&live_run_bytes).context("parse live exact-attempt workflow run")?;
    if live_run != archived_run {
        bail!(
            "live GitHub exact-attempt run identity differs from archived run for gate {}",
            manifest.gate_id
        );
    }
    if live_run.status != "completed" || live_run.conclusion.as_deref() != Some("success") {
        bail!("live GitHub run is not a terminal success");
    }

    let jobs_endpoint = format!(
        "repos/{}/actions/runs/{}/attempts/{}/jobs?per_page=100",
        manifest.repository, manifest.run_id, manifest.run_attempt
    );
    let live_pages_bytes = gh_api(&["--paginate", "--slurp", &jobs_endpoint])?;
    let live_pages: Vec<JobsPage> =
        serde_json::from_slice(&live_pages_bytes).context("parse paginated live workflow jobs")?;
    if live_pages.is_empty() {
        bail!("live GitHub jobs response contained no pages");
    }
    let declared_total = live_pages[0].total_count;
    if live_pages
        .iter()
        .any(|page| page.total_count != declared_total)
    {
        bail!("live GitHub jobs pages disagree on total_count");
    }
    let live_jobs: Vec<JobIdentity> = live_pages.into_iter().flat_map(|page| page.jobs).collect();
    if live_jobs.len() != declared_total {
        bail!(
            "live GitHub job pagination incomplete: total_count={} fetched={}",
            declared_total,
            live_jobs.len()
        );
    }

    let archived_by_id: BTreeMap<u64, (&str, &str, Option<&str>)> = archived_jobs
        .jobs
        .iter()
        .map(|job| {
            (
                job.id,
                (job.name.as_str(), job.status.as_str(), job.conclusion.as_deref()),
            )
        })
        .collect();
    let live_by_id: BTreeMap<u64, (&str, &str, Option<&str>)> = live_jobs
        .iter()
        .map(|job| {
            (
                job.id,
                (job.name.as_str(), job.status.as_str(), job.conclusion.as_deref()),
            )
        })
        .collect();
    if archived_by_id != live_by_id {
        bail!(
            "live GitHub jobs differ from archived exact-attempt jobs for gate {}",
            manifest.gate_id
        );
    }

    let workflow_endpoint = format!(
        "repos/{}/contents/{}?ref={}",
        manifest.repository, manifest.workflow_path, manifest.subject_commit
    );
    let live_workflow = gh_api(&[
        "-H",
        "Accept: application/vnd.github.raw+json",
        &workflow_endpoint,
    ])?;
    let live_workflow_sha256 = sha256_bytes(&live_workflow);
    if live_workflow_sha256 != manifest.workflow_file_sha256 {
        bail!(
            "live exact-SHA workflow source differs from archived workflow source for gate {}",
            manifest.gate_id
        );
    }

    // Cross-check the result returned by the offline archive verifier rather than
    // trusting two independently parsed manifests that happen to share strings.
    if archived_verified.gate_id != manifest.gate_id
        || archived_verified.subject_commit != manifest.subject_commit
        || archived_verified.workflow != manifest.workflow
        || archived_verified.run_id != manifest.run_id
        || archived_verified.run_attempt != manifest.run_attempt
    {
        bail!("offline archive verification result disagrees with live-verification manifest");
    }

    Ok(LiveActionsVerification {
        gate_id: manifest.gate_id,
        repository: manifest.repository,
        workflow: manifest.workflow,
        subject_commit: manifest.subject_commit,
        run_id: manifest.run_id,
        run_attempt: manifest.run_attempt,
        verified_job_count: declared_total,
        workflow_file_sha256: live_workflow_sha256,
        verification_transport: "GitHub REST API over gh authenticated HTTPS transport".into(),
    })
}

fn gh_api(args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new("gh")
        .arg("api")
        .args(args)
        .output()
        .with_context(|| format!("execute gh api {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "gh api failed for {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output.stdout)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    use std::fmt::Write as _;
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}
