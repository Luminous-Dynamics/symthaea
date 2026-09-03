use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    process::Command,
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    interoception_capsule_archive::{self, VerifiedEvidenceCapsuleArchive},
    interoception_qualification::{
        self, ActionsArchiveManifest, VerifiedActionsGate, VerifiedLocalGate,
        VERIFIER_POLICY_VERSION,
    },
    symthaea_interoception::{
        GateStatus, QualificationEvidenceBundle, QualificationGateEvidence,
        INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
    },
};

pub const PROMOTION_AUTHORIZATION_ENVELOPE_SCHEMA_VERSION: u16 = 1;
pub const FROZEN_V01_SOURCE_COMMIT: &str = "1007949d5c60fd2d7dd650e8bb4521e2b2803c48";

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
    pub workflow_git_blob_sha1: String,
    pub verification_transport: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedGateBinding {
    pub gate_id: String,
    pub evidence_kind: String,
    pub evidence_object_sha256: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedQualificationEvidence {
    pub source_commit: String,
    pub model_semantics_version: u16,
    pub qualification_bundle_sha256: String,
    pub evidence_capsule_archive: VerifiedEvidenceCapsuleArchive,
    pub gates: Vec<VerifiedGateBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotionAuthorizationEnvelope {
    pub schema_version: u16,
    pub verifier_policy_version: String,
    pub verification_mode: String,
    pub verified_evidence: VerifiedQualificationEvidence,
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

#[derive(Debug, Deserialize)]
struct ContentIdentity {
    sha: String,
}

pub fn authorize_promotion_live(
    bundle_path: &Path,
    repo_root: &Path,
    evidence_root: &Path,
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

    let bundle = read_canonical_bundle(bundle_path)?;
    bundle
        .validate()
        .map_err(|errors| anyhow::anyhow!("qualification bundle validation failed: {}", errors.join("; ")))?;
    if !bundle.is_qualified() {
        bail!("raw bundle does not satisfy structural qualification");
    }
    if bundle.source_commit != FROZEN_V01_SOURCE_COMMIT {
        bail!(
            "verifier policy {} is frozen to source {}, but bundle names {}",
            VERIFIER_POLICY_VERSION,
            FROZEN_V01_SOURCE_COMMIT,
            bundle.source_commit
        );
    }
    if bundle.model_semantics_version != INTEROCEPTIVE_MODEL_SEMANTICS_VERSION {
        bail!("model semantics version mismatch");
    }

    let capsule = interoception_capsule_archive::verify_capsule_archive(
        bundle_path,
        evidence_root,
        Some(repo_root),
    )?;

    let local = [
        verify_local_dir(local_fmt_dir, repo_root)?,
        verify_local_dir(local_test_dir, repo_root)?,
        verify_local_dir(local_clippy_dir, repo_root)?,
    ];
    let archives = [
        interoception_qualification::verify_actions_archive(workspace_ci_dir, Some(repo_root))?,
        interoception_qualification::verify_actions_archive(showroom_dir, Some(repo_root))?,
    ];

    for verified in &local {
        bind_local_to_bundle(&bundle, verified)?;
    }
    for verified in &archives {
        bind_actions_to_bundle(&bundle, verified)?;
    }

    let mut live_actions = vec![
        verify_actions_live(workspace_ci_dir, Some(repo_root))?,
        verify_actions_live(showroom_dir, Some(repo_root))?,
    ];
    live_actions.sort_by(|left, right| left.gate_id.cmp(&right.gate_id));
    for live in &live_actions {
        if live.subject_commit != FROZEN_V01_SOURCE_COMMIT {
            bail!("live Actions verification resolved a non-frozen source commit");
        }
    }

    let mut gates = Vec::with_capacity(5);
    for verified in local {
        gates.push(VerifiedGateBinding {
            gate_id: verified.gate_id,
            evidence_kind: "LocalCommand".into(),
            evidence_object_sha256: vec![verified.environment_sha256, verified.transcript_sha256],
        });
    }
    for verified in archives {
        gates.push(VerifiedGateBinding {
            gate_id: verified.gate_id,
            evidence_kind: "GitHubActionsArchiveAndLiveExactAttempt".into(),
            evidence_object_sha256: vec![
                verified.archive_manifest_sha256,
                verified.run_json_sha256,
                verified.jobs_json_sha256,
                verified.workflow_file_sha256,
            ],
        });
    }
    gates.sort_by(|left, right| left.gate_id.cmp(&right.gate_id));

    let verified_evidence = VerifiedQualificationEvidence {
        source_commit: bundle.source_commit.clone(),
        model_semantics_version: bundle.model_semantics_version,
        qualification_bundle_sha256: bundle_sha256(&bundle)?,
        evidence_capsule_archive: capsule,
        gates,
    };

    let envelope = PromotionAuthorizationEnvelope {
        schema_version: PROMOTION_AUTHORIZATION_ENVELOPE_SCHEMA_VERSION,
        verifier_policy_version: VERIFIER_POLICY_VERSION.into(),
        verification_mode:
            "frozen-source+canonical-bundle+local-exact-bytes+capsule-bytes+live-github-exact-attempt+content-addressed-archive-v1"
                .into(),
        verified_evidence,
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
    let archived_verified = interoception_qualification::verify_actions_archive(archive_dir, repo_root)?;
    let manifest_bytes = fs::read(archive_dir.join("manifest.json"))?;
    let manifest: ActionsArchiveManifest =
        serde_json::from_slice(&manifest_bytes).context("parse Actions archive manifest")?;

    let archived_run: RunIdentity = serde_json::from_slice(&fs::read(archive_dir.join("run.json"))?)
        .context("parse archived run identity for live comparison")?;
    let archived_jobs: JobsPage = serde_json::from_slice(&fs::read(archive_dir.join("jobs.json"))?)
        .context("parse archived jobs identity for live comparison")?;
    if archived_jobs.total_count != archived_jobs.jobs.len() || archived_jobs.jobs.is_empty() {
        bail!("archived jobs object is incomplete or empty before live verification");
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
    if declared_total == 0 {
        bail!("live GitHub exact attempt contains zero jobs");
    }
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
    for job in &live_jobs {
        if job.status != "completed" || job.conclusion.as_deref() != Some("success") {
            bail!(
                "required v0.1 evidence attempt contains non-success job {:?}: status={}, conclusion={:?}",
                job.name,
                job.status,
                job.conclusion
            );
        }
    }

    let archived_by_id = jobs_by_id(&archived_jobs.jobs, "archived")?;
    let live_by_id = jobs_by_id(&live_jobs, "live")?;
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
    let live_content_bytes = gh_api(&[&workflow_endpoint])?;
    let live_content: ContentIdentity =
        serde_json::from_slice(&live_content_bytes).context("parse live workflow content identity")?;
    validate_sha1("live workflow Git blob SHA", &live_content.sha)?;
    let archived_workflow_blob = git_hash_object(&archive_dir.join("workflow.yml"))?;
    if archived_workflow_blob != live_content.sha {
        bail!(
            "live exact-SHA workflow Git blob differs from archived workflow source for gate {}",
            manifest.gate_id
        );
    }

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
        workflow_file_sha256: archived_verified.workflow_file_sha256,
        workflow_git_blob_sha1: live_content.sha,
        verification_transport: "GitHub REST API via gh authenticated HTTPS request".into(),
    })
}

fn jobs_by_id<'a>(
    jobs: &'a [JobIdentity],
    label: &str,
) -> Result<BTreeMap<u64, (&'a str, &'a str, Option<&'a str>)>> {
    let mut mapped = BTreeMap::new();
    for job in jobs {
        let previous = mapped.insert(
            job.id,
            (job.name.as_str(), job.status.as_str(), job.conclusion.as_deref()),
        );
        if previous.is_some() {
            bail!("{label} jobs contain duplicate job id {}", job.id);
        }
    }
    Ok(mapped)
}

fn verify_local_dir(dir: &Path, repo_root: &Path) -> Result<VerifiedLocalGate> {
    interoception_qualification::verify_local_gate(
        &interoception_qualification::local_manifest_path(dir),
        &interoception_qualification::local_transcript_path(dir),
        Some(repo_root),
    )
}

fn bind_local_to_bundle(bundle: &QualificationEvidenceBundle, verified: &VerifiedLocalGate) -> Result<()> {
    if verified.subject_commit != FROZEN_V01_SOURCE_COMMIT {
        bail!("verified local evidence is not bound to the frozen v0.1 source");
    }
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
        _ => bail!(
            "bundle local gate {} does not bind the independently verified local evidence",
            verified.gate_id
        ),
    }
}

fn bind_actions_to_bundle(
    bundle: &QualificationEvidenceBundle,
    verified: &VerifiedActionsGate,
) -> Result<()> {
    if verified.subject_commit != FROZEN_V01_SOURCE_COMMIT {
        bail!("verified Actions evidence is not bound to the frozen v0.1 source");
    }
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
        _ => bail!(
            "bundle Actions gate {} does not bind the independently verified exact attempt",
            verified.gate_id
        ),
    }
}

fn read_canonical_bundle(path: &Path) -> Result<QualificationEvidenceBundle> {
    let bytes = fs::read(path).with_context(|| format!("read qualification bundle {}", path.display()))?;
    let bundle: QualificationEvidenceBundle =
        serde_json::from_slice(&bytes).context("parse qualification evidence bundle")?;
    let canonical = serde_json::to_vec(&bundle).context("re-serialize qualification bundle canonically")?;
    if bytes != canonical {
        bail!(
            "qualification bundle bytes are not the canonical compact JSON representation required by the frozen v0.1 bundle contract"
        );
    }
    Ok(bundle)
}

fn bundle_sha256(bundle: &QualificationEvidenceBundle) -> Result<String> {
    bundle
        .sha256()
        .map_err(|errors| anyhow::anyhow!("qualification bundle digest failed: {}", errors.join("; ")))
}

fn git_hash_object(path: &Path) -> Result<String> {
    let output = Command::new("git")
        .arg("hash-object")
        .arg(path)
        .output()
        .with_context(|| format!("git hash-object {}", path.display()))?;
    if !output.status.success() {
        bail!(
            "git hash-object failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let value = String::from_utf8(output.stdout).context("git hash-object emitted non-UTF-8 output")?;
    let value = value.trim().to_string();
    validate_sha1("archived workflow Git blob SHA", &value)?;
    Ok(value)
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
