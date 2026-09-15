// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mediated pre-scoring freeze control for calibration evidence.
//!
//! This module is intentionally additive to the v1 frozen-calibration receipt.
//! A declared plan grants no freeze authority. The stronger v2 evidence object
//! is produced only by a controller path that observes the actual Git subject,
//! fsyncs a pre-scoring guard, invokes the scorer itself, replays the Git
//! subject afterward, binds the exact result bytes, and fsyncs a checkpoint.
//!
//! The theorem is boundary-ordered mediation, not continuous sandboxing. The
//! receipts therefore record `continuous_runtime_mediation_performed = false`.

use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;

pub const DECLARED_FREEZE_PLAN_V2_SCHEMA: &str = "psych-calibration-freeze-plan-v2";
pub const PRE_SCORING_GUARD_V2_SCHEMA: &str = "psych-calibration-pre-scoring-guard-v2";
pub const POST_SCORING_CHECKPOINT_V2_SCHEMA: &str =
    "psych-calibration-post-scoring-checkpoint-v2";

/// A declaration of the experiment identity to freeze.
///
/// This structure is not authority-bearing. In particular it contains no
/// caller-selectable `VerifiedFrozen`/`PreScoringEvidenceBound` flag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclaredCalibrationFreezePlanV2 {
    pub schema_version: String,
    pub benchmark: String,
    pub parameter_manifest_digest: String,
    /// Optional pre-scoring calculation-authority root, e.g. the qualified
    /// transitive authority binding introduced by the #3150 lineage.
    pub calculation_authority_digest: Option<String>,
    pub task_set_id: String,
    pub baseline_or_holdout_id: String,
    pub lineage_id: String,
    /// Identity of a separately captured environment/reproducibility capsule.
    /// This tranche binds the digest; it does not independently recapture or
    /// validate the capsule contents.
    pub environment_capsule_digest: String,
    pub result_label: String,
}

impl DeclaredCalibrationFreezePlanV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        benchmark: impl Into<String>,
        parameter_manifest_digest: impl Into<String>,
        calculation_authority_digest: Option<String>,
        task_set_id: impl Into<String>,
        baseline_or_holdout_id: impl Into<String>,
        lineage_id: impl Into<String>,
        environment_capsule_digest: impl Into<String>,
        result_label: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: DECLARED_FREEZE_PLAN_V2_SCHEMA.to_string(),
            benchmark: benchmark.into(),
            parameter_manifest_digest: parameter_manifest_digest.into(),
            calculation_authority_digest,
            task_set_id: task_set_id.into(),
            baseline_or_holdout_id: baseline_or_holdout_id.into(),
            lineage_id: lineage_id.into(),
            environment_capsule_digest: environment_capsule_digest.into(),
            result_label: result_label.into(),
        }
    }

    pub fn validate(&self) -> Result<(), MediationControlError> {
        if self.schema_version != DECLARED_FREEZE_PLAN_V2_SCHEMA {
            return Err(MediationControlError::UnsupportedPlanSchema);
        }
        for (name, value) in [
            ("benchmark", self.benchmark.as_str()),
            ("task_set_id", self.task_set_id.as_str()),
            ("baseline_or_holdout_id", self.baseline_or_holdout_id.as_str()),
            ("lineage_id", self.lineage_id.as_str()),
            ("result_label", self.result_label.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(MediationControlError::EmptyPlanField(name));
            }
        }
        validate_digest("parameter_manifest_digest", &self.parameter_manifest_digest)?;
        validate_digest(
            "environment_capsule_digest",
            &self.environment_capsule_digest,
        )?;
        if let Some(digest) = &self.calculation_authority_digest {
            validate_digest("calculation_authority_digest", digest)?;
        }
        Ok(())
    }

    pub fn digest_hex(&self) -> Result<String, MediationControlError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.psych.calibration-freeze-plan.v2\0");
        hash_field(&mut hasher, self.schema_version.as_bytes());
        hash_field(&mut hasher, self.benchmark.as_bytes());
        hash_field(&mut hasher, self.parameter_manifest_digest.as_bytes());
        match &self.calculation_authority_digest {
            Some(value) => {
                hasher.update(&[1]);
                hash_field(&mut hasher, value.as_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hash_field(&mut hasher, self.task_set_id.as_bytes());
        hash_field(&mut hasher, self.baseline_or_holdout_id.as_bytes());
        hash_field(&mut hasher, self.lineage_id.as_bytes());
        hash_field(&mut hasher, self.environment_capsule_digest.as_bytes());
        hash_field(&mut hasher, self.result_label.as_bytes());
        Ok(hasher.finalize().to_hex().to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreScoringGuardPayloadV2 {
    pub schema_version: String,
    pub plan_digest: String,
    /// Exact Git HEAD observed by the controller immediately before the guard
    /// is materialized. The plan cannot choose this value.
    pub observed_code_subject: String,
    pub continuous_runtime_mediation_performed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreScoringGuardReceiptV2 {
    pub payload: PreScoringGuardPayloadV2,
    pub receipt_digest: String,
}

impl PreScoringGuardReceiptV2 {
    fn new(
        plan_digest: String,
        observed_code_subject: String,
    ) -> Result<Self, MediationControlError> {
        let payload = PreScoringGuardPayloadV2 {
            schema_version: PRE_SCORING_GUARD_V2_SCHEMA.to_string(),
            plan_digest,
            observed_code_subject,
            continuous_runtime_mediation_performed: false,
        };
        let receipt_digest = guard_payload_digest_hex(&payload)?;
        Ok(Self {
            payload,
            receipt_digest,
        })
    }

    pub fn validate(
        &self,
        plan: &DeclaredCalibrationFreezePlanV2,
    ) -> Result<(), MediationControlError> {
        if self.payload.schema_version != PRE_SCORING_GUARD_V2_SCHEMA {
            return Err(MediationControlError::UnsupportedGuardSchema);
        }
        if self.payload.continuous_runtime_mediation_performed {
            return Err(MediationControlError::UnsupportedContinuousMediationClaim);
        }
        validate_full_git_sha(&self.payload.observed_code_subject)?;
        if self.payload.plan_digest != plan.digest_hex()? {
            return Err(MediationControlError::PlanDigestMismatch);
        }
        if self.receipt_digest != guard_payload_digest_hex(&self.payload)? {
            return Err(MediationControlError::GuardDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostScoringCheckpointPayloadV2 {
    pub schema_version: String,
    pub plan_digest: String,
    pub guard_receipt_digest: String,
    pub observed_code_subject_before: String,
    pub observed_code_subject_after: String,
    /// BLAKE3 of the exact result artifact bytes returned by the mediated scorer.
    pub result_artifact_digest: String,
    pub result_artifact_len: u64,
    pub continuous_runtime_mediation_performed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostScoringCheckpointReceiptV2 {
    pub payload: PostScoringCheckpointPayloadV2,
    pub receipt_digest: String,
}

impl PostScoringCheckpointReceiptV2 {
    fn new(
        plan_digest: String,
        guard_receipt_digest: String,
        code_subject_before: String,
        code_subject_after: String,
        result_artifact_digest: String,
        result_artifact_len: u64,
    ) -> Result<Self, MediationControlError> {
        let payload = PostScoringCheckpointPayloadV2 {
            schema_version: POST_SCORING_CHECKPOINT_V2_SCHEMA.to_string(),
            plan_digest,
            guard_receipt_digest,
            observed_code_subject_before: code_subject_before,
            observed_code_subject_after: code_subject_after,
            result_artifact_digest,
            result_artifact_len,
            continuous_runtime_mediation_performed: false,
        };
        let receipt_digest = checkpoint_payload_digest_hex(&payload)?;
        Ok(Self {
            payload,
            receipt_digest,
        })
    }

    pub fn validate(
        &self,
        plan: &DeclaredCalibrationFreezePlanV2,
        guard: &PreScoringGuardReceiptV2,
    ) -> Result<(), MediationControlError> {
        guard.validate(plan)?;
        if self.payload.schema_version != POST_SCORING_CHECKPOINT_V2_SCHEMA {
            return Err(MediationControlError::UnsupportedCheckpointSchema);
        }
        if self.payload.continuous_runtime_mediation_performed {
            return Err(MediationControlError::UnsupportedContinuousMediationClaim);
        }
        if self.payload.plan_digest != plan.digest_hex()? {
            return Err(MediationControlError::PlanDigestMismatch);
        }
        if self.payload.guard_receipt_digest != guard.receipt_digest {
            return Err(MediationControlError::GuardDigestMismatch);
        }
        validate_full_git_sha(&self.payload.observed_code_subject_before)?;
        validate_full_git_sha(&self.payload.observed_code_subject_after)?;
        if self.payload.observed_code_subject_before != guard.payload.observed_code_subject
            || self.payload.observed_code_subject_after != guard.payload.observed_code_subject
        {
            return Err(MediationControlError::CodeSubjectMismatch);
        }
        validate_digest("result_artifact_digest", &self.payload.result_artifact_digest)?;
        if self.payload.result_artifact_len == 0 {
            return Err(MediationControlError::EmptyResultArtifact);
        }
        if self.receipt_digest != checkpoint_payload_digest_hex(&self.payload)? {
            return Err(MediationControlError::CheckpointDigestMismatch);
        }
        Ok(())
    }
}

/// In-process capability proving that the current process obtained these
/// receipts through the mediated controller path below.
///
/// Deliberately not `Deserialize`: loading the serialized receipts later does
/// not recreate this authority capability. A future external verifier may
/// introduce an independently qualified rehydration theorem.
#[derive(Debug, Clone)]
pub struct MediatedFreezeEvidenceV2 {
    guard: PreScoringGuardReceiptV2,
    checkpoint: PostScoringCheckpointReceiptV2,
}

impl MediatedFreezeEvidenceV2 {
    pub fn guard_receipt(&self) -> &PreScoringGuardReceiptV2 {
        &self.guard
    }

    pub fn checkpoint_receipt(&self) -> &PostScoringCheckpointReceiptV2 {
        &self.checkpoint
    }

    pub fn validate(
        &self,
        plan: &DeclaredCalibrationFreezePlanV2,
    ) -> Result<(), MediationControlError> {
        self.checkpoint.validate(plan, &self.guard)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediationControlError {
    UnsupportedPlanSchema,
    UnsupportedGuardSchema,
    UnsupportedCheckpointSchema,
    UnsupportedContinuousMediationClaim,
    EmptyPlanField(&'static str),
    InvalidDigest(&'static str),
    InvalidCodeSubject,
    GitCommandFailed(String),
    DirtySubjectBeforeScoring,
    DirtySubjectAfterScoring,
    GuardPathAlreadyExists,
    CheckpointPathAlreadyExists,
    Io(String),
    Serialization(String),
    GuardPersistenceMismatch,
    GuardMutatedDuringScoring,
    CodeSubjectChanged { before: String, after: String },
    PlanDigestMismatch,
    GuardDigestMismatch,
    CheckpointDigestMismatch,
    CodeSubjectMismatch,
    EmptyResultArtifact,
}

#[derive(Debug)]
pub enum MediatedFreezeError<E> {
    Control(MediationControlError),
    Scorer(E),
}

/// Mediate one scoring operation under a pre-scoring guard and post-scoring
/// checkpoint.
///
/// Ordering is controller-owned:
///
/// `observe clean Git subject -> fsync guard -> scorer -> replay Git subject -> fsync checkpoint`
///
/// The scorer returns the exact artifact bytes that should be hash-bound by the
/// checkpoint. If the scorer fails, mutates the guard, dirties/changes the Git
/// subject, or returns an empty artifact, no checkpoint is produced.
pub fn mediate_frozen_scoring<E, F>(
    repository_root: &Path,
    guard_path: &Path,
    checkpoint_path: &Path,
    plan: &DeclaredCalibrationFreezePlanV2,
    scorer: F,
) -> Result<(Vec<u8>, MediatedFreezeEvidenceV2), MediatedFreezeError<E>>
where
    F: FnOnce(&PreScoringGuardReceiptV2) -> Result<Vec<u8>, E>,
{
    plan.validate().map_err(MediatedFreezeError::Control)?;
    if guard_path.exists() {
        return Err(MediatedFreezeError::Control(
            MediationControlError::GuardPathAlreadyExists,
        ));
    }
    if checkpoint_path.exists() {
        return Err(MediatedFreezeError::Control(
            MediationControlError::CheckpointPathAlreadyExists,
        ));
    }

    let code_subject_before = observe_clean_git_subject(repository_root, true)
        .map_err(MediatedFreezeError::Control)?;
    let plan_digest = plan.digest_hex().map_err(MediatedFreezeError::Control)?;
    let guard = PreScoringGuardReceiptV2::new(plan_digest.clone(), code_subject_before.clone())
        .map_err(MediatedFreezeError::Control)?;
    guard
        .validate(plan)
        .map_err(MediatedFreezeError::Control)?;

    let guard_bytes = serialize_pretty(&guard).map_err(MediatedFreezeError::Control)?;
    write_new_and_sync(guard_path, &guard_bytes).map_err(MediatedFreezeError::Control)?;
    let persisted_guard = fs::read(guard_path)
        .map_err(|error| MediatedFreezeError::Control(io_error(error)))?;
    if persisted_guard != guard_bytes {
        return Err(MediatedFreezeError::Control(
            MediationControlError::GuardPersistenceMismatch,
        ));
    }

    let result_bytes = scorer(&guard).map_err(MediatedFreezeError::Scorer)?;
    if result_bytes.is_empty() {
        return Err(MediatedFreezeError::Control(
            MediationControlError::EmptyResultArtifact,
        ));
    }

    let guard_after = fs::read(guard_path)
        .map_err(|error| MediatedFreezeError::Control(io_error(error)))?;
    if guard_after != guard_bytes {
        return Err(MediatedFreezeError::Control(
            MediationControlError::GuardMutatedDuringScoring,
        ));
    }

    let code_subject_after = observe_clean_git_subject(repository_root, false)
        .map_err(MediatedFreezeError::Control)?;
    if code_subject_after != code_subject_before {
        return Err(MediatedFreezeError::Control(
            MediationControlError::CodeSubjectChanged {
                before: code_subject_before,
                after: code_subject_after,
            },
        ));
    }

    let result_artifact_digest = result_artifact_digest_hex(&result_bytes);
    let checkpoint = PostScoringCheckpointReceiptV2::new(
        plan_digest,
        guard.receipt_digest.clone(),
        code_subject_before,
        code_subject_after,
        result_artifact_digest,
        result_bytes.len() as u64,
    )
    .map_err(MediatedFreezeError::Control)?;
    checkpoint
        .validate(plan, &guard)
        .map_err(MediatedFreezeError::Control)?;
    let checkpoint_bytes =
        serialize_pretty(&checkpoint).map_err(MediatedFreezeError::Control)?;
    write_new_and_sync(checkpoint_path, &checkpoint_bytes)
        .map_err(MediatedFreezeError::Control)?;

    Ok((
        result_bytes,
        MediatedFreezeEvidenceV2 { guard, checkpoint },
    ))
}

pub fn result_artifact_digest_hex(bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.psych.calibration-result-artifact.v2\0");
    hash_field(&mut hasher, bytes);
    hasher.finalize().to_hex().to_string()
}

fn observe_clean_git_subject(
    repository_root: &Path,
    before_scoring: bool,
) -> Result<String, MediationControlError> {
    let head = run_git(repository_root, &["rev-parse", "HEAD"])?;
    validate_full_git_sha(&head)?;
    let status = run_git(
        repository_root,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    if !status.is_empty() {
        return Err(if before_scoring {
            MediationControlError::DirtySubjectBeforeScoring
        } else {
            MediationControlError::DirtySubjectAfterScoring
        });
    }
    Ok(head)
}

fn run_git(repository_root: &Path, args: &[&str]) -> Result<String, MediationControlError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository_root)
        .args(args)
        .output()
        .map_err(|error| MediationControlError::GitCommandFailed(error.to_string()))?;
    if !output.status.success() {
        return Err(MediationControlError::GitCommandFailed(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn validate_full_git_sha(value: &str) -> Result<(), MediationControlError> {
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(MediationControlError::InvalidCodeSubject);
    }
    Ok(())
}

fn validate_digest(name: &'static str, value: &str) -> Result<(), MediationControlError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(MediationControlError::InvalidDigest(name));
    }
    Ok(())
}

fn write_new_and_sync(path: &Path, bytes: &[u8]) -> Result<(), MediationControlError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(io_error)?;
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(io_error)?;
    file.write_all(bytes).map_err(io_error)?;
    file.sync_all().map_err(io_error)?;
    Ok(())
}

fn serialize_pretty<T: Serialize>(value: &T) -> Result<Vec<u8>, MediationControlError> {
    serde_json::to_vec_pretty(value)
        .map(|mut bytes| {
            bytes.push(b'\n');
            bytes
        })
        .map_err(|error| MediationControlError::Serialization(error.to_string()))
}

fn guard_payload_digest_hex(
    payload: &PreScoringGuardPayloadV2,
) -> Result<String, MediationControlError> {
    let bytes = serde_json::to_vec(payload)
        .map_err(|error| MediationControlError::Serialization(error.to_string()))?;
    Ok(domain_hash(
        b"symthaea.psych.calibration-pre-scoring-guard.v2\0",
        &bytes,
    ))
}

fn checkpoint_payload_digest_hex(
    payload: &PostScoringCheckpointPayloadV2,
) -> Result<String, MediationControlError> {
    let bytes = serde_json::to_vec(payload)
        .map_err(|error| MediationControlError::Serialization(error.to_string()))?;
    Ok(domain_hash(
        b"symthaea.psych.calibration-post-scoring-checkpoint.v2\0",
        &bytes,
    ))
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hash_field(&mut hasher, bytes);
    hasher.finalize().to_hex().to_string()
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn io_error(error: std::io::Error) -> MediationControlError {
    MediationControlError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_CASE: AtomicU64 = AtomicU64::new(0);

    fn case_root(label: &str) -> std::path::PathBuf {
        let serial = NEXT_CASE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "symthaea-psych-freeze-v2-{label}-{}-{serial}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn git(repo: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn init_repo(root: &Path) -> std::path::PathBuf {
        let repo = root.join("repo");
        fs::create_dir_all(&repo).unwrap();
        git(&repo, &["init", "-q"]);
        git(
            &repo,
            &["config", "user.email", "qualification@example.invalid"],
        );
        git(&repo, &["config", "user.name", "Qualification Fixture"]);
        fs::write(repo.join("subject.txt"), b"frozen subject\n").unwrap();
        git(&repo, &["add", "subject.txt"]);
        git(&repo, &["commit", "-qm", "fixture subject"]);
        repo
    }

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn plan() -> DeclaredCalibrationFreezePlanV2 {
        DeclaredCalibrationFreezePlanV2::new(
            "WorM::N-back",
            digest('a'),
            Some(digest('b')),
            "nback-holdout-v2",
            "human-baseline-v2",
            "psych-freeze-v2-fixture",
            digest('c'),
            "nback-score-receipt",
        )
    }

    #[test]
    fn guard_is_fsynced_before_scorer_and_checkpoint_binds_result() {
        let root = case_root("success");
        let repo = init_repo(&root);
        let guard_path = root.join("control/guard.json");
        let checkpoint_path = root.join("control/checkpoint.json");
        let plan = plan();
        let expected_result = br#"{"accuracy":0.81}"#.to_vec();

        let (result, evidence) = mediate_frozen_scoring(
            &repo,
            &guard_path,
            &checkpoint_path,
            &plan,
            |guard| {
                assert!(guard_path.exists());
                let bytes = fs::read(&guard_path).unwrap();
                let on_disk: PreScoringGuardReceiptV2 = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(&on_disk, guard);
                Ok::<Vec<u8>, &'static str>(expected_result.clone())
            },
        )
        .unwrap();

        assert_eq!(result, expected_result);
        assert!(checkpoint_path.exists());
        assert!(evidence.validate(&plan).is_ok());
        assert_eq!(
            evidence.checkpoint_receipt().payload.result_artifact_digest,
            result_artifact_digest_hex(&result)
        );
        assert!(
            !evidence
                .checkpoint_receipt()
                .payload
                .continuous_runtime_mediation_performed
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn scorer_failure_produces_no_checkpoint() {
        let root = case_root("scorer-failure");
        let repo = init_repo(&root);
        let guard_path = root.join("guard.json");
        let checkpoint_path = root.join("checkpoint.json");
        let result = mediate_frozen_scoring(
            &repo,
            &guard_path,
            &checkpoint_path,
            &plan(),
            |_| Err::<Vec<u8>, _>("scorer failed"),
        );
        assert!(matches!(
            result,
            Err(MediatedFreezeError::Scorer("scorer failed"))
        ));
        assert!(guard_path.exists());
        assert!(!checkpoint_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn dirty_subject_is_rejected_before_scorer_runs() {
        let root = case_root("dirty-before");
        let repo = init_repo(&root);
        fs::write(repo.join("subject.txt"), b"changed before scoring\n").unwrap();
        let guard_path = root.join("guard.json");
        let checkpoint_path = root.join("checkpoint.json");
        let mut scorer_ran = false;
        let result = mediate_frozen_scoring(
            &repo,
            &guard_path,
            &checkpoint_path,
            &plan(),
            |_| {
                scorer_ran = true;
                Ok::<_, &'static str>(b"score".to_vec())
            },
        );
        assert!(matches!(
            result,
            Err(MediatedFreezeError::Control(
                MediationControlError::DirtySubjectBeforeScoring
            ))
        ));
        assert!(!scorer_ran);
        assert!(!guard_path.exists());
        assert!(!checkpoint_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn guard_mutation_during_scoring_fails_without_checkpoint() {
        let root = case_root("guard-mutation");
        let repo = init_repo(&root);
        let guard_path = root.join("guard.json");
        let checkpoint_path = root.join("checkpoint.json");
        let result = mediate_frozen_scoring(
            &repo,
            &guard_path,
            &checkpoint_path,
            &plan(),
            |_| {
                fs::write(&guard_path, b"tampered\n").unwrap();
                Ok::<_, &'static str>(b"score".to_vec())
            },
        );
        assert!(matches!(
            result,
            Err(MediatedFreezeError::Control(
                MediationControlError::GuardMutatedDuringScoring
            ))
        ));
        assert!(!checkpoint_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn tracked_subject_mutation_during_scoring_fails_without_checkpoint() {
        let root = case_root("dirty-after");
        let repo = init_repo(&root);
        let guard_path = root.join("guard.json");
        let checkpoint_path = root.join("checkpoint.json");
        let result = mediate_frozen_scoring(
            &repo,
            &guard_path,
            &checkpoint_path,
            &plan(),
            |_| {
                fs::write(repo.join("subject.txt"), b"changed during scoring\n").unwrap();
                Ok::<_, &'static str>(b"score".to_vec())
            },
        );
        assert!(matches!(
            result,
            Err(MediatedFreezeError::Control(
                MediationControlError::DirtySubjectAfterScoring
            ))
        ));
        assert!(!checkpoint_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn clean_commit_change_during_scoring_is_detected() {
        let root = case_root("subject-change");
        let repo = init_repo(&root);
        let before = git(&repo, &["rev-parse", "HEAD"]);
        let guard_path = root.join("guard.json");
        let checkpoint_path = root.join("checkpoint.json");
        let result = mediate_frozen_scoring(
            &repo,
            &guard_path,
            &checkpoint_path,
            &plan(),
            |_| {
                fs::write(repo.join("subject.txt"), b"new committed subject\n").unwrap();
                git(&repo, &["add", "subject.txt"]);
                git(&repo, &["commit", "-qm", "change subject during scoring"]);
                Ok::<_, &'static str>(b"score".to_vec())
            },
        );
        match result {
            Err(MediatedFreezeError::Control(MediationControlError::CodeSubjectChanged {
                before: got_before,
                after,
            })) => {
                assert_eq!(got_before, before);
                assert_ne!(after, before);
            }
            other => panic!("unexpected result: {other:?}"),
        }
        assert!(!checkpoint_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn empty_result_artifact_is_rejected() {
        let root = case_root("empty-result");
        let repo = init_repo(&root);
        let checkpoint_path = root.join("checkpoint.json");
        let result = mediate_frozen_scoring(
            &repo,
            &root.join("guard.json"),
            &checkpoint_path,
            &plan(),
            |_| Ok::<_, &'static str>(Vec::new()),
        );
        assert!(matches!(
            result,
            Err(MediatedFreezeError::Control(
                MediationControlError::EmptyResultArtifact
            ))
        ));
        assert!(!checkpoint_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn malformed_declared_digests_fail_before_authority_is_created() {
        let mut plan = plan();
        plan.parameter_manifest_digest = "not-a-digest".to_string();
        assert_eq!(
            plan.validate(),
            Err(MediationControlError::InvalidDigest(
                "parameter_manifest_digest"
            ))
        );
    }

    #[test]
    fn tampered_serialized_checkpoint_fails_consistency_validation() {
        let root = case_root("checkpoint-tamper");
        let repo = init_repo(&root);
        let plan = plan();
        let (_, evidence) = mediate_frozen_scoring(
            &repo,
            &root.join("guard.json"),
            &root.join("checkpoint.json"),
            &plan,
            |_| Ok::<_, &'static str>(b"score".to_vec()),
        )
        .unwrap();
        let mut checkpoint = evidence.checkpoint_receipt().clone();
        checkpoint.payload.result_artifact_len += 1;
        assert_eq!(
            checkpoint.validate(&plan, evidence.guard_receipt()),
            Err(MediationControlError::CheckpointDigestMismatch)
        );
        let _ = fs::remove_dir_all(root);
    }
}
