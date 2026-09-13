// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure qualification receipts for EUREKA exact execution subjects.
//!
//! This module deliberately contains no GitHub/API access. A provider-specific
//! materializer may normalize workflow/run/job observations into these types;
//! this kernel validates those immutable observations against a frozen profile.
//!
//! Governing non-equivalences:
//!
//! ```text
//! provider head metadata != checked-out execution subject
//! source-sanity green     != full exact-head qualification
//! merge-ref green         != exact-head green
//! queued/cancelled        != pass
//! diagnostic fingerprint  != scientific identity
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_evidence_plane::{EvidenceCounters, Expectation, RunEvidence, RunId};

pub(super) const QUALIFICATION_RECEIPT_REVISION: &str =
    "EUREKA.002W.QUALIFICATION_RECEIPT.v0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(super) enum QualificationLane {
    SourceSanity,
    FullExactHead,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(super) enum QualificationSubject {
    ExactHead {
        head_sha: String,
    },
    Integration {
        head_sha: String,
        base_sha: String,
        merge_sha: String,
    },
}

impl QualificationSubject {
    fn expected_checkout_sha(&self) -> &str {
        match self {
            Self::ExactHead { head_sha } => head_sha,
            Self::Integration { merge_sha, .. } => merge_sha,
        }
    }

    fn provider_head_sha(&self) -> &str {
        match self {
            Self::ExactHead { head_sha } | Self::Integration { head_sha, .. } => head_sha,
        }
    }

    fn provider_base_sha(&self) -> Option<&str> {
        match self {
            Self::ExactHead { .. } => None,
            Self::Integration { base_sha, .. } => Some(base_sha),
        }
    }

    fn validate(&self) -> bool {
        match self {
            Self::ExactHead { head_sha } => is_git_sha(head_sha),
            Self::Integration {
                head_sha,
                base_sha,
                merge_sha,
            } => is_git_sha(head_sha) && is_git_sha(base_sha) && is_git_sha(merge_sha),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(super) enum JobDisposition {
    Queued,
    InProgress,
    Success,
    Failure,
    Cancelled,
    Skipped,
    Neutral,
    TimedOut,
    ActionRequired,
    StartupFailure,
    Stale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(super) enum FailureAttribution {
    Subject,
    Qualifier,
    Infrastructure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ObservedQualificationJob {
    pub job_id: u64,
    pub name: String,
    pub disposition: JobDisposition,
    pub failure_attribution: Option<FailureAttribution>,
    /// Content identity for the log/step evidence supporting a failure
    /// attribution. Descriptive provider URLs are intentionally insufficient.
    pub failure_evidence_identity: Option<String>,
}

impl ObservedQualificationJob {
    fn validate(&self) -> bool {
        if self.job_id == 0 || self.name.trim().is_empty() {
            return false;
        }

        match self.disposition {
            JobDisposition::Failure
            | JobDisposition::TimedOut
            | JobDisposition::StartupFailure => match self.failure_attribution {
                Some(_) => self
                    .failure_evidence_identity
                    .as_deref()
                    .is_some_and(valid_identity),
                None => self.failure_evidence_identity.is_none(),
            },
            _ => self.failure_attribution.is_none() && self.failure_evidence_identity.is_none(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct QualificationProfile {
    pub revision: String,
    pub lane: QualificationLane,
    pub workflow_path: String,
    /// Git blob SHA or a canonical workflow-content digest.
    pub workflow_definition_identity: String,
    pub allowed_events: BTreeSet<String>,
    pub required_jobs: BTreeSet<String>,
    /// Required jobs that may be skipped under a predeclared workflow
    /// condition. This is exact-name scoped; a wildcard is never accepted.
    pub allowed_skipped_jobs: BTreeSet<String>,
    pub replay_identity: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum QualificationBuildError {
    EmptyRevision,
    EmptyWorkflowPath,
    InvalidWorkflowIdentity,
    EmptyAllowedEvents,
    EmptyRequiredJobs,
    AllowedSkipNotRequired(String),
    EmptyJobName,
    InvalidSubject,
    InvalidObservation,
    SerializationFailed,
}

impl QualificationProfile {
    pub fn try_new(
        revision: impl Into<String>,
        lane: QualificationLane,
        workflow_path: impl Into<String>,
        workflow_definition_identity: impl Into<String>,
        allowed_events: impl IntoIterator<Item = String>,
        required_jobs: impl IntoIterator<Item = String>,
        allowed_skipped_jobs: impl IntoIterator<Item = String>,
    ) -> Result<Self, QualificationBuildError> {
        let revision = revision.into();
        let workflow_path = workflow_path.into();
        let workflow_definition_identity = workflow_definition_identity.into();
        let allowed_events: BTreeSet<_> = allowed_events.into_iter().collect();
        let required_jobs: BTreeSet<_> = required_jobs.into_iter().collect();
        let allowed_skipped_jobs: BTreeSet<_> = allowed_skipped_jobs.into_iter().collect();

        if revision.trim().is_empty() {
            return Err(QualificationBuildError::EmptyRevision);
        }
        if workflow_path.trim().is_empty() {
            return Err(QualificationBuildError::EmptyWorkflowPath);
        }
        if !valid_identity(&workflow_definition_identity) {
            return Err(QualificationBuildError::InvalidWorkflowIdentity);
        }
        if allowed_events.is_empty() {
            return Err(QualificationBuildError::EmptyAllowedEvents);
        }
        if required_jobs.is_empty() {
            return Err(QualificationBuildError::EmptyRequiredJobs);
        }
        if required_jobs.iter().any(|name| name.trim().is_empty())
            || allowed_skipped_jobs
                .iter()
                .any(|name| name.trim().is_empty())
        {
            return Err(QualificationBuildError::EmptyJobName);
        }
        if let Some(name) = allowed_skipped_jobs
            .iter()
            .find(|name| !required_jobs.contains(*name))
        {
            return Err(QualificationBuildError::AllowedSkipNotRequired(
                name.clone(),
            ));
        }

        let mut profile = Self {
            revision,
            lane,
            workflow_path,
            workflow_definition_identity,
            allowed_events,
            required_jobs,
            allowed_skipped_jobs,
            replay_identity: String::new(),
        };
        profile.replay_identity = digest_serializable(&ProfileCommitment::from(&profile))?;
        Ok(profile)
    }

    fn replay_valid(&self) -> bool {
        digest_serializable(&ProfileCommitment::from(self))
            .is_ok_and(|digest| digest == self.replay_identity)
    }
}

#[derive(Serialize)]
struct ProfileCommitment<'a> {
    revision: &'a str,
    lane: QualificationLane,
    workflow_path: &'a str,
    workflow_definition_identity: &'a str,
    allowed_events: &'a BTreeSet<String>,
    required_jobs: &'a BTreeSet<String>,
    allowed_skipped_jobs: &'a BTreeSet<String>,
}

impl<'a> From<&'a QualificationProfile> for ProfileCommitment<'a> {
    fn from(profile: &'a QualificationProfile) -> Self {
        Self {
            revision: &profile.revision,
            lane: profile.lane,
            workflow_path: &profile.workflow_path,
            workflow_definition_identity: &profile.workflow_definition_identity,
            allowed_events: &profile.allowed_events,
            required_jobs: &profile.required_jobs,
            allowed_skipped_jobs: &profile.allowed_skipped_jobs,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct QualificationRunObservation {
    pub repository: String,
    pub workflow_run_id: u64,
    pub workflow_id: u64,
    pub workflow_path: String,
    pub workflow_definition_identity: String,
    pub run_attempt: u32,
    pub event: String,
    /// Provider metadata. This is not accepted as proof of checkout subject.
    pub provider_head_sha: String,
    pub provider_base_sha: Option<String>,
    /// Must come from an explicit in-run `git rev-parse HEAD` assertion/receipt.
    pub checked_out_subject_sha: String,
    pub materializer_revision: String,
    pub jobs: Vec<ObservedQualificationJob>,
    pub replay_identity: String,
}

impl QualificationRunObservation {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        repository: impl Into<String>,
        workflow_run_id: u64,
        workflow_id: u64,
        workflow_path: impl Into<String>,
        workflow_definition_identity: impl Into<String>,
        run_attempt: u32,
        event: impl Into<String>,
        provider_head_sha: impl Into<String>,
        provider_base_sha: Option<String>,
        checked_out_subject_sha: impl Into<String>,
        materializer_revision: impl Into<String>,
        mut jobs: Vec<ObservedQualificationJob>,
    ) -> Result<Self, QualificationBuildError> {
        let repository = repository.into();
        let workflow_path = workflow_path.into();
        let workflow_definition_identity = workflow_definition_identity.into();
        let event = event.into();
        let provider_head_sha = provider_head_sha.into();
        let checked_out_subject_sha = checked_out_subject_sha.into();
        let materializer_revision = materializer_revision.into();

        jobs.sort_by(|left, right| {
            left.name
                .cmp(&right.name)
                .then_with(|| left.job_id.cmp(&right.job_id))
        });

        let unique_job_ids: BTreeSet<_> = jobs.iter().map(|job| job.job_id).collect();
        let unique_job_names: BTreeSet<_> = jobs.iter().map(|job| job.name.as_str()).collect();

        if repository.trim().is_empty()
            || workflow_run_id == 0
            || workflow_id == 0
            || workflow_path.trim().is_empty()
            || !valid_identity(&workflow_definition_identity)
            || run_attempt == 0
            || event.trim().is_empty()
            || !is_git_sha(&provider_head_sha)
            || provider_base_sha
                .as_deref()
                .is_some_and(|sha| !is_git_sha(sha))
            || !is_git_sha(&checked_out_subject_sha)
            || materializer_revision.trim().is_empty()
            || jobs.is_empty()
            || jobs.iter().any(|job| !job.validate())
            || unique_job_ids.len() != jobs.len()
            || unique_job_names.len() != jobs.len()
        {
            return Err(QualificationBuildError::InvalidObservation);
        }

        let mut observation = Self {
            repository,
            workflow_run_id,
            workflow_id,
            workflow_path,
            workflow_definition_identity,
            run_attempt,
            event,
            provider_head_sha,
            provider_base_sha,
            checked_out_subject_sha,
            materializer_revision,
            jobs,
            replay_identity: String::new(),
        };
        observation.replay_identity = digest_serializable(&ObservationCommitment::from(&observation))?;
        Ok(observation)
    }

    fn replay_valid(&self) -> bool {
        digest_serializable(&ObservationCommitment::from(self))
            .is_ok_and(|digest| digest == self.replay_identity)
    }
}

#[derive(Serialize)]
struct ObservationCommitment<'a> {
    repository: &'a str,
    workflow_run_id: u64,
    workflow_id: u64,
    workflow_path: &'a str,
    workflow_definition_identity: &'a str,
    run_attempt: u32,
    event: &'a str,
    provider_head_sha: &'a str,
    provider_base_sha: &'a Option<String>,
    checked_out_subject_sha: &'a str,
    materializer_revision: &'a str,
    jobs: &'a [ObservedQualificationJob],
}

impl<'a> From<&'a QualificationRunObservation> for ObservationCommitment<'a> {
    fn from(observation: &'a QualificationRunObservation) -> Self {
        Self {
            repository: &observation.repository,
            workflow_run_id: observation.workflow_run_id,
            workflow_id: observation.workflow_id,
            workflow_path: &observation.workflow_path,
            workflow_definition_identity: &observation.workflow_definition_identity,
            run_attempt: observation.run_attempt,
            event: &observation.event,
            provider_head_sha: &observation.provider_head_sha,
            provider_base_sha: &observation.provider_base_sha,
            checked_out_subject_sha: &observation.checked_out_subject_sha,
            materializer_revision: &observation.materializer_revision,
            jobs: &observation.jobs,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum QualificationClassification {
    Pass,
    NotExecuted,
    WrongLane,
    StaleSubject,
    IncompleteJobSet,
    FailSubject,
    FailQualifier,
    InfrastructureIndeterminate,
    UnclassifiedFailure,
    MultipleFailureClasses,
    InvalidObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(super) enum QualificationIssue {
    InvalidProfileReplay,
    InvalidObservationReplay,
    WrongLane,
    WrongWorkflowPath,
    WrongWorkflowDefinition,
    DisallowedEvent,
    ProviderHeadMismatch,
    ProviderBaseMismatch,
    CheckedOutSubjectMismatch,
    MissingRequiredJob(String),
    RequiredJobNotSuccessful {
        name: String,
        disposition: JobDisposition,
    },
}

#[derive(Debug, Clone)]
pub(super) struct QualificationEvaluation {
    pub classification: QualificationClassification,
    pub issues: Vec<QualificationIssue>,
    /// Canonically sorted integer defect counters. These, not
    /// `RunEvidence::config_hash`, are safe to include in replay identities.
    pub defect_counters: BTreeMap<String, u64>,
    /// Generic declared-vs-measured mechanical integrity report. This is
    /// intentionally diagnostic/enforcement evidence, not the receipt identity.
    pub mechanical_evidence: RunEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct QualifiedHeadReceipt {
    pub receipt_revision: String,
    pub profile_replay_identity: String,
    pub lane: QualificationLane,
    pub subject: QualificationSubject,
    pub workflow_run_id: u64,
    pub workflow_id: u64,
    pub run_attempt: u32,
    pub workflow_definition_identity: String,
    pub materializer_revision: String,
    pub checked_out_subject_sha: String,
    pub observation_replay_identity: String,
    pub normalized_job_set_digest: String,
    pub defect_counters: BTreeMap<String, u64>,
    pub classification: QualificationClassification,
    pub replay_identity: String,
}

impl QualifiedHeadReceipt {
    fn replay_valid(&self) -> bool {
        digest_serializable(&ReceiptCommitment::from(self))
            .is_ok_and(|digest| digest == self.replay_identity)
    }
}

#[derive(Serialize)]
struct ReceiptCommitment<'a> {
    receipt_revision: &'a str,
    profile_replay_identity: &'a str,
    lane: QualificationLane,
    subject: &'a QualificationSubject,
    workflow_run_id: u64,
    workflow_id: u64,
    run_attempt: u32,
    workflow_definition_identity: &'a str,
    materializer_revision: &'a str,
    checked_out_subject_sha: &'a str,
    observation_replay_identity: &'a str,
    normalized_job_set_digest: &'a str,
    defect_counters: &'a BTreeMap<String, u64>,
    classification: QualificationClassification,
}

impl<'a> From<&'a QualifiedHeadReceipt> for ReceiptCommitment<'a> {
    fn from(receipt: &'a QualifiedHeadReceipt) -> Self {
        Self {
            receipt_revision: &receipt.receipt_revision,
            profile_replay_identity: &receipt.profile_replay_identity,
            lane: receipt.lane,
            subject: &receipt.subject,
            workflow_run_id: receipt.workflow_run_id,
            workflow_id: receipt.workflow_id,
            run_attempt: receipt.run_attempt,
            workflow_definition_identity: &receipt.workflow_definition_identity,
            materializer_revision: &receipt.materializer_revision,
            checked_out_subject_sha: &receipt.checked_out_subject_sha,
            observation_replay_identity: &receipt.observation_replay_identity,
            normalized_job_set_digest: &receipt.normalized_job_set_digest,
            defect_counters: &receipt.defect_counters,
            classification: receipt.classification,
        }
    }
}

pub(super) fn evaluate_qualification(
    profile: &QualificationProfile,
    subject: &QualificationSubject,
    observation: &QualificationRunObservation,
) -> QualificationEvaluation {
    let mut issues = Vec::new();
    let mut defects = zero_defect_counters();

    if !profile.replay_valid() {
        issues.push(QualificationIssue::InvalidProfileReplay);
        increment(&mut defects, "invalid_profile_replay");
    }
    if !observation.replay_valid() {
        issues.push(QualificationIssue::InvalidObservationReplay);
        increment(&mut defects, "invalid_observation_replay");
    }
    if profile.lane != QualificationLane::FullExactHead {
        issues.push(QualificationIssue::WrongLane);
        increment(&mut defects, "wrong_lane");
    }
    if observation.workflow_path != profile.workflow_path {
        issues.push(QualificationIssue::WrongWorkflowPath);
        increment(&mut defects, "wrong_workflow_path");
    }
    if observation.workflow_definition_identity != profile.workflow_definition_identity {
        issues.push(QualificationIssue::WrongWorkflowDefinition);
        increment(&mut defects, "wrong_workflow_definition");
    }
    if !profile.allowed_events.contains(&observation.event) {
        issues.push(QualificationIssue::DisallowedEvent);
        increment(&mut defects, "disallowed_event");
    }
    if observation.provider_head_sha != subject.provider_head_sha() {
        issues.push(QualificationIssue::ProviderHeadMismatch);
        increment(&mut defects, "provider_head_mismatch");
    }
    if observation.provider_base_sha.as_deref() != subject.provider_base_sha() {
        issues.push(QualificationIssue::ProviderBaseMismatch);
        increment(&mut defects, "provider_base_mismatch");
    }
    if observation.checked_out_subject_sha != subject.expected_checkout_sha() {
        issues.push(QualificationIssue::CheckedOutSubjectMismatch);
        increment(&mut defects, "checked_out_subject_mismatch");
    }

    let jobs_by_name: BTreeMap<_, _> = observation
        .jobs
        .iter()
        .map(|job| (job.name.as_str(), job))
        .collect();

    let mut failure_classes = BTreeSet::new();
    let mut has_not_executed = false;
    let mut has_incomplete = false;

    for required_name in &profile.required_jobs {
        let Some(job) = jobs_by_name.get(required_name.as_str()) else {
            issues.push(QualificationIssue::MissingRequiredJob(required_name.clone()));
            increment(&mut defects, "missing_required_jobs");
            has_incomplete = true;
            continue;
        };

        let allowed_skip = profile.allowed_skipped_jobs.contains(required_name);
        let accepted = job.disposition == JobDisposition::Success
            || (allowed_skip && job.disposition == JobDisposition::Skipped);
        if accepted {
            continue;
        }

        issues.push(QualificationIssue::RequiredJobNotSuccessful {
            name: required_name.clone(),
            disposition: job.disposition,
        });
        increment(&mut defects, "required_non_success");

        match job.disposition {
            JobDisposition::Queued | JobDisposition::InProgress => {
                increment(&mut defects, "required_not_executed");
                has_not_executed = true;
            }
            JobDisposition::Skipped => {
                increment(&mut defects, "required_skipped");
                has_incomplete = true;
            }
            JobDisposition::Cancelled => {
                increment(&mut defects, "required_cancelled");
                failure_classes.insert(FailureAttribution::Infrastructure);
            }
            JobDisposition::Failure
            | JobDisposition::TimedOut
            | JobDisposition::StartupFailure => {
                if let Some(class) = job.failure_attribution {
                    failure_classes.insert(class);
                } else {
                    increment(&mut defects, "unclassified_failure");
                }
            }
            JobDisposition::Neutral
            | JobDisposition::ActionRequired
            | JobDisposition::Stale => {
                increment(&mut defects, "required_incomplete_terminal_state");
                has_incomplete = true;
            }
            JobDisposition::Success => unreachable!("success handled above"),
        }
    }

    if !subject.validate() {
        increment(&mut defects, "invalid_subject");
    }

    let classification = if !profile.replay_valid()
        || !observation.replay_valid()
        || !subject.validate()
        || issues.iter().any(|issue| {
            matches!(
                issue,
                QualificationIssue::WrongWorkflowPath
                    | QualificationIssue::WrongWorkflowDefinition
                    | QualificationIssue::DisallowedEvent
            )
        })
    {
        QualificationClassification::InvalidObservation
    } else if profile.lane != QualificationLane::FullExactHead {
        QualificationClassification::WrongLane
    } else if issues.iter().any(|issue| {
        matches!(
            issue,
            QualificationIssue::ProviderHeadMismatch
                | QualificationIssue::ProviderBaseMismatch
                | QualificationIssue::CheckedOutSubjectMismatch
        )
    }) {
        QualificationClassification::StaleSubject
    } else if has_not_executed {
        QualificationClassification::NotExecuted
    } else if has_incomplete {
        QualificationClassification::IncompleteJobSet
    } else if defects.get("unclassified_failure").copied().unwrap_or(0) > 0 {
        QualificationClassification::UnclassifiedFailure
    } else if failure_classes.len() > 1 {
        QualificationClassification::MultipleFailureClasses
    } else if failure_classes.contains(&FailureAttribution::Infrastructure) {
        QualificationClassification::InfrastructureIndeterminate
    } else if failure_classes.contains(&FailureAttribution::Qualifier) {
        QualificationClassification::FailQualifier
    } else if failure_classes.contains(&FailureAttribution::Subject) {
        QualificationClassification::FailSubject
    } else if issues.is_empty() {
        QualificationClassification::Pass
    } else {
        QualificationClassification::IncompleteJobSet
    };

    let mechanical_evidence = mechanical_run_evidence(profile, subject, observation, &defects);

    QualificationEvaluation {
        classification,
        issues,
        defect_counters: defects,
        mechanical_evidence,
    }
}

pub(super) fn build_qualified_head_receipt(
    profile: &QualificationProfile,
    subject: &QualificationSubject,
    observation: &QualificationRunObservation,
) -> Result<QualifiedHeadReceipt, QualificationEvaluation> {
    let evaluation = evaluate_qualification(profile, subject, observation);
    if evaluation.classification != QualificationClassification::Pass
        || !evaluation.mechanical_evidence.satisfied
    {
        return Err(evaluation);
    }

    let normalized_job_set_digest = match digest_serializable(&observation.jobs) {
        Ok(digest) => digest,
        Err(_) => {
            let mut failed = evaluation;
            failed.classification = QualificationClassification::InvalidObservation;
            increment(&mut failed.defect_counters, "serialization_failure");
            return Err(failed);
        }
    };

    let mut receipt = QualifiedHeadReceipt {
        receipt_revision: QUALIFICATION_RECEIPT_REVISION.into(),
        profile_replay_identity: profile.replay_identity.clone(),
        lane: profile.lane,
        subject: subject.clone(),
        workflow_run_id: observation.workflow_run_id,
        workflow_id: observation.workflow_id,
        run_attempt: observation.run_attempt,
        workflow_definition_identity: observation.workflow_definition_identity.clone(),
        materializer_revision: observation.materializer_revision.clone(),
        checked_out_subject_sha: observation.checked_out_subject_sha.clone(),
        observation_replay_identity: observation.replay_identity.clone(),
        normalized_job_set_digest,
        defect_counters: evaluation.defect_counters,
        classification: QualificationClassification::Pass,
        replay_identity: String::new(),
    };

    receipt.replay_identity = digest_serializable(&ReceiptCommitment::from(&receipt))
        .expect("receipt contains only serializable deterministic fields");
    debug_assert!(receipt.replay_valid());
    Ok(receipt)
}

fn mechanical_run_evidence(
    profile: &QualificationProfile,
    subject: &QualificationSubject,
    observation: &QualificationRunObservation,
    defects: &BTreeMap<String, u64>,
) -> RunEvidence {
    let declared: BTreeMap<String, Expectation> = defects
        .keys()
        .map(|name| (name.clone(), Expectation::MustBeZero))
        .collect();
    let mut measured = EvidenceCounters::new();
    for (name, value) in defects {
        measured.record(name.clone(), *value as f64);
    }

    RunEvidence::new(
        RunId::new(format!(
            "eureka-qualification-run-{}-attempt-{}",
            observation.workflow_run_id, observation.run_attempt
        )),
        &(
            profile.replay_identity.as_str(),
            subject,
            observation.replay_identity.as_str(),
        ),
        declared,
        measured,
    )
}

fn zero_defect_counters() -> BTreeMap<String, u64> {
    [
        "invalid_profile_replay",
        "invalid_observation_replay",
        "wrong_lane",
        "wrong_workflow_path",
        "wrong_workflow_definition",
        "disallowed_event",
        "provider_head_mismatch",
        "provider_base_mismatch",
        "checked_out_subject_mismatch",
        "missing_required_jobs",
        "required_non_success",
        "required_not_executed",
        "required_skipped",
        "required_cancelled",
        "required_incomplete_terminal_state",
        "unclassified_failure",
        "invalid_subject",
        "serialization_failure",
    ]
    .into_iter()
    .map(|name| (name.to_string(), 0))
    .collect()
}

fn increment(counters: &mut BTreeMap<String, u64>, name: &str) {
    *counters.entry(name.to_string()).or_insert(0) += 1;
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, QualificationBuildError> {
    let bytes = serde_json::to_vec(value).map_err(|_| QualificationBuildError::SerializationFailed)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn is_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_identity(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = "1111111111111111111111111111111111111111";
    const BASE: &str = "2222222222222222222222222222222222222222";
    const MERGE: &str = "3333333333333333333333333333333333333333";
    const WORKFLOW: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn profile(lane: QualificationLane) -> QualificationProfile {
        QualificationProfile::try_new(
            "profile-v1",
            lane,
            ".github/workflows/ci.yml",
            WORKFLOW,
            ["workflow_dispatch".to_string()],
            ["Format Check".to_string(), "Tests".to_string()],
            Vec::<String>::new(),
        )
        .unwrap()
    }

    fn job(id: u64, name: &str, disposition: JobDisposition) -> ObservedQualificationJob {
        ObservedQualificationJob {
            job_id: id,
            name: name.into(),
            disposition,
            failure_attribution: None,
            failure_evidence_identity: None,
        }
    }

    fn observation(
        checked_out_subject_sha: &str,
        jobs: Vec<ObservedQualificationJob>,
    ) -> QualificationRunObservation {
        QualificationRunObservation::try_new(
            "Luminous-Dynamics/symthaea",
            42,
            7,
            ".github/workflows/ci.yml",
            WORKFLOW,
            1,
            "workflow_dispatch",
            HEAD,
            None,
            checked_out_subject_sha,
            "github-materializer-v1",
            jobs,
        )
        .unwrap()
    }

    fn successful_jobs() -> Vec<ObservedQualificationJob> {
        vec![
            job(1, "Format Check", JobDisposition::Success),
            job(2, "Tests", JobDisposition::Success),
        ]
    }

    #[test]
    fn exact_head_all_success_mints_receipt() {
        let profile = profile(QualificationLane::FullExactHead);
        let subject = QualificationSubject::ExactHead {
            head_sha: HEAD.into(),
        };
        let observation = observation(HEAD, successful_jobs());

        let receipt = build_qualified_head_receipt(&profile, &subject, &observation).unwrap();
        assert_eq!(receipt.classification, QualificationClassification::Pass);
        assert_eq!(receipt.checked_out_subject_sha, HEAD);
        assert!(receipt.defect_counters.values().all(|value| *value == 0));
        assert!(receipt.replay_valid());
    }

    #[test]
    fn source_sanity_green_cannot_mint_authoritative_receipt() {
        let profile = profile(QualificationLane::SourceSanity);
        let subject = QualificationSubject::ExactHead {
            head_sha: HEAD.into(),
        };
        let observation = observation(HEAD, successful_jobs());

        let err = build_qualified_head_receipt(&profile, &subject, &observation).unwrap_err();
        assert_eq!(err.classification, QualificationClassification::WrongLane);
    }

    #[test]
    fn queued_required_job_is_not_executed_not_pass() {
        let profile = profile(QualificationLane::FullExactHead);
        let subject = QualificationSubject::ExactHead {
            head_sha: HEAD.into(),
        };
        let observation = observation(
            HEAD,
            vec![
                job(1, "Format Check", JobDisposition::Success),
                job(2, "Tests", JobDisposition::Queued),
            ],
        );

        let evaluation = evaluate_qualification(&profile, &subject, &observation);
        assert_eq!(evaluation.classification, QualificationClassification::NotExecuted);
        assert!(!evaluation.mechanical_evidence.satisfied);
    }

    #[test]
    fn skipped_required_job_is_incomplete_unless_predeclared() {
        let profile = profile(QualificationLane::FullExactHead);
        let subject = QualificationSubject::ExactHead {
            head_sha: HEAD.into(),
        };
        let observation = observation(
            HEAD,
            vec![
                job(1, "Format Check", JobDisposition::Success),
                job(2, "Tests", JobDisposition::Skipped),
            ],
        );
        assert_eq!(
            evaluate_qualification(&profile, &subject, &observation).classification,
            QualificationClassification::IncompleteJobSet
        );

        let allowed_profile = QualificationProfile::try_new(
            "profile-v2",
            QualificationLane::FullExactHead,
            ".github/workflows/ci.yml",
            WORKFLOW,
            ["workflow_dispatch".to_string()],
            ["Format Check".to_string(), "Tests".to_string()],
            ["Tests".to_string()],
        )
        .unwrap();
        assert!(build_qualified_head_receipt(&allowed_profile, &subject, &observation).is_ok());
    }

    #[test]
    fn provider_head_metadata_does_not_substitute_for_checkout_identity() {
        let profile = profile(QualificationLane::FullExactHead);
        let subject = QualificationSubject::ExactHead {
            head_sha: HEAD.into(),
        };
        // Provider still says HEAD while the actual in-run checkout is MERGE.
        let observation = observation(MERGE, successful_jobs());
        let evaluation = evaluate_qualification(&profile, &subject, &observation);
        assert_eq!(evaluation.classification, QualificationClassification::StaleSubject);
    }

    #[test]
    fn integration_subject_accepts_only_exact_recorded_merge_checkout() {
        let profile = profile(QualificationLane::FullExactHead);
        let subject = QualificationSubject::Integration {
            head_sha: HEAD.into(),
            base_sha: BASE.into(),
            merge_sha: MERGE.into(),
        };
        let observation = QualificationRunObservation::try_new(
            "Luminous-Dynamics/symthaea",
            43,
            7,
            ".github/workflows/ci.yml",
            WORKFLOW,
            1,
            "workflow_dispatch",
            HEAD,
            Some(BASE.into()),
            MERGE,
            "github-materializer-v1",
            successful_jobs(),
        )
        .unwrap();

        assert!(build_qualified_head_receipt(&profile, &subject, &observation).is_ok());
    }

    #[test]
    fn subject_failure_requires_explicit_attribution_evidence() {
        let profile = profile(QualificationLane::FullExactHead);
        let subject = QualificationSubject::ExactHead {
            head_sha: HEAD.into(),
        };
        let failed = ObservedQualificationJob {
            job_id: 2,
            name: "Tests".into(),
            disposition: JobDisposition::Failure,
            failure_attribution: Some(FailureAttribution::Subject),
            failure_evidence_identity: Some("b".repeat(64)),
        };
        let observation = observation(
            HEAD,
            vec![job(1, "Format Check", JobDisposition::Success), failed],
        );

        let evaluation = evaluate_qualification(&profile, &subject, &observation);
        assert_eq!(evaluation.classification, QualificationClassification::FailSubject);
    }

    #[test]
    fn unattributed_failure_remains_unclassified() {
        let profile = profile(QualificationLane::FullExactHead);
        let subject = QualificationSubject::ExactHead {
            head_sha: HEAD.into(),
        };
        let observation = observation(
            HEAD,
            vec![
                job(1, "Format Check", JobDisposition::Success),
                job(2, "Tests", JobDisposition::Failure),
            ],
        );

        assert_eq!(
            evaluate_qualification(&profile, &subject, &observation).classification,
            QualificationClassification::UnclassifiedFailure
        );
    }

    #[test]
    fn observation_identity_is_invariant_to_job_input_order() {
        let one = observation(
            HEAD,
            vec![
                job(1, "Format Check", JobDisposition::Success),
                job(2, "Tests", JobDisposition::Success),
            ],
        );
        let two = observation(
            HEAD,
            vec![
                job(2, "Tests", JobDisposition::Success),
                job(1, "Format Check", JobDisposition::Success),
            ],
        );
        assert_eq!(one.replay_identity, two.replay_identity);
    }

    #[test]
    fn run_attempt_changes_qualification_identity() {
        let profile = profile(QualificationLane::FullExactHead);
        let subject = QualificationSubject::ExactHead {
            head_sha: HEAD.into(),
        };
        let one = observation(HEAD, successful_jobs());
        let two = QualificationRunObservation::try_new(
            "Luminous-Dynamics/symthaea",
            42,
            7,
            ".github/workflows/ci.yml",
            WORKFLOW,
            2,
            "workflow_dispatch",
            HEAD,
            None,
            HEAD,
            "github-materializer-v1",
            successful_jobs(),
        )
        .unwrap();

        let receipt_one = build_qualified_head_receipt(&profile, &subject, &one).unwrap();
        let receipt_two = build_qualified_head_receipt(&profile, &subject, &two).unwrap();
        assert_ne!(receipt_one.replay_identity, receipt_two.replay_identity);
    }

    #[test]
    fn invalid_profile_workflow_identity_rejects() {
        let err = QualificationProfile::try_new(
            "profile-v1",
            QualificationLane::FullExactHead,
            ".github/workflows/ci.yml",
            "not-a-digest",
            ["workflow_dispatch".to_string()],
            ["Format Check".to_string()],
            Vec::<String>::new(),
        )
        .unwrap_err();
        assert_eq!(err, QualificationBuildError::InvalidWorkflowIdentity);
    }
}
