// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provider-neutral exact-head qualification evidence.
//!
//! CI/API access belongs outside this module. Authority identities use explicit
//! canonical bytes + domain-separated BLAKE3. The crate-level `config_hash`
//! remains a non-cryptographic diagnostic fingerprint and is never used here.

pub mod witness;

use crate::{EvidenceCounters, Expectation, RunEvidence, RunId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

const PROFILE_DOMAIN: &[u8] = b"symthaea.qualification.profile.v1\0";
const OBS_DOMAIN: &[u8] = b"symthaea.qualification.observation.v1\0";
const JOBS_DOMAIN: &[u8] = b"symthaea.qualification.jobs.v1\0";
const MECH_DOMAIN: &[u8] = b"symthaea.qualification.mechanical.v1\0";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.qualification.receipt.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QualificationDigest(pub [u8; 32]);

impl fmt::Display for QualificationDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArtifactIdentity {
    GitBlobSha1([u8; 20]),
    Sha256([u8; 32]),
    Blake3([u8; 32]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationLane {
    SourceSanity,
    FullExactHead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobDisposition {
    Passed,
    FailSubject,
    FailQualifier,
    InfrastructureIndeterminate,
    Skipped,
    Cancelled,
    NotExecuted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationClassification {
    Pass,
    FailSubject,
    FailQualifier,
    InfrastructureIndeterminate,
    NotExecuted,
    WrongLane,
    StaleSubject,
    IncompleteJobSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationJobRequirement {
    pub name: String,
    pub allow_skip: bool,
}

impl QualificationJobRequirement {
    pub fn required(name: impl Into<String>) -> Self {
        Self { name: name.into(), allow_skip: false }
    }

    pub fn conditionally_skippable(name: impl Into<String>) -> Self {
        Self { name: name.into(), allow_skip: true }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationProfile {
    pub revision: u32,
    pub lane: QualificationLane,
    pub repository: String,
    pub workflow_path: String,
    pub workflow_definition: ArtifactIdentity,
    pub allowed_events: Vec<String>,
    pub required_jobs: Vec<QualificationJobRequirement>,
}

impl QualificationProfile {
    pub fn new(
        revision: u32,
        lane: QualificationLane,
        repository: impl Into<String>,
        workflow_path: impl Into<String>,
        workflow_definition: ArtifactIdentity,
        mut allowed_events: Vec<String>,
        mut required_jobs: Vec<QualificationJobRequirement>,
    ) -> Result<Self, QualificationError> {
        let repository = repository.into();
        let workflow_path = workflow_path.into();
        if repository.is_empty() {
            return Err(QualificationError::EmptyField("repository"));
        }
        if workflow_path.is_empty() {
            return Err(QualificationError::EmptyField("workflow_path"));
        }
        if allowed_events.is_empty() {
            return Err(QualificationError::EmptyAllowedEvents);
        }
        if required_jobs.is_empty() {
            return Err(QualificationError::EmptyRequiredJobs);
        }
        if allowed_events.iter().any(String::is_empty) {
            return Err(QualificationError::EmptyField("allowed_event"));
        }
        if required_jobs.iter().any(|j| j.name.is_empty()) {
            return Err(QualificationError::EmptyField("required_job"));
        }

        allowed_events.sort();
        if allowed_events.windows(2).any(|w| w[0] == w[1]) {
            return Err(QualificationError::DuplicateAllowedEvent);
        }
        required_jobs.sort_by(|a, b| a.name.cmp(&b.name));
        if required_jobs.windows(2).any(|w| w[0].name == w[1].name) {
            return Err(QualificationError::DuplicateRequiredJob);
        }

        Ok(Self {
            revision,
            lane,
            repository,
            workflow_path,
            workflow_definition,
            allowed_events,
            required_jobs,
        })
    }

    pub fn identity(&self) -> QualificationDigest {
        let mut w = Writer::new(PROFILE_DOMAIN);
        w.u32(self.revision);
        w.u8(lane_tag(self.lane));
        w.str(&self.repository);
        w.str(&self.workflow_path);
        w.artifact(self.workflow_definition);
        w.u32(self.allowed_events.len() as u32);
        for event in &self.allowed_events {
            w.str(event);
        }
        w.u32(self.required_jobs.len() as u32);
        for job in &self.required_jobs {
            w.str(&job.name);
            w.bool(job.allow_skip);
        }
        w.finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedQualificationStep {
    pub ordinal: u32,
    pub name: String,
    pub disposition: JobDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedQualificationJob {
    pub job_id: u64,
    pub name: String,
    pub disposition: JobDisposition,
    pub steps: Vec<ObservedQualificationStep>,
    /// Descriptive only; excluded from authority identity.
    pub provider_status: Option<String>,
    /// Descriptive only; excluded from authority identity.
    pub provider_conclusion: Option<String>,
}

impl ObservedQualificationJob {
    fn validate(&self) -> Result<(), QualificationError> {
        if self.name.is_empty() {
            return Err(QualificationError::EmptyField("job_name"));
        }
        if self.steps.iter().any(|step| step.name.is_empty()) {
            return Err(QualificationError::EmptyField("step_name"));
        }

        let mut seen = std::collections::BTreeSet::new();
        for step in &self.steps {
            if !seen.insert((step.ordinal, step.name.as_str())) {
                return Err(QualificationError::DuplicateObservedStep);
            }
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut w = Writer::empty();
        w.u64(self.job_id);
        w.str(&self.name);
        w.u8(job_tag(self.disposition));

        let mut steps: Vec<_> = self.steps.iter().collect();
        steps.sort_by(|a, b| (a.ordinal, a.name.as_str()).cmp(&(b.ordinal, b.name.as_str())));
        w.u32(steps.len() as u32);
        for step in steps {
            w.u32(step.ordinal);
            w.str(&step.name);
            w.u8(job_tag(step.disposition));
        }

        w.into_bytes()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationRunObservation {
    pub repository: String,
    pub workflow_run_id: u64,
    pub workflow_id: u64,
    pub workflow_path: String,
    pub workflow_definition: ArtifactIdentity,
    pub run_attempt: u32,
    pub event: String,
    pub exact_head_sha: String,
    pub jobs: Vec<ObservedQualificationJob>,
    pub materializer_revision: u32,
    /// Descriptive refs/status are intentionally non-authoritative.
    pub head_ref: Option<String>,
    pub base_ref: Option<String>,
    pub provider_status: Option<String>,
    pub provider_conclusion: Option<String>,
}

impl QualificationRunObservation {
    pub fn validate(&self) -> Result<(), QualificationError> {
        if self.repository.is_empty() {
            return Err(QualificationError::EmptyField("repository"));
        }
        if self.workflow_path.is_empty() {
            return Err(QualificationError::EmptyField("workflow_path"));
        }
        if self.event.is_empty() {
            return Err(QualificationError::EmptyField("event"));
        }
        validate_sha(&self.exact_head_sha)?;
        for job in &self.jobs {
            job.validate()?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<QualificationDigest, QualificationError> {
        self.validate()?;
        let mut w = Writer::new(OBS_DOMAIN);
        w.str(&self.repository);
        w.u64(self.workflow_run_id);
        w.u64(self.workflow_id);
        w.str(&self.workflow_path);
        w.artifact(self.workflow_definition);
        w.u32(self.run_attempt);
        w.str(&self.event);
        w.str(&self.exact_head_sha);
        w.u32(self.materializer_revision);
        write_jobs(&mut w, &self.jobs);
        Ok(w.finish())
    }

    pub fn job_set_identity(&self) -> QualificationDigest {
        let mut w = Writer::new(JOBS_DOMAIN);
        write_jobs(&mut w, &self.jobs);
        w.finish()
    }
}

#[derive(Debug, Clone)]
pub struct QualificationEvaluation {
    pub classification: QualificationClassification,
    pub profile_identity: QualificationDigest,
    pub observation_identity: QualificationDigest,
    pub job_set_identity: QualificationDigest,
    pub mechanical_identity: QualificationDigest,
    pub mechanical_integrity: RunEvidence,
}

impl QualificationProfile {
    pub fn evaluate(
        &self,
        expected_subject_sha: &str,
        observation: &QualificationRunObservation,
    ) -> Result<QualificationEvaluation, QualificationError> {
        validate_sha(expected_subject_sha)?;
        observation.validate()?;

        let mut counts: BTreeMap<&'static str, u64> = [
            "missing", "duplicate", "disallowed_skip", "cancelled",
            "subject_fail", "qualifier_fail", "infra", "not_executed",
            "stale", "wrong_repository", "wrong_workflow", "wrong_event", "unexpected_job",
        ]
        .into_iter()
        .map(|k| (k, 0))
        .collect();

        let mut observed_by_name: BTreeMap<&str, Vec<&ObservedQualificationJob>> = BTreeMap::new();
        for job in &observation.jobs {
            observed_by_name.entry(&job.name).or_default().push(job);
        }

        let required_names: std::collections::BTreeSet<&str> =
            self.required_jobs.iter().map(|job| job.name.as_str()).collect();
        counts.insert(
            "unexpected_job",
            observation
                .jobs
                .iter()
                .filter(|job| !required_names.contains(job.name.as_str()))
                .count() as u64,
        );

        for requirement in &self.required_jobs {
            let jobs = observed_by_name.get(requirement.name.as_str());
            match jobs {
                None => *counts.get_mut("missing").unwrap() += 1,
                Some(jobs) if jobs.len() != 1 => *counts.get_mut("duplicate").unwrap() += 1,
                Some(jobs) => match jobs[0].disposition {
                    JobDisposition::Passed => {}
                    JobDisposition::FailSubject => *counts.get_mut("subject_fail").unwrap() += 1,
                    JobDisposition::FailQualifier => *counts.get_mut("qualifier_fail").unwrap() += 1,
                    JobDisposition::InfrastructureIndeterminate => *counts.get_mut("infra").unwrap() += 1,
                    JobDisposition::Skipped if requirement.allow_skip => {}
                    JobDisposition::Skipped => *counts.get_mut("disallowed_skip").unwrap() += 1,
                    JobDisposition::Cancelled => *counts.get_mut("cancelled").unwrap() += 1,
                    JobDisposition::NotExecuted => *counts.get_mut("not_executed").unwrap() += 1,
                },
            }
        }

        counts.insert("stale", u64::from(observation.exact_head_sha != expected_subject_sha));
        counts.insert(
            "wrong_repository",
            u64::from(observation.repository != self.repository),
        );
        counts.insert(
            "wrong_workflow",
            u64::from(
                observation.workflow_path != self.workflow_path
                    || observation.workflow_definition != self.workflow_definition,
            ),
        );
        counts.insert(
            "wrong_event",
            u64::from(self.allowed_events.binary_search(&observation.event).is_err()),
        );

        let mut declared = BTreeMap::new();
        let mut measured = EvidenceCounters::new();
        for (name, value) in &counts {
            declared.insert((*name).to_string(), Expectation::MustBeZero);
            measured.record(*name, *value as f64);
        }

        let profile_identity = self.identity();
        let observation_identity = observation.identity()?;
        let mechanical_integrity = RunEvidence::new(
            RunId::new(format!(
                "qualification:{}:{}",
                observation.workflow_run_id, observation.run_attempt
            )),
            &(profile_identity, observation_identity),
            declared,
            measured,
        );

        let classification = if self.lane != QualificationLane::FullExactHead {
            QualificationClassification::WrongLane
        } else if counts["stale"] > 0 {
            QualificationClassification::StaleSubject
        } else if counts["wrong_repository"] > 0
            || counts["wrong_workflow"] > 0
            || counts["wrong_event"] > 0
            || counts["duplicate"] > 0
            || counts["unexpected_job"] > 0
        {
            QualificationClassification::FailQualifier
        } else if counts["missing"] > 0 || counts["disallowed_skip"] > 0 {
            QualificationClassification::IncompleteJobSet
        } else if counts["not_executed"] > 0 {
            QualificationClassification::NotExecuted
        } else if counts["infra"] > 0 || counts["cancelled"] > 0 {
            QualificationClassification::InfrastructureIndeterminate
        } else if counts["qualifier_fail"] > 0 {
            QualificationClassification::FailQualifier
        } else if counts["subject_fail"] > 0 {
            QualificationClassification::FailSubject
        } else {
            QualificationClassification::Pass
        };

        let mechanical_identity =
            mechanical_digest(profile_identity, observation_identity, classification, &counts);

        Ok(QualificationEvaluation {
            classification,
            profile_identity,
            observation_identity,
            job_set_identity: observation.job_set_identity(),
            mechanical_identity,
            mechanical_integrity,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualifiedHeadReceipt {
    revision: u32,
    profile_identity: QualificationDigest,
    expected_subject_sha: String,
    observed_head_sha: String,
    workflow_definition: ArtifactIdentity,
    workflow_run_id: u64,
    run_attempt: u32,
    materializer_revision: u32,
    job_set_identity: QualificationDigest,
    mechanical_identity: QualificationDigest,
}

impl QualifiedHeadReceipt {
    pub fn try_new(
        profile: &QualificationProfile,
        expected_subject_sha: &str,
        observation: &QualificationRunObservation,
    ) -> Result<Self, QualificationError> {
        validate_sha(expected_subject_sha)?;
        if profile.lane != QualificationLane::FullExactHead {
            return Err(QualificationError::ReceiptRequiresFullExactHead);
        }

        // Recompute classification and mechanical integrity inside the minting
        // boundary. Never accept caller-supplied evaluation state as authority.
        let evaluation = profile.evaluate(expected_subject_sha, observation)?;
        if evaluation.classification != QualificationClassification::Pass
            || !evaluation.mechanical_integrity.satisfied
        {
            return Err(QualificationError::ReceiptRequiresPass);
        }
        if observation.exact_head_sha != expected_subject_sha {
            return Err(QualificationError::ReceiptSubjectMismatch);
        }

        Ok(Self {
            revision: 1,
            profile_identity: evaluation.profile_identity,
            expected_subject_sha: expected_subject_sha.to_string(),
            observed_head_sha: observation.exact_head_sha.clone(),
            workflow_definition: observation.workflow_definition,
            workflow_run_id: observation.workflow_run_id,
            run_attempt: observation.run_attempt,
            materializer_revision: observation.materializer_revision,
            job_set_identity: evaluation.job_set_identity,
            mechanical_identity: evaluation.mechanical_identity,
        })
    }

    pub fn profile_identity(&self) -> QualificationDigest {
        self.profile_identity
    }

    pub fn expected_subject_sha(&self) -> &str {
        &self.expected_subject_sha
    }

    pub fn observed_head_sha(&self) -> &str {
        &self.observed_head_sha
    }

    pub fn workflow_definition(&self) -> ArtifactIdentity {
        self.workflow_definition
    }

    pub fn workflow_run_id(&self) -> u64 {
        self.workflow_run_id
    }

    pub fn run_attempt(&self) -> u32 {
        self.run_attempt
    }

    pub fn materializer_revision(&self) -> u32 {
        self.materializer_revision
    }

    pub fn job_set_identity(&self) -> QualificationDigest {
        self.job_set_identity
    }

    pub fn mechanical_identity(&self) -> QualificationDigest {
        self.mechanical_identity
    }

    pub fn identity(&self) -> QualificationDigest {
        let mut w = Writer::new(RECEIPT_DOMAIN);
        w.u32(self.revision);
        w.digest(self.profile_identity);
        w.str(&self.expected_subject_sha);
        w.str(&self.observed_head_sha);
        w.artifact(self.workflow_definition);
        w.u64(self.workflow_run_id);
        w.u32(self.run_attempt);
        w.u32(self.materializer_revision);
        w.digest(self.job_set_identity);
        w.digest(self.mechanical_identity);
        w.finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationError {
    EmptyField(&'static str),
    EmptyAllowedEvents,
    EmptyRequiredJobs,
    DuplicateAllowedEvent,
    DuplicateRequiredJob,
    DuplicateObservedStep,
    InvalidSha,
    ReceiptRequiresFullExactHead,
    ReceiptRequiresPass,
    ReceiptSubjectMismatch,
}

impl fmt::Display for QualificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for QualificationError {}

fn validate_sha(sha: &str) -> Result<(), QualificationError> {
    if matches!(sha.len(), 40 | 64)
        && sha.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(())
    } else {
        Err(QualificationError::InvalidSha)
    }
}

fn write_jobs(w: &mut Writer, jobs: &[ObservedQualificationJob]) {
    let mut jobs: Vec<_> = jobs.iter().collect();
    jobs.sort_by(|a, b| (a.name.as_str(), a.job_id).cmp(&(b.name.as_str(), b.job_id)));
    w.u32(jobs.len() as u32);
    for job in jobs {
        w.bytes(&job.canonical_bytes());
    }
}

fn mechanical_digest(
    profile: QualificationDigest,
    observation: QualificationDigest,
    classification: QualificationClassification,
    counts: &BTreeMap<&'static str, u64>,
) -> QualificationDigest {
    let mut w = Writer::new(MECH_DOMAIN);
    w.digest(profile);
    w.digest(observation);
    w.u8(classification_tag(classification));
    w.u32(counts.len() as u32);
    for (name, value) in counts {
        w.str(name);
        w.u64(*value);
    }
    w.finish()
}

fn lane_tag(v: QualificationLane) -> u8 {
    match v {
        QualificationLane::SourceSanity => 0,
        QualificationLane::FullExactHead => 1,
    }
}
fn job_tag(v: JobDisposition) -> u8 {
    match v {
        JobDisposition::Passed => 0,
        JobDisposition::FailSubject => 1,
        JobDisposition::FailQualifier => 2,
        JobDisposition::InfrastructureIndeterminate => 3,
        JobDisposition::Skipped => 4,
        JobDisposition::Cancelled => 5,
        JobDisposition::NotExecuted => 6,
    }
}
fn classification_tag(v: QualificationClassification) -> u8 {
    match v {
        QualificationClassification::Pass => 0,
        QualificationClassification::FailSubject => 1,
        QualificationClassification::FailQualifier => 2,
        QualificationClassification::InfrastructureIndeterminate => 3,
        QualificationClassification::NotExecuted => 4,
        QualificationClassification::WrongLane => 5,
        QualificationClassification::StaleSubject => 6,
        QualificationClassification::IncompleteJobSet => 7,
    }
}

struct Writer {
    buf: Vec<u8>,
}
impl Writer {
    fn new(domain: &[u8]) -> Self {
        let mut w = Self::empty();
        w.bytes(domain);
        w
    }
    fn empty() -> Self {
        Self { buf: Vec::new() }
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn bool(&mut self, v: bool) {
        self.u8(u8::from(v));
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, v: &str) {
        self.bytes(v.as_bytes());
    }
    fn digest(&mut self, v: QualificationDigest) {
        self.buf.extend_from_slice(&v.0);
    }
    fn artifact(&mut self, v: ArtifactIdentity) {
        match v {
            ArtifactIdentity::GitBlobSha1(bytes) => {
                self.u8(0);
                self.bytes(&bytes);
            }
            ArtifactIdentity::Sha256(bytes) => {
                self.u8(1);
                self.bytes(&bytes);
            }
            ArtifactIdentity::Blake3(bytes) => {
                self.u8(2);
                self.bytes(&bytes);
            }
        }
    }
    fn bytes(&mut self, v: &[u8]) {
        self.u32(v.len() as u32);
        self.buf.extend_from_slice(v);
    }
    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
    fn finish(self) -> QualificationDigest {
        QualificationDigest(*blake3::hash(&self.buf).as_bytes())
    }
}

#[cfg(test)]
mod tests;
