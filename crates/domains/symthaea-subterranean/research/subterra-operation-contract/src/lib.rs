//! Bounded physical-operation state machine for Subterra.
//!
//! This crate is a dependency-free reference contract. It validates local
//! state transitions and evidence metadata; it does not authenticate
//! signatures, control hardware, prove physical truth, or replace an
//! independent engineering verifier.
//!
//! The central rule is that command receipt, actuator response, measured
//! physical change, and independently accepted postconditions are distinct.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

/// Upper bound on retained events per attempt; unresolved work remains locked
/// when this limit is reached.
pub const MAX_ATTEMPT_EVENTS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkScopeId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttemptId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkPackageId(pub u64);

/// Opaque evidence identifier. This type does not calculate or authenticate a
/// digest; callers must supply one from a reviewed digest/signature layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EvidenceDigest(pub [u8; 32]);

impl EvidenceDigest {
    pub const ZERO: Self = Self([0; 32]);

    pub const fn is_present(self) -> bool {
        let mut index = 0;
        while index < self.0.len() {
            if self.0[index] != 0 {
                return true;
            }
            index += 1;
        }
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationSpec {
    pub scope_id: WorkScopeId,
    pub attempt_id: AttemptId,
    pub work_package_id: WorkPackageId,
    pub work_package_revision: u64,
    pub work_package_digest: EvidenceDigest,
    pub design_revision: u64,
    pub design_digest: EvidenceDigest,
    pub coordinate_frame_revision: u64,
    pub coordinate_frame_digest: EvidenceDigest,
    pub calibration_revision: u64,
    pub calibration_digest: EvidenceDigest,
    pub authority_revision: u64,
    pub authority_evidence_digest: EvidenceDigest,
    pub executor_id: u64,
    pub executor_sensor_chain_id: u64,
    /// Declared common-cause domains for the executor's sensing/evidence path.
    pub executor_common_cause_domains: Vec<u64>,
    /// Required fraction of the declared inspection surface/volume, in 0..=1000.
    pub required_coverage_per_mille: u16,
    pub expires_at_step: u64,
}

impl OperationSpec {
    pub fn validate(&self, now_step: u64) -> Result<(), ContractError> {
        if self.scope_id.0 == 0
            || self.attempt_id.0 == 0
            || self.work_package_id.0 == 0
            || self.work_package_revision == 0
            || !self.work_package_digest.is_present()
            || self.design_revision == 0
            || !self.design_digest.is_present()
            || self.coordinate_frame_revision == 0
            || !self.coordinate_frame_digest.is_present()
            || self.calibration_revision == 0
            || !self.calibration_digest.is_present()
            || self.authority_revision == 0
            || !self.authority_evidence_digest.is_present()
            || self.executor_id == 0
            || self.executor_sensor_chain_id == 0
            || self.executor_common_cause_domains.is_empty()
            || self.required_coverage_per_mille == 0
            || self.required_coverage_per_mille > 1000
            || self.expires_at_step < now_step
        {
            return Err(ContractError::InvalidSpecification);
        }

        let mut domains = BTreeSet::new();
        if self.executor_common_cause_domains.contains(&0)
            || self
                .executor_common_cause_domains
                .iter()
                .any(|domain| !domains.insert(*domain))
        {
            return Err(ContractError::InvalidCommonCauseDeclaration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptState {
    Approved,
    PreflightAccepted,
    CommandIssued,
    Acknowledged,
    ActuationObserved,
    OutcomeUnknown,
    ReconciledNotExecuted,
    ReconciledExecutedAwaitingInspection,
    PhysicalDeltaObserved,
    VerificationPending,
    InsufficientEvidence,
    Accepted,
    Rejected,
    ReworkReview,
    ReworkSuperseded,
    Closed,
}

impl AttemptState {
    /// States in which this attempt still blocks a new mutation in its scope.
    pub const fn locks_scope(self) -> bool {
        !matches!(self, Self::ReconciledNotExecuted | Self::ReworkSuperseded | Self::Closed)
    }

    pub const fn is_terminal_for_attempt(self) -> bool {
        matches!(self, Self::ReconciledNotExecuted | Self::ReworkSuperseded | Self::Closed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationEvent {
    PreflightAccepted {
        work_package_revision: u64,
        design_revision: u64,
        coordinate_frame_revision: u64,
        calibration_revision: u64,
        authority_revision: u64,
        work_package_digest: EvidenceDigest,
        design_digest: EvidenceDigest,
        coordinate_frame_digest: EvidenceDigest,
        calibration_digest: EvidenceDigest,
        authority_evidence_digest: EvidenceDigest,
    },
    CommandIssued {
        command_digest: EvidenceDigest,
        idempotency_key: EvidenceDigest,
    },
    DeviceAcknowledged {
        command_digest: EvidenceDigest,
    },
    ActuatorTelemetryObserved {
        command_digest: EvidenceDigest,
        telemetry_digest: EvidenceDigest,
        calibration_revision: u64,
        calibration_digest: EvidenceDigest,
    },
    OutcomeTimedOut,
    ReconciledNotExecuted {
        reconciler_id: u64,
        evidence_digest: EvidenceDigest,
    },
    ReconciledExecuted {
        reconciler_id: u64,
        evidence_digest: EvidenceDigest,
    },
    PhysicalDeltaObserved {
        evidence: PhysicalDeltaEvidence,
    },
    VerificationSubmitted {
        receipt: VerificationReceipt,
    },
    AdditionalEvidenceReceived {
        evidence_digest: EvidenceDigest,
    },
    CloseAccepted,
    RecordReworkReview {
        authority_revision: u64,
        disposition_digest: EvidenceDigest,
    },
    /// A responsible authority supersedes this rejected attempt with a new,
    /// separately identified work package. Registry admission requires the
    /// next attempt to match this exact authorization.
    AuthorizeRework {
        authorization: ReworkAuthorization,
        approval_digest: EvidenceDigest,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReworkAuthorization {
    pub work_package_id: WorkPackageId,
    pub work_package_revision: u64,
    pub work_package_digest: EvidenceDigest,
    pub design_revision: u64,
    pub design_digest: EvidenceDigest,
    pub coordinate_frame_revision: u64,
    pub coordinate_frame_digest: EvidenceDigest,
    pub calibration_revision: u64,
    pub calibration_digest: EvidenceDigest,
    pub authority_revision: u64,
    pub authority_evidence_digest: EvidenceDigest,
}

impl ReworkAuthorization {
    fn validate_against(&self, prior: &OperationSpec) -> bool {
        self.work_package_id.0 != 0
            && self.work_package_revision > prior.work_package_revision
            && self.work_package_digest.is_present()
            && self.work_package_digest != prior.work_package_digest
            && self.design_revision > 0
            && self.design_digest.is_present()
            && self.coordinate_frame_revision > 0
            && self.coordinate_frame_digest.is_present()
            && self.calibration_revision > 0
            && self.calibration_digest.is_present()
            && self.authority_revision > prior.authority_revision
            && self.authority_evidence_digest.is_present()
    }

    fn matches_spec(&self, spec: &OperationSpec) -> bool {
        self.work_package_id == spec.work_package_id
            && self.work_package_revision == spec.work_package_revision
            && self.work_package_digest == spec.work_package_digest
            && self.design_revision == spec.design_revision
            && self.design_digest == spec.design_digest
            && self.coordinate_frame_revision == spec.coordinate_frame_revision
            && self.coordinate_frame_digest == spec.coordinate_frame_digest
            && self.calibration_revision == spec.calibration_revision
            && self.calibration_digest == spec.calibration_digest
            && self.authority_revision == spec.authority_revision
            && self.authority_evidence_digest == spec.authority_evidence_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalDeltaEvidence {
    pub source_robot_id: u64,
    pub sensor_chain_id: u64,
    pub work_package_digest: EvidenceDigest,
    pub design_revision: u64,
    pub design_digest: EvidenceDigest,
    pub coordinate_frame_revision: u64,
    pub coordinate_frame_digest: EvidenceDigest,
    pub calibration_revision: u64,
    pub calibration_digest: EvidenceDigest,
    pub coverage_per_mille: u16,
    pub unobserved_required_regions: u32,
    pub digest: EvidenceDigest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReceipt {
    pub verifier_id: u64,
    pub sensor_chain_id: u64,
    pub work_package_digest: EvidenceDigest,
    pub design_revision: u64,
    pub design_digest: EvidenceDigest,
    pub coordinate_frame_revision: u64,
    pub coordinate_frame_digest: EvidenceDigest,
    pub calibration_revision: u64,
    pub calibration_digest: EvidenceDigest,
    pub coverage_per_mille: u16,
    pub unobserved_required_regions: u32,
    pub common_cause_domains: Vec<u64>,
    /// Evidence from a reviewed common-cause/dependency assessment; this crate
    /// checks presence and declared overlap, not the assessment's truth.
    pub common_cause_assessment_digest: EvidenceDigest,
    pub physical_delta_digest: EvidenceDigest,
    pub postconditions_passed: bool,
    pub digest: EvidenceDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationIssue {
    MissingDigest,
    VerifierIsExecutor,
    SensorChainShared,
    CoordinateFrameMismatch,
    CalibrationMismatch,
    CoverageInsufficient,
    RequiredRegionUnobserved,
    CommonCauseOverlap,
    InvalidCommonCauseDeclaration,
    MissingIndependenceAssessment,
    PhysicalDeltaMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationAssessment {
    pub issues: Vec<VerificationIssue>,
    pub accepted: bool,
    pub rejected: bool,
}

impl VerificationReceipt {
    pub fn assess(
        &self,
        spec: &OperationSpec,
        physical_delta: &PhysicalDeltaEvidence,
    ) -> VerificationAssessment {
        let mut issues = Vec::new();

        if !self.digest.is_present() {
            issues.push(VerificationIssue::MissingDigest);
        }
        if !self.common_cause_assessment_digest.is_present() {
            issues.push(VerificationIssue::MissingIndependenceAssessment);
        }
        if self.physical_delta_digest != physical_delta.digest {
            issues.push(VerificationIssue::PhysicalDeltaMismatch);
        }
        if self.verifier_id == 0 || self.verifier_id == spec.executor_id {
            issues.push(VerificationIssue::VerifierIsExecutor);
        }
        if self.sensor_chain_id == 0 || self.sensor_chain_id == spec.executor_sensor_chain_id {
            issues.push(VerificationIssue::SensorChainShared);
        }
        if self.work_package_digest != spec.work_package_digest
            || self.design_revision != spec.design_revision
            || self.design_digest != spec.design_digest
        {
            issues.push(VerificationIssue::CoordinateFrameMismatch);
        }
        if self.coordinate_frame_revision != spec.coordinate_frame_revision
            || self.coordinate_frame_digest != spec.coordinate_frame_digest
        {
            issues.push(VerificationIssue::CoordinateFrameMismatch);
        }
        if self.calibration_revision != spec.calibration_revision
            || self.calibration_digest != spec.calibration_digest {
            issues.push(VerificationIssue::CalibrationMismatch);
        }
        if self.coverage_per_mille < spec.required_coverage_per_mille
            || self.coverage_per_mille > 1000
        {
            issues.push(VerificationIssue::CoverageInsufficient);
        }
        if self.unobserved_required_regions != 0 {
            issues.push(VerificationIssue::RequiredRegionUnobserved);
        }

        let mut domains = BTreeSet::new();
        if self.common_cause_domains.is_empty() || self.common_cause_domains.contains(&0)
            || self
                .common_cause_domains
                .iter()
                .any(|domain| !domains.insert(*domain))
        {
            issues.push(VerificationIssue::InvalidCommonCauseDeclaration);
        } else if self
            .common_cause_domains
            .iter()
            .any(|domain| spec.executor_common_cause_domains.contains(domain))
        {
            issues.push(VerificationIssue::CommonCauseOverlap);
        }

        let qualified = issues.is_empty();
        VerificationAssessment {
            issues,
            accepted: qualified && self.postconditions_passed,
            rejected: qualified && !self.postconditions_passed,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StampedOperationEvent {
    /// Strictly increasing per-attempt event sequence, beginning at one.
    pub sequence: u64,
    /// Monotonic mission step; wall-clock timestamps are intentionally not used
    /// as causal proof by this local reference contract.
    pub observed_at_step: u64,
    pub work_package_id: WorkPackageId,
    pub work_package_revision: u64,
    pub work_package_digest: EvidenceDigest,
    pub event: OperationEvent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionRecord {
    pub sequence: u64,
    pub observed_at_step: u64,
    pub from: AttemptState,
    pub to: AttemptState,
    /// Full typed event retained so a transition cannot be separated from the
    /// evidence references and revision bindings that caused it.
    pub event: OperationEvent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    InvalidSpecification,
    InvalidCommonCauseDeclaration,
    EventSequenceGap { expected: u64, received: u64 },
    EventSequenceExhausted,
    AttemptEventLimitExceeded,
    WorkPackageIdentityMismatch,
    StepRegression,
    WorkPackageRevisionMismatch,
    WorkPackageExpired,
    InvalidTransition { from: AttemptState, event: &'static str },
    MissingDigest,
    IdempotencyKeyReused,
    CommandDigestMismatch,
    CalibrationMismatch,
    CoordinateFrameMismatch,
    InvalidEvidenceSource,
    ReconcilerMustBeIndependent,
    DuplicateEvidence,
    InvalidAuthorityRevision,
    DuplicateAttempt,
    ScopeAlreadyLocked { scope_id: WorkScopeId, holder: AttemptId },
    UnknownAttempt(AttemptId),
}

#[derive(Debug, Clone)]
pub struct OperationAttempt {
    spec: OperationSpec,
    state: AttemptState,
    next_sequence: u64,
    last_observed_at_step: u64,
    command_digest: Option<EvidenceDigest>,
    idempotency_key: Option<EvidenceDigest>,
    physical_delta: Option<PhysicalDeltaEvidence>,
    last_verification: Option<VerificationAssessment>,
    seen_evidence: BTreeSet<EvidenceDigest>,
    history: Vec<TransitionRecord>,
}

impl OperationAttempt {
    pub fn new(spec: OperationSpec, now_step: u64) -> Result<Self, ContractError> {
        spec.validate(now_step)?;
        Ok(Self {
            spec,
            state: AttemptState::Approved,
            next_sequence: 1,
            last_observed_at_step: now_step,
            command_digest: None,
            idempotency_key: None,
            physical_delta: None,
            last_verification: None,
            seen_evidence: BTreeSet::new(),
            history: Vec::new(),
        })
    }

    pub const fn state(&self) -> AttemptState {
        self.state
    }

    pub fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    pub fn history(&self) -> &[TransitionRecord] {
        &self.history
    }

    pub fn last_verification(&self) -> Option<&VerificationAssessment> {
        self.last_verification.as_ref()
    }

    pub fn physical_delta(&self) -> Option<&PhysicalDeltaEvidence> {
        self.physical_delta.as_ref()
    }

    pub fn apply(
        &mut self,
        stamped: StampedOperationEvent,
    ) -> Result<AttemptState, ContractError> {
        if self.history.len() >= MAX_ATTEMPT_EVENTS {
            return Err(ContractError::AttemptEventLimitExceeded);
        }
        if stamped.sequence != self.next_sequence {
            return Err(ContractError::EventSequenceGap {
                expected: self.next_sequence,
                received: stamped.sequence,
            });
        }
        if stamped.observed_at_step < self.last_observed_at_step {
            return Err(ContractError::StepRegression);
        }
        if stamped.work_package_id != self.spec.work_package_id
            || stamped.work_package_digest != self.spec.work_package_digest
        {
            return Err(ContractError::WorkPackageIdentityMismatch);
        }
        if stamped.work_package_revision != self.spec.work_package_revision {
            return Err(ContractError::WorkPackageRevisionMismatch);
        }
        if stamped.observed_at_step > self.spec.expires_at_step
            && !matches!(
                &stamped.event,
                OperationEvent::OutcomeTimedOut
                    | OperationEvent::ReconciledNotExecuted { .. }
                    | OperationEvent::ReconciledExecuted { .. }
                    | OperationEvent::PhysicalDeltaObserved { .. }
                    | OperationEvent::VerificationSubmitted { .. }
                    | OperationEvent::AdditionalEvidenceReceived { .. }
                    | OperationEvent::CloseAccepted
                    | OperationEvent::RecordReworkReview { .. }
            )
        {
            return Err(ContractError::WorkPackageExpired);
        }

        let next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(ContractError::EventSequenceExhausted)?;
        let from = self.state;
        let to = self.transition(&stamped.event)?;
        self.state = to;
        self.last_observed_at_step = stamped.observed_at_step;
        self.next_sequence = next_sequence;
        self.history.push(TransitionRecord {
            sequence: stamped.sequence,
            observed_at_step: stamped.observed_at_step,
            from,
            to,
            event: stamped.event.clone(),
        });
        Ok(to)
    }

    fn transition(&mut self, event: &OperationEvent) -> Result<AttemptState, ContractError> {
        match event {
            OperationEvent::PreflightAccepted {
                work_package_revision,
                design_revision,
                coordinate_frame_revision,
                calibration_revision,
                authority_revision,
                work_package_digest,
                design_digest,
                coordinate_frame_digest,
                calibration_digest,
                authority_evidence_digest,
            } if self.state == AttemptState::Approved => {
                if *work_package_revision != self.spec.work_package_revision
                    || *design_revision != self.spec.design_revision
                    || *coordinate_frame_revision != self.spec.coordinate_frame_revision
                    || *calibration_revision != self.spec.calibration_revision
                {
                    return Err(ContractError::WorkPackageRevisionMismatch);
                }
                if *authority_revision != self.spec.authority_revision
                    || *authority_evidence_digest != self.spec.authority_evidence_digest
                {
                    return Err(ContractError::InvalidAuthorityRevision);
                }
                if *work_package_digest != self.spec.work_package_digest
                    || *design_digest != self.spec.design_digest
                    || *coordinate_frame_digest != self.spec.coordinate_frame_digest
                    || *calibration_digest != self.spec.calibration_digest
                {
                    return Err(ContractError::WorkPackageIdentityMismatch);
                }
                Ok(AttemptState::PreflightAccepted)
            }
            OperationEvent::CommandIssued {
                command_digest,
                idempotency_key,
            } if self.state == AttemptState::PreflightAccepted => {
                if !command_digest.is_present() || !idempotency_key.is_present() {
                    return Err(ContractError::MissingDigest);
                }
                if self.idempotency_key.is_some() {
                    return Err(ContractError::IdempotencyKeyReused);
                }
                self.command_digest = Some(*command_digest);
                self.idempotency_key = Some(*idempotency_key);
                Ok(AttemptState::CommandIssued)
            }
            OperationEvent::DeviceAcknowledged { command_digest }
                if self.state == AttemptState::CommandIssued =>
            {
                if Some(*command_digest) != self.command_digest {
                    return Err(ContractError::CommandDigestMismatch);
                }
                Ok(AttemptState::Acknowledged)
            }
            OperationEvent::ActuatorTelemetryObserved {
                command_digest,
                telemetry_digest,
                calibration_revision,
                calibration_digest,
            } if self.state == AttemptState::Acknowledged => {
                if Some(*command_digest) != self.command_digest {
                    return Err(ContractError::CommandDigestMismatch);
                }
                if !telemetry_digest.is_present() {
                    return Err(ContractError::MissingDigest);
                }
                if !self.seen_evidence.insert(*telemetry_digest) {
                    return Err(ContractError::DuplicateEvidence);
                }
                if *calibration_revision != self.spec.calibration_revision
                    || *calibration_digest != self.spec.calibration_digest
                {
                    return Err(ContractError::CalibrationMismatch);
                }
                Ok(AttemptState::ActuationObserved)
            }
            OperationEvent::OutcomeTimedOut
                if matches!(
                    self.state,
                    AttemptState::CommandIssued
                        | AttemptState::Acknowledged
                        | AttemptState::ActuationObserved
                ) =>
            {
                Ok(AttemptState::OutcomeUnknown)
            }
            OperationEvent::ReconciledNotExecuted {
                reconciler_id,
                evidence_digest,
            } if self.state == AttemptState::OutcomeUnknown => {
                self.validate_reconciliation(*reconciler_id, *evidence_digest)?;
                Ok(AttemptState::ReconciledNotExecuted)
            }
            OperationEvent::ReconciledExecuted {
                reconciler_id,
                evidence_digest,
            } if self.state == AttemptState::OutcomeUnknown => {
                self.validate_reconciliation(*reconciler_id, *evidence_digest)?;
                Ok(AttemptState::ReconciledExecutedAwaitingInspection)
            }
            OperationEvent::PhysicalDeltaObserved { evidence }
                if matches!(
                    self.state,
                    AttemptState::ActuationObserved
                        | AttemptState::ReconciledExecutedAwaitingInspection
                ) =>
            {
                if evidence.source_robot_id != self.spec.executor_id
                    || evidence.sensor_chain_id != self.spec.executor_sensor_chain_id
                {
                    return Err(ContractError::InvalidEvidenceSource);
                }
                if evidence.work_package_digest != self.spec.work_package_digest
                    || evidence.design_revision != self.spec.design_revision
                    || evidence.design_digest != self.spec.design_digest
                {
                    return Err(ContractError::WorkPackageIdentityMismatch);
                }
                if evidence.coordinate_frame_revision != self.spec.coordinate_frame_revision
                    || evidence.coordinate_frame_digest != self.spec.coordinate_frame_digest
                {
                    return Err(ContractError::CoordinateFrameMismatch);
                }
                if evidence.calibration_revision != self.spec.calibration_revision
                    || evidence.calibration_digest != self.spec.calibration_digest {
                    return Err(ContractError::CalibrationMismatch);
                }
                if !evidence.digest.is_present() {
                    return Err(ContractError::MissingDigest);
                }
                if !self.seen_evidence.insert(evidence.digest) {
                    return Err(ContractError::DuplicateEvidence);
                }
                self.physical_delta = Some(evidence.clone());
                Ok(AttemptState::VerificationPending)
            }
            OperationEvent::VerificationSubmitted { receipt }
                if self.state == AttemptState::VerificationPending =>
            {
                if self.seen_evidence.contains(&receipt.digest) {
                    return Err(ContractError::DuplicateEvidence);
                }
                let physical_delta = self.physical_delta.as_ref()
                    .ok_or(ContractError::InvalidEvidenceSource)?;
                let assessment = receipt.assess(&self.spec, physical_delta);
                self.seen_evidence.insert(receipt.digest);
                let next = if assessment.accepted {
                    AttemptState::Accepted
                } else if assessment.rejected {
                    AttemptState::Rejected
                } else {
                    AttemptState::InsufficientEvidence
                };
                self.last_verification = Some(assessment);
                Ok(next)
            }
            OperationEvent::AdditionalEvidenceReceived { evidence_digest }
                if self.state == AttemptState::InsufficientEvidence =>
            {
                if !evidence_digest.is_present() {
                    return Err(ContractError::MissingDigest);
                }
                if !self.seen_evidence.insert(*evidence_digest) {
                    return Err(ContractError::DuplicateEvidence);
                }
                Ok(AttemptState::VerificationPending)
            }
            OperationEvent::CloseAccepted if self.state == AttemptState::Accepted => {
                Ok(AttemptState::Closed)
            }
            OperationEvent::RecordReworkReview {
                authority_revision,
                disposition_digest,
            } if self.state == AttemptState::Rejected => {
                if *authority_revision != self.spec.authority_revision
                    || !disposition_digest.is_present()
                {
                    return Err(ContractError::InvalidAuthorityRevision);
                }
                Ok(AttemptState::ReworkReview)
            }
            OperationEvent::AuthorizeRework {
                authorization,
                approval_digest,
            } if self.state == AttemptState::ReworkReview => {
                if !authorization.validate_against(&self.spec) || !approval_digest.is_present() {
                    return Err(ContractError::InvalidAuthorityRevision);
                }
                Ok(AttemptState::ReworkSuperseded)
            }
            _ => Err(ContractError::InvalidTransition {
                from: self.state,
                event: event.name(),
            }),
        }
    }

    fn validate_reconciliation(
        &mut self,
        reconciler_id: u64,
        evidence_digest: EvidenceDigest,
    ) -> Result<(), ContractError> {
        if reconciler_id == 0 || reconciler_id == self.spec.executor_id {
            return Err(ContractError::ReconcilerMustBeIndependent);
        }
        if !evidence_digest.is_present() {
            return Err(ContractError::MissingDigest);
        }
        if !self.seen_evidence.insert(evidence_digest) {
            return Err(ContractError::DuplicateEvidence);
        }
        Ok(())
    }
}

impl OperationEvent {
    const fn name(&self) -> &'static str {
        match self {
            Self::PreflightAccepted { .. } => "PreflightAccepted",
            Self::CommandIssued { .. } => "CommandIssued",
            Self::DeviceAcknowledged { .. } => "DeviceAcknowledged",
            Self::ActuatorTelemetryObserved { .. } => "ActuatorTelemetryObserved",
            Self::OutcomeTimedOut => "OutcomeTimedOut",
            Self::ReconciledNotExecuted { .. } => "ReconciledNotExecuted",
            Self::ReconciledExecuted { .. } => "ReconciledExecuted",
            Self::PhysicalDeltaObserved { .. } => "PhysicalDeltaObserved",
            Self::VerificationSubmitted { .. } => "VerificationSubmitted",
            Self::AdditionalEvidenceReceived { .. } => "AdditionalEvidenceReceived",
            Self::CloseAccepted => "CloseAccepted",
            Self::RecordReworkReview { .. } => "RecordReworkReview",
            Self::AuthorizeRework { .. } => "AuthorizeRework",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct OperationRegistry {
    attempts: BTreeMap<AttemptId, OperationAttempt>,
    active_by_scope: BTreeMap<WorkScopeId, AttemptId>,
    idempotency_keys: BTreeSet<EvidenceDigest>,
    authorized_rework: BTreeMap<WorkScopeId, ReworkAuthorization>,
}

impl OperationRegistry {
    pub fn begin_attempt(
        &mut self,
        spec: OperationSpec,
        now_step: u64,
    ) -> Result<(), ContractError> {
        if self.attempts.contains_key(&spec.attempt_id) {
            return Err(ContractError::DuplicateAttempt);
        }
        if let Some(holder) = self.active_by_scope.get(&spec.scope_id) {
            return Err(ContractError::ScopeAlreadyLocked {
                scope_id: spec.scope_id,
                holder: *holder,
            });
        }
        if let Some(authorization) = self.authorized_rework.get(&spec.scope_id) {
            if !authorization.matches_spec(&spec) {
                return Err(ContractError::InvalidAuthorityRevision);
            }
        }
        let scope_id = spec.scope_id;
        let attempt_id = spec.attempt_id;
        let attempt = OperationAttempt::new(spec, now_step)?;
        self.attempts.insert(attempt_id, attempt);
        self.active_by_scope.insert(scope_id, attempt_id);
        self.authorized_rework.remove(&scope_id);
        Ok(())
    }

    pub fn append(
        &mut self,
        attempt_id: AttemptId,
        stamped: StampedOperationEvent,
    ) -> Result<AttemptState, ContractError> {
        let command_key = match &stamped.event {
            OperationEvent::CommandIssued { idempotency_key, .. } => Some(*idempotency_key),
            _ => None,
        };
        if command_key.is_some_and(|key| self.idempotency_keys.contains(&key)) {
            return Err(ContractError::IdempotencyKeyReused);
        }
        let rework_authorization = match &stamped.event {
            OperationEvent::AuthorizeRework { authorization, .. } => Some(authorization.clone()),
            _ => None,
        };
        let attempt = self
            .attempts
            .get_mut(&attempt_id)
            .ok_or(ContractError::UnknownAttempt(attempt_id))?;
        let scope_id = attempt.spec.scope_id;
        let state = attempt.apply(stamped)?;
        if let Some(key) = command_key {
            self.idempotency_keys.insert(key);
        }
        if state == AttemptState::ReworkSuperseded {
            if let Some(authorization) = rework_authorization {
                self.authorized_rework.insert(scope_id, authorization);
            }
        }
        if !state.locks_scope()
            && self.active_by_scope.get(&scope_id) == Some(&attempt_id)
        {
            self.active_by_scope.remove(&scope_id);
        }
        Ok(state)
    }

    pub fn attempt(&self, attempt_id: AttemptId) -> Option<&OperationAttempt> {
        self.attempts.get(&attempt_id)
    }

    pub fn scope_lock_holder(&self, scope_id: WorkScopeId) -> Option<AttemptId> {
        self.active_by_scope.get(&scope_id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> EvidenceDigest {
        EvidenceDigest([byte; 32])
    }

    fn spec() -> OperationSpec {
        OperationSpec {
            scope_id: WorkScopeId(7),
            attempt_id: AttemptId(11),
            work_package_id: WorkPackageId(3),
            work_package_revision: 4,
            work_package_digest: digest(20),
            design_revision: 9,
            design_digest: digest(21),
            coordinate_frame_revision: 2,
            coordinate_frame_digest: digest(22),
            calibration_revision: 5,
            calibration_digest: digest(23),
            authority_revision: 6,
            authority_evidence_digest: digest(24),
            executor_id: 100,
            executor_sensor_chain_id: 200,
            executor_common_cause_domains: vec![1, 2],
            required_coverage_per_mille: 950,
            expires_at_step: 100,
        }
    }

    fn event(sequence: u64, step: u64, event: OperationEvent) -> StampedOperationEvent {
        StampedOperationEvent {
            sequence,
            observed_at_step: step,
            work_package_id: WorkPackageId(3),
            work_package_revision: 4,
            work_package_digest: digest(20),
            event,
        }
    }

    fn preflight(registry: &mut OperationRegistry) {
        registry
            .append(
                AttemptId(11),
                event(
                    1,
                    2,
                    OperationEvent::PreflightAccepted {
                        work_package_revision: 4,
                        design_revision: 9,
                        coordinate_frame_revision: 2,
                        calibration_revision: 5,
                        authority_revision: 6,
                        work_package_digest: digest(20),
                        design_digest: digest(21),
                        coordinate_frame_digest: digest(22),
                        calibration_digest: digest(23),
                        authority_evidence_digest: digest(24),
                    },
                ),
            )
            .unwrap();
    }

    fn command_and_actuation(registry: &mut OperationRegistry) {
        preflight(registry);
        registry
            .append(
                AttemptId(11),
                event(
                    2,
                    3,
                    OperationEvent::CommandIssued {
                        command_digest: digest(10),
                        idempotency_key: digest(11),
                    },
                ),
            )
            .unwrap();
        registry
            .append(
                AttemptId(11),
                event(
                    3,
                    4,
                    OperationEvent::DeviceAcknowledged {
                        command_digest: digest(10),
                    },
                ),
            )
            .unwrap();
        registry
            .append(
                AttemptId(11),
                event(
                    4,
                    5,
                    OperationEvent::ActuatorTelemetryObserved {
                        command_digest: digest(10),
                        telemetry_digest: digest(12),
                        calibration_revision: 5,
                        calibration_digest: digest(23),
                    },
                ),
            )
            .unwrap();
    }

    fn physical_delta() -> PhysicalDeltaEvidence {
        PhysicalDeltaEvidence {
            source_robot_id: 100,
            sensor_chain_id: 200,
            work_package_digest: digest(20),
            design_revision: 9,
            design_digest: digest(21),
            coordinate_frame_revision: 2,
            coordinate_frame_digest: digest(22),
            calibration_revision: 5,
            calibration_digest: digest(23),
            coverage_per_mille: 900,
            unobserved_required_regions: 1,
            digest: digest(13),
        }
    }

    fn verification() -> VerificationReceipt {
        VerificationReceipt {
            verifier_id: 300,
            sensor_chain_id: 400,
            work_package_digest: digest(20),
            design_revision: 9,
            design_digest: digest(21),
            coordinate_frame_revision: 2,
            coordinate_frame_digest: digest(22),
            calibration_revision: 5,
            calibration_digest: digest(23),
            coverage_per_mille: 1000,
            unobserved_required_regions: 0,
            common_cause_domains: vec![3, 4],
            common_cause_assessment_digest: digest(19),
            physical_delta_digest: digest(13),
            postconditions_passed: true,
            digest: digest(14),
        }
    }

    fn start_verification(registry: &mut OperationRegistry) {
        command_and_actuation(registry);
        registry
            .append(
                AttemptId(11),
                event(
                    5,
                    6,
                    OperationEvent::PhysicalDeltaObserved {
                        evidence: physical_delta(),
                    },
                ),
            )
            .unwrap();
    }

    #[test]
    fn acknowledgement_and_actuator_telemetry_do_not_close_physical_work() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        command_and_actuation(&mut registry);
        assert_eq!(
            registry.attempt(AttemptId(11)).unwrap().state(),
            AttemptState::ActuationObserved
        );
        assert!(registry.scope_lock_holder(WorkScopeId(7)).is_some());
        assert!(matches!(
            registry.append(
                AttemptId(11),
                event(5, 6, OperationEvent::CloseAccepted)
            ),
            Err(ContractError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn normal_operation_requires_independent_verified_postconditions() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        start_verification(&mut registry);
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(
                    6,
                    7,
                    OperationEvent::VerificationSubmitted {
                        receipt: verification(),
                    },
                ),
            ).unwrap(),
            AttemptState::Accepted
        );
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(7, 8, OperationEvent::CloseAccepted),
            ).unwrap(),
            AttemptState::Closed
        );
        assert_eq!(registry.scope_lock_holder(WorkScopeId(7)), None);
    }

    #[test]
    fn timeout_locks_scope_until_independent_reconciliation() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        preflight(&mut registry);
        registry.append(
            AttemptId(11),
            event(2, 3, OperationEvent::CommandIssued {
                command_digest: digest(10),
                idempotency_key: digest(11),
            }),
        ).unwrap();
        registry.append(
            AttemptId(11),
            event(3, 4, OperationEvent::OutcomeTimedOut),
        ).unwrap();
        assert_eq!(registry.attempt(AttemptId(11)).unwrap().state(), AttemptState::OutcomeUnknown);
        let mut retry = spec();
        retry.attempt_id = AttemptId(12);
        assert!(matches!(
            registry.begin_attempt(retry.clone(), 5),
            Err(ContractError::ScopeAlreadyLocked { .. })
        ));
        registry.append(
            AttemptId(11),
            event(4, 6, OperationEvent::ReconciledNotExecuted {
                reconciler_id: 301,
                evidence_digest: digest(15),
            }),
        ).unwrap();
        assert_eq!(registry.scope_lock_holder(WorkScopeId(7)), None);
        registry.begin_attempt(retry, 7).unwrap();
    }

    #[test]
    fn executor_cannot_reconcile_its_own_ambiguous_command() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        preflight(&mut registry);
        registry.append(
            AttemptId(11),
            event(2, 3, OperationEvent::CommandIssued {
                command_digest: digest(10),
                idempotency_key: digest(11),
            }),
        ).unwrap();
        registry.append(
            AttemptId(11),
            event(3, 4, OperationEvent::OutcomeTimedOut),
        ).unwrap();
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(4, 5, OperationEvent::ReconciledNotExecuted {
                    reconciler_id: 100,
                    evidence_digest: digest(15),
                }),
            ),
            Err(ContractError::ReconcilerMustBeIndependent)
        );
    }

    #[test]
    fn reconciled_execution_requires_physical_observation_before_verification() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        preflight(&mut registry);
        registry.append(
            AttemptId(11),
            event(2, 3, OperationEvent::CommandIssued {
                command_digest: digest(10),
                idempotency_key: digest(11),
            }),
        ).unwrap();
        registry.append(
            AttemptId(11),
            event(3, 4, OperationEvent::OutcomeTimedOut),
        ).unwrap();
        registry.append(
            AttemptId(11),
            event(4, 5, OperationEvent::ReconciledExecuted {
                reconciler_id: 301,
                evidence_digest: digest(15),
            }),
        ).unwrap();
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(5, 6, OperationEvent::VerificationSubmitted {
                    receipt: verification(),
                }),
            ),
            Err(ContractError::InvalidTransition {
                from: AttemptState::ReconciledExecutedAwaitingInspection,
                event: "VerificationSubmitted",
            })
        );
    }

    #[test]
    fn insufficient_coverage_is_not_accepted_and_can_be_reinspected() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        start_verification(&mut registry);
        let mut poor = verification();
        poor.coverage_per_mille = 800;
        poor.unobserved_required_regions = 2;
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(6, 7, OperationEvent::VerificationSubmitted { receipt: poor }),
            ).unwrap(),
            AttemptState::InsufficientEvidence
        );
        assert!(registry.attempt(AttemptId(11)).unwrap().last_verification().unwrap().issues.contains(&VerificationIssue::CoverageInsufficient));
        registry.append(
            AttemptId(11),
            event(7, 8, OperationEvent::AdditionalEvidenceReceived {
                evidence_digest: digest(16),
            }),
        ).unwrap();
        let mut improved = verification();
        improved.digest = digest(18);
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(8, 9, OperationEvent::VerificationSubmitted {
                    receipt: improved,
                }),
            ).unwrap(),
            AttemptState::Accepted
        );
    }

    #[test]
    fn common_cause_overlap_prevents_independent_acceptance() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        start_verification(&mut registry);
        let mut receipt = verification();
        receipt.common_cause_domains = vec![2, 9];
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(6, 7, OperationEvent::VerificationSubmitted { receipt }),
            ).unwrap(),
            AttemptState::InsufficientEvidence
        );
        assert!(registry.attempt(AttemptId(11)).unwrap().last_verification().unwrap().issues.contains(&VerificationIssue::CommonCauseOverlap));
    }

    #[test]
    fn verifier_must_not_reuse_executor_or_its_sensor_chain() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        start_verification(&mut registry);
        let mut receipt = verification();
        receipt.verifier_id = 100;
        receipt.sensor_chain_id = 200;
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(6, 7, OperationEvent::VerificationSubmitted { receipt }),
            ).unwrap(),
            AttemptState::InsufficientEvidence
        );
        let assessment = registry.attempt(AttemptId(11)).unwrap().last_verification().unwrap();
        assert!(assessment.issues.contains(&VerificationIssue::VerifierIsExecutor));
        assert!(assessment.issues.contains(&VerificationIssue::SensorChainShared));
    }

    #[test]
    fn stale_preflight_and_frame_or_calibration_mismatches_fail_closed() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(1, 2, OperationEvent::PreflightAccepted {
                    work_package_revision: 3,
                    design_revision: 9,
                    coordinate_frame_revision: 2,
                    calibration_revision: 5,
                    authority_revision: 6,
                    work_package_digest: digest(20),
                    design_digest: digest(21),
                    coordinate_frame_digest: digest(22),
                    calibration_digest: digest(23),
                    authority_evidence_digest: digest(24),
                }),
            ),
            Err(ContractError::WorkPackageRevisionMismatch)
        );
        assert_eq!(registry.attempt(AttemptId(11)).unwrap().state(), AttemptState::Approved);
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(1, 2, OperationEvent::PreflightAccepted {
                    work_package_revision: 4,
                    design_revision: 9,
                    coordinate_frame_revision: 2,
                    calibration_revision: 5,
                    authority_revision: 6,
                    work_package_digest: digest(20),
                    design_digest: digest(21),
                    coordinate_frame_digest: digest(22),
                    calibration_digest: digest(23),
                    authority_evidence_digest: digest(24),
                }),
            ).unwrap(),
            AttemptState::PreflightAccepted
        );
    }

    #[test]
    fn duplicate_or_out_of_order_events_do_not_mutate_the_attempt() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        preflight(&mut registry);
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(3, 4, OperationEvent::CommandIssued {
                    command_digest: digest(10),
                    idempotency_key: digest(11),
                }),
            ),
            Err(ContractError::EventSequenceGap { expected: 2, received: 3 })
        );
        assert_eq!(registry.attempt(AttemptId(11)).unwrap().state(), AttemptState::PreflightAccepted);
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(2, 3, OperationEvent::CommandIssued {
                    command_digest: digest(10),
                    idempotency_key: digest(11),
                }),
            ).unwrap(),
            AttemptState::CommandIssued
        );
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(3, 2, OperationEvent::DeviceAcknowledged {
                    command_digest: digest(10),
                }),
            ),
            Err(ContractError::StepRegression)
        );
    }

    #[test]
    fn failed_postcondition_requires_rework_review_and_keeps_scope_locked() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        start_verification(&mut registry);
        let mut receipt = verification();
        receipt.postconditions_passed = false;
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(6, 7, OperationEvent::VerificationSubmitted { receipt }),
            ).unwrap(),
            AttemptState::Rejected
        );
        registry.append(
            AttemptId(11),
            event(7, 8, OperationEvent::RecordReworkReview {
                authority_revision: 6,
                disposition_digest: digest(17),
            }),
        ).unwrap();
        assert_eq!(registry.attempt(AttemptId(11)).unwrap().state(), AttemptState::ReworkReview);
        assert_eq!(registry.scope_lock_holder(WorkScopeId(7)), Some(AttemptId(11)));
    }
    #[test]
    fn idempotency_keys_cannot_be_reused_across_attempts() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        preflight(&mut registry);
        registry.append(
            AttemptId(11),
            event(2, 3, OperationEvent::CommandIssued {
                command_digest: digest(10),
                idempotency_key: digest(11),
            }),
        ).unwrap();
        registry.append(
            AttemptId(11),
            event(3, 4, OperationEvent::OutcomeTimedOut),
        ).unwrap();
        registry.append(
            AttemptId(11),
            event(4, 5, OperationEvent::ReconciledNotExecuted {
                reconciler_id: 301,
                evidence_digest: digest(15),
            }),
        ).unwrap();

        let mut retry = spec();
        retry.attempt_id = AttemptId(12);
        registry.begin_attempt(retry, 6).unwrap();
        registry.append(
            AttemptId(12),
            event(1, 7, OperationEvent::PreflightAccepted {
                work_package_revision: 4,
                design_revision: 9,
                coordinate_frame_revision: 2,
                calibration_revision: 5,
                authority_revision: 6,
                work_package_digest: digest(20),
                design_digest: digest(21),
                coordinate_frame_digest: digest(22),
                calibration_digest: digest(23),
                authority_evidence_digest: digest(24),
            }),
        ).unwrap();
        assert_eq!(
            registry.append(
                AttemptId(12),
                event(2, 8, OperationEvent::CommandIssued {
                    command_digest: digest(30),
                    idempotency_key: digest(11),
                }),
            ),
            Err(ContractError::IdempotencyKeyReused)
        );
        assert_eq!(
            registry.attempt(AttemptId(12)).unwrap().state(),
            AttemptState::PreflightAccepted
        );
    }

    #[test]
    fn rejected_work_scope_requires_new_authorized_package_before_retry() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        start_verification(&mut registry);
        let mut failed = verification();
        failed.postconditions_passed = false;
        registry.append(
            AttemptId(11),
            event(6, 7, OperationEvent::VerificationSubmitted { receipt: failed }),
        ).unwrap();
        registry.append(
            AttemptId(11),
            event(7, 8, OperationEvent::RecordReworkReview {
                authority_revision: 6,
                disposition_digest: digest(31),
            }),
        ).unwrap();

        let mut premature_retry = spec();
        premature_retry.attempt_id = AttemptId(12);
        assert!(matches!(
            registry.begin_attempt(premature_retry, 9),
            Err(ContractError::ScopeAlreadyLocked { .. })
        ));

        let authorization = ReworkAuthorization {
            work_package_id: WorkPackageId(4),
            work_package_revision: 5,
            work_package_digest: digest(32),
            design_revision: 10,
            design_digest: digest(33),
            coordinate_frame_revision: 2,
            coordinate_frame_digest: digest(22),
            calibration_revision: 5,
            calibration_digest: digest(23),
            authority_revision: 7,
            authority_evidence_digest: digest(34),
        };
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(8, 10, OperationEvent::AuthorizeRework {
                    authorization: authorization.clone(),
                    approval_digest: digest(35),
                }),
            ).unwrap(),
            AttemptState::ReworkSuperseded
        );
        assert_eq!(registry.scope_lock_holder(WorkScopeId(7)), None);

        let mut wrong_retry = spec();
        wrong_retry.attempt_id = AttemptId(12);
        wrong_retry.work_package_revision = 5;
        assert_eq!(
            registry.begin_attempt(wrong_retry, 11),
            Err(ContractError::InvalidAuthorityRevision)
        );

        let mut authorized_retry = spec();
        authorized_retry.attempt_id = AttemptId(13);
        authorized_retry.work_package_id = WorkPackageId(4);
        authorized_retry.work_package_revision = 5;
        authorized_retry.work_package_digest = digest(32);
        authorized_retry.design_revision = 10;
        authorized_retry.design_digest = digest(33);
        authorized_retry.authority_revision = 7;
        authorized_retry.authority_evidence_digest = digest(34);
        registry.begin_attempt(authorized_retry, 11).unwrap();
        assert_eq!(
            registry.scope_lock_holder(WorkScopeId(7)),
            Some(AttemptId(13))
        );
    }

    #[test]
    fn transition_history_retains_the_typed_command_and_evidence_events() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        preflight(&mut registry);
        registry.append(
            AttemptId(11),
            event(2, 3, OperationEvent::CommandIssued {
                command_digest: digest(10),
                idempotency_key: digest(11),
            }),
        ).unwrap();
        let history = registry.attempt(AttemptId(11)).unwrap().history();
        assert_eq!(history.len(), 2);
        assert!(matches!(
            &history[1].event,
            OperationEvent::CommandIssued {
                command_digest: EvidenceDigest([10; 32]),
                idempotency_key: EvidenceDigest([11; 32]),
            }
        ));
    }

    #[test]
    fn verifier_must_bind_its_receipt_to_the_observed_physical_delta() {
        let mut registry = OperationRegistry::default();
        registry.begin_attempt(spec(), 1).unwrap();
        start_verification(&mut registry);
        let mut receipt = verification();
        receipt.physical_delta_digest = digest(99);
        assert_eq!(
            registry.append(
                AttemptId(11),
                event(6, 7, OperationEvent::VerificationSubmitted { receipt }),
            ).unwrap(),
            AttemptState::InsufficientEvidence
        );
        assert!(registry.attempt(AttemptId(11)).unwrap()
            .last_verification().unwrap().issues.contains(&VerificationIssue::PhysicalDeltaMismatch));
    }

}
