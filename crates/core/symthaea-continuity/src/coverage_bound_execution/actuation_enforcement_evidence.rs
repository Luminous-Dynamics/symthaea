// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Closed-world evidence contract for backend/resource actuation enforcement.
//!
//! This module still grants no physical authority. It answers a narrower question:
//! did an adapter provide one exact, context-bound evidence record for every V1
//! enforcement obligation, using the required evidence basis, with no UNKNOWN/failed
//! result or omitted obligation?
//!
//! Structural completeness is not truth and is not verifier qualification:
//!
//! `DeclaredEvidence != StructurallyCompleteEvidence != VerifiedEnforcement`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::actuation_interlock::{
    ActuationEnforcementBoundaryProfileId, ActuationEnforcementBoundaryProfileV1,
};
use super::actuation_interlock_auth::{
    ActuationInterlockAuthenticationProfileId, ActuationInterlockAuthenticationProfileV1,
};
use crate::execution_capability::ExecutionBackendId;

pub const ACTUATION_ENFORCEMENT_EVIDENCE_RECORD_SCHEMA_V1: &str =
    "symthaea-continuity-actuation-enforcement-evidence-record-v1";
pub const ACTUATION_ENFORCEMENT_EVIDENCE_SET_SCHEMA_V1: &str =
    "symthaea-continuity-actuation-enforcement-evidence-set-v1";

const RECORD_DOMAIN: &[u8] =
    b"symthaea.continuity.actuation-enforcement-evidence-record.v1\0";
const COMPLETE_SET_DOMAIN: &[u8] =
    b"symthaea.continuity.structurally-complete-actuation-enforcement-evidence.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name {
            pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
        }
    };
}

digest_id!(ActuationEnforcementEvidenceRecordId);
digest_id!(StructurallyCompleteActuationEnforcementEvidenceId);

/// Fixed V1 enforcement obligations. Callers cannot supply a smaller obligation set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActuationEnforcementObligationV1 {
    BoundaryIdentity,
    SameBoundaryChecksAndMutates,
    DurableMonotonicFence,
    RejectStaleGeneration,
    RejectReplay,
    RejectDenyDisposition,
    EmergencyStopDominates,
    OneUsePermitConsumption,
    CrashRecoveryPreservesFence,
}

impl ActuationEnforcementObligationV1 {
    fn tag(self) -> u8 {
        match self {
            Self::BoundaryIdentity => 1,
            Self::SameBoundaryChecksAndMutates => 2,
            Self::DurableMonotonicFence => 3,
            Self::RejectStaleGeneration => 4,
            Self::RejectReplay => 5,
            Self::RejectDenyDisposition => 6,
            Self::EmergencyStopDominates => 7,
            Self::OneUsePermitConsumption => 8,
            Self::CrashRecoveryPreservesFence => 9,
        }
    }
}

pub const REQUIRED_ACTUATION_ENFORCEMENT_OBLIGATIONS_V1:
    [ActuationEnforcementObligationV1; 9] = [
    ActuationEnforcementObligationV1::BoundaryIdentity,
    ActuationEnforcementObligationV1::SameBoundaryChecksAndMutates,
    ActuationEnforcementObligationV1::DurableMonotonicFence,
    ActuationEnforcementObligationV1::RejectStaleGeneration,
    ActuationEnforcementObligationV1::RejectReplay,
    ActuationEnforcementObligationV1::RejectDenyDisposition,
    ActuationEnforcementObligationV1::EmergencyStopDominates,
    ActuationEnforcementObligationV1::OneUsePermitConsumption,
    ActuationEnforcementObligationV1::CrashRecoveryPreservesFence,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActuationEnforcementEvidenceBasisV1 {
    StaticImplementationInspection,
    AtomicCheckAndActuateScenario,
    CrashRestartScenario,
    StaleHolderScenario,
    ReplayScenario,
    DenyScenario,
    EmergencyStopScenario,
    OneUseConsumptionScenario,
}

impl ActuationEnforcementEvidenceBasisV1 {
    fn tag(self) -> u8 {
        match self {
            Self::StaticImplementationInspection => 1,
            Self::AtomicCheckAndActuateScenario => 2,
            Self::CrashRestartScenario => 3,
            Self::StaleHolderScenario => 4,
            Self::ReplayScenario => 5,
            Self::DenyScenario => 6,
            Self::EmergencyStopScenario => 7,
            Self::OneUseConsumptionScenario => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActuationEnforcementEvidenceOutcomeV1 {
    Satisfied,
    Failed,
    Unknown,
}

impl ActuationEnforcementEvidenceOutcomeV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Satisfied => 1,
            Self::Failed => 2,
            Self::Unknown => 3,
        }
    }
}

/// One descriptive evidence record. This is not trusted merely because a caller can
/// construct it; future verifier-owned admission must establish evidence truth.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationEnforcementEvidenceRecordV1 {
    schema_version: String,
    enforcement_profile_id: ActuationEnforcementBoundaryProfileId,
    authentication_profile_id: ActuationInterlockAuthenticationProfileId,
    backend_id: ExecutionBackendId,
    obligation: ActuationEnforcementObligationV1,
    basis: ActuationEnforcementEvidenceBasisV1,
    scenario_digest: [u8; 32],
    outcome: ActuationEnforcementEvidenceOutcomeV1,
    observed_at_unix_ms: u64,
    raw_evidence_digest: [u8; 32],
    record_id: ActuationEnforcementEvidenceRecordId,
}

impl ActuationEnforcementEvidenceRecordV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        enforcement: &ActuationEnforcementBoundaryProfileV1,
        authentication: &ActuationInterlockAuthenticationProfileV1,
        obligation: ActuationEnforcementObligationV1,
        basis: ActuationEnforcementEvidenceBasisV1,
        scenario_digest: [u8; 32],
        outcome: ActuationEnforcementEvidenceOutcomeV1,
        observed_at_unix_ms: u64,
        raw_evidence_digest: [u8; 32],
    ) -> Result<Self, ActuationEnforcementEvidenceError> {
        enforcement.validate()?;
        authentication.validate()?;
        require_profiles_match(enforcement, authentication)?;
        if scenario_digest == [0; 32] || raw_evidence_digest == [0; 32] {
            return Err(ActuationEnforcementEvidenceError::ZeroDigest);
        }
        if observed_at_unix_ms == 0 {
            return Err(ActuationEnforcementEvidenceError::ZeroObservationTime);
        }
        let expected_basis = required_basis(obligation);
        if basis != expected_basis {
            return Err(ActuationEnforcementEvidenceError::WrongEvidenceBasis {
                obligation,
                expected: expected_basis,
                observed: basis,
            });
        }
        let record_id = ActuationEnforcementEvidenceRecordId(hash_record(
            enforcement.id(),
            authentication.id(),
            enforcement.backend_id(),
            obligation,
            basis,
            scenario_digest,
            outcome,
            observed_at_unix_ms,
            raw_evidence_digest,
        ));
        Ok(Self {
            schema_version: ACTUATION_ENFORCEMENT_EVIDENCE_RECORD_SCHEMA_V1.to_owned(),
            enforcement_profile_id: enforcement.id(),
            authentication_profile_id: authentication.id(),
            backend_id: enforcement.backend_id(),
            obligation,
            basis,
            scenario_digest,
            outcome,
            observed_at_unix_ms,
            raw_evidence_digest,
            record_id,
        })
    }

    pub fn validate(&self) -> Result<(), ActuationEnforcementEvidenceError> {
        if self.schema_version != ACTUATION_ENFORCEMENT_EVIDENCE_RECORD_SCHEMA_V1 {
            return Err(ActuationEnforcementEvidenceError::UnsupportedRecordSchema(
                self.schema_version.clone(),
            ));
        }
        if self.scenario_digest == [0; 32] || self.raw_evidence_digest == [0; 32] {
            return Err(ActuationEnforcementEvidenceError::ZeroDigest);
        }
        if self.observed_at_unix_ms == 0 {
            return Err(ActuationEnforcementEvidenceError::ZeroObservationTime);
        }
        let expected_basis = required_basis(self.obligation);
        if self.basis != expected_basis {
            return Err(ActuationEnforcementEvidenceError::WrongEvidenceBasis {
                obligation: self.obligation,
                expected: expected_basis,
                observed: self.basis,
            });
        }
        let expected = ActuationEnforcementEvidenceRecordId(hash_record(
            self.enforcement_profile_id,
            self.authentication_profile_id,
            self.backend_id,
            self.obligation,
            self.basis,
            self.scenario_digest,
            self.outcome,
            self.observed_at_unix_ms,
            self.raw_evidence_digest,
        ));
        if expected != self.record_id {
            return Err(ActuationEnforcementEvidenceError::RecordIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> ActuationEnforcementEvidenceRecordId { self.record_id }
    pub fn enforcement_profile_id(&self) -> ActuationEnforcementBoundaryProfileId {
        self.enforcement_profile_id
    }
    pub fn authentication_profile_id(&self) -> ActuationInterlockAuthenticationProfileId {
        self.authentication_profile_id
    }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn obligation(&self) -> ActuationEnforcementObligationV1 { self.obligation }
    pub fn basis(&self) -> ActuationEnforcementEvidenceBasisV1 { self.basis }
    pub fn outcome(&self) -> ActuationEnforcementEvidenceOutcomeV1 { self.outcome }
    pub fn observed_at_unix_ms(&self) -> u64 { self.observed_at_unix_ms }
}

/// Closed-world structural evidence bundle. This proves only that every fixed V1
/// obligation has exactly one context-matching Satisfied record with its required
/// basis. It does not establish that those records are truthful.
#[derive(Debug, Clone)]
pub struct StructurallyCompleteActuationEnforcementEvidenceV1 {
    complete_id: StructurallyCompleteActuationEnforcementEvidenceId,
    enforcement_profile_id: ActuationEnforcementBoundaryProfileId,
    authentication_profile_id: ActuationInterlockAuthenticationProfileId,
    backend_id: ExecutionBackendId,
    records: Vec<ActuationEnforcementEvidenceRecordV1>,
}

impl StructurallyCompleteActuationEnforcementEvidenceV1 {
    pub fn assemble(
        enforcement: &ActuationEnforcementBoundaryProfileV1,
        authentication: &ActuationInterlockAuthenticationProfileV1,
        records: Vec<ActuationEnforcementEvidenceRecordV1>,
    ) -> Result<Self, ActuationEnforcementEvidenceError> {
        enforcement.validate()?;
        authentication.validate()?;
        require_profiles_match(enforcement, authentication)?;
        if records.len() != REQUIRED_ACTUATION_ENFORCEMENT_OBLIGATIONS_V1.len() {
            return Err(ActuationEnforcementEvidenceError::WrongRecordCount {
                expected: REQUIRED_ACTUATION_ENFORCEMENT_OBLIGATIONS_V1.len(),
                observed: records.len(),
            });
        }

        let mut by_obligation = BTreeMap::new();
        for record in records {
            record.validate()?;
            if record.enforcement_profile_id() != enforcement.id()
                || record.authentication_profile_id() != authentication.id()
                || record.backend_id() != enforcement.backend_id()
            {
                return Err(ActuationEnforcementEvidenceError::EvidenceContextMismatch);
            }
            if record.outcome() != ActuationEnforcementEvidenceOutcomeV1::Satisfied {
                return Err(ActuationEnforcementEvidenceError::NonQualifyingOutcome {
                    obligation: record.obligation(),
                    outcome: record.outcome(),
                });
            }
            if by_obligation.insert(record.obligation(), record).is_some() {
                return Err(ActuationEnforcementEvidenceError::DuplicateObligation);
            }
        }

        let mut ordered = Vec::with_capacity(REQUIRED_ACTUATION_ENFORCEMENT_OBLIGATIONS_V1.len());
        for obligation in REQUIRED_ACTUATION_ENFORCEMENT_OBLIGATIONS_V1 {
            let record = by_obligation
                .remove(&obligation)
                .ok_or(ActuationEnforcementEvidenceError::MissingObligation(obligation))?;
            ordered.push(record);
        }
        if !by_obligation.is_empty() {
            return Err(ActuationEnforcementEvidenceError::UnexpectedObligation);
        }

        let complete_id = StructurallyCompleteActuationEnforcementEvidenceId(hash_complete_set(
            enforcement.id(),
            authentication.id(),
            enforcement.backend_id(),
            &ordered,
        ));
        Ok(Self {
            complete_id,
            enforcement_profile_id: enforcement.id(),
            authentication_profile_id: authentication.id(),
            backend_id: enforcement.backend_id(),
            records: ordered,
        })
    }

    pub fn id(&self) -> StructurallyCompleteActuationEnforcementEvidenceId { self.complete_id }
    pub fn enforcement_profile_id(&self) -> ActuationEnforcementBoundaryProfileId {
        self.enforcement_profile_id
    }
    pub fn authentication_profile_id(&self) -> ActuationInterlockAuthenticationProfileId {
        self.authentication_profile_id
    }
    pub fn backend_id(&self) -> ExecutionBackendId { self.backend_id }
    pub fn records(&self) -> &[ActuationEnforcementEvidenceRecordV1] { &self.records }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ActuationEnforcementEvidenceError {
    #[error(transparent)]
    Interlock(#[from] super::actuation_interlock::ActuationInterlockError),
    #[error(transparent)]
    Authentication(#[from] super::actuation_interlock_auth::ActuationInterlockAuthenticationError),
    #[error("unsupported actuation enforcement evidence record schema: {0}")]
    UnsupportedRecordSchema(String),
    #[error("actuation enforcement evidence digest must be non-zero")]
    ZeroDigest,
    #[error("actuation enforcement evidence observation time must be non-zero")]
    ZeroObservationTime,
    #[error("actuation enforcement/authentication profiles do not bind the same backend boundary")]
    ProfileContextMismatch,
    #[error("actuation enforcement evidence record context mismatch")]
    EvidenceContextMismatch,
    #[error("actuation enforcement evidence record identity mismatch")]
    RecordIdentityMismatch,
    #[error("wrong evidence basis for {obligation:?}: expected {expected:?}, observed {observed:?}")]
    WrongEvidenceBasis {
        obligation: ActuationEnforcementObligationV1,
        expected: ActuationEnforcementEvidenceBasisV1,
        observed: ActuationEnforcementEvidenceBasisV1,
    },
    #[error("wrong actuation enforcement evidence record count: expected {expected}, observed {observed}")]
    WrongRecordCount { expected: usize, observed: usize },
    #[error("duplicate actuation enforcement obligation")]
    DuplicateObligation,
    #[error("missing actuation enforcement obligation: {0:?}")]
    MissingObligation(ActuationEnforcementObligationV1),
    #[error("unexpected actuation enforcement obligation remained after closed-world assembly")]
    UnexpectedObligation,
    #[error("actuation enforcement obligation {obligation:?} has non-qualifying outcome {outcome:?}")]
    NonQualifyingOutcome {
        obligation: ActuationEnforcementObligationV1,
        outcome: ActuationEnforcementEvidenceOutcomeV1,
    },
}

fn require_profiles_match(
    enforcement: &ActuationEnforcementBoundaryProfileV1,
    authentication: &ActuationInterlockAuthenticationProfileV1,
) -> Result<(), ActuationEnforcementEvidenceError> {
    if authentication.enforcement_profile_id() != enforcement.id()
        || authentication.backend_id() != enforcement.backend_id()
    {
        return Err(ActuationEnforcementEvidenceError::ProfileContextMismatch);
    }
    Ok(())
}

fn required_basis(
    obligation: ActuationEnforcementObligationV1,
) -> ActuationEnforcementEvidenceBasisV1 {
    match obligation {
        ActuationEnforcementObligationV1::BoundaryIdentity => {
            ActuationEnforcementEvidenceBasisV1::StaticImplementationInspection
        }
        ActuationEnforcementObligationV1::SameBoundaryChecksAndMutates => {
            ActuationEnforcementEvidenceBasisV1::AtomicCheckAndActuateScenario
        }
        ActuationEnforcementObligationV1::DurableMonotonicFence => {
            ActuationEnforcementEvidenceBasisV1::CrashRestartScenario
        }
        ActuationEnforcementObligationV1::RejectStaleGeneration => {
            ActuationEnforcementEvidenceBasisV1::StaleHolderScenario
        }
        ActuationEnforcementObligationV1::RejectReplay => {
            ActuationEnforcementEvidenceBasisV1::ReplayScenario
        }
        ActuationEnforcementObligationV1::RejectDenyDisposition => {
            ActuationEnforcementEvidenceBasisV1::DenyScenario
        }
        ActuationEnforcementObligationV1::EmergencyStopDominates => {
            ActuationEnforcementEvidenceBasisV1::EmergencyStopScenario
        }
        ActuationEnforcementObligationV1::OneUsePermitConsumption => {
            ActuationEnforcementEvidenceBasisV1::OneUseConsumptionScenario
        }
        ActuationEnforcementObligationV1::CrashRecoveryPreservesFence => {
            ActuationEnforcementEvidenceBasisV1::CrashRestartScenario
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn hash_record(
    enforcement_profile_id: ActuationEnforcementBoundaryProfileId,
    authentication_profile_id: ActuationInterlockAuthenticationProfileId,
    backend_id: ExecutionBackendId,
    obligation: ActuationEnforcementObligationV1,
    basis: ActuationEnforcementEvidenceBasisV1,
    scenario_digest: [u8; 32],
    outcome: ActuationEnforcementEvidenceOutcomeV1,
    observed_at_unix_ms: u64,
    raw_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(RECORD_DOMAIN);
    h.update(enforcement_profile_id.as_bytes());
    h.update(authentication_profile_id.as_bytes());
    h.update(backend_id.as_bytes());
    h.update(&[obligation.tag()]);
    h.update(&[basis.tag()]);
    h.update(&scenario_digest);
    h.update(&[outcome.tag()]);
    h.update(&observed_at_unix_ms.to_le_bytes());
    h.update(&raw_evidence_digest);
    *h.finalize().as_bytes()
}

fn hash_complete_set(
    enforcement_profile_id: ActuationEnforcementBoundaryProfileId,
    authentication_profile_id: ActuationInterlockAuthenticationProfileId,
    backend_id: ExecutionBackendId,
    records: &[ActuationEnforcementEvidenceRecordV1],
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(COMPLETE_SET_DOMAIN);
    h.update(enforcement_profile_id.as_bytes());
    h.update(authentication_profile_id.as_bytes());
    h.update(backend_id.as_bytes());
    h.update(&(records.len() as u64).to_le_bytes());
    for record in records {
        h.update(record.id().as_bytes());
    }
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_obligation_set_is_closed_and_stable() {
        assert_eq!(REQUIRED_ACTUATION_ENFORCEMENT_OBLIGATIONS_V1.len(), 9);
        assert_ne!(RECORD_DOMAIN, COMPLETE_SET_DOMAIN);
    }

    #[test]
    fn every_v1_obligation_has_a_required_basis() {
        for obligation in REQUIRED_ACTUATION_ENFORCEMENT_OBLIGATIONS_V1 {
            let _ = required_basis(obligation);
        }
    }
}