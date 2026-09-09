// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Replayable evidence for one verifier-profile adoption commit transaction.
//!
//! The compact adoption head is sufficient for lineage/CAS but intentionally too
//! small to preserve the policy provenance that led to a write. This module stores
//! that provenance as a canonical, serializable evidence record built only from the
//! #1198 grant-bound trusted-time commit capsule.
//!
//! The record is deliberately **not** a committed receipt and is never authority by
//! deserialization. It proves only intrinsic/canonical consistency of historical
//! evidence. A future persistence adapter must still require exact Xenia proof plus
//! a successful atomic registry CAS before asserting that this record was committed.
//!
//! Core theorem:
//!
//! `commit evidence != committed receipt != current verifier authority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::profile_adoption::{
    VerifierAdoptionScopeV1, VerifierProfileAdoptionError, VerifierProfileAdoptionPredecessorV1,
    VerifierProfileAdoptionTransitionV1,
};
use crate::profile_adoption_admission::VerifierProfileAdoptionHeadV1;
use crate::profile_adoption_grant::{
    VerifierAdoptionAuthorityGrantError, VerifierAdoptionAuthorityGrantV1,
};
use crate::profile_adoption_grant_commit::GrantBoundVerifierProfileAdoptionCommitPreconditionsV1;
use crate::profile_adoption_root::{
    VerifierProfileAdoptionAuthorityRootSnapshotV1, VerifierProfileAdoptionRootBindingError,
};
use crate::witness::EvidenceClass;

pub const VERIFIER_PROFILE_ADOPTION_COMMIT_EVIDENCE_SCHEMA_V1: &str =
    "symthaea-continuity-verifier-profile-adoption-commit-evidence-v1";
const RECORD_DOMAIN: &[u8] =
    b"symthaea.continuity.verifier-profile-adoption.commit-evidence.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionCommitEvidenceIdV1([u8; 32]);

impl VerifierProfileAdoptionCommitEvidenceIdV1 {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Serializable historical projection of one exact locally provisioned adoption
/// root snapshot. Reading this value does not make the root trusted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionRootEvidenceV1 {
    authority_subject: String,
    authority_root_id: String,
    authority_root_digest: [u8; 32],
    provisioning_epoch: u64,
    snapshot_id: [u8; 32],
}

impl VerifierProfileAdoptionRootEvidenceV1 {
    fn from_snapshot(snapshot: &VerifierProfileAdoptionAuthorityRootSnapshotV1) -> Self {
        Self {
            authority_subject: snapshot.authority_subject().to_owned(),
            authority_root_id: snapshot.authority_root_id().to_owned(),
            authority_root_digest: snapshot.authority_root_digest(),
            provisioning_epoch: snapshot.provisioning_epoch(),
            snapshot_id: *snapshot.id().as_bytes(),
        }
    }

    fn reconstruct(
        &self,
    ) -> Result<VerifierProfileAdoptionAuthorityRootSnapshotV1, VerifierProfileAdoptionCommitEvidenceError>
    {
        let snapshot = VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            self.authority_subject.clone(),
            self.authority_root_id.clone(),
            self.authority_root_digest,
            self.provisioning_epoch,
        )?;
        if snapshot.id().as_bytes() != &self.snapshot_id {
            return Err(VerifierProfileAdoptionCommitEvidenceError::RootSnapshotIdentityMismatch);
        }
        Ok(snapshot)
    }

    pub fn authority_subject(&self) -> &str {
        &self.authority_subject
    }
    pub fn authority_root_id(&self) -> &str {
        &self.authority_root_id
    }
    pub fn authority_root_digest(&self) -> [u8; 32] {
        self.authority_root_digest
    }
    pub fn provisioning_epoch(&self) -> u64 {
        self.provisioning_epoch
    }
    pub fn snapshot_id(&self) -> &[u8; 32] {
        &self.snapshot_id
    }
}

/// Serializable historical projection of the exact local capability envelope used
/// for the adoption. It is evidence about policy, not live policy after restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierAdoptionAuthorityGrantEvidenceV1 {
    grant_slot_id: String,
    verifier_role_id: String,
    grant_epoch: u64,
    maximum_evidence_class: EvidenceClass,
    allowed_scope: VerifierAdoptionScopeV1,
    grant_id: [u8; 32],
}

impl VerifierAdoptionAuthorityGrantEvidenceV1 {
    fn from_grant(grant: &VerifierAdoptionAuthorityGrantV1) -> Self {
        Self {
            grant_slot_id: grant.grant_slot_id().to_owned(),
            verifier_role_id: grant.verifier_role_id().to_owned(),
            grant_epoch: grant.grant_epoch(),
            maximum_evidence_class: grant.maximum_evidence_class(),
            allowed_scope: grant.allowed_scope().clone(),
            grant_id: *grant.id().as_bytes(),
        }
    }

    fn reconstruct(
        &self,
        root: VerifierProfileAdoptionAuthorityRootSnapshotV1,
    ) -> Result<VerifierAdoptionAuthorityGrantV1, VerifierProfileAdoptionCommitEvidenceError> {
        let grant = VerifierAdoptionAuthorityGrantV1::new(
            self.grant_slot_id.clone(),
            root,
            self.verifier_role_id.clone(),
            self.grant_epoch,
            self.maximum_evidence_class,
            self.allowed_scope.clone(),
        )?;
        if grant.id().as_bytes() != &self.grant_id {
            return Err(VerifierProfileAdoptionCommitEvidenceError::GrantIdentityMismatch);
        }
        Ok(grant)
    }

    pub fn grant_slot_id(&self) -> &str {
        &self.grant_slot_id
    }
    pub fn verifier_role_id(&self) -> &str {
        &self.verifier_role_id
    }
    pub fn grant_epoch(&self) -> u64 {
        self.grant_epoch
    }
    pub fn maximum_evidence_class(&self) -> EvidenceClass {
        self.maximum_evidence_class
    }
    pub fn allowed_scope(&self) -> &VerifierAdoptionScopeV1 {
        &self.allowed_scope
    }
    pub fn grant_id(&self) -> &[u8; 32] {
        &self.grant_id
    }
}

/// Historical clock-policy lineage captured by #1149/#1198.
///
/// The observation ID is retained for provenance, but the record does not claim that
/// this clock observation remains current. Runtime use must obtain a fresh local
/// observation and re-run #1204.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionClockPolicyEvidenceV1 {
    source_id: String,
    source_epoch: u64,
    checked_observation_id: [u8; 32],
    max_uncertainty_ms: u64,
}

impl VerifierProfileAdoptionClockPolicyEvidenceV1 {
    fn validate(&self) -> Result<(), VerifierProfileAdoptionCommitEvidenceError> {
        checked_text("clock_source_id", &self.source_id)?;
        if self.source_epoch == 0 {
            return Err(VerifierProfileAdoptionCommitEvidenceError::ZeroClockEpoch);
        }
        if self.checked_observation_id == [0; 32] {
            return Err(VerifierProfileAdoptionCommitEvidenceError::ZeroClockObservationId);
        }
        Ok(())
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }
    pub fn source_epoch(&self) -> u64 {
        self.source_epoch
    }
    pub fn checked_observation_id(&self) -> &[u8; 32] {
        &self.checked_observation_id
    }
    pub fn max_uncertainty_ms(&self) -> u64 {
        self.max_uncertainty_ms
    }
}

/// Canonical replayable evidence projected from one exact #1198 commit capsule.
///
/// `validate()` checks only intrinsic/canonical consistency. It does not establish
/// that Xenia authenticated the transition, that the registry write occurred, or
/// that any root/grant/clock state is still current.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionCommitEvidenceV1 {
    schema_version: String,
    transition: VerifierProfileAdoptionTransitionV1,
    predecessor_transition: Option<VerifierProfileAdoptionTransitionV1>,
    root: VerifierProfileAdoptionRootEvidenceV1,
    grant: VerifierAdoptionAuthorityGrantEvidenceV1,
    clock_policy: VerifierProfileAdoptionClockPolicyEvidenceV1,
    record_id: VerifierProfileAdoptionCommitEvidenceIdV1,
}

impl VerifierProfileAdoptionCommitEvidenceV1 {
    pub fn project_from_commit_preconditions(
        preconditions: &GrantBoundVerifierProfileAdoptionCommitPreconditionsV1,
        predecessor_transition: Option<&VerifierProfileAdoptionTransitionV1>,
    ) -> Result<Self, VerifierProfileAdoptionCommitEvidenceError> {
        let grant_bound = preconditions.grant_bound();
        let root_bound = grant_bound.root_bound();
        let transition = root_bound.transition().clone();
        validate_predecessor_projection(
            &transition,
            root_bound.expected_predecessor_head(),
            predecessor_transition,
        )?;

        let root = VerifierProfileAdoptionRootEvidenceV1::from_snapshot(
            root_bound.authority_root_snapshot(),
        );
        let grant = VerifierAdoptionAuthorityGrantEvidenceV1::from_grant(
            grant_bound.authority_grant(),
        );
        let time_bound = preconditions.time_bound();
        let clock_policy = VerifierProfileAdoptionClockPolicyEvidenceV1 {
            source_id: time_bound.expected_clock_source_id().to_owned(),
            source_epoch: time_bound.expected_clock_epoch(),
            checked_observation_id: *time_bound.checked_clock_observation_id().as_bytes(),
            max_uncertainty_ms: time_bound.max_uncertainty_ms(),
        };

        let mut record = Self {
            schema_version: VERIFIER_PROFILE_ADOPTION_COMMIT_EVIDENCE_SCHEMA_V1.to_owned(),
            transition,
            predecessor_transition: predecessor_transition.cloned(),
            root,
            grant,
            clock_policy,
            record_id: VerifierProfileAdoptionCommitEvidenceIdV1([0; 32]),
        };
        record.record_id = VerifierProfileAdoptionCommitEvidenceIdV1(record.hash_record()?);
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), VerifierProfileAdoptionCommitEvidenceError> {
        if self.schema_version != VERIFIER_PROFILE_ADOPTION_COMMIT_EVIDENCE_SCHEMA_V1 {
            return Err(VerifierProfileAdoptionCommitEvidenceError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        self.transition.validate()?;
        validate_predecessor_record(&self.transition, self.predecessor_transition.as_ref())?;

        let root = self.root.reconstruct()?;
        let grant = self.grant.reconstruct(root.clone())?;
        self.clock_policy.validate()?;

        let subject = self.transition.subject();
        if subject.authority_subject() != root.authority_subject()
            || subject.authority_root_id() != root.authority_root_id()
            || subject.authority_root_digest() != root.authority_root_digest()
        {
            return Err(VerifierProfileAdoptionCommitEvidenceError::TransitionRootMismatch);
        }
        if subject.verifier_role_id() != grant.verifier_role_id() {
            return Err(VerifierProfileAdoptionCommitEvidenceError::TransitionGrantRoleMismatch);
        }
        if subject.evidence_class_ceiling() > grant.maximum_evidence_class() {
            return Err(VerifierProfileAdoptionCommitEvidenceError::TransitionExceedsGrantClass);
        }
        if !scope_is_subset(subject.scope(), grant.allowed_scope()) {
            return Err(VerifierProfileAdoptionCommitEvidenceError::TransitionExceedsGrantScope);
        }

        let expected = VerifierProfileAdoptionCommitEvidenceIdV1(self.hash_record()?);
        if expected != self.record_id {
            return Err(VerifierProfileAdoptionCommitEvidenceError::RecordIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> VerifierProfileAdoptionCommitEvidenceIdV1 {
        self.record_id
    }
    pub fn transition(&self) -> &VerifierProfileAdoptionTransitionV1 {
        &self.transition
    }
    pub fn predecessor_transition(&self) -> Option<&VerifierProfileAdoptionTransitionV1> {
        self.predecessor_transition.as_ref()
    }
    pub fn root(&self) -> &VerifierProfileAdoptionRootEvidenceV1 {
        &self.root
    }
    pub fn grant(&self) -> &VerifierAdoptionAuthorityGrantEvidenceV1 {
        &self.grant
    }
    pub fn clock_policy(&self) -> &VerifierProfileAdoptionClockPolicyEvidenceV1 {
        &self.clock_policy
    }

    /// Check that this durable evidence was projected from the exact same in-process
    /// commit capsule. This remains evidence equality, not current authorization.
    pub fn matches_commit_preconditions(
        &self,
        preconditions: &GrantBoundVerifierProfileAdoptionCommitPreconditionsV1,
    ) -> Result<(), VerifierProfileAdoptionCommitEvidenceError> {
        self.validate()?;
        let grant_bound = preconditions.grant_bound();
        let root_bound = grant_bound.root_bound();
        if self.transition.transition_digest()? != root_bound.transition_digest() {
            return Err(VerifierProfileAdoptionCommitEvidenceError::CommitTransitionMismatch);
        }
        if self.root.snapshot_id() != root_bound.authority_root_snapshot().id().as_bytes() {
            return Err(VerifierProfileAdoptionCommitEvidenceError::CommitRootSnapshotMismatch);
        }
        if self.grant.grant_id() != grant_bound.authority_grant().id().as_bytes() {
            return Err(VerifierProfileAdoptionCommitEvidenceError::CommitGrantMismatch);
        }
        let time_bound = preconditions.time_bound();
        if self.clock_policy.source_id() != time_bound.expected_clock_source_id()
            || self.clock_policy.source_epoch() != time_bound.expected_clock_epoch()
            || self.clock_policy.checked_observation_id()
                != time_bound.checked_clock_observation_id().as_bytes()
            || self.clock_policy.max_uncertainty_ms() != time_bound.max_uncertainty_ms()
        {
            return Err(VerifierProfileAdoptionCommitEvidenceError::CommitClockPolicyMismatch);
        }
        Ok(())
    }

    fn hash_record(&self) -> Result<[u8; 32], VerifierProfileAdoptionCommitEvidenceError> {
        let mut bytes = Vec::new();
        put_str(&mut bytes, VERIFIER_PROFILE_ADOPTION_COMMIT_EVIDENCE_SCHEMA_V1);
        put_bytes(&mut bytes, &self.transition.canonical_signing_bytes()?);
        match &self.predecessor_transition {
            None => bytes.push(0),
            Some(predecessor) => {
                bytes.push(1);
                put_bytes(&mut bytes, &predecessor.canonical_signing_bytes()?);
            }
        }
        encode_root(&mut bytes, &self.root);
        encode_grant(&mut bytes, &self.grant);
        encode_clock_policy(&mut bytes, &self.clock_policy);
        let mut hasher = blake3::Hasher::new();
        hasher.update(RECORD_DOMAIN);
        hasher.update(&bytes);
        Ok(*hasher.finalize().as_bytes())
    }
}

fn validate_predecessor_projection(
    transition: &VerifierProfileAdoptionTransitionV1,
    expected_head: &VerifierProfileAdoptionHeadV1,
    predecessor: Option<&VerifierProfileAdoptionTransitionV1>,
) -> Result<(), VerifierProfileAdoptionCommitEvidenceError> {
    match expected_head {
        VerifierProfileAdoptionHeadV1::Uninitialized => {
            if predecessor.is_some() {
                return Err(VerifierProfileAdoptionCommitEvidenceError::UnexpectedPredecessor);
            }
        }
        VerifierProfileAdoptionHeadV1::Current(_) => {
            let predecessor = predecessor
                .ok_or(VerifierProfileAdoptionCommitEvidenceError::MissingPredecessor)?;
            let observed = VerifierProfileAdoptionHeadV1::from_transition(predecessor)?;
            if &observed != expected_head {
                return Err(VerifierProfileAdoptionCommitEvidenceError::PredecessorHeadMismatch);
            }
        }
    }
    validate_predecessor_record(transition, predecessor)
}

fn validate_predecessor_record(
    transition: &VerifierProfileAdoptionTransitionV1,
    predecessor: Option<&VerifierProfileAdoptionTransitionV1>,
) -> Result<(), VerifierProfileAdoptionCommitEvidenceError> {
    match (transition.predecessor(), predecessor) {
        (VerifierProfileAdoptionPredecessorV1::Bootstrap, None) => {
            let rebuilt = VerifierProfileAdoptionTransitionV1::bootstrap(
                transition.subject().clone(),
            )?;
            if rebuilt != *transition {
                return Err(VerifierProfileAdoptionCommitEvidenceError::PredecessorRecordMismatch);
            }
        }
        (VerifierProfileAdoptionPredecessorV1::Previous(expected), Some(predecessor)) => {
            predecessor.validate()?;
            if predecessor.transition_digest()? != expected {
                return Err(VerifierProfileAdoptionCommitEvidenceError::PredecessorRecordMismatch);
            }
            let rebuilt = VerifierProfileAdoptionTransitionV1::successor(
                transition.subject().clone(),
                predecessor,
            )?;
            if rebuilt != *transition {
                return Err(VerifierProfileAdoptionCommitEvidenceError::PredecessorRecordMismatch);
            }
        }
        _ => return Err(VerifierProfileAdoptionCommitEvidenceError::PredecessorRecordMismatch),
    }
    Ok(())
}

fn scope_is_subset(candidate: &VerifierAdoptionScopeV1, allowed: &VerifierAdoptionScopeV1) -> bool {
    match allowed {
        VerifierAdoptionScopeV1::AllContinuityVerification => true,
        VerifierAdoptionScopeV1::Contract {
            contract_id: allowed_contract,
        } => match candidate {
            VerifierAdoptionScopeV1::AllContinuityVerification => false,
            VerifierAdoptionScopeV1::Contract { contract_id }
            | VerifierAdoptionScopeV1::Requirements { contract_id, .. } => {
                contract_id == allowed_contract
            }
        },
        VerifierAdoptionScopeV1::Requirements {
            contract_id: allowed_contract,
            requirement_ids: allowed_requirements,
        } => match candidate {
            VerifierAdoptionScopeV1::Requirements {
                contract_id,
                requirement_ids,
            } if contract_id == allowed_contract => requirement_ids
                .iter()
                .all(|id| allowed_requirements.binary_search(id).is_ok()),
            _ => false,
        },
    }
}

fn encode_root(out: &mut Vec<u8>, root: &VerifierProfileAdoptionRootEvidenceV1) {
    put_str(out, &root.authority_subject);
    put_str(out, &root.authority_root_id);
    out.extend_from_slice(&root.authority_root_digest);
    out.extend_from_slice(&root.provisioning_epoch.to_le_bytes());
    out.extend_from_slice(&root.snapshot_id);
}

fn encode_grant(out: &mut Vec<u8>, grant: &VerifierAdoptionAuthorityGrantEvidenceV1) {
    put_str(out, &grant.grant_slot_id);
    put_str(out, &grant.verifier_role_id);
    out.extend_from_slice(&grant.grant_epoch.to_le_bytes());
    out.push(evidence_class_tag(grant.maximum_evidence_class));
    encode_scope(out, &grant.allowed_scope);
    out.extend_from_slice(&grant.grant_id);
}

fn encode_clock_policy(out: &mut Vec<u8>, clock: &VerifierProfileAdoptionClockPolicyEvidenceV1) {
    put_str(out, &clock.source_id);
    out.extend_from_slice(&clock.source_epoch.to_le_bytes());
    out.extend_from_slice(&clock.checked_observation_id);
    out.extend_from_slice(&clock.max_uncertainty_ms.to_le_bytes());
}

fn encode_scope(out: &mut Vec<u8>, scope: &VerifierAdoptionScopeV1) {
    match scope {
        VerifierAdoptionScopeV1::AllContinuityVerification => out.push(0),
        VerifierAdoptionScopeV1::Contract { contract_id } => {
            out.push(1);
            out.extend_from_slice(contract_id.as_bytes());
        }
        VerifierAdoptionScopeV1::Requirements {
            contract_id,
            requirement_ids,
        } => {
            out.push(2);
            out.extend_from_slice(contract_id.as_bytes());
            put_len(out, requirement_ids.len());
            for requirement_id in requirement_ids {
                out.extend_from_slice(requirement_id.as_bytes());
            }
        }
    }
}

fn evidence_class_tag(class: EvidenceClass) -> u8 {
    match class {
        EvidenceClass::Declared => 1,
        EvidenceClass::Observed => 2,
        EvidenceClass::StaticAnalysis => 3,
        EvidenceClass::Simulated => 4,
        EvidenceClass::DifferentiallyVerified => 5,
        EvidenceClass::HardwareVerified => 6,
        EvidenceClass::IndependentlyReplicated => 7,
    }
}

fn checked_text(
    field: &'static str,
    value: &str,
) -> Result<(), VerifierProfileAdoptionCommitEvidenceError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(VerifierProfileAdoptionCommitEvidenceError::BlankText { field });
    }
    if trimmed.len() > 1024 {
        return Err(VerifierProfileAdoptionCommitEvidenceError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(VerifierProfileAdoptionCommitEvidenceError::ControlCharacters { field });
    }
    Ok(())
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value);
}

fn put_len(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u64).to_le_bytes());
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionCommitEvidenceError {
    #[error(transparent)]
    Adoption(#[from] VerifierProfileAdoptionError),
    #[error(transparent)]
    Root(#[from] VerifierProfileAdoptionRootBindingError),
    #[error(transparent)]
    Grant(#[from] VerifierAdoptionAuthorityGrantError),
    #[error("unsupported verifier-profile adoption commit-evidence schema {0}")]
    UnsupportedSchema(String),
    #[error("stored adoption-root snapshot identity is not canonical")]
    RootSnapshotIdentityMismatch,
    #[error("stored adoption-grant identity is not canonical")]
    GrantIdentityMismatch,
    #[error("stored commit-evidence identity is not canonical")]
    RecordIdentityMismatch,
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("clock source epoch must be non-zero")]
    ZeroClockEpoch,
    #[error("checked clock observation identity must be non-zero")]
    ZeroClockObservationId,
    #[error("commit evidence contains unexpected predecessor transition")]
    UnexpectedPredecessor,
    #[error("commit evidence is missing required predecessor transition")]
    MissingPredecessor,
    #[error("commit evidence predecessor does not equal the expected predecessor head")]
    PredecessorHeadMismatch,
    #[error("commit evidence predecessor lineage is not exact")]
    PredecessorRecordMismatch,
    #[error("commit evidence transition does not match its historical root snapshot")]
    TransitionRootMismatch,
    #[error("commit evidence transition role does not match its historical grant")]
    TransitionGrantRoleMismatch,
    #[error("commit evidence transition exceeds historical grant evidence class")]
    TransitionExceedsGrantClass,
    #[error("commit evidence transition exceeds historical grant scope")]
    TransitionExceedsGrantScope,
    #[error("commit evidence transition differs from supplied commit preconditions")]
    CommitTransitionMismatch,
    #[error("commit evidence root snapshot differs from supplied commit preconditions")]
    CommitRootSnapshotMismatch,
    #[error("commit evidence grant differs from supplied commit preconditions")]
    CommitGrantMismatch,
    #[error("commit evidence clock policy differs from supplied commit preconditions")]
    CommitClockPolicyMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GrantBoundVerifierProfileAdoptionCommitPreconditionsV1,
        RootBoundPolicyCheckedVerifierProfileAdoptionV1,
        TimeBoundVerifierProfileAdoptionCommitPreconditionsV1,
        VerifierAdoptionAuthorityGrantV1, VerifierAdoptionScopeV1,
        VerifierProfileAdoptionAdmissionPolicyV1,
        VerifierProfileAdoptionAuthorityRootSnapshotV1,
        VerifierProfileAdoptionClockObservationV1,
        VerifierProfileAdoptionCommitPreconditionsV1, VerifierProfileAdoptionHeadV1,
        VerifierProfileAdoptionSubjectV1, VerifierProfileAdoptionTransitionV1,
        VerifierProfileV1, bind_root_bound_adoption_to_authority_grant,
    };

    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "hardware-verifier-v1",
            [9; 32],
            7,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    fn root(epoch: u64) -> VerifierProfileAdoptionAuthorityRootSnapshotV1 {
        VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            epoch,
        )
        .unwrap()
    }

    fn grant(
        root: VerifierProfileAdoptionAuthorityRootSnapshotV1,
        epoch: u64,
        maximum: EvidenceClass,
    ) -> VerifierAdoptionAuthorityGrantV1 {
        VerifierAdoptionAuthorityGrantV1::new(
            "grant-slot-1",
            root,
            "hardware-verifier-v1",
            epoch,
            maximum,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap()
    }

    fn fixture(
        root_epoch: u64,
        grant_epoch: u64,
    ) -> (
        GrantBoundVerifierProfileAdoptionCommitPreconditionsV1,
        VerifierProfileAdoptionAuthorityRootSnapshotV1,
    ) {
        let profile = profile();
        let root = root(root_epoch);
        let subject = VerifierProfileAdoptionSubjectV1::new(
            "adopt-1",
            root.authority_subject(),
            root.authority_root_id(),
            root.authority_root_digest(),
            &profile,
            1,
            1_000,
            2_000,
            EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let checked = VerifierProfileAdoptionAdmissionPolicyV1::new(
            root.authority_subject(),
            root.authority_root_id(),
            root.authority_root_digest(),
            profile.profile_name(),
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap()
        .check(1_500, &transition, &profile, None)
        .unwrap();
        let root_bound =
            RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked, root.clone()).unwrap();
        let grant = grant(root.clone(), grant_epoch, EvidenceClass::DifferentiallyVerified);
        let grant_bound = bind_root_bound_adoption_to_authority_grant(root_bound.clone(), &grant, None)
            .unwrap();
        let commit = VerifierProfileAdoptionCommitPreconditionsV1::from_root_bound(root_bound)
            .unwrap();
        let checked_clock = VerifierProfileAdoptionClockObservationV1::new(
            "trusted-clock-1",
            4,
            1_490,
            1_510,
        )
        .unwrap();
        let time_bound = TimeBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
            commit,
            &checked_clock,
            50,
        )
        .unwrap();
        let preconditions = GrantBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
            grant_bound,
            time_bound,
        )
        .unwrap();
        (preconditions, root)
    }

    #[test]
    fn projection_is_deterministic_and_matches_exact_commit_capsule() {
        let (preconditions, _) = fixture(9, 3);
        let a = VerifierProfileAdoptionCommitEvidenceV1::project_from_commit_preconditions(
            &preconditions,
            None,
        )
        .unwrap();
        let b = VerifierProfileAdoptionCommitEvidenceV1::project_from_commit_preconditions(
            &preconditions,
            None,
        )
        .unwrap();
        assert_eq!(a.id(), b.id());
        a.validate().unwrap();
        a.matches_commit_preconditions(&preconditions).unwrap();
    }

    #[test]
    fn root_reprovisioning_changes_evidence_identity() {
        let (a, _) = fixture(9, 3);
        let (b, _) = fixture(10, 3);
        let a = VerifierProfileAdoptionCommitEvidenceV1::project_from_commit_preconditions(&a, None)
            .unwrap();
        let b = VerifierProfileAdoptionCommitEvidenceV1::project_from_commit_preconditions(&b, None)
            .unwrap();
        assert_ne!(a.root().snapshot_id(), b.root().snapshot_id());
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn grant_epoch_changes_evidence_identity() {
        let (a, _) = fixture(9, 3);
        let (b, _) = fixture(9, 4);
        let a = VerifierProfileAdoptionCommitEvidenceV1::project_from_commit_preconditions(&a, None)
            .unwrap();
        let b = VerifierProfileAdoptionCommitEvidenceV1::project_from_commit_preconditions(&b, None)
            .unwrap();
        assert_ne!(a.grant().grant_id(), b.grant().grant_id());
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn tampered_root_epoch_fails_intrinsic_validation() {
        let (preconditions, _) = fixture(9, 3);
        let mut record = VerifierProfileAdoptionCommitEvidenceV1::project_from_commit_preconditions(
            &preconditions,
            None,
        )
        .unwrap();
        record.root.provisioning_epoch += 1;
        assert_eq!(
            record.validate().unwrap_err(),
            VerifierProfileAdoptionCommitEvidenceError::RootSnapshotIdentityMismatch
        );
    }

    #[test]
    fn bootstrap_record_rejects_unexpected_predecessor() {
        let (preconditions, _) = fixture(9, 3);
        let transition = preconditions.grant_bound().root_bound().transition().clone();
        let err = VerifierProfileAdoptionCommitEvidenceV1::project_from_commit_preconditions(
            &preconditions,
            Some(&transition),
        )
        .unwrap_err();
        assert_eq!(err, VerifierProfileAdoptionCommitEvidenceError::UnexpectedPredecessor);
    }

    #[test]
    fn record_is_evidence_not_currentness() {
        let (preconditions, _) = fixture(9, 3);
        let record = VerifierProfileAdoptionCommitEvidenceV1::project_from_commit_preconditions(
            &preconditions,
            None,
        )
        .unwrap();
        // The public API exposes only historical facts and equality checks. It has
        // no method that returns PolicyCurrentVerifierRuntimeEnvelopeV1 or an
        // authorized verifier profile after deserialization.
        assert_eq!(
            record.transition().transition_digest().unwrap(),
            preconditions.grant_bound().root_bound().transition_digest()
        );
    }
}
