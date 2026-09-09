// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact typed-subject retention for qualified continuity witnesses.
//!
//! A closed-world witness over an exact contract and target is still incomplete as
//! an enterprise transition fact if it forgets whether that contract concerned a
//! machine, service, cluster, network device, fabric, site, or fleet. This module
//! wraps the existing qualified witness with the constructor-qualified
//! subject/contract relation established by `subject_contract`.
//!
//! Core theorem:
//!
//! `QualifiedContinuityWitnessV1 != SubjectBoundQualifiedContinuityWitnessV1 != ExecutionAuthority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::contract::ContinuityContractId;
use crate::scope::{ContinuitySubjectId, ContinuitySubjectV1};
use crate::subject_contract::{
    SubjectBoundContinuityContractId, SubjectBoundContinuityContractV1,
    SubjectContractBindingError,
};
use crate::witness::{QualifiedContinuityWitnessV1, TargetRealizationId, WitnessId};

const SUBJECT_WITNESS_BINDING_DOMAIN: &[u8] =
    b"symthaea.continuity.subject-qualified-witness.v1\0";

/// Stable reference identity for one exact subject-bound qualified witness.
///
/// This ID is serializable evidence/reference material only. It cannot recreate
/// the constructor-only wrapper and does not grant migration or execution
/// authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SubjectBoundQualifiedContinuityWitnessId([u8; 32]);

impl SubjectBoundQualifiedContinuityWitnessId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Non-Serde qualified witness that retains the exact typed operational subject.
///
/// Production callers cannot deserialize this value. It is constructed only by
/// the crate-owned closed-world composer after verifier-owned evidence has
/// qualified the exact contract/target world.
#[derive(Debug, Clone)]
pub struct SubjectBoundQualifiedContinuityWitnessV1 {
    subject_contract: SubjectBoundContinuityContractV1,
    witness: QualifiedContinuityWitnessV1,
    binding_id: SubjectBoundQualifiedContinuityWitnessId,
}

impl SubjectBoundQualifiedContinuityWitnessV1 {
    pub(crate) fn from_qualified(
        subject_contract: SubjectBoundContinuityContractV1,
        witness: QualifiedContinuityWitnessV1,
    ) -> Result<Self, SubjectWitnessBindingError> {
        subject_contract.validate()?;
        if witness.contract_id() != subject_contract.contract_id() {
            return Err(SubjectWitnessBindingError::CrossContractWitness);
        }

        let binding_id = SubjectBoundQualifiedContinuityWitnessId(hash_binding(
            subject_contract.id(),
            witness.id(),
        ));

        Ok(Self {
            subject_contract,
            witness,
            binding_id,
        })
    }

    /// Recheck the subject/contract relation, exact witness contract and binding
    /// identity. This does not establish freshness, commit eligibility, owner
    /// authority, recoverability, or execution permission.
    pub fn validate(&self) -> Result<(), SubjectWitnessBindingError> {
        self.subject_contract.validate()?;
        if self.witness.contract_id() != self.subject_contract.contract_id() {
            return Err(SubjectWitnessBindingError::CrossContractWitness);
        }
        let expected = SubjectBoundQualifiedContinuityWitnessId(hash_binding(
            self.subject_contract.id(),
            self.witness.id(),
        ));
        if expected != self.binding_id {
            return Err(SubjectWitnessBindingError::BindingIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> SubjectBoundQualifiedContinuityWitnessId {
        self.binding_id
    }

    pub fn subject_id(&self) -> ContinuitySubjectId {
        self.subject_contract.subject_id()
    }

    pub fn subject_contract_binding_id(&self) -> SubjectBoundContinuityContractId {
        self.subject_contract.id()
    }

    pub fn contract_id(&self) -> ContinuityContractId {
        self.witness.contract_id()
    }

    pub fn witness_id(&self) -> WitnessId {
        self.witness.id()
    }

    pub fn target_realization_id(&self) -> TargetRealizationId {
        self.witness.target_realization_id()
    }

    pub fn nonblocking_issues(&self) -> usize {
        self.witness.nonblocking_issues()
    }

    pub fn subject(&self) -> &ContinuitySubjectV1 {
        self.subject_contract.subject()
    }

    pub fn subject_contract(&self) -> &SubjectBoundContinuityContractV1 {
        &self.subject_contract
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SubjectWitnessBindingError {
    #[error(transparent)]
    SubjectContract(#[from] SubjectContractBindingError),
    #[error("qualified witness belongs to a different continuity contract than the exact typed subject binding")]
    CrossContractWitness,
    #[error("stored subject-bound qualified witness identity does not match exact subject/contract binding and witness")]
    BindingIdentityMismatch,
}

fn hash_binding(
    subject_contract_id: SubjectBoundContinuityContractId,
    witness_id: WitnessId,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SUBJECT_WITNESS_BINDING_DOMAIN);
    hasher.update(subject_contract_id.as_bytes());
    hasher.update(witness_id.as_bytes());
    *hasher.finalize().as_bytes()
}
