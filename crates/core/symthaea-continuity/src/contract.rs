// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validated continuity contracts.
//!
//! A dependency claim can suggest what matters, but only an explicitly approved
//! continuity contract defines what a transition must preserve.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::observation::DependencyClaimId;

const REQUIREMENT_DOMAIN: &[u8] = b"symthaea.continuity.requirement.v1\0";
const CONTRACT_DOMAIN: &[u8] = b"symthaea.continuity.contract.v1\0";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct ContinuityRequirementId([u8; 32]);

impl ContinuityRequirementId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct ContinuityContractId([u8; 32]);

impl ContinuityContractId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequirementCriticality {
    Must,
    Should,
    Optional,
}

impl RequirementCriticality {
    fn tag(self) -> u8 {
        match self {
            Self::Must => 1,
            Self::Should => 2,
            Self::Optional => 3,
        }
    }
}

/// How the organization claims that a dependency became a continuity requirement.
///
/// `LegacyUnspecified` is deliberately representable for imported/historical
/// records, but cannot appear in a validated continuity contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalBasis {
    ExplicitHuman,
    ExplicitPolicy,
    ImportedApproved,
    LegacyUnspecified,
}

impl ApprovalBasis {
    fn tag(self) -> u8 {
        match self {
            Self::ExplicitHuman => 1,
            Self::ExplicitPolicy => 2,
            Self::ImportedApproved => 3,
            Self::LegacyUnspecified => 4,
        }
    }

    fn is_explicitly_approved(self) -> bool {
        !matches!(self, Self::LegacyUnspecified)
    }
}

/// What kind of equivalence is sufficient for one continuity requirement.
///
/// V1 intentionally avoids a universal scalar score and avoids embedding
/// arbitrary executable scripts. Concrete verifier implementations live above
/// this pure contract layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum EquivalencePredicate {
    ExactContent,
    StructuredInvariant { schema_id: String },
    BehavioralScenario { scenario_id: String },
    ServiceLevel { objective_id: String },
    HumanAcceptance { rubric_id: String },
}

impl EquivalencePredicate {
    fn validate(&self) -> Result<(), ContractError> {
        match self {
            Self::ExactContent => Ok(()),
            Self::StructuredInvariant { schema_id } => {
                checked_text("equivalence schema_id", schema_id.clone()).map(|_| ())
            }
            Self::BehavioralScenario { scenario_id } => {
                checked_text("equivalence scenario_id", scenario_id.clone()).map(|_| ())
            }
            Self::ServiceLevel { objective_id } => {
                checked_text("equivalence objective_id", objective_id.clone()).map(|_| ())
            }
            Self::HumanAcceptance { rubric_id } => {
                checked_text("equivalence rubric_id", rubric_id.clone()).map(|_| ())
            }
        }
    }

    fn encode(&self, bytes: &mut Vec<u8>) {
        match self {
            Self::ExactContent => bytes.push(1),
            Self::StructuredInvariant { schema_id } => {
                bytes.push(2);
                put_str(bytes, schema_id);
            }
            Self::BehavioralScenario { scenario_id } => {
                bytes.push(3);
                put_str(bytes, scenario_id);
            }
            Self::ServiceLevel { objective_id } => {
                bytes.push(4);
                put_str(bytes, objective_id);
            }
            Self::HumanAcceptance { rubric_id } => {
                bytes.push(5);
                put_str(bytes, rubric_id);
            }
        }
    }
}

/// Raw serializable continuity requirement.
///
/// Deserialization does not make this value approved. Only a validated contract
/// can promote structurally valid requirements into the contract set consumed by
/// later witness construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuityRequirementV1 {
    dependency_claim: DependencyClaimId,
    capability: String,
    criticality: RequirementCriticality,
    equivalence: EquivalencePredicate,
    approval_basis: ApprovalBasis,
    approval_evidence_digest: [u8; 32],
    requirement_id: ContinuityRequirementId,
}

impl ContinuityRequirementV1 {
    pub fn new(
        dependency_claim: DependencyClaimId,
        capability: impl Into<String>,
        criticality: RequirementCriticality,
        equivalence: EquivalencePredicate,
        approval_basis: ApprovalBasis,
        approval_evidence_digest: [u8; 32],
    ) -> Result<Self, ContractError> {
        let capability = checked_text("capability", capability.into())?;
        equivalence.validate()?;
        if approval_basis.is_explicitly_approved() && approval_evidence_digest == [0; 32] {
            return Err(ContractError::MissingApprovalEvidence);
        }
        let requirement_id = ContinuityRequirementId(hash_requirement(
            dependency_claim,
            &capability,
            criticality,
            &equivalence,
            approval_basis,
            approval_evidence_digest,
        ));
        Ok(Self {
            dependency_claim,
            capability,
            criticality,
            equivalence,
            approval_basis,
            approval_evidence_digest,
            requirement_id,
        })
    }

    pub fn id(&self) -> ContinuityRequirementId {
        self.requirement_id
    }

    pub fn dependency_claim(&self) -> DependencyClaimId {
        self.dependency_claim
    }

    pub fn capability(&self) -> &str {
        &self.capability
    }

    pub fn criticality(&self) -> RequirementCriticality {
        self.criticality
    }

    pub fn equivalence(&self) -> &EquivalencePredicate {
        &self.equivalence
    }

    pub fn approval_basis(&self) -> ApprovalBasis {
        self.approval_basis
    }

    pub fn approval_evidence_digest(&self) -> [u8; 32] {
        self.approval_evidence_digest
    }

    fn validate_structure(&self) -> Result<(), ContractError> {
        checked_text("capability", self.capability.clone())?;
        self.equivalence.validate()?;
        if self.approval_basis.is_explicitly_approved() && self.approval_evidence_digest == [0; 32]
        {
            return Err(ContractError::MissingApprovalEvidence);
        }
        let expected = ContinuityRequirementId(hash_requirement(
            self.dependency_claim,
            &self.capability,
            self.criticality,
            &self.equivalence,
            self.approval_basis,
            self.approval_evidence_digest,
        ));
        if self.requirement_id != expected {
            return Err(ContractError::RequirementIdentityMismatch);
        }
        Ok(())
    }
}

/// Raw serializable contract candidate.
///
/// A raw contract may intentionally preserve `LegacyUnspecified` requirements so
/// historical/imported ambiguity is visible. Call `validate()` before the value
/// can participate in witness construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuityContractV1 {
    subject: String,
    source_snapshot_digest: [u8; 32],
    requirements: Vec<ContinuityRequirementV1>,
    contract_id: ContinuityContractId,
}

impl ContinuityContractV1 {
    pub fn new(
        subject: impl Into<String>,
        source_snapshot_digest: [u8; 32],
        mut requirements: Vec<ContinuityRequirementV1>,
    ) -> Result<Self, ContractError> {
        let subject = checked_text("contract subject", subject.into())?;
        if source_snapshot_digest == [0; 32] {
            return Err(ContractError::ZeroSourceSnapshot);
        }
        if requirements.is_empty() {
            return Err(ContractError::NoRequirements);
        }
        for requirement in &requirements {
            requirement.validate_structure()?;
        }
        requirements.sort_by_key(ContinuityRequirementV1::id);
        if requirements.windows(2).any(|pair| pair[0].id() == pair[1].id()) {
            return Err(ContractError::DuplicateRequirement);
        }
        let contract_id = ContinuityContractId(hash_contract(
            &subject,
            source_snapshot_digest,
            &requirements,
        ));
        Ok(Self {
            subject,
            source_snapshot_digest,
            requirements,
            contract_id,
        })
    }

    pub fn id(&self) -> ContinuityContractId {
        self.contract_id
    }

    pub fn subject(&self) -> &str {
        &self.subject
    }

    pub fn source_snapshot_digest(&self) -> [u8; 32] {
        self.source_snapshot_digest
    }

    pub fn requirements(&self) -> &[ContinuityRequirementV1] {
        &self.requirements
    }

    pub fn validate(&self) -> Result<ValidatedContinuityContractV1, ContractError> {
        checked_text("contract subject", self.subject.clone())?;
        if self.source_snapshot_digest == [0; 32] {
            return Err(ContractError::ZeroSourceSnapshot);
        }
        if self.requirements.is_empty() {
            return Err(ContractError::NoRequirements);
        }
        let mut canonical = self.requirements.clone();
        for requirement in &canonical {
            requirement.validate_structure()?;
            if !requirement.approval_basis.is_explicitly_approved() {
                return Err(ContractError::UnapprovedRequirement {
                    requirement: requirement.id(),
                });
            }
        }
        canonical.sort_by_key(ContinuityRequirementV1::id);
        if canonical.windows(2).any(|pair| pair[0].id() == pair[1].id()) {
            return Err(ContractError::DuplicateRequirement);
        }
        if canonical != self.requirements {
            return Err(ContractError::NonCanonicalRequirementOrder);
        }
        let expected = ContinuityContractId(hash_contract(
            &self.subject,
            self.source_snapshot_digest,
            &self.requirements,
        ));
        if self.contract_id != expected {
            return Err(ContractError::ContractIdentityMismatch);
        }
        Ok(ValidatedContinuityContractV1 { inner: self.clone() })
    }
}

/// Non-Serde validated wrapper. This is the only contract form intended for
/// downstream witness construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedContinuityContractV1 {
    inner: ContinuityContractV1,
}

impl ValidatedContinuityContractV1 {
    pub fn id(&self) -> ContinuityContractId {
        self.inner.id()
    }

    pub fn subject(&self) -> &str {
        self.inner.subject()
    }

    pub fn source_snapshot_digest(&self) -> [u8; 32] {
        self.inner.source_snapshot_digest()
    }

    pub fn requirements(&self) -> &[ContinuityRequirementV1] {
        self.inner.requirements()
    }

    pub fn as_raw(&self) -> &ContinuityContractV1 {
        &self.inner
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ContractError {
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("approved requirement requires non-zero approval evidence")]
    MissingApprovalEvidence,
    #[error("source estate snapshot digest must be non-zero")]
    ZeroSourceSnapshot,
    #[error("continuity contract must contain at least one requirement")]
    NoRequirements,
    #[error("continuity contract contains duplicate requirement identity")]
    DuplicateRequirement,
    #[error("continuity contract requirements must be in canonical identity order")]
    NonCanonicalRequirementOrder,
    #[error("requirement {requirement:?} has no explicit approval basis")]
    UnapprovedRequirement {
        requirement: ContinuityRequirementId,
    },
    #[error("stored requirement identity does not match canonical fields")]
    RequirementIdentityMismatch,
    #[error("stored contract identity does not match canonical fields")]
    ContractIdentityMismatch,
}

fn checked_text(field: &'static str, value: String) -> Result<String, ContractError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ContractError::BlankText { field });
    }
    if trimmed.len() > 1024 {
        return Err(ContractError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(ContractError::ControlCharacters { field });
    }
    Ok(trimmed.to_string())
}

fn hash_requirement(
    dependency_claim: DependencyClaimId,
    capability: &str,
    criticality: RequirementCriticality,
    equivalence: &EquivalencePredicate,
    approval_basis: ApprovalBasis,
    approval_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(dependency_claim.as_bytes());
    put_str(&mut bytes, capability);
    bytes.push(criticality.tag());
    equivalence.encode(&mut bytes);
    bytes.push(approval_basis.tag());
    bytes.extend_from_slice(&approval_evidence_digest);
    domain_hash(REQUIREMENT_DOMAIN, &bytes)
}

fn hash_contract(
    subject: &str,
    source_snapshot_digest: [u8; 32],
    requirements: &[ContinuityRequirementV1],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, subject);
    bytes.extend_from_slice(&source_snapshot_digest);
    put_len(&mut bytes, requirements.len());
    for requirement in requirements {
        bytes.extend_from_slice(requirement.id().as_bytes());
    }
    domain_hash(CONTRACT_DOMAIN, &bytes)
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn put_len(bytes: &mut Vec<u8>, len: usize) {
    bytes.extend_from_slice(&(len as u64).to_le_bytes());
}

fn put_str(bytes: &mut Vec<u8>, value: &str) {
    put_len(bytes, value.len());
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::{
        DependencyBasis, DependencyClaimV1, EvidenceBasis, ObservationCoverage,
        ObservationEnvelopeV1,
    };

    fn dependency(seed: u8) -> DependencyClaimId {
        let observation = ObservationEnvelopeV1::new(
            "machine-1",
            "workflow.dependency",
            "fixture",
            "1",
            1_700_000_000_000,
            ObservationCoverage::Complete,
            EvidenceBasis::Declared,
            [seed; 32],
            vec![],
        )
        .unwrap();
        DependencyClaimV1::new(
            "role:research",
            "requires",
            format!("capability:{seed}"),
            DependencyBasis::Declared,
            vec![observation.id()],
            vec![],
        )
        .unwrap()
        .id()
    }

    fn approved_requirement(seed: u8) -> ContinuityRequirementV1 {
        ContinuityRequirementV1::new(
            dependency(seed),
            format!("workflow-{seed}"),
            RequirementCriticality::Must,
            EquivalencePredicate::BehavioralScenario {
                scenario_id: format!("scenario-{seed}"),
            },
            ApprovalBasis::ExplicitHuman,
            [seed.wrapping_add(20); 32],
        )
        .unwrap()
    }

    #[test]
    fn legacy_unspecified_is_representable_but_not_validatable() {
        let requirement = ContinuityRequirementV1::new(
            dependency(1),
            "quarterly-payroll",
            RequirementCriticality::Must,
            EquivalencePredicate::BehavioralScenario {
                scenario_id: "payroll-v1".into(),
            },
            ApprovalBasis::LegacyUnspecified,
            [0; 32],
        )
        .unwrap();
        let raw = ContinuityContractV1::new("finance-fleet", [9; 32], vec![requirement]).unwrap();
        assert!(matches!(
            raw.validate(),
            Err(ContractError::UnapprovedRequirement { .. })
        ));
    }

    #[test]
    fn explicit_approval_yields_validated_contract() {
        let raw = ContinuityContractV1::new(
            "research-fleet",
            [8; 32],
            vec![approved_requirement(1), approved_requirement(2)],
        )
        .unwrap();
        let validated = raw.validate().unwrap();
        assert_eq!(validated.id(), raw.id());
        assert_eq!(validated.requirements().len(), 2);
    }

    #[test]
    fn requirement_input_order_does_not_change_contract_identity() {
        let first = approved_requirement(3);
        let second = approved_requirement(4);
        let left = ContinuityContractV1::new(
            "fleet-a",
            [7; 32],
            vec![first.clone(), second.clone()],
        )
        .unwrap();
        let right = ContinuityContractV1::new("fleet-a", [7; 32], vec![second, first]).unwrap();
        assert_eq!(left.id(), right.id());
        left.validate().unwrap();
        right.validate().unwrap();
    }

    #[test]
    fn changed_equivalence_changes_requirement_identity() {
        let dep = dependency(5);
        let exact = ContinuityRequirementV1::new(
            dep,
            "document-preservation",
            RequirementCriticality::Must,
            EquivalencePredicate::ExactContent,
            ApprovalBasis::ExplicitPolicy,
            [31; 32],
        )
        .unwrap();
        let scenario = ContinuityRequirementV1::new(
            dep,
            "document-preservation",
            RequirementCriticality::Must,
            EquivalencePredicate::BehavioralScenario {
                scenario_id: "roundtrip-v1".into(),
            },
            ApprovalBasis::ExplicitPolicy,
            [31; 32],
        )
        .unwrap();
        assert_ne!(exact.id(), scenario.id());
    }

    #[test]
    fn source_snapshot_changes_contract_identity() {
        let requirement = approved_requirement(6);
        let left = ContinuityContractV1::new("fleet-a", [1; 32], vec![requirement.clone()])
            .unwrap();
        let right = ContinuityContractV1::new("fleet-a", [2; 32], vec![requirement]).unwrap();
        assert_ne!(left.id(), right.id());
    }

    #[test]
    fn duplicate_requirements_fail_closed() {
        let requirement = approved_requirement(7);
        let result = ContinuityContractV1::new(
            "fleet-a",
            [3; 32],
            vec![requirement.clone(), requirement],
        );
        assert_eq!(result.unwrap_err(), ContractError::DuplicateRequirement);
    }
}
