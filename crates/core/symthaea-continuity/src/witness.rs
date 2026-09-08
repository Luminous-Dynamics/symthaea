// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Closed-world continuity witness composition.
//!
//! Core theorem:
//!
//! `Complete + Coherent + SufficientEvidence = QualifiedWitness`.
//!
//! A witness still does not grant migration authority. It proves only that the
//! preregistered continuity obligations for one exact contract/target have been
//! completely dispositioned and that every blocking obligation is satisfied at
//! or above its declared evidence floor.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::contract::{
    ContinuityContractId, ContinuityRequirementId, RequirementCriticality,
    ValidatedContinuityContractV1,
};

const TARGET_DOMAIN: &[u8] = b"symthaea.continuity.target-realization.v1\0";
const POLICY_DOMAIN: &[u8] = b"symthaea.continuity.verification-policy.v1\0";
const OBLIGATION_DOMAIN: &[u8] = b"symthaea.continuity.verification-obligation.v1\0";
const MANIFEST_DOMAIN: &[u8] = b"symthaea.continuity.witness-manifest.v1\0";
const WITNESS_DOMAIN: &[u8] = b"symthaea.continuity.witness-evaluation.v1\0";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct TargetRealizationId([u8; 32]);

impl TargetRealizationId {
    pub fn from_digest(digest: [u8; 32]) -> Result<Self, WitnessError> {
        if digest == [0; 32] {
            return Err(WitnessError::ZeroTargetRealization);
        }
        Ok(Self(domain_hash(TARGET_DOMAIN, &digest)))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct VerificationPolicyId([u8; 32]);

impl VerificationPolicyId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct VerificationObligationId([u8; 32]);

impl VerificationObligationId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct WitnessManifestId([u8; 32]);

impl WitnessManifestId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct WitnessId([u8; 32]);

impl WitnessId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Evidence classes are ordered only for satisfying an explicit minimum evidence
/// floor. They are not a universal confidence score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    Declared,
    Observed,
    StaticAnalysis,
    Simulated,
    DifferentiallyVerified,
    HardwareVerified,
    IndependentlyReplicated,
}

impl EvidenceClass {
    fn tag(self) -> u8 {
        match self {
            Self::Declared => 1,
            Self::Observed => 2,
            Self::StaticAnalysis => 3,
            Self::Simulated => 4,
            Self::DifferentiallyVerified => 5,
            Self::HardwareVerified => 6,
            Self::IndependentlyReplicated => 7,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationPolicyEntryV1 {
    requirement_id: ContinuityRequirementId,
    minimum_evidence_class: EvidenceClass,
}

impl VerificationPolicyEntryV1 {
    pub fn new(
        requirement_id: ContinuityRequirementId,
        minimum_evidence_class: EvidenceClass,
    ) -> Self {
        Self {
            requirement_id,
            minimum_evidence_class,
        }
    }

    pub fn requirement_id(&self) -> ContinuityRequirementId {
        self.requirement_id
    }

    pub fn minimum_evidence_class(&self) -> EvidenceClass {
        self.minimum_evidence_class
    }
}

/// Exact verifier policy for one contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationPolicyV1 {
    verifier_profile_id: String,
    entries: Vec<VerificationPolicyEntryV1>,
    policy_id: VerificationPolicyId,
}

impl VerificationPolicyV1 {
    pub fn new(
        verifier_profile_id: impl Into<String>,
        mut entries: Vec<VerificationPolicyEntryV1>,
    ) -> Result<Self, WitnessError> {
        let verifier_profile_id = checked_text("verifier_profile_id", verifier_profile_id.into())?;
        if entries.is_empty() {
            return Err(WitnessError::EmptyVerificationPolicy);
        }
        entries.sort_by_key(VerificationPolicyEntryV1::requirement_id);
        if entries
            .windows(2)
            .any(|pair| pair[0].requirement_id == pair[1].requirement_id)
        {
            return Err(WitnessError::DuplicatePolicyRequirement);
        }
        let policy_id = VerificationPolicyId(hash_policy(&verifier_profile_id, &entries));
        Ok(Self {
            verifier_profile_id,
            entries,
            policy_id,
        })
    }

    pub fn id(&self) -> VerificationPolicyId {
        self.policy_id
    }

    pub fn verifier_profile_id(&self) -> &str {
        &self.verifier_profile_id
    }

    pub fn entries(&self) -> &[VerificationPolicyEntryV1] {
        &self.entries
    }

    fn validate(&self) -> Result<(), WitnessError> {
        checked_text("verifier_profile_id", self.verifier_profile_id.clone())?;
        if self.entries.is_empty() {
            return Err(WitnessError::EmptyVerificationPolicy);
        }
        let mut canonical = self.entries.clone();
        canonical.sort_by_key(VerificationPolicyEntryV1::requirement_id);
        if canonical
            .windows(2)
            .any(|pair| pair[0].requirement_id == pair[1].requirement_id)
        {
            return Err(WitnessError::DuplicatePolicyRequirement);
        }
        if canonical != self.entries {
            return Err(WitnessError::NonCanonicalPolicyOrder);
        }
        let expected = VerificationPolicyId(hash_policy(&self.verifier_profile_id, &self.entries));
        if expected != self.policy_id {
            return Err(WitnessError::PolicyIdentityMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationObligationV1 {
    obligation_id: VerificationObligationId,
    requirement_id: ContinuityRequirementId,
    criticality: RequirementCriticality,
    minimum_evidence_class: EvidenceClass,
}

impl VerificationObligationV1 {
    pub fn id(&self) -> VerificationObligationId {
        self.obligation_id
    }

    pub fn requirement_id(&self) -> ContinuityRequirementId {
        self.requirement_id
    }

    pub fn criticality(&self) -> RequirementCriticality {
        self.criticality
    }

    pub fn minimum_evidence_class(&self) -> EvidenceClass {
        self.minimum_evidence_class
    }

    fn blocking(&self) -> bool {
        self.criticality == RequirementCriticality::Must
    }
}

/// Closed-world manifest derived from one validated contract, one exact target,
/// and one exact verifier policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessManifestV1 {
    contract_id: ContinuityContractId,
    source_snapshot_digest: [u8; 32],
    subject: String,
    target_realization_id: TargetRealizationId,
    verification_policy_id: VerificationPolicyId,
    obligations: Vec<VerificationObligationV1>,
    manifest_id: WitnessManifestId,
}

impl WitnessManifestV1 {
    pub fn new(
        contract: &ValidatedContinuityContractV1,
        target_realization_id: TargetRealizationId,
        policy: &VerificationPolicyV1,
    ) -> Result<Self, WitnessError> {
        policy.validate()?;
        let contract_ids: Vec<ContinuityRequirementId> =
            contract.requirements().iter().map(|r| r.id()).collect();
        let policy_ids: Vec<ContinuityRequirementId> =
            policy.entries().iter().map(|entry| entry.requirement_id()).collect();
        if contract_ids != policy_ids {
            return Err(WitnessError::PolicyContractMismatch);
        }

        let mut policy_by_requirement = BTreeMap::new();
        for entry in policy.entries() {
            policy_by_requirement.insert(entry.requirement_id(), entry.minimum_evidence_class());
        }

        let mut obligations = Vec::with_capacity(contract.requirements().len());
        for requirement in contract.requirements() {
            let minimum_evidence_class = policy_by_requirement
                .get(&requirement.id())
                .copied()
                .ok_or(WitnessError::PolicyContractMismatch)?;
            let obligation_id = VerificationObligationId(hash_obligation(
                contract.id(),
                target_realization_id,
                policy.id(),
                requirement.id(),
                requirement.criticality(),
                minimum_evidence_class,
            ));
            obligations.push(VerificationObligationV1 {
                obligation_id,
                requirement_id: requirement.id(),
                criticality: requirement.criticality(),
                minimum_evidence_class,
            });
        }
        obligations.sort_by_key(VerificationObligationV1::id);
        let manifest_id = WitnessManifestId(hash_manifest(
            contract.id(),
            contract.source_snapshot_digest(),
            contract.subject(),
            target_realization_id,
            policy.id(),
            &obligations,
        ));
        Ok(Self {
            contract_id: contract.id(),
            source_snapshot_digest: contract.source_snapshot_digest(),
            subject: contract.subject().to_string(),
            target_realization_id,
            verification_policy_id: policy.id(),
            obligations,
            manifest_id,
        })
    }

    pub fn id(&self) -> WitnessManifestId {
        self.manifest_id
    }

    pub fn contract_id(&self) -> ContinuityContractId {
        self.contract_id
    }

    pub fn source_snapshot_digest(&self) -> [u8; 32] {
        self.source_snapshot_digest
    }

    pub fn subject(&self) -> &str {
        &self.subject
    }

    pub fn target_realization_id(&self) -> TargetRealizationId {
        self.target_realization_id
    }

    pub fn verification_policy_id(&self) -> VerificationPolicyId {
        self.verification_policy_id
    }

    pub fn obligations(&self) -> &[VerificationObligationV1] {
        &self.obligations
    }

    fn obligation(&self, id: VerificationObligationId) -> Option<&VerificationObligationV1> {
        self.obligations
            .binary_search_by_key(&id, VerificationObligationV1::id)
            .ok()
            .map(|index| &self.obligations[index])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum ObligationDispositionV1 {
    Satisfied {
        evidence_digest: [u8; 32],
        evidence_class: EvidenceClass,
    },
    Failed {
        evidence_digest: [u8; 32],
    },
    Inconclusive {
        evidence_digest: [u8; 32],
    },
    InfrastructureFailure {
        evidence_digest: [u8; 32],
    },
    NotExecuted {
        evidence_digest: [u8; 32],
    },
}

impl ObligationDispositionV1 {
    fn evidence_digest(&self) -> [u8; 32] {
        match self {
            Self::Satisfied { evidence_digest, .. }
            | Self::Failed { evidence_digest }
            | Self::Inconclusive { evidence_digest }
            | Self::InfrastructureFailure { evidence_digest }
            | Self::NotExecuted { evidence_digest } => *evidence_digest,
        }
    }

    fn tag(&self) -> u8 {
        match self {
            Self::Satisfied { .. } => 1,
            Self::Failed { .. } => 2,
            Self::Inconclusive { .. } => 3,
            Self::InfrastructureFailure { .. } => 4,
            Self::NotExecuted { .. } => 5,
        }
    }

    fn evidence_class(&self) -> Option<EvidenceClass> {
        match self {
            Self::Satisfied { evidence_class, .. } => Some(*evidence_class),
            _ => None,
        }
    }
}

/// Append-only in-memory ledger for one manifest. V1 deliberately has no Serde
/// derive so callers cannot deserialize around duplicate/unknown-obligation checks.
#[derive(Debug, Clone)]
pub struct WitnessLedgerV1 {
    manifest: WitnessManifestV1,
    dispositions: BTreeMap<VerificationObligationId, ObligationDispositionV1>,
}

impl WitnessLedgerV1 {
    pub fn new(manifest: WitnessManifestV1) -> Self {
        Self {
            manifest,
            dispositions: BTreeMap::new(),
        }
    }

    pub fn manifest(&self) -> &WitnessManifestV1 {
        &self.manifest
    }

    pub fn record(
        &mut self,
        obligation_id: VerificationObligationId,
        disposition: ObligationDispositionV1,
    ) -> Result<(), WitnessError> {
        if self.manifest.obligation(obligation_id).is_none() {
            return Err(WitnessError::UnknownObligation);
        }
        if self.dispositions.contains_key(&obligation_id) {
            return Err(WitnessError::DuplicateDisposition);
        }
        if disposition.evidence_digest() == [0; 32] {
            return Err(WitnessError::ZeroDispositionEvidence);
        }
        self.dispositions.insert(obligation_id, disposition);
        Ok(())
    }

    pub fn disposition_count(&self) -> usize {
        self.dispositions.len()
    }

    pub fn finalize(self) -> Result<WitnessEvaluationV1, WitnessError> {
        if self.dispositions.len() != self.manifest.obligations.len() {
            return Err(WitnessError::IncompleteWitness {
                expected: self.manifest.obligations.len(),
                observed: self.dispositions.len(),
            });
        }

        let mut blocking_failures = 0usize;
        let mut nonblocking_issues = 0usize;
        for obligation in &self.manifest.obligations {
            let disposition = self
                .dispositions
                .get(&obligation.id())
                .ok_or(WitnessError::IncompleteWitness {
                    expected: self.manifest.obligations.len(),
                    observed: self.dispositions.len(),
                })?;
            let satisfied = matches!(
                disposition.evidence_class(),
                Some(class) if class >= obligation.minimum_evidence_class()
            );
            if !satisfied {
                if obligation.blocking() {
                    blocking_failures += 1;
                } else {
                    nonblocking_issues += 1;
                }
            }
        }

        let witness_id = WitnessId(hash_witness(
            self.manifest.id(),
            &self.manifest.obligations,
            &self.dispositions,
        ));
        Ok(WitnessEvaluationV1 {
            manifest: self.manifest,
            dispositions: self.dispositions,
            witness_id,
            blocking_failures,
            nonblocking_issues,
        })
    }
}

#[derive(Debug, Clone)]
pub struct WitnessEvaluationV1 {
    manifest: WitnessManifestV1,
    dispositions: BTreeMap<VerificationObligationId, ObligationDispositionV1>,
    witness_id: WitnessId,
    blocking_failures: usize,
    nonblocking_issues: usize,
}

impl WitnessEvaluationV1 {
    pub fn id(&self) -> WitnessId {
        self.witness_id
    }

    pub fn manifest(&self) -> &WitnessManifestV1 {
        &self.manifest
    }

    pub fn blocking_failures(&self) -> usize {
        self.blocking_failures
    }

    pub fn nonblocking_issues(&self) -> usize {
        self.nonblocking_issues
    }

    pub fn dispositions(
        &self,
    ) -> &BTreeMap<VerificationObligationId, ObligationDispositionV1> {
        &self.dispositions
    }

    pub fn qualify(self) -> Result<QualifiedContinuityWitnessV1, WitnessError> {
        if self.blocking_failures != 0 {
            return Err(WitnessError::BlockingObligationsUnsatisfied {
                count: self.blocking_failures,
            });
        }
        Ok(QualifiedContinuityWitnessV1 { evaluation: self })
    }
}

/// Non-Serde verifier-owned value that can only arise after complete witness
/// finalization with every blocking obligation satisfied at its policy floor.
#[derive(Debug, Clone)]
pub struct QualifiedContinuityWitnessV1 {
    evaluation: WitnessEvaluationV1,
}

impl QualifiedContinuityWitnessV1 {
    pub fn id(&self) -> WitnessId {
        self.evaluation.id()
    }

    pub fn manifest_id(&self) -> WitnessManifestId {
        self.evaluation.manifest.id()
    }

    pub fn contract_id(&self) -> ContinuityContractId {
        self.evaluation.manifest.contract_id()
    }

    pub fn target_realization_id(&self) -> TargetRealizationId {
        self.evaluation.manifest.target_realization_id()
    }

    pub fn nonblocking_issues(&self) -> usize {
        self.evaluation.nonblocking_issues()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum WitnessError {
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("target realization digest must be non-zero")]
    ZeroTargetRealization,
    #[error("verification policy must contain at least one entry")]
    EmptyVerificationPolicy,
    #[error("verification policy contains duplicate requirement identity")]
    DuplicatePolicyRequirement,
    #[error("verification policy entries must be in canonical requirement order")]
    NonCanonicalPolicyOrder,
    #[error("stored verification policy identity does not match canonical fields")]
    PolicyIdentityMismatch,
    #[error("verification policy requirement set does not exactly match contract")]
    PolicyContractMismatch,
    #[error("witness disposition references an obligation outside the manifest")]
    UnknownObligation,
    #[error("witness obligation already has a disposition")]
    DuplicateDisposition,
    #[error("witness disposition requires non-zero evidence digest")]
    ZeroDispositionEvidence,
    #[error("witness is incomplete: expected {expected} dispositions, observed {observed}")]
    IncompleteWitness { expected: usize, observed: usize },
    #[error("{count} blocking continuity obligations remain unsatisfied")]
    BlockingObligationsUnsatisfied { count: usize },
}

fn checked_text(field: &'static str, value: String) -> Result<String, WitnessError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(WitnessError::BlankText { field });
    }
    if trimmed.len() > 1024 {
        return Err(WitnessError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(WitnessError::ControlCharacters { field });
    }
    Ok(trimmed.to_string())
}

fn hash_policy(
    verifier_profile_id: &str,
    entries: &[VerificationPolicyEntryV1],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, verifier_profile_id);
    put_len(&mut bytes, entries.len());
    for entry in entries {
        bytes.extend_from_slice(entry.requirement_id().as_bytes());
        bytes.push(entry.minimum_evidence_class().tag());
    }
    domain_hash(POLICY_DOMAIN, &bytes)
}

fn hash_obligation(
    contract_id: ContinuityContractId,
    target_id: TargetRealizationId,
    policy_id: VerificationPolicyId,
    requirement_id: ContinuityRequirementId,
    criticality: RequirementCriticality,
    minimum_evidence_class: EvidenceClass,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(contract_id.as_bytes());
    bytes.extend_from_slice(target_id.as_bytes());
    bytes.extend_from_slice(policy_id.as_bytes());
    bytes.extend_from_slice(requirement_id.as_bytes());
    bytes.push(match criticality {
        RequirementCriticality::Must => 1,
        RequirementCriticality::Should => 2,
        RequirementCriticality::Optional => 3,
    });
    bytes.push(minimum_evidence_class.tag());
    domain_hash(OBLIGATION_DOMAIN, &bytes)
}

fn hash_manifest(
    contract_id: ContinuityContractId,
    source_snapshot_digest: [u8; 32],
    subject: &str,
    target_id: TargetRealizationId,
    policy_id: VerificationPolicyId,
    obligations: &[VerificationObligationV1],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(contract_id.as_bytes());
    bytes.extend_from_slice(&source_snapshot_digest);
    put_str(&mut bytes, subject);
    bytes.extend_from_slice(target_id.as_bytes());
    bytes.extend_from_slice(policy_id.as_bytes());
    put_len(&mut bytes, obligations.len());
    for obligation in obligations {
        bytes.extend_from_slice(obligation.id().as_bytes());
    }
    domain_hash(MANIFEST_DOMAIN, &bytes)
}

fn hash_witness(
    manifest_id: WitnessManifestId,
    obligations: &[VerificationObligationV1],
    dispositions: &BTreeMap<VerificationObligationId, ObligationDispositionV1>,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(manifest_id.as_bytes());
    put_len(&mut bytes, obligations.len());
    for obligation in obligations {
        let disposition = dispositions
            .get(&obligation.id())
            .expect("finalized witness contains every manifest obligation");
        bytes.extend_from_slice(obligation.id().as_bytes());
        bytes.push(disposition.tag());
        bytes.extend_from_slice(&disposition.evidence_digest());
        match disposition.evidence_class() {
            Some(class) => {
                bytes.push(1);
                bytes.push(class.tag());
            }
            None => bytes.push(0),
        }
    }
    domain_hash(WITNESS_DOMAIN, &bytes)
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
    use crate::contract::{
        ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, EquivalencePredicate,
    };
    use crate::observation::{
        DependencyBasis, DependencyClaimV1, EvidenceBasis, ObservationCoverage,
        ObservationEnvelopeV1,
    };

    fn contract_with_criticalities(
        criticalities: &[RequirementCriticality],
    ) -> ValidatedContinuityContractV1 {
        let mut requirements = Vec::new();
        for (index, criticality) in criticalities.iter().copied().enumerate() {
            let seed = (index + 1) as u8;
            let observation = ObservationEnvelopeV1::new(
                "machine-1",
                "workflow.dependency",
                "fixture",
                "1",
                1_700_000_000_000 + index as u64,
                ObservationCoverage::Complete,
                EvidenceBasis::Declared,
                [seed; 32],
                vec![],
            )
            .unwrap();
            let dependency = DependencyClaimV1::new(
                "role:test",
                "requires",
                format!("capability:{seed}"),
                DependencyBasis::Declared,
                vec![observation.id()],
                vec![],
            )
            .unwrap();
            requirements.push(
                ContinuityRequirementV1::new(
                    dependency.id(),
                    format!("capability-{seed}"),
                    criticality,
                    EquivalencePredicate::BehavioralScenario {
                        scenario_id: format!("scenario-{seed}"),
                    },
                    ApprovalBasis::ExplicitPolicy,
                    [seed.wrapping_add(20); 32],
                )
                .unwrap(),
            );
        }
        ContinuityContractV1::new("test-fleet", [44; 32], requirements)
            .unwrap()
            .validate()
            .unwrap()
    }

    fn policy(
        contract: &ValidatedContinuityContractV1,
        minimum: EvidenceClass,
    ) -> VerificationPolicyV1 {
        VerificationPolicyV1::new(
            "fixture-verifier-v1",
            contract
                .requirements()
                .iter()
                .map(|requirement| VerificationPolicyEntryV1::new(requirement.id(), minimum))
                .collect(),
        )
        .unwrap()
    }

    fn manifest(
        contract: &ValidatedContinuityContractV1,
        minimum: EvidenceClass,
    ) -> WitnessManifestV1 {
        WitnessManifestV1::new(
            contract,
            TargetRealizationId::from_digest([99; 32]).unwrap(),
            &policy(contract, minimum),
        )
        .unwrap()
    }

    #[test]
    fn policy_must_exactly_cover_contract_requirements() {
        let contract = contract_with_criticalities(&[
            RequirementCriticality::Must,
            RequirementCriticality::Must,
        ]);
        let incomplete = VerificationPolicyV1::new(
            "fixture",
            vec![VerificationPolicyEntryV1::new(
                contract.requirements()[0].id(),
                EvidenceClass::Simulated,
            )],
        )
        .unwrap();
        let result = WitnessManifestV1::new(
            &contract,
            TargetRealizationId::from_digest([1; 32]).unwrap(),
            &incomplete,
        );
        assert_eq!(result.unwrap_err(), WitnessError::PolicyContractMismatch);
    }

    #[test]
    fn incomplete_witness_refuses_finalization() {
        let contract = contract_with_criticalities(&[
            RequirementCriticality::Must,
            RequirementCriticality::Must,
        ]);
        let manifest = manifest(&contract, EvidenceClass::Simulated);
        let mut ledger = WitnessLedgerV1::new(manifest.clone());
        ledger
            .record(
                manifest.obligations()[0].id(),
                ObligationDispositionV1::Satisfied {
                    evidence_digest: [1; 32],
                    evidence_class: EvidenceClass::Simulated,
                },
            )
            .unwrap();
        assert!(matches!(
            ledger.finalize(),
            Err(WitnessError::IncompleteWitness { .. })
        ));
    }

    #[test]
    fn failed_must_obligation_prevents_qualification() {
        let contract = contract_with_criticalities(&[RequirementCriticality::Must]);
        let manifest = manifest(&contract, EvidenceClass::Simulated);
        let mut ledger = WitnessLedgerV1::new(manifest.clone());
        ledger
            .record(
                manifest.obligations()[0].id(),
                ObligationDispositionV1::Failed {
                    evidence_digest: [2; 32],
                },
            )
            .unwrap();
        let evaluation = ledger.finalize().unwrap();
        assert_eq!(evaluation.blocking_failures(), 1);
        assert!(matches!(
            evaluation.qualify(),
            Err(WitnessError::BlockingObligationsUnsatisfied { count: 1 })
        ));
    }

    #[test]
    fn insufficient_evidence_class_prevents_qualification() {
        let contract = contract_with_criticalities(&[RequirementCriticality::Must]);
        let manifest = manifest(&contract, EvidenceClass::HardwareVerified);
        let mut ledger = WitnessLedgerV1::new(manifest.clone());
        ledger
            .record(
                manifest.obligations()[0].id(),
                ObligationDispositionV1::Satisfied {
                    evidence_digest: [3; 32],
                    evidence_class: EvidenceClass::Simulated,
                },
            )
            .unwrap();
        let evaluation = ledger.finalize().unwrap();
        assert_eq!(evaluation.blocking_failures(), 1);
    }

    #[test]
    fn nonblocking_failure_remains_visible_but_can_qualify() {
        let contract = contract_with_criticalities(&[
            RequirementCriticality::Must,
            RequirementCriticality::Should,
        ]);
        let manifest = manifest(&contract, EvidenceClass::Simulated);
        let mut ledger = WitnessLedgerV1::new(manifest.clone());
        for obligation in manifest.obligations() {
            let disposition = if obligation.criticality() == RequirementCriticality::Must {
                ObligationDispositionV1::Satisfied {
                    evidence_digest: [4; 32],
                    evidence_class: EvidenceClass::Simulated,
                }
            } else {
                ObligationDispositionV1::Inconclusive {
                    evidence_digest: [5; 32],
                }
            };
            ledger.record(obligation.id(), disposition).unwrap();
        }
        let evaluation = ledger.finalize().unwrap();
        assert_eq!(evaluation.blocking_failures(), 0);
        assert_eq!(evaluation.nonblocking_issues(), 1);
        let qualified = evaluation.qualify().unwrap();
        assert_eq!(qualified.nonblocking_issues(), 1);
    }

    #[test]
    fn exact_target_participates_in_manifest_identity() {
        let contract = contract_with_criticalities(&[RequirementCriticality::Must]);
        let policy = policy(&contract, EvidenceClass::Simulated);
        let left = WitnessManifestV1::new(
            &contract,
            TargetRealizationId::from_digest([10; 32]).unwrap(),
            &policy,
        )
        .unwrap();
        let right = WitnessManifestV1::new(
            &contract,
            TargetRealizationId::from_digest([11; 32]).unwrap(),
            &policy,
        )
        .unwrap();
        assert_ne!(left.id(), right.id());
    }

    #[test]
    fn complete_sufficient_must_evidence_yields_qualified_witness() {
        let contract = contract_with_criticalities(&[RequirementCriticality::Must]);
        let manifest = manifest(&contract, EvidenceClass::DifferentiallyVerified);
        let mut ledger = WitnessLedgerV1::new(manifest.clone());
        ledger
            .record(
                manifest.obligations()[0].id(),
                ObligationDispositionV1::Satisfied {
                    evidence_digest: [12; 32],
                    evidence_class: EvidenceClass::HardwareVerified,
                },
            )
            .unwrap();
        let qualified = ledger.finalize().unwrap().qualify().unwrap();
        assert_eq!(qualified.contract_id(), contract.id());
        assert_eq!(qualified.manifest_id(), manifest.id());
    }
}
