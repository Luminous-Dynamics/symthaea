// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact typed-subject binding for validated continuity contracts.
//!
//! The historical v1 continuity contract carries a human-readable `subject` label.
//! That label remains useful for compatibility and audit display, but it is not a
//! sufficient distributed-infrastructure identity: the same label can name a
//! machine, service, cluster, network device, or fabric in different namespaces.
//!
//! This module adds a constructor-only, non-Serde binding between one already
//! validated continuity contract and one exact [`ContinuitySubjectV1`]. It does not
//! change the serialized v1 contract schema and grants no migration or execution
//! authority.
//!
//! Core theorem:
//!
//! `ValidatedContinuityContractV1 != SubjectBoundContinuityContractV1 != ExecutionAuthority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::contract::{ContinuityContractId, ValidatedContinuityContractV1};
use crate::scope::{ContinuitySubjectError, ContinuitySubjectId, ContinuitySubjectV1};

const SUBJECT_CONTRACT_BINDING_DOMAIN: &[u8] =
    b"symthaea.continuity.subject-contract-binding.v1\0";

/// Stable content identity of one exact typed-subject / validated-contract pair.
///
/// This identity is serializable reference material only. Possessing or
/// deserializing it does not reconstruct the non-Serde validated binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SubjectBoundContinuityContractId([u8; 32]);

impl SubjectBoundContinuityContractId {
    /// Raw BLAKE3-256 identity bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Constructor-qualified binding between an exact typed continuity subject and an
/// already validated continuity contract.
///
/// This wrapper deliberately implements neither `Serialize` nor `Deserialize`.
/// Transport bytes may describe a subject or a raw contract, but callers must
/// independently validate both layers and construct this value locally before a
/// future witness path may rely on the exact subject relation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubjectBoundContinuityContractV1 {
    subject: ContinuitySubjectV1,
    contract: ValidatedContinuityContractV1,
    binding_id: SubjectBoundContinuityContractId,
}

impl SubjectBoundContinuityContractV1 {
    /// Bind one exact typed subject to one validated contract.
    ///
    /// V1 retains a compatibility ratchet: the historical free-form contract
    /// subject label must equal the typed subject's canonical `logical_id` exactly.
    /// A caller cannot silently reinterpret a legacy `db-primary` contract as
    /// `payments-cluster`, nor can whitespace/case drift create an implicit alias.
    pub fn bind(
        subject: ContinuitySubjectV1,
        contract: ValidatedContinuityContractV1,
    ) -> Result<Self, SubjectContractBindingError> {
        subject.validate()?;
        require_legacy_label_match(&subject, &contract)?;

        let binding_id = SubjectBoundContinuityContractId(hash_binding(
            subject.id(),
            contract.id(),
        ));

        Ok(Self {
            subject,
            contract,
            binding_id,
        })
    }

    /// Recheck the exact subject identity, compatibility label, and stored binding
    /// identity. This remains validation of content relationships, not ownership,
    /// authentication, currentness, or execution permission.
    pub fn validate(&self) -> Result<(), SubjectContractBindingError> {
        self.subject.validate()?;
        require_legacy_label_match(&self.subject, &self.contract)?;

        let expected = SubjectBoundContinuityContractId(hash_binding(
            self.subject.id(),
            self.contract.id(),
        ));
        if expected != self.binding_id {
            return Err(SubjectContractBindingError::BindingIdentityMismatch);
        }
        Ok(())
    }

    /// Exact typed subject / contract binding identity.
    pub fn id(&self) -> SubjectBoundContinuityContractId {
        self.binding_id
    }

    /// Exact typed subject identity.
    pub fn subject_id(&self) -> ContinuitySubjectId {
        self.subject.id()
    }

    /// Exact validated contract identity.
    pub fn contract_id(&self) -> ContinuityContractId {
        self.contract.id()
    }

    /// Exact typed continuity subject.
    pub fn subject(&self) -> &ContinuitySubjectV1 {
        &self.subject
    }

    /// Exact validated continuity contract.
    pub fn contract(&self) -> &ValidatedContinuityContractV1 {
        &self.contract
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SubjectContractBindingError {
    #[error(transparent)]
    Subject(#[from] ContinuitySubjectError),
    #[error(
        "legacy contract subject label {contract_subject:?} does not exactly match typed subject logical_id {subject_logical_id:?}"
    )]
    LegacySubjectLabelMismatch {
        contract_subject: String,
        subject_logical_id: String,
    },
    #[error("stored subject-bound continuity contract identity does not match exact subject and contract")]
    BindingIdentityMismatch,
}

fn require_legacy_label_match(
    subject: &ContinuitySubjectV1,
    contract: &ValidatedContinuityContractV1,
) -> Result<(), SubjectContractBindingError> {
    if contract.subject() != subject.logical_id() {
        return Err(SubjectContractBindingError::LegacySubjectLabelMismatch {
            contract_subject: contract.subject().to_owned(),
            subject_logical_id: subject.logical_id().to_owned(),
        });
    }
    Ok(())
}

fn hash_binding(subject_id: ContinuitySubjectId, contract_id: ContinuityContractId) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SUBJECT_CONTRACT_BINDING_DOMAIN);
    hasher.update(subject_id.as_bytes());
    hasher.update(contract_id.as_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{
        ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, EquivalencePredicate,
        RequirementCriticality,
    };
    use crate::observation::{
        DependencyBasis, DependencyClaimV1, EvidenceBasis, ObservationCoverage,
        ObservationEnvelopeV1,
    };
    use crate::scope::ContinuityScopeV1;

    fn contract(label: &str, seed: u8) -> ValidatedContinuityContractV1 {
        let observation = ObservationEnvelopeV1::new(
            format!("source-{label}"),
            "continuity.dependency",
            "fixture",
            "1",
            1_700_000_000_000 + seed as u64,
            ObservationCoverage::Complete,
            EvidenceBasis::Tested,
            [seed; 32],
            vec![],
        )
        .unwrap();
        let dependency = DependencyClaimV1::new(
            format!("role:{label}"),
            "requires",
            format!("capability:{seed}"),
            DependencyBasis::Declared,
            vec![observation.id()],
            vec![],
        )
        .unwrap();
        let requirement = ContinuityRequirementV1::new(
            dependency.id(),
            format!("workflow-{seed}"),
            RequirementCriticality::Must,
            EquivalencePredicate::BehavioralScenario {
                scenario_id: format!("scenario-{seed}"),
            },
            ApprovalBasis::ExplicitPolicy,
            [seed.wrapping_add(1); 32],
        )
        .unwrap();

        ContinuityContractV1::new(label, [seed.wrapping_add(2); 32], vec![requirement])
            .unwrap()
            .validate()
            .unwrap()
    }

    fn subject(
        namespace: &str,
        logical_id: &str,
        scope: ContinuityScopeV1,
        parent: Option<ContinuitySubjectId>,
    ) -> ContinuitySubjectV1 {
        ContinuitySubjectV1::new(namespace, logical_id, scope, parent).unwrap()
    }

    #[test]
    fn exact_subject_and_contract_bind_deterministically() {
        let contract = contract("payments", 1);
        let subject = subject("org.example", "payments", ContinuityScopeV1::Service, None);

        let a = SubjectBoundContinuityContractV1::bind(subject.clone(), contract.clone()).unwrap();
        let b = SubjectBoundContinuityContractV1::bind(subject, contract).unwrap();

        assert_eq!(a.id(), b.id());
        assert_eq!(a.subject_id(), b.subject_id());
        assert_eq!(a.contract_id(), b.contract_id());
        a.validate().unwrap();
    }

    #[test]
    fn legacy_label_mismatch_fails_closed() {
        let contract = contract("db-primary", 2);
        let subject = subject("org.example", "db-replica", ContinuityScopeV1::Machine, None);

        assert!(matches!(
            SubjectBoundContinuityContractV1::bind(subject, contract),
            Err(SubjectContractBindingError::LegacySubjectLabelMismatch { .. })
        ));
    }

    #[test]
    fn same_label_different_scope_produces_different_binding() {
        let contract = contract("edge-01", 3);
        let machine = subject("org.example", "edge-01", ContinuityScopeV1::Machine, None);
        let device = subject(
            "org.example",
            "edge-01",
            ContinuityScopeV1::NetworkDevice,
            None,
        );

        let machine_bound =
            SubjectBoundContinuityContractV1::bind(machine, contract.clone()).unwrap();
        let device_bound = SubjectBoundContinuityContractV1::bind(device, contract).unwrap();

        assert_ne!(machine_bound.subject_id(), device_bound.subject_id());
        assert_ne!(machine_bound.id(), device_bound.id());
    }

    #[test]
    fn same_label_and_scope_different_namespace_produces_different_binding() {
        let contract = contract("payments", 4);
        let a = subject("org.a", "payments", ContinuityScopeV1::Service, None);
        let b = subject("org.b", "payments", ContinuityScopeV1::Service, None);

        let a_bound = SubjectBoundContinuityContractV1::bind(a, contract.clone()).unwrap();
        let b_bound = SubjectBoundContinuityContractV1::bind(b, contract).unwrap();

        assert_ne!(a_bound.subject_id(), b_bound.subject_id());
        assert_ne!(a_bound.id(), b_bound.id());
    }

    #[test]
    fn parent_context_is_preserved_in_contract_binding() {
        let contract = contract("leaf-01", 5);
        let fabric_a = subject(
            "org.example",
            "fabric-a",
            ContinuityScopeV1::NetworkFabric,
            None,
        );
        let fabric_b = subject(
            "org.example",
            "fabric-b",
            ContinuityScopeV1::NetworkFabric,
            None,
        );
        let leaf_a = subject(
            "org.example",
            "leaf-01",
            ContinuityScopeV1::NetworkDevice,
            Some(fabric_a.id()),
        );
        let leaf_b = subject(
            "org.example",
            "leaf-01",
            ContinuityScopeV1::NetworkDevice,
            Some(fabric_b.id()),
        );

        let a = SubjectBoundContinuityContractV1::bind(leaf_a, contract.clone()).unwrap();
        let b = SubjectBoundContinuityContractV1::bind(leaf_b, contract).unwrap();

        assert_ne!(a.subject_id(), b.subject_id());
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn contract_identity_is_part_of_binding_identity() {
        let subject = subject("org.example", "cluster-a", ContinuityScopeV1::Cluster, None);
        let a = SubjectBoundContinuityContractV1::bind(subject.clone(), contract("cluster-a", 6))
            .unwrap();
        let b = SubjectBoundContinuityContractV1::bind(subject, contract("cluster-a", 7)).unwrap();

        assert_ne!(a.contract_id(), b.contract_id());
        assert_ne!(a.id(), b.id());
    }
}
