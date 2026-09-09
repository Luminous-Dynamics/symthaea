// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Internal composition of verifier-owned evidence into a closed-world witness.

use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::contract::ContinuityRequirementId;
use crate::subject_contract::SubjectBoundContinuityContractV1;
use crate::subject_witness::{
    SubjectBoundQualifiedContinuityWitnessV1, SubjectWitnessBindingError,
};
use crate::verifier::AuthenticatedVerificationEvidenceV1;
use crate::witness::{TargetRealizationId, VerificationPolicyV1, WitnessError, WitnessLedgerV1, WitnessManifestV1};

pub(crate) fn compose_qualified_witness(
    subject_contract: &SubjectBoundContinuityContractV1,
    target: TargetRealizationId,
    policy: &VerificationPolicyV1,
    evidence: Vec<AuthenticatedVerificationEvidenceV1>,
) -> Result<SubjectBoundQualifiedContinuityWitnessV1, ComposeError> {
    subject_contract.validate()?;
    let contract = subject_contract.contract();
    let manifest = WitnessManifestV1::new(contract, target, policy)?;
    let mut obligations = BTreeMap::new();
    for obligation in manifest.obligations() {
        obligations.insert(obligation.requirement_id(), obligation.id());
    }

    let mut seen = BTreeSet::new();
    let mut ledger = WitnessLedgerV1::new(manifest);
    for item in evidence {
        if item.contract_id() != contract.id() {
            return Err(ComposeError::CrossContractEvidence);
        }
        if item.target_realization_id() != target {
            return Err(ComposeError::CrossTargetEvidence);
        }
        let requirement = item.requirement_id();
        if !seen.insert(requirement) {
            return Err(ComposeError::DuplicateRequirementEvidence { requirement });
        }
        let obligation = obligations
            .get(&requirement)
            .copied()
            .ok_or(ComposeError::UnknownRequirementEvidence { requirement })?;
        ledger.record(obligation, item.disposition())?;
    }

    let witness = ledger.finalize()?.qualify()?;
    Ok(SubjectBoundQualifiedContinuityWitnessV1::from_qualified(
        subject_contract.clone(),
        witness,
    )?)
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum ComposeError {
    #[error(transparent)]
    Witness(#[from] WitnessError),
    #[error(transparent)]
    SubjectContract(#[from] crate::subject_contract::SubjectContractBindingError),
    #[error(transparent)]
    SubjectWitness(#[from] SubjectWitnessBindingError),
    #[error("authenticated evidence belongs to a different continuity contract")]
    CrossContractEvidence,
    #[error("authenticated evidence belongs to a different target realization")]
    CrossTargetEvidence,
    #[error("authenticated evidence references requirement outside the validated contract: {requirement:?}")]
    UnknownRequirementEvidence { requirement: ContinuityRequirementId },
    #[error("requirement received more than one authenticated evidence item: {requirement:?}")]
    DuplicateRequirementEvidence { requirement: ContinuityRequirementId },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{
        ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, EquivalencePredicate,
        RequirementCriticality, ValidatedContinuityContractV1,
    };
    use crate::observation::{
        DependencyBasis, DependencyClaimV1, EvidenceBasis, ObservationCoverage,
        ObservationEnvelopeV1,
    };
    use crate::scope::{ContinuityScopeV1, ContinuitySubjectV1};
    use crate::subject_contract::SubjectBoundContinuityContractV1;
    use crate::verifier::{
        AuthenticatedVerificationEvidenceV1, VerificationEvidenceClaimV1, VerificationOutcomeV1,
        VerifierProfileV1, policy_check_verification_evidence,
    };
    use crate::witness::{EvidenceClass, VerificationPolicyEntryV1};

    fn contract(seed: u8) -> ValidatedContinuityContractV1 {
        let observation = ObservationEnvelopeV1::new(
            format!("machine-{seed}"),
            "workflow.dependency",
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
            "role:test",
            "requires",
            format!("capability-{seed}"),
            DependencyBasis::Declared,
            vec![observation.id()],
            vec![],
        )
        .unwrap();
        let requirement = ContinuityRequirementV1::new(
            dependency.id(),
            format!("test-workflow-{seed}"),
            RequirementCriticality::Must,
            EquivalencePredicate::BehavioralScenario {
                scenario_id: format!("scenario-{seed}"),
            },
            ApprovalBasis::ExplicitPolicy,
            [seed.wrapping_add(20); 32],
        )
        .unwrap();
        ContinuityContractV1::new(
            format!("test-fleet-{seed}"),
            [seed.wrapping_add(40); 32],
            vec![requirement],
        )
        .unwrap()
        .validate()
        .unwrap()
    }

    fn bound_contract(seed: u8) -> SubjectBoundContinuityContractV1 {
        let contract = contract(seed);
        let logical_id = contract.subject().to_owned();
        let subject = ContinuitySubjectV1::new(
            "org.example",
            logical_id,
            ContinuityScopeV1::Fleet,
            None,
        )
        .unwrap();
        SubjectBoundContinuityContractV1::bind(subject, contract).unwrap()
    }

    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "fixture-verifier",
            [8; 32],
            1,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    fn policy(contract: &ValidatedContinuityContractV1) -> VerificationPolicyV1 {
        VerificationPolicyV1::new(
            "fixture-verifier",
            contract
                .requirements()
                .iter()
                .map(|r| VerificationPolicyEntryV1::new(r.id(), EvidenceClass::Simulated))
                .collect(),
        )
        .unwrap()
    }

    fn authenticated(
        contract: &ValidatedContinuityContractV1,
        target: TargetRealizationId,
        profile: &VerifierProfileV1,
        challenge: [u8; 32],
        evidence_digest: [u8; 32],
    ) -> AuthenticatedVerificationEvidenceV1 {
        let policy = policy(contract);
        let claim = VerificationEvidenceClaimV1::new(
            contract.id(),
            target,
            contract.requirements()[0].id(),
            profile.id(),
            challenge,
            1_700_000_000_100,
            VerificationOutcomeV1::Satisfied,
            evidence_digest,
        )
        .unwrap();
        let checked = policy_check_verification_evidence(
            contract,
            target,
            &policy,
            profile,
            challenge,
            claim,
        )
        .unwrap();
        AuthenticatedVerificationEvidenceV1::authenticate_for_test(checked, [9; 32]).unwrap()
    }

    #[test]
    fn exact_subject_context_evidence_can_qualify_witness() {
        let bound = bound_contract(1);
        let contract = bound.contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let qualified = compose_qualified_witness(
            &bound,
            target,
            &policy(contract),
            vec![authenticated(contract, target, &profile, [5; 32], [6; 32])],
        )
        .unwrap();

        assert_eq!(qualified.contract_id(), contract.id());
        assert_eq!(qualified.subject_id(), bound.subject_id());
        assert_eq!(qualified.subject_contract_binding_id(), bound.id());
        assert_eq!(qualified.target_realization_id(), target);
        qualified.validate().unwrap();
    }

    #[test]
    fn missing_authenticated_evidence_remains_incomplete() {
        let bound = bound_contract(1);
        let contract = bound.contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        assert!(matches!(
            compose_qualified_witness(&bound, target, &policy(contract), vec![]),
            Err(ComposeError::Witness(WitnessError::IncompleteWitness { .. }))
        ));
    }

    #[test]
    fn duplicate_authenticated_requirement_fails_closed() {
        let bound = bound_contract(1);
        let contract = bound.contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let a = authenticated(contract, target, &profile, [5; 32], [6; 32]);
        let b = authenticated(contract, target, &profile, [5; 32], [7; 32]);
        assert!(matches!(
            compose_qualified_witness(&bound, target, &policy(contract), vec![a, b]),
            Err(ComposeError::DuplicateRequirementEvidence { .. })
        ));
    }

    #[test]
    fn authenticated_evidence_cannot_cross_target_context() {
        let bound = bound_contract(1);
        let contract = bound.contract();
        let target_a = TargetRealizationId::from_digest([4; 32]).unwrap();
        let target_b = TargetRealizationId::from_digest([5; 32]).unwrap();
        let profile = profile();
        let evidence = authenticated(contract, target_a, &profile, [6; 32], [7; 32]);
        assert_eq!(
            compose_qualified_witness(&bound, target_b, &policy(contract), vec![evidence])
                .unwrap_err(),
            ComposeError::CrossTargetEvidence,
        );
    }

    #[test]
    fn authenticated_evidence_cannot_cross_contract_context() {
        let bound_a = bound_contract(1);
        let bound_b = bound_contract(2);
        let contract_a = bound_a.contract();
        let contract_b = bound_b.contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let evidence = authenticated(contract_a, target, &profile, [6; 32], [7; 32]);
        assert_eq!(
            compose_qualified_witness(&bound_b, target, &policy(contract_b), vec![evidence])
                .unwrap_err(),
            ComposeError::CrossContractEvidence,
        );
    }

    #[test]
    fn same_legacy_contract_under_different_scope_cannot_collapse_subject_identity() {
        let contract = contract(3);
        let logical_id = contract.subject().to_owned();
        let machine = SubjectBoundContinuityContractV1::bind(
            ContinuitySubjectV1::new(
                "org.example",
                logical_id.clone(),
                ContinuityScopeV1::Machine,
                None,
            )
            .unwrap(),
            contract.clone(),
        )
        .unwrap();
        let cluster = SubjectBoundContinuityContractV1::bind(
            ContinuitySubjectV1::new(
                "org.example",
                logical_id,
                ContinuityScopeV1::Cluster,
                None,
            )
            .unwrap(),
            contract,
        )
        .unwrap();
        let target = TargetRealizationId::from_digest([11; 32]).unwrap();
        let profile = profile();
        let evidence_a = authenticated(machine.contract(), target, &profile, [12; 32], [13; 32]);
        let evidence_b = authenticated(cluster.contract(), target, &profile, [12; 32], [13; 32]);

        let machine_witness = compose_qualified_witness(
            &machine,
            target,
            &policy(machine.contract()),
            vec![evidence_a],
        )
        .unwrap();
        let cluster_witness = compose_qualified_witness(
            &cluster,
            target,
            &policy(cluster.contract()),
            vec![evidence_b],
        )
        .unwrap();

        assert_ne!(machine_witness.subject_id(), cluster_witness.subject_id());
        assert_ne!(machine_witness.id(), cluster_witness.id());
    }
}
