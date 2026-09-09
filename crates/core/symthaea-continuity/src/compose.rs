// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Internal composition of verifier-owned evidence into a closed-world witness.

use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::contract::{ContinuityRequirementId, ValidatedContinuityContractV1};
use crate::verifier::AuthenticatedVerificationEvidenceV1;
use crate::witness::{
    QualifiedContinuityWitnessV1, TargetRealizationId, VerificationPolicyV1, WitnessError,
    WitnessLedgerV1, WitnessManifestV1,
};

pub(crate) fn compose_qualified_witness(
    contract: &ValidatedContinuityContractV1,
    target: TargetRealizationId,
    policy: &VerificationPolicyV1,
    evidence: Vec<AuthenticatedVerificationEvidenceV1>,
) -> Result<QualifiedContinuityWitnessV1, ComposeError> {
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

    Ok(ledger.finalize()?.qualify()?)
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum ComposeError {
    #[error(transparent)]
    Witness(#[from] WitnessError),
    #[error("authenticated evidence belongs to a different continuity contract")]
    CrossContractEvidence,
    #[error("authenticated evidence belongs to a different target realization")]
    CrossTargetEvidence,
    #[error(
        "authenticated evidence references requirement outside the validated contract: {requirement:?}"
    )]
    UnknownRequirementEvidence {
        requirement: ContinuityRequirementId,
    },
    #[error("requirement received more than one authenticated evidence item: {requirement:?}")]
    DuplicateRequirementEvidence {
        requirement: ContinuityRequirementId,
    },
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
    use crate::profile_adoption::{
        VerifierAdoptionScopeV1, VerifierProfileAdoptionSubjectV1,
        VerifierProfileAdoptionTransitionV1,
    };
    use crate::profile_adoption_root::VerifierProfileAdoptionAuthorityRootSnapshotV1;
    use crate::verifier::{
        AuthenticatedVerificationEvidenceV1, AuthorityCheckedVerificationEvidenceV1,
        VerificationEvidenceClaimV1, VerificationOutcomeV1, VerifierProfileV1,
        policy_check_verification_evidence,
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
        let profile_checked = policy_check_verification_evidence(
            contract, target, &policy, profile, challenge, claim,
        )
        .unwrap();

        let adoption = VerifierProfileAdoptionTransitionV1::bootstrap(
            VerifierProfileAdoptionSubjectV1::new(
                "fixture-adoption",
                "organization:test",
                "adoption-root-1",
                [0x55; 32],
                profile,
                1,
                1_600_000_000_000,
                1_800_000_000_000,
                EvidenceClass::HardwareVerified,
                VerifierAdoptionScopeV1::AllContinuityVerification,
            )
            .unwrap(),
        )
        .unwrap();
        let root_snapshot = VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            9,
        )
        .unwrap();
        let authority_checked = AuthorityCheckedVerificationEvidenceV1::authorize_for_test(
            profile_checked,
            &adoption,
            &root_snapshot,
        )
        .unwrap();

        AuthenticatedVerificationEvidenceV1::authenticate_for_test(authority_checked, [9; 32])
            .unwrap()
    }

    #[test]
    fn exact_context_evidence_can_qualify_witness() {
        let contract = contract(1);
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let qualified = compose_qualified_witness(
            &contract,
            target,
            &policy(&contract),
            vec![authenticated(&contract, target, &profile, [5; 32], [6; 32])],
        )
        .unwrap();
        assert_eq!(qualified.contract_id(), contract.id());
        assert_eq!(qualified.target_realization_id(), target);
    }

    #[test]
    fn missing_authenticated_evidence_remains_incomplete() {
        let contract = contract(1);
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        assert!(matches!(
            compose_qualified_witness(&contract, target, &policy(&contract), vec![]),
            Err(ComposeError::Witness(
                WitnessError::IncompleteWitness { .. }
            ))
        ));
    }

    #[test]
    fn duplicate_authenticated_requirement_fails_closed() {
        let contract = contract(1);
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let a = authenticated(&contract, target, &profile, [5; 32], [6; 32]);
        let b = authenticated(&contract, target, &profile, [5; 32], [7; 32]);
        assert!(matches!(
            compose_qualified_witness(&contract, target, &policy(&contract), vec![a, b]),
            Err(ComposeError::DuplicateRequirementEvidence { .. })
        ));
    }

    #[test]
    fn authenticated_evidence_cannot_cross_target_context() {
        let contract = contract(1);
        let target_a = TargetRealizationId::from_digest([4; 32]).unwrap();
        let target_b = TargetRealizationId::from_digest([5; 32]).unwrap();
        let profile = profile();
        let evidence = authenticated(&contract, target_a, &profile, [6; 32], [7; 32]);
        assert_eq!(
            compose_qualified_witness(&contract, target_b, &policy(&contract), vec![evidence])
                .unwrap_err(),
            ComposeError::CrossTargetEvidence,
        );
    }

    #[test]
    fn authenticated_evidence_cannot_cross_contract_context() {
        let contract_a = contract(1);
        let contract_b = contract(2);
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let evidence = authenticated(&contract_a, target, &profile, [6; 32], [7; 32]);
        assert_eq!(
            compose_qualified_witness(&contract_b, target, &policy(&contract_b), vec![evidence])
                .unwrap_err(),
            ComposeError::CrossContractEvidence,
        );
    }
}
