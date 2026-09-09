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

/// Compose one qualified witness exclusively from verifier-owned authenticated
/// evidence. No caller-supplied evidence class crosses this boundary.
pub(crate) fn compose_qualified_witness(
    contract: &ValidatedContinuityContractV1,
    target: TargetRealizationId,
    policy: &VerificationPolicyV1,
    evidence: Vec<AuthenticatedVerificationEvidenceV1>,
) -> Result<QualifiedContinuityWitnessV1, ComposeError> {
    let manifest = WitnessManifestV1::new(contract, target, policy)?;
    let mut by_requirement = BTreeMap::new();
    for obligation in manifest.obligations() {
        by_requirement.insert(obligation.requirement_id(), obligation.id());
    }

    let mut seen = BTreeSet::new();
    let mut ledger = WitnessLedgerV1::new(manifest);
    for item in evidence {
        let requirement_id = item.requirement_id();
        if !seen.insert(requirement_id) {
            return Err(ComposeError::DuplicateRequirementEvidence {
                requirement: requirement_id,
            });
        }
        let obligation_id = by_requirement
            .get(&requirement_id)
            .copied()
            .ok_or(ComposeError::UnknownRequirementEvidence {
                requirement: requirement_id,
            })?;
        ledger.record(obligation_id, item.disposition())?;
    }

    Ok(ledger.finalize()?.qualify()?)
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum ComposeError {
    #[error(transparent)]
    Witness(#[from] WitnessError),
    #[error("authenticated evidence references requirement outside the validated contract: {requirement:?}")]
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
    use crate::verifier::{
        policy_check_verification_evidence, AuthenticatedVerificationEvidenceV1,
        VerificationEvidenceClaimV1, VerificationOutcomeV1, VerifierProfileV1,
    };
    use crate::witness::{EvidenceClass, VerificationPolicyEntryV1};

    fn contract() -> ValidatedContinuityContractV1 {
        let observation = ObservationEnvelopeV1::new(
            "machine-1",
            "workflow.dependency",
            "fixture",
            "1",
            1_700_000_000_000,
            ObservationCoverage::Complete,
            EvidenceBasis::Tested,
            [1; 32],
            vec![],
        )
        .unwrap();
        let dependency = DependencyClaimV1::new(
            "role:test",
            "requires",
            "capability:test",
            DependencyBasis::Declared,
            vec![observation.id()],
            vec![],
        )
        .unwrap();
        let requirement = ContinuityRequirementV1::new(
            dependency.id(),
            "test-workflow",
            RequirementCriticality::Must,
            EquivalencePredicate::BehavioralScenario {
                scenario_id: "scenario-v1".into(),
            },
            ApprovalBasis::ExplicitPolicy,
            [2; 32],
        )
        .unwrap();
        ContinuityContractV1::new("test-fleet", [3; 32], vec![requirement])
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
                .map(|requirement| {
                    VerificationPolicyEntryV1::new(requirement.id(), EvidenceClass::Simulated)
                })
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
    fn verifier_owned_evidence_can_qualify_witness() {
        let contract = contract();
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
        let contract = contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        assert!(matches!(
            compose_qualified_witness(&contract, target, &policy(&contract), vec![]),
            Err(ComposeError::Witness(WitnessError::IncompleteWitness { .. }))
        ));
    }

    #[test]
    fn duplicate_authenticated_requirement_fails_closed() {
        let contract = contract();
        let target = TargetRealizationId::from_digest([4; 32]).unwrap();
        let profile = profile();
        let first = authenticated(&contract, target, &profile, [5; 32], [6; 32]);
        let second = authenticated(&contract, target, &profile, [5; 32], [7; 32]);
        assert!(matches!(
            compose_qualified_witness(
                &contract,
                target,
                &policy(&contract),
                vec![first, second],
            ),
            Err(ComposeError::DuplicateRequirementEvidence { .. })
        ));
    }
}
