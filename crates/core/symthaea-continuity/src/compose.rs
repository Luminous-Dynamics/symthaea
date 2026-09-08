// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public safe composition surface for continuity witnesses.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::contract::{ContinuityRequirementId, ValidatedContinuityContractV1};
use crate::witness::{
    ObligationDispositionV1, QualifiedContinuityWitnessV1, TargetRealizationId,
    VerificationPolicyV1, WitnessError, WitnessLedgerV1, WitnessManifestV1,
};

/// One caller-supplied disposition keyed by the contract requirement it addresses.
///
/// Callers never construct witness obligations or manifests directly. The verifier
/// derives those from the validated contract, target, and verification policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuityEvidenceSubmissionV1 {
    requirement_id: ContinuityRequirementId,
    disposition: ObligationDispositionV1,
}

impl ContinuityEvidenceSubmissionV1 {
    pub fn new(
        requirement_id: ContinuityRequirementId,
        disposition: ObligationDispositionV1,
    ) -> Self {
        Self {
            requirement_id,
            disposition,
        }
    }

    pub fn requirement_id(&self) -> ContinuityRequirementId {
        self.requirement_id
    }

    pub fn disposition(&self) -> &ObligationDispositionV1 {
        &self.disposition
    }
}

/// Compose a qualified continuity witness without exposing a deserializable
/// manifest/ledger construction path to external callers.
///
/// The function derives the closed-world obligation set internally and requires
/// submitted requirement IDs to be unique and known. Missing submissions remain
/// visible to `WitnessLedgerV1::finalize()` as an incomplete witness.
pub fn compose_qualified_witness(
    contract: &ValidatedContinuityContractV1,
    target: TargetRealizationId,
    policy: &VerificationPolicyV1,
    submissions: Vec<ContinuityEvidenceSubmissionV1>,
) -> Result<QualifiedContinuityWitnessV1, ComposeError> {
    let manifest = WitnessManifestV1::new(contract, target, policy)?;
    let mut by_requirement = BTreeMap::new();
    for obligation in manifest.obligations() {
        by_requirement.insert(obligation.requirement_id(), obligation.id());
    }

    let mut seen = BTreeSet::new();
    let mut ledger = WitnessLedgerV1::new(manifest);
    for submission in submissions {
        if !seen.insert(submission.requirement_id) {
            return Err(ComposeError::DuplicateRequirementSubmission {
                requirement: submission.requirement_id,
            });
        }
        let obligation = by_requirement
            .get(&submission.requirement_id)
            .copied()
            .ok_or(ComposeError::UnknownRequirementSubmission {
                requirement: submission.requirement_id,
            })?;
        ledger.record(obligation, submission.disposition)?;
    }

    Ok(ledger.finalize()?.qualify()?)
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ComposeError {
    #[error(transparent)]
    Witness(#[from] WitnessError),
    #[error("submission references requirement outside the validated contract: {requirement:?}")]
    UnknownRequirementSubmission {
        requirement: ContinuityRequirementId,
    },
    #[error("requirement received more than one submitted disposition: {requirement:?}")]
    DuplicateRequirementSubmission {
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
    use crate::witness::{EvidenceClass, VerificationPolicyEntryV1};

    fn contract() -> ValidatedContinuityContractV1 {
        let observation = ObservationEnvelopeV1::new(
            "machine-1",
            "workflow.dependency",
            "fixture",
            "1",
            1_700_000_000_000,
            ObservationCoverage::Complete,
            EvidenceBasis::Declared,
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

    #[test]
    fn safe_composition_yields_qualified_witness() {
        let contract = contract();
        let requirement = contract.requirements()[0].id();
        let policy = VerificationPolicyV1::new(
            "fixture-verifier",
            vec![VerificationPolicyEntryV1::new(
                requirement,
                EvidenceClass::Simulated,
            )],
        )
        .unwrap();
        let qualified = compose_qualified_witness(
            &contract,
            TargetRealizationId::from_digest([4; 32]).unwrap(),
            &policy,
            vec![ContinuityEvidenceSubmissionV1::new(
                requirement,
                ObligationDispositionV1::Satisfied {
                    evidence_digest: [5; 32],
                    evidence_class: EvidenceClass::HardwareVerified,
                },
            )],
        )
        .unwrap();
        assert_eq!(qualified.contract_id(), contract.id());
    }

    #[test]
    fn missing_submission_remains_incomplete() {
        let contract = contract();
        let requirement = contract.requirements()[0].id();
        let policy = VerificationPolicyV1::new(
            "fixture-verifier",
            vec![VerificationPolicyEntryV1::new(
                requirement,
                EvidenceClass::Simulated,
            )],
        )
        .unwrap();
        assert!(matches!(
            compose_qualified_witness(
                &contract,
                TargetRealizationId::from_digest([6; 32]).unwrap(),
                &policy,
                vec![],
            ),
            Err(ComposeError::Witness(WitnessError::IncompleteWitness { .. }))
        ));
    }

    #[test]
    fn duplicate_requirement_submission_fails_before_ledger_finalization() {
        let contract = contract();
        let requirement = contract.requirements()[0].id();
        let policy = VerificationPolicyV1::new(
            "fixture-verifier",
            vec![VerificationPolicyEntryV1::new(
                requirement,
                EvidenceClass::Observed,
            )],
        )
        .unwrap();
        let disposition = ObligationDispositionV1::Satisfied {
            evidence_digest: [7; 32],
            evidence_class: EvidenceClass::Observed,
        };
        let result = compose_qualified_witness(
            &contract,
            TargetRealizationId::from_digest([8; 32]).unwrap(),
            &policy,
            vec![
                ContinuityEvidenceSubmissionV1::new(requirement, disposition.clone()),
                ContinuityEvidenceSubmissionV1::new(requirement, disposition),
            ],
        );
        assert!(matches!(
            result,
            Err(ComposeError::DuplicateRequirementSubmission { .. })
        ));
    }
}
