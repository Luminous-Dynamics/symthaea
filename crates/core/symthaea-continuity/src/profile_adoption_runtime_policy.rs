// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public guard for attenuation-aware verifier runtime policy.
//!
//! The underlying runtime evaluator supports newer grant epochs so policy can
//! narrow without re-adoption. This guard adds the anti-equivocation rule:
//!
//! `same grant epoch => same grant semantics`.
//!
//! It also grounds the historical persisted grant scope in the exact contract
//! before returning any runtime policy envelope.

use thiserror::Error;

use crate::contract::{ContinuityRequirementId, ValidatedContinuityContractV1};
use crate::profile_adoption::VerifierAdoptionScopeV1;
use crate::profile_adoption_admission::VerifierProfileAdoptionHeadV1;
use crate::profile_adoption_authority::VerifierAdoptionAuthorityGrantV1;
use crate::profile_adoption_commit::VerifierAdoptionAuthorityRootSnapshotV1;
use crate::profile_adoption_currentness::VerifierProfileAdoptionRegistryRecordV1;
use crate::profile_adoption_runtime::{
    PolicyCurrentVerifierRuntimeEnvelopeV1, VerifierProfileRuntimePolicyError,
    check_verifier_runtime_policy as check_runtime_inner,
};
use crate::verifier::VerifierProfileV1;

#[allow(clippy::too_many_arguments)]
pub fn check_verifier_runtime_policy(
    record: &VerifierProfileAdoptionRegistryRecordV1,
    now_unix_ms: u64,
    current_root: &VerifierAdoptionAuthorityRootSnapshotV1,
    current_grant: Option<&VerifierAdoptionAuthorityGrantV1>,
    current_head: &VerifierProfileAdoptionHeadV1,
    current_profile: &VerifierProfileV1,
    scope_contract: Option<&ValidatedContinuityContractV1>,
) -> Result<PolicyCurrentVerifierRuntimeEnvelopeV1, VerifierRuntimePolicyGuardError> {
    // Evaluate all ordinary currentness first. The inner value is not exposed unless
    // the stricter persisted-policy/epoch checks below also succeed.
    let envelope = check_runtime_inner(
        record,
        now_unix_ms,
        current_root,
        current_grant,
        current_head,
        current_profile,
        scope_contract,
    )?;

    ground_historical_scope(record.authority_grant().allowed_scope(), scope_contract)?;

    let current_grant = current_grant
        .ok_or(VerifierRuntimePolicyGuardError::MissingCurrentGrantAfterInnerSuccess)?;
    let historical = record.authority_grant();
    if current_grant.grant_epoch() == historical.grant_epoch()
        && (current_grant.maximum_evidence_class() != historical.maximum_evidence_class()
            || current_grant.allowed_scope() != historical.allowed_scope())
    {
        return Err(VerifierRuntimePolicyGuardError::SameEpochGrantEquivocation {
            epoch: current_grant.grant_epoch(),
        });
    }

    Ok(envelope)
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierRuntimePolicyGuardError {
    #[error(transparent)]
    Runtime(#[from] VerifierProfileRuntimePolicyError),
    #[error("runtime evaluator succeeded without a current grant")]
    MissingCurrentGrantAfterInnerSuccess,
    #[error("historical authority grant scope requires an exact continuity contract")]
    MissingHistoricalScopeContract,
    #[error("historical authority grant scope references a different continuity contract")]
    HistoricalScopeContractMismatch,
    #[error("historical authority grant references requirement outside the exact contract: {requirement:?}")]
    UnknownHistoricalGrantRequirement { requirement: ContinuityRequirementId },
    #[error("authority grant semantics changed without advancing grant epoch {epoch}")]
    SameEpochGrantEquivocation { epoch: u64 },
}

fn ground_historical_scope(
    scope: &VerifierAdoptionScopeV1,
    contract: Option<&ValidatedContinuityContractV1>,
) -> Result<(), VerifierRuntimePolicyGuardError> {
    match scope {
        VerifierAdoptionScopeV1::AllContinuityVerification => Ok(()),
        VerifierAdoptionScopeV1::Contract { contract_id } => {
            let contract = contract
                .ok_or(VerifierRuntimePolicyGuardError::MissingHistoricalScopeContract)?;
            if contract.id() != *contract_id {
                return Err(VerifierRuntimePolicyGuardError::HistoricalScopeContractMismatch);
            }
            Ok(())
        }
        VerifierAdoptionScopeV1::Requirements { contract_id, requirement_ids } => {
            let contract = contract
                .ok_or(VerifierRuntimePolicyGuardError::MissingHistoricalScopeContract)?;
            if contract.id() != *contract_id {
                return Err(VerifierRuntimePolicyGuardError::HistoricalScopeContractMismatch);
            }
            for requirement in requirement_ids {
                if !contract.requirements().iter().any(|item| item.id() == *requirement) {
                    return Err(VerifierRuntimePolicyGuardError::UnknownHistoricalGrantRequirement {
                        requirement: *requirement,
                    });
                }
            }
            Ok(())
        }
    }
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
        VerifierProfileAdoptionSubjectV1, VerifierProfileAdoptionTransitionV1,
    };
    use crate::profile_adoption_admission::VerifierProfileAdoptionAdmissionPolicyV1;
    use crate::profile_adoption_authority::bind_policy_checked_adoption_to_authority_grant;
    use crate::profile_adoption_currentness::VerifierProfileAdoptionRegistryRecordV1;
    use crate::witness::EvidenceClass;

    fn contract() -> ValidatedContinuityContractV1 {
        let observation = ObservationEnvelopeV1::new(
            "machine-1", "workflow.dependency", "fixture", "1", 1_700_000_000_000,
            ObservationCoverage::Complete, EvidenceBasis::Tested, [1; 32], vec![],
        ).unwrap();
        let dependency = DependencyClaimV1::new(
            "role:research", "requires", "capability:cuda", DependencyBasis::Observed,
            vec![observation.id()], vec![],
        ).unwrap();
        let requirement = ContinuityRequirementV1::new(
            dependency.id(), "cuda-workflow", RequirementCriticality::Must,
            EquivalencePredicate::BehavioralScenario { scenario_id: "cuda-fixture-v1".into() },
            ApprovalBasis::ExplicitPolicy, [2; 32],
        ).unwrap();
        ContinuityContractV1::new("research-fleet", [3; 32], vec![requirement])
            .unwrap().validate().unwrap()
    }

    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "hardware-verifier-v1", [9; 32], 7, EvidenceClass::HardwareVerified,
        ).unwrap()
    }

    fn root() -> VerifierAdoptionAuthorityRootSnapshotV1 {
        VerifierAdoptionAuthorityRootSnapshotV1::new(
            "organization:test", "adoption-root-1", [0x55; 32], 4,
        ).unwrap()
    }

    fn grant(epoch: u64, class: EvidenceClass, scope: VerifierAdoptionScopeV1) -> VerifierAdoptionAuthorityGrantV1 {
        VerifierAdoptionAuthorityGrantV1::new(
            "grant-1", "organization:test", "adoption-root-1", [0x55; 32],
            "hardware-verifier-v1", epoch, class, scope,
        ).unwrap()
    }

    fn fixture() -> (
        ValidatedContinuityContractV1,
        VerifierProfileV1,
        VerifierAdoptionAuthorityGrantV1,
        VerifierAdoptionAuthorityRootSnapshotV1,
        VerifierProfileAdoptionRegistryRecordV1,
        VerifierProfileAdoptionHeadV1,
    ) {
        let contract = contract();
        let profile = profile();
        let grant = grant(
            3,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::Contract { contract_id: contract.id() },
        );
        let subject = VerifierProfileAdoptionSubjectV1::new(
            "adopt-1", "organization:test", "adoption-root-1", [0x55; 32], &profile,
            1, 1_000, 2_000, EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::Contract { contract_id: contract.id() },
        ).unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let admission = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test", "adoption-root-1", [0x55; 32],
            "hardware-verifier-v1", VerifierProfileAdoptionHeadV1::Uninitialized,
        ).unwrap();
        let checked = admission.check(1_500, &transition, &profile, Some(&contract)).unwrap();
        let granted = bind_policy_checked_adoption_to_authority_grant(
            checked, &grant, Some(&contract),
        ).unwrap();
        let root = root();
        let record = VerifierProfileAdoptionRegistryRecordV1::new_uncommitted_projection(
            &granted, &root, None,
        ).unwrap();
        let head = VerifierProfileAdoptionHeadV1::from_transition(&transition).unwrap();
        (contract, profile, grant, root, record, head)
    }

    #[test]
    fn same_epoch_same_semantics_remains_valid() {
        let (contract, profile, grant, root, record, head) = fixture();
        check_verifier_runtime_policy(
            &record, 1_500, &root, Some(&grant), &head, &profile, Some(&contract),
        ).unwrap();
    }

    #[test]
    fn same_epoch_changed_ceiling_is_equivocation() {
        let (contract, profile, _, root, record, head) = fixture();
        let conflicting = grant(
            3,
            EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::Contract { contract_id: contract.id() },
        );
        assert_eq!(
            check_verifier_runtime_policy(
                &record, 1_500, &root, Some(&conflicting), &head, &profile, Some(&contract),
            ).unwrap_err(),
            VerifierRuntimePolicyGuardError::SameEpochGrantEquivocation { epoch: 3 },
        );
    }

    #[test]
    fn newer_epoch_may_narrow_without_equivocation() {
        let (contract, profile, _, root, record, head) = fixture();
        let narrowed = grant(
            4,
            EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::Contract { contract_id: contract.id() },
        );
        let current = check_verifier_runtime_policy(
            &record, 1_500, &root, Some(&narrowed), &head, &profile, Some(&contract),
        ).unwrap();
        assert_eq!(current.effective_evidence_class(), EvidenceClass::DifferentiallyVerified);
    }
}
