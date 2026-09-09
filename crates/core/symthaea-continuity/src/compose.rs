// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Internal composition of fresh verifier-owned evidence into a witness context.

use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::contract::{ContinuityRequirementId, ValidatedContinuityContractV1};
use crate::freshness::{
    FreshAuthenticatedVerificationEvidenceV1, FreshnessError, FreshnessPolicyV1,
    QualifiedContinuityWitnessContextV1,
};
use crate::witness::{TargetRealizationId, VerificationPolicyV1, WitnessError, WitnessLedgerV1, WitnessManifestV1};

pub(crate) fn compose_qualified_witness_context(
    contract: &ValidatedContinuityContractV1,
    target: TargetRealizationId,
    verification_policy: &VerificationPolicyV1,
    freshness_policy: &FreshnessPolicyV1,
    evidence: Vec<FreshAuthenticatedVerificationEvidenceV1>,
) -> Result<QualifiedContinuityWitnessContextV1, ComposeError> {
    if freshness_policy.contract_id() != contract.id() {
        return Err(ComposeError::FreshnessPolicyContractMismatch);
    }
    let manifest = WitnessManifestV1::new(contract, target, verification_policy)?;
    let mut obligations = BTreeMap::new();
    for obligation in manifest.obligations() {
        obligations.insert(obligation.requirement_id(), obligation.id());
    }

    let mut seen = BTreeSet::new();
    let mut ledger = WitnessLedgerV1::new(manifest);
    for item in &evidence {
        if item.contract_id() != contract.id() { return Err(ComposeError::CrossContractEvidence); }
        if item.target_realization_id() != target { return Err(ComposeError::CrossTargetEvidence); }
        if item.freshness_policy_id() != freshness_policy.id() { return Err(ComposeError::CrossFreshnessPolicyEvidence); }
        let requirement = item.requirement_id();
        if !seen.insert(requirement) { return Err(ComposeError::DuplicateRequirementEvidence { requirement }); }
        let obligation = obligations.get(&requirement).copied().ok_or(ComposeError::UnknownRequirementEvidence { requirement })?;
        ledger.record(obligation, item.disposition())?;
    }

    let witness = ledger.finalize()?.qualify()?;
    Ok(QualifiedContinuityWitnessContextV1::new(witness, freshness_policy.id(), &evidence)?)
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum ComposeError {
    #[error(transparent)]
    Witness(#[from] WitnessError),
    #[error(transparent)]
    Freshness(#[from] FreshnessError),
    #[error("freshness policy belongs to a different continuity contract")]
    FreshnessPolicyContractMismatch,
    #[error("fresh evidence belongs to a different continuity contract")]
    CrossContractEvidence,
    #[error("fresh evidence belongs to a different target realization")]
    CrossTargetEvidence,
    #[error("fresh evidence was admitted under a different freshness policy")]
    CrossFreshnessPolicyEvidence,
    #[error("fresh evidence references requirement outside the validated contract: {requirement:?}")]
    UnknownRequirementEvidence { requirement: ContinuityRequirementId },
    #[error("requirement received more than one fresh evidence item: {requirement:?}")]
    DuplicateRequirementEvidence { requirement: ContinuityRequirementId },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, EquivalencePredicate, RequirementCriticality};
    use crate::freshness::{admit_freshness, EvidenceFreshnessRequirementV1, EvidenceStabilityClassV1, FreshnessPolicyEntryV1, TrustedClockSnapshotV1};
    use crate::observation::{DependencyBasis, DependencyClaimV1, EvidenceBasis, ObservationCoverage, ObservationEnvelopeV1};
    use crate::verifier::{policy_check_verification_evidence, AuthenticatedVerificationEvidenceV1, VerificationEvidenceClaimV1, VerificationOutcomeV1, VerifierProfileV1};
    use crate::witness::{EvidenceClass, VerificationPolicyEntryV1};

    fn contract() -> ValidatedContinuityContractV1 {
        let o = ObservationEnvelopeV1::new("m", "dep", "fixture", "1", 1_700_000_000_000, ObservationCoverage::Complete, EvidenceBasis::Tested, [1;32], vec![]).unwrap();
        let d = DependencyClaimV1::new("role", "requires", "cap", DependencyBasis::Observed, vec![o.id()], vec![]).unwrap();
        let r = ContinuityRequirementV1::new(d.id(), "cap", RequirementCriticality::Must, EquivalencePredicate::BehavioralScenario { scenario_id: "s".into() }, ApprovalBasis::ExplicitPolicy, [2;32]).unwrap();
        ContinuityContractV1::new("fleet", [3;32], vec![r]).unwrap().validate().unwrap()
    }
    fn verification_policy(c: &ValidatedContinuityContractV1) -> VerificationPolicyV1 {
        VerificationPolicyV1::new("v", vec![VerificationPolicyEntryV1::new(c.requirements()[0].id(), EvidenceClass::Simulated)]).unwrap()
    }
    fn freshness_policy(c: &ValidatedContinuityContractV1, max_age: u64) -> FreshnessPolicyV1 {
        FreshnessPolicyV1::new(c, vec![FreshnessPolicyEntryV1::new(c.requirements()[0].id(), EvidenceFreshnessRequirementV1::MaxAge { class: EvidenceStabilityClassV1::Volatile, max_age_ms: max_age }).unwrap()]).unwrap()
    }
    fn authenticated(c: &ValidatedContinuityContractV1, target: TargetRealizationId, observed: u64) -> AuthenticatedVerificationEvidenceV1 {
        let profile = VerifierProfileV1::new("v", [5;32], 2, EvidenceClass::HardwareVerified).unwrap();
        let vp = verification_policy(c);
        let claim = VerificationEvidenceClaimV1::new(c.id(), target, c.requirements()[0].id(), profile.id(), [6;32], observed, VerificationOutcomeV1::Satisfied, [7;32]).unwrap();
        let checked = policy_check_verification_evidence(c, target, &vp, &profile, [6;32], claim).unwrap();
        AuthenticatedVerificationEvidenceV1::authenticate_for_test(checked, [8;32]).unwrap()
    }

    #[test]
    fn fresh_exact_context_yields_witness_context() {
        let c = contract();
        let target = TargetRealizationId::from_digest([4;32]).unwrap();
        let fp = freshness_policy(&c, 1000);
        let clock = TrustedClockSnapshotV1::new("clock", 1, 1500, 1550).unwrap();
        let fresh = admit_freshness(&fp, Some(&clock), authenticated(&c, target, 1000)).unwrap();
        let context = compose_qualified_witness_context(&c, target, &verification_policy(&c), &fp, vec![fresh]).unwrap();
        assert_eq!(context.contract_id(), c.id());
        assert_eq!(context.target_realization_id(), target);
        assert_eq!(context.valid_until_unix_ms(), Some(2000));
        assert_eq!(context.clock_lineage().unwrap().epoch(), 1);
    }

    #[test]
    fn evidence_admitted_under_other_freshness_policy_is_rejected() {
        let c = contract();
        let target = TargetRealizationId::from_digest([4;32]).unwrap();
        let lenient = freshness_policy(&c, 2000);
        let strict = freshness_policy(&c, 1000);
        let clock = TrustedClockSnapshotV1::new("clock", 1, 1500, 1550).unwrap();
        let fresh = admit_freshness(&lenient, Some(&clock), authenticated(&c, target, 1000)).unwrap();
        assert_eq!(
            compose_qualified_witness_context(&c, target, &verification_policy(&c), &strict, vec![fresh]).unwrap_err(),
            ComposeError::CrossFreshnessPolicyEvidence,
        );
    }
}
