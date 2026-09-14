// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime activation authority for clock-governed fabrication upgrades.
//!
//! Upgrade authorization is intentionally not executable authority. This crate mints a separate
//! opaque permit only after trusted time has definitely reached the handoff activation instant,
//! the finalization window is definitely still open, and every policy migration required by the
//! handoff has become the exact one-step current hardened policy lineage under the same fresh
//! operational-clock basis.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_policy_exact_evidence::ExactEvidenceBoundPolicyHeadV1;
use symthaea_fabrication_policy_head_observation::QuorumObservedPolicyHeadV1;
use symthaea_fabrication_policy_lineage::ClockGovernedPolicyLineageV1;
use symthaea_fabrication_policy_temporal_validity::ClockGovernedPolicyTemporalValidityPermitV1;
use symthaea_fabrication_upgrade_authority::{
    ClockGovernedUpgradeHandoffIdV1, ClockGovernedUpgradeHandoffV1,
    UpgradePolicyRequirementV1,
};
use symthaea_fabrication_witness_authority::RegistryBoundPolicyHeadV1;
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const CLOCK_GOVERNED_UPGRADE_ACTIVATION_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-upgrade-activation.v1";
pub const MAX_UPGRADE_ACTIVATION_CLOCK_HOPS: usize = 4096;

const POLICY_ACTIVATION_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-policy-activation-set.v1\0";
const UPGRADE_ACTIVATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-upgrade-activation.v1\0";

/// Borrowed post-migration authority bundle for one policy domain. It must describe the exact
/// successor lineage required by the handoff and be freshly re-observed under the execution clock.
pub struct UpgradeActivatedPolicyAuthorityInputV1<'a> {
    pub lineage: &'a ClockGovernedPolicyLineageV1,
    pub temporal: &'a ClockGovernedPolicyTemporalValidityPermitV1,
    pub observed: &'a QuorumObservedPolicyHeadV1,
    pub registry_bound: &'a RegistryBoundPolicyHeadV1,
    pub exact_evidence: &'a ExactEvidenceBoundPolicyHeadV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedUpgradeActivationPermitIdV1(Sha256Digest);

impl ClockGovernedUpgradeActivationPermitIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque permission to begin one exact authorized upgrade handoff now.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedUpgradeActivationPermitV1 {
    id: ClockGovernedUpgradeActivationPermitIdV1,
    handoff_id: ClockGovernedUpgradeHandoffIdV1,
    handoff_plan_digest: Sha256Digest,
    policy_authority_set_digest: Sha256Digest,
    policy_activation_set_digest: Sha256Digest,
    authorization_operational_basis_id: OperationalClockBasisIdV1,
    current_operational_basis_id: OperationalClockBasisIdV1,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    clock_bridge_hops: usize,
    activated_policy_count: usize,
    activates_at_unix_ms: u64,
    finalization_deadline_unix_ms: u64,
}

impl ClockGovernedUpgradeActivationPermitV1 {
    pub fn id(&self) -> ClockGovernedUpgradeActivationPermitIdV1 {
        self.id
    }
    pub fn handoff_id(&self) -> ClockGovernedUpgradeHandoffIdV1 {
        self.handoff_id
    }
    pub fn handoff_plan_digest(&self) -> Sha256Digest {
        self.handoff_plan_digest
    }
    pub fn policy_authority_set_digest(&self) -> Sha256Digest {
        self.policy_authority_set_digest
    }
    pub fn policy_activation_set_digest(&self) -> Sha256Digest {
        self.policy_activation_set_digest
    }
    pub fn authorization_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.authorization_operational_basis_id
    }
    pub fn current_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.current_operational_basis_id
    }
    pub fn current_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.current_clock_envelope_id
    }
    pub fn clock_bridge_hops(&self) -> usize {
        self.clock_bridge_hops
    }
    pub fn activated_policy_count(&self) -> usize {
        self.activated_policy_count
    }
    pub fn activates_at_unix_ms(&self) -> u64 {
        self.activates_at_unix_ms
    }
    pub fn finalization_deadline_unix_ms(&self) -> u64 {
        self.finalization_deadline_unix_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernedUpgradeActivationError {
    AuthorizationBasisMismatch,
    AuthorizationEnvelopeMismatch,
    TooManyClockBridgeHops { actual: usize, maximum: usize },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    Clock(ClockGovernanceTimeError),
    ActivationNotDefinitelyReached {
        activates_at_unix_ms: u64,
        current_lower_unix_ms: u64,
    },
    FinalizationMayBeClosed {
        finalization_deadline_unix_ms: u64,
        current_upper_unix_ms: u64,
    },
    PolicyAuthorityCountMismatch { required: usize, supplied: usize },
    DuplicatePolicyDomain(String),
    MissingPolicyDomain(String),
    UnexpectedPolicyDomain(String),
    PolicyLineageNotExactSuccessor(String),
    PolicyMigrationNotActivated(String),
    PolicySuccessorMismatch(String),
    PolicyTemporalMismatch(String),
    PolicyObservedHeadMismatch(String),
    PolicyRegistryHeadMismatch(String),
    PolicyExactEvidenceMismatch(String),
    PolicyNotFreshOnExecutionClock(String),
    SequenceOverflow(String),
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct ActivatedPolicyCommitment {
    policy_domain: String,
    required_migration_authority_id: String,
    predecessor_lineage_id: String,
    predecessor_lineage_sequence: u64,
    current_lineage_id: String,
    current_lineage_sequence: u64,
    current_policy_binding_digest: String,
    migration_activation_permit_id: String,
    temporal_validity_permit_id: String,
    observed_head_id: String,
    registry_bound_head_id: String,
    exact_evidence_head_id: String,
}

/// Mint runtime permission for an already-authorized upgrade handoff.
///
/// `clock_bridge` contains any operational-clock bases strictly between `authorization_basis` and
/// `current_basis`. There is no caller-selected scalar current time.
pub fn derive_clock_governed_upgrade_activation_permit_v1(
    handoff: &ClockGovernedUpgradeHandoffV1,
    authorization_basis: &OperationalClockBasisV1,
    clock_bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
    activated_policies: &[UpgradeActivatedPolicyAuthorityInputV1<'_>],
) -> Result<ClockGovernedUpgradeActivationPermitV1, Vec<ClockGovernedUpgradeActivationError>> {
    let mut violations = Vec::new();

    if authorization_basis.id() != handoff.authorization_operational_basis_id() {
        violations.push(ClockGovernedUpgradeActivationError::AuthorizationBasisMismatch);
    }
    let authorization_clock = match derive_clock_governance_evaluation_envelope_v1(authorization_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(ClockGovernedUpgradeActivationError::Clock(error));
            return Err(violations);
        }
    };
    if authorization_clock.id() != handoff.authorization_clock_envelope_id() {
        violations.push(ClockGovernedUpgradeActivationError::AuthorizationEnvelopeMismatch);
    }
    if clock_bridge.len() > MAX_UPGRADE_ACTIVATION_CLOCK_HOPS {
        violations.push(ClockGovernedUpgradeActivationError::TooManyClockBridgeHops {
            actual: clock_bridge.len(),
            maximum: MAX_UPGRADE_ACTIVATION_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(
        handoff.authorization_operational_basis_id(),
        clock_bridge,
        current_basis,
    ) {
        violations.push(error);
    }

    let current_clock = match derive_clock_governance_evaluation_envelope_v1(current_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(ClockGovernedUpgradeActivationError::Clock(error));
            return Err(violations);
        }
    };

    let plan = handoff.plan();
    if current_clock.lower_unix_ms() < plan.activates_at_unix_ms {
        violations.push(
            ClockGovernedUpgradeActivationError::ActivationNotDefinitelyReached {
                activates_at_unix_ms: plan.activates_at_unix_ms,
                current_lower_unix_ms: current_clock.lower_unix_ms(),
            },
        );
    }
    if current_clock.upper_unix_ms() >= plan.finalization_deadline_unix_ms {
        violations.push(ClockGovernedUpgradeActivationError::FinalizationMayBeClosed {
            finalization_deadline_unix_ms: plan.finalization_deadline_unix_ms,
            current_upper_unix_ms: current_clock.upper_unix_ms(),
        });
    }

    let activated_policy_commitments = match qualify_activated_policies(
        &plan.policy_requirements,
        activated_policies,
        current_basis,
        current_clock.id(),
    ) {
        Ok(value) => value,
        Err(mut errors) => {
            violations.append(&mut errors);
            Vec::new()
        }
    };

    if !violations.is_empty() {
        return Err(violations);
    }

    let policy_activation_set_digest = hash_serializable(
        POLICY_ACTIVATION_SET_DOMAIN,
        &activated_policy_commitments,
    )
    .map_err(|error| vec![error])?;
    let id = ClockGovernedUpgradeActivationPermitIdV1(
        digest_upgrade_activation(
            handoff.id(),
            handoff.plan_digest(),
            handoff.policy_authority_set_digest(),
            policy_activation_set_digest,
            handoff.authorization_operational_basis_id(),
            current_basis.id(),
            current_clock.id(),
            clock_bridge.len(),
            activated_policy_commitments.len(),
            plan.activates_at_unix_ms,
            plan.finalization_deadline_unix_ms,
        )
        .map_err(|error| vec![error])?,
    );

    Ok(ClockGovernedUpgradeActivationPermitV1 {
        id,
        handoff_id: handoff.id(),
        handoff_plan_digest: handoff.plan_digest(),
        policy_authority_set_digest: handoff.policy_authority_set_digest(),
        policy_activation_set_digest,
        authorization_operational_basis_id: handoff.authorization_operational_basis_id(),
        current_operational_basis_id: current_basis.id(),
        current_clock_envelope_id: current_clock.id(),
        clock_bridge_hops: clock_bridge.len(),
        activated_policy_count: activated_policy_commitments.len(),
        activates_at_unix_ms: plan.activates_at_unix_ms,
        finalization_deadline_unix_ms: plan.finalization_deadline_unix_ms,
    })
}

fn verify_clock_lineage(
    authorization_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), ClockGovernedUpgradeActivationError> {
    if current_basis.id() == authorization_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(ClockGovernedUpgradeActivationError::BrokenClockLineage {
            hop: 1,
            expected_predecessor: authorization_basis_id.to_hex(),
            actual_predecessor: bridge[0]
                .predecessor_operational_basis_id()
                .map(|value| value.to_hex()),
        });
    }

    let mut expected = authorization_basis_id;
    for (index, basis) in bridge.iter().enumerate() {
        let actual = basis.predecessor_operational_basis_id();
        if actual != Some(expected) {
            return Err(ClockGovernedUpgradeActivationError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(ClockGovernedUpgradeActivationError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn qualify_activated_policies(
    requirements: &[UpgradePolicyRequirementV1],
    activated_policies: &[UpgradeActivatedPolicyAuthorityInputV1<'_>],
    current_basis: &OperationalClockBasisV1,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
) -> Result<Vec<ActivatedPolicyCommitment>, Vec<ClockGovernedUpgradeActivationError>> {
    let mut violations = Vec::new();
    if requirements.len() != activated_policies.len() {
        violations.push(ClockGovernedUpgradeActivationError::PolicyAuthorityCountMismatch {
            required: requirements.len(),
            supplied: activated_policies.len(),
        });
    }

    let mut inputs = BTreeMap::new();
    for input in activated_policies {
        let domain = input.lineage.domain().to_string();
        if inputs.insert(domain.clone(), input).is_some() {
            violations.push(ClockGovernedUpgradeActivationError::DuplicatePolicyDomain(domain));
        }
    }

    let required_domains = requirements
        .iter()
        .map(|requirement| requirement.policy_domain.clone())
        .collect::<BTreeSet<_>>();
    for domain in inputs.keys() {
        if !required_domains.contains(domain) {
            violations.push(ClockGovernedUpgradeActivationError::UnexpectedPolicyDomain(
                domain.clone(),
            ));
        }
    }

    let mut commitments = Vec::with_capacity(requirements.len());
    for requirement in requirements {
        let Some(input) = inputs.get(&requirement.policy_domain).copied() else {
            violations.push(ClockGovernedUpgradeActivationError::MissingPolicyDomain(
                requirement.policy_domain.clone(),
            ));
            continue;
        };
        let domain = requirement.policy_domain.clone();
        let expected_sequence = match requirement.current_lineage_sequence.checked_add(1) {
            Some(value) => value,
            None => {
                violations.push(ClockGovernedUpgradeActivationError::SequenceOverflow(
                    domain,
                ));
                continue;
            }
        };
        let previous_matches = input
            .lineage
            .previous_lineage_id()
            .is_some_and(|id| id.to_hex() == requirement.current_lineage_id);
        if input.lineage.sequence() != expected_sequence || !previous_matches {
            violations.push(
                ClockGovernedUpgradeActivationError::PolicyLineageNotExactSuccessor(
                    requirement.policy_domain.clone(),
                ),
            );
        }
        let migration_matches = input
            .lineage
            .last_migration_id()
            .is_some_and(|id| id.to_hex() == requirement.migration_authority_id);
        let Some(last_activation_permit_id) = input.lineage.last_activation_permit_id() else {
            violations.push(ClockGovernedUpgradeActivationError::PolicyMigrationNotActivated(
                requirement.policy_domain.clone(),
            ));
            continue;
        };
        if !migration_matches {
            violations.push(ClockGovernedUpgradeActivationError::PolicyMigrationNotActivated(
                requirement.policy_domain.clone(),
            ));
        }
        if input.lineage.current_policy_binding_digest()
            != requirement.successor_policy_binding_digest
        {
            violations.push(ClockGovernedUpgradeActivationError::PolicySuccessorMismatch(
                requirement.policy_domain.clone(),
            ));
        }

        if input.temporal.lineage_id() != input.lineage.id()
            || input.temporal.lineage_sequence() != input.lineage.sequence()
            || input.temporal.current_policy_binding_digest()
                != input.lineage.current_policy_binding_digest()
        {
            violations.push(ClockGovernedUpgradeActivationError::PolicyTemporalMismatch(
                requirement.policy_domain.clone(),
            ));
        }
        if input.temporal.current_operational_basis_id() != current_basis.id()
            || input.temporal.current_clock_envelope_id() != current_clock_envelope_id
        {
            violations.push(
                ClockGovernedUpgradeActivationError::PolicyNotFreshOnExecutionClock(
                    requirement.policy_domain.clone(),
                ),
            );
        }
        if input.observed.lineage_id() != input.lineage.id()
            || input.observed.lineage_sequence() != input.lineage.sequence()
            || input.observed.temporal_validity_permit_id() != input.temporal.id()
            || input.observed.clock_envelope_id() != current_clock_envelope_id
        {
            violations.push(ClockGovernedUpgradeActivationError::PolicyObservedHeadMismatch(
                requirement.policy_domain.clone(),
            ));
        }
        if input.registry_bound.observed_head_id() != input.observed.id() {
            violations.push(ClockGovernedUpgradeActivationError::PolicyRegistryHeadMismatch(
                requirement.policy_domain.clone(),
            ));
        }
        if input.exact_evidence.observed_head_id() != input.observed.id()
            || input.exact_evidence.registry_bound_head_id() != input.registry_bound.id()
        {
            violations.push(ClockGovernedUpgradeActivationError::PolicyExactEvidenceMismatch(
                requirement.policy_domain.clone(),
            ));
        }

        commitments.push(ActivatedPolicyCommitment {
            policy_domain: requirement.policy_domain.clone(),
            required_migration_authority_id: requirement.migration_authority_id.clone(),
            predecessor_lineage_id: requirement.current_lineage_id.clone(),
            predecessor_lineage_sequence: requirement.current_lineage_sequence,
            current_lineage_id: input.lineage.id().to_hex(),
            current_lineage_sequence: input.lineage.sequence(),
            current_policy_binding_digest: input.lineage.current_policy_binding_digest().to_hex(),
            migration_activation_permit_id: last_activation_permit_id.to_hex(),
            temporal_validity_permit_id: input.temporal.id().to_hex(),
            observed_head_id: input.observed.id().to_hex(),
            registry_bound_head_id: input.registry_bound.id().to_hex(),
            exact_evidence_head_id: input.exact_evidence.id().to_hex(),
        });
    }

    if !violations.is_empty() {
        return Err(violations);
    }
    commitments.sort_by(|left, right| left.policy_domain.cmp(&right.policy_domain));
    Ok(commitments)
}

#[derive(Serialize)]
struct UpgradeActivationCommitment {
    schema: &'static str,
    handoff_id: String,
    handoff_plan_digest: String,
    policy_authority_set_digest: String,
    policy_activation_set_digest: String,
    authorization_operational_basis_id: String,
    current_operational_basis_id: String,
    current_clock_envelope_id: String,
    clock_bridge_hops: usize,
    activated_policy_count: usize,
    activates_at_unix_ms: u64,
    finalization_deadline_unix_ms: u64,
}

#[allow(clippy::too_many_arguments)]
fn digest_upgrade_activation(
    handoff_id: ClockGovernedUpgradeHandoffIdV1,
    handoff_plan_digest: Sha256Digest,
    policy_authority_set_digest: Sha256Digest,
    policy_activation_set_digest: Sha256Digest,
    authorization_operational_basis_id: OperationalClockBasisIdV1,
    current_operational_basis_id: OperationalClockBasisIdV1,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    clock_bridge_hops: usize,
    activated_policy_count: usize,
    activates_at_unix_ms: u64,
    finalization_deadline_unix_ms: u64,
) -> Result<Sha256Digest, ClockGovernedUpgradeActivationError> {
    hash_serializable(
        UPGRADE_ACTIVATION_DOMAIN,
        &UpgradeActivationCommitment {
            schema: CLOCK_GOVERNED_UPGRADE_ACTIVATION_SCHEMA,
            handoff_id: handoff_id.to_hex(),
            handoff_plan_digest: handoff_plan_digest.to_hex(),
            policy_authority_set_digest: policy_authority_set_digest.to_hex(),
            policy_activation_set_digest: policy_activation_set_digest.to_hex(),
            authorization_operational_basis_id: authorization_operational_basis_id.to_hex(),
            current_operational_basis_id: current_operational_basis_id.to_hex(),
            current_clock_envelope_id: current_clock_envelope_id.to_hex(),
            clock_bridge_hops,
            activated_policy_count,
            activates_at_unix_ms,
            finalization_deadline_unix_ms,
        },
    )
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, ClockGovernedUpgradeActivationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ClockGovernedUpgradeActivationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
