// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime activation authority that preserves the globally-current finalized predecessor lineage.
//!
//! The ordinary runtime theorem is sound for an ordinary clock-governed handoff, but its permit only
//! carries the ordinary handoff ID. This bridge independently re-proves activation from the public
//! requirements of `LineageBoundClockGovernedUpgradeHandoffV1` and preserves the predecessor-root
//! provenance in a distinct opaque activation capability.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_lineage_bound_upgrade_handoff::{
    LineageBoundClockGovernedUpgradeHandoffIdV1, LineageBoundClockGovernedUpgradeHandoffV1,
};
use symthaea_fabrication_upgrade_authority::{
    ClockGovernedUpgradeHandoffIdV1, UpgradePolicyRequirementV1,
};
use symthaea_fabrication_upgrade_runtime::UpgradeActivatedPolicyAuthorityInputV1;
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const LINEAGE_BOUND_UPGRADE_ACTIVATION_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-upgrade-activation.v1";
pub const MAX_LINEAGE_BOUND_ACTIVATION_CLOCK_HOPS: usize = 4096;

const POLICY_ACTIVATION_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-upgrade-policy-activation-set.v1\0";
const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-upgrade-activation-clock-lineage.v1\0";
const ACTIVATION_DOMAIN: &[u8] = b"symthaea.fabrication.lineage-bound-upgrade-activation.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundUpgradeActivationPermitIdV1(Sha256Digest);

impl LineageBoundUpgradeActivationPermitIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque runtime permission for one exact lineage-bound handoff.
#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundUpgradeActivationPermitV1 {
    id: LineageBoundUpgradeActivationPermitIdV1,
    lineage_handoff_id: LineageBoundClockGovernedUpgradeHandoffIdV1,
    inner_handoff_id: ClockGovernedUpgradeHandoffIdV1,
    predecessor_root_digest: Sha256Digest,
    current_head_digest: Sha256Digest,
    governance_view_digest: Sha256Digest,
    registry_head_digest: Sha256Digest,
    predecessor_endpoint_digest: Sha256Digest,
    predecessor_finalization_sequence: u64,
    predecessor_checkpoint_digest: Sha256Digest,
    predecessor_transparency_log_digest: Sha256Digest,
    handoff_plan_digest: Sha256Digest,
    authorization_operational_basis_id: OperationalClockBasisIdV1,
    current_operational_basis_id: OperationalClockBasisIdV1,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_bridge_hops: usize,
    policy_activation_set_digest: Sha256Digest,
    activated_policy_count: usize,
    activates_at_unix_ms: u64,
    finalization_deadline_unix_ms: u64,
}

impl LineageBoundUpgradeActivationPermitV1 {
    pub fn id(&self) -> LineageBoundUpgradeActivationPermitIdV1 {
        self.id
    }
    pub fn lineage_handoff_id(&self) -> LineageBoundClockGovernedUpgradeHandoffIdV1 {
        self.lineage_handoff_id
    }
    pub fn inner_handoff_id(&self) -> ClockGovernedUpgradeHandoffIdV1 {
        self.inner_handoff_id
    }
    pub fn predecessor_root_digest(&self) -> Sha256Digest {
        self.predecessor_root_digest
    }
    pub fn current_head_digest(&self) -> Sha256Digest {
        self.current_head_digest
    }
    pub fn governance_view_digest(&self) -> Sha256Digest {
        self.governance_view_digest
    }
    pub fn registry_head_digest(&self) -> Sha256Digest {
        self.registry_head_digest
    }
    pub fn predecessor_endpoint_digest(&self) -> Sha256Digest {
        self.predecessor_endpoint_digest
    }
    pub fn predecessor_finalization_sequence(&self) -> u64 {
        self.predecessor_finalization_sequence
    }
    pub fn predecessor_checkpoint_digest(&self) -> Sha256Digest {
        self.predecessor_checkpoint_digest
    }
    pub fn predecessor_transparency_log_digest(&self) -> Sha256Digest {
        self.predecessor_transparency_log_digest
    }
    pub fn handoff_plan_digest(&self) -> Sha256Digest {
        self.handoff_plan_digest
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
    pub fn clock_lineage_digest(&self) -> Sha256Digest {
        self.clock_lineage_digest
    }
    pub fn clock_bridge_hops(&self) -> usize {
        self.clock_bridge_hops
    }
    pub fn policy_activation_set_digest(&self) -> Sha256Digest {
        self.policy_activation_set_digest
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
pub enum LineageBoundUpgradeActivationError {
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

#[derive(Debug, Clone, Serialize)]
struct ClockLineageCommitment {
    authorization_basis_id: String,
    bridge_basis_ids: Vec<String>,
    current_basis_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct ActivationCommitment {
    schema: &'static str,
    lineage_handoff_id: String,
    inner_handoff_id: String,
    predecessor_root_digest: String,
    current_head_digest: String,
    governance_view_digest: String,
    registry_head_digest: String,
    predecessor_endpoint_digest: String,
    predecessor_finalization_sequence: u64,
    predecessor_checkpoint_digest: String,
    predecessor_transparency_log_digest: String,
    handoff_plan_digest: String,
    authorization_operational_basis_id: String,
    current_operational_basis_id: String,
    current_clock_envelope_id: String,
    clock_lineage_digest: String,
    clock_bridge_hops: usize,
    policy_activation_set_digest: String,
    activated_policy_count: usize,
    activates_at_unix_ms: u64,
    finalization_deadline_unix_ms: u64,
}

pub fn derive_lineage_bound_upgrade_activation_permit_v1(
    handoff: &LineageBoundClockGovernedUpgradeHandoffV1,
    authorization_basis: &OperationalClockBasisV1,
    clock_bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
    activated_policies: &[UpgradeActivatedPolicyAuthorityInputV1<'_>],
) -> Result<LineageBoundUpgradeActivationPermitV1, Vec<LineageBoundUpgradeActivationError>> {
    let mut violations = Vec::new();

    if authorization_basis.id() != handoff.operational_basis_id() {
        violations.push(LineageBoundUpgradeActivationError::AuthorizationBasisMismatch);
    }
    let authorization_clock = match derive_clock_governance_evaluation_envelope_v1(authorization_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundUpgradeActivationError::Clock(error));
            return Err(violations);
        }
    };
    if authorization_clock.id() != handoff.clock_envelope_id() {
        violations.push(LineageBoundUpgradeActivationError::AuthorizationEnvelopeMismatch);
    }

    if clock_bridge.len() > MAX_LINEAGE_BOUND_ACTIVATION_CLOCK_HOPS {
        violations.push(LineageBoundUpgradeActivationError::TooManyClockBridgeHops {
            actual: clock_bridge.len(),
            maximum: MAX_LINEAGE_BOUND_ACTIVATION_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(
        handoff.operational_basis_id(),
        clock_bridge,
        current_basis,
    ) {
        violations.push(error);
    }

    let current_clock = match derive_clock_governance_evaluation_envelope_v1(current_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundUpgradeActivationError::Clock(error));
            return Err(violations);
        }
    };

    let plan = handoff.plan();
    if current_clock.lower_unix_ms() < plan.activates_at_unix_ms {
        violations.push(LineageBoundUpgradeActivationError::ActivationNotDefinitelyReached {
            activates_at_unix_ms: plan.activates_at_unix_ms,
            current_lower_unix_ms: current_clock.lower_unix_ms(),
        });
    }
    if current_clock.upper_unix_ms() >= plan.finalization_deadline_unix_ms {
        violations.push(LineageBoundUpgradeActivationError::FinalizationMayBeClosed {
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
    let clock_lineage_commitment = ClockLineageCommitment {
        authorization_basis_id: authorization_basis.id().to_hex(),
        bridge_basis_ids: clock_bridge.iter().map(|basis| basis.id().to_hex()).collect(),
        current_basis_id: current_basis.id().to_hex(),
    };
    let clock_lineage_digest = hash_serializable(CLOCK_LINEAGE_DOMAIN, &clock_lineage_commitment)
        .map_err(|error| vec![error])?;

    let commitment = ActivationCommitment {
        schema: LINEAGE_BOUND_UPGRADE_ACTIVATION_SCHEMA,
        lineage_handoff_id: handoff.id().to_hex(),
        inner_handoff_id: handoff.inner_handoff_id().to_hex(),
        predecessor_root_digest: handoff.predecessor_root_id().as_digest().to_hex(),
        current_head_digest: handoff.current_head_id().as_digest().to_hex(),
        governance_view_digest: handoff.governance_view_id().as_digest().to_hex(),
        registry_head_digest: handoff.registry_head_id().as_digest().to_hex(),
        predecessor_endpoint_digest: handoff.predecessor_endpoint_digest().to_hex(),
        predecessor_finalization_sequence: handoff.predecessor_finalization_sequence(),
        predecessor_checkpoint_digest: handoff.predecessor_checkpoint_digest().to_hex(),
        predecessor_transparency_log_digest: handoff.predecessor_transparency_log_digest().to_hex(),
        handoff_plan_digest: handoff.plan_digest().to_hex(),
        authorization_operational_basis_id: handoff.operational_basis_id().to_hex(),
        current_operational_basis_id: current_basis.id().to_hex(),
        current_clock_envelope_id: current_clock.id().to_hex(),
        clock_lineage_digest: clock_lineage_digest.to_hex(),
        clock_bridge_hops: clock_bridge.len(),
        policy_activation_set_digest: policy_activation_set_digest.to_hex(),
        activated_policy_count: activated_policy_commitments.len(),
        activates_at_unix_ms: plan.activates_at_unix_ms,
        finalization_deadline_unix_ms: plan.finalization_deadline_unix_ms,
    };
    let id = LineageBoundUpgradeActivationPermitIdV1(
        hash_serializable(ACTIVATION_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(LineageBoundUpgradeActivationPermitV1 {
        id,
        lineage_handoff_id: handoff.id(),
        inner_handoff_id: handoff.inner_handoff_id(),
        predecessor_root_digest: handoff.predecessor_root_id().as_digest(),
        current_head_digest: handoff.current_head_id().as_digest(),
        governance_view_digest: handoff.governance_view_id().as_digest(),
        registry_head_digest: handoff.registry_head_id().as_digest(),
        predecessor_endpoint_digest: handoff.predecessor_endpoint_digest(),
        predecessor_finalization_sequence: handoff.predecessor_finalization_sequence(),
        predecessor_checkpoint_digest: handoff.predecessor_checkpoint_digest(),
        predecessor_transparency_log_digest: handoff.predecessor_transparency_log_digest(),
        handoff_plan_digest: handoff.plan_digest(),
        authorization_operational_basis_id: handoff.operational_basis_id(),
        current_operational_basis_id: current_basis.id(),
        current_clock_envelope_id: current_clock.id(),
        clock_lineage_digest,
        clock_bridge_hops: clock_bridge.len(),
        policy_activation_set_digest,
        activated_policy_count: activated_policy_commitments.len(),
        activates_at_unix_ms: plan.activates_at_unix_ms,
        finalization_deadline_unix_ms: plan.finalization_deadline_unix_ms,
    })
}

fn verify_clock_lineage(
    authorization_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), LineageBoundUpgradeActivationError> {
    if current_basis.id() == authorization_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(LineageBoundUpgradeActivationError::BrokenClockLineage {
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
            return Err(LineageBoundUpgradeActivationError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(LineageBoundUpgradeActivationError::BrokenClockLineage {
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
) -> Result<Vec<ActivatedPolicyCommitment>, Vec<LineageBoundUpgradeActivationError>> {
    let mut violations = Vec::new();
    if requirements.len() != activated_policies.len() {
        violations.push(LineageBoundUpgradeActivationError::PolicyAuthorityCountMismatch {
            required: requirements.len(),
            supplied: activated_policies.len(),
        });
    }

    let mut inputs = BTreeMap::new();
    for input in activated_policies {
        let domain = input.lineage.domain().to_string();
        if inputs.insert(domain.clone(), input).is_some() {
            violations.push(LineageBoundUpgradeActivationError::DuplicatePolicyDomain(domain));
        }
    }

    let required_domains = requirements
        .iter()
        .map(|requirement| requirement.policy_domain.clone())
        .collect::<BTreeSet<_>>();
    for domain in inputs.keys() {
        if !required_domains.contains(domain) {
            violations.push(LineageBoundUpgradeActivationError::UnexpectedPolicyDomain(
                domain.clone(),
            ));
        }
    }

    let mut commitments = Vec::with_capacity(requirements.len());
    for requirement in requirements {
        let Some(input) = inputs.get(&requirement.policy_domain).copied() else {
            violations.push(LineageBoundUpgradeActivationError::MissingPolicyDomain(
                requirement.policy_domain.clone(),
            ));
            continue;
        };
        let domain = requirement.policy_domain.clone();
        let expected_sequence = match requirement.current_lineage_sequence.checked_add(1) {
            Some(value) => value,
            None => {
                violations.push(LineageBoundUpgradeActivationError::SequenceOverflow(domain));
                continue;
            }
        };
        let previous_matches = input
            .lineage
            .previous_lineage_id()
            .is_some_and(|id| id.to_hex() == requirement.current_lineage_id);
        if input.lineage.sequence() != expected_sequence || !previous_matches {
            violations.push(LineageBoundUpgradeActivationError::PolicyLineageNotExactSuccessor(
                requirement.policy_domain.clone(),
            ));
        }
        let migration_matches = input
            .lineage
            .last_migration_id()
            .is_some_and(|id| id.to_hex() == requirement.migration_authority_id);
        let Some(last_activation_permit_id) = input.lineage.last_activation_permit_id() else {
            violations.push(LineageBoundUpgradeActivationError::PolicyMigrationNotActivated(
                requirement.policy_domain.clone(),
            ));
            continue;
        };
        if !migration_matches {
            violations.push(LineageBoundUpgradeActivationError::PolicyMigrationNotActivated(
                requirement.policy_domain.clone(),
            ));
        }
        if input.lineage.current_policy_binding_digest()
            != requirement.successor_policy_binding_digest
        {
            violations.push(LineageBoundUpgradeActivationError::PolicySuccessorMismatch(
                requirement.policy_domain.clone(),
            ));
        }

        if input.temporal.lineage_id() != input.lineage.id()
            || input.temporal.lineage_sequence() != input.lineage.sequence()
            || input.temporal.current_policy_binding_digest()
                != input.lineage.current_policy_binding_digest()
        {
            violations.push(LineageBoundUpgradeActivationError::PolicyTemporalMismatch(
                requirement.policy_domain.clone(),
            ));
        }
        if input.temporal.current_operational_basis_id() != current_basis.id()
            || input.temporal.current_clock_envelope_id() != current_clock_envelope_id
        {
            violations.push(LineageBoundUpgradeActivationError::PolicyNotFreshOnExecutionClock(
                requirement.policy_domain.clone(),
            ));
        }
        if input.observed.lineage_id() != input.lineage.id()
            || input.observed.lineage_sequence() != input.lineage.sequence()
            || input.observed.temporal_validity_permit_id() != input.temporal.id()
            || input.observed.clock_envelope_id() != current_clock_envelope_id
        {
            violations.push(LineageBoundUpgradeActivationError::PolicyObservedHeadMismatch(
                requirement.policy_domain.clone(),
            ));
        }
        if input.registry_bound.observed_head_id() != input.observed.id() {
            violations.push(LineageBoundUpgradeActivationError::PolicyRegistryHeadMismatch(
                requirement.policy_domain.clone(),
            ));
        }
        if input.exact_evidence.observed_head_id() != input.observed.id()
            || input.exact_evidence.registry_bound_head_id() != input.registry_bound.id()
        {
            violations.push(LineageBoundUpgradeActivationError::PolicyExactEvidenceMismatch(
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

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, LineageBoundUpgradeActivationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| LineageBoundUpgradeActivationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
