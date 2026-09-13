// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime activation permits for interval-safe fabrication governance.
//!
//! Authorization to migrate is intentionally distinct from authority to make a
//! migration effective. This crate requires a fresh operational-clock lineage
//! descending from the clock basis that authorized the migration, then proves
//! activation has definitely been reached for every possible true time in the
//! current clock envelope.

#![deny(unsafe_code)]

use serde::Serialize;
use symthaea_fabrication_governance_bridge::{
    ClockGovernedPolicyMigrationIdV1, ClockGovernedPolicyMigrationV1,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::policy_migration::PolicyInvariantDisposition;
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError,
    OperationalClockBasisIdV1, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const CLOCK_GOVERNED_POLICY_ACTIVATION_PERMIT_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-policy-activation-permit.v1";
pub const MAX_POLICY_ACTIVATION_CLOCK_HOPS: usize = 4096;

const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-policy-activation-lineage.v1\0";
const ACTIVATION_PERMIT_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-policy-activation-permit.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedPolicyActivationPermitIdV1(Sha256Digest);

impl ClockGovernedPolicyActivationPermitIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that one previously authorized migration may now become
/// effective under a fresh descendant operational-clock basis.
///
/// This capability deliberately does not claim that the migration's predecessor
/// policy is the globally current policy. Durable policy-currentness is a
/// separate lineage theorem. It proves only temporal activation permission for
/// this exact migration and its exact clock ancestry.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedPolicyActivationPermitV1 {
    id: ClockGovernedPolicyActivationPermitIdV1,
    migration_id: ClockGovernedPolicyMigrationIdV1,
    authorization_operational_basis_id: OperationalClockBasisIdV1,
    activation_operational_basis_id: OperationalClockBasisIdV1,
    authorization_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    activation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_lineage_hops: usize,
    activates_at_unix_s: u64,
    rollback_deadline_unix_s: u64,
}

impl ClockGovernedPolicyActivationPermitV1 {
    pub fn id(&self) -> ClockGovernedPolicyActivationPermitIdV1 {
        self.id
    }

    pub fn migration_id(&self) -> ClockGovernedPolicyMigrationIdV1 {
        self.migration_id
    }

    pub fn authorization_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.authorization_operational_basis_id
    }

    pub fn activation_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.activation_operational_basis_id
    }

    pub fn authorization_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.authorization_clock_envelope_id
    }

    pub fn activation_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.activation_clock_envelope_id
    }

    pub fn clock_lineage_digest(&self) -> Sha256Digest {
        self.clock_lineage_digest
    }

    pub fn clock_lineage_hops(&self) -> usize {
        self.clock_lineage_hops
    }

    pub fn activates_at_unix_s(&self) -> u64 {
        self.activates_at_unix_s
    }

    pub fn rollback_deadline_unix_s(&self) -> u64 {
        self.rollback_deadline_unix_s
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernedPolicyActivationError {
    TooManyClockLineageHops {
        actual: usize,
        maximum: usize,
    },
    ClockEnvelope(ClockGovernanceTimeError),
    AuthorizationClockMismatch,
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    TimeScaleOverflow,
    ActivationNotYetCertain {
        activation_unix_ms: u64,
        current_lower_unix_ms: u64,
    },
    RollbackWindowMayBeClosed {
        rollback_deadline_unix_ms: u64,
        current_upper_unix_ms: u64,
    },
    WaiverMayBeExpired {
        invariant: String,
        expires_at_unix_ms: u64,
        current_upper_unix_ms: u64,
    },
    Encoding(String),
}

/// Derive runtime activation permission from an already opaque migration
/// authorization and an explicit bounded chain of opaque operational clocks.
///
/// `successor_chain` must begin with an immediate successor of
/// `authorization_basis` and remain unbroken through the final current basis.
/// There is deliberately no caller-selected current timestamp.
pub fn derive_clock_governed_policy_activation_permit_v1(
    migration: &ClockGovernedPolicyMigrationV1,
    authorization_basis: &OperationalClockBasisV1,
    successor_chain: &[OperationalClockBasisV1],
) -> Result<ClockGovernedPolicyActivationPermitV1, ClockGovernedPolicyActivationError> {
    if successor_chain.len() > MAX_POLICY_ACTIVATION_CLOCK_HOPS {
        return Err(ClockGovernedPolicyActivationError::TooManyClockLineageHops {
            actual: successor_chain.len(),
            maximum: MAX_POLICY_ACTIVATION_CLOCK_HOPS,
        });
    }

    let authorization_envelope = derive_clock_governance_evaluation_envelope_v1(authorization_basis)
        .map_err(ClockGovernedPolicyActivationError::ClockEnvelope)?;
    if authorization_envelope.id() != migration.clock_envelope_id() {
        return Err(ClockGovernedPolicyActivationError::AuthorizationClockMismatch);
    }

    let mut lineage_ids = Vec::with_capacity(successor_chain.len() + 1);
    lineage_ids.push(authorization_basis.id());
    let mut expected_predecessor = authorization_basis.id();
    for (index, basis) in successor_chain.iter().enumerate() {
        let actual = basis.predecessor_operational_basis_id();
        if actual != Some(expected_predecessor) {
            return Err(ClockGovernedPolicyActivationError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected_predecessor.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected_predecessor = basis.id();
        lineage_ids.push(basis.id());
    }

    let activation_basis = successor_chain.last().unwrap_or(authorization_basis);
    let activation_envelope = derive_clock_governance_evaluation_envelope_v1(activation_basis)
        .map_err(ClockGovernedPolicyActivationError::ClockEnvelope)?;

    let plan = migration.plan();
    let activation_unix_ms = seconds_to_millis(plan.activates_at_unix_s)?;
    if activation_unix_ms > activation_envelope.lower_unix_ms() {
        return Err(ClockGovernedPolicyActivationError::ActivationNotYetCertain {
            activation_unix_ms,
            current_lower_unix_ms: activation_envelope.lower_unix_ms(),
        });
    }

    let rollback_deadline_unix_ms = seconds_to_millis(plan.rollback_deadline_unix_s)?;
    if rollback_deadline_unix_ms <= activation_envelope.upper_unix_ms() {
        return Err(ClockGovernedPolicyActivationError::RollbackWindowMayBeClosed {
            rollback_deadline_unix_ms,
            current_upper_unix_ms: activation_envelope.upper_unix_ms(),
        });
    }

    for item in &plan.migrations {
        if let PolicyInvariantDisposition::Waived {
            expires_at_unix_s, ..
        } = &item.disposition
        {
            let expires_at_unix_ms = seconds_to_millis(*expires_at_unix_s)?;
            if expires_at_unix_ms <= activation_envelope.upper_unix_ms() {
                return Err(ClockGovernedPolicyActivationError::WaiverMayBeExpired {
                    invariant: item.name.clone(),
                    expires_at_unix_ms,
                    current_upper_unix_ms: activation_envelope.upper_unix_ms(),
                });
            }
        }
    }

    let clock_lineage_digest = digest_clock_lineage(&lineage_ids)?;
    let id = ClockGovernedPolicyActivationPermitIdV1(digest_activation_permit(
        migration.id(),
        authorization_basis.id(),
        activation_basis.id(),
        authorization_envelope.id(),
        activation_envelope.id(),
        clock_lineage_digest,
        successor_chain.len(),
        plan.activates_at_unix_s,
        plan.rollback_deadline_unix_s,
    )?);

    Ok(ClockGovernedPolicyActivationPermitV1 {
        id,
        migration_id: migration.id(),
        authorization_operational_basis_id: authorization_basis.id(),
        activation_operational_basis_id: activation_basis.id(),
        authorization_clock_envelope_id: authorization_envelope.id(),
        activation_clock_envelope_id: activation_envelope.id(),
        clock_lineage_digest,
        clock_lineage_hops: successor_chain.len(),
        activates_at_unix_s: plan.activates_at_unix_s,
        rollback_deadline_unix_s: plan.rollback_deadline_unix_s,
    })
}

fn seconds_to_millis(value: u64) -> Result<u64, ClockGovernedPolicyActivationError> {
    value
        .checked_mul(1_000)
        .ok_or(ClockGovernedPolicyActivationError::TimeScaleOverflow)
}

#[derive(Serialize)]
struct ClockLineageCommitment {
    basis_ids: Vec<String>,
}

fn digest_clock_lineage(
    basis_ids: &[OperationalClockBasisIdV1],
) -> Result<Sha256Digest, ClockGovernedPolicyActivationError> {
    let bytes = serde_json::to_vec(&ClockLineageCommitment {
        basis_ids: basis_ids.iter().map(|value| value.to_hex()).collect(),
    })
    .map_err(|error| ClockGovernedPolicyActivationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(CLOCK_LINEAGE_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[derive(Serialize)]
struct ActivationPermitCommitment {
    schema: &'static str,
    migration_id: String,
    authorization_operational_basis_id: String,
    activation_operational_basis_id: String,
    authorization_clock_envelope_id: String,
    activation_clock_envelope_id: String,
    clock_lineage_digest: String,
    clock_lineage_hops: usize,
    activates_at_unix_s: u64,
    rollback_deadline_unix_s: u64,
}

#[allow(clippy::too_many_arguments)]
fn digest_activation_permit(
    migration_id: ClockGovernedPolicyMigrationIdV1,
    authorization_operational_basis_id: OperationalClockBasisIdV1,
    activation_operational_basis_id: OperationalClockBasisIdV1,
    authorization_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    activation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_lineage_hops: usize,
    activates_at_unix_s: u64,
    rollback_deadline_unix_s: u64,
) -> Result<Sha256Digest, ClockGovernedPolicyActivationError> {
    let bytes = serde_json::to_vec(&ActivationPermitCommitment {
        schema: CLOCK_GOVERNED_POLICY_ACTIVATION_PERMIT_SCHEMA,
        migration_id: migration_id.to_hex(),
        authorization_operational_basis_id: authorization_operational_basis_id.to_hex(),
        activation_operational_basis_id: activation_operational_basis_id.to_hex(),
        authorization_clock_envelope_id: authorization_clock_envelope_id.to_hex(),
        activation_clock_envelope_id: activation_clock_envelope_id.to_hex(),
        clock_lineage_digest: clock_lineage_digest.to_hex(),
        clock_lineage_hops,
        activates_at_unix_s,
        rollback_deadline_unix_s,
    })
    .map_err(|error| ClockGovernedPolicyActivationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(ACTIVATION_PERMIT_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
