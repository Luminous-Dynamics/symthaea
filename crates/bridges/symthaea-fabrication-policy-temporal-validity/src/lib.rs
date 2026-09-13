// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh temporal-validity permits for durable policy-lineage state.
//!
//! A durable lineage state is not timeless live authority when it carries temporary waivers.
//! This crate re-evaluates that exact state against a fresh descendant operational clock without
//! mutating history. It deliberately does not claim that the supplied lineage state is the newest
//! externally stored head; durable-head currentness remains a separate authority theorem.

#![deny(unsafe_code)]

use serde::Serialize;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_policy_lineage::{
    ClockGovernedPolicyLineageIdV1, ClockGovernedPolicyLineageV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const CLOCK_GOVERNED_POLICY_TEMPORAL_VALIDITY_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-policy-temporal-validity.v1";
pub const MAX_POLICY_TEMPORAL_VALIDITY_CLOCK_HOPS: usize = 4096;

const TEMPORAL_VALIDITY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-policy-temporal-validity.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedPolicyTemporalValidityPermitIdV1(Sha256Digest);

impl ClockGovernedPolicyTemporalValidityPermitIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that one exact durable policy-lineage state remains temporally usable under a
/// fresh descendant operational-clock basis.
///
/// This capability proves two things only:
///
/// 1. the supplied clock descends without a skipped/forked hop from the lineage state's latest
///    operational basis; and
/// 2. every unresolved waiver in that exact lineage state remains unexpired for every possible
///    true time in the fresh clock envelope.
///
/// It does not prove that another newer policy-lineage state has not been durably published.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedPolicyTemporalValidityPermitV1 {
    id: ClockGovernedPolicyTemporalValidityPermitIdV1,
    lineage_id: ClockGovernedPolicyLineageIdV1,
    lineage_sequence: u64,
    current_policy_binding_digest: Sha256Digest,
    active_waiver_count: usize,
    prior_operational_basis_id: OperationalClockBasisIdV1,
    current_operational_basis_id: OperationalClockBasisIdV1,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    clock_bridge_hops: usize,
}

impl ClockGovernedPolicyTemporalValidityPermitV1 {
    pub fn id(&self) -> ClockGovernedPolicyTemporalValidityPermitIdV1 {
        self.id
    }

    pub fn lineage_id(&self) -> ClockGovernedPolicyLineageIdV1 {
        self.lineage_id
    }

    pub fn lineage_sequence(&self) -> u64 {
        self.lineage_sequence
    }

    pub fn current_policy_binding_digest(&self) -> Sha256Digest {
        self.current_policy_binding_digest
    }

    pub fn active_waiver_count(&self) -> usize {
        self.active_waiver_count
    }

    pub fn prior_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.prior_operational_basis_id
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernedPolicyTemporalValidityError {
    TooManyClockBridgeHops {
        actual: usize,
        maximum: usize,
    },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    Clock(ClockGovernanceTimeError),
    WaiverMayBeExpired {
        invariant: String,
        expires_at_unix_ms: u64,
        current_upper_unix_ms: u64,
    },
    TimeScaleOverflow,
    Encoding(String),
}

/// Re-evaluate an exact durable lineage state under a fresh descendant clock.
///
/// `clock_bridge` contains any intermediate operational bases strictly between the lineage state's
/// latest basis and `current_basis`. No caller-selected scalar time is accepted.
pub fn derive_clock_governed_policy_temporal_validity_permit_v1(
    lineage: &ClockGovernedPolicyLineageV1,
    clock_bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<ClockGovernedPolicyTemporalValidityPermitV1, ClockGovernedPolicyTemporalValidityError> {
    if clock_bridge.len() > MAX_POLICY_TEMPORAL_VALIDITY_CLOCK_HOPS {
        return Err(
            ClockGovernedPolicyTemporalValidityError::TooManyClockBridgeHops {
                actual: clock_bridge.len(),
                maximum: MAX_POLICY_TEMPORAL_VALIDITY_CLOCK_HOPS,
            },
        );
    }
    verify_clock_lineage(
        lineage.latest_operational_basis_id(),
        clock_bridge,
        current_basis,
    )?;
    let envelope = derive_clock_governance_evaluation_envelope_v1(current_basis)
        .map_err(ClockGovernedPolicyTemporalValidityError::Clock)?;

    for waiver in lineage.active_waivers() {
        let expires_at_unix_ms = waiver
            .expires_at_unix_s()
            .checked_mul(1_000)
            .ok_or(ClockGovernedPolicyTemporalValidityError::TimeScaleOverflow)?;
        if expires_at_unix_ms <= envelope.upper_unix_ms() {
            return Err(
                ClockGovernedPolicyTemporalValidityError::WaiverMayBeExpired {
                    invariant: waiver.invariant().to_string(),
                    expires_at_unix_ms,
                    current_upper_unix_ms: envelope.upper_unix_ms(),
                },
            );
        }
    }

    let id = ClockGovernedPolicyTemporalValidityPermitIdV1(digest_temporal_validity(
        lineage.id(),
        lineage.sequence(),
        lineage.current_policy_binding_digest(),
        lineage.active_waivers().len(),
        lineage.latest_operational_basis_id(),
        current_basis.id(),
        envelope.id(),
        clock_bridge.len(),
    )?);

    Ok(ClockGovernedPolicyTemporalValidityPermitV1 {
        id,
        lineage_id: lineage.id(),
        lineage_sequence: lineage.sequence(),
        current_policy_binding_digest: lineage.current_policy_binding_digest(),
        active_waiver_count: lineage.active_waivers().len(),
        prior_operational_basis_id: lineage.latest_operational_basis_id(),
        current_operational_basis_id: current_basis.id(),
        current_clock_envelope_id: envelope.id(),
        clock_bridge_hops: clock_bridge.len(),
    })
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), ClockGovernedPolicyTemporalValidityError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(ClockGovernedPolicyTemporalValidityError::BrokenClockLineage {
            hop: 1,
            expected_predecessor: prior_basis_id.to_hex(),
            actual_predecessor: bridge[0]
                .predecessor_operational_basis_id()
                .map(|value| value.to_hex()),
        });
    }

    let mut expected = prior_basis_id;
    for (index, basis) in bridge.iter().enumerate() {
        let actual = basis.predecessor_operational_basis_id();
        if actual != Some(expected) {
            return Err(ClockGovernedPolicyTemporalValidityError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(ClockGovernedPolicyTemporalValidityError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

#[derive(Serialize)]
struct TemporalValidityCommitment {
    schema: &'static str,
    lineage_id: String,
    lineage_sequence: u64,
    current_policy_binding_digest: String,
    active_waiver_count: usize,
    prior_operational_basis_id: String,
    current_operational_basis_id: String,
    current_clock_envelope_id: String,
    clock_bridge_hops: usize,
}

#[allow(clippy::too_many_arguments)]
fn digest_temporal_validity(
    lineage_id: ClockGovernedPolicyLineageIdV1,
    lineage_sequence: u64,
    current_policy_binding_digest: Sha256Digest,
    active_waiver_count: usize,
    prior_operational_basis_id: OperationalClockBasisIdV1,
    current_operational_basis_id: OperationalClockBasisIdV1,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    clock_bridge_hops: usize,
) -> Result<Sha256Digest, ClockGovernedPolicyTemporalValidityError> {
    let bytes = serde_json::to_vec(&TemporalValidityCommitment {
        schema: CLOCK_GOVERNED_POLICY_TEMPORAL_VALIDITY_SCHEMA,
        lineage_id: lineage_id.to_hex(),
        lineage_sequence,
        current_policy_binding_digest: current_policy_binding_digest.to_hex(),
        active_waiver_count,
        prior_operational_basis_id: prior_operational_basis_id.to_hex(),
        current_operational_basis_id: current_operational_basis_id.to_hex(),
        current_clock_envelope_id: current_clock_envelope_id.to_hex(),
        clock_bridge_hops,
    })
    .map_err(|error| ClockGovernedPolicyTemporalValidityError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(TEMPORAL_VALIDITY_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
