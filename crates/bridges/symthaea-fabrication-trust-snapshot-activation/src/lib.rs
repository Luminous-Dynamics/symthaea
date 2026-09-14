// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Trusted-lineage activation permits for interval-safe fabrication trust rotation.
//!
//! A legitimately authorized future trust rotation is not yet an active trust snapshot. This bridge
//! proves, using an explicit bounded descendant `OperationalClockBasisV1` chain, that activation has
//! definitely occurred for every possible true time in a fresh trusted clock interval and that the
//! proposed snapshot remains valid across that complete interval.

#![deny(unsafe_code)]

use serde::Serialize;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_trust_rotation_authority::{
    ClockGovernedTrustRotationIdV1, ClockGovernedTrustRotationV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const CLOCK_GOVERNED_TRUST_SNAPSHOT_ACTIVATION_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-trust-snapshot-activation.v1";
pub const MAX_TRUST_SNAPSHOT_ACTIVATION_CLOCK_HOPS: usize = 4096;

const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.trust-snapshot-activation-clock-lineage.v1\0";
const ACTIVATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-trust-snapshot-activation.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedTrustSnapshotActivationPermitIdV1(Sha256Digest);

impl ClockGovernedTrustSnapshotActivationPermitIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedTrustSnapshotActivationPermitV1 {
    id: ClockGovernedTrustSnapshotActivationPermitIdV1,
    rotation_id: ClockGovernedTrustRotationIdV1,
    proposed_snapshot_digest: Sha256Digest,
    proposed_snapshot_sequence: u64,
    authorization_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    authorization_operational_basis_id: OperationalClockBasisIdV1,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    current_operational_basis_id: OperationalClockBasisIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_hop_count: usize,
    activates_at_unix_ms: u64,
}

impl ClockGovernedTrustSnapshotActivationPermitV1 {
    pub fn id(&self) -> ClockGovernedTrustSnapshotActivationPermitIdV1 {
        self.id
    }
    pub fn rotation_id(&self) -> ClockGovernedTrustRotationIdV1 {
        self.rotation_id
    }
    pub fn proposed_snapshot_digest(&self) -> Sha256Digest {
        self.proposed_snapshot_digest
    }
    pub fn proposed_snapshot_sequence(&self) -> u64 {
        self.proposed_snapshot_sequence
    }
    pub fn authorization_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.authorization_clock_envelope_id
    }
    pub fn authorization_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.authorization_operational_basis_id
    }
    pub fn current_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.current_clock_envelope_id
    }
    pub fn current_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.current_operational_basis_id
    }
    pub fn clock_lineage_digest(&self) -> Sha256Digest {
        self.clock_lineage_digest
    }
    pub fn clock_hop_count(&self) -> usize {
        self.clock_hop_count
    }
    pub fn activates_at_unix_ms(&self) -> u64 {
        self.activates_at_unix_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustSnapshotActivationError {
    AuthorizationBasisMismatch,
    AuthorizationEnvelopeMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    Clock(ClockGovernanceTimeError),
    ActivationNotYetCertain,
    SnapshotNotValidAcrossCurrentEnvelope(ClockGovernanceTimeError),
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct ActivationCommitment {
    schema: &'static str,
    rotation_id: String,
    proposed_snapshot_digest: String,
    proposed_snapshot_sequence: u64,
    authorization_clock_envelope_id: String,
    authorization_operational_basis_id: String,
    current_clock_envelope_id: String,
    current_operational_basis_id: String,
    clock_lineage_digest: String,
    clock_hop_count: usize,
    activates_at_unix_ms: u64,
}

pub fn derive_clock_governed_trust_snapshot_activation_permit_v1(
    rotation: &ClockGovernedTrustRotationV1,
    authorization_basis: &OperationalClockBasisV1,
    clock_bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<ClockGovernedTrustSnapshotActivationPermitV1, TrustSnapshotActivationError> {
    if authorization_basis.id() != rotation.operational_basis_id() {
        return Err(TrustSnapshotActivationError::AuthorizationBasisMismatch);
    }
    let authorization_clock = derive_clock_governance_evaluation_envelope_v1(authorization_basis)
        .map_err(TrustSnapshotActivationError::Clock)?;
    if authorization_clock.id() != rotation.clock_envelope_id() {
        return Err(TrustSnapshotActivationError::AuthorizationEnvelopeMismatch);
    }
    if clock_bridge.len() > MAX_TRUST_SNAPSHOT_ACTIVATION_CLOCK_HOPS {
        return Err(TrustSnapshotActivationError::TooManyClockHops {
            actual: clock_bridge.len(),
            maximum: MAX_TRUST_SNAPSHOT_ACTIVATION_CLOCK_HOPS,
        });
    }
    verify_clock_lineage(authorization_basis.id(), clock_bridge, current_basis)?;

    let current_clock = derive_clock_governance_evaluation_envelope_v1(current_basis)
        .map_err(TrustSnapshotActivationError::Clock)?;
    if rotation.activates_at_unix_ms() > current_clock.lower_unix_ms() {
        return Err(TrustSnapshotActivationError::ActivationNotYetCertain);
    }
    let snapshot = rotation.proposed_snapshot();
    current_clock
        .require_valid_across_seconds_window(snapshot.issued_at_unix_s, snapshot.expires_at_unix_s)
        .map_err(TrustSnapshotActivationError::SnapshotNotValidAcrossCurrentEnvelope)?;

    let mut lineage = Vec::with_capacity(clock_bridge.len() + 2);
    lineage.push(authorization_basis.id().to_hex());
    lineage.extend(clock_bridge.iter().map(|basis| basis.id().to_hex()));
    if current_basis.id() != authorization_basis.id() {
        lineage.push(current_basis.id().to_hex());
    }
    let clock_lineage_digest =
        hash_serializable(CLOCK_LINEAGE_DOMAIN, &lineage)?;
    let clock_hop_count = if current_basis.id() == authorization_basis.id() {
        0
    } else {
        clock_bridge.len() + 1
    };

    let commitment = ActivationCommitment {
        schema: CLOCK_GOVERNED_TRUST_SNAPSHOT_ACTIVATION_SCHEMA,
        rotation_id: rotation.id().to_hex(),
        proposed_snapshot_digest: rotation.proposed_snapshot_digest().to_hex(),
        proposed_snapshot_sequence: snapshot.sequence,
        authorization_clock_envelope_id: authorization_clock.id().to_hex(),
        authorization_operational_basis_id: authorization_basis.id().to_hex(),
        current_clock_envelope_id: current_clock.id().to_hex(),
        current_operational_basis_id: current_basis.id().to_hex(),
        clock_lineage_digest: clock_lineage_digest.to_hex(),
        clock_hop_count,
        activates_at_unix_ms: rotation.activates_at_unix_ms(),
    };
    let id = ClockGovernedTrustSnapshotActivationPermitIdV1(hash_serializable(
        ACTIVATION_DOMAIN,
        &commitment,
    )?);

    Ok(ClockGovernedTrustSnapshotActivationPermitV1 {
        id,
        rotation_id: rotation.id(),
        proposed_snapshot_digest: rotation.proposed_snapshot_digest(),
        proposed_snapshot_sequence: snapshot.sequence,
        authorization_clock_envelope_id: authorization_clock.id(),
        authorization_operational_basis_id: authorization_basis.id(),
        current_clock_envelope_id: current_clock.id(),
        current_operational_basis_id: current_basis.id(),
        clock_lineage_digest,
        clock_hop_count,
        activates_at_unix_ms: rotation.activates_at_unix_ms(),
    })
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), TrustSnapshotActivationError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(TrustSnapshotActivationError::BrokenClockLineage {
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
            return Err(TrustSnapshotActivationError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(TrustSnapshotActivationError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, TrustSnapshotActivationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| TrustSnapshotActivationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_limit_is_bounded() {
        assert_eq!(MAX_TRUST_SNAPSHOT_ACTIVATION_CLOCK_HOPS, 4096);
    }
}
