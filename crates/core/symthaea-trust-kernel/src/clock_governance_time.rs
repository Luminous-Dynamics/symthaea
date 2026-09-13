// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interval-safe governance time derived from recursive operational clock authority.
//!
//! Governance code must not collapse an accepted clock interval to one scalar
//! `now`. This capability preserves the exact interval and exposes conservative
//! temporal predicates that must hold for every possible true time inside it.

use crate::clock_operational::{OperationalClockBasisIdV1, OperationalClockBasisV1};
use crate::digest::{Sha256Digest, domain_hash};
use serde_json::Value;
use std::collections::BTreeMap;

pub const CLOCK_GOVERNANCE_EVALUATION_ENVELOPE_SCHEMA: &str =
    "symthaea.trust.clock-governance-evaluation-envelope.v1";
const CLOCK_GOVERNANCE_EVALUATION_ENVELOPE_DOMAIN: &[u8] =
    b"symthaea.trust.clock-governance-evaluation-envelope.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernanceEvaluationEnvelopeIdV1(Sha256Digest);

#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernanceEvaluationEnvelopeV1 {
    id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
    clock_window_evidence_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    epoch: u64,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    consensus_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernanceTimeError {
    InvalidEnvelope,
    InvalidValidityWindow,
    NotValidAcrossEnvelope,
    InactiveAuthority,
    UsageNotAllowed,
    ActivationMayBeInPast,
    ActivationMayBeTooLate,
    TimeScaleOverflow,
    ActivationWindowOverflow,
    Encoding(String),
}

impl ClockGovernanceEvaluationEnvelopeIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl ClockGovernanceEvaluationEnvelopeV1 {
    pub fn id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.id
    }

    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }

    pub fn clock_window_evidence_digest(&self) -> Sha256Digest {
        self.clock_window_evidence_digest
    }

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn lower_unix_ms(&self) -> u64 {
        self.lower_unix_ms
    }

    pub fn upper_unix_ms(&self) -> u64 {
        self.upper_unix_ms
    }

    pub fn consensus_unix_ms(&self) -> u64 {
        self.consensus_unix_ms
    }

    /// Require a second-granularity validity interval to cover every possible
    /// true time in the trusted millisecond envelope.
    pub fn require_valid_across_seconds_window(
        &self,
        not_before_unix_s: u64,
        not_after_unix_s: u64,
    ) -> Result<(), ClockGovernanceTimeError> {
        if not_before_unix_s >= not_after_unix_s {
            return Err(ClockGovernanceTimeError::InvalidValidityWindow);
        }
        let not_before_ms = seconds_to_millis(not_before_unix_s)?;
        let not_after_ms = seconds_to_millis(not_after_unix_s)?;
        if not_before_ms > self.lower_unix_ms || not_after_ms <= self.upper_unix_ms {
            return Err(ClockGovernanceTimeError::NotValidAcrossEnvelope);
        }
        Ok(())
    }

    /// Lifecycle helper for key/provider authority. Status and usage are
    /// supplied by the caller's already-validated semantic record; temporal
    /// eligibility is checked against the entire trusted interval.
    pub fn require_authority_valid_across_seconds_window(
        &self,
        not_before_unix_s: u64,
        not_after_unix_s: u64,
        active: bool,
        usage_allowed: bool,
    ) -> Result<(), ClockGovernanceTimeError> {
        if !active {
            return Err(ClockGovernanceTimeError::InactiveAuthority);
        }
        if !usage_allowed {
            return Err(ClockGovernanceTimeError::UsageNotAllowed);
        }
        self.require_valid_across_seconds_window(not_before_unix_s, not_after_unix_s)
    }

    /// Require an activation to be safe for *every* possible true current time
    /// in the envelope. It must be no earlier than the upper bound and no later
    /// than lower + maximum_delay.
    pub fn require_activation_within_delay_seconds(
        &self,
        activates_at_unix_s: u64,
        maximum_delay_s: u64,
    ) -> Result<(), ClockGovernanceTimeError> {
        let activation_ms = seconds_to_millis(activates_at_unix_s)?;
        let delay_ms = seconds_to_millis(maximum_delay_s)?;
        let latest_ms = self
            .lower_unix_ms
            .checked_add(delay_ms)
            .ok_or(ClockGovernanceTimeError::ActivationWindowOverflow)?;
        if activation_ms < self.upper_unix_ms {
            return Err(ClockGovernanceTimeError::ActivationMayBeInPast);
        }
        if activation_ms > latest_ms {
            return Err(ClockGovernanceTimeError::ActivationMayBeTooLate);
        }
        Ok(())
    }
}

/// Derive governance-time authority solely from an opaque operational clock
/// capability. No caller-supplied current time participates in construction.
pub fn derive_clock_governance_evaluation_envelope_v1(
    basis: &OperationalClockBasisV1,
) -> Result<ClockGovernanceEvaluationEnvelopeV1, ClockGovernanceTimeError> {
    let epoch = basis.epoch();
    let lower_unix_ms = basis.lower_unix_ms();
    let upper_unix_ms = basis.upper_unix_ms();
    let consensus_unix_ms = basis.consensus_unix_ms();
    if epoch == 0
        || lower_unix_ms > consensus_unix_ms
        || consensus_unix_ms > upper_unix_ms
        || basis.clock_window_evidence_digest().0 == [0; 32]
        || basis.trust_snapshot_digest().0 == [0; 32]
    {
        return Err(ClockGovernanceTimeError::InvalidEnvelope);
    }
    let id = ClockGovernanceEvaluationEnvelopeIdV1(compute_envelope_id(
        basis.id(),
        basis.clock_window_evidence_digest(),
        basis.trust_snapshot_digest(),
        epoch,
        lower_unix_ms,
        upper_unix_ms,
        consensus_unix_ms,
    )?);
    Ok(ClockGovernanceEvaluationEnvelopeV1 {
        id,
        operational_basis_id: basis.id(),
        clock_window_evidence_digest: basis.clock_window_evidence_digest(),
        trust_snapshot_digest: basis.trust_snapshot_digest(),
        epoch,
        lower_unix_ms,
        upper_unix_ms,
        consensus_unix_ms,
    })
}

#[allow(clippy::too_many_arguments)]
fn compute_envelope_id(
    operational_basis_id: OperationalClockBasisIdV1,
    clock_window_evidence_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    epoch: u64,
    lower_unix_ms: u64,
    upper_unix_ms: u64,
    consensus_unix_ms: u64,
) -> Result<Sha256Digest, ClockGovernanceTimeError> {
    let bytes = canonical_json_bytes([
        (
            "clock_window_evidence_digest",
            Value::String(clock_window_evidence_digest.to_hex()),
        ),
        ("consensus_unix_ms", Value::from(consensus_unix_ms)),
        ("epoch", Value::from(epoch)),
        ("lower_unix_ms", Value::from(lower_unix_ms)),
        (
            "operational_basis_id",
            Value::String(operational_basis_id.to_hex()),
        ),
        (
            "schema",
            Value::String(CLOCK_GOVERNANCE_EVALUATION_ENVELOPE_SCHEMA.to_string()),
        ),
        (
            "trust_snapshot_digest",
            Value::String(trust_snapshot_digest.to_hex()),
        ),
        ("upper_unix_ms", Value::from(upper_unix_ms)),
    ])?;
    Ok(domain_hash(
        CLOCK_GOVERNANCE_EVALUATION_ENVELOPE_DOMAIN,
        &bytes,
    ))
}

fn seconds_to_millis(value: u64) -> Result<u64, ClockGovernanceTimeError> {
    value
        .checked_mul(1_000)
        .ok_or(ClockGovernanceTimeError::TimeScaleOverflow)
}

fn canonical_json_bytes<const N: usize>(
    entries: [(&str, Value); N],
) -> Result<Vec<u8>, ClockGovernanceTimeError> {
    let map = entries
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect::<BTreeMap<_, _>>();
    serde_json::to_vec(&map).map_err(|error| ClockGovernanceTimeError::Encoding(error.to_string()))
}
