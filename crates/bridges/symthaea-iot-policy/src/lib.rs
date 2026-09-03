// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Anti-rollback policy selection for consequential cyber-physical actuation.
//!
//! A capability says who may attempt an operation. A safety policy says what
//! concrete firmware, command parameters, physical observations, consequence
//! charge and command lifetime are acceptable for that operation. Those are
//! different authority domains and should not be silently collapsed.
//!
//! This crate removes a dangerous convenience from the product path: callers do
//! not hand an arbitrary [`SafetyEnvelope`] and arbitrary non-zero risk charge to
//! the actuation validator. Instead, a configured [`ActuationPolicyRegistry`]
//! issues an opaque [`ActuationPolicyHandle`] for one currently valid policy.
//!
//! Registry snapshots are sequence-numbered, hash chained, canonicalized before
//! hashing, and can be restored only against an externally retained
//! [`ActuationPolicyHead`]. This makes policy rollback/collision detectable once
//! the owner persists the latest head.
//!
//! ## Trust boundary
//!
//! This crate does **not** authenticate where a policy snapshot came from. The
//! process or configuration subsystem that constructs/updates a registry remains
//! part of the trusted computing base. A future signed-policy / Mycelix governance
//! adapter can authenticate snapshots before passing them here without changing
//! the actuation API. The important v0.1 property is that downstream physical I/O
//! cannot accidentally substitute a permissive raw safety envelope after a policy
//! has been selected.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_authority::{Digest32, Operation, ResourceRef, RiskBudget};
use symthaea_iot_authority::{SAFETY_ENVELOPE_SCHEMA_VERSION, SafetyEnvelope};
use thiserror::Error;

/// Current schema generation for an individual actuation policy.
pub const ACTUATION_POLICY_SCHEMA_VERSION: u16 = 1;
/// Current schema generation for a policy registry snapshot.
pub const ACTUATION_POLICY_SNAPSHOT_SCHEMA_VERSION: u16 = 1;
/// Hard ceiling for one registry snapshot.
pub const MAX_ACTUATION_POLICIES: usize = 4_096;
/// Domain separator for individual policy commitments.
pub const ACTUATION_POLICY_DOMAIN: &[u8] = b"symthaea-iot-actuation-policy-v1\0";
/// Domain separator for policy-snapshot commitments.
pub const ACTUATION_POLICY_SNAPSHOT_DOMAIN: &[u8] = b"symthaea-iot-policy-snapshot-v1\0";

/// Trusted-configuration policy for one exact physical resource and operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationPolicyV1 {
    /// Fail-closed schema version.
    pub schema_version: u16,
    /// Stable policy identity used for selection and audit.
    pub policy_id: String,
    /// Monotonic semantic revision of this policy identity.
    pub revision: u64,
    /// Exact protected physical resource.
    pub device: ResourceRef,
    /// Exact protected semantic operation.
    pub operation: Operation,
    /// Complete safety envelope selected by policy rather than the command caller.
    pub safety: SafetyEnvelope,
    /// Exact consequence charge that every command under this policy must reserve.
    pub risk_charge: RiskBudget,
    /// Maximum command validity interval permitted by this policy.
    pub max_command_lifetime_s: u64,
    /// Earliest trusted Unix time when this policy may be selected.
    pub not_before_unix_s: u64,
    /// Optional exclusive policy expiry.
    pub not_after_unix_s: Option<u64>,
}

impl ActuationPolicyV1 {
    /// Validate structural policy invariants independent of current wall clock.
    pub fn validate(&self) -> Result<(), ActuationPolicyError> {
        if self.schema_version != ACTUATION_POLICY_SCHEMA_VERSION {
            return Err(ActuationPolicyError::UnsupportedPolicySchema);
        }
        if self.policy_id.is_empty() || self.policy_id.trim() != self.policy_id {
            return Err(ActuationPolicyError::InvalidPolicyId);
        }
        if self.revision == 0 {
            return Err(ActuationPolicyError::RevisionZero);
        }
        if self.safety.schema_version != SAFETY_ENVELOPE_SCHEMA_VERSION {
            return Err(ActuationPolicyError::UnsupportedSafetySchema);
        }
        if self.safety.device != self.device {
            return Err(ActuationPolicyError::SafetyDeviceMismatch);
        }
        if self.safety.operation != self.operation {
            return Err(ActuationPolicyError::SafetyOperationMismatch);
        }
        if self.safety.policy_id.is_empty()
            || self.safety.allowed_firmware.is_empty()
            || self
                .safety
                .parameter_ranges
                .values()
                .any(|range| !range.is_valid())
            || self
                .safety
                .required_observations
                .values()
                .any(|range| !range.is_valid())
        {
            return Err(ActuationPolicyError::MalformedSafetyEnvelope);
        }
        if self.risk_charge == RiskBudget::default() {
            return Err(ActuationPolicyError::ZeroRiskCharge);
        }
        if self.max_command_lifetime_s == 0 {
            return Err(ActuationPolicyError::ZeroCommandLifetime);
        }
        if self
            .not_after_unix_s
            .is_some_and(|not_after| not_after <= self.not_before_unix_s)
        {
            return Err(ActuationPolicyError::InvalidPolicyWindow);
        }
        Ok(())
    }

    /// Whether this policy is selectable at the trusted wall-clock value.
    pub fn active_at(&self, now_unix_s: u64) -> bool {
        now_unix_s >= self.not_before_unix_s
            && self
                .not_after_unix_s
                .is_none_or(|not_after| now_unix_s < not_after)
    }

    /// Infallible domain-separated commitment after structural validation.
    ///
    /// The safety envelope already has its own deterministic commitment; binding
    /// that commitment keeps this transcript small while still committing every
    /// safety-relevant field.
    pub fn digest(&self) -> Digest32 {
        let mut h = blake3::Hasher::new();
        h.update(ACTUATION_POLICY_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        update_string(&mut h, &self.policy_id);
        h.update(&self.revision.to_be_bytes());
        update_string(&mut h, &self.device.0);
        update_string(&mut h, &self.operation.0);
        let Digest32(safety_digest) = self.safety.digest();
        h.update(&safety_digest);
        update_risk(&mut h, self.risk_charge);
        h.update(&self.max_command_lifetime_s.to_be_bytes());
        h.update(&self.not_before_unix_s.to_be_bytes());
        match self.not_after_unix_s {
            Some(value) => {
                h.update(&[1]);
                h.update(&value.to_be_bytes());
            }
            None => h.update(&[0]),
        }
        Digest32(*h.finalize().as_bytes())
    }
}

/// One complete generation of configured actuation policies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationPolicySnapshotV1 {
    /// Fail-closed schema version.
    pub schema_version: u16,
    /// Monotonic policy-registry generation. Generation zero is invalid.
    pub sequence: u64,
    /// Trusted Unix time at which this snapshot was issued.
    pub issued_at_unix_s: u64,
    /// Exclusive expiry of the snapshot itself.
    pub expires_at_unix_s: u64,
    /// Previous snapshot commitment; absent only for sequence 1.
    pub previous_snapshot_digest: Option<Digest32>,
    /// Active policies in this generation. Policy IDs are unique within a snapshot.
    pub policies: Vec<ActuationPolicyV1>,
}

impl ActuationPolicySnapshotV1 {
    /// Validate snapshot and contained policy structure.
    pub fn validate(&self) -> Result<(), ActuationPolicyError> {
        if self.schema_version != ACTUATION_POLICY_SNAPSHOT_SCHEMA_VERSION {
            return Err(ActuationPolicyError::UnsupportedSnapshotSchema);
        }
        if self.sequence == 0 {
            return Err(ActuationPolicyError::SnapshotSequenceZero);
        }
        if self.issued_at_unix_s >= self.expires_at_unix_s {
            return Err(ActuationPolicyError::InvalidSnapshotWindow);
        }
        if self.policies.is_empty() {
            return Err(ActuationPolicyError::EmptySnapshot);
        }
        if self.policies.len() > MAX_ACTUATION_POLICIES {
            return Err(ActuationPolicyError::TooManyPolicies {
                actual: self.policies.len(),
                maximum: MAX_ACTUATION_POLICIES,
            });
        }
        if self.sequence == 1 && self.previous_snapshot_digest.is_some() {
            return Err(ActuationPolicyError::GenesisHasPredecessor);
        }
        if self.sequence > 1 && self.previous_snapshot_digest.is_none() {
            return Err(ActuationPolicyError::MissingPredecessor);
        }

        let mut ids = BTreeSet::new();
        for policy in &self.policies {
            policy.validate()?;
            if !ids.insert(policy.policy_id.clone()) {
                return Err(ActuationPolicyError::DuplicatePolicyId(
                    policy.policy_id.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Canonical snapshot digest independent of input vector ordering.
    pub fn digest(&self) -> Result<Digest32, ActuationPolicyError> {
        self.validate()?;
        let mut policies = self.policies.iter().collect::<Vec<_>>();
        policies.sort_by(|left, right| left.policy_id.cmp(&right.policy_id));

        let mut h = blake3::Hasher::new();
        h.update(ACTUATION_POLICY_SNAPSHOT_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.sequence.to_be_bytes());
        h.update(&self.issued_at_unix_s.to_be_bytes());
        h.update(&self.expires_at_unix_s.to_be_bytes());
        match self.previous_snapshot_digest {
            Some(Digest32(value)) => {
                h.update(&[1]);
                h.update(&value);
            }
            None => h.update(&[0]),
        }
        h.update(&(policies.len() as u64).to_be_bytes());
        for policy in policies {
            let Digest32(value) = policy.digest();
            h.update(&value);
        }
        Ok(Digest32(*h.finalize().as_bytes()))
    }
}

/// Externally retained anti-rollback anchor for the current policy generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationPolicyHead {
    /// Current registry generation.
    pub sequence: u64,
    /// Commitment to the exact canonical snapshot.
    pub digest: Digest32,
}

/// In-process policy registry whose lineage has been structurally verified.
///
/// The registry is intentionally not `Serialize`/`Deserialize`. Persist the public
/// snapshot plus its externally retained [`ActuationPolicyHead`] and reconstruct via
/// [`Self::restore`] instead.
#[derive(Debug)]
pub struct ActuationPolicyRegistry {
    snapshot: ActuationPolicySnapshotV1,
    head: ActuationPolicyHead,
}

impl ActuationPolicyRegistry {
    /// Accept generation 1 of a newly provisioned policy lineage.
    pub fn genesis(snapshot: ActuationPolicySnapshotV1) -> Result<Self, ActuationPolicyError> {
        snapshot.validate()?;
        if snapshot.sequence != 1 || snapshot.previous_snapshot_digest.is_some() {
            return Err(ActuationPolicyError::NotGenesis);
        }
        let head = ActuationPolicyHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        Ok(Self { snapshot, head })
    }

    /// Verify and accept the immediate successor of this registry generation.
    pub fn successor(
        &self,
        snapshot: ActuationPolicySnapshotV1,
    ) -> Result<Self, ActuationPolicyError> {
        snapshot.validate()?;
        let expected_sequence = self
            .head
            .sequence
            .checked_add(1)
            .ok_or(ActuationPolicyError::SequenceOverflow)?;
        if snapshot.sequence != expected_sequence {
            return Err(ActuationPolicyError::SequenceNotNext {
                expected: expected_sequence,
                proposed: snapshot.sequence,
            });
        }
        if snapshot.previous_snapshot_digest != Some(self.head.digest) {
            return Err(ActuationPolicyError::PredecessorMismatch);
        }
        if snapshot.issued_at_unix_s < self.snapshot.issued_at_unix_s {
            return Err(ActuationPolicyError::IssuedAtRegressed);
        }
        let head = ActuationPolicyHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        Ok(Self { snapshot, head })
    }

    /// Restore a persisted snapshot only when it matches the external trusted head.
    pub fn restore(
        snapshot: ActuationPolicySnapshotV1,
        trusted_head: ActuationPolicyHead,
    ) -> Result<Self, ActuationPolicyError> {
        snapshot.validate()?;
        let head = ActuationPolicyHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        if head != trusted_head {
            return Err(ActuationPolicyError::TrustedHeadMismatch);
        }
        Ok(Self { snapshot, head })
    }

    /// Current externally retainable anti-rollback head.
    pub const fn head(&self) -> ActuationPolicyHead {
        self.head
    }

    /// Read-only current snapshot for persistence/audit.
    pub fn snapshot(&self) -> &ActuationPolicySnapshotV1 {
        &self.snapshot
    }

    /// Select one exact currently active policy from the current snapshot.
    ///
    /// The returned handle borrows this registry. It therefore cannot outlive the
    /// registry generation that authorized it and cannot be deserialized from bytes.
    pub fn policy(
        &self,
        policy_id: &str,
        now_unix_s: u64,
    ) -> Result<ActuationPolicyHandle<'_>, ActuationPolicyError> {
        if now_unix_s < self.snapshot.issued_at_unix_s
            || now_unix_s >= self.snapshot.expires_at_unix_s
        {
            return Err(ActuationPolicyError::SnapshotNotFresh);
        }
        let policy = self
            .snapshot
            .policies
            .iter()
            .find(|policy| policy.policy_id == policy_id)
            .ok_or_else(|| ActuationPolicyError::UnknownPolicy(policy_id.to_string()))?;
        if !policy.active_at(now_unix_s) {
            return Err(ActuationPolicyError::PolicyNotActive);
        }
        Ok(ActuationPolicyHandle {
            registry_head: self.head,
            policy,
            policy_digest: policy.digest(),
            selected_at_unix_s: now_unix_s,
        })
    }
}

/// Opaque, generation-bound policy selection used by the product actuation path.
///
/// This is a borrowing handle with private fields and no serialization. A caller may
/// inspect the selected policy, but cannot manufacture a handle by deserializing a
/// permissive safety envelope.
#[derive(Debug)]
pub struct ActuationPolicyHandle<'a> {
    registry_head: ActuationPolicyHead,
    policy: &'a ActuationPolicyV1,
    policy_digest: Digest32,
    selected_at_unix_s: u64,
}

impl ActuationPolicyHandle<'_> {
    /// Exact selected policy.
    pub fn policy(&self) -> &ActuationPolicyV1 {
        self.policy
    }

    /// Exact policy commitment.
    pub const fn policy_digest(&self) -> Digest32 {
        self.policy_digest
    }

    /// Registry generation that authorized this handle.
    pub const fn registry_head(&self) -> ActuationPolicyHead {
        self.registry_head
    }

    /// Trusted selection time.
    pub const fn selected_at_unix_s(&self) -> u64 {
        self.selected_at_unix_s
    }

    /// Selected safety envelope.
    pub fn safety(&self) -> &SafetyEnvelope {
        &self.policy.safety
    }

    /// Exact consequence charge dictated by policy.
    pub const fn risk_charge(&self) -> RiskBudget {
        self.policy.risk_charge
    }

    /// Maximum command validity interval dictated by policy.
    pub const fn max_command_lifetime_s(&self) -> u64 {
        self.policy.max_command_lifetime_s
    }
}

/// Structural, lineage, or selection failure for actuation policy.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ActuationPolicyError {
    #[error("unsupported actuation policy schema")]
    UnsupportedPolicySchema,
    #[error("unsupported actuation policy snapshot schema")]
    UnsupportedSnapshotSchema,
    #[error("unsupported safety envelope schema")]
    UnsupportedSafetySchema,
    #[error("policy id must be non-empty and canonical")]
    InvalidPolicyId,
    #[error("policy revision must be non-zero")]
    RevisionZero,
    #[error("policy safety envelope protects a different device")]
    SafetyDeviceMismatch,
    #[error("policy safety envelope protects a different operation")]
    SafetyOperationMismatch,
    #[error("policy contains a malformed safety envelope")]
    MalformedSafetyEnvelope,
    #[error("policy consequence charge must be non-zero")]
    ZeroRiskCharge,
    #[error("policy command lifetime must be non-zero")]
    ZeroCommandLifetime,
    #[error("policy validity window is invalid")]
    InvalidPolicyWindow,
    #[error("policy snapshot sequence must be non-zero")]
    SnapshotSequenceZero,
    #[error("policy snapshot validity window is invalid")]
    InvalidSnapshotWindow,
    #[error("policy snapshot must contain at least one policy")]
    EmptySnapshot,
    #[error("policy snapshot contains {actual} policies, maximum is {maximum}")]
    TooManyPolicies { actual: usize, maximum: usize },
    #[error("generation 1 policy snapshot must not have a predecessor")]
    GenesisHasPredecessor,
    #[error("non-genesis policy snapshot must bind its predecessor")]
    MissingPredecessor,
    #[error("duplicate policy id in one snapshot: {0}")]
    DuplicatePolicyId(String),
    #[error("snapshot is not a valid genesis generation")]
    NotGenesis,
    #[error("policy registry sequence overflow")]
    SequenceOverflow,
    #[error("policy snapshot sequence is not the immediate successor: expected {expected}, proposed {proposed}")]
    SequenceNotNext { expected: u64, proposed: u64 },
    #[error("policy snapshot does not bind the current registry head")]
    PredecessorMismatch,
    #[error("policy snapshot issued-at time regressed")]
    IssuedAtRegressed,
    #[error("restored policy snapshot does not match externally retained head")]
    TrustedHeadMismatch,
    #[error("policy snapshot is not fresh at the selection time")]
    SnapshotNotFresh,
    #[error("unknown actuation policy: {0}")]
    UnknownPolicy(String),
    #[error("actuation policy is not active at the selection time")]
    PolicyNotActive,
}

fn update_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn update_risk(hasher: &mut blake3::Hasher, risk: RiskBudget) {
    hasher.update(&risk.mutation_units.to_be_bytes());
    hasher.update(&risk.irreversible_units.to_be_bytes());
    hasher.update(&risk.external_disclosure_bytes.to_be_bytes());
    hasher.update(&risk.monetary_microunits.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use symthaea_iot_authority::InclusiveRangeI64;

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn risk(units: u64) -> RiskBudget {
        RiskBudget {
            mutation_units: units,
            ..RiskBudget::default()
        }
    }

    fn policy(id: &str, revision: u64) -> ActuationPolicyV1 {
        let device = ResourceRef(format!("iot:valve:{id}"));
        let operation = Operation("valve.open".into());
        ActuationPolicyV1 {
            schema_version: ACTUATION_POLICY_SCHEMA_VERSION,
            policy_id: id.into(),
            revision,
            device: device.clone(),
            operation: operation.clone(),
            safety: SafetyEnvelope {
                schema_version: SAFETY_ENVELOPE_SCHEMA_VERSION,
                policy_id: format!("safety-{id}"),
                device,
                operation,
                allowed_firmware: BTreeSet::from([digest(7)]),
                parameter_ranges: BTreeMap::from([(
                    "duration_ms".into(),
                    InclusiveRangeI64 {
                        min: 1_000,
                        max: 120_000,
                    },
                )]),
                required_observations: BTreeMap::from([(
                    "pressure_x100".into(),
                    InclusiveRangeI64 {
                        min: 100,
                        max: 350_000,
                    },
                )]),
            },
            risk_charge: risk(2),
            max_command_lifetime_s: 30,
            not_before_unix_s: 100,
            not_after_unix_s: Some(900),
        }
    }

    fn genesis(policies: Vec<ActuationPolicyV1>) -> ActuationPolicySnapshotV1 {
        ActuationPolicySnapshotV1 {
            schema_version: ACTUATION_POLICY_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_s: 100,
            expires_at_unix_s: 1_000,
            previous_snapshot_digest: None,
            policies,
        }
    }

    #[test]
    fn snapshot_digest_is_independent_of_policy_input_order() {
        let left = genesis(vec![policy("b", 1), policy("a", 1)]);
        let right = genesis(vec![policy("a", 1), policy("b", 1)]);
        assert_eq!(left.digest().unwrap(), right.digest().unwrap());
    }

    #[test]
    fn handle_is_bound_to_exact_registry_generation_and_policy() {
        let registry = ActuationPolicyRegistry::genesis(genesis(vec![policy("a", 1)])).unwrap();
        let handle = registry.policy("a", 500).unwrap();
        assert_eq!(handle.registry_head(), registry.head());
        assert_eq!(handle.policy_digest(), handle.policy().digest());
        assert_eq!(handle.risk_charge(), risk(2));
        assert_eq!(handle.max_command_lifetime_s(), 30);
    }

    #[test]
    fn successor_must_be_immediate_and_hash_chained() {
        let registry = ActuationPolicyRegistry::genesis(genesis(vec![policy("a", 1)])).unwrap();
        let next = ActuationPolicySnapshotV1 {
            schema_version: ACTUATION_POLICY_SNAPSHOT_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_s: 200,
            expires_at_unix_s: 1_100,
            previous_snapshot_digest: Some(registry.head().digest),
            policies: vec![policy("a", 2)],
        };
        let successor = registry.successor(next).unwrap();
        assert_eq!(successor.head().sequence, 2);

        let skipped = ActuationPolicySnapshotV1 {
            schema_version: ACTUATION_POLICY_SNAPSHOT_SCHEMA_VERSION,
            sequence: 4,
            issued_at_unix_s: 300,
            expires_at_unix_s: 1_200,
            previous_snapshot_digest: Some(successor.head().digest),
            policies: vec![policy("a", 3)],
        };
        assert!(matches!(
            successor.successor(skipped),
            Err(ActuationPolicyError::SequenceNotNext { .. })
        ));
    }

    #[test]
    fn restore_rejects_wrong_external_head() {
        let snapshot = genesis(vec![policy("a", 1)]);
        let registry = ActuationPolicyRegistry::genesis(snapshot.clone()).unwrap();
        let wrong = ActuationPolicyHead {
            sequence: 1,
            digest: digest(99),
        };
        assert!(matches!(
            ActuationPolicyRegistry::restore(snapshot, wrong),
            Err(ActuationPolicyError::TrustedHeadMismatch)
        ));
        assert_ne!(registry.head(), wrong);
    }

    #[test]
    fn expired_snapshot_or_policy_cannot_mint_handle() {
        let registry = ActuationPolicyRegistry::genesis(genesis(vec![policy("a", 1)])).unwrap();
        assert!(matches!(
            registry.policy("a", 1_000),
            Err(ActuationPolicyError::SnapshotNotFresh)
        ));
        assert!(matches!(
            registry.policy("a", 950),
            Err(ActuationPolicyError::PolicyNotActive)
        ));
    }

    #[test]
    fn malformed_or_zero_risk_policy_is_rejected() {
        let mut zero = policy("a", 1);
        zero.risk_charge = RiskBudget::default();
        assert_eq!(zero.validate(), Err(ActuationPolicyError::ZeroRiskCharge));

        let mut mismatched = policy("a", 1);
        mismatched.safety.device = ResourceRef("iot:other".into());
        assert_eq!(
            mismatched.validate(),
            Err(ActuationPolicyError::SafetyDeviceMismatch)
        );
    }
}
