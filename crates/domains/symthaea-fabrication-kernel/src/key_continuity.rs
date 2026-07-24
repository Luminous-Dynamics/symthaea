// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Long-horizon key continuity across trust-snapshot rotation.

use crate::attestation::SignatureAlgorithm;
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const KEY_CONTINUITY_SCHEMA: &str = "symthaea.fabrication.key-continuity.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyContinuityPolicy {
    pub required_usages: BTreeSet<KeyUsage>,
    pub minimum_bridge_keys_per_usage: usize,
    pub minimum_successor_keys_per_usage: usize,
    pub minimum_overlap_s: u64,
    pub require_successor_algorithm_diversity: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedKeyContinuity {
    pub schema_version: String,
    pub previous_snapshot_digest: Sha256Digest,
    pub successor_snapshot_digest: Sha256Digest,
    pub previous_snapshot_sequence: u64,
    pub successor_snapshot_sequence: u64,
    pub transition_at_unix_s: u64,
    pub overlap_ends_at_unix_s: u64,
    pub bridge_keys_by_usage: BTreeMap<KeyUsage, Vec<(SignatureAlgorithm, String)>>,
    pub successor_algorithms_by_usage: BTreeMap<KeyUsage, Vec<SignatureAlgorithm>>,
    pub evidence_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyContinuityError {
    InvalidPolicy,
    SnapshotInvalid(String),
    SequenceNotAdvanced,
    TransitionOutsideSnapshot,
    OverlapOverflow,
    MissingBridgeKeys(KeyUsage),
    InsufficientSuccessorKeys(KeyUsage),
    MissingAlgorithmDiversity(KeyUsage),
    Encoding(String),
}

pub fn verify_key_continuity(
    previous: &TrustSnapshot,
    successor: &TrustSnapshot,
    transition_at_unix_s: u64,
    policy: &KeyContinuityPolicy,
) -> Result<VerifiedKeyContinuity, KeyContinuityError> {
    validate_policy(policy)?;
    previous
        .validate()
        .map_err(|error| KeyContinuityError::SnapshotInvalid(format!("{error:?}")))?;
    successor
        .validate()
        .map_err(|error| KeyContinuityError::SnapshotInvalid(format!("{error:?}")))?;
    if successor.sequence <= previous.sequence {
        return Err(KeyContinuityError::SequenceNotAdvanced);
    }
    if !previous.is_fresh_at(transition_at_unix_s) || !successor.is_fresh_at(transition_at_unix_s) {
        return Err(KeyContinuityError::TransitionOutsideSnapshot);
    }
    let overlap_ends_at_unix_s = transition_at_unix_s
        .checked_add(policy.minimum_overlap_s)
        .ok_or(KeyContinuityError::OverlapOverflow)?;
    if overlap_ends_at_unix_s > previous.expires_at_unix_s
        || overlap_ends_at_unix_s > successor.expires_at_unix_s
    {
        return Err(KeyContinuityError::TransitionOutsideSnapshot);
    }

    let mut bridge_keys_by_usage = BTreeMap::new();
    let mut successor_algorithms_by_usage = BTreeMap::new();
    for usage in &policy.required_usages {
        let previous_keys = eligible_identities(
            previous,
            *usage,
            transition_at_unix_s,
            overlap_ends_at_unix_s,
        );
        let successor_keys = eligible_identities(
            successor,
            *usage,
            transition_at_unix_s,
            overlap_ends_at_unix_s,
        );
        let bridge = previous_keys
            .intersection(&successor_keys)
            .cloned()
            .collect::<Vec<_>>();
        if bridge.len() < policy.minimum_bridge_keys_per_usage {
            return Err(KeyContinuityError::MissingBridgeKeys(*usage));
        }
        if successor_keys.len() < policy.minimum_successor_keys_per_usage {
            return Err(KeyContinuityError::InsufficientSuccessorKeys(*usage));
        }
        let algorithms = successor_keys
            .iter()
            .map(|(algorithm, _)| algorithm.clone())
            .collect::<BTreeSet<_>>();
        if policy.require_successor_algorithm_diversity && algorithms.len() < 2 {
            return Err(KeyContinuityError::MissingAlgorithmDiversity(*usage));
        }
        bridge_keys_by_usage.insert(*usage, bridge);
        successor_algorithms_by_usage.insert(*usage, algorithms.into_iter().collect());
    }
    let previous_snapshot_digest = digest_trust_snapshot(previous)
        .map_err(|error| KeyContinuityError::SnapshotInvalid(format!("{error:?}")))?;
    let successor_snapshot_digest = digest_trust_snapshot(successor)
        .map_err(|error| KeyContinuityError::SnapshotInvalid(format!("{error:?}")))?;
    let mut evidence = VerifiedKeyContinuity {
        schema_version: KEY_CONTINUITY_SCHEMA.into(),
        previous_snapshot_digest,
        successor_snapshot_digest,
        previous_snapshot_sequence: previous.sequence,
        successor_snapshot_sequence: successor.sequence,
        transition_at_unix_s,
        overlap_ends_at_unix_s,
        bridge_keys_by_usage,
        successor_algorithms_by_usage,
        evidence_digest: Sha256Digest([0; 32]),
    };
    validate_key_continuity_shape(&evidence)?;
    evidence.evidence_digest = digest_key_continuity_fields(&evidence)?;
    Ok(evidence)
}

pub fn digest_key_continuity(
    evidence: &VerifiedKeyContinuity,
) -> Result<Sha256Digest, KeyContinuityError> {
    validate_key_continuity_shape(evidence)?;
    let expected = digest_key_continuity_fields(evidence)?;
    if evidence.evidence_digest != expected {
        return Err(KeyContinuityError::SnapshotInvalid(
            "evidence digest mismatch".into(),
        ));
    }
    Ok(expected)
}

fn validate_key_continuity_shape(
    evidence: &VerifiedKeyContinuity,
) -> Result<(), KeyContinuityError> {
    if evidence.schema_version != KEY_CONTINUITY_SCHEMA
        || evidence.previous_snapshot_digest.0 == [0; 32]
        || evidence.successor_snapshot_digest.0 == [0; 32]
        || evidence.previous_snapshot_sequence == 0
        || evidence.successor_snapshot_sequence <= evidence.previous_snapshot_sequence
        || evidence.transition_at_unix_s >= evidence.overlap_ends_at_unix_s
        || evidence.bridge_keys_by_usage.is_empty()
        || evidence.bridge_keys_by_usage.len() != evidence.successor_algorithms_by_usage.len()
    {
        return Err(KeyContinuityError::SnapshotInvalid(
            "invalid evidence shape".into(),
        ));
    }
    for (usage, bridge) in &evidence.bridge_keys_by_usage {
        let Some(algorithms) = evidence.successor_algorithms_by_usage.get(usage) else {
            return Err(KeyContinuityError::SnapshotInvalid("usage mismatch".into()));
        };
        if bridge.is_empty()
            || algorithms.is_empty()
            || bridge.windows(2).any(|pair| pair[0] >= pair[1])
            || algorithms.windows(2).any(|pair| pair[0] >= pair[1])
            || bridge.iter().any(|(algorithm, key_id)| {
                !algorithm.is_canonical()
                    || key_id.trim().is_empty()
                    || key_id != key_id.trim()
                    || key_id.len() > 256
                    || key_id.chars().any(char::is_control)
            })
        {
            return Err(KeyContinuityError::SnapshotInvalid(
                "invalid continuity set".into(),
            ));
        }
    }
    Ok(())
}

fn digest_key_continuity_fields(
    evidence: &VerifiedKeyContinuity,
) -> Result<Sha256Digest, KeyContinuityError> {
    let mut canonical = evidence.clone();
    canonical.evidence_digest = Sha256Digest([0; 32]);
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| KeyContinuityError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.key-continuity-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn eligible_identities(
    snapshot: &TrustSnapshot,
    usage: KeyUsage,
    start: u64,
    end: u64,
) -> BTreeSet<(SignatureAlgorithm, String)> {
    snapshot
        .keys
        .iter()
        .filter(|key| eligible_for_interval(key, usage, start, end))
        .map(|key| (key.algorithm.clone(), key.key_id.clone()))
        .collect()
}

fn eligible_for_interval(key: &KeyTrustRecord, usage: KeyUsage, start: u64, end: u64) -> bool {
    key.status == KeyLifecycleStatus::Active
        && key.usages.contains(&usage)
        && key.not_before_unix_s <= start
        && key
            .not_after_unix_s
            .is_none_or(|not_after| not_after >= end)
}

fn validate_policy(policy: &KeyContinuityPolicy) -> Result<(), KeyContinuityError> {
    if policy.required_usages.is_empty()
        || policy.minimum_bridge_keys_per_usage == 0
        || policy.minimum_successor_keys_per_usage == 0
        || policy.minimum_overlap_s == 0
    {
        return Err(KeyContinuityError::InvalidPolicy);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_required_usage_set_is_rejected() {
        let policy = KeyContinuityPolicy {
            required_usages: BTreeSet::new(),
            minimum_bridge_keys_per_usage: 1,
            minimum_successor_keys_per_usage: 1,
            minimum_overlap_s: 1,
            require_successor_algorithm_diversity: false,
        };
        assert_eq!(
            validate_policy(&policy),
            Err(KeyContinuityError::InvalidPolicy)
        );
    }
}
