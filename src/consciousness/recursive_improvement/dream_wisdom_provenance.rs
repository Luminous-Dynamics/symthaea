// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical provenance identities for generated dream wisdom.
//!
//! `DreamResult` carries aggregate counters and best-score summaries; it does not
//! identify the concrete counterfactual records that caused downstream effects.
//! This module binds each generated `Wisdom<Vec<f32>>` value to a deterministic,
//! collision-resistant BLAKE3 identity and can bind an ordered batch produced by
//! one dream cycle to a second digest.
//!
//! These identities are provenance only. They do not make generated wisdom
//! empirical, validated, or eligible to promote confidence.

use symthaea_dream::Wisdom;

const WISDOM_DOMAIN: &[u8] = b"symthaea/dream-wisdom/v1\0";
const BATCH_DOMAIN: &[u8] = b"symthaea/dream-wisdom-batch/v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DreamWisdomIdentity {
    pub digest: [u8; 32],
}

impl DreamWisdomIdentity {
    pub fn to_hex(self) -> String {
        hex::encode(self.digest)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DreamWisdomBatchIdentity {
    pub digest: [u8; 32],
    pub wisdom_count: usize,
}

impl DreamWisdomBatchIdentity {
    pub fn to_hex(self) -> String {
        hex::encode(self.digest)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DreamWisdomProvenanceError {
    EmptyContext,
    EmptyAction,
    EmptyBatch,
    NonFiniteContext { index: usize },
    NonFiniteAction { index: usize },
    NonFinitePhiImprovement,
    NonFiniteEffectiveInformation,
    NonFiniteConfidence,
    ConfidenceOutOfRange,
}

/// Compute a deterministic content identity for one generated wisdom record.
///
/// Canonicalization is deliberately explicit and architecture-independent:
/// vector lengths are encoded as little-endian `u64`, every `f32` contributes
/// its IEEE-754 little-endian bytes, and a versioned domain separator prevents
/// accidental cross-protocol reuse of the digest.
pub fn dream_wisdom_identity(
    wisdom: &Wisdom<Vec<f32>>,
) -> Result<DreamWisdomIdentity, DreamWisdomProvenanceError> {
    validate_wisdom(wisdom)?;

    let mut hasher = blake3::Hasher::new();
    hasher.update(WISDOM_DOMAIN);
    hash_f32_slice(&mut hasher, &wisdom.context_state);
    hash_f32_slice(&mut hasher, &wisdom.better_action);
    hasher.update(&wisdom.phi_improvement.to_le_bytes());
    hasher.update(&wisdom.effective_information.to_le_bytes());
    hasher.update(&wisdom.confidence.to_le_bytes());

    Ok(DreamWisdomIdentity {
        digest: *hasher.finalize().as_bytes(),
    })
}

/// Bind an ordered set of newly generated wisdom records to one cycle identity.
///
/// Ordering is significant because the dream engine's output order is part of the
/// produced artifact. The batch commits the validated per-record identities rather
/// than reimplementing record serialization a second time.
pub fn dream_wisdom_batch_identity(
    wisdom: &[Wisdom<Vec<f32>>],
) -> Result<DreamWisdomBatchIdentity, DreamWisdomProvenanceError> {
    if wisdom.is_empty() {
        return Err(DreamWisdomProvenanceError::EmptyBatch);
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(BATCH_DOMAIN);
    hasher.update(&(wisdom.len() as u64).to_le_bytes());
    for item in wisdom {
        hasher.update(&dream_wisdom_identity(item)?.digest);
    }

    Ok(DreamWisdomBatchIdentity {
        digest: *hasher.finalize().as_bytes(),
        wisdom_count: wisdom.len(),
    })
}

fn validate_wisdom(
    wisdom: &Wisdom<Vec<f32>>,
) -> Result<(), DreamWisdomProvenanceError> {
    if wisdom.context_state.is_empty() {
        return Err(DreamWisdomProvenanceError::EmptyContext);
    }
    if wisdom.better_action.is_empty() {
        return Err(DreamWisdomProvenanceError::EmptyAction);
    }
    for (index, value) in wisdom.context_state.iter().enumerate() {
        if !value.is_finite() {
            return Err(DreamWisdomProvenanceError::NonFiniteContext { index });
        }
    }
    for (index, value) in wisdom.better_action.iter().enumerate() {
        if !value.is_finite() {
            return Err(DreamWisdomProvenanceError::NonFiniteAction { index });
        }
    }
    if !wisdom.phi_improvement.is_finite() {
        return Err(DreamWisdomProvenanceError::NonFinitePhiImprovement);
    }
    if !wisdom.effective_information.is_finite() {
        return Err(DreamWisdomProvenanceError::NonFiniteEffectiveInformation);
    }
    if !wisdom.confidence.is_finite() {
        return Err(DreamWisdomProvenanceError::NonFiniteConfidence);
    }
    if !(0.0..=1.0).contains(&wisdom.confidence) {
        return Err(DreamWisdomProvenanceError::ConfidenceOutOfRange);
    }
    Ok(())
}

fn hash_f32_slice(hasher: &mut blake3::Hasher, values: &[f32]) {
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wisdom() -> Wisdom<Vec<f32>> {
        Wisdom {
            context_state: vec![0.1, -0.2, 0.3],
            better_action: vec![0.4, 0.5],
            phi_improvement: 0.25,
            effective_information: 0.15,
            confidence: 0.8,
        }
    }

    #[test]
    fn identity_is_deterministic_and_hex_is_complete() {
        let item = wisdom();
        let first = dream_wisdom_identity(&item).unwrap();
        let second = dream_wisdom_identity(&item).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.to_hex().len(), 64);
    }

    #[test]
    fn every_semantic_field_is_committed() {
        let baseline = wisdom();
        let baseline_id = dream_wisdom_identity(&baseline).unwrap();

        let mut changed_context = baseline.clone();
        changed_context.context_state[0] += 0.01;
        assert_ne!(baseline_id, dream_wisdom_identity(&changed_context).unwrap());

        let mut changed_action = baseline.clone();
        changed_action.better_action[0] += 0.01;
        assert_ne!(baseline_id, dream_wisdom_identity(&changed_action).unwrap());

        let mut changed_phi = baseline.clone();
        changed_phi.phi_improvement += 0.01;
        assert_ne!(baseline_id, dream_wisdom_identity(&changed_phi).unwrap());

        let mut changed_ei = baseline.clone();
        changed_ei.effective_information += 0.01;
        assert_ne!(baseline_id, dream_wisdom_identity(&changed_ei).unwrap());

        let mut changed_confidence = baseline;
        changed_confidence.confidence -= 0.01;
        assert_ne!(
            baseline_id,
            dream_wisdom_identity(&changed_confidence).unwrap()
        );
    }

    #[test]
    fn batch_identity_commits_order_and_count() {
        let first = wisdom();
        let mut second = wisdom();
        second.better_action[0] = -0.4;

        let forward = dream_wisdom_batch_identity(&[first.clone(), second.clone()]).unwrap();
        let reverse = dream_wisdom_batch_identity(&[second, first]).unwrap();
        assert_ne!(forward, reverse);
        assert_eq!(forward.wisdom_count, 2);
        assert_eq!(forward.to_hex().len(), 64);
    }

    #[test]
    fn malformed_generated_values_fail_closed() {
        let mut item = wisdom();
        item.context_state[1] = f32::NAN;
        assert_eq!(
            dream_wisdom_identity(&item),
            Err(DreamWisdomProvenanceError::NonFiniteContext { index: 1 })
        );

        let mut item = wisdom();
        item.confidence = 1.1;
        assert_eq!(
            dream_wisdom_identity(&item),
            Err(DreamWisdomProvenanceError::ConfidenceOutOfRange)
        );

        assert_eq!(
            dream_wisdom_batch_identity(&[]),
            Err(DreamWisdomProvenanceError::EmptyBatch)
        );
    }
}
