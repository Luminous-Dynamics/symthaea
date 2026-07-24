// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent anti-equivocation tracking for transparency witnesses.

use crate::attestation::SignatureAlgorithm;
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::transparency_witness::{SignedTransparencyWitness, VerifiedTransparencyWitnessQuorum};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessObservationState {
    pub checkpoint_log_size: u64,
    pub checkpoint_root_digest: Sha256Digest,
    pub checkpoint_digest: Sha256Digest,
    pub statement_digest: Sha256Digest,
    pub observed_at_unix_s: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransparencyWitnessTracker {
    latest_by_signer: BTreeMap<(SignatureAlgorithm, String), WitnessObservationState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyWitnessTrackingError {
    InvalidSigner,
    LogSizeRollback { latest: u64, proposed: u64 },
    SameSizeEquivocation,
    CheckpointDigestSubstitution,
    ObservationTimeRegressed { latest: u64, proposed: u64 },
    WitnessNotInVerifiedQuorum,
    CheckpointMismatch,
    InvalidState,
    Encoding(String),
}

impl TransparencyWitnessTracker {
    pub fn accept_verified(
        &mut self,
        quorum: &VerifiedTransparencyWitnessQuorum,
        witness: &SignedTransparencyWitness,
    ) -> Result<(), TransparencyWitnessTrackingError> {
        if witness.statement.checkpoint_digest != quorum.checkpoint_digest() {
            return Err(TransparencyWitnessTrackingError::CheckpointMismatch);
        }
        let identity = (
            witness.signature.algorithm.clone(),
            witness.signature.key_id.clone(),
            witness.statement.witness_organization.clone(),
            witness.statement.witness_region.clone(),
        );
        if !quorum.witnesses().contains(&identity) {
            return Err(TransparencyWitnessTrackingError::WitnessNotInVerifiedQuorum);
        }
        self.accept_observation(witness)
    }

    fn accept_observation(
        &mut self,
        witness: &SignedTransparencyWitness,
    ) -> Result<(), TransparencyWitnessTrackingError> {
        let key_id = witness.signature.key_id.as_str();
        if !witness.signature.algorithm.is_canonical()
            || key_id.trim().is_empty()
            || key_id != key_id.trim()
            || key_id.len() > 256
        {
            return Err(TransparencyWitnessTrackingError::InvalidSigner);
        }
        let identity = (
            witness.signature.algorithm.clone(),
            witness.signature.key_id.clone(),
        );
        let proposed = WitnessObservationState {
            checkpoint_log_size: witness.statement.checkpoint_log_size,
            checkpoint_root_digest: witness.statement.checkpoint_root_digest,
            checkpoint_digest: witness.statement.checkpoint_digest,
            statement_digest: witness.statement_digest,
            observed_at_unix_s: witness.statement.observed_at_unix_s,
        };
        if let Some(latest) = self.latest_by_signer.get(&identity) {
            if proposed.checkpoint_log_size < latest.checkpoint_log_size {
                return Err(TransparencyWitnessTrackingError::LogSizeRollback {
                    latest: latest.checkpoint_log_size,
                    proposed: proposed.checkpoint_log_size,
                });
            }
            if proposed.observed_at_unix_s < latest.observed_at_unix_s {
                return Err(TransparencyWitnessTrackingError::ObservationTimeRegressed {
                    latest: latest.observed_at_unix_s,
                    proposed: proposed.observed_at_unix_s,
                });
            }
            if proposed.checkpoint_log_size == latest.checkpoint_log_size {
                if proposed == *latest {
                    return Ok(());
                }
                if proposed.checkpoint_root_digest != latest.checkpoint_root_digest {
                    return Err(TransparencyWitnessTrackingError::SameSizeEquivocation);
                }
                return Err(TransparencyWitnessTrackingError::CheckpointDigestSubstitution);
            }
        }
        self.latest_by_signer.insert(identity, proposed);
        Ok(())
    }

    pub fn latest(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
    ) -> Option<&WitnessObservationState> {
        self.latest_by_signer
            .get(&(algorithm.clone(), key_id.to_string()))
    }

    pub fn witness_count(&self) -> usize {
        self.latest_by_signer.len()
    }

    pub fn validate(&self) -> Result<(), TransparencyWitnessTrackingError> {
        for ((algorithm, key_id), state) in &self.latest_by_signer {
            if !algorithm.is_canonical()
                || key_id.trim().is_empty()
                || key_id != key_id.trim()
                || key_id.len() > 256
                || state.checkpoint_log_size == 0
                || state.observed_at_unix_s == 0
                || state.checkpoint_root_digest == Sha256Digest([0; 32])
                || state.checkpoint_digest == Sha256Digest([0; 32])
                || state.statement_digest == Sha256Digest([0; 32])
            {
                return Err(TransparencyWitnessTrackingError::InvalidState);
            }
        }
        Ok(())
    }

    pub fn verify_successor_of(
        &self,
        previous: &Self,
    ) -> Result<(), TransparencyWitnessTrackingError> {
        self.validate()?;
        previous.validate()?;
        for (identity, old) in &previous.latest_by_signer {
            let Some(new) = self.latest_by_signer.get(identity) else {
                return Err(TransparencyWitnessTrackingError::LogSizeRollback {
                    latest: old.checkpoint_log_size,
                    proposed: 0,
                });
            };
            if new.checkpoint_log_size < old.checkpoint_log_size {
                return Err(TransparencyWitnessTrackingError::LogSizeRollback {
                    latest: old.checkpoint_log_size,
                    proposed: new.checkpoint_log_size,
                });
            }
            if new.observed_at_unix_s < old.observed_at_unix_s {
                return Err(TransparencyWitnessTrackingError::ObservationTimeRegressed {
                    latest: old.observed_at_unix_s,
                    proposed: new.observed_at_unix_s,
                });
            }
            if new.checkpoint_log_size == old.checkpoint_log_size && new != old {
                return Err(TransparencyWitnessTrackingError::SameSizeEquivocation);
            }
        }
        Ok(())
    }
}

pub fn digest_transparency_witness_tracker(
    tracker: &TransparencyWitnessTracker,
) -> Result<Sha256Digest, TransparencyWitnessTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker)
        .map_err(|error| TransparencyWitnessTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.transparency-witness-tracker-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::DetachedSignature;
    use crate::transparency_witness::{
        SIGNED_TRANSPARENCY_WITNESS_SCHEMA, TRANSPARENCY_WITNESS_SCHEMA,
        TransparencyWitnessStatement,
    };

    fn witness(size: u64, root: u8, observed: u64) -> SignedTransparencyWitness {
        SignedTransparencyWitness {
            schema_version: SIGNED_TRANSPARENCY_WITNESS_SCHEMA.into(),
            statement: TransparencyWitnessStatement {
                schema_version: TRANSPARENCY_WITNESS_SCHEMA.into(),
                checkpoint_digest: Sha256Digest([root; 32]),
                checkpoint_log_size: size,
                checkpoint_root_digest: Sha256Digest([root; 32]),
                witness_organization: "org-a".into(),
                witness_region: "region-a".into(),
                observed_at_unix_s: observed,
            },
            statement_digest: Sha256Digest([root; 32]),
            signature: DetachedSignature {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "witness-a".into(),
                signature: vec![1],
            },
        }
    }

    #[test]
    fn same_size_different_root_is_equivocation() {
        let mut tracker = TransparencyWitnessTracker::default();
        tracker.accept_observation(&witness(10, 1, 100)).unwrap();
        assert_eq!(
            tracker.accept_observation(&witness(10, 2, 101)),
            Err(TransparencyWitnessTrackingError::SameSizeEquivocation)
        );
    }
}
