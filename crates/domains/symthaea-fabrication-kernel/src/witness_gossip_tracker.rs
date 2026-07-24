// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent transparency-gossip observations and equivocation evidence.

use crate::attestation::SignatureAlgorithm;
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::witness_gossip::{VerifiedWitnessEquivocation, VerifiedWitnessGossip};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const WITNESS_GOSSIP_TRACKER_SCHEMA: &str = "symthaea.fabrication.witness-gossip-tracker.v1";
pub const MAX_GOSSIP_OBSERVATIONS: usize = 16_384;
pub const MAX_EQUIVOCATION_RECORDS: usize = 4_096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessGossipObservationRecord {
    pub signer_algorithm: SignatureAlgorithm,
    pub signer_key_id: String,
    pub witness_organization: String,
    pub checkpoint_log_size: u64,
    pub checkpoint_root_digest: Sha256Digest,
    pub statement_digest: Sha256Digest,
    pub observed_at_unix_s: u64,
    pub trust_snapshot_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessEquivocationRecord {
    pub proof_digest: Sha256Digest,
    pub signer_algorithm: SignatureAlgorithm,
    pub signer_key_id: String,
    pub checkpoint_log_size: u64,
    pub first_root_digest: Sha256Digest,
    pub second_root_digest: Sha256Digest,
    pub proved_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessGossipTracker {
    pub schema_version: String,
    observations: Vec<WitnessGossipObservationRecord>,
    equivocations: Vec<WitnessEquivocationRecord>,
}

impl Default for WitnessGossipTracker {
    fn default() -> Self {
        Self {
            schema_version: WITNESS_GOSSIP_TRACKER_SCHEMA.into(),
            observations: Vec::new(),
            equivocations: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessGossipTrackingError {
    UnsupportedSchema,
    ObservationCapacityExceeded,
    EquivocationCapacityExceeded,
    InvalidObservation,
    InvalidEquivocation,
    SameStatementSubstitution,
    ObservationTimeRegressed,
    EquivocationSubstitution,
    ProofNotBackedByObservations,
    Encoding(String),
}

impl WitnessGossipTracker {
    pub fn observations(&self) -> &[WitnessGossipObservationRecord] {
        &self.observations
    }
    pub fn equivocations(&self) -> &[WitnessEquivocationRecord] {
        &self.equivocations
    }

    pub fn validate(&self) -> Result<(), WitnessGossipTrackingError> {
        if self.schema_version != WITNESS_GOSSIP_TRACKER_SCHEMA {
            return Err(WitnessGossipTrackingError::UnsupportedSchema);
        }
        if self.observations.len() > MAX_GOSSIP_OBSERVATIONS {
            return Err(WitnessGossipTrackingError::ObservationCapacityExceeded);
        }
        if self.equivocations.len() > MAX_EQUIVOCATION_RECORDS {
            return Err(WitnessGossipTrackingError::EquivocationCapacityExceeded);
        }
        let mut statements = BTreeMap::new();
        let mut latest_time = BTreeMap::new();
        for record in &self.observations {
            validate_observation(record)?;
            let statement_identity = (
                record.signer_algorithm.clone(),
                record.signer_key_id.clone(),
                record.statement_digest,
            );
            if let Some(previous_root) =
                statements.insert(statement_identity, record.checkpoint_root_digest)
            {
                if previous_root != record.checkpoint_root_digest {
                    return Err(WitnessGossipTrackingError::SameStatementSubstitution);
                }
            }
            let witness_identity = (
                record.signer_algorithm.clone(),
                record.signer_key_id.clone(),
            );
            if let Some(previous) = latest_time.insert(witness_identity, record.observed_at_unix_s)
            {
                if record.observed_at_unix_s < previous {
                    return Err(WitnessGossipTrackingError::ObservationTimeRegressed);
                }
            }
        }
        let mut proof_digests = BTreeMap::new();
        for record in &self.equivocations {
            validate_equivocation(record)?;
            if let Some(previous) = proof_digests.insert(
                record.proof_digest,
                (record.first_root_digest, record.second_root_digest),
            ) {
                if previous != (record.first_root_digest, record.second_root_digest) {
                    return Err(WitnessGossipTrackingError::EquivocationSubstitution);
                }
            }
            let roots_present = self.observations.iter().any(|observation| {
                observation.signer_algorithm == record.signer_algorithm
                    && observation.signer_key_id == record.signer_key_id
                    && observation.checkpoint_log_size == record.checkpoint_log_size
                    && observation.checkpoint_root_digest == record.first_root_digest
            }) && self.observations.iter().any(|observation| {
                observation.signer_algorithm == record.signer_algorithm
                    && observation.signer_key_id == record.signer_key_id
                    && observation.checkpoint_log_size == record.checkpoint_log_size
                    && observation.checkpoint_root_digest == record.second_root_digest
            });
            if !roots_present {
                return Err(WitnessGossipTrackingError::ProofNotBackedByObservations);
            }
        }
        Ok(())
    }

    pub fn observe(
        &mut self,
        gossip: &VerifiedWitnessGossip,
    ) -> Result<Sha256Digest, WitnessGossipTrackingError> {
        self.validate()?;
        if self.observations.len() >= MAX_GOSSIP_OBSERVATIONS {
            return Err(WitnessGossipTrackingError::ObservationCapacityExceeded);
        }
        let record = WitnessGossipObservationRecord {
            signer_algorithm: gossip.signer_algorithm().clone(),
            signer_key_id: gossip.signer_key_id().to_string(),
            witness_organization: gossip.statement().witness_organization.clone(),
            checkpoint_log_size: gossip.statement().checkpoint_log_size,
            checkpoint_root_digest: gossip.statement().checkpoint_root_digest,
            statement_digest: gossip.statement_digest(),
            observed_at_unix_s: gossip.statement().observed_at_unix_s,
            trust_snapshot_digest: gossip.trust_snapshot_digest(),
        };
        if self.observations.iter().any(|existing| existing == &record) {
            return Ok(record.statement_digest);
        }
        if self.observations.iter().any(|existing| {
            existing.signer_algorithm == record.signer_algorithm
                && existing.signer_key_id == record.signer_key_id
                && existing.statement_digest == record.statement_digest
                && existing.checkpoint_root_digest != record.checkpoint_root_digest
        }) {
            return Err(WitnessGossipTrackingError::SameStatementSubstitution);
        }
        if let Some(latest) = self.observations.iter().rev().find(|existing| {
            existing.signer_algorithm == record.signer_algorithm
                && existing.signer_key_id == record.signer_key_id
        }) {
            if record.observed_at_unix_s < latest.observed_at_unix_s {
                return Err(WitnessGossipTrackingError::ObservationTimeRegressed);
            }
        }
        self.observations.push(record);
        Ok(gossip.statement_digest())
    }

    pub fn record_equivocation(
        &mut self,
        proof: &VerifiedWitnessEquivocation,
    ) -> Result<Sha256Digest, WitnessGossipTrackingError> {
        self.validate()?;
        if self.equivocations.len() >= MAX_EQUIVOCATION_RECORDS {
            return Err(WitnessGossipTrackingError::EquivocationCapacityExceeded);
        }
        let body = proof.proof();
        let record = WitnessEquivocationRecord {
            proof_digest: proof.proof_digest(),
            signer_algorithm: body.signer_algorithm.clone(),
            signer_key_id: body.signer_key_id.clone(),
            checkpoint_log_size: body.checkpoint_log_size,
            first_root_digest: body.first_root_digest,
            second_root_digest: body.second_root_digest,
            proved_at_unix_s: body.proved_at_unix_s,
        };
        if self
            .equivocations
            .iter()
            .any(|existing| existing == &record)
        {
            return Ok(record.proof_digest);
        }
        let roots_present = self.observations.iter().any(|observation| {
            observation.signer_algorithm == record.signer_algorithm
                && observation.signer_key_id == record.signer_key_id
                && observation.checkpoint_log_size == record.checkpoint_log_size
                && observation.checkpoint_root_digest == record.first_root_digest
        }) && self.observations.iter().any(|observation| {
            observation.signer_algorithm == record.signer_algorithm
                && observation.signer_key_id == record.signer_key_id
                && observation.checkpoint_log_size == record.checkpoint_log_size
                && observation.checkpoint_root_digest == record.second_root_digest
        });
        if !roots_present {
            return Err(WitnessGossipTrackingError::ProofNotBackedByObservations);
        }
        self.equivocations.push(record);
        Ok(proof.proof_digest())
    }
}

pub fn digest_witness_gossip_tracker(
    tracker: &WitnessGossipTracker,
) -> Result<Sha256Digest, WitnessGossipTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker)
        .map_err(|error| WitnessGossipTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.witness-gossip-tracker-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn validate_observation(
    record: &WitnessGossipObservationRecord,
) -> Result<(), WitnessGossipTrackingError> {
    if !record.signer_algorithm.is_canonical()
        || record.signer_key_id.trim().is_empty()
        || record.witness_organization.trim().is_empty()
        || record.checkpoint_log_size == 0
    {
        return Err(WitnessGossipTrackingError::InvalidObservation);
    }
    Ok(())
}

fn validate_equivocation(
    record: &WitnessEquivocationRecord,
) -> Result<(), WitnessGossipTrackingError> {
    if !record.signer_algorithm.is_canonical()
        || record.signer_key_id.trim().is_empty()
        || record.checkpoint_log_size == 0
        || record.first_root_digest == record.second_root_digest
    {
        return Err(WitnessGossipTrackingError::InvalidEquivocation);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbacked_equivocation_is_rejected() {
        let tracker = WitnessGossipTracker {
            schema_version: WITNESS_GOSSIP_TRACKER_SCHEMA.into(),
            observations: Vec::new(),
            equivocations: vec![WitnessEquivocationRecord {
                proof_digest: Sha256Digest([1; 32]),
                signer_algorithm: SignatureAlgorithm::Ed25519,
                signer_key_id: "witness-a".into(),
                checkpoint_log_size: 4,
                first_root_digest: Sha256Digest([2; 32]),
                second_root_digest: Sha256Digest([3; 32]),
                proved_at_unix_s: 10,
            }],
        };
        assert_eq!(
            tracker.validate(),
            Err(WitnessGossipTrackingError::ProofNotBackedByObservations)
        );
    }
}
