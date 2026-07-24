// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-domain monotonic epoch seals.
//!
//! Fabrication authority spans several independently versioned state machines.
//! A single sequence number is insufficient: trust, membership, gateway,
//! resilience, containment, transparency, lineage, and incident histories must
//! all move monotonically. This module seals that vector into one digest and
//! rejects partial rollback or same-vector substitution.

use crate::crypto_digest::{Sha256, Sha256Digest};
use serde::{Deserialize, Serialize};

pub const AUTHORITY_EPOCH_SCHEMA: &str = "symthaea.fabrication.authority-epoch.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityEpochVector {
    pub schema_version: String,
    pub trust_sequence: u64,
    pub membership_epoch: u64,
    pub gateway_generation: u64,
    pub resilience_generation: u64,
    pub containment_generation: u64,
    pub transparency_tree_size: u64,
    pub release_lineage_events: u64,
    pub incident_events: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorityEpochError {
    UnsupportedSchema,
    ZeroTrustSequence,
    ZeroMembershipEpoch,
    ZeroGatewayGeneration,
    Encoding(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorityEpochTrackingError {
    Invalid(AuthorityEpochError),
    ComponentRollback {
        component: &'static str,
        latest: u64,
        proposed: u64,
    },
    NoProgress,
    SameVectorSubstitution,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityEpochTracker {
    latest: Option<AuthorityEpochVector>,
    latest_digest: Option<Sha256Digest>,
}

impl AuthorityEpochVector {
    pub fn new(
        trust_sequence: u64,
        membership_epoch: u64,
        gateway_generation: u64,
        resilience_generation: u64,
        containment_generation: u64,
        transparency_tree_size: u64,
        release_lineage_events: u64,
        incident_events: u64,
    ) -> Result<Self, AuthorityEpochError> {
        let value = Self {
            schema_version: AUTHORITY_EPOCH_SCHEMA.into(),
            trust_sequence,
            membership_epoch,
            gateway_generation,
            resilience_generation,
            containment_generation,
            transparency_tree_size,
            release_lineage_events,
            incident_events,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), AuthorityEpochError> {
        if self.schema_version != AUTHORITY_EPOCH_SCHEMA {
            return Err(AuthorityEpochError::UnsupportedSchema);
        }
        if self.trust_sequence == 0 {
            return Err(AuthorityEpochError::ZeroTrustSequence);
        }
        if self.membership_epoch == 0 {
            return Err(AuthorityEpochError::ZeroMembershipEpoch);
        }
        if self.gateway_generation == 0 {
            return Err(AuthorityEpochError::ZeroGatewayGeneration);
        }
        Ok(())
    }

    pub fn dominates(&self, previous: &Self) -> Result<bool, AuthorityEpochTrackingError> {
        self.validate()
            .map_err(AuthorityEpochTrackingError::Invalid)?;
        previous
            .validate()
            .map_err(AuthorityEpochTrackingError::Invalid)?;
        let pairs = [
            (
                "trust_sequence",
                previous.trust_sequence,
                self.trust_sequence,
            ),
            (
                "membership_epoch",
                previous.membership_epoch,
                self.membership_epoch,
            ),
            (
                "gateway_generation",
                previous.gateway_generation,
                self.gateway_generation,
            ),
            (
                "resilience_generation",
                previous.resilience_generation,
                self.resilience_generation,
            ),
            (
                "containment_generation",
                previous.containment_generation,
                self.containment_generation,
            ),
            (
                "transparency_tree_size",
                previous.transparency_tree_size,
                self.transparency_tree_size,
            ),
            (
                "release_lineage_events",
                previous.release_lineage_events,
                self.release_lineage_events,
            ),
            (
                "incident_events",
                previous.incident_events,
                self.incident_events,
            ),
        ];
        let mut advanced = false;
        for (component, latest, proposed) in pairs {
            if proposed < latest {
                return Err(AuthorityEpochTrackingError::ComponentRollback {
                    component,
                    latest,
                    proposed,
                });
            }
            advanced |= proposed > latest;
        }
        Ok(advanced)
    }
}

impl AuthorityEpochTracker {
    pub fn validate(&self) -> Result<(), AuthorityEpochTrackingError> {
        match (&self.latest, self.latest_digest) {
            (None, None) => Ok(()),
            (Some(latest), Some(digest)) => {
                latest
                    .validate()
                    .map_err(AuthorityEpochTrackingError::Invalid)?;
                let expected =
                    digest_authority_epoch(latest).map_err(AuthorityEpochTrackingError::Invalid)?;
                if expected != digest {
                    return Err(AuthorityEpochTrackingError::SameVectorSubstitution);
                }
                Ok(())
            }
            _ => Err(AuthorityEpochTrackingError::SameVectorSubstitution),
        }
    }

    pub fn accept(
        &mut self,
        proposed: AuthorityEpochVector,
    ) -> Result<Sha256Digest, AuthorityEpochTrackingError> {
        self.validate()?;
        proposed
            .validate()
            .map_err(AuthorityEpochTrackingError::Invalid)?;
        let digest =
            digest_authority_epoch(&proposed).map_err(AuthorityEpochTrackingError::Invalid)?;
        if let Some(latest) = &self.latest {
            match proposed.dominates(latest)? {
                true => {}
                false if self.latest_digest == Some(digest) => return Ok(digest),
                false => return Err(AuthorityEpochTrackingError::NoProgress),
            }
        }
        self.latest = Some(proposed);
        self.latest_digest = Some(digest);
        Ok(digest)
    }

    pub fn latest(&self) -> Option<&AuthorityEpochVector> {
        self.latest.as_ref()
    }
    pub fn latest_digest(&self) -> Option<Sha256Digest> {
        self.latest_digest
    }
}

pub fn digest_authority_epoch_tracker(
    tracker: &AuthorityEpochTracker,
) -> Result<Sha256Digest, AuthorityEpochTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker).map_err(|error| {
        AuthorityEpochTrackingError::Invalid(AuthorityEpochError::Encoding(error.to_string()))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.authority-epoch-tracker.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn digest_authority_epoch(
    vector: &AuthorityEpochVector,
) -> Result<Sha256Digest, AuthorityEpochError> {
    vector.validate()?;
    let bytes = serde_json::to_vec(vector)
        .map_err(|error| AuthorityEpochError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.authority-epoch-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(gateway: u64, containment: u64) -> AuthorityEpochVector {
        AuthorityEpochVector::new(3, 2, gateway, 1, containment, 10, 4, 7).unwrap()
    }

    #[test]
    fn partial_rollback_is_rejected() {
        let mut tracker = AuthorityEpochTracker::default();
        tracker.accept(vector(4, 4)).unwrap();
        let regressed = AuthorityEpochVector::new(3, 2, 5, 1, 3, 10, 4, 7).unwrap();
        assert!(matches!(
            tracker.accept(regressed),
            Err(AuthorityEpochTrackingError::ComponentRollback {
                component: "containment_generation",
                ..
            })
        ));
    }

    #[test]
    fn identical_vector_is_idempotent_but_changed_vector_requires_progress() {
        let mut tracker = AuthorityEpochTracker::default();
        let value = vector(4, 4);
        let first = tracker.accept(value.clone()).unwrap();
        assert_eq!(tracker.accept(value).unwrap(), first);
    }
}
