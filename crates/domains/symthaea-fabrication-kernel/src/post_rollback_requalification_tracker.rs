// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent post-rollback qualification capabilities.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::post_rollback_requalification::AuthorizedPostRollbackRequalification;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const POST_ROLLBACK_REQUALIFICATION_TRACKER_SCHEMA: &str =
    "symthaea.fabrication.post-rollback-requalification-tracker.v1";
pub const MAX_REQUALIFICATION_RECORDS: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostRollbackRequalificationRecord {
    pub requalification_sequence: u64,
    pub rollback_digest: Sha256Digest,
    pub target_promotion_digest: Sha256Digest,
    pub requalification_digest: Sha256Digest,
    pub ceremony_digest: Sha256Digest,
    pub authorized_machine_ids: BTreeSet<String>,
    pub authorized_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostRollbackRequalificationTracker {
    pub schema_version: String,
    records: Vec<PostRollbackRequalificationRecord>,
}

impl Default for PostRollbackRequalificationTracker {
    fn default() -> Self {
        Self {
            schema_version: POST_ROLLBACK_REQUALIFICATION_TRACKER_SCHEMA.into(),
            records: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostRollbackRequalificationTrackingError {
    UnsupportedSchema,
    CapacityExceeded,
    InvalidRecord,
    SequenceRollback,
    SameSequenceSubstitution,
    RollbackRequalifiedDifferently,
    MissingPriorRecord,
    PriorRecordChanged,
    Encoding(String),
}

impl PostRollbackRequalificationTracker {
    pub fn records(&self) -> &[PostRollbackRequalificationRecord] {
        &self.records
    }

    pub fn validate(&self) -> Result<(), PostRollbackRequalificationTrackingError> {
        if self.schema_version != POST_ROLLBACK_REQUALIFICATION_TRACKER_SCHEMA {
            return Err(PostRollbackRequalificationTrackingError::UnsupportedSchema);
        }
        if self.records.len() > MAX_REQUALIFICATION_RECORDS {
            return Err(PostRollbackRequalificationTrackingError::CapacityExceeded);
        }
        let mut previous_sequence = 0;
        let mut rollback_records = BTreeMap::new();
        for record in &self.records {
            if record.requalification_sequence == 0
                || record.requalification_sequence <= previous_sequence
                || record.authorized_at_unix_s >= record.expires_at_unix_s
                || record.authorized_machine_ids.is_empty()
                || record.authorized_machine_ids.iter().any(|machine_id| {
                    machine_id.trim().is_empty()
                        || machine_id != machine_id.trim()
                        || machine_id.len() > 256
                        || machine_id.chars().any(char::is_control)
                })
            {
                return Err(PostRollbackRequalificationTrackingError::InvalidRecord);
            }
            if let Some(previous_digest) =
                rollback_records.insert(record.rollback_digest, record.requalification_digest)
            {
                if previous_digest != record.requalification_digest {
                    return Err(
                        PostRollbackRequalificationTrackingError::RollbackRequalifiedDifferently,
                    );
                }
            }
            previous_sequence = record.requalification_sequence;
        }
        Ok(())
    }

    pub fn apply(
        &mut self,
        authorization: &AuthorizedPostRollbackRequalification,
    ) -> Result<Sha256Digest, PostRollbackRequalificationTrackingError> {
        self.validate()?;
        if self.records.len() >= MAX_REQUALIFICATION_RECORDS {
            return Err(PostRollbackRequalificationTrackingError::CapacityExceeded);
        }
        let evidence = authorization.evidence();
        if let Some(existing) = self
            .records
            .iter()
            .find(|record| record.rollback_digest == evidence.rollback_digest)
        {
            if existing.requalification_digest == authorization.requalification_digest() {
                return Ok(existing.requalification_digest);
            }
            return Err(PostRollbackRequalificationTrackingError::RollbackRequalifiedDifferently);
        }
        if let Some(latest) = self.records.last() {
            if evidence.requalification_sequence < latest.requalification_sequence {
                return Err(PostRollbackRequalificationTrackingError::SequenceRollback);
            }
            if evidence.requalification_sequence == latest.requalification_sequence {
                return Err(PostRollbackRequalificationTrackingError::SameSequenceSubstitution);
            }
        }
        self.records.push(PostRollbackRequalificationRecord {
            requalification_sequence: evidence.requalification_sequence,
            rollback_digest: evidence.rollback_digest,
            target_promotion_digest: evidence.target_promotion_digest,
            requalification_digest: authorization.requalification_digest(),
            ceremony_digest: authorization.ceremony_digest(),
            authorized_machine_ids: evidence.authorized_machine_ids.clone(),
            authorized_at_unix_s: evidence.authorized_at_unix_s,
            expires_at_unix_s: evidence.expires_at_unix_s,
        });
        Ok(authorization.requalification_digest())
    }

    pub fn permits(
        &self,
        target_promotion_digest: Sha256Digest,
        machine_id: &str,
        unix_s: u64,
    ) -> bool {
        self.records.iter().rev().any(|record| {
            record.target_promotion_digest == target_promotion_digest
                && unix_s >= record.authorized_at_unix_s
                && unix_s < record.expires_at_unix_s
                && record.authorized_machine_ids.contains(machine_id)
        })
    }

    pub fn verify_successor_of(
        &self,
        previous: &Self,
    ) -> Result<(), PostRollbackRequalificationTrackingError> {
        self.validate()?;
        previous.validate()?;
        if self.records.len() < previous.records.len() {
            return Err(PostRollbackRequalificationTrackingError::MissingPriorRecord);
        }
        if self.records[..previous.records.len()] != previous.records[..] {
            return Err(PostRollbackRequalificationTrackingError::PriorRecordChanged);
        }
        Ok(())
    }
}

pub fn digest_post_rollback_requalification_tracker(
    tracker: &PostRollbackRequalificationTracker,
) -> Result<Sha256Digest, PostRollbackRequalificationTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker)
        .map_err(|error| PostRollbackRequalificationTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.post-rollback-requalification-tracker-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successor_cannot_remove_qualification() {
        let previous = PostRollbackRequalificationTracker {
            schema_version: POST_ROLLBACK_REQUALIFICATION_TRACKER_SCHEMA.into(),
            records: vec![PostRollbackRequalificationRecord {
                requalification_sequence: 1,
                rollback_digest: Sha256Digest([1; 32]),
                target_promotion_digest: Sha256Digest([2; 32]),
                requalification_digest: Sha256Digest([3; 32]),
                ceremony_digest: Sha256Digest([4; 32]),
                authorized_machine_ids: BTreeSet::from(["machine-a".into()]),
                authorized_at_unix_s: 10,
                expires_at_unix_s: 20,
            }],
        };
        let current = PostRollbackRequalificationTracker::default();
        assert_eq!(
            current.verify_successor_of(&previous),
            Err(PostRollbackRequalificationTrackingError::MissingPriorRecord)
        );
    }
}
