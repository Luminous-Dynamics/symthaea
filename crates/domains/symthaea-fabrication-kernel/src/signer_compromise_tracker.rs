// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent anti-rollback state for emergency signer containment.

use crate::attestation::SignatureAlgorithm;
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::signer_compromise::{AuthorizedSignerCompromise, CompromisedSignerIdentity};
use crate::trust::KeyUsage;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SIGNER_COMPROMISE_TRACKER_SCHEMA: &str =
    "symthaea.fabrication.signer-compromise-tracker.v1";
pub const MAX_COMPROMISED_SIGNERS: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompromiseContainmentRecord {
    pub signer: CompromisedSignerIdentity,
    pub affected_usages: BTreeSet<KeyUsage>,
    pub sequence: u64,
    pub effective_at_unix_s: u64,
    pub notice_digest: Sha256Digest,
    pub ceremony_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignerCompromiseTracker {
    pub schema_version: String,
    records: Vec<CompromiseContainmentRecord>,
}

impl Default for SignerCompromiseTracker {
    fn default() -> Self {
        Self {
            schema_version: SIGNER_COMPROMISE_TRACKER_SCHEMA.into(),
            records: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignerCompromiseTrackingError {
    UnsupportedSchema,
    CapacityExceeded,
    InvalidRecord,
    GlobalSequenceRollback { latest: u64, proposed: u64 },
    SameSequenceSubstitution { sequence: u64 },
    SignerScopeNarrowed(String),
    SignerTimeRegressed(String),
    Encoding(String),
}

impl SignerCompromiseTracker {
    pub fn records(&self) -> &[CompromiseContainmentRecord] {
        &self.records
    }

    pub fn latest_sequence(&self) -> Option<u64> {
        self.records.last().map(|record| record.sequence)
    }

    pub fn validate(&self) -> Result<(), SignerCompromiseTrackingError> {
        if self.schema_version != SIGNER_COMPROMISE_TRACKER_SCHEMA {
            return Err(SignerCompromiseTrackingError::UnsupportedSchema);
        }
        if self.records.len() > MAX_COMPROMISED_SIGNERS {
            return Err(SignerCompromiseTrackingError::CapacityExceeded);
        }
        let mut latest_by_signer: BTreeMap<
            (SignatureAlgorithm, String),
            &CompromiseContainmentRecord,
        > = BTreeMap::new();
        let mut previous_sequence = 0;
        let mut sequence_digests = BTreeMap::new();
        for record in &self.records {
            validate_record(record)?;
            if previous_sequence != 0 && record.sequence <= previous_sequence {
                return Err(SignerCompromiseTrackingError::GlobalSequenceRollback {
                    latest: previous_sequence,
                    proposed: record.sequence,
                });
            }
            sequence_digests.insert(record.sequence, record.notice_digest);
            let signer_identity = (
                record.signer.algorithm.clone(),
                record.signer.key_id.clone(),
            );
            if let Some(previous) = latest_by_signer.get(&signer_identity) {
                if !record
                    .affected_usages
                    .is_superset(&previous.affected_usages)
                {
                    return Err(SignerCompromiseTrackingError::SignerScopeNarrowed(
                        record.signer.key_id.clone(),
                    ));
                }
                if record.effective_at_unix_s < previous.effective_at_unix_s {
                    return Err(SignerCompromiseTrackingError::SignerTimeRegressed(
                        record.signer.key_id.clone(),
                    ));
                }
            }
            latest_by_signer.insert(signer_identity, record);
            previous_sequence = record.sequence;
        }
        Ok(())
    }

    pub fn apply(
        &mut self,
        authorization: &AuthorizedSignerCompromise,
    ) -> Result<Sha256Digest, SignerCompromiseTrackingError> {
        self.validate()?;
        let notice = authorization.notice();
        if self.records.len() >= MAX_COMPROMISED_SIGNERS {
            return Err(SignerCompromiseTrackingError::CapacityExceeded);
        }
        if let Some(latest) = self.latest_sequence() {
            if notice.sequence < latest {
                return Err(SignerCompromiseTrackingError::GlobalSequenceRollback {
                    latest,
                    proposed: notice.sequence,
                });
            }
            if notice.sequence == latest {
                if self
                    .records
                    .last()
                    .is_some_and(|record| record.notice_digest == authorization.notice_digest())
                {
                    return Ok(authorization.notice_digest());
                }
                return Err(SignerCompromiseTrackingError::SameSequenceSubstitution {
                    sequence: notice.sequence,
                });
            }
        }
        if let Some(previous) = self
            .records
            .iter()
            .rev()
            .find(|record| record.signer == notice.signer)
        {
            if !notice
                .affected_usages
                .is_superset(&previous.affected_usages)
            {
                return Err(SignerCompromiseTrackingError::SignerScopeNarrowed(
                    notice.signer.key_id.clone(),
                ));
            }
            if notice.effective_at_unix_s < previous.effective_at_unix_s {
                return Err(SignerCompromiseTrackingError::SignerTimeRegressed(
                    notice.signer.key_id.clone(),
                ));
            }
        }
        self.records.push(CompromiseContainmentRecord {
            signer: notice.signer.clone(),
            affected_usages: notice.affected_usages.clone(),
            sequence: notice.sequence,
            effective_at_unix_s: notice.effective_at_unix_s,
            notice_digest: authorization.notice_digest(),
            ceremony_digest: authorization.ceremony_digest(),
        });
        Ok(authorization.notice_digest())
    }

    pub fn is_compromised_at(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        usage: KeyUsage,
        unix_s: u64,
    ) -> bool {
        self.records.iter().rev().any(|record| {
            &record.signer.algorithm == algorithm
                && record.signer.key_id == key_id
                && record.effective_at_unix_s <= unix_s
                && record.affected_usages.contains(&usage)
        })
    }
}

pub fn digest_signer_compromise_tracker(
    tracker: &SignerCompromiseTracker,
) -> Result<Sha256Digest, SignerCompromiseTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker)
        .map_err(|error| SignerCompromiseTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.signer-compromise-tracker-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn validate_record(
    record: &CompromiseContainmentRecord,
) -> Result<(), SignerCompromiseTrackingError> {
    if record.sequence == 0
        || record.signer.key_id.trim().is_empty()
        || record.signer.key_id != record.signer.key_id.trim()
        || !record.signer.algorithm.is_canonical()
        || record.affected_usages.is_empty()
    {
        return Err(SignerCompromiseTrackingError::InvalidRecord);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_scope_cannot_shrink() {
        let signer = CompromisedSignerIdentity {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "key-a".into(),
        };
        let mut first_scope = BTreeSet::new();
        first_scope.insert(KeyUsage::ReleasePromotion);
        first_scope.insert(KeyUsage::ReleaseRollback);
        let mut second_scope = BTreeSet::new();
        second_scope.insert(KeyUsage::ReleaseRollback);
        let tracker = SignerCompromiseTracker {
            schema_version: SIGNER_COMPROMISE_TRACKER_SCHEMA.into(),
            records: vec![
                CompromiseContainmentRecord {
                    signer: signer.clone(),
                    affected_usages: first_scope,
                    sequence: 1,
                    effective_at_unix_s: 10,
                    notice_digest: Sha256Digest([1; 32]),
                    ceremony_digest: Sha256Digest([2; 32]),
                },
                CompromiseContainmentRecord {
                    signer,
                    affected_usages: second_scope,
                    sequence: 2,
                    effective_at_unix_s: 11,
                    notice_digest: Sha256Digest([3; 32]),
                    ceremony_digest: Sha256Digest([4; 32]),
                },
            ],
        };
        assert!(matches!(
            tracker.validate(),
            Err(SignerCompromiseTrackingError::SignerScopeNarrowed(_))
        ));
    }
}
