// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only policy migration lineage and waiver-expiry tracking.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::policy_migration::{
    AuthorizedPolicyMigration, PolicyInvariantDisposition, PolicyMigrationError,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const POLICY_MIGRATION_RECORD_SCHEMA: &str = "symthaea.fabrication.policy-migration-record.v1";
pub const MAX_POLICY_MIGRATION_RECORDS: usize = 16_384;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyMigrationRecord {
    pub schema_version: String,
    pub sequence: u64,
    pub domain: String,
    pub predecessor_policy_digest: Sha256Digest,
    pub successor_policy_digest: Sha256Digest,
    pub migration_digest: Sha256Digest,
    pub activates_at_unix_s: u64,
    pub rollback_deadline_unix_s: u64,
    pub active_waiver_expiries: Vec<(String, u64)>,
    pub previous_record_digest: Sha256Digest,
    pub record_digest: Sha256Digest,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyMigrationTracker {
    pub records: Vec<PolicyMigrationRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyMigrationTrackingError {
    CapacityExceeded,
    UnsupportedSchema,
    InvalidSequence {
        expected: u64,
        actual: u64,
    },
    DuplicateMigration,
    DomainFork(String),
    ActivationRegression(String),
    PreviousDigestMismatch,
    RecordDigestMismatch,
    ExpiredWaiver {
        domain: String,
        invariant: String,
        expired_at: u64,
    },
    InvalidMigration(PolicyMigrationError),
    Encoding(String),
}

impl PolicyMigrationTracker {
    pub fn validate_at(&self, now_unix_s: u64) -> Result<(), PolicyMigrationTrackingError> {
        if self.records.len() > MAX_POLICY_MIGRATION_RECORDS {
            return Err(PolicyMigrationTrackingError::CapacityExceeded);
        }
        let mut previous_record_digest = empty_policy_migration_record_digest();
        let mut domains = BTreeMap::<String, (Sha256Digest, u64)>::new();
        let mut migrations = BTreeSet::new();
        for (index, record) in self.records.iter().enumerate() {
            if record.schema_version != POLICY_MIGRATION_RECORD_SCHEMA {
                return Err(PolicyMigrationTrackingError::UnsupportedSchema);
            }
            let expected = index as u64 + 1;
            if record.sequence != expected {
                return Err(PolicyMigrationTrackingError::InvalidSequence {
                    expected,
                    actual: record.sequence,
                });
            }
            if !migrations.insert(record.migration_digest) {
                return Err(PolicyMigrationTrackingError::DuplicateMigration);
            }
            if record.previous_record_digest != previous_record_digest {
                return Err(PolicyMigrationTrackingError::PreviousDigestMismatch);
            }
            if let Some((latest_policy, latest_activation)) = domains.get(&record.domain) {
                if record.predecessor_policy_digest != *latest_policy {
                    return Err(PolicyMigrationTrackingError::DomainFork(
                        record.domain.clone(),
                    ));
                }
                if record.activates_at_unix_s < *latest_activation {
                    return Err(PolicyMigrationTrackingError::ActivationRegression(
                        record.domain.clone(),
                    ));
                }
            }
            let mut waiver_names = BTreeSet::new();
            for (invariant, expiry) in &record.active_waiver_expiries {
                if !waiver_names.insert(invariant.clone()) {
                    return Err(PolicyMigrationTrackingError::InvalidMigration(
                        PolicyMigrationError::DuplicateInvariant(invariant.clone()),
                    ));
                }
                if now_unix_s >= *expiry {
                    return Err(PolicyMigrationTrackingError::ExpiredWaiver {
                        domain: record.domain.clone(),
                        invariant: invariant.clone(),
                        expired_at: *expiry,
                    });
                }
            }
            let expected_digest = digest_policy_migration_record_fields(record)?;
            if expected_digest != record.record_digest {
                return Err(PolicyMigrationTrackingError::RecordDigestMismatch);
            }
            domains.insert(
                record.domain.clone(),
                (record.successor_policy_digest, record.activates_at_unix_s),
            );
            previous_record_digest = record.record_digest;
        }
        Ok(())
    }

    pub fn accept(
        &mut self,
        migration: &AuthorizedPolicyMigration,
        now_unix_s: u64,
    ) -> Result<Sha256Digest, PolicyMigrationTrackingError> {
        self.validate_at(now_unix_s)?;
        if self.records.len() >= MAX_POLICY_MIGRATION_RECORDS {
            return Err(PolicyMigrationTrackingError::CapacityExceeded);
        }
        if self
            .records
            .iter()
            .any(|record| record.migration_digest == migration.plan_digest)
        {
            return self
                .records
                .iter()
                .find(|record| record.migration_digest == migration.plan_digest)
                .map(|record| record.record_digest)
                .ok_or(PolicyMigrationTrackingError::DuplicateMigration);
        }
        if let Some(latest) = self
            .records
            .iter()
            .rev()
            .find(|record| record.domain == migration.plan.predecessor.domain)
        {
            if latest.successor_policy_digest != migration.plan.predecessor.policy_digest {
                return Err(PolicyMigrationTrackingError::DomainFork(
                    latest.domain.clone(),
                ));
            }
            if migration.plan.activates_at_unix_s < latest.activates_at_unix_s {
                return Err(PolicyMigrationTrackingError::ActivationRegression(
                    latest.domain.clone(),
                ));
            }
        }
        let mut active_waiver_expiries = migration
            .plan
            .migrations
            .iter()
            .filter_map(|item| match &item.disposition {
                PolicyInvariantDisposition::Waived {
                    expires_at_unix_s, ..
                } => Some((item.name.clone(), *expires_at_unix_s)),
                _ => None,
            })
            .collect::<Vec<_>>();
        active_waiver_expiries.sort_by(|left, right| left.0.cmp(&right.0));
        let previous_record_digest = self
            .records
            .last()
            .map_or_else(empty_policy_migration_record_digest, |record| {
                record.record_digest
            });
        let mut record = PolicyMigrationRecord {
            schema_version: POLICY_MIGRATION_RECORD_SCHEMA.into(),
            sequence: self.records.len() as u64 + 1,
            domain: migration.plan.predecessor.domain.clone(),
            predecessor_policy_digest: migration.plan.predecessor.policy_digest,
            successor_policy_digest: migration.plan.successor.policy_digest,
            migration_digest: migration.plan_digest,
            activates_at_unix_s: migration.plan.activates_at_unix_s,
            rollback_deadline_unix_s: migration.plan.rollback_deadline_unix_s,
            active_waiver_expiries,
            previous_record_digest,
            record_digest: Sha256Digest([0; 32]),
        };
        record.record_digest = digest_policy_migration_record_fields(&record)?;
        let digest = record.record_digest;
        self.records.push(record);
        Ok(digest)
    }

    pub fn head(&self) -> Sha256Digest {
        self.records
            .last()
            .map_or_else(empty_policy_migration_record_digest, |record| {
                record.record_digest
            })
    }
}

pub fn digest_policy_migration_tracker(
    tracker: &PolicyMigrationTracker,
    now_unix_s: u64,
) -> Result<Sha256Digest, PolicyMigrationTrackingError> {
    tracker.validate_at(now_unix_s)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.policy-migration-tracker.v1\0");
    hasher.update(&(tracker.records.len() as u64).to_le_bytes());
    hasher.update(&tracker.head().0);
    Ok(hasher.finalize())
}

pub fn empty_policy_migration_record_digest() -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.policy-migration-record-empty.v1\0");
    hasher.finalize()
}

fn digest_policy_migration_record_fields(
    record: &PolicyMigrationRecord,
) -> Result<Sha256Digest, PolicyMigrationTrackingError> {
    let bytes = serde_json::to_vec(&(
        &record.schema_version,
        record.sequence,
        &record.domain,
        record.predecessor_policy_digest,
        record.successor_policy_digest,
        record.migration_digest,
        record.activates_at_unix_s,
        record.rollback_deadline_unix_s,
        &record.active_waiver_expiries,
        record.previous_record_digest,
    ))
    .map_err(|error| PolicyMigrationTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.policy-migration-record-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;
    use crate::policy_migration::{
        POLICY_MIGRATION_SCHEMA, PolicyBinding, PolicyInvariantBinding, PolicyInvariantMigration,
        PolicyMigrationPlan,
    };

    fn authorized(
        domain: &str,
        from: &[u8],
        to: &[u8],
        activation: u64,
    ) -> AuthorizedPolicyMigration {
        let invariant = |digest: Sha256Digest| {
            vec![PolicyInvariantBinding {
                name: "authority".into(),
                digest,
            }]
        };
        AuthorizedPolicyMigration {
            plan: PolicyMigrationPlan {
                schema_version: POLICY_MIGRATION_SCHEMA.into(),
                predecessor: PolicyBinding::new(domain, "1", sha256(from), invariant(sha256(from)))
                    .unwrap(),
                successor: PolicyBinding::new(domain, "2", sha256(to), invariant(sha256(to)))
                    .unwrap(),
                activates_at_unix_s: activation,
                rollback_deadline_unix_s: activation + 100,
                rationale: "strengthen".into(),
                migrations: vec![PolicyInvariantMigration {
                    name: "authority".into(),
                    predecessor_digest: sha256(from),
                    successor_digest: Some(sha256(to)),
                    disposition: PolicyInvariantDisposition::Strengthened,
                }],
            },
            plan_digest: sha256(&[from, to].concat()),
            ceremony_digest: sha256(b"ceremony"),
            trust_snapshot_digest: sha256(b"trust"),
        }
    }

    #[test]
    fn policy_history_cannot_fork() {
        let mut tracker = PolicyMigrationTracker::default();
        tracker
            .accept(&authorized("machine", b"a", b"b", 100), 50)
            .unwrap();
        let fork = authorized("machine", b"x", b"c", 200);
        assert_eq!(
            tracker.accept(&fork, 150),
            Err(PolicyMigrationTrackingError::DomainFork("machine".into()))
        );
    }
}
