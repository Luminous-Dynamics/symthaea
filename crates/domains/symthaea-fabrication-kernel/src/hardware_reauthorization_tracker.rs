// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent machine-specific anti-rollback state for hardware reauthorization.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::hardware_reauthorization::VerifiedHardwareReauthorization;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const HARDWARE_REAUTHORIZATION_TRACKER_SCHEMA: &str =
    "symthaea.fabrication.hardware-reauthorization-tracker.v1";
pub const MAX_TRACKED_REAUTHORIZED_MACHINES: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareReauthorizationRecord {
    pub handoff_digest: Sha256Digest,
    pub sequence: u64,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub statement_digest: Sha256Digest,
    pub hardware_identity_digest: Sha256Digest,
    pub machine_profile_digest: Sha256Digest,
    pub firmware_digest: Sha256Digest,
    pub calibration_digest: Sha256Digest,
    pub capability_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareReauthorizationTracker {
    pub schema_version: String,
    pub records: BTreeMap<String, HardwareReauthorizationRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HardwareReauthorizationTrackingError {
    UnsupportedSchema,
    CapacityExceeded,
    InvalidMachineId,
    InvalidRecord,
    HandoffSubstitution,
    SequenceRollback { latest: u64, proposed: u64 },
    SequenceCollision { sequence: u64 },
    IssueTimeRegression { latest: u64, proposed: u64 },
    Encoding(String),
}

impl Default for HardwareReauthorizationTracker {
    fn default() -> Self {
        Self {
            schema_version: HARDWARE_REAUTHORIZATION_TRACKER_SCHEMA.into(),
            records: BTreeMap::new(),
        }
    }
}

impl HardwareReauthorizationTracker {
    pub fn validate(&self) -> Result<(), HardwareReauthorizationTrackingError> {
        if self.schema_version != HARDWARE_REAUTHORIZATION_TRACKER_SCHEMA {
            return Err(HardwareReauthorizationTrackingError::UnsupportedSchema);
        }
        if self.records.len() > MAX_TRACKED_REAUTHORIZED_MACHINES {
            return Err(HardwareReauthorizationTrackingError::CapacityExceeded);
        }
        for (machine_id, record) in &self.records {
            validate_machine_id(machine_id)?;
            if record.sequence == 0
                || record.issued_at_unix_s >= record.expires_at_unix_s
                || record.handoff_digest.0 == [0; 32]
                || record.statement_digest.0 == [0; 32]
                || record.hardware_identity_digest.0 == [0; 32]
                || record.machine_profile_digest.0 == [0; 32]
                || record.firmware_digest.0 == [0; 32]
                || record.calibration_digest.0 == [0; 32]
                || record.capability_digest.0 == [0; 32]
            {
                return Err(HardwareReauthorizationTrackingError::InvalidRecord);
            }
        }
        Ok(())
    }

    pub fn accept(
        &mut self,
        authorization: &VerifiedHardwareReauthorization,
    ) -> Result<Sha256Digest, HardwareReauthorizationTrackingError> {
        self.validate()?;
        let statement = authorization.statement();
        validate_machine_id(&statement.machine_id)?;
        let proposed = HardwareReauthorizationRecord {
            handoff_digest: statement.handoff_digest,
            sequence: statement.reauthorization_sequence,
            issued_at_unix_s: statement.issued_at_unix_s,
            expires_at_unix_s: statement.expires_at_unix_s,
            statement_digest: authorization.statement_digest(),
            hardware_identity_digest: statement.hardware_identity_digest,
            machine_profile_digest: statement.machine_profile_digest,
            firmware_digest: statement.firmware_digest,
            calibration_digest: statement.calibration_digest,
            capability_digest: statement.capability_digest,
        };
        if let Some(latest) = self.records.get(&statement.machine_id) {
            if latest.handoff_digest != proposed.handoff_digest {
                return Err(HardwareReauthorizationTrackingError::HandoffSubstitution);
            }
            if proposed.sequence < latest.sequence {
                return Err(HardwareReauthorizationTrackingError::SequenceRollback {
                    latest: latest.sequence,
                    proposed: proposed.sequence,
                });
            }
            if proposed.sequence == latest.sequence {
                if latest == &proposed {
                    return Ok(proposed.statement_digest);
                }
                return Err(HardwareReauthorizationTrackingError::SequenceCollision {
                    sequence: proposed.sequence,
                });
            }
            if proposed.issued_at_unix_s < latest.issued_at_unix_s {
                return Err(HardwareReauthorizationTrackingError::IssueTimeRegression {
                    latest: latest.issued_at_unix_s,
                    proposed: proposed.issued_at_unix_s,
                });
            }
        } else if self.records.len() >= MAX_TRACKED_REAUTHORIZED_MACHINES {
            return Err(HardwareReauthorizationTrackingError::CapacityExceeded);
        }
        self.records
            .insert(statement.machine_id.clone(), proposed.clone());
        Ok(proposed.statement_digest)
    }

    pub fn permits(&self, machine_id: &str, handoff_digest: Sha256Digest, unix_s: u64) -> bool {
        self.records.get(machine_id).is_some_and(|record| {
            record.handoff_digest == handoff_digest
                && unix_s >= record.issued_at_unix_s
                && unix_s < record.expires_at_unix_s
        })
    }
}

pub fn digest_hardware_reauthorization_tracker(
    tracker: &HardwareReauthorizationTracker,
) -> Result<Sha256Digest, HardwareReauthorizationTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker)
        .map_err(|error| HardwareReauthorizationTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.hardware-reauthorization-tracker-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn validate_machine_id(value: &str) -> Result<(), HardwareReauthorizationTrackingError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
    {
        return Err(HardwareReauthorizationTrackingError::InvalidMachineId);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    #[test]
    fn malformed_persisted_record_is_rejected() {
        let mut tracker = HardwareReauthorizationTracker::default();
        tracker.records.insert(
            "machine-a".into(),
            HardwareReauthorizationRecord {
                handoff_digest: sha256(b"handoff"),
                sequence: 0,
                issued_at_unix_s: 1,
                expires_at_unix_s: 2,
                statement_digest: sha256(b"statement"),
                hardware_identity_digest: sha256(b"hardware"),
                machine_profile_digest: sha256(b"profile"),
                firmware_digest: sha256(b"firmware"),
                calibration_digest: sha256(b"calibration"),
                capability_digest: sha256(b"capability"),
            },
        );
        assert_eq!(
            tracker.validate(),
            Err(HardwareReauthorizationTrackingError::InvalidRecord)
        );
    }

    #[test]
    fn empty_tracker_digest_is_stable() {
        let tracker = HardwareReauthorizationTracker::default();
        assert_eq!(
            digest_hardware_reauthorization_tracker(&tracker).unwrap(),
            digest_hardware_reauthorization_tracker(&tracker).unwrap()
        );
    }
}
