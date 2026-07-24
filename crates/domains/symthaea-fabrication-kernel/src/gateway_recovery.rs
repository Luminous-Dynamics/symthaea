// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verified disaster-recovery bundles for durable gateway state.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_consensus::VerifiedGatewayConsensus;
use crate::gateway_state::{
    FabricationGatewayState, GatewayStateEnvelope, GatewayStateError,
    verify_gateway_state_successor,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const GATEWAY_RECOVERY_BUNDLE_SCHEMA: &str = "symthaea.fabrication.gateway-recovery-bundle.v1";
pub const MAX_RECOVERY_CHECKPOINTS: usize = 4096;
pub const MAX_RECOVERY_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayConsensusEvidence {
    pub state_digest: Sha256Digest,
    pub generation: u64,
    pub consensus_digest: Sha256Digest,
    pub trust_snapshot_digest: Sha256Digest,
    pub gateways: Vec<String>,
}

impl GatewayConsensusEvidence {
    pub fn from_verified(consensus: &VerifiedGatewayConsensus) -> Self {
        Self {
            state_digest: consensus.state_digest(),
            generation: consensus.generation(),
            consensus_digest: consensus.consensus_digest(),
            trust_snapshot_digest: consensus.trust_snapshot_digest(),
            gateways: consensus.gateways().to_vec(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayRecoveryCheckpoint {
    pub backup_id: String,
    pub captured_at_unix_ms: u64,
    pub envelope: GatewayStateEnvelope,
    pub consensus: GatewayConsensusEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayRecoveryBundle {
    pub schema_version: String,
    pub recovery_set_id: String,
    pub exported_at_unix_ms: u64,
    pub checkpoints: Vec<GatewayRecoveryCheckpoint>,
    pub bundle_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayRecoveryError {
    UnsupportedSchema,
    EmptyRecoverySetId,
    NonCanonicalRecoverySetId,
    RecoverySetIdTooLong,
    EmptyBundle,
    TooManyCheckpoints {
        actual: usize,
        maximum: usize,
    },
    EmptyBackupId {
        index: usize,
    },
    NonCanonicalBackupId {
        index: usize,
    },
    DuplicateBackupId(String),
    CaptureBeforeCommit {
        index: usize,
    },
    CaptureTimeRegressed {
        index: usize,
    },
    Envelope {
        index: usize,
        error: GatewayStateError,
    },
    ConsensusStateMismatch {
        index: usize,
    },
    ConsensusGenerationMismatch {
        index: usize,
    },
    ConsensusTrustMismatch {
        index: usize,
    },
    EmptyConsensus {
        index: usize,
    },
    DuplicateConsensusGateway {
        index: usize,
        gateway_id: String,
    },
    NonContiguousGeneration {
        index: usize,
        previous: u64,
        current: u64,
    },
    Successor {
        index: usize,
        error: GatewayStateError,
    },
    ExportBeforeLatestCapture,
    DigestMismatch,
    Encoding(String),
}

impl GatewayRecoveryBundle {
    pub fn build(
        recovery_set_id: impl Into<String>,
        exported_at_unix_ms: u64,
        checkpoints: Vec<GatewayRecoveryCheckpoint>,
    ) -> Result<Self, GatewayRecoveryError> {
        let mut bundle = Self {
            schema_version: GATEWAY_RECOVERY_BUNDLE_SCHEMA.into(),
            recovery_set_id: recovery_set_id.into(),
            exported_at_unix_ms,
            checkpoints,
            bundle_digest: Sha256Digest([0; 32]),
        };
        bundle.validate_without_digest()?;
        bundle.bundle_digest = digest_recovery_bundle_body(&bundle)?;
        Ok(bundle)
    }

    pub fn validate(&self) -> Result<(), GatewayRecoveryError> {
        self.validate_without_digest()?;
        if digest_recovery_bundle_body(self)? != self.bundle_digest {
            return Err(GatewayRecoveryError::DigestMismatch);
        }
        Ok(())
    }

    pub fn latest_state(&self) -> Result<FabricationGatewayState, GatewayRecoveryError> {
        self.validate()?;
        self.checkpoints
            .last()
            .ok_or(GatewayRecoveryError::EmptyBundle)?
            .envelope
            .clone()
            .open()
            .map_err(|error| GatewayRecoveryError::Envelope {
                index: self.checkpoints.len().saturating_sub(1),
                error,
            })
    }

    fn validate_without_digest(&self) -> Result<(), GatewayRecoveryError> {
        if self.schema_version != GATEWAY_RECOVERY_BUNDLE_SCHEMA {
            return Err(GatewayRecoveryError::UnsupportedSchema);
        }
        validate_id(&self.recovery_set_id).map_err(|kind| match kind {
            IdError::Empty => GatewayRecoveryError::EmptyRecoverySetId,
            IdError::NonCanonical => GatewayRecoveryError::NonCanonicalRecoverySetId,
            IdError::TooLong => GatewayRecoveryError::RecoverySetIdTooLong,
        })?;
        if self.checkpoints.is_empty() {
            return Err(GatewayRecoveryError::EmptyBundle);
        }
        if self.checkpoints.len() > MAX_RECOVERY_CHECKPOINTS {
            return Err(GatewayRecoveryError::TooManyCheckpoints {
                actual: self.checkpoints.len(),
                maximum: MAX_RECOVERY_CHECKPOINTS,
            });
        }
        let mut backup_ids = BTreeSet::new();
        let mut previous_state: Option<FabricationGatewayState> = None;
        let mut previous_capture = 0;
        for (index, checkpoint) in self.checkpoints.iter().enumerate() {
            validate_id(&checkpoint.backup_id).map_err(|kind| match kind {
                IdError::Empty => GatewayRecoveryError::EmptyBackupId { index },
                IdError::NonCanonical | IdError::TooLong => {
                    GatewayRecoveryError::NonCanonicalBackupId { index }
                }
            })?;
            if !backup_ids.insert(checkpoint.backup_id.clone()) {
                return Err(GatewayRecoveryError::DuplicateBackupId(
                    checkpoint.backup_id.clone(),
                ));
            }
            let state = checkpoint
                .envelope
                .clone()
                .open()
                .map_err(|error| GatewayRecoveryError::Envelope { index, error })?;
            if checkpoint.captured_at_unix_ms < state.committed_at_unix_ms {
                return Err(GatewayRecoveryError::CaptureBeforeCommit { index });
            }
            if index > 0 && checkpoint.captured_at_unix_ms < previous_capture {
                return Err(GatewayRecoveryError::CaptureTimeRegressed { index });
            }
            if checkpoint.consensus.state_digest != checkpoint.envelope.state_digest {
                return Err(GatewayRecoveryError::ConsensusStateMismatch { index });
            }
            if checkpoint.consensus.generation != state.generation {
                return Err(GatewayRecoveryError::ConsensusGenerationMismatch { index });
            }
            if checkpoint.consensus.trust_snapshot_digest
                != crate::trust::digest_trust_snapshot(&state.trust_snapshot).map_err(|error| {
                    GatewayRecoveryError::Envelope {
                        index,
                        error: GatewayStateError::TrustSnapshot(error),
                    }
                })?
            {
                return Err(GatewayRecoveryError::ConsensusTrustMismatch { index });
            }
            if checkpoint.consensus.gateways.is_empty() {
                return Err(GatewayRecoveryError::EmptyConsensus { index });
            }
            let mut gateway_ids = BTreeSet::new();
            for gateway_id in &checkpoint.consensus.gateways {
                if validate_id(gateway_id).is_err() || !gateway_ids.insert(gateway_id.clone()) {
                    return Err(GatewayRecoveryError::DuplicateConsensusGateway {
                        index,
                        gateway_id: gateway_id.clone(),
                    });
                }
            }
            if let Some(previous) = previous_state.as_ref() {
                if state.generation != previous.generation.saturating_add(1) {
                    return Err(GatewayRecoveryError::NonContiguousGeneration {
                        index,
                        previous: previous.generation,
                        current: state.generation,
                    });
                }
                verify_gateway_state_successor(previous, &state)
                    .map_err(|error| GatewayRecoveryError::Successor { index, error })?;
            }
            previous_capture = checkpoint.captured_at_unix_ms;
            previous_state = Some(state);
        }
        if self
            .checkpoints
            .last()
            .is_some_and(|checkpoint| self.exported_at_unix_ms < checkpoint.captured_at_unix_ms)
        {
            return Err(GatewayRecoveryError::ExportBeforeLatestCapture);
        }
        Ok(())
    }
}

pub fn digest_recovery_bundle(
    bundle: &GatewayRecoveryBundle,
) -> Result<Sha256Digest, GatewayRecoveryError> {
    bundle.validate()?;
    Ok(bundle.bundle_digest)
}

fn digest_recovery_bundle_body(
    bundle: &GatewayRecoveryBundle,
) -> Result<Sha256Digest, GatewayRecoveryError> {
    #[derive(Serialize)]
    struct Body<'a> {
        schema_version: &'a str,
        recovery_set_id: &'a str,
        exported_at_unix_ms: u64,
        checkpoints: &'a [GatewayRecoveryCheckpoint],
    }
    let bytes = serde_json::to_vec(&Body {
        schema_version: &bundle.schema_version,
        recovery_set_id: &bundle.recovery_set_id,
        exported_at_unix_ms: bundle.exported_at_unix_ms,
        checkpoints: &bundle.checkpoints,
    })
    .map_err(|error| GatewayRecoveryError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.gateway-recovery-bundle-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

enum IdError {
    Empty,
    NonCanonical,
    TooLong,
}
fn validate_id(value: &str) -> Result<(), IdError> {
    if value.trim().is_empty() {
        return Err(IdError::Empty);
    }
    if value != value.trim() {
        return Err(IdError::NonCanonical);
    }
    if value.len() > MAX_RECOVERY_ID_BYTES {
        return Err(IdError::TooLong);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::SignatureAlgorithm;
    use crate::audit::AuditJournal;
    use crate::gateway_consensus_tracker::GatewayConsensusTracker;
    use crate::gateway_state::FabricationGatewayState;
    use crate::incident_ledger::IncidentLedger;
    use crate::operator_command_tracker::OperatorCommandTracker;
    use crate::session::MachineSessionTracker;
    use crate::submission_ledger::SubmissionLedger;
    use crate::telemetry_tracker::MachineTelemetryTracker;
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot};

    fn state() -> FabricationGatewayState {
        FabricationGatewayState::genesis(
            500_000,
            TrustSnapshot::new(
                1,
                100,
                1_000,
                vec![KeyTrustRecord {
                    algorithm: SignatureAlgorithm::Ed25519,
                    key_id: "root".into(),
                    not_before_unix_s: 100,
                    not_after_unix_s: None,
                    status: KeyLifecycleStatus::Active,
                    usages: BTreeSet::from([KeyUsage::GatewayConsensus]),
                }],
            )
            .unwrap(),
            AuditJournal::default(),
            MachineSessionTracker::default(),
            MachineTelemetryTracker::default(),
            SubmissionLedger::default(),
            OperatorCommandTracker::default(),
            GatewayConsensusTracker::default(),
            IncidentLedger::default(),
        )
        .unwrap()
    }

    fn checkpoint(state: FabricationGatewayState) -> GatewayRecoveryCheckpoint {
        let envelope = GatewayStateEnvelope::seal(state.clone()).unwrap();
        GatewayRecoveryCheckpoint {
            backup_id: format!("backup-{}", state.generation),
            captured_at_unix_ms: state.committed_at_unix_ms,
            consensus: GatewayConsensusEvidence {
                state_digest: envelope.state_digest,
                generation: state.generation,
                consensus_digest: Sha256Digest([7; 32]),
                trust_snapshot_digest: crate::trust::digest_trust_snapshot(&state.trust_snapshot)
                    .unwrap(),
                gateways: vec!["gateway-a".into()],
            },
            envelope,
        }
    }

    #[test]
    fn intact_recovery_bundle_restores_latest_state() {
        let state = state();
        let bundle =
            GatewayRecoveryBundle::build("site-a", 600_000, vec![checkpoint(state.clone())])
                .unwrap();
        assert_eq!(bundle.latest_state().unwrap(), state);
    }

    #[test]
    fn altered_checkpoint_breaks_bundle_digest() {
        let state = state();
        let mut bundle =
            GatewayRecoveryBundle::build("site-a", 600_000, vec![checkpoint(state)]).unwrap();
        bundle.checkpoints[0].captured_at_unix_ms += 1;
        assert_eq!(bundle.validate(), Err(GatewayRecoveryError::DigestMismatch));
    }
}
