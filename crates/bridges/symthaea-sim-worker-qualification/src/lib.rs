// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Host-issued authority for exact simulation-worker qualification.
//!
//! Persisted qualification evidence is not executable authority. The boundary is:
//!
//! ```text
//! WorkerQualificationEvidence
//!     != WorkerQualificationRecord
//!     != ActiveWorkerQualification
//! ```
//!
//! A host may issue a [`WorkerQualificationRecord`] only after it has reviewed
//! the exact worker image against the named qualification evidence (for example,
//! exact-head CI receipts and negative containment probes). Activating the
//! record requires the exact sealed image digest and a live currentness source.
//! The active token remains non-serializable and must be rechecked at use sites.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use symthaea_sim_worker::{SUPERVISOR_PROFILE_V1, WORKER_PROTOCOL_V1};
use symthaea_sim_worker_containment::WORKER_CONTAINMENT_PROFILE_V1;
use symthaea_sim_worker_filesystem::WORKER_FILESYSTEM_PROFILE_V1;
use symthaea_sim_worker_image::{SEALED_WORKER_IMAGE_PROFILE_V1, SealedWorkerImage};
use thiserror::Error;

pub const WORKER_QUALIFICATION_PROFILE_V1: &str =
    "symthaea.simulation.worker-qualification.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerQualificationContext {
    pub current_generation: u64,
    pub revoked: bool,
}

impl WorkerQualificationContext {
    pub const fn active(current_generation: u64) -> Self {
        Self {
            current_generation,
            revoked: false,
        }
    }

    pub const fn revoked(current_generation: u64) -> Self {
        Self {
            current_generation,
            revoked: true,
        }
    }
}

pub trait WorkerQualificationCurrentnessSource: Debug + Send + Sync {
    fn current_context(&self, worker_sha256: [u8; 32]) -> Option<WorkerQualificationContext>;
}

/// Host-issued qualification record. Serializable for audit transport, but
/// deliberately not deserializable back into authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerQualificationRecord {
    worker_sha256: [u8; 32],
    qualification_evidence_sha256: [u8; 32],
    generation: u64,
}

/// Persistable evidence-only representation. There is intentionally no upgrade
/// API from this type to [`WorkerQualificationRecord`] or
/// [`ActiveWorkerQualification`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerQualificationEvidence {
    pub profile: String,
    pub worker_sha256: String,
    pub qualification_evidence_sha256: String,
    pub generation: u64,
    pub worker_image_profile: String,
    pub supervisor_profile: String,
    pub worker_protocol: String,
    pub process_containment_profile: String,
    pub filesystem_containment_profile: String,
}

impl WorkerQualificationRecord {
    pub fn issue(
        worker_sha256: [u8; 32],
        qualification_evidence_sha256: [u8; 32],
        generation: u64,
    ) -> Result<Self, WorkerQualificationError> {
        if worker_sha256 == [0; 32] {
            return Err(WorkerQualificationError::ZeroWorkerDigest);
        }
        if qualification_evidence_sha256 == [0; 32] {
            return Err(WorkerQualificationError::ZeroEvidenceDigest);
        }
        if generation == 0 {
            return Err(WorkerQualificationError::ZeroGeneration);
        }
        Ok(Self {
            worker_sha256,
            qualification_evidence_sha256,
            generation,
        })
    }

    pub fn worker_sha256(&self) -> [u8; 32] {
        self.worker_sha256
    }

    pub fn qualification_evidence_sha256(&self) -> [u8; 32] {
        self.qualification_evidence_sha256
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn evidence(&self) -> WorkerQualificationEvidence {
        WorkerQualificationEvidence {
            profile: WORKER_QUALIFICATION_PROFILE_V1.into(),
            worker_sha256: hex_digest(self.worker_sha256),
            qualification_evidence_sha256: hex_digest(self.qualification_evidence_sha256),
            generation: self.generation,
            worker_image_profile: SEALED_WORKER_IMAGE_PROFILE_V1.into(),
            supervisor_profile: SUPERVISOR_PROFILE_V1.into(),
            worker_protocol: WORKER_PROTOCOL_V1.into(),
            process_containment_profile: WORKER_CONTAINMENT_PROFILE_V1.into(),
            filesystem_containment_profile: WORKER_FILESYSTEM_PROFILE_V1.into(),
        }
    }

    pub fn activate(
        &self,
        image: &SealedWorkerImage,
        currentness: &dyn WorkerQualificationCurrentnessSource,
    ) -> Result<ActiveWorkerQualification, WorkerQualificationError> {
        if image.image_sha256() != self.worker_sha256 {
            return Err(WorkerQualificationError::WorkerDigestMismatch);
        }
        validate_currentness(
            self.generation,
            currentness
                .current_context(self.worker_sha256)
                .ok_or(WorkerQualificationError::CurrentnessUnavailable)?,
        )?;
        Ok(ActiveWorkerQualification {
            record: self.clone(),
        })
    }
}

/// Non-serializable proof that one exact worker image was activated from the
/// host's qualification authority.
#[derive(Debug, PartialEq, Eq)]
pub struct ActiveWorkerQualification {
    record: WorkerQualificationRecord,
}

impl ActiveWorkerQualification {
    pub fn worker_sha256(&self) -> [u8; 32] {
        self.record.worker_sha256()
    }

    pub fn qualification_evidence_sha256(&self) -> [u8; 32] {
        self.record.qualification_evidence_sha256()
    }

    pub fn generation(&self) -> u64 {
        self.record.generation()
    }

    pub fn matches_image(&self, image: &SealedWorkerImage) -> bool {
        image.image_sha256() == self.worker_sha256()
    }

    pub fn recheck_currentness(
        &self,
        currentness: &dyn WorkerQualificationCurrentnessSource,
    ) -> Result<(), WorkerQualificationError> {
        let context = currentness
            .current_context(self.worker_sha256())
            .ok_or(WorkerQualificationError::CurrentnessUnavailable)?;
        validate_currentness(self.generation(), context)
    }

    pub fn evidence(&self) -> WorkerQualificationEvidence {
        self.record.evidence()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum WorkerQualificationError {
    #[error("worker qualification digest cannot be zero")]
    ZeroWorkerDigest,
    #[error("worker qualification evidence digest cannot be zero")]
    ZeroEvidenceDigest,
    #[error("worker qualification generation must be nonzero")]
    ZeroGeneration,
    #[error("sealed worker image does not match qualification digest")]
    WorkerDigestMismatch,
    #[error("worker qualification currentness is unavailable")]
    CurrentnessUnavailable,
    #[error("worker qualification has been revoked")]
    Revoked,
    #[error("worker qualification generation mismatch: qualified={qualified}, current={current}")]
    GenerationMismatch { qualified: u64, current: u64 },
}

fn validate_currentness(
    generation: u64,
    context: WorkerQualificationContext,
) -> Result<(), WorkerQualificationError> {
    if context.revoked {
        return Err(WorkerQualificationError::Revoked);
    }
    if context.current_generation != generation {
        return Err(WorkerQualificationError::GenerationMismatch {
            qualified: generation,
            current: context.current_generation,
        });
    }
    Ok(())
}

fn hex_digest(bytes: [u8; 32]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(TABLE[(byte >> 4) as usize] as char);
        out.push(TABLE[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct FixedCurrentness(WorkerQualificationContext);

    impl WorkerQualificationCurrentnessSource for FixedCurrentness {
        fn current_context(&self, _worker_sha256: [u8; 32]) -> Option<WorkerQualificationContext> {
            Some(self.0)
        }
    }

    #[test]
    fn evidence_is_structural_and_profile_explicit() {
        let record = WorkerQualificationRecord::issue([0x11; 32], [0x22; 32], 7).unwrap();
        let evidence = record.evidence();
        assert_eq!(evidence.profile, WORKER_QUALIFICATION_PROFILE_V1);
        assert_eq!(evidence.worker_image_profile, SEALED_WORKER_IMAGE_PROFILE_V1);
        assert_eq!(evidence.process_containment_profile, WORKER_CONTAINMENT_PROFILE_V1);
        assert_eq!(evidence.filesystem_containment_profile, WORKER_FILESYSTEM_PROFILE_V1);
    }

    #[test]
    fn currentness_generation_is_exact_and_revocation_fails_closed() {
        let record = WorkerQualificationRecord::issue([0x11; 32], [0x22; 32], 7).unwrap();
        assert!(matches!(
            validate_currentness(7, WorkerQualificationContext::active(8)),
            Err(WorkerQualificationError::GenerationMismatch { .. })
        ));
        assert_eq!(
            validate_currentness(7, WorkerQualificationContext::revoked(7)),
            Err(WorkerQualificationError::Revoked)
        );
        let source = FixedCurrentness(WorkerQualificationContext::active(7));
        assert_eq!(source.current_context([0x11; 32]).unwrap().current_generation, 7);
    }
}
