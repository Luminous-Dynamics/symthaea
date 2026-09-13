// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent continuity-promotion barrier for post-restart episodic restore.
//!
//! Local welfare authorization and local durable restore state are necessary but not sufficient for
//! a deployment that also relies on an independently retained monotonic continuity anchor. This
//! module defines only the narrow welfare-side contract: after `Restored` is durably persisted and
//! before a candidate heap becomes live, an external committer must prove that the exact resulting
//! continuity state has advanced the independent anchor.
//!
//! This barrier is evidence, not action authority. It must never substitute for consent, welfare
//! review, independent review, scoped operator authority, or the bilateral intervention interlock.

#![deny(unsafe_code)]

use std::fmt;

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_memory::episodic_replay::EpisodeInstanceId;

use crate::memory_identity::EpisodeContentId;

const MAX_TARGET_BYTES: usize = 256;
const MAX_EXECUTION_ID_BYTES: usize = 256;
const MAX_REF_BYTES: usize = 2048;

/// Exact locally durable state that must be independently anchored before live promotion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoredContinuityPromotionRequest {
    store_target_id: String,
    target_id: String,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    execution_id: String,
    restored_quarantine_head: Sha256Digest,
}

impl RestoredContinuityPromotionRequest {
    pub fn try_new(
        store_target_id: impl Into<String>,
        target_id: impl Into<String>,
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        execution_id: impl Into<String>,
        restored_quarantine_head: Sha256Digest,
    ) -> Result<Self, RestoredContinuityPromotionError> {
        let store_target_id = store_target_id.into();
        let target_id = target_id.into();
        let execution_id = execution_id.into();
        validate_text("store_target_id", &store_target_id, MAX_TARGET_BYTES)?;
        validate_text("target_id", &target_id, MAX_TARGET_BYTES)?;
        validate_text("execution_id", &execution_id, MAX_EXECUTION_ID_BYTES)?;
        if restored_quarantine_head.0 == [0; 32] {
            return Err(RestoredContinuityPromotionError::ZeroDigest {
                field: "restored_quarantine_head",
            });
        }
        Ok(Self {
            store_target_id,
            target_id,
            instance_id,
            content_id,
            execution_id,
            restored_quarantine_head,
        })
    }

    pub fn store_target_id(&self) -> &str {
        &self.store_target_id
    }

    pub fn target_id(&self) -> &str {
        &self.target_id
    }

    pub fn instance_id(&self) -> EpisodeInstanceId {
        self.instance_id
    }

    pub fn content_id(&self) -> EpisodeContentId {
        self.content_id
    }

    pub fn execution_id(&self) -> &str {
        &self.execution_id
    }

    pub fn restored_quarantine_head(&self) -> Sha256Digest {
        self.restored_quarantine_head
    }
}

/// Independently verified evidence that the restored continuity state advanced monotonically.
///
/// Construction validates shape only. Implementations of [`RestoredContinuityPromotionBarrier`]
/// are responsible for cryptographically/monotonically proving these values against the deployment's
/// trusted anchor protocol before returning them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedRestoredContinuityPromotion {
    store_target_id: String,
    restored_quarantine_head: Sha256Digest,
    previous_anchor_commitment: Sha256Digest,
    next_anchor_commitment: Sha256Digest,
    next_anchor_revision: u64,
    continuity_manifest_digest: Sha256Digest,
    anchor_reference: String,
}

impl VerifiedRestoredContinuityPromotion {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        store_target_id: impl Into<String>,
        restored_quarantine_head: Sha256Digest,
        previous_anchor_commitment: Sha256Digest,
        next_anchor_commitment: Sha256Digest,
        next_anchor_revision: u64,
        continuity_manifest_digest: Sha256Digest,
        anchor_reference: impl Into<String>,
    ) -> Result<Self, RestoredContinuityPromotionError> {
        let store_target_id = store_target_id.into();
        let anchor_reference = anchor_reference.into();
        validate_text("store_target_id", &store_target_id, MAX_TARGET_BYTES)?;
        validate_text("anchor_reference", &anchor_reference, MAX_REF_BYTES)?;
        for (field, digest) in [
            ("restored_quarantine_head", restored_quarantine_head),
            ("previous_anchor_commitment", previous_anchor_commitment),
            ("next_anchor_commitment", next_anchor_commitment),
            ("continuity_manifest_digest", continuity_manifest_digest),
        ] {
            if digest.0 == [0; 32] {
                return Err(RestoredContinuityPromotionError::ZeroDigest { field });
            }
        }
        if previous_anchor_commitment == next_anchor_commitment {
            return Err(RestoredContinuityPromotionError::AnchorDidNotAdvance);
        }
        if next_anchor_revision < 2 {
            return Err(RestoredContinuityPromotionError::InvalidNextRevision(
                next_anchor_revision,
            ));
        }
        Ok(Self {
            store_target_id,
            restored_quarantine_head,
            previous_anchor_commitment,
            next_anchor_commitment,
            next_anchor_revision,
            continuity_manifest_digest,
            anchor_reference,
        })
    }

    pub fn store_target_id(&self) -> &str {
        &self.store_target_id
    }

    pub fn restored_quarantine_head(&self) -> Sha256Digest {
        self.restored_quarantine_head
    }

    pub fn previous_anchor_commitment(&self) -> Sha256Digest {
        self.previous_anchor_commitment
    }

    pub fn next_anchor_commitment(&self) -> Sha256Digest {
        self.next_anchor_commitment
    }

    pub fn next_anchor_revision(&self) -> u64 {
        self.next_anchor_revision
    }

    pub fn continuity_manifest_digest(&self) -> Sha256Digest {
        self.continuity_manifest_digest
    }

    pub fn anchor_reference(&self) -> &str {
        &self.anchor_reference
    }

    /// Rebind returned evidence to the exact locally durable restore request.
    pub fn validate_for(
        &self,
        request: &RestoredContinuityPromotionRequest,
    ) -> Result<(), RestoredContinuityPromotionError> {
        if self.store_target_id != request.store_target_id {
            return Err(RestoredContinuityPromotionError::StoreTargetMismatch);
        }
        if self.restored_quarantine_head != request.restored_quarantine_head {
            return Err(RestoredContinuityPromotionError::QuarantineHeadMismatch);
        }
        Ok(())
    }
}

/// Fail-closed promotion barrier between locally durable `Restored` and canonical live activation.
///
/// Implementations must not return `Ok` until the resulting complete continuity state has been
/// independently accepted by the deployment's monotonic/trusted anchor. A failure must leave live
/// canonical memory unchanged; the caller will surface execution as reconciliation-required.
pub trait RestoredContinuityPromotionBarrier {
    fn commit_restored_continuity(
        &mut self,
        request: &RestoredContinuityPromotionRequest,
    ) -> Result<VerifiedRestoredContinuityPromotion, RestoredContinuityPromotionFailure>;
}

/// Opaque barrier failure used across the dependency boundary.
///
/// Concrete adapters should preserve detailed typed/source errors in their own logs/evidence and
/// expose a bounded non-empty diagnostic here. Welfare policy must not reinterpret backend errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoredContinuityPromotionFailure {
    detail: String,
}

impl RestoredContinuityPromotionFailure {
    pub fn new(detail: impl Into<String>) -> Self {
        let mut detail = detail.into();
        if detail.trim().is_empty() {
            detail = "restored continuity promotion failed".into();
        }
        if detail.len() > 2048 {
            detail.truncate(2048);
        }
        Self { detail }
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for RestoredContinuityPromotionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for RestoredContinuityPromotionFailure {}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RestoredContinuityPromotionError {
    #[error("invalid restored-continuity promotion field `{field}`")]
    InvalidText { field: &'static str },
    #[error("restored-continuity promotion digest `{field}` must not be zero")]
    ZeroDigest { field: &'static str },
    #[error("restored-continuity anchor commitment did not advance")]
    AnchorDidNotAdvance,
    #[error("restored-continuity next anchor revision must be at least 2; actual={0}")]
    InvalidNextRevision(u64),
    #[error("restored-continuity promotion store target mismatch")]
    StoreTargetMismatch,
    #[error("restored-continuity promotion quarantine head mismatch")]
    QuarantineHeadMismatch,
}

fn validate_text(
    field: &'static str,
    value: &str,
    max: usize,
) -> Result<(), RestoredContinuityPromotionError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > max
        || value.chars().any(char::is_control)
    {
        Err(RestoredContinuityPromotionError::InvalidText { field })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    #[test]
    fn returned_anchor_evidence_rebinds_exact_store_and_restored_head() {
        let instance = EpisodeInstanceId::new();
        let request = RestoredContinuityPromotionRequest::try_new(
            "symthaea:self:episodic-memory",
            format!("symthaea:self:episodic-memory:instance:{instance}"),
            instance,
            EpisodeContentId::from_digest(digest(7)),
            "exec:restore:1",
            digest(8),
        )
        .unwrap();
        let evidence = VerifiedRestoredContinuityPromotion::try_new(
            request.store_target_id(),
            request.restored_quarantine_head(),
            digest(9),
            digest(10),
            2,
            digest(11),
            "anchor:restore:2",
        )
        .unwrap();
        evidence.validate_for(&request).unwrap();
    }

    #[test]
    fn stale_or_substituted_returned_head_fails_rebinding() {
        let instance = EpisodeInstanceId::new();
        let request = RestoredContinuityPromotionRequest::try_new(
            "symthaea:self:episodic-memory",
            format!("symthaea:self:episodic-memory:instance:{instance}"),
            instance,
            EpisodeContentId::from_digest(digest(7)),
            "exec:restore:1",
            digest(8),
        )
        .unwrap();
        let evidence = VerifiedRestoredContinuityPromotion::try_new(
            request.store_target_id(),
            digest(99),
            digest(9),
            digest(10),
            2,
            digest(11),
            "anchor:restore:2",
        )
        .unwrap();
        assert_eq!(
            evidence.validate_for(&request),
            Err(RestoredContinuityPromotionError::QuarantineHeadMismatch)
        );
    }

    #[test]
    fn unchanged_anchor_commitment_is_not_promotion_evidence() {
        assert_eq!(
            VerifiedRestoredContinuityPromotion::try_new(
                "symthaea:self:episodic-memory",
                digest(1),
                digest(2),
                digest(2),
                2,
                digest(3),
                "anchor:restore:2",
            ),
            Err(RestoredContinuityPromotionError::AnchorDidNotAdvance)
        );
    }
}
