// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only deployment lineage across promotion and rollback decisions.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::release_promotion::AuthorizedReleasePromotion;
use crate::release_rollback::AuthorizedReleaseRollback;
use serde::{Deserialize, Serialize};

pub const RELEASE_LINEAGE_SCHEMA: &str = "symthaea.fabrication.release-lineage.v1";
pub const MAX_RELEASE_LINEAGE_EVENTS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReleaseLineageAction {
    Promotion,
    Rollback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseLineageEvent {
    pub sequence: u64,
    pub recorded_at_unix_s: u64,
    pub action: ReleaseLineageAction,
    pub authority_digest: Sha256Digest,
    pub previous_active_promotion_digest: Option<Sha256Digest>,
    pub resulting_active_promotion_digest: Sha256Digest,
    pub previous_event_digest: Option<Sha256Digest>,
    pub event_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseLineage {
    pub schema_version: String,
    pub events: Vec<ReleaseLineageEvent>,
}

impl Default for ReleaseLineage {
    fn default() -> Self {
        Self {
            schema_version: RELEASE_LINEAGE_SCHEMA.into(),
            events: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseLineageError {
    UnsupportedSchema,
    CapacityExceeded,
    TimeRegressed,
    InvalidSequence,
    PreviousDigestMismatch,
    EventDigestMismatch,
    ActivePromotionMismatch,
    PromotionAlreadyActive,
    RollbackFromMismatch,
    RollbackTargetAlreadyActive,
    RollbackExpired,
    Encoding(String),
}

impl ReleaseLineage {
    pub fn validate(&self) -> Result<(), ReleaseLineageError> {
        if self.schema_version != RELEASE_LINEAGE_SCHEMA {
            return Err(ReleaseLineageError::UnsupportedSchema);
        }
        if self.events.len() > MAX_RELEASE_LINEAGE_EVENTS {
            return Err(ReleaseLineageError::CapacityExceeded);
        }
        let mut previous_event_digest = None;
        let mut active = None;
        let mut previous_time = None;
        for (index, event) in self.events.iter().enumerate() {
            if event.sequence != index as u64 + 1 {
                return Err(ReleaseLineageError::InvalidSequence);
            }
            if previous_time.is_some_and(|time| event.recorded_at_unix_s < time) {
                return Err(ReleaseLineageError::TimeRegressed);
            }
            if event.previous_event_digest != previous_event_digest {
                return Err(ReleaseLineageError::PreviousDigestMismatch);
            }
            if event.previous_active_promotion_digest != active {
                return Err(ReleaseLineageError::ActivePromotionMismatch);
            }
            let expected = compute_release_lineage_event_digest(
                event.sequence,
                event.recorded_at_unix_s,
                event.action,
                event.authority_digest,
                event.previous_active_promotion_digest,
                event.resulting_active_promotion_digest,
                event.previous_event_digest,
            )?;
            if expected != event.event_digest {
                return Err(ReleaseLineageError::EventDigestMismatch);
            }
            active = Some(event.resulting_active_promotion_digest);
            previous_event_digest = Some(event.event_digest);
            previous_time = Some(event.recorded_at_unix_s);
        }
        Ok(())
    }

    pub fn active_promotion_digest(&self) -> Option<Sha256Digest> {
        self.events
            .last()
            .map(|event| event.resulting_active_promotion_digest)
    }

    pub fn chain_head(&self) -> Option<Sha256Digest> {
        self.events.last().map(|event| event.event_digest)
    }

    pub fn record_promotion(
        &mut self,
        promotion: &AuthorizedReleasePromotion,
        recorded_at_unix_s: u64,
    ) -> Result<Sha256Digest, ReleaseLineageError> {
        self.validate()?;
        if self.events.len() >= MAX_RELEASE_LINEAGE_EVENTS {
            return Err(ReleaseLineageError::CapacityExceeded);
        }
        if self.active_promotion_digest() == Some(promotion.promotion_digest()) {
            return Err(ReleaseLineageError::PromotionAlreadyActive);
        }
        self.append(
            recorded_at_unix_s,
            ReleaseLineageAction::Promotion,
            promotion.promotion_digest(),
            promotion.promotion_digest(),
        )
    }

    pub fn record_rollback(
        &mut self,
        rollback: &AuthorizedReleaseRollback,
        recorded_at_unix_s: u64,
    ) -> Result<Sha256Digest, ReleaseLineageError> {
        self.validate()?;
        if recorded_at_unix_s >= rollback.evidence().expires_at_unix_s {
            return Err(ReleaseLineageError::RollbackExpired);
        }
        if self.active_promotion_digest() != Some(rollback.evidence().from_promotion_digest) {
            return Err(ReleaseLineageError::RollbackFromMismatch);
        }
        if self.active_promotion_digest() == Some(rollback.evidence().target_promotion_digest) {
            return Err(ReleaseLineageError::RollbackTargetAlreadyActive);
        }
        self.append(
            recorded_at_unix_s,
            ReleaseLineageAction::Rollback,
            rollback.rollback_digest(),
            rollback.evidence().target_promotion_digest,
        )
    }

    fn append(
        &mut self,
        recorded_at_unix_s: u64,
        action: ReleaseLineageAction,
        authority_digest: Sha256Digest,
        resulting_active_promotion_digest: Sha256Digest,
    ) -> Result<Sha256Digest, ReleaseLineageError> {
        if self
            .events
            .last()
            .is_some_and(|event| recorded_at_unix_s < event.recorded_at_unix_s)
        {
            return Err(ReleaseLineageError::TimeRegressed);
        }
        let sequence = self.events.len() as u64 + 1;
        let previous_active_promotion_digest = self.active_promotion_digest();
        let previous_event_digest = self.chain_head();
        let event_digest = compute_release_lineage_event_digest(
            sequence,
            recorded_at_unix_s,
            action,
            authority_digest,
            previous_active_promotion_digest,
            resulting_active_promotion_digest,
            previous_event_digest,
        )?;
        self.events.push(ReleaseLineageEvent {
            sequence,
            recorded_at_unix_s,
            action,
            authority_digest,
            previous_active_promotion_digest,
            resulting_active_promotion_digest,
            previous_event_digest,
            event_digest,
        });
        Ok(event_digest)
    }
}

pub fn digest_release_lineage(
    lineage: &ReleaseLineage,
) -> Result<Sha256Digest, ReleaseLineageError> {
    lineage.validate()?;
    let bytes = serde_json::to_vec(lineage)
        .map_err(|error| ReleaseLineageError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.release-lineage-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[allow(clippy::too_many_arguments)]
fn compute_release_lineage_event_digest(
    sequence: u64,
    recorded_at_unix_s: u64,
    action: ReleaseLineageAction,
    authority_digest: Sha256Digest,
    previous_active_promotion_digest: Option<Sha256Digest>,
    resulting_active_promotion_digest: Sha256Digest,
    previous_event_digest: Option<Sha256Digest>,
) -> Result<Sha256Digest, ReleaseLineageError> {
    let bytes = serde_json::to_vec(&(
        sequence,
        recorded_at_unix_s,
        action,
        authority_digest,
        previous_active_promotion_digest,
        resulting_active_promotion_digest,
        previous_event_digest,
    ))
    .map_err(|error| ReleaseLineageError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.release-lineage-event.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_lineage_is_valid_and_has_no_head() {
        let lineage = ReleaseLineage::default();
        assert_eq!(lineage.validate(), Ok(()));
        assert_eq!(lineage.chain_head(), None);
    }
}
