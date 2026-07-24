// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Threshold-authorized anchors for compacted operational evidence.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::evidence_compaction::{
    CompactedEvidence, EvidenceCompactionError, EvidenceCompactionPolicy, digest_compacted_evidence,
};
use crate::threshold::VerifiedThresholdCeremony;
use serde::{Deserialize, Serialize};

pub const EVIDENCE_COMPACTION_ANCHOR_SCHEMA: &str =
    "symthaea.fabrication.evidence-compaction-anchor.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCompactionAnchor {
    pub schema_version: String,
    pub checkpoint_digest: Sha256Digest,
    pub total_count: u64,
    pub prefix_count: u64,
    pub final_head: Sha256Digest,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizedEvidenceCompactionAnchor {
    pub anchor: EvidenceCompactionAnchor,
    pub anchor_digest: Sha256Digest,
    pub ceremony_digest: Sha256Digest,
    pub trust_snapshot_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceAnchorPolicy {
    pub maximum_lifetime_s: u64,
}

impl Default for EvidenceAnchorPolicy {
    fn default() -> Self {
        Self {
            maximum_lifetime_s: 30 * 24 * 60 * 60,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceAnchorError {
    UnsupportedSchema,
    InvalidWindow,
    LifetimeTooLong,
    CheckpointMismatch,
    CountMismatch,
    FinalHeadMismatch,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Compaction(EvidenceCompactionError),
    Encoding(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceAnchorTracker {
    latest_total_count: Option<u64>,
    latest_anchor_digest: Option<Sha256Digest>,
    latest_final_head: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceAnchorTrackingError {
    CountRollback { latest: u64, proposed: u64 },
    SameCountSubstitution,
}

pub fn build_evidence_compaction_anchor(
    compacted: &CompactedEvidence,
    compaction_policy: &EvidenceCompactionPolicy,
    issued_at_unix_s: u64,
    expires_at_unix_s: u64,
    anchor_policy: &EvidenceAnchorPolicy,
) -> Result<EvidenceCompactionAnchor, EvidenceAnchorError> {
    if issued_at_unix_s >= expires_at_unix_s {
        return Err(EvidenceAnchorError::InvalidWindow);
    }
    if expires_at_unix_s - issued_at_unix_s > anchor_policy.maximum_lifetime_s {
        return Err(EvidenceAnchorError::LifetimeTooLong);
    }
    let checkpoint_digest = digest_compacted_evidence(compacted, compaction_policy)
        .map_err(EvidenceAnchorError::Compaction)?;
    Ok(EvidenceCompactionAnchor {
        schema_version: EVIDENCE_COMPACTION_ANCHOR_SCHEMA.into(),
        checkpoint_digest,
        total_count: compacted.total_count,
        prefix_count: compacted.prefix_count,
        final_head: compacted.final_head,
        issued_at_unix_s,
        expires_at_unix_s,
    })
}

pub fn verify_evidence_compaction_anchor(
    anchor: &EvidenceCompactionAnchor,
    compacted: &CompactedEvidence,
    compaction_policy: &EvidenceCompactionPolicy,
    now_unix_s: u64,
) -> Result<(), EvidenceAnchorError> {
    if anchor.schema_version != EVIDENCE_COMPACTION_ANCHOR_SCHEMA {
        return Err(EvidenceAnchorError::UnsupportedSchema);
    }
    if anchor.issued_at_unix_s >= anchor.expires_at_unix_s
        || now_unix_s < anchor.issued_at_unix_s
        || now_unix_s >= anchor.expires_at_unix_s
    {
        return Err(EvidenceAnchorError::InvalidWindow);
    }
    let expected = digest_compacted_evidence(compacted, compaction_policy)
        .map_err(EvidenceAnchorError::Compaction)?;
    if anchor.checkpoint_digest != expected {
        return Err(EvidenceAnchorError::CheckpointMismatch);
    }
    if anchor.total_count != compacted.total_count || anchor.prefix_count != compacted.prefix_count
    {
        return Err(EvidenceAnchorError::CountMismatch);
    }
    if anchor.final_head != compacted.final_head {
        return Err(EvidenceAnchorError::FinalHeadMismatch);
    }
    Ok(())
}

pub fn digest_evidence_compaction_anchor(
    anchor: &EvidenceCompactionAnchor,
) -> Result<Sha256Digest, EvidenceAnchorError> {
    if anchor.schema_version != EVIDENCE_COMPACTION_ANCHOR_SCHEMA {
        return Err(EvidenceAnchorError::UnsupportedSchema);
    }
    if anchor.issued_at_unix_s >= anchor.expires_at_unix_s {
        return Err(EvidenceAnchorError::InvalidWindow);
    }
    let bytes = serde_json::to_vec(anchor)
        .map_err(|error| EvidenceAnchorError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.evidence-compaction-anchor-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_evidence_compaction_anchor(
    anchor: EvidenceCompactionAnchor,
    compacted: &CompactedEvidence,
    compaction_policy: &EvidenceCompactionPolicy,
    now_unix_s: u64,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedEvidenceCompactionAnchor, EvidenceAnchorError> {
    verify_evidence_compaction_anchor(&anchor, compacted, compaction_policy, now_unix_s)?;
    let anchor_digest = digest_evidence_compaction_anchor(&anchor)?;
    if ceremony.purpose() != "evidence-compaction-anchor" {
        return Err(EvidenceAnchorError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != anchor_digest {
        return Err(EvidenceAnchorError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedEvidenceCompactionAnchor {
        anchor,
        anchor_digest,
        ceremony_digest: ceremony.ceremony_digest(),
        trust_snapshot_digest: ceremony.trust_snapshot_digest(),
    })
}

impl EvidenceAnchorTracker {
    pub fn accept(
        &mut self,
        authorized: &AuthorizedEvidenceCompactionAnchor,
    ) -> Result<(), EvidenceAnchorTrackingError> {
        if let Some(latest) = self.latest_total_count {
            if authorized.anchor.total_count < latest {
                return Err(EvidenceAnchorTrackingError::CountRollback {
                    latest,
                    proposed: authorized.anchor.total_count,
                });
            }
            if authorized.anchor.total_count == latest {
                if self.latest_anchor_digest == Some(authorized.anchor_digest) {
                    return Ok(());
                }
                return Err(EvidenceAnchorTrackingError::SameCountSubstitution);
            }
        }
        self.latest_total_count = Some(authorized.anchor.total_count);
        self.latest_anchor_digest = Some(authorized.anchor_digest);
        self.latest_final_head = Some(authorized.anchor.final_head);
        Ok(())
    }

    pub fn latest_anchor_digest(&self) -> Option<Sha256Digest> {
        self.latest_anchor_digest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;
    use crate::evidence_compaction::{EvidenceJournal, compact_evidence};

    #[test]
    fn anchor_binds_exact_checkpoint_count_and_head() {
        let mut journal = EvidenceJournal::default();
        journal.append(1, "event", sha256(b"one")).unwrap();
        journal.append(2, "event", sha256(b"two")).unwrap();
        let policy = EvidenceCompactionPolicy {
            minimum_retained_tail: 1,
            maximum_retained_tail: 4,
        };
        let compacted = compact_evidence(&journal, 1, None, &policy).unwrap();
        let anchor = build_evidence_compaction_anchor(
            &compacted,
            &policy,
            10,
            20,
            &EvidenceAnchorPolicy::default(),
        )
        .unwrap();
        verify_evidence_compaction_anchor(&anchor, &compacted, &policy, 15).unwrap();
        let mut altered = compacted.clone();
        altered.total_count += 1;
        assert!(verify_evidence_compaction_anchor(&anchor, &altered, &policy, 15).is_err());
    }
}
