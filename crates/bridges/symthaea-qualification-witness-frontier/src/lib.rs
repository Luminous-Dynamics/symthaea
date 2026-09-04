// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ancestry-aware recovery semantics for externally anchored witness frontiers.
//!
//! This crate intentionally does not implement Xenia, TPM NV, SCITT, or another
//! anchor transport. It consumes an externally verified *current enough* anchor
//! claim through a verifier trait and a locally audited history view through a
//! separate trait. The classifier then answers whether local state is exactly
//! anchored, safely ahead on the same reservation chain, missing/rolled back, or
//! divergent.
//!
//! The critical asymmetry is:
//!
//! ```text
//! local < trusted external frontier  => contain as rollback/missing local state
//! local = trusted external frontier  => publication may proceed
//! local > trusted external frontier  => only safe when trusted head is a proven
//!                                       ancestor; preserve local and re-anchor
//! ```
//!
//! A larger sequence number alone is never accepted as proof of ancestry.
//! Evidence chronology remains separate from execution authority.

#![deny(unsafe_code)]

use std::fmt;

use symthaea_authority::Digest32;
use symthaea_qualification_witness_sequence::WitnessSequenceFrontierStatementV1;
use thiserror::Error;

pub const EXTERNAL_ANCHOR_SCHEMA_VERSION: u16 = 1;

const FRONTIER_DOMAIN: &[u8] = b"symthaea.qualification-witness.sequence-frontier.v1\0";
const ZERO32: [u8; 32] = [0; 32];

/// Structural frontier point independent of one storage or anchor transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessFrontierPointV1 {
    pub witness_id: [u8; 16],
    pub high_watermark: u64,
    pub reservation_head: Digest32,
    pub statement_digest: Digest32,
}

impl WitnessFrontierPointV1 {
    pub fn from_local_statement(statement: WitnessSequenceFrontierStatementV1) -> Self {
        Self {
            witness_id: statement.witness_id(),
            high_watermark: statement.high_watermark(),
            reservation_head: statement.reservation_head(),
            statement_digest: statement.digest(),
        }
    }

    pub fn validate(self) -> Result<(), FrontierRecoveryError> {
        if self.witness_id == [0; 16]
            || self.high_watermark == 0
            || self.reservation_head.0 == ZERO32
            || self.statement_digest.0 == ZERO32
            || self.statement_digest != frontier_statement_digest(
                self.witness_id,
                self.high_watermark,
                self.reservation_head,
            )
        {
            return Err(FrontierRecoveryError::MalformedFrontier);
        }
        Ok(())
    }
}

/// Raw claim returned by an external anchoring system.
///
/// `source_sequence` is the anchoring system's own monotonic ordering domain,
/// not the witness reservation sequence. `freshness_evidence_digest` binds the
/// external verifier's proof/challenge/checkpoint used to decide this claim is
/// current enough. This crate does not interpret that proof itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalWitnessFrontierClaimV1 {
    pub schema_version: u16,
    pub source_id: [u8; 16],
    pub source_epoch: u64,
    pub source_sequence: u64,
    pub witness_id: [u8; 16],
    pub high_watermark: u64,
    pub reservation_head: Digest32,
    pub frontier_statement_digest: Digest32,
    pub freshness_evidence_digest: Digest32,
}

impl ExternalWitnessFrontierClaimV1 {
    fn validate(self) -> Result<(), FrontierRecoveryError> {
        if self.schema_version != EXTERNAL_ANCHOR_SCHEMA_VERSION
            || self.source_id == [0; 16]
            || self.source_epoch == 0
            || self.source_sequence == 0
            || self.witness_id == [0; 16]
            || self.high_watermark == 0
            || self.reservation_head.0 == ZERO32
            || self.frontier_statement_digest.0 == ZERO32
            || self.freshness_evidence_digest.0 == ZERO32
            || self.frontier_statement_digest
                != frontier_statement_digest(
                    self.witness_id,
                    self.high_watermark,
                    self.reservation_head,
                )
        {
            return Err(FrontierRecoveryError::MalformedExternalAnchor);
        }
        Ok(())
    }
}

/// Adapter contract for Xenia/TPM/transparency or another independent source.
///
/// Implementations MUST verify authentication/integrity and source freshness,
/// including the source's own anti-rollback/currentness policy. A merely valid
/// old signature is insufficient when the source can provide a newer frontier.
pub trait ExternalWitnessFrontierVerifier {
    fn verify_current(
        &self,
        claim: &ExternalWitnessFrontierClaimV1,
    ) -> Result<(), ExternalAnchorVerificationError>;
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("external frontier source rejected claim: {reason}")]
pub struct ExternalAnchorVerificationError {
    pub reason: String,
}

/// Opaque result of structural validation plus an external source verifier.
#[derive(Debug)]
pub struct VerifiedExternalWitnessFrontierV1 {
    claim: ExternalWitnessFrontierClaimV1,
}

impl VerifiedExternalWitnessFrontierV1 {
    pub fn point(&self) -> WitnessFrontierPointV1 {
        WitnessFrontierPointV1 {
            witness_id: self.claim.witness_id,
            high_watermark: self.claim.high_watermark,
            reservation_head: self.claim.reservation_head,
            statement_digest: self.claim.frontier_statement_digest,
        }
    }

    pub fn source_id(&self) -> [u8; 16] {
        self.claim.source_id
    }

    pub fn source_epoch(&self) -> u64 {
        self.claim.source_epoch
    }

    pub fn source_sequence(&self) -> u64 {
        self.claim.source_sequence
    }

    pub fn freshness_evidence_digest(&self) -> Digest32 {
        self.claim.freshness_evidence_digest
    }
}

pub fn verify_external_witness_frontier_v1<V: ExternalWitnessFrontierVerifier>(
    claim: ExternalWitnessFrontierClaimV1,
    verifier: &V,
) -> Result<VerifiedExternalWitnessFrontierV1, FrontierRecoveryError> {
    claim.validate()?;
    verifier.verify_current(&claim)?;
    Ok(VerifiedExternalWitnessFrontierV1 { claim })
}

/// Trusted local-history contract used by the recovery classifier.
///
/// `audit_witness` must validate the complete local reservation chain/current
/// frontier before any comparison. `reservation_head_at` must return the exact
/// local chain head at that historical reservation sequence, not synthesize it
/// from the current counter.
pub trait LocalWitnessFrontierHistory {
    fn audit_witness(&self, witness_id: [u8; 16]) -> Result<(), FrontierHistoryError>;

    fn current_frontier(
        &self,
        witness_id: [u8; 16],
    ) -> Result<Option<WitnessFrontierPointV1>, FrontierHistoryError>;

    fn reservation_head_at(
        &self,
        witness_id: [u8; 16],
        high_watermark: u64,
    ) -> Result<Option<Digest32>, FrontierHistoryError>;
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("local witness history unavailable or invalid: {reason}")]
pub struct FrontierHistoryError {
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WitnessFrontierRecoveryRelationV1 {
    /// No local reservations and no external anchor.
    EmptyUnanchored,
    /// Local state exists but no external anchor exists yet.
    InitialAnchorRequired {
        local_high_watermark: u64,
        local_statement_digest: Digest32,
    },
    /// Exact local/current equality with the verified external anchor.
    AnchoredCurrent {
        high_watermark: u64,
        statement_digest: Digest32,
    },
    /// Local state is newer and the trusted external head is proven to be a
    /// prefix of the audited local reservation chain.
    LocalAheadVerifiedDescendant {
        trusted_high_watermark: u64,
        local_high_watermark: u64,
        local_statement_digest: Digest32,
    },
    /// External source proves state exists beyond the local high watermark, or
    /// proves state while local history is missing entirely.
    RollbackOrMissingLocal {
        trusted_high_watermark: u64,
        local_high_watermark: Option<u64>,
    },
    /// Same sequence but different chain head/statement.
    DivergentAtSameHeight {
        high_watermark: u64,
    },
    /// Local counter is ahead, but its historical chain head at the trusted
    /// sequence does not match the external anchor.
    DivergentTrustedPrefix {
        trusted_high_watermark: u64,
        local_high_watermark: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WitnessFrontierPublicationDispositionV1 {
    /// External anchor and local audited frontier are exactly current.
    PublishAllowed,
    /// Preserve local state; obtain/advance external anchor before publication.
    AnchorRequired,
    /// Stop publication and require explicit recovery/investigation.
    Contained,
}

impl WitnessFrontierRecoveryRelationV1 {
    pub fn publication_disposition(self) -> WitnessFrontierPublicationDispositionV1 {
        match self {
            Self::AnchoredCurrent { .. } => WitnessFrontierPublicationDispositionV1::PublishAllowed,
            Self::EmptyUnanchored | Self::InitialAnchorRequired { .. } | Self::LocalAheadVerifiedDescendant { .. } => {
                WitnessFrontierPublicationDispositionV1::AnchorRequired
            }
            Self::RollbackOrMissingLocal { .. }
            | Self::DivergentAtSameHeight { .. }
            | Self::DivergentTrustedPrefix { .. } => WitnessFrontierPublicationDispositionV1::Contained,
        }
    }
}

/// Classify one audited local witness history against the latest externally
/// verified/current-enough anchor claim.
pub fn classify_witness_frontier_recovery_v1<H: LocalWitnessFrontierHistory>(
    history: &H,
    witness_id: [u8; 16],
    external: Option<&VerifiedExternalWitnessFrontierV1>,
) -> Result<WitnessFrontierRecoveryRelationV1, FrontierRecoveryError> {
    if witness_id == [0; 16] {
        return Err(FrontierRecoveryError::MalformedFrontier);
    }
    if let Some(external) = external {
        if external.point().witness_id != witness_id {
            return Err(FrontierRecoveryError::WitnessIdentityMismatch);
        }
    }

    history.audit_witness(witness_id)?;
    let local = history.current_frontier(witness_id)?;
    if let Some(local) = local {
        local.validate()?;
        if local.witness_id != witness_id {
            return Err(FrontierRecoveryError::WitnessIdentityMismatch);
        }
    }

    match (local, external) {
        (None, None) => Ok(WitnessFrontierRecoveryRelationV1::EmptyUnanchored),
        (Some(local), None) => Ok(WitnessFrontierRecoveryRelationV1::InitialAnchorRequired {
            local_high_watermark: local.high_watermark,
            local_statement_digest: local.statement_digest,
        }),
        (None, Some(external)) => Ok(WitnessFrontierRecoveryRelationV1::RollbackOrMissingLocal {
            trusted_high_watermark: external.point().high_watermark,
            local_high_watermark: None,
        }),
        (Some(local), Some(external)) => {
            let trusted = external.point();
            if local.high_watermark < trusted.high_watermark {
                return Ok(WitnessFrontierRecoveryRelationV1::RollbackOrMissingLocal {
                    trusted_high_watermark: trusted.high_watermark,
                    local_high_watermark: Some(local.high_watermark),
                });
            }
            if local.high_watermark == trusted.high_watermark {
                if local.reservation_head == trusted.reservation_head
                    && local.statement_digest == trusted.statement_digest
                {
                    return Ok(WitnessFrontierRecoveryRelationV1::AnchoredCurrent {
                        high_watermark: local.high_watermark,
                        statement_digest: local.statement_digest,
                    });
                }
                return Ok(WitnessFrontierRecoveryRelationV1::DivergentAtSameHeight {
                    high_watermark: local.high_watermark,
                });
            }

            let historical_head = history
                .reservation_head_at(witness_id, trusted.high_watermark)?
                .ok_or(FrontierRecoveryError::HistoricalPrefixUnavailable)?;
            if historical_head != trusted.reservation_head
                || frontier_statement_digest(witness_id, trusted.high_watermark, historical_head)
                    != trusted.statement_digest
            {
                return Ok(WitnessFrontierRecoveryRelationV1::DivergentTrustedPrefix {
                    trusted_high_watermark: trusted.high_watermark,
                    local_high_watermark: local.high_watermark,
                });
            }

            Ok(WitnessFrontierRecoveryRelationV1::LocalAheadVerifiedDescendant {
                trusted_high_watermark: trusted.high_watermark,
                local_high_watermark: local.high_watermark,
                local_statement_digest: local.statement_digest,
            })
        }
    }
}

fn frontier_statement_digest(
    witness_id: [u8; 16],
    high_watermark: u64,
    reservation_head: Digest32,
) -> Digest32 {
    let mut transcript = Transcript::new(FRONTIER_DOMAIN);
    transcript.u16(symthaea_qualification_witness_sequence::WITNESS_SEQUENCE_SCHEMA_VERSION);
    transcript.fixed(&witness_id);
    transcript.u64(high_watermark);
    transcript.fixed(&reservation_head.0);
    Digest32(transcript.finish())
}

#[derive(Debug, Error)]
pub enum FrontierRecoveryError {
    #[error("malformed witness frontier")]
    MalformedFrontier,
    #[error("malformed external witness frontier anchor")]
    MalformedExternalAnchor,
    #[error("external/local witness identity mismatch")]
    WitnessIdentityMismatch,
    #[error("local audited history lacks the trusted historical prefix")]
    HistoricalPrefixUnavailable,
    #[error(transparent)]
    ExternalVerification(#[from] ExternalAnchorVerificationError),
    #[error(transparent)]
    History(#[from] FrontierHistoryError),
}

struct Transcript {
    bytes: Vec<u8>,
}

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(domain.len() + 128);
        bytes.extend_from_slice(domain);
        Self { bytes }
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn fixed(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn finish(self) -> [u8; 32] {
        *blake3::hash(&self.bytes).as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    struct AcceptExternal;

    impl ExternalWitnessFrontierVerifier for AcceptExternal {
        fn verify_current(
            &self,
            _claim: &ExternalWitnessFrontierClaimV1,
        ) -> Result<(), ExternalAnchorVerificationError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct FakeHistory {
        witness_id: [u8; 16],
        heads: BTreeMap<u64, Digest32>,
        audit_ok: bool,
    }

    impl FakeHistory {
        fn with_heads(witness_id: [u8; 16], heads: &[(u64, u8)]) -> Self {
            Self {
                witness_id,
                heads: heads
                    .iter()
                    .map(|(sequence, byte)| (*sequence, Digest32([*byte; 32])))
                    .collect(),
                audit_ok: true,
            }
        }
    }

    impl LocalWitnessFrontierHistory for FakeHistory {
        fn audit_witness(&self, witness_id: [u8; 16]) -> Result<(), FrontierHistoryError> {
            if self.audit_ok && (self.heads.is_empty() || witness_id == self.witness_id) {
                Ok(())
            } else {
                Err(FrontierHistoryError {
                    reason: "fixture audit rejected".to_string(),
                })
            }
        }

        fn current_frontier(
            &self,
            witness_id: [u8; 16],
        ) -> Result<Option<WitnessFrontierPointV1>, FrontierHistoryError> {
            let Some((&high, &head)) = self.heads.last_key_value() else {
                return Ok(None);
            };
            Ok(Some(WitnessFrontierPointV1 {
                witness_id,
                high_watermark: high,
                reservation_head: head,
                statement_digest: frontier_statement_digest(witness_id, high, head),
            }))
        }

        fn reservation_head_at(
            &self,
            witness_id: [u8; 16],
            high_watermark: u64,
        ) -> Result<Option<Digest32>, FrontierHistoryError> {
            if witness_id != self.witness_id && !self.heads.is_empty() {
                return Err(FrontierHistoryError {
                    reason: "wrong witness".to_string(),
                });
            }
            Ok(self.heads.get(&high_watermark).copied())
        }
    }

    fn external(
        witness_id: [u8; 16],
        high_watermark: u64,
        head: Digest32,
    ) -> VerifiedExternalWitnessFrontierV1 {
        let claim = ExternalWitnessFrontierClaimV1 {
            schema_version: EXTERNAL_ANCHOR_SCHEMA_VERSION,
            source_id: [0xa1; 16],
            source_epoch: 4,
            source_sequence: 12,
            witness_id,
            high_watermark,
            reservation_head: head,
            frontier_statement_digest: frontier_statement_digest(witness_id, high_watermark, head),
            freshness_evidence_digest: Digest32([0xee; 32]),
        };
        verify_external_witness_frontier_v1(claim, &AcceptExternal).unwrap()
    }

    #[test]
    fn exact_anchor_allows_publication() {
        let witness = [1; 16];
        let history = FakeHistory::with_heads(witness, &[(1, 0x11), (2, 0x22)]);
        let anchor = external(witness, 2, Digest32([0x22; 32]));
        let relation = classify_witness_frontier_recovery_v1(&history, witness, Some(&anchor)).unwrap();
        assert!(matches!(
            relation,
            WitnessFrontierRecoveryRelationV1::AnchoredCurrent { high_watermark: 2, .. }
        ));
        assert_eq!(
            relation.publication_disposition(),
            WitnessFrontierPublicationDispositionV1::PublishAllowed
        );
    }

    #[test]
    fn local_ahead_requires_matching_trusted_ancestor() {
        let witness = [1; 16];
        let history = FakeHistory::with_heads(witness, &[(1, 0x11), (2, 0x22), (3, 0x33)]);
        let anchor = external(witness, 2, Digest32([0x22; 32]));
        let relation = classify_witness_frontier_recovery_v1(&history, witness, Some(&anchor)).unwrap();
        assert!(matches!(
            relation,
            WitnessFrontierRecoveryRelationV1::LocalAheadVerifiedDescendant {
                trusted_high_watermark: 2,
                local_high_watermark: 3,
                ..
            }
        ));
        assert_eq!(
            relation.publication_disposition(),
            WitnessFrontierPublicationDispositionV1::AnchorRequired
        );
    }

    #[test]
    fn larger_local_counter_with_wrong_prefix_is_contained() {
        let witness = [1; 16];
        let history = FakeHistory::with_heads(witness, &[(1, 0x11), (2, 0x99), (3, 0x33)]);
        let anchor = external(witness, 2, Digest32([0x22; 32]));
        let relation = classify_witness_frontier_recovery_v1(&history, witness, Some(&anchor)).unwrap();
        assert!(matches!(
            relation,
            WitnessFrontierRecoveryRelationV1::DivergentTrustedPrefix {
                trusted_high_watermark: 2,
                local_high_watermark: 3
            }
        ));
        assert_eq!(
            relation.publication_disposition(),
            WitnessFrontierPublicationDispositionV1::Contained
        );
    }

    #[test]
    fn external_ahead_is_rollback_and_contained() {
        let witness = [1; 16];
        let history = FakeHistory::with_heads(witness, &[(1, 0x11)]);
        let anchor = external(witness, 2, Digest32([0x22; 32]));
        let relation = classify_witness_frontier_recovery_v1(&history, witness, Some(&anchor)).unwrap();
        assert!(matches!(
            relation,
            WitnessFrontierRecoveryRelationV1::RollbackOrMissingLocal {
                trusted_high_watermark: 2,
                local_high_watermark: Some(1)
            }
        ));
        assert_eq!(
            relation.publication_disposition(),
            WitnessFrontierPublicationDispositionV1::Contained
        );
    }

    #[test]
    fn same_height_different_head_is_contained() {
        let witness = [1; 16];
        let history = FakeHistory::with_heads(witness, &[(1, 0x11), (2, 0x99)]);
        let anchor = external(witness, 2, Digest32([0x22; 32]));
        let relation = classify_witness_frontier_recovery_v1(&history, witness, Some(&anchor)).unwrap();
        assert!(matches!(
            relation,
            WitnessFrontierRecoveryRelationV1::DivergentAtSameHeight { high_watermark: 2 }
        ));
    }

    #[test]
    fn missing_local_with_external_anchor_is_contained() {
        let witness = [1; 16];
        let history = FakeHistory::default();
        let anchor = external(witness, 1, Digest32([0x11; 32]));
        let relation = classify_witness_frontier_recovery_v1(&history, witness, Some(&anchor)).unwrap();
        assert!(matches!(
            relation,
            WitnessFrontierRecoveryRelationV1::RollbackOrMissingLocal {
                trusted_high_watermark: 1,
                local_high_watermark: None
            }
        ));
    }

    #[test]
    fn no_external_anchor_requires_anchor_before_publication() {
        let witness = [1; 16];
        let history = FakeHistory::with_heads(witness, &[(1, 0x11)]);
        let relation = classify_witness_frontier_recovery_v1(&history, witness, None).unwrap();
        assert!(matches!(
            relation,
            WitnessFrontierRecoveryRelationV1::InitialAnchorRequired {
                local_high_watermark: 1,
                ..
            }
        ));
        assert_eq!(
            relation.publication_disposition(),
            WitnessFrontierPublicationDispositionV1::AnchorRequired
        );
    }

    #[test]
    fn external_claim_must_be_structurally_consistent_before_source_verification() {
        let witness = [1; 16];
        let mut claim = ExternalWitnessFrontierClaimV1 {
            schema_version: EXTERNAL_ANCHOR_SCHEMA_VERSION,
            source_id: [0xa1; 16],
            source_epoch: 1,
            source_sequence: 1,
            witness_id: witness,
            high_watermark: 1,
            reservation_head: Digest32([0x11; 32]),
            frontier_statement_digest: frontier_statement_digest(witness, 1, Digest32([0x11; 32])),
            freshness_evidence_digest: Digest32([0xee; 32]),
        };
        claim.reservation_head = Digest32([0x99; 32]);
        assert!(matches!(
            verify_external_witness_frontier_v1(claim, &AcceptExternal),
            Err(FrontierRecoveryError::MalformedExternalAnchor)
        ));
    }
}
