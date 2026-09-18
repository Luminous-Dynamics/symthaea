// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Passive multi-epoch revision-history chain.
//!
//! EKM-066 restores the complete immutable legacy V1 audit. EKM-077 defines one
//! passive V2 receipt segment per authority epoch. This module binds the archival
//! prefix and ordered epoch segments into one globally contiguous audit lineage.
//! It does not construct `BeliefRevisionHistory` or grant mutation authority.

use crate::knowledge::belief_revision_receipt::BeliefRevisionReceiptId;
use crate::knowledge::epistemic_restart_continuity::epoch_bound_receipt_segment::{
    EpochBoundRevisionReceiptSegmentError, EpochBoundRevisionReceiptSegmentV2,
};
use crate::knowledge::epistemic_restart_revision_audit_restoration::ImmutableRevisionAuditRestorationV1;
use std::error::Error;
use std::fmt;

const MAX_EPOCH_SEGMENTS: usize = 1_000_000;
const MAX_LEGACY_RECEIPTS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiEpochRevisionHistoryChainVersion {
    V2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultiEpochRevisionHistoryChainDigestV2([u8; 32]);

impl MultiEpochRevisionHistoryChainDigestV2 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiEpochRevisionHistoryChainV2 {
    version: MultiEpochRevisionHistoryChainVersion,
    captured_at_cycle: u64,
    legacy_audit_restoration_digest: [u8; 32],
    legacy_receipt_count: u64,
    legacy_next_receipt_id: BeliefRevisionReceiptId,
    segments: Vec<EpochBoundRevisionReceiptSegmentV2>,
    final_next_receipt_id: BeliefRevisionReceiptId,
    receipt_cursor_continuity_proven: bool,
    epoch_sequence_continuity_proven: bool,
    epoch_issuance_chain_verified: bool,
    operational_history_constructed: bool,
    mutation_authority: bool,
    activation_authorized: bool,
    chain_digest: MultiEpochRevisionHistoryChainDigestV2,
}

impl MultiEpochRevisionHistoryChainV2 {
    /// Bind the complete immutable legacy audit to ordered passive V2 epoch segments.
    ///
    /// This is an audit-lineage operation only. It does not turn any receipt or
    /// segment into operational mutation authority.
    pub fn capture(
        legacy_audit: &ImmutableRevisionAuditRestorationV1,
        segments: Vec<EpochBoundRevisionReceiptSegmentV2>,
        captured_at_cycle: u64,
    ) -> Result<Self, MultiEpochRevisionHistoryChainError> {
        if !legacy_audit.complete_immutable_revision_audit_restored()
            || legacy_audit.operational_revision_history_constructed()
            || legacy_audit.mutation_authority()
            || legacy_audit.writable_hydration_authorized()
            || legacy_audit.writable_state_export_authorized()
            || legacy_audit.activation_authorized()
        {
            return Err(MultiEpochRevisionHistoryChainError::UnexpectedLegacyAuditAuthority);
        }
        if captured_at_cycle < legacy_audit.restored_at_cycle() {
            return Err(MultiEpochRevisionHistoryChainError::CapturePredatesLegacyRestoration {
                captured_at_cycle,
                restored_at_cycle: legacy_audit.restored_at_cycle(),
            });
        }
        if legacy_audit.records().len() > MAX_LEGACY_RECEIPTS {
            return Err(MultiEpochRevisionHistoryChainError::TooManyLegacyReceipts {
                actual: legacy_audit.records().len(),
                maximum: MAX_LEGACY_RECEIPTS,
            });
        }
        if segments.len() > MAX_EPOCH_SEGMENTS {
            return Err(MultiEpochRevisionHistoryChainError::TooManyEpochSegments {
                actual: segments.len(),
                maximum: MAX_EPOCH_SEGMENTS,
            });
        }

        let legacy_count = u64::try_from(legacy_audit.records().len())
            .map_err(|_| MultiEpochRevisionHistoryChainError::LengthOverflow)?;
        for (index, record) in legacy_audit.records().iter().enumerate() {
            let index = u64::try_from(index)
                .map_err(|_| MultiEpochRevisionHistoryChainError::LengthOverflow)?;
            let expected = index
                .checked_add(1)
                .ok_or(MultiEpochRevisionHistoryChainError::ReceiptIdOverflow)?;
            if record.receipt_id().0 != expected {
                return Err(MultiEpochRevisionHistoryChainError::LegacyReceiptIdDiscontinuity {
                    expected: BeliefRevisionReceiptId(expected),
                    actual: record.receipt_id(),
                });
            }
        }
        let legacy_next = BeliefRevisionReceiptId(
            legacy_count
                .checked_add(1)
                .ok_or(MultiEpochRevisionHistoryChainError::ReceiptIdOverflow)?,
        );

        let mut cursor = legacy_next;
        let mut expected_epoch_sequence = 1u64;
        let mut previous_segment_capture_cycle = None;
        for segment in &segments {
            segment
                .verify()
                .map_err(MultiEpochRevisionHistoryChainError::SegmentInvalid)?;
            if segment.first_receipt_id() != cursor {
                return Err(MultiEpochRevisionHistoryChainError::SegmentReceiptCursorMismatch {
                    expected: cursor,
                    actual: segment.first_receipt_id(),
                });
            }
            if segment.authority_epoch_sequence() != expected_epoch_sequence {
                return Err(MultiEpochRevisionHistoryChainError::EpochSequenceDiscontinuity {
                    expected: expected_epoch_sequence,
                    actual: segment.authority_epoch_sequence(),
                });
            }
            if segment.captured_at_cycle() > captured_at_cycle {
                return Err(MultiEpochRevisionHistoryChainError::SegmentAfterChainCapture {
                    segment_cycle: segment.captured_at_cycle(),
                    chain_cycle: captured_at_cycle,
                });
            }
            if let Some(previous) = previous_segment_capture_cycle {
                if segment.captured_at_cycle() < previous {
                    return Err(MultiEpochRevisionHistoryChainError::SegmentCaptureCycleRegression {
                        previous,
                        current: segment.captured_at_cycle(),
                    });
                }
            }
            previous_segment_capture_cycle = Some(segment.captured_at_cycle());
            cursor = segment.next_receipt_id();
            expected_epoch_sequence = expected_epoch_sequence
                .checked_add(1)
                .ok_or(MultiEpochRevisionHistoryChainError::EpochSequenceOverflow)?;
        }

        let mut out = Self {
            version: MultiEpochRevisionHistoryChainVersion::V2,
            captured_at_cycle,
            legacy_audit_restoration_digest: legacy_audit.restoration_digest().as_bytes(),
            legacy_receipt_count: legacy_count,
            legacy_next_receipt_id: legacy_next,
            segments,
            final_next_receipt_id: cursor,
            receipt_cursor_continuity_proven: true,
            epoch_sequence_continuity_proven: true,
            // Segment sequence/digest continuity is structural. Actual epoch issuance
            // must later be proven from successful activation-commit receipts.
            epoch_issuance_chain_verified: false,
            operational_history_constructed: false,
            mutation_authority: false,
            activation_authorized: false,
            chain_digest: MultiEpochRevisionHistoryChainDigestV2([0; 32]),
        };
        out.chain_digest = digest_chain(&out)?;
        Ok(out)
    }

    pub fn version(&self) -> MultiEpochRevisionHistoryChainVersion {
        self.version
    }
    pub fn captured_at_cycle(&self) -> u64 {
        self.captured_at_cycle
    }
    pub fn legacy_audit_restoration_digest(&self) -> [u8; 32] {
        self.legacy_audit_restoration_digest
    }
    pub fn legacy_receipt_count(&self) -> u64 {
        self.legacy_receipt_count
    }
    pub fn legacy_next_receipt_id(&self) -> BeliefRevisionReceiptId {
        self.legacy_next_receipt_id
    }
    pub fn segments(&self) -> &[EpochBoundRevisionReceiptSegmentV2] {
        &self.segments
    }
    pub fn final_next_receipt_id(&self) -> BeliefRevisionReceiptId {
        self.final_next_receipt_id
    }
    pub fn receipt_cursor_continuity_proven(&self) -> bool {
        self.receipt_cursor_continuity_proven
    }
    pub fn epoch_sequence_continuity_proven(&self) -> bool {
        self.epoch_sequence_continuity_proven
    }
    pub fn epoch_issuance_chain_verified(&self) -> bool {
        self.epoch_issuance_chain_verified
    }
    pub fn operational_history_constructed(&self) -> bool {
        self.operational_history_constructed
    }
    pub fn mutation_authority(&self) -> bool {
        self.mutation_authority
    }
    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
    pub fn chain_digest(&self) -> MultiEpochRevisionHistoryChainDigestV2 {
        self.chain_digest
    }

    pub fn verify_against(
        &self,
        legacy_audit: &ImmutableRevisionAuditRestorationV1,
    ) -> Result<(), MultiEpochRevisionHistoryChainError> {
        let live = Self::capture(legacy_audit, self.segments.clone(), self.captured_at_cycle)?;
        if &live != self {
            return Err(MultiEpochRevisionHistoryChainError::ChainMismatch);
        }
        if digest_chain(self)? != self.chain_digest {
            return Err(MultiEpochRevisionHistoryChainError::ChainDigestMismatch);
        }
        Ok(())
    }
}

fn digest_chain(
    chain: &MultiEpochRevisionHistoryChainV2,
) -> Result<MultiEpochRevisionHistoryChainDigestV2, MultiEpochRevisionHistoryChainError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-multi-epoch-revision-history-chain-v2");
    hasher.update(&[2]);
    hasher.update(&chain.captured_at_cycle.to_le_bytes());
    hasher.update(&chain.legacy_audit_restoration_digest);
    hasher.update(&chain.legacy_receipt_count.to_le_bytes());
    hasher.update(&chain.legacy_next_receipt_id.0.to_le_bytes());
    let count = u64::try_from(chain.segments.len())
        .map_err(|_| MultiEpochRevisionHistoryChainError::LengthOverflow)?;
    hasher.update(&count.to_le_bytes());
    for segment in &chain.segments {
        hasher.update(&segment.segment_digest().as_bytes());
    }
    hasher.update(&chain.final_next_receipt_id.0.to_le_bytes());
    for value in [
        chain.receipt_cursor_continuity_proven,
        chain.epoch_sequence_continuity_proven,
        chain.epoch_issuance_chain_verified,
        chain.operational_history_constructed,
        chain.mutation_authority,
        chain.activation_authorized,
    ] {
        hasher.update(&[u8::from(value)]);
    }
    Ok(MultiEpochRevisionHistoryChainDigestV2(
        *hasher.finalize().as_bytes(),
    ))
}

#[derive(Debug, Clone, PartialEq)]
pub enum MultiEpochRevisionHistoryChainError {
    UnexpectedLegacyAuditAuthority,
    CapturePredatesLegacyRestoration {
        captured_at_cycle: u64,
        restored_at_cycle: u64,
    },
    TooManyLegacyReceipts { actual: usize, maximum: usize },
    TooManyEpochSegments { actual: usize, maximum: usize },
    LengthOverflow,
    ReceiptIdOverflow,
    EpochSequenceOverflow,
    LegacyReceiptIdDiscontinuity {
        expected: BeliefRevisionReceiptId,
        actual: BeliefRevisionReceiptId,
    },
    SegmentInvalid(EpochBoundRevisionReceiptSegmentError),
    SegmentReceiptCursorMismatch {
        expected: BeliefRevisionReceiptId,
        actual: BeliefRevisionReceiptId,
    },
    EpochSequenceDiscontinuity { expected: u64, actual: u64 },
    SegmentAfterChainCapture { segment_cycle: u64, chain_cycle: u64 },
    SegmentCaptureCycleRegression { previous: u64, current: u64 },
    ChainMismatch,
    ChainDigestMismatch,
}

impl fmt::Display for MultiEpochRevisionHistoryChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "multi-epoch revision history chain invalid: {self:?}")
    }
}

impl Error for MultiEpochRevisionHistoryChainError {}
