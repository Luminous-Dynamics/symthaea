// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Passive persistence segment for future epoch-bound revision receipts.
//!
//! EKM-076 defines one immutable V2 receipt. This module defines how receipts from
//! one authority epoch must be grouped and chained without constructing an
//! operational history or exporting mutation authority.

#[path = "epistemic_restart_multi_epoch_revision_chain.rs"]
pub mod multi_epoch_history_chain;

use crate::knowledge::belief_revision_receipt::BeliefRevisionReceiptId;
use crate::knowledge::epistemic_restart_continuity::epoch_bound_receipt_data::{
    EpochBoundRevisionReceiptDataError, EpochBoundRevisionReceiptDataV2,
};
use std::error::Error;
use std::fmt;

const MAX_SEGMENT_RECORDS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochBoundRevisionReceiptSegmentVersion {
    V2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpochBoundRevisionReceiptSegmentDigestV2([u8; 32]);

impl EpochBoundRevisionReceiptSegmentDigestV2 {
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

/// One append-only receipt segment belonging to exactly one authority epoch.
///
/// EKM-077 intentionally provides no production constructor. A later operational
/// history layer may create a segment only after EKM-076 receipts can be produced
/// by a fresh evaluation under a committed epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochBoundRevisionReceiptSegmentV2 {
    version: EpochBoundRevisionReceiptSegmentVersion,
    authority_epoch_digest: [u8; 32],
    authority_epoch_sequence: u64,
    first_receipt_id: BeliefRevisionReceiptId,
    next_receipt_id: BeliefRevisionReceiptId,
    captured_at_cycle: u64,
    records: Vec<EpochBoundRevisionReceiptDataV2>,
    segment_digest: EpochBoundRevisionReceiptSegmentDigestV2,
}

impl EpochBoundRevisionReceiptSegmentV2 {
    pub fn version(&self) -> EpochBoundRevisionReceiptSegmentVersion {
        self.version
    }

    pub fn authority_epoch_digest(&self) -> [u8; 32] {
        self.authority_epoch_digest
    }

    pub fn authority_epoch_sequence(&self) -> u64 {
        self.authority_epoch_sequence
    }

    pub fn first_receipt_id(&self) -> BeliefRevisionReceiptId {
        self.first_receipt_id
    }

    pub fn next_receipt_id(&self) -> BeliefRevisionReceiptId {
        self.next_receipt_id
    }

    pub fn captured_at_cycle(&self) -> u64 {
        self.captured_at_cycle
    }

    pub fn records(&self) -> &[EpochBoundRevisionReceiptDataV2] {
        &self.records
    }

    pub fn segment_digest(&self) -> EpochBoundRevisionReceiptSegmentDigestV2 {
        self.segment_digest
    }

    pub fn operational_history_constructed(&self) -> bool {
        false
    }

    pub fn authority_epoch_issuance_verified(&self) -> bool {
        false
    }

    pub fn mutation_authority(&self) -> bool {
        false
    }

    pub fn activation_authorized(&self) -> bool {
        false
    }

    pub fn verify(&self) -> Result<(), EpochBoundRevisionReceiptSegmentError> {
        validate_segment(self)?;
        if digest_segment(self)? != self.segment_digest {
            return Err(EpochBoundRevisionReceiptSegmentError::SegmentDigestMismatch);
        }
        Ok(())
    }
}

fn validate_segment(
    segment: &EpochBoundRevisionReceiptSegmentV2,
) -> Result<(), EpochBoundRevisionReceiptSegmentError> {
    if segment.version != EpochBoundRevisionReceiptSegmentVersion::V2 {
        return Err(EpochBoundRevisionReceiptSegmentError::UnsupportedVersion);
    }
    if segment.authority_epoch_sequence == 0 {
        return Err(EpochBoundRevisionReceiptSegmentError::InvalidAuthorityEpochSequence);
    }
    if segment.first_receipt_id.0 == 0 || segment.next_receipt_id.0 == 0 {
        return Err(EpochBoundRevisionReceiptSegmentError::InvalidReceiptIdBoundary);
    }
    if segment.records.len() > MAX_SEGMENT_RECORDS {
        return Err(EpochBoundRevisionReceiptSegmentError::TooManyRecords {
            actual: segment.records.len(),
            maximum: MAX_SEGMENT_RECORDS,
        });
    }

    let count = u64::try_from(segment.records.len())
        .map_err(|_| EpochBoundRevisionReceiptSegmentError::LengthOverflow)?;
    let expected_next = segment
        .first_receipt_id
        .0
        .checked_add(count)
        .ok_or(EpochBoundRevisionReceiptSegmentError::ReceiptIdOverflow)?;
    if segment.next_receipt_id.0 != expected_next {
        return Err(EpochBoundRevisionReceiptSegmentError::NextReceiptIdMismatch {
            expected: BeliefRevisionReceiptId(expected_next),
            actual: segment.next_receipt_id,
        });
    }

    let mut previous_cycle = None;
    for (offset, record) in segment.records.iter().enumerate() {
        record
            .verify()
            .map_err(EpochBoundRevisionReceiptSegmentError::RecordInvalid)?;
        if record.authority_epoch_digest() != segment.authority_epoch_digest
            || record.authority_epoch_sequence() != segment.authority_epoch_sequence
        {
            return Err(EpochBoundRevisionReceiptSegmentError::RecordEpochMismatch(
                record.receipt_id(),
            ));
        }

        let offset = u64::try_from(offset)
            .map_err(|_| EpochBoundRevisionReceiptSegmentError::LengthOverflow)?;
        let expected_id = segment
            .first_receipt_id
            .0
            .checked_add(offset)
            .ok_or(EpochBoundRevisionReceiptSegmentError::ReceiptIdOverflow)?;
        if record.receipt_id().0 != expected_id {
            return Err(EpochBoundRevisionReceiptSegmentError::NonContiguousReceiptId {
                expected: BeliefRevisionReceiptId(expected_id),
                actual: record.receipt_id(),
            });
        }
        if record.evaluated_at_cycle() > segment.captured_at_cycle {
            return Err(EpochBoundRevisionReceiptSegmentError::RecordAfterCapture {
                receipt_id: record.receipt_id(),
                evaluated_at_cycle: record.evaluated_at_cycle(),
                captured_at_cycle: segment.captured_at_cycle,
            });
        }
        if let Some(previous) = previous_cycle {
            if record.evaluated_at_cycle() < previous {
                return Err(EpochBoundRevisionReceiptSegmentError::EvaluationCycleRegression {
                    previous,
                    current: record.evaluated_at_cycle(),
                });
            }
        }
        previous_cycle = Some(record.evaluated_at_cycle());
    }

    Ok(())
}

fn digest_segment(
    segment: &EpochBoundRevisionReceiptSegmentV2,
) -> Result<EpochBoundRevisionReceiptSegmentDigestV2, EpochBoundRevisionReceiptSegmentError> {
    validate_segment(segment)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-epoch-bound-revision-receipt-segment-v2");
    hasher.update(&[2]);
    hasher.update(&segment.authority_epoch_digest);
    hasher.update(&segment.authority_epoch_sequence.to_le_bytes());
    hasher.update(&segment.first_receipt_id.0.to_le_bytes());
    hasher.update(&segment.next_receipt_id.0.to_le_bytes());
    hasher.update(&segment.captured_at_cycle.to_le_bytes());
    let count = u64::try_from(segment.records.len())
        .map_err(|_| EpochBoundRevisionReceiptSegmentError::LengthOverflow)?;
    hasher.update(&count.to_le_bytes());
    for record in &segment.records {
        hasher.update(&record.receipt_digest().as_bytes());
    }
    Ok(EpochBoundRevisionReceiptSegmentDigestV2(
        *hasher.finalize().as_bytes(),
    ))
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpochBoundRevisionReceiptSegmentError {
    UnsupportedVersion,
    InvalidAuthorityEpochSequence,
    InvalidReceiptIdBoundary,
    TooManyRecords { actual: usize, maximum: usize },
    LengthOverflow,
    ReceiptIdOverflow,
    NextReceiptIdMismatch {
        expected: BeliefRevisionReceiptId,
        actual: BeliefRevisionReceiptId,
    },
    RecordInvalid(EpochBoundRevisionReceiptDataError),
    RecordEpochMismatch(BeliefRevisionReceiptId),
    NonContiguousReceiptId {
        expected: BeliefRevisionReceiptId,
        actual: BeliefRevisionReceiptId,
    },
    RecordAfterCapture {
        receipt_id: BeliefRevisionReceiptId,
        evaluated_at_cycle: u64,
        captured_at_cycle: u64,
    },
    EvaluationCycleRegression { previous: u64, current: u64 },
    SegmentDigestMismatch,
}

impl fmt::Display for EpochBoundRevisionReceiptSegmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epoch-bound revision receipt segment invalid: {self:?}")
    }
}

impl Error for EpochBoundRevisionReceiptSegmentError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_segment() -> EpochBoundRevisionReceiptSegmentV2 {
        let mut out = EpochBoundRevisionReceiptSegmentV2 {
            version: EpochBoundRevisionReceiptSegmentVersion::V2,
            authority_epoch_digest: [7; 32],
            authority_epoch_sequence: 2,
            first_receipt_id: BeliefRevisionReceiptId(41),
            next_receipt_id: BeliefRevisionReceiptId(41),
            captured_at_cycle: 100,
            records: vec![],
            segment_digest: EpochBoundRevisionReceiptSegmentDigestV2([0; 32]),
        };
        out.segment_digest = digest_segment(&out).unwrap();
        out
    }

    #[test]
    fn empty_epoch_segment_is_well_formed_but_non_authoritative() {
        let segment = empty_segment();
        segment.verify().unwrap();
        assert_eq!(segment.first_receipt_id(), segment.next_receipt_id());
        assert!(!segment.operational_history_constructed());
        assert!(!segment.authority_epoch_issuance_verified());
        assert!(!segment.mutation_authority());
        assert!(!segment.activation_authorized());
    }

    #[test]
    fn next_receipt_id_must_match_segment_length() {
        let mut segment = empty_segment();
        segment.next_receipt_id = BeliefRevisionReceiptId(42);
        assert!(matches!(
            segment.verify(),
            Err(EpochBoundRevisionReceiptSegmentError::NextReceiptIdMismatch { .. })
        ));
    }
}
