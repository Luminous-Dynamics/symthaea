// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-object verification for historical materials-extraction receipts.
//!
//! A receipt being internally well-formed is not sufficient evidence that it belongs
//! to the protocol/corpus bytes a benchmark is actually consuming. This crate binds
//! those objects again at the consumption boundary.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use symthaea_materials_historical_extraction::{
    CompactHistoricalCorpus, HistoricalExtractionError, HistoricalExtractionReceipt,
    OqmdExtractionProtocol,
};
use thiserror::Error;

/// Verified cross-object binding for one extraction receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExtractionBinding {
    /// Digest of the exact protocol used for verification.
    pub protocol_sha256: String,
    /// Digest of the exact compact corpus used for verification.
    pub compact_corpus_sha256: String,
    /// Digest of the exact extraction receipt.
    pub receipt_sha256: String,
    /// Number of records in the verified compact corpus.
    pub record_count: u64,
}

/// Revalidate an extraction receipt against the actual protocol and compact corpus.
///
/// This proves only cross-object identity/lineage consistency. It does not establish
/// that the historical dump was authentic, that the extractor was scientifically
/// correct, that benchmark targets are uncontaminated, or that a material is valid.
pub fn verify_extraction_binding(
    protocol: &OqmdExtractionProtocol,
    corpus: &CompactHistoricalCorpus,
    receipt: &HistoricalExtractionReceipt,
) -> Result<VerifiedExtractionBinding, ExtractionVerificationError> {
    protocol.validate()?;
    corpus.validate_for(protocol)?;
    let expected_protocol = protocol.protocol_sha256()?;
    let expected_corpus = corpus.corpus_sha256(protocol)?;
    let expected_record_count = u64::try_from(corpus.records.len())
        .map_err(|_| ExtractionVerificationError::RecordCountOverflow)?;

    // Revalidate all receipt fields/digest shapes before comparing lineage.
    let receipt_sha = receipt.receipt_sha256()?;

    if !receipt.protocol_sha256.eq_ignore_ascii_case(&expected_protocol) {
        return Err(ExtractionVerificationError::ProtocolMismatch {
            expected: expected_protocol,
            observed: receipt.protocol_sha256.clone(),
        });
    }
    if !receipt
        .compact_corpus_sha256
        .eq_ignore_ascii_case(&expected_corpus)
    {
        return Err(ExtractionVerificationError::CorpusMismatch {
            expected: expected_corpus,
            observed: receipt.compact_corpus_sha256.clone(),
        });
    }
    if receipt.record_count != expected_record_count {
        return Err(ExtractionVerificationError::RecordCountMismatch {
            expected: expected_record_count,
            observed: receipt.record_count,
        });
    }

    Ok(VerifiedExtractionBinding {
        protocol_sha256: expected_protocol,
        compact_corpus_sha256: expected_corpus,
        receipt_sha256: receipt_sha,
        record_count: expected_record_count,
    })
}

/// Failures in cross-object historical extraction verification.
#[derive(Debug, Error)]
pub enum ExtractionVerificationError {
    /// Underlying protocol/corpus/receipt validation failed.
    #[error(transparent)]
    Historical(#[from] HistoricalExtractionError),
    /// Protocol identity in the receipt differs from the actual protocol.
    #[error("extraction receipt protocol mismatch: expected {expected}, observed {observed}")]
    ProtocolMismatch {
        /// Protocol digest computed from the supplied protocol.
        expected: String,
        /// Protocol digest carried by the receipt.
        observed: String,
    },
    /// Compact-corpus identity in the receipt differs from the actual corpus.
    #[error("extraction receipt corpus mismatch: expected {expected}, observed {observed}")]
    CorpusMismatch {
        /// Corpus digest computed from the supplied corpus.
        expected: String,
        /// Corpus digest carried by the receipt.
        observed: String,
    },
    /// Receipt record count differs from the actual compact corpus.
    #[error("extraction receipt record-count mismatch: expected {expected}, observed {observed}")]
    RecordCountMismatch {
        /// Actual corpus record count.
        expected: u64,
        /// Receipt-declared record count.
        observed: u64,
    },
    /// Platform cannot represent the in-memory record count as u64.
    #[error("compact corpus record count cannot be represented as u64")]
    RecordCountOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_materials_historical_extraction::{
        NormalizedOqmdRecord, bind_extraction_receipt, oqmd_v17_fe_co_zr_protocol,
    };

    fn sha(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn record(entry_id: u64) -> NormalizedOqmdRecord {
        NormalizedOqmdRecord {
            entry_id,
            name: format!("Fe-{entry_id}"),
            element_set: vec!["Fe".to_string()],
            composition_sha256: sha('a'),
            structure_sha256: Some(sha('b')),
            duplicate_entry_id: None,
            spacegroup: Some("Im-3m".to_string()),
            prototype: None,
            natoms: 1,
            ntypes: 1,
            delta_e_ev_atom: Some("0".to_string()),
            stability_ev_atom: Some("0".to_string()),
            band_gap_ev: None,
            calculation_label: Some("static".to_string()),
            fit: Some("standard".to_string()),
            icsd_id: None,
            property_condition_signature: "oqmd-v1.7|fit=standard|calculation=static".to_string(),
        }
    }

    fn fixture() -> (
        OqmdExtractionProtocol,
        CompactHistoricalCorpus,
        HistoricalExtractionReceipt,
    ) {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let corpus = CompactHistoricalCorpus::from_records(
            &protocol,
            vec![record(1), record(2)],
        )
        .unwrap();
        let receipt = bind_extraction_receipt(
            &protocol,
            &corpus,
            &sha('1'),
            &sha('2'),
            &sha('3'),
            &sha('4'),
            &sha('5'),
            &sha('6'),
            &sha('7'),
            &sha('8'),
            &sha('9'),
        )
        .unwrap();
        (protocol, corpus, receipt)
    }

    #[test]
    fn exact_protocol_corpus_and_receipt_bind() {
        let (protocol, corpus, receipt) = fixture();
        let verified = verify_extraction_binding(&protocol, &corpus, &receipt).unwrap();
        assert_eq!(verified.record_count, 2);
        assert_eq!(verified.protocol_sha256, protocol.protocol_sha256().unwrap());
        assert_eq!(
            verified.compact_corpus_sha256,
            corpus.corpus_sha256(&protocol).unwrap()
        );
        assert_eq!(verified.receipt_sha256, receipt.receipt_sha256().unwrap());
    }

    #[test]
    fn receipt_from_different_protocol_fails() {
        let (protocol, corpus, mut receipt) = fixture();
        receipt.protocol_sha256 = sha('c');
        assert!(matches!(
            verify_extraction_binding(&protocol, &corpus, &receipt),
            Err(ExtractionVerificationError::ProtocolMismatch { .. })
        ));
    }

    #[test]
    fn receipt_from_different_corpus_fails() {
        let (protocol, corpus, mut receipt) = fixture();
        receipt.compact_corpus_sha256 = sha('c');
        assert!(matches!(
            verify_extraction_binding(&protocol, &corpus, &receipt),
            Err(ExtractionVerificationError::CorpusMismatch { .. })
        ));
    }

    #[test]
    fn record_count_drift_fails() {
        let (protocol, corpus, mut receipt) = fixture();
        receipt.record_count = 3;
        assert!(matches!(
            verify_extraction_binding(&protocol, &corpus, &receipt),
            Err(ExtractionVerificationError::RecordCountMismatch {
                expected: 2,
                observed: 3
            })
        ));
    }

    #[test]
    fn mutated_corpus_fails_before_receipt_comparison() {
        let (protocol, mut corpus, receipt) = fixture();
        corpus.records.swap(0, 1);
        assert!(matches!(
            verify_extraction_binding(&protocol, &corpus, &receipt),
            Err(ExtractionVerificationError::Historical(
                HistoricalExtractionError::NonCanonicalRecordOrder
            ))
        ));
    }
}
