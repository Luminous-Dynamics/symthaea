// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compact external-attestation checkpoints for the Golden qualification ledger.
//!
//! A local hash chain detects accidental/internal inconsistency but is not, by itself,
//! an external trust anchor: an actor able to rewrite the entire ledger could recompute
//! local hashes. This module emits a compact, domain-separated checkpoint suitable for
//! signing or anchoring by Xenia/operator/trusted storage without making that checkpoint
//! execution authority or copying private benchmark artifacts into it.

use crate::golden_binding_ledger::{
    GoldenQualificationBindingLedgerV1, GoldenQualificationLedgerErrorV1,
    GOLDEN_QUALIFICATION_LEDGER_SCHEMA_V1,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

pub const GOLDEN_QUALIFICATION_CHECKPOINT_SCHEMA_V1: &str =
    "symthaea-golden-qualification-checkpoint-v1";
pub const GOLDEN_QUALIFICATION_ATTESTATION_DOMAIN_V1: &str =
    "symthaea-golden-qualification-checkpoint-attestation-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenQualificationCheckpointV1 {
    pub schema_version: String,
    pub ledger_schema_version: String,
    pub entry_count: usize,
    pub ledger_digest: String,
    pub head_entry_digest: Option<String>,
    /// Digest of the previous checkpoint object, if this checkpoint extends a
    /// checkpoint chain. This is not the ledger-entry predecessor digest.
    pub previous_checkpoint_digest: Option<String>,
    pub checkpointed_at_unix_ms: u64,
}

impl GoldenQualificationCheckpointV1 {
    pub fn from_ledger(
        ledger: &GoldenQualificationBindingLedgerV1,
        previous_checkpoint_digest: Option<String>,
        checkpointed_at_unix_ms: u64,
    ) -> Result<Self, GoldenCheckpointErrorV1> {
        ledger.validate_chain()?;
        if ledger.schema_version != GOLDEN_QUALIFICATION_LEDGER_SCHEMA_V1 {
            return Err(GoldenCheckpointErrorV1::LedgerSchemaMismatch);
        }
        if let Some(previous) = &previous_checkpoint_digest {
            validate_hex_digest(previous, "previous checkpoint digest")?;
        }
        if let Some(last) = ledger.entries().last() {
            if checkpointed_at_unix_ms < last.recorded_at_unix_ms {
                return Err(GoldenCheckpointErrorV1::CheckpointPredatesLedgerHead);
            }
        }

        Ok(Self {
            schema_version: GOLDEN_QUALIFICATION_CHECKPOINT_SCHEMA_V1.into(),
            ledger_schema_version: ledger.schema_version.clone(),
            entry_count: ledger.entries().len(),
            ledger_digest: ledger.digest()?,
            head_entry_digest: ledger.head_digest()?,
            previous_checkpoint_digest,
            checkpointed_at_unix_ms,
        })
    }

    pub fn successor_from_ledger(
        previous: &Self,
        ledger: &GoldenQualificationBindingLedgerV1,
        checkpointed_at_unix_ms: u64,
    ) -> Result<Self, GoldenCheckpointErrorV1> {
        previous.validate_shape()?;
        if checkpointed_at_unix_ms < previous.checkpointed_at_unix_ms {
            return Err(GoldenCheckpointErrorV1::CheckpointTimeRegressed);
        }
        if ledger.entries().len() < previous.entry_count {
            return Err(GoldenCheckpointErrorV1::LedgerEntryCountRegressed {
                previous: previous.entry_count,
                current: ledger.entries().len(),
            });
        }
        Self::from_ledger(
            ledger,
            Some(previous.digest()?),
            checkpointed_at_unix_ms,
        )
    }

    pub fn validate_shape(&self) -> Result<(), GoldenCheckpointErrorV1> {
        if self.schema_version != GOLDEN_QUALIFICATION_CHECKPOINT_SCHEMA_V1 {
            return Err(GoldenCheckpointErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.ledger_schema_version != GOLDEN_QUALIFICATION_LEDGER_SCHEMA_V1 {
            return Err(GoldenCheckpointErrorV1::LedgerSchemaMismatch);
        }
        validate_hex_digest(&self.ledger_digest, "ledger digest")?;
        if let Some(head) = &self.head_entry_digest {
            validate_hex_digest(head, "head entry digest")?;
        }
        if let Some(previous) = &self.previous_checkpoint_digest {
            validate_hex_digest(previous, "previous checkpoint digest")?;
        }
        if self.entry_count == 0 && self.head_entry_digest.is_some() {
            return Err(GoldenCheckpointErrorV1::UnexpectedHeadForEmptyLedger);
        }
        if self.entry_count > 0 && self.head_entry_digest.is_none() {
            return Err(GoldenCheckpointErrorV1::MissingHeadForNonEmptyLedger);
        }
        Ok(())
    }

    pub fn verify_against_ledger(
        &self,
        ledger: &GoldenQualificationBindingLedgerV1,
    ) -> Result<(), GoldenCheckpointErrorV1> {
        self.validate_shape()?;
        ledger.validate_chain()?;
        if self.ledger_schema_version != ledger.schema_version {
            return Err(GoldenCheckpointErrorV1::LedgerSchemaMismatch);
        }
        if self.entry_count != ledger.entries().len() {
            return Err(GoldenCheckpointErrorV1::EntryCountMismatch {
                checkpoint: self.entry_count,
                ledger: ledger.entries().len(),
            });
        }
        if self.ledger_digest != ledger.digest()? {
            return Err(GoldenCheckpointErrorV1::LedgerDigestMismatch);
        }
        if self.head_entry_digest != ledger.head_digest()? {
            return Err(GoldenCheckpointErrorV1::HeadDigestMismatch);
        }
        if let Some(last) = ledger.entries().last() {
            if self.checkpointed_at_unix_ms < last.recorded_at_unix_ms {
                return Err(GoldenCheckpointErrorV1::CheckpointPredatesLedgerHead);
            }
        }
        Ok(())
    }

    /// Domain-separated digest of the checkpoint object itself.
    pub fn digest(&self) -> Result<String, GoldenCheckpointErrorV1> {
        self.validate_shape()?;
        digest_serializable("symthaea-golden-qualification-checkpoint-object-v1", self)
    }

    /// Small deterministic signing/attestation preimage. External code may sign or
    /// anchor these bytes, but this crate deliberately does not define signer trust,
    /// signature algorithms, key custody or verification authority.
    pub fn attestation_preimage(&self) -> Result<Vec<u8>, GoldenCheckpointErrorV1> {
        let checkpoint_digest = self.digest()?;
        serde_json::to_vec(&(
            GOLDEN_QUALIFICATION_ATTESTATION_DOMAIN_V1,
            checkpoint_digest,
        ))
        .map_err(|err| GoldenCheckpointErrorV1::Serialization(err.to_string()))
    }
}

fn digest_serializable<T: Serialize + ?Sized>(
    domain: &'static str,
    value: &T,
) -> Result<String, GoldenCheckpointErrorV1> {
    let bytes = serde_json::to_vec(&(domain, value))
        .map_err(|err| GoldenCheckpointErrorV1::Serialization(err.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn validate_hex_digest(value: &str, field: &'static str) -> Result<(), GoldenCheckpointErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(GoldenCheckpointErrorV1::InvalidDigest(field))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum GoldenCheckpointErrorV1 {
    Ledger(GoldenQualificationLedgerErrorV1),
    UnsupportedSchema(String),
    LedgerSchemaMismatch,
    InvalidDigest(&'static str),
    Serialization(String),
    CheckpointPredatesLedgerHead,
    CheckpointTimeRegressed,
    LedgerEntryCountRegressed { previous: usize, current: usize },
    UnexpectedHeadForEmptyLedger,
    MissingHeadForNonEmptyLedger,
    EntryCountMismatch { checkpoint: usize, ledger: usize },
    LedgerDigestMismatch,
    HeadDigestMismatch,
}

impl fmt::Display for GoldenCheckpointErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ledger(err) => write!(f, "golden qualification ledger invalid: {err}"),
            Self::UnsupportedSchema(value) => write!(f, "unsupported checkpoint schema {value}"),
            Self::LedgerSchemaMismatch => write!(f, "qualification ledger schema mismatch"),
            Self::InvalidDigest(field) => write!(f, "invalid 32-byte hex digest for {field}"),
            Self::Serialization(message) => write!(f, "checkpoint serialization failed: {message}"),
            Self::CheckpointPredatesLedgerHead => write!(f, "checkpoint predates ledger head"),
            Self::CheckpointTimeRegressed => write!(f, "checkpoint timestamp regressed"),
            Self::LedgerEntryCountRegressed { previous, current } => write!(
                f,
                "ledger entry count regressed from {previous} to {current}"
            ),
            Self::UnexpectedHeadForEmptyLedger => write!(f, "empty ledger checkpoint has a head digest"),
            Self::MissingHeadForNonEmptyLedger => write!(f, "non-empty ledger checkpoint lacks a head digest"),
            Self::EntryCountMismatch { checkpoint, ledger } => write!(
                f,
                "checkpoint entry count {checkpoint} does not match ledger {ledger}"
            ),
            Self::LedgerDigestMismatch => write!(f, "checkpoint ledger digest mismatch"),
            Self::HeadDigestMismatch => write!(f, "checkpoint ledger-head digest mismatch"),
        }
    }
}

impl Error for GoldenCheckpointErrorV1 {}

impl From<GoldenQualificationLedgerErrorV1> for GoldenCheckpointErrorV1 {
    fn from(value: GoldenQualificationLedgerErrorV1) -> Self {
        Self::Ledger(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden_binding_ledger::GoldenQualificationBindingLedgerV1;
    use crate::golden_bound_result::GoldenBoundQualificationResultV1;
    use crate::golden_qualification_binding::GoldenQualificationBindingV1;
    use crate::it_qualification::{
        QualificationCaseIdV1, QualificationCaseKeyV1, QualificationResultIdV1,
        QualificationRunIdV1,
    };

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn binding(suffix: &str, evaluated_at: u64) -> GoldenBoundQualificationResultV1 {
        GoldenBoundQualificationResultV1 {
            lineage: GoldenQualificationBindingV1 {
                result_id: QualificationResultIdV1(format!("result-{suffix}")),
                run_id: QualificationRunIdV1(format!("run-{suffix}")),
                case_key: QualificationCaseKeyV1 {
                    id: QualificationCaseIdV1("case-1".into()),
                    revision: 1,
                },
                corpus_digest: digest('a'),
                case_digest: digest('b'),
                grading_artifact_digest: digest('c'),
                run_context_digest: digest('d'),
                metrics_digest: digest('e'),
            },
            private_evaluation_digest: digest('f'),
            derived_metrics_digest: digest('e'),
            evaluated_at_unix_ms: evaluated_at,
        }
    }

    #[test]
    fn empty_ledger_checkpoint_is_well_formed_and_verifiable() {
        let ledger = GoldenQualificationBindingLedgerV1::new();
        let checkpoint = GoldenQualificationCheckpointV1::from_ledger(&ledger, None, 100).unwrap();
        checkpoint.verify_against_ledger(&ledger).unwrap();
        assert_eq!(checkpoint.entry_count, 0);
        assert!(checkpoint.head_entry_digest.is_none());
        assert_eq!(checkpoint.digest().unwrap().len(), 64);
        assert!(!checkpoint.attestation_preimage().unwrap().is_empty());
    }

    #[test]
    fn checkpoint_detects_ledger_growth_until_successor_is_created() {
        let mut ledger = GoldenQualificationBindingLedgerV1::new();
        ledger.append(binding("a", 100), 110).unwrap();
        let first = GoldenQualificationCheckpointV1::from_ledger(&ledger, None, 120).unwrap();
        first.verify_against_ledger(&ledger).unwrap();

        ledger.append(binding("b", 130), 140).unwrap();
        assert!(matches!(
            first.verify_against_ledger(&ledger),
            Err(GoldenCheckpointErrorV1::EntryCountMismatch { .. })
        ));

        let second = GoldenQualificationCheckpointV1::successor_from_ledger(&first, &ledger, 150).unwrap();
        second.verify_against_ledger(&ledger).unwrap();
        assert_eq!(second.previous_checkpoint_digest, Some(first.digest().unwrap()));
        assert_eq!(second.entry_count, 2);
    }

    #[test]
    fn successor_rejects_ledger_rollback() {
        let mut ledger = GoldenQualificationBindingLedgerV1::new();
        ledger.append(binding("a", 100), 110).unwrap();
        let first = GoldenQualificationCheckpointV1::from_ledger(&ledger, None, 120).unwrap();

        let empty = GoldenQualificationBindingLedgerV1::new();
        assert!(matches!(
            GoldenQualificationCheckpointV1::successor_from_ledger(&first, &empty, 130),
            Err(GoldenCheckpointErrorV1::LedgerEntryCountRegressed { .. })
        ));
    }
}
