// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public protocol boundary for SYM-RSI-SEM-002.
//!
//! The raw benchmark implementation is crate-private. External callers receive a
//! receipt that cryptographically binds the frozen preregistration commit, exact
//! implementation subject, and inner benchmark evidence digest. This is provenance
//! and self-integrity only; authenticity still requires an independent evidence
//! lineage such as a frozen Git commit, signed manifest, or append-only ledger.

pub use super::sym_rsi_semantic_ambiguity_experiment::{
    Sem2AdmissionRates, Sem2DimensionMetrics, Sem2Disposition, Sem2ExperimentError,
    Sem2MarginThreshold, Sem2Receipt,
};
use super::sym_rsi_semantic_ambiguity_experiment::run_sym_rsi_sem_002;

pub const SYM_RSI_SEM_002_PREREGISTRATION_SHA: &str =
    "97e47ae4b95844ceeae857e9500de15871859422";
pub const SYM_RSI_SEM_002_PROTOCOL_RECEIPT_SCHEMA: &str =
    "symthaea.sym-rsi-sem-002.protocol-bound.v1";

#[derive(Debug, Clone, PartialEq)]
pub struct ProtocolBoundSem2Receipt {
    pub schema: &'static str,
    pub preregistration_sha: &'static str,
    pub implementation_subject_sha: String,
    pub measurement: Sem2Receipt,
    pub protocol_evidence_digest: [u8; 32],
}

impl ProtocolBoundSem2Receipt {
    pub fn protocol_evidence_digest_hex(&self) -> String {
        hex::encode(self.protocol_evidence_digest)
    }

    /// Verify the protocol/measurement self-binding.
    ///
    /// This detects local mutation or detachment. It does not authenticate the
    /// receipt's origin; an independent frozen/signed evidence lineage is still
    /// required for that stronger claim.
    pub fn validate(&self) -> Result<(), Sem2ProtocolError> {
        if self.schema != SYM_RSI_SEM_002_PROTOCOL_RECEIPT_SCHEMA {
            return Err(Sem2ProtocolError::SchemaMismatch);
        }
        if self.preregistration_sha != SYM_RSI_SEM_002_PREREGISTRATION_SHA {
            return Err(Sem2ProtocolError::PreregistrationMismatch);
        }
        if self.implementation_subject_sha != self.measurement.subject_sha {
            return Err(Sem2ProtocolError::SubjectMismatch);
        }
        let expected = protocol_digest(
            self.preregistration_sha,
            &self.implementation_subject_sha,
            &self.measurement,
        );
        if expected != self.protocol_evidence_digest {
            return Err(Sem2ProtocolError::ProtocolDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sem2ProtocolError {
    Experiment(Sem2ExperimentError),
    SchemaMismatch,
    PreregistrationMismatch,
    SubjectMismatch,
    ProtocolDigestMismatch,
}

impl From<Sem2ExperimentError> for Sem2ProtocolError {
    fn from(value: Sem2ExperimentError) -> Self {
        Self::Experiment(value)
    }
}

/// Execute SEM-002 only through its frozen preregistration boundary.
///
/// A returned receipt is still benchmark measurement, not executable qualification
/// of `implementation_subject_sha` and not empirical/confidence authority.
pub fn run_preregistered_sym_rsi_sem_002(
    implementation_subject_sha: &str,
) -> Result<ProtocolBoundSem2Receipt, Sem2ProtocolError> {
    let measurement = run_sym_rsi_sem_002(implementation_subject_sha)?;
    let protocol_evidence_digest = protocol_digest(
        SYM_RSI_SEM_002_PREREGISTRATION_SHA,
        implementation_subject_sha,
        &measurement,
    );
    let receipt = ProtocolBoundSem2Receipt {
        schema: SYM_RSI_SEM_002_PROTOCOL_RECEIPT_SCHEMA,
        preregistration_sha: SYM_RSI_SEM_002_PREREGISTRATION_SHA,
        implementation_subject_sha: implementation_subject_sha.to_owned(),
        measurement,
        protocol_evidence_digest,
    };
    receipt.validate()?;
    Ok(receipt)
}

fn protocol_digest(
    preregistration_sha: &str,
    implementation_subject_sha: &str,
    measurement: &Sem2Receipt,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SYM_RSI_SEM_002_PROTOCOL_RECEIPT_SCHEMA.as_bytes());
    hasher.update(preregistration_sha.as_bytes());
    hasher.update(measurement.schema.as_bytes());
    hasher.update(implementation_subject_sha.as_bytes());
    hasher.update(&measurement.semantic_calibration_digest);
    hasher.update(&measurement.margin_calibration_digest);
    hasher.update(&measurement.evidence_digest);
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUBJECT_A: &str = "0123456789abcdef0123456789abcdef01234567";
    const SUBJECT_B: &str = "89abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn public_receipt_binds_frozen_preregistration() {
        let receipt = run_preregistered_sym_rsi_sem_002(SUBJECT_A).unwrap();
        assert_eq!(receipt.preregistration_sha, SYM_RSI_SEM_002_PREREGISTRATION_SHA);
        assert_eq!(receipt.implementation_subject_sha, SUBJECT_A);
        assert_eq!(receipt.measurement.subject_sha, SUBJECT_A);
        receipt.validate().unwrap();
    }

    #[test]
    fn different_implementation_subject_changes_protocol_digest() {
        let left = run_preregistered_sym_rsi_sem_002(SUBJECT_A).unwrap();
        let right = run_preregistered_sym_rsi_sem_002(SUBJECT_B).unwrap();
        assert_ne!(left.protocol_evidence_digest, right.protocol_evidence_digest);
    }

    #[test]
    fn tampered_inner_measurement_digest_fails_validation() {
        let mut receipt = run_preregistered_sym_rsi_sem_002(SUBJECT_A).unwrap();
        receipt.measurement.evidence_digest[0] ^= 0x01;
        assert_eq!(receipt.validate(), Err(Sem2ProtocolError::ProtocolDigestMismatch));
    }

    #[test]
    fn detached_subject_fails_before_digest_check() {
        let mut receipt = run_preregistered_sym_rsi_sem_002(SUBJECT_A).unwrap();
        receipt.implementation_subject_sha = SUBJECT_B.to_owned();
        assert_eq!(receipt.validate(), Err(Sem2ProtocolError::SubjectMismatch));
    }

    #[test]
    fn schema_or_preregistration_substitution_fails_closed() {
        let mut receipt = run_preregistered_sym_rsi_sem_002(SUBJECT_A).unwrap();
        receipt.schema = "wrong-schema";
        assert_eq!(receipt.validate(), Err(Sem2ProtocolError::SchemaMismatch));

        let mut receipt = run_preregistered_sym_rsi_sem_002(SUBJECT_A).unwrap();
        receipt.preregistration_sha = "0000000000000000000000000000000000000000";
        assert_eq!(
            receipt.validate(),
            Err(Sem2ProtocolError::PreregistrationMismatch)
        );
    }
}
