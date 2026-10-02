// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! External cryptographic attestation adapter for observation-fabric receipts.
//!
//! This crate deliberately lives outside symthaea-core. The core defines the
//! receipt/envelope semantics and canonical payload; this adapter adds one concrete
//! cryptographic proof suite (detached Ed25519) without making cryptographic key
//! resolution, authorization, revocation, or substantive truth part of the core.
//!
//! The adapter is Symthaea-native. It is not a claim of W3C Data Integrity
//! conformance: W3C Data Integrity 1.1 is currently a Working Draft, and its
//! document-processing and cryptosuite rules are distinct from this envelope's
//! explicit canonical byte contract.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_core::observation_fabric::{
    IndependenceVerificationReceipt, ReceiptAttestationEnvelope,
    ReceiptAttestationTemporalStatus,
};

pub const CRYPTOSUITE: &str = "symthaea-ed25519-detached-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptAttestationVerificationOutcome {
    Verified,
    InvalidEnvelope,
    ReceiptCommitmentMismatch,
    NotYetValid,
    Expired,
    MissingProof,
    InvalidProofEncoding,
    CryptosuiteMismatch,
    VerificationMethodMismatch,
    ProofPurposeMismatch,
    DomainMismatch,
    ChallengeMismatch,
    InvalidSignature,
    VerificationMethodUnavailable,
    VerificationMethodRevoked,
    VerificationMethodExpired,
    ProofPurposeUnauthorized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStage {
    Passed,
    Failed(ReceiptAttestationVerificationOutcome),
    NotEvaluated,
}

/// Structured stage-by-stage verification evidence. This deliberately does not
/// collapse evidence into an aggregate trust score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptAttestationVerificationReport {
    pub outcome: ReceiptAttestationVerificationOutcome,
    pub structural_validation: VerificationStage,
    pub receipt_commitment: VerificationStage,
    pub temporal_validity: VerificationStage,
    pub cryptosuite: VerificationStage,
    pub verification_method: VerificationStage,
    pub lifecycle: VerificationStage,
    pub proof_purpose_authorization: VerificationStage,
    pub proof_policy: VerificationStage,
    pub cryptographic_proof: VerificationStage,
}

impl ReceiptAttestationVerificationReport {
    fn failed(
        outcome: ReceiptAttestationVerificationOutcome,
        stage: fn(ReceiptAttestationVerificationOutcome) -> VerificationStage,
    ) -> Self {
        let failed = stage(outcome);
        let mut report = Self {
            outcome,
            structural_validation: VerificationStage::NotEvaluated,
            receipt_commitment: VerificationStage::NotEvaluated,
            temporal_validity: VerificationStage::NotEvaluated,
            cryptosuite: VerificationStage::NotEvaluated,
            verification_method: VerificationStage::NotEvaluated,
            lifecycle: VerificationStage::NotEvaluated,
            proof_purpose_authorization: VerificationStage::NotEvaluated,
            proof_policy: VerificationStage::NotEvaluated,
            cryptographic_proof: VerificationStage::NotEvaluated,
        };

        match outcome {
            ReceiptAttestationVerificationOutcome::InvalidEnvelope => {
                report.structural_validation = failed;
            }
            ReceiptAttestationVerificationOutcome::ReceiptCommitmentMismatch => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = failed;
            }
            ReceiptAttestationVerificationOutcome::NotYetValid
            | ReceiptAttestationVerificationOutcome::Expired => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = failed;
            }
            ReceiptAttestationVerificationOutcome::CryptosuiteMismatch => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = VerificationStage::Passed;
                report.cryptosuite = failed;
            }
            ReceiptAttestationVerificationOutcome::VerificationMethodMismatch
            | ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = VerificationStage::Passed;
                report.cryptosuite = VerificationStage::Passed;
                report.verification_method = failed;
            }
            ReceiptAttestationVerificationOutcome::VerificationMethodRevoked
            | ReceiptAttestationVerificationOutcome::VerificationMethodExpired => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = VerificationStage::Passed;
                report.cryptosuite = VerificationStage::Passed;
                report.verification_method = VerificationStage::Passed;
                report.lifecycle = failed;
            }
            ReceiptAttestationVerificationOutcome::ProofPurposeUnauthorized => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = VerificationStage::Passed;
                report.cryptosuite = VerificationStage::Passed;
                report.verification_method = VerificationStage::Passed;
                report.lifecycle = VerificationStage::Passed;
                report.proof_purpose_authorization = failed;
            }
            ReceiptAttestationVerificationOutcome::ProofPurposeMismatch
            | ReceiptAttestationVerificationOutcome::DomainMismatch
            | ReceiptAttestationVerificationOutcome::ChallengeMismatch => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = VerificationStage::Passed;
                report.cryptosuite = VerificationStage::Passed;
                report.verification_method = VerificationStage::Passed;
                report.lifecycle = VerificationStage::Passed;
                report.proof_purpose_authorization = VerificationStage::Passed;
                report.proof_policy = failed;
            }
            ReceiptAttestationVerificationOutcome::MissingProof
            | ReceiptAttestationVerificationOutcome::InvalidProofEncoding
            | ReceiptAttestationVerificationOutcome::InvalidSignature => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = VerificationStage::Passed;
                report.cryptosuite = VerificationStage::Passed;
                report.verification_method = VerificationStage::Passed;
                report.lifecycle = VerificationStage::Passed;
                report.proof_purpose_authorization = VerificationStage::Passed;
                report.proof_policy = VerificationStage::Passed;
                report.cryptographic_proof = failed;
            }
            ReceiptAttestationVerificationOutcome::Verified => {}
        }
        report
    }

    fn passed() -> Self {
        Self {
            outcome: ReceiptAttestationVerificationOutcome::Verified,
            structural_validation: VerificationStage::Passed,
            receipt_commitment: VerificationStage::Passed,
            temporal_validity: VerificationStage::Passed,
            cryptosuite: VerificationStage::Passed,
            verification_method: VerificationStage::Passed,
            lifecycle: VerificationStage::Passed,
            proof_purpose_authorization: VerificationStage::Passed,
            proof_policy: VerificationStage::Passed,
            cryptographic_proof: VerificationStage::Passed,
        }
    }
}

