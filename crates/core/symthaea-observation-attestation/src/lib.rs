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
pub const VERIFIER_VERSION: &str = "symthaea-observation-attestation-report-v4";
const LEGACY_REPORT_VERIFIER_VERSION: &str = "symthaea-observation-attestation-report-v3";

pub const EVALUATION_PROCEDURE_VERSION: &str =
    "symthaea-observation-evaluation-procedure-v2";
pub const EVALUATION_PROCEDURE_ID: &str =
    "symthaea-observation-attestation-ed25519-procedure-v1";

pub const POLICY_VERSION: &str = "symthaea-observation-verification-policy-v1";
pub const VERIFIER_IMPLEMENTATION_ID: &str = "symthaea-observation-attestation-ed25519-v1";
pub const ENVIRONMENT_IDENTITY_VERSION: &str = "symthaea-verifier-environment-v1";

fn is_blake3_fingerprint(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationPolicyInputs {
    pub policy_version: &'static str,
    pub cryptosuite: &'static str,
    pub expected_proof_purpose: Option<String>,
    pub expected_domain: Option<String>,
    pub expected_challenge_fingerprint: Option<String>,
    pub require_active_verification_method: bool,
}

impl VerificationPolicyInputs {
    /// Validate the semantic policy identity before trusting its fingerprint.
    ///
    /// A self-consistent fingerprint is not sufficient because callers may
    /// deserialize and mutate public fields before recomputing that fingerprint.
    pub fn is_well_formed(&self) -> bool {
        let optional_nonempty = |value: &Option<String>| {
            value.as_deref().is_none_or(|value| !value.is_empty())
        };

        !self.policy_version.is_empty()
            && self.cryptosuite == CRYPTOSUITE
            && self.require_active_verification_method
            && optional_nonempty(&self.expected_proof_purpose)
            && optional_nonempty(&self.expected_domain)
            && self
                .expected_challenge_fingerprint
                .as_deref()
                .is_none_or(is_blake3_fingerprint)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        fn write_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        fn write_option(bytes: &mut Vec<u8>, value: Option<&str>) {
            match value {
                Some(value) => { bytes.push(1); write_string(bytes, value); }
                None => bytes.push(0),
            }
        }
        write_string(&mut bytes, self.policy_version);
        write_string(&mut bytes, self.cryptosuite);
        write_option(&mut bytes, self.expected_proof_purpose.as_deref());
        write_option(&mut bytes, self.expected_domain.as_deref());
        write_option(&mut bytes, self.expected_challenge_fingerprint.as_deref());
        bytes.push(self.require_active_verification_method as u8);
        bytes
    }

    pub fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:observation-verification-policy:v1\n");
        hasher.update(&self.canonical_bytes());
        hasher.finalize().to_hex().to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierEnvironmentIdentity {
    pub identity_version: &'static str,
    pub implementation_id: &'static str,
    pub build_fingerprint: String,
    pub runtime_profile: Option<String>,
}

impl VerifierEnvironmentIdentity {
    pub fn new(build_fingerprint: impl Into<String>) -> Self {
        Self {
            identity_version: ENVIRONMENT_IDENTITY_VERSION,
            implementation_id: VERIFIER_IMPLEMENTATION_ID,
            build_fingerprint: build_fingerprint.into(),
            runtime_profile: None,
        }
    }

    pub fn with_runtime_profile(mut self, profile: impl Into<String>) -> Self {
        self.runtime_profile = Some(profile.into());
        self
    }

    /// Validate the semantic environment identity before trusting its fingerprint.
    pub fn is_well_formed(&self) -> bool {
        !self.build_fingerprint.is_empty()
            && self.identity_version == ENVIRONMENT_IDENTITY_VERSION
            && self.implementation_id == VERIFIER_IMPLEMENTATION_ID
            && self
                .runtime_profile
                .as_deref()
                .is_none_or(|profile| !profile.is_empty())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        fn write_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        write_string(&mut bytes, self.identity_version);
        write_string(&mut bytes, self.implementation_id);
        write_string(&mut bytes, &self.build_fingerprint);
        match self.runtime_profile.as_deref() {
            Some(profile) => { bytes.push(1); write_string(&mut bytes, profile); }
            None => bytes.push(0),
        }
        bytes
    }

    pub fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:verifier-environment:v1\n");
        hasher.update(&self.canonical_bytes());
        hasher.finalize().to_hex().to_string()
    }
}

const REPORT_DOMAIN_SEPARATOR_V3: &[u8] = b"symthaea:observation-attestation-report:v3\n";
const REPORT_DOMAIN_SEPARATOR: &[u8] = b"symthaea:observation-attestation-report:v4\n";

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

/// Stable wire/canonicalization tags for outcomes.
///
/// Do not derive these from enum discriminants: inserting or reordering enum variants
/// must never silently change the identity of historical evidence.
const fn verification_outcome_tag(outcome: ReceiptAttestationVerificationOutcome) -> u8 {
    match outcome {
        ReceiptAttestationVerificationOutcome::Verified => 0,
        ReceiptAttestationVerificationOutcome::InvalidEnvelope => 1,
        ReceiptAttestationVerificationOutcome::ReceiptCommitmentMismatch => 2,
        ReceiptAttestationVerificationOutcome::NotYetValid => 3,
        ReceiptAttestationVerificationOutcome::Expired => 4,
        ReceiptAttestationVerificationOutcome::MissingProof => 5,
        ReceiptAttestationVerificationOutcome::InvalidProofEncoding => 6,
        ReceiptAttestationVerificationOutcome::CryptosuiteMismatch => 7,
        ReceiptAttestationVerificationOutcome::VerificationMethodMismatch => 8,
        ReceiptAttestationVerificationOutcome::ProofPurposeMismatch => 9,
        ReceiptAttestationVerificationOutcome::DomainMismatch => 10,
        ReceiptAttestationVerificationOutcome::ChallengeMismatch => 11,
        ReceiptAttestationVerificationOutcome::InvalidSignature => 12,
        ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable => 13,
        ReceiptAttestationVerificationOutcome::VerificationMethodRevoked => 14,
        ReceiptAttestationVerificationOutcome::VerificationMethodExpired => 15,
        ReceiptAttestationVerificationOutcome::ProofPurposeUnauthorized => 16,
    }
}

const CURRENT_ENVELOPE_STRUCTURAL_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::InvalidEnvelope),
];
const CURRENT_RECEIPT_COMMITMENT_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::ReceiptCommitmentMismatch),
];
const CURRENT_TEMPORAL_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::NotYetValid),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::Expired),
];
const CURRENT_CRYPTOSUITE_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::CryptosuiteMismatch),
];
const CURRENT_VERIFICATION_METHOD_RESOLUTION_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::VerificationMethodMismatch),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable),
];
const CURRENT_VERIFICATION_METHOD_LIFECYCLE_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::VerificationMethodRevoked),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::VerificationMethodExpired),
];
const CURRENT_PROOF_PURPOSE_AUTHORIZATION_FAILURE_TAGS: &[u8] = &[verification_outcome_tag(
    ReceiptAttestationVerificationOutcome::ProofPurposeUnauthorized,
)];
const CURRENT_PROOF_POLICY_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::ProofPurposeMismatch),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::DomainMismatch),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::ChallengeMismatch),
];
const CURRENT_CRYPTOGRAPHIC_PROOF_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::MissingProof),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::InvalidProofEncoding),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::InvalidSignature),
];

const LEGACY_ENVELOPE_STRUCTURAL_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::InvalidEnvelope),
];
const LEGACY_RECEIPT_COMMITMENT_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::ReceiptCommitmentMismatch),
];
const LEGACY_TEMPORAL_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::NotYetValid),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::Expired),
];
const LEGACY_CRYPTOSUITE_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::CryptosuiteMismatch),
];
const LEGACY_VERIFICATION_METHOD_RESOLUTION_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::VerificationMethodMismatch),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable),
];
const LEGACY_VERIFICATION_METHOD_LIFECYCLE_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::VerificationMethodRevoked),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::VerificationMethodExpired),
];
const LEGACY_PROOF_PURPOSE_AUTHORIZATION_FAILURE_TAGS: &[u8] = &[verification_outcome_tag(
    ReceiptAttestationVerificationOutcome::ProofPurposeUnauthorized,
)];
const LEGACY_PROOF_POLICY_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::ProofPurposeMismatch),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::DomainMismatch),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::ChallengeMismatch),
];
const LEGACY_CRYPTOGRAPHIC_PROOF_FAILURE_TAGS: &[u8] = &[
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::MissingProof),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::InvalidProofEncoding),
    verification_outcome_tag(ReceiptAttestationVerificationOutcome::InvalidSignature),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStage {
    Passed,
    Failed(ReceiptAttestationVerificationOutcome),
    NotEvaluated,
}

/// Typed identifiers for each verification procedure check.
/// Keeping the identifier and report-stage mapping together prevents a second
/// positional check-id list from silently diverging from the verification report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationCheck {
    EnvelopeStructuralValidation,
    ReceiptCommitment,
    TemporalValidity,
    CryptosuiteConformance,
    VerificationMethodResolution,
    VerificationMethodLifecycle,
    ProofPurposeAuthorization,
    ProofPolicyConformance,
    CryptographicProof,
}

impl EvaluationCheck {
    pub const ALL: &'static [Self] = &[
        Self::EnvelopeStructuralValidation,
        Self::ReceiptCommitment,
        Self::TemporalValidity,
        Self::CryptosuiteConformance,
        Self::VerificationMethodResolution,
        Self::VerificationMethodLifecycle,
        Self::ProofPurposeAuthorization,
        Self::ProofPolicyConformance,
        Self::CryptographicProof,
    ];

    pub const fn definition_version(self) -> &'static str {
        "v1"
    }

    pub const fn id(self) -> &'static str {
        match self {
            Self::EnvelopeStructuralValidation => "envelope-structural-validation",
            Self::ReceiptCommitment => "receipt-commitment",
            Self::TemporalValidity => "temporal-validity",
            Self::CryptosuiteConformance => "cryptosuite-conformance",
            Self::VerificationMethodResolution => "verification-method-resolution",
            Self::VerificationMethodLifecycle => "verification-method-lifecycle",
            Self::ProofPurposeAuthorization => "proof-purpose-authorization",
            Self::ProofPolicyConformance => "proof-policy-conformance",
            Self::CryptographicProof => "cryptographic-proof",
        }
    }

    /// Stable failure-outcome semantics for this check.
    ///
    /// These tags are part of the current procedure fingerprint. If the set of
    /// outcomes a check may emit changes, the procedure identity must change too.
    pub const fn allowed_failure_outcome_tags(self) -> &'static [u8] {
        match self {
            Self::EnvelopeStructuralValidation => CURRENT_ENVELOPE_STRUCTURAL_FAILURE_TAGS,
            Self::ReceiptCommitment => CURRENT_RECEIPT_COMMITMENT_FAILURE_TAGS,
            Self::TemporalValidity => CURRENT_TEMPORAL_FAILURE_TAGS,
            Self::CryptosuiteConformance => CURRENT_CRYPTOSUITE_FAILURE_TAGS,
            Self::VerificationMethodResolution => CURRENT_VERIFICATION_METHOD_RESOLUTION_FAILURE_TAGS,
            Self::VerificationMethodLifecycle => CURRENT_VERIFICATION_METHOD_LIFECYCLE_FAILURE_TAGS,
            Self::ProofPurposeAuthorization => CURRENT_PROOF_PURPOSE_AUTHORIZATION_FAILURE_TAGS,
            Self::ProofPolicyConformance => CURRENT_PROOF_POLICY_FAILURE_TAGS,
            Self::CryptographicProof => CURRENT_CRYPTOGRAPHIC_PROOF_FAILURE_TAGS,
        }
    }

    /// Frozen failure-outcome semantics for the historical v1 procedure.
    const fn legacy_allowed_failure_outcome_tags(self) -> &'static [u8] {
        match self {
            Self::EnvelopeStructuralValidation => LEGACY_ENVELOPE_STRUCTURAL_FAILURE_TAGS,
            Self::ReceiptCommitment => LEGACY_RECEIPT_COMMITMENT_FAILURE_TAGS,
            Self::TemporalValidity => LEGACY_TEMPORAL_FAILURE_TAGS,
            Self::CryptosuiteConformance => LEGACY_CRYPTOSUITE_FAILURE_TAGS,
            Self::VerificationMethodResolution => LEGACY_VERIFICATION_METHOD_RESOLUTION_FAILURE_TAGS,
            Self::VerificationMethodLifecycle => LEGACY_VERIFICATION_METHOD_LIFECYCLE_FAILURE_TAGS,
            Self::ProofPurposeAuthorization => LEGACY_PROOF_PURPOSE_AUTHORIZATION_FAILURE_TAGS,
            Self::ProofPolicyConformance => LEGACY_PROOF_POLICY_FAILURE_TAGS,
            Self::CryptographicProof => LEGACY_CRYPTOGRAPHIC_PROOF_FAILURE_TAGS,
        }
    }

    fn allows_legacy_failure_outcome(
        self,
        outcome: ReceiptAttestationVerificationOutcome,
    ) -> bool {
        self.legacy_allowed_failure_outcome_tags()
            .contains(&verification_outcome_tag(outcome))
    }

    fn allows_failure_outcome(
        self,
        outcome: ReceiptAttestationVerificationOutcome,
    ) -> bool {
        self.allowed_failure_outcome_tags()
            .contains(&verification_outcome_tag(outcome))
    }

    fn stage(self, report: &ReceiptAttestationVerificationReport) -> VerificationStage {
        match self {
            Self::EnvelopeStructuralValidation => report.structural_validation,
            Self::ReceiptCommitment => report.receipt_commitment,
            Self::TemporalValidity => report.temporal_validity,
            Self::CryptosuiteConformance => report.cryptosuite,
            Self::VerificationMethodResolution => report.verification_method,
            Self::VerificationMethodLifecycle => report.lifecycle,
            Self::ProofPurposeAuthorization => report.proof_purpose_authorization,
            Self::ProofPolicyConformance => report.proof_policy,
            Self::CryptographicProof => report.cryptographic_proof,
        }
    }

    fn stage_mut(self, report: &mut ReceiptAttestationVerificationReport) -> &mut VerificationStage {
        match self {
            Self::EnvelopeStructuralValidation => &mut report.structural_validation,
            Self::ReceiptCommitment => &mut report.receipt_commitment,
            Self::TemporalValidity => &mut report.temporal_validity,
            Self::CryptosuiteConformance => &mut report.cryptosuite,
            Self::VerificationMethodResolution => &mut report.verification_method,
            Self::VerificationMethodLifecycle => &mut report.lifecycle,
            Self::ProofPurposeAuthorization => &mut report.proof_purpose_authorization,
            Self::ProofPolicyConformance => &mut report.proof_policy,
            Self::CryptographicProof => &mut report.cryptographic_proof,
        }
    }
}

/// Frozen check sequence for the historical v1 attestation procedure.
///
/// Never append, remove, reorder, or reinterpret entries in this table. A future
/// procedure evolution must change EvaluationCheck::ALL and leave this table
/// untouched unless the historical v1 identity is intentionally retired.
const LEGACY_EVALUATION_CHECKS_V1: &[EvaluationCheck] = &[
    EvaluationCheck::EnvelopeStructuralValidation,
    EvaluationCheck::ReceiptCommitment,
    EvaluationCheck::TemporalValidity,
    EvaluationCheck::CryptosuiteConformance,
    EvaluationCheck::VerificationMethodResolution,
    EvaluationCheck::VerificationMethodLifecycle,
    EvaluationCheck::ProofPurposeAuthorization,
    EvaluationCheck::ProofPolicyConformance,
    EvaluationCheck::CryptographicProof,
];
/// Structured stage-by-stage verification evidence. This deliberately does not
/// collapse evidence into an aggregate trust score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvaluationProcedure {
    pub procedure_version: &'static str,
    pub procedure_id: &'static str,
    pub checks: &'static [EvaluationCheck],
}

impl EvaluationProcedure {
    pub const fn attestation_ed25519() -> Self {
        Self {
            procedure_version: EVALUATION_PROCEDURE_VERSION,
            procedure_id: EVALUATION_PROCEDURE_ID,
            checks: EvaluationCheck::ALL,
        }
    }

    /// Historical v1 procedure definition retained for legacy report evidence.
    pub const fn attestation_ed25519_v1() -> Self {
        Self {
            procedure_version: "symthaea-observation-evaluation-procedure-v1",
            procedure_id: EVALUATION_PROCEDURE_ID,
            checks: LEGACY_EVALUATION_CHECKS_V1,
        }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        fn write_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }

        let mut bytes = Vec::new();
        let legacy_v1 =
            self.procedure_version == "symthaea-observation-evaluation-procedure-v1";
        bytes.extend_from_slice(if legacy_v1 {
            b"symthaea:observation-evaluation-procedure:v1\n"
        } else {
            b"symthaea:observation-evaluation-procedure:v2\n"
        });
        write_string(&mut bytes, self.procedure_version);
        write_string(&mut bytes, self.procedure_id);
        bytes.extend_from_slice(&(self.checks.len() as u64).to_be_bytes());
        for check in self.checks {
            write_string(&mut bytes, check.id());
            if !legacy_v1 {
                write_string(&mut bytes, check.definition_version());
                let allowed_outcomes = check.allowed_failure_outcome_tags();
                bytes.extend_from_slice(&(allowed_outcomes.len() as u64).to_be_bytes());
                bytes.extend_from_slice(allowed_outcomes);
            }
        }
        bytes
    }

    pub fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        let domain = if self.procedure_version == "symthaea-observation-evaluation-procedure-v1" {
            b"symthaea:observation-evaluation-procedure:v1\n"
        } else {
            b"symthaea:observation-evaluation-procedure:v2\n"
        };
        hasher.update(domain);
        hasher.update(&self.canonical_bytes());
        hasher.finalize().to_hex().to_string()
    }

    /// Return the checks that actually executed for a report.
    ///
    /// Compatibility helper retained for callers that only have a legacy report.
    /// New code should consume `EvaluationTrace::executed_check_ids` instead.
    #[deprecated(note = "use EvaluationTrace::executed_check_ids instead")]
    pub fn executed_check_ids(
        &self,
        report: &ReceiptAttestationVerificationReport,
    ) -> Vec<&'static str> {
        self.checks
            .iter()
            .copied()
            .filter_map(|check| {
                (!matches!(check.stage(report), VerificationStage::NotEvaluated))
                    .then_some(check.id())
            })
            .collect()
    }
}

/// Immutable result for one typed verification check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationCheckResult {
    pub sequence: u32,
    pub check: EvaluationCheck,
    pub stage: VerificationStage,
}

impl EvaluationCheckResult {
    pub fn id(&self) -> &'static str {
        self.check.id()
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.sequence.to_be_bytes());
        bytes.extend_from_slice(&(self.id().len() as u64).to_be_bytes());
        bytes.extend_from_slice(self.id().as_bytes());
        match self.stage {
            VerificationStage::Passed => bytes.push(0),
            VerificationStage::Failed(outcome) => {
                bytes.push(1);
                bytes.push(verification_outcome_tag(outcome));
            }
            VerificationStage::NotEvaluated => bytes.push(2),
        }
        bytes
    }
}

/// Durable execution trace for an evaluation procedure.
///
/// The trace contains only checks that executed. Its order is explicit and its
/// procedure fingerprint binds the trace to the procedure that defined the check
/// semantics. The legacy report remains the compatibility projection.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationTrace {
    pub procedure_fingerprint: String,
    pub results: Vec<EvaluationCheckResult>,
}

impl EvaluationTrace {
    /// Reconstruct an execution trace from compatibility report stages.
    ///
    /// Current reports capture the trace directly. This method exists for
    /// historical reports serialized before `execution_trace` was introduced.
    #[deprecated(note = "use the report's captured execution_trace; this is a legacy compatibility path")]
    pub fn from_report(report: &ReceiptAttestationVerificationReport) -> Self {
        Self::from_report_legacy(report)
    }

    fn from_report_legacy(report: &ReceiptAttestationVerificationReport) -> Self {
        let procedure = EvaluationProcedure::attestation_ed25519_v1();
        let results = procedure
            .checks
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, check)| {
                let stage = check.stage(report);
                (!matches!(stage, VerificationStage::NotEvaluated)).then_some(
                    EvaluationCheckResult {
                        sequence: index as u32,
                        check,
                        stage,
                    },
                )
            })
            .collect();
        Self {
            procedure_fingerprint: report.procedure_fingerprint.clone(),
            results,
        }
    }

    pub fn executed_check_ids(&self) -> Vec<&'static str> {
        self.results.iter().map(EvaluationCheckResult::id).collect()
    }

    /// Verify that this trace is the report's captured execution evidence.
    ///
    /// Current v4 reports carry the authoritative trace directly. Legacy reports
    /// without that field use the compatibility stage projection.
    pub fn matches_report(&self, report: &ReceiptAttestationVerificationReport) -> bool {
        if self.procedure_fingerprint != report.procedure_fingerprint {
            return false;
        }

        let matches_trace = if report.verifier_version == LEGACY_REPORT_VERIFIER_VERSION {
            // The execution_trace field did not exist in legacy v3 evidence.
            // Validate only the historical stage projection, regardless of any
            // newer field that may have been attached by a forward serializer.
            self == &EvaluationTrace::from_report_legacy(report)
        } else if report.execution_trace.is_well_formed() {
            self == &report.execution_trace
        } else {
            false
        };
        matches_trace
            && self.terminal_outcome() == Some(report.outcome)
            && self
                .results
                .iter()
                .all(|result| result.check.stage(report) == result.stage)
    }

    /// Return the aggregate outcome represented by this trace.
    ///
    /// A successful outcome is terminal only when the full procedure executed.
    /// A partial prefix of passing checks is intentionally outcome-less.
    pub fn terminal_outcome(&self) -> Option<ReceiptAttestationVerificationOutcome> {
        if !self.is_well_formed() {
            return None;
        }

        match self.results.last().map(|result| result.stage) {
            Some(VerificationStage::Failed(outcome)) => Some(outcome),
            Some(VerificationStage::Passed) => {
                Some(ReceiptAttestationVerificationOutcome::Verified)
            }
            Some(VerificationStage::NotEvaluated) | None => None,
        }
    }

    /// Validate the structural invariants of the durable execution trace.
    pub fn is_well_formed(&self) -> bool {
        let current_procedure = EvaluationProcedure::attestation_ed25519();
        let legacy_procedure = EvaluationProcedure::attestation_ed25519_v1();
        let (procedure, legacy_semantics) =
            if self.procedure_fingerprint == current_procedure.fingerprint() {
                (current_procedure, false)
            } else if self.procedure_fingerprint == legacy_procedure.fingerprint() {
                (legacy_procedure, true)
            } else {
                return false;
            };

        !self.results.is_empty()
            && self.results.iter().enumerate().all(|(index, result)| {
                result.sequence == index as u32
                    && procedure.checks.get(index).copied() == Some(result.check)
                    && match result.stage {
                        VerificationStage::Passed => true,
                        VerificationStage::Failed(outcome) => {
                            if legacy_semantics {
                                result.check.allows_legacy_failure_outcome(outcome)
                            } else {
                                result.check.allows_failure_outcome(outcome)
                            }
                        }
                        VerificationStage::NotEvaluated => false,
                    }
            })
            && self
                .results
                .windows(2)
                .all(|pair| {
                    !matches!(pair[0].stage, VerificationStage::Failed(_))
                })
            && match self.results.last().map(|result| result.stage) {
                Some(VerificationStage::Failed(_)) => true,
                Some(VerificationStage::Passed) => {
                    self.results.len() == procedure.checks.len()
                }
                _ => false,
            }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        fn write_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea:observation-evaluation-trace:v1\n");
        write_string(&mut bytes, &self.procedure_fingerprint);
        bytes.extend_from_slice(&(self.results.len() as u64).to_be_bytes());
        for result in &self.results {
            let encoded = result.canonical_bytes();
            bytes.extend_from_slice(&(encoded.len() as u64).to_be_bytes());
            bytes.extend_from_slice(&encoded);
        }
        bytes
    }

    /// Stable content identity for the executed verification trace.
    pub fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:observation-evaluation-trace:v1\n");
        hasher.update(&self.canonical_bytes());
        hasher.finalize().to_hex().to_string()
    }
}

/// Structured stage-by-stage verification evidence. This deliberately does not
/// collapse evidence into an aggregate trust score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptAttestationVerificationReport {
    pub outcome: ReceiptAttestationVerificationOutcome,
    pub verifier_version: &'static str,
    pub receipt_fingerprint: String,
    pub evaluated_at_unix_ns: i128,
    pub policy_inputs: VerificationPolicyInputs,
    pub policy_fingerprint: String,
    pub environment_identity: VerifierEnvironmentIdentity,
    pub environment_fingerprint: String,
    /// Fingerprint of the exact verification procedure executed to produce this report.
    pub procedure_fingerprint: String,
    /// Execution evidence captured while verification actually ran.
    ///
    /// Legacy serialized reports may deserialize this with the default empty trace;
    /// current verifier-produced reports always contain the populated trace.
    #[serde(default)]
    pub execution_trace: EvaluationTrace,
    /// Fingerprint of the resolver's durable view, when resolution returned a paired result.
    ///
    /// A resolver error does not fabricate this field from a later verifier-side
    /// observation. A paired None is likewise authoritative and means that the
    /// resolver did not provide a durable snapshot identifier for that result.
    pub resolution_snapshot_fingerprint: Option<String>,
    /// Method identifier supplied to the resolver and, when available, confirmed by it.
    ///
    /// On resolver failure this may still contain the requested identifier for
    /// diagnostics; consumers must not interpret Some here as proof that resolution
    /// succeeded. Resolver success is established by the execution trace and outcome.
    pub resolved_verification_method: Option<String>,
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
    /// Check that stored identity fingerprints still commit to their semantic inputs.
    ///
    /// Reports are serializable public data, so callers must not assume these
    /// redundant fields remain mutually consistent after deserialization.
    pub fn has_consistent_identity_bindings(&self) -> bool {
        let supported_version = self.verifier_version == VERIFIER_VERSION
            || self.verifier_version == LEGACY_REPORT_VERIFIER_VERSION;
        let expected_procedure_fingerprint = if self.verifier_version == VERIFIER_VERSION {
            EvaluationProcedure::attestation_ed25519().fingerprint()
        } else if self.verifier_version == LEGACY_REPORT_VERIFIER_VERSION {
            EvaluationProcedure::attestation_ed25519_v1().fingerprint()
        } else {
            return false;
        };

        supported_version
            && is_blake3_fingerprint(&self.receipt_fingerprint)
            && self.policy_inputs.is_well_formed()
            && self.environment_identity.is_well_formed()
            && self.policy_fingerprint == self.policy_inputs.fingerprint()
            && self.environment_fingerprint == self.environment_identity.fingerprint()
            && self.procedure_fingerprint == expected_procedure_fingerprint
    }

    /// Validate metadata that accompanies the execution trace but is not itself
    /// represented by a trace result.
    ///
    /// Once verification-method resolution has passed, a report must retain the
    /// method identity that was actually bound to that successful resolution.
    fn has_consistent_resolution_metadata(&self) -> bool {
        let snapshot_is_well_formed = self
            .resolution_snapshot_fingerprint
            .as_deref()
            .is_none_or(|snapshot| !snapshot.trim().is_empty());
        let resolved_method_is_well_formed = self
            .resolved_verification_method
            .as_deref()
            .is_none_or(|method| !method.trim().is_empty());
        let resolved_method_is_required =
            matches!(self.verification_method, VerificationStage::Passed);

        snapshot_is_well_formed
            && resolved_method_is_well_formed
            && (!resolved_method_is_required || self.resolved_verification_method.is_some())
    }

    /// Validate that this report is internally coherent without requiring
    /// construction of an EvidenceEvaluation.
    ///
    /// Legacy v3 reports are validated against their historical stage projection
    /// because first-class execution traces were not part of the v3 evidence identity.
    /// Current v4 reports must validate their captured execution trace directly.
    pub fn is_well_formed(&self) -> bool {
        if !self.has_consistent_identity_bindings()
            || !self.has_consistent_resolution_metadata()
        {
            return false;
        }

        if self.verifier_version == LEGACY_REPORT_VERIFIER_VERSION {
            let legacy_trace = EvaluationTrace::from_report_legacy(self);
            legacy_trace.is_well_formed()
                && legacy_trace.terminal_outcome() == Some(self.outcome)
                && legacy_trace.matches_report(self)
        } else {
            self.execution_trace.is_well_formed()
                && self.execution_trace.terminal_outcome() == Some(self.outcome)
                && self.execution_trace.matches_report(self)
        }
    }

    fn failed(
        outcome: ReceiptAttestationVerificationOutcome,
        failed_check: EvaluationCheck,
        receipt_fingerprint: String,
        resolved_verification_method: Option<String>,
        evaluated_at_unix_ns: i128,
        policy_inputs: VerificationPolicyInputs,
        environment_identity: VerifierEnvironmentIdentity,
    ) -> Self {
        let procedure = EvaluationProcedure::attestation_ed25519();
        let mut execution_results = Vec::new();
        let mut report = Self {
            outcome,
            verifier_version: VERIFIER_VERSION,
            receipt_fingerprint,
            evaluated_at_unix_ns,
            policy_fingerprint: policy_inputs.fingerprint(),
            environment_fingerprint: environment_identity.fingerprint(),
            procedure_fingerprint: procedure.fingerprint(),
            execution_trace: EvaluationTrace::default(),
            policy_inputs,
            environment_identity,
            resolution_snapshot_fingerprint: None,
            resolved_verification_method,
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

        for (index, check) in procedure.checks.iter().copied().enumerate() {
            let stage = check.stage_mut(&mut report);
            if check == failed_check {
                let result = VerificationStage::Failed(outcome);
                *stage = result;
                execution_results.push(EvaluationCheckResult {
                    sequence: index as u32,
                    check,
                    stage: result,
                });
                break;
            }
            *stage = VerificationStage::Passed;
            execution_results.push(EvaluationCheckResult {
                sequence: index as u32,
                check,
                stage: VerificationStage::Passed,
            });
        }
        report.execution_trace = EvaluationTrace {
            procedure_fingerprint: report.procedure_fingerprint.clone(),
            results: execution_results,
        };
        report
    }

    fn failed_with_resolution_snapshot(
        outcome: ReceiptAttestationVerificationOutcome,
        failed_check: EvaluationCheck,
        receipt_fingerprint: String,
        resolved_verification_method: Option<String>,
        resolution_snapshot_fingerprint: Option<String>,
        evaluated_at_unix_ns: i128,
        policy_inputs: VerificationPolicyInputs,
        environment_identity: VerifierEnvironmentIdentity,
    ) -> Self {
        let mut report = Self::failed(
            outcome,
            failed_check,
            receipt_fingerprint,
            resolved_verification_method,
            evaluated_at_unix_ns,
            policy_inputs,
            environment_identity,
        );
        report.resolution_snapshot_fingerprint = resolution_snapshot_fingerprint;
        report
    }

    fn passed(
        receipt_fingerprint: String,
        resolved_verification_method: Option<String>,
        evaluated_at_unix_ns: i128,
        policy_inputs: VerificationPolicyInputs,
        environment_identity: VerifierEnvironmentIdentity,
    ) -> Self {
        let procedure = EvaluationProcedure::attestation_ed25519();
        let execution_trace = EvaluationTrace {
            procedure_fingerprint: procedure.fingerprint(),
            results: procedure
                .checks
                .iter()
                .copied()
                .enumerate()
                .map(|(index, check)| EvaluationCheckResult {
                    sequence: index as u32,
                    check,
                    stage: VerificationStage::Passed,
                })
                .collect(),
        };
        Self {
            outcome: ReceiptAttestationVerificationOutcome::Verified,
            verifier_version: VERIFIER_VERSION,
            receipt_fingerprint,
            evaluated_at_unix_ns,
            policy_fingerprint: policy_inputs.fingerprint(),
            environment_fingerprint: environment_identity.fingerprint(),
            procedure_fingerprint: procedure.fingerprint(),
            execution_trace,
            policy_inputs,
            environment_identity,
            resolution_snapshot_fingerprint: None,
            resolved_verification_method,
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

    pub fn canonical_bytes(&self) -> Vec<u8> {
        fn write_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        fn write_stage(bytes: &mut Vec<u8>, stage: VerificationStage) {
            match stage {
                VerificationStage::Passed => bytes.push(0),
                VerificationStage::Failed(outcome) => {
                    bytes.push(1);
                    bytes.push(verification_outcome_tag(outcome));
                }
                VerificationStage::NotEvaluated => bytes.push(2),
            }
        }

        let mut bytes = Vec::new();
        let legacy_v3 = self.verifier_version == LEGACY_REPORT_VERIFIER_VERSION;
        bytes.extend_from_slice(if legacy_v3 {
            REPORT_DOMAIN_SEPARATOR_V3
        } else {
            REPORT_DOMAIN_SEPARATOR
        });
        write_string(&mut bytes, self.verifier_version);
        write_string(&mut bytes, &self.receipt_fingerprint);
        bytes.extend_from_slice(&self.evaluated_at_unix_ns.to_be_bytes());
        write_string(&mut bytes, &self.policy_fingerprint);
        write_string(&mut bytes, &self.environment_fingerprint);
        write_string(&mut bytes, &self.procedure_fingerprint);
        if !legacy_v3 {
            let trace_bytes = self.execution_trace.canonical_bytes();
            bytes.extend_from_slice(&(trace_bytes.len() as u64).to_be_bytes());
            bytes.extend_from_slice(&trace_bytes);
        }
        match &self.resolution_snapshot_fingerprint {
            Some(value) => { bytes.push(1); write_string(&mut bytes, value); }
            None => bytes.push(0),
        }
        match &self.resolved_verification_method {
            Some(method) => {
                bytes.push(1);
                write_string(&mut bytes, method);
            }
            None => bytes.push(0),
        }
        bytes.push(verification_outcome_tag(self.outcome));
        write_stage(&mut bytes, self.structural_validation);
        write_stage(&mut bytes, self.receipt_commitment);
        write_stage(&mut bytes, self.temporal_validity);
        write_stage(&mut bytes, self.cryptosuite);
        write_stage(&mut bytes, self.verification_method);
        write_stage(&mut bytes, self.lifecycle);
        write_stage(&mut bytes, self.proof_purpose_authorization);
        write_stage(&mut bytes, self.proof_policy);
        write_stage(&mut bytes, self.cryptographic_proof);
        bytes
    }

    /// Stable content identity of the captured execution trace.
    pub fn execution_trace_fingerprint(&self) -> String {
        self.execution_trace.fingerprint()
    }

    pub fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        let domain = if self.verifier_version == LEGACY_REPORT_VERIFIER_VERSION {
            REPORT_DOMAIN_SEPARATOR_V3
        } else {
            REPORT_DOMAIN_SEPARATOR
        };
        hasher.update(domain);
        hasher.update(&self.canonical_bytes());
        hasher.finalize().to_hex().to_string()
    }

}

pub const VERIFICATION_CONTEXT_VERSION: &str =
    "symthaea-observation-verification-context-v4";
pub const EVIDENCE_EVALUATION_VERSION: &str =
    "symthaea-observation-evaluation-v7";
pub const ATTESTATION_VERIFICATION_EVALUATION_TYPE: &str =
    "receipt-attestation-verification";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationContext {
    pub context_version: &'static str,
    pub policy_fingerprint: String,
    /// Stable semantic identity of the verifier implementation.
    pub verifier_id: &'static str,
    /// Report/protocol version used by the verifier.
    pub verifier_version: &'static str,
    pub environment_fingerprint: String,
    /// Stable semantic identity of the evaluation procedure family.
    pub procedure_id: &'static str,
    pub procedure_fingerprint: String,
    /// Stable identity of the evaluator instance or organization, when available.
    ///
    /// This is deliberately distinct from verifier implementation/version: two
    /// independent evaluators may run the same procedure and implementation.
    pub evaluator_identity_fingerprint: Option<String>,
    pub resolution_snapshot_fingerprint: Option<String>,
    pub trust_root_fingerprint: Option<String>,
    pub authorization_policy_fingerprint: Option<String>,
    pub evaluated_at_unix_ns: i128,
}

impl VerificationContext {
    pub fn from_report(report: &ReceiptAttestationVerificationReport) -> Self {
        Self {
            context_version: VERIFICATION_CONTEXT_VERSION,
            policy_fingerprint: report.policy_fingerprint.clone(),
            verifier_id: VERIFIER_IMPLEMENTATION_ID,
            verifier_version: report.verifier_version,
            environment_fingerprint: report.environment_fingerprint.clone(),
            procedure_id: EVALUATION_PROCEDURE_ID,
            procedure_fingerprint: report.procedure_fingerprint.clone(),
            evaluator_identity_fingerprint: None,
            resolution_snapshot_fingerprint: report.resolution_snapshot_fingerprint.clone(),
            trust_root_fingerprint: None,
            authorization_policy_fingerprint: None,
            evaluated_at_unix_ns: report.evaluated_at_unix_ns,
        }
    }

    pub fn with_trust_root_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.trust_root_fingerprint = Some(fingerprint.into());
        self
    }

    pub fn with_authorization_policy_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.authorization_policy_fingerprint = Some(fingerprint.into());
        self
    }

    pub fn with_evaluator_identity_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.evaluator_identity_fingerprint = Some(fingerprint.into());
        self
    }

    /// Validate semantic context identity independently of any source report.
    ///
    /// A matching self-fingerprint is not sufficient: the fingerprint can be
    /// recomputed after semantic identifiers are maliciously replaced. The
    /// context therefore also has to use the supported schema, verifier identity,
    /// procedure identity, and the procedure fingerprint bound to its report version.
    pub fn is_well_formed(&self) -> bool {
        let expected_procedure_fingerprint = if self.verifier_version == VERIFIER_VERSION {
            EvaluationProcedure::attestation_ed25519().fingerprint()
        } else if self.verifier_version == LEGACY_REPORT_VERIFIER_VERSION {
            EvaluationProcedure::attestation_ed25519_v1().fingerprint()
        } else {
            return false;
        };

        let optional_identity_is_well_formed = |value: &Option<String>| {
            value.as_deref().is_none_or(|identity| !identity.is_empty())
        };

        self.context_version == VERIFICATION_CONTEXT_VERSION
            && self.verifier_id == VERIFIER_IMPLEMENTATION_ID
            && self.procedure_id == EVALUATION_PROCEDURE_ID
            && self.procedure_fingerprint == expected_procedure_fingerprint
            && !self.policy_fingerprint.is_empty()
            && !self.environment_fingerprint.is_empty()
            && optional_identity_is_well_formed(&self.evaluator_identity_fingerprint)
            && optional_identity_is_well_formed(&self.resolution_snapshot_fingerprint)
            && optional_identity_is_well_formed(&self.trust_root_fingerprint)
            && optional_identity_is_well_formed(&self.authorization_policy_fingerprint)
    }

    /// Verify that execution-bound context fields still correspond to the report.
    /// Supplemental evaluator/trust/authorization identities are deliberately
    /// excluded because they can be supplied by an external evaluation authority.
    pub fn matches_report(&self, report: &ReceiptAttestationVerificationReport) -> bool {
        self.is_well_formed()
            && report.is_well_formed()
            && self.context_version == VERIFICATION_CONTEXT_VERSION
            && self.policy_fingerprint == report.policy_fingerprint
            && self.verifier_id == VERIFIER_IMPLEMENTATION_ID
            && self.verifier_version == report.verifier_version
            && self.environment_fingerprint == report.environment_fingerprint
            && self.procedure_id == EVALUATION_PROCEDURE_ID
            && self.procedure_fingerprint == report.procedure_fingerprint
            && self.resolution_snapshot_fingerprint == report.resolution_snapshot_fingerprint
            && self.evaluated_at_unix_ns == report.evaluated_at_unix_ns
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        fn write_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        fn write_option(bytes: &mut Vec<u8>, value: Option<&str>) {
            match value {
                Some(value) => { bytes.push(1); write_string(bytes, value); }
                None => bytes.push(0),
            }
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea:observation-verification-context:v4\n");
        write_string(&mut bytes, self.context_version);
        write_string(&mut bytes, &self.policy_fingerprint);
        write_string(&mut bytes, self.verifier_id);
        write_string(&mut bytes, self.verifier_version);
        write_string(&mut bytes, &self.environment_fingerprint);
        write_string(&mut bytes, self.procedure_id);
        write_string(&mut bytes, &self.procedure_fingerprint);
        write_option(&mut bytes, self.evaluator_identity_fingerprint.as_deref());
        write_option(&mut bytes, self.resolution_snapshot_fingerprint.as_deref());
        write_option(&mut bytes, self.trust_root_fingerprint.as_deref());
        write_option(&mut bytes, self.authorization_policy_fingerprint.as_deref());
        bytes.extend_from_slice(&self.evaluated_at_unix_ns.to_be_bytes());
        bytes
    }

    pub fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:observation-verification-context:v4\n");
        hasher.update(&self.canonical_bytes());
        hasher.finalize().to_hex().to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvaluationClaim {
    ReceiptIntegrity,
    AttestationAuthenticity,
    TemporalValidity,
    CryptosuiteConformance,
    VerificationMethodResolution,
    VerificationMethodLifecycle,
    ProofPurposeAuthorization,
    ProofPolicyConformance,
    CryptographicProofValidity,
    UnderlyingObservationTruth,
    SemanticValidity,
    ExternalWorldCorrespondence,
    AttesterIntent,
    EvaluatorIndependence,
    EnvelopeStructuralValidity,
}

impl EvaluationClaim {
    /// Complete claim universe for the Evidence Fabric boundary.
    ///
    /// A well-formed boundary must classify every known claim exactly once.
    pub const ALL: &'static [Self] = &[
        Self::ReceiptIntegrity,
        Self::AttestationAuthenticity,
        Self::TemporalValidity,
        Self::CryptosuiteConformance,
        Self::VerificationMethodResolution,
        Self::VerificationMethodLifecycle,
        Self::ProofPurposeAuthorization,
        Self::ProofPolicyConformance,
        Self::CryptographicProofValidity,
        Self::UnderlyingObservationTruth,
        Self::SemanticValidity,
        Self::ExternalWorldCorrespondence,
        Self::AttesterIntent,
        Self::EvaluatorIndependence,
        Self::EnvelopeStructuralValidity,
    ];
}

fn evaluation_claim_tag(claim: EvaluationClaim) -> u8 {
    match claim {
        EvaluationClaim::ReceiptIntegrity => 0,
        EvaluationClaim::AttestationAuthenticity => 1,
        EvaluationClaim::TemporalValidity => 2,
        EvaluationClaim::CryptosuiteConformance => 3,
        EvaluationClaim::VerificationMethodResolution => 4,
        EvaluationClaim::VerificationMethodLifecycle => 5,
        EvaluationClaim::ProofPurposeAuthorization => 6,
        EvaluationClaim::ProofPolicyConformance => 7,
        EvaluationClaim::CryptographicProofValidity => 8,
        EvaluationClaim::UnderlyingObservationTruth => 9,
        EvaluationClaim::SemanticValidity => 10,
        EvaluationClaim::ExternalWorldCorrespondence => 11,
        EvaluationClaim::AttesterIntent => 12,
        EvaluationClaim::EvaluatorIndependence => 13,
        // Append-only: never renumber existing canonical claim tags.
        EvaluationClaim::EnvelopeStructuralValidity => 14,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationBoundary {
    pub established: Vec<EvaluationClaim>,
    pub not_established: Vec<EvaluationClaim>,
    pub indeterminate: Vec<EvaluationClaim>,
}

impl EvaluationBoundary {
    pub fn from_report(report: &ReceiptAttestationVerificationReport) -> Self {
        fn classify(
            stage: VerificationStage,
            claim: EvaluationClaim,
            established: &mut Vec<EvaluationClaim>,
            not_established: &mut Vec<EvaluationClaim>,
            indeterminate: &mut Vec<EvaluationClaim>,
        ) {
            match stage {
                VerificationStage::Passed => established.push(claim),
                VerificationStage::Failed(_) => not_established.push(claim),
                VerificationStage::NotEvaluated => indeterminate.push(claim),
            }
        }

        let mut established = Vec::new();
        let mut not_established = Vec::new();
        let mut indeterminate = Vec::new();

        classify(
            report.structural_validation,
            EvaluationClaim::EnvelopeStructuralValidity,
            &mut established,
            &mut not_established,
            &mut indeterminate,
        );
        classify(
            report.receipt_commitment,
            EvaluationClaim::ReceiptIntegrity,
            &mut established,
            &mut not_established,
            &mut indeterminate,
        );
        classify(
            report.temporal_validity,
            EvaluationClaim::TemporalValidity,
            &mut established,
            &mut not_established,
            &mut indeterminate,
        );
        classify(
            report.cryptosuite,
            EvaluationClaim::CryptosuiteConformance,
            &mut established,
            &mut not_established,
            &mut indeterminate,
        );
        classify(
            report.verification_method,
            EvaluationClaim::VerificationMethodResolution,
            &mut established,
            &mut not_established,
            &mut indeterminate,
        );
        classify(
            report.lifecycle,
            EvaluationClaim::VerificationMethodLifecycle,
            &mut established,
            &mut not_established,
            &mut indeterminate,
        );
        classify(
            report.proof_purpose_authorization,
            EvaluationClaim::ProofPurposeAuthorization,
            &mut established,
            &mut not_established,
            &mut indeterminate,
        );
        classify(
            report.proof_policy,
            EvaluationClaim::ProofPolicyConformance,
            &mut established,
            &mut not_established,
            &mut indeterminate,
        );
        classify(
            report.cryptographic_proof,
            EvaluationClaim::CryptographicProofValidity,
            &mut established,
            &mut not_established,
            &mut indeterminate,
        );

        // Authenticity is a compound claim: the receipt must be structurally
        // acceptable, bound to its commitment, and cryptographically valid.
        match (
            report.structural_validation,
            report.receipt_commitment,
            report.cryptographic_proof,
        ) {
            (
                VerificationStage::Passed,
                VerificationStage::Passed,
                VerificationStage::Passed,
            ) => established.push(EvaluationClaim::AttestationAuthenticity),
            (VerificationStage::NotEvaluated, _, _)
            | (_, VerificationStage::NotEvaluated, _)
            | (_, _, VerificationStage::NotEvaluated) => {
                indeterminate.push(EvaluationClaim::AttestationAuthenticity)
            }
            _ => not_established.push(EvaluationClaim::AttestationAuthenticity),
        }

        // These claims are outside the scope of this cryptographic verifier.
        // They are therefore explicitly not-established, not merely omitted.
        not_established.extend([
            EvaluationClaim::UnderlyingObservationTruth,
            EvaluationClaim::SemanticValidity,
            EvaluationClaim::ExternalWorldCorrespondence,
            EvaluationClaim::AttesterIntent,
            EvaluationClaim::EvaluatorIndependence,
        ]);

        Self {
            established,
            not_established,
            indeterminate,
        }
    }

    /// Validate the epistemic partition: every known claim occupies exactly one bucket.
    pub fn is_well_formed(&self) -> bool {
        let established: std::collections::BTreeSet<_> =
            self.established.iter().copied().collect();
        let not_established: std::collections::BTreeSet<_> =
            self.not_established.iter().copied().collect();
        let indeterminate: std::collections::BTreeSet<_> =
            self.indeterminate.iter().copied().collect();
        let all_claims: std::collections::BTreeSet<_> =
            EvaluationClaim::ALL.iter().copied().collect();

        let mut present = established.clone();
        present.extend(not_established.iter().copied());
        present.extend(indeterminate.iter().copied());

        established.len() == self.established.len()
            && not_established.len() == self.not_established.len()
            && indeterminate.len() == self.indeterminate.len()
            && established.is_disjoint(&not_established)
            && established.is_disjoint(&indeterminate)
            && not_established.is_disjoint(&indeterminate)
            && present == all_claims
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut established = self.established.clone();
        let mut not_established = self.not_established.clone();
        let mut indeterminate = self.indeterminate.clone();
        // Canonicalization sorts claims for stable ordering but deliberately does
        // not deduplicate malformed inputs. Validation and canonical identity are
        // separate: malformed evidence must not hash to the same identity as a
        // well-formed boundary merely because canonicalization repaired it.
        established.sort_by_key(|v| evaluation_claim_tag(*v));
        not_established.sort_by_key(|v| evaluation_claim_tag(*v));
        indeterminate.sort_by_key(|v| evaluation_claim_tag(*v));

        let mut bytes = Vec::new();
        for claims in [&established, &not_established, &indeterminate] {
            bytes.extend_from_slice(&(claims.len() as u64).to_be_bytes());
            for claim in claims.iter().copied() {
                bytes.push(evaluation_claim_tag(claim));
            }
        }
        bytes
    }
}

/// Legacy limitation summary retained for source compatibility.
///
/// New evidence should use `EvaluationBoundary`, which distinguishes
/// established, not-established, and indeterminate claims.
#[deprecated(note = "use EvaluationBoundary instead")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationLimitation {
    UnderlyingObservationTruthNotEvaluated,
    SemanticValidityNotEvaluated,
    ExternalWorldStateNotEvaluated,
    AttesterIntentNotEvaluated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceEvaluation {
    pub evaluation_version: &'static str,
    pub subject_fingerprint: String,
    pub evaluation_type: &'static str,
    pub context: VerificationContext,
    pub context_fingerprint: String,
    /// Immutable execution trace; this is the authoritative record of checks that ran.
    pub execution_trace: EvaluationTrace,
    pub verification_report_fingerprint: String,
    /// Compatibility summary of the evaluation outcome. Detailed execution evidence
    /// lives exclusively in `execution_trace`.
    pub outcome: ReceiptAttestationVerificationOutcome,
    pub boundary: EvaluationBoundary,
}

impl EvidenceEvaluation {
    pub fn from_report(report: &ReceiptAttestationVerificationReport) -> Self {
        let context = VerificationContext::from_report(report);
        let execution_trace = if report.verifier_version == LEGACY_REPORT_VERIFIER_VERSION {
            // Legacy v3 canonical identity predates first-class execution traces.
            // Always reconstruct from the historical stage projection; any newer
            // trace field is outside the v3 evidence contract.
            EvaluationTrace::from_report_legacy(report)
        } else {
            report.execution_trace.clone()
        };
        Self {
            evaluation_version: EVIDENCE_EVALUATION_VERSION,
            subject_fingerprint: report.receipt_fingerprint.clone(),
            evaluation_type: ATTESTATION_VERIFICATION_EVALUATION_TYPE,
            context_fingerprint: context.fingerprint(),
            context,
            execution_trace,
            verification_report_fingerprint: report.fingerprint(),
            outcome: report.outcome,
            boundary: EvaluationBoundary::from_report(report),
        }
    }

    /// Construct an evaluation from a report and an explicitly supplied context.
    ///
    /// The context is part of the evaluation's evidence identity. Callers should
    /// only supply context that actually governed the evaluation; changing context
    /// after construction is intentionally not supported.
    pub fn from_report_with_context(
        report: &ReceiptAttestationVerificationReport,
        context: VerificationContext,
    ) -> Self {
        // Only supplemental evaluator/trust/authorization context may be
        // supplied here. Procedure, policy, verifier, environment, resolver
        // snapshot, and evaluation time are execution facts already bound by the report and
        // therefore cannot be replaced after the evaluation occurred.
        let mut evaluation = Self::from_report(report);
        let mut merged_context = VerificationContext::from_report(report);
        merged_context.evaluator_identity_fingerprint =
            context.evaluator_identity_fingerprint;
        merged_context.trust_root_fingerprint = context.trust_root_fingerprint;
        merged_context.authorization_policy_fingerprint =
            context.authorization_policy_fingerprint;
        evaluation.context_fingerprint = merged_context.fingerprint();
        evaluation.context = merged_context;
        evaluation
    }

    /// Validate the internal integrity of this evaluation without requiring
    /// access to the source verification report.
    ///
    /// This is intentionally weaker than report consistency: it proves that the
    /// evaluation is internally coherent, not that it still matches the exact
    /// report from which it was materialized.
    pub fn is_well_formed(&self) -> bool {
        self.evaluation_version == EVIDENCE_EVALUATION_VERSION
            && self.evaluation_type == ATTESTATION_VERIFICATION_EVALUATION_TYPE
            && is_blake3_fingerprint(&self.subject_fingerprint)
            && is_blake3_fingerprint(&self.verification_report_fingerprint)
            && self.context.is_well_formed()
            && self.context_fingerprint == self.context.fingerprint()
            && self.execution_trace.is_well_formed()
            && self.execution_trace.terminal_outcome() == Some(self.outcome)
            && self.boundary.is_well_formed()
    }

    /// Validate that this evaluation remains consistent with the report that
    /// materialized it. This catches post-hoc mutation of outcome, subject,
    /// context identity, execution evidence, or epistemic boundary.
    pub fn is_consistent_with_report(
        &self,
        report: &ReceiptAttestationVerificationReport,
    ) -> bool {
        // Refuse to materialize consistency from a report whose own execution
        // evidence no longer satisfies the report contract. This keeps the
        // report-level invariant as the single integrity gate for consumers.
        report.is_well_formed()
            && self.is_well_formed()
            && self.subject_fingerprint == report.receipt_fingerprint
            && self.verification_report_fingerprint == report.fingerprint()
            && self.context_fingerprint == self.context.fingerprint()
            && self.context.matches_report(report)
            && self.execution_trace.procedure_fingerprint == report.procedure_fingerprint
            && self.execution_trace.matches_report(report)
            && self.execution_trace.terminal_outcome() == Some(self.outcome)
            && self.execution_trace.is_well_formed()
            && self.boundary == EvaluationBoundary::from_report(report)
            && self.boundary.is_well_formed()
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        fn write_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea:evidence-evaluation:v7\n");
        write_string(&mut bytes, self.evaluation_version);
        write_string(&mut bytes, &self.subject_fingerprint);
        write_string(&mut bytes, self.evaluation_type);
        write_string(&mut bytes, &self.context_fingerprint);
        let trace_bytes = self.execution_trace.canonical_bytes();
        bytes.extend_from_slice(&(trace_bytes.len() as u64).to_be_bytes());
        bytes.extend_from_slice(&trace_bytes);
        write_string(&mut bytes, &self.verification_report_fingerprint);
        bytes.push(verification_outcome_tag(self.outcome));
        bytes.extend_from_slice(&self.boundary.canonical_bytes());
        bytes
    }

    /// Stable content identity of the execution evidence referenced by this evaluation.
    pub fn execution_trace_fingerprint(&self) -> String {
        self.execution_trace.fingerprint()
    }

    pub fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:evidence-evaluation:v7\n");
        hasher.update(&self.canonical_bytes());
        hasher.finalize().to_hex().to_string()
    }
}

impl ReceiptAttestationVerificationReport {
    pub fn to_evidence_evaluation(&self) -> EvidenceEvaluation {
        EvidenceEvaluation::from_report(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationMethodStatus {
    Active,
    Revoked,
    Expired,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedVerificationMethod {
    pub verification_method: String,
    pub verifying_key: VerifyingKey,
    pub status: VerificationMethodStatus,
    pub allowed_proof_purposes: Vec<String>,
}

impl ResolvedVerificationMethod {
    pub fn is_authorized_for(&self, proof_purpose: &str) -> bool {
        self.allowed_proof_purposes
            .iter()
            .any(|purpose| purpose == proof_purpose)
    }
}

/// A resolved verification method paired with the resolver snapshot that
/// produced it.
///
/// The pairing is intentional: a bare resolved key plus a separately queried
/// snapshot fingerprint can describe two different resolver states when the
/// backing registry is mutable. Implementations with mutable or remote state
/// should override the resolver's resolve_with_snapshot method so this pair
/// is captured atomically from one durable view.
#[derive(Debug, Clone)]
pub struct ResolvedVerificationMethodSnapshot {
    pub resolved: ResolvedVerificationMethod,
    pub snapshot_fingerprint: Option<String>,
}

/// Application-supplied verification-method resolver.
///
/// Resolution, controller authorization, key lifecycle, and status are deliberately
/// injected rather than performed through network access in this crate.
pub trait VerificationMethodResolver {
    fn resolve(
        &self,
        verification_method: &str,
    ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError>;

    /// Resolve a verification method and, when the implementation can prove the
    /// pairing came from one consistency-preserving view, bind it to a snapshot.
    ///
    /// The compatibility default intentionally returns no snapshot. Combining
    /// resolve() with a separate snapshot_fingerprint_for() observation can create
    /// split-brain evidence for mutable resolvers, so a resolver must override this
    /// method to claim a paired snapshot.
    fn resolve_with_snapshot(
        &self,
        verification_method: &str,
    ) -> Result<ResolvedVerificationMethodSnapshot, VerificationMethodResolutionError> {
        Ok(ResolvedVerificationMethodSnapshot {
            resolved: self.resolve(verification_method)?,
            snapshot_fingerprint: None,
        })
    }

    /// Return a deterministic fingerprint of the resolver state consulted by verification.
    ///
    /// Implementations backed by mutable or remote state should return a fingerprint of
    /// the exact durable snapshot used for resolution. Returning `None` means the caller
    /// must supply an explicit snapshot fingerprint when durable auditability is required.
    fn snapshot_fingerprint(&self) -> Option<String> {
        None
    }

    /// Return a fingerprint scoped to the verification-method resolution being performed.
    ///
    /// This accessor is independent from resolve_with_snapshot(): calling it does not
    /// establish that its value was observed in the same state as a prior resolve().
    /// Mutable or remote implementations should override resolve_with_snapshot() when they
    /// can prove that the resolved method and snapshot came from one consistency-preserving view.
    fn snapshot_fingerprint_for(&self, verification_method: &str) -> Option<String> {
        let _ = verification_method;
        self.snapshot_fingerprint()
    }
}

/// Internal fail-closed resolver used by the resolver-free compatibility entry point.
struct UnresolvedVerificationMethodResolver;

impl VerificationMethodResolver for UnresolvedVerificationMethodResolver {
    fn resolve(
        &self,
        _verification_method: &str,
    ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
        Err(VerificationMethodResolutionError::Unavailable)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VerificationMethodResolutionError {
    #[error("verification method is unavailable")]
    Unavailable,
}

#[derive(Debug, Clone, Default)]
pub struct InMemoryVerificationMethodResolver {
    methods: std::collections::BTreeMap<String, ResolvedVerificationMethod>,
}

impl InMemoryVerificationMethodResolver {
    pub fn new(methods: impl IntoIterator<Item = ResolvedVerificationMethod>) -> Self {
        Self {
            methods: methods
                .into_iter()
                .map(|method| (method.verification_method.clone(), method))
                .collect(),
        }
    }
}

impl VerificationMethodResolver for InMemoryVerificationMethodResolver {
    fn resolve(
        &self,
        verification_method: &str,
    ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
        self.methods
            .get(verification_method)
            .cloned()
            .ok_or(VerificationMethodResolutionError::Unavailable)
    }

    fn resolve_with_snapshot(
        &self,
        verification_method: &str,
    ) -> Result<ResolvedVerificationMethodSnapshot, VerificationMethodResolutionError> {
        let resolved = self
            .methods
            .get(verification_method)
            .cloned()
            .ok_or(VerificationMethodResolutionError::Unavailable)?;
        let snapshot_fingerprint = Some(Self::method_snapshot_fingerprint(&resolved));
        Ok(ResolvedVerificationMethodSnapshot {
            resolved,
            snapshot_fingerprint,
        })
    }

    fn snapshot_fingerprint(&self) -> Option<String> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:verification-method-resolver-snapshot:v1\n");
        for (method_id, method) in &self.methods {
            hasher.update(&(method_id.len() as u64).to_be_bytes());
            hasher.update(method_id.as_bytes());
            hasher.update(&method.verifying_key.to_bytes());
            hasher.update(&[match method.status {
                VerificationMethodStatus::Active => 0,
                VerificationMethodStatus::Revoked => 1,
                VerificationMethodStatus::Expired => 2,
                VerificationMethodStatus::Unknown => 3,
            }]);
            let mut purposes = method.allowed_proof_purposes.clone();
            purposes.sort();
            purposes.dedup();
            hasher.update(&(purposes.len() as u64).to_be_bytes());
            for purpose in &purposes {
                hasher.update(&(purpose.len() as u64).to_be_bytes());
                hasher.update(purpose.as_bytes());
            }
        }
        Some(hasher.finalize().to_hex().to_string())
    }

    fn snapshot_fingerprint_for(&self, verification_method: &str) -> Option<String> {
        self.methods
            .get(verification_method)
            .map(Self::method_snapshot_fingerprint)
    }
}

impl InMemoryVerificationMethodResolver {
    fn method_snapshot_fingerprint(method: &ResolvedVerificationMethod) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:verification-method-resolution-snapshot:v2\n");
        hasher.update(&(method.verification_method.len() as u64).to_be_bytes());
        hasher.update(method.verification_method.as_bytes());
        hasher.update(&method.verifying_key.to_bytes());
        hasher.update(&[match method.status {
            VerificationMethodStatus::Active => 0,
            VerificationMethodStatus::Revoked => 1,
            VerificationMethodStatus::Expired => 2,
            VerificationMethodStatus::Unknown => 3,
        }]);
        let mut purposes = method.allowed_proof_purposes.clone();
        purposes.sort();
        purposes.dedup();
        hasher.update(&(purposes.len() as u64).to_be_bytes());
        for purpose in &purposes {
            hasher.update(&(purpose.len() as u64).to_be_bytes());
            hasher.update(purpose.as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }
}

#[derive(Debug, Clone)]
pub struct Ed25519ReceiptVerifier {
    verification_method: String,
    verifying_key: VerifyingKey,
    expected_proof_purpose: Option<String>,
    expected_domain: Option<String>,
    expected_challenge: Option<String>,
    now_unix_ns: i128,
    policy_inputs: VerificationPolicyInputs,
    environment_identity: VerifierEnvironmentIdentity,
    resolution_snapshot_fingerprint: Option<String>,
}

impl Ed25519ReceiptVerifier {
    pub fn new(
        verification_method: impl Into<String>,
        verifying_key: VerifyingKey,
        now_unix_ns: i128,
    ) -> Self {
        Self {
            verification_method: verification_method.into(),
            verifying_key,
            expected_proof_purpose: None,
            expected_domain: None,
            expected_challenge: None,
            now_unix_ns,
            policy_inputs: VerificationPolicyInputs {
                policy_version: POLICY_VERSION,
                cryptosuite: CRYPTOSUITE,
                expected_proof_purpose: None,
                expected_domain: None,
                expected_challenge_fingerprint: None,
                require_active_verification_method: true,
            },
            environment_identity: VerifierEnvironmentIdentity::new("unspecified"),
            resolution_snapshot_fingerprint: None,
        }
    }

    pub fn with_expected_proof_purpose(mut self, purpose: impl Into<String>) -> Self {
        self.expected_proof_purpose = Some(purpose.into());
        self.policy_inputs.expected_proof_purpose = self.expected_proof_purpose.clone();
        self
    }

    pub fn with_expected_domain(mut self, domain: impl Into<String>) -> Self {
        self.expected_domain = Some(domain.into());
        self.policy_inputs.expected_domain = self.expected_domain.clone();
        self
    }

    pub fn with_expected_challenge(mut self, challenge: impl Into<String>) -> Self {
        self.expected_challenge = Some(challenge.into());
        self.policy_inputs.expected_challenge_fingerprint = self.expected_challenge.as_deref().map(|v| blake3::hash(v.as_bytes()).to_hex().to_string());
        self
    }

    pub fn with_environment_identity(mut self, identity: VerifierEnvironmentIdentity) -> Self {
        self.environment_identity = identity;
        self
    }

    pub fn with_policy_version(mut self, version: &'static str) -> Self {
        self.policy_inputs.policy_version = version;
        self
    }

    /// Bind the exact resolver snapshot used for verification.
    /// The snapshot is represented by a content-derived fingerprint so the
    /// durable report does not depend on mutable remote resolver state.
    pub fn with_resolution_snapshot_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.resolution_snapshot_fingerprint = Some(fingerprint.into());
        self
    }

    /// Verify the full Evidence Fabric procedure without a resolver.
    ///
    /// This entry point fails closed once resolver-backed verification is required:
    /// a locally supplied key is not evidence of current method lifecycle or
    /// proof-purpose authorization. Use verify_detached_proof() for narrow
    /// cryptographic validation with the pinned key, or verify_with_resolver()
    /// for the complete appraisal procedure.
    pub fn verify(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
    ) -> ReceiptAttestationVerificationOutcome {
        self.verify_report(envelope, receipt).outcome
    }

    /// Produce a full Evidence Fabric report.
    ///
    /// This resolver-free entry point deliberately fails closed after the
    /// structural/receipt/temporal/cryptosuite preflight: the configured key alone
    /// is not evidence that the verification method is currently active or authorized.
    /// Use verify_with_resolver_report() for a terminal full-procedure result.
    pub fn verify_report(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
    ) -> ReceiptAttestationVerificationReport {
        let resolver = UnresolvedVerificationMethodResolver;
        self.verify_with_resolver_report(envelope, receipt, &resolver)
    }

    /// Verify the detached Ed25519 proof with the verifier's pinned key.
    ///
    /// This is intentionally narrower than Evidence Fabric evaluation: it can establish
    /// the cryptographic proof under this verifier's configured policy inputs, but it does
    /// not establish resolver-backed method lifecycle or authorization semantics.
    pub fn verify_detached_proof(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
    ) -> ReceiptAttestationVerificationOutcome {
        self.verify_with_resolved_key_report(
            envelope,
            receipt,
            &self.verification_method,
            &self.verifying_key,
        )
        .outcome
    }

    /// Verify using an application-controlled resolver.
    ///
    /// This adds verification-method resolution, lifecycle status, and proof-purpose
    /// authorization without allowing this crate to fetch or trust remote key
    /// material implicitly.
    pub fn verify_with_resolver<R: VerificationMethodResolver>(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
        resolver: &R,
    ) -> ReceiptAttestationVerificationOutcome {
        self.verify_with_resolver_report(envelope, receipt, resolver).outcome
    }

    pub fn verify_with_resolver_report<R: VerificationMethodResolver>(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
        resolver: &R,
    ) -> ReceiptAttestationVerificationReport {
        let mut report = self.verify_with_resolver_report_inner(envelope, receipt, resolver);
        if report.resolution_snapshot_fingerprint.is_none()
            && report.resolved_verification_method.is_none()
        {
            report.resolution_snapshot_fingerprint = self.resolution_snapshot_fingerprint.clone();
        }
        report
    }

    fn verify_with_resolver_report_inner<R: VerificationMethodResolver>(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
        resolver: &R,
    ) -> ReceiptAttestationVerificationReport {
        if envelope.validate().is_err() {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::InvalidEnvelope,
                EvaluationCheck::EnvelopeStructuralValidation,
                receipt.fingerprint(),
                None,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }
        if !envelope.verify_against_receipt(receipt) {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::ReceiptCommitmentMismatch,
                EvaluationCheck::ReceiptCommitment,
                receipt.fingerprint(),
                None,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }
        match envelope.temporal_status_at(self.now_unix_ns) {
            ReceiptAttestationTemporalStatus::NotYetValid => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::NotYetValid,
                    EvaluationCheck::TemporalValidity,
                    receipt.fingerprint(),
                    None,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
            ReceiptAttestationTemporalStatus::Expired => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::Expired,
                    EvaluationCheck::TemporalValidity,
                    receipt.fingerprint(),
                    None,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
            ReceiptAttestationTemporalStatus::Valid => {}
        }
        if envelope.cryptosuite.as_deref() != Some(CRYPTOSUITE) {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::CryptosuiteMismatch,
                EvaluationCheck::CryptosuiteConformance,
                receipt.fingerprint(),
                None,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }

        let Some(method) = envelope.verification_method.as_deref() else {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable,
                EvaluationCheck::VerificationMethodResolution,
                receipt.fingerprint(),
                None,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        };
        let resolved_snapshot = match resolver.resolve_with_snapshot(method) {
            Ok(resolved_snapshot) => resolved_snapshot,
            Err(_) => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable,
                    EvaluationCheck::VerificationMethodResolution,
                    receipt.fingerprint(),
                    Some(method.to_string()),
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
        };
        if resolved_snapshot
            .snapshot_fingerprint
            .as_deref()
            .is_some_and(|snapshot| snapshot.trim().is_empty())
        {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable,
                EvaluationCheck::VerificationMethodResolution,
                receipt.fingerprint(),
                Some(method.to_string()),
                self.now_unix_ns,
                self.policy_inputs.clone(),
                self.environment_identity.clone(),
            );
        }
        let resolved = resolved_snapshot.resolved;
        if resolved.verification_method != method {
            return ReceiptAttestationVerificationReport::failed_with_resolution_snapshot(
                ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable,
                EvaluationCheck::VerificationMethodResolution,
                receipt.fingerprint(),
                Some(method.to_string()),
                resolved_snapshot.snapshot_fingerprint.clone(),
                self.now_unix_ns,
                self.policy_inputs.clone(),
                self.environment_identity.clone(),
            );
        }
        match resolved.status {
            VerificationMethodStatus::Active => {}
            VerificationMethodStatus::Revoked => {
                return ReceiptAttestationVerificationReport::failed_with_resolution_snapshot(
                ReceiptAttestationVerificationOutcome::VerificationMethodRevoked,
                EvaluationCheck::VerificationMethodLifecycle,
                receipt.fingerprint(),
                Some(method.to_string()),
                resolved_snapshot.snapshot_fingerprint.clone(),
                self.now_unix_ns,
                self.policy_inputs.clone(),
                self.environment_identity.clone(),
            );
            }
            VerificationMethodStatus::Expired => {
                return ReceiptAttestationVerificationReport::failed_with_resolution_snapshot(
                ReceiptAttestationVerificationOutcome::VerificationMethodExpired,
                EvaluationCheck::VerificationMethodLifecycle,
                receipt.fingerprint(),
                Some(method.to_string()),
                resolved_snapshot.snapshot_fingerprint.clone(),
                self.now_unix_ns,
                self.policy_inputs.clone(),
                self.environment_identity.clone(),
            );
            }
            VerificationMethodStatus::Unknown => {
                return ReceiptAttestationVerificationReport::failed_with_resolution_snapshot(
                ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable,
                EvaluationCheck::VerificationMethodResolution,
                receipt.fingerprint(),
                Some(method.to_string()),
                resolved_snapshot.snapshot_fingerprint.clone(),
                self.now_unix_ns,
                self.policy_inputs.clone(),
                self.environment_identity.clone(),
            );
            }
        }
        if !resolved.is_authorized_for(&envelope.proof_purpose) {
            return ReceiptAttestationVerificationReport::failed_with_resolution_snapshot(
                ReceiptAttestationVerificationOutcome::ProofPurposeUnauthorized,
                EvaluationCheck::ProofPurposeAuthorization,
                receipt.fingerprint(),
                Some(method.to_string()),
                resolved_snapshot.snapshot_fingerprint.clone(),
                self.now_unix_ns,
                self.policy_inputs.clone(),
                self.environment_identity.clone(),
            );
        }
        let mut report =
            self.verify_with_resolved_key_report(envelope, receipt, method, &resolved.verifying_key);
        // The resolver's paired result is authoritative once resolution occurs.
        // In particular, a paired None means that no resolver snapshot was supplied;
        // do not silently substitute a later verifier-side observation.
        report.resolution_snapshot_fingerprint = resolved_snapshot.snapshot_fingerprint;
        report
    }

    fn verify_with_resolved_key(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
        verification_method: &str,
        verifying_key: &VerifyingKey,
    ) -> ReceiptAttestationVerificationOutcome {
        self.verify_with_resolved_key_report(
            envelope, receipt, verification_method, verifying_key,
        ).outcome
    }

    fn verify_with_resolved_key_report(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
        verification_method: &str,
        verifying_key: &VerifyingKey,
    ) -> ReceiptAttestationVerificationReport {
        let method = Some(verification_method.to_string());
        let fingerprint = receipt.fingerprint();

        if envelope.validate().is_err() {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::InvalidEnvelope,
                EvaluationCheck::EnvelopeStructuralValidation,
                fingerprint,
                method,
                self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }
        if !envelope.verify_against_receipt(receipt) {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::ReceiptCommitmentMismatch,
                EvaluationCheck::ReceiptCommitment,
                fingerprint,
                method,
                self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }

        match envelope.temporal_status_at(self.now_unix_ns) {
            ReceiptAttestationTemporalStatus::NotYetValid => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::NotYetValid,
                    EvaluationCheck::TemporalValidity,
                    fingerprint,
                method,
                self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
            ReceiptAttestationTemporalStatus::Expired => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::Expired,
                    EvaluationCheck::TemporalValidity,
                    fingerprint,
                method,
                self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
            ReceiptAttestationTemporalStatus::Valid => {}
        }

        if envelope.cryptosuite.as_deref() != Some(CRYPTOSUITE) {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::CryptosuiteMismatch,
                EvaluationCheck::CryptosuiteConformance,
                fingerprint,
                method,
                self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }
        if envelope.verification_method.as_deref() != Some(verification_method) {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::VerificationMethodMismatch,
                EvaluationCheck::VerificationMethodResolution,
                fingerprint,
                method,
                self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }
        if let Some(expected) = &self.expected_proof_purpose {
            if envelope.proof_purpose != *expected {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::ProofPurposeMismatch,
                    EvaluationCheck::ProofPolicyConformance,
                    fingerprint,
                method,
                self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
        }
        if let Some(expected) = &self.expected_domain {
            if envelope.domain.as_deref() != Some(expected.as_str()) {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::DomainMismatch,
                    EvaluationCheck::ProofPolicyConformance,
                    fingerprint,
                method,
                self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
        }
        if let Some(expected) = &self.expected_challenge {
            if envelope.challenge.as_deref() != Some(expected.as_str()) {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::ChallengeMismatch,
                    EvaluationCheck::ProofPolicyConformance,
                    fingerprint,
                method,
                self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
        }

        let Some(proof) = envelope.proof.as_deref() else {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::MissingProof,
                EvaluationCheck::CryptographicProof,
                fingerprint,
                method,
                self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        };
        let Ok(proof_bytes) = <[u8; 64]>::try_from(proof) else {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::InvalidProofEncoding,
                EvaluationCheck::CryptographicProof,
                fingerprint,
                method,
                self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        };
        let signature = Signature::from_bytes(&proof_bytes);

        match verifying_key.verify(&envelope.canonical_payload_bytes(), &signature) {
            Ok(()) => ReceiptAttestationVerificationReport::passed(
                fingerprint,
                method,
                self.now_unix_ns,
                self.policy_inputs.clone(),
                self.environment_identity.clone(),
            ),
            Err(_) => ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::InvalidSignature,
                EvaluationCheck::CryptographicProof,
                fingerprint,
                method,
                self.now_unix_ns,
                self.policy_inputs.clone(),
                self.environment_identity.clone(),
            ),
        }
    }
}

/// Attach a detached Ed25519 proof. Key publication, authorization, rotation,
/// and revocation remain outside this adapter.
pub fn sign_envelope(
    envelope: &mut ReceiptAttestationEnvelope,
    signing_key: &SigningKey,
    verification_method: impl Into<String>,
) -> Result<(), SignEnvelopeError> {
    envelope.cryptosuite = Some(CRYPTOSUITE.to_string());
    envelope.verification_method = Some(verification_method.into());
    envelope.validate().map_err(|_| SignEnvelopeError::InvalidEnvelope)?;
    let signature = signing_key.sign(&envelope.canonical_payload_bytes());
    envelope.proof = Some(signature.to_bytes().to_vec());
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SignEnvelopeError {
    #[error("receipt attestation envelope is structurally invalid")]
    InvalidEnvelope,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::observation_fabric::{
        AssetRef, DisclosurePolicy, Observation, ObservationGraph, ObservationModality,
        ObservationProvenance, ObservationQuality, ObservationTime, ProvenanceCoverage,
        ProvenanceVerification, SensorIdentity,
    };

    fn receipt() -> IndependenceVerificationReceipt {
        let base = Observation {
            id: "obs-a".into(),
            modality: ObservationModality::Optical,
            time: ObservationTime { observed_at_unix_ns: 100, time_uncertainty_ns: 1 },
            location: None,
            feature_of_interest_id: None,
            quality: ObservationQuality {
                confidence: 0.9,
                measurement_uncertainty: None,
                calibrated: true,
            },
            provenance: ObservationProvenance {
                source: SensorIdentity::new("sensor-a"),
                verification: ProvenanceVerification::Unverified,
                coverage: ProvenanceCoverage::Complete,
                attestation_id: None,
                acquired_by: None,
                parent_observation_ids: Vec::new(),
                processing_fingerprint: None,
                processing_activity: None,
            },
            asset: Some(AssetRef::blake3(b"a")),
            disclosure: DisclosurePolicy::restricted(),
        };
        let mut other = base.clone();
        other.id = "obs-b".into();
        other.provenance.source = SensorIdentity::new("sensor-b");
        let graph = ObservationGraph { observations: vec![base, other], relations: vec![] };
        let assessment = graph.assess_independence_detailed("obs-a", "obs-b").expect("assessment");
        IndependenceVerificationReceipt::from_assessment(&assessment)
    }

    fn resolved_report(
        verifier: &Ed25519ReceiptVerifier,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
    ) -> ReceiptAttestationVerificationReport {
        let method = envelope
            .verification_method
            .clone()
            .expect("test envelope has a verification method");
        let resolver = InMemoryVerificationMethodResolver::new([ResolvedVerificationMethod {
            verification_method: method,
            verifying_key: verifier.verifying_key,
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec![envelope.proof_purpose.clone()],
        }]);
        verifier.verify_with_resolver_report(envelope, receipt, &resolver)
    }

    fn resolved_report(
        verifier: &Ed25519ReceiptVerifier,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
    ) -> ReceiptAttestationVerificationReport {
        let method = envelope
            .verification_method
            .clone()
            .expect("test envelope has a verification method");
        let resolver = InMemoryVerificationMethodResolver::new([ResolvedVerificationMethod {
            verification_method: method,
            verifying_key: verifier.verifying_key.clone(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec![envelope.proof_purpose.clone()],
        }]);
        verifier.verify_with_resolver_report(envelope, receipt, &resolver)
    }

    fn envelope_and_key() -> (
        ReceiptAttestationEnvelope,
        SigningKey,
        IndependenceVerificationReceipt,
    ) {
        let receipt = receipt();
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let mut envelope = ReceiptAttestationEnvelope::from_receipt(
            &receipt, "attester-a", "observation-independence", 100,
        );
        envelope.expires_at_unix_ns = Some(200);
        sign_envelope(&mut envelope, &signing_key, "did:example:attester-a#key-1")
            .expect("sign");
        (envelope, signing_key, receipt)
    }

    #[test]
    fn signs_and_verifies_detached_receipt() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        assert_eq!(
            verifier.verify_detached_proof(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::Verified
        );
    }

    #[test]
    fn payload_mutation_invalidates_signature() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        assert_eq!(
            verifier.verify_detached_proof(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::Verified
        );
        envelope.challenge = Some("challenge".into());
        assert_eq!(
            verifier.verify_detached_proof(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::InvalidSignature
        );
    }

    #[test]
    fn temporal_and_key_policy_failures_are_distinct() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            200,
        );
        assert_eq!(
            verifier.verify_detached_proof(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::Expired
        );

        envelope.expires_at_unix_ns = Some(300);
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:wrong#key-1",
            signing_key.verifying_key(),
            150,
        );
        assert_eq!(
            verifier.verify_detached_proof(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::VerificationMethodMismatch
        );
    }

    #[test]
    fn resolver_separates_key_lifecycle_and_authorization() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let method = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            signing_key.verifying_key(),
            150,
        );
        assert_eq!(
            verifier.verify_with_resolver(&envelope, &receipt, &resolver),
            ReceiptAttestationVerificationOutcome::Verified
        );

        let revoked = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Revoked,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([revoked]);
        assert_eq!(
            verifier.verify_with_resolver(&envelope, &receipt, &resolver),
            ReceiptAttestationVerificationOutcome::VerificationMethodRevoked
        );

        let unauthorized = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["authentication".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([unauthorized]);
        assert_eq!(
            verifier.verify_with_resolver(&envelope, &receipt, &resolver),
            ReceiptAttestationVerificationOutcome::ProofPurposeUnauthorized
        );
    }

    #[test]
    fn resolver_mode_rejects_invalid_envelope_before_resolution() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope.attester_id.clear();
        let resolver = InMemoryVerificationMethodResolver::new([]);
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            signing_key.verifying_key(),
            150,
        );

        assert_eq!(
            verifier.verify_with_resolver(&envelope, &receipt, &resolver),
            ReceiptAttestationVerificationOutcome::InvalidEnvelope
        );
    }

    #[test]
    fn resolver_uses_resolved_key_not_embedded_verifier_key() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let unrelated_key = SigningKey::from_bytes(&[9u8; 32]);
        let method = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            unrelated_key.verifying_key(),
            150,
        );

        assert_eq!(
            verifier.verify_with_resolver(&envelope, &receipt, &resolver),
            ReceiptAttestationVerificationOutcome::Verified
        );
    }

    #[test]
    fn proof_purpose_domain_and_challenge_are_policy_checks() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope.domain = Some("example.org".into());
        envelope.challenge = Some("c1".into());
        sign_envelope(&mut envelope, &signing_key, "did:example:attester-a#key-1")
            .expect("resign");
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .with_expected_proof_purpose("observation-independence")
        .with_expected_domain("example.org")
        .with_expected_challenge("c1");
        assert_eq!(
            verifier.verify_detached_proof(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::Verified
        );
    }

    #[test]
    fn verification_report_preserves_stage_boundaries() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let report = verifier.verify_report(&envelope, &receipt);
        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable
        );
        assert_eq!(report.structural_validation, VerificationStage::Passed);
        assert_eq!(report.receipt_commitment, VerificationStage::Passed);
        assert_eq!(report.temporal_validity, VerificationStage::Passed);
        assert_eq!(report.cryptosuite, VerificationStage::Passed);
        assert_eq!(
            report.verification_method,
            VerificationStage::Failed(
                ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable
            )
        );
        assert_eq!(report.lifecycle, VerificationStage::NotEvaluated);
        assert_eq!(report.proof_purpose_authorization, VerificationStage::NotEvaluated);
        assert_eq!(report.proof_policy, VerificationStage::NotEvaluated);
        assert_eq!(report.cryptographic_proof, VerificationStage::NotEvaluated);
        assert_eq!(
            report.execution_trace.executed_check_ids(),
            vec![
                "envelope-structural-validation",
                "receipt-commitment",
                "temporal-validity",
                "cryptosuite-conformance",
                "verification-method-resolution",
            ]
        );
        assert!(report.is_well_formed());
    }

    #[test]
    fn verification_report_marks_unexecuted_stages() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope.attester_id.clear();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let report = verifier.verify_report(&envelope, &receipt);
        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::InvalidEnvelope
        );
        assert_eq!(
            report.structural_validation,
            VerificationStage::Failed(ReceiptAttestationVerificationOutcome::InvalidEnvelope)
        );
        assert_eq!(report.receipt_commitment, VerificationStage::NotEvaluated);
        assert_eq!(report.cryptographic_proof, VerificationStage::NotEvaluated);
    }

    #[test]
    fn resolver_report_records_lifecycle_failure() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let method = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Revoked,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            signing_key.verifying_key(),
            150,
        );
        let report = verifier.verify_with_resolver_report(&envelope, &receipt, &resolver);
        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::VerificationMethodRevoked
        );
        assert_eq!(report.verification_method, VerificationStage::Passed);
        assert_eq!(
            report.lifecycle,
            VerificationStage::Failed(
                ReceiptAttestationVerificationOutcome::VerificationMethodRevoked
            )
        );
        assert_eq!(
            report.proof_policy,
            VerificationStage::NotEvaluated
        );
    }

    #[test]
    fn fingerprint_shape_validation_rejects_noncanonical_digests() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = verifier.verify_report(&envelope, &receipt);

        assert!(is_blake3_fingerprint(&report.receipt_fingerprint));
        assert!(report.is_well_formed());

        report.receipt_fingerprint = "not-a-fingerprint".into();
        assert!(!report.is_well_formed());

        let report = verifier.verify_report(&envelope, &receipt);
        let mut evaluation = report.to_evidence_evaluation();
        assert!(evaluation.is_well_formed());

        evaluation.subject_fingerprint = "not-a-fingerprint".into();
        assert!(!evaluation.is_well_formed());

        let mut evaluation = report.to_evidence_evaluation();
        evaluation.verification_report_fingerprint = "not-a-fingerprint".into();
        assert!(!evaluation.is_well_formed());

        let mut policy = report.policy_inputs.clone();
        policy.expected_challenge_fingerprint = Some("A".repeat(64));
        assert!(!policy.is_well_formed());
    }

    #[test]
    fn legacy_procedure_uses_frozen_check_sequence() {
        let procedure = EvaluationProcedure::attestation_ed25519_v1();

        assert_eq!(procedure.checks, LEGACY_EVALUATION_CHECKS_V1);
        assert_eq!(procedure.checks.len(), 9);
        assert_eq!(
            procedure
                .checks
                .iter()
                .map(|check| check.id())
                .collect::<Vec<_>>(),
            vec![
                "envelope-structural-validation",
                "receipt-commitment",
                "temporal-validity",
                "cryptosuite-conformance",
                "verification-method-resolution",
                "verification-method-lifecycle",
                "proof-purpose-authorization",
                "proof-policy-conformance",
                "cryptographic-proof",
            ]
        );
    }
    #[test]
    fn policy_self_validation_rejects_unsupported_lifecycle_mode() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let mut policy = report.policy_inputs;

        assert!(policy.is_well_formed());
        policy.require_active_verification_method = false;
        assert!(!policy.is_well_formed());
    }

    #[test]
    fn custom_policy_version_remains_a_valid_policy_identity() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .with_policy_version("symthaea-observation-verification-policy-v2")
        .verify_report(&envelope, &receipt);

        assert!(report.policy_inputs.is_well_formed());
        assert!(report.has_consistent_identity_bindings());
        assert_eq!(
            report.policy_inputs.policy_version,
            "symthaea-observation-verification-policy-v2"
        );
    }

    #[test]
    fn verification_policy_self_validation_rejects_semantic_rebinding() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let mut report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);

        assert!(report.policy_inputs.is_well_formed());
        report.policy_inputs.cryptosuite = "attacker-cryptosuite";
        report.policy_fingerprint = report.policy_inputs.fingerprint();
        assert!(!report.policy_inputs.is_well_formed());
        assert!(!report.has_consistent_identity_bindings());
    }

    #[test]
    fn verifier_environment_self_validation_rejects_semantic_rebinding() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let mut report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);

        assert!(report.environment_identity.is_well_formed());
        report.environment_identity.implementation_id = "attacker-verifier";
        report.environment_fingerprint = report.environment_identity.fingerprint();
        assert!(!report.environment_identity.is_well_formed());
        assert!(!report.has_consistent_identity_bindings());
    }

    #[test]
    fn verification_report_binds_policy_environment_and_evaluation_time() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .with_expected_proof_purpose("observation-independence")
        .with_expected_domain("mycelix")
        .with_environment_identity(
            VerifierEnvironmentIdentity::new("build-sha-abc")
                .with_runtime_profile("portable"),
        );
        let report = verifier.verify_report(&envelope, &receipt);
        assert_eq!(report.evaluated_at_unix_ns, 150);
        assert_eq!(report.policy_inputs.expected_proof_purpose.as_deref(), Some("observation-independence"));
        assert_eq!(report.policy_inputs.expected_domain.as_deref(), Some("mycelix"));
        assert_eq!(report.environment_identity.build_fingerprint, "build-sha-abc");
        assert_eq!(report.policy_fingerprint, report.policy_inputs.fingerprint());
        assert_eq!(report.environment_fingerprint, report.environment_identity.fingerprint());
    }

    #[test]
    fn evidence_evaluation_rejects_mutated_policy_or_environment_inputs() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();
        assert!(evaluation.is_consistent_with_report(&report));

        let mut changed_policy = report.clone();
        changed_policy.policy_inputs.expected_domain = Some("attacker-domain".into());
        assert!(!evaluation.is_consistent_with_report(&changed_policy));

        let mut changed_environment = report.clone();
        changed_environment.environment_identity.build_fingerprint =
            "different-build".into();
        assert!(!evaluation.is_consistent_with_report(&changed_environment));
    }

    #[test]
    fn verification_report_fingerprint_changes_when_policy_or_environment_changes() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let base = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let policy = base.clone().with_expected_domain("mycelix");
        let environment = base.clone().with_environment_identity(
            VerifierEnvironmentIdentity::new("build-sha-abc")
        );
        assert_ne!(
            base.verify_report(&envelope, &receipt).fingerprint(),
            policy.verify_report(&envelope, &receipt).fingerprint()
        );
        assert_ne!(
            base.verify_report(&envelope, &receipt).fingerprint(),
            environment.verify_report(&envelope, &receipt).fingerprint()
        );
    }

    #[test]
    fn in_memory_resolver_binds_automatic_snapshot() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let method = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            signing_key.verifying_key(),
            150,
        );
        let report = verifier.verify_with_resolver_report(&envelope, &receipt, &resolver);
        assert!(report.resolution_snapshot_fingerprint.is_some());

        let changed = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Revoked,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let changed_resolver = InMemoryVerificationMethodResolver::new([changed]);
        let changed_report =
            verifier.verify_with_resolver_report(&envelope, &receipt, &changed_resolver);
        assert_ne!(
            report.resolution_snapshot_fingerprint,
            changed_report.resolution_snapshot_fingerprint
        );
    }

    #[test]
    fn verification_report_canonical_encoding_uses_stable_outcome_tags() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = verifier.verify_report(&envelope, &receipt);
        let verified = report.canonical_bytes();
        assert_eq!(verified[verified.len() - 1], 0);

        report.outcome = ReceiptAttestationVerificationOutcome::InvalidSignature;
        report.cryptographic_proof =
            VerificationStage::Failed(ReceiptAttestationVerificationOutcome::InvalidSignature);
        let invalid = report.canonical_bytes();
        assert!(invalid.windows(2).any(|pair| pair == [1, 12]));
    }

    #[test]
    fn resolver_snapshot_treats_authorized_purposes_as_a_set() {
        let (_, signing_key, _) = envelope_and_key();
        let first = InMemoryVerificationMethodResolver::new([ResolvedVerificationMethod {
            verification_method: "did:example:key".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec![
                "z-purpose".into(),
                "a-purpose".into(),
                "z-purpose".into(),
            ],
        }]);
        let second = InMemoryVerificationMethodResolver::new([ResolvedVerificationMethod {
            verification_method: "did:example:key".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec![
                "a-purpose".into(),
                "z-purpose".into(),
            ],
        }]);
        assert_eq!(first.snapshot_fingerprint(), second.snapshot_fingerprint());
    }

    #[test]
    fn resolver_snapshot_is_scoped_to_requested_method() {
        let (_, signing_key, _) = envelope_and_key();
        let method_a = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let method_b = ResolvedVerificationMethod {
            verification_method: "did:example:attester-b#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };

        let resolver = InMemoryVerificationMethodResolver::new([method_a.clone(), method_b.clone()]);
        let base = resolver
            .snapshot_fingerprint_for(&method_a.verification_method)
            .expect("method-a snapshot");

        let mut unrelated = method_b.clone();
        unrelated.status = VerificationMethodStatus::Revoked;
        let unrelated_changed =
            InMemoryVerificationMethodResolver::new([method_a.clone(), unrelated]);
        assert_eq!(
            unrelated_changed.snapshot_fingerprint_for(&method_a.verification_method),
            Some(base.clone())
        );

        let mut relevant = method_a;
        relevant.status = VerificationMethodStatus::Revoked;
        let relevant_changed = InMemoryVerificationMethodResolver::new([relevant, method_b]);
        assert_ne!(
            relevant_changed.snapshot_fingerprint_for("did:example:attester-a#key-1"),
            Some(base)
        );
    }

    #[test]
    fn resolve_with_snapshot_uses_scoped_snapshot_fingerprint() {
        let (_, signing_key, _) = envelope_and_key();
        let method_a = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let method_b = ResolvedVerificationMethod {
            verification_method: "did:example:attester-b#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus.Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver =
            InMemoryVerificationMethodResolver::new([method_a.clone(), method_b]);

        let resolved = resolver
            .resolve_with_snapshot(&method_a.verification_method)
            .expect("method-a resolution");

        assert_eq!(resolved.resolved, method_a);
        assert_eq!(
            resolved.snapshot_fingerprint,
            resolver.snapshot_fingerprint_for(&method_a.verification_method)
        );
        assert_ne!(
            resolved.snapshot_fingerprint,
            resolver.snapshot_fingerprint()
        );
    }

    #[test]
    fn resolver_snapshot_binds_key_and_authorization_state() {
        let (_, signing_key, _) = envelope_and_key();
        let alternate_key = SigningKey::from_bytes(&[0x42; 32]);

        let base = InMemoryVerificationMethodResolver::new([ResolvedVerificationMethod {
            verification_method: "did:example:key".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        }]);
        let changed_key = InMemoryVerificationMethodResolver::new([ResolvedVerificationMethod {
            verification_method: "did:example:key".into(),
            verifying_key: alternate_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        }]);
        let changed_authorization = InMemoryVerificationMethodResolver::new([ResolvedVerificationMethod {
            verification_method: "did:example:key".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["different-purpose".into()],
        }]);

        assert_ne!(base.snapshot_fingerprint(), changed_key.snapshot_fingerprint());
        assert_ne!(
            base.snapshot_fingerprint(),
            changed_authorization.snapshot_fingerprint()
        );
    }

    #[test]
    fn default_resolve_with_snapshot_refuses_unpaired_snapshot_claims() {
        struct SplitBrainDefaultResolver {
            method: ResolvedVerificationMethod,
            snapshot_calls: std::cell::Cell<u32>,
        }

        impl VerificationMethodResolver for SplitBrainDefaultResolver {
            fn resolve(
                &self,
                _verification_method: &str,
            ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
                Ok(self.method.clone())
            }

            fn snapshot_fingerprint_for(&self, _verification_method: &str) -> Option<String> {
                self.snapshot_calls.set(self.snapshot_calls.get() + 1);
                Some("later-state".into())
            }
        }

        let (envelope, signing_key, _) = envelope_and_key();
        let resolver = SplitBrainDefaultResolver {
            method: ResolvedVerificationMethod {
                verification_method: envelope.attester_id,
                verifying_key: signing_key.verifying_key(),
                status: VerificationMethodStatus::Active,
                allowed_proof_purposes: vec!["observation-independence".into()],
            },
            snapshot_calls: std::cell::Cell::new(0),
        };

        let paired = resolver
            .resolve_with_snapshot("did:example:attester-a#key-1")
            .expect("compatibility resolution should succeed");

        assert_eq!(paired.snapshot_fingerprint, None);
        assert_eq!(resolver.snapshot_calls.get(), 0);
    }

    #[test]
    fn paired_resolution_none_snapshot_is_not_recombined_with_later_resolver_state() {
        struct NoSnapshotResolver {
            snapshot_calls: std::cell::Cell<u32>,
            method: ResolvedVerificationMethod,
        }

        impl VerificationMethodResolver for NoSnapshotResolver {
            fn resolve(
                &self,
                _verification_method: &str,
            ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
                Ok(self.method.clone())
            }

            fn resolve_with_snapshot(
                &self,
                _verification_method: &str,
            ) -> Result<ResolvedVerificationMethodSnapshot, VerificationMethodResolutionError> {
                Ok(ResolvedVerificationMethodSnapshot {
                    resolved: self.method.clone(),
                    snapshot_fingerprint: None,
                })
            }

            fn snapshot_fingerprint(&self) -> Option<String> {
                self.snapshot_calls
                    .set(self.snapshot_calls.get() + 1);
                Some("later-state".to_string())
            }
        }

        let (envelope, signing_key, receipt) = envelope_and_key();
        let method = ResolvedVerificationMethod {
            verification_method: envelope.attester_id.clone(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = NoSnapshotResolver {
            snapshot_calls: std::cell::Cell::new(0),
            method,
        };
        let verifier = Ed25519ReceiptVerifier::new(
            envelope.attester_id.clone(),
            signing_key.verifying_key(),
            150,
        )
        .with_resolution_snapshot_fingerprint("verifier-later-state");

        let report = verifier.verify_with_resolver_report(&envelope, &receipt, &resolver);

        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::Verified
        );
        assert_eq!(report.resolved_verification_method.as_deref(), Some(envelope.attester_id.as_str()));
        assert_eq!(report.resolution_snapshot_fingerprint, None);
        assert_eq!(resolver.snapshot_calls.get(), 0);
    }

    #[test]
    fn resolver_snapshot_is_not_attached_when_resolution_never_occurs() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope.attester_id.clear();

        let resolver = InMemoryVerificationMethodResolver::default();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = verifier.verify_with_resolver_report(&envelope, &receipt, &resolver);

        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::InvalidEnvelope
        );
        assert_eq!(report.resolved_verification_method, None);
        assert_eq!(report.resolution_snapshot_fingerprint, None);
    }

    #[test]
    fn resolver_snapshot_is_bound_to_the_resolution_result() {
        struct AtomicResolver {
            method: ResolvedVerificationMethod,
            snapshot: String,
        }

        impl VerificationMethodResolver for AtomicResolver {
            fn resolve(
                &self,
                _verification_method: &str,
            ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
                Ok(self.method.clone())
            }

            fn resolve_with_snapshot(
                &self,
                _verification_method: &str,
            ) -> Result<ResolvedVerificationMethodSnapshot, VerificationMethodResolutionError> {
                Ok(ResolvedVerificationMethodSnapshot {
                    resolved: self.method.clone(),
                    snapshot_fingerprint: Some(self.snapshot.clone()),
                })
            }

            fn snapshot_fingerprint(&self) -> Option<String> {
                Some("separately-observed-state".into())
            }
        }

        let (envelope, signing_key, receipt) = envelope_and_key();
        let resolver = AtomicResolver {
            method: ResolvedVerificationMethod {
                verification_method: "did:example:attester-a#key-1".into(),
                verifying_key: signing_key.verifying_key(),
                status: VerificationMethodStatus::Active,
                allowed_proof_purposes: vec!["observation-independence".into()],
            },
            snapshot: "atomic-snapshot".into(),
        };
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            signing_key.verifying_key(),
            150,
        );
        let report = verifier.verify_with_resolver_report(&envelope, &receipt, &resolver);

        assert_eq!(
            report.resolution_snapshot_fingerprint.as_deref(),
            Some("atomic-snapshot")
        );
    }

    #[test]
    fn verification_report_binds_resolution_snapshot() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let method = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            signing_key.verifying_key(),
            150,
        )
        .with_resolution_snapshot_fingerprint("resolver-snapshot-a");
        let report = verifier.verify_with_resolver_report(&envelope, &receipt, &resolver);
        assert_eq!(
            report.resolution_snapshot_fingerprint,
            resolver.snapshot_fingerprint_for(&envelope.attester_id)
        );

        // A paired resolver result is authoritative for the resolution event.
        // The verifier's compatibility snapshot must not overwrite a snapshot
        // returned by the resolver, because doing so would recombine two
        // independently observed states.
        let explicit_snapshot = verifier
            .clone()
            .with_resolution_snapshot_fingerprint("verifier-observed-later-state")
            .verify_with_resolver_report(&envelope, &receipt, &resolver);
        assert_eq!(
            explicit_snapshot.resolution_snapshot_fingerprint,
            report.resolution_snapshot_fingerprint
        );
        assert_eq!(explicit_snapshot.fingerprint(), report.fingerprint());
    }

    #[test]
    fn explicit_snapshot_is_preserved_when_resolution_never_occurs() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope.attester_id.clear();

        let resolver = InMemoryVerificationMethodResolver::default();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .with_resolution_snapshot_fingerprint("prebound-verifier-state");

        let report = verifier.verify_with_resolver_report(&envelope, &receipt, &resolver);

        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::InvalidEnvelope
        );
        assert_eq!(report.resolved_verification_method, None);
        assert_eq!(
            report.resolution_snapshot_fingerprint.as_deref(),
            Some("prebound-verifier-state")
        );
    }

    #[test]
    fn evidence_evaluation_rejects_mutated_resolution_snapshot() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let method = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            signing_key.verifying_key(),
            150,
        );
        let report = verifier.verify_with_resolver_report(&envelope, &receipt, &resolver);
        let mut evaluation = report.to_evidence_evaluation();

        assert!(evaluation.is_consistent_with_report(&report));

        evaluation.verification_report_fingerprint = {
            let mut mutated = report.clone();
            mutated.resolution_snapshot_fingerprint =
                Some("different-resolution-state".into());
            mutated.fingerprint()
        };

        assert!(!evaluation.is_consistent_with_report(&report));
    }

    #[test]
    fn verification_report_fingerprint_is_deterministic_and_binds_receipt() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let first = verifier.verify_report(&envelope, &receipt);
        let second = verifier.verify_report(&envelope, &receipt);
        assert_eq!(first.fingerprint(), second.fingerprint());
        assert_eq!(first.receipt_fingerprint, receipt.fingerprint());

        let other_receipt = {
            let mut r = receipt.clone();
            r.source_observation_id = "different-source".into();
            r
        };
        assert_ne!(first.receipt_fingerprint, other_receipt.fingerprint());
    }

    #[test]
    fn evidence_evaluation_boundary_marks_failed_checks_as_not_established() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope
            .proof
            .as_mut()
            .expect("signed envelope proof")[0] ^= 0x01;
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let report = resolved_report(&verifier, &envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();
        assert!(evaluation
            .boundary
            .not_established
            .contains(&EvaluationClaim::CryptographicProofValidity));
        assert!(evaluation
            .boundary
            .not_established
            .contains(&EvaluationClaim::AttestationAuthenticity));
        assert!(!evaluation
            .boundary
            .indeterminate
            .contains(&EvaluationClaim::CryptographicProofValidity));
    }

    #[test]
    fn evidence_evaluation_boundary_distinguishes_established_from_not_established() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();

        assert!(evaluation
            .boundary
            .established
            .contains(&EvaluationClaim::ReceiptIntegrity));
        assert!(evaluation
            .boundary
            .established
            .contains(&EvaluationClaim::CryptographicProofValidity));
        assert!(evaluation
            .boundary
            .not_established
            .contains(&EvaluationClaim::UnderlyingObservationTruth));
        assert!(evaluation
            .boundary
            .not_established
            .contains(&EvaluationClaim::SemanticValidity));
        assert!(evaluation.boundary.indeterminate.is_empty());
    }

    #[test]
    fn evidence_evaluation_boundary_changes_identity() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let mut first = report.to_evidence_evaluation();
        let mut second = first.clone();
        second.boundary.established.clear();
        assert_ne!(first.fingerprint(), second.fingerprint());
        first.boundary.not_established.reverse();
        assert_eq!(first.fingerprint(), report.to_evidence_evaluation().fingerprint());
    }

    #[test]
    fn evidence_evaluation_preserves_context_and_boundaries() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .with_environment_identity(VerifierEnvironmentIdentity::new("build-a"))
        .verify_report(&envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();
        assert_eq!(evaluation.subject_fingerprint, receipt.fingerprint());
        assert_eq!(evaluation.context_fingerprint, evaluation.context.fingerprint());
        assert_eq!(evaluation.verification_report_fingerprint, report.fingerprint());
        assert!(evaluation.execution_trace.is_well_formed());
        assert!(evaluation
            .boundary
            .not_established
            .contains(&EvaluationClaim::UnderlyingObservationTruth));
    }

    #[test]
    fn evidence_evaluation_context_changes_identity_without_changing_subject() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let base = report.to_evidence_evaluation();
        let changed = EvidenceEvaluation::from_report_with_context(
            &report,
            VerificationContext::from_report(&report)
                .with_trust_root_fingerprint("trust-root-a"),
        );
        assert_eq!(base.subject_fingerprint, changed.subject_fingerprint);
        assert_ne!(base.context_fingerprint, changed.context_fingerprint);
        assert_ne!(base.fingerprint(), changed.fingerprint());
    }

    #[test]
    fn evidence_evaluation_uses_v7_fingerprint_domain() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();
        assert!(evaluation
            .canonical_bytes()
            .starts_with(b"symthaea:evidence-evaluation:v7\n"));
    }

    #[test]
    fn evaluation_trace_empty_has_no_terminal_outcome() {
        let trace = EvaluationTrace {
            procedure_fingerprint:
                EvaluationProcedure::attestation_ed25519().fingerprint(),
            results: Vec::new(),
        };
        assert_eq!(trace.terminal_outcome(), None);
        assert!(!trace.is_well_formed());
    }

    #[test]
    fn evaluation_trace_malformed_all_pass_has_no_terminal_outcome() {
        let procedure = EvaluationProcedure::attestation_ed25519();
        let trace = EvaluationTrace {
            procedure_fingerprint: procedure.fingerprint(),
            results: procedure
                .checks
                .iter()
                .copied()
                .enumerate()
                .map(|(index, check)| EvaluationCheckResult {
                    sequence: index as u32 + 1,
                    check,
                    stage: VerificationStage::Passed,
                })
                .collect(),
        };
        assert!(!trace.is_well_formed());
        assert_eq!(trace.terminal_outcome(), None);
    }

    #[test]
    fn evaluation_trace_partial_success_has_no_terminal_outcome() {
        let procedure = EvaluationProcedure::attestation_ed25519();
        let trace = EvaluationTrace {
            procedure_fingerprint: procedure.fingerprint(),
            results: vec![EvaluationCheckResult {
                sequence: 0,
                check: EvaluationCheck::EnvelopeStructuralValidation,
                stage: VerificationStage::Passed,
            }],
        };
        assert_eq!(trace.terminal_outcome(), None);
        assert!(!trace.is_well_formed());
    }

    #[test]
    fn report_with_context_cannot_rebind_execution_identity() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);

        let mut supplied = VerificationContext::from_report(&report)
            .with_evaluator_identity_fingerprint("evaluator-1")
            .with_trust_root_fingerprint("trust-root-1")
            .with_authorization_policy_fingerprint("authz-policy-1");
        supplied.policy_fingerprint = "attacker-policy".into();
        supplied.verifier_id = "attacker-verifier";
        supplied.environment_fingerprint = "attacker-environment".into();
        supplied.procedure_fingerprint = "attacker-procedure".into();
        supplied.evaluated_at_unix_ns = 999;

        let evaluation = EvidenceEvaluation::from_report_with_context(&report, supplied);

        assert_eq!(
            evaluation.context.policy_fingerprint,
            report.policy_fingerprint
        );
        assert_eq!(
            evaluation.context.verifier_id,
            VERIFIER_IMPLEMENTATION_ID
        );
        assert_eq!(
            evaluation.context.environment_fingerprint,
            report.environment_fingerprint
        );
        assert_eq!(
            evaluation.context.procedure_fingerprint,
            report.procedure_fingerprint
        );
        assert_eq!(
            evaluation.context.evaluated_at_unix_ns,
            report.evaluated_at_unix_ns
        );
        assert_eq!(
            evaluation.context.evaluator_identity_fingerprint.as_deref(),
            Some("evaluator-1")
        );
        assert_eq!(
            evaluation.context.trust_root_fingerprint.as_deref(),
            Some("trust-root-1")
        );
        assert_eq!(
            evaluation.context.authorization_policy_fingerprint.as_deref(),
            Some("authz-policy-1")
        );
        assert!(evaluation.is_consistent_with_report(&report));
    }

    #[test]
    fn evidence_evaluation_self_validation_rejects_internal_mutation() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();

        assert!(evaluation.is_well_formed());

        let mut changed_context = evaluation.clone();
        changed_context.context_fingerprint = "tampered-context".into();
        assert!(!changed_context.is_well_formed());

        let mut empty_subject = evaluation.clone();
        empty_subject.subject_fingerprint.clear();
        assert!(!empty_subject.is_well_formed());

        let mut empty_report_binding = evaluation.clone();
        empty_report_binding.verification_report_fingerprint.clear();
        assert!(!empty_report_binding.is_well_formed());

        let mut recomputed_context = evaluation.clone();
        recomputed_context.context.verifier_id = "attacker-verifier";
        recomputed_context.context_fingerprint = recomputed_context.context.fingerprint();
        assert!(!recomputed_context.is_well_formed());

        let mut changed_trace = evaluation;
        changed_trace.execution_trace = EvaluationTrace::default();
        assert!(!changed_trace.is_well_formed());
    }

    #[test]
    fn verification_context_self_validation_rejects_semantic_identifier_mutation() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let mut context = VerificationContext::from_report(&report);

        assert!(context.is_well_formed());

        context.verifier_id = "attacker-verifier";
        assert_eq!(context.fingerprint(), VerificationContext {
            verifier_id: "attacker-verifier",
            ..context.clone()
        }.fingerprint());
        assert!(!context.is_well_formed());
    }

    #[test]
    fn evaluation_boundary_canonical_identity_preserves_duplicates() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let boundary = EvaluationBoundary::from_report(&report);

        let mut duplicate = boundary.clone();
        duplicate
            .not_established
            .push(EvaluationClaim::EvaluatorIndependence);

        assert!(!duplicate.is_well_formed());
        assert_ne!(duplicate.canonical_bytes(), boundary.canonical_bytes());

        let clean_evaluation = report.to_evidence_evaluation();
        let mut tampered_evaluation = clean_evaluation.clone();
        tampered_evaluation.boundary = duplicate;
        assert!(!tampered_evaluation.is_well_formed());
        assert_ne!(tampered_evaluation.fingerprint(), clean_evaluation.fingerprint());
    }

    #[test]
    fn evaluation_boundary_self_validation_rejects_omitted_claim() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let boundary = EvaluationBoundary::from_report(&report);

        assert!(boundary.is_well_formed());

        let mut tampered = boundary.clone();
        tampered
            .not_established
            .retain(|claim| *claim != EvaluationClaim::EvaluatorIndependence);
        assert!(!tampered.is_well_formed());
    }

    #[test]
    fn verification_report_self_validation_rejects_empty_resolved_method() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = verifier.verify_report(&envelope, &receipt);

        assert!(report.is_well_formed());
        report.resolved_verification_method = Some(String::new());
        assert!(!report.is_well_formed());

        report.verification_method = VerificationStage::Failed(
            ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable,
        );
        assert!(!report.is_well_formed());
        report.resolved_verification_method = Some(
            "did:example:attester-a#key-1".into(),
        );
        assert!(report.is_well_formed());
    }

    #[test]
    fn verification_report_self_validation_rejects_empty_receipt_identity() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = verifier.verify_report(&envelope, &receipt);

        assert!(report.is_well_formed());
        report.receipt_fingerprint.clear();
        assert!(!report.is_well_formed());
    }

    #[test]
    fn verification_report_self_validation_rejects_empty_resolution_snapshot() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            signing_key.verifying_key(),
            150,
        );
        let method = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let mut report = verifier.verify_with_resolver_report(&envelope, &receipt, &resolver);

        assert!(report.is_well_formed());
        report.resolution_snapshot_fingerprint = Some(String::new());
        assert!(!report.is_well_formed());
    }

    #[test]
    fn from_report_with_context_cannot_rebind_execution_identity() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);

        let mut supplied = VerificationContext::from_report(&report);
        supplied.verifier_id = "attacker-verifier";
        supplied.procedure_id = "attacker-procedure";
        supplied.policy_fingerprint = "attacker-policy".into();
        supplied.environment_fingerprint = "attacker-environment".into();
        supplied.evaluated_at_unix_ns = 999;
        supplied.evaluator_identity_fingerprint = Some("evaluator-a".into());

        let evaluation = EvidenceEvaluation::from_report_with_context(&report, supplied);
        let expected = VerificationContext::from_report(&report);

        assert_eq!(evaluation.context.verifier_id, expected.verifier_id);
        assert_eq!(evaluation.context.procedure_id, expected.procedure_id);
        assert_eq!(
            evaluation.context.policy_fingerprint,
            expected.policy_fingerprint
        );
        assert_eq!(
            evaluation.context.environment_fingerprint,
            expected.environment_fingerprint
        );
        assert_eq!(
            evaluation.context.evaluated_at_unix_ns,
            expected.evaluated_at_unix_ns
        );
        assert_eq!(
            evaluation.context.evaluator_identity_fingerprint.as_deref(),
            Some("evaluator-a")
        );
        assert!(evaluation.is_consistent_with_report(&report));
    }
    #[test]
    fn verification_context_self_validation_rejects_empty_optional_identities() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let base = VerificationContext::from_report(&report);

        for context in [
            VerificationContext {
                evaluator_identity_fingerprint: Some(String::new()),
                ..base.clone()
            },
            VerificationContext {
                trust_root_fingerprint: Some(String::new()),
                ..base.clone()
            },
            VerificationContext {
                authorization_policy_fingerprint: Some(String::new()),
                ..base.clone()
            },
        ] {
            assert!(!context.is_well_formed());
        }
    }

    #[test]
    fn verification_context_matches_report_rejects_malformed_report() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let context = VerificationContext::from_report(&report);

        assert!(context.matches_report(&report));

        let mut malformed_report = report;
        malformed_report.structural_validation = VerificationStage::Failed(
            ReceiptAttestationVerificationOutcome::InvalidEnvelope,
        );

        assert!(!malformed_report.is_well_formed());
        assert!(!context.matches_report(&malformed_report));
    }

    #[test]
    fn evidence_evaluation_consistency_binds_trace_and_outcome() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();
        assert!(evaluation.is_consistent_with_report(&report));
        assert!(evaluation.boundary.is_well_formed());

        let mut tampered = evaluation.clone();
        tampered.outcome =
            ReceiptAttestationVerificationOutcome::InvalidSignature;
        assert!(!tampered.is_consistent_with_report(&report));
    }

    #[test]
    fn evidence_boundary_tracks_structural_validity_separately() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope.attester_id.clear();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();
        assert!(evaluation
            .boundary
            .not_established
            .contains(&EvaluationClaim::EnvelopeStructuralValidity));
        assert!(evaluation
            .boundary
            .indeterminate
            .contains(&EvaluationClaim::ReceiptIntegrity));
        assert!(evaluation.boundary.is_well_formed());
    }

    #[test]
    fn verification_report_captures_execution_trace_at_execution_time() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope
            .proof
            .as_mut()
            .expect("signed envelope proof")[0] ^= 0x01;
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);

        assert!(report.execution_trace.is_well_formed());
        assert!(report.execution_trace.matches_report(&report));
        assert_eq!(
            report.execution_trace.terminal_outcome(),
            Some(ReceiptAttestationVerificationOutcome::InvalidSignature)
        );

        let evaluation = report.to_evidence_evaluation();
        assert_eq!(evaluation.execution_trace, report.execution_trace);
    }

    #[test]
    fn current_v4_malformed_trace_is_not_reconstructed_from_stages() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let mut report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        report.execution_trace.results.pop();

        assert!(!report.execution_trace.is_well_formed());
        assert!(!report.execution_trace.matches_report(&report));

        let evaluation = report.to_evidence_evaluation();
        assert_eq!(evaluation.execution_trace, report.execution_trace);
        assert!(!evaluation.is_consistent_with_report(&report));
    }

    #[test]
    fn legacy_v1_trace_validates_against_its_bound_procedure() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);
        let mut trace = EvaluationTrace::from_report_legacy(&report);
        trace.procedure_fingerprint = EvaluationProcedure::attestation_ed25519_v1().fingerprint();

        assert!(trace.is_well_formed());
        assert_eq!(
            trace.terminal_outcome(),
            Some(ReceiptAttestationVerificationOutcome::Verified)
        );

        trace.results.pop();
        assert!(!trace.is_well_formed());
    }

    #[test]
    fn legacy_v3_report_canonicalization_excludes_execution_trace() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = resolved_report(&verifier, &envelope, &receipt);
        let v4 = report.canonical_bytes();
        report.verifier_version = "symthaea-observation-attestation-report-v3";
        report.execution_trace = EvaluationTrace::default();
        let v3 = report.canonical_bytes();
        assert!(v3.starts_with(b"symthaea:observation-attestation-report:v3\n"));
        assert!(v4.starts_with(b"symthaea:observation-attestation-report:v4\n"));
        assert_ne!(v3, v4);
    }

    #[test]
    fn legacy_v3_report_fingerprint_uses_v3_hash_domain() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = resolved_report(&verifier, &envelope, &receipt);
        report.verifier_version = "symthaea-observation-attestation-report-v3";
        report.execution_trace = EvaluationTrace::default();

        let expected = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(REPORT_DOMAIN_SEPARATOR_V3);
            hasher.update(&report.canonical_bytes());
            hasher.finalize().to_hex().to_string()
        };
        assert_eq!(report.fingerprint(), expected);
    }

    #[test]
    fn current_report_uses_captured_trace_as_evaluation_source() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();
        assert_eq!(evaluation.execution_trace, report.execution_trace);
        assert!(evaluation.is_consistent_with_report(&report));
    }

    #[test]
    fn evaluation_context_binds_verifier_and_procedure_semantic_ids() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();

        assert_eq!(evaluation.context.verifier_id, VERIFIER_IMPLEMENTATION_ID);
        assert_eq!(evaluation.context.procedure_id, EVALUATION_PROCEDURE_ID);
        assert!(evaluation.context.matches_report(&report));

        let mut tampered = evaluation.clone();
        tampered.context.procedure_id = "different-procedure";
        tampered.context_fingerprint = tampered.context.fingerprint();
        assert!(!tampered.is_consistent_with_report(&report));
    }

    #[test]
    fn evaluator_identity_changes_context_without_changing_subject() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);

        let base = report.to_evidence_evaluation();
        let independent = EvidenceEvaluation::from_report_with_context(
            &report,
            VerificationContext::from_report(&report)
                .with_evaluator_identity_fingerprint("evaluator-b"),
        );

        assert_eq!(base.subject_fingerprint, independent.subject_fingerprint);
        assert_ne!(base.context_fingerprint, independent.context_fingerprint);
        assert_ne!(base.fingerprint(), independent.fingerprint());
        assert_eq!(
            independent.context.evaluator_identity_fingerprint.as_deref(),
            Some("evaluator-b")
        );
    }

    #[test]
    fn trace_fingerprint_is_exposed_by_report_and_evaluation() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();

        assert_eq!(
            report.execution_trace_fingerprint(),
            report.execution_trace.fingerprint()
        );
        assert_eq!(
            evaluation.execution_trace_fingerprint(),
            report.execution_trace.fingerprint()
        );
    }

    #[test]
    fn evidence_evaluation_has_single_source_of_stage_results() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();

        assert!(evaluation.execution_trace.is_well_formed());
        assert_eq!(
            evaluation.execution_trace.results.last().map(|result| result.stage),
            Some(VerificationStage::Passed)
        );
        assert!(evaluation.boundary.established.contains(&EvaluationClaim::CryptographicProofValidity));
    }

    #[test]
    fn executed_check_trace_matches_reached_stages() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope.attester_id.clear();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();

        assert_eq!(
            evaluation.execution_trace.executed_check_ids(),
            vec!["envelope-structural-validation"]
        );
    }

    #[test]
    fn executed_check_trace_contains_all_checks_on_success() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();

        assert_eq!(
            evaluation.execution_trace.executed_check_ids(),
            EvaluationProcedure::attestation_ed25519()
                .checks
                .iter()
                .map(|check| check.id())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn verification_report_binds_evaluation_procedure() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);

        assert_eq!(
            report.procedure_fingerprint,
            EvaluationProcedure::attestation_ed25519().fingerprint()
        );
    }

    #[test]
    fn evaluation_context_cannot_rebind_execution_facts() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);

        let mut attempted_rebinding = VerificationContext::from_report(&report);
        attempted_rebinding.procedure_fingerprint = "procedure-alternate".into();
        attempted_rebinding.evaluated_at_unix_ns = 999;

        let evaluation =
            EvidenceEvaluation::from_report_with_context(&report, attempted_rebinding);

        assert_eq!(
            evaluation.context.procedure_fingerprint,
            report.procedure_fingerprint
        );
        assert_eq!(
            evaluation.context.evaluated_at_unix_ns,
            report.evaluated_at_unix_ns
        );
        assert_eq!(
            evaluation.context.policy_fingerprint,
            report.policy_fingerprint
        );
        assert_eq!(
            evaluation.context.environment_fingerprint,
            report.environment_fingerprint
        );
    }

    #[test]
    fn evaluation_context_allows_supplemental_trust_and_authorization_roots() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);

        let base = report.to_evidence_evaluation();
        let changed = EvidenceEvaluation::from_report_with_context(
            &report,
            VerificationContext::from_report(&report)
                .with_trust_root_fingerprint("trust-root-a")
                .with_authorization_policy_fingerprint("authorization-policy-a"),
        );

        assert_eq!(base.subject_fingerprint, changed.subject_fingerprint);
        assert_ne!(base.context_fingerprint, changed.context_fingerprint);
        assert_ne!(base.fingerprint(), changed.fingerprint());
        assert_eq!(
            changed.context.trust_root_fingerprint.as_deref(),
            Some("trust-root-a")
        );
        assert_eq!(
            changed.context.authorization_policy_fingerprint.as_deref(),
            Some("authorization-policy-a")
        );
    }

    #[test]
    fn legacy_v1_execution_trace_remains_well_formed() {
        let procedure = EvaluationProcedure::attestation_ed25519_v1();
        let trace = EvaluationTrace {
            procedure_fingerprint: procedure.fingerprint(),
            results: procedure
                .checks
                .iter()
                .copied()
                .enumerate()
                .map(|(index, check)| EvaluationCheckResult {
                    sequence: index as u32,
                    check,
                    stage: VerificationStage::Passed,
                })
                .collect(),
        };
        assert!(trace.is_well_formed());
        assert_eq!(
            trace.terminal_outcome(),
            Some(ReceiptAttestationVerificationOutcome::Verified)
        );
        assert_ne!(
            procedure.fingerprint(),
            EvaluationProcedure::attestation_ed25519().fingerprint()
        );
    }

    #[test]
    fn legacy_v1_execution_trace_preserves_historical_failure_semantics() {
        let procedure = EvaluationProcedure::attestation_ed25519_v1();
        let trace = EvaluationTrace {
            procedure_fingerprint: procedure.fingerprint(),
            results: procedure
                .checks
                .iter()
                .copied()
                .enumerate()
                .map(|(index, check)| EvaluationCheckResult {
                    sequence: index as u32,
                    check,
                    stage: if check == EvaluationCheck::CryptographicProof {
                        VerificationStage::Failed(
                            ReceiptAttestationVerificationOutcome::InvalidSignature,
                        )
                    } else {
                        VerificationStage::Passed
                    },
                })
                .collect(),
        };

        assert!(trace.is_well_formed());
        assert_eq!(
            trace.terminal_outcome(),
            Some(ReceiptAttestationVerificationOutcome::InvalidSignature)
        );
    }

    #[test]
    fn current_report_cannot_adopt_legacy_procedure_semantics() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = resolved_report(&verifier, &envelope, &receipt);

        report.procedure_fingerprint =
            EvaluationProcedure::attestation_ed25519_v1().fingerprint();
        report.execution_trace = EvaluationTrace {
            procedure_fingerprint: report.procedure_fingerprint.clone(),
            results: EvaluationProcedure::attestation_ed25519_v1()
                .checks
                .iter()
                .copied()
                .enumerate()
                .map(|(index, check)| EvaluationCheckResult {
                    sequence: index as u32,
                    check,
                    stage: VerificationStage::Passed,
                })
                .collect(),
        };

        assert!(!report.has_consistent_identity_bindings());
        assert!(!report.to_evidence_evaluation().is_consistent_with_report(&report));
    }

    #[test]
    fn verification_report_self_validation_rejects_resolution_metadata_mutation() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = resolved_report(&verifier, &envelope, &receipt);

        assert!(report.is_well_formed());
        assert_eq!(report.verification_method, VerificationStage::Passed);

        report.resolved_verification_method = None;
        assert!(!report.is_well_formed());
    }

    #[test]
    fn verification_report_self_validation_rejects_stage_mutation() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = resolved_report(&verifier, &envelope, &receipt);

        assert!(report.is_well_formed());
        report.cryptographic_proof =
            VerificationStage::Failed(ReceiptAttestationVerificationOutcome::InvalidSignature);
        assert!(!report.is_well_formed());
    }

    #[test]
    fn current_report_does_not_repair_malformed_execution_trace() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let mut report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);

        report.execution_trace = EvaluationTrace::default();

        let evaluation = report.to_evidence_evaluation();

        assert!(evaluation.execution_trace.results.is_empty());
        assert_eq!(
            evaluation.execution_trace,
            EvaluationTrace::default()
        );
        assert!(!evaluation.execution_trace.is_well_formed());
        assert!(!evaluation.is_consistent_with_report(&report));
    }

    #[test]
    fn legacy_v3_ignores_unbound_current_execution_trace() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = resolved_report(&verifier, &envelope, &receipt);
        report.verifier_version = "symthaea-observation-attestation-report-v3";
        report.procedure_fingerprint =
            EvaluationProcedure::attestation_ed25519_v1().fingerprint();
        report.execution_trace = EvaluationTrace {
            procedure_fingerprint: report.procedure_fingerprint.clone(),
            results: vec![EvaluationCheckResult {
                sequence: 7,
                check: EvaluationCheck::CryptographicProof,
                stage: VerificationStage::Passed,
            }],
        };

        let evaluation = report.to_evidence_evaluation();
        let legacy_projection = EvaluationTrace::from_report_legacy(&report);

        assert_eq!(evaluation.execution_trace, legacy_projection);
        assert_ne!(evaluation.execution_trace, report.execution_trace);
        assert!(evaluation.execution_trace.is_well_formed());
        assert!(evaluation.is_consistent_with_report(&report));
    }

    #[test]
    fn legacy_v3_report_self_validation_uses_historical_projection() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = resolved_report(&verifier, &envelope, &receipt);
        report.verifier_version = "symthaea-observation-attestation-report-v3";
        report.procedure_fingerprint =
            EvaluationProcedure::attestation_ed25519_v1().fingerprint();
        report.execution_trace = EvaluationTrace::default();

        assert!(report.is_well_formed());

        report.outcome = ReceiptAttestationVerificationOutcome::InvalidSignature;
        assert!(!report.is_well_formed());
    }

    #[test]
    fn legacy_v3_report_can_materialize_current_evidence_evaluation() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = resolved_report(&verifier, &envelope, &receipt);
        report.verifier_version = "symthaea-observation-attestation-report-v3";
        report.procedure_fingerprint =
            EvaluationProcedure::attestation_ed25519_v1().fingerprint();
        report.execution_trace = EvaluationTrace::default();

        let evaluation = report.to_evidence_evaluation();
        assert!(evaluation.execution_trace.is_well_formed());
        assert!(evaluation.is_consistent_with_report(&report));
    }

    #[test]
    fn evaluation_procedure_canonicalization_binds_check_definition_versions() {
        let procedure = EvaluationProcedure::attestation_ed25519();
        let canonical = procedure.canonical_bytes();
        for check in procedure.checks {
            let id = check.id();
            let version = check.definition_version();
            assert!(canonical.windows(id.len()).any(|window| window == id.as_bytes()));
            assert!(canonical
                .windows(version.len())
                .any(|window| window == version.as_bytes()));
        }
        assert!(canonical.starts_with(
            b"symthaea:observation-evaluation-procedure:v2\n"
        ));
    }

    #[test]
    fn evaluation_procedure_fingerprint_binds_check_order() {
        let procedure = EvaluationProcedure::attestation_ed25519();
        let reordered = EvaluationProcedure {
            procedure_version: procedure.procedure_version,
            procedure_id: procedure.procedure_id,
            checks: &[
                EvaluationCheck::ReceiptCommitment,
                EvaluationCheck::EnvelopeStructuralValidation,
                EvaluationCheck::TemporalValidity,
                EvaluationCheck::CryptosuiteConformance,
                EvaluationCheck::VerificationMethodResolution,
                EvaluationCheck::VerificationMethodLifecycle,
                EvaluationCheck::ProofPurposeAuthorization,
                EvaluationCheck::ProofPolicyConformance,
                EvaluationCheck::CryptographicProof,
            ],
        };

        assert_ne!(procedure.fingerprint(), reordered.fingerprint());
    }

    #[test]
    fn evaluation_trace_rejects_unknown_procedure_identity() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);
        let mut trace = EvaluationTrace::from_report(&report);
        trace.procedure_fingerprint = "unknown-procedure".into();
        assert!(!trace.is_well_formed());
    }

    #[test]
    fn evaluation_trace_binds_check_results_and_procedure() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope
            .proof
            .as_mut()
            .expect("signed envelope proof")[0] ^= 0x01;
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);
        let evaluation = report.to_evidence_evaluation();

        assert_eq!(
            evaluation.execution_trace.procedure_fingerprint,
            report.procedure_fingerprint
        );
        assert!(evaluation.execution_trace.is_well_formed());
        assert!(evaluation.boundary.is_well_formed());
        assert_eq!(evaluation.execution_trace.results.len(), 9);
        let last = evaluation.execution_trace.results.last().expect("trace has a result");
        assert_eq!(last.check, EvaluationCheck::CryptographicProof);
        assert_eq!(
            last.stage,
            VerificationStage::Failed(
                ReceiptAttestationVerificationOutcome::InvalidSignature
            )
        );
        assert_eq!(
            evaluation.execution_trace.results.iter().map(|r| r.sequence).collect::<Vec<_>>(),
            (0..9).collect::<Vec<_>>()
        );
    }

    #[test]
    fn evaluation_trace_changes_when_check_result_changes() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);
        let mut trace = EvaluationTrace::from_report(&report);
        let original = trace.fingerprint();
        trace.results.last_mut().expect("trace has a result").stage =
            VerificationStage::Failed(ReceiptAttestationVerificationOutcome::InvalidSignature);
        assert_ne!(original, trace.fingerprint());
    }

    #[test]
    fn evidence_evaluation_fingerprint_is_order_independent_for_boundary_claims() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);
        let first = report.to_evidence_evaluation();
        let mut second = first.clone();
        second.boundary.established.reverse();
        second.boundary.not_established.reverse();
        second.boundary.indeterminate.reverse();
        assert_eq!(first.fingerprint(), second.fingerprint());
    }

    #[test]
    fn evidence_evaluation_rejects_post_hoc_boundary_inflation() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );

        let report = resolved_report(&verifier, &envelope, &receipt);
        let mut evaluation = report.to_evidence_evaluation();
        assert!(evaluation.is_consistent_with_report(&report));
        evaluation
            .boundary
            .established
            .push(EvaluationClaim::UnderlyingObservationTruth);
        assert!(!evaluation.is_consistent_with_report(&report));
    }

    #[test]
    fn procedure_fingerprint_binds_failure_outcome_semantics() {
        let current = EvaluationProcedure::attestation_ed25519();
        let legacy = EvaluationProcedure::attestation_ed25519_v1();

        assert_ne!(current.fingerprint(), legacy.fingerprint());

        let current_canonical = current.canonical_bytes();

        // Reconstruct the pre-hardening v2 canonical form. The current form
        // must differ because failure semantics are now part of procedure identity.
        fn write_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        let mut pre_hardening_v2 = Vec::new();
        pre_hardening_v2.extend_from_slice(b"symthaea:observation-evaluation-procedure:v2\n");
        write_string(&mut pre_hardening_v2, current.procedure_version);
        write_string(&mut pre_hardening_v2, current.procedure_id);
        pre_hardening_v2.extend_from_slice(&(current.checks.len() as u64).to_be_bytes());
        for check in current.checks {
            write_string(&mut pre_hardening_v2, check.id());
            write_string(&mut pre_hardening_v2, check.definition_version());
        }
        assert_ne!(current_canonical, pre_hardening_v2);

        // Reconstruct the historical v1 canonical form exactly. This guards
        // against accidentally changing the identity of legacy evidence while
        // strengthening the current procedure.
        let mut historical = Vec::new();
        historical.extend_from_slice(b"symthaea:observation-evaluation-procedure:v1\n");
        write_string(&mut historical, legacy.procedure_version);
        write_string(&mut historical, legacy.procedure_id);
        historical.extend_from_slice(&(legacy.checks.len() as u64).to_be_bytes());
        for check in legacy.checks {
            write_string(&mut historical, check.id());
        }
        assert_eq!(legacy.canonical_bytes(), historical);
    }

    #[test]
    fn execution_trace_rejects_outcome_for_wrong_check() {
        let procedure = EvaluationProcedure::attestation_ed25519();
        let mut trace = EvaluationTrace {
            procedure_fingerprint: procedure.fingerprint(),
            results: vec![EvaluationCheckResult {
                sequence: 0,
                check: EvaluationCheck::EnvelopeStructuralValidation,
                stage: VerificationStage::Failed(
                    ReceiptAttestationVerificationOutcome::InvalidSignature,
                ),
            }],
        };
        assert!(!trace.is_well_formed());

        trace.results[0].stage = VerificationStage::Failed(
            ReceiptAttestationVerificationOutcome::InvalidEnvelope,
        );
        assert!(trace.is_well_formed());
    }

    #[test]
    fn execution_trace_rejects_contradictory_report_stage_projection() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let mut report = resolved_report(&verifier, &envelope, &receipt);
        assert!(report.execution_trace.matches_report(&report));
        report.structural_validation =
            VerificationStage::Failed(ReceiptAttestationVerificationOutcome::InvalidEnvelope);
        assert!(!report.execution_trace.matches_report(&report));
    }

    #[test]
    fn verification_report_fingerprint_binds_resolution_identity_fields() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let report = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .verify_report(&envelope, &receipt);

        assert!(report.resolved_verification_method.is_none());
        assert!(report.resolution_snapshot_fingerprint.is_none());

        let mut resolved_method = report.clone();
        resolved_method.resolved_verification_method =
            Some("did:example:attester-a#key-2".into());
        assert_ne!(report.fingerprint(), resolved_method.fingerprint());

        let mut snapshot = report.clone();
        snapshot.resolution_snapshot_fingerprint = Some("opaque-snapshot-1".into());
        assert_ne!(report.fingerprint(), snapshot.fingerprint());

        let mut paired = report.clone();
        paired.resolved_verification_method =
            Some("did:example:attester-a#key-2".into());
        paired.resolution_snapshot_fingerprint = Some("opaque-snapshot-1".into());
        assert_ne!(resolved_method.fingerprint(), paired.fingerprint());
        assert_ne!(snapshot.fingerprint(), paired.fingerprint());

        let evaluation = report.to_evidence_evaluation();

        let mut rebound_method = report.clone();
        rebound_method.resolved_verification_method =
            Some("did:example:attester-a#key-2".into());
        assert!(!evaluation.is_consistent_with_report(&rebound_method));

        let mut rebound_snapshot = report.clone();
        rebound_snapshot.resolution_snapshot_fingerprint = Some("opaque-snapshot-1".into());
        assert!(!evaluation.is_consistent_with_report(&rebound_snapshot));
    }

    #[test]
    fn resolver_snapshot_is_retained_when_post_resolution_crypto_failure_occurs() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope
            .proof
            .as_mut()
            .expect("signed envelope proof")[0] ^= 0x01;

        let method = ResolvedVerificationMethod {
            verification_method: envelope.attester_id.clone(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let expected_snapshot = resolver
            .snapshot_fingerprint_for(&envelope.attester_id)
            .expect("active method has a scoped snapshot");

        let report = Ed25519ReceiptVerifier::new(
            envelope.attester_id.clone(),
            signing_key.verifying_key(),
            150,
        )
        .verify_with_resolver_report(&envelope, &receipt, &resolver);

        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::InvalidSignature
        );
        assert_eq!(
            report.resolved_verification_method.as_deref(),
            Some(envelope.attester_id.as_str())
        );
        assert_eq!(
            report.resolution_snapshot_fingerprint.as_deref(),
            Some(expected_snapshot.as_str())
        );
        assert!(report.execution_trace.is_well_formed());
        assert_eq!(
            report.execution_trace.results.last().map(|result| result.stage),
            Some(VerificationStage::Failed(
                ReceiptAttestationVerificationOutcome::InvalidSignature
            ))
        );
    }

    #[test]
    fn failed_resolution_report_binds_snapshot_into_evidence_evaluation() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let method = ResolvedVerificationMethod {
            verification_method: envelope.attester_id.clone(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Revoked,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let report = Ed25519ReceiptVerifier::new(
            envelope.attester_id.clone(),
            signing_key.verifying_key(),
            150,
        )
        .verify_with_resolver_report(&envelope, &receipt, &resolver);

        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::VerificationMethodRevoked
        );
        assert!(report.resolution_snapshot_fingerprint.is_some());

        let mut evaluation = report.to_evidence_evaluation();
        assert!(evaluation.is_consistent_with_report(&report));

        evaluation.verification_report_fingerprint = {
            let mut mutated = report.clone();
            mutated.resolution_snapshot_fingerprint =
                Some("different-resolution-state".into());
            mutated.fingerprint()
        };

        assert!(!evaluation.is_consistent_with_report(&report));
    }

    #[test]
    fn resolver_identity_mismatch_cannot_fall_through_to_crypto_verification() {
        struct MismatchedKeyResolver {
            returned_method: String,
            verifying_key: VerifyingKey,
        }

        impl VerificationMethodResolver for MismatchedKeyResolver {
            fn resolve(
                &self,
                _verification_method: &str,
            ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
                Ok(ResolvedVerificationMethod {
                    verification_method: self.returned_method.clone(),
                    verifying_key: self.verifying_key,
                    status: VerificationMethodStatus::Active,
                    allowed_proof_purposes: vec!["observation-independence".into()],
                })
            }

            fn resolve_with_snapshot(
                &self,
                verification_method: &str,
            ) -> Result<ResolvedVerificationMethodSnapshot, VerificationMethodResolutionError> {
                Ok(ResolvedVerificationMethodSnapshot {
                    resolved: self.resolve(verification_method)?,
                    snapshot_fingerprint: Some("identity-mismatch-snapshot".into()),
                })
            }
        }

        let (envelope, signing_key, receipt) = envelope_and_key();
        let resolver = MismatchedKeyResolver {
            returned_method: "did:example:unexpected#key-9".into(),
            verifying_key: signing_key.verifying_key(),
        };

        let report = Ed25519ReceiptVerifier::new(
            envelope.attester_id.clone(),
            signing_key.verifying_key(),
            150,
        )
        .verify_with_resolver_report(&envelope, &receipt, &resolver);

        // The returned key is intentionally valid for the envelope. Only the
        // resolver's method identity mismatch should prevent verification.
        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable
        );
        assert_ne!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::Verified
        );
        assert_eq!(
            report.execution_trace.results.last().map(|result| result.check),
            Some(EvaluationCheck::VerificationMethodResolution)
        );
        assert_eq!(
            report.resolution_snapshot_fingerprint.as_deref(),
            Some("identity-mismatch-snapshot")
        );
        assert!(report.execution_trace.is_well_formed());
    }

    #[test]
    fn resolver_empty_snapshot_fails_closed_before_lifecycle_or_proof() {
        struct EmptySnapshotResolver;

        impl VerificationMethodResolver for EmptySnapshotResolver {
            fn resolve(
                &self,
                verification_method: &str,
            ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
                Ok(ResolvedVerificationMethod {
                    verification_method: verification_method.to_string(),
                    verifying_key: SigningKey::from_bytes(&[7u8; 32]).verifying_key(),
                    status: VerificationMethodStatus::Active,
                    allowed_proof_purposes: vec!["observation-independence".into()],
                })
            }

            fn resolve_with_snapshot(
                &self,
                verification_method: &str,
            ) -> Result<ResolvedVerificationMethodSnapshot, VerificationMethodResolutionError> {
                Ok(ResolvedVerificationMethodSnapshot {
                    resolved: self.resolve(verification_method)?,
                    snapshot_fingerprint: Some(String::new()),
                })
            }
        }

        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let report = verifier.verify_with_resolver_report(&envelope, &receipt, &EmptySnapshotResolver);

        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable
        );
        assert_eq!(
            report.verification_method,
            VerificationStage::Failed(
                ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable
            )
        );
        assert_eq!(report.lifecycle, VerificationStage::NotEvaluated);
        assert_eq!(report.cryptographic_proof, VerificationStage::NotEvaluated);
        assert!(report.resolution_snapshot_fingerprint.is_none());
        assert!(report.is_well_formed());
    }

    #[test]
    fn resolver_snapshot_is_retained_when_resolver_returns_mismatched_method() {
        struct MismatchedResolver {
            expected: String,
            returned: ResolvedVerificationMethod,
            snapshot: String,
        }

        impl VerificationMethodResolver for MismatchedResolver {
            fn resolve(
                &self,
                verification_method: &str,
            ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
                assert_eq!(verification_method, self.expected);
                Ok(self.returned.clone())
            }

            fn resolve_with_snapshot(
                &self,
                verification_method: &str,
            ) -> Result<ResolvedVerificationMethodSnapshot, VerificationMethodResolutionError> {
                Ok(ResolvedVerificationMethodSnapshot {
                    resolved: self.resolve(verification_method)?,
                    snapshot_fingerprint: Some(self.snapshot.clone()),
                })
            }
        }

        let (envelope, signing_key, receipt) = envelope_and_key();
        let requested_method = envelope.attester_id.clone();
        let returned_method = ResolvedVerificationMethod {
            verification_method: "did:example:unexpected#key-9".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = MismatchedResolver {
            expected: requested_method.clone(),
            returned: returned_method,
            snapshot: "mismatched-resolution-snapshot".into(),
        };

        let report = Ed25519ReceiptVerifier::new(
            requested_method.clone(),
            signing_key.verifying_key(),
            150,
        )
        .verify_with_resolver_report(&envelope, &receipt, &resolver);

        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable
        );
        assert_eq!(
            report.resolved_verification_method.as_deref(),
            Some(requested_method.as_str())
        );
        assert_eq!(
            report.resolution_snapshot_fingerprint.as_deref(),
            Some("mismatched-resolution-snapshot")
        );
        assert!(report.execution_trace.is_well_formed());
        assert_eq!(
            report.execution_trace.results.last().map(|result| result.stage),
            Some(VerificationStage::Failed(
                ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable
            ))
        );
    }

    #[test]
    fn resolver_snapshot_is_retained_when_proof_purpose_is_unauthorized() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let method = ResolvedVerificationMethod {
            verification_method: envelope.attester_id.clone(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["authentication".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let expected_snapshot = resolver
            .snapshot_fingerprint_for(&envelope.attester_id)
            .expect("active method has a scoped snapshot");

        let report = Ed25519ReceiptVerifier::new(
            envelope.attester_id.clone(),
            signing_key.verifying_key(),
            150,
        )
        .verify_with_resolver_report(&envelope, &receipt, &resolver);

        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::ProofPurposeUnauthorized
        );
        assert_eq!(
            report.resolved_verification_method.as_deref(),
            Some(envelope.attester_id.as_str())
        );
        assert_eq!(
            report.resolution_snapshot_fingerprint.as_deref(),
            Some(expected_snapshot.as_str())
        );
        assert!(report.execution_trace.is_well_formed());
        assert_eq!(
            report.execution_trace.results.last().map(|result| result.stage),
            Some(VerificationStage::Failed(
                ReceiptAttestationVerificationOutcome::ProofPurposeUnauthorized
            ))
        );
    }

    #[test]
    fn resolver_error_does_not_attach_unpaired_verifier_snapshot() {
        struct UnavailableResolver;

        impl VerificationMethodResolver for UnavailableResolver {
            fn resolve(
                &self,
                _verification_method: &str,
            ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
                Err(VerificationMethodResolutionError::Unavailable)
            }
        }

        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            envelope.attester_id.clone(),
            signing_key.verifying_key(),
            150,
        )
        .with_resolution_snapshot_fingerprint("verifier-later-state");

        let report =
            verifier.verify_with_resolver_report(&envelope, &receipt, &UnavailableResolver);

        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable
        );
        assert_eq!(
            report.resolved_verification_method.as_deref(),
            Some(envelope.attester_id.as_str())
        );
        assert!(report.resolution_snapshot_fingerprint.is_none());
        assert!(report.execution_trace.is_well_formed());
    }

    #[test]
    fn resolver_snapshot_is_retained_when_resolution_reaches_terminal_failure() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let method = ResolvedVerificationMethod {
            verification_method: envelope.attester_id.clone(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Revoked,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let expected_snapshot = resolver
            .snapshot_fingerprint_for(&envelope.attester_id)
            .expect("revoked method has a scoped snapshot");

        let report = Ed25519ReceiptVerifier::new(
            envelope.attester_id.clone(),
            signing_key.verifying_key(),
            150,
        )
        .verify_with_resolver_report(&envelope, &receipt, &resolver);

        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::VerificationMethodRevoked
        );
        assert_eq!(
            report.resolved_verification_method.as_deref(),
            Some(envelope.attester_id.as_str())
        );
        assert_eq!(
            report.resolution_snapshot_fingerprint.as_deref(),
            Some(expected_snapshot.as_str())
        );
        assert!(report.execution_trace.is_well_formed());
    }

}
