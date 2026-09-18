// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Authenticated Lean execution -> qualification-bound evidence attestation.
//!
//! Generic `EvidenceReceipt` values remain structural and caller-constructible.
//! This module provides the narrower path that binds a real Lean execution to:
//! the exact frozen mathematical subject, the exact review qualification that
//! authorized it, the exact sealed claim, and the exact audited Lean artifact.

use crate::axiom_gate::{
    AxiomPolicy, GateReport, ProofAuditOutcome, audit_authenticated_lean_source,
    with_authenticated_spec_probe,
};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use symthaea_math_research::{
    EvidenceIssue, EvidenceKind, EvidenceReceipt, EvidenceReceiptDraft, EvidenceVerdict,
    FrozenChallenge, QualifiedClaim, Sha256Digest,
};

const TRUST_DOMAIN: &str = "symthaea.lean-bridge.authenticated-kernel.v1";
const AXIOM_POLICY_DOMAIN: &str = "symthaea.proof-audit.axiom-policy.v1";
const ENVIRONMENT_DOMAIN: &str = "symthaea.lean-bridge.verifier-environment.v1";
const ATTESTATION_DOMAIN: &str = "symthaea.lean-bridge.qualified-attestation.v1";
const PRODUCER_ID: &str = "symthaea-lean-bridge/authenticated-kernel-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
struct BinarySnapshot {
    configured_path: PathBuf,
    canonical_path: PathBuf,
    sha256: Sha256Digest,
}

/// Qualification-bound evidence that can only be constructed after successful
/// authenticated Lean execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedLeanReceipt {
    receipt: EvidenceReceipt,
    qualification_sha256: Sha256Digest,
    qualified_claim_binding_sha256: Sha256Digest,
    audited_artifact_sha256: Sha256Digest,
    lean_binary_sha256: Sha256Digest,
    environment_sha256: Sha256Digest,
    axiom_policy_sha256: Sha256Digest,
    attestation_sha256: Sha256Digest,
    lean_binary_path: PathBuf,
}

impl AuthenticatedLeanReceipt {
    /// Structural receipt. Possession of this value alone does not retain the
    /// qualification-bound authority of the authenticated wrapper.
    pub fn receipt(&self) -> &EvidenceReceipt {
        &self.receipt
    }

    /// Deliberately drops verifier-specific authority back to the generic
    /// structural evidence layer.
    pub fn into_receipt(self) -> EvidenceReceipt {
        self.receipt
    }

    pub fn qualification_sha256(&self) -> &Sha256Digest {
        &self.qualification_sha256
    }

    pub fn qualified_claim_binding_sha256(&self) -> &Sha256Digest {
        &self.qualified_claim_binding_sha256
    }

    pub fn audited_artifact_sha256(&self) -> &Sha256Digest {
        &self.audited_artifact_sha256
    }

    pub fn lean_binary_sha256(&self) -> &Sha256Digest {
        &self.lean_binary_sha256
    }

    pub fn environment_sha256(&self) -> &Sha256Digest {
        &self.environment_sha256
    }

    pub fn axiom_policy_sha256(&self) -> &Sha256Digest {
        &self.axiom_policy_sha256
    }

    pub fn attestation_sha256(&self) -> &Sha256Digest {
        &self.attestation_sha256
    }

    pub fn lean_binary_path(&self) -> &Path {
        &self.lean_binary_path
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptVerificationError {
    ChallengeMismatch,
    QualificationMismatch {
        claim_qualification: Sha256Digest,
        challenge_qualification: Sha256Digest,
    },
    StatementDigestMismatch {
        claim: Sha256Digest,
        expected: Sha256Digest,
    },
    EmptyTimestamp,
    MissingPinnedLeanBinary,
    LeanBinaryNotAbsolute { configured: PathBuf },
    LeanBinaryCanonicalize { configured: PathBuf, message: String },
    LeanBinaryNotFile { canonical: PathBuf },
    LeanBinaryRead { path: PathBuf, message: String },
    LeanBinaryChanged {
        before_path: PathBuf,
        after_path: PathBuf,
        before_sha256: Sha256Digest,
        after_sha256: Sha256Digest,
    },
    ProbeConstruction(String),
    LeanNotInstalled,
    LeanProcess(String),
    GateRejected(GateReport),
    ReceiptRejected(Vec<EvidenceIssue>),
}

impl fmt::Display for ReceiptVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ReceiptVerificationError {}

/// Execute Lean and issue an authenticated receipt only for a claim already
/// bound to this exact challenge qualification.
pub fn verify_claim_with_lean(
    challenge: &FrozenChallenge,
    qualified_claim: &QualifiedClaim,
    script_body: &str,
    theorem: &str,
    expected_statement: &str,
    policy: &AxiomPolicy,
    created_at_utc: &str,
) -> Result<AuthenticatedLeanReceipt, ReceiptVerificationError> {
    if qualified_claim.challenge_sha256() != challenge.challenge_sha256() {
        return Err(ReceiptVerificationError::ChallengeMismatch);
    }
    if qualified_claim.qualification_sha256() != challenge.qualification_sha256() {
        return Err(ReceiptVerificationError::QualificationMismatch {
            claim_qualification: qualified_claim.qualification_sha256().clone(),
            challenge_qualification: challenge.qualification_sha256().clone(),
        });
    }

    let claim = qualified_claim.claim();
    let normalized_statement = expected_statement.trim();
    let expected_statement_sha256 = Sha256Digest::of_bytes(normalized_statement.as_bytes());
    if claim.statement_sha256() != &expected_statement_sha256 {
        return Err(ReceiptVerificationError::StatementDigestMismatch {
            claim: claim.statement_sha256().clone(),
            expected: expected_statement_sha256,
        });
    }
    if created_at_utc.trim().is_empty() {
        return Err(ReceiptVerificationError::EmptyTimestamp);
    }

    let audited_source = with_authenticated_spec_probe(script_body, theorem, normalized_statement)
        .map_err(ReceiptVerificationError::ProbeConstruction)?;
    let artifact_sha256 = Sha256Digest::of_bytes(audited_source.as_bytes());

    let before = snapshot_pinned_lean_binary()?;
    let axiom_policy_sha256 = axiom_policy_digest(policy);
    let environment_sha256 = verifier_environment_digest(challenge, &before, &axiom_policy_sha256);

    let audit_outcome =
        audit_authenticated_lean_source(script_body, theorem, normalized_statement, policy);

    let after = snapshot_pinned_lean_binary()?;
    if before != after {
        return Err(ReceiptVerificationError::LeanBinaryChanged {
            before_path: before.canonical_path,
            after_path: after.canonical_path,
            before_sha256: before.sha256,
            after_sha256: after.sha256,
        });
    }

    let report = match audit_outcome {
        ProofAuditOutcome::Audited(report) => report,
        ProofAuditOutcome::LeanNotInstalled => return Err(ReceiptVerificationError::LeanNotInstalled),
        ProofAuditOutcome::ProcessError(message) => {
            return Err(ReceiptVerificationError::LeanProcess(message));
        }
    };
    if !report.accepted() {
        return Err(ReceiptVerificationError::GateRejected(report));
    }

    let receipt_id = format!(
        "lean-kernel:{}:{}",
        claim.claim_id(),
        &artifact_sha256.as_str()[..16]
    );
    let receipt = EvidenceReceiptDraft::for_claim(
        claim,
        receipt_id,
        EvidenceKind::LeanKernel,
        EvidenceVerdict::Supports,
        artifact_sha256.clone(),
        PRODUCER_ID,
        trust_domain_sha256(),
        Some(before.sha256.clone()),
        environment_sha256.clone(),
        created_at_utc.trim(),
    )
    .seal_for_claim(claim)
    .map_err(ReceiptVerificationError::ReceiptRejected)?;

    let qualification_sha256 = challenge.qualification_sha256().clone();
    let qualified_claim_binding_sha256 = qualified_claim.binding_sha256().clone();
    let attestation_sha256 = attestation_digest(
        receipt.receipt_sha256(),
        &qualification_sha256,
        &qualified_claim_binding_sha256,
        &artifact_sha256,
        &before.sha256,
        &environment_sha256,
        &axiom_policy_sha256,
    );

    Ok(AuthenticatedLeanReceipt {
        receipt,
        qualification_sha256,
        qualified_claim_binding_sha256,
        audited_artifact_sha256: artifact_sha256,
        lean_binary_sha256: before.sha256,
        environment_sha256,
        axiom_policy_sha256,
        attestation_sha256,
        lean_binary_path: before.canonical_path,
    })
}

fn snapshot_pinned_lean_binary() -> Result<BinarySnapshot, ReceiptVerificationError> {
    let configured = std::env::var_os("LEAN_PATH_BIN")
        .ok_or(ReceiptVerificationError::MissingPinnedLeanBinary)?;
    snapshot_binary_path(Path::new(&configured))
}

fn snapshot_binary_path(path: &Path) -> Result<BinarySnapshot, ReceiptVerificationError> {
    if !path.is_absolute() {
        return Err(ReceiptVerificationError::LeanBinaryNotAbsolute {
            configured: path.to_path_buf(),
        });
    }
    let canonical_path = path.canonicalize().map_err(|error| {
        ReceiptVerificationError::LeanBinaryCanonicalize {
            configured: path.to_path_buf(),
            message: error.to_string(),
        }
    })?;
    if !canonical_path.is_file() {
        return Err(ReceiptVerificationError::LeanBinaryNotFile {
            canonical: canonical_path,
        });
    }
    let sha256 = sha256_file(&canonical_path)?;
    Ok(BinarySnapshot {
        configured_path: path.to_path_buf(),
        canonical_path,
        sha256,
    })
}

fn sha256_file(path: &Path) -> Result<Sha256Digest, ReceiptVerificationError> {
    let mut file = File::open(path).map_err(|error| ReceiptVerificationError::LeanBinaryRead {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            ReceiptVerificationError::LeanBinaryRead {
                path: path.to_path_buf(),
                message: error.to_string(),
            }
        })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let bytes = hasher.finalize();
    let mut text = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut text, "{byte:02x}").expect("writing SHA-256 text cannot fail");
    }
    Sha256Digest::parse(text).map_err(|_| ReceiptVerificationError::LeanBinaryRead {
        path: path.to_path_buf(),
        message: "internal SHA-256 encoding failure".into(),
    })
}

fn trust_domain_sha256() -> Sha256Digest {
    Sha256Digest::of_bytes(TRUST_DOMAIN.as_bytes())
}

fn axiom_policy_digest(policy: &AxiomPolicy) -> Sha256Digest {
    let mut bytes = Vec::new();
    put_text(&mut bytes, AXIOM_POLICY_DOMAIN);
    for axiom in policy.allowed() {
        put_text(&mut bytes, axiom);
    }
    Sha256Digest::of_bytes(&bytes)
}

fn verifier_environment_digest(
    challenge: &FrozenChallenge,
    binary: &BinarySnapshot,
    axiom_policy_sha256: &Sha256Digest,
) -> Sha256Digest {
    let mut bytes = Vec::new();
    put_text(&mut bytes, ENVIRONMENT_DOMAIN);
    put_text(&mut bytes, &challenge.specification().lean_toolchain);
    put_text(&mut bytes, &challenge.specification().mathlib_revision);
    put_text(&mut bytes, &binary.configured_path.to_string_lossy());
    put_text(&mut bytes, &binary.canonical_path.to_string_lossy());
    put_text(&mut bytes, binary.sha256.as_str());
    put_text(&mut bytes, axiom_policy_sha256.as_str());
    Sha256Digest::of_bytes(&bytes)
}

#[allow(clippy::too_many_arguments)]
fn attestation_digest(
    receipt_sha256: &Sha256Digest,
    qualification_sha256: &Sha256Digest,
    qualified_claim_binding_sha256: &Sha256Digest,
    artifact_sha256: &Sha256Digest,
    binary_sha256: &Sha256Digest,
    environment_sha256: &Sha256Digest,
    axiom_policy_sha256: &Sha256Digest,
) -> Sha256Digest {
    let mut bytes = Vec::new();
    put_text(&mut bytes, ATTESTATION_DOMAIN);
    put_text(&mut bytes, receipt_sha256.as_str());
    put_text(&mut bytes, qualification_sha256.as_str());
    put_text(&mut bytes, qualified_claim_binding_sha256.as_str());
    put_text(&mut bytes, artifact_sha256.as_str());
    put_text(&mut bytes, binary_sha256.as_str());
    put_text(&mut bytes, environment_sha256.as_str());
    put_text(&mut bytes, axiom_policy_sha256.as_str());
    Sha256Digest::of_bytes(&bytes)
}

fn put_text(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_be_bytes());
    output.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_math_research::{
        ClaimDraft, FormalizationReview, FormalizationReviewKind, MathematicalClaimKind,
        MathematicalSpecification, QualificationBindingIssue, SourceReference, SpecificationStatus,
    };

    fn digest(label: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(label.as_bytes())
    }

    fn challenge(label: &str, statement: &str, independent_domain: &str) -> FrozenChallenge {
        let mut specification = MathematicalSpecification::new(
            format!("TEST-{label}"),
            SourceReference {
                locator: format!("https://example.invalid/{label}"),
                revision: "v1".into(),
                retrieved_at_utc: "2026-09-18T00:00:00Z".into(),
            },
            digest(&format!("source:{label}")),
            Sha256Digest::of_bytes(statement.trim().as_bytes()),
            digest(&format!("definitions:{label}")),
            "leanprover/lean4:v4.24.0",
            "mathlib-test-revision",
        );
        let statement_hash = specification.lean_statement_sha256.clone();
        let definitions_hash = specification.definitions_sha256.clone();
        for (reviewer, domain, kind) in [
            ("reviewer-a", "lineage-a", FormalizationReviewKind::DefinitionsAudit),
            ("reviewer-a", "lineage-a", FormalizationReviewKind::SemanticReview),
            (
                "reviewer-independent",
                independent_domain,
                FormalizationReviewKind::IndependentFormalization,
            ),
        ] {
            specification.add_review(FormalizationReview {
                reviewer_id: reviewer.into(),
                independence_domain_sha256: digest(domain),
                method_sha256: digest(&format!("method:{reviewer}:{domain}:{kind:?}")),
                kind,
                lean_statement_sha256: statement_hash.clone(),
                definitions_sha256: definitions_hash.clone(),
                notes: "reviewed".into(),
            });
        }
        specification.advance_to(SpecificationStatus::FrozenChallenge).unwrap();
        specification.freeze().unwrap()
    }

    fn qualified_claim(challenge: &FrozenChallenge, statement: &str) -> QualifiedClaim {
        let claim = ClaimDraft::for_challenge(
            challenge,
            "claim-1",
            Sha256Digest::of_bytes(statement.trim().as_bytes()),
            digest("assumptions"),
            MathematicalClaimKind::Theorem,
            "test",
            digest("context"),
        )
        .seal(challenge)
        .unwrap();
        QualifiedClaim::bind(claim, challenge).unwrap()
    }

    #[test]
    fn statement_mismatch_fails_before_binary_lookup() {
        let challenge = challenge("A", "True", "lineage-b");
        let claim = qualified_claim(&challenge, "True");
        let error = verify_claim_with_lean(
            &challenge,
            &claim,
            "theorem t : True := by trivial",
            "t",
            "False",
            &AxiomPolicy::constitutional(),
            "2026-09-18T00:00:00Z",
        )
        .unwrap_err();
        assert!(matches!(error, ReceiptVerificationError::StatementDigestMismatch { .. }));
    }

    #[test]
    fn challenge_mismatch_fails_before_binary_lookup() {
        let left = challenge("A", "True", "lineage-b");
        let right = challenge("B", "True", "lineage-b");
        let claim = qualified_claim(&left, "True");
        assert_eq!(
            verify_claim_with_lean(
                &right,
                &claim,
                "theorem t : True := by trivial",
                "t",
                "True",
                &AxiomPolicy::constitutional(),
                "2026-09-18T00:00:00Z",
            )
            .unwrap_err(),
            ReceiptVerificationError::ChallengeMismatch
        );
    }

    #[test]
    fn same_subject_different_qualification_fails_before_binary_lookup() {
        let left = challenge("A", "True", "lineage-b");
        let right = challenge("A", "True", "lineage-c");
        assert_eq!(left.challenge_sha256(), right.challenge_sha256());
        assert_ne!(left.qualification_sha256(), right.qualification_sha256());
        let claim = qualified_claim(&left, "True");
        assert!(matches!(
            verify_claim_with_lean(
                &right,
                &claim,
                "theorem t : True := by trivial",
                "t",
                "True",
                &AxiomPolicy::constitutional(),
                "2026-09-18T00:00:00Z",
            )
            .unwrap_err(),
            ReceiptVerificationError::QualificationMismatch { .. }
        ));
    }

    #[test]
    fn empty_timestamp_fails_before_binary_lookup() {
        let challenge = challenge("A", "True", "lineage-b");
        let claim = qualified_claim(&challenge, "True");
        assert_eq!(
            verify_claim_with_lean(
                &challenge,
                &claim,
                "theorem t : True := by trivial",
                "t",
                "True",
                &AxiomPolicy::constitutional(),
                " ",
            )
            .unwrap_err(),
            ReceiptVerificationError::EmptyTimestamp
        );
    }

    #[test]
    fn relative_binary_path_is_rejected() {
        assert_eq!(
            snapshot_binary_path(Path::new("lean")).unwrap_err(),
            ReceiptVerificationError::LeanBinaryNotAbsolute {
                configured: PathBuf::from("lean")
            }
        );
    }

    #[test]
    fn policy_and_environment_identities_are_sensitive() {
        assert_ne!(
            axiom_policy_digest(&AxiomPolicy::constitutional()),
            axiom_policy_digest(&AxiomPolicy::classical())
        );
        let challenge = challenge("A", "True", "lineage-b");
        let a = BinarySnapshot {
            configured_path: PathBuf::from("/opt/lean/bin/lean"),
            canonical_path: PathBuf::from("/nix/store/lean-a/bin/lean"),
            sha256: digest("lean-a"),
        };
        let b = BinarySnapshot {
            configured_path: a.configured_path.clone(),
            canonical_path: PathBuf::from("/nix/store/lean-b/bin/lean"),
            sha256: digest("lean-b"),
        };
        let policy = axiom_policy_digest(&AxiomPolicy::constitutional());
        assert_ne!(
            verifier_environment_digest(&challenge, &a, &policy),
            verifier_environment_digest(&challenge, &b, &policy)
        );
    }

    #[test]
    fn attestation_identity_binds_qualification() {
        let common_receipt = digest("receipt");
        let binding = digest("binding");
        let artifact = digest("artifact");
        let binary = digest("binary");
        let environment = digest("environment");
        let policy = digest("policy");
        assert_ne!(
            attestation_digest(
                &common_receipt,
                &digest("qualification-a"),
                &binding,
                &artifact,
                &binary,
                &environment,
                &policy,
            ),
            attestation_digest(
                &common_receipt,
                &digest("qualification-b"),
                &binding,
                &artifact,
                &binary,
                &environment,
                &policy,
            )
        );
    }

    #[test]
    fn qualified_claim_binding_cannot_cross_subjects() {
        let left = challenge("A", "True", "lineage-b");
        let right = challenge("B", "True", "lineage-b");
        let structural = ClaimDraft::for_challenge(
            &left,
            "claim",
            digest("statement"),
            digest("assumptions"),
            MathematicalClaimKind::Lemma,
            "agent",
            digest("context"),
        )
        .seal(&left)
        .unwrap();
        assert_eq!(
            QualifiedClaim::bind(structural, &right),
            Err(QualificationBindingIssue::ChallengeMismatch)
        );
    }
}
