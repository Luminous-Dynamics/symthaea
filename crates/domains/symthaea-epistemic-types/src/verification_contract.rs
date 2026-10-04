//! Substrate-neutral boundary for adapter-produced claim verification evidence.
//!
//! This module does not resolve controller documents, perform cryptography, or make
//! network calls. It defines the exact request an adapter must satisfy and the
//! evidence envelope it may return after doing those operations.
//!
//! The separation is intentional:
//!
//! `ClaimAuthorship` expresses declarations.
//! `VerificationRequest` states what an adapter is being asked to establish.
//! `VerificationEvidence` records which external resolution/cryptographic artifacts
//! were used to establish it.
//!
//! A `Verified` outcome therefore means "the adapter attests that it performed the
//! required checks", not that the substrate-neutral core itself performed them.

use crate::{
    ClaimAuthorIdentity, ClaimControllerIdentity, ClaimProofPurpose, ClaimVerificationMethod,
    FederatedClaim, FederationDependency,
};
use serde::{Deserialize, Serialize};

pub const VERIFICATION_EVIDENCE_SCHEMA_VERSION: u16 = 1;

/// Typed identifier for the verification relationship under which a verification
/// method is permitted to validate a proof.
///
/// This is deliberately distinct from proof purpose: adapters must establish both
/// the requested purpose and the controller-document relationship rather than
/// assuming that one string semantically subsumes the other.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClaimVerificationRelationship(String);

impl ClaimVerificationRelationship {
    pub fn new(value: impl Into<String>) -> Result<Self, VerificationFailure> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(VerificationFailure::Structural(
                "claim verification relationship must be non-empty".into(),
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        if self.0.trim().is_empty() {
            Err(VerificationFailure::Structural(
                "claim verification relationship must be non-empty".into(),
            ))
        } else {
            Ok(())
        }
    }
}

fn is_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// The exact structural and identity inputs an external verification adapter must
/// operate over before it can return cryptographic/controller evidence.
///
/// The request is derived from the claim itself, so adapters cannot accidentally
/// verify one claim while attaching evidence to another representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationRequest {
    pub claim_representation_digest: String,
    pub statement_digest: String,
    pub author: ClaimAuthorIdentity,
    pub proof_purpose: ClaimProofPurpose,
    pub verification_method: ClaimVerificationMethod,
    pub expected_controller: ClaimControllerIdentity,
    pub expected_verification_relationship: ClaimVerificationRelationship,
}

impl VerificationRequest {
    /// Create a request only when the claim has a complete local authorship binding
    /// for the exact expected purpose and controller.
    ///
    /// This is still structural. The adapter must separately resolve the verification
    /// method, bind it to the expected controller document, validate the permitted
    /// relationship, and verify the cryptographic proof.
    pub fn from_claim(
        claim: &FederatedClaim,
        expected_purpose: ClaimProofPurpose,
        expected_controller: ClaimControllerIdentity,
        expected_verification_relationship: ClaimVerificationRelationship,
    ) -> Result<Self, VerificationFailure> {
        claim
            .validate_structure()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?;

        let authorship = claim
            .authorship_binding()
            .ok_or(VerificationFailure::MissingAuthorship)?;

        if authorship.author() != &claim.author {
            return Err(VerificationFailure::Structural(
                "claim authorship author must match declared claim author".into(),
            ));
        }
        if !authorship.proof_purpose_matches(&expected_purpose) {
            return Err(VerificationFailure::ProofPurposeMismatch {
                expected: expected_purpose,
                actual: authorship.proof_purpose().clone(),
            });
        }

        let verification_method = authorship
            .verification_method()
            .cloned()
            .ok_or(VerificationFailure::MissingVerificationMethod)?;
        let controller = authorship
            .verification_controller()
            .cloned()
            .ok_or(VerificationFailure::MissingVerificationController)?;

        if controller != expected_controller {
            return Err(VerificationFailure::ControllerMismatch {
                expected: expected_controller,
                actual: controller,
            });
        }

        let statement_digest = claim
            .statement_identity()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?
            .digest();

        Ok(Self {
            claim_representation_digest: claim.canonical_digest(),
            statement_digest,
            author: claim.author.clone(),
            proof_purpose: expected_purpose,
            verification_method,
            expected_controller,
            expected_verification_relationship,
        })
    }

    /// Typed external dependencies required for the adapter-side resolution step.
    ///
    /// The core deliberately does not turn these into fetches; an adapter may resolve
    /// them through a DID/controller-document system, Holochain, a local cache, or any
    /// other substrate.
    pub fn dependencies(&self) -> Vec<FederationDependency> {
        vec![
            FederationDependency::VerificationMethod(self.verification_method.as_str().to_owned()),
            FederationDependency::ControllerDocument(
                self.expected_controller.as_str().to_owned(),
            ),
        ]
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        if !is_hex_digest(&self.claim_representation_digest) {
            return Err(VerificationFailure::Structural(
                "claim representation digest must be a 64-character hexadecimal digest".into(),
            ));
        }
        if !is_hex_digest(&self.statement_digest) {
            return Err(VerificationFailure::Structural(
                "statement digest must be a 64-character hexadecimal digest".into(),
            ));
        }
        self.author
            .validate_structure()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?;
        self.proof_purpose
            .validate_structure()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?;
        self.verification_method
            .validate_structure()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?;
        self.expected_controller
            .validate_structure()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?;
        self.expected_verification_relationship.validate_structure()?;
        Ok(())
    }
}

/// A successful adapter attestation.
///
/// The presence of this type means the adapter claims all required steps for this
/// exact request succeeded: method resolution, controller binding, permitted
/// verification relationship, proof-purpose match, and cryptographic verification.
/// The core does not independently establish any of those external facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationEvidence {
    pub schema_version: u16,
    pub claim_representation_digest: String,
    pub statement_digest: String,
    pub author: ClaimAuthorIdentity,
    pub proof_purpose: ClaimProofPurpose,
    pub verification_method: ClaimVerificationMethod,
    pub controller: ClaimControllerIdentity,
    /// Controller identity actually read from the resolved controller document.
    pub controller_document_controller: ClaimControllerIdentity,
    /// Verification-method identity actually read from the resolved controller document.
    pub controller_document_verification_method: ClaimVerificationMethod,
    pub verification_relationship: ClaimVerificationRelationship,
    pub controller_document_ref: String,
    pub controller_document_digest: String,
    pub cryptosuite: String,
    pub signed_payload_digest: String,
    pub proof_digest: String,
}

impl VerificationEvidence {
    /// Construct adapter evidence after the adapter has completed its external
    /// resolution and cryptographic checks.
    pub fn from_adapter_attestation(
        request: &VerificationRequest,
        controller_document_ref: impl Into<String>,
        controller_document_digest: impl Into<String>,
        controller_document_controller: ClaimControllerIdentity,
        controller_document_verification_method: ClaimVerificationMethod,
        verification_relationship: impl Into<String>,
        cryptosuite: impl Into<String>,
        signed_payload_digest: impl Into<String>,
        proof_digest: impl Into<String>,
    ) -> Result<Self, VerificationFailure> {
        request.validate_structure()?;

        let controller_document_ref = controller_document_ref.into();
        let controller_document_digest = controller_document_digest.into();
        let verification_relationship =
            ClaimVerificationRelationship::new(verification_relationship)?;
        if verification_relationship != request.expected_verification_relationship {
            return Err(VerificationFailure::VerificationRelationshipMismatch {
                expected: request.expected_verification_relationship.clone(),
                actual: verification_relationship,
            });
        }
        controller_document_controller.validate_structure()?;
        controller_document_verification_method.validate_structure()?;
        if controller_document_controller != request.expected_controller {
            return Err(VerificationFailure::ControllerMismatch {
                expected: request.expected_controller.clone(),
                actual: controller_document_controller,
            });
        }
        if controller_document_verification_method != request.verification_method {
            return Err(VerificationFailure::VerificationMethodMismatch {
                expected: request.verification_method.clone(),
                actual: controller_document_verification_method,
            });
        }
        let cryptosuite = cryptosuite.into();
        let signed_payload_digest = signed_payload_digest.into();
        let proof_digest = proof_digest.into();

        for (name, value) in [
            ("controller document reference", controller_document_ref.as_str()),
            ("verification relationship", verification_relationship.as_str()),
            ("cryptosuite", cryptosuite.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(VerificationFailure::Structural(format!(
                    "{name} must be non-empty"
                )));
            }
        }

        for (name, value) in [
            ("controller document digest", controller_document_digest.as_str()),
            ("signed payload digest", signed_payload_digest.as_str()),
            ("proof digest", proof_digest.as_str()),
        ] {
            if !is_hex_digest(value) {
                return Err(VerificationFailure::Structural(format!(
                    "{name} must be a 64-character hexadecimal digest"
                )));
            }
        }

        Ok(Self {
            schema_version: VERIFICATION_EVIDENCE_SCHEMA_VERSION,
            claim_representation_digest: request.claim_representation_digest.clone(),
            statement_digest: request.statement_digest.clone(),
            author: request.author.clone(),
            proof_purpose: request.proof_purpose.clone(),
            verification_method: request.verification_method.clone(),
            controller: request.expected_controller.clone(),
            controller_document_controller,
            controller_document_verification_method,
            verification_relationship,
            controller_document_ref,
            controller_document_digest,
            verification_relationship,
            cryptosuite,
            signed_payload_digest,
            proof_digest,
        })
    }

    /// A deterministic representation of the adapter's attestation record.
    ///
    /// This is an evidence-record identity, not a substitute for the cryptographic
    /// verification it describes.
    pub fn evidence_digest(&self) -> String {
        let encoded = (
            "symthaea:verification-evidence:v1",
            self.schema_version,
            &self.claim_representation_digest,
            &self.statement_digest,
            self.author.as_str(),
            self.proof_purpose.as_str(),
            self.verification_method.as_str(),
            self.controller.as_str(),
            &self.controller_document_ref,
            &self.controller_document_digest,
            &self.verification_relationship,
            &self.cryptosuite,
            &self.signed_payload_digest,
            &self.proof_digest,
        );
        let bytes = serde_json::to_vec(&encoded)
            .expect("verification evidence is serializable");
        crate::sha256_hex(&bytes)
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        if self.schema_version != VERIFICATION_EVIDENCE_SCHEMA_VERSION {
            return Err(VerificationFailure::Structural(
                "unsupported verification evidence schema version".into(),
            ));
        }
        let request = VerificationRequest {
            claim_representation_digest: self.claim_representation_digest.clone(),
            statement_digest: self.statement_digest.clone(),
            author: self.author.clone(),
            proof_purpose: self.proof_purpose.clone(),
            verification_method: self.verification_method.clone(),
            expected_controller: self.controller.clone(),
            expected_verification_relationship: self.verification_relationship.clone(),
        };
        request.validate_structure()?;
        self.controller_document_controller.validate_structure()?;
        self.controller_document_verification_method.validate_structure()?;
        if self.controller_document_controller != self.controller {
            return Err(VerificationFailure::ControllerMismatch {
                expected: self.controller.clone(),
                actual: self.controller_document_controller.clone(),
            });
        }
        if self.controller_document_verification_method != self.verification_method {
            return Err(VerificationFailure::VerificationMethodMismatch {
                expected: self.verification_method.clone(),
                actual: self.controller_document_verification_method.clone(),
            });
        }

        for (name, value) [
            ("controller document reference", self.controller_document_ref.as_str()),
            ("verification relationship", self.verification_relationship.as_str()),
            ("cryptosuite", self.cryptosuite.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(VerificationFailure::Structural(format!(
                    "{name} must be non-empty"
                )));
            }
        }
        for (name, value) in [
            ("controller document digest", self.controller_document_digest.as_str()),
            ("signed payload digest", self.signed_payload_digest.as_str()),
            ("proof digest", self.proof_digest.as_str()),
        ] {
            if !is_hex_digest(value) {
                return Err(VerificationFailure::Structural(format!(
                    "{name} must be a 64-character hexadecimal digest"
                )));
            }
        }
        Ok(())
    }

    pub fn matches_request(&self, request: &VerificationRequest) -> bool {
        self.validate_structure().is_ok()
            && self.claim_representation_digest == request.claim_representation_digest
            && self.statement_digest == request.statement_digest
            && self.author == request.author
            && self.proof_purpose == request.proof_purpose
            && self.verification_method == request.verification_method
            && self.controller == request.expected_controller
            && self.controller_document_controller == request.expected_controller
            && self.controller_document_verification_method == request.verification_method
            && self.verification_relationship == request.expected_verification_relationship
    }
}

/// Outcome vocabulary for an adapter.
///
/// `Unresolved` is intentionally separate from `Invalid`: a missing controller
/// document or verification method may be retryable, while a cryptographic or
/// structural failure is definitive for the current artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationOutcome {
    Verified(VerificationEvidence),
    Invalid(VerificationFailure),
    Unresolved(Vec<FederationDependency>),
}

impl VerificationOutcome {
    pub fn is_verified(&self) -> bool {
        matches!(self, Self::Verified(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationFailure {
    Structural(String),
    MissingAuthorship,
    MissingVerificationMethod,
    MissingVerificationController,
    ProofPurposeMismatch {
        expected: ClaimProofPurpose,
        actual: ClaimProofPurpose,
    },
    ControllerMismatch {
        expected: ClaimControllerIdentity,
        actual: ClaimControllerIdentity,
    },
    VerificationMethodMismatch {
        expected: ClaimVerificationMethod,
        actual: ClaimVerificationMethod,
    },
    VerificationRelationshipMismatch {
        expected: ClaimVerificationRelationship,
        actual: ClaimVerificationRelationship,
    },
    CryptographicVerificationFailed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CanonicalAdmissionReceipt, ProvenanceRelation, ProvenanceRelationKind,
        ProvenanceValidationReport, ProvenanceView,
    };

    fn fixture_claim() -> FederatedClaim {
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let validation = ProvenanceValidationReport::from_relations(std::slice::from_ref(&relation));
        let view = ProvenanceView::from_relations(std::slice::from_ref(&relation), validation.clone())
            .unwrap();
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-verification",
            Some("frontier:verification".into()),
            "canonical:verification",
            Some("family:verification".into()),
            view.snapshot_digest.clone(),
            view.validation.validator_version.clone(),
            view.validation.snapshot_schema_version,
        )
        .unwrap();

        FederatedClaim::new(
            "claim:verification",
            "canonical:verification",
            "family:verification",
            "author:verification",
            "statement:verification",
            view,
            receipt,
        )
        .unwrap()
        .with_authorship(
            crate::ClaimAuthorship::new(
                crate::ClaimAuthorIdentity::new("author:verification").unwrap(),
                ClaimProofPurpose::new("assertionMethod").unwrap(),
                Some(ClaimVerificationMethod::new(
                    "https://example.test/controller#key-1",
                ).unwrap()),
            )
            .unwrap()
            .with_verification_controller(
                ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn request_binds_exact_claim_purpose_and_controller() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
        )
        .unwrap();

        assert_eq!(request.author.as_str(), "author:verification");
        assert_eq!(
            request.verification_method.as_str(),
            "https://example.test/controller#key-1"
        );
        assert_eq!(request.dependencies().len(), 2);
        assert_eq!(
            request.expected_verification_relationship.as_str(),
            "assertionMethod"
        );
    }

    #[test]
    fn request_rejects_wrong_purpose_and_controller() {
        let claim = fixture_claim();
        let wrong_purpose = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("authentication").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
        )
        .unwrap_err();
        assert!(matches!(
            wrong_purpose,
            VerificationFailure::ProofPurposeMismatch { .. }
        ));

        let wrong_controller = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/other").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
        )
        .unwrap_err();
        assert!(matches!(
            wrong_controller,
            VerificationFailure::ControllerMismatch { .. }
        ));
    }

    #[test]
    fn evidence_binds_adapter_artifacts_to_exact_request() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
        )
        .unwrap();

        let evidence = VerificationEvidence::from_adapter_attestation(
            &request,
            "https://example.test/controller",
            &"11".repeat(32),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationMethod::new("https://example.test/controller#key-1").unwrap(),
            "assertionMethod",
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        assert!(evidence.validate_structure().is_ok());
        assert!(evidence.matches_request(&request));
        assert!(!evidence.evidence_digest().is_empty());
    }

    #[test]
    fn evidence_rejects_malformed_external_artifacts() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
        )
        .unwrap();

        assert!(matches!(
            VerificationEvidence::from_adapter_attestation(
                &request,
                "https://example.test/controller",
                "not-a-digest",
                ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
                ClaimVerificationMethod::new("https://example.test/controller#key-1").unwrap(),
                "assertionMethod",
                "ed25519",
                &"22".repeat(32),
                &"33".repeat(32),
            ),
            Err(VerificationFailure::Structural(_))
        ));
    }

    #[test]
    fn evidence_rejects_a_controller_relationship_different_from_the_request() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
        )
        .unwrap();

        assert!(matches!(
            VerificationEvidence::from_adapter_attestation(
                &request,
                "https://example.test/controller",
                &"11".repeat(32),
                ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
                ClaimVerificationMethod::new("https://example.test/controller#key-1").unwrap(),
                "authentication",
                "ed25519",
                &"22".repeat(32),
                &"33".repeat(32),
            ),
            Err(VerificationFailure::VerificationRelationshipMismatch { .. })
        ));
    }

    #[test]
    fn evidence_changes_when_controller_snapshot_or_claim_changes() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
        )
        .unwrap();
        let a = VerificationEvidence::from_adapter_attestation(
            &request,
            "https://example.test/controller",
            &"11".repeat(32),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationMethod::new("https://example.test/controller#key-1").unwrap(),
            "assertionMethod",
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        let b = VerificationEvidence::from_adapter_attestation(
            &request,
            "https://example.test/controller",
            &"44".repeat(32),
            "assertionMethod",
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        assert_ne!(a.evidence_digest(), b.evidence_digest());

        let mut changed_claim = claim.clone();
        changed_claim.source_event = Some("event:changed".into());
        let changed_request = VerificationRequest::from_claim(
            &changed_claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
        )
        .unwrap();
        let c = VerificationEvidence::from_adapter_attestation(
            &changed_request,
            "https://example.test/controller",
            &"11".repeat(32),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationMethod::new("https://example.test/controller#key-1").unwrap(),
            "assertionMethod",
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        assert_ne!(a.evidence_digest(), c.evidence_digest());
    }
}
