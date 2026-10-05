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
use chrono::{DateTime, FixedOffset};
use serde::{Deserialize, Serialize};
use url::Url;

pub const VERIFICATION_REQUEST_SCHEMA_VERSION: u16 = 1;
pub const VERIFICATION_EVIDENCE_SCHEMA_VERSION: u16 = 1;
pub const VERIFICATION_EVIDENCE_DIGEST_VERSION: u16 = 2;

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

/// Temporal and anti-replay inputs supplied by the proof and the verifier.
///
/// The core validates syntax, temporal ordering, and exact domain/challenge matching.
/// It does not maintain a challenge-consumption store; one-time challenge tracking
/// remains an adapter/application responsibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationFreshnessContext {
    pub proof_created: Option<String>,
    pub proof_expires: Option<String>,
    pub proof_domain: Option<String>,
    pub proof_challenge: Option<String>,
    pub verification_time: String,
    pub expected_domain: Option<String>,
    pub expected_challenge: Option<String>,
}

impl VerificationFreshnessContext {
    pub fn validate(&self) -> Result<(), VerificationFailure> {
        let verification_time = parse_timestamp("verification time", &self.verification_time)?;

        let created = self
            .proof_created
            .as_deref()
            .map(|value| parse_timestamp("proof created", value))
            .transpose()?;

        let expires = self
            .proof_expires
            .as_deref()
            .map(|value| parse_timestamp("proof expires", value))
            .transpose()?;

        if let (Some(created), Some(expires)) = (created, expires) {
            if expires < created {
                return Err(VerificationFailure::InvalidValidityWindow);
            }
        }

        if created.is_some_and(|created| created > verification_time) {
            return Err(VerificationFailure::ProofCreatedInFuture);
        }

        if expires.is_some_and(|expires| verification_time >= expires) {
            return Err(VerificationFailure::ProofExpired);
        }

        for (name, value) in [
            ("proof domain", self.proof_domain.as_deref()),
            ("proof challenge", self.proof_challenge.as_deref()),
            ("expected domain", self.expected_domain.as_deref()),
            ("expected challenge", self.expected_challenge.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                return Err(VerificationFailure::Structural(format!(
                    "{name} must be non-empty when present"
                )));
            }
        }

        if let Some(expected_domain) = &self.expected_domain {
            if self.proof_domain.as_deref() != Some(expected_domain.as_str()) {
                return Err(VerificationFailure::DomainMismatch {
                    expected: expected_domain.clone(),
                    actual: self.proof_domain.clone(),
                });
            }
        }

        if let Some(expected_challenge) = &self.expected_challenge {
            if self.proof_challenge.as_deref() != Some(expected_challenge.as_str()) {
                return Err(VerificationFailure::ChallengeMismatch {
                    expected: expected_challenge.clone(),
                    actual: self.proof_challenge.clone(),
                });
            }
        }

        Ok(())
    }

    pub fn replay_context_digest(&self) -> String {
        let encoded = (
            "symthaea:verification-freshness:v1",
            &self.proof_domain,
            &self.proof_challenge,
            &self.expected_domain,
            &self.expected_challenge,
        );
        let bytes = serde_json::to_vec(&encoded)
            .expect("verification freshness context is serializable");
        crate::sha256_hex(&bytes)
    }
}

fn parse_timestamp(
    field: &'static str,
    value: &str,
) -> Result<DateTime<FixedOffset>, VerificationFailure> {
    DateTime::parse_from_rfc3339(value).map_err(|_| {
        VerificationFailure::InvalidTimestamp {
            field,
            value: value.to_owned(),
        }
    })
}

/// Typed identity of the concrete controlled-identifier document resource
/// dereferenced for a verification method. This is intentionally distinct from the
/// controller identity expressed by a verification method.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClaimControllerDocumentIdentity(String);

impl ClaimControllerDocumentIdentity {
    pub fn new(value: impl Into<String>) -> Result<Self, VerificationFailure> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(VerificationFailure::Structural(
                "controller document identity must be non-empty".into(),
            ));
        }
        Url::parse(&value)
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        Self::new(self.0.clone()).map(|_| ())
    }
}

/// Adapter attestation for the exact Controlled Identifiers verification-method
/// retrieval boundary.
///
/// This captures the security-critical result of the W3C retrieval algorithm without
/// performing dereferencing itself. The constructor receives the already-dereferenced
/// controller document facts and checks the invariants that can be evaluated purely
/// from those facts:
///
/// * the method identifier's primary resource is the controller document URL;
/// * the controller document's `id` is that URL;
/// * the resolved method identifier is exactly the requested method;
/// * the method's declared controller is exactly that controller-document URL; and
/// * the exact method is a member of the requested verification relationship.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationMethodResolution {
    pub schema_version: u16,
    pub verification_method: ClaimVerificationMethod,
    pub controller_document_ref: String,
    pub controller_document_id: ClaimControllerDocumentIdentity,
    pub resolved_verification_method_controller: ClaimControllerIdentity,
    pub verification_relationship: ClaimVerificationRelationship,
    pub relationship_methods: Vec<ClaimVerificationMethod>,
    pub relationship_methods_digest: String,
    pub controller_document_digest: String,
}

impl VerificationMethodResolution {
    pub const SCHEMA_VERSION: u16 = 1;

    pub fn from_controller_document(
        request: &VerificationRequest,
        controller_document_ref: impl Into<String>,
        controller_document_id: ClaimControllerDocumentIdentity,
        resolved_verification_method: ClaimVerificationMethod,
        resolved_verification_method_controller: ClaimControllerIdentity,
        relationship_methods: &[ClaimVerificationMethod],
        controller_document_digest: impl Into<String>,
    ) -> Result<Self, VerificationFailure> {
        request.validate_structure()?;
        let controller_document_ref = controller_document_ref.into();
        let controller_document_digest = controller_document_digest.into();

        let method_url = Url::parse(request.verification_method.as_str())
            .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
        if method_url.fragment().is_none() {
            return Err(VerificationFailure::InvalidVerificationMethodUrl);
        }
        let mut expected_document_url = method_url.clone();
        expected_document_url.set_fragment(None);
        let expected_document_url = expected_document_url.to_string();

        if controller_document_ref != expected_document_url {
            return Err(VerificationFailure::ControllerDocumentMismatch {
                expected: expected_document_url,
                actual: controller_document_ref,
            });
        }

        if resolved_verification_method != request.verification_method {
            return Err(VerificationFailure::VerificationMethodMismatch {
                expected: request.verification_method.clone(),
                actual: resolved_verification_method,
            });
        }

        controller_document_id.validate_structure()
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        let document_id = Url::parse(controller_document_id.as_str())
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        if document_id.as_str() != controller_document_ref {
            return Err(VerificationFailure::ControllerDocumentMismatch {
                expected: controller_document_ref.clone(),
                actual: controller_document_id.as_str().to_owned(),
            });
        }

        resolved_verification_method_controller
            .validate_structure()
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        let resolved_controller = Url::parse(resolved_verification_method_controller.as_str())
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        if resolved_controller.as_str() != controller_document_ref {
            return Err(VerificationFailure::ControllerMismatch {
                expected: ClaimControllerIdentity::new(controller_document_ref.clone())
                    .expect("validated controller document URL"),
                actual: resolved_verification_method_controller,
            });
        }

        if request.expected_controller.as_str() != controller_document_ref {
            return Err(VerificationFailure::ControllerMismatch {
                expected: request.expected_controller.clone(),
                actual: ClaimControllerIdentity::new(controller_document_ref.clone())
                    .expect("validated controller document URL"),
            });
        }

        let mut methods = relationship_methods.to_vec();
        for method in &methods {
            method.validate_structure()
                .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
            Url::parse(method.as_str())
                .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
        }
        methods.sort();
        methods.dedup();

        if !methods.iter().any(|method| method == &request.verification_method) {
            return Err(VerificationFailure::VerificationRelationshipMismatch {
                expected: request.expected_verification_relationship.clone(),
                actual: ClaimVerificationRelationship::new(
                    "unbound-verification-method"
                )?,
            });
        }

        let encoded_members =
            ("symthaea:verification-relationship-members:v1",
             request.expected_verification_relationship.as_str(),
             methods.iter().map(ClaimVerificationMethod::as_str).collect::<Vec<_>>());
        let bytes = serde_json::to_vec(&encoded_members)
            .expect("verification relationship member set is serializable");
        let relationship_methods_digest = crate::sha256_hex(&bytes);

        if !is_hex_digest(&controller_document_digest) {
            return Err(VerificationFailure::Structural(
                "controller document digest must be a 64-character hexadecimal digest".into(),
            ));
        }

        Ok(Self {
            schema_version: Self::SCHEMA_VERSION,
            verification_method: request.verification_method.clone(),
            controller_document_ref,
            controller_document_id,
            resolved_verification_method_controller,
            verification_relationship: request.expected_verification_relationship.clone(),
            relationship_methods: methods,
            relationship_methods_digest,
            controller_document_digest,
        })
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        if self.schema_version != Self::SCHEMA_VERSION {
            return Err(VerificationFailure::Structural(
                "unsupported verification method resolution schema version".into(),
            ));
        }
        Url::parse(self.verification_method.as_str())
            .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;

        let mut expected_document_url =
            Url::parse(self.verification_method.as_str())
                .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
        expected_document_url.set_fragment(None);
        if expected_document_url.as_str() != self.controller_document_ref {
            return Err(VerificationFailure::ControllerDocumentMismatch {
                expected: expected_document_url.to_string(),
                actual: self.controller_document_ref.clone(),
            });
        }

        let document_id = Url::parse(self.controller_document_id.as_str())
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        if document_id.as_str() != self.controller_document_ref {
            return Err(VerificationFailure::ControllerDocumentMismatch {
                expected: self.controller_document_ref.clone(),
                actual: self.controller_document_id.as_str().to_owned(),
            });
        }

        let resolved_controller =
            Url::parse(self.resolved_verification_method_controller.as_str())
                .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        if resolved_controller.as_str() != self.controller_document_ref {
            return Err(VerificationFailure::ControllerMismatch {
                expected: ClaimControllerIdentity::new(self.controller_document_ref.clone())
                    .expect("validated controller document URL"),
                actual: self.resolved_verification_method_controller.clone(),
            });
        }

        self.verification_relationship.validate_structure()?;
        let mut canonical_methods = self.relationship_methods.clone();
        for method in &canonical_methods {
            method
                .validate_structure()
                .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
            Url::parse(method.as_str())
                .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
        }
        canonical_methods.sort();
        canonical_methods.dedup();
        if canonical_methods != self.relationship_methods {
            return Err(VerificationFailure::Structural(
                "relationship method set must be sorted and unique".into(),
            ));
        }
        if !self
            .relationship_methods
            .iter()
            .any(|method| method == &self.verification_method)
        {
            return Err(VerificationFailure::VerificationMethodNotInRelationship);
        }
        let encoded_members = (
            "symthaea:verification-relationship-members:v1",
            self.verification_relationship.as_str(),
            self.relationship_methods
                .iter()
                .map(ClaimVerificationMethod::as_str)
                .collect::<Vec<_>>(),
        );
        let bytes = serde_json::to_vec(&encoded_members)
            .expect("verification relationship member set is serializable");
        if crate::sha256_hex(&bytes) != self.relationship_methods_digest {
            return Err(VerificationFailure::Structural(
                "relationship methods digest does not match member set".into(),
            ));
        }
        if !is_hex_digest(&self.relationship_methods_digest) {
            return Err(VerificationFailure::Structural(
                "relationship methods digest must be a 64-character hexadecimal digest".into(),
            ));
        }
        if !is_hex_digest(&self.controller_document_digest) {
            return Err(VerificationFailure::Structural(
                "controller document digest must be a 64-character hexadecimal digest".into(),
            ));
        }
        Ok(())
    }

    pub fn resolution_digest(&self) -> String {
        let encoded = (
            "symthaea:verification-method-resolution:v1",
            self.schema_version,
            self.verification_method.as_str(),
            self.controller_document_ref.as_str(),
            self.controller_document_id.as_str(),
            self.resolved_verification_method_controller.as_str(),
            self.verification_relationship.as_str(),
            self.relationship_methods_digest.as_str(),
            self.controller_document_digest.as_str(),
        );
        let bytes = serde_json::to_vec(&encoded)
            .expect("verification method resolution is serializable");
        crate::sha256_hex(&bytes)
    }

    pub fn matches_request(&self, request: &VerificationRequest) -> bool {
        self.validate_structure().is_ok()
            && self.verification_method == request.verification_method
            && self.resolved_verification_method_controller == request.expected_controller
            && self.verification_relationship == request.expected_verification_relationship
    }
}

/// The exact structural and identity inputs an external verification adapter must
/// operate over before it can return cryptographic/controller evidence.
///
/// The request is derived from the claim itself, so adapters cannot accidentally
/// verify one claim while attaching evidence to another representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationRequest {
    pub schema_version: u16,
    pub claim_representation_digest: String,
    pub statement_digest: String,
    pub author: ClaimAuthorIdentity,
    pub proof_purpose: ClaimProofPurpose,
    pub verification_method: ClaimVerificationMethod,
    pub expected_controller: ClaimControllerIdentity,
    pub expected_verification_relationship: ClaimVerificationRelationship,
    pub freshness: VerificationFreshnessContext,
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
        freshness: VerificationFreshnessContext,
    ) -> Result<Self, VerificationFailure> {
        claim
            .validate_structure()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?;
        freshness.validate()?;

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
            schema_version: VERIFICATION_REQUEST_SCHEMA_VERSION,
            claim_representation_digest: claim.canonical_digest(),
            statement_digest,
            author: claim.author.clone(),
            proof_purpose: expected_purpose,
            verification_method,
            expected_controller,
            expected_verification_relationship,
            freshness,
        })
    }

    pub fn controller_document_ref(&self) -> Result<String, VerificationFailure> {
        let method_url = Url::parse(self.verification_method.as_str())
            .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
        if method_url.fragment().is_none() {
            return Err(VerificationFailure::InvalidVerificationMethodUrl);
        }
        let mut document_url = method_url;
        document_url.set_fragment(None);
        Ok(document_url.to_string())
    }

    /// Typed external dependencies required for the adapter-side resolution step.
    ///
    /// The core deliberately does not turn these into fetches; an adapter may resolve
    /// them through a DID/controller-document system, Holochain, a local cache, or any
    /// other substrate.
    pub fn dependencies(&self) -> Result<Vec<FederationDependency>, VerificationFailure> {
        Ok(vec![
            FederationDependency::VerificationMethod(self.verification_method.as_str().to_owned()),
            FederationDependency::ControllerDocument(self.controller_document_ref()?),
        ])
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        if self.schema_version != VERIFICATION_REQUEST_SCHEMA_VERSION {
            return Err(VerificationFailure::Structural(
                "unsupported verification request schema version".into(),
            ));
        }
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
        self.freshness.validate()?;
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
    /// Controller identity actually read from the resolved verification-method definition.
    ///
    /// This is intentionally distinct from the identity of the controller document
    /// itself: a verification method's controller MUST be checked explicitly.
    pub resolved_verification_method_controller: ClaimControllerIdentity,
    /// Verification-method identity actually read from the resolved controller document.
    pub controller_document_verification_method: ClaimVerificationMethod,
    pub verification_relationship: ClaimVerificationRelationship,
    pub controller_document_ref: String,
    pub controller_document_digest: String,
    pub resolution: VerificationMethodResolution,
    pub cryptosuite: String,
    pub freshness: VerificationFreshnessContext,
    pub signed_payload_digest: String,
    pub proof_digest: String,
}

impl VerificationEvidence {
    /// Construct adapter evidence after the adapter has completed its external
    /// resolution and cryptographic checks.
    pub fn from_adapter_attestation(
        request: &VerificationRequest,
        resolution: VerificationMethodResolution,
        cryptosuite: impl Into<String>,
        signed_payload_digest: impl Into<String>,
        proof_digest: impl Into<String>,
    ) -> Result<Self, VerificationFailure> {
        request.validate_structure()?;
        resolution.validate_structure()?;
        if !resolution.matches_request(request) {
            return Err(VerificationFailure::ResolutionRequestMismatch);
        }
        let cryptosuite = cryptosuite.into();
        let signed_payload_digest = signed_payload_digest.into();
        let proof_digest = proof_digest.into();

        for (name, value) in [
            ("cryptosuite", cryptosuite.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(VerificationFailure::Structural(format!(
                    "{name} must be non-empty"
                )));
            }
        }

        for (name, value) in [
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
            verification_method: resolution.verification_method.clone(),
            controller: request.expected_controller.clone(),
            resolved_verification_method_controller:
                resolution.resolved_verification_method_controller.clone(),
            controller_document_verification_method: resolution.verification_method.clone(),
            verification_relationship: resolution.verification_relationship.clone(),
            controller_document_ref: resolution.controller_document_ref.clone(),
            controller_document_digest: resolution.controller_document_digest.clone(),
            resolution,
            cryptosuite,
            freshness: request.freshness.clone(),
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
            "symthaea:verification-evidence:v2",
            self.schema_version,
            VERIFICATION_EVIDENCE_DIGEST_VERSION,
            &self.claim_representation_digest,
            &self.statement_digest,
            self.author.as_str(),
            self.proof_purpose.as_str(),
            self.verification_method.as_str(),
            self.controller.as_str(),
            self.resolved_verification_method_controller.as_str(),
            self.controller_document_verification_method.as_str(),
            &self.controller_document_ref,
            &self.controller_document_digest,
            &self.resolution.resolution_digest(),
            &self.verification_relationship,
            &self.cryptosuite,
            &self.signed_payload_digest,
            &self.proof_digest,
            &self.freshness.replay_context_digest(),
            &self.freshness.verification_time,
            &self.freshness.proof_created,
            &self.freshness.proof_expires,
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
            schema_version: VERIFICATION_REQUEST_SCHEMA_VERSION,
            claim_representation_digest: self.claim_representation_digest.clone(),
            statement_digest: self.statement_digest.clone(),
            author: self.author.clone(),
            proof_purpose: self.proof_purpose.clone(),
            verification_method: self.verification_method.clone(),
            expected_controller: self.controller.clone(),
            expected_verification_relationship: self.verification_relationship.clone(),
            freshness: self.freshness.clone(),
        };
        request.validate_structure()?;
        self.resolution.validate_structure()?;
        if !self.resolution.matches_request(&request) {
            return Err(VerificationFailure::ResolutionRequestMismatch);
        }
        if self.resolution.controller_document_ref != self.controller_document_ref
            || self.resolution.controller_document_digest != self.controller_document_digest
            || self.resolution.verification_method != self.controller_document_verification_method
            || self.resolution.resolved_verification_method_controller
                != self.resolved_verification_method_controller
            || self.resolution.verification_relationship != self.verification_relationship
        {
            return Err(VerificationFailure::ResolutionEvidenceMismatch);
        }
        if self.freshness != request.freshness {
            return Err(VerificationFailure::FreshnessMismatch);
        }
        self.freshness.validate()?;
        if self.freshness != request.freshness {
            return Err(VerificationFailure::FreshnessMismatch);
        }
        self.resolved_verification_method_controller.validate_structure()?;
        self.controller_document_verification_method.validate_structure()?;
        if self.resolved_verification_method_controller != self.controller {
            return Err(VerificationFailure::ControllerMismatch {
                expected: self.controller.clone(),
                actual: self.resolved_verification_method_controller.clone(),
            });
        }
        if self.controller_document_verification_method != self.verification_method {
            return Err(VerificationFailure::VerificationMethodMismatch {
                expected: self.verification_method.clone(),
                actual: self.controller_document_verification_method.clone(),
            });
        }

        for (name, value) in [
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
        if !is_hex_digest(&self.resolution_digest) {
            return Err(VerificationFailure::Structural(
                "resolution digest must be a 64-character hexadecimal digest".into(),
            ));
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
            && self.resolved_verification_method_controller == request.expected_controller
            && self.controller_document_verification_method == request.verification_method
            && self.verification_relationship == request.expected_verification_relationship
            && self.freshness == request.freshness
            && self.resolution.matches_request(request)
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
    InvalidVerificationMethodUrl,
    InvalidControllerDocumentId,
    ControllerDocumentMismatch {
        expected: String,
        actual: String,
    },
    VerificationRelationshipMismatch {
        expected: ClaimVerificationRelationship,
        actual: ClaimVerificationRelationship,
    },
    VerificationMethodNotInRelationship,
    InvalidTimestamp {
        field: &'static str,
        value: String,
    },
    InvalidValidityWindow,
    ProofCreatedInFuture,
    ProofExpired,
    DomainMismatch {
        expected: String,
        actual: Option<String>,
    },
    ChallengeMismatch {
        expected: String,
        actual: Option<String>,
    },
    FreshnessMismatch,
    ResolutionRequestMismatch,
    ResolutionEvidenceMismatch,
    CryptographicVerificationFailed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CanonicalAdmissionReceipt, ProvenanceRelation, ProvenanceRelationKind,
        ProvenanceValidationReport, ProvenanceView,
    };

    fn freshness(
        proof_created: Option<&str>,
        proof_expires: Option<&str>,
        proof_domain: Option<&str>,
        proof_challenge: Option<&str>,
        verification_time: &str,
        expected_domain: Option<&str>,
        expected_challenge: Option<&str>,
    ) -> VerificationFreshnessContext {
        VerificationFreshnessContext {
            proof_created: proof_created.map(str::to_owned),
            proof_expires: proof_expires.map(str::to_owned),
            proof_domain: proof_domain.map(str::to_owned),
            proof_challenge: proof_challenge.map(str::to_owned),
            verification_time: verification_time.to_owned(),
            expected_domain: expected_domain.map(str::to_owned),
            expected_challenge: expected_challenge.map(str::to_owned),
        }
    }

    fn default_freshness() -> VerificationFreshnessContext {
        freshness(
            Some("2026-10-05T00:00:00Z"),
            Some("2026-10-05T03:00:00Z"),
            Some("example.test"),
            Some("challenge-1"),
            "2026-10-05T02:00:00Z",
            Some("example.test"),
            Some("challenge-1"),
        )
    }

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

    fn resolved_method(request: &VerificationRequest) -> VerificationMethodResolution {
        VerificationMethodResolution::from_controller_document(
            request,
            "https://example.test/controller",
            ClaimControllerDocumentIdentity::new("https://example.test/controller").unwrap(),
            request.verification_method.clone(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            &[request.verification_method.clone()],
            &"11".repeat(32),
        )
        .unwrap()
    }

    fn make_evidence(
        request: &VerificationRequest,
        controller_document_digest: &str,
        cryptosuite: &str,
        signed_payload_digest: &str,
        proof_digest: &str,
    ) -> Result<VerificationEvidence, VerificationFailure> {
        let mut resolution = resolved_method(request);
        resolution.controller_document_digest = controller_document_digest.to_owned();
        VerificationEvidence::from_adapter_attestation(
            request,
            resolution,
            cryptosuite,
            signed_payload_digest,
            proof_digest,
        )
    }




    #[test]
    fn freshness_context_rejects_future_expiry_and_mismatched_replay_inputs() {
        let valid = default_freshness();
        assert!(valid.validate().is_ok());

        let mut future_created = valid.clone();
        future_created.proof_created = Some("2026-10-05T04:00:00Z".into());
        assert!(matches!(
            future_created.validate(),
            Err(VerificationFailure::ProofCreatedInFuture)
        ));

        let mut expired = valid.clone();
        expired.proof_expires = Some("2026-10-05T02:00:00Z".into());
        assert!(matches!(
            expired.validate(),
            Err(VerificationFailure::ProofExpired)
        ));

        let mut invalid_window = valid.clone();
        invalid_window.proof_created = Some("2026-10-05T02:30:00Z".into());
        invalid_window.proof_expires = Some("2026-10-05T02:15:00Z".into());
        assert!(matches!(
            invalid_window.validate(),
            Err(VerificationFailure::InvalidValidityWindow)
        ));

        let mut wrong_domain = valid.clone();
        wrong_domain.proof_domain = Some("other.example".into());
        assert!(matches!(
            wrong_domain.validate(),
            Err(VerificationFailure::DomainMismatch { .. })
        ));

        let mut wrong_challenge = valid;
        wrong_challenge.proof_challenge = Some("challenge-2".into());
        assert!(matches!(
            wrong_challenge.validate(),
            Err(VerificationFailure::ChallengeMismatch { .. })
        ));
    }

    #[test]
    fn freshness_rejects_malformed_timestamps_and_blank_security_context() {
        let mut malformed = default_freshness();
        malformed.verification_time = "2026-10-05T02:00:00".into();
        assert!(matches!(
            malformed.validate(),
            Err(VerificationFailure::InvalidTimestamp { field: "verification time", .. })
        ));

        let mut blank_domain = default_freshness();
        blank_domain.expected_domain = Some("   ".into());
        assert!(matches!(
            blank_domain.validate(),
            Err(VerificationFailure::Structural(_))
        ));
    }

    #[test]
    fn replay_context_digest_changes_when_domain_or_challenge_changes() {
        let base = default_freshness();
        let mut domain = base.clone();
        domain.proof_domain = Some("other.example".into());
        let mut challenge = base.clone();
        challenge.proof_challenge = Some("challenge-2".into());
        assert_ne!(base.replay_context_digest(), domain.replay_context_digest());
        assert_ne!(base.replay_context_digest(), challenge.replay_context_digest());
    }


    #[test]
    fn request_binds_exact_claim_purpose_and_controller() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        assert_eq!(request.schema_version, VERIFICATION_REQUEST_SCHEMA_VERSION);
        assert_eq!(request.author.as_str(), "author:verification");
        assert_eq!(
            request.verification_method.as_str(),
            "https://example.test/controller#key-1"
        );
        assert_eq!(request.dependencies().unwrap().len(), 2);
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
            default_freshness(),
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
            default_freshness(),
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
            default_freshness(),
        )
        .unwrap();

        let evidence = make_evidence(
            &request,
            &"11".repeat(32),
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
            default_freshness(),
        )
        .unwrap();

        assert!(matches!(
            {
            let result = {
                let mut resolution = resolved_method(&request);
                resolution.controller_document_digest = "not-a-digest".into();
                VerificationEvidence::from_adapter_attestation(
                    &request,
                    resolution,
                    "ed25519",
                    &"22".repeat(32),
                    &"33".repeat(32),
                )
            },
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
            default_freshness(),
        )
        .unwrap();

        assert!(matches!(
            {
            let mut resolution = resolved_method(&request);
            resolution.verification_relationship =
                ClaimVerificationRelationship::new("authentication").unwrap();
            VerificationEvidence::from_adapter_attestation(
                &request,
                resolution,
                "ed25519",
                &"22".repeat(32),
                &"33".repeat(32),
            )
        },
            Err(VerificationFailure::VerificationRelationshipMismatch { .. })
        ));
    }

    #[test]
    fn evidence_rejects_resolved_verification_method_controller_substitution() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let result = VerificationEvidence::from_adapter_attestation(
            &request,
            "https://example.test/controller",
            &"11".repeat(32),
            ClaimControllerIdentity::new("https://example.test/other").unwrap(),
            ClaimVerificationMethod::new("https://example.test/controller#key-1").unwrap(),
            &"55".repeat(32),
            "assertionMethod",
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        );

        assert!(matches!(
            result,
            Err(VerificationFailure::ControllerMismatch { .. })
        ));
    }

    #[test]
    fn evidence_rejects_controller_document_method_substitution() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let result = VerificationEvidence::from_adapter_attestation(
            &request,
            "https://example.test/controller",
            &"11".repeat(32),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationMethod::new("https://example.test/controller#key-2").unwrap(),
            "assertionMethod",
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        );

        assert!(matches!(
            result,
            Err(VerificationFailure::VerificationMethodMismatch { .. })
        ));
    }

    #[test]
    #[test]
    fn evidence_cannot_be_replayed_under_a_different_freshness_context() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let evidence = make_evidence(
            &request,
            &"11".repeat(32),
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        let mut replayed = evidence.clone();
        replayed.freshness.proof_challenge = Some("challenge-2".into());

        assert!(!replayed.matches_request(&request));
        assert_ne!(evidence.evidence_digest(), replayed.evidence_digest());
    }

    #[test]
    fn evidence_digest_covers_resolved_method_and_controller_assertions() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let base = make_evidence(
            &request,
            &"11".repeat(32),
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        let mut method = base.clone();
        method.controller_document_verification_method =
            ClaimVerificationMethod::new("https://example.test/controller#key-2").unwrap();
        assert_ne!(base.evidence_digest(), method.evidence_digest());

        let mut controller = base;
        controller.resolved_verification_method_controller =
            ClaimControllerIdentity::new("https://example.test/other").unwrap();
        assert_ne!(controller.evidence_digest(), method.evidence_digest());
    }

    fn evidence_changes_when_controller_snapshot_or_claim_changes() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();
        let a = make_evidence(
            &request,
            &"11".repeat(32),
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        let b = make_evidence(
            &request,
            &"44".repeat(32),
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
            default_freshness(),
        )
        .unwrap();
        let c = VerificationEvidence::from_adapter_attestation(
            &changed_request,
            "https://example.test/controller",
            &"11".repeat(32),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationMethod::new("https://example.test/controller#key-1").unwrap(),
            &"55".repeat(32),
            "assertionMethod",
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        assert_ne!(a.evidence_digest(), c.evidence_digest());
    }
}
