// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Final capability boundary for scientific qualification.
//!
//! `AuthorityLevel` intentionally stops at `Bound`; ordinary caller-constructible
//! records can never select `Qualified`. Qualification exists only as this
//! private-field capability, minted from an already lifecycle-authenticated claim
//! decision whose policy profile was separately lifecycle-authenticated.
//!
//! Qualification means the exact scientific claim lineage satisfied the exact
//! authenticated qualification profile and the exact qualification decision was
//! authorized. It does **not** mean scientific truth, global evidence
//! exhaustiveness, universal causal validity, global independence, or perpetual
//! current validity.

use serde::Serialize;
use symthaea_trust_core::Sha256Digest as TrustSha256Digest;

use crate::{
    ResearchId, SCIENCE_QUALIFICATION_DECISION_USAGE, Sha256Digest,
    VerifiedQualificationDecision,
};

pub const QUALIFIED_SCIENTIFIC_CLAIM_SCHEMA: &str =
    "symthaea.qualified-scientific-claim.v1";
const QUALIFICATION_DIGEST_DOMAIN: &str =
    "symthaea.qualified-scientific-claim.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScientificQualificationError {
    DecisionNotAuthenticated,
    DecisionAlreadyClaimsQualification,
    ReviewReadyAlreadyClaimsQualification,
    ProfileAuthorityAlreadyClaimsQualification,
    ProfileLineageMismatch,
    DecisionPurposeMismatch,
    DecisionSubjectMismatch,
    DecisionPayloadMismatch,
    DecisionContextMismatch,
    DecisionPredatesProfileAuthorization,
}

/// Non-forgeable scientific qualification capability.
///
/// The capability is serializable as retained evidence but intentionally not
/// deserializable. Rehydration must rerun the full chain:
/// evidence/claim binding -> coverage -> eligibility -> review readiness ->
/// profile authority -> decision authority -> qualification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualifiedScientificClaim {
    schema_version: String,
    claim_id: ResearchId,
    subject_sha256: Sha256Digest,
    qualification_sha256: Sha256Digest,
    decision: VerifiedQualificationDecision,
}

impl QualifiedScientificClaim {
    pub fn try_new(
        decision: VerifiedQualificationDecision,
    ) -> Result<Self, ScientificQualificationError> {
        if !decision.decision_authenticated() {
            return Err(ScientificQualificationError::DecisionNotAuthenticated);
        }
        if decision.qualification_established() {
            return Err(ScientificQualificationError::DecisionAlreadyClaimsQualification);
        }
        if decision.review_ready().qualification_established() {
            return Err(ScientificQualificationError::ReviewReadyAlreadyClaimsQualification);
        }
        if decision.authorized_profile().qualification_established() {
            return Err(ScientificQualificationError::ProfileAuthorityAlreadyClaimsQualification);
        }
        if decision.review_ready().profile_sha256()
            != decision.authorized_profile().profile_sha256()
        {
            return Err(ScientificQualificationError::ProfileLineageMismatch);
        }

        let profile = decision.authorized_profile().profile().profile();
        let envelope = decision.verified_attestation().envelope();
        if envelope.purpose.as_str() != SCIENCE_QUALIFICATION_DECISION_USAGE {
            return Err(ScientificQualificationError::DecisionPurposeMismatch);
        }
        if envelope.subject_sha256 != bridge_digest(&profile.subject_sha256) {
            return Err(ScientificQualificationError::DecisionSubjectMismatch);
        }
        if envelope.payload_sha256 != bridge_digest(decision.review_ready().readiness_sha256()) {
            return Err(ScientificQualificationError::DecisionPayloadMismatch);
        }
        if envelope.context_sha256.as_ref()
            != Some(
                decision
                    .authorized_profile()
                    .authorization_authority_sha256(),
            )
        {
            return Err(ScientificQualificationError::DecisionContextMismatch);
        }

        let profile_authorized_at = decision
            .authorized_profile()
            .verified_attestation()
            .evaluation_time_unix_s();
        let decision_authorized_at = decision
            .verified_attestation()
            .evaluation_time_unix_s();
        if decision_authorized_at < profile_authorized_at {
            return Err(ScientificQualificationError::DecisionPredatesProfileAuthorization);
        }

        let claim_id = profile.claim_id.clone();
        let subject_sha256 = profile.subject_sha256.clone();
        let qualification_sha256 = qualification_digest(
            &claim_id,
            &subject_sha256,
            decision.review_ready().readiness_sha256(),
            decision.authorized_profile().profile_sha256(),
            decision
                .authorized_profile()
                .authorization_authority_sha256(),
            decision.decision_authority_sha256(),
        );

        Ok(Self {
            schema_version: QUALIFIED_SCIENTIFIC_CLAIM_SCHEMA.into(),
            claim_id,
            subject_sha256,
            qualification_sha256,
            decision,
        })
    }

    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    pub fn claim_id(&self) -> &ResearchId {
        &self.claim_id
    }

    pub fn subject_sha256(&self) -> &Sha256Digest {
        &self.subject_sha256
    }

    pub fn qualification_sha256(&self) -> &Sha256Digest {
        &self.qualification_sha256
    }

    pub fn qualification_profile_sha256(&self) -> &Sha256Digest {
        self.decision.authorized_profile().profile_sha256()
    }

    /// Full profile authority grant identity, including its exact signature
    /// policy, trust snapshot, and evaluation time.
    pub fn profile_authority_sha256(&self) -> &TrustSha256Digest {
        self.decision
            .authorized_profile()
            .authorization_authority_sha256()
    }

    /// Full claim-decision authority grant identity, including its exact
    /// signature policy, trust snapshot, and evaluation time.
    pub fn decision_authority_sha256(&self) -> &TrustSha256Digest {
        self.decision.decision_authority_sha256()
    }

    pub fn profile_authorized_at_unix_s(&self) -> u64 {
        self.decision
            .authorized_profile()
            .verified_attestation()
            .evaluation_time_unix_s()
    }

    pub fn decision_authorized_at_unix_s(&self) -> u64 {
        self.decision
            .verified_attestation()
            .evaluation_time_unix_s()
    }

    pub fn decision(&self) -> &VerifiedQualificationDecision {
        &self.decision
    }

    /// Qualification is represented by possession of this capability, not by a
    /// caller-selectable `AuthorityLevel` variant.
    pub const fn qualification_established(&self) -> bool {
        true
    }

    /// Protocol-local evidence coverage is not global evidence exhaustiveness.
    pub const fn global_exhaustiveness_established(&self) -> bool {
        false
    }

    /// Graph-local replication lineage is not proof of global independence.
    pub const fn global_independence_established(&self) -> bool {
        false
    }

    /// This capability records a qualification event at retained evaluation
    /// times. Current validity requires a later revocation/supersession check.
    pub const fn current_validity_established(&self) -> bool {
        false
    }

    /// Scientific qualification is not a claim of immutable scientific truth.
    pub const fn scientific_truth_established(&self) -> bool {
        false
    }
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn qualification_digest(
    claim_id: &ResearchId,
    subject_sha256: &Sha256Digest,
    readiness_sha256: &Sha256Digest,
    profile_sha256: &Sha256Digest,
    profile_authority_sha256: &TrustSha256Digest,
    decision_authority_sha256: &TrustSha256Digest,
) -> Sha256Digest {
    let mut bytes = Vec::new();
    append_frame(&mut bytes, QUALIFICATION_DIGEST_DOMAIN);
    append_frame(&mut bytes, QUALIFIED_SCIENTIFIC_CLAIM_SCHEMA);
    append_frame(&mut bytes, claim_id.as_str());
    append_frame(&mut bytes, subject_sha256.as_str());
    append_frame(&mut bytes, readiness_sha256.as_str());
    append_frame(&mut bytes, profile_sha256.as_str());
    append_frame(&mut bytes, profile_authority_sha256.as_str());
    append_frame(&mut bytes, decision_authority_sha256.as_str());
    append_frame(&mut bytes, "qualification-established");
    append_frame(&mut bytes, "scientific-truth-not-established");
    append_frame(&mut bytes, "global-exhaustiveness-not-established");
    append_frame(&mut bytes, "global-independence-not-established");
    append_frame(&mut bytes, "current-validity-requires-recheck");
    Sha256Digest::of_bytes(&bytes)
}

fn append_frame(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AuthorityLevel;

    #[test]
    fn ordinary_authority_levels_still_have_no_qualified_variant() {
        let levels = [
            AuthorityLevel::None,
            AuthorityLevel::Declared,
            AuthorityLevel::Bound,
        ];
        assert_eq!(levels.len(), 3);
    }

    #[test]
    fn qualification_schema_is_explicitly_versioned() {
        assert_eq!(
            QUALIFIED_SCIENTIFIC_CLAIM_SCHEMA,
            "symthaea.qualified-scientific-claim.v1"
        );
    }
}
