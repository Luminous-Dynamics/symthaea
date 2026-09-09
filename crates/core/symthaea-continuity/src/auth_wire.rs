// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical bytes for cryptographic authentication of continuity verification claims.
//!
//! This module defines only the application payload that an external authenticator
//! (for example Xenia) may sign/verify. It does not perform cryptography and it does
//! not turn a valid signature into continuity qualification or execution authority.

use crate::verifier::{
    VerificationAdmissionError, VerificationEvidenceClaimV1, VerificationOutcomeV1,
};

/// Stable schema label for the exact authentication payload defined by this module.
pub const CONTINUITY_VERIFICATION_CLAIM_AUTH_SCHEMA: &str =
    "symthaea-continuity-verification-claim-auth-v1";

/// Stable Xenia statement purpose reserved for continuity verification claims.
///
/// This string belongs in Xenia's outer purpose/domain binding. The payload below
/// independently carries its own Symthaea domain and schema so neither layer silently
/// substitutes for the other.
pub const CONTINUITY_VERIFICATION_XENIA_PURPOSE: &str =
    "org.luminous.symthaea.continuity.verification-claim.v1";

/// Hash algorithm used by [`canonical_verification_claim_digest`].
pub const CONTINUITY_VERIFICATION_CLAIM_HASH_ALGORITHM: &str = "blake3-256";

const AUTH_BYTES_DOMAIN: &[u8] = b"symthaea.continuity.verification-claim.auth-bytes.v1\0";

/// Return the exact v1 byte string that an external authenticator must authenticate.
///
/// The layout is deliberately hand-written rather than delegated to Serde/bincode so
/// a serializer upgrade cannot silently change the cryptographic statement.
///
/// Layout, in order:
///
/// 1. fixed domain bytes;
/// 2. `u64` little-endian schema length + UTF-8 schema bytes;
/// 3. contract id (32 bytes);
/// 4. target realization id (32 bytes);
/// 5. continuity requirement id (32 bytes);
/// 6. verifier profile id (32 bytes);
/// 7. transaction challenge (32 bytes);
/// 8. observation time as `u64` little-endian milliseconds;
/// 9. verification outcome tag (one byte);
/// 10. raw evidence digest (32 bytes);
/// 11. canonical verification-claim id (32 bytes).
///
/// `claim.validate()` runs first, so malformed/deserialized claim identities cannot
/// acquire valid authentication bytes.
pub fn canonical_verification_claim_bytes(
    claim: &VerificationEvidenceClaimV1,
) -> Result<Vec<u8>, VerificationAdmissionError> {
    claim.validate()?;

    let mut bytes = Vec::with_capacity(
        AUTH_BYTES_DOMAIN.len()
            + 8
            + CONTINUITY_VERIFICATION_CLAIM_AUTH_SCHEMA.len()
            + 7 * 32
            + 8
            + 1,
    );
    bytes.extend_from_slice(AUTH_BYTES_DOMAIN);
    put_str(&mut bytes, CONTINUITY_VERIFICATION_CLAIM_AUTH_SCHEMA);
    bytes.extend_from_slice(claim.contract_id().as_bytes());
    bytes.extend_from_slice(claim.target_realization_id().as_bytes());
    bytes.extend_from_slice(claim.requirement_id().as_bytes());
    bytes.extend_from_slice(claim.verifier_profile_id().as_bytes());
    bytes.extend_from_slice(&claim.transaction_challenge());
    bytes.extend_from_slice(&claim.observed_at_unix_ms().to_le_bytes());
    bytes.push(outcome_tag(claim.outcome()));
    bytes.extend_from_slice(&claim.raw_evidence_digest());
    bytes.extend_from_slice(claim.id().as_bytes());
    Ok(bytes)
}

/// BLAKE3-256 digest of the exact authentication payload.
///
/// A future Xenia adapter can compare this value directly with the payload digest on
/// Xenia's opaque authenticated-statement result after Xenia has authenticated the
/// exact bytes returned by [`canonical_verification_claim_bytes`].
pub fn canonical_verification_claim_digest(
    claim: &VerificationEvidenceClaimV1,
) -> Result<[u8; 32], VerificationAdmissionError> {
    let bytes = canonical_verification_claim_bytes(claim)?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

fn outcome_tag(outcome: VerificationOutcomeV1) -> u8 {
    match outcome {
        VerificationOutcomeV1::Satisfied => 1,
        VerificationOutcomeV1::Failed => 2,
        VerificationOutcomeV1::Inconclusive => 3,
        VerificationOutcomeV1::InfrastructureFailure => 4,
        VerificationOutcomeV1::NotExecuted => 5,
    }
}

fn put_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{
        ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, EquivalencePredicate,
        RequirementCriticality, ValidatedContinuityContractV1,
    };
    use crate::observation::{
        DependencyBasis, DependencyClaimV1, EvidenceBasis, ObservationCoverage,
        ObservationEnvelopeV1,
    };
    use crate::verifier::{VerificationEvidenceClaimV1, VerifierProfileV1};
    use crate::witness::{EvidenceClass, TargetRealizationId};

    fn contract() -> ValidatedContinuityContractV1 {
        let observation = ObservationEnvelopeV1::new(
            "machine-1",
            "workflow.dependency",
            "fixture",
            "1",
            1_700_000_000_000,
            ObservationCoverage::Complete,
            EvidenceBasis::Tested,
            [1; 32],
            vec![],
        )
        .unwrap();
        let dependency = DependencyClaimV1::new(
            "role:research",
            "requires",
            "capability:cuda",
            DependencyBasis::Observed,
            vec![observation.id()],
            vec![],
        )
        .unwrap();
        let requirement = ContinuityRequirementV1::new(
            dependency.id(),
            "cuda-workflow",
            RequirementCriticality::Must,
            EquivalencePredicate::BehavioralScenario {
                scenario_id: "cuda-fixture-v1".into(),
            },
            ApprovalBasis::ExplicitPolicy,
            [2; 32],
        )
        .unwrap();
        ContinuityContractV1::new("research-fleet", [3; 32], vec![requirement])
            .unwrap()
            .validate()
            .unwrap()
    }

    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "hardware-verifier-v1",
            [9; 32],
            7,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    fn claim(
        target: TargetRealizationId,
        challenge: [u8; 32],
        outcome: VerificationOutcomeV1,
        raw_evidence_digest: [u8; 32],
    ) -> VerificationEvidenceClaimV1 {
        let contract = contract();
        let profile = profile();
        VerificationEvidenceClaimV1::new(
            contract.id(),
            target,
            contract.requirements()[0].id(),
            profile.id(),
            challenge,
            1_700_000_000_111,
            outcome,
            raw_evidence_digest,
        )
        .unwrap()
    }

    #[test]
    fn authentication_bytes_are_deterministic_and_versioned() {
        let claim = claim(
            TargetRealizationId::from_digest([4; 32]).unwrap(),
            [5; 32],
            VerificationOutcomeV1::Satisfied,
            [8; 32],
        );
        let a = canonical_verification_claim_bytes(&claim).unwrap();
        let b = canonical_verification_claim_bytes(&claim).unwrap();
        assert_eq!(a, b);
        assert!(a.starts_with(AUTH_BYTES_DOMAIN));
        assert_eq!(
            a.len(),
            AUTH_BYTES_DOMAIN.len()
                + 8
                + CONTINUITY_VERIFICATION_CLAIM_AUTH_SCHEMA.len()
                + 7 * 32
                + 8
                + 1
        );
        assert_eq!(
            CONTINUITY_VERIFICATION_XENIA_PURPOSE,
            "org.luminous.symthaea.continuity.verification-claim.v1"
        );
        assert_eq!(
            canonical_verification_claim_digest(&claim).unwrap(),
            *blake3::hash(&a).as_bytes()
        );
    }

    #[test]
    fn exact_semantic_changes_change_authentication_bytes() {
        let base = claim(
            TargetRealizationId::from_digest([4; 32]).unwrap(),
            [5; 32],
            VerificationOutcomeV1::Satisfied,
            [8; 32],
        );
        let changed_target = claim(
            TargetRealizationId::from_digest([6; 32]).unwrap(),
            [5; 32],
            VerificationOutcomeV1::Satisfied,
            [8; 32],
        );
        let changed_challenge = claim(
            TargetRealizationId::from_digest([4; 32]).unwrap(),
            [7; 32],
            VerificationOutcomeV1::Satisfied,
            [8; 32],
        );
        let changed_outcome = claim(
            TargetRealizationId::from_digest([4; 32]).unwrap(),
            [5; 32],
            VerificationOutcomeV1::Failed,
            [8; 32],
        );
        let changed_evidence = claim(
            TargetRealizationId::from_digest([4; 32]).unwrap(),
            [5; 32],
            VerificationOutcomeV1::Satisfied,
            [10; 32],
        );

        let base_bytes = canonical_verification_claim_bytes(&base).unwrap();
        for candidate in [
            changed_target,
            changed_challenge,
            changed_outcome,
            changed_evidence,
        ] {
            assert_ne!(
                base_bytes,
                canonical_verification_claim_bytes(&candidate).unwrap()
            );
        }
    }

    #[test]
    fn layout_commits_observation_time_outcome_and_claim_identity() {
        let claim = claim(
            TargetRealizationId::from_digest([4; 32]).unwrap(),
            [5; 32],
            VerificationOutcomeV1::InfrastructureFailure,
            [8; 32],
        );
        let bytes = canonical_verification_claim_bytes(&claim).unwrap();
        let prefix = AUTH_BYTES_DOMAIN.len()
            + 8
            + CONTINUITY_VERIFICATION_CLAIM_AUTH_SCHEMA.len()
            + 5 * 32;

        assert_eq!(
            &bytes[prefix..prefix + 8],
            &claim.observed_at_unix_ms().to_le_bytes()
        );
        assert_eq!(bytes[prefix + 8], 4);
        assert_eq!(&bytes[bytes.len() - 32..], claim.id().as_bytes());
    }
}
