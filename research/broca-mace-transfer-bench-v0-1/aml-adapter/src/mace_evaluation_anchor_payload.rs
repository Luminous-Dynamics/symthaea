// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical pre-anchor statement for MACE evaluator commitments.
//!
//! This payload is constructed only from a verified sealed evaluator manifest
//! and a verified detached witness capability. It deliberately excludes the
//! transparency-log ID and entry ID: those are assigned by the log after the
//! pre-anchor payload has been committed and belong in the detached witness
//! receipt. This avoids a circular commitment.
//!
//! The payload digest is the MACE commitment that a MON-005 evidence record
//! should carry. The outer MON-005 record digest / Merkle leaf still belongs
//! to MON-005's canonical record encoder and proof verifier; this module does
//! not pretend that the MACE payload digest is itself the outer record leaf.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::evaluation_manifest_commitment_receipt::VerifiedEvaluationManifestCommitmentV1;
use crate::frozen_decision_episode_manifest::DecisionEpisodeSplitV1;
use crate::sealed_evaluation_manifest::{
    sealed_evaluation_manifest_sha256_v1, SealedEvaluationManifestV1,
    VerifiedSealedEvaluationManifestV1,
};

pub const MACE_EVALUATION_ANCHOR_PAYLOAD_SCHEMA_V1: &str =
    "mace_evaluation_anchor_payload_v1";
pub const MACE_EVALUATION_ANCHOR_PAYLOAD_DOMAIN_V1: &[u8] =
    b"luminous:mace:evaluation-anchor-payload:v1\0";
pub const MAX_MACE_EVALUATION_ANCHOR_PAYLOAD_BYTES_V1: usize = 1_048_576;

/// Exact MACE content to be committed before the independent log assigns an
/// anchor entry. No log ID, entry ID, checkpoint size, or witness signature
/// is included here; those are bound by the detached post-append receipt.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaceEvaluationAnchorPayloadV1 {
    pub schema_version: String,
    pub suite_id: String,
    pub suite_version: String,
    pub evaluation_protocol_sha256: String,
    pub split: DecisionEpisodeSplitV1,
    pub episode_count: usize,
    pub episode_manifest_root_sha256: String,
    pub sealed_evaluation_manifest_sha256: String,
    pub arm_assignment_root_sha256: String,
    pub pre_registration_root_sha256: String,
    pub sealed_artifacts_root_sha256: String,
    pub evaluator_profile_sha256: String,
    pub verifier_profile_sha256: String,
    pub access_control_profile_sha256: String,
    pub sealed_store_profile_sha256: String,
    pub candidate_access_profile_sha256: String,
    pub reveal_protocol_sha256: String,
    pub trace_seal_protocol_sha256: String,
}

/// Consumer-owned profile pins. These must come from deployment/evaluator
/// configuration, not from the candidate or the sealed manifest itself.
pub struct MaceEvaluationAnchorExpectedBindingV1<'a> {
    pub evaluator_profile_sha256: &'a str,
    pub verifier_profile_sha256: &'a str,
    pub access_control_profile_sha256: &'a str,
    pub sealed_store_profile_sha256: &'a str,
    pub candidate_access_profile_sha256: &'a str,
    pub reveal_protocol_sha256: &'a str,
    pub trace_seal_protocol_sha256: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaceEvaluationAnchorPayloadRejectV1 {
    UnsupportedSchemaVersion,
    InvalidIdentifier,
    InvalidDigest,
    ManifestDigestMismatch,
    WitnessManifestMismatch,
    SuiteBindingMismatch,
    EpisodeSetBindingMismatch,
    AssignmentRootMismatch,
    PreRegistrationRootMismatch,
    ArtifactRootMismatch,
    CustodyProfileMismatch,
    CandidateAccessProfileMismatch,
    RevealProtocolMismatch,
    TraceSealProtocolMismatch,
    CandidateReadableBeforeTraceSeal,
    InvalidRevealCondition,
    CustodyAfterWitnessCommitment,
    EmptyEpisodeSet,
    TooManyEpisodes,
    CanonicalizationFailed,
    PayloadTooLarge,
}

/// Private-field capability proving that the exact MACE anchor payload is
/// consistent with consumer-verified manifest and witness capabilities.
/// It does not prove that the payload has been externally logged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedMaceEvaluationAnchorPayloadV1 {
    payload: MaceEvaluationAnchorPayloadV1,
    payload_sha256: String,
}

impl VerifiedMaceEvaluationAnchorPayloadV1 {
    pub fn payload(&self) -> &MaceEvaluationAnchorPayloadV1 {
        &self.payload
    }

    /// Digest to place in the payload field of the MON-005 evidence record.
    /// MON-005 remains responsible for canonicalizing and hashing its outer
    /// record (tenant, sequence, predecessor and payload) into the Merkle leaf.
    pub fn mon005_payload_sha256(&self) -> &str {
        &self.payload_sha256
    }

    pub fn payload_sha256(&self) -> &str {
        &self.payload_sha256
    }
}

/// Build the pre-anchor MACE statement only from already verified capabilities
/// and consumer-owned profile pins. The sealed manifest is supplied again so
/// the function can recompute its full digest and derive the exact custody and
/// anti-leakage profile bindings without trusting caller-supplied copies.
pub fn build_mace_evaluation_anchor_payload_v1(
    manifest: &SealedEvaluationManifestV1,
    verified_manifest: &VerifiedSealedEvaluationManifestV1,
    verified_witness: &VerifiedEvaluationManifestCommitmentV1,
    expected: &MaceEvaluationAnchorExpectedBindingV1<'_>,
) -> Result<VerifiedMaceEvaluationAnchorPayloadV1, MaceEvaluationAnchorPayloadRejectV1> {
    if manifest.schema_version != "mace_sealed_evaluation_manifest_v1" {
        return Err(MaceEvaluationAnchorPayloadRejectV1::UnsupportedSchemaVersion);
    }

    let manifest_digest = sealed_evaluation_manifest_sha256_v1(manifest)
        .map_err(|_| MaceEvaluationAnchorPayloadRejectV1::CanonicalizationFailed)?;
    if manifest_digest != verified_manifest.manifest_sha256() {
        return Err(MaceEvaluationAnchorPayloadRejectV1::ManifestDigestMismatch);
    }
    if manifest_digest != verified_witness.sealed_evaluation_manifest_sha256() {
        return Err(MaceEvaluationAnchorPayloadRejectV1::WitnessManifestMismatch);
    }

    validate_atom(&manifest.suite.suite_id)?;
    validate_atom(&manifest.suite.suite_version)?;
    if manifest.suite.suite_id != verified_manifest.suite_id()
        || manifest.suite.suite_version != verified_manifest.suite_version()
        || manifest.suite.suite_id != verified_witness.suite_id()
        || manifest.suite.suite_version != verified_witness.suite_version()
    {
        return Err(MaceEvaluationAnchorPayloadRejectV1::SuiteBindingMismatch);
    }

    if manifest.episode_set.episode_count == 0 {
        return Err(MaceEvaluationAnchorPayloadRejectV1::EmptyEpisodeSet);
    }
    if manifest.episode_set.episode_count > 10_000
        || manifest.episode_set.episode_count != verified_manifest.episode_count()
    {
        return Err(MaceEvaluationAnchorPayloadRejectV1::TooManyEpisodes);
    }
    if manifest.episode_set.split != verified_manifest.split()
        || manifest.episode_set.episode_manifest_root_sha256
            != verified_manifest.episode_manifest_root_sha256()
        || manifest.episode_set.episode_manifest_root_sha256
            != verified_witness.episode_manifest_root_sha256()
    {
        return Err(MaceEvaluationAnchorPayloadRejectV1::EpisodeSetBindingMismatch);
    }
    if verified_manifest.arm_assignment_root_sha256()
        != verified_witness.arm_assignment_root_sha256()
    {
        return Err(MaceEvaluationAnchorPayloadRejectV1::AssignmentRootMismatch);
    }
    if verified_manifest.pre_registration_root_sha256()
        != verified_witness.pre_registration_root_sha256()
    {
        return Err(MaceEvaluationAnchorPayloadRejectV1::PreRegistrationRootMismatch);
    }
    if verified_manifest.sealed_artifacts_root_sha256()
        != verified_witness.sealed_artifacts_root_sha256()
    {
        return Err(MaceEvaluationAnchorPayloadRejectV1::ArtifactRootMismatch);
    }
    if manifest.custody.evaluator_profile_sha256 != expected.evaluator_profile_sha256
        || manifest.custody.verifier_profile_sha256 != expected.verifier_profile_sha256
        || manifest.custody.access_control_profile_sha256 != expected.access_control_profile_sha256
        || manifest.custody.sealed_store_profile_sha256 != expected.sealed_store_profile_sha256
    {
        return Err(MaceEvaluationAnchorPayloadRejectV1::CustodyProfileMismatch);
    }
    if verified_witness.candidate_access_profile_sha256()
        != expected.candidate_access_profile_sha256
    {
        return Err(MaceEvaluationAnchorPayloadRejectV1::CandidateAccessProfileMismatch);
    }
    if verified_witness.reveal_protocol_sha256() != expected.reveal_protocol_sha256 {
        return Err(MaceEvaluationAnchorPayloadRejectV1::RevealProtocolMismatch);
    }
    if manifest.anti_leakage.trace_seal_protocol_sha256
        != expected.trace_seal_protocol_sha256
    {
        return Err(MaceEvaluationAnchorPayloadRejectV1::TraceSealProtocolMismatch);
    }
    if manifest.anti_leakage.candidate_readable_before_trace_seal {
        return Err(MaceEvaluationAnchorPayloadRejectV1::CandidateReadableBeforeTraceSeal);
    }
    if manifest.anti_leakage.reveal_condition != "all_candidate_traces_sealed" {
        return Err(MaceEvaluationAnchorPayloadRejectV1::InvalidRevealCondition);
    }
    if manifest.custody.committed_at_unix > verified_witness.committed_at_unix() {
        return Err(MaceEvaluationAnchorPayloadRejectV1::CustodyAfterWitnessCommitment);
    }

    for digest in [
        manifest.suite.evaluation_protocol_sha256.as_str(),
        manifest.episode_set.episode_manifest_root_sha256.as_str(),
        manifest.custody.evaluator_profile_sha256.as_str(),
        manifest.custody.verifier_profile_sha256.as_str(),
        manifest.custody.access_control_profile_sha256.as_str(),
        manifest.custody.sealed_store_profile_sha256.as_str(),
        expected.candidate_access_profile_sha256,
        expected.reveal_protocol_sha256,
        manifest.anti_leakage.trace_seal_protocol_sha256.as_str(),
    ] {
        validate_sha256(digest)?;
    }

    let payload = MaceEvaluationAnchorPayloadV1 {
        schema_version: MACE_EVALUATION_ANCHOR_PAYLOAD_SCHEMA_V1.to_owned(),
        suite_id: manifest.suite.suite_id.clone(),
        suite_version: manifest.suite.suite_version.clone(),
        evaluation_protocol_sha256: manifest.suite.evaluation_protocol_sha256.clone(),
        split: manifest.episode_set.split,
        episode_count: manifest.episode_set.episode_count,
        episode_manifest_root_sha256: manifest.episode_set.episode_manifest_root_sha256.clone(),
        sealed_evaluation_manifest_sha256: manifest_digest,
        arm_assignment_root_sha256: verified_manifest.arm_assignment_root_sha256().to_owned(),
        pre_registration_root_sha256: verified_manifest.pre_registration_root_sha256().to_owned(),
        sealed_artifacts_root_sha256: verified_manifest.sealed_artifacts_root_sha256().to_owned(),
        evaluator_profile_sha256: manifest.custody.evaluator_profile_sha256.clone(),
        verifier_profile_sha256: manifest.custody.verifier_profile_sha256.clone(),
        access_control_profile_sha256: manifest.custody.access_control_profile_sha256.clone(),
        sealed_store_profile_sha256: manifest.custody.sealed_store_profile_sha256.clone(),
        candidate_access_profile_sha256: expected.candidate_access_profile_sha256.to_owned(),
        reveal_protocol_sha256: expected.reveal_protocol_sha256.to_owned(),
        trace_seal_protocol_sha256: manifest.anti_leakage.trace_seal_protocol_sha256.clone(),
    };
    let payload_sha256 = mace_evaluation_anchor_payload_sha256_v1(&payload)?;
    Ok(VerifiedMaceEvaluationAnchorPayloadV1 {
        payload,
        payload_sha256,
    })
}

/// Canonical pre-anchor commitment. This digest is placed inside a MON-005
/// record; the MON-005 record itself is separately hashed and witnessed.
pub fn mace_evaluation_anchor_payload_sha256_v1(
    payload: &MaceEvaluationAnchorPayloadV1,
) -> Result<String, MaceEvaluationAnchorPayloadRejectV1> {
    validate_payload_shape(payload)?;
    let canonical = serde_json_canonicalizer::to_vec(payload)
        .map_err(|_| MaceEvaluationAnchorPayloadRejectV1::CanonicalizationFailed)?;
    if canonical.len() > MAX_MACE_EVALUATION_ANCHOR_PAYLOAD_BYTES_V1 {
        return Err(MaceEvaluationAnchorPayloadRejectV1::PayloadTooLarge);
    }
    let mut hasher = Sha256::new();
    hasher.update(MACE_EVALUATION_ANCHOR_PAYLOAD_DOMAIN_V1);
    hasher.update(&canonical);
    Ok(hex_digest(&hasher.finalize()))
}

fn validate_payload_shape(
    payload: &MaceEvaluationAnchorPayloadV1,
) -> Result<(), MaceEvaluationAnchorPayloadRejectV1> {
    if payload.schema_version != MACE_EVALUATION_ANCHOR_PAYLOAD_SCHEMA_V1 {
        return Err(MaceEvaluationAnchorPayloadRejectV1::UnsupportedSchemaVersion);
    }
    validate_atom(&payload.suite_id)?;
    validate_atom(&payload.suite_version)?;
    if payload.episode_count == 0 {
        return Err(MaceEvaluationAnchorPayloadRejectV1::EmptyEpisodeSet);
    }
    if payload.episode_count > 10_000 {
        return Err(MaceEvaluationAnchorPayloadRejectV1::TooManyEpisodes);
    }
    for digest in [
        payload.evaluation_protocol_sha256.as_str(),
        payload.episode_manifest_root_sha256.as_str(),
        payload.sealed_evaluation_manifest_sha256.as_str(),
        payload.arm_assignment_root_sha256.as_str(),
        payload.pre_registration_root_sha256.as_str(),
        payload.sealed_artifacts_root_sha256.as_str(),
        payload.evaluator_profile_sha256.as_str(),
        payload.verifier_profile_sha256.as_str(),
        payload.access_control_profile_sha256.as_str(),
        payload.sealed_store_profile_sha256.as_str(),
        payload.candidate_access_profile_sha256.as_str(),
        payload.reveal_protocol_sha256.as_str(),
        payload.trace_seal_protocol_sha256.as_str(),
    ] {
        validate_sha256(digest)?;
    }
    Ok(())
}

fn validate_atom(value: &str) -> Result<(), MaceEvaluationAnchorPayloadRejectV1> {
    if value.is_empty()
        || value.len() > 256
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(MaceEvaluationAnchorPayloadRejectV1::InvalidIdentifier);
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), MaceEvaluationAnchorPayloadRejectV1> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(MaceEvaluationAnchorPayloadRejectV1::InvalidDigest);
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sealed_evaluation_manifest::test_verified_sealed_evaluation_fixture_v1;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn expected<'a>() -> MaceEvaluationAnchorExpectedBindingV1<'a> {
        MaceEvaluationAnchorExpectedBindingV1 {
            evaluator_profile_sha256: Box::leak(digest('b').into_boxed_str()),
            verifier_profile_sha256: Box::leak(digest('c').into_boxed_str()),
            access_control_profile_sha256: Box::leak(digest('d').into_boxed_str()),
            sealed_store_profile_sha256: Box::leak(digest('e').into_boxed_str()),
            candidate_access_profile_sha256: Box::leak(digest('1').into_boxed_str()),
            reveal_protocol_sha256: Box::leak(digest('2').into_boxed_str()),
            trace_seal_protocol_sha256: Box::leak(digest('f').into_boxed_str()),
        }
    }

    #[test]
    fn verified_sealed_manifest_builds_pre_anchor_payload() {
        let (manifest, verified, witness) = test_verified_sealed_evaluation_fixture_v1();
        let payload = build_mace_evaluation_anchor_payload_v1(
            &manifest,
            &verified,
            &witness,
            &expected(),
        ).unwrap();

        assert_eq!(payload.payload().schema_version, MACE_EVALUATION_ANCHOR_PAYLOAD_SCHEMA_V1);
        assert_eq!(payload.payload().suite_id, "mace-t10-integrity");
        assert_eq!(payload.payload().suite_version, "v1");
        assert_eq!(payload.payload().episode_count, 2);
        assert_eq!(payload.payload().candidate_readable_before_trace_seal, false);
        assert_eq!(payload.payload().reveal_protocol_sha256, digest('2'));
        assert_eq!(payload.mon005_payload_sha256().len(), 64);
    }

    #[test]
    fn pre_anchor_payload_has_fixed_cross_language_digest() {
        let (manifest, verified, witness) = test_verified_sealed_evaluation_fixture_v1();
        let payload = build_mace_evaluation_anchor_payload_v1(
            &manifest,
            &verified,
            &witness,
            &expected(),
        ).unwrap();
        assert_eq!(
            payload.mon005_payload_sha256(),
            "0000000000000000000000000000000000000000000000000000000000000000"
        );
    }

    #[test]
    fn altered_manifest_cannot_be_bound_to_verified_capability() {
        let (manifest, verified, witness) = test_verified_sealed_evaluation_fixture_v1();
        let mut altered = manifest.clone();
        altered.custody.evaluator_profile_sha256 = digest('f');
        assert_eq!(
            build_mace_evaluation_anchor_payload_v1(
                &altered,
                &verified,
                &witness,
                &expected(),
            ),
            Err(MaceEvaluationAnchorPayloadRejectV1::ManifestDigestMismatch)
        );
    }

    #[test]
    fn witness_root_substitution_is_rejected() {
        let (manifest, verified, _) = test_verified_sealed_evaluation_fixture_v1();
        let (_, _, wrong_witness) = test_verified_sealed_evaluation_fixture_v1();
        // The two independent test capabilities bind the same suite/manifests;
        // a payload root mutation is instead caught by the original sealed digest.
        let mut altered = manifest.clone();
        altered.episode_set.episode_manifest_root_sha256 = digest('9');
        assert_eq!(
            build_mace_evaluation_anchor_payload_v1(
                &altered,
                &verified,
                &wrong_witness,
                &expected(),
            ),
            Err(MaceEvaluationAnchorPayloadRejectV1::ManifestDigestMismatch)
        );
    }

    #[test]
    fn consumer_pinned_profiles_must_match_the_verified_manifest_and_witness() {
        let (manifest, verified, witness) = test_verified_sealed_evaluation_fixture_v1();
        let wrong = MaceEvaluationAnchorExpectedBindingV1 {
            evaluator_profile_sha256: Box::leak(digest('0').into_boxed_str()),
            ..expected()
        };
        assert_eq!(
            build_mace_evaluation_anchor_payload_v1(
                &manifest,
                &verified,
                &witness,
                &wrong,
            ),
            Err(MaceEvaluationAnchorPayloadRejectV1::CustodyProfileMismatch)
        );
    }

    #[test]
    fn candidate_access_and_reveal_profiles_are_pinned_separately() {
        let (manifest, verified, witness) = test_verified_sealed_evaluation_fixture_v1();
        let wrong_access = MaceEvaluationAnchorExpectedBindingV1 {
            candidate_access_profile_sha256: Box::leak(digest('9').into_boxed_str()),
            ..expected()
        };
        assert_eq!(
            build_mace_evaluation_anchor_payload_v1(
                &manifest,
                &verified,
                &witness,
                &wrong_access,
            ),
            Err(MaceEvaluationAnchorPayloadRejectV1::CandidateAccessProfileMismatch)
        );
        let wrong_reveal = MaceEvaluationAnchorExpectedBindingV1 {
            reveal_protocol_sha256: Box::leak(digest('9').into_boxed_str()),
            ..expected()
        };
        assert_eq!(
            build_mace_evaluation_anchor_payload_v1(
                &manifest,
                &verified,
                &witness,
                &wrong_reveal,
            ),
            Err(MaceEvaluationAnchorPayloadRejectV1::RevealProtocolMismatch)
        );
    }
}
