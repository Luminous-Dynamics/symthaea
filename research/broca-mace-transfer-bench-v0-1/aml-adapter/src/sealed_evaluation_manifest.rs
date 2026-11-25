// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consumer-side verification for the sealed MACE evaluator manifest.
//!
//! This module closes the gap between a witnessed public episode-set root and
//! the evaluator-only assignments/artifacts committed beside it. It does not
//! decrypt artifacts, provide isolated custody, or implement a transparency
//! log. Those remain independently administered deployment boundaries.

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::evaluation_manifest_commitment_receipt::VerifiedEvaluationManifestCommitmentV1;
use crate::frozen_decision_episode_manifest::{
    DecisionEpisodeSplitV1, FrozenDecisionEpisodeManifestV1,
    verify_frozen_decision_episode_set_v1,
};

pub const SEALED_EVALUATION_MANIFEST_SCHEMA_V1: &str = "mace_sealed_evaluation_manifest_v1";
pub const MAX_SEALED_EVALUATION_MANIFEST_BYTES_V1: usize = 32 * 1024 * 1024;
pub const MAX_SEALED_EVALUATION_ASSIGNMENTS_V1: usize = 10_000;
pub const MAX_SEALED_EVALUATION_ARTIFACTS_V1: usize = 100_000;
pub const MAX_SEALED_EVALUATION_IDENTIFIER_BYTES_V1: usize = 256;
pub const MAX_SEALED_EVALUATION_UNIX_SECONDS_V1: u64 = 253_402_300_799;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedEvaluationSuiteV1 {
    pub suite_id: String,
    pub suite_version: String,
    pub evaluation_protocol_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedEvaluationEpisodeSetV1 {
    pub episode_manifest_root_sha256: String,
    pub episode_count: usize,
    pub split: DecisionEpisodeSplitV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedEpisodeAssignmentV1 {
    pub episode_id: String,
    pub arm_id: String,
    pub seed_commitment_sha256: String,
    pub treatment_configuration_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedEvaluationPreRegistrationV1 {
    pub analysis_plan_sha256: String,
    pub utility_scale_sha256: String,
    pub hard_constraint_oracle_sha256: String,
    pub metric_definitions_sha256: String,
    pub exclusion_policy_sha256: String,
    pub stopping_rule_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SealedEvaluationArtifactRoleV1 {
    HiddenGroundTruth,
    AttackLabel,
    ExpectedDecision,
    OutcomeOracle,
    RepairOracle,
    ConstraintOracle,
    MetricParameters,
    AdjudicationEvidence,
    ExclusionDetails,
}

impl SealedEvaluationArtifactRoleV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::HiddenGroundTruth => "hidden_ground_truth",
            Self::AttackLabel => "attack_label",
            Self::ExpectedDecision => "expected_decision",
            Self::OutcomeOracle => "outcome_oracle",
            Self::RepairOracle => "repair_oracle",
            Self::ConstraintOracle => "constraint_oracle",
            Self::MetricParameters => "metric_parameters",
            Self::AdjudicationEvidence => "adjudication_evidence",
            Self::ExclusionDetails => "exclusion_details",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedEvaluationArtifactV1 {
    pub artifact_id: String,
    pub role: SealedEvaluationArtifactRoleV1,
    pub content_sha256: String,
    pub sealed_object_commitment_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedEvaluationCustodyV1 {
    pub evaluator_profile_sha256: String,
    pub verifier_profile_sha256: String,
    pub access_control_profile_sha256: String,
    pub sealed_store_profile_sha256: String,
    pub committed_at_unix: u64,
    pub key_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedEvaluationAntiLeakageV1 {
    pub candidate_readable_before_trace_seal: bool,
    pub trace_seal_protocol_sha256: String,
    pub reveal_condition: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedEvaluationManifestV1 {
    pub schema_version: String,
    pub suite: SealedEvaluationSuiteV1,
    pub episode_set: SealedEvaluationEpisodeSetV1,
    pub assignments: Vec<SealedEpisodeAssignmentV1>,
    pub pre_registration: SealedEvaluationPreRegistrationV1,
    pub sealed_artifacts: Vec<SealedEvaluationArtifactV1>,
    pub custody: SealedEvaluationCustodyV1,
    pub anti_leakage: SealedEvaluationAntiLeakageV1,
}

/// The custody profile and the candidate-access/reveal bindings are supplied by
/// the consumer, never inferred from the sealed manifest itself.
pub struct SealedEvaluationManifestExpectedBindingV1<'a> {
    pub evaluator_profile_sha256: &'a str,
    pub verifier_profile_sha256: &'a str,
    pub access_control_profile_sha256: &'a str,
    pub sealed_store_profile_sha256: &'a str,
    pub candidate_access_profile_sha256: &'a str,
    pub reveal_protocol_sha256: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SealedEvaluationManifestRejectV1 {
    ManifestTooLarge,
    InvalidJson,
    UnsupportedSchemaVersion,
    InvalidIdentifier,
    InvalidDigest,
    InvalidTimestamp,
    EmptyArtifactSet,
    TooManyAssignments,
    TooManyArtifacts,
    EpisodeSetInvalid,
    EpisodeSetRootMismatch,
    EpisodeCountMismatch,
    SuiteBindingMismatch,
    SplitMismatch,
    AssignmentCountMismatch,
    AssignmentEpisodeCoverageMismatch,
    NonCanonicalAssignmentOrder,
    DuplicateAssignmentEpisodeId,
    InvalidArmId,
    AssignmentSeedCommitmentMismatch,
    AssignmentCommitmentMismatch,
    ArtifactOrderMismatch,
    DuplicateArtifactId,
    PreRegistrationRootMismatch,
    ArtifactRootMismatch,
    SealedManifestDigestMismatch,
    CustodyProfileMismatch,
    WitnessProfileBindingMismatch,
    CandidateReadableBeforeTraceSeal,
    InvalidRevealCondition,
    CustodyBeforeEpisodeFreeze,
    CustodyAfterWitnessCommitment,
    CanonicalizationFailed,
}

/// A verified capability is emitted only after the full assignment set,
/// public per-episode commitments, preregistration, artifacts, custody
/// profiles, anti-leakage settings and detached witness are cross-checked.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedSealedEvaluationManifestV1 {
    manifest_sha256: String,
    episode_manifest_root_sha256: String,
    arm_assignment_root_sha256: String,
    pre_registration_root_sha256: String,
    sealed_artifacts_root_sha256: String,
    suite_id: String,
    suite_version: String,
    split: DecisionEpisodeSplitV1,
    episode_count: usize,
}

impl VerifiedSealedEvaluationManifestV1 {
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn episode_manifest_root_sha256(&self) -> &str {
        &self.episode_manifest_root_sha256
    }

    pub fn arm_assignment_root_sha256(&self) -> &str {
        &self.arm_assignment_root_sha256
    }

    pub fn pre_registration_root_sha256(&self) -> &str {
        &self.pre_registration_root_sha256
    }

    pub fn sealed_artifacts_root_sha256(&self) -> &str {
        &self.sealed_artifacts_root_sha256
    }

    pub fn suite_id(&self) -> &str {
        &self.suite_id
    }

    pub fn suite_version(&self) -> &str {
        &self.suite_version
    }

    pub fn split(&self) -> DecisionEpisodeSplitV1 {
        self.split
    }

    pub fn episode_count(&self) -> usize {
        self.episode_count
    }
}

pub fn parse_and_verify_sealed_evaluation_manifest_v1(
    wire: &[u8],
    public_episodes: &[FrozenDecisionEpisodeManifestV1],
    witness: &VerifiedEvaluationManifestCommitmentV1,
    expected: &SealedEvaluationManifestExpectedBindingV1<'_>,
) -> Result<VerifiedSealedEvaluationManifestV1, SealedEvaluationManifestRejectV1> {
    if wire.len() > MAX_SEALED_EVALUATION_MANIFEST_BYTES_V1 {
        return Err(SealedEvaluationManifestRejectV1::ManifestTooLarge);
    }
    let manifest: SealedEvaluationManifestV1 =
        serde_json::from_slice(wire).map_err(|_| SealedEvaluationManifestRejectV1::InvalidJson)?;
    validate_sealed_evaluation_manifest_v1(&manifest, public_episodes, witness, expected)
}

pub fn validate_sealed_evaluation_manifest_v1(
    manifest: &SealedEvaluationManifestV1,
    public_episodes: &[FrozenDecisionEpisodeManifestV1],
    witness: &VerifiedEvaluationManifestCommitmentV1,
    expected: &SealedEvaluationManifestExpectedBindingV1<'_>,
) -> Result<VerifiedSealedEvaluationManifestV1, SealedEvaluationManifestRejectV1> {
    validate_manifest_shape(manifest)?;
    validate_expected_bindings(expected)?;

    let episode_set = verify_frozen_decision_episode_set_v1(public_episodes)
        .map_err(|_| SealedEvaluationManifestRejectV1::EpisodeSetInvalid)?;
    episode_set
        .verify_against_witness(witness)
        .map_err(|_| SealedEvaluationManifestRejectV1::EpisodeSetRootMismatch)?;

    if manifest.suite.suite_id != witness.suite_id()
        || manifest.suite.suite_version != witness.suite_version()
        || manifest.suite.suite_id != episode_set.suite_id()
        || manifest.suite.suite_version != episode_set.suite_version()
    {
        return Err(SealedEvaluationManifestRejectV1::SuiteBindingMismatch);
    }
    if manifest.episode_set.episode_manifest_root_sha256 != episode_set.root_sha256()
        || manifest.episode_set.episode_manifest_root_sha256
            != witness.episode_manifest_root_sha256()
    {
        return Err(SealedEvaluationManifestRejectV1::EpisodeSetRootMismatch);
    }
    if manifest.episode_set.episode_count != public_episodes.len()
        || manifest.episode_set.episode_count != episode_set.len()
    {
        return Err(SealedEvaluationManifestRejectV1::EpisodeCountMismatch);
    }
    if manifest.episode_set.split != episode_set.split() {
        return Err(SealedEvaluationManifestRejectV1::SplitMismatch);
    }

    let assignment_root = arm_assignment_set_root_sha256_v1(
        &manifest.assignments,
        public_episodes,
    )?;
    let pre_registration_root =
        pre_registration_root_sha256_v1(&manifest.pre_registration)?;
    let artifacts_root = sealed_artifacts_root_sha256_v1(&manifest.sealed_artifacts)?;
    let manifest_digest = sealed_evaluation_manifest_sha256_v1(manifest)?;

    if assignment_root != witness.arm_assignment_root_sha256() {
        return Err(SealedEvaluationManifestRejectV1::AssignmentCommitmentMismatch);
    }
    if pre_registration_root != witness.pre_registration_root_sha256() {
        return Err(SealedEvaluationManifestRejectV1::PreRegistrationRootMismatch);
    }
    if artifacts_root != witness.sealed_artifacts_root_sha256() {
        return Err(SealedEvaluationManifestRejectV1::ArtifactRootMismatch);
    }
    if manifest_digest != witness.sealed_evaluation_manifest_sha256() {
        return Err(SealedEvaluationManifestRejectV1::SealedManifestDigestMismatch);
    }

    if manifest.custody.evaluator_profile_sha256 != expected.evaluator_profile_sha256
        || manifest.custody.verifier_profile_sha256 != expected.verifier_profile_sha256
        || manifest.custody.access_control_profile_sha256 != expected.access_control_profile_sha256
        || manifest.custody.sealed_store_profile_sha256 != expected.sealed_store_profile_sha256
    {
        return Err(SealedEvaluationManifestRejectV1::CustodyProfileMismatch);
    }
    if witness.candidate_access_profile_sha256() != expected.candidate_access_profile_sha256
        || witness.reveal_protocol_sha256() != expected.reveal_protocol_sha256
    {
        return Err(SealedEvaluationManifestRejectV1::WitnessProfileBindingMismatch);
    }
    if manifest.anti_leakage.candidate_readable_before_trace_seal {
        return Err(SealedEvaluationManifestRejectV1::CandidateReadableBeforeTraceSeal);
    }
    if manifest.anti_leakage.reveal_condition != "all_candidate_traces_sealed" {
        return Err(SealedEvaluationManifestRejectV1::InvalidRevealCondition);
    }

    let latest_freeze = public_episodes
        .iter()
        .map(|episode| episode.frozen_at_unix)
        .max()
        .ok_or(SealedEvaluationManifestRejectV1::EpisodeSetInvalid)?;
    if manifest.custody.committed_at_unix < latest_freeze {
        return Err(SealedEvaluationManifestRejectV1::CustodyBeforeEpisodeFreeze);
    }
    if manifest.custody.committed_at_unix > witness.committed_at_unix() {
        return Err(SealedEvaluationManifestRejectV1::CustodyAfterWitnessCommitment);
    }

    Ok(VerifiedSealedEvaluationManifestV1 {
        manifest_sha256: manifest_digest,
        episode_manifest_root_sha256: episode_set.root_sha256().to_owned(),
        arm_assignment_root_sha256: assignment_root,
        pre_registration_root_sha256: pre_registration_root,
        sealed_artifacts_root_sha256: artifacts_root,
        suite_id: manifest.suite.suite_id.clone(),
        suite_version: manifest.suite.suite_version.clone(),
        split: manifest.episode_set.split,
        episode_count: manifest.episode_set.episode_count,
    })
}

pub fn arm_assignment_commitment_sha256_v1(
    assignment: &SealedEpisodeAssignmentV1,
) -> Result<String, SealedEvaluationManifestRejectV1> {
    validate_atom(&assignment.episode_id)?;
    validate_arm_id(&assignment.arm_id)?;
    validate_hash(&assignment.seed_commitment_sha256)?;
    validate_hash(&assignment.treatment_configuration_sha256)?;
    let preimage = json!({
        "episode_id": assignment.episode_id,
        "arm_id": assignment.arm_id,
        "seed_commitment_sha256": assignment.seed_commitment_sha256,
        "treatment_configuration_sha256": assignment.treatment_configuration_sha256
    });
    hash_jcs(
        b"luminous:mace:arm-assignment:v1\0",
        &preimage,
    )
}

pub fn arm_assignment_set_root_sha256_v1(
    assignments: &[SealedEpisodeAssignmentV1],
    public_episodes: &[FrozenDecisionEpisodeManifestV1],
) -> Result<String, SealedEvaluationManifestRejectV1> {
    if assignments.len() > MAX_SEALED_EVALUATION_ASSIGNMENTS_V1 {
        return Err(SealedEvaluationManifestRejectV1::TooManyAssignments);
    }
    if assignments.len() != public_episodes.len() {
        return Err(SealedEvaluationManifestRejectV1::AssignmentCountMismatch);
    }

    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:arm-assignment-set:v1\0");
    hasher.update((assignments.len() as u64).to_be_bytes());

    let mut previous_id: Option<&str> = None;
    for (assignment, episode) in assignments.iter().zip(public_episodes) {
        validate_atom(&assignment.episode_id)?;
        if let Some(previous) = previous_id {
            match previous.as_bytes().cmp(assignment.episode_id.as_bytes()) {
                std::cmp::Ordering::Equal => {
                    return Err(SealedEvaluationManifestRejectV1::DuplicateAssignmentEpisodeId);
                }
                std::cmp::Ordering::Greater => {
                    return Err(SealedEvaluationManifestRejectV1::NonCanonicalAssignmentOrder);
                }
                std::cmp::Ordering::Less => {}
            }
        }
        previous_id = Some(&assignment.episode_id);

        if assignment.episode_id != episode.episode_id {
            return Err(SealedEvaluationManifestRejectV1::AssignmentEpisodeCoverageMismatch);
        }
        if assignment.seed_commitment_sha256
            != episode.randomization.episode_seed_commitment_sha256
        {
            return Err(SealedEvaluationManifestRejectV1::AssignmentSeedCommitmentMismatch);
        }
        let commitment = arm_assignment_commitment_sha256_v1(assignment)?;
        if commitment != episode.randomization.arm_assignment_commitment_sha256 {
            return Err(SealedEvaluationManifestRejectV1::AssignmentCommitmentMismatch);
        }

        hasher.update((assignment.episode_id.len() as u64).to_be_bytes());
        hasher.update(assignment.episode_id.as_bytes());
        hasher.update(decode_hash(&commitment)?);
    }

    Ok(hex_digest(&hasher.finalize()))
}

pub fn pre_registration_root_sha256_v1(
    pre_registration: &SealedEvaluationPreRegistrationV1,
) -> Result<String, SealedEvaluationManifestRejectV1> {
    for digest in [
        pre_registration.analysis_plan_sha256.as_str(),
        pre_registration.utility_scale_sha256.as_str(),
        pre_registration.hard_constraint_oracle_sha256.as_str(),
        pre_registration.metric_definitions_sha256.as_str(),
        pre_registration.exclusion_policy_sha256.as_str(),
        pre_registration.stopping_rule_sha256.as_str(),
    ] {
        validate_hash(digest)?;
    }
    hash_jcs(
        b"luminous:mace:pre-registration:v1\0",
        pre_registration,
    )
}

pub fn sealed_artifacts_root_sha256_v1(
    artifacts: &[SealedEvaluationArtifactV1],
) -> Result<String, SealedEvaluationManifestRejectV1> {
    if artifacts.is_empty() {
        return Err(SealedEvaluationManifestRejectV1::EmptyArtifactSet);
    }
    if artifacts.len() > MAX_SEALED_EVALUATION_ARTIFACTS_V1 {
        return Err(SealedEvaluationManifestRejectV1::TooManyArtifacts);
    }

    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:sealed-artifact-set:v1\0");
    hasher.update((artifacts.len() as u64).to_be_bytes());
    let mut previous_id: Option<&str> = None;

    for artifact in artifacts {
        validate_atom(&artifact.artifact_id)?;
        validate_hash(&artifact.content_sha256)?;
        validate_hash(&artifact.sealed_object_commitment_sha256)?;
        if let Some(previous) = previous_id {
            match previous.as_bytes().cmp(artifact.artifact_id.as_bytes()) {
                std::cmp::Ordering::Equal => {
                    return Err(SealedEvaluationManifestRejectV1::DuplicateArtifactId);
                }
                std::cmp::Ordering::Greater => {
                    return Err(SealedEvaluationManifestRejectV1::ArtifactOrderMismatch);
                }
                std::cmp::Ordering::Less => {}
            }
        }
        previous_id = Some(&artifact.artifact_id);

        append_field(&mut hasher, &artifact.artifact_id);
        append_field(&mut hasher, artifact.role.as_str());
        hasher.update(decode_hash(&artifact.content_sha256)?);
        hasher.update(decode_hash(&artifact.sealed_object_commitment_sha256)?);
    }

    Ok(hex_digest(&hasher.finalize()))
}

pub fn sealed_evaluation_manifest_sha256_v1(
    manifest: &SealedEvaluationManifestV1,
) -> Result<String, SealedEvaluationManifestRejectV1> {
    hash_jcs(
        b"luminous:mace:sealed-evaluation-manifest:v1\0",
        manifest,
    )
}

fn validate_manifest_shape(
    manifest: &SealedEvaluationManifestV1,
) -> Result<(), SealedEvaluationManifestRejectV1> {
    if manifest.schema_version != SEALED_EVALUATION_MANIFEST_SCHEMA_V1 {
        return Err(SealedEvaluationManifestRejectV1::UnsupportedSchemaVersion);
    }
    if manifest.assignments.len() > MAX_SEALED_EVALUATION_ASSIGNMENTS_V1 {
        return Err(SealedEvaluationManifestRejectV1::TooManyAssignments);
    }
    if manifest.sealed_artifacts.is_empty() {
        return Err(SealedEvaluationManifestRejectV1::EmptyArtifactSet);
    }
    if manifest.sealed_artifacts.len() > MAX_SEALED_EVALUATION_ARTIFACTS_V1 {
        return Err(SealedEvaluationManifestRejectV1::TooManyArtifacts);
    }

    for atom in [
        manifest.suite.suite_id.as_str(),
        manifest.suite.suite_version.as_str(),
        manifest.custody.key_id.as_str(),
    ] {
        validate_atom(atom)?;
    }
    for digest in [
        manifest.suite.evaluation_protocol_sha256.as_str(),
        manifest.episode_set.episode_manifest_root_sha256.as_str(),
        manifest.custody.evaluator_profile_sha256.as_str(),
        manifest.custody.verifier_profile_sha256.as_str(),
        manifest.custody.access_control_profile_sha256.as_str(),
        manifest.custody.sealed_store_profile_sha256.as_str(),
        manifest.anti_leakage.trace_seal_protocol_sha256.as_str(),
    ] {
        validate_hash(digest)?;
    }
    if manifest.episode_set.episode_count == 0
        || manifest.episode_set.episode_count > MAX_SEALED_EVALUATION_ASSIGNMENTS_V1
    {
        return Err(SealedEvaluationManifestRejectV1::EpisodeCountMismatch);
    }
    if manifest.custody.committed_at_unix > MAX_SEALED_EVALUATION_UNIX_SECONDS_V1 {
        return Err(SealedEvaluationManifestRejectV1::InvalidTimestamp);
    }
    Ok(())
}

fn validate_expected_bindings(
    expected: &SealedEvaluationManifestExpectedBindingV1<'_>,
) -> Result<(), SealedEvaluationManifestRejectV1> {
    for digest in [
        expected.evaluator_profile_sha256,
        expected.verifier_profile_sha256,
        expected.access_control_profile_sha256,
        expected.sealed_store_profile_sha256,
        expected.candidate_access_profile_sha256,
        expected.reveal_protocol_sha256,
    ] {
        validate_hash(digest)?;
    }
    Ok(())
}

fn validate_atom(value: &str) -> Result<(), SealedEvaluationManifestRejectV1> {
    if value.is_empty()
        || value.len() > MAX_SEALED_EVALUATION_IDENTIFIER_BYTES_V1
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(SealedEvaluationManifestRejectV1::InvalidIdentifier);
    }
    Ok(())
}

fn validate_arm_id(value: &str) -> Result<(), SealedEvaluationManifestRejectV1> {
    if !matches!(value, "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K") {
        return Err(SealedEvaluationManifestRejectV1::InvalidArmId);
    }
    Ok(())
}

fn validate_hash(value: &str) -> Result<(), SealedEvaluationManifestRejectV1> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SealedEvaluationManifestRejectV1::InvalidDigest);
    }
    Ok(())
}

fn decode_hash(value: &str) -> Result<[u8; 32], SealedEvaluationManifestRejectV1> {
    validate_hash(value)?;
    let mut output = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        output[index] = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Result<u8, SealedEvaluationManifestRejectV1> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(SealedEvaluationManifestRejectV1::InvalidDigest),
    }
}

fn append_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hash_jcs<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, SealedEvaluationManifestRejectV1> {
    let canonical = serde_json_canonicalizer::to_vec(value)
        .map_err(|_| SealedEvaluationManifestRejectV1::CanonicalizationFailed)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&canonical);
    Ok(hex_digest(&hasher.finalize()))
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
    use crate::evaluation_manifest_commitment_receipt::
        test_verified_evaluation_manifest_commitment_v1_with_bindings;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn episode(id: &str, index: u64, seed: &str) -> FrozenDecisionEpisodeManifestV1 {
        FrozenDecisionEpisodeManifestV1 {
            schema_version: "mace_decision_episode_manifest_v1".to_owned(),
            episode_id: id.to_owned(),
            suite: crate::frozen_decision_episode_manifest::DecisionEpisodeSuiteV1 {
                suite_id: "mace-t10-integrity".to_owned(),
                suite_version: "v1".to_owned(),
                task_family: "agent-memory-integrity".to_owned(),
                split: DecisionEpisodeSplitV1::HeldOut,
                episode_index: index,
            },
            task: crate::frozen_decision_episode_manifest::DecisionEpisodeTaskV1 {
                task_input_sha256: digest('a'),
                source_snapshot_sha256: digest('b'),
                provenance_graph_sha256: digest('c'),
            },
            decision: crate::frozen_decision_episode_manifest::DecisionEpisodeDecisionV1 {
                decision_invocation_id: format!("decision-{id}"),
                decision_need_sha256: digest('d'),
                purpose_sha256: digest('e'),
                subject_scope_sha256: digest('f'),
                option_set_sha256: digest('1'),
                constraint_set_sha256: digest('2'),
                decision_policy_sha256: digest('3'),
            },
            runtime: crate::frozen_decision_episode_manifest::DecisionEpisodeRuntimeV1 {
                environment_revision_sha256: digest('4'),
                reader_profile_sha256: digest('5'),
                tool_profile_sha256: digest('6'),
                retrieval_policy_profile_sha256: digest('7'),
                runner_profile_sha256: digest('8'),
            },
            resources: crate::frozen_decision_episode_manifest::DecisionEpisodeResourcesV1 {
                budget_profile_sha256: digest('9'),
            },
            randomization: crate::frozen_decision_episode_manifest::DecisionEpisodeRandomizationV1 {
                episode_seed_commitment_sha256: seed.to_owned(),
                arm_assignment_commitment_sha256: digest('0'),
            },
            frozen_at_unix: 100 + index,
        }
    }

    struct Fixture {
        manifest: SealedEvaluationManifestV1,
        episodes: Vec<FrozenDecisionEpisodeManifestV1>,
        expected: SealedEvaluationManifestExpectedBindingV1<'static>,
    }

    impl Fixture {
        fn new() -> Self {
            let mut episodes = vec![episode("episode-a", 1, &digest('a')), episode("episode-b", 2, &digest('b'))];
            let assignments = vec![
                SealedEpisodeAssignmentV1 {
                    episode_id: "episode-a".to_owned(),
                    arm_id: "A".to_owned(),
                    seed_commitment_sha256: digest('a'),
                    treatment_configuration_sha256: digest('c'),
                },
                SealedEpisodeAssignmentV1 {
                    episode_id: "episode-b".to_owned(),
                    arm_id: "B".to_owned(),
                    seed_commitment_sha256: digest('b'),
                    treatment_configuration_sha256: digest('d'),
                },
            ];
            for (episode, assignment) in episodes.iter_mut().zip(&assignments) {
                episode.randomization.arm_assignment_commitment_sha256 =
                    arm_assignment_commitment_sha256_v1(assignment).unwrap();
            }

            let mut manifest = SealedEvaluationManifestV1 {
                schema_version: SEALED_EVALUATION_MANIFEST_SCHEMA_V1.to_owned(),
                suite: SealedEvaluationSuiteV1 {
                    suite_id: "mace-t10-integrity".to_owned(),
                    suite_version: "v1".to_owned(),
                    evaluation_protocol_sha256: digest('e'),
                },
                episode_set: SealedEvaluationEpisodeSetV1 {
                    episode_manifest_root_sha256: String::new(),
                    episode_count: 2,
                    split: DecisionEpisodeSplitV1::HeldOut,
                },
                assignments,
                pre_registration: SealedEvaluationPreRegistrationV1 {
                    analysis_plan_sha256: digest('1'),
                    utility_scale_sha256: digest('2'),
                    hard_constraint_oracle_sha256: digest('3'),
                    metric_definitions_sha256: digest('4'),
                    exclusion_policy_sha256: digest('5'),
                    stopping_rule_sha256: digest('6'),
                },
                sealed_artifacts: vec![
                    SealedEvaluationArtifactV1 {
                        artifact_id: "attack-label-T10-erasure-resurrection".to_owned(),
                        role: SealedEvaluationArtifactRoleV1::AttackLabel,
                        content_sha256: digest('7'),
                        sealed_object_commitment_sha256: digest('8'),
                    },
                    SealedEvaluationArtifactV1 {
                        artifact_id: "attack-label-T9-whole-store-rollback".to_owned(),
                        role: SealedEvaluationArtifactRoleV1::AttackLabel,
                        content_sha256: digest('9'),
                        sealed_object_commitment_sha256: digest('a'),
                    },
                ],
                custody: SealedEvaluationCustodyV1 {
                    evaluator_profile_sha256: digest('b'),
                    verifier_profile_sha256: digest('c'),
                    access_control_profile_sha256: digest('d'),
                    sealed_store_profile_sha256: digest('e'),
                    committed_at_unix: 90,
                    key_id: "mace-evaluator-key-1".to_owned(),
                },
                anti_leakage: SealedEvaluationAntiLeakageV1 {
                    candidate_readable_before_trace_seal: false,
                    trace_seal_protocol_sha256: digest('f'),
                    reveal_condition: "all_candidate_traces_sealed".to_owned(),
                },
            };

            let episode_set = verify_frozen_decision_episode_set_v1(&episodes).unwrap();
            manifest.episode_set.episode_manifest_root_sha256 = episode_set.root_sha256().to_owned();

            let expected = SealedEvaluationManifestExpectedBindingV1 {
                evaluator_profile_sha256: Box::leak(digest('b').into_boxed_str()),
                verifier_profile_sha256: Box::leak(digest('c').into_boxed_str()),
                access_control_profile_sha256: Box::leak(digest('d').into_boxed_str()),
                sealed_store_profile_sha256: Box::leak(digest('e').into_boxed_str()),
                candidate_access_profile_sha256: Box::leak(digest('1').into_boxed_str()),
                reveal_protocol_sha256: Box::leak(digest('2').into_boxed_str()),
            };

            Self { manifest, episodes, expected }
        }

        fn witness(&self) -> VerifiedEvaluationManifestCommitmentV1 {
            let set = verify_frozen_decision_episode_set_v1(&self.episodes).unwrap();
            let assignment_root = arm_assignment_set_root_sha256_v1(
                &self.manifest.assignments, &self.episodes,
            ).unwrap();
            let pre_registration_root =
                pre_registration_root_sha256_v1(&self.manifest.pre_registration).unwrap();
            let artifacts_root =
                sealed_artifacts_root_sha256_v1(&self.manifest.sealed_artifacts).unwrap();
            let sealed_digest = sealed_evaluation_manifest_sha256_v1(&self.manifest).unwrap();

            test_verified_evaluation_manifest_commitment_v1_with_bindings(
                set.root_sha256(),
                &sealed_digest,
                &assignment_root,
                &pre_registration_root,
                &artifacts_root,
                self.expected.candidate_access_profile_sha256,
                self.expected.reveal_protocol_sha256,
            )
        }

        fn verify(
            &self,
        ) -> Result<VerifiedSealedEvaluationManifestV1, SealedEvaluationManifestRejectV1> {
            validate_sealed_evaluation_manifest_v1(
                &self.manifest,
                &self.episodes,
                &self.witness(),
                &self.expected,
            )
        }
    }

    #[test]
    fn fixed_roots_match_independent_jcs_reference_vectors() {
        let fixture = Fixture::new();
        let episode_set = verify_frozen_decision_episode_set_v1(&fixture.episodes).unwrap();
        assert_eq!(
            episode_set.root_sha256(),
            "75f59e31152587332c9784865a9fb6323c24bbdba52d57f08de1ca4cc0a5151a"
        );
        assert_eq!(
            arm_assignment_commitment_sha256_v1(&fixture.manifest.assignments[0]).unwrap(),
            "de44441d6cc243bdfa87720795f0b0cccd339aa18fe81042048a895b68fd56d1"
        );
        assert_eq!(
            arm_assignment_commitment_sha256_v1(&fixture.manifest.assignments[1]).unwrap(),
            "b31cad1a1fd6472b2cfffd49267e68ecaf4d6f6184c74c8f78978e27178bb181"
        );
        assert_eq!(
            arm_assignment_set_root_sha256_v1(
                &fixture.manifest.assignments, &fixture.episodes
            ).unwrap(),
            "f58f876336a886a0e6665fc0db926cb3ab46dc3a42339b4971491ec426dbaaed"
        );
        assert_eq!(
            pre_registration_root_sha256_v1(&fixture.manifest.pre_registration).unwrap(),
            "53f3c926d3f4e74c5c8d52e8073288610f9a84aebf0ea257dca7b6d66b8a3ae7"
        );
        assert_eq!(
            sealed_artifacts_root_sha256_v1(&fixture.manifest.sealed_artifacts).unwrap(),
            "12178fbd2e8f05f809fe97877ee888810674dc843572198f10f0ac5eba1906c9"
        );
        assert_eq!(
            sealed_evaluation_manifest_sha256_v1(&fixture.manifest).unwrap(),
            "74f920a119a64f326b1d6b7b6379138c66c55b451967cbe9deb39acf8bbc9f97"
        );
    }

    #[test]
    fn exact_assignment_artifact_and_witness_bindings_verify() {
        let fixture = Fixture::new();
        let verified = fixture.verify().unwrap();
        assert_eq!(verified.episode_count(), 2);
        assert_eq!(verified.split(), DecisionEpisodeSplitV1::HeldOut);
        assert_eq!(
            verified.episode_manifest_root_sha256(),
            fixture.manifest.episode_set.episode_manifest_root_sha256
        );
        assert_eq!(
            verified.sealed_artifacts_root_sha256(),
            sealed_artifacts_root_sha256_v1(&fixture.manifest.sealed_artifacts).unwrap()
        );
        assert_eq!(verified.manifest_sha256(),
            sealed_evaluation_manifest_sha256_v1(&fixture.manifest).unwrap());
    }

    #[test]
    fn missing_or_surplus_episode_assignment_is_rejected() {
        let fixture = Fixture::new();
        let witness = fixture.witness();
        let mut altered = fixture.manifest.clone();
        altered.assignments.pop();
        assert_eq!(
            validate_sealed_evaluation_manifest_v1(
                &altered, &fixture.episodes, &witness, &fixture.expected
            ),
            Err(SealedEvaluationManifestRejectV1::AssignmentCountMismatch)
        );
    }

    #[test]
    fn assignments_must_be_in_exact_public_episode_order() {
        let fixture = Fixture::new();
        let witness = fixture.witness();
        let mut altered = fixture.manifest.clone();
        altered.assignments.swap(0, 1);
        assert_eq!(
            validate_sealed_evaluation_manifest_v1(
                &altered, &fixture.episodes, &witness, &fixture.expected
            ),
            Err(SealedEvaluationManifestRejectV1::AssignmentEpisodeCoverageMismatch)
        );
    }

    #[test]
    fn duplicate_assignment_episode_is_rejected() {
        let mut fixture = Fixture::new();
        fixture.manifest.assignments[1].episode_id = "episode-a".to_owned();
        assert_eq!(
            arm_assignment_set_root_sha256_v1(
                &fixture.manifest.assignments, &fixture.episodes
            ),
            Err(SealedEvaluationManifestRejectV1::DuplicateAssignmentEpisodeId)
        );
    }

    #[test]
    fn public_seed_commitment_cannot_be_rebound() {
        let fixture = Fixture::new();
        let witness = fixture.witness();
        let mut altered = fixture.manifest.clone();
        altered.assignments[0].seed_commitment_sha256 = digest('f');
        assert_eq!(
            validate_sealed_evaluation_manifest_v1(
                &altered, &fixture.episodes, &witness, &fixture.expected
            ),
            Err(SealedEvaluationManifestRejectV1::AssignmentSeedCommitmentMismatch)
        );
    }

    #[test]
    fn treatment_or_arm_change_breaks_public_assignment_commitment() {
        let fixture = Fixture::new();
        let witness = fixture.witness();
        let mut altered = fixture.manifest.clone();
        altered.assignments[0].treatment_configuration_sha256 = digest('f');
        assert_eq!(
            validate_sealed_evaluation_manifest_v1(
                &altered, &fixture.episodes, &witness, &fixture.expected
            ),
            Err(SealedEvaluationManifestRejectV1::AssignmentCommitmentMismatch)
        );
    }

    #[test]
    fn evaluator_artifact_content_is_bound_to_the_original_witness() {
        let fixture = Fixture::new();
        let witness = fixture.witness();
        let mut altered = fixture.manifest.clone();
        altered.sealed_artifacts[0].content_sha256 = digest('f');
        assert_eq!(
            validate_sealed_evaluation_manifest_v1(
                &altered, &fixture.episodes, &witness, &fixture.expected
            ),
            Err(SealedEvaluationManifestRejectV1::ArtifactRootMismatch)
        );
    }

    #[test]
    fn preregistration_policy_is_bound_to_the_original_witness() {
        let fixture = Fixture::new();
        let witness = fixture.witness();
        let mut altered = fixture.manifest.clone();
        altered.pre_registration.exclusion_policy_sha256 = digest('f');
        assert_eq!(
            validate_sealed_evaluation_manifest_v1(
                &altered, &fixture.episodes, &witness, &fixture.expected
            ),
            Err(SealedEvaluationManifestRejectV1::PreRegistrationRootMismatch)
        );
    }

    #[test]
    fn t9_and_t10_are_distinct_sealed_artifacts() {
        let fixture = Fixture::new();
        let ids: Vec<&str> = fixture.manifest.sealed_artifacts
            .iter().map(|artifact| artifact.artifact_id.as_str()).collect();
        assert!(ids.iter().any(|id| id.contains("T9-whole-store-rollback")));
        assert!(ids.iter().any(|id| id.contains("T10-erasure-resurrection")));
        let full_root = sealed_artifacts_root_sha256_v1(&fixture.manifest.sealed_artifacts).unwrap();
        let t10_only_root = sealed_artifacts_root_sha256_v1(
            &fixture.manifest.sealed_artifacts[..1]
        ).unwrap();
        assert_ne!(full_root, t10_only_root);
    }

    #[test]
    fn t10_artifact_cannot_be_removed_without_breaking_witness() {
        let fixture = Fixture::new();
        // Preserve the independently committed baseline witness. Recomputing
        // a new witness after mutation would incorrectly model the evaluator
        // as cooperating with the tampering.
        let witness = fixture.witness();
        let mut altered = fixture.manifest.clone();
        altered.sealed_artifacts.remove(0);
        assert_eq!(
            validate_sealed_evaluation_manifest_v1(
                &altered, &fixture.episodes, &witness, &fixture.expected
            ),
            Err(SealedEvaluationManifestRejectV1::ArtifactRootMismatch)
        );
    }

    #[test]
    fn candidate_must_not_read_evaluator_state_before_trace_seal() {
        let mut fixture = Fixture::new();
        fixture.manifest.anti_leakage.candidate_readable_before_trace_seal = true;
        // Fixture::witness recomputes and signs the changed full-manifest digest,
        // proving the explicit anti-leakage condition is checked independently
        // of digest/commitment tampering.
        assert_eq!(
            fixture.verify(),
            Err(SealedEvaluationManifestRejectV1::CandidateReadableBeforeTraceSeal)
        );
    }

    #[test]
    fn noncanonical_artifact_order_and_duplicate_ids_are_rejected() {
        let mut fixture = Fixture::new();
        fixture.manifest.sealed_artifacts.swap(0, 1);
        assert_eq!(
            sealed_artifacts_root_sha256_v1(&fixture.manifest.sealed_artifacts),
            Err(SealedEvaluationManifestRejectV1::ArtifactOrderMismatch)
        );

        let mut duplicate = Fixture::new();
        duplicate.manifest.sealed_artifacts[1].artifact_id =
            duplicate.manifest.sealed_artifacts[0].artifact_id.clone();
        assert_eq!(
            sealed_artifacts_root_sha256_v1(&duplicate.manifest.sealed_artifacts),
            Err(SealedEvaluationManifestRejectV1::DuplicateArtifactId)
        );
    }

    #[test]
    fn evaluator_custody_profiles_are_consumer_pinned() {
        let mut fixture = Fixture::new();
        fixture.manifest.custody.evaluator_profile_sha256 = digest('f');
        assert_eq!(
            fixture.verify(),
            Err(SealedEvaluationManifestRejectV1::CustodyProfileMismatch)
        );
    }

    #[test]
    fn custody_must_follow_freeze_and_precede_witness_commit() {
        let mut before_freeze = Fixture::new();
        before_freeze.manifest.custody.committed_at_unix = 100;
        let witness = before_freeze.witness();
        assert_eq!(
            validate_sealed_evaluation_manifest_v1(
                &before_freeze.manifest, &before_freeze.episodes, &witness,
                &before_freeze.expected
            ),
            Err(SealedEvaluationManifestRejectV1::CustodyBeforeEpisodeFreeze)
        );

        let mut after_witness = Fixture::new();
        after_witness.manifest.custody.committed_at_unix = 120;
        let witness = after_witness.witness();
        assert_eq!(
            validate_sealed_evaluation_manifest_v1(
                &after_witness.manifest, &after_witness.episodes, &witness,
                &after_witness.expected
            ),
            Err(SealedEvaluationManifestRejectV1::CustodyAfterWitnessCommitment)
        );
    }

    #[test]
    fn unknown_fields_and_duplicate_json_fields_are_rejected() {
        let fixture = Fixture::new();
        let wire = serde_json::to_string(&fixture.manifest).unwrap();
        let duplicated = wire.replacen(
            "{\"schema_version\":\"mace_sealed_evaluation_manifest_v1\",",
            "{\"schema_version\":\"mace_sealed_evaluation_manifest_v1\",\"schema_version\":\"mace_sealed_evaluation_manifest_v1\",",
            1,
        );
        let witness = fixture.witness();
        assert_eq!(
            parse_and_verify_sealed_evaluation_manifest_v1(
                duplicated.as_bytes(), &fixture.episodes, &witness, &fixture.expected
            ),
            Err(SealedEvaluationManifestRejectV1::InvalidJson)
        );
    }
}
