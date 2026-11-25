// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical public frozen-episode manifest verification and witnessed set
//! membership. Outcome verification consumes a private membership capability,
//! not an arbitrary per-episode digest supplied by the caller.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::evaluation_manifest_commitment_receipt::VerifiedEvaluationManifestCommitmentV1;

pub const DECISION_EPISODE_MANIFEST_SCHEMA_V1: &str = "mace_decision_episode_manifest_v1";
pub const MAX_DECISION_EPISODE_MANIFESTS_V1: usize = 10_000;
pub const MAX_DECISION_EPISODE_IDENTIFIER_BYTES_V1: usize = 256;
pub const MAX_DECISION_EPISODE_INDEX_V1: u64 = 1_000_000_000;
pub const MAX_DECISION_EPISODE_UNIX_SECONDS_V1: u64 = 253_402_300_799;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionEpisodeSplitV1 {
    Development,
    Validation,
    HeldOut,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionEpisodeSuiteV1 {
    pub suite_id: String,
    pub suite_version: String,
    pub task_family: String,
    pub split: DecisionEpisodeSplitV1,
    pub episode_index: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionEpisodeTaskV1 {
    pub task_input_sha256: String,
    pub source_snapshot_sha256: String,
    pub provenance_graph_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionEpisodeDecisionV1 {
    pub decision_invocation_id: String,
    pub decision_need_sha256: String,
    pub purpose_sha256: String,
    pub subject_scope_sha256: String,
    pub option_set_sha256: String,
    pub constraint_set_sha256: String,
    pub decision_policy_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionEpisodeRuntimeV1 {
    pub environment_revision_sha256: String,
    pub reader_profile_sha256: String,
    pub tool_profile_sha256: String,
    pub retrieval_policy_profile_sha256: String,
    pub runner_profile_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionEpisodeResourcesV1 {
    pub budget_profile_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionEpisodeRandomizationV1 {
    pub episode_seed_commitment_sha256: String,
    pub arm_assignment_commitment_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenDecisionEpisodeManifestV1 {
    pub schema_version: String,
    pub episode_id: String,
    pub suite: DecisionEpisodeSuiteV1,
    pub task: DecisionEpisodeTaskV1,
    pub decision: DecisionEpisodeDecisionV1,
    pub runtime: DecisionEpisodeRuntimeV1,
    pub resources: DecisionEpisodeResourcesV1,
    pub randomization: DecisionEpisodeRandomizationV1,
    pub frozen_at_unix: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrozenDecisionEpisodeManifestRejectV1 {
    TooManyManifests,
    EmptyManifestSet,
    InvalidJson,
    UnsupportedSchemaVersion,
    InvalidIdentifier,
    InvalidDigest,
    InvalidTimestamp,
    InvalidEpisodeIndex,
    NonCanonicalEpisodeOrder,
    DuplicateEpisodeId,
    MixedSuiteIdentity,
    EpisodeSetRootMismatch,
    EpisodeNotInWitnessedSet,
    WitnessSuiteMismatch,
    CanonicalizationFailed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedFrozenDecisionEpisodeSetV1 {
    root_sha256: String,
    suite_id: String,
    suite_version: String,
    split: DecisionEpisodeSplitV1,
    members: BTreeMap<String, String>,
}

impl VerifiedFrozenDecisionEpisodeSetV1 {
    pub fn root_sha256(&self) -> &str {
        &self.root_sha256
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

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Bind the complete verified public episode set to the detached
    /// commitment witness before issuing any per-episode membership.
    pub fn verify_against_witness(
        &self,
        witness: &VerifiedEvaluationManifestCommitmentV1,
    ) -> Result<(), FrozenDecisionEpisodeManifestRejectV1> {
        if self.root_sha256 != witness.episode_manifest_root_sha256() {
            return Err(FrozenDecisionEpisodeManifestRejectV1::EpisodeSetRootMismatch);
        }
        if self.suite_id != witness.suite_id() || self.suite_version != witness.suite_version() {
            return Err(FrozenDecisionEpisodeManifestRejectV1::WitnessSuiteMismatch);
        }
        Ok(())
    }

    /// Return a private-field capability for one exact episode only after
    /// proving it belongs to the complete set bound by the witness.
    pub fn bind_episode(
        &self,
        episode_id: &str,
        manifest_sha256: &str,
        witness: &VerifiedEvaluationManifestCommitmentV1,
    ) -> Result<VerifiedDecisionEpisodeMembershipV1, FrozenDecisionEpisodeManifestRejectV1> {
        self.verify_against_witness(witness)?;
        validate_atom(episode_id)?;
        validate_hash(manifest_sha256)?;
        match self.members.get(episode_id) {
            Some(actual) if actual == manifest_sha256 => {}
            _ => return Err(FrozenDecisionEpisodeManifestRejectV1::EpisodeNotInWitnessedSet),
        }
        Ok(VerifiedDecisionEpisodeMembershipV1 {
            episode_set_root_sha256: self.root_sha256.clone(),
            episode_id: episode_id.to_owned(),
            manifest_sha256: manifest_sha256.to_owned(),
            suite_id: self.suite_id.clone(),
            suite_version: self.suite_version.clone(),
            split: self.split,
        })
    }
}

/// Capability with no public constructor. It proves both the exact manifest
/// digest and membership in the full episode-set root tied to a verified
/// pre-run commitment witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedDecisionEpisodeMembershipV1 {
    episode_set_root_sha256: String,
    episode_id: String,
    manifest_sha256: String,
    suite_id: String,
    suite_version: String,
    split: DecisionEpisodeSplitV1,
}

impl VerifiedDecisionEpisodeMembershipV1 {
    pub fn episode_set_root_sha256(&self) -> &str {
        &self.episode_set_root_sha256
    }

    pub fn episode_id(&self) -> &str {
        &self.episode_id
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
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
}

pub fn parse_frozen_decision_episode_manifest_v1(
    wire: &[u8],
) -> Result<FrozenDecisionEpisodeManifestV1, FrozenDecisionEpisodeManifestRejectV1> {
    let manifest: FrozenDecisionEpisodeManifestV1 =
        serde_json::from_slice(wire).map_err(|_| FrozenDecisionEpisodeManifestRejectV1::InvalidJson)?;
    validate_frozen_decision_episode_manifest_v1(&manifest)?;
    Ok(manifest)
}

pub fn validate_frozen_decision_episode_manifest_v1(
    manifest: &FrozenDecisionEpisodeManifestV1,
) -> Result<(), FrozenDecisionEpisodeManifestRejectV1> {
    if manifest.schema_version != DECISION_EPISODE_MANIFEST_SCHEMA_V1 {
        return Err(FrozenDecisionEpisodeManifestRejectV1::UnsupportedSchemaVersion);
    }
    for atom in [
        manifest.episode_id.as_str(),
        manifest.suite.suite_id.as_str(),
        manifest.suite.suite_version.as_str(),
        manifest.suite.task_family.as_str(),
        manifest.decision.decision_invocation_id.as_str(),
    ] {
        validate_atom(atom)?;
    }
    if manifest.frozen_at_unix > MAX_DECISION_EPISODE_UNIX_SECONDS_V1 {
        return Err(FrozenDecisionEpisodeManifestRejectV1::InvalidTimestamp);
    }
    if manifest.suite.episode_index > MAX_DECISION_EPISODE_INDEX_V1 {
        return Err(FrozenDecisionEpisodeManifestRejectV1::InvalidEpisodeIndex);
    }
    for digest in [
        manifest.task.task_input_sha256.as_str(),
        manifest.task.source_snapshot_sha256.as_str(),
        manifest.task.provenance_graph_sha256.as_str(),
        manifest.decision.decision_need_sha256.as_str(),
        manifest.decision.purpose_sha256.as_str(),
        manifest.decision.subject_scope_sha256.as_str(),
        manifest.decision.option_set_sha256.as_str(),
        manifest.decision.constraint_set_sha256.as_str(),
        manifest.decision.decision_policy_sha256.as_str(),
        manifest.runtime.environment_revision_sha256.as_str(),
        manifest.runtime.reader_profile_sha256.as_str(),
        manifest.runtime.tool_profile_sha256.as_str(),
        manifest.runtime.retrieval_policy_profile_sha256.as_str(),
        manifest.runtime.runner_profile_sha256.as_str(),
        manifest.resources.budget_profile_sha256.as_str(),
        manifest.randomization.episode_seed_commitment_sha256.as_str(),
        manifest.randomization.arm_assignment_commitment_sha256.as_str(),
    ] {
        validate_hash(digest)?;
    }
    Ok(())
}

pub fn frozen_decision_episode_manifest_sha256_v1(
    manifest: &FrozenDecisionEpisodeManifestV1,
) -> Result<String, FrozenDecisionEpisodeManifestRejectV1> {
    validate_frozen_decision_episode_manifest_v1(manifest)?;
    let canonical = serde_json_canonicalizer::to_vec(manifest)
        .map_err(|_| FrozenDecisionEpisodeManifestRejectV1::CanonicalizationFailed)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:decision-episode-manifest:v1\0");
    hasher.update(&canonical);
    Ok(hex_digest(&hasher.finalize()))
}

/// Input order MUST already be strict ascending UTF-8 episode_id order. The
/// root uses each manifest's canonical digest and the same length-prefixed
/// ID encoding documented by MACE Frozen Decision Evaluation Manifests v0.1.
pub fn verify_frozen_decision_episode_set_v1(
    manifests: &[FrozenDecisionEpisodeManifestV1],
) -> Result<VerifiedFrozenDecisionEpisodeSetV1, FrozenDecisionEpisodeManifestRejectV1> {
    if manifests.is_empty() {
        return Err(FrozenDecisionEpisodeManifestRejectV1::EmptyManifestSet);
    }
    if manifests.len() > MAX_DECISION_EPISODE_MANIFESTS_V1 {
        return Err(FrozenDecisionEpisodeManifestRejectV1::TooManyManifests);
    }

    let suite_id = manifests[0].suite.suite_id.clone();
    let suite_version = manifests[0].suite.suite_version.clone();
    let split = manifests[0].suite.split;
    let mut members = BTreeMap::new();
    let mut previous_id: Option<&str> = None;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:decision-episode-set:v1\0");
    hasher.update((manifests.len() as u64).to_be_bytes());

    for manifest in manifests {
        validate_frozen_decision_episode_manifest_v1(manifest)?;
        if manifest.suite.suite_id != suite_id
            || manifest.suite.suite_version != suite_version
            || manifest.suite.split != split
        {
            return Err(FrozenDecisionEpisodeManifestRejectV1::MixedSuiteIdentity);
        }
        if let Some(previous) = previous_id {
            match previous.as_bytes().cmp(manifest.episode_id.as_bytes()) {
                std::cmp::Ordering::Equal => {
                    return Err(FrozenDecisionEpisodeManifestRejectV1::DuplicateEpisodeId);
                }
                std::cmp::Ordering::Greater => {
                    return Err(FrozenDecisionEpisodeManifestRejectV1::NonCanonicalEpisodeOrder);
                }
                std::cmp::Ordering::Less => {}
            }
        }
        previous_id = Some(&manifest.episode_id);

        let manifest_sha256 = frozen_decision_episode_manifest_sha256_v1(manifest)?;
        hasher.update((manifest.episode_id.len() as u64).to_be_bytes());
        hasher.update(manifest.episode_id.as_bytes());
        hasher.update(decode_hash(&manifest_sha256)?);
        members.insert(manifest.episode_id.clone(), manifest_sha256);
    }

    Ok(VerifiedFrozenDecisionEpisodeSetV1 {
        root_sha256: hex_digest(&hasher.finalize()),
        suite_id,
        suite_version,
        split,
        members,
    })
}

fn validate_atom(value: &str) -> Result<(), FrozenDecisionEpisodeManifestRejectV1> {
    if value.is_empty()
        || value.len() > MAX_DECISION_EPISODE_IDENTIFIER_BYTES_V1
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(FrozenDecisionEpisodeManifestRejectV1::InvalidIdentifier);
    }
    Ok(())
}

fn validate_hash(value: &str) -> Result<(), FrozenDecisionEpisodeManifestRejectV1> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(FrozenDecisionEpisodeManifestRejectV1::InvalidDigest);
    }
    Ok(())
}

fn decode_hash(value: &str) -> Result<[u8; 32], FrozenDecisionEpisodeManifestRejectV1> {
    validate_hash(value)?;
    let mut output = [0u8; 32];
    for (i, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        output[i] = (high << 4) | low;
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Result<u8, FrozenDecisionEpisodeManifestRejectV1> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(FrozenDecisionEpisodeManifestRejectV1::InvalidDigest),
    }
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

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    pub(super) fn fixture_manifest(id: &str) -> FrozenDecisionEpisodeManifestV1 {
        FrozenDecisionEpisodeManifestV1 {
            schema_version: DECISION_EPISODE_MANIFEST_SCHEMA_V1.to_owned(),
            episode_id: id.to_owned(),
            suite: DecisionEpisodeSuiteV1 {
                suite_id: "mace-test".to_owned(),
                suite_version: "v1".to_owned(),
                task_family: "memory-governance".to_owned(),
                split: DecisionEpisodeSplitV1::HeldOut,
                episode_index: 1,
            },
            task: DecisionEpisodeTaskV1 {
                task_input_sha256: digest('a'),
                source_snapshot_sha256: digest('b'),
                provenance_graph_sha256: digest('c'),
            },
            decision: DecisionEpisodeDecisionV1 {
                decision_invocation_id: format!("decision-{id}"),
                decision_need_sha256: digest('d'),
                purpose_sha256: digest('e'),
                subject_scope_sha256: digest('f'),
                option_set_sha256: digest('1'),
                constraint_set_sha256: digest('2'),
                decision_policy_sha256: digest('3'),
            },
            runtime: DecisionEpisodeRuntimeV1 {
                environment_revision_sha256: digest('4'),
                reader_profile_sha256: digest('5'),
                tool_profile_sha256: digest('6'),
                retrieval_policy_profile_sha256: digest('7'),
                runner_profile_sha256: digest('8'),
            },
            resources: DecisionEpisodeResourcesV1 {
                budget_profile_sha256: digest('9'),
            },
            randomization: DecisionEpisodeRandomizationV1 {
                episode_seed_commitment_sha256: digest('0'),
                arm_assignment_commitment_sha256: digest('a'),
            },
            frozen_at_unix: 100,
        }
    }

    #[test]
    fn episode_set_root_is_deterministic_for_fixed_manifests() {
        let a = fixture_manifest("episode-a");
        let b = fixture_manifest("episode-b");
        let set = verify_frozen_decision_episode_set_v1(&[a.clone(), b.clone()]).unwrap();
        let reparsed = parse_frozen_decision_episode_manifest_v1(
            &serde_json::to_vec(&a).unwrap(),
        ).unwrap();
        assert_eq!(
            frozen_decision_episode_manifest_sha256_v1(&reparsed).unwrap(),
            frozen_decision_episode_manifest_sha256_v1(&a).unwrap()
        );
        let repeated = verify_frozen_decision_episode_set_v1(&[a, b]).unwrap();
        assert_eq!(set.root_sha256(), repeated.root_sha256());
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn episode_set_requires_canonical_order_and_rejects_duplicates() {
        let a = fixture_manifest("episode-a");
        let b = fixture_manifest("episode-b");
        assert_eq!(
            verify_frozen_decision_episode_set_v1(&[b.clone(), a.clone()]),
            Err(FrozenDecisionEpisodeManifestRejectV1::NonCanonicalEpisodeOrder)
        );
        assert_eq!(
            verify_frozen_decision_episode_set_v1(&[a.clone(), a]),
            Err(FrozenDecisionEpisodeManifestRejectV1::DuplicateEpisodeId)
        );
    }

    #[test]
    fn mixed_suite_identity_is_rejected() {
        let a = fixture_manifest("episode-a");
        let mut b = fixture_manifest("episode-b");
        b.suite.suite_version = "v2".to_owned();
        assert_eq!(
            verify_frozen_decision_episode_set_v1(&[a, b]),
            Err(FrozenDecisionEpisodeManifestRejectV1::MixedSuiteIdentity)
        );
    }

    #[test]
    fn mixed_development_and_held_out_splits_are_rejected() {
        let a = fixture_manifest("episode-a");
        let mut b = fixture_manifest("episode-b");
        b.suite.split = DecisionEpisodeSplitV1::Development;
        assert_eq!(
            verify_frozen_decision_episode_set_v1(&[a, b]),
            Err(FrozenDecisionEpisodeManifestRejectV1::MixedSuiteIdentity)
        );
    }

    #[test]
    fn episode_index_is_bounded_by_the_schema() {
        let mut manifest = fixture_manifest("episode-a");
        manifest.suite.episode_index = MAX_DECISION_EPISODE_INDEX_V1 + 1;
        assert_eq!(
            validate_frozen_decision_episode_manifest_v1(&manifest),
            Err(FrozenDecisionEpisodeManifestRejectV1::InvalidEpisodeIndex)
        );
    }

    #[test]
    fn episode_membership_requires_the_exact_witnessed_root_and_digest() {
        let manifest = fixture_manifest("episode-a");
        let manifest_sha256 = frozen_decision_episode_manifest_sha256_v1(&manifest).unwrap();
        let set = verify_frozen_decision_episode_set_v1(&[manifest.clone()]).unwrap();
        let witness =
            crate::evaluation_manifest_commitment_receipt::test_verified_evaluation_manifest_commitment_with_episode_root_v1(
                set.root_sha256(),
            );
        let membership = set
            .bind_episode(&manifest.episode_id, &manifest_sha256, &witness)
            .unwrap();
        assert_eq!(membership.episode_id(), "episode-a");
        assert_eq!(membership.manifest_sha256(), manifest_sha256);
        assert_eq!(membership.episode_set_root_sha256(), witness.episode_manifest_root_sha256());

        let wrong_digest = digest('0');
        assert_eq!(
            set.bind_episode(&manifest.episode_id, &wrong_digest, &witness),
            Err(FrozenDecisionEpisodeManifestRejectV1::EpisodeNotInWitnessedSet)
        );
    }

    #[test]
    fn episode_set_root_must_match_detached_witness() {
        let manifest = fixture_manifest("episode-a");
        let set = verify_frozen_decision_episode_set_v1(&[manifest]).unwrap();
        let wrong_root = digest('0');
        let witness =
            crate::evaluation_manifest_commitment_receipt::test_verified_evaluation_manifest_commitment_with_episode_root_v1(
                &wrong_root,
            );
        assert_eq!(
            set.verify_against_witness(&witness),
            Err(FrozenDecisionEpisodeManifestRejectV1::EpisodeSetRootMismatch)
        );
    }

    #[test]
    fn malformed_digest_and_unknown_json_fields_are_rejected() {
        let mut a = fixture_manifest("episode-a");
        a.task.task_input_sha256 = "not-a-digest".to_owned();
        assert_eq!(
            validate_frozen_decision_episode_manifest_v1(&a),
            Err(FrozenDecisionEpisodeManifestRejectV1::InvalidDigest)
        );
        let wire = serde_json::to_string(&fixture_manifest("episode-a")).unwrap();
        let with_unknown = wire.replacen(
            "{\"schema_version\":\"mace_decision_episode_manifest_v1\",",
            "{\"schema_version\":\"mace_decision_episode_manifest_v1\",\"untrusted\":true,",
            1,
        );
        assert_eq!(
            parse_frozen_decision_episode_manifest_v1(with_unknown.as_bytes()),
            Err(FrozenDecisionEpisodeManifestRejectV1::InvalidJson)
        );
    }
}

#[cfg(test)]
pub(crate) fn test_verified_episode_membership_v1() -> VerifiedDecisionEpisodeMembershipV1 {
    let manifest = tests::fixture_manifest("episode-1");
    let manifest_sha256 = frozen_decision_episode_manifest_sha256_v1(&manifest).unwrap();
    let set = verify_frozen_decision_episode_set_v1(&[manifest.clone()]).unwrap();
    let witness =
        crate::evaluation_manifest_commitment_receipt::test_verified_evaluation_manifest_commitment_with_episode_root_v1(
            set.root_sha256(),
        );
    set.bind_episode(&manifest.episode_id, &manifest_sha256, &witness)
        .unwrap()
}
