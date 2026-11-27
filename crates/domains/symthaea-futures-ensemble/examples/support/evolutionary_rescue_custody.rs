// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Custody-safe projection of the evolutionary-rescue recorded-world fixture.
//!
//! The source SerializedWorld is an offline research artifact containing both observations and
//! evaluator truth. This adapter projects exactly one permitted observation stream into a
//! custodian-only replay archive and keeps the trajectory, privileged observations, seed and split
//! in an evaluator-only capsule/manifest. The archive contains future observations and MUST NOT
//! be mounted into the candidate process; a trusted replay driver may expose only decision-time
//! prefixes through CandidateDecisionContext.
//!
//! Digests identify serialized bytes; they are not signatures, proof of truth, or access control.
//! Candidate/evaluator separation must also be enforced by filesystem/service custody.

use serde::{Deserialize, Serialize};
use symthaea_futures_symtropy::evolutionary_rescue::EvolutionaryRescueObservation;

use crate::common;

pub const CAPSULE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationView {
    Census,
    NoisyTrait,
    ShuffledTraitControl,
}

impl ObservationView {
    pub fn policy_id(self) -> &'static str {
        match self {
            Self::Census => "population_census_v1",
            Self::NoisyTrait => "noisy_trait_v1",
            Self::ShuffledTraitControl => "shuffled_trait_negative_control_v1",
        }
    }

    fn observations(self, world: &common::SerializedWorld) -> Vec<EvolutionaryRescueObservation> {
        match self {
            Self::Census => world.census_obs.clone(),
            Self::NoisyTrait => world.noisy_obs.clone(),
            Self::ShuffledTraitControl => world.shuffled_obs.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldSplit {
    Training,
    HeldOut,
}

/// Custodian-only full observation stream. Never pass this archive wholesale to a candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayObservationArchive {
    pub schema_version: u16,
    pub world_id: String,
    pub world_format_version: String,
    pub observation_policy_id: String,
    pub observations: Vec<EvolutionaryRescueObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayArchiveReceipt {
    pub schema_version: u16,
    pub world_id: String,
    pub observation_policy_id: String,
    pub replay_archive_blake3: String,
}

/// The only observation artifact a candidate should receive for one replay decision.
/// This type is constructed by the trusted replay driver from a custodian-only archive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionObservationPrefix {
    pub schema_version: u16,
    pub world_id: String,
    pub observation_policy_id: String,
    pub decision_tick: u64,
    pub observations: Vec<EvolutionaryRescueObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedDecisionPrefix {
    pub prefix_json: Vec<u8>,
    pub prefix_blake3: String,
}

/// Construct a time-bounded candidate input. The full replay archive must remain outside the
/// candidate process; only this prefix is eligible for delivery at the declared decision tick.
pub fn build_decision_prefix(
    archive: &ReplayObservationArchive,
    decision_tick: u64,
) -> Result<EncodedDecisionPrefix, String> {
    if archive.schema_version != CAPSULE_SCHEMA_VERSION {
        return Err("unsupported replay archive schema version".into());
    }
    if archive.observations.is_empty() {
        return Err("replay archive has no observations".into());
    }
    for (index, observation) in archive.observations.iter().enumerate() {
        if observation.tick != index as u64 {
            return Err(format!(
                "replay archive tick discontinuity at index {index}"
            ));
        }
    }
    let cutoff = usize::try_from(decision_tick)
        .map_err(|_| "decision tick does not fit platform index".to_owned())?;
    if cutoff >= archive.observations.len() {
        return Err("decision tick is beyond the replay archive".into());
    }

    let prefix = DecisionObservationPrefix {
        schema_version: CAPSULE_SCHEMA_VERSION,
        world_id: archive.world_id.clone(),
        observation_policy_id: archive.observation_policy_id.clone(),
        decision_tick,
        observations: archive.observations[..=cutoff].to_vec(),
    };
    let prefix_json = serde_json::to_vec(&prefix)
        .map_err(|e| format!("serialize decision observation prefix: {e}"))?;
    let prefix_blake3 = blake3::hash(&prefix_json).to_hex().to_string();
    Ok(EncodedDecisionPrefix {
        prefix_json,
        prefix_blake3,
    })
}

/// Evaluator-only material. Never pass this struct or its serialized bytes to candidate code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluatorOutcomeCapsule {
    pub schema_version: u16,
    pub world_id: String,
    pub world_format_version: String,
    pub seed: u64,
    pub trajectory_collapsed: Vec<bool>,
    pub privileged_observations: Vec<EvolutionaryRescueObservation>,
    pub max_population: usize,
    pub first_extinction_tick: Option<u64>,
    pub trait_level_just_before_extinction: Option<f64>,
}

/// Evaluator-only join record. It binds both capsule digests without revealing the evaluator
/// payload to the candidate. Keep it in the evaluator custody domain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldCustodyManifest {
    pub schema_version: u16,
    pub world_id: String,
    pub world_format_version: String,
    pub source_commit: String,
    pub split: WorldSplit,
    pub seed: u64,
    pub observation_policy_id: String,
    pub replay_archive_blake3: String,
    pub evaluator_capsule_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedReplayArtifacts {
    pub replay_archive_json: Vec<u8>,
    pub replay_receipt_json: Vec<u8>,
    pub evaluator_capsule_json: Vec<u8>,
    pub evaluator_manifest_json: Vec<u8>,
    pub replay_digest: String,
    pub evaluator_digest: String,
}

pub fn encode_world(
    world: &common::SerializedWorld,
    world_id: &str,
    split: WorldSplit,
    view: ObservationView,
    source_commit: &str,
) -> Result<EncodedReplayArtifacts, String> {
    validate_world(world, view)?;
    require_nonempty("world_id", world_id)?;
    validate_commit_sha(source_commit)?;

    let replay_archive = ReplayObservationArchive {
        schema_version: CAPSULE_SCHEMA_VERSION,
        world_id: world_id.to_owned(),
        world_format_version: world.format_version.clone(),
        observation_policy_id: view.policy_id().to_owned(),
        observations: view.observations(world),
    };
    let evaluator = EvaluatorOutcomeCapsule {
        schema_version: CAPSULE_SCHEMA_VERSION,
        world_id: world_id.to_owned(),
        world_format_version: world.format_version.clone(),
        seed: world.seed,
        trajectory_collapsed: world.trajectory.clone(),
        privileged_observations: world.privileged_obs.clone(),
        max_population: world.max_population,
        first_extinction_tick: world.first_extinction_tick,
        trait_level_just_before_extinction: world.trait_level_just_before_extinction,
    };

    let replay_archive_json = serde_json::to_vec(&replay_archive)
        .map_err(|e| format!("serialize replay archive: {e}"))?;
    let evaluator_capsule_json =
        serde_json::to_vec(&evaluator).map_err(|e| format!("serialize evaluator capsule: {e}"))?;
    let replay_digest = blake3::hash(&replay_archive_json).to_hex().to_string();
    let evaluator_digest = blake3::hash(&evaluator_capsule_json).to_hex().to_string();

    let receipt = ReplayArchiveReceipt {
        schema_version: CAPSULE_SCHEMA_VERSION,
        world_id: world_id.to_owned(),
        observation_policy_id: view.policy_id().to_owned(),
        replay_archive_blake3: replay_digest.clone(),
    };
    let manifest = WorldCustodyManifest {
        schema_version: CAPSULE_SCHEMA_VERSION,
        world_id: world_id.to_owned(),
        world_format_version: world.format_version.clone(),
        source_commit: source_commit.to_owned(),
        split,
        seed: world.seed,
        observation_policy_id: view.policy_id().to_owned(),
        replay_archive_blake3: replay_digest.clone(),
        evaluator_capsule_blake3: evaluator_digest.clone(),
    };

    Ok(EncodedReplayArtifacts {
        replay_archive_json,
        replay_receipt_json: serde_json::to_vec(&receipt)
            .map_err(|e| format!("serialize replay archive receipt: {e}"))?,
        evaluator_capsule_json,
        evaluator_manifest_json: serde_json::to_vec(&manifest)
            .map_err(|e| format!("serialize evaluator manifest: {e}"))?,
        replay_digest,
        evaluator_digest,
    })
}

/// Recomputes and verifies the evaluator's digest links. Call this before opening a world for
/// scoring. It deliberately operates on exact serialized bytes, not parsed/reformatted JSON.
pub fn verify_encoded_replay_artifacts(
    set: &EncodedReplayArtifacts,
    expected_replay_digest: &str,
    expected_evaluator_digest: &str,
) -> Result<(), String> {
    let replay_actual = blake3::hash(&set.replay_archive_json).to_hex().to_string();
    let evaluator_actual = blake3::hash(&set.evaluator_capsule_json)
        .to_hex()
        .to_string();
    if replay_actual != expected_replay_digest || replay_actual != set.replay_digest {
        return Err("replay archive digest mismatch".into());
    }
    if evaluator_actual != expected_evaluator_digest || evaluator_actual != set.evaluator_digest {
        return Err("evaluator capsule digest mismatch".into());
    }
    let receipt: ReplayArchiveReceipt = serde_json::from_slice(&set.replay_receipt_json)
        .map_err(|e| format!("decode replay archive receipt: {e}"))?;
    let manifest: WorldCustodyManifest = serde_json::from_slice(&set.evaluator_manifest_json)
        .map_err(|e| format!("decode evaluator manifest: {e}"))?;
    let candidate: ReplayObservationArchive = serde_json::from_slice(&set.replay_archive_json)
        .map_err(|e| format!("decode replay archive: {e}"))?;
    let evaluator: EvaluatorOutcomeCapsule = serde_json::from_slice(&set.evaluator_capsule_json)
        .map_err(|e| format!("decode evaluator capsule: {e}"))?;
    if receipt.schema_version != CAPSULE_SCHEMA_VERSION
        || manifest.schema_version != CAPSULE_SCHEMA_VERSION
        || candidate.schema_version != CAPSULE_SCHEMA_VERSION
        || evaluator.schema_version != CAPSULE_SCHEMA_VERSION
        || receipt.replay_archive_blake3 != replay_actual
        || manifest.replay_archive_blake3 != replay_actual
        || manifest.evaluator_capsule_blake3 != evaluator_actual
        || receipt.world_id != manifest.world_id
        || candidate.world_id != manifest.world_id
        || evaluator.world_id != manifest.world_id
        || candidate.world_format_version != manifest.world_format_version
        || evaluator.world_format_version != manifest.world_format_version
        || receipt.observation_policy_id != manifest.observation_policy_id
        || candidate.observation_policy_id != manifest.observation_policy_id
        || evaluator.seed != manifest.seed
    {
        return Err("capsule receipt/manifest linkage mismatch".into());
    }
    Ok(())
}

fn validate_world(world: &common::SerializedWorld, view: ObservationView) -> Result<(), String> {
    if world.format_version != common::WORLD_FORMAT_VERSION {
        return Err(format!("unsupported world format {}", world.format_version));
    }
    if world.trajectory.is_empty() {
        return Err("world trajectory must not be empty".into());
    }
    validate_stream("census", &world.census_obs, world.trajectory.len(), false)?;
    validate_stream(
        "noisy_trait",
        &world.noisy_obs,
        world.trajectory.len(),
        true,
    )?;
    validate_stream(
        "shuffled_trait",
        &world.shuffled_obs,
        world.trajectory.len(),
        true,
    )?;
    validate_stream(
        "privileged",
        &world.privileged_obs,
        world.trajectory.len(),
        true,
    )?;

    let expected_first_extinction = world
        .trajectory
        .iter()
        .position(|&collapsed| collapsed)
        .map(|i| i as u64);
    if world.first_extinction_tick != expected_first_extinction {
        return Err("first_extinction_tick disagrees with the recorded trajectory".into());
    }
    if world
        .trait_level_just_before_extinction
        .is_some_and(|v| !v.is_finite())
    {
        return Err("trait_level_just_before_extinction must be finite".into());
    }
    match view {
        ObservationView::Census => {}
        ObservationView::NoisyTrait => validate_trait_presence("noisy_trait", &world.noisy_obs)?,
        ObservationView::ShuffledTraitControl => {
            validate_trait_presence("shuffled_trait", &world.shuffled_obs)?
        }
    }
    Ok(())
}

fn validate_stream(
    label: &str,
    stream: &[EvolutionaryRescueObservation],
    expected_len: usize,
    trait_allowed: bool,
) -> Result<(), String> {
    if stream.len() != expected_len {
        return Err(format!("{label} stream length does not match trajectory"));
    }
    for (index, observation) in stream.iter().enumerate() {
        if observation.tick != index as u64 {
            return Err(format!(
                "{label} stream tick discontinuity at index {index}"
            ));
        }
        if let Some(sample) = observation.sample {
            if let Some(value) = sample.observed_mean_forage_efficiency {
                if !trait_allowed {
                    return Err(format!("{label} stream unexpectedly contains trait data"));
                }
                if !value.is_finite() {
                    return Err(format!(
                        "{label} stream contains non-finite trait observation"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_trait_presence(
    label: &str,
    stream: &[EvolutionaryRescueObservation],
) -> Result<(), String> {
    for (index, observation) in stream.iter().enumerate() {
        if observation
            .sample
            .is_some_and(|s| s.observed_mean_forage_efficiency.is_none())
        {
            return Err(format!(
                "{label} stream is missing a trait value at observed index {index}"
            ));
        }
    }
    Ok(())
}

fn validate_commit_sha(value: &str) -> Result<(), String> {
    let valid_len = value.len() == 40 || value.len() == 64;
    if !valid_len || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("source_commit must be a 40- or 64-character hexadecimal commit ID".into());
    }
    Ok(())
}

fn require_nonempty(field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{field} must not be empty"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_futures_symtropy::evolutionary_rescue::EvolutionaryRescueSample;

    fn sample(trait_value: Option<f64>) -> Option<EvolutionaryRescueSample> {
        Some(EvolutionaryRescueSample {
            sampled_alive_count: 12,
            observed_mean_forage_efficiency: trait_value,
        })
    }

    fn world() -> common::SerializedWorld {
        let census = vec![
            EvolutionaryRescueObservation {
                tick: 0,
                sample: sample(None),
            },
            EvolutionaryRescueObservation {
                tick: 1,
                sample: sample(None),
            },
        ];
        let noisy = vec![
            EvolutionaryRescueObservation {
                tick: 0,
                sample: sample(Some(0.2)),
            },
            EvolutionaryRescueObservation {
                tick: 1,
                sample: sample(Some(0.3)),
            },
        ];
        common::SerializedWorld {
            format_version: common::WORLD_FORMAT_VERSION.to_owned(),
            seed: 11,
            census_obs: census,
            noisy_obs: noisy.clone(),
            shuffled_obs: noisy.clone(),
            privileged_obs: noisy,
            trajectory: vec![false, true],
            max_population: 12,
            first_extinction_tick: Some(1),
            trait_level_just_before_extinction: Some(0.3),
        }
    }

    #[test]
    fn replay_archive_excludes_seed_truth_and_privileged_streams() {
        let encoded = encode_world(
            &world(),
            "d5a4dfc1-7e8c-4a9b-8e57-2bd995c5ce21",
            WorldSplit::Training,
            ObservationView::Census,
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();
        let json = String::from_utf8(encoded.replay_archive_json).unwrap();
        for forbidden in [
            "\"seed\"",
            "\"trajectory_collapsed\"",
            "\"privileged_observations\"",
            "\"first_extinction_tick\"",
            "\"split\"",
        ] {
            assert!(
                !json.contains(forbidden),
                "replay archive leaked {forbidden}"
            );
        }
        assert!(json.contains("population_census_v1"));
    }

    #[test]
    fn evaluator_manifest_binds_both_capsules_without_candidate_exposure() {
        let encoded = encode_world(
            &world(),
            "world-opaque-1",
            WorldSplit::HeldOut,
            ObservationView::NoisyTrait,
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();
        verify_encoded_replay_artifacts(
            &encoded,
            &encoded.replay_digest,
            &encoded.evaluator_digest,
        )
        .unwrap();
        let receipt = String::from_utf8(encoded.replay_receipt_json).unwrap();
        assert!(!receipt.contains("evaluator_capsule_blake3"));
        assert!(!receipt.contains("\"seed\""));
        let manifest = String::from_utf8(encoded.evaluator_manifest_json).unwrap();
        let parsed_manifest: WorldCustodyManifest = serde_json::from_str(&manifest).unwrap();
        assert_eq!(parsed_manifest.seed, 11);
        assert_eq!(parsed_manifest.split, WorldSplit::HeldOut);
    }

    #[test]
    fn decision_prefix_contains_no_future_observations() {
        let encoded = encode_world(
            &world(),
            "d5a4dfc1-7e8c-4a9b-8e57-2bd995c5ce21",
            WorldSplit::Training,
            ObservationView::Census,
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();
        let archive: ReplayObservationArchive =
            serde_json::from_slice(&encoded.replay_archive_json).unwrap();
        let prefix = build_decision_prefix(&archive, 0).unwrap();
        let parsed: DecisionObservationPrefix =
            serde_json::from_slice(&prefix.prefix_json).unwrap();
        assert_eq!(parsed.decision_tick, 0);
        assert_eq!(parsed.observations.len(), 1);
        assert_eq!(parsed.observations[0].tick, 0);
        assert!(prefix.prefix_blake3.len() == 64);
    }

    #[test]
    fn rejects_decision_tick_beyond_available_history() {
        let encoded = encode_world(
            &world(),
            "opaque",
            WorldSplit::Training,
            ObservationView::Census,
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();
        let archive: ReplayObservationArchive =
            serde_json::from_slice(&encoded.replay_archive_json).unwrap();
        assert!(
            build_decision_prefix(&archive, 2)
                .unwrap_err()
                .contains("beyond")
        );
    }

    #[test]
    fn rejects_tick_discontinuity_before_materialization() {
        let mut w = world();
        w.census_obs[1].tick = 4;
        assert!(
            encode_world(
                &w,
                "opaque",
                WorldSplit::Training,
                ObservationView::Census,
                "0123456789abcdef0123456789abcdef01234567",
            )
            .unwrap_err()
            .contains("tick discontinuity")
        );
    }

    #[test]
    fn rejects_inconsistent_extinction_metadata() {
        let mut w = world();
        w.first_extinction_tick = None;
        assert!(
            encode_world(
                &w,
                "opaque",
                WorldSplit::Training,
                ObservationView::Census,
                "0123456789abcdef0123456789abcdef01234567",
            )
            .unwrap_err()
            .contains("first_extinction_tick")
        );
    }

    #[test]
    fn rejects_census_capsule_with_trait_leakage() {
        let mut w = world();
        w.census_obs[0]
            .sample
            .as_mut()
            .unwrap()
            .observed_mean_forage_efficiency = Some(0.2);
        assert!(
            encode_world(
                &w,
                "opaque",
                WorldSplit::Training,
                ObservationView::Census,
                "0123456789abcdef0123456789abcdef01234567",
            )
            .unwrap_err()
            .contains("unexpectedly contains trait data")
        );
    }

    #[test]
    fn rejects_receipt_or_manifest_relinking() {
        let mut encoded = encode_world(
            &world(),
            "opaque",
            WorldSplit::Training,
            ObservationView::Census,
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();
        let mut manifest: WorldCustodyManifest =
            serde_json::from_slice(&encoded.evaluator_manifest_json).unwrap();
        manifest.world_id = "different-world".into();
        encoded.evaluator_manifest_json = serde_json::to_vec(&manifest).unwrap();
        assert!(
            verify_encoded_replay_artifacts(
                &encoded,
                &encoded.replay_digest,
                &encoded.evaluator_digest,
            )
            .unwrap_err()
            .contains("linkage mismatch")
        );
    }

    #[test]
    fn rejects_tampered_capsule_bytes() {
        let mut encoded = encode_world(
            &world(),
            "opaque",
            WorldSplit::Training,
            ObservationView::Census,
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();
        encoded.evaluator_capsule_json.push(b' ');
        assert!(
            verify_encoded_replay_artifacts(
                &encoded,
                &encoded.replay_digest,
                &encoded.evaluator_digest,
            )
            .unwrap_err()
            .contains("evaluator capsule digest mismatch")
        );
    }
}
