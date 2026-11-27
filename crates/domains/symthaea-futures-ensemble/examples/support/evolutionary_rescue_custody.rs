// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Custody-safe projection of the evolutionary-rescue recorded-world fixture.
//!
//! The source SerializedWorld is an offline research artifact containing both observations and
//! evaluator truth. This adapter projects exactly one permitted observation stream into a
//! candidate capsule and keeps the trajectory, privileged observations, seed and split in an
//! evaluator-only capsule/manifest.
//!
//! Digests identify serialized bytes; they are not signatures, proof of truth, or access control.
//! Candidate/evaluator separation must also be enforced by filesystem/service custody.

use serde::{Deserialize, Serialize};
use symthaea_futures_symtropy::evolutionary_rescue::EvolutionaryRescueObservation;

#[path = "evolutionary_rescue_common.rs"]
mod common;

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateObservationCapsule {
    pub schema_version: u16,
    pub world_id: String,
    pub world_format_version: String,
    pub observation_policy_id: String,
    pub observations: Vec<EvolutionaryRescueObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateCapsuleReceipt {
    pub schema_version: u16,
    pub world_id: String,
    pub observation_policy_id: String,
    pub candidate_capsule_blake3: String,
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
    pub candidate_capsule_blake3: String,
    pub evaluator_capsule_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedCapsuleSet {
    pub candidate_capsule_json: Vec<u8>,
    pub candidate_receipt_json: Vec<u8>,
    pub evaluator_capsule_json: Vec<u8>,
    pub evaluator_manifest_json: Vec<u8>,
    pub candidate_digest: String,
    pub evaluator_digest: String,
}

pub fn encode_world(
    world: &common::SerializedWorld,
    world_id: &str,
    split: WorldSplit,
    view: ObservationView,
    source_commit: &str,
) -> Result<EncodedCapsuleSet, String> {
    validate_world(world, view)?;
    require_nonempty("world_id", world_id)?;
    validate_commit_sha(source_commit)?;

    let candidate = CandidateObservationCapsule {
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

    let candidate_capsule_json = serde_json::to_vec(&candidate)
        .map_err(|e| format!("serialize candidate capsule: {e}"))?;
    let evaluator_capsule_json = serde_json::to_vec(&evaluator)
        .map_err(|e| format!("serialize evaluator capsule: {e}"))?;
    let candidate_digest = blake3::hash(&candidate_capsule_json).to_hex().to_string();
    let evaluator_digest = blake3::hash(&evaluator_capsule_json).to_hex().to_string();

    let receipt = CandidateCapsuleReceipt {
        schema_version: CAPSULE_SCHEMA_VERSION,
        world_id: world_id.to_owned(),
        observation_policy_id: view.policy_id().to_owned(),
        candidate_capsule_blake3: candidate_digest.clone(),
    };
    let manifest = WorldCustodyManifest {
        schema_version: CAPSULE_SCHEMA_VERSION,
        world_id: world_id.to_owned(),
        world_format_version: world.format_version.clone(),
        source_commit: source_commit.to_owned(),
        split,
        seed: world.seed,
        observation_policy_id: view.policy_id().to_owned(),
        candidate_capsule_blake3: candidate_digest.clone(),
        evaluator_capsule_blake3: evaluator_digest.clone(),
    };

    Ok(EncodedCapsuleSet {
        candidate_capsule_json,
        candidate_receipt_json: serde_json::to_vec(&receipt)
            .map_err(|e| format!("serialize candidate receipt: {e}"))?,
        evaluator_capsule_json,
        evaluator_manifest_json: serde_json::to_vec(&manifest)
            .map_err(|e| format!("serialize evaluator manifest: {e}"))?,
        candidate_digest,
        evaluator_digest,
    })
}

/// Recomputes and verifies the evaluator's digest links. Call this before opening a world for
/// scoring. It deliberately operates on exact serialized bytes, not parsed/reformatted JSON.
pub fn verify_encoded_capsules(
    set: &EncodedCapsuleSet,
    expected_candidate_digest: &str,
    expected_evaluator_digest: &str,
) -> Result<(), String> {
    let candidate_actual = blake3::hash(&set.candidate_capsule_json).to_hex().to_string();
    let evaluator_actual = blake3::hash(&set.evaluator_capsule_json).to_hex().to_string();
    if candidate_actual != expected_candidate_digest || candidate_actual != set.candidate_digest {
        return Err("candidate capsule digest mismatch".into());
    }
    if evaluator_actual != expected_evaluator_digest || evaluator_actual != set.evaluator_digest {
        return Err("evaluator capsule digest mismatch".into());
    }
    let receipt: CandidateCapsuleReceipt = serde_json::from_slice(&set.candidate_receipt_json)
        .map_err(|e| format!("decode candidate receipt: {e}"))?;
    let manifest: WorldCustodyManifest = serde_json::from_slice(&set.evaluator_manifest_json)
        .map_err(|e| format!("decode evaluator manifest: {e}"))?;
    if receipt.candidate_capsule_blake3 != candidate_actual
        || manifest.candidate_capsule_blake3 != candidate_actual
        || manifest.evaluator_capsule_blake3 != evaluator_actual
        || receipt.world_id != manifest.world_id
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
    validate_stream("noisy_trait", &world.noisy_obs, world.trajectory.len(), true)?;
    validate_stream("shuffled_trait", &world.shuffled_obs, world.trajectory.len(), true)?;
    validate_stream("privileged", &world.privileged_obs, world.trajectory.len(), true)?;

    let expected_first_extinction = world.trajectory.iter().position(|&collapsed| collapsed).map(|i| i as u64);
    if world.first_extinction_tick != expected_first_extinction {
        return Err("first_extinction_tick disagrees with the recorded trajectory".into());
    }
    if world.trait_level_just_before_extinction.is_some_and(|v| !v.is_finite()) {
        return Err("trait_level_just_before_extinction must be finite".into());
    }
    match view {
        ObservationView::Census => {}
        ObservationView::NoisyTrait => validate_trait_presence("noisy_trait", &world.noisy_obs)?,
        ObservationView::ShuffledTraitControl => validate_trait_presence("shuffled_trait", &world.shuffled_obs)?,
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
            return Err(format!("{label} stream tick discontinuity at index {index}"));
        }
        if let Some(sample) = observation.sample {
            if let Some(value) = sample.observed_mean_forage_efficiency {
                if !trait_allowed {
                    return Err(format!("{label} stream unexpectedly contains trait data"));
                }
                if !value.is_finite() {
                    return Err(format!("{label} stream contains non-finite trait observation"));
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
        if observation.sample.is_some_and(|s| s.observed_mean_forage_efficiency.is_none()) {
            return Err(format!("{label} stream is missing a trait value at observed index {index}"));
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
            EvolutionaryRescueObservation { tick: 0, sample: sample(None) },
            EvolutionaryRescueObservation { tick: 1, sample: sample(None) },
        ];
        let noisy = vec![
            EvolutionaryRescueObservation { tick: 0, sample: sample(Some(0.2)) },
            EvolutionaryRescueObservation { tick: 1, sample: sample(Some(0.3)) },
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
    fn candidate_capsule_excludes_seed_truth_and_privileged_streams() {
        let encoded = encode_world(
            &world(), "d5a4dfc1-7e8c-4a9b-8e57-2bd995c5ce21",
            WorldSplit::Training, ObservationView::Census,
            "0123456789abcdef0123456789abcdef01234567",
        ).unwrap();
        let json = String::from_utf8(encoded.candidate_capsule_json).unwrap();
        for forbidden in ["\"seed\"", "\"trajectory_collapsed\"", "\"privileged_observations\"", "\"first_extinction_tick\"", "\"split\""] {
            assert!(!json.contains(forbidden), "candidate capsule leaked {forbidden}");
        }
        assert!(json.contains("population_census_v1"));
    }

    #[test]
    fn evaluator_manifest_binds_both_capsules_without_candidate_exposure() {
        let encoded = encode_world(
            &world(), "world-opaque-1", WorldSplit::HeldOut,
            ObservationView::NoisyTrait, "0123456789abcdef0123456789abcdef01234567",
        ).unwrap();
        verify_encoded_capsules(&encoded, &encoded.candidate_digest, &encoded.evaluator_digest).unwrap();
        let receipt = String::from_utf8(encoded.candidate_receipt_json).unwrap();
        assert!(!receipt.contains("evaluator_capsule_blake3"));
        assert!(!receipt.contains("\"seed\""));
        let manifest = String::from_utf8(encoded.evaluator_manifest_json).unwrap();
        assert!(manifest.contains("\"seed\":66") == false); // seed is evaluator-only; fixture seed is 11.
        assert!(manifest.contains("\"seed\":11"));
    }

    #[test]
    fn rejects_tick_discontinuity_before_materialization() {
        let mut w = world();
        w.census_obs[1].tick = 4;
        assert!(encode_world(
            &w, "opaque", WorldSplit::Training, ObservationView::Census,
            "0123456789abcdef0123456789abcdef01234567",
        ).unwrap_err().contains("tick discontinuity"));
    }

    #[test]
    fn rejects_inconsistent_extinction_metadata() {
        let mut w = world();
        w.first_extinction_tick = None;
        assert!(encode_world(
            &w, "opaque", WorldSplit::Training, ObservationView::Census,
            "0123456789abcdef0123456789abcdef01234567",
        ).unwrap_err().contains("first_extinction_tick"));
    }

    #[test]
    fn rejects_census_capsule_with_trait_leakage() {
        let mut w = world();
        w.census_obs[0].sample.as_mut().unwrap().observed_mean_forage_efficiency = Some(0.2);
        assert!(encode_world(
            &w, "opaque", WorldSplit::Training, ObservationView::Census,
            "0123456789abcdef0123456789abcdef01234567",
        ).unwrap_err().contains("unexpectedly contains trait data"));
    }

    #[test]
    fn rejects_tampered_capsule_bytes() {
        let mut encoded = encode_world(
            &world(), "opaque", WorldSplit::Training, ObservationView::Census,
            "0123456789abcdef0123456789abcdef01234567",
        ).unwrap();
        encoded.evaluator_capsule_json.push(b' ');
        assert!(verify_encoded_capsules(
            &encoded, &encoded.candidate_digest, &encoded.evaluator_digest,
        ).unwrap_err().contains("evaluator capsule digest mismatch"));
    }
}
