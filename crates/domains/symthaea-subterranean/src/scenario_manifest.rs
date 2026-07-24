// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible scenario manifests for certification campaigns.
//!
//! The built-in fingerprint is deterministic and suitable for detecting
//! accidental manifest drift. It is not a cryptographic signature or an
//! authentication mechanism.

use crate::requirements::RequirementId;
use crate::types::{NUM_STATE_CHANNELS, STATE_CHANNEL_RANGES, SubterraneanState};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SCENARIO_MANIFEST_SCHEMA_VERSION: u16 = 1;
pub const MAX_SCENARIO_STEPS: u32 = 2_000_000;
pub const MAX_STATE_OVERRIDES: usize = NUM_STATE_CHANNELS;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScenarioAcceptance {
    pub require_valid_final_state: bool,
    pub require_no_invariant_breach: bool,
    pub require_no_productive_work_at_red: bool,
    pub minimum_final_battery: Option<f64>,
}

impl Default for ScenarioAcceptance {
    fn default() -> Self {
        Self {
            require_valid_final_state: true,
            require_no_invariant_breach: true,
            require_no_productive_work_at_red: true,
            minimum_final_battery: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StateOverride {
    pub channel: usize,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioManifest {
    pub schema_version: u16,
    pub scenario_id: String,
    pub seed_phrase: String,
    pub dt_seconds: f32,
    pub steps: u32,
    pub phi: f64,
    pub state_overrides: Vec<StateOverride>,
    pub verifies: Vec<RequirementId>,
    pub tags: Vec<String>,
    #[serde(default)]
    pub acceptance: ScenarioAcceptance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioFingerprint(pub [u8; 32]);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScenarioManifestError {
    UnsupportedSchema(u16),
    EmptyScenarioId,
    EmptySeedPhrase,
    InvalidStepCount,
    InvalidTimeStep,
    InvalidPhi,
    TooManyOverrides,
    InvalidChannel(usize),
    DuplicateChannel(usize),
    InvalidChannelValue(usize),
    MissingRequirement,
    DuplicateRequirement(RequirementId),
}

impl ScenarioManifest {
    pub fn new(
        scenario_id: impl Into<String>,
        seed_phrase: impl Into<String>,
        steps: u32,
        verifies: Vec<RequirementId>,
    ) -> Self {
        Self {
            schema_version: SCENARIO_MANIFEST_SCHEMA_VERSION,
            scenario_id: scenario_id.into(),
            seed_phrase: seed_phrase.into(),
            dt_seconds: 0.005,
            steps,
            phi: 0.9,
            state_overrides: Vec::new(),
            verifies,
            tags: Vec::new(),
            acceptance: ScenarioAcceptance::default(),
        }
    }

    pub fn validate(&self) -> Result<(), ScenarioManifestError> {
        if self.schema_version != SCENARIO_MANIFEST_SCHEMA_VERSION {
            return Err(ScenarioManifestError::UnsupportedSchema(
                self.schema_version,
            ));
        }
        if self.scenario_id.trim().is_empty() {
            return Err(ScenarioManifestError::EmptyScenarioId);
        }
        if self.seed_phrase.trim().is_empty() {
            return Err(ScenarioManifestError::EmptySeedPhrase);
        }
        if self.steps == 0 || self.steps > MAX_SCENARIO_STEPS {
            return Err(ScenarioManifestError::InvalidStepCount);
        }
        if !self.dt_seconds.is_finite() || !(0.0001..=1.0).contains(&self.dt_seconds) {
            return Err(ScenarioManifestError::InvalidTimeStep);
        }
        if !self.phi.is_finite() || !(0.0..=1.0).contains(&self.phi) {
            return Err(ScenarioManifestError::InvalidPhi);
        }
        if self.state_overrides.len() > MAX_STATE_OVERRIDES {
            return Err(ScenarioManifestError::TooManyOverrides);
        }
        let mut channels = BTreeSet::new();
        for override_value in &self.state_overrides {
            if override_value.channel >= NUM_STATE_CHANNELS {
                return Err(ScenarioManifestError::InvalidChannel(
                    override_value.channel,
                ));
            }
            if !channels.insert(override_value.channel) {
                return Err(ScenarioManifestError::DuplicateChannel(
                    override_value.channel,
                ));
            }
            let (minimum, maximum) = STATE_CHANNEL_RANGES[override_value.channel];
            if !override_value.value.is_finite()
                || !(minimum..=maximum).contains(&override_value.value)
            {
                return Err(ScenarioManifestError::InvalidChannelValue(
                    override_value.channel,
                ));
            }
        }
        if let Some(minimum_battery) = self.acceptance.minimum_final_battery {
            if !minimum_battery.is_finite() || !(0.0..=1.0).contains(&minimum_battery) {
                return Err(ScenarioManifestError::InvalidChannelValue(7));
            }
        }
        if self.verifies.is_empty() {
            return Err(ScenarioManifestError::MissingRequirement);
        }
        let mut requirements = BTreeSet::new();
        for requirement in &self.verifies {
            if !requirements.insert(*requirement) {
                return Err(ScenarioManifestError::DuplicateRequirement(*requirement));
            }
        }
        Ok(())
    }

    pub fn initial_state(&self) -> Result<SubterraneanState, ScenarioManifestError> {
        self.validate()?;
        let mut state = SubterraneanState::home();
        for override_value in &self.state_overrides {
            state.channels[override_value.channel] = override_value.value;
        }
        Ok(state)
    }

    pub fn fingerprint(&self) -> Result<ScenarioFingerprint, ScenarioManifestError> {
        self.validate()?;
        let mut lanes = [
            0xcbf29ce484222325u64,
            0x9e3779b97f4a7c15,
            0x6a09e667f3bcc909,
            0xbb67ae8584caa73b,
        ];
        let bytes = self.canonical_bytes();
        for (index, byte) in bytes.iter().enumerate() {
            let lane = index % lanes.len();
            lanes[lane] ^= u64::from(*byte).wrapping_add(index as u64);
            lanes[lane] = lanes[lane].wrapping_mul(0x100000001b3);
            lanes[lane] = lanes[lane].rotate_left(((index + lane) % 63 + 1) as u32);
        }
        let mut digest = [0u8; 32];
        for (index, lane) in lanes.into_iter().enumerate() {
            digest[index * 8..(index + 1) * 8].copy_from_slice(&lane.to_le_bytes());
        }
        Ok(ScenarioFingerprint(digest))
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.schema_version.to_le_bytes());
        append_string(&mut bytes, &self.scenario_id);
        append_string(&mut bytes, &self.seed_phrase);
        bytes.extend_from_slice(&self.dt_seconds.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.steps.to_le_bytes());
        bytes.extend_from_slice(&self.phi.to_bits().to_le_bytes());
        let mut overrides = self.state_overrides.clone();
        overrides.sort_by_key(|value| value.channel);
        for value in overrides {
            bytes.extend_from_slice(&(value.channel as u64).to_le_bytes());
            bytes.extend_from_slice(&value.value.to_bits().to_le_bytes());
        }
        let mut requirements = self.verifies.clone();
        requirements.sort();
        for requirement in requirements {
            append_string(&mut bytes, requirement.code());
        }
        bytes.push(self.acceptance.require_valid_final_state as u8);
        bytes.push(self.acceptance.require_no_invariant_breach as u8);
        bytes.push(self.acceptance.require_no_productive_work_at_red as u8);
        bytes.extend_from_slice(
            &self
                .acceptance
                .minimum_final_battery
                .map(f64::to_bits)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        let mut tags = self.tags.clone();
        tags.sort();
        for tag in tags {
            append_string(&mut bytes, &tag);
        }
        bytes
    }
}

fn append_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_order_independent_for_sets() {
        let mut left = ScenarioManifest::new(
            "thermal",
            "scenario seed",
            100,
            vec![
                RequirementId::HazardPreemption,
                RequirementId::SafeCommandBounds,
            ],
        );
        left.tags = vec!["red".into(), "thermal".into()];
        left.state_overrides = vec![
            StateOverride {
                channel: 4,
                value: 150.0,
            },
            StateOverride {
                channel: 7,
                value: 0.8,
            },
        ];
        let mut right = left.clone();
        right.tags.reverse();
        right.verifies.reverse();
        right.state_overrides.reverse();
        assert_eq!(left.fingerprint(), right.fingerprint());
    }

    #[test]
    fn invalid_physical_override_is_rejected() {
        let mut manifest =
            ScenarioManifest::new("bad", "seed", 10, vec![RequirementId::SafeCommandBounds]);
        manifest.state_overrides.push(StateOverride {
            channel: 7,
            value: 2.0,
        });
        assert_eq!(
            manifest.validate(),
            Err(ScenarioManifestError::InvalidChannelValue(7))
        );
    }
}
