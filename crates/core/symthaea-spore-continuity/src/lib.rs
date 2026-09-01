// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Semantic continuity between independently isolated Spore lifecycle renderers.
//!
//! This crate intentionally carries no framebuffer contents, process metadata,
//! file paths, journal text, credentials, or boot-control authority. A producer
//! can disappear entirely and the next renderer must fall back safely.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

pub const CONTINUITY_VERSION: u16 = 1;
pub const MAX_CONTINUITY_BYTES: usize = 2048;
pub const PHASE_SCALE: u32 = 1_000_000;

pub type Digest32 = [u8; 32];
pub type VisualSeed = [u8; 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LifecycleTransition {
    BootToGreeter,
    GreeterToSession,
    SessionToLock,
    LockToSession,
    SessionToSuspend,
    SuspendToSession,
    SessionToShutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ContinuityHealth {
    Normal,
    Delayed,
    Degraded,
    Failed,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QualityProfile {
    Calm,
    Standard,
    Rich,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MotionProfile {
    Reduced,
    Standard,
}

/// Bounded semantic state that may cross a lifecycle renderer boundary.
///
/// `scene_digest` names the exact visual scene/package or agreed built-in scene
/// semantics. `visual_seed` carries deterministic identity without carrying a
/// machine identifier. `phase_micros` is fixed-point in [0, PHASE_SCALE] so
/// continuity does not depend on cross-runtime floating-point behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuityState {
    pub version: u16,
    pub scene_digest: Digest32,
    pub visual_seed: VisualSeed,
    pub visual_plan_digest: Option<Digest32>,
    pub phase_micros: u32,
    pub world_age_ticks: u64,
    pub transition: LifecycleTransition,
    pub health: ContinuityHealth,
    pub quality: QualityProfile,
    pub motion: MotionProfile,
}

impl ContinuityState {
    pub fn new(
        scene_digest: Digest32,
        visual_seed: VisualSeed,
        transition: LifecycleTransition,
    ) -> Self {
        Self {
            version: CONTINUITY_VERSION,
            scene_digest,
            visual_seed,
            visual_plan_digest: None,
            phase_micros: 0,
            world_age_ticks: 0,
            transition,
            health: ContinuityHealth::Unknown,
            quality: QualityProfile::Standard,
            motion: MotionProfile::Standard,
        }
    }

    pub fn validate(&self) -> Result<(), ContinuityError> {
        if self.version != CONTINUITY_VERSION {
            return Err(ContinuityError::UnsupportedVersion(self.version));
        }
        if self.phase_micros > PHASE_SCALE {
            return Err(ContinuityError::PhaseOutOfRange(self.phase_micros));
        }
        if self.scene_digest.iter().all(|byte| *byte == 0) {
            return Err(ContinuityError::ZeroSceneDigest);
        }
        if self.visual_seed.iter().all(|byte| *byte == 0) {
            return Err(ContinuityError::ZeroVisualSeed);
        }
        Ok(())
    }

    pub fn encode_json(&self) -> Result<Vec<u8>, ContinuityError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| ContinuityError::Serialization(error.to_string()))?;
        if bytes.len() > MAX_CONTINUITY_BYTES {
            return Err(ContinuityError::TooLarge {
                bytes: bytes.len(),
                max: MAX_CONTINUITY_BYTES,
            });
        }
        Ok(bytes)
    }

    pub fn decode_json(bytes: &[u8]) -> Result<Self, ContinuityError> {
        if bytes.len() > MAX_CONTINUITY_BYTES {
            return Err(ContinuityError::TooLarge {
                bytes: bytes.len(),
                max: MAX_CONTINUITY_BYTES,
            });
        }
        let state: Self = serde_json::from_slice(bytes)
            .map_err(|error| ContinuityError::Serialization(error.to_string()))?;
        state.validate()?;
        Ok(state)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContinuityError {
    UnsupportedVersion(u16),
    PhaseOutOfRange(u32),
    ZeroSceneDigest,
    ZeroVisualSeed,
    TooLarge { bytes: usize, max: usize },
    Serialization(String),
}

impl std::fmt::Display for ContinuityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported Spore continuity version {version}")
            }
            Self::PhaseOutOfRange(phase) => {
                write!(f, "continuity phase exceeds {PHASE_SCALE}: {phase}")
            }
            Self::ZeroSceneDigest => write!(f, "continuity scene digest may not be all-zero"),
            Self::ZeroVisualSeed => write!(f, "continuity visual seed may not be all-zero"),
            Self::TooLarge { bytes, max } => {
                write!(f, "continuity payload exceeds size bound: {bytes} > {max}")
            }
            Self::Serialization(error) => write!(f, "continuity serialization error: {error}"),
        }
    }
}

impl std::error::Error for ContinuityError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_state() -> ContinuityState {
        let mut scene = [0u8; 32];
        scene[0] = 1;
        let mut seed = [0u8; 32];
        seed[31] = 2;
        ContinuityState::new(scene, seed, LifecycleTransition::BootToGreeter)
    }

    #[test]
    fn round_trip_is_stable_and_bounded() {
        let mut state = valid_state();
        state.phase_micros = 420_000;
        state.world_age_ticks = 12_345;
        state.health = ContinuityHealth::Normal;
        state.motion = MotionProfile::Reduced;

        let bytes = state.encode_json().unwrap();
        assert!(bytes.len() <= MAX_CONTINUITY_BYTES);
        assert_eq!(ContinuityState::decode_json(&bytes).unwrap(), state);
    }

    #[test]
    fn rejects_invalid_version_and_phase() {
        let mut state = valid_state();
        state.version = CONTINUITY_VERSION + 1;
        assert!(matches!(
            state.validate(),
            Err(ContinuityError::UnsupportedVersion(_))
        ));

        let mut state = valid_state();
        state.phase_micros = PHASE_SCALE + 1;
        assert!(matches!(
            state.validate(),
            Err(ContinuityError::PhaseOutOfRange(_))
        ));
    }

    #[test]
    fn zero_identity_material_is_rejected() {
        let state = ContinuityState::new(
            [0u8; 32],
            [1u8; 32],
            LifecycleTransition::BootToGreeter,
        );
        assert_eq!(state.validate(), Err(ContinuityError::ZeroSceneDigest));

        let state = ContinuityState::new(
            [1u8; 32],
            [0u8; 32],
            LifecycleTransition::BootToGreeter,
        );
        assert_eq!(state.validate(), Err(ContinuityError::ZeroVisualSeed));
    }
}
