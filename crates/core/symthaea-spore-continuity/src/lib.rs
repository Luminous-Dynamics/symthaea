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

/// Every 32-byte digest in this ABI is BLAKE3 unless a future ABI version says otherwise.
pub type Digest32 = [u8; 32];
pub type VisualSeed = [u8; 32];
pub type ContinuityLineage = [u8; 16];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LifecycleSurface {
    Boot,
    Greeter,
    Session,
    Lock,
    Suspended,
    Recovery,
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleTransition {
    pub from: LifecycleSurface,
    pub to: LifecycleSurface,
}

impl LifecycleTransition {
    pub const fn new(from: LifecycleSurface, to: LifecycleSurface) -> Self {
        Self { from, to }
    }

    /// Conservative lifecycle graph. Direct Boot -> Session permits autologin;
    /// Session/Lock -> Greeter permits logout and user switching; recovery is
    /// reachable without asserting why recovery was entered.
    pub const fn is_allowed(self) -> bool {
        use LifecycleSurface::{Boot, Greeter, Lock, Recovery, Session, Shutdown, Suspended};
        matches!(
            (self.from, self.to),
            (Boot, Greeter | Session | Recovery)
                | (Greeter, Session | Suspended | Recovery | Shutdown)
                | (Session, Greeter | Lock | Suspended | Recovery | Shutdown)
                | (Lock, Greeter | Session | Suspended | Recovery | Shutdown)
                | (Suspended, Greeter | Session | Lock | Recovery)
                | (Recovery, Greeter | Session | Shutdown)
        )
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ContrastProfile {
    Standard,
    High,
}

/// Bounded semantic state that may cross a lifecycle renderer boundary.
///
/// `scene_digest` names the exact visual scene/package or agreed built-in scene
/// semantics. `visual_seed` carries deterministic identity without carrying a
/// machine identifier. `phase_micros` is fixed-point in [0, PHASE_SCALE] so
/// continuity does not depend on cross-runtime floating-point behavior.
///
/// `continuity_lineage` is a fresh random value for one ephemeral lifecycle
/// chain, not a credential or machine identifier. `handoff_sequence` must
/// increase within that lineage so consumers can reject stale/replayed state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuityState {
    pub version: u16,
    pub continuity_lineage: ContinuityLineage,
    pub handoff_sequence: u64,
    pub scene_digest: Digest32,
    pub visual_seed: VisualSeed,
    pub visual_plan_digest: Option<Digest32>,
    pub phase_micros: u32,
    pub world_age_ticks: u64,
    pub transition: LifecycleTransition,
    pub health: ContinuityHealth,
    pub quality: QualityProfile,
    pub motion: MotionProfile,
    pub contrast: ContrastProfile,
}

impl ContinuityState {
    pub fn new(
        continuity_lineage: ContinuityLineage,
        scene_digest: Digest32,
        visual_seed: VisualSeed,
        transition: LifecycleTransition,
    ) -> Self {
        Self {
            version: CONTINUITY_VERSION,
            continuity_lineage,
            handoff_sequence: 1,
            scene_digest,
            visual_seed,
            visual_plan_digest: None,
            phase_micros: 0,
            world_age_ticks: 0,
            transition,
            health: ContinuityHealth::Unknown,
            quality: QualityProfile::Standard,
            motion: MotionProfile::Standard,
            contrast: ContrastProfile::Standard,
        }
    }

    pub fn validate(&self) -> Result<(), ContinuityError> {
        if self.version != CONTINUITY_VERSION {
            return Err(ContinuityError::UnsupportedVersion(self.version));
        }
        if self.continuity_lineage.iter().all(|byte| *byte == 0) {
            return Err(ContinuityError::ZeroLineage);
        }
        if self.handoff_sequence == 0 {
            return Err(ContinuityError::ZeroSequence);
        }
        if !self.transition.is_allowed() {
            return Err(ContinuityError::InvalidTransition(self.transition));
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

    /// Validate that `next` may advance this same ephemeral continuity lineage.
    /// A new lineage is accepted by the owning lifecycle coordinator, not by
    /// silently treating unrelated packets as successors here.
    pub fn validate_successor(&self, next: &Self) -> Result<(), ContinuityError> {
        self.validate()?;
        next.validate()?;
        if self.continuity_lineage != next.continuity_lineage {
            return Err(ContinuityError::LineageChanged);
        }
        if next.handoff_sequence <= self.handoff_sequence {
            return Err(ContinuityError::SequenceNotAdvanced {
                previous: self.handoff_sequence,
                observed: next.handoff_sequence,
            });
        }
        if next.world_age_ticks < self.world_age_ticks {
            return Err(ContinuityError::WorldAgeRegressed {
                previous: self.world_age_ticks,
                observed: next.world_age_ticks,
            });
        }
        if self.transition.to != next.transition.from {
            return Err(ContinuityError::SurfaceDiscontinuity {
                previous_to: self.transition.to,
                next_from: next.transition.from,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContinuityError {
    UnsupportedVersion(u16),
    ZeroLineage,
    ZeroSequence,
    InvalidTransition(LifecycleTransition),
    PhaseOutOfRange(u32),
    ZeroSceneDigest,
    ZeroVisualSeed,
    LineageChanged,
    SequenceNotAdvanced { previous: u64, observed: u64 },
    WorldAgeRegressed { previous: u64, observed: u64 },
    SurfaceDiscontinuity {
        previous_to: LifecycleSurface,
        next_from: LifecycleSurface,
    },
    TooLarge { bytes: usize, max: usize },
    Serialization(String),
}

impl std::fmt::Display for ContinuityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported Spore continuity version {version}")
            }
            Self::ZeroLineage => write!(f, "continuity lineage may not be all-zero"),
            Self::ZeroSequence => write!(f, "continuity handoff sequence must start above zero"),
            Self::InvalidTransition(transition) => write!(
                f,
                "invalid lifecycle transition {:?} -> {:?}",
                transition.from, transition.to
            ),
            Self::PhaseOutOfRange(phase) => {
                write!(f, "continuity phase exceeds {PHASE_SCALE}: {phase}")
            }
            Self::ZeroSceneDigest => write!(f, "continuity scene digest may not be all-zero"),
            Self::ZeroVisualSeed => write!(f, "continuity visual seed may not be all-zero"),
            Self::LineageChanged => write!(f, "continuity successor changed lineage"),
            Self::SequenceNotAdvanced { previous, observed } => write!(
                f,
                "continuity sequence did not advance: previous={previous}, observed={observed}"
            ),
            Self::WorldAgeRegressed { previous, observed } => write!(
                f,
                "continuity world age regressed: previous={previous}, observed={observed}"
            ),
            Self::SurfaceDiscontinuity {
                previous_to,
                next_from,
            } => write!(
                f,
                "continuity surface chain broke: previous ended at {previous_to:?}, next starts at {next_from:?}"
            ),
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

    fn transition(from: LifecycleSurface, to: LifecycleSurface) -> LifecycleTransition {
        LifecycleTransition::new(from, to)
    }

    fn valid_state() -> ContinuityState {
        let mut lineage = [0u8; 16];
        lineage[0] = 9;
        let mut scene = [0u8; 32];
        scene[0] = 1;
        let mut seed = [0u8; 32];
        seed[31] = 2;
        ContinuityState::new(
            lineage,
            scene,
            seed,
            transition(LifecycleSurface::Boot, LifecycleSurface::Greeter),
        )
    }

    #[test]
    fn representative_real_lifecycle_transitions_are_supported() {
        assert!(transition(LifecycleSurface::Boot, LifecycleSurface::Greeter).is_allowed());
        assert!(transition(LifecycleSurface::Boot, LifecycleSurface::Session).is_allowed());
        assert!(transition(LifecycleSurface::Greeter, LifecycleSurface::Suspended).is_allowed());
        assert!(transition(LifecycleSurface::Session, LifecycleSurface::Greeter).is_allowed());
        assert!(transition(LifecycleSurface::Lock, LifecycleSurface::Greeter).is_allowed());
        assert!(transition(LifecycleSurface::Lock, LifecycleSurface::Suspended).is_allowed());
        assert!(transition(LifecycleSurface::Suspended, LifecycleSurface::Lock).is_allowed());
        assert!(!transition(LifecycleSurface::Shutdown, LifecycleSurface::Session).is_allowed());
        assert!(!transition(LifecycleSurface::Session, LifecycleSurface::Session).is_allowed());
    }

    #[test]
    fn round_trip_is_stable_and_bounded() {
        let mut state = valid_state();
        state.phase_micros = 420_000;
        state.world_age_ticks = 12_345;
        state.health = ContinuityHealth::Normal;
        state.motion = MotionProfile::Reduced;
        state.contrast = ContrastProfile::High;

        let bytes = state.encode_json().unwrap();
        assert!(bytes.len() <= MAX_CONTINUITY_BYTES);
        assert_eq!(ContinuityState::decode_json(&bytes).unwrap(), state);
    }

    #[test]
    fn rejects_invalid_version_phase_lineage_sequence_and_transition() {
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

        let mut state = valid_state();
        state.continuity_lineage = [0u8; 16];
        assert_eq!(state.validate(), Err(ContinuityError::ZeroLineage));

        let mut state = valid_state();
        state.handoff_sequence = 0;
        assert_eq!(state.validate(), Err(ContinuityError::ZeroSequence));

        let mut state = valid_state();
        state.transition = transition(LifecycleSurface::Shutdown, LifecycleSurface::Session);
        assert!(matches!(
            state.validate(),
            Err(ContinuityError::InvalidTransition(_))
        ));
    }

    #[test]
    fn zero_identity_material_is_rejected() {
        let state = ContinuityState::new(
            [9u8; 16],
            [0u8; 32],
            [1u8; 32],
            transition(LifecycleSurface::Boot, LifecycleSurface::Greeter),
        );
        assert_eq!(state.validate(), Err(ContinuityError::ZeroSceneDigest));

        let state = ContinuityState::new(
            [9u8; 16],
            [1u8; 32],
            [0u8; 32],
            transition(LifecycleSurface::Boot, LifecycleSurface::Greeter),
        );
        assert_eq!(state.validate(), Err(ContinuityError::ZeroVisualSeed));
    }

    #[test]
    fn successor_rejects_replay_lineage_age_and_surface_errors() {
        let mut previous = valid_state();
        previous.handoff_sequence = 4;
        previous.world_age_ticks = 100;

        let mut replay = previous.clone();
        assert!(matches!(
            previous.validate_successor(&replay),
            Err(ContinuityError::SequenceNotAdvanced { .. })
        ));

        replay.handoff_sequence = 5;
        replay.continuity_lineage = [8u8; 16];
        assert_eq!(
            previous.validate_successor(&replay),
            Err(ContinuityError::LineageChanged)
        );

        let mut rewind = previous.clone();
        rewind.handoff_sequence = 5;
        rewind.world_age_ticks = 99;
        rewind.transition = transition(LifecycleSurface::Greeter, LifecycleSurface::Session);
        assert!(matches!(
            previous.validate_successor(&rewind),
            Err(ContinuityError::WorldAgeRegressed { .. })
        ));

        let mut discontinuity = previous.clone();
        discontinuity.handoff_sequence = 5;
        discontinuity.world_age_ticks = 101;
        discontinuity.transition = transition(LifecycleSurface::Session, LifecycleSurface::Lock);
        assert!(matches!(
            previous.validate_successor(&discontinuity),
            Err(ContinuityError::SurfaceDiscontinuity { .. })
        ));

        let mut next = previous.clone();
        next.handoff_sequence = 5;
        next.world_age_ticks = 101;
        next.transition = transition(LifecycleSurface::Greeter, LifecycleSurface::Session);
        previous.validate_successor(&next).unwrap();
    }
}
