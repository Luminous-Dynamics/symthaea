// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Renderer-neutral final semantic input for Spore boot presentation.
//!
//! This crate combines already-validated live modulation with the deterministic
//! elastic visual clock. It does not inspect Linux/systemd, choose a BootGenome,
//! render pixels, own DRM, or authorize display handoff.

#![forbid(unsafe_code)]

use std::fmt;

use symthaea_boot_ecology_live::{
    DiagnosticFloor, DomainMask, LiveEcologyModulation, REVEAL_SCALE, SemanticBootAnchor,
    VisualAccent,
};
use symthaea_boot_protocol::BootHealth;
use symthaea_boot_visual_clock::{
    ClockAdvance, ClockError, ClockMode, ElasticVisualClock, VisualClockPolicy, truth_band,
};

pub const FRAME_SCHEMA_VERSION: u16 = 1;
pub const TRACE_SCHEMA_VERSION: u16 = 1;

/// One complete, bounded semantic frame supplied to diagnostics and the exact
/// visual renderer. It contains no raw logs, process metadata, strings, or user
/// content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EcologyFrameInput {
    pub schema_version: u16,
    pub observation_sequence: u64,
    pub anchor: SemanticBootAnchor,
    pub health: BootHealth,
    pub visual_phase: u32,
    pub reveal_floor: u32,
    pub truth_ceiling: u32,
    pub delayed_domains: DomainMask,
    pub degraded_domains: DomainMask,
    pub failed_domains: DomainMask,
    pub diagnostic_floor: DiagnosticFloor,
    pub accent_token: u64,
    pub accent: VisualAccent,
    pub clock_mode: ClockMode,
    /// Presentation fact only. Host lifecycle policy remains authoritative.
    pub handoff_ready: bool,
}

impl EcologyFrameInput {
    pub fn from_step(
        modulation: &LiveEcologyModulation,
        advance: ClockAdvance,
    ) -> Result<Self, FrameError> {
        let band = truth_band(modulation.anchor);
        let frame = Self {
            schema_version: FRAME_SCHEMA_VERSION,
            observation_sequence: modulation.observation_sequence,
            anchor: modulation.anchor,
            health: modulation.health,
            visual_phase: advance.after,
            reveal_floor: modulation.reveal_floor,
            truth_ceiling: band.ceiling,
            delayed_domains: modulation.delayed_domains,
            degraded_domains: modulation.degraded_domains,
            failed_domains: modulation.failed_domains,
            diagnostic_floor: modulation.diagnostic_floor,
            accent_token: modulation.accent_token,
            accent: modulation.accent,
            clock_mode: advance.mode,
            handoff_ready: modulation.handoff_ready,
        };
        frame.validate()?;
        Ok(frame)
    }

    pub fn validate(&self) -> Result<(), FrameError> {
        if self.schema_version != FRAME_SCHEMA_VERSION {
            return Err(FrameError::UnsupportedVersion(self.schema_version));
        }

        let band = truth_band(self.anchor);
        if self.reveal_floor != self.anchor.reveal_floor() || self.reveal_floor != band.floor {
            return Err(FrameError::RevealFloorMismatch {
                expected: band.floor,
                observed: self.reveal_floor,
            });
        }
        if self.truth_ceiling != band.ceiling {
            return Err(FrameError::TruthCeilingMismatch {
                expected: band.ceiling,
                observed: self.truth_ceiling,
            });
        }
        if self.visual_phase > self.truth_ceiling || self.visual_phase > REVEAL_SCALE {
            return Err(FrameError::PhaseOutsideTruthBand {
                phase: self.visual_phase,
                ceiling: self.truth_ceiling,
            });
        }

        let delayed = self.delayed_domains.bits();
        let degraded = self.degraded_domains.bits();
        let failed = self.failed_domains.bits();
        if (delayed & degraded) != 0 || (delayed & failed) != 0 || (degraded & failed) != 0 {
            return Err(FrameError::OverlappingDomainStates);
        }

        if (!degraded_is_empty(degraded) || !failed_is_empty(failed))
            && self.diagnostic_floor < DiagnosticFloor::Diagnostics
        {
            return Err(FrameError::DiagnosticFloorTooLow);
        }
        if !delayed_is_empty(delayed) && self.diagnostic_floor < DiagnosticFloor::Status {
            return Err(FrameError::DiagnosticFloorTooLow);
        }

        if self.handoff_ready && self.anchor != SemanticBootAnchor::SessionReady {
            return Err(FrameError::InvalidHandoffFact);
        }

        match self.clock_mode {
            ClockMode::CatchUp => {
                if self.visual_phase > self.reveal_floor {
                    return Err(FrameError::ClockModeMismatch);
                }
            }
            ClockMode::AmbientDrift => {
                if self.health != BootHealth::Normal
                    || self.diagnostic_floor != DiagnosticFloor::Ambient
                    || self.visual_phase < self.reveal_floor
                    || self.visual_phase > self.truth_ceiling
                {
                    return Err(FrameError::ClockModeMismatch);
                }
            }
            ClockMode::Complete => {
                if self.anchor != SemanticBootAnchor::SessionReady
                    || self.visual_phase != REVEAL_SCALE
                {
                    return Err(FrameError::ClockModeMismatch);
                }
            }
            ClockMode::Hold => {}
        }

        match self.accent {
            VisualAccent::Ready => {
                if !self.handoff_ready || self.health != BootHealth::Normal {
                    return Err(FrameError::AccentMismatch);
                }
            }
            VisualAccent::Failed => {
                if failed_is_empty(failed) && self.health != BootHealth::Failed {
                    return Err(FrameError::AccentMismatch);
                }
            }
            VisualAccent::Degraded => {
                if degraded_is_empty(degraded) && self.health != BootHealth::Degraded {
                    return Err(FrameError::AccentMismatch);
                }
            }
            VisualAccent::Delay => {
                if delayed_is_empty(delayed) && self.health != BootHealth::Delayed {
                    return Err(FrameError::AccentMismatch);
                }
            }
            VisualAccent::None | VisualAccent::Progress | VisualAccent::Recovery => {}
        }

        Ok(())
    }

    /// Deterministic semantic digest for replay/evidence correlation. This is
    /// not a credential, signature, or authorization token.
    pub fn semantic_digest(&self) -> FrameDigest {
        let mut hasher = blake3::Hasher::new();
        hash_frame(&mut hasher, self);
        FrameDigest(*hasher.finalize().as_bytes())
    }
}

const fn delayed_is_empty(bits: u16) -> bool {
    bits == 0
}
const fn degraded_is_empty(bits: u16) -> bool {
    bits == 0
}
const fn failed_is_empty(bits: u16) -> bool {
    bits == 0
}

/// Convenience owner for the elastic clock. It turns one validated modulation
/// plus elapsed presentation time into one validated renderer-neutral frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentationDriver {
    clock: ElasticVisualClock,
}

impl PresentationDriver {
    pub fn new(policy: VisualClockPolicy) -> Result<Self, FrameError> {
        Ok(Self {
            clock: ElasticVisualClock::new(policy)?,
        })
    }

    pub fn restore(
        anchor: SemanticBootAnchor,
        phase: u32,
        policy: VisualClockPolicy,
    ) -> Result<Self, FrameError> {
        Ok(Self {
            clock: ElasticVisualClock::from_phase(anchor, phase, policy)?,
        })
    }

    pub const fn phase(&self) -> u32 {
        self.clock.phase()
    }

    pub fn reset(&mut self) {
        self.clock.reset();
    }

    pub fn advance_ms(
        &mut self,
        elapsed_ms: u32,
        modulation: &LiveEcologyModulation,
    ) -> Result<EcologyFrameInput, FrameError> {
        let step = self.clock.advance_ms(elapsed_ms, modulation)?;
        EcologyFrameInput::from_step(modulation, step)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameDigest([u8; 32]);

impl FrameDigest {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for FrameDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceDigest {
    pub frame_count: u64,
    pub digest: FrameDigest,
}

impl fmt::Display for TraceDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.frame_count, self.digest)
    }
}

/// Streaming deterministic digest over the sequence of semantic frames. The
/// renderer may include this in replay evidence without retaining raw telemetry.
#[derive(Debug, Clone)]
pub struct SemanticTraceHasher {
    hasher: blake3::Hasher,
    frame_count: u64,
}

impl Default for SemanticTraceHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticTraceHasher {
    pub fn new() -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"spore-ecology-semantic-trace-v1\0");
        hasher.update(&TRACE_SCHEMA_VERSION.to_le_bytes());
        Self {
            hasher,
            frame_count: 0,
        }
    }

    pub fn push(&mut self, frame: &EcologyFrameInput) -> Result<FrameDigest, FrameError> {
        frame.validate()?;
        let digest = frame.semantic_digest();
        self.hasher.update(&self.frame_count.to_le_bytes());
        self.hasher.update(digest.as_bytes());
        self.frame_count = self.frame_count.saturating_add(1);
        Ok(digest)
    }

    pub fn finalize(self) -> TraceDigest {
        TraceDigest {
            frame_count: self.frame_count,
            digest: FrameDigest(*self.hasher.finalize().as_bytes()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    Clock(ClockError),
    UnsupportedVersion(u16),
    RevealFloorMismatch { expected: u32, observed: u32 },
    TruthCeilingMismatch { expected: u32, observed: u32 },
    PhaseOutsideTruthBand { phase: u32, ceiling: u32 },
    OverlappingDomainStates,
    DiagnosticFloorTooLow,
    InvalidHandoffFact,
    ClockModeMismatch,
    AccentMismatch,
}

impl From<ClockError> for FrameError {
    fn from(value: ClockError) -> Self {
        Self::Clock(value)
    }
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(error) => write!(f, "visual clock rejected frame: {error}"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported ecology frame schema version {version}")
            }
            Self::RevealFloorMismatch { expected, observed } => write!(
                f,
                "ecology frame reveal floor mismatch: expected={expected}, observed={observed}"
            ),
            Self::TruthCeilingMismatch { expected, observed } => write!(
                f,
                "ecology frame truth ceiling mismatch: expected={expected}, observed={observed}"
            ),
            Self::PhaseOutsideTruthBand { phase, ceiling } => write!(
                f,
                "ecology frame phase outside truth band: phase={phase}, ceiling={ceiling}"
            ),
            Self::OverlappingDomainStates => write!(f, "ecology frame domain state masks overlap"),
            Self::DiagnosticFloorTooLow => {
                write!(f, "ecology frame diagnostics visibility underreports domain state")
            }
            Self::InvalidHandoffFact => write!(f, "ecology frame carries invalid handoff-ready fact"),
            Self::ClockModeMismatch => write!(f, "ecology frame clock mode contradicts semantic state"),
            Self::AccentMismatch => write!(f, "ecology frame accent contradicts semantic state"),
        }
    }
}

impl std::error::Error for FrameError {}

fn hash_frame(hasher: &mut blake3::Hasher, frame: &EcologyFrameInput) {
    hasher.update(b"spore-ecology-frame-v1\0");
    hasher.update(&frame.schema_version.to_le_bytes());
    hasher.update(&frame.observation_sequence.to_le_bytes());
    hasher.update(&[anchor_tag(frame.anchor)]);
    hasher.update(&[health_tag(frame.health)]);
    hasher.update(&frame.visual_phase.to_le_bytes());
    hasher.update(&frame.reveal_floor.to_le_bytes());
    hasher.update(&frame.truth_ceiling.to_le_bytes());
    hasher.update(&frame.delayed_domains.bits().to_le_bytes());
    hasher.update(&frame.degraded_domains.bits().to_le_bytes());
    hasher.update(&frame.failed_domains.bits().to_le_bytes());
    hasher.update(&[diagnostic_tag(frame.diagnostic_floor)]);
    hasher.update(&frame.accent_token.to_le_bytes());
    hasher.update(&[accent_tag(frame.accent)]);
    hasher.update(&[clock_mode_tag(frame.clock_mode)]);
    hasher.update(&[u8::from(frame.handoff_ready)]);
}

const fn anchor_tag(value: SemanticBootAnchor) -> u8 {
    match value {
        SemanticBootAnchor::KernelPhase => 0,
        SemanticBootAnchor::InitrdPhase => 1,
        SemanticBootAnchor::StoragePhase => 2,
        SemanticBootAnchor::FilesystemsPhase => 3,
        SemanticBootAnchor::SecurityPhase => 4,
        SemanticBootAnchor::NetworkPhase => 5,
        SemanticBootAnchor::ServicesPhase => 6,
        SemanticBootAnchor::GraphicsPhase => 7,
        SemanticBootAnchor::SessionPhase => 8,
        SemanticBootAnchor::SessionReady => 9,
    }
}

const fn health_tag(value: BootHealth) -> u8 {
    match value {
        BootHealth::Normal => 0,
        BootHealth::Delayed => 1,
        BootHealth::Degraded => 2,
        BootHealth::Failed => 3,
        BootHealth::Unknown => 4,
    }
}

const fn diagnostic_tag(value: DiagnosticFloor) -> u8 {
    match value {
        DiagnosticFloor::Ambient => 0,
        DiagnosticFloor::Status => 1,
        DiagnosticFloor::Diagnostics => 2,
    }
}

const fn accent_tag(value: VisualAccent) -> u8 {
    match value {
        VisualAccent::None => 0,
        VisualAccent::Progress => 1,
        VisualAccent::Delay => 2,
        VisualAccent::Degraded => 3,
        VisualAccent::Failed => 4,
        VisualAccent::Recovery => 5,
        VisualAccent::Ready => 6,
    }
}

const fn clock_mode_tag(value: ClockMode) -> u8 {
    match value {
        ClockMode::CatchUp => 0,
        ClockMode::AmbientDrift => 1,
        ClockMode::Hold => 2,
        ClockMode::Complete => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modulation(anchor: SemanticBootAnchor, health: BootHealth) -> LiveEcologyModulation {
        LiveEcologyModulation {
            observation_sequence: 7,
            anchor,
            health,
            reveal_floor: anchor.reveal_floor(),
            delayed_domains: DomainMask::empty(),
            degraded_domains: DomainMask::empty(),
            failed_domains: DomainMask::empty(),
            diagnostic_floor: if health == BootHealth::Normal {
                DiagnosticFloor::Ambient
            } else if health == BootHealth::Delayed {
                DiagnosticFloor::Status
            } else if matches!(health, BootHealth::Degraded | BootHealth::Failed) {
                DiagnosticFloor::Diagnostics
            } else {
                DiagnosticFloor::Ambient
            },
            accent_token: 0,
            accent: VisualAccent::None,
            handoff_ready: anchor == SemanticBootAnchor::SessionReady,
        }
    }

    #[test]
    fn driver_produces_valid_renderer_neutral_frames() {
        let mut driver = PresentationDriver::new(VisualClockPolicy::default()).unwrap();
        let target = modulation(SemanticBootAnchor::ServicesPhase, BootHealth::Normal);
        let frame = driver.advance_ms(16, &target).unwrap();
        frame.validate().unwrap();
        assert!(frame.visual_phase <= frame.truth_ceiling);
        assert_eq!(frame.anchor, SemanticBootAnchor::ServicesPhase);
    }

    #[test]
    fn semantic_digest_is_deterministic_and_phase_sensitive() {
        let target = modulation(SemanticBootAnchor::NetworkPhase, BootHealth::Normal);
        let mut a = PresentationDriver::new(VisualClockPolicy::default()).unwrap();
        let mut b = PresentationDriver::new(VisualClockPolicy::default()).unwrap();
        let frame_a = a.advance_ms(16, &target).unwrap();
        let frame_b = b.advance_ms(16, &target).unwrap();
        assert_eq!(frame_a.semantic_digest(), frame_b.semantic_digest());

        let frame_c = a.advance_ms(16, &target).unwrap();
        assert_ne!(frame_a.semantic_digest(), frame_c.semantic_digest());
    }

    #[test]
    fn trace_digest_is_deterministic_and_order_sensitive() {
        let target = modulation(SemanticBootAnchor::NetworkPhase, BootHealth::Normal);
        let mut driver = PresentationDriver::new(VisualClockPolicy::default()).unwrap();
        let first = driver.advance_ms(16, &target).unwrap();
        let second = driver.advance_ms(33, &target).unwrap();

        let mut left = SemanticTraceHasher::new();
        left.push(&first).unwrap();
        left.push(&second).unwrap();

        let mut right = SemanticTraceHasher::new();
        right.push(&first).unwrap();
        right.push(&second).unwrap();
        assert_eq!(left.clone().finalize(), right.finalize());

        let mut reversed = SemanticTraceHasher::new();
        reversed.push(&second).unwrap();
        reversed.push(&first).unwrap();
        assert_ne!(left.finalize(), reversed.finalize());
    }

    #[test]
    fn ready_accent_requires_known_normal_health() {
        let mut frame = EcologyFrameInput {
            schema_version: FRAME_SCHEMA_VERSION,
            observation_sequence: 9,
            anchor: SemanticBootAnchor::SessionReady,
            health: BootHealth::Unknown,
            visual_phase: REVEAL_SCALE,
            reveal_floor: REVEAL_SCALE,
            truth_ceiling: REVEAL_SCALE,
            delayed_domains: DomainMask::empty(),
            degraded_domains: DomainMask::empty(),
            failed_domains: DomainMask::empty(),
            diagnostic_floor: DiagnosticFloor::Ambient,
            accent_token: 9,
            accent: VisualAccent::Ready,
            clock_mode: ClockMode::Complete,
            handoff_ready: true,
        };
        assert_eq!(frame.validate(), Err(FrameError::AccentMismatch));
        frame.health = BootHealth::Normal;
        frame.validate().unwrap();
    }

    #[test]
    fn overlapping_domain_state_masks_are_rejected() {
        let mut delayed = DomainMask::empty();
        delayed.insert(symthaea_boot_protocol::BootDomain::Network);
        let mut failed = DomainMask::empty();
        failed.insert(symthaea_boot_protocol::BootDomain::Network);

        let frame = EcologyFrameInput {
            schema_version: FRAME_SCHEMA_VERSION,
            observation_sequence: 3,
            anchor: SemanticBootAnchor::NetworkPhase,
            health: BootHealth::Failed,
            visual_phase: SemanticBootAnchor::NetworkPhase.reveal_floor(),
            reveal_floor: SemanticBootAnchor::NetworkPhase.reveal_floor(),
            truth_ceiling: truth_band(SemanticBootAnchor::NetworkPhase).ceiling,
            delayed_domains: delayed,
            degraded_domains: DomainMask::empty(),
            failed_domains: failed,
            diagnostic_floor: DiagnosticFloor::Diagnostics,
            accent_token: 3,
            accent: VisualAccent::Failed,
            clock_mode: ClockMode::Hold,
            handoff_ready: false,
        };
        assert_eq!(frame.validate(), Err(FrameError::OverlappingDomainStates));
    }
}
