// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic elastic timing for Spore boot presentation.
//!
//! The clock may interpolate only inside the visual band earned by an already-
//! validated boot modulation. It cannot create a new semantic boot anchor and it
//! carries no boot/session authority.

#![forbid(unsafe_code)]

use symthaea_boot_ecology_live::{
    DiagnosticFloor, LiveEcologyModulation, REVEAL_SCALE, SemanticBootAnchor,
};
use symthaea_boot_protocol::BootHealth;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruthBand {
    pub floor: u32,
    pub ceiling: u32,
}

impl TruthBand {
    pub const fn validate(self) -> bool {
        self.floor <= self.ceiling && self.ceiling <= REVEAL_SCALE
    }
}

/// Visual room available after the factual phase floor has been reached but
/// before the next unearned factual phase would be implied.
pub const fn truth_band(anchor: SemanticBootAnchor) -> TruthBand {
    let ceiling = match anchor {
        SemanticBootAnchor::KernelPhase => 110_000,
        SemanticBootAnchor::InitrdPhase => 210_000,
        SemanticBootAnchor::StoragePhase => 325_000,
        SemanticBootAnchor::FilesystemsPhase => 430_000,
        SemanticBootAnchor::SecurityPhase => 540_000,
        SemanticBootAnchor::NetworkPhase => 650_000,
        SemanticBootAnchor::ServicesPhase => 790_000,
        SemanticBootAnchor::GraphicsPhase => 900_000,
        SemanticBootAnchor::SessionPhase => 975_000,
        SemanticBootAnchor::SessionReady => REVEAL_SCALE,
    };
    TruthBand {
        floor: anchor.reveal_floor(),
        ceiling,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisualClockPolicy {
    /// Fixed-point phase units per second used when factual state is ahead of
    /// rendered state. This is presentation velocity, not Linux boot progress.
    pub catch_up_per_second: u32,
    /// Fixed-point phase units per second allowed for calm decorative motion
    /// within one already-earned truth band.
    pub ambient_per_second: u32,
    /// Bound a single scheduling step so a paused renderer cannot consume an
    /// entire sequence from one giant elapsed-time sample.
    pub max_step_ms: u32,
}

impl Default for VisualClockPolicy {
    fn default() -> Self {
        Self {
            catch_up_per_second: 1_500_000,
            ambient_per_second: 40_000,
            max_step_ms: 250,
        }
    }
}

impl VisualClockPolicy {
    pub const fn validate(self) -> bool {
        self.catch_up_per_second > 0
            && self.max_step_ms > 0
            && self.ambient_per_second <= self.catch_up_per_second
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockMode {
    CatchUp,
    AmbientDrift,
    Hold,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockAdvance {
    pub before: u32,
    pub after: u32,
    pub mode: ClockMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockError {
    InvalidPolicy,
    InvalidModulation,
    RevealFloorMismatch {
        expected: u32,
        observed: u32,
    },
    AnchorRegressed {
        previous: SemanticBootAnchor,
        observed: SemanticBootAnchor,
    },
    PhaseBeyondTruthBand {
        phase: u32,
        ceiling: u32,
    },
}

impl std::fmt::Display for ClockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPolicy => write!(f, "invalid Spore visual clock policy"),
            Self::InvalidModulation => write!(f, "invalid live ecology modulation"),
            Self::RevealFloorMismatch { expected, observed } => write!(
                f,
                "visual reveal floor mismatch: expected={expected}, observed={observed}"
            ),
            Self::AnchorRegressed { previous, observed } => write!(
                f,
                "visual clock anchor regressed: previous={previous:?}, observed={observed:?}"
            ),
            Self::PhaseBeyondTruthBand { phase, ceiling } => write!(
                f,
                "visual phase exceeds current truth band: phase={phase}, ceiling={ceiling}"
            ),
        }
    }
}

impl std::error::Error for ClockError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElasticVisualClock {
    phase: u32,
    policy: VisualClockPolicy,
    last_anchor: Option<SemanticBootAnchor>,
}

impl Default for ElasticVisualClock {
    fn default() -> Self {
        Self::new(VisualClockPolicy::default()).expect("default visual clock policy is valid")
    }
}

impl ElasticVisualClock {
    pub fn new(policy: VisualClockPolicy) -> Result<Self, ClockError> {
        if !policy.validate() {
            return Err(ClockError::InvalidPolicy);
        }
        Ok(Self {
            phase: 0,
            policy,
            last_anchor: None,
        })
    }

    pub fn from_phase(
        anchor: SemanticBootAnchor,
        phase: u32,
        policy: VisualClockPolicy,
    ) -> Result<Self, ClockError> {
        if !policy.validate() {
            return Err(ClockError::InvalidPolicy);
        }
        let band = truth_band(anchor);
        if phase > band.ceiling {
            return Err(ClockError::PhaseBeyondTruthBand {
                phase,
                ceiling: band.ceiling,
            });
        }
        Ok(Self {
            phase,
            policy,
            last_anchor: Some(anchor),
        })
    }

    pub const fn phase(&self) -> u32 {
        self.phase
    }

    pub fn reset(&mut self) {
        self.phase = 0;
        self.last_anchor = None;
    }

    /// Advance one presentation step. The returned phase is monotonic and can
    /// never cross the current anchor's truthful decorative ceiling.
    pub fn advance_ms(
        &mut self,
        elapsed_ms: u32,
        modulation: &LiveEcologyModulation,
    ) -> Result<ClockAdvance, ClockError> {
        if !modulation.validate() {
            return Err(ClockError::InvalidModulation);
        }

        let band = truth_band(modulation.anchor);
        if band.floor != modulation.reveal_floor {
            return Err(ClockError::RevealFloorMismatch {
                expected: band.floor,
                observed: modulation.reveal_floor,
            });
        }
        if let Some(previous) = self.last_anchor
            && modulation.anchor < previous
        {
            return Err(ClockError::AnchorRegressed {
                previous,
                observed: modulation.anchor,
            });
        }
        if self.phase > band.ceiling {
            return Err(ClockError::PhaseBeyondTruthBand {
                phase: self.phase,
                ceiling: band.ceiling,
            });
        }

        let before = self.phase;
        let elapsed_ms = elapsed_ms.min(self.policy.max_step_ms);

        let mode = if self.phase < band.floor {
            self.phase = advance_toward(
                self.phase,
                band.floor,
                phase_step(self.policy.catch_up_per_second, elapsed_ms),
            );
            ClockMode::CatchUp
        } else if self.phase < band.ceiling && may_decoratively_advance(modulation) {
            self.phase = advance_toward(
                self.phase,
                band.ceiling,
                phase_step(self.policy.ambient_per_second, elapsed_ms),
            );
            ClockMode::AmbientDrift
        } else if self.phase == REVEAL_SCALE
            && modulation.anchor == SemanticBootAnchor::SessionReady
        {
            ClockMode::Complete
        } else {
            ClockMode::Hold
        };

        self.last_anchor = Some(modulation.anchor);
        debug_assert!(self.phase >= before);
        debug_assert!(self.phase <= band.ceiling);
        Ok(ClockAdvance {
            before,
            after: self.phase,
            mode,
        })
    }
}

const fn may_decoratively_advance(modulation: &LiveEcologyModulation) -> bool {
    matches!(modulation.health, BootHealth::Normal)
        && matches!(modulation.diagnostic_floor, DiagnosticFloor::Ambient)
}

const fn phase_step(rate_per_second: u32, elapsed_ms: u32) -> u32 {
    (((rate_per_second as u64) * (elapsed_ms as u64)) / 1_000).min(u32::MAX as u64) as u32
}

const fn advance_toward(current: u32, target: u32, step: u32) -> u32 {
    current.saturating_add(step).min(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_boot_ecology_live::{DomainMask, VisualAccent};

    fn modulation(anchor: SemanticBootAnchor, health: BootHealth) -> LiveEcologyModulation {
        LiveEcologyModulation {
            observation_sequence: 1,
            anchor,
            health,
            reveal_floor: anchor.reveal_floor(),
            delayed_domains: DomainMask::empty(),
            degraded_domains: DomainMask::empty(),
            failed_domains: DomainMask::empty(),
            diagnostic_floor: match health {
                BootHealth::Normal | BootHealth::Unknown => DiagnosticFloor::Ambient,
                BootHealth::Delayed => DiagnosticFloor::Status,
                BootHealth::Degraded | BootHealth::Failed => DiagnosticFloor::Diagnostics,
            },
            accent_token: 0,
            accent: VisualAccent::None,
            handoff_ready: anchor == SemanticBootAnchor::SessionReady,
        }
    }

    #[test]
    fn truth_bands_match_reveal_floors_and_do_not_cross_next_anchor() {
        let anchors = [
            SemanticBootAnchor::KernelPhase,
            SemanticBootAnchor::InitrdPhase,
            SemanticBootAnchor::StoragePhase,
            SemanticBootAnchor::FilesystemsPhase,
            SemanticBootAnchor::SecurityPhase,
            SemanticBootAnchor::NetworkPhase,
            SemanticBootAnchor::ServicesPhase,
            SemanticBootAnchor::GraphicsPhase,
            SemanticBootAnchor::SessionPhase,
            SemanticBootAnchor::SessionReady,
        ];

        for (index, anchor) in anchors.iter().copied().enumerate() {
            let band = truth_band(anchor);
            assert!(band.validate());
            assert_eq!(band.floor, anchor.reveal_floor());
            if let Some(next) = anchors.get(index + 1) {
                assert!(band.ceiling < next.reveal_floor());
            }
        }
    }

    #[test]
    fn factual_jump_catches_up_monotonically_without_overshoot() {
        let policy = VisualClockPolicy::default();
        let mut clock = ElasticVisualClock::new(policy).unwrap();
        let target = modulation(SemanticBootAnchor::GraphicsPhase, BootHealth::Normal);

        let mut previous = 0;
        for _ in 0..64 {
            let step = clock.advance_ms(16, &target).unwrap();
            assert!(step.after >= previous);
            assert!(step.after <= target.reveal_floor);
            previous = step.after;
            if clock.phase() == target.reveal_floor {
                break;
            }
        }
        assert_eq!(clock.phase(), target.reveal_floor);
    }

    #[test]
    fn normal_slow_boot_may_drift_only_inside_truth_band() {
        let anchor = SemanticBootAnchor::NetworkPhase;
        let target = modulation(anchor, BootHealth::Normal);
        let band = truth_band(anchor);
        let mut clock = ElasticVisualClock::from_phase(
            anchor,
            target.reveal_floor,
            VisualClockPolicy::default(),
        )
        .unwrap();

        for _ in 0..1_000 {
            clock.advance_ms(50, &target).unwrap();
        }
        assert_eq!(clock.phase(), band.ceiling);
    }

    #[test]
    fn delayed_degraded_failed_and_unknown_do_not_decoratively_advance() {
        for health in [
            BootHealth::Unknown,
            BootHealth::Delayed,
            BootHealth::Degraded,
            BootHealth::Failed,
        ] {
            let anchor = SemanticBootAnchor::ServicesPhase;
            let target = modulation(anchor, health);
            let mut clock = ElasticVisualClock::from_phase(
                anchor,
                target.reveal_floor,
                VisualClockPolicy::default(),
            )
            .unwrap();
            let step = clock.advance_ms(250, &target).unwrap();
            assert_eq!(step.mode, ClockMode::Hold);
            assert_eq!(step.before, step.after);
        }
    }

    #[test]
    fn scheduling_gap_is_bounded_by_policy() {
        let policy = VisualClockPolicy::default();
        let target = modulation(SemanticBootAnchor::SessionPhase, BootHealth::Normal);
        let mut long_gap = ElasticVisualClock::new(policy).unwrap();
        let mut bounded = ElasticVisualClock::new(policy).unwrap();

        let long = long_gap.advance_ms(10_000, &target).unwrap();
        let short = bounded.advance_ms(policy.max_step_ms, &target).unwrap();
        assert_eq!(long.after, short.after);
    }

    #[test]
    fn ready_reaches_complete_without_conferring_host_authority() {
        let target = modulation(SemanticBootAnchor::SessionReady, BootHealth::Normal);
        let mut clock = ElasticVisualClock::from_phase(
            SemanticBootAnchor::SessionReady,
            950_000,
            VisualClockPolicy::default(),
        )
        .unwrap();
        for _ in 0..10 {
            clock.advance_ms(50, &target).unwrap();
            if clock.phase() == REVEAL_SCALE {
                break;
            }
        }
        assert_eq!(clock.phase(), REVEAL_SCALE);
        let final_step = clock.advance_ms(16, &target).unwrap();
        assert_eq!(final_step.mode, ClockMode::Complete);
        assert!(target.handoff_ready);
        // `handoff_ready` remains a presentation fact; host lifecycle policy is
        // intentionally outside this crate.
    }

    #[test]
    fn clock_rejects_regressed_or_mismatched_modulation() {
        let mut clock = ElasticVisualClock::new(VisualClockPolicy::default()).unwrap();
        let services = modulation(SemanticBootAnchor::ServicesPhase, BootHealth::Normal);
        clock.advance_ms(16, &services).unwrap();

        let network = modulation(SemanticBootAnchor::NetworkPhase, BootHealth::Normal);
        assert!(matches!(
            clock.advance_ms(16, &network),
            Err(ClockError::AnchorRegressed { .. })
        ));

        let mut mismatched = services;
        mismatched.reveal_floor += 1;
        assert!(matches!(
            clock.advance_ms(16, &mismatched),
            Err(ClockError::RevealFloorMismatch { .. })
        ));
    }

    #[test]
    fn restored_phase_must_fit_declared_truth_band() {
        let anchor = SemanticBootAnchor::NetworkPhase;
        let band = truth_band(anchor);
        assert!(matches!(
            ElasticVisualClock::from_phase(
                anchor,
                band.ceiling + 1,
                VisualClockPolicy::default()
            ),
            Err(ClockError::PhaseBeyondTruthBand { .. })
        ));
    }
}
