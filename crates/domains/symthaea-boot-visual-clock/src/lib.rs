// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic elastic timing for Spore boot presentation.
//!
//! The clock may interpolate only inside the visual band earned by an already-
//! validated boot modulation. It cannot create a new semantic boot anchor and it
//! carries no boot/session authority.

#![forbid(unsafe_code)]

use symthaea_boot_ecology_live::{
    DiagnosticFloor, DomainMask, LiveEcologyModulation, REVEAL_SCALE, SemanticBootAnchor,
    VisualAccent,
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
    let band = match anchor {
        SemanticBootAnchor::KernelPhase => TruthBand {
            floor: 50_000,
            ceiling: 110_000,
        },
        SemanticBootAnchor::InitrdPhase => TruthBand {
            floor: 120_000,
            ceiling: 210_000,
        },
        SemanticBootAnchor::StoragePhase => TruthBand {
            floor: 220_000,
            ceiling: 325_000,
        },
        SemanticBootAnchor::FilesystemsPhase => TruthBand {
            floor: 340_000,
            ceiling: 430_000,
        },
        SemanticBootAnchor::SecurityPhase => TruthBand {
            floor: 450_000,
            ceiling: 540_000,
        },
        SemanticBootAnchor::NetworkPhase => TruthBand {
            floor: 560_000,
            ceiling: 650_000,
        },
        SemanticBootAnchor::ServicesPhase => TruthBand {
            floor: 680_000,
            ceiling: 790_000,
        },
        SemanticBootAnchor::GraphicsPhase => TruthBand {
            floor: 820_000,
            ceiling: 900_000,
        },
        SemanticBootAnchor::SessionPhase => TruthBand {
            floor: 920_000,
            ceiling: 975_000,
        },
        SemanticBootAnchor::SessionReady => TruthBand {
            floor: REVEAL_SCALE,
            ceiling: REVEAL_SCALE,
        },
    };
    debug_assert!(band.validate());
    band
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
pub struct ElasticVisualClock {
    phase: u32,
    policy: VisualClockPolicy,
}

impl Default for ElasticVisualClock {
    fn default() -> Self {
        Self::new(VisualClockPolicy::default())
    }
}

impl ElasticVisualClock {
    pub fn new(policy: VisualClockPolicy) -> Self {
        assert!(policy.validate(), "invalid Spore visual clock policy");
        Self { phase: 0, policy }
    }

    pub fn from_phase(phase: u32, policy: VisualClockPolicy) -> Self {
        assert!(policy.validate(), "invalid Spore visual clock policy");
        Self {
            phase: phase.min(REVEAL_SCALE),
            policy,
        }
    }

    pub const fn phase(&self) -> u32 {
        self.phase
    }

    pub fn reset(&mut self) {
        self.phase = 0;
    }

    /// Advance one presentation step. The returned phase is monotonic and can
    /// never cross the current anchor's truthful decorative ceiling.
    pub fn advance_ms(
        &mut self,
        elapsed_ms: u32,
        modulation: &LiveEcologyModulation,
    ) -> ClockAdvance {
        debug_assert!(modulation.validate());
        let before = self.phase;
        let elapsed_ms = elapsed_ms.min(self.policy.max_step_ms);
        let band = truth_band(modulation.anchor);
        debug_assert_eq!(band.floor, modulation.reveal_floor);

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

        debug_assert!(self.phase >= before);
        debug_assert!(self.phase <= band.ceiling);
        ClockAdvance {
            before,
            after: self.phase,
            mode,
        }
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
        let mut clock = ElasticVisualClock::new(policy);
        let target = modulation(SemanticBootAnchor::GraphicsPhase, BootHealth::Normal);

        let mut previous = 0;
        for _ in 0..20 {
            let step = clock.advance_ms(16, &target);
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
        let mut clock = ElasticVisualClock::from_phase(target.reveal_floor, VisualClockPolicy::default());

        for _ in 0..1_000 {
            clock.advance_ms(50, &target);
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
            let mut clock = ElasticVisualClock::from_phase(target.reveal_floor, VisualClockPolicy::default());
            let step = clock.advance_ms(250, &target);
            assert_eq!(step.mode, ClockMode::Hold);
            assert_eq!(step.before, step.after);
        }
    }

    #[test]
    fn scheduling_gap_is_bounded_by_policy() {
        let policy = VisualClockPolicy::default();
        let target = modulation(SemanticBootAnchor::SessionPhase, BootHealth::Normal);
        let mut long_gap = ElasticVisualClock::new(policy);
        let mut bounded = ElasticVisualClock::new(policy);

        let long = long_gap.advance_ms(10_000, &target);
        let short = bounded.advance_ms(policy.max_step_ms, &target);
        assert_eq!(long.after, short.after);
    }

    #[test]
    fn ready_reaches_complete_without_conferring_host_authority() {
        let target = modulation(SemanticBootAnchor::SessionReady, BootHealth::Normal);
        let mut clock = ElasticVisualClock::from_phase(950_000, VisualClockPolicy::default());
        for _ in 0..10 {
            clock.advance_ms(50, &target);
            if clock.phase() == REVEAL_SCALE {
                break;
            }
        }
        assert_eq!(clock.phase(), REVEAL_SCALE);
        let final_step = clock.advance_ms(16, &target);
        assert_eq!(final_step.mode, ClockMode::Complete);
        assert!(target.handoff_ready);
        // `handoff_ready` remains a presentation fact; host lifecycle policy is
        // intentionally outside this crate.
    }
}
