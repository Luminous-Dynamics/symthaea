// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic environmental perturbation schedules for evolvability experiments.
//!
//! A perturbation schedule is deliberately external to [`crate::Population`]. It transforms a
//! caller-supplied baseline resource signal without changing organism, reproduction, mutation,
//! or environment semantics. This keeps stress-test design separable from the substrate being
//! tested and makes A/B replay straightforward.
//!
//! Overlapping windows compose in an order-independent way:
//!
//! `perturbed = baseline * product(active multipliers) + sum(active deltas)`
//!
//! followed by one clamp to `[0, 1]`. Deltas are therefore never accidentally multiplied by
//! whichever window happened to be inserted first.

/// One bounded resource perturbation window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourcePerturbation {
    start_tick: u64,
    end_tick_exclusive: u64,
    multiplier: f64,
    delta: f64,
}

impl ResourcePerturbation {
    /// Construct a resource perturbation active for exactly `duration_ticks` ticks.
    pub fn new(
        start_tick: u64,
        duration_ticks: u64,
        multiplier: f64,
        delta: f64,
    ) -> Result<Self, PerturbationError> {
        if duration_ticks == 0 {
            return Err(PerturbationError::ZeroDuration { start_tick });
        }
        if !multiplier.is_finite() || multiplier < 0.0 {
            return Err(PerturbationError::InvalidMultiplier { multiplier });
        }
        if !delta.is_finite() {
            return Err(PerturbationError::InvalidDelta { delta });
        }
        let end_tick_exclusive = start_tick
            .checked_add(duration_ticks)
            .ok_or(PerturbationError::TickOverflow {
                start_tick,
                duration_ticks,
            })?;
        Ok(Self {
            start_tick,
            end_tick_exclusive,
            multiplier,
            delta,
        })
    }

    pub fn start_tick(self) -> u64 {
        self.start_tick
    }

    pub fn end_tick_exclusive(self) -> u64 {
        self.end_tick_exclusive
    }

    pub fn duration_ticks(self) -> u64 {
        self.end_tick_exclusive - self.start_tick
    }

    pub fn multiplier(self) -> f64 {
        self.multiplier
    }

    pub fn delta(self) -> f64 {
        self.delta
    }

    pub fn is_active(self, tick: u64) -> bool {
        self.start_tick <= tick && tick < self.end_tick_exclusive
    }
}

/// A deterministic collection of perturbations.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PerturbationSchedule {
    windows: Vec<ResourcePerturbation>,
}

impl PerturbationSchedule {
    /// Normalize windows into a deterministic order. Runtime composition is commutative, but the
    /// normalization also guarantees stable debug/evidence output for equivalent schedules.
    pub fn new(mut windows: Vec<ResourcePerturbation>) -> Self {
        windows.sort_by(|a, b| {
            a.start_tick
                .cmp(&b.start_tick)
                .then_with(|| a.end_tick_exclusive.cmp(&b.end_tick_exclusive))
                .then_with(|| a.multiplier.total_cmp(&b.multiplier))
                .then_with(|| a.delta.total_cmp(&b.delta))
        });
        Self { windows }
    }

    pub fn windows(&self) -> &[ResourcePerturbation] {
        &self.windows
    }

    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    pub fn active_count(&self, tick: u64) -> usize {
        self.windows
            .iter()
            .filter(|window| window.is_active(tick))
            .count()
    }

    /// Transform one baseline resource observation through all perturbations active at `tick`.
    pub fn apply_resource(&self, tick: u64, baseline: f64) -> Result<f64, PerturbationError> {
        if !baseline.is_finite() {
            return Err(PerturbationError::InvalidBaseline { baseline });
        }

        let mut multiplier = 1.0;
        let mut delta = 0.0;
        for window in self.windows.iter().filter(|window| window.is_active(tick)) {
            multiplier *= window.multiplier;
            delta += window.delta;
            if !multiplier.is_finite() || !delta.is_finite() {
                return Err(PerturbationError::NonFiniteComposition { tick });
            }
        }

        let perturbed = baseline * multiplier + delta;
        if !perturbed.is_finite() {
            return Err(PerturbationError::NonFiniteComposition { tick });
        }
        Ok(perturbed.clamp(0.0, 1.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerturbationError {
    ZeroDuration {
        start_tick: u64,
    },
    TickOverflow {
        start_tick: u64,
        duration_ticks: u64,
    },
    InvalidMultiplier {
        multiplier: f64,
    },
    InvalidDelta {
        delta: f64,
    },
    InvalidBaseline {
        baseline: f64,
    },
    NonFiniteComposition {
        tick: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_window_is_half_open_and_exact_duration() {
        let shock = ResourcePerturbation::new(10, 3, 0.5, 0.0).expect("valid shock");
        assert!(!shock.is_active(9));
        assert!(shock.is_active(10));
        assert!(shock.is_active(11));
        assert!(shock.is_active(12));
        assert!(!shock.is_active(13));
        assert_eq!(shock.duration_ticks(), 3);
    }

    #[test]
    fn outside_all_windows_the_baseline_is_unchanged_except_domain_clamp() {
        let schedule = PerturbationSchedule::new(vec![
            ResourcePerturbation::new(10, 5, 0.5, -0.1).expect("window"),
        ]);
        assert_eq!(schedule.apply_resource(0, 0.6).expect("baseline"), 0.6);
        assert_eq!(schedule.apply_resource(0, 1.4).expect("clamped baseline"), 1.0);
    }

    #[test]
    fn overlapping_windows_compose_order_independently() {
        let a = ResourcePerturbation::new(5, 10, 0.5, -0.1).expect("a");
        let b = ResourcePerturbation::new(7, 4, 0.8, 0.05).expect("b");
        let ab = PerturbationSchedule::new(vec![a, b]);
        let ba = PerturbationSchedule::new(vec![b, a]);

        for tick in 0..20 {
            assert_eq!(
                ab.apply_resource(tick, 0.75).expect("ab"),
                ba.apply_resource(tick, 0.75).expect("ba")
            );
        }

        let expected = (0.75_f64 * 0.5 * 0.8 - 0.1 + 0.05).clamp(0.0, 1.0);
        assert!((ab.apply_resource(8, 0.75).expect("overlap") - expected).abs() < 1e-12);
        assert_eq!(ab.active_count(8), 2);
    }

    #[test]
    fn equivalent_input_order_normalizes_to_one_schedule_identity() {
        let a = ResourcePerturbation::new(20, 2, 0.9, 0.0).expect("a");
        let b = ResourcePerturbation::new(5, 4, 0.7, -0.05).expect("b");
        assert_eq!(
            PerturbationSchedule::new(vec![a, b]),
            PerturbationSchedule::new(vec![b, a])
        );
    }

    #[test]
    fn invalid_windows_fail_at_construction() {
        assert!(matches!(
            ResourcePerturbation::new(0, 0, 1.0, 0.0),
            Err(PerturbationError::ZeroDuration { .. })
        ));
        assert!(matches!(
            ResourcePerturbation::new(0, 1, -1.0, 0.0),
            Err(PerturbationError::InvalidMultiplier { .. })
        ));
        assert!(matches!(
            ResourcePerturbation::new(u64::MAX, 1, 1.0, 0.0),
            Err(PerturbationError::TickOverflow { .. })
        ));
    }

    #[test]
    fn non_finite_baseline_fails_closed() {
        let schedule = PerturbationSchedule::default();
        assert!(matches!(
            schedule.apply_resource(0, f64::NAN),
            Err(PerturbationError::InvalidBaseline { .. })
        ));
    }
}
