// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Passive islanding detection: rate-of-change-of-frequency (ROCOF) and
//! voltage/frequency threshold methods.
//!
//! **Honest limitation, demonstrated rather than hidden**: passive detection
//! has a well-known "non-detection zone" (NDZ) — if a DER's real-power
//! output happens to closely match the local load at the moment of
//! islanding, the post-island frequency shift (governed by the DER's own
//! droop control once it must supply the island alone) can be too small to
//! trip either a ROCOF or a simple threshold relay. This is *why* active
//! anti-islanding schemes exist in real inverters; `test_near_zero_power_mismatch_demonstrates_non_detection_zone`
//! below reproduces the mechanism using this crate's own `FrequencyDroop`
//! model rather than an unverified closed-form NDZ formula.

use crate::droop::FrequencyDroop;
use serde::{Deserialize, Serialize};

/// Rate-of-change-of-frequency detector.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RocofDetector {
    /// Trip threshold, Hz/s. Typical utility-interconnection relay settings
    /// are in the 0.5-1.0 Hz/s range.
    pub threshold_hz_per_s: f64,
}

impl RocofDetector {
    /// `dt_s` must be > 0; returns `false` (cannot compute a rate) otherwise.
    pub fn detect(&self, frequency_hz_prev: f64, frequency_hz_now: f64, dt_s: f64) -> bool {
        if dt_s <= 0.0 {
            return false;
        }
        let rocof = (frequency_hz_now - frequency_hz_prev).abs() / dt_s;
        rocof >= self.threshold_hz_per_s
    }

    /// The computed rate of change, Hz/s (signed). `dt_s` must be > 0.
    pub fn rate_hz_per_s(&self, frequency_hz_prev: f64, frequency_hz_now: f64, dt_s: f64) -> f64 {
        if dt_s <= 0.0 {
            return 0.0;
        }
        (frequency_hz_now - frequency_hz_prev) / dt_s
    }
}

/// Simple over/under voltage and frequency threshold detector.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ThresholdDetector {
    pub min_frequency_hz: f64,
    pub max_frequency_hz: f64,
    pub min_voltage_pu: f64,
    pub max_voltage_pu: f64,
}

impl ThresholdDetector {
    pub fn detect(&self, frequency_hz: f64, voltage_pu: f64) -> bool {
        frequency_hz < self.min_frequency_hz
            || frequency_hz > self.max_frequency_hz
            || voltage_pu < self.min_voltage_pu
            || voltage_pu > self.max_voltage_pu
    }
}

/// Combined passive detector: trips if EITHER criterion trips (standard
/// practice — ORing multiple passive criteria shrinks, but does not
/// eliminate, the non-detection zone).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PassiveIslandingDetector {
    pub rocof: RocofDetector,
    pub threshold: ThresholdDetector,
}

impl PassiveIslandingDetector {
    pub fn detect(
        &self,
        frequency_hz_prev: f64,
        frequency_hz_now: f64,
        dt_s: f64,
        voltage_pu: f64,
    ) -> bool {
        self.rocof.detect(frequency_hz_prev, frequency_hz_now, dt_s)
            || self.threshold.detect(frequency_hz_now, voltage_pu)
    }
}

/// The steady-state frequency a droop-controlled DER settles at once it
/// must alone supply `local_load_kw` after islanding (no more utility to
/// share the load). While grid-tied, the DER's frequency is set by the
/// (stiff) utility regardless of its own droop curve; droop only governs
/// behavior once the DER is grid-forming, i.e. after the tie opens.
pub fn steady_state_frequency_after_islanding(
    freq_droop: &FrequencyDroop,
    local_load_kw: f64,
) -> f64 {
    freq_droop.frequency_for_power(local_load_kw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rocof_detects_large_rate_of_change() {
        let detector = RocofDetector {
            threshold_hz_per_s: 0.5,
        };
        // 60.0 -> 59.0 in 0.02s = 50 Hz/s, far above threshold.
        assert!(detector.detect(60.0, 59.0, 0.02));
    }

    #[test]
    fn test_rocof_does_not_detect_small_rate_of_change() {
        let detector = RocofDetector {
            threshold_hz_per_s: 0.5,
        };
        // 60.0 -> 59.995 in 0.02s = 0.25 Hz/s, below threshold.
        assert!(!detector.detect(60.0, 59.995, 0.02));
    }

    #[test]
    fn test_rocof_boundary_is_inclusive() {
        let detector = RocofDetector {
            threshold_hz_per_s: 0.5,
        };
        // 60.0, 59.5, and 0.5 are all exactly representable in binary
        // floating point (halves of integers), so 0.5/1.0 = 0.5 Hz/s is
        // exact here -- unlike e.g. 60.0 -> 59.99, where 59.99 isn't
        // exactly representable and the computed rate lands fractionally
        // below 0.5 due to rounding, not the intended boundary case.
        assert!(detector.detect(60.0, 59.5, 1.0));
    }

    #[test]
    fn test_rocof_zero_or_negative_dt_never_detects() {
        let detector = RocofDetector {
            threshold_hz_per_s: 0.5,
        };
        assert!(!detector.detect(60.0, 50.0, 0.0));
        assert!(!detector.detect(60.0, 50.0, -1.0));
    }

    #[test]
    fn test_threshold_detector_triggers_on_underfrequency() {
        let detector = ThresholdDetector {
            min_frequency_hz: 59.3,
            max_frequency_hz: 60.5,
            min_voltage_pu: 0.88,
            max_voltage_pu: 1.10,
        };
        assert!(detector.detect(59.0, 1.0));
        assert!(!detector.detect(59.5, 1.0));
    }

    #[test]
    fn test_threshold_detector_triggers_on_overvoltage() {
        let detector = ThresholdDetector {
            min_frequency_hz: 59.3,
            max_frequency_hz: 60.5,
            min_voltage_pu: 0.88,
            max_voltage_pu: 1.10,
        };
        assert!(detector.detect(60.0, 1.15));
        assert!(!detector.detect(60.0, 1.05));
    }

    #[test]
    fn test_large_power_mismatch_is_detected_after_islanding() {
        // DER droop-controlled, was exporting far more than local load
        // needed (grid was absorbing the surplus pre-island). Once islanded,
        // it must throttle back hard to match local load alone -> large
        // frequency excursion -> clearly detected.
        let freq_droop = FrequencyDroop {
            nominal_frequency_hz: 60.0,
            nominal_power_kw: 100.0,
            droop_hz_per_kw: 0.02,
        };
        let local_load_kw = 20.0; // DER was set up to output ~100kW, load is only 20kW
        let post_island_freq = steady_state_frequency_after_islanding(&freq_droop, local_load_kw);

        let detector = PassiveIslandingDetector {
            rocof: RocofDetector {
                threshold_hz_per_s: 0.5,
            },
            threshold: ThresholdDetector {
                min_frequency_hz: 59.3,
                max_frequency_hz: 60.5,
                min_voltage_pu: 0.88,
                max_voltage_pu: 1.10,
            },
        };
        // Simulate a fast transition (20ms) from nominal to the new
        // post-island steady state.
        let detected = detector.detect(60.0, post_island_freq, 0.02, 1.0);
        assert!(
            detected,
            "post-island frequency {post_island_freq} Hz should have been detected"
        );
    }

    /// The real, well-documented non-detection zone: DG output already
    /// matches local load almost exactly at the moment of islanding, so the
    /// droop-governed post-island frequency barely moves from nominal.
    /// Neither ROCOF nor a simple threshold relay trips. This is not a gap
    /// in this crate's detector logic — it is the actual physical
    /// limitation of passive islanding detection that motivates active
    /// detection schemes in real inverters.
    #[test]
    fn test_near_zero_power_mismatch_demonstrates_non_detection_zone() {
        let freq_droop = FrequencyDroop {
            nominal_frequency_hz: 60.0,
            nominal_power_kw: 100.0,
            droop_hz_per_kw: 0.001, // stiff (small) droop gain, typical of a well-tuned inverter
        };
        let local_load_kw = 100.05; // essentially matches nominal DG output
        let post_island_freq = steady_state_frequency_after_islanding(&freq_droop, local_load_kw);

        let detector = PassiveIslandingDetector {
            rocof: RocofDetector {
                threshold_hz_per_s: 0.5,
            },
            threshold: ThresholdDetector {
                min_frequency_hz: 59.3,
                max_frequency_hz: 60.5,
                min_voltage_pu: 0.88,
                max_voltage_pu: 1.10,
            },
        };
        let detected = detector.detect(60.0, post_island_freq, 0.02, 1.0);
        assert!(
            !detected,
            "expected the classic non-detection zone: post-island frequency {post_island_freq} Hz \
             should NOT trip a passive relay when generation nearly matches local load"
        );
    }
}
