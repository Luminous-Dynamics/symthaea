// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Voltage ride-through / trip envelope, modeling the abnormal-voltage
//! must-trip behavior required of grid-tied DER inverters.
//!
//! **For simulation only, not certification.** This models the *shape* of
//! IEEE 1547-2018's abnormal-voltage ride-through requirement (a continuous-
//! operation band around nominal voltage, with progressively shorter
//! mandatory clearing times as voltage deviates further from nominal) but
//! the specific band edges and clearing times below are simplified,
//! illustrative round numbers — they have NOT been checked against the
//! published standard's exact Category I/II/III tables. Do not use this for
//! interconnection compliance testing; use it for simulating microgrid
//! disturbance response and islanding scenarios.

use serde::{Deserialize, Serialize};

/// One abnormal-voltage band: `[min_pu, max_pu)`, with a maximum ride-through
/// duration before the DER is required to trip.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VoltageBand {
    pub min_pu: f64,
    pub max_pu: f64,
    pub max_clearing_time_s: f64,
}

/// A full voltage trip envelope: a continuous-operation region (indefinite
/// ride-through) plus abnormal bands on either side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoltageTripEnvelope {
    pub continuous_operation_min_pu: f64,
    pub continuous_operation_max_pu: f64,
    /// Abnormal bands, outside the continuous-operation region. Order does
    /// not matter — `max_ride_through_seconds` scans all of them.
    pub bands: Vec<VoltageBand>,
}

impl VoltageTripEnvelope {
    /// Illustrative default envelope in the shape of IEEE 1547-2018's
    /// Category II/III abnormal-voltage ride-through — see module doc for
    /// the "simulation only" caveat on the specific numbers.
    pub fn illustrative_default() -> Self {
        Self {
            continuous_operation_min_pu: 0.917,
            continuous_operation_max_pu: 1.05,
            bands: vec![
                // Undervoltage, most to least severe.
                VoltageBand {
                    min_pu: f64::NEG_INFINITY,
                    max_pu: 0.45,
                    max_clearing_time_s: 0.16,
                },
                VoltageBand {
                    min_pu: 0.45,
                    max_pu: 0.60,
                    max_clearing_time_s: 1.0,
                },
                VoltageBand {
                    min_pu: 0.60,
                    max_pu: 0.917,
                    max_clearing_time_s: 10.0,
                },
                // Overvoltage, least to most severe.
                VoltageBand {
                    min_pu: 1.05,
                    max_pu: 1.10,
                    max_clearing_time_s: 5.0,
                },
                VoltageBand {
                    min_pu: 1.10,
                    max_pu: 1.20,
                    max_clearing_time_s: 1.0,
                },
                VoltageBand {
                    min_pu: 1.20,
                    max_pu: f64::INFINITY,
                    max_clearing_time_s: 0.16,
                },
            ],
        }
    }

    /// Whether `voltage_pu` is within the continuous-operation region
    /// (indefinite ride-through, no mandatory trip).
    pub fn is_continuous_operation(&self, voltage_pu: f64) -> bool {
        voltage_pu >= self.continuous_operation_min_pu
            && voltage_pu <= self.continuous_operation_max_pu
    }

    /// Maximum ride-through duration (seconds) for `voltage_pu`, or `None`
    /// if it's within the continuous-operation region (indefinite).
    pub fn max_ride_through_seconds(&self, voltage_pu: f64) -> Option<f64> {
        if self.is_continuous_operation(voltage_pu) {
            return None;
        }
        self.bands
            .iter()
            .find(|band| voltage_pu >= band.min_pu && voltage_pu < band.max_pu)
            .map(|band| band.max_clearing_time_s)
    }

    /// Whether a DER that has been at `voltage_pu` for `elapsed_seconds`
    /// (continuously, within the same abnormal condition) is now required
    /// to trip.
    pub fn should_trip(&self, voltage_pu: f64, elapsed_seconds: f64) -> bool {
        match self.max_ride_through_seconds(voltage_pu) {
            None => false,
            Some(max_clearing_time_s) => elapsed_seconds >= max_clearing_time_s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nominal_voltage_never_trips() {
        let envelope = VoltageTripEnvelope::illustrative_default();
        assert!(envelope.is_continuous_operation(1.0));
        assert!(!envelope.should_trip(1.0, 1_000_000.0));
    }

    #[test]
    fn test_severe_undervoltage_trips_after_clearing_time() {
        let envelope = VoltageTripEnvelope::illustrative_default();
        assert!(!envelope.should_trip(0.3, 0.1));
        assert!(envelope.should_trip(0.3, 0.2));
    }

    #[test]
    fn test_moderate_overvoltage_has_longer_ride_through_than_severe() {
        let envelope = VoltageTripEnvelope::illustrative_default();
        // 1.15 pu (moderate) must ride through longer than 1.25 pu (severe).
        let moderate = envelope.max_ride_through_seconds(1.15).unwrap();
        let severe = envelope.max_ride_through_seconds(1.25).unwrap();
        assert!(moderate > severe, "moderate={moderate}s severe={severe}s");

        assert!(!envelope.should_trip(1.15, moderate - 0.1));
        assert!(envelope.should_trip(1.15, moderate + 0.1));
    }

    #[test]
    fn test_continuous_operation_boundary_is_inclusive() {
        let envelope = VoltageTripEnvelope::illustrative_default();
        assert!(envelope.is_continuous_operation(0.917));
        assert!(envelope.is_continuous_operation(1.05));
        assert!(!envelope.is_continuous_operation(0.9169999));
        assert!(!envelope.is_continuous_operation(1.0500001));
    }

    #[test]
    fn test_severity_monotonic_further_from_nominal_undervoltage() {
        let envelope = VoltageTripEnvelope::illustrative_default();
        let near = envelope.max_ride_through_seconds(0.70).unwrap();
        let far = envelope.max_ride_through_seconds(0.50).unwrap();
        let extreme = envelope.max_ride_through_seconds(0.20).unwrap();
        assert!(
            near > far && far > extreme,
            "expected monotonically shorter ride-through further from nominal: near={near} far={far} extreme={extreme}"
        );
    }

    #[test]
    fn test_voltage_outside_all_bands_has_no_ride_through_limit_only_if_gap_exists() {
        // Sanity check: the illustrative envelope's bands plus continuous
        // region should cover the entire real line with no gaps, so every
        // voltage has a defined answer (continuous, or some max ride-through).
        let envelope = VoltageTripEnvelope::illustrative_default();
        for millivolts_pu in -500..2500 {
            let v = millivolts_pu as f64 / 1000.0;
            let continuous = envelope.is_continuous_operation(v);
            let has_band = envelope.max_ride_through_seconds(v).is_some();
            assert!(
                continuous || has_band,
                "voltage {v} pu has neither continuous-operation nor a matching band"
            );
        }
    }
}
