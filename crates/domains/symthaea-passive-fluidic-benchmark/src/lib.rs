// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Paired benchmark evaluation for fixed-geometry passive fluidic devices.
//!
//! This crate does not simulate rectification. It evaluates paired results
//! produced by an external CFD, analytical, or measured backend and makes the
//! comparison deterministic and auditable.

/// One operating point for one flow direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalResult {
    /// Absolute pressure drop across the device (Pa).
    pub pressure_drop_pa: f64,
    /// Signed or unsigned volumetric flow rate; comparison uses magnitude.
    pub flow_rate_m3_s: f64,
}

impl DirectionalResult {
    pub fn validate(&self) -> bool {
        self.pressure_drop_pa.is_finite()
            && self.pressure_drop_pa > 0.0
            && self.flow_rate_m3_s.is_finite()
            && self.flow_rate_m3_s != 0.0
    }
}

/// Paired forward/reverse observation from the same fixed geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectificationObservation {
    pub forward: DirectionalResult,
    pub reverse: DirectionalResult,
}

impl RectificationObservation {
    pub fn validate(&self, relative_flow_tolerance: f64) -> bool {
        self.forward.validate()
            && self.reverse.validate()
            && relative_flow_mismatch(self.forward.flow_rate_m3_s, self.reverse.flow_rate_m3_s)
                <= relative_flow_tolerance.max(0.0)
    }

    /// Diodicity proxy: reverse pressure drop divided by forward pressure drop.
    ///
    /// Values above 1.0 indicate greater reverse pressure loss under the paired
    /// operating condition. This is a comparison metric, not a proof of useful
    /// one-way flow in an arbitrary application.
    pub fn diodicity(&self) -> Option<f64> {
        if !self.forward.validate() || !self.reverse.validate() {
            return None;
        }
        let value = self.reverse.pressure_drop_pa / self.forward.pressure_drop_pa;
        value.is_finite().then_some(value)
    }

    /// Forward pressure-drop penalty relative to reverse pressure drop.
    pub fn pressure_drop_ratio(&self) -> Option<f64> {
        self.diodicity()
    }
}

/// Compute Reynolds number for a paired benchmark operating point.
pub fn reynolds_number(density: f64, flow_speed: f64, hydraulic_diameter: f64, viscosity: f64) -> Option<f64> {
    if !density.is_finite()
        || !flow_speed.is_finite()
        || !hydraulic_diameter.is_finite()
        || !viscosity.is_finite()
        || density <= 0.0
        || hydraulic_diameter <= 0.0
        || viscosity <= 0.0
    {
        return None;
    }

    let re = symthaea_thermofluids::fluids::reynolds_number(
        density,
        flow_speed.abs(),
        hydraulic_diameter,
        viscosity,
    );
    re.is_finite().then_some(re)
}

fn relative_flow_mismatch(a: f64, b: f64) -> f64 {
    let a = a.abs();
    let b = b.abs();
    if !a.is_finite() || !b.is_finite() || a == 0.0 || b == 0.0 {
        return f64::INFINITY;
    }
    (a - b).abs() / a.max(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_paired_result_produces_diodicity() {
        let observation = RectificationObservation {
            forward: DirectionalResult {
                pressure_drop_pa: 100.0,
                flow_rate_m3_s: 1.0,
            },
            reverse: DirectionalResult {
                pressure_drop_pa: 220.0,
                flow_rate_m3_s: -1.0,
            },
        };

        assert!(observation.validate(1.0e-12));
        assert!((observation.diodicity().unwrap() - 2.2).abs() < 1.0e-12);
    }

    #[test]
    fn mismatched_flow_rates_are_not_a_valid_paired_comparison() {
        let observation = RectificationObservation {
            forward: DirectionalResult {
                pressure_drop_pa: 100.0,
                flow_rate_m3_s: 1.0,
            },
            reverse: DirectionalResult {
                pressure_drop_pa: 220.0,
                flow_rate_m3_s: -0.7,
            },
        };

        assert!(!observation.validate(0.01));
        assert_eq!(observation.diodicity(), Some(2.2));
    }

    #[test]
    fn invalid_pressure_drop_is_rejected() {
        let observation = RectificationObservation {
            forward: DirectionalResult {
                pressure_drop_pa: 0.0,
                flow_rate_m3_s: 1.0,
            },
            reverse: DirectionalResult {
                pressure_drop_pa: 220.0,
                flow_rate_m3_s: -1.0,
            },
        };
        assert!(!observation.validate(0.01));
        assert!(observation.diodicity().is_none());
    }

    #[test]
    fn reynolds_delegates_to_shared_fluid_physics() {
        let re = reynolds_number(1000.0, 2.0, 0.05, 1e-3).unwrap();
        assert!((re - 100_000.0).abs() < 1e-6);
    }

    #[test]
    fn invalid_reynolds_inputs_are_rejected() {
        assert!(reynolds_number(0.0, 2.0, 0.05, 1e-3).is_none());
        assert!(reynolds_number(1000.0, 2.0, 0.05, 0.0).is_none());
    }
}
