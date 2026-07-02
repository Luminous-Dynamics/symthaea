// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pedestrian Dead Reckoning (PDR): position estimation from phone IMU sensors.
//!
//! When GPS is denied and no peer anchors are available, PDR uses the phone's
//! built-in accelerometer, gyroscope, and barometer to estimate relative position
//! from the last known fix.
//!
//! # Algorithm
//!
//! Each step detected by the accelerometer advances the position:
//!   position += stride_length × [cos(heading), sin(heading)]
//!
//! Heading is integrated from the gyroscope rotation rate, corrected by
//! magnetometer when available.
//!
//! # Practical Envelope
//!
//! This is a bounded fallback, not a long-duration navigation solution.
//! In commodity-phone conditions it is most useful for tens of seconds to a
//! few minutes between stronger external fixes. Beyond that, heading and
//! stride errors dominate.
//!
//! # References
//!
//! - Weinberg, H. (2002). Using the ADXL202 in Pedometer and Personal Navigation Applications.
//! - Harle, R. (2013). A Survey of Indoor Inertial Navigation Systems for Pedestrians. IEEE.
//! - Jimenez et al. (2009). A Comparison of Pedestrian Dead-Reckoning Algorithms.

use serde::{Deserialize, Serialize};

/// Standard atmosphere (hPa) for barometric altitude.
const P0_HPA: f64 = 1013.25;

/// Pedestrian Dead Reckoning engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PedestrianDeadReckoning {
    /// Current heading in radians (0 = north/+y, π/2 = east/+x).
    heading_rad: f64,
    /// Adaptive stride length (meters).
    stride_length_m: f64,
    /// 2D position relative to origin [east_m, north_m].
    position: [f64; 2],
    /// Altitude from barometric pressure (meters).
    altitude_m: f64,
    /// Reference barometric pressure at origin (hPa).
    reference_pressure_hpa: f64,
    /// Total steps counted.
    total_steps: u64,
    /// Estimated cumulative drift (meters).
    drift_accumulated_m: f64,
    /// Accelerometer history for stride estimation (ring buffer, last 10).
    accel_history: Vec<f32>,
    /// Whether magnetometer correction is active.
    mag_correction_active: bool,
    /// Configuration.
    config: PdrConfig,
}

/// PDR configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdrConfig {
    /// Default stride length when no accelerometer calibration (meters).
    pub default_stride_m: f64,
    /// Weinberg constant k for stride = k * (a_max - a_min)^0.25.
    pub weinberg_k: f64,
    /// Gyro drift rate without magnetometer (rad/s).
    pub gyro_drift_rate: f64,
    /// Drift accumulation rate per step (meters).
    pub drift_per_step_m: f64,
    /// Drift accumulation rate per step with magnetometer (meters).
    pub drift_per_step_mag_m: f64,
    /// Minimum accelerometer variance to detect walking.
    pub walk_threshold: f64,
    /// Maximum complementary-filter gain applied to magnetometer corrections.
    pub mag_max_correction_gain: f64,
}

impl Default for PdrConfig {
    fn default() -> Self {
        Self {
            default_stride_m: 0.71,     // Average adult walking stride
            weinberg_k: 0.42,           // Weinberg 2002 constant
            gyro_drift_rate: 0.0025,    // Conservative commodity MEMS drift
            drift_per_step_m: 0.25,     // ~25cm drift per step without mag
            drift_per_step_mag_m: 0.10, // ~10cm drift per step with mag
            walk_threshold: 1.5,        // m/s² accel variance threshold
            mag_max_correction_gain: 0.15,
        }
    }
}

impl PedestrianDeadReckoning {
    /// Create a new PDR engine.
    pub fn new(config: PdrConfig) -> Self {
        Self {
            heading_rad: 0.0,
            stride_length_m: config.default_stride_m,
            position: [0.0, 0.0],
            altitude_m: 0.0,
            reference_pressure_hpa: P0_HPA,
            total_steps: 0,
            drift_accumulated_m: 0.0,
            accel_history: Vec::with_capacity(10),
            mag_correction_active: false,
            config,
        }
    }

    /// Process one cycle of IMU data.
    ///
    /// # Arguments
    /// - `accel_magnitude`: Accelerometer magnitude (m/s²), ~9.8 when stationary
    /// - `rotation_rate`: Gyroscope rotation rate around vertical axis (rad/s)
    /// - `barometer_hpa`: Barometric pressure (hPa), 0 to skip
    /// - `step_delta`: Steps detected since last call (from Android step counter)
    /// - `dt_s`: Time delta in seconds
    pub fn step(
        &mut self,
        accel_magnitude: f32,
        rotation_rate: f32,
        barometer_hpa: f32,
        step_delta: u32,
        dt_s: f64,
    ) {
        // 1. Update heading from gyroscope
        self.heading_rad += rotation_rate as f64 * dt_s;
        // Normalize to [0, 2π)
        self.heading_rad = self.heading_rad.rem_euclid(std::f64::consts::TAU);

        // 2. Update accelerometer history for stride estimation
        self.accel_history.push(accel_magnitude);
        if self.accel_history.len() > 10 {
            self.accel_history.remove(0);
        }

        // 3. Estimate stride length from accelerometer variance (Weinberg model)
        if self.accel_history.len() >= 4 {
            let a_max = self.accel_history.iter().cloned().fold(f32::MIN, f32::max);
            let a_min = self.accel_history.iter().cloned().fold(f32::MAX, f32::min);
            let a_range = (a_max - a_min) as f64;
            if a_range > self.config.walk_threshold {
                // Weinberg formula: stride = k * (a_max - a_min)^0.25
                self.stride_length_m = self.config.weinberg_k * a_range.powf(0.25);
                // Clamp to reasonable range
                self.stride_length_m = self.stride_length_m.clamp(0.3, 1.5);
            }
        }

        // 4. Advance position by detected steps
        for _ in 0..step_delta {
            self.position[0] += self.stride_length_m * self.heading_rad.sin(); // east
            self.position[1] += self.stride_length_m * self.heading_rad.cos(); // north
            self.total_steps += 1;

            // Accumulate drift estimate
            let drift_per_step = if self.mag_correction_active {
                self.config.drift_per_step_mag_m
            } else {
                self.config.drift_per_step_m
            };
            self.drift_accumulated_m += drift_per_step;
        }

        // 5. Add gyro drift (even when not stepping)
        if !self.mag_correction_active {
            self.drift_accumulated_m += self.config.gyro_drift_rate * dt_s;
        }

        // 6. Update altitude from barometer
        if barometer_hpa > 100.0 {
            self.altitude_m =
                barometric_altitude(barometer_hpa as f64, self.reference_pressure_hpa);
        }
    }

    /// Correct heading using magnetometer reading.
    ///
    /// # Arguments
    /// - `magnetic_heading_deg`: Magnetic north heading (0-360°)
    /// - `accuracy_deg`: Heading accuracy (typically 5-15°)
    pub fn correct_heading(&mut self, magnetic_heading_deg: f32, accuracy_deg: f32) {
        if accuracy_deg > 45.0 {
            return;
        } // Too noisy
        let mag_rad = (magnetic_heading_deg as f64).to_radians();
        // Indoor magnetometers are often badly distorted, so keep the correction
        // gain modest and let the reported accuracy modulate how strongly we trust it.
        let quality = ((45.0 - accuracy_deg as f64) / 40.0).clamp(0.0, 1.0);
        let alpha = (self.config.mag_max_correction_gain * quality).max(0.02);
        // Handle angle wrapping
        let diff = angle_diff(self.heading_rad, mag_rad);
        self.heading_rad += alpha * diff;
        self.heading_rad = self.heading_rad.rem_euclid(std::f64::consts::TAU);
        self.mag_correction_active = true;
    }

    /// Get current relative position [east_m, north_m].
    pub fn get_position(&self) -> [f64; 2] {
        self.position
    }

    /// Get current altitude (meters, relative to reference pressure).
    pub fn get_altitude(&self) -> f64 {
        self.altitude_m
    }

    /// Get current heading in degrees (0-360, north-referenced).
    pub fn get_heading_deg(&self) -> f64 {
        self.heading_rad.to_degrees().rem_euclid(360.0)
    }

    /// Get estimated cumulative drift in meters.
    pub fn get_drift_estimate(&self) -> f64 {
        self.drift_accumulated_m
    }

    /// Get total steps counted.
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Get current stride length estimate.
    pub fn stride_length(&self) -> f64 {
        self.stride_length_m
    }

    /// Reset position to a known fix (GPS or mesh). Clears drift.
    pub fn reset_to_origin(&mut self) {
        self.position = [0.0, 0.0];
        self.drift_accumulated_m = 0.0;
    }

    /// Set reference barometric pressure for altitude computation.
    pub fn set_reference_pressure(&mut self, pressure_hpa: f64) {
        self.reference_pressure_hpa = pressure_hpa;
    }

    /// Set initial heading (e.g., from magnetometer at startup).
    pub fn set_heading(&mut self, heading_deg: f64) {
        self.heading_rad = heading_deg.to_radians();
    }
}

/// Compute altitude from barometric pressure using the hypsometric formula.
///
/// alt = 44330 × (1 - (p/p0)^0.1903) meters
pub fn barometric_altitude(pressure_hpa: f64, reference_hpa: f64) -> f64 {
    44330.0 * (1.0 - (pressure_hpa / reference_hpa).powf(0.1903))
}

/// Compute shortest signed angle difference, handling wrapping.
fn angle_diff(a: f64, b: f64) -> f64 {
    let diff = b - a;
    let pi = std::f64::consts::PI;
    if diff > pi {
        diff - 2.0 * pi
    } else if diff < -pi {
        diff + 2.0 * pi
    } else {
        diff
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_straight_north() {
        let mut pdr = PedestrianDeadReckoning::new(PdrConfig::default());
        pdr.set_heading(0.0); // North
                              // Walk 10 steps
        for _ in 0..10 {
            pdr.step(10.5, 0.0, 0.0, 1, 0.5);
        }
        let pos = pdr.get_position();
        assert!(pos[0].abs() < 0.01, "Should not drift east: {}", pos[0]);
        assert!(
            (pos[1] - 7.1).abs() < 1.0,
            "Should be ~7.1m north: {}",
            pos[1]
        );
        assert_eq!(pdr.total_steps(), 10);
    }

    #[test]
    fn walk_east() {
        let mut pdr = PedestrianDeadReckoning::new(PdrConfig::default());
        pdr.set_heading(90.0); // East
        for _ in 0..10 {
            pdr.step(10.5, 0.0, 0.0, 1, 0.5);
        }
        let pos = pdr.get_position();
        assert!(
            (pos[0] - 7.1).abs() < 1.0,
            "Should be ~7.1m east: {}",
            pos[0]
        );
        assert!(pos[1].abs() < 0.01, "Should not drift north: {}", pos[1]);
    }

    #[test]
    fn walk_square_returns_near_origin() {
        let mut pdr = PedestrianDeadReckoning::new(PdrConfig::default());
        let steps_per_side = 14; // ~10m per side

        // Walk north
        pdr.set_heading(0.0);
        for _ in 0..steps_per_side {
            pdr.step(10.5, 0.0, 0.0, 1, 0.5);
        }
        // Walk east
        pdr.set_heading(90.0);
        for _ in 0..steps_per_side {
            pdr.step(10.5, 0.0, 0.0, 1, 0.5);
        }
        // Walk south
        pdr.set_heading(180.0);
        for _ in 0..steps_per_side {
            pdr.step(10.5, 0.0, 0.0, 1, 0.5);
        }
        // Walk west
        pdr.set_heading(270.0);
        for _ in 0..steps_per_side {
            pdr.step(10.5, 0.0, 0.0, 1, 0.5);
        }

        let pos = pdr.get_position();
        let error = (pos[0].powi(2) + pos[1].powi(2)).sqrt();
        // With perfect heading (no gyro), should return exactly to origin
        assert!(error < 0.1, "Square walk return error: {}m", error);
    }

    #[test]
    fn gyro_heading_integration() {
        let mut pdr = PedestrianDeadReckoning::new(PdrConfig::default());
        pdr.set_heading(0.0);
        // Rotate 90° (π/2 rad) over 1 second at π/2 rad/s
        pdr.step(9.8, std::f64::consts::FRAC_PI_2 as f32, 0.0, 0, 1.0);
        assert!(
            (pdr.get_heading_deg() - 90.0).abs() < 1.0,
            "Heading: {}",
            pdr.get_heading_deg()
        );
    }

    #[test]
    fn magnetometer_correction() {
        let mut pdr = PedestrianDeadReckoning::new(PdrConfig::default());
        pdr.set_heading(0.0);
        // Gyro drifts heading to 10°
        pdr.heading_rad = 10.0_f64.to_radians();
        // Magnetometer says true heading is 0°
        pdr.correct_heading(0.0, 5.0);
        // Should blend toward 0° without fully trusting the magnetometer
        assert!(
            pdr.get_heading_deg() < 10.0,
            "Should correct: {}",
            pdr.get_heading_deg()
        );
        assert!(
            pdr.get_heading_deg() > 0.0,
            "Should not overcorrect: {}",
            pdr.get_heading_deg()
        );
        assert!(pdr.mag_correction_active);
    }

    #[test]
    fn drift_accumulates() {
        let mut pdr = PedestrianDeadReckoning::new(PdrConfig::default());
        assert_eq!(pdr.get_drift_estimate(), 0.0);
        for _ in 0..100 {
            pdr.step(10.5, 0.0, 0.0, 1, 0.5);
        }
        assert!(
            pdr.get_drift_estimate() > 10.0,
            "100 steps should accumulate drift"
        );
    }

    #[test]
    fn drift_lower_with_magnetometer() {
        let mut pdr_no_mag = PedestrianDeadReckoning::new(PdrConfig::default());
        let mut pdr_mag = PedestrianDeadReckoning::new(PdrConfig::default());
        pdr_mag.mag_correction_active = true;

        for _ in 0..100 {
            pdr_no_mag.step(10.5, 0.0, 0.0, 1, 0.5);
            pdr_mag.step(10.5, 0.0, 0.0, 1, 0.5);
        }
        assert!(
            pdr_mag.get_drift_estimate() < pdr_no_mag.get_drift_estimate(),
            "Magnetometer should reduce drift: {} vs {}",
            pdr_mag.get_drift_estimate(),
            pdr_no_mag.get_drift_estimate()
        );
    }

    #[test]
    fn reset_clears_drift() {
        let mut pdr = PedestrianDeadReckoning::new(PdrConfig::default());
        for _ in 0..50 {
            pdr.step(10.5, 0.0, 0.0, 1, 0.5);
        }
        assert!(pdr.get_drift_estimate() > 0.0);
        pdr.reset_to_origin();
        assert_eq!(pdr.get_drift_estimate(), 0.0);
        assert_eq!(pdr.get_position(), [0.0, 0.0]);
    }

    #[test]
    fn barometric_altitude_standard() {
        let alt = barometric_altitude(1013.25, 1013.25);
        assert!(alt.abs() < 0.1, "Sea level: {}", alt);
        let alt_1000m = barometric_altitude(898.76, 1013.25);
        assert!((alt_1000m - 1000.0).abs() < 50.0, "1000m: {}", alt_1000m);
    }

    #[test]
    fn barometric_floor_detection() {
        // ~3m floor height ≈ 0.36 hPa pressure change
        let floor0 = barometric_altitude(1013.25, 1013.25);
        let floor1 = barometric_altitude(1012.89, 1013.25);
        let diff = floor1 - floor0;
        assert!((diff - 3.0).abs() < 1.0, "Floor height: {}m", diff);
    }

    #[test]
    fn adaptive_stride_from_accel() {
        let mut pdr = PedestrianDeadReckoning::new(PdrConfig::default());
        // Simulate vigorous walking (high accel variance)
        let accels = [8.0, 12.0, 7.5, 13.0, 8.0, 12.5, 7.0, 13.5, 8.5, 11.5];
        for &a in &accels {
            pdr.step(a, 0.0, 0.0, 0, 0.1);
        }
        // Stride should adapt from default (0.71) to reflect high variance
        assert!(pdr.stride_length() > 0.5, "Stride: {}", pdr.stride_length());
        assert!(pdr.stride_length() < 1.5, "Stride: {}", pdr.stride_length());
    }

    #[test]
    fn no_steps_no_position_change() {
        let mut pdr = PedestrianDeadReckoning::new(PdrConfig::default());
        pdr.step(9.8, 0.0, 0.0, 0, 1.0);
        pdr.step(9.8, 0.0, 0.0, 0, 1.0);
        assert_eq!(pdr.get_position(), [0.0, 0.0]);
        assert_eq!(pdr.total_steps(), 0);
    }
}
