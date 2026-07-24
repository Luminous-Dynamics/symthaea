// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MPU6050 6-axis IMU decoder.
//!
//! Implements [`SensorDecoder`] for the InvenSense MPU-6050, providing
//! 3-axis accelerometer + 3-axis gyroscope readings.
//!
//! # Register Layout (burst read from 0x3B)
//!
//! ```text
//! Offset  Register        Axis
//! 0–1     ACCEL_XOUT      Accelerometer X  (16-bit signed, ±2g default → /16384)
//! 2–3     ACCEL_YOUT      Accelerometer Y
//! 4–5     ACCEL_ZOUT      Accelerometer Z
//! 6–7     TEMP_OUT        Temperature (skipped)
//! 8–9     GYRO_XOUT       Gyroscope X  (16-bit signed, ±250°/s default → /131)
//! 10–11   GYRO_YOUT       Gyroscope Y
//! 12–13   GYRO_ZOUT       Gyroscope Z
//! ```

use std::time::Instant;

use crate::sensor::{HalSensorAdapter, SensorDecoder};

/// MPU6050 I2C address (AD0 pin low).
const MPU6050_ADDR: u8 = 0x68;

/// WHO_AM_I register.
const WHO_AM_I_REG: u8 = 0x75;
/// Expected WHO_AM_I value.
const WHO_AM_I_VAL: u8 = 0x68;

/// Start of accelerometer data (burst read).
const ACCEL_XOUT_H: u8 = 0x3B;

/// Bytes per burst read: 6 accel + 2 temp + 6 gyro = 14.
const BURST_LEN: usize = 14;

/// Accelerometer scale factor for ±2g range (LSB/g).
const ACCEL_SCALE: f32 = 16384.0;

/// Gyroscope scale factor for ±250°/s range (LSB/°/s).
const GYRO_SCALE: f32 = 131.0;

/// MPU6050 6-axis IMU decoder.
///
/// Decodes 14-byte burst reads into 6 f32 values:
/// `[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]`
///
/// Units: accelerometer in g, gyroscope in °/s.
pub struct Mpu6050Decoder;

impl SensorDecoder for Mpu6050Decoder {
    fn name(&self) -> &str {
        "mpu6050"
    }

    fn address(&self) -> u8 {
        MPU6050_ADDR
    }

    fn read_register(&self) -> u8 {
        ACCEL_XOUT_H
    }

    fn read_len(&self) -> usize {
        BURST_LEN
    }

    fn decode(&self, raw: &[u8]) -> Vec<f32> {
        if raw.len() < BURST_LEN {
            return vec![f32::NAN; 6];
        }

        let ax = i16_from_be(raw[0], raw[1]) as f32 / ACCEL_SCALE;
        let ay = i16_from_be(raw[2], raw[3]) as f32 / ACCEL_SCALE;
        let az = i16_from_be(raw[4], raw[5]) as f32 / ACCEL_SCALE;
        // raw[6..8] = temperature, skipped
        let gx = i16_from_be(raw[8], raw[9]) as f32 / GYRO_SCALE;
        let gy = i16_from_be(raw[10], raw[11]) as f32 / GYRO_SCALE;
        let gz = i16_from_be(raw[12], raw[13]) as f32 / GYRO_SCALE;

        vec![ax, ay, az, gx, gy, gz]
    }

    fn who_am_i(&self) -> Option<(u8, u8)> {
        Some((WHO_AM_I_REG, WHO_AM_I_VAL))
    }
}

/// Parse big-endian 16-bit signed integer from two bytes.
fn i16_from_be(high: u8, low: u8) -> i16 {
    ((high as i16) << 8) | (low as i16)
}

// ============================================================================
// COMPLEMENTARY FILTER
// ============================================================================

/// Complementary filter: fuses accelerometer + gyroscope into roll/pitch angles.
///
/// Wraps any [`HalSensorAdapter`] that produces 6-element readings
/// `[ax, ay, az, gx, gy, gz]` (accel in g, gyro in °/s) and outputs
/// `[roll_deg, pitch_deg]`.
///
/// **Yaw is not computed** — requires a magnetometer, unavailable on MPU6050.
///
/// Filter formula:
/// `angle = alpha * (angle + gyro * dt) + (1 - alpha) * atan2(accel)`
pub struct ComplementaryFilter {
    inner: Box<dyn HalSensorAdapter>,
    alpha: f32,
    roll: f32,
    pitch: f32,
    last_time: Option<Instant>,
    initialized: bool,
}

impl ComplementaryFilter {
    /// Create a new complementary filter wrapping a 6-axis IMU sensor.
    ///
    /// Default alpha = 0.98 (heavy gyro weighting for smooth output).
    pub fn new(inner: Box<dyn HalSensorAdapter>) -> Self {
        Self {
            inner,
            alpha: 0.98,
            roll: 0.0,
            pitch: 0.0,
            last_time: None,
            initialized: false,
        }
    }

    /// Set the complementary filter coefficient (0.0–1.0).
    ///
    /// Higher alpha → more gyro trust, less accel noise.
    /// Lower alpha → faster convergence to accelerometer angle.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        if alpha.is_finite() {
            self.alpha = alpha.clamp(0.0, 1.0);
        }
        self
    }

    /// Compute roll angle from accelerometer (degrees).
    fn accel_roll(ay: f32, az: f32) -> f32 {
        ay.atan2(az).to_degrees()
    }

    /// Compute pitch angle from accelerometer (degrees).
    fn accel_pitch(ax: f32, ay: f32, az: f32) -> f32 {
        (-ax).atan2((ay * ay + az * az).sqrt()).to_degrees()
    }
}

impl HalSensorAdapter for ComplementaryFilter {
    fn name(&self) -> &str {
        "complementary_filter"
    }

    fn read_raw(&mut self) -> Option<Vec<f32>> {
        let values = self.inner.read_raw()?;
        if values.len() < 6 || values[..6].iter().any(|value| !value.is_finite()) {
            return None;
        }

        let (ax, ay, az) = (values[0], values[1], values[2]);
        let (gx, gy, _gz) = (values[3], values[4], values[5]);

        let accel_roll = Self::accel_roll(ay, az);
        let accel_pitch = Self::accel_pitch(ax, ay, az);

        if !self.initialized {
            // First reading: initialize from accel only
            self.roll = accel_roll;
            self.pitch = accel_pitch;
            self.last_time = Some(Instant::now());
            self.initialized = true;
            return Some(vec![self.roll, self.pitch]);
        }

        let now = Instant::now();
        let dt = if let Some(last) = self.last_time {
            now.duration_since(last).as_secs_f32()
        } else {
            0.0
        };
        self.last_time = Some(now);

        // Complementary filter fusion
        self.roll = self.alpha * (self.roll + gx * dt) + (1.0 - self.alpha) * accel_roll;
        self.pitch = self.alpha * (self.pitch + gy * dt) + (1.0 - self.alpha) * accel_pitch;

        Some(vec![self.roll, self.pitch])
    }

    fn is_available(&self) -> bool {
        self.inner.is_available()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockHalSensor;

    #[test]
    fn test_mpu6050_decode_zeros() {
        let decoder = Mpu6050Decoder;
        let raw = [0u8; BURST_LEN];
        let values = decoder.decode(&raw);
        assert_eq!(values.len(), 6);
        for &v in &values {
            assert!(v.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_mpu6050_decode_1g_z() {
        let decoder = Mpu6050Decoder;
        let mut raw = [0u8; BURST_LEN];
        // Accel Z = +16384 = 1g at ±2g scale
        raw[4] = 0x40; // 16384 >> 8 = 0x40
        raw[5] = 0x00;
        let values = decoder.decode(&raw);
        assert!((values[2] - 1.0).abs() < 0.001, "az = {}", values[2]);
    }

    #[test]
    fn test_mpu6050_decode_negative_accel() {
        let decoder = Mpu6050Decoder;
        let mut raw = [0u8; BURST_LEN];
        // Accel X = -16384 = -1g
        raw[0] = 0xC0; // -16384 >> 8 = 0xC0
        raw[1] = 0x00;
        let values = decoder.decode(&raw);
        assert!((values[0] + 1.0).abs() < 0.001, "ax = {}", values[0]);
    }

    #[test]
    fn test_mpu6050_decode_gyro() {
        let decoder = Mpu6050Decoder;
        let mut raw = [0u8; BURST_LEN];
        // Gyro X = 131 = 1°/s at ±250°/s scale
        raw[8] = 0x00;
        raw[9] = 0x83; // 131 = 0x0083
        let values = decoder.decode(&raw);
        assert!((values[3] - 1.0).abs() < 0.01, "gx = {}", values[3]);
    }

    #[test]
    fn test_mpu6050_short_buffer() {
        let decoder = Mpu6050Decoder;
        let raw = [0u8; 4]; // too short
        let values = decoder.decode(&raw);
        assert_eq!(values.len(), 6);
        // Malformed evidence must not be represented as a valid level reading.
        for &v in &values {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn test_mpu6050_who_am_i() {
        let decoder = Mpu6050Decoder;
        let (reg, val) = decoder.who_am_i().unwrap();
        assert_eq!(reg, 0x75);
        assert_eq!(val, 0x68);
    }

    #[test]
    fn test_mpu6050_metadata() {
        let decoder = Mpu6050Decoder;
        assert_eq!(decoder.name(), "mpu6050");
        assert_eq!(decoder.address(), 0x68);
        assert_eq!(decoder.read_register(), 0x3B);
        assert_eq!(decoder.read_len(), 14);
    }

    #[test]
    fn test_i16_from_be() {
        assert_eq!(i16_from_be(0x00, 0x01), 1);
        assert_eq!(i16_from_be(0xFF, 0xFF), -1);
        assert_eq!(i16_from_be(0x40, 0x00), 16384);
        assert_eq!(i16_from_be(0x80, 0x00), -32768);
    }

    // ── Complementary filter tests ───────────────────────────────────

    #[test]
    fn test_complementary_filter_level() {
        // Gravity along Z → roll≈0, pitch≈0
        let readings = vec![vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]; 3];
        let sensor = MockHalSensor::new("mpu", readings);
        let mut filter = ComplementaryFilter::new(Box::new(sensor));

        let out = filter.read_raw().unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].abs() < 1.0, "roll should be ~0, got {}", out[0]);
        assert!(out[1].abs() < 1.0, "pitch should be ~0, got {}", out[1]);
    }

    #[test]
    fn test_complementary_filter_roll_90() {
        // Gravity along Y → roll ≈ 90°
        let readings = vec![vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; 3];
        let sensor = MockHalSensor::new("mpu", readings);
        let mut filter = ComplementaryFilter::new(Box::new(sensor));

        let out = filter.read_raw().unwrap();
        assert!(
            (out[0] - 90.0).abs() < 1.0,
            "roll should be ~90, got {}",
            out[0]
        );
    }

    #[test]
    fn test_complementary_filter_pitch_90() {
        // Gravity along -X → pitch ≈ 90°
        let readings = vec![vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; 3];
        let sensor = MockHalSensor::new("mpu", readings);
        let mut filter = ComplementaryFilter::new(Box::new(sensor));

        let out = filter.read_raw().unwrap();
        assert!(
            (out[1] - 90.0).abs() < 1.0,
            "pitch should be ~90, got {}",
            out[1]
        );
    }

    #[test]
    fn test_complementary_filter_gyro_fusion() {
        // Two reads exercise the gyro integration path
        let readings = vec![
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 10.0, 10.0, 0.0], // some gyro rotation
        ];
        let sensor = MockHalSensor::new("mpu", readings);
        let mut filter = ComplementaryFilter::new(Box::new(sensor));

        let _first = filter.read_raw().unwrap();
        let second = filter.read_raw().unwrap();
        // Just verify it produces finite output — exact value depends on dt
        assert!(second[0].is_finite());
        assert!(second[1].is_finite());
    }

    #[test]
    fn test_complementary_filter_config_and_metadata() {
        let sensor = MockHalSensor::new("mpu", vec![vec![0.0; 6]; 3]);
        let filter = ComplementaryFilter::new(Box::new(sensor)).with_alpha(0.95);
        assert_eq!(filter.name(), "complementary_filter");
        assert!(filter.is_available());
        assert!((filter.alpha - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_complementary_filter_rejects_non_finite_input() {
        let readings = vec![vec![f32::NAN, 0.0, 1.0, 0.0, 0.0, 0.0]];
        let sensor = MockHalSensor::new("mpu", readings);
        let mut filter = ComplementaryFilter::new(Box::new(sensor)).with_alpha(f32::NAN);
        assert!((filter.alpha - 0.98).abs() < f32::EPSILON);
        assert!(filter.read_raw().is_none());
    }

    #[test]
    fn test_complementary_filter_short_reading() {
        // Only 4 values — should return None (need 6)
        let readings = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let sensor = MockHalSensor::new("mpu", readings);
        let mut filter = ComplementaryFilter::new(Box::new(sensor));
        assert!(filter.read_raw().is_none());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn decode_always_produces_6_finite_f32s(raw in proptest::collection::vec(any::<u8>(), 14..=14)) {
            let decoder = Mpu6050Decoder;
            let values = decoder.decode(&raw);
            prop_assert_eq!(values.len(), 6);
            for (i, v) in values.iter().enumerate() {
                prop_assert!(v.is_finite(), "value[{}] = {} is not finite", i, v);
            }
        }

        #[test]
        fn decode_short_buffer_produces_non_finite_evidence(len in 0usize..14) {
            let decoder = Mpu6050Decoder;
            let raw = vec![0u8; len];
            let values = decoder.decode(&raw);
            prop_assert_eq!(values.len(), 6);
            for v in &values {
                prop_assert!(v.is_nan());
            }
        }
    }
}
