// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HAL sensor adapter: generic I2C sensor → f32 readings.
//!
//! The [`HalSensorAdapter`] trait provides a uniform interface for reading
//! sensor data as `Vec<f32>`. The generic [`EmbeddedSensor`] combines an
//! I2C bus with a [`SensorDecoder`] to implement this trait for any
//! embedded-hal-compatible device.

use embedded_hal::i2c::I2c;
use tracing::warn;

use crate::error::{HalError, HalResult, i2c_error_detail};

// ============================================================================
// SENSOR ADAPTER TRAIT
// ============================================================================

/// Uniform interface for reading sensor data as f32 values.
///
/// Implemented by [`EmbeddedSensor`] (hardware) and [`MockHalSensor`](crate::mock::MockHalSensor) (testing).
pub trait HalSensorAdapter: Send {
    /// Human-readable sensor name.
    fn name(&self) -> &str;

    /// Read raw sensor values. Returns `None` if no data available.
    fn read_raw(&mut self) -> Option<Vec<f32>>;

    /// Whether the sensor is currently operational.
    fn is_available(&self) -> bool;
}

// ============================================================================
// SENSOR DECODER TRAIT
// ============================================================================

/// Decodes raw I2C register bytes into f32 sensor values.
///
/// Implement this for each sensor chip (MPU6050, BMP280, etc.).
pub trait SensorDecoder: Send {
    /// Sensor name (e.g., `"mpu6050"`).
    fn name(&self) -> &str;

    /// I2C address of the sensor.
    fn address(&self) -> u8;

    /// Register address to start reading from.
    fn read_register(&self) -> u8;

    /// Number of bytes to read per sample.
    fn read_len(&self) -> usize;

    /// Decode raw bytes into f32 values.
    fn decode(&self, raw: &[u8]) -> Vec<f32>;

    /// WHO_AM_I register address (for availability check). `None` to skip.
    fn who_am_i(&self) -> Option<(u8, u8)> {
        None
    }
}

// ============================================================================
// GENERIC EMBEDDED SENSOR
// ============================================================================

/// Generic I2C sensor that combines a bus with a decoder.
pub struct EmbeddedSensor<I, D> {
    bus: I,
    decoder: D,
    available: bool,
}

impl<I: I2c, D: SensorDecoder> EmbeddedSensor<I, D> {
    /// Create a new embedded sensor.
    pub fn new(bus: I, decoder: D) -> Self {
        Self {
            bus,
            decoder,
            available: false,
        }
    }

    /// Probe the sensor by reading WHO_AM_I (if supported).
    pub fn probe(&mut self) -> HalResult<bool> {
        if let Some((reg, expected)) = self.decoder.who_am_i() {
            let mut buf = [0u8; 1];
            match self
                .bus
                .write_read(self.decoder.address(), &[reg], &mut buf)
            {
                Ok(()) => {
                    self.available = buf[0] == expected;
                    if !self.available {
                        warn!(
                            sensor = self.decoder.name(),
                            expected = format!("0x{:02X}", expected),
                            got = format!("0x{:02X}", buf[0]),
                            "WHO_AM_I mismatch"
                        );
                    }
                    Ok(self.available)
                }
                Err(e) => {
                    self.available = false;
                    Err(HalError::I2c {
                        bus: format!("0x{:02X}", self.decoder.address()),
                        detail: i2c_error_detail(embedded_hal::i2c::Error::kind(&e)),
                    })
                }
            }
        } else {
            // No WHO_AM_I — assume available
            self.available = true;
            Ok(true)
        }
    }

    /// Get a reference to the decoder.
    pub fn decoder(&self) -> &D {
        &self.decoder
    }
}

impl<I: I2c + Send, D: SensorDecoder> HalSensorAdapter for EmbeddedSensor<I, D> {
    fn name(&self) -> &str {
        self.decoder.name()
    }

    fn read_raw(&mut self) -> Option<Vec<f32>> {
        if !self.available {
            return None;
        }

        let reg = self.decoder.read_register();
        let len = self.decoder.read_len();
        let mut buf = vec![0u8; len];

        match self
            .bus
            .write_read(self.decoder.address(), &[reg], &mut buf)
        {
            Ok(()) => Some(self.decoder.decode(&buf)),
            Err(e) => {
                warn!(
                    sensor = self.decoder.name(),
                    error = %i2c_error_detail(embedded_hal::i2c::Error::kind(&e)),
                    "sensor read failed"
                );
                None
            }
        }
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imu::Mpu6050Decoder;
    use crate::mock::MockI2cBus;

    #[test]
    fn test_embedded_sensor_probe_success() {
        // MPU6050 WHO_AM_I register 0x75 should return 0x68
        let bus = MockI2cBus::new().with_responses(vec![vec![0x68]]);
        let decoder = Mpu6050Decoder;
        let mut sensor = EmbeddedSensor::new(bus, decoder);

        assert!(!sensor.is_available());
        let result = sensor.probe().unwrap();
        assert!(result);
        assert!(sensor.is_available());
    }

    #[test]
    fn test_embedded_sensor_probe_wrong_id() {
        let bus = MockI2cBus::new().with_responses(vec![vec![0xFF]]);
        let decoder = Mpu6050Decoder;
        let mut sensor = EmbeddedSensor::new(bus, decoder);

        let result = sensor.probe().unwrap();
        assert!(!result);
        assert!(!sensor.is_available());
    }

    #[test]
    fn test_embedded_sensor_read() {
        // Probe response + one sensor read (14 bytes for MPU6050)
        let accel_gyro = vec![
            0x00, 0x01, // accel_x high/low
            0x00, 0x02, // accel_y
            0x00, 0x03, // accel_z
            0x00, 0x00, // temp (ignored)
            0x00, 0x04, // gyro_x
            0x00, 0x05, // gyro_y
            0x00, 0x06, // gyro_z
        ];
        let bus = MockI2cBus::new().with_responses(vec![vec![0x68], accel_gyro]);
        let decoder = Mpu6050Decoder;
        let mut sensor = EmbeddedSensor::new(bus, decoder);

        sensor.probe().unwrap();
        let values = sensor.read_raw().unwrap();
        assert_eq!(values.len(), 6); // ax, ay, az, gx, gy, gz
    }

    #[test]
    fn test_embedded_sensor_read_when_unavailable() {
        let bus = MockI2cBus::new();
        let decoder = Mpu6050Decoder;
        let mut sensor = EmbeddedSensor::new(bus, decoder);

        // Not probed → not available → returns None
        assert!(sensor.read_raw().is_none());
    }
}
