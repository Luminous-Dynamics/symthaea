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

    /// Validate decoder configuration before probing hardware.
    fn validate_config(&self) -> Result<(), String> {
        let len = self.read_len();
        if len == 0 || len > 4096 {
            return Err(format!("invalid read length: {len}"));
        }
        if self.address() > 0x7F {
            return Err(format!(
                "invalid 7-bit I2C address: 0x{:02X}",
                self.address()
            ));
        }
        Ok(())
    }

    /// WHO_AM_I register address (for availability check). `None` means the
    /// probe performs a real sample read instead of assuming availability.
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

    /// Probe the sensor by reading WHO_AM_I when available, otherwise by
    /// performing and validating one real sample read.
    pub fn probe(&mut self) -> HalResult<bool> {
        self.available = false;
        self.decoder
            .validate_config()
            .map_err(|detail| HalError::Sensor {
                name: self.decoder.name().to_string(),
                detail,
            })?;

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
            let reg = self.decoder.read_register();
            let mut buf = vec![0u8; self.decoder.read_len()];
            if let Err(e) = self
                .bus
                .write_read(self.decoder.address(), &[reg], &mut buf)
            {
                self.available = false;
                return Err(HalError::I2c {
                    bus: format!("0x{:02X}", self.decoder.address()),
                    detail: i2c_error_detail(embedded_hal::i2c::Error::kind(&e)),
                });
            }
            self.validate_decoded_sample(self.decoder.decode(&buf))?;
            self.available = true;
            Ok(true)
        }
    }

    fn validate_decoded_sample(&self, values: Vec<f32>) -> HalResult<Vec<f32>> {
        if values.is_empty() {
            return Err(HalError::Sensor {
                name: self.decoder.name().to_string(),
                detail: "decoder returned an empty sample".to_string(),
            });
        }
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(HalError::Sensor {
                name: self.decoder.name().to_string(),
                detail: format!("sample field {index} is non-finite: {value}"),
            });
        }
        Ok(values)
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
            Ok(()) => match self.validate_decoded_sample(self.decoder.decode(&buf)) {
                Ok(values) => Some(values),
                Err(error) => {
                    self.available = false;
                    warn!(sensor = self.decoder.name(), error = %error, "invalid sensor sample");
                    None
                }
            },
            Err(e) => {
                self.available = false;
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
    use crate::ina219::Ina219Decoder;
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
    fn test_probe_without_identity_performs_real_read() {
        let bus = MockI2cBus::new().with_responses(vec![vec![0, 0, 0, 0]]);
        let decoder = Ina219Decoder::new(0.1);
        let mut sensor = EmbeddedSensor::new(bus, decoder);
        assert!(sensor.probe().unwrap());
        assert!(sensor.is_available());
    }

    #[test]
    fn test_invalid_decoder_configuration_fails_probe() {
        let bus = MockI2cBus::new();
        let decoder = Ina219Decoder::new(0.0);
        let mut sensor = EmbeddedSensor::new(bus, decoder);
        assert!(matches!(sensor.probe(), Err(HalError::Sensor { .. })));
        assert!(!sensor.is_available());
    }

    #[test]
    fn test_read_failure_revokes_availability() {
        use embedded_hal::i2c;

        let bus = MockI2cBus::new().with_responses(vec![vec![0x68]]);
        let mut sensor = EmbeddedSensor::new(bus, Mpu6050Decoder);
        assert!(sensor.probe().unwrap());
        sensor.bus.inject_error(i2c::ErrorKind::Bus);
        assert!(sensor.read_raw().is_none());
        assert!(!sensor.is_available());
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
