// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! INA219 current/voltage sensor decoder.
//!
//! Implements [`SensorDecoder`] for the Texas Instruments INA219 high-side
//! current sensor. Reads shunt voltage and bus voltage from register 0x01,
//! then derives current using configurable shunt resistance.
//!
//! # Register Layout (burst read from 0x01)
//!
//! ```text
//! Offset  Register        Encoding
//! 0–1     Shunt Voltage   16-bit signed, LSB = 10µV
//! 2–3     Bus Voltage     Bits [15:3] = voltage, LSB = 4mV
//! ```
//!
//! # Default Address
//!
//! 0x45 (avoids PCA9685 at 0x40/0x41). The INA219 supports addresses
//! 0x40–0x4F via A0/A1 pin strapping.

use crate::sensor::SensorDecoder;

/// Default I2C address (avoids PCA9685 conflict).
const DEFAULT_ADDRESS: u8 = 0x45;

/// Shunt voltage register (start of 4-byte burst read).
const SHUNT_VOLTAGE_REG: u8 = 0x01;

/// Shunt voltage LSB in volts (10µV).
const SHUNT_LSB_V: f32 = 10e-6;

/// Bus voltage LSB in volts (4mV).
const BUS_LSB_V: f32 = 4e-3;

/// INA219 current/voltage sensor decoder.
///
/// Decodes 4-byte burst reads into `[current_amps, bus_voltage_volts]`.
///
/// # Example
///
/// ```
/// use symthaea_hal::ina219::Ina219Decoder;
/// use symthaea_hal::sensor::SensorDecoder;
///
/// let decoder = Ina219Decoder::new(0.1); // 100mΩ shunt
/// assert_eq!(decoder.name(), "ina219");
/// assert_eq!(decoder.read_len(), 4);
/// ```
pub struct Ina219Decoder {
    /// Shunt resistance in ohms.
    shunt_resistance: f32,
    /// I2C address.
    address: u8,
}

impl Ina219Decoder {
    /// Create a new INA219 decoder with the given shunt resistance (ohms).
    ///
    /// Uses the default I2C address (0x45).
    pub fn new(shunt_resistance_ohms: f32) -> Self {
        Self {
            shunt_resistance: shunt_resistance_ohms,
            address: DEFAULT_ADDRESS,
        }
    }

    /// Override the I2C address (default 0x45).
    pub fn with_address(mut self, address: u8) -> Self {
        self.address = address;
        self
    }

    /// Get the configured shunt resistance in ohms.
    pub fn shunt_resistance(&self) -> f32 {
        self.shunt_resistance
    }
}

impl SensorDecoder for Ina219Decoder {
    fn name(&self) -> &str {
        "ina219"
    }

    fn address(&self) -> u8 {
        self.address
    }

    fn read_register(&self) -> u8 {
        SHUNT_VOLTAGE_REG
    }

    fn read_len(&self) -> usize {
        4 // 2 bytes shunt + 2 bytes bus
    }

    fn decode(&self, raw: &[u8]) -> Vec<f32> {
        if raw.len() < 4 {
            return vec![0.0, 0.0];
        }

        // Shunt voltage: 16-bit signed, LSB = 10µV
        let shunt_raw = ((raw[0] as i16) << 8) | (raw[1] as i16);
        let shunt_v = shunt_raw as f32 * SHUNT_LSB_V;

        // Current: I = V_shunt / R_shunt
        let current_a = if self.shunt_resistance.abs() > f32::EPSILON {
            shunt_v / self.shunt_resistance
        } else {
            0.0
        };

        // Bus voltage: bits [15:3], LSB = 4mV
        let bus_raw = ((raw[2] as u16) << 8) | (raw[3] as u16);
        let bus_voltage_v = ((bus_raw >> 3) & 0x1FFF) as f32 * BUS_LSB_V;

        vec![current_a, bus_voltage_v]
    }

    fn who_am_i(&self) -> Option<(u8, u8)> {
        // INA219 doesn't have a WHO_AM_I register
        None
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ina219_metadata() {
        let decoder = Ina219Decoder::new(0.1);
        assert_eq!(decoder.name(), "ina219");
        assert_eq!(decoder.address(), DEFAULT_ADDRESS);
        assert_eq!(decoder.read_register(), 0x01);
        assert_eq!(decoder.read_len(), 4);
        assert!(decoder.who_am_i().is_none());
    }

    #[test]
    fn test_ina219_custom_address() {
        let decoder = Ina219Decoder::new(0.1).with_address(0x40);
        assert_eq!(decoder.address(), 0x40);
    }

    #[test]
    fn test_ina219_decode_zeros() {
        let decoder = Ina219Decoder::new(0.1);
        let raw = [0u8; 4];
        let values = decoder.decode(&raw);
        assert_eq!(values.len(), 2);
        assert!(values[0].abs() < f32::EPSILON, "current = {}", values[0]);
        assert!(values[1].abs() < f32::EPSILON, "voltage = {}", values[1]);
    }

    #[test]
    fn test_ina219_decode_known_current() {
        let decoder = Ina219Decoder::new(0.1); // 100mΩ shunt
        // Shunt voltage = 1000 * 10µV = 10mV → current = 0.01V / 0.1Ω = 0.1A
        let raw = [
            0x03, 0xE8, // shunt = 1000 (big-endian signed)
            0x00, 0x00, // bus = 0
        ];
        let values = decoder.decode(&raw);
        assert!(
            (values[0] - 0.1).abs() < 0.001,
            "expected ~0.1A, got {}",
            values[0]
        );
    }

    #[test]
    fn test_ina219_decode_bus_voltage() {
        let decoder = Ina219Decoder::new(0.1);
        // Bus voltage = 1000 in bits [15:3] → 1000 * 4mV = 4.0V
        // 1000 << 3 = 8000 = 0x1F40
        let raw = [
            0x00, 0x00, // shunt = 0
            0x1F, 0x40, // bus: bits[15:3] = 1000
        ];
        let values = decoder.decode(&raw);
        assert!(
            (values[1] - 4.0).abs() < 0.01,
            "expected ~4.0V, got {}",
            values[1]
        );
    }

    #[test]
    fn test_ina219_decode_negative_current() {
        let decoder = Ina219Decoder::new(0.1);
        // Shunt voltage = -1000 * 10µV → current = -0.01V / 0.1Ω = -0.1A
        // -1000 in i16 = 0xFC18
        let raw = [
            0xFC, 0x18, // shunt = -1000 (signed)
            0x00, 0x00, // bus = 0
        ];
        let values = decoder.decode(&raw);
        assert!(
            (values[0] + 0.1).abs() < 0.001,
            "expected ~-0.1A, got {}",
            values[0]
        );
    }

    #[test]
    fn test_ina219_short_buffer() {
        let decoder = Ina219Decoder::new(0.1);
        let raw = [0u8; 2]; // too short
        let values = decoder.decode(&raw);
        assert_eq!(values.len(), 2);
        assert!(values[0].abs() < f32::EPSILON);
        assert!(values[1].abs() < f32::EPSILON);
    }

    #[test]
    fn test_ina219_zero_shunt_resistance() {
        let decoder = Ina219Decoder::new(0.0); // zero resistance
        let raw = [0x03, 0xE8, 0x00, 0x00];
        let values = decoder.decode(&raw);
        assert!(
            values[0].abs() < f32::EPSILON,
            "zero resistance should give zero current"
        );
    }

    #[test]
    fn test_ina219_shunt_resistance_getter() {
        let decoder = Ina219Decoder::new(0.05);
        assert!((decoder.shunt_resistance() - 0.05).abs() < f32::EPSILON);
    }
}
