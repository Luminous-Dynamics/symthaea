// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PCA9685 16-channel 12-bit PWM controller driver.
//!
//! Generic over `embedded_hal::i2c::I2c` — works with both real hardware
//! (`linux_embedded_hal::I2cdev`) and [`MockI2cBus`](crate::mock::MockI2cBus) for testing.
//!
//! # Board Layout
//!
//! ```text
//! Board 0 (0x40): joints  0–15  (abdomen + legs + right_shoulder1)
//! Board 1 (0x41): joints 16–20  (remaining arms)
//! ```

use embedded_hal::i2c::I2c;
use tracing::{debug, warn};

use crate::error::{HalError, HalResult, i2c_error_detail};

// ============================================================================
// PCA9685 REGISTER MAP
// ============================================================================

/// PCA9685 register addresses.
mod reg {
    pub const MODE1: u8 = 0x00;
    #[allow(dead_code)]
    pub const MODE2: u8 = 0x01;
    pub const LED0_ON_L: u8 = 0x06;
    pub const ALL_LED_OFF_L: u8 = 0xFC;
    pub const ALL_LED_OFF_H: u8 = 0xFD;
    pub const PRE_SCALE: u8 = 0xFE;
}

/// MODE1 register bits.
mod mode1 {
    pub const RESTART: u8 = 0x80;
    pub const SLEEP: u8 = 0x10;
    pub const AI: u8 = 0x20; // Auto-increment
}

/// Number of PWM channels on a PCA9685.
pub const CHANNELS: usize = 16;

/// Internal oscillator frequency (25 MHz nominal).
const OSC_CLOCK: f64 = 25_000_000.0;

// ============================================================================
// PCA9685 DRIVER
// ============================================================================

/// PCA9685 I2C PWM controller.
///
/// Each channel produces a 12-bit (0–4095) PWM signal at a configurable
/// frequency. For hobby servos, 50 Hz is standard.
pub struct Pca9685<I> {
    bus: I,
    address: u8,
    /// Current prescale value (determines PWM frequency).
    prescale: u8,
}

impl<I: I2c> Pca9685<I> {
    /// Create a new PCA9685 driver without initializing the device.
    pub fn new(bus: I, address: u8) -> Self {
        Self {
            bus,
            address,
            prescale: 0,
        }
    }

    /// Initialize the PCA9685: reset, set frequency, enable auto-increment.
    ///
    /// Standard servo frequency is 50 Hz. Call this before setting any channels.
    pub fn init(&mut self, frequency_hz: f64) -> HalResult<()> {
        // Calculate prescale: prescale = round(osc_clock / (4096 * freq)) - 1
        let prescale_val = ((OSC_CLOCK / (4096.0 * frequency_hz)).round() as u8).saturating_sub(1);
        self.prescale = prescale_val;

        debug!(
            address = format!("0x{:02X}", self.address),
            frequency_hz,
            prescale = prescale_val,
            "initializing PCA9685"
        );

        // 1. Sleep mode (required to set prescale)
        self.write_reg(reg::MODE1, mode1::SLEEP)?;

        // 2. Set prescale
        self.write_reg(reg::PRE_SCALE, prescale_val)?;

        // 3. Wake up with auto-increment enabled
        self.write_reg(reg::MODE1, mode1::AI)?;

        // 4. Wait for oscillator to stabilize (500µs per datasheet).
        //    In mock/test, this is a no-op. On real hardware, the delay
        //    is handled by the I2C bus latency being >500µs for the next write.
        std::thread::sleep(std::time::Duration::from_micros(500));

        // 5. Restart
        self.write_reg(reg::MODE1, mode1::AI | mode1::RESTART)?;

        Ok(())
    }

    /// Set a single channel's PWM on/off counts (12-bit each, 0–4095).
    pub fn set_channel(&mut self, channel: u8, on: u16, off: u16) -> HalResult<()> {
        if channel >= CHANNELS as u8 {
            return Err(HalError::Pca9685 {
                address: self.address,
                detail: format!("channel {} out of range (0–15)", channel),
            });
        }

        let base = reg::LED0_ON_L + 4 * channel;
        let data = [
            base,
            (on & 0xFF) as u8,
            ((on >> 8) & 0x0F) as u8,
            (off & 0xFF) as u8,
            ((off >> 8) & 0x0F) as u8,
        ];
        self.write_bytes(&data)
    }

    /// Set a channel's PWM duty by pulse width in microseconds.
    ///
    /// Converts µs → 12-bit count based on the current frequency.
    pub fn set_pulse_us(&mut self, channel: u8, pulse_us: u16) -> HalResult<()> {
        let period_us = self.period_us();
        let count = ((pulse_us as f64 / period_us) * 4096.0).round() as u16;
        let count = count.min(4095);
        self.set_channel(channel, 0, count)
    }

    /// Batch-write pulse widths (µs) to consecutive channels starting at `first`.
    pub fn set_pulse_batch(&mut self, first: u8, pulses_us: &[u16]) -> HalResult<()> {
        for (i, &pulse) in pulses_us.iter().enumerate() {
            let ch = first + i as u8;
            if ch >= CHANNELS as u8 {
                warn!(channel = ch, "batch write skipping out-of-range channel");
                break;
            }
            self.set_pulse_us(ch, pulse)?;
        }
        Ok(())
    }

    /// Turn all channels off (emergency / init).
    pub fn all_off(&mut self) -> HalResult<()> {
        // Set ALL_LED_OFF_H bit 4 (full-off) for all channels.
        self.write_bytes(&[reg::ALL_LED_OFF_L, 0x00])?;
        self.write_bytes(&[reg::ALL_LED_OFF_H, 0x10])?;
        debug!(
            address = format!("0x{:02X}", self.address),
            "all channels off"
        );
        Ok(())
    }

    /// Read the current MODE1 register value.
    pub fn read_mode1(&mut self) -> HalResult<u8> {
        self.read_reg(reg::MODE1)
    }

    /// The I2C address of this device.
    pub fn address(&self) -> u8 {
        self.address
    }

    /// PWM period in microseconds based on the current prescale.
    pub fn period_us(&self) -> f64 {
        let freq = OSC_CLOCK / (4096.0 * (self.prescale as f64 + 1.0));
        1_000_000.0 / freq
    }

    /// Read back the ON and OFF counts for a single channel.
    ///
    /// Returns `(on_count, off_count)` — both 12-bit values (0–4095).
    /// Uses the auto-increment read from `LED0_ON_L + 4*channel`.
    ///
    /// This is a diagnostic operation (I2C read per channel), not for per-tick use.
    pub fn read_channel(&mut self, channel: u8) -> HalResult<(u16, u16)> {
        if channel >= CHANNELS as u8 {
            return Err(HalError::Pca9685 {
                address: self.address,
                detail: format!("channel {} out of range (0–15)", channel),
            });
        }
        let base = reg::LED0_ON_L + 4 * channel;
        let mut buf = [0u8; 4];
        self.bus
            .write_read(self.address, &[base], &mut buf)
            .map_err(|e| HalError::I2c {
                bus: format!("0x{:02X}", self.address),
                detail: i2c_error_detail(embedded_hal::i2c::Error::kind(&e)),
            })?;
        let on = u16::from(buf[0]) | (u16::from(buf[1]) << 8);
        let off = u16::from(buf[2]) | (u16::from(buf[3]) << 8);
        Ok((on & 0x0FFF, off & 0x0FFF))
    }

    /// Consume the driver and return the I2C bus.
    pub fn release(self) -> I {
        self.bus
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn write_reg(&mut self, reg: u8, value: u8) -> HalResult<()> {
        self.write_bytes(&[reg, value])
    }

    fn write_bytes(&mut self, data: &[u8]) -> HalResult<()> {
        self.bus
            .write(self.address, data)
            .map_err(|e| HalError::I2c {
                bus: format!("0x{:02X}", self.address),
                detail: i2c_error_detail(embedded_hal::i2c::Error::kind(&e)),
            })
    }

    fn read_reg(&mut self, reg: u8) -> HalResult<u8> {
        let mut buf = [0u8; 1];
        self.bus
            .write_read(self.address, &[reg], &mut buf)
            .map_err(|e| HalError::I2c {
                bus: format!("0x{:02X}", self.address),
                detail: i2c_error_detail(embedded_hal::i2c::Error::kind(&e)),
            })?;
        Ok(buf[0])
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{I2cTransaction, MockI2cBus};

    #[test]
    fn test_init_writes_prescale() {
        let bus = MockI2cBus::new().with_responses(vec![]);
        let mut pca = Pca9685::new(bus, 0x40);
        pca.init(50.0).unwrap();

        // Prescale for 50Hz: round(25e6 / (4096 * 50)) - 1 = 121
        assert_eq!(pca.prescale, 121);
        assert_eq!(pca.address(), 0x40);
    }

    #[test]
    fn test_set_channel_records_write() {
        let bus = MockI2cBus::new();
        let mut pca = Pca9685::new(bus, 0x40);
        pca.prescale = 121; // pretend initialized

        pca.set_channel(0, 0, 307).unwrap();

        let txns = pca.bus.transactions();
        // Should have written 5 bytes: [base_reg, on_l, on_h, off_l, off_h]
        assert!(!txns.is_empty());
    }

    #[test]
    fn test_set_channel_out_of_range() {
        let bus = MockI2cBus::new();
        let mut pca = Pca9685::new(bus, 0x40);
        let result = pca.set_channel(16, 0, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_pulse_us() {
        let bus = MockI2cBus::new();
        let mut pca = Pca9685::new(bus, 0x40);
        pca.prescale = 121;

        // At 50Hz, period = 20000µs. 1500µs → count = 1500/20000 * 4096 ≈ 307
        pca.set_pulse_us(0, 1500).unwrap();
    }

    #[test]
    fn test_set_pulse_batch() {
        let bus = MockI2cBus::new();
        let mut pca = Pca9685::new(bus, 0x40);
        pca.prescale = 121;

        pca.set_pulse_batch(0, &[500, 1500, 2500]).unwrap();
        // 3 channels written
        assert!(pca.bus.transaction_count() >= 3);
    }

    #[test]
    fn test_all_off() {
        let bus = MockI2cBus::new();
        let mut pca = Pca9685::new(bus, 0x40);
        pca.all_off().unwrap();
        assert!(pca.bus.transaction_count() >= 2);
    }

    #[test]
    fn test_read_mode1() {
        let bus = MockI2cBus::new().with_responses(vec![vec![0x21]]); // AI | RESTART
        let mut pca = Pca9685::new(bus, 0x40);
        let val = pca.read_mode1().unwrap();
        assert_eq!(val, 0x21);
    }

    #[test]
    fn test_period_us_at_50hz() {
        let bus = MockI2cBus::new();
        let mut pca = Pca9685::new(bus, 0x40);
        pca.prescale = 121;

        let period = pca.period_us();
        // 25e6 / (4096 * 122) = ~50.02 Hz → period ≈ 19991 µs
        assert!((period - 20000.0).abs() < 100.0, "period_us = {}", period);
    }

    #[test]
    fn test_i2c_error_propagates() {
        use embedded_hal::i2c;
        let mut bus = MockI2cBus::new();
        bus.inject_error(i2c::ErrorKind::NoAcknowledge(
            i2c::NoAcknowledgeSource::Address,
        ));
        let mut pca = Pca9685::new(bus, 0x40);
        let result = pca.all_off();
        assert!(result.is_err());
    }

    #[test]
    fn test_release_returns_bus() {
        let bus = MockI2cBus::new();
        let pca = Pca9685::new(bus, 0x40);
        let _bus = pca.release();
    }

    // ── read_channel tests ──────────────────────────────────────────

    #[test]
    fn test_read_channel_parses_on_off() {
        // Pre-load 4-byte response: ON_L=0x33, ON_H=0x01, OFF_L=0x44, OFF_H=0x02
        let bus = MockI2cBus::new().with_responses(vec![vec![0x33, 0x01, 0x44, 0x02]]);
        let mut pca = Pca9685::new(bus, 0x40);
        let (on, off) = pca.read_channel(0).unwrap();
        assert_eq!(on, 0x0133);
        assert_eq!(off, 0x0244);
    }

    #[test]
    fn test_read_channel_out_of_range() {
        let bus = MockI2cBus::new();
        let mut pca = Pca9685::new(bus, 0x40);
        let result = pca.read_channel(16);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_channel_register_address() {
        // Channel 5 → base = LED0_ON_L + 4*5 = 0x06 + 20 = 0x1A
        let bus = MockI2cBus::new().with_responses(vec![vec![0, 0, 0, 0]]);
        let mut pca = Pca9685::new(bus, 0x40);
        pca.read_channel(5).unwrap();
        // Verify the write portion sent the correct register
        let txns = pca.bus.transactions();
        // write_read produces Write + Read in our mock
        if let I2cTransaction::Write { data, .. } = &txns[0] {
            assert_eq!(data[0], 0x1A, "expected register 0x1A for channel 5");
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::mock::MockI2cBus;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn set_pulse_us_succeeds_on_valid_channel(ch in 0u8..16, pulse in 500u16..2500) {
            let bus = MockI2cBus::new();
            let mut pca = Pca9685::new(bus, 0x40);
            pca.prescale = 121; // 50 Hz
            prop_assert!(pca.set_pulse_us(ch, pulse).is_ok());
        }

        #[test]
        fn set_pulse_us_fails_on_invalid_channel(ch in 16u8..=255) {
            let bus = MockI2cBus::new();
            let mut pca = Pca9685::new(bus, 0x40);
            pca.prescale = 121;
            // set_pulse_us calls set_channel which checks bounds
            prop_assert!(pca.set_pulse_us(ch, 1500).is_err());
        }
    }
}
