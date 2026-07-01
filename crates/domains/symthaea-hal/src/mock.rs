// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mock I2C bus and HAL sensor for testing without hardware.

use embedded_hal::digital;
use embedded_hal::i2c::{self, ErrorType, I2c, Operation};
use parking_lot::Mutex;
use std::sync::Arc;

use crate::sensor::HalSensorAdapter;

// ============================================================================
// MOCK I2C BUS
// ============================================================================

/// Recorded I2C transaction for verification.
#[derive(Debug, Clone, PartialEq)]
pub enum I2cTransaction {
    /// Write to address.
    Write { address: u8, data: Vec<u8> },
    /// Read from address (returns the provided data).
    Read { address: u8, len: usize },
    /// Write-then-read (register read pattern).
    WriteRead {
        address: u8,
        write: Vec<u8>,
        read_len: usize,
    },
}

/// Mock I2C error.
#[derive(Debug, Clone)]
pub struct MockI2cError {
    pub kind: i2c::ErrorKind,
}

impl i2c::Error for MockI2cError {
    fn kind(&self) -> i2c::ErrorKind {
        self.kind
    }
}

impl std::fmt::Display for MockI2cError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MockI2cError({:?})", self.kind)
    }
}

/// Mock I2C bus implementing `embedded_hal::i2c::I2c`.
///
/// Records all transactions for later verification and returns
/// pre-configured responses for reads.
pub struct MockI2cBus {
    /// Pre-loaded read responses. Each `write_read` or `read` pops the front.
    read_responses: Vec<Vec<u8>>,
    read_index: usize,
    /// Recorded transactions.
    transactions: Vec<I2cTransaction>,
    /// If set, the next operation will return this error.
    inject_error: Option<i2c::ErrorKind>,
}

impl MockI2cBus {
    /// Create a new mock bus with no pre-loaded responses.
    pub fn new() -> Self {
        Self {
            read_responses: Vec::new(),
            read_index: 0,
            transactions: Vec::new(),
            inject_error: None,
        }
    }

    /// Pre-load read responses (consumed in order).
    pub fn with_responses(mut self, responses: Vec<Vec<u8>>) -> Self {
        self.read_responses = responses;
        self
    }

    /// Inject an error for the next operation.
    pub fn inject_error(&mut self, kind: i2c::ErrorKind) {
        self.inject_error = Some(kind);
    }

    /// Get recorded transactions.
    pub fn transactions(&self) -> &[I2cTransaction] {
        &self.transactions
    }

    /// Number of recorded transactions.
    pub fn transaction_count(&self) -> usize {
        self.transactions.len()
    }

    /// Clear recorded transactions.
    pub fn clear_transactions(&mut self) {
        self.transactions.clear();
    }

    fn check_error(&mut self) -> Result<(), MockI2cError> {
        if let Some(kind) = self.inject_error.take() {
            Err(MockI2cError { kind })
        } else {
            Ok(())
        }
    }

    fn next_response(&mut self, len: usize) -> Vec<u8> {
        if self.read_index < self.read_responses.len() {
            let resp = self.read_responses[self.read_index].clone();
            self.read_index += 1;
            resp
        } else {
            vec![0u8; len]
        }
    }
}

impl Default for MockI2cBus {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorType for MockI2cBus {
    type Error = MockI2cError;
}

impl I2c for MockI2cBus {
    fn transaction(
        &mut self,
        address: u8,
        operations: &mut [Operation<'_>],
    ) -> Result<(), Self::Error> {
        self.check_error()?;

        for op in operations {
            match op {
                Operation::Write(data) => {
                    self.transactions.push(I2cTransaction::Write {
                        address,
                        data: data.to_vec(),
                    });
                }
                Operation::Read(buf) => {
                    let resp = self.next_response(buf.len());
                    let copy_len = buf.len().min(resp.len());
                    buf[..copy_len].copy_from_slice(&resp[..copy_len]);
                    self.transactions.push(I2cTransaction::Read {
                        address,
                        len: buf.len(),
                    });
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// MOCK HAL SENSOR
// ============================================================================

/// Mock HAL sensor for testing the sensor pipeline without hardware.
pub struct MockHalSensor {
    name: String,
    readings: Arc<Mutex<Vec<Vec<f32>>>>,
    index: usize,
}

impl MockHalSensor {
    /// Create a mock sensor with pre-loaded readings.
    pub fn new(name: impl Into<String>, readings: Vec<Vec<f32>>) -> Self {
        Self {
            name: name.into(),
            readings: Arc::new(Mutex::new(readings)),
            index: 0,
        }
    }
}

impl HalSensorAdapter for MockHalSensor {
    fn name(&self) -> &str {
        &self.name
    }

    fn read_raw(&mut self) -> Option<Vec<f32>> {
        let readings = self.readings.lock();
        if self.index < readings.len() {
            let data = readings[self.index].clone();
            self.index += 1;
            Some(data)
        } else {
            None
        }
    }

    fn is_available(&self) -> bool {
        let readings = self.readings.lock();
        self.index < readings.len()
    }
}

// ============================================================================
// MOCK GPIO INPUT PIN
// ============================================================================

/// Mock GPIO pin error.
#[derive(Debug)]
pub struct MockPinError;

impl digital::Error for MockPinError {
    fn kind(&self) -> digital::ErrorKind {
        digital::ErrorKind::Other
    }
}

impl std::fmt::Display for MockPinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MockPinError")
    }
}

/// Mock GPIO input pin for testing [`GpioEstop`](crate::gpio_estop::GpioEstop).
///
/// Can be configured to return a fixed state or an error.
pub struct MockInputPin {
    is_low: Option<bool>,
}

impl MockInputPin {
    /// Create a pin that returns the given low state.
    pub fn new(is_low: bool) -> Self {
        Self {
            is_low: Some(is_low),
        }
    }

    /// Create a pin that always returns an error.
    pub fn with_error() -> Self {
        Self { is_low: None }
    }
}

impl digital::ErrorType for MockInputPin {
    type Error = MockPinError;
}

impl digital::InputPin for MockInputPin {
    fn is_high(&mut self) -> Result<bool, Self::Error> {
        match self.is_low {
            Some(low) => Ok(!low),
            None => Err(MockPinError),
        }
    }

    fn is_low(&mut self) -> Result<bool, Self::Error> {
        match self.is_low {
            Some(low) => Ok(low),
            None => Err(MockPinError),
        }
    }
}

// ============================================================================
// MOCK ESTOP POLLER
// ============================================================================

/// Mock e-stop poller for testing the runtime without GPIO hardware.
pub struct MockEstopPoller {
    /// If `true`, the next `poll()` returns `true` (e-stop triggered).
    pub triggered: bool,
}

impl MockEstopPoller {
    /// Create a mock e-stop poller.
    pub fn new(triggered: bool) -> Self {
        Self { triggered }
    }
}

impl crate::gpio_estop::EstopPoller for MockEstopPoller {
    fn poll(&mut self) -> bool {
        self.triggered
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_bus_write() {
        let mut bus = MockI2cBus::new();
        bus.transaction(0x40, &mut [Operation::Write(&[0x00, 0x10])])
            .unwrap();

        assert_eq!(bus.transaction_count(), 1);
        assert_eq!(
            bus.transactions()[0],
            I2cTransaction::Write {
                address: 0x40,
                data: vec![0x00, 0x10],
            }
        );
    }

    #[test]
    fn test_mock_bus_read_with_responses() {
        let mut bus = MockI2cBus::new().with_responses(vec![vec![0xAB, 0xCD]]);
        let mut buf = [0u8; 2];
        bus.transaction(0x40, &mut [Operation::Read(&mut buf)])
            .unwrap();

        assert_eq!(buf, [0xAB, 0xCD]);
    }

    #[test]
    fn test_mock_bus_write_read() {
        let mut bus = MockI2cBus::new().with_responses(vec![vec![0x42]]);
        let mut buf = [0u8; 1];
        bus.transaction(
            0x40,
            &mut [Operation::Write(&[0x00]), Operation::Read(&mut buf)],
        )
        .unwrap();

        assert_eq!(buf, [0x42]);
        assert_eq!(bus.transaction_count(), 2);
    }

    #[test]
    fn test_mock_bus_error_injection() {
        let mut bus = MockI2cBus::new();
        bus.inject_error(i2c::ErrorKind::NoAcknowledge(
            i2c::NoAcknowledgeSource::Address,
        ));

        let result = bus.transaction(0x40, &mut [Operation::Write(&[0x00])]);
        assert!(result.is_err());

        // Error consumed, next operation succeeds
        let result = bus.transaction(0x40, &mut [Operation::Write(&[0x00])]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_bus_default_read_zeros() {
        let mut bus = MockI2cBus::new();
        let mut buf = [0xFF; 4];
        bus.transaction(0x40, &mut [Operation::Read(&mut buf)])
            .unwrap();
        assert_eq!(buf, [0, 0, 0, 0]);
    }

    #[test]
    fn test_mock_sensor_read_sequence() {
        let mut sensor = MockHalSensor::new("imu", vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        assert!(sensor.is_available());
        assert_eq!(sensor.name(), "imu");

        let r1 = sensor.read_raw().unwrap();
        assert_eq!(r1, vec![1.0, 2.0]);

        let r2 = sensor.read_raw().unwrap();
        assert_eq!(r2, vec![3.0, 4.0]);

        assert!(!sensor.is_available());
        assert!(sensor.read_raw().is_none());
    }

    #[test]
    fn test_mock_input_pin_low() {
        use embedded_hal::digital::InputPin;
        let mut pin = MockInputPin::new(true);
        assert!(pin.is_low().unwrap());
        assert!(!pin.is_high().unwrap());
    }

    #[test]
    fn test_mock_input_pin_high() {
        use embedded_hal::digital::InputPin;
        let mut pin = MockInputPin::new(false);
        assert!(!pin.is_low().unwrap());
        assert!(pin.is_high().unwrap());
    }

    #[test]
    fn test_mock_input_pin_error() {
        use embedded_hal::digital::InputPin;
        let mut pin = MockInputPin::with_error();
        assert!(pin.is_low().is_err());
        assert!(pin.is_high().is_err());
    }
}
