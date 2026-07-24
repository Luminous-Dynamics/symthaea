// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HAL error types.

/// Errors from the hardware abstraction layer.
#[derive(Debug, thiserror::Error)]
pub enum HalError {
    /// I2C bus communication failure.
    #[error("I2C error on bus {bus}: {detail}")]
    I2c { bus: String, detail: String },

    /// Safety interlock tripped.
    #[error("safety interlock: {0}")]
    Safety(String),

    /// Watchdog timeout — no command received within deadline.
    #[error("watchdog timeout: no command for {elapsed_ms}ms (limit {limit_ms}ms)")]
    WatchdogTimeout { elapsed_ms: u64, limit_ms: u64 },

    /// Overcurrent detected on a joint.
    #[error("overcurrent on joint {joint} ({name}): {amps:.2}A exceeds {limit:.2}A")]
    Overcurrent {
        joint: usize,
        name: &'static str,
        amps: f32,
        limit: f32,
    },

    /// Emergency stop is active.
    #[error("e-stop active")]
    EStop,

    /// Joint angle out of mechanical bounds.
    #[error("joint {joint} ({name}) angle {angle:.1}° exceeds bounds [{min:.1}°, {max:.1}°]")]
    AngleBounds {
        joint: usize,
        name: &'static str,
        angle: f32,
        min: f32,
        max: f32,
    },

    /// Calibration profile error (missing joint, bad JSON, etc.).
    #[error("calibration error: {0}")]
    Calibration(String),

    /// Sensor configuration, identity, or sample validation failure.
    #[error("sensor '{name}': {detail}")]
    Sensor { name: String, detail: String },

    /// PCA9685 device not responding or misconfigured.
    #[error("PCA9685 at address 0x{address:02X}: {detail}")]
    Pca9685 { address: u8, detail: String },
}

/// Convenience alias.
pub type HalResult<T> = Result<T, HalError>;

/// Convert an `embedded_hal::i2c::ErrorKind` into a string for `HalError::I2c`.
pub(crate) fn i2c_error_detail(kind: embedded_hal::i2c::ErrorKind) -> String {
    // ErrorKind does not implement Display in embedded-hal 1.0, so we format via Debug.
    format!("{:?}", kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = HalError::I2c {
            bus: "/dev/i2c-1".into(),
            detail: "NACK".into(),
        };
        assert!(e.to_string().contains("I2C error"));
        assert!(e.to_string().contains("NACK"));
    }

    #[test]
    fn test_estop_display() {
        let e = HalError::EStop;
        assert_eq!(e.to_string(), "e-stop active");
    }

    #[test]
    fn test_watchdog_display() {
        let e = HalError::WatchdogTimeout {
            elapsed_ms: 150,
            limit_ms: 100,
        };
        assert!(e.to_string().contains("150ms"));
    }

    #[test]
    fn test_overcurrent_display() {
        let e = HalError::Overcurrent {
            joint: 3,
            name: "right_hip_x",
            amps: 2.5,
            limit: 2.0,
        };
        assert!(e.to_string().contains("right_hip_x"));
        assert!(e.to_string().contains("2.50A"));
    }

    #[test]
    fn test_angle_bounds_display() {
        let e = HalError::AngleBounds {
            joint: 6,
            name: "right_knee",
            angle: 185.0,
            min: -10.0,
            max: 180.0,
        };
        assert!(e.to_string().contains("right_knee"));
        assert!(e.to_string().contains("185.0"));
    }
}
