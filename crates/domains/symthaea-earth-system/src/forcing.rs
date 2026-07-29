// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Small, explicit radiative-forcing protocols for transient experiments.
//!
//! Protocols are deterministic functions of elapsed time. They carry no claim
//! of being emissions scenarios or observational reconstructions; callers are
//! responsible for the provenance and interpretation of the forcing values.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};

/// A bounded set of forcing protocols useful for reproducible experiments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ForcingProtocol {
    /// Constant forcing in W/m².
    Constant { forcing: f64 },
    /// Linear transition from `initial` to `final_forcing`, then a hold.
    LinearRamp {
        initial: f64,
        final_forcing: f64,
        duration_seconds: f64,
    },
    /// Rectangular anomaly added to a baseline over `[start, end)`.
    Pulse {
        baseline: f64,
        anomaly: f64,
        start_seconds: f64,
        end_seconds: f64,
    },
    /// Smooth periodic forcing in W/m².
    Sinusoidal {
        mean: f64,
        amplitude: f64,
        period_seconds: f64,
        phase_radians: f64,
    },
}

impl ForcingProtocol {
    pub fn constant(forcing: f64) -> Result<Self, ModelError> {
        let protocol = Self::Constant { forcing };
        protocol.validate()?;
        Ok(protocol)
    }

    pub fn linear_ramp(
        initial: f64,
        final_forcing: f64,
        duration_seconds: f64,
    ) -> Result<Self, ModelError> {
        let protocol = Self::LinearRamp {
            initial,
            final_forcing,
            duration_seconds,
        };
        protocol.validate()?;
        Ok(protocol)
    }

    pub fn pulse(
        baseline: f64,
        anomaly: f64,
        start_seconds: f64,
        end_seconds: f64,
    ) -> Result<Self, ModelError> {
        let protocol = Self::Pulse {
            baseline,
            anomaly,
            start_seconds,
            end_seconds,
        };
        protocol.validate()?;
        Ok(protocol)
    }

    pub fn sinusoidal(
        mean: f64,
        amplitude: f64,
        period_seconds: f64,
        phase_radians: f64,
    ) -> Result<Self, ModelError> {
        let protocol = Self::Sinusoidal {
            mean,
            amplitude,
            period_seconds,
            phase_radians,
        };
        protocol.validate()?;
        Ok(protocol)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        match *self {
            Self::Constant { forcing } => require_finite("forcing", forcing),
            Self::LinearRamp {
                initial,
                final_forcing,
                duration_seconds,
            } => {
                require_finite("initial_forcing", initial)?;
                require_finite("final_forcing", final_forcing)?;
                require_positive("duration_seconds", duration_seconds)
            }
            Self::Pulse {
                baseline,
                anomaly,
                start_seconds,
                end_seconds,
            } => {
                require_finite("baseline_forcing", baseline)?;
                require_finite("forcing_anomaly", anomaly)?;
                require_non_negative("start_seconds", start_seconds)?;
                require_non_negative("end_seconds", end_seconds)?;
                if start_seconds >= end_seconds {
                    return Err(ModelError::InvalidOrdering {
                        lower: "start_seconds",
                        lower_value: start_seconds,
                        upper: "end_seconds",
                        upper_value: end_seconds,
                    });
                }
                Ok(())
            }
            Self::Sinusoidal {
                mean,
                amplitude,
                period_seconds,
                phase_radians,
            } => {
                require_finite("mean_forcing", mean)?;
                require_finite("forcing_amplitude", amplitude)?;
                require_positive("period_seconds", period_seconds)?;
                require_finite("phase_radians", phase_radians)
            }
        }
    }

    /// Times where the protocol changes value or slope. Event-aligned
    /// integrators split exactly at these points.
    pub fn integration_events(&self) -> Vec<f64> {
        match *self {
            Self::Constant { .. } | Self::Sinusoidal { .. } => Vec::new(),
            Self::LinearRamp {
                duration_seconds, ..
            } => vec![duration_seconds],
            Self::Pulse {
                start_seconds,
                end_seconds,
                ..
            } => {
                vec![start_seconds, end_seconds]
            }
        }
    }

    /// Left-hand value used by an RK4 endpoint that lands exactly on a
    /// discontinuity. At ordinary times this equals [`Self::at`].
    pub fn at_left_limit(&self, time_seconds: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_non_negative("time_seconds", time_seconds)?;
        Ok(match *self {
            Self::Pulse {
                baseline,
                anomaly,
                start_seconds: _,
                end_seconds,
            } if time_seconds == end_seconds => baseline + anomaly,
            Self::Pulse {
                baseline,
                start_seconds,
                ..
            } if time_seconds == start_seconds => baseline,
            _ => return self.at(time_seconds),
        })
    }

    /// Evaluate the forcing at non-negative elapsed time, in W/m².
    pub fn at(&self, time_seconds: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_non_negative("time_seconds", time_seconds)?;
        Ok(match *self {
            Self::Constant { forcing } => forcing,
            Self::LinearRamp {
                initial,
                final_forcing,
                duration_seconds,
            } => {
                let fraction = (time_seconds / duration_seconds).clamp(0.0, 1.0);
                initial + fraction * (final_forcing - initial)
            }
            Self::Pulse {
                baseline,
                anomaly,
                start_seconds,
                end_seconds,
            } => {
                if time_seconds >= start_seconds && time_seconds < end_seconds {
                    baseline + anomaly
                } else {
                    baseline
                }
            }
            Self::Sinusoidal {
                mean,
                amplitude,
                period_seconds,
                phase_radians,
            } => {
                let angle = core::f64::consts::TAU * time_seconds / period_seconds + phase_radians;
                mean + amplitude * angle.sin()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_ramp_interpolates_and_holds() {
        let p = ForcingProtocol::linear_ramp(0.0, 4.0, 10.0).unwrap();
        assert_eq!(p.at(0.0).unwrap(), 0.0);
        assert!((p.at(2.5).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(p.at(10.0).unwrap(), 4.0);
        assert_eq!(p.at(100.0).unwrap(), 4.0);
    }

    #[test]
    fn pulse_is_half_open() {
        let p = ForcingProtocol::pulse(1.0, 3.0, 5.0, 8.0).unwrap();
        assert_eq!(p.at(4.999).unwrap(), 1.0);
        assert_eq!(p.at(5.0).unwrap(), 4.0);
        assert_eq!(p.at(7.999).unwrap(), 4.0);
        assert_eq!(p.at(8.0).unwrap(), 1.0);
    }

    #[test]
    fn event_metadata_and_left_limits_are_explicit() {
        let pulse = ForcingProtocol::pulse(1.0, 3.0, 5.0, 8.0).unwrap();
        assert_eq!(pulse.integration_events(), vec![5.0, 8.0]);
        assert_eq!(pulse.at_left_limit(5.0).unwrap(), 1.0);
        assert_eq!(pulse.at(5.0).unwrap(), 4.0);
        assert_eq!(pulse.at_left_limit(8.0).unwrap(), 4.0);
        assert_eq!(pulse.at(8.0).unwrap(), 1.0);
    }

    #[test]
    fn sinusoidal_forcing_has_declared_period_and_mean() {
        let p = ForcingProtocol::sinusoidal(2.0, 1.5, 10.0, 0.0).unwrap();
        assert!((p.at(0.0).unwrap() - 2.0).abs() < 1e-12);
        assert!((p.at(2.5).unwrap() - 3.5).abs() < 1e-12);
        assert!((p.at(7.5).unwrap() - 0.5).abs() < 1e-12);
        assert!((p.at(10.0).unwrap() - 2.0).abs() < 1e-12);
        assert!(p.integration_events().is_empty());
    }

    #[test]
    fn invalid_protocols_are_rejected() {
        assert!(ForcingProtocol::constant(f64::NAN).is_err());
        assert!(ForcingProtocol::linear_ramp(0.0, 1.0, 0.0).is_err());
        assert!(ForcingProtocol::pulse(0.0, 1.0, 2.0, 2.0).is_err());
        assert!(ForcingProtocol::sinusoidal(0.0, 1.0, 0.0, 0.0).is_err());
    }
}
