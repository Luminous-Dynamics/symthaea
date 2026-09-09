// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Measurement-only temporal alignment evidence for universal sensorimotor state.
//!
//! R3.9 exists because physical semantic identity is insufficient for temporal
//! learning: an LTC/CfC dynamics model also needs to know whether two samples
//! inhabit a declared common clock domain and what positive physical `dt`
//! separates them. This module never infers synchronization from raw timestamp
//! units or matching numeric values.

use super::sensor_fusion::SemanticImuFrameV1;
use symthaea_humanoid::SemanticHumanoidFrameV1;

/// Explicit clock identity for one observation stream.
///
/// Every variant includes a domain identifier. Even Unix-epoch timestamps from
/// two machines are not considered comparable merely because they share an epoch
/// convention; callers must explicitly place them in the same synchronization
/// domain. That declaration is metadata, not proof of bounded clock error.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SensorimotorClockDomainV1 {
    UnixEpoch(String),
    Monotonic(String),
    Simulation(String),
    Custom {
        namespace: String,
        synchronization_domain_id: String,
    },
}

impl SensorimotorClockDomainV1 {
    pub fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::UnixEpoch(domain_id)
            | Self::Monotonic(domain_id)
            | Self::Simulation(domain_id) => {
                if domain_id.trim().is_empty() {
                    Err("clock synchronization-domain id must not be empty")
                } else {
                    Ok(())
                }
            }
            Self::Custom {
                namespace,
                synchronization_domain_id,
            } => {
                if namespace.trim().is_empty() {
                    return Err("custom clock-domain namespace must not be empty");
                }
                if synchronization_domain_id.trim().is_empty() {
                    return Err("clock synchronization-domain id must not be empty");
                }
                Ok(())
            }
        }
    }
}

/// One capture coordinate with an explicitly declared clock domain.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorimotorCaptureTimeV1 {
    pub schema_id: String,
    pub clock_domain: SensorimotorClockDomainV1,
    pub capture_seconds: f64,
    /// Declared timestamp granularity when the source representation makes it
    /// explicit. This is not a bound on physical sensor latency, synchronization
    /// error, transport latency, or capture uncertainty.
    pub nominal_resolution_seconds: Option<f64>,
}

impl SensorimotorCaptureTimeV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.sensorimotor.capture-time.v1";

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err("unsupported sensorimotor capture-time schema");
        }
        self.clock_domain.validate()?;
        if !self.capture_seconds.is_finite() || self.capture_seconds < 0.0 {
            return Err("capture time must be finite and non-negative");
        }
        if let Some(resolution) = self.nominal_resolution_seconds {
            if !resolution.is_finite() || resolution <= 0.0 {
                return Err("nominal timestamp resolution must be finite and positive");
            }
        }
        Ok(())
    }

    /// Build an explicit time coordinate for an R3.2 IMU semantic frame.
    ///
    /// The caller MUST supply the clock domain. The current `ImuReading` type
    /// exposes microseconds but its documentation does not establish one
    /// unambiguous clock interpretation across producers.
    pub fn from_imu(
        frame: &SemanticImuFrameV1,
        clock_domain: SensorimotorClockDomainV1,
    ) -> Result<Self, &'static str> {
        clock_domain.validate()?;
        let value = Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            clock_domain,
            capture_seconds: frame.capture_timestamp_us as f64 / 1_000_000.0,
            nominal_resolution_seconds: Some(1.0e-6),
        };
        value.validate()?;
        Ok(value)
    }

    /// Build an explicit time coordinate for an R3.3 humanoid semantic frame.
    ///
    /// The humanoid source currently carries simulation seconds, but the caller
    /// still supplies a concrete simulation-domain identity so independent
    /// simulators/runs are never assumed synchronized by unit alone.
    pub fn from_humanoid(
        frame: &SemanticHumanoidFrameV1,
        clock_domain: SensorimotorClockDomainV1,
    ) -> Result<Self, String> {
        frame.validate().map_err(|error| error.to_string())?;
        clock_domain.validate().map_err(str::to_string)?;
        let value = Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            clock_domain,
            capture_seconds: frame.timestamp_seconds,
            nominal_resolution_seconds: None,
        };
        value.validate().map_err(str::to_string)?;
        Ok(value)
    }
}

/// Evidence for the temporal relationship between two capture coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorimotorTemporalAlignmentEvidenceV1 {
    pub schema_id: String,
    pub source: SensorimotorCaptureTimeV1,
    pub target: SensorimotorCaptureTimeV1,
    /// Caller-chosen tolerance for treating two samples as co-temporal for a
    /// sensor-fusion/alignment experiment. It is independent from whether the
    /// pair forms a valid forward temporal transition.
    pub coobserved_tolerance_seconds: f64,
    /// True when subtraction is meaningful under the caller-declared identical
    /// clock-domain identity. This does not prove actual synchronization error.
    pub comparable: bool,
    /// `target - source` when the domains match; otherwise `None`.
    pub signed_delta_seconds: Option<f64>,
    pub absolute_delta_seconds: Option<f64>,
    /// Co-temporality predicate only. A valid LTC transition may intentionally
    /// be outside this tolerance and still have a valid positive `dt`.
    pub coobserved_within_tolerance: bool,
}

impl SensorimotorTemporalAlignmentEvidenceV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.sensorimotor.temporal-alignment.v1";

    /// Positive forward `dt` suitable for a temporal prediction transition.
    ///
    /// This deliberately rejects cross-domain, reversed, and duplicate-time
    /// pairs rather than teaching an LTC model an invented or zero-duration step.
    /// It is intentionally independent from `coobserved_within_tolerance`.
    pub fn positive_forward_dt_seconds(&self) -> Result<f64, &'static str> {
        if !self.comparable {
            return Err("capture times are from different declared clock domains");
        }
        let dt = self
            .signed_delta_seconds
            .ok_or("comparable temporal evidence must carry a delta")?;
        if dt <= 0.0 {
            return Err("temporal prediction requires positive forward dt");
        }
        Ok(dt)
    }
}

/// Compare two explicit capture coordinates without performing clock conversion
/// or claiming that a shared domain identifier proves clock synchronization.
pub fn measure_temporal_alignment_v1(
    source: SensorimotorCaptureTimeV1,
    target: SensorimotorCaptureTimeV1,
    coobserved_tolerance_seconds: f64,
) -> Result<SensorimotorTemporalAlignmentEvidenceV1, &'static str> {
    source.validate()?;
    target.validate()?;
    if !coobserved_tolerance_seconds.is_finite() || coobserved_tolerance_seconds < 0.0 {
        return Err("coobserved temporal tolerance must be finite and non-negative");
    }

    if source.clock_domain != target.clock_domain {
        return Ok(SensorimotorTemporalAlignmentEvidenceV1 {
            schema_id: SensorimotorTemporalAlignmentEvidenceV1::SCHEMA_ID.to_string(),
            source,
            target,
            coobserved_tolerance_seconds,
            comparable: false,
            signed_delta_seconds: None,
            absolute_delta_seconds: None,
            coobserved_within_tolerance: false,
        });
    }

    let delta = target.capture_seconds - source.capture_seconds;
    let absolute_delta = delta.abs();
    Ok(SensorimotorTemporalAlignmentEvidenceV1 {
        schema_id: SensorimotorTemporalAlignmentEvidenceV1::SCHEMA_ID.to_string(),
        source,
        target,
        coobserved_tolerance_seconds,
        comparable: true,
        signed_delta_seconds: Some(delta),
        absolute_delta_seconds: Some(absolute_delta),
        coobserved_within_tolerance: absolute_delta <= coobserved_tolerance_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::sensor_fusion::ImuReading;
    use symthaea_humanoid::{HumanoidMorphology, HumanoidState, SemanticHumanoidEncoderV1};

    fn humanoid_frame(timestamp_seconds: f64) -> SemanticHumanoidFrameV1 {
        let morphology = HumanoidMorphology::Dmc21;
        let mut state = HumanoidState::default_for(morphology);
        state.timestamp = timestamp_seconds;
        SemanticHumanoidEncoderV1::new()
            .frame(&state, morphology)
            .unwrap()
    }

    #[test]
    fn explicit_shared_simulation_domain_allows_cross_adapter_time_comparison() {
        let imu = SemanticImuFrameV1::from_reading(&ImuReading {
            accel: [0.0; 3],
            gyro: [0.0; 3],
            timestamp_us: 1_500_001,
        });
        let humanoid = humanoid_frame(1.5);
        let domain = SensorimotorClockDomainV1::Simulation("shared-testbed".into());

        let source = SensorimotorCaptureTimeV1::from_humanoid(&humanoid, domain.clone()).unwrap();
        let target = SensorimotorCaptureTimeV1::from_imu(&imu, domain).unwrap();
        let evidence = measure_temporal_alignment_v1(source, target, 2.0e-6).unwrap();

        assert!(evidence.comparable);
        assert!(evidence.coobserved_within_tolerance);
        let dt = evidence.positive_forward_dt_seconds().unwrap();
        assert!((dt - 1.0e-6).abs() < 1.0e-12);
    }

    #[test]
    fn different_clock_domains_are_not_comparable_even_when_numbers_match() {
        let imu = SemanticImuFrameV1::from_reading(&ImuReading {
            accel: [0.0; 3],
            gyro: [0.0; 3],
            timestamp_us: 1_000_000,
        });
        let humanoid = humanoid_frame(1.0);

        let source = SensorimotorCaptureTimeV1::from_imu(
            &imu,
            SensorimotorClockDomainV1::UnixEpoch("ptp-domain-a".into()),
        )
        .unwrap();
        let target = SensorimotorCaptureTimeV1::from_humanoid(
            &humanoid,
            SensorimotorClockDomainV1::Simulation("sim-a".into()),
        )
        .unwrap();
        let evidence = measure_temporal_alignment_v1(source, target, 0.0).unwrap();

        assert!(!evidence.comparable);
        assert_eq!(evidence.signed_delta_seconds, None);
        assert!(!evidence.coobserved_within_tolerance);
        assert!(evidence.positive_forward_dt_seconds().is_err());
    }

    #[test]
    fn same_epoch_convention_does_not_align_independent_sync_domains() {
        let base = |domain_id: &str| SensorimotorCaptureTimeV1 {
            schema_id: SensorimotorCaptureTimeV1::SCHEMA_ID.to_string(),
            clock_domain: SensorimotorClockDomainV1::UnixEpoch(domain_id.into()),
            capture_seconds: 1_800_000_000.0,
            nominal_resolution_seconds: Some(1.0e-6),
        };
        assert!(
            !measure_temporal_alignment_v1(base("host-a"), base("host-b"), 0.0)
                .unwrap()
                .comparable
        );
    }

    #[test]
    fn independent_monotonic_clocks_do_not_align_by_unit_alone() {
        let a = SensorimotorCaptureTimeV1 {
            schema_id: SensorimotorCaptureTimeV1::SCHEMA_ID.to_string(),
            clock_domain: SensorimotorClockDomainV1::Monotonic("robot-a".into()),
            capture_seconds: 5.0,
            nominal_resolution_seconds: Some(1.0e-6),
        };
        let b = SensorimotorCaptureTimeV1 {
            schema_id: SensorimotorCaptureTimeV1::SCHEMA_ID.to_string(),
            clock_domain: SensorimotorClockDomainV1::Monotonic("robot-b".into()),
            capture_seconds: 5.0,
            nominal_resolution_seconds: Some(1.0e-6),
        };
        assert!(!measure_temporal_alignment_v1(a, b, 0.0).unwrap().comparable);
    }

    #[test]
    fn forward_prediction_dt_is_independent_of_cotemporal_tolerance() {
        let domain = SensorimotorClockDomainV1::Simulation("sim".into());
        let base = |seconds| SensorimotorCaptureTimeV1 {
            schema_id: SensorimotorCaptureTimeV1::SCHEMA_ID.to_string(),
            clock_domain: domain.clone(),
            capture_seconds: seconds,
            nominal_resolution_seconds: None,
        };

        let evidence = measure_temporal_alignment_v1(base(1.0), base(1.020), 0.001).unwrap();
        assert!(!evidence.coobserved_within_tolerance);
        assert!((evidence.positive_forward_dt_seconds().unwrap() - 0.020).abs() < 1.0e-12);
    }

    #[test]
    fn reversed_and_duplicate_time_pairs_are_not_valid_prediction_steps() {
        let domain = SensorimotorClockDomainV1::Simulation("sim".into());
        let base = |seconds| SensorimotorCaptureTimeV1 {
            schema_id: SensorimotorCaptureTimeV1::SCHEMA_ID.to_string(),
            clock_domain: domain.clone(),
            capture_seconds: seconds,
            nominal_resolution_seconds: None,
        };

        let reversed = measure_temporal_alignment_v1(base(2.0), base(1.0), 2.0).unwrap();
        assert!(reversed.comparable);
        assert!(reversed.positive_forward_dt_seconds().is_err());

        let duplicate = measure_temporal_alignment_v1(base(1.0), base(1.0), 0.0).unwrap();
        assert!(duplicate.coobserved_within_tolerance);
        assert!(duplicate.positive_forward_dt_seconds().is_err());
    }

    #[test]
    fn invalid_domain_and_tolerance_fail_closed() {
        assert!(SensorimotorClockDomainV1::Simulation(String::new()).validate().is_err());
        assert!(SensorimotorClockDomainV1::UnixEpoch(String::new()).validate().is_err());
        let time = SensorimotorCaptureTimeV1 {
            schema_id: SensorimotorCaptureTimeV1::SCHEMA_ID.to_string(),
            clock_domain: SensorimotorClockDomainV1::UnixEpoch("test-domain".into()),
            capture_seconds: 1.0,
            nominal_resolution_seconds: Some(1.0e-6),
        };
        assert!(measure_temporal_alignment_v1(time.clone(), time, f64::NAN).is_err());
    }

    #[test]
    fn imu_constructor_exposes_resolution_without_assuming_clock_semantics() {
        let imu = SemanticImuFrameV1::from_reading(&ImuReading {
            accel: [0.0; 3],
            gyro: [0.0; 3],
            timestamp_us: 123,
        });
        let time = SensorimotorCaptureTimeV1::from_imu(
            &imu,
            SensorimotorClockDomainV1::Custom {
                namespace: "bench-clock".into(),
                synchronization_domain_id: "shared-capture-domain".into(),
            },
        )
        .unwrap();
        assert_eq!(time.nominal_resolution_seconds, Some(1.0e-6));
        assert!((time.capture_seconds - 123.0e-6).abs() < 1.0e-15);
    }
}
