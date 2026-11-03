#![forbid(unsafe_code)]

//! Evidence-qualified occupancy resource curves for dishwasher dispatch.
//!
//! These curves are built from physical measurements for one exact machine,
//! programme, soil class, surface and protocol. They intentionally reject
//! extrapolation. Piecewise-linear interpolation is a diagnostic estimate,
//! not a validated prediction or a substitute for testing the target occupancy.

use std::collections::HashSet;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurveScope {
    pub machine_model: String,
    pub programme: String,
    pub soil_class: String,
    pub surface_material: String,
    pub load_set_id: String,
    pub detergent_batch_id: String,
    pub water_hardness_ppm_as_caco3: u16,
    pub inlet_temperature_millic: u32,
    pub protocol_id: String,
    /// SHA-256 digest of the frozen raw-data bundle (64 hexadecimal characters).
    pub dataset_sha256: String,
}

impl CurveScope {
    fn validate(&self) -> Result<(), CurveError> {
        for (name, value) in [
            ("machine_model", &self.machine_model),
            ("programme", &self.programme),
            ("soil_class", &self.soil_class),
            ("surface_material", &self.surface_material),
            ("load_set_id", &self.load_set_id),
            ("detergent_batch_id", &self.detergent_batch_id),
            ("protocol_id", &self.protocol_id),
        ] {
            if value.trim().is_empty() {
                return Err(CurveError::InvalidScope(name));
            }
        }
        if self.water_hardness_ppm_as_caco3 > 1_000
            || self.inlet_temperature_millic > 90_000
        {
            return Err(CurveError::InvalidWaterCondition);
        }
        if self.dataset_sha256.len() != 64
            || !self.dataset_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(CurveError::InvalidDatasetDigest);
        }
        Ok(())
    }
}

/// Min/mean/max observed across independent physical runs.
/// These are descriptive observed bounds, not a confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservedBand {
    pub minimum: f64,
    pub mean: f64,
    pub maximum: f64,
}

impl ObservedBand {
    fn validate(self, field: &'static str) -> Result<(), CurveError> {
        if [self.minimum, self.mean, self.maximum]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(CurveError::InvalidObservedBand(field));
        }
        if self.minimum > self.mean || self.mean > self.maximum {
            return Err(CurveError::InvalidObservedBand(field));
        }
        Ok(())
    }

    fn interpolate(self, other: Self, fraction: f64) -> Self {
        let blend = |left: f64, right: f64| left + (right - left) * fraction;
        Self {
            minimum: blend(self.minimum, other.minimum),
            mean: blend(self.mean, other.mean),
            maximum: blend(self.maximum, other.maximum),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OccupancyObservation {
    /// Number of place settings or items under the frozen protocol's definition.
    pub occupancy: usize,
    /// Distinct run IDs included in the frozen raw-data bundle.
    pub replicate_ids: Vec<String>,
    pub cleaning_failures: usize,
    pub rinse_failures: usize,
    pub energy_kwh: ObservedBand,
    pub fresh_water_l: ObservedBand,
    pub duration_s: ObservedBand,
}

impl OccupancyObservation {
    fn validate(&self) -> Result<(), CurveError> {
        if self.occupancy == 0 {
            return Err(CurveError::ZeroOccupancy);
        }
        // Five distinct physical run IDs is the minimum pilot gate in issue
        // #10136. The raw-data digest must include these run identifiers.
        // Unique IDs are a provenance check, not proof of independence by themselves.
        if self.replicate_ids.len() < 5 {
            return Err(CurveError::InsufficientReplicates {
                occupancy: self.occupancy,
                observed: self.replicate_ids.len(),
                required: 5,
            });
        }
        if self.replicate_ids.iter().any(|id| id.trim().is_empty()) {
            return Err(CurveError::InvalidReplicateId);
        }
        let unique_ids: HashSet<&str> = self.replicate_ids.iter().map(String::as_str).collect();
        if unique_ids.len() != self.replicate_ids.len() {
            return Err(CurveError::DuplicateReplicateId {
                occupancy: self.occupancy,
            });
        }
        if self.cleaning_failures > self.replicate_count
            || self.rinse_failures > self.replicate_count
        {
            return Err(CurveError::FailureCountExceedsReplicates {
                occupancy: self.occupancy,
            });
        }
        if self.cleaning_failures != 0 || self.rinse_failures != 0 {
            return Err(CurveError::QualityGateFailed {
                occupancy: self.occupancy,
                cleaning_failures: self.cleaning_failures,
                rinse_failures: self.rinse_failures,
            });
        }
        self.energy_kwh.validate("energy_kwh")?;
        self.fresh_water_l.validate("fresh_water_l")?;
        self.duration_s.validate("duration_s")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceEstimate {
    pub requested_occupancy: usize,
    pub energy_kwh: ObservedBand,
    pub fresh_water_l: ObservedBand,
    pub duration_s: ObservedBand,
    pub lower_bracket_occupancy: usize,
    pub upper_bracket_occupancy: usize,
    pub interpolation_fraction: f64,
    /// False for interpolated rows, true only when the occupancy was measured.
    pub exact_measurement: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredOccupancyCurve {
    pub scope: CurveScope,
    points: Vec<OccupancyObservation>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CurveError {
    InvalidScope(&'static str),
    InvalidDatasetDigest,
    InvalidWaterCondition,
    InvalidReplicateId,
    DuplicateReplicateId {
        occupancy: usize,
    },
    TooFewOccupancyPoints,
    ZeroOccupancy,
    InsufficientReplicates {
        occupancy: usize,
        observed: usize,
        required: usize,
    },
    FailureCountExceedsReplicates {
        occupancy: usize,
    },
    QualityGateFailed {
        occupancy: usize,
        cleaning_failures: usize,
        rinse_failures: usize,
    },
    InvalidObservedBand(&'static str),
    OccupancyNotStrictlyIncreasing,
    ExtrapolationRejected {
        requested: usize,
        minimum: usize,
        maximum: usize,
    },
}

impl fmt::Display for CurveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for CurveError {}

impl MeasuredOccupancyCurve {
    pub fn new(
        scope: CurveScope,
        mut points: Vec<OccupancyObservation>,
    ) -> Result<Self, CurveError> {
        scope.validate()?;
        if points.len() < 2 {
            return Err(CurveError::TooFewOccupancyPoints);
        }
        points.sort_by_key(|point| point.occupancy);
        for point in &points {
            point.validate()?;
        }
        if points
            .windows(2)
            .any(|pair| pair[0].occupancy >= pair[1].occupancy)
        {
            return Err(CurveError::OccupancyNotStrictlyIncreasing);
        }
        Ok(Self { scope, points })
    }

    pub fn points(&self) -> &[OccupancyObservation] {
        &self.points
    }

    /// Estimate a value only inside the measured occupancy range.
    ///
    /// Exact measured occupancies return their physical observation. Values
    /// between observations use piecewise-linear interpolation. The returned
    /// min/mean/max bands remain descriptive; interpolation does not establish
    /// statistical coverage. Confirm target occupancies with held-out runs.
    pub fn estimate(&self, occupancy: usize) -> Result<ResourceEstimate, CurveError> {
        let minimum = self.points.first().expect("curve has at least two points").occupancy;
        let maximum = self.points.last().expect("curve has at least two points").occupancy;
        if occupancy < minimum || occupancy > maximum {
            return Err(CurveError::ExtrapolationRejected {
                requested: occupancy,
                minimum,
                maximum,
            });
        }

        if let Some(point) = self.points.iter().find(|point| point.occupancy == occupancy) {
            return Ok(ResourceEstimate {
                requested_occupancy: occupancy,
                energy_kwh: point.energy_kwh,
                fresh_water_l: point.fresh_water_l,
                duration_s: point.duration_s,
                lower_bracket_occupancy: occupancy,
                upper_bracket_occupancy: occupancy,
                interpolation_fraction: 0.0,
                exact_measurement: true,
            });
        }

        let pair = self
            .points
            .windows(2)
            .find(|pair| pair[0].occupancy < occupancy && occupancy < pair[1].occupancy)
            .expect("in-range non-exact occupancy must have a bracket");
        let lower = &pair[0];
        let upper = &pair[1];
        let fraction = (occupancy - lower.occupancy) as f64
            / (upper.occupancy - lower.occupancy) as f64;

        Ok(ResourceEstimate {
            requested_occupancy: occupancy,
            energy_kwh: lower.energy_kwh.interpolate(upper.energy_kwh, fraction),
            fresh_water_l: lower.fresh_water_l.interpolate(upper.fresh_water_l, fraction),
            duration_s: lower.duration_s.interpolate(upper.duration_s, fraction),
            lower_bracket_occupancy: lower.occupancy,
            upper_bracket_occupancy: upper.occupancy,
            interpolation_fraction: fraction,
            exact_measurement: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> CurveScope {
        CurveScope {
            machine_model: "Reference-Machine-A".to_owned(),
            programme: "Eco".to_owned(),
            soil_class: "starch".to_owned(),
            surface_material: "ceramic".to_owned(),
            load_set_id: "standardized-ceramic-load-v1".to_owned(),
            detergent_batch_id: "detergent-lot-2026-01".to_owned(),
            water_hardness_ppm_as_caco3: 120,
            inlet_temperature_millic: 15_000,
            protocol_id: "IEC60436-2025-COR1-test-protocol-v1".to_owned(),
            dataset_sha256: "a".repeat(64),
        }
    }

    fn point(occupancy: usize, energy: f64, water: f64, duration: f64) -> OccupancyObservation {
        OccupancyObservation {
            occupancy,
            replicate_ids: vec![
                "run-01".to_owned(),
                "run-02".to_owned(),
                "run-03".to_owned(),
                "run-04".to_owned(),
                "run-05".to_owned(),
            ],
            cleaning_failures: 0,
            rinse_failures: 0,
            energy_kwh: ObservedBand {
                minimum: energy * 0.9,
                mean: energy,
                maximum: energy * 1.1,
            },
            fresh_water_l: ObservedBand {
                minimum: water * 0.9,
                mean: water,
                maximum: water * 1.1,
            },
            duration_s: ObservedBand {
                minimum: duration * 0.9,
                mean: duration,
                maximum: duration * 1.1,
            },
        }
    }

    fn curve() -> MeasuredOccupancyCurve {
        MeasuredOccupancyCurve::new(
            scope(),
            vec![
                point(1, 0.20, 2.0, 900.0),
                point(4, 0.28, 3.5, 1_200.0),
                point(8, 0.40, 5.0, 1_500.0),
            ],
        )
        .unwrap()
    }

    #[test]
    fn exact_measured_occupancy_is_returned_without_interpolation() {
        let result = curve().estimate(4).unwrap();
        assert!(result.exact_measurement);
        assert_eq!(result.energy_kwh.mean, 0.28);
        assert_eq!(result.fresh_water_l.mean, 3.5);
        assert_eq!(result.duration_s.mean, 1_200.0);
    }

    #[test]
    fn intermediate_occupancy_is_piecewise_linearly_interpolated() {
        let result = curve().estimate(2).unwrap();
        assert!(!result.exact_measurement);
        assert_eq!(result.lower_bracket_occupancy, 1);
        assert_eq!(result.upper_bracket_occupancy, 4);
        assert!((result.energy_kwh.mean - (0.20 + (0.28 - 0.20) / 3.0)).abs() < 1e-12);
        assert!((result.fresh_water_l.mean - 2.5).abs() < 1e-12);
    }

    #[test]
    fn extrapolation_below_or_above_measured_range_is_rejected() {
        assert!(matches!(
            curve().estimate(0),
            Err(CurveError::ExtrapolationRejected { requested: 0, .. })
        ));
        assert!(matches!(
            curve().estimate(9),
            Err(CurveError::ExtrapolationRejected { requested: 9, .. })
        ));
    }

    #[test]
    fn cleaning_or_rinse_failure_disqualifies_the_occupancy_curve() {
        let mut failed = point(4, 0.28, 3.5, 1_200.0);
        failed.cleaning_failures = 1;
        let result = MeasuredOccupancyCurve::new(
            scope(),
            vec![point(1, 0.2, 2.0, 900.0), failed],
        );
        assert!(matches!(result, Err(CurveError::QualityGateFailed { .. })));
    }

    #[test]
    fn insufficient_replicates_are_rejected() {
        let mut underpowered = point(4, 0.28, 3.5, 1_200.0);
        underpowered.replicate_ids.pop();
        let result = MeasuredOccupancyCurve::new(
            scope(),
            vec![point(1, 0.2, 2.0, 900.0), underpowered],
        );
        assert!(matches!(result, Err(CurveError::InsufficientReplicates { .. })));
    }

    #[test]
    fn invalid_dataset_digest_and_scope_are_rejected() {
        let mut bad_scope = scope();
        bad_scope.dataset_sha256 = "not-a-sha256".to_owned();
        assert!(matches!(
            MeasuredOccupancyCurve::new(
                bad_scope,
                vec![point(1, 0.2, 2.0, 900.0), point(2, 0.24, 2.5, 1_000.0)],
            ),
            Err(CurveError::InvalidDatasetDigest)
        ));

        let mut missing_programme = scope();
        missing_programme.programme.clear();
        assert!(matches!(
            MeasuredOccupancyCurve::new(
                missing_programme,
                vec![point(1, 0.2, 2.0, 900.0), point(2, 0.24, 2.5, 1_000.0)],
            ),
            Err(CurveError::InvalidScope("programme"))
        ));
    }

    #[test]
    fn duplicate_or_blank_run_ids_are_rejected() {
        let mut duplicate = point(4, 0.28, 3.5, 1_200.0);
        duplicate.replicate_ids[4] = duplicate.replicate_ids[0].clone();
        assert!(matches!(
            MeasuredOccupancyCurve::new(
                scope(),
                vec![point(1, 0.2, 2.0, 900.0), duplicate],
            ),
            Err(CurveError::DuplicateReplicateId { .. })
        ));

        let mut blank = point(4, 0.28, 3.5, 1_200.0);
        blank.replicate_ids[0].clear();
        assert!(matches!(
            MeasuredOccupancyCurve::new(
                scope(),
                vec![point(1, 0.2, 2.0, 900.0), blank],
            ),
            Err(CurveError::InvalidReplicateId)
        ));
    }

    #[test]
    fn invalid_water_conditions_are_rejected() {
        let mut invalid = scope();
        invalid.water_hardness_ppm_as_caco3 = 1_001;
        assert!(matches!(
            MeasuredOccupancyCurve::new(
                invalid,
                vec![point(1, 0.2, 2.0, 900.0), point(2, 0.24, 2.5, 1_000.0)],
            ),
            Err(CurveError::InvalidWaterCondition)
        ));
    }

    #[test]
    fn duplicate_occupancy_is_rejected() {
        assert!(matches!(
            MeasuredOccupancyCurve::new(
                scope(),
                vec![point(2, 0.2, 2.0, 900.0), point(2, 0.24, 2.5, 1_000.0)],
            ),
            Err(CurveError::OccupancyNotStrictlyIncreasing)
        ));
    }

    #[test]
    fn malformed_observed_bands_are_rejected() {
        let mut malformed = point(4, 0.28, 3.5, 1_200.0);
        malformed.energy_kwh = ObservedBand {
            minimum: 0.3,
            mean: 0.28,
            maximum: 0.4,
        };
        assert!(matches!(
            MeasuredOccupancyCurve::new(
                scope(),
                vec![point(1, 0.2, 2.0, 900.0), malformed],
            ),
            Err(CurveError::InvalidObservedBand("energy_kwh"))
        ));
    }

    #[test]
    fn failure_counts_cannot_exceed_replicates() {
        let mut malformed = point(4, 0.28, 3.5, 1_200.0);
        malformed.rinse_failures = 6;
        assert!(matches!(
            MeasuredOccupancyCurve::new(
                scope(),
                vec![point(1, 0.2, 2.0, 900.0), malformed],
            ),
            Err(CurveError::FailureCountExceedsReplicates { .. })
        ));
    }
}
