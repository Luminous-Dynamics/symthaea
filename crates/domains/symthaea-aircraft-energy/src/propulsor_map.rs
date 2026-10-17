// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Provenance-bound rectangular propulsor performance maps.
//!
//! A validated map represents one propulsor configuration, one immutable source
//! artifact, one model/acquisition method, and one fixed reference-condition
//! profile. Bilinear interpolation is allowed only inside the complete declared
//! speed-by-RPM grid. The implementation never extrapolates or clamps.
//!
//! Interpolation accuracy does not establish measurement accuracy or physical
//! validity. Source identity is metadata: the SHA-256 syntax is checked here,
//! but this module does not fetch/authenticate the source bytes, establish
//! calibration, or qualify a propulsor for flight.

/// What kind of evidence populated a map. The categories remain separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropulsorMapDataKind {
    Experimental,
    NumericalSimulation,
    AnalyticalModel,
}

/// Provenance and fixed conditions for one propulsor map.
///
/// The locator/revision/digest identify the declared source artifact. A digest
/// match is not proof of accuracy or qualification. Applicability and use
/// limitations must be explicit even for synthetic research maps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropulsorMapProvenance {
    pub map_id: String,
    pub propulsor_configuration_id: String,
    pub source_locator: String,
    pub source_revision: String,
    pub source_sha256: String,
    pub data_kind: PropulsorMapDataKind,
    /// Identifies fixed factors not represented as axes, such as air-density
    /// profile, propeller pitch, voltage range, hardware version and setup.
    pub reference_conditions_id: String,
    pub method_id: String,
    pub applicability_notes: String,
    pub use_limitations: String,
}

/// One grid cell's observed or modelled operating point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropulsorPerformancePoint {
    pub thrust_n: f64,
    pub torque_nm: f64,
    pub shaft_power_w: f64,
    pub electrical_input_power_w: f64,
}

/// Unvalidated speed-by-RPM data, with speed-major then RPM-minor row order.
///
/// There must be exactly speeds_m_s.len() * rpms.len() points. Each cell is
/// explicit; sparse grids must be represented as separate complete maps or
/// rejected by the importer, never silently filled with favorable values.
#[derive(Debug, Clone, PartialEq)]
pub struct PropulsorPerformanceMap {
    pub provenance: PropulsorMapProvenance,
    pub speeds_m_s: Vec<f64>,
    pub rpms: Vec<f64>,
    pub points: Vec<PropulsorPerformancePoint>,
}

/// Errors for schema validation, compatibility and bounded interpolation.
#[derive(Debug, Clone, PartialEq)]
pub enum PropulsorMapError {
    EmptyField { field: &'static str },
    InvalidSha256,
    AxisTooShort { axis: &'static str, actual: usize },
    NonFiniteAxis { axis: &'static str, index: usize },
    InvalidAxisValue { axis: &'static str, index: usize },
    NonIncreasingAxis { axis: &'static str, index: usize },
    GridSizeOverflow,
    PointCountMismatch { expected: usize, actual: usize },
    NonFinitePoint { index: usize, field: &'static str },
    NegativePoint { index: usize, field: &'static str },
    NonFiniteQuery { field: &'static str },
    OutOfDomain {
        axis: &'static str,
        value: f64,
        minimum: f64,
        maximum: f64,
    },
    PropulsorConfigurationMismatch,
    ReferenceConditionsMismatch,
    DerivedValueOutOfRange,
}

/// A successfully validated map. Fields stay private so repeated queries cannot
/// bypass schema checks after deserialization or manual construction.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedPropulsorPerformanceMap {
    provenance: PropulsorMapProvenance,
    speeds_m_s: Vec<f64>,
    rpms: Vec<f64>,
    points: Vec<PropulsorPerformancePoint>,
}

/// Query coordinates and compatibility identities.
///
/// Other environmental and hardware conditions must match the map's declared
/// reference-condition profile; they are not inferred from speed/RPM.
#[derive(Debug, Clone, PartialEq)]
pub struct PropulsorMapQuery {
    pub propulsor_configuration_id: String,
    pub reference_conditions_id: String,
    pub forward_speed_m_s: f64,
    pub rpm: f64,
}

/// The interpolated estimate and exact cell used to obtain it.
///
/// Values are estimates inside a tabulated domain. This result does not itself
/// carry a calibrated uncertainty interval or a physical-qualification state.
#[derive(Debug, Clone, PartialEq)]
pub struct PropulsorMapEstimate {
    pub map_id: String,
    pub source_sha256: String,
    pub data_kind: PropulsorMapDataKind,
    pub reference_conditions_id: String,
    pub forward_speed_m_s: f64,
    pub rpm: f64,
    pub thrust_n: f64,
    pub torque_nm: f64,
    pub shaft_power_w: f64,
    pub electrical_input_power_w: f64,
    pub cell_speed_lower_m_s: f64,
    pub cell_speed_upper_m_s: f64,
    pub cell_rpm_lower: f64,
    pub cell_rpm_upper: f64,
    pub speed_fraction: f64,
    pub rpm_fraction: f64,
}

impl PropulsorPerformanceMap {
    /// Validate all metadata, axes and operating points before enabling queries.
    pub fn validate(self) -> Result<ValidatedPropulsorPerformanceMap, PropulsorMapError> {
        for (field, value) in [
            ("map_id", self.provenance.map_id.as_str()),
            ("propulsor_configuration_id", self.provenance.propulsor_configuration_id.as_str()),
            ("source_locator", self.provenance.source_locator.as_str()),
            ("source_revision", self.provenance.source_revision.as_str()),
            ("reference_conditions_id", self.provenance.reference_conditions_id.as_str()),
            ("method_id", self.provenance.method_id.as_str()),
            ("applicability_notes", self.provenance.applicability_notes.as_str()),
            ("use_limitations", self.provenance.use_limitations.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(PropulsorMapError::EmptyField { field });
            }
        }
        if self.provenance.source_sha256.len() != 64
            || !self.provenance.source_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(PropulsorMapError::InvalidSha256);
        }

        validate_axis("forward_speed_m_s", &self.speeds_m_s, true)?;
        validate_axis("rpm", &self.rpms, false)?;
        let expected = self.speeds_m_s.len()
            .checked_mul(self.rpms.len())
            .ok_or(PropulsorMapError::GridSizeOverflow)?;
        if self.points.len() != expected {
            return Err(PropulsorMapError::PointCountMismatch {
                expected,
                actual: self.points.len(),
            });
        }

        for (index, point) in self.points.iter().enumerate() {
            for (field, value) in [
                ("thrust_n", point.thrust_n),
                ("torque_nm", point.torque_nm),
                ("shaft_power_w", point.shaft_power_w),
                ("electrical_input_power_w", point.electrical_input_power_w),
            ] {
                if !value.is_finite() {
                    return Err(PropulsorMapError::NonFinitePoint { index, field });
                }
                if value < 0.0 {
                    return Err(PropulsorMapError::NegativePoint { index, field });
                }
            }
        }

        Ok(ValidatedPropulsorPerformanceMap {
            provenance: self.provenance,
            speeds_m_s: self.speeds_m_s,
            rpms: self.rpms,
            points: self.points,
        })
    }
}

impl ValidatedPropulsorPerformanceMap {
    pub fn provenance(&self) -> &PropulsorMapProvenance {
        &self.provenance
    }

    /// Bilinearly interpolate inside the map. Queries outside either axis,
    /// under another configuration, or under another fixed condition profile
    /// fail closed. There is intentionally no extrapolation/clamping policy.
    pub fn interpolate(
        &self,
        query: &PropulsorMapQuery,
    ) -> Result<PropulsorMapEstimate, PropulsorMapError> {
        if query.propulsor_configuration_id != self.provenance.propulsor_configuration_id {
            return Err(PropulsorMapError::PropulsorConfigurationMismatch);
        }
        if query.reference_conditions_id != self.provenance.reference_conditions_id {
            return Err(PropulsorMapError::ReferenceConditionsMismatch);
        }
        for (field, value) in [
            ("forward_speed_m_s", query.forward_speed_m_s),
            ("rpm", query.rpm),
        ] {
            if !value.is_finite() {
                return Err(PropulsorMapError::NonFiniteQuery { field });
            }
        }

        let (speed_index, speed_fraction) =
            bracket("forward_speed_m_s", &self.speeds_m_s, query.forward_speed_m_s)?;
        let (rpm_index, rpm_fraction) = bracket("rpm", &self.rpms, query.rpm)?;
        let rpm_count = self.rpms.len();
        let p00 = self.points[speed_index * rpm_count + rpm_index];
        let p01 = self.points[speed_index * rpm_count + rpm_index + 1];
        let p10 = self.points[(speed_index + 1) * rpm_count + rpm_index];
        let p11 = self.points[(speed_index + 1) * rpm_count + rpm_index + 1];

        let mix = |a: f64, b: f64, c: f64, d: f64| {
            let low_speed = a * (1.0 - rpm_fraction) + b * rpm_fraction;
            let high_speed = c * (1.0 - rpm_fraction) + d * rpm_fraction;
            low_speed * (1.0 - speed_fraction) + high_speed * speed_fraction
        };
        let estimate = PropulsorMapEstimate {
            map_id: self.provenance.map_id.clone(),
            source_sha256: self.provenance.source_sha256.clone(),
            data_kind: self.provenance.data_kind,
            reference_conditions_id: self.provenance.reference_conditions_id.clone(),
            forward_speed_m_s: query.forward_speed_m_s,
            rpm: query.rpm,
            thrust_n: mix(p00.thrust_n, p01.thrust_n, p10.thrust_n, p11.thrust_n),
            torque_nm: mix(p00.torque_nm, p01.torque_nm, p10.torque_nm, p11.torque_nm),
            shaft_power_w: mix(p00.shaft_power_w, p01.shaft_power_w, p10.shaft_power_w, p11.shaft_power_w),
            electrical_input_power_w: mix(
                p00.electrical_input_power_w,
                p01.electrical_input_power_w,
                p10.electrical_input_power_w,
                p11.electrical_input_power_w,
            ),
            cell_speed_lower_m_s: self.speeds_m_s[speed_index],
            cell_speed_upper_m_s: self.speeds_m_s[speed_index + 1],
            cell_rpm_lower: self.rpms[rpm_index],
            cell_rpm_upper: self.rpms[rpm_index + 1],
            speed_fraction,
            rpm_fraction,
        };
        for value in [
            estimate.thrust_n,
            estimate.torque_nm,
            estimate.shaft_power_w,
            estimate.electrical_input_power_w,
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(PropulsorMapError::DerivedValueOutOfRange);
            }
        }
        Ok(estimate)
    }
}

fn validate_axis(
    axis: &'static str,
    values: &[f64],
    allow_zero: bool,
) -> Result<(), PropulsorMapError> {
    if values.len() < 2 {
        return Err(PropulsorMapError::AxisTooShort { axis, actual: values.len() });
    }
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PropulsorMapError::NonFiniteAxis { axis, index });
        }
        if value < 0.0 || (!allow_zero && value == 0.0) {
            return Err(PropulsorMapError::InvalidAxisValue { axis, index });
        }
        if index > 0 && value <= values[index - 1] {
            return Err(PropulsorMapError::NonIncreasingAxis { axis, index });
        }
    }
    Ok(())
}

fn bracket(
    axis: &'static str,
    values: &[f64],
    query: f64,
) -> Result<(usize, f64), PropulsorMapError> {
    let minimum = values[0];
    let maximum = values[values.len() - 1];
    if query < minimum || query > maximum {
        return Err(PropulsorMapError::OutOfDomain {
            axis,
            value: query,
            minimum,
            maximum,
        });
    }
    let mut high = values.partition_point(|&value| value <= query);
    if high == 0 {
        return Err(PropulsorMapError::OutOfDomain { axis, value: query, minimum, maximum });
    }
    if high == values.len() {
        high = values.len() - 1;
    }
    let low = high - 1;
    let fraction = (query - values[low]) / (values[high] - values[low]);
    if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
        return Err(PropulsorMapError::DerivedValueOutOfRange);
    }
    Ok((low, fraction))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_map() -> PropulsorPerformanceMap {
        PropulsorPerformanceMap {
            provenance: PropulsorMapProvenance {
                map_id: "synthetic-map-v1".into(),
                propulsor_configuration_id: "synthetic-propulsor-a".into(),
                source_locator: "synthetic://propulsor-map".into(),
                source_revision: "fixture-v1".into(),
                source_sha256: "a".repeat(64),
                data_kind: PropulsorMapDataKind::Experimental,
                reference_conditions_id: "synthetic-standard-day-v1".into(),
                method_id: "synthetic-known-answer-v1".into(),
                applicability_notes: "synthetic fixture only".into(),
                use_limitations: "not measured and not for operational decisions".into(),
            },
            speeds_m_s: vec![0.0, 10.0],
            rpms: vec![1_000.0, 2_000.0],
            points: vec![
                PropulsorPerformancePoint { thrust_n: 100.0, torque_nm: 2.0, shaft_power_w: 200.0, electrical_input_power_w: 250.0 },
                PropulsorPerformancePoint { thrust_n: 200.0, torque_nm: 8.0, shaft_power_w: 1_600.0, electrical_input_power_w: 2_000.0 },
                PropulsorPerformancePoint { thrust_n: 80.0, torque_nm: 2.0, shaft_power_w: 200.0, electrical_input_power_w: 250.0 },
                PropulsorPerformancePoint { thrust_n: 160.0, torque_nm: 8.0, shaft_power_w: 1_600.0, electrical_input_power_w: 2_000.0 },
            ],
        }
    }

    fn query(speed: f64, rpm: f64) -> PropulsorMapQuery {
        PropulsorMapQuery {
            propulsor_configuration_id: "synthetic-propulsor-a".into(),
            reference_conditions_id: "synthetic-standard-day-v1".into(),
            forward_speed_m_s: speed,
            rpm,
        }
    }

    #[test]
    fn center_point_matches_independent_bilinear_oracle() {
        let map = synthetic_map().validate().unwrap();
        let result = map.interpolate(&query(5.0, 1_500.0)).unwrap();
        assert!((result.thrust_n - 135.0).abs() < 1e-12);
        assert!((result.torque_nm - 5.0).abs() < 1e-12);
        assert!((result.shaft_power_w - 900.0).abs() < 1e-12);
        assert!((result.electrical_input_power_w - 1_125.0).abs() < 1e-12);
    }

    #[test]
    fn exact_corners_and_grid_edges_are_interpolated_without_extrapolation() {
        let map = synthetic_map().validate().unwrap();
        assert_eq!(map.interpolate(&query(0.0, 1_000.0)).unwrap().thrust_n, 100.0);
        assert_eq!(map.interpolate(&query(10.0, 2_000.0)).unwrap().thrust_n, 160.0);
        assert_eq!(map.interpolate(&query(5.0, 1_000.0)).unwrap().thrust_n, 90.0);
        assert_eq!(map.interpolate(&query(10.0, 1_500.0)).unwrap().thrust_n, 130.0);
    }

    #[test]
    fn rejects_out_of_domain_speed_and_rpm_instead_of_clamping() {
        let map = synthetic_map().validate().unwrap();
        assert!(matches!(map.interpolate(&query(-0.1, 1_500.0)), Err(PropulsorMapError::OutOfDomain { axis: "forward_speed_m_s", .. })));
        assert!(matches!(map.interpolate(&query(5.0, 2_000.1)), Err(PropulsorMapError::OutOfDomain { axis: "rpm", .. })));
    }

    #[test]
    fn rejects_incompatible_configuration_and_reference_conditions() {
        let map = synthetic_map().validate().unwrap();
        let mut wrong_config = query(5.0, 1_500.0);
        wrong_config.propulsor_configuration_id = "different-propulsor".into();
        assert_eq!(map.interpolate(&wrong_config), Err(PropulsorMapError::PropulsorConfigurationMismatch));
        let mut wrong_conditions = query(5.0, 1_500.0);
        wrong_conditions.reference_conditions_id = "different-day".into();
        assert_eq!(map.interpolate(&wrong_conditions), Err(PropulsorMapError::ReferenceConditionsMismatch));
    }

    #[test]
    fn rejects_bad_provenance_digest_and_missing_grid_cells() {
        let mut bad_digest = synthetic_map();
        bad_digest.provenance.source_sha256 = "not-a-sha".into();
        assert_eq!(bad_digest.validate().unwrap_err(), PropulsorMapError::InvalidSha256);

        let mut missing_cell = synthetic_map();
        missing_cell.points.pop();
        assert_eq!(missing_cell.validate().unwrap_err(), PropulsorMapError::PointCountMismatch { expected: 4, actual: 3 });
    }

    #[test]
    fn rejects_duplicate_axis_values_and_non_finite_measurements() {
        let mut duplicate_axis = synthetic_map();
        duplicate_axis.rpms = vec![1_000.0, 1_000.0];
        assert_eq!(duplicate_axis.validate().unwrap_err(), PropulsorMapError::NonIncreasingAxis { axis: "rpm", index: 1 });

        let mut invalid_point = synthetic_map();
        invalid_point.points[2].thrust_n = f64::NAN;
        assert_eq!(invalid_point.validate().unwrap_err(), PropulsorMapError::NonFinitePoint { index: 2, field: "thrust_n" });
    }

    #[test]
    fn rejects_negative_performance_values() {
        let mut invalid_point = synthetic_map();
        invalid_point.points[0].torque_nm = -0.1;
        assert_eq!(invalid_point.validate().unwrap_err(), PropulsorMapError::NegativePoint { index: 0, field: "torque_nm" });
    }

    #[test]
    fn rejects_non_finite_query_coordinates() {
        let map = synthetic_map().validate().unwrap();
        assert_eq!(map.interpolate(&query(f64::NAN, 1_500.0)), Err(PropulsorMapError::NonFiniteQuery { field: "forward_speed_m_s" }));
    }

    #[test]
    fn same_query_and_map_produce_identical_estimates() {
        let map = synthetic_map().validate().unwrap();
        assert_eq!(map.interpolate(&query(3.0, 1_250.0)).unwrap(), map.interpolate(&query(3.0, 1_250.0)).unwrap());
    }
}
