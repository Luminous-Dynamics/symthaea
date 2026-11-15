//! G2d deterministic central finite-difference sensitivities.
//!
//! Sensitivities describe the selected polyhedral hydrostatic model only.
//! Both perturbations must independently pass the same section, mesh and
//! hydrostatic pipeline. There is no one-sided fallback or automatic repair.

use serde::{Deserialize, Serialize};

use crate::hydrostatics::{
    calculate_hydrostatic_properties, HydrostaticProperties, HydrostaticViolation,
};
use crate::loft::{loft_waterplane_capped_hull, LoftedHullArtifact, LoftViolation};
use crate::section::{
    compute_section_spec_digest, generate_section_artifact, SectionHullSpec, SectionViolation,
};
use crate::{AdmissionProfile, Disposition};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SectionParameter {
    LengthM,
    BeamM,
    DraftM,
    StationHalfBreadthFraction {
        station_index: usize,
    },
    StationDraftFraction {
        station_index: usize,
    },
    ControlBreadthFraction {
        station_index: usize,
        control_index: usize,
    },
    ControlDepthFraction {
        station_index: usize,
        control_index: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HydrostaticObservable {
    DisplacementVolumeM3,
    DisplacementTonnes,
    LongitudinalCentreOfBuoyancyM,
    TransverseCentreOfBuoyancyM,
    VerticalCentreOfBuoyancyM,
    WaterplaneAreaM2,
    LongitudinalCentreOfFlotationM,
    TransverseCentreOfFlotationM,
    TransverseMetacentricRadiusM,
    LongitudinalMetacentricRadiusM,
    TonnesPerCentimetreImmersion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensitivityViolation {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HydrostaticSensitivity {
    pub result_schema_version: u32,
    pub candidate_id: String,
    pub profile_id: String,
    pub profile_digest_blake3: String,
    pub parameter: SectionParameter,
    pub parameter_code: String,
    pub parameter_units: String,
    pub observable: HydrostaticObservable,
    pub observable_units: String,
    pub finite_difference_step: f64,
    pub water_density_kg_m3: f64,
    pub water_density_source: String,
    pub base_spec_digest_blake3: String,
    pub minus_spec_digest_blake3: String,
    pub plus_spec_digest_blake3: String,
    pub minus_half_step_spec_digest_blake3: String,
    pub plus_half_step_spec_digest_blake3: String,
    pub base_geometry_digest_blake3: String,
    pub minus_geometry_digest_blake3: String,
    pub plus_geometry_digest_blake3: String,
    pub minus_half_step_geometry_digest_blake3: String,
    pub plus_half_step_geometry_digest_blake3: String,
    pub base_value: f64,
    pub minus_value: f64,
    pub plus_value: f64,
    pub minus_half_step_value: f64,
    pub plus_half_step_value: f64,
    pub coarse_derivative_per_parameter_unit: f64,
    pub fine_derivative_per_parameter_unit: f64,
    pub estimated_truncation_error: f64,
    /// Richardson-extrapolated central difference using h and h/2.
    pub derivative_per_parameter_unit: f64,
    /// BLAKE3 of this result with the digest field cleared.
    pub result_digest_blake3: String,
}

fn violation(
    violations: &mut Vec<SensitivityViolation>,
    code: &str,
    detail: impl Into<String>,
) {
    violations.push(SensitivityViolation {
        code: code.to_owned(),
        detail: detail.into(),
    });
}

fn parameter_metadata(parameter: SectionParameter) -> (&'static str, &'static str) {
    match parameter {
        SectionParameter::LengthM => ("length_m", "m"),
        SectionParameter::BeamM => ("beam_m", "m"),
        SectionParameter::DraftM => ("draft_m", "m"),
        SectionParameter::StationHalfBreadthFraction { .. } => {
            ("station_half_breadth_fraction", "1")
        }
        SectionParameter::StationDraftFraction { .. } => ("station_draft_fraction", "1"),
        SectionParameter::ControlBreadthFraction { .. } => ("control_breadth_fraction", "1"),
        SectionParameter::ControlDepthFraction { .. } => ("control_depth_fraction", "1"),
    }
}

fn observable_units(observable: HydrostaticObservable) -> &'static str {
    match observable {
        HydrostaticObservable::DisplacementVolumeM3 => "m^3",
        HydrostaticObservable::DisplacementTonnes => "tonne",
        HydrostaticObservable::LongitudinalCentreOfBuoyancyM
        | HydrostaticObservable::TransverseCentreOfBuoyancyM
        | HydrostaticObservable::VerticalCentreOfBuoyancyM
        | HydrostaticObservable::LongitudinalCentreOfFlotationM
        | HydrostaticObservable::TransverseCentreOfFlotationM
        | HydrostaticObservable::TransverseMetacentricRadiusM
        | HydrostaticObservable::LongitudinalMetacentricRadiusM => "m",
        HydrostaticObservable::WaterplaneAreaM2 => "m^2",
        HydrostaticObservable::TonnesPerCentimetreImmersion => "tonne/cm",
    }
}

fn observable_value(
    properties: &HydrostaticProperties,
    observable: HydrostaticObservable,
) -> f64 {
    match observable {
        HydrostaticObservable::DisplacementVolumeM3 => properties.displacement_volume_m3,
        HydrostaticObservable::DisplacementTonnes => properties.displacement_tonnes,
        HydrostaticObservable::LongitudinalCentreOfBuoyancyM => {
            properties.longitudinal_centre_of_buoyancy_m
        }
        HydrostaticObservable::TransverseCentreOfBuoyancyM => {
            properties.transverse_centre_of_buoyancy_m
        }
        HydrostaticObservable::VerticalCentreOfBuoyancyM => {
            properties.vertical_centre_of_buoyancy_below_waterplane_m
        }
        HydrostaticObservable::WaterplaneAreaM2 => properties.waterplane_area_m2,
        HydrostaticObservable::LongitudinalCentreOfFlotationM => {
            properties.longitudinal_centre_of_flotation_m
        }
        HydrostaticObservable::TransverseCentreOfFlotationM => {
            properties.transverse_centre_of_flotation_m
        }
        HydrostaticObservable::TransverseMetacentricRadiusM => {
            properties.transverse_metacentric_radius_bm_t_m
        }
        HydrostaticObservable::LongitudinalMetacentricRadiusM => {
            properties.longitudinal_metacentric_radius_bm_l_m
        }
        HydrostaticObservable::TonnesPerCentimetreImmersion => {
            properties.tonnes_per_centimetre_immersion
        }
    }
}

fn perturb_spec(
    source: &SectionHullSpec,
    parameter: SectionParameter,
    delta: f64,
) -> Result<SectionHullSpec, SensitivityViolation> {
    let mut result = source.clone();
    let value = match parameter {
        SectionParameter::LengthM => &mut result.length_m,
        SectionParameter::BeamM => &mut result.beam_m,
        SectionParameter::DraftM => &mut result.draft_m,
        SectionParameter::StationHalfBreadthFraction { station_index } => &mut result
            .stations
            .get_mut(station_index)
            .ok_or_else(|| SensitivityViolation {
                code: "parameter_station_index".into(),
                detail: format!("station index {station_index} is outside the source specification"),
            })?
            .half_breadth_fraction,
        SectionParameter::StationDraftFraction { station_index } => &mut result
            .stations
            .get_mut(station_index)
            .ok_or_else(|| SensitivityViolation {
                code: "parameter_station_index".into(),
                detail: format!("station index {station_index} is outside the source specification"),
            })?
            .draft_fraction,
        SectionParameter::ControlBreadthFraction {
            station_index,
            control_index,
        } => &mut result
            .stations
            .get_mut(station_index)
            .ok_or_else(|| SensitivityViolation {
                code: "parameter_station_index".into(),
                detail: format!("station index {station_index} is outside the source specification"),
            })?
            .control_points
            .get_mut(control_index)
            .ok_or_else(|| SensitivityViolation {
                code: "parameter_control_index".into(),
                detail: format!(
                    "control index {control_index} is outside station {station_index}"
                ),
            })?
            .breadth_fraction,
        SectionParameter::ControlDepthFraction {
            station_index,
            control_index,
        } => &mut result
            .stations
            .get_mut(station_index)
            .ok_or_else(|| SensitivityViolation {
                code: "parameter_station_index".into(),
                detail: format!("station index {station_index} is outside the source specification"),
            })?
            .control_points
            .get_mut(control_index)
            .ok_or_else(|| SensitivityViolation {
                code: "parameter_control_index".into(),
                detail: format!(
                    "control index {control_index} is outside station {station_index}"
                ),
            })?
            .depth_fraction,
    };
    *value += delta;
    Ok(result)
}

fn evaluate_spec(
    spec: &SectionHullSpec,
    profile: &AdmissionProfile,
    water_density_kg_m3: f64,
) -> Result<(String, String, HydrostaticProperties), Vec<SensitivityViolation>> {
    let section_artifact = generate_section_artifact(spec).map_err(|errors| {
        errors
            .into_iter()
            .map(|item: SectionViolation| SensitivityViolation {
                code: format!("section:{}", item.code),
                detail: item.detail,
            })
            .collect::<Vec<_>>()
    })?;
    let lofted: LoftedHullArtifact =
        loft_waterplane_capped_hull(&section_artifact, profile.clone()).map_err(|errors| {
            errors
                .into_iter()
                .map(|item: LoftViolation| SensitivityViolation {
                    code: format!("loft:{}", item.code),
                    detail: item.detail,
                })
                .collect::<Vec<_>>()
        })?;
    if lofted.admission_report.disposition != Disposition::AdmissibleForProfile {
        return Err(vec![SensitivityViolation {
            code: "mesh_admission".into(),
            detail: "perturbed geometry did not pass its declared mesh-integrity profile".into(),
        }]);
    }
    let properties = calculate_hydrostatic_properties(&lofted, water_density_kg_m3).map_err(
        |errors: Vec<HydrostaticViolation>| {
            errors
                .into_iter()
                .map(|item| SensitivityViolation {
                    code: format!("hydrostatics:{}", item.code),
                    detail: item.detail,
                })
                .collect::<Vec<_>>()
        },
    )?;
    Ok((
        compute_section_spec_digest(spec).map_err(|error| {
            vec![SensitivityViolation {
                code: "spec_digest".into(),
                detail: format!("could not digest section spec: {error}"),
            }]
        })?,
        lofted.candidate.geometry_digest_blake3,
        properties,
    ))
}

/// Compute the deterministic digest of a sensitivity receipt payload.
/// The digest field is cleared before serialization to avoid recursion.
pub fn compute_sensitivity_digest(
    result: &HydrostaticSensitivity,
) -> Result<String, serde_json::Error> {
    let mut payload = result.clone();
    payload.result_digest_blake3.clear();
    let bytes = serde_json::to_vec(&payload)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Verify all sensitivity provenance fields remain bound to the result digest.
pub fn verify_sensitivity_result(
    result: &HydrostaticSensitivity,
) -> Result<(), SensitivityViolation> {
    let actual = compute_sensitivity_digest(result).map_err(|error| SensitivityViolation {
        code: "sensitivity_digest".into(),
        detail: format!("could not serialize sensitivity result: {error}"),
    })?;
    if actual.eq_ignore_ascii_case(&result.result_digest_blake3) {
        Ok(())
    } else {
        Err(SensitivityViolation {
            code: "sensitivity_digest_mismatch".into(),
            detail: "sensitivity receipt payload does not match its result digest".into(),
        })
    }
}

/// Compute a central finite-difference derivative of one hydrostatic
/// observable with respect to one section-family parameter.
///
/// Both plus and minus perturbations must independently generate an
/// admissible loft and hydrostatic result. No one-sided fallback, clamping,
/// mesh repair, resampling or tolerance change is permitted.
pub fn calculate_hydrostatic_sensitivity(
    source: &SectionHullSpec,
    profile: &AdmissionProfile,
    parameter: SectionParameter,
    step: f64,
    observable: HydrostaticObservable,
    water_density_kg_m3: f64,
    water_density_source: &str,
) -> Result<HydrostaticSensitivity, Vec<SensitivityViolation>> {
    if !step.is_finite() || step <= 0.0 {
        return Err(vec![SensitivityViolation {
            code: "finite_difference_step".into(),
            detail: "finite-difference step must be finite and strictly positive".into(),
        }]);
    }

    if water_density_source.trim().is_empty() {
        return Err(vec![SensitivityViolation {
            code: "water_density_source".into(),
            detail: "water density source/assumption identifier must be explicit".into(),
        }]);
    }
    if !water_density_kg_m3.is_finite() || water_density_kg_m3 <= 0.0 {
        return Err(vec![SensitivityViolation {
            code: "water_density".into(),
            detail: "water density must be finite and strictly positive in kg/m^3".into(),
        }]);
    }

    let (parameter_code, parameter_units) = parameter_metadata(parameter);
    let minus_spec = perturb_spec(source, parameter, -step).map_err(|error| vec![error])?;
    let plus_spec = perturb_spec(source, parameter, step).map_err(|error| vec![error])?;

    let half_step = step * 0.5;
    if !half_step.is_finite() || half_step <= 0.0 {
        return Err(vec![SensitivityViolation {
            code: "half_step_underflow".into(),
            detail: "half-step underflowed or is not representable".into(),
        }]);
    }
    let minus_half_spec =
        perturb_spec(source, parameter, -half_step).map_err(|error| vec![error])?;
    let plus_half_spec =
        perturb_spec(source, parameter, half_step).map_err(|error| vec![error])?;

    let (base_spec_digest, base_geometry_digest, base_properties) =
        evaluate_spec(source, profile, water_density_kg_m3)?;
    let (minus_spec_digest, minus_geometry_digest, minus_properties) =
        evaluate_spec(&minus_spec, profile, water_density_kg_m3).map_err(|mut errors| {
            errors.insert(
                0,
                SensitivityViolation {
                    code: "minus_perturbation_invalid".into(),
                    detail: "negative h perturbation did not pass the unchanged geometry/hydrostatic pipeline".into(),
                },
            );
            errors
        })?;
    let (plus_spec_digest, plus_geometry_digest, plus_properties) =
        evaluate_spec(&plus_spec, profile, water_density_kg_m3).map_err(|mut errors| {
            errors.insert(
                0,
                SensitivityViolation {
                    code: "plus_perturbation_invalid".into(),
                    detail: "positive h perturbation did not pass the unchanged geometry/hydrostatic pipeline".into(),
                },
            );
            errors
        })?;
    let (minus_half_spec_digest, minus_half_geometry_digest, minus_half_properties) =
        evaluate_spec(&minus_half_spec, profile, water_density_kg_m3).map_err(|mut errors| {
            errors.insert(
                0,
                SensitivityViolation {
                    code: "minus_half_step_invalid".into(),
                    detail: "negative h/2 perturbation did not pass the unchanged geometry/hydrostatic pipeline".into(),
                },
            );
            errors
        })?;
    let (plus_half_spec_digest, plus_half_geometry_digest, plus_half_properties) =
        evaluate_spec(&plus_half_spec, profile, water_density_kg_m3).map_err(|mut errors| {
            errors.insert(
                0,
                SensitivityViolation {
                    code: "plus_half_step_invalid".into(),
                    detail: "positive h/2 perturbation did not pass the unchanged geometry/hydrostatic pipeline".into(),
                },
            );
            errors
        })?;

    let base_value = observable_value(&base_properties, observable);
    let minus_value = observable_value(&minus_properties, observable);
    let plus_value = observable_value(&plus_properties, observable);
    let minus_half_step_value = observable_value(&minus_half_properties, observable);
    let plus_half_step_value = observable_value(&plus_half_properties, observable);
    let coarse_derivative = (plus_value - minus_value) / (2.0 * step);
    let fine_derivative = (plus_half_step_value - minus_half_step_value) / (2.0 * half_step);
    let estimated_truncation_error = (fine_derivative - coarse_derivative).abs() / 3.0;
    let derivative = fine_derivative + (fine_derivative - coarse_derivative) / 3.0;

    if [
        base_value,
        minus_value,
        plus_value,
        minus_half_step_value,
        plus_half_step_value,
        coarse_derivative,
        fine_derivative,
        estimated_truncation_error,
        derivative,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(vec![SensitivityViolation {
            code: "non_finite_sensitivity".into(),
            detail: "base, perturbed or derivative hydrostatic value is non-finite".into(),
        }]);
    }

    let profile_bytes = serde_json::to_vec(profile).map_err(|error| {
        vec![SensitivityViolation {
            code: "profile_digest".into(),
            detail: format!("could not serialize admission profile: {error}"),
        }]
    })?;
    let profile_digest_blake3 = blake3::hash(&profile_bytes).to_hex().to_string();

    let mut result = HydrostaticSensitivity {
        result_schema_version: 1,
        candidate_id: source.candidate_id.clone(),
        profile_id: profile.id.clone(),
        profile_digest_blake3,
        parameter,
        parameter_code: parameter_code.into(),
        parameter_units: parameter_units.into(),
        observable,
        observable_units: observable_units(observable).into(),
        finite_difference_step: step,
        water_density_kg_m3,
        water_density_source: water_density_source.to_owned(),
        base_spec_digest_blake3: base_spec_digest,
        minus_spec_digest_blake3: minus_spec_digest,
        plus_spec_digest_blake3: plus_spec_digest,
        minus_half_step_spec_digest_blake3: minus_half_spec_digest,
        plus_half_step_spec_digest_blake3: plus_half_spec_digest,
        base_geometry_digest_blake3: base_geometry_digest,
        minus_geometry_digest_blake3: minus_geometry_digest,
        plus_geometry_digest_blake3: plus_geometry_digest,
        minus_half_step_geometry_digest_blake3: minus_half_geometry_digest,
        plus_half_step_geometry_digest_blake3: plus_half_geometry_digest,
        base_value,
        minus_value,
        plus_value,
        minus_half_step_value,
        plus_half_step_value,
        coarse_derivative_per_parameter_unit: coarse_derivative,
        fine_derivative_per_parameter_unit: fine_derivative,
        estimated_truncation_error,
        derivative_per_parameter_unit: derivative,
        result_digest_blake3: String::new(),
    };
    result.result_digest_blake3 =
        compute_sensitivity_digest(&result).map_err(|error| {
            vec![SensitivityViolation {
                code: "sensitivity_digest".into(),
                detail: format!("could not serialize sensitivity receipt: {error}"),
            }]
        })?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::section::{SectionControlPoint, SectionStation};
    use crate::CoordinateFrame;

    fn prismatic_spec() -> SectionHullSpec {
        let controls = vec![
            SectionControlPoint {
                breadth_fraction: 0.0,
                depth_fraction: 1.0,
            },
            SectionControlPoint {
                breadth_fraction: 1.0,
                depth_fraction: 0.0,
            },
        ];
        SectionHullSpec {
            schema_version: 1,
            candidate_id: "sensitivity-prism-001".into(),
            length_m: 100.0,
            beam_m: 20.0,
            draft_m: 6.0,
            coordinate_frame: CoordinateFrame {
                id: "ship-local".into(),
                axis_convention: "x-forward,y-starboard,z-down".into(),
                handedness: "right-handed".into(),
            },
            samples_per_segment: 1,
            stations: vec![
                SectionStation {
                    x_over_length: 0.0,
                    half_breadth_fraction: 1.0,
                    draft_fraction: 1.0,
                    control_points: controls.clone(),
                },
                SectionStation {
                    x_over_length: 1.0,
                    half_breadth_fraction: 1.0,
                    draft_fraction: 1.0,
                    control_points: controls,
                },
            ],
        }
    }

    fn profile() -> AdmissionProfile {
        AdmissionProfile {
            id: "sensitivity-prism-v1".into(),
            require_closed_surface: true,
            minimum_face_area_m2: 1.0e-12,
            minimum_enclosed_volume_m3: 1.0e-12,
            additional_required_checks: vec![],
        }
    }

    #[test]
    fn d_volume_d_length_matches_triangular_prism_analytic_value() {
        let result = calculate_hydrostatic_sensitivity(
            &prismatic_spec(),
            &profile(),
            SectionParameter::LengthM,
            0.1,
            HydrostaticObservable::DisplacementVolumeM3,
            1_025.0,
            "synthetic-prism-known-answer",
        )
        .unwrap();
        assert!((result.base_value - 6_000.0).abs() < 1.0e-9);
        assert!((result.derivative_per_parameter_unit - 60.0).abs() < 1.0e-8);
        assert_ne!(
            result.minus_spec_digest_blake3,
            result.plus_spec_digest_blake3
        );
        assert_ne!(
            result.minus_geometry_digest_blake3,
            result.plus_geometry_digest_blake3
        );
    }

    #[test]
    fn d_volume_d_beam_matches_triangular_prism_analytic_value() {
        let result = calculate_hydrostatic_sensitivity(
            &prismatic_spec(),
            &profile(),
            SectionParameter::BeamM,
            0.1,
            HydrostaticObservable::DisplacementVolumeM3,
            1_025.0,
            "synthetic-prism-known-answer",
        )
        .unwrap();
        assert!((result.derivative_per_parameter_unit - 300.0).abs() < 1.0e-8);
    }

    #[test]
    fn d_volume_d_draft_matches_triangular_prism_analytic_value() {
        let result = calculate_hydrostatic_sensitivity(
            &prismatic_spec(),
            &profile(),
            SectionParameter::DraftM,
            0.1,
            HydrostaticObservable::DisplacementVolumeM3,
            1_025.0,
            "synthetic-prism-known-answer",
        )
        .unwrap();
        assert!((result.derivative_per_parameter_unit - 1_000.0).abs() < 1.0e-8);
    }

    #[test]
    fn d_lcb_d_length_matches_half_length_relation() {
        let result = calculate_hydrostatic_sensitivity(
            &prismatic_spec(),
            &profile(),
            SectionParameter::LengthM,
            0.1,
            HydrostaticObservable::LongitudinalCentreOfBuoyancyM,
            1_025.0,
            "synthetic-prism-known-answer",
        )
        .unwrap();
        assert!((result.base_value - 50.0).abs() < 1.0e-9);
        assert!((result.derivative_per_parameter_unit - 0.5).abs() < 1.0e-8);
    }

    #[test]
    fn sensitivity_receipt_digest_binds_both_step_sizes_and_profile() {
        let mut result = calculate_hydrostatic_sensitivity(
            &prismatic_spec(),
            &profile(),
            SectionParameter::LengthM,
            0.1,
            HydrostaticObservable::DisplacementVolumeM3,
            1_025.0,
            "synthetic-prism-known-answer",
        )
        .unwrap();
        assert!(verify_sensitivity_result(&result).is_ok());
        assert!(
            (result.coarse_derivative_per_parameter_unit
                - result.fine_derivative_per_parameter_unit)
                .abs()
                < 1.0e-8
        );
        assert!(result.estimated_truncation_error < 1.0e-8);
        result.water_density_source = "changed-source".into();
        assert_eq!(
            verify_sensitivity_result(&result).unwrap_err().code,
            "sensitivity_digest_mismatch"
        );
    }

    #[test]
    fn zero_or_non_finite_step_is_rejected() {
        let error = calculate_hydrostatic_sensitivity(
            &prismatic_spec(),
            &profile(),
            SectionParameter::LengthM,
            0.0,
            HydrostaticObservable::DisplacementVolumeM3,
            1_025.0,
            "synthetic-prism-known-answer",
        )
        .unwrap_err();
        assert!(error
            .iter()
            .any(|item| item.code == "finite_difference_step"));
    }

    #[test]
    fn out_of_domain_perturbation_is_not_clamped_or_changed_to_one_sided() {
        let mut spec = prismatic_spec();
        spec.stations[0].half_breadth_fraction = 0.05;
        let error = calculate_hydrostatic_sensitivity(
            &spec,
            &profile(),
            SectionParameter::StationHalfBreadthFraction { station_index: 0 },
            0.1,
            HydrostaticObservable::DisplacementVolumeM3,
            1_025.0,
            "synthetic-prism-known-answer",
        )
        .unwrap_err();
        assert!(error
            .iter()
            .any(|item| item.code == "minus_perturbation_invalid"));
    }

    #[test]
    fn invalid_parameter_index_is_rejected() {
        let error = calculate_hydrostatic_sensitivity(
            &prismatic_spec(),
            &profile(),
            SectionParameter::StationHalfBreadthFraction { station_index: 99 },
            0.01,
            HydrostaticObservable::DisplacementVolumeM3,
            1_025.0,
            "synthetic-prism-known-answer",
        )
        .unwrap_err();
        assert!(error
            .iter()
            .any(|item| item.code == "parameter_station_index"));
    }
}
