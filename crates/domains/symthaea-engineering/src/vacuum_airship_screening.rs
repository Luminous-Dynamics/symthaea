// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conservative arithmetic screening for persistent-platform buoyancy and external pressure.
//!
//! This module deliberately does not predict structural strength. Buoyancy is calculated from
//! explicit SI inputs; pressure capacity is compared only when a caller supplies a capacity
//! record with non-empty provenance fields. Those fields are carried, not authenticated.
//! A declared ratio is not a safety factor, certification, flight-readiness result, or substitute
//! for imperfection-sensitive nonlinear analysis and physical tests.

use std::error::Error;
use std::fmt;

/// Invalid input or non-finite derived arithmetic in a screening calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreeningError {
    /// An input value was NaN or infinite.
    NonFiniteInput(&'static str),
    /// An input that must exceed zero was zero or negative.
    MustBePositive(&'static str),
    /// A mass or density that must be non-negative was negative.
    MustBeNonNegative(&'static str),
    /// Arithmetic overflowed while deriving a result.
    NonFiniteResult(&'static str),
}

impl fmt::Display for ScreeningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput(name) => write!(f, "{name} must be finite"),
            Self::MustBePositive(name) => write!(f, "{name} must be greater than zero"),
            Self::MustBeNonNegative(name) => write!(f, "{name} must be non-negative"),
            Self::NonFiniteResult(name) => write!(f, "derived {name} is not finite"),
        }
    }
}

impl Error for ScreeningError {}

fn validate_finite(value: f64, name: &'static str) -> Result<(), ScreeningError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreeningError::NonFiniteInput(name))
    }
}

fn validate_positive(value: f64, name: &'static str) -> Result<(), ScreeningError> {
    validate_finite(value, name)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ScreeningError::MustBePositive(name))
    }
}

fn validate_non_negative(value: f64, name: &'static str) -> Result<(), ScreeningError> {
    validate_finite(value, name)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ScreeningError::MustBeNonNegative(name))
    }
}

fn validate_result(value: f64, name: &'static str) -> Result<(), ScreeningError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreeningError::NonFiniteResult(name))
    }
}

/// Exact sphere volume for a positive radius, useful for an initial reference geometry.
pub fn sphere_volume_m3(radius_m: f64) -> Result<f64, ScreeningError> {
    validate_positive(radius_m, "radius_m")?;
    let volume = (4.0 / 3.0) * std::f64::consts::PI * radius_m.powi(3);
    validate_result(volume, "sphere volume")?;
    Ok(volume)
}

/// Explicit non-payload mass terms. No mass is implicitly estimated or defaulted.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VehicleMassBreakdown {
    /// Pressure boundary, sealing skin, coatings, and attached envelope parts.
    pub envelope_and_seals_kg: f64,
    /// Primary frame, lattice members, joints, and bracing.
    pub frame_lattice_and_joints_kg: f64,
    /// Motors, propellers, and propulsion hardware.
    pub propulsion_kg: f64,
    /// Batteries and other stored-energy hardware.
    pub energy_storage_kg: f64,
    /// Avionics, navigation, communications, and control hardware.
    pub avionics_and_communications_kg: f64,
    /// Landing systems, thermal systems, pumps, redundancy, and all other non-payload mass.
    pub other_systems_kg: f64,
}

impl VehicleMassBreakdown {
    /// Validate each explicit mass and return total vehicle mass excluding payload.
    pub fn total_vehicle_mass_kg(self) -> Result<f64, ScreeningError> {
        let terms = [
            ("envelope_and_seals_kg", self.envelope_and_seals_kg),
            ("frame_lattice_and_joints_kg", self.frame_lattice_and_joints_kg),
            ("propulsion_kg", self.propulsion_kg),
            ("energy_storage_kg", self.energy_storage_kg),
            ("avionics_and_communications_kg", self.avionics_and_communications_kg),
            ("other_systems_kg", self.other_systems_kg),
        ];
        for (name, value) in terms {
            validate_non_negative(value, name)?;
        }
        let total = terms.iter().map(|(_, value)| *value).sum::<f64>();
        validate_result(total, "vehicle mass")?;
        Ok(total)
    }
}

/// Fully explicit inputs for an idealized buoyancy mass balance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuoyancyRequest {
    /// Enclosed displacement volume in cubic metres.
    pub enclosed_volume_m3: f64,
    /// Ambient air density at the declared operating condition, kg/m³.
    pub ambient_air_density_kg_m3: f64,
    /// Interior gas density at the same reference condition, kg/m³; use 0 for ideal vacuum.
    pub internal_gas_density_kg_m3: f64,
    /// Complete non-payload vehicle mass breakdown.
    pub vehicle_mass: VehicleMassBreakdown,
}

/// Arithmetic mass balance. A negative payload allowance means the vehicle is too heavy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuoyancyResult {
    /// Mass of displaced ambient air, kg.
    pub displaced_air_mass_kg: f64,
    /// Mass of gas inside the envelope, kg.
    pub internal_gas_mass_kg: f64,
    /// Ideal net buoyant mass before vehicle mass is subtracted, kg.
    pub net_buoyant_mass_kg: f64,
    /// Explicit sum of all non-payload vehicle masses, kg.
    pub non_payload_vehicle_mass_kg: f64,
    /// Maximum theoretical payload allowance, kg; signed to preserve negative results.
    pub payload_allowance_kg: f64,
    /// Payload allowance divided by net buoyant mass when that mass is positive.
    /// This can be negative and is an arithmetic ratio, not an operating margin.
    pub payload_fraction_of_net_buoyant_mass: Option<f64>,
}

/// Compute an idealized SI mass balance. No material strength or pressure resistance is inferred.
pub fn screen_buoyancy(request: BuoyancyRequest) -> Result<BuoyancyResult, ScreeningError> {
    validate_positive(request.enclosed_volume_m3, "enclosed_volume_m3")?;
    validate_positive(request.ambient_air_density_kg_m3, "ambient_air_density_kg_m3")?;
    validate_non_negative(request.internal_gas_density_kg_m3, "internal_gas_density_kg_m3")?;

    let displaced_air_mass_kg =
        request.ambient_air_density_kg_m3 * request.enclosed_volume_m3;
    let internal_gas_mass_kg =
        request.internal_gas_density_kg_m3 * request.enclosed_volume_m3;
    validate_result(displaced_air_mass_kg, "displaced air mass")?;
    validate_result(internal_gas_mass_kg, "internal gas mass")?;

    let net_buoyant_mass_kg = displaced_air_mass_kg - internal_gas_mass_kg;
    validate_result(net_buoyant_mass_kg, "net buoyant mass")?;
    let non_payload_vehicle_mass_kg = request.vehicle_mass.total_vehicle_mass_kg()?;
    let payload_allowance_kg = net_buoyant_mass_kg - non_payload_vehicle_mass_kg;
    validate_result(payload_allowance_kg, "payload allowance")?;

    let payload_fraction_of_net_buoyant_mass = if net_buoyant_mass_kg > 0.0 {
        let fraction = payload_allowance_kg / net_buoyant_mass_kg;
        validate_result(fraction, "payload fraction")?;
        Some(fraction)
    } else {
        None
    };

    Ok(BuoyancyResult {
        displaced_air_mass_kg,
        internal_gas_mass_kg,
        net_buoyant_mass_kg,
        non_payload_vehicle_mass_kg,
        payload_allowance_kg,
        payload_fraction_of_net_buoyant_mass,
    })
}

/// Geometry-based mass budget for an idealized spherical envelope.
///
/// The radius is used for both enclosed volume and pressure-boundary surface area. This is a
//// thin-envelope screening approximation: it ignores thickness-induced volume/area differences,
//// local appendages, deformation, and the mass of structural supports unless the caller includes
//// those in `fixed_non_envelope_vehicle_mass_kg`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalEnvelopeBudgetRequest {
    /// Reference radius of the spherical pressure boundary, metres.
    pub radius_m: f64,
    /// Ambient air density at the declared operating condition, kg/m³.
    pub ambient_air_density_kg_m3: f64,
    /// Interior gas density at the same reference condition, kg/m³; use 0 for ideal vacuum.
    pub internal_gas_density_kg_m3: f64,
    /// All non-envelope, non-payload mass, including frame, joints, propulsion and systems, kg.
    pub fixed_non_envelope_vehicle_mass_kg: f64,
    /// Desired payload mass, kg.
    pub target_payload_mass_kg: f64,
}

/// Signed upper bound for the spherical envelope's remaining mass budget.
///
/// A negative `maximum_envelope_mass_kg` or areal density means the target is already impossible
/// before an envelope is added. A non-negative value only passes arithmetic bookkeeping; it does
/// not imply the envelope can resist external pressure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalEnvelopeBudgetResult {
    /// Idealized enclosed volume, m³.
    pub enclosed_volume_m3: f64,
    /// Idealized pressure-boundary area, m².
    pub envelope_surface_area_m2: f64,
    /// Ideal net buoyant mass before vehicle mass is subtracted, kg.
    pub net_buoyant_mass_kg: f64,
    /// Declared mass of all non-envelope vehicle components, kg.
    pub fixed_non_envelope_vehicle_mass_kg: f64,
    /// Declared target payload mass, kg.
    pub target_payload_mass_kg: f64,
    /// Maximum mass left for the entire envelope and its directly attached layers, kg.
    pub maximum_envelope_mass_kg: f64,
    /// Maximum mean envelope areal mass, kg/m²; signed so infeasibility is preserved.
    pub maximum_envelope_areal_density_kg_m2: f64,
}

/// Determine the mass ceiling available to a spherical pressure envelope.
///
/// This is useful for quickly rejecting candidate materials/architectures that exceed the
/// buoyancy budget. It cannot establish a minimum required thickness or pressure capacity.
pub fn screen_spherical_envelope_budget(
    request: SphericalEnvelopeBudgetRequest,
) -> Result<SphericalEnvelopeBudgetResult, ScreeningError> {
    validate_positive(request.radius_m, "radius_m")?;
    validate_positive(
        request.ambient_air_density_kg_m3,
        "ambient_air_density_kg_m3",
    )?;
    validate_non_negative(
        request.internal_gas_density_kg_m3,
        "internal_gas_density_kg_m3",
    )?;
    validate_non_negative(
        request.fixed_non_envelope_vehicle_mass_kg,
        "fixed_non_envelope_vehicle_mass_kg",
    )?;
    validate_non_negative(request.target_payload_mass_kg, "target_payload_mass_kg")?;

    let enclosed_volume_m3 = sphere_volume_m3(request.radius_m)?;
    let envelope_surface_area_m2 =
        4.0 * std::f64::consts::PI * request.radius_m.powi(2);
    validate_result(envelope_surface_area_m2, "envelope surface area")?;

    let net_buoyant_mass_kg = (request.ambient_air_density_kg_m3
        - request.internal_gas_density_kg_m3)
        * enclosed_volume_m3;
    validate_result(net_buoyant_mass_kg, "net buoyant mass")?;

    let maximum_envelope_mass_kg = net_buoyant_mass_kg
        - request.fixed_non_envelope_vehicle_mass_kg
        - request.target_payload_mass_kg;
    validate_result(maximum_envelope_mass_kg, "maximum envelope mass")?;

    let maximum_envelope_areal_density_kg_m2 =
        maximum_envelope_mass_kg / envelope_surface_area_m2;
    validate_result(
        maximum_envelope_areal_density_kg_m2,
        "maximum envelope areal density",
    )?;

    Ok(SphericalEnvelopeBudgetResult {
        enclosed_volume_m3,
        envelope_surface_area_m2,
        net_buoyant_mass_kg,
        fixed_non_envelope_vehicle_mass_kg: request.fixed_non_envelope_vehicle_mass_kg,
        target_payload_mass_kg: request.target_payload_mass_kg,
        maximum_envelope_mass_kg,
        maximum_envelope_areal_density_kg_m2,
    })
}

/// An externally sourced pressure-capacity declaration plus the references needed to audit it.
///
/// The caller must supply a conservative allowable pressure for the exact applicable design
/// generation, not merely a perfect-geometry linear eigenvalue. This structure checks that
/// references are present; it cannot authenticate a solver, test, or independence claim.
#[derive(Debug, Clone, PartialEq)]
pub struct PressureCapacityRecord {
    /// Declared allowable external differential pressure, Pa.
    pub allowable_external_pressure_pa: f64,
    /// Immutable record identifying the capacity result.
    pub evidence_record_id: String,
    /// Exact model/test profile and configuration reference.
    pub basis_profile_id: String,
    /// Independent check or physical-test record reference.
    pub independent_check_or_test_id: String,
}

/// Explicit external-pressure case. All pressures are absolute SI pascals.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalPressureRequest {
    pub external_pressure_pa: f64,
    pub internal_pressure_pa: f64,
    pub capacity: Option<PressureCapacityRecord>,
}

/// Pressure comparison status. No variant declares a design safe or flight-ready.
#[derive(Debug, Clone, PartialEq)]
pub enum PressureAssessment {
    /// Interior pressure equals or exceeds exterior pressure: no positive external differential.
    NoExternalPressureDifferential {
        differential_pressure_pa: f64,
    },
    /// No external capacity record was supplied, so resistance is unknown to this screen.
    CapacityUnknown {
        differential_pressure_pa: f64,
    },
    /// Capacity was supplied, but one or more audit references are absent.
    IncompleteProvenance {
        differential_pressure_pa: f64,
        missing_fields: Vec<&'static str>,
    },
    /// Arithmetic comparison only; the references have not been authenticated by this module.
    DeclaredCapacityComparison {
        differential_pressure_pa: f64,
        allowable_external_pressure_pa: f64,
        declared_capacity_ratio: f64,
        evidence_record_id: String,
        basis_profile_id: String,
        independent_check_or_test_id: String,
    },
}

/// Compare an explicit external differential pressure against an externally qualified capacity.
///
/// A ratio above 1.0 means only that the declared capacity exceeds this declared pressure load.
/// It does not account for mismatched model scope, unmodeled damage, reliability, certification,
/// dynamic loading, or whether a purported independent review is actually independent.
pub fn assess_external_pressure(
    request: ExternalPressureRequest,
) -> Result<PressureAssessment, ScreeningError> {
    validate_non_negative(request.external_pressure_pa, "external_pressure_pa")?;
    validate_non_negative(request.internal_pressure_pa, "internal_pressure_pa")?;

    let differential_pressure_pa = request.external_pressure_pa - request.internal_pressure_pa;
    validate_result(differential_pressure_pa, "differential pressure")?;
    if differential_pressure_pa <= 0.0 {
        return Ok(PressureAssessment::NoExternalPressureDifferential {
            differential_pressure_pa,
        });
    }

    let Some(capacity) = request.capacity else {
        return Ok(PressureAssessment::CapacityUnknown {
            differential_pressure_pa,
        });
    };
    validate_positive(
        capacity.allowable_external_pressure_pa,
        "allowable_external_pressure_pa",
    )?;

    let mut missing_fields = Vec::new();
    if capacity.evidence_record_id.trim().is_empty() {
        missing_fields.push("evidence_record_id");
    }
    if capacity.basis_profile_id.trim().is_empty() {
        missing_fields.push("basis_profile_id");
    }
    if capacity.independent_check_or_test_id.trim().is_empty() {
        missing_fields.push("independent_check_or_test_id");
    }
    if !missing_fields.is_empty() {
        return Ok(PressureAssessment::IncompleteProvenance {
            differential_pressure_pa,
            missing_fields,
        });
    }

    let declared_capacity_ratio =
        capacity.allowable_external_pressure_pa / differential_pressure_pa;
    validate_result(declared_capacity_ratio, "declared capacity ratio")?;

    Ok(PressureAssessment::DeclaredCapacityComparison {
        differential_pressure_pa,
        allowable_external_pressure_pa: capacity.allowable_external_pressure_pa,
        declared_capacity_ratio,
        evidence_record_id: capacity.evidence_record_id,
        basis_profile_id: capacity.basis_profile_id,
        independent_check_or_test_id: capacity.independent_check_or_test_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_mass() -> VehicleMassBreakdown {
        VehicleMassBreakdown::default()
    }

    fn qualified_record(capacity_pa: f64) -> PressureCapacityRecord {
        PressureCapacityRecord {
            allowable_external_pressure_pa: capacity_pa,
            evidence_record_id: "evidence://pressure-result/42".into(),
            basis_profile_id: "profile://nonlinear-imperfect-shell/v1".into(),
            independent_check_or_test_id: "evidence://independent-check/7".into(),
        }
    }

    #[test]
    fn unit_sphere_volume_matches_analytic_reference() {
        let volume = sphere_volume_m3(1.0).unwrap();
        assert!((volume - 4.0 * std::f64::consts::PI / 3.0).abs() < 1e-12);
    }

    #[test]
    fn twenty_metre_radius_vacuum_sphere_has_expected_gross_lift() {
        let volume = sphere_volume_m3(20.0).unwrap();
        let result = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: volume,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.0,
            vehicle_mass: zero_mass(),
        })
        .unwrap();
        assert!((result.net_buoyant_mass_kg - 41_050.14400690663).abs() < 1e-6);
        assert_eq!(result.internal_gas_mass_kg, 0.0);
        assert_eq!(result.payload_allowance_kg, result.net_buoyant_mass_kg);
    }

    #[test]
    fn twenty_metre_payload_allowance_subtracts_every_explicit_mass() {
        let volume = sphere_volume_m3(20.0).unwrap();
        let mass = VehicleMassBreakdown {
            envelope_and_seals_kg: 10.0,
            frame_lattice_and_joints_kg: 20.0,
            propulsion_kg: 30.0,
            energy_storage_kg: 40.0,
            avionics_and_communications_kg: 5.0,
            other_systems_kg: 5.0,
        };
        let result = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: volume,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.0,
            vehicle_mass: mass,
        })
        .unwrap();
        assert_eq!(result.non_payload_vehicle_mass_kg, 110.0);
        assert!((result.payload_allowance_kg - (41_050.14400690663 - 110.0)).abs() < 1e-6);
    }

    #[test]
    fn heavier_than_buoyancy_vehicle_keeps_negative_payload_allowance() {
        let result = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: 1.0,
            ambient_air_density_kg_m3: 1.0,
            internal_gas_density_kg_m3: 0.0,
            vehicle_mass: VehicleMassBreakdown {
                frame_lattice_and_joints_kg: 2.0,
                ..zero_mass()
            },
        })
        .unwrap();
        assert_eq!(result.payload_allowance_kg, -1.0);
        assert_eq!(result.payload_fraction_of_net_buoyant_mass, Some(-1.0));
    }

    #[test]
    fn nonzero_internal_gas_mass_reduces_net_lift() {
        let vacuum = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: 100.0,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.0,
            vehicle_mass: zero_mass(),
        })
        .unwrap();
        let helium_like = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: 100.0,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.169_280_218_717_622_25,
            vehicle_mass: zero_mass(),
        })
        .unwrap();
        assert!(helium_like.net_buoyant_mass_kg < vacuum.net_buoyant_mass_kg);
    }

    #[test]
    fn negative_mass_is_rejected() {
        let result = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: 1.0,
            ambient_air_density_kg_m3: 1.0,
            internal_gas_density_kg_m3: 0.0,
            vehicle_mass: VehicleMassBreakdown {
                propulsion_kg: -1.0,
                ..zero_mass()
            },
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::MustBeNonNegative("propulsion_kg")
        );
    }

    #[test]
    fn nan_density_is_rejected() {
        let result = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: 1.0,
            ambient_air_density_kg_m3: f64::NAN,
            internal_gas_density_kg_m3: 0.0,
            vehicle_mass: zero_mass(),
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::NonFiniteInput("ambient_air_density_kg_m3")
        );
    }

    #[test]
    fn twenty_metre_vacuum_sphere_has_8_1667_kg_m2_envelope_ceiling_before_other_mass() {
        let result = screen_spherical_envelope_budget(SphericalEnvelopeBudgetRequest {
            radius_m: 20.0,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.0,
            fixed_non_envelope_vehicle_mass_kg: 0.0,
            target_payload_mass_kg: 0.0,
        })
        .unwrap();

        // For a spherical shell with no other mass or payload:
        // (rho * 4/3*pi*R^3) / (4*pi*R^2) = rho*R/3.
        assert!(
            (result.maximum_envelope_areal_density_kg_m2 - (1.225 * 20.0 / 3.0)).abs() < 1e-12
        );
    }

    #[test]
    fn fixed_vehicle_mass_and_target_payload_reduce_envelope_budget() {
        let result = screen_spherical_envelope_budget(SphericalEnvelopeBudgetRequest {
            radius_m: 20.0,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.0,
            fixed_non_envelope_vehicle_mass_kg: 10_000.0,
            target_payload_mass_kg: 5_000.0,
        })
        .unwrap();

        assert!(
            (result.maximum_envelope_mass_kg - (41_050.14400690663 - 15_000.0)).abs() < 1e-6
        );
        assert!(result.maximum_envelope_areal_density_kg_m2 < 8.166666666666667);
    }

    #[test]
    fn envelope_budget_preserves_infeasible_negative_mass_ceiling() {
        let result = screen_spherical_envelope_budget(SphericalEnvelopeBudgetRequest {
            radius_m: 1.0,
            ambient_air_density_kg_m3: 1.0,
            internal_gas_density_kg_m3: 0.0,
            fixed_non_envelope_vehicle_mass_kg: 100.0,
            target_payload_mass_kg: 1.0,
        })
        .unwrap();

        assert!(result.maximum_envelope_mass_kg < 0.0);
        assert!(result.maximum_envelope_areal_density_kg_m2 < 0.0);
    }

    #[test]
    fn negative_target_payload_is_rejected() {
        let result = screen_spherical_envelope_budget(SphericalEnvelopeBudgetRequest {
            radius_m: 1.0,
            ambient_air_density_kg_m3: 1.0,
            internal_gas_density_kg_m3: 0.0,
            fixed_non_envelope_vehicle_mass_kg: 0.0,
            target_payload_mass_kg: -1.0,
        });

        assert_eq!(
            result.unwrap_err(),
            ScreeningError::MustBeNonNegative("target_payload_mass_kg")
        );
    }

    #[test]
    fn absent_pressure_capacity_stays_unknown() {
        let assessment = assess_external_pressure(ExternalPressureRequest {
            external_pressure_pa: 101_325.0,
            internal_pressure_pa: 0.0,
            capacity: None,
        })
        .unwrap();
        assert_eq!(
            assessment,
            PressureAssessment::CapacityUnknown {
                differential_pressure_pa: 101_325.0
            }
        );
    }

    #[test]
    fn incomplete_pressure_provenance_blocks_ratio() {
        let assessment = assess_external_pressure(ExternalPressureRequest {
            external_pressure_pa: 101_325.0,
            internal_pressure_pa: 0.0,
            capacity: Some(PressureCapacityRecord {
                allowable_external_pressure_pa: 250_000.0,
                evidence_record_id: String::new(),
                basis_profile_id: "profile/v1".into(),
                independent_check_or_test_id: "check/1".into(),
            }),
        })
        .unwrap();
        assert_eq!(
            assessment,
            PressureAssessment::IncompleteProvenance {
                differential_pressure_pa: 101_325.0,
                missing_fields: vec!["evidence_record_id"]
            }
        );
    }

    #[test]
    fn declared_pressure_capacity_ratio_is_arithmetic_not_a_pass_flag() {
        let assessment = assess_external_pressure(ExternalPressureRequest {
            external_pressure_pa: 101_325.0,
            internal_pressure_pa: 0.0,
            capacity: Some(qualified_record(202_650.0)),
        })
        .unwrap();
        match assessment {
            PressureAssessment::DeclaredCapacityComparison {
                declared_capacity_ratio,
                ..
            } => assert!((declared_capacity_ratio - 2.0).abs() < 1e-12),
            other => panic!("unexpected assessment: {other:?}"),
        }
    }

    #[test]
    fn pressure_case_without_external_differential_does_not_claim_capacity() {
        let assessment = assess_external_pressure(ExternalPressureRequest {
            external_pressure_pa: 90_000.0,
            internal_pressure_pa: 101_325.0,
            capacity: Some(qualified_record(202_650.0)),
        })
        .unwrap();
        assert_eq!(
            assessment,
            PressureAssessment::NoExternalPressureDifferential {
                differential_pressure_pa: -11_325.0
            }
        );
    }

    #[test]
    fn zero_radius_is_rejected() {
        assert_eq!(
            sphere_volume_m3(0.0).unwrap_err(),
            ScreeningError::MustBePositive("radius_m")
        );
    }
}
