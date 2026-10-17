// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-carrying contract for candidate-specific mission demand profiles.
//!
//! This module validates the identity, provenance, mass closure, environment,
//! and phase-demand record supplied by an upstream mission model. It does not
//! compute aerodynamics, propulsion, transition performance, or flight control.
//! A validated contract means the record is internally consistent, not that
//! the cited model or its predictions are physically accurate.

use std::collections::HashSet;

use super::{DutyClass, EnergySystemEnvelope, MissionProfile, MissionSegment};

/// Content-addressed reference to a model, input dataset, or run artifact.
///
/// A digest provides identity only. The caller remains responsible for making
/// the referenced source resolvable and evaluating its validity/applicability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifactRef {
    pub locator: String,
    pub revision: String,
    pub sha256: String,
    pub retrieved_at: String,
    /// Units or an explicit "not applicable" declaration for this artifact.
    pub units: String,
    /// Reference state/conditions or an explicit "not applicable" declaration.
    pub reference_conditions: String,
    /// Supported conditions or an explicit non-physical-artifact declaration.
    pub applicability_domain: String,
    /// Quantified uncertainty or an explicit reason it does not apply.
    pub uncertainty_description: String,
    pub limitations: String,
}

/// Mutually exclusive mass accounting for one candidate configuration.
///
/// Fixed vehicle mass excludes the energy-system and propulsion masses below.
/// Energy-system mass must use the same boundary as EnergySystemEnvelope.
/// Payload includes all occupants, baggage, cargo, and mission equipment.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateMassProperties {
    pub fixed_vehicle_mass_kg: f64,
    pub energy_system_installed_mass_kg: f64,
    pub propulsion_installed_mass_kg: f64,
    /// Fuel, coolant, or other consumables not already counted above.
    pub consumables_mass_kg: f64,
    pub payload_mass_kg: f64,
    pub maximum_takeoff_mass_limit_kg: f64,
    /// Explicit coordinate frame/datum for x/y/z, not an implicit default.
    pub center_of_gravity_frame: String,
    /// x/y/z in that datum, in metres.
    pub center_of_gravity_m: [f64; 3],
}

impl CandidateMassProperties {
    /// Sum of the five non-overlapping mass terms, if the arithmetic is finite.
    pub fn mission_takeoff_mass_kg(&self) -> Option<f64> {
        let total = self.fixed_vehicle_mass_kg
            + self.energy_system_installed_mass_kg
            + self.propulsion_installed_mass_kg
            + self.consumables_mass_kg
            + self.payload_mass_kg;
        total.is_finite().then_some(total)
    }
}

/// Frozen environmental conditions passed to the upstream mission model.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionEnvironment {
    pub air_density_kg_m3: f64,
    pub ambient_temperature_k: f64,
    pub ambient_pressure_pa: f64,
    /// Explicit coordinate frame in which the wind vector is expressed.
    pub wind_reference_frame: String,
    /// Wind vector in that local frame, metres per second.
    pub wind_vector_mps: [f64; 3],
}

/// Explicit uncertainty interval for a phase power estimate.
///
/// These are engineering bounds, not a probability distribution or confidence
/// interval unless the method field says so and its assumptions are documented.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerUncertaintyBounds {
    pub lower_bound_w: f64,
    pub upper_bound_w: f64,
    pub method: String,
}

/// One phase load emitted by the upstream candidate-specific model.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionPhaseDemand {
    pub id: String,
    pub duration_s: f64,
    pub required_bus_power_w: f64,
    pub power_uncertainty: PowerUncertaintyBounds,
    pub duty_class: DutyClass,
}

/// Traceable model-run record. The two input digests bind the run to the
/// candidate configuration and energy-system envelope used by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateMissionModelRun {
    pub run_id: String,
    pub model_version: String,
    pub source_revision: String,
    pub run_artifact: ModelArtifactRef,
    /// Artifact for the canonical candidate configuration manifest.
    pub configuration_artifact: ModelArtifactRef,
    pub model_artifact: ModelArtifactRef,
    pub aerodynamics_artifact: ModelArtifactRef,
    pub propulsion_artifact: ModelArtifactRef,
    pub energy_system_artifact: ModelArtifactRef,
    pub mass_properties_artifact: ModelArtifactRef,
    pub environment_artifact: ModelArtifactRef,
    /// Content-addressed phase list including nominal power and uncertainty bounds.
    pub phase_demand_artifact: ModelArtifactRef,
    pub route_artifact: ModelArtifactRef,
    pub mission_family_artifact: ModelArtifactRef,
    pub input_configuration_sha256: String,
    pub input_energy_system_envelope_sha256: String,
    pub input_route_sha256: String,
    pub input_mission_family_sha256: String,
    pub input_mass_properties_sha256: String,
    pub input_environment_sha256: String,
    pub output_phase_demand_sha256: String,
}

/// Complete provenance-bearing output from a candidate-specific mission model.
///
/// Digests are supplied by the producer; this crate validates syntax and
/// equality bindings but does not canonicalize/serialize inputs or recompute
/// their SHA-256 hashes. The upstream tool must document the canonicalization
/// process and make every referenced artifact independently retrievable.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateMissionManifest {
    pub candidate_id: String,
    pub mission_family_id: String,
    pub configuration_sha256: String,
    pub energy_system_envelope_sha256: String,
    pub route_sha256: String,
    pub mission_family_sha256: String,
    pub mass_properties_sha256: String,
    pub environment_sha256: String,
    pub phase_demand_sha256: String,
    pub route_distance_m: f64,
    pub reserve_fraction: f64,
    pub mass_properties: CandidateMassProperties,
    pub environment: MissionEnvironment,
    pub model_run: CandidateMissionModelRun,
    pub phases: Vec<MissionPhaseDemand>,
}

/// Contract validation finding. Any finding fails closed.
#[derive(Debug, Clone, PartialEq)]
pub enum MissionModelContractFinding {
    MissingField { field: String },
    InvalidValue { field: String, reason: String },
    InvalidSha256 { field: String },
    DigestMismatch { field: String },
    DuplicatePhaseId { id: String },
    MassLimitExceeded { total_mass_kg: f64, limit_kg: f64 },
    AccumulationOverflow { field: String },
}

/// Validated mission profile returned with the source manifest and its nominal
/// energy total so downstream reports can preserve provenance and uncertainty.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedCandidateMissionProfile {
    pub profile: MissionProfile,
    pub manifest: CandidateMissionManifest,
    pub nominal_total_energy_wh: f64,
}

/// Validate the candidate/model-run contract and construct the mission profile
/// consumed by the existing energy screen.
///
/// This function verifies consistency, not accuracy. It confirms that the
/// model run declares the same candidate and system-envelope digests, that
/// model/data artifacts have versioned content references, that mass closes,
/// and that phase inputs are valid. The returned object retains the manifest
/// and uncertainty intervals. It does not verify the underlying physics.
pub fn validate_candidate_mission_manifest(
    manifest: &CandidateMissionManifest,
    energy_system: &EnergySystemEnvelope,
) -> Result<ValidatedCandidateMissionProfile, Vec<MissionModelContractFinding>> {
    let mut findings = Vec::new();

    for (field, value) in [
        ("candidate_id", manifest.candidate_id.as_str()),
        ("mission_family_id", manifest.mission_family_id.as_str()),
        ("model_run.run_id", manifest.model_run.run_id.as_str()),
        ("model_run.model_version", manifest.model_run.model_version.as_str()),
        ("model_run.source_revision", manifest.model_run.source_revision.as_str()),
    ] {
        validate_nonempty(field, value, &mut findings);
    }

    validate_digest("configuration_sha256", &manifest.configuration_sha256, &mut findings);
    validate_digest(
        "energy_system_envelope_sha256",
        &manifest.energy_system_envelope_sha256,
        &mut findings,
    );
    validate_digest("route_sha256", &manifest.route_sha256, &mut findings);
    validate_digest(
        "mission_family_sha256",
        &manifest.mission_family_sha256,
        &mut findings,
    );
    validate_digest(
        "mass_properties_sha256",
        &manifest.mass_properties_sha256,
        &mut findings,
    );
    validate_digest(
        "environment_sha256",
        &manifest.environment_sha256,
        &mut findings,
    );
    validate_digest(
        "phase_demand_sha256",
        &manifest.phase_demand_sha256,
        &mut findings,
    );
    validate_digest(
        "model_run.output_phase_demand_sha256",
        &manifest.model_run.output_phase_demand_sha256,
        &mut findings,
    );
    validate_digest(
        "model_run.input_configuration_sha256",
        &manifest.model_run.input_configuration_sha256,
        &mut findings,
    );
    validate_digest(
        "model_run.input_energy_system_envelope_sha256",
        &manifest.model_run.input_energy_system_envelope_sha256,
        &mut findings,
    );
    validate_digest("model_run.input_route_sha256", &manifest.model_run.input_route_sha256, &mut findings);
    validate_digest(
        "model_run.input_mission_family_sha256",
        &manifest.model_run.input_mission_family_sha256,
        &mut findings,
    );
    validate_digest(
        "model_run.input_mass_properties_sha256",
        &manifest.model_run.input_mass_properties_sha256,
        &mut findings,
    );
    validate_digest(
        "model_run.input_environment_sha256",
        &manifest.model_run.input_environment_sha256,
        &mut findings,
    );

    if !manifest.configuration_sha256.is_empty()
        && !manifest.model_run.input_configuration_sha256.is_empty()
        && !manifest.configuration_sha256.eq_ignore_ascii_case(
            &manifest.model_run.input_configuration_sha256,
        )
    {
        findings.push(MissionModelContractFinding::DigestMismatch {
            field: "model_run.input_configuration_sha256".into(),
        });
    }
    if !manifest.energy_system_envelope_sha256.is_empty()
        && !manifest.model_run.input_energy_system_envelope_sha256.is_empty()
        && !manifest.energy_system_envelope_sha256.eq_ignore_ascii_case(
            &manifest.model_run.input_energy_system_envelope_sha256,
        )
    {
        findings.push(MissionModelContractFinding::DigestMismatch {
            field: "model_run.input_energy_system_envelope_sha256".into(),
        });
    }

    for ((
        expected_field,
        expected,
        actual_field,
        actual,
    )) in [
        (
            "route_sha256",
            manifest.route_sha256.as_str(),
            "model_run.input_route_sha256",
            manifest.model_run.input_route_sha256.as_str(),
        ),
        (
            "mission_family_sha256",
            manifest.mission_family_sha256.as_str(),
            "model_run.input_mission_family_sha256",
            manifest.model_run.input_mission_family_sha256.as_str(),
        ),
        (
            "mass_properties_sha256",
            manifest.mass_properties_sha256.as_str(),
            "model_run.input_mass_properties_sha256",
            manifest.model_run.input_mass_properties_sha256.as_str(),
        ),
        (
            "environment_sha256",
            manifest.environment_sha256.as_str(),
            "model_run.input_environment_sha256",
            manifest.model_run.input_environment_sha256.as_str(),
        ),
    ] {
        validate_digest_binding(expected_field, expected, actual_field, actual, &mut findings);
    }

    for (field, artifact) in [
        ("model_run.run_artifact", &manifest.model_run.run_artifact),
        ("model_run.configuration_artifact", &manifest.model_run.configuration_artifact),
        ("model_run.model_artifact", &manifest.model_run.model_artifact),
        ("model_run.aerodynamics_artifact", &manifest.model_run.aerodynamics_artifact),
        ("model_run.propulsion_artifact", &manifest.model_run.propulsion_artifact),
        ("model_run.energy_system_artifact", &manifest.model_run.energy_system_artifact),
        ("model_run.mass_properties_artifact", &manifest.model_run.mass_properties_artifact),
        ("model_run.environment_artifact", &manifest.model_run.environment_artifact),
        ("model_run.phase_demand_artifact", &manifest.model_run.phase_demand_artifact),
        ("model_run.route_artifact", &manifest.model_run.route_artifact),
        ("model_run.mission_family_artifact", &manifest.model_run.mission_family_artifact),
    ] {
        validate_artifact(field, artifact, &mut findings);
    }

    validate_digest_binding(
        "configuration_sha256",
        &manifest.configuration_sha256,
        "model_run.configuration_artifact.sha256",
        &manifest.model_run.configuration_artifact.sha256,
        &mut findings,
    );
    validate_digest_binding(
        "energy_system_envelope_sha256",
        &manifest.energy_system_envelope_sha256,
        "model_run.energy_system_artifact.sha256",
        &manifest.model_run.energy_system_artifact.sha256,
        &mut findings,
    );
    validate_digest_binding(
        "route_sha256",
        &manifest.route_sha256,
        "model_run.route_artifact.sha256",
        &manifest.model_run.route_artifact.sha256,
        &mut findings,
    );
    validate_digest_binding(
        "mission_family_sha256",
        &manifest.mission_family_sha256,
        "model_run.mission_family_artifact.sha256",
        &manifest.model_run.mission_family_artifact.sha256,
        &mut findings,
    );
    validate_digest_binding(
        "mass_properties_sha256",
        &manifest.mass_properties_sha256,
        "model_run.mass_properties_artifact.sha256",
        &manifest.model_run.mass_properties_artifact.sha256,
        &mut findings,
    );
    validate_digest_binding(
        "environment_sha256",
        &manifest.environment_sha256,
        "model_run.environment_artifact.sha256",
        &manifest.model_run.environment_artifact.sha256,
        &mut findings,
    );
    validate_digest_binding(
        "phase_demand_sha256",
        &manifest.phase_demand_sha256,
        "model_run.phase_demand_artifact.sha256",
        &manifest.model_run.phase_demand_artifact.sha256,
        &mut findings,
    );
    validate_digest_binding(
        "phase_demand_sha256",
        &manifest.phase_demand_sha256,
        "model_run.output_phase_demand_sha256",
        &manifest.model_run.output_phase_demand_sha256,
        &mut findings,
    );


    validate_positive("route_distance_m", manifest.route_distance_m, &mut findings);
    validate_fraction("reserve_fraction", manifest.reserve_fraction, &mut findings);
    validate_environment(&manifest.environment, &mut findings);
    validate_mass_properties(&manifest.mass_properties, &mut findings);

    match energy_system.installed_mass_kg {
        Some(value) if value.is_finite() && value > 0.0 => {
            if !approximately_equal(value, manifest.mass_properties.energy_system_installed_mass_kg) {
                findings.push(MissionModelContractFinding::InvalidValue {
                    field: "energy_system.installed_mass_kg".into(),
                    reason: "does not match the candidate mass-properties record".into(),
                });
            }
        }
        Some(_) => findings.push(MissionModelContractFinding::InvalidValue {
            field: "energy_system.installed_mass_kg".into(),
            reason: "must be finite and greater than zero".into(),
        }),
        None => findings.push(MissionModelContractFinding::MissingField {
            field: "energy_system.installed_mass_kg".into(),
        }),
    }
    match energy_system.reserve_fraction {
        Some(value) if value.is_finite() && (0.0..1.0).contains(&value) => {
            if !approximately_equal(value, manifest.reserve_fraction) {
                findings.push(MissionModelContractFinding::InvalidValue {
                    field: "energy_system.reserve_fraction".into(),
                    reason: "does not match the mission manifest's reserve policy".into(),
                });
            }
        }
        Some(_) => findings.push(MissionModelContractFinding::InvalidValue {
            field: "energy_system.reserve_fraction".into(),
            reason: "must be finite and lie in [0, 1)".into(),
        }),
        None => findings.push(MissionModelContractFinding::MissingField {
            field: "energy_system.reserve_fraction".into(),
        }),
    }

    if manifest.phases.is_empty() {
        findings.push(MissionModelContractFinding::MissingField {
            field: "phases".into(),
        });
    }
    let mut phase_ids = HashSet::with_capacity(manifest.phases.len());
    let mut total_energy_wh = 0.0;
    for phase in &manifest.phases {
        validate_nonempty("phases.id", &phase.id, &mut findings);
        if !phase_ids.insert(phase.id.clone()) {
            findings.push(MissionModelContractFinding::DuplicatePhaseId {
                id: phase.id.clone(),
            });
        }
        validate_positive("phases.duration_s", phase.duration_s, &mut findings);
        validate_nonnegative(
            "phases.required_bus_power_w",
            phase.required_bus_power_w,
            &mut findings,
        );
        validate_nonempty(
            "phases.power_uncertainty.method",
            &phase.power_uncertainty.method,
            &mut findings,
        );
        validate_nonnegative(
            "phases.power_uncertainty.lower_bound_w",
            phase.power_uncertainty.lower_bound_w,
            &mut findings,
        );
        validate_nonnegative(
            "phases.power_uncertainty.upper_bound_w",
            phase.power_uncertainty.upper_bound_w,
            &mut findings,
        );
        if phase.power_uncertainty.lower_bound_w.is_finite()
            && phase.required_bus_power_w.is_finite()
            && phase.power_uncertainty.lower_bound_w > phase.required_bus_power_w
        {
            findings.push(MissionModelContractFinding::InvalidValue {
                field: "phases.power_uncertainty.lower_bound_w".into(),
                reason: "lower bound must not exceed nominal bus power".into(),
            });
        }
        if phase.power_uncertainty.upper_bound_w.is_finite()
            && phase.required_bus_power_w.is_finite()
            && phase.power_uncertainty.upper_bound_w < phase.required_bus_power_w
        {
            findings.push(MissionModelContractFinding::InvalidValue {
                field: "phases.power_uncertainty.upper_bound_w".into(),
                reason: "upper bound must not be below nominal bus power".into(),
            });
        }
        if phase.power_uncertainty.lower_bound_w.is_finite()
            && phase.power_uncertainty.upper_bound_w.is_finite()
            && phase.power_uncertainty.lower_bound_w > phase.power_uncertainty.upper_bound_w
        {
            findings.push(MissionModelContractFinding::InvalidValue {
                field: "phases.power_uncertainty".into(),
                reason: "lower bound must not exceed upper bound".into(),
            });
        }
        if phase.duration_s.is_finite()
            && phase.duration_s > 0.0
            && phase.required_bus_power_w.is_finite()
            && phase.required_bus_power_w >= 0.0
        {
            let phase_energy_wh =
                phase.required_bus_power_w * phase.duration_s / 3_600.0;
            total_energy_wh += phase_energy_wh;
            if !phase_energy_wh.is_finite() || !total_energy_wh.is_finite() {
                findings.push(MissionModelContractFinding::AccumulationOverflow {
                    field: "phases.total_energy_wh".into(),
                });
            }
        }
    }

    if !findings.is_empty() {
        return Err(findings);
    }

    Ok(ValidatedCandidateMissionProfile {
        profile: MissionProfile {
            id: format!("{}::{}", manifest.mission_family_id, manifest.candidate_id),
            segments: manifest
                .phases
                .iter()
                .map(|phase| {
                    MissionSegment::new(
                        phase.id.clone(),
                        phase.duration_s,
                        phase.required_bus_power_w,
                        phase.duty_class,
                    )
                })
                .collect(),
        },
        manifest: manifest.clone(),
        nominal_total_energy_wh: total_energy_wh,
    })
}

fn validate_digest_binding(
    expected_field: &str,
    expected: &str,
    artifact_field: &str,
    artifact_digest: &str,
    findings: &mut Vec<MissionModelContractFinding>,
) {
    if !expected.is_empty()
        && !artifact_digest.is_empty()
        && !expected.eq_ignore_ascii_case(artifact_digest)
    {
        findings.push(MissionModelContractFinding::DigestMismatch {
            field: format!("{expected_field} != {artifact_field}"),
        });
    }
}

fn validate_artifact(
    field: &str,
    artifact: &ModelArtifactRef,
    findings: &mut Vec<MissionModelContractFinding>,
) {
    for (suffix, value) in [
        ("locator", artifact.locator.as_str()),
        ("revision", artifact.revision.as_str()),
        ("retrieved_at", artifact.retrieved_at.as_str()),
        ("units", artifact.units.as_str()),
        ("reference_conditions", artifact.reference_conditions.as_str()),
        ("applicability_domain", artifact.applicability_domain.as_str()),
        ("uncertainty_description", artifact.uncertainty_description.as_str()),
        ("limitations", artifact.limitations.as_str()),
    ] {
        validate_nonempty(&format!("{field}.{suffix}"), value, findings);
    }
    validate_digest(&format!("{field}.sha256"), &artifact.sha256, findings);
}

fn validate_mass_properties(
    mass: &CandidateMassProperties,
    findings: &mut Vec<MissionModelContractFinding>,
) {
    validate_positive("mass.fixed_vehicle_mass_kg", mass.fixed_vehicle_mass_kg, findings);
    validate_positive(
        "mass.energy_system_installed_mass_kg",
        mass.energy_system_installed_mass_kg,
        findings,
    );
    validate_positive(
        "mass.propulsion_installed_mass_kg",
        mass.propulsion_installed_mass_kg,
        findings,
    );
    validate_nonnegative("mass.consumables_mass_kg", mass.consumables_mass_kg, findings);
    validate_nonnegative("mass.payload_mass_kg", mass.payload_mass_kg, findings);
    validate_positive(
        "mass.maximum_takeoff_mass_limit_kg",
        mass.maximum_takeoff_mass_limit_kg,
        findings,
    );
    validate_nonempty(
        "mass.center_of_gravity_frame",
        &mass.center_of_gravity_frame,
        findings,
    );
    for (index, coordinate) in mass.center_of_gravity_m.iter().enumerate() {
        if !coordinate.is_finite() {
            findings.push(MissionModelContractFinding::InvalidValue {
                field: format!("mass.center_of_gravity_m[{index}]"),
                reason: "must be finite".into(),
            });
        }
    }
    if let Some(total_mass) = mass.mission_takeoff_mass_kg() {
        if total_mass > mass.maximum_takeoff_mass_limit_kg {
            findings.push(MissionModelContractFinding::MassLimitExceeded {
                total_mass_kg: total_mass,
                limit_kg: mass.maximum_takeoff_mass_limit_kg,
            });
        }
    } else {
        findings.push(MissionModelContractFinding::AccumulationOverflow {
            field: "mass.mission_takeoff_mass_kg".into(),
        });
    }
}

fn validate_environment(
    environment: &MissionEnvironment,
    findings: &mut Vec<MissionModelContractFinding>,
) {
    validate_positive("environment.air_density_kg_m3", environment.air_density_kg_m3, findings);
    validate_positive(
        "environment.ambient_temperature_k",
        environment.ambient_temperature_k,
        findings,
    );
    validate_positive(
        "environment.ambient_pressure_pa",
        environment.ambient_pressure_pa,
        findings,
    );
    validate_nonempty(
        "environment.wind_reference_frame",
        &environment.wind_reference_frame,
        findings,
    );
    for (index, component) in environment.wind_vector_mps.iter().enumerate() {
        if !component.is_finite() {
            findings.push(MissionModelContractFinding::InvalidValue {
                field: format!("environment.wind_vector_mps[{index}]"),
                reason: "must be finite".into(),
            });
        }
    }
}

fn validate_nonempty(
    field: &str,
    value: &str,
    findings: &mut Vec<MissionModelContractFinding>,
) {
    if value.trim().is_empty() || value.trim() != value {
        findings.push(MissionModelContractFinding::MissingField {
            field: field.into(),
        });
    }
}

fn validate_digest(
    field: &str,
    value: &str,
    findings: &mut Vec<MissionModelContractFinding>,
) {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        findings.push(MissionModelContractFinding::InvalidSha256 {
            field: field.into(),
        });
    }
}

fn validate_positive(
    field: &str,
    value: f64,
    findings: &mut Vec<MissionModelContractFinding>,
) {
    if !value.is_finite() || value <= 0.0 {
        findings.push(MissionModelContractFinding::InvalidValue {
            field: field.into(),
            reason: "must be finite and greater than zero".into(),
        });
    }
}

fn validate_nonnegative(
    field: &str,
    value: f64,
    findings: &mut Vec<MissionModelContractFinding>,
) {
    if !value.is_finite() || value < 0.0 {
        findings.push(MissionModelContractFinding::InvalidValue {
            field: field.into(),
            reason: "must be finite and non-negative".into(),
        });
    }
}

fn validate_fraction(
    field: &str,
    value: f64,
    findings: &mut Vec<MissionModelContractFinding>,
) {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        findings.push(MissionModelContractFinding::InvalidValue {
            field: field.into(),
            reason: "must be finite and lie in [0, 1)".into(),
        });
    }
}

fn approximately_equal(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 1.0e-9 * left.abs().max(right.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(ch: char, name: &str) -> ModelArtifactRef {
        ModelArtifactRef {
            locator: format!("https://example.invalid/{name}"),
            revision: "fixture-v1".into(),
            sha256: ch.to_string().repeat(64),
            retrieved_at: "2026-10-17".into(),
            units: "Not applicable: synthetic test fixture artifact.".into(),
            reference_conditions: "Not applicable: synthetic test fixture.".into(),
            applicability_domain: "Unit tests only; no physical aircraft applicability.".into(),
            uncertainty_description: "No physical uncertainty represented in this synthetic fixture.".into(),
            limitations: "Synthetic fixture only; not flight evidence.".into(),
        }
    }

    fn manifest() -> CandidateMissionManifest {
        let config_digest = "1".repeat(64);
        let energy_digest = "2".repeat(64);
        CandidateMissionManifest {
            candidate_id: "candidate-alpha".into(),
            mission_family_id: "two-seat-100km-v1".into(),
            configuration_sha256: config_digest.clone(),
            energy_system_envelope_sha256: energy_digest.clone(),
            route_sha256: "8".repeat(64),
            mission_family_sha256: "9".repeat(64),
            mass_properties_sha256: "f".repeat(64),
            environment_sha256: "7".repeat(64),
            phase_demand_sha256: "6".repeat(64),
            route_distance_m: 100_000.0,
            reserve_fraction: 0.1,
            mass_properties: CandidateMassProperties {
                fixed_vehicle_mass_kg: 600.0,
                energy_system_installed_mass_kg: 50.0,
                propulsion_installed_mass_kg: 40.0,
                consumables_mass_kg: 0.0,
                payload_mass_kg: 160.0,
                maximum_takeoff_mass_limit_kg: 900.0,
                center_of_gravity_frame: "body-datum-v1".into(),
                center_of_gravity_m: [1.2, 0.0, 0.3],
            },
            environment: MissionEnvironment {
                air_density_kg_m3: 1.225,
                ambient_temperature_k: 288.15,
                ambient_pressure_pa: 101_325.0,
                wind_reference_frame: "local-north-east-down-v1".into(),
                wind_vector_mps: [0.0, 0.0, 0.0],
            },
            model_run: CandidateMissionModelRun {
                run_id: "run-001".into(),
                model_version: "fixture-model-v1".into(),
                source_revision: "git-sha-fixture".into(),
                run_artifact: artifact('a', "run"),
                configuration_artifact: artifact('1', "configuration"),
                model_artifact: artifact('b', "model"),
                aerodynamics_artifact: artifact('c', "aero"),
                propulsion_artifact: artifact('d', "propulsion"),
                energy_system_artifact: artifact('2', "energy"),
                mass_properties_artifact: artifact('f', "mass"),
                environment_artifact: artifact('7', "environment"),
                phase_demand_artifact: artifact('6', "phase-demand"),
                route_artifact: artifact('8', "route"),
                mission_family_artifact: artifact('9', "mission-family"),
                input_configuration_sha256: config_digest,
                input_energy_system_envelope_sha256: energy_digest,
                input_route_sha256: "8".repeat(64),
                input_mission_family_sha256: "9".repeat(64),
                input_mass_properties_sha256: "f".repeat(64),
                input_environment_sha256: "7".repeat(64),
                output_phase_demand_sha256: "6".repeat(64),
            },
            phases: vec![
                MissionPhaseDemand {
                    id: "cruise".into(),
                    duration_s: 100.0,
                    required_bus_power_w: 1_000.0,
                    power_uncertainty: PowerUncertaintyBounds {
                        lower_bound_w: 900.0,
                        upper_bound_w: 1_100.0,
                        method: "synthetic invariant fixture; not flight evidence".into(),
                    },
                    duty_class: DutyClass::Continuous,
                },
                MissionPhaseDemand {
                    id: "climb".into(),
                    duration_s: 10.0,
                    required_bus_power_w: 1_500.0,
                    power_uncertainty: PowerUncertaintyBounds {
                        lower_bound_w: 1_300.0,
                        upper_bound_w: 1_700.0,
                        method: "synthetic invariant fixture; not flight evidence".into(),
                    },
                    duty_class: DutyClass::Peak,
                },
            ],
        }
    }

    fn energy_system() -> EnergySystemEnvelope {
        EnergySystemEnvelope {
            installed_mass_kg: Some(50.0),
            usable_energy_at_bus_wh: Some(50_000.0),
            reserve_fraction: Some(0.1),
            max_continuous_bus_power_w: Some(30_000.0),
            max_peak_bus_power_w: Some(50_000.0),
            max_peak_duration_s: Some(60.0),
        }
    }

    #[test]
    fn valid_manifest_builds_a_candidate_specific_profile() {
        let validated = validate_candidate_mission_manifest(&manifest(), &energy_system()).unwrap();
        assert_eq!(validated.profile.id, "two-seat-100km-v1::candidate-alpha");
        assert_eq!(validated.profile.segments.len(), 2);
        assert_eq!(validated.profile.segments[0].id, "cruise");
        assert!((validated.nominal_total_energy_wh - 31.944444444444443).abs() < 1e-9);
        assert_eq!(validated.manifest.phases[0].power_uncertainty.lower_bound_w, 900.0);
    }

    #[test]
    fn mismatched_configuration_digest_fails_closed() {
        let mut input = manifest();
        input.model_run.input_configuration_sha256 = "3".repeat(64);
        let result = validate_candidate_mission_manifest(&input, &energy_system());
        assert!(matches!(
            result,
            Err(findings) if findings.iter().any(|finding| matches!(
                finding,
                MissionModelContractFinding::DigestMismatch { field }
                    if field == "model_run.input_configuration_sha256"
            ))
        ));
    }

    #[test]
    fn mismatched_energy_system_mass_and_reserve_fail_closed() {
        let input = manifest();
        let mut system = energy_system();
        system.installed_mass_kg = Some(55.0);
        system.reserve_fraction = Some(0.2);
        let result = validate_candidate_mission_manifest(&input, &system);
        assert!(matches!(
            result,
            Err(findings) if findings.len() >= 2
        ));
    }

    #[test]
    fn mass_over_limit_is_rejected() {
        let mut input = manifest();
        input.mass_properties.payload_mass_kg = 300.0;
        let result = validate_candidate_mission_manifest(&input, &energy_system());
        assert!(matches!(
            result,
            Err(findings) if findings.iter().any(|finding| matches!(
                finding,
                MissionModelContractFinding::MassLimitExceeded { .. }
            ))
        ));
    }

    #[test]
    fn configuration_artifact_digest_must_match_manifest() {
        let mut input = manifest();
        input.model_run.configuration_artifact.sha256 = "4".repeat(64);
        let result = validate_candidate_mission_manifest(&input, &energy_system());
        assert!(matches!(
            result,
            Err(findings) if findings.iter().any(|finding| matches!(
                finding,
                MissionModelContractFinding::DigestMismatch { field }
                    if field.contains("configuration_sha256")
            ))
        ));
    }

    #[test]
    fn route_and_mission_family_artifacts_are_bound_to_declared_digests() {
        let mut input = manifest();
        input.route_sha256 = "3".repeat(64);
        input.mission_family_sha256 = "5".repeat(64);
        let result = validate_candidate_mission_manifest(&input, &energy_system());
        assert!(matches!(
            result,
            Err(findings) if findings.iter().filter(|finding| matches!(
                finding,
                MissionModelContractFinding::DigestMismatch { .. }
            )).count() >= 2
        ));
    }

    #[test]
    fn phase_demand_digest_must_match_output_artifact_and_record() {
        let mut input = manifest();
        input.model_run.output_phase_demand_sha256 = "0".repeat(64);
        let result = validate_candidate_mission_manifest(&input, &energy_system());
        assert!(matches!(
            result,
            Err(findings) if findings.iter().any(|finding| matches!(
                finding,
                MissionModelContractFinding::DigestMismatch { field }
                    if field.contains("phase_demand_sha256")
            ))
        ));
    }

    #[test]
    fn uncertainty_bounds_must_contain_nominal_power() {
        let mut input = manifest();
        input.phases[0].power_uncertainty.lower_bound_w = 1_100.0;
        let result = validate_candidate_mission_manifest(&input, &energy_system());
        assert!(matches!(
            result,
            Err(findings) if findings.iter().any(|finding| matches!(
                finding,
                MissionModelContractFinding::InvalidValue { field, .. }
                    if field == "phases.power_uncertainty.lower_bound_w"
            ))
        ));
    }

    #[test]
    fn invalid_source_digest_is_rejected() {
        let mut input = manifest();
        input.model_run.aerodynamics_artifact.sha256 = "unknown".into();
        let result = validate_candidate_mission_manifest(&input, &energy_system());
        assert!(matches!(
            result,
            Err(findings) if findings.iter().any(|finding| matches!(
                finding,
                MissionModelContractFinding::InvalidSha256 { field }
                    if field == "model_run.aerodynamics_artifact.sha256"
            ))
        ));
    }

    #[test]
    fn duplicate_phase_ids_are_rejected() {
        let mut input = manifest();
        input.phases[1].id = input.phases[0].id.clone();
        let result = validate_candidate_mission_manifest(&input, &energy_system());
        assert!(matches!(
            result,
            Err(findings) if findings.iter().any(|finding| matches!(
                finding,
                MissionModelContractFinding::DuplicatePhaseId { .. }
            ))
        ));
    }

    #[test]
    fn non_finite_phase_and_environment_values_are_rejected() {
        let mut input = manifest();
        input.phases[0].required_bus_power_w = f64::INFINITY;
        input.environment.air_density_kg_m3 = f64::NAN;
        let result = validate_candidate_mission_manifest(&input, &energy_system());
        assert!(matches!(
            result,
            Err(findings) if findings.iter().any(|finding| matches!(
                finding,
                MissionModelContractFinding::InvalidValue { .. }
            ))
        ));
    }
}
