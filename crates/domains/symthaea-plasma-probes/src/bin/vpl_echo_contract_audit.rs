// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Cross-field semantic auditor for the synthetic K-RESP-CL Vlasov-Poisson-Landau contract.
//!
//! JSON Schema validates shape. This binary checks the physical scaling law,
//! forward-time preparation, disjoint cohort roles, collision-parameter range,
//! immutable state identities, required observables / structure gates, and
//! the literature-only claim boundary. It does not execute a Landau solver,
//! prove the cited theorem, or produce a physical echo-growth certificate.

use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("kresp_cl_contract_audit=FAIL: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let fixture_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| "usage: vpl_echo_contract_audit <fixture.json>".to_owned())?;
    let bytes = fs::read(&fixture_path)
        .map_err(|error| format!("cannot read {}: {error}", fixture_path.display()))?;
    let fixture: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid JSON {}: {error}", fixture_path.display()))?;
    audit(&fixture)?;
    println!("kresp_cl_vpl_semantic_validation=PASS");
    Ok(())
}

fn audit(fixture: &Value) -> Result<(), String> {
    require_string(fixture, "schema_version", "0.1")?;
    require_string(fixture, "record_status", "synthetic_benchmark_contract_only")?;
    require_string(fixture, "benchmark_id", "KRESP-CL-VPL-1D1V")?;
    require_string(fixture, "equation", "vlasov_poisson_landau_coulomb")?;
    require_string(fixture, "collision_operator", "landau_coulomb")?;
    require_string(fixture, "equilibrium", "global_maxwellian")?;

    let domain = field(fixture, "domain")?;
    require_string(domain, "spatial_topology", "periodic_torus")?;
    require_string(domain, "velocity_domain_model", "truncated_real_line_with_decay_tail")?;
    require_string(domain, "time_direction", "forward_only_from_physical_initial_data")?;
    if integer(domain, "velocity_dimension")? != 1
        || number(domain, "spatial_domain_length")? <= 0.0
        || number(domain, "velocity_cutoff")? <= 0.0
        || !boolean(domain, "velocity_tail_extension_required")?
    {
        return Err(
            "the VPL domain or velocity-tail convergence contract is invalid".to_owned(),
        );
    }

    let scaling = field(fixture, "perturbation_scaling")?;
    require_string(
        scaling,
        "formula",
        "epsilon=c*nu^(1/3)*(log(1/nu))^p",
    )?;
    require_string(scaling, "log_argument", "1/nu")?;
    if !boolean(scaling, "amplitude_is_computed_from_formula")?
        || !boolean(scaling, "p_values_are_exploratory_not_source_theorem_constants")?
    {
        return Err(
            "the perturbation scaling must be computed and exploratory p values labelled".to_owned(),
        );
    }
    let nu_range = array_at(scaling, "collision_parameter_nu_range")?;
    if nu_range.len() != 2 {
        return Err("collision_parameter_nu_range must contain exactly two bounds".to_owned());
    }
    let nu_min = numeric_value(&nu_range[0], "nu lower bound")?;
    let nu_max = numeric_value(&nu_range[1], "nu upper bound")?;
    let amplitude_limit = number(scaling, "maximum_initial_perturbation_amplitude")?;
    let horizon_fraction = number(
        scaling,
        "observation_horizon_fraction_of_nu_minus_one_third",
    )?;
    if nu_min <= 0.0
        || nu_max >= 1.0
        || nu_max <= nu_min
        || amplitude_limit <= 0.0
        || !horizon_fraction.is_finite()
        || horizon_fraction <= 0.0
        || horizon_fraction > 1.0
    {
        return Err(
            "the scaling domain, amplitude limit or pre-dissipation horizon is invalid".to_owned(),
        );
    }

    let c_values = numeric_array(scaling, "prefactor_c_values")?;
    let p_values = numeric_array(scaling, "log_power_p_values")?;
    if c_values.len() < 2 || p_values.is_empty()
        || c_values.iter().any(|value| *value <= 0.0)
        || p_values.iter().any(|value| *value < 0.0)
    {
        return Err("the exploratory c/p sweep must be non-empty and admissible".to_owned());
    }

    let cohorts = field(fixture, "trajectory_cohorts")?;
    let role_specs = [
        ("training", "training", 3_usize),
        ("validation", "validation", 2_usize),
        ("sealed_holdout", "sealed_holdout", 3_usize),
    ];
    let mut trajectory_ids = HashSet::new();
    let mut initial_state_digests = HashSet::new();
    let mut observed_roles = HashSet::new();
    for (group_name, expected_role, expected_count) in role_specs {
        let trajectories = array_at(cohorts, group_name)?;
        if trajectories.len() != expected_count {
            return Err(format!(
                "{group_name} must contain exactly {expected_count} trajectories"
            ));
        }
        for trajectory in trajectories {
            let trajectory_id = string(trajectory, "trajectory_id")?;
            if trajectory_id.trim().is_empty()
                || !trajectory_ids.insert(trajectory_id.to_owned())
            {
                return Err(format!("trajectory id {trajectory_id} is empty or duplicated"));
            }
            let role = string(trajectory, "split_role")?;
            if role != expected_role {
                return Err(format!(
                    "{trajectory_id} has role {role}, expected {expected_role}"
                ));
            }
            observed_roles.insert(role.to_owned());
            if !boolean(trajectory, "forward_initialization")? {
                return Err(format!("{trajectory_id} is not initialized at physical time zero"));
            }

            let nu = number(trajectory, "collision_parameter_nu")?;
            let c = number(trajectory, "prefactor_c")?;
            let p = number(trajectory, "log_power_p")?;
            let epsilon = number(trajectory, "computed_initial_perturbation_amplitude")?;
            if nu < nu_min || nu > nu_max || c <= 0.0 || p < 0.0 {
                return Err(format!(
                    "{trajectory_id} lies outside the registered nu/c/p domain"
                ));
            }
            if !c_values.iter().any(|value| *value == c)
                || !p_values.iter().any(|value| *value == p)
            {
                return Err(format!("{trajectory_id} uses an unregistered c or p value"));
            }
            let expected_epsilon = c * nu.cbrt() * (1.0 / nu).ln().powf(p);
            if !expected_epsilon.is_finite() || expected_epsilon <= 0.0 {
                return Err(format!("{trajectory_id} has non-finite formula amplitude"));
            }
            let formula_tolerance = 32.0 * f64::EPSILON * expected_epsilon.abs().max(1.0);
            if (epsilon - expected_epsilon).abs() > formula_tolerance {
                return Err(format!(
                    "{trajectory_id} amplitude does not match epsilon=c*nu^(1/3)*(log(1/nu))^p"
                ));
            }
            if epsilon > amplitude_limit {
                return Err(format!(
                    "{trajectory_id} exceeds the small-perturbation design limit"
                ));
            }

            let modes = integer_array(trajectory, "initial_spatial_modes")?;
            if modes.len() < 2 || modes.iter().any(|mode| *mode <= 0) {
                return Err(format!(
                    "{trajectory_id} requires at least two positive spatial modes"
                ));
            }
            let distinct_modes = modes.iter().copied().collect::<HashSet<_>>();
            if distinct_modes.len() != modes.len() {
                return Err(format!("{trajectory_id} contains duplicate initial spatial modes"));
            }
            if integer(trajectory, "echo_cascade_depth")? < 2 {
                return Err(format!(
                    "{trajectory_id} must register at least a two-wave interaction"
                ));
            }
            if number(
                trajectory,
                "observation_horizon_fraction_of_nu_minus_one_third",
            )? != horizon_fraction
            {
                return Err(format!(
                    "{trajectory_id} does not use the registered observation horizon"
                ));
            }

            let digest = string(trajectory, "initial_state_sha256")?;
            if !is_sha256_hex(digest) || !initial_state_digests.insert(digest.to_owned()) {
                return Err(format!(
                    "{trajectory_id} has an invalid or reused initial-state digest"
                ));
            }
        }
    }
    if observed_roles.len() != 3 {
        return Err(
            "training, validation and sealed-holdout roles must all be present".to_owned(),
        );
    }

    let qois = string_array(fixture, "registered_qois")?;
    for qoi in [
        "electric_field_mode_amplitude",
        "transport_profile_sobolev_norm",
        "high_spatial_frequency_tail_energy",
        "collision_entropy_production",
        "branch_identity",
    ] {
        if !qois.iter().any(|value| value == qoi) {
            return Err(format!("required VPL QoI {qoi} is missing"));
        }
    }

    let structure = field(fixture, "structure_contract")?;
    for key in [
        "mass_conservation_required",
        "momentum_conservation_required",
        "kinetic_energy_exchange_required",
        "landau_entropy_dissipation_required",
        "field_poisson_residual_required",
        "positivity_required",
        "full_order_fallback_on_unqualified_branch",
    ] {
        if !boolean(structure, key)? {
            return Err(format!("required VPL structure gate {key} cannot be disabled"));
        }
    }

    let convergence = field(fixture, "reference_convergence")?;
    for key in [
        "spatial_refinement_levels",
        "velocity_refinement_levels",
        "time_refinement_levels",
    ] {
        if integer(convergence, key)? < 2 {
            return Err(format!("{key} must be at least two"));
        }
    }
    for key in [
        "velocity_domain_extension_required",
        "collision_operator_quadrature_refinement_required",
        "independent_full_order_confirmation_required",
        "all_reference_trajectories_forward_integrated",
    ] {
        if !boolean(convergence, key)? {
            return Err(format!("reference convergence requirement {key} cannot be disabled"));
        }
    }

    let comparators = string_array(fixture, "comparators")?;
    for comparator in [
        "full_order_vlasov_poisson_landau",
        "linearized_landau_reference",
        "collisionless_vlasov_poisson_control",
        "snapshot_pod",
        "adaptive_hermite",
        "response_balanced_kinetic_reduction",
        "symplectic_particle_vlasov_rom",
        "entropy_vpd_rom_not_same_equation",
    ] {
        if !comparators.iter().any(|value| value == comparator) {
            return Err(format!("mandatory comparator {comparator} is missing"));
        }
    }

    let success = field(fixture, "success_targets")?;
    require_string(success, "status", "prospective_only_not_achieved")?;
    for key in [
        "all_targets_are_prospective",
        "echo_growth_before_enhanced_dissipation_scale_required",
        "reduced_model_must_not_suppress_verified_echo_cascade",
        "all_structure_gates_must_hold",
    ] {
        if !boolean(success, key)? {
            return Err(format!("success target {key} cannot be disabled"));
        }
    }

    let boundary = field(fixture, "claim_boundary")?;
    for key in [
        "physical_solver_executed",
        "physical_echo_growth_measured",
        "reference_converged",
        "stability_theorem_proved_for_this_implementation",
        "competitive_result_reproduced",
    ] {
        if boolean(boundary, key)? {
            return Err(format!("synthetic fixture cannot claim {key}"));
        }
    }

    let prior_art = field(fixture, "prior_art")?;
    require_string(prior_art, "arxiv_id", "2611.20080")?;
    require_string(prior_art, "source_date", "2026-11-15")?;
    require_string(prior_art, "status", "literature_claim_not_reproduced_here")?;
    let source_claims = string_array(prior_art, "source_claims")?;
    for claim in [
        "echo_growth_stable_under_linearized_operator_changes",
        "echo_growth_stable_with_exact_evolving_background_coefficients",
        "echo_growth_persists_under_nonlinear_perturbations",
        "sobolev_norm_inflation_before_nu_minus_one_third_enhanced_dissipation_scale",
    ] {
        if !source_claims.iter().any(|value| value == claim) {
            return Err(format!("prior-art claim {claim} is not registered"));
        }
    }
    let provenance = field(fixture, "provenance")?;
    require_string(provenance, "fixture_class", "synthetic_kresp_cl_design_fixture_only")?;
    if !boolean(provenance, "no_solver_execution")? {
        return Err("synthetic K-RESP-CL provenance cannot claim solver execution".to_owned());
    }

    Ok(())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value.get(key).ok_or_else(|| format!("missing field {key}"))
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| format!("{key} must be a string"))
}

fn require_string(value: &Value, key: &str, expected: &str) -> Result<(), String> {
    let actual = string(value, key)?;
    if actual != expected {
        return Err(format!("{key} must equal {expected}, got {actual}"));
    }
    Ok(())
}

fn number(value: &Value, key: &str) -> Result<f64, String> {
    numeric_value(field(value, key)?, key)
}

fn numeric_value(value: &Value, name: &str) -> Result<f64, String> {
    let number = value
        .as_f64()
        .ok_or_else(|| format!("{name} must be a finite number"))?;
    if !number.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    Ok(number)
}

fn integer(value: &Value, key: &str) -> Result<i64, String> {
    field(value, key)?
        .as_i64()
        .ok_or_else(|| format!("{key} must be an integer"))
}

fn boolean(value: &Value, key: &str) -> Result<bool, String> {
    field(value, key)?
        .as_bool()
        .ok_or_else(|| format!("{key} must be a boolean"))
}

fn array_at<'a>(value: &'a Value, key: &str) -> Result<&'a [Value], String> {
    field(value, key)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{key} must be an array"))
}

fn numeric_array(value: &Value, key: &str) -> Result<Vec<f64>, String> {
    array_at(value, key)?
        .iter()
        .map(|entry| numeric_value(entry, key))
        .collect()
}

fn integer_array(value: &Value, key: &str) -> Result<Vec<i64>, String> {
    array_at(value, key)?
        .iter()
        .map(|entry| {
            entry
                .as_i64()
                .ok_or_else(|| format!("{key} entries must be integers"))
        })
        .collect()
}

fn string_array(value: &Value, key: &str) -> Result<Vec<String>, String> {
    array_at(value, key)?
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{key} entries must be strings"))
        })
        .collect()
}
