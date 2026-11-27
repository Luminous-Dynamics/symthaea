// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Cross-field semantic auditor for the synthetic K-RESP-1 experiment contract.
//!
//! JSON Schema validates field shape. This executable validates the physical
//! relationships between fields: shared checkpoint history, admissible and
//! ordered pulse paths, disjoint splits, held-out second-pulse parameters,
//! matched QoI/convergence contracts, mandatory comparators, and claim limits.
//! It does not execute a plasma solver or certify any physical response.

use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq)]
struct FirstPulseSignature {
    spatial_mode: i64,
    kick_amplitude: f64,
    phase_rad: f64,
    application_time: f64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("response_balanced_contract_audit=FAIL: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let fixture_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| "usage: response_balanced_contract_audit <fixture.json>".to_owned())?;
    let bytes = fs::read(&fixture_path)
        .map_err(|error| format!("cannot read {}: {error}", fixture_path.display()))?;
    let fixture: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid JSON {}: {error}", fixture_path.display()))?;
    audit(&fixture)?;
    println!("response_balanced_kresp1_semantic_validation=PASS");
    Ok(())
}

fn audit(fixture: &Value) -> Result<(), String> {
    require_string(fixture, "schema_version", "0.1")?;
    require_string(
        fixture,
        "record_status",
        "synthetic_experiment_contract_only",
    )?;
    require_string(
        fixture,
        "source_pulse_family_contract_id",
        "AKPD-SYNTHETIC-PULSE-FAMILY-001",
    )?;

    let cohort = field(fixture, "cohort")?;
    let spatial = field(cohort, "spatial_domain")?;
    let velocity = field(cohort, "velocity_domain")?;
    let maximum_time = number(cohort, "maximum_time")?;
    if string(cohort, "state_metric_id")?.trim().is_empty() {
        return Err("cohort state metric identifier is empty".to_owned());
    }
    if string(cohort, "state_coordinate_id")?.trim().is_empty() {
        return Err("cohort state coordinate identifier is empty".to_owned());
    }
    if maximum_time <= 0.0
        || number(spatial, "maximum")? <= number(spatial, "minimum")?
        || number(velocity, "maximum")? <= number(velocity, "minimum")?
    {
        return Err("cohort domains or maximum time are invalid".to_owned());
    }

    let checkpoint = field(fixture, "frozen_checkpoint")?;
    let checkpoint_time = number(checkpoint, "time")?;
    if checkpoint_time < 0.0
        || checkpoint_time >= maximum_time
        || !boolean(checkpoint, "phase_mixing_complete")?
        || !boolean(checkpoint, "first_pulse_applied")?
        || !boolean(checkpoint, "second_pulse_not_yet_applied")?
    {
        return Err("checkpoint phase or pulse history is inconsistent".to_owned());
    }
    let solver_revision = string(checkpoint, "solver_revision")?;
    if solver_revision.trim().is_empty() {
        return Err("checkpoint solver revision is empty".to_owned());
    }

    let linearization = field(fixture, "linearization_contract")?;
    require_string(
        linearization,
        "status",
        "synthetic_derivatives_not_computed",
    )?;
    require_string(linearization, "derivative_source", "not_computed_synthetic")?;
    for key in [
        "future_solver_transition_jacobians_required",
        "pulse_map_state_jacobian_required",
        "pulse_map_parameter_jacobian_required",
        "split_solver_composition_order_required",
        "independent_finite_difference_cross_check_required",
    ] {
        if !boolean(linearization, key)? {
            return Err(format!(
                "linearization requirement {key} cannot be disabled"
            ));
        }
    }
    require_string(
        linearization,
        "branch_change_disposition",
        "UNKNOWN_OR_HANDOFF",
    )?;
    if !boolean(linearization, "pulse_application_time_jacobian_required")?
        || !boolean(
            linearization,
            "pulse_time_finite_difference_cross_check_required",
        )?
    {
        return Err(
            "pulse application-time sensitivity and independent cross-check are mandatory"
                .to_owned(),
        );
    }
    require_string(
        linearization,
        "pulse_time_sensitivity_convention",
        "scheduled_reset_D_R_f_minus_plus_R_t_minus_f_plus",
    )?;
    require_string(
        linearization,
        "pulse_time_sensitivity_status",
        "synthetic_not_computed",
    )?;
    if !boolean(linearization, "distribution_pulse_pullback_required")?
        || !boolean(
            linearization,
            "distribution_pulse_remap_mass_ledger_required",
        )?
    {
        return Err("distribution pulse pullback and mass ledger are mandatory".to_owned());
    }
    require_string(
        linearization,
        "distribution_pulse_pullback_status",
        "first_order_piecewise_constant_module_only",
    )?;
    require_string(
        linearization,
        "distribution_pulse_remap_jacobian_status",
        "fixed_topology_piecewise_constant_analytic_module_only",
    )?;
    if !boolean(
        linearization,
        "distribution_pulse_remap_jacobian_finite_difference_required",
    )? {
        return Err("distribution remap Jacobian finite-difference cross-check is mandatory".to_owned());
    }
    require_string(
        linearization,
        "distribution_pulse_remap_topology_change_disposition",
        "UNKNOWN_OR_HANDOFF",
    )?;
    require_string(
        linearization,
        "distribution_pulse_boundary_outflow_disposition",
        "UNQUALIFIED_UNLESS_GRID_CONVERGENCE_OR_EXPLICIT_BOUNDARY_ERROR",
    )?;
    if !boolean(
        linearization,
        "distribution_pulse_boundary_outflow_jacobian_required",
    )? || !boolean(
        linearization,
        "distribution_pulse_boundary_outflow_finite_difference_required",
    )? {
        return Err("boundary outflow Jacobian and finite-difference check are mandatory".to_owned());
    }
    require_string(
        linearization,
        "distribution_pulse_boundary_outflow_jacobian_status",
        "fixed_topology_piecewise_constant_analytic_module_only",
    )?;

    if !boolean(
        linearization,
        "distribution_pulse_spatial_integral_weights_required",
    )? {
        return Err("physical spatial mass integration requires explicit quadrature weights".to_owned());
    }
    require_string(
        linearization,
        "distribution_pulse_unweighted_total_mass_interpretation",
        "unit_weight_row_sum_not_general_spatial_integral",
    )?;
    require_string(
        linearization,
        "distribution_pulse_weighted_quadrature_api_status",
        "positive_spatial_weight_api_implemented_module_only",
    )?;

    let horizon = field(fixture, "prediction_horizon")?;
    let horizon_start = number(horizon, "start_time")?;
    let horizon_end = number(horizon, "end_time")?;
    if horizon_start != checkpoint_time
        || horizon_end <= horizon_start
        || horizon_end > maximum_time
    {
        return Err("prediction horizon is not aligned to the frozen checkpoint".to_owned());
    }

    let family = field(fixture, "pulse_family")?;
    require_string(family, "operator_id", "sinusoidal_velocity_kick_v1")?;
    require_string(family, "amplitude_semantics", "maximum_absolute_delta_v")?;
    require_string(
        family,
        "application_semantics",
        "instantaneous_canonical_velocity_kick",
    )?;
    let mode_range = field(family, "allowed_spatial_mode")?;
    let mode_min = integer(mode_range, "minimum")?;
    let mode_max = integer(mode_range, "maximum")?;
    let phase_range = field(family, "allowed_phase_rad")?;
    let phase_min = number(phase_range, "minimum")?;
    let phase_max = number(phase_range, "maximum")?;
    let amplitude_range = field(family, "allowed_kick_amplitude")?;
    let amplitude_min = number(amplitude_range, "minimum")?;
    let amplitude_max = number(amplitude_range, "maximum")?;
    let maximum_application_time = number(family, "maximum_application_time")?;
    if mode_min <= 0
        || mode_max < mode_min
        || phase_max <= phase_min
        || amplitude_min < 0.0
        || amplitude_max < amplitude_min
        || maximum_application_time > maximum_time
    {
        return Err("pulse-family admissible ranges are invalid".to_owned());
    }

    let split = field(fixture, "split")?;
    let training = array_at(split, "training")?;
    let validation = array_at(split, "validation")?;
    let held_out = array_at(split, "held_out")?;
    if training.len() < 2 || validation.is_empty() || held_out.len() < 3 {
        return Err("training, validation, or held-out split is too small".to_owned());
    }

    let held_out_axes = string_array(split, "held_out_axes")?;
    for axis in [
        "pulse_phase",
        "kick_amplitude",
        "spatial_mode",
        "pulse_separation",
    ] {
        if !held_out_axes.iter().any(|value| value == axis) {
            return Err(format!("required held-out axis {axis} is not declared"));
        }
    }

    let mut trial_ids = HashSet::new();
    let mut pulse_ids = HashSet::new();
    let mut first_signature: Option<FirstPulseSignature> = None;
    let mut training_second = Vec::new();
    let mut held_out_second = Vec::new();

    for (split_name, trials, is_training, is_held_out) in [
        ("training", training, true, false),
        ("validation", validation, false, false),
        ("held_out", held_out, false, true),
    ] {
        for trial in trials {
            let trial_id = string(trial, "trial_id")?;
            if !trial_ids.insert(trial_id.to_owned()) {
                return Err(format!("trial id {trial_id} is duplicated across splits"));
            }
            let pulses = array_at(trial, "pulses")?;
            if pulses.len() != 2 {
                return Err(format!(
                    "{split_name}/{trial_id} must contain exactly two pulses"
                ));
            }
            let first = &pulses[0];
            let second = &pulses[1];
            for pulse in pulses {
                let pulse_id = string(pulse, "pulse_id")?;
                if !pulse_ids.insert(pulse_id.to_owned()) {
                    return Err(format!("pulse id {pulse_id} is duplicated globally"));
                }
                let mode = integer(pulse, "spatial_mode")?;
                let amplitude = number(pulse, "kick_amplitude")?;
                let phase = number(pulse, "phase_rad")?;
                let time = number(pulse, "application_time")?;
                if mode < mode_min
                    || mode > mode_max
                    || amplitude < amplitude_min
                    || amplitude > amplitude_max
                    || phase < phase_min
                    || phase > phase_max
                    || time < 0.0
                    || time > maximum_application_time
                    || time >= horizon_end
                {
                    return Err(format!(
                        "{split_name}/{trial_id} has a pulse outside the registered family"
                    ));
                }
            }

            let first_time = number(first, "application_time")?;
            let second_time = number(second, "application_time")?;
            if first_time >= checkpoint_time || second_time <= checkpoint_time {
                return Err(format!(
                    "{split_name}/{trial_id} checkpoint must lie strictly between pulse one and pulse two"
                ));
            }

            let signature = FirstPulseSignature {
                spatial_mode: integer(first, "spatial_mode")?,
                kick_amplitude: number(first, "kick_amplitude")?,
                phase_rad: number(first, "phase_rad")?,
                application_time: first_time,
            };
            match first_signature {
                None => first_signature = Some(signature),
                Some(existing) if existing == signature => {}
                Some(_) => {
                    return Err(
                        "training, validation, and held-out paths do not share one first pulse"
                            .to_owned(),
                    );
                }
            }

            if is_training {
                training_second.push(second);
            }
            if is_held_out {
                held_out_second.push(second);
            }
        }
    }

    if training_second.is_empty() || held_out_second.is_empty() {
        return Err("training or held-out second-pulse cohort is empty".to_owned());
    }
    if !held_out_second.iter().any(|held| {
        !training_second
            .iter()
            .any(|train| same_number(held, train, "phase_rad"))
    }) {
        return Err("held-out second pulses do not vary phase beyond training".to_owned());
    }
    if !held_out_second.iter().any(|held| {
        !training_second
            .iter()
            .any(|train| same_number(held, train, "kick_amplitude"))
    }) {
        return Err("held-out second pulses do not vary amplitude beyond training".to_owned());
    }
    if !held_out_second.iter().any(|held| {
        !training_second
            .iter()
            .any(|train| same_integer(held, train, "spatial_mode"))
    }) {
        return Err("held-out second pulses do not vary spatial mode beyond training".to_owned());
    }

    let training_separations = training
        .iter()
        .map(|trial| pulse_separation(trial))
        .collect::<Result<Vec<_>, _>>()?;
    let held_out_separations = held_out
        .iter()
        .map(|trial| pulse_separation(trial))
        .collect::<Result<Vec<_>, _>>()?;
    if !held_out_separations
        .iter()
        .any(|held| !training_separations.iter().any(|train| held == train))
    {
        return Err("held-out second pulses do not vary separation beyond training".to_owned());
    }

    let observables = array_at(fixture, "observables")?;
    if observables.len() < 3 {
        return Err("at least three observables are required".to_owned());
    }
    let mut qoi_ids = HashSet::new();
    for observable in observables {
        let id = string(observable, "qoi_id")?;
        if !qoi_ids.insert(id.to_owned()) {
            return Err(format!("observable id {id} is duplicated"));
        }
        if number(observable, "normalization_scale")? <= 0.0
            || number(observable, "priority")? < 0.0
        {
            return Err(format!("observable {id} has invalid scale or priority"));
        }
    }

    let convergence = field(fixture, "reference_convergence")?;
    for key in [
        "minimum_spatial_refinement_levels",
        "minimum_velocity_refinement_levels",
        "minimum_time_refinement_levels",
    ] {
        if integer(convergence, key)? < 2 {
            return Err(format!("{key} must be at least two"));
        }
    }
    if !boolean(convergence, "domain_extension_required")?
        || !boolean(convergence, "independent_reference_confirmation_required")?
    {
        return Err("reference convergence requirements cannot be disabled".to_owned());
    }
    for qoi in string_array(convergence, "convergence_observables")? {
        if !qoi_ids.contains(&qoi) {
            return Err(format!(
                "convergence observable {qoi} is not a registered QoI"
            ));
        }
    }

    let prior_art = string_array(fixture, "methodological_prior_art")?;
    for method in [
        "semi_infinite_worst_case_scenario_generation",
        "adaptive_symplectic_particle_vlasov_rom",
    ] {
        if !prior_art.iter().any(|value| value == method) {
            return Err(format!(
                "methodological prior-art entry {method} is missing"
            ));
        }
    }
    let comparators = string_array(fixture, "comparators")?;
    let adjacent_methods = string_array(fixture, "adjacent_methods")?;
    for method in [
        "adaptive_opinf_port",
        "adaptive_nitrom_port",
        "adaptive_opinf_nitrom_hybrid_port",
        "symmetry_augmented_kinetic_closure",
        "entropy_based_vlasov_poisson_dougherty_moment_port",
        "adaptive_symplectic_particle_vlasov_rom_port",
    ] {
        if !adjacent_methods.iter().any(|value| value == method) {
            return Err(format!("adjacent method {method} is not registered"));
        }
    }
    for comparator in [
        "snapshot_pod",
        "padovan_nonintrusive_balancing",
        "time_windowed_pod",
        "energy_windowed_pod",
        "conservative_fluid_kinetic_rom",
        "adaptive_hermite",
        "adaptive_fourier",
        "tensor_network_vlasov_poisson",
        "non_markovian_fno_closure",
        "full_order_reference",
    ] {
        if !comparators.iter().any(|value| value == comparator) {
            return Err(format!("mandatory comparator {comparator} is missing"));
        }
    }

    let response_qualification = field(fixture, "response_qualification")?;
    if !boolean(response_qualification, "checkpoint_state_residual_required")?
        || !boolean(
            response_qualification,
            "future_pulse_injection_residual_required",
        )?
        || !boolean(
            response_qualification,
            "nonlinear_full_order_response_required",
        )?
    {
        return Err(
            "both linearized response channels and nonlinear full-order response are mandatory"
                .to_owned(),
        );
    }
    if number(response_qualification, "cvar_alpha")? != 0.9
        || string(response_qualification, "residual_threshold_status")?
            != "calibrate_on_independently_converged_physical_reference"
    {
        return Err("response-risk summary or calibration status is invalid".to_owned());
    }
    let risk_metrics = string_array(response_qualification, "risk_summary_metrics")?;
    for metric in ["weighted_mean", "worst_case", "cvar"] {
        if !risk_metrics.iter().any(|value| value == metric) {
            return Err(format!("response risk metric {metric} is missing"));
        }
    }

    let scalar_search = field(response_qualification, "scalar_separation_oracle")?;
    require_string(scalar_search, "status", "design_only_not_executed")?;
    require_string(
        scalar_search,
        "method",
        "branch_conditioned_lipschitz_branch_and_bound_1d",
    )?;
    if !boolean(scalar_search, "interior_search_required")?
        || boolean(scalar_search, "boundary_only_search_sufficient")?
        || !boolean(scalar_search, "full_order_confirmation_required")?
    {
        return Err(
            "adversarial pulse search must include interior points and full-order confirmation"
                .to_owned(),
        );
    }
    require_string(
        scalar_search,
        "lipschitz_certificate_status",
        "not_issued_synthetic",
    )?;
    require_string(
        scalar_search,
        "branch_stability_evidence_status",
        "not_issued_synthetic",
    )?;
    require_string(
        scalar_search,
        "budget_exhaustion_disposition",
        "unqualified",
    )?;
    require_string(
        scalar_search,
        "global_bound_claim_status",
        "conditional_on_valid_lipschitz_and_branch_evidence",
    )?;
    let parameter_axes = string_array(scalar_search, "continuous_parameter_axes")?;
    for axis in ["pulse_phase", "kick_amplitude", "pulse_separation"] {
        if !parameter_axes.iter().any(|value| value == axis) {
            return Err(format!(
                "continuous adversarial search axis {axis} is missing"
            ));
        }
    }

    let joint_search = field(response_qualification, "joint_box_oracle")?;
    require_string(joint_search, "status", "design_only_not_executed")?;
    require_string(
        joint_search,
        "method",
        "anisotropic_lipschitz_branch_and_bound_nd",
    )?;
    require_string(
        joint_search,
        "coordinate_lipschitz_semantics",
        "sum_i_L_i_abs_coordinate_delta_i",
    )?;
    require_string(
        joint_search,
        "subdivision_policy",
        "largest_Li_times_half_width",
    )?;
    require_string(
        joint_search,
        "lipschitz_certificate_status",
        "not_issued_synthetic",
    )?;
    require_string(
        joint_search,
        "branch_stability_evidence_status",
        "not_issued_synthetic",
    )?;
    require_string(
        joint_search,
        "budget_exhaustion_disposition",
        "unqualified",
    )?;
    require_string(
        joint_search,
        "global_bound_claim_status",
        "conditional_on_valid_box_lipschitz_and_branch_evidence",
    )?;
    require_string(
        joint_search,
        "objective_normalization",
        "declared_qoi_scale_and_fixed_positive_floor",
    )?;
    if !boolean(joint_search, "interior_search_required")?
        || boolean(joint_search, "boundary_only_search_sufficient")?
        || !boolean(joint_search, "full_order_confirmation_required")?
        || !boolean(
            joint_search,
            "reference_response_denominator_lower_bound_required",
        )?
    {
        return Err(
            "joint response-box search must retain interior, full-order and denominator gates"
                .to_owned(),
        );
    }
    let joint_axes = string_array(joint_search, "continuous_parameter_axes")?;
    if joint_axes.len() < 2 || joint_axes.len() > 3 {
        return Err("joint response-box search requires two or three axes".to_owned());
    }
    for axis in ["pulse_phase", "kick_amplitude", "pulse_separation"] {
        if !joint_axes.iter().any(|value| value == axis) {
            return Err(format!("joint response-box axis {axis} is missing"));
        }
    }

    let resources = field(fixture, "resource_budget")?;
    for comparison in [
        "equal_rank",
        "equal_stored_bytes",
        "equal_end_to_end_wall_time",
    ] {
        if !string_array(resources, "comparison_modes")?
            .iter()
            .any(|value| value == comparison)
        {
            return Err(format!(
                "matched-cost comparison mode {comparison} is missing"
            ));
        }
    }
    if integer(resources, "maximum_reference_steps")? <= 0
        || integer(resources, "maximum_wall_time_seconds")? <= 0
        || integer(resources, "maximum_memory_bytes")? <= 0
    {
        return Err("resource limits must be positive".to_owned());
    }

    let targets = field(fixture, "success_targets")?;
    require_string(targets, "status", "prospective_only_not_achieved")?;
    if !boolean(targets, "all_targets_are_prospective")? {
        return Err("prospective physical targets cannot be marked achieved".to_owned());
    }
    let claim_boundary = field(fixture, "claim_boundary")?;
    for key in [
        "physical_solver_executed",
        "physical_response_measured",
        "competitive_ranking_claimed",
        "independent_reference_converged",
    ] {
        if boolean(claim_boundary, key)? {
            return Err(format!("synthetic fixture falsely claims {key}"));
        }
    }
    Ok(())
}

fn pulse_separation(trial: &Value) -> Result<f64, String> {
    let pulses = array_at(trial, "pulses")?;
    if pulses.len() != 2 {
        return Err("pulse separation requires exactly two pulses".to_owned());
    }
    Ok(number(&pulses[1], "application_time")? - number(&pulses[0], "application_time")?)
}

fn same_number(left: &Value, right: &Value, key: &str) -> bool {
    match (
        left.get(key).and_then(Value::as_f64),
        right.get(key).and_then(Value::as_f64),
    ) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

fn same_integer(left: &Value, right: &Value, key: &str) -> bool {
    match (
        left.get(key).and_then(Value::as_i64),
        right.get(key).and_then(Value::as_i64),
    ) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
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
    let result = field(value, key)?
        .as_f64()
        .ok_or_else(|| format!("{key} must be a finite number"))?;
    if !result.is_finite() {
        return Err(format!("{key} must be finite"));
    }
    Ok(result)
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
