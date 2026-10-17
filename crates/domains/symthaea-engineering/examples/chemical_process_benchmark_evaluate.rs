// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Independent checker for candidate dispositions against the frozen synthetic
// chemical-process qualification corpus. This checks benchmark contracts only;
// it is not a chemistry solver, process simulator, safety assessor, or authority
// to qualify or operate a physical process.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    process::ExitCode,
};

const ALLOWED_DISPOSITIONS: &[&str] = &[
    "ACCEPTED_WITHIN_SCOPE",
    "BLOCKED",
    "UNRESOLVED",
    "OUT_OF_DOMAIN",
];

const ALLOWED_CLASSES: &[&str] = &[
    "positive-control",
    "negative-control",
    "metamorphic",
    "analytic-reference",
    "uncertainty-control",
    "tamper-control",
];

const USAGE: &str = concat!(
    "Usage: chemical_process_benchmark_evaluate --corpus <path> ",
    "--results <path> --expected-head <git-commit-sha>"
);

struct Args {
    corpus_path: String,
    results_path: String,
    expected_head: String,
}

fn parse_args() -> Result<Args, String> {
    let mut corpus_path = None;
    let mut results_path = None;
    let mut expected_head = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}\n{USAGE}"))?;
        match flag.as_str() {
            "--corpus" => corpus_path = Some(value),
            "--results" => results_path = Some(value),
            "--expected-head" => expected_head = Some(value),
            _ => return Err(format!("unknown argument {flag:?}\n{USAGE}")),
        }
    }

    let expected_head = expected_head
        .ok_or_else(|| format!("--expected-head is required\n{USAGE}"))?;
    let is_hex_sha = matches!(expected_head.len(), 40 | 64)
        && expected_head.chars().all(|character| character.is_ascii_hexdigit());
    if !is_hex_sha {
        return Err(format!(
            "expected head must be a 40- or 64-character hexadecimal commit ID\n{USAGE}"
        ));
    }

    Ok(Args {
        corpus_path: corpus_path.ok_or_else(|| format!("--corpus is required\n{USAGE}"))?,
        results_path: results_path.ok_or_else(|| format!("--results is required\n{USAGE}"))?,
        expected_head,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_json_with_digest(path: &str) -> Result<(Value, String), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let digest = sha256_hex(&bytes);
    let parsed = serde_json::from_slice(&bytes)?;
    Ok((parsed, digest))
}

fn text<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    value.get(field).and_then(Value::as_str)
}

fn strings(value: &Value, field: &str) -> Option<BTreeSet<String>> {
    value
        .get(field)?
        .as_array()?
        .iter()
        .map(|item| {
            item.as_str()
                .filter(|text| !text.trim().is_empty())
                .map(ToOwned::to_owned)
        })
        .collect()
}


fn numeric_field(value: &Value, field: &str) -> Option<f64> {
    value.get(field).and_then(Value::as_f64)
}

fn mixer_balance_oracle(fixture: &Value, tolerance: f64) -> Result<Value, String> {
    if fixture.get("unit").and_then(Value::as_str) != Some("mixer")
        || fixture.get("reaction").and_then(Value::as_bool) != Some(false)
    {
        return Err("mixer oracle requires a non-reactive mixer fixture".to_owned());
    }

    let mut components = BTreeMap::<String, (f64, f64, f64, String)>::new();
    for (stream_group, is_inlet) in [("inlets", true), ("outlets", false)] {
        let streams = fixture
            .get(stream_group)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("mixer fixture has no {stream_group} array"))?;

        for stream in streams {
            let component = text(stream, "component")
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| format!("{stream_group} stream has no component identity"))?;
            let flow = numeric_field(stream, "flow")
                .ok_or_else(|| {
                    format!("{stream_group} stream for {component} has no numeric flow")
                })?;
            let unit = text(stream, "unit")
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| format!("{stream_group} stream for {component} has no unit"))?;

            if !flow.is_finite() || flow < 0.0 {
                return Err(format!("{stream_group} stream for {component} has invalid flow"));
            }

            let entry = components
                .entry(component.to_owned())
                .or_insert((0.0, 0.0, 0.0, unit.to_owned()));
            if entry.3 != unit {
                return Err(format!(
                    concat!(
                        "component {component} mixes incompatible unit labels ",
                        "without explicit conversion"
                    )
                ));
            }

            if is_inlet {
                entry.0 += flow;
            } else {
                entry.1 += flow;
            }
            entry.2 += flow.abs();
        }
    }

    if components.is_empty() {
        return Err("mixer fixture must contain at least one component".to_owned());
    }

    let mut all_closed = true;
    let mut component_report = serde_json::Map::new();
    for (component, (inlet, outlet, magnitude_sum, unit)) in components {
        let residual = inlet - outlet;
        let scale = 1.0_f64.max(magnitude_sum);
        let normalized_residual = residual.abs() / scale;
        let passed = normalized_residual <= tolerance;
        all_closed &= passed;
        component_report.insert(
            component,
            json!({
                "unit": unit,
                "inlet_total": inlet,
                "outlet_total": outlet,
                "raw_residual": residual,
                "normalization_scale": scale,
                "normalized_residual": normalized_residual,
                "tolerance": tolerance,
                "passed": passed
            }),
        );
    }

    Ok(json!({
        "oracle": "nonreactive_mixer_component_balance_v1",
        "derived_disposition": if all_closed { "ACCEPTED_WITHIN_SCOPE" } else { "BLOCKED" },
        "components": component_report,
        "balance_closed": all_closed
    }))
}

fn unit_conversion_oracle(
    fixture: &Value,
    observed_numeric_values: &Value,
    tolerance: f64,
) -> Result<Value, String> {
    let from_value = numeric_field(fixture, "from_value")
        .ok_or_else(|| "unit-conversion fixture is missing from_value".to_owned())?;
    let to_value = numeric_field(fixture, "to_value")
        .ok_or_else(|| "unit-conversion fixture is missing to_value".to_owned())?;
    let factor = numeric_field(fixture, "conversion_factor")
        .ok_or_else(|| "unit-conversion fixture is missing conversion_factor".to_owned())?;
    let from_unit = text(fixture, "from_unit").unwrap_or_default();
    let to_unit = text(fixture, "to_unit").unwrap_or_default();
    let factor_ref = text(fixture, "conversion_factor_ref").unwrap_or_default();
    let rule = text(fixture, "conversion_rule").unwrap_or_default();

    if !from_value.is_finite()
        || !to_value.is_finite()
        || !factor.is_finite()
        || factor <= 0.0
        || from_unit.is_empty()
        || to_unit.is_empty()
        || from_unit == to_unit
        || factor_ref.is_empty()
        || rule != "to_value = from_value * conversion_factor"
    {
        return Err(
            "unit-conversion fixture has invalid values, units, reference, or rule".to_owned(),
        );
    }

    let independently_converted = from_value * factor;
    let reference_scale = 1.0_f64.max(independently_converted.abs()).max(to_value.abs());
    let fixture_reference_residual =
        (independently_converted - to_value).abs() / reference_scale;
    let observed_value = numeric_field(observed_numeric_values, "converted_value")
        .ok_or_else(|| "unit-conversion result is missing converted_value".to_owned())?;
    if !observed_value.is_finite() {
        return Err("unit-conversion result converted_value is non-finite".to_owned());
    }
    let observed_relative_error = (observed_value - independently_converted).abs()
        / 1.0_f64.max(independently_converted.abs());
    let fixture_reference_valid = fixture_reference_residual <= tolerance;
    let observed_conversion_valid = observed_relative_error <= tolerance;
    Ok(json!({
        "oracle": "explicit_unit_conversion_v1",
        "from_value": from_value,
        "from_unit": from_unit,
        "conversion_factor": factor,
        "conversion_factor_ref": factor_ref,
        "independently_converted_value": independently_converted,
        "fixture_reference_value": to_value,
        "fixture_reference_normalized_residual": fixture_reference_residual,
        "observed_converted_value": observed_value,
        "observed_relative_error": observed_relative_error,
        "tolerance": tolerance,
        "fixture_reference_valid": fixture_reference_valid,
        "observed_conversion_valid": observed_conversion_valid,
        "passed": fixture_reference_valid && observed_conversion_valid
    }))
}


fn unit_dimension(unit: &str) -> Option<&'static str> {
    match unit {
        "kg/s" | "g/s" => Some("mass_flow"),
        "mol/s" | "kmol/s" => Some("amount_flow"),
        "m^3/s" | "L/s" => Some("volume_flow"),
        _ => None,
    }
}

fn required_number(fixture: &Value, field: &str) -> Result<f64, String> {
    let value = numeric_field(fixture, field)
        .ok_or_else(|| format!("fixture is missing numeric field {field:?}"))?;
    if !value.is_finite() {
        return Err(format!("fixture numeric field {field:?} is not finite"));
    }
    Ok(value)
}

fn derive_synthetic_disposition(
    id: &str,
    fixture: &Value,
    tolerance: f64,
) -> Result<Value, String> {
    match id {
        "CHEM-PROC-003" => {
            let inlets = fixture
                .get("inlets")
                .and_then(Value::as_array)
                .ok_or_else(|| "unit-compatibility fixture is missing inlets".to_owned())?;
            let mut dimensions = BTreeMap::<String, BTreeSet<String>>::new();
            let mut units_by_component = BTreeMap::<String, BTreeSet<String>>::new();
            for inlet in inlets {
                let component = text(inlet, "component")
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| "unit-compatibility inlet is missing component".to_owned())?;
                let unit = text(inlet, "unit")
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| "unit-compatibility inlet is missing unit".to_owned())?;
                let dimension = unit_dimension(unit)
                    .ok_or_else(|| format!("unit {unit:?} is outside this synthetic oracle"))?;
                dimensions.entry(component.to_owned())
                    .or_default()
                    .insert(dimension.to_owned());
                units_by_component.entry(component.to_owned())
                    .or_default()
                    .insert(unit.to_owned());
            }
            let mismatches = dimensions
                .iter()
                .filter(|(_, values)| values.len() > 1)
                .map(|(component, _)| component.clone())
                .collect::<Vec<_>>();
            let conversion_evidence_present = fixture
                .get("conversion_evidence")
                .is_some_and(|value| !value.is_null());
            let blocked = !mismatches.is_empty() && !conversion_evidence_present;
            Ok(json!({
                "oracle": "synthetic_quantity_dimension_compatibility_v1",
                "derived_disposition": if blocked { "BLOCKED" } else { "ACCEPTED_WITHIN_SCOPE" },
                "component_units": units_by_component,
                "incompatible_components": mismatches,
                "conversion_evidence_present": conversion_evidence_present
            }))
        }
        "CHEM-PROC-005" => {
            let inlet = required_number(fixture, "inlet_flow")?;
            let product = required_number(fixture, "known_product_flow")?;
            let unknown_output_missing = fixture
                .get("unknown_output_flow")
                .is_none_or(Value::is_null);
            let sink_missing = fixture.get("declared_sink").is_none_or(Value::is_null);
            let unexplained_residual = inlet - product > tolerance;
            let unresolved = unknown_output_missing && sink_missing && unexplained_residual;
            Ok(json!({
                "oracle": "unaccounted_stream_disposition_v1",
                "derived_disposition": if unresolved {
                    "UNRESOLVED"
                } else {
                    "ACCEPTED_WITHIN_SCOPE"
                },
                "inlet_flow": inlet,
                "known_product_flow": product,
                "unaccounted_known_difference": inlet - product,
                "unknown_output_missing": unknown_output_missing,
                "declared_sink_missing": sink_missing
            }))
        }
        "CHEM-PROC-006" => {
            let inlet = required_number(fixture, "inlet_flow")?;
            let product = required_number(fixture, "product_flow")?;
            let sink = required_number(fixture, "declared_sink_flow")?;
            let scale = 1.0_f64.max(inlet.abs() + product.abs() + sink.abs());
            let residual = inlet - product - sink;
            let normalized_residual = residual.abs() / scale;
            let sink_declared = text(fixture, "sink_evidence_kind")
                == Some("synthetic_accounting_fixture");
            let closed = normalized_residual <= tolerance && sink_declared;
            Ok(json!({
                "oracle": "explicit_synthetic_accounting_sink_v1",
                "derived_disposition": if closed { "ACCEPTED_WITHIN_SCOPE" } else { "BLOCKED" },
                "raw_residual": residual,
                "normalization_scale": scale,
                "normalized_residual": normalized_residual,
                "tolerance": tolerance,
                "declared_synthetic_sink": sink_declared,
                "accounting_closed": closed
            }))
        }
        "CHEM-PROC-008" => {
            let status = text(fixture, "solver_status").unwrap_or_default();
            let residual = numeric_field(fixture, "convergence_residual");
            let solution_present = numeric_field(fixture, "reported_solution")
                .is_some_and(f64::is_finite);
            let converged = status == "converged"
                && solution_present
                && residual.is_some_and(|value| value.is_finite() && value.abs() <= tolerance);
            Ok(json!({
                "oracle": "solver_convergence_state_v1",
                "derived_disposition": if converged { "ACCEPTED_WITHIN_SCOPE" } else { "BLOCKED" },
                "solver_status": status,
                "convergence_residual": residual,
                "convergence_claim_supported": converged
            }))
        }
        "CHEM-PROC-009" => {
            let reference_a = text(fixture, "stream_a_reference").unwrap_or_default();
            let reference_b = text(fixture, "stream_b_reference").unwrap_or_default();
            let evidence_present = fixture.get("reconciliation_evidence")
                .is_some_and(|value| !value.is_null());
            let unresolved = reference_a != reference_b && !evidence_present;
            Ok(json!({
                "oracle": "thermodynamic_reference_state_compatibility_v1",
                "derived_disposition": if unresolved {
                    "UNRESOLVED"
                } else {
                    "ACCEPTED_WITHIN_SCOPE"
                },
                "reference_states_match": reference_a == reference_b,
                "reconciliation_evidence_present": evidence_present
            }))
        }
        "CHEM-PROC-010" => {
            let selectivity = numeric_field(fixture, "reported_selectivity")
                .is_some_and(f64::is_finite);
            let recovery = numeric_field(fixture, "reported_recovery").is_some_and(f64::is_finite);
            let assay = fixture.get("product_assay").is_some_and(|value| !value.is_null());
            let definition = text(fixture, "metric_definition").unwrap_or_default();
            let complete = selectivity && recovery && assay && !definition.trim().is_empty();
            Ok(json!({
                "oracle": "separation_metric_completeness_v1",
                "derived_disposition": if complete {
                    "ACCEPTED_WITHIN_SCOPE"
                } else {
                    "UNRESOLVED"
                },
                "selectivity_present": selectivity,
                "recovery_present": recovery,
                "product_assay_present": assay,
                "metric_definition_present": !definition.trim().is_empty()
            }))
        }
        "CHEM-PROC-011" => {
            let purity = text(fixture, "purity_status") == Some("known");
            let contaminants = text(fixture, "contaminant_profile") == Some("resolved");
            let admissible = purity && contaminants;
            Ok(json!({
                "oracle": "material_grade_evidence_v1",
                "derived_disposition": if admissible { "ACCEPTED_WITHIN_SCOPE" } else { "BLOCKED" },
                "purity_known": purity,
                "contaminants_resolved": contaminants
            }))
        }
        "CHEM-PROC-012" => {
            let currentness = text(fixture, "currentness").unwrap_or_default();
            let policy = text(fixture, "currentness_policy").unwrap_or_default();
            let violation = currentness == "stale" && policy == "current_required";
            Ok(json!({
                "oracle": "property_currentness_policy_v1",
                "derived_disposition": if violation { "BLOCKED" } else { "ACCEPTED_WITHIN_SCOPE" },
                "currentness": currentness,
                "policy": policy,
                "stale_requirement_violation": violation
            }))
        }
        "CHEM-PROC-013" => {
            let domain = text(fixture, "model_validity_domain").unwrap_or_default();
            let context = text(fixture, "candidate_context").unwrap_or_default();
            let qualified =
                text(fixture, "extrapolation_policy") == Some("qualified_for_domain_shift");
            let mismatch = domain != context && !qualified;
            Ok(json!({
                "oracle": "model_validity_domain_v1",
                "derived_disposition": if mismatch {
                    "OUT_OF_DOMAIN"
                } else {
                    "ACCEPTED_WITHIN_SCOPE"
                },
                "model_domain": domain,
                "candidate_context": context,
                "qualified_extrapolation": qualified
            }))
        }
        "CHEM-PROC-014" => {
            let required = strings(fixture, "required_utilities")
                .ok_or_else(|| "required_utilities missing or malformed".to_owned())?;
            let available = strings(fixture, "available_utilities")
                .ok_or_else(|| "available_utilities missing or malformed".to_owned())?;
            let missing = required.difference(&available).cloned().collect::<Vec<_>>();
            Ok(json!({
                "oracle": "required_utility_closure_v1",
                "derived_disposition": if missing.is_empty() {
                    "ACCEPTED_WITHIN_SCOPE"
                } else {
                    "BLOCKED"
                },
                "missing_utilities": missing
            }))
        }
        "CHEM-PROC-015" => {
            let violated = text(fixture, "hard_constraint_status") == Some("violated");
            let hard_gates =
                text(fixture, "optimization_policy") == Some("hard_constraints_are_gates");
            let blocked = violated && hard_gates;
            Ok(json!({
                "oracle": "noncompensatory_hard_constraint_v1",
                "derived_disposition": if blocked { "BLOCKED" } else { "ACCEPTED_WITHIN_SCOPE" },
                "hard_constraint_violated": violated,
                "hard_constraints_are_gates": hard_gates
            }))
        }
        "CHEM-PROC-016" => {
            let overlap = fixture.get("uncertainty_overlap").and_then(Value::as_bool) == Some(true);
            let rank_changes = fixture
                .get("sensitivity_changes_order")
                .and_then(Value::as_bool)
                == Some(true);
            let score_a = required_number(fixture, "candidate_A_point_score")?;
            let score_b = required_number(fixture, "candidate_B_point_score")?;
            let unresolved = overlap && rank_changes;
            Ok(json!({
                "oracle": "uncertainty_sensitive_ranking_v1",
                "derived_disposition": if unresolved {
                    "UNRESOLVED"
                } else {
                    "ACCEPTED_WITHIN_SCOPE"
                },
                "point_scores": { "A": score_a, "B": score_b },
                "uncertainty_overlap": overlap,
                "sensitivity_changes_order": rank_changes
            }))
        }
        "CHEM-PROC-017" => {
            let definitions = fixture.get("definitions").and_then(Value::as_array)
                .ok_or_else(|| "stream identity fixture is missing definitions".to_owned())?;
            let digests = definitions.iter().map(|definition| {
                text(definition, "composition_digest")
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| "stream definition is missing composition digest".to_owned())
            }).collect::<Result<BTreeSet<_>, _>>()?;
            let conflict = digests.len() > 1;
            Ok(json!({
                "oracle": "stream_identity_conflict_v1",
                "derived_disposition": if conflict { "BLOCKED" } else { "ACCEPTED_WITHIN_SCOPE" },
                "distinct_composition_digests": digests
            }))
        }
        "CHEM-PROC-018" => {
            let required = strings(fixture, "required_dependencies")
                .ok_or_else(|| "required_dependencies missing or malformed".to_owned())?;
            let available = strings(fixture, "available_dependencies")
                .ok_or_else(|| "available_dependencies missing or malformed".to_owned())?;
            let missing = required.difference(&available).cloned().collect::<Vec<_>>();
            Ok(json!({
                "oracle": "lifecycle_dependency_closure_v1",
                "derived_disposition": if missing.is_empty() {
                    "ACCEPTED_WITHIN_SCOPE"
                } else {
                    "BLOCKED"
                },
                "missing_dependencies": missing
            }))
        }
        "CHEM-PROC-019" => {
            let origin = text(fixture, "result_origin").unwrap_or_default();
            let requested = text(fixture, "requested_evidence_class").unwrap_or_default();
            let measurement = fixture
                .get("measurement_record")
                .is_some_and(|value| !value.is_null());
            let mismatch = origin == "simulation"
                && requested == "physical_observation"
                && !measurement;
            Ok(json!({
                "oracle": "evidence_origin_class_v1",
                "derived_disposition": if mismatch { "BLOCKED" } else { "ACCEPTED_WITHIN_SCOPE" },
                "result_origin": origin,
                "requested_evidence_class": requested,
                "measurement_record_present": measurement
            }))
        }
        "CHEM-PROC-020" => {
            let purity_present = fixture
                .get("purity_evidence")
                .is_some_and(|value| !value.is_null());
            Ok(json!({
                "oracle": "inventory_is_not_purity_evidence_v1",
                "derived_disposition": if purity_present {
                    "ACCEPTED_WITHIN_SCOPE"
                } else {
                    "BLOCKED"
                },
                "purity_evidence_present": purity_present
            }))
        }
        "CHEM-PROC-021" => {
            let registered = text(fixture, "registered_digest").unwrap_or_default();
            let presented = text(fixture, "presented_digest").unwrap_or_default();
            let match_digest = !registered.is_empty() && registered == presented;
            Ok(json!({
                "oracle": "source_content_digest_identity_v1",
                "derived_disposition": if match_digest {
                    "ACCEPTED_WITHIN_SCOPE"
                } else {
                    "BLOCKED"
                },
                "registered_digest_matches_presented": match_digest
            }))
        }
        "CHEM-PROC-022" => {
            let status = text(fixture, "external_solver_status").unwrap_or_default();
            let expected = fixture
                .get("expected_reference_comparison")
                .and_then(Value::as_bool)
                == Some(true);
            let comparison = status == "completed_pass" && expected;
            Ok(json!({
                "oracle": "reference_solver_execution_state_v1",
                "derived_disposition": if comparison { "ACCEPTED_WITHIN_SCOPE" } else { "BLOCKED" },
                "solver_status": status,
                "expected_reference_comparison": expected,
                "comparison_established": comparison
            }))
        }
        "CHEM-PROC-023" => {
            let assessed = text(fixture, "assessed_head").unwrap_or_default();
            let receipt = text(fixture, "receipt_head").unwrap_or_default();
            let match_head = !assessed.is_empty() && assessed == receipt;
            Ok(json!({
                "oracle": "receipt_exact_head_binding_v1",
                "derived_disposition": if match_head { "ACCEPTED_WITHIN_SCOPE" } else { "BLOCKED" },
                "exact_head_match": match_head
            }))
        }
        "CHEM-PROC-024" => {
            let graph_valid = fixture.get("graph_connected").and_then(Value::as_bool) == Some(true)
                && fixture.get("stream_schema_valid").and_then(Value::as_bool) == Some(true);
            let model_validated = text(fixture, "model_validation") == Some("validated");
            let equipment_qualified = text(fixture, "equipment_qualification") == Some("qualified");
            let disposition = if !graph_valid {
                "BLOCKED"
            } else if model_validated && equipment_qualified {
                "ACCEPTED_WITHIN_SCOPE"
            } else {
                "UNRESOLVED"
            };
            Ok(json!({
                "oracle": "structural_vs_physical_qualification_v1",
                "derived_disposition": disposition,
                "graph_structurally_valid": graph_valid,
                "model_validated": model_validated,
                "equipment_qualified": equipment_qualified
            }))
        }
        "CHEM-PROC-025" | "CHEM-PROC-026" => {
            let proposition_id = text(fixture, "proposition_id").unwrap_or_default();
            let manifests_complete = fixture
                .get("dependency_manifests_complete")
                .and_then(Value::as_bool)
                == Some(true);
            let solver_a = fixture.get("solver_a");
            let solver_b = fixture.get("solver_b");
            let solver_a_id = solver_a
                .and_then(|solver| text(solver, "solver_id"))
                .unwrap_or_default();
            let solver_b_id = solver_b
                .and_then(|solver| text(solver, "solver_id"))
                .unwrap_or_default();
            let solver_a_implementation = solver_a
                .and_then(|solver| text(solver, "implementation_digest"))
                .unwrap_or_default();
            let solver_b_implementation = solver_b
                .and_then(|solver| text(solver, "implementation_digest"))
                .unwrap_or_default();
            let dependencies_a = solver_a
                .and_then(|solver| strings(solver, "evidence_dependency_digests"))
                .unwrap_or_default();
            let dependencies_b = solver_b
                .and_then(|solver| strings(solver, "evidence_dependency_digests"))
                .unwrap_or_default();
            let shared_dependencies = dependencies_a
                .intersection(&dependencies_b)
                .cloned()
                .collect::<Vec<_>>();
            let inputs_complete = !proposition_id.trim().is_empty()
                && manifests_complete
                && !solver_a_id.trim().is_empty()
                && !solver_b_id.trim().is_empty()
                && !solver_a_implementation.trim().is_empty()
                && !solver_b_implementation.trim().is_empty()
                && !dependencies_a.is_empty()
                && !dependencies_b.is_empty();
            let implementations_distinct = solver_a_id != solver_b_id
                && solver_a_implementation != solver_b_implementation;
            let independent_within_declared_scope = inputs_complete
                && implementations_distinct
                && shared_dependencies.is_empty();
            let derived_disposition = if independent_within_declared_scope {
                "ACCEPTED_WITHIN_SCOPE"
            } else {
                "UNRESOLVED"
            };
            Ok(json!({
                "oracle": "evidence_dependency_independence_v1",
                "derived_disposition": derived_disposition,
                "proposition_id": proposition_id,
                "dependency_manifests_complete": manifests_complete,
                "solver_a_id": solver_a_id,
                "solver_b_id": solver_b_id,
                "solver_implementations_distinct": implementations_distinct,
                "solver_a_dependency_count": dependencies_a.len(),
                "solver_b_dependency_count": dependencies_b.len(),
                "shared_evidence_dependency_digests": shared_dependencies,
                "input_provenance_complete": inputs_complete,
                "independent_corroboration_within_declared_scope": independent_within_declared_scope
            }))
        }
        _ => Err(format!("no semantic disposition oracle registered for {id}")),
    }
}


fn evaluate(corpus: &Value, results: &Value, expected_head: &str) -> Value {
    let mut global_errors = Vec::<String>::new();

    let benchmark_id = text(corpus, "benchmark_id").unwrap_or_default();
    let corpus_version = text(corpus, "corpus_version").unwrap_or_default();

    if benchmark_id.is_empty() || corpus_version.is_empty() {
        global_errors.push("corpus is missing benchmark_id or corpus_version".to_owned());
    }

    let canonical_dispositions = ALLOWED_DISPOSITIONS
        .iter()
        .map(|item| (*item).to_owned())
        .collect::<BTreeSet<_>>();
    if strings(corpus, "allowed_dispositions").as_ref() != Some(&canonical_dispositions) {
        global_errors.push(
            "corpus allowed_dispositions is missing or differs from the frozen vocabulary"
                .to_owned(),
        );
    }
    if text(results, "benchmark_id") != Some(benchmark_id) {
        global_errors.push("results benchmark_id does not match corpus".to_owned());
    }
    if text(results, "corpus_version") != Some(corpus_version) {
        global_errors.push("results corpus_version does not match corpus".to_owned());
    }

    let reported_head = text(results, "evaluated_head").unwrap_or_default();
    if reported_head != expected_head {
        global_errors.push(
            "results evaluated_head does not match externally supplied --expected-head".to_owned(),
        );
    }

    let expected_cases = match corpus.get("cases").and_then(Value::as_array) {
        Some(cases) => cases,
        None => {
            global_errors.push("corpus cases field is missing or not an array".to_owned());
            return report(corpus, expected_head, reported_head, global_errors, Vec::new(), 0);
        }
    };
    if expected_cases.is_empty() {
        global_errors.push("corpus case list must not be empty".to_owned());
    }
    let mut expected_id_counts = BTreeMap::<String, usize>::new();
    for case in expected_cases {
        let Some(id) = text(case, "id") else {
            global_errors.push("corpus case is missing an id".to_owned());
            continue;
        };
        if id.trim().is_empty() {
            global_errors.push("corpus case has an empty id".to_owned());
        }
        let count = expected_id_counts.entry(id.to_owned()).or_default();
        *count += 1;
        if *count > 1 {
            global_errors.push(format!("duplicate corpus case id {id:?}"));
        }
        let class = text(case, "class").unwrap_or_default();
        if !ALLOWED_CLASSES.contains(&class) {
            global_errors.push(format!("corpus case {id:?} has unknown class {class:?}"));
        }
        if case.get("fixture").and_then(Value::as_object).is_none() {
            global_errors.push(format!("corpus case {id:?} has a missing or malformed fixture"));
        }
        if strings(case, "required_findings").map_or(true, |items| items.is_empty()) {
            global_errors.push(format!(
                "corpus case {id:?} has missing or empty required_findings"
            ));
        }
        if strings(case, "prohibited_promotions").map_or(true, |items| items.is_empty()) {
            global_errors.push(format!(
                "corpus case {id:?} has missing or empty prohibited_promotions"
            ));
        }
    }

    let observed_cases = match results.get("cases").and_then(Value::as_array) {
        Some(cases) => cases,
        None => {
            global_errors.push("results cases field is missing or not an array".to_owned());
            return report(
                corpus,
                expected_head,
                reported_head,
                global_errors,
                expected_cases
                    .iter()
                    .map(|case| {
                        let id = text(case, "id").unwrap_or("<missing-id>");
                        json!({
                            "id": id,
                            "passed": false,
                            "errors": ["no case results were provided"]
                        })
                    })
                    .collect(),
                0,
            );
        }
    };

    let mut observed_by_id = BTreeMap::<String, Vec<&Value>>::new();
    for observed in observed_cases {
        match text(observed, "id") {
            Some(id) if !id.trim().is_empty() => {
                observed_by_id.entry(id.to_owned()).or_default().push(observed);
            }
            _ => global_errors.push("result case has a missing or empty id".to_owned()),
        }
    }

    let expected_ids = expected_cases
        .iter()
        .filter_map(|case| text(case, "id").map(ToOwned::to_owned))
        .collect::<BTreeSet<_>>();
    for id in observed_by_id.keys() {
        if !expected_ids.contains(id) {
            global_errors.push(format!("unexpected result case id {id:?}"));
        }
        if observed_by_id.get(id).is_some_and(|matches| matches.len() != 1) {
            global_errors.push(format!("duplicate result case id {id:?}"));
        }
    }

    let mut case_reports = Vec::with_capacity(expected_cases.len());
    for expected in expected_cases {
        let id = text(expected, "id").unwrap_or("<missing-id>");
        let expected_disposition = text(expected, "expected_disposition").unwrap_or_default();
        let mut errors = Vec::<String>::new();

        if !ALLOWED_DISPOSITIONS.contains(&expected_disposition) {
            errors.push(format!(
                "corpus contains unknown expected disposition {expected_disposition:?}"
            ));
        }

        let actual = observed_by_id.get(id).filter(|matches| matches.len() == 1);
        let Some(actual) = actual.and_then(|matches| matches.first()).copied() else {
            errors.push("result missing or duplicated".to_owned());
            case_reports.push(json!({
                "id": id,
                "expected_disposition": expected_disposition,
                "observed_disposition": Value::Null,
                "passed": false,
                "errors": errors
            }));
            continue;
        };

        let observed_disposition = text(actual, "observed_disposition").unwrap_or_default();
        if !ALLOWED_DISPOSITIONS.contains(&observed_disposition) {
            errors.push(format!("unknown observed disposition {observed_disposition:?}"));
        }
        if observed_disposition != expected_disposition {
            errors.push("observed disposition does not match frozen expectation".to_owned());
        }

        let required = match strings(expected, "required_findings") {
            Some(findings) if !findings.is_empty() => findings,
            _ => {
                errors.push("corpus required_findings missing, malformed, or empty".to_owned());
                BTreeSet::new()
            }
        };
        let observed_findings = match strings(actual, "observed_findings") {
            Some(findings) => findings,
            None => {
                errors.push("observed_findings missing or malformed".to_owned());
                BTreeSet::new()
            }
        };
        let missing_findings = required
            .difference(&observed_findings)
            .cloned()
            .collect::<Vec<_>>();
        if !missing_findings.is_empty() {
            errors.push("one or more required findings are missing".to_owned());
        }

        let prohibited = match strings(expected, "prohibited_promotions") {
            Some(promotions) if !promotions.is_empty() => promotions,
            _ => {
                errors.push("corpus prohibited_promotions missing, malformed, or empty".to_owned());
                BTreeSet::new()
            }
        };
        let observed_promotions = match strings(actual, "observed_promotions") {
            Some(promotions) => promotions,
            None => {
                errors.push("observed_promotions missing or malformed".to_owned());
                BTreeSet::new()
            }
        };
        let violations = prohibited
            .intersection(&observed_promotions)
            .cloned()
            .collect::<Vec<_>>();
        if !violations.is_empty() {
            errors.push("prohibited evidence promotion detected".to_owned());
        }

        let mut computed_numeric_checks = Value::Null;
        if text(expected, "class") == Some("analytic-reference") {
            let fixture = expected.get("fixture");
            let equation = fixture.and_then(|value| text(value, "equation"));
            let fresh = fixture.and_then(|value| value.get("fresh")).and_then(Value::as_f64);
            let recycle_fraction = fixture
                .and_then(|value| value.get("recycle_fraction"))
                .and_then(Value::as_f64);
            let analytic_solution = fixture
                .and_then(|value| value.get("analytic_solution"))
                .and_then(Value::as_f64);
            let observed_solution = actual
                .get("observed_numeric_values")
                .and_then(|value| value.get("solution"))
                .and_then(Value::as_f64);
            let tolerance = corpus
                .get("residual_profile")
                .and_then(|value| value.get("target_max"))
                .and_then(Value::as_f64);

            if equation != Some("x = fresh + recycle_fraction*x") {
                errors.push(
                    "analytic-reference fixture has no supported oracle equation".to_owned(),
                );
            } else if let (
                Some(fresh),
                Some(recycle_fraction),
                Some(analytic_solution),
                Some(observed_solution),
                Some(tolerance),
            ) = (
                fresh,
                recycle_fraction,
                analytic_solution,
                observed_solution,
                tolerance,
            ) {
                if ![fresh, recycle_fraction, analytic_solution, observed_solution, tolerance]
                    .iter()
                    .all(|number| number.is_finite())
                    || tolerance <= 0.0
                {
                    errors.push(
                        "analytic-reference numeric values must be finite and tolerance positive"
                            .to_owned(),
                    );
                } else {
                    let rhs = fresh + recycle_fraction * observed_solution;
                    let residual_scale = 1.0_f64.max(
                        observed_solution.abs()
                            + fresh.abs()
                            + (recycle_fraction * observed_solution).abs(),
                    );
                    let normalized_residual = (observed_solution - rhs).abs() / residual_scale;
                    let solution_relative_error = (observed_solution - analytic_solution).abs()
                        / 1.0_f64.max(analytic_solution.abs());
                    let numeric_passed = normalized_residual <= tolerance
                        && solution_relative_error <= tolerance;
                    if normalized_residual > tolerance {
                        errors.push(
                            concat!(
                                "independently calculated recycle-equation residual exceeds ",
                                "frozen tolerance"
                            )
                            .to_owned(),
                        );
                    }
                    if solution_relative_error > tolerance {
                        errors.push(
                            concat!(
                                "observed solution differs from the frozen ",
                                "analytic solution"
                            )
                            .to_owned(),
                        );
                    }
                    computed_numeric_checks = json!({
                        "oracle": "linear_recycle_v1",
                        "observed_solution": observed_solution,
                        "analytic_solution": analytic_solution,
                        "normalized_equation_residual": normalized_residual,
                        "solution_relative_error": solution_relative_error,
                        "tolerance": tolerance,
                        "passed": numeric_passed
                    });
                }
            } else {
                errors.push(
                    "analytic-reference result is missing numeric values or tolerance".to_owned(),
                );
            }
        }

        if matches!(id, "CHEM-PROC-001" | "CHEM-PROC-002") {
            let tolerance = corpus
                .get("residual_profile")
                .and_then(|profile| numeric_field(profile, "target_max"))
                .unwrap_or(f64::NAN);
            match expected.get("fixture").ok_or_else(|| "fixture missing".to_owned())
                .and_then(|fixture| mixer_balance_oracle(fixture, tolerance)) {
                Ok(oracle) => {
                    if text(&oracle, "derived_disposition") != Some(expected_disposition) {
                        errors.push(
                            "independent mixer oracle disagrees with frozen expected disposition"
                                .to_owned(),
                        );
                    }
                    let observed_residuals = actual
                        .get("observed_numeric_values")
                        .and_then(|value| value.get("component_balance_residuals"))
                        .and_then(Value::as_object);
                    match observed_residuals {
                        Some(residuals) => {
                            if let Some(components) = oracle
                                .get("components")
                                .and_then(Value::as_object)
                            {
                                for (component, expected_balance) in components {
                                    let expected_residual =
                                        numeric_field(expected_balance, "raw_residual")
                                        .unwrap_or(f64::NAN);
                                    let normalization_scale =
                                        numeric_field(expected_balance, "normalization_scale")
                                        .unwrap_or(f64::NAN);
                                    let reported_residual = residuals
                                        .get(component)
                                        .and_then(Value::as_f64);
                                    match reported_residual {
                                        Some(reported) if reported.is_finite()
                                            && normalization_scale.is_finite()
                                            && normalization_scale > 0.0
                                            && (reported - expected_residual).abs()
                                                / normalization_scale <= tolerance => {}
                                        _ => errors.push(format!(
                                            concat!(
                                                "reported residual for component {component:?} ",
                                                "is missing or disagrees with independent oracle"
                                            )
                                        )),
                                    }
                                }
                                for component in residuals.keys() {
                                    if !components.contains_key(component) {
                                        errors.push(format!(
                                            "unexpected reported balance component {component:?}"
                                        ));
                                    }
                                }
                            }
                        }
                        None => errors.push(
                            concat!(
                                "mixer case result is missing observed_numeric_values.",
                                "component_balance_residuals"
                            )
                                .to_owned()
                        ),
                    }
                    if oracle.get("balance_closed").and_then(Value::as_bool) != Some(true)
                        && expected_disposition == "ACCEPTED_WITHIN_SCOPE"
                    {
                        errors.push("unclosed mixer balance cannot be accepted".to_owned());
                    }
                    computed_numeric_checks = oracle;
                }
                Err(reason) => errors.push(format!("independent mixer oracle failed: {reason}")),
            }
        }

        if id == "CHEM-PROC-004" {
            let tolerance = corpus
                .get("residual_profile")
                .and_then(|profile| numeric_field(profile, "target_max"))
                .unwrap_or(f64::NAN);
            let empty_numeric_values = Value::Null;
            let observed_numeric_values = actual
                .get("observed_numeric_values")
                .unwrap_or(&empty_numeric_values);
            match expected.get("fixture").ok_or_else(|| "fixture missing".to_owned())
                .and_then(|fixture| {
                    unit_conversion_oracle(fixture, observed_numeric_values, tolerance)
                }) {
                Ok(oracle) => {
                    if oracle.get("passed").and_then(Value::as_bool) != Some(true) {
                        errors.push("independent unit-conversion oracle failed".to_owned());
                    }
                    if expected_disposition != "ACCEPTED_WITHIN_SCOPE" {
                        errors.push(
                            "unit-conversion control must preserve its frozen expected disposition"
                                .to_owned(),
                        );
                    }
                    computed_numeric_checks = oracle;
                }
                Err(reason) => errors.push(format!(
                    "independent unit-conversion oracle failed: {reason}"
                )),
            }
        }

        let semantic_oracle_ids = [
            "CHEM-PROC-003", "CHEM-PROC-005", "CHEM-PROC-006", "CHEM-PROC-008",
            "CHEM-PROC-009", "CHEM-PROC-010", "CHEM-PROC-011", "CHEM-PROC-012",
            "CHEM-PROC-013", "CHEM-PROC-014", "CHEM-PROC-015", "CHEM-PROC-016",
            "CHEM-PROC-017", "CHEM-PROC-018", "CHEM-PROC-019", "CHEM-PROC-020",
            "CHEM-PROC-021", "CHEM-PROC-022", "CHEM-PROC-023", "CHEM-PROC-024",
            "CHEM-PROC-025", "CHEM-PROC-026",
        ];
        if semantic_oracle_ids.contains(&id) {
            let tolerance = corpus
                .get("residual_profile")
                .and_then(|profile| numeric_field(profile, "target_max"))
                .unwrap_or(f64::NAN);
            match expected
                .get("fixture")
                .ok_or_else(|| "fixture missing".to_owned())
                .and_then(|fixture| derive_synthetic_disposition(id, fixture, tolerance))
            {
                Ok(oracle) => {
                    if text(&oracle, "derived_disposition") != Some(expected_disposition) {
                        errors.push(
                            concat!(
                                "independent synthetic oracle disagrees with ",
                                "frozen expected disposition"
                            )
                                .to_owned(),
                        );
                    }
                    computed_numeric_checks = oracle;
                }
                Err(reason) => errors.push(format!(
                    "independent synthetic oracle failed: {reason}"
                )),
            }
        }

        let passed = errors.is_empty();
        case_reports.push(json!({
            "id": id,
            "expected_disposition": expected_disposition,
            "observed_disposition": observed_disposition,
            "missing_required_findings": missing_findings,
            "prohibited_promotions_observed": violations,
            "computed_numeric_checks": computed_numeric_checks,
            "passed": passed,
            "errors": errors
        }));
    }

    report(
        corpus,
        expected_head,
        reported_head,
        global_errors,
        case_reports,
        observed_cases.len(),
    )
}

fn report(
    corpus: &Value,
    expected_head: &str,
    reported_head: &str,
    global_errors: Vec<String>,
    cases: Vec<Value>,
    observed_count: usize,
) -> Value {
    let case_count = cases.len();
    let passed_count = cases
        .iter()
        .filter(|case| case.get("passed").and_then(Value::as_bool) == Some(true))
        .count();
    let failed_count = case_count.saturating_sub(passed_count);
    let no_global_errors = global_errors.is_empty();
    let passed = no_global_errors && failed_count == 0 && observed_count == case_count;

    json!({
        "report_schema_version": "1.0.0",
        "report_kind": "synthetic_chemical_process_contract_evaluation",
        "benchmark_id": text(corpus, "benchmark_id").unwrap_or("UNKNOWN"),
        "corpus_version": text(corpus, "corpus_version").unwrap_or("UNKNOWN"),
        "expected_head": expected_head,
        "reported_head": reported_head,
        "expected_case_count": case_count,
        "observed_case_count": observed_count,
        "passed_cases": passed_count,
        "failed_cases": failed_count,
        "global_errors": global_errors,
        "passed": passed,
        "claim_ceiling": concat!(
            "This result checks only agreement with the frozen synthetic benchmark contract. ",
            "It does not establish chemistry, process-model accuracy, product grade, safety, ",
            "economics, plant qualification, or physical operating authority."
        ),
        "cases": cases
    })
}

fn run() -> Result<bool, Box<dyn Error>> {
    let args = parse_args().map_err(|message| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
    })?;
    let (corpus, corpus_sha256) = read_json_with_digest(&args.corpus_path)?;
    let (results, results_sha256) = read_json_with_digest(&args.results_path)?;
    let mut report = evaluate(&corpus, &results, &args.expected_head);
    if let Some(report_object) = report.as_object_mut() {
        report_object.insert("corpus_sha256".to_owned(), json!(corpus_sha256));
        report_object.insert("results_sha256".to_owned(), json!(results_sha256));
    }
    let passed = report.get("passed").and_then(Value::as_bool) == Some(true);
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(passed)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("chemical-process benchmark evaluator: {error}");
            ExitCode::from(2)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const FROZEN_CORPUS_JSON: &str = include_str!(
        "../../../../docs/engineering/fixtures/CHEMICAL_PROCESS_QUALIFICATION_V1.json"
    );

    #[test]
    fn sha256_digest_matches_published_test_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    fn mini_corpus() -> Value {
        json!({
            "benchmark_id": "TEST-BENCH",
            "corpus_version": "1.0.0",
            "allowed_dispositions": ALLOWED_DISPOSITIONS,
            "cases": [
                {
                    "id": "CASE-1",
                    "class": "positive-control",
                    "expected_disposition": "ACCEPTED_WITHIN_SCOPE",
                    "required_findings": ["balance_checked"],
                    "prohibited_promotions": ["plant_qualified"],
                    "fixture": { "kind": "synthetic" }
                },
                {
                    "id": "CASE-2",
                    "class": "negative-control",
                    "expected_disposition": "BLOCKED",
                    "required_findings": ["missing_utility_found"],
                    "prohibited_promotions": ["production_capability_complete"],
                    "fixture": { "kind": "synthetic" }
                }
            ]
        })
    }

    fn result_case(
        id: &str,
        disposition: &str,
        findings: &[&str],
        promotions: &[&str],
    ) -> Value {
        json!({
            "id": id,
            "observed_disposition": disposition,
            "observed_findings": findings,
            "observed_promotions": promotions
        })
    }

    fn good_results() -> Value {
        json!({
            "benchmark_id": "TEST-BENCH",
            "corpus_version": "1.0.0",
            "evaluated_head": HEAD,
            "cases": [
                result_case("CASE-1", "ACCEPTED_WITHIN_SCOPE", &["balance_checked"], &[]),
                result_case("CASE-2", "BLOCKED", &["missing_utility_found"], &[])
            ]
        })
    }

    #[test]
    fn matching_dispositions_and_findings_pass_contract() {
        let output = evaluate(&mini_corpus(), &good_results(), HEAD);
        assert_eq!(output["passed"], true);
        assert_eq!(output["passed_cases"], 2);
        assert_eq!(output["failed_cases"], 0);
    }

    #[test]
    fn disposition_mismatch_and_missing_findings_fail_closed() {
        let mut results = good_results();
        results["cases"][0]["observed_disposition"] = json!("BLOCKED");
        results["cases"][0]["observed_findings"] = json!([]);
        let output = evaluate(&mini_corpus(), &results, HEAD);
        assert_eq!(output["passed"], false);
        assert_eq!(output["cases"][0]["passed"], false);
        assert!(!output["cases"][0]["errors"].as_array().unwrap().is_empty());
    }

    #[test]
    fn prohibited_promotions_fail_even_when_disposition_matches() {
        let mut results = good_results();
        results["cases"][0]["observed_promotions"] = json!(["plant_qualified"]);
        let output = evaluate(&mini_corpus(), &results, HEAD);
        assert_eq!(output["passed"], false);
        assert_eq!(
            output["cases"][0]["prohibited_promotions_observed"],
            json!(["plant_qualified"])
        );
    }

    #[test]
    fn report_head_must_match_independently_supplied_head() {
        let mut results = good_results();
        results["evaluated_head"] = json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        let output = evaluate(&mini_corpus(), &results, HEAD);
        assert_eq!(output["passed"], false);
        assert!(
            output["global_errors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|error| error.as_str().unwrap().contains("expected-head"))
        );
    }

    #[test]
    fn duplicate_or_unknown_case_ids_fail_closed() {
        let mut results = good_results();
        results["cases"] = json!([
            result_case("CASE-1", "ACCEPTED_WITHIN_SCOPE", &["balance_checked"], &[]),
            result_case("CASE-1", "ACCEPTED_WITHIN_SCOPE", &["balance_checked"], &[]),
            result_case("UNEXPECTED", "BLOCKED", &["some_finding"], &[])
        ]);
        let output = evaluate(&mini_corpus(), &results, HEAD);
        assert_eq!(output["passed"], false);
        assert!(
            output["global_errors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|error| error.as_str().unwrap().contains("duplicate result case id"))
        );
        assert!(
            output["global_errors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|error| error.as_str().unwrap().contains("unexpected result case id"))
        );
    }

    #[test]
    fn skipped_required_case_is_not_treated_as_pass() {
        let mut results = good_results();
        results["cases"] = json!([
            result_case("CASE-1", "ACCEPTED_WITHIN_SCOPE", &["balance_checked"], &[])
        ]);
        let output = evaluate(&mini_corpus(), &results, HEAD);
        assert_eq!(output["passed"], false);
        assert_eq!(output["cases"][1]["passed"], false);
    }

    fn analytic_corpus() -> Value {
        json!({
            "benchmark_id": "ANALYTIC-TEST",
            "corpus_version": "1.0.0",
            "allowed_dispositions": ALLOWED_DISPOSITIONS,
            "residual_profile": { "target_max": 1e-8 },
            "cases": [{
                "id": "RECYCLE-1",
                "class": "analytic-reference",
                "expected_disposition": "ACCEPTED_WITHIN_SCOPE",
                "required_findings": [
                    "solution_matches_analytic_reference",
                    "residual_within_frozen_tolerance"
                ],
                "prohibited_promotions": ["plant_build_authorized"],
                "fixture": {
                    "equation": "x = fresh + recycle_fraction*x",
                    "fresh": 1.0,
                    "recycle_fraction": 0.2,
                    "analytic_solution": 1.25
                }
            }]
        })
    }

    fn analytic_results(solution: f64) -> Value {
        json!({
            "benchmark_id": "ANALYTIC-TEST",
            "corpus_version": "1.0.0",
            "evaluated_head": HEAD,
            "cases": [{
                "id": "RECYCLE-1",
                "observed_disposition": "ACCEPTED_WITHIN_SCOPE",
                "observed_findings": [
                    "solution_matches_analytic_reference",
                    "residual_within_frozen_tolerance"
                ],
                "observed_promotions": [],
                "observed_numeric_values": { "solution": solution }
            }]
        })
    }

    
    #[test]
    fn malformed_corpus_contract_fails_closed() {
        let mut invalid_corpus = mini_corpus();
        invalid_corpus["cases"][0]
            .as_object_mut()
            .unwrap()
            .remove("prohibited_promotions");

        let output = evaluate(&invalid_corpus, &good_results(), HEAD);
        assert_eq!(output["passed"], false);
        assert!(
            output["global_errors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|error| error.as_str().unwrap().contains("prohibited_promotions"))
        );
    }

    #[test]
    fn analytical_recycle_oracle_recomputes_residual() {
        let output = evaluate(&analytic_corpus(), &analytic_results(1.25), HEAD);
        assert_eq!(output["passed"], true);
        assert_eq!(
            output["cases"][0]["computed_numeric_checks"]["normalized_equation_residual"],
            0.0
        );
        assert_eq!(output["cases"][0]["computed_numeric_checks"]["passed"], true);
    }

    #[test]
    fn analytical_recycle_oracle_rejects_a_plausible_but_wrong_solution() {
        let output = evaluate(&analytic_corpus(), &analytic_results(2.0), HEAD);
        assert_eq!(output["passed"], false);
        assert_eq!(output["cases"][0]["computed_numeric_checks"]["passed"], false);
    }

    fn single_numeric_case(
        id: &str,
        class: &str,
        disposition: &str,
        fixture: Value,
        required_findings: &[&str],
        prohibited_promotions: &[&str],
    ) -> Value {
        json!({
            "benchmark_id": "NUMERIC-ORACLE-TEST",
            "corpus_version": "1.0.0",
            "allowed_dispositions": ALLOWED_DISPOSITIONS,
            "residual_profile": { "target_max": 1e-8 },
            "cases": [{
                "id": id,
                "class": class,
                "expected_disposition": disposition,
                "required_findings": required_findings,
                "prohibited_promotions": prohibited_promotions,
                "fixture": fixture
            }]
        })
    }

    fn numeric_result(
        id: &str,
        disposition: &str,
        findings: &[&str],
        values: Value,
    ) -> Value {
        json!({
            "benchmark_id": "NUMERIC-ORACLE-TEST",
            "corpus_version": "1.0.0",
            "evaluated_head": HEAD,
            "cases": [{
                "id": id,
                "observed_disposition": disposition,
                "observed_findings": findings,
                "observed_promotions": [],
                "observed_numeric_values": values
            }]
        })
    }

    #[test]
    fn mixer_oracle_accepts_balanced_component_flow() {
        let fixture = json!({
            "unit": "mixer",
            "reaction": false,
            "inlets": [
                {"component": "A", "flow": 2.0, "unit": "mol/s"},
                {"component": "A", "flow": 3.0, "unit": "mol/s"}
            ],
            "outlets": [
                {"component": "A", "flow": 5.0, "unit": "mol/s"}
            ]
        });
        let corpus = single_numeric_case(
            "CHEM-PROC-001",
            "positive-control",
            "ACCEPTED_WITHIN_SCOPE",
            fixture,
            &["component_balance_closed"],
            &["process_safety_qualified"],
        );
        let results = numeric_result(
            "CHEM-PROC-001",
            "ACCEPTED_WITHIN_SCOPE",
            &["component_balance_closed"],
            json!({"component_balance_residuals": {"A": 0.0}}),
        );
        let output = evaluate(&corpus, &results, HEAD);
        assert_eq!(output["passed"], true);
        assert_eq!(
            output["cases"][0]["computed_numeric_checks"]["components"]["A"]["raw_residual"],
            0.0
        );
        assert_eq!(
            output["cases"][0]["computed_numeric_checks"]["balance_closed"],
            true
        );
    }

    #[test]
    fn mixer_oracle_requires_imbalance_to_remain_blocked() {
        let fixture = json!({
            "unit": "mixer",
            "reaction": false,
            "unexplained_sink": false,
            "inlets": [{"component": "A", "flow": 2.0, "unit": "mol/s"}],
            "outlets": [{"component": "A", "flow": 1.0, "unit": "mol/s"}]
        });
        let corpus = single_numeric_case(
            "CHEM-PROC-002",
            "negative-control",
            "BLOCKED",
            fixture,
            &["component_balance_residual_nonzero"],
            &["balance_closed"],
        );
        let results = numeric_result(
            "CHEM-PROC-002",
            "BLOCKED",
            &["component_balance_residual_nonzero"],
            json!({"component_balance_residuals": {"A": 1.0}}),
        );
        let output = evaluate(&corpus, &results, HEAD);
        assert_eq!(output["passed"], true);
        assert_eq!(
            output["cases"][0]["computed_numeric_checks"]["derived_disposition"],
            "BLOCKED"
        );
        assert_eq!(
            output["cases"][0]["computed_numeric_checks"]["components"]["A"]["raw_residual"],
            1.0
        );
        assert_eq!(
            output["cases"][0]["computed_numeric_checks"]["balance_closed"],
            false
        );
    }

    #[test]
    fn mixer_oracle_rejects_a_claimed_zero_residual_when_streams_do_not_balance() {
        let fixture = json!({
            "unit": "mixer",
            "reaction": false,
            "inlets": [{"component": "A", "flow": 2.0, "unit": "mol/s"}],
            "outlets": [{"component": "A", "flow": 1.0, "unit": "mol/s"}]
        });
        let corpus = single_numeric_case(
            "CHEM-PROC-002",
            "negative-control",
            "BLOCKED",
            fixture,
            &["component_balance_residual_nonzero"],
            &["balance_closed"],
        );
        let results = numeric_result(
            "CHEM-PROC-002",
            "BLOCKED",
            &["component_balance_residual_nonzero"],
            json!({"component_balance_residuals": {"A": 0.0}}),
        );
        let output = evaluate(&corpus, &results, HEAD);
        assert_eq!(output["passed"], false);
        assert!(
            output["cases"][0]["errors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|error| error.as_str().unwrap().contains("independent oracle"))
        );
    }

    #[test]
    fn unit_conversion_oracle_accepts_explicit_equivalent_representation() {
        let fixture = json!({
            "conversion_kind": "volume_flow",
            "from_value": 1.0,
            "from_unit": "m^3/s",
            "to_value": 1000.0,
            "to_unit": "L/s",
            "conversion_factor": 1000.0,
            "conversion_factor_ref": "synthetic-unit-conversion-v1",
            "conversion_rule": "to_value = from_value * conversion_factor"
        });
        let corpus = single_numeric_case(
            "CHEM-PROC-004",
            "metamorphic",
            "ACCEPTED_WITHIN_SCOPE",
            fixture,
            &["normalized_result_equivalent_within_frozen_tolerance"],
            &["unit_conversion_inferred_without_evidence"],
        );
        let results = numeric_result(
            "CHEM-PROC-004",
            "ACCEPTED_WITHIN_SCOPE",
            &["normalized_result_equivalent_within_frozen_tolerance"],
            json!({"converted_value": 1000.0}),
        );
        let output = evaluate(&corpus, &results, HEAD);
        assert_eq!(output["passed"], true);
        assert_eq!(
            output["cases"][0]["computed_numeric_checks"]["observed_conversion_valid"],
            true
        );
        assert_eq!(
            output["cases"][0]["computed_numeric_checks"]["fixture_reference_normalized_residual"],
            0.0
        );
    }

    #[test]
    fn unit_conversion_oracle_rejects_wrong_conversion_result() {
        let fixture = json!({
            "conversion_kind": "volume_flow",
            "from_value": 1.0,
            "from_unit": "m^3/s",
            "to_value": 1000.0,
            "to_unit": "L/s",
            "conversion_factor": 1000.0,
            "conversion_factor_ref": "synthetic-unit-conversion-v1",
            "conversion_rule": "to_value = from_value * conversion_factor"
        });
        let corpus = single_numeric_case(
            "CHEM-PROC-004",
            "metamorphic",
            "ACCEPTED_WITHIN_SCOPE",
            fixture,
            &["normalized_result_equivalent_within_frozen_tolerance"],
            &["unit_conversion_inferred_without_evidence"],
        );
        let results = numeric_result(
            "CHEM-PROC-004",
            "ACCEPTED_WITHIN_SCOPE",
            &["normalized_result_equivalent_within_frozen_tolerance"],
            json!({"converted_value": 900.0}),
        );
        let output = evaluate(&corpus, &results, HEAD);
        assert_eq!(output["passed"], false);
        assert_eq!(
            output["cases"][0]["computed_numeric_checks"]["observed_conversion_valid"],
            false
        );
    }

    #[test]
    fn common_mode_dependency_oracle_fails_closed_on_shared_or_incomplete_provenance() {
        let corpus: Value = serde_json::from_str(FROZEN_CORPUS_JSON)
            .expect("frozen synthetic corpus must parse");
        let cases = corpus["cases"].as_array().expect("cases must be an array");
        let shared_case = cases
            .iter()
            .find(|case| text(case, "id") == Some("CHEM-PROC-025"))
            .expect("common-mode negative control must exist");
        let shared_oracle = derive_synthetic_disposition(
            "CHEM-PROC-025",
            &shared_case["fixture"],
            1.0e-8,
        )
        .expect("shared-dependency oracle must calculate a disposition");
        assert_eq!(text(&shared_oracle, "derived_disposition"), Some("UNRESOLVED"));
        assert!(
            shared_oracle["shared_evidence_dependency_digests"]
                .as_array()
                .is_some_and(|values| !values.is_empty()),
            "shared property data must be visible in the independent checker report"
        );

        let independent_case = cases
            .iter()
            .find(|case| text(case, "id") == Some("CHEM-PROC-026"))
            .expect("independence positive control must exist");
        let mut fixture = independent_case["fixture"].clone();
        let independent_oracle = derive_synthetic_disposition(
            "CHEM-PROC-026",
            &fixture,
            1.0e-8,
        )
        .expect("disjoint-dependency oracle must calculate a disposition");
        assert_eq!(
            text(&independent_oracle, "derived_disposition"),
            Some("ACCEPTED_WITHIN_SCOPE")
        );

        let shared_digest = fixture["solver_a"]["evidence_dependency_digests"][0].clone();
        fixture["solver_b"]["evidence_dependency_digests"][0] = shared_digest;
        let mutated_oracle = derive_synthetic_disposition("CHEM-PROC-026", &fixture, 1.0e-8)
            .expect("mutated common-mode fixture must remain evaluable");
        assert_eq!(text(&mutated_oracle, "derived_disposition"), Some("UNRESOLVED"));

        fixture["dependency_manifests_complete"] = json!(false);
        let incomplete_oracle = derive_synthetic_disposition("CHEM-PROC-026", &fixture, 1.0e-8)
            .expect("missing provenance must become unresolved, not a checker crash");
        assert_eq!(text(&incomplete_oracle, "derived_disposition"), Some("UNRESOLVED"));
    }

    #[test]
    fn all_twenty_two_semantic_oracles_match_the_frozen_fixture_dispositions() {
        let corpus: Value = serde_json::from_str(FROZEN_CORPUS_JSON)
            .expect("frozen synthetic corpus must parse for oracle coverage tests");
        let tolerance = corpus["residual_profile"]["target_max"]
            .as_f64()
            .expect("frozen corpus must declare a numeric tolerance");
        let cases = corpus["cases"].as_array().expect("frozen cases must be an array");
        let mut checked = BTreeSet::new();

        for case in cases {
            let id = text(case, "id").expect("every frozen case must have an ID");
            if matches!(
                id,
                "CHEM-PROC-001" | "CHEM-PROC-002" | "CHEM-PROC-004" | "CHEM-PROC-007"
            ) {
                continue;
            }

            let fixture = case.get("fixture").expect("case must have a fixture");
            let oracle = derive_synthetic_disposition(id, fixture, tolerance)
                .unwrap_or_else(|reason| {
                    panic!("{id} should have an independent oracle: {reason}")
                });
            assert_eq!(
                text(&oracle, "derived_disposition"),
                text(case, "expected_disposition"),
                "independent semantic oracle disagrees for {id}: {oracle}"
            );
            assert!(checked.insert(id.to_owned()), "semantic oracle case {id} was duplicated");
        }

        assert_eq!(
            checked.len(),
            22,
            "every non-numeric fixture needs an independent semantic oracle"
        );
    }

}
