use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn required_text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty text field {key}"))
}

fn finite_number(value: &Value, context: &str) -> Result<f64, String> {
    let number = value
        .as_f64()
        .ok_or_else(|| format!("{context} must be numeric"))?;
    if !number.is_finite() {
        return Err(format!("{context} must be finite"));
    }
    Ok(number)
}

fn numeric_array(value: &Value, context: &str) -> Result<(), String> {
    let array = value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?;
    for (index, item) in array.iter().enumerate() {
        if let Some(row) = item.as_array() {
            for (column, element) in row.iter().enumerate() {
                finite_number(element, &format!("{context}[{index}][{column}]"))?;
            }
        } else {
            finite_number(item, &format!("{context}[{index}]"))?;
        }
    }
    Ok(())
}

fn validate_observable_value(observation: &Value) -> Result<(), String> {
    let id = required_text(observation, "observable_id")?;
    let kind = required_text(observation, "observable_kind")?;
    let value_kind = required_text(observation, "value_kind")?;
    let value = observation
        .get("value")
        .ok_or_else(|| format!("observable {id} missing value field"))?;

    let uncertainty = observation
        .get("uncertainty")
        .ok_or_else(|| format!("observable {id} missing uncertainty"))?;
    if let Some(value) = uncertainty.as_f64() {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("observable {id}: uncertainty must be finite and non-negative"));
        }
    } else if let Some(values) = uncertainty.as_array() {
        numeric_array(&Value::Array(values.clone()), &format!("observable {id} uncertainty"))?;
        if values.iter().any(|value| value.as_f64().is_some_and(|number| number < 0.0)) {
            return Err(format!("observable {id}: uncertainty entries must be non-negative"));
        }
    } else if !uncertainty.is_null() {
        return Err(format!("observable {id}: uncertainty must be numeric, array or null"));
    }

    if value_kind == "not_reported" {
        if !value.is_null() {
            return Err(format!("observable {id}: not_reported must have null value"));
        }
        return Ok(());
    }
    if value.is_null() {
        return Err(format!("observable {id}: reported value_kind requires a value"));
    }
    if value_kind == "scalar" {
        finite_number(value, &format!("observable {id} value"))?;
    } else if value_kind == "vector" || value_kind == "matrix" || value_kind == "spectrum" {
        numeric_array(value, &format!("observable {id} value"))?;
    } else {
        return Err(format!("observable {id}: unsupported value_kind {value_kind}"));
    }

    if kind == "lambda_L_scalar" {
        if value_kind != "scalar" {
            return Err(format!("observable {id}: lambda_L_scalar must be scalar"));
        }
        if required_text(observation, "isotropy_assumption")? != "isotropic_declared" {
            return Err(format!(
                "observable {id}: scalar lambda_L requires an explicit isotropy assumption"
            ));
        }
    }
    if kind == "lambda_L_tensor" {
        if value_kind != "matrix"
            || required_text(observation, "isotropy_assumption")? != "anisotropic_tensor"
        {
            return Err(format!(
                "observable {id}: lambda_L_tensor requires a matrix and anisotropic_tensor declaration"
            ));
        }
        let axes = observation
            .get("tensor_axes")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("observable {id}: tensor_axes required"))?;
        if axes.len() < 2 {
            return Err(format!("observable {id}: lambda tensor requires at least two declared axes"));
        }
    }
    if kind == "xi0_microscopic" {
        if required_text(observation, "length_scale_definition")? != "microscopic_xi0" {
            return Err(format!(
                "observable {id}: microscopic xi0 cannot be relabelled as xi_GL"
            ));
        }
    }
    if kind == "xi_GL" {
        if required_text(observation, "length_scale_definition")? != "GL_near_Tc_xi" {
            return Err(format!(
                "observable {id}: xi_GL requires the near-Tc GL definition"
            ));
        }
    }
    if kind != "xi0_microscopic" && kind != "xi_GL" {
        if observation
            .get("length_scale_definition")
            .and_then(Value::as_str)
            .is_some_and(|definition| definition != "not_applicable")
        {
            return Err(format!(
                "observable {id}: length_scale_definition is only valid for xi0/xi_GL"
            ));
        }
    }
    Ok(())
}

fn uncertainty_is_present_and_finite(value: &Value, context: &str) -> Result<bool, String> {
    if value.is_null() {
        return Ok(false);
    }
    if let Some(number) = value.as_f64() {
        if !number.is_finite() || number < 0.0 {
            return Err(format!("{context} must be finite and non-negative"));
        }
        return Ok(true);
    }
    numeric_array(value, context)?;
    Ok(true)
}

fn exportable_coefficients<'a>(coefficients: &'a [Value]) -> Vec<&'a Value> {
    coefficients
        .iter()
        .filter(|coefficient| {
            matches!(
                coefficient.get("status").and_then(Value::as_str),
                Some("derived" | "constrained" | "phenomenological_fit")
            ) && coefficient.get("value").is_some_and(|value| !value.is_null())
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct ScalarGlInvariants {
    xi_squared: f64,
    inverse_lambda_squared_proxy: f64,
}

fn scalar_gl_invariants(a: f64, b: f64, k: f64) -> Result<ScalarGlInvariants, String> {
    if !a.is_finite() || !b.is_finite() || !k.is_finite() || a >= 0.0 || b <= 0.0 || k <= 0.0 {
        return Err("scalar GL invariants require a<0, b>0 and K>0".to_owned());
    }
    let rho0_squared = -a / b;
    let xi_squared = k / -a;
    let inverse_lambda_squared_proxy = k * rho0_squared;
    if !xi_squared.is_finite() || !inverse_lambda_squared_proxy.is_finite() {
        return Err("scalar GL invariants are non-finite".to_owned());
    }
    Ok(ScalarGlInvariants {
        xi_squared,
        inverse_lambda_squared_proxy,
    })
}

fn scalar_gl_basis_rescaling(a: f64, b: f64, k: f64, scale: f64) -> Result<Value, String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err("order-parameter basis scale must be positive and finite".to_owned());
    }
    let original = scalar_gl_invariants(a, b, k)?;
    let scale_squared = scale * scale;
    let transformed = (
        a / scale_squared,
        b / (scale_squared * scale_squared),
        k / scale_squared,
    );
    let rescaled = scalar_gl_invariants(transformed.0, transformed.1, transformed.2)?;
    let invariant_residual = (original.xi_squared - rescaled.xi_squared)
        .abs()
        .max(
            (original.inverse_lambda_squared_proxy - rescaled.inverse_lambda_squared_proxy)
                .abs(),
        );
    Ok(json!({
        "basis_scale": scale,
        "original_coefficients": {"a": a, "b": b, "K": k},
        "rescaled_coefficients": {"a": transformed.0, "b": transformed.1, "K": transformed.2},
        "original_observable_proxies": {
            "xi_squared": original.xi_squared,
            "inverse_lambda_squared_proxy": original.inverse_lambda_squared_proxy
        },
        "rescaled_observable_proxies": {
            "xi_squared": rescaled.xi_squared,
            "inverse_lambda_squared_proxy": rescaled.inverse_lambda_squared_proxy
        },
        "maximum_invariant_residual": invariant_residual,
        "coefficient_tuple_changed": true,
        "interpretation": "Tc, xi and lambda-like response do not identify absolute GL coefficient magnitudes unless the order-parameter basis normalization is fixed; this scalar GL demonstration is synthetic and convention-specific."
    }))
}

fn validate_coefficient(
    coefficient: &Value,
    observables: &HashMap<String, &Value>,
    state_ids: &HashSet<String>,
    calculation_family: &str,
    mechanism_scope: &str,
) -> Result<(), String> {
    let id = required_text(coefficient, "coefficient_id")?;
    let kind = required_text(coefficient, "coefficient_kind")?;
    let status = required_text(coefficient, "status")?;
    let value_kind = required_text(coefficient, "value_kind")?;
    let value = coefficient
        .get("value")
        .ok_or_else(|| format!("coefficient {id} missing value"))?;
    let state_id = required_text(coefficient, "state_id")?;
    if !state_ids.contains(state_id) {
        return Err(format!("coefficient {id} references unknown state {state_id}"));
    }
    let _ = required_text(coefficient, "basis_normalization")?;
    let _ = required_text(coefficient, "temperature_regime")?;
    let _ = required_text(coefficient, "field_regime")?;
    let derivation_method = required_text(coefficient, "derivation_method")?;
    let _ = required_text(coefficient, "unit")?;
    let model_family = required_text(coefficient, "model_family")?;

    let support_ids = coefficient
        .get("supporting_observable_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("coefficient {id} supporting_observable_ids must be an array"))?;
    let mut supporting = Vec::with_capacity(support_ids.len());
    for support in support_ids {
        let observable_id = support
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| format!("coefficient {id} has empty supporting observable ID"))?;
        let observation = observables
            .get(observable_id)
            .ok_or_else(|| {
                format!("coefficient {id} references unknown observable {observable_id}")
            })?;
        if required_text(observation, "state_id")? != state_id {
            return Err(format!(
                "coefficient {id} mixes material states through supporting observable {observable_id}"
            ));
        }
        supporting.push(*observation);
    }

    if status == "not_identifiable" || status == "not_applicable" {
        if !value.is_null() || value_kind != "null" {
            return Err(format!(
                "coefficient {id}: {status} must not carry a numeric value"
            ));
        }
        let reason = required_text(coefficient, "non_identifiability_reason")?;
        if status == "not_applicable" && reason.is_empty() {
            return Err(format!("coefficient {id}: not_applicable requires a reason"));
        }
        return Ok(());
    }

    if value.is_null() || value_kind == "null" {
        return Err(format!("coefficient {id}: {status} requires an explicit value"));
    }
    if matches!(status, "derived" | "constrained" | "phenomenological_fit") {
        if required_text(coefficient, "unit")? == "not specified" {
            return Err(format!("coefficient {id}: exportable coefficient requires an explicit unit"));
        }
        if status == "derived"
            && required_text(coefficient, "basis_normalization")? == "not specified"
        {
            return Err(format!("derived coefficient {id} requires an explicit basis normalization"));
        }
        let uncertainty = coefficient
            .get("uncertainty_or_covariance")
            .ok_or_else(|| format!("coefficient {id} missing uncertainty_or_covariance"))?;
        if !uncertainty_is_present_and_finite(
            uncertainty,
            &format!("coefficient {id} uncertainty_or_covariance"),
        )? {
            return Err(format!(
                "coefficient {id}: exportable status {status} requires uncertainty or documented bounds"
            ));
        }
    }
    if value_kind == "scalar" {
        finite_number(value, &format!("coefficient {id} value"))?;
    } else if value_kind == "vector" || value_kind == "matrix" {
        numeric_array(value, &format!("coefficient {id} value"))?;
    } else {
        return Err(format!("coefficient {id}: unsupported value_kind {value_kind}"));
    }

    if matches!(status, "derived" | "constrained" | "phenomenological_fit")
        && supporting.is_empty()
    {
        return Err(format!(
            "exportable coefficient {id} requires supporting observations from the same material state"
        ));
    }
    if status == "phenomenological_fit" {
        if model_family != "experimental_effective"
            || !supporting.iter().any(|observation| {
                observation["evidence_status"] == "experimental_measurement"
            })
        {
            return Err(format!(
                "phenomenological coefficient {id} requires experimental evidence and experimental_effective model family"
            ));
        }
    }

    if kind == "a_temperature_slope" && status == "derived" {
        let supported = supporting.iter().any(|observation| {
            observation["observable_kind"] == "quadratic_pair_susceptibility"
        });
        if !supported || !derivation_method.contains("quadratic-free-energy-expansion") {
            return Err(format!(
                "derived a(T) slope {id} requires quadratic pair-susceptibility/free-energy expansion evidence"
            ));
        }
    }
    if kind == "b_quartic" && status == "derived" {
        let supported = supporting.iter().any(|observation| {
            observation["observable_kind"] == "quartic_free_energy_response"
        });
        if !supported || !derivation_method.contains("quartic-free-energy-expansion") {
            return Err(format!(
                "derived quartic coefficient {id} requires quartic free-energy response"
            ));
        }
    }
    if kind == "K_gradient_tensor" && status == "derived" {
        let supported = supporting.iter().any(|observation| {
            observation["observable_kind"] == "finite_q_pair_susceptibility"
        });
        if calculation_family != "scdft"
            && mechanism_scope != "correlated_unconventional"
        {
            return Err(format!(
                "derived K coefficient {id} must use a declared finite-q microscopic response family"
            ));
        }
        if !supported
            || !derivation_method.contains("q-gradient-expansion")
            || required_text(coefficient, "basis_normalization")? == "not specified"
        {
            return Err(format!(
                "derived K coefficient {id} requires finite-q response, q-gradient expansion and basis normalization"
            ));
        }
    }
    if calculation_family == "scdft"
        && mechanism_scope == "conventional_electron_phonon"
        && model_family == "correlated_unconventional"
    {
        return Err(format!(
            "coefficient {id} cannot promote a conventional SCDFT result as a correlated-unconventional GL coefficient"
        ));
    }
    if mechanism_scope == "correlated_unconventional"
        && model_family == "conventional_electron_phonon"
    {
        return Err(format!(
            "coefficient {id} cannot promote a correlated-unconventional result as a conventional electron-phonon GL coefficient"
        ));
    }
    Ok(())
}

fn audit_bundle(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator =
        validator_for(&schema).map_err(|error| format!("schema compilation failed: {error}"))?;
    if let Some(error) = validator.iter_errors(&fixture).next() {
        return Err(format!(
            "microscopic-GL schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }
    if fixture["contract"] != "SC-MICROSCOPIC-GL-TRANSLATION-0.1" {
        return Err("unsupported microscopic-GL contract".to_owned());
    }
    if fixture["fixture_only"].as_bool() != Some(true) {
        return Err("initial microscopic-GL auditor accepts synthetic/source metadata fixtures only".to_owned());
    }

    let source = fixture
        .get("source_record")
        .ok_or_else(|| "missing source_record".to_owned())?;
    let source_evidence_status = required_text(source, "evidence_status")?;
    let calculation = fixture
        .get("microscopic_method")
        .ok_or_else(|| "missing microscopic_method".to_owned())?;
    let calculation_family = required_text(calculation, "calculation_family")?;
    let mechanism_scope = required_text(calculation, "mechanism_scope")?;
    let includes_finite_q = calculation
        .get("includes_finite_q_pair_response")
        .and_then(Value::as_bool)
        .ok_or_else(|| "includes_finite_q_pair_response must be boolean".to_owned())?;

    let states = fixture
        .get("material_states")
        .and_then(Value::as_array)
        .ok_or_else(|| "material_states must be an array".to_owned())?;
    let mut state_ids = HashSet::new();
    let mut material_state_summaries = Vec::with_capacity(states.len());
    for state in states {
        let state_id = required_text(state, "state_id")?;
        if !state_ids.insert(state_id.to_owned()) {
            return Err(format!("duplicate material state ID {state_id}"));
        }
        let material_id = required_text(state, "material_id")?;
        let phase_id = required_text(state, "phase_id")?;
        let structure_id = required_text(state, "structure_id")?;
        material_state_summaries.push(json!({
            "state_id": state_id,
            "material_id": material_id,
            "phase_id": phase_id,
            "structure_id": structure_id
        }));
    }

    let observable_values = fixture
        .get("derived_observables")
        .and_then(Value::as_array)
        .ok_or_else(|| "derived_observables must be an array".to_owned())?;
    let mut observable_ids = HashSet::new();
    let mut observables = HashMap::with_capacity(observable_values.len());
    let mut observable_kind_counts: HashMap<String, usize> = HashMap::new();
    for observable in observable_values {
        let id = required_text(observable, "observable_id")?;
        if !observable_ids.insert(id.to_owned()) {
            return Err(format!("duplicate observable ID {id}"));
        }
        let state_id = required_text(observable, "state_id")?;
        if !state_ids.contains(state_id) {
            return Err(format!("observable {id} references unknown material state {state_id}"));
        }
        validate_observable_value(observable)?;
        let kind = required_text(observable, "observable_kind")?.to_owned();
        *observable_kind_counts.entry(kind).or_insert(0) += 1;
        observables.insert(id.to_owned(), observable);
    }

    let coefficient_values = fixture
        .get("effective_gl_coefficients")
        .and_then(Value::as_array)
        .ok_or_else(|| "effective_gl_coefficients must be an array".to_owned())?;
    let mut coefficient_ids = HashSet::new();
    let mut coefficient_status_counts: HashMap<String, usize> = HashMap::new();
    for coefficient in coefficient_values {
        let id = required_text(coefficient, "coefficient_id")?;
        if !coefficient_ids.insert(id.to_owned()) {
            return Err(format!("duplicate GL coefficient ID {id}"));
        }
        validate_coefficient(
            coefficient,
            &observables,
            &state_ids,
            calculation_family,
            mechanism_scope,
        )?;
        let status = required_text(coefficient, "status")?.to_owned();
        *coefficient_status_counts.entry(status).or_insert(0) += 1;
    }

    if source_evidence_status == "paper_reported"
        && fixture["fixture_only"].as_bool() == Some(true)
        && observable_values.iter().any(|observable| {
            observable["evidence_status"] == "independently_reproduced"
        })
    {
        return Err(
            "paper_reported method metadata fixture cannot label embedded results independently_reproduced"
                .to_owned(),
        );
    }
    if includes_finite_q && calculation_family == "scdft" && mechanism_scope != "conventional_electron_phonon" {
        return Err(
            "the selected SCDFT method scope must be declared conventional electron-phonon in this initial bridge"
                .to_owned(),
        );
    }

    let source_title = required_text(source, "title")?;
    let bundle_id = required_text(&fixture, "bundle_id")?;
    let exportable = exportable_coefficients(coefficient_values);
    let basis_rescaling = scalar_gl_basis_rescaling(-1.0, 2.0, 0.5, 3.0)?;
    Ok(json!({
        "status": "pass",
        "contract": "SC-MICROSCOPIC-GL-AUDIT-0.1",
        "bundle_id": bundle_id,
        "schema_sha256": format!("{:x}", Sha256::digest(schema_bytes)),
        "fixture_sha256": format!("{:x}", Sha256::digest(fixture_bytes)),
        "source_record": {
            "title": source_title,
            "doi": required_text(source, "doi")?,
            "record_kind": required_text(source, "record_kind")?,
            "evidence_status": source_evidence_status
        },
        "microscopic_method": {
            "calculation_family": calculation_family,
            "mechanism_scope": mechanism_scope,
            "includes_finite_q_pair_response": includes_finite_q
        },
        "material_states": material_state_summaries,
        "observable_count": observable_values.len(),
        "observable_kind_counts": observable_kind_counts,
        "gl_coefficient_count": coefficient_values.len(),
        "coefficient_status_counts": coefficient_status_counts,
        "exportable_gl_coefficients": exportable,
        "exportable_coefficient_count": exportable.len(),
        "basis_normalization_identifiability_demo": basis_rescaling,
        "translation_policy": {
            "xi0_and_xi_GL_are_distinct_observable_kinds": true,
            "scalar_lambda_requires_explicit_isotropy": true,
            "derived_K_requires_finite_q_response_and_basis_normalization": true,
            "derived_b_requires_quartic_response": true,
            "cross_state_coefficient_support_is_rejected": true,
            "unidentified_coefficients_are_not_exportable_as_numbers": true
        },
        "claim_ceiling": "Microscopic-to-effective evidence audit only. This receipt does not independently reproduce the cited calculations, derive unprovided coefficients, or establish a material Tc."
    }))
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <schema.json> --fixture <microscopic-gl-bundle.json> [--out <audit-receipt.json>]"
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut schema_path = None;
    let mut fixture_path = None;
    let mut output_path = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--fixture" | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(path),
                    "--fixture" => fixture_path = Some(path),
                    "--out" => output_path = Some(path),
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => {
                usage(&args[0]);
                return;
            }
            unknown => {
                eprintln!("unknown argument: {unknown}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }
    let (Some(schema_path), Some(fixture_path)) = (schema_path, fixture_path) else {
        usage(&args[0]);
        process::exit(2);
    };
    let receipt = audit_bundle(&read(&schema_path), &read(&fixture_path))
        .unwrap_or_else(|error| {
            eprintln!("SC-MICROSCOPIC-GL-FAIL: {error}");
            process::exit(1);
        });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing microscopic-GL receipt should not fail");
    if let Some(path) = output_path {
        if let Err(error) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write {}: {error}", path.display());
            process::exit(1);
        }
    } else {
        println!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn inputs() -> (Vec<u8>, Value) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_microscopic_gl_translation_v0_1.schema.json"),
        )
        .unwrap();
        let fixture: Value = serde_json::from_slice(
            &fs::read(root.join("data/research/superconductivity_microscopic_gl_translation_fixture_v0_1.json"))
                .unwrap(),
        )
        .unwrap();
        (schema, fixture)
    }

    fn audit_mutated(fixture: Value) -> Result<Value, String> {
        let (schema, _) = inputs();
        audit_bundle(&schema, &serde_json::to_vec(&fixture).unwrap())
    }

    #[test]
    fn metadata_only_source_record_passes_without_fabricated_values() {
        let (_, fixture) = inputs();
        let receipt = audit_mutated(fixture).unwrap();
        assert_eq!(receipt["status"], "pass");
        assert_eq!(receipt["observable_count"], 0);
        assert_eq!(receipt["coefficient_status_counts"]["not_identifiable"], 3);
        assert_eq!(receipt["exportable_coefficient_count"], 0);
        assert!(
            receipt["basis_normalization_identifiability_demo"]["maximum_invariant_residual"]
                .as_f64()
                .unwrap()
                < 1.0e-12
        );
    }

    #[test]
    fn xi0_cannot_be_relabelled_as_gl_coherence_length() {
        let (_, mut fixture) = inputs();
        fixture["derived_observables"] = json!([{
            "observable_id":"obs-xi-0",
            "state_id":"method_summary_only",
            "observable_kind":"xi0_microscopic",
            "length_scale_definition":"GL_near_Tc_xi",
            "value_kind":"scalar","value":10.0,"unit":"angstrom",
            "tensor_axes":[],
            "isotropy_assumption":"not_applicable",
            "uncertainty":0.1,"evidence_status":"paper_reported",
            "measurement_or_derivation_method":"microscopic nonlocal pair response",
            "source_artifact_digest":null,
            "source_reference":"https://doi.org/10.1103/lrhl-j9bt"
        }]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn xi0_and_xi_gl_are_distinct_typed_observables() {
        let (_, mut fixture) = inputs();
        fixture["derived_observables"] = json!([{
            "observable_id":"obs-xi-gl",
            "state_id":"method_summary_only",
            "observable_kind":"xi_GL",
            "length_scale_definition":"microscopic_xi0",
            "value_kind":"scalar","value":10.0,"unit":"angstrom",
            "tensor_axes":[],"isotropy_assumption":"not_applicable",
            "uncertainty":0.1,"evidence_status":"paper_reported",
            "measurement_or_derivation_method":"near-Tc GL fit",
            "source_artifact_digest":null,
            "source_reference":"https://doi.org/10.1103/lrhl-j9bt"
        }]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn non_finite_or_negative_observable_uncertainty_is_rejected() {
        let (_, mut fixture) = inputs();
        fixture["derived_observables"] = json!([{
            "observable_id":"obs-tc",
            "state_id":"method_summary_only",
            "observable_kind":"Tc",
            "value_kind":"scalar","value":8.0,"unit":"K",
            "tensor_axes":[],"isotropy_assumption":"not_applicable",
            "uncertainty":-0.5,"evidence_status":"paper_reported",
            "measurement_or_derivation_method":"source report",
            "source_artifact_digest":null,
            "source_reference":"https://doi.org/10.1103/lrhl-j9bt"
        }]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn scalar_penetration_depth_requires_explicit_isotropy() {
        let (_, mut fixture) = inputs();
        fixture["derived_observables"] = json!([{
            "observable_id":"lambda-scalar",
            "state_id":"method_summary_only",
            "observable_kind":"lambda_L_scalar",
            "value_kind":"scalar","value":100.0,"unit":"nm",
            "tensor_axes":[],
            "isotropy_assumption":"not_specified",
            "uncertainty":1.0,"evidence_status":"paper_reported",
            "measurement_or_derivation_method":"calculated",
            "source_artifact_digest":null,
            "source_reference":"https://doi.org/10.1103/lrhl-j9bt"
        }]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn lambda_tensor_requires_anisotropic_matrix_and_axes() {
        let (_, mut fixture) = inputs();
        fixture["derived_observables"] = json!([{
            "observable_id":"lambda-matrix",
            "state_id":"method_summary_only",
            "observable_kind":"lambda_L_tensor",
            "value_kind":"matrix","value":[[100.0,0.0],[0.0,120.0]],"unit":"nm",
            "tensor_axes":["a","b"],
            "isotropy_assumption":"anisotropic_tensor",
            "uncertainty":null,"evidence_status":"paper_reported",
            "measurement_or_derivation_method":"calculated tensor",
            "source_artifact_digest":null,
            "source_reference":"https://doi.org/10.1103/lrhl-j9bt"
        }]);
        assert!(audit_mutated(fixture).is_ok());
    }

    #[test]
    fn identified_coefficient_is_exported_with_basis_and_uncertainty() {
        let (_, mut fixture) = inputs();
        fixture["source_record"]["evidence_status"] = json!("independently_reproduced");
        fixture["derived_observables"] = json!([{
            "observable_id":"pair-susc",
            "state_id":"method_summary_only",
            "observable_kind":"quadratic_pair_susceptibility",
            "value_kind":"scalar","value":1.0,"unit":"arb",
            "tensor_axes":[],"isotropy_assumption":"not_applicable",
            "uncertainty":0.1,"evidence_status":"independently_reproduced",
            "measurement_or_derivation_method":"independent quadratic response calculation",
            "source_artifact_digest":null,
            "source_reference":"https://doi.org/10.1103/lrhl-j9bt",
            "length_scale_definition":"not_applicable"
        }]);
        let coefficient = &mut fixture["effective_gl_coefficients"][0];
        coefficient["status"] = json!("derived");
        coefficient["value_kind"] = json!("scalar");
        coefficient["value"] = json!(1.0);
        coefficient["unit"] = json!("dimensionless_per_kelvin");
        coefficient["basis_normalization"] = json!("psi_normalized_to_pair_susceptibility_v1");
        coefficient["uncertainty_or_covariance"] = json!(0.05);
        coefficient["supporting_observable_ids"] = json!(["pair-susc"]);
        coefficient["derivation_method"] = json!("quadratic-free-energy-expansion");
        let receipt = audit_mutated(fixture).unwrap();
        assert_eq!(receipt["exportable_coefficient_count"], 1);
        assert_eq!(
            receipt["exportable_gl_coefficients"][0]["coefficient_id"],
            "scalar_GL_a_slope"
        );
        assert_eq!(
            receipt["exportable_gl_coefficients"][0]["basis_normalization"],
            "psi_normalized_to_pair_susceptibility_v1"
        );
    }

    #[test]
    fn cross_mechanism_coefficient_promotion_is_rejected() {
        let (_, mut fixture) = inputs();
        fixture["derived_observables"] = json!([{
            "observable_id":"pair-susc",
            "state_id":"method_summary_only",
            "observable_kind":"quadratic_pair_susceptibility",
            "value_kind":"scalar","value":1.0,"unit":"arb",
            "tensor_axes":[],"isotropy_assumption":"not_applicable",
            "uncertainty":0.1,"evidence_status":"paper_reported",
            "measurement_or_derivation_method":"source response",
            "source_artifact_digest":null,
            "source_reference":"https://doi.org/10.1103/lrhl-j9bt",
            "length_scale_definition":"not_applicable"
        }]);
        let coefficient = &mut fixture["effective_gl_coefficients"][0];
        coefficient["status"] = json!("derived");
        coefficient["value_kind"] = json!("scalar");
        coefficient["value"] = json!(1.0);
        coefficient["unit"] = json!("arb");
        coefficient["basis_normalization"] = json!("psi-v1");
        coefficient["uncertainty_or_covariance"] = json!(0.1);
        coefficient["model_family"] = json!("correlated_unconventional");
        coefficient["supporting_observable_ids"] = json!(["pair-susc"]);
        coefficient["derivation_method"] = json!("quadratic-free-energy-expansion");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn derived_quartic_b_requires_quartic_response() {
        let (_, mut fixture) = inputs();
        fixture["effective_gl_coefficients"][1]["status"] = json!("derived");
        fixture["effective_gl_coefficients"][1]["value_kind"] = json!("scalar");
        fixture["effective_gl_coefficients"][1]["value"] = json!(1.0);
        fixture["effective_gl_coefficients"][1]["supporting_observable_ids"] = json!([]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn derived_gradient_K_requires_finite_q_evidence_and_expansion() {
        let (_, mut fixture) = inputs();
        fixture["effective_gl_coefficients"][2]["status"] = json!("derived");
        fixture["effective_gl_coefficients"][2]["value_kind"] = json!("matrix");
        fixture["effective_gl_coefficients"][2]["value"] = json!([[1.0,0.0],[0.0,1.0]]);
        fixture["effective_gl_coefficients"][2]["supporting_observable_ids"] = json!([]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn coefficient_cannot_join_observations_from_another_material_state() {
        let (_, mut fixture) = inputs();
        let second_state = fixture["material_states"][0].clone();
        fixture["material_states"].as_array_mut().unwrap().push(second_state);
        fixture["material_states"][1]["state_id"] = json!("second-pressure-state");
        fixture["derived_observables"] = json!([{
            "observable_id":"xi-at-state-2",
            "state_id":"second-pressure-state",
            "observable_kind":"xi0_microscopic",
            "length_scale_definition":"microscopic_xi0",
            "value_kind":"scalar","value":20.0,"unit":"angstrom",
            "tensor_axes":[],"isotropy_assumption":"not_applicable",
            "uncertainty":1.0,"evidence_status":"paper_reported",
            "measurement_or_derivation_method":"microscopic response",
            "source_artifact_digest":null,
            "source_reference":"https://doi.org/10.1103/lrhl-j9bt"
        }]);
        fixture["effective_gl_coefficients"][0]["supporting_observable_ids"] =
            json!(["xi-at-state-2"]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn paper_reported_metadata_fixture_cannot_claim_independent_reproduction() {
        let (_, mut fixture) = inputs();
        fixture["derived_observables"] = json!([{
            "observable_id":"made-up-reproduced",
            "state_id":"method_summary_only",
            "observable_kind":"Tc",
            "value_kind":"scalar","value":300.0,"unit":"K",
            "tensor_axes":[],"isotropy_assumption":"not_applicable",
            "uncertainty":null,"evidence_status":"independently_reproduced",
            "measurement_or_derivation_method":"not run",
            "source_artifact_digest":null,
            "source_reference":"https://doi.org/10.1103/lrhl-j9bt"
        }]);
        assert!(audit_mutated(fixture).is_err());
    }
}
