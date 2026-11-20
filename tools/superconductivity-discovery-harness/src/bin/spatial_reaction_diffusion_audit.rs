use jsonschema::validator_for;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Copy, Debug)]
struct TransportMatrix {
    d_aa: f64,
    d_ab: f64,
    d_bb: f64,
}
#[derive(Clone, Debug)]
struct Case {
    id: String,
    length: f64,
    cells: usize,
    duration: f64,
    time_step: f64,
    reservoir_a: f64,
    reservoir_b: f64,
    initial_a: f64,
    initial_b: f64,
    transport: TransportMatrix,
    target_rate: f64,
    competitor_rate: f64,
    thermodynamic_preference: String,
}
#[derive(Clone, Copy, Debug)]
struct Balance {
    initial: f64,
    boundary_influx: f64,
    final_inventory: f64,
    target_consumption: f64,
    competitor_consumption: f64,
    residual: f64,
    relative_residual: f64,
}
#[derive(Clone, Debug)]
struct ResultCase {
    dt_limit_diffusion: f64,
    dt_limit_reaction: f64,
    steps: usize,
    dx: f64,
    final_a: Vec<f64>,
    final_b: Vec<f64>,
    target_product: f64,
    competitor_product: f64,
    influx_a: f64,
    influx_b: f64,
    balance_a: Balance,
    balance_b: Balance,
    min_a: f64,
    min_b: f64,
    max_a: f64,
    max_b: f64,
    transport_eigen_min: f64,
    transport_eigen_max: f64,
}

fn text_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str).filter(|x| !x.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}"))
}
fn number_field(value: &Value, key: &str) -> Result<f64, String> {
    let x = value.get(key).and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric field {key}"))?;
    if !x.is_finite() { return Err(format!("{key} must be finite")); }
    Ok(x)
}
fn balance(initial: f64, influx: f64, final_inventory: f64, target: f64, competitor: f64) -> Balance {
    let residual = initial + influx - final_inventory - target - competitor;
    let scale = (initial.abs() + influx.abs()).max(1.0e-30);
    Balance {
        initial,
        boundary_influx: influx,
        final_inventory,
        target_consumption: target,
        competitor_consumption: competitor,
        residual,
        relative_residual: residual.abs() / scale,
    }
}
fn min_max(values: &[f64]) -> (f64, f64) {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (minimum, maximum)
}
fn eigenvalues(matrix: TransportMatrix) -> (f64, f64, f64) {
    let determinant = matrix.d_aa * matrix.d_bb - matrix.d_ab * matrix.d_ab;
    let trace = matrix.d_aa + matrix.d_bb;
    let discriminant = ((matrix.d_aa - matrix.d_bb).powi(2) + 4.0 * matrix.d_ab.powi(2)).sqrt();
    ((trace - discriminant) / 2.0, (trace + discriminant) / 2.0, determinant)
}
fn parse_case(value: &Value) -> Result<Case, String> {
    let matrix = value.get("transport_matrix_m2_s").ok_or("case missing transport_matrix_m2_s")?;
    let transport = TransportMatrix {
        d_aa: number_field(matrix, "d_aa")?,
        d_ab: number_field(matrix, "d_ab")?,
        d_bb: number_field(matrix, "d_bb")?,
    };
    let (lambda_min, _, determinant) = eigenvalues(transport);
    if determinant <= 0.0 || lambda_min <= 0.0 {
        return Err(format!("case {} transport matrix is not positive definite", text_field(value, "case_id")?));
    }
    let case = Case {
        id: text_field(value, "case_id")?.to_owned(),
        length: number_field(value, "length_m")?,
        cells: value.get("grid_cells").and_then(Value::as_u64).ok_or("grid_cells must be integer")? as usize,
        duration: number_field(value, "duration_s")?,
        time_step: number_field(value, "time_step_s")?,
        reservoir_a: number_field(value, "reservoir_a")?,
        reservoir_b: number_field(value, "reservoir_b")?,
        initial_a: number_field(value, "initial_a")?,
        initial_b: number_field(value, "initial_b")?,
        transport,
        target_rate: number_field(value, "target_rate_s_inverse")?,
        competitor_rate: number_field(value, "competitor_rate_s_inverse")?,
        thermodynamic_preference: text_field(value, "thermodynamic_preference")?.to_owned(),
    };
    if case.length <= 0.0 || case.cells < 4 || case.cells > 512
        || case.duration <= 0.0 || case.time_step <= 0.0
        || case.target_rate <= 0.0 || case.competitor_rate <= 0.0
        || [case.reservoir_a, case.reservoir_b, case.initial_a, case.initial_b]
            .iter().any(|x| *x < 0.0 || *x > 2.0) {
        return Err(format!("case {} has invalid length, grid, time, rate or concentration", case.id));
    }
    Ok(case)
}
fn explicit_stability_limits(case: &Case) -> (f64, f64, f64, f64, f64) {
    let dx = case.length / case.cells as f64;
    let (lambda_min, lambda_max, determinant) = eigenvalues(case.transport);
    let diffusion_limit = 0.20 * dx * dx / lambda_max;
    let maximum_reaction_rate = case.target_rate * 2.0 + case.competitor_rate;
    let reaction_limit = 0.20 / maximum_reaction_rate;
    (dx, diffusion_limit, reaction_limit, lambda_min, determinant)
}
fn fluxes(
    a: &[f64],
    b: &[f64],
    case: &Case,
    dx: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = case.cells;
    let mut flux_a = vec![0.0; n + 1];
    let mut flux_b = vec![0.0; n + 1];
    // Left boundary: fixed reservoir concentration, half-cell distance.
    flux_a[0] = -2.0 / dx * (
        case.transport.d_aa * (a[0] - case.reservoir_a)
            + case.transport.d_ab * (b[0] - case.reservoir_b)
    );
    flux_b[0] = -2.0 / dx * (
        case.transport.d_ab * (a[0] - case.reservoir_a)
            + case.transport.d_bb * (b[0] - case.reservoir_b)
    );
    // Internal faces: each face flux is calculated once and shared by both cells.
    for face in 1..n {
        let grad_a = (a[face] - a[face - 1]) / dx;
        let grad_b = (b[face] - b[face - 1]) / dx;
        flux_a[face] = -(case.transport.d_aa * grad_a + case.transport.d_ab * grad_b);
        flux_b[face] = -(case.transport.d_ab * grad_a + case.transport.d_bb * grad_b);
    }
    // Right boundary: no-flux.
    flux_a[n] = 0.0;
    flux_b[n] = 0.0;
    (flux_a, flux_b)
}
fn inventory(profile: &[f64], dx: f64) -> f64 {
    profile.iter().sum::<f64>() * dx
}
fn simulate(case: &Case) -> Result<ResultCase, String> {
    let (dx, diffusion_limit, reaction_limit, lambda_min, determinant) =
        explicit_stability_limits(case);
    let (_, lambda_max, _) = eigenvalues(case.transport);
    let stability_limit = diffusion_limit.min(reaction_limit);
    if case.time_step > stability_limit * (1.0 + 1.0e-12) {
        return Err(format!(
            "case {} time step {} exceeds explicit stability limit {} (diffusion {}, reaction {})",
            case.id, case.time_step, stability_limit, diffusion_limit, reaction_limit
        ));
    }
    let ratio = case.duration / case.time_step;
    let steps = ratio.round();
    if (ratio - steps).abs() > 1.0e-8 || steps < 1.0 || steps > 2_000_000.0 {
        return Err(format!("case {} duration/time_step must be an integer in [1,2000000]", case.id));
    }
    let steps = steps as usize;
    let n = case.cells;
    let mut a = vec![case.initial_a; n];
    let mut b = vec![case.initial_b; n];
    let initial_a = inventory(&a, dx);
    let initial_b = inventory(&b, dx);
    let mut target_product = 0.0;
    let mut competitor_product = 0.0;
    let mut influx_a = 0.0;
    let mut influx_b = 0.0;
    let dt = case.time_step;
    for step_index in 0..steps {
        let maximum_current_reaction_rate = (0..n).fold(0.0_f64, |maximum, i| {
            maximum
                .max(case.target_rate * b[i] + case.competitor_rate)
                .max(case.target_rate * a[i])
        });
        let dynamic_reaction_limit = if maximum_current_reaction_rate > 0.0 {
            0.20 / maximum_current_reaction_rate
        } else {
            f64::INFINITY
        };
        if dt > dynamic_reaction_limit * (1.0 + 1.0e-12) {
            return Err(format!(
                "case {} time step {} exceeds dynamic reaction stability limit {} at step {}",
                case.id, dt, dynamic_reaction_limit, step_index
            ));
        }
        let (ja, jb) = fluxes(&a, &b, case, dx);
        influx_a += ja[0] * dt;
        influx_b += jb[0] * dt;
        let mut next_a = vec![0.0; n];
        let mut next_b = vec![0.0; n];
        for i in 0..n {
            let reaction_target = case.target_rate * a[i] * b[i];
            let reaction_competitor = case.competitor_rate * a[i];
            target_product += reaction_target * dx * dt;
            competitor_product += reaction_competitor * dx * dt;
            let diffusion_a = -(ja[i + 1] - ja[i]) / dx;
            let diffusion_b = -(jb[i + 1] - jb[i]) / dx;
            next_a[i] = a[i] + dt * (diffusion_a - reaction_target - reaction_competitor);
            next_b[i] = b[i] + dt * (diffusion_b - reaction_target);
            if next_a[i] < -1.0e-12 || next_b[i] < -1.0e-12 {
                return Err(format!(
                    "case {} generated negative concentration at cell {}: A={}, B={}",
                    case.id, i, next_a[i], next_b[i]
                ));
            }
            // Only remove floating-point dust inside the declared tolerance.
            if next_a[i] < 0.0 { next_a[i] = 0.0; }
            if next_b[i] < 0.0 { next_b[i] = 0.0; }
        }
        a = next_a;
        b = next_b;
    }
    let final_a = inventory(&a, dx);
    let final_b = inventory(&b, dx);
    let balance_a = balance(initial_a, influx_a, final_a, target_product, competitor_product);
    let balance_b = balance(initial_b, influx_b, final_b, target_product, 0.0);
    if initial_a + influx_a <= 0.0 || initial_b + influx_b <= 0.0 {
        return Err(format!("case {} has no positive available A or B inventory", case.id));
    }
    if balance_a.relative_residual > 1.0e-8 || balance_b.relative_residual > 1.0e-8 {
        return Err(format!(
            "case {} conservation residual exceeds tolerance: A={}, B={}",
            case.id, balance_a.relative_residual, balance_b.relative_residual
        ));
    }
    let (min_a, max_a) = min_max(&a);
    let (min_b, max_b) = min_max(&b);
    if min_a < 0.0 || min_b < 0.0 {
        return Err(format!("case {} profile has a negative final concentration", case.id));
    }
    Ok(ResultCase {
        dt_limit_diffusion: diffusion_limit,
        dt_limit_reaction: reaction_limit,
        steps,
        dx,
        final_a: a,
        final_b: b,
        target_product,
        competitor_product,
        influx_a,
        influx_b,
        balance_a,
        balance_b,
        min_a,
        min_b,
        max_a,
        max_b,
        transport_eigen_min: lambda_min,
        transport_eigen_max: lambda_max,
    })
}
fn case_receipt(case: &Case, result: &ResultCase) -> Value {
    let total_a_consumed = result.target_product + result.competitor_product;
    let available_a = result.balance_a.initial + result.balance_a.boundary_influx;
    let available_b = result.balance_b.initial + result.balance_b.boundary_influx;
    let phase_denominator_a = available_a.max(1.0e-30);
    let phase_denominator_b = available_b.max(1.0e-30);
    json!({
        "case_id": case.id,
        "thermodynamic_preference": case.thermodynamic_preference,
        "domain": {
            "length_m": case.length,
            "grid_cells": case.cells,
            "cell_width_m": result.dx,
            "duration_s": case.duration,
            "time_step_s": case.time_step,
            "steps": result.steps
        },
        "transport": {
            "d_aa_m2_s": case.transport.d_aa,
            "d_ab_m2_s": case.transport.d_ab,
            "d_bb_m2_s": case.transport.d_bb,
            "minimum_eigenvalue_m2_s": result.transport_eigen_min,
            "maximum_eigenvalue_m2_s": result.transport_eigen_max,
            "positive_definite": true,
            "explicit_diffusion_time_step_limit_s": result.dt_limit_diffusion,
            "explicit_reaction_time_step_limit_s": result.dt_limit_reaction
        },
        "boundary_influx_per_area": {
            "a": result.influx_a,
            "b": result.influx_b
        },
        "phase_products_per_area": {
            "target": result.target_product,
            "competitor": result.competitor_product,
            "target_fraction_of_a_available": result.target_product / phase_denominator_a,
            "competitor_fraction_of_a_available": result.competitor_product / phase_denominator_a,
            "unreacted_a_fraction_of_a_available": result.balance_a.final_inventory / phase_denominator_a,
            "target_fraction_of_b_available": result.target_product / phase_denominator_b,
            "unreacted_b_fraction_of_b_available": result.balance_b.final_inventory / phase_denominator_b,
            "converted_a_fraction_of_a_available": total_a_consumed / phase_denominator_a
        },
        "final_mobile_inventory_per_area": {
            "a": result.balance_a.final_inventory,
            "b": result.balance_b.final_inventory
        },
        "conservation": {
            "a": {
                "initial": result.balance_a.initial,
                "boundary_influx": result.balance_a.boundary_influx,
                "final_inventory": result.balance_a.final_inventory,
                "target_consumption": result.balance_a.target_consumption,
                "competitor_consumption": result.balance_a.competitor_consumption,
                "residual": result.balance_a.residual,
                "relative_residual": result.balance_a.relative_residual
            },
            "b": {
                "initial": result.balance_b.initial,
                "boundary_influx": result.balance_b.boundary_influx,
                "final_inventory": result.balance_b.final_inventory,
                "target_consumption": result.balance_b.target_consumption,
                "competitor_consumption": result.balance_b.competitor_consumption,
                "residual": result.balance_b.residual,
                "relative_residual": result.balance_b.relative_residual
            }
        },
        "concentration_profile": {
            "a": result.final_a,
            "b": result.final_b,
            "minimum_a": result.min_a,
            "maximum_a": result.max_a,
            "minimum_b": result.min_b,
            "maximum_b": result.max_b
        },
        "interpretation_ceiling": "Synthetic conservative finite-volume result for the declared 1D model; not measured phase yield or a spatially complete synthesis prediction."
    })
}
fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes).map_err(|e| format!("schema JSON parse failed: {e}"))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes).map_err(|e| format!("fixture JSON parse failed: {e}"))?;
    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator = validator_for(&schema).map_err(|e| format!("schema compilation failed: {e}"))?;
    if let Some(error) = validator.iter_errors(&fixture).next() {
        return Err(format!("spatial reaction-diffusion schema validation failed: {error} at {}", error.instance_path()));
    }
    if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
        return Err("spatial reaction-diffusion auditor accepts synthetic fixtures only".to_owned());
    }
    if text_field(&fixture, "contract")? != "SC-SPATIAL-REACTION-DIFFUSION-0.1" {
        return Err("unsupported spatial reaction-diffusion contract".to_owned());
    }
    let cases = fixture.get("cases").and_then(Value::as_array).ok_or("cases must be array")?;
    let mut seen = std::collections::HashSet::new();
    let mut receipts = Vec::new();
    for row in cases {
        let case = parse_case(row)?;
        if !seen.insert(case.id.clone()) { return Err(format!("duplicate spatial case {}", case.id)); }
        let result = simulate(&case)?;
        receipts.push(case_receipt(&case, &result));
    }
    Ok(json!({
        "status": "synthetic_spatial_reaction_diffusion_pass",
        "contract": "SC-SPATIAL-REACTION-DIFFUSION-AUDIT-0.1",
        "benchmark_id": text_field(&fixture, "benchmark_id")?,
        "schema_sha256": format!("{:x}", Sha256::digest(schema_bytes)),
        "fixture_sha256": format!("{:x}", Sha256::digest(fixture_bytes)),
        "concentration_reference": text_field(&fixture, "concentration_reference")?,
        "cases": receipts,
        "claim_ceiling": "Synthetic one-dimensional conservative reaction-diffusion reference only. No real reaction kinetics, measured transport, phase yield, superconductivity or laboratory procedure is established."
    }))
}
fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| {
        eprintln!("could not read {}: {e}", path.display());
        process::exit(1);
    })
}
fn usage(program: &str) {
    eprintln!("Usage: {program} --schema <schema.json> --fixture <fixture.json> [--out <receipt.json>]");
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
                if index + 1 >= args.len() { usage(&args[0]); process::exit(2); }
                let path = PathBuf::from(&args[index+1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(path),
                    "--fixture" => fixture_path = Some(path),
                    "--out" => output_path = Some(path),
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => { usage(&args[0]); return; }
            other => { eprintln!("unknown argument {other}"); usage(&args[0]); process::exit(2); }
        }
    }
    let (Some(schema_path), Some(fixture_path)) = (schema_path, fixture_path) else {
        usage(&args[0]); process::exit(2);
    };
    let output = audit(&read(&schema_path), &read(&fixture_path)).unwrap_or_else(|error| {
        eprintln!("SC-SPATIAL-REACTION-DIFFUSION-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&output).expect("JSON receipt serialization cannot fail");
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

    fn inputs_for(file: &str) -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        (
            fs::read(root.join("data/schemas/superconductivity_spatial_reaction_diffusion_v0_1.schema.json")).unwrap(),
            fs::read(root.join("tools/superconductivity-discovery-harness/fixtures").join(file)).unwrap(),
        )
    }
    fn inputs() -> (Vec<u8>, Vec<u8>) {
        inputs_for("spatial_reaction_diffusion_smoke_v0_1.json")
    }

    #[test]
    fn finite_volume_profiles_remain_nonnegative_and_conserve_a_and_b() {
        let (schema, fixture) = inputs();
        let output = audit(&schema, &fixture).unwrap();
        for case in output["cases"].as_array().unwrap() {
            assert!(case["concentration_profile"]["minimum_a"].as_f64().unwrap() >= 0.0);
            assert!(case["concentration_profile"]["minimum_b"].as_f64().unwrap() >= 0.0);
            assert!(case["conservation"]["a"]["relative_residual"].as_f64().unwrap() < 1.0e-8);
            assert!(case["conservation"]["b"]["relative_residual"].as_f64().unwrap() < 1.0e-8);
        }
    }

    #[test]
    fn cross_diffusion_changes_target_and_competitor_products() {
        let (schema, fixture) = inputs();
        let output = audit(&schema, &fixture).unwrap();
        let cases = output["cases"].as_array().unwrap();
        let negative = cases.iter().find(|x| x["case_id"] == "negative_cross_correlation_long_hold").unwrap();
        let positive = cases.iter().find(|x| x["case_id"] == "positive_cross_correlation_long_hold").unwrap();
        assert_ne!(
            negative["phase_products_per_area"]["target"],
            positive["phase_products_per_area"]["target"]
        );
        assert!(negative["phase_products_per_area"]["target"].as_f64().unwrap()
            < positive["phase_products_per_area"]["target"].as_f64().unwrap());
    }

    #[test]
    fn layer_thickness_changes_transport_limited_phase_products() {
        let (schema, fixture) = inputs();
        let output = audit(&schema, &fixture).unwrap();
        let cases = output["cases"].as_array().unwrap();
        let thin = cases.iter().find(|x| x["case_id"] == "negative_cross_correlation_long_hold").unwrap();
        let thick = cases.iter().find(|x| x["case_id"] == "thicker_product_layer").unwrap();
        assert_ne!(
            thin["phase_products_per_area"]["target"],
            thick["phase_products_per_area"]["target"]
        );
    }

    #[test]
    fn non_positive_definite_transport_is_rejected() {
        let (schema, fixture) = inputs_for("spatial_reaction_diffusion_non_psd_v0_1.json");
        let error = audit(&schema, &fixture).unwrap_err();
        assert!(error.contains("transport matrix is not positive definite"));
    }

    #[test]
    fn unstable_explicit_time_step_is_rejected() {
        let (schema, fixture) = inputs_for("spatial_reaction_diffusion_unstable_step_v0_1.json");
        let error = audit(&schema, &fixture).unwrap_err();
        assert!(error.contains("exceeds explicit stability limit"));
    }

    #[test]
    fn zero_flux_right_boundary_is_conservative() {
        let (schema, fixture) = inputs();
        let output = audit(&schema, &fixture).unwrap();
        for case in output["cases"].as_array().unwrap() {
            assert!(case["conservation"]["a"]["relative_residual"].as_f64().unwrap() < 1.0e-8);
            assert!(case["conservation"]["b"]["relative_residual"].as_f64().unwrap() < 1.0e-8);
        }
    }
}
