use jsonschema::validator_for;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Clone, Copy, Debug)]
struct Interval {
    lower: f64,
    upper: f64,
}
#[derive(Clone, Debug)]
struct TransportSpecies {
    diffusivity_prefactor: f64,
    diffusion_activation: f64,
}
#[derive(Clone, Debug)]
struct Transport {
    species_a: TransportSpecies,
    species_b: TransportSpecies,
    cross_correlation: f64,
}
#[derive(Clone, Debug)]
struct Route {
    id: String,
    mobile_requirement: String,
    thermodynamic_conditions_id: String,
    reaction_extent_basis_id: String,
    standard_state_id: String,
    activation_barrier: f64,
    prefactor: f64,
    delta_g: Interval,
    availability: f64,
}
#[derive(Clone, Debug)]
struct Case {
    id: String,
    thermodynamic_conditions_id: String,
    temperature: f64,
    hold_time: f64,
    layer_thickness: f64,
    transport: Transport,
    target: Route,
    competitor: Route,
    selectivity_margin: f64,
}
#[derive(Clone, Copy, Debug)]
struct PhaseFractions {
    target: f64,
    competitor: f64,
    unreacted: f64,
    total: f64,
}
#[derive(Clone, Debug)]
struct TransportState {
    diffusivity_a: f64,
    diffusivity_b: f64,
    diffusivity_ab: f64,
    determinant: f64,
    eigenvalue_min: f64,
    eigenvalue_max: f64,
    effective_target_diffusivity: f64,
    target_transport_rate: f64,
    competitor_transport_rate: f64,
}
#[derive(Clone, Debug)]
struct RouteKinetics {
    target_reaction_rate: f64,
    competitor_reaction_rate: f64,
}
#[derive(Clone, Debug)]
struct CaseResult {
    transport: TransportState,
    kinetics: RouteKinetics,
    coupled: PhaseFractions,
    diagonal: PhaseFractions,
    reaction_only: PhaseFractions,
    target_transport_time: f64,
    competitor_transport_time: f64,
    target_reaction_time: f64,
    competitor_reaction_time: f64,
    thermodynamic_status: String,
    kinetic_phase_ranking: String,
    conservation_error: f64,
}

fn text_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}"))
}
fn number_field(value: &Value, key: &str) -> Result<f64, String> {
    let x = value.get(key).and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric field {key}"))?;
    if !x.is_finite() { return Err(format!("{key} must be finite")); }
    Ok(x)
}
fn interval(value: &Value, label: &str) -> Result<Interval, String> {
    let lower = number_field(value, "lower")?;
    let upper = number_field(value, "upper")?;
    if lower > upper { return Err(format!("{label} interval lower bound exceeds upper bound")); }
    Ok(Interval { lower, upper })
}
fn exp_rate(prefactor: f64, activation: f64, k_b: f64, temperature: f64) -> f64 {
    prefactor * (-activation / (k_b * temperature)).exp()
}
fn transport_survival(diffusion_rate: f64, reaction_rate: f64, time: f64) -> f64 {
    if time <= 0.0 { return 1.0; }
    let scale = diffusion_rate.max(reaction_rate).max(1.0e-300);
    if (diffusion_rate - reaction_rate).abs() <= 1.0e-10 * scale {
        return ((-diffusion_rate * time).exp() * (1.0 + diffusion_rate * time)).clamp(0.0, 1.0);
    }
    let value = (
        reaction_rate * (-diffusion_rate * time).exp()
            - diffusion_rate * (-reaction_rate * time).exp()
    ) / (reaction_rate - diffusion_rate);
    value.clamp(0.0, 1.0)
}
fn transport_density(diffusion_rate: f64, reaction_rate: f64, time: f64) -> f64 {
    if time <= 0.0 { return 0.0; }
    let scale = diffusion_rate.max(reaction_rate).max(1.0e-300);
    if (diffusion_rate - reaction_rate).abs() <= 1.0e-10 * scale {
        return (diffusion_rate * diffusion_rate * time * (-diffusion_rate * time).exp()).max(0.0);
    }
    let value = diffusion_rate * reaction_rate
        * ((-diffusion_rate * time).exp() - (-reaction_rate * time).exp())
        / (reaction_rate - diffusion_rate);
    value.max(0.0)
}
fn simpson<F: Fn(f64) -> f64>(function: F, end: f64, steps: usize) -> f64 {
    let h = end / steps as f64;
    let mut sum = function(0.0) + function(end);
    for i in 1..steps {
        let x = i as f64 * h;
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * function(x);
    }
    sum * h / 3.0
}
fn competing_route_fractions(
    target_transport_rate: f64,
    target_reaction_rate: f64,
    competitor_transport_rate: f64,
    competitor_reaction_rate: f64,
    hold_time: f64,
    steps: usize,
) -> PhaseFractions {
    let target_density = |t: f64| transport_density(target_transport_rate, target_reaction_rate, t);
    let target_survival = |t: f64| transport_survival(target_transport_rate, target_reaction_rate, t);
    let competitor_density = |t: f64| transport_density(competitor_transport_rate, competitor_reaction_rate, t);
    let competitor_survival = |t: f64| transport_survival(competitor_transport_rate, competitor_reaction_rate, t);
    let target = simpson(|t| target_density(t) * competitor_survival(t), hold_time, steps).clamp(0.0, 1.0);
    let competitor = simpson(|t| competitor_density(t) * target_survival(t), hold_time, steps).clamp(0.0, 1.0);
    let unreacted = (target_survival(hold_time) * competitor_survival(hold_time)).clamp(0.0, 1.0);
    let total = target + competitor + unreacted;
    PhaseFractions { target, competitor, unreacted, total }
}
fn reaction_only_fractions(target_rate: f64, competitor_rate: f64, hold_time: f64) -> PhaseFractions {
    let total_rate = target_rate + competitor_rate;
    let unreacted = (-total_rate * hold_time).exp();
    let converted = 1.0 - unreacted;
    PhaseFractions {
        target: converted * target_rate / total_rate,
        competitor: converted * competitor_rate / total_rate,
        unreacted,
        total: 1.0,
    }
}
fn thermodynamic_status(target: Interval, competitor: Interval, margin: f64) -> &'static str {
    if target.upper + margin < competitor.lower {
        "target_reaction_robustly_favored"
    } else if competitor.upper + margin < target.lower {
        "competitor_reaction_robustly_favored"
    } else {
        "reaction_free_energy_intervals_overlap_or_are_unresolved"
    }
}
fn parse_route(value: &Value, label: &str) -> Result<Route, String> {
    let delta_g = interval(
        value.get("delta_g_interval_eV_per_route_extent")
            .ok_or_else(|| format!("{label} missing ΔG interval"))?,
        &format!("{label} ΔG"),
    )?;
    let activation_barrier = number_field(value, "reaction_activation_barrier_eV")?;
    if activation_barrier < 0.0 { return Err(format!("{label} activation barrier must be non-negative")); }
    let prefactor = number_field(value, "reaction_prefactor_s_inverse")?;
    if prefactor <= 0.0 { return Err(format!("{label} reaction prefactor must be positive")); }
    let availability = number_field(value, "availability_fraction")?;
    if availability <= 0.0 || availability > 1.0 {
        return Err(format!("{label} availability fraction must be in (0,1]"));
    }
    Ok(Route {
        id: text_field(value, "route_id")?.to_owned(),
        mobile_requirement: text_field(value, "mobile_species_requirement")?.to_owned(),
        thermodynamic_conditions_id: text_field(value, "thermodynamic_conditions_id")?.to_owned(),
        reaction_extent_basis_id: text_field(value, "reaction_extent_basis_id")?.to_owned(),
        standard_state_id: text_field(value, "standard_state_id")?.to_owned(),
        activation_barrier,
        prefactor,
        delta_g,
        availability,
    })
}
fn parse_case(value: &Value) -> Result<Case, String> {
    let transport = value.get("transport").ok_or("case missing transport")?;
    let target_route = parse_route(value.get("target_route").ok_or("case missing target_route")?, "target route")?;
    let competitor_route = parse_route(value.get("competitor_route").ok_or("case missing competitor_route")?, "competitor route")?;
    if target_route.mobile_requirement != "co_transport_a_b" {
        return Err("v0.1 target route must require co-transport of A and B".to_owned());
    }
    if competitor_route.mobile_requirement != "species_a_only" {
        return Err("v0.1 competitor route must require species A only".to_owned());
    }
    if target_route.id == competitor_route.id { return Err("target and competitor route IDs must differ".to_owned()); }
    let transport = Transport {
        species_a: TransportSpecies {
            diffusivity_prefactor: number_field(transport, "species_a_diffusivity_prefactor_m2_s")?,
            diffusion_activation: number_field(transport, "species_a_diffusion_activation_eV")?,
        },
        species_b: TransportSpecies {
            diffusivity_prefactor: number_field(transport, "species_b_diffusivity_prefactor_m2_s")?,
            diffusion_activation: number_field(transport, "species_b_diffusion_activation_eV")?,
        },
        cross_correlation: number_field(transport, "cross_correlation_coefficient")?,
    };
    if transport.species_a.diffusivity_prefactor <= 0.0
        || transport.species_b.diffusivity_prefactor <= 0.0
        || transport.species_a.diffusion_activation < 0.0
        || transport.species_b.diffusion_activation < 0.0 {
        return Err("diffusivity prefactors must be positive and diffusion activation energies non-negative".to_owned());
    }
    if transport.cross_correlation.abs() >= 1.0 {
        return Err("cross-correlation coefficient must be strictly between -1 and +1 for a positive-definite 2x2 transport matrix".to_owned());
    }
    let case = Case {
        id: text_field(value, "case_id")?.to_owned(),
        thermodynamic_conditions_id: text_field(value, "thermodynamic_conditions_id")?.to_owned(),
        temperature: number_field(value, "temperature_k")?,
        hold_time: number_field(value, "hold_time_s")?,
        layer_thickness: number_field(value, "product_layer_thickness_m")?,
        transport,
        target: target_route,
        competitor: competitor_route,
        selectivity_margin: number_field(value, "thermodynamic_selectivity_margin_eV")?,
    };
    if case.temperature <= 0.0 || case.hold_time <= 0.0 || case.layer_thickness <= 0.0
        || case.selectivity_margin < 0.0 {
        return Err(format!("case {} has invalid temperature, hold time, layer thickness or selectivity margin", case.id));
    }
    for route in [&case.target, &case.competitor] {
        if route.thermodynamic_conditions_id != case.thermodynamic_conditions_id {
            return Err(format!("case {} route {} thermodynamic conditions do not match the case", case.id, route.id));
        }
    }
    if case.target.reaction_extent_basis_id != case.competitor.reaction_extent_basis_id {
        return Err(format!("case {} target and competitor reaction extent bases do not match", case.id));
    }
    if case.target.standard_state_id != case.competitor.standard_state_id {
        return Err(format!("case {} target and competitor standard states do not match", case.id));
    }
    Ok(case)
}
fn evaluate_case(case: &Case, k_b: f64, steps: usize) -> Result<CaseResult, String> {
    let d_a = exp_rate(
        case.transport.species_a.diffusivity_prefactor,
        case.transport.species_a.diffusion_activation,
        k_b,
        case.temperature,
    );
    let d_b = exp_rate(
        case.transport.species_b.diffusivity_prefactor,
        case.transport.species_b.diffusion_activation,
        k_b,
        case.temperature,
    );
    let d_ab = case.transport.cross_correlation * (d_a * d_b).sqrt();
    let determinant = d_a * d_b - d_ab * d_ab;
    let trace = d_a + d_b;
    let discriminant = ((d_a - d_b).powi(2) + 4.0 * d_ab.powi(2)).sqrt();
    let eigenvalue_min = (trace - discriminant) / 2.0;
    let eigenvalue_max = (trace + discriminant) / 2.0;
    if determinant <= 0.0 || eigenvalue_min <= 0.0 {
        return Err(format!("case {} transport matrix is not positive definite", case.id));
    }
    let effective_target_diffusivity = (d_a + d_b + 2.0 * d_ab) / 2.0;
    if effective_target_diffusivity <= 0.0 {
        return Err(format!("case {} co-transport projection is not positive", case.id));
    }
    let target_transport_rate = 2.0 * effective_target_diffusivity / case.layer_thickness.powi(2);
    let competitor_transport_rate = 2.0 * d_a / case.layer_thickness.powi(2);
    let target_reaction_rate = exp_rate(
        case.target.prefactor,
        case.target.activation_barrier,
        k_b,
        case.temperature,
    ) * case.target.availability;
    let competitor_reaction_rate = exp_rate(
        case.competitor.prefactor,
        case.competitor.activation_barrier,
        k_b,
        case.temperature,
    ) * case.competitor.availability;
    if target_transport_rate <= 0.0 || competitor_transport_rate <= 0.0
        || target_reaction_rate <= 0.0 || competitor_reaction_rate <= 0.0 {
        return Err(format!("case {} generated a non-positive route rate", case.id));
    }
    let coupled = competing_route_fractions(
        target_transport_rate,
        target_reaction_rate,
        competitor_transport_rate,
        competitor_reaction_rate,
        case.hold_time,
        steps,
    );
    let diagonal_target_diffusivity = (d_a + d_b) / 2.0;
    let diagonal_target_transport_rate = 2.0 * diagonal_target_diffusivity / case.layer_thickness.powi(2);
    let diagonal = competing_route_fractions(
        diagonal_target_transport_rate,
        target_reaction_rate,
        competitor_transport_rate,
        competitor_reaction_rate,
        case.hold_time,
        steps,
    );
    let reaction_only = reaction_only_fractions(
        target_reaction_rate,
        competitor_reaction_rate,
        case.hold_time,
    );
    let thermodynamic = thermodynamic_status(
        case.target.delta_g,
        case.competitor.delta_g,
        case.selectivity_margin,
    ).to_owned();
    let kinetic_phase_ranking = if coupled.target > coupled.competitor {
        "target_phase_fraction_exceeds_competitor"
    } else if coupled.competitor > coupled.target {
        "competitor_phase_fraction_exceeds_target"
    } else {
        "target_competitor_phase_fraction_tie"
    }.to_owned();
    let conservation_error = (coupled.total - 1.0).abs();
    if conservation_error > 1.0e-6 || (diagonal.total - 1.0).abs() > 1.0e-6 {
        return Err(format!("case {} competing-risk phase fractions do not sum to one", case.id));
    }
    Ok(CaseResult {
        transport: TransportState {
            diffusivity_a: d_a,
            diffusivity_b: d_b,
            diffusivity_ab: d_ab,
            determinant,
            eigenvalue_min,
            eigenvalue_max,
            effective_target_diffusivity,
            target_transport_rate,
            competitor_transport_rate,
        },
        kinetics: RouteKinetics { target_reaction_rate, competitor_reaction_rate },
        coupled,
        diagonal,
        reaction_only,
        target_transport_time: 1.0 / target_transport_rate,
        competitor_transport_time: 1.0 / competitor_transport_rate,
        target_reaction_time: 1.0 / target_reaction_rate,
        competitor_reaction_time: 1.0 / competitor_reaction_rate,
        thermodynamic_status: thermodynamic,
        kinetic_phase_ranking,
        conservation_error,
    })
}
fn fractions_json(fractions: PhaseFractions) -> Value {
    json!({
        "target_phase_fraction": fractions.target,
        "competitor_phase_fraction": fractions.competitor,
        "unreacted_fraction": fractions.unreacted,
        "total_fraction": fractions.total
    })
}
fn case_receipt(case: &Case, result: &CaseResult) -> Value {
    json!({
        "case_id": case.id,
        "thermodynamic_conditions_id": case.thermodynamic_conditions_id,
        "reaction_extent_basis_id": case.target.reaction_extent_basis_id,
        "standard_state_id": case.target.standard_state_id,
        "temperature_k": case.temperature,
        "hold_time_s": case.hold_time,
        "product_layer_thickness_m": case.layer_thickness,
        "thermodynamic_selectivity": result.thermodynamic_status,
        "target_delta_g_interval_eV_per_route_extent": {
            "lower": case.target.delta_g.lower, "upper": case.target.delta_g.upper
        },
        "competitor_delta_g_interval_eV_per_route_extent": {
            "lower": case.competitor.delta_g.lower, "upper": case.competitor.delta_g.upper
        },
        "transport_matrix_at_temperature": {
            "species_a_diffusivity_m2_s": result.transport.diffusivity_a,
            "species_b_diffusivity_m2_s": result.transport.diffusivity_b,
            "cross_diffusivity_m2_s": result.transport.diffusivity_ab,
            "cross_correlation_coefficient": case.transport.cross_correlation,
            "determinant": result.transport.determinant,
            "minimum_eigenvalue": result.transport.eigenvalue_min,
            "maximum_eigenvalue": result.transport.eigenvalue_max,
            "positive_definite": true,
            "target_co_transport_projection_m2_s": result.transport.effective_target_diffusivity
        },
        "route_kinetics": {
            "target_route_id": case.target.id,
            "target_activation_barrier_eV": case.target.activation_barrier,
            "target_reaction_prefactor_s_inverse": case.target.prefactor,
            "target_reaction_rate_s_inverse": result.kinetics.target_reaction_rate,
            "competitor_route_id": case.competitor.id,
            "competitor_activation_barrier_eV": case.competitor.activation_barrier,
            "competitor_reaction_prefactor_s_inverse": case.competitor.prefactor,
            "competitor_reaction_rate_s_inverse": result.kinetics.competitor_reaction_rate
        },
        "characteristic_times_s": {
            "target_transport": result.target_transport_time,
            "competitor_transport": result.competitor_transport_time,
            "target_reaction": result.target_reaction_time,
            "competitor_reaction": result.competitor_reaction_time
        },
        "coupled_transport_competing_risks": fractions_json(result.coupled),
        "diagonal_diffusion_baseline_cross_term_zero": fractions_json(result.diagonal),
        "reaction_only_baseline_transport_ignored": fractions_json(result.reaction_only),
        "modelled_phase_ranking": result.kinetic_phase_ranking,
        "phase_fraction_conservation_absolute_error": result.conservation_error,
        "observation_status": "unknown_not_reported",
        "interpretation_ceiling": "Modelled competing-risk phase fractions under the stated reduced two-stage assumptions; not measured phase yield or a spatial reaction-diffusion solution."
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
        return Err(format!("transport kinetics schema validation failed: {error} at {}", error.instance_path()));
    }
    if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
        return Err("transport kinetics auditor accepts synthetic fixtures only".to_owned());
    }
    if text_field(&fixture, "contract")? != "SC-TRANSPORT-KINETIC-SELECTIVITY-0.1" {
        return Err("unsupported transport kinetic selectivity contract".to_owned());
    }
    let k_b = number_field(&fixture, "boltzmann_constant_eV_per_K")?;
    let steps = fixture.get("integration_steps").and_then(Value::as_u64)
        .ok_or("integration_steps must be an integer")? as usize;
    if k_b <= 0.0 || steps < 100 || steps > 100_000 || steps % 2 != 0 {
        return Err("Boltzmann constant must be positive and integration steps even in [100,100000]".to_owned());
    }
    let rows = fixture.get("cases").and_then(Value::as_array).ok_or("cases must be array")?;
    let mut seen = std::collections::HashSet::new();
    let mut receipts = Vec::new();
    for row in rows {
        let case = parse_case(row)?;
        if !seen.insert(case.id.clone()) { return Err(format!("duplicate transport case {}", case.id)); }
        let result = evaluate_case(&case, k_b, steps)?;
        receipts.push(case_receipt(&case, &result));
    }
    Ok(json!({
        "status": "synthetic_transport_kinetic_selectivity_pass",
        "contract": "SC-TRANSPORT-KINETIC-SELECTIVITY-AUDIT-0.1",
        "benchmark_id": text_field(&fixture, "benchmark_id")?,
        "schema_sha256": format!("{:x}", Sha256::digest(schema_bytes)),
        "fixture_sha256": format!("{:x}", Sha256::digest(fixture_bytes)),
        "boltzmann_constant_eV_per_K": k_b,
        "integration_steps": steps,
        "numerical_model": {
            "route_completion_time": "sum of independent exponential transport and reaction waiting times",
            "competing_risk": "target_density * competitor_survival; competitor_density * target_survival",
            "numerical_integration": "composite Simpson rule",
            "assumptions": [
                "route completion clocks are independent",
                "target requires common-mode co-transport of A and B",
                "competitor requires transport of A only",
                "cross diffusivity is rho * sqrt(D_A * D_B)",
                "transport coefficients are constant during one case",
                "no shared precursor depletion, spatial concentration field or moving layer is modelled"
            ]
        },
        "case_count": receipts.len(),
        "cases": receipts,
        "claim_ceiling": "Synthetic two-route transport-coupled kinetic-selectivity model only; no measured diffusion coefficient, reaction rate, phase yield, superconducting material or laboratory procedure is established."
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
    let receipt = audit(&read(&schema_path), &read(&fixture_path)).unwrap_or_else(|e| {
        eprintln!("SC-TRANSPORT-KINETIC-SELECTIVITY-FAIL: {e}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt).expect("JSON receipt serialization cannot fail");
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

    fn inputs_for(filename: &str) -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        (
            fs::read(root.join("data/schemas/superconductivity_transport_kinetic_selectivity_v0_1.schema.json")).unwrap(),
            fs::read(root.join("tools/superconductivity-discovery-harness/fixtures").join(filename)).unwrap(),
        )
    }
    fn inputs() -> (Vec<u8>, Vec<u8>) {
        inputs_for("transport_kinetic_selectivity_smoke_v0_1.json")
    }

    #[test]
    fn cross_ion_correlation_changes_target_transport_projection() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        let cases = receipt["cases"].as_array().unwrap();
        let negative = cases.iter().find(|x| x["case_id"] == "negative_cross_correlation_long_hold_competitor_takes_over").unwrap();
        let positive = cases.iter().find(|x| x["case_id"] == "positive_cross_correlation_target_co_transport_wins").unwrap();
        assert!(negative["transport_matrix_at_temperature"]["target_co_transport_projection_m2_s"].as_f64().unwrap()
            < positive["transport_matrix_at_temperature"]["target_co_transport_projection_m2_s"].as_f64().unwrap());
        assert!(negative["coupled_transport_competing_risks"]["target_phase_fraction"].as_f64().unwrap()
            < positive["coupled_transport_competing_risks"]["target_phase_fraction"].as_f64().unwrap());
    }

    #[test]
    fn hold_time_can_reverse_phase_selection_under_identical_thermodynamics() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        let cases = receipt["cases"].as_array().unwrap();
        let short = cases.iter().find(|x| x["case_id"] == "negative_cross_correlation_short_hold_target_initially_leads").unwrap();
        let long = cases.iter().find(|x| x["case_id"] == "negative_cross_correlation_long_hold_competitor_takes_over").unwrap();
        assert_eq!(short["thermodynamic_selectivity"], "target_reaction_robustly_favored");
        assert_eq!(long["thermodynamic_selectivity"], "target_reaction_robustly_favored");
        assert!(short["coupled_transport_competing_risks"]["target_phase_fraction"].as_f64().unwrap()
            > short["coupled_transport_competing_risks"]["competitor_phase_fraction"].as_f64().unwrap());
        assert!(long["coupled_transport_competing_risks"]["target_phase_fraction"].as_f64().unwrap()
            < long["coupled_transport_competing_risks"]["competitor_phase_fraction"].as_f64().unwrap());
    }

    #[test]
    fn phase_fractions_conserve_probability_mass_for_all_cases() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        for case in receipt["cases"].as_array().unwrap() {
            let fractions = &case["coupled_transport_competing_risks"];
            let sum = fractions["target_phase_fraction"].as_f64().unwrap()
                + fractions["competitor_phase_fraction"].as_f64().unwrap()
                + fractions["unreacted_fraction"].as_f64().unwrap();
            assert!((sum - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn diagonal_transport_and_reaction_only_baselines_are_separate() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        let negative = receipt["cases"].as_array().unwrap().iter()
            .find(|x| x["case_id"] == "negative_cross_correlation_long_hold_competitor_takes_over").unwrap();
        let coupled = negative["coupled_transport_competing_risks"]["target_phase_fraction"].as_f64().unwrap();
        let diagonal = negative["diagonal_diffusion_baseline_cross_term_zero"]["target_phase_fraction"].as_f64().unwrap();
        let reaction_only = negative["reaction_only_baseline_transport_ignored"]["target_phase_fraction"].as_f64().unwrap();
        assert!(coupled < diagonal);
        assert!(coupled < reaction_only);
    }

    #[test]
    fn non_positive_definite_transport_matrix_is_rejected() {
        let (schema, fixture) = inputs_for("transport_kinetic_selectivity_non_psd_v0_1.json");
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["cases"][0]["transport"]["cross_correlation_coefficient"] = json!(1.1);
        assert!(audit(&schema, &serde_json::to_vec(&data).unwrap()).is_err());
    }

    #[test]
    fn target_and_competitor_must_share_conditions_extent_and_standard_state() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["cases"][0]["competitor_route"]["standard_state_id"] = json!("different_reference");
        let error = audit(&schema, &serde_json::to_vec(&data).unwrap()).unwrap_err();
        assert!(error.contains("standard states do not match"));

        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["cases"][0]["target_route"]["reaction_extent_basis_id"] = json!("different_extent");
        let error = audit(&schema, &serde_json::to_vec(&data).unwrap()).unwrap_err();
        assert!(error.contains("reaction extent bases do not match"));

        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["cases"][0]["target_route"]["thermodynamic_conditions_id"] = json!("different_conditions");
        let error = audit(&schema, &serde_json::to_vec(&data).unwrap()).unwrap_err();
        assert!(error.contains("thermodynamic conditions do not match"));
    }

    #[test]
    fn equal_transport_and_reaction_rates_use_the_erlang_limit() {
        let d = 0.03;
        let t = 10.0;
        let s = transport_survival(d, d, t);
        let expected = (-d * t).exp() * (1.0 + d * t);
        assert!((s - expected).abs() < 1.0e-12);
        let f = transport_density(d, d, t);
        assert!((f - d*d*t*(-d*t).exp()).abs() < 1.0e-12);
    }
}
