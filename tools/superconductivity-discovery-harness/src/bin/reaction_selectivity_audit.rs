use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
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
struct Reaction {
    id: String,
    role: String,
    balanced_equation: String,
    product_phase_id: String,
    conditions_id: String,
    normalization_basis: String,
    basis_unit: String,
    balance_status: String,
    delta_g: Interval,
    activation_barrier: Interval,
}
#[derive(Clone, Debug)]
struct PhaseObservation {
    phase_id: String,
    identity_status: String,
    fraction: f64,
}
#[derive(Clone, Debug)]
struct Characterization {
    sample_id: String,
    execution_run_id: String,
    method: String,
    conditions_id: String,
    source_locator: String,
    raw_data_sha256: String,
    detection_limit: f64,
    phases: Vec<PhaseObservation>,
}
#[derive(Clone, Debug)]
struct Case {
    id: String,
    protocol_id: String,
    protocol_revision: String,
    conditions_id: String,
    temperature_k: f64,
    pressure_gpa: f64,
    atmosphere: String,
    chemical_potential_context: String,
    basis: String,
    basis_unit: String,
    standard_state: String,
    basis_balance_status: String,
    hull: Interval,
    hull_threshold: f64,
    hull_source_class: String,
    target: Reaction,
    competitors: Vec<Reaction>,
    selectivity_margin: f64,
    kinetic_limit: f64,
    outcome_status: String,
    characterization: Option<Characterization>,
}
fn text_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}"))
}
fn number_field(value: &Value, key: &str) -> Result<f64, String> {
    let x = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric field {key}"))?;
    if !x.is_finite() {
        return Err(format!("{key} must be finite"));
    }
    Ok(x)
}
fn interval(value: &Value, context: &str) -> Result<Interval, String> {
    let lower = number_field(value, "lower")?;
    let upper = number_field(value, "upper")?;
    if lower > upper {
        return Err(format!(
            "{context} interval lower bound exceeds upper bound"
        ));
    }
    Ok(Interval { lower, upper })
}
fn parse_reaction(value: &Value, context: &str) -> Result<Reaction, String> {
    let delta_g = interval(
        value
            .get("delta_g_interval_eV_per_basis")
            .ok_or_else(|| format!("{context} missing delta_g_interval_eV_per_basis"))?,
        &format!("{context} delta_g"),
    )?;
    let activation_barrier = interval(
        value
            .get("activation_barrier_interval_eV")
            .ok_or_else(|| format!("{context} missing activation_barrier_interval_eV"))?,
        &format!("{context} activation barrier"),
    )?;
    if activation_barrier.lower < 0.0 {
        return Err(format!("{context} activation barrier must be non-negative"));
    }
    Ok(Reaction {
        id: text_field(value, "reaction_id")?.to_owned(),
        role: text_field(value, "role")?.to_owned(),
        balanced_equation: text_field(value, "balanced_equation")?.to_owned(),
        product_phase_id: text_field(value, "product_phase_id")?.to_owned(),
        conditions_id: text_field(value, "conditions_id")?.to_owned(),
        normalization_basis: text_field(value, "normalization_basis")?.to_owned(),
        basis_unit: text_field(value, "basis_unit")?.to_owned(),
        balance_status: text_field(value, "balance_status")?.to_owned(),
        delta_g,
        activation_barrier,
    })
}
fn parse_characterization(value: &Value) -> Result<Option<Characterization>, String> {
    let evidence = value
        .get("characterization_evidence")
        .ok_or("case missing characterization_evidence")?;
    if evidence.is_null() {
        return Ok(None);
    }
    let evidence = evidence
        .as_object()
        .ok_or("characterization_evidence must be an object or null")?;
    let evidence_value = Value::Object(evidence.clone());
    let detection_limit = number_field(&evidence_value, "detection_limit_phase_fraction")?;
    if !(0.0..=1.0).contains(&detection_limit) || detection_limit == 0.0 {
        return Err("detection_limit_phase_fraction must be in (0,1]".to_owned());
    }
    let digest = text_field(&evidence_value, "raw_data_sha256")?;
    if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("raw_data_sha256 must contain 64 hexadecimal characters".to_owned());
    }
    let mut phases = Vec::new();
    let mut phase_ids = HashSet::new();
    let mut fraction_sum = 0.0;
    let phase_rows = evidence_value
        .get("realized_phases")
        .and_then(Value::as_array)
        .ok_or("characterization realized_phases must be an array")?;
    for row in phase_rows {
        let phase_id = text_field(row, "phase_id")?.to_owned();
        if !phase_ids.insert(phase_id.clone()) {
            return Err(format!("duplicate realized phase {phase_id}"));
        }
        let fraction = number_field(row, "phase_fraction")?;
        if !(0.0..=1.0).contains(&fraction) {
            return Err(format!("phase {phase_id} fraction must be in [0,1]"));
        }
        fraction_sum += fraction;
        phases.push(PhaseObservation {
            phase_id,
            identity_status: text_field(row, "identity_status")?.to_owned(),
            fraction,
        });
    }
    if fraction_sum > 1.0 + 1.0e-9 {
        return Err(format!(
            "realized phase fractions sum to {fraction_sum}, exceeding one"
        ));
    }
    Ok(Some(Characterization {
        sample_id: text_field(&evidence_value, "sample_id")?.to_owned(),
        execution_run_id: text_field(&evidence_value, "execution_run_id")?.to_owned(),
        method: text_field(&evidence_value, "method")?.to_owned(),
        conditions_id: text_field(&evidence_value, "conditions_id")?.to_owned(),
        source_locator: text_field(&evidence_value, "source_locator")?.to_owned(),
        raw_data_sha256: digest.to_owned(),
        detection_limit,
        phases,
    }))
}
fn parse_case(value: &Value) -> Result<Case, String> {
    let id = text_field(value, "case_id")?.to_owned();
    let conditions = value.get("conditions").ok_or("case missing conditions")?;
    let basis = value
        .get("reaction_basis")
        .ok_or("case missing reaction_basis")?;
    let hull = value.get("hull_screen").ok_or("case missing hull_screen")?;
    let hull_interval = interval(
        hull.get("above_hull_interval_mev_atom")
            .ok_or("hull_screen missing interval")?,
        "above-hull",
    )?;
    if hull_interval.lower < 0.0 {
        return Err(format!("{id} above-hull interval must be non-negative"));
    }
    let target = parse_reaction(
        value
            .get("target_reaction")
            .ok_or("case missing target_reaction")?,
        &format!("{id} target reaction"),
    )?;
    if target.role != "target" {
        return Err(format!("{id} target_reaction must have role=target"));
    }
    let mut competitors = Vec::new();
    let mut reaction_ids = HashSet::new();
    if !reaction_ids.insert(target.id.clone()) {
        return Err(format!("{id} duplicate target reaction id"));
    }
    let competitor_rows = value
        .get("competing_reactions")
        .and_then(Value::as_array)
        .ok_or("competing_reactions must be an array")?;
    if competitor_rows.is_empty() {
        return Err(format!("{id} requires at least one competing reaction"));
    }
    for (index, row) in competitor_rows.iter().enumerate() {
        let reaction = parse_reaction(row, &format!("{id} competitor[{index}]"))?;
        if reaction.role != "competing" {
            return Err(format!(
                "{id} competing reaction {} must have role=competing",
                reaction.id
            ));
        }
        if !reaction_ids.insert(reaction.id.clone()) {
            return Err(format!("{id} duplicate reaction id {}", reaction.id));
        }
        competitors.push(reaction);
    }
    let case = Case {
        id,
        protocol_id: text_field(value, "protocol_id")?.to_owned(),
        protocol_revision: text_field(value, "protocol_revision")?.to_owned(),
        conditions_id: text_field(conditions, "conditions_id")?.to_owned(),
        temperature_k: number_field(conditions, "temperature_k")?,
        pressure_gpa: number_field(conditions, "pressure_gpa")?,
        atmosphere: text_field(conditions, "atmosphere")?.to_owned(),
        chemical_potential_context: text_field(conditions, "chemical_potential_context")?
            .to_owned(),
        basis: text_field(basis, "normalization_basis")?.to_owned(),
        basis_unit: text_field(basis, "basis_unit")?.to_owned(),
        standard_state: text_field(basis, "standard_state")?.to_owned(),
        basis_balance_status: text_field(basis, "balance_status")?.to_owned(),
        hull: hull_interval,
        hull_threshold: number_field(hull, "screening_threshold_mev_atom")?,
        hull_source_class: text_field(hull, "source_class")?.to_owned(),
        target,
        competitors,
        selectivity_margin: number_field(value, "selectivity_margin_eV_per_basis")?,
        kinetic_limit: number_field(value, "kinetic_barrier_limit_eV")?,
        outcome_status: text_field(value, "observed_target_phase_status")?.to_owned(),
        characterization: parse_characterization(value)?,
    };
    if case.temperature_k <= 0.0
        || case.pressure_gpa < 0.0
        || case.hull_threshold < 0.0
        || case.selectivity_margin < 0.0
        || case.kinetic_limit < 0.0
    {
        return Err(format!(
            "{} has an invalid non-negative condition or threshold",
            case.id
        ));
    }
    let all_reactions = std::iter::once(&case.target).chain(case.competitors.iter());
    for reaction in all_reactions {
        if reaction.conditions_id != case.conditions_id {
            return Err(format!(
                "{} reaction {} condition ID does not match case conditions",
                case.id, reaction.id
            ));
        }
        if reaction.normalization_basis != case.basis || reaction.basis_unit != case.basis_unit {
            return Err(format!(
                "{} reaction {} normalization basis/unit does not match case basis",
                case.id, reaction.id
            ));
        }
    }
    if case.target.balance_status != case.basis_balance_status {
        return Err(format!(
            "{} target reaction and case reaction-basis balance status disagree",
            case.id
        ));
    }
    validate_observation(&case)?;
    Ok(case)
}
fn validate_observation(case: &Case) -> Result<(), String> {
    match case.outcome_status.as_str() {
        "unknown_not_reported" => {
            if case.characterization.is_some() {
                return Err(format!(
                    "{} unknown_not_reported must not attach a characterized outcome",
                    case.id
                ));
            }
        }
        "target_phase_observed" => {
            let evidence = case.characterization.as_ref().ok_or_else(|| {
                format!(
                    "{} target observed requires characterization evidence",
                    case.id
                )
            })?;
            let phase = evidence.phases.iter().find(|p| p.phase_id == case.target.product_phase_id)
                .ok_or_else(|| format!("{} target phase observed status lacks target phase in characterized phases", case.id))?;
            if !matches!(phase.identity_status.as_str(), "refined" | "identified") {
                return Err(format!(
                    "{} target phase observation is not identified/refined",
                    case.id
                ));
            }
            if phase.fraction < evidence.detection_limit {
                return Err(format!(
                    "{} target phase fraction is below the declared detection limit",
                    case.id
                ));
            }
        }
        "target_phase_not_detected_under_conditions" => {
            let evidence = case.characterization.as_ref().ok_or_else(|| {
                format!(
                    "{} non-detection requires characterization evidence and a detection limit",
                    case.id
                )
            })?;
            if evidence
                .phases
                .iter()
                .any(|p| p.phase_id == case.target.product_phase_id)
            {
                return Err(format!(
                    "{} non-detection cannot include the target phase in realized_phases",
                    case.id
                ));
            }
            if evidence.detection_limit <= 0.0 {
                return Err(format!(
                    "{} non-detection requires a positive detection limit",
                    case.id
                ));
            }
        }
        other => {
            return Err(format!(
                "{} has unsupported outcome status {other}",
                case.id
            ));
        }
    }
    Ok(())
}
fn hull_assessment(case: &Case) -> &'static str {
    if case.hull.upper <= case.hull_threshold {
        "within_declared_hull_screen_threshold"
    } else if case.hull.lower > case.hull_threshold {
        "above_declared_hull_screen_threshold"
    } else {
        "hull_interval_overlaps_declared_threshold"
    }
}
fn reaction_selectivity(case: &Case) -> &'static str {
    let reactions_verified = case.basis_balance_status == "verified"
        && case.target.balance_status == "verified"
        && case
            .competitors
            .iter()
            .all(|r| r.balance_status == "verified");
    if !reactions_verified {
        return "abstained_unverified_reaction_balance";
    }
    if case.competitors.iter().all(|competitor| {
        case.target.delta_g.upper + case.selectivity_margin < competitor.delta_g.lower
    }) {
        return "target_reaction_robustly_favored";
    }
    if case.competitors.iter().any(|competitor| {
        competitor.delta_g.upper + case.selectivity_margin < case.target.delta_g.lower
    }) {
        return "at_least_one_competing_reaction_robustly_favored";
    }
    "reaction_free_energy_intervals_overlap_or_are_unresolved"
}
fn kinetic_assessment(case: &Case) -> &'static str {
    if case.target.activation_barrier.upper <= case.kinetic_limit {
        "target_barrier_below_declared_limit"
    } else if case.target.activation_barrier.lower > case.kinetic_limit {
        "target_barrier_above_declared_limit"
    } else {
        "target_barrier_interval_overlaps_declared_limit"
    }
}
fn combined_assessment(selectivity: &str, kinetics: &str) -> &'static str {
    match (selectivity, kinetics) {
        ("target_reaction_robustly_favored", "target_barrier_below_declared_limit") => {
            "target_route_selective_and_below_declared_kinetic_limit"
        }
        ("target_reaction_robustly_favored", "target_barrier_above_declared_limit") => {
            "target_route_selective_but_barrier_above_declared_limit"
        }
        ("target_reaction_robustly_favored", _) => {
            "target_route_selective_but_kinetics_indeterminate"
        }
        ("at_least_one_competing_reaction_robustly_favored", _) => {
            "competing_route_robustly_preferred"
        }
        ("abstained_unverified_reaction_balance", _) => {
            "abstained_unverified_or_noncomparable_reactions"
        }
        _ => "route_selectivity_indeterminate",
    }
}
fn case_receipt(case: &Case) -> Value {
    let selectivity = reaction_selectivity(case);
    let kinetics = kinetic_assessment(case);
    let observed = case.characterization.as_ref().map(|evidence| {
        json!({
            "sample_id": evidence.sample_id,
            "execution_run_id": evidence.execution_run_id,
            "method": evidence.method,
            "characterization_conditions_id": evidence.conditions_id,
            "source_locator": evidence.source_locator,
            "raw_data_sha256": evidence.raw_data_sha256,
            "detection_limit_phase_fraction": evidence.detection_limit,
            "realized_phases": evidence.phases.iter().map(|p| json!({
                "phase_id": p.phase_id,
                "identity_status": p.identity_status,
                "phase_fraction": p.fraction
            })).collect::<Vec<_>>()
        })
    });
    json!({
        "case_id": case.id,
        "protocol_id": case.protocol_id,
        "protocol_revision": case.protocol_revision,
        "conditions": {
            "conditions_id": case.conditions_id,
            "temperature_k": case.temperature_k,
            "pressure_gpa": case.pressure_gpa,
            "atmosphere": case.atmosphere,
            "chemical_potential_context": case.chemical_potential_context
        },
        "reaction_basis": {
            "normalization_basis": case.basis,
            "basis_unit": case.basis_unit,
            "standard_state": case.standard_state,
            "declared_balance_status": case.basis_balance_status,
            "balance_status_semantics": "Caller/source-declared assertion; the auditor checks internal consistency but does not independently balance elemental or charge stoichiometry.",
            "normalization_comparison_semantics": "Target and competing reactions are compared only after exact matching of the declared common extent, unit and reaction-condition identifier."
        },
        "hull_screen_assessment": hull_assessment(case),
        "above_hull_interval_mev_atom": {"lower":case.hull.lower,"upper":case.hull.upper},
        "hull_screen_threshold_mev_atom": case.hull_threshold,
        "hull_source_class": case.hull_source_class,
        "reaction_selectivity_assessment": selectivity,
        "target_reaction": {
            "reaction_id": case.target.id,
            "balanced_equation": case.target.balanced_equation,
            "product_phase_id": case.target.product_phase_id,
            "delta_g_interval_eV_per_basis": {"lower":case.target.delta_g.lower,"upper":case.target.delta_g.upper},
            "activation_barrier_interval_eV": {"lower":case.target.activation_barrier.lower,"upper":case.target.activation_barrier.upper},
            "balance_status": case.target.balance_status
        },
        "competing_reactions": case.competitors.iter().map(|r| json!({
            "reaction_id": r.id,
            "balanced_equation": r.balanced_equation,
            "product_phase_id": r.product_phase_id,
            "delta_g_interval_eV_per_basis": {"lower":r.delta_g.lower,"upper":r.delta_g.upper},
            "activation_barrier_interval_eV": {"lower":r.activation_barrier.lower,"upper":r.activation_barrier.upper},
            "balance_status": r.balance_status
        })).collect::<Vec<_>>(),
        "selectivity_margin_eV_per_basis": case.selectivity_margin,
        "kinetic_assessment": kinetics,
        "target_activation_barrier_interval_eV": {"lower":case.target.activation_barrier.lower,"upper":case.target.activation_barrier.upper},
        "kinetic_barrier_limit_eV": case.kinetic_limit,
        "model_route_assessment": combined_assessment(selectivity, kinetics),
        "observed_target_phase_status": case.outcome_status,
        "characterization_evidence": observed,
        "interpretation_ceiling": "Thermodynamic interval comparison, hull screen and kinetic barrier are distinct model diagnostics. They do not prove a reaction will proceed or a phase will form."
    })
}
fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|e| format!("schema JSON parse failed: {e}"))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes)
        .map_err(|e| format!("fixture JSON parse failed: {e}"))?;
    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator =
        validator_for(&schema).map_err(|e| format!("schema compilation failed: {e}"))?;
    if let Some(error) = validator.iter_errors(&fixture).next() {
        return Err(format!(
            "reaction selectivity schema validation failed: {error} at {}",
            error.instance_path()
        ));
    }
    if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
        return Err("reaction selectivity auditor accepts synthetic fixtures only".to_owned());
    }
    if text_field(&fixture, "contract")? != "SC-REACTION-SELECTIVITY-0.1" {
        return Err("unsupported reaction selectivity contract".to_owned());
    }
    let rows = fixture
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| "cases must be an array".to_owned())?;
    let mut cases = Vec::with_capacity(rows.len());
    let mut ids = HashSet::new();
    for row in rows {
        let case = parse_case(row)?;
        if !ids.insert(case.id.clone()) {
            return Err(format!("duplicate reaction selectivity case {}", case.id));
        }
        cases.push(case);
    }
    let suite_id = text_field(&fixture, "suite_id")?;
    let receipts: Vec<Value> = cases.iter().map(case_receipt).collect();
    Ok(json!({
        "status": "synthetic_reaction_selectivity_contract_pass",
        "contract": "SC-REACTION-SELECTIVITY-AUDIT-0.1",
        "suite_id": suite_id,
        "schema_sha256": format!("{:x}", Sha256::digest(schema_bytes)),
        "fixture_sha256": format!("{:x}", Sha256::digest(fixture_bytes)),
        "case_count": cases.len(),
        "cases": receipts,
        "claim_ceiling": "Synthetic thermodynamic-selectivity and characterization-evidence audit only. No route recommendation, reaction feasibility, observed superconductivity, material synthesis or laboratory authorization is established."
    }))
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| {
        eprintln!("could not read {}: {e}", path.display());
        process::exit(1);
    })
}
fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <schema.json> --fixture <fixture.json> [--out <receipt.json>]"
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
            other => {
                eprintln!("unknown argument {other}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }
    let (Some(schema_path), Some(fixture_path)) = (schema_path, fixture_path) else {
        usage(&args[0]);
        process::exit(2);
    };
    let receipt = audit(&read(&schema_path), &read(&fixture_path)).unwrap_or_else(|e| {
        eprintln!("SC-REACTION-SELECTIVITY-FAIL: {e}");
        process::exit(1);
    });
    let rendered =
        serde_json::to_string_pretty(&receipt).expect("JSON receipt serialization cannot fail");
    if let Some(path) = output_path {
        if let Err(e) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write {}: {e}", path.display());
            process::exit(1);
        }
    } else {
        println!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        (
            fs::read(root.join("data/schemas/superconductivity_reaction_selectivity_v0_1.schema.json")).unwrap(),
            fs::read(root.join("tools/superconductivity-discovery-harness/fixtures/reaction_selectivity_smoke_v0_1.json")).unwrap(),
        )
    }
    fn receipt() -> Value {
        let (schema, fixture) = inputs();
        audit(&schema, &fixture).unwrap()
    }

    #[test]
    fn low_hull_screen_does_not_override_a_competing_reaction() {
        let result = receipt();
        let case = result["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["case_id"] == "low_hull_wrong_reaction_route")
            .unwrap();
        assert_eq!(
            case["hull_screen_assessment"],
            "within_declared_hull_screen_threshold"
        );
        assert_eq!(
            case["reaction_selectivity_assessment"],
            "at_least_one_competing_reaction_robustly_favored"
        );
        assert_eq!(
            case["kinetic_assessment"],
            "target_barrier_above_declared_limit"
        );
        assert_eq!(
            case["model_route_assessment"],
            "competing_route_robustly_preferred"
        );
    }

    #[test]
    fn metastable_hull_does_not_automatically_reject_a_selective_route() {
        let result = receipt();
        let case = result["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["case_id"] == "metastable_hull_selective_route_unknown")
            .unwrap();
        assert_eq!(
            case["hull_screen_assessment"],
            "above_declared_hull_screen_threshold"
        );
        assert_eq!(
            case["reaction_selectivity_assessment"],
            "target_reaction_robustly_favored"
        );
        assert_eq!(
            case["kinetic_assessment"],
            "target_barrier_below_declared_limit"
        );
        assert_eq!(case["observed_target_phase_status"], "unknown_not_reported");
        assert!(case["characterization_evidence"].is_null());
    }

    #[test]
    fn overlapping_reaction_intervals_abstain_from_selectivity() {
        let result = receipt();
        let case = result["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["case_id"] == "overlapping_reaction_intervals_unknown")
            .unwrap();
        assert_eq!(
            case["reaction_selectivity_assessment"],
            "reaction_free_energy_intervals_overlap_or_are_unresolved"
        );
        assert_eq!(
            case["kinetic_assessment"],
            "target_barrier_interval_overlaps_declared_limit"
        );
        assert_eq!(
            case["hull_screen_assessment"],
            "hull_interval_overlaps_declared_threshold"
        );
        assert_eq!(case["observed_target_phase_status"], "unknown_not_reported");
    }

    #[test]
    fn multiphase_target_observation_preserves_all_phase_fractions() {
        let result = receipt();
        let case = result["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["case_id"] == "selective_multiphase_target_observed")
            .unwrap();
        assert_eq!(
            case["observed_target_phase_status"],
            "target_phase_observed"
        );
        assert_eq!(
            case["characterization_evidence"]["realized_phases"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            case["characterization_evidence"]["detection_limit_phase_fraction"],
            0.01
        );
        assert_eq!(
            case["characterization_evidence"]["execution_run_id"],
            "run_selective_multiphase_target_observed"
        );
        assert_ne!(
            case["characterization_evidence"]["characterization_conditions_id"],
            case["conditions"]["conditions_id"]
        );
    }

    #[test]
    fn unknown_is_not_a_failed_synthesis() {
        let result = receipt();
        let case = result["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["case_id"] == "metastable_hull_selective_route_unknown")
            .unwrap();
        assert_eq!(case["observed_target_phase_status"], "unknown_not_reported");
        assert!(case["characterization_evidence"].is_null());
        assert_eq!(
            case["model_route_assessment"],
            "target_route_selective_and_below_declared_kinetic_limit"
        );
    }

    #[test]
    fn unverified_reaction_balance_abstains() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["cases"][1]["reaction_basis"]["balance_status"] = json!("unverified");
        data["cases"][1]["target_reaction"]["balance_status"] = json!("unverified");
        let output = audit(&schema, &serde_json::to_vec(&data).unwrap()).unwrap();
        let case = output["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["case_id"] == "metastable_hull_selective_route_unknown")
            .unwrap();
        assert_eq!(
            case["reaction_selectivity_assessment"],
            "abstained_unverified_reaction_balance"
        );
        assert_eq!(
            case["model_route_assessment"],
            "abstained_unverified_or_noncomparable_reactions"
        );
    }

    #[test]
    fn non_detection_requires_conditions_and_positive_detection_limit() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["cases"][0]["characterization_evidence"] = Value::Null;
        assert!(audit(&schema, &serde_json::to_vec(&data).unwrap()).is_err());

        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["cases"][0]["characterization_evidence"]["detection_limit_phase_fraction"] =
            json!(0.0);
        assert!(audit(&schema, &serde_json::to_vec(&data).unwrap()).is_err());
    }

    #[test]
    fn characterized_phase_fractions_cannot_exceed_one() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["cases"][3]["characterization_evidence"]["realized_phases"][1]["phase_fraction"] =
            json!(0.8);
        assert!(audit(&schema, &serde_json::to_vec(&data).unwrap()).is_err());
    }

    #[test]
    fn reaction_condition_and_normalization_mismatch_is_rejected() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["cases"][0]["competing_reactions"][0]["conditions_id"] = json!("different-conditions");
        assert!(audit(&schema, &serde_json::to_vec(&data).unwrap()).is_err());

        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["cases"][0]["competing_reactions"][0]["basis_unit"] = json!("eV_per_atom");
        assert!(audit(&schema, &serde_json::to_vec(&data).unwrap()).is_err());
    }
    #[test]
    fn tentative_target_phase_identity_is_not_promoted_to_observed() {
        let (schema, fixture) = inputs();
        let mut data: Value = serde_json::from_slice(&fixture).unwrap();
        data["cases"][3]["characterization_evidence"]["realized_phases"][0]["identity_status"] =
            json!("tentative");
        assert!(audit(&schema, &serde_json::to_vec(&data).unwrap()).is_err());
    }

    #[test]
    fn characterization_conditions_are_distinct_from_reaction_conditions() {
        let result = receipt();
        let case = result["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["case_id"] == "low_hull_wrong_reaction_route")
            .unwrap();
        assert_ne!(
            case["conditions"]["conditions_id"],
            case["characterization_evidence"]["characterization_conditions_id"]
        );
        assert_eq!(
            case["characterization_evidence"]["execution_run_id"],
            "run_low_hull_wrong_reaction_route"
        );
    }
}
