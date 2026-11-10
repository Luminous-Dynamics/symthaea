use jsonschema::{Validator, draft202012};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/transport/ev-cell-limited-power-envelope/0.1";
const CORPUS_VERSION: &str = "ev-cell-limited-power-envelope-corpus/0.1";
const PROFILE_VERSION: &str = "ev-cell-limited-power-envelope/0.1";
const EPSILON: f64 = 1e-12;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    corpus_version: String,
    schema_id: String,
    cases: Vec<CorpusCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusCase {
    case_id: String,
    expectation: String,
    expected_limiting_cell: Option<String>,
    expected_limiting_constraint: Option<String>,
    expected_max_current_a: Option<f64>,
    expected_power_w: Option<f64>,
    profile: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    profile_version: String,
    profile_id: String,
    asset: Asset,
    topology: Topology,
    model: Model,
    pulse: Pulse,
    cells: Vec<Cell>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Asset {
    vehicle_ref: String,
    vehicle_configuration_digest: String,
    pack_ref: String,
    pack_configuration_digest: String,
    cell_inventory_digest: String,
    chemistry: String,
    declared_cell_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Topology {
    status: String,
    architecture: String,
    topology_ref: String,
    topology_digest: String,
    qualification: Option<TopologyQualification>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TopologyQualification {
    qualification_ref: String,
    authority_ref: String,
    evidence_digest: String,
    evidence_class: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {
    model_ref: String,
    model_digest: String,
    equation_ref: String,
    parameter_digest: String,
    soc_valid_min: f64,
    soc_valid_max: f64,
    temperature_valid_min_c: f64,
    temperature_valid_max_c: f64,
    calibration_valid: bool,
    domain_supported: bool,
    protocol_ref: String,
    protocol_digest: String,
    calibration_digest: String,
    calibration_evidence_class: String,
    domain_scope_digest: String,
    domain_scope_evidence_class: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pulse {
    direction: String,
    duration_s: f64,
    ambient_temperature_upper_c: f64,
    ambient_evidence_digest: String,
    ambient_evidence_class: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cell {
    cell_ref: String,
    cell_identity_digest: String,
    ocv_lower_v: f64,
    ocv_upper_v: f64,
    ocv_evidence_digest: String,
    ocv_evidence_class: String,
    resistance_lower_ohm: f64,
    resistance_upper_ohm: f64,
    resistance_evidence_digest: String,
    resistance_evidence_class: String,
    soc_fraction: f64,
    soc_uncertainty: f64,
    soc_evidence_digest: String,
    soc_evidence_class: String,
    temperature_estimate_c: f64,
    temperature_uncertainty_c: f64,
    temperature_evidence_digest: String,
    temperature_evidence_class: String,
    minimum_voltage_v: f64,
    voltage_limit_evidence_digest: String,
    voltage_limit_evidence_class: String,
    maximum_discharge_current_lower_a: f64,
    current_limit_evidence_digest: String,
    current_limit_evidence_class: String,
    thermal_resistance_upper_k_per_w: f64,
    thermal_capacity_lower_j_per_k: f64,
    thermal_evidence_digest: String,
    thermal_evidence_class: String,
    maximum_temperature_lower_c: f64,
    cell_configuration_digest: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EvidenceRef {
    pub role: String,
    pub digest: String,
    pub evidence_class: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CellLimit {
    pub cell_ref: String,
    pub voltage_limited_current_a: f64,
    pub current_rating_limit_a: f64,
    pub thermal_limited_current_a: f64,
    pub admissible_current_a: f64,
    pub limiting_constraint: String,
    pub terminal_voltage_lower_v_at_selected_current: Option<f64>,
    pub temperature_upper_c_at_selected_current: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PowerEnvelope {
    pub profile_id: String,
    pub status: String,
    pub pulse_duration_s: f64,
    pub maximum_admissible_current_a: f64,
    pub selected_current_a: Option<f64>,
    pub pack_terminal_voltage_lower_v: Option<f64>,
    pub conservative_terminal_power_lower_w: Option<f64>,
    pub limiting_cell: String,
    pub limiting_constraint: String,
    pub topology_qualification_ref: String,
    pub topology_authority_ref: String,
    pub model_ref: String,
    pub model_digest: String,
    pub cells: Vec<CellLimit>,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Serialize)]
struct CaseAudit {
    case_id: String,
    expectation: String,
    accepted: bool,
    errors: Vec<String>,
    envelope: Option<PowerEnvelope>,
}

#[derive(Debug, Serialize)]
struct ReportPayload {
    corpus_version: String,
    schema_id: String,
    schema_sha256: String,
    canonical_corpus_sha256: String,
    case_count: usize,
    computed_count: usize,
    no_discharge_count: usize,
    rejected_count: usize,
    cases: Vec<CaseAudit>,
}

#[derive(Debug, Serialize)]
pub struct CorpusReport {
    #[serde(flatten)]
    payload: ReportPayload,
    report_sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_class(value: &str) -> bool {
    matches!(
        value,
        "reference_test_measurement"
            | "derived_estimate"
            | "model_prediction"
            | "simulation_output"
            | "synthetic_fixture"
            | "vendor_claim"
    )
}

fn schema_validator(bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("schema JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if schema.get("$id").and_then(Value::as_str) != Some(SCHEMA_ID) {
        return Err("schema identity does not match power-envelope v0.1".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("schema compilation: {error}"))
}

fn validate_profile(profile: &Profile) -> Vec<String> {
    let mut errors = Vec::new();
    if profile.profile_version != PROFILE_VERSION || !nonempty(&profile.profile_id) {
        errors.push("profile version or identity is invalid".to_owned());
    }
    if !nonempty(&profile.asset.vehicle_ref)
        || !nonempty(&profile.asset.pack_ref)
        || !nonempty(&profile.asset.chemistry)
        || profile.asset.declared_cell_count != profile.cells.len()
        || profile.cells.len() < 2
        || !is_digest(&profile.asset.vehicle_configuration_digest)
        || !is_digest(&profile.asset.pack_configuration_digest)
        || !is_digest(&profile.asset.cell_inventory_digest)
    {
        errors.push("asset identity or declared series-cell count is invalid".to_owned());
    }
    if profile.topology.status != "qualified"
        || profile.topology.architecture != "series_single_parallel"
        || !nonempty(&profile.topology.topology_ref)
        || !is_digest(&profile.topology.topology_digest)
    {
        errors.push("v0.1 requires explicitly qualified series-single-parallel topology".to_owned());
    }
    match &profile.topology.qualification {
        Some(qualification)
            if nonempty(&qualification.qualification_ref)
                && nonempty(&qualification.authority_ref)
                && is_digest(&qualification.evidence_digest)
                && valid_class(&qualification.evidence_class) => {}
        _ => errors.push("topology requires authority-referenced qualification evidence".to_owned()),
    }

    let model = &profile.model;
    if !nonempty(&model.model_ref)
        || !is_digest(&model.model_digest)
        || !nonempty(&model.equation_ref)
        || !is_digest(&model.parameter_digest)
        || !is_digest(&model.protocol_digest)
        || !nonempty(&model.protocol_ref)
        || !is_digest(&model.calibration_digest)
        || !valid_class(&model.calibration_evidence_class)
        || !is_digest(&model.domain_scope_digest)
        || !valid_class(&model.domain_scope_evidence_class)
        || !model.soc_valid_min.is_finite()
        || !model.soc_valid_max.is_finite()
        || model.soc_valid_min < 0.0
        || model.soc_valid_max > 1.0
        || model.soc_valid_min >= model.soc_valid_max
        || !model.temperature_valid_min_c.is_finite()
        || !model.temperature_valid_max_c.is_finite()
        || model.temperature_valid_min_c >= model.temperature_valid_max_c
    {
        errors.push("model, protocol, calibration, domain or validity-domain contract is invalid".to_owned());
    }
    if !model.calibration_valid {
        errors.push("model calibration is not valid".to_owned());
    }
    if !model.domain_supported {
        errors.push("model does not support this operating domain".to_owned());
    }

    if profile.pulse.direction != "discharge"
        || !profile.pulse.duration_s.is_finite()
        || profile.pulse.duration_s <= 0.0
        || profile.pulse.duration_s > 600.0
        || !profile.pulse.ambient_temperature_upper_c.is_finite()
        || !is_digest(&profile.pulse.ambient_evidence_digest)
        || !valid_class(&profile.pulse.ambient_evidence_class)
    {
        errors.push("pulse direction, duration or ambient evidence is invalid".to_owned());
    }
    if profile.pulse.ambient_temperature_upper_c < model.temperature_valid_min_c
        || profile.pulse.ambient_temperature_upper_c > model.temperature_valid_max_c
    {
        errors.push("ambient temperature is outside the model's declared temperature domain".to_owned());
    }

    let mut ids = BTreeSet::new();
    for cell in &profile.cells {
        if !nonempty(&cell.cell_ref)
            || !ids.insert(cell.cell_ref.clone())
            || !is_digest(&cell.cell_identity_digest)
            || !is_digest(&cell.cell_configuration_digest)
        {
            errors.push(format!("cell {} identity/configuration is invalid or duplicated", cell.cell_ref));
        }
        if !cell.ocv_lower_v.is_finite()
            || !cell.ocv_upper_v.is_finite()
            || cell.ocv_lower_v <= 0.0
            || cell.ocv_upper_v < cell.ocv_lower_v
            || !is_digest(&cell.ocv_evidence_digest)
            || !valid_class(&cell.ocv_evidence_class)
        {
            errors.push(format!("cell {} OCV interval/evidence is invalid", cell.cell_ref));
        }
        if !cell.resistance_lower_ohm.is_finite()
            || !cell.resistance_upper_ohm.is_finite()
            || cell.resistance_lower_ohm <= 0.0
            || cell.resistance_upper_ohm < cell.resistance_lower_ohm
            || !is_digest(&cell.resistance_evidence_digest)
            || !valid_class(&cell.resistance_evidence_class)
        {
            errors.push(format!("cell {} resistance interval/evidence is invalid", cell.cell_ref));
        }
        if !cell.soc_fraction.is_finite()
            || !cell.soc_uncertainty.is_finite()
            || cell.soc_uncertainty < 0.0
            || cell.soc_fraction - cell.soc_uncertainty < model.soc_valid_min
            || cell.soc_fraction + cell.soc_uncertainty > model.soc_valid_max
            || !is_digest(&cell.soc_evidence_digest)
            || !valid_class(&cell.soc_evidence_class)
        {
            errors.push(format!("cell {} SOC is outside its qualified model domain", cell.cell_ref));
        }
        let temperature_lower = cell.temperature_estimate_c - cell.temperature_uncertainty_c;
        let temperature_upper = cell.temperature_estimate_c + cell.temperature_uncertainty_c;
        if !cell.temperature_estimate_c.is_finite()
            || !cell.temperature_uncertainty_c.is_finite()
            || cell.temperature_uncertainty_c < 0.0
            || temperature_lower < model.temperature_valid_min_c
            || temperature_upper > model.temperature_valid_max_c
            || !is_digest(&cell.temperature_evidence_digest)
            || !valid_class(&cell.temperature_evidence_class)
        {
            errors.push(format!("cell {} temperature is outside its qualified model domain", cell.cell_ref));
        }
        if !cell.minimum_voltage_v.is_finite()
            || cell.minimum_voltage_v <= 0.0
            || !is_digest(&cell.voltage_limit_evidence_digest)
            || !valid_class(&cell.voltage_limit_evidence_class)
        {
            errors.push(format!("cell {} voltage floor/evidence is invalid", cell.cell_ref));
        }
        if !cell.maximum_discharge_current_lower_a.is_finite()
            || cell.maximum_discharge_current_lower_a <= 0.0
            || !is_digest(&cell.current_limit_evidence_digest)
            || !valid_class(&cell.current_limit_evidence_class)
        {
            errors.push(format!("cell {} current rating/evidence is invalid", cell.cell_ref));
        }
        if !cell.thermal_resistance_upper_k_per_w.is_finite()
            || cell.thermal_resistance_upper_k_per_w <= 0.0
            || !cell.thermal_capacity_lower_j_per_k.is_finite()
            || cell.thermal_capacity_lower_j_per_k <= 0.0
            || !is_digest(&cell.thermal_evidence_digest)
            || !valid_class(&cell.thermal_evidence_class)
            || !cell.maximum_temperature_lower_c.is_finite()
        {
            errors.push(format!("cell {} thermal parameters/evidence are invalid", cell.cell_ref));
        }
    }
    errors
}

#[derive(Debug, Clone)]
struct RawCellLimit {
    cell_ref: String,
    voltage_limit_a: f64,
    current_limit_a: f64,
    thermal_limit_a: f64,
    admissible_limit_a: f64,
    limiting_constraint: String,
}

fn cell_limit(cell: &Cell, profile: &Profile) -> RawCellLimit {
    let voltage_limit = ((cell.ocv_lower_v - cell.minimum_voltage_v)
        / cell.resistance_upper_ohm)
        .max(0.0);
    let current_limit = cell.maximum_discharge_current_lower_a;
    let temperature_upper =
        cell.temperature_estimate_c + cell.temperature_uncertainty_c;
    let baseline_temperature = temperature_upper.max(profile.pulse.ambient_temperature_upper_c);
    let margin = cell.maximum_temperature_lower_c - baseline_temperature;
    let thermal_gain = cell.resistance_upper_ohm
        * cell.thermal_resistance_upper_k_per_w
        * (1.0
            - (-profile.pulse.duration_s
                / (cell.thermal_resistance_upper_k_per_w * cell.thermal_capacity_lower_j_per_k))
                .exp());
    let thermal_limit = if margin <= 0.0 || thermal_gain <= 0.0 || !thermal_gain.is_finite() {
        0.0
    } else {
        (margin / thermal_gain).sqrt()
    };
    let candidates = [
        (voltage_limit, 0, "cell_voltage_floor"),
        (current_limit, 1, "cell_current_rating"),
        (thermal_limit, 2, "cell_thermal_ceiling"),
    ];
    let selected = candidates
        .iter()
        .min_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        })
        .expect("three cell constraints exist");
    RawCellLimit {
        cell_ref: cell.cell_ref.clone(),
        voltage_limit_a: voltage_limit,
        current_limit_a: current_limit,
        thermal_limit_a: thermal_limit,
        admissible_limit_a: selected.0,
        limiting_constraint: selected.2.to_owned(),
    }
}

fn evidence(role: &str, digest: &str, class: Option<&str>) -> EvidenceRef {
    EvidenceRef {
        role: role.to_owned(),
        digest: digest.to_owned(),
        evidence_class: class.map(str::to_owned),
    }
}

fn build_evidence(profile: &Profile) -> Vec<EvidenceRef> {
    let mut rows = vec![
        evidence(
            "vehicle_configuration",
            &profile.asset.vehicle_configuration_digest,
            None,
        ),
        evidence(
            "pack_configuration",
            &profile.asset.pack_configuration_digest,
            None,
        ),
        evidence("cell_inventory", &profile.asset.cell_inventory_digest, None),
        evidence("topology_descriptor", &profile.topology.topology_digest, None),
        evidence("model", &profile.model.model_digest, None),
        evidence("model_equation_parameters", &profile.model.parameter_digest, None),
        evidence(
            "calibration",
            &profile.model.calibration_digest,
            Some(&profile.model.calibration_evidence_class),
        ),
        evidence(
            "domain_scope",
            &profile.model.domain_scope_digest,
            Some(&profile.model.domain_scope_evidence_class),
        ),
        evidence("model_protocol", &profile.model.protocol_digest, None),
        evidence(
            "ambient_temperature",
            &profile.pulse.ambient_evidence_digest,
            Some(&profile.pulse.ambient_evidence_class),
        ),
    ];
    if let Some(qualification) = &profile.topology.qualification {
        rows.push(evidence(
            "topology_qualification",
            &qualification.evidence_digest,
            Some(&qualification.evidence_class),
        ));
    }
    for cell in &profile.cells {
        rows.extend([
            evidence(
                &format!("cell_identity:{}", cell.cell_ref),
                &cell.cell_identity_digest,
                None,
            ),
            evidence(
                &format!("cell_configuration:{}", cell.cell_ref),
                &cell.cell_configuration_digest,
                None,
            ),
            evidence(
                &format!("ocv:{}", cell.cell_ref),
                &cell.ocv_evidence_digest,
                Some(&cell.ocv_evidence_class),
            ),
            evidence(
                &format!("resistance:{}", cell.cell_ref),
                &cell.resistance_evidence_digest,
                Some(&cell.resistance_evidence_class),
            ),
            evidence(
                &format!("soc:{}", cell.cell_ref),
                &cell.soc_evidence_digest,
                Some(&cell.soc_evidence_class),
            ),
            evidence(
                &format!("temperature:{}", cell.cell_ref),
                &cell.temperature_evidence_digest,
                Some(&cell.temperature_evidence_class),
            ),
            evidence(
                &format!("voltage_limit:{}", cell.cell_ref),
                &cell.voltage_limit_evidence_digest,
                Some(&cell.voltage_limit_evidence_class),
            ),
            evidence(
                &format!("current_limit:{}", cell.cell_ref),
                &cell.current_limit_evidence_digest,
                Some(&cell.current_limit_evidence_class),
            ),
            evidence(
                &format!("thermal:{}", cell.cell_ref),
                &cell.thermal_evidence_digest,
                Some(&cell.thermal_evidence_class),
            ),
        ]);
    }
    rows.sort_by(|left, right| {
        (&left.role, &left.digest, &left.evidence_class)
            .cmp(&(&right.role, &right.digest, &right.evidence_class))
    });
    rows
}

fn compute(profile: &Profile) -> PowerEnvelope {
    let mut limits: Vec<RawCellLimit> = profile
        .cells
        .iter()
        .map(|cell| cell_limit(cell, profile))
        .collect();
    limits.sort_by(|left, right| {
        left.admissible_limit_a
            .total_cmp(&right.admissible_limit_a)
            .then_with(|| left.cell_ref.cmp(&right.cell_ref))
    });
    let limiting = limits.first().expect("validated profile contains cells");
    let maximum_current = limiting.admissible_limit_a.max(0.0);
    let evidence = build_evidence(profile);

    if maximum_current <= EPSILON {
        let cells = limits
            .iter()
            .map(|limit| CellLimit {
                cell_ref: limit.cell_ref.clone(),
                voltage_limited_current_a: limit.voltage_limit_a,
                current_rating_limit_a: limit.current_limit_a,
                thermal_limited_current_a: limit.thermal_limit_a,
                admissible_current_a: limit.admissible_limit_a,
                limiting_constraint: limit.limiting_constraint.clone(),
                terminal_voltage_lower_v_at_selected_current: None,
                temperature_upper_c_at_selected_current: None,
            })
            .collect();
        return PowerEnvelope {
            profile_id: profile.profile_id.clone(),
            status: "NO_ADMISSIBLE_DISCHARGE".to_owned(),
            pulse_duration_s: profile.pulse.duration_s,
            maximum_admissible_current_a: 0.0,
            selected_current_a: None,
            pack_terminal_voltage_lower_v: None,
            conservative_terminal_power_lower_w: None,
            limiting_cell: limiting.cell_ref.clone(),
            limiting_constraint: limiting.limiting_constraint.clone(),
            topology_qualification_ref: profile.topology.qualification.as_ref().unwrap().qualification_ref.clone(),
            topology_authority_ref: profile.topology.qualification.as_ref().unwrap().authority_ref.clone(),
            model_ref: profile.model.model_ref.clone(),
            model_digest: profile.model.model_digest.clone(),
            cells,
            evidence,
        };
    }

    let sum_ocv_lower: f64 = profile.cells.iter().map(|cell| cell.ocv_lower_v).sum();
    let sum_resistance_upper: f64 = profile
        .cells
        .iter()
        .map(|cell| cell.resistance_upper_ohm)
        .sum();
    let maximum_power_current = sum_ocv_lower / (2.0 * sum_resistance_upper);
    let selected_current = maximum_current.min(maximum_power_current);
    let pack_voltage_lower = sum_ocv_lower - selected_current * sum_resistance_upper;
    let power_lower = selected_current * pack_voltage_lower;

    let mut cells = Vec::new();
    for cell in &profile.cells {
        let temperature_upper = cell.temperature_estimate_c + cell.temperature_uncertainty_c;
        let baseline_temperature =
            temperature_upper.max(profile.pulse.ambient_temperature_upper_c);
        let tau = cell.thermal_resistance_upper_k_per_w * cell.thermal_capacity_lower_j_per_k;
        let thermal_gain = cell.resistance_upper_ohm
            * cell.thermal_resistance_upper_k_per_w
            * (1.0 - (-profile.pulse.duration_s / tau).exp());
        let predicted_temperature = baseline_temperature
            + selected_current.powi(2) * thermal_gain;
        cells.push(CellLimit {
            cell_ref: cell.cell_ref.clone(),
            voltage_limited_current_a: ((cell.ocv_lower_v - cell.minimum_voltage_v)
                / cell.resistance_upper_ohm)
                .max(0.0),
            current_rating_limit_a: cell.maximum_discharge_current_lower_a,
            thermal_limited_current_a: cell_limit(cell, profile).thermal_limit_a,
            admissible_current_a: cell_limit(cell, profile).admissible_limit_a,
            limiting_constraint: cell_limit(cell, profile).limiting_constraint,
            terminal_voltage_lower_v_at_selected_current: Some(
                cell.ocv_lower_v - selected_current * cell.resistance_upper_ohm,
            ),
            temperature_upper_c_at_selected_current: Some(predicted_temperature),
        });
    }
    cells.sort_by(|left, right| left.cell_ref.cmp(&right.cell_ref));
    PowerEnvelope {
        profile_id: profile.profile_id.clone(),
        status: "COMPUTED".to_owned(),
        pulse_duration_s: profile.pulse.duration_s,
        maximum_admissible_current_a: maximum_current,
        selected_current_a: Some(selected_current),
        pack_terminal_voltage_lower_v: Some(pack_voltage_lower),
        conservative_terminal_power_lower_w: Some(power_lower),
        limiting_cell: limiting.cell_ref.clone(),
        limiting_constraint: limiting.limiting_constraint.clone(),
        topology_qualification_ref: profile.topology.qualification.as_ref().unwrap().qualification_ref.clone(),
        topology_authority_ref: profile.topology.qualification.as_ref().unwrap().authority_ref.clone(),
        model_ref: profile.model.model_ref.clone(),
        model_digest: profile.model.model_digest.clone(),
        cells,
        evidence,
    }
}

fn canonicalize_corpus(value: &mut Value) {
    let Some(cases) = value.get_mut("cases").and_then(Value::as_array_mut) else {
        return;
    };
    cases.sort_by(|left, right| {
        left.get("case_id")
            .and_then(Value::as_str)
            .cmp(&right.get("case_id").and_then(Value::as_str))
    });
    for case in cases {
        if let Some(cells) = case
            .get_mut("profile")
            .and_then(|profile| profile.get_mut("cells"))
            .and_then(Value::as_array_mut)
        {
            cells.sort_by(|left, right| {
                left.get("cell_ref")
                    .and_then(Value::as_str)
                    .cmp(&right.get("cell_ref").and_then(Value::as_str))
            });
        }
    }
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<CorpusReport, String> {
    let validator = schema_validator(schema_bytes)?;
    let mut corpus_value: Value =
        serde_json::from_slice(corpus_bytes).map_err(|error| format!("corpus JSON: {error}"))?;
    canonicalize_corpus(&mut corpus_value);
    let canonical = serde_json::to_vec(&corpus_value)
        .map_err(|error| format!("canonical corpus: {error}"))?;
    let corpus: Corpus =
        serde_json::from_slice(&canonical).map_err(|error| format!("typed corpus: {error}"))?;
    if corpus.corpus_version != CORPUS_VERSION || corpus.schema_id != SCHEMA_ID {
        return Err("corpus version or schema identity mismatch".to_owned());
    }

    let mut seen = BTreeSet::new();
    let mut cases = Vec::new();
    let mut computed_count = 0usize;
    let mut no_discharge_count = 0usize;
    let mut rejected_count = 0usize;
    for case in corpus.cases {
        if !nonempty(&case.case_id) || !seen.insert(case.case_id.clone()) {
            return Err(format!("case identity is empty or duplicated: {}", case.case_id));
        }
        if !matches!(
            case.expectation.as_str(),
            "computed" | "no_admissible_discharge" | "reject"
        ) {
            return Err(format!("case {} has unknown expectation", case.case_id));
        }
        let schema_errors: Vec<String> = validator
            .iter_errors(&case.profile)
            .map(|error| error.to_string())
            .collect();
        let outcome = if schema_errors.is_empty() {
            match serde_json::from_value::<Profile>(case.profile) {
                Ok(profile) => {
                    let errors = validate_profile(&profile);
                    if errors.is_empty() {
                        let envelope = compute(&profile);
                        let numeric_values = [
                            Some(envelope.maximum_admissible_current_a),
                            envelope.selected_current_a,
                            envelope.pack_terminal_voltage_lower_v,
                            envelope.conservative_terminal_power_lower_w,
                        ];
                        if numeric_values
                            .iter()
                            .flatten()
                            .any(|value| !value.is_finite() || *value < 0.0)
                        {
                            Err(vec![
                                "power envelope contains a non-finite or negative output"
                                    .to_owned(),
                            ])
                        } else {
                            Ok(envelope)
                        }
                    } else {
                        Err(errors)
                    }
                }
                Err(error) => Err(vec![format!("typed profile parse: {error}")]),
            }
        } else {
            Err(schema_errors)
        };

        let (envelope, errors) = match outcome {
            Ok(envelope) => {
                let expected_status = match case.expectation.as_str() {
                    "computed" => "COMPUTED",
                    "no_admissible_discharge" => "NO_ADMISSIBLE_DISCHARGE",
                    _ => unreachable!(),
                };
                if envelope.status != expected_status
                    || case.expected_limiting_cell.as_deref() != Some(envelope.limiting_cell.as_str())
                    || case.expected_limiting_constraint.as_deref()
                        != Some(envelope.limiting_constraint.as_str())
                    || case.expected_max_current_a.is_none_or(|expected| {
                        !expected.is_finite()
                            || (envelope.maximum_admissible_current_a - expected).abs() > 1e-8
                    })
                    || match (case.expected_power_w, envelope.conservative_terminal_power_lower_w) {
                        (Some(expected), Some(actual)) => {
                            !expected.is_finite() || (actual - expected).abs() > 1e-6
                        }
                        (None, None) => false,
                        _ => true,
                    }
                {
                    return Err(format!(
                        "case {} known-truth disposition/limiting-cell/current/power mismatch",
                        case.case_id
                    ));
                }
                if envelope.status == "COMPUTED" {
                    computed_count += 1;
                } else {
                    no_discharge_count += 1;
                }
                (Some(envelope), Vec::new())
            }
            Err(errors) => {
                if case.expectation != "reject"
                    || case.expected_limiting_cell.is_some()
                    || case.expected_limiting_constraint.is_some()
                    || case.expected_max_current_a.is_some()
                    || case.expected_power_w.is_some()
                {
                    return Err(format!("case {} expected a valid envelope but input was rejected: {}", case.case_id, errors.join("; ")));
                }
                rejected_count += 1;
                (None, errors)
            }
        };
        cases.push(CaseAudit {
            case_id: case.case_id,
            expectation: case.expectation,
            accepted: envelope.is_some(),
            errors,
            envelope,
        });
    }

    let payload = ReportPayload {
        corpus_version: CORPUS_VERSION.to_owned(),
        schema_id: SCHEMA_ID.to_owned(),
        schema_sha256: sha256(schema_bytes),
        canonical_corpus_sha256: sha256(&canonical),
        case_count: cases.len(),
        computed_count,
        no_discharge_count,
        rejected_count,
        cases,
    };
    let bytes =
        serde_json::to_vec(&payload).map_err(|error| format!("report serialization: {error}"))?;
    Ok(CorpusReport {
        payload,
        report_sha256: sha256(&bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_cell_limited_power_envelope_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/cell_limited_power_envelope_scenarios_v0_1.json");

    fn report() -> CorpusReport {
        audit_corpus(SCHEMA, CORPUS).expect("frozen power-envelope corpus")
    }

    #[test]
    fn frozen_corpus_computes_four_rejects_four_and_has_one_no_discharge() {
        let report = report();
        assert_eq!(report.payload.case_count, 9);
        assert_eq!(report.payload.computed_count, 4);
        assert_eq!(report.payload.no_discharge_count, 1);
        assert_eq!(report.payload.rejected_count, 4);
    }

    #[test]
    fn weakest_cell_sets_series_current_and_terminal_power() {
        let result = report();
        let envelope = result.payload.cases.iter()
            .find(|case| case.case_id == "weak-cell-voltage-limited")
            .and_then(|case| case.envelope.as_ref())
            .expect("weak-cell envelope");
        assert_eq!(envelope.limiting_cell, "cell-003");
        assert_eq!(envelope.limiting_constraint, "cell_voltage_floor");
        assert!(envelope.selected_current_a.unwrap() <= envelope.maximum_admissible_current_a);
    }

    #[test]
    fn thermal_and_current_rating_constraints_are_not_hidden_by_average_resistance() {
        let result = report();
        let thermal = result.payload.cases.iter()
            .find(|case| case.case_id == "thermal-limited")
            .and_then(|case| case.envelope.as_ref())
            .expect("thermal envelope");
        assert_eq!(thermal.limiting_constraint, "cell_thermal_ceiling");
        let current = result.payload.cases.iter()
            .find(|case| case.case_id == "current-rating-limited")
            .and_then(|case| case.envelope.as_ref())
            .expect("current-limited envelope");
        assert_eq!(current.limiting_cell, "cell-002");
        assert_eq!(current.limiting_constraint, "cell_current_rating");
    }

    #[test]
    fn no_thermal_margin_never_emits_positive_power() {
        let result = report();
        let envelope = result.payload.cases.iter()
            .find(|case| case.case_id == "no-admissible-discharge-when-thermal-margin-exhausted")
            .and_then(|case| case.envelope.as_ref())
            .expect("no-discharge result");
        assert_eq!(envelope.status, "NO_ADMISSIBLE_DISCHARGE");
        assert_eq!(envelope.maximum_admissible_current_a, 0.0);
        assert!(envelope.conservative_terminal_power_lower_w.is_none());
    }

    #[test]
    fn every_cell_hard_limit_bounds_the_common_series_current() {
        let audited = report();
        let case = audited.payload.cases.iter()
            .find(|case| case.case_id == "thermal-limited")
            .expect("thermal-limited case");
        let envelope = case.envelope.as_ref().expect("computed envelope");
        let corpus_value: Value = serde_json::from_slice(CORPUS).expect("corpus JSON");
        let thermal_profile = corpus_value["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["case_id"] == "thermal-limited")
            .expect("thermal-limited fixture")["profile"]
            .clone();
        let profile: Profile =
            serde_json::from_value(thermal_profile).expect("thermal profile");
        for cell in &envelope.cells {
            assert!(envelope.maximum_admissible_current_a
                <= cell.voltage_limited_current_a + EPSILON);
            assert!(envelope.maximum_admissible_current_a
                <= cell.current_rating_limit_a + EPSILON);
            assert!(envelope.maximum_admissible_current_a
                <= cell.thermal_limited_current_a + EPSILON);
            assert!(cell.terminal_voltage_lower_v_at_selected_current.unwrap()
                >= profile.cells.iter().find(|input| input.cell_ref == cell.cell_ref)
                    .unwrap().minimum_voltage_v - EPSILON);
            assert!(cell.temperature_upper_c_at_selected_current.unwrap()
                <= profile.cells.iter().find(|input| input.cell_ref == cell.cell_ref)
                    .unwrap().maximum_temperature_lower_c + EPSILON);
        }
    }

    #[test]
    fn power_receipt_preserves_model_calibration_domain_and_limit_evidence_classes() {
        let audited = report();
        let envelope = audited.payload.cases.iter()
            .find(|case| case.case_id == "equal-series-reference")
            .and_then(|case| case.envelope.as_ref())
            .expect("reference envelope");
        assert!(envelope.evidence.iter().any(|row| {
            row.role == "calibration" && row.evidence_class.as_deref() == Some("synthetic_fixture")
        }));
        assert!(envelope.evidence.iter().any(|row| {
            row.role == "voltage_limit:cell-001"
                && row.evidence_class.as_deref() == Some("synthetic_fixture")
        }));
        assert!(envelope.evidence.iter().any(|row| row.role == "cell_inventory"));
        assert_eq!(envelope.topology_authority_ref, "synthetic-authority");
    }

    #[test]
    fn cell_order_permutation_preserves_canonical_report_digest() {
        let original = report();
        let mut value: Value = serde_json::from_slice(CORPUS).expect("corpus JSON");
        value["cases"].as_array_mut().unwrap().reverse();
        for case in value["cases"].as_array_mut().unwrap() {
            case["profile"]["cells"].as_array_mut().unwrap().reverse();
        }
        let permuted = serde_json::to_vec(&value).expect("permuted corpus");
        let replay = audit_corpus(SCHEMA, &permuted).expect("permuted power-envelope audit");
        assert_eq!(original.report_sha256, replay.report_sha256);
        assert_eq!(
            original.payload.canonical_corpus_sha256,
            replay.payload.canonical_corpus_sha256
        );
    }
}
