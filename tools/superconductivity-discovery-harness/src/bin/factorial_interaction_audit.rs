use jsonschema::validator_for;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
const CLAIM_CEILING: &str = "synthetic-factorial-design-and-contrast-arithmetic-only";
const DEFAULT_SCHEMA: &str = "data/schemas/sc_factorial_interaction_v0_1.schema.json";
const DEFAULT_FIXTURE: &str =
    "tools/superconductivity-discovery-harness/fixtures/factorial_interaction_smoke_v0_1.json";
const RESPONSE_CHANNELS: [&str; 2] = [
    "local_pairing_gap_mev",
    "bulk_superfluid_stiffness_mev",
];

#[derive(Clone, Debug)]
struct Factor {
    factor_id: String,
    off_level: f64,
    on_level: f64,
    unit: String,
}

#[derive(Clone, Debug)]
struct Arm {
    arm_id: String,
    specimen_id: String,
    material_state_id: String,
    factor_a_on: bool,
    factor_b_on: bool,
    randomization_order: u64,
    outcomes: BTreeMap<String, f64>,
}

#[derive(Clone, Debug)]
struct Block {
    block_id: String,
    batch_id: String,
    campaign_id: String,
    arms: BTreeMap<String, Arm>,
}

fn fail(path: &str, message: impl AsRef<str>) -> String {
    format!("{path}: {}", message.as_ref())
}

fn object<'a>(value: &'a Value, path: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| fail(path, "must be an object"))
}

fn array<'a>(value: &'a Value, path: &str) -> Result<&'a Vec<Value>, String> {
    value
        .as_array()
        .ok_or_else(|| fail(path, "must be an array"))
}

fn string_field<'a>(value: &'a Value, key: &str, path: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| fail(path, format!("{key} must be a non-empty string")))
}

fn bool_field(value: &Value, key: &str, path: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| fail(path, format!("{key} must be boolean")))
}

fn number_field(value: &Value, key: &str, path: &str) -> Result<f64, String> {
    let number = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| fail(path, format!("{key} must be a finite number")))?;
    if !number.is_finite() {
        return Err(fail(path, format!("{key} must be a finite number")));
    }
    Ok(number)
}

fn integer_field(value: &Value, key: &str, path: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| fail(path, format!("{key} must be a non-negative integer")))
}

fn digest(value: &str, path: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(fail(path, "must be lowercase 64-character SHA-256 hex"));
    }
    Ok(())
}

fn parse_factor(value: &Value, path: &str) -> Result<Factor, String> {
    let factor = Factor {
        factor_id: string_field(value, "factor_id", path)?.to_owned(),
        off_level: number_field(value, "off_level", path)?,
        on_level: number_field(value, "on_level", path)?,
        unit: string_field(value, "unit", path)?.to_owned(),
    };
    if (factor.on_level - factor.off_level).abs() <= f64::EPSILON {
        return Err(fail(path, "off_level and on_level must differ"));
    }
    Ok(factor)
}

fn expected_arm_id(a_on: bool, b_on: bool) -> &'static str {
    match (a_on, b_on) {
        (false, false) => "control",
        (true, false) => "a_only",
        (false, true) => "b_only",
        (true, true) => "combined",
    }
}

fn close_enough(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9_f64.max(left.abs().max(right.abs()) * 1.0e-9)
}

fn parse_arm(
    value: &Value,
    index: usize,
    target_material_id: &str,
    factor_a: &Factor,
    factor_b: &Factor,
    specimen_ids: &mut HashSet<String>,
    state_ids: &mut HashSet<String>,
) -> Result<Arm, String> {
    let path = format!("arm[{index}]");
    let arm_id = string_field(value, "arm_id", &path)?.to_owned();
    let specimen_id = string_field(value, "specimen_id", &path)?.to_owned();
    let material_id = string_field(value, "material_id", &path)?;
    let material_state_id = string_field(value, "material_state_id", &path)?.to_owned();
    if material_id != target_material_id {
        return Err(fail(&path, "arm material_id does not match target_material"));
    }
    if !specimen_ids.insert(specimen_id.clone()) {
        return Err(fail(&path, format!("specimen_id {specimen_id} is reused")));
    }
    if !state_ids.insert(material_state_id.clone()) {
        return Err(fail(
            &path,
            format!("material_state_id {material_state_id} is reused"),
        ));
    }

    let factor_a_on = bool_field(value, "factor_a_on", &path)?;
    let factor_b_on = bool_field(value, "factor_b_on", &path)?;
    if arm_id != expected_arm_id(factor_a_on, factor_b_on) {
        return Err(fail(
            &path,
            "arm_id does not match the declared factor A/B assignment",
        ));
    }

    let randomization_order = integer_field(value, "randomization_order", &path)?;
    if !(1..=4).contains(&randomization_order) {
        return Err(fail(&path, "randomization_order must be in 1..=4"));
    }
    if string_field(value, "assignment_state", &path)? != "sensor_verified" {
        return Err(fail(
            &path,
            "assignment_state must be sensor_verified",
        ));
    }

    let readback = value
        .get("delivery_readback")
        .ok_or_else(|| fail(&path, "delivery_readback is required"))?;
    if string_field(readback, "verification", &format!("{path}.delivery_readback"))?
        != "sensor_verified"
    {
        return Err(fail(
            &path,
            "delivery_readback.verification must be sensor_verified",
        ));
    }
    let a_value = number_field(readback, "factor_a_value", &format!("{path}.delivery_readback"))?;
    let b_value = number_field(readback, "factor_b_value", &format!("{path}.delivery_readback"))?;
    let a_unit = string_field(readback, "factor_a_unit", &format!("{path}.delivery_readback"))?;
    let b_unit = string_field(readback, "factor_b_unit", &format!("{path}.delivery_readback"))?;
    if a_unit != factor_a.unit || b_unit != factor_b.unit {
        return Err(fail(
            &path,
            "delivery readback units do not match declared factor units",
        ));
    }
    let expected_a = if factor_a_on {
        factor_a.on_level
    } else {
        factor_a.off_level
    };
    let expected_b = if factor_b_on {
        factor_b.on_level
    } else {
        factor_b.off_level
    };
    if !close_enough(a_value, expected_a) || !close_enough(b_value, expected_b) {
        return Err(fail(
            &path,
            "sensor-verified delivered levels do not match the declared factorial assignment",
        ));
    }
    digest(
        string_field(readback, "source_sha256", &format!("{path}.delivery_readback"))?,
        &format!("{path}.delivery_readback.source_sha256"),
    )?;

    let outcomes = array(
        value
            .get("outcomes")
            .ok_or_else(|| fail(&path, "outcomes are required"))?,
        &format!("{path}.outcomes"),
    )?;
    let mut outcome_map = BTreeMap::new();
    for (outcome_index, outcome) in outcomes.iter().enumerate() {
        let outcome_path = format!("{path}.outcomes[{outcome_index}]");
        let channel = string_field(outcome, "response_channel", &outcome_path)?.to_owned();
        if !RESPONSE_CHANNELS.contains(&channel.as_str()) {
            return Err(fail(&outcome_path, format!("unsupported response channel {channel}")));
        }
        let unit = string_field(outcome, "unit", &outcome_path)?;
        if unit != "meV" {
            return Err(fail(
                &outcome_path,
                "v0.1 pairing-gap and superfluid-stiffness outcomes must use meV",
            ));
        }
        if string_field(outcome, "evidence_authority", &outcome_path)?
            != "synthetic_measurement"
        {
            return Err(fail(
                &outcome_path,
                "synthetic fixture outcomes require synthetic_measurement authority",
            ));
        }
        let response = number_field(outcome, "value", &outcome_path)?;
        if outcome_map.insert(channel.clone(), response).is_some() {
            return Err(fail(
                &outcome_path,
                format!("duplicate response channel {channel}"),
            ));
        }
    }
    for channel in RESPONSE_CHANNELS {
        if !outcome_map.contains_key(channel) {
            return Err(fail(&path, format!("missing response channel {channel}")));
        }
    }

    Ok(Arm {
        arm_id,
        specimen_id,
        material_state_id,
        factor_a_on,
        factor_b_on,
        randomization_order,
        outcomes: outcome_map,
    })
}

fn contrast(block: &Block, response_channel: &str) -> Result<f64, String> {
    let control = block
        .arms
        .get("control")
        .ok_or_else(|| format!("{}: missing control arm", block.block_id))?;
    let a_only = block
        .arms
        .get("a_only")
        .ok_or_else(|| format!("{}: missing a_only arm", block.block_id))?;
    let b_only = block
        .arms
        .get("b_only")
        .ok_or_else(|| format!("{}: missing b_only arm", block.block_id))?;
    let combined = block
        .arms
        .get("combined")
        .ok_or_else(|| format!("{}: missing combined arm", block.block_id))?;
    Ok(combined.outcomes[response_channel] - a_only.outcomes[response_channel]
        - b_only.outcomes[response_channel]
        + control.outcomes[response_channel])
}

fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator = validator_for(&schema)
        .map_err(|error| format!("schema compilation failed: {error}"))?;
    if let Some(error) = validator.iter_errors(&fixture).next() {
        return Err(format!(
            "fixture schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }
    if fixture.get("evidence_class").and_then(Value::as_str) != Some("synthetic") {
        return Err("factorial interaction auditor accepts synthetic fixtures only".to_owned());
    }
    if fixture.get("claim_ceiling").and_then(Value::as_str) != Some(CLAIM_CEILING) {
        return Err("fixture claim ceiling does not match factorial interaction audit".to_owned());
    }
    let benchmark_id = string_field(&fixture, "benchmark_id", "$")?;
    let material = fixture
        .get("target_material")
        .ok_or_else(|| "target_material is required".to_owned())?;
    let material_id = string_field(material, "material_id", "target_material")?;
    let factor_a = parse_factor(
        fixture
            .get("factor_a")
            .ok_or_else(|| "factor_a is required".to_owned())?,
        "factor_a",
    )?;
    let factor_b = parse_factor(
        fixture
            .get("factor_b")
            .ok_or_else(|| "factor_b is required".to_owned())?,
        "factor_b",
    )?;
    if factor_a.factor_id == factor_b.factor_id {
        return Err("factor_a and factor_b must have distinct factor IDs".to_owned());
    }
    let condition_profile = fixture
        .get("condition_profile")
        .ok_or_else(|| "condition_profile is required".to_owned())?;
    let condition_profile_id =
        string_field(condition_profile, "condition_profile_id", "condition_profile")?;
    let condition_temperature = number_field(
        condition_profile,
        "temperature_k",
        "condition_profile",
    )?;
    if condition_temperature <= 0.0 {
        return Err("condition_profile.temperature_k must be positive".to_owned());
    }
    if number_field(
        condition_profile,
        "field_magnitude_t",
        "condition_profile",
    )? < 0.0
    {
        return Err("condition_profile.field_magnitude_t must be non-negative".to_owned());
    }
    string_field(
        condition_profile,
        "field_orientation",
        "condition_profile",
    )?;
    let measurement_protocol =
        string_field(condition_profile, "measurement_protocol_id", "condition_profile")?;

    let block_values = array(
        fixture
            .get("blocks")
            .ok_or_else(|| "blocks are required".to_owned())?,
        "blocks",
    )?;
    if block_values.len() < 2 {
        return Err("factorial smoke audit requires at least two independent blocks".to_owned());
    }
    let mut block_ids = BTreeSet::new();
    let mut batch_ids = BTreeSet::new();
    let mut campaign_ids = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    let mut specimen_ids = HashSet::new();
    let mut state_ids = HashSet::new();
    let mut blocks = Vec::with_capacity(block_values.len());

    for (block_index, block_value) in block_values.iter().enumerate() {
        let path = format!("blocks[{block_index}]");
        let block_id = string_field(block_value, "block_id", &path)?.to_owned();
        let batch_id = string_field(block_value, "batch_id", &path)?.to_owned();
        let campaign_id = string_field(block_value, "campaign_id", &path)?.to_owned();
        let laboratory_id = string_field(block_value, "laboratory_id", &path)?;
        let instrument_id = string_field(block_value, "instrument_id", &path)?;
        let protocol_id = string_field(block_value, "protocol_id", &path)?;
        if !block_ids.insert(block_id.clone()) {
            return Err(format!("duplicate block_id {block_id}"));
        }
        if !batch_ids.insert(batch_id.clone()) {
            return Err(format!("batch_id {batch_id} is reused across independent blocks"));
        }
        if !campaign_ids.insert(campaign_id.clone()) {
            return Err(format!("campaign_id {campaign_id} is reused across independent blocks"));
        }
        if !seeds.insert(integer_field(block_value, "randomization_seed", &path)?) {
            return Err(format!("{path}: randomization_seed must be unique across blocks"));
        }
        if string_field(block_value, "condition_profile_id", &path)? != condition_profile_id {
            return Err(format!("{path}: block condition profile differs from benchmark profile"));
        }
        if protocol_id != measurement_protocol {
            return Err(format!("{path}: block protocol differs from declared measurement protocol"));
        }
        if laboratory_id.is_empty() || instrument_id.is_empty() {
            return Err(format!("{path}: laboratory/instrument identity is missing"));
        }

        let arms_value = array(
            block_value
                .get("arms")
                .ok_or_else(|| fail(&path, "arms are required"))?,
            &format!("{path}.arms"),
        )?;
        if arms_value.len() != 4 {
            return Err(format!("{path}: exactly four factorial arms are required"));
        }
        let mut order = BTreeSet::new();
        let mut arms = BTreeMap::new();
        for (arm_index, arm_value) in arms_value.iter().enumerate() {
            let arm = parse_arm(
                arm_value,
                arm_index,
                material_id,
                &factor_a,
                &factor_b,
                &mut specimen_ids,
                &mut state_ids,
            )?;
            if !order.insert(arm.randomization_order) {
                return Err(format!(
                    "{path}: randomization_order {} is duplicated",
                    arm.randomization_order
                ));
            }
            if arms.insert(arm.arm_id.clone(), arm).is_some() {
                return Err(format!("{path}: duplicate factorial arm"));
            }
        }
        if order != BTreeSet::from([1, 2, 3, 4]) {
            return Err(format!("{path}: randomization_order must contain exactly 1,2,3,4"));
        }
        if arms.len() != 4
            || !["control", "a_only", "b_only", "combined"]
                .iter()
                .all(|arm_id| arms.contains_key(*arm_id))
        {
            return Err(format!("{path}: all four factorial arms are required"));
        }
        blocks.push(Block {
            block_id,
            batch_id,
            campaign_id,
            arms,
        });
    }

    let mut response_reports = Map::new();
    let expected_truth = fixture
        .get("synthetic_truth")
        .and_then(|value| value.get("expected_mean_interaction_by_channel"))
        .ok_or_else(|| "synthetic_truth.expected_mean_interaction_by_channel is required".to_owned())?;
    for channel in RESPONSE_CHANNELS {
        let mut per_block = Vec::with_capacity(blocks.len());
        let mut sum = 0.0;
        for block in &blocks {
            let interaction = contrast(block, channel)?;
            sum += interaction;
            per_block.push(json!({
                "block_id": block.block_id,
                "batch_id": block.batch_id,
                "campaign_id": block.campaign_id,
                "interaction_contrast": interaction
            }));
        }
        let mean_interaction = sum / blocks.len() as f64;
        let expected = number_field(expected_truth, channel, "synthetic_truth")?;
        let absolute_error = (mean_interaction - expected).abs();
        if !close_enough(mean_interaction, expected) {
            return Err(format!(
                "{channel}: computed synthetic contrast {mean_interaction} does not match generator truth {expected}"
            ));
        }
        response_reports.insert(
            channel.to_owned(),
            json!({
                "response_unit": "meV",
                "per_block_interaction_contrast": per_block,
                "mean_interaction_contrast": mean_interaction,
                "synthetic_generator_truth": expected,
                "absolute_error_vs_generator_truth": absolute_error
            }),
        );
    }

    Ok(json!({
        "schema_version": "sc-factorial-interaction-receipt.v0.1",
        "benchmark_id": benchmark_id,
        "schema_sha256": format!("{:x}", Sha256::digest(schema_bytes)),
        "fixture_sha256": format!("{:x}", Sha256::digest(fixture_bytes)),
        "target_material_id": material_id,
        "factor_a_id": factor_a.factor_id,
        "factor_b_id": factor_b.factor_id,
        "condition_profile_id": condition_profile_id,
        "independent_block_count": blocks.len(),
        "specimen_count": specimen_ids.len(),
        "response_channels": response_reports,
        "claim_ceiling": CLAIM_CEILING
    }))
}

fn parse_args() -> Result<(PathBuf, PathBuf, Option<PathBuf>), String> {
    let mut schema = PathBuf::from(DEFAULT_SCHEMA);
    let mut fixture = PathBuf::from(DEFAULT_FIXTURE);
    let mut output = None;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for argument {argument}"))?;
        match argument.as_str() {
            "--schema" => schema = PathBuf::from(value),
            "--fixture" => fixture = PathBuf::from(value),
            "--out" => output = Some(PathBuf::from(value)),
            other => return Err(format!("unsupported argument {other}")),
        }
    }
    Ok((schema, fixture, output))
}

fn run() -> Result<(), String> {
    let (schema_path, fixture_path, output_path) = parse_args()?;
    let schema_bytes = fs::read(&schema_path)
        .map_err(|error| format!("cannot read schema {}: {error}", schema_path.display()))?;
    let fixture_bytes = fs::read(&fixture_path)
        .map_err(|error| format!("cannot read fixture {}: {error}", fixture_path.display()))?;
    let report = audit(&schema_bytes, &fixture_bytes)?;
    let encoded = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("cannot serialize receipt: {error}"))?;
    if let Some(path) = output_path {
        fs::write(&path, &encoded)
            .map_err(|error| format!("cannot write receipt {}: {error}", path.display()))?;
    } else {
        println!(
            "{}",
            String::from_utf8(encoded).map_err(|error| error.to_string())?
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sc_factorial_interaction_audit: {error}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_bytes() -> &'static [u8] {
        include_bytes!("../../../../data/schemas/sc_factorial_interaction_v0_1.schema.json")
    }

    fn fixture_bytes() -> &'static [u8] {
        include_bytes!("../../fixtures/factorial_interaction_smoke_v0_1.json")
    }

    fn fixture_value() -> Value {
        serde_json::from_slice(fixture_bytes()).expect("fixture JSON must parse")
    }

    fn audit_fixture(fixture: &Value) -> Result<Value, String> {
        let bytes = serde_json::to_vec(fixture).expect("fixture serializes");
        audit(schema_bytes(), &bytes)
    }

    fn run_fixture() -> Value {
        audit(schema_bytes(), fixture_bytes()).expect("factorial synthetic fixture must pass")
    }

    #[test]
    fn synthetic_pairing_and_coherence_contrasts_are_separate() {
        let receipt = run_fixture();
        assert_eq!(receipt["independent_block_count"], 2);
        assert_eq!(receipt["specimen_count"], 8);
        assert_eq!(
            receipt["response_channels"]["local_pairing_gap_mev"]["mean_interaction_contrast"],
            3.0
        );
        assert_eq!(
            receipt["response_channels"]["bulk_superfluid_stiffness_mev"]
                ["mean_interaction_contrast"],
            0.0
        );
        assert_eq!(
            receipt["claim_ceiling"],
            CLAIM_CEILING
        );
    }

    #[test]
    fn requires_complete_factorial_arms() {
        let mut fixture = fixture_value();
        fixture["blocks"][0]["arms"].as_array_mut().expect("array").pop();
        assert!(audit_fixture(&fixture).unwrap_err().contains("exactly four factorial arms"));
    }

    #[test]
    fn rejects_duplicate_factorial_arm() {
        let mut fixture = fixture_value();
        fixture["blocks"][0]["arms"][3]["arm_id"] = json!("control");
        assert!(audit_fixture(&fixture).unwrap_err().contains("duplicate factorial arm"));
    }

    #[test]
    fn rejects_same_specimen_reused_across_arms() {
        let mut fixture = fixture_value();
        fixture["blocks"][0]["arms"][1]["specimen_id"] =
            fixture["blocks"][0]["arms"][0]["specimen_id"].clone();
        assert!(audit_fixture(&fixture).unwrap_err().contains("specimen_id"));
    }

    #[test]
    fn rejects_nominal_material_mismatch() {
        let mut fixture = fixture_value();
        fixture["blocks"][1]["arms"][2]["material_id"] = json!("OTHER-MATERIAL");
        assert!(audit_fixture(&fixture).unwrap_err().contains("does not match target_material"));
    }

    #[test]
    fn rejects_reused_batch_or_campaign_as_independent_block() {
        let mut fixture = fixture_value();
        fixture["blocks"][1]["batch_id"] = json!("BATCH-001");
        assert!(audit_fixture(&fixture).unwrap_err().contains("batch_id"));
    }

    #[test]
    fn rejects_duplicate_randomization_order() {
        let mut fixture = fixture_value();
        fixture["blocks"][0]["arms"][1]["randomization_order"] = json!(3);
        assert!(audit_fixture(&fixture).unwrap_err().contains("randomization_order"));
    }

    #[test]
    fn refuses_commanded_only_treatment_assignment() {
        let mut fixture = fixture_value();
        fixture["blocks"][0]["arms"][0]["assignment_state"] = json!("commanded_only");
        assert!(audit_fixture(&fixture).unwrap_err().contains("sensor_verified"));
    }

    #[test]
    fn refuses_commanded_or_unverified_delivery_readback() {
        let mut fixture = fixture_value();
        fixture["blocks"][0]["arms"][0]["delivery_readback"]["verification"] = json!("commanded");
        assert!(audit_fixture(&fixture).unwrap_err().contains("must be sensor_verified"));
    }

    #[test]
    fn refuses_wrong_delivered_factor_level() {
        let mut fixture = fixture_value();
        fixture["blocks"][0]["arms"][3]["delivery_readback"]["factor_a_value"] = json!(0.0);
        assert!(audit_fixture(&fixture).unwrap_err().contains("delivered levels do not match"));
    }

    #[test]
    fn refuses_malformed_assignment_receipt_digest() {
        let mut fixture = fixture_value();
        fixture["blocks"][0]["arms"][0]["delivery_readback"]["source_sha256"] = json!("not-a-digest");
        assert!(audit_fixture(&fixture).unwrap_err().contains("SHA-256"));
    }

    #[test]
    fn additive_only_control_has_zero_interaction() {
        let mut fixture = fixture_value();
        for block in fixture["blocks"].as_array_mut().expect("blocks array") {
            let arms = block["arms"].as_array_mut().expect("arms array");
            let baseline = 10.0;
            for arm in arms {
                let a = arm["factor_a_on"].as_bool().expect("factor A bool");
                let b = arm["factor_b_on"].as_bool().expect("factor B bool");
                let response = baseline + if a { 2.0 } else { 0.0 } + if b { 1.0 } else { 0.0 };
                arm["outcomes"][0]["value"] = json!(response);
            }
        }
        fixture["synthetic_truth"]["expected_mean_interaction_by_channel"]["local_pairing_gap_mev"] = json!(0.0);
        let receipt = audit_fixture(&fixture).expect("additive control must be accepted");
        assert_eq!(
            receipt["response_channels"]["local_pairing_gap_mev"]["mean_interaction_contrast"],
            0.0
        );
    }

    #[test]
    fn refuses_response_channel_unit_mixing() {
        let mut fixture = fixture_value();
        fixture["blocks"][0]["arms"][0]["outcomes"][1]["unit"] = json!("K");
        assert!(audit_fixture(&fixture).unwrap_err().contains("must use meV"));
    }

    #[test]
    fn refuses_changed_condition_profile() {
        let mut fixture = fixture_value();
        fixture["blocks"][1]["condition_profile_id"] = json!("OTHER-CONDITIONS");
        assert!(audit_fixture(&fixture).unwrap_err().contains("condition profile differs"));
    }
}
