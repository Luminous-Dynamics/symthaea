use chrono::DateTime;
use jsonschema::{Validator, draft202012};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/transport/ev-battery-observation-batch/0.1";
const CORPUS_VERSION: &str = "ev-battery-observation-corpus/0.1";

#[derive(Debug, Clone, Serialize)]
pub struct CaseResult {
    pub case_id: String,
    pub expectation: String,
    pub accepted: bool,
    pub direct_observations_usable: usize,
    pub observations_withheld: usize,
    pub bms_or_derived_values: usize,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditSummary {
    pub corpus_version: String,
    pub schema_sha256: String,
    pub corpus_sha256: String,
    pub case_count: usize,
    pub expected_accept_count: usize,
    pub expected_reject_count: usize,
    pub cases: Vec<CaseResult>,
    pub report_sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn text_at<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn timestamp_ms(value: &Value, key: &str) -> Option<i64> {
    let raw = text_at(value, key)?;
    DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

fn resolution_rank(value: &str) -> Option<u8> {
    match value {
        "pack" => Some(0),
        "module" => Some(1),
        "cell" => Some(2),
        _ => None,
    }
}

fn expected_unit(channel: &str) -> Option<&'static str> {
    match channel {
        "voltage" => Some("V"),
        "current" => Some("A"),
        "temperature" => Some("Cel"),
        "soc" => Some("1"),
        "charge_power" | "discharge_power" => Some("kW"),
        _ => None,
    }
}

fn schema_validator(schema_bytes: &[u8]) -> Result<Validator, String> {
    let schema: Value =
        serde_json::from_slice(schema_bytes).map_err(|e| format!("schema JSON: {e}"))?;
    if text_at(&schema, "$schema") != Some("https://json-schema.org/draft/2020-12/schema") {
        return Err("schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    if text_at(&schema, "$id") != Some(SCHEMA_ID) {
        return Err("schema $id does not match the frozen contract identity".to_owned());
    }
    if !draft202012::meta::is_valid(&schema) {
        return Err("schema fails Draft 2020-12 meta-schema validation".to_owned());
    }
    draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|e| format!("schema compilation: {e}"))
}

fn semantic_errors(batch: &Value) -> (Vec<String>, usize, usize, usize) {
    let mut errors = Vec::new();
    let asset = match batch.get("asset") {
        Some(v) => v,
        None => return (vec!["asset missing".to_owned()], 0, 0, 0),
    };
    let max_resolution = text_at(asset, "max_observable_resolution").unwrap_or("");
    let max_rank = resolution_rank(max_resolution).unwrap_or(0);

    let mut scopes = BTreeSet::new();
    if let Some(items) = batch.get("scope_inventory").and_then(Value::as_array) {
        for item in items {
            let resolution = text_at(item, "resolution").unwrap_or("");
            let subject_ref = text_at(item, "subject_ref").unwrap_or("");
            let key = (resolution.to_owned(), subject_ref.to_owned());
            if !scopes.insert(key) {
                errors.push(format!(
                    "duplicate scope inventory entry: {resolution}/{subject_ref}"
                ));
            }
            if resolution == "pack" && subject_ref != text_at(asset, "pack_ref").unwrap_or("") {
                errors.push("pack scope inventory identity differs from asset.pack_ref".to_owned());
            }
        }
    }

    let calibrations: BTreeMap<String, &Value> = batch
        .get("calibrations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|cal| text_at(cal, "calibration_ref").map(|id| (id.to_owned(), cal)))
        .collect();
    let uncertainty_ms = batch
        .pointer("/clock/time_uncertainty_ms")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let max_delay_ms = batch
        .pointer("/clock/max_transport_delay_ms")
        .and_then(Value::as_i64)
        .unwrap_or(0);

    let mut ids = BTreeSet::new();
    let mut streams: BTreeMap<String, (i64, i64)> = BTreeMap::new();
    let mut usable_direct = 0;
    let mut withheld = 0;
    let mut bms_or_derived = 0;

    for (index, observation) in batch
        .get("observations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let prefix = format!("observations[{index}]");
        let id = text_at(observation, "observation_id").unwrap_or("");
        if !ids.insert(id.to_owned()) {
            errors.push(format!("{prefix}: duplicate observation_id {id}"));
        }

        let channel = text_at(observation, "channel").unwrap_or("");
        let unit = text_at(observation, "unit").unwrap_or("");
        if expected_unit(channel) != Some(unit) {
            errors.push(format!(
                "{prefix}: unit {unit} is invalid for channel {channel}"
            ));
        }

        let resolution = text_at(observation, "subject_resolution").unwrap_or("");
        let subject_ref = text_at(observation, "subject_ref").unwrap_or("");
        let rank = resolution_rank(resolution).unwrap_or(0);
        if rank > max_rank {
            errors.push(format!(
                "{prefix}: {resolution}-level observation exceeds declared maximum {max_resolution}"
            ));
        }
        if !scopes.contains(&(resolution.to_owned(), subject_ref.to_owned())) {
            errors.push(format!(
                "{prefix}: subject {resolution}/{subject_ref} is absent from exact scope inventory"
            ));
        }

        let state = text_at(
            observation.pointer("/quality").unwrap_or(&Value::Null),
            "state",
        )
        .unwrap_or("");
        let flags: Vec<&str> = observation
            .pointer("/quality/flags")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        if state == "valid" && !flags.is_empty() {
            errors.push(format!("{prefix}: valid state cannot carry quality flags"));
        }
        if state != "valid" && flags.is_empty() {
            errors.push(format!(
                "{prefix}: non-valid state requires at least one reason flag"
            ));
        }

        let value = observation.get("value").unwrap_or(&Value::Null);
        if state == "unavailable" {
            if !value.is_null() || !flags.contains(&"missing_value") {
                errors.push(format!("{prefix}: unavailable must use null value and missing_value flag; numeric zero is not a missing sentinel"));
            }
        } else if value.is_null() {
            errors.push(format!(
                "{prefix}: null value is permitted only for unavailable observations"
            ));
        } else if let Some(number) = value.as_f64() {
            if !number.is_finite() {
                errors.push(format!("{prefix}: value must be finite"));
            }
            if channel == "soc" && !(0.0..=1.0).contains(&number) {
                errors.push(format!("{prefix}: SOC fraction must lie in [0, 1]"));
            }
            if matches!(channel, "voltage" | "charge_power" | "discharge_power") && number < 0.0 {
                errors.push(format!(
                    "{prefix}: {channel} cannot be negative under this contract"
                ));
            }
        }

        let sampled = timestamp_ms(observation, "sampled_at");
        let received = timestamp_ms(observation, "received_at");
        if let (Some(sampled_ms), Some(received_ms)) = (sampled, received) {
            let latency = received_ms.saturating_sub(sampled_ms);
            if latency < -uncertainty_ms && (state == "valid" || !flags.contains(&"clock_skew")) {
                errors.push(format!("{prefix}: sample time exceeds receive time beyond clock uncertainty without a quarantined clock_skew flag"));
            }
            if latency > max_delay_ms
                && (state == "valid"
                    || !(flags.contains(&"stale") || flags.contains(&"transport_delay")))
            {
                errors.push(format!("{prefix}: transport delay exceeds policy but observation is not quarantined as stale"));
            }
        }

        let measurement_class = text_at(observation, "measurement_class").unwrap_or("");
        let calibration_ref = observation.get("calibration_ref").and_then(Value::as_str);
        if measurement_class == "direct_sensor_observation" {
            match calibration_ref {
                None => errors.push(format!(
                    "{prefix}: direct sensor observation requires calibration_ref"
                )),
                Some(reference) => {
                    match calibrations.get(reference) {
                        None => errors.push(format!(
                            "{prefix}: calibration_ref {reference} is not registered in this batch"
                        )),
                        Some(cal) => {
                            if text_at(cal, "sensor_ref") != text_at(observation, "sensor_ref") {
                                errors.push(format!("{prefix}: calibration sensor identity does not match observation sensor"));
                            }
                            if let (Some(sample_ms), Some(from_ms), Some(until_ms)) = (
                                sampled,
                                timestamp_ms(cal, "valid_from"),
                                timestamp_ms(cal, "valid_until"),
                            ) {
                                if sample_ms < from_ms
                                    && (state == "valid"
                                        || !flags.contains(&"calibration_not_yet_valid"))
                                {
                                    errors.push(format!("{prefix}: sample precedes calibration validity without quarantine"));
                                }
                                if sample_ms > until_ms
                                    && (state == "valid" || !flags.contains(&"calibration_expired"))
                                {
                                    errors.push(format!("{prefix}: sample follows calibration expiry without quarantine"));
                                }
                            }
                        }
                    }
                }
            }
        }

        let sensor = text_at(observation, "sensor_ref").unwrap_or("");
        let stream_key = format!("{sensor}|{channel}|{subject_ref}");
        if let (Some(sample_ms), Some(sequence)) = (
            sampled,
            observation
                .get("monotonic_sequence")
                .and_then(Value::as_i64),
        ) {
            if let Some((previous_sequence, previous_sample)) = streams.get(&stream_key) {
                if (sequence <= *previous_sequence || sample_ms < *previous_sample)
                    && (state == "valid"
                        || !(flags.contains(&"out_of_order")
                            || flags.contains(&"duplicate_sequence")))
                {
                    errors.push(format!(
                        "{prefix}: stream sequence/time regressed without explicit quarantine"
                    ));
                }
            }
            streams.insert(stream_key, (sequence, sample_ms));
        }

        if state == "valid" && measurement_class == "direct_sensor_observation" {
            usable_direct += 1;
        } else if state != "valid" {
            withheld += 1;
        }
        if matches!(
            measurement_class,
            "bms_reported_value"
                | "derived_estimate"
                | "model_prediction"
                | "simulation_output"
                | "vendor_claim"
        ) {
            bms_or_derived += 1;
        }
    }

    (errors, usable_direct, withheld, bms_or_derived)
}

fn audit_case(case: &Value, validator: &Validator) -> CaseResult {
    let case_id = text_at(case, "case_id")
        .unwrap_or("missing-case-id")
        .to_owned();
    let expectation = text_at(case, "expectation").unwrap_or("reject").to_owned();
    let batch = case.get("batch").unwrap_or(&Value::Null);
    let mut errors: Vec<String> = validator
        .iter_errors(batch)
        .map(|e| e.to_string())
        .collect();
    let (semantic, direct, withheld, estimates) = if errors.is_empty() {
        semantic_errors(batch)
    } else {
        (Vec::new(), 0, 0, 0)
    };
    errors.extend(semantic);
    CaseResult {
        case_id,
        expectation,
        accepted: errors.is_empty(),
        direct_observations_usable: direct,
        observations_withheld: withheld,
        bms_or_derived_values: estimates,
        errors,
    }
}

#[derive(Serialize)]
struct DigestPayload<'a> {
    corpus_version: &'a str,
    schema_sha256: &'a str,
    corpus_sha256: &'a str,
    case_count: usize,
    expected_accept_count: usize,
    expected_reject_count: usize,
    cases: &'a [CaseResult],
}

pub fn audit_corpus(schema_bytes: &[u8], corpus_bytes: &[u8]) -> Result<AuditSummary, String> {
    let validator = schema_validator(schema_bytes)?;
    let corpus: Value =
        serde_json::from_slice(corpus_bytes).map_err(|e| format!("corpus JSON: {e}"))?;
    let version = text_at(&corpus, "corpus_version").ok_or("corpus_version missing")?;
    if version != CORPUS_VERSION {
        return Err(format!("unsupported corpus_version {version}"));
    }
    let cases = corpus
        .get("cases")
        .and_then(Value::as_array)
        .ok_or("cases must be an array")?;
    let results: Vec<CaseResult> = cases
        .iter()
        .map(|case| audit_case(case, &validator))
        .collect();

    let mut ids = BTreeSet::new();
    for result in &results {
        if !ids.insert(result.case_id.clone()) {
            return Err(format!("duplicate case_id {}", result.case_id));
        }
        let expected = result.expectation == "accept";
        if expected != result.accepted {
            return Err(format!(
                "fixture {} expected {} but audit accepted={}: {}",
                result.case_id,
                result.expectation,
                result.accepted,
                result.errors.join("; ")
            ));
        }
        if !matches!(result.expectation.as_str(), "accept" | "reject") {
            return Err(format!(
                "fixture {} has invalid expectation",
                result.case_id
            ));
        }
    }

    let expected_accept_count = results.iter().filter(|r| r.expectation == "accept").count();
    let expected_reject_count = results.iter().filter(|r| r.expectation == "reject").count();
    let schema_digest = sha256(schema_bytes);
    let corpus_digest = sha256(corpus_bytes);
    let payload = DigestPayload {
        corpus_version: version,
        schema_sha256: &schema_digest,
        corpus_sha256: &corpus_digest,
        case_count: results.len(),
        expected_accept_count,
        expected_reject_count,
        cases: &results,
    };
    let report_bytes =
        serde_json::to_vec(&payload).map_err(|e| format!("report serialization: {e}"))?;
    let report_digest = sha256(&report_bytes);

    Ok(AuditSummary {
        corpus_version: version.to_owned(),
        schema_sha256: schema_digest,
        corpus_sha256: corpus_digest,
        case_count: results.len(),
        expected_accept_count,
        expected_reject_count,
        cases: results,
        report_sha256: report_digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &[u8] =
        include_bytes!("../../../schemas/transport/ev_battery_observation_batch_v0_1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../fixtures/ev_battery/observation_batches_v0_1.json");

    #[test]
    fn schema_and_corpus_meta_contract_are_valid() {
        let summary = audit_corpus(SCHEMA, CORPUS).expect("frozen corpus must audit");
        assert_eq!(summary.case_count, 18);
        assert_eq!(summary.expected_accept_count, 5);
        assert_eq!(summary.expected_reject_count, 13);
    }

    #[test]
    fn stale_and_missing_values_are_retained_but_not_usable() {
        let summary = audit_corpus(SCHEMA, CORPUS).expect("frozen corpus must audit");
        let stale = summary
            .cases
            .iter()
            .find(|c| c.case_id == "positive-quarantined-stale-observation")
            .unwrap();
        assert_eq!(stale.accepted, true);
        assert_eq!(stale.observations_withheld, 1);
        let missing = summary
            .cases
            .iter()
            .find(|c| c.case_id == "positive-explicit-unavailable-reading")
            .unwrap();
        assert_eq!(missing.accepted, true);
        assert_eq!(missing.observations_withheld, 1);
    }

    #[test]
    fn estimates_are_not_counted_as_direct_sensor_observations() {
        let summary = audit_corpus(SCHEMA, CORPUS).expect("frozen corpus must audit");
        let mixed = summary
            .cases
            .iter()
            .find(|c| c.case_id == "positive-measurement-versus-bms-estimate")
            .unwrap();
        assert_eq!(mixed.direct_observations_usable, 2);
        assert_eq!(mixed.bms_or_derived_values, 1);
    }

    #[test]
    fn every_authored_negative_control_is_rejected() {
        let summary = audit_corpus(SCHEMA, CORPUS).expect("frozen corpus must audit");
        assert!(
            summary
                .cases
                .iter()
                .filter(|c| c.expectation == "reject")
                .all(|c| !c.accepted)
        );
    }
}
