//! Strict SFEB JSONL trace parsing and hash-chain verification.
//! Event and payload digests are SHA-256 over RFC 8785 JCS bytes.

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const MAX_TRACE_BYTES: usize = 256 * 1024 * 1024;
const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;
const MAX_EVENTS: usize = 100_000;
const MAX_SAFE_INTEGER: i128 = 9_007_199_254_740_991;

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictValueVisitor)
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = StrictValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an I-JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where E: de::Error {
        Ok(StrictValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where E: de::Error {
        let wide = value as i128;
        if wide < -MAX_SAFE_INTEGER || wide > MAX_SAFE_INTEGER {
            return Err(E::custom("integer exceeds the RFC 8785 safe range; encode it as a string"));
        }
        Ok(StrictValue(Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where E: de::Error {
        if value as u128 > MAX_SAFE_INTEGER as u128 {
            return Err(E::custom("integer exceeds the RFC 8785 safe range; encode it as a string"));
        }
        Ok(StrictValue(Value::Number(value.into())))
    }

    fn visit_i128<E>(self, value: i128) -> Result<Self::Value, E>
    where E: de::Error {
        if value < -MAX_SAFE_INTEGER || value > MAX_SAFE_INTEGER {
            return Err(E::custom("integer exceeds the RFC 8785 safe range; encode it as a string"));
        }
        let value = i64::try_from(value).map_err(E::custom)?;
        Ok(StrictValue(Value::Number(value.into())))
    }

    fn visit_u128<E>(self, value: u128) -> Result<Self::Value, E>
    where E: de::Error {
        if value > MAX_SAFE_INTEGER as u128 {
            return Err(E::custom("integer exceeds the RFC 8785 safe range; encode it as a string"));
        }
        let value = u64::try_from(value).map_err(E::custom)?;
        Ok(StrictValue(Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where E: de::Error {
        if !value.is_finite() {
            return Err(E::custom("non-finite number is not valid I-JSON"));
        }
        if value.fract() == 0.0 && value.abs() > MAX_SAFE_INTEGER as f64 {
            return Err(E::custom("integral floating-point value exceeds the RFC 8785 safe range; encode it as a string"));
        }
        let number = serde_json::Number::from_f64(value)
            .ok_or_else(|| E::custom("number cannot be represented as JSON"))?;
        Ok(StrictValue(Value::Number(number)))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where E: de::Error {
        Ok(StrictValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where E: de::Error {
        Ok(StrictValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where E: de::Error {
        Ok(StrictValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where E: de::Error {
        Ok(StrictValue(Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where A: SeqAccess<'de> {
        let mut values = Vec::new();
        while let Some(StrictValue(value)) = sequence.next_element::<StrictValue>()? {
            values.push(value);
        }
        Ok(StrictValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where A: MapAccess<'de> {
        let mut seen = BTreeSet::new();
        let mut values = Map::new();
        while let Some((key, StrictValue(value))) =
            object.next_entry::<String, StrictValue>()?
        {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate object key: {key}")));
            }
            values.insert(key, value);
        }
        Ok(StrictValue(Value::Object(values)))
    }
}

pub fn parse_strict_json_line(line: &[u8], line_number: usize) -> Result<Value, String> {
    if line.len() > MAX_EVENT_BYTES {
        return Err(format!("trace line {line_number} exceeds {MAX_EVENT_BYTES} bytes"));
    }
    let mut deserializer = serde_json::Deserializer::from_slice(line);
    let StrictValue(value) = StrictValue::deserialize(&mut deserializer)
        .map_err(|error| format!("trace line {line_number}: {error}"))?;
    deserializer.end()
        .map_err(|error| format!("trace line {line_number}: trailing data: {error}"))?;
    Ok(value)
}

pub fn parse_jsonl(bytes: &[u8]) -> Result<Vec<Value>, String> {
    if bytes.is_empty() {
        return Err("trace bundle is empty".to_string());
    }
    if bytes.len() > MAX_TRACE_BYTES {
        return Err(format!("trace bundle exceeds {MAX_TRACE_BYTES} bytes"));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("trace bundle is not UTF-8: {error}"))?;
    let mut lines: Vec<&str> = text.split('\n').collect();
    if text.ends_with('\n') {
        lines.pop();
    }
    if lines.is_empty() || lines.len() > MAX_EVENTS {
        return Err(format!("trace event count must be between 1 and {MAX_EVENTS}"));
    }
    let mut events = Vec::with_capacity(lines.len());
    for (index, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            return Err(format!("trace line {} is blank", index + 1));
        }
        events.push(parse_strict_json_line(line.as_bytes(), index + 1)?);
    }
    Ok(events)
}

fn required_string<'a>(event: &'a Value, field: &str, index: usize) -> Result<&'a str, String> {
    event.get(field).and_then(Value::as_str)
        .ok_or_else(|| format!("trace event {index}: {field} must be a string"))
}

fn required_u64(event: &Value, field: &str, index: usize) -> Result<u64, String> {
    event.get(field).and_then(Value::as_u64)
        .ok_or_else(|| format!("trace event {index}: {field} must be an unsigned integer"))
}

fn jcs_sha256(value: &Value) -> Result<String, String> {
    let canonical = serde_jcs::to_vec(value)
        .map_err(|error| format!("RFC 8785 canonicalization failed: {error}"))?;
    Ok(hex::encode(Sha256::digest(canonical)))
}

fn event_hash(event: &Value) -> Result<String, String> {
    let mut unsigned = event.clone();
    let object = unsigned.as_object_mut()
        .ok_or_else(|| "trace event must be a JSON object".to_string())?;
    if object.remove("event_sha256").is_none() {
        return Err("trace event is missing event_sha256".to_string());
    }
    jcs_sha256(&unsigned)
}

pub fn verify_events(
    events: &[Value],
    raw_bundle: &[u8],
    expected_run_id: Option<&str>,
) -> Result<Value, String> {
    if events.len() < 2 || events.len() > MAX_EVENTS {
        return Err(format!("trace must contain 2..={MAX_EVENTS} events"));
    }
    let reparsed = parse_jsonl(raw_bundle)?;
    if reparsed.as_slice() != events {
        return Err("supplied trace events do not match the exact raw JSONL bundle".to_string());
    }
    let first_run_id = required_string(&events[0], "run_id", 0)?;
    if let Some(expected) = expected_run_id {
        if first_run_id != expected {
            return Err(format!("trace run_id {first_run_id} does not match expected {expected}"));
        }
    }

    let mut previous_hash = ZERO_SHA256.to_string();
    let mut previous_elapsed = 0_u64;
    let mut last_hash = String::new();

    for (index, event) in events.iter().enumerate() {
        let label = index + 1;
        if event.get("schema").and_then(Value::as_str) != Some("sfeb-trace-event-v1") {
            return Err(format!("trace event {label}: unsupported schema"));
        }
        let run_id = required_string(event, "run_id", label)?;
        if run_id != first_run_id {
            return Err(format!("trace event {label}: cross-run event replay detected"));
        }
        let sequence = required_u64(event, "sequence", label)?;
        if sequence != index as u64 {
            return Err(format!("trace event {label}: sequence must be {}, got {sequence}", index));
        }
        let elapsed = required_u64(event, "elapsed_ns", label)?;
        if index == 0 && elapsed != 0 {
            return Err("first trace event must have elapsed_ns=0".to_string());
        }
        if index > 0 && elapsed < previous_elapsed {
            return Err(format!("trace event {label}: elapsed_ns regressed"));
        }
        previous_elapsed = elapsed;

        let event_type = required_string(event, "event_type", label)?;
        if index == 0 && event_type != "run_started" {
            return Err("first trace event must be run_started".to_string());
        }
        if event_type == "run_started" && index != 0 {
            return Err(format!("trace event {label}: duplicate or misplaced run_started"));
        }
        if event_type == "run_finished" && index + 1 != events.len() {
            return Err(format!("trace event {label}: run_finished must be terminal"));
        }
        if index + 1 == events.len() && event_type != "run_finished" {
            return Err("trace is truncated or lacks terminal run_finished event".to_string());
        }

        let payload = event.get("payload").filter(|value| value.is_object())
            .ok_or_else(|| format!("trace event {label}: payload must be an object"))?;
        let payload_hash = jcs_sha256(payload)?;
        if required_string(event, "payload_sha256", label)? != payload_hash {
            return Err(format!("trace event {label}: payload_sha256 mismatch"));
        }
        if required_string(event, "previous_event_sha256", label)? != previous_hash {
            return Err(format!("trace event {label}: previous_event_sha256 mismatch"));
        }
        let calculated_hash = event_hash(event)?;
        if required_string(event, "event_sha256", label)? != calculated_hash {
            return Err(format!("trace event {label}: event_sha256 mismatch"));
        }
        previous_hash = calculated_hash.clone();
        last_hash = calculated_hash;
    }

    Ok(serde_json::json!({
        "schema": "sfeb-trace-verification-summary-v1",
        "run_id": first_run_id,
        "event_count": events.len(),
        "trace_bundle_sha256": hex::encode(Sha256::digest(raw_bundle)),
        "final_event_sha256": last_hash,
        "elapsed_ns": previous_elapsed,
        "sequence_contiguous": true,
        "event_chain_valid": true,
        "payload_hashes_valid": true,
        "terminal_event_valid": true
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_event(run_id: &str, sequence: u64, elapsed_ns: u64, event_type: &str, previous: &str) -> Value {
        let mut event = json!({
            "schema": "sfeb-trace-event-v1",
            "run_id": run_id,
            "sequence": sequence,
            "elapsed_ns": elapsed_ns,
            "event_type": event_type,
            "actor": "controller",
            "redaction_state": "non_sensitive",
            "payload": {},
            "payload_sha256": "",
            "previous_event_sha256": previous,
            "event_sha256": ""
        });
        let payload_hash = jcs_sha256(&event["payload"]).unwrap();
        event["payload_sha256"] = Value::String(payload_hash);
        let hash = event_hash(&event).unwrap();
        event["event_sha256"] = Value::String(hash);
        event
    }

    fn valid_events() -> Vec<Value> {
        let start = make_event("run-01", 0, 0, "run_started", ZERO_SHA256);
        let start_hash = start["event_sha256"].as_str().unwrap().to_owned();
        let observe = make_event("run-01", 1, 10, "dom_observation", &start_hash);
        let observe_hash = observe["event_sha256"].as_str().unwrap().to_owned();
        let finish = make_event("run-01", 2, 20, "run_finished", &observe_hash);
        vec![start, observe, finish]
    }

    fn to_jsonl(events: &[Value]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for event in events {
            bytes.extend(serde_json::to_vec(event).unwrap());
            bytes.push(b'\n');
        }
        bytes
    }

    fn verify(events: &[Value]) -> Result<Value, String> {
        let bytes = to_jsonl(events);
        verify_events(events, &bytes, None)
    }

    #[test]
    fn jcs_serialization_matches_rfc8785_object_key_order() {
        let value = json!({"b": 2, "a": 1});
        let bytes = serde_jcs::to_vec(&value).expect("JCS serialization");
        assert_eq!(bytes, br#"{"a":1,"b":2}"#);
    }

    #[test]
    fn checked_in_trace_schema_accepts_a_valid_event_triplet() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../attestation/sfeb-trace-event-v1.schema.json"
        )).expect("trace schema JSON");
        let validator = jsonschema::validator_for(&schema).expect("trace schema compiles");
        for event in valid_events() {
            assert!(validator.is_valid(&event), "generated trace event must satisfy schema");
        }
    }

    #[test]
    fn accepts_a_valid_two_event_trace() {
        let events = valid_events();
        let summary = verify(&events).expect("valid trace");
        assert_eq!(summary["event_count"], 2);
        assert_eq!(summary["run_id"], "run-01");
        assert_eq!(summary["event_chain_valid"], true);
    }

    #[test]
    fn rejects_duplicate_json_object_keys_before_value_construction() {
        let result = parse_strict_json_line(br#"{"run_id":"a","run_id":"b"}"#, 1);
        assert!(result.unwrap_err().contains("duplicate object key"));
    }

    #[test]
    fn rejects_modified_payload_even_when_json_remains_well_formed() {
        let mut events = valid_events();
        events[1]["payload"]["verdict"] = Value::String("accepted".to_string());
        assert!(verify(&events).unwrap_err().contains("payload_sha256 mismatch"));
    }

    #[test]
    fn rejects_reordered_events() {
        let mut events = valid_events();
        events.swap(1, 2);
        assert!(verify(&events).unwrap_err().contains("sequence"));
    }

    #[test]
    fn rejects_truncated_trace_without_terminal_event() {
        let mut events = valid_events();
        events.pop();
        assert!(verify(&events).unwrap_err().contains("lacks terminal run_finished"));
    }

    #[test]
    fn rejects_cross_run_event_replay() {
        let mut events = valid_events();
        events[1]["run_id"] = Value::String("run-other".to_string());
        assert!(verify(&events).unwrap_err().contains("cross-run"));
    }

    #[test]
    fn rejects_duplicate_or_skipped_sequence_numbers() {
        let mut events = valid_events();
        events[1]["sequence"] = Value::from(0);
        assert!(verify(&events).unwrap_err().contains("sequence"));
    }

    #[test]
    fn rejects_elapsed_time_regression() {
        let mut events = valid_events();
        events[2]["elapsed_ns"] = Value::from(5);
        let previous = events[1]["event_sha256"].as_str().unwrap().to_owned();
        events[2]["previous_event_sha256"] = Value::String(previous);
        let hash = event_hash(&events[2]).unwrap();
        events[2]["event_sha256"] = Value::String(hash);
        assert!(verify(&events).unwrap_err().contains("elapsed_ns regressed"));
    }

    #[test]
    fn rejects_event_hash_tampering() {
        let mut events = valid_events();
        events[1]["actor"] = Value::String("worker".to_string());
        assert!(verify(&events).unwrap_err().contains("event_sha256 mismatch"));
    }

    #[test]
    fn rejects_previous_hash_discontinuity() {
        let mut events = valid_events();
        events[1]["previous_event_sha256"] = Value::String("0".repeat(64));
        let hash = event_hash(&events[1]).unwrap();
        events[1]["event_sha256"] = Value::String(hash);
        assert!(verify(&events).unwrap_err().contains("previous_event_sha256 mismatch"));
    }

    #[test]
    fn rejects_trace_values_that_do_not_match_supplied_raw_bytes() {
        let events = valid_events();
        let result = verify_events(&events, br#"{"not":"the trace"}"#, None);
        assert!(result.unwrap_err().contains("supplied trace events do not match"));
    }

    #[test]
    fn rejects_integers_outside_the_jcs_safe_range() {
        let result = parse_strict_json_line(b"9007199254740992", 1);
        assert!(result.unwrap_err().contains("safe range"));
    }

    #[test]
    fn rejects_large_integral_floats_outside_the_jcs_safe_range() {
        let result = parse_strict_json_line(b"9.007199254740992e15", 1);
        assert!(result.unwrap_err().contains("safe range"));
    }

    #[test]
    fn rejects_expected_run_identity_mismatch() {
        let events = valid_events();
        let bytes = to_jsonl(&events);
        let result = verify_events(&events, &bytes, Some("different-run"));
        assert!(result.unwrap_err().contains("does not match expected"));
    }
}
