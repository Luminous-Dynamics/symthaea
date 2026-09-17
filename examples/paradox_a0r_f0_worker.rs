// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PARADOX-A0R-F0 V2 research-only, label-blind Plane-F transport worker.

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::{self, BufRead, Write};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use uuid::Uuid;

const REQUEST_SCHEMA: &str = "PARADOX-A0R-F0-WORKER-REQUEST-V2";
const RESPONSE_SCHEMA: &str = "PARADOX-A0R-F0-WORKER-RESPONSE-V2";
const SCIENCE_DOMAIN: &[u8] = b"PARADOX-A0R-F0-SCIENTIFIC-V1";
const RECEIPT_DOMAIN: &[u8] = b"PARADOX-A0R-F0-RECEIPT-V1";
const FEATURE_DOMAIN: &[u8] = b"PARADOX-A0R-F0-FEATURE-BUNDLE-V1";
const PRODUCTION_SHA: &str = "eb73527d05a913e79d1f05135ad6b06c1da8e2ee";
const G2B_SHA: &str = "09d83a1d1fddbbbd30e4eba7cc95946c8eab871f";
const A0_SHA: &str = "8cc2651576f0068a1ccac3ef21c8a0a0eb3c2afb";
const M0_SHA: &str = "306b0471a0854d2e77c1d50438fc493806fe1564";
const GENESIS: &str = "PARADOX-A0R-DEV-V1-GENESIS-2026-09-16";
const CONFIG_PROJECTION_SHA256: &str =
    "2cd99b06cf2cf11cfc0612c0db818355099b48b04bd687ca66a31a08680be1f9";

const ALLOWED_REQUEST_KEYS: &[&str] = &[
    "schema_version",
    "opaque_measurement_id",
    "opaque_base_fixture_id",
    "opaque_transform_id",
    "technical_pair_id",
    "technical_replicate_index",
    "agent_visible_events",
    "measurement_cycle_index",
    "f0_subject_sha",
    "runner_config_projection_sha256",
    "executable_sha256",
    "environment_capsule_sha256",
    "source_binding_sha256",
];

const FORBIDDEN_KEY_FRAGMENTS: &[&str] = &[
    "label",
    "condition",
    "expected_response",
    "oracle",
    "capability_atom",
    "score",
    "split",
    "probe",
    "prediction",
    "preprocessing",
    "semantic_fixture",
];

#[derive(Debug)]
struct Request {
    opaque_measurement_id: String,
    opaque_base_fixture_id: String,
    opaque_transform_id: String,
    technical_pair_id: String,
    technical_replicate_index: u64,
    agent_visible_events: Vec<String>,
    measurement_cycle_index: usize,
    f0_subject_sha: String,
    executable_sha256: String,
    environment_capsule_sha256: String,
    source_binding_sha256: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("F0_WORKER_ERROR:{err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    if std::env::args_os().len() != 1 {
        return Err("SIDECAR_ARGUMENT_FORBIDDEN".into());
    }

    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());

    for (line_index, line) in stdin.lock().lines().enumerate() {
        let line = line.map_err(|e| format!("STDIN_READ_FAILURE:{line_index}:{e}"))?;
        if line.trim().is_empty() {
            continue;
        }

        let value: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| format!("REQUEST_JSON_INVALID:{line_index}:{e}"))?;
        reject_forbidden_keys(&value)?;
        let request = parse_request(&value)?;
        let response = execute_request(&request)?;

        serde_json::to_writer(&mut stdout, &response)
            .map_err(|e| format!("RESPONSE_JSON_FAILURE:{e}"))?;
        stdout
            .write_all(b"\n")
            .map_err(|e| format!("STDOUT_WRITE_FAILURE:{e}"))?;
        stdout
            .flush()
            .map_err(|e| format!("STDOUT_FLUSH_FAILURE:{e}"))?;
    }

    Ok(())
}

fn reject_forbidden_keys(value: &serde_json::Value) -> Result<(), String> {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                let lowered = key.to_ascii_lowercase();
                if FORBIDDEN_KEY_FRAGMENTS
                    .iter()
                    .any(|fragment| lowered.contains(fragment))
                {
                    return Err(format!("SEMANTIC_KEY_FORBIDDEN:{key}"));
                }
                reject_forbidden_keys(child)?;
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                reject_forbidden_keys(item)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn parse_request(value: &serde_json::Value) -> Result<Request, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "REQUEST_MUST_BE_OBJECT".to_string())?;

    let allowed: BTreeSet<&str> = ALLOWED_REQUEST_KEYS.iter().copied().collect();
    for key in obj.keys() {
        if !allowed.contains(key.as_str()) {
            return Err(format!("REQUEST_KEY_FORBIDDEN:{key}"));
        }
    }
    for key in ALLOWED_REQUEST_KEYS {
        if !obj.contains_key(*key) {
            return Err(format!("REQUEST_KEY_MISSING:{key}"));
        }
    }

    if require_str(obj, "schema_version")? != REQUEST_SCHEMA {
        return Err("REQUEST_SCHEMA_MISMATCH".into());
    }
    if require_str(obj, "runner_config_projection_sha256")? != CONFIG_PROJECTION_SHA256 {
        return Err("RUNNER_CONFIG_PROJECTION_MISMATCH".into());
    }

    let technical_replicate_index = require_u64(obj, "technical_replicate_index")?;
    if technical_replicate_index > 1 {
        return Err("TECHNICAL_REPLICATE_INDEX_INVALID".into());
    }

    let events = obj["agent_visible_events"]
        .as_array()
        .ok_or_else(|| "AGENT_VISIBLE_EVENTS_MUST_BE_ARRAY".to_string())?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| "AGENT_VISIBLE_EVENT_MUST_BE_STRING".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if events.is_empty() {
        return Err("AGENT_VISIBLE_EVENTS_EMPTY".into());
    }

    let measurement_cycle_index = require_u64(obj, "measurement_cycle_index")? as usize;
    if measurement_cycle_index >= events.len() {
        return Err("MEASUREMENT_CYCLE_OUT_OF_RANGE".into());
    }

    Ok(Request {
        opaque_measurement_id: require_opaque_id(obj, "opaque_measurement_id")?,
        opaque_base_fixture_id: require_opaque_id(obj, "opaque_base_fixture_id")?,
        opaque_transform_id: require_opaque_id(obj, "opaque_transform_id")?,
        technical_pair_id: require_opaque_id(obj, "technical_pair_id")?,
        technical_replicate_index,
        agent_visible_events: events,
        measurement_cycle_index,
        f0_subject_sha: require_hex_id(obj, "f0_subject_sha", 40)?,
        executable_sha256: require_hex_id(obj, "executable_sha256", 64)?,
        environment_capsule_sha256: require_hex_id(obj, "environment_capsule_sha256", 64)?,
        source_binding_sha256: require_hex_id(obj, "source_binding_sha256", 64)?,
    })
}

fn require_str<'a>(
    obj: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<&'a str, String> {
    obj.get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("REQUEST_STRING_REQUIRED:{key}"))
}

fn require_u64(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<u64, String> {
    obj.get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("REQUEST_U64_REQUIRED:{key}"))
}

fn require_opaque_id(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<String, String> {
    let value = require_str(obj, key)?;
    if value.len() != 32
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("OPAQUE_ID_INVALID:{key}"));
    }
    Ok(value.to_owned())
}

fn require_hex_id(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    len: usize,
) -> Result<String, String> {
    let value = require_str(obj, key)?;
    if value.len() != len || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("REQUEST_HEX_ID_INVALID:{key}"));
    }
    Ok(value.to_ascii_lowercase())
}

fn frozen_config() -> CognitiveLoopConfig {
    let mut config = CognitiveLoopConfig::with_cfc();
    config.genesis_phrase = Some(GENESIS.to_owned());
    config.cfc_config.num_neurons = 256;
    config.cfc_config.input_dim = 256;
    config.cfc_config.delta_t = 0.02;
    config.cfc_config.prediction_horizons = vec![0.02, 0.1, 0.2];
    config.async_training = false;
    config.enable_online_learning = false;
    config.episodic_replay_training = false;
    config.memory_graduation = false;
    config.enable_recurrent_dim_masking = false;
    config.enable_spectral_entropy_masking = false;
    config.effective_dim_fraction_override = None;
    config.attention_budget_override_us = Some(60_000_000);
    config.timezone_offset_hours = 0.0;
    config
}

fn execute_request(request: &Request) -> Result<serde_json::Value, String> {
    let mut service = CognitiveLoopService::new(frozen_config())
        .map_err(|error| format!("SERVICE_CONSTRUCTION_FAILURE:{error}"))?;

    let mut selected = None;
    for (cycle_index, event) in request.agent_visible_events.iter().enumerate() {
        let result = service.cycle(event);
        if cycle_index == request.measurement_cycle_index {
            selected = Some(result);
        }
    }
    let result = selected.ok_or_else(|| "MEASUREMENT_CYCLE_NOT_OBSERVED".to_string())?;

    let recurrent_bytes = f32le_bytes(&result.output);
    let thought_bytes = f32le_bytes(&result.thought_vector);
    let wisdom_bytes: &[u8] = &result.wisdom_hv.0;

    let recurrent_nonfinite_count = result.output.iter().filter(|value| !value.is_finite()).count();
    let thought_nonfinite_count = result
        .thought_vector
        .iter()
        .filter(|value| !value.is_finite())
        .count();
    let recurrent_all_zero = result.output.iter().all(|value| value.to_bits() == 0);

    let mut invalidity_reasons = Vec::new();
    if result.output.len() != 256 {
        invalidity_reasons.push("INVALID_MEASUREMENT_DIMENSION_RECURRENT");
    }
    if result.thought_vector.len() != 32 {
        invalidity_reasons.push("INVALID_MEASUREMENT_DIMENSION_THOUGHT");
    }
    if recurrent_nonfinite_count != 0 || thought_nonfinite_count != 0 {
        invalidity_reasons.push("INVALID_MEASUREMENT_NONFINITE");
    }
    if wisdom_bytes.len() != 2048 {
        invalidity_reasons.push("INVALID_MEASUREMENT_DIMENSION_WISDOM_HV");
    }

    let measurement_validity = if invalidity_reasons.is_empty() {
        "ELIGIBLE_F0_TRANSPORT"
    } else {
        "INVALID_F0_TRANSPORT"
    };

    let recurrent_sha256 = sha256_hex(&recurrent_bytes);
    let thought_sha256 = sha256_hex(&thought_bytes);
    let wisdom_sha256 = sha256_hex(wisdom_bytes);
    let sealed_feature_bundle_sha256 = hash_frames(
        FEATURE_DOMAIN,
        &[
            ("recurrent_f32le", recurrent_bytes.as_slice()),
            ("thought_f32le", thought_bytes.as_slice()),
            ("wisdom_hv", wisdom_bytes),
        ],
    );

    let scientific_payload_bytes = canonical_scientific_payload(
        request.measurement_cycle_index,
        result.output.len(),
        &recurrent_sha256,
        recurrent_nonfinite_count,
        recurrent_all_zero,
        result.thought_vector.len(),
        &thought_sha256,
        thought_nonfinite_count,
        wisdom_bytes.len(),
        &wisdom_sha256,
        &sealed_feature_bundle_sha256,
        measurement_validity,
        &invalidity_reasons,
    );
    let scientific_payload_sha256 = domain_hash(SCIENCE_DOMAIN, &scientific_payload_bytes);

    // Provenance-only randomness is created only after all scientific bytes
    // and the scientific commitment have been sealed.
    let service_instance_id = Uuid::new_v4().to_string();
    let provenance_bytes =
        canonical_provenance_envelope(request, &service_instance_id, &scientific_payload_sha256);
    let receipt_binding_sha256 = domain_hash(RECEIPT_DOMAIN, &provenance_bytes);

    Ok(serde_json::json!({
        "schema_version": RESPONSE_SCHEMA,
        "raw_feature_bundle": {
            "recurrent_f32le_hex": to_hex(&recurrent_bytes),
            "thought_f32le_hex": to_hex(&thought_bytes),
            "wisdom_hv_hex": to_hex(wisdom_bytes),
            "sealed_feature_bundle_sha256": sealed_feature_bundle_sha256.as_str()
        },
        "scientific_payload": {
            "measurement_cycle_index": request.measurement_cycle_index,
            "recurrent_length": result.output.len(),
            "recurrent_f32le_sha256": recurrent_sha256,
            "recurrent_nonfinite_count": recurrent_nonfinite_count,
            "recurrent_all_zero": recurrent_all_zero,
            "thought_vector_length": result.thought_vector.len(),
            "thought_vector_f32le_sha256": thought_sha256,
            "thought_vector_nonfinite_count": thought_nonfinite_count,
            "wisdom_hv_byte_length": wisdom_bytes.len(),
            "wisdom_hv_sha256": wisdom_sha256,
            "sealed_feature_bundle_sha256": sealed_feature_bundle_sha256.as_str(),
            "recurrent_masking_enabled": false,
            "spectral_entropy_masking_enabled": false,
            "effective_dim_fraction_override_is_none": true,
            "measurement_validity": measurement_validity,
            "invalidity_reasons": invalidity_reasons
        },
        "scientific_payload_sha256": scientific_payload_sha256.as_str(),
        "provenance_envelope": {
            "production_subject_sha": PRODUCTION_SHA,
            "g2b_subject_sha": G2B_SHA,
            "a0_subject_sha": A0_SHA,
            "m0_subject_sha": M0_SHA,
            "f0_subject_sha": request.f0_subject_sha.as_str(),
            "opaque_measurement_id": request.opaque_measurement_id.as_str(),
            "opaque_base_fixture_id": request.opaque_base_fixture_id.as_str(),
            "opaque_transform_id": request.opaque_transform_id.as_str(),
            "technical_pair_id": request.technical_pair_id.as_str(),
            "technical_replicate_index": request.technical_replicate_index,
            "service_instance_id": service_instance_id.as_str(),
            "runner_config_projection_sha256": CONFIG_PROJECTION_SHA256,
            "executable_sha256": request.executable_sha256.as_str(),
            "environment_capsule_sha256": request.environment_capsule_sha256.as_str(),
            "source_binding_sha256": request.source_binding_sha256.as_str(),
            "scientific_payload_sha256": scientific_payload_sha256.as_str()
        },
        "receipt_binding_sha256": receipt_binding_sha256.as_str()
    }))
}

fn f32le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    to_hex(&hasher.finalize())
}

fn domain_hash(domain: &[u8], payload: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher.update((payload.len() as u64).to_le_bytes());
    hasher.update(payload);
    to_hex(&hasher.finalize())
}

fn push_frame(out: &mut Vec<u8>, name: &str, value: &[u8]) {
    out.extend_from_slice(&(name.len() as u32).to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value);
}

fn hash_frames(domain: &[u8], fields: &[(&str, &[u8])]) -> String {
    let mut bytes = Vec::new();
    for (name, value) in fields {
        push_frame(&mut bytes, name, value);
    }
    domain_hash(domain, &bytes)
}

#[allow(clippy::too_many_arguments)]
fn canonical_scientific_payload(
    measurement_cycle_index: usize,
    recurrent_length: usize,
    recurrent_sha256: &str,
    recurrent_nonfinite_count: usize,
    recurrent_all_zero: bool,
    thought_length: usize,
    thought_sha256: &str,
    thought_nonfinite_count: usize,
    wisdom_byte_length: usize,
    wisdom_sha256: &str,
    sealed_feature_bundle_sha256: &str,
    measurement_validity: &str,
    invalidity_reasons: &[&str],
) -> Vec<u8> {
    let mut out = Vec::new();
    push_frame(&mut out, "schema", b"PARADOX-A0R-F0-SCIENTIFIC-PAYLOAD-V1");
    push_frame(
        &mut out,
        "measurement_cycle_index",
        &(measurement_cycle_index as u64).to_le_bytes(),
    );
    push_frame(
        &mut out,
        "recurrent_length",
        &(recurrent_length as u64).to_le_bytes(),
    );
    push_frame(&mut out, "recurrent_sha256", recurrent_sha256.as_bytes());
    push_frame(
        &mut out,
        "recurrent_nonfinite_count",
        &(recurrent_nonfinite_count as u64).to_le_bytes(),
    );
    push_frame(
        &mut out,
        "recurrent_all_zero",
        &[u8::from(recurrent_all_zero)],
    );
    push_frame(
        &mut out,
        "thought_length",
        &(thought_length as u64).to_le_bytes(),
    );
    push_frame(&mut out, "thought_sha256", thought_sha256.as_bytes());
    push_frame(
        &mut out,
        "thought_nonfinite_count",
        &(thought_nonfinite_count as u64).to_le_bytes(),
    );
    push_frame(
        &mut out,
        "wisdom_byte_length",
        &(wisdom_byte_length as u64).to_le_bytes(),
    );
    push_frame(&mut out, "wisdom_sha256", wisdom_sha256.as_bytes());
    push_frame(
        &mut out,
        "feature_bundle_sha256",
        sealed_feature_bundle_sha256.as_bytes(),
    );
    push_frame(&mut out, "recurrent_masking_enabled", &[0]);
    push_frame(&mut out, "spectral_entropy_masking_enabled", &[0]);
    push_frame(
        &mut out,
        "effective_dim_fraction_override_is_none",
        &[1],
    );
    push_frame(
        &mut out,
        "measurement_validity",
        measurement_validity.as_bytes(),
    );
    for reason in invalidity_reasons {
        push_frame(&mut out, "invalidity_reason", reason.as_bytes());
    }
    out
}

fn canonical_provenance_envelope(
    request: &Request,
    service_instance_id: &str,
    scientific_payload_sha256: &str,
) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, value) in [
        ("schema", "PARADOX-A0R-F0-PROVENANCE-ENVELOPE-V1"),
        ("production_subject_sha", PRODUCTION_SHA),
        ("g2b_subject_sha", G2B_SHA),
        ("a0_subject_sha", A0_SHA),
        ("m0_subject_sha", M0_SHA),
        ("f0_subject_sha", request.f0_subject_sha.as_str()),
        (
            "opaque_measurement_id",
            request.opaque_measurement_id.as_str(),
        ),
        (
            "opaque_base_fixture_id",
            request.opaque_base_fixture_id.as_str(),
        ),
        ("opaque_transform_id", request.opaque_transform_id.as_str()),
        ("technical_pair_id", request.technical_pair_id.as_str()),
        ("service_instance_id", service_instance_id),
        (
            "runner_config_projection_sha256",
            CONFIG_PROJECTION_SHA256,
        ),
        ("executable_sha256", request.executable_sha256.as_str()),
        (
            "environment_capsule_sha256",
            request.environment_capsule_sha256.as_str(),
        ),
        (
            "source_binding_sha256",
            request.source_binding_sha256.as_str(),
        ),
        ("scientific_payload_sha256", scientific_payload_sha256),
    ] {
        push_frame(&mut out, name, value.as_bytes());
    }
    push_frame(
        &mut out,
        "technical_replicate_index",
        &request.technical_replicate_index.to_le_bytes(),
    );
    out
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
