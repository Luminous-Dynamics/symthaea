// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PARADOX-A0R-F0 V4 research-only, label-blind Plane-F transport worker.
//!
//! Deliberately self-contained: this binary adds no Cargo dependency or feature.

use serde::Deserialize;
use std::io::{self, BufRead, Write};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use uuid::Uuid;

const REQUEST_SCHEMA: &str = "PARADOX-A0R-F0-WORKER-REQUEST-V4";
// The response wire format is unchanged from V3; keeping the same schema is an
// explicit compatibility claim, independently checked by the pair auditor.
const RESPONSE_SCHEMA: &str = "PARADOX-A0R-F0-WORKER-RESPONSE-V3";
const SCIENCE_DOMAIN: &[u8] = b"PARADOX-A0R-F0-SCIENTIFIC-V1";
const RECEIPT_DOMAIN: &[u8] = b"PARADOX-A0R-F0-RECEIPT-V2";
const FEATURE_DOMAIN: &[u8] = b"PARADOX-A0R-F0-FEATURE-BUNDLE-V1";
const EVENTS_DOMAIN: &[u8] = b"PARADOX-A0R-F0-EVENTS-V1";
const PRODUCTION_SHA: &str = "eb73527d05a913e79d1f05135ad6b06c1da8e2ee";
const G2B_SHA: &str = "09d83a1d1fddbbbd30e4eba7cc95946c8eab871f";
const A0_SHA: &str = "8cc2651576f0068a1ccac3ef21c8a0a0eb3c2afb";
const M0_SHA: &str = "306b0471a0854d2e77c1d50438fc493806fe1564";
const GENESIS: &str = "PARADOX-A0R-DEV-V1-GENESIS-2026-09-16";
const CONFIG_PROJECTION_SHA256: &str =
    "2cd99b06cf2cf11cfc0612c0db818355099b48b04bd687ca66a31a08680be1f9";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestWire {
    schema_version: String,
    opaque_measurement_id: String,
    opaque_base_fixture_id: String,
    opaque_transform_id: String,
    technical_pair_id: String,
    technical_replicate_index: u8,
    agent_visible_events: Vec<String>,
    measurement_cycle_index: u64,
    f0_subject_sha: String,
    runner_config_projection_sha256: String,
    executable_sha256: String,
    environment_capsule_sha256: String,
    source_binding_sha256: String,
}

#[derive(Debug)]
struct Request {
    opaque_measurement_id: String,
    opaque_base_fixture_id: String,
    opaque_transform_id: String,
    technical_pair_id: String,
    technical_replicate_index: u8,
    agent_visible_events: Vec<String>,
    measurement_cycle_index: usize,
    f0_subject_sha: String,
    executable_sha256: String,
    environment_capsule_sha256: String,
    source_binding_sha256: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("F0_WORKER_ERROR:{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    if std::env::args_os().len() != 1 {
        return Err("SIDECAR_ARGUMENT_FORBIDDEN".into());
    }
    verify_sha256_implementation()?;

    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());

    for (line_index, line) in stdin.lock().lines().enumerate() {
        let line = line.map_err(|error| format!("STDIN_READ_FAILURE:{line_index}:{error}"))?;
        if line.trim().is_empty() {
            continue;
        }

        let wire: RequestWire = serde_json::from_str(&line)
            .map_err(|error| format!("REQUEST_JSON_OR_SCHEMA_INVALID:{line_index}:{error}"))?;
        let request = validate_request(wire)?;
        let response = execute_request(&request)?;

        serde_json::to_writer(&mut stdout, &response)
            .map_err(|error| format!("RESPONSE_JSON_FAILURE:{error}"))?;
        stdout
            .write_all(b"\n")
            .map_err(|error| format!("STDOUT_WRITE_FAILURE:{error}"))?;
        stdout
            .flush()
            .map_err(|error| format!("STDOUT_FLUSH_FAILURE:{error}"))?;
    }
    Ok(())
}

fn validate_request(wire: RequestWire) -> Result<Request, String> {
    if wire.schema_version != REQUEST_SCHEMA {
        return Err("REQUEST_SCHEMA_MISMATCH".into());
    }
    if wire.runner_config_projection_sha256 != CONFIG_PROJECTION_SHA256 {
        return Err("RUNNER_CONFIG_PROJECTION_MISMATCH".into());
    }
    if wire.technical_replicate_index > 1 {
        return Err("TECHNICAL_REPLICATE_INDEX_INVALID".into());
    }
    if wire.agent_visible_events.is_empty() {
        return Err("AGENT_VISIBLE_EVENTS_EMPTY".into());
    }
    let final_index = u64::try_from(wire.agent_visible_events.len() - 1)
        .map_err(|_| "EVENT_COUNT_UNREPRESENTABLE".to_string())?;
    if wire.measurement_cycle_index != final_index {
        return Err("MEASUREMENT_CYCLE_NOT_FINAL".into());
    }

    Ok(Request {
        opaque_measurement_id: validate_opaque_id(
            wire.opaque_measurement_id,
            "opaque_measurement_id",
        )?,
        opaque_base_fixture_id: validate_opaque_id(
            wire.opaque_base_fixture_id,
            "opaque_base_fixture_id",
        )?,
        opaque_transform_id: validate_opaque_id(
            wire.opaque_transform_id,
            "opaque_transform_id",
        )?,
        technical_pair_id: validate_opaque_id(wire.technical_pair_id, "technical_pair_id")?,
        technical_replicate_index: wire.technical_replicate_index,
        agent_visible_events: wire.agent_visible_events,
        measurement_cycle_index: usize::try_from(wire.measurement_cycle_index)
            .map_err(|_| "MEASUREMENT_CYCLE_UNREPRESENTABLE".to_string())?,
        f0_subject_sha: validate_lower_hex(wire.f0_subject_sha, 40, "f0_subject_sha")?,
        executable_sha256: validate_lower_hex(
            wire.executable_sha256,
            64,
            "executable_sha256",
        )?,
        environment_capsule_sha256: validate_lower_hex(
            wire.environment_capsule_sha256,
            64,
            "environment_capsule_sha256",
        )?,
        source_binding_sha256: validate_lower_hex(
            wire.source_binding_sha256,
            64,
            "source_binding_sha256",
        )?,
    })
}

fn validate_opaque_id(value: String, field: &str) -> Result<String, String> {
    if value.len() != 32
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("OPAQUE_ID_INVALID:{field}"));
    }
    Ok(value)
}

fn validate_lower_hex(value: String, len: usize, field: &str) -> Result<String, String> {
    if value.len() != len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("LOWER_HEX_INVALID:{field}"));
    }
    Ok(value)
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
    let agent_visible_event_count = u64::try_from(request.agent_visible_events.len())
        .map_err(|_| "EVENT_COUNT_UNREPRESENTABLE".to_string())?;
    let agent_visible_events_sha256 = event_stream_digest(&request.agent_visible_events);

    let mut service = CognitiveLoopService::new(frozen_config())
        .map_err(|error| format!("SERVICE_CONSTRUCTION_FAILURE:{error}"))?;

    let mut measured = None;
    for (cycle_index, event) in request.agent_visible_events.iter().enumerate() {
        let result = service.cycle(event);
        if cycle_index == request.measurement_cycle_index {
            measured = Some(result);
        }
    }
    let result = measured.ok_or_else(|| "MEASUREMENT_CYCLE_NOT_OBSERVED".to_string())?;

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

    let service_instance_id = Uuid::new_v4().to_string();
    let provenance_bytes = canonical_provenance_envelope(
        request,
        &service_instance_id,
        agent_visible_event_count,
        &agent_visible_events_sha256,
        &scientific_payload_sha256,
    );
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
            "agent_visible_event_count": agent_visible_event_count,
            "agent_visible_events_sha256": agent_visible_events_sha256.as_str(),
            "runner_config_projection_sha256": CONFIG_PROJECTION_SHA256,
            "executable_sha256": request.executable_sha256.as_str(),
            "environment_capsule_sha256": request.environment_capsule_sha256.as_str(),
            "source_binding_sha256": request.source_binding_sha256.as_str(),
            "scientific_payload_sha256": scientific_payload_sha256.as_str()
        },
        "receipt_binding_sha256": receipt_binding_sha256.as_str()
    }))
}

fn event_stream_digest(events: &[String]) -> String {
    let mut payload = Vec::new();
    for event in events {
        push_frame(&mut payload, "event_utf8", event.as_bytes());
    }
    domain_hash(EVENTS_DOMAIN, &payload)
}

fn f32le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn verify_sha256_implementation() -> Result<(), String> {
    let empty = sha256_hex(b"");
    if empty != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" {
        return Err("SHA256_SELF_TEST_EMPTY_FAILED".into());
    }
    let abc = sha256_hex(b"abc");
    if abc != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" {
        return Err("SHA256_SELF_TEST_ABC_FAILED".into());
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    to_hex(&sha256_digest(bytes))
}

fn sha256_digest(input: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut message = Vec::with_capacity(input.len() + 72);
    message.extend_from_slice(input);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7)
                ^ w[i - 15].rotate_right(18)
                ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17)
                ^ w[i - 2].rotate_right(19)
                ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut digest = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn domain_hash(domain: &[u8], payload: &[u8]) -> String {
    let mut preimage = Vec::with_capacity(16 + domain.len() + payload.len());
    preimage.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    preimage.extend_from_slice(domain);
    preimage.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    preimage.extend_from_slice(payload);
    sha256_hex(&preimage)
}

fn push_frame(out: &mut Vec<u8>, name: &str, value: &[u8]) {
    out.extend_from_slice(&(name.len() as u32).to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value);
}

fn hash_frames(domain: &[u8], fields: &[(&str, &[u8])]) -> String {
    let mut payload = Vec::new();
    for (name, value) in fields {
        push_frame(&mut payload, name, value);
    }
    domain_hash(domain, &payload)
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
    push_frame(&mut out, "recurrent_length", &(recurrent_length as u64).to_le_bytes());
    push_frame(&mut out, "recurrent_sha256", recurrent_sha256.as_bytes());
    push_frame(
        &mut out,
        "recurrent_nonfinite_count",
        &(recurrent_nonfinite_count as u64).to_le_bytes(),
    );
    push_frame(&mut out, "recurrent_all_zero", &[u8::from(recurrent_all_zero)]);
    push_frame(&mut out, "thought_length", &(thought_length as u64).to_le_bytes());
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
    push_frame(&mut out, "effective_dim_fraction_override_is_none", &[1]);
    push_frame(&mut out, "measurement_validity", measurement_validity.as_bytes());
    for reason in invalidity_reasons {
        push_frame(&mut out, "invalidity_reason", reason.as_bytes());
    }
    out
}

fn canonical_provenance_envelope(
    request: &Request,
    service_instance_id: &str,
    agent_visible_event_count: u64,
    agent_visible_events_sha256: &str,
    scientific_payload_sha256: &str,
) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, value) in [
        ("schema", "PARADOX-A0R-F0-PROVENANCE-ENVELOPE-V2"),
        ("production_subject_sha", PRODUCTION_SHA),
        ("g2b_subject_sha", G2B_SHA),
        ("a0_subject_sha", A0_SHA),
        ("m0_subject_sha", M0_SHA),
        ("f0_subject_sha", request.f0_subject_sha.as_str()),
        ("opaque_measurement_id", request.opaque_measurement_id.as_str()),
        ("opaque_base_fixture_id", request.opaque_base_fixture_id.as_str()),
        ("opaque_transform_id", request.opaque_transform_id.as_str()),
        ("technical_pair_id", request.technical_pair_id.as_str()),
        ("service_instance_id", service_instance_id),
        ("agent_visible_events_sha256", agent_visible_events_sha256),
        ("runner_config_projection_sha256", CONFIG_PROJECTION_SHA256),
        ("executable_sha256", request.executable_sha256.as_str()),
        (
            "environment_capsule_sha256",
            request.environment_capsule_sha256.as_str(),
        ),
        ("source_binding_sha256", request.source_binding_sha256.as_str()),
        ("scientific_payload_sha256", scientific_payload_sha256),
    ] {
        push_frame(&mut out, name, value.as_bytes());
    }
    push_frame(
        &mut out,
        "technical_replicate_index",
        &u64::from(request.technical_replicate_index).to_le_bytes(),
    );
    push_frame(
        &mut out,
        "agent_visible_event_count",
        &agent_visible_event_count.to_le_bytes(),
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
