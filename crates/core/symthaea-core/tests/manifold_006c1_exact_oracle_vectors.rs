// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::{env, fs, path::Path};

use serde_json::{Value, json};
use sha3::{Digest, Sha3_256};
use symthaea_core::kinodynamic_reachability::{
    BoundedSingleIntegrator1D, ControlInterval1D, PlantTimeProfile,
};
use symthaea_core::robust_reachability::{
    BoundedDisturbance1D, RobustReachabilityPolicy1D, RobustReachabilityResult1D,
    RobustSingleIntegratorQuery1D, TerminalInterval1D, fixed_control_terminal_envelope,
    solve_robust_single_integrator_analytic,
};

const VECTOR_IDENTITY_DOMAIN: &[u8] = b"symthaea.manifold-006c1.vector-set.v1\0";

fn bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn model() -> BoundedSingleIntegrator1D {
    BoundedSingleIntegrator1D::new(ControlInterval1D::new(-1.0, 1.0).unwrap())
}

fn query(
    model: &BoundedSingleIntegrator1D,
    initial_state: f64,
    target_lower: f64,
    target_upper: f64,
    disturbance_lower: f64,
    disturbance_upper: f64,
    horizon: f64,
) -> RobustSingleIntegratorQuery1D {
    RobustSingleIntegratorQuery1D::new(
        model,
        initial_state,
        TerminalInterval1D::new(target_lower, target_upper).unwrap(),
        BoundedDisturbance1D::new(disturbance_lower, disturbance_upper).unwrap(),
        PlantTimeProfile::new(horizon).unwrap(),
        RobustReachabilityPolicy1D::reference_v1(),
    )
    .unwrap()
}

fn classification(result: &RobustReachabilityResult1D) -> &'static str {
    match result {
        RobustReachabilityResult1D::RobustFeasible { .. } => "RobustFeasible",
        RobustReachabilityResult1D::CertifiedNotRobust { .. } => "CertifiedNotRobust",
        RobustReachabilityResult1D::Unknown { .. } => "Unknown",
    }
}

fn promoted_control(result: &RobustReachabilityResult1D) -> String {
    match result {
        RobustReachabilityResult1D::RobustFeasible { witness, .. } => bits(witness.control()),
        RobustReachabilityResult1D::CertifiedNotRobust { .. }
        | RobustReachabilityResult1D::Unknown { .. } => "none".to_string(),
    }
}

fn envelope_vector(
    name: &str,
    initial_state: f64,
    control: f64,
    disturbance_lower: f64,
    disturbance_upper: f64,
    horizon: f64,
    target_lower: f64,
    target_upper: f64,
) -> Value {
    let model = model();
    let query = query(
        &model,
        initial_state,
        target_lower,
        target_upper,
        disturbance_lower,
        disturbance_upper,
        horizon,
    );
    let envelope = fixed_control_terminal_envelope(&model, &query, control).unwrap();
    json!({
        "name": name,
        "initial_state_bits": bits(initial_state),
        "control_bits": bits(control),
        "disturbance_min_bits": bits(disturbance_lower),
        "disturbance_max_bits": bits(disturbance_upper),
        "horizon_bits": bits(horizon),
        "target_lower_bits": bits(target_lower),
        "target_upper_bits": bits(target_upper),
        "rust_envelope_lower_bits": bits(envelope.lower()),
        "rust_envelope_upper_bits": bits(envelope.upper()),
    })
}

fn classification_vector(
    name: &str,
    initial_state: f64,
    target_lower: f64,
    target_upper: f64,
    disturbance_lower: f64,
    disturbance_upper: f64,
    horizon: f64,
) -> Value {
    let model = model();
    let query = query(
        &model,
        initial_state,
        target_lower,
        target_upper,
        disturbance_lower,
        disturbance_upper,
        horizon,
    );
    let result = solve_robust_single_integrator_analytic(&model, &query).unwrap();
    json!({
        "name": name,
        "control_min_bits": bits(model.controls().minimum()),
        "control_max_bits": bits(model.controls().maximum()),
        "initial_state_bits": bits(initial_state),
        "disturbance_min_bits": bits(disturbance_lower),
        "disturbance_max_bits": bits(disturbance_upper),
        "horizon_bits": bits(horizon),
        "target_lower_bits": bits(target_lower),
        "target_upper_bits": bits(target_upper),
        "rust_classification": classification(&result),
        "rust_promoted_control_bits": promoted_control(&result),
    })
}

fn push_text(hasher: &mut Sha3_256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn push_bits(hasher: &mut Sha3_256, hex_bits: &str) {
    let raw = u64::from_str_radix(hex_bits, 16).unwrap();
    hasher.update(raw.to_be_bytes());
}

fn vector_set_identity(envelopes: &[Value], classes: &[Value]) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(VECTOR_IDENTITY_DOMAIN);
    hasher.update((envelopes.len() as u64).to_be_bytes());
    for vector in envelopes {
        hasher.update(b"E");
        push_text(&mut hasher, vector["name"].as_str().unwrap());
        for key in [
            "initial_state_bits",
            "control_bits",
            "disturbance_min_bits",
            "disturbance_max_bits",
            "horizon_bits",
            "target_lower_bits",
            "target_upper_bits",
            "rust_envelope_lower_bits",
            "rust_envelope_upper_bits",
        ] {
            push_bits(&mut hasher, vector[key].as_str().unwrap());
        }
    }
    hasher.update((classes.len() as u64).to_be_bytes());
    for vector in classes {
        hasher.update(b"C");
        push_text(&mut hasher, vector["name"].as_str().unwrap());
        for key in [
            "control_min_bits",
            "control_max_bits",
            "initial_state_bits",
            "disturbance_min_bits",
            "disturbance_max_bits",
            "horizon_bits",
            "target_lower_bits",
            "target_upper_bits",
        ] {
            push_bits(&mut hasher, vector[key].as_str().unwrap());
        }
        push_text(
            &mut hasher,
            vector["rust_classification"].as_str().unwrap(),
        );
        push_text(
            &mut hasher,
            vector["rust_promoted_control_bits"].as_str().unwrap(),
        );
    }
    format!("{:x}", hasher.finalize())
}

#[test]
fn theorem_vectors_cover_rounding_underflow_exact_scaling_and_classification() {
    let tiny = f64::MIN_POSITIVE;
    let envelope_vectors = vec![
        envelope_vector("decimal-rounding-trap", 0.0, 0.0, 0.0, 0.1, 0.3, 0.0, 0.03),
        envelope_vector("nonzero-underflow", 0.0, tiny, 0.0, 0.0, tiny, -1.0, 1.0),
        envelope_vector("power-of-two-exact", 0.0, 0.5, 0.0, 0.0, 2.0, 1.0, 1.0),
        envelope_vector("signed-normal", 0.5, -0.25, -0.125, 0.125, 2.0, -1.0, 1.0),
    ];
    let classification_vectors = vec![
        classification_vector("robust-feasible", 0.0, -0.5, 0.5, -0.25, 0.25, 2.0),
        classification_vector("rounding-boundary-unknown", 0.0, 0.0, 0.03, 0.0, 0.1, 0.3),
        classification_vector("certified-not-robust", 0.0, -0.1, 0.1, -0.25, 0.25, 2.0),
    ];
    let vector_set_identity = vector_set_identity(&envelope_vectors, &classification_vectors);
    let vectors = json!({
        "schema": "symthaea.manifold-006c1.oracle-vectors.v2",
        "bit_encoding": "ieee754-binary64-u64-hex",
        "vector_identity_algorithm": "sha3-256-domain-separated-semantic-v1",
        "vector_set_identity": vector_set_identity,
        "envelope_vectors": envelope_vectors,
        "classification_vectors": classification_vectors,
    });

    let envelopes = vectors["envelope_vectors"].as_array().unwrap();
    assert_eq!(envelopes.len(), 4);
    assert_ne!(
        envelopes[0]["rust_envelope_upper_bits"],
        Value::String(bits(0.03))
    );
    assert_eq!(tiny * tiny, 0.0);
    assert_ne!(
        envelopes[1]["rust_envelope_upper_bits"],
        Value::String(bits(0.0))
    );
    assert_eq!(
        envelopes[2]["rust_envelope_lower_bits"],
        Value::String(bits(1.0))
    );
    assert_eq!(
        envelopes[2]["rust_envelope_upper_bits"],
        Value::String(bits(1.0))
    );

    let classes = vectors["classification_vectors"].as_array().unwrap();
    assert_eq!(classes[0]["rust_classification"], "RobustFeasible");
    assert_ne!(classes[0]["rust_promoted_control_bits"], "none");
    assert_eq!(classes[1]["rust_classification"], "Unknown");
    assert_eq!(classes[1]["rust_promoted_control_bits"], "none");
    assert_eq!(classes[2]["rust_classification"], "CertifiedNotRobust");
    assert_eq!(classes[2]["rust_promoted_control_bits"], "none");
    assert_eq!(vectors["vector_set_identity"].as_str().unwrap().len(), 64);

    if let Ok(output_path) = env::var("MANIFOLD_006C1_VECTOR_PATH") {
        let path = Path::new(&output_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, serde_json::to_vec_pretty(&vectors).unwrap()).unwrap();
    }
}
