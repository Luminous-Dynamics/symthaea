// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic known-answer runner for static vs rectangular model ambiguity.

use std::env;
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};

use symthaea_fep::static_model_robust_diagnostics::solve_two_stage_static_model_fixture;

fn main() -> Result<(), Box<dyn Error>> {
    let source_sha = env::var("SYMTHAEA_SOURCE_SHA")?;
    if source_sha.len() != 40 || !source_sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "SYMTHAEA_SOURCE_SHA must be a 40-character hexadecimal Git SHA",
        )
        .into());
    }

    let result = solve_two_stage_static_model_fixture();
    if result.policies_evaluated != 64
        || result.static_robust_risk != 0.0
        || result.rectangular_robust_risk != 0.5
        || result.clairvoyant_lower_bound != 0.0
        || result.calibrated_policy_rectangular_risk != 1.0
        || result.static_optimal_risk_by_vertex != [0.0, 0.0]
    {
        return Err(IoError::other(
            "static-vs-rectangular known-answer policy fixture changed",
        )
        .into());
    }

    println!("schema_version=hlb-static-model-robust-seq-v1");
    println!("status=synthetic_exact_policy_tree_oracle");
    println!("source_sha={source_sha}");
    println!("policies_evaluated={}", result.policies_evaluated);
    println!("static_robust_risk={}", result.static_robust_risk);
    println!("rectangular_robust_risk={}", result.rectangular_robust_risk);
    println!("clairvoyant_lower_bound={}", result.clairvoyant_lower_bound);
    println!(
        "calibrated_policy_rectangular_risk={}",
        result.calibrated_policy_rectangular_risk
    );
    println!(
        "static_optimal_risk_by_vertex={:?}",
        result.static_optimal_risk_by_vertex
    );
    println!("static_optimal_policy={:?}", result.static_optimal_policy);
    println!("rectangular_optimal_policy={:?}", result.rectangular_optimal_policy);
    Ok(())
}
