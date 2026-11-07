// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::env;
use symthaea_science_evidence::{
    ArtifactDigest,
    ClaimBoundary,
    ExperimentManifest,
    RunEnvironment,
    RunReceipt,
    VerificationStatus,
    SCIENCE_EVIDENCE_SCHEMA,
};

fn required_env(name: &str) -> Result<String, Box<dyn Error>> {
    env::var(name).map_err(|_| format!("set {name} to an explicit run value").into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let source_commit_sha = required_env("SYMTHAEA_SOURCE_COMMIT")?;
    let toolchain = required_env("SYMTHAEA_TOOLCHAIN")?;
    let platform = required_env("SYMTHAEA_PLATFORM")?;
    let clean = match required_env("SYMTHAEA_SOURCE_TREE_CLEAN")?.as_str() {
        "true" => true,
        "false" => false,
        _ => return Err("SYMTHAEA_SOURCE_TREE_CLEAN must be exactly true or false".into()),
    };

    let fixture_bytes = b"{\"seed\":42,\"events\":1500}";
    let manifest = ExperimentManifest {
        schema: SCIENCE_EVIDENCE_SCHEMA.to_string(),
        study_id: "example-minkowski-causal-set".to_string(),
        protocol_id: "1p1d-fixed-n-reference-v1".to_string(),
        source_commit_sha,
        source_tree_clean: clean,
        toolchain: toolchain.clone(),
        platform: platform.clone(),
        coordinate_and_unit_conventions: vec![
            "u=t+x, v=t-x".to_string(),
            "c=1; T=1".to_string(),
        ],
        equations_and_assumptions: vec![
            "u_a<=u_b && v_a<=v_b with at least one strict inequality".to_string(),
            "fixed-cardinality Poisson sprinkling in flat 1+1D Minkowski".to_string(),
        ],
        configuration: serde_json::json!({"seed": 42, "events": 1500}),
        input_artifacts: vec![ArtifactDigest::from_bytes("fixture.json", fixture_bytes)],
        input_scope_note: "one frozen, content-addressed fixture".to_string(),
        seed_policy: "deterministic SplitMix64; seed is explicit".to_string(),
        seeds: vec![42],
        baseline_description: "analytic Myrheim ordering fraction r=0.5".to_string(),
        primary_metrics: vec![
            "ordering_fraction".to_string(),
            "Myrheim_Meyer_dimension".to_string(),
            "longest_chain_length".to_string(),
        ],
        uncertainty_method: "multi-seed Monte Carlo SD and t interval".to_string(),
        claim_boundary: ClaimBoundary::NumericalSimulation,
        claims: vec![
            "the finite-sample estimator approaches the flat 1+1D reference".to_string(),
        ],
        non_claims: vec![
            "does not establish causal-set quantum gravity".to_string(),
            "does not prove spacetime is fundamentally discrete".to_string(),
        ],
        falsification_conditions: vec![
            "ordering fraction fails to approach 0.5 as event count increases".to_string(),
        ],
    };
    manifest.validate()?;
    let manifest_digest = manifest.digest()?;

    // This example does not run the experiment. Its receipt is deliberately
    // NotRun, with no checks and no outputs, and therefore cannot qualify.
    let receipt = RunReceipt {
        manifest_blake3_hex: manifest_digest.clone(),
        source_commit_sha: manifest.source_commit_sha.clone(),
        status: VerificationStatus::NotRun,
        checks: vec![],
        output_artifacts: vec![],
        environment: RunEnvironment {
            toolchain,
            platform,
            runner_label: "manifest-demo-not-a-runner".to_string(),
        },
    };

    println!("manifest_blake3={manifest_digest}");
    println!("receipt_status={:?}", receipt.status);
    println!("qualifies_as_pass={}", receipt.qualifies_as_pass(&manifest));
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
