// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Fabrication — Verification & Safety Sweettest
//!
//! Integration tests for verification submission, safety claims,
//! epistemic scoring, and safety class enforcement.
//!
//! ## Running
//! ```bash
//! cd mycelix-workspace/happs/fabrication/tests/sweettest
//! cargo test --release --test sweettest_verification -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

use fabrication_sweettest::common::*;

// ============================================================================
// Mirror types
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitVerificationInput {
    pub design_hash: ActionHash,
    pub verification_type: String,
    pub result: serde_json::Value,
    pub evidence: Vec<ActionHash>,
    pub credentials: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitClaimInput {
    pub design_hash: ActionHash,
    pub claim_type: serde_json::Value,
    pub claim_text: String,
    pub supporting_evidence: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EpistemicScore {
    pub empirical: f32,
    pub normative: f32,
    pub mythic: f32,
    pub overall_confidence: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerificationSummary {
    pub design_hash: ActionHash,
    pub total_verifications: u32,
    pub passed: u32,
    pub failed: u32,
    pub claims_count: u32,
    pub average_confidence: f32,
}

// ============================================================================
// Verification Workflow Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_verification_submit_and_query() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a design first
    let design_input = CreateDesignInput {
        title: "Verification Test Bracket".to_string(),
        description: "For verification workflow testing".to_string(),
        category: "Parts".to_string(),
        intent_vector: None,
        parametric_schema: None,
        constraint_graph: None,
        material_compatibility: vec![],
        circularity_score: 0.5,
        embodied_energy_kwh: 1.0,
        repair_manifest: None,
        license: serde_json::json!("PublicDomain"),
        safety_class: "Class2LoadBearing".to_string(),
    };

    let design_record: Record = conductor
        .call(
            &alice.zome("designs_coordinator"),
            "create_design",
            design_input,
        )
        .await;

    let design_hash = design_record.action_address().clone();

    // Submit a verification
    let verification_input = SubmitVerificationInput {
        design_hash: design_hash.clone(),
        verification_type: "StructuralAnalysis".to_string(),
        result: serde_json::json!({
            "Passed": {
                "confidence": 0.92,
                "notes": "FEA analysis complete"
            }
        }),
        evidence: vec![],
        credentials: vec!["PE License #12345".to_string()],
    };

    let verification_record: Record = conductor
        .call(
            &alice.zome("verification_coordinator"),
            "submit_verification",
            verification_input,
        )
        .await;

    assert_eq!(
        verification_record.action().author(),
        alice.agent_pubkey()
    );

    // Query verifications for this design (now takes HashPaginationInput)
    let verifications: serde_json::Value = conductor
        .call(
            &alice.zome("verification_coordinator"),
            "get_design_verifications",
            HashPaginationInput {
                hash: design_hash.clone(),
                pagination: None,
            },
        )
        .await;

    let items = verifications.get("items").and_then(|v| v.as_array()).unwrap();
    assert_eq!(items.len(), 1);

    // Get verification summary
    let summary: VerificationSummary = conductor
        .call(
            &alice.zome("verification_coordinator"),
            "get_verification_summary",
            design_hash.clone(),
        )
        .await;

    assert_eq!(summary.total_verifications, 1);
    assert_eq!(summary.passed, 1);
    assert_eq!(summary.failed, 0);

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_safety_claim_and_epistemic_score() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a design
    let design_input = CreateDesignInput {
        title: "Epistemic Score Test".to_string(),
        description: "For epistemic scoring test".to_string(),
        category: "Parts".to_string(),
        intent_vector: None,
        parametric_schema: None,
        constraint_graph: None,
        material_compatibility: vec![],
        circularity_score: 0.5,
        embodied_energy_kwh: 1.0,
        repair_manifest: None,
        license: serde_json::json!("PublicDomain"),
        safety_class: "Class1Functional".to_string(),
    };

    let design_record: Record = conductor
        .call(
            &alice.zome("designs_coordinator"),
            "create_design",
            design_input,
        )
        .await;

    let design_hash = design_record.action_address().clone();

    // Submit a safety claim
    let claim_input = SubmitClaimInput {
        design_hash: design_hash.clone(),
        claim_type: serde_json::json!({
            "LoadCapacity": "Supports 100kg static load"
        }),
        claim_text: "Bracket supports up to 100kg per ISO 14122".to_string(),
        supporting_evidence: vec!["FEA report v3.0".to_string()],
    };

    let claim_record: Record = conductor
        .call(
            &alice.zome("verification_coordinator"),
            "submit_safety_claim",
            claim_input,
        )
        .await;

    assert_eq!(claim_record.action().author(), alice.agent_pubkey());

    // Get epistemic score (should have defaults since Knowledge hApp is not present)
    let score: EpistemicScore = conductor
        .call(
            &alice.zome("verification_coordinator"),
            "get_epistemic_score",
            design_hash,
        )
        .await;

    // Should have default epistemic values (0.5, 0.3, 0.2)
    assert!(score.empirical > 0.0);
    assert!(score.overall_confidence > 0.0);

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_verification_summary_with_multiple_results() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let design_input = CreateDesignInput {
        title: "Multi-Verification Design".to_string(),
        description: "For multi-verification summary test".to_string(),
        category: "Parts".to_string(),
        intent_vector: None,
        parametric_schema: None,
        constraint_graph: None,
        material_compatibility: vec![],
        circularity_score: 0.5,
        embodied_energy_kwh: 1.0,
        repair_manifest: None,
        license: serde_json::json!("PublicDomain"),
        safety_class: "Class3BodyContact".to_string(),
    };

    let design_record: Record = conductor
        .call(
            &alice.zome("designs_coordinator"),
            "create_design",
            design_input,
        )
        .await;

    let design_hash = design_record.action_address().clone();

    // Submit passing verification
    let pass_input = SubmitVerificationInput {
        design_hash: design_hash.clone(),
        verification_type: "StructuralAnalysis".to_string(),
        result: serde_json::json!({
            "Passed": { "confidence": 0.95, "notes": "Passed" }
        }),
        evidence: vec![],
        credentials: vec!["PE #1".to_string()],
    };

    let _: Record = conductor
        .call(
            &alice.zome("verification_coordinator"),
            "submit_verification",
            pass_input,
        )
        .await;

    // Submit failing verification
    let fail_input = SubmitVerificationInput {
        design_hash: design_hash.clone(),
        verification_type: "MaterialCompatibility".to_string(),
        result: serde_json::json!({
            "Failed": { "reasons": ["Material mismatch"] }
        }),
        evidence: vec![],
        credentials: vec!["PE #2".to_string()],
    };

    let _: Record = conductor
        .call(
            &alice.zome("verification_coordinator"),
            "submit_verification",
            fail_input,
        )
        .await;

    // Submit a claim too
    let claim_input = SubmitClaimInput {
        design_hash: design_hash.clone(),
        claim_type: serde_json::json!({
            "TemperatureRange": "Safe to 80C"
        }),
        claim_text: "Operates safely up to 80°C".to_string(),
        supporting_evidence: vec![],
    };

    let _: Record = conductor
        .call(
            &alice.zome("verification_coordinator"),
            "submit_safety_claim",
            claim_input,
        )
        .await;

    // Check summary
    let summary: VerificationSummary = conductor
        .call(
            &alice.zome("verification_coordinator"),
            "get_verification_summary",
            design_hash,
        )
        .await;

    assert_eq!(summary.total_verifications, 2);
    assert_eq!(summary.passed, 1);
    assert_eq!(summary.failed, 1);
    assert_eq!(summary.claims_count, 1);
    assert!(summary.average_confidence > 0.0);

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
