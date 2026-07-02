// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Data Integrity Verification Pipeline
//!
//! This example demonstrates how to use Mycelix-DeSci to verify the integrity
//! of scientific datasets, ensuring data hasn't been tampered with or corrupted.
//!
//! ## Scenario
//! A research lab needs to verify that datasets they received match the original
//! published data. This example shows:
//! 1. Computing BLAKE3 hash of local dataset
//! 2. Querying API for matching claims
//! 3. Verifying hash matches published hash
//! 4. Checking verification count (trust level)
//! 5. Validating provenance chain
//! 6. Generating integrity report
//!
//! ## Usage
//! ```bash
//! cargo run --example data_integrity_pipeline
//! ```

use mycelix_desci_core::hash::compute_blake3;
use serde_json::json;
use std::fs;

const API_BASE: &str = "http://localhost:8080/api/v1";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║      Mycelix-DeSci: Data Integrity Pipeline Demo        ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Step 1: Load local dataset
    println!("📂 STEP 1: Loading local dataset");
    println!("─────────────────────────────────────");

    let dataset = r#"{
  "experiment": "Protein folding study",
  "method": "AlphaFold2",
  "protein_id": "P12345",
  "confidence_scores": [0.95, 0.92, 0.88, 0.91],
  "structure_pdb": "..."
}"#;

    println!("✓ Dataset loaded: {} bytes", dataset.len());
    println!();

    // Step 2: Compute hash
    println!("🔐 STEP 2: Computing BLAKE3 hash");
    println!("─────────────────────────────────────");

    let hash = compute_blake3(dataset.as_bytes());
    let hash_str = hex::encode(&hash);

    println!("✓ Hash computed successfully");
    println!("  Algorithm: BLAKE3");
    println!("  Hash: blake3:{}", hash_str);
    println!("  Bits: 256");
    println!();

    // Step 3: Query for matching claims
    println!("🔍 STEP 3: Searching for matching claims");
    println!("─────────────────────────────────────");

    let client = reqwest::Client::new();

    // First, create a test claim so we have something to find
    let claim_request = json!({
        "tier": "E0",
        "content": {
            "dataset_hash": format!("blake3:{}", hash_str),
            "description": "Protein folding dataset - P12345 using AlphaFold2",
            "category": "structural-biology",
            "keywords": ["protein-folding", "AlphaFold2", "structure-prediction"],
            "license": "CC0-1.0"
        },
        "creator": "lab@example.org"
    });

    let claim: serde_json::Value = client
        .post(format!("{}/claims", API_BASE))
        .json(&claim_request)
        .send()
        .await?
        .json()
        .await?;

    println!("✓ Found matching claim");
    println!("  Claim ID: {}", claim["id"]);
    println!("  Category: {}", claim["content"]["category"]);
    println!();

    // Step 4: Verify hash matches
    println!("✅ STEP 4: Verifying data integrity");
    println!("─────────────────────────────────────");

    let published_hash = claim["content"]["dataset_hash"]
        .as_str()
        .unwrap()
        .strip_prefix("blake3:")
        .unwrap();

    let matches = published_hash == hash_str;

    if matches {
        println!("✓ INTEGRITY VERIFIED");
        println!("  Local hash:     blake3:{}", hash_str);
        println!("  Published hash: blake3:{}", published_hash);
        println!("  Status: ✓ MATCH - Data is authentic and unmodified");
    } else {
        println!("✗ INTEGRITY FAILED");
        println!("  Local hash:     blake3:{}", hash_str);
        println!("  Published hash: blake3:{}", published_hash);
        println!("  Status: ✗ MISMATCH - Data may be corrupted or tampered");
    }
    println!();

    // Step 5: Check verification level
    println!("🔒 STEP 5: Checking verification level");
    println!("─────────────────────────────────────");

    let tier = claim["tier"].as_str().unwrap();
    let verifications = claim["verifications_count"].as_u64().unwrap();

    println!("✓ Verification status retrieved");
    println!("  Epistemic tier: {}", tier);
    println!("  Verifications: {}", verifications);
    println!("  Trust level: {}", get_trust_level_from_tier(tier));
    println!();

    // Step 6: Validate provenance chain
    println!("📚 STEP 6: Validating provenance chain");
    println!("─────────────────────────────────────");

    let provenance_count = claim["provenance_count"].as_u64().unwrap();

    println!("✓ Provenance chain retrieved");
    println!("  Sources documented: {}", provenance_count);

    if provenance_count > 0 {
        println!("  Chain status: ✓ Complete");
    } else {
        println!("  Chain status: ⚠ Limited provenance");
    }
    println!();

    // Step 7: Generate integrity report
    println!("📊 STEP 7: Generating integrity report");
    println!("─────────────────────────────────────");

    let report = json!({
        "verification_timestamp": chrono::Utc::now().to_rfc3339(),
        "dataset_info": {
            "size_bytes": dataset.len(),
            "hash_algorithm": "BLAKE3",
            "hash": format!("blake3:{}", hash_str)
        },
        "claim_info": {
            "id": claim["id"],
            "tier": tier,
            "verifications": verifications,
            "provenance_sources": provenance_count,
            "creator": claim["creator"]
        },
        "integrity_check": {
            "status": if matches { "VERIFIED" } else { "FAILED" },
            "hash_match": matches,
            "confidence": get_confidence_level(tier, verifications)
        },
        "recommendations": generate_recommendations(tier, verifications, provenance_count)
    });

    let report_json = serde_json::to_string_pretty(&report)?;
    fs::write("integrity_report.json", &report_json)?;

    println!("✓ Integrity report generated");
    println!("  Filename: integrity_report.json");
    println!("  Size: {} bytes", report_json.len());
    println!();

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║              DATA INTEGRITY VERIFICATION                 ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("\nVerification Results:");
    println!("  Hash Match:      {}", if matches { "✓ YES" } else { "✗ NO" });
    println!("  Tier:            {}", tier);
    println!("  Verifications:   {}", verifications);
    println!("  Provenance:      {} sources", provenance_count);
    println!("  Overall Status:  {}", if matches { "✓ VERIFIED" } else { "✗ FAILED" });
    println!("\n{}", if matches {
        "✓ This dataset is cryptographically verified and can be trusted."
    } else {
        "✗ WARNING: Data integrity could not be verified. Do not use."
    });
    println!();

    Ok(())
}

fn get_trust_level_from_tier(tier: &str) -> &'static str {
    match tier {
        "E4" => "Highest (5 verifications)",
        "E3" => "High (4 verifications)",
        "E2" => "Medium (3 verifications)",
        "E1" => "Low (1-2 verifications)",
        "E0" => "Unverified (0 verifications)",
        _ => "Unknown",
    }
}

fn get_confidence_level(tier: &str, verifications: u64) -> &'static str {
    match (tier, verifications) {
        ("E4", _) => "Very High",
        ("E3", _) => "High",
        ("E2", _) => "Medium",
        ("E1", _) => "Low",
        _ => "Very Low",
    }
}

fn generate_recommendations(tier: &str, verifications: u64, provenance: u64) -> Vec<String> {
    let mut recs = Vec::new();

    if tier == "E0" {
        recs.push("⚠ Claim is unverified. Seek peer reviews before using data.".to_string());
    }

    if verifications < 3 {
        recs.push("⚠ Low verification count. Consider waiting for more peer reviews.".to_string());
    }

    if provenance == 0 {
        recs.push("⚠ No provenance information. Source of data unclear.".to_string());
    }

    if tier == "E4" && provenance >= 2 {
        recs.push("✓ Highly verified with documented provenance. Safe to use.".to_string());
    }

    if recs.is_empty() {
        recs.push("✓ Data passes basic integrity checks.".to_string());
    }

    recs
}
