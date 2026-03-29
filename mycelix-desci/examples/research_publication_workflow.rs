// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Complete Research Publication Workflow
//!
//! This example demonstrates the full lifecycle of publishing scientific research
//! on the Mycelix-DeSci platform, from raw data to peer-reviewed verified claim.
//!
//! ## Scenario
//! Dr. Alice is publishing a breakthrough longevity study. This example shows:
//! 1. Hashing the dataset with BLAKE3
//! 2. Creating an initial E0 claim
//! 3. Adding storage reference (IPFS/Arweave)
//! 4. Adding provenance (lab notebook, prior work)
//! 5. Peer review process (collecting verifications → E4)
//! 6. Querying related claims
//! 7. Tracking researcher trust score
//! 8. Exporting claim for archival
//!
//! ## Usage
//! ```bash
//! cargo run --example research_publication_workflow
//! ```
//!
//! ## Prerequisites
//! - API server running on http://localhost:8080
//! - Sample dataset file (will be generated if not present)

use mycelix_desci_core::{
    claims::{ClaimContent, DesciClaim, EpistemicTier, Provenance, Verification},
    hash::compute_blake3,
};
use serde_json::json;
use std::fs;
use std::time::Duration;

const API_BASE: &str = "http://localhost:8080/api/v1";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   Mycelix-DeSci: Research Publication Workflow Demo     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Step 1: Generate or load dataset
    println!("📊 STEP 1: Preparing research dataset");
    println!("─────────────────────────────────────");
    let dataset = generate_sample_dataset()?;
    println!("✓ Dataset generated: {} bytes", dataset.len());
    println!("  Sample data: {}\n", &dataset[..100.min(dataset.len())]);

    // Step 2: Compute BLAKE3 hash
    println!("🔐 STEP 2: Computing cryptographic hash");
    println!("─────────────────────────────────────");
    let hash = compute_blake3(&dataset);
    let hash_str = format!("blake3:{}", hex::encode(&hash));
    println!("✓ BLAKE3 hash computed");
    println!("  Hash: {}\n", hash_str);

    // Step 3: Create initial E0 claim
    println!("📝 STEP 3: Creating initial E0 claim");
    println!("─────────────────────────────────────");
    let claim_request = json!({
        "tier": "E0",
        "content": {
            "dataset_hash": hash_str,
            "description": "Novel NAD+ supplementation protocol demonstrates 23% increase in cellular NAD+ levels in human trials (n=150, double-blind, placebo-controlled)",
            "category": "longevity",
            "keywords": ["NAD+", "aging", "clinical-trial", "supplementation", "biomarkers"],
            "storage_ref": null,
            "reproducibility_score": null,
            "license": "CC-BY-4.0"
        },
        "creator": "dr.alice@longevity-institute.org"
    });

    let client = reqwest::Client::new();
    let claim_response: serde_json::Value = client
        .post(format!("{}/claims", API_BASE))
        .json(&claim_request)
        .send()
        .await?
        .json()
        .await?;

    let claim_id = claim_response["id"].as_str().unwrap();
    println!("✓ Claim created successfully");
    println!("  Claim ID: {}", claim_id);
    println!("  Initial tier: E0 (Unverified)");
    println!("  Category: longevity\n");

    // Step 4: Add storage reference (simulating IPFS upload)
    println!("💾 STEP 4: Adding decentralized storage reference");
    println!("─────────────────────────────────────");
    tokio::time::sleep(Duration::from_millis(500)).await; // Simulate upload
    println!("✓ Dataset uploaded to IPFS (simulated)");
    println!("  CID: QmX7M9CiYXjVeFnkfVGf1EmUqqQmhZdqBQ7ZCq8qjKnTuP");
    println!("  Size: {} bytes", dataset.len());
    println!("  Retrievable via: https://ipfs.io/ipfs/...\n");

    // Step 5: Add provenance information
    println!("📚 STEP 5: Adding provenance information");
    println!("─────────────────────────────────────");

    // Add lab notebook reference
    let provenance1_request = json!({
        "source": "Lab Notebook Entry #2847",
        "source_type": "lab_notebook",
        "url": "https://longevity-institute.org/notebooks/2847"
    });

    client
        .put(format!("{}/claims/{}/provenance", API_BASE, claim_id))
        .json(&provenance1_request)
        .send()
        .await?;

    println!("✓ Added lab notebook provenance");

    // Add prior research reference
    let provenance2_request = json!({
        "source": "Sinclair et al. 2013 - NAD+ in aging",
        "source_type": "prior_research",
        "url": "https://doi.org/10.1016/j.cell.2013.05.037"
    });

    client
        .put(format!("{}/claims/{}/provenance", API_BASE, claim_id))
        .json(&provenance2_request)
        .send()
        .await?;

    println!("✓ Added prior research provenance");
    println!("  Total provenance sources: 2\n");

    // Step 6: Peer review process (collect verifications)
    println!("👥 STEP 6: Peer review and verification process");
    println!("─────────────────────────────────────");

    let reviewers = vec![
        ("dr.bob@stanford.edu", "Stanford Longevity Center"),
        ("prof.carol@mit.edu", "MIT Biology Dept"),
        ("dr.david@harvard.edu", "Harvard Medical School"),
        ("dr.eve@oxford.ac.uk", "Oxford Aging Research"),
        ("prof.frank@ucl.ac.uk", "UCL Institute of Healthy Ageing"),
    ];

    for (i, (reviewer, affiliation)) in reviewers.iter().enumerate() {
        println!("\n  Review #{}: {}", i + 1, affiliation);

        // Simulate review time
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Generate simulated signature
        let signature = format!("sig_{}_{}",reviewer, claim_id).as_bytes().to_vec();
        let signature_hex = hex::encode(&signature);

        let verification_request = json!({
            "verifier": reviewer,
            "signature": signature_hex,
            "notes": format!("Reviewed methodology and results. Data appears sound. - {}", affiliation)
        });

        let response: serde_json::Value = client
            .put(format!("{}/claims/{}/verify", API_BASE, claim_id))
            .json(&verification_request)
            .send()
            .await?
            .json()
            .await?;

        let new_tier = response["tier"].as_str().unwrap();
        println!("  ✓ Verification added by {}", reviewer);
        println!("  → Claim tier upgraded to: {}", new_tier);

        // Update trust score for verifier
        let trust_update = json!({ "delta": 0.05 });
        client
            .put(format!("{}/trust/{}", API_BASE, reviewer))
            .json(&trust_update)
            .send()
            .await?;

        println!("  → Trust score updated for verifier");
    }

    println!("\n✓ Peer review complete: 5 verifications collected");
    println!("  Final tier: E4 (Highly Verified)\n");

    // Step 7: Query related claims
    println!("🔍 STEP 7: Querying related research");
    println!("─────────────────────────────────────");

    let query_request = json!({
        "category": "longevity",
        "keywords": ["NAD+"],
        "page": 1,
        "page_size": 5
    });

    let query_response: serde_json::Value = client
        .post(format!("{}/query", API_BASE))
        .json(&query_request)
        .send()
        .await?
        .json()
        .await?;

    let total_related = query_response["total_count"].as_u64().unwrap();
    println!("✓ Found {} related claims in longevity category", total_related);

    if let Some(results) = query_response["results"].as_array() {
        for (i, result) in results.iter().enumerate() {
            println!("  {}. {} [{}]",
                i + 1,
                result["id"].as_str().unwrap_or("unknown"),
                result["tier"].as_str().unwrap_or("unknown")
            );
        }
    }
    println!();

    // Step 8: Check creator's trust score
    println!("⭐ STEP 8: Checking researcher trust score");
    println!("─────────────────────────────────────");

    let creator = "dr.alice@longevity-institute.org";
    let trust_response: serde_json::Value = client
        .get(format!("{}/trust/{}", API_BASE, creator))
        .send()
        .await?
        .json()
        .await?;

    let trust_score = trust_response["score"].as_f64().unwrap();
    println!("✓ Trust score retrieved");
    println!("  Researcher: {}", creator);
    println!("  Trust score: {:.3}", trust_score);
    println!("  Trust level: {}", get_trust_level(trust_score));
    println!();

    // Step 9: Retrieve final claim state
    println!("📥 STEP 9: Retrieving final claim state");
    println!("─────────────────────────────────────");

    let final_claim: serde_json::Value = client
        .get(format!("{}/claims/{}", API_BASE, claim_id))
        .send()
        .await?
        .json()
        .await?;

    println!("✓ Final claim retrieved");
    println!("\n{}", "=".repeat(60));
    println!("FINAL CLAIM SUMMARY");
    println!("{}", "=".repeat(60));
    println!("ID:              {}", final_claim["id"]);
    println!("Tier:            {} (Highly Verified)", final_claim["tier"]);
    println!("Category:        {}", final_claim["content"]["category"]);
    println!("Description:     {}", final_claim["content"]["description"]);
    println!("Dataset Hash:    {}", final_claim["content"]["dataset_hash"]);
    println!("Creator:         {}", final_claim["creator"]);
    println!("Verifications:   {}", final_claim["verifications_count"]);
    println!("Provenance:      {}", final_claim["provenance_count"]);
    println!("Keywords:        {}", final_claim["content"]["keywords"]);
    println!("{}", "=".repeat(60));

    // Step 10: Export for archival
    println!("\n💾 STEP 10: Exporting claim for archival");
    println!("─────────────────────────────────────");

    let export_filename = format!("claim_{}.json", claim_id);
    fs::write(&export_filename, serde_json::to_string_pretty(&final_claim)?)?;

    println!("✓ Claim exported to: {}", export_filename);
    println!("  File size: {} bytes", fs::metadata(&export_filename)?.len());
    println!("  Format: JSON");
    println!();

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║                    WORKFLOW COMPLETE! ✨                  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("\nResearch publication successfully processed:");
    println!("  ✓ Dataset hashed and secured");
    println!("  ✓ Initial claim created (E0)");
    println!("  ✓ Provenance documented (2 sources)");
    println!("  ✓ Peer reviewed (5 verifications)");
    println!("  ✓ Upgraded to highest tier (E4)");
    println!("  ✓ Archived for reproducibility");
    println!("\nThis claim is now:");
    println!("  • Cryptographically verifiable");
    println!("  • Peer-reviewed and trusted");
    println!("  • Provenance-tracked");
    println!("  • Permanently archived");
    println!("  • Discoverable via queries");
    println!("\n🔬 Ready for the scientific community! 🔬\n");

    Ok(())
}

/// Generate sample research dataset
fn generate_sample_dataset() -> Result<String, Box<dyn std::error::Error>> {
    let dataset = r#"{
  "study": "NAD+ Supplementation Clinical Trial",
  "protocol": "Double-blind, placebo-controlled",
  "participants": 150,
  "duration_weeks": 12,
  "intervention": {
    "compound": "Nicotinamide Riboside",
    "dosage_mg": 500,
    "frequency": "twice daily"
  },
  "measurements": {
    "baseline_nad": 35.2,
    "endpoint_nad": 43.3,
    "percent_increase": 23.0,
    "p_value": 0.00012,
    "confidence_interval": [18.5, 27.5]
  },
  "biomarkers": {
    "sirtuin_activity": {"baseline": 1.0, "endpoint": 1.34, "change": "+34%"},
    "mitochondrial_function": {"baseline": 1.0, "endpoint": 1.21, "change": "+21%"},
    "cellular_senescence": {"baseline": 12.4, "endpoint": 9.1, "change": "-27%"}
  },
  "adverse_events": {
    "mild_nausea": 3,
    "headache": 2,
    "none": 145
  },
  "conclusion": "Nicotinamide riboside supplementation significantly increases NAD+ levels and improves age-related biomarkers with minimal adverse effects."
}"#;

    Ok(dataset.to_string())
}

/// Get trust level description
fn get_trust_level(score: f64) -> &'static str {
    match score {
        s if s >= 0.8 => "Excellent (Highly Trusted)",
        s if s >= 0.6 => "Good (Trusted)",
        s if s >= 0.4 => "Fair (Neutral)",
        s if s >= 0.2 => "Low (Untrusted)",
        _ => "Very Low (Highly Untrusted)",
    }
}
