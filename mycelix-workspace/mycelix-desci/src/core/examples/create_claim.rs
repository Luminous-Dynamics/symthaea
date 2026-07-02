// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Example: Creating and Saving an Epistemic Claim
//!
//! This example demonstrates how to:
//! 1. Create a dataset claim with metadata
//! 2. Add provenance information
//! 3. Save the claim to storage
//!
//! Run with: cargo run --example create_claim

use mycelix_desci_core::{
    claims::{ClaimContent, DesciClaim, EpistemicTier, Provenance},
    storage::{MemoryStorage, StorageBackend},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Creating an Epistemic Claim ===\n");

    // 1. Create claim content with dataset metadata
    let content = ClaimContent {
        dataset_hash: "blake3:abc123def456...".to_string(),
        description: "Reproducible study on NAD+ supplementation effects in aging mice".to_string(),
        category: "longevity".to_string(),
        keywords: vec![
            "NAD+".to_string(),
            "aging".to_string(),
            "supplementation".to_string(),
            "mice".to_string(),
        ],
        storage_ref: Some("ipfs://QmX8Y9Z...".to_string()),
        reproducibility_score: Some(0.92),
        license: Some("CC-BY-4.0".to_string()),
    };

    println!("Created claim content:");
    println!("  Category: {}", content.category);
    println!("  Description: {}", content.description);
    println!("  Keywords: {:?}", content.keywords);
    println!();

    // 2. Create the claim with E3 tier (reproducible)
    let mut claim = DesciClaim::new(
        EpistemicTier::E3,
        content,
        "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK".to_string(),
    );

    println!("Initial claim:");
    println!("  ID: {}", claim.id);
    println!("  Tier: {:?} ({})", claim.epistemic_tier, claim.epistemic_tier.description());
    println!("  Creator: {}", claim.creator);
    println!();

    // 3. Add provenance information
    let prov1 = Provenance::new(
        "Lab Notebook ID:2024-042".to_string(),
        "laboratory_record".to_string(),
    )
    .with_url("https://lab.example.edu/notebook/2024-042".to_string())
    .with_metadata(
        "institution",
        serde_json::json!("Example University"),
    );

    let prov2 = Provenance::new(
        "PubChem ID:123456".to_string(),
        "database".to_string(),
    )
    .with_url("https://pubchem.ncbi.nlm.nih.gov/compound/123456".to_string());

    claim.add_provenance(prov1);
    claim.add_provenance(prov2);

    println!("Added provenance chain:");
    for (i, prov) in claim.provenance.iter().enumerate() {
        println!("  {}. {} ({})", i + 1, prov.source, prov.source_type);
    }
    println!();

    // 4. Validate tier requirements
    println!("Tier validation:");
    println!("  Required verifications: {}", claim.epistemic_tier.min_verifications());
    println!("  Current verifications: {}", claim.verifications.len());
    println!("  Valid: {}", claim.is_valid_for_tier());
    println!();

    // 5. Save to storage
    let storage = MemoryStorage::new();
    let claim_id = storage.store(&claim).await?;

    println!("Claim saved to storage:");
    println!("  Storage ID: {}", claim_id);
    println!();

    // 6. Retrieve from storage
    let retrieved_claim = storage.retrieve(&claim_id).await?;

    println!("Retrieved claim:");
    println!("  ID matches: {}", retrieved_claim.id == claim.id);
    println!("  Description: {}", retrieved_claim.content.description);
    println!();

    // 7. Export to JSON
    let json = claim.to_json()?;
    println!("JSON export (first 200 chars):");
    println!("{}", &json[..200.min(json.len())]);
    println!("...\n");

    println!("✓ Example completed successfully!");

    Ok(())
}
