// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use mycelix_desci_core::{claims::DesciClaim, hash, Result};
use std::path::PathBuf;
use tracing::info;

pub async fn execute(claim_id: String, file: Option<PathBuf>) -> Result<()> {
    info!("Verifying claim: {}", claim_id);

    // Load claim from local storage
    let claim_file = PathBuf::from(format!(".mycelix/claims/{}.json", claim_id));

    if !claim_file.exists() {
        return Err(mycelix_desci_core::Error::NotFound(format!(
            "Claim not found: {}",
            claim_id
        )));
    }

    let claim_json = std::fs::read_to_string(&claim_file).map_err(|e| {
        mycelix_desci_core::Error::Generic(format!("Failed to read claim file: {}", e))
    })?;

    let claim: DesciClaim = DesciClaim::from_json(&claim_json).map_err(|e| {
        mycelix_desci_core::Error::Generic(format!("Failed to parse claim: {}", e))
    })?;

    println!("Claim Information:");
    println!("  ID: {}", claim.id);
    println!("  Tier: {:?}", claim.epistemic_tier);
    println!("  Category: {}", claim.content.category);
    println!("  Description: {}", claim.content.description);
    println!("  Hash: {}", claim.content.dataset_hash);
    println!("  Verifications: {}", claim.verifications.len());

    // Verify tier requirements
    let is_valid_tier = claim.is_valid_for_tier();
    println!("\nTier Validation:");
    println!("  Required verifications: {}", claim.epistemic_tier.min_verifications());
    println!("  Current verifications: {}", claim.verifications.len());
    println!("  Status: {}", if is_valid_tier { "✓ Valid" } else { "✗ Invalid - needs more verifications" });

    // If file provided, verify hash
    if let Some(file_path) = file {
        println!("\nHash Verification:");

        if !file_path.exists() {
            println!("  ✗ File not found: {}", file_path.display());
        } else {
            // Parse the stored hash
            let expected_hash = hash::Hash::from_string(&claim.content.dataset_hash)?;
            let computed_hash = hash::hash_file_with_algorithm(&file_path, expected_hash.algorithm)?;

            if computed_hash.bytes == expected_hash.bytes {
                println!("  ✓ Hash matches!");
                println!("    File: {}", file_path.display());
                println!("    Hash: {}", computed_hash.hex());
            } else {
                println!("  ✗ Hash mismatch!");
                println!("    Expected: {}", expected_hash.hex());
                println!("    Computed: {}", computed_hash.hex());
            }
        }
    }

    // Show provenance
    if !claim.provenance.is_empty() {
        println!("\nProvenance:");
        for (i, prov) in claim.provenance.iter().enumerate() {
            println!("  {}. {} ({})", i + 1, prov.source, prov.source_type);
            if let Some(url) = &prov.url {
                println!("     URL: {}", url);
            }
        }
    }

    Ok(())
}
