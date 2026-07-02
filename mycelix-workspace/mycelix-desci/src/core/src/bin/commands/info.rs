// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use mycelix_desci_core::{claims::DesciClaim, Result};
use std::path::PathBuf;
use tracing::info;

pub async fn execute(claim_id: String, format: String) -> Result<()> {
    info!("Fetching info for claim: {}", claim_id);

    // Load claim
    let claim_file = PathBuf::from(format!(".mycelix/claims/{}.json", claim_id));

    if !claim_file.exists() {
        return Err(mycelix_desci_core::Error::NotFound(format!(
            "Claim not found: {}",
            claim_id
        )));
    }

    let claim_json = std::fs::read_to_string(&claim_file).map_err(|e| {
        mycelix_desci_core::Error::Generic(format!("Failed to read claim: {}", e))
    })?;

    let claim: DesciClaim = DesciClaim::from_json(&claim_json).map_err(|e| {
        mycelix_desci_core::Error::Generic(format!("Failed to parse claim: {}", e))
    })?;

    match format.as_str() {
        "json" => {
            println!("{}", claim.to_json().unwrap_or_default());
        }
        _ => {
            println!("Claim Information");
            println!("{}", "=".repeat(80));
            println!("ID: {}", claim.id);
            println!("Epistemic Tier: {:?} ({})", claim.epistemic_tier, claim.epistemic_tier.description());
            println!("Category: {}", claim.content.category);
            println!("Description: {}", claim.content.description);
            println!("\nDataset:");
            println!("  Hash: {}", claim.content.dataset_hash);

            if let Some(storage_ref) = &claim.content.storage_ref {
                println!("  Storage: {}", storage_ref);
            }

            if let Some(score) = claim.content.reproducibility_score {
                println!("  Reproducibility: {:.2}%", score * 100.0);
            }

            if let Some(license) = &claim.content.license {
                println!("  License: {}", license);
            }

            if !claim.content.keywords.is_empty() {
                println!("\nKeywords:");
                for keyword in &claim.content.keywords {
                    println!("  - {}", keyword);
                }
            }

            println!("\nMetadata:");
            println!("  Creator: {}", claim.creator);
            println!("  Created: {}", claim.created_at);
            println!("  Updated: {}", claim.updated_at);

            println!("\nVerification:");
            println!("  Verifications: {} / {} required",
                claim.verifications.len(),
                claim.epistemic_tier.min_verifications()
            );

            println!("  Status: {}",
                if claim.is_valid_for_tier() {
                    "✓ Valid"
                } else {
                    "⚠ Needs more verifications"
                }
            );

            if !claim.verifications.is_empty() {
                println!("\nVerifiers:");
                for (i, ver) in claim.verifications.iter().enumerate() {
                    println!("  {}. {} ({})", i + 1, ver.verifier, ver.timestamp);
                    if let Some(notes) = &ver.notes {
                        println!("     Notes: {}", notes);
                    }
                }
            }

            if !claim.provenance.is_empty() {
                println!("\nProvenance Chain:");
                for (i, prov) in claim.provenance.iter().enumerate() {
                    println!("  {}. {} ({})", i + 1, prov.source, prov.source_type);
                    if let Some(url) = &prov.url {
                        println!("     URL: {}", url);
                    }
                    println!("     Timestamp: {}", prov.timestamp);
                }
            }
        }
    }

    Ok(())
}
