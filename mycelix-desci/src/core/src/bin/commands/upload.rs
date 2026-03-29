// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use mycelix_desci_core::{
    claims::{ClaimContent, DesciClaim, EpistemicTier, Provenance},
    hash, storage::{MemoryStorage, StorageBackend},
    Result,
};
use std::path::PathBuf;
use tracing::info;

pub async fn execute(
    file: PathBuf,
    tier: String,
    category: String,
    description: String,
    provenance: Option<String>,
    license: Option<String>,
    keywords: Option<String>,
) -> Result<()> {
    info!("Uploading dataset from {:?}", file);

    // Verify file exists
    if !file.exists() {
        return Err(mycelix_desci_core::Error::Generic(format!(
            "File does not exist: {}",
            file.display()
        )));
    }

    // Parse epistemic tier
    let epistemic_tier = parse_tier(&tier)?;

    // Calculate file hash
    let file_hash = hash::hash_file(&file)?;

    println!("Calculating dataset hash...");
    println!("Hash: {}", file_hash.to_string());

    // Parse keywords
    let keyword_list = keywords
        .map(|k| k.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    // Create claim content
    let content = ClaimContent {
        dataset_hash: file_hash.to_string(),
        description,
        category,
        keywords: keyword_list,
        storage_ref: None, // TODO: Upload to IPFS
        reproducibility_score: None,
        license,
    };

    // Create claim
    let mut claim = DesciClaim::new(
        epistemic_tier,
        content,
        "local_creator".to_string(), // TODO: Use actual identity
    );

    // Add provenance if provided
    if let Some(prov_str) = provenance {
        let prov = Provenance::new(prov_str, "manual".to_string());
        claim.add_provenance(prov);
    }

    // Store claim (using memory storage for now)
    let storage = MemoryStorage::new();
    let claim_id = storage.store(&claim).await?;

    println!("\n✓ Dataset uploaded successfully!");
    println!("  Claim ID: {}", claim_id);
    println!("  Tier: {:?}", epistemic_tier);
    println!("  Category: {}", claim.content.category);
    println!("  Hash: {}", claim.content.dataset_hash);

    // Save claim to local file for persistence
    let claim_file = PathBuf::from(format!(".mycelix/claims/{}.json", claim_id));
    if let Some(parent) = claim_file.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let claim_json = claim.to_json().map_err(|e| {
        mycelix_desci_core::Error::Generic(format!("Failed to serialize claim: {}", e))
    })?;

    std::fs::write(&claim_file, claim_json).map_err(|e| {
        mycelix_desci_core::Error::Generic(format!("Failed to write claim file: {}", e))
    })?;

    println!("  Saved to: {}", claim_file.display());

    Ok(())
}

fn parse_tier(tier_str: &str) -> Result<EpistemicTier> {
    match tier_str.to_uppercase().as_str() {
        "E0" => Ok(EpistemicTier::E0),
        "E1" => Ok(EpistemicTier::E1),
        "E2" => Ok(EpistemicTier::E2),
        "E3" => Ok(EpistemicTier::E3),
        "E4" => Ok(EpistemicTier::E4),
        _ => Err(mycelix_desci_core::Error::Generic(format!(
            "Invalid epistemic tier: {}. Must be E0-E4",
            tier_str
        ))),
    }
}
