// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use mycelix_desci_core::{claims::DesciClaim, Result};
use std::path::PathBuf;
use tracing::info;

pub async fn execute(
    category: Option<String>,
    min_tier: Option<String>,
    keywords: Option<String>,
    format: String,
    limit: usize,
) -> Result<()> {
    info!("Querying claims with filters");

    // Load all claims from local storage
    let claims_dir = PathBuf::from(".mycelix/claims");

    if !claims_dir.exists() {
        println!("No claims found. Upload some datasets first!");
        return Ok(());
    }

    let mut claims = Vec::new();

    // Read all claim files
    let entries = std::fs::read_dir(claims_dir).map_err(|e| {
        mycelix_desci_core::Error::Generic(format!("Failed to read claims directory: {}", e))
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| {
            mycelix_desci_core::Error::Generic(format!("Failed to read directory entry: {}", e))
        })?;

        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(claim) = DesciClaim::from_json(&content) {
                    claims.push(claim);
                }
            }
        }
    }

    // Apply filters
    if let Some(cat) = &category {
        claims.retain(|c| c.content.category.to_lowercase().contains(&cat.to_lowercase()));
    }

    if let Some(tier_str) = &min_tier {
        if let Ok(tier) = parse_tier(tier_str) {
            claims.retain(|c| c.epistemic_tier >= tier);
        }
    }

    if let Some(kw) = &keywords {
        let keywords_lower = kw.to_lowercase();
        claims.retain(|c| {
            c.content.keywords.iter().any(|k| k.to_lowercase().contains(&keywords_lower))
                || c.content.description.to_lowercase().contains(&keywords_lower)
        });
    }

    // Limit results
    claims.truncate(limit);

    // Display results
    match format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&claims).map_err(|e| {
                mycelix_desci_core::Error::Generic(format!("Failed to serialize: {}", e))
            })?;
            println!("{}", json);
        }
        "table" => {
            println!("\nFound {} claims:\n", claims.len());
            println!("{:<38} {:<6} {:<15} {}", "ID", "Tier", "Category", "Description");
            println!("{}", "-".repeat(100));

            for claim in claims {
                println!(
                    "{:<38} {:<6} {:<15} {}",
                    claim.id.to_string(),
                    format!("{:?}", claim.epistemic_tier),
                    truncate(&claim.content.category, 15),
                    truncate(&claim.content.description, 40)
                );
            }
        }
        _ => {
            for claim in claims {
                println!("\nClaim: {}", claim.id);
                println!("  Tier: {:?}", claim.epistemic_tier);
                println!("  Category: {}", claim.content.category);
                println!("  Description: {}", claim.content.description);
            }
        }
    }

    Ok(())
}

fn parse_tier(tier_str: &str) -> Result<mycelix_desci_core::claims::EpistemicTier> {
    use mycelix_desci_core::claims::EpistemicTier;

    match tier_str.to_uppercase().as_str() {
        "E0" => Ok(EpistemicTier::E0),
        "E1" => Ok(EpistemicTier::E1),
        "E2" => Ok(EpistemicTier::E2),
        "E3" => Ok(EpistemicTier::E3),
        "E4" => Ok(EpistemicTier::E4),
        _ => Err(mycelix_desci_core::Error::Generic(format!(
            "Invalid tier: {}",
            tier_str
        ))),
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
