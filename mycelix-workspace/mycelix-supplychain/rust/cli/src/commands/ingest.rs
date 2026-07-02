// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use claim_model::SupplyEventVC;
use colored::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize)]
struct IngestResponse {
    vc_jwt: String,
    claim_id: String,
    lineage_hash: String,
    previous_claims: Option<Vec<String>>,
}

pub async fn run(base_url: &str, file: Option<&Path>, stdin: bool) -> Result<()> {
    // Read event JSON
    let event_json = if stdin || file.map(|p| p.as_os_str() == "-").unwrap_or(false) {
        println!("{}", "Reading event from stdin...".cyan());
        std::io::read_to_string(std::io::stdin()).context("Failed to read from stdin")?
    } else if let Some(path) = file {
        std::fs::read_to_string(path)
            .context(format!("Failed to read file: {}", path.display()))?
    } else {
        anyhow::bail!("Either --file or --stdin must be specified");
    };

    // Parse and validate
    let vc: SupplyEventVC = serde_json::from_str(&event_json).context("Invalid event JSON")?;

    println!(
        "{} event for batch {}",
        "Ingesting".cyan(),
        vc.credential_subject.batch_id.yellow()
    );

    // Send to API
    let client = reqwest::Client::new();
    let url = format!("{}/v1/events", base_url);

    let response = client
        .post(&url)
        .json(&vc)
        .send()
        .await
        .context("Failed to send request")?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        anyhow::bail!("API error ({}): {}", status, error_text);
    }

    let result: IngestResponse = response.json().await.context("Invalid response")?;

    println!();
    println!("{}", "✓ Event ingested successfully!".green().bold());
    println!();
    println!("  {} {}", "Claim ID:".cyan(), result.claim_id.bright_white());
    println!(
        "  {} {}",
        "Lineage hash:".cyan(),
        result.lineage_hash.bright_white()
    );

    if let Some(prev) = result.previous_claims {
        if !prev.is_empty() {
            println!("  {} {}", "Previous claims:".cyan(), prev.len());
            for (i, id) in prev.iter().enumerate() {
                println!("    {}. {}", i + 1, id.dimmed());
            }
        }
    }

    println!();

    Ok(())
}
