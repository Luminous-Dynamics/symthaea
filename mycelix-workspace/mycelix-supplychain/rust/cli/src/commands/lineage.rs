// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use colored::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct LineageResponse {
    batch_id: String,
    claims: Vec<serde_json::Value>,
}

pub async fn run(base_url: &str, batch_id: &str, format: &str, _depth: usize) -> Result<()> {
    println!("{} lineage for batch {}...", "Fetching".cyan(), batch_id.yellow());

    let client = reqwest::Client::new();
    let url = format!("{}/v1/batches/{}/lineage", base_url, batch_id);

    let response = client.get(&url).send().await.context("Failed to send request")?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        println!();
        println!("{}", "ℹ No lineage found for this batch".yellow());
        println!("This could mean:");
        println!("  • The batch doesn't exist");
        println!("  • No events have been recorded for this batch");
        println!();
        return Ok(());
    }

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        anyhow::bail!("API error ({}): {}", status, error_text);
    }

    let result: LineageResponse = response.json().await.context("Invalid response")?;

    println!();
    println!("{}", "✓ Lineage retrieved successfully!".green().bold());
    println!();

    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&result.claims)?);
        }
        "dot" => {
            println!("digraph Lineage {{");
            println!("  rankdir=LR;");
            for claim in &result.claims {
                if let Some(id) = claim.get("id").and_then(|v| v.as_str()) {
                    println!("  \"{}\" [label=\"{}\"];", id, &id[..8]);
                    if let Some(prev) = claim.get("lineage")
                        .and_then(|l| l.get("previous_claims"))
                        .and_then(|p| p.as_array())
                    {
                        for parent_id in prev {
                            if let Some(parent) = parent_id.as_str() {
                                println!("  \"{}\" -> \"{}\";", parent, id);
                            }
                        }
                    }
                }
            }
            println!("}}");
        }
        "mermaid" => {
            println!("graph LR");
            for claim in &result.claims {
                if let Some(id) = claim.get("id").and_then(|v| v.as_str()) {
                    let short_id = &id[..8];
                    if let Some(prev) = claim.get("lineage")
                        .and_then(|l| l.get("previous_claims"))
                        .and_then(|p| p.as_array())
                    {
                        for parent_id in prev {
                            if let Some(parent) = parent_id.as_str() {
                                println!("  {} --> {}", &parent[..8], short_id);
                            }
                        }
                    } else {
                        println!("  {}", short_id);
                    }
                }
            }
        }
        _ => anyhow::bail!("Unknown format: {} (supported: json, dot, mermaid)", format),
    }

    println!();
    println!("  {} {}", "Total claims:".cyan(), result.claims.len());
    println!();

    Ok(())
}
