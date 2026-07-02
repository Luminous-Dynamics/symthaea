// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use colored::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct ClaimResponse {
    claim: serde_json::Value,
    vc_jwt: String,
}

pub async fn run(base_url: &str, claim_id: &str) -> Result<()> {
    println!("{} claim {}...", "Fetching".cyan(), claim_id.yellow());

    let client = reqwest::Client::new();
    let url = format!("{}/v1/claims/{}", base_url, claim_id);

    let response = client.get(&url).send().await.context("Failed to send request")?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        anyhow::bail!("Claim not found: {}", claim_id);
    }

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        anyhow::bail!("API error ({}): {}", status, error_text);
    }

    let result: ClaimResponse = response.json().await.context("Invalid response")?;

    println!();
    println!("{}", "✓ Claim retrieved successfully!".green().bold());
    println!();
    println!("{}", serde_json::to_string_pretty(&result.claim)?);
    println!();
    println!("{}", "VC JWT:".cyan().bold());
    println!("{}", result.vc_jwt.dimmed());
    println!();

    Ok(())
}
