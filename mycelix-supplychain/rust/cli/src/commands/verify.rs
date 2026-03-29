// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use colored::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize)]
struct VerifyRequest {
    vc_jwt: String,
}

#[derive(Deserialize)]
struct VerifyResponse {
    valid: bool,
    claim_id: Option<String>,
    issuer: Option<String>,
    error: Option<String>,
}

pub async fn run(base_url: &str, jwt: Option<&str>, file: Option<&Path>) -> Result<()> {
    let vc_jwt = if let Some(jwt_str) = jwt {
        jwt_str.to_string()
    } else if let Some(path) = file {
        std::fs::read_to_string(path)
            .context(format!("Failed to read file: {}", path.display()))?
            .trim()
            .to_string()
    } else {
        anyhow::bail!("Either --jwt or --file must be specified");
    };

    println!("{}", "Verifying Verifiable Credential...".cyan());

    let client = reqwest::Client::new();
    let url = format!("{}/v1/verify", base_url);

    let request = VerifyRequest { vc_jwt };

    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .context("Failed to send request")?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        anyhow::bail!("API error ({}): {}", status, error_text);
    }

    let result: VerifyResponse = response.json().await.context("Invalid response")?;

    println!();
    if result.valid {
        println!("{}", "✓ Verification successful!".green().bold());
        println!();
        if let Some(claim_id) = result.claim_id {
            println!("  {} {}", "Claim ID:".cyan(), claim_id.bright_white());
        }
        if let Some(issuer) = result.issuer {
            println!("  {} {}", "Issuer:".cyan(), issuer.bright_white());
        }
    } else {
        println!("{}", "✗ Verification failed!".red().bold());
        println!();
        if let Some(error) = result.error {
            println!("  {} {}", "Error:".red(), error);
        }
    }
    println!();

    Ok(())
}
