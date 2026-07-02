// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use colored::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct HealthResponse {
    status: String,
    version: String,
    timestamp: String,
}

pub async fn run(base_url: &str) -> Result<()> {
    println!("{} service health...", "Checking".cyan());

    let client = reqwest::Client::new();
    let url = format!("{}/health", base_url);

    let start = std::time::Instant::now();
    let response = client.get(&url).send().await.context("Failed to connect to service")?;
    let duration = start.elapsed();

    if !response.status().is_success() {
        let status = response.status();
        anyhow::bail!("Service unhealthy (status: {})", status);
    }

    let result: HealthResponse = response.json().await.context("Invalid response")?;

    println!();
    println!("{}", "✓ Service is healthy!".green().bold());
    println!();
    println!("  {} {}", "Status:".cyan(), result.status.green());
    println!("  {} {}", "Version:".cyan(), result.version.bright_white());
    println!("  {} {:?}", "Response time:".cyan(), duration);
    println!("  {} {}", "URL:".cyan(), base_url.dimmed());
    println!();

    Ok(())
}
