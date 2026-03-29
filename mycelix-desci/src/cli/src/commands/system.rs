// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! System commands

use anyhow::Result;
use clap::Subcommand;

use crate::client::ApiClient;
use crate::output::{self, OutputMode};

use super::{HealthResponse, MetricsResponse, VersionResponse};

#[derive(Subcommand)]
pub enum SystemCommand {
    /// Check system health
    Health,

    /// Get system metrics
    Metrics,

    /// Get version information
    Version,
}

pub async fn execute(
    client: ApiClient,
    command: SystemCommand,
    output_mode: OutputMode,
) -> Result<()> {
    match command {
        SystemCommand::Health => check_health(client, output_mode).await,
        SystemCommand::Metrics => get_metrics(client, output_mode).await,
        SystemCommand::Version => get_version(client, output_mode).await,
    }
}

async fn check_health(client: ApiClient, output_mode: OutputMode) -> Result<()> {
    output::info("Checking system health...");

    let response: HealthResponse = client.get("/api/v1/system/health").await?;

    if response.status == "healthy" {
        output::success("System is healthy");
    } else {
        output::warning("System is unhealthy");
    }

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table => {
            output::print_key_value_table(&[
                ("Status", response.status.clone()),
                ("Version", response.version.clone()),
                ("Uptime", format!("{} seconds", response.uptime_seconds)),
            ]);

            println!("\nHealth Checks:");
            let mut table = output::create_table(&["Component", "Status"]);
            for check in &response.checks {
                table.add_row(vec![check.name.clone(), check.status.clone()]);
            }
            println!("{}", table);
        }
        OutputMode::Plain => {
            println!("Status: {}", response.status);
            println!("Version: {}", response.version);
            for check in &response.checks {
                println!("  {}: {}", check.name, check.status);
            }
        }
    }

    Ok(())
}

async fn get_metrics(client: ApiClient, output_mode: OutputMode) -> Result<()> {
    output::info("Retrieving system metrics...");

    let response: MetricsResponse = client.get("/api/v1/system/metrics").await?;

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table => {
            output::print_key_value_table(&[
                ("Uptime", format!("{} seconds", response.uptime_seconds)),
                ("Total Claims", response.total_claims.to_string()),
                ("Total Participants", response.total_participants.to_string()),
                ("Queries Executed", response.queries_executed.to_string()),
                ("Claims Created", response.claims_created.to_string()),
                ("Verifications Added", response.verifications_added.to_string()),
                ("Avg Response Time", format!("{:.2} ms", response.average_response_time_ms)),
            ]);
        }
        OutputMode::Plain => {
            println!("Uptime: {} seconds", response.uptime_seconds);
            println!("Total claims: {}", response.total_claims);
            println!("Queries executed: {}", response.queries_executed);
        }
    }

    Ok(())
}

async fn get_version(client: ApiClient, output_mode: OutputMode) -> Result<()> {
    output::info("Retrieving version information...");

    let response: VersionResponse = client.get("/api/v1/system/version").await?;

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table => {
            output::print_key_value_table(&[
                ("Version", response.version.clone()),
                ("Build Date", response.build_date.clone()),
                ("Git Commit", response.git_commit.clone()),
                ("Rust Version", response.rust_version.clone()),
            ]);
        }
        OutputMode::Plain => {
            println!("Version: {}", response.version);
            println!("Build date: {}", response.build_date);
            println!("Git commit: {}", response.git_commit);
        }
    }

    Ok(())
}
