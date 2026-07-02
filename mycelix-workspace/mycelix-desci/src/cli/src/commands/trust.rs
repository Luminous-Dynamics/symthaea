// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust commands

use anyhow::Result;
use clap::Subcommand;
use serde::Serialize;

use crate::client::ApiClient;
use crate::output::{self, OutputMode};

use super::TrustScoreResponse;

#[derive(Subcommand)]
pub enum TrustCommand {
    /// Get trust score for a participant
    Get {
        /// Participant identifier
        participant: String,
    },

    /// Update trust score for a participant
    Update {
        /// Participant identifier
        participant: String,

        /// Score delta (-1.0 to 1.0)
        #[arg(long)]
        delta: f64,
    },

    /// Get trust network statistics
    Stats,
}

#[derive(Debug, Serialize)]
struct UpdateTrustScoreRequest {
    delta: f64,
}

pub async fn execute(
    client: ApiClient,
    command: TrustCommand,
    output_mode: OutputMode,
) -> Result<()> {
    match command {
        TrustCommand::Get { participant } => {
            get_trust_score(client, &participant, output_mode).await
        }
        TrustCommand::Update { participant, delta } => {
            update_trust_score(client, &participant, delta, output_mode).await
        }
        TrustCommand::Stats => get_trust_stats(client, output_mode).await,
    }
}

async fn get_trust_score(
    client: ApiClient,
    participant: &str,
    output_mode: OutputMode,
) -> Result<()> {
    output::info(&format!("Retrieving trust score for {}", participant));

    let response: TrustScoreResponse = client
        .get(&format!("/api/v1/trust/{}", participant))
        .await?;

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table => {
            output::print_key_value_table(&[
                ("Participant", response.participant.clone()),
                ("Score", format!("{:.3}", response.score)),
                ("Last Updated", response.last_updated.clone()),
            ]);
        }
        OutputMode::Plain => {
            println!("{}: {:.3}", response.participant, response.score);
        }
    }

    Ok(())
}

async fn update_trust_score(
    client: ApiClient,
    participant: &str,
    delta: f64,
    output_mode: OutputMode,
) -> Result<()> {
    output::info(&format!(
        "Updating trust score for {} (delta: {:+.3})",
        participant, delta
    ));

    let request = UpdateTrustScoreRequest { delta };

    let response: TrustScoreResponse = client
        .put(&format!("/api/v1/trust/{}", participant), &request)
        .await?;

    output::success(&format!("New score: {:.3}", response.score));

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table => {
            output::print_key_value_table(&[
                ("Participant", response.participant.clone()),
                ("New Score", format!("{:.3}", response.score)),
                ("Last Updated", response.last_updated.clone()),
            ]);
        }
        OutputMode::Plain => {
            println!("New score: {:.3}", response.score);
        }
    }

    Ok(())
}

async fn get_trust_stats(client: ApiClient, output_mode: OutputMode) -> Result<()> {
    output::info("Retrieving trust network statistics...");

    #[derive(serde::Deserialize, serde::Serialize)]
    struct TrustStatsResponse {
        total_participants: usize,
        average_score: f64,
        median_score: f64,
        highest_score: f64,
        lowest_score: f64,
    }

    let response: TrustStatsResponse = client.get("/api/v1/trust/stats").await?;

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table => {
            output::print_key_value_table(&[
                ("Total Participants", response.total_participants.to_string()),
                ("Average Score", format!("{:.3}", response.average_score)),
                ("Median Score", format!("{:.3}", response.median_score)),
                ("Highest Score", format!("{:.3}", response.highest_score)),
                ("Lowest Score", format!("{:.3}", response.lowest_score)),
            ]);
        }
        OutputMode::Plain => {
            println!("Total participants: {}", response.total_participants);
            println!("Average score: {:.3}", response.average_score);
        }
    }

    Ok(())
}
