// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claims commands

use anyhow::Result;
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::client::ApiClient;
use crate::output::{self, OutputMode};

use super::ClaimResponse;

#[derive(Subcommand)]
pub enum ClaimsCommand {
    /// Create a new claim from JSON file
    Create {
        /// Path to JSON file with claim data
        file: String,
    },

    /// Get claim by ID
    Get {
        /// Claim ID (UUID)
        id: String,
    },

    /// Add verification to a claim
    Verify {
        /// Claim ID
        id: String,

        /// Verifier identifier
        #[arg(long)]
        verifier: String,

        /// Signature (hex)
        #[arg(long)]
        signature: String,

        /// Optional notes
        #[arg(long)]
        notes: Option<String>,
    },

    /// Add provenance to a claim
    Provenance {
        /// Claim ID
        id: String,

        /// Source identifier
        #[arg(long)]
        source: String,

        /// Source type
        #[arg(long)]
        source_type: String,

        /// Optional URL
        #[arg(long)]
        url: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct CreateClaimRequest {
    tier: String,
    content: ClaimContentRequest,
    creator: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ClaimContentRequest {
    dataset_hash: String,
    description: String,
    category: String,
    keywords: Vec<String>,
    storage_ref: Option<String>,
    reproducibility_score: Option<f64>,
    license: Option<String>,
}

#[derive(Debug, Serialize)]
struct AddVerificationRequest {
    verifier: String,
    signature: Vec<u8>,
    notes: Option<String>,
}

#[derive(Debug, Serialize)]
struct AddProvenanceRequest {
    source: String,
    source_type: String,
    url: Option<String>,
}

pub async fn execute(
    client: ApiClient,
    command: ClaimsCommand,
    output_mode: OutputMode,
) -> Result<()> {
    match command {
        ClaimsCommand::Create { file } => create_claim(client, &file, output_mode).await,
        ClaimsCommand::Get { id } => get_claim(client, &id, output_mode).await,
        ClaimsCommand::Verify { id, verifier, signature, notes } => {
            verify_claim(client, &id, &verifier, &signature, notes, output_mode).await
        }
        ClaimsCommand::Provenance { id, source, source_type, url } => {
            add_provenance(client, &id, &source, &source_type, url, output_mode).await
        }
    }
}

async fn create_claim(
    client: ApiClient,
    file_path: &str,
    output_mode: OutputMode,
) -> Result<()> {
    output::info(&format!("Creating claim from {}", file_path));

    // Read and parse JSON file
    let contents = std::fs::read_to_string(file_path)?;
    let request: CreateClaimRequest = serde_json::from_str(&contents)?;

    // Create claim via API
    let response: ClaimResponse = client.post("/api/v1/claims", &request).await?;

    output::success(&format!("Claim created with ID: {}", response.id));

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table => print_claim_table(&response),
        OutputMode::Plain => {
            println!("ID: {}", response.id);
            println!("Tier: {}", response.tier);
            println!("Category: {}", response.content.category);
        }
    }

    Ok(())
}

async fn get_claim(
    client: ApiClient,
    id: &str,
    output_mode: OutputMode,
) -> Result<()> {
    let uuid = Uuid::parse_str(id)?;
    output::info(&format!("Retrieving claim {}", uuid));

    let response: ClaimResponse = client.get(&format!("/api/v1/claims/{}", uuid)).await?;

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table => print_claim_table(&response),
        OutputMode::Plain => {
            println!("ID: {}", response.id);
            println!("Tier: {}", response.tier);
            println!("Category: {}", response.content.category);
            println!("Description: {}", response.content.description);
        }
    }

    Ok(())
}

async fn verify_claim(
    client: ApiClient,
    id: &str,
    verifier: &str,
    signature_hex: &str,
    notes: Option<String>,
    output_mode: OutputMode,
) -> Result<()> {
    let uuid = Uuid::parse_str(id)?;
    output::info(&format!("Adding verification to claim {}", uuid));

    // Convert hex signature to bytes
    let signature = hex::decode(signature_hex)?;

    let request = AddVerificationRequest {
        verifier: verifier.to_string(),
        signature,
        notes,
    };

    let response: ClaimResponse = client
        .put(&format!("/api/v1/claims/{}/verify", uuid), &request)
        .await?;

    output::success(&format!(
        "Verification added. New tier: {}",
        response.tier
    ));

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table => print_claim_table(&response),
        OutputMode::Plain => {
            println!("Claim ID: {}", response.id);
            println!("New tier: {}", response.tier);
            println!("Verifications: {}", response.verifications_count);
        }
    }

    Ok(())
}

async fn add_provenance(
    client: ApiClient,
    id: &str,
    source: &str,
    source_type: &str,
    url: Option<String>,
    output_mode: OutputMode,
) -> Result<()> {
    let uuid = Uuid::parse_str(id)?;
    output::info(&format!("Adding provenance to claim {}", uuid));

    let request = AddProvenanceRequest {
        source: source.to_string(),
        source_type: source_type.to_string(),
        url,
    };

    let response: ClaimResponse = client
        .put(&format!("/api/v1/claims/{}/provenance", uuid), &request)
        .await?;

    output::success("Provenance added");

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table => print_claim_table(&response),
        OutputMode::Plain => {
            println!("Claim ID: {}", response.id);
            println!("Provenance count: {}", response.provenance_count);
        }
    }

    Ok(())
}

fn print_claim_table(claim: &ClaimResponse) {
    output::print_key_value_table(&[
        ("ID", claim.id.to_string()),
        ("Tier", claim.tier.clone()),
        ("Category", claim.content.category.clone()),
        ("Description", claim.content.description.clone()),
        ("Creator", claim.creator.clone()),
        ("Created At", claim.created_at.clone()),
        ("Verifications", claim.verifications_count.to_string()),
        ("Provenance", claim.provenance_count.to_string()),
        ("Keywords", claim.content.keywords.join(", ")),
        ("Dataset Hash", claim.content.dataset_hash.clone()),
    ]);
}
