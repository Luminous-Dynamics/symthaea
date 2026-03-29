// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Query commands

use anyhow::Result;
use clap::Subcommand;
use serde::Serialize;

use crate::client::ApiClient;
use crate::output::{self, OutputMode};

use super::QueryResponse;

#[derive(Subcommand)]
pub enum QueryCommand {
    /// Search claims with filters
    Search {
        /// Filter by category
        #[arg(long)]
        category: Option<String>,

        /// Filter by minimum tier (E0, E1, E2, E3, E4)
        #[arg(long)]
        tier: Option<String>,

        /// Filter by keywords (comma-separated)
        #[arg(long)]
        keywords: Option<String>,

        /// Page number (default: 1)
        #[arg(long, default_value = "1")]
        page: usize,

        /// Page size (default: 20)
        #[arg(long, default_value = "20")]
        page_size: usize,
    },

    /// List all categories
    Categories,

    /// Get query statistics
    Stats,
}

#[derive(Debug, Serialize)]
struct QueryRequest {
    category: Option<String>,
    tier: Option<String>,
    keywords: Option<Vec<String>>,
    page: Option<usize>,
    page_size: Option<usize>,
}

pub async fn execute(
    client: ApiClient,
    command: QueryCommand,
    output_mode: OutputMode,
) -> Result<()> {
    match command {
        QueryCommand::Search { category, tier, keywords, page, page_size } => {
            search_claims(client, category, tier, keywords, page, page_size, output_mode).await
        }
        QueryCommand::Categories => list_categories(client, output_mode).await,
        QueryCommand::Stats => get_stats(client, output_mode).await,
    }
}

async fn search_claims(
    client: ApiClient,
    category: Option<String>,
    tier: Option<String>,
    keywords: Option<String>,
    page: usize,
    page_size: usize,
    output_mode: OutputMode,
) -> Result<()> {
    output::info("Searching claims...");

    let keywords_vec = keywords.map(|k| {
        k.split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<_>>()
    });

    let request = QueryRequest {
        category,
        tier,
        keywords: keywords_vec,
        page: Some(page),
        page_size: Some(page_size),
    };

    let response: QueryResponse = client.post("/api/v1/query", &request).await?;

    output::success(&format!(
        "Found {} claims (page {}/{})",
        response.total_count, response.page, response.total_pages
    ));

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table => {
            let mut table = output::create_table(&["ID", "Tier", "Category", "Description"]);
            for claim in &response.results {
                table.add_row(vec![
                    claim.id.to_string(),
                    claim.tier.clone(),
                    claim.content.category.clone(),
                    claim.content.description.chars().take(50).collect::<String>() + "...",
                ]);
            }
            println!("{}", table);
        }
        OutputMode::Plain => {
            for claim in &response.results {
                println!("{}: {} - {}", claim.id, claim.tier, claim.content.category);
            }
        }
    }

    Ok(())
}

async fn list_categories(client: ApiClient, output_mode: OutputMode) -> Result<()> {
    output::info("Retrieving categories...");

    #[derive(serde::Deserialize, serde::Serialize)]
    struct CategoriesResponse {
        categories: Vec<String>,
    }

    let response: CategoriesResponse = client.get("/api/v1/query/categories").await?;

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table | OutputMode::Plain => {
            for category in &response.categories {
                println!("  • {}", category);
            }
        }
    }

    Ok(())
}

async fn get_stats(client: ApiClient, output_mode: OutputMode) -> Result<()> {
    output::info("Retrieving statistics...");

    #[derive(serde::Deserialize, serde::Serialize)]
    struct QueryStatsResponse {
        total_claims: usize,
        claims_by_tier: std::collections::HashMap<String, usize>,
        total_categories: usize,
        total_keywords: usize,
    }

    let response: QueryStatsResponse = client.get("/api/v1/query/stats").await?;

    match output_mode {
        OutputMode::Json => output::print_json(&response)?,
        OutputMode::Table => {
            output::print_key_value_table(&[
                ("Total Claims", response.total_claims.to_string()),
                ("Total Categories", response.total_categories.to_string()),
                ("Total Keywords", response.total_keywords.to_string()),
            ]);
        }
        OutputMode::Plain => {
            println!("Total claims: {}", response.total_claims);
            println!("Total categories: {}", response.total_categories);
            println!("Total keywords: {}", response.total_keywords);
        }
    }

    Ok(())
}
