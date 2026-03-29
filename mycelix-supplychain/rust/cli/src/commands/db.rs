// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use colored::*;
use sqlx::SqlitePool;

pub async fn migrate(database_url: &str) -> Result<()> {
    println!("{} database migrations...", "Running".cyan());
    println!("  {} {}", "Database:".cyan(), database_url.dimmed());
    println!();

    let pool = SqlitePool::connect(database_url)
        .await
        .context("Failed to connect to database")?;

    // Run the initial schema migration manually
    // In a production system, you would use sqlx-cli or embed migrations
    let initial_schema = include_str!("../../../service/migrations/20251115000001_initial_schema.sql");

    for statement in initial_schema.split(';') {
        let trimmed = statement.trim();
        if !trimmed.is_empty() {
            sqlx::query(trimmed)
                .execute(&pool)
                .await
                .context(format!("Failed to execute migration statement"))?;
        }
    }

    println!("{}", "✓ Migrations completed successfully!".green().bold());
    println!();

    Ok(())
}

pub async fn stats(database_url: &str) -> Result<()> {
    println!("{} database statistics...", "Fetching".cyan());

    let pool = SqlitePool::connect(database_url)
        .await
        .context("Failed to connect to database")?;

    let total_claims: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM claims")
        .fetch_one(&pool)
        .await
        .context("Failed to query claims count")?;

    let total_lineage_links: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM lineage")
        .fetch_one(&pool)
        .await
        .context("Failed to query lineage count")?;

    let event_type_stats: Vec<(String, i64)> =
        sqlx::query_as("SELECT event_type, COUNT(*) as count FROM claims GROUP BY event_type")
            .fetch_all(&pool)
            .await
            .context("Failed to query event types")?;

    println!();
    println!("{}", "✓ Database statistics".green().bold());
    println!();
    println!("  {} {}", "Total claims:".cyan(), total_claims.to_string().bright_white());
    println!("  {} {}", "Lineage links:".cyan(), total_lineage_links.to_string().bright_white());
    println!();
    println!("{}", "  Events by type:".cyan().bold());
    for (event_type, count) in event_type_stats {
        println!("    {} {}", format!("{:12}", event_type).cyan(), count.to_string().bright_white());
    }
    println!();

    Ok(())
}

pub async fn health_check(database_url: &str) -> Result<()> {
    println!("{} database connection...", "Checking".cyan());

    let start = std::time::Instant::now();
    let pool = SqlitePool::connect(database_url)
        .await
        .context("Failed to connect to database")?;

    sqlx::query("SELECT 1")
        .fetch_one(&pool)
        .await
        .context("Failed to execute health check query")?;

    let duration = start.elapsed();

    println!();
    println!("{}", "✓ Database is healthy!".green().bold());
    println!();
    println!("  {} {}", "Database:".cyan(), database_url.dimmed());
    println!("  {} {:?}", "Response time:".cyan(), duration);
    println!();

    Ok(())
}
