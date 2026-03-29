// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Simple API Usage Example
//!
//! This example demonstrates basic API operations for getting started
//! with the Mycelix-DeSci platform quickly.
//!
//! ## What this example shows:
//! - Creating a claim
//! - Retrieving a claim
//! - Searching claims
//! - Checking system health
//!
//! ## Usage
//! ```bash
//! cargo run --example simple_api_usage
//! ```

use serde_json::json;

const API_BASE: &str = "http://localhost:8080/api/v1";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Mycelix-DeSci: Simple API Usage Example\n");

    let client = reqwest::Client::new();

    // 1. Check system health
    println!("1️⃣  Checking system health...");
    let health: serde_json::Value = client
        .get(format!("{}/system/health", API_BASE))
        .send()
        .await?
        .json()
        .await?;

    println!("   Status: {}", health["status"]);
    println!("   Version: {}\n", health["version"]);

    // 2. Create a simple claim
    println!("2️⃣  Creating a claim...");
    let claim_request = json!({
        "tier": "E0",
        "content": {
            "dataset_hash": "blake3:abc123...",
            "description": "Example research claim",
            "category": "test",
            "keywords": ["example", "demo"]
        },
        "creator": "user@example.com"
    });

    let claim: serde_json::Value = client
        .post(format!("{}/claims", API_BASE))
        .json(&claim_request)
        .send()
        .await?
        .json()
        .await?;

    let claim_id = claim["id"].as_str().unwrap();
    println!("   Created claim: {}", claim_id);
    println!("   Tier: {}\n", claim["tier"]);

    // 3. Retrieve the claim
    println!("3️⃣  Retrieving claim...");
    let retrieved: serde_json::Value = client
        .get(format!("{}/claims/{}", API_BASE, claim_id))
        .send()
        .await?
        .json()
        .await?;

    println!("   Description: {}", retrieved["content"]["description"]);
    println!("   Creator: {}\n", retrieved["creator"]);

    // 4. Search claims
    println!("4️⃣  Searching claims...");
    let search_request = json!({
        "category": "test",
        "page": 1,
        "page_size": 5
    });

    let search_results: serde_json::Value = client
        .post(format!("{}/query", API_BASE))
        .json(&search_request)
        .send()
        .await?
        .json()
        .await?;

    println!("   Found {} claims", search_results["total_count"]);
    println!("   Page: {}/{}\n", search_results["page"], search_results["total_pages"]);

    // 5. Get system metrics
    println!("5️⃣  Getting system metrics...");
    let metrics: serde_json::Value = client
        .get(format!("{}/system/metrics", API_BASE))
        .send()
        .await?
        .json()
        .await?;

    println!("   Total claims: {}", metrics["total_claims"]);
    println!("   Uptime: {} seconds\n", metrics["uptime_seconds"]);

    println!("✅ All API operations completed successfully!");

    Ok(())
}
