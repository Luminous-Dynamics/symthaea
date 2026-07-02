// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use claim_model::{CredentialSubject, EventType, Facility, SupplyEventVC};
use chrono::Utc;
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;

pub async fn run(base_url: &str, file: &Path, has_header: bool, batch_size: usize) -> Result<()> {
    println!("{} CSV file: {}", "Reading".cyan(), file.display());

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(has_header)
        .from_path(file)
        .context("Failed to read CSV file")?;

    let mut events: Vec<SupplyEventVC> = Vec::new();

    // Parse CSV records
    // Expected format: event_type,issuer,product_id,batch_id,quantity,unit,facility_id,facility_name
    for result in rdr.records() {
        let record = result.context("Failed to parse CSV record")?;

        if record.len() < 8 {
            eprintln!("{} Skipping invalid record (not enough fields)", "⚠".yellow());
            continue;
        }

        let event_type = record[0].parse::<EventType>().ok();
        if event_type.is_none() {
            eprintln!("{} Skipping record with invalid event_type: {}", "⚠".yellow(), &record[0]);
            continue;
        }

        let vc = SupplyEventVC {
            context: vec!["https://www.w3.org/2018/credentials/v1".to_string()],
            vc_type: vec!["VerifiableCredential".to_string()],
            issuer: record[1].to_string(),
            issuance_date: Utc::now(),
            expiration_date: None,
            credential_subject: CredentialSubject {
                event_type: event_type.unwrap(),
                product_id: record[2].to_string(),
                batch_id: record[3].to_string(),
                prev_batch_ids: None,
                quantity: record[4].parse().unwrap_or(0.0),
                unit: record[5].to_string(),
                facility: Facility {
                    id: record[6].to_string(),
                    name: record[7].to_string(),
                    location: None,
                },
                timestamp: Utc::now(),
                shipment: None,
                certification: None,
                metadata: None,
            },
            proof: None,
        };

        events.push(vc);
    }

    if events.is_empty() {
        println!("{}", "No valid events found in CSV".yellow());
        return Ok(());
    }

    println!("{} {} events from CSV", "Parsed".green(), events.len());
    println!("{} events to API...", "Uploading".cyan());

    let client = reqwest::Client::new();
    let url = format!("{}/v1/events", base_url);

    let pb = ProgressBar::new(events.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let mut success = 0;
    let mut failed = 0;

    for chunk in events.chunks(batch_size) {
        let mut handles = Vec::new();

        for vc in chunk {
            let client = client.clone();
            let url = url.clone();
            let vc = vc.clone();

            let handle = tokio::spawn(async move {
                client.post(&url).json(&vc).send().await
            });

            handles.push(handle);
        }

        for handle in handles {
            match handle.await {
                Ok(Ok(response)) if response.status().is_success() => {
                    success += 1;
                }
                _ => {
                    failed += 1;
                }
            }
            pb.inc(1);
        }
    }

    pb.finish_with_message("Done");

    println!();
    println!("{}", "✓ Import completed!".green().bold());
    println!();
    println!("  {} {}", "Successful:".green(), success);
    if failed > 0 {
        println!("  {} {}", "Failed:".red(), failed);
    }
    println!();

    Ok(())
}
