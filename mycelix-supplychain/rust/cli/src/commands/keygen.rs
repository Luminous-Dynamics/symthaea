// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use colored::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize)]
struct KeypairFile {
    public_key: String,
    private_key: String,
    seed: String,
}

pub async fn run(output: &Path, seed: Option<&str>) -> Result<()> {
    println!("{}", "Generating Ed25519 keypair...".cyan().bold());

    let keypair = if let Some(seed_hex) = seed {
        // Decode hex seed
        let seed_bytes = hex::decode(seed_hex).context("Invalid hex seed")?;
        if seed_bytes.len() != 32 {
            anyhow::bail!("Seed must be 32 bytes (64 hex characters)");
        }
        let mut seed_arr = [0u8; 32];
        seed_arr.copy_from_slice(&seed_bytes);
        crypto::KeyPair::from_seed(&seed_arr)
    } else {
        crypto::KeyPair::generate()
    };

    let public_key_hex = hex::encode(keypair.public_key().to_bytes());
    let seed_hex = hex::encode(keypair.to_bytes());

    let keypair_file = KeypairFile {
        public_key: public_key_hex.clone(),
        private_key: seed_hex.clone(), // For Ed25519, private key = seed
        seed: seed_hex,
    };

    // Write to file
    let json = serde_json::to_string_pretty(&keypair_file)?;
    std::fs::write(output, json).context("Failed to write keypair file")?;

    println!();
    println!("{}", "✓ Keypair generated successfully!".green().bold());
    println!();
    println!("  {} {}", "Public Key:".cyan(), public_key_hex);
    println!("  {} {}", "Output file:".cyan(), output.display());
    println!();
    println!("{}", "⚠ Warning: Keep the private key secure!".yellow());
    println!();

    Ok(())
}
