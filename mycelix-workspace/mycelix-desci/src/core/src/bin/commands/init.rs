// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use mycelix_desci_core::{Config, Result};
use std::path::PathBuf;
use tracing::info;

pub async fn execute(output: PathBuf) -> Result<()> {
    info!("Initializing Mycelix-DeSci configuration in {:?}", output);

    // Create configuration directory
    std::fs::create_dir_all(&output).map_err(|e| {
        mycelix_desci_core::Error::Generic(format!("Failed to create directory: {}", e))
    })?;

    // Create data directory
    let data_dir = output.join("data");
    std::fs::create_dir_all(&data_dir).map_err(|e| {
        mycelix_desci_core::Error::Generic(format!("Failed to create data directory: {}", e))
    })?;

    // Write default configuration
    let config = Config::default();
    let config_file = output.join("config.toml");
    config.save(&config_file)?;

    println!("✓ Configuration initialized successfully!");
    println!("  Config file: {}", config_file.display());
    println!("  Data directory: {}", data_dir.display());
    println!("\nNext steps:");
    println!("  1. Edit {} to customize settings", config_file.display());
    println!("  2. Run 'mycelix-desci upload' to add datasets");
    println!("  3. Run 'mycelix-desci query' to search claims");

    Ok(())
}
