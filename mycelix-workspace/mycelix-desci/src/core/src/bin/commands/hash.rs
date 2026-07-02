// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use mycelix_desci_core::{hash, Result};
use std::path::PathBuf;
use tracing::info;

pub async fn execute(path: PathBuf, algorithm: String) -> Result<()> {
    info!("Computing hash of {:?} using {}", path, algorithm);

    let hash_algorithm = match algorithm.to_lowercase().as_str() {
        "blake3" => hash::HashAlgorithm::Blake3,
        "sha256" => hash::HashAlgorithm::Sha256,
        _ => {
            return Err(mycelix_desci_core::Error::Generic(format!(
                "Unknown hash algorithm: {}. Supported: blake3, sha256",
                algorithm
            )))
        }
    };

    if !path.exists() {
        return Err(mycelix_desci_core::Error::Generic(format!(
            "Path does not exist: {}",
            path.display()
        )));
    }

    let file_hash = hash::hash_file_with_algorithm(&path, hash_algorithm)?;

    println!("File: {}", path.display());
    println!("Algorithm: {}", hash_algorithm.as_str());
    println!("Hash: {}", file_hash.hex());
    println!("\nFormatted: {}", file_hash.to_string());

    Ok(())
}
