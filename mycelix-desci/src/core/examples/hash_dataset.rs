// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Example: Hashing Datasets
//!
//! This example demonstrates:
//! 1. Hashing files with BLAKE3 and SHA-256
//! 2. Verifying file integrity
//! 3. Building Merkle trees for large datasets
//!
//! Run with: cargo run --example hash_dataset

use mycelix_desci_core::hash::{self, HashAlgorithm};
use std::io::Write;
use tempfile::NamedTempFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Dataset Hashing Examples ===\n");

    // 1. Create a temporary file to hash
    let mut temp_file = NamedTempFile::new()?;
    temp_file.write_all(b"Example dataset content for Mycelix-DeSci\nLine 2\nLine 3")?;

    println!("Created temporary dataset file:");
    println!("  Path: {:?}", temp_file.path());
    println!();

    // 2. Hash with BLAKE3 (default, fast)
    let blake3_hash = hash::hash_file(temp_file.path())?;

    println!("BLAKE3 Hash:");
    println!("  Hex: {}", blake3_hash.hex());
    println!("  Formatted: {}", blake3_hash.to_string());
    println!();

    // 3. Hash with SHA-256 (for compatibility)
    let sha256_hash = hash::hash_file_with_algorithm(temp_file.path(), HashAlgorithm::Sha256)?;

    println!("SHA-256 Hash:");
    println!("  Hex: {}", sha256_hash.hex());
    println!("  Formatted: {}", sha256_hash.to_string());
    println!();

    // 4. Verify file integrity
    println!("Verifying file integrity...");
    let is_valid = hash::verify_file(temp_file.path(), &blake3_hash)?;
    println!("  ✓ Verification: {}", if is_valid { "PASSED" } else { "FAILED" });
    println!();

    // 5. Hash bytes directly
    let data = b"Quick hash of small data";
    let bytes_hash = hash::hash_bytes(data);

    println!("Direct bytes hashing:");
    println!("  Data: {:?}", std::str::from_utf8(data)?);
    println!("  Hash: {}", bytes_hash.to_string());
    println!();

    // 6. Merkle tree for large datasets
    println!("Building Merkle tree for dataset chunks...");

    let chunks = vec![
        b"Chunk 1: Dataset header".as_slice(),
        b"Chunk 2: Data rows 1-1000".as_slice(),
        b"Chunk 3: Data rows 1001-2000".as_slice(),
        b"Chunk 4: Dataset footer".as_slice(),
    ];

    let chunk_hashes: Vec<_> = chunks
        .iter()
        .map(|chunk| hash::hash_bytes(chunk))
        .collect();

    println!("  Individual chunk hashes:");
    for (i, h) in chunk_hashes.iter().enumerate() {
        println!("    Chunk {}: {}", i + 1, h.hex()[..16].to_string() + "...");
    }

    let merkle_tree = hash::build_merkle_tree(chunk_hashes)?;
    let root_hash = merkle_tree.root_hash();

    println!("  Merkle root: {}", root_hash.to_string());
    println!("  ✓ Tree built successfully!");
    println!();

    // 7. Parse hash from string
    let hash_string = blake3_hash.to_string();
    let parsed_hash = hash::Hash::from_string(&hash_string)?;

    println!("Hash string parsing:");
    println!("  Original: {}", hash_string);
    println!("  Parsed algorithm: {:?}", parsed_hash.algorithm);
    println!("  Hashes match: {}", parsed_hash.bytes == blake3_hash.bytes);
    println!();

    println!("✓ All examples completed successfully!");

    Ok(())
}
