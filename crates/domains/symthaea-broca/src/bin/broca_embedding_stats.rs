// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-embedding-stats: Extract per-dimension embedding statistics from a Mamba model.
//!
//! Computes per-dimension mean and variance of the model's embedding table [vocab_size, d_model]
//! and saves them in a compact binary format for manifold moment matching adapter initialization.
//!
//! Usage:
//!   broca-embedding-stats [--model MODEL_ID] --output stats.bin

use std::process;

use symthaea_broca::mamba::{MambaWrapper, best_device};
use symthaea_broca::temporal_projection::EmbeddingStats;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let (model_id, output_path) = match parse_args(&args) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {e}");
            eprintln!("Usage: broca-embedding-stats [--model MODEL_ID] --output stats.bin");
            process::exit(1);
        }
    };

    tracing::info!(model = %model_id, "Loading Mamba model");
    let device = best_device();
    let wrapper = match MambaWrapper::load(&model_id, device) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            process::exit(1);
        }
    };

    tracing::info!("Extracting embedding table");
    let table = wrapper.embedding_table();
    let flat: Vec<f32> = match table.flatten_all().and_then(|t| t.to_vec1()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to flatten embedding table: {e}");
            process::exit(1);
        }
    };

    let d_model = wrapper.d_model();
    let stats = EmbeddingStats::compute(&flat, d_model);

    // Print summary
    let mean_min = stats.mean.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean_max = stats.mean.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let var_min = stats.variance.iter().cloned().fold(f32::INFINITY, f32::min);
    let var_max = stats
        .variance
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    println!("Embedding stats:");
    println!("  dim:       {}", stats.dim);
    println!("  count:     {}", stats.count);
    println!("  mean:      [{mean_min:.6}, {mean_max:.6}]");
    println!("  variance:  [{var_min:.6}, {var_max:.6}]");
    println!("  file size: {} bytes", 16 + stats.dim * 8);

    tracing::info!(path = %output_path, "Saving embedding stats");
    if let Err(e) = stats.save(&output_path) {
        eprintln!("Failed to save: {e}");
        process::exit(1);
    }
    println!("Saved to: {output_path}");
}

fn parse_args(args: &[String]) -> Result<(String, String), String> {
    let mut model_id = "state-spaces/mamba-130m".to_string();
    let mut output_path = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_id = args.get(i).cloned().ok_or("--model requires a model ID")?;
            }
            "--output" | "-o" => {
                i += 1;
                output_path = args.get(i).cloned().ok_or("--output requires a path")?;
            }
            "--help" | "-h" => {
                eprintln!("Usage: broca-embedding-stats [--model MODEL_ID] --output stats.bin");
                std::process::exit(0);
            }
            arg => return Err(format!("Unknown argument: {arg}")),
        }
        i += 1;
    }

    if output_path.is_empty() {
        return Err("--output is required".to_string());
    }

    Ok((model_id, output_path))
}
