// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GPT-2 Cross-Model Validation
//!
//! Tests whether the late-layer phenomenal effect replicates in GPT-2.
//! If the effect appears across architectures, it strengthens the claim
//! that this is a general property of transformers.
//!
//! GPT-2 architecture:
//! - 12 layers (gpt2-small) or 24 layers (gpt2-medium)
//! - 768 hidden dimensions (small) or 1024 (medium)
//! - Decoder-only (causal attention)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example gpt2_layer_topology --features neural-bridge --release
//! ```

use anyhow::{Context, Result};
use std::time::Instant;

#[cfg(feature = "neural-bridge")]
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "neural-bridge")]
use candle_nn::VarBuilder;

#[cfg(feature = "neural-bridge")]
use candle_transformers::models::gpt2::{Config as Gpt2Config, GPT2};

#[cfg(feature = "neural-bridge")]
use tokenizers::Tokenizer;

#[cfg(feature = "neural-bridge")]
use hf_hub::{Repo, RepoType, api::sync::Api};

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::binary_hv::BinaryHV;

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::consciousness_topology::{ConsciousnessTopology, TopologyConfig};

fn main() -> Result<()> {
    #[cfg(not(feature = "neural-bridge"))]
    {
        println!("This example requires the 'neural-bridge' feature.");
        return Ok(());
    }

    #[cfg(feature = "neural-bridge")]
    run_experiment()
}

#[cfg(feature = "neural-bridge")]
fn run_experiment() -> Result<()> {
    println!("\n");
    println!("================================================================");
    println!("   GPT-2 CROSS-MODEL VALIDATION");
    println!("   Does the late-layer phenomenal effect replicate?");
    println!("================================================================\n");

    // Load corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    // Use subset for faster testing
    let phenomenal: Vec<_> = phenomenal.into_iter().take(50).collect();
    let functional: Vec<_> = functional.into_iter().take(50).collect();

    println!(
        "Using {} phenomenal, {} functional concepts\n",
        phenomenal.len(),
        functional.len()
    );

    // Load GPT-2
    println!("Loading GPT-2 medium (24 layers)...");
    let load_start = Instant::now();

    let device = Device::Cpu;
    let api = Api::new()?;
    let repo = api.repo(Repo::new(
        "openai-community/gpt2-medium".to_string(),
        RepoType::Model,
    ));

    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("Failed to download tokenizer")?;
    let config_path = repo
        .get("config.json")
        .context("Failed to download config")?;
    let weights_path = repo
        .get("model.safetensors")
        .or_else(|_| repo.get("pytorch_model.bin"))
        .context("Failed to download weights")?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

    let config_str = std::fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_str)?;

    let gpt2_config = Gpt2Config {
        vocab_size: config["vocab_size"].as_u64().unwrap_or(50257) as usize,
        n_embd: config["n_embd"].as_u64().unwrap_or(1024) as usize,
        n_head: config["n_head"].as_u64().unwrap_or(16) as usize,
        n_layer: config["n_layer"].as_u64().unwrap_or(24) as usize,
        n_positions: config["n_positions"].as_u64().unwrap_or(1024) as usize,
        ..Default::default()
    };

    println!(
        "  Config: {} layers, {} hidden dim",
        gpt2_config.n_layer, gpt2_config.n_embd
    );

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)? };
    let model = GPT2::load(vb, &gpt2_config)?;

    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    // Since GPT-2 doesn't have built-in layer extraction, we'll use the final hidden state
    // and compare it to early layers by modifying forward pass

    // For now, let's just use the final hidden state and see if there's a difference
    println!("================================================================");
    println!("   EXTRACTING FINAL HIDDEN STATES");
    println!("================================================================\n");

    let topology_config = TopologyConfig {
        min_persistence: 0.1,
        max_scale: 1.0,
        num_scales: 10,
        detect_cycles: true,
        detect_voids: false,
    };

    // Extract phenomenal
    println!("Processing phenomenal concepts...");
    let mut phen_unity = Vec::new();
    for (i, concept) in phenomenal.iter().enumerate() {
        if i % 10 == 0 {
            print!("  {}/{}\r", i, phenomenal.len());
        }

        let encoding = tokenizer
            .encode(concept.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let input = Tensor::new(&input_ids[..], &device)?.unsqueeze(0)?;

        // Forward through GPT-2
        let hidden = model.forward(&input)?;

        // Mean pool over sequence
        let pooled = hidden.mean(1)?;
        let activation: Vec<f32> = pooled.squeeze(0)?.to_vec1()?;

        let hv = activation_to_hv16(&activation);
        phen_unity.push(compute_unity(&hv, &topology_config));
    }
    println!("  Done                    ");

    // Extract functional
    println!("Processing functional concepts...");
    let mut func_unity = Vec::new();
    for (i, concept) in functional.iter().enumerate() {
        if i % 10 == 0 {
            print!("  {}/{}\r", i, functional.len());
        }

        let encoding = tokenizer
            .encode(concept.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let input = Tensor::new(&input_ids[..], &device)?.unsqueeze(0)?;

        let hidden = model.forward(&input)?;
        let pooled = hidden.mean(1)?;
        let activation: Vec<f32> = pooled.squeeze(0)?.to_vec1()?;

        let hv = activation_to_hv16(&activation);
        func_unity.push(compute_unity(&hv, &topology_config));
    }
    println!("  Done\n");

    // Statistics
    println!("================================================================");
    println!("   RESULTS: GPT-2 FINAL LAYER");
    println!("================================================================\n");

    let phen_mean = mean(&phen_unity);
    let func_mean = mean(&func_unity);
    let diff = phen_mean - func_mean;

    let phen_std = std_dev(&phen_unity);
    let func_std = std_dev(&func_unity);
    let pooled_std = ((phen_std.powi(2) + func_std.powi(2)) / 2.0).sqrt();
    let cohens_d = if pooled_std > 0.0 {
        diff / pooled_std
    } else {
        0.0
    };

    let p_value = permutation_test(&phen_unity, &func_unity, 10000);

    println!("Phenomenal Unity: {:.4} (std: {:.4})", phen_mean, phen_std);
    println!("Functional Unity: {:.4} (std: {:.4})", func_mean, func_std);
    println!("Difference: {:+.4}", diff);
    println!("Cohen's d: {:+.4}", cohens_d);
    println!(
        "p-value: {:.4} {}",
        p_value,
        if p_value < 0.05 { "*" } else { "" }
    );

    // Interpretation
    println!("\n================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    if diff > 0.0 && p_value < 0.05 {
        println!("✓ EFFECT REPLICATES IN GPT-2");
        println!("  Phenomenal concepts show higher topological unity");
        println!("  This suggests the effect is architecture-general");
    } else if diff < 0.0 && p_value < 0.05 {
        println!("✗ OPPOSITE EFFECT IN GPT-2");
        println!("  Functional concepts show higher unity");
        println!("  Architecture differences may matter");
    } else {
        println!("○ NO SIGNIFICANT DIFFERENCE IN GPT-2");
        println!("  Effect may be specific to encoder models (BGE-M3)");
    }

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn load_corpus(path: &str) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    let mut concepts = Vec::new();
    if let Some(items) = json["concepts"].as_array() {
        for item in items {
            if let Some(text) = item["text"].as_str() {
                concepts.push(text.to_string());
            }
        }
    }
    Ok(concepts)
}

#[cfg(feature = "neural-bridge")]
fn activation_to_hv16(activation: &[f32]) -> BinaryHV {
    use symthaea_core::hdc::HDC_DIMENSION;

    let mut expanded = Vec::with_capacity(HDC_DIMENSION);
    let tiles = HDC_DIMENSION / activation.len();
    let remainder = HDC_DIMENSION % activation.len();

    for tile in 0..tiles {
        for (i, &val) in activation.iter().enumerate() {
            let perturbation = ((tile * activation.len() + i) as f32 * 0.001).sin() * 0.01;
            expanded.push(val + perturbation);
        }
    }

    // Handle remainder if dimensions don't divide evenly
    for i in 0..remainder {
        expanded.push(activation[i]);
    }

    BinaryHV::from_bipolar(&expanded)
}

#[cfg(feature = "neural-bridge")]
fn compute_unity(hv: &BinaryHV, config: &TopologyConfig) -> f64 {
    let mut topology = ConsciousnessTopology::new(config.clone());

    topology.add_state(*hv);
    for shift in 1..5 {
        topology.add_state(hv.permute(shift * 100));
    }

    topology.analyze(0.5).unity_score
}

#[cfg(feature = "neural-bridge")]
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(feature = "neural-bridge")]
fn std_dev(values: &[f64]) -> f64 {
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

#[cfg(feature = "neural-bridge")]
fn permutation_test(a: &[f64], b: &[f64], n_perm: usize) -> f64 {
    let observed = mean(a) - mean(b);
    let mut combined: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    let n_a = a.len();

    let mut more_extreme = 0;
    let mut rng: u64 = 42;

    for _ in 0..n_perm {
        for i in (1..combined.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng as usize) % (i + 1);
            combined.swap(i, j);
        }

        let perm_a = combined[..n_a].iter().sum::<f64>() / n_a as f64;
        let perm_b = combined[n_a..].iter().sum::<f64>() / (combined.len() - n_a) as f64;

        if (perm_a - perm_b).abs() >= observed.abs() {
            more_extreme += 1;
        }
    }

    more_extreme as f64 / n_perm as f64
}
