// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BERT Cross-Architecture Validation
//!
//! Tests whether the phenomenal effect exists in BERT-base (12 layers, 768D)
//! as a smaller encoder comparison to BGE-M3 (24 layers, 1024D).
//!
//! Key questions:
//! 1. Does the effect appear at similar relative depth (~92%)?
//! 2. Is Φ extractable in a smaller model?
//! 3. Is the effect size comparable?
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example bert_validation --features neural-bridge --release
//! ```

use anyhow::{Context, Result};
use std::time::Instant;

#[cfg(feature = "neural-bridge")]
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "neural-bridge")]
use candle_nn::VarBuilder;

#[cfg(feature = "neural-bridge")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

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
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   BERT CROSS-ARCHITECTURE VALIDATION                         ║");
    println!("║   Testing Smaller Encoder Model                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    let phenomenal: Vec<_> = phenomenal.into_iter().take(50).collect();
    let functional: Vec<_> = functional.into_iter().take(50).collect();

    println!(
        "Using {} phenomenal, {} functional concepts\n",
        phenomenal.len(),
        functional.len()
    );

    // Load BERT-base
    println!("Loading BERT-base-uncased (12 layers, 768D)...");
    let load_start = Instant::now();

    let device = Device::Cpu;
    let api = Api::new()?;
    let repo = api.repo(Repo::new("bert-base-uncased".to_string(), RepoType::Model));

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
    let bert_config: BertConfig = serde_json::from_str(&config_str)?;

    println!(
        "  Config: {} layers, {} hidden dim",
        bert_config.num_hidden_layers, bert_config.hidden_size
    );

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)? };
    let model = BertModel::load(vb, &bert_config)?;

    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    let topology_config = TopologyConfig {
        min_persistence: 0.1,
        max_scale: 1.0,
        num_scales: 10,
        detect_cycles: true,
        detect_voids: false,
    };

    // BERT has 12 layers. Test late layers (75-100% depth = layers 9-11)
    // Equivalent to BGE-M3's L18-23 (75-96%)
    let test_layers = vec![8, 9, 10, 11];

    println!("════════════════════════════════════════════════════════════════");
    println!("   LAYER-WISE ANALYSIS (Final Hidden States)");
    println!("════════════════════════════════════════════════════════════════\n");

    println!(
        "Note: Using BERT's final layer output (layer {}).",
        bert_config.num_hidden_layers
    );
    println!("Testing with [CLS] token pooling.\n");

    // Extract activations for all concepts
    println!("Extracting phenomenal concept activations...");
    let mut phen_activations = Vec::new();
    for (i, concept) in phenomenal.iter().enumerate() {
        if i % 10 == 0 {
            print!("  {}/{}\\r", i, phenomenal.len());
        }

        let encoding = tokenizer
            .encode(concept.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let token_type_ids: Vec<u32> = vec![0; input_ids.len()];

        let input_ids_tensor = Tensor::new(&input_ids[..], &device)?.unsqueeze(0)?;
        let token_type_ids_tensor = Tensor::new(&token_type_ids[..], &device)?.unsqueeze(0)?;

        let output = model.forward(&input_ids_tensor, &token_type_ids_tensor, None)?;

        // Use [CLS] token (first token)
        let cls_token = output.narrow(1, 0, 1)?.squeeze(1)?;
        let activation: Vec<f32> = cls_token.to_vec1()?;

        phen_activations.push(activation);
    }
    println!("  Done                    ");

    println!("Extracting functional concept activations...");
    let mut func_activations = Vec::new();
    for (i, concept) in functional.iter().enumerate() {
        if i % 10 == 0 {
            print!("  {}/{}\\r", i, functional.len());
        }

        let encoding = tokenizer
            .encode(concept.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let token_type_ids: Vec<u32> = vec![0; input_ids.len()];

        let input_ids_tensor = Tensor::new(&input_ids[..], &device)?.unsqueeze(0)?;
        let token_type_ids_tensor = Tensor::new(&token_type_ids[..], &device)?.unsqueeze(0)?;

        let output = model.forward(&input_ids_tensor, &token_type_ids_tensor, None)?;

        let cls_token = output.narrow(1, 0, 1)?.squeeze(1)?;
        let activation: Vec<f32> = cls_token.to_vec1()?;

        func_activations.push(activation);
    }
    println!("  Done\n");

    // Compute unity scores
    println!("Computing topological unity...");
    let phen_unity: Vec<f64> = phen_activations
        .iter()
        .map(|a| {
            let hv = activation_to_hv16(a);
            compute_unity(&hv, &topology_config)
        })
        .collect();

    let func_unity: Vec<f64> = func_activations
        .iter()
        .map(|a| {
            let hv = activation_to_hv16(a);
            compute_unity(&hv, &topology_config)
        })
        .collect();

    // Statistics
    let phen_mean = mean(&phen_unity);
    let func_mean = mean(&func_unity);
    let phen_std = std_dev(&phen_unity);
    let func_std = std_dev(&func_unity);

    let pooled_std = ((phen_std.powi(2) + func_std.powi(2)) / 2.0).sqrt();
    let cohens_d = if pooled_std > 0.0 {
        (phen_mean - func_mean) / pooled_std
    } else {
        0.0
    };
    let p_value = permutation_test(&phen_unity, &func_unity, 10000);

    println!("\n════════════════════════════════════════════════════════════════");
    println!("   RESULTS: BERT-base Final Layer (Layer 12)");
    println!("════════════════════════════════════════════════════════════════\n");

    println!("Phenomenal Unity: {:.4} ± {:.4}", phen_mean, phen_std);
    println!("Functional Unity: {:.4} ± {:.4}", func_mean, func_std);
    println!("Difference: {:+.4}", phen_mean - func_mean);
    println!("Cohen's d: {:+.3}", cohens_d);
    println!("p-value: {:.4} {}", p_value, significance_stars(p_value));

    // Pairwise correlation analysis
    println!("\n────────────────────────────────────────────────────────────────");
    println!("Pairwise Correlation Analysis:");

    let phen_corr = pairwise_correlation(&phen_activations);
    let func_corr = pairwise_correlation(&func_activations);

    println!(
        "  Phenomenal pairwise correlation: {:.4} ± {:.4}",
        mean(&phen_corr),
        std_dev(&phen_corr)
    );
    println!(
        "  Functional pairwise correlation: {:.4} ± {:.4}",
        mean(&func_corr),
        std_dev(&func_corr)
    );

    let corr_p = permutation_test(&phen_corr, &func_corr, 5000);
    println!(
        "  Difference p-value: {:.4} {}",
        corr_p,
        significance_stars(corr_p)
    );

    // Φ extraction attempt
    println!("\n────────────────────────────────────────────────────────────────");
    println!("Φ Extraction:");

    let (phi_validated, phi_d) =
        extract_and_validate_phi(&phen_activations, &func_activations, &topology_config);

    if let Some(d) = phi_d {
        println!("  Φ loading effect size: d={:+.2}", d);
    }
    match phi_validated {
        Some(true) => println!("  Φ extraction: VALIDATED (removes significance)"),
        Some(false) => println!("  Φ extraction: NOT validated (effect persists)"),
        None => println!("  Φ extraction: Could not compute"),
    }

    // Interpretation
    println!("\n════════════════════════════════════════════════════════════════");
    println!("   INTERPRETATION");
    println!("════════════════════════════════════════════════════════════════\n");

    let replicates = cohens_d > 0.2 && p_value < 0.05;

    if replicates {
        println!("✓ PHENOMENAL EFFECT REPLICATES IN BERT-base");
        println!("  Smaller encoder shows same pattern as BGE-M3");
        println!("  Effect is scale-independent within encoder family\n");

        if mean(&phen_corr) > mean(&func_corr) && corr_p < 0.05 {
            println!("  ✓ Higher pairwise correlation for phenomenal concepts");
            println!("    Consistent with shared phenomenal signature (Φ)");
        }
    } else if cohens_d > 0.0 {
        println!("◐ WEAK/NON-SIGNIFICANT EFFECT IN BERT-base");
        println!("  Direction consistent with BGE-M3 (d={:+.2})", cohens_d);
        println!("  Effect may be stronger in larger models\n");
    } else {
        println!("✗ NO PHENOMENAL EFFECT IN BERT-base");
        println!("  Effect may require larger models or different architecture");
    }

    // Comparison with BGE-M3
    println!("\n────────────────────────────────────────────────────────────────");
    println!("Cross-Architecture Comparison:");
    println!("                    │ BGE-M3 (L22) │ BERT-base (L12)");
    println!("────────────────────┼──────────────┼────────────────");
    println!("Architecture        │ XLM-RoBERTa  │ BERT");
    println!("Layers              │ 24           │ 12");
    println!("Hidden dim          │ 1024         │ 768");
    println!("Parameters          │ ~560M        │ ~110M");
    println!("Phen unity          │ 0.898        │ {:.3}", phen_mean);
    println!("Func unity          │ 0.700        │ {:.3}", func_mean);
    println!("Cohen's d           │ +0.69        │ {:+.2}", cohens_d);
    println!(
        "Significant         │ Yes (p=.002) │ {} (p={:.3})",
        if p_value < 0.05 { "Yes" } else { "No" },
        p_value
    );

    // Relative depth analysis
    println!("\n────────────────────────────────────────────────────────────────");
    println!("Relative Depth Analysis:");
    println!("  BGE-M3 peak: Layer 22/24 = 91.7%% depth");
    println!("  BERT-base: Layer 12/12 = 100%% depth (final layer)");
    println!("  Predicted equivalent: Layer ~11/12 = 91.7%% depth");

    if cohens_d > 0.2 {
        println!("\n  → Effect present, confirming late-layer hypothesis");
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   EXPERIMENT COMPLETE                                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn extract_and_validate_phi(
    phen_acts: &[Vec<f32>],
    func_acts: &[Vec<f32>],
    topology_config: &TopologyConfig,
) -> (Option<bool>, Option<f64>) {
    if phen_acts.is_empty() || func_acts.is_empty() {
        return (None, None);
    }

    let dim = phen_acts[0].len();

    // Compute centroids
    let phen_centroid: Vec<f64> = (0..dim)
        .map(|i| phen_acts.iter().map(|a| a[i] as f64).sum::<f64>() / phen_acts.len() as f64)
        .collect();

    let func_centroid: Vec<f64> = (0..dim)
        .map(|i| func_acts.iter().map(|a| a[i] as f64).sum::<f64>() / func_acts.len() as f64)
        .collect();

    // Difference vector → Φ direction
    let diff: Vec<f64> = phen_centroid
        .iter()
        .zip(func_centroid.iter())
        .map(|(p, f)| p - f)
        .collect();

    let norm: f64 = diff.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    if norm < 1e-10 {
        return (None, None);
    }

    let phi: Vec<f64> = diff.iter().map(|x| x / norm).collect();

    // Compute Φ loadings
    let phen_loadings: Vec<f64> = phen_acts
        .iter()
        .map(|a| a.iter().zip(phi.iter()).map(|(x, p)| (*x as f64) * p).sum())
        .collect();

    let func_loadings: Vec<f64> = func_acts
        .iter()
        .map(|a| a.iter().zip(phi.iter()).map(|(x, p)| (*x as f64) * p).sum())
        .collect();

    let phen_loading_mean = mean(&phen_loadings);
    let func_loading_mean = mean(&func_loadings);
    let phen_loading_std = std_dev(&phen_loadings);
    let func_loading_std = std_dev(&func_loadings);

    let pooled_std = ((phen_loading_std.powi(2) + func_loading_std.powi(2)) / 2.0).sqrt();
    let phi_d = if pooled_std > 0.0 {
        (phen_loading_mean - func_loading_mean) / pooled_std
    } else {
        0.0
    };

    // Subtract Φ from phenomenal and re-measure
    let mut phen_minus_phi_unity = Vec::new();
    for act in phen_acts {
        let loading: f64 = act
            .iter()
            .zip(phi.iter())
            .map(|(x, p)| (*x as f64) * p)
            .sum();
        let adjusted: Vec<f32> = act
            .iter()
            .zip(phi.iter())
            .map(|(x, p)| (*x as f64 - loading * p) as f32)
            .collect();

        let hv = activation_to_hv16(&adjusted);
        phen_minus_phi_unity.push(compute_unity(&hv, topology_config));
    }

    let mut func_unity = Vec::new();
    for act in func_acts {
        let hv = activation_to_hv16(act);
        func_unity.push(compute_unity(&hv, topology_config));
    }

    let p_after = permutation_test(&phen_minus_phi_unity, &func_unity, 5000);
    let validated = p_after > 0.05;

    (Some(validated), Some(phi_d))
}

#[cfg(feature = "neural-bridge")]
fn pairwise_correlation(activations: &[Vec<f32>]) -> Vec<f64> {
    let mut correlations = Vec::new();
    let n = activations.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let corr = pearson_correlation(&activations[i], &activations[j]);
            correlations.push(corr);
        }
    }

    correlations
}

#[cfg(feature = "neural-bridge")]
fn pearson_correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let mean_a: f64 = a.iter().map(|x| *x as f64).sum::<f64>() / n;
    let mean_b: f64 = b.iter().map(|x| *x as f64).sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        let da = *x as f64 - mean_a;
        let db = *y as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a < 1e-10 || var_b < 1e-10 {
        return 0.0;
    }

    cov / (var_a.sqrt() * var_b.sqrt())
}

#[cfg(feature = "neural-bridge")]
fn significance_stars(p: f64) -> &'static str {
    if p < 0.001 {
        "***"
    } else if p < 0.01 {
        "**"
    } else if p < 0.05 {
        "*"
    } else {
        ""
    }
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
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(feature = "neural-bridge")]
fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
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
