// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Robustness Validation
//!
//! Tests whether the Layer 21 phenomenal effect is robust across:
//! 1. Different random subsets of concepts
//! 2. Different data splits
//! 3. Bootstrap resampling
//!
//! This validates that the effect is not an artifact of specific concept selection.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example robustness_validation --features neural-bridge --release
//! ```

use anyhow::Result;
use std::time::Instant;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::{LayerExtractor, PoolingMethod, layer_extractor::LayerExtractorConfig};

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
    println!("   ROBUSTNESS VALIDATION");
    println!("   Is the Layer 21 phenomenal effect stable across samples?");
    println!("================================================================\n");

    // Load full corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    println!(
        "Full corpus: {} phenomenal, {} functional concepts\n",
        phenomenal.len(),
        functional.len()
    );

    // Load model
    println!("Loading BGE-M3...");
    let load_start = Instant::now();
    let config = LayerExtractorConfig {
        pooling: PoolingMethod::Mean,
        ..Default::default()
    };
    let extractor = LayerExtractor::load(config)?;
    println!("  Loaded in {:.2}s\n", load_start.elapsed().as_secs_f64());

    let topology_config = TopologyConfig {
        min_persistence: 0.1,
        max_scale: 1.0,
        num_scales: 10,
        detect_cycles: true,
        detect_voids: false,
    };

    // Pre-extract all Layer 21 activations for speed
    println!("================================================================");
    println!("   PRE-EXTRACTING LAYER 21 ACTIVATIONS");
    println!("================================================================\n");

    println!("Extracting phenomenal...");
    let mut phen_hvs: Vec<BinaryHV> = Vec::new();
    for (i, concept) in phenomenal.iter().enumerate() {
        if i % 20 == 0 {
            print!("  {}/{}\\r", i, phenomenal.len());
        }
        let acts = extractor.extract_layers(concept, &[21])?;
        phen_hvs.push(activation_to_hv16(&acts[0].activation));
    }
    println!("  Done                    ");

    println!("Extracting functional...");
    let mut func_hvs: Vec<BinaryHV> = Vec::new();
    for (i, concept) in functional.iter().enumerate() {
        if i % 20 == 0 {
            print!("  {}/{}\\r", i, functional.len());
        }
        let acts = extractor.extract_layers(concept, &[21])?;
        func_hvs.push(activation_to_hv16(&acts[0].activation));
    }
    println!("  Done\n");

    // Test 1: Random subsets
    println!("================================================================");
    println!("   TEST 1: RANDOM SUBSETS (5 trials, 50 concepts each)");
    println!("================================================================\n");

    let mut subset_diffs = Vec::new();
    let mut subset_pvalues = Vec::new();

    println!("Trial │ Phen Unity │ Func Unity │ Diff   │ p-value");
    println!("──────┼────────────┼────────────┼────────┼────────");

    for trial in 0..5 {
        let seed = 42 + trial as u64 * 1000;

        // Random subset selection
        let phen_indices: Vec<usize> = random_subset(phen_hvs.len(), 50, seed);
        let func_indices: Vec<usize> = random_subset(func_hvs.len(), 50, seed + 1);

        let phen_sample: Vec<f64> = phen_indices
            .iter()
            .map(|&i| compute_unity(&phen_hvs[i], &topology_config))
            .collect();
        let func_sample: Vec<f64> = func_indices
            .iter()
            .map(|&i| compute_unity(&func_hvs[i], &topology_config))
            .collect();

        let phen_mean = mean(&phen_sample);
        let func_mean = mean(&func_sample);
        let diff = phen_mean - func_mean;
        let p = permutation_test(&phen_sample, &func_sample, 2000);

        println!(
            "{:5} │ {:10.4} │ {:10.4} │ {:+6.4} │ {:.4}{}",
            trial + 1,
            phen_mean,
            func_mean,
            diff,
            p,
            if p < 0.05 { "*" } else { "" }
        );

        subset_diffs.push(diff);
        subset_pvalues.push(p);
    }

    let consistent_direction = subset_diffs.iter().all(|&d| d > 0.0);
    let all_significant = subset_pvalues.iter().all(|&p| p < 0.05);
    let any_significant = subset_pvalues.iter().any(|&p| p < 0.05);

    println!("\nSubset Results:");
    println!(
        "  Consistent positive direction: {}",
        if consistent_direction {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!("  Mean difference: {:+.4}", mean(&subset_diffs));
    println!(
        "  Significant trials: {}/5",
        subset_pvalues.iter().filter(|&&p| p < 0.05).count()
    );

    // Test 2: Bootstrap resampling
    println!("\n================================================================");
    println!("   TEST 2: BOOTSTRAP RESAMPLING (1000 iterations)");
    println!("================================================================\n");

    let mut bootstrap_diffs = Vec::new();

    for boot in 0..1000 {
        if boot % 100 == 0 {
            print!("  Bootstrap {}/1000\\r", boot);
        }
        let seed = boot as u64 * 17;

        // Bootstrap resample with replacement
        let phen_sample: Vec<f64> = bootstrap_sample(&phen_hvs, seed)
            .iter()
            .map(|hv| compute_unity(hv, &topology_config))
            .collect();
        let func_sample: Vec<f64> = bootstrap_sample(&func_hvs, seed + 1)
            .iter()
            .map(|hv| compute_unity(hv, &topology_config))
            .collect();

        bootstrap_diffs.push(mean(&phen_sample) - mean(&func_sample));
    }
    println!("  Done                    \n");

    let boot_mean = mean(&bootstrap_diffs);
    let boot_std = std_dev(&bootstrap_diffs);
    let ci_lower = percentile(&bootstrap_diffs, 2.5);
    let ci_upper = percentile(&bootstrap_diffs, 97.5);

    println!("Bootstrap Results:");
    println!("  Mean difference: {:+.4}", boot_mean);
    println!("  Standard error: {:.4}", boot_std);
    println!("  95% CI: [{:+.4}, {:+.4}]", ci_lower, ci_upper);
    println!(
        "  CI excludes zero: {}",
        if ci_lower > 0.0 {
            "✓ YES (significant)"
        } else {
            "✗ NO"
        }
    );

    // Test 3: Cross-validation folds
    println!("\n================================================================");
    println!("   TEST 3: 5-FOLD CROSS-VALIDATION");
    println!("================================================================\n");

    let mut fold_diffs = Vec::new();

    println!("Fold │ Train Phen │ Train Func │ Test Phen │ Test Func │ Test Diff");
    println!("─────┼────────────┼────────────┼───────────┼───────────┼──────────");

    for fold in 0..5 {
        let (train_phen, test_phen) = cross_val_split(&phen_hvs, fold, 5);
        let (train_func, test_func) = cross_val_split(&func_hvs, fold, 5);

        let train_phen_unity: Vec<f64> = train_phen
            .iter()
            .map(|hv| compute_unity(hv, &topology_config))
            .collect();
        let train_func_unity: Vec<f64> = train_func
            .iter()
            .map(|hv| compute_unity(hv, &topology_config))
            .collect();
        let test_phen_unity: Vec<f64> = test_phen
            .iter()
            .map(|hv| compute_unity(hv, &topology_config))
            .collect();
        let test_func_unity: Vec<f64> = test_func
            .iter()
            .map(|hv| compute_unity(hv, &topology_config))
            .collect();

        let test_diff = mean(&test_phen_unity) - mean(&test_func_unity);
        fold_diffs.push(test_diff);

        println!(
            "{:4} │ {:10.4} │ {:10.4} │ {:9.4} │ {:9.4} │ {:+8.4}",
            fold + 1,
            mean(&train_phen_unity),
            mean(&train_func_unity),
            mean(&test_phen_unity),
            mean(&test_func_unity),
            test_diff
        );
    }

    let cv_mean = mean(&fold_diffs);
    let cv_std = std_dev(&fold_diffs);

    println!("\nCross-Validation Results:");
    println!("  Mean test difference: {:+.4}", cv_mean);
    println!("  Standard deviation: {:.4}", cv_std);
    println!(
        "  Coefficient of variation: {:.2}%",
        (cv_std / cv_mean.abs()) * 100.0
    );

    // Summary
    println!("\n================================================================");
    println!("   ROBUSTNESS SUMMARY");
    println!("================================================================\n");

    let robust = consistent_direction && ci_lower > 0.0;

    if robust {
        println!("✓ EFFECT IS ROBUST");
        println!("  - Consistent positive direction across all random subsets");
        println!("  - Bootstrap 95% CI excludes zero");
        println!(
            "  - Cross-validation shows stable effect (CV: {:.1}%)",
            (cv_std / cv_mean.abs()) * 100.0
        );
    } else if consistent_direction {
        println!("◐ EFFECT IS CONSISTENT BUT VARIABLE");
        println!("  - Direction is consistent (+phenomenal) across samples");
        println!("  - But magnitude varies - may need larger corpus");
    } else {
        println!("✗ EFFECT IS NOT ROBUST");
        println!("  - Direction varies across samples");
        println!("  - Original finding may be sample-specific");
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
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

#[cfg(feature = "neural-bridge")]
fn percentile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(feature = "neural-bridge")]
fn random_subset(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = seed;

    for i in (1..indices.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng as usize) % (i + 1);
        indices.swap(i, j);
    }

    indices.into_iter().take(k).collect()
}

#[cfg(feature = "neural-bridge")]
fn bootstrap_sample<T: Clone>(items: &[T], seed: u64) -> Vec<T> {
    let n = items.len();
    let mut rng = seed;
    let mut result = Vec::with_capacity(n);

    for _ in 0..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (rng as usize) % n;
        result.push(items[idx].clone());
    }

    result
}

#[cfg(feature = "neural-bridge")]
fn cross_val_split<T: Clone>(items: &[T], fold: usize, n_folds: usize) -> (Vec<T>, Vec<T>) {
    let fold_size = items.len() / n_folds;
    let test_start = fold * fold_size;
    let test_end = if fold == n_folds - 1 {
        items.len()
    } else {
        (fold + 1) * fold_size
    };

    let test: Vec<T> = items[test_start..test_end].to_vec();
    let train: Vec<T> = items[..test_start]
        .iter()
        .chain(items[test_end..].iter())
        .cloned()
        .collect();

    (train, test)
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
