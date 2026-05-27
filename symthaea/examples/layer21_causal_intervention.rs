// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Layer 21 Causal Intervention Experiment
//!
//! Tests whether Layer 21 is *causally necessary* for the phenomenal distinction.
//!
//! ## Intervention Methods
//!
//! 1. **Zero-out**: Set Layer 21 activations to zero
//! 2. **Noise injection**: Add Gaussian noise to Layer 21
//! 3. **Shuffle**: Randomly permute activations within Layer 21
//!
//! ## Hypothesis
//!
//! If Layer 21 is causally necessary, interventions should:
//! - Reduce or eliminate the phenomenal/functional distinction
//! - Effect should be stronger for Layer 21 than control layers (18, 23)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example layer21_causal_intervention --features neural-bridge --release
//! ```

use anyhow::Result;
use std::time::Instant;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::{LayerExtractor, PoolingMethod, layer_extractor::LayerExtractorConfig};

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::{HDC_DIMENSION, binary_hv::BinaryHV};

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
    println!("   LAYER 21 CAUSAL INTERVENTION");
    println!("   Is Layer 21 causally necessary for phenomenal distinction?");
    println!("================================================================\n");

    // Load corpora
    let phenomenal = load_corpus("data/consciousness_probe/phenomenal_concepts_expanded.json")?;
    let functional = load_corpus("data/consciousness_probe/functional_concepts_expanded.json")?;

    // Use subset
    let phenomenal: Vec<_> = phenomenal.into_iter().take(50).collect();
    let functional: Vec<_> = functional.into_iter().take(50).collect();

    println!(
        "Using {} phenomenal, {} functional concepts\n",
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

    // Layers to test interventions on
    let intervention_layers = vec![18, 21, 23];

    println!("================================================================");
    println!("   BASELINE: NO INTERVENTION");
    println!("================================================================\n");

    // First, get baseline at Layer 21
    let (baseline_phen, baseline_func) = extract_unity_scores(
        &extractor,
        &phenomenal,
        &functional,
        21,
        &topology_config,
        None,
    )?;

    let baseline_diff = mean(&baseline_phen) - mean(&baseline_func);
    let baseline_p = permutation_test(&baseline_phen, &baseline_func, 5000);

    println!("Layer 21 Baseline:");
    println!("  Phenomenal unity: {:.4}", mean(&baseline_phen));
    println!("  Functional unity: {:.4}", mean(&baseline_func));
    println!("  Difference: {:+.4}", baseline_diff);
    println!(
        "  p-value: {:.4} {}\n",
        baseline_p,
        if baseline_p < 0.05 { "*" } else { "" }
    );

    // Intervention types
    let interventions = vec![
        ("Zero-out", InterventionType::ZeroOut),
        ("Noise (σ=0.5)", InterventionType::Noise(0.5)),
        ("Noise (σ=1.0)", InterventionType::Noise(1.0)),
        ("Shuffle", InterventionType::Shuffle),
    ];

    println!("================================================================");
    println!("   INTERVENTIONS AT EACH LAYER");
    println!("================================================================\n");

    // Results storage
    let mut results: Vec<(usize, &str, f64, f64, f64)> = Vec::new(); // layer, intervention, diff, p, effect_reduction

    for &layer in &intervention_layers {
        println!("────────────────────────────────────────");
        println!("Layer {}", layer);
        println!("────────────────────────────────────────\n");

        for (name, intervention) in &interventions {
            let (phen_unity, func_unity) = extract_unity_scores(
                &extractor,
                &phenomenal,
                &functional,
                layer,
                &topology_config,
                Some(intervention.clone()),
            )?;

            let diff = mean(&phen_unity) - mean(&func_unity);
            let p = permutation_test(&phen_unity, &func_unity, 2000);

            // Effect reduction: how much did the intervention reduce the phenomenal advantage?
            let effect_reduction = if baseline_diff != 0.0 {
                ((baseline_diff - diff) / baseline_diff) * 100.0
            } else {
                0.0
            };

            println!(
                "  {}: diff={:+.4}, p={:.4}{}, reduction={:.1}%",
                name,
                diff,
                p,
                if p < 0.05 { "*" } else { " " },
                effect_reduction
            );

            results.push((layer, name, diff, p, effect_reduction));
        }
        println!();
    }

    // Analysis
    println!("================================================================");
    println!("   CAUSAL ANALYSIS");
    println!("================================================================\n");

    println!("Effect Reduction by Layer and Intervention:\n");
    println!("Layer │ Zero-out │ Noise 0.5 │ Noise 1.0 │ Shuffle");
    println!("──────┼──────────┼───────────┼───────────┼────────");

    for &layer in &intervention_layers {
        let layer_results: Vec<_> = results
            .iter()
            .filter(|(l, _, _, _, _)| *l == layer)
            .collect();

        print!("{:5} │", layer);
        for (_, _, _, _, reduction) in layer_results {
            print!(" {:8.1}% │", reduction);
        }
        println!();
    }

    // Determine if Layer 21 is special
    println!("\n----------------------------------------------------------------");

    let layer21_reductions: Vec<f64> = results
        .iter()
        .filter(|(l, _, _, _, _)| *l == 21)
        .map(|(_, _, _, _, r)| *r)
        .collect();

    let layer18_reductions: Vec<f64> = results
        .iter()
        .filter(|(l, _, _, _, _)| *l == 18)
        .map(|(_, _, _, _, r)| *r)
        .collect();

    let layer23_reductions: Vec<f64> = results
        .iter()
        .filter(|(l, _, _, _, _)| *l == 23)
        .map(|(_, _, _, _, r)| *r)
        .collect();

    let mean_21 = mean(&layer21_reductions);
    let mean_18 = mean(&layer18_reductions);
    let mean_23 = mean(&layer23_reductions);

    println!("Mean effect reduction:");
    println!("  Layer 18: {:.1}%", mean_18);
    println!("  Layer 21: {:.1}%", mean_21);
    println!("  Layer 23: {:.1}%", mean_23);

    // Interpretation
    println!("\n================================================================");
    println!("   INTERPRETATION");
    println!("================================================================\n");

    if mean_21 > mean_18 && mean_21 > mean_23 && mean_21 > 30.0 {
        println!("✓ LAYER 21 IS CAUSALLY IMPORTANT");
        println!("  Interventions at Layer 21 cause the largest reduction in phenomenal effect");
        println!(
            "  Mean reduction: {:.1}% (vs {:.1}% at L18, {:.1}% at L23)",
            mean_21, mean_18, mean_23
        );
        println!("\n  This supports the hypothesis that Layer 21 is where phenomenal");
        println!("  structure is encoded, not just correlated with.");
    } else if mean_21 > 20.0 {
        println!("◐ LAYER 21 CONTRIBUTES TO PHENOMENAL EFFECT");
        println!("  Interventions reduce the effect, but not more than other layers");
        println!("  The effect may be distributed across late layers");
    } else {
        println!("○ NO CLEAR CAUSAL ROLE FOR LAYER 21");
        println!("  Interventions don't substantially reduce the phenomenal effect");
        println!("  The effect may arise from earlier processing");
    }

    // Strongest intervention
    let strongest = results
        .iter()
        .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap())
        .unwrap();

    println!(
        "\nStrongest intervention: {} at Layer {} ({:.1}% reduction)",
        strongest.1, strongest.0, strongest.4
    );

    println!("\n================================================================");
    println!("   EXPERIMENT COMPLETE");
    println!("================================================================\n");

    Ok(())
}

#[cfg(feature = "neural-bridge")]
#[derive(Clone)]
enum InterventionType {
    ZeroOut,
    Noise(f64),
    Shuffle,
}

#[cfg(feature = "neural-bridge")]
fn extract_unity_scores(
    extractor: &LayerExtractor,
    phenomenal: &[String],
    functional: &[String],
    layer: usize,
    topology_config: &TopologyConfig,
    intervention: Option<InterventionType>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut phen_unity = Vec::new();
    let mut func_unity = Vec::new();

    for concept in phenomenal {
        let acts = extractor.extract_layers(concept, &[layer])?;
        let mut activation = acts[0].activation.clone();

        // Apply intervention
        if let Some(ref int_type) = intervention {
            apply_intervention(&mut activation, int_type);
        }

        let hv = activation_to_hv16(&activation);
        phen_unity.push(compute_unity(&hv, topology_config));
    }

    for concept in functional {
        let acts = extractor.extract_layers(concept, &[layer])?;
        let mut activation = acts[0].activation.clone();

        if let Some(ref int_type) = intervention {
            apply_intervention(&mut activation, int_type);
        }

        let hv = activation_to_hv16(&activation);
        func_unity.push(compute_unity(&hv, topology_config));
    }

    Ok((phen_unity, func_unity))
}

#[cfg(feature = "neural-bridge")]
fn apply_intervention(activation: &mut [f32], intervention: &InterventionType) {
    match intervention {
        InterventionType::ZeroOut => {
            for v in activation.iter_mut() {
                *v = 0.0;
            }
        }
        InterventionType::Noise(sigma) => {
            let mut rng: u64 = 12345;
            for v in activation.iter_mut() {
                // Box-Muller for Gaussian
                let u1 = pseudo_random(&mut rng);
                let u2 = pseudo_random(&mut rng);
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                *v += (normal * sigma) as f32;
            }
        }
        InterventionType::Shuffle => {
            let mut rng: u64 = 54321;
            for i in (1..activation.len()).rev() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (rng as usize) % (i + 1);
                activation.swap(i, j);
            }
        }
    }
}

#[cfg(feature = "neural-bridge")]
fn pseudo_random(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*seed as f64) / (u64::MAX as f64)
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
