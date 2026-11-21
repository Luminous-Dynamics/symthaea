// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic HDC representation diagnostic (legacy path: benchmark_pci_validation.rs).
//!
//! This example does NOT implement clinical PCI, TMS–EEG, or an intervention-based
//! perturbational-complexity algorithm. It constructs hand-parameterized synthetic
//! HDC networks, computes Symthaea's internal PhiEngine output, and measures the
//! entropy of the pairwise similarity distribution around selected reference nodes.
//! No perturbation is applied to the network in this example.
//!
//! The profile parameters are arbitrary synthetic controls, not encodings of
//! wakefulness, sleep, anesthesia, vegetative state, or minimally conscious state.
//! These values cannot be compared with clinical PCI values or cutoffs. The
//! PhiEngine output is an internal architecture metric, not a measurement of
//! phenomenal consciousness or biologically measured integrated information.
//!
//! This executable is retained as a smoke-test / exploratory synthetic fixture.
//! It reports no biological, clinical, or theory-validation result.
//!
//! ## Run
//!
//! \`\`\`bash
//! cargo run --example benchmark_pci_validation --release
//! \`\`\`

use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::phi_engine::PhiEngine;

const DIM: usize = 4096;
const NODE_COUNT: usize = 16;
const TRIAL_COUNT: usize = 5;

#[derive(Clone, Copy)]
struct SyntheticNetworkRegime {
    name: &'static str,
    coupling_strength: f32,
    recurrence: f32,
    noise_level: f32,
}

fn synthetic_network_regimes() -> Vec<SyntheticNetworkRegime> {
    // These are arbitrary coverage points in parameter space, not human/clinical states.
    vec![
        SyntheticNetworkRegime { name: "profile-01", coupling_strength: 0.60, recurrence: 0.70, noise_level: 0.10 },
        SyntheticNetworkRegime { name: "profile-02", coupling_strength: 0.50, recurrence: 0.50, noise_level: 0.15 },
        SyntheticNetworkRegime { name: "profile-03", coupling_strength: 0.30, recurrence: 0.30, noise_level: 0.20 },
        SyntheticNetworkRegime { name: "profile-04", coupling_strength: 0.15, recurrence: 0.15, noise_level: 0.25 },
        SyntheticNetworkRegime { name: "profile-05", coupling_strength: 0.10, recurrence: 0.10, noise_level: 0.30 },
        SyntheticNetworkRegime { name: "profile-06", coupling_strength: 0.35, recurrence: 0.20, noise_level: 0.40 },
        SyntheticNetworkRegime { name: "profile-07", coupling_strength: 0.08, recurrence: 0.05, noise_level: 0.30 },
        SyntheticNetworkRegime { name: "profile-08", coupling_strength: 0.25, recurrence: 0.25, noise_level: 0.20 },
    ]
}

fn create_synthetic_network(
    regime: &SyntheticNetworkRegime,
    n_nodes: usize,
    dim: usize,
    seed: u64,
) -> Vec<ContinuousHV> {
    let base_hvs: Vec<ContinuousHV> = (0..n_nodes)
        .map(|i| ContinuousHV::random(dim, seed + i as u64))
        .collect();

    let shared_signal = ContinuousHV::random(dim, seed + 10_000);
    let recurrent_template = ContinuousHV::random(dim, seed + 20_000);

    base_hvs
        .iter()
        .enumerate()
        .map(|(i, base)| {
            let mut values = vec![0.0f32; dim];

            let identity_weight = 1.0 - regime.coupling_strength;
            for (value, &base_value) in values.iter_mut().zip(base.values.iter()) {
                *value += identity_weight * base_value;
            }

            for (value, &shared_value) in values.iter_mut().zip(shared_signal.values.iter()) {
                *value += regime.coupling_strength * shared_value;
            }

            if i < (n_nodes as f32 * regime.recurrence) as usize {
                let recurrent_hv = base.bind(&recurrent_template);
                for (value, &recurrent_value) in values.iter_mut().zip(recurrent_hv.values.iter()) {
                    *value += regime.recurrence * 0.3 * recurrent_value;
                }
            }

            let noise = ContinuousHV::random(dim, seed + 30_000 + i as u64);
            for (value, &noise_value) in values.iter_mut().zip(noise.values.iter()) {
                *value += regime.noise_level * noise_value;
            }

            let norm: f32 = values.iter().map(|value| value * value).sum::<f32>().sqrt();
            if norm > 0.0 {
                for value in &mut values {
                    *value /= norm;
                }
            }

            ContinuousHV::from_vec(values)
        })
        .collect()
}

/// Shannon entropy of pairwise cosine-similarity bins around one reference node.
///
/// This is a descriptive statistic of a synthetic HDC representation. It is not
/// Lempel–Ziv complexity, PCI, causal perturbation propagation, or a consciousness
/// measure. The selected node is observed, not physically or computationally perturbed.
fn reference_node_similarity_entropy(network: &[ContinuousHV], reference_idx: usize) -> f64 {
    if network.len() < 2 || reference_idx >= network.len() {
        return 0.0;
    }

    let reference = &network[reference_idx];
    let similarities: Vec<f32> = network
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != reference_idx)
        .map(|(_, vector)| reference.similarity(vector))
        .collect();

    if similarities.is_empty() {
        return 0.0;
    }

    const BIN_COUNT: usize = 10;
    let mut bins = [0usize; BIN_COUNT];

    for similarity in similarities.iter().copied() {
        let bin = (((similarity.clamp(-1.0, 1.0) + 1.0) * 0.5)
            * (BIN_COUNT - 1) as f32) as usize;
        bins[bin.min(BIN_COUNT - 1)] += 1;
    }

    let total = similarities.len() as f64;
    let entropy = bins
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let probability = count as f64 / total;
            -probability * probability.ln()
        })
        .sum::<f64>();

    entropy / (BIN_COUNT as f64).ln()
}

fn mean_and_sample_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (mean, variance.sqrt())
}

struct RegimeResult {
    profile: &'static str,
    phi_engine_output_mean: f64,
    phi_engine_output_sample_std: f64,
    reference_node_similarity_entropy_mean: f64,
    reference_node_similarity_entropy_sample_std: f64,
}

fn main() {
    println!("Synthetic HDC representation diagnostic");
    println!("No TMS–EEG, clinical PCI, human data, or intervention is used.\n");

    let phi_engine = PhiEngine::auto();
    let regimes = synthetic_network_regimes();
    let mut results = Vec::with_capacity(regimes.len());

    println!(
        "{:14} │ {:>16} │ {:>21}",
        "Profile", "PhiEngine output", "Similarity entropy"
    );
    println!("───────────────┼──────────────────┼───────────────────────");

    for (regime_index, regime) in regimes.iter().enumerate() {
        let mut phi_values = Vec::with_capacity(TRIAL_COUNT);
        let mut entropy_values = Vec::with_capacity(TRIAL_COUNT);

        for trial in 0..TRIAL_COUNT {
            let seed = 42 + (regime_index as u64) * 100_000 + (trial as u64) * 1_000;
            let network = create_synthetic_network(regime, NODE_COUNT, DIM, seed);
            let phi_result = phi_engine.compute(&network);
            phi_values.push(phi_result.phi);

            // Several reference nodes are summarized; no node is perturbed.
            let reference_nodes = [0, NODE_COUNT / 4, NODE_COUNT / 2];
            let trial_entropy = reference_nodes
                .iter()
                .map(|&index| reference_node_similarity_entropy(&network, index))
                .sum::<f64>()
                / reference_nodes.len() as f64;
            entropy_values.push(trial_entropy);
        }

        let (phi_mean, phi_std) = mean_and_sample_std(&phi_values);
        let (entropy_mean, entropy_std) = mean_and_sample_std(&entropy_values);

        println!(
            "{:14} │ {:>8.4} ± {:>5.3} │ {:>10.4} ± {:>5.3}",
            regime.name, phi_mean, phi_std, entropy_mean, entropy_std
        );

        results.push(RegimeResult {
            profile: regime.name,
            phi_engine_output_mean: phi_mean,
            phi_engine_output_sample_std: phi_std,
            reference_node_similarity_entropy_mean: entropy_mean,
            reference_node_similarity_entropy_sample_std: entropy_std,
        });
    }

    let all_finite = results.iter().all(|result| {
        result.phi_engine_output_mean.is_finite()
            && result.phi_engine_output_sample_std.is_finite()
            && result.reference_node_similarity_entropy_mean.is_finite()
            && result.reference_node_similarity_entropy_sample_std.is_finite()
    });
    let entropy_in_range = results.iter().all(|result| {
        (0.0..=1.0).contains(&result.reference_node_similarity_entropy_mean)
    });

    println!("\nSynthetic fixture checks:");
    println!("  All reported values finite: {}", all_finite);
    println!("  Similarity entropy lies in [0, 1]: {}", entropy_in_range);
    println!("\nInterpretation: synthetic representation diagnostics only; not clinical PCI, biological consciousness evidence, or a clinical classifier.");

    let output = serde_json::json!({
        "benchmark": "Synthetic HDC representation diagnostic",
        "legacy_source_path": "examples/benchmark_pci_validation.rs",
        "scope": "synthetic_only_not_clinical_pci",
        "no_intervention_applied": true,
        "timestamp_utc": chrono::Utc::now().to_rfc3339(),
        "node_count": NODE_COUNT,
        "trial_count_per_profile": TRIAL_COUNT,
        "hypervector_dimension": DIM,
        "profiles": results.iter().map(|result| serde_json::json!({
            "profile": result.profile,
            "phi_engine_output_mean": result.phi_engine_output_mean,
            "phi_engine_output_sample_std": result.phi_engine_output_sample_std,
            "reference_node_similarity_entropy_mean": result.reference_node_similarity_entropy_mean,
            "reference_node_similarity_entropy_sample_std": result.reference_node_similarity_entropy_sample_std
        })).collect::<Vec<_>>(),
        "fixture_checks": {
            "all_values_finite": all_finite,
            "similarity_entropy_in_unit_interval": entropy_in_range
        }
    });

    let output_path = "synthetic_hdc_representation_diagnostic.json";
    match std::fs::File::create(output_path) {
        Ok(file) => {
            if let Err(error) = serde_json::to_writer_pretty(file, &output) {
                eprintln!("Could not serialize result JSON: {error}");
            } else {
                println!("Result manifest written to {output_path}");
            }
        }
        Err(error) => eprintln!("Could not create result JSON at {output_path}: {error}"),
    }
}
