#![no_main]

use risc0_zkvm::guest::env;
use symthaea_zkproof_core::{EvolutionInput, EvolutionOutput};

risc0_zkvm::guest::entry!(main);

fn main() {
    let input: EvolutionInput = env::read();

    // 1. Compute the verified Phi breakthrough
    // In the guest, we use a simplified version of the Spectral Connectivity proxy
    // to fit within the cycle limits.
    
    let mut total_phi = 0.0;
    let n = input.episodes.len() as f32;

    for episode in &input.episodes {
        // Simplified Laplacian connectivity proxy:
        // Sum of squared distances to mean (proxy for clustering/coherence)
        let mean = episode.iter().sum::<f32>() / episode.len() as f32;
        let variance = episode.iter().map(|x| (x - mean).powi(2)).sum::<f32>();
        
        // Scale by tau (simulating the CfC effect)
        let phi = variance * input.tau_scale;
        total_phi += phi;
    }

    let avg_phi = if n > 0.0 { total_phi / n } else { 0.0 };

    // 2. Commit the output as a public proof
    let output = EvolutionOutput {
        average_phi: avg_phi,
        tau_scale: input.tau_scale,
        episode_count: input.episodes.len() as u32,
    };

    env::commit(&output);
}
