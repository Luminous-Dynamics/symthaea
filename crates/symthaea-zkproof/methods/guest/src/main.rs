#![no_main]

use risc0_zkvm::guest::env;
use symthaea_zkproof_core::{
    BalanceProofInput, BalanceProofOutput, EvolutionInput, EvolutionOutput,
};

risc0_zkvm::guest::entry!(main);

/// Guest entry point.
///
/// The host selects the proof type by writing either an `EvolutionInput` or a
/// `BalanceProofInput`.  We use a `u8` tag to dispatch:
///   0 => consciousness attestation (EvolutionInput → EvolutionOutput)
///   1 => balance sufficiency    (BalanceProofInput → BalanceProofOutput)
fn main() {
    let tag: u8 = env::read();

    match tag {
        0 => prove_evolution(),
        1 => prove_balance(),
        _ => panic!("unknown proof tag: {tag}"),
    }
}

/// Consciousness attestation proof.
fn prove_evolution() {
    let input: EvolutionInput = env::read();

    let mut total_phi = 0.0_f32;
    let n = input.episodes.len() as f32;

    for episode in &input.episodes {
        if episode.is_empty() {
            continue;
        }
        let mean = episode.iter().sum::<f32>() / episode.len() as f32;
        let variance = episode.iter().map(|x| (x - mean).powi(2)).sum::<f32>();
        let phi = variance * input.tau_scale;
        total_phi += phi;
    }

    let avg_phi = if n > 0.0 { total_phi / n } else { 0.0 };

    env::commit(&EvolutionOutput {
        average_phi: avg_phi,
        tau_scale: input.tau_scale,
        episode_count: input.episodes.len() as u32,
    });
}

/// Balance sufficiency proof.
/// Proves: balance >= required_minimum WITHOUT revealing the actual balance.
fn prove_balance() {
    let input: BalanceProofInput = env::read();
    let sufficient = input.balance >= input.required_minimum;

    env::commit(&BalanceProofOutput {
        sufficient,
        required_minimum: input.required_minimum,
        nonce: input.nonce,
    });
}
