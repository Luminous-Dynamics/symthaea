#![no_main]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![no_std]

use risc0_zkvm::guest::env;
use symthaea_zkproof_core::{
    BalanceProofInput, BalanceProofOutput, BalanceProofRequestBoundInputV1,
    BalanceProofRequestBoundOutputV1, EvolutionInput, EvolutionOutput,
};

risc0_zkvm::guest::entry!(main);

fn main() {
    // Tag-dispatched entry point:
    //   tag 0 = consciousness attestation (EvolutionInput → EvolutionOutput)
    //   tag 1 = legacy balance proof     (BalanceProofInput → BalanceProofOutput)
    //   tag 2 = request-bound balance proof (BalanceProofRequestBoundInputV1 → BalanceProofRequestBoundOutputV1)
    let tag: u8 = env::read();

    match tag {
        0 => run_consciousness_attestation(),
        1 => run_balance_proof(),
        2 => run_request_bound_balance_proof_v1(),
        _ => panic!("unknown guest program tag"),
    }
}

/// Tag 0: Consciousness attestation — simplified Laplacian connectivity proxy.
fn run_consciousness_attestation() {
    let input: EvolutionInput = env::read();

    let mut total_phi = 0.0_f32;
    let n = input.episodes.len() as f32;

    for episode in &input.episodes {
        if episode.is_empty() {
            continue;
        }
        let mean = episode.iter().sum::<f32>() / episode.len() as f32;
        let variance = episode
            .iter()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f32>();
        let phi = variance * input.tau_scale;
        total_phi += phi;
    }

    let avg_phi = if n > 0.0 { total_phi / n } else { 0.0 };

    let output = EvolutionOutput {
        average_phi: avg_phi,
        tau_scale: input.tau_scale,
        episode_count: input.episodes.len() as u32,
    };

    env::commit(&output);
}

/// Tag 1: Balance sufficiency proof — proves `balance >= required_minimum`
/// without revealing the actual balance.
fn run_balance_proof() {
    let input: BalanceProofInput = env::read();

    let output = BalanceProofOutput {
        sufficient: input.balance >= input.required_minimum,
        required_minimum: input.required_minimum,
        nonce: input.nonce,
    };

    env::commit(&output);
}


/// Tag 2: request-bound balance predicate.
///
/// This binds the public request context to the journal, but does NOT verify
/// the referenced issuer statement, prove that the private balance opens a
/// source commitment, issue/consume the challenge, or establish authority.
fn run_request_bound_balance_proof_v1() {
    let input: BalanceProofRequestBoundInputV1 = env::read();

    // Malformed request contexts are not eligible for a positive receipt.
    assert!(input.request.validate().is_ok());

    let output = BalanceProofRequestBoundOutputV1 {
        sufficient: input.balance >= input.request.required_minimum,
        required_minimum: input.request.required_minimum,
        request_digest: input.request.canonical_digest(),
    };

    env::commit(&output);
}
