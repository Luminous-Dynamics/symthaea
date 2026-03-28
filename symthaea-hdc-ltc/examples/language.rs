// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mini thought-to-text demonstration.
//!
//! Creates 26 letter embeddings as random hypervectors, encodes a "thought"
//! as a bundle of letter HVs, feeds it through an HDC-LTC neuron, and
//! decodes the output by finding the most similar letter embedding.

use symthaea_hdc_ltc::{ContinuousHV, HdcLtcUnifiedNeuron, NeuronConfig};

fn main() {
    let dim = 1024;
    let config = NeuronConfig {
        dim,
        tau_base: 0.08,
        ..NeuronConfig::default()
    };

    // Create deterministic letter embeddings (a=0, b=1, ..., z=25)
    let letters: Vec<ContinuousHV> = (0..26)
        .map(|i| ContinuousHV::new_random(dim, 5000 + i as u64))
        .collect();
    let letter_names: Vec<char> = ('a'..='z').collect();

    // Encode a "thought" as the bundle of letter HVs for "hello"
    let thought_letters = [7, 4, 11, 11, 14]; // h, e, l, l, o
    let thought_refs: Vec<&ContinuousHV> = thought_letters.iter().map(|&i| &letters[i]).collect();
    let thought = ContinuousHV::bundle(&thought_refs);

    println!("Thought: 'hello' encoded as bundle of letter HVs");
    println!("\nDecoding via HDC-LTC neuron evolution:\n");

    let mut neuron = HdcLtcUnifiedNeuron::new(config, 42);

    // Generate a sequence by repeatedly evolving and decoding
    let mut generated = String::new();
    let dt = 0.1;

    for step in 0..20 {
        // Feed the thought (input stays constant, neuron state evolves)
        neuron.evolve_closed_form(dt, &thought);

        // Decode: find the letter most similar to the current state
        let state = neuron.state();
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for (i, letter_hv) in letters.iter().enumerate() {
            let sim = state.similarity(letter_hv);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        let decoded = letter_names[best_idx];
        generated.push(decoded);

        println!(
            "  Step {:2}: decoded '{}' (similarity: {:.4}, state norm: {:.3})",
            step,
            decoded,
            best_sim,
            state.norm()
        );
    }

    println!("\nGenerated sequence: \"{}\"", generated);
    println!("\nNote: This is a minimal demo. Real language generation uses");
    println!("Broca's 43-channel ThoughtEncoder with epistemic gating.");
}
