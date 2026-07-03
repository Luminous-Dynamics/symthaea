// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use mycelix_zkp_core::consciousness::CivicTier;
use std::error::Error;
use std::io::{self, Write};
use std::path::Path;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator};
use symthaea_broca_tools::invariant_guard::AxiomaticInvariantGuard;
use symthaea_broca_tools::memory_ring::HdcContextRing;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

struct SemanticPathIntegrator {
    coord_x: f64,
    coord_y: f64,
}

impl SemanticPathIntegrator {
    fn new() -> Self {
        Self {
            coord_x: 0.0,
            coord_y: 0.0,
        }
    }
    fn compute_step(&mut self, accuracy: f64, surprise: f64) {
        self.coord_x += accuracy * 0.15;
        self.coord_y += surprise * 0.15;
    }
    fn distance_from_origin(&self) -> f64 {
        ((self.coord_x * self.coord_x) + (self.coord_y * self.coord_y)).sqrt()
    }
}

fn sample_uniform(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed as f32) / (u32::MAX as f32)
}

fn extract_active_primes(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut primes = Vec::new();
    let prime_words: &[(&str, &str)] = &[
        ("i ", "I"),
        ("i'", "I"),
        ("you", "YOU"),
        ("someone", "SOMEONE"),
        ("something", "SOMETHING"),
        ("people", "PEOPLE"),
        ("body", "BODY"),
        ("think", "THINK"),
        ("know", "KNOW"),
        ("want", "WANT"),
        ("feel", "FEEL"),
        ("see", "SEE"),
        ("hear", "HEAR"),
        ("say", "SAY"),
        ("word", "WORDS"),
        ("true", "TRUE"),
        ("do ", "DO"),
        ("happen", "HAPPEN"),
        ("move", "MOVE"),
        ("good", "GOOD"),
        ("bad", "BAD"),
        ("not ", "NOT"),
        ("maybe", "MAYBE"),
        ("can ", "CAN"),
        ("because", "BECAUSE"),
        ("if ", "IF"),
        ("now", "NOW"),
        ("before", "BEFORE"),
        ("after", "AFTER"),
        ("here", "HERE"),
        ("inside", "INSIDE"),
        ("big", "BIG"),
        ("small", "SMALL"),
        ("very", "VERY"),
        ("more", "MORE"),
        ("some", "SOME"),
        ("all ", "ALL"),
        ("live", "LIVE"),
        ("alive", "LIVE"),
    ];
    for &(word, prime) in prime_words {
        if lower.contains(word) && !primes.contains(&prime.to_string()) {
            primes.push(prime.to_string());
        }
    }
    primes
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Symthaea Whitelist-Isolated Cognitive Node [v25.2-PROD]");
    println!("└─────────────────────────────────────────────────────────");

    let genesis = GenesisSeed::from_phrase("symthaea-fused-node-2026");
    let checkpoint_path = "data/models/broca-checkpoint-latest.bin";

    let mut generator_instance = if Path::new(checkpoint_path).exists() {
        println!(
            "Ingesting active background weights from {}...",
            checkpoint_path
        );
        match BrocaGenerator::from_checkpoint(checkpoint_path, &genesis) {
            Ok((loaded_gen, _adam, _metrics, _epoch_label)) => loaded_gen,
            Err(e) => {
                println!(
                    "!] Checkpoint load failed ({}), falling back to genesis blueprint.",
                    e
                );
                BrocaGenerator::new_4k(&genesis, BrocaConfig::default())
            }
        }
    } else {
        println!("!] No active background weights found. Launching default substrate.");
        BrocaGenerator::new_4k(&genesis, BrocaConfig::default())
    };

    let guard_rails = AxiomaticInvariantGuard::new(0.70, 0.03);
    let vocabulary_size = 4048;
    let mut rng_seed = 1337u32;
    let mut conversational_turn_count = 0usize;

    println!("\nEntering interactive session. Type \"exit\" or \"quit\" to leave.");
    println!("Commands: \"/reset\" (Flushes conversation memory context)\n");

    loop {
        print!("[turn-{} ] symthaea> ", conversational_turn_count);
        io::stdout().flush()?;

        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line)?;
        let trimmed = input_line.trim();

        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "exit" || trimmed == "quit" {
            println!("Shutting down multi-turn session.");
            break;
        }

        if trimmed == "/reset" {
            conversational_turn_count = 0;
            println!("-> Conversational context ring buffer successfully cleared.");
            continue;
        }

        let active_primes = extract_active_primes(trimmed);
        let token_count = generator_instance.tokenizer().encode(trimmed).len();
        let primitive_grounding = if token_count == 0 {
            0.0f32
        } else {
            (active_primes.len() as f32 / token_count as f32).min(1.0)
        };

        let mut active_intent = 1; // Answer
        let mut intent_label = "Answer";
        let mut epistemic_confidence = 0.85f32;

        if active_primes.contains(&"KNOW".to_string())
            || active_primes.contains(&"TRUE".to_string())
        {
            active_intent = 1;
            intent_label = "Answer";
            epistemic_confidence = 0.95f32;
        } else if active_primes.contains(&"MAYBE".to_string())
            || active_primes.contains(&"NOT".to_string())
        {
            active_intent = 4;
            intent_label = "Uncertainty";
            epistemic_confidence = 0.35f32;
        } else if active_primes.contains(&"THINK".to_string())
            || active_primes.contains(&"FEEL".to_string())
        {
            active_intent = 5;
            intent_label = "Reflect";
            epistemic_confidence = 0.65f32;
        } else if active_primes.is_empty() {
            active_intent = 6;
            intent_label = "Continue";
        }

        let mut channels = ThoughtChannels::default();
        for i in 0..8 {
            channels.channels[i] = if i == active_intent { 1.0 } else { 0.0 };
        }

        channels.set_epistemic(1.0 - epistemic_confidence);
        channels.set_emotion(0.3, 0.4, 0.6);
        channels.set_consciousness(0.8, 0.7, 0.85);
        channels.channels[19] = (active_primes.len() as f32 * 0.5).clamp(0.0, 10.0);
        channels.channels[21] = 0.4 * 0.5 + primitive_grounding * 0.5;
        channels.channels[23] = epistemic_confidence * 0.8 + primitive_grounding * 0.2;
        channels.channels[17] = if active_intent == 4 { 1.4 } else { 0.9 };

        // --- ENCODE PROMPT TEXT DIRECTLY INTO MANIFOLD SPACE ---
        let mut sensory_values = vec![0.0f32; HDC_DIMENSION];
        let prompt_tokens = generator_instance.tokenizer().encode(trimmed);
        if !prompt_tokens.is_empty() {
            let token_embeddings = generator_instance.controller().token_embeddings();
            for &t_id in &prompt_tokens {
                if (t_id as usize) < token_embeddings.len() {
                    let t_slice = token_embeddings[t_id as usize].as_slice();
                    for i in 0..HDC_DIMENSION {
                        sensory_values[i] += t_slice[i];
                    }
                }
            }
            for i in 0..HDC_DIMENSION {
                sensory_values[i] /= prompt_tokens.len() as f32;
            }
        }
        let base_sensory_thought = ContinuousHV::from_slice(&sensory_values);

        let mut memory_ring = HdcContextRing::new(0.85);
        generator_instance.controller_mut().reset();

        let mut active_token_id = prompt_tokens.last().copied().unwrap_or(1u32);
        let generation_steps = 14;

        let active_temperature = 0.85f32;
        let sensory_precision = 0.45f32;
        let context_precision = 0.35f32;
        let macro_influence = 0.20f32;

        let mut macro_thought_anchor = base_sensory_thought.clone();
        let mut spatial_map = SemanticPathIntegrator::new();
        let mut active_prediction_error_vector = vec![0.0f32; HDC_DIMENSION];
        let error_gain_factor = 0.25f32;

        print!("syncing temporal tracking matrices... ");
        io::stdout().flush()?;

        let mut response_tokens = Vec::with_capacity(generation_steps);

        for pos in 0..generation_steps {
            if pos > 0 && pos % 4 == 0 {
                let current_context = memory_ring.current_context().clone();
                let macro_slice = macro_thought_anchor.as_slice();
                let ctx_slice = current_context.as_slice();
                let mut updated_macro = vec![0.0f32; HDC_DIMENSION];
                for i in 0..HDC_DIMENSION {
                    updated_macro[i] = (macro_slice[i] * 0.60) + (ctx_slice[i] * 0.40);
                }
                macro_thought_anchor = ContinuousHV::from_slice(&updated_macro);
            }

            let current_drift = spatial_map.distance_from_origin();
            let anchor_pull = (current_drift * 0.10).min(0.35) as f32;

            let mut fused_values = vec![0.0f32; HDC_DIMENSION];
            let base_slice = base_sensory_thought.as_slice();
            let context_slice = memory_ring.current_context().as_slice();
            let macro_slice = macro_thought_anchor.as_slice();

            for i in 0..HDC_DIMENSION {
                fused_values[i] = (base_slice[i] * (sensory_precision + anchor_pull))
                    + (context_slice[i] * (context_precision - anchor_pull).max(0.05))
                    + (macro_slice[i] * macro_influence)
                    + (active_prediction_error_vector[i] * error_gain_factor);
            }
            let effective_thought = ContinuousHV::from_slice(&fused_values);

            let _logits = generator_instance.controller_mut().forward_step(
                &effective_thought,
                active_token_id,
                pos,
            );

            let mut chosen_token_id = 0u32;
            {
                let live_target_hv = generator_instance.controller().output_hv();
                let token_embeddings = generator_instance.controller().token_embeddings();

                let mut candidates: Vec<(u32, f32)> = (0..vocabulary_size)
                    .map(|idx| {
                        let token_str = generator_instance.tokenizer().decode(&[idx]);
                        let mut sim = live_target_hv.similarity(&token_embeddings[idx as usize]);

                        // POSITIVE PROSE WHITELIST: Token must contain ONLY letters or standard spacing
                        let is_pure_prose = !token_str.is_empty()
                            && token_str
                                .chars()
                                .all(|c| c.is_alphabetic() || c.is_whitespace());

                        if !is_pure_prose {
                            sim = -2.0; // Absolute programmatic elimination
                        }
                        (idx, sim)
                    })
                    .collect();

                candidates
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut sampling_pool = candidates[..5usize].to_vec();
                let mut sum_exp = 0.0f32;
                for item in &mut sampling_pool {
                    item.1 = (item.1 / active_temperature).exp();
                    sum_exp += item.1;
                }

                let mut choice_threshold = sample_uniform(&mut rng_seed) * sum_exp;
                chosen_token_id = sampling_pool.last().unwrap().0;
                for item in sampling_pool {
                    choice_threshold -= item.1;
                    if choice_threshold <= 0.0 {
                        chosen_token_id = item.0;
                        break;
                    }
                }
            }

            let (is_valid, coherence, baseline) = guard_rails.verify_emission_path(
                &mut generator_instance,
                &effective_thought,
                chosen_token_id,
            );
            let target_token = if !is_valid {
                (chosen_token_id + 1) % (vocabulary_size as u32)
            } else {
                chosen_token_id
            };

            response_tokens.push(target_token);

            let thought_slice = effective_thought.as_slice();
            {
                let token_embeddings = generator_instance.controller().token_embeddings();
                let chosen_embedding_slice = token_embeddings[target_token as usize].as_slice();
                for i in 0..HDC_DIMENSION {
                    active_prediction_error_vector[i] =
                        chosen_embedding_slice[i] - thought_slice[i];
                }
            }

            spatial_map.compute_step(coherence as f64, (baseline - coherence).abs() as f64);
            let current_output = generator_instance.controller().output_hv().clone();
            memory_ring.push_trajectory(&current_output);
            active_token_id = target_token;
        }

        conversational_turn_count += 1;

        print!("\r\x1b[Kresponse: ");
        let mut words = Vec::new();
        for &t_id in &response_tokens {
            let token_str = generator_instance.tokenizer().decode(&[t_id]);
            let cleaned = token_str.trim().replace('\u{FFFD}', "");
            if !cleaned.is_empty() {
                words.push(cleaned);
            }
        }
        println!("{}", words.join(" "));

        let unique_tokens: std::collections::HashSet<u32> =
            response_tokens.iter().copied().collect();
        let type_token_ratio = if response_tokens.is_empty() {
            0.0f32
        } else {
            unique_tokens.len() as f32 / response_tokens.len() as f32
        };

        println!(
            "  └─ [Telemetry Profile - TTR: {:.2} | Drift: {:.4} | Gate: {:?}]",
            type_token_ratio,
            spatial_map.distance_from_origin(),
            CivicTier::Participant
        );
        println!("");
    }

    Ok(())
}
