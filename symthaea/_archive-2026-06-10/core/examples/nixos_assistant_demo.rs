// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS Conscious Assistant Demo
//!
//! A demonstration of Symthaea's HDC+LTC+Primitives-first architecture:
//!
//! - **ConsciousnessLanguageCore**: HDC+LTC+Primitives understanding
//! - **2D Consciousness Space**: Phi (integration) x Confidence (epistemic certainty)
//! - **Four Quadrants**: Analytical, Intuitive, Emotional, Somatic
//! - **Execution Strategies**: Confident, Curious, Autopilot, Lost
//!
//! ## Architecture
//!
//! ```text
//! User Input
//!     |
//!     v
//! ConsciousnessLanguageCore  <-- HDC+LTC+Primitives (PRIMARY)
//!   - NSM Semantic Primes
//!   - Semantic Frame Matching
//!   - NixOS Intent Recognition
//!     |
//!     v
//!   2D Consciousness Space   <-- Phi x Confidence
//!     |
//!     v
//!   Execution Strategy
//!   - Confident: Execute now
//!   - Curious: Explore first
//!   - Autopilot: Fast execute
//!   - Lost: Ask for help
//! ```
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example nixos_assistant_demo
//! ```

use std::io::{self, Write};
use symthaea::language::{
    ConsciousnessLanguageConfig, ConsciousnessLanguageCore, ConsciousnessQuadrant,
    ExecutionStrategy,
};

fn main() {
    println!("=================================================================");
    println!("  Symthaea: HDC+LTC+Primitives-First NixOS Assistant");
    println!("  Understanding through semantic primitives, not LLM prompts");
    println!("=================================================================\n");

    // Initialize the ConsciousnessLanguageCore
    let config = ConsciousnessLanguageConfig {
        dimension: 512,
        emotional_enabled: true,
        metacognition_enabled: true,
        confidence_threshold: 0.5,
        ..Default::default()
    };

    let mut core = ConsciousnessLanguageCore::new(config);

    println!("2D Consciousness Space:");
    println!("   The system uses TWO independent dimensions:");
    println!("   - Phi (Integration): How deeply unified the processing is");
    println!("   - Confidence: How certain about the interpretation\n");
    println!("   Quadrants:     Analytical | Intuitive | Emotional | Somatic");
    println!("   Strategies:    Confident  | Autopilot | Curious   | Lost\n");
    println!("Commands:");
    println!("   - Type a NixOS question or command request");
    println!("   - 'status' - Show current consciousness state");
    println!("   - 'quit' or 'exit' - Exit the demo\n");

    // Main interaction loop
    loop {
        let phi = core.phi();
        let indicator = if phi >= 0.7 {
            "HIGH"
        } else if phi >= 0.4 {
            "MED"
        } else {
            "LOW"
        };
        print!("\n[Phi={:.2} {}] You: ", phi, indicator);
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" {
            println!("\nThank you for exploring consciousness-aware AI!");
            break;
        }

        if input == "status" {
            println!("\n  Current Phi: {:.3}", core.phi());
            println!("  Consciousness State: {:?}", core.consciousness_state());
            continue;
        }

        // Process through the ConsciousnessLanguageCore
        let result = core.process(input);

        println!("\n===============================================================");

        // Show consciousness quadrant
        let quadrant_desc = match result.quadrant {
            ConsciousnessQuadrant::Analytical => "Analytical (logical/structured processing)",
            ConsciousnessQuadrant::Intuitive => "Intuitive (pattern-matched processing)",
            ConsciousnessQuadrant::Emotional => "Emotional (affective processing)",
            ConsciousnessQuadrant::Somatic => "Somatic (embodied processing)",
        };
        println!("  Quadrant:      {:?} - {}", result.quadrant, quadrant_desc);

        // Show consciousness metrics
        println!("  Phi:           {:.3}", result.consciousness_phi);
        println!("  Confidence:    {:.3}", result.epistemic_confidence);
        println!("  Free Energy:   {:.3}", result.unified_free_energy);
        println!("  State:         {:?}", result.consciousness_state);

        // Show execution strategy
        println!("\n  Execution Strategy:");
        match &result.execution_strategy {
            ExecutionStrategy::Confident {
                execute_immediately,
                explain_reasoning,
                ..
            } => {
                println!("    CONFIDENT (High Phi + High Confidence)");
                println!("    Execute immediately: {}", execute_immediately);
                println!("    Explain reasoning: {}", explain_reasoning);
            }
            ExecutionStrategy::Curious {
                explore_first,
                targeted_questions,
                ..
            } => {
                println!("    CURIOUS (High Phi + Low Confidence)");
                println!("    Explore first (dry-run): {}", explore_first);
                if !targeted_questions.is_empty() {
                    println!("    Targeted questions:");
                    for q in targeted_questions.iter().take(3) {
                        println!("      - {}", q.question);
                    }
                }
            }
            ExecutionStrategy::Autopilot {
                execute_efficiently,
                minimal_response,
                ..
            } => {
                println!("    AUTOPILOT (Low Phi + High Confidence)");
                println!("    Execute efficiently: {}", execute_efficiently);
                println!("    Minimal response: {}", minimal_response);
            }
            ExecutionStrategy::Lost {
                request_help,
                generic_questions,
                ..
            } => {
                println!("    LOST (Low Phi + Low Confidence)");
                println!("    Request help: {}", request_help);
                if !generic_questions.is_empty() {
                    println!("    Questions:");
                    for q in generic_questions.iter().take(3) {
                        println!("      - {}", q.question);
                    }
                }
            }
        }

        // Show NixOS understanding
        let nix = &result.nix_understanding;
        println!("\n  NixOS Understanding:");
        println!("    Intent:      {:?}", nix.intent);
        println!("    Confidence:  {:.1}%", nix.confidence * 100.0);
        if !nix.description.is_empty() {
            println!("    Description: {}", nix.description);
        }
        if !nix.parameters.is_empty() {
            println!("    Parameters:");
            for (key, value) in &nix.parameters {
                println!("      {}: {}", key, value);
            }
        }

        // Show clarifying questions if any
        if !result.clarifying_questions.is_empty() {
            println!("\n  Clarifying Questions:");
            for q in &result.clarifying_questions {
                println!("    - {}", q.question);
            }
        }

        // Show primitive tiers (semantic grounding)
        if !result.primitive_tiers.is_empty() {
            println!("\n  Primitive Tiers: {:?}", result.primitive_tiers);
        }

        println!("===============================================================");
    }
}
