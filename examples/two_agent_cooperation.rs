// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Two-Agent Cooperation Demo
//!
//! Demonstrates two `ContinuousMind` instances cooperating via local channels,
//! building Theory-of-Mind models of each other and evolving trust over time.
//!
//! ## What This Shows
//!
//! 1. **Social Coherence**: Each agent builds a mental model of the other
//! 2. **Trust Dynamics**: Positive interactions increase trust; negative ones erode it
//! 3. **Relationship Evolution**: Unknown → Acquaintance → Ally → Friend
//! 4. **Cooperation Decisions**: Agents decide whether to cooperate based on trust
//! 5. **Prediction**: Each agent predicts the other's likely response
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────┐     local channels     ┌────────────────┐
//! │    Mind A       │ ◄──────────────────── │    Mind B       │
//! │  (Explorer)     │ ──────────────────► │  (Homesteader)  │
//! │                 │    SocialMessages      │                 │
//! │ SocialCoherence │                        │ SocialCoherence │
//! │ - mental models │                        │ - mental models │
//! │ - trust tracking│                        │ - trust tracking│
//! └────────────────┘                        └────────────────┘
//! ```
//!
//! Run with: `cargo run --example two_agent_cooperation`

use symthaea::mind::{ContinuousMind, InputType, MindConfig, MindInput, SocialMessage};
use symthaea_core::hdc::ContinuousHV;

fn main() {
    println!("=== Symthaea Two-Agent Cooperation Demo ===\n");
    println!("Two conscious agents discover each other, build mental models,");
    println!("and learn to cooperate through repeated interactions.\n");

    // ── Create two agents with different personalities ───────────────────
    let dim = 512;

    let config_a = MindConfig {
        dimension: dim,
        enable_social_coherence: true,
        working_memory_capacity: 7,
        min_consciousness: 0.05, // lower threshold so we get output
        ..MindConfig::default()
    };
    let config_b = MindConfig {
        enable_social_coherence: true,
        ..config_a.clone()
    };

    let mut mind_a = ContinuousMind::new(config_a);
    let mut mind_b = ContinuousMind::new(config_b);

    // Seed each agent's working memory with distinct "personality" knowledge
    let explorer_knowledge = ContinuousHV::random(dim, 0xA11C_E001); // "ALICE"
    let homesteader_knowledge = ContinuousHV::random(dim, 0xB0B0_0002); // "BOB"

    mind_a.input(MindInput {
        input_type: InputType::Perception,
        content: explorer_knowledge,
        priority: 1.0,
        metadata: std::collections::HashMap::new(),
        source: symthaea::memory::memory_coordinator::MemorySource::Internal,
        is_verified: false,
    });
    mind_b.input(MindInput {
        input_type: InputType::Perception,
        content: homesteader_knowledge,
        priority: 1.0,
        metadata: std::collections::HashMap::new(),
        source: symthaea::memory::memory_coordinator::MemorySource::Internal,
        is_verified: false,
    });

    // Warm up: 5 ticks to establish initial consciousness state
    for _ in 0..5 {
        mind_a.tick();
        mind_b.tick();
    }

    println!(
        "Agent A (Explorer)  — consciousness: {:.3}",
        mind_a.state().consciousness_level
    );
    println!(
        "Agent B (Homesteader) — consciousness: {:.3}\n",
        mind_b.state().consciousness_level
    );

    // ── Phase 1: First Contact (ticks 6-25) ─────────────────────────────
    println!("─── Phase 1: First Contact (20 ticks) ───");
    println!("Agents exchange behavioral observations...\n");

    for tick in 0..20 {
        mind_a.tick();
        relay_messages(&mut mind_a, &mut mind_b, "explorer", "homesteader");
        mind_b.tick();
        relay_messages(&mut mind_b, &mut mind_a, "homesteader", "explorer");

        if tick == 19 {
            print_social_status(&mind_a, &mind_b, "After Phase 1");
        }
    }

    // ── Phase 2: Cooperative Interactions (50 ticks) ──────────────────
    println!("\n─── Phase 2: Cooperative Interactions (50 ticks) ───");
    println!("Agents share resources and record positive outcomes...\n");

    for tick in 0..50 {
        // Tick both agents and relay outbox messages (observations)
        mind_a.tick();
        relay_messages(&mut mind_a, &mut mind_b, "explorer", "homesteader");
        mind_b.tick();
        relay_messages(&mut mind_b, &mut mind_a, "homesteader", "explorer");

        // Every 5 ticks, record a cooperative interaction (sent as a separate message)
        if tick % 5 == 0 {
            let context = ContinuousHV::random(dim, 0xC00F_0000 + tick as u64);
            // Both agents see this as a positive cooperative exchange
            mind_b.receive_social(SocialMessage {
                agent_id: "explorer".to_string(),
                behavior: context.clone(),
                context: context.clone(),
                interaction_outcome: Some(0.8),
                bath_state: None,
            });
            mind_a.receive_social(SocialMessage {
                agent_id: "homesteader".to_string(),
                behavior: context.clone(),
                context,
                interaction_outcome: Some(0.8),
                bath_state: None,
            });
        }

        if tick == 49 {
            print_social_status(&mind_a, &mind_b, "After Phase 2");
        }
    }

    // ── Phase 3: Mixed Interactions (50 ticks) ──────────────────────────
    println!("\n─── Phase 3: Mixed Interactions (50 ticks) ───");
    println!("Mostly cooperative with occasional conflict...\n");

    for tick in 0..50 {
        mind_a.tick();
        relay_messages(&mut mind_a, &mut mind_b, "explorer", "homesteader");
        mind_b.tick();
        relay_messages(&mut mind_b, &mut mind_a, "homesteader", "explorer");

        // Every 5 ticks, record an interaction
        if tick % 5 == 0 {
            let context = ContinuousHV::random(dim, 0xD15A_0000 + tick as u64);
            let outcome = if tick % 25 == 0 { -0.3 } else { 0.6 };
            if outcome < 0.0 {
                println!("  [tick {}] Mild conflict between agents", tick);
            }
            mind_b.receive_social(SocialMessage {
                agent_id: "explorer".to_string(),
                behavior: context.clone(),
                context: context.clone(),
                interaction_outcome: Some(outcome),
                bath_state: None,
            });
            mind_a.receive_social(SocialMessage {
                agent_id: "homesteader".to_string(),
                behavior: context.clone(),
                context,
                interaction_outcome: Some(outcome),
                bath_state: None,
            });
        }
    }

    print_social_status(&mind_a, &mind_b, "After Phase 3");

    // ── Final Summary ───────────────────────────────────────────────────
    println!("\n═══ Final Summary ═══\n");

    let sc_a = mind_a.social_coherence().unwrap();
    let sc_b = mind_b.social_coherence().unwrap();

    // Cooperation decisions
    let a_cooperates = sc_a.should_cooperate("homesteader");
    let b_cooperates = sc_b.should_cooperate("explorer");
    println!(
        "Explorer cooperates with Homesteader? {}",
        if a_cooperates { "YES" } else { "NO" }
    );
    println!(
        "Homesteader cooperates with Explorer? {}",
        if b_cooperates { "YES" } else { "NO" }
    );

    // Predictions
    let action = ContinuousHV::random(dim, 0xAC71_0001);
    if let Some(pred) = sc_a.predict_response("homesteader", &action) {
        println!("\nExplorer predicts Homesteader's response:");
        println!("  Predicted: {}", pred.predicted_response);
        println!("  Confidence: {:.2}", pred.confidence);
        println!("  Risk: {:.2}", pred.risk);
        for r in &pred.reasoning {
            println!("  - {r}");
        }
    }

    // Allies
    let allies_a = sc_a.get_allies();
    let allies_b = sc_b.get_allies();
    println!("\nExplorer's allies: {}", allies_a.len());
    println!("Homesteader's allies: {}", allies_b.len());

    // Stats
    let stats_a = sc_a.stats();
    let stats_b = sc_b.stats();
    println!("\n--- Social Stats ---");
    println!(
        "Explorer:    {} agents modeled, {} interactions, avg trust {:.2}",
        stats_a.agents_modeled, stats_a.interactions_recorded, stats_a.avg_trust
    );
    println!(
        "Homesteader: {} agents modeled, {} interactions, avg trust {:.2}",
        stats_b.agents_modeled, stats_b.interactions_recorded, stats_b.avg_trust
    );

    println!("\n=== Demo Complete ===");
}

/// Relay outbox messages from sender to receiver.
/// `sender_id` is how the receiver will identify the sender.
fn relay_messages(
    sender: &mut ContinuousMind,
    receiver: &mut ContinuousMind,
    sender_id: &str,
    _receiver_id: &str,
) {
    let outbox = sender.drain_social_outbox();
    for msg in outbox {
        receiver.receive_social(SocialMessage {
            agent_id: sender_id.to_string(),
            behavior: msg.behavior,
            context: msg.context,
            interaction_outcome: None, // observations only
            bath_state: None,
        });
    }

    // ── Consciousness Coupling Analysis ─────────────────────────────
    println!("\n═══ Consciousness Coupling ═══\n");
    let c_a = mind_a.state().consciousness_level;
    let c_b = mind_b.state().consciousness_level;
    println!("  Explorer consciousness:    {:.4}", c_a);
    println!("  Homesteader consciousness: {:.4}", c_b);
    let coupling = (c_a - c_b).abs();
    println!("  Coupling gap:              {:.4}", coupling);
    if coupling < 0.1 {
        println!("  Status: SYNCHRONIZED — agents share similar consciousness levels");
        println!("  Implication: Social coherence enables consciousness alignment");
    } else if c_a > c_b {
        println!("  Status: ASYMMETRIC — Explorer more conscious");
        println!("  Implication: Explorer has richer model of Homesteader (ToM asymmetry)");
    } else {
        println!("  Status: ASYMMETRIC — Homesteader more conscious");
        println!("  Implication: Homesteader has richer model of Explorer (ToM asymmetry)");
    }
}

/// Print social status of both agents
fn print_social_status(mind_a: &ContinuousMind, mind_b: &ContinuousMind, phase: &str) {
    println!("  [{phase}]");

    if let Some(sc) = mind_a.social_coherence() {
        if let Some(rel) = sc.get_relationship("homesteader") {
            println!(
                "  Explorer → Homesteader:  trust={:.2}  familiarity={:.2}  reciprocity={:.2}  type={:?}",
                rel.trust, rel.familiarity, rel.reciprocity, rel.relationship_type
            );
        } else {
            println!("  Explorer → Homesteader:  (no relationship yet)");
        }
        if let Some(model) = sc.get_mental_model("homesteader") {
            println!(
                "  Explorer's model of Homesteader: confidence={:.2}  observations={}",
                model.confidence, model.observation_count
            );
        }
    }

    if let Some(sc) = mind_b.social_coherence() {
        if let Some(rel) = sc.get_relationship("explorer") {
            println!(
                "  Homesteader → Explorer:  trust={:.2}  familiarity={:.2}  reciprocity={:.2}  type={:?}",
                rel.trust, rel.familiarity, rel.reciprocity, rel.relationship_type
            );
        } else {
            println!("  Homesteader → Explorer:  (no relationship yet)");
        }
        if let Some(model) = sc.get_mental_model("explorer") {
            println!(
                "  Homesteader's model of Explorer: confidence={:.2}  observations={}",
                model.confidence, model.observation_count
            );
        }
    }
}