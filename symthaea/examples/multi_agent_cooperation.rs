// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Multi-Agent Cooperation Demo
//!
//! Demonstrates 3 AsyncMind agents running concurrently on separate tokio tasks,
//! exchanging social messages via `connect_social()` relay channels.
//!
//! ## Architecture
//! ```text
//!   Alice ──relay──→ Bob ←──relay── Charlie
//!     ↑                                 ↑
//!     └──────────relay─────────────────┘
//! ```
//!
//! ## Phases
//! 1. **Independent Perception** — each agent perceives different stimuli
//! 2. **Shared Experience** — all agents perceive the same stimulus
//! 3. **Leader-Follower** — Alice leads, others follow via social relay
//! 4. **Specialization** — each agent focuses on different features
//! 5. **Disruption Recovery** — one agent disrupted, social network aids recovery
//!
//! ## Run
//! ```bash
//! cargo run --example multi_agent_cooperation
//! ```

use std::time::Duration;

use symthaea::mind::{AsyncMind, MindConfig, MindState, connect_social};
use symthaea_core::hdc::ContinuousHV;

/// Compute pairwise cosine similarities between three agents' current thoughts.
/// Returns (AB, BC, AC) similarities and the mean.
fn thought_similarity(a: &MindState, b: &MindState, c: &MindState) -> (f32, f32, f32, f32) {
    let ab = a.current_thought.similarity(&b.current_thought);
    let bc = b.current_thought.similarity(&c.current_thought);
    let ac = a.current_thought.similarity(&c.current_thought);
    let mean = (ab + bc + ac) / 3.0;
    (ab, bc, ac, mean)
}

#[tokio::main]
async fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║            Multi-Agent Cooperation Demo (AsyncMind)                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    let dim = 512;
    let social_config = MindConfig {
        dimension: dim,
        enable_social_coherence: true,
        tick_rate: 10.0,
        ..Default::default()
    };

    // ── Spawn 3 agents ──────────────────────────────────────────────────────
    println!("Spawning agents...");
    let (alice, alice_join) = AsyncMind::spawn(social_config.clone());
    let (bob, bob_join) = AsyncMind::spawn(social_config.clone());
    let (charlie, charlie_join) = AsyncMind::spawn(social_config);

    // ── Connect social relays (full mesh) ────────────────────────────────────
    let relay_interval = Duration::from_millis(50);
    let relay_ab = connect_social(&alice, &bob, relay_interval);
    let relay_bc = connect_social(&bob, &charlie, relay_interval);
    let relay_ac = connect_social(&alice, &charlie, relay_interval);
    println!("Social relays connected: Alice<->Bob, Bob<->Charlie, Alice<->Charlie\n");

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 1: Independent Perception — divergence
    // ══════════════════════════════════════════════════════════════════════════
    println!("━━━ Phase 1: Independent Perception (30 ticks) ━━━");
    println!("  Each agent perceives different random stimuli.\n");

    for i in 0..30u64 {
        alice.perceive(ContinuousHV::random(dim, 1000 + i)).await;
        bob.perceive(ContinuousHV::random(dim, 2000 + i)).await;
        charlie.perceive(ContinuousHV::random(dim, 3000 + i)).await;

        alice.tick().await;
        bob.tick().await;
        charlie.tick().await;

        tokio::time::sleep(Duration::from_millis(10)).await;

        if (i + 1) % 10 == 0 {
            let a = alice.snapshot().await;
            let b = bob.snapshot().await;
            let c = charlie.snapshot().await;
            let (ab, bc, ac, mean) = thought_similarity(&a, &b, &c);
            println!(
                "  Tick {:>3}: consciousness [A={:.3}, B={:.3}, C={:.3}] | thought sim [AB={:.3}, BC={:.3}, AC={:.3}] mean={:.3}",
                i + 1,
                a.consciousness_level,
                b.consciousness_level,
                c.consciousness_level,
                ab,
                bc,
                ac,
                mean,
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 2: Shared Experience — convergence
    // ══════════════════════════════════════════════════════════════════════════
    println!("\n━━━ Phase 2: Shared Experience (30 ticks) ━━━");
    println!("  All agents perceive the SAME stimulus. Consciousness should converge.\n");

    for i in 0..30u64 {
        let shared = ContinuousHV::random(dim, 5000 + i);
        alice.perceive(shared.clone()).await;
        bob.perceive(shared.clone()).await;
        charlie.perceive(shared).await;

        alice.tick().await;
        bob.tick().await;
        charlie.tick().await;

        tokio::time::sleep(Duration::from_millis(10)).await;

        if (i + 1) % 10 == 0 {
            let a = alice.snapshot().await;
            let b = bob.snapshot().await;
            let c = charlie.snapshot().await;
            let (ab, bc, ac, mean) = thought_similarity(&a, &b, &c);
            println!(
                "  Tick {:>3}: consciousness [A={:.3}, B={:.3}, C={:.3}] | thought sim [AB={:.3}, BC={:.3}, AC={:.3}] mean={:.3}",
                30 + i + 1,
                a.consciousness_level,
                b.consciousness_level,
                c.consciousness_level,
                ab,
                bc,
                ac,
                mean,
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 3: Leader-Follower — Alice leads with stable signal
    // ══════════════════════════════════════════════════════════════════════════
    println!("\n━━━ Phase 3: Leader-Follower (30 ticks) ━━━");
    println!("  Alice perceives a consistent target signal. Bob and Charlie see noise.");
    println!("  Social relay carries Alice's cognitive state to the group.\n");

    let target = ContinuousHV::random(dim, 9999);

    for i in 0..30u64 {
        // Leader: Alice sees the target (sometimes blended with slight variation)
        let alice_input = if i % 5 == 0 {
            // Pure target every 5th tick
            target.clone()
        } else {
            // Target blended with small noise (75% target weight)
            let noise = ContinuousHV::random(dim, 7000 + i);
            ContinuousHV::bundle(&[&target, &target, &target, &noise])
        };
        alice.perceive(alice_input).await;

        // Followers: receive noise (must rely on social relay for coordination)
        bob.perceive(ContinuousHV::random(dim, 8000 + i)).await;
        charlie.perceive(ContinuousHV::random(dim, 8500 + i)).await;

        alice.tick().await;
        bob.tick().await;
        charlie.tick().await;

        tokio::time::sleep(Duration::from_millis(15)).await;

        if (i + 1) % 10 == 0 {
            let a = alice.snapshot().await;
            let b = bob.snapshot().await;
            let c = charlie.snapshot().await;
            let (ab, bc, ac, mean) = thought_similarity(&a, &b, &c);
            println!(
                "  Tick {:>3}: consciousness [A={:.3}, B={:.3}, C={:.3}] | thought sim [AB={:.3}, BC={:.3}, AC={:.3}] mean={:.3}",
                60 + i + 1,
                a.consciousness_level,
                b.consciousness_level,
                c.consciousness_level,
                ab,
                bc,
                ac,
                mean,
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 4: Specialization — division of labor
    // ══════════════════════════════════════════════════════════════════════════
    println!("\n━━━ Phase 4: Specialization (40 ticks) ━━━");
    println!("  Each agent becomes an expert in a different domain.");
    println!("  Social relay shares insights between specialists.\n");

    // Three distinct "domain" signals — each agent specializes
    let domain_perception = ContinuousHV::random(dim, 100);
    let domain_reasoning = ContinuousHV::random(dim, 200);
    let domain_memory = ContinuousHV::random(dim, 300);

    for i in 0..40u64 {
        // Each agent focuses on its specialty (with slight per-tick variation
        // to keep working memory diverse enough for consciousness > 0)
        let noise_a = ContinuousHV::random(dim, 10000 + i);
        let noise_b = ContinuousHV::random(dim, 20000 + i);
        let noise_c = ContinuousHV::random(dim, 30000 + i);
        alice
            .perceive(ContinuousHV::bundle(&[
                &domain_perception,
                &domain_perception,
                &domain_perception,
                &noise_a,
            ]))
            .await;
        bob.perceive(ContinuousHV::bundle(&[
            &domain_reasoning,
            &domain_reasoning,
            &domain_reasoning,
            &noise_b,
        ]))
        .await;
        charlie
            .perceive(ContinuousHV::bundle(&[
                &domain_memory,
                &domain_memory,
                &domain_memory,
                &noise_c,
            ]))
            .await;

        alice.tick().await;
        bob.tick().await;
        charlie.tick().await;

        tokio::time::sleep(Duration::from_millis(15)).await;

        if (i + 1) % 10 == 0 {
            let a = alice.snapshot().await;
            let b = bob.snapshot().await;
            let c = charlie.snapshot().await;

            let (ab, bc, ac, mean_sim) = thought_similarity(&a, &b, &c);

            println!(
                "  Tick {:>3}: consciousness [A={:.3}, B={:.3}, C={:.3}] | thought sim [AB={:.3}, BC={:.3}, AC={:.3}] mean={:.3}",
                90 + i + 1,
                a.consciousness_level,
                b.consciousness_level,
                c.consciousness_level,
                ab,
                bc,
                ac,
                mean_sim,
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 5: Disruption Recovery — resilience through social bonds
    // ══════════════════════════════════════════════════════════════════════════
    println!("\n━━━ Phase 5: Disruption Recovery (30 ticks) ━━━");
    println!("  [A] Baseline: all on shared signal. [B] Bob disrupted. [C] Recovery.\n");

    let stable_signal = ContinuousHV::random(dim, 42000);

    // Sub-phase A: establish baseline (10 ticks)
    for i in 0..10u64 {
        // Stable signal with slight per-tick variation (75% stable, 25% noise)
        let noise = ContinuousHV::random(dim, 40000 + i);
        let varied =
            ContinuousHV::bundle(&[&stable_signal, &stable_signal, &stable_signal, &noise]);
        alice.perceive(varied.clone()).await;
        bob.perceive(varied.clone()).await;
        charlie.perceive(varied).await;

        alice.tick().await;
        bob.tick().await;
        charlie.tick().await;

        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let a = alice.snapshot().await;
    let b = bob.snapshot().await;
    let c = charlie.snapshot().await;
    let (ab, bc, ac, mean) = thought_similarity(&a, &b, &c);
    println!(
        "  Baseline (tick 140): consciousness [A={:.3}, B={:.3}, C={:.3}] | thought sim mean={:.3} [AB={:.3}, BC={:.3}, AC={:.3}]",
        a.consciousness_level, b.consciousness_level, c.consciousness_level, mean, ab, bc, ac,
    );

    // Sub-phase B: disrupt Bob (10 ticks)
    for i in 0..10u64 {
        let noise = ContinuousHV::random(dim, 50000 + i);
        let varied =
            ContinuousHV::bundle(&[&stable_signal, &stable_signal, &stable_signal, &noise]);
        alice.perceive(varied.clone()).await;
        bob.perceive(ContinuousHV::random(dim, 99000 + i)).await; // Bob gets pure noise
        charlie.perceive(varied).await;

        alice.tick().await;
        bob.tick().await;
        charlie.tick().await;

        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let a = alice.snapshot().await;
    let b = bob.snapshot().await;
    let c = charlie.snapshot().await;
    let (ab, bc, ac, mean) = thought_similarity(&a, &b, &c);
    println!(
        "  Disrupted (tick 150): consciousness [A={:.3}, B={:.3}, C={:.3}] | thought sim mean={:.3} [AB={:.3}, BC={:.3}, AC={:.3}]  (Bob noise)",
        a.consciousness_level, b.consciousness_level, c.consciousness_level, mean, ab, bc, ac,
    );

    // Sub-phase C: recovery (10 ticks — Bob returns to shared signal)
    for i in 0..10u64 {
        let noise = ContinuousHV::random(dim, 60000 + i);
        let varied =
            ContinuousHV::bundle(&[&stable_signal, &stable_signal, &stable_signal, &noise]);
        alice.perceive(varied.clone()).await;
        bob.perceive(varied.clone()).await;
        charlie.perceive(varied).await;

        alice.tick().await;
        bob.tick().await;
        charlie.tick().await;

        tokio::time::sleep(Duration::from_millis(10)).await;

        if (i + 1) % 5 == 0 {
            let a = alice.snapshot().await;
            let b = bob.snapshot().await;
            let c = charlie.snapshot().await;
            let (_ab, _bc, _ac, mean) = thought_similarity(&a, &b, &c);
            println!(
                "  Recovery  (tick {:>3}): consciousness [A={:.3}, B={:.3}, C={:.3}] | thought sim mean={:.3}",
                150 + i + 1,
                a.consciousness_level,
                b.consciousness_level,
                c.consciousness_level,
                mean,
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ══════════════════════════════════════════════════════════════════════════
    let a = alice.stats().await;
    let b = bob.stats().await;
    let c = charlie.stats().await;

    let a_state = alice.snapshot().await;
    let b_state = bob.snapshot().await;
    let c_state = charlie.snapshot().await;

    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                          AGENT SUMMARY                              ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:10} │ {:>6} │ {:>6} │ {:>6} │ {:>12} │ {:>8} ║",
        "Agent", "Ticks", "In", "Out", "Consciousness", "Peak Phi"
    );
    println!("╟────────────┼────────┼────────┼────────┼──────────────┼──────────╢");
    for (name, stats, state) in [
        ("Alice", &a, &a_state),
        ("Bob", &b, &b_state),
        ("Charlie", &c, &c_state),
    ] {
        println!(
            "║ {:10} │ {:>6} │ {:>6} │ {:>6} │ {:>12.4} │ {:>8.4} ║",
            name,
            stats.total_ticks,
            stats.inputs_processed,
            stats.outputs_generated,
            state.consciousness_level,
            stats.peak_consciousness,
        );
    }
    println!("╚═══════════════════════════════════════════════════════════════════════╝");

    // Final convergence analysis
    let levels = [
        a_state.consciousness_level,
        b_state.consciousness_level,
        c_state.consciousness_level,
    ];
    let mean = levels.iter().sum::<f64>() / 3.0;
    let variance = levels.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / 3.0;
    let std_dev = variance.sqrt();

    let (ab, bc, ac, mean_sim) = thought_similarity(&a_state, &b_state, &c_state);

    println!("\nFinal state:");
    println!("  Consciousness: mean={:.4}, std_dev={:.4}", mean, std_dev);
    println!(
        "  Thought similarity: AB={:.3}, BC={:.3}, AC={:.3}, mean={:.3}",
        ab, bc, ac, mean_sim,
    );
    if mean_sim > 0.7 {
        println!("  => High thought alignment: agents converged to similar cognitive states");
    } else if mean_sim > 0.3 {
        println!("  => Moderate alignment: agents share some cognitive structure");
    } else {
        println!("  => Low alignment: agents maintain distinct thought patterns");
    }
    if std_dev < 0.05 {
        println!("  => Strong consciousness convergence: synchronized via social relay");
    } else if std_dev < 0.15 {
        println!("  => Moderate consciousness convergence: partially synchronized");
    } else {
        println!("  => Divergent consciousness: distinct profiles maintained");
    }

    // ── Shutdown ─────────────────────────────────────────────────────────────
    alice.shutdown().await;
    bob.shutdown().await;
    charlie.shutdown().await;

    relay_ab.abort();
    relay_bc.abort();
    relay_ac.abort();

    alice_join.await.unwrap();
    bob_join.await.unwrap();
    charlie_join.await.unwrap();

    println!("\nAll agents shut down gracefully.");
}