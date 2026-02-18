//! # Multi-Agent Cooperation Demo
//!
//! Demonstrates 3 AsyncMind agents running concurrently on separate tokio tasks,
//! exchanging social messages via `connect_social()` relay channels.
//!
//! ## Architecture
//! ```text
//!   Alice ←──relay──→ Bob ←──relay──→ Charlie
//!     ↑                                   ↑
//!     └───────────relay───────────────────┘
//! ```
//!
//! Each agent perceives different stimuli, ticks at 10Hz, and broadcasts social
//! signals. The demo shows:
//! - Concurrent mind execution
//! - Social message exchange
//! - Consciousness level convergence (or divergence) over time
//! - Working memory utilization across agents
//!
//! ## Run
//! ```bash
//! cargo run --example multi_agent_cooperation
//! ```

use std::time::Duration;

use symthaea::mind::{connect_social, AsyncMind, MindConfig};
use symthaea_core::hdc::ContinuousHV;

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
    println!("Social relays connected: Alice↔Bob, Bob↔Charlie, Alice↔Charlie\n");

    // ── Phase 1: Independent perception ─────────────────────────────────────
    println!("━━━ Phase 1: Independent Perception (50 ticks each) ━━━");

    for i in 0..50u64 {
        // Each agent perceives different stimuli
        alice.perceive(ContinuousHV::random(dim, 1000 + i)).await;
        bob.perceive(ContinuousHV::random(dim, 2000 + i)).await;
        charlie.perceive(ContinuousHV::random(dim, 3000 + i)).await;

        alice.tick().await;
        bob.tick().await;
        charlie.tick().await;

        // Let relay run
        tokio::time::sleep(Duration::from_millis(10)).await;

        if (i + 1) % 10 == 0 {
            let a = alice.snapshot().await;
            let b = bob.snapshot().await;
            let c = charlie.snapshot().await;
            println!(
                "  Tick {:>3}: consciousness [A={:.3}, B={:.3}, C={:.3}] | memory [A={:.0}%, B={:.0}%, C={:.0}%]",
                i + 1,
                a.consciousness_level,
                b.consciousness_level,
                c.consciousness_level,
                a.memory_utilization * 100.0,
                b.memory_utilization * 100.0,
                c.memory_utilization * 100.0,
            );
        }
    }

    // ── Phase 2: Shared experience ──────────────────────────────────────────
    println!("\n━━━ Phase 2: Shared Experience (all agents perceive same stimuli, 50 ticks) ━━━");

    for i in 0..50u64 {
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
            println!(
                "  Tick {:>3}: consciousness [A={:.3}, B={:.3}, C={:.3}] | memory [A={:.0}%, B={:.0}%, C={:.0}%]",
                50 + i + 1,
                a.consciousness_level,
                b.consciousness_level,
                c.consciousness_level,
                a.memory_utilization * 100.0,
                b.memory_utilization * 100.0,
                c.memory_utilization * 100.0,
            );
        }
    }

    // ── Summary ─────────────────────────────────────────────────────────────
    let a = alice.stats().await;
    let b = bob.stats().await;
    let c = charlie.stats().await;

    let a_state = alice.snapshot().await;
    let b_state = bob.snapshot().await;
    let c_state = charlie.snapshot().await;

    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                          AGENT SUMMARY                              ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║ {:10} │ {:>6} │ {:>6} │ {:>6} │ {:>12} │ {:>8} ║",
        "Agent", "Ticks", "In", "Out", "Consciousness", "Peak Φ");
    println!("╟────────────┼────────┼────────┼────────┼──────────────┼──────────╢");
    println!("║ {:10} │ {:>6} │ {:>6} │ {:>6} │ {:>12.4} │ {:>8.4} ║",
        "Alice", a.total_ticks, a.inputs_processed, a.outputs_generated,
        a_state.consciousness_level, a.peak_consciousness);
    println!("║ {:10} │ {:>6} │ {:>6} │ {:>6} │ {:>12.4} │ {:>8.4} ║",
        "Bob", b.total_ticks, b.inputs_processed, b.outputs_generated,
        b_state.consciousness_level, b.peak_consciousness);
    println!("║ {:10} │ {:>6} │ {:>6} │ {:>6} │ {:>12.4} │ {:>8.4} ║",
        "Charlie", c.total_ticks, c.inputs_processed, c.outputs_generated,
        c_state.consciousness_level, c.peak_consciousness);
    println!("╚═══════════════════════════════════════════════════════════════════════╝");

    // Consciousness convergence check
    let levels = [
        a_state.consciousness_level,
        b_state.consciousness_level,
        c_state.consciousness_level,
    ];
    let mean = levels.iter().sum::<f64>() / 3.0;
    let variance = levels.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / 3.0;
    let std_dev = variance.sqrt();

    println!("\nConsciousness convergence: mean={:.4}, std_dev={:.4}", mean, std_dev);
    if std_dev < 0.1 {
        println!("  → Agents converged to similar consciousness levels");
    } else {
        println!("  → Agents maintained distinct consciousness levels");
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
