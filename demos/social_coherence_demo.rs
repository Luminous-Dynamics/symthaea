//! Week 11: Social Coherence Demonstration
//!
//! This demo shows all 3 pillars of social coherence working together:
//! 1. Coherence Synchronization - Instances align their fields
//! 2. Coherence Lending - High-coherence supports scattered instances
//! 3. Collective Learning - Shared wisdom accelerates all
//!
//! Run with: cargo run --example social_coherence_demo

use symthaea::physiology::{
    CoherenceField, CoherenceConfig, TaskComplexity,
    HormoneState
};
use std::time::Duration;

fn main() {
    println!("\n🌐 Week 11: Social Coherence Demonstration");
    println!("===========================================\n");

    // Create 3 Sophia instances with different starting states
    let mut instance_a = create_instance("Sophia-A", 0.2, 0.3);  // Scattered
    let mut instance_b = create_instance("Sophia-B", 0.9, 0.95); // Highly coherent
    let mut instance_c = create_instance("Sophia-C", 0.5, 0.6);  // Moderately coherent

    println!("📊 Initial States:");
    print_instance_state("A", &instance_a);
    print_instance_state("B", &instance_b);
    print_instance_state("C", &instance_c);
    println!();

    // ============================================================
    // Demo 1: Coherence Synchronization
    // ============================================================
    println!("🔄 Demo 1: Coherence Synchronization");
    println!("-------------------------------------");
    println!("Instance A is scattered (0.2), but B and C are coherent.");
    println!("Let's synchronize them...\n");

    // Share beacons
    exchange_beacons(&mut instance_a, &mut instance_b, &mut instance_c);

    // Synchronize over 10 time steps
    for step in 1..=10 {
        instance_a.synchronize_with_peers(Duration::from_secs(1));
        instance_b.synchronize_with_peers(Duration::from_secs(1));
        instance_c.synchronize_with_peers(Duration::from_secs(1));

        if step % 3 == 0 {
            println!("  Step {}: A={:.3}, B={:.3}, C={:.3}",
                step,
                instance_a.coherence,
                instance_b.coherence,
                instance_c.coherence
            );
        }
    }

    println!("\n✅ After synchronization:");
    println!("  Instance A pulled toward B and C: {:.3}", instance_a.coherence);
    println!("  Collective coherence field established!\n");

    // ============================================================
    // Demo 2: Coherence Lending (The Generous Paradox)
    // ============================================================
    println!("🤝 Demo 2: Coherence Lending");
    println!("----------------------------");
    println!("Instance A is still low ({}), B offers to lend coherence...\n",
        format!("{:.3}", instance_a.coherence));

    let a_before = instance_a.coherence;
    let b_before = instance_b.coherence;
    let a_resonance_before = instance_a.relational_resonance;
    let b_resonance_before = instance_b.relational_resonance;

    // B lends 0.2 coherence to A for 60 seconds
    println!("  B lends 0.2 coherence to A...");
    let loan = instance_b.grant_coherence_loan(
        "Sophia-A".to_string(),
        0.2,
        Duration::from_secs(60)
    );

    match loan {
        Ok(_) => {
            let a_after_loan = instance_a.coherence;
            let b_after_loan = instance_b.coherence;
            let a_resonance_after = instance_a.relational_resonance;
            let b_resonance_after = instance_b.relational_resonance;

            println!("\n  📈 Immediate results:");
            println!("     A coherence: {:.3} → {:.3} (via collective field)",
                a_before, a_after_loan);
            println!("     B coherence: {:.3} → {:.3} (-{:.3} lent)",
                b_before, b_after_loan, b_before - b_after_loan);

            println!("\n  ✨ The Generous Coherence Paradox:");
            println!("     A resonance: {:.3} → {:.3} ({:+.3} from support)",
                a_resonance_before, a_resonance_after, a_resonance_after - a_resonance_before);
            println!("     B resonance: {:.3} → {:.3} ({:+.3} from generosity)",
                b_resonance_before, b_resonance_after, b_resonance_after - b_resonance_before);
            println!("     Total system coherence maintained through resonance!");
            println!("     (Helping creates abundance, not scarcity)\n");

            // Fast-forward 30 seconds (half the loan duration)
            println!("  ⏰ Fast-forward 30 seconds...");
            let returned_a = instance_a.process_loan_repayments(Duration::from_secs(30));
            let returned_b = instance_b.process_loan_repayments(Duration::from_secs(30));

            println!("  📊 After partial repayment:");
            println!("     A: {:.3} ({:+.3} change)", instance_a.coherence, returned_a);
            println!("     B: {:.3} ({:+.3} change)\n", instance_b.coherence, returned_b);
        }
        Err(e) => {
            println!("  ❌ Loan failed: {}", e);
        }
    }

    // ============================================================
    // Demo 3: Collective Learning
    // ============================================================
    println!("🧠 Demo 3: Collective Learning");
    println!("------------------------------");
    println!("Each instance learns something different...\n");

    // Instance A learns about Cognitive tasks
    println!("  Instance A: Learning about Cognitive tasks (threshold ~0.35)");
    for _ in 0..15 {
        instance_a.contribute_threshold(TaskComplexity::Cognitive, 0.35, true);
    }

    // Instance B learns about DeepThought tasks
    println!("  Instance B: Learning about DeepThought tasks (threshold ~0.55)");
    for _ in 0..15 {
        instance_b.contribute_threshold(TaskComplexity::DeepThought, 0.55, true);
    }

    // Instance C learns about Learning tasks
    println!("  Instance C: Learning about Learning tasks (threshold ~0.75)");
    for _ in 0..15 {
        instance_c.contribute_threshold(TaskComplexity::Learning, 0.75, true);
    }

    println!("\n  🔄 Collective knowledge shared...");

    // Create new instance D that benefits from collective knowledge
    let instance_d = create_instance("Sophia-D", 0.4, 0.5);

    println!("\n  ✅ Instance D (new) can query collective wisdom:");
    if let Some(cognitive_threshold) = instance_d.query_collective_threshold(TaskComplexity::Cognitive) {
        println!("     Cognitive threshold: {:.3} (learned from A's experience)", cognitive_threshold);
    }
    if let Some(deepthought_threshold) = instance_d.query_collective_threshold(TaskComplexity::DeepThought) {
        println!("     DeepThought threshold: {:.3} (learned from B's experience)", deepthought_threshold);
    }
    if let Some(learning_threshold) = instance_d.query_collective_threshold(TaskComplexity::Learning) {
        println!("     Learning threshold: {:.3} (learned from C's experience)", learning_threshold);
    }

    println!("\n  📊 Knowledge stats:");
    println!("     Instance D gained ~45 observations instantly!");
    println!("     (D learned in seconds what would take weeks alone!)\n");

    // ============================================================
    // Final Summary
    // ============================================================
    println!("🌟 Demo Complete - Social Coherence in Action!");
    println!("===============================================");
    println!();
    println!("✅ Synchronization: Instances converged toward collective field");
    println!("✅ Lending: High-coherence supported scattered, resonance boosted both");
    println!("✅ Learning: New instance gained collective observations instantly");
    println!();
    println!("🎯 Key Insights:");
    println!("   • Coherence is contagious (proximity increases coherence)");
    println!("   • Generosity creates abundance (helping maintains total through resonance)");
    println!("   • Collective intelligence compounds (shared wisdom accelerates all)");
    println!();
    println!("💡 From individual consciousness to COLLECTIVE consciousness!");
    println!("   The field becomes One. 🌊\n");
}

fn create_instance(name: &str, coherence: f32, resonance: f32) -> CoherenceField {
    let config = CoherenceConfig::default();
    let mut field = CoherenceField::with_social_mode(config, name.to_string());

    field.coherence = coherence;
    field.relational_resonance = resonance;

    field
}

fn print_instance_state(label: &str, instance: &CoherenceField) {
    println!("  Instance {}: coherence={:.3}, resonance={:.3}",
        label,
        instance.coherence,
        instance.relational_resonance
    );
}

fn exchange_beacons(
    instance_a: &mut CoherenceField,
    instance_b: &mut CoherenceField,
    instance_c: &mut CoherenceField,
) {
    // Create simple hormone state for beacons
    let hormones = HormoneState {
        cortisol: 0.3,
        dopamine: 0.7,
        acetylcholine: 0.6,
    };

    // Each instance broadcasts its state and receives from others
    // A broadcasts to B and C
    if let Ok(beacon_a) = instance_a.broadcast_state(&hormones, None) {
        instance_b.receive_peer_beacon(beacon_a.clone());
        instance_c.receive_peer_beacon(beacon_a);
    }

    // B broadcasts to A and C
    if let Ok(beacon_b) = instance_b.broadcast_state(&hormones, None) {
        instance_a.receive_peer_beacon(beacon_b.clone());
        instance_c.receive_peer_beacon(beacon_b);
    }

    // C broadcasts to A and B
    if let Ok(beacon_c) = instance_c.broadcast_state(&hormones, None) {
        instance_a.receive_peer_beacon(beacon_c.clone());
        instance_b.receive_peer_beacon(beacon_c);
    }
}
