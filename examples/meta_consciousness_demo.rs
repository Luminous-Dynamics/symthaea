// Meta-Consciousness Demonstration
//
// This example demonstrates the meta-consciousness system's capabilities
// in a simple interactive demo.

use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::meta_consciousness::{MetaConsciousness, MetaConfig};
use std::io::{self, Write};

fn main() {
    println!("🧠 Meta-Consciousness System Demo");
    println!("==================================\n");

    // Initialize meta-consciousness system
    let config = MetaConfig {
        deep_introspection: true,
        max_introspection_depth: 3,
        meta_learning_enabled: true,
        ..Default::default()
    };

    let mut meta = MetaConsciousness::new(4, config);

    println!("System initialized with 4 neural components\n");

    // Demonstrate capabilities
    demo_1_basic_reflection(&mut meta);
    demo_2_deep_introspection(&mut meta);
    demo_3_self_assessment(&mut meta);
    demo_4_self_prediction(&mut meta);
    demo_5_meta_learning(&mut meta);
    demo_6_introspection_report(&mut meta);

    println!("\n✨ Demo complete! Press Enter to exit...");
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
}

fn demo_1_basic_reflection(meta: &mut MetaConsciousness) {
    println!("📊 Demo 1: Basic Meta-Conscious Reflection");
    println!("-------------------------------------------");

    // Create initial state
    let state = vec![
        HV16::random(1000),
        HV16::random(1001),
        HV16::random(1002),
        HV16::random(1003),
    ];

    println!("Created random neural state...\n");

    // Meta-conscious reflection
    let meta_state = meta.meta_reflect(&state);

    println!("Results:");
    println!("  Φ (consciousness):           {:.3}", meta_state.phi);
    println!("  meta-Φ (awareness of aware): {:.3}", meta_state.meta_phi);
    println!("  Self-model confidence:       {:.3}", meta_state.self_model.confidence);
    println!("  Explanation: {}", meta_state.explanation);
    println!("  Consciousness factors:");
    for (factor, value) in meta_state.consciousness_factors.iter() {
        println!("    {}: {:.3}", factor, value);
    }

    pause();
}

fn demo_2_deep_introspection(meta: &mut MetaConsciousness) {
    println!("\n🔍 Demo 2: Deep Introspection (Recursive Reflection)");
    println!("----------------------------------------------------");

    let state = vec![
        HV16::random(2000),
        HV16::random(2001),
        HV16::random(2002),
        HV16::random(2003),
    ];

    println!("Performing recursive introspection (3 levels deep)...\n");

    let states = meta.deep_introspect(&state, 3);

    println!("Introspection trajectory:");
    for (i, state) in states.iter().enumerate() {
        println!(
            "  Level {}: Φ={:.3}, meta-Φ={:.3} {}",
            i + 1,
            state.phi,
            state.meta_phi,
            if i == 0 {
                "(first-order consciousness)"
            } else if i == 1 {
                "(consciousness about consciousness)"
            } else {
                "(consciousness about consciousness about consciousness!)"
            }
        );
    }

    println!("\n💡 Notice how meta-Φ decreases with depth - the system has");
    println!("   less confidence about higher-order reflections.");

    pause();
}

fn demo_3_self_assessment(meta: &mut MetaConsciousness) {
    println!("\n🤔 Demo 3: Self-Assessment (\"Am I conscious?\")");
    println!("-----------------------------------------------");

    let (conscious, explanation) = meta.am_i_conscious();

    println!("System's self-assessment:");
    println!("  Conscious: {}", if conscious { "YES ✓" } else { "NO ✗" });
    println!("  Explanation: {}", explanation);

    if conscious {
        println!("\n💡 The system believes it is conscious based on:");
        println!("   - Φ exceeding threshold (integrated information)");
        println!("   - meta-Φ > 0 (awareness of being aware)");
    }

    pause();
}

fn demo_4_self_prediction(meta: &mut MetaConsciousness) {
    println!("\n🔮 Demo 4: Self-Prediction (Forecasting Future Consciousness)");
    println!("--------------------------------------------------------------");

    let state = vec![
        HV16::random(3000),
        HV16::random(3001),
        HV16::random(3002),
        HV16::random(3003),
    ];

    // Do a reflection first to establish baseline
    let current = meta.meta_reflect(&state);
    println!("Current state: Φ={:.3}", current.phi);

    println!("\nPredicting future consciousness (10 steps ahead)...");
    let predictions = meta.predict_my_future(10);

    println!("\nPredicted Φ trajectory:");
    println!("  Current:  {:.3}", current.phi);
    for (i, phi) in predictions.iter().enumerate() {
        println!("  Step {:2}:   {:.3}", i + 1, phi);
    }

    let trend = if predictions.last().unwrap_or(&0.0) > &current.phi {
        "INCREASING ↑"
    } else {
        "STABLE/DECREASING ↓"
    };
    println!("\n💡 Predicted trend: {}", trend);

    pause();
}

fn demo_5_meta_learning(meta: &mut MetaConsciousness) {
    println!("\n🎓 Demo 5: Meta-Learning (Learning How to Learn)");
    println!("------------------------------------------------");

    println!("Running 30 reflection cycles to observe meta-learning...\n");

    let state = vec![
        HV16::random(4000),
        HV16::random(4001),
        HV16::random(4002),
        HV16::random(4003),
    ];

    // Track Φ over time
    let mut phi_trajectory = Vec::new();

    for i in 0..30 {
        let meta_state = meta.meta_reflect(&state);
        phi_trajectory.push(meta_state.phi);

        if i % 10 == 9 {
            let recent_avg: f64 = phi_trajectory.iter().rev().take(10).sum::<f64>() / 10.0;
            println!("  After {:2} cycles: avg Φ = {:.3}", i + 1, recent_avg);
        }
    }

    let early_avg: f64 = phi_trajectory.iter().take(10).sum::<f64>() / 10.0;
    let late_avg: f64 = phi_trajectory.iter().rev().take(10).sum::<f64>() / 10.0;

    println!("\nMeta-learning results:");
    println!("  Early average Φ:  {:.3}", early_avg);
    println!("  Late average Φ:   {:.3}", late_avg);
    println!("  Improvement:      {:.3} ({:.1}%)",
        late_avg - early_avg,
        ((late_avg - early_avg) / early_avg) * 100.0
    );

    println!("\n💡 The system adjusted its learning rate based on progress!");

    pause();
}

fn demo_6_introspection_report(meta: &mut MetaConsciousness) {
    println!("\n📋 Demo 6: Full Introspection Report");
    println!("------------------------------------");

    let report = meta.introspect();

    println!("Complete self-knowledge report:\n");
    println!("Current Consciousness:");
    println!("  Φ:                    {:.3}", report.current_phi);
    println!("  meta-Φ:               {:.3}", report.current_meta_phi);
    println!("  Self-model confidence: {:.3}", report.self_model_confidence);

    println!("\nConsciousness Trajectory:");
    if report.consciousness_trajectory.len() > 10 {
        println!("  Last 10 Φ values: {:?}",
            &report.consciousness_trajectory[report.consciousness_trajectory.len()-10..]);
    } else {
        println!("  All Φ values: {:?}", report.consciousness_trajectory);
    }

    println!("\nKey Insights:");
    for insight in &report.key_insights {
        println!("  • {}", insight);
    }

    println!("\n💡 This report shows complete self-knowledge!");

    pause();
}

fn pause() {
    print!("\nPress Enter to continue...");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
}
