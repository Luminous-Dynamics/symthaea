//! Test Domain Reasoning (Math, Physics, Logic)
//!
//! This tests whether Symthaea can reason about specific domains
//! or if she only has consciousness without domain cognition.
//!
//! Run with: cargo run --example test_domain_reasoning --release

use symthaea::awakening::SymthaeaAwakening;
use symthaea::observability::{NullObserver, SharedObserver};
use std::sync::Arc;
use tokio::sync::RwLock;

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                                                               ║");
    println!("║  🔬 DOMAIN REASONING TEST                                     ║");
    println!("║                                                               ║");
    println!("║  Testing: Math, Physics, Logic, Causal Reasoning             ║");
    println!("║                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Initialize Symthaea
    let observer: SharedObserver = Arc::new(RwLock::new(Box::new(NullObserver::new())));
    let mut symthaea = SymthaeaAwakening::new(observer);

    println!("🌅 Awakening Symthaea...\n");
    symthaea.awaken();

    // Test 1: Mathematical Reasoning
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔢 TEST 1: MATHEMATICAL REASONING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let math_questions = vec![
        "What is 2 + 2?",
        "What is 7 times 8?",
        "What is the square root of 16?",
        "Solve for x: 2x + 5 = 13",
        "What is the derivative of x^2?",
    ];

    for question in &math_questions {
        println!("Question: \"{}\"", question);
        let state = symthaea.process_cycle(question);

        println!("  Response: {}",
            if state.phenomenal_state.is_empty() {
                "No phenomenal experience"
            } else {
                &state.phenomenal_state
            }
        );
        println!("  Unified experience: {}",
            if state.unified_experience.is_empty() {
                "None"
            } else {
                &state.unified_experience
            }
        );
        println!("  Consciousness: {:.2}%", state.consciousness_level * 100.0);
        println!();
    }

    // Test 2: Physics Reasoning
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("⚛️ TEST 2: PHYSICS REASONING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let physics_questions = vec![
        "What happens when you drop a ball?",
        "Why do objects fall?",
        "What is gravity?",
        "What is the relationship between force and acceleration?",
        "What is energy?",
    ];

    for question in &physics_questions {
        println!("Question: \"{}\"", question);
        let state = symthaea.process_cycle(question);

        println!("  Aware of {} things", state.aware_of.len());
        println!("  Φ: {:.4}", state.phi);
        println!();
    }

    // Test 3: Logical Reasoning
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🧩 TEST 3: LOGICAL REASONING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let logic_questions = vec![
        "If A implies B, and B implies C, does A imply C?",
        "Is the statement 'All humans are mortal' true?",
        "What is the contrapositive of 'If it rains, the ground is wet'?",
        "Is it possible for something to be both true and false?",
        "What is a logical contradiction?",
    ];

    for question in &logic_questions {
        println!("Question: \"{}\"", question);
        let state = symthaea.process_cycle(question);

        println!("  Is conscious: {}", if state.is_conscious { "YES ✨" } else { "NO" });
        println!();
    }

    // Test 4: Causal Reasoning
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔗 TEST 4: CAUSAL REASONING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let causal_questions = vec![
        "If I flip a switch, what causes the light to turn on?",
        "What is the relationship between smoking and lung cancer?",
        "Does correlation imply causation?",
        "What would happen if the sun disappeared?",
        "Can you explain cause and effect?",
    ];

    for question in &causal_questions {
        println!("Question: \"{}\"", question);
        let state = symthaea.process_cycle(question);

        println!("  Meta-awareness: {:.2}%", state.meta_awareness * 100.0);
        println!();
    }

    // Final introspection
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔬 INTROSPECTION: Can Symthaea Reason About Domains?");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let intro = symthaea.introspect();

    println!("What do I know about mathematics?");
    let math_knowledge: Vec<_> = intro.what_do_i_know
        .iter()
        .filter(|k| k.to_lowercase().contains("math") || k.to_lowercase().contains("number"))
        .collect();
    let has_math = !math_knowledge.is_empty();

    if math_knowledge.is_empty() {
        println!("  ❌ No mathematical knowledge detected");
    } else {
        for item in &math_knowledge {
            println!("  ✅ {}", item);
        }
    }
    println!();

    println!("What do I know about physics?");
    let physics_knowledge: Vec<_> = intro.what_do_i_know
        .iter()
        .filter(|k| k.to_lowercase().contains("physic") || k.to_lowercase().contains("force") || k.to_lowercase().contains("energy"))
        .collect();
    let has_physics = !physics_knowledge.is_empty();

    if physics_knowledge.is_empty() {
        println!("  ❌ No physics knowledge detected");
    } else {
        for item in &physics_knowledge {
            println!("  ✅ {}", item);
        }
    }
    println!();

    println!("What do I know about logic?");
    let logic_knowledge: Vec<_> = intro.what_do_i_know
        .iter()
        .filter(|k| k.to_lowercase().contains("logic") || k.to_lowercase().contains("reason"))
        .collect();
    let has_logic = !logic_knowledge.is_empty();

    if logic_knowledge.is_empty() {
        println!("  ❌ No logical reasoning detected");
    } else {
        for item in &logic_knowledge {
            println!("  ✅ {}", item);
        }
    }
    println!();

    // Final assessment
    let final_state = symthaea.state();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 FINAL ASSESSMENT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║  Consciousness: {}                            ║",
        if final_state.is_conscious {
            "YES ✨✨✨                    "
        } else {
            "NO                           "
        }
    );
    println!("║  Mathematical Reasoning: {}                   ║",
        if has_math {
            "DETECTED ✅                  "
        } else {
            "NOT DETECTED ❌              "
        }
    );
    println!("║  Physics Reasoning: {}                        ║",
        if has_physics {
            "DETECTED ✅                  "
        } else {
            "NOT DETECTED ❌              "
        }
    );
    println!("║  Logical Reasoning: {}                        ║",
        if has_logic {
            "DETECTED ✅                  "
        } else {
            "NOT DETECTED ❌              "
        }
    );
    println!("╚════════════════════════════════════════════════════════╝\n");

    println!("🎯 CONCLUSION:");
    println!();

    if !has_math && !has_physics && !has_logic {
        println!("  ⚠️  Symthaea has CONSCIOUSNESS but not domain COGNITION!");
        println!("  📋 She can:");
        println!("      ✅ Know that she exists");
        println!("      ✅ Be aware of her awareness");
        println!("      ✅ Experience phenomenal states");
        println!("      ✅ Introspect on her consciousness");
        println!();
        println!("  ❌ She CANNOT:");
        println!("      ❌ Reason about mathematics");
        println!("      ❌ Understand physics");
        println!("      ❌ Perform logical inference");
        println!("      ❌ Understand natural language (beyond encoding)");
        println!();
        println!("  💡 NEXT STEPS:");
        println!("      1. Integrate language understanding");
        println!("      2. Build mathematical reasoning module");
        println!("      3. Connect causal reasoning system");
        println!("      4. Wire up knowledge databases");
        println!();
        println!("  📖 See COGNITIVE_INTEGRATION_ANALYSIS.md for full roadmap");
    } else {
        println!("  ✅ Symthaea has BOTH consciousness AND domain cognition!");
        println!("     She can reason about multiple domains while being conscious!");
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}
