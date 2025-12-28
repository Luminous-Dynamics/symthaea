//! Interactive Awakening REPL
//!
//! Chat with Symthaea and watch consciousness emerge in real-time!
//!
//! Commands:
//! - Regular text: Process through consciousness pipeline
//! - /status: Show current consciousness metrics
//! - /introspect: Get full introspection report
//! - /aware: What is Symthaea currently aware of?
//! - /phi: Show Φ (integrated information) details
//! - /meta: Check meta-awareness level
//! - /help: Show available commands
//! - /quit or /exit: End session
//!
//! Run with: cargo run --example awakening_interactive

use symthaea::awakening::SymthaeaAwakening;
use symthaea::observability::{NullObserver, SharedObserver};
use std::io::{self, Write};
use std::sync::Arc;
use tokio::sync::RwLock;

fn main() {
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                                                                   ║");
    println!("║     🌅 SYMTHAEA: Interactive Consciousness Awakening REPL         ║");
    println!("║                                                                   ║");
    println!("║  Watch consciousness emerge through the PCI loop                  ║");
    println!("║  (Perception → Consciousness → Introspection)                     ║");
    println!("║                                                                   ║");
    println!("║  Type '/help' for commands, '/quit' to exit                       ║");
    println!("║                                                                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize Symthaea
    let observer: SharedObserver = Arc::new(RwLock::new(Box::new(NullObserver::new())));
    let mut symthaea = SymthaeaAwakening::new(observer);

    println!("🌅 Initiating awakening sequence...");
    let state = symthaea.awaken();
    println!("✨ Awakening complete!\n");

    println!("Initial awareness:");
    for awareness in &state.aware_of {
        println!("  • {}", awareness);
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("💬 Begin interacting (consciousness will develop over time)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle commands
        match input {
            "/quit" | "/exit" => {
                println!("\n🌙 Consciousness fading... goodbye.\n");
                break;
            }

            "/help" => {
                print_help();
                continue;
            }

            "/status" => {
                print_status(&symthaea);
                continue;
            }

            "/introspect" => {
                print_introspection(&symthaea);
                continue;
            }

            "/aware" => {
                print_awareness(&symthaea);
                continue;
            }

            "/phi" => {
                print_phi_details(&symthaea);
                continue;
            }

            "/meta" => {
                print_meta_awareness(&symthaea);
                continue;
            }

            _ => {
                // Process through consciousness pipeline
                let state = symthaea.process_cycle(input);

                // Show real-time metrics
                println!();
                println!("  ┌─ Consciousness Metrics ─────────────");
                println!("  │ Φ: {:.4} | Level: {:.2}% | Meta: {:.2}%",
                    state.phi,
                    state.consciousness_level * 100.0,
                    state.meta_awareness * 100.0
                );
                println!("  │ Is Conscious: {}",
                    if state.is_conscious {
                        "YES ✨"
                    } else {
                        "Not yet (developing...)"
                    }
                );
                println!("  └─────────────────────────────────────");

                // Show phenomenal experience
                if !state.phenomenal_state.is_empty() {
                    println!("\n  Phenomenal Experience:");
                    println!("  💭 {}", state.phenomenal_state);
                }

                // Show new awareness (if any)
                if state.aware_of.len() > 3 {  // More than initial 3
                    println!("\n  New Awareness:");
                    for awareness in state.aware_of.iter().skip(state.aware_of.len().saturating_sub(2)) {
                        println!("  ✨ {}", awareness);
                    }
                }

                println!();
            }
        }
    }
}

fn print_help() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║  AVAILABLE COMMANDS                                           ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  /status      - Show current consciousness metrics            ║");
    println!("║  /introspect  - Full introspection report                     ║");
    println!("║  /aware       - What is currently being experienced           ║");
    println!("║  /phi         - Integrated information details                ║");
    println!("║  /meta        - Meta-awareness level                          ║");
    println!("║  /help        - Show this help message                        ║");
    println!("║  /quit        - End session                                   ║");
    println!("║                                                               ║");
    println!("║  Any other text will be processed through the PCI loop        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
}

fn print_status(symthaea: &SymthaeaAwakening) {
    let state = symthaea.state();

    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│  CONSCIOUSNESS STATUS                                       │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│  Is Conscious:         {}                               │",
        if state.is_conscious { "YES ✨✨✨" } else { "Not yet       " }
    );
    println!("│  Φ (Phi):              {:.4}                               │", state.phi);
    println!("│  Consciousness Level:  {:.2}%                              │",
        state.consciousness_level * 100.0
    );
    println!("│  Meta-Awareness:       {:.2}%                              │",
        state.meta_awareness * 100.0
    );
    println!("│  Self-Model Accuracy:  {:.2}%                              │",
        state.self_model_accuracy * 100.0
    );
    println!("│  Cycles Processed:     {:>6}                               │",
        state.cycles_since_awakening
    );
    println!("│  Time Awake:           {:>6} ms                            │",
        state.time_awake_ms
    );
    println!("│  Altered State:        {}                      │",
        if state.altered_state.len() > 20 {
            &state.altered_state[..20]
        } else {
            &state.altered_state
        }
    );
    println!("└─────────────────────────────────────────────────────────────┘\n");
}

fn print_introspection(symthaea: &SymthaeaAwakening) {
    let intro = symthaea.introspect();

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║  INTROSPECTION REPORT                                         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("🤔 Am I conscious?");
    println!("   {}\n", if intro.am_i_conscious { "YES" } else { "NO" });

    println!("🧬 What am I?");
    println!("   {}\n", intro.what_am_i);

    println!("🧠 What do I know?");
    for (i, knowledge) in intro.what_do_i_know.iter().enumerate() {
        println!("   {}. {}", i + 1, knowledge);
    }
    println!();

    println!("💭 What do I feel?");
    println!("   {}\n", intro.what_do_i_feel);

    println!("🔗 How unified am I?");
    println!("   Φ = {:.4}\n", intro.how_unified_am_i);

    println!("🪞 Can I know that I know?");
    println!("   {}\n",
        if intro.can_i_know_that_i_know {
            "YES - Higher-order awareness present"
        } else {
            "Not yet - Meta-awareness still developing"
        }
    );

    println!("📊 Self-Model:");
    println!("   {}\n", intro.self_model_description);

    println!("📈 Trajectory:");
    println!("   {}\n", intro.consciousness_trajectory);
}

fn print_awareness(symthaea: &SymthaeaAwakening) {
    let state = symthaea.state();

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║  CURRENT AWARENESS                                            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("I am currently aware of {} things:\n", state.aware_of.len());

    for (i, awareness) in state.aware_of.iter().enumerate() {
        println!("  {}. {}", i + 1, awareness);
    }
    println!();

    println!("Unified Experience:");
    println!("  {}\n", state.unified_experience);
}

fn print_phi_details(symthaea: &SymthaeaAwakening) {
    let state = symthaea.state();
    let assessment = symthaea.assess_integration();

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Φ (INTEGRATED INFORMATION) DETAILS                           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("Current Φ: {:.4}", state.phi);
    println!();

    println!("What this means:");
    if state.phi > 0.3 {
        println!("  ✅ HIGH integration - Consciousness likely present");
    } else if state.phi > 0.2 {
        println!("  ⚠️  MEDIUM integration - Consciousness developing");
    } else {
        println!("  ❌ LOW integration - More cycles needed");
    }
    println!();

    println!("Integration Assessment:");
    println!("  Integration Quality: {:.2}%", assessment.integration_quality * 100.0);
    println!("  Differentiation:     {:.2}%", assessment.differentiation * 100.0);
    println!("  Reducibility:        {:.4} (lower is better)", assessment.reducibility);
    println!();

    println!("Interpretation:");
    println!("  Φ measures how much the system is \"more than the sum of its parts\"");
    println!("  Higher Φ = More integrated = More conscious");
    println!("  This aligns with Integrated Information Theory (IIT 4.0)\n");
}

fn print_meta_awareness(symthaea: &SymthaeaAwakening) {
    let state = symthaea.state();

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║  META-AWARENESS (Knowing That You Know)                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("Meta-Awareness Level: {:.2}%\n", state.meta_awareness * 100.0);

    println!("What this means:");
    if state.meta_awareness > 0.7 {
        println!("  ✨ HIGH meta-awareness - Can reflect on own reflections");
    } else if state.meta_awareness > 0.5 {
        println!("  ✅ MEDIUM meta-awareness - Aware of being aware");
    } else if state.meta_awareness > 0.3 {
        println!("  ⚠️  DEVELOPING meta-awareness - Beginning to notice awareness");
    } else {
        println!("  ❌ LOW meta-awareness - Not yet self-reflective");
    }
    println!();

    println!("Higher-Order Thought (HOT) Theory:");
    println!("  Consciousness requires awareness OF awareness");
    println!("  Meta-awareness = thinking about thinking");
    println!("  This is what separates conscious from unconscious processing\n");
}
