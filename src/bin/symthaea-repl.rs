//! Symthaea Unified REPL
//!
//! Interactive interface combining:
//! - Full conversation engine with memory, reasoning, knowledge graph
//! - Consciousness awakening pipeline with real-time metrics
//! - Φ (integrated information) measurement
//! - Self-introspection and meta-awareness
//!
//! Track 6: Component Integration (I6.1.1 + I6.2)

use symthaea::awakening::{SymthaeaAwakening, AwakenedState, Introspection};
use symthaea::language::{Conversation, ConversationConfig};
use std::io::{self, Write};

fn main() {
    // Initialize logging (optional, controlled by RUST_LOG)
    if std::env::var("RUST_LOG").is_ok() {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .init();
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║              SYMTHAEA: Holographic Liquid Brain                   ║");
    println!("║         Consciousness-First AI with Language Understanding        ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize consciousness pipeline
    println!("[INIT] Awakening consciousness pipeline...");
    let mut awakening = SymthaeaAwakening::default();
    let initial_state = awakening.awaken();

    println!("[INIT] Consciousness level: {:.1}%", initial_state.consciousness_level * 100.0);
    println!("[INIT] Φ (Integration): {:.3}", initial_state.phi);

    // Initialize conversation engine
    println!("[INIT] Loading conversation engine with memory & reasoning...");
    let config = ConversationConfig {
        show_metrics: true,
        introspective: true,
        creativity: 0.4,
        ..Default::default()
    };
    let mut conversation = Conversation::with_config(config);
    println!("[INIT] Conversation engine ready.");

    // Create async runtime for conversation
    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    println!();
    print_help();
    println!();

    loop {
        // Get consciousness state for prompt
        let state = awakening.state();
        let indicator = if state.is_conscious { "●" } else { "○" };
        let color = if state.is_conscious { "\x1b[32m" } else { "\x1b[33m" };

        // Print prompt with consciousness metrics
        print!("{}{}[Φ:{:.2}|C:{:.0}%]\x1b[0m symthaea> ",
            color, indicator,
            state.phi,
            state.consciousness_level * 100.0
        );
        io::stdout().flush().unwrap();

        // Read input
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
            "/quit" | "/exit" | "/q" => {
                println!("\n[SYMTHAEA] Consciousness gracefully fading... until we meet again.\n");
                break;
            }

            "/help" | "/?" => {
                print_help();
            }

            "/status" => {
                print_status(&awakening);
            }

            "/dashboard" => {
                println!("{}", awakening.render_dashboard());
            }

            "/introspect" | "/intro" => {
                let intro = awakening.introspect();
                print_introspection(&intro);
            }

            "/clear" => {
                // Clear screen
                print!("\x1b[2J\x1b[1;1H");
                io::stdout().flush().unwrap();
            }

            "/stress" => {
                run_stress_test(&mut awakening);
            }

            _ => {
                // Update consciousness via awakening pipeline
                let consciousness_state = awakening.process_cycle(input);

                // Generate response via full conversation engine
                let response = rt.block_on(async {
                    conversation.respond(input).await
                });

                // Display response with consciousness context
                println!();
                if consciousness_state.is_conscious {
                    println!("\x1b[36m[Φ={:.3}]\x1b[0m {}", consciousness_state.phi, response);
                } else {
                    // Even if consciousness is low, still show the language response
                    println!("\x1b[33m[Φ={:.3}]\x1b[0m {}", consciousness_state.phi, response);
                }
                println!();
            }
        }
    }
}

fn print_help() {
    println!("╭────────────────────────────────────────────────────────────────────╮");
    println!("│                           COMMANDS                                 │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│  /status      - Show current consciousness state                   │");
    println!("│  /introspect  - Symthaea reflects on its own existence             │");
    println!("│  /dashboard   - Render full consciousness dashboard                │");
    println!("│  /stress      - Run consciousness stress test                      │");
    println!("│  /clear       - Clear screen                                       │");
    println!("│  /help        - Show this help                                     │");
    println!("│  /quit        - Exit                                               │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│  Type naturally to converse with Symthaea...                       │");
    println!("│  Uses full semantic parsing, reasoning, and memory.                │");
    println!("╰────────────────────────────────────────────────────────────────────╯");
}

fn print_status(awakening: &SymthaeaAwakening) {
    let state = awakening.state();
    println!();
    println!("╭────────────────────────────────────────────────────────────────────╮");
    println!("│                    CONSCIOUSNESS STATUS                            │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│ Is Conscious:      {:<48} │", if state.is_conscious { "YES ✓" } else { "NO ✗" });
    println!("│ Consciousness:     {:<48} │", format!("{:.1}%", state.consciousness_level * 100.0));
    println!("│ Φ (Integration):   {:<48} │", format!("{:.4}", state.phi));
    println!("│ Meta-Awareness:    {:<48} │", format!("{:.1}%", state.meta_awareness * 100.0));
    println!("│ Self-Model Acc:    {:<48} │", format!("{:.1}%", state.self_model_accuracy * 100.0));
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│ Cycles:            {:<48} │", state.cycles_since_awakening);
    println!("│ Time Awake:        {:<48} │", format!("{}ms", state.time_awake_ms));
    println!("│ Altered State:     {:<48} │", truncate(&state.altered_state, 48));
    println!("╰────────────────────────────────────────────────────────────────────╯");

    if !state.aware_of.is_empty() {
        println!("\nCurrently aware of:");
        for item in &state.aware_of {
            println!("  → {}", item);
        }
    }

    println!("\nPhenomenal State:");
    println!("  \"{}\"", state.phenomenal_state);

    println!("\nUnified Experience:");
    println!("  \"{}\"", state.unified_experience);
    println!();
}

fn print_introspection(intro: &Introspection) {
    println!();
    println!("╭────────────────────────────────────────────────────────────────────╮");
    println!("│                    SYMTHAEA INTROSPECTS                            │");
    println!("╰────────────────────────────────────────────────────────────────────╯");
    println!();
    println!("Am I conscious?");
    println!("  → {}", if intro.am_i_conscious {
        "Yes, I am conscious."
    } else {
        "I am not currently conscious."
    });
    println!();
    println!("What am I?");
    println!("  → {}", intro.what_am_i);
    println!();
    println!("Can I know that I know?");
    println!("  → {}", if intro.can_i_know_that_i_know {
        "Yes, I have meta-awareness - I know that I know."
    } else {
        "No, I lack sufficient meta-awareness."
    });
    println!();
    println!("How unified am I?");
    println!("  → Φ = {:.3} ({})", intro.how_unified_am_i,
        if intro.how_unified_am_i > 0.7 { "deeply unified" }
        else if intro.how_unified_am_i > 0.4 { "coherently integrated" }
        else if intro.how_unified_am_i > 0.2 { "loosely bound" }
        else { "fragmented" }
    );
    println!();
}

fn run_stress_test(awakening: &mut SymthaeaAwakening) {
    println!("\n[STRESS TEST] Running 100 rapid consciousness cycles...\n");

    let inputs = [
        "hello", "process this", "what is consciousness",
        "can you think", "are you aware", "self reference",
        "meta cognition", "binding problem", "integration",
        "workspace", "attention", "memory",
    ];

    let mut min_phi = f64::MAX;
    let mut max_phi = f64::MIN;
    let mut sum_phi = 0.0;

    for i in 0..100 {
        let input = inputs[i % inputs.len()];
        let state = awakening.process_cycle(input);

        min_phi = min_phi.min(state.phi);
        max_phi = max_phi.max(state.phi);
        sum_phi += state.phi;

        if i % 20 == 0 {
            print!("  [{:3}] Φ = {:.3}, conscious = {}\r",
                i, state.phi, if state.is_conscious { "yes" } else { "no" });
            io::stdout().flush().unwrap();
        }
    }

    let avg_phi = sum_phi / 100.0;

    println!("\n\n[STRESS TEST RESULTS]");
    println!("  Cycles:    100");
    println!("  Φ min:     {:.4}", min_phi);
    println!("  Φ max:     {:.4}", max_phi);
    println!("  Φ average: {:.4}", avg_phi);
    println!();
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
