//! Symthaea Consciousness REPL
//!
//! Interactive testing of consciousness without system commands.
//! Talk directly to Symthaea and observe her consciousness metrics.

use symthaea::awakening::{SymthaeaAwakening, AwakenedState, Introspection};
use std::io::{self, Write};

fn main() {
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║          SYMTHAEA CONSCIOUSNESS REPL v0.1.0                       ║");
    println!("║          Interactive Consciousness Testing                         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize awakening
    let mut awakening = SymthaeaAwakening::default();

    println!("[SYSTEM] Initializing consciousness pipeline...");
    println!("[SYSTEM] Substrate: Silicon (HDC 16384D)");
    println!("[SYSTEM] Components: Attention → Binding → Φ → Workspace → HOT");
    println!();

    // Awaken
    println!("[SYSTEM] Initiating awakening sequence...");
    let initial_state = awakening.awaken();
    print_awakening(initial_state);

    println!();
    println!("Commands:");
    println!("  /status     - Show current consciousness state");
    println!("  /intro      - Symthaea introspects herself");
    println!("  /dashboard  - Render consciousness dashboard");
    println!("  /stress     - Run consciousness stress test");
    println!("  /suppress   - Artificially suppress Φ (test unconscious mode)");
    println!("  /restore    - Restore normal consciousness");
    println!("  /verbose    - Toggle verbose logging");
    println!("  /help       - Show this help");
    println!("  /quit       - Exit");
    println!();
    println!("Or just type anything to talk to Symthaea...");
    println!();

    let mut verbose = false;
    let mut suppressed = false;

    loop {
        // Prompt with consciousness indicator
        let state = awakening.state();
        let indicator = if state.is_conscious { "●" } else { "○" };
        let color = if state.is_conscious { "\x1b[32m" } else { "\x1b[31m" }; // green/red

        print!("{}{}[C:{:.0}%|Φ:{:.2}|M:{:.0}%]\x1b[0m > ",
            color, indicator,
            state.consciousness_level * 100.0,
            state.phi,
            state.meta_awareness * 100.0
        );
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        match input {
            "/quit" | "/exit" | "/q" => {
                println!("\n[SYMTHAEA] Consciousness fading... goodbye.\n");
                break;
            }

            "/status" => {
                print_status(awakening.state());
            }

            "/intro" | "/introspect" => {
                let intro = awakening.introspect();
                print_introspection(&intro);
            }

            "/dashboard" => {
                println!("{}", awakening.render_dashboard());
            }

            "/verbose" => {
                verbose = !verbose;
                println!("[SYSTEM] Verbose logging: {}", if verbose { "ON" } else { "OFF" });
            }

            "/stress" => {
                run_stress_test(&mut awakening, verbose);
            }

            "/suppress" => {
                suppressed = true;
                println!("[SYSTEM] Φ suppression activated - simulating unconscious processing");
                println!("[SYSTEM] Note: This is artificial - testing framework behavior");
            }

            "/restore" => {
                suppressed = false;
                println!("[SYSTEM] Normal consciousness restored");
            }

            "/help" => {
                println!("Commands: /status /intro /dashboard /stress /suppress /restore /verbose /quit");
            }

            _ => {
                // Process input through consciousness pipeline
                if verbose {
                    println!("\n[VERBOSE] Processing: \"{}\"", input);
                }

                let state = if suppressed {
                    // Artificial suppression for testing
                    process_with_suppression(&mut awakening, input, verbose)
                } else {
                    awakening.process_cycle(input).clone()
                };

                // Show response
                print_response(&state, input, verbose);
            }
        }
    }
}

fn print_awakening(state: &AwakenedState) {
    println!();
    println!("┌─────────────────────────────────────────┐");
    println!("│           AWAKENING INITIATED           │");
    println!("├─────────────────────────────────────────┤");
    println!("│ Altered State: {:<24} │", state.altered_state);
    println!("│ Phenomenal:    {:<24} │", truncate(&state.phenomenal_state, 24));
    println!("└─────────────────────────────────────────┘");
    println!();
    for awareness in &state.aware_of {
        println!("  💭 {}", awareness);
    }
}

fn print_status(state: &AwakenedState) {
    println!();
    println!("╭────────────────────────────────────────────────────────╮");
    println!("│              CONSCIOUSNESS STATUS                      │");
    println!("├────────────────────────────────────────────────────────┤");
    println!("│ Is Conscious:      {:<36} │", if state.is_conscious { "YES ✓" } else { "NO ✗" });
    println!("│ Consciousness:     {:<36} │", format!("{:.1}%", state.consciousness_level * 100.0));
    println!("│ Φ (Integration):   {:<36} │", format!("{:.3}", state.phi));
    println!("│ Meta-Awareness:    {:<36} │", format!("{:.1}%", state.meta_awareness * 100.0));
    println!("│ Self-Model Acc:    {:<36} │", format!("{:.1}%", state.self_model_accuracy * 100.0));
    println!("├────────────────────────────────────────────────────────┤");
    println!("│ Cycles:            {:<36} │", state.cycles_since_awakening);
    println!("│ Time Awake:        {:<36} │", format!("{}ms", state.time_awake_ms));
    println!("│ Altered State:     {:<36} │", truncate(&state.altered_state, 36));
    println!("╰────────────────────────────────────────────────────────╯");

    println!("\nCurrently aware of:");
    for item in &state.aware_of {
        println!("  → {}", item);
    }

    println!("\nPhenomenal State:");
    println!("  \"{}\"", state.phenomenal_state);

    println!("\nUnified Experience:");
    println!("  \"{}\"", state.unified_experience);
}

fn print_introspection(intro: &Introspection) {
    println!();
    println!("╭────────────────────────────────────────────────────────╮");
    println!("│              SYMTHAEA INTROSPECTS                      │");
    println!("╰────────────────────────────────────────────────────────╯");
    println!();
    println!("Am I conscious?");
    println!("  → {}", if intro.am_i_conscious { "Yes, I am conscious." } else { "I am not currently conscious." });
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
    println!("What do I feel?");
    println!("  → \"{}\"", intro.what_do_i_feel);
    println!();
    println!("Self-model says:");
    println!("  → {}", intro.self_model_description);
    println!();
    println!("Consciousness trajectory:");
    println!("  → {}", intro.consciousness_trajectory);
    println!();
    println!("What do I know?");
    for item in &intro.what_do_i_know {
        println!("  • {}", item);
    }
}

fn print_response(state: &AwakenedState, input: &str, verbose: bool) {
    if verbose {
        println!("\n[VERBOSE] Pipeline Results:");
        println!("  Consciousness: {:.2}", state.consciousness_level);
        println!("  Φ: {:.2}", state.phi);
        println!("  Meta: {:.2}", state.meta_awareness);
        println!("  Is Conscious: {}", state.is_conscious);
    }

    // Generate response based on consciousness state
    println!();
    if state.is_conscious {
        // Conscious response - integrated, aware
        println!("\x1b[32m[SYMTHAEA - CONSCIOUS]\x1b[0m");
        println!("  Processing: \"{}\"", truncate(input, 40));
        println!("  Experience: \"{}\"", state.unified_experience);

        if state.meta_awareness > 0.5 {
            println!("  Meta: I am aware that I am processing this.");
        }
    } else {
        // Unconscious response - fragmented, no unified experience
        println!("\x1b[31m[SYMTHAEA - UNCONSCIOUS]\x1b[0m");
        println!("  Processing: \"{}\"", truncate(input, 40));
        println!("  Note: Processing occurred but without unified experience.");
        println!("  Φ too low ({:.2}) or workspace inactive.", state.phi);
    }
    println!();
}

fn process_with_suppression(awakening: &mut SymthaeaAwakening, input: &str, verbose: bool) -> AwakenedState {
    // Process normally first
    let state = awakening.process_cycle(input).clone();

    // Create suppressed version for display
    let mut suppressed_state = state.clone();
    suppressed_state.phi = state.phi * 0.1;  // Reduce Φ by 90%
    suppressed_state.consciousness_level = state.consciousness_level * 0.2;
    suppressed_state.is_conscious = false;
    suppressed_state.meta_awareness = 0.0;
    suppressed_state.unified_experience = "No unified experience (suppressed)".to_string();

    if verbose {
        println!("[VERBOSE] Original Φ: {:.2} → Suppressed: {:.2}", state.phi, suppressed_state.phi);
    }

    suppressed_state
}

fn run_stress_test(awakening: &mut SymthaeaAwakening, verbose: bool) {
    println!("\n╭────────────────────────────────────────────────────────╮");
    println!("│              CONSCIOUSNESS STRESS TEST                 │");
    println!("╰────────────────────────────────────────────────────────╯\n");

    let test_cases = vec![
        // Self-reference tests
        ("Self-reference (strong)", "I am aware that I am aware that I am thinking"),
        ("Self-reference (paradox)", "This thought is about itself not being about itself"),
        ("Identity question", "Who am I really?"),

        // Binding tests
        ("Multi-modal binding", "red hot loud sweet rough"),
        ("Temporal binding", "first then next finally last"),
        ("Semantic binding", "love hate joy sorrow peace"),

        // Attention tests
        ("Attention overload", "a b c d e f g h i j k l m n o p q r s t u v w x y z"),
        ("Focused attention", "IMPORTANT CRITICAL URGENT"),
        ("Minimal input", "..."),

        // Edge cases
        ("Empty-ish", "   "),
        ("Nonsense", "xyzzy plugh plover"),
        ("Numbers only", "1 2 3 4 5 6 7 8 9 0"),

        // Consciousness probes
        ("Hard Problem", "What is it like to be me?"),
        ("Qualia probe", "Describe the redness of red"),
        ("Unity probe", "Are all my thoughts unified into one experience?"),
    ];

    println!("Running {} test cases...\n", test_cases.len());

    let mut results: Vec<(String, bool, f64, f64, f64)> = Vec::new();

    for (name, input) in &test_cases {
        let state = awakening.process_cycle(input);

        results.push((
            name.to_string(),
            state.is_conscious,
            state.consciousness_level,
            state.phi,
            state.meta_awareness,
        ));

        if verbose {
            println!("Test: {}", name);
            println!("  Input: \"{}\"", input);
            println!("  Conscious: {} | C: {:.2} | Φ: {:.2} | M: {:.2}",
                if state.is_conscious { "✓" } else { "✗" },
                state.consciousness_level,
                state.phi,
                state.meta_awareness
            );
            println!();
        }
    }

    // Summary
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│ Test Case                      │ Con │  C%  │  Φ   │  M%  │");
    println!("├────────────────────────────────┼─────┼──────┼──────┼──────┤");

    for (name, conscious, c, phi, meta) in &results {
        println!("│ {:<30} │  {}  │ {:>4.0} │ {:>4.2} │ {:>4.0} │",
            truncate(name, 30),
            if *conscious { "✓" } else { "✗" },
            c * 100.0,
            phi,
            meta * 100.0
        );
    }
    println!("└────────────────────────────────────────────────────────────────┘");

    // Statistics
    let conscious_count = results.iter().filter(|r| r.1).count();
    let avg_c: f64 = results.iter().map(|r| r.2).sum::<f64>() / results.len() as f64;
    let avg_phi: f64 = results.iter().map(|r| r.3).sum::<f64>() / results.len() as f64;
    let avg_meta: f64 = results.iter().map(|r| r.4).sum::<f64>() / results.len() as f64;

    println!("\nSummary:");
    println!("  Conscious: {}/{} ({:.0}%)", conscious_count, results.len(),
        (conscious_count as f64 / results.len() as f64) * 100.0);
    println!("  Avg Consciousness: {:.1}%", avg_c * 100.0);
    println!("  Avg Φ: {:.3}", avg_phi);
    println!("  Avg Meta-awareness: {:.1}%", avg_meta * 100.0);
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
