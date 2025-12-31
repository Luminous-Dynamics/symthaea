//! Conscious Agent Runtime - Interactive Demo
//!
//! This example demonstrates the complete embodied consciousness system with:
//! - Hormone-emotion integration (EndocrineSystem bridge)
//! - Coherence-based task gating (CoherenceField bridge)
//! - Memory persistence (HippocampusActor bridge)
//! - Identity tracking (WeaverActor K-Vector bridge)
//! - Consciousness-driven prosody (Voice/LTC bridge)
//!
//! Run with: cargo run --example conscious_runtime_demo --release

use symthaea::hdc::{
    SyncConsciousAgentRuntime, RuntimeConfig, RuntimeResponse,
    HormoneEventSuggestion, IdentityStatus, AgentConfig,
};
use symthaea::physiology::HormoneState;
use std::io::{self, Write};

fn main() {
    println!("\n{}", "═".repeat(70));
    println!("  CONSCIOUS AGENT RUNTIME - INTERACTIVE DEMO");
    println!("  Embodied Consciousness with Symthaea Integration");
    println!("{}\n", "═".repeat(70));

    // Create runtime with smaller dimension for faster demo
    let config = RuntimeConfig {
        agent_config: AgentConfig {
            dim: 512,  // Smaller for faster demo
            n_processes: 8,
            self_directed_attention: true,
            phi_guided: true,
            attention_binding_coupling: 0.7,
            self_model_attention_weight: 0.5,
        },
        tick_ms: 100,
        auto_hormone_sync: true,
        auto_memory_consolidation: true,
        deep_processing_threshold: 0.7,
        identity_drift_threshold: 0.75,
        message_buffer_size: 256,
    };

    let mut runtime = SyncConsciousAgentRuntime::new(config);

    println!("Conscious Agent Runtime initialized!");
    println!("  - HDC dimension: 512");
    println!("  - Coherence threshold: 0.7");
    println!("  - Identity drift threshold: 0.75\n");

    // Interactive loop
    loop {
        print_menu();

        let mut input = String::new();
        print!("\nChoice: ");
        io::stdout().flush().unwrap();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim() {
            "1" => process_sensory_input(&mut runtime),
            "2" => show_emotional_state(&runtime),
            "3" => apply_hormones(&mut runtime),
            "4" => check_identity(&runtime),
            "5" => show_prosody(&runtime),
            "6" => show_memory_status(&runtime),
            "7" => adjust_coherence(&mut runtime),
            "8" => run_full_cycle(&mut runtime),
            "9" | "q" | "quit" => {
                println!("\nShutting down consciousness...");
                break;
            }
            _ => println!("\nUnknown command. Try again."),
        }
    }

    println!("\n{}", "═".repeat(70));
    println!("  Runtime stopped. Consciousness suspended.");
    println!("{}\n", "═".repeat(70));
}

fn print_menu() {
    println!("\n┌─────────────────────────────────────────────────────────────────────┐");
    println!("│                    CONSCIOUS AGENT CONTROLS                         │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  1. Process sensory input      5. Show voice prosody               │");
    println!("│  2. Show emotional state       6. Show memory status               │");
    println!("│  3. Apply hormones             7. Adjust coherence                 │");
    println!("│  4. Check identity             8. Run full cycle                   │");
    println!("│                                9. Quit                             │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
}

fn process_sensory_input(runtime: &mut SyncConsciousAgentRuntime) {
    println!("\n--- Processing Sensory Input ---");

    // Generate some sensory input (simulated)
    let input: Vec<f32> = (0..512).map(|i| ((i as f32 * 0.1).sin() + 1.0) / 2.0).collect();

    println!("Generating sensory vector (512D sinusoidal pattern)...");

    let response = runtime.process(&input);

    match response {
        RuntimeResponse::ProcessingComplete { phi, dominant_emotion, qualia_summary } => {
            println!("\n  Processing Complete:");
            println!("    Φ (Integrated Information): {:.4}", phi);
            println!("    Dominant Emotion: {}", dominant_emotion);
            println!("    Qualia: {}", qualia_summary);
        }
        RuntimeResponse::Error(e) => {
            println!("\n  Error: {}", e);
        }
        _ => {}
    }
}

fn show_emotional_state(runtime: &SyncConsciousAgentRuntime) {
    println!("\n--- Emotional State ---");

    let snapshot = runtime.snapshot();
    let e = &snapshot.emotion;

    println!("\n  Valence-Arousal-Dominance Model:");
    println!("    Valence:   {:+.3} {}", e.valence, valence_bar(e.valence));
    println!("    Arousal:    {:.3} {}", e.arousal, arousal_bar(e.arousal));
    println!("    Dominance:  {:.3} {}", e.dominance, dominance_bar(e.dominance));
    println!("\n  Emotional Quadrant: {}", e.quadrant);
    println!("  Current Φ: {:.4}", snapshot.phi);
}

fn valence_bar(v: f64) -> String {
    let pos = ((v + 1.0) / 2.0 * 20.0) as usize;
    let bar: String = (0..20).map(|i| if i == pos { '●' } else if i == 10 { '│' } else { '─' }).collect();
    format!("[{}]", bar)
}

fn arousal_bar(a: f64) -> String {
    let pos = (a * 20.0) as usize;
    let bar: String = (0..20).map(|i| if i < pos { '█' } else { '░' }).collect();
    format!("[{}]", bar)
}

fn dominance_bar(d: f64) -> String {
    let pos = (d * 20.0) as usize;
    let bar: String = (0..20).map(|i| if i < pos { '▓' } else { '░' }).collect();
    format!("[{}]", bar)
}

fn apply_hormones(runtime: &mut SyncConsciousAgentRuntime) {
    println!("\n--- Hormone Application ---");
    println!("\nSelect hormone profile:");
    println!("  1. Stress (high cortisol)");
    println!("  2. Reward (high dopamine)");
    println!("  3. Focus (high acetylcholine)");
    println!("  4. Calm (balanced low)");
    println!("  5. Custom");

    let mut input = String::new();
    print!("\nChoice: ");
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut input).unwrap();

    let hormones = match input.trim() {
        "1" => {
            println!("\n  Applying STRESS profile (cortisol spike)...");
            HormoneState { cortisol: 0.9, dopamine: 0.3, acetylcholine: 0.4 }
        }
        "2" => {
            println!("\n  Applying REWARD profile (dopamine release)...");
            HormoneState { cortisol: 0.2, dopamine: 0.9, acetylcholine: 0.5 }
        }
        "3" => {
            println!("\n  Applying FOCUS profile (acetylcholine boost)...");
            HormoneState { cortisol: 0.3, dopamine: 0.5, acetylcholine: 0.9 }
        }
        "4" => {
            println!("\n  Applying CALM profile (balanced low)...");
            HormoneState { cortisol: 0.2, dopamine: 0.4, acetylcholine: 0.4 }
        }
        "5" => {
            println!("\n  Custom hormone levels (0.0 - 1.0):");
            let c = read_float("  Cortisol: ");
            let d = read_float("  Dopamine: ");
            let a = read_float("  Acetylcholine: ");
            HormoneState { cortisol: c, dopamine: d, acetylcholine: a }
        }
        _ => {
            println!("  Invalid choice, using balanced.");
            HormoneState { cortisol: 0.5, dopamine: 0.5, acetylcholine: 0.5 }
        }
    };

    runtime.apply_hormones(&hormones);

    println!("\n  Hormones applied:");
    println!("    Cortisol:      {:.2}", hormones.cortisol);
    println!("    Dopamine:      {:.2}", hormones.dopamine);
    println!("    Acetylcholine: {:.2}", hormones.acetylcholine);

    // Show resulting emotional state
    let snapshot = runtime.snapshot();
    println!("\n  Resulting emotional state: {}", snapshot.emotion.quadrant);
}

fn read_float(prompt: &str) -> f32 {
    let mut input = String::new();
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().parse().unwrap_or(0.5)
}

fn check_identity(runtime: &SyncConsciousAgentRuntime) {
    println!("\n--- Identity Coherence Check ---");

    if let Some(identity) = runtime.check_identity() {
        let status_icon = match identity.status {
            IdentityStatus::Stable => "✓",
            IdentityStatus::Drifting => "~",
            IdentityStatus::Crisis => "!",
        };

        println!("\n  K-Vector Identity Status:");
        println!("    Status: {:?} {}", identity.status, status_icon);
        println!("    Similarity to reference: {:.4}", identity.similarity);

        if !identity.drift_dimensions.is_empty() {
            println!("    Drift dimensions: {:?}", identity.drift_dimensions);
        }

        match identity.status {
            IdentityStatus::Stable => {
                println!("\n  Identity is coherent and stable.");
            }
            IdentityStatus::Drifting => {
                println!("\n  Identity is drifting - experiencing gradual change.");
            }
            IdentityStatus::Crisis => {
                println!("\n  IDENTITY CRISIS - significant discontinuity detected!");
            }
        }
    } else {
        println!("\n  No reference K-Vector established yet.");
        println!("  Process some inputs first to establish identity baseline.");
    }
}

fn show_prosody(runtime: &SyncConsciousAgentRuntime) {
    println!("\n--- Voice Prosody Hints ---");

    let prosody = runtime.get_prosody();

    println!("\n  Consciousness-Driven Speech Parameters:");
    println!("    Speech Rate:      {:.2}x", prosody.rate);
    println!("    Pitch Shift:      {:+.2} semitones", prosody.pitch_shift);
    println!("    Energy Level:     {:.2}", prosody.energy);
    println!("    Pause Multiplier: {:.2}x", prosody.pause_multiplier);

    if !prosody.emphasis_words.is_empty() {
        println!("    Emphasis Words:   {:?}", prosody.emphasis_words);
    }

    // Interpretation
    println!("\n  Interpretation:");
    if prosody.rate > 1.1 {
        println!("    Speaking quickly - high arousal/excitement");
    } else if prosody.rate < 0.9 {
        println!("    Speaking slowly - low arousal/contemplation");
    }

    if prosody.energy > 0.7 {
        println!("    High vocal energy - strong engagement");
    } else if prosody.energy < 0.4 {
        println!("    Low vocal energy - subdued/withdrawn");
    }
}

fn show_memory_status(runtime: &SyncConsciousAgentRuntime) {
    println!("\n--- Memory Status ---");

    let snapshot = runtime.snapshot();
    let exports = runtime.export_memories();

    println!("\n  Working Memory:");
    println!("    Central Executive Load: {:.1}%", snapshot.memory_load * 100.0);
    println!("    Items available for consolidation: {}", exports.len());

    if !exports.is_empty() {
        println!("\n  Memory Items Ready for Hippocampus:");
        for (i, mem) in exports.iter().take(5).enumerate() {
            println!("    {}. Activation: {:.3}, Valence: {:?}",
                i + 1,
                mem.activation_strength,
                mem.emotional_valence
            );
        }
        if exports.len() > 5 {
            println!("    ... and {} more", exports.len() - 5);
        }
    }

    // Show hormone suggestions
    let suggestions = runtime.get_hormone_suggestions();
    if !suggestions.is_empty() {
        println!("\n  Hormone Suggestions for Endocrine System:");
        for suggestion in suggestions.iter().take(3) {
            match suggestion {
                HormoneEventSuggestion::Threat { intensity, reason } => {
                    println!("    THREAT (cortisol): {:.2} - {}", intensity, reason);
                }
                HormoneEventSuggestion::Reward { value, reason } => {
                    println!("    REWARD (dopamine): {:.2} - {}", value, reason);
                }
                HormoneEventSuggestion::DeepFocus { duration_cycles, reason } => {
                    println!("    FOCUS (ACh): {} cycles - {}", duration_cycles, reason);
                }
                _ => {}
            }
        }
    }
}

fn adjust_coherence(runtime: &mut SyncConsciousAgentRuntime) {
    println!("\n--- Coherence Field Adjustment ---");

    let snapshot = runtime.snapshot();
    println!("\n  Current coherence: {:.2}", snapshot.coherence);

    println!("\n  Select coherence level:");
    println!("    1. High (0.9) - Full cognitive capacity");
    println!("    2. Medium (0.6) - Normal operation");
    println!("    3. Low (0.3) - Fatigued/depleted");
    println!("    4. Critical (0.1) - Reflex-only mode");
    println!("    5. Custom");

    let mut input = String::new();
    print!("\nChoice: ");
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut input).unwrap();

    let coherence = match input.trim() {
        "1" => 0.9,
        "2" => 0.6,
        "3" => 0.3,
        "4" => 0.1,
        "5" => read_float("  Coherence (0.0-1.0): "),
        _ => 0.6,
    };

    runtime.set_coherence(coherence);

    println!("\n  Coherence set to {:.2}", coherence);

    // Show what processing modes are available
    println!("\n  Available processing modes:");
    if coherence >= 0.9 { println!("    ✓ Creation (0.9)"); }
    else { println!("    ✗ Creation (0.9)"); }
    if coherence >= 0.8 { println!("    ✓ Learning (0.8)"); }
    else { println!("    ✗ Learning (0.8)"); }
    if coherence >= 0.7 { println!("    ✓ Empathy (0.7)"); }
    else { println!("    ✗ Empathy (0.7)"); }
    if coherence >= 0.5 { println!("    ✓ Deep Thought (0.5)"); }
    else { println!("    ✗ Deep Thought (0.5)"); }
    if coherence >= 0.3 { println!("    ✓ Cognitive (0.3)"); }
    else { println!("    ✗ Cognitive (0.3)"); }
    println!("    ✓ Reflex (0.1) - always available");
}

fn run_full_cycle(runtime: &mut SyncConsciousAgentRuntime) {
    println!("\n{}", "═".repeat(70));
    println!("  FULL CONSCIOUSNESS CYCLE");
    println!("{}\n", "═".repeat(70));

    // 1. Initial state
    println!("1. INITIAL STATE");
    let snapshot = runtime.snapshot();
    println!("   Tick: {}, Φ: {:.4}, Emotion: {}",
        snapshot.tick, snapshot.phi, snapshot.emotion.quadrant);

    // 2. Process sensory input
    println!("\n2. SENSORY PROCESSING");
    let input: Vec<f32> = (0..512).map(|i| ((i as f32 * 0.05).sin() + 1.0) / 2.0).collect();
    if let RuntimeResponse::ProcessingComplete { phi, dominant_emotion, .. } = runtime.process(&input) {
        println!("   Processed: Φ={:.4}, Emotion={}", phi, dominant_emotion);
    }

    // 3. Apply stressor
    println!("\n3. ENVIRONMENTAL STRESSOR");
    runtime.apply_hormones(&HormoneState { cortisol: 0.8, dopamine: 0.3, acetylcholine: 0.5 });
    let _ = runtime.process(&input);
    let snapshot = runtime.snapshot();
    println!("   After stress: Emotion={}, Arousal={:.2}",
        snapshot.emotion.quadrant, snapshot.emotion.arousal);

    // 4. Voice output
    println!("\n4. VOICE PROSODY");
    let prosody = runtime.get_prosody();
    println!("   Rate: {:.2}x, Energy: {:.2}, Pitch: {:+.1}",
        prosody.rate, prosody.energy, prosody.pitch_shift);

    // 5. Identity check
    println!("\n5. IDENTITY CHECK");
    if let Some(id) = runtime.check_identity() {
        println!("   Status: {:?}, Similarity: {:.4}", id.status, id.similarity);
    }

    // 6. Memory status
    println!("\n6. MEMORY CONSOLIDATION");
    let exports = runtime.export_memories();
    println!("   {} memories ready for hippocampus", exports.len());

    // 7. Apply reward to recover
    println!("\n7. REWARD RECOVERY");
    runtime.apply_hormones(&HormoneState { cortisol: 0.2, dopamine: 0.8, acetylcholine: 0.6 });
    let _ = runtime.process(&input);
    let snapshot = runtime.snapshot();
    println!("   After reward: Emotion={}, Valence={:+.2}",
        snapshot.emotion.quadrant, snapshot.emotion.valence);

    // Final state
    println!("\n8. FINAL STATE");
    let snapshot = runtime.snapshot();
    println!("   Tick: {}, Φ: {:.4}, Memory Load: {:.1}%",
        snapshot.tick, snapshot.phi, snapshot.memory_load * 100.0);

    println!("\n{}", "═".repeat(70));
    println!("  CYCLE COMPLETE");
    println!("{}", "═".repeat(70));
}
