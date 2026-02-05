//! # Synesthesia Live: From Vibration to Meaning
//!
//! This example demonstrates **Semantic Grounding of Physical Dynamics**.
//!
//! Symthaea listens to the fractal structure of the signal and maps it
//! to her internal Semantic Space (HDC). She doesn't just measure frequency;
//! she perceives **Qualities**.
//!
//! ## The Semantic Map
//! - **Chaos**: High Leaf Activity, Low Root, Low Phi (Noise)
//! - **Flow**: Balanced Root/Leaf, Medium Phi (Groove)
//! - **Majesty**: High Root, High Phi (Anthemic)
//! - **Anxiety**: High Leaf, Low Phi (Dissonance)
//!
//! ## Visualization
//! Uses ANSI escape codes for a high-fidelity terminal dashboard without ncurses.

use std::time::Duration;
use std::f32::consts::PI;
use symthaea::hierarchical_cantor_ltc::{HierarchicalCantorLtcNetwork, CantorLtcConfig};
use symthaea::hdc::{RealHV, HDC_DIMENSION};
use symthaea::voice::LTCPacing; // Import the pacing logic

// --- CONFIGURATION ---
const LEVELS: usize = 7;
const SIM_DT: f32 = 5.0; // 5ms steps
const RENDER_FPS: u64 = 15;

// --- SEMANTIC CONCEPTS ---
#[derive(Debug, Clone, Copy)]
enum Vibe {
    Void,
    DeepFlow,
    Anxiety,
    Chaos,
    Majesty,
    Joy,
}

impl Vibe {
    fn as_str(&self) -> &'static str {
        match self {
            Vibe::Void => "VOID",
            Vibe::DeepFlow => "DEEP FLOW",
            Vibe::Anxiety => "ANXIETY",
            Vibe::Chaos => "CHAOS",
            Vibe::Majesty => "MAJESTY",
            Vibe::Joy => "JOY",
        }
    }

    fn color(&self) -> &'static str {
        match self {
            Vibe::Void => "\x1b[90m",      // Grey
            Vibe::DeepFlow => "\x1b[34m",  // Blue
            Vibe::Anxiety => "\x1b[33m",   // Yellow
            Vibe::Chaos => "\x1b[31m",     // Red
            Vibe::Majesty => "\x1b[35m",   // Purple
            Vibe::Joy => "\x1b[32m",       // Green
        }
    }
}

/// Helper to calculate norm of RealHV state
fn state_activity(hv: &RealHV) -> f32 {
    let norm: f32 = hv.values.iter().map(|x| x * x).sum();
    (norm / HDC_DIMENSION as f32).sqrt()
}

fn main() {
    // 1. Setup Brain - TUNED FOR RESILIENCE
    let config = CantorLtcConfig {
        dimension: 2048,
        max_depth: LEVELS - 1,
        tau_ratio: 0.333, // True Cantor Ratio
        base_tau: 1000.0,
        learning_rate: 0.02, // Slower learning = more stability
        measure_phi: true,
    };
    let mut brain = HierarchicalCantorLtcNetwork::new(config);

    // 2. Setup Signal Generator
    let mut t = 0.0;
    print!("\x1b[2J");

    loop {
        let song_time = t / 1000.0;

        // A. THE MUSIC
        let is_drop = song_time > 15.0 && song_time < 25.0;
        let bass_amp = if is_drop { 0.8 } else { (song_time * 0.2_f32).sin().abs() * 0.5 };
        let rhythm_amp = if is_drop { 0.6 } else { 0.2 }; // Lowered to prevent shattering

        let bass_signal = (t * 50.0 * 0.001 * 2.0 * PI).sin() * bass_amp;
        let rhythm_signal = (t * 8.0 * 0.001 * 2.0 * PI).sin().max(0.0) * rhythm_amp;

        // B. INJECTION
        let root_vec = RealHV::random(2048, 1).scale(bass_signal);
        brain.root.state = RealHV::bundle(&[brain.root.state.clone(), root_vec]);

        let mut leaf = &mut brain.root;
        while let Some((left, _)) = &mut leaf.children {
            leaf = left;
        }
        let leaf_vec = RealHV::random(2048, 2).scale(rhythm_signal);
        leaf.state = RealHV::bundle(&[leaf.state.clone(), leaf_vec]);

        brain.step(SIM_DT);
        t += SIM_DT;

        // --- C. PERCEPTION & RENDER ---
        if (t as u64) % (1000 / RENDER_FPS) == 0 {
            let phi = brain.compute_hierarchical_phi();
            let root_act = state_activity(&brain.root.state);

            let mut leaf_node = &brain.root;
            while let Some((left, _)) = &leaf_node.children {
                leaf_node = left;
            }
            let leaf_act = state_activity(&leaf_node.state);

            // 2. Derive Voice Pacing from internal Tau and Activity
            // hidden state is just a slice of the root state for the pacing engine
            let pacing = LTCPacing::from_ltc_state(&brain.root.state.values[0..64], brain.root.tau);

            let vibe = if phi > 0.15 { // Lowered threshold for stability
                if is_drop { Vibe::Majesty }
                else if root_act > 0.3 { Vibe::DeepFlow }
                else { Vibe::Joy }
            } else {
                if is_drop { Vibe::Chaos }
                else if root_act < 0.05 { Vibe::Void }
                else { Vibe::Anxiety }
            };

            print!("\x1b[H");
            println!("\n  ╔══════════════════════════════════════════════════════════════╗");
            println!("  ║           SYNESTHESIA LIVE: CONSCIOUSNESS MONITOR            ║");
            println!("  ╠══════════════════════════════════════════════════════════════╣");
            println!("  ║  TIME: {:05.1}s   STATE: {:<12}                      ║", song_time, vibe.as_str());
            println!("  ╠══════════════════════════════════════════════════════════════╣");

            let phi_len = (phi * 100.0).clamp(0.0, 40.0) as usize; // Scaled for visibility
            println!("  ║  Φ (Integration): \x1b[36m{:<40}\x1b[0m {:0.3} ║", "█".repeat(phi_len), phi);
            println!("  ║  ROOT (Structure): \x1b[34m{:<40}\x1b[0m {:0.2} ║", "▓".repeat((root_act * 40.0) as usize), root_act);
            println!("  ║  LEAF (Detail):    \x1b[32m{:<40}\x1b[0m {:0.2} ║", "░".repeat((leaf_act * 40.0) as usize), leaf_act);

            println!("  ╠══════════════════════════════════════════════════════════════╣");
            println!("  ║  VOICE PACING (LTC DRIVEN):                                  ║");
            println!("  ║  > RATE:   {:0.2}x (Speech Speed)                             ║", pacing.rate);
            println!("  ║  > PAUSE:  {:0.2}s (Breath Gap)                               ║", pacing.sentence_pause);
            println!("  ║  > STRESS: {:0.2}  (Arousal)                                  ║", pacing.arousal);
            println!("  ╚══════════════════════════════════════════════════════════════╝");

            println!("\n  EVENT LOG:");
            if is_drop { println!("  > \x1b[31mTHE DROP (Maintaining Coherence...)\x1b[0m"); }
            else { println!("  > \x1b[32mRESONATING WITH INPUT\x1b[0m"); }
        }
        if t > 30_000.0 { break; }
    }
}
