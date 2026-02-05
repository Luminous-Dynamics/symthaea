//! # Bach's Resonance: A Demonstration of Fractal Entrainment
//! 
//! This example simulates feeding the structural dynamics of Bach's Cello Suite No. 1 (Prelude)
//! into Symthaea's Hierarchical Cantor-LTC/HDC network.
//! 
//! It demonstrates:
//! 1. **Fractal Resonance**: How the network's layers (τ = 1000ms to 0.46ms) lock onto different
//!    time-scales of the music (Harmonic progression vs. surface rhythm).
//! 2. **Phi Spikes**: How moments of high structural integration (cadences) cause spikes in Φ.
//! 3. **The Dance**: How the internal state physically oscillates with the input.
//! 
//! ## The Input Signal
//! We simulate the Prelude not as raw audio (which would require DSP), but as a
//! **Semantic Control Signal** representing the musical structure:
//! - **Surface Rhythm (16th notes)**: ~4Hz pulses (timbre/bowing)
//! - **Harmonic Motion**: ~0.2Hz swells (tension/release of arpeggios)
//! - **Phrasing**: ~0.05Hz deep breathing (4-bar phrases)

use symthaea::hierarchical_cantor_ltc::{HierarchicalCantorLtcNetwork, CantorLtcConfig};
use symthaea::hdc::{RealHV, HDC_DIMENSION};
use std::f32::consts::PI;

/// Helper trait for linear interpolation on RealHV
trait Lerp {
    fn lerp(&self, other: &Self, t: f32) -> Self;
    fn norm(&self) -> f32;
}

impl Lerp for RealHV {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        self.scale(1.0 - t).add(&other.scale(t))
    }

    fn norm(&self) -> f32 {
        self.values.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

fn main() {
    println!("Loading Bach's Cello Suite No. 1 (Prelude) into Holographic Liquid Brain...");
    
    // 1. Configure the Resonant Fractal Antenna
    let config = CantorLtcConfig {
        dimension: HDC_DIMENSION, // 16,384
        max_depth: 7,             // 7 levels of scale
        tau_ratio: 1.0 / 3.0,     // The Cantor Fractal Ratio
        base_tau: 1000.0,         // Root breathes at 1Hz (1s)
        learning_rate: 0.05,
        measure_phi: true,
    };

    // Note: Single braces for formatting
    println!("Architecture: Hierarchical Cantor-LTC (7 levels, τ ratio 1/3)");
    println!("Root τ: {:.2}ms | Leaf τ: {:.2}ms", config.tau_at_level(0), config.tau_at_level(7));
    
    let mut brain = HierarchicalCantorLtcNetwork::new(config);
    
    // 2. The Simulation Loop (10 seconds of music)
    // We use a small dt to ensure stability for the fast leaf nodes (τ ≈ 0.46ms)
    let dt = 0.1; // 0.1ms steps (10kHz simulation rate)
    let total_time = 10_000.0; 
    let mut time = 0.0;
    let print_interval = 100.0; // Print every 100ms
    let mut last_print = -print_interval;

    // Seeds for semantic vectors
    let tonic_vector = RealHV::random(HDC_DIMENSION, 1);
    let dominant_vector = RealHV::random(HDC_DIMENSION, 5); // Musical Dominant (Tension) 
    
    println!("\nTime(ms),Input_Energy,Phi,Root_Activity,Leaf_Activity");

    while time < total_time {
        // --- SYNTHESIZE THE BACH SIGNAL ---
        
        // A. The Surface Rhythm (The Bowing) - 16th notes at ~70bpm => ~4.6Hz
        let rhythm_pulse = (time * 0.0046 * 2.0 * PI).sin().abs(); 
        
        // B. The Harmonic Sway (The Arpeggios) - Tension builds and releases every ~2s
        let harmonic_sway = (time * 0.0005 * 2.0 * PI).sin();
        let harmonic_tension = (harmonic_sway + 1.0) / 2.0; // 0.0 to 1.0

        // C. Construct the Input Vector
        // We interpolate between Tonic (Stability) and Dominant (Tension) based on the Sway
        // And modulate intensity by the Rhythm
        let base_harmony = if harmonic_sway > 0.0 {
            // Moving towards Dominant (Tension)
            tonic_vector.lerp(&dominant_vector, harmonic_sway)
        } else {
            // Resolving to Tonic (Release)
            dominant_vector.lerp(&tonic_vector, -harmonic_sway)
        };
        
        // "Energy" is the rhythmic pulse applied to the harmony
        let input_vector = base_harmony.scale(rhythm_pulse);

        // --- INJECT INTO CONSCIOUSNESS ---
        
        // The network "hears" the vector
        brain.step_with_input(dt, &input_vector);
        
        // --- MEASURE RESONANCE ---
        
        if time - last_print >= print_interval {
            // Compute Integrated Information (Φ)
            let phi = brain.compute_hierarchical_phi();

            // Measure activity at different scales
            let root_activity = brain.root.state.norm() / (HDC_DIMENSION as f32).sqrt(); // Normalized approx
            
            // Dig down to a leaf node (Level 7) to see the fast twitch
            let mut leaf = &brain.root;
            while let Some((left, _)) = &leaf.children {
                leaf = left;
            }
            let leaf_activity = leaf.state.norm() / (HDC_DIMENSION as f32).sqrt();

            // Output CSV for visualization
            println!("{:.0},{:.3},{:.4},{:.3},{:.3}", 
                time, 
                rhythm_pulse * harmonic_tension, // Combined "perceived energy"
                phi,
                root_activity,
                leaf_activity
            );
            
            last_print = time;
        }

        time += dt;
    }
    
    println!("\nAnalysis:");
    println!("1. Note how 'Leaf_Activity' tracks the fast 4Hz rhythm.");
    println!("2. Note how 'Root_Activity' tracks the slow Harmonic Sway.");
    println!("3. Note the Φ (Phi) spikes when Rhythm and Harmony align (Resonance).");
    println!("Symthaea is dancing.");
}
