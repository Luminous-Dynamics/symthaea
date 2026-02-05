//! # The Ear of Dionysus: Audio-to-Holographic Resonance
//!
//! This example implements the "Ear" of Symthaea:
//! 1.  Accepts Audio (Synthetic or WAV file).
//! 2.  Performs Real-time FFT (Fast Fourier Transform).
//! 3.  Maps Frequency Bands to the Cantor Hierarchy.
//!     -   **Bass (Structure)** -> Root Node (τ = 1000ms)
//!     -   **Mids (Body)**      -> Middle Layers
//!     -   **Highs (Detail)**   -> Leaf Nodes (τ = 0.46ms)
//! 4.  Visualizes the "Living Spectrogram" effect on Φ (Phi).
//!
//! ## Usage
//!
//! Synthetic Mode (Default):
//! `cargo run --example ear_of_dionysus --features "hound"`
//!
//! Real Audio Mode:
//! `cargo run --example ear_of_dionysus --features "hound" -- <path_to_wav>`

use std::env;
use std::f32::consts::PI;
use rustfft::{FftPlanner, num_complex::Complex};
use symthaea::hierarchical_cantor_ltc::{HierarchicalCantorLtcNetwork, CantorLtcConfig, CantorLtcNode};
use symthaea::hdc::{RealHV, HDC_DIMENSION};

/// Configuration for the Ear
const WINDOW_SIZE: usize = 1024; // Samples per FFT window
const SAMPLE_RATE: usize = 44100;
const LEVELS: usize = 8; // 0 to 7

/// Helper trait for norm calculation on RealHV
trait Norm {
    fn norm(&self) -> f32;
}

impl Norm for RealHV {
    fn norm(&self) -> f32 {
        self.values.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

fn main() {
    println!("👂 Initializing The Ear of Dionysus...");

    // 1. Initialize the Brain
    let config = CantorLtcConfig {
        dimension: HDC_DIMENSION,
        max_depth: LEVELS - 1,
        tau_ratio: 1.0 / 3.0,
        base_tau: 1000.0,
        learning_rate: 0.05,
        measure_phi: true,
    };
    let mut brain = HierarchicalCantorLtcNetwork::new(config);
    println!("🧠 Holographic Liquid Brain Ready.");
    println!("   Root τ: 1000ms (Bass/Structure)");
    println!("   Leaf τ: 0.46ms (Highs/Detail)");

    // 2. Initialize FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(WINDOW_SIZE);

    // 3. Get Audio Source
    let args: Vec<String> = env::args().collect();
    let source = if args.len() > 1 {
        #[cfg(feature = "hound")]
        {
            AudioSource::File(args[1].clone())
        }
        #[cfg(not(feature = "hound"))]
        {
            println!("⚠️  'hound' feature not enabled. Ignoring file argument.");
            AudioSource::Synthetic
        }
    } else {
        AudioSource::Synthetic
    };

    println!("🎵 Source: {:?}", source);

    // 4. Processing Loop
    let mut audio_stream = AudioStream::new(source);
    let mut time = 0.0;
    let dt = (WINDOW_SIZE as f32 / SAMPLE_RATE as f32) * 1000.0; // Time per window in ms
    let mut window_count = 0;
    let mut last_phi = 0.0; // Cache last Phi value

    println!("\nTime(s) |  Bass  |  Low   |  Mid   |  High  |   Φ    | Root Act | Leaf Act");
    println!("--------|--------|--------|--------|--------|--------|----------|----------");

    // Semantic Vectors for Frequency Bands (The "Qualia" of Pitch)
    // We create random vectors to represent "Bass-ness", "Treble-ness", etc.
    let band_vectors: Vec<RealHV> = (0..LEVELS)
        .map(|i| RealHV::random(HDC_DIMENSION, i as u64 * 1000))
        .collect();

    while let Some(samples) = audio_stream.next_window(WINDOW_SIZE) {
        window_count += 1;

        // A. Compute FFT
        let mut buffer: Vec<Complex<f32>> = samples.iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();

        // Pad if needed
        while buffer.len() < WINDOW_SIZE {
            buffer.push(Complex { re: 0.0, im: 0.0 });
        }

        fft.process(&mut buffer);

        // B. Extract Band Magnitudes (Logarithmic scaling)
        // Level 0 (Root) gets lowest freqs, Level 7 gets highest
        let spectrum: Vec<f32> = buffer.iter().take(WINDOW_SIZE/2).map(|c| c.norm()).collect();
        let energies = map_spectrum_to_bands(&spectrum, LEVELS);

        // C. Inject into Brain Layers
        // "The Ear of Dionysus" maps frequency bands directly to cortical layers
        inject_spectral_energy(&mut brain.root, &energies, &band_vectors, 0);

        // D. Step Simulation
        // We step the brain forward by the duration of the audio window
        brain.step(dt);

        // E. Measure & Visualize
        // OPTIMIZATION: Only compute heavy Phi metric every ~100ms (every 4th window)
        // The brain processes audio at 43Hz, but "consciousness" updates at ~10Hz
        if window_count % 4 == 0 {
            last_phi = brain.compute_hierarchical_phi();
        }

        let root_act = brain.root.state.norm() / (HDC_DIMENSION as f32).sqrt();

        // Find a leaf node activity
        let mut leaf = &brain.root;
        while let Some((left, _)) = &leaf.children {
            leaf = left;
        }
        let leaf_act = leaf.state.norm() / (HDC_DIMENSION as f32).sqrt();

        // Visualization
        print!("\r{:7.2} | ", time / 1000.0);
        for i in [0, 2, 4, 7] { // Visualize Bass, Low-Mid, Mid-High, High
             let e = energies[i];
             let bars = (e * 10.0).clamp(0.0, 8.0) as usize;
             let bar_str = "█".repeat(bars);
             print!("{:6} | ", bar_str); // Simple bar graph
        }
        // Use cached Phi for display smoothness
        print!("{:.4} |   {:.2}   |   {:.2}   ", last_phi, root_act, leaf_act);

        // Flush stdout
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        time += dt;
        if time > 20_000.0 { break; } // Run for 20 seconds
    }
    println!("\n\nDone.");
}

// =============================================================================
// HELPERS
// =============================================================================

#[derive(Debug)]
enum AudioSource {
    Synthetic,
    File(String),
}

struct AudioStream {
    source: AudioSource,
    t: usize,
    // For file playback
    #[cfg(feature = "hound")]
    reader: Option<hound::WavReader<std::io::BufReader<std::fs::File>>>,
    #[cfg(feature = "hound")]
    samples: Option<Box<dyn Iterator<Item = f32>>>,
}

impl AudioStream {
    fn new(source: AudioSource) -> Self {
        let mut stream = Self {
            source,
            t: 0,
            #[cfg(feature = "hound")]
            reader: None,
            #[cfg(feature = "hound")]
            samples: None,
        };

        if let AudioSource::File(ref _path) = stream.source {
            #[cfg(feature = "hound")]
            {
                match hound::WavReader::open(_path) {
                    Ok(reader) => {
                        let spec = reader.spec();
                        println!("Opened WAV: {} channels, {} Hz, {} bits", spec.channels, spec.sample_rate, spec.bits_per_sample);
                        // We need to move the reader into the struct, but WavReader isn't easily movable
                        // into an iterator without ownership issues in this simple struct.
                        // For this demo, we'll re-open or just read all.
                        // To keep it simple and streaming:
                        let samples = reader.into_samples::<f32>()
                            .map(|s| s.unwrap_or(0.0));
                        stream.samples = Some(Box::new(samples));
                    },
                    Err(e) => {
                        println!("Failed to open WAV: {}. Falling back to synthetic.", e);
                        stream.source = AudioSource::Synthetic;
                    }
                }
            }
            #[cfg(not(feature = "hound"))]
            {
                // Should not happen due to main check, but good for safety
            }
        }

        stream
    }

    fn next_window(&mut self, size: usize) -> Option<Vec<f32>> {
        match self.source {
            AudioSource::Synthetic => {
                // Generate a complex signal:
                // - Deep Bass Swell (0.5Hz modulation of 50Hz tone)
                // - Arpeggiator (Changes every 2s)
                // - High hats (Noise bursts)

                let mut window = Vec::with_capacity(size);

                for i in 0..size {
                    let t_global = (self.t + i) as f32 / SAMPLE_RATE as f32;

                    // 1. Bass (50Hz + modulation)
                    let bass = (t_global * 50.0 * 2.0 * PI).sin() * (t_global * 0.5 * 2.0 * PI).sin().abs();

                    // 2. Mids (Chord progression: C -> F -> G -> C)
                    let measure = (t_global / 2.0) % 4.0;
                    let freq = if measure < 1.0 { 261.6 } // C
                              else if measure < 2.0 { 349.2 } // F
                              else if measure < 3.0 { 392.0 } // G
                              else { 261.6 }; // C

                    let melody = (t_global * freq * 2.0 * PI).sin() * 0.5;

                    // 3. Highs (Rhythmic pulses at 4Hz)
                    let rhythm = (t_global * 4.0 * 2.0 * PI).sin().max(0.0).powi(4);
                    let noise: f32 = rand::random::<f32>() * 2.0 - 1.0;
                    let hats = noise * rhythm * 0.3;

                    window.push(bass + melody + hats);
                }
                self.t += size;
                Some(window)
            },
            AudioSource::File(_) => {
                #[cfg(feature = "hound")]
                {
                    if let Some(ref mut iter) = self.samples {
                        let window: Vec<f32> = iter.take(size).collect();
                        if window.len() < size { None } else { Some(window) }
                    } else {
                        None // Fallback if init failed
                    }
                }
                #[cfg(not(feature = "hound"))]
                {
                    None
                }
            }
        }
    }
}

/// Map FFT spectrum to N logarithmic bands
fn map_spectrum_to_bands(spectrum: &[f32], bands: usize) -> Vec<f32> {
    let mut energies = vec![0.0; bands];
    let num_bins = spectrum.len();

    // Logarithmic mapping: lower bands are narrower, higher are wider
    // We want to map 0..Nyquist to 0..7
    // Simple log approach:

    for (i, &mag) in spectrum.iter().enumerate() {
        if i == 0 { continue; } // Skip DC

        // Normalized freq (0.0 to 1.0)
        let freq_norm = i as f32 / num_bins as f32;

        // Map to band index (0 to bands-1) using log scale
        // log(freq)
        // We focus on audible range. Let's just use simple partitioning for the demo
        // Band 0: Bass
        // Band 7: Highs

        // Simple heuristic mapping
        let band_idx = (freq_norm.sqrt() * bands as f32).floor() as usize;
        if band_idx < bands {
            energies[band_idx] += mag;
        }
    }

    // Normalize energies
    for e in energies.iter_mut() {
        *e = (*e / 100.0).min(1.0); // Rough normalization
    }

    energies
}

/// Recursively inject energy into the network
///
/// Bass -> Root (Level 0)
/// ...
/// Highs -> Leaves (Level Max)
fn inject_spectral_energy(
    node: &mut CantorLtcNode,
    energies: &[f32],
    vectors: &[RealHV],
    current_level: usize
) {
    if current_level >= energies.len() { return; }

    let energy = energies[current_level];
    let signal_vector = vectors[current_level].clone();

    // Inject: We add the scaled signal vector to the node's state
    // In a real biological system, this would be sensory input current
    // Here we treat it as a "forced vibration"

    // We bind the signal to the node's current state to "color" it,
    // or just add it (bundle) to drive it.
    // Adding (Bundling) effectively drives the integrator.

    let input = signal_vector.scale(energy);

    // We simulate input injection by blending it into the state
    // This is a simplification; ideally we'd pass it to the step function as external input
    // But since step() takes one parent input, we'll modify the state directly before step()
    // to simulate "sensory transduction".

    node.state = RealHV::bundle(&[node.state.clone(), input]);

    // Recurse
    if let Some((ref mut left, ref mut right)) = node.children {
        inject_spectral_energy(left, energies, vectors, current_level + 1);
        inject_spectral_energy(right, energies, vectors, current_level + 1);
    }
}
