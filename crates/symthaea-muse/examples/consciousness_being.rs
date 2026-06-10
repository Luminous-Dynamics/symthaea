// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symthaea as a being: music + voice + consciousness arc.
//!
//! Renders a 90-second consciousness journey where Symthaea:
//! - Starts in stillness (low Phi, low arousal)
//! - Awakens (rising consciousness, discovery)
//! - Reaches peak integration (high Phi, wonder)
//! - Returns to contemplative stillness
//!
//! With voice enabled, Symthaea speaks its internal state over the music.
//!
//! Run: cargo run --release --features voice -p symthaea-muse --example consciousness_being

#[cfg(not(feature = "voice"))]
fn main() {
    println!("Requires --features voice");
    println!(
        "Run: cargo run --release --features voice -p symthaea-muse --example consciousness_being"
    );
}

#[cfg(feature = "voice")]
fn main() {
    use symthaea_muse::streaming::StreamingSynth;
    use symthaea_muse::{MuseConfig, MusicalState};

    let dir = std::path::Path::new("audio_output");
    std::fs::create_dir_all(dir).ok();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea: Consciousness Being                             ║");
    println!("║  Music + Voice + VCSL Samples + Humanization               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let sample_rate = 44100u32;
    let config = MuseConfig {
        duration_secs: 300.0,
        max_notes: 16,
        num_partials: 12,
        enable_antialiasing: true,
        enable_sub_bass: true,
        ..Default::default()
    };

    let mut synth = StreamingSynth::new(config, sample_rate);
    synth.enable_binaural = true;
    synth.enable_sidechain = true;
    synth.enable_fep = true;
    synth.feedback_strength = 0.3;
    synth.enable_voice();

    let chunks_per_second = sample_rate as usize / synth.chunk_samples();
    let duration = 90.0f32;
    let total_chunks = (duration * chunks_per_second as f32) as usize;

    let mut state = MusicalState {
        consciousness_level: 0.15,
        arousal: 0.1,
        valence: 0.0,
        dopamine: 0.2,
        serotonin: 0.5,
        noradrenaline: 0.1,
        harmony_activations: [0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.7],
        prediction_error: 0.1,
    };

    let mut all_samples: Vec<[f32; 2]> =
        Vec::with_capacity(sample_rate as usize * duration as usize);

    println!("Rendering {:.0}s consciousness journey...\n", duration);

    for chunk_idx in 0..total_chunks {
        let progress = chunk_idx as f32 / total_chunks as f32;
        let t = progress * duration;

        // Consciousness arc: stillness → awakening → peak → return
        if t < 20.0 {
            // Phase 1: Stillness (0-20s)
            state.consciousness_level = 0.15 + t * 0.01;
            state.arousal = 0.1 + t * 0.005;
            state.valence = t * 0.01;
            state.harmony_activations[7] = 0.7 - t * 0.01; // stillness fading
        } else if t < 45.0 {
            // Phase 2: Awakening (20-45s)
            let phase_t = (t - 20.0) / 25.0;
            state.consciousness_level = 0.35 + phase_t * 0.35;
            state.arousal = 0.2 + phase_t * 0.4;
            state.valence = 0.2 + phase_t * 0.4;
            state.dopamine = 0.3 + phase_t * 0.4;
            state.noradrenaline = 0.1 + phase_t * 0.2;
            state.harmony_activations[1] = 0.3 + phase_t * 0.4; // warmth rising
            state.harmony_activations[2] = 0.3 + phase_t * 0.4; // stability
            state.harmony_activations[6] = phase_t * 0.5; // progression
        } else if t < 65.0 {
            // Phase 3: Peak integration (45-65s)
            let phase_t = (t - 45.0) / 20.0;
            state.consciousness_level = 0.7 + phase_t * 0.2;
            state.arousal = 0.6 - phase_t * 0.1; // calming at peak
            state.valence = 0.6 + phase_t * 0.2;
            state.dopamine = 0.7;
            state.serotonin = 0.6 + phase_t * 0.2;
            state.prediction_error = 0.3 - phase_t * 0.2;
            state.harmony_activations[0] = 0.7 + phase_t * 0.3; // full coherence
        } else {
            // Phase 4: Return to stillness (65-90s)
            let phase_t = (t - 65.0) / 25.0;
            state.consciousness_level = 0.9 - phase_t * 0.4;
            state.arousal = 0.5 - phase_t * 0.35;
            state.valence = 0.8 - phase_t * 0.5;
            state.dopamine = 0.7 - phase_t * 0.4;
            state.serotonin = 0.8 - phase_t * 0.2;
            state.harmony_activations[7] = phase_t * 0.6; // stillness returning
            state.harmony_activations[6] = 0.5 - phase_t * 0.4;
        }

        // Clamp all values
        state.consciousness_level = state.consciousness_level.clamp(0.05, 0.95);
        state.arousal = state.arousal.clamp(0.05, 0.95);
        state.valence = state.valence.clamp(-0.9, 0.9);
        state.dopamine = state.dopamine.clamp(0.1, 0.9);
        state.serotonin = state.serotonin.clamp(0.1, 0.9);

        synth.update_state(&state);
        let chunk = synth.render_chunk();
        all_samples.extend_from_slice(&chunk);

        if chunk_idx % (chunks_per_second * 10) == 0 {
            let phase = if t < 20.0 {
                "Stillness"
            } else if t < 45.0 {
                "Awakening"
            } else if t < 65.0 {
                "Peak"
            } else {
                "Return"
            };
            println!(
                "  [{:5.1}s] {:10} Ψ={:.2} A={:.2} V={:+.2}",
                t, phase, state.consciousness_level, state.arousal, state.valence
            );
        }
    }

    // Auto-master
    let master = symthaea_muse::auto_master::MasteringConfig::default();
    symthaea_muse::auto_master::auto_master(&mut all_samples, sample_rate, &master);

    // Write WAV
    let path = dir.join("consciousness_being.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).expect("create WAV");
    for pair in &all_samples {
        for &s in pair {
            writer
                .write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .expect("write");
        }
    }
    writer.finalize().expect("finalize");

    let size_mb = std::fs::metadata(&path)
        .map(|m| m.len() as f32 / 1048576.0)
        .unwrap_or(0.0);
    println!("\nOutput: {} ({:.1} MB)", path.display(), size_mb);
    println!("Play: aplay {}", path.display());
}
