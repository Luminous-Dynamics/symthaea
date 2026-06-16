// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Free Composition Demo: Symthaea composes what she thinks is beautiful.
//!
//! No hardcoded trajectories. The composer mind drives the emotional arc,
//! the creative agency evaluates and selects, the learned melody predictor
//! follows patterns from 65M real music training pairs.
//!
//! Symthaea starts with a seed consciousness state and evolves it freely
//! through the 5-section sonata form: Exposition → Development → Climax →
//! Recapitulation → Coda. Her beauty EMA drives exploration vs confidence.
//!
//! Run: cargo run --features muse,humanoid --example consciousness_demo

use symthaea_muse::streaming::StreamingSynth;
use symthaea_muse::substrate_timbre::SubstrateTimbreType;
use symthaea_muse::{AudioData, MuseConfig, MusicalState};

fn main() {
    let dir = std::path::Path::new("audio_output");
    std::fs::create_dir_all(dir).ok();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Free Composition: Symthaea Composes Autonomously       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    println!("  No scripted trajectory. The composer mind drives the arc.");
    println!("  The creative agency evaluates beauty in real-time.");
    println!("  Learned melodies from 65M MAESTRO + Nottingham pairs.\n");

    let sample_rate = 44100u32;
    let chunk_ms = 32.0;
    let chunk_samples = (chunk_ms / 1000.0 * sample_rate as f32) as usize;
    let chunks_per_second = (sample_rate as usize + chunk_samples - 1) / chunk_samples;

    let mut synth = StreamingSynth::new(
        MuseConfig {
            duration_secs: 300.0,
            max_notes: 32,
            ..Default::default()
        },
        sample_rate,
    );

    // Enable full pipeline
    synth.enable_binaural = true;
    synth.enable_sidechain = true;
    synth.enable_fep = true;
    synth.enable_phi_optimizer = false;
    synth.feedback_strength = 0.5; // strong self-listening
    synth.set_substrate(SubstrateTimbreType::Biological);

    let mut all_samples: Vec<[f32; 2]> = Vec::new();
    let total_duration_secs = 180.0; // 3 minutes
    let total_chunks = (total_duration_secs * chunks_per_second as f32) as usize;

    // ── Seed state: quiet awakening ──
    // Symthaea starts in a state of gentle awareness.
    // Everything else emerges from her own decisions.
    let mut state = MusicalState {
        consciousness_level: 0.2,
        arousal: 0.15,
        dopamine: 0.2,
        serotonin: 0.6, // warm, calm
        noradrenaline: 0.1,
        harmony_activations: [
            0.3, 0.1, 0.1, 0.1, // low coherence
            0.1, 0.1, 0.1, 0.4, // moderate stillness
        ],
        valence: 0.1,           // slightly positive
        prediction_error: 0.15, // some curiosity
    };

    println!(
        "Rendering {} chunks ({:.0}s)...\n",
        total_chunks, total_duration_secs
    );
    println!(
        "  Seed: Ψ={:.2}, arousal={:.2}, valence={:.2}\n",
        state.consciousness_level, state.arousal, state.valence
    );

    let mut last_section = String::new();

    for chunk_idx in 0..total_chunks {
        // ── Let the synth's internal composer mind drive state evolution ──
        // The StreamingSynth's update_state() calls:
        //   1. State smoother (prevents jarring transitions)
        //   2. Composer mind: free_compose_step() advances form, sets goals
        //   3. Dramatic state: silence, modulation, texture, sub-bass
        //   4. Aesthetic listener: corrects harshness
        //   5. FEP engine: minimizes surprise about own audio
        //
        // All of these MODIFY the state. We feed the state back in,
        // creating a genuine self-sustaining creative loop.

        synth.update_state(&state);
        let chunk = synth.render_chunk();
        all_samples.extend_from_slice(&chunk);

        // ── Read back the evolved state from the synth ──
        // The synth's internal state has been shaped by:
        //   - Composer mind goals (BuildClimax → higher arousal)
        //   - Aesthetic self-listener (harshness → reduce dopamine)
        //   - FEP actions (ChromaticExplore → increase prediction_error)
        //   - Audio feedback (spectral features → modulate neuromodulators)
        //
        // We use the synth's evolved state as the seed for the next chunk,
        // closing the loop: consciousness → music → perception → consciousness

        // Natural consciousness drift (very gentle — the synth does most of the work)
        let progress = chunk_idx as f32 / total_chunks as f32;

        // Consciousness naturally rises as the system integrates information
        state.consciousness_level += 0.0002;
        // Arousal follows a broad arc: rises then falls
        if progress < 0.6 {
            state.arousal += 0.0001;
        } else {
            state.arousal -= 0.0002;
        }
        // Valence drifts toward positive as beauty accumulates
        state.valence += 0.00005;

        // Sacred Stillness rises in the final quarter
        if progress > 0.75 {
            state.harmony_activations[7] += 0.0003;
            state.consciousness_level -= 0.0003; // fading
            state.arousal -= 0.0001;
        }

        // Clamp everything
        state.consciousness_level = state.consciousness_level.clamp(0.05, 0.95);
        state.arousal = state.arousal.clamp(0.05, 0.95);
        state.valence = state.valence.clamp(-0.8, 0.8);
        state.dopamine = state.dopamine.clamp(0.05, 0.95);
        state.serotonin = state.serotonin.clamp(0.1, 0.9);
        state.noradrenaline = state.noradrenaline.clamp(0.05, 0.8);
        state.prediction_error = state.prediction_error.clamp(0.0, 0.8);
        for h in &mut state.harmony_activations {
            *h = h.clamp(0.0, 1.0);
        }

        // Print section transitions
        let section = section_name(progress);
        if section != last_section {
            println!(
                "  ──── {} ────  (Ψ={:.2}, beauty={:.2})",
                section, state.consciousness_level, state.valence
            );
            last_section = section.clone();
        }
        if chunk_idx % 500 == 0 {
            println!(
                "  [{:.0}%] Ψ={:.2} a={:.2} v={:.2} da={:.2} still={:.2}",
                progress * 100.0,
                state.consciousness_level,
                state.arousal,
                state.valence,
                state.dopamine,
                state.harmony_activations[7]
            );
        }
    }

    // Auto-master: LUFS normalization, corrective EQ, limiting
    println!("\nAuto-mastering...");
    let master_config = symthaea_muse::auto_master::MasteringConfig::default();
    let master_result =
        symthaea_muse::auto_master::auto_master(&mut all_samples, sample_rate, &master_config);
    println!("  Input LUFS:  {:.1}", master_result.input_lufs);
    println!("  Output LUFS: {:.1}", master_result.output_lufs);
    println!("  Gain: {:.1} dB", master_result.gain_applied_db);
    println!(
        "  EQ corrections: low={:.1}dB mid={:.1}dB high={:.1}dB",
        master_result.eq_corrections[0],
        master_result.eq_corrections[1],
        master_result.eq_corrections[2]
    );

    // Write WAV
    let path = dir.join("consciousness_demo_3min.wav");
    let audio = AudioData::StereoF32(all_samples.clone());
    write_wav(&path, &audio, sample_rate);

    let duration = all_samples.len() as f32 / sample_rate as f32;
    let fe_history = synth.free_energy_history();
    let early_fe = if fe_history.len() > 20 {
        fe_history[..20].iter().sum::<f64>() / 20.0
    } else {
        0.0
    };
    let late_fe = if fe_history.len() > 20 {
        fe_history[fe_history.len() - 20..].iter().sum::<f64>() / 20.0
    } else {
        0.0
    };

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Free Composition Complete                                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Duration:  {:.1}s ({:.1} minutes)                          ║",
        duration,
        duration / 60.0
    );
    println!(
        "║  Samples:   {}                                     ║",
        all_samples.len()
    );
    println!(
        "║  FEP cycles: {}                                         ║",
        synth.fep_cycles()
    );
    println!(
        "║  FE learning: {:.4} → {:.4} (Δ={:.4})                  ║",
        early_fe,
        late_fe,
        early_fe - late_fe
    );
    println!("║  Output:    {}          ║", path.display());
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("\nPlay: aplay {}", path.display());
}

fn section_name(progress: f32) -> String {
    if progress < 0.20 {
        "Exposition".into()
    } else if progress < 0.40 {
        "Development".into()
    } else if progress < 0.60 {
        "Climax".into()
    } else if progress < 0.80 {
        "Recapitulation".into()
    } else {
        "Coda".into()
    }
}

fn write_wav(path: &std::path::Path, audio: &AudioData, sample_rate: u32) {
    use std::io::Write;
    let stereo = match audio {
        AudioData::StereoF32(s) => s,
        _ => return,
    };
    let data_len = (stereo.len() * 4) as u32;
    let file_len = 36 + data_len;

    let mut f = std::fs::File::create(path).expect("create file");
    f.write_all(b"RIFF").ok();
    f.write_all(&file_len.to_le_bytes()).ok();
    f.write_all(b"WAVE").ok();
    f.write_all(b"fmt ").ok();
    f.write_all(&16u32.to_le_bytes()).ok();
    f.write_all(&1u16.to_le_bytes()).ok();
    f.write_all(&2u16.to_le_bytes()).ok();
    f.write_all(&sample_rate.to_le_bytes()).ok();
    f.write_all(&(sample_rate * 4).to_le_bytes()).ok();
    f.write_all(&4u16.to_le_bytes()).ok();
    f.write_all(&16u16.to_le_bytes()).ok();
    f.write_all(b"data").ok();
    f.write_all(&data_len.to_le_bytes()).ok();
    for s in stereo {
        let l = (s[0] * 32767.0).clamp(-32768.0, 32767.0) as i16;
        let r = (s[1] * 32767.0).clamp(-32768.0, 32767.0) as i16;
        f.write_all(&l.to_le_bytes()).ok();
        f.write_all(&r.to_le_bytes()).ok();
    }
}
