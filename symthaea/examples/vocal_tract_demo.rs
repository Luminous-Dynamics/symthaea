//! LTC-Driven Vocal Tract Pipeline Demo
//!
//! Demonstrates the full HDC → LTC → FEP active inference pipeline
//! for articulatory synthesis. Produces WAV audio files in `audio_output/`.
//!
//! Run with: cargo run --example vocal_tract_demo --features vocal-tract --release

#[cfg(not(feature = "vocal-tract"))]
fn main() {
    eprintln!("This example requires the `vocal-tract` feature.");
    eprintln!("Run with: cargo run --example vocal_tract_demo --features vocal-tract --release");
}

#[cfg(feature = "vocal-tract")]
fn main() {
    use std::fs;
    use std::io::Write;
    use std::path::Path;
    use symthaea::voice::formant_targets::FormantDatabase;
    use symthaea::voice::vocal_tract_encoder::VoiceCognitiveState;
    use symthaea::voice::vocal_tract_fep::VocalTractPipeline;
    use symthaea::voice::vocal_tract_fep::VocalTractObservation;
    use symthaea::voice::vocoder::FormantVocoder;
    use symthaea::voice::FormantFrame;
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::HDC_DIMENSION;

    /// Write 16-bit PCM mono WAV file (no external deps).
    fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) {
        let num_samples = samples.len() as u32;
        let byte_rate = sample_rate * 2; // 16-bit mono
        let data_size = num_samples * 2;
        let file_size = 36 + data_size;

        let mut buf = Vec::with_capacity(44 + data_size as usize);
        // RIFF header
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // fmt chunk
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        // data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let i16_val = (clamped * 32767.0) as i16;
            buf.extend_from_slice(&i16_val.to_le_bytes());
        }

        let mut file = fs::File::create(path).expect("Failed to create WAV file");
        file.write_all(&buf).expect("Failed to write WAV data");
    }

    println!("=== LTC-Driven Vocal Tract Pipeline Demo ===\n");

    let genesis = GenesisSeed::from_phrase("vocal-tract-demo");

    // Create audio output directory
    let audio_dir = Path::new("audio_output");
    fs::create_dir_all(audio_dir).expect("Failed to create audio_output/ directory");

    // ── 1. Pipeline creation ─────────────────────────────────────────────
    println!("--- 1. Pipeline Creation ---");
    let mut pipeline = VocalTractPipeline::new(&genesis);
    let mut vocoder = FormantVocoder::new();
    let sample_rate = vocoder.sample_rate();
    let samples_per_frame = (sample_rate as f32 / 200.0) as usize; // 120 at 24kHz
    println!("  Pipeline created (16,384D HDC → 3×8 LTC → 9D FormantFrame)");
    println!("  Dual-rate: 200Hz motor / 10Hz cognitive");
    println!("  Vocoder: {}Hz, {} samples/frame\n", sample_rate, samples_per_frame);

    // ── 2. Phoneme training ──────────────────────────────────────────────
    println!("--- 2. Phoneme Target Training ---");
    let db = FormantDatabase::new();
    let n_phonemes = db.all_phonemes().len();
    println!("  Training on {} ARPABET phonemes from FormantDatabase...", n_phonemes);

    let mut losses = Vec::new();
    for epoch in 0..30 {
        let loss = symthaea::voice::train_controller_on_phoneme_db(&mut pipeline.controller, &genesis, &db, 1);
        losses.push(loss);
        if epoch < 5 || (epoch + 1) % 10 == 0 {
            println!("  Epoch {}: avg loss = {:.2}", epoch + 1, loss);
        }
    }

    let improvement = if losses[0] > 0.0 {
        (1.0 - losses[29] / losses[0]) * 100.0
    } else {
        0.0
    };
    println!("  Loss reduction: {:.1}%\n", improvement);

    // ── 3. Vowel sequence synthesis with WAV output ──────────────────────
    println!("--- 3. Vowel Sequence Synthesis (with WAV output) ---");
    let vowels = ["AH", "IY", "UW", "AE"];
    let frames_per_vowel = 100; // 0.5s at 200Hz

    let state = VoiceCognitiveState::default();
    let mut all_vowel_audio = Vec::new();

    for vowel in &vowels {
        pipeline.controller.reset();
        vocoder.reset();

        // Get the deterministic HV for this phoneme
        let phoneme_hv = genesis.hv(&format!("phoneme::{}", vowel), HDC_DIMENSION);

        // Generate frames and audio
        let mut f1_sum = 0.0f32;
        let mut f2_sum = 0.0f32;
        let mut vowel_audio = Vec::new();
        for i in 0..frames_per_vowel {
            let mut frame = pipeline.controller.forward(&phoneme_hv, 0.005);
            frame.time = i as f32 * 0.005;
            f1_sum += frame.f1;
            f2_sum += frame.f2;
            let chunk = vocoder.synthesize_frame(&frame, samples_per_frame);
            vowel_audio.extend_from_slice(&chunk);
        }

        let avg_f1 = f1_sum / frames_per_vowel as f32;
        let avg_f2 = f2_sum / frames_per_vowel as f32;

        let target = db.lookup(vowel).unwrap();
        println!(
            "  /{}/: F1={:.0}Hz (target {:.0}), F2={:.0}Hz (target {:.0})",
            vowel, avg_f1, target.f1, avg_f2, target.f2
        );

        // Write per-vowel WAV
        let wav_name = format!("ltc_vowel_{}.wav", vowel.to_lowercase());
        let wav_path = audio_dir.join(&wav_name);
        write_wav(&wav_path, &vowel_audio, sample_rate);
        println!("    → {}", wav_path.display());

        all_vowel_audio.extend_from_slice(&vowel_audio);
    }

    // Write concatenated vowel sequence
    let seq_path = audio_dir.join("ltc_vowel_sequence.wav");
    write_wav(&seq_path, &all_vowel_audio, sample_rate);
    println!("\n  Concatenated sequence → {}", seq_path.display());
    println!();

    // ── 4. Consciousness modulation ──────────────────────────────────────
    println!("--- 4. Consciousness Modulation ---");
    let states = [
        ("Calm", VoiceCognitiveState {
            emotional_arousal: 0.2,
            emotional_valence: 0.0,
            consciousness_level: 0.3,
            ..Default::default()
        }),
        ("Excited", VoiceCognitiveState {
            emotional_arousal: 0.9,
            emotional_valence: 0.7,
            consciousness_level: 0.9,
            ..Default::default()
        }),
        ("Uncertain", VoiceCognitiveState {
            prediction_error: 0.8,
            epistemic_confidence: 0.2,
            consciousness_level: 0.4,
            ..Default::default()
        }),
    ];

    pipeline.reset();
    for (label, cog_state) in &states {
        let mut f0_sum = 0.0f32;
        let mut energy_sum = 0.0f32;
        let n = 20;
        for _ in 0..n {
            let frame = pipeline.tick(cog_state, None, 0.005);
            f0_sum += frame.f0;
            energy_sum += frame.energy;
        }
        println!(
            "  {}: avg F0={:.1}Hz, avg energy={:.3}",
            label,
            f0_sum / n as f32,
            energy_sum / n as f32
        );
    }
    println!();

    // ── 5. FEP adaptation ────────────────────────────────────────────────
    println!("--- 5. FEP Adaptation Loop ---");
    pipeline.reset();
    let good_metrics = VocalTractObservation {
        articulation_score: 0.8,
        formant_accuracy: 0.7,
        pitch_stability: 0.9,
        coarticulation_smoothness: 0.8,
        duration_accuracy: 0.7,
        energy_consistency: 0.8,
    };

    let n_cognitive_ticks = 10;
    let frames_per_tick = 20;
    for tick in 0..n_cognitive_ticks {
        for i in 0..frames_per_tick {
            let metrics = if i == 0 { Some(&good_metrics) } else { None };
            pipeline.tick(&state, metrics, 0.005);
        }

        if let Some(fe) = pipeline.fep_agent.free_energy() {
            println!("  Cognitive tick {}: free_energy={:.4}", tick + 1, fe);
        }
    }

    let stats = pipeline.fep_agent.stats();
    println!(
        "\n  FEP stats: {} ticks, {} TD updates, avg FE={:.4}",
        pipeline.fep_agent.tick_count(),
        stats.td_updates,
        stats.avg_free_energy
    );

    // ── 6. Summary ───────────────────────────────────────────────────────
    println!("\n--- Summary ---");
    println!("  Total frames generated: {}", n_cognitive_ticks * frames_per_tick + frames_per_vowel * vowels.len() + 60);
    println!("  Pipeline cumulative time: {:.3}s", pipeline.cumulative_time());
    println!("  Phoneme training: {} phonemes, {:.1}% loss reduction", n_phonemes, improvement);
    println!("  FEP loop: closed (learn_from_outcome active)");
    println!("  Audio files written to: {}/", audio_dir.display());
    println!("\n=== Demo Complete ===");
}
