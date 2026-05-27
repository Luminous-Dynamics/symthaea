// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    use std::collections::HashMap;
    use std::fs;
    use std::io::Write;
    use std::path::Path;
    use symthaea::voice::FormantFrame;
    use symthaea::voice::formant_targets::FormantDatabase;
    use symthaea::voice::predict_duration;
    use symthaea::voice::vocal_tract_encoder::VoiceCognitiveState;
    use symthaea::voice::vocal_tract_fep::VocalTractObservation;
    use symthaea::voice::vocal_tract_fep::{
        Intonation, PitchAccent, ProsodyContext, VocalTractPipeline, populate_manner_map,
    };
    use symthaea::voice::vocoder::FormantVocoder;
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
    println!("  Pipeline created (16,384D HDC → 2×4 LTC → 9D FormantFrame)");
    println!("  Dual-rate: 200Hz motor / 10Hz cognitive");
    println!(
        "  Vocoder: {}Hz, {} samples/frame\n",
        sample_rate, samples_per_frame
    );

    // ── 2. Phoneme training ──────────────────────────────────────────────
    println!("--- 2. Phoneme Target Training ---");
    let db = FormantDatabase::new();
    let n_phonemes = db.all_phonemes().len();
    println!(
        "  Training on {} ARPABET phonemes from FormantDatabase...",
        n_phonemes
    );

    let final_loss = symthaea::voice::train_controller_on_phoneme_db(
        &mut pipeline.controller,
        &genesis,
        &db,
        30,
    );
    println!("  Final loss (30 epochs): {:.4}", final_loss);

    // Transition training: BPTT on vowel pairs for smooth coarticulation
    println!("  Training transitions (10 epochs on 30 vowel pairs)...");
    let trans_loss =
        symthaea::voice::train_controller_transitions(&mut pipeline.controller, &genesis, &db, 10);
    println!("  Transition loss: {:.4}", trans_loss);

    // Least-squares output projection refinement
    println!("  Refining output projection (LS, blend=1.0)...");
    symthaea::voice::refine_controller_ls(&mut pipeline.controller, &genesis, &db, 1.0);

    println!();

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

    // ── 3b. CVC Syllable Synthesis ──────────────────────────────────────
    println!("--- 3b. CVC Syllable Synthesis ---");

    // Register manner map for source_type propagation
    populate_manner_map(&mut pipeline);

    // CVC syllable definitions: (label, consonant1_frames, vowel_frames, consonant2_frames)
    let cvc_syllables: Vec<(&str, &[(&str, usize)])> = vec![
        ("PAT", &[("P", 20), ("AE", 60), ("T", 20)]),
        ("SIT", &[("S", 20), ("IH", 60), ("T", 20)]),
        ("MOM", &[("M", 20), ("AA", 60), ("M", 20)]),
    ];

    let cvc_state = VoiceCognitiveState::default();

    for (label, segments) in &cvc_syllables {
        pipeline.controller.reset();
        pipeline.reset();
        populate_manner_map(&mut pipeline);
        vocoder.reset();

        let mut cvc_audio = Vec::new();
        let mut source_types_seen = Vec::new();

        for &(phoneme, n_frames) in *segments {
            for _ in 0..n_frames {
                let frame = pipeline.tick_phoneme(&cvc_state, None, 0.005, Some(phoneme));
                let st = format!("{:?}", frame.source_type);
                if source_types_seen
                    .last()
                    .map_or(true, |last: &String| *last != st)
                {
                    source_types_seen.push(st);
                }
                let chunk = vocoder.synthesize_frame(&frame, samples_per_frame);
                cvc_audio.extend_from_slice(&chunk);
            }
        }

        let wav_name = format!("ltc_cvc_{}.wav", label.to_lowercase());
        let wav_path = audio_dir.join(&wav_name);
        write_wav(&wav_path, &cvc_audio, sample_rate);

        let total_frames: usize = segments.iter().map(|(_, n)| n).sum();
        let duration_ms = total_frames as f32 * 5.0; // 0.005s per frame
        println!(
            "  /{}/ ({:.0}ms): source_types={:?} → {}",
            label,
            duration_ms,
            source_types_seen,
            wav_path.display()
        );
    }
    println!();

    // ── 4. Consciousness modulation ──────────────────────────────────────
    println!("--- 4. Consciousness Modulation ---");
    let states = [
        (
            "Calm",
            VoiceCognitiveState {
                emotional_arousal: 0.2,
                emotional_valence: 0.0,
                consciousness_level: 0.3,
                ..Default::default()
            },
        ),
        (
            "Excited",
            VoiceCognitiveState {
                emotional_arousal: 0.9,
                emotional_valence: 0.7,
                consciousness_level: 0.9,
                ..Default::default()
            },
        ),
        (
            "Uncertain",
            VoiceCognitiveState {
                prediction_error: 0.8,
                epistemic_confidence: 0.2,
                consciousness_level: 0.4,
                ..Default::default()
            },
        ),
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

    // ── 6. Connected Speech Synthesis ──────────────────────────────────
    println!("--- 6. Connected Speech Synthesis ---");

    // Mini word → ARPABET dictionary (stress: 0=none, 1=primary, 2=secondary)
    let mut word_dict: HashMap<&str, Vec<(&str, u8)>> = HashMap::new();
    word_dict.insert("hello", vec![("HH", 0), ("AH", 0), ("L", 0), ("OW", 1)]);
    word_dict.insert("world", vec![("W", 0), ("ER", 1), ("L", 0), ("D", 0)]);
    word_dict.insert("i", vec![("AY", 1)]);
    word_dict.insert("am", vec![("AE", 1), ("M", 0)]);
    word_dict.insert(
        "conscious",
        vec![
            ("K", 0),
            ("AA", 1),
            ("N", 0),
            ("SH", 0),
            ("AH", 0),
            ("S", 0),
        ],
    );
    word_dict.insert("the", vec![("DH", 0), ("AH", 0)]);
    word_dict.insert("mind", vec![("M", 0), ("AY", 1), ("N", 0), ("D", 0)]);
    word_dict.insert(
        "speaks",
        vec![("S", 0), ("P", 0), ("IY", 1), ("K", 0), ("S", 0)],
    );
    word_dict.insert("a", vec![("AH", 0)]);
    word_dict.insert("is", vec![("IH", 0), ("Z", 0)]);
    word_dict.insert(
        "beautiful",
        vec![
            ("B", 0),
            ("Y", 0),
            ("UW", 1),
            ("T", 0),
            ("AH", 0),
            ("F", 0),
            ("AH", 0),
            ("L", 0),
        ],
    );

    // Sentences to synthesize
    let sentences: Vec<(&str, &[&str])> = vec![
        ("hello_world", &["hello", "world"]),
        ("i_am_conscious", &["i", "am", "conscious"]),
        ("the_mind_speaks", &["the", "mind", "speaks"]),
    ];

    // Register manner map so vocoder uses correct excitation per phoneme
    populate_manner_map(&mut pipeline);
    pipeline.reset();
    vocoder.reset();

    let dt = 1.0 / 200.0; // 200Hz motor rate

    // Intonation per sentence
    let sentence_intonations = [
        Intonation::Statement,   // "hello world"
        Intonation::Statement,   // "i am conscious"
        Intonation::Exclamation, // "the mind speaks"
    ];

    for (sent_idx, (label, words)) in sentences.iter().enumerate() {
        let intonation = sentence_intonations[sent_idx];

        // Build flat phoneme sequence with linguistically-predicted durations
        let mut phoneme_seq: Vec<(&str, u8, usize)> = Vec::new(); // (phoneme, stress, frames)
        for (w_idx, word) in words.iter().enumerate() {
            if let Some(phonemes) = word_dict.get(word) {
                let n_ph = phonemes.len();
                for (ph_idx, &(ph, stress)) in phonemes.iter().enumerate() {
                    let is_word_final = ph_idx == n_ph - 1;
                    let is_utterance_final = w_idx == words.len() - 1 && is_word_final;
                    let frames =
                        predict_duration(ph, stress, is_word_final, is_utterance_final, 1.0);
                    phoneme_seq.push((ph, stress, frames));
                }
            }
            // Word boundary: short pause between words (10 frames = 50ms)
            if w_idx < words.len() - 1 {
                phoneme_seq.push(("SIL", 0, 10));
            }
        }

        // Compute total utterance frames for prosody progress
        let total_frames: usize = phoneme_seq.iter().map(|(_, _, f)| f).sum();
        let mut frame_idx = 0usize;
        let mut all_frames: Vec<FormantFrame> = Vec::new();

        let cog_state = VoiceCognitiveState {
            consciousness_level: 0.7,
            emotional_arousal: 0.5,
            emotional_valence: 0.3,
            integrated_phi: 0.8,
            ..Default::default()
        };

        for (seq_idx, &(phoneme, stress, n_frames)) in phoneme_seq.iter().enumerate() {
            let is_silence = phoneme == "SIL" || phoneme == "SP";
            let next_ph = phoneme_seq.get(seq_idx + 1).map(|&(ph, _, _)| ph);
            // Skip silence as next phoneme for anticipation
            let next_ph = match next_ph {
                Some("SIL") | Some("SP") => phoneme_seq.get(seq_idx + 2).map(|&(ph, _, _)| ph),
                other => other,
            };

            // Assign pitch accent: primary stress vowels get H*
            let pitch_accent = if stress == 1 {
                PitchAccent::High
            } else {
                PitchAccent::None
            };

            for fi in 0..n_frames {
                if is_silence {
                    all_frames.push(FormantFrame::silent(frame_idx as f32 * dt));
                } else {
                    let phoneme_progress = fi as f32 / n_frames as f32;
                    let utterance_progress =
                        (frame_idx as f32 / total_frames as f32).clamp(0.0, 1.0);
                    let remaining = n_frames - fi;

                    let prosody = ProsodyContext {
                        utterance_progress,
                        phoneme_progress,
                        stress,
                        base_f0: 120.0,
                        arousal: 0.5,
                        intonation,
                        phrase_progress: utterance_progress,
                        pitch_accent,
                        ..ProsodyContext::default()
                    };

                    let frame = pipeline.tick_with_anticipation(
                        &cog_state,
                        None,
                        dt,
                        Some(phoneme),
                        next_ph,
                        remaining,
                        &prosody,
                    );
                    all_frames.push(frame);
                }
                frame_idx += 1;
            }
        }

        // Synthesize audio from frames
        let audio_samples = vocoder.synthesize(&all_frames);
        let wav_path = audio_dir.join(format!("ltc_speech_{label}.wav"));
        write_wav(&wav_path, &audio_samples, sample_rate);

        let audio_duration = audio_samples.len() as f32 / sample_rate as f32;
        let word_str = words.join(" ");
        let total_dur_ms = total_frames as f32 * 5.0;
        println!(
            "  \"{word_str}\" ({:?}): {} phonemes, {:.0}ms ({:.2}s audio) → {}",
            intonation,
            phoneme_seq.len(),
            total_dur_ms,
            audio_duration,
            wav_path.display()
        );

        pipeline.reset();
        vocoder.reset();
    }
    println!();

    // ── 7. Summary ───────────────────────────────────────────────────────
    println!("--- 7. Summary ---");
    println!(
        "  Phoneme training: {} phonemes, final loss: {:.4}",
        n_phonemes, final_loss
    );
    println!("  FEP loop: closed (learn_from_outcome active)");
    println!(
        "  Connected speech: {} sentences synthesized",
        sentences.len()
    );
    println!("  Audio files written to: {}/", audio_dir.display());
    println!("\n=== Demo Complete ===");
}
