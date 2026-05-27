// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hi-Fi Audio Validation Suite
//!
//! Run: cargo run --features muse,humanoid --example validate_hifi

use symthaea_muse::{
    AudioData, MelodyMode, MuseConfig, MusicalState, OutputFormat, ReverbConfig, compose,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          Hi-Fi Audio Validation Suite                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut pass = 0u32;
    let mut fail = 0u32;

    // ─── 1. A/B Comparison ──────────────────────────────────────────────────
    println!("━━━ 1. A/B Listening Comparison ━━━");
    let state = MusicalState {
        consciousness_level: 0.8,
        arousal: 0.5,
        dopamine: 0.6,
        serotonin: 0.4,
        noradrenaline: 0.3,
        harmony_activations: [0.7, 0.5, 0.6, 0.4, 0.3, 0.5, 0.6, 0.3],
        valence: 0.3,
        prediction_error: 0.2,
    };
    let old_config = MuseConfig {
        duration_secs: 4.0,
        max_notes: 16,
        output_format: OutputFormat::Mono16,
        num_partials: 4,
        enable_antialiasing: false,
        melody_mode: MelodyMode::Classic,
        cfc_layer_sizes: vec![4, 4],
        ..Default::default()
    };
    let new_config = MuseConfig {
        duration_secs: 4.0,
        max_notes: 16,
        melody_mode: MelodyMode::Neural,
        ..Default::default()
    };
    let old_comp = compose(&old_config, &state, 42);
    let new_comp = compose(&new_config, &state, 42);
    let old_rms = audio_rms(&old_comp.audio);
    let new_rms = audio_rms(&new_comp.audio);
    println!(
        "  Old (4 partial, mono): {} samples, RMS={:.4}",
        old_comp.audio.len(),
        old_rms
    );
    println!(
        "  New (8 partial, stereo): {} samples, RMS={:.4}",
        new_comp.audio.len(),
        new_rms
    );
    check(&mut pass, &mut fail, "Old produces audio", old_rms > 0.001);
    check(&mut pass, &mut fail, "New produces audio", new_rms > 0.001);
    check(
        &mut pass,
        &mut fail,
        "New is stereo",
        matches!(new_comp.audio, AudioData::StereoF32(_)),
    );
    println!(
        "  Notes: old={}, new={}\n",
        old_comp.notes.len(),
        new_comp.notes.len()
    );

    // ─── 2. Output Format Matrix ────────────────────────────────────────────
    println!("━━━ 2. Output Format Matrix ━━━");
    for (fmt, name) in [
        (OutputFormat::Mono16, "Mono16"),
        (OutputFormat::MonoF32, "MonoF32"),
        (OutputFormat::StereoF32, "StereoF32"),
    ] {
        let cfg = MuseConfig {
            duration_secs: 2.0,
            max_notes: 8,
            output_format: fmt,
            ..Default::default()
        };
        let comp = compose(&cfg, &MusicalState::default(), 42);
        let ok = match (&comp.audio, fmt) {
            (AudioData::I16(_), OutputFormat::Mono16) => true,
            (AudioData::F32(_), OutputFormat::MonoF32) => true,
            (AudioData::StereoF32(_), OutputFormat::StereoF32) => true,
            _ => false,
        };
        check(&mut pass, &mut fail, &format!("{name} correct"), ok);
        println!(
            "  {name:12} — {} samples, RMS={:.4}",
            comp.audio.len(),
            audio_rms(&comp.audio)
        );
    }

    // Neural mode
    let neural_cfg = MuseConfig {
        duration_secs: 2.0,
        max_notes: 8,
        melody_mode: MelodyMode::Neural,
        ..Default::default()
    };
    let neural = compose(&neural_cfg, &MusicalState::default(), 42);
    check(
        &mut pass,
        &mut fail,
        "Neural mode produces audio",
        !neural.audio.is_empty(),
    );
    println!(
        "  Neural — {} notes, {} samples\n",
        neural.notes.len(),
        neural.audio.len()
    );

    // ─── 3. Consciousness-Coupled Variation ─────────────────────────────────
    println!("━━━ 3. Consciousness-Coupled Variation ━━━");
    let calm = MusicalState {
        consciousness_level: 0.9,
        arousal: 0.1,
        serotonin: 0.8,
        ..Default::default()
    };
    let intense = MusicalState {
        consciousness_level: 0.2,
        arousal: 0.9,
        noradrenaline: 0.8,
        dopamine: 0.9,
        ..Default::default()
    };
    let cfg = MuseConfig {
        duration_secs: 3.0,
        max_notes: 12,
        ..Default::default()
    };
    let calm_comp = compose(&cfg, &calm, 42);
    let intense_comp = compose(&cfg, &intense, 42);
    let calm_rms = audio_rms(&calm_comp.audio);
    let intense_rms = audio_rms(&intense_comp.audio);
    println!(
        "  Calm:    {} notes, RMS={:.4}",
        calm_comp.notes.len(),
        calm_rms
    );
    println!(
        "  Intense: {} notes, RMS={:.4}",
        intense_comp.notes.len(),
        intense_rms
    );
    check(
        &mut pass,
        &mut fail,
        "Different states → different note counts",
        calm_comp.notes.len() != intense_comp.notes.len(),
    );
    println!();

    // ─── 4. Streaming Pipeline ──────────────────────────────────────────────
    println!("━━━ 4. Streaming Pipeline ━━━");
    {
        use symthaea_muse::streaming::StreamingSynth;
        let mut synth = StreamingSynth::new(
            MuseConfig {
                duration_secs: 4.0,
                max_notes: 16,
                ..Default::default()
            },
            44100,
        );
        synth.update_state(&MusicalState {
            consciousness_level: 0.8,
            ..Default::default()
        });
        let mut total_rms = 0.0f32;
        let mut chunks = 0;
        for _ in 0..100 {
            let chunk = synth.render_chunk();
            let rms: f32 = (chunk.iter().map(|s| s[0] * s[0] + s[1] * s[1]).sum::<f32>()
                / chunk.len() as f32)
                .sqrt();
            total_rms += rms;
            chunks += 1;
            for s in &chunk {
                assert!(s[0].is_finite() && s[1].is_finite());
            }
        }
        let avg_rms = total_rms / chunks as f32;
        println!("  100 chunks rendered, avg RMS={:.4}", avg_rms);
        println!("  Total samples: {}", synth.total_samples_rendered());
        let features = synth.feedback_features();
        println!(
            "  Feedback: centroid={:.3}, flux={:.3}, entropy={:.3}",
            features.spectral_centroid, features.spectral_flux, features.rhythm_entropy
        );
        check(&mut pass, &mut fail, "Streaming renders finite audio", true);
        check(
            &mut pass,
            &mut fail,
            "Feedback loop extracts features",
            features.spectral_centroid > 0.0 || features.rms_energy > 0.0,
        );
    }
    println!();

    // ─── 5. Module Smoke Tests ──────────────────────────────────────────────
    println!("━━━ 5. Module Smoke Tests ━━━");
    {
        // Wavetable
        use symthaea_muse::wavetable::{WavetableBank, WavetableOscillator};
        let bank = WavetableBank::default_bank();
        let mut osc = WavetableOscillator::new();
        osc.set_consciousness_morph(0.7);
        let s = osc.tick(&bank, 440.0, 44100.0);
        check(
            &mut pass,
            &mut fail,
            "Wavetable produces signal",
            s.abs() > 0.0 || osc.phase > 0.0,
        );

        // Sidechain
        use symthaea_muse::sidechain::DuckingMatrix;
        let mut matrix = DuckingMatrix::default_matrix(44100);
        let roles = [
            symthaea_muse::voice::VoiceRole::Lead,
            symthaea_muse::voice::VoiceRole::Bass,
        ];
        let mut bufs = vec![vec![0.5f32; 100], vec![0.3f32; 100]];
        matrix.apply(&mut bufs, &roles, 100);
        check(
            &mut pass,
            &mut fail,
            "Sidechain ducks bass",
            bufs[1].iter().sum::<f32>() / 100.0 < 0.3,
        );

        // Consciousness reverb
        use symthaea_muse::consciousness_reverb::ConsciousnessReverb;
        let mut reverb = ConsciousnessReverb::new(44100);
        reverb.update_state(&MusicalState::default());
        let (l, r) = reverb.process_stereo(0.5, 0.5);
        check(
            &mut pass,
            &mut fail,
            "Consciousness reverb produces output",
            l.abs() > 0.0 || r.abs() > 0.0,
        );

        // Binaural
        use symthaea_muse::binaural::BinauralConsciousnessRenderer;
        let mut bin = BinauralConsciousnessRenderer::new(2, 44100);
        bin.update_state(&MusicalState {
            consciousness_level: 0.8,
            ..Default::default()
        });
        let out = bin.render(&[vec![0.5; 256], vec![0.3; 256]]);
        check(
            &mut pass,
            &mut fail,
            "Binaural renders stereo",
            out.iter().any(|s| s[0].abs() > 0.01),
        );

        // Substrate timbre
        use symthaea_muse::substrate_timbre::{SubstrateTimbreModifier, SubstrateTimbreType};
        let quantum = SubstrateTimbreModifier::for_substrate(SubstrateTimbreType::Quantum);
        let (l, r) = quantum.stereo_decorrelation(42);
        check(
            &mut pass,
            &mut fail,
            "Quantum stereo decorrelates",
            (l - r).abs() > 0.001,
        );

        // Phi optimizer
        use symthaea_muse::phi_optimizer::{PhiOptimizer, PhiTarget};
        let mut opt = PhiOptimizer::new(PhiTarget::Maximize);
        opt.update_phi(0.2);
        let mut s = MusicalState::default();
        opt.modulate_state(&mut s);
        check(
            &mut pass,
            &mut fail,
            "Phi optimizer modulates state",
            s.prediction_error > 0.1,
        );

        // Audio feedback
        use symthaea_muse::audio_feedback::AudioFeedbackEncoder;
        let mut fb = AudioFeedbackEncoder::new();
        let chunk: Vec<[f32; 2]> = (0..1024)
            .map(|i| {
                let s = (i as f32 * 0.1).sin() * 0.5;
                [s, s]
            })
            .collect();
        let features = fb.extract(&chunk, 44100);
        check(
            &mut pass,
            &mut fail,
            "Audio feedback extracts features",
            features.rms_energy > 0.0,
        );

        // Timbre space
        use symthaea_core::genesis::GenesisSeed;
        use symthaea_muse::timbre_space::TimbreManifold;
        let genesis = GenesisSeed::from_phrase("test");
        let mut manifold = TimbreManifold::default_manifold(&genesis);
        let partials = manifold.lookup(0.5, 0.5);
        check(
            &mut pass,
            &mut fail,
            "Timbre space returns partials",
            !partials.is_empty(),
        );

        // Collaborative
        use symthaea_muse::collaborative::{
            BlendStrategy, CollaborativeComposer, HarmonyContribution,
        };
        let mut composer = CollaborativeComposer::new(BlendStrategy::PhiWeighted);
        composer.contribute(HarmonyContribution {
            participant_id: 1,
            harmonies: [0.8; 8],
            phi: 0.9,
            cycle: 0,
        });
        composer.contribute(HarmonyContribution {
            participant_id: 2,
            harmonies: [0.3; 8],
            phi: 0.3,
            cycle: 0,
        });
        let collective = composer.blend();
        check(
            &mut pass,
            &mut fail,
            "Collaborative blends participants",
            collective.consciousness_level != 0.5,
        );

        // Spectral vocoder
        use symthaea_muse::spectral_vocoder::SpectralVocoder;
        let genesis = GenesisSeed::from_phrase("test-vocoder");
        let mut vocoder = SpectralVocoder::new(&genesis, 44100);
        let chunk = vocoder.render_chunk(&MusicalState::default(), 512);
        check(
            &mut pass,
            &mut fail,
            "Spectral vocoder renders audio",
            chunk.iter().any(|s| s[0].abs() > 0.0001),
        );
    }
    println!();

    // ─── Summary ────────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════╗");
    if fail == 0 {
        println!(
            "║  ALL {} CHECKS PASSED                                      ║",
            pass
        );
    } else {
        println!(
            "║  {} passed, {} failed                                       ║",
            pass, fail
        );
    }
    println!("╚══════════════════════════════════════════════════════════════╝");

    if fail > 0 {
        std::process::exit(1);
    }
}

fn check(pass: &mut u32, fail: &mut u32, name: &str, ok: bool) {
    if ok {
        println!("  [PASS] {name}");
        *pass += 1;
    } else {
        println!("  [FAIL] {name}");
        *fail += 1;
    }
}

fn audio_rms(audio: &AudioData) -> f32 {
    let samples: Vec<f32> = match audio {
        AudioData::I16(s) => s.iter().map(|&x| x as f32 / 32768.0).collect(),
        AudioData::F32(s) => s.clone(),
        AudioData::StereoF32(s) => s.iter().map(|p| (p[0] + p[1]) * 0.5).collect(),
    };
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}