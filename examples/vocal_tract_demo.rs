//! LTC-Driven Vocal Tract Pipeline Demo
//!
//! Demonstrates the full HDC → LTC → FEP active inference pipeline
//! for articulatory synthesis.
//!
//! Run with: cargo run --example vocal_tract_demo --features vocal-tract --release

#[cfg(not(feature = "vocal-tract"))]
fn main() {
    eprintln!("This example requires the `vocal-tract` feature.");
    eprintln!("Run with: cargo run --example vocal_tract_demo --features vocal-tract --release");
}

#[cfg(feature = "vocal-tract")]
fn main() {
    use symthaea::voice::formant_targets::FormantDatabase;
    use symthaea::voice::vocal_tract_controller::{VocalTractConfig, VocalTractController};
    use symthaea::voice::vocal_tract_encoder::{VoiceCognitiveState, VocalTractHdcEncoder};
    use symthaea::voice::vocal_tract_fep::{VocalTractFepAgent, VocalTractPipeline};
    use symthaea::voice::voice_feedback::VoiceOutputMetrics;
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::HDC_DIMENSION;

    println!("=== LTC-Driven Vocal Tract Pipeline Demo ===\n");

    let genesis = GenesisSeed::from_phrase("vocal-tract-demo");

    // ── 1. Pipeline creation ─────────────────────────────────────────────
    println!("--- 1. Pipeline Creation ---");
    let mut pipeline = VocalTractPipeline::new(&genesis);
    println!("  Pipeline created (16,384D HDC → LTC → 9D FormantFrame)");
    println!("  Dual-rate: 200Hz motor / 10Hz cognitive\n");

    // ── 2. Phoneme training ──────────────────────────────────────────────
    println!("--- 2. Phoneme Target Training ---");
    let db = FormantDatabase::new();
    let n_phonemes = db.all_phonemes().len();
    println!("  Training on {} ARPABET phonemes from FormantDatabase...", n_phonemes);

    let mut losses = Vec::new();
    for epoch in 0..5 {
        let loss = pipeline.controller.train_on_phoneme_targets(&genesis, &db, 1);
        losses.push(loss);
        println!("  Epoch {}: avg loss = {:.2}", epoch + 1, loss);
    }

    let improvement = if losses[0] > 0.0 {
        (1.0 - losses[4] / losses[0]) * 100.0
    } else {
        0.0
    };
    println!("  Loss reduction: {:.1}%\n", improvement);

    // ── 3. Vowel sequence synthesis ──────────────────────────────────────
    println!("--- 3. Vowel Sequence Synthesis ---");
    let vowels = ["AH", "IY", "UW", "AE"];
    let frames_per_vowel = 100; // 0.5s at 200Hz

    let state = VoiceCognitiveState::default();

    for vowel in &vowels {
        pipeline.controller.reset();

        // Get the deterministic HV for this phoneme
        let phoneme_hv = genesis.hv(&format!("phoneme::{}", vowel), HDC_DIMENSION);

        // Generate frames by feeding the phoneme HV through the pipeline
        let mut f1_sum = 0.0f32;
        let mut f2_sum = 0.0f32;
        for _ in 0..frames_per_vowel {
            let frame = pipeline.controller.forward(&phoneme_hv, 0.005);
            f1_sum += frame.f1;
            f2_sum += frame.f2;
        }

        let avg_f1 = f1_sum / frames_per_vowel as f32;
        let avg_f2 = f2_sum / frames_per_vowel as f32;

        let target = db.lookup(vowel).unwrap();
        println!(
            "  /{}/: F1={:.0}Hz (target {:.0}), F2={:.0}Hz (target {:.0})",
            vowel, avg_f1, target.f1, avg_f2, target.f2
        );
    }
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
    let good_metrics = VoiceOutputMetrics {
        articulation_score: 0.8,
        formant_accuracy: 0.7,
        pitch_stability: 0.9,
        coarticulation_smoothness: 0.8,
        duration_accuracy: 0.7,
        energy_consistency: 0.8,
        ..Default::default()
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
    println!("\n=== Demo Complete ===");
}
