//! Benchmark: LTC-Driven vs Rule-Based Articulatory Synthesis
//!
//! Compares the HdcLtcUnifiedNetwork vocal tract controller against
//! the existing rule-based ArticulatorySynthesizer on:
//! 1. Formant accuracy (F1/F2/F3 vs ground truth)
//! 2. Transition smoothness (max frame-to-frame delta)
//! 3. Timing (ms per frame)
//!
//! Run with: cargo run --example vocal_tract_benchmark --features vocal-tract --release

#[cfg(not(feature = "vocal-tract"))]
fn main() {
    eprintln!("This example requires the `vocal-tract` feature.");
    eprintln!(
        "Run with: cargo run --example vocal_tract_benchmark --features vocal-tract --release"
    );
}

#[cfg(feature = "vocal-tract")]
fn main() {
    use std::time::Instant;
    use symthaea::voice::formant_targets::FormantDatabase;
    use symthaea::voice::vocal_tract_controller::{VocalTractConfig, VocalTractController};
    use symthaea::voice::vocal_tract_encoder::VoiceCognitiveState;
    use symthaea::voice::vocal_tract_fep::{populate_manner_map, VocalTractPipeline};
    use symthaea::voice::{ArticulatoryConfig, ArticulatorySynthesizer, LTCPacing, TimedPhoneme};
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::HDC_DIMENSION;

    println!("=== Vocal Tract Benchmark: LTC vs Rule-Based ===\n");

    let genesis = GenesisSeed::from_phrase("vocal-tract-benchmark");
    let db = FormantDatabase::new();

    // ── Setup LTC controller ─────────────────────────────────────────────
    let config = VocalTractConfig::default();
    let mut ltc = VocalTractController::new(&genesis, &config);

    println!(
        "Training LTC controller on {} phonemes (100 epochs)...",
        db.all_phonemes().len()
    );
    for epoch in 0..100 {
        let loss = symthaea::voice::train_controller_on_phoneme_db(&mut ltc, &genesis, &db, 1);
        println!("  Epoch {}: avg loss = {:.2}", epoch + 1, loss);
    }
    println!();

    // ── Setup rule-based synthesizer ─────────────────────────────────────
    let mut articulatory = ArticulatorySynthesizer::with_config(ArticulatoryConfig {
        base_f0: 120.0,
        base_tau: 0.05,
        frame_rate: 200.0,
        coarticulation: true,
        ..Default::default()
    });

    // ── Benchmark 1: Vowel formant accuracy ──────────────────────────────
    println!("--- Benchmark 1: Vowel Formant Accuracy ---");
    println!(
        "  {:>6} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} | {:>8} {:>8}",
        "Phon", "LTC-F1", "LTC-F2", "LTC-F3", "Tgt-F1", "Tgt-F2", "Tgt-F3", "LTC-err", "Rule-err"
    );
    println!("  {}", "-".repeat(100));

    let test_vowels = ["AH", "IY", "UW", "AE", "EH", "IH", "AA", "OW", "AO", "UH"];
    let mut ltc_total_err = 0.0f32;
    let mut rule_total_err = 0.0f32;
    let mut vowel_count = 0;

    for vowel in &test_vowels {
        if let Some(target) = db.lookup(vowel) {
            // LTC prediction
            ltc.reset();
            let phoneme_hv = genesis.hv(&format!("phoneme::{}", vowel), HDC_DIMENSION);
            // Warm up network state
            for _ in 0..20 {
                ltc.forward(&phoneme_hv, 0.005);
            }
            // Average over 10 steady-state frames
            let mut f1_sum = 0.0f32;
            let mut f2_sum = 0.0f32;
            let mut f3_sum = 0.0f32;
            for _ in 0..10 {
                let frame = ltc.forward(&phoneme_hv, 0.005);
                f1_sum += frame.f1;
                f2_sum += frame.f2;
                f3_sum += frame.f3;
            }
            let ltc_f1 = f1_sum / 10.0;
            let ltc_f2 = f2_sum / 10.0;
            let ltc_f3 = f3_sum / 10.0;

            let ltc_err = ((ltc_f1 - target.f1).powi(2)
                + (ltc_f2 - target.f2).powi(2)
                + (ltc_f3 - target.f3).powi(2))
            .sqrt();

            // Rule-based prediction: synthesize the vowel phoneme
            let pacing = LTCPacing::default();
            let timed = vec![TimedPhoneme {
                phoneme: vowel.to_string(),
                start_time: 0.0,
                duration: 0.1,
                stress: 1,
            }];
            let rule_frames = articulatory.synthesize(&timed, &pacing);

            // Average formants from rule-based frames (skip first few for onset)
            let (_rule_f1, _rule_f2, _rule_f3, rule_err) = if rule_frames.len() > 5 {
                let stable = &rule_frames[5..];
                let n = stable.len() as f32;
                let rf1: f32 = stable.iter().map(|f| f.f1).sum::<f32>() / n;
                let rf2: f32 = stable.iter().map(|f| f.f2).sum::<f32>() / n;
                let rf3: f32 = stable.iter().map(|f| f.f3).sum::<f32>() / n;
                let err = ((rf1 - target.f1).powi(2)
                    + (rf2 - target.f2).powi(2)
                    + (rf3 - target.f3).powi(2))
                .sqrt();
                (rf1, rf2, rf3, err)
            } else {
                (0.0, 0.0, 0.0, f32::MAX)
            };

            println!(
                "  {:>6} | {:>10.1} {:>10.1} {:>10.1} | {:>10.1} {:>10.1} {:>10.1} | {:>8.1} {:>8.1}",
                vowel, ltc_f1, ltc_f2, ltc_f3, target.f1, target.f2, target.f3, ltc_err, rule_err
            );

            ltc_total_err += ltc_err;
            rule_total_err += rule_err;
            vowel_count += 1;
        }
    }

    if vowel_count > 0 {
        println!("  {}", "-".repeat(100));
        println!(
            "  {:>6} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} | {:>8.1} {:>8.1}",
            "AVG",
            "",
            "",
            "",
            "",
            "",
            "",
            ltc_total_err / vowel_count as f32,
            rule_total_err / vowel_count as f32
        );
    }
    println!();

    // ── Benchmark 2: Transition smoothness (pipeline-based) ─────────────
    println!("--- Benchmark 2: Transition Smoothness (AH → IY, pipeline) ---");

    // Pipeline-based transition (includes coarticulation + adaptive rate limiting)
    let mut pipeline = VocalTractPipeline::new(&genesis);
    populate_manner_map(&mut pipeline);
    // Train pipeline's controller to match the standalone one
    symthaea::voice::train_controller_on_phoneme_db(
        &mut pipeline.controller,
        &genesis,
        &db,
        100,
    );

    let state = VoiceCognitiveState::default();
    let dt = 0.005;

    // Run 40 frames of /AH/ then 40 frames of /IY/ through pipeline
    let mut pipeline_f1_values = Vec::new();
    for _ in 0..40 {
        let frame = pipeline.tick_phoneme(&state, None, dt, Some("AH"));
        pipeline_f1_values.push(frame.f1);
    }
    for _ in 0..40 {
        let frame = pipeline.tick_phoneme(&state, None, dt, Some("IY"));
        pipeline_f1_values.push(frame.f1);
    }

    // Separate transition frames (around frame 40) from steady-state
    let transition_start = 35; // 5 frames before switch
    let transition_end = 56; // 16 frames after switch (coarticulation window)
    let pipeline_transition_max: f32 = pipeline_f1_values[transition_start..transition_end]
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);
    let pipeline_steady_max: f32 = pipeline_f1_values[60..]
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);

    // Rule-based transition for comparison
    let pacing = LTCPacing::default();
    let timed = vec![
        TimedPhoneme {
            phoneme: "AH".to_string(),
            start_time: 0.0,
            duration: 0.2,
            stress: 1,
        },
        TimedPhoneme {
            phoneme: "IY".to_string(),
            start_time: 0.2,
            duration: 0.2,
            stress: 1,
        },
    ];
    let rule_frames = articulatory.synthesize(&timed, &pacing);
    let rule_max_delta: f32 = rule_frames
        .windows(2)
        .map(|w| (w[1].f1 - w[0].f1).abs())
        .fold(0.0f32, f32::max);

    println!(
        "  Pipeline max F1 delta (transition): {:.2} Hz/frame",
        pipeline_transition_max
    );
    println!(
        "  Pipeline max F1 delta (steady):     {:.2} Hz/frame",
        pipeline_steady_max
    );
    println!("  Rule     max F1 delta:              {:.2} Hz/frame", rule_max_delta);
    println!();

    // ── Benchmark 3: Timing ──────────────────────────────────────────────
    println!("--- Benchmark 3: Timing ---");
    let n_frames = 1000;

    // LTC timing
    ltc.reset();
    let hv = genesis.hv("phoneme::AH", HDC_DIMENSION);
    let start = Instant::now();
    for _ in 0..n_frames {
        ltc.forward(&hv, 0.005);
    }
    let ltc_elapsed = start.elapsed();
    let ltc_us_per_frame = ltc_elapsed.as_micros() as f64 / n_frames as f64;

    // Rule-based timing
    let timed_long: Vec<TimedPhoneme> = (0..10)
        .map(|i| TimedPhoneme {
            phoneme: "AH".to_string(),
            start_time: i as f32 * 0.1,
            duration: 0.1,
            stress: 1,
        })
        .collect();
    let start = Instant::now();
    for _ in 0..10 {
        let _ = articulatory.synthesize(&timed_long, &pacing);
    }
    let rule_elapsed = start.elapsed();
    let rule_frames_total = 10 * timed_long.len() * 20; // ~20 frames per 0.1s phoneme at 200Hz
    let rule_us_per_frame = if rule_frames_total > 0 {
        rule_elapsed.as_micros() as f64 / rule_frames_total as f64
    } else {
        0.0
    };

    println!(
        "  LTC:  {:.1} us/frame ({} frames)",
        ltc_us_per_frame, n_frames
    );
    println!(
        "  Rule: {:.1} us/frame (~{} frames)",
        rule_us_per_frame, rule_frames_total
    );
    let ltc_hz = 1_000_000.0 / ltc_us_per_frame;
    println!("  LTC throughput: {:.0} Hz (target: 200Hz)", ltc_hz);
    println!();

    // ── Benchmark 4: Consonant formant accuracy ─────────────────────────
    println!("--- Benchmark 4: Consonant Formant Accuracy ---");
    println!(
        "  {:>6} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} | {:>8} {:>8} | {:>10}",
        "Phon", "LTC-F1", "LTC-F2", "LTC-F3", "Tgt-F1", "Tgt-F2", "Tgt-F3", "LTC-err",
        "Rule-err", "Manner"
    );
    println!("  {}", "-".repeat(115));

    let test_consonants = ["P", "T", "K", "B", "D", "G", "S", "F", "M", "N", "L", "R"];
    let mut cons_ltc_total_err = 0.0f32;
    let mut cons_rule_total_err = 0.0f32;
    let mut cons_count = 0;

    for consonant in &test_consonants {
        if let Some(target) = db.lookup(consonant) {
            // LTC prediction
            ltc.reset();
            let phoneme_hv = genesis.hv(&format!("phoneme::{}", consonant), HDC_DIMENSION);
            for _ in 0..20 {
                ltc.forward(&phoneme_hv, 0.005);
            }
            let mut f1_sum = 0.0f32;
            let mut f2_sum = 0.0f32;
            let mut f3_sum = 0.0f32;
            for _ in 0..10 {
                let frame = ltc.forward(&phoneme_hv, 0.005);
                f1_sum += frame.f1;
                f2_sum += frame.f2;
                f3_sum += frame.f3;
            }
            let ltc_f1 = f1_sum / 10.0;
            let ltc_f2 = f2_sum / 10.0;
            let ltc_f3 = f3_sum / 10.0;

            let ltc_err = ((ltc_f1 - target.f1).powi(2)
                + (ltc_f2 - target.f2).powi(2)
                + (ltc_f3 - target.f3).powi(2))
            .sqrt();

            // Rule-based prediction
            let timed = vec![TimedPhoneme {
                phoneme: consonant.to_string(),
                start_time: 0.0,
                duration: 0.1,
                stress: 0,
            }];
            let rule_frames = articulatory.synthesize(&timed, &pacing);
            let rule_err = if rule_frames.len() > 5 {
                let stable = &rule_frames[5..];
                let n = stable.len() as f32;
                let rf1: f32 = stable.iter().map(|f| f.f1).sum::<f32>() / n;
                let rf2: f32 = stable.iter().map(|f| f.f2).sum::<f32>() / n;
                let rf3: f32 = stable.iter().map(|f| f.f3).sum::<f32>() / n;
                ((rf1 - target.f1).powi(2)
                    + (rf2 - target.f2).powi(2)
                    + (rf3 - target.f3).powi(2))
                .sqrt()
            } else {
                f32::MAX
            };

            let manner_str = format!("{:?}", target.manner);
            println!(
                "  {:>6} | {:>10.1} {:>10.1} {:>10.1} | {:>10.1} {:>10.1} {:>10.1} | {:>8.1} {:>8.1} | {:>10}",
                consonant, ltc_f1, ltc_f2, ltc_f3, target.f1, target.f2, target.f3,
                ltc_err, rule_err, manner_str
            );

            cons_ltc_total_err += ltc_err;
            cons_rule_total_err += rule_err;
            cons_count += 1;
        }
    }

    if cons_count > 0 {
        println!("  {}", "-".repeat(115));
        println!(
            "  {:>6} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} | {:>8.1} {:>8.1}",
            "AVG", "", "", "", "", "", "",
            cons_ltc_total_err / cons_count as f32,
            cons_rule_total_err / cons_count as f32
        );
    }
    println!();

    // ── Benchmark 5: Source type verification ─────────────────────────────
    println!("--- Benchmark 5: Source Type Verification (pipeline) ---");

    // Use pipeline since source_type is set in tick_phoneme() from manner map
    pipeline.reset();

    let source_type_tests: Vec<(&str, &str)> = vec![
        ("AH", "Vowel"),
        ("IY", "Vowel"),
        ("P", "Stop"),
        ("T", "Stop"),
        ("K", "Stop"),
        ("B", "Stop"),
        ("S", "Fricative"),
        ("F", "Fricative"),
        ("M", "Nasal"),
        ("N", "Nasal"),
        ("CH", "Affricate"),
        ("L", "Liquid"),
    ];

    let mut pass_count = 0;
    let total = source_type_tests.len();

    for (phoneme, expected_str) in &source_type_tests {
        let frame = pipeline.tick_phoneme(&state, None, dt, Some(phoneme));
        let actual_str = format!("{:?}", frame.source_type);
        let ok = actual_str == *expected_str;
        if ok {
            pass_count += 1;
        }
        println!(
            "  {:>4} → expected {:>10}, got {:>10} [{}]",
            phoneme,
            expected_str,
            actual_str,
            if ok { "PASS" } else { "FAIL" }
        );
    }

    println!(
        "\n  Source type verification: {}/{} passed",
        pass_count, total
    );
    println!();

    // ── Summary ──────────────────────────────────────────────────────────
    println!("=== Benchmark Complete ===");
}
