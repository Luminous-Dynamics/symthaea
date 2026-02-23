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
    eprintln!("Run with: cargo run --example vocal_tract_benchmark --features vocal-tract --release");
}

#[cfg(feature = "vocal-tract")]
fn main() {
    use std::time::Instant;
    use symthaea::voice::formant_targets::FormantDatabase;
    use symthaea::voice::vocal_tract_controller::{VocalTractConfig, VocalTractController};
    use symthaea::voice::{
        ArticulatoryConfig, ArticulatorySynthesizer, LTCPacing, TimedPhoneme,
    };
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::HDC_DIMENSION;

    println!("=== Vocal Tract Benchmark: LTC vs Rule-Based ===\n");

    let genesis = GenesisSeed::from_phrase("vocal-tract-benchmark");
    let db = FormantDatabase::new();

    // ── Setup LTC controller ─────────────────────────────────────────────
    let config = VocalTractConfig::default();
    let mut ltc = VocalTractController::new(&genesis, &config);

    println!("Training LTC controller on {} phonemes (10 epochs)...", db.all_phonemes().len());
    let final_loss = ltc.train_on_phoneme_targets(&genesis, &db, 10);
    println!("  Final training loss: {:.2}\n", final_loss);

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
            "AVG", "", "", "", "", "", "",
            ltc_total_err / vowel_count as f32,
            rule_total_err / vowel_count as f32
        );
    }
    println!();

    // ── Benchmark 2: Transition smoothness ───────────────────────────────
    println!("--- Benchmark 2: Transition Smoothness (AH → IY) ---");

    // LTC transition
    ltc.reset();
    let ah_hv = genesis.hv("phoneme::AH", HDC_DIMENSION);
    let iy_hv = genesis.hv("phoneme::IY", HDC_DIMENSION);

    let mut ltc_f1_values = Vec::new();
    for _ in 0..40 {
        let frame = ltc.forward(&ah_hv, 0.005);
        ltc_f1_values.push(frame.f1);
    }
    for _ in 0..40 {
        let frame = ltc.forward(&iy_hv, 0.005);
        ltc_f1_values.push(frame.f1);
    }

    let ltc_max_delta: f32 = ltc_f1_values
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);

    // Rule-based transition
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

    println!("  LTC  max F1 delta: {:.2} Hz/frame", ltc_max_delta);
    println!("  Rule max F1 delta: {:.2} Hz/frame", rule_max_delta);
    println!(
        "  Winner: {}",
        if ltc_max_delta < rule_max_delta { "LTC (smoother)" } else { "Rule-based (smoother)" }
    );
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

    println!("  LTC:  {:.1} us/frame ({} frames)", ltc_us_per_frame, n_frames);
    println!("  Rule: {:.1} us/frame (~{} frames)", rule_us_per_frame, rule_frames_total);
    let ltc_hz = 1_000_000.0 / ltc_us_per_frame;
    println!("  LTC throughput: {:.0} Hz (target: 200Hz)", ltc_hz);
    println!();

    // ── Summary ──────────────────────────────────────────────────────────
    println!("=== Benchmark Complete ===");
}
