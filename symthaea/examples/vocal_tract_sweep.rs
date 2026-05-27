// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Parameter sweep for vocal tract LTC controller.
//!
//! Tests one parameter at a time while holding others at Phase 24 baseline
//! (gradient training + LS refinement). Includes blend, lambda, and neurons sweeps.
//!
//! Run with: cargo run --example vocal_tract_sweep --features vocal-tract --release

#[cfg(not(feature = "vocal-tract"))]
fn main() {
    eprintln!("Requires `vocal-tract` feature.");
    eprintln!("Run: cargo run --example vocal_tract_sweep --features vocal-tract --release");
}

#[cfg(feature = "vocal-tract")]
fn main() {
    use std::time::Instant;
    use symthaea::voice::formant_targets::FormantDatabase;
    use symthaea::voice::vocal_tract_controller::{
        TrainingHyperparams, VocalTractConfig, VocalTractController,
    };
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::HDC_DIMENSION;

    let genesis = GenesisSeed::from_phrase("vocal-tract-benchmark");
    let db = FormantDatabase::new();

    let test_vowels = ["AH", "IY", "UW", "AE", "EH", "IH", "AA", "OW", "AO", "UH"];
    let epochs = 100;

    /// Train + LS refine + evaluate. Returns (avg_vowel_err, per_vowel_errs, loss).
    fn evaluate(
        genesis: &GenesisSeed,
        db: &FormantDatabase,
        config: &VocalTractConfig,
        params: &TrainingHyperparams,
        epochs: usize,
        test_vowels: &[&str],
        ls_blend: f32,
        ls_lambda: f32,
    ) -> (f32, Vec<(String, f32)>, f32) {
        let mut ltc =
            VocalTractController::new_with_weight_init(genesis, config, params.weight_init_scale);

        let phonemes = db.all_phonemes();
        let targets: Vec<(&str, &symthaea::voice::formant_targets::FormantTarget)> = phonemes
            .iter()
            .filter_map(|name| db.lookup(name).map(|t| (name.as_str(), t)))
            .collect();
        let loss = ltc.train_on_phoneme_targets_configured(genesis, &targets, epochs, params);

        // Transition training (always 10 epochs)
        symthaea::voice::train_controller_transitions(&mut ltc, genesis, db, 10);

        // LS refinement
        if ls_blend > 0.0 {
            ltc.refine_output_projection_ls_configured(genesis, &targets, ls_blend, ls_lambda);
        }

        // Evaluate vowels
        let mut total_err = 0.0f32;
        let mut per_vowel = Vec::new();
        for vowel in test_vowels {
            if let Some(target) = db.lookup(vowel) {
                ltc.reset();
                let phoneme_hv = genesis.hv(&format!("phoneme::{}", vowel), HDC_DIMENSION);
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
                let err = ((ltc_f1 - target.f1).powi(2)
                    + (ltc_f2 - target.f2).powi(2)
                    + (ltc_f3 - target.f3).powi(2))
                .sqrt();
                total_err += err;
                per_vowel.push((vowel.to_string(), err));
            }
        }
        let avg_err = total_err / per_vowel.len().max(1) as f32;
        (avg_err, per_vowel, loss)
    }

    println!("=== Vocal Tract Parameter Sweep (Phase 24 + LS) ===\n");
    println!(
        "Epochs: {}, Phonemes: {}\n",
        epochs,
        db.all_phonemes().len()
    );

    let baseline_params = TrainingHyperparams::default();
    let baseline_config = VocalTractConfig::default();
    let baseline_blend = 0.7f32;
    let baseline_lambda = 0.01f32;

    // ── Baseline (Phase 24: gradient + LS) ──────────────────────────
    print!("Baseline (Phase 24, blend=0.7, lambda=0.01)... ");
    let start = Instant::now();
    let (baseline_avg, baseline_per, baseline_loss) = evaluate(
        &genesis,
        &db,
        &baseline_config,
        &baseline_params,
        epochs,
        &test_vowels,
        baseline_blend,
        baseline_lambda,
    );
    println!("done in {:.0}s", start.elapsed().as_secs_f32());
    print_result("BASELINE+LS", baseline_avg, baseline_loss, &baseline_per);

    // Also show gradient-only baseline for reference
    print!("Gradient-only (no LS)... ");
    let start = Instant::now();
    let (grad_avg, grad_per, grad_loss) = evaluate(
        &genesis,
        &db,
        &baseline_config,
        &baseline_params,
        epochs,
        &test_vowels,
        0.0,
        baseline_lambda,
    );
    println!("done in {:.0}s", start.elapsed().as_secs_f32());
    print_result("GRADIENT-ONLY", grad_avg, grad_loss, &grad_per);
    println!();

    let mut best_avg = baseline_avg;
    let mut best_label = "BASELINE+LS".to_string();

    // ── 1. LS Blend sweep ───────────────────────────────────────────
    println!("── Sweep: LS blend ──");
    for &val in &[0.5f32, 0.6, 0.8, 0.9, 1.0] {
        let label = format!("blend={:.1}", val);
        print!("  {}... ", label);
        let start = Instant::now();
        let (avg, per, loss) = evaluate(
            &genesis,
            &db,
            &baseline_config,
            &baseline_params,
            epochs,
            &test_vowels,
            val,
            baseline_lambda,
        );
        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(&label, avg, loss, &per);
        if avg < best_avg {
            best_avg = avg;
            best_label = label;
        }
    }
    println!();

    // ── 2. LS Lambda sweep ──────────────────────────────────────────
    println!("── Sweep: LS lambda ──");
    for &val in &[0.001f32, 0.005, 0.05, 0.1] {
        let label = format!("lambda={}", val);
        print!("  {}... ", label);
        let start = Instant::now();
        let (avg, per, loss) = evaluate(
            &genesis,
            &db,
            &baseline_config,
            &baseline_params,
            epochs,
            &test_vowels,
            baseline_blend,
            val,
        );
        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(&label, avg, loss, &per);
        if avg < best_avg {
            best_avg = avg;
            best_label = label;
        }
    }
    println!();

    // ── 3. Neurons per layer + LS ───────────────────────────────────
    println!("── Sweep: neurons_per_layer (with LS) ──");
    for &val in &[3usize, 5, 6, 8] {
        let mut c = baseline_config.clone();
        c.neurons_per_layer = val;
        let label = format!("neurons={}", val);
        print!("  {}... ", label);
        let start = Instant::now();
        let (avg, per, loss) = evaluate(
            &genesis,
            &db,
            &c,
            &baseline_params,
            epochs,
            &test_vowels,
            baseline_blend,
            baseline_lambda,
        );
        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(&label, avg, loss, &per);
        if avg < best_avg {
            best_avg = avg;
            best_label = label;
        }
    }
    println!();

    // ── 4. Iterative LS (apply LS twice) ────────────────────────────
    println!("── Sweep: iterative LS ──");
    {
        let label = "LS×2 (iterative)";
        print!("  {}... ", label);
        let start = Instant::now();

        let mut ltc = VocalTractController::new_with_weight_init(
            &genesis,
            &baseline_config,
            baseline_params.weight_init_scale,
        );
        let phonemes = db.all_phonemes();
        let targets: Vec<(&str, &symthaea::voice::formant_targets::FormantTarget)> = phonemes
            .iter()
            .filter_map(|name| db.lookup(name).map(|t| (name.as_str(), t)))
            .collect();
        let loss =
            ltc.train_on_phoneme_targets_configured(&genesis, &targets, epochs, &baseline_params);
        symthaea::voice::train_controller_transitions(&mut ltc, &genesis, &db, 10);

        // First LS pass
        ltc.refine_output_projection_ls_configured(
            &genesis,
            &targets,
            baseline_blend,
            baseline_lambda,
        );
        // Second LS pass (re-collects HVs with updated weights)
        ltc.refine_output_projection_ls_configured(
            &genesis,
            &targets,
            baseline_blend,
            baseline_lambda,
        );

        // Evaluate
        let mut total_err = 0.0f32;
        let mut per_vowel = Vec::new();
        for vowel in &test_vowels {
            if let Some(target) = db.lookup(vowel) {
                ltc.reset();
                let phoneme_hv = genesis.hv(&format!("phoneme::{}", vowel), HDC_DIMENSION);
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
                let err = ((f1_sum / 10.0 - target.f1).powi(2)
                    + (f2_sum / 10.0 - target.f2).powi(2)
                    + (f3_sum / 10.0 - target.f3).powi(2))
                .sqrt();
                total_err += err;
                per_vowel.push((vowel.to_string(), err));
            }
        }
        let avg = total_err / per_vowel.len().max(1) as f32;

        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(label, avg, loss, &per_vowel);
        if avg < best_avg {
            best_avg = avg;
            best_label = label.to_string();
        }
    }
    println!();

    // ── 5. Gradient training epochs sweep (with LS) ─────────────────
    println!("── Sweep: gradient epochs (with LS) ──");
    for &val in &[50usize, 150, 200] {
        let label = format!("epochs={}", val);
        print!("  {}... ", label);
        let start = Instant::now();
        let (avg, per, loss) = evaluate(
            &genesis,
            &db,
            &baseline_config,
            &baseline_params,
            val,
            &test_vowels,
            baseline_blend,
            baseline_lambda,
        );
        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(&label, avg, loss, &per);
        if avg < best_avg {
            best_avg = avg;
            best_label = label;
        }
    }
    println!();

    // ── Summary ─────────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════");
    println!(
        "  BEST: {} → avg {:.1} Hz (baseline {:.1} Hz, grad-only {:.1} Hz)",
        best_label, best_avg, baseline_avg, grad_avg
    );
    println!("═══════════════════════════════════════════════════");
}

#[cfg(feature = "vocal-tract")]
fn print_result(label: &str, avg: f32, loss: f32, per_vowel: &[(String, f32)]) {
    let iy_err = per_vowel
        .iter()
        .find(|(n, _)| n == "IY")
        .map(|(_, e)| *e)
        .unwrap_or(0.0);
    let uw_err = per_vowel
        .iter()
        .find(|(n, _)| n == "UW")
        .map(|(_, e)| *e)
        .unwrap_or(0.0);
    let aa_err = per_vowel
        .iter()
        .find(|(n, _)| n == "AA")
        .map(|(_, e)| *e)
        .unwrap_or(0.0);
    println!(
        "    {:>20} | avg {:.1} Hz | IY {:.1} | UW {:.1} | AA {:.1} | loss {:.4}",
        label, avg, iy_err, uw_err, aa_err, loss
    );
}