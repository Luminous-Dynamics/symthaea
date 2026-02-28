//! Parameter sweep for vocal tract LTC controller.
//!
//! Tests one parameter at a time while holding others at Phase 23 baseline,
//! then tests the combined best configuration.
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

    /// Train + evaluate a single configuration. Returns (avg_vowel_err, per_vowel_errs, loss).
    fn evaluate(
        genesis: &GenesisSeed,
        db: &FormantDatabase,
        config: &VocalTractConfig,
        params: &TrainingHyperparams,
        epochs: usize,
        test_vowels: &[&str],
    ) -> (f32, Vec<(String, f32)>, f32) {
        let mut ltc = VocalTractController::new_with_weight_init(
            genesis,
            config,
            params.weight_init_scale,
        );

        let phonemes = db.all_phonemes();
        let targets: Vec<(&str, &symthaea::voice::formant_targets::FormantTarget)> = phonemes
            .iter()
            .filter_map(|name| db.lookup(name).map(|t| (name.as_str(), t)))
            .collect();
        let loss = ltc.train_on_phoneme_targets_configured(genesis, &targets, epochs, params);

        // Transition training (always 10 epochs with default transition LR)
        symthaea::voice::train_controller_transitions(&mut ltc, genesis, db, 10);

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

    println!("=== Vocal Tract Parameter Sweep ===\n");
    println!("Epochs: {}, Phonemes: {}\n", epochs, db.all_phonemes().len());

    // ── Baseline ──────────────────────────────────────────────────
    let baseline_params = TrainingHyperparams::default();
    let baseline_config = VocalTractConfig::default();

    print!("Baseline (Phase 23 defaults)... ");
    let start = Instant::now();
    let (baseline_avg, baseline_per, baseline_loss) =
        evaluate(&genesis, &db, &baseline_config, &baseline_params, epochs, &test_vowels);
    println!("done in {:.0}s", start.elapsed().as_secs_f32());
    print_result("BASELINE", baseline_avg, baseline_loss, &baseline_per);
    println!();

    // ── Sweep definitions ─────────────────────────────────────────
    // Each sweep: (label, parameter name, values to test)
    // We test one param at a time, keeping all others at default.

    let mut best_overall_avg = baseline_avg;
    let mut best_overall_label = "BASELINE".to_string();
    let mut best_params = baseline_params.clone();
    let mut best_config = baseline_config.clone();

    // 1. LR annealing floor
    println!("── Sweep: lr_min_mult ──");
    for &val in &[3.0f32, 5.0, 7.0, 15.0] {
        let mut p = baseline_params.clone();
        p.lr_min_mult = val;
        let label = format!("lr_min={}", val);
        print!("  {}... ", label);
        let start = Instant::now();
        let (avg, per, loss) = evaluate(&genesis, &db, &baseline_config, &p, epochs, &test_vowels);
        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(&label, avg, loss, &per);
        if avg < best_overall_avg {
            best_overall_avg = avg;
            best_overall_label = label;
            best_params = p;
        }
    }
    println!();

    // 2. LR peak
    println!("── Sweep: lr_peak_mult ──");
    for &val in &[20.0f32, 25.0, 40.0] {
        let mut p = baseline_params.clone();
        p.lr_peak_mult = val;
        let label = format!("lr_peak={}", val);
        print!("  {}... ", label);
        let start = Instant::now();
        let (avg, per, loss) = evaluate(&genesis, &db, &baseline_config, &p, epochs, &test_vowels);
        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(&label, avg, loss, &per);
        if avg < best_overall_avg {
            best_overall_avg = avg;
            best_overall_label = label;
            best_params = p;
        }
    }
    println!();

    // 3. Weight init scale
    println!("── Sweep: weight_init_scale ──");
    for &val in &[0.10f32, 0.20, 0.25] {
        let mut p = baseline_params.clone();
        p.weight_init_scale = val;
        let label = format!("w_init={}", val);
        print!("  {}... ", label);
        let start = Instant::now();
        let (avg, per, loss) = evaluate(&genesis, &db, &baseline_config, &p, epochs, &test_vowels);
        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(&label, avg, loss, &per);
        if avg < best_overall_avg {
            best_overall_avg = avg;
            best_overall_label = label;
            best_params = p;
        }
    }
    println!();

    // 4. Distance LR cap
    println!("── Sweep: distance_lr_cap ──");
    for &val in &[2.0f32, 2.5, 4.0] {
        let mut p = baseline_params.clone();
        p.distance_lr_cap = val;
        let label = format!("dist_cap={}", val);
        print!("  {}... ", label);
        let start = Instant::now();
        let (avg, per, loss) = evaluate(&genesis, &db, &baseline_config, &p, epochs, &test_vowels);
        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(&label, avg, loss, &per);
        if avg < best_overall_avg {
            best_overall_avg = avg;
            best_overall_label = label;
            best_params = p;
        }
    }
    println!();

    // 5. Outlier steps
    println!("── Sweep: outlier_steps ──");
    for &val in &[15usize, 25, 30] {
        let mut p = baseline_params.clone();
        p.outlier_steps = val;
        let label = format!("out_steps={}", val);
        print!("  {}... ", label);
        let start = Instant::now();
        let (avg, per, loss) = evaluate(&genesis, &db, &baseline_config, &p, epochs, &test_vowels);
        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(&label, avg, loss, &per);
        if avg < best_overall_avg {
            best_overall_avg = avg;
            best_overall_label = label;
            best_params = p;
        }
    }
    println!();

    // 6. Warmup steps
    println!("── Sweep: warmup_steps ──");
    for &val in &[10usize, 15, 30] {
        let mut p = baseline_params.clone();
        p.warmup_steps = val;
        let label = format!("warmup={}", val);
        print!("  {}... ", label);
        let start = Instant::now();
        let (avg, per, loss) = evaluate(&genesis, &db, &baseline_config, &p, epochs, &test_vowels);
        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(&label, avg, loss, &per);
        if avg < best_overall_avg {
            best_overall_avg = avg;
            best_overall_label = label;
            best_params = p;
        }
    }
    println!();

    // 7. Neurons per layer (requires new config)
    println!("── Sweep: neurons_per_layer ──");
    for &val in &[3usize, 5] {
        let mut c = baseline_config.clone();
        c.neurons_per_layer = val;
        let label = format!("neurons={}", val);
        print!("  {}... ", label);
        let start = Instant::now();
        let (avg, per, loss) = evaluate(&genesis, &db, &c, &baseline_params, epochs, &test_vowels);
        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(&label, avg, loss, &per);
        if avg < best_overall_avg {
            best_overall_avg = avg;
            best_overall_label = label;
            best_params = baseline_params.clone();
            best_config = c;
        }
    }
    println!();

    // 8. F2 distance weight
    println!("── Sweep: f2_distance_weight ──");
    for &val in &[1.0f32, 2.0, 6.0] {
        let mut p = baseline_params.clone();
        p.f2_distance_weight = val;
        let label = format!("f2_wt={}", val);
        print!("  {}... ", label);
        let start = Instant::now();
        let (avg, per, loss) = evaluate(&genesis, &db, &baseline_config, &p, epochs, &test_vowels);
        println!("done in {:.0}s", start.elapsed().as_secs_f32());
        print_result(&label, avg, loss, &per);
        if avg < best_overall_avg {
            best_overall_avg = avg;
            best_overall_label = label;
            best_params = p;
        }
    }
    println!();

    // ── Summary ───────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════");
    println!("  BEST SINGLE-PARAM: {} → avg {:.1} Hz (baseline {:.1} Hz)",
        best_overall_label, best_overall_avg, baseline_avg);
    println!("  Best params: {:?}", best_params);
    println!("  Best config neurons: {}", best_config.neurons_per_layer);
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
