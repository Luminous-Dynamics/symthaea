// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # SPARC Rotation-Curve Model Comparison
//!
//! Fits every SPARC galaxy's rotation curve with four gravity models
//! (Newtonian baryonic-only, NFW dark-matter halo, MOND, conformal gravity)
//! and compares them with honest per-galaxy and sample-level χ²/AIC/BIC.
//! Trains one HDC+CfC+GLU residual regressor per model as a supplementary
//! learnability diagnostic (see crate README for the important caveat that
//! this diagnostic is NOT apples-to-apples across models with different
//! free-parameter counts).
//!
//! ## Data
//! Requires the SPARC dataset (`scripts/download_sparc.sh`). Override the
//! location with `SYMTHAEA_SPARC_DATA_DIR`.
//!
//! ## Run
//! ```bash
//! bash scripts/download_sparc.sh
//! cargo run --release -p symthaea-galactic --example benchmark_sparc
//! ```

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use symthaea_galactic::fit::{aic, bic, reduced_chi2};
use symthaea_galactic::gravity_models::{
    ConformalGravity, Mond, Newtonian, NfwHalo, RotationModel,
};
use symthaea_galactic::sparc::{Galaxy, load_sparc};
use symthaea_galactic::validation::{
    evaluate_residual_regressor, low_surface_brightness_holdout, train_test_split,
};

fn sparc_data_dir() -> PathBuf {
    env::var("SYMTHAEA_SPARC_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/sparc"))
}

/// Epochs for the HDC residual regressor's manual gradient descent.
/// Override with `SYMTHAEA_SPARC_RESIDUAL_EPOCHS` — the 16,384-D vector ops
/// scale linearly with this, so it's the main lever for trading diagnostic
/// depth against wall-clock time on slower/contended machines.
fn residual_epochs() -> usize {
    env::var("SYMTHAEA_SPARC_RESIDUAL_EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
}

fn ephemeral_results_path() -> PathBuf {
    env::var("SYMTHAEA_SPARC_RESULTS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/sparc/results.json"))
}

/// Committed provenance artifact — deliberately outside `data/**` AND
/// outside any directory literally named `results/` (both are blanket
/// gitignored at the symthaea workspace root; verified against
/// `symthaea/.gitignore` before picking this path).
fn committed_results_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmark_provenance/sparc_benchmark.json")
}

fn git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Per-model summary across the full quality-cut sample.
struct ModelSummary {
    name: &'static str,
    n_free_params_per_galaxy: usize,
    total_chi2: f64,
    total_points: usize,
    total_params: usize,
    n_galaxies: usize,
    n_converged: usize,
}

fn summarize(model: &dyn RotationModel, galaxies: &[&Galaxy]) -> ModelSummary {
    let mut total_chi2 = 0.0;
    let mut total_points = 0;
    let mut n_converged = 0;
    for g in galaxies {
        let fit = model.fit(g);
        total_chi2 += fit.chi2;
        total_points += g.points.len();
        if fit.converged {
            n_converged += 1;
        }
    }
    ModelSummary {
        name: model.name(),
        n_free_params_per_galaxy: model.n_free_params(),
        total_chi2,
        total_points,
        total_params: model.n_free_params() * galaxies.len(),
        n_galaxies: galaxies.len(),
        n_converged,
    }
}

fn main() {
    println!("=== SPARC Rotation-Curve Model Comparison ===\n");

    let data_dir = sparc_data_dir();
    let galaxies = match load_sparc(&data_dir) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Could not load SPARC data: {e}");
            eprintln!("Run: bash scripts/download_sparc.sh");
            eprintln!("Or set SYMTHAEA_SPARC_DATA_DIR to an existing SPARC directory.");
            std::process::exit(1);
        }
    };
    println!(
        "Loaded {} galaxies from {}",
        galaxies.len(),
        data_dir.display()
    );

    // Quality cut: Q<=2 (high/medium quality), inclination >= 30deg
    // (Lelli, McGaugh & Schombert 2016 RAR convention).
    let cut: Vec<&Galaxy> = galaxies
        .iter()
        .filter(|g| g.quality <= 2 && g.inclination_deg >= 30.0)
        .collect();
    println!(
        "Quality cut (Q<=2, inc>=30deg): {} of {} galaxies\n",
        cut.len(),
        galaxies.len()
    );

    let models: Vec<Box<dyn RotationModel>> = vec![
        Box::new(Newtonian),
        Box::new(Mond),
        Box::new(ConformalGravity),
        Box::new(NfwHalo),
    ];

    println!(
        "{:<20} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "model", "k/gal", "chi2_total", "chi2/dof", "AIC", "BIC"
    );
    let mut model_json = Vec::new();
    for model in &models {
        let summary = summarize(model.as_ref(), &cut);
        let reduced = reduced_chi2(
            summary.total_chi2,
            summary.total_points,
            summary.total_params,
        );
        let total_aic = aic(summary.total_chi2, summary.total_params);
        let total_bic = bic(
            summary.total_chi2,
            summary.total_params,
            summary.total_points,
        );
        println!(
            "{:<20} {:>10} {:>12.1} {:>12.3} {:>12.1} {:>10.1}",
            summary.name,
            summary.n_free_params_per_galaxy,
            summary.total_chi2,
            reduced,
            total_aic,
            total_bic
        );
        if summary.n_converged < summary.n_galaxies {
            println!(
                "  [warn] {}/{} galaxies did not converge",
                summary.n_galaxies - summary.n_converged,
                summary.n_galaxies
            );
        }
        model_json.push(serde_json::json!({
            "name": summary.name,
            "n_free_params_per_galaxy": summary.n_free_params_per_galaxy,
            "total_chi2": summary.total_chi2,
            "total_points": summary.total_points,
            "total_params": summary.total_params,
            "reduced_chi2": reduced,
            "aic": total_aic,
            "bic": total_bic,
            "n_galaxies": summary.n_galaxies,
            "n_converged": summary.n_converged,
        }));
    }

    println!(
        "\nNote: AIC/BIC totals sum per-galaxy k across the sample. NFW's {} params\n\
         (2 x {} galaxies) buy it structural flexibility no 0-parameter model has —\n\
         see README for why this makes cross-model AIC/BIC comparison, not a fair fight\n\
         in the strict sense, but a real question about whether that flexibility is needed.",
        cut.len() * 2,
        cut.len()
    );

    // ── Residual-learnability diagnostic ────────────────────────────────
    println!("\n=== HDC Residual-Regressor Learnability Diagnostic ===");
    println!(
        "(held-out R^2 of a CfC+GLU regressor trained on each model's residuals;\n\
         see README: NOT directly comparable across models with different k)\n"
    );

    let (train_galaxies, test_galaxies) = train_test_split(&cut);
    println!(
        "Train/test split: {} train, {} test galaxies\n",
        train_galaxies.len(),
        test_galaxies.len()
    );

    println!(
        "{:<20} {:>12} {:>12} {:>10} {:>10}",
        "model", "R^2", "baseline_R^2", "MAE", "base_MAE"
    );
    let epochs = residual_epochs();
    let mut residual_json = Vec::new();
    for (i, model) in models.iter().enumerate() {
        let result = evaluate_residual_regressor(
            model.as_ref(),
            &train_galaxies,
            &test_galaxies,
            epochs,
            i as u64,
        );
        println!(
            "{:<20} {:>12.4} {:>12.4} {:>10.4} {:>10.4}",
            result.model_name,
            result.r_squared,
            result.baseline_r_squared,
            result.mae,
            result.baseline_mae
        );
        residual_json.push(serde_json::to_value(&result).unwrap());
    }

    // ── Extrapolation holdout: low-surface-brightness galaxies ─────────
    println!("\n=== Extrapolation Holdout: Low-Surface-Brightness Galaxies ===");
    let (kept, lsb_held_out) = low_surface_brightness_holdout(&cut);
    println!(
        "{} galaxies held out (lowest-quintile SBeff), {} used for training\n",
        lsb_held_out.len(),
        kept.len()
    );
    println!("{:<20} {:>12} {:>12}", "model", "R^2 (LSB)", "baseline_R^2");
    let mut lsb_json = Vec::new();
    for (i, model) in models.iter().enumerate() {
        let result = evaluate_residual_regressor(
            model.as_ref(),
            &kept,
            &lsb_held_out,
            epochs,
            1000 + i as u64,
        );
        println!(
            "{:<20} {:>12.4} {:>12.4}",
            result.model_name, result.r_squared, result.baseline_r_squared
        );
        lsb_json.push(serde_json::to_value(&result).unwrap());
    }

    // ── Write results ────────────────────────────────────────────────
    let results = serde_json::json!({
        "provenance": {
            "git_sha": git_sha(),
            "dataset": "SPARC (Lelli, McGaugh & Schombert 2016)",
            "data_dir": data_dir.display().to_string(),
            "n_galaxies_loaded": galaxies.len(),
            "n_galaxies_after_cut": cut.len(),
            "quality_cut": "Q<=2, inclination>=30deg",
            "upsilon_disk": symthaea_galactic::constants::UPSILON_DISK,
            "upsilon_bulge": symthaea_galactic::constants::UPSILON_BULGE,
            "residual_regressor_epochs": epochs,
        },
        "models": model_json,
        "residual_learnability": {
            "note": "NOT directly comparable across models with different free-parameter counts (see README)",
            "results": residual_json,
        },
        "lsb_extrapolation_holdout": {
            "n_held_out": lsb_held_out.len(),
            "results": lsb_json,
        },
    });

    let ephemeral = ephemeral_results_path();
    if let Some(parent) = ephemeral.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&ephemeral, serde_json::to_string_pretty(&results).unwrap())
        .unwrap_or_else(|e| eprintln!("warning: could not write {}: {e}", ephemeral.display()));
    println!("\nWrote ephemeral results to {}", ephemeral.display());

    let committed = committed_results_path();
    if let Some(parent) = committed.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&committed, serde_json::to_string_pretty(&results).unwrap())
        .unwrap_or_else(|e| eprintln!("warning: could not write {}: {e}", committed.display()));
    println!(
        "Wrote committed provenance artifact to {}",
        committed.display()
    );
}
