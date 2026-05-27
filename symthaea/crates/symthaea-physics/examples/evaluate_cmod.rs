// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evaluate Fusion Digital Twin against real C-Mod density limit data.
//!
//! Runs V1 (original), V2 (temporal windowing + per-shot reference), and
//! V3 (physics-informed: Greenwald fraction + rate-of-change features)
//! evaluations and prints a side-by-side comparison.
//!
//! Usage:
//!   cargo run -p symthaea-physics --example evaluate_cmod
//!   cargo run -p symthaea-physics --example evaluate_cmod -- --json report.json

use std::path::PathBuf;

use symthaea_physics::cmod_evaluation::{
    EvaluationConfig, EvaluationConfigV2, EvaluationConfigV3, evaluate_density_limit,
    evaluate_density_limit_v2, evaluate_density_limit_v3,
};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Default path relative to workspace root
    let csv_path = PathBuf::from(
        std::env::var("CMOD_CSV_PATH").unwrap_or_else(|_| "data/cmod/DL_DataFrame.csv".into()),
    );

    // Optional JSON output path
    let json_path = if let Some(idx) = args.iter().position(|a| a == "--json") {
        args.get(idx + 1).map(PathBuf::from)
    } else {
        None
    };

    eprintln!(
        "Loading C-Mod density limit data from: {}",
        csv_path.display()
    );

    // ── V1: Original evaluation ──────────────────────────────────────────
    eprintln!("\n=== Running V1 (original) evaluation ===");
    let config_v1 = EvaluationConfig::default();
    let report_v1 = evaluate_density_limit(&csv_path, &config_v1)?;

    println!("{}", report_v1.to_markdown());

    eprintln!("--- V1 Summary ---");
    eprintln!("Best threshold: {:.4}", report_v1.best_threshold);
    eprintln!("Best F1:        {:.4}", report_v1.best_f1);
    eprintln!("AUC:            {:.4}", report_v1.auc);

    // ── V2: Improved evaluation ──────────────────────────────────────────
    eprintln!("\n=== Running V2 (temporal window + per-shot ref) evaluation ===");
    let config_v2 = EvaluationConfigV2::default();
    let report_v2 = evaluate_density_limit_v2(&csv_path, &config_v2)?;

    println!("{}", report_v2.to_markdown());

    eprintln!("--- V2 Summary ---");
    eprintln!("Best threshold: {:.4}", report_v2.best_threshold);
    eprintln!("Best F1:        {:.4}", report_v2.best_f1);
    eprintln!("AUC:            {:.4}", report_v2.auc);

    // ── V3: Physics-informed evaluation ──────────────────────────────────
    eprintln!("\n=== Running V3 (physics-informed: Greenwald + rates) evaluation ===");
    let config_v3 = EvaluationConfigV3::default();
    let report_v3 = evaluate_density_limit_v3(&csv_path, &config_v3)?;

    println!("{}", report_v3.to_markdown());

    eprintln!("--- V3 Summary ---");
    eprintln!("Best threshold: {:.4}", report_v3.best_threshold);
    eprintln!("Best F1:        {:.4}", report_v3.best_f1);
    eprintln!("AUC:            {:.4}", report_v3.auc);

    // ── Side-by-side comparison ──────────────────────────────────────────
    let auc_v1v2 = report_v2.auc - report_v1.auc;
    let auc_v2v3 = report_v3.auc - report_v2.auc;
    let auc_v1v3 = report_v3.auc - report_v1.auc;

    eprintln!("\n================================================================");
    eprintln!("           V1 vs V2 vs V3 Comparison");
    eprintln!("================================================================");
    eprintln!("              V1        V2        V3        V1→V3");
    eprintln!(
        "AUC:          {:.4}    {:.4}    {:.4}    {:+.4}",
        report_v1.auc, report_v2.auc, report_v3.auc, auc_v1v3
    );
    eprintln!(
        "Best F1:      {:.4}    {:.4}    {:.4}    {:+.4}",
        report_v1.best_f1,
        report_v2.best_f1,
        report_v3.best_f1,
        report_v3.best_f1 - report_v1.best_f1
    );
    eprintln!(
        "Threshold:    {:.4}    {:.4}    {:.4}",
        report_v1.best_threshold, report_v2.best_threshold, report_v3.best_threshold
    );
    eprintln!(
        "Precision:    {:.4}    {:.4}    {:.4}",
        report_v1.confusion_matrix.precision(),
        report_v2.confusion_matrix.precision(),
        report_v3.confusion_matrix.precision()
    );
    eprintln!(
        "Recall:       {:.4}    {:.4}    {:.4}",
        report_v1.confusion_matrix.recall(),
        report_v2.confusion_matrix.recall(),
        report_v3.confusion_matrix.recall()
    );
    eprintln!(
        "Lead time:    {:.4}s   {:.4}s   {:.4}s",
        report_v1.lead_time_stats.mean_s,
        report_v2.lead_time_stats.mean_s,
        report_v3.lead_time_stats.mean_s
    );
    eprintln!("----------------------------------------------------------------");
    eprintln!("V1→V2 AUC: {:+.4}", auc_v1v2);
    eprintln!("V2→V3 AUC: {:+.4}", auc_v2v3);
    eprintln!("V1→V3 AUC: {:+.4}", auc_v1v3);
    eprintln!("================================================================");

    // Optionally write V3 JSON (the best one)
    if let Some(path) = json_path {
        let json = report_v3.to_json()?;
        std::fs::write(&path, &json)?;
        eprintln!("V3 JSON report written to: {}", path.display());
    }

    Ok(())
}
