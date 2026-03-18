//! Evaluate Fusion Digital Twin against real C-Mod density limit data.
//!
//! Usage:
//!   cargo run -p symthaea-physics --example evaluate_cmod
//!   cargo run -p symthaea-physics --example evaluate_cmod -- --json report.json

use std::path::PathBuf;

use symthaea_physics::cmod_evaluation::{evaluate_density_limit, EvaluationConfig};

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

    eprintln!("Loading C-Mod density limit data from: {}", csv_path.display());

    let config = EvaluationConfig::default();
    let report = evaluate_density_limit(&csv_path, &config)?;

    // Print markdown report to stdout
    println!("{}", report.to_markdown());

    // Optionally write JSON
    if let Some(path) = json_path {
        let json = report.to_json()?;
        std::fs::write(&path, &json)?;
        eprintln!("JSON report written to: {}", path.display());
    }

    // Print summary to stderr
    eprintln!("--- Summary ---");
    eprintln!("Best threshold: {:.4}", report.best_threshold);
    eprintln!("Best F1:        {:.4}", report.best_f1);
    eprintln!("AUC:            {:.4}", report.auc);
    eprintln!(
        "Train/Test:     {} / {} shots",
        report.total_shots_train, report.total_shots_test
    );
    eprintln!(
        "Test samples:   {} normal, {} disruption",
        report.samples_per_class.0, report.samples_per_class.1
    );

    Ok(())
}
