// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-machine disruption prediction evaluation.
//!
//! If ITPA data exists in data/itpa/, runs cross-machine evaluation.
//! Otherwise, runs on C-Mod data only using the generic evaluator.
//! Falls back to synthetic data if no real data is available.
//!
//! Usage:
//!   cargo run -p symthaea-physics --example cross_machine_eval

use std::path::Path;

use symthaea_physics::cross_machine_evaluation::*;

fn main() -> anyhow::Result<()> {
    let itpa_dir = Path::new("data/itpa");
    let cmod_path = Path::new("data/cmod/DL_DataFrame.csv");

    let eval_config = GenericEvalConfig::default();

    // Check what data is available
    let has_itpa = itpa_dir.is_dir()
        && std::fs::read_dir(itpa_dir)
            .map(|mut d| {
                d.any(|e| {
                    e.map(|e| e.path().extension().map_or(false, |ext| ext == "csv"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);
    let has_cmod = cmod_path.is_file();

    if has_itpa {
        eprintln!("=== ITPA data found — running cross-machine evaluation ===\n");

        // Collect all CSV files in the ITPA directory
        let mut configs = Vec::new();

        if has_cmod {
            eprintln!("  Including C-Mod data");
            configs.push(MachineDatasetConfig::cmod());
        }

        // Add ITPA CSVs (each file assumed to be one machine)
        for entry in std::fs::read_dir(itpa_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "csv") {
                let name = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                eprintln!("  Found ITPA file: {}", path.display());

                // Use the ITPA template and override the path/name
                let mut cfg = MachineDatasetConfig::itpa_template();
                cfg.machine_name = name;
                cfg.csv_path = path;
                configs.push(cfg);
            }
        }

        if configs.is_empty() {
            eprintln!("No valid configs found, falling back to synthetic data.");
            run_synthetic(&eval_config)?;
        } else {
            let result = evaluate_cross_machine(&configs, &eval_config)?;
            let report = generate_cross_machine_report(&result);
            println!("{}", report);
        }
    } else if has_cmod {
        eprintln!("=== C-Mod data found — running single-machine generic evaluation ===\n");

        let config = MachineDatasetConfig::cmod();
        let result = evaluate_machine(&config, &eval_config)?;

        println!("# Single-Machine Evaluation: {}\n", result.machine_name);
        println!("| Metric | Value |");
        println!("|--------|-------|");
        println!("| Total shots | {} |", result.total_shots);
        println!("| Disrupted shots | {} |", result.disrupted_shots);
        println!("| Test samples | {} |", result.total_samples);
        println!("| Sensors | {} |", result.n_sensors);
        println!("| AUC V1 | {:.4} |", result.auc_v1);
        println!("| AUC V2 | {:.4} |", result.auc_v2);
        println!("| F1 V1 | {:.4} |", result.best_f1_v1);
        println!("| F1 V2 | {:.4} |", result.best_f1_v2);
        println!("| Lead time mean | {:.1} ms |", result.lead_time_mean_ms);
        println!(
            "| Lead time median | {:.1} ms |",
            result.lead_time_median_ms
        );
    } else {
        eprintln!("=== No real data found — running on synthetic data ===\n");
        run_synthetic(&eval_config)?;
    }

    Ok(())
}

fn run_synthetic(eval_config: &GenericEvalConfig) -> anyhow::Result<()> {
    eprintln!("Generating synthetic data for 2 machines...\n");

    // Create temp directory for synthetic CSVs
    let dir = tempfile::tempdir()?;

    let shots_a = generate_synthetic_machine("Alpha", 6, 30, 100, 0.3, 1234);
    let shots_b = generate_synthetic_machine("Beta", 6, 30, 100, 0.3, 5678);

    let path_a = write_synthetic_csv(&dir, "alpha.csv", &shots_a, 6);
    let path_b = write_synthetic_csv(&dir, "beta.csv", &shots_b, 6);

    let sensor_cols: Vec<String> = (0..6).map(|i| format!("sensor_{}", i)).collect();

    let config_a = MachineDatasetConfig {
        machine_name: "Synthetic Alpha".to_string(),
        csv_path: path_a,
        shot_id_column: "shot_id".to_string(),
        time_column: "time".to_string(),
        label_column: "disrupted".to_string(),
        sensor_columns: sensor_cols.clone(),
        label_positive_value: "1".to_string(),
    };

    let config_b = MachineDatasetConfig {
        machine_name: "Synthetic Beta".to_string(),
        csv_path: path_b,
        shot_id_column: "shot_id".to_string(),
        time_column: "time".to_string(),
        label_column: "disrupted".to_string(),
        sensor_columns: sensor_cols,
        label_positive_value: "1".to_string(),
    };

    let result = evaluate_cross_machine(&[config_a, config_b], eval_config)?;
    let report = generate_cross_machine_report(&result);
    println!("{}", report);

    Ok(())
}

/// Write synthetic shots to CSV.
fn write_synthetic_csv(
    dir: &tempfile::TempDir,
    filename: &str,
    shots: &[GenericShot],
    n_sensors: usize,
) -> std::path::PathBuf {
    use std::io::Write;

    let path = dir.path().join(filename);
    let mut f = std::fs::File::create(&path).unwrap();

    let mut header = "shot_id,time,disrupted".to_string();
    for i in 0..n_sensors {
        header.push_str(&format!(",sensor_{}", i));
    }
    writeln!(f, "{}", header).unwrap();

    for shot in shots {
        for sample in &shot.samples {
            let label = if sample.is_disruption { "1" } else { "0" };
            let mut row = format!("{},{},{}", sample.shot_id, sample.time_s, label);
            for v in &sample.sensor_values {
                row.push_str(&format!(",{}", v));
            }
            writeln!(f, "{}", row).unwrap();
        }
    }

    path
}
