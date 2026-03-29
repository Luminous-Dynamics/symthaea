// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Language FL Test Fixture Generator
//!
//! Generates deterministic JSON fixtures that Rust, TypeScript, and Python
//! SDKs can all load and verify. Ensures all implementations produce
//! identical results for identical inputs.
//!
//! Run: `cargo run --example generate_fl_fixtures --release`
//! Output: `fixtures/fl_test_vectors.json`

use mycelix_fl_core::*;
use std::collections::HashMap;
use std::fs;

fn round6(v: f32) -> f64 {
    ((v as f64) * 1_000_000.0).round() / 1_000_000.0
}

fn round6_vec(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| round6(x)).collect()
}

fn main() {
    println!("=== FL Cross-Language Fixture Generator ===\n");

    // Deterministic inputs: 10 gradient updates with 5 dimensions
    let dim = 5;
    let updates: Vec<GradientUpdate> = vec![
        // 5 honest participants with consistent gradients
        GradientUpdate::new(
            "honest_0".into(),
            1,
            vec![0.10, 0.20, 0.30, 0.40, 0.50],
            100,
            0.50,
        ),
        GradientUpdate::new(
            "honest_1".into(),
            1,
            vec![0.12, 0.22, 0.28, 0.42, 0.48],
            200,
            0.45,
        ),
        GradientUpdate::new(
            "honest_2".into(),
            1,
            vec![0.11, 0.19, 0.31, 0.39, 0.51],
            150,
            0.48,
        ),
        GradientUpdate::new(
            "honest_3".into(),
            1,
            vec![0.09, 0.21, 0.29, 0.41, 0.49],
            100,
            0.52,
        ),
        GradientUpdate::new(
            "honest_4".into(),
            1,
            vec![0.13, 0.18, 0.32, 0.38, 0.52],
            120,
            0.47,
        ),
        // 3 Byzantine participants with extreme gradients
        GradientUpdate::new(
            "byz_0".into(),
            1,
            vec![50.0, -30.0, 80.0, -60.0, 100.0],
            100,
            0.90,
        ),
        GradientUpdate::new(
            "byz_1".into(),
            1,
            vec![-40.0, 70.0, -50.0, 90.0, -80.0],
            100,
            0.85,
        ),
        GradientUpdate::new(
            "byz_2".into(),
            1,
            vec![30.0, 30.0, 30.0, 30.0, 30.0],
            100,
            0.88,
        ),
        // 2 low-reputation participants with mild gradients
        GradientUpdate::new(
            "lowrep_0".into(),
            1,
            vec![0.50, 0.50, 0.50, 0.50, 0.50],
            80,
            0.60,
        ),
        GradientUpdate::new(
            "lowrep_1".into(),
            1,
            vec![0.40, 0.60, 0.40, 0.60, 0.40],
            80,
            0.55,
        ),
    ];

    let mut reputations: HashMap<String, f32> = HashMap::new();
    reputations.insert("honest_0".into(), 0.90);
    reputations.insert("honest_1".into(), 0.85);
    reputations.insert("honest_2".into(), 0.88);
    reputations.insert("honest_3".into(), 0.87);
    reputations.insert("honest_4".into(), 0.86);
    reputations.insert("byz_0".into(), 0.15);
    reputations.insert("byz_1".into(), 0.12);
    reputations.insert("byz_2".into(), 0.18);
    reputations.insert("lowrep_0".into(), 0.40);
    reputations.insert("lowrep_1".into(), 0.35);

    // Compute expected outputs for each algorithm
    println!("Computing FedAvg...");
    let fedavg_result = fedavg(&updates).unwrap();

    println!("Computing TrimmedMean(0.2)...");
    let trimmed_result = trimmed_mean(&updates, 0.2).unwrap();

    println!("Computing CoordinateMedian...");
    let median_result = coordinate_median(&updates).unwrap();

    println!("Computing Krum(3)...");
    let krum_result = krum(&updates, 3).unwrap();

    println!("Computing TrustWeighted(0.5)...");
    let trust_result = trust_weighted(&updates, &reputations, 0.5).unwrap();

    // Run Byzantine detection
    println!("Running EarlyByzantineDetector...");
    let early_detector = EarlyByzantineDetector::new();
    let early_result = early_detector.detect(&updates);

    println!("Running MultiSignalByzantineDetector...");
    let multi_detector = MultiSignalByzantineDetector::new();
    let multi_result = multi_detector.detect(&updates);

    // Build JSON fixture
    let fixture = format!(
        r#"{{
  "version": "1.0.0",
  "description": "Cross-language FL test vectors for Rust, TypeScript, and Python SDKs",
  "precision_note": "All values rounded to 6 decimal places. Cross-language tolerance: 1e-4",
  "inputs": {{
    "dimension": {dim},
    "updates": [
{updates_json}
    ],
    "reputations": {{
{reps_json}
    }}
  }},
  "expected_outputs": {{
    "fedavg": {{
      "gradients": {fedavg_json}
    }},
    "trimmed_mean_0_2": {{
      "trim_percentage": 0.2,
      "gradients": {trimmed_json}
    }},
    "coordinate_median": {{
      "gradients": {median_json}
    }},
    "krum_3": {{
      "num_select": 3,
      "gradients": {krum_json}
    }},
    "trust_weighted_0_5": {{
      "trust_threshold": 0.5,
      "gradients": {trust_json},
      "participant_count": {trust_count},
      "excluded_count": {trust_excluded}
    }}
  }},
  "byzantine_detection": {{
    "early_detector": {{
      "byzantine_indices": {early_byz_indices:?},
      "early_terminated": {early_terminated}
    }},
    "multi_signal_detector": {{
      "byzantine_indices": {multi_byz_indices:?},
      "early_terminated": {multi_terminated},
      "stats": {{
        "mean_norm": {mean_norm},
        "std_norm": {std_norm},
        "mean_cosine_sim": {mean_cos},
        "participants_analyzed": {participants_analyzed}
      }}
    }}
  }}
}}"#,
        dim = dim,
        updates_json = updates
            .iter()
            .map(|u| format!(
                r#"      {{
        "participant_id": "{}",
        "model_version": {},
        "gradients": {:?},
        "batch_size": {},
        "loss": {}
      }}"#,
                u.participant_id,
                u.model_version,
                round6_vec(&u.gradients),
                u.metadata.batch_size,
                round6(u.metadata.loss)
            ))
            .collect::<Vec<_>>()
            .join(",\n"),
        reps_json = {
            let mut sorted_reps: Vec<_> = reputations.iter().collect();
            sorted_reps.sort_by_key(|(k, _)| (*k).clone());
            sorted_reps
                .iter()
                .map(|(k, v)| format!(r#"      "{}": {}"#, k, round6(**v)))
                .collect::<Vec<_>>()
                .join(",\n")
        },
        fedavg_json = format!("{:?}", round6_vec(&fedavg_result)),
        trimmed_json = format!("{:?}", round6_vec(&trimmed_result)),
        median_json = format!("{:?}", round6_vec(&median_result)),
        krum_json = format!("{:?}", round6_vec(&krum_result)),
        trust_json = format!("{:?}", round6_vec(&trust_result.gradients)),
        trust_count = trust_result.participant_count,
        trust_excluded = trust_result.excluded_count,
        early_byz_indices = early_result.byzantine_indices,
        early_terminated = early_result.early_terminated,
        multi_byz_indices = multi_result.byzantine_indices,
        multi_terminated = multi_result.early_terminated,
        mean_norm = round6(multi_result.stats.mean_norm),
        std_norm = round6(multi_result.stats.std_norm),
        mean_cos = round6(multi_result.stats.mean_cosine_sim),
        participants_analyzed = multi_result.stats.participants_analyzed,
    );

    // Write to fixtures directory
    let fixtures_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    fs::create_dir_all(&fixtures_dir).unwrap();
    let output_path = fixtures_dir.join("fl_test_vectors.json");
    fs::write(&output_path, &fixture).unwrap();
    println!("\nFixtures written to: {}", output_path.display());

    // Print summary
    println!("\n=== Summary ===");
    println!(
        "Updates: {} ({} honest, 3 Byzantine, 2 low-rep)",
        updates.len(),
        5
    );
    println!("FedAvg: {:?}", round6_vec(&fedavg_result));
    println!("TrimmedMean: {:?}", round6_vec(&trimmed_result));
    println!("Median: {:?}", round6_vec(&median_result));
    println!("Krum(3): {:?}", round6_vec(&krum_result));
    println!(
        "TrustWeighted: {:?} (included={}, excluded={})",
        round6_vec(&trust_result.gradients),
        trust_result.participant_count,
        trust_result.excluded_count
    );
    println!("Early Byz: {:?}", early_result.byzantine_indices);
    println!("MultiSignal Byz: {:?}", multi_result.byzantine_indices);
    println!("\n=== ALL FIXTURES GENERATED ===");
}
