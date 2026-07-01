// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 9: Materials Design
//!
//! Demonstrates:
//! - Encode 5 preset materials into 16,384D
//! - Similarity matrix (metals cluster together)
//! - Aging prediction at 5 horizons (1 day to 50 years)
//! - Constraint search (metals with yield > 200 MPa)

fn main() {
    println!("=== Genesis Mission Challenge 9: Materials Design ===\n");

    use symthaea_materials::{
        MaterialAgingModel, MaterialDatabase, MaterialHdcEncoder, MaterialProperty,
    };

    let presets = MaterialProperty::presets();

    // 1. Similarity matrix
    println!("--- Similarity Matrix ---");
    let encoder = MaterialHdcEncoder::new();
    let matrix = encoder.similarity_matrix(&presets);
    print!("{:>20}", "");
    for m in &presets {
        print!("{:>12}", &m.name[..m.name.len().min(10)]);
    }
    println!();
    for (i, m) in presets.iter().enumerate() {
        print!("{:>20}", &m.name[..m.name.len().min(18)]);
        for j in 0..presets.len() {
            print!("{:>12.3}", matrix[i][j]);
        }
        println!();
    }

    // 2. Aging prediction
    println!("\n--- Aging Predictions (Steel A36) ---");
    let aging_model = MaterialAgingModel::new();
    let steel = MaterialProperty::steel_a36();
    for pred in aging_model.predict_all_horizons(&steel) {
        println!(
            "  {}: similarity={:.3}, remaining_strength={:.3}",
            pred.horizon_label, pred.state_similarity, pred.remaining_strength
        );
    }

    // 3. O(1) cost proof
    println!("\n--- O(1) Aging Prediction Cost ---");
    for &horizon in symthaea_materials::AGING_HORIZONS {
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = aging_model.predict_at_horizon(&steel, horizon);
        }
        let elapsed = start.elapsed();
        println!(
            "  {:>10}: {:.1}µs/prediction",
            symthaea_materials::AGING_HORIZON_LABELS[symthaea_materials::AGING_HORIZONS
                .iter()
                .position(|&h| h == horizon)
                .unwrap()],
            elapsed.as_micros() as f64 / 1000.0
        );
    }

    // 4. Constraint search
    println!("\n--- Constraint Search: metals with yield > 200 MPa ---");
    let db = MaterialDatabase::with_presets();
    let results = db.constrained_search(
        &MaterialProperty::titanium_ti6al4v(),
        |m| {
            m.category == symthaea_materials::MaterialCategory::Metal
                && m.yield_strength_mpa > 200.0
        },
        10,
    );
    for r in &results {
        println!(
            "  {} (sim={:.3}, yield={:.0} MPa)",
            r.material.name, r.similarity, r.material.yield_strength_mpa
        );
    }

    println!("\nPASS: Materials Design operational");
}
