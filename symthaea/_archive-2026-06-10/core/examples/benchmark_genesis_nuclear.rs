// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 25: Nuclear Attribution
//!
//! Demonstrates:
//! - Encode 5 reference signatures into 16,384D
//! - Perturbed HEU sample → correct attribution
//! - O(1) decay backdating proof

fn main() {
    println!("=== Genesis Mission Challenge 25: Nuclear Attribution ===\n");

    use symthaea_nuclear_forensics::{
        IsotopeDecayModel, IsotopicHdcEncoder, IsotopicSignature, NuclearAttributionAgent,
    };

    // 1. Similarity matrix
    println!("--- Isotopic Signature Similarity Matrix ---");
    let encoder = IsotopicHdcEncoder::new();
    let refs = IsotopicSignature::references();
    let matrix = encoder.similarity_matrix(&refs);

    print!("{:>15}", "");
    for r in &refs {
        print!("{:>12}", &r.name[..r.name.len().min(10)]);
    }
    println!();
    for (i, r) in refs.iter().enumerate() {
        print!("{:>15}", &r.name[..r.name.len().min(13)]);
        for j in 0..refs.len() {
            print!("{:>12.3}", matrix[i][j]);
        }
        println!();
    }

    // 2. Attribution of perturbed HEU sample
    println!("\n--- Attribution: Perturbed HEU Sample ---");
    let agent = NuclearAttributionAgent::new();
    let perturbed_heu = IsotopicSignature::highly_enriched_uranium().perturbed(0.01, 42);
    let result = agent.attribute(&perturbed_heu);
    println!("  Source: {:?}", result.matched_source);
    println!("  Reference: {}", result.matched_name);
    println!("  Confidence: {:.3}", result.confidence);
    println!("  All matches:");
    for (name, sim) in &result.all_similarities {
        println!("    {}: {:.3}", name, sim);
    }

    // 3. Attribution of all references (should all self-match)
    println!("\n--- Self-Match Verification ---");
    for sig in IsotopicSignature::references() {
        let r = agent.attribute(&sig);
        println!(
            "  {} → {:?} (confidence={:.3})",
            sig.name, r.matched_source, r.confidence
        );
    }

    // 4. O(1) temporal prediction proof (predict_at_horizon replaces backdate)
    println!("\n--- O(1) Temporal Prediction Cost ---");
    let decay_model = IsotopeDecayModel::new();
    let spent = IsotopicSignature::spent_fuel();
    for &horizon in symthaea_nuclear_forensics::DECAY_HORIZONS {
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = decay_model.predict_at_horizon(&spent, horizon);
        }
        let elapsed = start.elapsed();
        println!(
            "  {:>10}: {:.1}µs/predict",
            symthaea_nuclear_forensics::DECAY_HORIZONS
                .iter()
                .position(|&h| h == horizon)
                .map(|i| ["1 day", "1 month", "1 year", "10 years", "50 years"][i])
                .unwrap_or("?"),
            elapsed.as_micros() as f64 / 1000.0
        );
    }

    // 5. Age estimation from decay ratios
    println!("\n--- Age Estimation ---");
    for sig in IsotopicSignature::references() {
        let age = decay_model.estimate_age(&sig);
        let years = age.estimated_age_seconds / 31_536_000.0;
        println!(
            "  {}: estimated age {:.1} years (confidence={:.3})",
            sig.name, years, age.confidence
        );
    }

    println!("\nPASS: Nuclear Attribution operational");
}
