// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Bound Phenomenal Concept Probe
//!
//! Tests the combined H1+H2 hypothesis:
//! Do BOUND phenomenal concepts show stronger topological signatures than unbound?
//!
//! This experiment:
//! 1. Encodes phenomenal concept pairs via Neural Bridge
//! 2. Binds them using HDC XOR operation
//! 3. Probes both bound and unbound concepts
//! 4. Compares topological properties (unity, Betti numbers)
//!
//! Run:
//! ```bash
//! cargo run --example bound_phenomenal_probe --features neural-bridge --release
//! ```

#[cfg(feature = "neural-bridge")]
use anyhow::Result;
#[cfg(feature = "neural-bridge")]
use std::collections::HashMap;

#[cfg(feature = "neural-bridge")]
use symthaea::perception::{Concept, ConsciousnessProbeV2};

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::BinaryHV;

/// Phenomenal concept pairs that humans report as unified experiences
#[cfg(feature = "neural-bridge")]
const UNIFIED_PHENOMENAL_PAIRS: &[(&str, &str)] = &[
    (
        "The vivid experience of seeing red",
        "The felt warmth of sunlight",
    ),
    (
        "The taste of sweet chocolate",
        "The sensation of smooth texture",
    ),
    ("The feeling of deep sadness", "The heaviness in the chest"),
    ("The sound of a violin", "The emotional stirring it evokes"),
    (
        "The smell of fresh coffee",
        "The anticipation of the first sip",
    ),
];

/// Functional concept pairs (control)
#[cfg(feature = "neural-bridge")]
const FUNCTIONAL_PAIRS: &[(&str, &str)] = &[
    (
        "The recursive algorithm terminates",
        "Memory is allocated on the heap",
    ),
    (
        "Hash tables provide O(1) lookup",
        "TCP ensures reliable delivery",
    ),
    (
        "The compiler optimizes bytecode",
        "Garbage collection frees memory",
    ),
    (
        "Binary search has logarithmic complexity",
        "The function returns an integer",
    ),
    (
        "The database indexes the primary key",
        "Matrix multiplication computes dot products",
    ),
];

/// Single phenomenal concepts for comparison
#[cfg(feature = "neural-bridge")]
const SINGLE_PHENOMENAL: &[&str] = &[
    "The vivid experience of seeing red",
    "What it is like to taste chocolate",
    "The subjective feeling of pain",
    "The unified field of conscious awareness",
    "The felt quality of sadness",
];

/// Single functional concepts for comparison
#[cfg(feature = "neural-bridge")]
const SINGLE_FUNCTIONAL: &[&str] = &[
    "The recursive algorithm terminates",
    "Hash tables provide O(1) lookup",
    "The compiler optimizes bytecode",
    "Binary search has logarithmic complexity",
    "The database indexes the primary key",
];

#[cfg(feature = "neural-bridge")]
fn main() -> Result<()> {
    println!("\n{}", "=".repeat(70));
    println!("   BOUND PHENOMENAL CONCEPT PROBE");
    println!("   Testing Combined H1+H2 Hypothesis");
    println!("{}\n", "=".repeat(70));

    // Load probe
    println!("Loading ConsciousnessProbeV2 with BGE-M3...");
    let mut probe = ConsciousnessProbeV2::load_default()?;
    println!("Probe loaded successfully.\n");

    // Results storage
    let mut results: HashMap<String, Vec<f64>> = HashMap::new();
    results.insert("single_phenomenal_unity".to_string(), Vec::new());
    results.insert("single_functional_unity".to_string(), Vec::new());
    results.insert("bound_phenomenal_unity".to_string(), Vec::new());
    results.insert("bound_functional_unity".to_string(), Vec::new());
    results.insert("single_phenomenal_betti0".to_string(), Vec::new());
    results.insert("single_functional_betti0".to_string(), Vec::new());
    results.insert("bound_phenomenal_betti0".to_string(), Vec::new());
    results.insert("bound_functional_betti0".to_string(), Vec::new());

    // 1. Probe single phenomenal concepts
    println!("Probing single phenomenal concepts...");
    for text in SINGLE_PHENOMENAL {
        let result = probe.probe_text(text)?;
        results
            .get_mut("single_phenomenal_unity")
            .unwrap()
            .push(result.unity_score);
        results
            .get_mut("single_phenomenal_betti0")
            .unwrap()
            .push(result.betti.beta_0 as f64);
        println!(
            "  {} -> unity={:.4}, β₀={}",
            &text[..40.min(text.len())],
            result.unity_score,
            result.betti.beta_0
        );
    }

    // 2. Probe single functional concepts
    println!("\nProbing single functional concepts...");
    for text in SINGLE_FUNCTIONAL {
        let result = probe.probe_text(text)?;
        results
            .get_mut("single_functional_unity")
            .unwrap()
            .push(result.unity_score);
        results
            .get_mut("single_functional_betti0")
            .unwrap()
            .push(result.betti.beta_0 as f64);
        println!(
            "  {} -> unity={:.4}, β₀={}",
            &text[..40.min(text.len())],
            result.unity_score,
            result.betti.beta_0
        );
    }

    // 3. Probe BOUND phenomenal pairs
    println!("\n{}", "-".repeat(70));
    println!("Probing BOUND phenomenal concept pairs...");
    for (a, b) in UNIFIED_PHENOMENAL_PAIRS {
        // Encode both concepts
        let packed_a = probe.bridge_mut().encode_to_hdc(a)?;
        let packed_b = probe.bridge_mut().encode_to_hdc(b)?;

        // Convert to BinaryHV
        let hv_a = packed_to_hv16(&packed_a);
        let hv_b = packed_to_hv16(&packed_b);

        // BIND them
        let hv_bound = hv_a.bind(&hv_b);

        // Probe the bound vector
        let concept = Concept {
            id: format!("bound_phen_{}", a.len() + b.len()),
            text: format!("{} ⊗ {}", &a[..30.min(a.len())], &b[..30.min(b.len())]),
            category: "bound_phenomenal".to_string(),
            subcategory: "unified_pair".to_string(),
        };
        let result = probe.probe_concept_from_hv(&concept, &hv_bound)?;

        results
            .get_mut("bound_phenomenal_unity")
            .unwrap()
            .push(result.unity_score);
        results
            .get_mut("bound_phenomenal_betti0")
            .unwrap()
            .push(result.betti.beta_0 as f64);

        println!("  {} ⊗ {}", &a[..30.min(a.len())], &b[..30.min(b.len())]);
        println!(
            "    -> unity={:.4}, β₀={}",
            result.unity_score, result.betti.beta_0
        );
    }

    // 4. Probe BOUND functional pairs (control)
    println!("\nProbing BOUND functional concept pairs...");
    for (a, b) in FUNCTIONAL_PAIRS {
        let packed_a = probe.bridge_mut().encode_to_hdc(a)?;
        let packed_b = probe.bridge_mut().encode_to_hdc(b)?;

        let hv_a = packed_to_hv16(&packed_a);
        let hv_b = packed_to_hv16(&packed_b);

        let hv_bound = hv_a.bind(&hv_b);

        let concept = Concept {
            id: format!("bound_func_{}", a.len() + b.len()),
            text: format!("{} ⊗ {}", &a[..30.min(a.len())], &b[..30.min(b.len())]),
            category: "bound_functional".to_string(),
            subcategory: "pair".to_string(),
        };
        let result = probe.probe_concept_from_hv(&concept, &hv_bound)?;

        results
            .get_mut("bound_functional_unity")
            .unwrap()
            .push(result.unity_score);
        results
            .get_mut("bound_functional_betti0")
            .unwrap()
            .push(result.betti.beta_0 as f64);

        println!("  {} ⊗ {}", &a[..30.min(a.len())], &b[..30.min(b.len())]);
        println!(
            "    -> unity={:.4}, β₀={}",
            result.unity_score, result.betti.beta_0
        );
    }

    // Summary statistics
    println!("\n{}", "=".repeat(70));
    println!("   SUMMARY STATISTICS");
    println!("{}\n", "=".repeat(70));

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let std = |v: &[f64]| {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };

    let sp_unity = &results["single_phenomenal_unity"];
    let sf_unity = &results["single_functional_unity"];
    let bp_unity = &results["bound_phenomenal_unity"];
    let bf_unity = &results["bound_functional_unity"];

    println!(
        "{:<30} {:>12} {:>12}",
        "Condition", "Unity (mean)", "Unity (std)"
    );
    println!("{}", "-".repeat(54));
    println!(
        "{:<30} {:>12.4} {:>12.4}",
        "Single Phenomenal",
        mean(sp_unity),
        std(sp_unity)
    );
    println!(
        "{:<30} {:>12.4} {:>12.4}",
        "Single Functional",
        mean(sf_unity),
        std(sf_unity)
    );
    println!(
        "{:<30} {:>12.4} {:>12.4}",
        "Bound Phenomenal",
        mean(bp_unity),
        std(bp_unity)
    );
    println!(
        "{:<30} {:>12.4} {:>12.4}",
        "Bound Functional",
        mean(bf_unity),
        std(bf_unity)
    );

    // Effect sizes
    println!("\n{}", "-".repeat(70));
    println!("   EFFECT SIZE ANALYSIS (Cohen's d)");
    println!("{}\n", "-".repeat(70));

    let cohens_d = |a: &[f64], b: &[f64]| {
        let pooled_std = ((std(a).powi(2) + std(b).powi(2)) / 2.0).sqrt();
        if pooled_std > 0.0 {
            (mean(a) - mean(b)) / pooled_std
        } else {
            0.0
        }
    };

    let d_single = cohens_d(sp_unity, sf_unity);
    let d_bound = cohens_d(bp_unity, bf_unity);
    let d_binding_phen = cohens_d(bp_unity, sp_unity);
    let d_binding_func = cohens_d(bf_unity, sf_unity);

    println!("Single phenomenal vs functional:     d = {:+.3}", d_single);
    println!("Bound phenomenal vs functional:      d = {:+.3}", d_bound);
    println!(
        "Binding effect (phenomenal):         d = {:+.3}",
        d_binding_phen
    );
    println!(
        "Binding effect (functional):         d = {:+.3}",
        d_binding_func
    );

    // Key finding
    println!("\n{}", "=".repeat(70));
    println!("   KEY FINDING");
    println!("{}\n", "=".repeat(70));

    if d_bound > d_single && d_bound > 0.5 {
        println!("✓ COMBINED H1+H2 SUPPORTED");
        println!("  Bound phenomenal concepts show STRONGER topological signature");
        println!(
            "  than single concepts (d={:.3} vs d={:.3})",
            d_bound, d_single
        );
    } else if d_bound > 0.2 {
        println!("◐ PARTIAL SUPPORT");
        println!(
            "  Bound phenomenal concepts show moderate effect (d={:.3})",
            d_bound
        );
    } else {
        println!("✗ NOT SUPPORTED");
        println!("  Binding does not enhance phenomenal topological signature");
    }

    if d_binding_phen > d_binding_func + 0.2 {
        println!("\n✓ BINDING SPECIFICITY");
        println!("  Binding has stronger effect on phenomenal concepts");
        println!("  (Δd = {:.3})", d_binding_phen - d_binding_func);
    }

    // Save results
    let output = serde_json::json!({
        "single_phenomenal": {
            "unity_mean": mean(sp_unity),
            "unity_std": std(sp_unity),
            "n": sp_unity.len(),
        },
        "single_functional": {
            "unity_mean": mean(sf_unity),
            "unity_std": std(sf_unity),
            "n": sf_unity.len(),
        },
        "bound_phenomenal": {
            "unity_mean": mean(bp_unity),
            "unity_std": std(bp_unity),
            "n": bp_unity.len(),
        },
        "bound_functional": {
            "unity_mean": mean(bf_unity),
            "unity_std": std(bf_unity),
            "n": bf_unity.len(),
        },
        "effect_sizes": {
            "single_phen_vs_func": d_single,
            "bound_phen_vs_func": d_bound,
            "binding_effect_phenomenal": d_binding_phen,
            "binding_effect_functional": d_binding_func,
        }
    });

    let output_path = "data/bound_phenomenal_probe_results.json";
    std::fs::write(output_path, serde_json::to_string_pretty(&output)?)?;
    println!("\nResults saved to: {}", output_path);

    println!("\n{}\n", "=".repeat(70));

    Ok(())
}

#[cfg(feature = "neural-bridge")]
fn packed_to_hv16(packed: &symthaea_core::hdc::PackedBipolar) -> BinaryHV {
    // Convert PackedBipolar to BinaryHV
    let bipolar = packed.to_bipolar();
    // Convert i8 to f32 for from_bipolar
    let bipolar_f32: Vec<f32> = bipolar.iter().map(|&v| v as f32).collect();
    BinaryHV::from_bipolar(&bipolar_f32)
}

#[cfg(not(feature = "neural-bridge"))]
fn main() {
    eprintln!("This example requires the 'neural-bridge' feature.");
    eprintln!(
        "Run with: cargo run --example bound_phenomenal_probe --features neural-bridge --release"
    );
    std::process::exit(1);
}