// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Consciousness Emergence Experiment
//!
//! Tests the hypothesis that phenomenal character emerges through compositional
//! binding in the physics hierarchy.
//!
//! ## Core Hypotheses
//!
//! **H1**: Higher levels of composition (molecules > atoms > hadrons > quarks)
//!         exhibit increasing phenomenal character.
//!
//! **H2**: HDC binding (⊗) produces higher phenomenal integration than
//!         bundling (⊕) for the same constituents.
//!
//! **H3**: "Embodied qualia" (physical substrate bound with phenomenal concepts)
//!         has measurably higher phenomenal index than bare physical substrate.
//!
//! ## Background
//!
//! Integrated Information Theory (IIT) suggests that consciousness correlates
//! with Φ (phi) - a measure of integrated information. We test whether:
//!
//! 1. Physical complexity correlates with phenomenal character
//! 2. Binding operations increase integration (unity)
//! 3. Phenomenal concepts can be "grafted onto" physical substrates
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example consciousness_emergence_experiment --release
//! ```

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::physics::chemistry::Chemistry;
use symthaea_core::physics::{Hadrons, PeriodicTable, PhysicsConsciousnessBridge, StandardModel};

fn main() {
    println!("\n");
    println!("================================================================");
    println!("   CONSCIOUSNESS EMERGENCE EXPERIMENT");
    println!("   Testing phenomenal character across physics scales");
    println!("================================================================\n");

    let genesis = GenesisSeed::from_phrase("Consciousness Research");
    let model = StandardModel::from_genesis(&genesis);
    let hadrons = Hadrons::from_model(&model, &genesis);
    let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
    let chemistry = Chemistry::from_genesis(&genesis, &table);
    let bridge = PhysicsConsciousnessBridge::from_genesis(&genesis);

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("EXPERIMENT 1: COMPOSITIONAL EMERGENCE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("H1: Does phenomenal character increase with compositional complexity?\n");

    let analysis = bridge.compositional_phenomenal_analysis(&model, &hadrons, &table, &chemistry);

    println!("Phenomenal Index by Layer:");
    println!("  (Range: -1.0 = purely functional, +1.0 = purely phenomenal)\n");

    println!("  Level 0 (Quarks):     {:+.4}", analysis.level_0_quarks);
    println!("  Level 1 (Hadrons):    {:+.4}", analysis.level_1_hadrons);
    println!("  Level 2 (Atoms):      {:+.4}", analysis.level_2_atoms);
    println!("  Level 3 (Molecules):  {:+.4}", analysis.level_3_molecules);

    let gradient = analysis.emergence_gradient();
    println!("\nEmergence Gradient: {:+.4} per level", gradient);

    let emergence = analysis.shows_emergence();
    println!(
        "\nH1 Result: {}",
        if emergence {
            "SUPPORTED - Phenomenal character increases with complexity"
        } else {
            "NOT SUPPORTED - No clear emergence pattern"
        }
    );

    // Detailed breakdown
    println!("\nDetailed Analysis:");
    let quark_phen = bridge.estimate_phenomenal_index(&model.up_quark);
    let electron_phen = bridge.estimate_phenomenal_index(&model.electron);
    let proton_phen = bridge.estimate_phenomenal_index(&hadrons.proton);
    let neutron_phen = bridge.estimate_phenomenal_index(&hadrons.neutron);

    let h = table.element(1).unwrap();
    let c = table.element(6).unwrap();
    let o = table.element(8).unwrap();
    let water = chemistry.water(&table);

    println!("  Up Quark:    {:+.4}", quark_phen);
    println!("  Electron:    {:+.4}", electron_phen);
    println!("  Proton:      {:+.4}", proton_phen);
    println!("  Neutron:     {:+.4}", neutron_phen);
    println!(
        "  Hydrogen:    {:+.4}",
        bridge.estimate_phenomenal_index(&h.vector)
    );
    println!(
        "  Carbon:      {:+.4}",
        bridge.estimate_phenomenal_index(&c.vector)
    );
    println!(
        "  Oxygen:      {:+.4}",
        bridge.estimate_phenomenal_index(&o.vector)
    );
    println!(
        "  Water:       {:+.4}",
        bridge.estimate_phenomenal_index(&water.vector)
    );

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("EXPERIMENT 2: BINDING vs BUNDLING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("H2: Does binding create higher unity than bundling?\n");

    // Test multiple pairs
    let test_pairs: Vec<(&str, &ContinuousHV, &str, &ContinuousHV)> = vec![
        ("Up Quark", &model.up_quark, "Down Quark", &model.down_quark),
        ("Proton", &hadrons.proton, "Neutron", &hadrons.neutron),
        ("Hydrogen", &h.vector, "Oxygen", &o.vector),
        ("Carbon", &c.vector, "Oxygen", &o.vector),
    ];

    println!(
        "{:<24} {:>12} {:>12} {:>10}",
        "Pair", "Bind Unity", "Bundle Unity", "Δ Bind-Bundle"
    );
    println!("{}", "-".repeat(60));

    let mut total_advantage = 0.0;
    let mut count = 0;

    for (name_a, vec_a, name_b, vec_b) in &test_pairs {
        let result = bridge.binding_unity_test(vec_a, vec_b);

        println!(
            "{:<24} {:>12.4} {:>12.4} {:>+10.4}",
            format!("{} + {}", name_a, name_b),
            result.bound_unity,
            result.bundled_unity,
            result.binding_advantage
        );

        total_advantage += result.binding_advantage;
        count += 1;
    }

    let avg_advantage = total_advantage / count as f32;
    println!("\nMean Binding Advantage: {:+.4}", avg_advantage);

    println!(
        "\nH2 Result: {}",
        if avg_advantage > 0.0 {
            "SUPPORTED - Binding produces higher unity than bundling"
        } else {
            "NOT SUPPORTED - Bundling produces equal or higher unity"
        }
    );

    // Integration comparison
    println!("\nIntegration Analysis (Bound vs Bundled):");
    for (name_a, vec_a, name_b, vec_b) in &test_pairs {
        let result = bridge.binding_unity_test(vec_a, vec_b);
        println!("  {} + {}:", name_a, name_b);
        println!("    Bound Integration:   {:.4}", result.bound_integration);
        println!("    Bundled Integration: {:.4}", result.bundled_integration);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("EXPERIMENT 3: EMBODIMENT EFFECT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("H3: Does embodiment increase phenomenal character?\n");

    // Test embodiment on various physical substrates
    let substrates: Vec<(&str, &ContinuousHV)> = vec![
        ("Electron", &model.electron),
        ("Proton", &hadrons.proton),
        ("Carbon", &c.vector),
        ("Water", &water.vector),
    ];

    println!(
        "{:<12} {:>14} {:>14} {:>12}",
        "Substrate", "Bare Phen", "Embodied Phen", "Δ Embodied"
    );
    println!("{}", "-".repeat(54));

    let mut total_delta = 0.0;
    let mut positive_count = 0;
    let mut test_count = 0;

    for (name, vec) in &substrates {
        let bare_phen = bridge.estimate_phenomenal_index(vec);
        let embodied = bridge.embody_qualia(vec);
        let embodied_phen = bridge.estimate_phenomenal_index(&embodied);
        let delta = embodied_phen - bare_phen;

        println!(
            "{:<12} {:>+14.4} {:>+14.4} {:>+12.4}",
            name, bare_phen, embodied_phen, delta
        );

        total_delta += delta;
        if delta > 0.0 {
            positive_count += 1;
        }
        test_count += 1;
    }

    let avg_delta = total_delta / test_count as f32;
    println!("\nMean Embodiment Effect: {:+.4}", avg_delta);
    println!("Positive Effects: {}/{}", positive_count, test_count);

    println!(
        "\nH3 Result: {}",
        if avg_delta > 0.0 && positive_count > test_count / 2 {
            "SUPPORTED - Embodiment increases phenomenal character"
        } else {
            "NOT SUPPORTED - Embodiment does not reliably increase phenomenal character"
        }
    );

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("EXPERIMENT 4: PHENOMENAL vs FUNCTIONAL DISCRIMINATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Testing class separation in consciousness concept space.\n");

    // Create labeled vectors
    let labeled_vectors: Vec<(&ContinuousHV, bool)> = vec![
        (&bridge.qualia, true),
        (&bridge.awareness, true),
        (&bridge.subjective, true),
        (&bridge.experience, true),
        (&bridge.unity, true),
        (&bridge.computation, false),
        (&bridge.information, false),
        (&bridge.processing, false),
        (&bridge.algorithm, false),
        (&bridge.function, false),
    ];

    let fisher = bridge.compute_discrimination(&labeled_vectors);

    println!("Fisher's Criterion: {:.4}", fisher);
    println!("  (Higher = better separation between phenomenal and functional)");

    println!("\nIndividual Phenomenal Indices:");
    println!("\n  Phenomenal Concepts:");
    println!(
        "    Qualia:     {:+.4}",
        bridge.estimate_phenomenal_index(&bridge.qualia)
    );
    println!(
        "    Awareness:  {:+.4}",
        bridge.estimate_phenomenal_index(&bridge.awareness)
    );
    println!(
        "    Subjective: {:+.4}",
        bridge.estimate_phenomenal_index(&bridge.subjective)
    );
    println!(
        "    Experience: {:+.4}",
        bridge.estimate_phenomenal_index(&bridge.experience)
    );
    println!(
        "    Unity:      {:+.4}",
        bridge.estimate_phenomenal_index(&bridge.unity)
    );

    println!("\n  Functional Concepts:");
    println!(
        "    Computation: {:+.4}",
        bridge.estimate_phenomenal_index(&bridge.computation)
    );
    println!(
        "    Information: {:+.4}",
        bridge.estimate_phenomenal_index(&bridge.information)
    );
    println!(
        "    Processing:  {:+.4}",
        bridge.estimate_phenomenal_index(&bridge.processing)
    );
    println!(
        "    Algorithm:   {:+.4}",
        bridge.estimate_phenomenal_index(&bridge.algorithm)
    );
    println!(
        "    Function:    {:+.4}",
        bridge.estimate_phenomenal_index(&bridge.function)
    );

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("EXPERIMENT 5: CROSS-DOMAIN ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("How do physics concepts relate to consciousness concepts?\n");

    // Physics to consciousness similarity
    println!("Physics → Consciousness Similarity Matrix:\n");
    println!(
        "{:<12} {:>10} {:>10} {:>10} {:>10}",
        "Physics", "Qualia", "Unity", "Computa", "Info"
    );
    println!("{}", "-".repeat(52));

    let physics_concepts: Vec<(&str, &ContinuousHV)> = vec![
        ("Electron", &model.electron),
        ("Proton", &hadrons.proton),
        ("Carbon", &c.vector),
        ("Water", &water.vector),
    ];

    for (name, vec) in &physics_concepts {
        println!(
            "{:<12} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            name,
            vec.similarity(&bridge.qualia),
            vec.similarity(&bridge.unity),
            vec.similarity(&bridge.computation),
            vec.similarity(&bridge.information)
        );
    }

    // Emergence concept similarities
    println!("\nBridge Concepts:");
    println!(
        "  Embodiment ↔ Qualia:      {:.4}",
        bridge.embodiment.similarity(&bridge.qualia)
    );
    println!(
        "  Embodiment ↔ Computation: {:.4}",
        bridge.embodiment.similarity(&bridge.computation)
    );
    println!(
        "  Emergence ↔ Unity:        {:.4}",
        bridge.emergence.similarity(&bridge.unity)
    );
    println!(
        "  Integration ↔ Unity:      {:.4}",
        bridge.integration.similarity(&bridge.unity)
    );

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Hypothesis Results:\n");
    println!(
        "  H1 (Compositional Emergence):  {:>10}",
        if emergence {
            "SUPPORTED"
        } else {
            "NOT SUPPORTED"
        }
    );
    println!("      Emergence gradient: {:+.4}/level", gradient);

    println!(
        "\n  H2 (Binding > Bundling):       {:>10}",
        if avg_advantage > 0.0 {
            "SUPPORTED"
        } else {
            "NOT SUPPORTED"
        }
    );
    println!("      Mean binding advantage: {:+.4}", avg_advantage);

    println!(
        "\n  H3 (Embodiment Effect):        {:>10}",
        if avg_delta > 0.0 {
            "SUPPORTED"
        } else {
            "NOT SUPPORTED"
        }
    );
    println!("      Mean embodiment delta: {:+.4}", avg_delta);

    println!("\n  Phenomenal/Functional Discrimination: {:.4}", fisher);

    println!("\nImplications:");
    if emergence && avg_advantage > 0.0 && avg_delta > 0.0 {
        println!("  ✓ Evidence consistent with compositional theory of consciousness");
        println!("  ✓ Binding operations may be computationally relevant to phenomenal unity");
        println!("  ✓ Physical-phenomenal bridging appears feasible in HDC space");
    } else {
        println!("  - Mixed results require further investigation");
        println!("  - Consider alternative binding operations or phenomenal encodings");
    }

    println!("\n================================================================");
    println!("   CONSCIOUSNESS EMERGENCE EXPERIMENT COMPLETE");
    println!("================================================================\n");
}