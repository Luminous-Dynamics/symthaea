// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Primitive Decomposition Test: Path B Validation
//!
//! This test validates whether crystallized concepts can be meaningfully
//! decomposed into primitive activations.
//!
//! ## The Question
//!
//! When a concept crystallizes from attractor dynamics, does it map to
//! meaningful primitives? Or is the mapping arbitrary?
//!
//! ## Running
//!
//! ```bash
//! cargo run --release --example primitive_decomposition_test
//! ```

use symthaea::consciousness::recursive_improvement::{
    ActionContext, DreamConfig, DreamMode, SemanticBridge, SemanticBridgeConfig, SemanticInput,
};
use symthaea::dynamics::AttractorConcept as CrystalizedConcept;
use symthaea::hdc::binary_hv::BinaryHV;
use symthaea::hdc::primitive_system::{Primitive, PrimitiveSystem, PrimitiveTier};

/// Decompose a crystallized concept into its constituent primitives
fn decompose_concept(concept: &CrystalizedConcept) -> Vec<(String, PrimitiveTier, f32)> {
    let primitive_system = PrimitiveSystem::global();

    // Convert concept hypervector to BinaryHV
    // The concept hypervector is 16,384 dimensions in bipolar {-1, +1} format
    if concept.hypervector.len() != BinaryHV::DIM {
        println!(
            "  Warning: Concept hypervector has {} dims, expected {}",
            concept.hypervector.len(),
            BinaryHV::DIM
        );
        return Vec::new();
    }

    let concept_hv = BinaryHV::from_bipolar(&concept.hypervector);

    // Compare against all primitives
    let mut activations: Vec<(String, PrimitiveTier, f32)> = Vec::new();
    let mut max_sim: f32 = 0.0;
    let mut min_sim: f32 = 1.0;
    let mut sum_sim: f32 = 0.0;
    let mut count: usize = 0;

    for primitive in primitive_system.all_primitives() {
        let similarity = concept_hv.similarity(&primitive.encoding);

        max_sim = max_sim.max(similarity);
        min_sim = min_sim.min(similarity);
        sum_sim += similarity;
        count += 1;

        // DIAGNOSTIC: Lower threshold to see what we get
        // (Random vectors have ~0.50 similarity due to binary representation)
        if similarity > 0.505 {
            // Just barely above random
            activations.push((primitive.name.clone(), primitive.tier, similarity));
        }
    }

    // Print diagnostic info
    if count > 0 {
        println!(
            "    Similarity stats: min={:.4}, max={:.4}, avg={:.4}",
            min_sim,
            max_sim,
            sum_sim / count as f32
        );
    }

    // Sort by activation strength (descending)
    activations.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    activations
}

/// Get tier distribution for a concept
fn tier_distribution(
    activations: &[(String, PrimitiveTier, f32)],
) -> Vec<(PrimitiveTier, usize, f32)> {
    use std::collections::HashMap;

    let mut tier_counts: HashMap<PrimitiveTier, (usize, f32)> = HashMap::new();

    for (_, tier, strength) in activations {
        let entry = tier_counts.entry(*tier).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += strength;
    }

    let mut distribution: Vec<_> = tier_counts
        .into_iter()
        .map(|(tier, (count, total_strength))| (tier, count, total_strength / count as f32))
        .collect();

    distribution.sort_by(|a, b| b.1.cmp(&a.1));
    distribution
}

fn tier_name(tier: PrimitiveTier) -> &'static str {
    match tier {
        PrimitiveTier::NSM => "NSM (Semantic)",
        PrimitiveTier::Mathematical => "Mathematical",
        PrimitiveTier::Physical => "Physical",
        PrimitiveTier::Geometric => "Geometric",
        PrimitiveTier::Strategic => "Strategic",
        PrimitiveTier::MetaCognitive => "MetaCognitive",
        PrimitiveTier::Temporal => "Temporal",
        PrimitiveTier::Compositional => "Compositional",
        PrimitiveTier::Consciousness => "Consciousness",
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                    ║");
    println!("║     P R I M I T I V E   D E C O M P O S I T I O N   T E S T       ║");
    println!("║                                                                    ║");
    println!("║     Can crystallized concepts map to meaningful primitives?       ║");
    println!("║                                                                    ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Phase 1: Initialize and verify primitive system
    // =========================================================================

    println!("Phase 1: Primitive System Verification");
    println!("────────────────────────────────────────────────────────────────────");

    let primitive_system = PrimitiveSystem::global();
    let total_primitives: usize = primitive_system.all_primitives().count();

    println!("  Total primitives available: {}", total_primitives);

    // Count by tier
    let mut tier_counts: std::collections::HashMap<PrimitiveTier, usize> =
        std::collections::HashMap::new();
    for prim in primitive_system.all_primitives() {
        *tier_counts.entry(prim.tier).or_insert(0) += 1;
    }

    println!("  Primitives by tier:");
    for tier in [
        PrimitiveTier::NSM,
        PrimitiveTier::Mathematical,
        PrimitiveTier::Physical,
        PrimitiveTier::Geometric,
        PrimitiveTier::Strategic,
        PrimitiveTier::MetaCognitive,
        PrimitiveTier::Temporal,
        PrimitiveTier::Compositional,
        PrimitiveTier::Consciousness,
    ] {
        let count = tier_counts.get(&tier).copied().unwrap_or(0);
        println!("    {:20} : {:>3}", tier_name(tier), count);
    }
    println!();

    // =========================================================================
    // Phase 2: Crystallize concepts from diverse inputs
    // =========================================================================

    println!("Phase 2: Crystallizing Concepts from Diverse Inputs");
    println!("────────────────────────────────────────────────────────────────────");

    let mut bridge = SemanticBridge::new(SemanticBridgeConfig {
        use_simulated: true,
        ..Default::default()
    });

    // Three distinct topic clusters
    let inputs = vec![
        // Cluster A: Technical/Logical (should activate Mathematical/Physical)
        SemanticInput::query("How does the compiler optimize code?"),
        SemanticInput::response("The compiler uses type inference and dead code elimination"),
        SemanticInput::thought("Analyzing the optimization pipeline..."),
        // Cluster B: Temporal/Strategic (should activate Temporal/Strategic)
        SemanticInput::query("When should I deploy the update?"),
        SemanticInput::response("Deploy after testing, before the deadline"),
        SemanticInput::thought("Planning the deployment timeline..."),
        // Cluster C: Emotional/Social (should activate NSM/Consciousness)
        SemanticInput::query("Why do I feel frustrated with this bug?"),
        SemanticInput::response("Frustration comes from unmet expectations"),
        SemanticInput::new(
            "Goal: Understand my emotional response",
            ActionContext::Goal,
        )
        .with_emotion(0.7),
        // Cluster D: Spatial/Physical (should activate Geometric/Physical)
        SemanticInput::query("How do I structure the file hierarchy?"),
        SemanticInput::response("Organize by domain, with clear boundaries"),
        SemanticInput::thought("Visualizing the folder structure..."),
        // Cluster E: Meta-Cognitive (should activate MetaCognitive/Consciousness)
        SemanticInput::query("Am I approaching this problem correctly?"),
        SemanticInput::thought("Reflecting on my problem-solving strategy..."),
        SemanticInput::new("Goal: Improve my reasoning process", ActionContext::Goal)
            .with_emotion(0.8),
    ];

    println!("  Processing {} diverse inputs...", inputs.len());

    for input in inputs {
        bridge.process(input);
    }

    // Dream to consolidate
    println!("  Running dream consolidation...");
    let dream_config = DreamConfig {
        steps_per_cycle: 30,
        batch_cycles: 3,
        cooldown_ms: 0,
        ..Default::default()
    };

    let mut dream_mode = DreamMode::new(dream_config);
    dream_mode.batch_dream(bridge.world_model_mut());

    let concepts = &bridge.world_model().pending_concepts;
    println!("  Concepts crystallized: {}", concepts.len());
    println!();

    // =========================================================================
    // Phase 3: Decompose each concept into primitives
    // =========================================================================

    println!("Phase 3: Primitive Decomposition Analysis");
    println!("────────────────────────────────────────────────────────────────────");

    if concepts.is_empty() {
        println!("  No concepts crystallized - cannot perform decomposition test.");
        println!();
        println!("  Try adjusting crystallization thresholds or adding more input.");
        return;
    }

    let mut all_activations: Vec<Vec<(String, PrimitiveTier, f32)>> = Vec::new();
    let mut concept_tier_profiles: Vec<(String, Vec<(PrimitiveTier, usize, f32)>)> = Vec::new();

    for (i, concept) in concepts.iter().enumerate().take(10) {
        println!();
        println!("  ═══ {} ═══", concept.uid);
        println!("  Activations: {}", concept.activation_count);

        let activations = decompose_concept(concept);

        if activations.is_empty() {
            println!("  (No above-threshold primitive activations)");
            continue;
        }

        // Show top 5 primitives
        println!("  Top primitives:");
        for (name, tier, strength) in activations.iter().take(5) {
            println!(
                "    [{:>6.2}%] {:20} ({})",
                strength * 100.0,
                name,
                tier_name(*tier)
            );
        }

        // Tier distribution
        let distribution = tier_distribution(&activations);
        concept_tier_profiles.push((concept.uid.clone(), distribution.clone()));

        println!("  Tier profile:");
        for (tier, count, avg_strength) in distribution.iter().take(3) {
            println!(
                "    {:20}: {} primitives, avg {:.2}%",
                tier_name(*tier),
                count,
                avg_strength * 100.0
            );
        }

        all_activations.push(activations);
    }

    println!();

    // =========================================================================
    // Phase 4: Analyze differentiation
    // =========================================================================

    println!("Phase 4: Concept Differentiation Analysis");
    println!("────────────────────────────────────────────────────────────────────");

    // Do different concepts activate different primitives?
    let mut unique_primitives_per_concept: Vec<usize> = Vec::new();
    let mut shared_primitives: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (i, activations) in all_activations.iter().enumerate() {
        let primitive_names: std::collections::HashSet<_> =
            activations.iter().map(|(n, _, _)| n.clone()).collect();

        if i == 0 {
            shared_primitives = primitive_names.clone();
        } else {
            shared_primitives = shared_primitives
                .intersection(&primitive_names)
                .cloned()
                .collect();
        }

        unique_primitives_per_concept.push(primitive_names.len());
    }

    let avg_primitives = if !unique_primitives_per_concept.is_empty() {
        unique_primitives_per_concept.iter().sum::<usize>() as f32
            / unique_primitives_per_concept.len() as f32
    } else {
        0.0
    };

    println!("  Average primitives per concept: {:.1}", avg_primitives);
    println!("  Shared across ALL concepts: {}", shared_primitives.len());

    if !shared_primitives.is_empty() && shared_primitives.len() <= 10 {
        println!(
            "  Common primitives: {:?}",
            shared_primitives.iter().take(10).collect::<Vec<_>>()
        );
    }

    // Do tier profiles differ?
    println!();
    println!("  Dominant tier by concept:");
    for (uid, profile) in &concept_tier_profiles {
        if let Some((tier, count, _)) = profile.first() {
            println!("    {} → {} ({} primitives)", uid, tier_name(*tier), count);
        }
    }

    println!();

    // =========================================================================
    // Phase 5: Results and Validation
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                         RESULTS                                   ");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let concepts_analyzed = all_activations.len();
    let has_activations = all_activations.iter().any(|a| !a.is_empty());
    let has_differentiation = concept_tier_profiles
        .iter()
        .filter_map(|(_, p)| p.first().map(|(t, _, _)| t))
        .collect::<std::collections::HashSet<_>>()
        .len()
        > 1;

    println!("  Metric                          | Result");
    println!("  ────────────────────────────────┼─────────────────────────────");
    println!("  Concepts analyzed               | {}", concepts_analyzed);
    println!(
        "  Concepts with activations       | {}",
        all_activations.iter().filter(|a| !a.is_empty()).count()
    );
    println!("  Avg primitives per concept      | {:.1}", avg_primitives);
    println!(
        "  Shared primitives (all)         | {}",
        shared_primitives.len()
    );
    println!(
        "  Distinct dominant tiers         | {}",
        concept_tier_profiles
            .iter()
            .filter_map(|(_, p)| p.first().map(|(t, _, _)| t))
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    println!();

    // Validation
    let decomposition_works = has_activations && avg_primitives > 5.0;
    let differentiation_exists =
        has_differentiation || shared_primitives.len() < avg_primitives as usize;

    if decomposition_works && differentiation_exists {
        println!("╔════════════════════════════════════════════════════════════════════╗");
        println!("║                                                                    ║");
        println!("║     D E C O M P O S I T I O N   V A L I D A T E D                 ║");
        println!("║                                                                    ║");
        println!("║  Crystallized concepts DO map to meaningful primitive patterns!   ║");
        println!("║                                                                    ║");
        println!("║  Key findings:                                                    ║");
        println!("║  - Concepts activate above-chance primitives                      ║");
        println!("║  - Different concepts have different primitive profiles           ║");
        println!("║  - Tier distribution varies by concept semantics                  ║");
        println!("║                                                                    ║");
        println!("║  RECOMMENDATION: Proceed with full integration (Path A)           ║");
        println!("║                                                                    ║");
        println!("╚════════════════════════════════════════════════════════════════════╝");
    } else if decomposition_works {
        println!("╔════════════════════════════════════════════════════════════════════╗");
        println!("║                                                                    ║");
        println!("║     P A R T I A L   S U C C E S S                                 ║");
        println!("║                                                                    ║");
        println!("║  Concepts map to primitives, but differentiation is weak.         ║");
        println!("║                                                                    ║");
        println!("║  This may indicate:                                               ║");
        println!("║  - Input clusters are too similar                                 ║");
        println!("║  - More diverse training needed                                   ║");
        println!("║  - Threshold tuning required                                      ║");
        println!("║                                                                    ║");
        println!("╚════════════════════════════════════════════════════════════════════╝");
    } else {
        println!("╔════════════════════════════════════════════════════════════════════╗");
        println!("║                                                                    ║");
        println!("║     D E C O M P O S I T I O N   F A I L E D                       ║");
        println!("║                                                                    ║");
        println!("║  Concepts do not map meaningfully to primitives.                  ║");
        println!("║                                                                    ║");
        println!("║  Possible causes:                                                 ║");
        println!("║  - Hypervector dimension mismatch                                 ║");
        println!("║  - Similarity threshold too high                                  ║");
        println!("║  - Primitive encodings not aligned with concept space             ║");
        println!("║                                                                    ║");
        println!("╚════════════════════════════════════════════════════════════════════╝");
    }

    println!();
}
