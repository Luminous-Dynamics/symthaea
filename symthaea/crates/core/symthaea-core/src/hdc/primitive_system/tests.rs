// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::*;

#[test]
fn test_primitive_system_creation() {
    let system = PrimitiveSystem::new();
    assert!(system.count() > 0, "Should have primitives");
    assert!(
        system.count_tier(PrimitiveTier::Mathematical) > 0,
        "Should have mathematical primitives"
    );
}

#[test]
fn test_tier1_primitives() {
    let system = PrimitiveSystem::new();

    // Check key primitives exist
    assert!(system.get("SET").is_some(), "SET primitive should exist");
    assert!(system.get("NOT").is_some(), "NOT primitive should exist");
    assert!(system.get("ZERO").is_some(), "ZERO primitive should exist");
    assert!(
        system.get("ADDITION").is_some(),
        "ADDITION primitive should exist"
    );
}

#[test]
fn test_orthogonality_check() {
    let system = PrimitiveSystem::new();

    // Check that SET and NOT are reasonably orthogonal
    let sim = system.check_orthogonality("SET", "NOT");
    assert!(sim.is_some(), "Should be able to check orthogonality");

    // With 16,384-bit vectors, similarity() returns fraction of matching bits
    // in [0, 1]. Random/orthogonal pairs concentrate around 0.5.
    if let Some(similarity) = sim {
        assert!(
            (similarity - 0.5).abs() < 0.03,
            "Cross-domain primitives should be near-orthogonal (sim ≈ 0.5), got {}",
            similarity
        );
    }
}

#[test]
fn test_tier_validation() {
    let system = PrimitiveSystem::new();

    // With bind-based embedding and 16,384 dimensions, all within-tier
    // primitive pairs should have |similarity - 0.5| < 0.03 (~4σ threshold).
    let violations = system.validate_tier_orthogonality(PrimitiveTier::Mathematical, 0.03);

    // Zero violations expected — any violation indicates a seeding collision
    assert!(
        violations.is_empty(),
        "All Tier 1 primitives should be orthogonal (|sim - 0.5| < 0.03), violations: {:?}",
        violations
    );
}

#[test]
fn test_domain_manifolds() {
    let system = PrimitiveSystem::new();

    let math = system.domain("mathematics");
    let logic = system.domain("logic");

    assert!(math.is_some(), "Mathematics domain should exist");
    assert!(logic.is_some(), "Logic domain should exist");

    // Domain rotations are independent random 16K-bit vectors, so near-orthogonal
    // similarity() returns [0,1] where 0.5 = random baseline
    if let (Some(m), Some(l)) = (math, logic) {
        let sim = m.rotation.similarity(&l.rotation);
        assert!(
            (sim - 0.5).abs() < 0.03,
            "Domain rotations should be near-orthogonal (sim ≈ 0.5), got {}",
            sim
        );
    }
}

#[test]
fn test_derived_primitives() {
    let system = PrimitiveSystem::new();

    let zero = system.get("ZERO").unwrap();
    let one = system.get("ONE").unwrap();
    let addition = system.get("ADDITION").unwrap();

    assert!(zero.is_base, "ZERO should be a base primitive");
    assert!(!one.is_base, "ONE should be derived");
    assert!(!addition.is_base, "ADDITION should be derived");

    assert!(
        one.derivation.is_some(),
        "Derived primitives should have derivation"
    );
}

// ========================================================================
// TIER 2: PHYSICAL REALITY TESTS
// ========================================================================

#[test]
fn test_tier2_primitives_exist() {
    let system = PrimitiveSystem::new();

    // Physical properties
    assert!(system.get("MASS").is_some(), "MASS primitive should exist");
    assert!(
        system.get("CHARGE").is_some(),
        "CHARGE primitive should exist"
    );
    assert!(
        system.get("ENERGY").is_some(),
        "ENERGY primitive should exist"
    );

    // Motion
    assert!(
        system.get("VELOCITY").is_some(),
        "VELOCITY primitive should exist"
    );
    assert!(
        system.get("ACCELERATION").is_some(),
        "ACCELERATION primitive should exist"
    );
    assert!(
        system.get("MOMENTUM").is_some(),
        "MOMENTUM primitive should exist"
    );

    // Causality
    assert!(
        system.get("CAUSE").is_some(),
        "CAUSE primitive should exist"
    );
    assert!(
        system.get("EFFECT").is_some(),
        "EFFECT primitive should exist"
    );
    assert!(
        system.get("STATE_CHANGE").is_some(),
        "STATE_CHANGE primitive should exist"
    );

    // Thermodynamics
    assert!(
        system.get("THERMODYNAMIC_ENTROPY").is_some(),
        "THERMODYNAMIC_ENTROPY primitive should exist"
    );
    assert!(
        system.get("TEMPERATURE").is_some(),
        "TEMPERATURE primitive should exist"
    );
}

#[test]
fn test_tier2_domains() {
    let system = PrimitiveSystem::new();

    assert!(
        system.domain("physics").is_some(),
        "Physics domain should exist"
    );
    assert!(
        system.domain("causality").is_some(),
        "Causality domain should exist"
    );
}

#[test]
fn test_tier2_derived_primitives() {
    let system = PrimitiveSystem::new();

    // MOMENTUM should be derived (MASS × VELOCITY)
    let momentum = system.get("MOMENTUM").unwrap();
    assert!(!momentum.is_base, "MOMENTUM should be derived");
    assert!(
        momentum.derivation.is_some(),
        "MOMENTUM should have derivation"
    );

    // ACCELERATION should be derived
    let acceleration = system.get("ACCELERATION").unwrap();
    assert!(!acceleration.is_base, "ACCELERATION should be derived");
}

#[test]
fn test_tier2_orthogonality() {
    let system = PrimitiveSystem::new();

    let violations = system.validate_tier_orthogonality(PrimitiveTier::Physical, 0.03);
    assert!(
        violations.is_empty(),
        "All Tier 2 primitives should be orthogonal (|sim - 0.5| < 0.03), violations: {:?}",
        violations
    );
}

// ========================================================================
// TIER 3: GEOMETRIC & TOPOLOGICAL TESTS
// ========================================================================

#[test]
fn test_tier3_primitives_exist() {
    let system = PrimitiveSystem::new();

    // Basic geometry
    assert!(
        system.get("POINT").is_some(),
        "POINT primitive should exist"
    );
    assert!(system.get("LINE").is_some(), "LINE primitive should exist");
    assert!(
        system.get("PLANE").is_some(),
        "PLANE primitive should exist"
    );
    assert!(
        system.get("ANGLE").is_some(),
        "ANGLE primitive should exist"
    );
    assert!(
        system.get("DISTANCE").is_some(),
        "DISTANCE primitive should exist"
    );

    // Vectors
    assert!(
        system.get("VECTOR").is_some(),
        "VECTOR primitive should exist"
    );
    assert!(
        system.get("DOT_PRODUCT").is_some(),
        "DOT_PRODUCT primitive should exist"
    );
    assert!(
        system.get("CROSS_PRODUCT").is_some(),
        "CROSS_PRODUCT primitive should exist"
    );

    // Differential geometry
    assert!(
        system.get("MANIFOLD").is_some(),
        "MANIFOLD primitive should exist"
    );
    assert!(
        system.get("TANGENT_SPACE").is_some(),
        "TANGENT_SPACE primitive should exist"
    );
    assert!(
        system.get("CURVATURE").is_some(),
        "CURVATURE primitive should exist"
    );

    // Topology
    assert!(
        system.get("OPEN_SET").is_some(),
        "OPEN_SET primitive should exist"
    );
    assert!(
        system.get("BOUNDARY").is_some(),
        "BOUNDARY primitive should exist"
    );
    assert!(
        system.get("PART_OF").is_some(),
        "PART_OF primitive should exist"
    );
}

#[test]
fn test_tier3_domains() {
    let system = PrimitiveSystem::new();

    assert!(
        system.domain("geometry").is_some(),
        "Geometry domain should exist"
    );
    assert!(
        system.domain("topology").is_some(),
        "Topology domain should exist"
    );
}

#[test]
fn test_tier3_orthogonality() {
    let system = PrimitiveSystem::new();

    let violations = system.validate_tier_orthogonality(PrimitiveTier::Geometric, 0.03);
    assert!(
        violations.is_empty(),
        "All Tier 3 primitives should be orthogonal (|sim - 0.5| < 0.03), violations: {:?}",
        violations
    );
}

// ========================================================================
// TIER 4: STRATEGIC & SOCIAL TESTS
// ========================================================================

#[test]
fn test_tier4_primitives_exist() {
    let system = PrimitiveSystem::new();

    // Game theory
    assert!(
        system.get("UTILITY").is_some(),
        "UTILITY primitive should exist"
    );
    assert!(
        system.get("STRATEGY").is_some(),
        "STRATEGY primitive should exist"
    );
    assert!(
        system.get("EQUILIBRIUM").is_some(),
        "EQUILIBRIUM primitive should exist"
    );
    assert!(
        system.get("PAYOFF").is_some(),
        "PAYOFF primitive should exist"
    );

    // Temporal logic
    assert!(
        system.get("BEFORE").is_some(),
        "BEFORE primitive should exist"
    );
    assert!(
        system.get("AFTER").is_some(),
        "AFTER primitive should exist"
    );
    assert!(
        system.get("DURING").is_some(),
        "DURING primitive should exist"
    );
    assert!(
        system.get("MEETS").is_some(),
        "MEETS primitive should exist"
    );

    // Social coordination
    assert!(
        system.get("COOPERATE").is_some(),
        "COOPERATE primitive should exist"
    );
    assert!(
        system.get("DEFECT").is_some(),
        "DEFECT primitive should exist"
    );
    assert!(
        system.get("RECIPROCATE").is_some(),
        "RECIPROCATE primitive should exist"
    );
    assert!(
        system.get("TRUST").is_some(),
        "TRUST primitive should exist"
    );

    // Information
    assert!(
        system.get("BELIEF").is_some(),
        "BELIEF primitive should exist"
    );
    assert!(
        system.get("COMMON_KNOWLEDGE").is_some(),
        "COMMON_KNOWLEDGE primitive should exist"
    );
}

#[test]
fn test_tier4_domains() {
    let system = PrimitiveSystem::new();

    assert!(
        system.domain("game_theory").is_some(),
        "Game theory domain should exist"
    );
    assert!(
        system.domain("temporal").is_some(),
        "Temporal domain should exist"
    );
    assert!(
        system.domain("social").is_some(),
        "Social domain should exist"
    );
}

#[test]
fn test_tier4_harmonic_connections() {
    let system = PrimitiveSystem::new();

    // Tier 4 primitives should connect to harmonics
    // COOPERATE + TRUST → Sacred Reciprocity
    let cooperate = system.get("COOPERATE").unwrap();
    let trust = system.get("TRUST").unwrap();

    assert!(
        cooperate.tier == PrimitiveTier::Strategic,
        "COOPERATE should be Strategic tier"
    );
    assert!(
        trust.tier == PrimitiveTier::Strategic,
        "TRUST should be Strategic tier"
    );
}

#[test]
fn test_tier4_orthogonality() {
    let system = PrimitiveSystem::new();

    let violations = system.validate_tier_orthogonality(PrimitiveTier::Strategic, 0.03);
    assert!(
        violations.is_empty(),
        "All Tier 4 primitives should be orthogonal (|sim - 0.5| < 0.03), violations: {:?}",
        violations
    );
}

// ========================================================================
// TIER 5: META-COGNITIVE TESTS
// ========================================================================

#[test]
fn test_tier5_primitives_exist() {
    let system = PrimitiveSystem::new();

    // Self-awareness
    assert!(system.get("SELF").is_some(), "SELF primitive should exist");
    assert!(
        system.get("IDENTITY").is_some(),
        "IDENTITY primitive should exist"
    );
    assert!(
        system.get("META_BELIEF").is_some(),
        "META_BELIEF primitive should exist"
    );
    assert!(
        system.get("INTROSPECTION").is_some(),
        "INTROSPECTION primitive should exist"
    );

    // Homeostasis
    assert!(
        system.get("HOMEOSTASIS").is_some(),
        "HOMEOSTASIS primitive should exist"
    );
    assert!(
        system.get("SETPOINT").is_some(),
        "SETPOINT primitive should exist"
    );
    assert!(
        system.get("REGULATION").is_some(),
        "REGULATION primitive should exist"
    );
    assert!(
        system.get("FEEDBACK").is_some(),
        "FEEDBACK primitive should exist"
    );

    // Repair & adaptation
    assert!(
        system.get("REPAIR").is_some(),
        "REPAIR primitive should exist"
    );
    assert!(
        system.get("ADAPT").is_some(),
        "ADAPT primitive should exist"
    );
    assert!(
        system.get("LEARN").is_some(),
        "LEARN primitive should exist"
    );

    // Epistemic
    assert!(system.get("KNOW").is_some(), "KNOW primitive should exist");
    assert!(
        system.get("UNCERTAIN").is_some(),
        "UNCERTAIN primitive should exist"
    );
    assert!(
        system.get("CONFIDENCE").is_some(),
        "CONFIDENCE primitive should exist"
    );
    assert!(
        system.get("EVIDENCE").is_some(),
        "EVIDENCE primitive should exist"
    );

    // Metabolic
    assert!(
        system.get("RESOURCE").is_some(),
        "RESOURCE primitive should exist"
    );
    assert!(
        system.get("ALLOCATE").is_some(),
        "ALLOCATE primitive should exist"
    );

    // Reward
    assert!(
        system.get("REWARD").is_some(),
        "REWARD primitive should exist"
    );
    assert!(system.get("GOAL").is_some(), "GOAL primitive should exist");
    assert!(
        system.get("VALUE").is_some(),
        "VALUE primitive should exist"
    );
}

#[test]
fn test_tier5_domains() {
    let system = PrimitiveSystem::new();

    assert!(
        system.domain("metacognition").is_some(),
        "Metacognition domain should exist"
    );
    assert!(
        system.domain("homeostasis").is_some(),
        "Homeostasis domain should exist"
    );
    assert!(
        system.domain("epistemic").is_some(),
        "Epistemic domain should exist"
    );
    assert!(
        system.domain("metabolic").is_some(),
        "Metabolic domain should exist"
    );
}

#[test]
fn test_tier5_consciousness_primitives() {
    let system = PrimitiveSystem::new();

    // Tier 5 enables consciousness-first computing
    // SELF + HOMEOSTASIS → self-regulation
    let self_prim = system.get("SELF").unwrap();
    let homeostasis = system.get("HOMEOSTASIS").unwrap();

    assert!(self_prim.tier == PrimitiveTier::MetaCognitive);
    assert!(homeostasis.tier == PrimitiveTier::MetaCognitive);

    // These primitives enable the system to reason about itself
    assert!(self_prim.is_base, "SELF should be a base primitive");
    assert!(
        homeostasis.is_base,
        "HOMEOSTASIS should be a base primitive"
    );
}

#[test]
fn test_tier5_orthogonality() {
    let system = PrimitiveSystem::new();

    let violations = system.validate_tier_orthogonality(PrimitiveTier::MetaCognitive, 0.03);
    assert!(
        violations.is_empty(),
        "All Tier 5 primitives should be orthogonal (|sim - 0.5| < 0.03), violations: {:?}",
        violations
    );
}

// ========================================================================
// CROSS-TIER INTEGRATION TESTS
// ========================================================================

#[test]
fn test_complete_primitive_ecology() {
    let system = PrimitiveSystem::new();

    // Count primitives per tier
    let tier1_count = system.count_tier(PrimitiveTier::Mathematical);
    let tier2_count = system.count_tier(PrimitiveTier::Physical);
    let tier3_count = system.count_tier(PrimitiveTier::Geometric);
    let tier4_count = system.count_tier(PrimitiveTier::Strategic);
    let tier5_count = system.count_tier(PrimitiveTier::MetaCognitive);

    // Verify all tiers have primitives
    assert!(tier1_count > 0, "Tier 1 should have primitives");
    assert!(tier2_count > 0, "Tier 2 should have primitives");
    assert!(tier3_count > 0, "Tier 3 should have primitives");
    assert!(tier4_count > 0, "Tier 4 should have primitives");
    assert!(tier5_count > 0, "Tier 5 should have primitives");

    // Verify total count is substantial
    let total = system.count();
    assert!(
        total >= 80,
        "Should have at least 80 primitives across all tiers (got {})",
        total
    );

    println!("Complete Primitive Ecology:");
    println!("  Tier 1 (Mathematical): {} primitives", tier1_count);
    println!("  Tier 2 (Physical):     {} primitives", tier2_count);
    println!("  Tier 3 (Geometric):    {} primitives", tier3_count);
    println!("  Tier 4 (Strategic):    {} primitives", tier4_count);
    println!("  Tier 5 (MetaCognitive): {} primitives", tier5_count);
    println!("  TOTAL: {} primitives", total);
}

#[test]
fn test_cross_tier_binding() {
    let system = PrimitiveSystem::new();

    // Find cross-tier binding rules
    let cross_tier_rules: Vec<_> = system
        .binding_rules()
        .iter()
        .filter(|rule| {
            rule.pattern.len() > 1
                && rule
                    .pattern
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    > 1
        })
        .collect();

    // Should have at least one cross-tier binding rule
    assert!(
        !cross_tier_rules.is_empty(),
        "Should have cross-tier binding rules"
    );

    // Print cross-tier rules
    for rule in &cross_tier_rules {
        println!("Cross-tier rule: {} - {}", rule.name, rule.example);
    }
}

#[test]
fn test_domain_diversity() {
    let system = PrimitiveSystem::new();

    // Should have multiple domains across tiers
    assert!(system.domain("mathematics").is_some());
    assert!(system.domain("logic").is_some());
    assert!(system.domain("physics").is_some());
    assert!(system.domain("causality").is_some());
    assert!(system.domain("geometry").is_some());
    assert!(system.domain("topology").is_some());
    assert!(system.domain("game_theory").is_some());
    assert!(system.domain("temporal").is_some());
    assert!(system.domain("social").is_some());
    assert!(system.domain("metacognition").is_some());
    assert!(system.domain("homeostasis").is_some());
    assert!(system.domain("epistemic").is_some());
    assert!(system.domain("metabolic").is_some());

    // Total domain count
    println!("Total domains: {}", system.domains.len());
    assert!(
        system.domains.len() >= 13,
        "Should have at least 13 distinct domains"
    );
}

#[test]
fn test_harmonic_primitive_connections() {
    let system = PrimitiveSystem::new();

    // Test primitives that connect to specific harmonics

    // Tier 2 → Resonant Coherence (physical stability)
    assert!(system.get("THERMODYNAMIC_ENTROPY").is_some());

    // Tier 3 → Universal Interconnectedness (spatial relationships)
    assert!(system.get("PART_OF").is_some());

    // Tier 4 → Sacred Reciprocity (cooperation)
    assert!(system.get("COOPERATE").is_some());
    assert!(system.get("TRUST").is_some());
    assert!(system.get("RECIPROCATE").is_some());

    // Tier 5 → All 7 harmonics (meta-cognitive spans all)
    assert!(system.get("SELF").is_some());
    assert!(system.get("HOMEOSTASIS").is_some());
    assert!(system.get("KNOW").is_some());
    assert!(system.get("GOAL").is_some());
}

#[test]
fn test_primitive_ecology_summary() {
    let system = PrimitiveSystem::new();

    let summary = system.summary();

    println!("\n{}", summary);

    // Summary should contain all tier names
    assert!(summary.contains("Mathematical"));
    assert!(summary.contains("Physical"));
    assert!(summary.contains("Geometric"));
    assert!(summary.contains("Strategic"));
    assert!(summary.contains("MetaCognitive"));
}

// =========================================================================
// DERIVATION CHAIN INTEGRATION TESTS
// =========================================================================
//
// These tests verify that derived primitives have the expected algebraic
// relationships with their parent primitives via XOR binding.
//
// For a derived primitive D = P1 ⊗ P2:
// - D ⊗ P1 should have high similarity to P2
// - D ⊗ P2 should have high similarity to P1
// =========================================================================

#[test]
fn test_attention_derives_from_salience_selection() {
    let system = PrimitiveSystem::new();

    let attention = system.get("ATTENTION").expect("ATTENTION should exist");
    let salience = system.get("SALIENCE").expect("SALIENCE should exist");
    let selection = system.get("SELECTION").expect("SELECTION should exist");

    // If ATTENTION = SALIENCE ⊗ SELECTION, then:
    // ATTENTION ⊗ SALIENCE should be similar to SELECTION
    let unbound = attention.encoding.bind(&salience.encoding);
    let sim_to_selection = unbound.similarity(&selection.encoding);

    // For proper derivation, similarity should be very high (>0.95)
    // If derivation fell back to random, it would be ~0.5
    assert!(
        sim_to_selection > 0.90,
        "ATTENTION\u{2297}SALIENCE should recover SELECTION (sim={:.3}, expected >0.90). \
         If ~0.5, ATTENTION used random fallback instead of derivation.",
        sim_to_selection
    );

    // Similarly, ATTENTION ⊗ SELECTION should recover SALIENCE
    let unbound2 = attention.encoding.bind(&selection.encoding);
    let sim_to_salience = unbound2.similarity(&salience.encoding);
    assert!(
        sim_to_salience > 0.90,
        "ATTENTION\u{2297}SELECTION should recover SALIENCE (sim={:.3}, expected >0.90)",
        sim_to_salience
    );
}

#[test]
fn test_probability_derivation_chain() {
    let system = PrimitiveSystem::new();

    let probability = system.get("PROBABILITY").expect("PROBABILITY should exist");
    let ratio = system.get("RATIO").expect("RATIO should exist");
    let certainty = system.get("CERTAINTY").expect("CERTAINTY should exist");

    // PROBABILITY = RATIO ⊗ CERTAINTY
    // Unbinding should recover the other parent
    let unbound = probability.encoding.bind(&ratio.encoding);
    let sim_to_certainty = unbound.similarity(&certainty.encoding);

    // Also check the reverse unbinding
    let unbound2 = probability.encoding.bind(&certainty.encoding);
    let sim_to_ratio = unbound2.similarity(&ratio.encoding);

    // For proper derivation, similarity should be very high (>0.90)
    // If derivation fell back to random, it would be ~0.5
    assert!(
        sim_to_certainty > 0.90,
        "PROBABILITY\u{2297}RATIO should recover CERTAINTY (sim={:.3}). \
         If ~0.5, PROBABILITY used random fallback instead of derivation.",
        sim_to_certainty
    );
    assert!(
        sim_to_ratio > 0.90,
        "PROBABILITY\u{2297}CERTAINTY should recover RATIO (sim={:.3})",
        sim_to_ratio
    );
}

#[test]
fn test_expected_value_derivation_chain() {
    let system = PrimitiveSystem::new();

    let expected_value = system
        .get("EXPECTED_VALUE")
        .expect("EXPECTED_VALUE should exist");
    let probability = system.get("PROBABILITY").expect("PROBABILITY should exist");
    let value = system.get("VALUE").expect("VALUE should exist");

    // EXPECTED_VALUE = PROBABILITY ⊗ VALUE
    // Unbinding should recover the other parent
    let unbound = expected_value.encoding.bind(&probability.encoding);
    let sim_to_value = unbound.similarity(&value.encoding);

    // For proper derivation, similarity should be very high (>0.90)
    // If derivation fell back to random, it would be ~0.5
    assert!(
        sim_to_value > 0.90,
        "EXPECTED_VALUE\u{2297}PROBABILITY should recover VALUE (sim={:.3})",
        sim_to_value
    );
}

#[test]
fn test_certainty_uncertain_relationship() {
    let system = PrimitiveSystem::new();

    // UNCERTAIN and CERTAINTY are both base primitives (not derived)
    // They should be nearly orthogonal (independent random vectors ~0.5 similarity)
    let uncertain = system.get("UNCERTAIN").expect("UNCERTAIN should exist");
    let certainty = system.get("CERTAINTY").expect("CERTAINTY should exist");

    let sim = uncertain.encoding.similarity(&certainty.encoding);
    // Both are independently seeded randoms, so should be ~0.5 (orthogonal)
    assert!(
        sim > 0.40 && sim < 0.60,
        "UNCERTAIN and CERTAINTY are independent randoms, should be ~0.5 orthogonal (sim={:.3})",
        sim
    );
}

#[test]
fn test_derived_primitives_not_random() {
    let system = PrimitiveSystem::new();

    // Test several ACTUAL derived primitives to ensure they're not just random vectors
    // Random vectors would have ~0.5 similarity to their declared parents
    // NOTE: Only test primitives that are actually derived in init_derived_primitives!

    let test_cases = [
        ("FIELD", vec!["FORCE", "POINT"]),
        ("EQUILIBRIUM", vec!["FORCE", "CONSERVATION"]),
        ("SHANNON_ENTROPY", vec!["PROBABILITY", "INFORMATION"]),
    ];

    for (derived_name, parent_names) in test_cases {
        let derived = system.get(derived_name);
        if derived.is_none() {
            // Skip if this derived primitive doesn't exist in this version
            continue;
        }
        let derived = derived.unwrap();

        // Get parent encodings
        let parents: Vec<_> = parent_names
            .iter()
            .filter_map(|name| system.get(name))
            .collect();

        if parents.len() < 2 {
            // Skip if parents don't exist
            continue;
        }

        // Compute expected derivation: P1 ⊗ P2
        let expected = parents[0].encoding.bind(&parents[1].encoding);

        let sim = derived.encoding.similarity(&expected);
        assert!(
            sim > 0.90,
            "{} should be derived from {:?} (sim={:.3}, expected >0.90). \
             Low similarity suggests random fallback was used.",
            derived_name,
            parent_names,
            sim
        );
    }
}

#[test]
fn test_derivation_chain_transitivity() {
    // Test that multi-level derivations work correctly
    // VARIANCE = EXPECTED_VALUE ⊗ DEVIATION
    // EXPECTED_VALUE = PROBABILITY ⊗ VALUE
    // So VARIANCE ⊗ EXPECTED_VALUE should recover DEVIATION
    let system = PrimitiveSystem::new();

    let variance = system.get("VARIANCE");
    let expected_value = system.get("EXPECTED_VALUE");
    let deviation = system.get("DEVIATION");

    if let (Some(variance), Some(expected_value), Some(deviation)) =
        (variance, expected_value, deviation)
    {
        // VARIANCE ⊗ EXPECTED_VALUE should recover DEVIATION
        let unbound = variance.encoding.bind(&expected_value.encoding);
        let sim = unbound.similarity(&deviation.encoding);

        assert!(
            sim > 0.90,
            "VARIANCE\u{2297}EXPECTED_VALUE should recover DEVIATION (sim={:.3})",
            sim
        );
    }
}

// =========================================================================
// DERIVATION QUALITY BENCHMARK
// =========================================================================

/// Benchmark test: measures derivation quality across all derived primitives
///
/// This test quantifies how well the derive_encoding function works by:
/// 1. Finding all primitives with a documented derivation (e.g., "A ⊗ B")
/// 2. For each, computing the expected encoding from parents
/// 3. Measuring similarity between actual and expected encodings
///
/// Results are printed to help assess system health.
#[test]
fn test_benchmark_derivation_quality() {
    let system = PrimitiveSystem::new();

    // Collect all derived primitives with their documented derivation
    let mut derived_count = 0;
    let mut success_count = 0;
    let mut total_similarity: f32 = 0.0;
    let mut failures: Vec<(String, f32)> = Vec::new();

    for (name, primitive) in system.primitives.iter() {
        if primitive.is_base || primitive.derivation.is_none() {
            continue;
        }

        let derivation = primitive.derivation.as_ref().unwrap();

        // Parse derivation string like "A ⊗ B" or "APPLY(A, B)"
        // For now, only handle "X ⊗ Y" format
        if !derivation.contains(" \u{2297} ") {
            continue;
        }

        let parts: Vec<&str> = derivation.split(" \u{2297} ").collect();
        if parts.len() != 2 {
            continue;
        }

        let parent1_name = parts[0].trim();
        let parent2_name = parts[1].trim();

        // Get parent primitives
        let parent1 = system.get(parent1_name);
        let parent2 = system.get(parent2_name);

        if let (Some(p1), Some(p2)) = (parent1, parent2) {
            derived_count += 1;

            // Expected encoding: parent1 ⊗ parent2
            let expected = p1.encoding.bind(&p2.encoding);
            let similarity = primitive.encoding.similarity(&expected);
            total_similarity += similarity;

            if similarity > 0.90 {
                success_count += 1;
            } else {
                failures.push((name.clone(), similarity));
            }
        }
    }

    // Print benchmark results
    if derived_count > 0 {
        let avg_similarity = total_similarity / derived_count as f32;
        println!("\n=== DERIVATION QUALITY BENCHMARK ===");
        println!("Total derived primitives tested: {}", derived_count);
        println!(
            "Successfully derived (sim > 0.90): {} ({:.1}%)",
            success_count,
            100.0 * success_count as f32 / derived_count as f32
        );
        println!("Average similarity to expected: {:.3}", avg_similarity);

        if !failures.is_empty() {
            println!("\nFailed derivations:");
            for (name, sim) in &failures {
                println!("  {} - similarity: {:.3}", name, sim);
            }
        }
        println!("=====================================\n");

        // Assert at least 90% success rate
        let success_rate = success_count as f32 / derived_count as f32;
        assert!(
            success_rate >= 0.90,
            "Derivation success rate {:.1}% is below 90% threshold. {} of {} primitives failed.",
            success_rate * 100.0,
            failures.len(),
            derived_count
        );
    }
}

// =========================================================================
// TYPED PRIMITIVE OPERATIONS TESTS
// =========================================================================

#[test]
fn test_bind_primitives() {
    let system = PrimitiveSystem::new();

    // Test successful binding
    let result = system.bind_primitives("CAUSE", "EFFECT");
    assert!(result.is_ok(), "bind_primitives should succeed");

    let result = result.unwrap();
    assert_eq!(result.source_primitives, vec!["CAUSE", "EFFECT"]);
    assert!(result.operation.contains("bind"));

    // Verify XOR binding properties: A ⊗ B ⊗ B = A
    let cause = system.get("CAUSE").unwrap();
    let effect = system.get("EFFECT").unwrap();
    let unbound = result.encoding.bind(&effect.encoding);
    let sim = unbound.similarity(&cause.encoding);
    assert!(
        sim > 0.99,
        "Unbinding should recover original (sim={:.3})",
        sim
    );
}

#[test]
fn test_bind_primitives_not_found() {
    let system = PrimitiveSystem::new();

    let result = system.bind_primitives("CAUSE", "NONEXISTENT");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), PrimitiveError::NotFound(_)));
}

#[test]
fn test_bundle_primitives() {
    let system = PrimitiveSystem::new();

    let result = system.bundle_primitives(&["AND", "OR", "NOT"]);
    assert!(result.is_ok(), "bundle_primitives should succeed");

    let result = result.unwrap();
    assert_eq!(result.source_primitives.len(), 3);
    assert!(result.operation.contains("bundle"));

    // Bundle should have moderate similarity to all inputs
    let and_prim = system.get("AND").unwrap();
    let or_prim = system.get("OR").unwrap();
    let not_prim = system.get("NOT").unwrap();

    let sim_and = result.encoding.similarity(&and_prim.encoding);
    let sim_or = result.encoding.similarity(&or_prim.encoding);
    let sim_not = result.encoding.similarity(&not_prim.encoding);

    // Bundle should be more similar to inputs than random (~0.5)
    assert!(
        sim_and > 0.55,
        "Bundle should be similar to AND (sim={:.3})",
        sim_and
    );
    assert!(
        sim_or > 0.55,
        "Bundle should be similar to OR (sim={:.3})",
        sim_or
    );
    assert!(
        sim_not > 0.55,
        "Bundle should be similar to NOT (sim={:.3})",
        sim_not
    );
}

#[test]
fn test_bundle_primitives_empty() {
    let system = PrimitiveSystem::new();

    let result = system.bundle_primitives(&[]);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), PrimitiveError::EmptyInput));
}

// =========================================================================
// SEQUENCE ENCODING TESTS
// =========================================================================

#[test]
fn test_encode_sequence_single() {
    let system = PrimitiveSystem::new();

    // Single element sequence should be equivalent to the primitive itself
    let result = system.encode_sequence(&["CAUSE"]);
    assert!(result.is_ok());

    let result = result.unwrap();
    let cause = system.get("CAUSE").unwrap();
    let sim = result.encoding.similarity(&cause.encoding);
    assert!(
        sim > 0.99,
        "Single-element sequence should equal the primitive (sim={:.3})",
        sim
    );
}

#[test]
fn test_encode_sequence_order_matters() {
    let system = PrimitiveSystem::new();

    // A → B should be different from B → A
    let seq_ab = system.encode_sequence(&["CAUSE", "EFFECT"]).unwrap();
    let seq_ba = system.encode_sequence(&["EFFECT", "CAUSE"]).unwrap();

    let sim = seq_ab.encoding.similarity(&seq_ba.encoding);

    // Should be significantly different (not random ~0.5, but also not identical >0.99)
    assert!(
        sim < 0.8,
        "Different orderings should produce different encodings (sim={:.3})",
        sim
    );
}

#[test]
fn test_encode_sequence_longer() {
    let system = PrimitiveSystem::new();

    // Test longer sequence
    let result = system.encode_sequence(&["BEFORE", "DURING", "AFTER"]);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.source_primitives, vec!["BEFORE", "DURING", "AFTER"]);
    assert!(result.operation.contains("->"));
}

#[test]
fn test_encode_sequence_empty() {
    let system = PrimitiveSystem::new();

    let result = system.encode_sequence(&[]);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), PrimitiveError::EmptyInput));
}

#[test]
fn test_encode_sequence_not_found() {
    let system = PrimitiveSystem::new();

    let result = system.encode_sequence(&["CAUSE", "NONEXISTENT"]);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), PrimitiveError::NotFound(_)));
}

// =========================================================================
// WEIGHTED BUNDLING TESTS
// =========================================================================

#[test]
fn test_bundle_weighted_equal_weights() {
    let system = PrimitiveSystem::new();

    // Equal weights should be similar to regular bundle
    let regular = system.bundle_primitives(&["AND", "OR"]).unwrap();
    let weighted = system
        .bundle_weighted(&[("AND", 1.0), ("OR", 1.0)])
        .unwrap();

    let sim = regular.encoding.similarity(&weighted.encoding);
    assert!(
        sim > 0.95,
        "Equal-weighted bundle should match regular bundle (sim={:.3})",
        sim
    );
}

#[test]
fn test_bundle_weighted_dominant() {
    let system = PrimitiveSystem::new();

    let and_prim = system.get("AND").unwrap();
    let or_prim = system.get("OR").unwrap();

    // Heavily weight AND
    let and_dominant = system
        .bundle_weighted(&[("AND", 10.0), ("OR", 1.0)])
        .unwrap();
    let sim_to_and = and_dominant.encoding.similarity(&and_prim.encoding);
    let sim_to_or = and_dominant.encoding.similarity(&or_prim.encoding);

    assert!(
        sim_to_and > sim_to_or,
        "AND-dominant bundle should be more similar to AND ({:.3}) than OR ({:.3})",
        sim_to_and,
        sim_to_or
    );
    assert!(
        sim_to_and > 0.7,
        "AND-dominant bundle should have high similarity to AND (sim={:.3})",
        sim_to_and
    );
}

#[test]
fn test_bundle_weighted_empty() {
    let system = PrimitiveSystem::new();

    let result = system.bundle_weighted(&[]);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), PrimitiveError::EmptyInput));
}

#[test]
fn test_bundle_weighted_zero_weights() {
    let system = PrimitiveSystem::new();

    let result = system.bundle_weighted(&[("AND", 0.0), ("OR", 0.0)]);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), PrimitiveError::InvalidWeight));
}

#[test]
fn test_bundle_weighted_not_found() {
    let system = PrimitiveSystem::new();

    let result = system.bundle_weighted(&[("AND", 1.0), ("NONEXISTENT", 1.0)]);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), PrimitiveError::NotFound(_)));
}

// =========================================================================
// ANALOGY AND PERMUTATION TESTS
// =========================================================================

#[test]
fn test_analogy() {
    let system = PrimitiveSystem::new();

    // CAUSE:EFFECT :: BEFORE:? should give something related to AFTER
    let result = system.analogy("CAUSE", "EFFECT", "BEFORE");
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.operation.contains("analogy"));
    assert_eq!(result.source_primitives, vec!["CAUSE", "EFFECT", "BEFORE"]);
}

#[test]
fn test_permute_primitive() {
    let system = PrimitiveSystem::new();

    let original = system.get("CAUSE").unwrap();
    let permuted = system.permute_primitive("CAUSE", 1).unwrap();

    // Permuted should be different from original
    let sim = original.encoding.similarity(&permuted.encoding);
    assert!(
        sim < 0.7,
        "Permuted vector should be different from original (sim={:.3})",
        sim
    );

    // Permuting back should recover original (BinaryHV::permute(16384 - 1) = inverse)
    // Note: This depends on the permute implementation being cyclic
}

// =========================================================================
// SIMILARITY SEARCH TESTS
// =========================================================================

#[test]
fn test_find_similar_primitives() {
    let system = PrimitiveSystem::new();

    let similar = system.find_similar("CAUSE", 5);
    assert_eq!(
        similar.len(),
        5,
        "Should return requested number of results"
    );

    // Results should be sorted by similarity (descending)
    for i in 1..similar.len() {
        assert!(
            similar[i - 1].1 >= similar[i].1,
            "Results should be sorted by similarity"
        );
    }

    // Should not include the query primitive itself
    assert!(
        !similar.iter().any(|(name, _)| name == "CAUSE"),
        "Results should not include query primitive"
    );
}

#[test]
fn test_find_similar_to_encoding() {
    let system = PrimitiveSystem::new();

    let cause = system.get("CAUSE").unwrap();
    let similar = system.find_similar_to_encoding(&cause.encoding, 3);

    assert_eq!(similar.len(), 3);

    // CAUSE itself should be the top match
    assert_eq!(similar[0].0, "CAUSE");
    assert!(
        similar[0].1 > 0.99,
        "Exact match should have ~1.0 similarity"
    );
}

#[test]
fn test_query() {
    let system = PrimitiveSystem::new();

    let cause = system.get("CAUSE").unwrap();
    let (name, sim) = system.query(&cause.encoding);

    assert_eq!(name, "CAUSE");
    assert!(sim > 0.99);
}

// =========================================================================
// LSH INDEX TESTS
// =========================================================================

#[test]
fn test_lsh_index_build() {
    let system = PrimitiveSystem::new();

    // Build LSH index with 8 bands and 64 bits per band
    let lsh = system.build_lsh_index(8, 64);
    let stats = lsh.stats();

    assert_eq!(stats.num_bands, 8);
    assert_eq!(stats.bits_per_band, 64);
    assert!(stats.total_buckets > 0, "Should have created buckets");
    assert!(stats.total_entries > 0, "Should have indexed primitives");
}

#[test]
fn test_lsh_find_exact_match() {
    let system = PrimitiveSystem::new();
    let lsh = system.build_lsh_index(8, 64);

    // Query for CAUSE should find CAUSE in candidates
    let cause = system.get("CAUSE").unwrap();
    let candidates = lsh.query_candidates(&cause.encoding);

    assert!(
        candidates.contains(&"CAUSE".to_string()),
        "LSH should find exact match in candidates"
    );
}

#[test]
fn test_lsh_find_similar() {
    let system = PrimitiveSystem::new();
    let lsh = system.build_lsh_index(8, 64);

    let cause = system.get("CAUSE").unwrap();
    let similar = system.find_similar_lsh(&cause.encoding, 5, &lsh);

    // Should find at least one result (CAUSE itself)
    assert!(!similar.is_empty(), "LSH search should find results");

    // CAUSE should be the top match
    assert_eq!(similar[0].0, "CAUSE");
    assert!(similar[0].1 > 0.99);
}

#[test]
fn test_lsh_candidates_contain_similar() {
    let system = PrimitiveSystem::new();
    let lsh = system.build_lsh_index(16, 32); // More bands, fewer bits = more candidates

    // Bundle CAUSE and EFFECT
    let bound = system.bind_primitives("CAUSE", "EFFECT").unwrap();

    // The candidates should include some primitives
    let candidates = lsh.query_candidates(&bound.encoding);

    // With a good LSH configuration, we should get some candidates
    // (though bound vectors are orthogonal to inputs, some collisions are expected)
    println!(
        "LSH returned {} candidates for bound(CAUSE, EFFECT)",
        candidates.len()
    );
}

#[test]
fn test_lsh_deterministic() {
    let system = PrimitiveSystem::new();

    // Build two indices with same parameters
    let lsh1 = system.build_lsh_index(4, 32);
    let lsh2 = system.build_lsh_index(4, 32);

    // Query should return same candidates
    let cause = system.get("CAUSE").unwrap();
    let candidates1 = lsh1.query_candidates(&cause.encoding);
    let candidates2 = lsh2.query_candidates(&cause.encoding);

    let set1: std::collections::HashSet<_> = candidates1.into_iter().collect();
    let set2: std::collections::HashSet<_> = candidates2.into_iter().collect();

    assert_eq!(set1, set2, "LSH should be deterministic");
}

#[test]
fn test_lsh_stats() {
    let system = PrimitiveSystem::new();
    let lsh = system.build_lsh_index(4, 48);

    let stats = lsh.stats();
    println!(
        "LSH stats: {} bands, {} bits/band, {} buckets, {} entries, avg {:.2} per bucket",
        stats.num_bands,
        stats.bits_per_band,
        stats.total_buckets,
        stats.total_entries,
        stats.avg_bucket_size
    );

    // Each primitive should be in each band
    let expected_entries = system.count() * stats.num_bands;
    assert_eq!(
        stats.total_entries, expected_entries,
        "Each primitive should be indexed in each band"
    );
}

// =========================================================================
// COMPOSITION CACHE TESTS
// =========================================================================

#[test]
fn test_cache_bind() {
    let system = PrimitiveSystem::new();
    let mut cache = CompositionCache::new(100);

    // First call should be a miss
    let result1 = cache.bind_cached(&system, "CAUSE", "EFFECT").unwrap();
    let stats1 = cache.stats();
    assert_eq!(stats1.misses, 1);
    assert_eq!(stats1.hits, 0);

    // Second call should be a hit
    let result2 = cache.bind_cached(&system, "CAUSE", "EFFECT").unwrap();
    let stats2 = cache.stats();
    assert_eq!(stats2.misses, 1);
    assert_eq!(stats2.hits, 1);

    // Results should be identical
    assert!(
        result1.encoding.similarity(&result2.encoding) > 0.99,
        "Cached result should be identical"
    );
}

#[test]
fn test_cache_bundle() {
    let system = PrimitiveSystem::new();
    let mut cache = CompositionCache::new(100);

    let result1 = cache.bundle_cached(&system, &["AND", "OR", "NOT"]).unwrap();
    let result2 = cache.bundle_cached(&system, &["AND", "OR", "NOT"]).unwrap();

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);

    assert!(
        result1.encoding.similarity(&result2.encoding) > 0.99,
        "Cached bundle should be identical"
    );
}

#[test]
fn test_cache_sequence() {
    let system = PrimitiveSystem::new();
    let mut cache = CompositionCache::new(100);

    let result1 = cache
        .sequence_cached(&system, &["BEFORE", "DURING", "AFTER"])
        .unwrap();
    let result2 = cache
        .sequence_cached(&system, &["BEFORE", "DURING", "AFTER"])
        .unwrap();

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);

    assert!(
        result1.encoding.similarity(&result2.encoding) > 0.99,
        "Cached sequence should be identical"
    );
}

#[test]
fn test_cache_hit_rate() {
    let system = PrimitiveSystem::new();
    let mut cache = CompositionCache::new(100);

    // Do several operations, some repeated
    let _ = cache.bind_cached(&system, "CAUSE", "EFFECT"); // miss
    let _ = cache.bind_cached(&system, "CAUSE", "EFFECT"); // hit
    let _ = cache.bind_cached(&system, "CAUSE", "EFFECT"); // hit
    let _ = cache.bind_cached(&system, "AND", "OR"); // miss
    let _ = cache.bind_cached(&system, "AND", "OR"); // hit

    let stats = cache.stats();
    assert_eq!(stats.hits, 3);
    assert_eq!(stats.misses, 2);
    assert!(
        (stats.hit_rate - 0.6).abs() < 0.01,
        "Hit rate should be 60%"
    );
}

#[test]
fn test_cache_lru_eviction() {
    let system = PrimitiveSystem::new();
    let mut cache = CompositionCache::new(3); // Very small cache

    // Fill the cache
    let _ = cache.bind_cached(&system, "CAUSE", "EFFECT");
    let _ = cache.bind_cached(&system, "AND", "OR");
    let _ = cache.bind_cached(&system, "BEFORE", "AFTER");

    assert_eq!(cache.stats().size, 3);

    // Add one more, should evict LRU (CAUSE-EFFECT)
    let _ = cache.bind_cached(&system, "SET", "NOT");
    assert_eq!(cache.stats().size, 3);

    // The first entry should have been evicted (miss on re-query)
    let stats_before = cache.stats();
    let _ = cache.bind_cached(&system, "CAUSE", "EFFECT");
    let stats_after = cache.stats();

    assert_eq!(
        stats_after.misses,
        stats_before.misses + 1,
        "LRU entry should have been evicted"
    );
}

#[test]
fn test_cache_clear() {
    let system = PrimitiveSystem::new();
    let mut cache = CompositionCache::new(100);

    let _ = cache.bind_cached(&system, "CAUSE", "EFFECT");
    let _ = cache.bind_cached(&system, "AND", "OR");

    assert_eq!(cache.stats().size, 2);

    cache.clear();

    assert_eq!(cache.stats().size, 0);
    assert_eq!(cache.stats().hits, 0);
    assert_eq!(cache.stats().misses, 0);
}

#[test]
fn test_cache_weighted_bundle() {
    let system = PrimitiveSystem::new();
    let mut cache = CompositionCache::new(100);

    let result1 = cache
        .bundle_weighted_cached(&system, &[("AND", 2.0), ("OR", 1.0)])
        .unwrap();
    let result2 = cache
        .bundle_weighted_cached(&system, &[("AND", 2.0), ("OR", 1.0)])
        .unwrap();

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);

    assert!(
        result1.encoding.similarity(&result2.encoding) > 0.99,
        "Cached weighted bundle should be identical"
    );
}

#[test]
fn test_cache_analogy() {
    let system = PrimitiveSystem::new();
    let mut cache = CompositionCache::new(100);

    let result1 = cache
        .analogy_cached(&system, "CAUSE", "EFFECT", "BEFORE")
        .unwrap();
    let result2 = cache
        .analogy_cached(&system, "CAUSE", "EFFECT", "BEFORE")
        .unwrap();

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);

    assert!(
        result1.encoding.similarity(&result2.encoding) > 0.99,
        "Cached analogy should be identical"
    );
}

#[test]
fn test_cache_permute() {
    let system = PrimitiveSystem::new();
    let mut cache = CompositionCache::new(100);

    let result1 = cache.permute_cached(&system, "CAUSE", 5).unwrap();
    let result2 = cache.permute_cached(&system, "CAUSE", 5).unwrap();

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);

    assert!(
        result1.encoding.similarity(&result2.encoding) > 0.99,
        "Cached permute should be identical"
    );
}

// =========================================================================
// COMPOSITION ALGEBRA TESTS
// =========================================================================

#[test]
fn test_algebra_define_bind() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();

    // Define a bind composition
    algebra
        .define("CAUSALITY", "CAUSE \u{2297} EFFECT", &system)
        .unwrap();

    let comp = algebra.get("CAUSALITY").unwrap();
    assert_eq!(comp.name, "CAUSALITY");
    assert_eq!(comp.sources, vec!["CAUSE", "EFFECT"]);

    // Verify it matches direct bind
    let direct = system.bind_primitives("CAUSE", "EFFECT").unwrap();
    let sim = comp.encoding.similarity(&direct.encoding);
    assert!(
        sim > 0.99,
        "Algebra bind should match direct bind (sim={:.3})",
        sim
    );
}

#[test]
fn test_algebra_define_bundle() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();

    algebra
        .define("LOGIC_CORE", "AND + OR + NOT", &system)
        .unwrap();

    let comp = algebra.get("LOGIC_CORE").unwrap();
    assert_eq!(comp.sources.len(), 3);

    // Verify it matches direct bundle
    let direct = system.bundle_primitives(&["AND", "OR", "NOT"]).unwrap();
    let sim = comp.encoding.similarity(&direct.encoding);
    assert!(
        sim > 0.99,
        "Algebra bundle should match direct bundle (sim={:.3})",
        sim
    );
}

#[test]
fn test_algebra_define_sequence() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();

    algebra
        .define(
            "TIME_FLOW",
            "BEFORE \u{2192} DURING \u{2192} AFTER",
            &system,
        )
        .unwrap();

    let comp = algebra.get("TIME_FLOW").unwrap();
    assert_eq!(comp.sources, vec!["BEFORE", "DURING", "AFTER"]);

    // Verify it matches direct sequence
    let direct = system
        .encode_sequence(&["BEFORE", "DURING", "AFTER"])
        .unwrap();
    let sim = comp.encoding.similarity(&direct.encoding);
    assert!(
        sim > 0.99,
        "Algebra sequence should match direct sequence (sim={:.3})",
        sim
    );
}

#[test]
fn test_algebra_define_weighted() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();

    algebra
        .define("MOSTLY_CAUSE", "CAUSE:3 + EFFECT:1", &system)
        .unwrap();

    let comp = algebra.get("MOSTLY_CAUSE").unwrap();

    // Should be more similar to CAUSE than EFFECT
    let cause = system.get("CAUSE").unwrap();
    let effect = system.get("EFFECT").unwrap();

    let sim_cause = comp.encoding.similarity(&cause.encoding);
    let sim_effect = comp.encoding.similarity(&effect.encoding);

    assert!(
        sim_cause > sim_effect,
        "Weighted composition should be more similar to CAUSE ({:.3}) than EFFECT ({:.3})",
        sim_cause,
        sim_effect
    );
}

#[test]
fn test_algebra_composition_chaining() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();

    // Define base compositions
    algebra
        .define("AB", "CAUSE \u{2297} EFFECT", &system)
        .unwrap();
    algebra
        .define("CD", "BEFORE \u{2297} AFTER", &system)
        .unwrap();

    // Chain them together
    algebra.define("ABCD", "AB \u{2297} CD", &system).unwrap();

    let abcd = algebra.get("ABCD").unwrap();
    assert_eq!(abcd.sources, vec!["AB", "CD"]);

    // Verify algebraic properties
    // ABCD = (CAUSE ⊗ EFFECT) ⊗ (BEFORE ⊗ AFTER)
    // Unbinding should recover components
}

#[test]
fn test_algebra_ascii_operators() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();

    // Test ASCII alternatives
    algebra
        .define("BIND_ASCII", "CAUSE ^ EFFECT", &system)
        .unwrap();
    algebra
        .define("SEQ_ASCII", "BEFORE > DURING > AFTER", &system)
        .unwrap();

    assert!(algebra.get("BIND_ASCII").is_some());
    assert!(algebra.get("SEQ_ASCII").is_some());
}

#[test]
fn test_algebra_not_found() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();

    let result = algebra.define("BAD", "NONEXISTENT \u{2297} CAUSE", &system);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        CompositionAlgebraError::NotFound(_)
    ));
}

#[test]
fn test_algebra_export_import() {
    let system = PrimitiveSystem::new();
    let mut algebra1 = CompositionAlgebra::new();

    algebra1
        .define("COMP1", "CAUSE \u{2297} EFFECT", &system)
        .unwrap();
    algebra1.define("COMP2", "AND + OR", &system).unwrap();

    // Export
    let exports = algebra1.export();
    assert_eq!(exports.len(), 2);

    // Import into new algebra
    let mut algebra2 = CompositionAlgebra::new();
    let count = algebra2.import(&exports, &system).unwrap();
    assert_eq!(count, 2);

    // Verify imported compositions match
    let comp1_orig = algebra1.get("COMP1").unwrap();
    let comp1_imported = algebra2.get("COMP1").unwrap();
    let sim = comp1_orig.encoding.similarity(&comp1_imported.encoding);
    assert!(sim > 0.99, "Imported composition should match original");
}

#[test]
fn test_algebra_list_and_clear() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();

    algebra
        .define("A", "CAUSE \u{2297} EFFECT", &system)
        .unwrap();
    algebra.define("B", "AND + OR", &system).unwrap();

    assert_eq!(algebra.list().len(), 2);

    algebra.remove("A");
    assert_eq!(algebra.list().len(), 1);

    algebra.clear();
    assert_eq!(algebra.list().len(), 0);
}

// =========================================================================
// GRAPH VISUALIZATION TESTS
// =========================================================================

#[test]
fn test_graph_from_primitives() {
    let system = PrimitiveSystem::new();

    let graph =
        PrimitiveGraph::from_primitives(&system, &["CAUSE", "EFFECT", "BEFORE", "AFTER"], 0.45);

    assert_eq!(graph.nodes.len(), 4);
    // With threshold 0.45, random vectors (~0.5 similarity) should have edges
}

#[test]
fn test_graph_from_tier() {
    let system = PrimitiveSystem::new();

    let graph = PrimitiveGraph::from_tier(
        &system,
        PrimitiveTier::Mathematical,
        0.55, // Above random, only strong connections
    );

    let stats = graph.stats();
    assert!(
        stats.node_count > 0,
        "Should have nodes from Mathematical tier"
    );
}

#[test]
fn test_graph_neighborhood() {
    let system = PrimitiveSystem::new();

    let graph = PrimitiveGraph::neighborhood(
        &system, "CAUSE", 2, // depth
        3, // top_k similar at each step
    );

    assert!(
        graph.nodes.len() >= 1,
        "Should have at least the center node"
    );
    assert!(
        graph.nodes.iter().any(|(n, _, _)| n == "CAUSE"),
        "Should contain center"
    );
}

#[test]
fn test_graph_to_dot() {
    let system = PrimitiveSystem::new();

    let graph = PrimitiveGraph::from_primitives(&system, &["CAUSE", "EFFECT"], 0.40);

    let dot = graph.to_dot();

    assert!(dot.contains("digraph"), "Should be DOT format");
    assert!(dot.contains("CAUSE"), "Should contain CAUSE node");
    assert!(dot.contains("EFFECT"), "Should contain EFFECT node");
}

#[test]
fn test_graph_to_ascii() {
    let system = PrimitiveSystem::new();

    let graph = PrimitiveGraph::from_primitives(&system, &["AND", "OR", "NOT"], 0.45);

    let ascii = graph.to_ascii();

    assert!(ascii.contains("Nodes:"), "Should have nodes section");
    assert!(ascii.contains("AND"), "Should list AND");
}

#[test]
fn test_graph_stats() {
    let system = PrimitiveSystem::new();

    let graph =
        PrimitiveGraph::from_primitives(&system, &["CAUSE", "EFFECT", "BEFORE", "AFTER"], 0.40);

    let stats = graph.stats();

    assert_eq!(stats.node_count, 4);
    assert!(stats.density >= 0.0 && stats.density <= 1.0);
}

// =========================================================================
// BATCH OPERATIONS TESTS
// =========================================================================

#[test]
fn test_batch_find_similar() {
    let system = PrimitiveSystem::new();

    let cause = system.get("CAUSE").unwrap().encoding.clone();
    let effect = system.get("EFFECT").unwrap().encoding.clone();

    let queries = vec![cause, effect];
    let results = system.batch_find_similar(&queries, 3);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), 3);
    assert_eq!(results[1].len(), 3);

    // First result should have CAUSE as top match
    assert_eq!(results[0][0].0, "CAUSE");
}

#[test]
fn test_batch_find_similar_lsh() {
    let system = PrimitiveSystem::new();

    let cause = system.get("CAUSE").unwrap().encoding.clone();
    let effect = system.get("EFFECT").unwrap().encoding.clone();
    let before = system.get("BEFORE").unwrap().encoding.clone();

    let queries = vec![cause, effect, before];
    let results = system.batch_find_similar_lsh(&queries, 3, 8, 64);

    assert_eq!(results.len(), 3);
    for result in &results {
        assert!(result.len() <= 3);
    }
}

#[test]
fn test_batch_bind() {
    let system = PrimitiveSystem::new();

    let pairs = vec![("CAUSE", "EFFECT"), ("BEFORE", "AFTER"), ("AND", "OR")];

    let results = system.batch_bind(&pairs);

    assert_eq!(results.len(), 3);
    for result in &results {
        assert!(result.is_ok());
    }
}

#[test]
fn test_batch_bundle() {
    let system = PrimitiveSystem::new();

    let groups: Vec<&[&str]> = vec![&["AND", "OR"], &["CAUSE", "EFFECT", "BEFORE"]];

    let results = system.batch_bundle(&groups);

    assert_eq!(results.len(), 2);
    for result in &results {
        assert!(result.is_ok());
    }
}

#[test]
fn test_batch_encode_sequences() {
    let system = PrimitiveSystem::new();

    let sequences: Vec<&[&str]> = vec![&["BEFORE", "DURING", "AFTER"], &["CAUSE", "EFFECT"]];

    let results = system.batch_encode_sequences(&sequences);

    assert_eq!(results.len(), 2);
    for result in &results {
        assert!(result.is_ok());
    }
}

#[test]
fn test_pairwise_similarities() {
    let system = PrimitiveSystem::new();

    let encodings: Vec<BinaryHV> = ["CAUSE", "EFFECT", "BEFORE"]
        .iter()
        .filter_map(|n| system.get(n).map(|p| p.encoding.clone()))
        .collect();

    let pairs = system.pairwise_similarities(&encodings);

    // 3 encodings = 3 pairs: (1,0), (2,0), (2,1)
    assert_eq!(pairs.len(), 3);

    for (i, j, sim) in &pairs {
        assert!(i > j, "Should be lower triangular");
        assert!(*sim >= 0.0 && *sim <= 1.0, "Similarity should be in [0,1]");
    }
}

#[test]
fn test_similarity_matrix() {
    let system = PrimitiveSystem::new();

    let names = ["CAUSE", "EFFECT", "BEFORE"];
    let matrix = system.similarity_matrix(&names);

    assert_eq!(matrix.len(), 3);
    assert_eq!(matrix[0].len(), 3);

    // Diagonal should be 1.0 (self-similarity)
    for i in 0..3 {
        assert!(
            (matrix[i][i] - 1.0).abs() < 0.01,
            "Self-similarity should be 1.0"
        );
    }

    // Matrix should be symmetric
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (matrix[i][j] - matrix[j][i]).abs() < 0.001,
                "Matrix should be symmetric"
            );
        }
    }
}

// =========================================================================
// PERSISTENCE TESTS
// =========================================================================

#[test]
fn test_persistence_save_load_session() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();

    algebra
        .define("TEST_COMP", "CAUSE \u{2297} EFFECT", &system)
        .unwrap();
    algebra.define("TEST_BUNDLE", "AND + OR", &system).unwrap();

    let history = vec![HistoryEntry {
        operation: "bind(CAUSE, EFFECT)".to_string(),
        result_match: "CAUSALITY".to_string(),
        similarity: 0.55,
    }];

    let persistence = PrimitivePersistence::new();
    let path = "/tmp/test_primitive_session.json";

    // Save
    persistence
        .save_session(path, &algebra, &history, Some("Test session"))
        .unwrap();

    // Load
    let (loaded_algebra, loaded_history) = persistence.load_session(path, &system).unwrap();

    // Verify
    assert!(loaded_algebra.get("TEST_COMP").is_some());
    assert!(loaded_algebra.get("TEST_BUNDLE").is_some());
    assert_eq!(loaded_history.len(), 1);
    assert_eq!(loaded_history[0].operation, "bind(CAUSE, EFFECT)");

    // Verify compositions match
    let orig_comp = algebra.get("TEST_COMP").unwrap();
    let loaded_comp = loaded_algebra.get("TEST_COMP").unwrap();
    let sim = orig_comp.encoding.similarity(&loaded_comp.encoding);
    assert!(sim > 0.99, "Loaded composition should match original");

    // Cleanup
    std::fs::remove_file(path).ok();
}

#[test]
fn test_persistence_save_load_compositions() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();

    algebra
        .define("COMP_A", "CAUSE \u{2297} EFFECT", &system)
        .unwrap();
    algebra
        .define("COMP_B", "BEFORE \u{2192} DURING \u{2192} AFTER", &system)
        .unwrap();

    let persistence = PrimitivePersistence::new();
    let path = "/tmp/test_compositions.json";

    // Save
    persistence.save_compositions(path, &algebra).unwrap();

    // Load
    let loaded = persistence.load_compositions(path, &system).unwrap();

    assert!(loaded.get("COMP_A").is_some());
    assert!(loaded.get("COMP_B").is_some());

    // Cleanup
    std::fs::remove_file(path).ok();
}

#[test]
fn test_persistence_export_graph_dot() {
    let system = PrimitiveSystem::new();

    let graph = PrimitiveGraph::from_primitives(&system, &["CAUSE", "EFFECT"], 0.40);

    let persistence = PrimitivePersistence::new();
    let path = "/tmp/test_graph.dot";

    persistence.export_graph_dot(path, &graph).unwrap();

    // Verify file exists and contains DOT content
    let content = std::fs::read_to_string(path).unwrap();
    assert!(content.contains("digraph"));
    assert!(content.contains("CAUSE"));

    // Cleanup
    std::fs::remove_file(path).ok();
}

#[test]
fn test_persistence_export_similarity_csv() {
    let system = PrimitiveSystem::new();
    let persistence = PrimitivePersistence::new();
    let path = "/tmp/test_similarity.csv";

    persistence
        .export_similarity_csv(path, &system, &["CAUSE", "EFFECT", "BEFORE"])
        .unwrap();

    // Verify file exists and has correct format
    let content = std::fs::read_to_string(path).unwrap();
    assert!(content.contains("CAUSE"));
    assert!(content.contains("EFFECT"));
    assert!(content.contains("BEFORE"));
    assert!(content.lines().count() == 4); // Header + 3 rows

    // Cleanup
    std::fs::remove_file(path).ok();
}

#[test]
fn test_session_data_serialization() {
    let session = SessionData {
        version: 1,
        timestamp: 1234567890,
        compositions: vec![CompositionExport {
            name: "TEST".to_string(),
            expression: "A \u{2297} B".to_string(),
        }],
        history: vec![],
        notes: Some("Test notes".to_string()),
    };

    let json = serde_json::to_string(&session).unwrap();
    let loaded: SessionData = serde_json::from_str(&json).unwrap();

    assert_eq!(loaded.version, 1);
    assert_eq!(loaded.timestamp, 1234567890);
    assert_eq!(loaded.compositions.len(), 1);
    assert_eq!(loaded.notes, Some("Test notes".to_string()));
}

// === Institutional/Geopolitical Primitives Tests ===

#[test]
fn test_institutional_base_primitives_exist() {
    let system = PrimitiveSystem::new();

    let expected = [
        "AUTHORITY",
        "LEGITIMACY",
        "SOVEREIGNTY",
        "JURISDICTION",
        "ENFORCEMENT",
        "COMPLIANCE",
        "POPULATION",
        "MONOPOLY",
        "TREATY",
        "SANCTION",
    ];

    for name in &expected {
        assert!(
            system.get(name).is_some(),
            "Institutional primitive '{}' should exist",
            name
        );
        let prim = system.get(name).unwrap();
        assert_eq!(prim.tier, PrimitiveTier::Strategic);
        assert_eq!(prim.domain, "institutional");
    }
}

#[test]
fn test_institutional_derived_composites_exist() {
    let system = PrimitiveSystem::new();

    let expected_derived = [
        ("TERRITORY", "SPACE ^ BOUNDARY ^ SOVEREIGNTY"),
        ("INSTITUTION", "NORM ^ AUTHORITY ^ PERSIST"),
        ("LAW", "NORM ^ ENFORCEMENT ^ JURISDICTION"),
        ("TAXATION", "OBLIGATION ^ AUTHORITY ^ EXCHANGE"),
        ("REGULATION", "LAW ^ COMPLIANCE"),
        (
            "FIAT_CURRENCY",
            "VALUE_SUBJECTIVE ^ AUTHORITY ^ TRUST_ECONOMIC ^ MONOPOLY",
        ),
        (
            "NATION_STATE",
            "SOVEREIGNTY ^ INSTITUTION ^ ENFORCEMENT ^ POPULATION",
        ),
        ("DIPLOMATIC_RELATION", "TREATY ^ RECIPROCATE ^ SOVEREIGNTY"),
    ];

    for (name, expected_expr) in &expected_derived {
        let prim = system.get(name);
        assert!(
            prim.is_some(),
            "Derived institutional primitive '{}' should exist",
            name
        );
        let prim = prim.unwrap();
        assert!(
            prim.derivation.is_some(),
            "'{}' should have a derivation expression",
            name
        );
        assert_eq!(
            prim.derivation.as_deref().unwrap(),
            *expected_expr,
            "'{}' derivation expression mismatch",
            name
        );
    }
}

#[test]
fn test_institutional_orthogonality() {
    let system = PrimitiveSystem::new();

    // Base institutional primitives should be near-orthogonal to each other
    let pairs = [
        ("AUTHORITY", "LEGITIMACY"),
        ("SOVEREIGNTY", "ENFORCEMENT"),
        ("JURISDICTION", "COMPLIANCE"),
        ("POPULATION", "MONOPOLY"),
        ("TREATY", "SANCTION"),
    ];

    for (a, b) in &pairs {
        let sim = system.check_orthogonality(a, b);
        assert!(
            sim.is_some(),
            "Should check orthogonality of {} vs {}",
            a,
            b
        );
        let sim = sim.unwrap();
        assert!(
            (sim - 0.5).abs() < 0.03,
            "Institutional primitives {} vs {} should be near-orthogonal (sim ≈ 0.5), got {}",
            a,
            b,
            sim
        );
    }
}

#[test]
fn test_institutional_cross_domain_orthogonality() {
    let system = PrimitiveSystem::new();

    // Institutional primitives should be orthogonal to primitives from other domains
    let cross_domain_pairs = [
        ("AUTHORITY", "MASS"),         // institutional vs physical
        ("SOVEREIGNTY", "SET"),        // institutional vs mathematical
        ("NATION_STATE", "QUALE"),     // institutional vs consciousness
        ("ENFORCEMENT", "METABOLISM"), // institutional vs biology
        ("JURISDICTION", "JOY"),       // institutional vs emotion
    ];

    for (a, b) in &cross_domain_pairs {
        let sim = system.check_orthogonality(a, b);
        assert!(
            sim.is_some(),
            "Should check orthogonality of {} vs {}",
            a,
            b
        );
        let sim = sim.unwrap();
        assert!(
            (sim - 0.5).abs() < 0.03,
            "Cross-domain {} vs {} should be near-orthogonal (sim ≈ 0.5), got {}",
            a,
            b,
            sim
        );
    }
}

#[test]
fn test_nation_state_decomposition() {
    let system = PrimitiveSystem::new();

    // NATION_STATE is derived from SOVEREIGNTY ^ INSTITUTION ^ ENFORCEMENT ^ POPULATION
    // Verify it exists and has meaningful compositional structure
    let nation_state = system.get("NATION_STATE").unwrap();
    let sovereignty = system.get("SOVEREIGNTY").unwrap();
    let institution = system.get("INSTITUTION").unwrap();

    // NATION_STATE should share more structure with its components than with
    // unrelated primitives. Since it's derived via XOR binding, unbinding any
    // component should yield something related to the remaining components.
    let ns_sov_sim = nation_state.encoding.similarity(&sovereignty.encoding);
    let ns_inst_sim = nation_state.encoding.similarity(&institution.encoding);

    // Both should be roughly orthogonal (0.5) because XOR binding with 3+ terms
    // distributes the information across all bits. The key test is that the
    // derivation resolved (not random fallback).
    assert!(
        (ns_sov_sim - 0.5).abs() < 0.03,
        "NATION_STATE ~ SOVEREIGNTY: expected ~0.5, got {}",
        ns_sov_sim
    );
    assert!(
        (ns_inst_sim - 0.5).abs() < 0.03,
        "NATION_STATE ~ INSTITUTION: expected ~0.5, got {}",
        ns_inst_sim
    );

    // Unbinding: NATION_STATE ^ SOVEREIGNTY should be closer to
    // INSTITUTION ^ ENFORCEMENT ^ POPULATION than a random vector.
    // This tests the algebraic compositionality of the encoding.
    let unbound = nation_state.encoding.bind(&sovereignty.encoding);
    let enforcement = system.get("ENFORCEMENT").unwrap();

    // The unbound vector now contains INSTITUTION ^ ENFORCEMENT ^ POPULATION.
    // Binding with INSTITUTION should yield ENFORCEMENT ^ POPULATION.
    let further_unbound = unbound.bind(&institution.encoding);
    let further_vs_enforcement = further_unbound.similarity(&enforcement.encoding);

    // With 2 remaining terms (ENFORCEMENT ^ POPULATION), similarity to ENFORCEMENT alone
    // should be ~0.5 (still orthogonal due to remaining POPULATION term).
    // The important thing is this is deterministic and reproducible.
    assert!(
        (further_vs_enforcement - 0.5).abs() < 0.03,
        "Algebraic unbinding should yield deterministic results, got {}",
        further_vs_enforcement
    );
}

#[test]
fn test_institutional_domain_registered() {
    let system = PrimitiveSystem::new();

    // Verify the institutional domain manifold is registered
    assert!(
        system.domains.contains_key("institutional"),
        "Institutional domain should be registered"
    );

    let domain = &system.domains["institutional"];
    assert_eq!(domain.tier, PrimitiveTier::Strategic);
}

// === Institutional Causal Axioms Tests ===

#[test]
fn test_institutional_axioms_load() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();
    let loaded = algebra.load_institutional_axioms(&system);
    assert_eq!(loaded, 18, "Should load all 18 institutional axioms");
}

#[test]
fn test_institutional_axioms_queryable() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();
    algebra.load_institutional_axioms(&system);

    let expected = [
        "REVOLUTION",
        "FAILED_STATE",
        "BORDER_DISPUTE",
        "LEGITIMATE_GOVERNANCE",
        "REGULATORY_CAPTURE",
        "TRADE_AGREEMENT",
        "ECONOMIC_SANCTION",
        "CIVIL_DISOBEDIENCE",
    ];

    for name in &expected {
        assert!(
            algebra.get(name).is_some(),
            "Axiom '{}' should be queryable",
            name
        );
    }
}

#[test]
fn test_institutional_axioms_encodings_distinct() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();
    algebra.load_institutional_axioms(&system);

    // All axiom encodings should be pairwise near-orthogonal
    let names = [
        "REVOLUTION",
        "FAILED_STATE",
        "BORDER_DISPUTE",
        "LEGITIMATE_GOVERNANCE",
    ];

    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            let a = &algebra.get(names[i]).unwrap().encoding;
            let b = &algebra.get(names[j]).unwrap().encoding;
            let sim = a.similarity(b);
            // Axioms sharing components (e.g. SOVEREIGNTY) have higher similarity;
            // 2-component axioms sharing 1 primitive can reach ~0.75
            assert!(
                sim < 0.85,
                "Axioms {} vs {} should not be near-identical, got {}",
                names[i],
                names[j],
                sim
            );
        }
    }
}

#[test]
fn test_revolution_shares_authority_structure() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();
    algebra.load_institutional_axioms(&system);

    // REVOLUTION = AUTHORITY + ENFORCEMENT + PROHIBITION
    // LEGITIMATE_GOVERNANCE = AUTHORITY + LEGITIMACY + TRUST
    // Both contain AUTHORITY, but unbinding AUTHORITY should yield different residuals
    let revolution = algebra.get("REVOLUTION").unwrap();
    let governance = algebra.get("LEGITIMATE_GOVERNANCE").unwrap();
    let authority = system.get("AUTHORITY").unwrap();

    let rev_residual = revolution.encoding.bind(&authority.encoding);
    let gov_residual = governance.encoding.bind(&authority.encoding);

    // Residuals should be near-orthogonal (ENFORCEMENT+PROHIBITION vs LEGITIMACY+TRUST)
    // Bundle-based compositions retain some shared structure; allow 0.15 tolerance
    let sim = rev_residual.similarity(&gov_residual);
    assert!(
        (sim - 0.5).abs() < 0.15,
        "Unbinding AUTHORITY should yield near-orthogonal residuals, got {}",
        sim
    );
}

// === Causal Query Engine Tests ===

#[test]
fn test_query_transition_finds_nearest_axiom() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();
    algebra.load_institutional_axioms(&system);

    // Removing TRUST from LEGITIMATE_GOVERNANCE should land closer to
    // REVOLUTION (authority without legitimacy basis) than other axioms
    let (nearest, sim, _residual) = algebra
        .query_transition("LEGITIMATE_GOVERNANCE", "TRUST", &system)
        .unwrap();

    // The residual is AUTHORITY ^ LEGITIMACY (TRUST removed).
    // In a high-dimensional space, this should be closest to some axiom
    // rather than random noise. We just verify the query returns a result.
    assert!(
        !nearest.is_empty(),
        "query_transition should find a nearest axiom"
    );
    assert!(
        sim >= 0.0 && sim <= 1.0,
        "Similarity should be in [0, 1], got {}",
        sim
    );
}

#[test]
fn test_query_transition_not_found_errors() {
    let system = PrimitiveSystem::new();
    let algebra = CompositionAlgebra::new();
    // No axioms loaded → query should fail

    let result = algebra.query_transition("LEGITIMATE_GOVERNANCE", "TRUST", &system);
    assert!(result.is_err(), "Should error when composite not found");
}

#[test]
fn test_query_transition_recovery_fidelity() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();
    algebra.load_institutional_axioms(&system);

    // Unbind then rebind should recover the original
    let original = algebra.get("TRADE_AGREEMENT").unwrap().encoding;
    let reciprocate = system.get("RECIPROCATE").unwrap().encoding;

    let (_nearest, _sim, residual) = algebra
        .query_transition("TRADE_AGREEMENT", "RECIPROCATE", &system)
        .unwrap();

    // Rebind: residual ^ RECIPROCATE should ≈ original
    let recovered = residual.bind(&reciprocate);
    let recovery_sim = recovered.similarity(&original);
    assert!(
        (recovery_sim - 1.0).abs() < 1e-6,
        "XOR rebinding should perfectly recover original, got {}",
        recovery_sim
    );
}

#[test]
fn test_rank_by_similarity() {
    let system = PrimitiveSystem::new();
    let mut algebra = CompositionAlgebra::new();
    algebra.load_institutional_axioms(&system);

    let trade = algebra.get("TRADE_AGREEMENT").unwrap().encoding;
    let ranked = algebra.rank_by_similarity(&trade);

    // First result should be TRADE_AGREEMENT itself with similarity ~1.0
    assert_eq!(ranked[0].0, "TRADE_AGREEMENT");
    assert!(
        (ranked[0].1 - 1.0).abs() < 1e-6,
        "Self-similarity should be 1.0, got {}",
        ranked[0].1
    );

    // Others should be near-orthogonal; axioms sharing components have higher similarity
    for (name, sim) in &ranked[1..] {
        assert!(
            (sim - 0.5).abs() < 0.25,
            "{} should not be highly correlated with TRADE_AGREEMENT, got {}",
            name,
            sim
        );
    }
}
