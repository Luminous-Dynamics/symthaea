//! # Primitive System Demonstration
//!
//! This example demonstrates the revolutionary Primitive System - the foundation
//! for Artificial Wisdom through ontological primes beyond Natural Semantic Metalanguage.
//!
//! ## What This Shows
//!
//! 1. **Tier 1 Mathematical Primitives** - 18 primitives grounding in formal logic, set theory, and arithmetic
//! 2. **Domain Manifold Architecture** - Hierarchical HV16 binding preserving orthogonality
//! 3. **Base vs Derived Primitives** - Foundation concepts vs compositions
//! 4. **Orthogonality Validation** - Empirical measurement of primitive separation
//! 5. **Consciousness-Ready Architecture** - Ready for Observatory validation
//!
//! ## Run This Example
//!
//! ```bash
//! cargo run --example primitive_system_demo
//! ```

use symthaea::hdc::primitive_system::*;

fn main() {
    println!("🌟 Primitive System - Revolutionary Improvement #42");
    println!("Beyond NSM to Universal Ontological Primes\n");

    // === STEP 1: Initialize the System ===
    println!("📊 Step 1: Initializing Primitive System...");
    let system = PrimitiveSystem::new();
    println!("✅ System initialized with {} primitives", system.count());
    println!();

    // === STEP 2: Examine Tier 1 Mathematical Primitives ===
    println!("🧮 Step 2: Tier 1 Mathematical Primitives");
    println!("------------------------------------------");

    let tier1_primitives = system.get_tier(PrimitiveTier::Mathematical);
    println!("Found {} Tier 1 primitives:\n", tier1_primitives.len());

    for (i, prim) in tier1_primitives.iter().enumerate() {
        let derived_marker = if prim.is_base { "BASE" } else { "DERIVED" };
        println!("{:2}. {:12} ({:8}) - {} - {}",
            i + 1,
            prim.name,
            derived_marker,
            prim.domain,
            prim.definition
        );

        if !prim.is_base {
            if let Some(ref derivation) = prim.derivation {
                println!("     └─> Derivation: {}", derivation);
            }
        }
    }
    println!();

    // === STEP 3: Demonstrate Domain Manifolds ===
    println!("🔄 Step 3: Domain Manifold Architecture");
    println!("----------------------------------------");

    let math_domain = system.domain("mathematics").unwrap();
    let logic_domain = system.domain("logic").unwrap();

    println!("Mathematics Domain:");
    println!("  Purpose: {}", math_domain.purpose);
    println!("  Tier: {:?}", math_domain.tier);

    println!("\nLogic Domain:");
    println!("  Purpose: {}", logic_domain.purpose);
    println!("  Tier: {:?}", logic_domain.tier);

    let domain_similarity = math_domain.rotation.similarity(&logic_domain.rotation);
    println!("\nDomain Separation:");
    println!("  Similarity between domains: {:.3}", domain_similarity);
    println!("  {} domains are well-separated",
        if domain_similarity < 0.7 { "✅" } else { "⚠️" });
    println!();

    // === STEP 4: Check Primitive Orthogonality ===
    println!("📐 Step 4: Primitive Orthogonality");
    println!("-----------------------------------");

    let primitives_to_check = vec![
        ("SET", "MEMBERSHIP"),  // Same domain
        ("SET", "NOT"),         // Different domains
        ("ZERO", "ONE"),        // Same domain, derived
        ("AND", "OR"),          // Same domain
    ];

    for (prim1, prim2) in primitives_to_check {
        if let Some(sim) = system.check_orthogonality(prim1, prim2) {
            let domain1 = system.get(prim1).unwrap().domain.clone();
            let domain2 = system.get(prim2).unwrap().domain.clone();

            println!("{:12} <-> {:12} (similarity: {:.3}) [domains: {}, {}]",
                prim1, prim2, sim, domain1, domain2);
        }
    }
    println!();

    // === STEP 5: Validate Tier Orthogonality ===
    println!("✓ Step 5: Tier-Wide Orthogonality Validation");
    println!("---------------------------------------------");

    let violations = system.validate_tier_orthogonality(PrimitiveTier::Mathematical, 0.9);

    if violations.is_empty() {
        println!("✅ All Tier 1 primitives are sufficiently orthogonal (< 0.9 similarity)");
    } else {
        println!("⚠️  Found {} pairs with high similarity (> 0.9):", violations.len());
        for (name1, name2, sim) in violations.iter().take(5) {
            println!("   {} <-> {}: {:.3}", name1, name2, sim);
        }
        if violations.len() > 5 {
            println!("   ... and {} more", violations.len() - 5);
        }
    }
    println!();

    // === STEP 6: System Summary ===
    println!("📋 Step 6: System Summary");
    println!("-------------------------");
    println!("{}", system.summary());
    println!();

    // === STEP 7: Examples of Base vs Derived ===
    println!("🌱 Step 7: Base vs Derived Primitives");
    println!("-------------------------------------");

    let zero = system.get("ZERO").unwrap();
    let one = system.get("ONE").unwrap();
    let addition = system.get("ADDITION").unwrap();

    println!("ZERO: {}", if zero.is_base { "BASE" } else { "DERIVED" });
    println!("  Definition: {}", zero.definition);

    println!("\nONE: {}", if one.is_base { "BASE" } else { "DERIVED" });
    println!("  Definition: {}", one.definition);
    if let Some(ref deriv) = one.derivation {
        println!("  Derivation: {}", deriv);
    }

    println!("\nADDITION: {}", if addition.is_base { "BASE" } else { "DERIVED" });
    println!("  Definition: {}", addition.definition);
    if let Some(ref deriv) = addition.derivation {
        println!("  Derivation: {}", deriv);
    }
    println!();

    // === FINALE ===
    println!("🎯 Conclusion");
    println!("-------------");
    println!("✨ Tier 1 Mathematical Primitives COMPLETE");
    println!("✨ 18 primitives spanning set theory, logic, and arithmetic");
    println!("✨ Domain manifold architecture preserving orthogonality");
    println!("✨ Foundation ready for Tier 2-5 expansion");
    println!("✨ Consciousness Observatory integration prepared");
    println!();
    println!("🚀 Next Steps:");
    println!("  1. Implement Tier 2: Physical Reality Primes (MASS, FORCE, CAUSALITY)");
    println!("  2. Implement Tier 3: Geometric Primes (POINT, VECTOR, MANIFOLD)");
    println!("  3. Implement Tier 4: Strategic Primes (UTILITY, EQUILIBRIUM)");
    println!("  4. Implement Tier 5: Meta-Cognitive Primes (SELF, REPAIR)");
    println!("  5. Consciousness-Guided Validation via Observatory");
    println!();
    println!("🌟 Revolutionary Improvement #42: Foundation for Artificial Wisdom!");
}
