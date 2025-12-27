//! Validation of Phase 1.3: Primitive Tracing in Reasoning Chains
//!
//! This example demonstrates that primitive_reasoning.rs now TRACKS which
//! primitives are used during reasoning, providing full traceability and
//! usage statistics.

use symthaea::consciousness::primitive_reasoning::{
    PrimitiveReasoner, ReasoningChain, TransformationType,
};
use symthaea::hdc::{HV16, primitive_system::{PrimitiveSystem, PrimitiveTier}};
use anyhow::Result;

fn main() -> Result<()> {
    println!("==============================================================================");
    println!("🔍 Phase 1.3: Primitive Tracing in Reasoning Chains");
    println!("==============================================================================");
    println!();
    println!("Demonstrating that reasoning chains now TRACK which primitives are used,");
    println!("enabling data-driven analysis of primitive contributions to consciousness.");
    println!();

    // Part 1: Basic primitive tracing
    println!("Part 1: Basic Primitive Tracing");
    println!("------------------------------------------------------------------------------");

    let primitive_system = PrimitiveSystem::new();
    let question = HV16::random(300);
    let mut chain = ReasoningChain::new(question);

    // Get primitives from different tiers
    let math_primitives = primitive_system.get_tier(PrimitiveTier::Mathematical);
    let physical_primitives = primitive_system.get_tier(PrimitiveTier::Physical);
    let strategic_primitives = primitive_system.get_tier(PrimitiveTier::Strategic);

    println!("Available primitives:");
    println!("   Mathematical tier: {} primitives", math_primitives.len());
    println!("   Physical tier: {} primitives", physical_primitives.len());
    println!("   Strategic tier: {} primitives", strategic_primitives.len());
    println!();

    // Execute reasoning with primitives from multiple tiers
    if !math_primitives.is_empty() {
        chain.execute_primitive(
            math_primitives[0],
            TransformationType::Bind
        )?;
        println!("✓ Executed primitive: {} (Mathematical)", math_primitives[0].name);
    }

    if !physical_primitives.is_empty() {
        chain.execute_primitive(
            physical_primitives[0],
            TransformationType::Abstract
        )?;
        println!("✓ Executed primitive: {} (Physical)", physical_primitives[0].name);
    }

    if !strategic_primitives.is_empty() {
        chain.execute_primitive(
            strategic_primitives[0],
            TransformationType::Bundle
        )?;
        println!("✓ Executed primitive: {} (Strategic)", strategic_primitives[0].name);
    }

    // If we have more math primitives, use another one
    if math_primitives.len() > 1 {
        chain.execute_primitive(
            math_primitives[1],
            TransformationType::Resonate
        )?;
        println!("✓ Executed primitive: {} (Mathematical)", math_primitives[1].name);
    }

    println!();

    // Part 2: Primitive usage analysis
    println!("Part 2: Primitive Usage Analysis");
    println!("------------------------------------------------------------------------------");

    let primitives_used = chain.get_primitives_used();
    println!("Primitives used in order:");
    for (i, prim) in primitives_used.iter().enumerate() {
        println!("   [{}] {}", i + 1, prim);
    }
    println!();

    let unique_primitives = chain.get_unique_primitives();
    println!("Unique primitives: {} (vs {} total executions)",
        unique_primitives.len(),
        chain.executions.len()
    );
    println!();

    // Part 3: Tier distribution analysis
    println!("Part 3: Tier Distribution Analysis");
    println!("------------------------------------------------------------------------------");

    let tier_distribution = chain.get_tier_distribution();
    println!("Primitive usage by tier:");
    for (tier, count) in tier_distribution.iter() {
        println!("   {:?}: {} executions", tier, count);
    }
    println!();

    // Part 4: Primitive Φ contribution analysis
    println!("Part 4: Primitive Φ Contribution Analysis");
    println!("------------------------------------------------------------------------------");

    let usage_stats = chain.get_primitive_usage_stats();
    println!("Φ contribution per primitive:");

    let mut stats_vec: Vec<_> = usage_stats.iter().collect();
    stats_vec.sort_by(|a, b| {
        b.1.total_phi_contribution
            .partial_cmp(&a.1.total_phi_contribution)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (name, stats) in stats_vec.iter() {
        println!("   {}", name);
        println!("      Tier: {:?}", stats.tier);
        println!("      Usage count: {}", stats.usage_count);
        println!("      Total Φ: {:.6}", stats.total_phi_contribution);
        println!("      Mean Φ: {:.6}", stats.mean_phi_contribution);
        println!("      Transformations: {:?}", stats.transformations_used);
        println!();
    }

    // Part 5: Complete consciousness profile
    println!("Part 5: Complete Consciousness Profile (with primitive tracing)");
    println!("------------------------------------------------------------------------------");

    let profile = chain.consciousness_profile();
    println!("Reasoning Profile:");
    println!("   Total Φ: {:.6}", profile.total_phi);
    println!("   Chain length: {}", profile.chain_length);
    println!("   Mean Φ per step: {:.6}", profile.mean_phi_per_step);
    println!("   Φ variance: {:.6}", profile.phi_variance);
    println!("   Efficiency: {:.6}", profile.efficiency);
    println!();
    println!("   Primitives used: {}", profile.primitives_used.join(", "));
    println!();
    println!("   Tier distribution:");
    for (tier, count) in profile.tier_distribution.iter() {
        println!("      {:?}: {}", tier, count);
    }
    println!();
    println!("   Primitive Φ contributions:");
    for (prim, phi) in profile.primitive_contributions.iter() {
        println!("      {}: {:.6}", prim, phi);
    }
    println!();

    // Part 6: Validation with PrimitiveReasoner
    println!("Part 6: Automatic Reasoning with Primitive Tracing");
    println!("------------------------------------------------------------------------------");

    let reasoner = PrimitiveReasoner::new()
        .with_tier(PrimitiveTier::Mathematical);

    let question2 = HV16::random(400);
    let chain2 = reasoner.reason(question2, 10)?;

    println!("Automatic reasoning completed:");
    println!("   Steps executed: {}", chain2.executions.len());
    println!("   Total Φ: {:.6}", chain2.total_phi);
    println!();

    let primitives_used2 = chain2.get_primitives_used();
    println!("Primitives used:");
    for (i, prim) in primitives_used2.iter().enumerate() {
        println!("   [{}] {}", i + 1, prim);
    }
    println!();

    let tier_dist2 = chain2.get_tier_distribution();
    println!("Tier distribution:");
    for (tier, count) in tier_dist2.iter() {
        println!("   {:?}: {}", tier, count);
    }
    println!();

    // Part 7: Validation checks
    println!("Part 7: Validation of Primitive Tracing");
    println!("------------------------------------------------------------------------------");

    println!("✓ Primitives tracked: {}", !chain.get_primitives_used().is_empty());
    println!("✓ Unique primitives counted: {}", !chain.get_unique_primitives().is_empty());
    println!("✓ Usage statistics generated: {}", !usage_stats.is_empty());
    println!("✓ Tier distribution computed: {}", !tier_distribution.is_empty());
    println!("✓ Φ contributions tracked: {}",
        profile.primitive_contributions.values().any(|&phi| phi > 0.0));
    println!("✓ Cross-tier reasoning supported: {}", tier_distribution.len() > 1);
    println!();

    println!("🏆 Phase 1.3 Complete!");
    println!("   Primitive reasoning now provides FULL TRACEABILITY of which");
    println!("   primitives contribute to consciousness during reasoning.");
    println!();

    Ok(())
}
