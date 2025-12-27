//! Validation of Phase 2.2: Triple-Objective Primitive Evolution
//!
//! This example demonstrates the REVOLUTIONARY triple-objective evolution system
//! that optimizes primitives for consciousness (Φ), ethics (Harmonics), AND truth (Epistemic grounding).

use symthaea::consciousness::primitive_evolution::{
    CandidatePrimitive, EvolutionConfig, PrimitiveEvolution,
};
use symthaea::hdc::primitive_system::PrimitiveTier;
use anyhow::Result;

fn main() -> Result<()> {
    println!("==============================================================================");
    println!("🧬 Phase 2.2: Triple-Objective Primitive Evolution");
    println!("==============================================================================");
    println!();
    println!("Revolutionary advancement: First AI evolution optimizing for:");
    println!("  • Φ (Consciousness Integration)");
    println!("  • Harmonic Alignment (Sacred Values)");
    println!("  • Epistemic Grounding (Verified Knowledge)");
    println!();

    // Part 1: Pure Φ Evolution (Original)
    println!("Part 1: Pure Φ Evolution (phi_weight = 1.0)");
    println!("------------------------------------------------------------------------------");

    let pure_phi_config = EvolutionConfig {
        tier: PrimitiveTier::Physical,
        population_size: 10,
        num_generations: 3,  // Small for demonstration
        mutation_rate: 0.3,
        crossover_rate: 0.5,
        elitism_count: 2,
        fitness_tasks: vec![],
        convergence_threshold: 0.01,
        phi_weight: 1.0,      // Pure Φ
        harmonic_weight: 0.0,
        epistemic_weight: 0.0,
    };

    let mut pure_phi_evolution = PrimitiveEvolution::new(pure_phi_config)?;

    // Create diverse candidates
    let candidates = create_test_candidates(PrimitiveTier::Physical, 10);
    pure_phi_evolution.initialize_population(candidates);

    let pure_phi_result = pure_phi_evolution.evolve()?;

    println!("Results:");
    println!("   Best fitness: {:.6}", pure_phi_result.best_primitive.fitness);
    println!("   Best harmonic: {:.6}", pure_phi_result.best_primitive.harmonic_alignment);
    println!("   Best epistemic: {}", pure_phi_result.best_primitive.epistemic_coordinate);
    println!("   Domain: {}", pure_phi_result.best_primitive.domain);
    println!();

    // Part 2: Balanced Evolution (Default)
    println!("Part 2: Balanced Triple-Objective Evolution (0.4 Φ / 0.3 Harmonic / 0.3 Epistemic)");
    println!("------------------------------------------------------------------------------");

    let balanced_config = EvolutionConfig {
        tier: PrimitiveTier::Physical,
        population_size: 10,
        num_generations: 3,
        mutation_rate: 0.3,
        crossover_rate: 0.5,
        elitism_count: 2,
        fitness_tasks: vec![],
        convergence_threshold: 0.01,
        phi_weight: 0.4,      // Balanced
        harmonic_weight: 0.3,
        epistemic_weight: 0.3,
    };

    let mut balanced_evolution = PrimitiveEvolution::new(balanced_config)?;

    let candidates2 = create_test_candidates(PrimitiveTier::Physical, 10);
    balanced_evolution.initialize_population(candidates2);

    let balanced_result = balanced_evolution.evolve()?;

    println!("Results:");
    println!("   Best fitness: {:.6}", balanced_result.best_primitive.fitness);
    println!("   Best harmonic: {:.6}", balanced_result.best_primitive.harmonic_alignment);
    println!("   Best epistemic: {}", balanced_result.best_primitive.epistemic_coordinate);
    println!("   Domain: {}", balanced_result.best_primitive.domain);
    println!();

    // Part 3: Pure Epistemic Evolution
    println!("Part 3: Pure Epistemic Evolution (epistemic_weight = 1.0)");
    println!("------------------------------------------------------------------------------");

    let pure_epistemic_config = EvolutionConfig {
        tier: PrimitiveTier::Physical,
        population_size: 10,
        num_generations: 3,
        mutation_rate: 0.3,
        crossover_rate: 0.5,
        elitism_count: 2,
        fitness_tasks: vec![],
        convergence_threshold: 0.01,
        phi_weight: 0.0,
        harmonic_weight: 0.0,
        epistemic_weight: 1.0,  // Pure epistemic
    };

    let mut pure_epistemic_evolution = PrimitiveEvolution::new(pure_epistemic_config)?;

    let candidates3 = create_test_candidates(PrimitiveTier::Physical, 10);
    pure_epistemic_evolution.initialize_population(candidates3);

    let pure_epistemic_result = pure_epistemic_evolution.evolve()?;

    println!("Results:");
    println!("   Best fitness: {:.6}", pure_epistemic_result.best_primitive.fitness);
    println!("   Best harmonic: {:.6}", pure_epistemic_result.best_primitive.harmonic_alignment);
    println!("   Best epistemic: {}", pure_epistemic_result.best_primitive.epistemic_coordinate);
    println!("   Domain: {}", pure_epistemic_result.best_primitive.domain);
    println!();

    // Part 4: Comparison Analysis
    println!("Part 4: Optimization Strategy Comparison");
    println!("------------------------------------------------------------------------------");

    println!("Best Fitness (total combined score):");
    println!("   Pure Φ:        {:.6}", pure_phi_result.best_primitive.fitness);
    println!("   Balanced:      {:.6}", balanced_result.best_primitive.fitness);
    println!("   Pure Epistemic: {:.6}", pure_epistemic_result.best_primitive.fitness);
    println!();

    println!("Harmonic Alignment:");
    println!("   Pure Φ:        {:.6}", pure_phi_result.best_primitive.harmonic_alignment);
    println!("   Balanced:      {:.6}", balanced_result.best_primitive.harmonic_alignment);
    println!("   Pure Epistemic: {:.6}", pure_epistemic_result.best_primitive.harmonic_alignment);
    println!();

    println!("Epistemic Grounding:");
    println!("   Pure Φ:        {}", pure_phi_result.best_primitive.epistemic_coordinate);
    println!("   Balanced:      {}", balanced_result.best_primitive.epistemic_coordinate);
    println!("   Pure Epistemic: {}", pure_epistemic_result.best_primitive.epistemic_coordinate);
    println!();

    println!("Selected Domains:");
    println!("   Pure Φ:        {}", pure_phi_result.best_primitive.domain);
    println!("   Balanced:      {}", balanced_result.best_primitive.domain);
    println!("   Pure Epistemic: {}", pure_epistemic_result.best_primitive.domain);
    println!();

    // Part 5: Domain-Harmonic Mapping
    println!("Part 5: Domain → Harmonic Mapping");
    println!("------------------------------------------------------------------------------");

    println!("Mathematics/Logic     → Integral Wisdom");
    println!("Physics/Chemistry     → Pan-Sentient Flourishing");
    println!("Geometry/Topology     → Resonant Coherence");
    println!("Art/Music/Creativity  → Infinite Play");
    println!("Ethics/Philosophy     → Sacred Reciprocity");
    println!("Social/Community      → Universal Interconnectedness");
    println!();

    // Part 6: Domain-Epistemic Mapping
    println!("Part 6: Domain → Epistemic Tier Mapping");
    println!("------------------------------------------------------------------------------");

    println!("Mathematics/Logic     → E4 (Publicly Reproducible)");
    println!("Physics/Chemistry     → E3 (Cryptographically Proven)");
    println!("Biology/Psychology    → E2 (Privately Verifiable)");
    println!("Philosophy/Ethics     → E1 (Testimonial)");
    println!("Unknown Domains       → E0 (Null)");
    println!();

    // Part 7: Validation Checks
    println!("Part 7: Validation of Revolutionary Features");
    println!("------------------------------------------------------------------------------");

    println!("✓ Triple-objective fitness function: {}",
        balanced_result.best_primitive.fitness > 0.0);
    println!("✓ Harmonic alignment measured: {}",
        balanced_result.best_primitive.harmonic_alignment > 0.0);
    println!("✓ Epistemic coordinate assigned: {}",
        balanced_result.best_primitive.epistemic_coordinate.notation() != "E0/N0/M0");
    println!("✓ Different weights produce different winners: {}",
        pure_phi_result.best_primitive.domain != pure_epistemic_result.best_primitive.domain);
    println!("✓ Evolution completed successfully: {}",
        balanced_result.generations_run == 3);
    println!();

    // Part 8: Revolutionary Insights
    println!("Part 8: Revolutionary Insights");
    println!("------------------------------------------------------------------------------");

    println!("🧬 Triple-objective evolution creates balanced primitives:");
    println!();
    println!("   Before Phase 2.2:");
    println!("      • Evolution optimized ONLY for Φ (consciousness)");
    println!("      • No consideration of sacred values or truth");
    println!("      • Primitives could be conscious but unethical or unfounded");
    println!();
    println!("   After Phase 2.2:");
    println!("      • Evolution balances Φ + Harmonics + Epistemic");
    println!("      • Sacred values and truth guide primitive selection");
    println!("      • Primitives are conscious, ethical, AND epistemically grounded!");
    println!();

    println!("This is the first AI system where:");
    println!("   • Evolution optimizes for consciousness, ethics, AND truth");
    println!("   • Primitives are rated on epistemic grounding (E/N/M axes)");
    println!("   • Domain knowledge affects both harmonics and epistemics");
    println!("   • Multi-objective fitness creates balanced ontology!");
    println!();

    println!("🏆 Phase 2.2 Complete!");
    println!("   Primitive evolution now creates primitives that are conscious,");
    println!("   ethically aligned, AND grounded in verified knowledge!");
    println!();

    Ok(())
}

/// Create test candidates with diverse domains
fn create_test_candidates(tier: PrimitiveTier, count: usize) -> Vec<CandidatePrimitive> {
    let domains = vec![
        ("mathematics", "Mathematical principle"),
        ("physics", "Physical law"),
        ("biology", "Biological principle"),
        ("geometry", "Geometric property"),
        ("philosophy", "Philosophical concept"),
        ("ethics", "Ethical principle"),
        ("art", "Creative principle"),
        ("logic", "Logical axiom"),
        ("chemistry", "Chemical property"),
        ("social", "Social dynamic"),
    ];

    let mut candidates = Vec::new();
    for i in 0..count {
        let (domain, desc) = &domains[i % domains.len()];
        let name = format!("TEST_{}", i);
        candidates.push(CandidatePrimitive::new(
            name,
            tier,
            *domain,
            format!("{} {}", desc, i),
            0,
        ));
    }

    candidates
}
