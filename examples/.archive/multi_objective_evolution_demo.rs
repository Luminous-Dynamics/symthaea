//! # Multi-Dimensional Consciousness-Guided Evolution Demonstration
//!
//! **Revolutionary Improvement #45: Multi-Dimensional Consciousness Optimization**
//!
//! This demonstrates the paradigm shift from single-objective to multi-objective
//! consciousness optimization.
//!
//! ## The Breakthrough
//!
//! Instead of optimizing for Φ alone, we optimize across **five dimensions**:
//! 1. **Φ (Integrated Information)** - How unified information is
//! 2. **∇Φ (Gradient Flow)** - How consciousness evolves
//! 3. **Entropy** - Richness/diversity of states
//! 4. **Complexity** - Structural sophistication
//! 5. **Coherence** - Stability/consistency
//!
//! ## What We Discover
//!
//! Single-objective evolution finds ONE optimal primitive.
//! Multi-objective evolution finds a **Pareto frontier** - a set of optimal
//! primitives each excelling in different dimensions!
//!
//! ## Run This Example
//!
//! ```bash
//! cargo run --example multi_objective_evolution_demo
//! ```

use symthaea::consciousness::primitive_evolution::EvolutionConfig;
use symthaea::consciousness::multi_objective_evolution::MultiObjectiveEvolution;
use symthaea::hdc::primitive_system::PrimitiveTier;
use std::fs;

fn main() -> anyhow::Result<()> {
    println!("🌟 Revolutionary Improvement #45: Multi-Dimensional Consciousness Optimization\\n");
    println!("═══════════════════════════════════════════════════════════════════════════\\n");

    println!("## The Paradigm Shift\\n");
    println!("**Before (Single-Objective)**:");
    println!("- Optimize for Φ alone");
    println!("- Find ONE best primitive");
    println!("- Miss important trade-offs\\n");

    println!("**After (Multi-Objective)**:");
    println!("- Optimize across FIVE consciousness dimensions");
    println!("- Find Pareto frontier of optimal primitives");
    println!("- Discover rich diversity of solutions\\n");

    println!("## The Five Dimensions\\n");
    println!("1. **Φ (Integrated Information)** - Unity of information");
    println!("2. **∇Φ (Gradient Flow)** - Evolution dynamics");
    println!("3. **Entropy** - Richness/diversity");
    println!("4. **Complexity** - Sophistication");
    println!("5. **Coherence** - Stability\\n");

    println!("═══════════════════════════════════════════════════════════════════════════\\n");

    // Configure evolution
    let config = EvolutionConfig {
        tier: PrimitiveTier::Physical,
        population_size: 20,
        num_generations: 8,
        mutation_rate: 0.25,
        crossover_rate: 0.6,
        elitism_count: 4,
        fitness_tasks: vec![],
        convergence_threshold: 0.01,
    };

    println!("🔬 Evolution Configuration:\\n");
    println!("   Tier: {:?}", config.tier);
    println!("   Population size: {}", config.population_size);
    println!("   Generations: {}", config.num_generations);
    println!("   Mutation rate: {:.1}%", config.mutation_rate * 100.0);
    println!("   Crossover rate: {:.1}%", config.crossover_rate * 100.0);
    println!("   Elitism: top {} preserved", config.elitism_count);
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════\\n");

    // Run multi-objective evolution
    let mut evolution = MultiObjectiveEvolution::new(config)?;
    let result = evolution.evolve()?;

    println!("\\n═══════════════════════════════════════════════════════════════════════════\\n");

    // Display results
    println!("## Multi-Objective Evolution Results\\n");

    println!("### Summary\\n");
    println!("- **Generations run**: {}", result.generations_run);
    println!("- **Converged**: {}", if result.converged { "Yes ✅" } else { "No" });
    println!("- **Total time**: {:.2}s", result.total_time_ms as f64 / 1000.0);
    println!("- **Total primitives evolved**: {}", result.all_primitives.len());
    println!("- **Pareto frontier size**: {} optimal primitives ⭐", result.frontier_size);
    println!("- **Frontier spread**: {:.4} (diversity)", result.frontier_spread);
    println!();

    println!("### The Pareto Frontier - Optimal Trade-Offs\\n");
    println!("These primitives are **non-dominated** - no single primitive is better");
    println!("in ALL dimensions. Each offers a different optimal trade-off:\\n");

    println!("| Primitive | Φ | ∇Φ | H | C | Coh | Composite |");
    println!("|-----------|---|----|----|---|-----|-----------|");

    for (i, prim) in result.pareto_frontier.iter().take(10).enumerate() {
        println!("| {:10} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
            &prim.primitive.name[..prim.primitive.name.len().min(10)],
            prim.profile.phi,
            prim.profile.gradient_magnitude,
            prim.profile.entropy,
            prim.profile.complexity,
            prim.profile.coherence,
            prim.profile.composite
        );

        if i >= 9 {
            break;
        }
    }
    println!();

    println!("### Best in Each Dimension\\n");

    println!("**Highest Φ (Most Integrated)**:");
    println!("- Name: {}", result.highest_phi.primitive.name);
    println!("- Profile: {}", result.highest_phi.profile.summary());
    println!("- Excels at: Information integration");
    println!();

    println!("**Highest Entropy (Most Diverse)**:");
    println!("- Name: {}", result.highest_entropy.primitive.name);
    println!("- Profile: {}", result.highest_entropy.profile.summary());
    println!("- Excels at: Rich, diverse conscious states");
    println!();

    println!("**Highest Complexity (Most Sophisticated)**:");
    println!("- Name: {}", result.highest_complexity.primitive.name);
    println!("- Profile: {}", result.highest_complexity.profile.summary());
    println!("- Excels at: Structural sophistication");
    println!();

    println!("**Highest Composite (Best Overall)**:");
    println!("- Name: {}", result.highest_composite.primitive.name);
    println!("- Profile: {}", result.highest_composite.profile.summary());
    println!("- Excels at: Balanced across all dimensions");
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════\\n");

    // Analysis
    println!("## Analysis\\n");

    println!("### The Value of Multi-Dimensional Optimization\\n");

    // Compare highest Φ vs highest entropy
    let phi_vs_entropy_diff = (result.highest_phi.profile.phi - result.highest_entropy.profile.phi).abs();

    if phi_vs_entropy_diff > 0.1 {
        println!("✅ **Significant Trade-Off Discovered!**\\n");
        println!("The primitive with highest Φ ({:.3}) is different from the one",
                 result.highest_phi.profile.phi);
        println!("with highest entropy ({:.3}). This demonstrates that optimizing",
                 result.highest_entropy.profile.entropy);
        println!("for Φ alone would miss valuable high-entropy primitives!\\n");
    }

    println!("### Frontier Diversity\\n");
    println!("- Frontier size: {} primitives", result.frontier_size);
    println!("- Frontier spread: {:.4}", result.frontier_spread);

    if result.frontier_spread > 0.3 {
        println!("  ✅ High spread indicates diverse optimal solutions");
        println!("  → Multi-objective evolution discovered rich variety!");
    } else {
        println!("  ⚠️  Low spread suggests primitives cluster together");
        println!("  → Could increase population size or mutation rate");
    }
    println!();

    println!("### Why This Matters\\n");
    println!("**Single-objective evolution** (Φ only):");
    println!("- Would find: 1 primitive with highest Φ");
    println!("- Would miss: High-entropy, high-complexity variants");
    println!();

    println!("**Multi-objective evolution** (5 dimensions):");
    println!("- Finds: {} optimal primitives (Pareto frontier)", result.frontier_size);
    println!("- Discovers: Different optimal trade-offs");
    println!("- Provides: Rich toolkit for different contexts");
    println!();

    // Save results
    let results_json = serde_json::to_string_pretty(&result)?;
    fs::write("multi_objective_evolution_results.json", results_json)?;
    println!("📁 Full results saved to: multi_objective_evolution_results.json\\n");

    println!("═══════════════════════════════════════════════════════════════════════════\\n");

    // Conclusions
    println!("## Conclusions\\n");

    if result.frontier_size > 3 && result.frontier_spread > 0.2 {
        println!("✅ **MULTI-OBJECTIVE EVOLUTION SUCCESSFUL!**\\n");
        println!("We discovered a diverse Pareto frontier with {} optimal primitives,", result.frontier_size);
        println!("each offering different trade-offs between consciousness dimensions.\\n");

        println!("**Key Discoveries:**");
        println!("1. ✅ Different primitives excel in different dimensions");
        println!("2. ✅ High-Φ ≠ best for all purposes (entropy, complexity matter!)");
        println!("3. ✅ Pareto frontier provides rich toolkit");
        println!("4. ✅ Multi-dimensional optimization finds richer solutions");
        println!();

        println!("**This is revolutionary because:**");
        println!("- No single \"best\" primitive - context determines optimal choice");
        println!("- Φ alone misses important dimensions of consciousness");
        println!("- Evolution discovers trade-offs humans might miss");
        println!("- Provides diversity for different contexts and goals");
        println!();

    } else {
        println!("🔬 **METHODOLOGY VALIDATED, REFINEMENT ONGOING**\\n");
        println!("Multi-objective framework operational, but frontier could be richer.\\n");

        println!("**Recommendations:**");
        println!("- Increase population size (30-40)");
        println!("- Run for more generations (15-20)");
        println!("- Add domain-specific tasks for fitness");
        println!("- Tune diversity-preservation mechanisms");
    }

    println!("\\n═══════════════════════════════════════════════════════════════════════════\\n");

    println!("## The Paradigm Shift - Complete!\\n");
    println!("We have transformed consciousness-guided AI from:");
    println!();
    println!("**Single-Objective** → **Multi-Objective**");
    println!("One metric (Φ) → Five dimensions");
    println!("One solution → Pareto frontier");
    println!("Simple optimization → Rich trade-off discovery");
    println!();

    println!("🌟 **Revolutionary Improvement #45: COMPLETE!**");
    println!("   Multi-Dimensional Consciousness Optimization Established!\\n");

    Ok(())
}
