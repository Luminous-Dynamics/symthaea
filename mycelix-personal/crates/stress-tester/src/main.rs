use luminous_sim_core::UnifiedConfig;
use serde::{Deserialize, Serialize};
use stress_tester::{GovernanceGenome, HeadlessSim};

/// Evolutionary manager for civilizational tuning.
pub struct Evolver {
    pub population: Vec<GovernanceGenome>,
    pub generation: u32,
}

impl Evolver {
    pub fn new(size: usize) -> Self {
        let mut population = Vec::with_capacity(size);
        for _ in 0..size {
            let mut genome = GovernanceGenome::default();
            genome.mutate(); // Initial variation
            population.push(genome);
        }
        Self {
            population,
            generation: 0,
        }
    }

    pub async fn evolve_generation(&mut self, config: UnifiedConfig) -> GovernanceGenome {
        self.generation += 1;
        println!(
            "\n🧬 Generation {}: Evaluating {} physics variations...",
            self.generation,
            self.population.len()
        );

        let mut results = Vec::new();
        for genome in &self.population {
            let mut sim = HeadlessSim::new(100, config.clone(), None, Some(genome.clone()));
            let report = sim.run(20).await;
            results.push((genome.clone(), report.fitness));
        }

        // Sort by fitness (highest first)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_genome = results[0].0.clone();
        println!(
            "  🏆 Best Fitness: {:.2} (Demurrage: {:.4}, Gate: {:.2})",
            results[0].1, best_genome.demurrage_rate, best_genome.resonance_gate_steward
        );

        // Next generation: Top 25% survive and mutate
        let survivor_count = self.population.len() / 4;
        let mut next_gen = Vec::new();
        for i in 0..self.population.len() {
            let mut child = results[i % survivor_count].0.clone();
            if i >= survivor_count {
                child.mutate();
            }
            next_gen.push(child);
        }
        self.population = next_gen;

        best_genome
    }
}

#[tokio::main]
async fn main() {
    println!("🍄 Mycelix: Starting Civilizational Neuroevolution (The Crucible)...");

    let config = UnifiedConfig::default();
    let mut evolver = Evolver::new(20); // Simulating 20 parallel civilizations

    for _ in 0..5 {
        // Run 5 evolutionary generations
        evolver.evolve_generation(config.clone()).await;
    }

    let final_best = &evolver.population[0];
    println!("\n✅ Neuroevolution Complete!");
    println!("--- Mathematically Optimal Physics Derived ---");
    println!("{:#?}", final_best);
}
