// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Code Discovery — Emergent solution finding through evolution + compilation
//!
//! When the emitter has no pattern for a problem, instead of returning `todo!()`,
//! this module DISCOVERS a solution by:
//!
//! 1. Seeding a population of code fragment compositions
//! 2. Evolving via mutation and crossover in HDC space
//! 3. Decoding each candidate HV to executable Rust
//! 4. Using `cargo check` / test execution as the FITNESS FUNCTION
//! 5. Persisting successful discoveries for future use
//!
//! This is genuine emergence: the system finds solutions it was never
//! programmed for, guided only by the test suite.

use symthaea_core::hdc::ContinuousHV;

use super::code_executor::{CodeExecutor, ExecutionResult};
use super::hv_code_decoder::{CodeFragment, FragmentCategory, HvCodeDecoder};
use crate::hdc::code_encoder::CodeHDEncoder;

/// Configuration for code discovery
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Population size for evolution
    pub population_size: usize,
    /// Maximum generations before giving up
    pub max_generations: usize,
    /// Mutation rate (probability of mutating each individual)
    pub mutation_rate: f32,
    /// Crossover rate
    pub crossover_rate: f32,
    /// Minimum fitness to consider a discovery
    pub discovery_threshold: f32,
    /// Whether to use compilation as fitness (slower but real)
    pub use_compilation_fitness: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            population_size: 30,
            max_generations: 20,
            mutation_rate: 0.3,
            crossover_rate: 0.5,
            discovery_threshold: 0.5,
            use_compilation_fitness: true,
        }
    }
}

/// A candidate solution during evolution
#[derive(Debug, Clone)]
struct Candidate {
    /// HDC representation
    hv: ContinuousHV,
    /// Decoded source code (cached after decoding)
    code: Option<String>,
    /// Fitness score (0.0 = doesn't compile, 1.0 = all tests pass)
    fitness: f32,
}

/// Result of a discovery attempt
#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    /// Whether a solution was found
    pub found: bool,
    /// The discovered source code (if found)
    pub source: Option<String>,
    /// Whether evaluation was simulated instead of actually compiled/tested
    pub simulated_execution: bool,
    /// Fitness of the best candidate
    pub best_fitness: f32,
    /// Number of generations evolved
    pub generations: usize,
    /// Number of candidates that compiled
    pub compiled_count: usize,
    /// Number of candidates evaluated
    pub total_evaluated: usize,
}

/// The discovery engine — evolves code solutions using compilation as fitness.
pub struct CodeDiscovery {
    config: DiscoveryConfig,
    encoder: CodeHDEncoder,
    decoder: HvCodeDecoder,
}

impl CodeDiscovery {
    pub fn new(dim: usize) -> Self {
        Self {
            config: DiscoveryConfig::default(),
            encoder: CodeHDEncoder::new(dim),
            decoder: HvCodeDecoder::new(dim),
        }
    }

    pub fn with_config(dim: usize, config: DiscoveryConfig) -> Self {
        Self {
            config,
            encoder: CodeHDEncoder::new(dim),
            decoder: HvCodeDecoder::new(dim),
        }
    }

    /// Get mutable reference to decoder (for registering discoveries)
    pub fn decoder_mut(&mut self) -> &mut HvCodeDecoder {
        &mut self.decoder
    }

    /// Attempt to discover a solution for a problem through evolution.
    ///
    /// This is the core emergence loop:
    /// 1. Seed population from code fragments
    /// 2. Evaluate fitness (compile + test)
    /// 3. Select, crossover, mutate
    /// 4. Repeat until solution found or max generations
    ///
    /// The `test_source` parameter contains the test assertions that serve
    /// as the fitness function. If the generated code passes all tests,
    /// fitness = 1.0.
    pub fn discover(
        &mut self,
        purpose: &str,
        signature: &str,
        param_name: &str,
        param_type: &str,
        return_type: &str,
        test_source: &str,
        executor: &mut CodeExecutor,
    ) -> DiscoveryResult {
        if self.config.use_compilation_fitness && !executor.supports_real_execution() {
            return DiscoveryResult {
                found: false,
                source: None,
                simulated_execution: true,
                best_fitness: 0.0,
                generations: 0,
                compiled_count: 0,
                total_evaluated: 0,
            };
        }

        // Encode the intent
        let intent_hv = self.encoder.encode_name(purpose);

        // Seed population from fragment compositions
        let mut population = self.seed_population(&intent_hv, param_type, return_type);

        let mut best_fitness = 0.0f32;
        let mut best_code: Option<String> = None;
        let mut compiled_count = 0usize;
        let mut total_evaluated = 0usize;

        for gen in 0..self.config.max_generations {
            // Decode and evaluate each candidate
            for candidate in &mut population {
                if candidate.code.is_none() {
                    // Decode HV to code
                    let code = self.decode_candidate(
                        &candidate.hv,
                        signature,
                        param_name,
                        param_type,
                        return_type,
                        purpose,
                    );
                    candidate.code = Some(code);
                }

                let code = candidate.code.as_ref().unwrap();
                total_evaluated += 1;

                // Evaluate fitness
                if self.config.use_compilation_fitness && !test_source.is_empty() {
                    let full_source = format!("{}\n\n{}", code, test_source);
                    let mut result = executor.execute_rust_with_inline_tests(&full_source);
                    result.parse_test_failures();

                    if result.simulated {
                        return DiscoveryResult {
                            found: false,
                            source: None,
                            simulated_execution: true,
                            best_fitness: 0.0,
                            generations: gen + 1,
                            compiled_count,
                            total_evaluated,
                        };
                    }

                    candidate.fitness = compute_rich_fitness(&result);
                    if result.compiled {
                        compiled_count += 1;
                    }
                } else {
                    // Fast fitness: similarity to intent (no compilation)
                    candidate.fitness = candidate.hv.similarity(&intent_hv).max(0.0);
                }

                // Track best
                if candidate.fitness > best_fitness {
                    best_fitness = candidate.fitness;
                    best_code = candidate.code.clone();
                }

                // Early exit if perfect solution found
                if candidate.fitness >= 1.0 {
                    // Register discovery in decoder for future use
                    if let Some(ref code) = candidate.code {
                        self.decoder.register_discovery(
                            &format!(
                                "discovered_{}",
                                purpose.chars().take(20).collect::<String>()
                            ),
                            code,
                        );
                    }

                    return DiscoveryResult {
                        found: true,
                        source: candidate.code.clone(),
                        simulated_execution: false,
                        best_fitness: 1.0,
                        generations: gen + 1,
                        compiled_count,
                        total_evaluated,
                    };
                }
            }

            // Select + crossover + mutate for next generation
            population = self.evolve_generation(population);
        }

        // Return best found (even if imperfect)
        DiscoveryResult {
            found: best_fitness >= self.config.discovery_threshold,
            source: if best_fitness >= self.config.discovery_threshold {
                best_code
            } else {
                None
            },
            simulated_execution: false,
            best_fitness,
            generations: self.config.max_generations,
            compiled_count,
            total_evaluated,
        }
    }

    /// Seed the initial population from fragment compositions.
    fn seed_population(
        &self,
        intent_hv: &ContinuousHV,
        param_type: &str,
        return_type: &str,
    ) -> Vec<Candidate> {
        let mut pop = Vec::with_capacity(self.config.population_size);

        // Seed 1: The intent HV itself
        pop.push(Candidate {
            hv: intent_hv.clone(),
            code: None,
            fitness: 0.0,
        });

        // Seed 2-5: Nearest known fragments
        let nearest = self.decoder.decode_nearest(intent_hv, 4);
        for (_, frag) in &nearest {
            pop.push(Candidate {
                hv: frag.hv.clone(),
                code: None,
                fitness: 0.0,
            });
        }

        // Seed 6+: Random compositions of fragments
        let sources = self
            .decoder
            .top_in_category(intent_hv, FragmentCategory::IteratorSource, 3);
        let adapters =
            self.decoder
                .top_in_category(intent_hv, FragmentCategory::IteratorAdapter, 5);
        let terminals =
            self.decoder
                .top_in_category(intent_hv, FragmentCategory::IteratorTerminal, 3);

        // Compose: each source × each adapter × each terminal
        for (_, src) in &sources {
            for (_, adp) in &adapters {
                for (_, trm) in &terminals {
                    if pop.len() >= self.config.population_size {
                        break;
                    }
                    // Bundle the three fragments into a composite HV
                    let composite = ContinuousHV::bundle(&[&src.hv, &adp.hv, &trm.hv]);
                    pop.push(Candidate {
                        hv: composite,
                        code: None,
                        fitness: 0.0,
                    });
                }
            }
        }

        // Fill remaining with random perturbations of intent
        while pop.len() < self.config.population_size {
            let seed = pop.len() as u64;
            let perturbed = intent_hv
                .add(&ContinuousHV::random(intent_hv.dim(), seed).scale(0.2))
                .normalize();
            pop.push(Candidate {
                hv: perturbed,
                code: None,
                fitness: 0.0,
            });
        }

        pop
    }

    /// Decode a candidate HV into executable Rust code.
    fn decode_candidate(
        &self,
        hv: &ContinuousHV,
        signature: &str,
        param_name: &str,
        param_type: &str,
        return_type: &str,
        purpose: &str,
    ) -> String {
        // Try iterator chain decoding first
        if let Some(chain) = self
            .decoder
            .decode_iterator_chain(hv, param_type, return_type)
        {
            // Replace generic "input" with actual param name
            let body = chain.replace("input", param_name).replace("v", param_name);
            return format!("{} {{\n    {}\n}}", signature, body);
        }

        // Try nearest complete body
        let nearest = self.decoder.decode_nearest(hv, 1);
        if let Some((sim, frag)) = nearest.first() {
            if *sim > 0.5 && frag.category == FragmentCategory::CompleteBody {
                return format!("{} {{\n    {}\n}}", signature, frag.code);
            }
        }

        // Fallback: compose from nearest fragments
        let top3 = self.decoder.decode_nearest(hv, 3);
        let body_parts: Vec<&str> = top3.iter().map(|(_, f)| f.code.as_str()).collect();
        let body = if body_parts.is_empty() {
            format!("todo!(\"Discover: {}\")", purpose)
        } else {
            body_parts.join("")
        };

        format!("{} {{\n    {}\n}}", signature, body)
    }

    /// Evolve population: select, crossover, mutate.
    fn evolve_generation(&self, mut population: Vec<Candidate>) -> Vec<Candidate> {
        // Sort by fitness descending
        population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut next_gen = Vec::with_capacity(self.config.population_size);

        // Elitism: keep top 2
        for candidate in population.iter().take(2) {
            next_gen.push(Candidate {
                hv: candidate.hv.clone(),
                code: None, // Force re-decode (might mutate)
                fitness: 0.0,
            });
        }

        // Fill rest with crossover + mutation
        let mut seed = 42u64;
        while next_gen.len() < self.config.population_size {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

            // Tournament selection: pick 2 random parents from top half
            let half = population.len() / 2;
            let p1_idx = (seed as usize) % half.max(1);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let p2_idx = (seed as usize) % half.max(1);

            let parent1 = &population[p1_idx];
            let parent2 = &population[p2_idx];

            // Crossover
            let frac = (seed >> 33) as f32 / u32::MAX as f32;
            let child_hv = if frac < self.config.crossover_rate {
                // Blend parents
                let alpha = 0.3 + frac * 0.4; // [0.3, 0.7]
                                              // Weighted blend: alpha * parent1 + (1-alpha) * parent2
                parent1
                    .hv
                    .scale(alpha)
                    .add(&parent2.hv.scale(1.0 - alpha))
                    .normalize()
            } else {
                parent1.hv.clone()
            };

            // Mutation
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut_frac = (seed >> 33) as f32 / u32::MAX as f32;
            let child_hv = if mut_frac < self.config.mutation_rate {
                // Perturb: add small random noise
                child_hv
                    .add(&ContinuousHV::random(child_hv.dim(), seed).scale(0.15))
                    .normalize()
            } else {
                child_hv
            };

            next_gen.push(Candidate {
                hv: child_hv,
                code: None,
                fitness: 0.0,
            });
        }

        next_gen
    }
}

/// Compute rich fitness from compilation + test results.
///
/// Instead of binary pass/fail, this captures the GRADIENT of correctness:
///
/// 0.00 = syntax error (can't even parse)
/// 0.05 = many compilation errors (far from compiling)
/// 0.15 = few compilation errors (close to compiling — fixable!)
/// 0.25 = compiles but no tests run
/// 0.30-0.95 = partial test passes (gradient of correctness)
/// 1.00 = all tests pass (perfect fitness)
///
/// This gradient turns evolution from random search into hill climbing.
/// A candidate passing 5/10 tests WILL be preferred over one passing 0/10,
/// driving evolution toward the solution.
fn compute_rich_fitness(result: &ExecutionResult) -> f32 {
    if !result.compiled {
        // Didn't compile — but HOW FAR from compiling?
        let error_count = result.compile_errors.len();

        if error_count == 0 {
            return 0.10; // Unknown failure
        }

        // Fewer errors = closer to compiling = higher fitness
        // 1 error → 0.15, 2 errors → 0.12, 5+ errors → 0.05
        let compilation_closeness = 0.05 + 0.10 / (error_count as f32);

        // Bonus for fixable errors (type mismatches are easier than missing functions)
        let fixable_count = result
            .compile_errors
            .iter()
            .filter(|e| {
                e.contains("E0308") // type mismatch
                || e.contains("E0277") // trait bound
                || e.contains("E0106") // lifetime
                || e.contains("mismatched types")
                || e.contains("expected")
            })
            .count();

        let fixability_bonus = if error_count > 0 {
            (fixable_count as f32 / error_count as f32) * 0.05
        } else {
            0.0
        };

        return (compilation_closeness + fixability_bonus).min(0.20);
    }

    // Compiled! Now check tests.
    let total_tests = result.tests_passed + result.tests_failed;

    if total_tests == 0 {
        return 0.25; // Compiled but no tests ran
    }

    if result.tests_failed == 0 {
        return 1.0; // Perfect — all tests pass
    }

    // Partial test passing: 0.30 base + 0.65 * pass_rate
    let pass_rate = result.tests_passed as f32 / total_tests as f32;
    let test_fitness = 0.30 + pass_rate * 0.65;

    // Bonus for test failures that are CLOSE to correct
    // (e.g., expected=6, actual=5 is better than expected=6, actual=42)
    let closeness_bonus = if !result.test_failures.is_empty() {
        let close_failures = result
            .test_failures
            .iter()
            .filter(|f| {
                // Check if expected and actual are numerically close
                if let (Some(exp), Some(act)) = (&f.expected, &f.actual) {
                    if let (Ok(e), Ok(a)) = (exp.parse::<f64>(), act.parse::<f64>()) {
                        let distance = (e - a).abs();
                        let scale = e.abs().max(1.0);
                        return distance / scale < 0.5; // Within 50% of expected
                    }
                }
                false
            })
            .count();
        (close_failures as f32 / result.test_failures.len().max(1) as f32) * 0.05
    } else {
        0.0
    };

    (test_fitness + closeness_bonus).min(0.99) // Reserve 1.0 for perfect
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();
        assert_eq!(config.population_size, 30);
        assert_eq!(config.max_generations, 20);
    }

    #[test]
    fn test_discover_rejects_simulated_execution() {
        let mut discovery = CodeDiscovery::new(512);
        let mut executor = CodeExecutor::new();

        let result = discovery.discover(
            "sum numbers",
            "fn sum_numbers(input: &[i32]) -> i32",
            "input",
            "&[i32]",
            "i32",
            "#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn smoke() { assert_eq!(sum_numbers(&[1,2,3]), 6); }\n}",
            &mut executor,
        );

        assert!(!result.found);
        assert!(result.simulated_execution);
        assert_eq!(result.best_fitness, 0.0);
        assert_eq!(result.total_evaluated, 0);
    }

    #[test]
    fn test_seed_population() {
        let discovery = CodeDiscovery::new(512);
        let intent = ContinuousHV::random(512, 42);
        let pop = discovery.seed_population(&intent, "&[i32]", "Vec<i32>");
        assert_eq!(pop.len(), discovery.config.population_size);
    }

    #[test]
    fn test_decode_candidate_produces_code() {
        let discovery = CodeDiscovery::new(512);
        let intent = ContinuousHV::random(512, 42);
        let code = discovery.decode_candidate(
            &intent,
            "pub fn example(v: &[i32]) -> Vec<i32>",
            "v",
            "&[i32]",
            "Vec<i32>",
            "filter and collect",
        );
        assert!(code.contains("pub fn example"));
        assert!(code.contains("{"));
        assert!(code.contains("}"));
    }

    #[test]
    fn test_evolve_generation_preserves_size() {
        let discovery = CodeDiscovery::new(512);
        let intent = ContinuousHV::random(512, 42);
        let pop = discovery.seed_population(&intent, "&[i32]", "Vec<i32>");
        let size = pop.len();
        let evolved = discovery.evolve_generation(pop);
        assert_eq!(evolved.len(), size);
    }
}
