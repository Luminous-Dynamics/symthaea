// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortingCompetencyMetrics {
    pub sortedness: f32,
    pub convergence_steps: usize,
    pub swap_count: usize,
    pub regressions: usize,
    pub recovery_after_damage: f32,
    pub mean_local_surprise: f32,
    pub mean_global_surprise: f32,
    pub coherence: f32,
    pub policy_diversity: f32,
}

impl Default for SortingCompetencyMetrics {
    fn default() -> Self {
        Self {
            sortedness: 0.0,
            convergence_steps: 0,
            swap_count: 0,
            regressions: 0,
            recovery_after_damage: 0.0,
            mean_local_surprise: 0.0,
            mean_global_surprise: 0.0,
            coherence: 0.0,
            policy_diversity: 0.0,
        }
    }
}

impl SortingCompetencyMetrics {
    pub fn fitness(&self) -> f32 {
        let sortedness = self.sortedness;
        let speed = 1.0 / (1.0 + self.convergence_steps as f32);
        let efficiency = 1.0 / (1.0 + self.swap_count as f32);
        let recovery = self.recovery_after_damage;
        let coherence = self.coherence;

        0.45 * sortedness + 0.15 * speed + 0.15 * efficiency + 0.15 * recovery + 0.10 * coherence
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortingMode {
    ClassicalController,
    CellView,
    SymthaeaMorpho,
}

#[derive(Debug, Clone)]
pub struct MorphoCell {
    pub value: f32,
    pub position: usize,
    pub damage: f32,
    pub energy: f32,
    pub policy: CellPolicy,
    pub surprise: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PolicyActivation {
    Linear,
    Tanh,
    Sigmoid,
    Step,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterizedPolicy {
    pub weights: [f32; 8],
    pub activation: PolicyActivation,
    pub swap_threshold: f32,
    pub energy_cost: f32,
    pub surprise_sensitivity: f32,
}

impl Default for ParameterizedPolicy {
    fn default() -> Self {
        Self {
            weights: [0.5, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
            activation: PolicyActivation::Sigmoid,
            swap_threshold: 0.5,
            energy_cost: 0.1,
            surprise_sensitivity: 0.1,
        }
    }
}

impl ParameterizedPolicy {
    pub fn decide_swap(
        &self,
        left_value: f32,
        right_value: f32,
        local_surprise: f32,
        energy: f32,
        damage: f32,
    ) -> bool {
        let inversion = left_value - right_value;
        let score = self.weights[0] * inversion
            + self.weights[1] * local_surprise
            + self.weights[2] * energy
            - self.weights[3] * damage;

        self.activate(score) > self.swap_threshold
    }

    fn activate(&self, x: f32) -> f32 {
        match self.activation {
            PolicyActivation::Linear => x,
            PolicyActivation::Tanh => x.tanh(),
            PolicyActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            PolicyActivation::Step => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    pub fn mutate(&mut self, seed: u64, magnitude: f32) {
        for (i, weight) in self.weights.iter_mut().enumerate() {
            *weight += ((seed + i as u64) % 100) as f32 / 100.0 * magnitude - (magnitude / 2.0);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellPolicy {
    Greedy,
    Stochastic,
    HdcPredictive,
    Parameterized(ParameterizedPolicy),
}

impl CellPolicy {
    pub fn decide_swap(
        &self,
        left_value: f32,
        right_value: f32,
        local_surprise: f32,
        energy: f32,
        damage: f32,
    ) -> bool {
        match self {
            CellPolicy::Greedy => left_value > right_value,
            CellPolicy::Stochastic => left_value > right_value,
            CellPolicy::HdcPredictive => false,
            CellPolicy::Parameterized(p) => {
                p.decide_swap(left_value, right_value, local_surprise, energy, damage)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MorphogeneticSortingSandbox {
    pub mode: SortingMode,
    pub cells: Vec<MorphoCell>,
    pub metrics: SortingCompetencyMetrics,
    pub step: usize,
}

impl MorphogeneticSortingSandbox {
    pub fn from_values(mode: SortingMode, values: Vec<f32>) -> Self {
        let cells = values
            .into_iter()
            .enumerate()
            .map(|(position, value)| MorphoCell {
                value,
                position,
                damage: 0.0,
                energy: 1.0,
                policy: CellPolicy::Greedy,
                surprise: 0.0,
            })
            .collect();

        let mut sandbox = Self {
            mode,
            cells,
            metrics: SortingCompetencyMetrics::default(),
            step: 0,
        };

        sandbox.update_metrics();
        sandbox
    }

    pub fn update_metrics(&mut self) {
        self.metrics.sortedness = self.calculate_sortedness();
        self.metrics.convergence_steps = self.step;
    }

    pub fn refresh_positions(&mut self) {
        for (idx, cell) in self.cells.iter_mut().enumerate() {
            cell.position = idx;
        }
    }

    pub fn step_once(&mut self) -> bool {
        match self.mode {
            SortingMode::ClassicalController => self.step_classical_controller(),
            SortingMode::CellView => self.step_cell_view(),
            SortingMode::SymthaeaMorpho => self.step_symthaea_morpho(),
        }
    }

    fn step_classical_controller(&mut self) -> bool {
        let mut swapped = false;
        for i in 0..self.cells.len().saturating_sub(1) {
            if self.cells[i].value > self.cells[i + 1].value {
                self.cells.swap(i, i + 1);
                self.metrics.swap_count += 1;
                swapped = true;
            }
        }
        self.refresh_positions();
        self.step += 1;
        swapped
    }

    fn step_cell_view(&mut self) -> bool {
        let mut swapped = false;
        if self.cells.len() < 2 {
            self.step += 1;
            return false;
        }

        for i in 0..self.cells.len() - 1 {
            if self.cells[i].damage >= 1.0 || self.cells[i + 1].damage >= 1.0 {
                continue;
            }

            if self.cells[i].energy <= 0.0 || self.cells[i + 1].energy <= 0.0 {
                continue;
            }

            let should_swap = self.cells[i].policy.decide_swap(
                self.cells[i].value,
                self.cells[i + 1].value,
                self.cells[i].surprise,
                self.cells[i].energy,
                self.cells[i].damage,
            );

            if should_swap {
                self.cells.swap(i, i + 1);
                self.metrics.swap_count += 1;
                self.cells[i].energy = (self.cells[i].energy - 0.01).max(0.0);
                self.cells[i + 1].energy = (self.cells[i + 1].energy - 0.01).max(0.0);
                swapped = true;
            }
        }
        self.refresh_positions();
        self.step += 1;
        swapped
    }

    fn step_symthaea_morpho(&mut self) -> bool {
        let mut swapped = false;
        let mut total_surprise = 0.0;
        let pair_count = self.cells.len().saturating_sub(1).max(1) as f32;

        if self.cells.len() < 2 {
            self.step += 1;
            return false;
        }

        for i in 0..self.cells.len() - 1 {
            if self.cells[i].damage >= 1.0 || self.cells[i + 1].damage >= 1.0 {
                continue;
            }

            if self.cells[i].energy <= 0.0 || self.cells[i + 1].energy <= 0.0 {
                continue;
            }

            let inversion_error = (self.cells[i].value - self.cells[i + 1].value).max(0.0);
            self.cells[i].surprise = 0.8 * self.cells[i].surprise + 0.2 * inversion_error;
            total_surprise += self.cells[i].surprise;

            let should_swap = self.cells[i].policy.decide_swap(
                self.cells[i].value,
                self.cells[i + 1].value,
                self.cells[i].surprise,
                self.cells[i].energy,
                self.cells[i].damage,
            );

            if should_swap {
                self.cells.swap(i, i + 1);
                self.metrics.swap_count += 1;
                self.cells[i].energy = (self.cells[i].energy - 0.01).max(0.0);
                self.cells[i + 1].energy = (self.cells[i + 1].energy - 0.01).max(0.0);
                swapped = true;
            }
        }

        self.metrics.mean_local_surprise = total_surprise / pair_count;
        self.metrics.mean_global_surprise = self.metrics.mean_local_surprise;

        let sortedness = self.calculate_sortedness();
        self.metrics.coherence =
            sortedness * (1.0 - self.metrics.mean_local_surprise.clamp(0.0, 1.0));

        self.refresh_positions();
        self.step += 1;
        swapped
    }

    pub fn run_until_converged(&mut self, max_steps: usize) -> SortingCompetencyMetrics {
        for _ in 0..max_steps {
            let changed = self.step_once();
            self.update_metrics();

            if !changed && self.metrics.sortedness >= 1.0 {
                break;
            }
        }
        self.metrics.clone()
    }

    pub fn calculate_sortedness(&self) -> f32 {
        if self.cells.len() < 2 {
            return 1.0;
        }

        let ordered_pairs = self
            .cells
            .windows(2)
            .filter(|pair| pair[0].value <= pair[1].value)
            .count();

        ordered_pairs as f32 / (self.cells.len() - 1) as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEvaluation {
    pub policy: CellPolicy,
    pub metrics: SortingCompetencyMetrics,
    pub fitness: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEvaluationSummary {
    pub policy: CellPolicy,
    pub evaluations: Vec<PolicyEvaluation>,
    pub mean_fitness: f32,
    pub min_fitness: f32,
    pub mean_sortedness: f32,
    pub mean_steps: f32,
    pub mean_swaps: f32,
    pub failure_count: usize,
}

pub struct DiscoveryHarness;

impl DiscoveryHarness {
    pub fn evaluate_policies(
        scenario: &[f32],
        policies: &[CellPolicy],
        max_steps: usize,
    ) -> Vec<PolicyEvaluation> {
        let mut results = Vec::new();

        for policy in policies {
            let mut sandbox = MorphogeneticSortingSandbox::from_values(
                SortingMode::SymthaeaMorpho,
                scenario.to_vec(),
            );

            for cell in &mut sandbox.cells {
                cell.policy = policy.clone();
            }

            let metrics = sandbox.run_until_converged(max_steps);
            let fitness = metrics.fitness();

            results.push(PolicyEvaluation {
                policy: policy.clone(),
                metrics,
                fitness,
            });
        }

        results.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    pub fn evaluate_policy_on_scenarios(
        scenarios: &[Vec<f32>],
        policy: &CellPolicy,
        max_steps: usize,
    ) -> PolicyEvaluationSummary {
        assert!(!scenarios.is_empty(), "scenarios must not be empty");
        let mut evaluations = Vec::new();
        let mut total_fitness = 0.0;
        let mut min_fitness = f32::MAX;
        let mut total_sortedness = 0.0;
        let mut total_steps = 0.0;
        let mut total_swaps = 0.0;
        let mut failure_count = 0;

        for scenario in scenarios {
            let mut sandbox = MorphogeneticSortingSandbox::from_values(
                SortingMode::SymthaeaMorpho,
                scenario.to_vec(),
            );
            for cell in &mut sandbox.cells {
                cell.policy = policy.clone();
            }

            let metrics = sandbox.run_until_converged(max_steps);
            let fitness = metrics.fitness();

            if metrics.sortedness < 1.0 {
                failure_count += 1;
            }

            total_fitness += fitness;
            min_fitness = min_fitness.min(fitness);
            total_sortedness += metrics.sortedness;
            total_steps += metrics.convergence_steps as f32;
            total_swaps += metrics.swap_count as f32;

            evaluations.push(PolicyEvaluation {
                policy: policy.clone(),
                metrics,
                fitness,
            });
        }

        let count = scenarios.len() as f32;
        PolicyEvaluationSummary {
            policy: policy.clone(),
            evaluations,
            mean_fitness: total_fitness / count,
            min_fitness,
            mean_sortedness: total_sortedness / count,
            mean_steps: total_steps / count,
            mean_swaps: total_swaps / count,
            failure_count,
        }
    }

    pub fn rank_policies_across_scenarios(
        scenarios: &[Vec<f32>],
        policies: &[CellPolicy],
        max_steps: usize,
    ) -> Vec<PolicyEvaluationSummary> {
        assert!(!scenarios.is_empty(), "scenarios must not be empty");
        let mut results: Vec<PolicyEvaluationSummary> = policies
            .iter()
            .map(|p| Self::evaluate_policy_on_scenarios(scenarios, p, max_steps))
            .collect();

        results.sort_by(|a, b| {
            b.mean_fitness
                .partial_cmp(&a.mean_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    b.min_fitness
                        .partial_cmp(&a.min_fitness)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        results
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyMetadata {
    pub seed: u64,
    pub mutation_magnitude: f32,
    pub generation: usize,
    pub timestamp: u64, // Epoch time
    pub run_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDiscoveryRecord {
    pub policy: CellPolicy,
    pub training_summary: PolicyEvaluationSummary,
    pub validation_summary: Option<PolicyEvaluationSummary>,
    pub metadata: PolicyMetadata,
    pub notes: String,
}

impl PolicyDiscoveryRecord {
    pub fn new(
        policy: CellPolicy,
        training_summary: PolicyEvaluationSummary,
        validation_summary: Option<PolicyEvaluationSummary>,
        metadata: PolicyMetadata,
    ) -> Self {
        Self {
            policy,
            training_summary,
            validation_summary,
            metadata,
            notes: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SingleGenerationEvolutionResult {
    pub parent_summary: PolicyEvaluationSummary,
    pub all_candidate_summaries: Vec<PolicyEvaluationSummary>,
    pub elite_summary: PolicyEvaluationSummary,
    pub validation_summary: PolicyEvaluationSummary,
}

pub struct EvolutionHarness;

impl EvolutionHarness {
    pub fn run_single_generation(
        parent: ParameterizedPolicy,
        population_size: usize,
        mutation_magnitude: f32,
        training_scenarios: &[Vec<f32>],
        validation_scenarios: &[Vec<f32>],
        max_steps: usize,
        seed: u64,
    ) -> SingleGenerationEvolutionResult {
        let mut candidates = vec![CellPolicy::Parameterized(parent.clone())];
        for i in 0..population_size {
            let mut mutant = parent.clone();
            mutant.mutate(seed + i as u64, mutation_magnitude);
            candidates.push(CellPolicy::Parameterized(mutant));
        }

        let ranked = DiscoveryHarness::rank_policies_across_scenarios(
            training_scenarios,
            &candidates,
            max_steps,
        );

        let elite = ranked[0].clone();

        let parent_summary = DiscoveryHarness::evaluate_policy_on_scenarios(
            training_scenarios,
            &CellPolicy::Parameterized(parent),
            max_steps,
        );

        let validation_summary = DiscoveryHarness::evaluate_policy_on_scenarios(
            validation_scenarios,
            &elite.policy,
            max_steps,
        );

        SingleGenerationEvolutionResult {
            parent_summary,
            all_candidate_summaries: ranked,
            elite_summary: elite,
            validation_summary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sortedness_logic() {
        let sorted = MorphogeneticSortingSandbox::from_values(
            SortingMode::ClassicalController,
            vec![1.0, 2.0, 3.0],
        );
        assert_eq!(sorted.metrics.sortedness, 1.0);

        let reversed = MorphogeneticSortingSandbox::from_values(
            SortingMode::ClassicalController,
            vec![3.0, 2.0, 1.0],
        );
        assert!(reversed.metrics.sortedness < 1.0);
    }

    #[test]
    fn test_classical_controller_convergence() {
        let mut sandbox = MorphogeneticSortingSandbox::from_values(
            SortingMode::ClassicalController,
            vec![3.0, 2.0, 1.0],
        );
        sandbox.run_until_converged(10);
        assert_eq!(sandbox.metrics.sortedness, 1.0);
        assert!(sandbox.metrics.convergence_steps > 0);
    }

    #[test]
    fn test_classical_controller_no_change_if_sorted() {
        let mut sandbox = MorphogeneticSortingSandbox::from_values(
            SortingMode::ClassicalController,
            vec![1.0, 2.0, 3.0],
        );
        sandbox.run_until_converged(10);
        assert_eq!(sandbox.metrics.swap_count, 0);
    }

    #[test]
    fn test_classical_controller_swaps_reversed() {
        let mut sandbox = MorphogeneticSortingSandbox::from_values(
            SortingMode::ClassicalController,
            vec![3.0, 2.0, 1.0],
        );
        sandbox.run_until_converged(10);
        assert!(sandbox.metrics.swap_count > 0);
    }

    #[test]
    fn test_cell_view_converges_undamaged() {
        let mut sandbox =
            MorphogeneticSortingSandbox::from_values(SortingMode::CellView, vec![3.0, 2.0, 1.0]);
        sandbox.run_until_converged(100);
        assert_eq!(sandbox.metrics.sortedness, 1.0);
    }

    #[test]
    fn test_cell_view_damage_blocks_swap() {
        let mut sandbox =
            MorphogeneticSortingSandbox::from_values(SortingMode::CellView, vec![2.0, 1.0]);
        sandbox.cells[0].damage = 1.0;
        sandbox.run_until_converged(10);
        assert!(sandbox.metrics.swap_count == 0);
    }

    #[test]
    fn test_cell_view_consumes_energy() {
        let mut sandbox =
            MorphogeneticSortingSandbox::from_values(SortingMode::CellView, vec![2.0, 1.0]);
        sandbox.step_once();
        assert!(sandbox.cells[0].energy < 1.0);
    }

    #[test]
    fn test_symthaea_morpho_converges() {
        let mut sandbox = MorphogeneticSortingSandbox::from_values(
            SortingMode::SymthaeaMorpho,
            vec![3.0, 2.0, 1.0],
        );
        sandbox.run_until_converged(100);
        assert_eq!(sandbox.metrics.sortedness, 1.0);
    }

    #[test]
    fn test_parameterized_policy_mutation() {
        let mut policy = ParameterizedPolicy::default();
        let old_weights = policy.weights;
        policy.mutate(123, 0.1);
        assert_ne!(old_weights, policy.weights);
    }

    #[test]
    fn test_symthaea_morpho_surprise_and_coherence() {
        let mut sandbox =
            MorphogeneticSortingSandbox::from_values(SortingMode::SymthaeaMorpho, vec![2.0, 1.0]);
        sandbox.step_once();
        assert!(sandbox.metrics.mean_local_surprise >= 0.0);
        assert!(sandbox.metrics.mean_global_surprise >= 0.0);
        assert!(sandbox.metrics.coherence >= 0.0);
        assert!(sandbox.metrics.coherence.is_finite());
    }

    #[test]
    fn test_policy_discovery_record_creation() {
        let parent = ParameterizedPolicy::default();
        let scenario = vec![vec![3.0, 2.0, 1.0]];
        let summary = DiscoveryHarness::evaluate_policy_on_scenarios(
            &scenario,
            &CellPolicy::Parameterized(parent.clone()),
            20,
        );

        let metadata = PolicyMetadata {
            seed: 123,
            mutation_magnitude: 0.1,
            generation: 1,
            timestamp: 1600000000,
            run_id: "test-run-001".to_string(),
        };

        let record = PolicyDiscoveryRecord::new(
            CellPolicy::Parameterized(parent),
            summary.clone(),
            Some(summary),
            metadata,
        );

        assert_eq!(record.metadata.run_id, "test-run-001");
        assert!(record.validation_summary.is_some());
    }

    #[test]
    fn test_symthaea_morpho_positions_consistent() {
        let mut sandbox = MorphogeneticSortingSandbox::from_values(
            SortingMode::SymthaeaMorpho,
            vec![3.0, 2.0, 1.0],
        );
        sandbox.step_once();

        for (idx, cell) in sandbox.cells.iter().enumerate() {
            assert_eq!(cell.position, idx);
        }
    }

    #[test]
    fn test_cell_policy_decide_swap_greedy() {
        let policy = CellPolicy::Greedy;
        assert!(policy.decide_swap(1.0, 0.0, 0.0, 1.0, 0.0));
        assert!(!policy.decide_swap(0.0, 1.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn test_fitness_increases_with_sortedness() {
        let low = SortingCompetencyMetrics {
            sortedness: 0.1,
            ..Default::default()
        };
        let high = SortingCompetencyMetrics {
            sortedness: 0.9,
            ..Default::default()
        };

        assert!(high.fitness() > low.fitness());
    }

    #[test]
    fn test_evaluate_policy_on_multiple_scenarios() {
        let scenarios = vec![vec![3.0, 2.0, 1.0], vec![1.0, 2.0, 3.0]];
        let policy = CellPolicy::Parameterized(ParameterizedPolicy::default());

        let summary = DiscoveryHarness::evaluate_policy_on_scenarios(&scenarios, &policy, 20);

        assert_eq!(summary.evaluations.len(), 2);
        assert!(summary.mean_fitness > 0.0);
    }

    #[test]
    fn test_multi_scenario_ranking_is_deterministic() {
        let scenarios = vec![vec![3.0, 2.0, 1.0], vec![2.0, 1.0]];
        let policies = vec![
            CellPolicy::Greedy,
            CellPolicy::Parameterized(ParameterizedPolicy::default()),
        ];

        let results1 = DiscoveryHarness::rank_policies_across_scenarios(&scenarios, &policies, 20);
        let results2 = DiscoveryHarness::rank_policies_across_scenarios(&scenarios, &policies, 20);

        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.mean_fitness, r2.mean_fitness);
        }
    }

    #[test]
    fn test_single_generation_returns_expected_candidate_count() {
        let parent = ParameterizedPolicy::default();
        let scenarios = vec![vec![3.0, 2.0, 1.0]];
        let population_size = 5;
        let result = EvolutionHarness::run_single_generation(
            parent,
            population_size,
            0.1,
            &scenarios,
            &scenarios,
            20,
            123,
        );
        assert_eq!(result.all_candidate_summaries.len(), population_size + 1);
    }

    #[test]
    fn test_single_generation_is_deterministic_for_same_seed() {
        let parent = ParameterizedPolicy::default();
        let scenarios = vec![vec![3.0, 2.0, 1.0]];
        let result1 = EvolutionHarness::run_single_generation(
            parent.clone(),
            5,
            0.1,
            &scenarios,
            &scenarios,
            20,
            123,
        );
        let result2 = EvolutionHarness::run_single_generation(
            parent, 5, 0.1, &scenarios, &scenarios, 20, 123,
        );
        assert_eq!(
            result1.elite_summary.mean_fitness,
            result2.elite_summary.mean_fitness
        );
    }

    #[test]
    fn test_discovery_harness_ranks_policies_single_scenario() {
        let scenario = vec![3.0, 2.0, 1.0];
        let policies = vec![
            CellPolicy::Greedy,
            CellPolicy::Parameterized(ParameterizedPolicy::default()),
        ];

        let results = DiscoveryHarness::evaluate_policies(&scenario, &policies, 20);

        assert_eq!(results.len(), 2);
        assert!(results[0].fitness >= results[1].fitness);
    }

    #[test]
    fn test_single_generation_computes_validation_summary() {
        let parent = ParameterizedPolicy::default();
        let training = vec![vec![3.0, 2.0, 1.0]];
        let validation = vec![vec![5.0, 4.0, 3.0, 2.0, 1.0]];

        let result = EvolutionHarness::run_single_generation(
            parent,
            5,
            0.1,
            &training,
            &validation,
            20,
            123,
        );

        assert_eq!(
            result.validation_summary.evaluations.len(),
            validation.len()
        );
        assert!(result.validation_summary.mean_fitness.is_finite());
    }

    #[test]
    fn test_elite_is_at_index_zero_in_all_candidates() {
        let parent = ParameterizedPolicy::default();
        let scenarios = vec![vec![3.0, 2.0, 1.0]];
        let result = EvolutionHarness::run_single_generation(
            parent, 5, 0.1, &scenarios, &scenarios, 20, 123,
        );
        assert_eq!(
            result.all_candidate_summaries[0].fitness,
            result.elite_summary.fitness
        );
    }
}
