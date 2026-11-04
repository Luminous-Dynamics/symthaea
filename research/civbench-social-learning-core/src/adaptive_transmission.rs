// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! State-aware cultural transmission over deterministic recipe DAGs.
//!
//! This is a CivBench mechanism benchmark inspired by Colas et al. (2026),
//! "Discovering Adaptive Transmission Programs for Collective Innovation".
//! It is not a reproduction of Collective LittleAlchemy or the authors'
//! LLM-guided evolutionary search. It isolates content-aware routing under
//! bounded attention using a transparent deterministic recipe task.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskFamily {
    Layered,
    Branching,
    Convergent,
}

impl TaskFamily {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Layered => "layered",
            Self::Branching => "branching",
            Self::Convergent => "convergent",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe {
    pub recipe_id: u16,
    pub ingredients: Vec<u16>,
    pub output_element: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskError {
    EmptyBaseSet,
    DuplicateBaseElement { element: u16 },
    EmptyRecipeSet,
    DuplicateRecipeId { recipe_id: u16 },
    DuplicateOutput { element: u16 },
    DuplicateIngredient { recipe_id: u16, ingredient: u16 },
    TooFewIngredients { recipe_id: u16 },
    UnknownIngredient { recipe_id: u16, ingredient: u16 },
    NonTopologicalRecipe { recipe_id: u16, output: u16 },
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBaseSet => write!(f, "task must have at least one base element"),
            Self::DuplicateBaseElement { element } => write!(f, "duplicate base element {element}"),
            Self::EmptyRecipeSet => write!(f, "task must contain at least one recipe"),
            Self::DuplicateRecipeId { recipe_id } => write!(f, "duplicate recipe id {recipe_id}"),
            Self::DuplicateOutput { element } => write!(f, "duplicate or base output element {element}"),
            Self::DuplicateIngredient { recipe_id, ingredient } => {
                write!(f, "recipe {recipe_id} repeats ingredient {ingredient}")
            }
            Self::TooFewIngredients { recipe_id } => write!(f, "recipe {recipe_id} needs at least two ingredients"),
            Self::UnknownIngredient { recipe_id, ingredient } => {
                write!(f, "recipe {recipe_id} references unknown or not-yet-produced ingredient {ingredient}")
            }
            Self::NonTopologicalRecipe { recipe_id, output } => {
                write!(f, "recipe {recipe_id} output {output} must be greater than every ingredient")
            }
        }
    }
}
impl std::error::Error for TaskError {}

/// A deterministic task family. Recipes are sorted topologically by output
/// element; each result can be produced only when all its ingredients are known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InnovationTask {
    pub family: TaskFamily,
    base_elements: BTreeSet<u16>,
    recipes: Vec<Recipe>,
    recipe_index_by_id: BTreeMap<u16, usize>,
    depth_by_element: BTreeMap<u16, u16>,
    descendant_count_by_element: BTreeMap<u16, usize>,
}

impl InnovationTask {
    pub fn new(
        family: TaskFamily,
        base_elements: Vec<u16>,
        mut recipes: Vec<Recipe>,
    ) -> Result<Self, TaskError> {
        if base_elements.is_empty() {
            return Err(TaskError::EmptyBaseSet);
        }
        if recipes.is_empty() {
            return Err(TaskError::EmptyRecipeSet);
        }

        let mut base = BTreeSet::new();
        for element in base_elements {
            if !base.insert(element) {
                return Err(TaskError::DuplicateBaseElement { element });
            }
        }

        let mut recipe_ids = BTreeSet::new();
        for recipe in &recipes {
            if !recipe_ids.insert(recipe.recipe_id) {
                return Err(TaskError::DuplicateRecipeId { recipe_id: recipe.recipe_id });
            }
        }

        recipes.sort_by_key(|recipe| (recipe.output_element, recipe.recipe_id));
        let mut available_elements = base.clone();
        let mut depth_by_element = BTreeMap::new();
        let mut ancestors_by_element: BTreeMap<u16, BTreeSet<u16>> = BTreeMap::new();
        let mut descendant_count_by_element: BTreeMap<u16, usize> = BTreeMap::new();
        for &element in &base {
            depth_by_element.insert(element, 0);
            ancestors_by_element.insert(element, BTreeSet::new());
            descendant_count_by_element.insert(element, 0);
        }

        let mut outputs = BTreeSet::new();
        for recipe in &recipes {
            if recipe.ingredients.len() < 2 {
                return Err(TaskError::TooFewIngredients { recipe_id: recipe.recipe_id });
            }
            let mut unique_ingredients = BTreeSet::new();
            for &ingredient in &recipe.ingredients {
                if !unique_ingredients.insert(ingredient) {
                    return Err(TaskError::DuplicateIngredient {
                        recipe_id: recipe.recipe_id,
                        ingredient,
                    });
                }
                if !available_elements.contains(&ingredient) {
                    return Err(TaskError::UnknownIngredient {
                        recipe_id: recipe.recipe_id,
                        ingredient,
                    });
                }
                if ingredient >= recipe.output_element {
                    return Err(TaskError::NonTopologicalRecipe {
                        recipe_id: recipe.recipe_id,
                        output: recipe.output_element,
                    });
                }
            }
            if !outputs.insert(recipe.output_element) || base.contains(&recipe.output_element) {
                return Err(TaskError::DuplicateOutput { element: recipe.output_element });
            }

            let mut ancestors = BTreeSet::new();
            let mut max_depth = 0u16;
            for &ingredient in &recipe.ingredients {
                ancestors.insert(ingredient);
                if let Some(previous) = ancestors_by_element.get(&ingredient) {
                    ancestors.extend(previous.iter().copied());
                }
                max_depth = max_depth.max(*depth_by_element.get(&ingredient).unwrap_or(&0));
            }
            for ancestor in &ancestors {
                *descendant_count_by_element.entry(*ancestor).or_insert(0) += 1;
            }
            depth_by_element.insert(recipe.output_element, max_depth.saturating_add(1));
            ancestors_by_element.insert(recipe.output_element, ancestors);
            descendant_count_by_element.entry(recipe.output_element).or_insert(0);
            available_elements.insert(recipe.output_element);
        }

        let recipe_index_by_id = recipes.iter().enumerate()
            .map(|(index, recipe)| (recipe.recipe_id, index))
            .collect();

        Ok(Self {
            family,
            base_elements: base,
            recipes,
            recipe_index_by_id,
            depth_by_element,
            descendant_count_by_element,
        })
    }

    pub fn family_task(family: TaskFamily) -> Self {
        let (bases, recipes) = match family {
            TaskFamily::Layered => (
                vec![0, 1, 2, 3, 4, 5],
                vec![
                    recipe(0, &[0, 1], 6),
                    recipe(1, &[1, 2], 7),
                    recipe(2, &[2, 3], 8),
                    recipe(3, &[3, 4], 9),
                    recipe(4, &[4, 5], 10),
                    recipe(5, &[6, 7], 11),
                    recipe(6, &[7, 8], 12),
                    recipe(7, &[8, 9], 13),
                    recipe(8, &[9, 10], 14),
                    recipe(9, &[11, 12], 15),
                    recipe(10, &[12, 13], 16),
                    recipe(11, &[13, 14], 17),
                    recipe(12, &[15, 16], 18),
                    recipe(13, &[16, 17], 19),
                ],
            ),
            TaskFamily::Branching => (
                vec![0, 1, 2, 3, 4, 5],
                vec![
                    recipe(0, &[0, 1], 6),
                    recipe(1, &[1, 2], 7),
                    recipe(2, &[2, 3], 8),
                    recipe(3, &[3, 4], 9),
                    recipe(4, &[4, 5], 10),
                    recipe(5, &[0, 2], 11),
                    recipe(6, &[2, 4], 12),
                    recipe(7, &[6, 11], 13),
                    recipe(8, &[7, 12], 14),
                    recipe(9, &[8, 10], 15),
                    recipe(10, &[9, 12], 16),
                    recipe(11, &[13, 14], 17),
                    recipe(12, &[14, 15], 18),
                    recipe(13, &[16, 17], 19),
                ],
            ),
            TaskFamily::Convergent => (
                vec![0, 1, 2, 3, 4, 5],
                vec![
                    recipe(0, &[0, 1], 6),
                    recipe(1, &[2, 3], 7),
                    recipe(2, &[4, 5], 8),
                    recipe(3, &[6, 7], 9),
                    recipe(4, &[7, 8], 10),
                    recipe(5, &[0, 5], 11),
                    recipe(6, &[9, 10], 12),
                    recipe(7, &[9, 11], 13),
                    recipe(8, &[10, 11], 14),
                    recipe(9, &[12, 13], 15),
                    recipe(10, &[13, 14], 16),
                    recipe(11, &[12, 14], 17),
                    recipe(12, &[15, 16], 18),
                    recipe(13, &[16, 17], 19),
                ],
            ),
        };
        Self::new(family, bases, recipes).expect("built-in recipe task is valid")
    }

    pub fn base_elements(&self) -> &BTreeSet<u16> {
        &self.base_elements
    }

    pub fn recipes(&self) -> &[Recipe] {
        &self.recipes
    }

    pub fn recipe(&self, recipe_id: u16) -> Option<&Recipe> {
        self.recipe_index_by_id.get(&recipe_id).map(|&index| &self.recipes[index])
    }

    pub fn total_recipe_count(&self) -> usize {
        self.recipes.len()
    }

    pub fn total_result_count(&self) -> usize {
        self.recipes.len()
    }

    pub fn depth(&self, element: u16) -> Option<u16> {
        self.depth_by_element.get(&element).copied()
    }

    pub fn descendant_count(&self, element: u16) -> usize {
        self.descendant_count_by_element.get(&element).copied().unwrap_or(0)
    }
}

fn recipe(recipe_id: u16, ingredients: &[u16], output_element: u16) -> Recipe {
    Recipe {
        recipe_id,
        ingredients: ingredients.to_vec(),
        output_element,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentKnowledge {
    pub known_elements: BTreeSet<u16>,
    pub known_recipes: BTreeSet<u16>,
    pub individual_discoveries: usize,
    pub social_recipe_acquisitions: usize,
}

impl AgentKnowledge {
    fn new(task: &InnovationTask) -> Self {
        Self {
            known_elements: task.base_elements.clone(),
            known_recipes: BTreeSet::new(),
            individual_discoveries: 0,
            social_recipe_acquisitions: 0,
        }
    }

    fn apply_closure(&mut self, task: &InnovationTask) -> usize {
        let mut total_added = 0usize;
        loop {
            let mut added_this_pass = 0usize;
            for recipe in task.recipes() {
                if self.known_recipes.contains(&recipe.recipe_id)
                    && !self.known_elements.contains(&recipe.output_element)
                    && recipe.ingredients.iter().all(|ingredient| self.known_elements.contains(ingredient))
                {
                    self.known_elements.insert(recipe.output_element);
                    added_this_pass += 1;
                    total_added += 1;
                }
            }
            if added_this_pass == 0 {
                break;
            }
        }
        total_added
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransmissionPolicy {
    NoSocial,
    FixedRing,
    FixedRingContentShuffled,
    RandomNetworkContent,
    StateAwareFrontier,
    StateAwareRandomContent,
    StateAwareRecipientShuffled,
}

impl TransmissionPolicy {
    pub const fn name(self) -> &'static str {
        match self {
            Self::NoSocial => "no_social",
            Self::FixedRing => "fixed_ring",
            Self::FixedRingContentShuffled => "fixed_ring_content_shuffled",
            Self::RandomNetworkContent => "random_network_content",
            Self::StateAwareFrontier => "state_aware_frontier",
            Self::StateAwareRandomContent => "state_aware_random_content",
            Self::StateAwareRecipientShuffled => "state_aware_recipient_shuffled",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub population_size: usize,
    pub steps: usize,
    pub network_shortcut_probability: f64,
    pub initial_seed_agents: usize,
    pub initial_seed_recipes_per_agent: usize,
    pub individual_attempts_per_agent_step: usize,
    pub individual_discovery_probability: f64,
    pub transmission_budget_per_step: usize,
    pub attention_capacity_per_recipient: usize,
    pub acquisition_probability: f64,
    pub policy: TransmissionPolicy,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            population_size: 12,
            steps: 12,
            network_shortcut_probability: 0.08,
            initial_seed_agents: 1,
            initial_seed_recipes_per_agent: 3,
            individual_attempts_per_agent_step: 2,
            individual_discovery_probability: 0.08,
            transmission_budget_per_step: 12,
            attention_capacity_per_recipient: 2,
            acquisition_probability: 0.9,
            policy: TransmissionPolicy::StateAwareFrontier,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    PopulationTooSmall,
    ZeroSteps,
    TooManySeedAgents { seeds: usize, population: usize },
    InvalidProbability { name: &'static str },
    ZeroAttentionCapacity,
    ZeroTransmissionBudget,
    SeedRecipesTooLarge { requested: usize, available: usize },
    InvalidProgramIdentifier,
    InvalidProgramVersion,
    InvalidProgramPrerequisiteThreshold { threshold_milli: u16 },
    InvalidProgramMessageLimit,
    InvalidProgramScoreWeights,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PopulationTooSmall => write!(f, "population_size must be at least three"),
            Self::ZeroSteps => write!(f, "steps must be positive"),
            Self::TooManySeedAgents { seeds, population } => {
                write!(f, "{seeds} seed agents exceed population size {population}")
            }
            Self::InvalidProbability { name } => write!(f, "{name} must be finite and in [0, 1]"),
            Self::ZeroAttentionCapacity => write!(f, "attention capacity must be positive"),
            Self::ZeroTransmissionBudget => write!(f, "transmission budget must be positive"),
            Self::SeedRecipesTooLarge { requested, available } => {
                write!(f, "requested {requested} initial recipes per seed agent, task has {available}")
            }
            Self::InvalidProgramIdentifier => write!(f, "transmission program ID must not be empty"),
            Self::InvalidProgramVersion => write!(f, "transmission program version must be positive"),
            Self::InvalidProgramPrerequisiteThreshold { threshold_milli } => write!(
                f,
                "minimum prerequisite match {threshold_milli} milli-units exceeds 1000"
            ),
            Self::InvalidProgramMessageLimit => write!(
                f,
                "program message limits must be positive and no greater than the configured per-step budget"
            ),
            Self::InvalidProgramScoreWeights => write!(
                f,
                "program score weights must be bounded, nonzero in total, and no greater than the declared maximum"
            ),
        }
    }
}
impl std::error::Error for ConfigError {}

impl Config {
    pub fn validate(self, task: &InnovationTask) -> Result<(), ConfigError> {
        if self.population_size < 3 {
            return Err(ConfigError::PopulationTooSmall);
        }
        if self.steps == 0 {
            return Err(ConfigError::ZeroSteps);
        }
        if self.initial_seed_agents > self.population_size {
            return Err(ConfigError::TooManySeedAgents {
                seeds: self.initial_seed_agents,
                population: self.population_size,
            });
        }
        for (name, value) in [
            ("network_shortcut_probability", self.network_shortcut_probability),
            ("individual_discovery_probability", self.individual_discovery_probability),
            ("acquisition_probability", self.acquisition_probability),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(ConfigError::InvalidProbability { name });
            }
        }
        if self.attention_capacity_per_recipient == 0 {
            return Err(ConfigError::ZeroAttentionCapacity);
        }
        if self.transmission_budget_per_step == 0 && self.policy != TransmissionPolicy::NoSocial {
            return Err(ConfigError::ZeroTransmissionBudget);
        }
        if self.initial_seed_recipes_per_agent > task.total_recipe_count() {
            return Err(ConfigError::SeedRecipesTooLarge {
                requested: self.initial_seed_recipes_per_agent,
                available: task.total_recipe_count(),
            });
        }
        Ok(())
    }
}

/// Bounded, deterministic score terms for a transmission program.
///
/// All weights are non-negative integer coefficients. The scheduler uses
/// integer arithmetic and stable action ordering; no executable code or
/// arbitrary expression evaluation is accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransmissionScoreWeights {
    pub immediate_executability: u32,
    pub prerequisite_match: u32,
    pub collective_novelty: u32,
    pub downstream_unlocks: u32,
    pub sender_knowledge: u32,
}

impl TransmissionScoreWeights {
    pub const MAX_WEIGHT: u32 = 1_000_000;

    fn is_valid(self) -> bool {
        let values = [
            self.immediate_executability,
            self.prerequisite_match,
            self.collective_novelty,
            self.downstream_unlocks,
            self.sender_knowledge,
        ];
        values.iter().any(|weight| *weight > 0)
            && values.iter().all(|weight| *weight <= Self::MAX_WEIGHT)
    }
}

/// A versioned, inspectable transmission program.
///
/// The program can only rank legal sender/receiver/recipe actions. It cannot
/// mutate agent state, inspect hidden task truth, bypass graph edges, or exceed
/// the global message and recipient-attention limits in Config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransmissionProgram {
    pub program_id: String,
    pub version: u16,
    /// Minimum proportion of the recipe's prerequisites already known by the
    /// recipient, expressed in thousandths (0..=1000).
    pub minimum_prerequisite_match_milli: u16,
    /// When true, only transmit recipes whose output is not yet known by the
    /// collective.
    pub require_collective_novelty: bool,
    pub max_messages_per_step: usize,
    pub max_messages_per_sender: usize,
    pub weights: TransmissionScoreWeights,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramIdentity {
    pub program_id: String,
    pub version: u16,
}

impl TransmissionProgram {
    /// Re-expresses the transparent StateAwareFrontier heuristic as a bounded
    /// data program. Its default weights are exactly proportional to that
    /// heuristic's score terms, preserving its deterministic ranking.
    pub fn frontier_v1(config: Config) -> Self {
        Self {
            program_id: "frontier-v1".to_owned(),
            version: 1,
            minimum_prerequisite_match_milli: 0,
            require_collective_novelty: false,
            max_messages_per_step: config.transmission_budget_per_step,
            max_messages_per_sender: config.transmission_budget_per_step,
            weights: TransmissionScoreWeights {
                immediate_executability: 100_000,
                prerequisite_match: 10_000,
                collective_novelty: 5_000,
                downstream_unlocks: 1_000,
                sender_knowledge: 1,
            },
        }
    }

    pub fn identity(&self) -> ProgramIdentity {
        ProgramIdentity {
            program_id: self.program_id.clone(),
            version: self.version,
        }
    }

    pub fn validate(&self, config: Config) -> Result<(), ConfigError> {
        if self.program_id.trim().is_empty() {
            return Err(ConfigError::InvalidProgramIdentifier);
        }
        if self.version == 0 {
            return Err(ConfigError::InvalidProgramVersion);
        }
        if self.minimum_prerequisite_match_milli > 1_000 {
            return Err(ConfigError::InvalidProgramPrerequisiteThreshold {
                threshold_milli: self.minimum_prerequisite_match_milli,
            });
        }
        if config.transmission_budget_per_step == 0
            || self.max_messages_per_step == 0
            || self.max_messages_per_sender == 0
            || self.max_messages_per_step > config.transmission_budget_per_step
            || self.max_messages_per_sender > config.transmission_budget_per_step
        {
            return Err(ConfigError::InvalidProgramMessageLimit);
        }
        if !self.weights.is_valid() {
            return Err(ConfigError::InvalidProgramScoreWeights);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TransmissionAction {
    pub sender: usize,
    pub receiver: usize,
    pub recipe_id: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocialGraph {
    adjacency: Vec<Vec<usize>>,
}

impl SocialGraph {
    fn generate(population_size: usize, shortcut_probability: f64, rng: &mut Rng) -> Self {
        let mut adjacency = vec![BTreeSet::new(); population_size];
        for node in 0..population_size {
            let next = (node + 1) % population_size;
            adjacency[node].insert(next);
            adjacency[next].insert(node);
        }
        for left in 0..population_size {
            for right in (left + 1)..population_size {
                if adjacency[left].contains(&right) {
                    continue;
                }
                if rng.bernoulli(shortcut_probability) {
                    adjacency[left].insert(right);
                    adjacency[right].insert(left);
                }
            }
        }
        Self {
            adjacency: adjacency.into_iter().map(|neighbors| neighbors.into_iter().collect()).collect(),
        }
    }

    pub fn neighbors(&self, agent: usize) -> Option<&[usize]> {
        self.adjacency.get(agent).map(Vec::as_slice)
    }

    pub fn density(&self) -> f64 {
        let directed_edges: usize = self.adjacency.iter().map(Vec::len).sum();
        let n = self.adjacency.len();
        directed_edges as f64 / (n * n.saturating_sub(1)) as f64
    }

    fn directed_pairs(&self) -> Vec<(usize, usize)> {
        self.adjacency.iter().enumerate()
            .flat_map(|(sender, neighbors)| neighbors.iter().copied().map(move |receiver| (sender, receiver)))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepMetrics {
    pub step: usize,
    pub unique_results_discovered: usize,
    pub result_coverage_count: usize,
    pub total_results: usize,
    pub max_discovered_depth: u16,
    pub mean_results_per_agent_milli: u32,
    pub mean_recipes_known_per_agent_milli: u32,
    pub transmissions_scheduled: usize,
    pub redundant_transmissions: usize,
    pub successful_recipe_acquisitions: usize,
    pub individual_recipe_discoveries: usize,
    pub new_collective_results: usize,
    pub attention_capacity_used: usize,
}

impl StepMetrics {
    pub fn result_coverage(self) -> f64 {
        self.result_coverage_count as f64 / self.total_results as f64
    }

    pub fn transmission_efficiency(self) -> Option<f64> {
        (self.transmissions_scheduled > 0)
            .then(|| self.new_collective_results as f64 / self.transmissions_scheduled as f64)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub seed: u64,
    pub graph: SocialGraph,
    pub agents: Vec<AgentKnowledge>,
    pub history: Vec<StepMetrics>,
    pub program_identity: Option<ProgramIdentity>,
}

impl Run {
    pub fn final_metrics(&self) -> Option<StepMetrics> {
        self.history.last().copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn bernoulli(&mut self, probability: f64) -> bool {
        self.uniform() < probability
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn index(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() as usize) % upper
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let selected = self.index(index + 1);
            values.swap(index, selected);
        }
    }
}

fn group_known_elements(agents: &[AgentKnowledge]) -> BTreeSet<u16> {
    agents.iter().flat_map(|agent| agent.known_elements.iter().copied()).collect()
}

fn group_known_recipes(agents: &[AgentKnowledge]) -> BTreeSet<u16> {
    agents.iter().flat_map(|agent| agent.known_recipes.iter().copied()).collect()
}

fn collective_metrics(
    step: usize,
    task: &InnovationTask,
    agents: &[AgentKnowledge],
    transmissions_scheduled: usize,
    redundant_transmissions: usize,
    successful_recipe_acquisitions: usize,
    individual_recipe_discoveries: usize,
    new_collective_results: usize,
    attention_capacity_used: usize,
) -> StepMetrics {
    let group_elements = group_known_elements(agents);
    let unique_results_discovered = task.recipes().iter()
        .filter(|recipe| group_elements.contains(&recipe.output_element)).count();
    let max_discovered_depth = task.recipes().iter()
        .filter(|recipe| group_elements.contains(&recipe.output_element))
        .filter_map(|recipe| task.depth(recipe.output_element))
        .max()
        .unwrap_or(0);
    let total_results = task.total_result_count();
    let total_element_count: usize = agents.iter()
        .map(|agent| agent.known_elements.iter()
            .filter(|element| !task.base_elements().contains(element))
            .count())
        .sum();
    let total_recipe_count: usize = agents.iter().map(|agent| agent.known_recipes.len()).sum();
    let scale = agents.len() as u64;
    let mean_results = ((total_element_count as u64 * 1000) / scale) as u32;
    let mean_recipes = ((total_recipe_count as u64 * 1000) / scale) as u32;

    StepMetrics {
        step,
        unique_results_discovered,
        result_coverage_count: unique_results_discovered,
        total_results,
        max_discovered_depth,
        mean_results_per_agent_milli: mean_results,
        mean_recipes_known_per_agent_milli: mean_recipes,
        transmissions_scheduled,
        redundant_transmissions,
        successful_recipe_acquisitions,
        individual_recipe_discoveries,
        new_collective_results,
        attention_capacity_used,
    }
}

fn individual_learning(
    task: &InnovationTask,
    agents: &mut [AgentKnowledge],
    attempts_per_agent: usize,
    discovery_probability: f64,
    rng: &mut Rng,
) -> usize {
    let mut discoveries = 0usize;
    for agent in agents {
        let mut candidates: Vec<u16> = task.recipes().iter()
            .filter(|recipe| !agent.known_recipes.contains(&recipe.recipe_id))
            .filter(|recipe| recipe.ingredients.iter().all(|ingredient| agent.known_elements.contains(ingredient)))
            .map(|recipe| recipe.recipe_id)
            .collect();
        rng.shuffle(&mut candidates);
        for recipe_id in candidates.into_iter().take(attempts_per_agent) {
            if rng.bernoulli(discovery_probability) {
                agent.known_recipes.insert(recipe_id);
                agent.individual_discoveries += 1;
                discoveries += 1;
                agent.apply_closure(task);
            }
        }
    }
    discoveries
}

fn action_score(
    task: &InnovationTask,
    agents: &[AgentKnowledge],
    group_elements: &BTreeSet<u16>,
    sender: usize,
    receiver: usize,
    recipe: &Recipe,
) -> f64 {
    let receiver_state = &agents[receiver];
    let ingredient_count = recipe.ingredients.len();
    let matched = recipe.ingredients.iter()
        .filter(|ingredient| receiver_state.known_elements.contains(ingredient))
        .count();
    let ready = matched == ingredient_count;
    let input_coverage = matched as f64 / ingredient_count as f64;
    let group_novelty = !group_elements.contains(&recipe.output_element);
    let future_unlocks = task.descendant_count(recipe.output_element);
    let sender_knowledge = agents[sender].known_recipes.len();

    (if ready { 100.0 } else { 0.0 })
        + input_coverage * 10.0
        + if group_novelty { 5.0 } else { 0.0 }
        + future_unlocks as f64
        + sender_knowledge as f64 * 0.001
}

fn state_aware_schedule(
    task: &InnovationTask,
    graph: &SocialGraph,
    agents: &[AgentKnowledge],
    config: Config,
) -> Vec<TransmissionAction> {
    let group_elements = group_known_elements(agents);
    let mut schedule = Vec::new();
    let mut recipient_load = vec![0usize; agents.len()];
    let mut used_actions = BTreeSet::new();

    while schedule.len() < config.transmission_budget_per_step {
        let mut best: Option<(f64, TransmissionAction)> = None;
        for sender in 0..agents.len() {
            let Some(neighbors) = graph.neighbors(sender) else { continue };
            if agents[sender].known_recipes.is_empty() { continue; }
            for &receiver in neighbors {
                if recipient_load[receiver] >= config.attention_capacity_per_recipient {
                    continue;
                }
                for &recipe_id in &agents[sender].known_recipes {
                    if agents[receiver].known_recipes.contains(&recipe_id) {
                        continue;
                    }
                    let action = TransmissionAction { sender, receiver, recipe_id };
                    if used_actions.contains(&(sender, receiver, recipe_id)) {
                        continue;
                    }
                    let recipe = task.recipe(recipe_id).expect("agent recipe id must exist in task");
                    let score = action_score(task, agents, &group_elements, sender, receiver, recipe);
                    let replace = match best {
                        None => true,
                        Some((best_score, best_action)) => {
                            score > best_score
                                || (score == best_score && action < best_action)
                        }
                    };
                    if replace {
                        best = Some((score, action));
                    }
                }
            }
        }
        let Some((_, action)) = best else { break };
        used_actions.insert((action.sender, action.receiver, action.recipe_id));
        recipient_load[action.receiver] += 1;
        schedule.push(action);
    }
    schedule
}

fn program_schedule(
    task: &InnovationTask,
    graph: &SocialGraph,
    agents: &[AgentKnowledge],
    config: Config,
    program: &TransmissionProgram,
) -> Vec<TransmissionAction> {
    let group_elements = group_known_elements(agents);
    let mut schedule = Vec::new();
    let mut recipient_load = vec![0usize; agents.len()];
    let mut sender_load = vec![0usize; agents.len()];
    let mut used_actions = BTreeSet::new();
    let budget = program.max_messages_per_step.min(config.transmission_budget_per_step);

    while schedule.len() < budget {
        let mut best: Option<(u64, TransmissionAction)> = None;
        for sender in 0..agents.len() {
            if sender_load[sender] >= program.max_messages_per_sender {
                continue;
            }
            let Some(neighbors) = graph.neighbors(sender) else { continue };
            if agents[sender].known_recipes.is_empty() {
                continue;
            }
            for &receiver in neighbors {
                if recipient_load[receiver] >= config.attention_capacity_per_recipient {
                    continue;
                }
                for &recipe_id in &agents[sender].known_recipes {
                    if agents[receiver].known_recipes.contains(&recipe_id) {
                        continue;
                    }
                    let recipe = task.recipe(recipe_id)
                        .expect("agent recipe ID must exist in task");
                    let matched = recipe.ingredients.iter()
                        .filter(|ingredient| agents[receiver].known_elements.contains(ingredient))
                        .count();
                    let prerequisite_match_milli =
                        (matched * 1_000 / recipe.ingredients.len()) as u16;
                    if prerequisite_match_milli < program.minimum_prerequisite_match_milli {
                        continue;
                    }

                    let group_novelty = !group_elements.contains(&recipe.output_element);
                    if program.require_collective_novelty && !group_novelty {
                        continue;
                    }

                    let action = TransmissionAction { sender, receiver, recipe_id };
                    if used_actions.contains(&(sender, receiver, recipe_id)) {
                        continue;
                    }

                    let ready = matched == recipe.ingredients.len();
                    let weights = program.weights;
                    let immediate_score = if ready {
                        u64::from(weights.immediate_executability)
                    } else {
                        0
                    };
                    let novelty_score = if group_novelty {
                        u64::from(weights.collective_novelty)
                    } else {
                        0
                    };
                    let score =
                        immediate_score
                        + (matched as u64 * u64::from(weights.prerequisite_match)
                            / recipe.ingredients.len() as u64)
                        + novelty_score
                        + task.descendant_count(recipe.output_element) as u64
                            * u64::from(weights.downstream_unlocks)
                        + agents[sender].known_recipes.len() as u64
                            * u64::from(weights.sender_knowledge);

                    let replace = match best {
                        None => true,
                        Some((best_score, best_action)) => {
                            score > best_score
                                || (score == best_score && action < best_action)
                        }
                    };
                    if replace {
                        best = Some((score, action));
                    }
                }
            }
        }

        let Some((_, action)) = best else { break };
        used_actions.insert((action.sender, action.receiver, action.recipe_id));
        recipient_load[action.receiver] += 1;
        sender_load[action.sender] += 1;
        schedule.push(action);
    }
    schedule
}

fn fixed_ring_pairs(population_size: usize) -> Vec<(usize, usize)> {
    (0..population_size)
        .map(|sender| (sender, (sender + 1) % population_size))
        .collect()
}

/// Shuffle selected payloads only among the same sender's scheduled
/// recipients. Sender/receiver action order, message count and each sender's
/// payload multiset remain unchanged.
fn shuffle_sender_payloads(actions: &mut [TransmissionAction], rng: &mut Rng) {
    let mut action_positions_by_sender: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (position, action) in actions.iter().enumerate() {
        action_positions_by_sender
            .entry(action.sender)
            .or_default()
            .push(position);
    }
    for positions in action_positions_by_sender.values() {
        let mut payloads: Vec<u16> = positions
            .iter()
            .map(|&position| actions[position].recipe_id)
            .collect();
        rng.shuffle(&mut payloads);
        for (&position, recipe_id) in positions.iter().zip(payloads) {
            actions[position].recipe_id = recipe_id;
        }
    }
}

fn build_schedule(
    policy: TransmissionPolicy,
    task: &InnovationTask,
    graph: &SocialGraph,
    agents: &[AgentKnowledge],
    config: Config,
    rng: &mut Rng,
) -> Vec<TransmissionAction> {
    match policy {
        TransmissionPolicy::NoSocial => Vec::new(),
        TransmissionPolicy::StateAwareFrontier => state_aware_schedule(task, graph, agents, config),
        TransmissionPolicy::StateAwareRandomContent => {
            let mut actions = state_aware_schedule(task, graph, agents, config);
            // Preserve the sender/receiver schedule, timing and message count,
            // but choose random content from each sender's current inventory.
            // This approximates the paper's schedule-preserving random-memory
            // ablation; it does not preserve the selected-content multiset.
            for action in &mut actions {
                let candidates: Vec<u16> = agents[action.sender]
                    .known_recipes
                    .iter()
                    .copied()
                    .collect();
                if !candidates.is_empty() {
                    action.recipe_id = candidates[rng.index(candidates.len())];
                }
            }
            actions
        }
        TransmissionPolicy::StateAwareRecipientShuffled => {
            let mut actions = state_aware_schedule(task, graph, agents, config);
            // A stricter control: hold each sender's selected payload multiset
            // fixed, but shuffle those payloads among that sender's scheduled
            // recipients. This isolates recipient matching specifically.
            shuffle_sender_payloads(&mut actions, rng);
            actions
        }
        TransmissionPolicy::FixedRing | TransmissionPolicy::FixedRingContentShuffled => {
            let mut actions = Vec::new();
            let mut loads = vec![0usize; agents.len()];
            for (sender, receiver) in fixed_ring_pairs(agents.len()) {
                if loads[receiver] >= config.attention_capacity_per_recipient {
                    continue;
                }
                let candidates: Vec<u16> = agents[sender].known_recipes.iter().copied().collect();
                if candidates.is_empty() {
                    continue;
                }
                let recipe_id = if policy == TransmissionPolicy::FixedRingContentShuffled {
                    candidates[rng.index(candidates.len())]
                } else {
                    *candidates.first().expect("candidate list is not empty")
                };
                actions.push(TransmissionAction { sender, receiver, recipe_id });
                loads[receiver] += 1;
                if actions.len() >= config.transmission_budget_per_step {
                    break;
                }
            }
            actions
        }
        TransmissionPolicy::RandomNetworkContent => {
            let mut pairs = graph.directed_pairs();
            rng.shuffle(&mut pairs);
            let mut actions = Vec::new();
            let mut loads = vec![0usize; agents.len()];
            for (sender, receiver) in pairs {
                if actions.len() >= config.transmission_budget_per_step {
                    break;
                }
                if loads[receiver] >= config.attention_capacity_per_recipient {
                    continue;
                }
                let candidates: Vec<u16> = agents[sender].known_recipes.iter().copied().collect();
                if candidates.is_empty() {
                    continue;
                }
                let recipe_id = candidates[rng.index(candidates.len())];
                actions.push(TransmissionAction { sender, receiver, recipe_id });
                loads[receiver] += 1;
            }
            actions
        }
    }
}

fn apply_actions(
    task: &InnovationTask,
    agents: &mut [AgentKnowledge],
    actions: &[TransmissionAction],
    acquisition_probability: f64,
    rng: &mut Rng,
) -> (usize, usize, usize, usize) {
    let mut redundant = 0usize;
    let mut acquisitions = 0usize;
    let mut new_group_results = 0usize;
    let mut attention_used = 0usize;
    let before_group = group_known_elements(agents);
    for action in actions {
        attention_used += 1;
        if agents[action.receiver].known_recipes.contains(&action.recipe_id) {
            redundant += 1;
            continue;
        }
        if !rng.bernoulli(acquisition_probability) {
            continue;
        }
        // The sender must genuinely possess the transmitted recipe. This is
        // checked at the transmission boundary, not inferred from the plan.
        if !agents[action.sender].known_recipes.contains(&action.recipe_id) {
            continue;
        }
        agents[action.receiver].known_recipes.insert(action.recipe_id);
        agents[action.receiver].social_recipe_acquisitions += 1;
        acquisitions += 1;
        agents[action.receiver].apply_closure(task);
    }
    let after_group = group_known_elements(agents);
    new_group_results = task.recipes().iter()
        .filter(|recipe| !before_group.contains(&recipe.output_element)
            && after_group.contains(&recipe.output_element))
        .count();
    (redundant, acquisitions, new_group_results, attention_used)
}

/// Simulate one state-aware transmission experiment on a fixed recipe DAG.
pub fn simulate(
    task: &InnovationTask,
    config: Config,
    seed: u64,
) -> Result<Run, ConfigError> {
    simulate_internal(task, config, seed, None)
}

/// Simulate with an inspectable, bounded data program instead of a fixed policy
/// enum. The program cannot bypass the graph, change agent state directly, or
/// exceed the configured message/recipient-attention budgets.
pub fn simulate_with_program(
    task: &InnovationTask,
    config: Config,
    seed: u64,
    program: &TransmissionProgram,
) -> Result<Run, ConfigError> {
    program.validate(config)?;
    simulate_internal(task, config, seed, Some(program))
}

fn simulate_internal(
    task: &InnovationTask,
    config: Config,
    seed: u64,
    program: Option<&TransmissionProgram>,
) -> Result<Run, ConfigError> {
    config.validate(task)?;
    let mut rng = Rng::new(seed);
    let graph = SocialGraph::generate(
        config.population_size,
        config.network_shortcut_probability,
        &mut rng,
    );
    let mut agents: Vec<AgentKnowledge> = (0..config.population_size)
        .map(|_| AgentKnowledge::new(task))
        .collect();

    // Seed knowledge is deterministic and explicit: initial recipes are
    // distributed round-robin among the configured seed agents.
    for agent_index in 0..config.initial_seed_agents {
        let first_recipe = agent_index
            .saturating_mul(config.initial_seed_recipes_per_agent)
            % task.recipes().len();
        for offset in 0..config.initial_seed_recipes_per_agent {
            let recipe_index = (first_recipe + offset) % task.recipes().len();
            let task_recipe = &task.recipes()[recipe_index];
            agents[agent_index].known_recipes.insert(task_recipe.recipe_id);
            agents[agent_index].apply_closure(task);
        }
    }

    let mut history = Vec::with_capacity(config.steps + 1);
    history.push(collective_metrics(0, task, &agents, 0, 0, 0, 0, 0, 0));

    for step in 1..=config.steps {
        let individual_discoveries = individual_learning(
            task,
            &mut agents,
            config.individual_attempts_per_agent_step,
            config.individual_discovery_probability,
            &mut rng,
        );
        let actions = match program {
            Some(program) => program_schedule(task, &graph, &agents, config, program),
            None => build_schedule(config.policy, task, &graph, &agents, config, &mut rng),
        };
        let (redundant, acquisitions, new_results, attention_used) = apply_actions(
            task,
            &mut agents,
            &actions,
            config.acquisition_probability,
            &mut rng,
        );
        history.push(collective_metrics(
            step,
            task,
            &agents,
            actions.len(),
            redundant,
            acquisitions,
            individual_discoveries,
            new_results,
            attention_used,
        ));
    }

    Ok(Run {
        seed,
        graph,
        agents,
        history,
        program_identity: program.map(TransmissionProgram::identity),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_config(policy: TransmissionPolicy) -> Config {
        Config {
            population_size: 6,
            steps: 4,
            network_shortcut_probability: 0.0,
            initial_seed_agents: 1,
            initial_seed_recipes_per_agent: 4,
            individual_attempts_per_agent_step: 0,
            individual_discovery_probability: 0.0,
            transmission_budget_per_step: 6,
            attention_capacity_per_recipient: 2,
            acquisition_probability: 1.0,
            policy,
        }
    }

    #[test]
    fn built_in_tasks_are_acyclic_and_have_distinct_families() {
        for family in [TaskFamily::Layered, TaskFamily::Branching, TaskFamily::Convergent] {
            let task = InnovationTask::family_task(family);
            assert_eq!(task.total_recipe_count(), 14);
            assert!(task.recipes().iter().all(|recipe| {
                recipe.ingredients.iter().all(|ingredient| *ingredient < recipe.output_element)
            }));
        }
    }

    #[test]
    fn task_rejects_unknown_and_non_topological_ingredients() {
        let unknown = InnovationTask::new(
            TaskFamily::Layered,
            vec![0, 1],
            vec![recipe(0, &[0, 9], 2)],
        );
        assert!(matches!(unknown, Err(TaskError::UnknownIngredient { .. })));

        let backwards = InnovationTask::new(
            TaskFamily::Layered,
            vec![0, 1],
            vec![recipe(0, &[0, 3], 2)],
        );
        assert!(matches!(backwards, Err(TaskError::UnknownIngredient { .. })));
    }

    #[test]
    fn acquired_recipe_can_be_stored_before_its_prerequisites_exist() {
        let task = InnovationTask::new(
            TaskFamily::Convergent,
            vec![0, 1, 2],
            vec![
                recipe(0, &[0, 1], 3),
                recipe(1, &[1, 2], 4),
                recipe(2, &[3, 4], 5),
            ],
        )
        .unwrap();
        let mut agent = AgentKnowledge::new(&task);
        agent.known_recipes.insert(2);
        assert_eq!(agent.apply_closure(&task), 0);
        agent.known_elements.insert(3);
        agent.known_elements.insert(4);
        assert_eq!(agent.apply_closure(&task), 1);
        assert!(agent.known_elements.contains(&5));
    }

    #[test]
    fn initial_seed_agents_receive_distinct_recipe_partitions() {
        let task = InnovationTask::family_task(TaskFamily::Layered);
        let config = Config {
            population_size: 6,
            steps: 1,
            initial_seed_agents: 2,
            initial_seed_recipes_per_agent: 3,
            individual_attempts_per_agent_step: 0,
            individual_discovery_probability: 0.0,
            policy: TransmissionPolicy::NoSocial,
            ..Config::default()
        };
        let run = simulate(&task, config, 919).unwrap();
        let first = &run.agents[0].known_recipes;
        let second = &run.agents[1].known_recipes;
        assert_eq!(first.len(), 3);
        assert_eq!(second.len(), 3);
        assert!(first.is_disjoint(second));
    }

    #[test]
    fn closure_derives_outputs_only_when_all_inputs_are_known() {
        let task = InnovationTask::family_task(TaskFamily::Layered);
        let mut agent = AgentKnowledge::new(&task);
        agent.known_recipes.insert(0);
        assert_eq!(agent.apply_closure(&task), 1);
        assert!(agent.known_elements.contains(&6));
        assert!(!agent.known_elements.contains(&11));
    }

    #[test]
    fn sender_local_shuffle_preserves_schedule_and_per_sender_payload_multisets() {
        let original = vec![
            TransmissionAction { sender: 0, receiver: 1, recipe_id: 3 },
            TransmissionAction { sender: 0, receiver: 2, recipe_id: 7 },
            TransmissionAction { sender: 1, receiver: 2, recipe_id: 9 },
        ];
        let original_pairs: Vec<_> = original.iter()
            .map(|action| (action.sender, action.receiver))
            .collect();
        let original_payloads: Vec<_> = original.iter()
            .map(|action| (action.sender, action.recipe_id))
            .collect();

        let mut at_least_one_non_identity_permutation = false;
        for seed in 0..64 {
            let mut actions = original.clone();
            let mut rng = Rng::new(seed);
            shuffle_sender_payloads(&mut actions, &mut rng);

            let shuffled_pairs: Vec<_> = actions.iter()
                .map(|action| (action.sender, action.receiver))
                .collect();
            let shuffled_payloads: Vec<_> = actions.iter()
                .map(|action| (action.sender, action.recipe_id))
                .collect();
            assert_eq!(original_pairs, shuffled_pairs);

            let mut before = original_payloads.clone();
            let mut after = shuffled_payloads;
            before.sort_unstable();
            after.sort_unstable();
            assert_eq!(before, after);

            at_least_one_non_identity_permutation |= actions
                .iter()
                .zip(&original)
                .any(|(shuffled, initial)| shuffled.recipe_id != initial.recipe_id);
        }
        assert!(at_least_one_non_identity_permutation);
    }

    #[test]
    fn recipient_shuffle_preserves_schedule_and_per_sender_content_multiset() {
        let task = InnovationTask::family_task(TaskFamily::Layered);
        let config = deterministic_config(TransmissionPolicy::StateAwareFrontier);
        let mut rng_a = Rng::new(77);
        let graph = SocialGraph::generate(config.population_size, 0.0, &mut rng_a);
        let agents = simulate(&task, Config { steps: 1, ..config }, 1).unwrap().agents;
        let mut rng_pairs = Rng::new(91);
        let state_aware = state_aware_schedule(&task, &graph, &agents, config);
        let shuffled = build_schedule(
            TransmissionPolicy::StateAwareRecipientShuffled,
            &task,
            &graph,
            &agents,
            config,
            &mut rng_pairs,
        );
        let pairs_a: Vec<_> = state_aware.iter().map(|a| (a.sender, a.receiver)).collect();
        let pairs_b: Vec<_> = shuffled.iter().map(|a| (a.sender, a.receiver)).collect();
        assert_eq!(pairs_a, pairs_b);

        let mut payloads_by_sender_a: BTreeMap<usize, Vec<u16>> = BTreeMap::new();
        let mut payloads_by_sender_b: BTreeMap<usize, Vec<u16>> = BTreeMap::new();
        for action in &state_aware {
            payloads_by_sender_a.entry(action.sender).or_default().push(action.recipe_id);
        }
        for action in &shuffled {
            assert!(agents[action.sender].known_recipes.contains(&action.recipe_id));
            payloads_by_sender_b.entry(action.sender).or_default().push(action.recipe_id);
        }
        for payloads in payloads_by_sender_a.values_mut() {
            payloads.sort_unstable();
        }
        for payloads in payloads_by_sender_b.values_mut() {
            payloads.sort_unstable();
        }
        assert_eq!(payloads_by_sender_a, payloads_by_sender_b);
    }

    #[test]
    fn random_content_ablation_preserves_state_aware_schedule_and_count() {
        let task = InnovationTask::family_task(TaskFamily::Branching);
        let config = deterministic_config(TransmissionPolicy::StateAwareFrontier);
        let mut graph_rng = Rng::new(31);
        let graph = SocialGraph::generate(config.population_size, 0.0, &mut graph_rng);
        let agents = simulate(&task, Config { steps: 1, ..config }, 51).unwrap().agents;
        let state_aware = state_aware_schedule(&task, &graph, &agents, config);
        let mut content_rng = Rng::new(52);
        let randomized = build_schedule(
            TransmissionPolicy::StateAwareRandomContent,
            &task,
            &graph,
            &agents,
            config,
            &mut content_rng,
        );
        let pairs_a: Vec<_> = state_aware.iter()
            .map(|action| (action.sender, action.receiver))
            .collect();
        let pairs_b: Vec<_> = randomized.iter()
            .map(|action| (action.sender, action.receiver))
            .collect();
        assert_eq!(pairs_a, pairs_b);
        assert_eq!(state_aware.len(), randomized.len());
        assert!(randomized.iter().all(|action| {
            agents[action.sender].known_recipes.contains(&action.recipe_id)
        }));
    }

    #[test]
    fn state_aware_actions_respect_sender_inventory_and_receiver_attention() {
        let task = InnovationTask::family_task(TaskFamily::Branching);
        let config = deterministic_config(TransmissionPolicy::StateAwareFrontier);
        let mut rng = Rng::new(8);
        let graph = SocialGraph::generate(config.population_size, 0.0, &mut rng);
        let agents = simulate(&task, Config { steps: 1, ..config }, 3).unwrap().agents;
        let actions = state_aware_schedule(&task, &graph, &agents, config);
        let mut loads = vec![0usize; config.population_size];
        for action in &actions {
            assert!(agents[action.sender].known_recipes.contains(&action.recipe_id));
            loads[action.receiver] += 1;
        }
        assert!(loads.iter().all(|count| *count <= config.attention_capacity_per_recipient));
        assert!(actions.len() <= config.transmission_budget_per_step);
    }

    #[test]
    fn no_social_baseline_never_schedules_transmissions() {
        let task = InnovationTask::family_task(TaskFamily::Layered);
        let config = deterministic_config(TransmissionPolicy::NoSocial);
        let run = simulate(&task, config, 44).unwrap();
        assert!(run.history.iter().all(|m| m.transmissions_scheduled == 0));
    }

    #[test]
    fn fixed_seed_simulation_is_deterministic() {
        let task = InnovationTask::family_task(TaskFamily::Convergent);
        let config = deterministic_config(TransmissionPolicy::StateAwareFrontier);
        let a = simulate(&task, config, 404).unwrap();
        let b = simulate(&task, config, 404).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn state_aware_frontier_does_not_send_recipe_already_known_by_receiver() {
        let task = InnovationTask::family_task(TaskFamily::Layered);
        let config = deterministic_config(TransmissionPolicy::StateAwareFrontier);
        let mut rng = Rng::new(9);
        let graph = SocialGraph::generate(config.population_size, 0.0, &mut rng);
        let agents = simulate(&task, Config { steps: 1, ..config }, 5).unwrap().agents;
        let actions = state_aware_schedule(&task, &graph, &agents, config);
        assert!(actions.iter().all(|action| {
            !agents[action.receiver].known_recipes.contains(&action.recipe_id)
        }));
    }

    #[test]
    fn metrics_keep_result_coverage_and_transmission_cost_separate() {
        let task = InnovationTask::family_task(TaskFamily::Layered);
        let config = deterministic_config(TransmissionPolicy::StateAwareFrontier);
        let run = simulate(&task, config, 606).unwrap();
        let metrics = run.final_metrics().unwrap();
        assert!(metrics.result_coverage() >= 0.0 && metrics.result_coverage() <= 1.0);
        assert!(metrics.transmissions_scheduled <= config.transmission_budget_per_step);
        assert!(metrics.mean_results_per_agent_milli <= (task.total_result_count() * 1000) as u32);
    }

    #[test]
    fn frontier_v1_program_reproduces_named_heuristic_exactly() {
        let task = InnovationTask::family_task(TaskFamily::Branching);
        let config = deterministic_config(TransmissionPolicy::StateAwareFrontier);
        let heuristic = simulate(&task, config, 818).unwrap();
        let program = TransmissionProgram::frontier_v1(config);
        let compiled = simulate_with_program(&task, config, 818, &program).unwrap();

        assert_eq!(heuristic.graph, compiled.graph);
        assert_eq!(heuristic.agents, compiled.agents);
        assert_eq!(heuristic.history, compiled.history);
        assert_eq!(
            compiled.program_identity,
            Some(ProgramIdentity {
                program_id: "frontier-v1".to_owned(),
                version: 1,
            })
        );
    }

    #[test]
    fn bounded_program_respects_sender_message_cap() {
        let task = InnovationTask::family_task(TaskFamily::Layered);
        let config = deterministic_config(TransmissionPolicy::StateAwareFrontier);
        let mut program = TransmissionProgram::frontier_v1(config);
        program.program_id = "one-message-per-sender".to_owned();
        program.max_messages_per_sender = 1;

        let mut graph_rng = Rng::new(818);
        let graph = SocialGraph::generate(
            config.population_size,
            config.network_shortcut_probability,
            &mut graph_rng,
        );
        let agents = simulate(&task, Config { steps: 1, ..config }, 819).unwrap().agents;
        let actions = program_schedule(&task, &graph, &agents, config, &program);
        let mut sender_loads = vec![0usize; config.population_size];
        for action in &actions {
            sender_loads[action.sender] += 1;
        }
        assert!(sender_loads.iter().all(|count| *count <= 1));
        assert!(actions.len() <= config.transmission_budget_per_step);
    }

    #[test]
    fn bounded_program_rejects_invalid_filters_and_unbounded_weights() {
        let task = InnovationTask::family_task(TaskFamily::Layered);
        let config = deterministic_config(TransmissionPolicy::StateAwareFrontier);
        let mut threshold = TransmissionProgram::frontier_v1(config);
        threshold.minimum_prerequisite_match_milli = 1_001;
        assert_eq!(
            simulate_with_program(&task, config, 1, &threshold),
            Err(ConfigError::InvalidProgramPrerequisiteThreshold {
                threshold_milli: 1_001,
            })
        );

        let mut weights = TransmissionProgram::frontier_v1(config);
        weights.weights.immediate_executability =
            TransmissionScoreWeights::MAX_WEIGHT + 1;
        assert_eq!(
            simulate_with_program(&task, config, 1, &weights),
            Err(ConfigError::InvalidProgramScoreWeights)
        );
    }

    #[test]
    fn invalid_config_fails_closed() {
        let task = InnovationTask::family_task(TaskFamily::Layered);
        let config = Config { individual_discovery_probability: 1.1, ..Config::default() };
        assert_eq!(
            simulate(&task, config, 1),
            Err(ConfigError::InvalidProbability { name: "individual_discovery_probability" })
        );
    }
}
