// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Intentional teaching under constrained bandwidth.
//!
//! This is a synthetic mechanism benchmark inspired by recent human cultural
//! transmission work. It separates what a teacher selects, what a learner
//! observes, acquires, integrates, correctly executes, retains, and transfers.
//! It is not a reproduction of any cited human experiment.

use core::fmt;

/// Knowledge item metadata available to the teacher-selection policy.
/// `true_benefit` and `validity` are hidden evaluator fields and are never
/// read by the selection policy.
#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeItem {
    pub item_id: u64,
    pub lineage_id: u64,
    pub task_family: String,
    /// Evaluator-only held-out benefit if the item is correctly used.
    pub true_benefit: f64,
    /// Teacher-visible claimed benefit (may be noisy or inflated).
    pub declared_benefit: f64,
    /// Teacher-visible estimated evidence reliability.
    pub evidence_reliability: f64,
    /// Teacher-visible uncertainty in the evidence, [0, 1].
    pub uncertainty: f64,
    /// Cost to transmit/learn, in abstract time units.
    pub learning_cost: f64,
    /// Intrinsic difficulty, separate from time/resource cost.
    pub difficulty: f64,
    /// Evaluator-only probability the item is valid in its declared scope.
    pub validity: f64,
    /// Evaluator-only probability it transfers to a held-out context.
    pub transferability: f64,
}

impl KnowledgeItem {
    pub fn validate(&self) -> Result<(), TeachingError> {
        if self.task_family.trim().is_empty() {
            return Err(TeachingError::EmptyTaskFamily { item_id: self.item_id });
        }
        for (name, value) in [
            ("true_benefit", self.true_benefit),
            ("declared_benefit", self.declared_benefit),
            ("evidence_reliability", self.evidence_reliability),
            ("uncertainty", self.uncertainty),
            ("learning_cost", self.learning_cost),
            ("difficulty", self.difficulty),
            ("validity", self.validity),
            ("transferability", self.transferability),
        ] {
            if !value.is_finite() {
                return Err(TeachingError::NonFiniteItemField { item_id: self.item_id, name });
            }
        }
        if self.true_benefit < 0.0 || self.declared_benefit < 0.0 {
            return Err(TeachingError::NegativeBenefit { item_id: self.item_id });
        }
        for (name, value) in [
            ("evidence_reliability", self.evidence_reliability),
            ("uncertainty", self.uncertainty),
            ("difficulty", self.difficulty),
            ("validity", self.validity),
            ("transferability", self.transferability),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(TeachingError::InvalidProbability { item_id: self.item_id, name });
            }
        }
        if self.learning_cost <= 0.0 {
            return Err(TeachingError::NonPositiveLearningCost { item_id: self.item_id });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TeachingPolicy {
    /// Select randomly from items that fit the bandwidth budget.
    Random,
    /// Select the largest claimed immediate benefit first.
    DeclaredBenefit,
    /// Prefer benefit times evidence reliability, discounted by uncertainty.
    EvidenceCalibrated,
    /// Prefer items predicted to benefit this learner/context, using declared
    /// benefit and an explicit context-fit score.
    LearnerBenefit,
    /// Combine evidence-calibrated benefit with a bounded lineage-rarity bonus.
    DiversityAware { novelty_weight: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearningMode {
    /// Teacher selects a capsule/lesson packet under bandwidth constraints.
    Social { policy: TeachingPolicy },
    /// Matched individual-only baseline; no demonstrations or teacher signal.
    IndividualOnly,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LearnerProfile {
    pub task_family: String,
    /// Fit score declared by an independent task/context matcher.
    pub context_fit_by_item: Vec<(u64, f64)>,
}

impl LearnerProfile {
    fn validate(&self, items: &[KnowledgeItem]) -> Result<(), TeachingError> {
        if self.task_family.trim().is_empty() {
            return Err(TeachingError::EmptyLearnerTaskFamily);
        }
        let mut seen = std::collections::BTreeSet::new();
        for (item_id, fit) in &self.context_fit_by_item {
            if !seen.insert(*item_id) {
                return Err(TeachingError::DuplicateContextFit { item_id: *item_id });
            }
            if !fit.is_finite() || !(0.0..=1.0).contains(fit) {
                return Err(TeachingError::InvalidContextFit { item_id: *item_id });
            }
            if !items.iter().any(|item| item.item_id == *item_id) {
                return Err(TeachingError::UnknownContextFitItem { item_id: *item_id });
            }
        }
        Ok(())
    }

    fn context_fit(&self, item_id: u64) -> f64 {
        self.context_fit_by_item.iter()
            .find(|(id, _)| *id == item_id)
            .map(|(_, fit)| *fit)
            .unwrap_or(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TeachingConfig {
    pub bandwidth_budget: f64,
    pub individual_learning_budget: f64,
    pub observation_probability: f64,
    pub acquisition_base_probability: f64,
    pub integration_probability: f64,
    pub retention_probability: f64,
    pub individual_discovery_base_probability: f64,
    pub task_family_mismatch_penalty: f64,
}

impl Default for TeachingConfig {
    fn default() -> Self {
        Self {
            bandwidth_budget: 8.0,
            individual_learning_budget: 8.0,
            observation_probability: 0.95,
            acquisition_base_probability: 0.9,
            integration_probability: 0.85,
            retention_probability: 0.9,
            individual_discovery_base_probability: 0.12,
            task_family_mismatch_penalty: 0.0,
        }
    }
}

impl TeachingConfig {
    pub fn validate(self) -> Result<(), TeachingError> {
        for (name, value) in [
            ("observation_probability", self.observation_probability),
            ("acquisition_base_probability", self.acquisition_base_probability),
            ("integration_probability", self.integration_probability),
            ("retention_probability", self.retention_probability),
            ("individual_discovery_base_probability", self.individual_discovery_base_probability),
            ("task_family_mismatch_penalty", self.task_family_mismatch_penalty),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(TeachingError::InvalidConfigProbability { name });
            }
        }
        if !self.bandwidth_budget.is_finite()
            || !self.individual_learning_budget.is_finite()
            || self.bandwidth_budget < 0.0
            || self.individual_learning_budget < 0.0
        {
            return Err(TeachingError::InvalidBudget);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TeachingError {
    EmptyTaskFamily { item_id: u64 },
    NonFiniteItemField { item_id: u64, name: &'static str },
    NegativeBenefit { item_id: u64 },
    InvalidProbability { item_id: u64, name: &'static str },
    NonPositiveLearningCost { item_id: u64 },
    InvalidConfigProbability { name: &'static str },
    InvalidBudget,
    InvalidNoveltyWeight,
    DuplicateItemId { item_id: u64 },
    EmptyLearnerTaskFamily,
    DuplicateContextFit { item_id: u64 },
    InvalidContextFit { item_id: u64 },
    UnknownContextFitItem { item_id: u64 },
}

impl fmt::Display for TeachingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTaskFamily { item_id } => write!(f, "item {item_id} has an empty task family"),
            Self::NonFiniteItemField { item_id, name } => write!(f, "item {item_id} has non-finite {name}"),
            Self::NegativeBenefit { item_id } => write!(f, "item {item_id} has negative benefit"),
            Self::InvalidProbability { item_id, name } => write!(f, "item {item_id} has invalid {name}"),
            Self::NonPositiveLearningCost { item_id } => write!(f, "item {item_id} has non-positive learning cost"),
            Self::InvalidConfigProbability { name } => write!(f, "configuration {name} must be finite and in [0,1]"),
            Self::InvalidBudget => write!(f, "learning budgets must be finite and non-negative"),
            Self::InvalidNoveltyWeight => write!(f, "novelty weight must be finite and in [0,1]"),
            Self::DuplicateItemId { item_id } => write!(f, "duplicate knowledge item id {item_id}"),
            Self::EmptyLearnerTaskFamily => write!(f, "learner task family must be non-empty"),
            Self::DuplicateContextFit { item_id } => write!(f, "duplicate learner context-fit entry for item {item_id}"),
            Self::InvalidContextFit { item_id } => write!(f, "learner context fit for item {item_id} must be finite and in [0,1]"),
            Self::UnknownContextFitItem { item_id } => write!(f, "learner context fit references unknown item {item_id}"),
        }
    }
}
impl std::error::Error for TeachingError {}

#[derive(Debug, Clone, PartialEq)]
pub struct ItemOutcome {
    pub item_id: u64,
    pub lineage_id: u64,
    pub selected_for_transmission: bool,
    pub attempted_individually: bool,
    pub observed: bool,
    pub acquired: bool,
    pub integrated: bool,
    pub correctly_executed: bool,
    pub retained: bool,
    pub transferred_to_held_out_context: bool,
    pub learning_cost_spent: f64,
    pub realized_benefit: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TeachingEpisode {
    pub mode: LearningMode,
    pub items: Vec<ItemOutcome>,
    pub offered_count: usize,
    pub attempted_item_count: usize,
    pub observed_count: usize,
    pub acquired_count: usize,
    pub integrated_count: usize,
    pub correct_execution_count: usize,
    pub retained_count: usize,
    pub transfer_success_count: usize,
    pub false_adoption_count: usize,
    pub unique_lineages_offered: usize,
    pub total_learning_cost: f64,
    pub realized_benefit: f64,
    pub unused_budget: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn bernoulli(&mut self, probability: f64) -> bool {
        self.uniform() < probability
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }
}

fn score_item(
    item: &KnowledgeItem,
    learner: &LearnerProfile,
    policy: TeachingPolicy,
    lineage_counts: &[(u64, usize)],
    total_knowledgeable: usize,
) -> f64 {
    let fit = learner.context_fit(item.item_id);
    let evidence_score = item.declared_benefit
        * item.evidence_reliability
        * (1.0 - item.uncertainty);
    match policy {
        TeachingPolicy::Random => 0.0,
        TeachingPolicy::DeclaredBenefit => item.declared_benefit,
        TeachingPolicy::EvidenceCalibrated => evidence_score,
        TeachingPolicy::LearnerBenefit => evidence_score * fit,
        TeachingPolicy::DiversityAware { novelty_weight } => {
            let count = lineage_counts.iter()
                .find(|(lineage, _)| *lineage == item.lineage_id)
                .map(|(_, count)| *count)
                .unwrap_or(0);
            let rarity = if total_knowledgeable == 0 {
                0.0
            } else {
                1.0 - count as f64 / total_knowledgeable as f64
            };
            evidence_score * fit + novelty_weight * rarity
        }
    }
}

fn validate_item_set(items: &[KnowledgeItem]) -> Result<(), TeachingError> {
    let mut seen = std::collections::BTreeSet::new();
    for item in items {
        item.validate()?;
        if !seen.insert(item.item_id) {
            return Err(TeachingError::DuplicateItemId { item_id: item.item_id });
        }
    }
    Ok(())
}

/// Select an ordered packet that fits the teacher's bandwidth. The policy
/// only reads declared metadata; evaluator-only true benefit/validity are
/// intentionally excluded from this function.
pub fn select_teaching_packet(
    items: &[KnowledgeItem],
    learner: &LearnerProfile,
    policy: TeachingPolicy,
    bandwidth_budget: f64,
    seed: u64,
) -> Result<Vec<usize>, TeachingError> {
    validate_item_set(items)?;
    learner.validate(items)?;
    if !bandwidth_budget.is_finite() || bandwidth_budget < 0.0 {
        return Err(TeachingError::InvalidBudget);
    }
    if let TeachingPolicy::DiversityAware { novelty_weight } = policy {
        if !novelty_weight.is_finite() || !(0.0..=1.0).contains(&novelty_weight) {
            return Err(TeachingError::InvalidNoveltyWeight);
        }
    }

    let mut lineage_counts = Vec::<(u64, usize)>::new();
    for item in items.iter().filter(|item| item.evidence_reliability > 0.0) {
        if let Some((_, count)) = lineage_counts.iter_mut().find(|(id, _)| *id == item.lineage_id) {
            *count += 1;
        } else {
            lineage_counts.push((item.lineage_id, 1));
        }
    }
    let total_knowledgeable: usize = lineage_counts.iter().map(|(_, count)| *count).sum();

    let mut ranked: Vec<(usize, f64)> = items.iter().enumerate()
        .map(|(index, item)| (index, score_item(item, learner, policy, &lineage_counts, total_knowledgeable)))
        .collect();

    if matches!(policy, TeachingPolicy::Random) {
        let mut rng = DeterministicRng::new(seed);
        for i in (1..ranked.len()).rev() {
            let j = rng.index(i + 1);
            ranked.swap(i, j);
        }
    } else {
        ranked.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score.total_cmp(left_score)
                .then_with(|| items[*left_index].item_id.cmp(&items[*right_index].item_id))
        });
    }

    let mut remaining = bandwidth_budget;
    let mut selected = Vec::new();
    for (index, _) in ranked {
        let cost = items[index].learning_cost;
        if cost <= remaining + f64::EPSILON {
            selected.push(index);
            remaining -= cost;
        }
    }
    Ok(selected)
}

fn evaluate_one(
    item: &KnowledgeItem,
    selected: bool,
    mode: LearningMode,
    learner: &LearnerProfile,
    config: TeachingConfig,
    rng: &mut DeterministicRng,
) -> ItemOutcome {
    let mut outcome = ItemOutcome {
        item_id: item.item_id,
        lineage_id: item.lineage_id,
        selected_for_transmission: matches!(mode, LearningMode::Social { .. }) && selected,
        attempted_individually: matches!(mode, LearningMode::IndividualOnly) && selected,
        observed: false,
        acquired: false,
        integrated: false,
        correctly_executed: false,
        retained: false,
        transferred_to_held_out_context: false,
        learning_cost_spent: 0.0,
        realized_benefit: 0.0,
    };

    match mode {
        LearningMode::Social { .. } => {
            if !selected {
                return outcome;
            }
            outcome.learning_cost_spent = item.learning_cost;
            outcome.observed = rng.bernoulli(config.observation_probability);
            if !outcome.observed {
                return outcome;
            }
            let fit = learner.context_fit(item.item_id);
            let mismatch = if item.task_family == learner.task_family {
                0.0
            } else {
                config.task_family_mismatch_penalty
            };
            let learnability = (1.0 - 0.8 * item.difficulty).clamp(0.1, 1.0);
            let acquisition_probability = (
                config.acquisition_base_probability
                    * item.evidence_reliability
                    * (1.0 - item.uncertainty)
                    * (0.5 + 0.5 * fit)
                    * learnability
                    * (1.0 - mismatch)
            ).clamp(0.0, 1.0);
            outcome.acquired = rng.bernoulli(acquisition_probability);
        }
        LearningMode::IndividualOnly => {
            if !selected {
                return outcome;
            }
            outcome.learning_cost_spent = item.learning_cost;
            let discovery_probability = (
                config.individual_discovery_base_probability
                    * (1.0 - 0.8 * item.difficulty)
                    * (1.0 - item.uncertainty)
            ).clamp(0.0, 1.0);
            outcome.acquired = rng.bernoulli(discovery_probability);
            outcome.observed = false;
        }
    }

    if !outcome.acquired {
        return outcome;
    }
    outcome.integrated = rng.bernoulli(config.integration_probability);
    if !outcome.integrated {
        return outcome;
    }

    outcome.correctly_executed = rng.bernoulli(item.validity);
    if outcome.correctly_executed {
        outcome.realized_benefit = item.true_benefit;
    }
    outcome.retained = rng.bernoulli(config.retention_probability);
    if outcome.retained {
        outcome.transferred_to_held_out_context = rng.bernoulli(item.transferability);
    }
    outcome
}

/// Execute one social-teaching episode or a matched individual-only control.
/// The evaluator receives true benefit and validity; the selection policy
/// never sees those fields.
pub fn run_teaching_episode(
    items: &[KnowledgeItem],
    learner: &LearnerProfile,
    mode: LearningMode,
    config: TeachingConfig,
    seed: u64,
) -> Result<TeachingEpisode, TeachingError> {
    config.validate()?;
    validate_item_set(items)?;
    learner.validate(items)?;
    let budget = match mode {
        LearningMode::Social { .. } => config.bandwidth_budget,
        LearningMode::IndividualOnly => config.individual_learning_budget,
    };
    let selected_indices = match mode {
        LearningMode::Social { policy } => {
            select_teaching_packet(items, learner, policy, budget, seed)?
        }
        LearningMode::IndividualOnly => {
            let mut indices: Vec<usize> = (0..items.len()).collect();
            indices.sort_by_key(|index| items[*index].item_id);
            let mut remaining = budget;
            indices.into_iter().filter(|index| {
                let cost = items[*index].learning_cost;
                if cost <= remaining + f64::EPSILON {
                    remaining -= cost;
                    true
                } else {
                    false
                }
            }).collect()
        }
    };

    let selected_set: std::collections::BTreeSet<usize> =
        selected_indices.iter().copied().collect();
    let mut rng = DeterministicRng::new(seed ^ 0xA11C_E5ED_5EED);
    let mut outcomes = Vec::with_capacity(items.len());
    let mut remaining = budget;
    for (index, item) in items.iter().enumerate() {
        let selected = selected_set.contains(&index);
        let outcome = evaluate_one(item, selected, mode, learner, config, &mut rng);
        if selected {
            remaining = (remaining - outcome.learning_cost_spent).max(0.0);
        }
        outcomes.push(outcome);
    }

    let offered_count = outcomes.iter().filter(|outcome| outcome.selected_for_transmission).count();
    let attempted_item_count = outcomes.iter().filter(|outcome| {
        outcome.selected_for_transmission || outcome.attempted_individually
    }).count();
    let observed_count = outcomes.iter().filter(|outcome| outcome.observed).count();
    let acquired_count = outcomes.iter().filter(|outcome| outcome.acquired).count();
    let integrated_count = outcomes.iter().filter(|outcome| outcome.integrated).count();
    let correct_execution_count = outcomes.iter().filter(|outcome| outcome.correctly_executed).count();
    let retained_count = outcomes.iter().filter(|outcome| outcome.retained).count();
    let transfer_success_count = outcomes.iter().filter(|outcome| outcome.transferred_to_held_out_context).count();
    let false_adoption_count = outcomes.iter().filter(|outcome| {
        outcome.integrated && !outcome.correctly_executed
    }).count();
    let total_learning_cost: f64 = outcomes.iter().map(|outcome| outcome.learning_cost_spent).sum();
    let realized_benefit: f64 = outcomes.iter().map(|outcome| outcome.realized_benefit).sum();
    let unique_lineages_offered = {
        let mut lineages = std::collections::BTreeSet::new();
        if matches!(mode, LearningMode::Social { .. }) {
            for outcome in outcomes.iter().filter(|outcome| outcome.selected_for_transmission) {
                lineages.insert(outcome.lineage_id);
            }
        }
        lineages.len()
    };

    Ok(TeachingEpisode {
        mode,
        items: outcomes,
        offered_count,
        attempted_item_count,
        observed_count,
        acquired_count,
        integrated_count,
        correct_execution_count,
        retained_count,
        transfer_success_count,
        false_adoption_count,
        unique_lineages_offered,
        total_learning_cost,
        realized_benefit,
        unused_budget: (budget - total_learning_cost).max(0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items() -> Vec<KnowledgeItem> {
        vec![
            KnowledgeItem {
                item_id: 1, lineage_id: 10, task_family: "navigation".into(),
                true_benefit: 8.0, declared_benefit: 9.0,
                evidence_reliability: 0.9, uncertainty: 0.1,
                learning_cost: 2.0, difficulty: 0.10, validity: 0.95, transferability: 0.8,
            },
            KnowledgeItem {
                item_id: 2, lineage_id: 10, task_family: "navigation".into(),
                true_benefit: 2.0, declared_benefit: 10.0,
                evidence_reliability: 0.25, uncertainty: 0.7,
                learning_cost: 2.0, difficulty: 0.80, validity: 0.4, transferability: 0.3,
            },
            KnowledgeItem {
                item_id: 3, lineage_id: 20, task_family: "navigation".into(),
                true_benefit: 6.0, declared_benefit: 7.0,
                evidence_reliability: 0.95, uncertainty: 0.05,
                learning_cost: 3.0, difficulty: 0.20, validity: 0.98, transferability: 0.9,
            },
            KnowledgeItem {
                item_id: 4, lineage_id: 30, task_family: "ecology".into(),
                true_benefit: 5.0, declared_benefit: 8.0,
                evidence_reliability: 0.8, uncertainty: 0.2,
                learning_cost: 4.0, difficulty: 0.40, validity: 0.9, transferability: 0.7,
            },
        ]
    }

    fn learner() -> LearnerProfile {
        LearnerProfile {
            task_family: "navigation".into(),
            context_fit_by_item: vec![(1, 0.95), (2, 0.9), (3, 0.8), (4, 0.2)],
        }
    }

    #[test]
    fn evidence_calibrated_teacher_avoids_high_claim_low_evidence_item() {
        let all = items();
        let selected = select_teaching_packet(
            &all,
            &learner(),
            TeachingPolicy::EvidenceCalibrated,
            2.0,
            1,
        ).unwrap();
        assert_eq!(selected, vec![0]);
        assert!(!selected.contains(&1));
    }

    #[test]
    fn learner_benefit_policy_uses_context_fit_without_access_to_true_benefit() {
        let all = items();
        let selected = select_teaching_packet(
            &all,
            &learner(),
            TeachingPolicy::LearnerBenefit,
            3.0,
            1,
        ).unwrap();
        assert_eq!(selected, vec![0]);
    }

    #[test]
    fn diversity_bonus_can_select_a_distinct_lineage() {
        let all = items();
        let selected = select_teaching_packet(
            &all,
            &learner(),
            TeachingPolicy::DiversityAware { novelty_weight: 1.0 },
            5.0,
            1,
        ).unwrap();
        let lineages: std::collections::BTreeSet<u64> =
            selected.iter().map(|index| all[*index].lineage_id).collect();
        assert!(lineages.len() >= 2);
    }

    #[test]
    fn packet_never_exceeds_bandwidth_and_preserves_rank_order() {
        let all = items();
        let selected = select_teaching_packet(
            &all,
            &learner(),
            TeachingPolicy::DeclaredBenefit,
            4.0,
            1,
        ).unwrap();
        let spent: f64 = selected.iter().map(|index| all[*index].learning_cost).sum();
        assert!(spent <= 4.0 + f64::EPSILON);
        assert_eq!(selected.first().copied(), Some(1));
    }

    #[test]
    fn stage_counts_never_inflate_acquisition_or_execution() {
        let mut all = items();
        for item in &mut all {
            item.evidence_reliability = 1.0;
            item.uncertainty = 0.0;
            item.difficulty = 0.0;
        }
        let mut learner = learner();
        for (_, fit) in &mut learner.context_fit_by_item {
            *fit = 1.0;
        }
        let config = TeachingConfig {
            bandwidth_budget: 20.0,
            observation_probability: 1.0,
            acquisition_base_probability: 1.0,
            integration_probability: 1.0,
            retention_probability: 1.0,
            ..TeachingConfig::default()
        };
        let episode = run_teaching_episode(
            &all,
            &learner,
            LearningMode::Social { policy: TeachingPolicy::EvidenceCalibrated },
            config,
            42,
        ).unwrap();
        assert_eq!(episode.offered_count, episode.observed_count);
        assert_eq!(episode.observed_count, episode.acquired_count);
        assert_eq!(episode.acquired_count, episode.integrated_count);
        assert!(episode.correct_execution_count <= episode.integrated_count);
        assert_eq!(episode.retained_count, episode.integrated_count);
        assert!(episode.transfer_success_count <= episode.retained_count);
    }

    #[test]
    fn individual_control_is_budget_matched_and_has_no_teacher_observations() {
        let all = items();
        let config = TeachingConfig {
            bandwidth_budget: 5.0,
            individual_learning_budget: 5.0,
            individual_discovery_base_probability: 1.0,
            integration_probability: 1.0,
            retention_probability: 1.0,
            ..TeachingConfig::default()
        };
        let episode = run_teaching_episode(
            &all,
            &learner(),
            LearningMode::IndividualOnly,
            config,
            9,
        ).unwrap();
        assert_eq!(episode.offered_count, 0);
        assert_eq!(episode.attempted_item_count, episode.items.iter().filter(|item| item.attempted_individually).count());
        assert!(episode.items.iter().all(|item| !item.selected_for_transmission));
        assert_eq!(episode.observed_count, 0);
        assert!(episode.items.iter().all(|item| {
            item.attempted_individually || item.learning_cost_spent == 0.0
        }));
        assert!(episode.total_learning_cost <= config.individual_learning_budget);
    }

    #[test]
    fn individual_only_path_rejects_duplicate_ids_and_invalid_context_fit() {
        let mut duplicate = items();
        duplicate[1].item_id = duplicate[0].item_id;
        assert_eq!(
            run_teaching_episode(
                &duplicate,
                &learner(),
                LearningMode::IndividualOnly,
                TeachingConfig::default(),
                1,
            ),
            Err(TeachingError::DuplicateItemId { item_id: duplicate[0].item_id })
        );

        let mut invalid_learner = learner();
        invalid_learner.context_fit_by_item[0].1 = f64::NAN;
        assert_eq!(
            run_teaching_episode(
                &items(),
                &invalid_learner,
                LearningMode::IndividualOnly,
                TeachingConfig::default(),
                1,
            ),
            Err(TeachingError::InvalidContextFit { item_id: invalid_learner.context_fit_by_item[0].0 })
        );
    }

    #[test]
    fn fixed_seed_is_reproducible_and_duplicate_ids_fail_closed() {
        let all = items();
        let config = TeachingConfig::default();
        let mode = LearningMode::Social { policy: TeachingPolicy::EvidenceCalibrated };
        assert_eq!(
            run_teaching_episode(&all, &learner(), mode, config, 2026).unwrap(),
            run_teaching_episode(&all, &learner(), mode, config, 2026).unwrap()
        );
        let mut duplicate = all.clone();
        duplicate[1].item_id = duplicate[0].item_id;
        assert_eq!(
            select_teaching_packet(&duplicate, &learner(), TeachingPolicy::Random, 10.0, 0),
            Err(TeachingError::DuplicateItemId { item_id: duplicate[0].item_id })
        );
    }
}
