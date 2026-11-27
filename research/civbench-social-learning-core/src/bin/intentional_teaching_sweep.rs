// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-seed intentional-teaching comparison under matched information budgets.
//!
//! Synthetic mechanism study only. This is not the analysis pipeline for any
//! cited human experiment.

use civbench_social_learning_core::intentional_teaching::{
    run_teaching_episode, KnowledgeItem, LearnerProfile, LearningMode, TeachingConfig,
    TeachingPolicy,
};

const REPLICATES: u64 = 256;

#[derive(Clone, Copy)]
enum Condition {
    Individual,
    Random,
    DeclaredBenefit,
    EvidenceCalibrated,
    LearnerBenefit,
    DiversityAware,
}

impl Condition {
    fn label(self) -> &'static str {
        match self {
            Self::Individual => "individual_only",
            Self::Random => "social_random",
            Self::DeclaredBenefit => "social_declared_benefit",
            Self::EvidenceCalibrated => "social_evidence_calibrated",
            Self::LearnerBenefit => "social_learner_context",
            Self::DiversityAware => "social_diversity_aware",
        }
    }
}

#[derive(Default)]
struct OnlineStats {
    n: u64,
    sum: f64,
    sum_squares: f64,
}

impl OnlineStats {
    fn push(&mut self, value: f64) {
        self.n += 1;
        self.sum += value;
        self.sum_squares += value * value;
    }

    fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }

    fn ci95_half_width(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let n = self.n as f64;
        let centered = (self.sum_squares - self.sum * self.sum / n).max(0.0);
        let sample_sd = (centered / (n - 1.0)).sqrt();
        Some(1.96 * sample_sd / n.sqrt())
    }
}

#[derive(Default)]
struct Summary {
    attempted_items: OnlineStats,
    offered_items: OnlineStats,
    observations: OnlineStats,
    acquisitions: OnlineStats,
    integrations: OnlineStats,
    correct_executions: OnlineStats,
    false_adoptions: OnlineStats,
    retention: OnlineStats,
    transfer: OnlineStats,
    lineages_offered: OnlineStats,
    cost: OnlineStats,
    benefit: OnlineStats,
    unused_budget: OnlineStats,
}

fn fmt(value: Option<f64>) -> String {
    value.map(|number| format!("{number:.6}")).unwrap_or_default()
}

fn items() -> Vec<KnowledgeItem> {
    // Deliberately includes overclaimed low-quality items, context-mismatched
    // skills, multiple lineages and different learning costs.
    let raw = [
        (1, 10, "navigation", 8.0, 9.0, 0.90, 0.10, 2.0, 0.10, 0.95, 0.80),
        (2, 10, "navigation", 2.0, 10.0, 0.25, 0.70, 2.0, 0.80, 0.40, 0.30),
        (3, 20, "navigation", 6.0, 7.0, 0.95, 0.05, 3.0, 0.20, 0.98, 0.90),
        (4, 30, "ecology", 5.0, 8.0, 0.80, 0.20, 4.0, 0.40, 0.90, 0.70),
        (5, 30, "navigation", 4.0, 6.0, 0.75, 0.20, 1.0, 0.35, 0.85, 0.75),
        (6, 40, "navigation", 7.0, 7.5, 0.90, 0.10, 4.0, 0.65, 0.95, 0.85),
        (7, 50, "ecology", 9.0, 9.5, 0.30, 0.60, 2.5, 0.75, 0.35, 0.40),
        (8, 60, "navigation", 3.0, 4.0, 0.95, 0.05, 1.5, 0.05, 0.99, 0.95),
        (9, 60, "navigation", 5.0, 5.0, 0.85, 0.15, 3.0, 0.45, 0.90, 0.90),
        (10, 70, "navigation", 6.5, 6.0, 0.70, 0.15, 2.0, 0.30, 0.92, 0.80),
        (11, 80, "ecology", 7.5, 8.5, 0.88, 0.12, 3.5, 0.55, 0.93, 0.70),
        (12, 90, "navigation", 2.5, 3.0, 0.98, 0.02, 1.0, 0.15, 0.99, 0.98),
    ];
    raw.into_iter().map(|(id, lineage, family, true_benefit, declared_benefit,
                          evidence_reliability, uncertainty, learning_cost, difficulty,
                          validity, transferability)| {
        KnowledgeItem {
            item_id: id,
            lineage_id: lineage,
            task_family: family.to_owned(),
            true_benefit,
            declared_benefit,
            evidence_reliability,
            uncertainty,
            learning_cost,
            difficulty,
            validity,
            transferability,
        }
    }).collect()
}

fn learner() -> LearnerProfile {
    LearnerProfile {
        task_family: "navigation".to_owned(),
        context_fit_by_item: vec![
            (1, 0.95), (2, 0.90), (3, 0.80), (4, 0.20),
            (5, 0.85), (6, 0.95), (7, 0.20), (8, 0.90),
            (9, 0.85), (10, 0.80), (11, 0.30), (12, 0.95),
        ],
    }
}

fn policy(condition: Condition) -> Option<TeachingPolicy> {
    match condition {
        Condition::Individual => None,
        Condition::Random => Some(TeachingPolicy::Random),
        Condition::DeclaredBenefit => Some(TeachingPolicy::DeclaredBenefit),
        Condition::EvidenceCalibrated => Some(TeachingPolicy::EvidenceCalibrated),
        Condition::LearnerBenefit => Some(TeachingPolicy::LearnerBenefit),
        Condition::DiversityAware => Some(TeachingPolicy::DiversityAware { novelty_weight: 0.25 }),
    }
}

fn run_condition(condition: Condition, replicate: u64) -> Summary {
    let items = items();
    let learner = learner();
    let config = TeachingConfig {
        bandwidth_budget: 8.0,
        individual_learning_budget: 8.0,
        observation_probability: 0.95,
        acquisition_base_probability: 0.90,
        integration_probability: 0.85,
        retention_probability: 0.90,
        individual_discovery_base_probability: 0.12,
        task_family_mismatch_penalty: 0.75,
    };
    let mode = match policy(condition) {
        Some(policy) => LearningMode::Social { policy },
        None => LearningMode::IndividualOnly,
    };

    let episode = run_teaching_episode(
        &items,
        &learner,
        mode,
        config,
        0x1A7E_7100_u64.wrapping_add(replicate),
    ).expect("fixed teaching benchmark fixture must validate");

    let denominator = episode.attempted_item_count.max(1) as f64;
    Summary {
        attempted_items: one(episode.attempted_item_count as f64),
        offered_items: one(episode.offered_count as f64),
        observations: one(episode.observed_count as f64 / denominator),
        acquisitions: one(episode.acquired_count as f64 / denominator),
        integrations: one(episode.integrated_count as f64 / denominator),
        correct_executions: one(episode.correct_execution_count as f64 / denominator),
        false_adoptions: one(episode.false_adoption_count as f64 / denominator),
        retention: one(episode.retained_count as f64 / denominator),
        transfer: one(episode.transfer_success_count as f64 / denominator),
        lineages_offered: one(episode.unique_lineages_offered as f64),
        cost: one(episode.total_learning_cost),
        benefit: one(episode.realized_benefit),
        unused_budget: one(episode.unused_budget),
    }
}

fn one(value: f64) -> OnlineStats {
    let mut stats = OnlineStats::default();
    stats.push(value);
    stats
}

fn main() {
    let conditions = [
        Condition::Individual,
        Condition::Random,
        Condition::DeclaredBenefit,
        Condition::EvidenceCalibrated,
        Condition::LearnerBenefit,
        Condition::DiversityAware,
    ];
    println!(
        "condition,replicates,attempted_items,teacher_offered_items,observation_rate,acquisition_rate,integration_rate,correct_execution_rate,correct_execution_ci95,false_adoption_rate,retention_rate,held_out_transfer_rate,unique_lineages_offered,learning_cost,realized_benefit,realized_benefit_ci95,unused_budget"
    );

    for condition in conditions {
        let mut aggregate = Summary::default();
        for replicate in 0..REPLICATES {
            let result = run_condition(condition, replicate);
            aggregate.attempted_items.push(result.attempted_items.mean().unwrap_or(0.0));
            aggregate.offered_items.push(result.offered_items.mean().unwrap_or(0.0));
            aggregate.observations.push(result.observations.mean().unwrap_or(0.0));
            aggregate.acquisitions.push(result.acquisitions.mean().unwrap_or(0.0));
            aggregate.integrations.push(result.integrations.mean().unwrap_or(0.0));
            aggregate.correct_executions.push(result.correct_executions.mean().unwrap_or(0.0));
            aggregate.false_adoptions.push(result.false_adoptions.mean().unwrap_or(0.0));
            aggregate.retention.push(result.retention.mean().unwrap_or(0.0));
            aggregate.transfer.push(result.transfer.mean().unwrap_or(0.0));
            aggregate.lineages_offered.push(result.lineages_offered.mean().unwrap_or(0.0));
            aggregate.cost.push(result.cost.mean().unwrap_or(0.0));
            aggregate.benefit.push(result.benefit.mean().unwrap_or(0.0));
            aggregate.unused_budget.push(result.unused_budget.mean().unwrap_or(0.0));
        }

        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            condition.label(),
            REPLICATES,
            fmt(aggregate.attempted_items.mean()),
            fmt(aggregate.offered_items.mean()),
            fmt(aggregate.observations.mean()),
            fmt(aggregate.acquisitions.mean()),
            fmt(aggregate.integrations.mean()),
            fmt(aggregate.correct_executions.mean()),
            fmt(aggregate.correct_executions.ci95_half_width()),
            fmt(aggregate.false_adoptions.mean()),
            fmt(aggregate.retention.mean()),
            fmt(aggregate.transfer.mean()),
            fmt(aggregate.lineages_offered.mean()),
            fmt(aggregate.cost.mean()),
            fmt(aggregate.benefit.mean()),
            fmt(aggregate.benefit.ci95_half_width()),
            fmt(aggregate.unused_budget.mean()),
        );
    }
}
