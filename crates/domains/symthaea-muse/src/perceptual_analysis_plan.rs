// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1B: frozen analysis and simulation-based sample-size planning
//! contracts for the first blinded perceptual study.
//!
//! This module does not simulate participants, fit a GLMM, or authorize human
//! collection. It defines the exact inputs/outputs that a separately qualified
//! planning runner must produce before P1A can bind a participant count.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_study_protocol::{
        CHANCE_PROBABILITY, MEL003_FIXED_SEEDS, PerceptualTaskV1,
        PrimaryAnalysisModelV1, SecondaryMultiplicityPolicyV1,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PERCEPTUAL_ANALYSIS_SPEC_VERSION: &str = "mel003-perceptual-analysis-spec-v1";
pub const PERCEPTUAL_POWER_PLAN_VERSION: &str = "mel003-perceptual-power-plan-v1";
pub const MIN_POWER_SIMULATION_REPLICATES: usize = 5_000;
pub const MIN_POWER_HETEROGENEITY_SCENARIOS: usize = 2;
/// 95th percentile of the standard normal for a one-sided lower bound.
pub const POWER_MONTE_CARLO_Z_95_ONE_SIDED: f64 = 1.644_853_626_951_472_2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryLinkV1 {
    Logit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFailurePolicyV1 {
    /// Convergence/singularity failure makes the registered analysis
    /// inconclusive. It may not silently switch to a simpler model after data
    /// inspection.
    InconclusiveNoAutomaticSubstitution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecondaryConditioningPolicyV1 {
    /// Directional judgments are analyzed on their own registered trials and
    /// are never restricted to participants/items that succeeded on ABX.
    IndependentTrialsNoConditioningOnPrimary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptualAnalysisExecutionIdentityV1 {
    pub source_revision: String,
    pub analysis_program_sha256: String,
    pub environment_sha256: String,
    pub model_engine: String,
    pub model_engine_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenPerceptualAnalysisSpecV1 {
    pub spec_version: String,
    pub primary_task: PerceptualTaskV1,
    pub key_secondary_task: PerceptualTaskV1,
    pub primary_model: PrimaryAnalysisModelV1,
    pub link: BinaryLinkV1,
    pub chance_probability: f64,
    pub alpha: f64,
    pub confidence_level: f64,
    pub participant_random_intercept_required: bool,
    pub item_random_intercept_required: bool,
    pub registered_item_seeds: [u64; 8],
    pub primary_alternative_greater_than_chance: bool,
    pub secondary_conditioning: SecondaryConditioningPolicyV1,
    pub secondary_multiplicity: SecondaryMultiplicityPolicyV1,
    pub report_fixed_effect_estimate: bool,
    pub report_odds_ratio: bool,
    pub report_confidence_interval: bool,
    pub report_participant_variance: bool,
    pub report_item_variance: bool,
    pub report_participant_summary: bool,
    pub report_item_summary: bool,
    pub model_failure_policy: ModelFailurePolicyV1,
    pub execution: PerceptualAnalysisExecutionIdentityV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualAnalysisSpecIssueV1 {
    WrongVersion,
    WrongPrimaryTask,
    WrongSecondaryTask,
    WrongPrimaryModel,
    WrongLink,
    WrongChanceProbability,
    InvalidAlpha,
    InvalidConfidenceLevel,
    MissingParticipantRandomIntercept,
    MissingItemRandomIntercept,
    WrongSeedPanel,
    WrongPrimaryAlternative,
    WrongSecondaryConditioning,
    WrongMultiplicityPolicy,
    MissingRequiredReportField { field: String },
    WrongFailurePolicy,
    InvalidDigest { field: String },
    EmptyExecutionField { field: String },
}

impl FrozenPerceptualAnalysisSpecV1 {
    pub fn validate(&self) -> Vec<PerceptualAnalysisSpecIssueV1> {
        let mut issues = Vec::new();
        if self.spec_version != PERCEPTUAL_ANALYSIS_SPEC_VERSION {
            issues.push(PerceptualAnalysisSpecIssueV1::WrongVersion);
        }
        if self.primary_task != PerceptualTaskV1::AbxDiscrimination {
            issues.push(PerceptualAnalysisSpecIssueV1::WrongPrimaryTask);
        }
        if self.key_secondary_task != PerceptualTaskV1::DirectionalRearticulation2Afc {
            issues.push(PerceptualAnalysisSpecIssueV1::WrongSecondaryTask);
        }
        if self.primary_model
            != PrimaryAnalysisModelV1::CrossedParticipantItemLogisticRandomIntercepts
        {
            issues.push(PerceptualAnalysisSpecIssueV1::WrongPrimaryModel);
        }
        if self.link != BinaryLinkV1::Logit {
            issues.push(PerceptualAnalysisSpecIssueV1::WrongLink);
        }
        if self.chance_probability != CHANCE_PROBABILITY {
            issues.push(PerceptualAnalysisSpecIssueV1::WrongChanceProbability);
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha >= 0.5 {
            issues.push(PerceptualAnalysisSpecIssueV1::InvalidAlpha);
        }
        if !self.confidence_level.is_finite()
            || self.confidence_level <= 0.5
            || self.confidence_level >= 1.0
        {
            issues.push(PerceptualAnalysisSpecIssueV1::InvalidConfidenceLevel);
        }
        if !self.participant_random_intercept_required {
            issues.push(PerceptualAnalysisSpecIssueV1::MissingParticipantRandomIntercept);
        }
        if !self.item_random_intercept_required {
            issues.push(PerceptualAnalysisSpecIssueV1::MissingItemRandomIntercept);
        }
        if self.registered_item_seeds != MEL003_FIXED_SEEDS {
            issues.push(PerceptualAnalysisSpecIssueV1::WrongSeedPanel);
        }
        if !self.primary_alternative_greater_than_chance {
            issues.push(PerceptualAnalysisSpecIssueV1::WrongPrimaryAlternative);
        }
        if self.secondary_conditioning
            != SecondaryConditioningPolicyV1::IndependentTrialsNoConditioningOnPrimary
        {
            issues.push(PerceptualAnalysisSpecIssueV1::WrongSecondaryConditioning);
        }
        if self.secondary_multiplicity
            != SecondaryMultiplicityPolicyV1::HolmWithinRegisteredSecondaryFamily
        {
            issues.push(PerceptualAnalysisSpecIssueV1::WrongMultiplicityPolicy);
        }
        for (field, present) in [
            ("fixed_effect_estimate", self.report_fixed_effect_estimate),
            ("odds_ratio", self.report_odds_ratio),
            ("confidence_interval", self.report_confidence_interval),
            ("participant_variance", self.report_participant_variance),
            ("item_variance", self.report_item_variance),
            ("participant_summary", self.report_participant_summary),
            ("item_summary", self.report_item_summary),
        ] {
            if !present {
                issues.push(PerceptualAnalysisSpecIssueV1::MissingRequiredReportField {
                    field: field.into(),
                });
            }
        }
        if self.model_failure_policy != ModelFailurePolicyV1::InconclusiveNoAutomaticSubstitution {
            issues.push(PerceptualAnalysisSpecIssueV1::WrongFailurePolicy);
        }
        for (field, digest) in [
            (
                "execution.analysis_program_sha256",
                self.execution.analysis_program_sha256.as_str(),
            ),
            (
                "execution.environment_sha256",
                self.execution.environment_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                issues.push(PerceptualAnalysisSpecIssueV1::InvalidDigest {
                    field: field.into(),
                });
            }
        }
        for (field, value) in [
            ("execution.source_revision", self.execution.source_revision.as_str()),
            ("execution.model_engine", self.execution.model_engine.as_str()),
            (
                "execution.model_engine_version",
                self.execution.model_engine_version.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                issues.push(PerceptualAnalysisSpecIssueV1::EmptyExecutionField {
                    field: field.into(),
                });
            }
        }
        issues
    }

    pub fn spec_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerEffectBasisV1 {
    /// The smallest effect of interest is declared on the human response scale,
    /// never derived from C5/C6 waveform proxy magnitude.
    HumanPerceptualSesoI,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerSelectionRuleV1 {
    /// A candidate N passes only when the one-sided 95% Wilson lower bound on
    /// simulated rejection probability reaches target power in every registered
    /// participant/item heterogeneity scenario.
    SmallestRobustNByOneSided95WilsonLowerBound,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerHeterogeneityScenarioV1 {
    pub scenario_id: String,
    /// Standard deviations on the logit scale. The human-scale SESOI is common
    /// across scenarios and lives on the plan itself; scenarios vary only the
    /// nuisance heterogeneity assumptions.
    pub participant_intercept_sd: f64,
    pub item_intercept_sd: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowerSimulationRunnerIdentityV1 {
    pub runner_version: String,
    pub source_revision: String,
    pub binary_sha256: String,
    pub environment_sha256: String,
    pub rng_algorithm: String,
    pub rng_seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerGridCellV1 {
    pub scenario_id: String,
    pub completed_participants: usize,
    pub simulation_replicates: usize,
    /// Number of simulated studies whose registered primary analysis rejected
    /// chance in the preregistered greater-than-chance direction.
    pub primary_rejection_count: usize,
}

impl PowerGridCellV1 {
    pub fn estimated_power(&self) -> Option<f64> {
        if self.simulation_replicates == 0
            || self.primary_rejection_count > self.simulation_replicates
        {
            None
        } else {
            Some(self.primary_rejection_count as f64 / self.simulation_replicates as f64)
        }
    }

    pub fn one_sided_95_wilson_lower_bound(&self) -> Option<f64> {
        let p = self.estimated_power()?;
        let n = self.simulation_replicates as f64;
        let z = POWER_MONTE_CARLO_Z_95_ONE_SIDED;
        let z2 = z * z;
        let denominator = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denominator;
        let margin = z / denominator * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
        Some((center - margin).max(0.0))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenPerceptualPowerPlanV1 {
    pub plan_version: String,
    pub analysis_spec_sha256: String,
    pub effect_basis: PowerEffectBasisV1,
    /// One human-scale smallest effect of interest shared across every power
    /// scenario. It may not be substituted with an acoustic proxy magnitude.
    pub primary_sesoi_success_probability: f64,
    pub item_count: usize,
    pub target_power: f64,
    pub alpha: f64,
    pub selection_rule: PowerSelectionRuleV1,
    pub minimum_simulation_replicates: usize,
    pub heterogeneity_scenarios: Vec<PowerHeterogeneityScenarioV1>,
    /// Strictly increasing candidate N values considered before selection.
    pub candidate_completed_participants: Vec<usize>,
    pub grid: Vec<PowerGridCellV1>,
    /// Must be the smallest registered N passing the frozen robust selection
    /// rule in every registered heterogeneity scenario.
    pub selected_completed_participants: usize,
    pub anticipated_noncompletion_rate: f64,
    pub maximum_enrolled_participants: usize,
    pub outcome_adaptive_stopping_allowed: bool,
    pub runner: PowerSimulationRunnerIdentityV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualPowerPlanIssueV1 {
    WrongVersion,
    InvalidAnalysisSpec,
    AnalysisSpecSerializationFailed,
    AnalysisSpecDigestMismatch,
    WrongEffectBasis,
    InvalidHumanSesoI,
    WrongItemCount,
    InvalidTargetPower,
    AlphaMismatch,
    WrongSelectionRule,
    TooFewSimulationReplicates,
    TooFewHeterogeneityScenarios { found: usize, required: usize },
    EmptyScenarioId { index: usize },
    DuplicateScenarioId { scenario_id: String },
    InvalidScenarioVariance { scenario_id: String },
    NoParticipantHeterogeneityScenario,
    NoItemHeterogeneityScenario,
    EmptyCandidateGrid,
    InvalidCandidateN { index: usize },
    CandidateGridNotStrictlyIncreasing,
    MissingGridCell { scenario_id: String, participants: usize },
    DuplicateGridCell { scenario_id: String, participants: usize },
    UnexpectedGridCell { scenario_id: String, participants: usize },
    InvalidGridCell { scenario_id: String, participants: usize },
    SelectedNNotRegistered,
    SelectedNNotSmallestRobustPassingN,
    NoRobustPassingN,
    InvalidNoncompletionRate,
    EnrollmentInflationMismatch,
    OutcomeAdaptiveStoppingAllowed,
    InvalidDigest { field: String },
    EmptyRunnerField { field: String },
}

impl FrozenPerceptualPowerPlanV1 {
    pub fn validate(
        &self,
        analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    ) -> Vec<PerceptualPowerPlanIssueV1> {
        let mut issues = Vec::new();
        if self.plan_version != PERCEPTUAL_POWER_PLAN_VERSION {
            issues.push(PerceptualPowerPlanIssueV1::WrongVersion);
        }
        if !analysis_spec.validate().is_empty() {
            issues.push(PerceptualPowerPlanIssueV1::InvalidAnalysisSpec);
        }
        match canonical_json_sha256(analysis_spec) {
            Ok(value) if value == self.analysis_spec_sha256 => {}
            Ok(_) => issues.push(PerceptualPowerPlanIssueV1::AnalysisSpecDigestMismatch),
            Err(_) => issues.push(PerceptualPowerPlanIssueV1::AnalysisSpecSerializationFailed),
        }
        if self.effect_basis != PowerEffectBasisV1::HumanPerceptualSesoI {
            issues.push(PerceptualPowerPlanIssueV1::WrongEffectBasis);
        }
        if !self.primary_sesoi_success_probability.is_finite()
            || self.primary_sesoi_success_probability <= CHANCE_PROBABILITY
            || self.primary_sesoi_success_probability >= 1.0
        {
            issues.push(PerceptualPowerPlanIssueV1::InvalidHumanSesoI);
        }
        if self.item_count != MEL003_FIXED_SEEDS.len() {
            issues.push(PerceptualPowerPlanIssueV1::WrongItemCount);
        }
        if !self.target_power.is_finite()
            || self.target_power <= 0.5
            || self.target_power >= 1.0
        {
            issues.push(PerceptualPowerPlanIssueV1::InvalidTargetPower);
        }
        if self.alpha != analysis_spec.alpha {
            issues.push(PerceptualPowerPlanIssueV1::AlphaMismatch);
        }
        if self.selection_rule != PowerSelectionRuleV1::SmallestRobustNByOneSided95WilsonLowerBound {
            issues.push(PerceptualPowerPlanIssueV1::WrongSelectionRule);
        }
        if self.minimum_simulation_replicates < MIN_POWER_SIMULATION_REPLICATES {
            issues.push(PerceptualPowerPlanIssueV1::TooFewSimulationReplicates);
        }

        if self.heterogeneity_scenarios.len() < MIN_POWER_HETEROGENEITY_SCENARIOS {
            issues.push(PerceptualPowerPlanIssueV1::TooFewHeterogeneityScenarios {
                found: self.heterogeneity_scenarios.len(),
                required: MIN_POWER_HETEROGENEITY_SCENARIOS,
            });
        }
        let mut scenario_ids = BTreeSet::new();
        let mut participant_heterogeneity_present = false;
        let mut item_heterogeneity_present = false;
        for (index, scenario) in self.heterogeneity_scenarios.iter().enumerate() {
            if scenario.scenario_id.trim().is_empty() {
                issues.push(PerceptualPowerPlanIssueV1::EmptyScenarioId { index });
            } else if !scenario_ids.insert(scenario.scenario_id.clone()) {
                issues.push(PerceptualPowerPlanIssueV1::DuplicateScenarioId {
                    scenario_id: scenario.scenario_id.clone(),
                });
            }
            if !scenario.participant_intercept_sd.is_finite()
                || scenario.participant_intercept_sd < 0.0
                || !scenario.item_intercept_sd.is_finite()
                || scenario.item_intercept_sd < 0.0
            {
                issues.push(PerceptualPowerPlanIssueV1::InvalidScenarioVariance {
                    scenario_id: scenario.scenario_id.clone(),
                });
            }
            participant_heterogeneity_present |= scenario.participant_intercept_sd > 0.0;
            item_heterogeneity_present |= scenario.item_intercept_sd > 0.0;
        }
        if !participant_heterogeneity_present {
            issues.push(PerceptualPowerPlanIssueV1::NoParticipantHeterogeneityScenario);
        }
        if !item_heterogeneity_present {
            issues.push(PerceptualPowerPlanIssueV1::NoItemHeterogeneityScenario);
        }

        if self.candidate_completed_participants.is_empty() {
            issues.push(PerceptualPowerPlanIssueV1::EmptyCandidateGrid);
        }
        let mut previous = 0usize;
        for (index, &n) in self.candidate_completed_participants.iter().enumerate() {
            if n == 0 {
                issues.push(PerceptualPowerPlanIssueV1::InvalidCandidateN { index });
            }
            if index > 0 && n <= previous {
                issues.push(PerceptualPowerPlanIssueV1::CandidateGridNotStrictlyIncreasing);
            }
            previous = n;
        }

        let registered_ns: BTreeSet<_> = self
            .candidate_completed_participants
            .iter()
            .copied()
            .collect();
        let mut cells: BTreeMap<(String, usize), &PowerGridCellV1> = BTreeMap::new();
        for cell in &self.grid {
            let key = (cell.scenario_id.clone(), cell.completed_participants);
            if !scenario_ids.contains(&cell.scenario_id)
                || !registered_ns.contains(&cell.completed_participants)
            {
                issues.push(PerceptualPowerPlanIssueV1::UnexpectedGridCell {
                    scenario_id: cell.scenario_id.clone(),
                    participants: cell.completed_participants,
                });
            }
            if cells.insert(key, cell).is_some() {
                issues.push(PerceptualPowerPlanIssueV1::DuplicateGridCell {
                    scenario_id: cell.scenario_id.clone(),
                    participants: cell.completed_participants,
                });
            }
            if cell.simulation_replicates < self.minimum_simulation_replicates
                || cell.primary_rejection_count > cell.simulation_replicates
            {
                issues.push(PerceptualPowerPlanIssueV1::InvalidGridCell {
                    scenario_id: cell.scenario_id.clone(),
                    participants: cell.completed_participants,
                });
            }
        }
        for scenario_id in &scenario_ids {
            for &n in &registered_ns {
                if !cells.contains_key(&(scenario_id.clone(), n)) {
                    issues.push(PerceptualPowerPlanIssueV1::MissingGridCell {
                        scenario_id: scenario_id.clone(),
                        participants: n,
                    });
                }
            }
        }

        let first_robust_passing = self
            .candidate_completed_participants
            .iter()
            .copied()
            .find(|n| {
                scenario_ids.iter().all(|scenario_id| {
                    cells
                        .get(&(scenario_id.clone(), *n))
                        .and_then(|cell| cell.one_sided_95_wilson_lower_bound())
                        .is_some_and(|lower| lower >= self.target_power)
                })
            });
        match first_robust_passing {
            None => issues.push(PerceptualPowerPlanIssueV1::NoRobustPassingN),
            Some(expected) => {
                if !registered_ns.contains(&self.selected_completed_participants) {
                    issues.push(PerceptualPowerPlanIssueV1::SelectedNNotRegistered);
                }
                if self.selected_completed_participants != expected {
                    issues.push(PerceptualPowerPlanIssueV1::SelectedNNotSmallestRobustPassingN);
                }
            }
        }

        if !self.anticipated_noncompletion_rate.is_finite()
            || self.anticipated_noncompletion_rate < 0.0
            || self.anticipated_noncompletion_rate >= 0.5
        {
            issues.push(PerceptualPowerPlanIssueV1::InvalidNoncompletionRate);
        } else if self.selected_completed_participants > 0 {
            let expected_max = (self.selected_completed_participants as f64
                / (1.0 - self.anticipated_noncompletion_rate))
                .ceil() as usize;
            if self.maximum_enrolled_participants != expected_max {
                issues.push(PerceptualPowerPlanIssueV1::EnrollmentInflationMismatch);
            }
        }
        if self.outcome_adaptive_stopping_allowed {
            issues.push(PerceptualPowerPlanIssueV1::OutcomeAdaptiveStoppingAllowed);
        }
        for (field, digest) in [
            ("runner.binary_sha256", self.runner.binary_sha256.as_str()),
            ("runner.environment_sha256", self.runner.environment_sha256.as_str()),
        ] {
            if !is_sha256(digest) {
                issues.push(PerceptualPowerPlanIssueV1::InvalidDigest {
                    field: field.into(),
                });
            }
        }
        for (field, value) in [
            ("runner.runner_version", self.runner.runner_version.as_str()),
            ("runner.source_revision", self.runner.source_revision.as_str()),
            ("runner.rng_algorithm", self.runner.rng_algorithm.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(PerceptualPowerPlanIssueV1::EmptyRunnerField {
                    field: field.into(),
                });
            }
        }
        issues
    }

    pub fn plan_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn analysis() -> FrozenPerceptualAnalysisSpecV1 {
        FrozenPerceptualAnalysisSpecV1 {
            spec_version: PERCEPTUAL_ANALYSIS_SPEC_VERSION.into(),
            primary_task: PerceptualTaskV1::AbxDiscrimination,
            key_secondary_task: PerceptualTaskV1::DirectionalRearticulation2Afc,
            primary_model: PrimaryAnalysisModelV1::CrossedParticipantItemLogisticRandomIntercepts,
            link: BinaryLinkV1::Logit,
            chance_probability: CHANCE_PROBABILITY,
            alpha: 0.05,
            confidence_level: 0.95,
            participant_random_intercept_required: true,
            item_random_intercept_required: true,
            registered_item_seeds: MEL003_FIXED_SEEDS,
            primary_alternative_greater_than_chance: true,
            secondary_conditioning: SecondaryConditioningPolicyV1::IndependentTrialsNoConditioningOnPrimary,
            secondary_multiplicity: SecondaryMultiplicityPolicyV1::HolmWithinRegisteredSecondaryFamily,
            report_fixed_effect_estimate: true,
            report_odds_ratio: true,
            report_confidence_interval: true,
            report_participant_variance: true,
            report_item_variance: true,
            report_participant_summary: true,
            report_item_summary: true,
            model_failure_policy: ModelFailurePolicyV1::InconclusiveNoAutomaticSubstitution,
            execution: PerceptualAnalysisExecutionIdentityV1 {
                source_revision: "analysis-source-v1".into(),
                analysis_program_sha256: DIGEST.into(),
                environment_sha256: DIGEST.into(),
                model_engine: "qualified-glmm-runner".into(),
                model_engine_version: "v1".into(),
            },
        }
    }

    fn plan(analysis: &FrozenPerceptualAnalysisSpecV1) -> FrozenPerceptualPowerPlanV1 {
        FrozenPerceptualPowerPlanV1 {
            plan_version: PERCEPTUAL_POWER_PLAN_VERSION.into(),
            analysis_spec_sha256: analysis.spec_sha256().unwrap(),
            effect_basis: PowerEffectBasisV1::HumanPerceptualSesoI,
            primary_sesoi_success_probability: 0.60,
            item_count: MEL003_FIXED_SEEDS.len(),
            target_power: 0.80,
            alpha: analysis.alpha,
            selection_rule: PowerSelectionRuleV1::SmallestRobustNByOneSided95WilsonLowerBound,
            minimum_simulation_replicates: 5_000,
            heterogeneity_scenarios: vec![
                PowerHeterogeneityScenarioV1 {
                    scenario_id: "moderate".into(),
                    participant_intercept_sd: 0.5,
                    item_intercept_sd: 0.35,
                },
                PowerHeterogeneityScenarioV1 {
                    scenario_id: "high-heterogeneity".into(),
                    participant_intercept_sd: 0.8,
                    item_intercept_sd: 0.55,
                },
            ],
            candidate_completed_participants: vec![32, 48, 64],
            grid: vec![
                PowerGridCellV1 { scenario_id: "moderate".into(), completed_participants: 32, simulation_replicates: 5_000, primary_rejection_count: 3_500 },
                PowerGridCellV1 { scenario_id: "moderate".into(), completed_participants: 48, simulation_replicates: 5_000, primary_rejection_count: 4_250 },
                PowerGridCellV1 { scenario_id: "moderate".into(), completed_participants: 64, simulation_replicates: 5_000, primary_rejection_count: 4_600 },
                PowerGridCellV1 { scenario_id: "high-heterogeneity".into(), completed_participants: 32, simulation_replicates: 5_000, primary_rejection_count: 3_000 },
                PowerGridCellV1 { scenario_id: "high-heterogeneity".into(), completed_participants: 48, simulation_replicates: 5_000, primary_rejection_count: 4_050 },
                PowerGridCellV1 { scenario_id: "high-heterogeneity".into(), completed_participants: 64, simulation_replicates: 5_000, primary_rejection_count: 4_400 },
            ],
            selected_completed_participants: 48,
            anticipated_noncompletion_rate: 0.10,
            maximum_enrolled_participants: 54,
            outcome_adaptive_stopping_allowed: false,
            runner: PowerSimulationRunnerIdentityV1 {
                runner_version: "perceptual-power-sim-v1".into(),
                source_revision: "planning-source-v1".into(),
                binary_sha256: DIGEST.into(),
                environment_sha256: DIGEST.into(),
                rng_algorithm: "ChaCha20".into(),
                rng_seed: 7,
            },
        }
    }

    #[test]
    fn valid_plan_selects_smallest_n_passing_every_scenario_with_mc_uncertainty() {
        let analysis = analysis();
        assert!(analysis.validate().is_empty());
        let plan = plan(&analysis);
        assert!(plan.validate(&analysis).is_empty());
        assert_eq!(plan.selected_completed_participants, 48);
        assert_eq!(plan.maximum_enrolled_participants, 54);
        assert!(plan.grid[4].estimated_power().unwrap() >= 0.80);
        assert!(plan.grid[4].one_sided_95_wilson_lower_bound().unwrap() >= 0.80);
        assert_eq!(plan.plan_sha256().unwrap().len(), 64);
    }

    #[test]
    fn point_estimate_at_target_is_not_enough_when_mc_lower_bound_misses() {
        let analysis = analysis();
        let mut plan = plan(&analysis);
        let cell = plan
            .grid
            .iter_mut()
            .find(|cell| cell.scenario_id == "high-heterogeneity" && cell.completed_participants == 48)
            .unwrap();
        cell.primary_rejection_count = 4_000;
        assert_eq!(cell.estimated_power(), Some(0.80));
        assert!(cell.one_sided_95_wilson_lower_bound().unwrap() < 0.80);
        assert!(plan.validate(&analysis).contains(
            &PerceptualPowerPlanIssueV1::SelectedNNotSmallestRobustPassingN
        ));
    }

    #[test]
    fn one_optimistic_scenario_is_not_a_robust_power_plan() {
        let analysis = analysis();
        let mut plan = plan(&analysis);
        plan.heterogeneity_scenarios.truncate(1);
        plan.grid.retain(|cell| cell.scenario_id == "moderate");
        assert!(plan.validate(&analysis).contains(
            &PerceptualPowerPlanIssueV1::TooFewHeterogeneityScenarios {
                found: 1,
                required: MIN_POWER_HETEROGENEITY_SCENARIOS,
            }
        ));
    }

    #[test]
    fn acoustic_proxy_cannot_replace_human_sesoi() {
        let analysis = analysis();
        let mut plan = plan(&analysis);
        plan.primary_sesoi_success_probability = CHANCE_PROBABILITY;
        assert!(plan
            .validate(&analysis)
            .contains(&PerceptualPowerPlanIssueV1::InvalidHumanSesoI));
    }

    #[test]
    fn outcome_adaptive_stopping_is_rejected() {
        let analysis = analysis();
        let mut plan = plan(&analysis);
        plan.outcome_adaptive_stopping_allowed = true;
        assert!(plan
            .validate(&analysis)
            .contains(&PerceptualPowerPlanIssueV1::OutcomeAdaptiveStoppingAllowed));
    }

    #[test]
    fn analysis_digest_is_load_bearing() {
        let analysis = analysis();
        let mut plan = plan(&analysis);
        plan.analysis_spec_sha256 = "b".repeat(64);
        assert!(plan
            .validate(&analysis)
            .contains(&PerceptualPowerPlanIssueV1::AnalysisSpecDigestMismatch));
    }
}
