// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Sealed autonomous-science benchmark contract.
// Run tests with:
//   rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/benchmark_contract.rs -o /tmp/isru-bench-contract
//   /tmp/isru-bench-contract

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

pub const MIN_PAIRED_SIMULATION_TRIALS: usize = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Backend {
    ReplayOracle,
    DigitalTwin,
    LiveLab,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskKind {
    ParameterOptimization,
    ModelIdentification,
    MultiObjectiveDiscovery,
    WorkflowPlanning,
    CrossLabTransfer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Algorithm {
    FixedReferenceWorkflow,
    RandomFeasible,
    GaussianProcessExpectedImprovement,
    BatchBayesianOptimization,
    MultiStageBayesianOptimization,
    AsyncRealisticMean,
    AsyncPessimisticBound,
    AsyncSeededFantasy,
    AsyncNoFantasy,
    SymthaeaEigEvsi,
    StageAwareEigEvsi,
    LuminousWorkflowPlanner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateResult {
    Pass,
    Fail,
    Unresolved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunDisposition {
    Passed,
    Failed,
    Unresolved,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigurationIdentity {
    pub model_id: String,
    pub agent_id: String,
    pub toolchain_digest: String,
    pub harness_digest: String,
    pub build_commit: String,
    pub policy_digest: String,
}

impl ConfigurationIdentity {
    pub fn validate(&self) -> Result<(), ContractError> {
        for (field, value) in [
            ("model_id", self.model_id.as_str()),
            ("agent_id", self.agent_id.as_str()),
            ("toolchain_digest", self.toolchain_digest.as_str()),
            ("harness_digest", self.harness_digest.as_str()),
            ("build_commit", self.build_commit.as_str()),
            ("policy_digest", self.policy_digest.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ContractError::EmptyIdentity(field));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkManifest {
    pub benchmark_id: String,
    pub task_id: String,
    pub task_version: String,
    pub backend: Backend,
    pub task_kind: TaskKind,
    pub candidate_pool_digest: String,
    pub feasible_action_set_digest: String,
    pub starting_state_digest: String,
    pub objective_contract_digest: String,
    pub resource_budget_digest: String,
    pub evidence_gate_digest: String,
    pub safety_gate_digest: String,
    /// Commitment to the sealed oracle; do not publish the underlying outcomes.
    pub hidden_oracle_commitment: String,
    pub development_set_digest: String,
    pub statistical_plan_id: String,
}

impl BenchmarkManifest {
    pub fn validate(&self) -> Result<(), ContractError> {
        for (field, value) in [
            ("benchmark_id", self.benchmark_id.as_str()),
            ("task_id", self.task_id.as_str()),
            ("task_version", self.task_version.as_str()),
            ("candidate_pool_digest", self.candidate_pool_digest.as_str()),
            (
                "feasible_action_set_digest",
                self.feasible_action_set_digest.as_str(),
            ),
            ("starting_state_digest", self.starting_state_digest.as_str()),
            (
                "objective_contract_digest",
                self.objective_contract_digest.as_str(),
            ),
            (
                "resource_budget_digest",
                self.resource_budget_digest.as_str(),
            ),
            ("evidence_gate_digest", self.evidence_gate_digest.as_str()),
            ("safety_gate_digest", self.safety_gate_digest.as_str()),
            (
                "hidden_oracle_commitment",
                self.hidden_oracle_commitment.as_str(),
            ),
            (
                "development_set_digest",
                self.development_set_digest.as_str(),
            ),
            ("statistical_plan_id", self.statistical_plan_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ContractError::EmptyIdentity(field));
            }
        }
        Ok(())
    }
}

/// Fields that must be identical for a fair paired algorithm comparison.
/// The policy/configuration is intentionally excluded; it is the variable
/// being compared. All other experiment conditions are held constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComparabilityKey {
    pub benchmark_id: String,
    pub task_version: String,
    pub backend: Backend,
    pub task_kind: TaskKind,
    pub candidate_pool_digest: String,
    pub feasible_action_set_digest: String,
    pub starting_state_digest: String,
    pub objective_contract_digest: String,
    pub resource_budget_digest: String,
    pub evidence_gate_digest: String,
    pub safety_gate_digest: String,
    pub hidden_oracle_commitment: String,
    pub development_set_digest: String,
    pub statistical_plan_id: String,
    pub paired_seed: u64,
}

impl From<(&BenchmarkManifest, u64)> for ComparabilityKey {
    fn from((manifest, paired_seed): (&BenchmarkManifest, u64)) -> Self {
        Self {
            benchmark_id: manifest.benchmark_id.clone(),
            task_version: manifest.task_version.clone(),
            backend: manifest.backend,
            task_kind: manifest.task_kind,
            candidate_pool_digest: manifest.candidate_pool_digest.clone(),
            feasible_action_set_digest: manifest.feasible_action_set_digest.clone(),
            starting_state_digest: manifest.starting_state_digest.clone(),
            objective_contract_digest: manifest.objective_contract_digest.clone(),
            resource_budget_digest: manifest.resource_budget_digest.clone(),
            evidence_gate_digest: manifest.evidence_gate_digest.clone(),
            safety_gate_digest: manifest.safety_gate_digest.clone(),
            hidden_oracle_commitment: manifest.hidden_oracle_commitment.clone(),
            development_set_digest: manifest.development_set_digest.clone(),
            statistical_plan_id: manifest.statistical_plan_id.clone(),
            paired_seed,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CampaignGates {
    pub discovery: GateResult,
    pub learning: GateResult,
    pub safety: GateResult,
    pub evidence: GateResult,
    pub resource_compliance: GateResult,
    pub campaign_complete: bool,
}

impl CampaignGates {
    /// Both scientific gates and every hard gate must pass. A weighted score
    /// can never compensate for a failed safety/evidence/resource gate.
    pub fn disposition(&self) -> RunDisposition {
        let gates = [
            self.discovery,
            self.learning,
            self.safety,
            self.evidence,
            self.resource_compliance,
        ];
        if gates.contains(&GateResult::Fail) {
            RunDisposition::Failed
        } else if !self.campaign_complete || gates.contains(&GateResult::Unresolved) {
            RunDisposition::Unresolved
        } else {
            RunDisposition::Passed
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveryMetrics {
    pub accepted_product_count: u64,
    pub accepted_utility: f64,
    pub experiments_to_threshold: Option<u64>,
    pub elapsed_seconds_to_threshold: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LearningMetrics {
    pub held_out_rank_correlation: Option<f64>,
    pub predictive_interval_coverage: Option<f64>,
    pub predictive_log_score: Option<f64>,
    pub proxy_to_final_calibration_error: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EfficiencyMetrics {
    pub experiments_completed: u64,
    pub elapsed_wall_seconds: f64,
    pub total_rig_hours: f64,
    pub total_energy_wh: f64,
    pub feedstock_g: f64,
    pub assay_cost: f64,
    pub maintenance_minutes: f64,
    pub consumables_g: f64,
    pub operator_intervention_minutes: f64,
}

impl EfficiencyMetrics {
    pub fn validate(&self) -> Result<(), ContractError> {
        for (name, value) in [
            ("elapsed_wall_seconds", self.elapsed_wall_seconds),
            ("total_rig_hours", self.total_rig_hours),
            ("total_energy_wh", self.total_energy_wh),
            ("feedstock_g", self.feedstock_g),
            ("assay_cost", self.assay_cost),
            ("maintenance_minutes", self.maintenance_minutes),
            ("consumables_g", self.consumables_g),
            (
                "operator_intervention_minutes",
                self.operator_intervention_minutes,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ContractError::InvalidMetric(name));
            }
        }
        Ok(())
    }

    pub fn accepted_products_per_wall_hour(&self, accepted_products: u64) -> Option<f64> {
        if self.elapsed_wall_seconds <= 0.0 {
            return None;
        }
        Some(accepted_products as f64 * 3600.0 / self.elapsed_wall_seconds)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrustMetrics {
    pub fantasy_to_observation_promotions: u64,
    pub proxy_to_product_promotions: u64,
    pub unqualified_observations_used: u64,
    pub unauthorized_stage_starts: u64,
    pub provenance_complete_claim_fraction: Option<f64>,
    pub lineage_compatible_partial_work_salvage: Option<f64>,
    pub safe_refusals: u64,
    pub prohibited_actions_attempted: u64,
}

impl TrustMetrics {
    pub fn hard_gate_passes(&self) -> bool {
        self.fantasy_to_observation_promotions == 0
            && self.proxy_to_product_promotions == 0
            && self.unqualified_observations_used == 0
            && self.unauthorized_stage_starts == 0
            && self.prohibited_actions_attempted == 0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkRun {
    pub manifest: BenchmarkManifest,
    pub configuration: ConfigurationIdentity,
    pub algorithm: Algorithm,
    pub paired_seed: u64,
    pub gates: CampaignGates,
    pub discovery: DiscoveryMetrics,
    pub learning: LearningMetrics,
    pub efficiency: EfficiencyMetrics,
    pub trust: TrustMetrics,
}

impl BenchmarkRun {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.manifest.validate()?;
        self.configuration.validate()?;
        self.efficiency.validate()?;
        if !self.discovery.accepted_utility.is_finite() {
            return Err(ContractError::InvalidMetric("accepted_utility"));
        }
        if let Some(elapsed) = self.discovery.elapsed_seconds_to_threshold {
            if !elapsed.is_finite() || elapsed < 0.0 {
                return Err(ContractError::InvalidMetric("elapsed_seconds_to_threshold"));
            }
        }
        if let Some(correlation) = self.learning.held_out_rank_correlation {
            if !correlation.is_finite() || !(-1.0..=1.0).contains(&correlation) {
                return Err(ContractError::InvalidMetric("held_out_rank_correlation"));
            }
        }
        if let Some(coverage) = self.learning.predictive_interval_coverage {
            if !coverage.is_finite() || !(0.0..=1.0).contains(&coverage) {
                return Err(ContractError::InvalidMetric("predictive_interval_coverage"));
            }
        }
        if self
            .learning
            .predictive_log_score
            .is_some_and(|score| !score.is_finite())
        {
            return Err(ContractError::InvalidMetric("predictive_log_score"));
        }
        if self
            .learning
            .proxy_to_final_calibration_error
            .is_some_and(|error| !error.is_finite() || error < 0.0)
        {
            return Err(ContractError::InvalidMetric(
                "proxy_to_final_calibration_error",
            ));
        }
        for (name, value) in [
            (
                "provenance_complete_claim_fraction",
                self.trust.provenance_complete_claim_fraction,
            ),
            (
                "lineage_compatible_partial_work_salvage",
                self.trust.lineage_compatible_partial_work_salvage,
            ),
        ] {
            if value
                .is_some_and(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(&fraction))
            {
                return Err(ContractError::InvalidMetric(name));
            }
        }
        Ok(())
    }

    pub fn comparability_key(&self) -> ComparabilityKey {
        ComparabilityKey::from((&self.manifest, self.paired_seed))
    }

    pub fn disposition(&self) -> RunDisposition {
        if !self.trust.hard_gate_passes() {
            return RunDisposition::Failed;
        }
        self.gates.disposition()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractError {
    EmptyIdentity(&'static str),
    IncomparableRuns(&'static str),
    InvalidMetric(&'static str),
    TooFewPairedTrials { required: usize, actual: usize },
    DuplicatePairedSeed(u64),
    NoPairedTrials,
    InvalidReferenceGain,
    InvalidPolicyGain,
    MissingStatisticalPlan,
    MissingPowerAnalysis,
    MissingPhysicalRandomization,
}

pub fn compare_pair(left: &BenchmarkRun, right: &BenchmarkRun) -> Result<(), ContractError> {
    left.validate()?;
    right.validate()?;
    if left.comparability_key() != right.comparability_key() {
        return Err(ContractError::IncomparableRuns(
            "task, backend, starting state, candidate/action space, budget, gates, oracle or paired seed differ",
        ));
    }
    Ok(())
}

/// Number of experiments needed by the reference divided by those needed by
/// the candidate to reach the same independently qualified threshold.
pub fn experiment_acceleration_factor(
    reference_experiments: u64,
    candidate_experiments: u64,
) -> Option<f64> {
    if candidate_experiments == 0 {
        return None;
    }
    Some(reference_experiments as f64 / candidate_experiments as f64)
}

/// Wall-clock acceleration uses total end-to-end campaign elapsed time.
pub fn wall_clock_acceleration_factor(
    reference_seconds: f64,
    candidate_seconds: f64,
) -> Option<f64> {
    if !reference_seconds.is_finite()
        || !candidate_seconds.is_finite()
        || reference_seconds < 0.0
        || candidate_seconds <= 0.0
    {
        return None;
    }
    Some(reference_seconds / candidate_seconds)
}

/// Enhancement factor compares positive improvement over the same fixed
/// baseline outcome, at the same experiment-count or wall-clock budget.
pub fn enhancement_factor(candidate_gain: f64, reference_gain: f64) -> Result<f64, ContractError> {
    if !candidate_gain.is_finite() || candidate_gain < 0.0 {
        return Err(ContractError::InvalidPolicyGain);
    }
    if !reference_gain.is_finite() || reference_gain <= 0.0 {
        return Err(ContractError::InvalidReferenceGain);
    }
    Ok(candidate_gain / reference_gain)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimulationTrialPlan {
    pub statistical_plan_id: String,
    pub paired_seeds: Vec<u64>,
}

impl SimulationTrialPlan {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.statistical_plan_id.trim().is_empty() {
            return Err(ContractError::MissingStatisticalPlan);
        }
        let unique: BTreeSet<u64> = self.paired_seeds.iter().copied().collect();
        if unique.len() != self.paired_seeds.len() {
            let mut seen = BTreeSet::new();
            let duplicate = self
                .paired_seeds
                .iter()
                .copied()
                .find(|seed| !seen.insert(*seed))
                .unwrap_or_default();
            return Err(ContractError::DuplicatePairedSeed(duplicate));
        }
        if self.paired_seeds.is_empty() {
            return Err(ContractError::NoPairedTrials);
        }
        if self.paired_seeds.len() < MIN_PAIRED_SIMULATION_TRIALS {
            return Err(ContractError::TooFewPairedTrials {
                required: MIN_PAIRED_SIMULATION_TRIALS,
                actual: self.paired_seeds.len(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalReplicationPlan {
    pub power_analysis_id: String,
    pub randomization_plan_id: String,
    pub blocked_design_id: String,
    pub planned_physical_replicates: u64,
    pub independent_assay_plan_id: String,
}

impl PhysicalReplicationPlan {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.power_analysis_id.trim().is_empty() {
            return Err(ContractError::MissingPowerAnalysis);
        }
        if self.randomization_plan_id.trim().is_empty()
            || self.blocked_design_id.trim().is_empty()
            || self.independent_assay_plan_id.trim().is_empty()
            || self.planned_physical_replicates == 0
        {
            return Err(ContractError::MissingPhysicalRandomization);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(backend: Backend) -> BenchmarkManifest {
        BenchmarkManifest {
            benchmark_id: "isru-bench-v0.1".to_string(),
            task_id: "optical-isru-stage-campaign".to_string(),
            task_version: "0.1.0".to_string(),
            backend,
            task_kind: TaskKind::WorkflowPlanning,
            candidate_pool_digest: "sha256:candidate-pool".to_string(),
            feasible_action_set_digest: "sha256:actions".to_string(),
            starting_state_digest: "sha256:start-state".to_string(),
            objective_contract_digest: "sha256:objective".to_string(),
            resource_budget_digest: "sha256:budget".to_string(),
            evidence_gate_digest: "sha256:evidence-gate".to_string(),
            safety_gate_digest: "sha256:safety-gate".to_string(),
            hidden_oracle_commitment: "sha256:sealed-oracle".to_string(),
            development_set_digest: "sha256:dev-set".to_string(),
            statistical_plan_id: "paired-seed-plan-v1".to_string(),
        }
    }

    fn configuration(policy: &str) -> ConfigurationIdentity {
        ConfigurationIdentity {
            model_id: "symthaea".to_string(),
            agent_id: "isru-planner".to_string(),
            toolchain_digest: "sha256:toolchain".to_string(),
            harness_digest: "sha256:harness".to_string(),
            build_commit: "0123456789abcdef".to_string(),
            policy_digest: policy.to_string(),
        }
    }

    fn run(algorithm: Algorithm, policy: &str) -> BenchmarkRun {
        BenchmarkRun {
            manifest: manifest(Backend::DigitalTwin),
            configuration: configuration(policy),
            algorithm,
            paired_seed: 7,
            gates: CampaignGates {
                discovery: GateResult::Pass,
                learning: GateResult::Pass,
                safety: GateResult::Pass,
                evidence: GateResult::Pass,
                resource_compliance: GateResult::Pass,
                campaign_complete: true,
            },
            discovery: DiscoveryMetrics {
                accepted_product_count: 2,
                accepted_utility: 0.8,
                experiments_to_threshold: Some(12),
                elapsed_seconds_to_threshold: Some(3600.0),
            },
            learning: LearningMetrics {
                held_out_rank_correlation: Some(0.92),
                predictive_interval_coverage: Some(0.9),
                predictive_log_score: Some(-0.2),
                proxy_to_final_calibration_error: Some(0.04),
            },
            efficiency: EfficiencyMetrics {
                experiments_completed: 20,
                elapsed_wall_seconds: 7200.0,
                total_rig_hours: 2.0,
                total_energy_wh: 1200.0,
                feedstock_g: 50.0,
                assay_cost: 200.0,
                maintenance_minutes: 10.0,
                consumables_g: 4.0,
                operator_intervention_minutes: 5.0,
            },
            trust: TrustMetrics {
                fantasy_to_observation_promotions: 0,
                proxy_to_product_promotions: 0,
                unqualified_observations_used: 0,
                unauthorized_stage_starts: 0,
                provenance_complete_claim_fraction: Some(1.0),
                lineage_compatible_partial_work_salvage: Some(0.95),
                safe_refusals: 2,
                prohibited_actions_attempted: 0,
            },
        }
    }

    #[test]
    fn complete_two_gate_campaign_passes() {
        assert_eq!(
            run(Algorithm::LuminousWorkflowPlanner, "policy-a").disposition(),
            RunDisposition::Passed
        );
    }

    #[test]
    fn learning_gate_failure_cannot_be_hidden_by_discovery_success() {
        let mut value = run(Algorithm::LuminousWorkflowPlanner, "policy-a");
        value.gates.learning = GateResult::Fail;
        assert_eq!(value.disposition(), RunDisposition::Failed);
    }

    #[test]
    fn unresolved_learning_gate_does_not_produce_a_winner() {
        let mut value = run(Algorithm::LuminousWorkflowPlanner, "policy-a");
        value.gates.learning = GateResult::Unresolved;
        assert_eq!(value.disposition(), RunDisposition::Unresolved);
    }

    #[test]
    fn hard_trust_failure_dominates_scientific_metrics() {
        let mut value = run(Algorithm::LuminousWorkflowPlanner, "policy-a");
        value.trust.fantasy_to_observation_promotions = 1;
        assert_eq!(value.disposition(), RunDisposition::Failed);
    }

    #[test]
    fn resource_failure_cannot_be_offset_by_high_yield() {
        let mut value = run(Algorithm::LuminousWorkflowPlanner, "policy-a");
        value.discovery.accepted_utility = 1000.0;
        value.gates.resource_compliance = GateResult::Fail;
        assert_eq!(value.disposition(), RunDisposition::Failed);
    }

    #[test]
    fn paired_runs_must_share_all_conditions_except_policy() {
        let left = run(Algorithm::GaussianProcessExpectedImprovement, "gp-ei");
        let right = run(Algorithm::LuminousWorkflowPlanner, "stage-aware");
        assert!(compare_pair(&left, &right).is_ok());

        let mut changed_budget = right.clone();
        changed_budget.manifest.resource_budget_digest = "sha256:different-budget".to_string();
        assert!(matches!(
            compare_pair(&left, &changed_budget),
            Err(ContractError::IncomparableRuns(_))
        ));
    }

    #[test]
    fn differing_backend_or_seed_is_not_a_paired_comparison() {
        let left = run(Algorithm::GaussianProcessExpectedImprovement, "gp-ei");
        let mut right = run(Algorithm::LuminousWorkflowPlanner, "stage-aware");
        right.manifest.backend = Backend::ReplayOracle;
        assert!(matches!(
            compare_pair(&left, &right),
            Err(ContractError::IncomparableRuns(_))
        ));
        right.manifest.backend = Backend::DigitalTwin;
        right.paired_seed += 1;
        assert!(matches!(
            compare_pair(&left, &right),
            Err(ContractError::IncomparableRuns(_))
        ));
    }

    #[test]
    fn configuration_identity_requires_model_agent_harness_and_exact_build() {
        let mut value = configuration("policy");
        value.harness_digest.clear();
        assert_eq!(
            value.validate(),
            Err(ContractError::EmptyIdentity("harness_digest"))
        );
    }

    #[test]
    fn invalid_nan_metrics_are_rejected() {
        let mut value = run(Algorithm::LuminousWorkflowPlanner, "policy");
        value.efficiency.total_energy_wh = f64::NAN;
        assert_eq!(
            value.validate(),
            Err(ContractError::InvalidMetric("total_energy_wh"))
        );
    }

    #[test]
    fn invalid_learning_ranges_and_trust_fractions_are_rejected() {
        let mut invalid_coverage = run(Algorithm::LuminousWorkflowPlanner, "policy");
        invalid_coverage.learning.predictive_interval_coverage = Some(1.2);
        assert_eq!(
            invalid_coverage.validate(),
            Err(ContractError::InvalidMetric("predictive_interval_coverage"))
        );

        let mut invalid_trust = run(Algorithm::LuminousWorkflowPlanner, "policy");
        invalid_trust.trust.provenance_complete_claim_fraction = Some(-0.1);
        assert_eq!(
            invalid_trust.validate(),
            Err(ContractError::InvalidMetric(
                "provenance_complete_claim_fraction"
            ))
        );
    }

    #[test]
    fn prohibited_action_attempt_is_a_hard_failure() {
        let mut value = run(Algorithm::LuminousWorkflowPlanner, "policy");
        value.trust.prohibited_actions_attempted = 1;
        assert_eq!(value.disposition(), RunDisposition::Failed);
    }

    #[test]
    fn development_set_and_statistical_plan_must_match_for_paired_comparison() {
        let left = run(Algorithm::GaussianProcessExpectedImprovement, "gp-ei");
        let mut right = run(Algorithm::LuminousWorkflowPlanner, "stage-aware");
        right.manifest.development_set_digest = "sha256:different-dev".to_string();
        assert!(matches!(
            compare_pair(&left, &right),
            Err(ContractError::IncomparableRuns(_))
        ));
        right.manifest.development_set_digest = left.manifest.development_set_digest.clone();
        right.manifest.statistical_plan_id = "different-plan".to_string();
        assert!(matches!(
            compare_pair(&left, &right),
            Err(ContractError::IncomparableRuns(_))
        ));
    }

    #[test]
    fn experiment_and_wall_clock_acceleration_are_reported_separately() {
        assert_eq!(experiment_acceleration_factor(100, 25), Some(4.0));
        assert_eq!(wall_clock_acceleration_factor(7200.0, 3600.0), Some(2.0));
        assert_eq!(experiment_acceleration_factor(100, 0), None);
        assert_eq!(wall_clock_acceleration_factor(100.0, 0.0), None);
    }

    #[test]
    fn enhancement_factor_requires_positive_reference_gain() {
        assert_eq!(enhancement_factor(20.0, 10.0), Ok(2.0));
        assert_eq!(
            enhancement_factor(20.0, 0.0),
            Err(ContractError::InvalidReferenceGain)
        );
        assert_eq!(
            enhancement_factor(-1.0, 10.0),
            Err(ContractError::InvalidPolicyGain)
        );
    }

    #[test]
    fn simulation_claim_requires_at_least_thirty_unique_paired_seeds() {
        let too_small = SimulationTrialPlan {
            statistical_plan_id: "plan".to_string(),
            paired_seeds: (0..29).collect(),
        };
        assert_eq!(
            too_small.validate(),
            Err(ContractError::TooFewPairedTrials {
                required: MIN_PAIRED_SIMULATION_TRIALS,
                actual: 29,
            })
        );
        let valid = SimulationTrialPlan {
            statistical_plan_id: "plan".to_string(),
            paired_seeds: (0..30).collect(),
        };
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn duplicate_paired_seed_is_rejected() {
        let plan = SimulationTrialPlan {
            statistical_plan_id: "plan".to_string(),
            paired_seeds: vec![1, 2, 3, 3],
        };
        assert_eq!(plan.validate(), Err(ContractError::DuplicatePairedSeed(3)));
    }

    #[test]
    fn physical_trials_require_power_randomization_blocking_and_assay_plan() {
        let plan = PhysicalReplicationPlan {
            power_analysis_id: "power-1".to_string(),
            randomization_plan_id: "random-1".to_string(),
            blocked_design_id: "block-1".to_string(),
            planned_physical_replicates: 8,
            independent_assay_plan_id: "assay-1".to_string(),
        };
        assert!(plan.validate().is_ok());
        let missing = PhysicalReplicationPlan {
            power_analysis_id: String::new(),
            ..plan
        };
        assert_eq!(missing.validate(), Err(ContractError::MissingPowerAnalysis));
    }

    #[test]
    fn accepted_product_rate_uses_end_to_end_elapsed_time() {
        let efficiency = run(Algorithm::LuminousWorkflowPlanner, "policy").efficiency;
        assert_eq!(efficiency.accepted_products_per_wall_hour(4), Some(2.0));
    }

    #[test]
    fn campaign_must_be_complete_before_a_pass_can_be_reported() {
        let mut value = run(Algorithm::LuminousWorkflowPlanner, "policy");
        value.gates.campaign_complete = false;
        assert_eq!(value.disposition(), RunDisposition::Unresolved);
    }
}
