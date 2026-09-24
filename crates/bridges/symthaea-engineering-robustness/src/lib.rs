// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing tolerance and robust-design campaign contracts.
//!
//! This crate defines *what was varied* and *what evidence came back*. It does
//! not mandate Monte Carlo, Latin-hypercube, Bayesian, or other sampling/search
//! algorithms. Failed scenarios are censored explicitly rather than imputed as
//! favorable outcomes.
//!
//! `simulation scenario feasibility != manufacturing yield`.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_engineering_optimization::{
    CandidateEvaluation, DesignCandidate, EvaluationStatus, FidelityClass, ObjectiveDirection,
    OptimizationError, OptimizationProfile,
};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UncertaintySourceKind {
    DatasheetTolerance,
    ManufacturingCapability,
    MeasuredPopulation,
    AssumedPrior,
    FittedPosterior,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UncertaintySource {
    pub kind: UncertaintySourceKind,
    pub provenance: String,
    pub evidence_ref: Option<String>,
    pub method: Option<String>,
}

impl UncertaintySource {
    pub fn validate(&self) -> Result<(), RobustnessError> {
        require_nonempty("uncertainty.provenance", &self.provenance)?;
        if matches!(
            self.kind,
            UncertaintySourceKind::DatasheetTolerance
                | UncertaintySourceKind::ManufacturingCapability
                | UncertaintySourceKind::MeasuredPopulation
                | UncertaintySourceKind::FittedPosterior
        ) && self
            .evidence_ref
            .as_ref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err(RobustnessError::MissingEvidenceReference(self.kind));
        }
        if matches!(self.kind, UncertaintySourceKind::FittedPosterior)
            && self
                .method
                .as_ref()
                .is_none_or(|value| value.trim().is_empty())
        {
            return Err(RobustnessError::MissingMethod(self.kind));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub lower: f64,
    pub upper: f64,
}

impl Bounds {
    fn validate(&self) -> Result<(), RobustnessError> {
        if !self.lower.is_finite() || !self.upper.is_finite() || self.lower > self.upper {
            return Err(RobustnessError::InvalidBounds {
                lower: self.lower,
                upper: self.upper,
            });
        }
        Ok(())
    }

    fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }
}

/// Declared uncertainty model. Sampling is performed by an external backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UncertaintyDistribution {
    BoundedUniform(Bounds),
    Normal {
        mean: f64,
        standard_deviation: f64,
        truncation: Option<Bounds>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncertainParameter {
    pub id: String,
    pub unit: String,
    pub distribution: UncertaintyDistribution,
    pub source: UncertaintySource,
}

impl UncertainParameter {
    fn validate(&self) -> Result<(), RobustnessError> {
        require_nonempty("uncertain_parameter.id", &self.id)?;
        require_nonempty("uncertain_parameter.unit", &self.unit)?;
        self.source.validate()?;
        match self.distribution {
            UncertaintyDistribution::BoundedUniform(bounds) => bounds.validate()?,
            UncertaintyDistribution::Normal {
                mean,
                standard_deviation,
                truncation,
            } => {
                if !mean.is_finite()
                    || !standard_deviation.is_finite()
                    || standard_deviation <= 0.0
                {
                    return Err(RobustnessError::InvalidNormalDistribution);
                }
                if let Some(bounds) = truncation {
                    bounds.validate()?;
                    if !bounds.contains(mean) {
                        return Err(RobustnessError::NormalMeanOutsideTruncation);
                    }
                }
            }
        }
        Ok(())
    }

    fn admits(&self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match self.distribution {
            UncertaintyDistribution::BoundedUniform(bounds) => bounds.contains(value),
            UncertaintyDistribution::Normal { truncation, .. } => {
                truncation.is_none_or(|bounds| bounds.contains(value))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingPolicy {
    /// Scenario points were supplied explicitly; no stochastic generator is implied.
    Explicit,
    /// Deterministic pseudo-random/sampling backend identity and seed.
    Seeded { method: String, seed: u64 },
}

impl SamplingPolicy {
    fn validate(&self) -> Result<(), RobustnessError> {
        if let Self::Seeded { method, .. } = self {
            require_nonempty("sampling.method", method)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioParameterValue {
    pub parameter_id: String,
    pub value: f64,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustScenario {
    pub id: String,
    /// Strictly ordered and complete for the uncertainty parameter registry.
    pub values: Vec<ScenarioParameterValue>,
}

/// Frozen uncertainty campaign definition for one nominal engineering candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustCampaignDefinition {
    pub id: String,
    pub optimization_profile_digest: String,
    pub nominal_candidate_digest: String,
    /// Strictly ordered by parameter ID.
    pub uncertain_parameters: Vec<UncertainParameter>,
    pub sampling_policy: SamplingPolicy,
    /// Strictly ordered by scenario ID; exact scenario population is load-bearing.
    pub scenarios: Vec<RobustScenario>,
}

impl RobustCampaignDefinition {
    pub fn validate(
        &self,
        profile: &OptimizationProfile,
        nominal_candidate: &DesignCandidate,
    ) -> Result<(), RobustnessError> {
        require_nonempty("campaign.id", &self.id)?;
        profile.validate()?;
        nominal_candidate.validate(profile)?;
        if self.optimization_profile_digest != profile.digest()? {
            return Err(RobustnessError::ProfileDigestMismatch);
        }
        if self.nominal_candidate_digest != nominal_candidate.digest(profile)? {
            return Err(RobustnessError::CandidateDigestMismatch);
        }
        if self.uncertain_parameters.is_empty() {
            return Err(RobustnessError::EmptyCollection(
                "campaign.uncertain_parameters",
            ));
        }
        if self.scenarios.is_empty() {
            return Err(RobustnessError::EmptyCollection("campaign.scenarios"));
        }
        ensure_ordered(
            "campaign.uncertain_parameters",
            self.uncertain_parameters.iter().map(|value| value.id.as_str()),
        )?;
        ensure_ordered(
            "campaign.scenarios",
            self.scenarios.iter().map(|value| value.id.as_str()),
        )?;
        self.sampling_policy.validate()?;
        for parameter in &self.uncertain_parameters {
            parameter.validate()?;
        }
        for scenario in &self.scenarios {
            self.validate_scenario(scenario)?;
        }
        Ok(())
    }

    fn validate_scenario(&self, scenario: &RobustScenario) -> Result<(), RobustnessError> {
        require_nonempty("scenario.id", &scenario.id)?;
        ensure_ordered(
            "scenario.values",
            scenario.values.iter().map(|value| value.parameter_id.as_str()),
        )?;
        if scenario.values.len() != self.uncertain_parameters.len() {
            return Err(RobustnessError::ScenarioDoesNotCoverRegistry(
                scenario.id.clone(),
            ));
        }
        for (value, parameter) in scenario.values.iter().zip(&self.uncertain_parameters) {
            if value.parameter_id != parameter.id || value.unit != parameter.unit {
                return Err(RobustnessError::ScenarioDoesNotCoverRegistry(
                    scenario.id.clone(),
                ));
            }
            if !parameter.admits(value.value) {
                return Err(RobustnessError::ScenarioValueOutsideDistribution {
                    scenario_id: scenario.id.clone(),
                    parameter_id: parameter.id.clone(),
                });
            }
        }
        Ok(())
    }

    /// Digest binds assumptions, source classes, sampling policy, seed, and exact
    /// scenario population. It is not evidence that the scenarios are independent.
    pub fn digest(
        &self,
        profile: &OptimizationProfile,
        nominal_candidate: &DesignCandidate,
    ) -> Result<String, RobustnessError> {
        self.validate(profile, nominal_candidate)?;
        let mut material = String::new();
        push(&mut material, "campaign", &self.id);
        push(
            &mut material,
            "profile_digest",
            &self.optimization_profile_digest,
        );
        push(
            &mut material,
            "candidate_digest",
            &self.nominal_candidate_digest,
        );
        match &self.sampling_policy {
            SamplingPolicy::Explicit => push(&mut material, "sampling", "explicit"),
            SamplingPolicy::Seeded { method, seed } => {
                push(&mut material, "sampling", "seeded");
                push(&mut material, "method", method);
                push(&mut material, "seed", &seed.to_string());
            }
        }
        for parameter in &self.uncertain_parameters {
            push(&mut material, "parameter", &parameter.id);
            push(&mut material, "unit", &parameter.unit);
            push(
                &mut material,
                "source_kind",
                uncertainty_source_name(parameter.source.kind),
            );
            push(&mut material, "provenance", &parameter.source.provenance);
            push(
                &mut material,
                "evidence",
                parameter.source.evidence_ref.as_deref().unwrap_or(""),
            );
            push(
                &mut material,
                "method",
                parameter.source.method.as_deref().unwrap_or(""),
            );
            match parameter.distribution {
                UncertaintyDistribution::BoundedUniform(bounds) => {
                    push(&mut material, "distribution", "bounded-uniform");
                    push_f64(&mut material, "lower", bounds.lower);
                    push_f64(&mut material, "upper", bounds.upper);
                }
                UncertaintyDistribution::Normal {
                    mean,
                    standard_deviation,
                    truncation,
                } => {
                    push(&mut material, "distribution", "normal");
                    push_f64(&mut material, "mean", mean);
                    push_f64(&mut material, "stddev", standard_deviation);
                    if let Some(bounds) = truncation {
                        push_f64(&mut material, "trunc_lower", bounds.lower);
                        push_f64(&mut material, "trunc_upper", bounds.upper);
                    }
                }
            }
        }
        for scenario in &self.scenarios {
            push(&mut material, "scenario", &scenario.id);
            for value in &scenario.values {
                push(&mut material, "parameter", &value.parameter_id);
                push(&mut material, "unit", &value.unit);
                push_f64(&mut material, "value", value.value);
            }
        }
        Ok(blake3::hash(material.as_bytes()).to_hex().to_string())
    }
}

/// Result of evaluating the nominal candidate under one uncertainty scenario.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioEvaluation {
    pub scenario_id: String,
    pub evaluation: CandidateEvaluation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustCampaignResult {
    pub campaign_digest: String,
    /// Strictly ordered by scenario ID and exactly one evaluation per scenario.
    pub evaluations: Vec<ScenarioEvaluation>,
}

impl RobustCampaignResult {
    pub fn validate(
        &self,
        campaign: &RobustCampaignDefinition,
        profile: &OptimizationProfile,
        nominal_candidate: &DesignCandidate,
    ) -> Result<(), RobustnessError> {
        campaign.validate(profile, nominal_candidate)?;
        if self.campaign_digest != campaign.digest(profile, nominal_candidate)? {
            return Err(RobustnessError::CampaignDigestMismatch);
        }
        ensure_ordered(
            "result.evaluations",
            self.evaluations.iter().map(|value| value.scenario_id.as_str()),
        )?;
        if self.evaluations.len() != campaign.scenarios.len() {
            return Err(RobustnessError::IncompleteScenarioPopulation);
        }
        let mut fidelity = None;
        for (evaluation, scenario) in self.evaluations.iter().zip(&campaign.scenarios) {
            if evaluation.scenario_id != scenario.id {
                return Err(RobustnessError::IncompleteScenarioPopulation);
            }
            evaluation.evaluation.validate(profile)?;
            if evaluation.evaluation.candidate.digest(profile)?
                != campaign.nominal_candidate_digest
            {
                return Err(RobustnessError::CandidateDigestMismatch);
            }
            match fidelity {
                None => fidelity = Some(evaluation.evaluation.fidelity.class()),
                Some(existing) if existing != evaluation.evaluation.fidelity.class() => {
                    return Err(RobustnessError::MixedFidelityCampaign);
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn summarize(
        &self,
        campaign: &RobustCampaignDefinition,
        profile: &OptimizationProfile,
        nominal_candidate: &DesignCandidate,
    ) -> Result<RobustSummary, RobustnessError> {
        self.validate(campaign, profile, nominal_candidate)?;
        let mut complete = 0usize;
        let mut infeasible = 0usize;
        let mut censored = 0usize;
        let mut samples: BTreeMap<String, Vec<f64>> = profile
            .objectives
            .iter()
            .map(|objective| (objective.id.clone(), Vec::new()))
            .collect();

        for scenario in &self.evaluations {
            match &scenario.evaluation.status {
                EvaluationStatus::Complete => {
                    complete += 1;
                    for objective in &scenario.evaluation.objectives {
                        if let Some(values) = samples.get_mut(&objective.objective_id) {
                            values.push(objective.value);
                        }
                    }
                }
                EvaluationStatus::Infeasible => infeasible += 1,
                EvaluationStatus::Failed { .. } => censored += 1,
            }
        }

        let classified = complete + infeasible;
        let evaluated_feasibility_fraction = if classified == 0 {
            None
        } else {
            Some(complete as f64 / classified as f64)
        };

        let mut objective_summaries = Vec::with_capacity(profile.objectives.len());
        for objective in &profile.objectives {
            let mut values = samples.remove(&objective.id).unwrap_or_default();
            values.sort_by(f64::total_cmp);
            if values.is_empty() {
                continue;
            }
            let minimum = values[0];
            let maximum = values[values.len() - 1];
            let p05 = nearest_rank(&values, 0.05);
            let p50 = nearest_rank(&values, 0.50);
            let p95 = nearest_rank(&values, 0.95);
            let worst_case = match objective.direction {
                ObjectiveDirection::Minimize => maximum,
                ObjectiveDirection::Maximize => minimum,
            };
            objective_summaries.push(ObjectiveRobustSummary {
                objective_id: objective.id.clone(),
                unit: objective.unit.clone(),
                sample_count: values.len(),
                minimum,
                p05,
                p50,
                p95,
                maximum,
                worst_case,
            });
        }

        Ok(RobustSummary {
            scenario_count: self.evaluations.len(),
            complete_feasible_count: complete,
            complete_infeasible_count: infeasible,
            failed_censored_count: censored,
            /// This is a scenario-classification fraction under the declared
            /// model/evidence fidelity, not manufacturing yield evidence.
            evaluated_feasibility_fraction,
            objective_summaries,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveRobustSummary {
    pub objective_id: String,
    pub unit: String,
    pub sample_count: usize,
    pub minimum: f64,
    pub p05: f64,
    pub p50: f64,
    pub p95: f64,
    pub maximum: f64,
    pub worst_case: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustSummary {
    pub scenario_count: usize,
    pub complete_feasible_count: usize,
    pub complete_infeasible_count: usize,
    pub failed_censored_count: usize,
    /// Fraction among classified (non-failed) scenarios. Never manufacturing yield.
    pub evaluated_feasibility_fraction: Option<f64>,
    pub objective_summaries: Vec<ObjectiveRobustSummary>,
}

fn nearest_rank(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

fn uncertainty_source_name(kind: UncertaintySourceKind) -> &'static str {
    match kind {
        UncertaintySourceKind::DatasheetTolerance => "datasheet-tolerance",
        UncertaintySourceKind::ManufacturingCapability => "manufacturing-capability",
        UncertaintySourceKind::MeasuredPopulation => "measured-population",
        UncertaintySourceKind::AssumedPrior => "assumed-prior",
        UncertaintySourceKind::FittedPosterior => "fitted-posterior",
    }
}

fn ensure_ordered<'a>(
    field: &'static str,
    values: impl Iterator<Item = &'a str>,
) -> Result<(), RobustnessError> {
    let values: Vec<_> = values.collect();
    if values.windows(2).any(|window| window[0] >= window[1]) {
        Err(RobustnessError::NonCanonicalOrder(field))
    } else {
        Ok(())
    }
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), RobustnessError> {
    if value.trim().is_empty() {
        Err(RobustnessError::EmptyIdentifier(field))
    } else {
        Ok(())
    }
}

fn push(material: &mut String, key: &str, value: &str) {
    material.push_str(&key.len().to_string());
    material.push(':');
    material.push_str(key);
    material.push('=');
    material.push_str(&value.len().to_string());
    material.push(':');
    material.push_str(value);
    material.push(';');
}

fn push_f64(material: &mut String, key: &str, value: f64) {
    push(material, key, &format!("{:016x}", value.to_bits()));
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum RobustnessError {
    #[error(transparent)]
    Optimization(#[from] OptimizationError),
    #[error("identifier {0} cannot be empty")]
    EmptyIdentifier(&'static str),
    #[error("collection {0} cannot be empty")]
    EmptyCollection(&'static str),
    #[error("{0} must be strictly ordered and duplicate-free")]
    NonCanonicalOrder(&'static str),
    #[error("{0:?} uncertainty source requires evidence_ref")]
    MissingEvidenceReference(UncertaintySourceKind),
    #[error("{0:?} uncertainty source requires method identity")]
    MissingMethod(UncertaintySourceKind),
    #[error("invalid bounds [{lower}, {upper}]")]
    InvalidBounds { lower: f64, upper: f64 },
    #[error("normal distribution requires finite mean and positive finite standard deviation")]
    InvalidNormalDistribution,
    #[error("normal mean must lie inside declared truncation")]
    NormalMeanOutsideTruncation,
    #[error("optimization profile digest mismatch")]
    ProfileDigestMismatch,
    #[error("nominal candidate digest mismatch")]
    CandidateDigestMismatch,
    #[error("scenario {0} does not exactly cover the uncertainty registry")]
    ScenarioDoesNotCoverRegistry(String),
    #[error("scenario {scenario_id} parameter {parameter_id} is outside admitted distribution")]
    ScenarioValueOutsideDistribution {
        scenario_id: String,
        parameter_id: String,
    },
    #[error("campaign digest mismatch")]
    CampaignDigestMismatch,
    #[error("result does not contain exactly one evaluation for every frozen scenario")]
    IncompleteScenarioPopulation,
    #[error("one robust campaign result cannot mix evaluation fidelity classes")]
    MixedFidelityCampaign,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_engineering_optimization::{
        ConstraintObservation, ConstraintRule, DesignAssignment, DesignValue, DesignVariable,
        EvaluationFidelity, HardConstraintSpec, ObjectiveObservation, ObjectiveSpec, VariableDomain,
    };

    fn profile() -> OptimizationProfile {
        OptimizationProfile {
            id: "robust-speaker-v1".into(),
            variables: vec![DesignVariable {
                id: "box_volume".into(),
                unit: Some("m^3".into()),
                domain: VariableDomain::Continuous {
                    lower: 0.01,
                    upper: 0.03,
                },
            }],
            objectives: vec![ObjectiveSpec {
                id: "max_spl".into(),
                unit: "dB".into(),
                direction: ObjectiveDirection::Maximize,
            }],
            hard_constraints: vec![HardConstraintSpec {
                id: "temp".into(),
                metric_id: "coil_temp".into(),
                unit: "degC".into(),
                rule: ConstraintRule::AtMost(180.0),
            }],
        }
    }

    fn candidate() -> DesignCandidate {
        DesignCandidate {
            id: "nominal-a".into(),
            assignments: vec![DesignAssignment {
                variable_id: "box_volume".into(),
                value: DesignValue::Continuous(0.02),
            }],
        }
    }

    fn measured_source() -> UncertaintySource {
        UncertaintySource {
            kind: UncertaintySourceKind::MeasuredPopulation,
            provenance: "driver-lot-study".into(),
            evidence_ref: Some("measurement:lot-study-001".into()),
            method: None,
        }
    }

    fn scenario(id: &str, bl: f64) -> RobustScenario {
        RobustScenario {
            id: id.into(),
            values: vec![ScenarioParameterValue {
                parameter_id: "force_factor".into(),
                value: bl,
                unit: "T*m".into(),
            }],
        }
    }

    fn campaign(seed: u64) -> RobustCampaignDefinition {
        let profile = profile();
        let candidate = candidate();
        RobustCampaignDefinition {
            id: "lot-robustness-v1".into(),
            optimization_profile_digest: profile.digest().unwrap(),
            nominal_candidate_digest: candidate.digest(&profile).unwrap(),
            uncertain_parameters: vec![UncertainParameter {
                id: "force_factor".into(),
                unit: "T*m".into(),
                distribution: UncertaintyDistribution::BoundedUniform(Bounds {
                    lower: 4.5,
                    upper: 5.5,
                }),
                source: measured_source(),
            }],
            sampling_policy: SamplingPolicy::Seeded {
                method: "latin-hypercube-v1".into(),
                seed,
            },
            scenarios: vec![scenario("s001", 4.7), scenario("s002", 5.3)],
        }
    }

    fn evaluation(id: &str, spl: f64, status: EvaluationStatus) -> CandidateEvaluation {
        let profile = profile();
        CandidateEvaluation {
            candidate: candidate(),
            profile_digest: profile.digest().unwrap(),
            fidelity: EvaluationFidelity::Numerical {
                solver_id: "solver-v1".into(),
            },
            evidence_ref: format!("solver-run:{id}"),
            objectives: match status {
                EvaluationStatus::Failed { .. } => vec![],
                _ => vec![ObjectiveObservation {
                    objective_id: "max_spl".into(),
                    value: spl,
                    unit: "dB".into(),
                    uncertainty: None,
                }],
            },
            constraints: match status {
                EvaluationStatus::Failed { .. } => vec![],
                EvaluationStatus::Infeasible => vec![ConstraintObservation {
                    constraint_id: "temp".into(),
                    value: 190.0,
                    unit: "degC".into(),
                    satisfied: false,
                }],
                EvaluationStatus::Complete => vec![ConstraintObservation {
                    constraint_id: "temp".into(),
                    value: 130.0,
                    unit: "degC".into(),
                    satisfied: true,
                }],
            },
            status,
        }
    }

    #[test]
    fn campaign_digest_binds_seed_and_scenario_population() {
        let profile = profile();
        let candidate = candidate();
        let a = campaign(7);
        let mut b = campaign(8);
        assert_ne!(
            a.digest(&profile, &candidate).unwrap(),
            b.digest(&profile, &candidate).unwrap()
        );
        b = campaign(7);
        b.scenarios[0].values[0].value = 4.8;
        assert_ne!(
            a.digest(&profile, &candidate).unwrap(),
            b.digest(&profile, &candidate).unwrap()
        );
    }

    #[test]
    fn assumed_prior_is_not_represented_as_measured_population() {
        let assumed = UncertaintySource {
            kind: UncertaintySourceKind::AssumedPrior,
            provenance: "engineering-prior".into(),
            evidence_ref: None,
            method: None,
        };
        assert!(assumed.validate().is_ok());
        let invalid_measured = UncertaintySource {
            kind: UncertaintySourceKind::MeasuredPopulation,
            provenance: "claimed-measurement".into(),
            evidence_ref: None,
            method: None,
        };
        assert!(matches!(
            invalid_measured.validate(),
            Err(RobustnessError::MissingEvidenceReference(
                UncertaintySourceKind::MeasuredPopulation
            ))
        ));
    }

    #[test]
    fn failed_scenario_is_censored_not_counted_as_infeasible() {
        let profile = profile();
        let candidate = candidate();
        let campaign = campaign(7);
        let result = RobustCampaignResult {
            campaign_digest: campaign.digest(&profile, &candidate).unwrap(),
            evaluations: vec![
                ScenarioEvaluation {
                    scenario_id: "s001".into(),
                    evaluation: evaluation("s001", 110.0, EvaluationStatus::Complete),
                },
                ScenarioEvaluation {
                    scenario_id: "s002".into(),
                    evaluation: evaluation(
                        "s002",
                        0.0,
                        EvaluationStatus::Failed {
                            reason: "solver failed".into(),
                        },
                    ),
                },
            ],
        };
        let summary = result.summarize(&campaign, &profile, &candidate).unwrap();
        assert_eq!(summary.complete_feasible_count, 1);
        assert_eq!(summary.complete_infeasible_count, 0);
        assert_eq!(summary.failed_censored_count, 1);
        assert_eq!(summary.evaluated_feasibility_fraction, Some(1.0));
        assert_eq!(summary.objective_summaries[0].sample_count, 1);
    }

    #[test]
    fn scenario_outside_declared_distribution_fails_closed() {
        let profile = profile();
        let candidate = candidate();
        let mut campaign = campaign(7);
        campaign.scenarios[1].values[0].value = 6.0;
        assert!(matches!(
            campaign.validate(&profile, &candidate),
            Err(RobustnessError::ScenarioValueOutsideDistribution { .. })
        ));
    }

    #[test]
    fn mixed_fidelity_campaign_is_rejected() {
        let profile = profile();
        let candidate = candidate();
        let campaign = campaign(7);
        let mut second = evaluation("s002", 111.0, EvaluationStatus::Complete);
        second.fidelity = EvaluationFidelity::Surrogate {
            model_id: "surrogate-v1".into(),
        };
        let result = RobustCampaignResult {
            campaign_digest: campaign.digest(&profile, &candidate).unwrap(),
            evaluations: vec![
                ScenarioEvaluation {
                    scenario_id: "s001".into(),
                    evaluation: evaluation("s001", 110.0, EvaluationStatus::Complete),
                },
                ScenarioEvaluation {
                    scenario_id: "s002".into(),
                    evaluation: second,
                },
            ],
        };
        assert_eq!(
            result.validate(&campaign, &profile, &candidate),
            Err(RobustnessError::MixedFidelityCampaign)
        );
    }

    #[test]
    fn robust_summary_reports_tail_and_worst_case_without_product_ranking() {
        let profile = profile();
        let candidate = candidate();
        let campaign = campaign(7);
        let result = RobustCampaignResult {
            campaign_digest: campaign.digest(&profile, &candidate).unwrap(),
            evaluations: vec![
                ScenarioEvaluation {
                    scenario_id: "s001".into(),
                    evaluation: evaluation("s001", 108.0, EvaluationStatus::Complete),
                },
                ScenarioEvaluation {
                    scenario_id: "s002".into(),
                    evaluation: evaluation("s002", 112.0, EvaluationStatus::Complete),
                },
            ],
        };
        let summary = result.summarize(&campaign, &profile, &candidate).unwrap();
        let objective = &summary.objective_summaries[0];
        assert_eq!(objective.minimum, 108.0);
        assert_eq!(objective.maximum, 112.0);
        assert_eq!(objective.worst_case, 108.0); // maximize objective
        assert_eq!(summary.evaluated_feasibility_fraction, Some(1.0));
    }
}
