// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preregistered multi-scenario campaigns for resource shadow benchmarks.
//!
//! A strong per-scenario benchmark can still be biased by choosing favorable
//! scenarios, seeds, retries, or stopping points. This crate freezes those choices
//! before campaign evidence exists and refuses aggregate evaluation until every
//! preregistered scenario has exactly one append-only disposition.
//!
//! Failed scenarios remain evidence. A failed attempt cannot be silently replaced
//! by a successful retry; a retry must be preregistered as its own scenario ID.
//! Campaign summaries retain per-scenario dispositions and rank histograms rather
//! than collapsing the campaign to one scalar score or universal winner.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use blake3::Hasher;
use symthaea_operations_research::ObjectiveDirection;
use symthaea_resource_objective_metric::{
    ObjectiveMetric, ObjectiveMetricSchema, ObjectiveMetricSchemaError, ObjectiveStatistic,
};
use symthaea_resource_shadow_benchmark::{
    EvaluatedSelectorDecision, EvaluationObjective, SelectorKind, ShadowBenchmarkEvaluation,
};
use thiserror::Error;

pub const SHADOW_CAMPAIGN_PROTOCOL_V1: &str = "symthaea.resource-shadow-campaign.v1";

const ALL_SELECTORS: [SelectorKind; 4] = [
    SelectorKind::CanonicalFeasible,
    SelectorKind::ParetoUniqueFrontier,
    SelectorKind::ParetoCanonicalFrontier,
    SelectorKind::HdcShadow,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CampaignManifestId([u8; 32]);

impl CampaignManifestId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CampaignEvidenceId([u8; 32]);

impl CampaignEvidenceId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// One preregistered scenario/attempt.
///
/// The four commitments are opaque content digests supplied by the surrounding
/// reproducibility system. This crate commits to those bytes but does not prove the
/// authenticity or correctness of the referenced artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CampaignScenarioSpec {
    pub scenario_id: String,
    pub seed: u64,
    pub planning_input_commitment: [u8; 32],
    pub normalization_profile_commitment: [u8; 32],
    pub hdc_preference_profile_commitment: [u8; 32],
    pub outcome_protocol_commitment: [u8; 32],
}

impl CampaignScenarioSpec {
    fn validate(&self) -> Result<(), CampaignManifestError> {
        if self.scenario_id.trim().is_empty() {
            return Err(CampaignManifestError::BlankScenarioId);
        }
        for (name, commitment) in [
            ("planning_input", self.planning_input_commitment),
            ("normalization_profile", self.normalization_profile_commitment),
            ("hdc_preference_profile", self.hdc_preference_profile_commitment),
            ("outcome_protocol", self.outcome_protocol_commitment),
        ] {
            if commitment == [0; 32] {
                return Err(CampaignManifestError::ZeroScenarioCommitment {
                    scenario_id: self.scenario_id.clone(),
                    field: name,
                });
            }
        }
        Ok(())
    }
}

/// Frozen campaign definition. Scenario and objective input order is canonicalized
/// before identity is computed, so semantically identical sets have one ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CampaignManifest {
    campaign_label: String,
    evaluation_schema: ObjectiveMetricSchema,
    evaluation_objectives: Vec<EvaluationObjective>,
    scenarios: Vec<CampaignScenarioSpec>,
    manifest_id: CampaignManifestId,
}

impl CampaignManifest {
    pub fn new(
        campaign_label: impl Into<String>,
        mut evaluation_objectives: Vec<EvaluationObjective>,
        mut scenarios: Vec<CampaignScenarioSpec>,
    ) -> Result<Self, CampaignManifestError> {
        let campaign_label = campaign_label.into();
        if campaign_label.trim().is_empty() {
            return Err(CampaignManifestError::BlankCampaignLabel);
        }
        if evaluation_objectives.is_empty() {
            return Err(CampaignManifestError::NoEvaluationObjectives);
        }
        if scenarios.is_empty() {
            return Err(CampaignManifestError::NoScenarios);
        }

        let evaluation_schema = ObjectiveMetricSchema::new(
            evaluation_objectives
                .iter()
                .map(|objective| objective.metric.clone()),
        )
        .map_err(CampaignManifestError::InvalidEvaluationSchema)?;

        evaluation_objectives.sort_by(|left, right| {
            left.metric.objective_name.cmp(&right.metric.objective_name)
        });

        scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
        let mut ids = BTreeSet::new();
        for scenario in &scenarios {
            scenario.validate()?;
            if !ids.insert(scenario.scenario_id.clone()) {
                return Err(CampaignManifestError::DuplicateScenarioId(
                    scenario.scenario_id.clone(),
                ));
            }
        }

        let manifest_id = CampaignManifestId(hash_manifest(
            &campaign_label,
            &evaluation_objectives,
            &scenarios,
        ));

        Ok(Self {
            campaign_label,
            evaluation_schema,
            evaluation_objectives,
            scenarios,
            manifest_id,
        })
    }

    pub fn campaign_label(&self) -> &str {
        &self.campaign_label
    }

    pub fn manifest_id(&self) -> CampaignManifestId {
        self.manifest_id
    }

    pub fn evaluation_schema(&self) -> &ObjectiveMetricSchema {
        &self.evaluation_schema
    }

    pub fn evaluation_objectives(&self) -> &[EvaluationObjective] {
        &self.evaluation_objectives
    }

    pub fn scenarios(&self) -> &[CampaignScenarioSpec] {
        &self.scenarios
    }

    pub fn scenario(&self, scenario_id: &str) -> Option<&CampaignScenarioSpec> {
        self.scenarios
            .binary_search_by(|scenario| scenario.scenario_id.as_str().cmp(scenario_id))
            .ok()
            .map(|index| &self.scenarios[index])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CampaignManifestError {
    #[error("campaign label must not be blank")]
    BlankCampaignLabel,
    #[error("campaign must declare at least one evaluation objective")]
    NoEvaluationObjectives,
    #[error("campaign must preregister at least one scenario")]
    NoScenarios,
    #[error("invalid campaign evaluation schema: {0}")]
    InvalidEvaluationSchema(ObjectiveMetricSchemaError),
    #[error("scenario id must not be blank")]
    BlankScenarioId,
    #[error("duplicate scenario id {0}")]
    DuplicateScenarioId(String),
    #[error("scenario {scenario_id} has unset all-zero {field} commitment")]
    ZeroScenarioCommitment {
        scenario_id: String,
        field: &'static str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CampaignFailureStage {
    Setup,
    Decision,
    OutcomeCollection,
    Evaluation,
    Infrastructure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedScenarioRecord {
    pub scenario_id: String,
    pub stage: CampaignFailureStage,
    pub reason_code: String,
    pub evidence_commitment: [u8; 32],
    pub evidence_ref: String,
}

impl FailedScenarioRecord {
    pub fn new(
        scenario_id: impl Into<String>,
        stage: CampaignFailureStage,
        reason_code: impl Into<String>,
        evidence_commitment: [u8; 32],
        evidence_ref: impl Into<String>,
    ) -> Result<Self, CampaignRecordError> {
        let record = Self {
            scenario_id: scenario_id.into(),
            stage,
            reason_code: reason_code.into(),
            evidence_commitment,
            evidence_ref: evidence_ref.into(),
        };
        if record.scenario_id.trim().is_empty() {
            return Err(CampaignRecordError::BlankScenarioId);
        }
        if record.reason_code.trim().is_empty() {
            return Err(CampaignRecordError::BlankFailureReason);
        }
        if record.evidence_commitment == [0; 32] {
            return Err(CampaignRecordError::ZeroEvidenceCommitment);
        }
        if record.evidence_ref.trim().is_empty() {
            return Err(CampaignRecordError::BlankEvidenceRef);
        }
        Ok(record)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CampaignSelectorOutcome {
    Recommended {
        selector_id: String,
        candidate_id: String,
        outcome_pareto_rank: usize,
    },
    Abstained {
        selector_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatedScenarioRecord {
    scenario_id: String,
    evaluation_evidence_commitment: [u8; 32],
    evaluation_evidence_ref: String,
    outcome_frontier_ids: Vec<String>,
    selector_outcomes: BTreeMap<SelectorKind, CampaignSelectorOutcome>,
}

impl EvaluatedScenarioRecord {
    pub fn from_evaluation(
        manifest: &CampaignManifest,
        scenario_id: &str,
        evaluation_evidence_commitment: [u8; 32],
        evaluation_evidence_ref: impl Into<String>,
        evaluation: &ShadowBenchmarkEvaluation,
    ) -> Result<Self, CampaignRecordError> {
        if manifest.scenario(scenario_id).is_none() {
            return Err(CampaignRecordError::ScenarioOutsideManifest(
                scenario_id.to_owned(),
            ));
        }
        if evaluation_evidence_commitment == [0; 32] {
            return Err(CampaignRecordError::ZeroEvidenceCommitment);
        }
        let evaluation_evidence_ref = evaluation_evidence_ref.into();
        if evaluation_evidence_ref.trim().is_empty() {
            return Err(CampaignRecordError::BlankEvidenceRef);
        }

        let actual_objectives = canonical_objectives(evaluation.evaluation_objectives())?;
        if actual_objectives != manifest.evaluation_objectives {
            return Err(CampaignRecordError::EvaluationSchemaMismatch);
        }

        let mut selector_outcomes = BTreeMap::new();
        for selector in ALL_SELECTORS {
            let decision = evaluation
                .evaluated_decision(selector)
                .ok_or(CampaignRecordError::MissingSelectorEvaluation(selector))?;
            let outcome = match decision {
                EvaluatedSelectorDecision::Recommended {
                    selector_id,
                    candidate_id,
                    outcome_pareto_rank,
                    ..
                } => CampaignSelectorOutcome::Recommended {
                    selector_id: selector_id.clone(),
                    candidate_id: candidate_id.clone(),
                    outcome_pareto_rank: *outcome_pareto_rank,
                },
                EvaluatedSelectorDecision::Abstained { selector_id } => {
                    CampaignSelectorOutcome::Abstained {
                        selector_id: selector_id.clone(),
                    }
                }
            };
            selector_outcomes.insert(selector, outcome);
        }

        let mut outcome_frontier_ids = evaluation
            .outcome_frontier_ids()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        outcome_frontier_ids.sort();

        Ok(Self {
            scenario_id: scenario_id.to_owned(),
            evaluation_evidence_commitment,
            evaluation_evidence_ref,
            outcome_frontier_ids,
            selector_outcomes,
        })
    }

    pub fn scenario_id(&self) -> &str {
        &self.scenario_id
    }

    pub fn evaluation_evidence_commitment(&self) -> &[u8; 32] {
        &self.evaluation_evidence_commitment
    }

    pub fn evaluation_evidence_ref(&self) -> &str {
        &self.evaluation_evidence_ref
    }

    pub fn outcome_frontier_ids(&self) -> &[String] {
        &self.outcome_frontier_ids
    }

    pub fn selector_outcome(&self, selector: SelectorKind) -> Option<&CampaignSelectorOutcome> {
        self.selector_outcomes.get(&selector)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScenarioDisposition {
    Evaluated(EvaluatedScenarioRecord),
    Failed(FailedScenarioRecord),
}

impl ScenarioDisposition {
    pub fn scenario_id(&self) -> &str {
        match self {
            Self::Evaluated(record) => record.scenario_id(),
            Self::Failed(record) => &record.scenario_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CampaignRecordError {
    #[error("scenario id must not be blank")]
    BlankScenarioId,
    #[error("scenario {0} is outside the preregistered manifest")]
    ScenarioOutsideManifest(String),
    #[error("scenario {0} already has a final disposition; retries require a separate preregistered scenario id")]
    DuplicateScenarioDisposition(String),
    #[error("evaluation evidence commitment must not be all-zero")]
    ZeroEvidenceCommitment,
    #[error("evaluation/failure evidence ref must not be blank")]
    BlankEvidenceRef,
    #[error("failure reason code must not be blank")]
    BlankFailureReason,
    #[error("scenario evaluation objective schema differs from campaign manifest")]
    EvaluationSchemaMismatch,
    #[error("scenario evaluation is missing selector {0:?}")]
    MissingSelectorEvaluation(SelectorKind),
    #[error("invalid scenario evaluation objective schema: {0}")]
    InvalidEvaluationSchema(ObjectiveMetricSchemaError),
}

/// Append-only disposition ledger. There is intentionally no update/replace API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CampaignLedger {
    manifest: CampaignManifest,
    dispositions: BTreeMap<String, ScenarioDisposition>,
}

impl CampaignLedger {
    pub fn new(manifest: CampaignManifest) -> Self {
        Self {
            manifest,
            dispositions: BTreeMap::new(),
        }
    }

    pub fn manifest(&self) -> &CampaignManifest {
        &self.manifest
    }

    pub fn disposition(&self, scenario_id: &str) -> Option<&ScenarioDisposition> {
        self.dispositions.get(scenario_id)
    }

    pub fn missing_scenario_ids(&self) -> Vec<&str> {
        self.manifest
            .scenarios
            .iter()
            .filter(|scenario| !self.dispositions.contains_key(&scenario.scenario_id))
            .map(|scenario| scenario.scenario_id.as_str())
            .collect()
    }

    pub fn is_complete(&self) -> bool {
        self.dispositions.len() == self.manifest.scenarios.len()
    }

    pub fn record_evaluation(
        &mut self,
        scenario_id: &str,
        evaluation_evidence_commitment: [u8; 32],
        evaluation_evidence_ref: impl Into<String>,
        evaluation: &ShadowBenchmarkEvaluation,
    ) -> Result<(), CampaignRecordError> {
        let record = EvaluatedScenarioRecord::from_evaluation(
            &self.manifest,
            scenario_id,
            evaluation_evidence_commitment,
            evaluation_evidence_ref,
            evaluation,
        )?;
        self.insert_disposition(ScenarioDisposition::Evaluated(record))
    }

    pub fn record_failure(
        &mut self,
        record: FailedScenarioRecord,
    ) -> Result<(), CampaignRecordError> {
        self.insert_disposition(ScenarioDisposition::Failed(record))
    }

    fn insert_disposition(
        &mut self,
        disposition: ScenarioDisposition,
    ) -> Result<(), CampaignRecordError> {
        let scenario_id = disposition.scenario_id().to_owned();
        if self.manifest.scenario(&scenario_id).is_none() {
            return Err(CampaignRecordError::ScenarioOutsideManifest(scenario_id));
        }
        if self.dispositions.contains_key(&scenario_id) {
            return Err(CampaignRecordError::DuplicateScenarioDisposition(scenario_id));
        }
        self.dispositions.insert(scenario_id, disposition);
        Ok(())
    }

    pub fn finalize(self) -> Result<CampaignEvaluation, CampaignFinalizeError> {
        let missing = self
            .manifest
            .scenarios
            .iter()
            .filter(|scenario| !self.dispositions.contains_key(&scenario.scenario_id))
            .map(|scenario| scenario.scenario_id.clone())
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(CampaignFinalizeError::IncompleteCampaign { missing });
        }
        Ok(CampaignEvaluation::from_complete_ledger(self))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CampaignFinalizeError {
    #[error("campaign cannot finalize before every preregistered scenario has a disposition; missing {missing:?}")]
    IncompleteCampaign { missing: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectorCampaignSummary {
    pub selector: SelectorKind,
    pub evaluated_scenarios: usize,
    pub recommended: usize,
    pub abstained: usize,
    pub outcome_frontier_selections: usize,
    pub outcome_rank_histogram: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CampaignEvaluation {
    manifest: CampaignManifest,
    dispositions: BTreeMap<String, ScenarioDisposition>,
    selector_summaries: BTreeMap<SelectorKind, SelectorCampaignSummary>,
    failure_histogram: BTreeMap<CampaignFailureStage, usize>,
    evidence_id: CampaignEvidenceId,
}

impl CampaignEvaluation {
    fn from_complete_ledger(ledger: CampaignLedger) -> Self {
        debug_assert!(ledger.is_complete());

        let mut selector_summaries = ALL_SELECTORS
            .into_iter()
            .map(|selector| {
                (
                    selector,
                    SelectorCampaignSummary {
                        selector,
                        evaluated_scenarios: 0,
                        recommended: 0,
                        abstained: 0,
                        outcome_frontier_selections: 0,
                        outcome_rank_histogram: BTreeMap::new(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut failure_histogram = BTreeMap::new();

        for disposition in ledger.dispositions.values() {
            match disposition {
                ScenarioDisposition::Evaluated(record) => {
                    for selector in ALL_SELECTORS {
                        let summary = selector_summaries
                            .get_mut(&selector)
                            .expect("all benchmark selectors are initialized");
                        summary.evaluated_scenarios += 1;
                        match record
                            .selector_outcome(selector)
                            .expect("evaluated record contains all selectors")
                        {
                            CampaignSelectorOutcome::Recommended {
                                outcome_pareto_rank,
                                ..
                            } => {
                                summary.recommended += 1;
                                if *outcome_pareto_rank == 0 {
                                    summary.outcome_frontier_selections += 1;
                                }
                                *summary
                                    .outcome_rank_histogram
                                    .entry(*outcome_pareto_rank)
                                    .or_insert(0) += 1;
                            }
                            CampaignSelectorOutcome::Abstained { .. } => {
                                summary.abstained += 1;
                            }
                        }
                    }
                }
                ScenarioDisposition::Failed(record) => {
                    *failure_histogram.entry(record.stage).or_insert(0) += 1;
                }
            }
        }

        let evidence_id = CampaignEvidenceId(hash_campaign_evidence(
            ledger.manifest.manifest_id,
            &ledger.dispositions,
        ));

        Self {
            manifest: ledger.manifest,
            dispositions: ledger.dispositions,
            selector_summaries,
            failure_histogram,
            evidence_id,
        }
    }

    pub fn manifest(&self) -> &CampaignManifest {
        &self.manifest
    }

    pub fn evidence_id(&self) -> CampaignEvidenceId {
        self.evidence_id
    }

    pub fn dispositions(&self) -> impl Iterator<Item = (&str, &ScenarioDisposition)> {
        self.dispositions
            .iter()
            .map(|(id, disposition)| (id.as_str(), disposition))
    }

    pub fn selector_summary(&self, selector: SelectorKind) -> &SelectorCampaignSummary {
        self.selector_summaries
            .get(&selector)
            .expect("all benchmark selectors have a campaign summary")
    }

    pub fn failure_histogram(&self) -> &BTreeMap<CampaignFailureStage, usize> {
        &self.failure_histogram
    }

    pub fn failed_scenarios(&self) -> usize {
        self.failure_histogram.values().sum()
    }

    pub fn evaluated_scenarios(&self) -> usize {
        self.dispositions.len() - self.failed_scenarios()
    }
}

fn canonical_objectives(
    objectives: &[EvaluationObjective],
) -> Result<Vec<EvaluationObjective>, CampaignRecordError> {
    ObjectiveMetricSchema::new(objectives.iter().map(|objective| objective.metric.clone()))
        .map_err(CampaignRecordError::InvalidEvaluationSchema)?;
    let mut ordered = objectives.to_vec();
    ordered.sort_by(|left, right| left.metric.objective_name.cmp(&right.metric.objective_name));
    Ok(ordered)
}

fn hash_manifest(
    campaign_label: &str,
    objectives: &[EvaluationObjective],
    scenarios: &[CampaignScenarioSpec],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    frame(&mut hasher, SHADOW_CAMPAIGN_PROTOCOL_V1.as_bytes());
    frame(&mut hasher, campaign_label.as_bytes());
    for selector in ALL_SELECTORS {
        frame(&mut hasher, selector_tag(selector));
    }
    for objective in objectives {
        hash_metric(&mut hasher, &objective.metric);
        frame(&mut hasher, direction_tag(objective.direction));
    }
    for scenario in scenarios {
        frame(&mut hasher, scenario.scenario_id.as_bytes());
        hasher.update(&scenario.seed.to_le_bytes());
        hasher.update(&scenario.planning_input_commitment);
        hasher.update(&scenario.normalization_profile_commitment);
        hasher.update(&scenario.hdc_preference_profile_commitment);
        hasher.update(&scenario.outcome_protocol_commitment);
    }
    *hasher.finalize().as_bytes()
}

fn hash_campaign_evidence(
    manifest_id: CampaignManifestId,
    dispositions: &BTreeMap<String, ScenarioDisposition>,
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    frame(&mut hasher, b"symthaea.resource-shadow-campaign.evidence.v1");
    hasher.update(manifest_id.as_bytes());
    for (scenario_id, disposition) in dispositions {
        frame(&mut hasher, scenario_id.as_bytes());
        match disposition {
            ScenarioDisposition::Evaluated(record) => {
                frame(&mut hasher, b"evaluated");
                hasher.update(&record.evaluation_evidence_commitment);
                frame(&mut hasher, record.evaluation_evidence_ref.as_bytes());
                for candidate_id in &record.outcome_frontier_ids {
                    frame(&mut hasher, candidate_id.as_bytes());
                }
                for selector in ALL_SELECTORS {
                    frame(&mut hasher, selector_tag(selector));
                    match record
                        .selector_outcome(selector)
                        .expect("evaluated record has all selectors")
                    {
                        CampaignSelectorOutcome::Recommended {
                            selector_id,
                            candidate_id,
                            outcome_pareto_rank,
                        } => {
                            frame(&mut hasher, b"recommend");
                            frame(&mut hasher, selector_id.as_bytes());
                            frame(&mut hasher, candidate_id.as_bytes());
                            hasher.update(&(*outcome_pareto_rank as u64).to_le_bytes());
                        }
                        CampaignSelectorOutcome::Abstained { selector_id } => {
                            frame(&mut hasher, b"abstain");
                            frame(&mut hasher, selector_id.as_bytes());
                        }
                    }
                }
            }
            ScenarioDisposition::Failed(record) => {
                frame(&mut hasher, b"failed");
                frame(&mut hasher, failure_stage_tag(record.stage));
                frame(&mut hasher, record.reason_code.as_bytes());
                hasher.update(&record.evidence_commitment);
                frame(&mut hasher, record.evidence_ref.as_bytes());
            }
        }
    }
    *hasher.finalize().as_bytes()
}

fn hash_metric(hasher: &mut Hasher, metric: &ObjectiveMetric) {
    frame(hasher, metric.objective_name.as_bytes());
    frame(hasher, metric.metric_id.as_bytes());
    frame(hasher, metric.unit_id.as_bytes());
    match metric.statistic {
        ObjectiveStatistic::CandidateTotal => frame(hasher, b"candidate_total"),
        ObjectiveStatistic::CandidateMean => frame(hasher, b"candidate_mean"),
        ObjectiveStatistic::CandidateMinimum => frame(hasher, b"candidate_minimum"),
        ObjectiveStatistic::CandidateMaximum => frame(hasher, b"candidate_maximum"),
        ObjectiveStatistic::CandidateFinal => frame(hasher, b"candidate_final"),
        ObjectiveStatistic::CandidateCount => frame(hasher, b"candidate_count"),
        ObjectiveStatistic::CandidateFraction => frame(hasher, b"candidate_fraction"),
        ObjectiveStatistic::PercentileBasisPoints(value) => {
            frame(hasher, b"percentile_basis_points");
            hasher.update(&value.to_le_bytes());
        }
    }
}

fn frame(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn direction_tag(direction: ObjectiveDirection) -> &'static [u8] {
    match direction {
        ObjectiveDirection::Minimize => b"minimize",
        ObjectiveDirection::Maximize => b"maximize",
    }
}

fn selector_tag(selector: SelectorKind) -> &'static [u8] {
    match selector {
        SelectorKind::CanonicalFeasible => b"canonical_feasible",
        SelectorKind::ParetoUniqueFrontier => b"pareto_unique_frontier",
        SelectorKind::ParetoCanonicalFrontier => b"pareto_canonical_frontier",
        SelectorKind::HdcShadow => b"hdc_shadow",
    }
}

fn failure_stage_tag(stage: CampaignFailureStage) -> &'static [u8] {
    match stage {
        CampaignFailureStage::Setup => b"setup",
        CampaignFailureStage::Decision => b"decision",
        CampaignFailureStage::OutcomeCollection => b"outcome_collection",
        CampaignFailureStage::Evaluation => b"evaluation",
        CampaignFailureStage::Infrastructure => b"infrastructure",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn cost() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "cost",
            "cost.realized.usd_micro.v1",
            "currency.usd_micro.v1",
            ObjectiveStatistic::CandidateTotal,
        )
        .unwrap()
    }

    fn latency() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "latency",
            "latency.realized.p95.second.v1",
            "si.second.v1",
            ObjectiveStatistic::PercentileBasisPoints(9500),
        )
        .unwrap()
    }

    fn objectives() -> Vec<EvaluationObjective> {
        vec![
            EvaluationObjective::new(cost(), ObjectiveDirection::Minimize),
            EvaluationObjective::new(latency(), ObjectiveDirection::Minimize),
        ]
    }

    fn scenario(id: &str, seed: u64, offset: u8) -> CampaignScenarioSpec {
        CampaignScenarioSpec {
            scenario_id: id.into(),
            seed,
            planning_input_commitment: digest(offset),
            normalization_profile_commitment: digest(offset + 1),
            hdc_preference_profile_commitment: digest(offset + 2),
            outcome_protocol_commitment: digest(offset + 3),
        }
    }

    fn manifest() -> CampaignManifest {
        CampaignManifest::new(
            "compute-commons-shadow-v1",
            objectives(),
            vec![scenario("b", 2, 10), scenario("a", 1, 20)],
        )
        .unwrap()
    }

    #[test]
    fn manifest_identity_is_independent_of_input_order() {
        let a = manifest();
        let mut reversed_objectives = objectives();
        reversed_objectives.reverse();
        let b = CampaignManifest::new(
            "compute-commons-shadow-v1",
            reversed_objectives,
            vec![scenario("a", 1, 20), scenario("b", 2, 10)],
        )
        .unwrap();
        assert_eq!(a.manifest_id(), b.manifest_id());
        assert_eq!(
            a.scenarios().iter().map(|s| s.scenario_id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn seed_and_metric_semantics_are_manifest_identity() {
        let base = manifest();
        let changed_seed = CampaignManifest::new(
            "compute-commons-shadow-v1",
            objectives(),
            vec![scenario("b", 3, 10), scenario("a", 1, 20)],
        )
        .unwrap();
        assert_ne!(base.manifest_id(), changed_seed.manifest_id());

        let milliseconds = ObjectiveMetric::new(
            "latency",
            "latency.realized.p95.millisecond.v1",
            "si.millisecond.v1",
            ObjectiveStatistic::PercentileBasisPoints(9500),
        )
        .unwrap();
        let changed_metric = CampaignManifest::new(
            "compute-commons-shadow-v1",
            vec![
                EvaluationObjective::new(cost(), ObjectiveDirection::Minimize),
                EvaluationObjective::new(milliseconds, ObjectiveDirection::Minimize),
            ],
            vec![scenario("b", 2, 10), scenario("a", 1, 20)],
        )
        .unwrap();
        assert_ne!(base.manifest_id(), changed_metric.manifest_id());
    }

    #[test]
    fn unset_commitment_fails_preregistration() {
        let mut bad = scenario("a", 1, 20);
        bad.planning_input_commitment = [0; 32];
        assert!(matches!(
            CampaignManifest::new("campaign", objectives(), vec![bad]),
            Err(CampaignManifestError::ZeroScenarioCommitment { field: "planning_input", .. })
        ));
    }

    #[test]
    fn incomplete_campaign_cannot_finalize() {
        let ledger = CampaignLedger::new(manifest());
        assert!(matches!(
            ledger.finalize(),
            Err(CampaignFinalizeError::IncompleteCampaign { missing }) if missing == vec!["a".to_string(), "b".to_string()]
        ));
    }

    #[test]
    fn failure_is_a_final_campaign_disposition_not_an_omitted_trial() {
        let manifest = CampaignManifest::new(
            "campaign",
            objectives(),
            vec![scenario("a", 1, 20)],
        )
        .unwrap();
        let mut ledger = CampaignLedger::new(manifest);
        ledger
            .record_failure(
                FailedScenarioRecord::new(
                    "a",
                    CampaignFailureStage::OutcomeCollection,
                    "sensor-timeout",
                    digest(90),
                    "evidence://scenario-a/failure",
                )
                .unwrap(),
            )
            .unwrap();
        let evaluation = ledger.finalize().unwrap();
        assert_eq!(evaluation.failed_scenarios(), 1);
        assert_eq!(evaluation.evaluated_scenarios(), 0);
        assert_eq!(
            evaluation
                .failure_histogram()
                .get(&CampaignFailureStage::OutcomeCollection),
            Some(&1)
        );
    }

    #[test]
    fn recorded_failure_cannot_be_replaced_by_retry() {
        let manifest = CampaignManifest::new(
            "campaign",
            objectives(),
            vec![scenario("a", 1, 20)],
        )
        .unwrap();
        let mut ledger = CampaignLedger::new(manifest);
        let failure = FailedScenarioRecord::new(
            "a",
            CampaignFailureStage::Infrastructure,
            "runner-lost",
            digest(91),
            "evidence://scenario-a/failure",
        )
        .unwrap();
        ledger.record_failure(failure.clone()).unwrap();
        assert!(matches!(
            ledger.record_failure(failure),
            Err(CampaignRecordError::DuplicateScenarioDisposition(id)) if id == "a"
        ));
    }

    fn synthetic_evaluated_record(id: &str, commitment: u8) -> EvaluatedScenarioRecord {
        let selector_outcomes = ALL_SELECTORS
            .into_iter()
            .enumerate()
            .map(|(index, selector)| {
                (
                    selector,
                    CampaignSelectorOutcome::Recommended {
                        selector_id: format!("selector-{index}"),
                        candidate_id: if index % 2 == 0 { "a" } else { "b" }.into(),
                        outcome_pareto_rank: index % 2,
                    },
                )
            })
            .collect();
        EvaluatedScenarioRecord {
            scenario_id: id.into(),
            evaluation_evidence_commitment: digest(commitment),
            evaluation_evidence_ref: format!("evidence://{id}/evaluation"),
            outcome_frontier_ids: vec!["a".into()],
            selector_outcomes,
        }
    }

    #[test]
    fn finalized_summary_preserves_rank_distribution_not_only_average() {
        let mut ledger = CampaignLedger::new(manifest());
        ledger
            .insert_disposition(ScenarioDisposition::Evaluated(synthetic_evaluated_record(
                "a", 70,
            )))
            .unwrap();
        ledger
            .insert_disposition(ScenarioDisposition::Evaluated(synthetic_evaluated_record(
                "b", 71,
            )))
            .unwrap();
        let evaluation = ledger.finalize().unwrap();
        let baseline = evaluation.selector_summary(SelectorKind::CanonicalFeasible);
        assert_eq!(baseline.evaluated_scenarios, 2);
        assert_eq!(baseline.outcome_frontier_selections, 2);
        assert_eq!(baseline.outcome_rank_histogram.get(&0), Some(&2));
        assert_eq!(evaluation.dispositions().count(), 2);
    }

    #[test]
    fn campaign_evidence_identity_commits_failure_evidence() {
        fn run(commitment: u8) -> CampaignEvidenceId {
            let manifest = CampaignManifest::new(
                "campaign",
                objectives(),
                vec![scenario("a", 1, 20)],
            )
            .unwrap();
            let mut ledger = CampaignLedger::new(manifest);
            ledger
                .record_failure(
                    FailedScenarioRecord::new(
                        "a",
                        CampaignFailureStage::Evaluation,
                        "evaluation-error",
                        digest(commitment),
                        "evidence://failure",
                    )
                    .unwrap(),
                )
                .unwrap();
            ledger.finalize().unwrap().evidence_id()
        }
        assert_ne!(run(80), run(81));
    }
}
