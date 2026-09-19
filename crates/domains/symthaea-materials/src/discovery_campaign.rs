// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-objective materials-discovery campaign contracts.
//!
//! A campaign defines the decision problem: independent objectives, hard
//! feasibility constraints, operating-condition signatures, scale, budgets,
//! evidence progression, and stop rules. It deliberately does not define an
//! acquisition algorithm and never collapses these dimensions into a universal
//! weighted `best_material` score.

use crate::evidence::MaterialsEvidenceStage;
use crate::material_subject::SubjectLineageRef;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Physical scale at which campaign conclusions are valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CampaignScale {
    /// Small coupon or specimen.
    Coupon,
    /// Laboratory batch.
    LabBatch,
    /// Representative component/device.
    Device,
    /// Pilot-process scale.
    Pilot,
    /// Industrial/full-process scale.
    Industrial,
}

/// Semantic class of a campaign metric.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CampaignMetricKind {
    /// Scientific property bound to MAT-008 conditions.
    ConditionedProperty,
    /// Economic/resource metric produced by MAT-006 or equivalent evidence.
    Economic,
    /// Broader impact metric such as embodied carbon.
    Impact,
    /// Campaign resource-use metric.
    Resource,
    /// Explicit domain-specific class.
    Other(String),
}

/// Exact metric definition referenced by objectives and hard constraints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignMetricRef {
    /// Stable metric identifier.
    pub metric_id: String,
    /// Semantic class.
    pub kind: CampaignMetricKind,
    /// Explicit unit; no conversion is implicit in this module.
    pub unit: String,
    /// MAT-008 condition signature for conditioned scientific properties.
    pub condition_signature: Option<String>,
}

/// Direction of one independent optimization objective.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObjectiveDirection {
    /// Lower values are preferred.
    Minimize,
    /// Higher values are preferred.
    Maximize,
    /// Values inside the interval have zero objective loss.
    TargetInterval {
        /// Inclusive lower target.
        lower: f64,
        /// Inclusive upper target.
        upper: f64,
    },
}

/// One independent campaign objective.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignObjective {
    /// Stable objective identifier.
    pub objective_id: String,
    /// Metric to optimize.
    pub metric: CampaignMetricRef,
    /// Optimization direction.
    pub direction: ObjectiveDirection,
}

/// Hard feasibility relation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HardConstraintRelation {
    /// Value must be at least the bound.
    AtLeast(f64),
    /// Value must be at most the bound.
    AtMost(f64),
    /// Value must lie in the inclusive interval.
    Interval {
        /// Lower bound.
        lower: f64,
        /// Upper bound.
        upper: f64,
    },
}

/// One hard feasibility constraint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignHardConstraint {
    /// Stable constraint identifier.
    pub constraint_id: String,
    /// Metric constrained.
    pub metric: CampaignMetricRef,
    /// Required relation.
    pub relation: HardConstraintRelation,
}

/// Monetary ceiling with explicit currency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonetaryBudget {
    /// Positive maximum amount.
    pub amount: f64,
    /// Currency identifier.
    pub currency: String,
}

/// Optional resource ceilings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CampaignBudgets {
    /// Maximum aggregate compute core-hours.
    pub max_compute_core_hours: Option<f64>,
    /// Maximum number of physical experiments.
    pub max_physical_experiments: Option<u32>,
    /// Maximum feedstock/material mass in kg.
    pub max_material_mass_kg: Option<f64>,
    /// Maximum direct monetary spend.
    pub max_direct_cost: Option<MonetaryBudget>,
    /// Maximum evaluated attempts.
    pub max_attempts: Option<u32>,
}

/// Explicit campaign stopping rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CampaignStopRule {
    /// Stop after a result reaches this evidence stage.
    EvidenceStageReached(MaterialsEvidenceStage),
    /// Stop when an acquisition layer reports expected information gain below this threshold.
    ExpectedInformationGainBelow(f64),
    /// Stop when any declared budget is exhausted.
    BudgetExhausted,
    /// Stop after a fixed number of attempts.
    MaxAttempts(u32),
    /// Stop for human scientific review.
    ManualReviewRequired,
}

/// Immutable definition of one materials-discovery decision problem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveryCampaign {
    /// Schema version used by canonical identity.
    pub schema_version: u32,
    /// Stable campaign identifier.
    pub campaign_id: String,
    /// Human-facing title; deliberately excluded from canonical identity.
    pub display_title: String,
    /// Physical scale for which the campaign is valid.
    pub scale: CampaignScale,
    /// Stable search-space identifier.
    pub search_space_id: String,
    /// Exact generator/search-space lineage.
    pub generator_lineage: SubjectLineageRef,
    /// Independent objectives; ordering is non-semantic.
    pub objectives: Vec<CampaignObjective>,
    /// Hard constraints; ordering is non-semantic.
    pub hard_constraints: Vec<CampaignHardConstraint>,
    /// Resource ceilings.
    pub budgets: CampaignBudgets,
    /// Minimum evidence stage required before a subject may pass this campaign's scientific gate.
    pub minimum_progression_stage: MaterialsEvidenceStage,
    /// Explicit stop rules; ordering is non-semantic.
    pub stop_rules: Vec<CampaignStopRule>,
}

impl DiscoveryCampaign {
    /// Validate structure, metric definitions, bounds, and budgets.
    pub fn validate(&self) -> Result<(), DiscoveryCampaignError> {
        if self.schema_version != 1 {
            return Err(DiscoveryCampaignError::UnsupportedSchemaVersion(self.schema_version));
        }
        nonempty("campaign_id", &self.campaign_id)?;
        nonempty("display_title", &self.display_title)?;
        nonempty("search_space_id", &self.search_space_id)?;
        validate_lineage(&self.generator_lineage)?;
        if self.objectives.is_empty() {
            return Err(DiscoveryCampaignError::NoObjectives);
        }
        if self.stop_rules.is_empty() {
            return Err(DiscoveryCampaignError::NoStopRules);
        }

        let mut objective_ids = HashSet::new();
        let mut constraint_ids = HashSet::new();
        let mut metric_definitions: HashMap<&str, String> = HashMap::new();

        for objective in &self.objectives {
            nonempty("objective_id", &objective.objective_id)?;
            if !objective_ids.insert(objective.objective_id.as_str()) {
                return Err(DiscoveryCampaignError::DuplicateObjectiveId(
                    objective.objective_id.clone(),
                ));
            }
            validate_metric(&objective.metric)?;
            register_metric_definition(&mut metric_definitions, &objective.metric)?;
            validate_objective_direction(&objective.direction)?;
        }
        for constraint in &self.hard_constraints {
            nonempty("constraint_id", &constraint.constraint_id)?;
            if !constraint_ids.insert(constraint.constraint_id.as_str()) {
                return Err(DiscoveryCampaignError::DuplicateConstraintId(
                    constraint.constraint_id.clone(),
                ));
            }
            validate_metric(&constraint.metric)?;
            register_metric_definition(&mut metric_definitions, &constraint.metric)?;
            validate_constraint_relation(&constraint.relation)?;
        }
        validate_budgets(&self.budgets)?;
        for rule in &self.stop_rules {
            validate_stop_rule(rule)?;
        }
        Ok(())
    }

    /// Deterministic campaign identity. Collection order and display title do not matter.
    pub fn canonical_identity(&self) -> Result<String, DiscoveryCampaignError> {
        self.validate()?;
        let mut objectives = self.objectives.iter().map(objective_key).collect::<Vec<_>>();
        let mut constraints = self
            .hard_constraints
            .iter()
            .map(constraint_key)
            .collect::<Vec<_>>();
        let mut stop_rules = self.stop_rules.iter().map(stop_rule_key).collect::<Vec<_>>();
        objectives.sort();
        constraints.sort();
        stop_rules.sort();
        Ok(format!(
            "materials-campaign:v{}|id={}|scale={}|space={}|generator={}|objectives=[{}]|constraints=[{}]|budgets={}|min_stage={:?}|stop=[{}]",
            self.schema_version,
            token(&self.campaign_id),
            scale_key(self.scale),
            token(&self.search_space_id),
            lineage_key(&self.generator_lineage),
            objectives.join(";"),
            constraints.join(";"),
            budgets_key(&self.budgets),
            self.minimum_progression_stage,
            stop_rules.join(";")
        ))
    }
}

/// Decision-layer value projected from one or more evidence records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignMetricValue {
    /// Metric identifier.
    pub metric_id: String,
    /// Numeric value.
    pub value: f64,
    /// Explicit unit.
    pub unit: String,
}

/// Campaign projection for one exact MAT-007 subject.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignEvaluationPoint {
    /// Canonical MAT-007 subject identity.
    pub subject_identity: String,
    /// Values used for campaign decisions.
    pub metrics: Vec<CampaignMetricValue>,
    /// Underlying evidence IDs. These remain external to the campaign score space.
    pub source_evidence_ids: Vec<String>,
}

impl CampaignEvaluationPoint {
    fn metric_map(&self) -> Result<HashMap<&str, &CampaignMetricValue>, DiscoveryCampaignError> {
        nonempty("subject_identity", &self.subject_identity)?;
        if self.source_evidence_ids.is_empty() {
            return Err(DiscoveryCampaignError::NoSourceEvidence);
        }
        let mut evidence_ids = HashSet::new();
        for id in &self.source_evidence_ids {
            nonempty("source_evidence_id", id)?;
            if !evidence_ids.insert(id.as_str()) {
                return Err(DiscoveryCampaignError::DuplicateSourceEvidenceId(id.clone()));
            }
        }
        let mut metrics = HashMap::new();
        for metric in &self.metrics {
            nonempty("metric_id", &metric.metric_id)?;
            nonempty("metric unit", &metric.unit)?;
            finite("metric value", metric.value)?;
            if metrics.insert(metric.metric_id.as_str(), metric).is_some() {
                return Err(DiscoveryCampaignError::DuplicateMetricValue(metric.metric_id.clone()));
            }
        }
        Ok(metrics)
    }
}

/// Return whether every hard constraint is satisfied. Violations are exclusions,
/// never hidden soft penalties.
pub fn campaign_point_is_feasible(
    campaign: &DiscoveryCampaign,
    point: &CampaignEvaluationPoint,
) -> Result<bool, DiscoveryCampaignError> {
    campaign.validate()?;
    let metrics = point.metric_map()?;
    constraints_satisfied(campaign, &metrics)
}

/// Return every feasible nondominated point under the independent objectives.
/// No weighted winner is selected.
pub fn campaign_pareto_front(
    campaign: &DiscoveryCampaign,
    points: &[CampaignEvaluationPoint],
) -> Result<Vec<CampaignEvaluationPoint>, DiscoveryCampaignError> {
    campaign.validate()?;
    let mut candidates = Vec::new();
    for point in points {
        let metrics = point.metric_map()?;
        if !constraints_satisfied(campaign, &metrics)? {
            continue;
        }
        let losses = campaign
            .objectives
            .iter()
            .map(|objective| {
                let value = metric_value(&metrics, &objective.metric)?;
                Ok(objective_loss(value, &objective.direction))
            })
            .collect::<Result<Vec<_>, DiscoveryCampaignError>>()?;
        candidates.push((point.clone(), losses));
    }

    let mut frontier = Vec::new();
    'candidate: for (i, (point, losses)) in candidates.iter().enumerate() {
        for (j, (_, other_losses)) in candidates.iter().enumerate() {
            if i != j && dominates(other_losses, losses) {
                continue 'candidate;
            }
        }
        frontier.push(point.clone());
    }
    Ok(frontier)
}

fn constraints_satisfied(
    campaign: &DiscoveryCampaign,
    metrics: &HashMap<&str, &CampaignMetricValue>,
) -> Result<bool, DiscoveryCampaignError> {
    for constraint in &campaign.hard_constraints {
        let value = metric_value(metrics, &constraint.metric)?;
        let passes = match &constraint.relation {
            HardConstraintRelation::AtLeast(bound) => value >= *bound,
            HardConstraintRelation::AtMost(bound) => value <= *bound,
            HardConstraintRelation::Interval { lower, upper } => {
                value >= *lower && value <= *upper
            }
        };
        if !passes {
            return Ok(false);
        }
    }
    Ok(true)
}

fn metric_value(
    metrics: &HashMap<&str, &CampaignMetricValue>,
    reference: &CampaignMetricRef,
) -> Result<f64, DiscoveryCampaignError> {
    let metric = metrics
        .get(reference.metric_id.as_str())
        .ok_or_else(|| DiscoveryCampaignError::MissingMetric(reference.metric_id.clone()))?;
    if metric.unit != reference.unit {
        return Err(DiscoveryCampaignError::UnitMismatch {
            metric_id: reference.metric_id.clone(),
            expected: reference.unit.clone(),
            actual: metric.unit.clone(),
        });
    }
    Ok(metric.value)
}

fn objective_loss(value: f64, direction: &ObjectiveDirection) -> f64 {
    match direction {
        ObjectiveDirection::Minimize => value,
        ObjectiveDirection::Maximize => -value,
        ObjectiveDirection::TargetInterval { lower, upper } if value < *lower => lower - value,
        ObjectiveDirection::TargetInterval { lower, upper } if value > *upper => value - upper,
        ObjectiveDirection::TargetInterval { .. } => 0.0,
    }
}

fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut strictly_better = false;
    for (left, right) in a.iter().zip(b.iter()) {
        if left > right {
            return false;
        }
        strictly_better |= left < right;
    }
    strictly_better
}

fn validate_metric(metric: &CampaignMetricRef) -> Result<(), DiscoveryCampaignError> {
    nonempty("metric_id", &metric.metric_id)?;
    nonempty("metric unit", &metric.unit)?;
    match &metric.kind {
        CampaignMetricKind::ConditionedProperty => {
            let signature = metric
                .condition_signature
                .as_deref()
                .ok_or_else(|| DiscoveryCampaignError::MissingConditionSignature {
                    metric_id: metric.metric_id.clone(),
                })?;
            nonempty("condition_signature", signature)?;
        }
        CampaignMetricKind::Other(name) => {
            nonempty("metric kind", name)?;
            if let Some(signature) = &metric.condition_signature {
                nonempty("condition_signature", signature)?;
            }
        }
        CampaignMetricKind::Economic | CampaignMetricKind::Impact | CampaignMetricKind::Resource => {
            if let Some(signature) = &metric.condition_signature {
                nonempty("condition_signature", signature)?;
            }
        }
    }
    Ok(())
}

fn register_metric_definition<'a>(
    definitions: &mut HashMap<&'a str, String>,
    metric: &'a CampaignMetricRef,
) -> Result<(), DiscoveryCampaignError> {
    let definition = metric_definition_key(metric);
    if let Some(existing) = definitions.insert(metric.metric_id.as_str(), definition.clone()) {
        if existing != definition {
            return Err(DiscoveryCampaignError::ConflictingMetricDefinition(
                metric.metric_id.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_objective_direction(direction: &ObjectiveDirection) -> Result<(), DiscoveryCampaignError> {
    if let ObjectiveDirection::TargetInterval { lower, upper } = direction {
        finite("objective lower", *lower)?;
        finite("objective upper", *upper)?;
        if lower > upper {
            return Err(DiscoveryCampaignError::InvalidInterval);
        }
    }
    Ok(())
}

fn validate_constraint_relation(relation: &HardConstraintRelation) -> Result<(), DiscoveryCampaignError> {
    match relation {
        HardConstraintRelation::AtLeast(value) | HardConstraintRelation::AtMost(value) => {
            finite("constraint bound", *value)
        }
        HardConstraintRelation::Interval { lower, upper } => {
            finite("constraint lower", *lower)?;
            finite("constraint upper", *upper)?;
            if lower > upper {
                Err(DiscoveryCampaignError::InvalidInterval)
            } else {
                Ok(())
            }
        }
    }
}

fn validate_budgets(budgets: &CampaignBudgets) -> Result<(), DiscoveryCampaignError> {
    if let Some(value) = budgets.max_compute_core_hours {
        positive("max_compute_core_hours", value)?;
    }
    if let Some(value) = budgets.max_material_mass_kg {
        positive("max_material_mass_kg", value)?;
    }
    if budgets.max_physical_experiments == Some(0) {
        return Err(DiscoveryCampaignError::ZeroBudget("max_physical_experiments"));
    }
    if budgets.max_attempts == Some(0) {
        return Err(DiscoveryCampaignError::ZeroBudget("max_attempts"));
    }
    if let Some(money) = &budgets.max_direct_cost {
        positive("max_direct_cost", money.amount)?;
        nonempty("budget currency", &money.currency)?;
    }
    Ok(())
}

fn validate_stop_rule(rule: &CampaignStopRule) -> Result<(), DiscoveryCampaignError> {
    match rule {
        CampaignStopRule::ExpectedInformationGainBelow(value) => nonnegative("information gain", *value),
        CampaignStopRule::MaxAttempts(0) => Err(DiscoveryCampaignError::ZeroStopRuleAttempts),
        CampaignStopRule::EvidenceStageReached(_)
        | CampaignStopRule::BudgetExhausted
        | CampaignStopRule::MaxAttempts(_)
        | CampaignStopRule::ManualReviewRequired => Ok(()),
    }
}

fn validate_lineage(lineage: &SubjectLineageRef) -> Result<(), DiscoveryCampaignError> {
    nonempty("generator lineage_id", &lineage.lineage_id)?;
    if lineage.artifact_sha256.len() != 64
        || !lineage.artifact_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(DiscoveryCampaignError::InvalidSha256);
    }
    Ok(())
}

fn objective_key(objective: &CampaignObjective) -> String {
    format!(
        "{}:{}:{}",
        token(&objective.objective_id),
        metric_definition_key(&objective.metric),
        direction_key(&objective.direction)
    )
}

fn constraint_key(constraint: &CampaignHardConstraint) -> String {
    format!(
        "{}:{}:{}",
        token(&constraint.constraint_id),
        metric_definition_key(&constraint.metric),
        relation_key(&constraint.relation)
    )
}

fn metric_definition_key(metric: &CampaignMetricRef) -> String {
    format!(
        "{}:{}:{}",
        metric_kind_key(&metric.kind),
        token(&metric.unit),
        metric
            .condition_signature
            .as_deref()
            .map(token)
            .unwrap_or_else(|| "none".to_string())
    )
}

fn metric_kind_key(kind: &CampaignMetricKind) -> String {
    match kind {
        CampaignMetricKind::ConditionedProperty => "property".to_string(),
        CampaignMetricKind::Economic => "economic".to_string(),
        CampaignMetricKind::Impact => "impact".to_string(),
        CampaignMetricKind::Resource => "resource".to_string(),
        CampaignMetricKind::Other(name) => format!("other-{}", token(name)),
    }
}

fn direction_key(direction: &ObjectiveDirection) -> String {
    match direction {
        ObjectiveDirection::Minimize => "min".to_string(),
        ObjectiveDirection::Maximize => "max".to_string(),
        ObjectiveDirection::TargetInterval { lower, upper } => {
            format!("target[{},{}]", float_key(*lower), float_key(*upper))
        }
    }
}

fn relation_key(relation: &HardConstraintRelation) -> String {
    match relation {
        HardConstraintRelation::AtLeast(value) => format!("at-least:{}", float_key(*value)),
        HardConstraintRelation::AtMost(value) => format!("at-most:{}", float_key(*value)),
        HardConstraintRelation::Interval { lower, upper } => {
            format!("interval[{},{}]", float_key(*lower), float_key(*upper))
        }
    }
}

fn budgets_key(budgets: &CampaignBudgets) -> String {
    format!(
        "compute={}|experiments={}|mass={}|cost={}|attempts={}",
        opt_float(budgets.max_compute_core_hours),
        budgets.max_physical_experiments.map_or_else(|| "none".to_string(), |v| v.to_string()),
        opt_float(budgets.max_material_mass_kg),
        budgets.max_direct_cost.as_ref().map_or_else(
            || "none".to_string(),
            |money| format!("{}:{}", float_key(money.amount), token(&money.currency)),
        ),
        budgets.max_attempts.map_or_else(|| "none".to_string(), |v| v.to_string())
    )
}

fn stop_rule_key(rule: &CampaignStopRule) -> String {
    match rule {
        CampaignStopRule::EvidenceStageReached(stage) => format!("stage:{stage:?}"),
        CampaignStopRule::ExpectedInformationGainBelow(value) => {
            format!("eig-below:{}", float_key(*value))
        }
        CampaignStopRule::BudgetExhausted => "budget-exhausted".to_string(),
        CampaignStopRule::MaxAttempts(value) => format!("max-attempts:{value}"),
        CampaignStopRule::ManualReviewRequired => "manual-review".to_string(),
    }
}

fn lineage_key(lineage: &SubjectLineageRef) -> String {
    format!("{}:{}", token(&lineage.lineage_id), lineage.artifact_sha256.to_ascii_lowercase())
}

fn scale_key(scale: CampaignScale) -> &'static str {
    match scale {
        CampaignScale::Coupon => "coupon",
        CampaignScale::LabBatch => "lab-batch",
        CampaignScale::Device => "device",
        CampaignScale::Pilot => "pilot",
        CampaignScale::Industrial => "industrial",
    }
}

fn nonempty(field: &'static str, value: &str) -> Result<(), DiscoveryCampaignError> {
    if value.trim().is_empty() {
        Err(DiscoveryCampaignError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn finite(field: &'static str, value: f64) -> Result<(), DiscoveryCampaignError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DiscoveryCampaignError::NonFiniteValue { field, value })
    }
}

fn nonnegative(field: &'static str, value: f64) -> Result<(), DiscoveryCampaignError> {
    finite(field, value)?;
    if value < 0.0 {
        Err(DiscoveryCampaignError::NegativeValue { field, value })
    } else {
        Ok(())
    }
}

fn positive(field: &'static str, value: f64) -> Result<(), DiscoveryCampaignError> {
    finite(field, value)?;
    if value <= 0.0 {
        Err(DiscoveryCampaignError::NonPositiveValue { field, value })
    } else {
        Ok(())
    }
}

fn opt_float(value: Option<f64>) -> String {
    value.map_or_else(|| "none".to_string(), float_key)
}

fn float_key(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

fn token(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            output.push(byte as char);
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

/// Campaign validation/evaluation failure.
#[derive(Debug, Clone, PartialEq)]
pub enum DiscoveryCampaignError {
    /// Unsupported schema version.
    UnsupportedSchemaVersion(u32),
    /// Required textual field was empty.
    EmptyField(&'static str),
    /// Campaign has no objectives.
    NoObjectives,
    /// Campaign has no stop rules.
    NoStopRules,
    /// Duplicate objective ID.
    DuplicateObjectiveId(String),
    /// Duplicate constraint ID.
    DuplicateConstraintId(String),
    /// Same metric ID was given incompatible units/kind/conditions.
    ConflictingMetricDefinition(String),
    /// Conditioned property omitted its MAT-008 condition signature.
    MissingConditionSignature {
        /// Metric ID.
        metric_id: String,
    },
    /// Interval bounds were inverted.
    InvalidInterval,
    /// Non-finite numeric value.
    NonFiniteValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Negative value where non-negative was required.
    NegativeValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Non-positive value where positive was required.
    NonPositiveValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Explicit zero count budget.
    ZeroBudget(&'static str),
    /// Zero-attempt stop rule.
    ZeroStopRuleAttempts,
    /// Invalid lineage digest.
    InvalidSha256,
    /// Evaluation point has no bound source evidence.
    NoSourceEvidence,
    /// Duplicate source-evidence ID.
    DuplicateSourceEvidenceId(String),
    /// Duplicate metric value.
    DuplicateMetricValue(String),
    /// Required metric absent from evaluation point.
    MissingMetric(String),
    /// Unit differs from the campaign definition.
    UnitMismatch {
        /// Metric ID.
        metric_id: String,
        /// Expected unit.
        expected: String,
        /// Actual unit.
        actual: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn metric(id: &str, kind: CampaignMetricKind, unit: &str) -> CampaignMetricRef {
        CampaignMetricRef {
            metric_id: id.to_string(),
            condition_signature: matches!(kind, CampaignMetricKind::ConditionedProperty)
                .then(|| "T=300K|fixture".to_string()),
            kind,
            unit: unit.to_string(),
        }
    }

    fn campaign() -> DiscoveryCampaign {
        DiscoveryCampaign {
            schema_version: 1,
            campaign_id: "TIM-001-fixture".to_string(),
            display_title: "Thermal interface fixture".to_string(),
            scale: CampaignScale::Coupon,
            search_space_id: "tim-space-v1".to_string(),
            generator_lineage: SubjectLineageRef {
                lineage_id: "fixture-generator-v1".to_string(),
                artifact_sha256: A64.to_string(),
            },
            objectives: vec![
                CampaignObjective {
                    objective_id: "min-rth".to_string(),
                    metric: metric("thermal_resistance", CampaignMetricKind::ConditionedProperty, "K/W"),
                    direction: ObjectiveDirection::Minimize,
                },
                CampaignObjective {
                    objective_id: "min-cost".to_string(),
                    metric: metric("direct_cost", CampaignMetricKind::Economic, "USD"),
                    direction: ObjectiveDirection::Minimize,
                },
            ],
            hard_constraints: vec![CampaignHardConstraint {
                constraint_id: "max-temp".to_string(),
                metric: metric("peak_temperature", CampaignMetricKind::ConditionedProperty, "K"),
                relation: HardConstraintRelation::AtMost(400.0),
            }],
            budgets: CampaignBudgets {
                max_compute_core_hours: Some(100.0),
                max_physical_experiments: Some(20),
                max_material_mass_kg: Some(2.0),
                max_direct_cost: Some(MonetaryBudget {
                    amount: 5_000.0,
                    currency: "USD".to_string(),
                }),
                max_attempts: Some(500),
            },
            minimum_progression_stage: MaterialsEvidenceStage::ExperimentallyCharacterized,
            stop_rules: vec![CampaignStopRule::BudgetExhausted, CampaignStopRule::MaxAttempts(500)],
        }
    }

    fn point(id: &str, rth: f64, cost: f64, peak_temperature: f64) -> CampaignEvaluationPoint {
        CampaignEvaluationPoint {
            subject_identity: id.to_string(),
            metrics: vec![
                CampaignMetricValue { metric_id: "thermal_resistance".to_string(), value: rth, unit: "K/W".to_string() },
                CampaignMetricValue { metric_id: "direct_cost".to_string(), value: cost, unit: "USD".to_string() },
                CampaignMetricValue { metric_id: "peak_temperature".to_string(), value: peak_temperature, unit: "K".to_string() },
            ],
            source_evidence_ids: vec![format!("evidence-{id}")],
        }
    }

    #[test]
    fn collection_order_and_title_do_not_change_identity() {
        let a = campaign();
        let mut b = a.clone();
        b.display_title = "renamed".to_string();
        b.objectives.reverse();
        b.hard_constraints.reverse();
        b.stop_rules.reverse();
        assert_eq!(a.canonical_identity().unwrap(), b.canonical_identity().unwrap());
    }

    #[test]
    fn condition_change_changes_identity() {
        let a = campaign();
        let mut b = a.clone();
        b.objectives[0].metric.condition_signature = Some("T=350K|fixture".to_string());
        assert_ne!(a.canonical_identity().unwrap(), b.canonical_identity().unwrap());
    }

    #[test]
    fn conflicting_metric_redefinition_is_rejected() {
        let mut invalid = campaign();
        invalid.hard_constraints.push(CampaignHardConstraint {
            constraint_id: "same-id-different-unit".to_string(),
            metric: metric("direct_cost", CampaignMetricKind::Economic, "EUR"),
            relation: HardConstraintRelation::AtMost(100.0),
        });
        assert_eq!(
            invalid.validate(),
            Err(DiscoveryCampaignError::ConflictingMetricDefinition("direct_cost".to_string()))
        );
    }

    #[test]
    fn hard_constraint_excludes_infeasible_point() {
        let campaign = campaign();
        assert!(campaign_point_is_feasible(&campaign, &point("ok", 0.1, 100.0, 390.0)).unwrap());
        assert!(!campaign_point_is_feasible(&campaign, &point("hot", 0.05, 50.0, 450.0)).unwrap());
    }

    #[test]
    fn pareto_front_preserves_conflicting_tradeoffs() {
        let points = vec![
            point("low-rth", 0.05, 300.0, 390.0),
            point("low-cost", 0.20, 50.0, 390.0),
            point("dominated", 0.25, 400.0, 390.0),
            point("infeasible", 0.01, 1.0, 450.0),
        ];
        let front = campaign_pareto_front(&campaign(), &points).unwrap();
        let ids: HashSet<_> = front.iter().map(|point| point.subject_identity.as_str()).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains("low-rth"));
        assert!(ids.contains("low-cost"));
    }

    #[test]
    fn target_interval_has_zero_loss_inside_range() {
        let direction = ObjectiveDirection::TargetInterval { lower: 10.0, upper: 20.0 };
        assert_eq!(objective_loss(15.0, &direction), 0.0);
        assert_eq!(objective_loss(5.0, &direction), 5.0);
        assert_eq!(objective_loss(25.0, &direction), 5.0);
    }

    #[test]
    fn conditioned_property_requires_conditions() {
        let mut invalid = campaign();
        invalid.objectives[0].metric.condition_signature = None;
        assert!(matches!(
            invalid.validate(),
            Err(DiscoveryCampaignError::MissingConditionSignature { .. })
        ));
    }

    #[test]
    fn wrong_unit_is_not_silently_converted() {
        let mut invalid_point = point("wrong-unit", 0.1, 100.0, 390.0);
        invalid_point.metrics[0].unit = "mK/W".to_string();
        assert!(matches!(
            campaign_point_is_feasible(&campaign(), &invalid_point),
            Err(DiscoveryCampaignError::UnitMismatch { .. })
        ));
    }
}
