// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bounded planning contracts for materials research opportunities.
//!
//! These types describe research priorities and bottleneck hypotheses. They are
//! proposal-plane data only: none of them is scientific property evidence.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

const FUNCTION_DOMAIN: &[u8] = b"symthaea-materials-opportunity-function-v1\0";
const BOTTLENECK_DOMAIN: &[u8] = b"symthaea-materials-opportunity-bottleneck-v1\0";
const INTERVENTION_DOMAIN: &[u8] = b"symthaea-materials-opportunity-intervention-v1\0";
const OPPORTUNITY_DOMAIN: &[u8] = b"symthaea-materials-opportunity-v1\0";

/// Deterministic BLAKE3 identity for one planning subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OpportunityId([u8; 32]);

impl OpportunityId {
    /// Returns the raw identity bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for OpportunityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Validation error for opportunity-planning data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpportunityError {
    /// A required field was empty.
    EmptyField(&'static str),
    /// A required collection was empty.
    EmptyCollection(&'static str),
    /// A set-like collection contained a duplicate.
    DuplicateValue(&'static str),
    /// An opportunity axis appeared more than once.
    DuplicateAxis(OpportunityAxis),
    /// A duplicate intervention was supplied.
    DuplicateIntervention,
    /// A numeric assessment was NaN or infinite.
    NonFiniteValue(OpportunityAxis),
    /// An interval lower bound exceeded its upper bound.
    InvalidInterval(OpportunityAxis),
    /// A numeric assessment had no unit.
    EmptyUnit(OpportunityAxis),
}

impl fmt::Display for OpportunityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(name) => write!(f, "required field `{name}` is empty"),
            Self::EmptyCollection(name) => write!(f, "required collection `{name}` is empty"),
            Self::DuplicateValue(name) => write!(f, "collection `{name}` contains a duplicate"),
            Self::DuplicateAxis(axis) => write!(f, "opportunity axis `{axis:?}` is duplicated"),
            Self::DuplicateIntervention => write!(f, "research intervention is duplicated"),
            Self::NonFiniteValue(axis) => write!(f, "axis `{axis:?}` contains a non-finite value"),
            Self::InvalidInterval(axis) => write!(f, "axis `{axis:?}` contains an invalid interval"),
            Self::EmptyUnit(axis) => write!(f, "axis `{axis:?}` has a numeric value without a unit"),
        }
    }
}

impl std::error::Error for OpportunityError {}

/// Evidence state for a bottleneck hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BottleneckEvidenceState {
    /// Evidence is weak or mainly hypothetical.
    Weak,
    /// Evidence suggests the bottleneck but alternatives remain plausible.
    Candidate,
    /// Evidence establishes the bottleneck only under an explicit bounded profile.
    EstablishedUnderProfile,
    /// Available evidence is insufficient for a stronger classification.
    EvidenceLimited,
}

/// Research intervention class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionClass {
    /// Replace an incumbent material while preserving its required function.
    MaterialSubstitution,
    /// Reduce material required per delivered function.
    MaterialIntensityReduction,
    /// Change geometry, layering, or system architecture.
    ArchitectureChange,
    /// Improve an interface rather than only the bulk material.
    InterfaceEngineering,
    /// Improve synthesis, manufacturing, refining, or another process route.
    ProcessImprovement,
    /// Recover or recycle material from an existing stream or product.
    RecoveryOrRecycling,
    /// Extend useful lifetime to reduce replacement burden.
    LifetimeExtension,
    /// Repair or reuse an existing material-bearing article.
    RepairOrReuse,
    /// Deliver the same function through a different technology route.
    AlternativeTechnologyFunction,
    /// Improve measurement or modeling when uncertainty is itself limiting.
    MeasurementOrModelImprovement,
}

/// Independent dimensions of a materials research opportunity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpportunityAxis {
    /// Expected leverage on the parent technology or service function.
    SystemLeverage,
    /// Remaining performance gap under the exact target envelope.
    PerformanceHeadroom,
    /// Potential to reduce critical-material or scarce-resource burden.
    CriticalityLeverage,
    /// Breadth of downstream functions that may benefit.
    DownstreamBreadth,
    /// Scientific tractability of the proposed question.
    ScientificTractability,
    /// Ability to observe the proposition computationally.
    ComputationalObservability,
    /// Availability and quality of relevant data.
    DataAvailability,
    /// Accessibility of meaningful physical experiments.
    ExperimentalAccessibility,
    /// Time or resource burden of decisive falsification.
    FalsificationCost,
    /// Readiness of the required scientific/evidence infrastructure.
    InfrastructureReadiness,
    /// Quality of evidence supporting the planning assessment.
    EvidenceQuality,
}

/// Coarse ordinal planning value with no implied cardinal spacing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanningOrdinal {
    /// Very low under the exact planning profile.
    VeryLow,
    /// Low under the exact planning profile.
    Low,
    /// Intermediate under the exact planning profile.
    Medium,
    /// High under the exact planning profile.
    High,
    /// Very high under the exact planning profile.
    VeryHigh,
}

/// Typed planning value for one independent axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentValue {
    /// Unknown rather than silently assumed to be zero.
    Unknown,
    /// Coarse ordinal assessment.
    Ordinal(PlanningOrdinal),
    /// Scalar value with declared unit.
    Scalar(f64, String),
    /// Closed interval with declared unit.
    Interval(f64, f64, String),
}

/// Technology or service function whose material dependence is under study.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TechnologyFunctionSubjectV1 {
    domain: String,
    functional_role: String,
    operating_context: String,
    required_performance: Vec<String>,
    evidence_refs: Vec<String>,
}

impl TechnologyFunctionSubjectV1 {
    /// Builds a validated technology-function planning subject.
    pub fn new(
        domain: impl Into<String>,
        functional_role: impl Into<String>,
        operating_context: impl Into<String>,
        required_performance: Vec<String>,
        evidence_refs: Vec<String>,
    ) -> Result<Self, OpportunityError> {
        let value = Self {
            domain: clean(domain.into(), "domain")?,
            functional_role: clean(functional_role.into(), "functional_role")?,
            operating_context: clean(operating_context.into(), "operating_context")?,
            required_performance: clean_set(required_performance, "required_performance", false)?,
            evidence_refs: clean_set(evidence_refs, "evidence_refs", false)?,
        };
        value.validate()?;
        Ok(value)
    }

    /// Validates the planning subject without upgrading its evidence.
    pub fn validate(&self) -> Result<(), OpportunityError> {
        check(&self.domain, "domain")?;
        check(&self.functional_role, "functional_role")?;
        check(&self.operating_context, "operating_context")?;
        check_set(&self.required_performance, "required_performance", false)?;
        check_set(&self.evidence_refs, "evidence_refs", false)?;
        Ok(())
    }

    /// Computes the deterministic identity of the exact function subject.
    pub fn id(&self) -> Result<OpportunityId, OpportunityError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(FUNCTION_DOMAIN);
        put_str(&mut h, self.domain.trim());
        put_str(&mut h, self.functional_role.trim());
        put_str(&mut h, self.operating_context.trim());
        put_set(&mut h, &self.required_performance);
        put_set(&mut h, &self.evidence_refs);
        Ok(finish(h))
    }
}

/// Hypothesis that a material, interface, or process is limiting a function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialBottleneckHypothesisV1 {
    function_id: OpportunityId,
    material_or_process_subject: String,
    limiting_proposition: String,
    evidence_state: BottleneckEvidenceState,
    alternative_explanations: Vec<String>,
    evidence_refs: Vec<String>,
}

impl MaterialBottleneckHypothesisV1 {
    /// Builds a validated bottleneck hypothesis.
    pub fn new(
        function_id: OpportunityId,
        material_or_process_subject: impl Into<String>,
        limiting_proposition: impl Into<String>,
        evidence_state: BottleneckEvidenceState,
        alternative_explanations: Vec<String>,
        evidence_refs: Vec<String>,
    ) -> Result<Self, OpportunityError> {
        let value = Self {
            function_id,
            material_or_process_subject: clean(
                material_or_process_subject.into(),
                "material_or_process_subject",
            )?,
            limiting_proposition: clean(limiting_proposition.into(), "limiting_proposition")?,
            evidence_state,
            alternative_explanations: clean_set(
                alternative_explanations,
                "alternative_explanations",
                true,
            )?,
            evidence_refs: clean_set(evidence_refs, "evidence_refs", false)?,
        };
        value.validate()?;
        Ok(value)
    }

    /// Validates the hypothesis without changing its evidence state.
    pub fn validate(&self) -> Result<(), OpportunityError> {
        check(&self.material_or_process_subject, "material_or_process_subject")?;
        check(&self.limiting_proposition, "limiting_proposition")?;
        check_set(&self.alternative_explanations, "alternative_explanations", true)?;
        check_set(&self.evidence_refs, "evidence_refs", false)?;
        Ok(())
    }

    /// Computes the deterministic identity of the exact bottleneck hypothesis.
    pub fn id(&self) -> Result<OpportunityId, OpportunityError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(BOTTLENECK_DOMAIN);
        h.update(self.function_id.as_bytes());
        put_str(&mut h, self.material_or_process_subject.trim());
        put_str(&mut h, self.limiting_proposition.trim());
        h.update(&[bottleneck_state_code(self.evidence_state)]);
        put_set(&mut h, &self.alternative_explanations);
        put_set(&mut h, &self.evidence_refs);
        Ok(finish(h))
    }
}

/// Non-authoritative intervention proposed for a material bottleneck.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchInterventionV1 {
    class: InterventionClass,
    description: String,
    evidence_refs: Vec<String>,
}

impl ResearchInterventionV1 {
    /// Builds a validated intervention proposal.
    pub fn new(
        class: InterventionClass,
        description: impl Into<String>,
        evidence_refs: Vec<String>,
    ) -> Result<Self, OpportunityError> {
        let value = Self {
            class,
            description: clean(description.into(), "description")?,
            evidence_refs: clean_set(evidence_refs, "evidence_refs", false)?,
        };
        value.validate()?;
        Ok(value)
    }

    /// Validates the intervention proposal.
    pub fn validate(&self) -> Result<(), OpportunityError> {
        check(&self.description, "description")?;
        check_set(&self.evidence_refs, "evidence_refs", false)?;
        Ok(())
    }

    /// Computes the deterministic identity of the intervention proposal.
    pub fn id(&self) -> Result<OpportunityId, OpportunityError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(INTERVENTION_DOMAIN);
        h.update(&[intervention_code(self.class)]);
        put_str(&mut h, self.description.trim());
        put_set(&mut h, &self.evidence_refs);
        Ok(finish(h))
    }
}

/// One independent axis in an opportunity assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpportunityDimensionV1 {
    axis: OpportunityAxis,
    value: AssessmentValue,
    evidence_refs: Vec<String>,
}

impl OpportunityDimensionV1 {
    /// Builds a validated planning dimension.
    pub fn new(
        axis: OpportunityAxis,
        value: AssessmentValue,
        evidence_refs: Vec<String>,
    ) -> Result<Self, OpportunityError> {
        let value = Self {
            axis,
            value: clean_assessment(axis, value)?,
            evidence_refs: clean_set(evidence_refs, "evidence_refs", false)?,
        };
        value.validate()?;
        Ok(value)
    }

    /// Returns this dimension's independent axis.
    pub fn axis(&self) -> OpportunityAxis {
        self.axis
    }

    /// Validates the value and evidence references.
    pub fn validate(&self) -> Result<(), OpportunityError> {
        check_assessment(self.axis, &self.value)?;
        check_set(&self.evidence_refs, "evidence_refs", false)?;
        Ok(())
    }
}

/// Multi-axis research opportunity with no built-in aggregate score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchOpportunityV1 {
    bottleneck_id: OpportunityId,
    interventions: Vec<ResearchInterventionV1>,
    dimensions: Vec<OpportunityDimensionV1>,
}

impl ResearchOpportunityV1 {
    /// Builds a validated multi-axis opportunity.
    pub fn new(
        bottleneck_id: OpportunityId,
        interventions: Vec<ResearchInterventionV1>,
        dimensions: Vec<OpportunityDimensionV1>,
    ) -> Result<Self, OpportunityError> {
        let value = Self {
            bottleneck_id,
            interventions,
            dimensions,
        };
        value.validate()?;
        Ok(value)
    }

    /// Validates nested contracts and prevents duplicate axes/interventions.
    pub fn validate(&self) -> Result<(), OpportunityError> {
        if self.interventions.is_empty() {
            return Err(OpportunityError::EmptyCollection("interventions"));
        }
        if self.dimensions.is_empty() {
            return Err(OpportunityError::EmptyCollection("dimensions"));
        }

        let mut intervention_ids = BTreeSet::new();
        for intervention in &self.interventions {
            let id = intervention.id()?;
            if !intervention_ids.insert(id) {
                return Err(OpportunityError::DuplicateIntervention);
            }
        }

        let mut axes = BTreeSet::new();
        for dimension in &self.dimensions {
            dimension.validate()?;
            if !axes.insert(dimension.axis()) {
                return Err(OpportunityError::DuplicateAxis(dimension.axis()));
            }
        }
        Ok(())
    }

    /// Computes an order-invariant identity over interventions and dimensions.
    pub fn id(&self) -> Result<OpportunityId, OpportunityError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(OPPORTUNITY_DOMAIN);
        h.update(self.bottleneck_id.as_bytes());

        let mut intervention_ids = self
            .interventions
            .iter()
            .map(ResearchInterventionV1::id)
            .collect::<Result<Vec<_>, _>>()?;
        intervention_ids.sort();
        put_u64(&mut h, intervention_ids.len() as u64);
        for id in intervention_ids {
            h.update(id.as_bytes());
        }

        let mut dimensions = self.dimensions.iter().collect::<Vec<_>>();
        dimensions.sort_by_key(|dimension| dimension.axis());
        put_u64(&mut h, dimensions.len() as u64);
        for dimension in dimensions {
            h.update(&[axis_code(dimension.axis)]);
            put_assessment(&mut h, dimension.axis, &dimension.value)?;
            put_set(&mut h, &dimension.evidence_refs);
        }
        Ok(finish(h))
    }

    /// Returns the independent planning dimensions without aggregating them.
    pub fn dimensions(&self) -> &[OpportunityDimensionV1] {
        &self.dimensions
    }

    /// Returns the intervention proposals for this bottleneck.
    pub fn interventions(&self) -> &[ResearchInterventionV1] {
        &self.interventions
    }
}

fn clean(value: String, field: &'static str) -> Result<String, OpportunityError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(OpportunityError::EmptyField(field));
    }
    Ok(value.to_owned())
}

fn check(value: &str, field: &'static str) -> Result<(), OpportunityError> {
    if value.trim().is_empty() {
        Err(OpportunityError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn clean_set(
    values: Vec<String>,
    field: &'static str,
    allow_empty: bool,
) -> Result<Vec<String>, OpportunityError> {
    if values.is_empty() && !allow_empty {
        return Err(OpportunityError::EmptyCollection(field));
    }
    let mut cleaned = values
        .into_iter()
        .map(|value| clean(value, field))
        .collect::<Result<Vec<_>, _>>()?;
    cleaned.sort();
    if cleaned.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(OpportunityError::DuplicateValue(field));
    }
    Ok(cleaned)
}

fn check_set(
    values: &[String],
    field: &'static str,
    allow_empty: bool,
) -> Result<(), OpportunityError> {
    if values.is_empty() && !allow_empty {
        return Err(OpportunityError::EmptyCollection(field));
    }
    let mut seen = BTreeSet::new();
    for value in values {
        check(value, field)?;
        if !seen.insert(value.trim()) {
            return Err(OpportunityError::DuplicateValue(field));
        }
    }
    Ok(())
}

fn clean_assessment(
    axis: OpportunityAxis,
    value: AssessmentValue,
) -> Result<AssessmentValue, OpportunityError> {
    let value = match value {
        AssessmentValue::Scalar(number, unit) => {
            let unit = unit.trim();
            if unit.is_empty() {
                return Err(OpportunityError::EmptyUnit(axis));
            }
            AssessmentValue::Scalar(number, unit.to_owned())
        }
        AssessmentValue::Interval(low, high, unit) => {
            let unit = unit.trim();
            if unit.is_empty() {
                return Err(OpportunityError::EmptyUnit(axis));
            }
            AssessmentValue::Interval(low, high, unit.to_owned())
        }
        other => other,
    };
    check_assessment(axis, &value)?;
    Ok(value)
}

fn check_assessment(axis: OpportunityAxis, value: &AssessmentValue) -> Result<(), OpportunityError> {
    match value {
        AssessmentValue::Unknown | AssessmentValue::Ordinal(_) => Ok(()),
        AssessmentValue::Scalar(number, unit) => {
            if !number.is_finite() {
                return Err(OpportunityError::NonFiniteValue(axis));
            }
            if unit.trim().is_empty() {
                return Err(OpportunityError::EmptyUnit(axis));
            }
            Ok(())
        }
        AssessmentValue::Interval(low, high, unit) => {
            if !low.is_finite() || !high.is_finite() {
                return Err(OpportunityError::NonFiniteValue(axis));
            }
            if low > high {
                return Err(OpportunityError::InvalidInterval(axis));
            }
            if unit.trim().is_empty() {
                return Err(OpportunityError::EmptyUnit(axis));
            }
            Ok(())
        }
    }
}

fn put_u64(h: &mut blake3::Hasher, value: u64) {
    h.update(&value.to_le_bytes());
}

fn put_str(h: &mut blake3::Hasher, value: &str) {
    put_u64(h, value.len() as u64);
    h.update(value.as_bytes());
}

fn put_set(h: &mut blake3::Hasher, values: &[String]) {
    let mut values = values.iter().map(|value| value.trim()).collect::<Vec<_>>();
    values.sort_unstable();
    put_u64(h, values.len() as u64);
    for value in values {
        put_str(h, value);
    }
}

fn put_assessment(
    h: &mut blake3::Hasher,
    axis: OpportunityAxis,
    value: &AssessmentValue,
) -> Result<(), OpportunityError> {
    check_assessment(axis, value)?;
    match value {
        AssessmentValue::Unknown => {
            h.update(&[0]);
        }
        AssessmentValue::Ordinal(value) => {
            h.update(&[1, ordinal_code(*value)]);
        }
        AssessmentValue::Scalar(number, unit) => {
            h.update(&[2]);
            h.update(&number.to_bits().to_le_bytes());
            put_str(h, unit.trim());
        }
        AssessmentValue::Interval(low, high, unit) => {
            h.update(&[3]);
            h.update(&low.to_bits().to_le_bytes());
            h.update(&high.to_bits().to_le_bytes());
            put_str(h, unit.trim());
        }
    }
    Ok(())
}

fn bottleneck_state_code(value: BottleneckEvidenceState) -> u8 {
    match value {
        BottleneckEvidenceState::Weak => 0,
        BottleneckEvidenceState::Candidate => 1,
        BottleneckEvidenceState::EstablishedUnderProfile => 2,
        BottleneckEvidenceState::EvidenceLimited => 3,
    }
}

fn intervention_code(value: InterventionClass) -> u8 {
    match value {
        InterventionClass::MaterialSubstitution => 0,
        InterventionClass::MaterialIntensityReduction => 1,
        InterventionClass::ArchitectureChange => 2,
        InterventionClass::InterfaceEngineering => 3,
        InterventionClass::ProcessImprovement => 4,
        InterventionClass::RecoveryOrRecycling => 5,
        InterventionClass::LifetimeExtension => 6,
        InterventionClass::RepairOrReuse => 7,
        InterventionClass::AlternativeTechnologyFunction => 8,
        InterventionClass::MeasurementOrModelImprovement => 9,
    }
}

fn axis_code(value: OpportunityAxis) -> u8 {
    match value {
        OpportunityAxis::SystemLeverage => 0,
        OpportunityAxis::PerformanceHeadroom => 1,
        OpportunityAxis::CriticalityLeverage => 2,
        OpportunityAxis::DownstreamBreadth => 3,
        OpportunityAxis::ScientificTractability => 4,
        OpportunityAxis::ComputationalObservability => 5,
        OpportunityAxis::DataAvailability => 6,
        OpportunityAxis::ExperimentalAccessibility => 7,
        OpportunityAxis::FalsificationCost => 8,
        OpportunityAxis::InfrastructureReadiness => 9,
        OpportunityAxis::EvidenceQuality => 10,
    }
}

fn ordinal_code(value: PlanningOrdinal) -> u8 {
    match value {
        PlanningOrdinal::VeryLow => 0,
        PlanningOrdinal::Low => 1,
        PlanningOrdinal::Medium => 2,
        PlanningOrdinal::High => 3,
        PlanningOrdinal::VeryHigh => 4,
    }
}

fn finish(h: blake3::Hasher) -> OpportunityId {
    OpportunityId(*h.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn function(refs: Vec<String>) -> TechnologyFunctionSubjectV1 {
        TechnologyFunctionSubjectV1::new(
            "electronics",
            "remove heat across chip/package interface",
            "high-power accelerator package",
            vec![
                "low total interface thermal resistance".into(),
                "survive thermal cycling".into(),
            ],
            refs,
        )
        .unwrap()
    }

    fn bottleneck(function_id: OpportunityId) -> MaterialBottleneckHypothesisV1 {
        MaterialBottleneckHypothesisV1::new(
            function_id,
            "thermal interface stack",
            "contact and bond-line resistance materially limit heat transfer",
            BottleneckEvidenceState::Candidate,
            vec!["cold-plate limitation".into(), "package-spreader limitation".into()],
            vec!["doi:example/tim-review".into()],
        )
        .unwrap()
    }

    #[test]
    fn set_order_does_not_change_function_identity() {
        let a = function(vec!["source:b".into(), "source:a".into()]);
        let b = function(vec!["source:a".into(), "source:b".into()]);
        assert_eq!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn operating_context_changes_function_identity() {
        let a = function(vec!["source:a".into()]);
        let b = TechnologyFunctionSubjectV1::new(
            "electronics",
            "remove heat across chip/package interface",
            "low-power sensor package",
            vec![
                "low total interface thermal resistance".into(),
                "survive thermal cycling".into(),
            ],
            vec!["source:a".into()],
        )
        .unwrap();
        assert_ne!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn alternative_explanation_changes_bottleneck_identity() {
        let function = function(vec!["source:a".into()]);
        let a = bottleneck(function.id().unwrap());
        let b = MaterialBottleneckHypothesisV1::new(
            function.id().unwrap(),
            "thermal interface stack",
            "contact and bond-line resistance materially limit heat transfer",
            BottleneckEvidenceState::Candidate,
            vec!["package-spreader limitation".into()],
            vec!["doi:example/tim-review".into()],
        )
        .unwrap();
        assert_ne!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn opportunity_identity_is_order_invariant() {
        let function = function(vec!["source:a".into()]);
        let bottleneck_id = bottleneck(function.id().unwrap()).id().unwrap();
        let substitute = ResearchInterventionV1::new(
            InterventionClass::MaterialSubstitution,
            "replace scarce filler family",
            vec!["source:sub".into()],
        )
        .unwrap();
        let interface = ResearchInterventionV1::new(
            InterventionClass::InterfaceEngineering,
            "reduce contact resistance",
            vec!["source:int".into()],
        )
        .unwrap();
        let leverage = OpportunityDimensionV1::new(
            OpportunityAxis::SystemLeverage,
            AssessmentValue::Ordinal(PlanningOrdinal::High),
            vec!["source:lev".into()],
        )
        .unwrap();
        let cost = OpportunityDimensionV1::new(
            OpportunityAxis::FalsificationCost,
            AssessmentValue::Interval(1.0, 3.0, "experiment-days".into()),
            vec!["source:cost".into()],
        )
        .unwrap();

        let a = ResearchOpportunityV1::new(
            bottleneck_id,
            vec![substitute.clone(), interface.clone()],
            vec![leverage.clone(), cost.clone()],
        )
        .unwrap();
        let b = ResearchOpportunityV1::new(
            bottleneck_id,
            vec![interface, substitute],
            vec![cost, leverage],
        )
        .unwrap();
        assert_eq!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn duplicate_axis_is_rejected() {
        let function = function(vec!["source:a".into()]);
        let bottleneck_id = bottleneck(function.id().unwrap()).id().unwrap();
        let intervention = ResearchInterventionV1::new(
            InterventionClass::ProcessImprovement,
            "improve bond-line process control",
            vec!["source:p".into()],
        )
        .unwrap();
        let first = OpportunityDimensionV1::new(
            OpportunityAxis::EvidenceQuality,
            AssessmentValue::Ordinal(PlanningOrdinal::Medium),
            vec!["source:e1".into()],
        )
        .unwrap();
        let second = OpportunityDimensionV1::new(
            OpportunityAxis::EvidenceQuality,
            AssessmentValue::Ordinal(PlanningOrdinal::High),
            vec!["source:e2".into()],
        )
        .unwrap();
        let result = ResearchOpportunityV1::new(
            bottleneck_id,
            vec![intervention],
            vec![first, second],
        );
        assert!(matches!(
            result,
            Err(OpportunityError::DuplicateAxis(OpportunityAxis::EvidenceQuality))
        ));
    }

    #[test]
    fn non_finite_assessment_is_rejected() {
        let result = OpportunityDimensionV1::new(
            OpportunityAxis::FalsificationCost,
            AssessmentValue::Scalar(f64::NAN, "usd".into()),
            vec!["source:cost".into()],
        );
        assert!(matches!(
            result,
            Err(OpportunityError::NonFiniteValue(OpportunityAxis::FalsificationCost))
        ));
    }

    #[test]
    fn unknown_is_distinct_from_numeric_zero() {
        let unknown = OpportunityDimensionV1::new(
            OpportunityAxis::DataAvailability,
            AssessmentValue::Unknown,
            vec!["source:data".into()],
        )
        .unwrap();
        let zero = OpportunityDimensionV1::new(
            OpportunityAxis::DataAvailability,
            AssessmentValue::Scalar(0.0, "records".into()),
            vec!["source:data".into()],
        )
        .unwrap();
        let function = function(vec!["source:a".into()]);
        let bottleneck_id = bottleneck(function.id().unwrap()).id().unwrap();
        let intervention = ResearchInterventionV1::new(
            InterventionClass::MeasurementOrModelImprovement,
            "measure missing baseline",
            vec!["source:m".into()],
        )
        .unwrap();
        let a = ResearchOpportunityV1::new(
            bottleneck_id,
            vec![intervention.clone()],
            vec![unknown],
        )
        .unwrap();
        let b = ResearchOpportunityV1::new(bottleneck_id, vec![intervention], vec![zero]).unwrap();
        assert_ne!(a.id().unwrap(), b.id().unwrap());
    }

    #[test]
    fn serde_round_trip_preserves_identity() {
        let function = function(vec!["source:a".into()]);
        let encoded = serde_json::to_string(&function).unwrap();
        let decoded: TechnologyFunctionSubjectV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(function.id().unwrap(), decoded.id().unwrap());
    }
}
