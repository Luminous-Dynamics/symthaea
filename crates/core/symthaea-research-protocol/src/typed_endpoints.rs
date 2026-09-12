//! Typed confirmatory endpoint contracts layered over a validated frozen research protocol.
//!
//! V1 protocol identity remains immutable. This module adds a V2 confirmatory envelope whose
//! digest commits metric value schemas, hypothesis-to-metric/baseline mappings, decision rules,
//! and exact run binding without rewriting the historical V1 protocol object.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::{
    FrozenProtocol, HypothesisRole, MetricRole, ProtocolError, ResearchRunRegistration,
};

const CONFIRMATORY_PROTOCOL_SCHEMA: &str = "symthaea-research-confirmatory-protocol/v2";
const CONFIRMATORY_RUN_SCHEMA: &str = "symthaea-research-confirmatory-run/v2";

pub type Result<T> = std::result::Result<T, ConfirmatoryProtocolError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfirmatoryProtocolError {
    Protocol(ProtocolError),
    EmptyField(&'static str),
    DuplicateId(String),
    MissingMetricSchema(String),
    UnknownMetricSchema(String),
    MissingPrimaryEndpoint(String),
    UnboundPrimaryMetric(String),
    UnknownEndpointHypothesis(String),
    UnknownEndpointMetric(String),
    UnknownEndpointBaseline(String),
    InvalidMetricSchema(String),
    InvalidEndpoint(String),
    Serialization(String),
    ProtocolDigestMismatch,
    RunBindingDigestMismatch,
}

impl Display for ConfirmatoryProtocolError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Protocol(error) => write!(f, "base research protocol invalid: {error}"),
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::DuplicateId(id) => write!(f, "duplicate confirmatory-contract id: {id}"),
            Self::MissingMetricSchema(id) => {
                write!(f, "metric {id} requires a frozen value schema")
            }
            Self::UnknownMetricSchema(id) => {
                write!(f, "value schema references unknown metric {id}")
            }
            Self::MissingPrimaryEndpoint(id) => {
                write!(f, "primary hypothesis {id} requires a confirmatory endpoint")
            }
            Self::UnboundPrimaryMetric(id) => write!(
                f,
                "primary metric {id} must belong to at least one primary confirmatory endpoint"
            ),
            Self::UnknownEndpointHypothesis(id) => {
                write!(f, "confirmatory endpoint references unknown hypothesis {id}")
            }
            Self::UnknownEndpointMetric(id) => {
                write!(f, "confirmatory endpoint references unknown metric {id}")
            }
            Self::UnknownEndpointBaseline(id) => {
                write!(f, "confirmatory endpoint references unknown baseline {id}")
            }
            Self::InvalidMetricSchema(message) => write!(f, "invalid metric schema: {message}"),
            Self::InvalidEndpoint(message) => write!(f, "invalid confirmatory endpoint: {message}"),
            Self::Serialization(message) => {
                write!(f, "confirmatory protocol serialization failed: {message}")
            }
            Self::ProtocolDigestMismatch => {
                write!(f, "confirmatory protocol digest does not match exact envelope contents")
            }
            Self::RunBindingDigestMismatch => {
                write!(f, "confirmatory run-binding digest does not match exact contents")
            }
        }
    }
}

impl Error for ConfirmatoryProtocolError {}

impl From<ProtocolError> for ConfirmatoryProtocolError {
    fn from(value: ProtocolError) -> Self {
        Self::Protocol(value)
    }
}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(ConfirmatoryProtocolError::EmptyField(field));
    }
    Ok(())
}

fn unique_ids<'a>(ids: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(ConfirmatoryProtocolError::DuplicateId(id.to_string()));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricValueSchema {
    Numeric { unit: String },
    Boolean,
    Categorical { allowed_values: Option<Vec<String>> },
}

impl MetricValueSchema {
    pub fn numeric(unit: impl Into<String>) -> Result<Self> {
        let value = Self::Numeric { unit: unit.into() };
        value.validate()?;
        Ok(value)
    }

    pub fn categorical(allowed_values: Option<Vec<String>>) -> Result<Self> {
        let value = Self::Categorical { allowed_values };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Numeric { unit } => non_empty(unit, "numeric metric unit"),
            Self::Boolean => Ok(()),
            Self::Categorical { allowed_values } => {
                if let Some(values) = allowed_values {
                    if values.is_empty() {
                        return Err(ConfirmatoryProtocolError::InvalidMetricSchema(
                            "categorical vocabulary must not be empty when declared".into(),
                        ));
                    }
                    for value in values {
                        non_empty(value, "categorical allowed value")?;
                    }
                    unique_ids(values.iter().map(String::as_str))?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricSchemaBinding {
    pub metric_id: String,
    pub value_schema: MetricValueSchema,
}

impl MetricSchemaBinding {
    pub fn new(metric_id: impl Into<String>, value_schema: MetricValueSchema) -> Result<Self> {
        let value = Self {
            metric_id: metric_id.into(),
            value_schema,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        non_empty(&self.metric_id, "metric schema binding id")?;
        self.value_schema.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryDecisionRule {
    BooleanMustBe(bool),
    CategoricalMustEqual(String),
    ExternalFrozenAnalysisRule {
        rule_id: String,
        artifact_digest: String,
    },
}

impl ConfirmatoryDecisionRule {
    pub fn external(
        rule_id: impl Into<String>,
        artifact_digest: impl Into<String>,
    ) -> Result<Self> {
        let value = Self::ExternalFrozenAnalysisRule {
            rule_id: rule_id.into(),
            artifact_digest: artifact_digest.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::BooleanMustBe(_) => Ok(()),
            Self::CategoricalMustEqual(value) => {
                non_empty(value, "categorical endpoint target")
            }
            Self::ExternalFrozenAnalysisRule {
                rule_id,
                artifact_digest,
            } => {
                non_empty(rule_id, "external analysis rule id")?;
                non_empty(artifact_digest, "external analysis rule digest")?;
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryEndpointSpec {
    pub endpoint_id: String,
    pub hypothesis_id: String,
    pub metric_ids: Vec<String>,
    pub baseline_ids: Vec<String>,
    pub decision_rule: ConfirmatoryDecisionRule,
}

impl ConfirmatoryEndpointSpec {
    pub fn new(
        endpoint_id: impl Into<String>,
        hypothesis_id: impl Into<String>,
        metric_ids: Vec<String>,
        baseline_ids: Vec<String>,
        decision_rule: ConfirmatoryDecisionRule,
    ) -> Result<Self> {
        let value = Self {
            endpoint_id: endpoint_id.into(),
            hypothesis_id: hypothesis_id.into(),
            metric_ids,
            baseline_ids,
            decision_rule,
        };
        value.validate_shallow()?;
        Ok(value)
    }

    pub fn validate_shallow(&self) -> Result<()> {
        non_empty(&self.endpoint_id, "confirmatory endpoint id")?;
        non_empty(&self.hypothesis_id, "confirmatory endpoint hypothesis id")?;
        if self.metric_ids.is_empty() {
            return Err(ConfirmatoryProtocolError::InvalidEndpoint(format!(
                "endpoint {} requires at least one metric",
                self.endpoint_id
            )));
        }
        for metric_id in &self.metric_ids {
            non_empty(metric_id, "confirmatory endpoint metric id")?;
        }
        for baseline_id in &self.baseline_ids {
            non_empty(baseline_id, "confirmatory endpoint baseline id")?;
        }
        unique_ids(self.metric_ids.iter().map(String::as_str))?;
        unique_ids(self.baseline_ids.iter().map(String::as_str))?;
        self.decision_rule.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenConfirmatoryProtocol {
    frozen_protocol: FrozenProtocol,
    metric_schemas: Vec<MetricSchemaBinding>,
    confirmatory_endpoints: Vec<ConfirmatoryEndpointSpec>,
    digest: String,
}

impl FrozenConfirmatoryProtocol {
    pub fn new(
        frozen_protocol: FrozenProtocol,
        metric_schemas: Vec<MetricSchemaBinding>,
        confirmatory_endpoints: Vec<ConfirmatoryEndpointSpec>,
    ) -> Result<Self> {
        let mut value = Self {
            frozen_protocol,
            metric_schemas,
            confirmatory_endpoints,
            digest: String::new(),
        };
        value.validate_semantics()?;
        value.digest = value.compute_digest()?;
        Ok(value)
    }

    pub fn frozen_protocol(&self) -> &FrozenProtocol {
        &self.frozen_protocol
    }

    pub fn metric_schemas(&self) -> &[MetricSchemaBinding] {
        &self.metric_schemas
    }

    pub fn confirmatory_endpoints(&self) -> &[ConfirmatoryEndpointSpec] {
        &self.confirmatory_endpoints
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn verify_digest(&self) -> Result<()> {
        if self.compute_digest()? != self.digest {
            return Err(ConfirmatoryProtocolError::ProtocolDigestMismatch);
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        self.verify_digest()?;
        self.validate_semantics()
    }

    fn compute_digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&(
            CONFIRMATORY_PROTOCOL_SCHEMA,
            self.frozen_protocol.digest(),
            &self.metric_schemas,
            &self.confirmatory_endpoints,
        ))
        .map_err(|error| ConfirmatoryProtocolError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    fn validate_semantics(&self) -> Result<()> {
        self.frozen_protocol.validate()?;
        for binding in &self.metric_schemas {
            binding.validate()?;
        }
        for endpoint in &self.confirmatory_endpoints {
            endpoint.validate_shallow()?;
        }
        unique_ids(self.metric_schemas.iter().map(|value| value.metric_id.as_str()))?;
        unique_ids(
            self.confirmatory_endpoints
                .iter()
                .map(|value| value.endpoint_id.as_str()),
        )?;

        let protocol = self.frozen_protocol.protocol();
        let hypotheses: HashMap<&str, _> = protocol
            .hypotheses
            .iter()
            .map(|value| (value.id.as_str(), value))
            .collect();
        let metrics: HashMap<&str, _> = protocol
            .metrics
            .iter()
            .map(|value| (value.id.as_str(), value))
            .collect();
        let baselines: HashSet<&str> = protocol
            .baselines
            .iter()
            .map(|value| value.id.as_str())
            .collect();
        let schemas: HashMap<&str, _> = self
            .metric_schemas
            .iter()
            .map(|value| (value.metric_id.as_str(), &value.value_schema))
            .collect();

        for binding in &self.metric_schemas {
            let Some(metric) = metrics.get(binding.metric_id.as_str()) else {
                return Err(ConfirmatoryProtocolError::UnknownMetricSchema(
                    binding.metric_id.clone(),
                ));
            };
            if let MetricValueSchema::Numeric { unit } = &binding.value_schema {
                if &metric.unit != unit {
                    return Err(ConfirmatoryProtocolError::InvalidMetricSchema(format!(
                        "metric {} numeric unit mismatch: protocol={}, schema={}",
                        binding.metric_id, metric.unit, unit
                    )));
                }
            }
        }
        for metric in &protocol.metrics {
            if !schemas.contains_key(metric.id.as_str()) {
                return Err(ConfirmatoryProtocolError::MissingMetricSchema(
                    metric.id.clone(),
                ));
            }
        }

        for endpoint in &self.confirmatory_endpoints {
            let Some(hypothesis) = hypotheses.get(endpoint.hypothesis_id.as_str()) else {
                return Err(ConfirmatoryProtocolError::UnknownEndpointHypothesis(
                    endpoint.hypothesis_id.clone(),
                ));
            };
            if hypothesis.role == HypothesisRole::Exploratory {
                return Err(ConfirmatoryProtocolError::InvalidEndpoint(format!(
                    "endpoint {} cannot target exploratory hypothesis {}",
                    endpoint.endpoint_id, endpoint.hypothesis_id
                )));
            }
            for metric_id in &endpoint.metric_ids {
                if !metrics.contains_key(metric_id.as_str()) {
                    return Err(ConfirmatoryProtocolError::UnknownEndpointMetric(
                        metric_id.clone(),
                    ));
                }
            }
            for baseline_id in &endpoint.baseline_ids {
                if !baselines.contains(baseline_id.as_str()) {
                    return Err(ConfirmatoryProtocolError::UnknownEndpointBaseline(
                        baseline_id.clone(),
                    ));
                }
            }

            match &endpoint.decision_rule {
                ConfirmatoryDecisionRule::BooleanMustBe(_) => {
                    self.validate_direct_endpoint_shape(endpoint, "BooleanMustBe")?;
                    if schemas[endpoint.metric_ids[0].as_str()] != &MetricValueSchema::Boolean {
                        return Err(ConfirmatoryProtocolError::InvalidEndpoint(format!(
                            "endpoint {} BooleanMustBe requires a Boolean metric",
                            endpoint.endpoint_id
                        )));
                    }
                }
                ConfirmatoryDecisionRule::CategoricalMustEqual(target) => {
                    self.validate_direct_endpoint_shape(endpoint, "CategoricalMustEqual")?;
                    match schemas[endpoint.metric_ids[0].as_str()] {
                        MetricValueSchema::Categorical { allowed_values } => {
                            if let Some(values) = allowed_values {
                                if !values.contains(target) {
                                    return Err(ConfirmatoryProtocolError::InvalidEndpoint(
                                        format!(
                                            "endpoint {} categorical target is outside the frozen vocabulary",
                                            endpoint.endpoint_id
                                        ),
                                    ));
                                }
                            }
                        }
                        _ => {
                            return Err(ConfirmatoryProtocolError::InvalidEndpoint(format!(
                                "endpoint {} CategoricalMustEqual requires a Categorical metric",
                                endpoint.endpoint_id
                            )));
                        }
                    }
                }
                ConfirmatoryDecisionRule::ExternalFrozenAnalysisRule { .. } => {}
            }
        }

        for hypothesis in protocol
            .hypotheses
            .iter()
            .filter(|value| value.role == HypothesisRole::Primary)
        {
            if !self
                .confirmatory_endpoints
                .iter()
                .any(|endpoint| endpoint.hypothesis_id == hypothesis.id)
            {
                return Err(ConfirmatoryProtocolError::MissingPrimaryEndpoint(
                    hypothesis.id.clone(),
                ));
            }
        }

        for metric in protocol
            .metrics
            .iter()
            .filter(|value| value.role == MetricRole::Primary)
        {
            let bound_to_primary = self.confirmatory_endpoints.iter().any(|endpoint| {
                endpoint.metric_ids.iter().any(|id| id == &metric.id)
                    && hypotheses
                        .get(endpoint.hypothesis_id.as_str())
                        .is_some_and(|hypothesis| hypothesis.role == HypothesisRole::Primary)
            });
            if !bound_to_primary {
                return Err(ConfirmatoryProtocolError::UnboundPrimaryMetric(
                    metric.id.clone(),
                ));
            }
        }

        Ok(())
    }

    fn validate_direct_endpoint_shape(
        &self,
        endpoint: &ConfirmatoryEndpointSpec,
        rule_name: &str,
    ) -> Result<()> {
        if endpoint.metric_ids.len() != 1 || !endpoint.baseline_ids.is_empty() {
            return Err(ConfirmatoryProtocolError::InvalidEndpoint(format!(
                "endpoint {} {rule_name} requires exactly one metric and no baseline comparator; comparator analyses must use ExternalFrozenAnalysisRule",
                endpoint.endpoint_id
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryRunBinding {
    run: ResearchRunRegistration,
    confirmatory_protocol_digest: String,
    digest: String,
}

impl ConfirmatoryRunBinding {
    pub fn new(protocol: &FrozenConfirmatoryProtocol, run: ResearchRunRegistration) -> Result<Self> {
        protocol.validate()?;
        run.validate_against(protocol.frozen_protocol())?;
        let mut value = Self {
            run,
            confirmatory_protocol_digest: protocol.digest().to_string(),
            digest: String::new(),
        };
        value.digest = value.compute_digest()?;
        Ok(value)
    }

    pub fn run(&self) -> &ResearchRunRegistration {
        &self.run
    }

    pub fn confirmatory_protocol_digest(&self) -> &str {
        &self.confirmatory_protocol_digest
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn validate_against(&self, protocol: &FrozenConfirmatoryProtocol) -> Result<()> {
        protocol.validate()?;
        if self.confirmatory_protocol_digest != protocol.digest() {
            return Err(ConfirmatoryProtocolError::ProtocolDigestMismatch);
        }
        self.run.validate_against(protocol.frozen_protocol())?;
        if self.compute_digest()? != self.digest {
            return Err(ConfirmatoryProtocolError::RunBindingDigestMismatch);
        }
        Ok(())
    }

    fn compute_digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(&(
            CONFIRMATORY_RUN_SCHEMA,
            &self.confirmatory_protocol_digest,
            &self.run,
        ))
        .map_err(|error| ConfirmatoryProtocolError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AnalysisPlanRef, BaselineSpec, HypothesisDirection, HypothesisSpec, MetricSpec,
        MultiplicityPolicy, ResearchProtocol, StoppingRule,
    };

    fn base_protocol(extra_primary_metric: bool) -> ResearchProtocol {
        let mut metrics = vec![
            MetricSpec::new(
                "utility-per-byte",
                "Mission-relevant information per transmitted byte",
                "utility/byte",
                MetricRole::Primary,
                "held-out mean",
            )
            .unwrap(),
        ];
        if extra_primary_metric {
            metrics.push(
                MetricSpec::new(
                    "second-primary",
                    "Second primary outcome",
                    "bool",
                    MetricRole::Primary,
                    "exact run outcome",
                )
                .unwrap(),
            );
        }
        ResearchProtocol::new(
            "wetland-watch-v1",
            "1",
            "Does semantic prioritization preserve more mission-relevant information?",
            vec![
                HypothesisSpec::new(
                    "h-primary",
                    "semantic prioritization beats the simple ROI baseline",
                    HypothesisRole::Primary,
                    HypothesisDirection::GreaterThan,
                )
                .unwrap(),
            ],
            metrics,
            vec![
                BaselineSpec::new(
                    "simple-roi",
                    "conventional codec plus simple cloud/change ROI",
                    "bench/simple_roi_v1",
                )
                .unwrap(),
            ],
            vec![],
            StoppingRule::FixedSampleCount(100),
            MultiplicityPolicy::SeparateConfirmatoryFromExploratory,
            AnalysisPlanRef::new("analysis", "1", "sha256:analysis-plan").unwrap(),
            "100 frozen paired Sentinel scenes",
            "fixed seed manifest",
        )
        .unwrap()
    }

    fn frozen_base() -> FrozenProtocol {
        base_protocol(false).freeze(1_000).unwrap()
    }

    fn contract() -> FrozenConfirmatoryProtocol {
        FrozenConfirmatoryProtocol::new(
            frozen_base(),
            vec![
                MetricSchemaBinding::new(
                    "utility-per-byte",
                    MetricValueSchema::numeric("utility/byte").unwrap(),
                )
                .unwrap(),
            ],
            vec![
                ConfirmatoryEndpointSpec::new(
                    "h-primary-endpoint",
                    "h-primary",
                    vec!["utility-per-byte".into()],
                    vec!["simple-roi".into()],
                    ConfirmatoryDecisionRule::external(
                        "analysis/h-primary",
                        "sha256:h-primary-rule",
                    )
                    .unwrap(),
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn envelope_is_content_addressed_and_semantically_valid() {
        let contract = contract();
        contract.validate().unwrap();
        assert!(!contract.digest().is_empty());
    }

    #[test]
    fn every_metric_requires_a_value_schema() {
        let error = FrozenConfirmatoryProtocol::new(
            frozen_base(),
            vec![],
            vec![
                ConfirmatoryEndpointSpec::new(
                    "h-primary-endpoint",
                    "h-primary",
                    vec!["utility-per-byte".into()],
                    vec!["simple-roi".into()],
                    ConfirmatoryDecisionRule::external(
                        "analysis/h-primary",
                        "sha256:h-primary-rule",
                    )
                    .unwrap(),
                )
                .unwrap(),
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ConfirmatoryProtocolError::MissingMetricSchema("utility-per-byte".into())
        );
    }

    #[test]
    fn every_primary_hypothesis_requires_an_endpoint() {
        let error = FrozenConfirmatoryProtocol::new(
            frozen_base(),
            vec![
                MetricSchemaBinding::new(
                    "utility-per-byte",
                    MetricValueSchema::numeric("utility/byte").unwrap(),
                )
                .unwrap(),
            ],
            vec![],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ConfirmatoryProtocolError::MissingPrimaryEndpoint("h-primary".into())
        );
    }

    #[test]
    fn every_primary_metric_must_be_bound_to_primary_endpoint() {
        let frozen = base_protocol(true).freeze(1_000).unwrap();
        let error = FrozenConfirmatoryProtocol::new(
            frozen,
            vec![
                MetricSchemaBinding::new(
                    "utility-per-byte",
                    MetricValueSchema::numeric("utility/byte").unwrap(),
                )
                .unwrap(),
                MetricSchemaBinding::new("second-primary", MetricValueSchema::Boolean).unwrap(),
            ],
            vec![
                ConfirmatoryEndpointSpec::new(
                    "h-primary-endpoint",
                    "h-primary",
                    vec!["utility-per-byte".into()],
                    vec!["simple-roi".into()],
                    ConfirmatoryDecisionRule::external(
                        "analysis/h-primary",
                        "sha256:h-primary-rule",
                    )
                    .unwrap(),
                )
                .unwrap(),
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ConfirmatoryProtocolError::UnboundPrimaryMetric("second-primary".into())
        );
    }

    #[test]
    fn direct_boolean_rule_cannot_launder_numeric_metric() {
        let error = FrozenConfirmatoryProtocol::new(
            frozen_base(),
            vec![
                MetricSchemaBinding::new(
                    "utility-per-byte",
                    MetricValueSchema::numeric("utility/byte").unwrap(),
                )
                .unwrap(),
            ],
            vec![
                ConfirmatoryEndpointSpec::new(
                    "h-primary-endpoint",
                    "h-primary",
                    vec!["utility-per-byte".into()],
                    vec![],
                    ConfirmatoryDecisionRule::BooleanMustBe(true),
                )
                .unwrap(),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ConfirmatoryProtocolError::InvalidEndpoint(_)
        ));
    }

    #[test]
    fn direct_rule_cannot_claim_unrepresented_baseline_comparison() {
        let frozen = base_protocol(false).freeze(1_000).unwrap();
        let error = FrozenConfirmatoryProtocol::new(
            frozen,
            vec![
                MetricSchemaBinding::new("utility-per-byte", MetricValueSchema::Boolean).unwrap(),
            ],
            vec![
                ConfirmatoryEndpointSpec::new(
                    "h-primary-endpoint",
                    "h-primary",
                    vec!["utility-per-byte".into()],
                    vec!["simple-roi".into()],
                    ConfirmatoryDecisionRule::BooleanMustBe(false),
                )
                .unwrap(),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ConfirmatoryProtocolError::InvalidEndpoint(_)
        ));
    }

    #[test]
    fn categorical_target_must_be_inside_frozen_vocabulary() {
        let error = FrozenConfirmatoryProtocol::new(
            frozen_base(),
            vec![
                MetricSchemaBinding::new(
                    "utility-per-byte",
                    MetricValueSchema::categorical(Some(vec![
                        "Suspended".into(),
                        "Current".into(),
                    ]))
                    .unwrap(),
                )
                .unwrap(),
            ],
            vec![
                ConfirmatoryEndpointSpec::new(
                    "h-primary-endpoint",
                    "h-primary",
                    vec!["utility-per-byte".into()],
                    vec![],
                    ConfirmatoryDecisionRule::CategoricalMustEqual("Unknown".into()),
                )
                .unwrap(),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ConfirmatoryProtocolError::InvalidEndpoint(_)
        ));
    }

    #[test]
    fn endpoint_change_changes_confirmatory_protocol_identity() {
        let first = contract();
        let mut endpoints = first.confirmatory_endpoints().to_vec();
        endpoints[0].decision_rule = ConfirmatoryDecisionRule::external(
            "analysis/h-primary",
            "sha256:different-rule",
        )
        .unwrap();
        let second = FrozenConfirmatoryProtocol::new(
            first.frozen_protocol().clone(),
            first.metric_schemas().to_vec(),
            endpoints,
        )
        .unwrap();
        assert_ne!(first.digest(), second.digest());
    }

    #[test]
    fn run_binding_commits_confirmatory_protocol_identity() {
        let protocol = contract();
        let run = ResearchRunRegistration::new(
            protocol.frozen_protocol(),
            "run-001",
            1_100,
            "deadbeef",
            "sha256:dataset",
            "sha256:repro",
            "sha256:seeds",
        )
        .unwrap();
        let binding = ConfirmatoryRunBinding::new(&protocol, run).unwrap();
        binding.validate_against(&protocol).unwrap();
        assert_eq!(binding.confirmatory_protocol_digest(), protocol.digest());
        assert!(!binding.digest().is_empty());
    }
}
