//! Canonical construction for V2 confirmatory protocol identities.
//!
//! The lower-level endpoint types preserve caller-provided ordering exactly. This module defines
//! the promotion-facing construction boundary for collections whose order is semantically a set.
//! Endpoint-local metric/baseline input order is intentionally preserved because an externally
//! frozen analysis rule may assign positional meaning to those inputs.

use serde::{Deserialize, Serialize};

use crate::{
    ConfirmatoryEndpointSpec, ConfirmatoryProtocolError, ConfirmatoryRunBinding, FrozenProtocol,
    FrozenConfirmatoryProtocol, MetricSchemaBinding, MetricValueSchema, ResearchRunRegistration,
};

pub type Result<T> = std::result::Result<T, ConfirmatoryProtocolError>;

/// Canonical, promotion-facing V2 confirmatory protocol identity.
///
/// Canonicalization normalizes only collections whose ordering has no declared semantics:
///
/// - metric schema bindings by `metric_id`;
/// - outer endpoint records by `endpoint_id`;
/// - categorical allowed-value vocabularies lexicographically.
///
/// It deliberately does not reorder `ConfirmatoryEndpointSpec::metric_ids` or `baseline_ids`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalConfirmatoryProtocol {
    inner: FrozenConfirmatoryProtocol,
}

impl CanonicalConfirmatoryProtocol {
    pub fn new(
        frozen_protocol: FrozenProtocol,
        mut metric_schemas: Vec<MetricSchemaBinding>,
        mut confirmatory_endpoints: Vec<ConfirmatoryEndpointSpec>,
    ) -> Result<Self> {
        // Validate before normalization so duplicate categorical vocabulary entries remain errors
        // rather than disappearing through accidental deduplication.
        for binding in &metric_schemas {
            binding.validate()?;
        }
        for endpoint in &confirmatory_endpoints {
            endpoint.validate_shallow()?;
        }

        for binding in &mut metric_schemas {
            if let MetricValueSchema::Categorical {
                allowed_values: Some(values),
            } = &mut binding.value_schema
            {
                values.sort();
            }
        }
        metric_schemas.sort_by(|left, right| left.metric_id.cmp(&right.metric_id));
        confirmatory_endpoints.sort_by(|left, right| left.endpoint_id.cmp(&right.endpoint_id));

        let inner = FrozenConfirmatoryProtocol::new(
            frozen_protocol,
            metric_schemas,
            confirmatory_endpoints,
        )?;
        let value = Self { inner };
        value.validate()?;
        Ok(value)
    }

    /// Validate both the lower-level V2 semantics/digest and the canonical representation.
    ///
    /// This check is intentionally required after deserialization as well as after construction.
    /// Constructor-time sorting alone would not prevent an imported noncanonical raw V2 envelope
    /// from being mislabeled as canonical.
    pub fn validate(&self) -> Result<()> {
        self.inner.validate()?;

        let schemas = self.inner.metric_schemas();
        if !schemas
            .windows(2)
            .all(|window| window[0].metric_id < window[1].metric_id)
        {
            return Err(ConfirmatoryProtocolError::InvalidMetricSchema(
                "canonical metric schema bindings must be strictly ordered by metric_id".into(),
            ));
        }
        for binding in schemas {
            if let MetricValueSchema::Categorical {
                allowed_values: Some(values),
            } = &binding.value_schema
            {
                if !values.windows(2).all(|window| window[0] < window[1]) {
                    return Err(ConfirmatoryProtocolError::InvalidMetricSchema(format!(
                        "categorical vocabulary for metric {} is not canonical",
                        binding.metric_id
                    )));
                }
            }
        }

        let endpoints = self.inner.confirmatory_endpoints();
        if !endpoints
            .windows(2)
            .all(|window| window[0].endpoint_id < window[1].endpoint_id)
        {
            return Err(ConfirmatoryProtocolError::InvalidEndpoint(
                "canonical confirmatory endpoints must be strictly ordered by endpoint_id".into(),
            ));
        }

        Ok(())
    }

    pub fn digest(&self) -> &str {
        self.inner.digest()
    }

    pub fn frozen_protocol(&self) -> &FrozenProtocol {
        self.inner.frozen_protocol()
    }

    pub fn metric_schemas(&self) -> &[MetricSchemaBinding] {
        self.inner.metric_schemas()
    }

    pub fn confirmatory_endpoints(&self) -> &[ConfirmatoryEndpointSpec] {
        self.inner.confirmatory_endpoints()
    }

    pub fn bind_run(&self, run: ResearchRunRegistration) -> Result<CanonicalConfirmatoryRunBinding> {
        self.validate()?;
        let binding = ConfirmatoryRunBinding::new(&self.inner, run)?;
        Ok(CanonicalConfirmatoryRunBinding { binding })
    }

    pub(crate) fn inner(&self) -> &FrozenConfirmatoryProtocol {
        &self.inner
    }
}

/// Run binding that can only be constructed from a canonical V2 protocol identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalConfirmatoryRunBinding {
    binding: ConfirmatoryRunBinding,
}

impl CanonicalConfirmatoryRunBinding {
    pub fn validate_against(&self, protocol: &CanonicalConfirmatoryProtocol) -> Result<()> {
        protocol.validate()?;
        self.binding.validate_against(protocol.inner())
    }

    pub fn digest(&self) -> &str {
        self.binding.digest()
    }

    pub fn confirmatory_protocol_digest(&self) -> &str {
        self.binding.confirmatory_protocol_digest()
    }

    pub fn run(&self) -> &ResearchRunRegistration {
        self.binding.run()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AnalysisPlanRef, BaselineSpec, ConfirmatoryDecisionRule, HypothesisDirection,
        HypothesisRole, HypothesisSpec, MetricRole, MetricSpec, MultiplicityPolicy,
        ResearchProtocol, StoppingRule,
    };

    fn frozen_two_metric_protocol() -> FrozenProtocol {
        ResearchProtocol::new(
            "canonical-v2",
            "1",
            "Are both frozen primary outcomes satisfied?",
            vec![
                HypothesisSpec::new(
                    "h-primary",
                    "both preregistered primary outcomes are evaluated",
                    HypothesisRole::Primary,
                    HypothesisDirection::Qualitative,
                )
                .unwrap(),
            ],
            vec![
                MetricSpec::new(
                    "metric-a",
                    "Boolean primary",
                    "bool",
                    MetricRole::Primary,
                    "exact",
                )
                .unwrap(),
                MetricSpec::new(
                    "metric-b",
                    "Categorical primary",
                    "category",
                    MetricRole::Primary,
                    "exact",
                )
                .unwrap(),
            ],
            vec![
                BaselineSpec::new("baseline", "frozen comparator", "bench/baseline-v1").unwrap(),
            ],
            vec![],
            StoppingRule::FixedEpisodeCount(8),
            MultiplicityPolicy::SeparateConfirmatoryFromExploratory,
            AnalysisPlanRef::new("analysis", "1", "sha256:analysis-plan").unwrap(),
            "frozen scenario set",
            "frozen seeds",
        )
        .unwrap()
        .freeze(1_000)
        .unwrap()
    }

    fn schemas_reversed(vocabulary_reversed: bool) -> Vec<MetricSchemaBinding> {
        let vocabulary = if vocabulary_reversed {
            vec!["Suspended".into(), "Current".into()]
        } else {
            vec!["Current".into(), "Suspended".into()]
        };
        vec![
            MetricSchemaBinding::new(
                "metric-b",
                MetricValueSchema::categorical(Some(vocabulary)).unwrap(),
            )
            .unwrap(),
            MetricSchemaBinding::new("metric-a", MetricValueSchema::Boolean).unwrap(),
        ]
    }

    fn endpoints_reversed() -> Vec<ConfirmatoryEndpointSpec> {
        vec![
            ConfirmatoryEndpointSpec::new(
                "endpoint-b",
                "h-primary",
                vec!["metric-b".into()],
                vec![],
                ConfirmatoryDecisionRule::CategoricalMustEqual("Current".into()),
            )
            .unwrap(),
            ConfirmatoryEndpointSpec::new(
                "endpoint-a",
                "h-primary",
                vec!["metric-a".into()],
                vec![],
                ConfirmatoryDecisionRule::BooleanMustBe(true),
            )
            .unwrap(),
        ]
    }

    #[test]
    fn outer_order_and_vocabulary_order_do_not_change_canonical_identity() {
        let frozen = frozen_two_metric_protocol();
        let first = CanonicalConfirmatoryProtocol::new(
            frozen.clone(),
            schemas_reversed(true),
            endpoints_reversed(),
        )
        .unwrap();

        let mut second_schemas = schemas_reversed(false);
        second_schemas.reverse();
        let mut second_endpoints = endpoints_reversed();
        second_endpoints.reverse();
        let second = CanonicalConfirmatoryProtocol::new(
            frozen,
            second_schemas,
            second_endpoints,
        )
        .unwrap();

        assert_eq!(first.digest(), second.digest());
        assert_eq!(first.metric_schemas()[0].metric_id, "metric-a");
        assert_eq!(first.confirmatory_endpoints()[0].endpoint_id, "endpoint-a");
        match &first.metric_schemas()[1].value_schema {
            MetricValueSchema::Categorical {
                allowed_values: Some(values),
            } => assert_eq!(values, &vec!["Current".to_string(), "Suspended".to_string()]),
            _ => panic!("metric-b should remain categorical"),
        }
    }

    #[test]
    fn duplicate_categorical_values_are_rejected_not_deduplicated() {
        let error = MetricValueSchema::categorical(Some(vec!["Current".into(), "Current".into()]))
            .unwrap_err();
        assert!(matches!(error, ConfirmatoryProtocolError::DuplicateId(_)));
    }

    #[test]
    fn imported_noncanonical_raw_envelope_is_rejected_by_canonical_validation() {
        let raw = FrozenConfirmatoryProtocol::new(
            frozen_two_metric_protocol(),
            schemas_reversed(true),
            endpoints_reversed(),
        )
        .unwrap();
        raw.validate().unwrap();

        let imported = CanonicalConfirmatoryProtocol { inner: raw };
        assert!(matches!(
            imported.validate().unwrap_err(),
            ConfirmatoryProtocolError::InvalidMetricSchema(_)
                | ConfirmatoryProtocolError::InvalidEndpoint(_)
        ));
    }

    #[test]
    fn endpoint_input_order_remains_identity_bearing_for_external_rules() {
        let frozen = frozen_two_metric_protocol();
        let schemas = schemas_reversed(false);
        let first = CanonicalConfirmatoryProtocol::new(
            frozen.clone(),
            schemas.clone(),
            vec![
                ConfirmatoryEndpointSpec::new(
                    "ordered-endpoint",
                    "h-primary",
                    vec!["metric-a".into(), "metric-b".into()],
                    vec!["baseline".into()],
                    ConfirmatoryDecisionRule::external("ordered-rule", "sha256:ordered-rule")
                        .unwrap(),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let second = CanonicalConfirmatoryProtocol::new(
            frozen,
            schemas,
            vec![
                ConfirmatoryEndpointSpec::new(
                    "ordered-endpoint",
                    "h-primary",
                    vec!["metric-b".into(), "metric-a".into()],
                    vec!["baseline".into()],
                    ConfirmatoryDecisionRule::external("ordered-rule", "sha256:ordered-rule")
                        .unwrap(),
                )
                .unwrap(),
            ],
        )
        .unwrap();

        assert_ne!(first.digest(), second.digest());
    }

    #[test]
    fn canonical_run_binding_commits_canonical_protocol_identity() {
        let protocol = CanonicalConfirmatoryProtocol::new(
            frozen_two_metric_protocol(),
            schemas_reversed(false),
            endpoints_reversed(),
        )
        .unwrap();
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
        let binding = protocol.bind_run(run).unwrap();
        binding.validate_against(&protocol).unwrap();
        assert_eq!(binding.confirmatory_protocol_digest(), protocol.digest());
    }
}
