// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification-only operationalization theorem for Economic Science.
//!
//! ETIR variables identify theoretical quantities. They intentionally do not
//! assert that a concrete external series measures that quantity. This test
//! freezes the higher-layer contract required before observations may stand in
//! for an ETIR variable.

use symthaea_economics::{EconomicVariable, StateDomain, UnitId, VariableId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MeasurementClass {
    Direct,
    Derived,
    Proxy,
    LatentEstimate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TemporalSemantics {
    PeriodAverage,
    PointInTime,
    FlowOverPeriod,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MeasurementAdmission {
    DirectObservation,
    DerivedObservation { method_artifact_id: String },
    ProxyObservation { basis: String },
    LatentEstimate { estimator_artifact_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MeasurementError {
    EmptyText(&'static str),
    TargetUnitMismatch,
    InvalidMethodContract,
    SourceNamespaceMismatch,
    SeriesMismatch,
    ConstructMismatch,
    SourceUnitMismatch,
    OutputUnitMismatch,
    PopulationScopeMismatch,
    TemporalSemanticsMismatch,
    MethodArtifactMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MeasurementSpecification {
    specification_id: String,
    variable_id: VariableId,
    construct_id: String,
    source_namespace: String,
    source_series_id: String,
    source_unit: UnitId,
    output_unit: UnitId,
    population_scope: String,
    temporal_semantics: TemporalSemantics,
    class: MeasurementClass,
    method_artifact_id: Option<String>,
    proxy_basis: Option<String>,
}

impl MeasurementSpecification {
    #[allow(clippy::too_many_arguments)]
    fn new(
        specification_id: impl Into<String>,
        variable: &EconomicVariable,
        construct_id: impl Into<String>,
        source_namespace: impl Into<String>,
        source_series_id: impl Into<String>,
        source_unit: UnitId,
        output_unit: UnitId,
        population_scope: impl Into<String>,
        temporal_semantics: TemporalSemantics,
        class: MeasurementClass,
        method_artifact_id: Option<String>,
        proxy_basis: Option<String>,
    ) -> Result<Self, MeasurementError> {
        let specification_id = specification_id.into();
        let construct_id = construct_id.into();
        let source_namespace = source_namespace.into();
        let source_series_id = source_series_id.into();
        let population_scope = population_scope.into();

        for (field, value) in [
            ("measurement specification id", specification_id.as_str()),
            ("measurement construct id", construct_id.as_str()),
            ("measurement source namespace", source_namespace.as_str()),
            ("measurement source series id", source_series_id.as_str()),
            ("measurement population scope", population_scope.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(MeasurementError::EmptyText(field));
            }
        }

        if &output_unit != variable.unit() {
            return Err(MeasurementError::TargetUnitMismatch);
        }

        let nonempty = |value: &Option<String>| {
            value
                .as_ref()
                .is_some_and(|item| !item.trim().is_empty())
        };

        match class {
            MeasurementClass::Direct => {
                if source_unit != output_unit
                    || method_artifact_id.is_some()
                    || proxy_basis.is_some()
                {
                    return Err(MeasurementError::InvalidMethodContract);
                }
            }
            MeasurementClass::Derived => {
                if !nonempty(&method_artifact_id) || proxy_basis.is_some() {
                    return Err(MeasurementError::InvalidMethodContract);
                }
            }
            MeasurementClass::Proxy => {
                if !nonempty(&proxy_basis) {
                    return Err(MeasurementError::InvalidMethodContract);
                }
            }
            MeasurementClass::LatentEstimate => {
                if !nonempty(&method_artifact_id) || proxy_basis.is_some() {
                    return Err(MeasurementError::InvalidMethodContract);
                }
            }
        }

        Ok(Self {
            specification_id,
            variable_id: variable.id().clone(),
            construct_id,
            source_namespace,
            source_series_id,
            source_unit,
            output_unit,
            population_scope,
            temporal_semantics,
            class,
            method_artifact_id,
            proxy_basis,
        })
    }

    fn specification_id(&self) -> &str {
        &self.specification_id
    }

    fn variable_id(&self) -> &VariableId {
        &self.variable_id
    }

    fn qualify(
        &self,
        observed: &ObservedMeasurementDescriptor,
    ) -> Result<MeasurementAdmission, MeasurementError> {
        if observed.source_namespace != self.source_namespace {
            return Err(MeasurementError::SourceNamespaceMismatch);
        }
        if observed.source_series_id != self.source_series_id {
            return Err(MeasurementError::SeriesMismatch);
        }
        if observed.construct_id != self.construct_id {
            return Err(MeasurementError::ConstructMismatch);
        }
        if observed.source_unit != self.source_unit {
            return Err(MeasurementError::SourceUnitMismatch);
        }
        if observed.output_unit != self.output_unit {
            return Err(MeasurementError::OutputUnitMismatch);
        }
        if observed.population_scope != self.population_scope {
            return Err(MeasurementError::PopulationScopeMismatch);
        }
        if observed.temporal_semantics != self.temporal_semantics {
            return Err(MeasurementError::TemporalSemanticsMismatch);
        }
        if observed.method_artifact_id != self.method_artifact_id {
            return Err(MeasurementError::MethodArtifactMismatch);
        }

        match self.class {
            MeasurementClass::Direct => Ok(MeasurementAdmission::DirectObservation),
            MeasurementClass::Derived => Ok(MeasurementAdmission::DerivedObservation {
                method_artifact_id: self.method_artifact_id.clone().expect("validated derived method"),
            }),
            MeasurementClass::Proxy => Ok(MeasurementAdmission::ProxyObservation {
                basis: self.proxy_basis.clone().expect("validated proxy basis"),
            }),
            MeasurementClass::LatentEstimate => Ok(MeasurementAdmission::LatentEstimate {
                estimator_artifact_id: self
                    .method_artifact_id
                    .clone()
                    .expect("validated latent estimator"),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObservedMeasurementDescriptor {
    construct_id: String,
    source_namespace: String,
    source_series_id: String,
    source_unit: UnitId,
    output_unit: UnitId,
    population_scope: String,
    temporal_semantics: TemporalSemantics,
    method_artifact_id: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn observed(
    construct_id: &str,
    source_namespace: &str,
    source_series_id: &str,
    source_unit: &str,
    output_unit: &str,
    population_scope: &str,
    temporal_semantics: TemporalSemantics,
    method_artifact_id: Option<&str>,
) -> ObservedMeasurementDescriptor {
    ObservedMeasurementDescriptor {
        construct_id: construct_id.into(),
        source_namespace: source_namespace.into(),
        source_series_id: source_series_id.into(),
        source_unit: UnitId::new(source_unit).unwrap(),
        output_unit: UnitId::new(output_unit).unwrap(),
        population_scope: population_scope.into(),
        temporal_semantics,
        method_artifact_id: method_artifact_id.map(str::to_string),
    }
}

fn unemployment_rate() -> EconomicVariable {
    EconomicVariable::new(
        VariableId::new("labor:unemployment_rate").unwrap(),
        StateDomain::Institutional,
        UnitId::new("percent").unwrap(),
        "Share of the labor-force construct classified as unemployed.",
    )
    .unwrap()
}

fn direct_unemployment_spec() -> MeasurementSpecification {
    MeasurementSpecification::new(
        "measure:unemployment:cps-u3-v1",
        &unemployment_rate(),
        "construct:unemployment:u3-v1",
        "statistics:household-survey",
        "series:unemployment-rate-u3",
        UnitId::new("percent").unwrap(),
        UnitId::new("percent").unwrap(),
        "civilian-noninstitutional-population-16plus",
        TemporalSemantics::PeriodAverage,
        MeasurementClass::Direct,
        None,
        None,
    )
    .unwrap()
}

#[test]
fn theory_variable_remains_separate_from_measurement_specification() {
    let variable = unemployment_rate();
    let household = direct_unemployment_spec();
    let administrative = MeasurementSpecification::new(
        "measure:unemployment:admin-claims-v1",
        &variable,
        "construct:registered-unemployment-claims-v1",
        "statistics:administrative",
        "series:registered-unemployment-claims-rate",
        UnitId::new("percent").unwrap(),
        UnitId::new("percent").unwrap(),
        "covered-workers",
        TemporalSemantics::PeriodAverage,
        MeasurementClass::Direct,
        None,
        None,
    )
    .unwrap();

    assert_eq!(household.variable_id(), variable.id());
    assert_eq!(administrative.variable_id(), variable.id());
    assert_ne!(household.specification_id(), administrative.specification_id());
}

#[test]
fn same_unit_does_not_make_a_different_construct_the_target_variable() {
    let spec = direct_unemployment_spec();
    let participation = observed(
        "construct:labor-force-participation-v1",
        "statistics:household-survey",
        "series:labor-force-participation-rate",
        "percent",
        "percent",
        "civilian-noninstitutional-population-16plus",
        TemporalSemantics::PeriodAverage,
        None,
    );

    assert_eq!(
        spec.qualify(&participation),
        Err(MeasurementError::SeriesMismatch)
    );
}

#[test]
fn population_scope_and_temporal_semantics_are_part_of_measurement_identity() {
    let spec = direct_unemployment_spec();
    let wrong_population = observed(
        "construct:unemployment:u3-v1",
        "statistics:household-survey",
        "series:unemployment-rate-u3",
        "percent",
        "percent",
        "youth-16-24",
        TemporalSemantics::PeriodAverage,
        None,
    );
    let wrong_temporal = observed(
        "construct:unemployment:u3-v1",
        "statistics:household-survey",
        "series:unemployment-rate-u3",
        "percent",
        "percent",
        "civilian-noninstitutional-population-16plus",
        TemporalSemantics::PointInTime,
        None,
    );

    assert_eq!(
        spec.qualify(&wrong_population),
        Err(MeasurementError::PopulationScopeMismatch)
    );
    assert_eq!(
        spec.qualify(&wrong_temporal),
        Err(MeasurementError::TemporalSemanticsMismatch)
    );
}

#[test]
fn derived_measurement_requires_explicit_method_and_unit_conversion() {
    let spec = MeasurementSpecification::new(
        "measure:unemployment:derived-bps-v1",
        &unemployment_rate(),
        "construct:unemployment:u3-v1",
        "statistics:derived-fixture",
        "series:unemployment-rate-bps",
        UnitId::new("basis_points").unwrap(),
        UnitId::new("percent").unwrap(),
        "civilian-noninstitutional-population-16plus",
        TemporalSemantics::FlowOverPeriod,
        MeasurementClass::Derived,
        Some("transform:basis-points-to-percent-v1".into()),
        None,
    )
    .unwrap();
    let matching = observed(
        "construct:unemployment:u3-v1",
        "statistics:derived-fixture",
        "series:unemployment-rate-bps",
        "basis_points",
        "percent",
        "civilian-noninstitutional-population-16plus",
        TemporalSemantics::FlowOverPeriod,
        Some("transform:basis-points-to-percent-v1"),
    );
    let wrong_transform = observed(
        "construct:unemployment:u3-v1",
        "statistics:derived-fixture",
        "series:unemployment-rate-bps",
        "basis_points",
        "percent",
        "civilian-noninstitutional-population-16plus",
        TemporalSemantics::FlowOverPeriod,
        Some("transform:post-hoc-v2"),
    );

    assert_eq!(
        spec.qualify(&matching),
        Ok(MeasurementAdmission::DerivedObservation {
            method_artifact_id: "transform:basis-points-to-percent-v1".into(),
        })
    );
    assert_eq!(
        spec.qualify(&wrong_transform),
        Err(MeasurementError::MethodArtifactMismatch)
    );
}

#[test]
fn proxy_and_latent_estimate_cannot_upgrade_to_direct_observation() {
    let tightness = EconomicVariable::new(
        VariableId::new("labor:market_tightness").unwrap(),
        StateDomain::Institutional,
        UnitId::new("index").unwrap(),
        "Latent labor-market tightness construct.",
    )
    .unwrap();
    let proxy = MeasurementSpecification::new(
        "measure:tightness:job-postings-proxy-v1",
        &tightness,
        "construct:job-postings-intensity-v1",
        "private:job-postings",
        "series:vacancy-intensity-index",
        UnitId::new("index").unwrap(),
        UnitId::new("index").unwrap(),
        "online-covered-vacancies",
        TemporalSemantics::PointInTime,
        MeasurementClass::Proxy,
        None,
        Some("postings cover only observed online vacancies".into()),
    )
    .unwrap();
    let proxy_observed = observed(
        "construct:job-postings-intensity-v1",
        "private:job-postings",
        "series:vacancy-intensity-index",
        "index",
        "index",
        "online-covered-vacancies",
        TemporalSemantics::PointInTime,
        None,
    );

    let expectations = EconomicVariable::new(
        VariableId::new("prices:inflation_expectations").unwrap(),
        StateDomain::Cognitive,
        UnitId::new("percent").unwrap(),
        "Latent expected inflation over the declared horizon.",
    )
    .unwrap();
    let latent = MeasurementSpecification::new(
        "measure:expectations:latent-v1",
        &expectations,
        "construct:inflation-expectations-latent-v1",
        "survey:expectations-panel",
        "series:expectations-input-panel",
        UnitId::new("percent").unwrap(),
        UnitId::new("percent").unwrap(),
        "survey-panel-respondents",
        TemporalSemantics::PeriodAverage,
        MeasurementClass::LatentEstimate,
        Some("estimator:state-space-v1".into()),
        None,
    )
    .unwrap();
    let latent_observed = observed(
        "construct:inflation-expectations-latent-v1",
        "survey:expectations-panel",
        "series:expectations-input-panel",
        "percent",
        "percent",
        "survey-panel-respondents",
        TemporalSemantics::PeriodAverage,
        Some("estimator:state-space-v1"),
    );

    let proxy_admission = proxy.qualify(&proxy_observed).unwrap();
    let latent_admission = latent.qualify(&latent_observed).unwrap();
    assert_eq!(
        proxy_admission,
        MeasurementAdmission::ProxyObservation {
            basis: "postings cover only observed online vacancies".into(),
        }
    );
    assert_eq!(
        latent_admission,
        MeasurementAdmission::LatentEstimate {
            estimator_artifact_id: "estimator:state-space-v1".into(),
        }
    );
    assert_ne!(proxy_admission, MeasurementAdmission::DirectObservation);
    assert_ne!(latent_admission, MeasurementAdmission::DirectObservation);
}

#[test]
fn direct_measurement_cannot_hide_unit_conversion_or_method_logic() {
    assert_eq!(
        MeasurementSpecification::new(
            "measure:bad-direct-v1",
            &unemployment_rate(),
            "construct:unemployment:u3-v1",
            "statistics:household-survey",
            "series:unemployment-rate-bps",
            UnitId::new("basis_points").unwrap(),
            UnitId::new("percent").unwrap(),
            "civilian-noninstitutional-population-16plus",
            TemporalSemantics::PeriodAverage,
            MeasurementClass::Direct,
            Some("transform:hidden-conversion".into()),
            None,
        ),
        Err(MeasurementError::InvalidMethodContract)
    );
}

#[test]
fn output_unit_must_match_the_etir_variable_unit() {
    assert_eq!(
        MeasurementSpecification::new(
            "measure:wrong-output-unit",
            &unemployment_rate(),
            "construct:unemployment:u3-v1",
            "statistics:household-survey",
            "series:unemployment-rate-u3",
            UnitId::new("fraction").unwrap(),
            UnitId::new("fraction").unwrap(),
            "civilian-noninstitutional-population-16plus",
            TemporalSemantics::PeriodAverage,
            MeasurementClass::Direct,
            None,
            None,
        ),
        Err(MeasurementError::TargetUnitMismatch)
    );
}
