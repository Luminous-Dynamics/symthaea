#![forbid(unsafe_code)]

//! Independent outcome evidence and uncertainty-aware pilot qualification.
//!
//! This module separates physical outcome evidence from process proxies such as
//! turbidity, conductivity, camera confidence, pump current or commanded dose.
//! Proxies can support operational monitoring after separate validation, but
//! they cannot by themselves qualify cleaning, rinse, cross-contamination or drying.
//!
//! The output is a narrowly scoped pilot evidence envelope, not certification,
//! a sanitization claim, a statistical guarantee or product release.

use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Endpoint {
    SoilRemovalResidual,
    FinalRinseResidue,
    CrossContamination,
    DryingResidualMoisture,
    // Relative microbial reduction only; this is not sanitization or disinfection.
    RelativeMicrobialReduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitDirection {
    AtMost,
    AtLeast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceKind {
    IndependentOutcomeAssay,
    CalibratedProcessProxy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Pass,
    Fail,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationScope {
    pub machine_model: String,
    pub programme: String,
    pub soil_class: String,
    pub surface_material: String,
    pub item_set_id: String,
    pub item_ids: Vec<String>,
    pub soil_batch_id: String,
    /// Seconds from soil application to the wash start, fixed per soil ladder.
    pub soil_age_s: u64,
    pub detergent_batch_id: String,
    pub water_batch_id: String,
    pub water_hardness_ppm_as_caco3: u16,
    pub inlet_temperature_millic: u32,
    pub ambient_temperature_millic: u32,
    pub ambient_relative_humidity_permille: u16,
    pub protocol_id: String,
    pub raw_archive_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EndpointCriterion {
    pub endpoint: Endpoint,
    pub criterion_id: String,
    pub metric_id: String,
    pub method_id: String,
    pub unit: String,
    pub direction: LimitDirection,
    pub limit: f64,
    pub criterion_evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutcomeObservation {
    pub run_id: String,
    pub item_id: String,
    pub endpoint: Endpoint,
    pub criterion_id: String,
    pub metric_id: String,
    pub evidence_kind: EvidenceKind,
    pub method_id: String,
    pub protocol_id: String,
    pub measured_value: f64,
    pub expanded_uncertainty: f64,
    pub unit: String,
    pub evidence_ref: String,
    pub calibration_bundle_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PilotPolicy {
    // Five independent runs is a pilot floor, not a confirmatory sample size.
    pub minimum_independent_runs: usize,
    pub require_drying_endpoint: bool,
    // This asks for relative microbiological evidence, not sanitization.
    pub require_relative_microbiology_endpoint: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualityError {
    InvalidScope(&'static str),
    InvalidItemSet,
    InvalidPilotRunCount,
    MissingRequiredCriterion(Endpoint),
    DuplicateCriterion(Endpoint),
    InvalidCriterion(Endpoint, &'static str),
    InvalidEvidenceDigest(&'static str),
    EmptyObservations,
    InvalidObservation(&'static str),
    ProtocolMismatch,
    UnknownItem(String),
    MissingCriterion(Endpoint),
    ProcessProxyCannotQualify { run_id: String, item_id: String, endpoint: Endpoint },
    UnitMismatch(Endpoint),
    CriterionMismatch(Endpoint),
    MetricMismatch(Endpoint),
    MethodMismatch(Endpoint),
    DuplicateObservation { run_id: String, item_id: String, endpoint: Endpoint },
    InsufficientIndependentRuns { item_id: String, endpoint: Endpoint, observed: usize, required: usize },
    UnbalancedRunMatrix { item_id: String, endpoint: Endpoint },
    OutcomeFailed { run_id: String, item_id: String, endpoint: Endpoint },
    OutcomeIndeterminate { run_id: String, item_id: String, endpoint: Endpoint },
}

impl fmt::Display for QualityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for QualityError {}

#[derive(Debug, Clone, PartialEq)]
pub struct PilotEvidenceEnvelope {
    scope: QualificationScope,
    required_endpoints: Vec<Endpoint>,
    criteria: Vec<EndpointCriterion>,
    method_ids: Vec<(Endpoint, String)>,
    run_ids: Vec<String>,
    observation_count: usize,
    minimum_independent_runs: usize,
}

impl PilotEvidenceEnvelope {
    pub fn scope(&self) -> &QualificationScope { &self.scope }
    pub fn required_endpoints(&self) -> &[Endpoint] { &self.required_endpoints }
    pub fn criteria(&self) -> &[EndpointCriterion] { &self.criteria }
    pub fn method_ids(&self) -> &[(Endpoint, String)] { &self.method_ids }
    pub fn run_ids(&self) -> &[String] { &self.run_ids }
    pub fn observation_count(&self) -> usize { self.observation_count }
    pub fn minimum_independent_runs(&self) -> usize { self.minimum_independent_runs }
}

fn canonical_non_empty(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_scope(scope: &QualificationScope) -> Result<(), QualityError> {
    for (field, value) in [
        ("machine_model", &scope.machine_model),
        ("programme", &scope.programme),
        ("soil_class", &scope.soil_class),
        ("surface_material", &scope.surface_material),
        ("item_set_id", &scope.item_set_id),
        ("soil_batch_id", &scope.soil_batch_id),
        ("detergent_batch_id", &scope.detergent_batch_id),
        ("water_batch_id", &scope.water_batch_id),
        ("protocol_id", &scope.protocol_id),
    ] {
        if !canonical_non_empty(value) {
            return Err(QualityError::InvalidScope(field));
        }
    }
    if scope.item_ids.is_empty() {
        return Err(QualityError::InvalidItemSet);
    }
    let mut item_ids = HashSet::with_capacity(scope.item_ids.len());
    for item_id in &scope.item_ids {
        if !canonical_non_empty(item_id) || !item_ids.insert(item_id.as_str()) {
            return Err(QualityError::InvalidItemSet);
        }
    }
    if scope.ambient_relative_humidity_permille > 1_000 {
        return Err(QualityError::InvalidScope("ambient_relative_humidity_permille"));
    }
    if !valid_digest(&scope.raw_archive_sha256) {
        return Err(QualityError::InvalidEvidenceDigest("raw_archive"));
    }
    Ok(())
}

fn required_endpoints(policy: PilotPolicy) -> Vec<Endpoint> {
    let mut endpoints = vec![
        Endpoint::SoilRemovalResidual,
        Endpoint::FinalRinseResidue,
        Endpoint::CrossContamination,
    ];
    if policy.require_drying_endpoint {
        endpoints.push(Endpoint::DryingResidualMoisture);
    }
    if policy.require_relative_microbiology_endpoint {
        endpoints.push(Endpoint::RelativeMicrobialReduction);
    }
    endpoints
}

fn classify(criterion: &EndpointCriterion, observation: &OutcomeObservation) -> Decision {
    let lower = observation.measured_value - observation.expanded_uncertainty;
    let upper = observation.measured_value + observation.expanded_uncertainty;
    match criterion.direction {
        LimitDirection::AtMost => {
            if upper <= criterion.limit {
                Decision::Pass
            } else if lower > criterion.limit {
                Decision::Fail
            } else {
                Decision::Indeterminate
            }
        }
        LimitDirection::AtLeast => {
            if lower >= criterion.limit {
                Decision::Pass
            } else if upper < criterion.limit {
                Decision::Fail
            } else {
                Decision::Indeterminate
            }
        }
    }
}

/// Build a narrowly scoped pilot envelope from independent item-level outcome data.
///
/// Soil residual, final-rinse residue and cross-contamination are mandatory.
/// Drying and relative microbiological reduction are separate opt-in endpoints.
/// Every item/endpoint pair needs the policy's minimum number of distinct physical
/// runs, and every uncertainty interval must clear the preregistered criterion.
/// Process proxies alone cannot qualify a scope.
///
/// Five runs are only a pilot floor. Calculate confirmatory sample size from pilot
/// variance and a preregistered effect size. This function does not certify a product.
pub fn qualify_pilot_evidence(
    scope: QualificationScope,
    policy: PilotPolicy,
    criteria: Vec<EndpointCriterion>,
    observations: Vec<OutcomeObservation>,
) -> Result<PilotEvidenceEnvelope, QualityError> {
    validate_scope(&scope)?;
    if policy.minimum_independent_runs < 5 || policy.minimum_independent_runs > 10_000 {
        return Err(QualityError::InvalidPilotRunCount);
    }
    if observations.is_empty() {
        return Err(QualityError::EmptyObservations);
    }

    let required = required_endpoints(policy);
    let required_set: HashSet<Endpoint> = required.iter().copied().collect();
    let mut criteria_by_endpoint = HashMap::with_capacity(criteria.len());

    for criterion in &criteria {
        if !required_set.contains(&criterion.endpoint) {
            continue;
        }
        if criteria_by_endpoint.contains_key(&criterion.endpoint) {
            return Err(QualityError::DuplicateCriterion(criterion.endpoint));
        }
        if !canonical_non_empty(&criterion.criterion_id) {
            return Err(QualityError::InvalidCriterion(criterion.endpoint, "criterion_id"));
        }
        if !canonical_non_empty(&criterion.metric_id) {
            return Err(QualityError::InvalidCriterion(criterion.endpoint, "metric_id"));
        }
        if !canonical_non_empty(&criterion.method_id) {
            return Err(QualityError::InvalidCriterion(criterion.endpoint, "method_id"));
        }
        if !canonical_non_empty(&criterion.unit) {
            return Err(QualityError::InvalidCriterion(criterion.endpoint, "unit"));
        }
        if !criterion.limit.is_finite() || criterion.limit < 0.0 {
            return Err(QualityError::InvalidCriterion(criterion.endpoint, "limit"));
        }
        if !valid_digest(&criterion.criterion_evidence_sha256) {
            return Err(QualityError::InvalidEvidenceDigest("criterion"));
        }
        criteria_by_endpoint.insert(criterion.endpoint, criterion);
    }
    for endpoint in &required {
        if !criteria_by_endpoint.contains_key(endpoint) {
            return Err(QualityError::MissingRequiredCriterion(*endpoint));
        }
    }

    let scope_items: HashSet<&str> = scope.item_ids.iter().map(String::as_str).collect();
    let mut seen: HashSet<(String, String, Endpoint)> = HashSet::with_capacity(observations.len());
    let mut run_ids = HashSet::new();
    let mut counts: HashMap<(String, Endpoint), HashSet<String>> = HashMap::new();
    let mut method_by_endpoint: HashMap<Endpoint, String> = HashMap::new();

    for observation in &observations {
        if !canonical_non_empty(&observation.run_id)
            || !canonical_non_empty(&observation.item_id)
            || !canonical_non_empty(&observation.criterion_id)
            || !canonical_non_empty(&observation.metric_id)
            || !canonical_non_empty(&observation.method_id)
            || !canonical_non_empty(&observation.protocol_id)
            || !canonical_non_empty(&observation.unit)
            || !canonical_non_empty(&observation.evidence_ref)
        {
            return Err(QualityError::InvalidObservation("blank or non-canonical identity"));
        }
        if observation.protocol_id != scope.protocol_id {
            return Err(QualityError::ProtocolMismatch);
        }
        if !scope_items.contains(observation.item_id.as_str()) {
            return Err(QualityError::UnknownItem(observation.item_id.clone()));
        }
        if observation.evidence_kind != EvidenceKind::IndependentOutcomeAssay {
            return Err(QualityError::ProcessProxyCannotQualify {
                run_id: observation.run_id.clone(),
                item_id: observation.item_id.clone(),
                endpoint: observation.endpoint,
            });
        }
        if !observation.measured_value.is_finite()
            || observation.measured_value < 0.0
            || !observation.expanded_uncertainty.is_finite()
            || observation.expanded_uncertainty < 0.0
        {
            return Err(QualityError::InvalidObservation(
                "measurement and expanded uncertainty must be finite and non-negative",
            ));
        }
        let lower = observation.measured_value - observation.expanded_uncertainty;
        let upper = observation.measured_value + observation.expanded_uncertainty;
        if !lower.is_finite() || !upper.is_finite() {
            return Err(QualityError::InvalidObservation("uncertainty interval overflow"));
        }
        if !valid_digest(&observation.calibration_bundle_sha256) {
            return Err(QualityError::InvalidEvidenceDigest("measurement calibration"));
        }

        let criterion = criteria_by_endpoint
            .get(&observation.endpoint)
            .copied()
            .ok_or(QualityError::MissingCriterion(observation.endpoint))?;
        if observation.criterion_id != criterion.criterion_id {
            return Err(QualityError::CriterionMismatch(observation.endpoint));
        }
        if observation.metric_id != criterion.metric_id {
            return Err(QualityError::MetricMismatch(observation.endpoint));
        }
        if observation.unit != criterion.unit {
            return Err(QualityError::UnitMismatch(observation.endpoint));
        }
        if observation.method_id != criterion.method_id {
            return Err(QualityError::MethodMismatch(observation.endpoint));
        }
        match method_by_endpoint.get(&observation.endpoint) {
            Some(method) if method != &observation.method_id => {
                return Err(QualityError::MethodMismatch(observation.endpoint));
            }
            None => { method_by_endpoint.insert(observation.endpoint, observation.method_id.clone()); }
            _ => {}
        }

        let key = (
            observation.run_id.clone(),
            observation.item_id.clone(),
            observation.endpoint,
        );
        if !seen.insert(key) {
            return Err(QualityError::DuplicateObservation {
                run_id: observation.run_id.clone(),
                item_id: observation.item_id.clone(),
                endpoint: observation.endpoint,
            });
        }
        run_ids.insert(observation.run_id.clone());
        counts.entry((observation.item_id.clone(), observation.endpoint))
            .or_default()
            .insert(observation.run_id.clone());

        match classify(criterion, observation) {
            Decision::Pass => {}
            Decision::Fail => return Err(QualityError::OutcomeFailed {
                run_id: observation.run_id.clone(),
                item_id: observation.item_id.clone(),
                endpoint: observation.endpoint,
            }),
            Decision::Indeterminate => return Err(QualityError::OutcomeIndeterminate {
                run_id: observation.run_id.clone(),
                item_id: observation.item_id.clone(),
                endpoint: observation.endpoint,
            }),
        }
    }

    let reference_key = (scope.item_ids[0].clone(), required[0]);
    let reference_runs = counts
        .get(&reference_key)
        .ok_or(QualityError::InsufficientIndependentRuns {
            item_id: reference_key.0.clone(),
            endpoint: reference_key.1,
            observed: 0,
            required: policy.minimum_independent_runs,
        })?;

    for item_id in &scope.item_ids {
        for endpoint in &required {
            let observed_runs = counts.get(&(item_id.clone(), *endpoint));
            let count = observed_runs.map_or(0, HashSet::len);
            if count < policy.minimum_independent_runs {
                return Err(QualityError::InsufficientIndependentRuns {
                    item_id: item_id.clone(),
                    endpoint: *endpoint,
                    observed: count,
                    required: policy.minimum_independent_runs,
                });
            }
            if observed_runs != Some(reference_runs) {
                return Err(QualityError::UnbalancedRunMatrix {
                    item_id: item_id.clone(),
                    endpoint: *endpoint,
                });
            }
        }
    }

    let criteria_for_envelope: Vec<EndpointCriterion> = required
        .iter()
        .map(|endpoint| {
            (**criteria_by_endpoint
                .get(endpoint)
                .expect("required criterion checked above"))
                .clone()
        })
        .collect();
    let methods_for_envelope: Vec<(Endpoint, String)> = required
        .iter()
        .map(|endpoint| {
            (
                *endpoint,
                method_by_endpoint
                    .get(endpoint)
                    .expect("required outcome observations checked above")
                    .clone(),
            )
        })
        .collect();
    let mut sorted_run_ids: Vec<String> = run_ids.into_iter().collect();
    sorted_run_ids.sort();
    Ok(PilotEvidenceEnvelope {
        scope,
        required_endpoints: required,
        criteria: criteria_for_envelope,
        method_ids: methods_for_envelope,
        run_ids: sorted_run_ids,
        observation_count: observations.len(),
        minimum_independent_runs: policy.minimum_independent_runs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> QualificationScope {
        QualificationScope {
            machine_model: "instantclean-prototype-a".to_owned(),
            programme: "quickcell-standard-v1".to_owned(),
            soil_class: "starch".to_owned(),
            surface_material: "ceramic".to_owned(),
            item_set_id: "two-ceramic-cups-v1".to_owned(),
            item_ids: vec!["cup-a".to_owned(), "cup-b".to_owned()],
            soil_batch_id: "soil-lot-2026-01".to_owned(),
            soil_age_s: 21_600,
            detergent_batch_id: "detergent-lot-1".to_owned(),
            water_batch_id: "water-lot-1".to_owned(),
            water_hardness_ppm_as_caco3: 75,
            inlet_temperature_millic: 15_000,
            ambient_temperature_millic: 20_000,
            ambient_relative_humidity_permille: 500,
            protocol_id: "independent-assay-v1".to_owned(),
            raw_archive_sha256: "a".repeat(64),
        }
    }

    fn policy() -> PilotPolicy {
        PilotPolicy {
            minimum_independent_runs: 5,
            require_drying_endpoint: false,
            require_relative_microbiology_endpoint: false,
        }
    }

    fn criteria(policy: PilotPolicy) -> Vec<EndpointCriterion> {
        let mut result = vec![
            EndpointCriterion {
                endpoint: Endpoint::SoilRemovalResidual,
                criterion_id: "soil-residual-limit-v1".to_owned(),
                metric_id: "residual_soil_area_fraction".to_owned(),
                method_id: "fluorescence-assay-v1".to_owned(),
                unit: "fraction".to_owned(),
                direction: LimitDirection::AtMost,
                limit: 0.05,
                criterion_evidence_sha256: "b".repeat(64),
            },
            EndpointCriterion {
                endpoint: Endpoint::FinalRinseResidue,
                criterion_id: "rinse-residue-limit-v1".to_owned(),
                metric_id: "final_rinse_residue".to_owned(),
                method_id: "rinse-residue-assay-v1".to_owned(),
                unit: "mg/L".to_owned(),
                direction: LimitDirection::AtMost,
                limit: 5.0,
                criterion_evidence_sha256: "c".repeat(64),
            },
            EndpointCriterion {
                endpoint: Endpoint::CrossContamination,
                criterion_id: "cross-transfer-limit-v1".to_owned(),
                metric_id: "marker_transfer_mass".to_owned(),
                method_id: "witness-coupon-assay-v1".to_owned(),
                unit: "mg".to_owned(),
                direction: LimitDirection::AtMost,
                limit: 0.01,
                criterion_evidence_sha256: "d".repeat(64),
            },
        ];
        if policy.require_drying_endpoint {
            result.push(EndpointCriterion {
                endpoint: Endpoint::DryingResidualMoisture,
                criterion_id: "drying-moisture-limit-v1".to_owned(),
                metric_id: "residual_moisture".to_owned(),
                method_id: "gravimetric-moisture-v1".to_owned(),
                unit: "g".to_owned(),
                direction: LimitDirection::AtMost,
                limit: 0.5,
                criterion_evidence_sha256: "e".repeat(64),
            });
        }
        if policy.require_relative_microbiology_endpoint {
            result.push(EndpointCriterion {
                endpoint: Endpoint::RelativeMicrobialReduction,
                criterion_id: "relative-microbe-limit-v1".to_owned(),
                metric_id: "log10_reduction".to_owned(),
                method_id: "relative-microbiology-v1".to_owned(),
                unit: "log10".to_owned(),
                direction: LimitDirection::AtLeast,
                limit: 2.0,
                criterion_evidence_sha256: "f".repeat(64),
            });
        }
        result
    }

    fn observations(policy: PilotPolicy) -> Vec<OutcomeObservation> {
        let endpoints = required_endpoints(policy);
        let acceptance_criteria = criteria(policy);
        let item_ids = scope().item_ids;
        let mut result = Vec::new();
        for run in 1..=5 {
            for item in &item_ids {
                for endpoint in &endpoints {
                    let (method, unit, value, uncertainty) = match endpoint {
                        Endpoint::SoilRemovalResidual => ("fluorescence-assay-v1", "fraction", 0.02, 0.005),
                        Endpoint::FinalRinseResidue => ("rinse-residue-assay-v1", "mg/L", 2.0, 0.5),
                        Endpoint::CrossContamination => ("witness-coupon-assay-v1", "mg", 0.003, 0.001),
                        Endpoint::DryingResidualMoisture => ("gravimetric-moisture-v1", "g", 0.2, 0.05),
                        Endpoint::RelativeMicrobialReduction => ("relative-microbiology-v1", "log10", 2.5, 0.2),
                    };
                    result.push(OutcomeObservation {
                        run_id: format!("run-{run:02}"),
                        item_id: item.clone(),
                        endpoint: *endpoint,
                        criterion_id: acceptance_criteria
                            .iter()
                            .find(|criterion| criterion.endpoint == *endpoint)
                            .unwrap()
                            .criterion_id
                            .clone(),
                        metric_id: acceptance_criteria
                            .iter()
                            .find(|criterion| criterion.endpoint == *endpoint)
                            .unwrap()
                            .metric_id
                            .clone(),
                        evidence_kind: EvidenceKind::IndependentOutcomeAssay,
                        method_id: method.to_owned(),
                        protocol_id: "independent-assay-v1".to_owned(),
                        measured_value: value,
                        expanded_uncertainty: uncertainty,
                        unit: unit.to_owned(),
                        evidence_ref: format!("evidence-{run}-{item}-{endpoint:?}"),
                        calibration_bundle_sha256: "1".repeat(64),
                    });
                }
            }
        }
        result
    }

    #[test]
    fn qualifies_only_complete_item_level_independent_pilot_evidence() {
        let policy = policy();
        let envelope = qualify_pilot_evidence(scope(), policy, criteria(policy), observations(policy)).unwrap();
        assert_eq!(envelope.run_ids().len(), 5);
        assert_eq!(envelope.observation_count(), 30);
        assert_eq!(envelope.required_endpoints(), &[
            Endpoint::SoilRemovalResidual,
            Endpoint::FinalRinseResidue,
            Endpoint::CrossContamination
        ]);
    }

    #[test]
    fn process_proxy_cannot_qualify_cleaning() {
        let policy = policy();
        let mut readings = observations(policy);
        readings[0].evidence_kind = EvidenceKind::CalibratedProcessProxy;
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::ProcessProxyCannotQualify { .. })
        ));
    }

    #[test]
    fn an_outcome_failure_rejects_the_entire_pilot_envelope() {
        let policy = policy();
        let mut readings = observations(policy);
        readings[0].measured_value = 0.08;
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::OutcomeFailed { endpoint: Endpoint::SoilRemovalResidual, .. })
        ));
    }

    #[test]
    fn uncertainty_overlapping_the_limit_is_indeterminate() {
        let policy = policy();
        let mut readings = observations(policy);
        readings[0].measured_value = 0.047;
        readings[0].expanded_uncertainty = 0.01;
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::OutcomeIndeterminate { endpoint: Endpoint::SoilRemovalResidual, .. })
        ));
    }

    #[test]
    fn requires_five_independent_runs_per_item_and_endpoint() {
        let policy = policy();
        let mut readings = observations(policy);
        readings.retain(|reading| !(reading.item_id == "cup-b"
            && reading.endpoint == Endpoint::FinalRinseResidue
            && reading.run_id == "run-05"));
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::InsufficientIndependentRuns {
                item_id, endpoint: Endpoint::FinalRinseResidue, observed: 4, required: 5
            }) if item_id == "cup-b"
        ));
    }

    #[test]
    fn requires_the_same_physical_run_matrix_for_every_item_and_endpoint() {
        let policy = policy();
        let mut readings = observations(policy);
        readings.retain(|reading| !(reading.item_id == "cup-b"
            && reading.endpoint == Endpoint::FinalRinseResidue
            && reading.run_id == "run-05"));
        readings.push(OutcomeObservation {
            run_id: "replacement-run".to_owned(),
            item_id: "cup-b".to_owned(),
            endpoint: Endpoint::FinalRinseResidue,
            criterion_id: criteria(policy).into_iter()
                .find(|criterion| criterion.endpoint == Endpoint::FinalRinseResidue)
                .unwrap().criterion_id,
            metric_id: "final_rinse_residue".to_owned(),
            evidence_kind: EvidenceKind::IndependentOutcomeAssay,
            method_id: "rinse-residue-assay-v1".to_owned(),
            protocol_id: "independent-assay-v1".to_owned(),
            measured_value: 2.0,
            expanded_uncertainty: 0.5,
            unit: "mg/L".to_owned(),
            evidence_ref: "replacement-rinse-evidence".to_owned(),
            calibration_bundle_sha256: "1".repeat(64),
        });
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::UnbalancedRunMatrix {
                item_id, endpoint: Endpoint::FinalRinseResidue
            }) if item_id == "cup-b"
        ));
    }

    #[test]
    fn rejects_duplicate_item_endpoint_run_observation() {
        let policy = policy();
        let mut readings = observations(policy);
        readings.push(readings[0].clone());
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::DuplicateObservation { .. })
        ));
    }

    #[test]
    fn requires_drying_and_relative_microbiology_only_when_requested() {
        let mut policy = policy();
        policy.require_drying_endpoint = true;
        policy.require_relative_microbiology_endpoint = true;
        let envelope = qualify_pilot_evidence(scope(), policy, criteria(policy), observations(policy)).unwrap();
        assert!(envelope.required_endpoints().contains(&Endpoint::DryingResidualMoisture));
        assert!(envelope.required_endpoints().contains(&Endpoint::RelativeMicrobialReduction));
    }

    #[test]
    fn relative_microbiology_is_not_a_sanitization_endpoint() {
        assert_eq!(required_endpoints(PilotPolicy {
            minimum_independent_runs: 5,
            require_drying_endpoint: false,
            require_relative_microbiology_endpoint: true,
        }), vec![
            Endpoint::SoilRemovalResidual,
            Endpoint::FinalRinseResidue,
            Endpoint::CrossContamination,
            Endpoint::RelativeMicrobialReduction,
        ]);
    }

    #[test]
    fn rejects_measurement_bound_to_wrong_criterion_or_metric() {
        let policy = policy();
        let mut readings = observations(policy);
        readings[0].criterion_id = "different-acceptance-rule".to_owned();
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::CriterionMismatch(Endpoint::SoilRemovalResidual))
        ));

        let mut readings = observations(policy);
        readings[0].metric_id = "rinse_conductivity".to_owned();
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::MetricMismatch(Endpoint::SoilRemovalResidual))
        ));
    }

    #[test]
    fn rejects_protocol_and_unit_mismatch() {
        let policy = policy();
        let mut readings = observations(policy);
        readings[0].protocol_id = "another-protocol".to_owned();
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::ProtocolMismatch)
        ));

        let mut readings = observations(policy);
        readings[0].unit = "mg".to_owned();
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::UnitMismatch(Endpoint::SoilRemovalResidual))
        ));
    }

    #[test]
    fn rejects_measurement_method_not_named_by_its_criterion() {
        let policy = policy();
        let mut readings = observations(policy);
        readings[0].method_id = "fluorescence-assay-v2".to_owned();
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::MethodMismatch(Endpoint::SoilRemovalResidual))
        ));
    }

    #[test]
    fn rejects_inconsistent_methods_and_malformed_evidence() {
        let policy = policy();
        let mut readings = observations(policy);
        readings[1].method_id = "different-fluorescence-model".to_owned();
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::MethodMismatch(Endpoint::SoilRemovalResidual))
        ));

        let mut readings = observations(policy);
        readings[0].calibration_bundle_sha256 = "unknown".to_owned();
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), readings),
            Err(QualityError::InvalidEvidenceDigest("measurement calibration"))
        ));
    }

    #[test]
    fn missing_criterion_for_requested_claim_is_rejected() {
        let mut policy = policy();
        policy.require_drying_endpoint = true;
        let mut acceptance = criteria(policy);
        acceptance.retain(|criterion| criterion.endpoint != Endpoint::DryingResidualMoisture);
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, acceptance, observations(policy)),
            Err(QualityError::MissingRequiredCriterion(Endpoint::DryingResidualMoisture))
        ));
    }

    #[test]
    fn rejects_duplicate_or_malformed_item_sets() {
        let mut invalid_scope = scope();
        invalid_scope.item_ids.push("cup-a".to_owned());
        let policy = policy();
        assert!(matches!(
            qualify_pilot_evidence(invalid_scope, policy, criteria(policy), observations(policy)),
            Err(QualityError::InvalidItemSet)
        ));
    }

    #[test]
    fn requires_five_runs_minimum_even_when_caller_requests_less() {
        let mut policy = policy();
        policy.minimum_independent_runs = 2;
        assert!(matches!(
            qualify_pilot_evidence(scope(), policy, criteria(policy), observations(policy)),
            Err(QualityError::InvalidPilotRunCount)
        ));
    }
}
