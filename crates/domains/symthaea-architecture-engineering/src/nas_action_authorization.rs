//! Fail-closed NAS action preflight across confidence, hardware, portfolio, and resource evidence.
//!
//! This module deliberately separates preflight eligibility from execution authority.
//! It recomputes the portfolio measurement action, validates the objective/hardware
//! profile, replays the anytime-confidence ledger, and re-computes the vector resource
//! frontier. A successful receipt is a preflight result only: it is not an atomic
//! reservation, a signed capability, or permission to execute an action.
//!
//! Promotion remains blocked until #9388 provides a private, receipt-replayed
//! qualification capability. Pruning preflight verifies the #9329 run-wide pruning
//! ledger and exact scope-bound event identity, but still does not reserve resources
//! or mutate an external run state. Those operations require a transactional
//! reservation service and independent qualification integration.
//!
//! SHA-256 digests here bind evidence identities. They do not prove truth, authorship,
//! chronology, or statistical independence.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::anytime_confidence_ledger::{
    verify_anytime_confidence_ledger, AnytimeConfidenceLedger,
};
use crate::anytime_qmc_confidence::AnytimeDecision;
use crate::hardware_objectives::{
    compute_hardware_profile_sha256, validate_hardware_profile,
    validate_objective_registry, ObjectiveDirection, ObjectiveRegistry, ObjectiveSource,
};
use crate::multi_resource_budget::{
    plan_resource_action_frontier, validate_resource_registry, ResourceBudget,
    ResourceBudgetAction, ResourceRegistry,
};
use crate::pareto_hypervolume_qmc::HypervolumeQmcDesign;
use crate::pareto_portfolio_voc::{
    evaluate_portfolio_measurement_voc, CandidatePortfolio, ParetoPortfolioConfig,
    ParetoPortfolioPosterior, PortfolioMeasurementAction, PortfolioParetoVoC,
};
use crate::run_wide_pruning_risk::RunWidePruningRiskLedger;

const SCOPE_DOMAIN: &[u8] = b"symthaea.nas.action-preflight-scope.v0.1";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.nas.action-preflight-receipt.v0.1";
const INSTRUMENT_DOMAIN: &[u8] = b"symthaea.nas.instrument-identity.v0.1";
const OBJECTIVE_ORDER_DOMAIN: &[u8] = b"symthaea.nas.objective-order.v0.1";
const PRUNE_EVENT_PREFIX: &str = "prune:";
const MEASUREMENT_STREAM_PREFIX: &str = "measurement:";
const TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NasActionKind {
    AcquireMeasurement,
    PromoteCandidate,
    PruneCandidate,
}

/// Complete identity tuple for one measurement decision.
///
/// Every digest must be computed from the exact upstream artifact. The scope hash
/// becomes the confidence-ledger stream identity, so a receipt for another candidate,
/// profile, posterior, QMC split, protocol, or budget cannot be substituted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NasMeasurementActionScope {
    pub run_id: String,
    /// Git commit object ID; accepts lowercase SHA-1 (40 hex) or SHA-256 (64 hex).
    pub target_commit_id: String,
    pub action_id: String,
    pub candidate_id: String,
    pub candidate_config_sha256: String,
    pub objective_id: String,
    pub objective_ids: Vec<String>,
    pub objective_registry_sha256: String,
    pub objective_order_sha256: String,
    pub posterior_sha256: String,
    pub portfolio_family_sha256: String,
    pub qmc_sampling_design_sha256: String,
    pub selection_replicates_sha256: String,
    pub validation_replicates_sha256: String,
    pub hardware_profile_sha256: String,
    pub measurement_protocol_sha256: String,
    pub instrument_identity_sha256: String,
    pub observation_likelihood_model_sha256: String,
    /// Exact frozen family manifest; excluded only from derived stream ID to avoid a hash cycle.
    pub confidence_family_manifest_sha256: String,
    pub resource_registry_sha256: String,
    pub resource_budget_sha256: String,
    pub resource_cost_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NasPruneActionScope {
    pub run_id: String,
    pub target_commit_id: String,
    pub action_id: String,
    pub candidate_id: String,
    pub candidate_config_sha256: String,
    pub selected_frontier_sha256: String,
    pub pruned_population_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NasActionPreflightReceipt {
    pub schema: String,
    pub version: String,
    pub action_kind: NasActionKind,
    pub disposition: String,
    pub run_id: String,
    pub target_commit_id: String,
    pub action_id: String,
    pub candidate_id: String,
    pub scope_sha256: String,
    pub confidence_family_manifest_sha256: Option<String>,
    pub confidence_ledger_sha256: Option<String>,
    pub confidence_receipt_sha256: Option<String>,
    pub portfolio_decision_sha256: Option<String>,
    pub objective_registry_sha256: Option<String>,
    pub hardware_profile_sha256: Option<String>,
    pub resource_registry_sha256: Option<String>,
    pub resource_budget_sha256: Option<String>,
    pub resource_cost_sha256: Option<String>,
    pub resource_frontier_sha256: Option<String>,
    pub pruning_risk_genesis_sha256: Option<String>,
    pub pruning_risk_receipt_sha256: Option<String>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NasAuthorizationError {
    pub field: String,
    pub message: String,
}

impl NasAuthorizationError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for NasAuthorizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for NasAuthorizationError {}

pub struct MeasurementPreflightRequest<'a> {
    pub scope: &'a NasMeasurementActionScope,
    pub confidence_ledger: &'a AnytimeConfidenceLedger,
    pub objective_registry: &'a ObjectiveRegistry,
    pub posterior: &'a ParetoPortfolioPosterior,
    pub portfolios: &'a [CandidatePortfolio],
    pub portfolio_config: &'a ParetoPortfolioConfig,
    pub qmc_design: &'a HypervolumeQmcDesign,
    pub measurement_action: &'a PortfolioMeasurementAction,
    pub resource_registry: &'a ResourceRegistry,
    pub resource_budget: &'a ResourceBudget,
    pub resource_actions: &'a [ResourceBudgetAction],
}

pub struct PrunePreflightRequest<'a> {
    pub scope: &'a NasPruneActionScope,
    pub risk_ledger: &'a RunWidePruningRiskLedger,
}

/// Compute the digest that binds all measurement evidence identities.
pub fn compute_measurement_scope_sha256(
    scope: &NasMeasurementActionScope,
) -> Result<String, NasAuthorizationError> {
    validate_measurement_scope(scope)?;
    let bytes = serde_json::to_vec(scope)
        .map_err(|error| NasAuthorizationError::new("scope", error.to_string()))?;
    Ok(digest_bytes(SCOPE_DOMAIN, &bytes))
}

/// The confidence ledger's preregistered stream ID for this exact measurement scope.
pub fn measurement_confidence_stream_id(
    scope: &NasMeasurementActionScope,
) -> Result<String, NasAuthorizationError> {
    // Derive the stream identity from the full action context but not the manifest digest:
    // the manifest contains this stream ID, so including the manifest digest would form a cycle.
    // The final scope digest does include the manifest digest and is what the preflight receipt binds.
    let mut stream_scope = scope.clone();
    stream_scope.confidence_family_manifest_sha256 = "0".repeat(64);
    Ok(format!(
        "{MEASUREMENT_STREAM_PREFIX}{}",
        compute_measurement_scope_sha256(&stream_scope)?
    ))
}

/// Check all preflight evidence for one decoupled hardware measurement action.
///
/// This recomputes PortfolioParetoVoC from the exact posterior, portfolio family, QMC
/// design and action. It then checks the matching preregistered confidence stream and
/// recomputes the component-wise resource frontier. No state is reserved or consumed.
pub fn verify_measurement_preflight(
    request: &MeasurementPreflightRequest<'_>,
) -> Result<NasActionPreflightReceipt, NasAuthorizationError> {
    let scope = request.scope;
    let scope_sha256 = compute_measurement_scope_sha256(scope)?;
    validate_objective_registry(request.objective_registry)
        .map_err(|error| NasAuthorizationError::new("objective_registry", error.to_string()))?;
    validate_resource_registry(request.resource_registry)
        .map_err(|error| NasAuthorizationError::new("resource_registry", error.to_string()))?;

    if request.objective_registry.registry_sha256 != scope.objective_registry_sha256 {
        return Err(NasAuthorizationError::new(
            "objective_registry_sha256",
            "objective registry does not match the preregistered action scope",
        ));
    }
    if request.resource_registry.registry_sha256 != scope.resource_registry_sha256
        || request.resource_budget.resource_registry_sha256 != scope.resource_registry_sha256
    {
        return Err(NasAuthorizationError::new(
            "resource_registry_sha256",
            "resource registry or budget is bound to a different resource registry",
        ));
    }

    let profile = request
        .objective_registry
        .hardware_profiles
        .iter()
        .find(|profile| profile.profile_sha256 == scope.hardware_profile_sha256)
        .ok_or_else(|| {
            NasAuthorizationError::new(
                "hardware_profile_sha256",
                "the exact hardware profile is absent from the objective registry",
            )
        })?;
    validate_hardware_profile(profile)
        .map_err(|error| NasAuthorizationError::new("hardware_profile", error.to_string()))?;
    if compute_hardware_profile_sha256(profile)
        .map_err(|error| NasAuthorizationError::new("hardware_profile", error.to_string()))?
        != scope.hardware_profile_sha256
    {
        return Err(NasAuthorizationError::new(
            "hardware_profile_sha256",
            "hardware profile digest does not recompute",
        ));
    }
    if profile.measurement_protocol_sha256 != scope.measurement_protocol_sha256 {
        return Err(NasAuthorizationError::new(
            "measurement_protocol_sha256",
            "hardware profile and action scope name different measurement protocols",
        ));
    }
    let instrument_identity_sha256 = compute_instrument_identity_sha256(profile)?;
    if instrument_identity_sha256 != scope.instrument_identity_sha256 {
        return Err(NasAuthorizationError::new(
            "instrument_identity_sha256",
            "instrument model, instrument ID, frequency, or profile identity changed",
        ));
    }

    let objective = request
        .objective_registry
        .objectives
        .iter()
        .find(|objective| objective.objective_id == scope.objective_id)
        .ok_or_else(|| {
            NasAuthorizationError::new(
                "objective_id",
                "the requested measurement objective is absent from the registry",
            )
        })?;
    if objective.hardware_profile_sha256.as_deref() != Some(scope.hardware_profile_sha256.as_str())
    {
        return Err(NasAuthorizationError::new(
            "hardware_profile_sha256",
            "objective is not bound to the exact hardware profile",
        ));
    }
    if objective.measurement_protocol_sha256 != scope.measurement_protocol_sha256 {
        return Err(NasAuthorizationError::new(
            "measurement_protocol_sha256",
            "objective and scope use different measurement protocols",
        ));
    }
    if !matches!(
        objective.source,
        ObjectiveSource::PhysicalDeviceMeasurement
            | ObjectiveSource::ImportedPhysicalDeviceMeasurement
    ) {
        return Err(NasAuthorizationError::new(
            "objective_source",
            "hardware measurement preflight requires a physical or imported physical measurement objective",
        ));
    }

    let mut objective_ids: Vec<String> = request
        .objective_registry
        .objectives
        .iter()
        .map(|definition| definition.objective_id.clone())
        .collect();
    objective_ids.sort();
    if objective_ids != scope.objective_ids {
        return Err(NasAuthorizationError::new(
            "objective_ids",
            "ordered objective coordinates do not match the objective registry",
        ));
    }
    if compute_objective_order_sha256(&objective_ids)? != scope.objective_order_sha256 {
        return Err(NasAuthorizationError::new(
            "objective_order_sha256",
            "objective coordinate-order digest does not recompute",
        ));
    }

    let action = request.measurement_action;
    if action.action_id != scope.action_id
        || action.candidate_id != scope.candidate_id
        || action.candidate_config_sha256 != scope.candidate_config_sha256
        || action.objective_id != scope.objective_id
        || action.objective_registry_sha256 != scope.objective_registry_sha256
        || action.hardware_profile_sha256.as_deref() != Some(scope.hardware_profile_sha256.as_str())
        || action.measurement_protocol_sha256 != scope.measurement_protocol_sha256
        || action.observation_likelihood_model_sha256 != scope.observation_likelihood_model_sha256
    {
        return Err(NasAuthorizationError::new(
            "measurement_action",
            "action identity, candidate configuration, objective, profile, protocol, or likelihood digest does not match the scope",
        ));
    }

    let candidate_index = request
        .posterior
        .candidate_ids
        .iter()
        .position(|candidate_id| candidate_id == &scope.candidate_id)
        .ok_or_else(|| {
            NasAuthorizationError::new(
                "candidate_id",
                "candidate is absent from the evaluated posterior",
            )
        })?;
    if request.posterior.candidate_config_sha256.get(candidate_index)
        != Some(&scope.candidate_config_sha256)
    {
        return Err(NasAuthorizationError::new(
            "candidate_config_sha256",
            "posterior candidate configuration differs from the action scope",
        ));
    }
    if request.posterior.objective_ids != scope.objective_ids {
        return Err(NasAuthorizationError::new(
            "objective_ids",
            "posterior coordinate order differs from the action scope",
        ));
    }

    let voc = evaluate_portfolio_measurement_voc(
        request.objective_registry,
        request.posterior,
        request.portfolios,
        request.portfolio_config,
        request.qmc_design,
        action,
    )
    .map_err(|error| NasAuthorizationError::new("portfolio_voc", error.to_string()))?;
    validate_portfolio_decision_scope(&voc, scope)?;

    if objective.measurement_cost_unit != voc.resource_unit
        || objective.measurement_cost_unit != action.resource_unit
        || action.incremental_cost_units != voc.incremental_cost_units
    {
        return Err(NasAuthorizationError::new(
            "measurement_cost",
            "objective acquisition-cost unit or units do not match the recomputed portfolio measurement action",
        ));
    }
    let cost_vector = action_cost_vector(request)?;
    let matching_cost_dimensions = cost_vector
        .resources
        .iter()
        .filter(|quantity| {
            quantity.unit == objective.measurement_cost_unit
                && quantity.units == voc.incremental_cost_units
                && quantity.cost_model_sha256 == objective.measurement_cost_model_sha256
        })
        .count();
    if matching_cost_dimensions != 1 {
        return Err(NasAuthorizationError::new(
            "resource_cost",
            "exactly one resource dimension must reconcile to the objective acquisition cost and unit",
        ));
    }

    let resource_cost_sha256 = digest_serde(b"symthaea.nas.resource-cost-vector.v0.1", cost_vector)?;
    if resource_cost_sha256 != scope.resource_cost_sha256 {
        return Err(NasAuthorizationError::new(
            "resource_cost_sha256",
            "resource cost vector digest differs from the declared action scope",
        ));
    }
    let resource_budget_sha256 = digest_serde(
        b"symthaea.nas.resource-budget.v0.1",
        request.resource_budget,
    )?;
    if resource_budget_sha256 != scope.resource_budget_sha256 {
        return Err(NasAuthorizationError::new(
            "resource_budget_sha256",
            "resource budget or protected reserves differ from the action scope",
        ));
    }

    let scoped_resource_action = request
        .resource_actions
        .iter()
        .find(|candidate| candidate.action_id == scope.action_id)
        .ok_or_else(|| {
            NasAuthorizationError::new(
                "resource_actions",
                "matching action is missing from the supplied resource frontier",
            )
        })?;
    if request
        .resource_actions
        .iter()
        .filter(|candidate| candidate.action_id == scope.action_id)
        .count()
        != 1
    {
        return Err(NasAuthorizationError::new(
            "resource_actions",
            "action must appear exactly once in the resource frontier input",
        ));
    }
    if scoped_resource_action.resource_cost != *action_cost_vector(request)?
        || !approximately_equal(
            scoped_resource_action.expected_regret_reduction,
            voc.estimated_regret_reduction,
        )
        || !approximately_equal(
            scoped_resource_action.numerical_standard_error,
            voc.estimated_regret_reduction_standard_error,
        )
    {
        return Err(NasAuthorizationError::new(
            "resource_actions",
            "resource action cost or benefit estimate does not match recomputed portfolio VoC",
        ));
    }

    let frontier = plan_resource_action_frontier(
        request.resource_registry,
        request.resource_budget,
        request.resource_actions,
    )
    .map_err(|error| NasAuthorizationError::new("resource_frontier", error.to_string()))?;
    if frontier.resource_registry_sha256 != scope.resource_registry_sha256
        || !frontier.frontier_action_ids.iter().any(|id| id == &scope.action_id)
    {
        return Err(NasAuthorizationError::new(
            "resource_frontier",
            "action is absent from the recomputed non-dominated resource frontier",
        ));
    }
    let assessment = frontier
        .assessments
        .iter()
        .find(|item| item.action_id == scope.action_id)
        .ok_or_else(|| NasAuthorizationError::new("resource_frontier", "missing action assessment"))?;
    if !assessment.resource_feasible || !assessment.benefit_eligible {
        return Err(NasAuthorizationError::new(
            "resource_frontier",
            "action is not both resource-feasible and benefit-eligible",
        ));
    }

    verify_anytime_confidence_ledger(request.confidence_ledger)
        .map_err(|error| NasAuthorizationError::new("confidence_ledger", error.to_string()))?;
    if request.confidence_ledger.manifest.manifest_sha256
        != scope.confidence_family_manifest_sha256
    {
        return Err(NasAuthorizationError::new(
            "confidence_family_manifest_sha256",
            "confidence ledger family manifest differs from the exact manifest committed in the action scope",
        ));
    }
    if request.confidence_ledger.manifest.family_id != scope.run_id {
        return Err(NasAuthorizationError::new(
            "confidence_family",
            "confidence family is not bound to this run ID",
        ));
    }
    let expected_stream_id = measurement_confidence_stream_id(scope)?;
    let stream = request
        .confidence_ledger
        .streams
        .iter()
        .find(|entry| entry.stream_id == expected_stream_id)
        .ok_or_else(|| {
            NasAuthorizationError::new(
                "confidence_stream",
                "no preregistered stream exists for this exact action scope",
            )
        })?;
    let latest = stream.looks.last().ok_or_else(|| {
        NasAuthorizationError::new(
            "confidence_stream",
            "the exact action stream has no committed evidence look",
        )
    })?;
    if latest.receipt.decision != AnytimeDecision::AtOrAboveThreshold
        || latest.receipt.lower_bound <= 0.0
    {
        return Err(NasAuthorizationError::new(
            "confidence_decision",
            "confidence evidence is inconclusive or does not establish positive decision value",
        ));
    }
    if !approximately_equal(latest.receipt.estimate, voc.estimated_regret_reduction) {
        return Err(NasAuthorizationError::new(
            "confidence_estimate",
            "replicate-level confidence estimate does not match recomputed portfolio regret reduction",
        ));
    }

    let portfolio_decision_sha256 = compute_portfolio_decision_binding_sha256(&voc)?;
    let resource_frontier_sha256 = compute_resource_frontier_binding_sha256(&frontier)?;
    let mut receipt = NasActionPreflightReceipt {
        schema: "symthaea.nas.action-preflight".to_owned(),
        version: "0.1.0".to_owned(),
        action_kind: NasActionKind::AcquireMeasurement,
        disposition: "PREFLIGHT_ELIGIBLE_NO_RESERVATION_NO_EXECUTION".to_owned(),
        run_id: scope.run_id.clone(),
        target_commit_id: scope.target_commit_id.clone(),
        action_id: scope.action_id.clone(),
        candidate_id: scope.candidate_id.clone(),
        scope_sha256,
        confidence_family_manifest_sha256: Some(
            request.confidence_ledger.manifest.manifest_sha256.clone(),
        ),
        confidence_ledger_sha256: Some(request.confidence_ledger.ledger_sha256.clone()),
        confidence_receipt_sha256: Some(latest.receipt.receipt_sha256.clone()),
        portfolio_decision_sha256: Some(portfolio_decision_sha256),
        objective_registry_sha256: Some(scope.objective_registry_sha256.clone()),
        hardware_profile_sha256: Some(scope.hardware_profile_sha256.clone()),
        resource_registry_sha256: Some(scope.resource_registry_sha256.clone()),
        resource_budget_sha256: Some(scope.resource_budget_sha256.clone()),
        resource_cost_sha256: Some(scope.resource_cost_sha256.clone()),
        resource_frontier_sha256: Some(resource_frontier_sha256),
        pruning_risk_genesis_sha256: None,
        pruning_risk_receipt_sha256: None,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = compute_preflight_receipt_sha256(&receipt)?;
    Ok(receipt)
}

/// Verify a pruning preflight against the exact scope-derived run-wide risk event.
///
/// The event ID must have been computed from the full pruning scope before the audit
/// was recorded. This makes the target commit, candidate configuration, frontier and
/// pruned population part of the run-wide ledger's event digest chain.
pub fn verify_pruning_preflight(
    request: &PrunePreflightRequest<'_>,
) -> Result<NasActionPreflightReceipt, NasAuthorizationError> {
    validate_prune_scope(request.scope)?;
    request
        .risk_ledger
        .verify_integrity()
        .map_err(|error| NasAuthorizationError::new("risk_ledger", error.to_string()))?;
    let scope_sha256 = compute_prune_scope_sha256(request.scope)?;
    let expected_event_id = format!("{PRUNE_EVENT_PREFIX}{scope_sha256}");
    let summary = request.risk_ledger.summary();
    if summary.run_id != request.scope.run_id || !summary.pruning_enabled {
        return Err(NasAuthorizationError::new(
            "risk_ledger",
            "risk ledger belongs to another run or has permanently disabled pruning",
        ));
    }
    let event = request
        .risk_ledger
        .events()
        .last()
        .filter(|event| event.event_id == expected_event_id)
        .ok_or_else(|| {
            NasAuthorizationError::new(
                "risk_event",
                "latest run-wide risk event is not bound to this exact pruning scope",
            )
        })?;
    if !event.verify_digest()
        || !event.pruning_remains_enabled
        || event.run_id != request.scope.run_id
        || event.selected_frontier_sha256 != request.scope.selected_frontier_sha256
        || event.pruned_population_sha256 != request.scope.pruned_population_sha256
    {
        return Err(NasAuthorizationError::new(
            "risk_event",
            "pruning event does not match the scope or does not leave pruning enabled",
        ));
    }
    let mut receipt = NasActionPreflightReceipt {
        schema: "symthaea.nas.action-preflight".to_owned(),
        version: "0.1.0".to_owned(),
        action_kind: NasActionKind::PruneCandidate,
        disposition: "PREFLIGHT_ELIGIBLE_NO_RESERVATION_NO_EXECUTION".to_owned(),
        run_id: request.scope.run_id.clone(),
        target_commit_id: request.scope.target_commit_id.clone(),
        action_id: request.scope.action_id.clone(),
        candidate_id: request.scope.candidate_id.clone(),
        scope_sha256,
        confidence_family_manifest_sha256: None,
        confidence_ledger_sha256: None,
        confidence_receipt_sha256: None,
        portfolio_decision_sha256: None,
        objective_registry_sha256: None,
        hardware_profile_sha256: None,
        resource_registry_sha256: None,
        resource_budget_sha256: None,
        resource_cost_sha256: None,
        resource_frontier_sha256: None,
        pruning_risk_genesis_sha256: Some(summary.genesis_sha256),
        pruning_risk_receipt_sha256: Some(event.receipt_sha256.clone()),
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = compute_preflight_receipt_sha256(&receipt)?;
    Ok(receipt)
}

/// Promotion is deliberately unavailable until #9388 provides an unforgeable,
/// receipt-replayed task-calibration capability. A boolean or caller-supplied digest
/// is never accepted as a substitute.
pub fn verify_promotion_preflight(
    _run_id: &str,
    _target_commit_id: &str,
    _action_id: &str,
    _candidate_id: &str,
    _candidate_config_sha256: &str,
) -> Result<NasActionPreflightReceipt, NasAuthorizationError> {
    Err(NasAuthorizationError::new(
        "qualification_capability",
        "candidate promotion is blocked until #9388 supplies a privately issued, exact-scope VerifiedVoCQualification capability",
    ))
}

pub fn compute_prune_scope_sha256(
    scope: &NasPruneActionScope,
) -> Result<String, NasAuthorizationError> {
    validate_prune_scope(scope)?;
    let bytes = serde_json::to_vec(scope)
        .map_err(|error| NasAuthorizationError::new("scope", error.to_string()))?;
    Ok(digest_bytes(SCOPE_DOMAIN, &bytes))
}

/// The run-wide pruning event identifier derived from the exact prune scope.
pub fn pruning_scope_event_id(
    scope: &NasPruneActionScope,
) -> Result<String, NasAuthorizationError> {
    Ok(format!(
        "{PRUNE_EVENT_PREFIX}{}",
        compute_prune_scope_sha256(scope)?
    ))
}

fn action_cost_vector(
    request: &MeasurementPreflightRequest<'_>,
) -> Result<&crate::multi_resource_budget::ResourceCostVector, NasAuthorizationError> {
    request
        .resource_actions
        .iter()
        .find(|action| action.action_id == request.scope.action_id)
        .map(|action| &action.resource_cost)
        .ok_or_else(|| NasAuthorizationError::new("resource_cost", "matching action cost is absent"))
}

fn validate_measurement_scope(scope: &NasMeasurementActionScope) -> Result<(), NasAuthorizationError> {
    if scope.run_id.trim().is_empty()
        || scope.action_id.trim().is_empty()
        || scope.candidate_id.trim().is_empty()
        || scope.objective_id.trim().is_empty()
    {
        return Err(NasAuthorizationError::new(
            "identity",
            "run, action, candidate and objective IDs must be non-empty",
        ));
    }
    if !is_git_object_id(&scope.target_commit_id) {
        return Err(NasAuthorizationError::new(
            "target_commit_id",
            "must be a lowercase 40- or 64-character Git object ID",
        ));
    }
    for (field, digest) in [
        ("candidate_config_sha256", &scope.candidate_config_sha256),
        ("objective_registry_sha256", &scope.objective_registry_sha256),
        ("objective_order_sha256", &scope.objective_order_sha256),
        ("posterior_sha256", &scope.posterior_sha256),
        ("portfolio_family_sha256", &scope.portfolio_family_sha256),
        ("qmc_sampling_design_sha256", &scope.qmc_sampling_design_sha256),
        ("selection_replicates_sha256", &scope.selection_replicates_sha256),
        ("validation_replicates_sha256", &scope.validation_replicates_sha256),
        ("hardware_profile_sha256", &scope.hardware_profile_sha256),
        ("measurement_protocol_sha256", &scope.measurement_protocol_sha256),
        ("instrument_identity_sha256", &scope.instrument_identity_sha256),
        ("observation_likelihood_model_sha256", &scope.observation_likelihood_model_sha256),
        ("confidence_family_manifest_sha256", &scope.confidence_family_manifest_sha256),
        ("resource_registry_sha256", &scope.resource_registry_sha256),
        ("resource_budget_sha256", &scope.resource_budget_sha256),
        ("resource_cost_sha256", &scope.resource_cost_sha256),
    ] {
        if !is_sha256(digest) {
            return Err(NasAuthorizationError::new(
                field,
                "must be a lowercase hexadecimal SHA-256 digest",
            ));
        }
    }
    if scope.objective_ids.is_empty()
        || scope.objective_ids.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(NasAuthorizationError::new(
            "objective_ids",
            "objective IDs must be non-empty, unique, and strictly sorted",
        ));
    }
    if scope.selection_replicates_sha256 == scope.validation_replicates_sha256 {
        return Err(NasAuthorizationError::new(
            "replicate_split",
            "selection and validation replicate sets must have distinct digests",
        ));
    }
    Ok(())
}

fn validate_prune_scope(scope: &NasPruneActionScope) -> Result<(), NasAuthorizationError> {
    if scope.run_id.trim().is_empty()
        || scope.action_id.trim().is_empty()
        || scope.candidate_id.trim().is_empty()
    {
        return Err(NasAuthorizationError::new(
            "identity",
            "run, action, and candidate IDs must be non-empty",
        ));
    }
    if !is_git_object_id(&scope.target_commit_id) {
        return Err(NasAuthorizationError::new(
            "target_commit_id",
            "must be a lowercase 40- or 64-character Git object ID",
        ));
    }
    for (field, digest) in [
        ("candidate_config_sha256", &scope.candidate_config_sha256),
        ("selected_frontier_sha256", &scope.selected_frontier_sha256),
        ("pruned_population_sha256", &scope.pruned_population_sha256),
    ] {
        if !is_sha256(digest) {
            return Err(NasAuthorizationError::new(
                field,
                "must be a lowercase hexadecimal SHA-256 digest",
            ));
        }
    }
    Ok(())
}

fn validate_portfolio_decision_scope(
    voc: &PortfolioParetoVoC,
    scope: &NasMeasurementActionScope,
) -> Result<(), NasAuthorizationError> {
    let decision = &voc.current_decision;
    for (field, actual, expected) in [
        ("posterior_sha256", decision.posterior_sha256.as_str(), scope.posterior_sha256.as_str()),
        ("portfolio_family_sha256", decision.portfolio_family_sha256.as_str(), scope.portfolio_family_sha256.as_str()),
        ("qmc_sampling_design_sha256", decision.qmc_sampling_design_sha256.as_str(), scope.qmc_sampling_design_sha256.as_str()),
        ("selection_replicates_sha256", decision.selection_replicates_sha256.as_str(), scope.selection_replicates_sha256.as_str()),
        ("validation_replicates_sha256", decision.validation_replicates_sha256.as_str(), scope.validation_replicates_sha256.as_str()),
    ] {
        if actual != expected {
            return Err(NasAuthorizationError::new(
                field,
                "recomputed portfolio decision identity does not match the declared action scope",
            ));
        }
    }
    if voc.objective_registry_sha256 != scope.objective_registry_sha256
        || voc.observation_likelihood_model_sha256 != scope.observation_likelihood_model_sha256
        || voc.action_id != scope.action_id
    {
        return Err(NasAuthorizationError::new(
            "portfolio_voc",
            "recomputed value-of-computation action identity differs from the declared scope",
        ));
    }
    Ok(())
}

fn compute_portfolio_decision_binding_sha256(
    voc: &PortfolioParetoVoC,
) -> Result<String, NasAuthorizationError> {
    let decision = &voc.current_decision;
    let binding = (
        voc.action_id.as_str(),
        voc.expected_posterior_regret,
        voc.expected_posterior_regret_standard_error,
        voc.estimated_regret_reduction,
        voc.estimated_regret_reduction_standard_error,
        voc.estimated_regret_reduction_interval_95,
        voc.regret_reduction_per_cost_unit,
        voc.incremental_cost_units,
        voc.resource_unit.as_str(),
        voc.observation_likelihood_model_sha256.as_str(),
        voc.objective_registry_sha256.as_str(),
        decision.selected_portfolio_id.as_str(),
        &decision.selected_candidate_ids,
        decision.oracle_expected_hypervolume,
        decision.selected_portfolio_expected_hypervolume,
        decision.portfolio_class_regret,
        decision.portfolio_class_regret_standard_error,
        &decision.validation_regret_replicates,
        &decision.selection_replicate_indices,
        &decision.validation_replicate_indices,
        decision.posterior_particle_count,
        decision.posterior_sha256.as_str(),
        decision.portfolio_family_sha256.as_str(),
        decision.qmc_sampling_design_sha256.as_str(),
        decision.selection_replicates_sha256.as_str(),
        decision.validation_replicates_sha256.as_str(),
    );
    digest_serde(
        b"symthaea.nas.portfolio-voc-decision-binding.v0.1",
        &binding,
    )
}

fn compute_resource_frontier_binding_sha256(
    frontier: &crate::multi_resource_budget::ResourceActionFrontier,
) -> Result<String, NasAuthorizationError> {
    let assessments: Vec<_> = frontier
        .assessments
        .iter()
        .map(|assessment| (
            assessment.action_id.as_str(),
            assessment.conservative_regret_reduction,
            assessment.resource_feasible,
            assessment.benefit_eligible,
            &assessment.blocked_resources,
        ))
        .collect();
    let binding = (
        frontier.resource_registry_sha256.as_str(),
        &assessments,
        &frontier.frontier_action_ids,
        &frontier.blocked_action_ids,
        &frontier.resource_headroom,
        frontier.lower_bound_z,
    );
    digest_serde(
        b"symthaea.nas.resource-frontier-binding.v0.1",
        &binding,
    )
}

fn compute_instrument_identity_sha256(
    profile: &crate::hardware_objectives::HardwareProfileManifest,
) -> Result<String, NasAuthorizationError> {
    let identity = (
        &profile.profile_sha256,
        &profile.instrument_model,
        &profile.instrument_id,
        profile.sampling_frequency_hz,
    );
    digest_serde(INSTRUMENT_DOMAIN, &identity)
}

fn compute_objective_order_sha256(ids: &[String]) -> Result<String, NasAuthorizationError> {
    digest_serde(OBJECTIVE_ORDER_DOMAIN, ids)
}

/// Check the integrity digest of a preflight receipt. This does not re-verify its source evidence
/// or turn a preflight receipt into an execution capability.
pub fn verify_action_preflight_receipt_digest(
    receipt: &NasActionPreflightReceipt,
) -> Result<(), NasAuthorizationError> {
    if !is_sha256(&receipt.receipt_sha256)
        || compute_preflight_receipt_sha256(receipt)? != receipt.receipt_sha256
    {
        return Err(NasAuthorizationError::new(
            "receipt_sha256",
            "preflight receipt digest does not match its canonical fields",
        ));
    }
    Ok(())
}

fn compute_preflight_receipt_sha256(
    receipt: &NasActionPreflightReceipt,
) -> Result<String, NasAuthorizationError> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    digest_serde(RECEIPT_DOMAIN, &canonical)
}

fn digest_serde<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, NasAuthorizationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| NasAuthorizationError::new("serialization", error.to_string()))?;
    Ok(digest_bytes(domain, &bytes))
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    encode_sha256(hasher.finalize().as_slice())
}

fn approximately_equal(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= TOLERANCE * left.abs().max(right.abs()).max(1.0)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_git_object_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn encode_sha256(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruning_audit::{prepare_pruning_audit_sample, AuditScore};
    use crate::anytime_confidence_ledger::{
        append_anytime_confidence_look, create_anytime_confidence_family_manifest,
        create_anytime_confidence_ledger, AnytimeStreamDeclaration,
    };
    use crate::hardware_objectives::{
        compute_objective_registry_sha256, HardwareProfileManifest, ObjectiveDefinition,
        ObjectiveDirection, ObjectiveNormalization,
    };
    use crate::multi_resource_budget::{
        compute_resource_registry_sha256, ResourceBudgetAction, ResourceBudgetLimit,
        ResourceClass, ResourceCostVector, ResourceDefinition, ResourceQuantity,
    };
    use crate::pareto_hypervolume_qmc::HypervolumeQmcConfig;
    use crate::pareto_portfolio_voc::{
        compute_portfolio_action_likelihood_sha256, CandidatePortfolio,
        ParetoPortfolioParticle, ParetoPortfolioPosterior, PortfolioActionOutcome,
        PortfolioMeasurementAction,
    };
    use crate::run_wide_pruning_risk::RunWidePruningRiskConfig;

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn prune_scope() -> NasPruneActionScope {
        NasPruneActionScope {
            run_id: "run-authz-001".into(),
            target_commit_id: "a".repeat(40),
            action_id: "prune-keep-001".into(),
            candidate_id: "candidate-prune-001".into(),
            candidate_config_sha256: sha('b'),
            selected_frontier_sha256: sha('c'),
            pruned_population_sha256: sha('d'),
        }
    }

    fn risk_config() -> RunWidePruningRiskConfig {
        RunWidePruningRiskConfig {
            run_id: "run-authz-001".into(),
            run_confidence_failure_budget: 0.05,
            max_audit_events: 1,
            max_false_prune_rate_per_event: 0.25,
            min_audit_samples: 2,
            fidelity_ids: vec!["epoch-30".into()],
        }
    }

    fn selected_scores() -> Vec<AuditScore> {
        vec![AuditScore {
            candidate_id: "frontier-A".into(),
            fidelity_index: 0,
            score: 0.9,
        }]
    }

    fn selected_frontier_digest(scores: &[AuditScore]) -> String {
        let mut rows: Vec<&AuditScore> = scores.iter().collect();
        rows.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.nas.pruning-selected-frontier.v0.1");
        for row in rows {
            hasher.update((row.candidate_id.len() as u64).to_be_bytes());
            hasher.update(row.candidate_id.as_bytes());
            hasher.update((row.fidelity_index as u64).to_be_bytes());
            hasher.update(row.score.to_bits().to_be_bytes());
        }
        encode_sha256(hasher.finalize().as_slice())
    }

    fn make_valid_prune_evidence(
        scope: &mut NasPruneActionScope,
    ) -> RunWidePruningRiskLedger {
        let population = vec!["discard-A".to_owned(), "discard-B".to_owned()];
        let sample = prepare_pruning_audit_sample(&population, 2, 77).unwrap();
        let selected = selected_scores();
        scope.selected_frontier_sha256 = selected_frontier_digest(&selected);
        scope.pruned_population_sha256 = sample.population_sha256().to_owned();
        let event_id = pruning_scope_event_id(scope).unwrap();
        let audited = sample
            .sampled_candidate_ids()
            .iter()
            .map(|candidate_id| AuditScore {
                candidate_id: candidate_id.clone(),
                fidelity_index: 0,
                score: 0.1,
            })
            .collect::<Vec<_>>();
        let mut ledger = RunWidePruningRiskLedger::new(risk_config()).unwrap();
        ledger
            .record_event(&event_id, "epoch-30", &selected, &sample, &audited)
            .unwrap();
        ledger
    }

    #[test]
    fn scope_digest_changes_when_target_commit_or_candidate_changes() {
        let original = prune_scope();
        let digest = compute_prune_scope_sha256(&original).unwrap();
        let mut changed_commit = original.clone();
        changed_commit.target_commit_id = "e".repeat(40);
        assert_ne!(digest, compute_prune_scope_sha256(&changed_commit).unwrap());
        let mut changed_candidate = original.clone();
        changed_candidate.candidate_config_sha256 = sha('f');
        assert_ne!(digest, compute_prune_scope_sha256(&changed_candidate).unwrap());
    }

    #[test]
    fn prune_event_id_binds_the_complete_scope() {
        let original = prune_scope();
        let mut changed = original.clone();
        changed.pruned_population_sha256 = sha('e');
        assert_ne!(
            pruning_scope_event_id(&original).unwrap(),
            pruning_scope_event_id(&changed).unwrap()
        );
    }

    #[test]
    fn verifies_pruning_preflight_only_for_the_exact_risk_event() {
        let mut scope = prune_scope();
        let ledger = make_valid_prune_evidence(&mut scope);
        ledger.verify_integrity().unwrap();
        let receipt = verify_pruning_preflight(&PrunePreflightRequest {
            scope: &scope,
            risk_ledger: &ledger,
        })
        .unwrap();
        assert_eq!(receipt.action_kind, NasActionKind::PruneCandidate);
        assert_eq!(
            receipt.disposition,
            "PREFLIGHT_ELIGIBLE_NO_RESERVATION_NO_EXECUTION"
        );
        assert!(receipt.pruning_risk_receipt_sha256.is_some());
        verify_action_preflight_receipt_digest(&receipt).unwrap();
        let mut tampered = receipt.clone();
        tampered.candidate_id = "other-candidate".into();
        assert!(verify_action_preflight_receipt_digest(&tampered).is_err());
        assert_eq!(
            receipt.receipt_sha256,
            compute_preflight_receipt_sha256(&receipt).unwrap()
        );
    }

    #[test]
    fn rejects_pruning_replay_against_another_commit_or_candidate() {
        let mut scope = prune_scope();
        let ledger = make_valid_prune_evidence(&mut scope);
        let mut changed = scope.clone();
        changed.target_commit_id = "e".repeat(40);
        assert!(verify_pruning_preflight(&PrunePreflightRequest {
            scope: &changed,
            risk_ledger: &ledger,
        })
        .is_err());

        let mut changed_candidate = scope.clone();
        changed_candidate.candidate_id = "other-candidate".into();
        assert!(verify_pruning_preflight(&PrunePreflightRequest {
            scope: &changed_candidate,
            risk_ledger: &ledger,
        })
        .is_err());
    }

    #[test]
    fn rejects_pruning_when_the_risk_ledger_has_disabled_it() {
        let mut scope = prune_scope();
        let population = vec!["discard-A".to_owned(), "discard-B".to_owned()];
        let sample = prepare_pruning_audit_sample(&population, 2, 77).unwrap();
        let selected = selected_scores();
        scope.selected_frontier_sha256 = selected_frontier_digest(&selected);
        scope.pruned_population_sha256 = sample.population_sha256().to_owned();
        let event_id = pruning_scope_event_id(&scope).unwrap();
        let audited = sample
            .sampled_candidate_ids()
            .iter()
            .map(|candidate_id| AuditScore {
                candidate_id: candidate_id.clone(),
                fidelity_index: 0,
                score: 0.95,
            })
            .collect::<Vec<_>>();
        let mut config = risk_config();
        config.max_false_prune_rate_per_event = 0.0;
        let mut ledger = RunWidePruningRiskLedger::new(config).unwrap();
        let event = ledger
            .record_event(&event_id, "epoch-30", &selected, &sample, &audited)
            .unwrap();
        assert!(!event.pruning_remains_enabled);
        assert!(verify_pruning_preflight(&PrunePreflightRequest {
            scope: &scope,
            risk_ledger: &ledger,
        })
        .is_err());
    }

    #[test]
    fn rejects_a_prune_event_with_the_wrong_run_id() {
        let mut scope = prune_scope();
        let mut ledger = make_valid_prune_evidence(&mut scope);
        // The event and ledger's internals are private; a changed scope is the public replay path.
        let mut changed = scope.clone();
        changed.run_id = "another-run".into();
        assert!(verify_pruning_preflight(&PrunePreflightRequest {
            scope: &changed,
            risk_ledger: &ledger,
        })
        .is_err());
        ledger.verify_integrity().unwrap();
    }

    #[test]
    fn promotion_is_blocked_without_the_private_qualification_capability() {
        let result = verify_promotion_preflight(
            "run-authz-001",
            &"a".repeat(40),
            "promote-A",
            "candidate-A",
            &sha('a'),
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("#9388"));
    }

    #[test]
    fn measurement_stream_id_is_bound_to_the_entire_action_scope() {
        let mut scope = NasMeasurementActionScope {
            run_id: "run-1".into(),
            target_commit_id: "a".repeat(40),
            action_id: "measure-a".into(),
            candidate_id: "candidate-a".into(),
            candidate_config_sha256: sha('a'),
            objective_id: "latency".into(),
            objective_ids: vec!["accuracy".into(), "latency".into()],
            objective_registry_sha256: sha('b'),
            objective_order_sha256: sha('c'),
            posterior_sha256: sha('d'),
            portfolio_family_sha256: sha('e'),
            qmc_sampling_design_sha256: sha('f'),
            selection_replicates_sha256: sha('1'),
            validation_replicates_sha256: sha('2'),
            hardware_profile_sha256: sha('3'),
            measurement_protocol_sha256: sha('4'),
            instrument_identity_sha256: sha('5'),
            observation_likelihood_model_sha256: sha('6'),
            resource_registry_sha256: sha('7'),
            resource_budget_sha256: sha('8'),
            resource_cost_sha256: sha('9'),
        };
        let first = measurement_confidence_stream_id(&scope).unwrap();
        scope.measurement_protocol_sha256 = sha('a');
        let second = measurement_confidence_stream_id(&scope).unwrap();
        assert_ne!(first, second);
        assert!(first.starts_with(MEASUREMENT_STREAM_PREFIX));
    }

    #[test]
    fn instrument_identity_binds_profile_instrument_and_sample_rate() {
        let mut profile = crate::hardware_objectives::HardwareProfileManifest {
            profile_id: "nrf5340-ppk2".into(),
            device_model: "nRF5340".into(),
            board_id: "board-01".into(),
            board_revision: "rev-a".into(),
            firmware_sha256: sha('a'),
            runtime_sha256: sha('b'),
            compiler_toolchain_sha256: sha('c'),
            quantization_spec_sha256: sha('d'),
            measurement_protocol_sha256: sha('e'),
            instrument_model: "Power Profiler Kit II".into(),
            instrument_id: "ppk2-001".into(),
            sampling_frequency_hz: 100_000,
            profile_sha256: String::new(),
        };
        profile.profile_sha256 =
            compute_hardware_profile_sha256(&profile).unwrap();
        let first = compute_instrument_identity_sha256(&profile).unwrap();
        profile.instrument_id = "ppk2-002".into();
        let second = compute_instrument_identity_sha256(&profile).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn objective_coordinate_order_digest_is_stable_and_order_sensitive() {
        let ids = vec!["accuracy".to_owned(), "latency".to_owned(), "energy".to_owned()];
        let first = compute_objective_order_sha256(&ids).unwrap();
        let mut reordered = ids.clone();
        reordered.swap(0, 1);
        let second = compute_objective_order_sha256(&reordered).unwrap();
        assert_ne!(first, second);
        assert_eq!(first, compute_objective_order_sha256(&ids).unwrap());
    }

    #[test]
    fn measurement_preflight_recomputes_and_binds_the_complete_evidence_chain() {
        let mut profile = HardwareProfileManifest {
            profile_id: "nrf5340-ppk2".into(),
            device_model: "Nordic nRF5340".into(),
            board_id: "board-01".into(),
            board_revision: "rev-a".into(),
            firmware_sha256: sha('1'),
            runtime_sha256: sha('2'),
            compiler_toolchain_sha256: sha('3'),
            quantization_spec_sha256: sha('4'),
            measurement_protocol_sha256: sha('5'),
            instrument_model: "Power Profiler Kit II".into(),
            instrument_id: "ppk2-001".into(),
            sampling_frequency_hz: 100_000,
            profile_sha256: String::new(),
        };
        profile.profile_sha256 = compute_hardware_profile_sha256(&profile).unwrap();

        let mut objective_registry = ObjectiveRegistry {
            schema: "symthaea.nas.objective-registry".into(),
            version: "0.1.0".into(),
            benchmark_id: "nats-bench-mcu-v2-preflight-fixture".into(),
            objectives: vec![
                ObjectiveDefinition {
                    objective_id: "accuracy".into(),
                    value_unit: "fraction".into(),
                    direction: ObjectiveDirection::Maximize,
                    source: ObjectiveSource::TrainedEvaluation,
                    normalization: ObjectiveNormalization {
                        raw_min: 0.0,
                        raw_max: 1.0,
                    },
                    measurement_protocol_sha256: sha('6'),
                    hardware_profile_sha256: None,
                    measurement_cost_unit: "objective-query".into(),
                    measurement_cost_model_sha256: sha('7'),
                },
                ObjectiveDefinition {
                    objective_id: "latency".into(),
                    value_unit: "milliseconds".into(),
                    direction: ObjectiveDirection::Minimize,
                    source: ObjectiveSource::PhysicalDeviceMeasurement,
                    normalization: ObjectiveNormalization {
                        raw_min: 0.0,
                        raw_max: 100.0,
                    },
                    measurement_protocol_sha256: sha('5'),
                    hardware_profile_sha256: Some(profile.profile_sha256.clone()),
                    measurement_cost_unit: "objective-query".into(),
                    measurement_cost_model_sha256: sha('8'),
                },
            ],
            hardware_profiles: vec![profile.clone()],
            registry_sha256: String::new(),
        };
        objective_registry.registry_sha256 =
            compute_objective_registry_sha256(&objective_registry).unwrap();

        let posterior = ParetoPortfolioPosterior {
            candidate_ids: vec!["A".into(), "B".into(), "C".into()],
            objective_ids: vec!["accuracy".into(), "latency".into()],
            objective_registry_sha256: objective_registry.registry_sha256.clone(),
            candidate_config_sha256: vec![sha('a'), sha('b'), sha('c')],
            candidate_feasible: vec![true, true, true],
            particles: vec![
                ParetoPortfolioParticle {
                    particle_id: "p1".into(),
                    probability: 0.5,
                    objective_values: vec![
                        Some(vec![0.95, 0.95]),
                        Some(vec![0.2, 0.2]),
                        Some(vec![0.4, 0.4]),
                    ],
                },
                ParetoPortfolioParticle {
                    particle_id: "p2".into(),
                    probability: 0.5,
                    objective_values: vec![
                        Some(vec![0.2, 0.2]),
                        Some(vec![0.95, 0.95]),
                        Some(vec![0.4, 0.4]),
                    ],
                },
            ],
        };
        let portfolios = vec![
            CandidatePortfolio {
                portfolio_id: "A-only".into(),
                candidate_indices: vec![0],
            },
            CandidatePortfolio {
                portfolio_id: "B-only".into(),
                candidate_indices: vec![1],
            },
        ];
        let portfolio_config = ParetoPortfolioConfig {
            target_k: 1,
            minimum_feasible_utility: vec![0.0, 0.0],
        };
        let qmc_design = HypervolumeQmcDesign::new(
            &[0.0, 0.0],
            &HypervolumeQmcConfig {
                samples_per_replicate: 4096,
                scramble_replicates: 128,
                seed: 19,
            },
        )
        .unwrap();
        let mut action = PortfolioMeasurementAction {
            action_id: "measure-A-latency".into(),
            candidate_id: "A".into(),
            candidate_config_sha256: sha('a'),
            objective_registry_sha256: objective_registry.registry_sha256.clone(),
            objective_id: "latency".into(),
            hardware_profile_sha256: Some(profile.profile_sha256.clone()),
            measurement_protocol_sha256: sha('5'),
            resource_unit: "objective-query".into(),
            incremental_cost_units: 2,
            observation_likelihood_model_sha256: String::new(),
            outcomes: vec![
                PortfolioActionOutcome {
                    outcome_id: "latency-low".into(),
                    likelihood_by_particle: vec![0.99, 0.01],
                },
                PortfolioActionOutcome {
                    outcome_id: "latency-high".into(),
                    likelihood_by_particle: vec![0.01, 0.99],
                },
            ],
        };
        action.observation_likelihood_model_sha256 =
            compute_portfolio_action_likelihood_sha256(&action).unwrap();
        let voc = evaluate_portfolio_measurement_voc(
            &objective_registry,
            &posterior,
            &portfolios,
            &portfolio_config,
            &qmc_design,
            &action,
        )
        .unwrap();
        assert!(voc.estimated_regret_reduction > 0.05);

        let mut resource_registry = ResourceRegistry {
            schema: "symthaea.nas.resource-registry".into(),
            version: "0.1.0".into(),
            resources: vec![
                ResourceDefinition {
                    resource_id: "device_measurement_queries".into(),
                    unit: "objective-query".into(),
                    class: ResourceClass::Custom,
                    cost_model_sha256: sha('8'),
                },
                ResourceDefinition {
                    resource_id: "instrument_time".into(),
                    unit: "milliseconds".into(),
                    class: ResourceClass::InstrumentTime,
                    cost_model_sha256: sha('9'),
                },
            ],
            registry_sha256: String::new(),
        };
        resource_registry.registry_sha256 =
            compute_resource_registry_sha256(&resource_registry).unwrap();
        let cost_vector = ResourceCostVector {
            action_id: action.action_id.clone(),
            resource_registry_sha256: resource_registry.registry_sha256.clone(),
            resources: vec![
                ResourceQuantity {
                    resource_id: "device_measurement_queries".into(),
                    unit: "objective-query".into(),
                    units: action.incremental_cost_units,
                    cost_model_sha256: sha('8'),
                },
                ResourceQuantity {
                    resource_id: "instrument_time".into(),
                    unit: "milliseconds".into(),
                    units: 1500,
                    cost_model_sha256: sha('9'),
                },
            ],
        };
        let resource_budget = ResourceBudget {
            resource_registry_sha256: resource_registry.registry_sha256.clone(),
            limits: vec![
                ResourceBudgetLimit {
                    resource_id: "device_measurement_queries".into(),
                    unit: "objective-query".into(),
                    total_units: 100,
                    already_spent_units: 10,
                    protected_reserve_units: 10,
                },
                ResourceBudgetLimit {
                    resource_id: "instrument_time".into(),
                    unit: "milliseconds".into(),
                    total_units: 100_000,
                    already_spent_units: 1_000,
                    protected_reserve_units: 5_000,
                },
            ],
        };
        let resource_action = ResourceBudgetAction {
            action_id: action.action_id.clone(),
            expected_regret_reduction: voc.estimated_regret_reduction,
            numerical_standard_error: voc.estimated_regret_reduction_standard_error,
            resource_cost: cost_vector.clone(),
        };
        let resource_actions = vec![resource_action];

        let mut scope = NasMeasurementActionScope {
            run_id: "run-authz-measurement-001".into(),
            target_commit_id: "a".repeat(40),
            action_id: action.action_id.clone(),
            candidate_id: action.candidate_id.clone(),
            candidate_config_sha256: action.candidate_config_sha256.clone(),
            objective_id: action.objective_id.clone(),
            objective_ids: vec!["accuracy".into(), "latency".into()],
            objective_registry_sha256: objective_registry.registry_sha256.clone(),
            objective_order_sha256: compute_objective_order_sha256(&[
                "accuracy".into(),
                "latency".into(),
            ])
            .unwrap(),
            posterior_sha256: voc.current_decision.posterior_sha256.clone(),
            portfolio_family_sha256: voc.current_decision.portfolio_family_sha256.clone(),
            qmc_sampling_design_sha256: voc.current_decision.qmc_sampling_design_sha256.clone(),
            selection_replicates_sha256: voc.current_decision.selection_replicates_sha256.clone(),
            validation_replicates_sha256: voc.current_decision.validation_replicates_sha256.clone(),
            hardware_profile_sha256: profile.profile_sha256.clone(),
            measurement_protocol_sha256: action.measurement_protocol_sha256.clone(),
            instrument_identity_sha256: compute_instrument_identity_sha256(&profile).unwrap(),
            observation_likelihood_model_sha256: action
                .observation_likelihood_model_sha256
                .clone(),
            confidence_family_manifest_sha256: String::new(),
            resource_registry_sha256: resource_registry.registry_sha256.clone(),
            resource_budget_sha256: digest_serde(
                b"symthaea.nas.resource-budget.v0.1",
                &resource_budget,
            )
            .unwrap(),
            resource_cost_sha256: digest_serde(
                b"symthaea.nas.resource-cost-vector.v0.1",
                &cost_vector,
            )
            .unwrap(),
        };

        let stream_id = measurement_confidence_stream_id(&scope).unwrap();
        let manifest = create_anytime_confidence_family_manifest(
            scope.run_id.clone(),
            0.05,
            vec![AnytimeStreamDeclaration {
                stream_id,
                range_min: -1.0,
                range_max: 1.0,
                threshold: 0.0,
            }],
        )
        .unwrap();
        scope.confidence_family_manifest_sha256 = manifest.manifest_sha256.clone();
        let mut confidence_ledger = create_anytime_confidence_ledger(manifest).unwrap();
        append_anytime_confidence_look(
            &mut confidence_ledger,
            &measurement_confidence_stream_id(&scope).unwrap(),
            &vec![voc.estimated_regret_reduction; 50_000],
        )
        .unwrap();

        let receipt = verify_measurement_preflight(&MeasurementPreflightRequest {
            scope: &scope,
            confidence_ledger: &confidence_ledger,
            objective_registry: &objective_registry,
            posterior: &posterior,
            portfolios: &portfolios,
            portfolio_config: &portfolio_config,
            qmc_design: &qmc_design,
            measurement_action: &action,
            resource_registry: &resource_registry,
            resource_budget: &resource_budget,
            resource_actions: &resource_actions,
        })
        .unwrap();
        assert_eq!(receipt.action_kind, NasActionKind::AcquireMeasurement);
        assert_eq!(
            receipt.disposition,
            "PREFLIGHT_ELIGIBLE_NO_RESERVATION_NO_EXECUTION"
        );
        verify_action_preflight_receipt_digest(&receipt).unwrap();

        let mut changed_manifest_scope = scope.clone();
        changed_manifest_scope.confidence_family_manifest_sha256 = sha('0');
        assert!(verify_measurement_preflight(&MeasurementPreflightRequest {
            scope: &changed_manifest_scope,
            confidence_ledger: &confidence_ledger,
            objective_registry: &objective_registry,
            posterior: &posterior,
            portfolios: &portfolios,
            portfolio_config: &portfolio_config,
            qmc_design: &qmc_design,
            measurement_action: &action,
            resource_registry: &resource_registry,
            resource_budget: &resource_budget,
            resource_actions: &resource_actions,
        })
        .is_err());

        scope.target_commit_id = "b".repeat(40);
        assert_ne!(
            receipt.scope_sha256,
            compute_measurement_scope_sha256(&scope).unwrap()
        );
    }

    #[test]
    fn rejects_malformed_scope_and_duplicate_replicate_split() {
        let mut invalid = prune_scope();
        invalid.target_commit_id = "not-a-git-sha".into();
        assert!(compute_prune_scope_sha256(&invalid).is_err());

        let mut measurement = NasMeasurementActionScope {
            run_id: "run-1".into(),
            target_commit_id: "a".repeat(40),
            action_id: "measure-a".into(),
            candidate_id: "a".into(),
            candidate_config_sha256: sha('a'),
            objective_id: "latency".into(),
            objective_ids: vec!["accuracy".into(), "latency".into()],
            objective_registry_sha256: sha('b'),
            objective_order_sha256: sha('c'),
            posterior_sha256: sha('d'),
            portfolio_family_sha256: sha('e'),
            qmc_sampling_design_sha256: sha('f'),
            selection_replicates_sha256: sha('1'),
            validation_replicates_sha256: sha('1'),
            hardware_profile_sha256: sha('2'),
            measurement_protocol_sha256: sha('3'),
            instrument_identity_sha256: sha('4'),
            observation_likelihood_model_sha256: sha('5'),
            confidence_family_manifest_sha256: sha('a'),
            resource_registry_sha256: sha('6'),
            resource_budget_sha256: sha('7'),
            resource_cost_sha256: sha('8'),
        };
        assert!(compute_measurement_scope_sha256(&measurement).is_err());
        measurement.validation_replicates_sha256 = sha('9');
        let digest = compute_measurement_scope_sha256(&measurement).unwrap();
        assert_eq!(digest.len(), 64);
    }
}
