//! Deterministic compilation of normalized audio edit requests into typed DAGs.
//!
//! This crate does not parse natural language, execute models, mutate audio,
//! or certify that an operator's declarations are true. It validates a frozen
//! capability registry, selects only compatible operators and independent
//! observers, checks graph/resource constraints, and returns an inspectable
//! plan. Runtime execution authorization and canonical plan hashing are
//! deliberately separate concerns.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const EDIT_PLAN_SCHEMA_V1: &str = "capability-qualified-edit-plan/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanErrorKindV1 {
    InvalidInput,
    Unsupported,
    BudgetExceeded,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanCompileErrorV1 {
    pub kind: PlanErrorKindV1,
    pub message: String,
}

impl PlanCompileErrorV1 {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            kind: PlanErrorKindV1::InvalidInput,
            message: message.into(),
        }
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self {
            kind: PlanErrorKindV1::Unsupported,
            message: message.into(),
        }
    }

    fn budget(message: impl Into<String>) -> Self {
        Self {
            kind: PlanErrorKindV1::BudgetExceeded,
            message: message.into(),
        }
    }
}

impl Display for PlanCompileErrorV1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {}", self.kind, self.message)
    }
}

impl Error for PlanCompileErrorV1 {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LocalityScopeV1 {
    SampleInterval,
    Phrase,
    Stem,
    Global,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeterminismV1 {
    Deterministic,
    Seeded,
    Nondeterministic,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactRefV1 {
    pub artifact_id: String,
    pub artifact_type: String,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageRequirementV1 {
    pub stage_id: String,
    pub input_artifact_ids: Vec<String>,
    pub output_artifact_id: String,
    pub output_artifact_type: String,
    pub dependencies: Vec<String>,
    pub required_capabilities: Vec<String>,
    pub required_preconditions: Vec<String>,
    pub required_postconditions: Vec<String>,
    pub required_observation_metric_ids: Vec<String>,
    pub maximum_affected_scope: LocalityScopeV1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanBudgetV1 {
    pub maximum_compute_units: f64,
    pub maximum_latency_ms: u64,
    pub maximum_risk_units: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectiveWeightsV1 {
    pub compute_weight: f64,
    pub latency_weight: f64,
    pub risk_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EditPlanRequestV1 {
    pub request_id: String,
    pub registry_snapshot_sha256: String,
    pub initial_artifacts: Vec<ArtifactRefV1>,
    pub stages: Vec<StageRequirementV1>,
    pub budget: PlanBudgetV1,
    pub objective_weights: ObjectiveWeightsV1,
    pub maximum_nondeterminism: DeterminismV1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OperatorProfileV1 {
    pub operator_id: String,
    pub version: String,
    pub artifact_sha256: String,
    pub input_artifact_types: Vec<String>,
    pub output_artifact_types: Vec<String>,
    pub capabilities: Vec<String>,
    pub supported_preconditions: Vec<String>,
    pub emitted_postconditions: Vec<String>,
    pub maximum_affected_scope: LocalityScopeV1,
    pub determinism: DeterminismV1,
    pub compute_units: f64,
    pub latency_ms: u64,
    /// Additive risk units used by the planner budget; not a calibrated
    /// probability and not a substitute for empirical reliability data.
    pub risk_units: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObserverProfileV1 {
    pub observer_id: String,
    pub version: String,
    pub artifact_sha256: String,
    pub metric_ids: Vec<String>,
    /// This is a registry declaration, not proof. Empirical independence
    /// qualification is a separate evidence activity.
    pub independently_qualified: bool,
    pub compute_units: f64,
    pub latency_ms: u64,
    pub risk_units: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlannedObserverV1 {
    pub metric_id: String,
    pub observer_id: String,
    pub version: String,
    pub artifact_sha256: String,
    pub compute_units: f64,
    pub latency_ms: u64,
    pub risk_units: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlannedStageV1 {
    pub stage_id: String,
    pub input_artifact_ids: Vec<String>,
    pub output_artifact_id: String,
    pub output_artifact_type: String,
    pub dependencies: Vec<String>,
    pub required_preconditions: Vec<String>,
    pub required_postconditions: Vec<String>,
    pub operator_id: String,
    pub operator_version: String,
    pub operator_artifact_sha256: String,
    pub capabilities_used: Vec<String>,
    pub affected_scope: LocalityScopeV1,
    pub determinism: DeterminismV1,
    pub observers: Vec<PlannedObserverV1>,
    pub compute_units: f64,
    pub latency_ms: u64,
    pub risk_units: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledEditPlanV1 {
    pub schema: String,
    pub request_id: String,
    pub registry_snapshot_sha256: String,
    /// Stages are emitted in deterministic topological order.
    pub stages: Vec<PlannedStageV1>,
    pub total_compute_units: f64,
    pub total_latency_ms: u64,
    pub total_risk_units: f64,
    pub objective_score: f64,
}

/// Compile a normalized request into a deterministic, capability-qualified DAG.
///
/// Operator selection is performed independently for each stage using the
/// declared weighted objective, subject to hard capability, artifact-type,
/// precondition, postcondition, locality and determinism constraints. For each
/// required metric, a separately declared independent observer must exist and
/// must not share the selected operator artifact digest.
pub fn compile_edit_plan(
    request: &EditPlanRequestV1,
    operators: &[OperatorProfileV1],
    observers: &[ObserverProfileV1],
) -> Result<CompiledEditPlanV1, PlanCompileErrorV1> {
    validate_request(request)?;
    validate_registry(operators, observers)?;

    let initial_types = initial_artifact_types(&request.initial_artifacts);
    let ordered_stage_ids = topological_order(&request.stages)?;

    let stage_by_id: BTreeMap<&str, &StageRequirementV1> = request
        .stages
        .iter()
        .map(|stage| (stage.stage_id.as_str(), stage))
        .collect();
    let output_types: BTreeMap<&str, &str> = request
        .stages
        .iter()
        .map(|stage| {
            (
                stage.output_artifact_id.as_str(),
                stage.output_artifact_type.as_str(),
            )
        })
        .collect();

    let mut selected_by_stage: BTreeMap<String, OperatorProfileV1> = BTreeMap::new();
    let mut planned_stages = Vec::with_capacity(request.stages.len());
    let mut total_compute = 0.0f64;
    let mut total_latency = 0u64;
    let mut total_risk = 0.0f64;

    for stage_id in ordered_stage_ids {
        let stage = stage_by_id
            .get(stage_id.as_str())
            .copied()
            .ok_or_else(|| PlanCompileErrorV1::invalid("validated stage disappeared"))?;

        let mut required_input_types = BTreeSet::new();
        for input_id in &stage.input_artifact_ids {
            let input_type = if let Some(initial_type) = initial_types.get(input_id.as_str()) {
                *initial_type
            } else {
                output_types
                    .get(input_id.as_str())
                    .copied()
                    .ok_or_else(|| {
                        PlanCompileErrorV1::invalid(format!(
                            "stage {} references an unresolved input artifact {}",
                            stage.stage_id, input_id
                        ))
                    })?
            };
            required_input_types.insert(input_type.to_owned());
        }

        let selected_operator = choose_operator(
            request,
            stage,
            &required_input_types,
            operators,
        )?;
        let mut stage_compute = selected_operator.compute_units;
        let mut stage_latency = selected_operator.latency_ms;
        let mut stage_risk = selected_operator.risk_units;

        let mut planned_observers = Vec::with_capacity(
            stage.required_observation_metric_ids.len(),
        );
        for metric_id in &stage.required_observation_metric_ids {
            let observer = choose_observer(
                request,
                stage,
                metric_id,
                &selected_operator,
                observers,
            )?;
            stage_compute += observer.compute_units;
            stage_latency = stage_latency
                .checked_add(observer.latency_ms)
                .ok_or_else(|| PlanCompileErrorV1::budget("stage latency overflow"))?;
            stage_risk += observer.risk_units;
            planned_observers.push(PlannedObserverV1 {
                metric_id: metric_id.clone(),
                observer_id: observer.observer_id.clone(),
                version: observer.version.clone(),
                artifact_sha256: observer.artifact_sha256.clone(),
                compute_units: observer.compute_units,
                latency_ms: observer.latency_ms,
                risk_units: observer.risk_units,
            });
        }

        total_compute += stage_compute;
        total_latency = total_latency
            .checked_add(stage_latency)
            .ok_or_else(|| PlanCompileErrorV1::budget("total plan latency overflow"))?;
        total_risk += stage_risk;

        selected_by_stage.insert(stage.stage_id.clone(), selected_operator.clone());
        planned_stages.push(PlannedStageV1 {
            stage_id: stage.stage_id.clone(),
            input_artifact_ids: stage.input_artifact_ids.clone(),
            output_artifact_id: stage.output_artifact_id.clone(),
            output_artifact_type: stage.output_artifact_type.clone(),
            dependencies: stage.dependencies.clone(),
            required_preconditions: stage.required_preconditions.clone(),
            required_postconditions: stage.required_postconditions.clone(),
            operator_id: selected_operator.operator_id.clone(),
            operator_version: selected_operator.version.clone(),
            operator_artifact_sha256: selected_operator.artifact_sha256.clone(),
            capabilities_used: stage.required_capabilities.clone(),
            affected_scope: selected_operator.maximum_affected_scope,
            determinism: selected_operator.determinism,
            observers: planned_observers,
            compute_units: stage_compute,
            latency_ms: stage_latency,
            risk_units: stage_risk,
        });
    }

    if total_compute > request.budget.maximum_compute_units {
        return Err(PlanCompileErrorV1::budget(format!(
            "plan requires {total_compute} compute units; budget is {}",
            request.budget.maximum_compute_units
        )));
    }
    if total_latency > request.budget.maximum_latency_ms {
        return Err(PlanCompileErrorV1::budget(format!(
            "plan requires {total_latency} ms; budget is {} ms",
            request.budget.maximum_latency_ms
        )));
    }
    if total_risk > request.budget.maximum_risk_units {
        return Err(PlanCompileErrorV1::budget(format!(
            "plan requires {total_risk} risk units; budget is {}",
            request.budget.maximum_risk_units
        )));
    }

    // Keep the local selected map alive through compilation to make it explicit
    // that each stage is selected against the same frozen operator registry.
    let _selected_operator_count = selected_by_stage.len();
    let objective_score = objective_score(
        total_compute,
        total_latency,
        total_risk,
        &request.objective_weights,
    );

    Ok(CompiledEditPlanV1 {
        schema: EDIT_PLAN_SCHEMA_V1.to_owned(),
        request_id: request.request_id.clone(),
        registry_snapshot_sha256: request.registry_snapshot_sha256.clone(),
        stages: planned_stages,
        total_compute_units: total_compute,
        total_latency_ms: total_latency,
        total_risk_units: total_risk,
        objective_score,
    })
}

fn validate_request(request: &EditPlanRequestV1) -> Result<(), PlanCompileErrorV1> {
    require_non_empty(&request.request_id, "request_id")?;
    validate_digest(
        &request.registry_snapshot_sha256,
        "registry_snapshot_sha256",
    )?;
    if request.stages.is_empty() {
        return Err(PlanCompileErrorV1::invalid(
            "edit plan must contain at least one stage",
        ));
    }
    validate_non_negative_finite(
        request.budget.maximum_compute_units,
        "maximum_compute_units",
    )?;
    validate_non_negative_finite(
        request.budget.maximum_risk_units,
        "maximum_risk_units",
    )?;
    validate_non_negative_finite(
        request.objective_weights.compute_weight,
        "compute_weight",
    )?;
    validate_non_negative_finite(
        request.objective_weights.latency_weight,
        "latency_weight",
    )?;
    validate_non_negative_finite(
        request.objective_weights.risk_weight,
        "risk_weight",
    )?;
    if request.objective_weights.compute_weight == 0.0
        && request.objective_weights.latency_weight == 0.0
        && request.objective_weights.risk_weight == 0.0
    {
        return Err(PlanCompileErrorV1::invalid(
            "at least one objective weight must be positive",
        ));
    }

    let mut initial_ids = BTreeSet::new();
    for artifact in &request.initial_artifacts {
        require_non_empty(&artifact.artifact_id, "initial artifact_id")?;
        require_non_empty(&artifact.artifact_type, "initial artifact_type")?;
        validate_digest(&artifact.sha256, "initial artifact sha256")?;
        if !initial_ids.insert(artifact.artifact_id.as_str()) {
            return Err(PlanCompileErrorV1::invalid(format!(
                "duplicate initial artifact ID {}",
                artifact.artifact_id
            )));
        }
    }

    let mut stage_ids = BTreeSet::new();
    let mut output_ids = BTreeSet::new();
    for stage in &request.stages {
        require_non_empty(&stage.stage_id, "stage_id")?;
        require_non_empty(&stage.output_artifact_id, "stage output_artifact_id")?;
        require_non_empty(&stage.output_artifact_type, "stage output_artifact_type")?;
        if !stage_ids.insert(stage.stage_id.as_str()) {
            return Err(PlanCompileErrorV1::invalid(format!(
                "duplicate stage ID {}",
                stage.stage_id
            )));
        }
        if !output_ids.insert(stage.output_artifact_id.as_str()) {
            return Err(PlanCompileErrorV1::invalid(format!(
                "duplicate output artifact ID {}",
                stage.output_artifact_id
            )));
        }
        if initial_ids.contains(stage.output_artifact_id.as_str()) {
            return Err(PlanCompileErrorV1::invalid(format!(
                "stage output {} shadows an initial artifact",
                stage.output_artifact_id
            )));
        }
        validate_unique_non_empty(
            &stage.input_artifact_ids,
            "stage input artifact IDs",
        )?;
        validate_unique_non_empty(&stage.dependencies, "stage dependencies")?;
        validate_unique_non_empty(
            &stage.required_capabilities,
            "required capabilities",
        )?;
        validate_unique_non_empty(
            &stage.required_preconditions,
            "required preconditions",
        )?;
        validate_unique_non_empty(
            &stage.required_postconditions,
            "required postconditions",
        )?;
        validate_unique_non_empty(
            &stage.required_observation_metric_ids,
            "required observation metric IDs",
        )?;
        if stage.required_capabilities.is_empty() {
            return Err(PlanCompileErrorV1::invalid(format!(
                "stage {} must declare at least one required capability",
                stage.stage_id
            )));
        }
        if stage.required_observation_metric_ids.is_empty() {
            return Err(PlanCompileErrorV1::invalid(format!(
                "stage {} must require at least one independent observation metric",
                stage.stage_id
            )));
        }
    }

    // Every stage input must be an initial artifact or the output of an
    // explicitly listed dependency. Dependencies cannot be decorative.
    let stage_by_id: BTreeMap<&str, &StageRequirementV1> = request
        .stages
        .iter()
        .map(|stage| (stage.stage_id.as_str(), stage))
        .collect();
    let output_producers: BTreeMap<&str, &StageRequirementV1> = request
        .stages
        .iter()
        .map(|stage| (stage.output_artifact_id.as_str(), stage))
        .collect();

    for stage in &request.stages {
        for dependency_id in &stage.dependencies {
            let dependency = stage_by_id.get(dependency_id.as_str()).copied().ok_or_else(|| {
                PlanCompileErrorV1::invalid(format!(
                    "stage {} references unknown dependency {}",
                    stage.stage_id, dependency_id
                ))
            })?;
            if dependency.stage_id == stage.stage_id {
                return Err(PlanCompileErrorV1::invalid(format!(
                    "stage {} cannot depend on itself",
                    stage.stage_id
                )));
            }
            if !stage
                .input_artifact_ids
                .contains(&dependency.output_artifact_id)
            {
                return Err(PlanCompileErrorV1::invalid(format!(
                    "stage {} declares dependency {} but does not consume its output artifact",
                    stage.stage_id, dependency_id
                )));
            }
        }

        for input_id in &stage.input_artifact_ids {
            if initial_ids.contains(input_id.as_str()) {
                continue;
            }
            let producer = output_producers.get(input_id.as_str()).copied().ok_or_else(|| {
                PlanCompileErrorV1::invalid(format!(
                    "stage {} references unknown input artifact {}",
                    stage.stage_id, input_id
                ))
            })?;
            if !stage.dependencies.contains(&producer.stage_id) {
                return Err(PlanCompileErrorV1::invalid(format!(
                    "stage {} consumes {} without declaring producer dependency {}",
                    stage.stage_id, input_id, producer.stage_id
                )));
            }
        }
    }

    Ok(())
}

fn validate_registry(
    operators: &[OperatorProfileV1],
    observers: &[ObserverProfileV1],
) -> Result<(), PlanCompileErrorV1> {
    let mut operator_ids = BTreeSet::new();
    for operator in operators {
        require_non_empty(&operator.operator_id, "operator_id")?;
        require_non_empty(&operator.version, "operator version")?;
        validate_digest(&operator.artifact_sha256, "operator artifact_sha256")?;
        validate_unique_non_empty(
            &operator.input_artifact_types,
            "operator input artifact types",
        )?;
        validate_unique_non_empty(
            &operator.output_artifact_types,
            "operator output artifact types",
        )?;
        validate_unique_non_empty(&operator.capabilities, "operator capabilities")?;
        validate_unique_non_empty(
            &operator.supported_preconditions,
            "operator supported preconditions",
        )?;
        validate_unique_non_empty(
            &operator.emitted_postconditions,
            "operator emitted postconditions",
        )?;
        if operator.output_artifact_types.is_empty() || operator.capabilities.is_empty() {
            return Err(PlanCompileErrorV1::invalid(format!(
                "operator {} must declare output types and capabilities",
                operator.operator_id
            )));
        }
        validate_non_negative_finite(operator.compute_units, "operator compute_units")?;
        validate_non_negative_finite(operator.risk_units, "operator risk_units")?;
        if !operator_ids.insert(operator.operator_id.as_str()) {
            return Err(PlanCompileErrorV1::invalid(format!(
                "duplicate operator ID {}",
                operator.operator_id
            )));
        }
    }

    let mut observer_ids = BTreeSet::new();
    for observer in observers {
        require_non_empty(&observer.observer_id, "observer_id")?;
        require_non_empty(&observer.version, "observer version")?;
        validate_digest(&observer.artifact_sha256, "observer artifact_sha256")?;
        validate_unique_non_empty(&observer.metric_ids, "observer metric IDs")?;
        if observer.metric_ids.is_empty() {
            return Err(PlanCompileErrorV1::invalid(format!(
                "observer {} must declare at least one metric",
                observer.observer_id
            )));
        }
        validate_non_negative_finite(observer.compute_units, "observer compute_units")?;
        validate_non_negative_finite(observer.risk_units, "observer risk_units")?;
        if !observer_ids.insert(observer.observer_id.as_str()) {
            return Err(PlanCompileErrorV1::invalid(format!(
                "duplicate observer ID {}",
                observer.observer_id
            )));
        }
    }
    Ok(())
}

fn choose_operator(
    request: &EditPlanRequestV1,
    stage: &StageRequirementV1,
    required_input_types: &BTreeSet<String>,
    operators: &[OperatorProfileV1],
) -> Result<OperatorProfileV1, PlanCompileErrorV1> {
    let mut candidates: Vec<&OperatorProfileV1> = operators
        .iter()
        .filter(|operator| {
            stage
                .required_capabilities
                .iter()
                .all(|required| operator.capabilities.contains(required))
                && required_input_types
                    .iter()
                    .all(|required| operator.input_artifact_types.contains(required))
                && operator
                    .output_artifact_types
                    .contains(&stage.output_artifact_type)
                && stage
                    .required_preconditions
                    .iter()
                    .all(|required| operator.supported_preconditions.contains(required))
                && stage
                    .required_postconditions
                    .iter()
                    .all(|required| operator.emitted_postconditions.contains(required))
                && operator.maximum_affected_scope <= stage.maximum_affected_scope
                && operator.determinism <= request.maximum_nondeterminism
        })
        .collect();

    candidates.sort_by(|left, right| {
        let left_score = objective_score(
            left.compute_units,
            left.latency_ms,
            left.risk_units,
            &request.objective_weights,
        );
        let right_score = objective_score(
            right.compute_units,
            right.latency_ms,
            right.risk_units,
            &request.objective_weights,
        );
        left_score
            .total_cmp(&right_score)
            .then_with(|| left.operator_id.cmp(&right.operator_id))
    });

    candidates
        .first()
        .copied()
        .cloned()
        .ok_or_else(|| {
            PlanCompileErrorV1::unsupported(format!(
                "no registered operator satisfies the capabilities, artifact types, pre/postconditions, locality and determinism contract for stage {}",
                stage.stage_id
            ))
        })
}

fn choose_observer(
    request: &EditPlanRequestV1,
    stage: &StageRequirementV1,
    metric_id: &str,
    operator: &OperatorProfileV1,
    observers: &[ObserverProfileV1],
) -> Result<ObserverProfileV1, PlanCompileErrorV1> {
    let mut candidates: Vec<&ObserverProfileV1> = observers
        .iter()
        .filter(|observer| {
            observer.independently_qualified
                && observer.metric_ids.iter().any(|metric| metric == metric_id)
                && observer.artifact_sha256 != operator.artifact_sha256
        })
        .collect();

    candidates.sort_by(|left, right| {
        let left_score = objective_score(
            left.compute_units,
            left.latency_ms,
            left.risk_units,
            &request.objective_weights,
        );
        let right_score = objective_score(
            right.compute_units,
            right.latency_ms,
            right.risk_units,
            &request.objective_weights,
        );
        left_score
            .total_cmp(&right_score)
            .then_with(|| left.observer_id.cmp(&right.observer_id))
    });

    candidates
        .first()
        .copied()
        .cloned()
        .ok_or_else(|| {
            PlanCompileErrorV1::unsupported(format!(
                "stage {} requires metric {}, but no independently qualified observer with a distinct artifact is registered",
                stage.stage_id, metric_id
            ))
        })
}

fn topological_order(
    stages: &[StageRequirementV1],
) -> Result<Vec<String>, PlanCompileErrorV1> {
    let mut indegree: BTreeMap<String, usize> = BTreeMap::new();
    let mut dependents: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for stage in stages {
        indegree.insert(stage.stage_id.clone(), stage.dependencies.len());
        for dependency in &stage.dependencies {
            dependents
                .entry(dependency.clone())
                .or_default()
                .push(stage.stage_id.clone());
        }
    }

    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(stage_id, _)| stage_id.clone())
        .collect();
    let mut ordered = Vec::with_capacity(stages.len());

    while let Some(stage_id) = ready.pop_first() {
        ordered.push(stage_id.clone());
        if let Some(children) = dependents.get(&stage_id) {
            for child in children {
                let degree = indegree.get_mut(child).ok_or_else(|| {
                    PlanCompileErrorV1::invalid("topological indegree entry disappeared")
                })?;
                *degree = degree.checked_sub(1).ok_or_else(|| {
                    PlanCompileErrorV1::invalid("topological indegree underflow")
                })?;
                if *degree == 0 {
                    ready.insert(child.clone());
                }
            }
        }
    }

    if ordered.len() != stages.len() {
        return Err(PlanCompileErrorV1::invalid(
            "stage dependency graph contains a cycle",
        ));
    }
    Ok(ordered)
}

fn initial_artifact_types<'a>(
    artifacts: &'a [ArtifactRefV1],
) -> BTreeMap<&'a str, &'a str> {
    artifacts
        .iter()
        .map(|artifact| (artifact.artifact_id.as_str(), artifact.artifact_type.as_str()))
        .collect()
}

fn objective_score(
    compute_units: f64,
    latency_ms: u64,
    risk_units: f64,
    weights: &ObjectiveWeightsV1,
) -> f64 {
    weights.compute_weight * compute_units
        + weights.latency_weight * latency_ms as f64
        + weights.risk_weight * risk_units
}

fn validate_unique_non_empty(
    values: &[String],
    field: &str,
) -> Result<(), PlanCompileErrorV1> {
    let mut seen = BTreeSet::new();
    for value in values {
        require_non_empty(value, field)?;
        if !seen.insert(value.as_str()) {
            return Err(PlanCompileErrorV1::invalid(format!(
                "{field} contains duplicate value {value}"
            )));
        }
    }
    Ok(())
}

fn validate_non_negative_finite(
    value: f64,
    field: &str,
) -> Result<(), PlanCompileErrorV1> {
    if !value.is_finite() || value < 0.0 {
        return Err(PlanCompileErrorV1::invalid(format!(
            "{field} must be finite and non-negative"
        )));
    }
    Ok(())
}

fn validate_digest(value: &str, field: &str) -> Result<(), PlanCompileErrorV1> {
    let valid = value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex.bytes().all(|byte| {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            })
    });
    if !valid {
        return Err(PlanCompileErrorV1::invalid(format!(
            "{field} must be sha256: followed by 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), PlanCompileErrorV1> {
    if value.trim().is_empty() {
        Err(PlanCompileErrorV1::invalid(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        format!("sha256:{}", character.to_string().repeat(64))
    }

    fn request() -> EditPlanRequestV1 {
        EditPlanRequestV1 {
            request_id: "request-001".to_owned(),
            registry_snapshot_sha256: digest('a'),
            initial_artifacts: vec![ArtifactRefV1 {
                artifact_id: "source-vocal".to_owned(),
                artifact_type: "audio/vocal-stem".to_owned(),
                sha256: digest('b'),
            }],
            stages: vec![StageRequirementV1 {
                stage_id: "replace-word".to_owned(),
                input_artifact_ids: vec!["source-vocal".to_owned()],
                output_artifact_id: "edited-vocal".to_owned(),
                output_artifact_type: "audio/vocal-stem".to_owned(),
                dependencies: Vec::new(),
                required_capabilities: vec!["vocal.lyric-local-edit".to_owned()],
                required_preconditions: vec!["source.aligned".to_owned()],
                required_postconditions: vec!["target.lyric-changed".to_owned()],
                required_observation_metric_ids: vec![
                    "phoneme-error-rate".to_owned(),
                    "f0-correlation".to_owned(),
                ],
                maximum_affected_scope: LocalityScopeV1::Phrase,
            }],
            budget: PlanBudgetV1 {
                maximum_compute_units: 20.0,
                maximum_latency_ms: 2_000,
                maximum_risk_units: 1.0,
            },
            objective_weights: ObjectiveWeightsV1 {
                compute_weight: 1.0,
                latency_weight: 0.001,
                risk_weight: 10.0,
            },
            maximum_nondeterminism: DeterminismV1::Seeded,
        }
    }

    fn operator(
        id: &str,
        artifact: char,
        scope: LocalityScopeV1,
        determinism: DeterminismV1,
        compute: f64,
        latency: u64,
        risk: f64,
    ) -> OperatorProfileV1 {
        OperatorProfileV1 {
            operator_id: id.to_owned(),
            version: "1.0".to_owned(),
            artifact_sha256: digest(artifact),
            input_artifact_types: vec!["audio/vocal-stem".to_owned()],
            output_artifact_types: vec!["audio/vocal-stem".to_owned()],
            capabilities: vec!["vocal.lyric-local-edit".to_owned()],
            supported_preconditions: vec!["source.aligned".to_owned()],
            emitted_postconditions: vec!["target.lyric-changed".to_owned()],
            maximum_affected_scope: scope,
            determinism,
            compute_units: compute,
            latency_ms: latency,
            risk_units: risk,
        }
    }

    fn observer(id: &str, artifact: char, metric: &str) -> ObserverProfileV1 {
        ObserverProfileV1 {
            observer_id: id.to_owned(),
            version: "pinned-1".to_owned(),
            artifact_sha256: digest(artifact),
            metric_ids: vec![metric.to_owned()],
            independently_qualified: true,
            compute_units: 0.5,
            latency_ms: 100,
            risk_units: 0.01,
        }
    }

    fn registries() -> (Vec<OperatorProfileV1>, Vec<ObserverProfileV1>) {
        (
            vec![operator(
                "formant-local-editor",
                'c',
                LocalityScopeV1::Phrase,
                DeterminismV1::Deterministic,
                5.0,
                500,
                0.05,
            )],
            vec![
                observer("independent-singing-asr", 'd', "phoneme-error-rate"),
                observer("score-independent-f0", 'e', "f0-correlation"),
            ],
        )
    }

    #[test]
    fn compiles_supported_local_edit_with_independent_observers() {
        let (operators, observers) = registries();
        let plan = compile_edit_plan(&request(), &operators, &observers).unwrap();
        assert_eq!(plan.schema, EDIT_PLAN_SCHEMA_V1);
        assert_eq!(plan.stages.len(), 1);
        assert_eq!(plan.stages[0].operator_id, "formant-local-editor");
        assert_eq!(plan.stages[0].observers.len(), 2);
        assert!(plan.stages[0]
            .observers
            .iter()
            .all(|observer| observer.artifact_sha256 != plan.stages[0].operator_artifact_sha256));
    }

    #[test]
    fn rejects_missing_capability_instead_of_falling_back() {
        let (mut operators, observers) = registries();
        operators[0].capabilities.clear();
        let error = compile_edit_plan(&request(), &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::Unsupported);
    }

    #[test]
    fn rejects_global_operator_for_phrase_local_request() {
        let (mut operators, observers) = registries();
        operators[0].maximum_affected_scope = LocalityScopeV1::Global;
        let error = compile_edit_plan(&request(), &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::Unsupported);
    }

    #[test]
    fn rejects_operator_that_exceeds_determinism_requirement() {
        let (mut operators, observers) = registries();
        operators[0].determinism = DeterminismV1::Nondeterministic;
        let error = compile_edit_plan(&request(), &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::Unsupported);
    }

    #[test]
    fn rejects_missing_independent_metric_observer() {
        let (operators, mut observers) = registries();
        observers.retain(|observer| observer.metric_ids[0] != "f0-correlation");
        let error = compile_edit_plan(&request(), &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::Unsupported);
    }

    #[test]
    fn rejects_observer_that_reuses_operator_artifact() {
        let (operators, mut observers) = registries();
        observers[0].artifact_sha256 = operators[0].artifact_sha256.clone();
        let error = compile_edit_plan(&request(), &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::Unsupported);
    }

    #[test]
    fn rejects_observer_not_independently_qualified() {
        let (operators, mut observers) = registries();
        observers[0].independently_qualified = false;
        let error = compile_edit_plan(&request(), &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::Unsupported);
    }

    #[test]
    fn enforces_compute_budget() {
        let (operators, observers) = registries();
        let mut request = request();
        request.budget.maximum_compute_units = 1.0;
        let error = compile_edit_plan(&request, &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::BudgetExceeded);
    }

    #[test]
    fn enforces_latency_budget_including_observers() {
        let (operators, observers) = registries();
        let mut request = request();
        request.budget.maximum_latency_ms = 100;
        let error = compile_edit_plan(&request, &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::BudgetExceeded);
    }

    #[test]
    fn enforces_additive_risk_budget() {
        let (operators, observers) = registries();
        let mut request = request();
        request.budget.maximum_risk_units = 0.01;
        let error = compile_edit_plan(&request, &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::BudgetExceeded);
    }

    #[test]
    fn rejects_unknown_input_artifact() {
        let (operators, observers) = registries();
        let mut request = request();
        request.stages[0].input_artifact_ids = vec!["missing-source".to_owned()];
        let error = compile_edit_plan(&request, &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::InvalidInput);
    }

    #[test]
    fn rejects_dependency_whose_output_is_not_consumed() {
        let (operators, observers) = registries();
        let mut request = request();
        request.stages.push(StageRequirementV1 {
            stage_id: "second-stage".to_owned(),
            input_artifact_ids: vec!["source-vocal".to_owned()],
            output_artifact_id: "second-output".to_owned(),
            output_artifact_type: "audio/vocal-stem".to_owned(),
            dependencies: vec!["replace-word".to_owned()],
            required_capabilities: vec!["vocal.lyric-local-edit".to_owned()],
            required_preconditions: vec!["source.aligned".to_owned()],
            required_postconditions: vec!["target.lyric-changed".to_owned()],
            required_observation_metric_ids: vec!["phoneme-error-rate".to_owned()],
            maximum_affected_scope: LocalityScopeV1::Phrase,
        });
        let error = compile_edit_plan(&request, &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::InvalidInput);
    }

    #[test]
    fn rejects_cycle_in_stage_graph() {
        let (operators, observers) = registries();
        let mut request = request();
        request.stages[0].output_artifact_id = "out-a".to_owned();
        request.stages[0].dependencies = vec!["stage-b".to_owned()];
        request.stages[0].input_artifact_ids = vec!["out-b".to_owned()];
        request.stages.push(StageRequirementV1 {
            stage_id: "stage-b".to_owned(),
            input_artifact_ids: vec!["out-a".to_owned()],
            output_artifact_id: "out-b".to_owned(),
            output_artifact_type: "audio/vocal-stem".to_owned(),
            dependencies: vec!["replace-word".to_owned()],
            required_capabilities: vec!["vocal.lyric-local-edit".to_owned()],
            required_preconditions: vec!["source.aligned".to_owned()],
            required_postconditions: vec!["target.lyric-changed".to_owned()],
            required_observation_metric_ids: vec!["phoneme-error-rate".to_owned()],
            maximum_affected_scope: LocalityScopeV1::Phrase,
        });
        let error = compile_edit_plan(&request, &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::InvalidInput);
    }

    #[test]
    fn rejects_missing_operator_precondition_or_postcondition() {
        let (mut operators, observers) = registries();
        operators[0].supported_preconditions.clear();
        assert!(compile_edit_plan(&request(), &operators, &observers).is_err());

        let (mut operators, observers) = registries();
        operators[0].emitted_postconditions.clear();
        assert!(compile_edit_plan(&request(), &operators, &observers).is_err());
    }

    #[test]
    fn rejects_output_type_mismatch() {
        let (mut operators, observers) = registries();
        operators[0].output_artifact_types = vec!["audio/full-mix".to_owned()];
        let error = compile_edit_plan(&request(), &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::Unsupported);
    }

    #[test]
    fn rejects_stage_consuming_dependency_without_declaring_it() {
        let (operators, observers) = registries();
        let mut request = request();
        request.stages[0].output_artifact_id = "stage-a-out".to_owned();
        request.stages[0].input_artifact_ids.push("stage-b-out".to_owned());
        request.stages.push(StageRequirementV1 {
            stage_id: "stage-b".to_owned(),
            input_artifact_ids: vec!["source-vocal".to_owned()],
            output_artifact_id: "stage-b-out".to_owned(),
            output_artifact_type: "audio/vocal-stem".to_owned(),
            dependencies: Vec::new(),
            required_capabilities: vec!["vocal.lyric-local-edit".to_owned()],
            required_preconditions: vec!["source.aligned".to_owned()],
            required_postconditions: vec!["target.lyric-changed".to_owned()],
            required_observation_metric_ids: vec!["phoneme-error-rate".to_owned()],
            maximum_affected_scope: LocalityScopeV1::Phrase,
        });
        let error = compile_edit_plan(&request, &operators, &observers).unwrap_err();
        assert_eq!(error.kind, PlanErrorKindV1::InvalidInput);
    }

    #[test]
    fn stage_order_is_topological_and_deterministic() {
        let (operators, observers) = registries();
        let mut request = request();
        request.stages[0].stage_id = "z-first".to_owned();
        request.stages[0].output_artifact_id = "intermediate".to_owned();
        request.stages.push(StageRequirementV1 {
            stage_id: "a-second".to_owned(),
            input_artifact_ids: vec!["intermediate".to_owned()],
            output_artifact_id: "final-output".to_owned(),
            output_artifact_type: "audio/vocal-stem".to_owned(),
            dependencies: vec!["z-first".to_owned()],
            required_capabilities: vec!["vocal.lyric-local-edit".to_owned()],
            required_preconditions: vec!["source.aligned".to_owned()],
            required_postconditions: vec!["target.lyric-changed".to_owned()],
            required_observation_metric_ids: vec!["phoneme-error-rate".to_owned()],
            maximum_affected_scope: LocalityScopeV1::Phrase,
        });
        let plan = compile_edit_plan(&request, &operators, &observers).unwrap();
        assert_eq!(plan.stages[0].stage_id, "z-first");
        assert_eq!(plan.stages[1].stage_id, "a-second");
    }

    #[test]
    fn selects_lowest_weighted_eligible_operator_deterministically() {
        let (mut operators, observers) = registries();
        operators.push(operator(
            "high-cost-local-editor",
            'f',
            LocalityScopeV1::SampleInterval,
            DeterminismV1::Deterministic,
            15.0,
            1_000,
            0.01,
        ));
        let plan = compile_edit_plan(&request(), &operators, &observers).unwrap();
        assert_eq!(plan.stages[0].operator_id, "formant-local-editor");
    }

    #[test]
    fn rejects_duplicate_stage_ids_and_duplicate_output_artifact_ids() {
        let (operators, observers) = registries();
        let mut duplicate_stage = request();
        duplicate_stage.stages.push(duplicate_stage.stages[0].clone());
        assert!(compile_edit_plan(&duplicate_stage, &operators, &observers).is_err());

        let mut duplicate_output = request();
        let mut second = duplicate_output.stages[0].clone();
        second.stage_id = "second".to_owned();
        duplicate_output.stages.push(second);
        assert!(compile_edit_plan(&duplicate_output, &operators, &observers).is_err());
    }

    #[test]
    fn rejects_zero_objective_weights_and_malformed_registry_digest() {
        let (operators, observers) = registries();
        let mut zero = request();
        zero.objective_weights = ObjectiveWeightsV1 {
            compute_weight: 0.0,
            latency_weight: 0.0,
            risk_weight: 0.0,
        };
        assert!(compile_edit_plan(&zero, &operators, &observers).is_err());

        let mut malformed = request();
        malformed.registry_snapshot_sha256 = "not-a-digest".to_owned();
        assert!(compile_edit_plan(&malformed, &operators, &observers).is_err());
    }
}
