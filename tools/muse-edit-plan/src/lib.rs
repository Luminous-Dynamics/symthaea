//! CQEP-001: capability-qualified edit-plan compiler.
//!
//! This crate consumes a normalized request and a frozen capability registry.
//! It emits a deterministic plan and evidence DAG, or an explicit
//! non-qualification result. It does not parse language, run operators, mutate
//! audio, establish that a capability declaration is true, or authorize a
//! commit.
//!
//! Profile qualification and rights verification are supplied by an external
//! governance/evaluation process. This compiler checks only that the exact
//! frozen declarations satisfy the request's static requirements.

use std::collections::{BTreeMap, BTreeSet};

pub const PLAN_SCHEMA_VERSION: &str = "cqep-001-plan-v1";
pub const CANONICAL_MANIFEST_ENCODING: &str = "cqep-tab-records-percent-utf8-v1";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Modality {
    Music,
    Speech,
    EnvironmentalSound,
    Mixed,
}

impl Modality {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Music => "music",
            Self::Speech => "speech",
            Self::EnvironmentalSound => "environmental_sound",
            Self::Mixed => "mixed",
        }
    }
}

/// Lower ranks are more local. A profile with a broader mutation scope than
/// the request permits is rejected, never silently promoted to a wider edit.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MutationScope {
    SampleRange,
    Phrase,
    Stem,
    Track,
    Mix,
}

impl MutationScope {
    pub const fn label(self) -> &'static str {
        match self {
            Self::SampleRange => "sample_range",
            Self::Phrase => "phrase",
            Self::Stem => "stem",
            Self::Track => "track",
            Self::Mix => "mix",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QualificationStatus {
    Qualified,
    Provisional,
    Revoked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RightsStatus {
    Verified,
    Restricted,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reproducibility {
    Deterministic,
    Seeded,
    Nondeterministic,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ProfileIdentity {
    pub id: String,
    pub version: String,
    pub artifact_sha256: String,
}

impl ProfileIdentity {
    pub fn stable_key(&self) -> String {
        format!("{}@{}#{}", self.id, self.version, self.artifact_sha256)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactRef {
    pub id: String,
    pub artifact_type: String,
    pub modality: Modality,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InputOrigin {
    SourceArtifact(String),
    StageOutput(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputBinding {
    pub input_id: String,
    pub expected_artifact_type: String,
    pub origin: InputOrigin,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericControlRequirement {
    pub control_id: String,
    pub minimum: f64,
    pub maximum: f64,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericControlRange {
    pub control_id: String,
    pub minimum: f64,
    pub maximum: f64,
    pub unit: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationRequirement {
    pub metric_id: String,
    pub minimum_observers: usize,
    pub must_be_independent_from_mutator: bool,
    pub distinct_observer_groups: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditStageRequest {
    pub stage_id: String,
    pub operation_id: String,
    pub inputs: Vec<InputBinding>,
    pub output_artifact_type: String,
    pub modality: Modality,
    pub maximum_permitted_mutation_scope: MutationScope,
    pub required_capabilities: BTreeSet<String>,
    pub protected_properties: BTreeSet<String>,
    pub required_receipts: BTreeSet<String>,
    pub required_preconditions: BTreeSet<String>,
    pub required_postconditions: BTreeSet<String>,
    pub controls: Vec<NumericControlRequirement>,
    pub observations: Vec<ObservationRequirement>,
    pub reproducibility_seed: Option<u64>,
    pub preferred_operator: Option<ProfileIdentity>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceBudget {
    pub maximum_total_latency_ms: u64,
    pub maximum_total_compute_units: u64,
    pub maximum_mutation_risk_score: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlannerPolicy {
    pub require_reproducible_operators: bool,
    pub require_verified_rights: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditRequest {
    pub request_id: String,
    pub sources: Vec<ArtifactRef>,
    pub stages: Vec<EditStageRequest>,
    pub resource_budget: ResourceBudget,
    pub policy: PlannerPolicy,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OperatorProfile {
    pub identity: ProfileIdentity,
    pub qualification: QualificationStatus,
    pub rights: RightsStatus,
    pub supported_operations: BTreeSet<String>,
    pub input_modalities: BTreeSet<Modality>,
    pub input_artifact_types: BTreeSet<String>,
    pub output_artifact_type: String,
    pub capabilities: BTreeSet<String>,
    pub maximum_mutation_scope: MutationScope,
    pub preservation_guarantees: BTreeSet<String>,
    pub emitted_receipts: BTreeSet<String>,
    pub supported_preconditions: BTreeSet<String>,
    pub guaranteed_postconditions: BTreeSet<String>,
    pub controls: Vec<NumericControlRange>,
    pub reproducibility: Reproducibility,
    pub independence_group: String,
    pub risk_score: u8,
    pub estimated_latency_ms: u64,
    pub estimated_compute_units: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObserverProfile {
    pub identity: ProfileIdentity,
    pub qualification: QualificationStatus,
    pub rights: RightsStatus,
    pub modalities: BTreeSet<Modality>,
    pub accepted_artifact_types: BTreeSet<String>,
    pub measured_metrics: BTreeSet<String>,
    pub independence_group: String,
    pub emitted_receipts: BTreeSet<String>,
    pub estimated_latency_ms: u64,
    pub estimated_compute_units: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CapabilityRegistry {
    pub registry_version: String,
    pub operators: Vec<OperatorProfile>,
    pub observers: Vec<ObserverProfile>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnsupportedReason {
    NoQualifiedOperator,
    NoIndependentObserver,
    ObserverCoverageMissing,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompileError {
    InvalidRequest(String),
    Unsupported {
        subject: String,
        reason: UnsupportedReason,
        detail: String,
    },
    Ambiguous {
        stage_id: String,
        candidates: Vec<ProfileIdentity>,
    },
    TypeMismatch {
        stage_id: String,
        input_id: String,
        expected: String,
        actual: String,
    },
    DependencyCycle {
        members: Vec<String>,
    },
    BudgetExceeded {
        resource: &'static str,
        estimated: u64,
        maximum: u64,
    },
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(reason) => write!(f, "invalid request: {reason}"),
            Self::Unsupported { subject, reason, detail } => {
                write!(f, "unsupported {subject}: {reason:?}: {detail}")
            }
            Self::Ambiguous { stage_id, candidates } => {
                write!(f, "ambiguous operator selection for {stage_id}: {candidates:?}")
            }
            Self::TypeMismatch { stage_id, input_id, expected, actual } => write!(
                f,
                "type mismatch at {stage_id}/{input_id}: expected {expected}, got {actual}"
            ),
            Self::DependencyCycle { members } => {
                write!(f, "dependency cycle among stages: {}", members.join(", "))
            }
            Self::BudgetExceeded { resource, estimated, maximum } => {
                write!(f, "{resource} budget exceeded: {estimated} > {maximum}")
            }
        }
    }
}

impl std::error::Error for CompileError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorSelection {
    pub stage_id: String,
    pub profile: ProfileIdentity,
    pub risk_score: u8,
    pub latency_ms: u64,
    pub compute_units: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObserverSelection {
    pub stage_id: String,
    pub metric_id: String,
    pub profile: ProfileIdentity,
    pub independence_group: String,
    pub latency_ms: u64,
    pub compute_units: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanNodeKind {
    VerifySource,
    Mutate,
    Observe,
    FinalQualification,
}

impl PlanNodeKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::VerifySource => "verify_source",
            Self::Mutate => "mutate",
            Self::Observe => "observe",
            Self::FinalQualification => "final_qualification",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanNode {
    pub node_id: String,
    pub kind: PlanNodeKind,
    pub depends_on: Vec<String>,
    pub stage_id: Option<String>,
    pub profile: Option<ProfileIdentity>,
    pub input_artifact_types: Vec<String>,
    pub output_artifact_type: Option<String>,
    pub evidence_metric: Option<String>,
    pub preconditions: BTreeSet<String>,
    pub postconditions: BTreeSet<String>,
    pub receipt_obligations: BTreeSet<String>,
    pub estimated_latency_ms: u64,
    pub estimated_compute_units: u64,
    pub risk_score: u8,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceEstimate {
    pub total_latency_ms: u64,
    pub total_compute_units: u64,
    pub maximum_mutation_risk_score: u8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledPlan {
    pub schema_version: String,
    pub request_id: String,
    pub registry_version: String,
    pub sources: Vec<ArtifactRef>,
    pub operator_selections: Vec<OperatorSelection>,
    pub observer_selections: Vec<ObserverSelection>,
    pub nodes: Vec<PlanNode>,
    pub resource_estimate: ResourceEstimate,
    pub execution_authorized: bool,
    pub digest_algorithm: String,
    pub digest_input_encoding: String,
}

impl CompiledPlan {
    /// Stable UTF-8 input for an external SHA-256 implementation.
    /// This crate does not implement a home-grown cryptographic digest.
    pub fn canonical_manifest(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("schema\t{}", atom(&self.schema_version)));
        lines.push(format!("request\t{}", atom(&self.request_id)));
        lines.push(format!("registry\t{}", atom(&self.registry_version)));
        lines.push(format!(
            "resources\t{}\t{}\t{}",
            self.resource_estimate.total_latency_ms,
            self.resource_estimate.total_compute_units,
            self.resource_estimate.maximum_mutation_risk_score
        ));
        lines.push(format!("execution_authorized\t{}", self.execution_authorized));

        let mut sources = self.sources.clone();
        sources.sort_by(|left, right| left.id.cmp(&right.id));
        for source in sources {
            lines.push(format!(
                "source\t{}\t{}\t{}\t{}",
                atom(&source.id),
                atom(&source.artifact_type),
                source.modality.label(),
                atom(&source.sha256)
            ));
        }

        for node in &self.nodes {
            let profile = node.profile.as_ref().map(|p| atom(&p.stable_key())).unwrap_or_default();
            let stage = node.stage_id.as_deref().map(atom).unwrap_or_default();
            let output = node.output_artifact_type.as_deref().map(atom).unwrap_or_default();
            let metric = node.evidence_metric.as_deref().map(atom).unwrap_or_default();
            let input_types = join_atoms(&node.input_artifact_types);
            let preconditions = join_atoms(&node.preconditions.iter().cloned().collect::<Vec<_>>());
            let postconditions = join_atoms(&node.postconditions.iter().cloned().collect::<Vec<_>>());
            let receipts = join_atoms(&node.receipt_obligations.iter().cloned().collect::<Vec<_>>());
            lines.push(format!(
                "node\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                atom(&node.node_id),
                node.kind.label(),
                join_atoms(&node.depends_on),
                stage,
                profile,
                input_types,
                output,
                metric,
                preconditions,
                postconditions,
                receipts,
                node.estimated_latency_ms,
                format!("{}:{}", node.estimated_compute_units, node.risk_score)
            ));
        }
        lines.join("\n") + "\n"
    }
}

fn atom(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            encoded.push(byte as char);
        } else {
            use std::fmt::Write as _;
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

fn join_atoms(values: &[String]) -> String {
    values.iter().map(|value| atom(value)).collect::<Vec<_>>().join(",")
}

fn valid_identifier(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= 256
        && !value.chars().any(char::is_control)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_numeric_range(
    id: &str,
    minimum: f64,
    maximum: f64,
    unit: &str,
) -> Result<(), CompileError> {
    if !valid_identifier(id) || !valid_identifier(unit) {
        return Err(CompileError::InvalidRequest(
            "control IDs and units must be non-empty identifiers".to_string(),
        ));
    }
    if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
        return Err(CompileError::InvalidRequest(format!(
            "invalid numeric control range for {id}"
        )));
    }
    Ok(())
}

fn validate_request(request: &EditRequest) -> Result<(), CompileError> {
    if !valid_identifier(&request.request_id) {
        return Err(CompileError::InvalidRequest("invalid request_id".to_string()));
    }
    if request.sources.is_empty() || request.stages.is_empty() {
        return Err(CompileError::InvalidRequest(
            "at least one source and one edit stage are required".to_string(),
        ));
    }

    let mut source_ids = BTreeSet::new();
    for source in &request.sources {
        if !valid_identifier(&source.id)
            || !valid_identifier(&source.artifact_type)
            || !valid_sha256(&source.sha256)
        {
            return Err(CompileError::InvalidRequest(format!(
                "invalid source artifact {}",
                source.id
            )));
        }
        if !source_ids.insert(source.id.clone()) {
            return Err(CompileError::InvalidRequest(format!(
                "duplicate source artifact ID {}",
                source.id
            )));
        }
    }

    let mut stage_ids = BTreeSet::new();
    for stage in &request.stages {
        if !valid_identifier(&stage.stage_id)
            || !valid_identifier(&stage.operation_id)
            || !valid_identifier(&stage.output_artifact_type)
        {
            return Err(CompileError::InvalidRequest(format!(
                "invalid stage identity or output type: {}",
                stage.stage_id
            )));
        }
        if !stage_ids.insert(stage.stage_id.clone()) {
            return Err(CompileError::InvalidRequest(format!(
                "duplicate stage ID {}",
                stage.stage_id
            )));
        }
        if stage.inputs.is_empty() {
            return Err(CompileError::InvalidRequest(format!(
                "stage {} has no typed inputs",
                stage.stage_id
            )));
        }
        let mut input_ids = BTreeSet::new();
        for input in &stage.inputs {
            if !valid_identifier(&input.input_id) || !valid_identifier(&input.expected_artifact_type) {
                return Err(CompileError::InvalidRequest(format!(
                    "invalid input binding in {}",
                    stage.stage_id
                )));
            }
            if !input_ids.insert(input.input_id.clone()) {
                return Err(CompileError::InvalidRequest(format!(
                    "duplicate input ID {} in {}",
                    input.input_id, stage.stage_id
                )));
            }
        }
        for control in &stage.controls {
            validate_numeric_range(&control.control_id, control.minimum, control.maximum, &control.unit)?;
        }
        for observation in &stage.observations {
            if !valid_identifier(&observation.metric_id) || observation.minimum_observers == 0 {
                return Err(CompileError::InvalidRequest(format!(
                    "invalid observation requirement in {}",
                    stage.stage_id
                )));
            }
        }
    }
    Ok(())
}

fn validate_registry(registry: &CapabilityRegistry) -> Result<(), CompileError> {
    if !valid_identifier(&registry.registry_version) {
        return Err(CompileError::InvalidRequest(
            "registry version must be a non-empty identifier".to_string(),
        ));
    }
    for profile in &registry.operators {
        if !valid_identifier(&profile.identity.id)
            || !valid_identifier(&profile.identity.version)
            || !valid_sha256(&profile.identity.artifact_sha256)
            || !valid_identifier(&profile.output_artifact_type)
            || !valid_identifier(&profile.independence_group)
            || profile.risk_score > 100
        {
            return Err(CompileError::InvalidRequest(format!(
                "invalid operator profile {}",
                profile.identity.id
            )));
        }
        for control in &profile.controls {
            validate_numeric_range(&control.control_id, control.minimum, control.maximum, &control.unit)?;
        }
    }
    for profile in &registry.observers {
        if !valid_identifier(&profile.identity.id)
            || !valid_identifier(&profile.identity.version)
            || !valid_sha256(&profile.identity.artifact_sha256)
            || !valid_identifier(&profile.independence_group)
        {
            return Err(CompileError::InvalidRequest(format!(
                "invalid observer profile {}",
                profile.identity.id
            )));
        }
    }
    Ok(())
}

fn validate_stage_graph(
    request: &EditRequest,
) -> Result<(BTreeMap<String, EditStageRequest>, Vec<String>), CompileError> {
    let stage_map: BTreeMap<String, EditStageRequest> = request
        .stages
        .iter()
        .cloned()
        .map(|stage| (stage.stage_id.clone(), stage))
        .collect();
    let source_types: BTreeMap<String, String> = request
        .sources
        .iter()
        .map(|source| (source.id.clone(), source.artifact_type.clone()))
        .collect();

    let mut indegree: BTreeMap<String, usize> =
        stage_map.keys().map(|id| (id.clone(), 0)).collect();
    let mut dependents: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for stage in stage_map.values() {
        let mut dependencies = BTreeSet::new();
        for input in &stage.inputs {
            let actual_type = match &input.origin {
                InputOrigin::SourceArtifact(source_id) => source_types.get(source_id).ok_or_else(|| {
                    CompileError::InvalidRequest(format!(
                        "stage {} references unknown source {}",
                        stage.stage_id, source_id
                    ))
                })?.clone(),
                InputOrigin::StageOutput(parent_id) => {
                    let parent = stage_map.get(parent_id).ok_or_else(|| {
                        CompileError::InvalidRequest(format!(
                            "stage {} references unknown parent stage {}",
                            stage.stage_id, parent_id
                        ))
                    })?;
                    if parent_id == &stage.stage_id {
                        return Err(CompileError::DependencyCycle {
                            members: vec![stage.stage_id.clone()],
                        });
                    }
                    dependencies.insert(parent_id.clone());
                    parent.output_artifact_type.clone()
                }
            };
            if actual_type != input.expected_artifact_type {
                return Err(CompileError::TypeMismatch {
                    stage_id: stage.stage_id.clone(),
                    input_id: input.input_id.clone(),
                    expected: input.expected_artifact_type.clone(),
                    actual: actual_type,
                });
            }
        }
        *indegree.get_mut(&stage.stage_id).expect("stage exists") = dependencies.len();
        for parent_id in dependencies {
            dependents.entry(parent_id).or_default().insert(stage.stage_id.clone());
        }
    }

    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(id.clone()))
        .collect();
    let mut order = Vec::with_capacity(stage_map.len());
    while let Some(id) = ready.pop_first() {
        order.push(id.clone());
        if let Some(children) = dependents.get(&id) {
            for child in children {
                let degree = indegree.get_mut(child).expect("dependent exists");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(child.clone());
                }
            }
        }
    }
    if order.len() != stage_map.len() {
        let members = indegree
            .into_iter()
            .filter_map(|(id, degree)| (degree > 0).then_some(id))
            .collect();
        return Err(CompileError::DependencyCycle { members });
    }
    Ok((stage_map, order))
}

fn control_is_supported(
    requirement: &NumericControlRequirement,
    profile: &OperatorProfile,
) -> bool {
    profile.controls.iter().any(|control| {
        control.control_id == requirement.control_id
            && control.unit == requirement.unit
            && control.minimum <= requirement.minimum
            && control.maximum >= requirement.maximum
    })
}

fn operator_compatibility_reasons(
    request: &EditRequest,
    stage: &EditStageRequest,
    input_types: &[String],
    profile: &OperatorProfile,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if profile.qualification != QualificationStatus::Qualified {
        reasons.push("profile is not Qualified".to_string());
    }
    if request.policy.require_verified_rights && profile.rights != RightsStatus::Verified {
        reasons.push("operator rights are not Verified".to_string());
    }
    if !profile.supported_operations.contains(&stage.operation_id) {
        reasons.push("operation is not supported".to_string());
    }
    if !profile.input_modalities.contains(&stage.modality) {
        reasons.push("modality is not supported".to_string());
    }
    if input_types
        .iter()
        .any(|artifact_type| !profile.input_artifact_types.contains(artifact_type))
    {
        reasons.push("one or more input artifact types are unsupported".to_string());
    }
    if profile.output_artifact_type != stage.output_artifact_type {
        reasons.push("output artifact type does not match".to_string());
    }
    if !profile.capabilities.is_superset(&stage.required_capabilities) {
        reasons.push("required capability is missing".to_string());
    }
    if profile.maximum_mutation_scope > stage.maximum_permitted_mutation_scope {
        reasons.push(format!(
            "mutation scope {} exceeds permitted {}",
            profile.maximum_mutation_scope.label(),
            stage.maximum_permitted_mutation_scope.label()
        ));
    }
    if !profile.preservation_guarantees.is_superset(&stage.protected_properties) {
        reasons.push("protected-property guarantee is missing".to_string());
    }
    if !profile.emitted_receipts.is_superset(&stage.required_receipts) {
        reasons.push("required execution receipt is not emitted".to_string());
    }
    if !profile.supported_preconditions.is_superset(&stage.required_preconditions) {
        reasons.push("required precondition is unsupported".to_string());
    }
    if !profile.guaranteed_postconditions.is_superset(&stage.required_postconditions) {
        reasons.push("required postcondition is not guaranteed".to_string());
    }
    if stage.controls.iter().any(|control| !control_is_supported(control, profile)) {
        reasons.push("one or more requested control ranges are unsupported".to_string());
    }
    if request.policy.require_reproducible_operators {
        match profile.reproducibility {
            Reproducibility::Deterministic => {}
            Reproducibility::Seeded if stage.reproducibility_seed.is_some() => {}
            Reproducibility::Seeded => reasons.push("seeded operator has no request seed".to_string()),
            Reproducibility::Nondeterministic => reasons.push("operator is nondeterministic".to_string()),
        }
    }
    if profile.risk_score > request.resource_budget.maximum_mutation_risk_score {
        reasons.push("operator risk exceeds the request budget".to_string());
    }
    reasons
}

fn select_operator<'a>(
    request: &EditRequest,
    stage: &EditStageRequest,
    input_types: &[String],
    registry: &'a CapabilityRegistry,
) -> Result<&'a OperatorProfile, CompileError> {
    let mut compatible: Vec<&OperatorProfile> = Vec::new();
    let mut rejection_reasons = BTreeSet::new();

    for profile in &registry.operators {
        let reasons = operator_compatibility_reasons(request, stage, input_types, profile);
        if reasons.is_empty() {
            compatible.push(profile);
        } else if profile.supported_operations.contains(&stage.operation_id) {
            rejection_reasons.extend(reasons);
        }
    }

    if let Some(preferred) = &stage.preferred_operator {
        compatible.retain(|profile| &profile.identity == preferred);
        if compatible.is_empty() {
            return Err(CompileError::Unsupported {
                subject: stage.stage_id.clone(),
                reason: UnsupportedReason::NoQualifiedOperator,
                detail: format!(
                    "preferred operator {} is absent or fails the contract; reasons: {}",
                    preferred.stable_key(),
                    rejection_reasons.into_iter().collect::<Vec<_>>().join("; ")
                ),
            });
        }
    }

    if compatible.is_empty() {
        return Err(CompileError::Unsupported {
            subject: stage.stage_id.clone(),
            reason: UnsupportedReason::NoQualifiedOperator,
            detail: if rejection_reasons.is_empty() {
                "registry contains no operator for this operation".to_string()
            } else {
                rejection_reasons.into_iter().collect::<Vec<_>>().join("; ")
            },
        });
    }

    compatible.sort_by_key(|profile| {
        (
            profile.risk_score,
            profile.estimated_latency_ms,
            profile.estimated_compute_units,
            profile.identity.clone(),
        )
    });
    let best = compatible[0];
    let best_objective = (
        best.risk_score,
        best.estimated_latency_ms,
        best.estimated_compute_units,
    );
    let tied: Vec<ProfileIdentity> = compatible
        .iter()
        .take_while(|profile| {
            (
                profile.risk_score,
                profile.estimated_latency_ms,
                profile.estimated_compute_units,
            ) == best_objective
        })
        .map(|profile| profile.identity.clone())
        .collect();
    if tied.len() > 1 {
        return Err(CompileError::Ambiguous {
            stage_id: stage.stage_id.clone(),
            candidates: tied,
        });
    }
    Ok(best)
}

fn select_observers<'a>(
    request: &EditRequest,
    stage: &EditStageRequest,
    operator: &OperatorProfile,
    registry: &'a CapabilityRegistry,
) -> Result<Vec<(&'a ObserverProfile, String)>, CompileError> {
    let mut selected = Vec::new();

    for requirement in &stage.observations {
        let mut candidates: Vec<&ObserverProfile> = registry
            .observers
            .iter()
            .filter(|observer| {
                observer.qualification == QualificationStatus::Qualified
                    && (!request.policy.require_verified_rights
                        || observer.rights == RightsStatus::Verified)
                    && observer.modalities.contains(&stage.modality)
                    && observer.accepted_artifact_types.contains(&stage.output_artifact_type)
                    && observer.measured_metrics.contains(&requirement.metric_id)
                    && (!requirement.must_be_independent_from_mutator
                        || observer.independence_group != operator.independence_group)
            })
            .collect();
        candidates.sort_by_key(|profile| {
            (
                profile.estimated_latency_ms,
                profile.estimated_compute_units,
                profile.identity.clone(),
            )
        });

        let mut groups = BTreeSet::new();
        let mut observers_for_metric = Vec::new();
        for observer in candidates {
            if requirement.distinct_observer_groups
                && !groups.insert(observer.independence_group.clone())
            {
                continue;
            }
            observers_for_metric.push(observer);
            if observers_for_metric.len() == requirement.minimum_observers {
                break;
            }
        }

        if observers_for_metric.len() < requirement.minimum_observers {
            let reason = if requirement.must_be_independent_from_mutator {
                UnsupportedReason::NoIndependentObserver
            } else {
                UnsupportedReason::ObserverCoverageMissing
            };
            return Err(CompileError::Unsupported {
                subject: format!("{}:{}", stage.stage_id, requirement.metric_id),
                reason,
                detail: format!(
                    "needed {} qualified observer(s), found {}",
                    requirement.minimum_observers,
                    observers_for_metric.len()
                ),
            });
        }

        for observer in observers_for_metric {
            selected.push((observer, requirement.metric_id.clone()));
        }
    }
    Ok(selected)
}

/// Compile a normalized request into a static execution/evidence DAG.
///
/// No node is executed. execution_authorized is always false. A downstream
/// executor must independently verify the registry, source digests, authority,
/// preconditions, receipts, budgets and live branch/base versions.
pub fn compile_plan(
    request: &EditRequest,
    registry: &CapabilityRegistry,
) -> Result<CompiledPlan, CompileError> {
    validate_request(request)?;
    validate_registry(registry)?;
    let (stage_map, stage_order) = validate_stage_graph(request)?;
    let source_map: BTreeMap<String, ArtifactRef> = request
        .sources
        .iter()
        .cloned()
        .map(|source| (source.id.clone(), source))
        .collect();

    let mut selected_operators = BTreeMap::<String, &OperatorProfile>::new();
    let mut selected_observers = BTreeMap::<String, Vec<(&ObserverProfile, String)>>::new();
    let mut estimate = ResourceEstimate::default();

    for stage_id in &stage_order {
        let stage = stage_map.get(stage_id).expect("stage exists");
        let mut input_types: Vec<String> = stage
            .inputs
            .iter()
            .map(|input| match &input.origin {
                InputOrigin::SourceArtifact(source_id) => {
                    source_map.get(source_id).expect("validated source").artifact_type.clone()
                }
                InputOrigin::StageOutput(parent_id) => {
                    stage_map.get(parent_id).expect("validated parent").output_artifact_type.clone()
                }
            })
            .collect();
        input_types.sort();

        let operator = select_operator(request, stage, &input_types, registry)?;
        let observers = select_observers(request, stage, operator, registry)?;

        estimate.total_latency_ms =
            estimate.total_latency_ms.saturating_add(operator.estimated_latency_ms);
        estimate.total_compute_units =
            estimate.total_compute_units.saturating_add(operator.estimated_compute_units);
        estimate.maximum_mutation_risk_score =
            estimate.maximum_mutation_risk_score.max(operator.risk_score);

        for (observer, _) in &observers {
            estimate.total_latency_ms =
                estimate.total_latency_ms.saturating_add(observer.estimated_latency_ms);
            estimate.total_compute_units =
                estimate.total_compute_units.saturating_add(observer.estimated_compute_units);
        }

        selected_operators.insert(stage_id.clone(), operator);
        selected_observers.insert(stage_id.clone(), observers);
    }

    if estimate.total_latency_ms > request.resource_budget.maximum_total_latency_ms {
        return Err(CompileError::BudgetExceeded {
            resource: "total latency",
            estimated: estimate.total_latency_ms,
            maximum: request.resource_budget.maximum_total_latency_ms,
        });
    }
    if estimate.total_compute_units > request.resource_budget.maximum_total_compute_units {
        return Err(CompileError::BudgetExceeded {
            resource: "total compute",
            estimated: estimate.total_compute_units,
            maximum: request.resource_budget.maximum_total_compute_units,
        });
    }
    if estimate.maximum_mutation_risk_score > request.resource_budget.maximum_mutation_risk_score {
        return Err(CompileError::BudgetExceeded {
            resource: "maximum mutation risk",
            estimated: estimate.maximum_mutation_risk_score as u64,
            maximum: request.resource_budget.maximum_mutation_risk_score as u64,
        });
    }

    let mut nodes = Vec::<PlanNode>::new();
    let mut source_node_ids = BTreeMap::<String, String>::new();
    let mut stage_node_ids = BTreeMap::<String, String>::new();
    let mut operator_selections = Vec::new();
    let mut observer_selections = Vec::new();

    let mut sorted_sources = request.sources.clone();
    sorted_sources.sort_by(|left, right| left.id.cmp(&right.id));
    for source in &sorted_sources {
        let node_id = format!("verify-source:{}", atom(&source.id));
        source_node_ids.insert(source.id.clone(), node_id.clone());
        nodes.push(PlanNode {
            node_id,
            kind: PlanNodeKind::VerifySource,
            depends_on: Vec::new(),
            stage_id: None,
            profile: None,
            input_artifact_types: vec![source.artifact_type.clone()],
            output_artifact_type: Some("verified_artifact_reference".to_string()),
            evidence_metric: None,
            preconditions: BTreeSet::new(),
            postconditions: BTreeSet::from([
                "source_sha256_matches".to_string(),
                "source_type_matches".to_string(),
            ]),
            receipt_obligations: BTreeSet::from(["source_integrity_receipt".to_string()]),
            estimated_latency_ms: 0,
            estimated_compute_units: 0,
            risk_score: 0,
        });
    }

    for stage_id in &stage_order {
        let stage = stage_map.get(stage_id).expect("stage exists");
        let operator = selected_operators.get(stage_id).expect("operator selected");
        let node_id = format!("mutate:{}", atom(stage_id));
        let mut dependencies = BTreeSet::new();
        let mut input_types = Vec::new();

        for input in &stage.inputs {
            match &input.origin {
                InputOrigin::SourceArtifact(source_id) => {
                    dependencies.insert(source_node_ids.get(source_id).expect("source node").clone());
                }
                InputOrigin::StageOutput(parent_id) => {
                    dependencies.insert(stage_node_ids.get(parent_id).expect("parent node").clone());
                }
            }
            input_types.push(input.expected_artifact_type.clone());
        }
        input_types.sort();
        stage_node_ids.insert(stage_id.clone(), node_id.clone());

        operator_selections.push(OperatorSelection {
            stage_id: stage_id.clone(),
            profile: operator.identity.clone(),
            risk_score: operator.risk_score,
            latency_ms: operator.estimated_latency_ms,
            compute_units: operator.estimated_compute_units,
        });
        nodes.push(PlanNode {
            node_id: node_id.clone(),
            kind: PlanNodeKind::Mutate,
            depends_on: dependencies.into_iter().collect(),
            stage_id: Some(stage_id.clone()),
            profile: Some(operator.identity.clone()),
            input_artifact_types: input_types,
            output_artifact_type: Some(stage.output_artifact_type.clone()),
            evidence_metric: None,
            preconditions: stage.required_preconditions.clone(),
            postconditions: stage.required_postconditions.clone(),
            receipt_obligations: stage.required_receipts.clone(),
            estimated_latency_ms: operator.estimated_latency_ms,
            estimated_compute_units: operator.estimated_compute_units,
            risk_score: operator.risk_score,
        });

        for (observer, metric_id) in selected_observers.get(stage_id).expect("observers selected") {
            let observer_node_id = format!(
                "observe:{}:{}:{}:{}",
                atom(stage_id),
                atom(metric_id),
                atom(&observer.identity.id),
                atom(&observer.identity.version)
            );
            observer_selections.push(ObserverSelection {
                stage_id: stage_id.clone(),
                metric_id: metric_id.clone(),
                profile: observer.identity.clone(),
                independence_group: observer.independence_group.clone(),
                latency_ms: observer.estimated_latency_ms,
                compute_units: observer.estimated_compute_units,
            });
            nodes.push(PlanNode {
                node_id: observer_node_id,
                kind: PlanNodeKind::Observe,
                depends_on: vec![node_id.clone()],
                stage_id: Some(stage_id.clone()),
                profile: Some(observer.identity.clone()),
                input_artifact_types: vec![stage.output_artifact_type.clone()],
                output_artifact_type: Some("independent_observation".to_string()),
                evidence_metric: Some(metric_id.clone()),
                preconditions: BTreeSet::from(["candidate_artifact_digest_verified".to_string()]),
                postconditions: BTreeSet::from([format!("metric:{}:observed", metric_id)]),
                receipt_obligations: observer.emitted_receipts.clone(),
                estimated_latency_ms: observer.estimated_latency_ms,
                estimated_compute_units: observer.estimated_compute_units,
                risk_score: 0,
            });
        }
    }

    let final_dependencies = nodes
        .iter()
        .filter(|node| matches!(node.kind, PlanNodeKind::Mutate | PlanNodeKind::Observe))
        .map(|node| node.node_id.clone())
        .collect::<BTreeSet<_>>();
    nodes.push(PlanNode {
        node_id: format!("qualify:{}", atom(&request.request_id)),
        kind: PlanNodeKind::FinalQualification,
        depends_on: final_dependencies.into_iter().collect(),
        stage_id: None,
        profile: None,
        input_artifact_types: vec!["candidate_artifact_and_evidence_set".to_string()],
        output_artifact_type: Some("qualified_candidate_proposal".to_string()),
        evidence_metric: None,
        preconditions: BTreeSet::from([
            "all_mutation_nodes_succeeded".to_string(),
            "all_required_observations_passed".to_string(),
            "no_unknown_or_indeterminate_evidence".to_string(),
        ]),
        postconditions: BTreeSet::from(["plan_static_qualification_complete".to_string()]),
        receipt_obligations: BTreeSet::from(["compiled_plan_receipt".to_string()]),
        estimated_latency_ms: 0,
        estimated_compute_units: 0,
        risk_score: 0,
    });

    nodes = topological_sort_nodes(nodes)?;
    Ok(CompiledPlan {
        schema_version: PLAN_SCHEMA_VERSION.to_string(),
        request_id: request.request_id.clone(),
        registry_version: registry.registry_version.clone(),
        sources: sorted_sources,
        operator_selections,
        observer_selections,
        nodes,
        resource_estimate: estimate,
        execution_authorized: false,
        digest_algorithm: "SHA-256".to_string(),
        digest_input_encoding: CANONICAL_MANIFEST_ENCODING.to_string(),
    })
}

fn topological_sort_nodes(nodes: Vec<PlanNode>) -> Result<Vec<PlanNode>, CompileError> {
    let mut by_id = BTreeMap::new();
    for node in nodes {
        if by_id.insert(node.node_id.clone(), node).is_some() {
            return Err(CompileError::InvalidRequest(
                "compiler generated duplicate plan node ID".to_string(),
            ));
        }
    }

    let mut indegree: BTreeMap<String, usize> = by_id.keys().map(|id| (id.clone(), 0)).collect();
    let mut dependents: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for node in by_id.values() {
        for dependency in &node.depends_on {
            if !by_id.contains_key(dependency) {
                return Err(CompileError::InvalidRequest(format!(
                    "plan node {} references missing dependency {}",
                    node.node_id, dependency
                )));
            }
            *indegree.get_mut(&node.node_id).expect("node exists") += 1;
            dependents.entry(dependency.clone()).or_default().insert(node.node_id.clone());
        }
    }

    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(id.clone()))
        .collect();
    let mut ordered = Vec::with_capacity(by_id.len());
    while let Some(id) = ready.pop_first() {
        ordered.push(by_id.get(&id).expect("ready node exists").clone());
        if let Some(children) = dependents.get(&id) {
            for child in children {
                let degree = indegree.get_mut(child).expect("child exists");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(child.clone());
                }
            }
        }
    }
    if ordered.len() != by_id.len() {
        return Err(CompileError::InvalidRequest(
            "compiled node graph unexpectedly contains a cycle".to_string(),
        ));
    }
    Ok(ordered)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_DIGEST: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OPERATOR_DIGEST: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const OBSERVER_DIGEST: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const OTHER_DIGEST: &str =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn source(id: &str, artifact_type: &str) -> ArtifactRef {
        ArtifactRef {
            id: id.to_string(),
            artifact_type: artifact_type.to_string(),
            modality: Modality::Music,
            sha256: SOURCE_DIGEST.to_string(),
        }
    }

    fn operator(id: &str, scope: MutationScope) -> OperatorProfile {
        OperatorProfile {
            identity: ProfileIdentity {
                id: id.to_string(),
                version: "1.0.0".to_string(),
                artifact_sha256: OPERATOR_DIGEST.to_string(),
            },
            qualification: QualificationStatus::Qualified,
            rights: RightsStatus::Verified,
            supported_operations: BTreeSet::from(["remove_instrument".to_string()]),
            input_modalities: BTreeSet::from([Modality::Music]),
            input_artifact_types: BTreeSet::from(["audio/wav".to_string()]),
            output_artifact_type: "audio/wav".to_string(),
            capabilities: BTreeSet::from(["stem_isolation".to_string()]),
            maximum_mutation_scope: scope,
            preservation_guarantees: BTreeSet::from(["non_target_stems".to_string()]),
            emitted_receipts: BTreeSet::from(["render_execution_receipt".to_string()]),
            supported_preconditions: BTreeSet::from(["source_digest_verified".to_string()]),
            guaranteed_postconditions: BTreeSet::from(["requested_instrument_removed".to_string()]),
            controls: vec![NumericControlRange {
                control_id: "separation_strength".to_string(),
                minimum: 0.0,
                maximum: 1.0,
                unit: "ratio".to_string(),
            }],
            reproducibility: Reproducibility::Seeded,
            independence_group: "sam-family".to_string(),
            risk_score: 20,
            estimated_latency_ms: 500,
            estimated_compute_units: 100,
        }
    }

    fn observer(id: &str, group: &str) -> ObserverProfile {
        ObserverProfile {
            identity: ProfileIdentity {
                id: id.to_string(),
                version: "2.0.0".to_string(),
                artifact_sha256: OBSERVER_DIGEST.to_string(),
            },
            qualification: QualificationStatus::Qualified,
            rights: RightsStatus::Verified,
            modalities: BTreeSet::from([Modality::Music]),
            accepted_artifact_types: BTreeSet::from(["audio/wav".to_string()]),
            measured_metrics: BTreeSet::from(["target_stem_absence".to_string()]),
            independence_group: group.to_string(),
            emitted_receipts: BTreeSet::from(["acoustic_observation_receipt".to_string()]),
            estimated_latency_ms: 100,
            estimated_compute_units: 10,
        }
    }

    fn request() -> EditRequest {
        let stage = EditStageRequest {
            stage_id: "remove-vocals".to_string(),
            operation_id: "remove_instrument".to_string(),
            inputs: vec![InputBinding {
                input_id: "source_mix".to_string(),
                expected_artifact_type: "audio/wav".to_string(),
                origin: InputOrigin::SourceArtifact("mix-001".to_string()),
            }],
            output_artifact_type: "audio/wav".to_string(),
            modality: Modality::Music,
            maximum_permitted_mutation_scope: MutationScope::Stem,
            required_capabilities: BTreeSet::from(["stem_isolation".to_string()]),
            protected_properties: BTreeSet::from(["non_target_stems".to_string()]),
            required_receipts: BTreeSet::from(["render_execution_receipt".to_string()]),
            required_preconditions: BTreeSet::from(["source_digest_verified".to_string()]),
            required_postconditions: BTreeSet::from(["requested_instrument_removed".to_string()]),
            controls: vec![NumericControlRequirement {
                control_id: "separation_strength".to_string(),
                minimum: 0.2,
                maximum: 0.8,
                unit: "ratio".to_string(),
            }],
            observations: vec![ObservationRequirement {
                metric_id: "target_stem_absence".to_string(),
                minimum_observers: 1,
                must_be_independent_from_mutator: true,
                distinct_observer_groups: true,
            }],
            reproducibility_seed: Some(42),
            preferred_operator: None,
        };
        EditRequest {
            request_id: "request-001".to_string(),
            sources: vec![source("mix-001", "audio/wav")],
            stages: vec![stage],
            resource_budget: ResourceBudget {
                maximum_total_latency_ms: 2_000,
                maximum_total_compute_units: 1_000,
                maximum_mutation_risk_score: 50,
            },
            policy: PlannerPolicy {
                require_reproducible_operators: true,
                require_verified_rights: true,
            },
        }
    }

    fn registry() -> CapabilityRegistry {
        CapabilityRegistry {
            registry_version: "registry-2026-11-12-v1".to_string(),
            operators: vec![operator("stem-editor", MutationScope::Stem)],
            observers: vec![observer("spectral-auditor", "spectral-family")],
        }
    }

    #[test]
    fn compiles_a_qualified_edit_and_independent_observer() {
        let plan = compile_plan(&request(), &registry()).unwrap();
        assert!(!plan.execution_authorized);
        assert_eq!(plan.operator_selections.len(), 1);
        assert_eq!(plan.observer_selections.len(), 1);
        assert!(plan.nodes.iter().any(|node| node.kind == PlanNodeKind::VerifySource));
        assert!(plan.nodes.iter().any(|node| node.kind == PlanNodeKind::Mutate));
        assert!(plan.nodes.iter().any(|node| node.kind == PlanNodeKind::Observe));
        assert!(plan.nodes.iter().any(|node| node.kind == PlanNodeKind::FinalQualification));
    }

    #[test]
    fn rejects_an_operator_that_can_mutate_a_broader_scope() {
        let mut registry = registry();
        registry.operators[0].maximum_mutation_scope = MutationScope::Track;
        assert!(matches!(
            compile_plan(&request(), &registry),
            Err(CompileError::Unsupported { reason: UnsupportedReason::NoQualifiedOperator, .. })
        ));
    }

    #[test]
    fn rejects_missing_capability_without_fallback() {
        let mut registry = registry();
        registry.operators[0].capabilities.clear();
        assert!(matches!(compile_plan(&request(), &registry), Err(CompileError::Unsupported { .. })));
    }

    #[test]
    fn rejects_non_independent_observer() {
        let mut registry = registry();
        registry.observers[0].independence_group = "sam-family".to_string();
        assert!(matches!(
            compile_plan(&request(), &registry),
            Err(CompileError::Unsupported { reason: UnsupportedReason::NoIndependentObserver, .. })
        ));
    }

    #[test]
    fn rejects_missing_observer_metric_coverage() {
        let mut registry = registry();
        registry.observers.clear();
        assert!(matches!(compile_plan(&request(), &registry), Err(CompileError::Unsupported { .. })));
    }

    #[test]
    fn rejects_input_artifact_type_mismatch() {
        let mut request = request();
        request.stages[0].inputs[0].expected_artifact_type = "audio/flac".to_string();
        assert!(matches!(compile_plan(&request, &registry()), Err(CompileError::TypeMismatch { .. })));
    }

    #[test]
    fn rejects_missing_stage_dependency() {
        let mut request = request();
        request.stages[0].inputs[0].origin = InputOrigin::StageOutput("missing".to_string());
        assert!(matches!(compile_plan(&request, &registry()), Err(CompileError::InvalidRequest(_))));
    }

    #[test]
    fn detects_stage_dependency_cycles() {
        let mut request = request();
        request.stages.push(EditStageRequest {
            stage_id: "second".to_string(),
            inputs: vec![InputBinding {
                input_id: "first".to_string(),
                expected_artifact_type: "audio/wav".to_string(),
                origin: InputOrigin::StageOutput("remove-vocals".to_string()),
            }],
            ..request.stages[0].clone()
        });
        request.stages[0].inputs = vec![InputBinding {
            input_id: "second".to_string(),
            expected_artifact_type: "audio/wav".to_string(),
            origin: InputOrigin::StageOutput("second".to_string()),
        }];
        assert!(matches!(compile_plan(&request, &registry()), Err(CompileError::DependencyCycle { .. })));
    }

    #[test]
    fn rejects_latency_budget_overrun_before_plan_is_returned() {
        let mut request = request();
        request.resource_budget.maximum_total_latency_ms = 100;
        assert!(matches!(
            compile_plan(&request, &registry()),
            Err(CompileError::BudgetExceeded { resource: "total latency", .. })
        ));
    }

    #[test]
    fn rejects_unverified_operator_rights_when_policy_requires_them() {
        let mut registry = registry();
        registry.operators[0].rights = RightsStatus::Unknown;
        assert!(matches!(compile_plan(&request(), &registry), Err(CompileError::Unsupported { .. })));
    }

    #[test]
    fn rejects_provisional_operator_profiles() {
        let mut registry = registry();
        registry.operators[0].qualification = QualificationStatus::Provisional;
        assert!(matches!(compile_plan(&request(), &registry), Err(CompileError::Unsupported { .. })));
    }

    #[test]
    fn seeded_operator_requires_a_request_seed() {
        let mut request = request();
        request.stages[0].reproducibility_seed = None;
        assert!(matches!(compile_plan(&request, &registry()), Err(CompileError::Unsupported { .. })));
        request.stages[0].reproducibility_seed = Some(42);
        assert!(compile_plan(&request, &registry()).is_ok());
    }

    #[test]
    fn rejects_nondeterministic_operator_under_reproducibility_policy() {
        let mut registry = registry();
        registry.operators[0].reproducibility = Reproducibility::Nondeterministic;
        assert!(matches!(compile_plan(&request(), &registry), Err(CompileError::Unsupported { .. })));
    }

    #[test]
    fn equal_best_operators_are_ambiguous_without_user_preference() {
        let mut registry = registry();
        let mut second = operator("stem-editor-alt", MutationScope::Stem);
        second.identity.artifact_sha256 = OTHER_DIGEST.to_string();
        registry.operators.push(second);
        assert!(matches!(compile_plan(&request(), &registry), Err(CompileError::Ambiguous { .. })));
    }

    #[test]
    fn exact_operator_preference_resolves_a_qualified_tie() {
        let mut registry = registry();
        let mut second = operator("stem-editor-alt", MutationScope::Stem);
        second.identity.artifact_sha256 = OTHER_DIGEST.to_string();
        let preferred = second.identity.clone();
        registry.operators.push(second);
        let mut request = request();
        request.stages[0].preferred_operator = Some(preferred.clone());
        let plan = compile_plan(&request, &registry).unwrap();
        assert_eq!(plan.operator_selections[0].profile, preferred);
    }

    #[test]
    fn requested_control_range_must_be_fully_supported() {
        let mut registry = registry();
        registry.operators[0].controls[0].maximum = 0.5;
        assert!(matches!(compile_plan(&request(), &registry), Err(CompileError::Unsupported { .. })));
    }

    #[test]
    fn missing_postcondition_or_receipt_blocks_operator_selection() {
        let mut registry = registry();
        registry.operators[0].guaranteed_postconditions.clear();
        assert!(matches!(compile_plan(&request(), &registry), Err(CompileError::Unsupported { .. })));
    }

    #[test]
    fn canonical_manifest_is_independent_of_registry_vector_order() {
        let mut left = registry();
        let mut right = registry();
        let mut extra = operator("extra-but-not-selected", MutationScope::Mix);
        extra.identity.artifact_sha256 = OTHER_DIGEST.to_string();
        left.operators.push(extra.clone());
        left.operators.swap(0, 1);
        right.operators.push(extra);
        let first = compile_plan(&request(), &left).unwrap();
        let second = compile_plan(&request(), &right).unwrap();
        assert_eq!(first.canonical_manifest(), second.canonical_manifest());
    }

    #[test]
    fn every_dependency_precedes_its_consumer() {
        let mut request = request();
        request.stages[0].stage_id = "first".to_string();
        request.stages.push(EditStageRequest {
            stage_id: "second".to_string(),
            inputs: vec![InputBinding {
                input_id: "previous".to_string(),
                expected_artifact_type: "audio/wav".to_string(),
                origin: InputOrigin::StageOutput("first".to_string()),
            }],
            ..request.stages[0].clone()
        });
        let plan = compile_plan(&request, &registry()).unwrap();
        let positions: BTreeMap<String, usize> = plan
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.node_id.clone(), index))
            .collect();
        for node in &plan.nodes {
            for dependency in &node.depends_on {
                assert!(positions[dependency] < positions[&node.node_id]);
            }
        }
    }

    #[test]
    fn source_digest_must_be_lowercase_sha256() {
        let mut request = request();
        request.sources[0].sha256 = "not-a-digest".to_string();
        assert!(matches!(compile_plan(&request, &registry()), Err(CompileError::InvalidRequest(_))));
    }

    #[test]
    fn output_artifact_type_must_match_operator_output() {
        let mut request = request();
        request.stages[0].output_artifact_type = "audio/flac".to_string();
        assert!(matches!(compile_plan(&request, &registry()), Err(CompileError::Unsupported { .. })));
    }

    #[test]
    fn plan_never_authorizes_mutation_or_commit() {
        let plan = compile_plan(&request(), &registry()).unwrap();
        assert!(!plan.execution_authorized);
        assert!(plan.nodes.iter().any(|node| {
            node.kind == PlanNodeKind::FinalQualification
                && node.preconditions.contains("no_unknown_or_indeterminate_evidence")
        }));
    }

    #[test]
    fn operator_profile_digest_is_bound_into_canonical_manifest() {
        let first = compile_plan(&request(), &registry()).unwrap();
        let mut other_registry = registry();
        other_registry.operators[0].identity.artifact_sha256 = OTHER_DIGEST.to_string();
        let second = compile_plan(&request(), &other_registry).unwrap();
        assert_ne!(first.canonical_manifest(), second.canonical_manifest());
    }

    #[test]
    fn distinct_observer_groups_are_required_when_configured() {
        let mut request = request();
        request.stages[0].observations[0].minimum_observers = 2;
        let mut registry = registry();
        registry.observers.push(observer("spectral-auditor-2", "spectral-family"));
        assert!(matches!(
            compile_plan(&request, &registry),
            Err(CompileError::Unsupported { reason: UnsupportedReason::NoIndependentObserver, .. })
        ));
        registry.observers[1].independence_group = "symbolic-audit".to_string();
        assert!(compile_plan(&request, &registry).is_ok());
    }

    #[test]
    fn independent_stages_share_source_and_final_gate_waits_for_both() {
        let mut request = request();
        request.stages[0].stage_id = "remove-vocals".to_string();
        request.stages.push(EditStageRequest {
            stage_id: "remove-drums".to_string(),
            ..request.stages[0].clone()
        });
        let mut registry = registry();
        let mut drums = operator("drum-remover", MutationScope::Stem);
        drums.identity.artifact_sha256 = OTHER_DIGEST.to_string();
        drums.supported_operations = BTreeSet::from(["remove_drums".to_string()]);
        request.stages[1].operation_id = "remove_drums".to_string();
        registry.operators.push(drums);
        let plan = compile_plan(&request, &registry).unwrap();
        let gate = plan.nodes.iter().find(|node| node.kind == PlanNodeKind::FinalQualification).unwrap();
        assert!(gate.depends_on.contains(&"mutate:remove-vocals".to_string()));
        assert!(gate.depends_on.contains(&"mutate:remove-drums".to_string()));
    }

    #[test]
    fn registry_version_must_be_present() {
        let mut registry = registry();
        registry.registry_version.clear();
        assert!(matches!(compile_plan(&request(), &registry), Err(CompileError::InvalidRequest(_))));
    }
}
