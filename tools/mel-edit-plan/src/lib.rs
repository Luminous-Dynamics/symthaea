//! CQEP-001: compile a normalized musical/audio edit contract into a typed DAG.
//!
//! This pure-std reference crate does not parse natural language, invoke models,
//! execute audio operations, verify an operator's real-world capability, or
//! authorize mutation. It checks only the frozen declarations supplied to it.
//! Every plan must still be checked against current artifact digests and live
//! preconditions before any mutating node is admitted.

use std::collections::{BTreeMap, BTreeSet};

/// Stable identifiers are plain strings in v1; production adapters should
/// validate them against the published operation/capability registries.
pub type Id = String;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ArtifactKind {
    Audio,
    StemSet,
    Midi,
    Score,
    PerformanceSchedule,
    FeatureMap,
    Text,
    EditManifest,
}

impl ArtifactKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Audio => "audio",
            Self::StemSet => "stem_set",
            Self::Midi => "midi",
            Self::Score => "score",
            Self::PerformanceSchedule => "performance_schedule",
            Self::FeatureMap => "feature_map",
            Self::Text => "text",
            Self::EditManifest => "edit_manifest",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Scope {
    SampleRegion,
    Phrase,
    Stem,
    Global,
}

impl Scope {
    fn as_str(self) -> &'static str {
        match self {
            Self::SampleRegion => "sample_region",
            Self::Phrase => "phrase",
            Self::Stem => "stem",
            Self::Global => "global",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Risk {
    Low,
    Medium,
    High,
    Critical,
}

impl Risk {
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Determinism {
    Deterministic,
    Seeded,
    Nondeterministic,
}

impl Determinism {
    fn as_str(self) -> &'static str {
        match self {
            Self::Deterministic => "deterministic",
            Self::Seeded => "seeded",
            Self::Nondeterministic => "nondeterministic",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EvidenceKind {
    EditExecution,
    ProtectedPropertyPreservation,
    ArtifactIntegrity,
    PitchAndTiming,
    StemIsolation,
    LoudnessAndPeak,
    SemanticCompatibility,
    HumanReview,
}

impl EvidenceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::EditExecution => "edit_execution",
            Self::ProtectedPropertyPreservation => "protected_property_preservation",
            Self::ArtifactIntegrity => "artifact_integrity",
            Self::PitchAndTiming => "pitch_and_timing",
            Self::StemIsolation => "stem_isolation",
            Self::LoudnessAndPeak => "loudness_and_peak",
            Self::SemanticCompatibility => "semantic_compatibility",
            Self::HumanReview => "human_review",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactRef {
    pub kind: ArtifactKind,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageRequest {
    pub id: Id,
    pub operation: Id,
    pub input_kind: ArtifactKind,
    pub output_kind: ArtifactKind,
    pub dependencies: BTreeSet<Id>,
    pub scope: Scope,
    pub required_capabilities: BTreeSet<Id>,
    pub changed_properties: BTreeSet<Id>,
    pub forbidden_side_effects: BTreeSet<Id>,
    pub required_preconditions: BTreeSet<Id>,
    pub required_postconditions: BTreeSet<Id>,
    pub required_evidence: BTreeSet<EvidenceKind>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditBudget {
    pub max_cost_units: u64,
    pub max_latency_ms: u64,
    pub max_risk: Risk,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditRequest {
    pub request_id: Id,
    pub source: ArtifactRef,
    pub stages: Vec<StageRequest>,
    pub protected_properties: BTreeSet<Id>,
    pub forbidden_side_effects: BTreeSet<Id>,
    pub budget: EditBudget,
    pub deterministic_required: bool,
    pub intent_ambiguous: bool,
    pub required_final_artifact_kind: ArtifactKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorProfile {
    pub id: Id,
    pub version: Id,
    pub artifact_sha256: String,
    pub operations: BTreeSet<Id>,
    pub capabilities: BTreeSet<Id>,
    pub input_kinds: BTreeSet<ArtifactKind>,
    pub output_kind: ArtifactKind,
    /// Maximum collateral scope the operator may affect.
    pub maximum_locality: Scope,
    pub determinism: Determinism,
    pub risk: Risk,
    pub cost_units: u64,
    pub latency_ms: u64,
    pub writes_properties: BTreeSet<Id>,
    pub side_effects: BTreeSet<Id>,
    pub preconditions: BTreeSet<Id>,
    pub postconditions: BTreeSet<Id>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObserverProfile {
    pub id: Id,
    pub version: Id,
    pub artifact_sha256: String,
    pub observes_kinds: BTreeSet<ArtifactKind>,
    pub produces: BTreeSet<EvidenceKind>,
    pub risk: Risk,
    pub cost_units: u64,
    pub latency_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorRegistry {
    pub operators: Vec<OperatorProfile>,
    pub observers: Vec<ObserverProfile>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanErrorKind {
    InvalidRequestDigest,
    InvalidOperatorDigest,
    InvalidObserverDigest,
    AmbiguousIntent,
    EmptyPlan,
    DuplicateStageId,
    MissingDependency,
    DependencyCycle,
    UnsupportedOperator,
    ProtectedPropertyWrite,
    ForbiddenSideEffect,
    MissingPrecondition,
    MissingPostcondition,
    MissingObserver,
    InputTypeMismatch,
    FinalArtifactMismatch,
    BudgetExceeded,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanError {
    pub kind: PlanErrorKind,
    pub subject: Option<Id>,
    pub detail: &'static str,
}

impl PlanError {
    fn new(kind: PlanErrorKind, subject: Option<&str>, detail: &'static str) -> Self {
        Self {
            kind,
            subject: subject.map(str::to_owned),
            detail,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanNodeKind {
    MutatingOperator,
    IndependentObserver,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanNode {
    pub id: Id,
    pub kind: PlanNodeKind,
    pub stage_id: Id,
    pub profile_id: Id,
    pub input_kind: ArtifactKind,
    pub output_kind: Option<ArtifactKind>,
    pub dependencies: BTreeSet<Id>,
    pub required_evidence: BTreeSet<EvidenceKind>,
    pub provides_evidence: Option<EvidenceKind>,
    pub risk: Risk,
    pub cost_units: u64,
    pub latency_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledPlan {
    pub request_id: Id,
    pub source: ArtifactRef,
    pub final_artifact_kind: ArtifactKind,
    pub ordered_nodes: Vec<PlanNode>,
    pub total_cost_units: u64,
    pub total_latency_ms: u64,
    pub maximum_risk: Risk,
    /// Stable canonical bytes for an external SHA-256 boundary.
    pub canonical_bytes: Vec<u8>,
}

impl CompiledPlan {
    /// Bind an externally computed SHA-256 digest to the canonical bytes.
    /// This function validates syntax only; callers must compute the digest
    /// over canonical_bytes using an independent trusted hashing library.
    pub fn seal_sha256(&self, digest: &str) -> Result<PlanSeal, PlanError> {
        if !is_sha256(digest) {
            return Err(PlanError::new(
                PlanErrorKind::InvalidRequestDigest,
                Some(&self.request_id),
                "expected 64 lowercase hexadecimal SHA-256 characters",
            ));
        }
        Ok(PlanSeal {
            request_id: self.request_id.clone(),
            plan_sha256: digest.to_owned(),
            canonical_bytes: self.canonical_bytes.clone(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanSeal {
    pub request_id: Id,
    pub plan_sha256: String,
    pub canonical_bytes: Vec<u8>,
}

/// Compile a request using deterministic, declaration-only profile selection.
///
/// Candidate ordering is risk, cost, latency, then stable profile ID. A stage
/// requires independent evidence for every declared evidence kind. A
/// downstream mutating stage depends on all upstream observer nodes, so
/// evidence failure or indeterminacy blocks subsequent mutation at execution.
pub fn compile_plan(
    request: &EditRequest,
    registry: &OperatorRegistry,
) -> Result<CompiledPlan, PlanError> {
    if request.intent_ambiguous {
        return Err(PlanError::new(
            PlanErrorKind::AmbiguousIntent,
            Some(&request.request_id),
            "normalized intent is marked ambiguous",
        ));
    }
    if !is_sha256(&request.source.sha256) {
        return Err(PlanError::new(
            PlanErrorKind::InvalidRequestDigest,
            Some(&request.request_id),
            "source artifact digest is not canonical SHA-256",
        ));
    }
    if request.stages.is_empty() {
        return Err(PlanError::new(
            PlanErrorKind::EmptyPlan,
            Some(&request.request_id),
            "at least one edit stage is required",
        ));
    }

    let mut stages = BTreeMap::<Id, &StageRequest>::new();
    for stage in &request.stages {
        if stages.insert(stage.id.clone(), stage).is_some() {
            return Err(PlanError::new(
                PlanErrorKind::DuplicateStageId,
                Some(&stage.id),
                "stage identifiers must be unique",
            ));
        }
    }

    for stage in &request.stages {
        for dependency in &stage.dependencies {
            if !stages.contains_key(dependency) {
                return Err(PlanError::new(
                    PlanErrorKind::MissingDependency,
                    Some(&stage.id),
                    "dependency does not name a stage in this plan",
                ));
            }
            if dependency == &stage.id {
                return Err(PlanError::new(
                    PlanErrorKind::DependencyCycle,
                    Some(&stage.id),
                    "stage cannot depend on itself",
                ));
            }
        }
    }

    let topological_stages = topological_stage_order(&stages)?;
    validate_artifact_flow(request, &stages, &topological_stages)?;

    let mut selected_operators = BTreeMap::<Id, &OperatorProfile>::new();
    let mut selected_observers = BTreeMap::<(Id, EvidenceKind), &ObserverProfile>::new();
    let mut total_cost = 0u64;
    let mut total_latency = 0u64;
    let mut maximum_risk = Risk::Low;

    for stage_id in &topological_stages {
        let stage = stages[stage_id];
        if let Some(property) = stage
            .changed_properties
            .intersection(&request.protected_properties)
            .next()
        {
            return Err(PlanError::new(
                PlanErrorKind::ProtectedPropertyWrite,
                Some(property),
                "requested edit changes a property protected by the request",
            ));
        }

        let mut candidates: Vec<&OperatorProfile> = registry
            .operators
            .iter()
            .filter(|profile| operator_compatible(request, stage, profile))
            .collect();
        candidates.sort_by(|a, b| {
            (a.risk, a.cost_units, a.latency_ms, &a.id)
                .cmp(&(b.risk, b.cost_units, b.latency_ms, &b.id))
        });
        let selected = candidates.first().copied().ok_or_else(|| {
            PlanError::new(
                PlanErrorKind::UnsupportedOperator,
                Some(&stage.id),
                "no registered operator satisfies the complete stage contract",
            )
        })?;
        selected_operators.insert(stage.id.clone(), selected);

        total_cost = total_cost.saturating_add(selected.cost_units);
        total_latency = total_latency.saturating_add(selected.latency_ms);
        maximum_risk = maximum_risk.max(selected.risk);

        for evidence in &stage.required_evidence {
            let mut observers: Vec<&ObserverProfile> = registry
                .observers
                .iter()
                .filter(|observer| {
                    observer.observes_kinds.contains(&stage.output_kind)
                        && observer.produces.contains(evidence)
                        && observer.risk <= request.budget.max_risk
                        && is_sha256(&observer.artifact_sha256)
                })
                .collect();
            observers.sort_by(|a, b| {
                (a.risk, a.cost_units, a.latency_ms, &a.id)
                    .cmp(&(b.risk, b.cost_units, b.latency_ms, &b.id))
            });
            let observer = observers.first().copied().ok_or_else(|| {
                PlanError::new(
                    PlanErrorKind::MissingObserver,
                    Some(&stage.id),
                    "no independent observer covers a required evidence obligation",
                )
            })?;
            selected_observers.insert((stage.id.clone(), *evidence), observer);
            total_cost = total_cost.saturating_add(observer.cost_units);
            total_latency = total_latency.saturating_add(observer.latency_ms);
            maximum_risk = maximum_risk.max(observer.risk);
        }

        if total_cost > request.budget.max_cost_units
            || total_latency > request.budget.max_latency_ms
            || maximum_risk > request.budget.max_risk
        {
            return Err(PlanError::new(
                PlanErrorKind::BudgetExceeded,
                Some(&stage.id),
                "selected operators and observers exceed a declared request budget",
            ));
        }
    }

    let outgoing = outgoing_edges(&stages);
    let sinks: Vec<&StageRequest> = request
        .stages
        .iter()
        .filter(|stage| outgoing.get(&stage.id).is_none_or(BTreeSet::is_empty))
        .collect();
    if sinks.len() != 1 || sinks[0].output_kind != request.required_final_artifact_kind {
        return Err(PlanError::new(
            PlanErrorKind::FinalArtifactMismatch,
            Some(&request.request_id),
            "plan must have one sink with the requested final artifact type",
        ));
    }

    let mut nodes = Vec::<PlanNode>::new();
    let mut stage_observer_nodes = BTreeMap::<Id, BTreeSet<Id>>::new();

    for stage_id in &topological_stages {
        let stage = stages[stage_id];
        let operator = selected_operators[stage_id];
        let mut dependencies = BTreeSet::new();
        for parent in &stage.dependencies {
            dependencies.insert(format!("op:{parent}"));
            if let Some(observer_nodes) = stage_observer_nodes.get(parent) {
                dependencies.extend(observer_nodes.iter().cloned());
            }
        }
        if stage.dependencies.is_empty() {
            dependencies.insert(format!("source:{}", request.source.sha256));
        }
        let operation_node_id = format!("op:{}", stage.id);
        nodes.push(PlanNode {
            id: operation_node_id.clone(),
            kind: PlanNodeKind::MutatingOperator,
            stage_id: stage.id.clone(),
            profile_id: operator.id.clone(),
            input_kind: stage.input_kind,
            output_kind: Some(stage.output_kind),
            dependencies,
            required_evidence: stage.required_evidence.clone(),
            provides_evidence: None,
            risk: operator.risk,
            cost_units: operator.cost_units,
            latency_ms: operator.latency_ms,
        });

        let mut observer_ids = BTreeSet::new();
        for evidence in &stage.required_evidence {
            let observer = selected_observers[&(stage.id.clone(), *evidence)];
            let observer_node_id = format!("observe:{}:{}", stage.id, evidence.as_str());
            nodes.push(PlanNode {
                id: observer_node_id.clone(),
                kind: PlanNodeKind::IndependentObserver,
                stage_id: stage.id.clone(),
                profile_id: observer.id.clone(),
                input_kind: stage.output_kind,
                output_kind: None,
                dependencies: BTreeSet::from([operation_node_id.clone()]),
                required_evidence: BTreeSet::new(),
                provides_evidence: Some(*evidence),
                risk: observer.risk,
                cost_units: observer.cost_units,
                latency_ms: observer.latency_ms,
            });
            observer_ids.insert(observer_node_id);
        }
        stage_observer_nodes.insert(stage.id.clone(), observer_ids);
    }

    let ordered_nodes = topological_node_order(nodes)?;
    let canonical_bytes = canonical_plan_bytes(
        &request.request_id,
        &request.source,
        request.required_final_artifact_kind,
        &ordered_nodes,
        total_cost,
        total_latency,
        maximum_risk,
    );

    Ok(CompiledPlan {
        request_id: request.request_id.clone(),
        source: request.source.clone(),
        final_artifact_kind: request.required_final_artifact_kind,
        ordered_nodes,
        total_cost_units: total_cost,
        total_latency_ms: total_latency,
        maximum_risk,
        canonical_bytes,
    })
}

fn operator_compatible(
    request: &EditRequest,
    stage: &StageRequest,
    profile: &OperatorProfile,
) -> bool {
    let has_forbidden_side_effect = profile.side_effects.iter().any(|effect| {
        request.forbidden_side_effects.contains(effect)
            || stage.forbidden_side_effects.contains(effect)
    });
    let writes_unrequested_property = stage
        .changed_properties
        .iter()
        .any(|property| !profile.writes_properties.contains(property));
    let writes_protected_property = profile
        .writes_properties
        .iter()
        .any(|property| request.protected_properties.contains(property));
    let determinism_ok = !request.deterministic_required
        || profile.determinism != Determinism::Nondeterministic;

    profile.operations.contains(&stage.operation)
        && stage.required_capabilities.is_subset(&profile.capabilities)
        && profile.input_kinds.contains(&stage.input_kind)
        && profile.output_kind == stage.output_kind
        && profile.maximum_locality <= stage.scope
        && profile.risk <= request.budget.max_risk
        && profile.cost_units <= request.budget.max_cost_units
        && profile.latency_ms <= request.budget.max_latency_ms
        && is_sha256(&profile.artifact_sha256)
        && stage.required_preconditions.is_subset(&profile.preconditions)
        && stage.required_postconditions.is_subset(&profile.postconditions)
        && !writes_unrequested_property
        && !writes_protected_property
        && !has_forbidden_side_effect
        && determinism_ok
}

fn validate_artifact_flow(
    request: &EditRequest,
    stages: &BTreeMap<Id, &StageRequest>,
    topological: &[Id],
) -> Result<(), PlanError> {
    for id in topological {
        let stage = stages[id];
        if stage.dependencies.is_empty() {
            if stage.input_kind != request.source.kind {
                return Err(PlanError::new(
                    PlanErrorKind::InputTypeMismatch,
                    Some(&stage.id),
                    "root stage input type does not match the source artifact",
                ));
            }
        } else {
            for dependency in &stage.dependencies {
                if stages[dependency].output_kind != stage.input_kind {
                    return Err(PlanError::new(
                        PlanErrorKind::InputTypeMismatch,
                        Some(&stage.id),
                        "dependent stage input type does not match its parent output",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn topological_stage_order(
    stages: &BTreeMap<Id, &StageRequest>,
) -> Result<Vec<Id>, PlanError> {
    let mut indegree = BTreeMap::<Id, usize>::new();
    let mut outgoing = BTreeMap::<Id, BTreeSet<Id>>::new();
    for (id, stage) in stages {
        indegree.insert(id.clone(), stage.dependencies.len());
        for dependency in &stage.dependencies {
            outgoing
                .entry(dependency.clone())
                .or_default()
                .insert(id.clone());
        }
    }
    let mut ready: BTreeSet<Id> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut ordered = Vec::with_capacity(stages.len());
    while let Some(id) = ready.pop_first() {
        ordered.push(id.clone());
        if let Some(children) = outgoing.get(&id) {
            for child in children {
                if let Some(degree) = indegree.get_mut(child) {
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert(child.clone());
                    }
                }
            }
        }
    }
    if ordered.len() != stages.len() {
        return Err(PlanError::new(
            PlanErrorKind::DependencyCycle,
            None,
            "stage dependency graph contains a cycle",
        ));
    }
    Ok(ordered)
}

fn outgoing_edges(stages: &BTreeMap<Id, &StageRequest>) -> BTreeMap<Id, BTreeSet<Id>> {
    let mut outgoing = BTreeMap::new();
    for (id, stage) in stages {
        for dependency in &stage.dependencies {
            outgoing
                .entry(dependency.clone())
                .or_insert_with(BTreeSet::new)
                .insert(id.clone());
        }
    }
    outgoing
}

fn topological_node_order(nodes: Vec<PlanNode>) -> Result<Vec<PlanNode>, PlanError> {
    let by_id: BTreeMap<Id, PlanNode> =
        nodes.into_iter().map(|node| (node.id.clone(), node)).collect();
    let mut indegree = BTreeMap::<Id, usize>::new();
    let mut outgoing = BTreeMap::<Id, BTreeSet<Id>>::new();
    for (id, node) in &by_id {
        let internal_dependencies = node
            .dependencies
            .iter()
            .filter(|dependency| by_id.contains_key(*dependency))
            .count();
        indegree.insert(id.clone(), internal_dependencies);
        for dependency in &node.dependencies {
            if by_id.contains_key(dependency) {
                outgoing
                    .entry(dependency.clone())
                    .or_default()
                    .insert(id.clone());
            }
        }
    }
    let mut ready: BTreeSet<Id> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut ordered = Vec::with_capacity(by_id.len());
    while let Some(id) = ready.pop_first() {
        ordered.push(by_id[&id].clone());
        if let Some(children) = outgoing.get(&id) {
            for child in children {
                if let Some(degree) = indegree.get_mut(child) {
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert(child.clone());
                    }
                }
            }
        }
    }
    if ordered.len() != by_id.len() {
        return Err(PlanError::new(
            PlanErrorKind::DependencyCycle,
            None,
            "compiled plan graph contains a cycle",
        ));
    }
    Ok(ordered)
}

fn canonical_plan_bytes(
    request_id: &str,
    source: &ArtifactRef,
    final_kind: ArtifactKind,
    nodes: &[PlanNode],
    total_cost: u64,
    total_latency: u64,
    maximum_risk: Risk,
) -> Vec<u8> {
    let mut out = String::from("MEL-EDIT-PLAN-V1\n");
    field(&mut out, request_id);
    field(&mut out, source.kind.as_str());
    field(&mut out, &source.sha256);
    field(&mut out, final_kind.as_str());
    field(&mut out, &total_cost.to_string());
    field(&mut out, &total_latency.to_string());
    field(&mut out, maximum_risk.as_str());
    for node in nodes {
        field(&mut out, node.id.as_str());
        field(
            &mut out,
            match node.kind {
                PlanNodeKind::MutatingOperator => "mutating_operator",
                PlanNodeKind::IndependentObserver => "independent_observer",
            },
        );
        field(&mut out, &node.stage_id);
        field(&mut out, &node.profile_id);
        field(&mut out, node.input_kind.as_str());
        field(
            &mut out,
            node.output_kind.map_or("none", ArtifactKind::as_str),
        );
        for dependency in &node.dependencies {
            field(&mut out, dependency);
        }
        field(&mut out, "end-dependencies");
        for evidence in &node.required_evidence {
            field(&mut out, evidence.as_str());
        }
        field(
            &mut out,
            node.provides_evidence.map_or("none", EvidenceKind::as_str),
        );
        field(&mut out, node.risk.as_str());
        field(&mut out, &node.cost_units.to_string());
        field(&mut out, &node.latency_ms.to_string());
        field(&mut out, "end-node");
    }
    out.into_bytes()
}

fn field(output: &mut String, value: &str) {
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
    output.push('\n');
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const DIGEST_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn request() -> EditRequest {
        EditRequest {
            request_id: "fixture-request-1".into(),
            source: ArtifactRef { kind: ArtifactKind::Audio, sha256: DIGEST_A.into() },
            stages: vec![StageRequest {
                id: "edit".into(),
                operation: "remove_vocal".into(),
                input_kind: ArtifactKind::Audio,
                output_kind: ArtifactKind::Audio,
                dependencies: BTreeSet::new(),
                scope: Scope::Stem,
                required_capabilities: BTreeSet::from(["source_separation".into()]),
                changed_properties: BTreeSet::from(["vocal_presence".into()]),
                forbidden_side_effects: BTreeSet::from(["duration_change".into()]),
                required_preconditions: BTreeSet::from(["source_available".into()]),
                required_postconditions: BTreeSet::from(["duration_preserved".into()]),
                required_evidence: BTreeSet::from([
                    EvidenceKind::EditExecution,
                    EvidenceKind::ProtectedPropertyPreservation,
                    EvidenceKind::ArtifactIntegrity,
                ]),
            }],
            protected_properties: BTreeSet::from(["duration".into(), "mix_timing".into()]),
            forbidden_side_effects: BTreeSet::from(["duration_change".into()]),
            budget: EditBudget {
                max_cost_units: 20,
                max_latency_ms: 10_000,
                max_risk: Risk::High,
            },
            deterministic_required: false,
            intent_ambiguous: false,
            required_final_artifact_kind: ArtifactKind::Audio,
        }
    }

    fn operator() -> OperatorProfile {
        OperatorProfile {
            id: "sam-audio-large".into(),
            version: "2026.10".into(),
            artifact_sha256: DIGEST_B.into(),
            operations: BTreeSet::from(["remove_vocal".into()]),
            capabilities: BTreeSet::from(["source_separation".into()]),
            input_kinds: BTreeSet::from([ArtifactKind::Audio]),
            output_kind: ArtifactKind::Audio,
            maximum_locality: Scope::Stem,
            determinism: Determinism::Seeded,
            risk: Risk::Medium,
            cost_units: 10,
            latency_ms: 4_000,
            writes_properties: BTreeSet::from(["vocal_presence".into()]),
            side_effects: BTreeSet::new(),
            preconditions: BTreeSet::from(["source_available".into()]),
            postconditions: BTreeSet::from(["duration_preserved".into()]),
        }
    }

    fn observer(id: &str, kind: EvidenceKind) -> ObserverProfile {
        ObserverProfile {
            id: id.into(),
            version: "v1".into(),
            artifact_sha256: DIGEST_C.into(),
            observes_kinds: BTreeSet::from([ArtifactKind::Audio]),
            produces: BTreeSet::from([kind]),
            risk: Risk::Low,
            cost_units: 1,
            latency_ms: 100,
        }
    }

    fn registry() -> OperatorRegistry {
        OperatorRegistry {
            operators: vec![operator()],
            observers: vec![
                observer("edit-observer", EvidenceKind::EditExecution),
                observer("preservation-observer", EvidenceKind::ProtectedPropertyPreservation),
                observer("integrity-observer", EvidenceKind::ArtifactIntegrity),
            ],
        }
    }

    #[test]
    fn compiles_typed_operation_with_independent_observers() {
        let plan = compile_plan(&request(), &registry()).unwrap();
        assert_eq!(plan.ordered_nodes.len(), 4);
        assert_eq!(plan.total_cost_units, 13);
        assert_eq!(plan.total_latency_ms, 4_300);
        assert_eq!(plan.final_artifact_kind, ArtifactKind::Audio);
        assert_eq!(plan.ordered_nodes[0].kind, PlanNodeKind::MutatingOperator);
        for observer in &plan.ordered_nodes[1..] {
            assert_eq!(observer.kind, PlanNodeKind::IndependentObserver);
            assert_eq!(observer.dependencies, BTreeSet::from(["op:edit".into()]));
        }
    }

    #[test]
    fn canonical_bytes_are_deterministic() {
        let left = compile_plan(&request(), &registry()).unwrap();
        let right = compile_plan(&request(), &registry()).unwrap();
        assert_eq!(left.canonical_bytes, right.canonical_bytes);
    }

    #[test]
    fn plan_seal_checks_sha256_syntax_but_does_not_compute_it() {
        let plan = compile_plan(&request(), &registry()).unwrap();
        assert!(plan.seal_sha256(DIGEST_C).is_ok());
        assert_eq!(
            plan.seal_sha256("not-a-digest").unwrap_err().kind,
            PlanErrorKind::InvalidRequestDigest
        );
    }

    #[test]
    fn rejects_ambiguous_intent_before_operator_selection() {
        let mut request = request();
        request.intent_ambiguous = true;
        assert_eq!(
            compile_plan(&request, &registry()).unwrap_err().kind,
            PlanErrorKind::AmbiguousIntent
        );
    }

    #[test]
    fn rejects_missing_capability_instead_of_falling_back() {
        let mut request = request();
        request.stages[0].required_capabilities.insert("isolated_execution".into());
        assert_eq!(
            compile_plan(&request, &registry()).unwrap_err().kind,
            PlanErrorKind::UnsupportedOperator
        );
    }

    #[test]
    fn rejects_unsupported_operation() {
        let mut request = request();
        request.stages[0].operation = "per_instrument_pitch_shift".into();
        assert_eq!(
            compile_plan(&request, &registry()).unwrap_err().kind,
            PlanErrorKind::UnsupportedOperator
        );
    }

    #[test]
    fn rejects_operator_that_can_expand_beyond_requested_locality() {
        let mut request = request();
        request.stages[0].scope = Scope::SampleRegion;
        assert_eq!(
            compile_plan(&request, &registry()).unwrap_err().kind,
            PlanErrorKind::UnsupportedOperator
        );
    }

    #[test]
    fn rejects_write_to_a_protected_property() {
        let mut request = request();
        request.protected_properties.insert("vocal_presence".into());
        assert_eq!(
            compile_plan(&request, &registry()).unwrap_err().kind,
            PlanErrorKind::ProtectedPropertyWrite
        );
    }

    #[test]
    fn rejects_forbidden_side_effect() {
        let mut registry = registry();
        registry.operators[0].side_effects.insert("duration_change".into());
        assert_eq!(
            compile_plan(&request(), &registry).unwrap_err().kind,
            PlanErrorKind::UnsupportedOperator
        );
    }

    #[test]
    fn rejects_missing_independent_evidence() {
        let mut registry = registry();
        registry.observers.pop();
        assert_eq!(
            compile_plan(&request(), &registry).unwrap_err().kind,
            PlanErrorKind::MissingObserver
        );
    }

    #[test]
    fn rejects_non_deterministic_operator_when_determinism_is_required() {
        let mut request = request();
        request.deterministic_required = true;
        let mut registry = registry();
        registry.operators[0].determinism = Determinism::Nondeterministic;
        assert_eq!(
            compile_plan(&request, &registry).unwrap_err().kind,
            PlanErrorKind::UnsupportedOperator
        );
    }

    #[test]
    fn rejects_stale_source_digest() {
        let mut request = request();
        request.source.sha256 = "not-a-digest".into();
        assert_eq!(
            compile_plan(&request, &registry()).unwrap_err().kind,
            PlanErrorKind::InvalidRequestDigest
        );
    }

    #[test]
    fn rejects_budget_overrun_including_observers() {
        let mut request = request();
        request.budget.max_cost_units = 12;
        assert_eq!(
            compile_plan(&request, &registry()).unwrap_err().kind,
            PlanErrorKind::BudgetExceeded
        );
    }

    #[test]
    fn rejects_dependency_cycle() {
        let mut request = request();
        request.stages[0].dependencies.insert("second".into());
        request.stages.push(StageRequest {
            id: "second".into(),
            operation: "other".into(),
            input_kind: ArtifactKind::Audio,
            output_kind: ArtifactKind::Audio,
            dependencies: BTreeSet::from(["edit".into()]),
            scope: Scope::Stem,
            required_capabilities: BTreeSet::new(),
            changed_properties: BTreeSet::new(),
            forbidden_side_effects: BTreeSet::new(),
            required_preconditions: BTreeSet::new(),
            required_postconditions: BTreeSet::new(),
            required_evidence: BTreeSet::new(),
        });
        assert_eq!(
            compile_plan(&request, &registry()).unwrap_err().kind,
            PlanErrorKind::DependencyCycle
        );
    }

    #[test]
    fn rejects_missing_dependency() {
        let mut request = request();
        request.stages[0].dependencies.insert("missing".into());
        assert_eq!(
            compile_plan(&request, &registry()).unwrap_err().kind,
            PlanErrorKind::MissingDependency
        );
    }

    #[test]
    fn rejects_artifact_type_mismatch() {
        let mut request = request();
        request.stages[0].input_kind = ArtifactKind::Midi;
        assert_eq!(
            compile_plan(&request, &registry()).unwrap_err().kind,
            PlanErrorKind::InputTypeMismatch
        );
    }

    #[test]
    fn rejects_wrong_final_artifact_kind() {
        let mut request = request();
        request.required_final_artifact_kind = ArtifactKind::StemSet;
        assert_eq!(
            compile_plan(&request, &registry()).unwrap_err().kind,
            PlanErrorKind::FinalArtifactMismatch
        );
    }

    #[test]
    fn observer_evidence_blocks_downstream_mutation() {
        let mut request = request();
        request.stages.push(StageRequest {
            id: "normalize".into(),
            operation: "loudness_normalize".into(),
            input_kind: ArtifactKind::Audio,
            output_kind: ArtifactKind::Audio,
            dependencies: BTreeSet::from(["edit".into()]),
            scope: Scope::Global,
            required_capabilities: BTreeSet::from(["loudness_control".into()]),
            changed_properties: BTreeSet::from(["integrated_loudness".into()]),
            forbidden_side_effects: BTreeSet::new(),
            required_preconditions: BTreeSet::new(),
            required_postconditions: BTreeSet::new(),
            required_evidence: BTreeSet::from([EvidenceKind::ArtifactIntegrity]),
        });
        let mut registry = registry();
        registry.operators.push(OperatorProfile {
            id: "loudness-normalizer".into(),
            version: "v1".into(),
            artifact_sha256: DIGEST_C.into(),
            operations: BTreeSet::from(["loudness_normalize".into()]),
            capabilities: BTreeSet::from(["loudness_control".into()]),
            input_kinds: BTreeSet::from([ArtifactKind::Audio]),
            output_kind: ArtifactKind::Audio,
            maximum_locality: Scope::Global,
            determinism: Determinism::Deterministic,
            risk: Risk::Low,
            cost_units: 2,
            latency_ms: 500,
            writes_properties: BTreeSet::from(["integrated_loudness".into()]),
            side_effects: BTreeSet::new(),
            preconditions: BTreeSet::new(),
            postconditions: BTreeSet::new(),
        });
        let plan = compile_plan(&request, &registry).unwrap();
        let normalize = plan.ordered_nodes.iter().find(|node| node.id == "op:normalize").unwrap();
        assert!(normalize.dependencies.contains(&"op:edit".into()));
        assert!(normalize.dependencies.contains(&"observe:edit:edit_execution".into()));
        assert!(normalize.dependencies.contains(&"observe:edit:protected_property_preservation".into()));
        assert!(normalize.dependencies.contains(&"observe:edit:artifact_integrity".into()));
    }
}
