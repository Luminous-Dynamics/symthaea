// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SCI-014A qualification fixture for owner-reconstructed scientific replay.
//!
//! This is deliberately an integration qualification, not a production API.
//! Synthetic fixtures prove composition without minting production-named
//! scientific replay capabilities.

use std::collections::{BTreeMap, BTreeSet};

use symthaea_scientific_replay::{
    DispositionEvaluationContextV1, DispositionEvaluationInputsV1, DispositionPolicyV1,
    DispositionRuleV1, PersistedDispositionEvaluationRecordV1, PredicateFactV1, PredicateValueV1,
    ReplayErrorV1, ReplayVerifiedDispositionEvaluationV1, RuleRequirementV1,
    build_persisted_evaluation_record_v1, evaluate_disposition_material_v1,
    verify_persisted_disposition_evaluation_v1,
};

const FIXTURE_RECONSTRUCTION_PROFILE: &str = "symthaea.science.qualified-reconstruction.fixture.v1";
const FIXTURE_DERIVATION_PROFILE: &str = "symthaea.science.predicate-derivation.fixture.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContributionRole {
    Support,
    Opposition,
    ActiveDefeater,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Candidate {
    id: String,
    reason_id: String,
    role: ContributionRole,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Accounting {
    Admitted,
    Excluded(String),
    Unresolved(String),
    AliasOf(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AccountingEntry {
    candidate_id: String,
    terminal: Accounting,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReasonNode {
    reason_id: String,
    candidate_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasonRelation {
    Supports,
    Opposes,
    Undercuts,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReasonEdge {
    from: String,
    to: String,
    relation: ReasonRelation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PredicateOperator {
    AnyAdmitted(ContributionRole),
    NoAdmitted(ContributionRole),
    AllAccounted,
    Mirror(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PredicateSpec {
    predicate_id: String,
    dependencies: Vec<String>,
    reason_ids: Vec<String>,
    operator: PredicateOperator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PolicyFixture {
    profile_id: String,
    rules: Vec<DispositionRuleV1>,
    fallback_output: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ScientificLineageFixture {
    proposition: String,
    scientific_use: String,
    evidence_scope: String,
    discovery_policy: String,
    source_registry_generation: String,
    source_lifecycle_generation: String,
    external_argument_generation: String,
    dependency_graph_generation: String,
    target_compatibility_generation: String,
    triangulation_generation: String,
    information_cutoff: String,
    predicate_derivation_execution_lineage: String,
    owner_reconstruction_artifact: String,
    owner_reconstruction_execution_lineage: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReconstructionFixture {
    lineage: ScientificLineageFixture,
    candidates: Vec<Candidate>,
    accounting: Vec<AccountingEntry>,
    reason_nodes: Vec<ReasonNode>,
    reason_edges: Vec<ReasonEdge>,
    predicate_specs: Vec<PredicateSpec>,
    policy: PolicyFixture,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PredicateReceipt {
    predicate_id: String,
    result: PredicateValueV1,
    predicate_spec_id: String,
    reason_subgraph_id: String,
    dependency_ids: Vec<String>,
    consulted_candidate_ids: Vec<String>,
    closed_world_receipt_id: Option<String>,
    derivation_profile_id: String,
    derivation_artifact_id: String,
    derivation_execution_lineage_id: String,
}

#[derive(Debug, PartialEq, Eq)]
struct QualifiedFixtureReconstruction {
    material_id: String,
    closed_view_receipt_id: String,
    predicate_dependency_graph_id: String,
    receipts: Vec<PredicateReceipt>,
    inputs: DispositionEvaluationInputsV1,
    policy: DispositionPolicyV1,
}

#[derive(Debug, PartialEq, Eq)]
struct FixtureReplayWitness {
    reconstruction_material_id: String,
    lower_replay: ReplayVerifiedDispositionEvaluationV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QualificationError {
    EmptyField,
    DuplicateCandidate,
    DuplicateAccounting,
    AccountingMismatch,
    InvalidAccounting,
    AliasTargetMissing,
    AliasSelfReference,
    AliasChain,
    AliasSemanticMismatch,
    DuplicateReason,
    UnknownReasonCandidate,
    CandidateReasonMismatch,
    DuplicateReasonEdge,
    UnknownReasonEndpoint,
    DuplicatePredicateSpec,
    DuplicatePredicateDependency,
    DuplicatePredicateReason,
    UnknownPredicateDependency,
    UnknownPredicateReason,
    OperatorDependencyMismatch,
    PredicateDependencyCycle,
    MissingPolicyPredicate,
    ExtraPredicateSpec,
    Lower(ReplayErrorV1),
}

impl From<ReplayErrorV1> for QualificationError {
    fn from(value: ReplayErrorV1) -> Self {
        Self::Lower(value)
    }
}

#[derive(Debug)]
struct QualifiedReasonTopology {
    nodes: BTreeMap<String, Option<String>>,
    edges: Vec<ReasonEdge>,
    material_id: String,
}

struct DerivationEnvironment<'a> {
    candidates: &'a BTreeMap<String, Candidate>,
    accounting: &'a BTreeMap<String, Accounting>,
    reason_topology: &'a QualifiedReasonTopology,
    closed_view_receipt_id: &'a str,
    derivation_artifact_id: &'a str,
    derivation_execution_lineage_id: &'a str,
}

fn require_nonempty(value: &str) -> Result<(), QualificationError> {
    if value.trim().is_empty() {
        Err(QualificationError::EmptyField)
    } else {
        Ok(())
    }
}

/// Fixture representation only: deterministic and domain-labeled, but not a
/// cryptographic digest or production content-identity theorem.
fn exact_id(kind: &str, parts: impl IntoIterator<Item = String>) -> String {
    let mut encoded = format!("symthaea.science.fixture.{kind}.v1|");
    for part in parts {
        encoded.push_str(&part.len().to_string());
        encoded.push(':');
        encoded.push_str(&part);
        encoded.push('|');
    }
    encoded
}

fn push_section(parts: &mut Vec<String>, name: &str, values: impl IntoIterator<Item = String>) {
    let values = values.into_iter().collect::<Vec<_>>();
    parts.push(format!("section:{name}"));
    parts.push(values.len().to_string());
    parts.extend(values);
}

fn role_code(role: ContributionRole) -> &'static str {
    match role {
        ContributionRole::Support => "support",
        ContributionRole::Opposition => "opposition",
        ContributionRole::ActiveDefeater => "active-defeater",
    }
}

fn relation_code(relation: ReasonRelation) -> &'static str {
    match relation {
        ReasonRelation::Supports => "supports",
        ReasonRelation::Opposes => "opposes",
        ReasonRelation::Undercuts => "undercuts",
    }
}

fn value_code(value: PredicateValueV1) -> &'static str {
    match value {
        PredicateValueV1::Satisfied => "satisfied",
        PredicateValueV1::NotSatisfied => "not-satisfied",
        PredicateValueV1::Unknown => "unknown",
        PredicateValueV1::NotApplicable => "not-applicable",
        PredicateValueV1::Blocked => "blocked",
    }
}

fn accounting_code(terminal: &Accounting) -> String {
    match terminal {
        Accounting::Admitted => "admitted".to_string(),
        Accounting::Excluded(reason) => format!("excluded:{reason}"),
        Accounting::Unresolved(reason) => format!("unresolved:{reason}"),
        Accounting::AliasOf(target) => format!("alias-of:{target}"),
    }
}

fn operator_code(operator: &PredicateOperator) -> String {
    match operator {
        PredicateOperator::AnyAdmitted(role) => format!("any-admitted:{}", role_code(*role)),
        PredicateOperator::NoAdmitted(role) => format!("no-admitted:{}", role_code(*role)),
        PredicateOperator::AllAccounted => "all-accounted".to_string(),
        PredicateOperator::Mirror(source) => format!("mirror:{source}"),
    }
}

fn canonical_candidates(
    fixture: &ReconstructionFixture,
) -> Result<BTreeMap<String, Candidate>, QualificationError> {
    let mut candidates = BTreeMap::new();
    for candidate in &fixture.candidates {
        require_nonempty(&candidate.id)?;
        require_nonempty(&candidate.reason_id)?;
        if candidates
            .insert(candidate.id.clone(), candidate.clone())
            .is_some()
        {
            return Err(QualificationError::DuplicateCandidate);
        }
    }
    Ok(candidates)
}

fn canonical_accounting(
    fixture: &ReconstructionFixture,
    candidates: &BTreeMap<String, Candidate>,
) -> Result<BTreeMap<String, Accounting>, QualificationError> {
    let mut accounting = BTreeMap::new();
    for entry in &fixture.accounting {
        require_nonempty(&entry.candidate_id)?;
        match &entry.terminal {
            Accounting::Excluded(reason) | Accounting::Unresolved(reason) => {
                require_nonempty(reason).map_err(|_| QualificationError::InvalidAccounting)?;
            }
            Accounting::AliasOf(target) => require_nonempty(target)?,
            Accounting::Admitted => {}
        }
        if accounting
            .insert(entry.candidate_id.clone(), entry.terminal.clone())
            .is_some()
        {
            return Err(QualificationError::DuplicateAccounting);
        }
    }

    let candidate_ids = candidates.keys().cloned().collect::<BTreeSet<_>>();
    let accounting_ids = accounting.keys().cloned().collect::<BTreeSet<_>>();
    if candidate_ids != accounting_ids {
        return Err(QualificationError::AccountingMismatch);
    }

    for (candidate_id, terminal) in &accounting {
        if let Accounting::AliasOf(target) = terminal {
            if candidate_id == target {
                return Err(QualificationError::AliasSelfReference);
            }
            let Some(target_terminal) = accounting.get(target) else {
                return Err(QualificationError::AliasTargetMissing);
            };
            if matches!(target_terminal, Accounting::AliasOf(_)) {
                return Err(QualificationError::AliasChain);
            }
            let alias = candidates
                .get(candidate_id)
                .expect("closed accounting references registered candidate");
            let canonical = candidates
                .get(target)
                .expect("alias target exists in closed candidate universe");
            if alias.role != canonical.role {
                return Err(QualificationError::AliasSemanticMismatch);
            }
        }
    }
    Ok(accounting)
}

fn closed_view_ids(
    fixture: &ReconstructionFixture,
    candidates: &BTreeMap<String, Candidate>,
    accounting: &BTreeMap<String, Accounting>,
) -> (String, String, String) {
    let view_id = exact_id(
        "closed-view-definition",
        [
            "proposition".to_string(),
            fixture.lineage.proposition.clone(),
            "scientific-use".to_string(),
            fixture.lineage.scientific_use.clone(),
            "scope".to_string(),
            fixture.lineage.evidence_scope.clone(),
            "discovery-policy".to_string(),
            fixture.lineage.discovery_policy.clone(),
            "source-registry-generation".to_string(),
            fixture.lineage.source_registry_generation.clone(),
            "cutoff".to_string(),
            fixture.lineage.information_cutoff.clone(),
        ],
    );

    let mut snapshot_parts = vec![view_id.clone()];
    let mut candidate_parts = Vec::new();
    for candidate in candidates.values() {
        candidate_parts.extend([
            candidate.id.clone(),
            candidate.reason_id.clone(),
            role_code(candidate.role).to_string(),
        ]);
    }
    push_section(&mut snapshot_parts, "candidates", candidate_parts);

    let mut accounting_parts = Vec::new();
    for (candidate_id, terminal) in accounting {
        accounting_parts.extend([candidate_id.clone(), accounting_code(terminal)]);
    }
    push_section(&mut snapshot_parts, "accounting", accounting_parts);

    let snapshot_id = exact_id("closed-view-snapshot", snapshot_parts);
    let receipt_id = exact_id(
        "closed-view-receipt",
        [
            "snapshot".to_string(),
            snapshot_id.clone(),
            "candidate-count".to_string(),
            candidates.len().to_string(),
            "accounting-count".to_string(),
            accounting.len().to_string(),
        ],
    );
    (view_id, snapshot_id, receipt_id)
}

fn qualify_reason_topology(
    fixture: &ReconstructionFixture,
    candidates: &BTreeMap<String, Candidate>,
) -> Result<QualifiedReasonTopology, QualificationError> {
    let mut nodes = BTreeMap::new();
    for node in &fixture.reason_nodes {
        require_nonempty(&node.reason_id)?;
        if let Some(candidate_id) = &node.candidate_id
            && !candidates.contains_key(candidate_id)
        {
            return Err(QualificationError::UnknownReasonCandidate);
        }
        if nodes
            .insert(node.reason_id.clone(), node.candidate_id.clone())
            .is_some()
        {
            return Err(QualificationError::DuplicateReason);
        }
    }

    for candidate in candidates.values() {
        if nodes.get(&candidate.reason_id) != Some(&Some(candidate.id.clone())) {
            return Err(QualificationError::CandidateReasonMismatch);
        }
    }

    let mut edges = fixture.reason_edges.clone();
    edges.sort_by_key(|edge| {
        (
            edge.from.clone(),
            edge.to.clone(),
            relation_code(edge.relation).to_string(),
        )
    });
    let mut edge_ids = BTreeSet::new();
    for edge in &edges {
        if !nodes.contains_key(&edge.from) || !nodes.contains_key(&edge.to) {
            return Err(QualificationError::UnknownReasonEndpoint);
        }
        let edge_id = exact_id(
            "reason-edge",
            [
                edge.from.clone(),
                edge.to.clone(),
                relation_code(edge.relation).to_string(),
            ],
        );
        if !edge_ids.insert(edge_id) {
            return Err(QualificationError::DuplicateReasonEdge);
        }
    }

    let mut parts = Vec::new();
    let mut node_parts = Vec::new();
    for (reason_id, candidate_id) in &nodes {
        node_parts.extend([
            reason_id.clone(),
            candidate_id.clone().unwrap_or_else(|| "none".to_string()),
        ]);
    }
    push_section(&mut parts, "nodes", node_parts);
    let mut edge_parts = Vec::new();
    for edge in &edges {
        edge_parts.extend([
            edge.from.clone(),
            edge.to.clone(),
            relation_code(edge.relation).to_string(),
        ]);
    }
    push_section(&mut parts, "edges", edge_parts);

    Ok(QualifiedReasonTopology {
        nodes,
        edges,
        material_id: exact_id("reason-topology", parts),
    })
}

fn canonical_specs(
    fixture: &ReconstructionFixture,
) -> Result<BTreeMap<String, PredicateSpec>, QualificationError> {
    let mut specs = BTreeMap::new();
    for spec in &fixture.predicate_specs {
        require_nonempty(&spec.predicate_id)?;
        let mut dependencies = spec.dependencies.clone();
        dependencies.sort();
        if dependencies.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(QualificationError::DuplicatePredicateDependency);
        }
        let mut reason_ids = spec.reason_ids.clone();
        reason_ids.sort();
        if reason_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(QualificationError::DuplicatePredicateReason);
        }
        match &spec.operator {
            PredicateOperator::Mirror(source) if dependencies.as_slice() == [source.as_str()] => {}
            PredicateOperator::Mirror(_) => {
                return Err(QualificationError::OperatorDependencyMismatch);
            }
            _ if dependencies.is_empty() => {}
            _ => return Err(QualificationError::OperatorDependencyMismatch),
        }
        let canonical = PredicateSpec {
            predicate_id: spec.predicate_id.clone(),
            dependencies,
            reason_ids,
            operator: spec.operator.clone(),
        };
        if specs.insert(spec.predicate_id.clone(), canonical).is_some() {
            return Err(QualificationError::DuplicatePredicateSpec);
        }
    }
    Ok(specs)
}

fn required_policy_predicates(policy: &PolicyFixture) -> BTreeSet<String> {
    policy
        .rules
        .iter()
        .flat_map(|rule| rule.requirements.iter())
        .map(|requirement| requirement.predicate_id.clone())
        .collect()
}

fn validate_coverage(
    policy: &PolicyFixture,
    specs: &BTreeMap<String, PredicateSpec>,
) -> Result<(), QualificationError> {
    let required = required_policy_predicates(policy);
    let supplied = specs.keys().cloned().collect::<BTreeSet<_>>();
    if required.difference(&supplied).next().is_some() {
        return Err(QualificationError::MissingPolicyPredicate);
    }
    if supplied.difference(&required).next().is_some() {
        return Err(QualificationError::ExtraPredicateSpec);
    }
    Ok(())
}

fn topological_order(
    specs: &BTreeMap<String, PredicateSpec>,
) -> Result<Vec<String>, QualificationError> {
    let mut indegree = specs
        .keys()
        .cloned()
        .map(|predicate| (predicate, 0_usize))
        .collect::<BTreeMap<_, _>>();
    let mut dependents = BTreeMap::<String, Vec<String>>::new();

    for (predicate_id, spec) in specs {
        for dependency in &spec.dependencies {
            if !specs.contains_key(dependency) {
                return Err(QualificationError::UnknownPredicateDependency);
            }
            *indegree
                .get_mut(predicate_id)
                .expect("registered predicate has indegree") += 1;
            dependents
                .entry(dependency.clone())
                .or_default()
                .push(predicate_id.clone());
        }
    }
    for children in dependents.values_mut() {
        children.sort();
    }

    let mut ready = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(predicate, _)| predicate.clone())
        .collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(specs.len());
    while let Some(predicate) = ready.pop_first() {
        order.push(predicate.clone());
        if let Some(children) = dependents.get(&predicate) {
            for child in children {
                let degree = indegree
                    .get_mut(child)
                    .expect("registered dependent has indegree");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(child.clone());
                }
            }
        }
    }
    if order.len() != specs.len() {
        return Err(QualificationError::PredicateDependencyCycle);
    }
    Ok(order)
}

fn dependency_graph_id(specs: &BTreeMap<String, PredicateSpec>) -> String {
    let mut parts = Vec::new();
    for (predicate_id, spec) in specs {
        parts.push(predicate_id.clone());
        push_section(&mut parts, "dependencies", spec.dependencies.clone());
    }
    exact_id("predicate-dependency-graph", parts)
}

fn predicate_spec_id(spec: &PredicateSpec) -> String {
    let mut parts = vec![spec.predicate_id.clone(), operator_code(&spec.operator)];
    push_section(&mut parts, "dependencies", spec.dependencies.clone());
    push_section(&mut parts, "reasons", spec.reason_ids.clone());
    exact_id("predicate-spec", parts)
}

fn reason_subgraph_id(
    predicate_id: &str,
    reason_ids: &[String],
    topology: &QualifiedReasonTopology,
) -> Result<String, QualificationError> {
    let selected = reason_ids.iter().cloned().collect::<BTreeSet<_>>();
    if selected.len() != reason_ids.len() {
        return Err(QualificationError::DuplicatePredicateReason);
    }
    if selected
        .iter()
        .any(|reason_id| !topology.nodes.contains_key(reason_id))
    {
        return Err(QualificationError::UnknownPredicateReason);
    }

    let mut parts = vec![predicate_id.to_string()];
    let mut node_parts = Vec::new();
    for reason_id in &selected {
        node_parts.extend([
            reason_id.clone(),
            topology
                .nodes
                .get(reason_id)
                .cloned()
                .flatten()
                .unwrap_or_else(|| "none".to_string()),
        ]);
    }
    push_section(&mut parts, "nodes", node_parts);
    let mut edge_parts = Vec::new();
    for edge in &topology.edges {
        if selected.contains(&edge.from) && selected.contains(&edge.to) {
            edge_parts.extend([
                edge.from.clone(),
                edge.to.clone(),
                relation_code(edge.relation).to_string(),
            ]);
        }
    }
    push_section(&mut parts, "edges", edge_parts);
    Ok(exact_id("reason-subgraph", parts))
}

fn role_result(
    role: ContributionRole,
    negative: bool,
    candidates: &BTreeMap<String, Candidate>,
    accounting: &BTreeMap<String, Accounting>,
) -> (PredicateValueV1, Vec<String>) {
    let consulted = candidates
        .values()
        .filter(|candidate| candidate.role == role)
        .filter(|candidate| !matches!(accounting.get(&candidate.id), Some(Accounting::AliasOf(_))))
        .map(|candidate| candidate.id.clone())
        .collect::<Vec<_>>();
    let admitted = consulted
        .iter()
        .any(|candidate| matches!(accounting.get(candidate), Some(Accounting::Admitted)));
    let unresolved = consulted
        .iter()
        .any(|candidate| matches!(accounting.get(candidate), Some(Accounting::Unresolved(_))));
    let result = if negative {
        if admitted {
            PredicateValueV1::NotSatisfied
        } else if unresolved {
            PredicateValueV1::Unknown
        } else {
            PredicateValueV1::Satisfied
        }
    } else if admitted {
        PredicateValueV1::Satisfied
    } else if unresolved {
        PredicateValueV1::Unknown
    } else {
        PredicateValueV1::NotSatisfied
    };
    (result, consulted)
}

fn derivation_artifact_id(
    dependency_graph_id: &str,
    specs: &BTreeMap<String, PredicateSpec>,
) -> String {
    let mut parts = vec![dependency_graph_id.to_string()];
    push_section(
        &mut parts,
        "predicate-specs",
        specs.values().map(predicate_spec_id),
    );
    exact_id("predicate-derivation-artifact", parts)
}

fn derive_receipts(
    specs: &BTreeMap<String, PredicateSpec>,
    order: &[String],
    environment: &DerivationEnvironment<'_>,
) -> Result<Vec<PredicateReceipt>, QualificationError> {
    let mut results = BTreeMap::<String, PredicateValueV1>::new();
    let mut receipts = BTreeMap::new();

    for predicate_id in order {
        let spec = specs
            .get(predicate_id)
            .expect("topological order contains registered predicate");
        let subgraph =
            reason_subgraph_id(predicate_id, &spec.reason_ids, environment.reason_topology)?;
        let (result, consulted, closed_world) = match &spec.operator {
            PredicateOperator::AnyAdmitted(role) => {
                let (result, consulted) =
                    role_result(*role, false, environment.candidates, environment.accounting);
                (result, consulted, None)
            }
            PredicateOperator::NoAdmitted(role) => {
                let (result, consulted) =
                    role_result(*role, true, environment.candidates, environment.accounting);
                (
                    result,
                    consulted,
                    Some(environment.closed_view_receipt_id.to_string()),
                )
            }
            PredicateOperator::AllAccounted => (
                PredicateValueV1::Satisfied,
                environment.candidates.keys().cloned().collect(),
                Some(environment.closed_view_receipt_id.to_string()),
            ),
            PredicateOperator::Mirror(source) => {
                let result = *results
                    .get(source)
                    .ok_or(QualificationError::UnknownPredicateDependency)?;
                (result, Vec::new(), None)
            }
        };
        results.insert(predicate_id.clone(), result);
        receipts.insert(
            predicate_id.clone(),
            PredicateReceipt {
                predicate_id: predicate_id.clone(),
                result,
                predicate_spec_id: predicate_spec_id(spec),
                reason_subgraph_id: subgraph,
                dependency_ids: spec.dependencies.clone(),
                consulted_candidate_ids: consulted,
                closed_world_receipt_id: closed_world,
                derivation_profile_id: FIXTURE_DERIVATION_PROFILE.to_string(),
                derivation_artifact_id: environment.derivation_artifact_id.to_string(),
                derivation_execution_lineage_id: environment
                    .derivation_execution_lineage_id
                    .to_string(),
            },
        );
    }
    Ok(receipts.into_values().collect())
}

fn policy_from_fixture(fixture: &ReconstructionFixture) -> DispositionPolicyV1 {
    let mut rules = fixture.policy.rules.clone();
    rules.sort_by_key(|rule| rule.priority);
    for rule in &mut rules {
        rule.requirements
            .sort_by_key(|requirement| requirement.predicate_id.clone());
    }

    let mut parts = vec![
        fixture.lineage.scientific_use.clone(),
        fixture.policy.profile_id.clone(),
        fixture.policy.fallback_output.clone(),
    ];
    for rule in &rules {
        let mut rule_parts = vec![
            rule.rule_id.clone(),
            rule.priority.to_string(),
            rule.output.clone(),
        ];
        for requirement in &rule.requirements {
            rule_parts.extend([
                requirement.predicate_id.clone(),
                value_code(requirement.expected).to_string(),
            ]);
        }
        push_section(&mut parts, "rule", rule_parts);
    }

    DispositionPolicyV1 {
        profile_id: fixture.policy.profile_id.clone(),
        artifact_id: exact_id("policy-artifact", parts),
        rules,
        fallback_output: fixture.policy.fallback_output.clone(),
    }
}

fn receipt_id(receipt: &PredicateReceipt) -> String {
    let mut parts = vec![
        receipt.predicate_id.clone(),
        value_code(receipt.result).to_string(),
        receipt.predicate_spec_id.clone(),
        receipt.reason_subgraph_id.clone(),
        receipt
            .closed_world_receipt_id
            .clone()
            .unwrap_or_else(|| "none".to_string()),
        receipt.derivation_profile_id.clone(),
        receipt.derivation_artifact_id.clone(),
        receipt.derivation_execution_lineage_id.clone(),
    ];
    push_section(&mut parts, "dependencies", receipt.dependency_ids.clone());
    push_section(
        &mut parts,
        "consulted-candidates",
        receipt.consulted_candidate_ids.clone(),
    );
    exact_id("predicate-receipt", parts)
}

fn qualify_fixture(
    fixture: &ReconstructionFixture,
) -> Result<QualifiedFixtureReconstruction, QualificationError> {
    for value in [
        fixture.lineage.proposition.as_str(),
        fixture.lineage.scientific_use.as_str(),
        fixture.lineage.evidence_scope.as_str(),
        fixture.lineage.discovery_policy.as_str(),
        fixture.lineage.source_registry_generation.as_str(),
        fixture.lineage.source_lifecycle_generation.as_str(),
        fixture.lineage.external_argument_generation.as_str(),
        fixture.lineage.dependency_graph_generation.as_str(),
        fixture.lineage.target_compatibility_generation.as_str(),
        fixture.lineage.triangulation_generation.as_str(),
        fixture.lineage.information_cutoff.as_str(),
        fixture
            .lineage
            .predicate_derivation_execution_lineage
            .as_str(),
        fixture.lineage.owner_reconstruction_artifact.as_str(),
        fixture
            .lineage
            .owner_reconstruction_execution_lineage
            .as_str(),
    ] {
        require_nonempty(value)?;
    }

    let candidates = canonical_candidates(fixture)?;
    let accounting = canonical_accounting(fixture, &candidates)?;
    let (view_id, view_snapshot_id, closed_view_receipt_id) =
        closed_view_ids(fixture, &candidates, &accounting);
    let reason_topology = qualify_reason_topology(fixture, &candidates)?;
    let specs = canonical_specs(fixture)?;
    validate_coverage(&fixture.policy, &specs)?;
    let order = topological_order(&specs)?;
    let predicate_dependency_graph_id = dependency_graph_id(&specs);
    let derivation_artifact_id = derivation_artifact_id(&predicate_dependency_graph_id, &specs);
    let environment = DerivationEnvironment {
        candidates: &candidates,
        accounting: &accounting,
        reason_topology: &reason_topology,
        closed_view_receipt_id: &closed_view_receipt_id,
        derivation_artifact_id: &derivation_artifact_id,
        derivation_execution_lineage_id: &fixture.lineage.predicate_derivation_execution_lineage,
    };
    let receipts = derive_receipts(&specs, &order, &environment)?;
    let policy = policy_from_fixture(fixture);

    let context = DispositionEvaluationContextV1 {
        evidence_view_id: view_id,
        evidence_view_snapshot_id: view_snapshot_id,
        source_lifecycle_generation_id: fixture.lineage.source_lifecycle_generation.clone(),
        external_argument_generation_id: fixture.lineage.external_argument_generation.clone(),
        dependency_graph_generation_id: fixture.lineage.dependency_graph_generation.clone(),
        target_compatibility_generation_id: fixture.lineage.target_compatibility_generation.clone(),
        triangulation_generation_id: fixture.lineage.triangulation_generation.clone(),
        reason_topology_snapshot_id: reason_topology.material_id.clone(),
        predicate_derivation_profile_id: FIXTURE_DERIVATION_PROFILE.to_string(),
        predicate_derivation_artifact_id: derivation_artifact_id,
        predicate_derivation_execution_lineage_id: fixture
            .lineage
            .predicate_derivation_execution_lineage
            .clone(),
        information_cutoff_id: fixture.lineage.information_cutoff.clone(),
    };
    let inputs = DispositionEvaluationInputsV1 {
        proposition_id: exact_id("proposition", [fixture.lineage.proposition.clone()]),
        scientific_use_id: exact_id("scientific-use", [fixture.lineage.scientific_use.clone()]),
        context,
        reason_topology_ids: reason_topology.nodes.keys().cloned().collect(),
        predicates: receipts
            .iter()
            .map(|receipt| PredicateFactV1 {
                predicate_id: receipt.predicate_id.clone(),
                value: receipt.result,
            })
            .collect(),
    };

    // SCI-014A invariant: lower replay material must be valid before this
    // private reconstruction capability can be constructed.
    let _ = evaluate_disposition_material_v1(&inputs, &policy)?;

    let mut material_parts = vec![
        FIXTURE_RECONSTRUCTION_PROFILE.to_string(),
        inputs.proposition_id.clone(),
        inputs.scientific_use_id.clone(),
        inputs.context.evidence_view_id.clone(),
        inputs.context.evidence_view_snapshot_id.clone(),
        inputs.context.source_lifecycle_generation_id.clone(),
        inputs.context.external_argument_generation_id.clone(),
        inputs.context.dependency_graph_generation_id.clone(),
        inputs.context.target_compatibility_generation_id.clone(),
        inputs.context.triangulation_generation_id.clone(),
        inputs.context.reason_topology_snapshot_id.clone(),
        inputs.context.predicate_derivation_profile_id.clone(),
        inputs.context.predicate_derivation_artifact_id.clone(),
        inputs
            .context
            .predicate_derivation_execution_lineage_id
            .clone(),
        inputs.context.information_cutoff_id.clone(),
        closed_view_receipt_id.clone(),
        predicate_dependency_graph_id.clone(),
        policy.artifact_id.clone(),
        fixture.lineage.owner_reconstruction_artifact.clone(),
        fixture
            .lineage
            .owner_reconstruction_execution_lineage
            .clone(),
    ];
    push_section(
        &mut material_parts,
        "predicate-receipts",
        receipts.iter().map(receipt_id),
    );
    let material_id = exact_id("scientific-reconstruction", material_parts);

    Ok(QualifiedFixtureReconstruction {
        material_id,
        closed_view_receipt_id,
        predicate_dependency_graph_id,
        receipts,
        inputs,
        policy,
    })
}

fn build_record(
    reconstruction: &QualifiedFixtureReconstruction,
) -> PersistedDispositionEvaluationRecordV1 {
    build_persisted_evaluation_record_v1(
        "fixture-assessment",
        &reconstruction.inputs,
        &reconstruction.policy,
    )
    .expect("qualified fixture already passed lower evaluation validation")
}

fn verify_fixture_replay(
    record: &PersistedDispositionEvaluationRecordV1,
    reconstruction: QualifiedFixtureReconstruction,
) -> Result<FixtureReplayWitness, QualificationError> {
    let lower_replay = verify_persisted_disposition_evaluation_v1(
        record,
        &reconstruction.inputs,
        &reconstruction.policy,
    )?;
    Ok(FixtureReplayWitness {
        reconstruction_material_id: reconstruction.material_id,
        lower_replay,
    })
}

fn requirement(predicate_id: &str, expected: PredicateValueV1) -> RuleRequirementV1 {
    RuleRequirementV1 {
        predicate_id: predicate_id.to_string(),
        expected,
    }
}

fn base_policy() -> PolicyFixture {
    PolicyFixture {
        profile_id: "fixture-policy-v1".to_string(),
        rules: vec![
            DispositionRuleV1 {
                rule_id: "blocked".to_string(),
                priority: 10,
                requirements: vec![requirement("active-defeater", PredicateValueV1::Satisfied)],
                output: "BlockedByQualifiedDefeater".to_string(),
            },
            DispositionRuleV1 {
                rule_id: "contested".to_string(),
                priority: 20,
                requirements: vec![
                    requirement("support-exists", PredicateValueV1::Satisfied),
                    requirement("opposition-exists", PredicateValueV1::Satisfied),
                ],
                output: "Contested".to_string(),
            },
            DispositionRuleV1 {
                rule_id: "supported".to_string(),
                priority: 30,
                requirements: vec![
                    requirement("support-exists", PredicateValueV1::Satisfied),
                    requirement("opposition-absent", PredicateValueV1::Satisfied),
                    requirement("active-defeater", PredicateValueV1::NotSatisfied),
                    requirement("all-accounted", PredicateValueV1::Satisfied),
                ],
                output: "SupportedWithinFixtureScope".to_string(),
            },
        ],
        fallback_output: "Underdetermined".to_string(),
    }
}

fn base_fixture() -> ReconstructionFixture {
    ReconstructionFixture {
        lineage: ScientificLineageFixture {
            proposition: "fixture proposition".to_string(),
            scientific_use: "fixture scientific review".to_string(),
            evidence_scope: "bounded-fixture-scope".to_string(),
            discovery_policy: "fixture-discovery-v1".to_string(),
            source_registry_generation: "registry:4".to_string(),
            source_lifecycle_generation: "lifecycle:4".to_string(),
            external_argument_generation: "argument:9".to_string(),
            dependency_graph_generation: "dependency:6".to_string(),
            target_compatibility_generation: "compatibility:3".to_string(),
            triangulation_generation: "triangulation:8".to_string(),
            information_cutoff: "ordinal:17".to_string(),
            predicate_derivation_execution_lineage: "predicate-execution:1".to_string(),
            owner_reconstruction_artifact: "reconstruction-artifact:1".to_string(),
            owner_reconstruction_execution_lineage: "reconstruction-execution:1".to_string(),
        },
        candidates: vec![
            Candidate {
                id: "evidence:support".to_string(),
                reason_id: "reason:support".to_string(),
                role: ContributionRole::Support,
            },
            Candidate {
                id: "evidence:opposition".to_string(),
                reason_id: "reason:opposition".to_string(),
                role: ContributionRole::Opposition,
            },
            Candidate {
                id: "evidence:defeater".to_string(),
                reason_id: "reason:defeater".to_string(),
                role: ContributionRole::ActiveDefeater,
            },
        ],
        accounting: vec![
            AccountingEntry {
                candidate_id: "evidence:support".to_string(),
                terminal: Accounting::Admitted,
            },
            AccountingEntry {
                candidate_id: "evidence:opposition".to_string(),
                terminal: Accounting::Excluded("outside-qualified-scope".to_string()),
            },
            AccountingEntry {
                candidate_id: "evidence:defeater".to_string(),
                terminal: Accounting::Excluded("inactive-at-cutoff".to_string()),
            },
        ],
        reason_nodes: vec![
            ReasonNode {
                reason_id: "reason:support".to_string(),
                candidate_id: Some("evidence:support".to_string()),
            },
            ReasonNode {
                reason_id: "reason:opposition".to_string(),
                candidate_id: Some("evidence:opposition".to_string()),
            },
            ReasonNode {
                reason_id: "reason:defeater".to_string(),
                candidate_id: Some("evidence:defeater".to_string()),
            },
            ReasonNode {
                reason_id: "reason:closure".to_string(),
                candidate_id: None,
            },
        ],
        // Deliberate scientific-reason cycle: reason topology may be cyclic.
        reason_edges: vec![
            ReasonEdge {
                from: "reason:support".to_string(),
                to: "reason:opposition".to_string(),
                relation: ReasonRelation::Opposes,
            },
            ReasonEdge {
                from: "reason:opposition".to_string(),
                to: "reason:support".to_string(),
                relation: ReasonRelation::Supports,
            },
            ReasonEdge {
                from: "reason:defeater".to_string(),
                to: "reason:support".to_string(),
                relation: ReasonRelation::Undercuts,
            },
        ],
        predicate_specs: vec![
            PredicateSpec {
                predicate_id: "support-exists".to_string(),
                dependencies: Vec::new(),
                reason_ids: vec!["reason:support".to_string()],
                operator: PredicateOperator::AnyAdmitted(ContributionRole::Support),
            },
            PredicateSpec {
                predicate_id: "opposition-exists".to_string(),
                dependencies: Vec::new(),
                reason_ids: vec!["reason:opposition".to_string()],
                operator: PredicateOperator::AnyAdmitted(ContributionRole::Opposition),
            },
            PredicateSpec {
                predicate_id: "opposition-absent".to_string(),
                dependencies: Vec::new(),
                reason_ids: vec![
                    "reason:closure".to_string(),
                    "reason:opposition".to_string(),
                ],
                operator: PredicateOperator::NoAdmitted(ContributionRole::Opposition),
            },
            PredicateSpec {
                predicate_id: "active-defeater".to_string(),
                dependencies: Vec::new(),
                reason_ids: vec!["reason:defeater".to_string()],
                operator: PredicateOperator::AnyAdmitted(ContributionRole::ActiveDefeater),
            },
            PredicateSpec {
                predicate_id: "all-accounted".to_string(),
                dependencies: Vec::new(),
                reason_ids: vec!["reason:closure".to_string()],
                operator: PredicateOperator::AllAccounted,
            },
        ],
        policy: base_policy(),
    }
}

fn receipt<'a>(
    reconstruction: &'a QualifiedFixtureReconstruction,
    predicate_id: &str,
) -> &'a PredicateReceipt {
    reconstruction
        .receipts
        .iter()
        .find(|receipt| receipt.predicate_id == predicate_id)
        .expect("fixture predicate has receipt")
}

#[test]
fn owner_reconstruction_composes_with_qualified_lower_replay() {
    let reconstruction = qualify_fixture(&base_fixture()).unwrap();
    let record = build_record(&reconstruction);
    let material_id = reconstruction.material_id.clone();
    let replay = verify_fixture_replay(&record, reconstruction).unwrap();

    assert_eq!(replay.reconstruction_material_id, material_id);
    assert_eq!(
        replay.lower_replay.primary_disposition(),
        "SupportedWithinFixtureScope"
    );
}

#[test]
fn lower_evaluation_must_validate_before_reconstruction_capability() {
    let mut fixture = base_fixture();
    fixture.policy.rules[1].priority = fixture.policy.rules[0].priority;

    assert!(matches!(
        qualify_fixture(&fixture),
        Err(QualificationError::Lower(
            ReplayErrorV1::DuplicatePriority { priority: 10 }
        ))
    ));
}

#[test]
fn omitted_candidate_breaks_closed_accounting() {
    let mut fixture = base_fixture();
    fixture
        .accounting
        .retain(|entry| entry.candidate_id != "evidence:opposition");
    assert_eq!(
        qualify_fixture(&fixture),
        Err(QualificationError::AccountingMismatch)
    );
}

#[test]
fn unresolved_opposition_remains_unknown_not_absent() {
    let mut fixture = base_fixture();
    fixture.accounting[1].terminal = Accounting::Unresolved("awaiting-review".to_string());
    let reconstruction = qualify_fixture(&fixture).unwrap();

    assert_eq!(
        receipt(&reconstruction, "opposition-absent").result,
        PredicateValueV1::Unknown
    );
}

#[test]
fn negative_by_absence_predicate_carries_closed_world_receipt() {
    let reconstruction = qualify_fixture(&base_fixture()).unwrap();
    assert_eq!(
        receipt(&reconstruction, "opposition-absent")
            .closed_world_receipt_id
            .as_deref(),
        Some(reconstruction.closed_view_receipt_id.as_str())
    );
}

#[test]
fn every_predicate_receipt_carries_derivation_lineage() {
    let fixture = base_fixture();
    let reconstruction = qualify_fixture(&fixture).unwrap();
    for receipt in &reconstruction.receipts {
        assert_eq!(receipt.derivation_profile_id, FIXTURE_DERIVATION_PROFILE);
        assert_eq!(
            receipt.derivation_execution_lineage_id,
            fixture.lineage.predicate_derivation_execution_lineage
        );
        assert_eq!(
            receipt.derivation_artifact_id,
            reconstruction
                .inputs
                .context
                .predicate_derivation_artifact_id
        );
    }
}

#[test]
fn owner_and_predicate_execution_lineages_are_independent_material() {
    let first = qualify_fixture(&base_fixture()).unwrap();

    let mut owner_changed = base_fixture();
    owner_changed.lineage.owner_reconstruction_execution_lineage =
        "reconstruction-execution:2".to_string();
    let second = qualify_fixture(&owner_changed).unwrap();

    assert_ne!(first.material_id, second.material_id);
    assert_eq!(first.inputs, second.inputs);
    assert_eq!(first.policy, second.policy);
}

#[test]
fn changed_predicate_derivation_execution_invalidates_old_lower_record() {
    let first = qualify_fixture(&base_fixture()).unwrap();
    let old_record = build_record(&first);

    let mut changed = base_fixture();
    changed.lineage.predicate_derivation_execution_lineage = "predicate-execution:2".to_string();
    let second = qualify_fixture(&changed).unwrap();

    assert_eq!(
        verify_persisted_disposition_evaluation_v1(&old_record, &second.inputs, &second.policy),
        Err(ReplayErrorV1::EvaluationContextMismatch)
    );
}

#[test]
fn scientific_reason_cycle_is_allowed_but_predicate_cycle_fails_closed() {
    assert!(qualify_fixture(&base_fixture()).is_ok());

    let mut fixture = base_fixture();
    let support = fixture
        .predicate_specs
        .iter_mut()
        .find(|spec| spec.predicate_id == "support-exists")
        .unwrap();
    support.dependencies = vec!["opposition-exists".to_string()];
    support.operator = PredicateOperator::Mirror("opposition-exists".to_string());
    let opposition = fixture
        .predicate_specs
        .iter_mut()
        .find(|spec| spec.predicate_id == "opposition-exists")
        .unwrap();
    opposition.dependencies = vec!["support-exists".to_string()];
    opposition.operator = PredicateOperator::Mirror("support-exists".to_string());

    assert_eq!(
        qualify_fixture(&fixture),
        Err(QualificationError::PredicateDependencyCycle)
    );
}

#[test]
fn reason_rewiring_changes_reconstruction_identity_without_changing_predicates() {
    let first = qualify_fixture(&base_fixture()).unwrap();
    let mut changed = base_fixture();
    changed.reason_edges[0].relation = ReasonRelation::Supports;
    let second = qualify_fixture(&changed).unwrap();

    assert_ne!(first.material_id, second.material_id);
    assert_eq!(first.inputs.predicates, second.inputs.predicates);
}

#[test]
fn same_predicate_vector_with_different_dependency_graph_is_distinct() {
    let first = qualify_fixture(&base_fixture()).unwrap();
    let mut changed = base_fixture();
    let active = changed
        .predicate_specs
        .iter_mut()
        .find(|spec| spec.predicate_id == "active-defeater")
        .unwrap();
    active.dependencies = vec!["opposition-exists".to_string()];
    active.operator = PredicateOperator::Mirror("opposition-exists".to_string());
    let second = qualify_fixture(&changed).unwrap();

    assert_eq!(first.inputs.predicates, second.inputs.predicates);
    assert_ne!(
        first.predicate_dependency_graph_id,
        second.predicate_dependency_graph_id
    );
    assert_ne!(first.material_id, second.material_id);
}

#[test]
fn every_policy_predicate_requires_exact_derivation_coverage() {
    let mut missing = base_fixture();
    missing
        .predicate_specs
        .retain(|spec| spec.predicate_id != "support-exists");
    assert_eq!(
        qualify_fixture(&missing),
        Err(QualificationError::MissingPolicyPredicate)
    );

    let mut extra = base_fixture();
    extra.predicate_specs.push(PredicateSpec {
        predicate_id: "hidden-auxiliary".to_string(),
        dependencies: Vec::new(),
        reason_ids: vec!["reason:closure".to_string()],
        operator: PredicateOperator::AllAccounted,
    });
    assert_eq!(
        qualify_fixture(&extra),
        Err(QualificationError::ExtraPredicateSpec)
    );
}

#[test]
fn duplicate_alias_chain_fails_closed() {
    let mut fixture = base_fixture();
    fixture.candidates.extend([
        Candidate {
            id: "evidence:support-alias-1".to_string(),
            reason_id: "reason:support-alias-1".to_string(),
            role: ContributionRole::Support,
        },
        Candidate {
            id: "evidence:support-alias-2".to_string(),
            reason_id: "reason:support-alias-2".to_string(),
            role: ContributionRole::Support,
        },
    ]);
    fixture.accounting.extend([
        AccountingEntry {
            candidate_id: "evidence:support-alias-1".to_string(),
            terminal: Accounting::AliasOf("evidence:support".to_string()),
        },
        AccountingEntry {
            candidate_id: "evidence:support-alias-2".to_string(),
            terminal: Accounting::AliasOf("evidence:support-alias-1".to_string()),
        },
    ]);

    assert_eq!(
        qualify_fixture(&fixture),
        Err(QualificationError::AliasChain)
    );
}

#[test]
fn alias_role_mismatch_fails_closed() {
    let mut fixture = base_fixture();
    fixture.candidates.push(Candidate {
        id: "evidence:bad-alias".to_string(),
        reason_id: "reason:bad-alias".to_string(),
        role: ContributionRole::Opposition,
    });
    fixture.accounting.push(AccountingEntry {
        candidate_id: "evidence:bad-alias".to_string(),
        terminal: Accounting::AliasOf("evidence:support".to_string()),
    });

    assert_eq!(
        qualify_fixture(&fixture),
        Err(QualificationError::AliasSemanticMismatch)
    );
}
