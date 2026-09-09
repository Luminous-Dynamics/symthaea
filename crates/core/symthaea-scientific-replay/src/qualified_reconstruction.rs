// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SCI-014A fixture qualification for full scientific-disposition replay.
//!
//! This module intentionally starts with synthetic, owner-verifiable fixture
//! state. It proves the composition shape required above disposition-evaluation
//! replay without claiming that the repository's real evidence, lifecycle,
//! dependency, compatibility, or triangulation stores are already unified.
//!
//! The authority boundary is strict:
//!
//! - ordinary fixture data is not qualified reconstruction;
//! - owner reconstruction must prove closed candidate accounting;
//! - scientific reason topology and predicate dependency topology are distinct;
//! - predicate dependencies must be acyclic in this first profile;
//! - every policy predicate must have exactly one owner-derived receipt;
//! - negative-by-absence predicates consume an already-closed evidence view;
//! - only a qualified reconstruction may be composed with persisted lower
//!   evaluation data to mint `ReplayVerifiedScientificDispositionV1`;
//! - full replay is historical evidence, never currentness or action authority.

use super::{
    DispositionEvaluationContextV1, DispositionEvaluationInputsV1, DispositionPolicyV1,
    DispositionRuleV1, PersistedDispositionEvaluationRecordV1, PredicateFactV1, PredicateValueV1,
    ReplayErrorV1, ReplayVerifiedDispositionEvaluationV1, verify_persisted_disposition_evaluation_v1,
};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const SCIENTIFIC_RECONSTRUCTION_FIXTURE_PROFILE_V1: &str =
    "symthaea.science.qualified-reconstruction.fixture.v1";
pub const PREDICATE_DERIVATION_FIXTURE_PROFILE_V1: &str =
    "symthaea.science.predicate-derivation.fixture.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContributionRoleV1 {
    Support,
    Opposition,
    ActiveDefeater,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateContributionFixtureV1 {
    pub contribution_id: String,
    pub reason_id: String,
    pub role: ContributionRoleV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateAccountingDispositionV1 {
    Admitted,
    Excluded { reason_id: String },
    Unresolved { reason_id: String },
    DuplicateAliasOf { canonical_contribution_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateAccountingEntryV1 {
    pub contribution_id: String,
    pub disposition: CandidateAccountingDispositionV1,
}

/// Ordinary fixture data describing one bounded evidence view.
///
/// Closure is not asserted by construction. The owner verifier checks that the
/// exact candidate set and accounting set are equal and that every candidate
/// has exactly one terminal accounting state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosedEvidenceViewFixtureV1 {
    pub scope_id: String,
    pub discovery_policy_id: String,
    pub source_registry_generation_id: String,
    pub candidates: Vec<CandidateContributionFixtureV1>,
    pub accounting: Vec<CandidateAccountingEntryV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasonRelationV1 {
    Supports,
    Opposes,
    RebuttingDefeater,
    UndercuttingDefeater,
    ScopeDefeater,
    Falsifies,
    Lifecycle,
    Dependency,
    TargetCompatibility,
    Triangulation,
    UnresolvedConflict,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasonNodeFixtureV1 {
    pub reason_id: String,
    pub contribution_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasonEdgeFixtureV1 {
    pub from_reason_id: String,
    pub to_reason_id: String,
    pub relation: ReasonRelationV1,
}

/// Scientific reason graphs may contain cycles. They are preserved as data and
/// are deliberately not reused as predicate execution topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasonTopologyFixtureV1 {
    pub nodes: Vec<ReasonNodeFixtureV1>,
    pub edges: Vec<ReasonEdgeFixtureV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateAtomFixtureV1 {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamScientificStateFixtureV1 {
    pub source_lifecycle: Vec<StateAtomFixtureV1>,
    pub external_argument: Vec<StateAtomFixtureV1>,
    pub dependency_graph: Vec<StateAtomFixtureV1>,
    pub target_compatibility: Vec<StateAtomFixtureV1>,
    pub triangulation: Vec<StateAtomFixtureV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropositionTargetFixtureV1 {
    pub canonical_statement: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScientificUseFixtureV1 {
    pub purpose: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InformationCutoffFixtureV1 {
    pub axis_id: String,
    pub position: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnerExecutionFixtureV1 {
    pub reconstruction_artifact_id: String,
    pub execution_capsule_id: String,
}

/// Policy definition before owner-derived artifact identity is attached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispositionPolicyFixtureV1 {
    pub profile_id: String,
    pub rules: Vec<DispositionRuleV1>,
    pub fallback_output: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredicateOperatorFixtureV1 {
    AnyAdmittedRole { role: ContributionRoleV1 },
    NoAdmittedRole { role: ContributionRoleV1 },
    AllCandidatesAccountedFor,
    MirrorPredicate { source_predicate_id: String },
}

/// Declarative predicate specification for the first non-recursive profile.
///
/// `direct_predicate_dependency_ids` is computational topology. `reason_ids`
/// names the scientific reason subgraph that explains the predicate. The two
/// graphs are intentionally distinct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateDerivationSpecFixtureV1 {
    pub predicate_id: String,
    pub direct_predicate_dependency_ids: Vec<String>,
    pub reason_ids: Vec<String>,
    pub operator: PredicateOperatorFixtureV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScientificReconstructionFixtureV1 {
    pub proposition: PropositionTargetFixtureV1,
    pub scientific_use: ScientificUseFixtureV1,
    pub information_cutoff: InformationCutoffFixtureV1,
    pub evidence_view: ClosedEvidenceViewFixtureV1,
    pub upstream_state: UpstreamScientificStateFixtureV1,
    pub reason_topology: ReasonTopologyFixtureV1,
    pub predicate_specs: Vec<PredicateDerivationSpecFixtureV1>,
    pub policy: DispositionPolicyFixtureV1,
    pub owner_execution: OwnerExecutionFixtureV1,
}

/// Ordinary proof material. Possessing or deserializing an equivalent record
/// would not recreate the private qualified reconstruction capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateDerivationReceiptV1 {
    pub predicate_id: String,
    pub predicate_spec_material_id: String,
    pub direct_predicate_dependency_ids: Vec<String>,
    pub reason_subgraph_material_id: String,
    pub consulted_candidate_ids: Vec<String>,
    pub closed_world_receipt_id: Option<String>,
    pub result: PredicateValueV1,
    pub derivation_profile_id: String,
    pub derivation_artifact_id: String,
    pub execution_lineage_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScientificReconstructionMaterialV1 {
    pub reconstruction_profile_id: String,
    pub proposition_id: String,
    pub scientific_use_id: String,
    pub context: DispositionEvaluationContextV1,
    pub policy_profile_id: String,
    pub policy_artifact_id: String,
    pub closed_evidence_view_receipt_id: String,
    pub predicate_dependency_graph_material_id: String,
    pub reason_topology_ids: Vec<String>,
    pub predicate_receipts: Vec<PredicateDerivationReceiptV1>,
    pub owner_reconstruction_execution_lineage_id: String,
}

/// Owner-created positive reconstruction capability.
///
/// Private fields and absence of Serde construction keep ordinary fixture or
/// persisted data from recreating this capability.
#[derive(Debug, PartialEq, Eq)]
pub struct QualifiedScientificReconstructionV1 {
    reconstruction_material_id: String,
    material: ScientificReconstructionMaterialV1,
    policy: DispositionPolicyV1,
    predicate_facts: Vec<PredicateFactV1>,
}

impl QualifiedScientificReconstructionV1 {
    pub fn material_id(&self) -> &str {
        &self.reconstruction_material_id
    }

    pub fn material(&self) -> &ScientificReconstructionMaterialV1 {
        &self.material
    }

    pub fn predicate_receipts(&self) -> &[PredicateDerivationReceiptV1] {
        &self.material.predicate_receipts
    }
}

/// Full historical replay capability for one owner-reconstructed scientific
/// state. This is not currentness, truth, consensus, recommendation, or action
/// authority.
#[derive(Debug, PartialEq, Eq)]
pub struct ReplayVerifiedScientificDispositionV1 {
    reconstruction: QualifiedScientificReconstructionV1,
    evaluation: ReplayVerifiedDispositionEvaluationV1,
}

impl ReplayVerifiedScientificDispositionV1 {
    pub fn reconstruction(&self) -> &QualifiedScientificReconstructionV1 {
        &self.reconstruction
    }

    pub fn evaluation(&self) -> &ReplayVerifiedDispositionEvaluationV1 {
        &self.evaluation
    }

    pub fn primary_disposition(&self) -> &str {
        self.evaluation.primary_disposition()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconstructionErrorV1 {
    EmptyField { field: &'static str },
    DuplicateCandidate { contribution_id: String },
    DuplicateAccounting { contribution_id: String },
    CandidateAccountingMismatch,
    InvalidAccountingReason { contribution_id: String },
    AliasTargetMissing { contribution_id: String, target_id: String },
    AliasSelfReference { contribution_id: String },
    AliasChain { contribution_id: String, target_id: String },
    DuplicateReasonNode { reason_id: String },
    UnknownReasonContribution { reason_id: String, contribution_id: String },
    CandidateReasonMismatch { contribution_id: String, reason_id: String },
    DuplicateReasonEdge { edge_id: String },
    UnknownReasonEndpoint { reason_id: String },
    DuplicateStateAtom { field: &'static str, key: String },
    DuplicatePredicateSpec { predicate_id: String },
    DuplicatePredicateDependency { predicate_id: String, dependency_id: String },
    DuplicatePredicateReason { predicate_id: String, reason_id: String },
    UnknownPredicateDependency { predicate_id: String, dependency_id: String },
    PredicateDependencyCycle,
    OperatorDependencyMismatch { predicate_id: String },
    UnknownPredicateReason { predicate_id: String, reason_id: String },
    MissingPolicyPredicateCoverage { predicate_id: String },
    ExtraPredicateSpec { predicate_id: String },
    LowerReplay(ReplayErrorV1),
}

impl fmt::Display for ReconstructionErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField { field } => write!(f, "empty required field: {field}"),
            Self::DuplicateCandidate { contribution_id } => {
                write!(f, "duplicate candidate contribution: {contribution_id}")
            }
            Self::DuplicateAccounting { contribution_id } => {
                write!(f, "duplicate candidate accounting: {contribution_id}")
            }
            Self::CandidateAccountingMismatch => {
                write!(f, "candidate and accounting identity sets differ")
            }
            Self::InvalidAccountingReason { contribution_id } => {
                write!(f, "invalid accounting reason for contribution: {contribution_id}")
            }
            Self::AliasTargetMissing {
                contribution_id,
                target_id,
            } => write!(
                f,
                "duplicate alias {contribution_id} targets missing contribution {target_id}"
            ),
            Self::AliasSelfReference { contribution_id } => {
                write!(f, "duplicate alias self-reference: {contribution_id}")
            }
            Self::AliasChain {
                contribution_id,
                target_id,
            } => write!(
                f,
                "duplicate alias chain is not canonical: {contribution_id} -> {target_id}"
            ),
            Self::DuplicateReasonNode { reason_id } => {
                write!(f, "duplicate reason node: {reason_id}")
            }
            Self::UnknownReasonContribution {
                reason_id,
                contribution_id,
            } => write!(
                f,
                "reason {reason_id} refers to unknown contribution {contribution_id}"
            ),
            Self::CandidateReasonMismatch {
                contribution_id,
                reason_id,
            } => write!(
                f,
                "candidate {contribution_id} is not bound by reason node {reason_id}"
            ),
            Self::DuplicateReasonEdge { edge_id } => {
                write!(f, "duplicate reason edge: {edge_id}")
            }
            Self::UnknownReasonEndpoint { reason_id } => {
                write!(f, "reason edge refers to unknown node: {reason_id}")
            }
            Self::DuplicateStateAtom { field, key } => {
                write!(f, "duplicate state atom in {field}: {key}")
            }
            Self::DuplicatePredicateSpec { predicate_id } => {
                write!(f, "duplicate predicate specification: {predicate_id}")
            }
            Self::DuplicatePredicateDependency {
                predicate_id,
                dependency_id,
            } => write!(
                f,
                "duplicate predicate dependency {dependency_id} for {predicate_id}"
            ),
            Self::DuplicatePredicateReason {
                predicate_id,
                reason_id,
            } => write!(f, "duplicate reason {reason_id} for predicate {predicate_id}"),
            Self::UnknownPredicateDependency {
                predicate_id,
                dependency_id,
            } => write!(
                f,
                "predicate {predicate_id} depends on unknown predicate {dependency_id}"
            ),
            Self::PredicateDependencyCycle => {
                write!(f, "predicate derivation dependency graph contains a cycle")
            }
            Self::OperatorDependencyMismatch { predicate_id } => {
                write!(f, "operator/dependency contract mismatch for {predicate_id}")
            }
            Self::UnknownPredicateReason {
                predicate_id,
                reason_id,
            } => write!(
                f,
                "predicate {predicate_id} refers to unknown reason {reason_id}"
            ),
            Self::MissingPolicyPredicateCoverage { predicate_id } => {
                write!(f, "missing derivation receipt path for policy predicate {predicate_id}")
            }
            Self::ExtraPredicateSpec { predicate_id } => write!(
                f,
                "auxiliary predicate {predicate_id} is outside fixture v1 coverage profile"
            ),
            Self::LowerReplay(error) => write!(f, "lower disposition replay failed: {error}"),
        }
    }
}

impl Error for ReconstructionErrorV1 {}

impl From<ReplayErrorV1> for ReconstructionErrorV1 {
    fn from(value: ReplayErrorV1) -> Self {
        Self::LowerReplay(value)
    }
}

#[derive(Debug)]
struct QualifiedClosedEvidenceViewFixtureV1 {
    view_id: String,
    snapshot_id: String,
    closure_receipt_id: String,
    candidates: BTreeMap<String, CandidateContributionFixtureV1>,
    accounting: BTreeMap<String, CandidateAccountingDispositionV1>,
}

#[derive(Debug)]
struct QualifiedReasonTopologyFixtureV1 {
    snapshot_id: String,
    node_ids: BTreeSet<String>,
    node_contributions: BTreeMap<String, Option<String>>,
    canonical_nodes: Vec<ReasonNodeFixtureV1>,
    canonical_edges: Vec<ReasonEdgeFixtureV1>,
}

fn require_non_empty(value: &str, field: &'static str) -> Result<(), ReconstructionErrorV1> {
    if value.trim().is_empty() {
        Err(ReconstructionErrorV1::EmptyField { field })
    } else {
        Ok(())
    }
}

/// Exact fixture identity: this is a length-prefixed canonical representation,
/// not a cryptographic digest. No collision-prone hash is being promoted to a
/// scientific identity theorem in this synthetic tranche.
fn exact_material_id(kind: &str, parts: Vec<String>) -> String {
    let mut encoded = format!("symthaea.science.fixture.{kind}.v1|");
    for part in parts {
        encoded.push_str(&part.len().to_string());
        encoded.push(':');
        encoded.push_str(&part);
        encoded.push('|');
    }
    encoded
}

fn role_code(role: ContributionRoleV1) -> &'static str {
    match role {
        ContributionRoleV1::Support => "support",
        ContributionRoleV1::Opposition => "opposition",
        ContributionRoleV1::ActiveDefeater => "active-defeater",
        ContributionRoleV1::Other => "other",
    }
}

fn predicate_value_code(value: PredicateValueV1) -> &'static str {
    match value {
        PredicateValueV1::Satisfied => "satisfied",
        PredicateValueV1::NotSatisfied => "not-satisfied",
        PredicateValueV1::Unknown => "unknown",
        PredicateValueV1::NotApplicable => "not-applicable",
        PredicateValueV1::Blocked => "blocked",
    }
}

fn relation_code(relation: &ReasonRelationV1) -> &'static str {
    match relation {
        ReasonRelationV1::Supports => "supports",
        ReasonRelationV1::Opposes => "opposes",
        ReasonRelationV1::RebuttingDefeater => "rebutting-defeater",
        ReasonRelationV1::UndercuttingDefeater => "undercutting-defeater",
        ReasonRelationV1::ScopeDefeater => "scope-defeater",
        ReasonRelationV1::Falsifies => "falsifies",
        ReasonRelationV1::Lifecycle => "lifecycle",
        ReasonRelationV1::Dependency => "dependency",
        ReasonRelationV1::TargetCompatibility => "target-compatibility",
        ReasonRelationV1::Triangulation => "triangulation",
        ReasonRelationV1::UnresolvedConflict => "unresolved-conflict",
    }
}

fn accounting_code(disposition: &CandidateAccountingDispositionV1) -> String {
    match disposition {
        CandidateAccountingDispositionV1::Admitted => "admitted".to_string(),
        CandidateAccountingDispositionV1::Excluded { reason_id } => {
            format!("excluded:{reason_id}")
        }
        CandidateAccountingDispositionV1::Unresolved { reason_id } => {
            format!("unresolved:{reason_id}")
        }
        CandidateAccountingDispositionV1::DuplicateAliasOf {
            canonical_contribution_id,
        } => format!("duplicate-alias-of:{canonical_contribution_id}"),
    }
}

fn canonical_state_atoms(
    atoms: &[StateAtomFixtureV1],
    field: &'static str,
) -> Result<Vec<StateAtomFixtureV1>, ReconstructionErrorV1> {
    let mut by_key = BTreeMap::new();
    for atom in atoms {
        require_non_empty(&atom.key, "state_atom_key")?;
        require_non_empty(&atom.value, "state_atom_value")?;
        if by_key.insert(atom.key.clone(), atom.clone()).is_some() {
            return Err(ReconstructionErrorV1::DuplicateStateAtom {
                field,
                key: atom.key.clone(),
            });
        }
    }
    Ok(by_key.into_values().collect())
}

fn state_snapshot_id(
    kind: &str,
    atoms: &[StateAtomFixtureV1],
    field: &'static str,
) -> Result<String, ReconstructionErrorV1> {
    let canonical = canonical_state_atoms(atoms, field)?;
    let mut parts = Vec::with_capacity(canonical.len() * 2);
    for atom in canonical {
        parts.push(atom.key);
        parts.push(atom.value);
    }
    Ok(exact_material_id(kind, parts))
}

fn proposition_id(
    proposition: &PropositionTargetFixtureV1,
) -> Result<String, ReconstructionErrorV1> {
    require_non_empty(&proposition.canonical_statement, "canonical_statement")?;
    Ok(exact_material_id(
        "proposition",
        vec![proposition.canonical_statement.clone()],
    ))
}

fn scientific_use_id(use_fixture: &ScientificUseFixtureV1) -> Result<String, ReconstructionErrorV1> {
    require_non_empty(&use_fixture.purpose, "scientific_use_purpose")?;
    Ok(exact_material_id(
        "scientific-use",
        vec![use_fixture.purpose.clone()],
    ))
}

fn information_cutoff_id(
    cutoff: &InformationCutoffFixtureV1,
) -> Result<String, ReconstructionErrorV1> {
    require_non_empty(&cutoff.axis_id, "information_cutoff_axis_id")?;
    require_non_empty(&cutoff.position, "information_cutoff_position")?;
    Ok(exact_material_id(
        "information-cutoff",
        vec![cutoff.axis_id.clone(), cutoff.position.clone()],
    ))
}

fn owner_execution_lineage_id(
    execution: &OwnerExecutionFixtureV1,
) -> Result<String, ReconstructionErrorV1> {
    require_non_empty(
        &execution.reconstruction_artifact_id,
        "reconstruction_artifact_id",
    )?;
    require_non_empty(&execution.execution_capsule_id, "execution_capsule_id")?;
    Ok(exact_material_id(
        "owner-reconstruction-execution",
        vec![
            execution.reconstruction_artifact_id.clone(),
            execution.execution_capsule_id.clone(),
        ],
    ))
}

fn qualify_closed_evidence_view(
    fixture: &ClosedEvidenceViewFixtureV1,
    proposition_id: &str,
    scientific_use_id: &str,
    cutoff_id: &str,
) -> Result<QualifiedClosedEvidenceViewFixtureV1, ReconstructionErrorV1> {
    require_non_empty(&fixture.scope_id, "evidence_view_scope_id")?;
    require_non_empty(&fixture.discovery_policy_id, "discovery_policy_id")?;
    require_non_empty(
        &fixture.source_registry_generation_id,
        "source_registry_generation_id",
    )?;

    let mut candidates = BTreeMap::new();
    for candidate in &fixture.candidates {
        require_non_empty(&candidate.contribution_id, "contribution_id")?;
        require_non_empty(&candidate.reason_id, "candidate_reason_id")?;
        if candidates
            .insert(candidate.contribution_id.clone(), candidate.clone())
            .is_some()
        {
            return Err(ReconstructionErrorV1::DuplicateCandidate {
                contribution_id: candidate.contribution_id.clone(),
            });
        }
    }

    let mut accounting = BTreeMap::new();
    for entry in &fixture.accounting {
        require_non_empty(&entry.contribution_id, "accounting_contribution_id")?;
        match &entry.disposition {
            CandidateAccountingDispositionV1::Admitted => {}
            CandidateAccountingDispositionV1::Excluded { reason_id }
            | CandidateAccountingDispositionV1::Unresolved { reason_id } => {
                if reason_id.trim().is_empty() {
                    return Err(ReconstructionErrorV1::InvalidAccountingReason {
                        contribution_id: entry.contribution_id.clone(),
                    });
                }
            }
            CandidateAccountingDispositionV1::DuplicateAliasOf {
                canonical_contribution_id,
            } => {
                require_non_empty(canonical_contribution_id, "canonical_contribution_id")?;
            }
        }
        if accounting
            .insert(entry.contribution_id.clone(), entry.disposition.clone())
            .is_some()
        {
            return Err(ReconstructionErrorV1::DuplicateAccounting {
                contribution_id: entry.contribution_id.clone(),
            });
        }
    }

    let candidate_ids: BTreeSet<_> = candidates.keys().cloned().collect();
    let accounting_ids: BTreeSet<_> = accounting.keys().cloned().collect();
    if candidate_ids != accounting_ids {
        return Err(ReconstructionErrorV1::CandidateAccountingMismatch);
    }

    for (contribution_id, disposition) in &accounting {
        if let CandidateAccountingDispositionV1::DuplicateAliasOf {
            canonical_contribution_id,
        } = disposition
        {
            if contribution_id == canonical_contribution_id {
                return Err(ReconstructionErrorV1::AliasSelfReference {
                    contribution_id: contribution_id.clone(),
                });
            }
            let Some(target) = accounting.get(canonical_contribution_id) else {
                return Err(ReconstructionErrorV1::AliasTargetMissing {
                    contribution_id: contribution_id.clone(),
                    target_id: canonical_contribution_id.clone(),
                });
            };
            if matches!(
                target,
                CandidateAccountingDispositionV1::DuplicateAliasOf { .. }
            ) {
                return Err(ReconstructionErrorV1::AliasChain {
                    contribution_id: contribution_id.clone(),
                    target_id: canonical_contribution_id.clone(),
                });
            }
        }
    }

    let view_id = exact_material_id(
        "closed-evidence-view-definition",
        vec![
            proposition_id.to_string(),
            scientific_use_id.to_string(),
            fixture.scope_id.clone(),
            fixture.discovery_policy_id.clone(),
            fixture.source_registry_generation_id.clone(),
            cutoff_id.to_string(),
        ],
    );

    let mut snapshot_parts = vec![view_id.clone()];
    for candidate in candidates.values() {
        snapshot_parts.push(candidate.contribution_id.clone());
        snapshot_parts.push(candidate.reason_id.clone());
        snapshot_parts.push(role_code(candidate.role).to_string());
    }
    for (contribution_id, disposition) in &accounting {
        snapshot_parts.push(contribution_id.clone());
        snapshot_parts.push(accounting_code(disposition));
    }
    let snapshot_id = exact_material_id("closed-evidence-view-snapshot", snapshot_parts);
    let closure_receipt_id = exact_material_id(
        "closed-evidence-view-receipt",
        vec![
            snapshot_id.clone(),
            candidates.len().to_string(),
            accounting.len().to_string(),
        ],
    );

    Ok(QualifiedClosedEvidenceViewFixtureV1 {
        view_id,
        snapshot_id,
        closure_receipt_id,
        candidates,
        accounting,
    })
}

fn qualify_reason_topology(
    fixture: &ReasonTopologyFixtureV1,
    view: &QualifiedClosedEvidenceViewFixtureV1,
) -> Result<QualifiedReasonTopologyFixtureV1, ReconstructionErrorV1> {
    let mut node_contributions = BTreeMap::new();
    for node in &fixture.nodes {
        require_non_empty(&node.reason_id, "reason_id")?;
        if let Some(contribution_id) = &node.contribution_id {
            require_non_empty(contribution_id, "reason_contribution_id")?;
            if !view.candidates.contains_key(contribution_id) {
                return Err(ReconstructionErrorV1::UnknownReasonContribution {
                    reason_id: node.reason_id.clone(),
                    contribution_id: contribution_id.clone(),
                });
            }
        }
        if node_contributions
            .insert(node.reason_id.clone(), node.contribution_id.clone())
            .is_some()
        {
            return Err(ReconstructionErrorV1::DuplicateReasonNode {
                reason_id: node.reason_id.clone(),
            });
        }
    }

    let node_ids: BTreeSet<_> = node_contributions.keys().cloned().collect();
    let mut edge_keys = BTreeSet::new();
    let mut canonical_edges = fixture.edges.clone();
    canonical_edges.sort_by_key(|edge| {
        format!(
            "{}\u{1f}{}\u{1f}{}",
            edge.from_reason_id,
            edge.to_reason_id,
            relation_code(&edge.relation)
        )
    });
    for edge in &canonical_edges {
        require_non_empty(&edge.from_reason_id, "edge_from_reason_id")?;
        require_non_empty(&edge.to_reason_id, "edge_to_reason_id")?;
        if !node_ids.contains(&edge.from_reason_id) {
            return Err(ReconstructionErrorV1::UnknownReasonEndpoint {
                reason_id: edge.from_reason_id.clone(),
            });
        }
        if !node_ids.contains(&edge.to_reason_id) {
            return Err(ReconstructionErrorV1::UnknownReasonEndpoint {
                reason_id: edge.to_reason_id.clone(),
            });
        }
        let key = format!(
            "{}\u{1f}{}\u{1f}{}",
            edge.from_reason_id,
            edge.to_reason_id,
            relation_code(&edge.relation)
        );
        if !edge_keys.insert(key.clone()) {
            return Err(ReconstructionErrorV1::DuplicateReasonEdge { edge_id: key });
        }
    }

    let canonical_nodes: Vec<_> = node_contributions
        .iter()
        .map(|(reason_id, contribution_id)| ReasonNodeFixtureV1 {
            reason_id: reason_id.clone(),
            contribution_id: contribution_id.clone(),
        })
        .collect();

    for candidate in view.candidates.values() {
        if node_contributions.get(&candidate.reason_id)
            != Some(&Some(candidate.contribution_id.clone()))
        {
            return Err(ReconstructionErrorV1::CandidateReasonMismatch {
                contribution_id: candidate.contribution_id.clone(),
                reason_id: candidate.reason_id.clone(),
            });
        }
    }

    let mut parts = Vec::new();
    for node in &canonical_nodes {
        parts.push(node.reason_id.clone());
        parts.push(node.contribution_id.clone().unwrap_or_default());
    }
    for edge in &canonical_edges {
        parts.push(edge.from_reason_id.clone());
        parts.push(edge.to_reason_id.clone());
        parts.push(relation_code(&edge.relation).to_string());
    }
    let snapshot_id = exact_material_id("reason-topology", parts);

    Ok(QualifiedReasonTopologyFixtureV1 {
        snapshot_id,
        node_ids,
        node_contributions,
        canonical_nodes,
        canonical_edges,
    })
}

fn reason_subgraph_material_id(
    topology: &QualifiedReasonTopologyFixtureV1,
    reason_ids: &[String],
    predicate_id: &str,
) -> Result<String, ReconstructionErrorV1> {
    let mut selected = BTreeSet::new();
    for reason_id in reason_ids {
        if !topology.node_ids.contains(reason_id) {
            return Err(ReconstructionErrorV1::UnknownPredicateReason {
                predicate_id: predicate_id.to_string(),
                reason_id: reason_id.clone(),
            });
        }
        if !selected.insert(reason_id.clone()) {
            return Err(ReconstructionErrorV1::DuplicatePredicateReason {
                predicate_id: predicate_id.to_string(),
                reason_id: reason_id.clone(),
            });
        }
    }

    let mut parts = Vec::new();
    for node in &topology.canonical_nodes {
        if selected.contains(&node.reason_id) {
            parts.push(node.reason_id.clone());
            parts.push(node.contribution_id.clone().unwrap_or_default());
        }
    }
    for edge in &topology.canonical_edges {
        if selected.contains(&edge.from_reason_id) && selected.contains(&edge.to_reason_id) {
            parts.push(edge.from_reason_id.clone());
            parts.push(edge.to_reason_id.clone());
            parts.push(relation_code(&edge.relation).to_string());
        }
    }
    Ok(exact_material_id("reason-subgraph", parts))
}

fn build_policy(
    fixture: &DispositionPolicyFixtureV1,
) -> Result<DispositionPolicyV1, ReconstructionErrorV1> {
    let provisional = DispositionPolicyV1 {
        profile_id: fixture.profile_id.clone(),
        artifact_id: "fixture-policy-artifact-placeholder".to_string(),
        rules: fixture.rules.clone(),
        fallback_output: fixture.fallback_output.clone(),
    };
    let mut canonical = super::canonical_policy(&provisional)?;

    let mut parts = vec![canonical.profile_id.clone(), canonical.fallback_output.clone()];
    for rule in &canonical.rules {
        parts.push(rule.rule_id.clone());
        parts.push(rule.priority.to_string());
        parts.push(rule.output.clone());
        for requirement in &rule.requirements {
            parts.push(requirement.predicate_id.clone());
            parts.push(predicate_value_code(requirement.expected).to_string());
        }
    }
    canonical.artifact_id = exact_material_id("disposition-policy-artifact", parts);
    Ok(canonical)
}

fn operator_code(operator: &PredicateOperatorFixtureV1) -> String {
    match operator {
        PredicateOperatorFixtureV1::AnyAdmittedRole { role } => {
            format!("any-admitted-role:{}", role_code(*role))
        }
        PredicateOperatorFixtureV1::NoAdmittedRole { role } => {
            format!("no-admitted-role:{}", role_code(*role))
        }
        PredicateOperatorFixtureV1::AllCandidatesAccountedFor => {
            "all-candidates-accounted-for".to_string()
        }
        PredicateOperatorFixtureV1::MirrorPredicate {
            source_predicate_id,
        } => format!("mirror-predicate:{source_predicate_id}"),
    }
}

fn canonical_predicate_specs(
    specs: &[PredicateDerivationSpecFixtureV1],
) -> Result<BTreeMap<String, PredicateDerivationSpecFixtureV1>, ReconstructionErrorV1> {
    let mut by_id = BTreeMap::new();
    for spec in specs {
        require_non_empty(&spec.predicate_id, "predicate_spec_id")?;
        if spec.reason_ids.is_empty() {
            return Err(ReconstructionErrorV1::EmptyField {
                field: "predicate_reason_ids",
            });
        }

        let mut dependencies = spec.direct_predicate_dependency_ids.clone();
        dependencies.sort();
        for pair in dependencies.windows(2) {
            if pair[0] == pair[1] {
                return Err(ReconstructionErrorV1::DuplicatePredicateDependency {
                    predicate_id: spec.predicate_id.clone(),
                    dependency_id: pair[0].clone(),
                });
            }
        }

        let mut reason_ids = spec.reason_ids.clone();
        reason_ids.sort();
        for pair in reason_ids.windows(2) {
            if pair[0] == pair[1] {
                return Err(ReconstructionErrorV1::DuplicatePredicateReason {
                    predicate_id: spec.predicate_id.clone(),
                    reason_id: pair[0].clone(),
                });
            }
        }

        match &spec.operator {
            PredicateOperatorFixtureV1::MirrorPredicate {
                source_predicate_id,
            } => {
                require_non_empty(source_predicate_id, "mirror_source_predicate_id")?;
                if dependencies.as_slice() != [source_predicate_id.as_str()] {
                    return Err(ReconstructionErrorV1::OperatorDependencyMismatch {
                        predicate_id: spec.predicate_id.clone(),
                    });
                }
            }
            _ if !dependencies.is_empty() => {
                return Err(ReconstructionErrorV1::OperatorDependencyMismatch {
                    predicate_id: spec.predicate_id.clone(),
                });
            }
            _ => {}
        }

        let canonical = PredicateDerivationSpecFixtureV1 {
            predicate_id: spec.predicate_id.clone(),
            direct_predicate_dependency_ids: dependencies,
            reason_ids,
            operator: spec.operator.clone(),
        };
        if by_id.insert(spec.predicate_id.clone(), canonical).is_some() {
            return Err(ReconstructionErrorV1::DuplicatePredicateSpec {
                predicate_id: spec.predicate_id.clone(),
            });
        }
    }
    Ok(by_id)
}

fn predicate_spec_material_id(spec: &PredicateDerivationSpecFixtureV1) -> String {
    let mut parts = vec![spec.predicate_id.clone(), operator_code(&spec.operator)];
    parts.extend(spec.direct_predicate_dependency_ids.iter().cloned());
    parts.push("reason-frontier".to_string());
    parts.extend(spec.reason_ids.iter().cloned());
    exact_material_id("predicate-specification", parts)
}

fn validate_policy_coverage(
    policy: &DispositionPolicyV1,
    specs: &BTreeMap<String, PredicateDerivationSpecFixtureV1>,
) -> Result<(), ReconstructionErrorV1> {
    let required: BTreeSet<_> = policy
        .rules
        .iter()
        .flat_map(|rule| rule.requirements.iter())
        .map(|requirement| requirement.predicate_id.clone())
        .collect();
    let supplied: BTreeSet<_> = specs.keys().cloned().collect();

    if let Some(predicate_id) = required.difference(&supplied).next() {
        return Err(ReconstructionErrorV1::MissingPolicyPredicateCoverage {
            predicate_id: predicate_id.clone(),
        });
    }
    if let Some(predicate_id) = supplied.difference(&required).next() {
        return Err(ReconstructionErrorV1::ExtraPredicateSpec {
            predicate_id: predicate_id.clone(),
        });
    }
    Ok(())
}

fn predicate_topological_order(
    specs: &BTreeMap<String, PredicateDerivationSpecFixtureV1>,
) -> Result<Vec<String>, ReconstructionErrorV1> {
    let mut indegree: BTreeMap<String, usize> =
        specs.keys().cloned().map(|id| (id, 0)).collect();
    let mut dependents: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (predicate_id, spec) in specs {
        for dependency_id in &spec.direct_predicate_dependency_ids {
            if !specs.contains_key(dependency_id) {
                return Err(ReconstructionErrorV1::UnknownPredicateDependency {
                    predicate_id: predicate_id.clone(),
                    dependency_id: dependency_id.clone(),
                });
            }
            *indegree
                .get_mut(predicate_id)
                .expect("predicate identity is initialized") += 1;
            dependents
                .entry(dependency_id.clone())
                .or_default()
                .push(predicate_id.clone());
        }
    }

    for values in dependents.values_mut() {
        values.sort();
    }

    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut order = Vec::with_capacity(specs.len());

    while let Some(predicate_id) = ready.pop_first() {
        order.push(predicate_id.clone());
        if let Some(children) = dependents.get(&predicate_id) {
            for child in children {
                let degree = indegree
                    .get_mut(child)
                    .expect("dependent predicate identity is initialized");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(child.clone());
                }
            }
        }
    }

    if order.len() != specs.len() {
        return Err(ReconstructionErrorV1::PredicateDependencyCycle);
    }
    Ok(order)
}

fn predicate_dependency_graph_material_id(
    specs: &BTreeMap<String, PredicateDerivationSpecFixtureV1>,
) -> String {
    let mut parts = Vec::new();
    for (predicate_id, spec) in specs {
        parts.push(predicate_id.clone());
        parts.extend(spec.direct_predicate_dependency_ids.iter().cloned());
        parts.push("next-predicate".to_string());
    }
    exact_material_id("predicate-dependency-graph", parts)
}

fn derivation_artifact_id(
    specs: &BTreeMap<String, PredicateDerivationSpecFixtureV1>,
    dependency_graph_id: &str,
) -> String {
    let mut parts = vec![
        PREDICATE_DERIVATION_FIXTURE_PROFILE_V1.to_string(),
        dependency_graph_id.to_string(),
    ];
    for spec in specs.values() {
        parts.push(predicate_spec_material_id(spec));
    }
    exact_material_id("predicate-derivation-artifact", parts)
}

fn role_predicate_result(
    view: &QualifiedClosedEvidenceViewFixtureV1,
    role: ContributionRoleV1,
    negative_by_absence: bool,
) -> Result<(PredicateValueV1, Vec<String>), ReconstructionErrorV1> {
    let consulted: Vec<_> = view
        .candidates
        .values()
        .filter(|candidate| candidate.role == role)
        .map(|candidate| candidate.contribution_id.clone())
        .collect();

    let mut admitted = false;
    let mut unresolved = false;
    for contribution_id in &consulted {
        let Some(disposition) = view.accounting.get(contribution_id) else {
            return Err(ReconstructionErrorV1::CandidateAccountingMismatch);
        };
        match disposition {
            CandidateAccountingDispositionV1::Admitted => admitted = true,
            CandidateAccountingDispositionV1::Unresolved { .. } => unresolved = true,
            CandidateAccountingDispositionV1::Excluded { .. }
            | CandidateAccountingDispositionV1::DuplicateAliasOf { .. } => {}
        }
    }

    let value = if negative_by_absence {
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

    Ok((value, consulted))
}

fn evaluate_predicate_specs(
    specs: &BTreeMap<String, PredicateDerivationSpecFixtureV1>,
    order: &[String],
    topology: &QualifiedReasonTopologyFixtureV1,
    view: &QualifiedClosedEvidenceViewFixtureV1,
    derivation_artifact_id: &str,
    execution_lineage_id: &str,
) -> Result<Vec<PredicateDerivationReceiptV1>, ReconstructionErrorV1> {
    let mut results = BTreeMap::new();
    let mut receipts = BTreeMap::new();

    for predicate_id in order {
        let spec = specs
            .get(predicate_id)
            .expect("topological order contains registered predicates only");
        let reason_subgraph_material_id =
            reason_subgraph_material_id(topology, &spec.reason_ids, predicate_id)?;

        let (result, consulted_candidate_ids, closed_world_receipt_id) = match &spec.operator {
            PredicateOperatorFixtureV1::AnyAdmittedRole { role } => {
                let (result, consulted) = role_predicate_result(view, *role, false)?;
                (result, consulted, None)
            }
            PredicateOperatorFixtureV1::NoAdmittedRole { role } => {
                let (result, consulted) = role_predicate_result(view, *role, true)?;
                (result, consulted, Some(view.closure_receipt_id.clone()))
            }
            PredicateOperatorFixtureV1::AllCandidatesAccountedFor => (
                PredicateValueV1::Satisfied,
                view.candidates.keys().cloned().collect(),
                Some(view.closure_receipt_id.clone()),
            ),
            PredicateOperatorFixtureV1::MirrorPredicate {
                source_predicate_id,
            } => {
                let Some(result) = results.get(source_predicate_id).copied() else {
                    return Err(ReconstructionErrorV1::UnknownPredicateDependency {
                        predicate_id: predicate_id.clone(),
                        dependency_id: source_predicate_id.clone(),
                    });
                };
                (result, Vec::new(), None)
            }
        };

        results.insert(predicate_id.clone(), result);
        receipts.insert(
            predicate_id.clone(),
            PredicateDerivationReceiptV1 {
                predicate_id: predicate_id.clone(),
                predicate_spec_material_id: predicate_spec_material_id(spec),
                direct_predicate_dependency_ids: spec.direct_predicate_dependency_ids.clone(),
                reason_subgraph_material_id,
                consulted_candidate_ids,
                closed_world_receipt_id,
                result,
                derivation_profile_id: PREDICATE_DERIVATION_FIXTURE_PROFILE_V1.to_string(),
                derivation_artifact_id: derivation_artifact_id.to_string(),
                execution_lineage_id: execution_lineage_id.to_string(),
            },
        );
    }

    Ok(receipts.into_values().collect())
}

fn receipt_material_id(receipt: &PredicateDerivationReceiptV1) -> String {
    let mut parts = vec![
        receipt.predicate_id.clone(),
        receipt.predicate_spec_material_id.clone(),
        predicate_value_code(receipt.result).to_string(),
        receipt.reason_subgraph_material_id.clone(),
        receipt.closed_world_receipt_id.clone().unwrap_or_default(),
        receipt.derivation_profile_id.clone(),
        receipt.derivation_artifact_id.clone(),
        receipt.execution_lineage_id.clone(),
    ];
    parts.extend(receipt.direct_predicate_dependency_ids.iter().cloned());
    parts.push("consulted-candidates".to_string());
    parts.extend(receipt.consulted_candidate_ids.iter().cloned());
    exact_material_id("predicate-derivation-receipt", parts)
}

fn reconstruction_material_id(material: &ScientificReconstructionMaterialV1) -> String {
    let mut parts = vec![
        material.reconstruction_profile_id.clone(),
        material.proposition_id.clone(),
        material.scientific_use_id.clone(),
        material.context.evidence_view_id.clone(),
        material.context.evidence_view_snapshot_id.clone(),
        material.context.source_lifecycle_generation_id.clone(),
        material.context.external_argument_generation_id.clone(),
        material.context.dependency_graph_generation_id.clone(),
        material.context.target_compatibility_generation_id.clone(),
        material.context.triangulation_generation_id.clone(),
        material.context.reason_topology_snapshot_id.clone(),
        material.context.predicate_derivation_profile_id.clone(),
        material.context.predicate_derivation_artifact_id.clone(),
        material
            .context
            .predicate_derivation_execution_lineage_id
            .clone(),
        material.context.information_cutoff_id.clone(),
        material.policy_profile_id.clone(),
        material.policy_artifact_id.clone(),
        material.closed_evidence_view_receipt_id.clone(),
        material.predicate_dependency_graph_material_id.clone(),
        material.owner_reconstruction_execution_lineage_id.clone(),
    ];
    parts.extend(material.reason_topology_ids.iter().cloned());
    for receipt in &material.predicate_receipts {
        parts.push(receipt_material_id(receipt));
    }
    exact_material_id("scientific-reconstruction", parts)
}

/// Owner-reconstruct one exact synthetic scientific state.
///
/// This function does not accept lower evaluation context, predicate facts, or
/// reason-topology identity from the caller. Those are reconstructed here from
/// the fixture material and retained behind a private capability boundary.
pub fn qualify_fixture_scientific_reconstruction_v1(
    fixture: &ScientificReconstructionFixtureV1,
) -> Result<QualifiedScientificReconstructionV1, ReconstructionErrorV1> {
    let proposition_id = proposition_id(&fixture.proposition)?;
    let scientific_use_id = scientific_use_id(&fixture.scientific_use)?;
    let information_cutoff_id = information_cutoff_id(&fixture.information_cutoff)?;
    let owner_execution_lineage_id = owner_execution_lineage_id(&fixture.owner_execution)?;

    let view = qualify_closed_evidence_view(
        &fixture.evidence_view,
        &proposition_id,
        &scientific_use_id,
        &information_cutoff_id,
    )?;
    let topology = qualify_reason_topology(&fixture.reason_topology, &view)?;

    let source_lifecycle_generation_id = state_snapshot_id(
        "source-lifecycle-generation",
        &fixture.upstream_state.source_lifecycle,
        "source_lifecycle",
    )?;
    let external_argument_generation_id = state_snapshot_id(
        "external-argument-generation",
        &fixture.upstream_state.external_argument,
        "external_argument",
    )?;
    let dependency_graph_generation_id = state_snapshot_id(
        "dependency-graph-generation",
        &fixture.upstream_state.dependency_graph,
        "dependency_graph",
    )?;
    let target_compatibility_generation_id = state_snapshot_id(
        "target-compatibility-generation",
        &fixture.upstream_state.target_compatibility,
        "target_compatibility",
    )?;
    let triangulation_generation_id = state_snapshot_id(
        "triangulation-generation",
        &fixture.upstream_state.triangulation,
        "triangulation",
    )?;

    let policy = build_policy(&fixture.policy)?;
    let specs = canonical_predicate_specs(&fixture.predicate_specs)?;
    validate_policy_coverage(&policy, &specs)?;

    for (predicate_id, spec) in &specs {
        for reason_id in &spec.reason_ids {
            if !topology.node_ids.contains(reason_id) {
                return Err(ReconstructionErrorV1::UnknownPredicateReason {
                    predicate_id: predicate_id.clone(),
                    reason_id: reason_id.clone(),
                });
            }
        }
    }

    let topological_order = predicate_topological_order(&specs)?;
    let predicate_dependency_graph_material_id = predicate_dependency_graph_material_id(&specs);
    let predicate_derivation_artifact_id =
        derivation_artifact_id(&specs, &predicate_dependency_graph_material_id);
    let predicate_receipts = evaluate_predicate_specs(
        &specs,
        &topological_order,
        &topology,
        &view,
        &predicate_derivation_artifact_id,
        &owner_execution_lineage_id,
    )?;

    let context = DispositionEvaluationContextV1 {
        evidence_view_id: view.view_id.clone(),
        evidence_view_snapshot_id: view.snapshot_id.clone(),
        source_lifecycle_generation_id,
        external_argument_generation_id,
        dependency_graph_generation_id,
        target_compatibility_generation_id,
        triangulation_generation_id,
        reason_topology_snapshot_id: topology.snapshot_id.clone(),
        predicate_derivation_profile_id: PREDICATE_DERIVATION_FIXTURE_PROFILE_V1.to_string(),
        predicate_derivation_artifact_id,
        predicate_derivation_execution_lineage_id: owner_execution_lineage_id.clone(),
        information_cutoff_id,
    };

    let predicate_facts: Vec<_> = predicate_receipts
        .iter()
        .map(|receipt| PredicateFactV1 {
            predicate_id: receipt.predicate_id.clone(),
            value: receipt.result,
        })
        .collect();
    let reason_topology_ids: Vec<_> = topology.node_ids.iter().cloned().collect();

    let material = ScientificReconstructionMaterialV1 {
        reconstruction_profile_id: SCIENTIFIC_RECONSTRUCTION_FIXTURE_PROFILE_V1.to_string(),
        proposition_id,
        scientific_use_id,
        context,
        policy_profile_id: policy.profile_id.clone(),
        policy_artifact_id: policy.artifact_id.clone(),
        closed_evidence_view_receipt_id: view.closure_receipt_id,
        predicate_dependency_graph_material_id,
        reason_topology_ids,
        predicate_receipts,
        owner_reconstruction_execution_lineage_id: owner_execution_lineage_id,
    };
    let reconstruction_material_id = reconstruction_material_id(&material);

    Ok(QualifiedScientificReconstructionV1 {
        reconstruction_material_id,
        material,
        policy,
        predicate_facts,
    })
}

/// Compose owner-qualified reconstruction with persisted lower replay data.
///
/// No API accepts an already-created lower replay witness plus caller strings;
/// the lower witness is created internally from reconstruction-owned material.
pub fn verify_replay_scientific_disposition_v1(
    record: &PersistedDispositionEvaluationRecordV1,
    reconstruction: QualifiedScientificReconstructionV1,
) -> Result<ReplayVerifiedScientificDispositionV1, ReconstructionErrorV1> {
    let inputs = DispositionEvaluationInputsV1 {
        proposition_id: reconstruction.material.proposition_id.clone(),
        scientific_use_id: reconstruction.material.scientific_use_id.clone(),
        context: reconstruction.material.context.clone(),
        reason_topology_ids: reconstruction.material.reason_topology_ids.clone(),
        predicates: reconstruction.predicate_facts.clone(),
    };
    let evaluation = verify_persisted_disposition_evaluation_v1(
        record,
        &inputs,
        &reconstruction.policy,
    )?;

    Ok(ReplayVerifiedScientificDispositionV1 {
        reconstruction,
        evaluation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RuleRequirementV1, build_persisted_evaluation_record_v1};

    fn atom(key: &str, value: &str) -> StateAtomFixtureV1 {
        StateAtomFixtureV1 {
            key: key.into(),
            value: value.into(),
        }
    }

    fn base_policy() -> DispositionPolicyFixtureV1 {
        DispositionPolicyFixtureV1 {
            profile_id: "fixture-disposition-policy-v1".into(),
            rules: vec![
                DispositionRuleV1 {
                    rule_id: "blocked".into(),
                    priority: 10,
                    requirements: vec![RuleRequirementV1 {
                        predicate_id: "active-defeater".into(),
                        expected: PredicateValueV1::Satisfied,
                    }],
                    output: "BlockedByQualifiedDefeater".into(),
                },
                DispositionRuleV1 {
                    rule_id: "contested".into(),
                    priority: 20,
                    requirements: vec![
                        RuleRequirementV1 {
                            predicate_id: "support-exists".into(),
                            expected: PredicateValueV1::Satisfied,
                        },
                        RuleRequirementV1 {
                            predicate_id: "opposition-exists".into(),
                            expected: PredicateValueV1::Satisfied,
                        },
                    ],
                    output: "Contested".into(),
                },
                DispositionRuleV1 {
                    rule_id: "supported".into(),
                    priority: 30,
                    requirements: vec![
                        RuleRequirementV1 {
                            predicate_id: "support-exists".into(),
                            expected: PredicateValueV1::Satisfied,
                        },
                        RuleRequirementV1 {
                            predicate_id: "opposition-exists".into(),
                            expected: PredicateValueV1::NotSatisfied,
                        },
                        RuleRequirementV1 {
                            predicate_id: "active-defeater".into(),
                            expected: PredicateValueV1::NotSatisfied,
                        },
                    ],
                    output: "SupportedWithinFixtureScope".into(),
                },
            ],
            fallback_output: "Underdetermined".into(),
        }
    }

    fn base_fixture() -> ScientificReconstructionFixtureV1 {
        let all_reasons = vec![
            "reason:closure".to_string(),
            "reason:defeater".to_string(),
            "reason:opposition".to_string(),
            "reason:support".to_string(),
        ];
        ScientificReconstructionFixtureV1 {
            proposition: PropositionTargetFixtureV1 {
                canonical_statement: "fixture proposition".into(),
            },
            scientific_use: ScientificUseFixtureV1 {
                purpose: "fixture-review".into(),
            },
            information_cutoff: InformationCutoffFixtureV1 {
                axis_id: "ordinal".into(),
                position: "17".into(),
            },
            evidence_view: ClosedEvidenceViewFixtureV1 {
                scope_id: "fixture-scope".into(),
                discovery_policy_id: "fixture-discovery-v1".into(),
                source_registry_generation_id: "registry:4".into(),
                candidates: vec![
                    CandidateContributionFixtureV1 {
                        contribution_id: "evidence:support".into(),
                        reason_id: "reason:support".into(),
                        role: ContributionRoleV1::Support,
                    },
                    CandidateContributionFixtureV1 {
                        contribution_id: "evidence:opposition".into(),
                        reason_id: "reason:opposition".into(),
                        role: ContributionRoleV1::Opposition,
                    },
                    CandidateContributionFixtureV1 {
                        contribution_id: "evidence:defeater".into(),
                        reason_id: "reason:defeater".into(),
                        role: ContributionRoleV1::ActiveDefeater,
                    },
                ],
                accounting: vec![
                    CandidateAccountingEntryV1 {
                        contribution_id: "evidence:support".into(),
                        disposition: CandidateAccountingDispositionV1::Admitted,
                    },
                    CandidateAccountingEntryV1 {
                        contribution_id: "evidence:opposition".into(),
                        disposition: CandidateAccountingDispositionV1::Excluded {
                            reason_id: "outside-qualified-scope".into(),
                        },
                    },
                    CandidateAccountingEntryV1 {
                        contribution_id: "evidence:defeater".into(),
                        disposition: CandidateAccountingDispositionV1::Excluded {
                            reason_id: "inactive-at-cutoff".into(),
                        },
                    },
                ],
            },
            upstream_state: UpstreamScientificStateFixtureV1 {
                source_lifecycle: vec![atom("generation", "4")],
                external_argument: vec![atom("generation", "9")],
                dependency_graph: vec![atom("generation", "6")],
                target_compatibility: vec![atom("generation", "3")],
                triangulation: vec![atom("generation", "8")],
            },
            reason_topology: ReasonTopologyFixtureV1 {
                nodes: vec![
                    ReasonNodeFixtureV1 {
                        reason_id: "reason:support".into(),
                        contribution_id: Some("evidence:support".into()),
                    },
                    ReasonNodeFixtureV1 {
                        reason_id: "reason:opposition".into(),
                        contribution_id: Some("evidence:opposition".into()),
                    },
                    ReasonNodeFixtureV1 {
                        reason_id: "reason:defeater".into(),
                        contribution_id: Some("evidence:defeater".into()),
                    },
                    ReasonNodeFixtureV1 {
                        reason_id: "reason:closure".into(),
                        contribution_id: None,
                    },
                ],
                edges: vec![
                    ReasonEdgeFixtureV1 {
                        from_reason_id: "reason:support".into(),
                        to_reason_id: "reason:opposition".into(),
                        relation: ReasonRelationV1::Opposes,
                    },
                    ReasonEdgeFixtureV1 {
                        from_reason_id: "reason:defeater".into(),
                        to_reason_id: "reason:support".into(),
                        relation: ReasonRelationV1::UndercuttingDefeater,
                    },
                ],
            },
            predicate_specs: vec![
                PredicateDerivationSpecFixtureV1 {
                    predicate_id: "support-exists".into(),
                    direct_predicate_dependency_ids: vec![],
                    reason_ids: all_reasons.clone(),
                    operator: PredicateOperatorFixtureV1::AnyAdmittedRole {
                        role: ContributionRoleV1::Support,
                    },
                },
                PredicateDerivationSpecFixtureV1 {
                    predicate_id: "opposition-exists".into(),
                    direct_predicate_dependency_ids: vec![],
                    reason_ids: all_reasons.clone(),
                    operator: PredicateOperatorFixtureV1::AnyAdmittedRole {
                        role: ContributionRoleV1::Opposition,
                    },
                },
                PredicateDerivationSpecFixtureV1 {
                    predicate_id: "active-defeater".into(),
                    direct_predicate_dependency_ids: vec![],
                    reason_ids: all_reasons,
                    operator: PredicateOperatorFixtureV1::AnyAdmittedRole {
                        role: ContributionRoleV1::ActiveDefeater,
                    },
                },
            ],
            policy: base_policy(),
            owner_execution: OwnerExecutionFixtureV1 {
                reconstruction_artifact_id: "fixture-reconstructor-v1".into(),
                execution_capsule_id: "fixture-capsule:1".into(),
            },
        }
    }

    fn lower_inputs(
        reconstruction: &QualifiedScientificReconstructionV1,
    ) -> DispositionEvaluationInputsV1 {
        DispositionEvaluationInputsV1 {
            proposition_id: reconstruction.material.proposition_id.clone(),
            scientific_use_id: reconstruction.material.scientific_use_id.clone(),
            context: reconstruction.material.context.clone(),
            reason_topology_ids: reconstruction.material.reason_topology_ids.clone(),
            predicates: reconstruction.predicate_facts.clone(),
        }
    }

    fn persisted_record(
        reconstruction: &QualifiedScientificReconstructionV1,
    ) -> PersistedDispositionEvaluationRecordV1 {
        build_persisted_evaluation_record_v1(
            "assessment:fixture",
            &lower_inputs(reconstruction),
            &reconstruction.policy,
        )
        .unwrap()
    }

    #[test]
    fn full_replay_requires_owner_reconstruction_then_lower_replay() {
        let reconstruction = qualify_fixture_scientific_reconstruction_v1(&base_fixture()).unwrap();
        let record = persisted_record(&reconstruction);
        let verified = verify_replay_scientific_disposition_v1(&record, reconstruction).unwrap();

        assert_eq!(verified.primary_disposition(), "SupportedWithinFixtureScope");
        assert_eq!(
            verified.reconstruction().material().reconstruction_profile_id,
            SCIENTIFIC_RECONSTRUCTION_FIXTURE_PROFILE_V1
        );
    }

    #[test]
    fn omitted_candidate_breaks_closed_view_accounting() {
        let mut fixture = base_fixture();
        fixture
            .evidence_view
            .accounting
            .retain(|entry| entry.contribution_id != "evidence:opposition");
        assert_eq!(
            qualify_fixture_scientific_reconstruction_v1(&fixture),
            Err(ReconstructionErrorV1::CandidateAccountingMismatch)
        );
    }

    #[test]
    fn unresolved_opposition_remains_unknown_not_absent() {
        let mut fixture = base_fixture();
        fixture.evidence_view.accounting[1].disposition =
            CandidateAccountingDispositionV1::Unresolved {
                reason_id: "awaiting-adjudication".into(),
            };
        let reconstruction = qualify_fixture_scientific_reconstruction_v1(&fixture).unwrap();
        let opposition = reconstruction
            .predicate_receipts()
            .iter()
            .find(|receipt| receipt.predicate_id == "opposition-exists")
            .unwrap();
        assert_eq!(opposition.result, PredicateValueV1::Unknown);

        let record = persisted_record(&reconstruction);
        let verified = verify_replay_scientific_disposition_v1(&record, reconstruction).unwrap();
        assert_eq!(verified.primary_disposition(), "Underdetermined");
    }

    #[test]
    fn scientific_reason_cycles_are_allowed_but_remain_material() {
        let mut fixture = base_fixture();
        fixture.reason_topology.edges.push(ReasonEdgeFixtureV1 {
            from_reason_id: "reason:opposition".into(),
            to_reason_id: "reason:defeater".into(),
            relation: ReasonRelationV1::RebuttingDefeater,
        });
        fixture.reason_topology.edges.push(ReasonEdgeFixtureV1 {
            from_reason_id: "reason:support".into(),
            to_reason_id: "reason:opposition".into(),
            relation: ReasonRelationV1::Supports,
        });
        assert!(qualify_fixture_scientific_reconstruction_v1(&fixture).is_ok());
    }

    #[test]
    fn same_reason_nodes_with_rewired_edges_change_reconstruction_identity() {
        let first = qualify_fixture_scientific_reconstruction_v1(&base_fixture()).unwrap();
        let mut changed = base_fixture();
        changed.reason_topology.edges[0].relation = ReasonRelationV1::Supports;
        let second = qualify_fixture_scientific_reconstruction_v1(&changed).unwrap();

        assert_ne!(first.material_id(), second.material_id());
        let first_values: Vec<_> = first
            .predicate_receipts()
            .iter()
            .map(|receipt| (receipt.predicate_id.clone(), receipt.result))
            .collect();
        let second_values: Vec<_> = second
            .predicate_receipts()
            .iter()
            .map(|receipt| (receipt.predicate_id.clone(), receipt.result))
            .collect();
        assert_eq!(first_values, second_values);
    }

    #[test]
    fn predicate_dependency_cycle_fails_closed() {
        let mut fixture = base_fixture();
        fixture.policy = DispositionPolicyFixtureV1 {
            profile_id: "cycle-policy".into(),
            rules: vec![DispositionRuleV1 {
                rule_id: "cycle".into(),
                priority: 10,
                requirements: vec![
                    RuleRequirementV1 {
                        predicate_id: "a".into(),
                        expected: PredicateValueV1::Satisfied,
                    },
                    RuleRequirementV1 {
                        predicate_id: "b".into(),
                        expected: PredicateValueV1::Satisfied,
                    },
                ],
                output: "CycleMatched".into(),
            }],
            fallback_output: "Underdetermined".into(),
        };
        fixture.predicate_specs = vec![
            PredicateDerivationSpecFixtureV1 {
                predicate_id: "a".into(),
                direct_predicate_dependency_ids: vec!["b".into()],
                reason_ids: vec!["reason:support".into()],
                operator: PredicateOperatorFixtureV1::MirrorPredicate {
                    source_predicate_id: "b".into(),
                },
            },
            PredicateDerivationSpecFixtureV1 {
                predicate_id: "b".into(),
                direct_predicate_dependency_ids: vec!["a".into()],
                reason_ids: vec!["reason:opposition".into()],
                operator: PredicateOperatorFixtureV1::MirrorPredicate {
                    source_predicate_id: "a".into(),
                },
            },
        ];

        assert_eq!(
            qualify_fixture_scientific_reconstruction_v1(&fixture),
            Err(ReconstructionErrorV1::PredicateDependencyCycle)
        );
    }

    #[test]
    fn lower_precedence_policy_predicates_still_require_derivation_coverage() {
        let mut fixture = base_fixture();
        fixture
            .predicate_specs
            .retain(|spec| spec.predicate_id != "opposition-exists");
        assert_eq!(
            qualify_fixture_scientific_reconstruction_v1(&fixture),
            Err(ReconstructionErrorV1::MissingPolicyPredicateCoverage {
                predicate_id: "opposition-exists".into()
            })
        );
    }

    #[test]
    fn hidden_auxiliary_predicate_is_rejected_in_fixture_v1() {
        let mut fixture = base_fixture();
        fixture.predicate_specs.push(PredicateDerivationSpecFixtureV1 {
            predicate_id: "hidden-helper".into(),
            direct_predicate_dependency_ids: vec![],
            reason_ids: vec!["reason:closure".into()],
            operator: PredicateOperatorFixtureV1::AllCandidatesAccountedFor,
        });
        assert_eq!(
            qualify_fixture_scientific_reconstruction_v1(&fixture),
            Err(ReconstructionErrorV1::ExtraPredicateSpec {
                predicate_id: "hidden-helper".into()
            })
        );
    }

    #[test]
    fn unknown_reason_reference_fails_before_predicate_derivation() {
        let mut fixture = base_fixture();
        fixture.predicate_specs[0].reason_ids = vec!["reason:not-present".into()];
        assert_eq!(
            qualify_fixture_scientific_reconstruction_v1(&fixture),
            Err(ReconstructionErrorV1::UnknownPredicateReason {
                predicate_id: "support-exists".into(),
                reason_id: "reason:not-present".into()
            })
        );
    }

    #[test]
    fn closed_world_negative_predicate_carries_closure_receipt() {
        let mut fixture = base_fixture();
        fixture.predicate_specs[1].operator = PredicateOperatorFixtureV1::NoAdmittedRole {
            role: ContributionRoleV1::Opposition,
        };
        fixture.policy.rules[1].requirements[1].expected = PredicateValueV1::NotSatisfied;
        fixture.policy.rules[2].requirements[1].expected = PredicateValueV1::Satisfied;

        let reconstruction = qualify_fixture_scientific_reconstruction_v1(&fixture).unwrap();
        let opposition = reconstruction
            .predicate_receipts()
            .iter()
            .find(|receipt| receipt.predicate_id == "opposition-exists")
            .unwrap();
        assert_eq!(opposition.result, PredicateValueV1::Satisfied);
        assert_eq!(
            opposition.closed_world_receipt_id.as_deref(),
            Some(
                reconstruction
                    .material()
                    .closed_evidence_view_receipt_id
                    .as_str()
            )
        );
    }

    #[test]
    fn changed_predicate_derivation_artifact_blocks_old_persisted_record() {
        let first = qualify_fixture_scientific_reconstruction_v1(&base_fixture()).unwrap();
        let record = persisted_record(&first);

        let mut changed_fixture = base_fixture();
        changed_fixture.predicate_specs[0].reason_ids = vec![
            "reason:closure".into(),
            "reason:support".into(),
        ];
        let changed = qualify_fixture_scientific_reconstruction_v1(&changed_fixture).unwrap();
        assert_ne!(
            first.material().context.predicate_derivation_artifact_id,
            changed.material().context.predicate_derivation_artifact_id
        );
        assert_eq!(
            verify_replay_scientific_disposition_v1(&record, changed),
            Err(ReconstructionErrorV1::LowerReplay(
                ReplayErrorV1::EvaluationContextMismatch
            ))
        );
    }

    #[test]
    fn changed_evidence_view_snapshot_blocks_old_persisted_record() {
        let first = qualify_fixture_scientific_reconstruction_v1(&base_fixture()).unwrap();
        let record = persisted_record(&first);

        let mut changed_fixture = base_fixture();
        changed_fixture.evidence_view.scope_id = "different-scope".into();
        let changed = qualify_fixture_scientific_reconstruction_v1(&changed_fixture).unwrap();
        assert_eq!(
            verify_replay_scientific_disposition_v1(&record, changed),
            Err(ReconstructionErrorV1::LowerReplay(
                ReplayErrorV1::EvaluationContextMismatch
            ))
        );
    }

    #[test]
    fn same_predicate_values_with_different_dependency_graph_are_distinct() {
        let mut first_fixture = base_fixture();
        first_fixture.policy = DispositionPolicyFixtureV1 {
            profile_id: "dependency-policy".into(),
            rules: vec![DispositionRuleV1 {
                rule_id: "both".into(),
                priority: 10,
                requirements: vec![
                    RuleRequirementV1 {
                        predicate_id: "a".into(),
                        expected: PredicateValueV1::Satisfied,
                    },
                    RuleRequirementV1 {
                        predicate_id: "b".into(),
                        expected: PredicateValueV1::Satisfied,
                    },
                ],
                output: "Both".into(),
            }],
            fallback_output: "Underdetermined".into(),
        };
        first_fixture.predicate_specs = vec![
            PredicateDerivationSpecFixtureV1 {
                predicate_id: "a".into(),
                direct_predicate_dependency_ids: vec![],
                reason_ids: vec!["reason:support".into()],
                operator: PredicateOperatorFixtureV1::AnyAdmittedRole {
                    role: ContributionRoleV1::Support,
                },
            },
            PredicateDerivationSpecFixtureV1 {
                predicate_id: "b".into(),
                direct_predicate_dependency_ids: vec!["a".into()],
                reason_ids: vec!["reason:support".into()],
                operator: PredicateOperatorFixtureV1::MirrorPredicate {
                    source_predicate_id: "a".into(),
                },
            },
        ];
        let first = qualify_fixture_scientific_reconstruction_v1(&first_fixture).unwrap();

        let mut second_fixture = first_fixture;
        second_fixture.predicate_specs[1] = PredicateDerivationSpecFixtureV1 {
            predicate_id: "b".into(),
            direct_predicate_dependency_ids: vec![],
            reason_ids: vec!["reason:support".into()],
            operator: PredicateOperatorFixtureV1::AnyAdmittedRole {
                role: ContributionRoleV1::Support,
            },
        };
        let second = qualify_fixture_scientific_reconstruction_v1(&second_fixture).unwrap();

        let first_values: Vec<_> = first
            .predicate_receipts()
            .iter()
            .map(|receipt| receipt.result)
            .collect();
        let second_values: Vec<_> = second
            .predicate_receipts()
            .iter()
            .map(|receipt| receipt.result)
            .collect();
        assert_eq!(first_values, second_values);
        assert_ne!(
            first.material().predicate_dependency_graph_material_id,
            second.material().predicate_dependency_graph_material_id
        );
        assert_ne!(first.material_id(), second.material_id());
    }
}
