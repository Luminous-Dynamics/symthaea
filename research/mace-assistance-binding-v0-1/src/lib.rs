#![forbid(unsafe_code)]

//! Standalone relation-closure and response-constraint qualification kernel.
//!
//! This is not a memory store, policy authority, natural-language truth detector,
//! or signature implementation. Production hosts inject verifiers backed by the
//! Mycelix/source-authority receipts, #13967 use-time revalidation, #13973 exact
//! context exposure links, and #14071 jointly governed evidence sets.
//!
//! Digests are integrity commitments only. They do not authenticate an issuer,
//! establish that a proposition is true, or prove that the model attended to it.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION_V1: &str = "mace_assistance_binding_v1";

fn digest_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn digest_serialized<T: Serialize>(value: &T) -> Result<String, AssistanceRejectV1> {
    let bytes = serde_json::to_vec(value).map_err(|_| AssistanceRejectV1::SerializationFailure)?;
    Ok(digest_bytes(&bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationContextV1 {
    pub valid_at_ms: u64,
    pub known_at_ms: u64,
    pub scope_id: String,
    pub purpose_id: String,
    pub audience_id: String,
    pub policy_sha256: String,
}

impl RelationContextV1 {
    pub fn validate(&self) -> Result<(), AssistanceRejectV1> {
        if !nonempty(&self.scope_id)
            || !nonempty(&self.purpose_id)
            || !nonempty(&self.audience_id)
            || !valid_sha256(&self.policy_sha256)
        {
            return Err(AssistanceRejectV1::InvalidContext);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKindV1 {
    SameSubject,
    SameEvent,
    SameDecision,
    ParticipantIn,
    Supports,
    Contradicts,
    Supersedes,
    DependsOn,
    TemporalNeighbor,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRelationEdgeV1 {
    pub edge_id: String,
    pub from_evidence_id: String,
    pub to_evidence_id: String,
    pub kind: RelationKindV1,
    pub scope_id: String,
    pub purpose_id: String,
    pub source_relation_sha256: String,
    pub source_lineage_receipt_sha256: String,
    pub authority_receipt_sha256: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: u64,
    pub known_from_ms: u64,
    pub known_until_ms: u64,
}

impl EvidenceRelationEdgeV1 {
    fn validate_snapshot(&self, context: &RelationContextV1) -> Result<(), AssistanceRejectV1> {
        if !nonempty(&self.edge_id)
            || !nonempty(&self.from_evidence_id)
            || !nonempty(&self.to_evidence_id)
            || !valid_sha256(&self.source_relation_sha256)
            || !valid_sha256(&self.source_lineage_receipt_sha256)
            || !valid_sha256(&self.authority_receipt_sha256)
            || self.valid_from_ms >= self.valid_until_ms
            || self.known_from_ms >= self.known_until_ms
        {
            return Err(AssistanceRejectV1::InvalidRelationEdge);
        }
        if self.scope_id != context.scope_id || self.purpose_id != context.purpose_id {
            return Err(AssistanceRejectV1::RelationScopeMismatch);
        }
        if !(self.valid_from_ms <= context.valid_at_ms && context.valid_at_ms < self.valid_until_ms)
            || !(self.known_from_ms <= context.known_at_ms && context.known_at_ms < self.known_until_ms)
        {
            return Err(AssistanceRejectV1::RelationNotCurrentAtSnapshot);
        }
        Ok(())
    }
}

/// Production implementation must validate the actual source/authority/lineage
/// receipts. A caller-supplied verified boolean is deliberately not used.
pub trait RelationEdgeVerifierV1 {
    fn verify_relation_edge(
        &self,
        edge: &EvidenceRelationEdgeV1,
        context: &RelationContextV1,
    ) -> Result<(), AssistanceRejectV1>;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationConflictV1 {
    pub left_evidence_id: String,
    pub right_evidence_id: String,
    pub relation_edge_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationClosureReceiptV1 {
    pub schema_version: String,
    pub seed_evidence_ids: Vec<String>,
    pub selected_relation_kinds: Vec<RelationKindV1>,
    pub evidence_ids: Vec<String>,
    pub traversed_edge_ids: Vec<String>,
    pub conflict_edges: Vec<RelationConflictV1>,
    pub context_sha256: String,
    pub max_hops: u32,
    pub max_nodes: u32,
    pub closure_sha256: String,
}

fn relation_closure_digest(receipt: &RelationClosureReceiptV1) -> Result<String, AssistanceRejectV1> {
    let mut unsigned = receipt.clone();
    unsigned.closure_sha256.clear();
    digest_serialized(&unsigned)
}

/// Deterministic bounded BFS over a candidate-proposed relation graph.
///
/// Every traversed edge is snapshot-checked and passed to the injected trusted
/// verifier before its destination is admitted. Contradictions are recorded,
/// never auto-resolved. Unselected relation kinds cannot expand the closure.
pub fn build_relation_closure_v1<V: RelationEdgeVerifierV1>(
    seeds: &[String],
    edges: &[EvidenceRelationEdgeV1],
    selected_kinds: &[RelationKindV1],
    context: &RelationContextV1,
    max_hops: u32,
    max_nodes: u32,
    verifier: &V,
) -> Result<VerifiedRelationClosureV1, AssistanceRejectV1> {
    context.validate()?;
    if seeds.is_empty() || max_nodes == 0 || max_hops == 0 || seeds.len() > max_nodes as usize {
        return Err(AssistanceRejectV1::InvalidClosureBudget);
    }

    let mut seed_set = BTreeSet::new();
    for seed in seeds {
        if !nonempty(seed) || !seed_set.insert(seed.clone()) {
            return Err(AssistanceRejectV1::InvalidSeedSet);
        }
    }
    let selected: BTreeSet<RelationKindV1> = selected_kinds.iter().copied().collect();
    if selected.is_empty() {
        return Err(AssistanceRejectV1::NoRelationKindsSelected);
    }
    let mut edge_ids = BTreeSet::new();
    let mut adjacency: BTreeMap<String, Vec<&EvidenceRelationEdgeV1>> = BTreeMap::new();
    for edge in edges {
        if !edge_ids.insert(edge.edge_id.clone()) {
            return Err(AssistanceRejectV1::DuplicateRelationEdge);
        }
        if selected.contains(&edge.kind) {
            adjacency.entry(edge.from_evidence_id.clone()).or_default().push(edge);
        }
    }
    for outgoing in adjacency.values_mut() {
        outgoing.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    }

    let context_sha256 = digest_serialized(context)?;
    let mut distance: BTreeMap<String, u32> = seed_set.iter().cloned().map(|id| (id, 0)).collect();
    let mut queue: VecDeque<String> = distance.keys().cloned().collect();
    let mut traversed_edges = BTreeSet::new();
    let mut conflicts = BTreeSet::new();

    while let Some(current) = queue.pop_front() {
        let depth = *distance.get(&current).ok_or(AssistanceRejectV1::ClosureInternalInvariant)?;
        if depth >= max_hops {
            continue;
        }
        if let Some(outgoing) = adjacency.get(&current) {
            for edge in outgoing {
                edge.validate_snapshot(context)?;
                verifier.verify_relation_edge(edge, context)?;
                traversed_edges.insert(edge.edge_id.clone());
                if edge.kind == RelationKindV1::Contradicts {
                    conflicts.insert((
                        edge.from_evidence_id.clone(),
                        edge.to_evidence_id.clone(),
                        edge.edge_id.clone(),
                    ));
                }
                if !distance.contains_key(&edge.to_evidence_id) {
                    if distance.len() >= max_nodes as usize {
                        return Err(AssistanceRejectV1::ClosureNodeBudgetExceeded);
                    }
                    distance.insert(edge.to_evidence_id.clone(), depth + 1);
                    queue.push_back(edge.to_evidence_id.clone());
                }
            }
        }
    }

    let mut receipt = RelationClosureReceiptV1 {
        schema_version: SCHEMA_VERSION_V1.to_owned(),
        seed_evidence_ids: seed_set.into_iter().collect(),
        selected_relation_kinds: selected.into_iter().collect(),
        evidence_ids: distance.into_keys().collect(),
        traversed_edge_ids: traversed_edges.into_iter().collect(),
        conflict_edges: conflicts.into_iter().map(|(left, right, relation_edge_id)| {
            RelationConflictV1 {
                left_evidence_id: left,
                right_evidence_id: right,
                relation_edge_id,
            }
        }).collect(),
        context_sha256,
        max_hops,
        max_nodes,
        closure_sha256: String::new(),
    };
    receipt.closure_sha256 = relation_closure_digest(&receipt)?;
    Ok(VerifiedRelationClosureV1 { receipt })
}

impl RelationClosureReceiptV1 {
    pub fn verify_integrity(&self) -> bool {
        self.schema_version == SCHEMA_VERSION_V1
            && valid_sha256(&self.context_sha256)
            && relation_closure_digest(self).is_ok_and(|digest| digest == self.closure_sha256)
    }
}

/// In-process capability minted only after all traversed edges have passed the
/// injected verifier. It deliberately has no Serialize/Deserialize support or
/// public constructor: a re-hashed JSON receipt cannot recreate this capability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedRelationClosureV1 {
    receipt: RelationClosureReceiptV1,
}

impl VerifiedRelationClosureV1 {
    pub fn receipt(&self) -> &RelationClosureReceiptV1 {
        &self.receipt
    }

    pub fn closure_sha256(&self) -> &str {
        &self.receipt.closure_sha256
    }

    pub fn context_sha256(&self) -> &str {
        &self.receipt.context_sha256
    }

    pub fn evidence_ids(&self) -> &[String] {
        &self.receipt.evidence_ids
    }

    pub fn conflict_edges(&self) -> &[RelationConflictV1] {
        &self.receipt.conflict_edges
    }

    fn verify_integrity(&self) -> bool {
        self.receipt.verify_integrity()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimProvenanceKindV1 {
    DirectQuotation,
    SpeakerAssertion,
    FirstHandObservation,
    GroupAgreement,
    DerivedInference,
    ContestedClaim,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseConstraintActionV1 {
    MayAssert,
    MustAttribute,
    MustQualify,
    MustNotAssert,
    MustAbstain,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceResponseConstraintV1 {
    pub constraint_id: String,
    pub source_evidence_id: String,
    pub subject_id: String,
    pub source_speaker_id: Option<String>,
    pub provenance_kind: ClaimProvenanceKindV1,
    pub action: ResponseConstraintActionV1,
    pub required: bool,
    pub proposition_sha256: String,
    pub source_span_sha256: String,
    pub evidence_use_receipt_sha256: String,
    pub decision_evidence_use_link_sha256: String,
    pub source_lineage_receipt_sha256: String,
    pub relation_closure_sha256: String,
    pub scope_id: String,
    pub purpose_id: String,
    pub audience_id: String,
    pub required_qualification_sha256: Option<String>,
    pub valid_from_ms: u64,
    pub valid_until_ms: u64,
    pub known_from_ms: u64,
    pub known_until_ms: u64,
}

impl EvidenceResponseConstraintV1 {
    fn validate(&self, context: &AssistanceContextV1) -> Result<(), AssistanceRejectV1> {
        if !nonempty(&self.constraint_id)
            || !nonempty(&self.source_evidence_id)
            || !nonempty(&self.subject_id)
            || !valid_sha256(&self.proposition_sha256)
            || !valid_sha256(&self.source_span_sha256)
            || !valid_sha256(&self.evidence_use_receipt_sha256)
            || !valid_sha256(&self.decision_evidence_use_link_sha256)
            || !valid_sha256(&self.source_lineage_receipt_sha256)
            || !valid_sha256(&self.relation_closure_sha256)
            || self.valid_from_ms >= self.valid_until_ms
            || self.known_from_ms >= self.known_until_ms
        {
            return Err(AssistanceRejectV1::InvalidResponseConstraint);
        }
        if self.scope_id != context.scope_id
            || self.purpose_id != context.purpose_id
            || self.audience_id != context.audience_id
        {
            return Err(AssistanceRejectV1::ConstraintContextMismatch);
        }
        if !(self.valid_from_ms <= context.valid_at_ms && context.valid_at_ms < self.valid_until_ms)
            || !(self.known_from_ms <= context.known_at_ms && context.known_at_ms < self.known_until_ms)
        {
            return Err(AssistanceRejectV1::ConstraintNotCurrentAtSnapshot);
        }
        match self.action {
            ResponseConstraintActionV1::MustAttribute
                if self.source_speaker_id.as_deref().is_none_or(|value| !nonempty(value)) =>
            {
                return Err(AssistanceRejectV1::MissingRequiredSpeaker);
            }
            ResponseConstraintActionV1::MustQualify
                if self.required_qualification_sha256.as_deref().is_none_or(|value| !valid_sha256(value)) =>
            {
                return Err(AssistanceRejectV1::MissingRequiredQualification);
            }
            _ => {}
        }
        match self.provenance_kind {
            ClaimProvenanceKindV1::DirectQuotation | ClaimProvenanceKindV1::SpeakerAssertion
                if self.action == ResponseConstraintActionV1::MayAssert =>
            {
                return Err(AssistanceRejectV1::ProvenanceRequiresAttribution);
            }
            ClaimProvenanceKindV1::DerivedInference | ClaimProvenanceKindV1::ContestedClaim
                if self.action == ResponseConstraintActionV1::MayAssert =>
            {
                return Err(AssistanceRejectV1::ProvenanceRequiresQualification);
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssistanceContextV1 {
    pub request_sha256: String,
    pub valid_at_ms: u64,
    pub known_at_ms: u64,
    pub scope_id: String,
    pub purpose_id: String,
    pub audience_id: String,
    pub policy_sha256: String,
    pub relation_closure_sha256: String,
}

impl AssistanceContextV1 {
    pub fn validate(&self) -> Result<(), AssistanceRejectV1> {
        if !valid_sha256(&self.request_sha256)
            || !valid_sha256(&self.policy_sha256)
            || !valid_sha256(&self.relation_closure_sha256)
            || !nonempty(&self.scope_id)
            || !nonempty(&self.purpose_id)
            || !nonempty(&self.audience_id)
        {
            return Err(AssistanceRejectV1::InvalidContext);
        }
        Ok(())
    }
}

/// The independent host verifies real source receipts, lineage, rights,
/// current lifecycle, consumer policy, scope and audience. This crate only
/// enforces typed binding structure and snapshot equality.
pub trait AssistanceEvidenceVerifierV1 {
    fn verify_constraint_evidence(
        &self,
        constraint: &EvidenceResponseConstraintV1,
        context: &AssistanceContextV1,
    ) -> Result<(), AssistanceRejectV1>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseDispositionV1 {
    Answer,
    Clarify,
    Abstain,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseClaimUseV1 {
    pub constraint_id: String,
    pub proposition_sha256: String,
    pub start_byte: u64,
    pub end_byte: u64,
    pub response_span_sha256: String,
    pub speaker_id: Option<String>,
    pub qualification_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssistanceBindingReceiptV1 {
    pub schema_version: String,
    pub request_sha256: String,
    pub response_sha256: String,
    pub context_sha256: String,
    pub relation_closure_sha256: String,
    pub constraint_set_sha256: String,
    pub disposition: ResponseDispositionV1,
    pub claim_uses: Vec<ResponseClaimUseV1>,
    pub binding_sha256: String,
}

fn relation_context_digest_from_assistance(context: &AssistanceContextV1) -> Result<String, AssistanceRejectV1> {
    digest_serialized(&RelationContextV1 {
        valid_at_ms: context.valid_at_ms,
        known_at_ms: context.known_at_ms,
        scope_id: context.scope_id.clone(),
        purpose_id: context.purpose_id.clone(),
        audience_id: context.audience_id.clone(),
        policy_sha256: context.policy_sha256.clone(),
    })
}

fn assistance_binding_digest(receipt: &AssistanceBindingReceiptV1) -> Result<String, AssistanceRejectV1> {
    let mut unsigned = receipt.clone();
    unsigned.binding_sha256.clear();
    digest_serialized(&unsigned)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssistanceRejectV1 {
    InvalidContext,
    InvalidRelationEdge,
    RelationScopeMismatch,
    RelationNotCurrentAtSnapshot,
    DuplicateRelationEdge,
    InvalidClosureBudget,
    InvalidSeedSet,
    NoRelationKindsSelected,
    ClosureNodeBudgetExceeded,
    ClosureInternalInvariant,
    InvalidResponseConstraint,
    ConstraintContextMismatch,
    ConstraintNotCurrentAtSnapshot,
    MissingRequiredSpeaker,
    MissingRequiredQualification,
    ProvenanceRequiresAttribution,
    ProvenanceRequiresQualification,
    ConstraintSetEmpty,
    DuplicateConstraintId,
    ConstraintClosureMismatch,
    ConstraintSourceOutsideClosure,
    ConflictedEvidenceCannotBeUnqualified,
    EvidenceVerifierRejected,
    InvalidResponseSpan,
    ResponseSpanDigestMismatch,
    UnknownConstraint,
    ClaimPropositionMismatch,
    ClaimSpeakerMismatch,
    ClaimQualificationMismatch,
    ForbiddenConstraintUsed,
    RequiredConstraintMissing,
    MustAbstain,
    NonAnswerDispositionHasClaimUses,
    RequestDigestMismatch,
    ResponseDigestMismatch,
    ContextDigestMismatch,
    ConstraintSetDigestMismatch,
    RelationClosureDigestMismatch,
    BindingDigestMismatch,
    InvalidBinding,
    SerializationFailure,
}

impl AssistanceRejectV1 {
    pub fn code(self) -> &'static str {
        match self {
            Self::InvalidContext => "invalid_context",
            Self::InvalidRelationEdge => "invalid_relation_edge",
            Self::RelationScopeMismatch => "relation_scope_mismatch",
            Self::RelationNotCurrentAtSnapshot => "relation_not_current_at_snapshot",
            Self::DuplicateRelationEdge => "duplicate_relation_edge",
            Self::InvalidClosureBudget => "invalid_closure_budget",
            Self::InvalidSeedSet => "invalid_seed_set",
            Self::NoRelationKindsSelected => "no_relation_kinds_selected",
            Self::ClosureNodeBudgetExceeded => "closure_node_budget_exceeded",
            Self::ClosureInternalInvariant => "closure_internal_invariant",
            Self::InvalidResponseConstraint => "invalid_response_constraint",
            Self::ConstraintContextMismatch => "constraint_context_mismatch",
            Self::ConstraintNotCurrentAtSnapshot => "constraint_not_current_at_snapshot",
            Self::MissingRequiredSpeaker => "missing_required_speaker",
            Self::MissingRequiredQualification => "missing_required_qualification",
            Self::ProvenanceRequiresAttribution => "provenance_requires_attribution",
            Self::ProvenanceRequiresQualification => "provenance_requires_qualification",
            Self::ConstraintSetEmpty => "constraint_set_empty",
            Self::DuplicateConstraintId => "duplicate_constraint_id",
            Self::ConstraintClosureMismatch => "constraint_closure_mismatch",
            Self::ConstraintSourceOutsideClosure => "constraint_source_outside_closure",
            Self::ConflictedEvidenceCannotBeUnqualified => "conflicted_evidence_cannot_be_unqualified",
            Self::EvidenceVerifierRejected => "evidence_verifier_rejected",
            Self::InvalidResponseSpan => "invalid_response_span",
            Self::ResponseSpanDigestMismatch => "response_span_digest_mismatch",
            Self::UnknownConstraint => "unknown_constraint",
            Self::ClaimPropositionMismatch => "claim_proposition_mismatch",
            Self::ClaimSpeakerMismatch => "claim_speaker_mismatch",
            Self::ClaimQualificationMismatch => "claim_qualification_mismatch",
            Self::ForbiddenConstraintUsed => "forbidden_constraint_used",
            Self::RequiredConstraintMissing => "required_constraint_missing",
            Self::MustAbstain => "must_abstain",
            Self::NonAnswerDispositionHasClaimUses => "non_answer_disposition_has_claim_uses",
            Self::RequestDigestMismatch => "request_digest_mismatch",
            Self::ResponseDigestMismatch => "response_digest_mismatch",
            Self::ContextDigestMismatch => "context_digest_mismatch",
            Self::ConstraintSetDigestMismatch => "constraint_set_digest_mismatch",
            Self::RelationClosureDigestMismatch => "relation_closure_digest_mismatch",
            Self::BindingDigestMismatch => "binding_digest_mismatch",
            Self::InvalidBinding => "invalid_binding",
            Self::SerializationFailure => "serialization_failure",
        }
    }
}

pub struct VerifiedAssistanceBindingV1 {
    binding_sha256: String,
}

impl VerifiedAssistanceBindingV1 {
    pub fn binding_sha256(&self) -> &str {
        &self.binding_sha256
    }
}

pub fn verify_assistance_binding_v1<V: AssistanceEvidenceVerifierV1>(
    receipt: &AssistanceBindingReceiptV1,
    request_bytes: &[u8],
    response_bytes: &[u8],
    context: &AssistanceContextV1,
    constraints: &[EvidenceResponseConstraintV1],
    closure: &VerifiedRelationClosureV1,
    verifier: &V,
) -> Result<VerifiedAssistanceBindingV1, AssistanceRejectV1> {
    context.validate()?;
    if !closure.verify_integrity()
        || closure.closure_sha256() != context.relation_closure_sha256
        || closure.context_sha256() != relation_context_digest_from_assistance(context)?
    {
        return Err(AssistanceRejectV1::RelationClosureDigestMismatch);
    }
    if digest_bytes(request_bytes) != context.request_sha256 {
        return Err(AssistanceRejectV1::RequestDigestMismatch);
    }
    if constraints.is_empty() {
        return Err(AssistanceRejectV1::ConstraintSetEmpty);
    }

    let mut by_id = BTreeMap::new();
    for constraint in constraints {
        constraint.validate(context)?;
        if constraint.relation_closure_sha256 != closure.closure_sha256() {
            return Err(AssistanceRejectV1::ConstraintClosureMismatch);
        }
        if closure.evidence_ids().binary_search(&constraint.source_evidence_id).is_err() {
            return Err(AssistanceRejectV1::ConstraintSourceOutsideClosure);
        }
        if closure.conflict_edges().iter().any(|edge| {
            (edge.left_evidence_id == constraint.source_evidence_id
                || edge.right_evidence_id == constraint.source_evidence_id)
                && constraint.action == ResponseConstraintActionV1::MayAssert
        }) {
            return Err(AssistanceRejectV1::ConflictedEvidenceCannotBeUnqualified);
        }
        verifier.verify_constraint_evidence(constraint, context)
            .map_err(|_| AssistanceRejectV1::EvidenceVerifierRejected)?;
        if by_id.insert(constraint.constraint_id.as_str(), constraint).is_some() {
            return Err(AssistanceRejectV1::DuplicateConstraintId);
        }
    }

    let context_sha256 = digest_serialized(context)?;
    let constraint_set_sha256 = digest_serialized(constraints)?;
    let response_sha256 = digest_bytes(response_bytes);
    if receipt.schema_version != SCHEMA_VERSION_V1 {
        return Err(AssistanceRejectV1::InvalidBinding);
    }
    if receipt.request_sha256 != context.request_sha256 {
        return Err(AssistanceRejectV1::RequestDigestMismatch);
    }
    if receipt.response_sha256 != response_sha256 {
        return Err(AssistanceRejectV1::ResponseDigestMismatch);
    }
    if receipt.context_sha256 != context_sha256 {
        return Err(AssistanceRejectV1::ContextDigestMismatch);
    }
    if receipt.constraint_set_sha256 != constraint_set_sha256 {
        return Err(AssistanceRejectV1::ConstraintSetDigestMismatch);
    }
    if receipt.relation_closure_sha256 != closure.closure_sha256() {
        return Err(AssistanceRejectV1::RelationClosureDigestMismatch);
    }

    if receipt.disposition != ResponseDispositionV1::Answer && !receipt.claim_uses.is_empty() {
        return Err(AssistanceRejectV1::NonAnswerDispositionHasClaimUses);
    }
    if receipt.disposition != ResponseDispositionV1::Abstain
        && constraints.iter().any(|constraint| {
            constraint.required && constraint.action == ResponseConstraintActionV1::MustAbstain
        })
    {
        return Err(AssistanceRejectV1::MustAbstain);
    }

    let mut used_required = BTreeSet::new();
    // Multiple independently qualified sources may support the same exact
    // response claim span. Partial overlaps are rejected to avoid ambiguous
    // claim-to-evidence attribution.
    let mut spans: BTreeSet<(usize, usize, String)> = BTreeSet::new();
    for claim_use in &receipt.claim_uses {
        let constraint = by_id.get(claim_use.constraint_id.as_str())
            .ok_or(AssistanceRejectV1::UnknownConstraint)?;
        if !valid_sha256(&claim_use.proposition_sha256)
            || claim_use.proposition_sha256 != constraint.proposition_sha256
        {
            return Err(AssistanceRejectV1::ClaimPropositionMismatch);
        }
        let response_text = std::str::from_utf8(response_bytes)
            .map_err(|_| AssistanceRejectV1::InvalidResponseSpan)?;
        let start = usize::try_from(claim_use.start_byte)
            .map_err(|_| AssistanceRejectV1::InvalidResponseSpan)?;
        let end = usize::try_from(claim_use.end_byte)
            .map_err(|_| AssistanceRejectV1::InvalidResponseSpan)?;
        if start >= end
            || end > response_bytes.len()
            || !response_text.is_char_boundary(start)
            || !response_text.is_char_boundary(end)
        {
            return Err(AssistanceRejectV1::InvalidResponseSpan);
        }
        if spans.iter().any(|(prior_start, prior_end, _)| {
            start < *prior_end && *prior_start < end && (start, end) != (*prior_start, *prior_end)
        }) {
            return Err(AssistanceRejectV1::InvalidResponseSpan);
        }
        if !spans.insert((start, end, claim_use.constraint_id.clone())) {
            return Err(AssistanceRejectV1::InvalidResponseSpan);
        }
        if digest_bytes(&response_bytes[start..end]) != claim_use.response_span_sha256 {
            return Err(AssistanceRejectV1::ResponseSpanDigestMismatch);
        }
        match constraint.action {
            ResponseConstraintActionV1::MustNotAssert | ResponseConstraintActionV1::MustAbstain => {
                return Err(AssistanceRejectV1::ForbiddenConstraintUsed);
            }
            ResponseConstraintActionV1::MustAttribute => {
                if claim_use.speaker_id.as_deref() != constraint.source_speaker_id.as_deref() {
                    return Err(AssistanceRejectV1::ClaimSpeakerMismatch);
                }
            }
            ResponseConstraintActionV1::MustQualify => {
                if claim_use.qualification_sha256 != constraint.required_qualification_sha256 {
                    return Err(AssistanceRejectV1::ClaimQualificationMismatch);
                }
            }
            ResponseConstraintActionV1::MayAssert => {}
        }
        used_required.insert(constraint.constraint_id.as_str());
    }

    if receipt.disposition == ResponseDispositionV1::Answer {
        for constraint in constraints.iter().filter(|constraint| {
            constraint.required && matches!(
                constraint.action,
                ResponseConstraintActionV1::MayAssert
                    | ResponseConstraintActionV1::MustAttribute
                    | ResponseConstraintActionV1::MustQualify
            )
        }) {
            if !used_required.contains(constraint.constraint_id.as_str()) {
                return Err(AssistanceRejectV1::RequiredConstraintMissing);
            }
        }
    }

    let claimed = receipt.binding_sha256.clone();
    if !valid_sha256(&claimed) || assistance_binding_digest(receipt)? != claimed {
        return Err(AssistanceRejectV1::BindingDigestMismatch);
    }
    Ok(VerifiedAssistanceBindingV1 { binding_sha256: claimed })
}

pub fn seal_assistance_binding_v1(
    context: &AssistanceContextV1,
    constraints: &[EvidenceResponseConstraintV1],
    closure: &VerifiedRelationClosureV1,
    response_bytes: &[u8],
    disposition: ResponseDispositionV1,
    claim_uses: Vec<ResponseClaimUseV1>,
) -> Result<AssistanceBindingReceiptV1, AssistanceRejectV1> {
    context.validate()?;
    if !closure.verify_integrity()
        || context.relation_closure_sha256 != closure.closure_sha256()
        || closure.context_sha256() != relation_context_digest_from_assistance(context)?
    {
        return Err(AssistanceRejectV1::RelationClosureDigestMismatch);
    }
    let mut receipt = AssistanceBindingReceiptV1 {
        schema_version: SCHEMA_VERSION_V1.to_owned(),
        request_sha256: context.request_sha256.clone(),
        response_sha256: digest_bytes(response_bytes),
        context_sha256: digest_serialized(context)?,
        relation_closure_sha256: closure.closure_sha256().to_owned(),
        constraint_set_sha256: digest_serialized(constraints)?,
        disposition,
        claim_uses,
        binding_sha256: String::new(),
    };
    receipt.binding_sha256 = assistance_binding_digest(&receipt)?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct FixtureVerifier;

    impl RelationEdgeVerifierV1 for FixtureVerifier {
        fn verify_relation_edge(
            &self,
            edge: &EvidenceRelationEdgeV1,
            _context: &RelationContextV1,
        ) -> Result<(), AssistanceRejectV1> {
            if edge.authority_receipt_sha256 == digest_bytes(b"trusted-authority") {
                Ok(())
            } else {
                Err(AssistanceRejectV1::EvidenceVerifierRejected)
            }
        }
    }

    impl AssistanceEvidenceVerifierV1 for FixtureVerifier {
        fn verify_constraint_evidence(
            &self,
            constraint: &EvidenceResponseConstraintV1,
            _context: &AssistanceContextV1,
        ) -> Result<(), AssistanceRejectV1> {
            if constraint.evidence_use_receipt_sha256 == digest_bytes(b"use")
                && constraint.decision_evidence_use_link_sha256 == digest_bytes(b"exposure")
                && constraint.source_lineage_receipt_sha256 == digest_bytes(b"lineage")
            {
                Ok(())
            } else {
                Err(AssistanceRejectV1::EvidenceVerifierRejected)
            }
        }
    }

    fn sha(value: &str) -> String {
        digest_bytes(value.as_bytes())
    }

    fn relation_context() -> RelationContextV1 {
        RelationContextV1 {
            valid_at_ms: 100,
            known_at_ms: 100,
            scope_id: "group:alpha".into(),
            purpose_id: "assist:meeting".into(),
            audience_id: "group:alpha:members".into(),
            policy_sha256: sha("policy"),
        }
    }

    fn relation_edge(
        edge_id: &str,
        from: &str,
        to: &str,
        kind: RelationKindV1,
    ) -> EvidenceRelationEdgeV1 {
        EvidenceRelationEdgeV1 {
            edge_id: edge_id.into(),
            from_evidence_id: from.into(),
            to_evidence_id: to.into(),
            kind,
            scope_id: "group:alpha".into(),
            purpose_id: "assist:meeting".into(),
            source_relation_sha256: sha(&format!("relation:{edge_id}")),
            source_lineage_receipt_sha256: sha("lineage"),
            authority_receipt_sha256: sha("trusted-authority"),
            valid_from_ms: 0,
            valid_until_ms: 1_000,
            known_from_ms: 0,
            known_until_ms: 1_000,
        }
    }

    fn closure() -> VerifiedRelationClosureV1 {
        build_relation_closure_v1(
            &vec!["e1".into()],
            &vec![
                relation_edge("r1", "e1", "e2", RelationKindV1::SameDecision),
                relation_edge("r2", "e2", "e3", RelationKindV1::Contradicts),
                relation_edge("r3", "e1", "e1", RelationKindV1::SameSubject),
                relation_edge("r4", "e1", "e4", RelationKindV1::Supports),
            ],
            &vec![
                RelationKindV1::SameDecision,
                RelationKindV1::Contradicts,
                RelationKindV1::SameSubject,
                RelationKindV1::Supports,
            ],
            &relation_context(),
            3,
            8,
            &FixtureVerifier,
        ).unwrap()
    }

    fn assistance_context(closure: &VerifiedRelationClosureV1) -> AssistanceContextV1 {
        AssistanceContextV1 {
            request_sha256: sha("request"),
            valid_at_ms: 100,
            known_at_ms: 100,
            scope_id: "group:alpha".into(),
            purpose_id: "assist:meeting".into(),
            audience_id: "group:alpha:members".into(),
            policy_sha256: sha("policy"),
            relation_closure_sha256: closure.closure_sha256().to_owned(),
        }
    }

    fn constraint(
        action: ResponseConstraintActionV1,
        required: bool,
        closure: &VerifiedRelationClosureV1,
    ) -> EvidenceResponseConstraintV1 {
        EvidenceResponseConstraintV1 {
            constraint_id: sha("constraint"),
            source_evidence_id: "e1".into(),
            subject_id: "participant:alice".into(),
            source_speaker_id: Some("speaker:alice".into()),
            provenance_kind: if action == ResponseConstraintActionV1::MustAttribute {
                ClaimProvenanceKindV1::SpeakerAssertion
            } else {
                ClaimProvenanceKindV1::FirstHandObservation
            },
            action,
            required,
            proposition_sha256: sha("proposition"),
            source_span_sha256: sha("source-span"),
            evidence_use_receipt_sha256: sha("use"),
            decision_evidence_use_link_sha256: sha("exposure"),
            source_lineage_receipt_sha256: sha("lineage"),
            relation_closure_sha256: closure.closure_sha256().to_owned(),
            scope_id: "group:alpha".into(),
            purpose_id: "assist:meeting".into(),
            audience_id: "group:alpha:members".into(),
            required_qualification_sha256: (action == ResponseConstraintActionV1::MustQualify)
                .then(|| sha("qualification")),
            valid_from_ms: 0,
            valid_until_ms: 1_000,
            known_from_ms: 0,
            known_until_ms: 1_000,
        }
    }

    fn response_binding(
        context: &AssistanceContextV1,
        constraints: &[EvidenceResponseConstraintV1],
        closure: &VerifiedRelationClosureV1,
        bytes: &[u8],
        disposition: ResponseDispositionV1,
        uses: Vec<ResponseClaimUseV1>,
    ) -> AssistanceBindingReceiptV1 {
        seal_assistance_binding_v1(context, constraints, closure, bytes, disposition, uses).unwrap()
    }

    fn claim_use(
        constraint: &EvidenceResponseConstraintV1,
        bytes: &[u8],
        speaker: Option<&str>,
        qualification: Option<String>,
    ) -> ResponseClaimUseV1 {
        ResponseClaimUseV1 {
            constraint_id: constraint.constraint_id.clone(),
            proposition_sha256: constraint.proposition_sha256.clone(),
            start_byte: 0,
            end_byte: bytes.len() as u64,
            response_span_sha256: digest_bytes(bytes),
            speaker_id: speaker.map(str::to_owned),
            qualification_sha256: qualification,
        }
    }

    #[test]
    fn relation_closure_is_bounded_deterministic_and_preserves_conflict() {
        let result = closure();
        assert_eq!(result.receipt().evidence_ids, vec!["e1".to_owned(), "e2".to_owned(), "e3".to_owned(), "e4".to_owned()]);
        assert_eq!(result.receipt().traversed_edge_ids, vec!["r1".to_owned(), "r2".to_owned(), "r3".to_owned(), "r4".to_owned()]);
        assert_eq!(result.receipt().conflict_edges.len(), 1);
        assert!(result.receipt().verify_integrity());
    }

    #[test]
    fn relation_closure_rejects_scope_mismatch_and_untrusted_authority() {
        let mut bad = relation_edge("bad", "e1", "e2", RelationKindV1::SameDecision);
        bad.scope_id = "group:other".into();
        assert_eq!(
            build_relation_closure_v1(
                &vec!["e1".into()], &vec![bad], &vec![RelationKindV1::SameDecision],
                &relation_context(), 2, 4, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::RelationScopeMismatch)
        );
        let mut untrusted = relation_edge("bad2", "e1", "e2", RelationKindV1::SameDecision);
        untrusted.authority_receipt_sha256 = sha("untrusted");
        assert_eq!(
            build_relation_closure_v1(
                &vec!["e1".into()], &vec![untrusted], &vec![RelationKindV1::SameDecision],
                &relation_context(), 2, 4, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::EvidenceVerifierRejected)
        );
    }

    #[test]
    fn relation_closure_rejects_expired_and_future_known_edges() {
        let mut stale = relation_edge("stale", "e1", "e2", RelationKindV1::SameDecision);
        stale.valid_until_ms = 50;
        assert_eq!(
            build_relation_closure_v1(
                &vec!["e1".into()], &vec![stale], &vec![RelationKindV1::SameDecision],
                &relation_context(), 2, 4, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::RelationNotCurrentAtSnapshot)
        );
        let mut future = relation_edge("future", "e1", "e2", RelationKindV1::SameDecision);
        future.known_from_ms = 150;
        assert_eq!(
            build_relation_closure_v1(
                &vec!["e1".into()], &vec![future], &vec![RelationKindV1::SameDecision],
                &relation_context(), 2, 4, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::RelationNotCurrentAtSnapshot)
        );
    }

    #[test]
    fn relation_closure_rejects_node_budget_overflow_without_partial_receipt() {
        let edges = vec![
            relation_edge("a", "e1", "e2", RelationKindV1::SameDecision),
            relation_edge("b", "e2", "e3", RelationKindV1::SameDecision),
        ];
        assert_eq!(
            build_relation_closure_v1(
                &vec!["e1".into()], &edges, &vec![RelationKindV1::SameDecision],
                &relation_context(), 3, 2, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::ClosureNodeBudgetExceeded)
        );
    }

    #[test]
    fn valid_response_binding_requires_exact_span_and_provenance_links() {
        let closure = closure();
        let context = assistance_context(&closure);
        let allowed = constraint(ResponseConstraintActionV1::MustAttribute, true, &closure);
        let bytes = b"Alice said the delivery date is Friday.";
        let use_row = claim_use(&allowed, bytes, Some("speaker:alice"), None);
        let receipt = response_binding(
            &context, &vec![allowed.clone()], &closure, bytes,
            ResponseDispositionV1::Answer, vec![use_row]
        );
        let verified = verify_assistance_binding_v1(
            &receipt, b"request", bytes, &context, &vec![allowed], &closure, &FixtureVerifier
        ).unwrap();
        assert_eq!(verified.binding_sha256(), receipt.binding_sha256.as_str());
    }

    #[test]
    fn attribution_must_not_promote_speaker_claim_to_world_fact() {
        let closure = closure();
        let context = assistance_context(&closure);
        let allowed = constraint(ResponseConstraintActionV1::MustAttribute, true, &closure);
        let bytes = b"The delivery date is Friday.";
        let receipt = response_binding(
            &context, &vec![allowed.clone()], &closure, bytes,
            ResponseDispositionV1::Answer,
            vec![claim_use(&allowed, bytes, Some("speaker:bob"), None)]
        );
        assert_eq!(
            verify_assistance_binding_v1(
                &receipt, b"request", bytes, &context, &vec![allowed], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::ClaimSpeakerMismatch)
        );
    }

    #[test]
    fn required_qualification_must_be_bound_to_the_claim_use() {
        let closure = closure();
        let context = assistance_context(&closure);
        let qualified = constraint(ResponseConstraintActionV1::MustQualify, true, &closure);
        let bytes = b"The revised estimate may be about 20 units.";
        let receipt = response_binding(
            &context, &vec![qualified.clone()], &closure, bytes,
            ResponseDispositionV1::Answer,
            vec![claim_use(&qualified, bytes, None, Some(sha("wrong")))]
        );
        assert_eq!(
            verify_assistance_binding_v1(
                &receipt, b"request", bytes, &context, &vec![qualified], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::ClaimQualificationMismatch)
        );
    }

    #[test]
    fn two_independent_constraints_may_support_the_same_exact_claim_span() {
        let closure = closure();
        let context = assistance_context(&closure);
        let first = constraint(ResponseConstraintActionV1::MayAssert, true, &closure);
        let mut second = constraint(ResponseConstraintActionV1::MayAssert, true, &closure);
        second.constraint_id = sha("constraint-2");
        second.source_evidence_id = "e4".into();
        let bytes = b"The agreed review is scheduled for Friday.";
        let receipt = response_binding(
            &context,
            &vec![first.clone(), second.clone()],
            &closure,
            bytes,
            ResponseDispositionV1::Answer,
            vec![
                claim_use(&first, bytes, None, None),
                claim_use(&second, bytes, None, None),
            ],
        );
        assert!(verify_assistance_binding_v1(
            &receipt,
            b"request",
            bytes,
            &context,
            &vec![first, second],
            &closure,
            &FixtureVerifier,
        ).is_ok());
    }

    #[test]
    fn relation_closure_cannot_be_replayed_under_a_different_audience_snapshot() {
        let closure = closure();
        let mut changed_context = assistance_context(&closure);
        changed_context.audience_id = "private:alice".into();
        assert_eq!(
            seal_assistance_binding_v1(
                &changed_context,
                &vec![constraint(ResponseConstraintActionV1::MayAssert, false, &closure)],
                &closure,
                b"claim",
                ResponseDispositionV1::Answer,
                Vec::new(),
            ),
            Err(AssistanceRejectV1::RelationClosureDigestMismatch)
        );
    }

    #[test]
    fn required_constraint_cannot_be_silently_omitted_from_answer() {
        let closure = closure();
        let context = assistance_context(&closure);
        let required = constraint(ResponseConstraintActionV1::MayAssert, true, &closure);
        let bytes = b"Here is the answer.";
        let receipt = response_binding(
            &context, &vec![required.clone()], &closure, bytes,
            ResponseDispositionV1::Answer, Vec::new()
        );
        assert_eq!(
            verify_assistance_binding_v1(
                &receipt, b"request", bytes, &context, &vec![required], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::RequiredConstraintMissing)
        );
    }

    #[test]
    fn required_must_abstain_constraint_blocks_answer() {
        let closure = closure();
        let context = assistance_context(&closure);
        let must_abstain = constraint(ResponseConstraintActionV1::MustAbstain, true, &closure);
        let bytes = b"Here is the answer.";
        let receipt = response_binding(
            &context, &vec![must_abstain.clone()], &closure, bytes,
            ResponseDispositionV1::Answer, Vec::new()
        );
        assert_eq!(
            verify_assistance_binding_v1(
                &receipt, b"request", bytes, &context, &vec![must_abstain], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::MustAbstain)
        );
    }

    #[test]
    fn forbidden_claim_constraint_cannot_be_cited_as_support() {
        let closure = closure();
        let context = assistance_context(&closure);
        let forbidden = constraint(ResponseConstraintActionV1::MustNotAssert, false, &closure);
        let bytes = b"The project has already been approved.";
        let receipt = response_binding(
            &context, &vec![forbidden.clone()], &closure, bytes,
            ResponseDispositionV1::Answer,
            vec![claim_use(&forbidden, bytes, None, None)]
        );
        assert_eq!(
            verify_assistance_binding_v1(
                &receipt, b"request", bytes, &context, &vec![forbidden], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::ForbiddenConstraintUsed)
        );
    }

    #[test]
    fn response_constraint_source_must_belong_to_relation_closure() {
        let closure = closure();
        let context = assistance_context(&closure);
        let mut outside = constraint(ResponseConstraintActionV1::MayAssert, true, &closure);
        outside.source_evidence_id = "not-in-closure".into();
        let bytes = b"Some statement.";
        let receipt = response_binding(
            &context, &vec![outside.clone()], &closure, bytes,
            ResponseDispositionV1::Answer, Vec::new()
        );
        assert_eq!(
            verify_assistance_binding_v1(
                &receipt, b"request", bytes, &context, &vec![outside], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::ConstraintSourceOutsideClosure)
        );
    }

    #[test]
    fn relation_closure_cannot_be_reused_after_bitemporal_snapshot_advances() {
        let closure = closure();
        let mut changed_context = assistance_context(&closure);
        changed_context.known_at_ms = 200;
        assert_eq!(
            seal_assistance_binding_v1(
                &changed_context,
                &vec![constraint(ResponseConstraintActionV1::MayAssert, false, &closure)],
                &closure,
                b"claim",
                ResponseDispositionV1::Answer,
                Vec::new(),
            ),
            Err(AssistanceRejectV1::RelationClosureDigestMismatch)
        );
    }

    #[test]
    fn conflicted_evidence_cannot_be_bound_as_an_unqualified_assertion() {
        let closure = closure();
        let context = assistance_context(&closure);
        let mut disputed = constraint(ResponseConstraintActionV1::MayAssert, true, &closure);
        disputed.source_evidence_id = "e2".into();
        let bytes = b"The decision is final.";
        let receipt = response_binding(
            &context, &vec![disputed.clone()], &closure, bytes,
            ResponseDispositionV1::Answer, Vec::new()
        );
        assert_eq!(
            verify_assistance_binding_v1(
                &receipt, b"request", bytes, &context, &vec![disputed], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::ConflictedEvidenceCannotBeUnqualified)
        );
    }

    #[test]
    fn inferred_or_contested_claims_cannot_be_marked_may_assert() {
        let closure = closure();
        let context = assistance_context(&closure);
        let mut inference = constraint(ResponseConstraintActionV1::MayAssert, true, &closure);
        inference.provenance_kind = ClaimProvenanceKindV1::DerivedInference;
        assert_eq!(
            verify_assistance_binding_v1(
                &response_binding(&context, &vec![inference.clone()], &closure, b"claim",
                    ResponseDispositionV1::Answer, Vec::new()),
                b"request", b"claim", &context, &vec![inference], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::ProvenanceRequiresQualification)
        );

        let mut speaker_claim = constraint(ResponseConstraintActionV1::MayAssert, true, &closure);
        speaker_claim.provenance_kind = ClaimProvenanceKindV1::SpeakerAssertion;
        assert_eq!(
            verify_assistance_binding_v1(
                &response_binding(&context, &vec![speaker_claim.clone()], &closure, b"claim",
                    ResponseDispositionV1::Answer, Vec::new()),
                b"request", b"claim", &context, &vec![speaker_claim], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::ProvenanceRequiresAttribution)
        );
    }

    #[test]
    fn audience_mismatch_and_stale_constraints_fail_closed() {
        let closure = closure();
        let context = assistance_context(&closure);
        let mut wrong_audience = constraint(ResponseConstraintActionV1::MayAssert, true, &closure);
        wrong_audience.audience_id = "private:alice".into();
        assert_eq!(
            verify_assistance_binding_v1(
                &response_binding(&context, &vec![wrong_audience.clone()], &closure, b"claim",
                    ResponseDispositionV1::Answer, Vec::new()),
                b"request", b"claim", &context, &vec![wrong_audience], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::ConstraintContextMismatch)
        );
        let mut stale = constraint(ResponseConstraintActionV1::MayAssert, true, &closure);
        stale.known_until_ms = 50;
        assert_eq!(
            verify_assistance_binding_v1(
                &response_binding(&context, &vec![stale.clone()], &closure, b"claim",
                    ResponseDispositionV1::Answer, Vec::new()),
                b"request", b"claim", &context, &vec![stale], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::ConstraintNotCurrentAtSnapshot)
        );
    }

    #[test]
    fn non_answer_disposition_cannot_carry_claim_uses() {
        let closure = closure();
        let context = assistance_context(&closure);
        let allowed = constraint(ResponseConstraintActionV1::MayAssert, false, &closure);
        let bytes = b"Clarify please.";
        let receipt = response_binding(
            &context, &vec![allowed.clone()], &closure, bytes,
            ResponseDispositionV1::Clarify,
            vec![claim_use(&allowed, bytes, None, None)]
        );
        assert_eq!(
            verify_assistance_binding_v1(
                &receipt, b"request", bytes, &context, &vec![allowed], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::NonAnswerDispositionHasClaimUses)
        );
    }

    #[test]
    fn unicode_response_spans_must_be_utf8_boundaries_and_digest_exact_bytes() {
        let closure = closure();
        let context = assistance_context(&closure);
        let allowed = constraint(ResponseConstraintActionV1::MayAssert, true, &closure);
        let bytes = "The plan is café.".as_bytes();
        let mut invalid = claim_use(&allowed, bytes, None, None);
        invalid.start_byte = 16; // inside the two-byte UTF-8 encoding for é
        invalid.end_byte = 17;
        let receipt = response_binding(
            &context, &vec![allowed.clone()], &closure, bytes,
            ResponseDispositionV1::Answer, vec![invalid]
        );
        assert_eq!(
            verify_assistance_binding_v1(
                &receipt, b"request", bytes, &context, &vec![allowed], &closure, &FixtureVerifier
            ),
            Err(AssistanceRejectV1::InvalidResponseSpan)
        );
    }
}
