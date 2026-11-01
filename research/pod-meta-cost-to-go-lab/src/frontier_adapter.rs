use std::collections::{BTreeMap, BTreeSet};

use pod_meta_lab::graph::{
    ClaimNode, EvidenceLifecycle, EvidenceRevision, PlannerCase, Polarity, ProofAlternative,
    ProofRequirement, QueryProfile, SourceIntegrity,
};
use pod_meta_lab::{Obligation, ObligationState};

use crate::{
    FrontierCostPrediction, FrontierModelError, LinearProofFrontierCostModel,
    ProofFrontierFeatures, ShiftStatus,
};

/// Execution facts absent from the planner graph. Source these from the
/// frozen pre-action operation plan; do not infer them from claim structure.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrontierAdapterContext {
    pub operation_family_ids: Vec<u32>,
    pub mandatory_verifier_operations: u64,
}

/// Canonical snapshot for one proof frontier.
///
/// The fingerprint is a deterministic, non-cryptographic identity checksum.
/// It is not an authentication tag, collision-resistant commitment, or signed
/// state receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontierSnapshot {
    pub features: ProofFrontierFeatures,
    pub fingerprint: [u8; 32],
    pub canonical_bytes: Vec<u8>,
}

impl FrontierSnapshot {
    /// Predict from this snapshot so the feature vector and frontier identity
    /// cannot be paired independently by the calling code.
    pub fn predict_cost_to_go(
        &self,
        model: LinearProofFrontierCostModel,
        population_cohort_digest: [u8; 32],
        shift_status: ShiftStatus,
    ) -> Result<FrontierCostPrediction, FrontierModelError> {
        model.predict(
            self.features,
            self.fingerprint,
            population_cohort_digest,
            shift_status,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrontierAdapterError {
    EmptyCaseId,
    EmptyQueryScope,
    EmptyQueryAuthority,
    NoTargets,
    DuplicateTarget(String),
    DuplicateClaim(String),
    DuplicateAlternative {
        claim_id: String,
        alternative_id: String,
    },
    EmptyAlternativeId {
        claim_id: String,
    },
    DuplicateEvidence(String),
    DuplicateObligation(String),
    MissingTargetClaim(String),
    MissingDependencyClaim {
        parent: String,
        dependency: String,
    },
    MissingObligation(String),
    MissingEvidence(String),
    EvidenceSubjectMismatch {
        claim_id: String,
        evidence_id: String,
    },
    EmptyEvidenceIdentity(String),
    EmptyObligationStateIdentity(String),
    DuplicateRequirement {
        claim_id: String,
        alternative_id: String,
    },
    DependencyCycle,
    CounterOverflow,
    ExecutionContextForEmptyFrontier,
    InvalidFeatureCounts,
}

impl std::fmt::Display for FrontierAdapterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for FrontierAdapterError {}

/// Derive cost-to-go features from a typed planner graph and scheduler
/// obligation states.
///
/// V1 requires Obligation.id == ClaimNode.id. Only unresolved claims reachable
/// from unresolved targets are traversed. A supported dependency is a
/// discharged leaf. Every reachable unresolved claim must have a non-empty
/// state identity.
///
/// An alternative is counted as locally admissible when every direct evidence
/// requirement is metadata-admissible and its direct evidence satisfies the
/// declared independence-group minimum. Claim dependencies must exist, but
/// their truth is not inferred here. The adapter does not read private oracle
/// labels and does not certify a proof path.
pub fn extract_frontier_snapshot(
    case: &PlannerCase,
    obligations: &[Obligation],
    context: &FrontierAdapterContext,
) -> Result<FrontierSnapshot, FrontierAdapterError> {
    if case.id.is_empty() {
        return Err(FrontierAdapterError::EmptyCaseId);
    }
    if case.query.required_scope.is_empty() {
        return Err(FrontierAdapterError::EmptyQueryScope);
    }
    if case.query.required_authority.is_empty() {
        return Err(FrontierAdapterError::EmptyQueryAuthority);
    }
    if case.target_claim_ids.is_empty() {
        return Err(FrontierAdapterError::NoTargets);
    }

    let mut claims_by_id = BTreeMap::<&str, &ClaimNode>::new();
    for claim in &case.claims {
        if claims_by_id.insert(claim.id.as_str(), claim).is_some() {
            return Err(FrontierAdapterError::DuplicateClaim(claim.id.clone()));
        }
    }

    let mut evidence_by_id = BTreeMap::<&str, &EvidenceRevision>::new();
    for evidence in &case.evidence {
        if evidence_by_id
            .insert(evidence.id.as_str(), evidence)
            .is_some()
        {
            return Err(FrontierAdapterError::DuplicateEvidence(evidence.id.clone()));
        }
    }

    let mut obligations_by_id = BTreeMap::<&str, &Obligation>::new();
    for obligation in obligations {
        if obligations_by_id
            .insert(obligation.id.as_str(), obligation)
            .is_some()
        {
            return Err(FrontierAdapterError::DuplicateObligation(
                obligation.id.clone(),
            ));
        }
    }

    let mut target_ids = case
        .target_claim_ids
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    target_ids.sort_unstable();
    for pair in target_ids.windows(2) {
        if pair[0] == pair[1] {
            return Err(FrontierAdapterError::DuplicateTarget(pair[0].to_owned()));
        }
    }

    let mut unresolved_targets = BTreeSet::<String>::new();
    let mut observed_claims = BTreeSet::<String>::new();
    for target_id in &target_ids {
        if !claims_by_id.contains_key(target_id) {
            return Err(FrontierAdapterError::MissingTargetClaim(
                (*target_id).to_owned(),
            ));
        }
        let obligation = obligations_by_id
            .get(target_id)
            .ok_or_else(|| FrontierAdapterError::MissingObligation((*target_id).to_owned()))?;
        observed_claims.insert((*target_id).to_owned());
        if obligation.state != ObligationState::SupportedUnderProfile {
            unresolved_targets.insert((*target_id).to_owned());
        }
    }

    let mut pending = unresolved_targets.iter().cloned().collect::<Vec<_>>();
    let mut active_claims = BTreeSet::<String>::new();
    let mut adjacency = BTreeMap::<String, BTreeSet<String>>::new();
    let mut referenced_evidence = BTreeSet::<String>::new();
    let mut available_evidence_families = BTreeSet::<String>::new();
    let mut dependency_edges = 0_u64;
    let mut admissible_alternatives = 0_u64;
    let mut missing_evidence_requirements = 0_u64;

    while let Some(claim_id) = pending.pop() {
        if !active_claims.insert(claim_id.clone()) {
            continue;
        }

        let claim = claims_by_id
            .get(claim_id.as_str())
            .copied()
            .ok_or_else(|| FrontierAdapterError::MissingTargetClaim(claim_id.clone()))?;
        let obligation = obligations_by_id
            .get(claim_id.as_str())
            .copied()
            .ok_or_else(|| FrontierAdapterError::MissingObligation(claim_id.clone()))?;
        if obligation.state == ObligationState::SupportedUnderProfile {
            return Err(FrontierAdapterError::InvalidFeatureCounts);
        }
        if obligation.state_identity.is_empty() {
            return Err(FrontierAdapterError::EmptyObligationStateIdentity(
                claim_id.clone(),
            ));
        }

        let mut alternative_ids = BTreeSet::new();
        for alternative in &claim.alternatives {
            if alternative.id.is_empty() {
                return Err(FrontierAdapterError::EmptyAlternativeId {
                    claim_id: claim_id.clone(),
                });
            }
            if !alternative_ids.insert(alternative.id.as_str()) {
                return Err(FrontierAdapterError::DuplicateAlternative {
                    claim_id: claim_id.clone(),
                    alternative_id: alternative.id.clone(),
                });
            }

            let mut requirement_keys = BTreeSet::new();
            let mut alternative_is_locally_admissible = true;
            let mut direct_independence_groups = BTreeSet::<String>::new();

            for requirement in &alternative.requirements {
                if !requirement_keys.insert(requirement_key(requirement)) {
                    return Err(FrontierAdapterError::DuplicateRequirement {
                        claim_id: claim_id.clone(),
                        alternative_id: alternative.id.clone(),
                    });
                }

                match requirement {
                    ProofRequirement::Evidence(evidence_id) => {
                        let evidence = evidence_by_id
                            .get(evidence_id.as_str())
                            .copied()
                            .ok_or_else(|| {
                                FrontierAdapterError::MissingEvidence(evidence_id.clone())
                            })?;
                        if evidence.subject_claim_id != claim_id {
                            return Err(FrontierAdapterError::EvidenceSubjectMismatch {
                                claim_id: claim_id.clone(),
                                evidence_id: evidence_id.clone(),
                            });
                        }
                        if evidence.id.is_empty()
                            || evidence.source_id.is_empty()
                            || evidence.source_family_id.is_empty()
                            || evidence.independence_group_id.is_empty()
                            || evidence.scope_id.is_empty()
                            || evidence.authority_id.is_empty()
                        {
                            return Err(FrontierAdapterError::EmptyEvidenceIdentity(
                                evidence_id.clone(),
                            ));
                        }
                        referenced_evidence.insert(evidence_id.clone());
                        if evidence_is_admissible(evidence, &case.query, &claim_id) {
                            direct_independence_groups
                                .insert(evidence.independence_group_id.clone());
                            available_evidence_families.insert(evidence.source_family_id.clone());
                        } else {
                            alternative_is_locally_admissible = false;
                            increment(&mut missing_evidence_requirements)?;
                        }
                    }
                    ProofRequirement::Claim {
                        claim_id: dependency_id,
                        ..
                    } => {
                        if !claims_by_id.contains_key(dependency_id.as_str()) {
                            return Err(FrontierAdapterError::MissingDependencyClaim {
                                parent: claim_id.clone(),
                                dependency: dependency_id.clone(),
                            });
                        }
                        let dependency_obligation = obligations_by_id
                            .get(dependency_id.as_str())
                            .copied()
                            .ok_or_else(|| {
                                FrontierAdapterError::MissingObligation(dependency_id.clone())
                            })?;
                        observed_claims.insert(dependency_id.clone());
                        if dependency_obligation.state != ObligationState::SupportedUnderProfile {
                            increment(&mut dependency_edges)?;
                            adjacency
                                .entry(claim_id.clone())
                                .or_default()
                                .insert(dependency_id.clone());
                            pending.push(dependency_id.clone());
                        }
                    }
                }
            }

            if direct_independence_groups.len() < alternative.min_independent_groups {
                alternative_is_locally_admissible = false;
            }
            if alternative_is_locally_admissible {
                increment(&mut admissible_alternatives)?;
            }
        }
    }

    let maximum_dependency_depth = longest_dependency_depth(&active_claims, &adjacency)?;
    let operation_families = context
        .operation_family_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();

    if active_claims.is_empty()
        && (!operation_families.is_empty() || context.mandatory_verifier_operations != 0)
    {
        return Err(FrontierAdapterError::ExecutionContextForEmptyFrontier);
    }

    let features = ProofFrontierFeatures {
        unresolved_obligations: usize_to_u64(active_claims.len())?,
        mandatory_obligations: usize_to_u64(unresolved_targets.len())?,
        maximum_dependency_depth,
        dependency_edges,
        admissible_alternatives,
        missing_evidence_requirements,
        distinct_operation_families: usize_to_u64(operation_families.len())?,
        distinct_evidence_families: usize_to_u64(available_evidence_families.len())?,
        mandatory_verifier_operations: context.mandatory_verifier_operations,
    };
    features
        .validate()
        .map_err(|_| FrontierAdapterError::InvalidFeatureCounts)?;

    let canonical_bytes = canonical_snapshot_bytes(
        case,
        &target_ids,
        &observed_claims,
        &active_claims,
        &claims_by_id,
        &obligations_by_id,
        &referenced_evidence,
        &evidence_by_id,
        &operation_families,
        context.mandatory_verifier_operations,
    )?;
    let fingerprint = identity_fingerprint_v1(&canonical_bytes);

    Ok(FrontierSnapshot {
        features,
        fingerprint,
        canonical_bytes,
    })
}

fn evidence_is_admissible(
    evidence: &EvidenceRevision,
    query: &QueryProfile,
    claim_id: &str,
) -> bool {
    let temporally_valid = evidence.valid_from <= query.valid_at
        && evidence.valid_until.is_none_or(|end| query.valid_at < end)
        && evidence.known_from <= query.known_at
        && evidence.known_until.is_none_or(|end| query.known_at < end);
    evidence.subject_claim_id == claim_id
        && temporally_valid
        && evidence.scope_id == query.required_scope
        && evidence.authority_id == query.required_authority
        && evidence.integrity == SourceIntegrity::Verified
        && evidence.lifecycle == EvidenceLifecycle::Active
}

fn requirement_key(requirement: &ProofRequirement) -> (u8, &str, u8) {
    match requirement {
        ProofRequirement::Evidence(evidence_id) => (0, evidence_id, 0),
        ProofRequirement::Claim { claim_id, polarity } => (1, claim_id, polarity_code(*polarity)),
    }
}

fn longest_dependency_depth(
    active_claims: &BTreeSet<String>,
    adjacency: &BTreeMap<String, BTreeSet<String>>,
) -> Result<u64, FrontierAdapterError> {
    if active_claims.is_empty() {
        return Ok(0);
    }

    let mut indegree = active_claims
        .iter()
        .cloned()
        .map(|claim_id| (claim_id, 0_usize))
        .collect::<BTreeMap<_, _>>();
    for dependencies in adjacency.values() {
        for dependency_id in dependencies {
            let degree = indegree
                .get_mut(dependency_id)
                .ok_or(FrontierAdapterError::InvalidFeatureCounts)?;
            *degree = degree
                .checked_add(1)
                .ok_or(FrontierAdapterError::CounterOverflow)?;
        }
    }

    let mut ready = indegree
        .iter()
        .filter_map(|(claim_id, degree)| (*degree == 0).then_some(claim_id.clone()))
        .collect::<BTreeSet<_>>();
    let mut depths = active_claims
        .iter()
        .cloned()
        .map(|claim_id| (claim_id, 0_u64))
        .collect::<BTreeMap<_, _>>();
    let mut processed = 0_usize;
    let mut maximum_depth = 0_u64;

    while let Some(claim_id) = ready.pop_first() {
        processed = processed
            .checked_add(1)
            .ok_or(FrontierAdapterError::CounterOverflow)?;
        let current_depth = *depths
            .get(&claim_id)
            .ok_or(FrontierAdapterError::InvalidFeatureCounts)?;
        maximum_depth = maximum_depth.max(current_depth);

        if let Some(dependencies) = adjacency.get(&claim_id) {
            for dependency_id in dependencies {
                let next_depth = current_depth
                    .checked_add(1)
                    .ok_or(FrontierAdapterError::CounterOverflow)?;
                let depth = depths
                    .get_mut(dependency_id)
                    .ok_or(FrontierAdapterError::InvalidFeatureCounts)?;
                *depth = (*depth).max(next_depth);

                let degree = indegree
                    .get_mut(dependency_id)
                    .ok_or(FrontierAdapterError::InvalidFeatureCounts)?;
                *degree = degree
                    .checked_sub(1)
                    .ok_or(FrontierAdapterError::InvalidFeatureCounts)?;
                if *degree == 0 {
                    ready.insert(dependency_id.clone());
                }
            }
        }
    }

    if processed != active_claims.len() {
        return Err(FrontierAdapterError::DependencyCycle);
    }
    Ok(maximum_depth)
}

#[allow(clippy::too_many_arguments)]
fn canonical_snapshot_bytes(
    case: &PlannerCase,
    target_ids: &[&str],
    observed_claims: &BTreeSet<String>,
    active_claims: &BTreeSet<String>,
    claims_by_id: &BTreeMap<&str, &ClaimNode>,
    obligations_by_id: &BTreeMap<&str, &Obligation>,
    referenced_evidence: &BTreeSet<String>,
    evidence_by_id: &BTreeMap<&str, &EvidenceRevision>,
    operation_families: &BTreeSet<u32>,
    mandatory_verifier_operations: u64,
) -> Result<Vec<u8>, FrontierAdapterError> {
    let mut bytes = b"POD_META_FRONTIER_SNAPSHOT_V1\0".to_vec();
    push_string(&mut bytes, &case.id)?;
    push_u64(&mut bytes, case.query.valid_at);
    push_u64(&mut bytes, case.query.known_at);
    push_string(&mut bytes, &case.query.required_scope)?;
    push_string(&mut bytes, &case.query.required_authority)?;

    push_len(&mut bytes, target_ids.len())?;
    for target_id in target_ids {
        push_string(&mut bytes, target_id)?;
    }

    push_len(&mut bytes, operation_families.len())?;
    for family_id in operation_families {
        push_u64(&mut bytes, u64::from(*family_id));
    }
    push_u64(&mut bytes, mandatory_verifier_operations);

    push_len(&mut bytes, observed_claims.len())?;
    for claim_id in observed_claims {
        push_string(&mut bytes, claim_id)?;
        let obligation = obligations_by_id
            .get(claim_id.as_str())
            .copied()
            .ok_or_else(|| FrontierAdapterError::MissingObligation(claim_id.clone()))?;
        push_u8(&mut bytes, obligation_state_code(obligation.state));
        push_string(&mut bytes, &obligation.state_identity)?;

        let Some(claim) = claims_by_id.get(claim_id.as_str()).copied() else {
            return Err(FrontierAdapterError::MissingTargetClaim(claim_id.clone()));
        };
        let alternatives = if active_claims.contains(claim_id) {
            let mut alternatives = claim.alternatives.iter().collect::<Vec<_>>();
            alternatives.sort_by(|left, right| left.id.cmp(&right.id));
            alternatives
        } else {
            Vec::new()
        };
        push_len(&mut bytes, alternatives.len())?;
        for alternative in alternatives {
            encode_alternative(&mut bytes, alternative)?;
        }
    }

    push_len(&mut bytes, referenced_evidence.len())?;
    for evidence_id in referenced_evidence {
        let evidence = evidence_by_id
            .get(evidence_id.as_str())
            .copied()
            .ok_or_else(|| FrontierAdapterError::MissingEvidence(evidence_id.clone()))?;
        encode_evidence(&mut bytes, evidence)?;
    }

    Ok(bytes)
}

fn encode_alternative(
    bytes: &mut Vec<u8>,
    alternative: &ProofAlternative,
) -> Result<(), FrontierAdapterError> {
    push_string(bytes, &alternative.id)?;
    push_u8(bytes, polarity_code(alternative.conclusion));
    push_len(bytes, alternative.min_independent_groups)?;
    let mut requirements = alternative.requirements.iter().collect::<Vec<_>>();
    requirements.sort_by_key(|requirement| requirement_key(requirement));
    push_len(bytes, requirements.len())?;
    for requirement in requirements {
        match requirement {
            ProofRequirement::Evidence(evidence_id) => {
                push_u8(bytes, 1);
                push_string(bytes, evidence_id)?;
            }
            ProofRequirement::Claim { claim_id, polarity } => {
                push_u8(bytes, 2);
                push_string(bytes, claim_id)?;
                push_u8(bytes, polarity_code(*polarity));
            }
        }
    }
    Ok(())
}

fn encode_evidence(
    bytes: &mut Vec<u8>,
    evidence: &EvidenceRevision,
) -> Result<(), FrontierAdapterError> {
    push_string(bytes, &evidence.id)?;
    push_string(bytes, &evidence.source_id)?;
    push_string(bytes, &evidence.source_family_id)?;
    push_string(bytes, &evidence.independence_group_id)?;
    push_string(bytes, &evidence.subject_claim_id)?;
    push_u8(bytes, polarity_code(evidence.asserted_polarity));
    push_u64(bytes, evidence.valid_from);
    push_option_u64(bytes, evidence.valid_until);
    push_u64(bytes, evidence.known_from);
    push_option_u64(bytes, evidence.known_until);
    push_string(bytes, &evidence.scope_id)?;
    push_string(bytes, &evidence.authority_id)?;
    push_u8(
        bytes,
        match evidence.integrity {
            SourceIntegrity::Verified => 1,
            SourceIntegrity::Unverified => 2,
            SourceIntegrity::Invalid => 3,
        },
    );
    push_u8(
        bytes,
        match evidence.lifecycle {
            EvidenceLifecycle::Active => 1,
            EvidenceLifecycle::Retracted => 2,
        },
    );
    Ok(())
}

fn push_string(bytes: &mut Vec<u8>, value: &str) -> Result<(), FrontierAdapterError> {
    push_len(bytes, value.len())?;
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_len(bytes: &mut Vec<u8>, value: usize) -> Result<(), FrontierAdapterError> {
    push_u64(bytes, usize_to_u64(value)?);
    Ok(())
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_option_u64(bytes: &mut Vec<u8>, value: Option<u64>) {
    match value {
        Some(value) => {
            push_u8(bytes, 1);
            push_u64(bytes, value);
        }
        None => push_u8(bytes, 0),
    }
}

fn push_u8(bytes: &mut Vec<u8>, value: u8) {
    bytes.push(value);
}

fn usize_to_u64(value: usize) -> Result<u64, FrontierAdapterError> {
    u64::try_from(value).map_err(|_| FrontierAdapterError::CounterOverflow)
}

fn increment(value: &mut u64) -> Result<(), FrontierAdapterError> {
    *value = value
        .checked_add(1)
        .ok_or(FrontierAdapterError::CounterOverflow)?;
    Ok(())
}

fn polarity_code(polarity: Polarity) -> u8 {
    match polarity {
        Polarity::Supports => 1,
        Polarity::Refutes => 2,
    }
}

fn obligation_state_code(state: ObligationState) -> u8 {
    match state {
        ObligationState::Open => 1,
        ObligationState::CandidateEvidenceAvailable => 2,
        ObligationState::SupportedUnderProfile => 3,
        ObligationState::Contested => 4,
        ObligationState::Contradicted => 5,
        ObligationState::StaleOrRetracted => 6,
        ObligationState::OutOfScope => 7,
        ObligationState::Unauthorized => 8,
        ObligationState::VerifierRejected => 9,
        ObligationState::UnresolvableWithoutExternalInput => 10,
        ObligationState::BudgetInsufficient => 11,
    }
}

/// Four seeded FNV-1a lanes provide a stable 256-bit identity checksum. This
/// deliberately has no cryptographic or authentication claim.
fn identity_fingerprint_v1(bytes: &[u8]) -> [u8; 32] {
    let mut output = [0_u8; 32];
    for (lane, chunk) in output.chunks_exact_mut(8).enumerate() {
        let lane_seed = (lane as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ lane_seed;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        chunk.copy_from_slice(&hash.to_le_bytes());
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query() -> QueryProfile {
        QueryProfile {
            valid_at: 10,
            known_at: 10,
            required_scope: "scope-a".into(),
            required_authority: "authority-a".into(),
        }
    }

    fn evidence(id: &str, claim_id: &str, family: &str, group: &str) -> EvidenceRevision {
        EvidenceRevision {
            id: id.into(),
            source_id: format!("source-{id}"),
            source_family_id: family.into(),
            independence_group_id: group.into(),
            subject_claim_id: claim_id.into(),
            asserted_polarity: Polarity::Supports,
            valid_from: 1,
            valid_until: None,
            known_from: 1,
            known_until: None,
            scope_id: "scope-a".into(),
            authority_id: "authority-a".into(),
            integrity: SourceIntegrity::Verified,
            lifecycle: EvidenceLifecycle::Active,
        }
    }

    fn evidence_alternative(id: &str, evidence_id: &str) -> ProofAlternative {
        ProofAlternative {
            id: id.into(),
            conclusion: Polarity::Supports,
            requirements: vec![ProofRequirement::Evidence(evidence_id.into())],
            min_independent_groups: 1,
        }
    }

    fn obligation(id: &str, state: ObligationState) -> Obligation {
        Obligation::new(id, state, 5).with_state_identity(format!("state-{id}"))
    }

    fn fixture() -> (PlannerCase, Vec<Obligation>, FrontierAdapterContext) {
        let root = ClaimNode {
            id: "root".into(),
            alternatives: vec![ProofAlternative {
                id: "root-path".into(),
                conclusion: Polarity::Supports,
                requirements: vec![
                    ProofRequirement::Evidence("e-root".into()),
                    ProofRequirement::Claim {
                        claim_id: "child".into(),
                        polarity: Polarity::Supports,
                    },
                ],
                min_independent_groups: 1,
            }],
        };
        let child = ClaimNode {
            id: "child".into(),
            alternatives: vec![evidence_alternative("child-path", "e-child")],
        };
        let case = PlannerCase {
            id: "fixture-case".into(),
            query: query(),
            target_claim_ids: vec!["root".into()],
            claims: vec![root, child],
            evidence: vec![
                evidence("e-root", "root", "family-a", "group-a"),
                evidence("e-child", "child", "family-b", "group-b"),
            ],
        };
        (
            case,
            vec![
                obligation("root", ObligationState::Open),
                obligation("child", ObligationState::CandidateEvidenceAvailable),
            ],
            FrontierAdapterContext {
                operation_family_ids: vec![3, 3, 5],
                mandatory_verifier_operations: 2,
            },
        )
    }

    #[test]
    fn extracts_hand_computed_frontier_features_from_the_typed_graph() {
        let (case, obligations, context) = fixture();
        let snapshot = extract_frontier_snapshot(&case, &obligations, &context).unwrap();
        assert_eq!(
            snapshot.features,
            ProofFrontierFeatures {
                unresolved_obligations: 2,
                mandatory_obligations: 1,
                maximum_dependency_depth: 1,
                dependency_edges: 1,
                admissible_alternatives: 2,
                missing_evidence_requirements: 0,
                distinct_operation_families: 2,
                distinct_evidence_families: 2,
                mandatory_verifier_operations: 2,
            }
        );
        assert!(!snapshot.canonical_bytes.is_empty());
        assert_ne!(snapshot.fingerprint, [0; 32]);
    }

    #[test]
    fn prediction_uses_the_snapshot_features_and_its_bound_identity() {
        let (case, obligations, context) = fixture();
        let snapshot = extract_frontier_snapshot(&case, &obligations, &context).unwrap();
        let point = crate::ResourceVector::new(10, 10, 10, 10, 10, 10);
        let scale = crate::ResourceVector::new(1, 1, 1, 1, 1, 1);
        let model = LinearProofFrontierCostModel {
            model_digest: [71; 32],
            point_cost: crate::FrontierCostCoefficients {
                intercept: point,
                ..Default::default()
            },
            uncertainty_scale: crate::FrontierCostCoefficients {
                intercept: scale,
                ..Default::default()
            },
        };
        let prediction = snapshot
            .predict_cost_to_go(model, [72; 32], ShiftStatus::InDomain)
            .unwrap();
        assert_eq!(prediction.features, snapshot.features);
        assert_eq!(prediction.frontier_digest, snapshot.fingerprint);
        assert_eq!(prediction.model_digest, [71; 32]);
        assert_eq!(prediction.point_cost, point);
        assert_eq!(prediction.uncertainty_scale, scale);
    }

    #[test]
    fn canonical_identity_is_invariant_to_input_permutation() {
        let (case, obligations, context) = fixture();
        let expected = extract_frontier_snapshot(&case, &obligations, &context).unwrap();

        let mut permuted_case = case;
        permuted_case.claims.reverse();
        permuted_case.evidence.reverse();
        for claim in &mut permuted_case.claims {
            claim.alternatives.reverse();
            for alternative in &mut claim.alternatives {
                alternative.requirements.reverse();
            }
        }
        let mut permuted_obligations = obligations;
        permuted_obligations.reverse();
        let mut permuted_context = context;
        permuted_context.operation_family_ids.reverse();

        let actual =
            extract_frontier_snapshot(&permuted_case, &permuted_obligations, &permuted_context)
                .unwrap();
        assert_eq!(actual.features, expected.features);
        assert_eq!(actual.canonical_bytes, expected.canonical_bytes);
        assert_eq!(actual.fingerprint, expected.fingerprint);
    }

    #[test]
    fn unreachable_nodes_and_unreferenced_evidence_do_not_change_frontier_identity() {
        let (mut case, obligations, context) = fixture();
        let expected = extract_frontier_snapshot(&case, &obligations, &context).unwrap();
        case.claims.push(ClaimNode {
            id: "unreachable".into(),
            alternatives: vec![evidence_alternative("unused-path", "unused-evidence")],
        });
        case.evidence.push(evidence(
            "unused-evidence",
            "unreachable",
            "unused-family",
            "unused-group",
        ));
        let mut obligations_with_unreachable = obligations;
        obligations_with_unreachable.push(obligation("unreachable", ObligationState::Open));

        let actual =
            extract_frontier_snapshot(&case, &obligations_with_unreachable, &context).unwrap();
        assert_eq!(actual.features, expected.features);
        assert_eq!(actual.canonical_bytes, expected.canonical_bytes);
        assert_eq!(actual.fingerprint, expected.fingerprint);
    }

    #[test]
    fn a_new_reachable_dependency_increases_depth_and_edge_count() {
        let (mut case, mut obligations, context) = fixture();
        let before = extract_frontier_snapshot(&case, &obligations, &context).unwrap();
        case.claims.push(ClaimNode {
            id: "leaf".into(),
            alternatives: vec![ProofAlternative {
                id: "leaf-path".into(),
                conclusion: Polarity::Supports,
                requirements: Vec::new(),
                min_independent_groups: 0,
            }],
        });
        let child = case
            .claims
            .iter_mut()
            .find(|claim| claim.id == "child")
            .unwrap();
        child.alternatives[0]
            .requirements
            .push(ProofRequirement::Claim {
                claim_id: "leaf".into(),
                polarity: Polarity::Supports,
            });
        obligations.push(obligation("leaf", ObligationState::Open));

        let after = extract_frontier_snapshot(&case, &obligations, &context).unwrap();
        assert_eq!(before.features.maximum_dependency_depth, 1);
        assert_eq!(after.features.maximum_dependency_depth, 2);
        assert_eq!(
            after.features.dependency_edges,
            before.features.dependency_edges + 1
        );
    }

    #[test]
    fn a_new_locally_admissible_or_alternative_is_counted() {
        let (mut case, obligations, context) = fixture();
        let before = extract_frontier_snapshot(&case, &obligations, &context).unwrap();
        case.evidence
            .push(evidence("e-root-2", "root", "family-c", "group-c"));
        let root = case
            .claims
            .iter_mut()
            .find(|claim| claim.id == "root")
            .unwrap();
        root.alternatives
            .push(evidence_alternative("root-alt-2", "e-root-2"));

        let after = extract_frontier_snapshot(&case, &obligations, &context).unwrap();
        assert_eq!(
            after.features.admissible_alternatives,
            before.features.admissible_alternatives + 1
        );
        assert_eq!(after.features.distinct_evidence_families, 3);
    }

    #[test]
    fn revoking_evidence_changes_availability_features_and_identity() {
        let (mut case, obligations, context) = fixture();
        let before = extract_frontier_snapshot(&case, &obligations, &context).unwrap();
        case.evidence[0].lifecycle = EvidenceLifecycle::Retracted;

        let after = extract_frontier_snapshot(&case, &obligations, &context).unwrap();
        assert_eq!(after.features.missing_evidence_requirements, 1);
        assert_eq!(after.features.distinct_evidence_families, 1);
        assert_ne!(after.fingerprint, before.fingerprint);
    }

    #[test]
    fn a_supported_target_has_an_empty_cost_to_go_frontier() {
        let (case, _, _) = fixture();
        let obligations = vec![obligation("root", ObligationState::SupportedUnderProfile)];
        let snapshot =
            extract_frontier_snapshot(&case, &obligations, &FrontierAdapterContext::default())
                .unwrap();
        assert_eq!(snapshot.features, ProofFrontierFeatures::default());
    }

    #[test]
    fn cycles_and_missing_references_fail_closed() {
        let (mut cycle, obligations, context) = fixture();
        let child = cycle
            .claims
            .iter_mut()
            .find(|claim| claim.id == "child")
            .unwrap();
        child.alternatives[0]
            .requirements
            .push(ProofRequirement::Claim {
                claim_id: "root".into(),
                polarity: Polarity::Supports,
            });
        assert_eq!(
            extract_frontier_snapshot(&cycle, &obligations, &context),
            Err(FrontierAdapterError::DependencyCycle)
        );

        let (mut missing_evidence, obligations, context) = fixture();
        missing_evidence.claims[0].alternatives[0].requirements[0] =
            ProofRequirement::Evidence("not-present".into());
        assert_eq!(
            extract_frontier_snapshot(&missing_evidence, &obligations, &context),
            Err(FrontierAdapterError::MissingEvidence("not-present".into()))
        );

        let (mut missing_dependency, obligations, context) = fixture();
        missing_dependency.claims[0].alternatives[0].requirements[1] = ProofRequirement::Claim {
            claim_id: "not-present".into(),
            polarity: Polarity::Supports,
        };
        assert!(matches!(
            extract_frontier_snapshot(&missing_dependency, &obligations, &context),
            Err(FrontierAdapterError::MissingDependencyClaim { .. })
        ));

        let (case, obligations, context) = fixture();
        let without_child = obligations
            .into_iter()
            .filter(|obligation| obligation.id != "child")
            .collect::<Vec<_>>();
        assert_eq!(
            extract_frontier_snapshot(&case, &without_child, &context),
            Err(FrontierAdapterError::MissingObligation("child".into()))
        );
    }

    #[test]
    fn duplicate_ids_and_duplicate_requirements_are_rejected() {
        let (mut case, obligations, context) = fixture();
        case.claims.push(case.claims[0].clone());
        assert!(matches!(
            extract_frontier_snapshot(&case, &obligations, &context),
            Err(FrontierAdapterError::DuplicateClaim(_))
        ));

        let (mut case, obligations, context) = fixture();
        let duplicate_requirement = case.claims[0].alternatives[0].requirements[0].clone();
        case.claims[0].alternatives[0]
            .requirements
            .push(duplicate_requirement);
        assert!(matches!(
            extract_frontier_snapshot(&case, &obligations, &context),
            Err(FrontierAdapterError::DuplicateRequirement { .. })
        ));

        let (mut case, obligations, context) = fixture();
        case.target_claim_ids.push("root".into());
        assert_eq!(
            extract_frontier_snapshot(&case, &obligations, &context),
            Err(FrontierAdapterError::DuplicateTarget("root".into()))
        );
    }

    #[test]
    fn long_dependency_chains_are_processed_iteratively() {
        const NODES: usize = 10_000;
        let claims = (0..NODES)
            .map(|index| ClaimNode {
                id: format!("claim-{index:05}"),
                alternatives: vec![ProofAlternative {
                    id: format!("path-{index:05}"),
                    conclusion: Polarity::Supports,
                    requirements: if index + 1 < NODES {
                        vec![ProofRequirement::Claim {
                            claim_id: format!("claim-{:05}", index + 1),
                            polarity: Polarity::Supports,
                        }]
                    } else {
                        Vec::new()
                    },
                    min_independent_groups: 0,
                }],
            })
            .collect::<Vec<_>>();
        let obligations = (0..NODES)
            .map(|index| obligation(&format!("claim-{index:05}"), ObligationState::Open))
            .collect::<Vec<_>>();
        let case = PlannerCase {
            id: "long-chain".into(),
            query: query(),
            target_claim_ids: vec!["claim-00000".into()],
            claims,
            evidence: Vec::new(),
        };
        let snapshot =
            extract_frontier_snapshot(&case, &obligations, &FrontierAdapterContext::default())
                .unwrap();
        assert_eq!(snapshot.features.unresolved_obligations, NODES as u64);
        assert_eq!(snapshot.features.dependency_edges, (NODES - 1) as u64);
        assert_eq!(
            snapshot.features.maximum_dependency_depth,
            (NODES - 1) as u64
        );
    }
}
