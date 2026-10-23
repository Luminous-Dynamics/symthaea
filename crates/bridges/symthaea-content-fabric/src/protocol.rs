use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::model::{
    ActionRefV1, ExternalPlannerProfileV1, ExternalPlannerRequestIdV1, ExternalPlannerRequestV1,
    FailureDomainKindV1, PlannerProfileIdV1, SoftEvidenceStateV1, SoftMetricKindV1,
    EXTERNAL_PLANNER_PROTOCOL_V1, PENALTY_PPM_MAX,
};

const REQUEST_MAGIC: &[u8] = b"MYCELIX-EXTERNAL-PLANNER-REQUEST\0";
const PROFILE_MAGIC: &[u8] = b"MYCELIX-CONTENT-PLANNER-PROFILE\0";
const PLANNER_SCHEMA_V1: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProtocolErrorV1 {
    #[error("unsupported external planner protocol schema")]
    UnsupportedSchemaVersion,
    #[error("external planner request id mismatch")]
    RequestIdMismatch,
    #[error("request contains no candidates")]
    EmptyCandidateSet,
    #[error("minimum replica count is invalid")]
    InvalidMinimumReplicas,
    #[error("normalization profile contains a zero ceiling")]
    InvalidNormalizationProfile,
    #[error("normalization profile id does not match its fields")]
    ProfileIdMismatch,
    #[error("target latency must be non-zero when present")]
    InvalidLatencyTarget,
    #[error("malformed action or agent reference")]
    MalformedOpaqueReference,
    #[error("duplicate candidate action")]
    DuplicateCandidate,
    #[error("candidate baseline ranks are not canonical and contiguous")]
    InvalidBaselineRank,
    #[error("candidate penalty exceeds one million ppm")]
    InvalidPenalty,
    #[error("missing weighted metric list is non-canonical")]
    InvalidMissingMetricList,
    #[error("failure-domain requirements are invalid or non-canonical")]
    InvalidFailureDomainRequirements,
    #[error("candidate failure-domain facts do not match canonical required dimensions")]
    InvalidCandidateFailureDomains,
    #[error("baseline selection is unknown, duplicated, or locally policy-invalid")]
    InvalidBaselineSelection,
    #[error("JSON decode failed: {0}")]
    Json(String),
}

fn put_field(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn put_string(hasher: &mut blake3::Hasher, value: &str) {
    put_field(hasher, value.as_bytes());
}

fn put_option_u32(hasher: &mut blake3::Hasher, value: Option<u32>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_be_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

pub fn recompute_profile_id_v1(profile: &ExternalPlannerProfileV1) -> PlannerProfileIdV1 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PROFILE_MAGIC);
    hasher.update(&PLANNER_SCHEMA_V1.to_be_bytes());
    hasher.update(&profile.cost_ceiling_microunits_per_gib.to_be_bytes());
    hasher.update(&profile.latency_ceiling_ms.to_be_bytes());
    hasher.update(&profile.energy_ceiling_millijoules_per_gib.to_be_bytes());
    hasher.update(&profile.locality_ceiling_km.to_be_bytes());
    PlannerProfileIdV1(*hasher.finalize().as_bytes())
}

pub fn recompute_request_id_v1(request: &ExternalPlannerRequestV1) -> ExternalPlannerRequestIdV1 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(REQUEST_MAGIC);
    hasher.update(&EXTERNAL_PLANNER_PROTOCOL_V1.to_be_bytes());
    hasher.update(&request.planner_input_id.0);
    hasher.update(&request.storage_intent_id.0);
    hasher.update(&request.target.object_id.0);
    put_field(&mut hasher, request.target.digest.algorithm.tag().as_bytes());
    hasher.update(&request.target.digest.bytes);
    hasher.update(&request.target.size_bytes.to_be_bytes());
    hasher.update(&[u8::from(request.target.client_side_encrypted)]);
    hasher.update(&request.evaluated_at_unix_ms.to_be_bytes());

    hasher.update(&request.profile.id.0);
    hasher.update(&request.profile.cost_ceiling_microunits_per_gib.to_be_bytes());
    hasher.update(&request.profile.latency_ceiling_ms.to_be_bytes());
    hasher.update(&request.profile.energy_ceiling_millijoules_per_gib.to_be_bytes());
    hasher.update(&request.profile.locality_ceiling_km.to_be_bytes());

    put_option_u32(&mut hasher, request.preferences.target_latency_ms);
    hasher.update(&request.preferences.cost_weight.to_be_bytes());
    hasher.update(&request.preferences.latency_weight.to_be_bytes());
    hasher.update(&request.preferences.energy_weight.to_be_bytes());
    hasher.update(&request.preferences.locality_weight.to_be_bytes());

    hasher.update(&request.minimum_replicas.to_be_bytes());
    hasher.update(&(request.failure_domain_requirements.len() as u64).to_be_bytes());
    for requirement in &request.failure_domain_requirements {
        hasher.update(&[requirement.kind.tag()]);
        hasher.update(&requirement.minimum_distinct.to_be_bytes());
    }

    hasher.update(&request.baseline_proposal_id.0);
    hasher.update(&(request.baseline_selected_availability_actions.len() as u64).to_be_bytes());
    for action in &request.baseline_selected_availability_actions {
        hasher.update(&action.0);
    }

    hasher.update(&(request.candidates.len() as u64).to_be_bytes());
    for candidate in &request.candidates {
        hasher.update(&candidate.availability_action.0);
        hasher.update(&candidate.advertisement_action.0);
        hasher.update(&candidate.provider.0);
        hasher.update(&candidate.baseline_rank.to_be_bytes());
        hasher.update(&candidate.baseline_weighted_penalty.to_be_bytes());
        hasher.update(&candidate.cost_penalty_ppm.to_be_bytes());
        hasher.update(&candidate.latency_penalty_ppm.to_be_bytes());
        hasher.update(&candidate.energy_penalty_ppm.to_be_bytes());
        hasher.update(&candidate.locality_penalty_ppm.to_be_bytes());
        hasher.update(&[evidence_state_tag(candidate.evidence_state)]);
        hasher.update(&(candidate.missing_weighted_metrics.len() as u64).to_be_bytes());
        for metric in &candidate.missing_weighted_metrics {
            hasher.update(&[metric_kind_tag(*metric)]);
        }
        hasher.update(&(candidate.accepted_failure_domains.len() as u64).to_be_bytes());
        for domain in &candidate.accepted_failure_domains {
            hasher.update(&[domain.kind.tag()]);
            put_string(&mut hasher, &domain.value);
        }
    }

    ExternalPlannerRequestIdV1(*hasher.finalize().as_bytes())
}

fn evidence_state_tag(state: SoftEvidenceStateV1) -> u8 {
    state.tag()
}

fn metric_kind_tag(kind: SoftMetricKindV1) -> u8 {
    kind.tag()
}

fn canonical_metric_list(metrics: &[SoftMetricKindV1]) -> bool {
    metrics.windows(2).all(|pair| pair[0] < pair[1])
}

fn canonical_domain_value(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 64
        && bytes.iter().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(*byte, b'-' | b'_' | b'.' | b':')
        })
}

fn canonical_domain_list(
    domains: &[crate::model::ExternalAcceptedFailureDomainV1],
) -> bool {
    domains.windows(2).all(|pair| pair[0].kind < pair[1].kind)
        && domains.iter().all(|domain| canonical_domain_value(&domain.value))
}

fn candidate_domains(
    request: &ExternalPlannerRequestV1,
    action: &ActionRefV1,
) -> Option<BTreeMap<FailureDomainKindV1, &str>> {
    let candidate = request
        .candidates
        .iter()
        .find(|candidate| &candidate.availability_action == action)?;
    Some(
        candidate
            .accepted_failure_domains
            .iter()
            .map(|domain| (domain.kind, domain.value.as_str()))
            .collect(),
    )
}

pub(crate) fn selection_satisfies_local_policy(
    request: &ExternalPlannerRequestV1,
    selected: &[ActionRefV1],
) -> bool {
    if selected.len() < usize::from(request.minimum_replicas) {
        return false;
    }
    let mut unique = BTreeSet::new();
    for action in selected {
        if !unique.insert(action) || candidate_domains(request, action).is_none() {
            return false;
        }
    }
    for requirement in &request.failure_domain_requirements {
        let distinct = selected
            .iter()
            .filter_map(|action| candidate_domains(request, action))
            .filter_map(|domains| domains.get(&requirement.kind).copied())
            .collect::<BTreeSet<_>>();
        if distinct.len() < usize::from(requirement.minimum_distinct) {
            return false;
        }
    }
    true
}

pub fn validate_request_v1(request: &ExternalPlannerRequestV1) -> Result<(), ProtocolErrorV1> {
    if request.schema_version != EXTERNAL_PLANNER_PROTOCOL_V1 {
        return Err(ProtocolErrorV1::UnsupportedSchemaVersion);
    }
    if request.candidates.is_empty() {
        return Err(ProtocolErrorV1::EmptyCandidateSet);
    }
    if request.minimum_replicas == 0
        || usize::from(request.minimum_replicas) > request.candidates.len()
    {
        return Err(ProtocolErrorV1::InvalidMinimumReplicas);
    }
    if request.profile.cost_ceiling_microunits_per_gib == 0
        || request.profile.latency_ceiling_ms == 0
        || request.profile.energy_ceiling_millijoules_per_gib == 0
        || request.profile.locality_ceiling_km == 0
    {
        return Err(ProtocolErrorV1::InvalidNormalizationProfile);
    }
    if request.profile.id != recompute_profile_id_v1(&request.profile) {
        return Err(ProtocolErrorV1::ProfileIdMismatch);
    }
    if request.preferences.target_latency_ms == Some(0) {
        return Err(ProtocolErrorV1::InvalidLatencyTarget);
    }

    let mut requirement_kinds = BTreeSet::new();
    let mut previous_requirement = None;
    for requirement in &request.failure_domain_requirements {
        if requirement.minimum_distinct == 0
            || requirement.minimum_distinct > request.minimum_replicas
            || !requirement_kinds.insert(requirement.kind)
            || previous_requirement.is_some_and(|previous| previous >= requirement.kind)
        {
            return Err(ProtocolErrorV1::InvalidFailureDomainRequirements);
        }
        previous_requirement = Some(requirement.kind);
    }

    let required_kinds = request
        .failure_domain_requirements
        .iter()
        .map(|requirement| requirement.kind)
        .collect::<BTreeSet<_>>();
    let mut candidate_actions = BTreeSet::new();
    for (index, candidate) in request.candidates.iter().enumerate() {
        if !candidate.availability_action.is_well_formed()
            || !candidate.advertisement_action.is_well_formed()
            || !candidate.provider.is_well_formed()
        {
            return Err(ProtocolErrorV1::MalformedOpaqueReference);
        }
        if !candidate_actions.insert(candidate.availability_action.clone()) {
            return Err(ProtocolErrorV1::DuplicateCandidate);
        }
        if candidate.baseline_rank != index as u64 {
            return Err(ProtocolErrorV1::InvalidBaselineRank);
        }
        if [
            candidate.cost_penalty_ppm,
            candidate.latency_penalty_ppm,
            candidate.energy_penalty_ppm,
            candidate.locality_penalty_ppm,
        ]
        .into_iter()
        .any(|penalty| penalty > PENALTY_PPM_MAX)
        {
            return Err(ProtocolErrorV1::InvalidPenalty);
        }
        if !canonical_metric_list(&candidate.missing_weighted_metrics) {
            return Err(ProtocolErrorV1::InvalidMissingMetricList);
        }
        if !canonical_domain_list(&candidate.accepted_failure_domains) {
            return Err(ProtocolErrorV1::InvalidCandidateFailureDomains);
        }
        let domains = candidate
            .accepted_failure_domains
            .iter()
            .map(|domain| (domain.kind, domain.value.as_str()))
            .collect::<BTreeMap<_, _>>();
        if domains.len() != candidate.accepted_failure_domains.len()
            || domains.keys().copied().collect::<BTreeSet<_>>() != required_kinds
        {
            return Err(ProtocolErrorV1::InvalidCandidateFailureDomains);
        }
    }

    if request
        .baseline_selected_availability_actions
        .iter()
        .any(|action| !action.is_well_formed())
        || !selection_satisfies_local_policy(request, &request.baseline_selected_availability_actions)
    {
        return Err(ProtocolErrorV1::InvalidBaselineSelection);
    }

    if request.id != recompute_request_id_v1(request) {
        return Err(ProtocolErrorV1::RequestIdMismatch);
    }
    Ok(())
}

pub fn decode_request_json_v1(json: &str) -> Result<ExternalPlannerRequestV1, ProtocolErrorV1> {
    let request = serde_json::from_str::<ExternalPlannerRequestV1>(json)
        .map_err(|error| ProtocolErrorV1::Json(error.to_string()))?;
    validate_request_v1(&request)?;
    Ok(request)
}
