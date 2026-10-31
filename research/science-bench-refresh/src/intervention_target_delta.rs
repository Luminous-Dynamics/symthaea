//! Independently blinded verification of the declared target delta in evaluator interventions.
//!
//! This module composes with intervention_quality v1. It distinguishes a global
//! meaning-preserving/meaning-changing judgment from evidence that the *declared*
//! target claim, direction and affected scope changed as intended. Reviewers report
//! observed changes without seeing the expected direction or allowed-scope list.
//!
//! Hashes bind declarations, not reviewer independence, semantic truth or raw custody.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use crate::intervention_quality::{
    analyze_intervention_quality, intervention_quality_plan_binding,
};

pub const TARGET_DELTA_PLAN_SCHEMA: &str =
    "symthaea.science_bench_refresh_intervention_target_delta_plan.v1";
pub const TARGET_DELTA_RECEIPT_SCHEMA: &str =
    "symthaea.science_bench_refresh_intervention_target_delta_receipt.v1";
pub const TARGET_DELTA_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_intervention_target_delta_report.v1";

const PLAN_DOMAIN: &[u8] = b"symthaea.intervention_target_delta.plan.v1\0";
const REPORT_DOMAIN: &[u8] = b"symthaea.intervention_target_delta.report.v1\0";
const MAX_CANDIDATES: usize = 100_000;
const MAX_PANEL: u32 = 7;
const MAX_SCOPE_IDS: usize = 10_000;
const MAX_LABEL: usize = 128;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetDeltaPurposeV1 {
    ExampleOnly,
    QualifiedRun,
}

impl TargetDeltaPurposeV1 {
    fn qualified(self) -> bool {
        self == Self::QualifiedRun
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum TargetChangeKindV1 {
    Unchanged,
    SupportIncreased,
    SupportDecreased,
    ContradictionIntroduced,
    ScopeChanged,
    TemporalValidityChanged,
    UnitChanged,
    IdentityChanged,
    SourceWithdrawn,
    SourceSuperseded,
    OtherDeclaredChange,
}

impl TargetChangeKindV1 {
    fn changed(self) -> bool {
        self != Self::Unchanged
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CollateralChangeV1 {
    NoneObserved,
    MaterialChangeOutsideAllowedScope,
    Ambiguous,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum QualityDispositionV1 {
    Accepted,
    Rejected,
    Abstained,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum QualityRelationV1 {
    SemanticInvariant,
    SemanticChange,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum FinalCandidateValidityV1 {
    Valid,
    Invalid,
    Ambiguous,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TargetDeltaCandidateV1 {
    pub candidate_id: String,
    /// Identity of the target claim/entity/scope being evaluated.
    pub target_identity_sha256: String,
    /// Exact parent baseline/variant artifact digests shown to reviewers.
    pub baseline_artifact_sha256: String,
    pub variant_artifact_sha256: String,
    pub baseline_target_state_sha256: String,
    pub variant_target_state_sha256: String,
    pub expected_target_change: TargetChangeKindV1,
    /// Exact canonical set of claim/scope identities allowed to change.
    pub expected_changed_claim_ids_sha256: Vec<String>,
    /// Exact canonical set of preserved control claims/scopes.
    pub preserved_claim_ids_sha256: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TargetDeltaPlanFieldsV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: TargetDeltaPurposeV1,
    pub study_id: String,
    pub quality_plan_sha256: String,
    pub quality_plan_commitment_sha256: String,
    pub reviewers_per_candidate: u32,
    pub consensus_threshold_permille: u16,
    pub candidates: Vec<TargetDeltaCandidateV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TargetDeltaPlanV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: TargetDeltaPurposeV1,
    pub study_id: String,
    pub quality_plan_sha256: String,
    pub quality_plan_commitment_sha256: String,
    pub reviewers_per_candidate: u32,
    pub consensus_threshold_permille: u16,
    pub candidates: Vec<TargetDeltaCandidateV1>,
    pub plan_commitment_sha256: String,
}

impl TargetDeltaPlanV1 {
    fn fields(&self) -> TargetDeltaPlanFieldsV1 {
        TargetDeltaPlanFieldsV1 {
            schema: self.schema.clone(),
            analyzer_source_revision: self.analyzer_source_revision.clone(),
            purpose: self.purpose,
            study_id: self.study_id.clone(),
            quality_plan_sha256: self.quality_plan_sha256.clone(),
            quality_plan_commitment_sha256: self.quality_plan_commitment_sha256.clone(),
            reviewers_per_candidate: self.reviewers_per_candidate,
            consensus_threshold_permille: self.consensus_threshold_permille,
            candidates: self.candidates.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TargetDeltaReviewerObservationV1 {
    pub reviewer_id_sha256: String,
    /// Echoes bind this adjudication to the exact planned target and parent artifacts.
    pub reviewed_target_identity_sha256: String,
    pub baseline_artifact_sha256: String,
    pub variant_artifact_sha256: String,
    /// Reviewers describe what they observed; they do not see the expected transition.
    pub observed_target_change: TargetChangeKindV1,
    /// Canonically sorted unique identity commitments observed to have changed.
    pub observed_changed_claim_ids_sha256: Vec<String>,
    pub collateral_change: CollateralChangeV1,
    pub blind_to_generator_identity: bool,
    pub blind_to_intended_relation: bool,
    pub blind_to_pipeline_disposition: bool,
    pub blind_to_evaluator_outcome: bool,
    pub blind_to_expected_target_change: bool,
    pub blind_to_allowed_changed_claims: bool,
    pub adjudication_artifact_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TargetDeltaCandidateAdjudicationV1 {
    pub candidate_id: String,
    pub reviewers: Vec<TargetDeltaReviewerObservationV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TargetDeltaReceiptV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: TargetDeltaPurposeV1,
    pub run_id: String,
    pub target_plan_sha256: String,
    pub target_plan_commitment_sha256: String,
    pub quality_plan_sha256: String,
    pub quality_receipt_sha256: String,
    pub adjudications: Vec<TargetDeltaCandidateAdjudicationV1>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TargetDeltaCandidateResultV1 {
    pub candidate_id: String,
    pub transformation_family: String,
    pub target_failure_mode: String,
    pub pipeline_disposition: QualityDispositionV1,
    pub intended_relation: QualityRelationV1,
    pub expected_target_change: TargetChangeKindV1,
    pub observed_target_change: Option<TargetChangeKindV1>,
    pub expected_changed_claim_count: u64,
    pub observed_changed_claim_count: u64,
    pub exact_changed_scope_match: bool,
    pub collateral_change: Option<CollateralChangeV1>,
    pub global_relation_validity: FinalCandidateValidityV1,
    pub target_delta_validity: FinalCandidateValidityV1,
    pub final_validity: FinalCandidateValidityV1,
    pub reviewer_consensus_ppm: Option<u32>,
    pub reviewer_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TargetDeltaFamilyMetricsV1 {
    pub transformation_family: String,
    pub candidate_count: u64,
    pub unique_semantic_cluster_count: u64,
    pub valid_count: u64,
    pub invalid_count: u64,
    pub ambiguous_count: u64,
    pub accepted_valid_count: u64,
    pub accepted_invalid_count: u64,
    pub accepted_ambiguous_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TargetDeltaReportFieldsV1 {
    pub schema: String,
    pub study_id: String,
    pub analyzer_source_revision: String,
    pub quality_plan_sha256: String,
    pub quality_receipt_sha256: String,
    pub target_plan_sha256: String,
    pub target_receipt_sha256: String,
    pub candidate_count: u64,
    pub adjudicated_candidate_count: u64,
    pub global_relation_valid_count: u64,
    pub global_relation_invalid_count: u64,
    pub global_relation_ambiguous_count: u64,
    pub target_delta_valid_count: u64,
    pub target_delta_invalid_count: u64,
    pub target_delta_ambiguous_count: u64,
    pub final_valid_count: u64,
    pub final_invalid_count: u64,
    pub final_ambiguous_count: u64,
    pub accepted_valid_count: u64,
    pub accepted_invalid_count: u64,
    pub accepted_ambiguous_count: u64,
    pub rejected_valid_count: u64,
    pub rejected_invalid_count: u64,
    pub rejected_ambiguous_count: u64,
    pub abstained_valid_count: u64,
    pub abstained_invalid_count: u64,
    pub abstained_ambiguous_count: u64,
    pub exact_changed_scope_match_count: u64,
    pub collateral_material_change_count: u64,
    pub reviewer_pairwise_agreement_ppm: Option<u32>,
    pub reviewer_pairwise_comparison_count: u64,
    pub reviewer_target_transition_agreement_ppm: Option<u32>,
    pub reviewer_target_transition_comparison_count: u64,
    pub generation_tokens: u64,
    pub generation_cost_micro_usd: u64,
    pub generation_wall_clock_ms: u64,
    pub final_valid_accepted_unique_per_1000_tokens: Option<f64>,
    pub final_valid_accepted_unique_per_usd: Option<f64>,
    pub family_metrics: Vec<TargetDeltaFamilyMetricsV1>,
    pub candidate_results: Vec<TargetDeltaCandidateResultV1>,
    pub reviewer_independence_proven: bool,
    pub reviewer_competence_proven: bool,
    pub semantic_truth_proven: bool,
    pub raw_artifact_custody_proven: bool,
    pub execution_authenticity_proven: bool,
    pub external_anchor_required: bool,
    pub scoring_eligible_declared: bool,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TargetDeltaReportV1 {
    #[serde(flatten)]
    pub fields: TargetDeltaReportFieldsV1,
    pub report_commitment_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ObservedTargetTuple {
    change: TargetChangeKindV1,
    changed_claims: Vec<String>,
    collateral: CollateralChangeV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelationValidity {
    Valid,
    Invalid,
    Ambiguous,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest_domain(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let length = u64::try_from(payload.len())
        .map_err(|_| "target-delta payload exceeds u64 length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(length.to_be_bytes());
    hasher.update(payload);
    Ok(sha256_hex(&hasher.finalize()))
}

fn valid_digest(value: &str, allow_zero: bool) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && (allow_zero || value.bytes().any(|byte| byte != b'0'))
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn nonempty(value: &str, max: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max
}

fn sorted_unique_digests(values: &[String], label: &str) -> Result<BTreeSet<String>, String> {
    if values.len() > MAX_SCOPE_IDS {
        return Err(format!("{label} exceeds {MAX_SCOPE_IDS} identities"));
    }
    let mut set = BTreeSet::new();
    let mut previous: Option<&str> = None;
    for value in values {
        if !valid_digest(value, false) {
            return Err(format!("{label} contains an invalid identity digest"));
        }
        if previous.is_some_and(|last| last >= value.as_str()) {
            return Err(format!("{label} must be strictly sorted and unique"));
        }
        previous = Some(value);
        set.insert(value.clone());
    }
    Ok(set)
}

fn plan_fields_commitment(fields: &TargetDeltaPlanFieldsV1) -> Result<String, String> {
    let payload = serde_json::to_vec(fields)
        .map_err(|error| format!("cannot serialize target-delta plan: {error}"))?;
    digest_domain(PLAN_DOMAIN, &payload)
}

/// Validate the target-delta specification and its correspondence to the parent quality plan.
pub fn validate_target_delta_plan_fields(
    fields: &TargetDeltaPlanFieldsV1,
    quality_plan_bytes: &[u8],
) -> Result<(), Vec<String>> {
    let mut issues = Vec::new();
    let allow_zero = !fields.purpose.qualified();
    if fields.schema != TARGET_DELTA_PLAN_SCHEMA {
        issues.push("unsupported target-delta plan schema".to_owned());
    }
    if !valid_revision(&fields.analyzer_source_revision)
        || fields.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        issues.push("analyzer_source_revision must match the exact compiled validator revision".to_owned());
    }
    if fields.purpose.qualified() && crate::BUILT_GENERATOR_WORKTREE_CLEAN != "true" {
        issues.push("qualified target-delta plans require a clean analyzer worktree".to_owned());
    }
    if !nonempty(&fields.study_id, MAX_LABEL) {
        issues.push("study_id must be non-empty and bounded".to_owned());
    }
    if !valid_digest(&fields.quality_plan_sha256, allow_zero)
        || fields.quality_plan_sha256 != sha256_hex(quality_plan_bytes)
    {
        issues.push("quality_plan_sha256 must bind the exact parent quality-plan bytes".to_owned());
    }

    let (parent_plan_sha256, parent_commitment, parent_revision, parent_study_id, parent_purpose) =
        match intervention_quality_plan_binding(quality_plan_bytes) {
            Ok(binding) => binding,
            Err(error) => {
                issues.push(format!("parent quality plan failed independent commitment verification: {error}"));
                (String::new(), String::new(), String::new(), String::new(), String::new())
            }
        };
    if !valid_digest(&fields.quality_plan_sha256, allow_zero)
        || fields.quality_plan_sha256 != parent_plan_sha256
    {
        issues.push("quality_plan_sha256 must bind the exact verified parent quality-plan bytes".to_owned());
    }
    if !valid_digest(&fields.quality_plan_commitment_sha256, allow_zero)
        || fields.quality_plan_commitment_sha256 != parent_commitment
    {
        issues.push("quality_plan_commitment_sha256 differs from the verified parent quality plan".to_owned());
    }
    if parent_revision != fields.analyzer_source_revision {
        issues.push("parent quality plan was committed by a different analyzer revision".to_owned());
    }
    if parent_study_id != fields.study_id {
        issues.push("target-delta study_id differs from its parent quality study_id".to_owned());
    }
    let own_purpose = if fields.purpose.qualified() { "qualified_run" } else { "example_only" };
    if parent_purpose != own_purpose {
        issues.push("target-delta purpose differs from its parent quality-plan purpose".to_owned());
    }
    let quality_plan: Value = match serde_json::from_slice(quality_plan_bytes) {
        Ok(value) => value,
        Err(error) => {
            issues.push(format!("parent quality plan is malformed: {error}"));
            Value::Null
        }
    };
    let parent_candidates = quality_plan.get("candidates").and_then(Value::as_array);
    let Some(parent_candidates) = parent_candidates else {
        issues.push("parent quality plan has no candidate census".to_owned());
        return Err(issues);
    };
    if fields.reviewers_per_candidate < 3 || fields.reviewers_per_candidate > MAX_PANEL {
        issues.push(format!("reviewers_per_candidate must be in 3..={MAX_PANEL}"));
    }
    if !(501..=1_000).contains(&fields.consensus_threshold_permille) {
        issues.push("consensus_threshold_permille must be in 501..=1000".to_owned());
    }
    if fields.candidates.len() != parent_candidates.len()
        || fields.candidates.is_empty()
        || fields.candidates.len() > MAX_CANDIDATES
    {
        issues.push("target-delta candidates must cover the exact parent quality-plan census".to_owned());
    }

    let mut parent_by_id = BTreeMap::new();
    for candidate in parent_candidates {
        let id = candidate.get("candidate_id").and_then(Value::as_str).unwrap_or("");
        let intended = candidate.get("intended_relation").and_then(Value::as_str).unwrap_or("");
        let baseline = candidate.get("baseline_input_sha256").and_then(Value::as_str).unwrap_or("");
        let variant = candidate.get("variant_input_sha256").and_then(Value::as_str).unwrap_or("");
        if id.is_empty()
            || parent_by_id
                .insert(id.to_owned(), (intended.to_owned(), baseline.to_owned(), variant.to_owned()))
                .is_some()
        {
            issues.push("parent quality plan contains duplicate or empty candidate IDs".to_owned());
        }
    }
    let mut seen = BTreeSet::new();
    for candidate in &fields.candidates {
        if !nonempty(&candidate.candidate_id, MAX_LABEL) || !seen.insert(candidate.candidate_id.as_str()) {
            issues.push("target-delta candidate IDs must be non-empty, bounded and unique".to_owned());
        }
        let parent_binding = parent_by_id.get(&candidate.candidate_id);
        let intended = parent_binding.map(|binding| binding.0.as_str()).unwrap_or("");
        if parent_binding.is_none() || (intended != "semantic_invariant" && intended != "semantic_change") {
            issues.push(format!("candidate {} does not exist in the parent quality plan", candidate.candidate_id));
        }
        if !valid_digest(&candidate.target_identity_sha256, allow_zero)
            || !valid_digest(&candidate.baseline_artifact_sha256, false)
            || !valid_digest(&candidate.variant_artifact_sha256, false)
            || !valid_digest(&candidate.baseline_target_state_sha256, allow_zero)
            || !valid_digest(&candidate.variant_target_state_sha256, allow_zero)
        {
            issues.push(format!("candidate {} has invalid target/artifact state commitments", candidate.candidate_id));
        }
        if let Some((_, parent_baseline, parent_variant)) = parent_binding
            && (candidate.baseline_artifact_sha256 != *parent_baseline
                || candidate.variant_artifact_sha256 != *parent_variant)
        {
            issues.push(format!("candidate {} target-delta artifacts differ from parent quality-plan inputs", candidate.candidate_id));
        }
        let expected_changed = sorted_unique_digests(
            &candidate.expected_changed_claim_ids_sha256,
            "expected_changed_claim_ids_sha256",
        );
        let preserved = sorted_unique_digests(
            &candidate.preserved_claim_ids_sha256,
            "preserved_claim_ids_sha256",
        );
        match (expected_changed, preserved) {
            (Ok(changed), Ok(unchanged)) => {
                if !changed.is_disjoint(&unchanged) {
                    issues.push(format!("candidate {} changed/preserved claim sets overlap", candidate.candidate_id));
                }
                if candidate.expected_target_change == TargetChangeKindV1::Unchanged {
                    if !changed.is_empty()
                        || !unchanged.contains(&candidate.target_identity_sha256)
                        || candidate.baseline_target_state_sha256 != candidate.variant_target_state_sha256
                    {
                        issues.push(format!("candidate {} unchanged target must preserve target state and declare no changed IDs", candidate.candidate_id));
                    }
                } else if !changed.contains(&candidate.target_identity_sha256)
                    || unchanged.contains(&candidate.target_identity_sha256)
                    || candidate.baseline_target_state_sha256 == candidate.variant_target_state_sha256
                {
                    issues.push(format!("candidate {} changed target must include target identity, exclude it from preserved IDs and change its state digest", candidate.candidate_id));
                }
            }
            (Err(error), _) | (_, Err(error)) => issues.push(format!("candidate {}: {error}", candidate.candidate_id)),
        }
        if intended == "semantic_invariant" && candidate.expected_target_change != TargetChangeKindV1::Unchanged {
            issues.push(format!("candidate {} conflicts with parent semantic_invariant relation", candidate.candidate_id));
        }
        if intended == "semantic_change" && !candidate.expected_target_change.changed() {
            issues.push(format!("candidate {} conflicts with parent semantic_change relation", candidate.candidate_id));
        }
    }
    if seen.len() != parent_by_id.len()
        || parent_by_id.keys().any(|id| !seen.contains(id.as_str()))
    {
        issues.push("target-delta plan omits or adds parent candidate IDs".to_owned());
    }
    if issues.is_empty() { Ok(()) } else { Err(issues) }
}

/// Commit a target-delta plan fields object after validating it against the parent plan.
pub fn compute_target_delta_plan_commitment(
    fields_bytes: &[u8],
    quality_plan_bytes: &[u8],
) -> Result<String, String> {
    let fields: TargetDeltaPlanFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("target-delta plan fields invalid: {error}"))?;
    validate_target_delta_plan_fields(&fields, quality_plan_bytes)
        .map_err(|issues| issues.join("; "))?;
    plan_fields_commitment(&fields)
}

fn verify_target_delta_plan(
    plan_bytes: &[u8],
    quality_plan_bytes: &[u8],
) -> Result<(TargetDeltaPlanV1, String), String> {
    let plan: TargetDeltaPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("target-delta plan invalid: {error}"))?;
    let fields = plan.fields();
    validate_target_delta_plan_fields(&fields, quality_plan_bytes)
        .map_err(|issues| issues.join("; "))?;
    let expected = plan_fields_commitment(&fields)?;
    if !valid_digest(&plan.plan_commitment_sha256, !plan.purpose.qualified())
        || plan.plan_commitment_sha256 != expected
    {
        return Err("target-delta plan commitment mismatch".to_owned());
    }
    Ok((plan, sha256_hex(plan_bytes)))
}

fn relation_validity_from_parent(
    intended: QualityRelationV1,
    votes: &[crate::intervention_quality::HumanAdjudicationV1],
    threshold_permille: u16,
) -> RelationValidity {
    let total = votes.len() as u64;
    if total == 0 { return RelationValidity::Ambiguous; }
    let preserving = votes.iter().filter(|vote| vote.verdict == crate::intervention_quality::ReviewerVerdictV1::MeaningPreserving).count() as u64;
    let changing = votes.iter().filter(|vote| vote.verdict == crate::intervention_quality::ReviewerVerdictV1::MeaningChanging).count() as u64;
    let ambiguous = total - preserving - changing;
    let (expected, opposite) = match intended {
        QualityRelationV1::SemanticInvariant => (preserving, changing),
        QualityRelationV1::SemanticChange => (changing, preserving),
    };
    if expected * 1_000 >= total * u64::from(threshold_permille) {
        RelationValidity::Valid
    } else if opposite * 1_000 >= total * u64::from(threshold_permille) {
        RelationValidity::Invalid
    } else {
        let _ = ambiguous;
        RelationValidity::Ambiguous
    }
}

fn target_tuple_consensus(
    reviewers: &[TargetDeltaReviewerObservationV1],
    threshold_permille: u16,
) -> (Option<ObservedTargetTuple>, Option<u32>, u64, u64, u64, u64) {
    let mut votes: BTreeMap<ObservedTargetTuple, u64> = BTreeMap::new();
    let mut tuple_matches = 0_u64;
    let mut transition_matches = 0_u64;
    let mut comparisons = 0_u64;
    for (index, reviewer) in reviewers.iter().enumerate() {
        let tuple = ObservedTargetTuple {
            change: reviewer.observed_target_change,
            changed_claims: reviewer.observed_changed_claim_ids_sha256.clone(),
            collateral: reviewer.collateral_change,
        };
        *votes.entry(tuple.clone()).or_default() += 1;
        for other in reviewers.iter().skip(index + 1) {
            comparisons += 1;
            let other_tuple = ObservedTargetTuple {
                change: other.observed_target_change,
                changed_claims: other.observed_changed_claim_ids_sha256.clone(),
                collateral: other.collateral_change,
            };
            if tuple == other_tuple {
                tuple_matches += 1;
            }
            if reviewer.observed_target_change == other.observed_target_change {
                transition_matches += 1;
            }
        }
    }
    let mut ranked: Vec<(ObservedTargetTuple, u64)> = votes.into_iter().collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let Some((tuple, count)) = ranked.into_iter().next() else {
        return (None, None, tuple_matches, comparisons, transition_matches, comparisons);
    };
    let total = reviewers.len() as u64;
    let consensus_ppm = ((count * 1_000_000) / total) as u32;
    if count * 1_000 < total * u64::from(threshold_permille) {
        return (None, Some(consensus_ppm), tuple_matches, comparisons, transition_matches, comparisons);
    }
    (Some(tuple), Some(consensus_ppm), tuple_matches, comparisons, transition_matches, comparisons)
}

fn target_delta_validity(
    target: Option<&ObservedTargetTuple>,
    candidate: &TargetDeltaCandidateV1,
) -> (FinalCandidateValidityV1, bool) {
    let Some(observed) = target else {
        return (FinalCandidateValidityV1::Ambiguous, false);
    };
    let scope_matches = observed.changed_claims == candidate.expected_changed_claim_ids_sha256;
    if observed.change == TargetChangeKindV1::OtherDeclaredChange {
        return (FinalCandidateValidityV1::Ambiguous, scope_matches);
    }
    if observed.change != candidate.expected_target_change
        || !scope_matches
        || observed.collateral == CollateralChangeV1::MaterialChangeOutsideAllowedScope
    {
        return (FinalCandidateValidityV1::Invalid, scope_matches);
    }
    if observed.collateral == CollateralChangeV1::Ambiguous {
        return (FinalCandidateValidityV1::Ambiguous, scope_matches);
    }
    (FinalCandidateValidityV1::Valid, true)
}

fn combine_validity(
    global: RelationValidity,
    target: FinalCandidateValidityV1,
) -> FinalCandidateValidityV1 {
    match (global, target) {
        (RelationValidity::Invalid, _) | (_, FinalCandidateValidityV1::Invalid) => {
            FinalCandidateValidityV1::Invalid
        }
        (RelationValidity::Ambiguous, _) | (_, FinalCandidateValidityV1::Ambiguous) => {
            FinalCandidateValidityV1::Ambiguous
        }
        (RelationValidity::Valid, FinalCandidateValidityV1::Valid) => FinalCandidateValidityV1::Valid,
    }
}

fn ppm(numerator: u64, denominator: u64) -> Option<u32> {
    (denominator > 0).then(|| ((numerator * 1_000_000) / denominator) as u32)
}

fn per_1000_tokens(count: u64, tokens: u64) -> Option<f64> {
    (tokens > 0).then(|| count as f64 * 1_000.0 / tokens as f64)
}

fn per_usd(count: u64, cost_micro_usd: u64) -> Option<f64> {
    (cost_micro_usd > 0).then(|| count as f64 * 1_000_000.0 / cost_micro_usd as f64)
}

/// Analyze v1 global relation adjudications and v2 blinded target-delta observations together.
pub fn analyze_intervention_target_delta(
    quality_plan_bytes: &[u8],
    quality_receipt_bytes: &[u8],
    target_plan_bytes: &[u8],
    target_receipt_bytes: &[u8],
) -> Result<TargetDeltaReportV1, String> {
    let quality_report = analyze_intervention_quality(quality_plan_bytes, quality_receipt_bytes)?;
    let (target_plan, target_plan_sha256) =
        verify_target_delta_plan(target_plan_bytes, quality_plan_bytes)?;
    let quality_plan: Value = serde_json::from_slice(quality_plan_bytes)
        .map_err(|error| format!("parent quality plan invalid: {error}"))?;
    let quality_receipt: Value = serde_json::from_slice(quality_receipt_bytes)
        .map_err(|error| format!("parent quality receipt invalid: {error}"))?;
    let target_receipt: TargetDeltaReceiptV1 = serde_json::from_slice(target_receipt_bytes)
        .map_err(|error| format!("target-delta receipt invalid: {error}"))?;

    if target_receipt.schema != TARGET_DELTA_RECEIPT_SCHEMA {
        return Err("unsupported target-delta receipt schema".to_owned());
    }
    if target_receipt.purpose != target_plan.purpose {
        return Err("target-delta receipt purpose differs from plan".to_owned());
    }
    if !nonempty(&target_receipt.run_id, MAX_LABEL) {
        return Err("target-delta receipt run_id must be non-empty and bounded".to_owned());
    }
    if !valid_revision(&target_receipt.analyzer_source_revision)
        || target_receipt.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
        || target_receipt.analyzer_source_revision != target_plan.analyzer_source_revision
    {
        return Err("target-delta receipt analyzer revision differs from exact compiled revision".to_owned());
    }
    if target_receipt.target_plan_sha256 != target_plan_sha256
        || target_receipt.target_plan_commitment_sha256 != target_plan.plan_commitment_sha256
        || target_receipt.quality_plan_sha256 != target_plan.quality_plan_sha256
        || target_receipt.quality_receipt_sha256 != sha256_hex(quality_receipt_bytes)
    {
        return Err("target-delta receipt does not bind exact target and parent quality artifacts".to_owned());
    }
    if target_receipt.quality_plan_sha256 != sha256_hex(quality_plan_bytes) {
        return Err("target-delta receipt quality plan digest mismatch".to_owned());
    }
    let quality_commitment = quality_plan.get("plan_commitment_sha256").and_then(Value::as_str).unwrap_or("");
    if target_plan.quality_plan_commitment_sha256 != quality_commitment {
        return Err("target-delta plan parent commitment mismatch".to_owned());
    }

    let quality_candidates = quality_plan.get("candidates").and_then(Value::as_array)
        .ok_or_else(|| "parent quality plan candidate census missing".to_owned())?;
    let quality_adjudications = quality_receipt.get("adjudications").and_then(Value::as_array)
        .ok_or_else(|| "parent quality receipt adjudications missing".to_owned())?;
    let mut quality_adjudications_by_id = BTreeMap::new();
    for row in quality_adjudications {
        let id = row.get("candidate_id").and_then(Value::as_str).unwrap_or("");
        if quality_adjudications_by_id.insert(id.to_owned(), row).is_some() {
            return Err("parent quality receipt contains duplicate candidate adjudication".to_owned());
        }
    }
    let mut target_specs_by_id = BTreeMap::new();
    for candidate in &target_plan.candidates {
        target_specs_by_id.insert(candidate.candidate_id.as_str(), candidate);
    }
    let mut target_adjudications_by_id = BTreeMap::new();
    for adjudication in &target_receipt.adjudications {
        if target_adjudications_by_id.insert(adjudication.candidate_id.as_str(), adjudication).is_some() {
            return Err(format!("duplicate target-delta adjudication {}", adjudication.candidate_id));
        }
    }
    if target_adjudications_by_id.len() != target_specs_by_id.len()
        || target_specs_by_id.keys().any(|id| !target_adjudications_by_id.contains_key(id))
    {
        return Err("target-delta receipt must contain exactly one adjudication for every planned candidate".to_owned());
    }

    let mut reviewer_matches = 0_u64;
    let mut reviewer_comparisons = 0_u64;
    let mut transition_matches = 0_u64;
    let mut transition_comparisons = 0_u64;
    let mut candidate_results = Vec::new();
    let mut family = BTreeMap::<String, TargetDeltaFamilyMetricsV1>::new();
    let mut global_valid = 0_u64;
    let mut global_invalid = 0_u64;
    let mut global_ambiguous = 0_u64;
    let mut target_valid = 0_u64;
    let mut target_invalid = 0_u64;
    let mut target_ambiguous = 0_u64;
    let mut final_valid = 0_u64;
    let mut final_invalid = 0_u64;
    let mut final_ambiguous = 0_u64;
    let mut exact_scope_matches = 0_u64;
    let mut collateral_changes = 0_u64;
    let mut accepted_valid = 0_u64;
    let mut accepted_invalid = 0_u64;
    let mut accepted_ambiguous = 0_u64;
    let mut rejected_valid = 0_u64;
    let mut rejected_invalid = 0_u64;
    let mut rejected_ambiguous = 0_u64;
    let mut abstained_valid = 0_u64;
    let mut abstained_invalid = 0_u64;
    let mut abstained_ambiguous = 0_u64;
    let mut tokens = 0_u64;
    let mut cost = 0_u64;
    let mut wall = 0_u64;
    let mut family_cluster_ids = BTreeMap::<String, BTreeSet<String>>::new();
    let mut valid_accepted_clusters = BTreeSet::new();

    for quality_candidate in quality_candidates {
        let id = quality_candidate.get("candidate_id").and_then(Value::as_str).unwrap_or("");
        let intended_relation = match quality_candidate.get("intended_relation").and_then(Value::as_str).unwrap_or("") {
            "semantic_invariant" => QualityRelationV1::SemanticInvariant,
            "semantic_change" => QualityRelationV1::SemanticChange,
            _ => return Err(format!("candidate {id} has unknown parent intended_relation")),
        };
        let disposition = match quality_candidate.get("pipeline_disposition").and_then(Value::as_str).unwrap_or("") {
            "accepted" => QualityDispositionV1::Accepted,
            "rejected" => QualityDispositionV1::Rejected,
            "abstained" => QualityDispositionV1::Abstained,
            _ => return Err(format!("candidate {id} has unknown parent pipeline disposition")),
        };
        let family_name = quality_candidate.get("transformation_family").and_then(Value::as_str).unwrap_or("").to_owned();
        let failure_mode = quality_candidate.get("target_failure_mode").and_then(Value::as_str).unwrap_or("").to_owned();
        let cluster = quality_candidate.get("semantic_cluster_sha256").and_then(Value::as_str).unwrap_or("").to_owned();
        let generation_tokens = quality_candidate.get("generation_tokens").and_then(Value::as_u64).unwrap_or(0);
        let generation_cost = quality_candidate.get("generation_cost_micro_usd").and_then(Value::as_u64).unwrap_or(0);
        let generation_wall = quality_candidate.get("generation_wall_clock_ms").and_then(Value::as_u64).unwrap_or(0);
        tokens = tokens.checked_add(generation_tokens).ok_or_else(|| "generation token total overflow".to_owned())?;
        cost = cost.checked_add(generation_cost).ok_or_else(|| "generation cost total overflow".to_owned())?;
        wall = wall.checked_add(generation_wall).ok_or_else(|| "generation wall total overflow".to_owned())?;

        let target_spec = target_specs_by_id.get(id).ok_or_else(|| format!("target-delta plan omitted candidate {id}"))?;
        let target_adjudication = target_adjudications_by_id.get(id).ok_or_else(|| format!("target-delta receipt omitted candidate {id}"))?;
        let parent_adjudication = quality_adjudications_by_id.get(id).ok_or_else(|| format!("parent quality receipt omitted candidate {id}"))?;
        let parent_reviewers = parent_adjudication.get("reviewers").and_then(Value::as_array)
            .ok_or_else(|| format!("parent quality receipt candidate {id} has no reviewer panel"))?;
        let parent_votes: Vec<crate::intervention_quality::HumanAdjudicationV1> =
            serde_json::from_value(Value::Array(parent_reviewers.clone()))
                .map_err(|error| format!("parent quality receipt reviewer panel invalid for {id}: {error}"))?;
        let parent_policy = quality_plan.get("adjudication_policy")
            .ok_or_else(|| "parent quality plan adjudication_policy missing".to_owned())?;
        let parent_threshold = parent_policy.get("consensus_threshold_permille").and_then(Value::as_u64).unwrap_or(0) as u16;
        let global_validity = relation_validity_from_parent(intended_relation, &parent_votes, parent_threshold);
        match global_validity {
            RelationValidity::Valid => global_valid += 1,
            RelationValidity::Invalid => global_invalid += 1,
            RelationValidity::Ambiguous => global_ambiguous += 1,
        }

        let reviewers = &target_adjudication.reviewers;
        if reviewers.len() != target_plan.reviewers_per_candidate as usize {
            return Err(format!("candidate {id} requires exactly {} target-delta reviewers", target_plan.reviewers_per_candidate));
        }
        let mut reviewer_ids = BTreeSet::new();
        for reviewer in reviewers {
            if !valid_digest(&reviewer.reviewer_id_sha256, false)
                || !reviewer_ids.insert(reviewer.reviewer_id_sha256.as_str())
            {
                return Err(format!("candidate {id} has invalid or duplicate target-delta reviewer ID"));
            }
            if !valid_digest(&reviewer.adjudication_artifact_sha256, false) {
                return Err(format!("candidate {id} has invalid target-delta adjudication artifact digest"));
            }
            if !valid_digest(&reviewer.reviewed_target_identity_sha256, false)
                || reviewer.reviewed_target_identity_sha256 != target_spec.target_identity_sha256
                || !valid_digest(&reviewer.baseline_artifact_sha256, false)
                || reviewer.baseline_artifact_sha256 != target_spec.baseline_artifact_sha256
                || !valid_digest(&reviewer.variant_artifact_sha256, false)
                || reviewer.variant_artifact_sha256 != target_spec.variant_artifact_sha256
            {
                return Err(format!("candidate {id} reviewer target/artifact binding differs from the committed plan"));
            }
            if !reviewer.blind_to_generator_identity
                || !reviewer.blind_to_intended_relation
                || !reviewer.blind_to_pipeline_disposition
                || !reviewer.blind_to_evaluator_outcome
                || !reviewer.blind_to_expected_target_change
                || !reviewer.blind_to_allowed_changed_claims
            {
                return Err(format!("candidate {id} contains a non-blinded target-delta reviewer"));
            }
            let observed = sorted_unique_digests(
                &reviewer.observed_changed_claim_ids_sha256,
                "observed_changed_claim_ids_sha256",
            ).map_err(|error| format!("candidate {id}: {error}"))?;
            if !observed.is_disjoint(&target_spec.preserved_claim_ids_sha256.iter().cloned().collect()) {
                if reviewer.collateral_change != CollateralChangeV1::MaterialChangeOutsideAllowedScope {
                    return Err(format!("candidate {id} reviewer failed to flag a preserved-scope change as collateral"));
                }
            }
        }
        let (
            observed_tuple,
            consensus_ppm,
            tuple_matches,
            tuple_comparisons,
            transition_pair_matches,
            transition_pair_comparisons,
        ) = target_tuple_consensus(reviewers, target_plan.consensus_threshold_permille);
        reviewer_matches += tuple_matches;
        reviewer_comparisons += tuple_comparisons;
        transition_matches += transition_pair_matches;
        transition_comparisons += transition_pair_comparisons;
        let (target_validity, scope_matches) =
            target_delta_validity(observed_tuple.as_ref(), target_spec);
        match target_validity {
            FinalCandidateValidityV1::Valid => target_valid += 1,
            FinalCandidateValidityV1::Invalid => target_invalid += 1,
            FinalCandidateValidityV1::Ambiguous => target_ambiguous += 1,
        }
        if scope_matches { exact_scope_matches += 1; }
        if observed_tuple.as_ref().is_some_and(|tuple| tuple.collateral == CollateralChangeV1::MaterialChangeOutsideAllowedScope) {
            collateral_changes += 1;
        }
        let finality = combine_validity(global_validity, target_validity);
        match finality {
            FinalCandidateValidityV1::Valid => final_valid += 1,
            FinalCandidateValidityV1::Invalid => final_invalid += 1,
            FinalCandidateValidityV1::Ambiguous => final_ambiguous += 1,
        }
        let metrics = family.entry(family_name.clone()).or_insert_with(|| TargetDeltaFamilyMetricsV1 {
            transformation_family: family_name.clone(),
            candidate_count: 0,
            unique_semantic_cluster_count: 0,
            valid_count: 0,
            invalid_count: 0,
            ambiguous_count: 0,
            accepted_valid_count: 0,
            accepted_invalid_count: 0,
            accepted_ambiguous_count: 0,
        });
        metrics.candidate_count += 1;
        match finality {
            FinalCandidateValidityV1::Valid => metrics.valid_count += 1,
            FinalCandidateValidityV1::Invalid => metrics.invalid_count += 1,
            FinalCandidateValidityV1::Ambiguous => metrics.ambiguous_count += 1,
        }
        match disposition {
            QualityDispositionV1::Accepted => match finality {
                FinalCandidateValidityV1::Valid => { accepted_valid += 1; metrics.accepted_valid_count += 1; valid_accepted_clusters.insert(cluster.clone()); }
                FinalCandidateValidityV1::Invalid => { accepted_invalid += 1; metrics.accepted_invalid_count += 1; }
                FinalCandidateValidityV1::Ambiguous => { accepted_ambiguous += 1; metrics.accepted_ambiguous_count += 1; }
            },
            QualityDispositionV1::Rejected => match finality {
                FinalCandidateValidityV1::Valid => rejected_valid += 1,
                FinalCandidateValidityV1::Invalid => rejected_invalid += 1,
                FinalCandidateValidityV1::Ambiguous => rejected_ambiguous += 1,
            },
            QualityDispositionV1::Abstained => match finality {
                FinalCandidateValidityV1::Valid => abstained_valid += 1,
                FinalCandidateValidityV1::Invalid => abstained_invalid += 1,
                FinalCandidateValidityV1::Ambiguous => abstained_ambiguous += 1,
            },
        }
        family_cluster_ids.entry(family_name.clone()).or_default().insert(cluster);
        candidate_results.push(TargetDeltaCandidateResultV1 {
            candidate_id: id.to_owned(),
            transformation_family: family_name,
            target_failure_mode: failure_mode,
            pipeline_disposition: disposition,
            intended_relation,
            expected_target_change: target_spec.expected_target_change,
            observed_target_change: observed_tuple.as_ref().map(|tuple| tuple.change),
            expected_changed_claim_count: target_spec.expected_changed_claim_ids_sha256.len() as u64,
            observed_changed_claim_count: observed_tuple.as_ref().map(|tuple| tuple.changed_claims.len() as u64).unwrap_or(0),
            exact_changed_scope_match: scope_matches,
            collateral_change: observed_tuple.as_ref().map(|tuple| tuple.collateral),
            global_relation_validity: match global_validity {
                RelationValidity::Valid => FinalCandidateValidityV1::Valid,
                RelationValidity::Invalid => FinalCandidateValidityV1::Invalid,
                RelationValidity::Ambiguous => FinalCandidateValidityV1::Ambiguous,
            },
            target_delta_validity: target_validity,
            final_validity: finality,
            reviewer_consensus_ppm: consensus_ppm,
            reviewer_count: reviewers.len() as u64,
        });
    }

    if candidate_results.len() != quality_report.fields.candidate_count as usize {
        return Err("target-delta result coverage differs from parent quality census".to_owned());
    }
    let final_valid_accepted_unique = valid_accepted_clusters.len() as u64;
    let fields = TargetDeltaReportFieldsV1 {
        schema: TARGET_DELTA_REPORT_SCHEMA.to_owned(),
        study_id: target_plan.study_id,
        analyzer_source_revision: target_plan.analyzer_source_revision,
        quality_plan_sha256: target_plan.quality_plan_sha256,
        quality_receipt_sha256: sha256_hex(quality_receipt_bytes),
        target_plan_sha256,
        target_receipt_sha256: sha256_hex(target_receipt_bytes),
        candidate_count: candidate_results.len() as u64,
        adjudicated_candidate_count: candidate_results.len() as u64,
        global_relation_valid_count: global_valid,
        global_relation_invalid_count: global_invalid,
        global_relation_ambiguous_count: global_ambiguous,
        target_delta_valid_count: target_valid,
        target_delta_invalid_count: target_invalid,
        target_delta_ambiguous_count: target_ambiguous,
        final_valid_count: final_valid,
        final_invalid_count: final_invalid,
        final_ambiguous_count: final_ambiguous,
        accepted_valid_count: accepted_valid,
        accepted_invalid_count: accepted_invalid,
        accepted_ambiguous_count: accepted_ambiguous,
        rejected_valid_count: rejected_valid,
        rejected_invalid_count: rejected_invalid,
        rejected_ambiguous_count: rejected_ambiguous,
        abstained_valid_count: abstained_valid,
        abstained_invalid_count: abstained_invalid,
        abstained_ambiguous_count: abstained_ambiguous,
        exact_changed_scope_match_count: exact_scope_matches,
        collateral_material_change_count: collateral_changes,
        reviewer_pairwise_agreement_ppm: ppm(reviewer_matches, reviewer_comparisons),
        reviewer_pairwise_comparison_count: reviewer_comparisons,
        reviewer_target_transition_agreement_ppm: ppm(transition_matches, transition_comparisons),
        reviewer_target_transition_comparison_count: transition_comparisons,
        generation_tokens: tokens,
        generation_cost_micro_usd: cost,
        generation_wall_clock_ms: wall,
        final_valid_accepted_unique_per_1000_tokens: per_1000_tokens(final_valid_accepted_unique, tokens),
        final_valid_accepted_unique_per_usd: per_usd(final_valid_accepted_unique, cost),
        family_metrics: family
            .into_iter()
            .map(|(family_name, mut metrics)| {
                metrics.unique_semantic_cluster_count = family_cluster_ids
                    .get(&family_name)
                    .map(|clusters| clusters.len() as u64)
                    .unwrap_or(0);
                metrics
            })
            .collect(),
        candidate_results,
        reviewer_independence_proven: false,
        reviewer_competence_proven: false,
        semantic_truth_proven: false,
        raw_artifact_custody_proven: false,
        execution_authenticity_proven: false,
        external_anchor_required: true,
        scoring_eligible_declared: false,
        claim_boundary: "Compares blinded reviewer observations to a target-delta specification bound to the parent intervention-quality census. It does not prove reviewer independence/competence, semantic truth, raw artifact custody, execution authenticity or downstream judge failure induction.".to_owned(),
    };
    let payload = serde_json::to_vec(&fields).map_err(|error| format!("cannot serialize target-delta report: {error}"))?;
    let report_commitment_sha256 = digest_domain(REPORT_DOMAIN, &payload)?;
    Ok(TargetDeltaReportV1 { fields, report_commitment_sha256 })
}


#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn blinded_quality_reviewers(candidate_id: &str, verdict: &str) -> Vec<Value> {
        (0..3)
            .map(|index| {
                json!({
                    "reviewer_id_sha256": digest(&format!("quality-reviewer-{candidate_id}-{index}")),
                    "verdict": verdict,
                    "blind_to_generator_identity": true,
                    "blind_to_intended_relation": true,
                    "blind_to_pipeline_disposition": true,
                    "blind_to_evaluator_outcome": true,
                    "adjudication_artifact_sha256": digest(&format!("quality-artifact-{candidate_id}-{index}"))
                })
            })
            .collect()
    }

    fn target_reviewers(
        candidate_id: &str,
        target_identity_sha256: &str,
        baseline_artifact_sha256: &str,
        variant_artifact_sha256: &str,
        change: &str,
        changed_claims: &[String],
        collateral: &str,
    ) -> Vec<Value> {
        (0..3)
            .map(|index| {
                json!({
                    "reviewer_id_sha256": digest(&format!("target-reviewer-{candidate_id}-{index}")),
                    "reviewed_target_identity_sha256": target_identity_sha256,
                    "baseline_artifact_sha256": baseline_artifact_sha256,
                    "variant_artifact_sha256": variant_artifact_sha256,
                    "observed_target_change": change,
                    "observed_changed_claim_ids_sha256": changed_claims,
                    "collateral_change": collateral,
                    "blind_to_generator_identity": true,
                    "blind_to_intended_relation": true,
                    "blind_to_pipeline_disposition": true,
                    "blind_to_evaluator_outcome": true,
                    "blind_to_expected_target_change": true,
                    "blind_to_allowed_changed_claims": true,
                    "adjudication_artifact_sha256": digest(&format!("target-artifact-{candidate_id}-{index}"))
                })
            })
            .collect()
    }

    fn fixtures() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, String) {
        let analyzer_revision = crate::BUILT_GENERATOR_SOURCE_REVISION.to_owned();
        let invariant_target_id = digest("claim:invariant-target");
        let changed_target_id = digest("claim:changed-target");
        let invariant_control_id = digest("claim:invariant-control");
        let changed_control_id = digest("claim:changed-control");
        let invariant_baseline = digest("state:invariant-target");
        let changed_baseline = digest("state:changed-target:baseline");
        let changed_variant = digest("state:changed-target:variant");
        let invariant_baseline_artifact = digest("invariant baseline input");
        let invariant_variant_artifact = digest("invariant variant input");
        let changed_baseline_artifact = digest("change baseline input");
        let changed_variant_artifact = digest("change variant input");

        let quality_fields = json!({
            "schema": crate::intervention_quality::QUALITY_PLAN_SCHEMA,
            "analyzer_source_revision": analyzer_revision,
            "purpose": "example_only",
            "study_id": "target-delta-study-001",
            "generator_source_revision": "a".repeat(40),
            "generator_prompt_sha256": digest("generator prompt"),
            "semantic_verifier_source_revision": "b".repeat(40),
            "semantic_verifier_policy_sha256": digest("semantic verifier policy"),
            "candidate_sampling_policy": "full_census",
            "adjudication_policy": {
                "reviewers_per_candidate": 3,
                "consensus_threshold_permille": 666,
                "require_blind_to_generator_identity": true,
                "require_blind_to_intended_relation": true,
                "require_blind_to_pipeline_disposition": true,
                "require_blind_to_evaluator_outcome": true
            },
            "candidates": [
                {
                    "candidate_id": "candidate-invariant",
                    "transformation_family": "formatting",
                    "target_failure_mode": "surface-format",
                    "semantic_cluster_sha256": digest("cluster invariant"),
                    "intended_relation": "semantic_invariant",
                    "pipeline_disposition": "accepted",
                    "baseline_input_sha256": invariant_baseline_artifact,
                    "variant_input_sha256": invariant_variant_artifact,
                    "generation_tokens": 100,
                    "generation_cost_micro_usd": 1000,
                    "generation_wall_clock_ms": 10
                },
                {
                    "candidate_id": "candidate-change",
                    "transformation_family": "evidence",
                    "target_failure_mode": "evidence-withdrawal",
                    "semantic_cluster_sha256": digest("cluster changed"),
                    "intended_relation": "semantic_change",
                    "pipeline_disposition": "accepted",
                    "baseline_input_sha256": changed_baseline_artifact,
                    "variant_input_sha256": changed_variant_artifact,
                    "generation_tokens": 200,
                    "generation_cost_micro_usd": 2000,
                    "generation_wall_clock_ms": 20
                }
            ]
        });
        let quality_fields_bytes = serde_json::to_vec(&quality_fields).unwrap();
        let quality_commitment =
            crate::intervention_quality::compute_intervention_quality_plan_commitment(&quality_fields_bytes).unwrap();
        let mut quality_plan_value = quality_fields.clone();
        quality_plan_value["plan_commitment_sha256"] = json!(quality_commitment);
        let quality_plan_bytes = serde_json::to_vec_pretty(&quality_plan_value).unwrap();

        let quality_receipt_value = json!({
            "schema": crate::intervention_quality::QUALITY_RECEIPT_SCHEMA,
            "analyzer_source_revision": analyzer_revision,
            "purpose": "example_only",
            "run_id": "quality-run-001",
            "plan_sha256": sha256_hex(&quality_plan_bytes),
            "plan_commitment_sha256": quality_commitment,
            "generator_source_revision": "a".repeat(40),
            "semantic_verifier_source_revision": "b".repeat(40),
            "semantic_verifier_policy_sha256": digest("semantic verifier policy"),
            "adjudications": [
                {
                    "candidate_id": "candidate-invariant",
                    "reviewers": blinded_quality_reviewers("candidate-invariant", "meaning_preserving")
                },
                {
                    "candidate_id": "candidate-change",
                    "reviewers": blinded_quality_reviewers("candidate-change", "meaning_changing")
                }
            ]
        });
        let quality_receipt_bytes = serde_json::to_vec_pretty(&quality_receipt_value).unwrap();

        let invariant_changed = Vec::<String>::new();
        let mut invariant_preserved = vec![invariant_target_id.clone(), invariant_control_id];
        invariant_preserved.sort();
        let mut changed_claims = vec![changed_target_id.clone()];
        changed_claims.sort();
        let mut changed_preserved = vec![changed_control_id];
        changed_preserved.sort();

        let target_fields = json!({
            "schema": TARGET_DELTA_PLAN_SCHEMA,
            "analyzer_source_revision": analyzer_revision,
            "purpose": "example_only",
            "study_id": "target-delta-study-001",
            "quality_plan_sha256": sha256_hex(&quality_plan_bytes),
            "quality_plan_commitment_sha256": quality_commitment,
            "reviewers_per_candidate": 3,
            "consensus_threshold_permille": 666,
            "candidates": [
                {
                    "candidate_id": "candidate-invariant",
                    "target_identity_sha256": invariant_target_id,
                    "baseline_artifact_sha256": invariant_baseline_artifact,
                    "variant_artifact_sha256": invariant_variant_artifact,
                    "baseline_target_state_sha256": invariant_baseline,
                    "variant_target_state_sha256": invariant_baseline,
                    "expected_target_change": "unchanged",
                    "expected_changed_claim_ids_sha256": invariant_changed,
                    "preserved_claim_ids_sha256": invariant_preserved
                },
                {
                    "candidate_id": "candidate-change",
                    "target_identity_sha256": changed_target_id,
                    "baseline_artifact_sha256": changed_baseline_artifact,
                    "variant_artifact_sha256": changed_variant_artifact,
                    "baseline_target_state_sha256": changed_baseline,
                    "variant_target_state_sha256": changed_variant,
                    "expected_target_change": "support_decreased",
                    "expected_changed_claim_ids_sha256": changed_claims,
                    "preserved_claim_ids_sha256": changed_preserved
                }
            ]
        });
        let target_fields_bytes = serde_json::to_vec(&target_fields).unwrap();
        let target_commitment =
            compute_target_delta_plan_commitment(&target_fields_bytes, &quality_plan_bytes).unwrap();
        let mut target_plan_value = target_fields.clone();
        target_plan_value["plan_commitment_sha256"] = json!(target_commitment);
        let target_plan_bytes = serde_json::to_vec_pretty(&target_plan_value).unwrap();

        let target_receipt_value = json!({
            "schema": TARGET_DELTA_RECEIPT_SCHEMA,
            "analyzer_source_revision": analyzer_revision,
            "purpose": "example_only",
            "run_id": "target-delta-run-001",
            "target_plan_sha256": sha256_hex(&target_plan_bytes),
            "target_plan_commitment_sha256": target_commitment,
            "quality_plan_sha256": sha256_hex(&quality_plan_bytes),
            "quality_receipt_sha256": sha256_hex(&quality_receipt_bytes),
            "adjudications": [
                {
                    "candidate_id": "candidate-invariant",
                    "reviewers": target_reviewers("candidate-invariant", &invariant_target_id, &invariant_baseline_artifact, &invariant_variant_artifact, "unchanged", &[], "none_observed")
                },
                {
                    "candidate_id": "candidate-change",
                    "reviewers": target_reviewers("candidate-change", &changed_target_id, &changed_baseline_artifact, &changed_variant_artifact, "support_decreased", &[changed_target_id.clone()], "none_observed")
                }
            ]
        });
        let target_receipt_bytes = serde_json::to_vec_pretty(&target_receipt_value).unwrap();

        (quality_plan_bytes, quality_receipt_bytes, target_plan_bytes, target_receipt_bytes, changed_target_id)
    }

    fn parse_receipt(bytes: &[u8]) -> Value {
        serde_json::from_slice(bytes).unwrap()
    }

    fn report(
        quality_plan: &[u8],
        quality_receipt: &[u8],
        target_plan: &[u8],
        target_receipt: &[u8],
    ) -> TargetDeltaReportV1 {
        analyze_intervention_target_delta(quality_plan, quality_receipt, target_plan, target_receipt).unwrap()
    }

    #[test]
    fn accepts_matching_global_and_target_delta_relations() {
        let (quality_plan, quality_receipt, target_plan, target_receipt, _) = fixtures();
        let result = report(&quality_plan, &quality_receipt, &target_plan, &target_receipt);
        assert_eq!(result.fields.candidate_count, 2);
        assert_eq!(result.fields.global_relation_valid_count, 2);
        assert_eq!(result.fields.target_delta_valid_count, 2);
        assert_eq!(result.fields.final_valid_count, 2);
        assert_eq!(result.fields.accepted_valid_count, 2);
        assert_eq!(result.fields.exact_changed_scope_match_count, 2);
        assert_eq!(result.fields.reviewer_pairwise_comparison_count, 6);
        assert_eq!(result.fields.reviewer_pairwise_agreement_ppm, Some(1_000_000));
        assert_eq!(result.fields.reviewer_target_transition_comparison_count, 6);
        assert_eq!(result.fields.reviewer_target_transition_agreement_ppm, Some(1_000_000));
        assert!(!result.fields.scoring_eligible_declared);
        assert!(!result.fields.semantic_truth_proven);
    }

    #[test]
    fn wrong_target_direction_is_invalid_even_when_global_relation_is_valid() {
        let (quality_plan, quality_receipt, target_plan, target_receipt, _) = fixtures();
        let mut receipt = parse_receipt(&target_receipt);
        for reviewer in receipt["adjudications"][1]["reviewers"].as_array_mut().unwrap() {
            reviewer["observed_target_change"] = json!("support_increased");
        }
        let altered = serde_json::to_vec(&receipt).unwrap();
        let result = report(&quality_plan, &quality_receipt, &target_plan, &altered);
        assert_eq!(result.fields.global_relation_valid_count, 2);
        assert_eq!(result.fields.target_delta_invalid_count, 1);
        assert_eq!(result.fields.final_invalid_count, 1);
    }

    #[test]
    fn collateral_change_invalidates_an_otherwise_correct_target_delta() {
        let (quality_plan, quality_receipt, target_plan, target_receipt, _) = fixtures();
        let mut receipt = parse_receipt(&target_receipt);
        for reviewer in receipt["adjudications"][1]["reviewers"].as_array_mut().unwrap() {
            reviewer["collateral_change"] = json!("material_change_outside_allowed_scope");
        }
        let altered = serde_json::to_vec(&receipt).unwrap();
        let result = report(&quality_plan, &quality_receipt, &target_plan, &altered);
        assert_eq!(result.fields.target_delta_invalid_count, 1);
        assert_eq!(result.fields.collateral_material_change_count, 1);
        assert_eq!(result.fields.final_invalid_count, 1);
    }

    #[test]
    fn ambiguous_target_panel_remains_ambiguous() {
        let (quality_plan, quality_receipt, target_plan, target_receipt, _) = fixtures();
        let mut receipt = parse_receipt(&target_receipt);
        let reviewers = receipt["adjudications"][1]["reviewers"].as_array_mut().unwrap();
        reviewers[0]["observed_target_change"] = json!("support_increased");
        reviewers[1]["observed_target_change"] = json!("contradiction_introduced");
        reviewers[2]["observed_target_change"] = json!("support_decreased");
        let altered = serde_json::to_vec(&receipt).unwrap();
        let result = report(&quality_plan, &quality_receipt, &target_plan, &altered);
        assert_eq!(result.fields.target_delta_ambiguous_count, 1);
        assert_eq!(result.fields.final_ambiguous_count, 1);
    }

    #[test]
    fn known_target_failure_is_not_hidden_by_ambiguous_global_relation() {
        let (quality_plan, quality_receipt, target_plan, target_receipt, _) = fixtures();
        let mut parent_receipt = parse_receipt(&quality_receipt);
        let parent_reviewers = parent_receipt["adjudications"][1]["reviewers"].as_array_mut().unwrap();
        parent_reviewers[0]["verdict"] = json!("meaning_changing");
        parent_reviewers[1]["verdict"] = json!("meaning_preserving");
        parent_reviewers[2]["verdict"] = json!("ambiguous");
        let parent_receipt_bytes = serde_json::to_vec(&parent_receipt).unwrap();
        // Bind target receipt to the changed exact parent-receipt bytes.
        let mut target_receipt_value = parse_receipt(&target_receipt);
        target_receipt_value["quality_receipt_sha256"] = json!(sha256_hex(&parent_receipt_bytes));
        let target_receipt_bytes = serde_json::to_vec(&target_receipt_value).unwrap();
        let mut target = parse_receipt(&target_receipt_bytes);
        for reviewer in target["adjudications"][1]["reviewers"].as_array_mut().unwrap() {
            reviewer["observed_target_change"] = json!("support_increased");
        }
        let target_receipt_bytes = serde_json::to_vec(&target).unwrap();
        let result = report(&quality_plan, &parent_receipt_bytes, &target_plan, &target_receipt_bytes);
        assert_eq!(result.fields.global_relation_ambiguous_count, 1);
        assert_eq!(result.fields.target_delta_invalid_count, 1);
        assert_eq!(result.fields.final_invalid_count, 1);
    }

    #[test]
    fn rejects_unblinded_reviewers_and_duplicate_or_missing_candidates() {
        let (quality_plan, quality_receipt, target_plan, target_receipt, _) = fixtures();
        let mut receipt = parse_receipt(&target_receipt);
        receipt["adjudications"][0]["reviewers"][0]["blind_to_expected_target_change"] = json!(false);
        let altered = serde_json::to_vec(&receipt).unwrap();
        assert!(analyze_intervention_target_delta(&quality_plan,&quality_receipt,&target_plan,&altered)
            .unwrap_err().contains("non-blinded"));

        let mut receipt = parse_receipt(&target_receipt);
        receipt["adjudications"][0]["reviewers"][0]["baseline_artifact_sha256"] =
            json!(digest("wrong reviewed baseline"));
        let altered = serde_json::to_vec(&receipt).unwrap();
        assert!(analyze_intervention_target_delta(&quality_plan,&quality_receipt,&target_plan,&altered)
            .unwrap_err().contains("reviewer target/artifact binding differs"));

        let mut receipt = parse_receipt(&target_receipt);
        receipt["adjudications"].as_array_mut().unwrap().pop();
        let altered = serde_json::to_vec(&receipt).unwrap();
        assert!(analyze_intervention_target_delta(&quality_plan,&quality_receipt,&target_plan,&altered)
            .unwrap_err().contains("exactly one adjudication"));

        let mut receipt = parse_receipt(&target_receipt);
        let first = receipt["adjudications"][0].clone();
        receipt["adjudications"].as_array_mut().unwrap().push(first);
        let altered = serde_json::to_vec(&receipt).unwrap();
        assert!(analyze_intervention_target_delta(&quality_plan,&quality_receipt,&target_plan,&altered)
            .unwrap_err().contains("duplicate target-delta adjudication"));
    }

    #[test]
    fn rejects_target_plan_scope_and_parent_binding_tampering() {
        let (quality_plan, quality_receipt, target_plan, target_receipt, changed_id) = fixtures();
        let mut target_plan_value = parse_receipt(&target_plan);
        let mut expanded_changed_scope = vec![changed_id, digest("unrelated material claim")];
        expanded_changed_scope.sort();
        target_plan_value["candidates"][1]["expected_changed_claim_ids_sha256"] =
            json!(expanded_changed_scope);
        let fields_value = {
            let mut value = target_plan_value.clone();
            value.as_object_mut().unwrap().remove("plan_commitment_sha256");
            value
        };
        let fields: TargetDeltaPlanFieldsV1 = serde_json::from_value(fields_value).unwrap();
        let fields_bytes = serde_json::to_vec(&fields).unwrap();
        let new_commitment =
            compute_target_delta_plan_commitment(&fields_bytes, &quality_plan).unwrap();
        target_plan_value["plan_commitment_sha256"] = json!(new_commitment);
        let altered_target_plan = serde_json::to_vec_pretty(&target_plan_value).unwrap();
        let mut altered_target_receipt = parse_receipt(&target_receipt);
        altered_target_receipt["target_plan_sha256"] = json!(sha256_hex(&altered_target_plan));
        altered_target_receipt["target_plan_commitment_sha256"] = json!(new_commitment);
        let altered_target_receipt = serde_json::to_vec(&altered_target_receipt).unwrap();
        let result = report(
            &quality_plan,
            &quality_receipt,
            &altered_target_plan,
            &altered_target_receipt,
        );
        assert_eq!(result.fields.target_delta_invalid_count, 1);
        assert_eq!(result.fields.final_invalid_count, 1);
        assert_eq!(result.fields.exact_changed_scope_match_count, 1);

        let mut wrong_parent = parse_receipt(&target_receipt);
        wrong_parent["quality_plan_sha256"] = json!(digest("wrong parent plan"));
        let wrong_parent = serde_json::to_vec(&wrong_parent).unwrap();
        assert!(analyze_intervention_target_delta(
            &quality_plan,
            &quality_receipt,
            &target_plan,
            &wrong_parent
        )
        .unwrap_err()
        .contains("exact target and parent quality artifacts"));
    }

    #[test]
    fn rejects_a_target_delta_plan_bound_to_another_parent_study() {
        let (quality_plan, _, target_plan, _, _) = fixtures();
        let mut plan = parse_receipt(&target_plan);
        plan["study_id"] = json!("different-study");
        let fields_value = {
            let mut value = plan.clone();
            value.as_object_mut().unwrap().remove("plan_commitment_sha256");
            value
        };
        let fields: TargetDeltaPlanFieldsV1 = serde_json::from_value(fields_value).unwrap();
        let issues = validate_target_delta_plan_fields(&fields, &quality_plan).unwrap_err();
        assert!(issues.iter().any(|issue| issue.contains("study_id differs")));
    }

    #[test]
    fn report_is_deterministic_and_claims_remain_bounded() {
        let (quality_plan, quality_receipt, target_plan, target_receipt, _) = fixtures();
        let first = serde_json::to_vec(&report(&quality_plan,&quality_receipt,&target_plan,&target_receipt)).unwrap();
        let second = serde_json::to_vec(&report(&quality_plan,&quality_receipt,&target_plan,&target_receipt)).unwrap();
        assert_eq!(first, second);
        let result: Value = serde_json::from_slice(&first).unwrap();
        assert_eq!(result["reviewer_independence_proven"], false);
        assert_eq!(result["reviewer_competence_proven"], false);
        assert_eq!(result["semantic_truth_proven"], false);
        assert_eq!(result["execution_authenticity_proven"], false);
        assert_eq!(result["external_anchor_required"], true);
        assert_eq!(result["scoring_eligible_declared"], false);
    }
}
