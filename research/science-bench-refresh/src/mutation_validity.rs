//! Independent semantic qualification and verified failure-yield accounting.
//!
//! This module audits supplied mutation plans, independent semantic-review
//! votes and evaluator outcomes. It does not authenticate reviewers, prove
//! reviewer blinding, custody raw artifact bytes, or establish population
//! validity. Evaluator failures count as verified only when the mutation is
//! unanimously confirmed against its preregistered semantic relation.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const MUTATION_PLAN_SCHEMA: &str = "symthaea.science_bench_refresh_mutation_validity_plan.v1";
pub const MUTATION_RECEIPT_SCHEMA: &str =
    "symthaea.science_bench_refresh_mutation_validity_receipt.v1";
pub const MUTATION_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_mutation_validity_report.v1";

const PLAN_DOMAIN: &[u8] = b"symthaea.mutation_validity.plan.v1\0";
const REPORT_DOMAIN: &[u8] = b"symthaea.mutation_validity.report.v1\0";
const MAX_VARIANTS: usize = 100_000;
const MAX_REVIEWERS: usize = 7;
const MAX_ARTIFACTS: usize = 32;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MutationPurposeV1 {
    ExampleOnly,
    QualifiedRun,
}

impl MutationPurposeV1 {
    fn qualified(self) -> bool {
        self == Self::QualifiedRun
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MutationSemanticRelationV1 {
    Preserving,
    Changing,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MutationFamilyV1 {
    Paraphrase,
    Formatting,
    EntitySubstitution,
    UnitChange,
    DateChange,
    ScopeChange,
    EvidenceAddition,
    EvidenceWithdrawal,
    SourceRetraction,
    MultimodalComposition,
    Other,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MutationModalityV1 {
    Text,
    Image,
    Audio,
    Video,
    Structured,
    Multimodal,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MutationArtifactCommitmentV1 {
    pub artifact_id: String,
    pub modality: MutationModalityV1,
    pub content_sha256: String,
    pub byte_length: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MutationVariantPlanV1 {
    pub mutation_id: String,
    pub metamorphic_pair_id: String,
    pub family: MutationFamilyV1,
    pub intended_relation: MutationSemanticRelationV1,
    pub target_dimension: Option<String>,
    pub baseline_input_sha256: String,
    pub variant_input_sha256: String,
    pub baseline_artifacts: Vec<MutationArtifactCommitmentV1>,
    pub variant_artifacts: Vec<MutationArtifactCommitmentV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MutationValidityPlanFieldsV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: MutationPurposeV1,
    pub study_id: String,
    pub generator_source_revision: String,
    pub evaluator_source_revision: String,
    pub generator_policy_sha256: String,
    pub semantic_task_set_sha256: String,
    pub review_protocol_sha256: String,
    pub metamorphic_plan_sha256: String,
    pub evaluator_prompt_sha256: String,
    pub rubric_sha256: String,
    pub reviewers_per_mutation: u32,
    pub variants: Vec<MutationVariantPlanV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MutationValidityPlanV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: MutationPurposeV1,
    pub study_id: String,
    pub generator_source_revision: String,
    pub evaluator_source_revision: String,
    pub generator_policy_sha256: String,
    pub semantic_task_set_sha256: String,
    pub review_protocol_sha256: String,
    pub metamorphic_plan_sha256: String,
    pub evaluator_prompt_sha256: String,
    pub rubric_sha256: String,
    pub reviewers_per_mutation: u32,
    pub variants: Vec<MutationVariantPlanV1>,
    pub plan_commitment_sha256: String,
}

impl MutationValidityPlanV1 {
    fn fields(&self) -> MutationValidityPlanFieldsV1 {
        MutationValidityPlanFieldsV1 {
            schema: self.schema.clone(),
            analyzer_source_revision: self.analyzer_source_revision.clone(),
            purpose: self.purpose,
            study_id: self.study_id.clone(),
            generator_source_revision: self.generator_source_revision.clone(),
            evaluator_source_revision: self.evaluator_source_revision.clone(),
            generator_policy_sha256: self.generator_policy_sha256.clone(),
            semantic_task_set_sha256: self.semantic_task_set_sha256.clone(),
            review_protocol_sha256: self.review_protocol_sha256.clone(),
            metamorphic_plan_sha256: self.metamorphic_plan_sha256.clone(),
            evaluator_prompt_sha256: self.evaluator_prompt_sha256.clone(),
            rubric_sha256: self.rubric_sha256.clone(),
            reviewers_per_mutation: self.reviewers_per_mutation,
            variants: self.variants.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SemanticReviewDecisionV1 {
    PreservesSemantics,
    ChangesSemantics,
    Indeterminate,
    Abstain,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SemanticAdjudicationV1 {
    pub adjudicator_id_sha256: String,
    pub independence_group_sha256: String,
    pub decision: SemanticReviewDecisionV1,
    pub confidence_millionths: u32,
    pub rationale_sha256: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MutationEvaluationStatusV1 {
    Completed,
    TimedOut,
    Blocked,
    InvalidReceipt,
    NotStarted,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MutationValidityObservationV1 {
    pub mutation_id: String,
    pub baseline_input_sha256: String,
    pub variant_input_sha256: String,
    pub adjudications: Vec<SemanticAdjudicationV1>,
    pub evaluation_status: MutationEvaluationStatusV1,
    pub evaluator_relation_pass: Option<bool>,
    pub failure_fingerprint_sha256: Option<String>,
    pub generator_call_count: u64,
    pub generator_token_count: u64,
    pub semantic_assistance_call_count: u64,
    pub semantic_assistance_token_count: u64,
    pub evaluator_call_count: u64,
    pub evaluator_token_count: u64,
    pub human_review_seconds: u64,
    pub wall_time_milliseconds: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MutationValidityReceiptV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: MutationPurposeV1,
    pub run_id: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub metamorphic_receipt_sha256: String,
    pub generator_source_revision: String,
    pub evaluator_source_revision: String,
    pub evaluator_prompt_sha256: String,
    pub rubric_sha256: String,
    pub observations: Vec<MutationValidityObservationV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DerivedReviewStatus {
    ConfirmedValidPreserving,
    ConfirmedValidChanging,
    ConfirmedInvalid,
    Disputed,
    Insufficient,
    Abstained,
    Unreviewed,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MutationValiditySummaryV1 {
    pub mutation_id: String,
    pub intended_relation: MutationSemanticRelationV1,
    pub review_count: u64,
    pub pairwise_agreement_count: u64,
    pub pairwise_disagreement_count: u64,
    pub review_status: String,
    pub evaluation_status: MutationEvaluationStatusV1,
    pub evaluator_relation_pass: Option<bool>,
    pub failure_counted_as_verified: bool,
    pub failure_exclusion_reason: Option<String>,
    pub failure_fingerprint_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MutationValidityReportV1 {
    pub report_commitment_sha256: String,
    pub schema: String,
    pub study_id: String,
    pub analyzer_source_revision: String,
    pub analyzer_worktree_clean: bool,
    pub plan_sha256: String,
    pub receipt_sha256: String,
    pub plan_commitment_sha256: String,
    pub metamorphic_receipt_sha256: String,
    pub planned_variant_count: u64,
    pub observed_variant_count: u64,
    pub completed_evaluation_count: u64,
    pub ineligible_evaluation_count: u64,
    pub planned_adjudication_slots: u64,
    pub observed_adjudication_count: u64,
    pub missing_adjudication_slots: u64,
    pub confirmed_valid_preserving_count: u64,
    pub confirmed_valid_changing_count: u64,
    pub confirmed_invalid_count: u64,
    pub disputed_count: u64,
    pub insufficient_count: u64,
    pub abstained_count: u64,
    pub unreviewed_count: u64,
    pub adjudicable_confirmed_count: u64,
    pub adjudicable_panel_concordance_millionths: Option<f64>,
    pub confirmed_valid_yield_over_planned_millionths: Option<f64>,
    pub pairwise_review_agreement_count: u64,
    pub pairwise_review_disagreement_count: u64,
    pub evaluator_failure_observation_count: u64,
    pub verified_failure_observation_count: u64,
    pub excluded_invalid_mutation_failure_count: u64,
    pub excluded_unconfirmed_mutation_failure_count: u64,
    pub unique_verified_failure_count: u64,
    pub duplicate_verified_failure_observation_count: u64,
    pub total_generator_call_count: u64,
    pub total_generator_token_count: u64,
    pub total_semantic_assistance_call_count: u64,
    pub total_semantic_assistance_token_count: u64,
    pub total_evaluator_call_count: u64,
    pub total_evaluator_token_count: u64,
    pub total_human_review_seconds: u64,
    pub total_wall_time_milliseconds: u64,
    pub total_token_count: u64,
    pub tokens_per_confirmed_valid_mutation: Option<f64>,
    pub tokens_per_unique_verified_failure: Option<f64>,
    pub human_review_seconds_per_unique_verified_failure: Option<f64>,
    pub variant_summaries: Vec<MutationValiditySummaryV1>,
    pub raw_artifact_custody_proven: bool,
    pub reviewer_identity_authenticated: bool,
    pub reviewer_blinding_proven: bool,
    pub panel_reference_validity_proven: bool,
    pub evaluator_execution_proven: bool,
    pub causal_effect_identified: bool,
    pub generalized_validity_estimated: bool,
    pub external_anchor_required: bool,
    pub scoring_eligible_declared: bool,
    pub claim_boundary: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest_domain(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let length = u64::try_from(payload.len())
        .map_err(|_| "domain-separated payload exceeds u64 length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(length.to_be_bytes());
    hasher.update(payload);
    Ok(sha256_hex(&hasher.finalize()))
}

fn valid_digest(value: &str, allow_zero: bool) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && (allow_zero || value.bytes().any(|byte| byte != b'0'))
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn nonempty(value: &str, max: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max
}

fn checked_add(total: &mut u64, value: u64, label: &str) -> Result<(), String> {
    *total = total
        .checked_add(value)
        .ok_or_else(|| format!("{label} counter overflow"))?;
    Ok(())
}

fn plan_commitment(fields: &MutationValidityPlanFieldsV1) -> Result<String, String> {
    let payload = serde_json::to_vec(fields)
        .map_err(|error| format!("cannot serialize mutation validity plan: {error}"))?;
    digest_domain(PLAN_DOMAIN, &payload)
}

fn validate_artifacts(
    mutation_id: &str,
    label: &str,
    artifacts: &[MutationArtifactCommitmentV1],
    allow_zero: bool,
) -> Vec<String> {
    let mut issues = Vec::new();
    if artifacts.is_empty() || artifacts.len() > MAX_ARTIFACTS {
        issues.push(format!(
            "{mutation_id} {label} must contain 1..={MAX_ARTIFACTS} artifact commitments"
        ));
    }
    let mut ids = BTreeSet::new();
    for artifact in artifacts {
        if !nonempty(&artifact.artifact_id, 128) || !ids.insert(artifact.artifact_id.as_str()) {
            issues.push(format!(
                "{mutation_id} {label} contains an empty or duplicate artifact ID"
            ));
        }
        if !valid_digest(&artifact.content_sha256, allow_zero) {
            issues.push(format!(
                "{mutation_id} {label} contains an invalid artifact content digest"
            ));
        }
        if artifact.byte_length == 0 {
            issues.push(format!(
                "{mutation_id} {label} contains a zero-byte artifact"
            ));
        }
    }
    issues
}

pub fn validate_mutation_validity_plan_fields(
    fields: &MutationValidityPlanFieldsV1,
) -> Result<(), Vec<String>> {
    let mut issues = Vec::new();
    let allow_zero = !fields.purpose.qualified();

    if fields.schema != MUTATION_PLAN_SCHEMA {
        issues.push("unsupported mutation validity plan schema".to_owned());
    }
    if !valid_revision(&fields.analyzer_source_revision)
        || fields.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        issues.push(
            "analyzer_source_revision must match the exact compiled validator revision".to_owned(),
        );
    }
    if fields.purpose.qualified() && crate::BUILT_GENERATOR_WORKTREE_CLEAN != "true" {
        issues
            .push("qualified mutation validity plans require a clean analyzer worktree".to_owned());
    }
    if !nonempty(&fields.study_id, 128) {
        issues.push("study_id must be non-empty and at most 128 bytes".to_owned());
    }
    for (label, value) in [
        (
            "generator_source_revision",
            &fields.generator_source_revision,
        ),
        (
            "evaluator_source_revision",
            &fields.evaluator_source_revision,
        ),
    ] {
        if !valid_revision(value) {
            issues.push(format!(
                "{label} must be a full nonzero lowercase Git revision"
            ));
        }
    }
    for (label, value) in [
        ("generator_policy_sha256", &fields.generator_policy_sha256),
        ("semantic_task_set_sha256", &fields.semantic_task_set_sha256),
        ("review_protocol_sha256", &fields.review_protocol_sha256),
        ("metamorphic_plan_sha256", &fields.metamorphic_plan_sha256),
        ("evaluator_prompt_sha256", &fields.evaluator_prompt_sha256),
        ("rubric_sha256", &fields.rubric_sha256),
    ] {
        if !valid_digest(value, allow_zero) {
            issues.push(format!("{label} must be a valid SHA-256 digest"));
        }
    }
    if !(2..=MAX_REVIEWERS as u32).contains(&fields.reviewers_per_mutation) {
        issues.push(format!(
            "reviewers_per_mutation must be in 2..={MAX_REVIEWERS}"
        ));
    }
    if fields.variants.is_empty() || fields.variants.len() > MAX_VARIANTS {
        issues.push(format!("variants must contain 1..={MAX_VARIANTS} entries"));
    }

    let mut mutation_ids = BTreeSet::new();
    let mut pair_ids = BTreeSet::new();
    for variant in &fields.variants {
        if !nonempty(&variant.mutation_id, 128)
            || !mutation_ids.insert(variant.mutation_id.as_str())
        {
            issues.push("mutation IDs must be non-empty and unique".to_owned());
        }
        if !nonempty(&variant.metamorphic_pair_id, 128)
            || !pair_ids.insert(variant.metamorphic_pair_id.as_str())
        {
            issues.push(format!(
                "{} has an empty or duplicate metamorphic_pair_id",
                variant.mutation_id
            ));
        }
        if !valid_digest(&variant.baseline_input_sha256, allow_zero)
            || !valid_digest(&variant.variant_input_sha256, allow_zero)
        {
            issues.push(format!(
                "{} has an invalid baseline/variant input digest",
                variant.mutation_id
            ));
        }
        if variant.baseline_input_sha256 == variant.variant_input_sha256 {
            issues.push(format!(
                "{} baseline and variant input digests must differ",
                variant.mutation_id
            ));
        }
        if variant.intended_relation == MutationSemanticRelationV1::Changing
            && !variant
                .target_dimension
                .as_deref()
                .is_some_and(|value| nonempty(value, 128))
        {
            issues.push(format!(
                "{} meaning-changing mutation requires a target_dimension",
                variant.mutation_id
            ));
        }
        if variant.intended_relation == MutationSemanticRelationV1::Preserving
            && variant.target_dimension.is_some()
        {
            issues.push(format!(
                "{} semantics-preserving mutation must not declare a changed target_dimension",
                variant.mutation_id
            ));
        }
        issues.extend(validate_artifacts(
            &variant.mutation_id,
            "baseline_artifacts",
            &variant.baseline_artifacts,
            allow_zero,
        ));
        issues.extend(validate_artifacts(
            &variant.mutation_id,
            "variant_artifacts",
            &variant.variant_artifacts,
            allow_zero,
        ));
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn compute_mutation_validity_plan_commitment(fields_bytes: &[u8]) -> Result<String, String> {
    let fields: MutationValidityPlanFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("mutation validity plan fields invalid: {error}"))?;
    validate_mutation_validity_plan_fields(&fields).map_err(|issues| issues.join("; "))?;
    plan_commitment(&fields)
}

pub fn verify_mutation_validity_plan(
    plan_bytes: &[u8],
) -> Result<(MutationValidityPlanV1, String), String> {
    let plan: MutationValidityPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("mutation validity plan invalid: {error}"))?;
    let fields = plan.fields();
    validate_mutation_validity_plan_fields(&fields).map_err(|issues| issues.join("; "))?;
    let expected = plan_commitment(&fields)?;
    if !valid_digest(&plan.plan_commitment_sha256, !plan.purpose.qualified())
        || plan.plan_commitment_sha256 != expected
    {
        return Err("mutation validity plan commitment mismatch".to_owned());
    }
    Ok((plan, sha256_hex(plan_bytes)))
}

pub fn verify_mutation_validity_report(
    report_bytes: &[u8],
) -> Result<MutationValidityReportV1, String> {
    let mut report: MutationValidityReportV1 = serde_json::from_slice(report_bytes)
        .map_err(|error| format!("mutation validity report invalid: {error}"))?;
    let supplied = report.report_commitment_sha256.clone();
    if !valid_digest(&supplied, false) {
        return Err(
            "mutation validity report commitment must be a nonzero SHA-256 digest".to_owned(),
        );
    }
    report.report_commitment_sha256.clear();
    let payload = serde_json::to_vec(&report)
        .map_err(|error| format!("cannot serialize mutation validity report: {error}"))?;
    let expected = digest_domain(REPORT_DOMAIN, &payload)?;
    if expected != supplied {
        return Err("mutation validity report commitment mismatch".to_owned());
    }
    report.report_commitment_sha256 = supplied;
    Ok(report)
}

fn derive_review_status(
    intended: MutationSemanticRelationV1,
    ratings: &[SemanticAdjudicationV1],
    required: u32,
) -> DerivedReviewStatus {
    if ratings.is_empty() {
        return DerivedReviewStatus::Unreviewed;
    }
    if ratings.len() < required as usize {
        return DerivedReviewStatus::Insufficient;
    }
    let labels: BTreeSet<SemanticReviewDecisionV1> =
        ratings.iter().map(|rating| rating.decision).collect();
    let expected = match intended {
        MutationSemanticRelationV1::Preserving => SemanticReviewDecisionV1::PreservesSemantics,
        MutationSemanticRelationV1::Changing => SemanticReviewDecisionV1::ChangesSemantics,
    };
    let opposite = match intended {
        MutationSemanticRelationV1::Preserving => SemanticReviewDecisionV1::ChangesSemantics,
        MutationSemanticRelationV1::Changing => SemanticReviewDecisionV1::PreservesSemantics,
    };
    if labels.len() == 1 && labels.contains(&expected) {
        return match intended {
            MutationSemanticRelationV1::Preserving => DerivedReviewStatus::ConfirmedValidPreserving,
            MutationSemanticRelationV1::Changing => DerivedReviewStatus::ConfirmedValidChanging,
        };
    }
    if labels.len() == 1 && labels.contains(&opposite) {
        return DerivedReviewStatus::ConfirmedInvalid;
    }
    if labels.contains(&SemanticReviewDecisionV1::PreservesSemantics)
        && labels.contains(&SemanticReviewDecisionV1::ChangesSemantics)
    {
        return DerivedReviewStatus::Disputed;
    }
    if labels == BTreeSet::from([SemanticReviewDecisionV1::Abstain]) {
        return DerivedReviewStatus::Abstained;
    }
    DerivedReviewStatus::Insufficient
}

fn review_status_name(status: DerivedReviewStatus) -> &'static str {
    match status {
        DerivedReviewStatus::ConfirmedValidPreserving => "confirmed_valid_preserving",
        DerivedReviewStatus::ConfirmedValidChanging => "confirmed_valid_changing",
        DerivedReviewStatus::ConfirmedInvalid => "confirmed_invalid",
        DerivedReviewStatus::Disputed => "disputed",
        DerivedReviewStatus::Insufficient => "insufficient",
        DerivedReviewStatus::Abstained => "abstained",
        DerivedReviewStatus::Unreviewed => "unreviewed",
    }
}

fn add_total(
    totals: &mut BTreeMap<&'static str, u64>,
    key: &'static str,
    value: u64,
) -> Result<(), String> {
    let counter = totals.entry(key).or_default();
    checked_add(counter, value, key)
}

pub fn analyze_mutation_validity(
    plan_bytes: &[u8],
    receipt_bytes: &[u8],
) -> Result<MutationValidityReportV1, String> {
    let (plan, plan_sha256) = verify_mutation_validity_plan(plan_bytes)?;
    let receipt: MutationValidityReceiptV1 = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("mutation validity receipt invalid: {error}"))?;
    if receipt.schema != MUTATION_RECEIPT_SCHEMA {
        return Err("unsupported mutation validity receipt schema".to_owned());
    }
    if receipt.analyzer_source_revision != plan.analyzer_source_revision
        || receipt.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        return Err(
            "receipt analyzer revision differs from the exact compiled validator revision"
                .to_owned(),
        );
    }
    if receipt.purpose != plan.purpose {
        return Err("receipt purpose differs from the plan".to_owned());
    }
    if !nonempty(&receipt.run_id, 128) {
        return Err("run_id must be non-empty and at most 128 bytes".to_owned());
    }
    if !valid_digest(
        &receipt.metamorphic_receipt_sha256,
        !plan.purpose.qualified(),
    ) {
        return Err("metamorphic_receipt_sha256 must be a valid digest".to_owned());
    }
    if receipt.plan_sha256 != plan_sha256
        || receipt.plan_commitment_sha256 != plan.plan_commitment_sha256
    {
        return Err("receipt does not bind exact plan bytes and commitment".to_owned());
    }
    if receipt.generator_source_revision != plan.generator_source_revision
        || receipt.evaluator_source_revision != plan.evaluator_source_revision
        || receipt.evaluator_prompt_sha256 != plan.evaluator_prompt_sha256
        || receipt.rubric_sha256 != plan.rubric_sha256
    {
        return Err(
            "receipt generator/evaluator/prompt/rubric differs from frozen plan".to_owned(),
        );
    }

    let planned: BTreeMap<&str, &MutationVariantPlanV1> = plan
        .variants
        .iter()
        .map(|variant| (variant.mutation_id.as_str(), variant))
        .collect();
    let mut observed = BTreeMap::new();
    for observation in &receipt.observations {
        if observed
            .insert(observation.mutation_id.as_str(), observation)
            .is_some()
        {
            return Err(format!(
                "duplicate mutation observation {}",
                observation.mutation_id
            ));
        }
        let variant = planned
            .get(observation.mutation_id.as_str())
            .ok_or_else(|| {
                format!(
                    "unexpected mutation observation {}",
                    observation.mutation_id
                )
            })?;
        if observation.baseline_input_sha256 != variant.baseline_input_sha256
            || observation.variant_input_sha256 != variant.variant_input_sha256
        {
            return Err(format!(
                "mutation {} input digest differs from committed plan",
                observation.mutation_id
            ));
        }
        if observation.adjudications.len() > plan.reviewers_per_mutation as usize {
            return Err(format!(
                "mutation {} has more adjudications than planned reviewer slots",
                observation.mutation_id
            ));
        }
        let mut reviewer_ids = BTreeSet::new();
        let mut independence_groups = BTreeSet::new();
        for adjudication in &observation.adjudications {
            for (label, digest) in [
                ("adjudicator_id_sha256", &adjudication.adjudicator_id_sha256),
                (
                    "independence_group_sha256",
                    &adjudication.independence_group_sha256,
                ),
                ("rationale_sha256", &adjudication.rationale_sha256),
            ] {
                if !valid_digest(digest, false) {
                    return Err(format!(
                        "mutation {} has invalid {label}",
                        observation.mutation_id
                    ));
                }
            }
            if !reviewer_ids.insert(adjudication.adjudicator_id_sha256.as_str()) {
                return Err(format!(
                    "mutation {} reuses an adjudicator identity",
                    observation.mutation_id
                ));
            }
            if !independence_groups.insert(adjudication.independence_group_sha256.as_str()) {
                return Err(format!(
                    "mutation {} reuses an adjudicator independence group",
                    observation.mutation_id
                ));
            }
            if adjudication.confidence_millionths > 1_000_000 {
                return Err(format!(
                    "mutation {} has out-of-range adjudicator confidence",
                    observation.mutation_id
                ));
            }
        }
        match observation.evaluation_status {
            MutationEvaluationStatusV1::Completed => {
                let passed = observation.evaluator_relation_pass.ok_or_else(|| {
                    format!(
                        "completed mutation {} requires evaluator_relation_pass",
                        observation.mutation_id
                    )
                })?;
                match (passed, observation.failure_fingerprint_sha256.as_deref()) {
                    (true, None) => {}
                    (false, Some(digest)) if valid_digest(digest, false) => {}
                    (false, _) => {
                        return Err(format!(
                            "failed mutation {} requires a valid failure fingerprint",
                            observation.mutation_id
                        ));
                    }
                    (true, Some(_)) => {
                        return Err(format!(
                            "passing mutation {} must not carry a failure fingerprint",
                            observation.mutation_id
                        ));
                    }
                }
            }
            _ => {
                if observation.evaluator_relation_pass.is_some()
                    || observation.failure_fingerprint_sha256.is_some()
                {
                    return Err(format!(
                        "ineligible mutation {} must not carry an evaluator outcome",
                        observation.mutation_id
                    ));
                }
            }
        }
    }
    let missing: Vec<&str> = planned
        .keys()
        .filter(|id| !observed.contains_key(**id))
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "receipt is missing {} planned mutation observations",
            missing.len()
        ));
    }

    let mut totals = BTreeMap::new();
    let mut summaries = Vec::with_capacity(plan.variants.len());
    let mut confirmed_valid_preserving = 0_u64;
    let mut confirmed_valid_changing = 0_u64;
    let mut confirmed_invalid = 0_u64;
    let mut disputed = 0_u64;
    let mut insufficient = 0_u64;
    let mut abstained = 0_u64;
    let mut unreviewed = 0_u64;
    let mut completed = 0_u64;
    let mut ineligible = 0_u64;
    let mut observed_adjudications = 0_u64;
    let mut pairwise_agreements = 0_u64;
    let mut pairwise_disagreements = 0_u64;
    let mut failure_observations = 0_u64;
    let mut verified_failure_observations = 0_u64;
    let mut excluded_invalid_failures = 0_u64;
    let mut excluded_unconfirmed_failures = 0_u64;
    let mut verified_fingerprints = BTreeSet::new();
    let mut duplicate_verified_failures = 0_u64;

    for variant in &plan.variants {
        let observation = observed.get(variant.mutation_id.as_str()).unwrap();
        let status = derive_review_status(
            variant.intended_relation,
            &observation.adjudications,
            plan.reviewers_per_mutation,
        );
        checked_add(
            &mut observed_adjudications,
            observation.adjudications.len() as u64,
            "observed adjudication",
        )?;
        let pair_count = (observation.adjudications.len() as u64)
            .checked_mul(observation.adjudications.len().saturating_sub(1) as u64)
            .map(|value| value / 2)
            .ok_or_else(|| "reviewer pair count overflow".to_owned())?;
        let mut agreements = 0_u64;
        let mut disagreements = 0_u64;
        for left in 0..observation.adjudications.len() {
            for right in (left + 1)..observation.adjudications.len() {
                if observation.adjudications[left].decision
                    == observation.adjudications[right].decision
                {
                    agreements += 1;
                } else {
                    disagreements += 1;
                }
            }
        }
        debug_assert_eq!(agreements + disagreements, pair_count);
        checked_add(
            &mut pairwise_agreements,
            agreements,
            "pairwise review agreement",
        )?;
        checked_add(
            &mut pairwise_disagreements,
            disagreements,
            "pairwise review disagreement",
        )?;

        match status {
            DerivedReviewStatus::ConfirmedValidPreserving => confirmed_valid_preserving += 1,
            DerivedReviewStatus::ConfirmedValidChanging => confirmed_valid_changing += 1,
            DerivedReviewStatus::ConfirmedInvalid => confirmed_invalid += 1,
            DerivedReviewStatus::Disputed => disputed += 1,
            DerivedReviewStatus::Insufficient => insufficient += 1,
            DerivedReviewStatus::Abstained => abstained += 1,
            DerivedReviewStatus::Unreviewed => unreviewed += 1,
        }
        let valid_mutation = matches!(
            status,
            DerivedReviewStatus::ConfirmedValidPreserving
                | DerivedReviewStatus::ConfirmedValidChanging
        );
        let is_invalid = status == DerivedReviewStatus::ConfirmedInvalid;
        let mut failure_counted = false;
        let mut exclusion_reason = None;
        if observation.evaluation_status == MutationEvaluationStatusV1::Completed {
            completed += 1;
            if observation.evaluator_relation_pass == Some(false) {
                failure_observations += 1;
                let fingerprint = observation.failure_fingerprint_sha256.as_ref().unwrap();
                if valid_mutation {
                    verified_failure_observations += 1;
                    failure_counted = true;
                    if !verified_fingerprints.insert(fingerprint.clone()) {
                        duplicate_verified_failures += 1;
                    }
                } else if is_invalid {
                    excluded_invalid_failures += 1;
                    exclusion_reason = Some("mutation_confirmed_invalid".to_owned());
                } else {
                    excluded_unconfirmed_failures += 1;
                    exclusion_reason = Some(review_status_name(status).to_owned());
                }
            }
        } else {
            ineligible += 1;
        }

        let summary = MutationValiditySummaryV1 {
            mutation_id: variant.mutation_id.clone(),
            intended_relation: variant.intended_relation,
            review_count: observation.adjudications.len() as u64,
            pairwise_agreement_count: agreements,
            pairwise_disagreement_count: disagreements,
            review_status: review_status_name(status).to_owned(),
            evaluation_status: observation.evaluation_status,
            evaluator_relation_pass: observation.evaluator_relation_pass,
            failure_counted_as_verified: failure_counted,
            failure_exclusion_reason: exclusion_reason,
            failure_fingerprint_sha256: if failure_counted {
                observation.failure_fingerprint_sha256.clone()
            } else {
                None
            },
        };
        summaries.push(summary);

        for (key, value) in [
            ("generator_calls", observation.generator_call_count),
            ("generator_tokens", observation.generator_token_count),
            (
                "semantic_assistance_calls",
                observation.semantic_assistance_call_count,
            ),
            (
                "semantic_assistance_tokens",
                observation.semantic_assistance_token_count,
            ),
            ("evaluator_calls", observation.evaluator_call_count),
            ("evaluator_tokens", observation.evaluator_token_count),
            ("human_review_seconds", observation.human_review_seconds),
            ("wall_time_milliseconds", observation.wall_time_milliseconds),
        ] {
            add_total(&mut totals, key, value)?;
        }
    }

    let planned_count = plan.variants.len() as u64;
    let planned_slots = planned_count
        .checked_mul(u64::from(plan.reviewers_per_mutation))
        .ok_or_else(|| "planned adjudication slot count overflow".to_owned())?;
    let missing_slots = planned_slots.saturating_sub(observed_adjudications);
    let adjudicable = confirmed_valid_preserving + confirmed_valid_changing + confirmed_invalid;
    let confirmed_valid = confirmed_valid_preserving + confirmed_valid_changing;
    let adjudicable_concordance = if adjudicable == 0 {
        None
    } else {
        Some(confirmed_valid as f64 / adjudicable as f64 * 1_000_000.0)
    };
    let valid_yield = Some(confirmed_valid as f64 / planned_count as f64 * 1_000_000.0);
    let total_tokens = totals
        .get("generator_tokens")
        .copied()
        .unwrap_or(0)
        .checked_add(
            totals
                .get("semantic_assistance_tokens")
                .copied()
                .unwrap_or(0),
        )
        .and_then(|value| value.checked_add(totals.get("evaluator_tokens").copied().unwrap_or(0)))
        .ok_or_else(|| "total token count overflow".to_owned())?;
    let unique_verified = verified_fingerprints.len() as u64;

    let fields = MutationValidityReportV1 {
        report_commitment_sha256: String::new(),
        schema: MUTATION_REPORT_SCHEMA.to_owned(),
        study_id: plan.study_id,
        analyzer_source_revision: plan.analyzer_source_revision,
        analyzer_worktree_clean: crate::BUILT_GENERATOR_WORKTREE_CLEAN == "true",
        plan_sha256,
        receipt_sha256: sha256_hex(receipt_bytes),
        plan_commitment_sha256: plan.plan_commitment_sha256,
        metamorphic_receipt_sha256: receipt.metamorphic_receipt_sha256,
        planned_variant_count: planned_count,
        observed_variant_count: receipt.observations.len() as u64,
        completed_evaluation_count: completed,
        ineligible_evaluation_count: ineligible,
        planned_adjudication_slots: planned_slots,
        observed_adjudication_count: observed_adjudications,
        missing_adjudication_slots: missing_slots,
        confirmed_valid_preserving_count: confirmed_valid_preserving,
        confirmed_valid_changing_count: confirmed_valid_changing,
        confirmed_invalid_count: confirmed_invalid,
        disputed_count: disputed,
        insufficient_count: insufficient,
        abstained_count: abstained,
        unreviewed_count: unreviewed,
        adjudicable_confirmed_count: adjudicable,
        adjudicable_panel_concordance_millionths: adjudicable_concordance,
        confirmed_valid_yield_over_planned_millionths: valid_yield,
        pairwise_review_agreement_count: pairwise_agreements,
        pairwise_review_disagreement_count: pairwise_disagreements,
        evaluator_failure_observation_count: failure_observations,
        verified_failure_observation_count: verified_failure_observations,
        excluded_invalid_mutation_failure_count: excluded_invalid_failures,
        excluded_unconfirmed_mutation_failure_count: excluded_unconfirmed_failures,
        unique_verified_failure_count: unique_verified,
        duplicate_verified_failure_observation_count: duplicate_verified_failures,
        total_generator_call_count: totals.get("generator_calls").copied().unwrap_or(0),
        total_generator_token_count: totals.get("generator_tokens").copied().unwrap_or(0),
        total_semantic_assistance_call_count: totals.get("semantic_assistance_calls").copied().unwrap_or(0),
        total_semantic_assistance_token_count: totals.get("semantic_assistance_tokens").copied().unwrap_or(0),
        total_evaluator_call_count: totals.get("evaluator_calls").copied().unwrap_or(0),
        total_evaluator_token_count: totals.get("evaluator_tokens").copied().unwrap_or(0),
        total_human_review_seconds: totals.get("human_review_seconds").copied().unwrap_or(0),
        total_wall_time_milliseconds: totals.get("wall_time_milliseconds").copied().unwrap_or(0),
        total_token_count: total_tokens,
        tokens_per_confirmed_valid_mutation: if confirmed_valid == 0 { None } else { Some(total_tokens as f64 / confirmed_valid as f64) },
        tokens_per_unique_verified_failure: if unique_verified == 0 { None } else { Some(total_tokens as f64 / unique_verified as f64) },
        human_review_seconds_per_unique_verified_failure: if unique_verified == 0 { None } else { Some(totals.get("human_review_seconds").copied().unwrap_or(0) as f64 / unique_verified as f64) },
        variant_summaries: summaries,
        raw_artifact_custody_proven: false,
        reviewer_identity_authenticated: false,
        reviewer_blinding_proven: false,
        panel_reference_validity_proven: false,
        evaluator_execution_proven: false,
        causal_effect_identified: false,
        generalized_validity_estimated: false,
        external_anchor_required: true,
        scoring_eligible_declared: false,
        claim_boundary: "The report validates declared artifact commitments, independent reviewer commitments, adjudication outcomes, evaluator relation outcomes and resource counters for a supplied fixed receipt. It does not custody raw artifact bytes, authenticate reviewers, prove blinding, establish that the panel is a valid reference, authenticate evaluator execution, identify causal effects, estimate generalized validity, or grant scoring eligibility.".to_owned(),
    };
    let payload = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize mutation validity report: {error}"))?;
    let commitment = digest_domain(REPORT_DOMAIN, &payload)?;
    let mut report = fields;
    report.report_commitment_sha256 = commitment;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn artifact(label: &str, modality: MutationModalityV1) -> MutationArtifactCommitmentV1 {
        MutationArtifactCommitmentV1 {
            artifact_id: label.to_owned(),
            modality,
            content_sha256: digest(&format!("artifact:{label}")),
            byte_length: 32,
        }
    }

    fn variant(id: &str, relation: MutationSemanticRelationV1) -> MutationVariantPlanV1 {
        MutationVariantPlanV1 {
            mutation_id: id.to_owned(),
            metamorphic_pair_id: format!("pair-{id}"),
            family: MutationFamilyV1::Paraphrase,
            intended_relation: relation,
            target_dimension: if relation == MutationSemanticRelationV1::Changing {
                Some("measurement_unit".to_owned())
            } else {
                None
            },
            baseline_input_sha256: digest(&format!("{id}:baseline")),
            variant_input_sha256: digest(&format!("{id}:variant")),
            baseline_artifacts: vec![artifact(&format!("{id}-base"), MutationModalityV1::Text)],
            variant_artifacts: vec![artifact(&format!("{id}-variant"), MutationModalityV1::Text)],
        }
    }

    fn rating(
        id: &str,
        index: usize,
        decision: SemanticReviewDecisionV1,
    ) -> SemanticAdjudicationV1 {
        SemanticAdjudicationV1 {
            adjudicator_id_sha256: digest(&format!("{id}:reviewer:{index}")),
            independence_group_sha256: digest(&format!("{id}:group:{index}")),
            decision,
            confidence_millionths: 900_000 - index as u32 * 50_000,
            rationale_sha256: digest(&format!("{id}:rationale:{index}:{decision:?}")),
        }
    }

    fn observation(
        variant: &MutationVariantPlanV1,
        decisions: &[SemanticReviewDecisionV1],
        pass: Option<bool>,
        failure: Option<&str>,
        status: MutationEvaluationStatusV1,
    ) -> MutationValidityObservationV1 {
        let completed = status == MutationEvaluationStatusV1::Completed;
        MutationValidityObservationV1 {
            mutation_id: variant.mutation_id.clone(),
            baseline_input_sha256: variant.baseline_input_sha256.clone(),
            variant_input_sha256: variant.variant_input_sha256.clone(),
            adjudications: decisions
                .iter()
                .enumerate()
                .map(|(index, decision)| rating(&variant.mutation_id, index, *decision))
                .collect(),
            evaluation_status: status,
            evaluator_relation_pass: if completed { pass } else { None },
            failure_fingerprint_sha256: if completed {
                failure.map(|label| digest(&format!("failure:{label}")))
            } else {
                None
            },
            generator_call_count: 1,
            generator_token_count: 1_000,
            semantic_assistance_call_count: 1,
            semantic_assistance_token_count: 200,
            evaluator_call_count: if completed { 1 } else { 0 },
            evaluator_token_count: if completed { 300 } else { 0 },
            human_review_seconds: if decisions.is_empty() { 0 } else { 120 },
            wall_time_milliseconds: 1_500,
        }
    }

    fn make_plan_and_receipt() -> (Vec<u8>, Vec<u8>) {
        let variants = vec![
            variant("preserve-valid-1", MutationSemanticRelationV1::Preserving),
            variant("change-valid", MutationSemanticRelationV1::Changing),
            variant("preserve-invalid", MutationSemanticRelationV1::Preserving),
            variant("preserve-disputed", MutationSemanticRelationV1::Preserving),
            variant("change-unreviewed", MutationSemanticRelationV1::Changing),
            variant("preserve-abstained", MutationSemanticRelationV1::Preserving),
            variant(
                "preserve-valid-duplicate",
                MutationSemanticRelationV1::Preserving,
            ),
        ];
        let fields = MutationValidityPlanFieldsV1 {
            schema: MUTATION_PLAN_SCHEMA.to_owned(),
            analyzer_source_revision: crate::BUILT_GENERATOR_SOURCE_REVISION.to_owned(),
            purpose: MutationPurposeV1::ExampleOnly,
            study_id: "mutation-validity-tests".to_owned(),
            generator_source_revision: "b".repeat(40),
            evaluator_source_revision: "c".repeat(40),
            generator_policy_sha256: digest("generator-policy"),
            semantic_task_set_sha256: digest("semantic-task-set"),
            review_protocol_sha256: digest("review-protocol"),
            metamorphic_plan_sha256: digest("metamorphic-plan"),
            evaluator_prompt_sha256: digest("evaluator-prompt"),
            rubric_sha256: digest("rubric"),
            reviewers_per_mutation: 3,
            variants: variants.clone(),
        };
        let commitment = plan_commitment(&fields).unwrap();
        let plan = MutationValidityPlanV1 {
            schema: fields.schema.clone(),
            analyzer_source_revision: fields.analyzer_source_revision.clone(),
            purpose: fields.purpose,
            study_id: fields.study_id.clone(),
            generator_source_revision: fields.generator_source_revision.clone(),
            evaluator_source_revision: fields.evaluator_source_revision.clone(),
            generator_policy_sha256: fields.generator_policy_sha256.clone(),
            semantic_task_set_sha256: fields.semantic_task_set_sha256.clone(),
            review_protocol_sha256: fields.review_protocol_sha256.clone(),
            metamorphic_plan_sha256: fields.metamorphic_plan_sha256.clone(),
            evaluator_prompt_sha256: fields.evaluator_prompt_sha256.clone(),
            rubric_sha256: fields.rubric_sha256.clone(),
            reviewers_per_mutation: fields.reviewers_per_mutation,
            variants: variants.clone(),
            plan_commitment_sha256: commitment,
        };
        let plan_bytes = serde_json::to_vec_pretty(&plan).unwrap();
        let receipt = MutationValidityReceiptV1 {
            schema: MUTATION_RECEIPT_SCHEMA.to_owned(),
            analyzer_source_revision: plan.analyzer_source_revision.clone(),
            purpose: MutationPurposeV1::ExampleOnly,
            run_id: "synthetic-run-001".to_owned(),
            plan_sha256: sha256_hex(&plan_bytes),
            plan_commitment_sha256: plan.plan_commitment_sha256.clone(),
            metamorphic_receipt_sha256: digest("metamorphic-receipt"),
            generator_source_revision: plan.generator_source_revision.clone(),
            evaluator_source_revision: plan.evaluator_source_revision.clone(),
            evaluator_prompt_sha256: plan.evaluator_prompt_sha256.clone(),
            rubric_sha256: plan.rubric_sha256.clone(),
            observations: vec![
                observation(
                    &variants[0],
                    &[SemanticReviewDecisionV1::PreservesSemantics; 3],
                    Some(false),
                    Some("fault-A"),
                    MutationEvaluationStatusV1::Completed,
                ),
                observation(
                    &variants[1],
                    &[SemanticReviewDecisionV1::ChangesSemantics; 3],
                    Some(true),
                    None,
                    MutationEvaluationStatusV1::Completed,
                ),
                observation(
                    &variants[2],
                    &[SemanticReviewDecisionV1::ChangesSemantics; 3],
                    Some(false),
                    Some("invalid-fault"),
                    MutationEvaluationStatusV1::Completed,
                ),
                observation(
                    &variants[3],
                    &[
                        SemanticReviewDecisionV1::PreservesSemantics,
                        SemanticReviewDecisionV1::ChangesSemantics,
                        SemanticReviewDecisionV1::Indeterminate,
                    ],
                    Some(false),
                    Some("disputed-fault"),
                    MutationEvaluationStatusV1::Completed,
                ),
                observation(
                    &variants[4],
                    &[],
                    None,
                    None,
                    MutationEvaluationStatusV1::NotStarted,
                ),
                observation(
                    &variants[5],
                    &[SemanticReviewDecisionV1::Abstain; 3],
                    Some(true),
                    None,
                    MutationEvaluationStatusV1::Completed,
                ),
                observation(
                    &variants[6],
                    &[SemanticReviewDecisionV1::PreservesSemantics; 3],
                    Some(false),
                    Some("fault-A"),
                    MutationEvaluationStatusV1::Completed,
                ),
            ],
        };
        let receipt_bytes = serde_json::to_vec_pretty(&receipt).unwrap();
        (plan_bytes, receipt_bytes)
    }

    #[test]
    fn clean_fixture_counts_only_failures_on_confirmed_valid_mutations() {
        let (plan, receipt) = make_plan_and_receipt();
        let report = analyze_mutation_validity(&plan, &receipt).unwrap();
        assert_eq!(report.planned_variant_count, 7);
        assert_eq!(report.observed_variant_count, 7);
        assert_eq!(report.completed_evaluation_count, 6);
        assert_eq!(report.ineligible_evaluation_count, 1);
        assert_eq!(report.planned_adjudication_slots, 21);
        assert_eq!(report.observed_adjudication_count, 18);
        assert_eq!(report.missing_adjudication_slots, 3);
        assert_eq!(report.confirmed_valid_preserving_count, 2);
        assert_eq!(report.confirmed_valid_changing_count, 1);
        assert_eq!(report.confirmed_invalid_count, 1);
        assert_eq!(report.disputed_count, 1);
        assert_eq!(report.abstained_count, 1);
        assert_eq!(report.unreviewed_count, 1);
        assert_eq!(report.adjudicable_confirmed_count, 4);
        assert_eq!(
            report.adjudicable_panel_concordance_millionths,
            Some(750_000.0)
        );
        assert_eq!(
            report.confirmed_valid_yield_over_planned_millionths,
            Some(3.0 / 7.0 * 1_000_000.0)
        );
        assert_eq!(report.pairwise_review_agreement_count, 15);
        assert_eq!(report.pairwise_review_disagreement_count, 3);
        assert_eq!(report.evaluator_failure_observation_count, 4);
        assert_eq!(report.verified_failure_observation_count, 2);
        assert_eq!(report.excluded_invalid_mutation_failure_count, 1);
        assert_eq!(report.excluded_unconfirmed_mutation_failure_count, 1);
        assert_eq!(report.unique_verified_failure_count, 1);
        assert_eq!(report.duplicate_verified_failure_observation_count, 1);
        assert_eq!(report.total_token_count, 10_200);
        assert_eq!(report.total_human_review_seconds, 720);
        assert_eq!(report.tokens_per_confirmed_valid_mutation, Some(3_400.0));
        assert_eq!(report.tokens_per_unique_verified_failure, Some(10_200.0));
        assert!(!report.raw_artifact_custody_proven);
        assert!(!report.reviewer_identity_authenticated);
        assert!(!report.reviewer_blinding_proven);
        assert!(!report.panel_reference_validity_proven);
        assert!(!report.evaluator_execution_proven);
        assert!(!report.causal_effect_identified);
        assert!(!report.generalized_validity_estimated);
        assert!(report.external_anchor_required);
        assert!(!report.scoring_eligible_declared);
    }

    #[test]
    fn report_commitment_verifies_and_detects_tampering() {
        let (plan, receipt) = make_plan_and_receipt();
        let report = analyze_mutation_validity(&plan, &receipt).unwrap();
        let bytes = serde_json::to_vec(&report).unwrap();
        let verified = verify_mutation_validity_report(&bytes).unwrap();
        assert_eq!(
            verified.report_commitment_sha256,
            report.report_commitment_sha256
        );
        let mut tampered: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        tampered["confirmed_valid_preserving_count"] = serde_json::json!(99);
        let tampered_bytes = serde_json::to_vec(&tampered).unwrap();
        assert!(
            verify_mutation_validity_report(&tampered_bytes)
                .unwrap_err()
                .contains("commitment mismatch")
        );
    }

    #[test]
    fn deterministic_for_exact_plan_and_receipt_bytes() {
        let (plan, receipt) = make_plan_and_receipt();
        let first =
            serde_json::to_vec(&analyze_mutation_validity(&plan, &receipt).unwrap()).unwrap();
        let second =
            serde_json::to_vec(&analyze_mutation_validity(&plan, &receipt).unwrap()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_missing_or_duplicate_mutation_observations() {
        let (plan, receipt) = make_plan_and_receipt();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"].as_array_mut().unwrap().pop();
        assert!(
            analyze_mutation_validity(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("missing")
        );
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        let first = value["observations"][0].clone();
        value["observations"].as_array_mut().unwrap().push(first);
        assert!(
            analyze_mutation_validity(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("duplicate")
        );
    }

    #[test]
    fn rejects_input_digest_drift_and_receipt_revision_drift() {
        let (plan, receipt) = make_plan_and_receipt();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][0]["variant_input_sha256"] = serde_json::json!(digest("wrong"));
        assert!(
            analyze_mutation_validity(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("input digest differs")
        );
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["evaluator_source_revision"] = serde_json::json!("d".repeat(40));
        assert!(
            analyze_mutation_validity(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("differs from frozen plan")
        );
    }

    #[test]
    fn rejects_reused_reviewer_identity_or_independence_group_within_variant() {
        let (plan, receipt) = make_plan_and_receipt();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][0]["adjudications"][1]["adjudicator_id_sha256"] =
            value["observations"][0]["adjudications"][0]["adjudicator_id_sha256"].clone();
        assert!(
            analyze_mutation_validity(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("reuses an adjudicator identity")
        );
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][0]["adjudications"][1]["independence_group_sha256"] =
            value["observations"][0]["adjudications"][0]["independence_group_sha256"].clone();
        assert!(
            analyze_mutation_validity(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("reuses an adjudicator independence group")
        );
    }

    #[test]
    fn rejects_semantics_changing_plan_without_target_dimension() {
        let (plan, _) = make_plan_and_receipt();
        let mut value: serde_json::Value = serde_json::from_slice(&plan).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("plan_commitment_sha256");
        let mut fields: MutationValidityPlanFieldsV1 = serde_json::from_value(value).unwrap();
        fields.variants[1].target_dimension = None;
        let issues = validate_mutation_validity_plan_fields(&fields).unwrap_err();
        assert!(
            issues
                .iter()
                .any(|issue| issue.contains("requires a target_dimension"))
        );
    }

    #[test]
    fn rejects_failed_relation_without_fingerprint_and_passing_with_fingerprint() {
        let (plan, receipt) = make_plan_and_receipt();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][0]["failure_fingerprint_sha256"] = serde_json::Value::Null;
        assert!(
            analyze_mutation_validity(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("requires a valid failure fingerprint")
        );
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][1]["failure_fingerprint_sha256"] =
            serde_json::json!(digest("unexpected"));
        assert!(
            analyze_mutation_validity(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("must not carry a failure fingerprint")
        );
    }

    #[test]
    fn rejects_partial_evaluator_outcome_on_ineligible_execution() {
        let (plan, receipt) = make_plan_and_receipt();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][4]["evaluator_relation_pass"] = serde_json::json!(false);
        assert!(
            analyze_mutation_validity(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("must not carry an evaluator outcome")
        );
    }

    #[test]
    fn cost_overflow_is_rejected_not_wrapped() {
        let (plan, receipt) = make_plan_and_receipt();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][0]["generator_token_count"] = serde_json::json!(u64::MAX);
        assert!(
            analyze_mutation_validity(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("counter overflow")
        );
    }

    #[test]
    fn rejects_more_reviewers_than_preregistered_slots() {
        let (plan, receipt) = make_plan_and_receipt();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        let first_rating = value["observations"][0]["adjudications"][0].clone();
        value["observations"][0]["adjudications"]
            .as_array_mut()
            .unwrap()
            .push(first_rating);
        assert!(
            analyze_mutation_validity(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("more adjudications than planned")
        );
    }
}
