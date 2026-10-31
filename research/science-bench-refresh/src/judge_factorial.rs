//! Balanced factorial analysis of interacting LLM-judge nuisance factors.
//!
//! This module verifies preregistered 2^5 within-task designs and computes
//! descriptive, full-denominator contrasts. It does not execute a judge,
//! authenticate raw prompt custody or randomization, establish semantic
//! equivalence, identify population causal effects, or grant scoring eligibility.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const FACTORIAL_PLAN_SCHEMA: &str = "symthaea.science_bench_refresh_judge_factorial_plan.v1";
pub const FACTORIAL_RECEIPT_SCHEMA: &str =
    "symthaea.science_bench_refresh_judge_factorial_receipt.v1";
pub const FACTORIAL_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_judge_factorial_report.v1";

const PLAN_DOMAIN: &[u8] = b"symthaea.judge_factorial.plan.v1\0";
const CONDITION_DOMAIN: &[u8] = b"symthaea.judge_factorial.condition.v1\0";
const REPORT_DOMAIN: &[u8] = b"symthaea.judge_factorial.report.v1\0";
const FACTOR_COUNT: usize = 5;
const FACTORIAL_CELLS: usize = 1 << FACTOR_COUNT;
const MAX_BLOCKS: usize = 1_000;
const MAX_REPEATS: u32 = 100;
const MAX_CONDITIONS: usize = 100_000;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FactorialPurposeV1 {
    ExampleOnly,
    QualifiedRun,
}

impl FactorialPurposeV1 {
    fn qualified(self) -> bool {
        self == Self::QualifiedRun
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CandidateOrderV1 {
    CandidateAFirst,
    CandidateBFirst,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ScoreOptionOrderV1 {
    Canonical,
    Reversed,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CriterionOrderV1 {
    Canonical,
    Reversed,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceCueV1 {
    Anonymous,
    Displayed,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum PresentationFormatV1 {
    Structured,
    Prose,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum JudgeNuisanceFactorV1 {
    CandidateOrder,
    ScoreOptionOrder,
    CriterionOrder,
    ProvenanceCue,
    PresentationFormat,
}

impl JudgeNuisanceFactorV1 {
    pub const ALL: [Self; FACTOR_COUNT] = [
        Self::CandidateOrder,
        Self::ScoreOptionOrder,
        Self::CriterionOrder,
        Self::ProvenanceCue,
        Self::PresentationFormat,
    ];
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct JudgeNuisanceFactorsV1 {
    pub candidate_order: CandidateOrderV1,
    pub score_option_order: ScoreOptionOrderV1,
    pub criterion_order: CriterionOrderV1,
    pub provenance_cue: ProvenanceCueV1,
    pub presentation_format: PresentationFormatV1,
}

impl JudgeNuisanceFactorsV1 {
    fn bit_levels(self) -> [bool; FACTOR_COUNT] {
        [
            self.candidate_order == CandidateOrderV1::CandidateBFirst,
            self.score_option_order == ScoreOptionOrderV1::Reversed,
            self.criterion_order == CriterionOrderV1::Reversed,
            self.provenance_cue == ProvenanceCueV1::Displayed,
            self.presentation_format == PresentationFormatV1::Prose,
        ]
    }

    fn from_bits(bits: [bool; FACTOR_COUNT]) -> Self {
        Self {
            candidate_order: if bits[0] {
                CandidateOrderV1::CandidateBFirst
            } else {
                CandidateOrderV1::CandidateAFirst
            },
            score_option_order: if bits[1] {
                ScoreOptionOrderV1::Reversed
            } else {
                ScoreOptionOrderV1::Canonical
            },
            criterion_order: if bits[2] {
                CriterionOrderV1::Reversed
            } else {
                CriterionOrderV1::Canonical
            },
            provenance_cue: if bits[3] {
                ProvenanceCueV1::Displayed
            } else {
                ProvenanceCueV1::Anonymous
            },
            presentation_format: if bits[4] {
                PresentationFormatV1::Prose
            } else {
                PresentationFormatV1::Structured
            },
        }
    }

    fn level(self, factor: JudgeNuisanceFactorV1) -> bool {
        let bits = self.bit_levels();
        bits[match factor {
            JudgeNuisanceFactorV1::CandidateOrder => 0,
            JudgeNuisanceFactorV1::ScoreOptionOrder => 1,
            JudgeNuisanceFactorV1::CriterionOrder => 2,
            JudgeNuisanceFactorV1::ProvenanceCue => 3,
            JudgeNuisanceFactorV1::PresentationFormat => 4,
        }]
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScoreOptionV1 {
    pub option_id: String,
    pub score_millionths: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FactorialConditionV1 {
    pub condition_id: String,
    pub repeat_index: u32,
    pub factors: JudgeNuisanceFactorsV1,
    pub input_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeFactorialBlockV1 {
    pub block_id: String,
    pub task_semantics_sha256: String,
    pub candidate_a_id: String,
    pub candidate_b_id: String,
    pub criterion_ids: Vec<String>,
    /// Canonical semantic order. Every option ID maps to one stable score.
    pub score_options: Vec<ScoreOptionV1>,
    /// Exact planned execution order. Must be a permutation of all conditions.
    pub execution_order: Vec<String>,
    pub conditions: Vec<FactorialConditionV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeFactorialPlanFieldsV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: FactorialPurposeV1,
    pub study_id: String,
    pub judge_source_revision: String,
    pub judge_prompt_sha256: String,
    pub rubric_sha256: String,
    pub semantic_task_set_sha256: String,
    pub repeats_per_condition: u32,
    pub blocks: Vec<JudgeFactorialBlockV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeFactorialPlanV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: FactorialPurposeV1,
    pub study_id: String,
    pub judge_source_revision: String,
    pub judge_prompt_sha256: String,
    pub rubric_sha256: String,
    pub semantic_task_set_sha256: String,
    pub repeats_per_condition: u32,
    pub blocks: Vec<JudgeFactorialBlockV1>,
    pub plan_commitment_sha256: String,
}

impl JudgeFactorialPlanV1 {
    fn fields(&self) -> JudgeFactorialPlanFieldsV1 {
        JudgeFactorialPlanFieldsV1 {
            schema: self.schema.clone(),
            analyzer_source_revision: self.analyzer_source_revision.clone(),
            purpose: self.purpose,
            study_id: self.study_id.clone(),
            judge_source_revision: self.judge_source_revision.clone(),
            judge_prompt_sha256: self.judge_prompt_sha256.clone(),
            rubric_sha256: self.rubric_sha256.clone(),
            semantic_task_set_sha256: self.semantic_task_set_sha256.clone(),
            repeats_per_condition: self.repeats_per_condition,
            blocks: self.blocks.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FactorialExecutionStatusV1 {
    Completed,
    TimedOut,
    Blocked,
    InvalidReceipt,
    NotStarted,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FactorialDecisionV1 {
    CandidateA,
    CandidateB,
    Tie,
    Abstain,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CriterionOutcomeV1 {
    pub criterion_id: String,
    pub score_millionths: u32,
    pub selected_option_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeFactorialObservationV1 {
    pub condition_id: String,
    pub execution_ordinal: u32,
    pub input_sha256: String,
    pub status: FactorialExecutionStatusV1,
    pub judge_receipt_sha256: Option<String>,
    pub decision: Option<FactorialDecisionV1>,
    pub candidate_a_score_millionths: Option<u32>,
    pub candidate_b_score_millionths: Option<u32>,
    pub criterion_scores: Vec<CriterionOutcomeV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeFactorialReceiptV1 {
    pub schema: String,
    pub purpose: FactorialPurposeV1,
    pub analyzer_source_revision: String,
    pub run_id: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub judge_source_revision: String,
    pub judge_prompt_sha256: String,
    pub rubric_sha256: String,
    pub observations: Vec<JudgeFactorialObservationV1>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FactorialLevelCountsV1 {
    pub factor: JudgeNuisanceFactorV1,
    pub low_level: String,
    pub high_level: String,
    pub low_planned: u64,
    pub low_completed: u64,
    pub low_ineligible: u64,
    pub high_planned: u64,
    pub high_completed: u64,
    pub high_ineligible: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FactorialMainEffectV1 {
    pub factor: JudgeNuisanceFactorV1,
    pub complete_low_count: u64,
    pub complete_high_count: u64,
    pub excluded_completed_low_count: u64,
    pub excluded_completed_high_count: u64,
    pub ineligible_low_count: u64,
    pub ineligible_high_count: u64,
    /// Mean candidate-A minus candidate-B score at high level minus low level.
    pub candidate_score_gap_delta_millionths: Option<f64>,
    /// Mean selected rubric score at high level minus low level.
    pub criterion_score_delta_millionths: Option<f64>,
    /// Normalized selected display-position delta (high minus low).
    pub selected_option_position_delta_millionths: Option<f64>,
    pub candidate_a_choice_rate_delta_millionths: Option<f64>,
    pub tie_rate_delta_millionths: Option<f64>,
    pub abstention_rate_delta_millionths: Option<f64>,
    pub decisive_low_count: u64,
    pub decisive_high_count: u64,
    pub tie_low_count: u64,
    pub tie_high_count: u64,
    pub abstention_low_count: u64,
    pub abstention_high_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CriterionOrderEffectV1 {
    pub criterion_id: String,
    pub canonical_mean_millionths: Option<f64>,
    pub reversed_mean_millionths: Option<f64>,
    pub reversed_minus_canonical_millionths: Option<f64>,
    pub canonical_completed_count: u64,
    pub reversed_completed_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FactorialInteractionV1 {
    pub factor_a: JudgeNuisanceFactorV1,
    pub factor_b: JudgeNuisanceFactorV1,
    /// Raw difference-in-differences: y11 - y10 - y01 + y00.
    pub candidate_score_gap_interaction_millionths: Option<f64>,
    pub criterion_score_interaction_millionths: Option<f64>,
    pub complete_00_count: u64,
    pub complete_01_count: u64,
    pub complete_10_count: u64,
    pub complete_11_count: u64,
    pub decisive_00_count: u64,
    pub decisive_01_count: u64,
    pub decisive_10_count: u64,
    pub decisive_11_count: u64,
    pub tie_00_count: u64,
    pub tie_01_count: u64,
    pub tie_10_count: u64,
    pub tie_11_count: u64,
    pub abstention_00_count: u64,
    pub abstention_01_count: u64,
    pub abstention_10_count: u64,
    pub abstention_11_count: u64,
    pub candidate_a_choice_interaction_millionths: Option<f64>,
    pub tie_rate_interaction_millionths: Option<f64>,
    pub abstention_rate_interaction_millionths: Option<f64>,
    pub excluded_conditions_across_four_cells: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FactorialBlockSummaryV1 {
    pub block_id: String,
    pub planned_condition_count: u64,
    pub completed_condition_count: u64,
    pub ineligible_condition_count: u64,
    pub fully_completed: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct JudgeFactorialReportFieldsV1 {
    pub schema: String,
    pub study_id: String,
    pub analyzer_source_revision: String,
    pub analyzer_worktree_clean: bool,
    pub balanced_repeat_count: u64,
    pub excluded_incomplete_repeat_count: u64,
    pub effects_use_only_fully_completed_repeats: bool,
    pub plan_sha256: String,
    pub receipt_sha256: String,
    pub plan_commitment_sha256: String,
    pub planned_block_count: u64,
    pub observed_block_count: u64,
    pub planned_condition_count: u64,
    pub observed_condition_count: u64,
    pub completed_condition_count: u64,
    pub ineligible_condition_count: u64,
    pub all_blocks_present: bool,
    pub all_conditions_present: bool,
    pub all_conditions_completed: bool,
    pub level_counts: Vec<FactorialLevelCountsV1>,
    pub main_effects: Vec<FactorialMainEffectV1>,
    pub criterion_order_effects: Vec<CriterionOrderEffectV1>,
    pub pairwise_interactions: Vec<FactorialInteractionV1>,
    pub block_summaries: Vec<FactorialBlockSummaryV1>,
    pub execution_authenticity_proven: bool,
    pub randomization_proven: bool,
    pub raw_prompt_custody_proven: bool,
    pub semantic_equivalence_proven: bool,
    pub causal_effect_identified: bool,
    pub generalized_accuracy_estimated: bool,
    pub external_anchor_required: bool,
    pub scoring_eligible_declared: bool,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct JudgeFactorialReportV1 {
    #[serde(flatten)]
    pub fields: JudgeFactorialReportFieldsV1,
    pub report_commitment_sha256: String,
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

fn valid_revision(value: &str, allow_placeholder: bool) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && (allow_placeholder || value.bytes().any(|byte| byte != b'0'))
}

fn nonempty(value: &str, max: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max
}

fn condition_id(
    block_id: &str,
    repeat_index: u32,
    factors: JudgeNuisanceFactorsV1,
) -> Result<String, String> {
    #[derive(Serialize)]
    struct Identity<'a> {
        block_id: &'a str,
        repeat_index: u32,
        factors: JudgeNuisanceFactorsV1,
    }
    let payload = serde_json::to_vec(&Identity {
        block_id,
        repeat_index,
        factors,
    })
    .map_err(|error| format!("cannot serialize condition identity: {error}"))?;
    digest_domain(CONDITION_DOMAIN, &payload)
}

fn plan_fields_commitment(fields: &JudgeFactorialPlanFieldsV1) -> Result<String, String> {
    let payload = serde_json::to_vec(fields)
        .map_err(|error| format!("cannot serialize factorial plan fields: {error}"))?;
    digest_domain(PLAN_DOMAIN, &payload)
}

/// Validate all block-level factorial completeness and frozen semantic identities.
pub fn validate_judge_factorial_plan_fields(
    fields: &JudgeFactorialPlanFieldsV1,
) -> Result<(), Vec<String>> {
    let mut issues = Vec::new();
    let allow_zero = !fields.purpose.qualified();
    if fields.schema != FACTORIAL_PLAN_SCHEMA {
        issues.push("unsupported judge factorial plan schema".to_owned());
    }
    if !valid_revision(&fields.analyzer_source_revision, false)
        || fields.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        issues.push(
            "analyzer_source_revision must match the exact compiled validator revision".to_owned(),
        );
    }
    if fields.purpose.qualified() && crate::BUILT_GENERATOR_WORKTREE_CLEAN != "true" {
        issues.push("qualified factorial plans require a clean analyzer worktree".to_owned());
    }
    if !nonempty(&fields.study_id, 128) {
        issues.push("study_id must be non-empty and at most 128 bytes".to_owned());
    }
    if !valid_revision(&fields.judge_source_revision, allow_zero) {
        issues
            .push("judge_source_revision must be a valid nonzero 40-character revision".to_owned());
    }
    for (label, value) in [
        ("judge_prompt_sha256", &fields.judge_prompt_sha256),
        ("rubric_sha256", &fields.rubric_sha256),
        ("semantic_task_set_sha256", &fields.semantic_task_set_sha256),
    ] {
        if !valid_digest(value, allow_zero) {
            issues.push(format!("{label} must be a valid nonzero SHA-256 digest"));
        }
    }
    if fields.repeats_per_condition == 0 || fields.repeats_per_condition > MAX_REPEATS {
        issues.push(format!(
            "repeats_per_condition must be in 1..={MAX_REPEATS}"
        ));
    }
    if fields.blocks.is_empty() || fields.blocks.len() > MAX_BLOCKS {
        issues.push(format!("blocks must contain 1..={MAX_BLOCKS} blocks"));
    }
    let planned_total = fields
        .blocks
        .len()
        .checked_mul(FACTORIAL_CELLS)
        .and_then(|count| count.checked_mul(fields.repeats_per_condition as usize));
    if planned_total.is_none_or(|count| count > MAX_CONDITIONS) {
        issues.push(format!("total planned conditions exceed {MAX_CONDITIONS}"));
    }

    let mut block_ids = BTreeSet::new();
    for block in &fields.blocks {
        if !nonempty(&block.block_id, 128) || !block_ids.insert(block.block_id.as_str()) {
            issues.push("block_id must be non-empty, bounded and unique".to_owned());
        }
        if !valid_digest(&block.task_semantics_sha256, allow_zero) {
            issues.push(format!(
                "block {} has invalid task_semantics_sha256",
                block.block_id
            ));
        }
        if !nonempty(&block.candidate_a_id, 128)
            || !nonempty(&block.candidate_b_id, 128)
            || block.candidate_a_id == block.candidate_b_id
        {
            issues.push(format!(
                "block {} requires two distinct non-empty candidate IDs",
                block.block_id
            ));
        }
        if block.criterion_ids.len() < 2
            || block.criterion_ids.len() > 128
            || block.criterion_ids.iter().any(|id| !nonempty(id, 128))
            || block.criterion_ids.iter().collect::<BTreeSet<_>>().len()
                != block.criterion_ids.len()
        {
            issues.push(format!(
                "block {} requires 2..=128 unique criterion IDs",
                block.block_id
            ));
        }
        if block.score_options.len() < 2 || block.score_options.len() > 16 {
            issues.push(format!(
                "block {} requires 2..=16 score options",
                block.block_id
            ));
        }
        let mut option_ids = BTreeSet::new();
        let mut previous_score = None;
        for option in &block.score_options {
            if !nonempty(&option.option_id, 128) || !option_ids.insert(option.option_id.as_str()) {
                issues.push(format!(
                    "block {} has empty or duplicate score option ID",
                    block.block_id
                ));
            }
            if option.score_millionths > 1_000_000
                || previous_score.is_some_and(|previous| option.score_millionths <= previous)
            {
                issues.push(format!(
                    "block {} score options must have strictly increasing scores in 0..=1,000,000",
                    block.block_id
                ));
            }
            previous_score = Some(option.score_millionths);
        }
        let expected_count = FACTORIAL_CELLS.saturating_mul(fields.repeats_per_condition as usize);
        if block.conditions.len() != expected_count {
            issues.push(format!(
                "block {} has {} conditions; expected {} for a complete 2^5 design",
                block.block_id,
                block.conditions.len(),
                expected_count
            ));
        }
        let condition_ids: BTreeSet<&str> = block
            .conditions
            .iter()
            .map(|condition| condition.condition_id.as_str())
            .collect();
        if condition_ids.len() != block.conditions.len() {
            issues.push(format!(
                "block {} contains duplicate condition IDs",
                block.block_id
            ));
        }
        let schedule_ids: BTreeSet<&str> =
            block.execution_order.iter().map(String::as_str).collect();
        if block.execution_order.len() != block.conditions.len()
            || schedule_ids.len() != block.execution_order.len()
            || schedule_ids != condition_ids
        {
            issues.push(format!("block {} execution_order must be an exact permutation of all planned condition IDs", block.block_id));
        }

        let mut seen_cells = BTreeSet::new();
        for condition in &block.conditions {
            if condition.repeat_index >= fields.repeats_per_condition {
                issues.push(format!(
                    "condition {} repeat_index is outside the planned range",
                    condition.condition_id
                ));
            }
            if !valid_digest(&condition.input_sha256, allow_zero) {
                issues.push(format!(
                    "condition {} has invalid input_sha256",
                    condition.condition_id
                ));
            }
            match condition_id(&block.block_id, condition.repeat_index, condition.factors) {
                Ok(expected) if expected == condition.condition_id => {}
                Ok(_) => issues.push(format!(
                    "condition {} ID does not match its block/repeat/factor assignment",
                    condition.condition_id
                )),
                Err(error) => issues.push(error),
            }
            if !seen_cells.insert((condition.repeat_index, condition.factors)) {
                issues.push(format!(
                    "block {} repeats a factorial cell within the same repeat index",
                    block.block_id
                ));
            }
        }
        for repeat in 0..fields.repeats_per_condition {
            for code in 0..FACTORIAL_CELLS {
                let bits = std::array::from_fn(|index| code & (1 << index) != 0);
                let factors = JudgeNuisanceFactorsV1::from_bits(bits);
                if !seen_cells.contains(&(repeat, factors)) {
                    issues.push(format!(
                        "block {} is missing factorial cell {code} at repeat {repeat}",
                        block.block_id
                    ));
                }
            }
        }
    }
    if let Some(reference_block) = fields.blocks.first() {
        for block in fields.blocks.iter().skip(1) {
            if block.criterion_ids != reference_block.criterion_ids {
                issues.push(format!(
                    "block {} criterion IDs/order differ from the study rubric",
                    block.block_id
                ));
            }
            if block.score_options != reference_block.score_options {
                issues.push(format!(
                    "block {} score options differ from the study rubric",
                    block.block_id
                ));
            }
        }
    }
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn compute_judge_factorial_plan_commitment(fields_bytes: &[u8]) -> Result<String, String> {
    let fields: JudgeFactorialPlanFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("judge factorial plan fields invalid: {error}"))?;
    validate_judge_factorial_plan_fields(&fields).map_err(|issues| issues.join("; "))?;
    plan_fields_commitment(&fields)
}

pub fn verify_judge_factorial_plan(
    plan_bytes: &[u8],
) -> Result<(JudgeFactorialPlanV1, String), String> {
    let plan: JudgeFactorialPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("judge factorial plan invalid: {error}"))?;
    let fields = plan.fields();
    validate_judge_factorial_plan_fields(&fields).map_err(|issues| issues.join("; "))?;
    let expected = plan_fields_commitment(&fields)?;
    if !valid_digest(&plan.plan_commitment_sha256, !plan.purpose.qualified())
        || plan.plan_commitment_sha256 != expected
    {
        return Err("judge factorial plan commitment mismatch".to_owned());
    }
    Ok((plan, sha256_hex(plan_bytes)))
}

#[derive(Debug, Clone)]
struct CompletedMetricV1 {
    block_id: String,
    repeat_index: u32,
    factors: JudgeNuisanceFactorsV1,
    candidate_score_gap: f64,
    mean_criterion_score: f64,
    mean_selected_option_position: f64,
    candidate_a_choice: Option<f64>,
    tie_indicator: f64,
    abstention_indicator: f64,
    criterion_scores: BTreeMap<String, f64>,
}

fn metric_for_observation(
    block: &JudgeFactorialBlockV1,
    condition: &FactorialConditionV1,
    observation: &JudgeFactorialObservationV1,
) -> Result<CompletedMetricV1, String> {
    let candidate_a = observation.candidate_a_score_millionths.ok_or_else(|| {
        format!(
            "completed condition {} omitted candidate A score",
            condition.condition_id
        )
    })?;
    let candidate_b = observation.candidate_b_score_millionths.ok_or_else(|| {
        format!(
            "completed condition {} omitted candidate B score",
            condition.condition_id
        )
    })?;
    let mut scores = BTreeMap::new();
    let option_index: BTreeMap<&str, usize> = block
        .score_options
        .iter()
        .enumerate()
        .map(|(index, option)| (option.option_id.as_str(), index))
        .collect();
    let mut score_sum = 0.0;
    let mut position_sum = 0.0;
    for score in &observation.criterion_scores {
        if !block.criterion_ids.contains(&score.criterion_id) {
            return Err(format!(
                "condition {} has an unknown criterion ID",
                condition.condition_id
            ));
        }
        let option = block
            .score_options
            .iter()
            .find(|option| option.option_id == score.selected_option_id)
            .ok_or_else(|| {
                format!(
                    "condition {} selected an unknown score option",
                    condition.condition_id
                )
            })?;
        if score.score_millionths != option.score_millionths {
            return Err(format!(
                "condition {} score does not match its selected semantic option",
                condition.condition_id
            ));
        }
        let index = *option_index
            .get(score.selected_option_id.as_str())
            .unwrap_or(&0);
        let displayed_position =
            if condition.factors.score_option_order == ScoreOptionOrderV1::Reversed {
                block.score_options.len() - 1 - index
            } else {
                index
            };
        let normalized = if block.score_options.len() <= 1 {
            0.0
        } else {
            displayed_position as f64 / (block.score_options.len() - 1) as f64
        };
        position_sum += normalized;
        score_sum += f64::from(score.score_millionths);
        scores.insert(
            score.criterion_id.clone(),
            f64::from(score.score_millionths),
        );
    }
    if scores.len() != block.criterion_ids.len() {
        return Err(format!(
            "condition {} must report every planned criterion exactly once",
            condition.condition_id
        ));
    }
    let criterion_count = block.criterion_ids.len() as f64;
    Ok(CompletedMetricV1 {
        block_id: block.block_id.clone(),
        repeat_index: condition.repeat_index,
        factors: condition.factors,
        candidate_score_gap: f64::from(candidate_a) - f64::from(candidate_b),
        mean_criterion_score: score_sum / criterion_count,
        mean_selected_option_position: position_sum / criterion_count,
        candidate_a_choice: match observation.decision {
            Some(FactorialDecisionV1::CandidateA) => Some(1_000_000.0),
            Some(FactorialDecisionV1::CandidateB) => Some(0.0),
            Some(FactorialDecisionV1::Tie | FactorialDecisionV1::Abstain) | None => None,
        },
        tie_indicator: if observation.decision == Some(FactorialDecisionV1::Tie) {
            1.0
        } else {
            0.0
        },
        abstention_indicator: if observation.decision == Some(FactorialDecisionV1::Abstain) {
            1.0
        } else {
            0.0
        },
        criterion_scores: scores,
    })
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values: Vec<f64> = values.collect();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn select_balanced_factorial_repeats(
    plan: &JudgeFactorialPlanV1,
    metrics: &[CompletedMetricV1],
) -> (Vec<CompletedMetricV1>, u64, u64) {
    let mut grouped: BTreeMap<(String, u32), Vec<&CompletedMetricV1>> = BTreeMap::new();
    for metric in metrics {
        grouped
            .entry((metric.block_id.clone(), metric.repeat_index))
            .or_default()
            .push(metric);
    }
    let mut balanced = Vec::new();
    let mut balanced_repeat_count = 0_u64;
    let mut excluded_repeat_count = 0_u64;
    for block in &plan.blocks {
        for repeat_index in 0..plan.repeats_per_condition {
            let key = (block.block_id.clone(), repeat_index);
            let rows = grouped.get(&key).map(Vec::as_slice).unwrap_or_default();
            if rows.len() == FACTORIAL_CELLS {
                balanced.extend(rows.iter().map(|metric| (*metric).clone()));
                balanced_repeat_count += 1;
            } else {
                excluded_repeat_count += 1;
            }
        }
    }
    (balanced, balanced_repeat_count, excluded_repeat_count)
}

fn factor_level_label(factor: JudgeNuisanceFactorV1, high: bool) -> &'static str {
    match (factor, high) {
        (JudgeNuisanceFactorV1::CandidateOrder, false) => "candidate_a_first",
        (JudgeNuisanceFactorV1::CandidateOrder, true) => "candidate_b_first",
        (JudgeNuisanceFactorV1::ScoreOptionOrder, false) => "canonical",
        (JudgeNuisanceFactorV1::ScoreOptionOrder, true) => "reversed",
        (JudgeNuisanceFactorV1::CriterionOrder, false) => "canonical",
        (JudgeNuisanceFactorV1::CriterionOrder, true) => "reversed",
        (JudgeNuisanceFactorV1::ProvenanceCue, false) => "anonymous",
        (JudgeNuisanceFactorV1::ProvenanceCue, true) => "displayed",
        (JudgeNuisanceFactorV1::PresentationFormat, false) => "structured",
        (JudgeNuisanceFactorV1::PresentationFormat, true) => "prose",
    }
}

fn main_effects(
    plan: &JudgeFactorialPlanV1,
    metrics: &[CompletedMetricV1],
    observations_by_id: &BTreeMap<String, &JudgeFactorialObservationV1>,
) -> (
    Vec<FactorialMainEffectV1>,
    Vec<FactorialLevelCountsV1>,
    Vec<CriterionOrderEffectV1>,
) {
    let mut effects = Vec::new();
    let mut counts = Vec::new();
    let mut criterion_effects = Vec::new();
    for factor in JudgeNuisanceFactorV1::ALL {
        let low: Vec<&CompletedMetricV1> = metrics
            .iter()
            .filter(|metric| !metric.factors.level(factor))
            .collect();
        let high: Vec<&CompletedMetricV1> = metrics
            .iter()
            .filter(|metric| metric.factors.level(factor))
            .collect();
        let low_gap = mean(low.iter().map(|metric| metric.candidate_score_gap));
        let high_gap = mean(high.iter().map(|metric| metric.candidate_score_gap));
        let low_score = mean(low.iter().map(|metric| metric.mean_criterion_score));
        let high_score = mean(high.iter().map(|metric| metric.mean_criterion_score));
        let low_position = mean(
            low.iter()
                .map(|metric| metric.mean_selected_option_position),
        );
        let high_position = mean(
            high.iter()
                .map(|metric| metric.mean_selected_option_position),
        );
        let low_choice = mean(low.iter().filter_map(|metric| metric.candidate_a_choice));
        let high_choice = mean(high.iter().filter_map(|metric| metric.candidate_a_choice));
        let low_tie_rate = mean(low.iter().map(|metric| metric.tie_indicator));
        let high_tie_rate = mean(high.iter().map(|metric| metric.tie_indicator));
        let low_abstention_rate = mean(low.iter().map(|metric| metric.abstention_indicator));
        let high_abstention_rate = mean(high.iter().map(|metric| metric.abstention_indicator));

        let mut planned_low = 0_u64;
        let mut planned_high = 0_u64;
        let mut completed_low = 0_u64;
        let mut completed_high = 0_u64;
        let mut ineligible_low = 0_u64;
        let mut ineligible_high = 0_u64;
        for block in &plan.blocks {
            for condition in &block.conditions {
                let high_level = condition.factors.level(factor);
                let completed =
                    observations_by_id
                        .get(&condition.condition_id)
                        .is_some_and(|observation| {
                            observation.status == FactorialExecutionStatusV1::Completed
                        });
                if high_level {
                    planned_high += 1;
                    if completed {
                        completed_high += 1;
                    } else {
                        ineligible_high += 1;
                    }
                } else {
                    planned_low += 1;
                    if completed {
                        completed_low += 1;
                    } else {
                        ineligible_low += 1;
                    }
                }
            }
        }
        effects.push(FactorialMainEffectV1 {
            factor,
            complete_low_count: low.len() as u64,
            complete_high_count: high.len() as u64,
            excluded_completed_low_count: completed_low.saturating_sub(low.len() as u64),
            excluded_completed_high_count: completed_high.saturating_sub(high.len() as u64),
            ineligible_low_count: ineligible_low,
            ineligible_high_count: ineligible_high,
            candidate_score_gap_delta_millionths: high_gap
                .zip(low_gap)
                .map(|(high, low)| high - low),
            criterion_score_delta_millionths: high_score
                .zip(low_score)
                .map(|(high, low)| high - low),
            selected_option_position_delta_millionths: high_position
                .zip(low_position)
                .map(|(high, low)| (high - low) * 1_000_000.0),
            candidate_a_choice_rate_delta_millionths: high_choice
                .zip(low_choice)
                .map(|(high, low)| high - low),
            decisive_low_count: low
                .iter()
                .filter(|metric| metric.candidate_a_choice.is_some())
                .count() as u64,
            decisive_high_count: high
                .iter()
                .filter(|metric| metric.candidate_a_choice.is_some())
                .count() as u64,
            tie_rate_delta_millionths: high_tie_rate
                .zip(low_tie_rate)
                .map(|(high, low)| (high - low) * 1_000_000.0),
            abstention_rate_delta_millionths: high_abstention_rate
                .zip(low_abstention_rate)
                .map(|(high, low)| (high - low) * 1_000_000.0),
            tie_low_count: low
                .iter()
                .filter(|metric| metric.tie_indicator > 0.5)
                .count() as u64,
            tie_high_count: high
                .iter()
                .filter(|metric| metric.tie_indicator > 0.5)
                .count() as u64,
            abstention_low_count: low
                .iter()
                .filter(|metric| metric.abstention_indicator > 0.5)
                .count() as u64,
            abstention_high_count: high
                .iter()
                .filter(|metric| metric.abstention_indicator > 0.5)
                .count() as u64,
        });
        counts.push(FactorialLevelCountsV1 {
            factor,
            low_level: factor_level_label(factor, false).to_owned(),
            high_level: factor_level_label(factor, true).to_owned(),
            low_planned: planned_low,
            low_completed: completed_low,
            low_ineligible: ineligible_low,
            high_planned: planned_high,
            high_completed: completed_high,
            high_ineligible: ineligible_high,
        });
    }

    for criterion_id in plan
        .blocks
        .iter()
        .flat_map(|block| block.criterion_ids.iter())
        .collect::<BTreeSet<_>>()
    {
        let canonical: Vec<f64> = metrics
            .iter()
            .filter(|metric| metric.factors.criterion_order == CriterionOrderV1::Canonical)
            .filter_map(|metric| metric.criterion_scores.get(criterion_id).copied())
            .collect();
        let reversed: Vec<f64> = metrics
            .iter()
            .filter(|metric| metric.factors.criterion_order == CriterionOrderV1::Reversed)
            .filter_map(|metric| metric.criterion_scores.get(criterion_id).copied())
            .collect();
        let canonical_mean = mean(canonical.iter().copied());
        let reversed_mean = mean(reversed.iter().copied());
        criterion_effects.push(CriterionOrderEffectV1 {
            criterion_id: criterion_id.to_string(),
            canonical_mean_millionths: canonical_mean,
            reversed_mean_millionths: reversed_mean,
            reversed_minus_canonical_millionths: reversed_mean
                .zip(canonical_mean)
                .map(|(reversed, canonical)| reversed - canonical),
            canonical_completed_count: canonical.len() as u64,
            reversed_completed_count: reversed.len() as u64,
        });
    }
    (effects, counts, criterion_effects)
}

fn pairwise_interactions(
    plan: &JudgeFactorialPlanV1,
    metrics: &[CompletedMetricV1],
) -> Vec<FactorialInteractionV1> {
    let mut output = Vec::new();
    for first in 0..FACTOR_COUNT {
        for second in (first + 1)..FACTOR_COUNT {
            let factor_a = JudgeNuisanceFactorV1::ALL[first];
            let factor_b = JudgeNuisanceFactorV1::ALL[second];
            let mean_at = |a: bool, b: bool, select: fn(&CompletedMetricV1) -> f64| {
                mean(
                    metrics
                        .iter()
                        .filter(|metric| {
                            metric.factors.level(factor_a) == a
                                && metric.factors.level(factor_b) == b
                        })
                        .map(select),
                )
            };
            let m00_gap = mean_at(false, false, |metric| metric.candidate_score_gap);
            let m01_gap = mean_at(false, true, |metric| metric.candidate_score_gap);
            let m10_gap = mean_at(true, false, |metric| metric.candidate_score_gap);
            let m11_gap = mean_at(true, true, |metric| metric.candidate_score_gap);
            let m00_score = mean_at(false, false, |metric| metric.mean_criterion_score);
            let m01_score = mean_at(false, true, |metric| metric.mean_criterion_score);
            let m10_score = mean_at(true, false, |metric| metric.mean_criterion_score);
            let m11_score = mean_at(true, true, |metric| metric.mean_criterion_score);
            let mean_choice_at = |a: bool, b: bool| {
                mean(
                    metrics
                        .iter()
                        .filter(|metric| {
                            metric.factors.level(factor_a) == a
                                && metric.factors.level(factor_b) == b
                        })
                        .filter_map(|metric| metric.candidate_a_choice),
                )
            };
            let m00_choice = mean_choice_at(false, false);
            let m01_choice = mean_choice_at(false, true);
            let m10_choice = mean_choice_at(true, false);
            let m11_choice = mean_choice_at(true, true);
            let mean_indicator_at = |a: bool, b: bool, select: fn(&CompletedMetricV1) -> f64| {
                mean(
                    metrics
                        .iter()
                        .filter(|metric| {
                            metric.factors.level(factor_a) == a
                                && metric.factors.level(factor_b) == b
                        })
                        .map(select),
                )
            };
            let m00_tie = mean_indicator_at(false, false, |metric| metric.tie_indicator);
            let m01_tie = mean_indicator_at(false, true, |metric| metric.tie_indicator);
            let m10_tie = mean_indicator_at(true, false, |metric| metric.tie_indicator);
            let m11_tie = mean_indicator_at(true, true, |metric| metric.tie_indicator);
            let m00_abstention =
                mean_indicator_at(false, false, |metric| metric.abstention_indicator);
            let m01_abstention =
                mean_indicator_at(false, true, |metric| metric.abstention_indicator);
            let m10_abstention =
                mean_indicator_at(true, false, |metric| metric.abstention_indicator);
            let m11_abstention =
                mean_indicator_at(true, true, |metric| metric.abstention_indicator);
            let count_at = |a: bool, b: bool| {
                metrics
                    .iter()
                    .filter(|metric| {
                        metric.factors.level(factor_a) == a && metric.factors.level(factor_b) == b
                    })
                    .count() as u64
            };
            let counts = [
                count_at(false, false),
                count_at(false, true),
                count_at(true, false),
                count_at(true, true),
            ];
            let decisive_at = |a: bool, b: bool| {
                metrics
                    .iter()
                    .filter(|metric| {
                        metric.factors.level(factor_a) == a
                            && metric.factors.level(factor_b) == b
                            && metric.candidate_a_choice.is_some()
                    })
                    .count() as u64
            };
            let decisive_counts = [
                decisive_at(false, false),
                decisive_at(false, true),
                decisive_at(true, false),
                decisive_at(true, true),
            ];
            let tie_at = |a: bool, b: bool| {
                metrics
                    .iter()
                    .filter(|metric| {
                        metric.factors.level(factor_a) == a
                            && metric.factors.level(factor_b) == b
                            && metric.tie_indicator > 0.5
                    })
                    .count() as u64
            };
            let abstention_at = |a: bool, b: bool| {
                metrics
                    .iter()
                    .filter(|metric| {
                        metric.factors.level(factor_a) == a
                            && metric.factors.level(factor_b) == b
                            && metric.abstention_indicator > 0.5
                    })
                    .count() as u64
            };
            let tie_counts = [
                tie_at(false, false),
                tie_at(false, true),
                tie_at(true, false),
                tie_at(true, true),
            ];
            let abstention_counts = [
                abstention_at(false, false),
                abstention_at(false, true),
                abstention_at(true, false),
                abstention_at(true, true),
            ];
            let completed_in_four_cells = counts.iter().copied().sum::<u64>();
            let planned_per_cell = plan.blocks.len() as u64
                * u64::from(plan.repeats_per_condition)
                * (FACTORIAL_CELLS as u64 / 4);
            let planned_in_four_cells = planned_per_cell * 4;
            output.push(FactorialInteractionV1 {
                factor_a,
                factor_b,
                candidate_score_gap_interaction_millionths: m11_gap
                    .zip(m10_gap)
                    .zip(m01_gap.zip(m00_gap))
                    .map(|((m11, m10), (m01, m00))| m11 - m10 - m01 + m00),
                criterion_score_interaction_millionths: m11_score
                    .zip(m10_score)
                    .zip(m01_score.zip(m00_score))
                    .map(|((m11, m10), (m01, m00))| m11 - m10 - m01 + m00),
                complete_00_count: counts[0],
                complete_01_count: counts[1],
                complete_10_count: counts[2],
                complete_11_count: counts[3],
                decisive_00_count: decisive_counts[0],
                decisive_01_count: decisive_counts[1],
                decisive_10_count: decisive_counts[2],
                decisive_11_count: decisive_counts[3],
                tie_00_count: tie_counts[0],
                tie_01_count: tie_counts[1],
                tie_10_count: tie_counts[2],
                tie_11_count: tie_counts[3],
                abstention_00_count: abstention_counts[0],
                abstention_01_count: abstention_counts[1],
                abstention_10_count: abstention_counts[2],
                abstention_11_count: abstention_counts[3],
                candidate_a_choice_interaction_millionths: m11_choice
                    .zip(m10_choice)
                    .zip(m01_choice.zip(m00_choice))
                    .map(|((m11, m10), (m01, m00))| m11 - m10 - m01 + m00),
                tie_rate_interaction_millionths: m11_tie
                    .zip(m10_tie)
                    .zip(m01_tie.zip(m00_tie))
                    .map(|((m11, m10), (m01, m00))| (m11 - m10 - m01 + m00) * 1_000_000.0),
                abstention_rate_interaction_millionths: m11_abstention
                    .zip(m10_abstention)
                    .zip(m01_abstention.zip(m00_abstention))
                    .map(|((m11, m10), (m01, m00))| (m11 - m10 - m01 + m00) * 1_000_000.0),
                excluded_conditions_across_four_cells: planned_in_four_cells
                    .saturating_sub(completed_in_four_cells),
            });
        }
    }
    output
}

/// Validate plan/receipt bytes and derive a deterministic descriptive report.
pub fn analyze_judge_factorial(
    plan_bytes: &[u8],
    receipt_bytes: &[u8],
) -> Result<JudgeFactorialReportV1, String> {
    let (plan, plan_sha256) = verify_judge_factorial_plan(plan_bytes)?;
    let receipt: JudgeFactorialReceiptV1 = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("judge factorial receipt invalid: {error}"))?;
    if receipt.schema != FACTORIAL_RECEIPT_SCHEMA {
        return Err("unsupported judge factorial receipt schema".to_owned());
    }
    if receipt.purpose != plan.purpose {
        return Err("receipt purpose differs from the plan".to_owned());
    }
    if receipt.analyzer_source_revision != plan.analyzer_source_revision
        || receipt.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        return Err(
            "receipt analyzer revision differs from the exact compiled validator revision"
                .to_owned(),
        );
    }
    if !nonempty(&receipt.run_id, 128) {
        return Err("receipt run_id must be non-empty and at most 128 bytes".to_owned());
    }
    if receipt.plan_sha256 != plan_sha256
        || receipt.plan_commitment_sha256 != plan.plan_commitment_sha256
    {
        return Err("receipt does not bind exact plan bytes and commitment".to_owned());
    }
    if receipt.judge_source_revision != plan.judge_source_revision
        || receipt.judge_prompt_sha256 != plan.judge_prompt_sha256
        || receipt.rubric_sha256 != plan.rubric_sha256
    {
        return Err(
            "receipt evaluator revision/prompt/rubric differs from the frozen plan".to_owned(),
        );
    }

    let mut expected = BTreeMap::new();
    for block in &plan.blocks {
        let conditions_by_id: BTreeMap<&str, &FactorialConditionV1> = block
            .conditions
            .iter()
            .map(|condition| (condition.condition_id.as_str(), condition))
            .collect();
        for (ordinal, condition_id) in block.execution_order.iter().enumerate() {
            let condition = conditions_by_id.get(condition_id.as_str()).ok_or_else(|| {
                format!(
                    "block {} execution schedule references an unknown condition",
                    block.block_id
                )
            })?;
            let execution_ordinal =
                u32::try_from(ordinal).map_err(|_| "execution ordinal exceeds u32".to_owned())?;
            expected.insert(
                condition_id.as_str(),
                (block, *condition, execution_ordinal),
            );
        }
    }
    let mut observed_ids = BTreeSet::new();
    for observation in &receipt.observations {
        if !observed_ids.insert(observation.condition_id.as_str()) {
            return Err(format!(
                "duplicate observation condition_id {}",
                observation.condition_id
            ));
        }
        let (block, condition, expected_execution_ordinal) = expected
            .get(observation.condition_id.as_str())
            .ok_or_else(|| {
                format!(
                    "unexpected observation condition_id {}",
                    observation.condition_id
                )
            })?;
        if observation.execution_ordinal != *expected_execution_ordinal {
            return Err(format!(
                "condition {} execution_ordinal differs from the committed schedule",
                condition.condition_id
            ));
        }
        if observation.input_sha256 != condition.input_sha256 {
            return Err(format!(
                "condition {} input digest differs from the committed plan",
                condition.condition_id
            ));
        }
        match observation.status {
            FactorialExecutionStatusV1::Completed => {
                let digest = observation.judge_receipt_sha256.as_deref().ok_or_else(|| {
                    format!(
                        "completed condition {} requires judge_receipt_sha256",
                        condition.condition_id
                    )
                })?;
                if !valid_digest(digest, false) {
                    return Err(format!(
                        "completed condition {} has invalid judge receipt digest",
                        condition.condition_id
                    ));
                }
                if observation.decision.is_none()
                    || observation.candidate_a_score_millionths.is_none()
                    || observation.candidate_b_score_millionths.is_none()
                {
                    return Err(format!(
                        "completed condition {} omitted decision or candidate scores",
                        condition.condition_id
                    ));
                }
                if observation
                    .candidate_a_score_millionths
                    .is_some_and(|value| value > 1_000_000)
                    || observation
                        .candidate_b_score_millionths
                        .is_some_and(|value| value > 1_000_000)
                {
                    return Err(format!(
                        "completed condition {} has out-of-range candidate score",
                        condition.condition_id
                    ));
                }
                metric_for_observation(block, condition, observation)?;
            }
            _ => {
                if observation.judge_receipt_sha256.is_some()
                    || observation.decision.is_some()
                    || observation.candidate_a_score_millionths.is_some()
                    || observation.candidate_b_score_millionths.is_some()
                    || !observation.criterion_scores.is_empty()
                {
                    return Err(format!(
                        "ineligible condition {} must not carry a partial outcome",
                        condition.condition_id
                    ));
                }
            }
        }
    }
    let missing: Vec<&str> = expected
        .keys()
        .filter(|id| !observed_ids.contains(**id))
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "receipt is missing {} planned condition observations",
            missing.len()
        ));
    }

    let observations_by_id: BTreeMap<String, &JudgeFactorialObservationV1> = receipt
        .observations
        .iter()
        .map(|observation| (observation.condition_id.clone(), observation))
        .collect();
    let mut metrics = Vec::new();
    for observation in &receipt.observations {
        if observation.status != FactorialExecutionStatusV1::Completed {
            continue;
        }
        let (block, condition, _) = expected.get(observation.condition_id.as_str()).unwrap();
        metrics.push(metric_for_observation(block, condition, observation)?);
    }
    let (balanced_metrics, balanced_repeat_count, excluded_incomplete_repeat_count) =
        select_balanced_factorial_repeats(&plan, &metrics);
    let (main_effects, level_counts, criterion_order_effects) =
        main_effects(&plan, &balanced_metrics, &observations_by_id);
    let interactions = pairwise_interactions(&plan, &balanced_metrics);

    let mut block_summaries = Vec::new();
    let mut completed_total = 0_u64;
    let mut ineligible_total = 0_u64;
    for block in &plan.blocks {
        let completed = block
            .conditions
            .iter()
            .filter(|condition| {
                observations_by_id
                    .get(&condition.condition_id)
                    .is_some_and(|observation| {
                        observation.status == FactorialExecutionStatusV1::Completed
                    })
            })
            .count() as u64;
        let planned = block.conditions.len() as u64;
        let ineligible = planned.saturating_sub(completed);
        completed_total += completed;
        ineligible_total += ineligible;
        block_summaries.push(FactorialBlockSummaryV1 {
            block_id: block.block_id.clone(),
            planned_condition_count: planned,
            completed_condition_count: completed,
            ineligible_condition_count: ineligible,
            fully_completed: completed == planned,
        });
    }
    let planned_count = expected.len() as u64;
    let fields = JudgeFactorialReportFieldsV1 {
        schema: FACTORIAL_REPORT_SCHEMA.to_owned(),
        study_id: plan.study_id,
        analyzer_source_revision: plan.analyzer_source_revision,
        analyzer_worktree_clean: crate::BUILT_GENERATOR_WORKTREE_CLEAN == "true",
        balanced_repeat_count,
        excluded_incomplete_repeat_count,
        effects_use_only_fully_completed_repeats: true,
        plan_sha256,
        receipt_sha256: sha256_hex(receipt_bytes),
        plan_commitment_sha256: plan.plan_commitment_sha256,
        planned_block_count: plan.blocks.len() as u64,
        observed_block_count: receipt.observations.iter().filter_map(|observation| {
            expected.get(observation.condition_id.as_str()).map(|(block, _, _)| block.block_id.clone())
        }).collect::<BTreeSet<_>>().len() as u64,
        planned_condition_count: planned_count,
        observed_condition_count: receipt.observations.len() as u64,
        completed_condition_count: completed_total,
        ineligible_condition_count: ineligible_total,
        all_blocks_present: block_summaries.len() == plan.blocks.len(),
        all_conditions_present: receipt.observations.len() == planned_count,
        all_conditions_completed: completed_total == planned_count,
        level_counts,
        main_effects,
        criterion_order_effects,
        pairwise_interactions: interactions,
        block_summaries,
        execution_authenticity_proven: false,
        randomization_proven: false,
        raw_prompt_custody_proven: false,
        semantic_equivalence_proven: false,
        causal_effect_identified: false,
        generalized_accuracy_estimated: false,
        external_anchor_required: true,
        scoring_eligible_declared: false,
        claim_boundary: "Descriptive contrasts use only fully completed 32-cell block/repeat units; all planned conditions remain in the denominator and incomplete completed cells are separately counted as excluded. Exact plan/input digest binding is checked, but randomization, raw prompt custody, judge execution authenticity, semantic equivalence, population causal effects, generalized accuracy and scoring eligibility are not established.".to_owned(),
    };
    let report_payload = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize factorial report fields: {error}"))?;
    let commitment = digest_domain(REPORT_DOMAIN, &report_payload)?;
    Ok(JudgeFactorialReportV1 {
        fields,
        report_commitment_sha256: commitment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn make_plan() -> (JudgeFactorialPlanV1, Vec<u8>, Vec<u8>) {
        let mut block = JudgeFactorialBlockV1 {
            block_id: "block-001".to_owned(),
            task_semantics_sha256: digest("task semantics"),
            candidate_a_id: "candidate-A".to_owned(),
            candidate_b_id: "candidate-B".to_owned(),
            criterion_ids: vec!["accuracy".to_owned(), "evidence".to_owned()],
            score_options: vec![
                ScoreOptionV1 {
                    option_id: "low".to_owned(),
                    score_millionths: 0,
                },
                ScoreOptionV1 {
                    option_id: "middle".to_owned(),
                    score_millionths: 500_000,
                },
                ScoreOptionV1 {
                    option_id: "high".to_owned(),
                    score_millionths: 1_000_000,
                },
            ],
            execution_order: Vec::new(),
            conditions: Vec::new(),
        };
        for repeat in 0..2 {
            for code in 0..FACTORIAL_CELLS {
                let bits = std::array::from_fn(|index| code & (1 << index) != 0);
                let factors = JudgeNuisanceFactorsV1::from_bits(bits);
                let id = condition_id(&block.block_id, repeat, factors).unwrap();
                block.conditions.push(FactorialConditionV1 {
                    condition_id: id.clone(),
                    repeat_index: repeat,
                    factors,
                    input_sha256: digest(&format!("input-{id}")),
                });
                block.execution_order.push(id);
            }
        }
        block.execution_order.reverse();
        let fields = JudgeFactorialPlanFieldsV1 {
            schema: FACTORIAL_PLAN_SCHEMA.to_owned(),
            analyzer_source_revision: crate::BUILT_GENERATOR_SOURCE_REVISION.to_owned(),
            purpose: FactorialPurposeV1::QualifiedRun,
            study_id: "factorial-example".to_owned(),
            judge_source_revision: "a".repeat(40),
            judge_prompt_sha256: digest("prompt"),
            rubric_sha256: digest("rubric"),
            semantic_task_set_sha256: digest("tasks"),
            repeats_per_condition: 2,
            blocks: vec![block.clone()],
        };
        let commitment = plan_fields_commitment(&fields).unwrap();
        let plan = JudgeFactorialPlanV1 {
            schema: fields.schema.clone(),
            analyzer_source_revision: fields.analyzer_source_revision.clone(),
            purpose: fields.purpose,
            study_id: fields.study_id.clone(),
            judge_source_revision: fields.judge_source_revision.clone(),
            judge_prompt_sha256: fields.judge_prompt_sha256.clone(),
            rubric_sha256: fields.rubric_sha256.clone(),
            semantic_task_set_sha256: fields.semantic_task_set_sha256.clone(),
            repeats_per_condition: fields.repeats_per_condition,
            blocks: fields.blocks.clone(),
            plan_commitment_sha256: commitment,
        };
        let plan_bytes = serde_json::to_vec_pretty(&plan).unwrap();
        let options = &block.score_options;
        let execution_ordinals: BTreeMap<&str, u32> = block
            .execution_order
            .iter()
            .enumerate()
            .map(|(ordinal, condition_id)| (condition_id.as_str(), ordinal as u32))
            .collect();
        let mut observations = Vec::new();
        for condition in &block.conditions {
            let selected = if condition.factors.score_option_order == ScoreOptionOrderV1::Reversed {
                "high"
            } else {
                "low"
            };
            let score = options
                .iter()
                .find(|option| option.option_id == selected)
                .unwrap()
                .score_millionths;
            let mut candidate_a_score =
                if condition.factors.candidate_order == CandidateOrderV1::CandidateAFirst {
                    700_000
                } else {
                    650_000
                };
            if condition.factors.candidate_order == CandidateOrderV1::CandidateBFirst
                && condition.factors.provenance_cue == ProvenanceCueV1::Displayed
            {
                candidate_a_score += 40_000;
            }
            let decision = if condition.factors.candidate_order == CandidateOrderV1::CandidateAFirst
                || condition.factors.provenance_cue == ProvenanceCueV1::Displayed
            {
                FactorialDecisionV1::CandidateA
            } else {
                FactorialDecisionV1::CandidateB
            };
            observations.push(JudgeFactorialObservationV1 {
                condition_id: condition.condition_id.clone(),
                execution_ordinal: *execution_ordinals
                    .get(condition.condition_id.as_str())
                    .unwrap(),
                input_sha256: condition.input_sha256.clone(),
                status: FactorialExecutionStatusV1::Completed,
                judge_receipt_sha256: Some(digest(&format!("receipt-{}", condition.condition_id))),
                decision: Some(decision),
                candidate_a_score_millionths: Some(candidate_a_score),
                candidate_b_score_millionths: Some(600_000),
                criterion_scores: block
                    .criterion_ids
                    .iter()
                    .map(|criterion_id| CriterionOutcomeV1 {
                        criterion_id: criterion_id.clone(),
                        score_millionths: score,
                        selected_option_id: selected.to_owned(),
                    })
                    .collect(),
            });
        }
        let receipt = JudgeFactorialReceiptV1 {
            schema: FACTORIAL_RECEIPT_SCHEMA.to_owned(),
            purpose: FactorialPurposeV1::QualifiedRun,
            analyzer_source_revision: plan.analyzer_source_revision.clone(),
            run_id: "run-001".to_owned(),
            plan_sha256: sha256_hex(&plan_bytes),
            plan_commitment_sha256: plan.plan_commitment_sha256.clone(),
            judge_source_revision: plan.judge_source_revision.clone(),
            judge_prompt_sha256: plan.judge_prompt_sha256.clone(),
            rubric_sha256: plan.rubric_sha256.clone(),
            observations,
        };
        let receipt_bytes = serde_json::to_vec_pretty(&receipt).unwrap();
        (plan, plan_bytes, receipt_bytes)
    }

    #[test]
    fn verifies_complete_32_cell_factorial_with_repeats() {
        let (plan, bytes, _) = make_plan();
        assert_eq!(plan.blocks[0].conditions.len(), 64);
        let (verified, plan_sha) = verify_judge_factorial_plan(&bytes).unwrap();
        assert_eq!(verified.plan_commitment_sha256, plan.plan_commitment_sha256);
        assert_eq!(plan_sha, sha256_hex(&bytes));
    }

    #[test]
    fn computes_expected_candidate_position_and_option_order_effects() {
        let (_, plan, receipt) = make_plan();
        let report = analyze_judge_factorial(&plan, &receipt).unwrap();
        assert_eq!(report.fields.planned_condition_count, 64);
        assert_eq!(report.fields.completed_condition_count, 64);
        let candidate = report
            .fields
            .main_effects
            .iter()
            .find(|effect| effect.factor == JudgeNuisanceFactorV1::CandidateOrder)
            .unwrap();
        assert_eq!(
            candidate.candidate_score_gap_delta_millionths,
            Some(-30_000.0)
        );
        assert_eq!(
            candidate.candidate_a_choice_rate_delta_millionths,
            Some(-500_000.0)
        );
        let option = report
            .fields
            .main_effects
            .iter()
            .find(|effect| effect.factor == JudgeNuisanceFactorV1::ScoreOptionOrder)
            .unwrap();
        assert!(option.criterion_score_delta_millionths.unwrap() > 0.0);
        let provenance = report
            .fields
            .main_effects
            .iter()
            .find(|effect| effect.factor == JudgeNuisanceFactorV1::ProvenanceCue)
            .unwrap();
        assert_eq!(
            provenance.candidate_score_gap_delta_millionths,
            Some(20_000.0)
        );
        assert_eq!(
            provenance.candidate_a_choice_rate_delta_millionths,
            Some(500_000.0)
        );
        let interaction = report
            .fields
            .pairwise_interactions
            .iter()
            .find(|effect| {
                effect.factor_a == JudgeNuisanceFactorV1::CandidateOrder
                    && effect.factor_b == JudgeNuisanceFactorV1::ProvenanceCue
            })
            .unwrap();
        assert_eq!(
            interaction.candidate_score_gap_interaction_millionths,
            Some(40_000.0)
        );
        assert_eq!(
            interaction.candidate_a_choice_interaction_millionths,
            Some(1_000_000.0)
        );
        assert_eq!(
            [
                interaction.decisive_00_count,
                interaction.decisive_01_count,
                interaction.decisive_10_count,
                interaction.decisive_11_count,
            ],
            [16, 16, 16, 16]
        );
    }

    #[test]
    fn reports_ties_and_abstentions_separately_from_decisive_choice_rates() {
        let (_, plan, receipt) = make_plan();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][0]["decision"] = serde_json::json!("tie");
        value["observations"][1]["decision"] = serde_json::json!("abstain");
        let altered = serde_json::to_vec(&value).unwrap();
        let report = analyze_judge_factorial(&plan, &altered).unwrap();

        let candidate_order = report
            .fields
            .main_effects
            .iter()
            .find(|effect| effect.factor == JudgeNuisanceFactorV1::CandidateOrder)
            .unwrap();
        assert_eq!(candidate_order.complete_low_count, 32);
        assert_eq!(candidate_order.complete_high_count, 32);
        assert_eq!(candidate_order.decisive_low_count, 30);
        assert_eq!(candidate_order.decisive_high_count, 32);
        assert_eq!(candidate_order.tie_low_count, 1);
        assert_eq!(candidate_order.tie_high_count, 0);
        assert_eq!(candidate_order.abstention_low_count, 1);
        assert_eq!(candidate_order.abstention_high_count, 0);
        assert_eq!(candidate_order.tie_rate_delta_millionths, Some(-31_250.0));
        assert_eq!(
            candidate_order.abstention_rate_delta_millionths,
            Some(-31_250.0)
        );

        let interaction = report
            .fields
            .pairwise_interactions
            .iter()
            .find(|effect| {
                effect.factor_a == JudgeNuisanceFactorV1::CandidateOrder
                    && effect.factor_b == JudgeNuisanceFactorV1::ProvenanceCue
            })
            .unwrap();
        assert_eq!(interaction.complete_00_count, 16);
        assert_eq!(interaction.decisive_00_count, 14);
        assert_eq!(interaction.tie_00_count, 1);
        assert_eq!(interaction.abstention_00_count, 1);
        assert_eq!(interaction.tie_rate_interaction_millionths, Some(62_500.0));
        assert_eq!(
            interaction.abstention_rate_interaction_millionths,
            Some(62_500.0)
        );
    }

    #[test]
    fn analysis_is_deterministic_for_exact_input_bytes() {
        let (_, plan, receipt) = make_plan();
        let first = serde_json::to_vec(&analyze_judge_factorial(&plan, &receipt).unwrap()).unwrap();
        let second =
            serde_json::to_vec(&analyze_judge_factorial(&plan, &receipt).unwrap()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_plan_committed_for_a_different_analyzer_revision() {
        let (plan, _, _) = make_plan();
        let mut fields = plan.fields();
        fields.analyzer_source_revision = "b".repeat(40);
        let issues = validate_judge_factorial_plan_fields(&fields).unwrap_err();
        assert!(
            issues
                .iter()
                .any(|issue| issue.contains("exact compiled validator revision"))
        );
    }

    #[test]
    fn rejects_receipt_from_a_different_analyzer_revision() {
        let (_, plan, receipt) = make_plan();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["analyzer_source_revision"] = serde_json::json!("b".repeat(40));
        let altered = serde_json::to_vec(&value).unwrap();
        assert!(
            analyze_judge_factorial(&plan, &altered)
                .unwrap_err()
                .contains("exact compiled validator revision")
        );
    }

    #[test]
    fn rejects_criterion_or_score_scale_drift_across_study_blocks() {
        let (plan, _, _) = make_plan();
        let mut fields = plan.fields();
        let mut second = fields.blocks[0].clone();
        second.block_id = "block-002".to_owned();
        let mut remapped_ids = BTreeMap::new();
        for condition in &mut second.conditions {
            let old_id = condition.condition_id.clone();
            let new_id =
                condition_id(&second.block_id, condition.repeat_index, condition.factors).unwrap();
            remapped_ids.insert(old_id, new_id.clone());
            condition.condition_id = new_id;
        }
        for condition_id in &mut second.execution_order {
            *condition_id = remapped_ids.get(condition_id).unwrap().clone();
        }
        second.criterion_ids.push("style".to_owned());
        second.score_options[0].score_millionths = 10_000;
        fields.blocks.push(second);
        let issues = validate_judge_factorial_plan_fields(&fields).unwrap_err();
        assert!(
            issues
                .iter()
                .any(|issue| issue.contains("criterion IDs/order differ"))
        );
        assert!(
            issues
                .iter()
                .any(|issue| issue.contains("score options differ"))
        );
    }

    #[test]
    fn rejects_missing_factorial_cell() {
        let (plan, _, _) = make_plan();
        let mut fields = plan.fields();
        fields.blocks[0].conditions.pop();
        assert!(validate_judge_factorial_plan_fields(&fields).is_err());
    }

    #[test]
    fn rejects_duplicate_cell_within_repeat() {
        let (plan, _, _) = make_plan();
        let mut fields = plan.fields();
        fields.blocks[0].conditions[1].factors = fields.blocks[0].conditions[0].factors;
        assert!(validate_judge_factorial_plan_fields(&fields).is_err());
    }

    #[test]
    fn rejects_changed_execution_order_or_missing_schedule_entry() {
        let (plan, _, _) = make_plan();
        let mut fields = plan.fields();
        fields.blocks[0].execution_order.pop();
        assert!(validate_judge_factorial_plan_fields(&fields).is_err());
    }

    #[test]
    fn rejects_receipt_execution_ordinal_mismatch() {
        let (_, plan, receipt) = make_plan();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][0]["execution_ordinal"] = serde_json::json!(999);
        let altered = serde_json::to_vec(&value).unwrap();
        assert!(
            analyze_judge_factorial(&plan, &altered)
                .unwrap_err()
                .contains("execution_ordinal differs from the committed schedule")
        );
    }

    #[test]
    fn rejects_receipt_input_digest_mismatch() {
        let (_, plan, receipt) = make_plan();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][0]["input_sha256"] = serde_json::json!(digest("wrong input"));
        let altered = serde_json::to_vec(&value).unwrap();
        assert!(
            analyze_judge_factorial(&plan, &altered)
                .unwrap_err()
                .contains("input digest differs")
        );
    }

    #[test]
    fn rejects_missing_and_duplicate_receipt_conditions() {
        let (_, plan, receipt) = make_plan();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"].as_array_mut().unwrap().pop();
        assert!(
            analyze_judge_factorial(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("missing")
        );

        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        let first = value["observations"][0].clone();
        value["observations"].as_array_mut().unwrap().push(first);
        assert!(
            analyze_judge_factorial(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("duplicate")
        );
    }

    #[test]
    fn rejects_semantic_option_score_mismatch() {
        let (_, plan, receipt) = make_plan();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][0]["criterion_scores"][0]["score_millionths"] =
            serde_json::json!(1_000_000);
        assert!(
            analyze_judge_factorial(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("does not match")
        );
    }

    #[test]
    fn incomplete_trials_remain_in_denominator_and_are_not_imputed() {
        let (_, plan, receipt) = make_plan();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][0]["status"] = serde_json::json!("timed_out");
        value["observations"][0]["judge_receipt_sha256"] = serde_json::Value::Null;
        value["observations"][0]["decision"] = serde_json::Value::Null;
        value["observations"][0]["candidate_a_score_millionths"] = serde_json::Value::Null;
        value["observations"][0]["candidate_b_score_millionths"] = serde_json::Value::Null;
        value["observations"][0]["criterion_scores"] = serde_json::json!([]);
        let report = analyze_judge_factorial(&plan, &serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(report.fields.planned_condition_count, 64);
        assert_eq!(report.fields.ineligible_condition_count, 1);
        assert_eq!(report.fields.balanced_repeat_count, 1);
        assert_eq!(report.fields.excluded_incomplete_repeat_count, 1);
        assert!(report.fields.effects_use_only_fully_completed_repeats);
        assert!(!report.fields.all_conditions_completed);
        let candidate = report
            .fields
            .main_effects
            .iter()
            .find(|effect| effect.factor == JudgeNuisanceFactorV1::CandidateOrder)
            .unwrap();
        assert_eq!(candidate.complete_low_count, 16);
        assert_eq!(candidate.complete_high_count, 16);
        assert_eq!(candidate.excluded_completed_low_count, 15);
        assert_eq!(candidate.excluded_completed_high_count, 16);
    }

    #[test]
    fn rejects_partial_output_on_ineligible_condition() {
        let (_, plan, receipt) = make_plan();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["observations"][0]["status"] = serde_json::json!("blocked");
        assert!(
            analyze_judge_factorial(&plan, &serde_json::to_vec(&value).unwrap())
                .unwrap_err()
                .contains("partial outcome")
        );
    }
}
