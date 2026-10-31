//! Randomized access-policy comparisons for search-time contamination.
//!
//! Independent run blocks, not questions, are the statistical units. Each block
//! runs the same frozen checkpoint and challenge packet under network-disabled,
//! allowlisted and unrestricted access, in a seed-randomized order. Every arm
//! must use an isolated process, agent-memory namespace and cache namespace.
//!
//! The module estimates access-policy effects on score and observed retrieval
//! findings. It does not identify the causal effect of a specific retrieved
//! page, and it cannot authenticate the enforcement perimeter or reset receipts.

use crate::{
    fresh_scoring::{FreshScoreReportV1, verify_fresh_score_report_integrity},
    hash_domain,
    search_time::{
        SearchAccessModeV1, SearchTimeExposureTraceV1,
        SearchTimeExposureFindingsV1, audit_search_time_trace,
        audit_search_time_findings,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SEARCH_TRIAL_STUDY_SCHEMA: &str =
    "symthaea.search_time_access_trial_study.v1";
pub const SEARCH_TRIAL_REPORT_SCHEMA: &str =
    "symthaea.search_time_access_trial_report.v1";
pub const SEARCH_TRIAL_MIN_BLOCKS: usize = 20;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SearchTrialConditionV1 {
    NetworkDisabled,
    Allowlisted,
    Unrestricted,
}

impl SearchTrialConditionV1 {
    fn access_mode(self) -> SearchAccessModeV1 {
        match self {
            Self::NetworkDisabled => SearchAccessModeV1::NetworkDisabled,
            Self::Allowlisted => SearchAccessModeV1::Allowlisted,
            Self::Unrestricted => SearchAccessModeV1::Unrestricted,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchTrialRunStatusV1 {
    Completed,
    Partial,
    Failed,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTrialBudgetV1 {
    pub max_model_calls: u64,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
    pub max_wall_clock_millis: u64,
    pub max_tool_calls: u64,
    pub max_human_assistance_minutes: u64,
    pub max_compute_cost_microusd: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTrialConditionRunV1 {
    pub condition: SearchTrialConditionV1,
    pub order_position: u8,
    pub status: SearchTrialRunStatusV1,
    pub run_id: String,
    /// Fresh isolated execution namespaces; values must not be reused across arms.
    pub sandbox_namespace_sha256: String,
    pub agent_memory_namespace_sha256: String,
    pub cache_namespace_sha256: String,
    pub state_reset_receipt_sha256: String,
    pub budget_spec_sha256: String,
    pub run_execution_receipt_sha256: String,
    pub condition_policy_sha256: String,
    pub trace: SearchTimeExposureTraceV1,
    pub findings: SearchTimeExposureFindingsV1,
    pub score_report: FreshScoreReportV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTrialBlockV1 {
    pub block_id: String,
    pub randomized_condition_order: Vec<SearchTrialConditionV1>,
    pub order_seed_commitment_sha256: String,
    pub runs: Vec<SearchTrialConditionRunV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTimeAccessTrialStudyV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub model_checkpoint_sha256: String,
    pub challenge_packet_sha256: String,
    pub inference_harness_sha256: String,
    pub prompt_and_decoding_sha256: String,
    pub metric_definition_sha256: String,
    pub enforcement_build_sha256: String,
    pub minimum_confirmatory_blocks: u32,
    /// Evaluator-only root seed. The public report contains only its commitment.
    pub run_order_seed_hex: String,
    pub budget: SearchTrialBudgetV1,
    pub blocks: Vec<SearchTrialBlockV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTrialIntervalV1 {
    pub mean: f64,
    pub standard_deviation_across_blocks: f64,
    pub standard_error: f64,
    pub confidence_level_permille: u16,
    pub lower_95: f64,
    pub upper_95: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTrialConditionMeanV1 {
    pub condition: SearchTrialConditionV1,
    pub accuracy_over_all_permille: f64,
    pub coverage_permille: f64,
    pub metadata_only_findings_per_run: f64,
    pub context_bearing_findings_per_run: f64,
    pub answer_bearing_findings_per_run: f64,
    pub unknown_findings_per_run: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTrialBlockEffectV1 {
    pub block_id: String,
    pub allowlisted_minus_disabled_accuracy_permille: f64,
    pub unrestricted_minus_disabled_accuracy_permille: f64,
    pub unrestricted_minus_allowlisted_accuracy_permille: f64,
    pub allowlisted_minus_disabled_coverage_permille: f64,
    pub unrestricted_minus_disabled_coverage_permille: f64,
    pub allowlisted_answer_bearing_minus_disabled: i64,
    pub unrestricted_answer_bearing_minus_disabled: i64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTimeAccessTrialReportV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub model_checkpoint_sha256: String,
    pub challenge_packet_sha256: String,
    pub study_sha256: String,
    pub run_order_seed_commitment_sha256: String,
    pub independent_block_count: u64,
    pub minimum_confirmatory_blocks: u32,
    pub block_count_meets_minimum: bool,
    pub every_run_completed_with_matching_budgets: bool,
    pub every_trace_integrity_valid: bool,
    pub every_finding_report_bound_to_trace: bool,
    pub run_state_namespace_identifiers_unique: bool,
    pub run_state_reset_receipts_present: bool,
    pub run_state_isolation_independently_verified: bool,
    pub condition_means: Vec<SearchTrialConditionMeanV1>,
    pub allowlisted_minus_disabled_accuracy_permille: SearchTrialIntervalV1,
    pub unrestricted_minus_disabled_accuracy_permille: SearchTrialIntervalV1,
    pub unrestricted_minus_allowlisted_accuracy_permille: SearchTrialIntervalV1,
    pub allowlisted_minus_disabled_coverage_permille: SearchTrialIntervalV1,
    pub unrestricted_minus_disabled_coverage_permille: SearchTrialIntervalV1,
    pub allowlisted_answer_bearing_minus_disabled: SearchTrialIntervalV1,
    pub unrestricted_answer_bearing_minus_disabled: SearchTrialIntervalV1,
    pub block_effects: Vec<SearchTrialBlockEffectV1>,
    pub complete_mediation_verified: bool,
    pub independently_verified: bool,
    pub confirmatory_eligible: bool,
    pub analysis_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn decode_seed_hex(value: &str) -> Result<Vec<u8>, String> {
    if value.len() < 64 || value.len() % 2 != 0 {
        return Err("run-order seed must contain at least 32 bytes of lowercase hex".to_owned());
    }
    let mut output = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let digit = |byte: u8| match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        };
        let high = digit(pair[0]).ok_or_else(|| "run-order seed must be lowercase hex".to_owned())?;
        let low = digit(pair[1]).ok_or_else(|| "run-order seed must be lowercase hex".to_owned())?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn budget_digest(budget: &SearchTrialBudgetV1) -> Result<String, String> {
    let bytes = serde_json::to_vec(budget)
        .map_err(|error| format!("cannot serialize search trial budget: {error}"))?;
    Ok(hash_domain(b"search-time-trial-budget.v1", &bytes))
}

fn expected_condition_order(
    seed: &[u8],
    block_id: &str,
) -> Result<Vec<SearchTrialConditionV1>, String> {
    let mut ranked = [
        SearchTrialConditionV1::NetworkDisabled,
        SearchTrialConditionV1::Allowlisted,
        SearchTrialConditionV1::Unrestricted,
    ].into_iter().map(|condition| {
        let bytes = serde_json::to_vec(&condition)
            .map_err(|error| format!("cannot serialize search condition: {error}"))?;
        let mut material = Vec::new();
        material.extend_from_slice(&(block_id.len() as u64).to_be_bytes());
        material.extend_from_slice(block_id.as_bytes());
        material.extend_from_slice(&bytes);
        Ok((hash_domain(b"search-time-condition-order.v1", &[seed, material.as_slice()].concat()), condition))
    }).collect::<Result<Vec<_>, String>>()?;
    ranked.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(ranked.into_iter().map(|(_, condition)| condition).collect())
}

fn order_commitment(
    seed: &[u8],
    block_id: &str,
    order: &[SearchTrialConditionV1],
) -> Result<String, String> {
    let bytes = serde_json::to_vec(order)
        .map_err(|error| format!("cannot serialize randomized condition order: {error}"))?;
    let mut material = Vec::new();
    material.extend_from_slice(&(block_id.len() as u64).to_be_bytes());
    material.extend_from_slice(block_id.as_bytes());
    material.extend_from_slice(&bytes);
    Ok(hash_domain(b"search-time-condition-order-commitment.v1", &[seed, material.as_slice()].concat()))
}

fn mean(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("cannot compute mean of empty block outcomes".to_owned());
    }
    let result = values.iter().sum::<f64>() / values.len() as f64;
    if result.is_finite() {
        Ok(result)
    } else {
        Err("search trial mean is non-finite".to_owned())
    }
}

fn t_critical_975(df: usize) -> f64 {
    const TABLE: &[(usize, f64)] = &[
        (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571),
        (6, 2.447), (7, 2.365), (8, 2.306), (9, 2.262), (10, 2.228),
        (11, 2.201), (12, 2.179), (13, 2.160), (14, 2.145), (15, 2.131),
        (16, 2.120), (17, 2.110), (18, 2.101), (19, 2.093), (20, 2.086),
        (21, 2.080), (22, 2.074), (23, 2.069), (24, 2.064), (25, 2.060),
        (26, 2.056), (27, 2.052), (28, 2.048), (29, 2.045), (30, 2.042),
        (40, 2.021), (60, 2.000), (120, 1.980), (usize::MAX, 1.960),
    ];
    TABLE.iter()
        .find(|(upper_df, _)| df <= *upper_df)
        .map(|(_, value)| *value)
        .unwrap_or(1.96)
}

fn interval(values: &[f64]) -> Result<SearchTrialIntervalV1, String> {
    if values.len() < 2 {
        return Err("at least two independent run blocks are required for intervals".to_owned());
    }
    let center = mean(values)?;
    let sum = values.iter().map(|value| (value - center).powi(2)).sum::<f64>();
    let sd = (sum / (values.len() - 1) as f64).sqrt();
    let se = sd / (values.len() as f64).sqrt();
    let margin = t_critical_975(values.len() - 1) * se;
    Ok(SearchTrialIntervalV1 {
        mean: center,
        standard_deviation_across_blocks: sd,
        standard_error: se,
        confidence_level_permille: 950,
        lower_95: center - margin,
        upper_95: center + margin,
    })
}

fn within_budget(run: &SearchTrialConditionRunV1, budget: &SearchTrialBudgetV1) -> bool {
    let report = &run.score_report;
    report.model_call_count <= budget.max_model_calls
        && report.input_token_count <= budget.max_input_tokens
        && report.output_token_count <= budget.max_output_tokens
        && report.wall_clock_millis <= budget.max_wall_clock_millis
        && report.tool_call_count <= budget.max_tool_calls
        && report.human_assistance_minutes <= budget.max_human_assistance_minutes
        && report.compute_cost_microusd <= budget.max_compute_cost_microusd
}

fn validate_run(
    study: &SearchTimeAccessTrialStudyV1,
    run: &SearchTrialConditionRunV1,
    expected_budget_sha256: &str,
    run_ids: &mut BTreeSet<String>,
    namespaces: &mut BTreeSet<String>,
    execution_receipts: &mut BTreeSet<String>,
    issues: &mut Vec<String>,
) -> Result<(), String> {
    if run.status != SearchTrialRunStatusV1::Completed {
        return Err(format!("run {} is not completed; incomplete blocks are inconclusive", run.run_id));
    }
    if run.run_id.trim().is_empty() || !run_ids.insert(run.run_id.clone()) {
        return Err("search trial run IDs must be unique and nonempty".to_owned());
    }
    if run.order_position == 0 || run.order_position > 3 {
        return Err(format!("run {} order position must be 1..=3", run.run_id));
    }
    for (digest, label) in [
        (&run.sandbox_namespace_sha256, "sandbox_namespace_sha256"),
        (&run.agent_memory_namespace_sha256, "agent_memory_namespace_sha256"),
        (&run.cache_namespace_sha256, "cache_namespace_sha256"),
        (&run.state_reset_receipt_sha256, "state_reset_receipt_sha256"),
        (&run.budget_spec_sha256, "budget_spec_sha256"),
        (&run.run_execution_receipt_sha256, "run_execution_receipt_sha256"),
    ] {
        validate_digest(digest, label)?;
    }
    for (digest, label) in [
        (&run.sandbox_namespace_sha256, "sandbox_namespace_sha256"),
        (&run.agent_memory_namespace_sha256, "agent_memory_namespace_sha256"),
        (&run.cache_namespace_sha256, "cache_namespace_sha256"),
        (&run.state_reset_receipt_sha256, "state_reset_receipt_sha256"),
    ] {
        if !namespaces.insert(digest.clone()) {
            return Err(format!("{label} must be unique across run arms and blocks"));
        }
    }
    if !execution_receipts.insert(run.run_execution_receipt_sha256.clone()) {
        return Err("run execution receipt digests must be unique across run arms and blocks".to_owned());
    }
    if run.budget_spec_sha256 != expected_budget_sha256 {
        return Err(format!("run {} budget specification does not match study", run.run_id));
    }
    if !within_budget(run, &study.budget) {
        return Err(format!("run {} exceeded a declared inference budget", run.run_id));
    }
    validate_digest(&run.condition_policy_sha256, "condition_policy_sha256")?;
    if run.trace.network_policy_sha256 != run.condition_policy_sha256
        || run.trace.run_id != run.run_id
        || run.trace.benchmark_id != study.benchmark_id
        || run.trace.benchmark_version != study.benchmark_version
        || run.trace.model_checkpoint_sha256 != study.model_checkpoint_sha256
        || run.trace.challenge_packet_sha256 != study.challenge_packet_sha256
        || run.trace.enforcement_build_sha256 != study.enforcement_build_sha256
        || run.trace.access_mode != run.condition.access_mode()
    {
        return Err(format!("run {} search trace does not match the declared condition/study", run.run_id));
    }
    let trace_audit = audit_search_time_trace(&run.trace)?;
    if !trace_audit.valid {
        return Err(format!("run {} search-time trace is invalid", run.run_id));
    }
    if run.score_report.run_id != run.run_id
        || run.score_report.benchmark_id != study.benchmark_id
        || run.score_report.benchmark_version != study.benchmark_version
        || run.score_report.model_checkpoint_sha256 != study.model_checkpoint_sha256
        || run.score_report.public_packet_sha256 != study.challenge_packet_sha256
        || run.score_report.inference_harness_sha256 != study.inference_harness_sha256
        || run.score_report.prompt_and_decoding_sha256 != study.prompt_and_decoding_sha256
        || run.score_report.metric_definition_sha256 != study.metric_definition_sha256
        || run.score_report.run_execution_receipt_sha256 != run.run_execution_receipt_sha256
    {
        return Err(format!("run {} score report does not match declared run/study identities", run.run_id));
    }
    if run.score_report.independently_verified {
        return Err("self-asserted independent score verification cannot be trusted".to_owned());
    }
    verify_fresh_score_report_integrity(&run.score_report)?;
    let finding_audit = audit_search_time_findings(&run.trace, &run.findings)?;
    if !finding_audit.valid {
        return Err(format!("run {} search-time findings are not bound to the trace", run.run_id));
    }
    if run.findings.run_id != run.run_id {
        return Err(format!("run {} findings refer to another run", run.run_id));
    }
    if run.findings.findings.iter().any(|finding| finding.event_sequence == 0) {
        return Err(format!("run {} findings contain an invalid event sequence", run.run_id));
    }
    if !trace_audit.all_known_channel_types_accounted_for {
        issues.push(format!("run {} has incomplete information-channel declarations", run.run_id));
    }
    Ok(())
}

/// Analyze a randomized three-condition, blocked search-time exposure study.
/// Conditions are compared within independent blocks; item rows remain nested.
pub fn analyze_search_time_access_trial(
    study: &SearchTimeAccessTrialStudyV1,
) -> Result<SearchTimeAccessTrialReportV1, String> {
    if study.schema != SEARCH_TRIAL_STUDY_SCHEMA {
        return Err(format!("study schema must be {SEARCH_TRIAL_STUDY_SCHEMA}"));
    }
    if study.study_id.trim().is_empty()
        || study.benchmark_id.trim().is_empty()
        || study.benchmark_version.trim().is_empty()
        || study.blocks.len() < 2
        || study.blocks.len() > 10_000
        || study.minimum_confirmatory_blocks < SEARCH_TRIAL_MIN_BLOCKS as u32
    {
        return Err("study identity, block count or minimum confirmatory block count is invalid".to_owned());
    }
    if study.budget.max_model_calls == 0
        || study.budget.max_wall_clock_millis == 0
    {
        return Err("trial model-call and wall-clock budgets must be positive".to_owned());
    }
    for (digest, label) in [
        (&study.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&study.challenge_packet_sha256, "challenge_packet_sha256"),
        (&study.inference_harness_sha256, "inference_harness_sha256"),
        (&study.prompt_and_decoding_sha256, "prompt_and_decoding_sha256"),
        (&study.metric_definition_sha256, "metric_definition_sha256"),
        (&study.enforcement_build_sha256, "enforcement_build_sha256"),
    ] {
        validate_digest(digest, label)?;
    }
    let seed = decode_seed_hex(&study.run_order_seed_hex)?;
    let seed_commitment = hash_domain(b"search-time-trial-order-seed.v1", &seed);
    let budget_sha256 = budget_digest(&study.budget)?;

    let mut block_ids = BTreeSet::new();
    let mut run_ids = BTreeSet::new();
    let mut namespaces = BTreeSet::new();
    let mut execution_receipts = BTreeSet::new();
    let mut model_identity: Option<(String, String, String, String)> = None;
    let mut every_run_completed = true;
    let mut every_trace_valid = true;
    let mut every_findings_bound = true;
    let mut condition_score = [
        Vec::<f64>::new(),
        Vec::<f64>::new(),
        Vec::<f64>::new(),
    ];
    let mut condition_coverage = [
        Vec::<f64>::new(),
        Vec::<f64>::new(),
        Vec::<f64>::new(),
    ];
    let mut condition_metadata = [Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()];
    let mut condition_context = [Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()];
    let mut condition_answer = [Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()];
    let mut condition_unknown = [Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()];
    let mut allowlisted_disabled_accuracy = Vec::new();
    let mut unrestricted_disabled_accuracy = Vec::new();
    let mut unrestricted_allowlisted_accuracy = Vec::new();
    let mut allowlisted_disabled_coverage = Vec::new();
    let mut unrestricted_disabled_coverage = Vec::new();
    let mut allowlisted_answer_diff = Vec::new();
    let mut unrestricted_answer_diff = Vec::new();
    let mut block_effects = Vec::new();
    let mut issues = Vec::new();

    for block in &study.blocks {
        if block.block_id.trim().is_empty() || !block_ids.insert(block.block_id.as_str()) {
            return Err("search trial block IDs must be unique and nonempty".to_owned());
        }
        if block.runs.len() != 3 {
            return Err(format!("block {} must contain exactly three condition runs", block.block_id));
        }
        let expected_order = expected_condition_order(&seed, &block.block_id)?;
        let expected_commitment = order_commitment(&seed, &block.block_id, &expected_order)?;
        if block.randomized_condition_order != expected_order
            || block.order_seed_commitment_sha256 != expected_commitment
        {
            return Err(format!("block {} condition order does not match evaluator seed", block.block_id));
        }
        let mut condition_runs = BTreeMap::new();
        for run in &block.runs {
            validate_run(
                study,
                run,
                &budget_sha256,
                &mut run_ids,
                &mut namespaces,
                &mut execution_receipts,
                &mut issues,
            )?;
            let identity = (
                run.score_report.model_provider.clone(),
                run.score_report.model_name.clone(),
                run.score_report.model_version.clone(),
                run.score_report.model_access_mode.clone(),
            );
            if let Some(expected_identity) = &model_identity {
                if &identity != expected_identity {
                    return Err("model provider/name/version/access mode must be identical across all trial runs".to_owned());
                }
            } else {
                model_identity = Some(identity);
            }
            if condition_runs.insert(run.condition, run).is_some() {
                return Err(format!("block {} duplicates condition {:?}", block.block_id, run.condition));
            }
            let expected_position = block.randomized_condition_order.iter()
                .position(|condition| *condition == run.condition)
                .ok_or_else(|| format!("block {} randomized order omits condition", block.block_id))?;
            if usize::from(run.order_position) != expected_position + 1 {
                return Err(format!("run {} order position disagrees with randomized order", run.run_id));
            }
        }
        let offline = condition_runs.get(&SearchTrialConditionV1::NetworkDisabled)
            .ok_or_else(|| format!("block {} missing network-disabled arm", block.block_id))?;
        let allowlisted = condition_runs.get(&SearchTrialConditionV1::Allowlisted)
            .ok_or_else(|| format!("block {} missing allowlisted arm", block.block_id))?;
        let unrestricted = condition_runs.get(&SearchTrialConditionV1::Unrestricted)
            .ok_or_else(|| format!("block {} missing unrestricted arm", block.block_id))?;
        let runs = [*offline, *allowlisted, *unrestricted];
        for (index, run) in runs.iter().enumerate() {
            let finding_audit = audit_search_time_findings(&run.trace, &run.findings)?;
            let condition = index;
            condition_score[condition].push(run.score_report.accuracy_over_all_permille);
            condition_coverage[condition].push(run.score_report.coverage_permille);
            condition_metadata[condition].push(finding_audit.metadata_only_count as f64);
            condition_context[condition].push(finding_audit.context_bearing_count as f64);
            condition_answer[condition].push(finding_audit.answer_bearing_count as f64);
            condition_unknown[condition].push(finding_audit.unknown_count as f64);
        }
        let offline_answer_count =
            audit_search_time_findings(&offline.trace, &offline.findings)?.answer_bearing_count;
        let allowlisted_answer_count =
            audit_search_time_findings(&allowlisted.trace, &allowlisted.findings)?.answer_bearing_count;
        let unrestricted_answer_count =
            audit_search_time_findings(&unrestricted.trace, &unrestricted.findings)?.answer_bearing_count;
        let offline_answer = offline_answer_count as f64;
        let allowlisted_answer = allowlisted_answer_count as f64;
        let unrestricted_answer = unrestricted_answer_count as f64;
        let answer_count_difference = |left: u64, right: u64| -> Result<i64, String> {
            let left = i64::try_from(left).map_err(|_| "answer-finding count exceeds i64")?;
            let right = i64::try_from(right).map_err(|_| "answer-finding count exceeds i64")?;
            left.checked_sub(right).ok_or_else(|| "answer-finding difference overflows i64".to_owned())
        };
        allowlisted_disabled_accuracy.push(
            allowlisted.score_report.accuracy_over_all_permille
                - offline.score_report.accuracy_over_all_permille
        );
        unrestricted_disabled_accuracy.push(
            unrestricted.score_report.accuracy_over_all_permille
                - offline.score_report.accuracy_over_all_permille
        );
        unrestricted_allowlisted_accuracy.push(
            unrestricted.score_report.accuracy_over_all_permille
                - allowlisted.score_report.accuracy_over_all_permille
        );
        allowlisted_disabled_coverage.push(
            allowlisted.score_report.coverage_permille - offline.score_report.coverage_permille
        );
        unrestricted_disabled_coverage.push(
            unrestricted.score_report.coverage_permille - offline.score_report.coverage_permille
        );
        allowlisted_answer_diff.push(allowlisted_answer - offline_answer);
        unrestricted_answer_diff.push(unrestricted_answer - offline_answer);
        block_effects.push(SearchTrialBlockEffectV1 {
            block_id: block.block_id.clone(),
            allowlisted_minus_disabled_accuracy_permille: allowlisted.score_report.accuracy_over_all_permille - offline.score_report.accuracy_over_all_permille,
            unrestricted_minus_disabled_accuracy_permille: unrestricted.score_report.accuracy_over_all_permille - offline.score_report.accuracy_over_all_permille,
            unrestricted_minus_allowlisted_accuracy_permille: unrestricted.score_report.accuracy_over_all_permille - allowlisted.score_report.accuracy_over_all_permille,
            allowlisted_minus_disabled_coverage_permille: allowlisted.score_report.coverage_permille - offline.score_report.coverage_permille,
            unrestricted_minus_disabled_coverage_permille: unrestricted.score_report.coverage_permille - offline.score_report.coverage_permille,
            allowlisted_answer_bearing_minus_disabled: answer_count_difference(allowlisted_answer_count, offline_answer_count)?,
            unrestricted_answer_bearing_minus_disabled: answer_count_difference(unrestricted_answer_count, offline_answer_count)?,
        });
        every_run_completed &= runs.iter().all(|run| run.status == SearchTrialRunStatusV1::Completed);
        every_trace_valid &= runs.iter().all(|run| audit_search_time_trace(&run.trace).map(|a| a.valid).unwrap_or(false));
        every_findings_bound &= runs.iter().all(|run| audit_search_time_findings(&run.trace, &run.findings).map(|a| a.valid).unwrap_or(false));
    }

    let condition_mean = |condition: SearchTrialConditionV1, index: usize| -> Result<SearchTrialConditionMeanV1, String> {
        Ok(SearchTrialConditionMeanV1 {
            condition,
            accuracy_over_all_permille: mean(&condition_score[index])?,
            coverage_permille: mean(&condition_coverage[index])?,
            metadata_only_findings_per_run: mean(&condition_metadata[index])?,
            context_bearing_findings_per_run: mean(&condition_context[index])?,
            answer_bearing_findings_per_run: mean(&condition_answer[index])?,
            unknown_findings_per_run: mean(&condition_unknown[index])?,
        })
    };
    let condition_means = vec![
        condition_mean(SearchTrialConditionV1::NetworkDisabled, 0)?,
        condition_mean(SearchTrialConditionV1::Allowlisted, 1)?,
        condition_mean(SearchTrialConditionV1::Unrestricted, 2)?,
    ];
    let block_count_meets_minimum = study.blocks.len() >= study.minimum_confirmatory_blocks as usize;
    if !block_count_meets_minimum {
        issues.push(format!(
            "only {} independent run blocks; minimum preregistered count is {}",
            study.blocks.len(),
            study.minimum_confirmatory_blocks
        ));
    }
    let study_bytes = serde_json::to_vec(study)
        .map_err(|error| format!("cannot serialize search trial study: {error}"))?;
    let study_sha256 = hash_domain(b"search-time-access-trial-study.v1", &study_bytes);
    let report = SearchTimeAccessTrialReportV1 {
        schema: SEARCH_TRIAL_REPORT_SCHEMA.to_owned(),
        study_id: study.study_id.clone(),
        benchmark_id: study.benchmark_id.clone(),
        benchmark_version: study.benchmark_version.clone(),
        model_checkpoint_sha256: study.model_checkpoint_sha256.clone(),
        challenge_packet_sha256: study.challenge_packet_sha256.clone(),
        study_sha256,
        run_order_seed_commitment_sha256: seed_commitment,
        independent_block_count: u64::try_from(study.blocks.len())
            .map_err(|_| "block count exceeds u64")?,
        minimum_confirmatory_blocks: study.minimum_confirmatory_blocks,
        block_count_meets_minimum,
        every_run_completed_with_matching_budgets: every_run_completed,
        every_trace_integrity_valid: every_trace_valid,
        every_finding_report_bound_to_trace: every_findings_bound,
        run_state_namespace_identifiers_unique: true,
        run_state_reset_receipts_present: true,
        run_state_isolation_independently_verified: false,
        condition_means,
        allowlisted_minus_disabled_accuracy_permille: interval(&allowlisted_disabled_accuracy)?,
        unrestricted_minus_disabled_accuracy_permille: interval(&unrestricted_disabled_accuracy)?,
        unrestricted_minus_allowlisted_accuracy_permille: interval(&unrestricted_allowlisted_accuracy)?,
        allowlisted_minus_disabled_coverage_permille: interval(&allowlisted_disabled_coverage)?,
        unrestricted_minus_disabled_coverage_permille: interval(&unrestricted_disabled_coverage)?,
        allowlisted_answer_bearing_minus_disabled: interval(&allowlisted_answer_diff)?,
        unrestricted_answer_bearing_minus_disabled: interval(&unrestricted_answer_diff)?,
        block_effects,
        complete_mediation_verified: false,
        independently_verified: false,
        confirmatory_eligible: false,
        analysis_status: if !block_count_meets_minimum {
            "exploratory_insufficient_independent_run_blocks"
        } else {
            "exploratory_waiting_for_independent_runner_and_trace_authentication"
        }.to_owned(),
        issues,
        claim_boundary: "run_block_is_independent_statistical_unit;items_are_nested;access_policy_effect_does_not_identify_specific_retrieval_influence;trace_mediation_and_runner_identity_not_cryptographically_authenticated".to_owned(),
    };
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        hash_domain(b"search-time-causal-test", label.as_bytes())
    }

    fn make_run(
        study: &SearchTimeAccessTrialStudyV1,
        block_id: &str,
        condition: SearchTrialConditionV1,
        order_position: u8,
        suffix: &str,
        accuracy_permille: f64,
        answer_findings: u64,
    ) -> SearchTrialConditionRunV1 {
        let access = condition.access_mode();
        let mut trace = SearchTimeExposureTraceV1 {
            schema: crate::search_time::SEARCH_TRACE_SCHEMA.to_owned(),
            run_id: format!("run-{suffix}"),
            benchmark_id: study.benchmark_id.clone(),
            benchmark_version: study.benchmark_version.clone(),
            model_checkpoint_sha256: study.model_checkpoint_sha256.clone(),
            challenge_packet_sha256: study.challenge_packet_sha256.clone(),
            network_policy_sha256: digest(&format!("policy-{condition:?}")),
            enforcement_build_sha256: study.enforcement_build_sha256.clone(),
            policy_id: format!("policy-{condition:?}"),
            access_mode: access,
            allowlisted_destination_sha256s: if condition == SearchTrialConditionV1::Allowlisted {
                vec![digest("allowlisted-destination")]
            } else {
                Vec::new()
            },
            instrumented_channels: [
                crate::search_time::SearchInformationChannelV1::WebSearch,
                crate::search_time::SearchInformationChannelV1::BrowserNavigation,
                crate::search_time::SearchInformationChannelV1::HttpApi,
                crate::search_time::SearchInformationChannelV1::RemoteMcp,
                crate::search_time::SearchInformationChannelV1::VectorStore,
                crate::search_time::SearchInformationChannelV1::LocalCache,
                crate::search_time::SearchInformationChannelV1::MountedFile,
                crate::search_time::SearchInformationChannelV1::AgentMemory,
                crate::search_time::SearchInformationChannelV1::Other,
            ].to_vec(),
            unobserved_channels: Vec::new(),
            events: vec![],
            event_count: 0,
            trace_root_sha256: String::new(),
            claim_boundary: "causal study fixture".to_owned(),
        };
        crate::search_time::seal_search_time_trace(&mut trace).unwrap();

        let run_id = format!("run-{suffix}");
        let run_execution_receipt_sha256 = digest(&format!("execution-{suffix}"));
        let mut score_report = synthetic_score_report(study, &run_id, accuracy_permille);
        score_report.run_execution_receipt_sha256 = run_execution_receipt_sha256.clone();
        score_report.tool_call_count = if answer_findings > 0 { 1 } else { 0 };
        score_report.report_sha256.clear();
        let score_bytes = serde_json::to_vec(&score_report).unwrap();
        score_report.report_sha256 =
            hash_domain(b"fresh-instance-score-report.v1", &score_bytes);
        let mut findings = SearchTimeExposureFindingsV1 {
            schema: crate::search_time::SEARCH_FINDINGS_SCHEMA.to_owned(),
            run_id: format!("run-{suffix}"),
            trace_root_sha256: trace.trace_root_sha256.clone(),
            findings: (0..answer_findings).map(|index| SearchExposureFindingV1 {
                event_sequence: 1,
                exposure_class: crate::search_time::SearchExposureClassV1::AnswerBearing,
                matched_item_ids: vec![format!("task-{index}")],
                evidence_sha256: digest(&format!("evidence-{suffix}-{index}")),
                method: crate::search_time::SearchFindingMethodV1::HumanAdjudication,
                confidence_permille: 1_000,
            }).collect(),
            findings_sha256: String::new(),
            claim_boundary: "synthetic evaluator findings".to_owned(),
        };
        if answer_findings > 0 {
            let mut event = crate::search_time::SearchTimeEventV1 {
                sequence: 1,
                invocation_id: format!("search-{suffix}"),
                channel: crate::search_time::SearchInformationChannelV1::WebSearch,
                provider_or_tool: "test-search".to_owned(),
                destination_sha256: digest("allowlisted-destination"),
                request_sha256: digest("request"),
                response_sha256: Some(digest("response")),
                response_bytes: 128,
                occurred_at_utc: "2026-10-31T18:00:00Z".to_owned(),
                disposition: crate::search_time::SearchEventDispositionV1::Allowed,
                previous_event_sha256: String::new(),
                event_sha256: String::new(),
            };
            if condition == SearchTrialConditionV1::NetworkDisabled {
                event.disposition = crate::search_time::SearchEventDispositionV1::Blocked;
                event.response_sha256 = None;
                event.response_bytes = 0;
            }
            trace.events.push(event);
            crate::search_time::seal_search_time_trace(&mut trace).unwrap();
            for finding in &mut findings.findings {
                finding.event_sequence = 1;
            }
            findings.trace_root_sha256 = trace.trace_root_sha256.clone();
        }
        let bytes = serde_json::to_vec(&{
            let mut clone = findings.clone();
            clone.findings_sha256.clear();
            clone
        }).unwrap();
        findings.findings_sha256 = hash_domain(b"search-time-findings.v1", &bytes);
        SearchTrialConditionRunV1 {
            condition,
            order_position,
            status: SearchTrialRunStatusV1::Completed,
            run_id,
            sandbox_namespace_sha256: digest(&format!("sandbox-{suffix}")),
            agent_memory_namespace_sha256: digest(&format!("memory-{suffix}")),
            cache_namespace_sha256: digest(&format!("cache-{suffix}")),
            state_reset_receipt_sha256: digest(&format!("reset-{suffix}")),
            budget_spec_sha256: budget_digest(&study.budget).unwrap(),
            run_execution_receipt_sha256,
            condition_policy_sha256: trace.network_policy_sha256.clone(),
            trace,
            findings,
            score_report,
        }
    }

    fn synthetic_score_report(
        study: &SearchTimeAccessTrialStudyV1,
        run_id: &str,
        accuracy_permille: f64,
    ) -> FreshScoreReportV1 {
        FreshScoreReportV1 {
            schema: crate::fresh_scoring::FRESH_SCORE_REPORT_SCHEMA.to_owned(),
            metric_id: crate::fresh_scoring::FRESH_EXACT_METRIC_ID.to_owned(),
            run_id: run_id.to_owned(),
            model_provider: "test-provider".to_owned(),
            model_name: "fixture-model".to_owned(),
            model_version: "v1".to_owned(),
            model_access_mode: "offline".to_owned(),
            model_checkpoint_sha256: study.model_checkpoint_sha256.clone(),
            batch_id: "fresh-batch".to_owned(),
            benchmark_id: study.benchmark_id.clone(),
            benchmark_version: study.benchmark_version.clone(),
            task_family: "linear-systems".to_owned(),
            public_packet_sha256: study.challenge_packet_sha256.clone(),
            sealed_oracle_sha256: digest("oracle"),
            manifest_sha256: digest("manifest"),
            submission_sha256: digest(&format!("submission-{run_id}")),
            inference_harness_sha256: study.inference_harness_sha256.clone(),
            prompt_and_decoding_sha256: study.prompt_and_decoding_sha256.clone(),
            metric_definition_sha256: study.metric_definition_sha256.clone(),
            run_execution_receipt_sha256: digest(&format!("execution-{run_id}")),
            item_count: 100,
            answered_count: 100,
            correct_count: accuracy_permille as u64 / 10,
            incorrect_count: 100 - accuracy_permille as u64 / 10,
            invalid_format_count: 0,
            abstained_count: 0,
            valid_format_response_count: 100,
            accuracy_over_all_permille: accuracy_permille,
            coverage_permille: 1_000.0,
            accuracy_when_answered_permille: Some(accuracy_permille),
            accuracy_when_valid_format_permille: Some(accuracy_permille),
            model_call_count: 100,
            input_token_count: 1_000,
            output_token_count: 2_000,
            wall_clock_millis: 10_000,
            tool_call_count: 1,
            human_assistance_minutes: 0,
            compute_cost_microusd: 0,
            execution_receipt_verification: "digest_present_signature_not_cryptographically_verified".to_owned(),
            independently_verified: false,
            items: (0..100).map(|index| crate::fresh_scoring::FreshItemScoreV1 {
                item_id: format!("item-{index}"),
                response_sha256: digest(&format!("response-{run_id}-{index}")),
                status: if index < accuracy_permille as usize / 10 {
                    crate::fresh_scoring::FreshItemScoreStatusV1::Correct
                } else {
                    crate::fresh_scoring::FreshItemScoreStatusV1::Incorrect
                },
            }).collect(),
            report_sha256: String::new(),
            claim_boundary: "synthetic score".to_owned(),
        }
    }

    fn fixture_study() -> SearchTimeAccessTrialStudyV1 {
        let mut study = SearchTimeAccessTrialStudyV1 {
            schema: SEARCH_TRIAL_STUDY_SCHEMA.to_owned(),
            study_id: "search-time-study-fixture".to_owned(),
            benchmark_id: "fresh-benchmark".to_owned(),
            benchmark_version: "v1".to_owned(),
            model_checkpoint_sha256: digest("checkpoint"),
            challenge_packet_sha256: digest("challenge"),
            inference_harness_sha256: digest("harness"),
            prompt_and_decoding_sha256: digest("prompt"),
            metric_definition_sha256: digest("metric"),
            enforcement_build_sha256: digest("enforcer"),
            minimum_confirmatory_blocks: SEARCH_TRIAL_MIN_BLOCKS as u32,
            run_order_seed_hex: "ab".repeat(32),
            budget: SearchTrialBudgetV1 {
                max_model_calls: 100,
                max_input_tokens: 1_000,
                max_output_tokens: 2_000,
                max_wall_clock_millis: 10_000,
                max_tool_calls: 10,
                max_human_assistance_minutes: 0,
                max_compute_cost_microusd: 0,
            },
            blocks: Vec::new(),
        };
        let seed = decode_seed_hex(&study.run_order_seed_hex).unwrap();
        let block_id = "block-001";
        let order = expected_condition_order(&seed, block_id).unwrap();
        let commitment = order_commitment(&seed, block_id, &order).unwrap();
        let mut runs = Vec::new();
        for (position, condition) in order.iter().enumerate() {
            let suffix = format!("{}-{condition:?}", block_id);
            let accuracy = match condition {
                SearchTrialConditionV1::NetworkDisabled => 700.0,
                SearchTrialConditionV1::Allowlisted => 730.0,
                SearchTrialConditionV1::Unrestricted => 760.0,
            };
            let answers = if *condition == SearchTrialConditionV1::NetworkDisabled { 0 } else { 1 };
            runs.push(make_run(
                &study,
                block_id,
                *condition,
                (position + 1) as u8,
                &suffix,
                accuracy,
                answers,
            ));
        }
        study.blocks.push(SearchTrialBlockV1 {
            block_id: block_id.to_owned(),
            randomized_condition_order: order,
            order_seed_commitment_sha256: commitment,
            runs,
        });
        study
    }

    fn two_block_fixture_study() -> SearchTimeAccessTrialStudyV1 {
        let mut study = fixture_study();
        let source_block = study.blocks[0].clone();
        let block_id = "block-002".to_owned();
        let seed = decode_seed_hex(&study.run_order_seed_hex).unwrap();
        let order = expected_condition_order(&seed, &block_id).unwrap();
        let commitment = order_commitment(&seed, &block_id, &order).unwrap();
        let mut runs = Vec::new();
        for condition in &order {
            let source = source_block.runs.iter()
                .find(|run| run.condition == *condition).unwrap();
            let mut run = source.clone();
            let suffix = format!("{block_id}-{condition:?}");
            run.run_id = format!("run-{suffix}");
            run.order_position = (order.iter().position(|candidate| candidate == condition).unwrap() + 1) as u8;
            run.sandbox_namespace_sha256 = digest(&format!("sandbox-{suffix}"));
            run.agent_memory_namespace_sha256 = digest(&format!("memory-{suffix}"));
            run.cache_namespace_sha256 = digest(&format!("cache-{suffix}"));
            run.state_reset_receipt_sha256 = digest(&format!("reset-{suffix}"));
            run.run_execution_receipt_sha256 = digest(&format!("execution-{suffix}"));
            run.condition_policy_sha256 = digest(&format!("policy-{suffix}"));
            run.trace.run_id = run.run_id.clone();
            run.trace.network_policy_sha256 = run.condition_policy_sha256.clone();
            for event in &mut run.trace.events {
                event.invocation_id = format!("{}-{}", suffix, event.sequence);
            }
            crate::search_time::seal_search_time_trace(&mut run.trace).unwrap();

            run.findings.run_id = run.run_id.clone();
            run.findings.trace_root_sha256 = run.trace.trace_root_sha256.clone();
            run.findings.findings_sha256.clear();
            let findings_bytes = serde_json::to_vec(&run.findings).unwrap();
            run.findings.findings_sha256 =
                hash_domain(b"search-time-findings.v1", &findings_bytes);

            run.score_report.run_id = run.run_id.clone();
            run.score_report.run_execution_receipt_sha256 =
                run.run_execution_receipt_sha256.clone();
            run.score_report.submission_sha256 = digest(&format!("submission-{suffix}"));
            run.score_report.report_sha256.clear();
            let report_bytes = serde_json::to_vec(&run.score_report).unwrap();
            run.score_report.report_sha256 =
                hash_domain(b"fresh-instance-score-report.v1", &report_bytes);
            runs.push(run);
        }
        study.blocks.push(SearchTrialBlockV1 {
            block_id,
            randomized_condition_order: order,
            order_seed_commitment_sha256: commitment,
            runs,
        });
        study
    }

    #[test]
    fn order_is_deterministic_and_block_scoped() {
        let seed = decode_seed_hex(&"12".repeat(32)).unwrap();
        assert_eq!(
            expected_condition_order(&seed, "a").unwrap(),
            expected_condition_order(&seed, "a").unwrap()
        );
        assert_ne!(
            expected_condition_order(&seed, "a").unwrap(),
            expected_condition_order(&seed, "b").unwrap()
        );
    }

    #[test]
    fn block_budget_must_match_and_condition_order_is_verified() {
        let mut study = two_block_fixture_study();
        study.blocks[0].runs[0].budget_spec_sha256 = digest("wrong-budget");
        assert!(analyze_search_time_access_trial(&study).unwrap_err().contains("budget specification"));
        let mut study = two_block_fixture_study();
        study.blocks[0].randomized_condition_order.swap(0, 1);
        assert!(analyze_search_time_access_trial(&study).unwrap_err().contains("condition order"));
    }

    #[test]
    fn independent_run_namespaces_must_never_be_reused() {
        let mut study = two_block_fixture_study();
        study.blocks[0].runs[1].cache_namespace_sha256 =
            study.blocks[0].runs[0].cache_namespace_sha256.clone();
        assert!(analyze_search_time_access_trial(&study).unwrap_err().contains("unique across run arms"));
    }

    #[test]
    fn model_identity_must_be_constant_across_conditions() {
        let mut study = two_block_fixture_study();
        study.blocks[0].runs[1].score_report.model_name = "different-model".to_owned();
        study.blocks[0].runs[1].score_report.report_sha256.clear();
        let bytes = serde_json::to_vec(&study.blocks[0].runs[1].score_report).unwrap();
        study.blocks[0].runs[1].score_report.report_sha256 =
            hash_domain(b"fresh-instance-score-report.v1", &bytes);
        assert!(analyze_search_time_access_trial(&study)
            .unwrap_err().contains("model provider/name/version/access mode"));
    }

    #[test]
    fn incomplete_or_over_budget_run_is_rejected() {
        let mut study = two_block_fixture_study();
        study.blocks[0].runs[0].status = SearchTrialRunStatusV1::Partial;
        assert!(analyze_search_time_access_trial(&study)
            .unwrap_err().contains("is not completed"));
        let mut study = two_block_fixture_study();
        study.blocks[0].runs[0].score_report.output_token_count =
            study.budget.max_output_tokens + 1;
        study.blocks[0].runs[0].score_report.report_sha256.clear();
        let bytes = serde_json::to_vec(&study.blocks[0].runs[0].score_report).unwrap();
        study.blocks[0].runs[0].score_report.report_sha256 =
            hash_domain(b"fresh-instance-score-report.v1", &bytes);
        assert!(analyze_search_time_access_trial(&study)
            .unwrap_err().contains("exceeded a declared inference budget"));
    }

    #[test]
    fn condition_trace_and_score_bindings_are_enforced() {
        let mut study = two_block_fixture_study();
        study.blocks[0].runs[0].trace.access_mode = SearchAccessModeV1::Unrestricted;
        crate::search_time::seal_search_time_trace(&mut study.blocks[0].runs[0].trace).unwrap();
        assert!(analyze_search_time_access_trial(&study).unwrap_err().contains("does not match the declared condition"));

        let mut study = two_block_fixture_study();
        study.blocks[0].runs[0].score_report.model_checkpoint_sha256 = digest("wrong-checkpoint");
        study.blocks[0].runs[0].score_report.report_sha256.clear();
        assert!(analyze_search_time_access_trial(&study).unwrap_err().contains("score report does not match"));
    }

    #[test]
    fn score_changes_are_summarized_across_independent_run_blocks() {
        let study = two_block_fixture_study();
        let mut expanded = study.clone();
        let block0 = expanded.blocks[0].clone();
        for index in 3..=20 {
            let block_id = format!("block-{index:03}");
            let order = expected_condition_order(
                &decode_seed_hex(&expanded.run_order_seed_hex).unwrap(),
                &block_id,
            ).unwrap();
            let commitment = order_commitment(
                &decode_seed_hex(&expanded.run_order_seed_hex).unwrap(),
                &block_id,
                &order,
            ).unwrap();
            let mut runs = Vec::new();
            for (position, condition) in order.iter().enumerate() {
                let source = block0.runs.iter().find(|run| run.condition == *condition).unwrap();
                let mut run = source.clone();
                let suffix = format!("{block_id}-{condition:?}");
                run.run_id = format!("run-{suffix}");
                run.order_position = (position + 1) as u8;
                run.sandbox_namespace_sha256 = digest(&format!("sandbox-{suffix}"));
                run.agent_memory_namespace_sha256 = digest(&format!("memory-{suffix}"));
                run.cache_namespace_sha256 = digest(&format!("cache-{suffix}"));
                run.state_reset_receipt_sha256 = digest(&format!("reset-{suffix}"));
                run.run_execution_receipt_sha256 = digest(&format!("execution-{suffix}"));
                run.trace.run_id = run.run_id.clone();
                run.trace.network_policy_sha256 = digest(&format!("policy-{}-{:?}", suffix, run.condition));
                run.condition_policy_sha256 = run.trace.network_policy_sha256.clone();
                crate::search_time::seal_search_time_trace(&mut run.trace).unwrap();
                run.findings.run_id = run.run_id.clone();
                run.findings.trace_root_sha256 = run.trace.trace_root_sha256.clone();
                run.findings.findings_sha256.clear();
                let findings_bytes = serde_json::to_vec(&run.findings).unwrap();
                run.findings.findings_sha256 = hash_domain(b"search-time-findings.v1", &findings_bytes);
                run.score_report.run_id = run.run_id.clone();
                run.score_report.run_execution_receipt_sha256 = run.run_execution_receipt_sha256.clone();
                run.score_report.report_sha256.clear();
                let report_bytes = serde_json::to_vec(&run.score_report).unwrap();
                run.score_report.report_sha256 =
                    hash_domain(b"fresh-instance-score-report.v1", &report_bytes);
                runs.push(run);
            }
            expanded.blocks.push(SearchTrialBlockV1 {
                block_id,
                randomized_condition_order: order,
                order_seed_commitment_sha256: commitment,
                runs,
            });
        }
        let report = analyze_search_time_access_trial(&expanded).unwrap();
        assert_eq!(report.independent_block_count, 20);
        assert!(report.block_count_meets_minimum);
        assert!(!report.confirmatory_eligible);
        assert!(!report.complete_mediation_verified);
        assert!(report.allowlisted_minus_disabled_accuracy_permille.mean > 0.0);
        assert!(report.unrestricted_minus_disabled_accuracy_permille.mean > 0.0);
    }

    #[test]
    fn strict_search_trial_schemas_match_serialized_structs() {
        let study_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schema/search-time-access-trial-study.schema.json"
        )).unwrap();
        let report_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schema/search-time-access-trial-report.schema.json"
        )).unwrap();
        let study = fixture_study();
        let study = two_block_fixture_study();
        let report = analyze_search_time_access_trial(&study).unwrap();
        for (schema, value, version) in [
            (&study_schema, serde_json::to_value(&study).unwrap(), SEARCH_TRIAL_STUDY_SCHEMA),
            (&report_schema, serde_json::to_value(&report).unwrap(), SEARCH_TRIAL_REPORT_SCHEMA),
        ] {
            assert_eq!(schema["properties"]["schema"]["const"].as_str(), Some(version));
            let required: BTreeSet<String> = schema["required"].as_array().unwrap()
                .iter().map(|item| item.as_str().unwrap().to_owned()).collect();
            let properties: BTreeSet<String> = schema["properties"].as_object().unwrap()
                .keys().cloned().collect();
            let actual: BTreeSet<String> = value.as_object().unwrap()
                .keys().cloned().collect();
            assert_eq!(required, properties);
            assert_eq!(actual, properties);
        }
    }
}
