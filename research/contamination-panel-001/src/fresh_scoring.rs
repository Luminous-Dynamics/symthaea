//! Exact scoring for secret-seeded fresh inference-time benchmark instances.
//!
//! The scorer receives the public packet, evaluator-only sealed oracle,
//! generation manifest and model response submission. It validates all
//! cross-artifact digests, exact item coverage, checkpoint binding and exact
//! oracle arithmetic before calculating accuracy, coverage and abstention.
//!
//! Exact-match scoring does not establish benchmark construct validity,
//! task-recipe novelty or absence from training data. The execution receipt is
//! self-reported unless a trusted signature verifier authenticates it.

use crate::{
    fresh_instance::{
        ExactLinearSolutionV1, FreshChallengeBundleV1, FreshChallengeManifestV1,
        FreshPublicChallengePacketV1, LinearSystemParametersV1, SealedFreshOraclePacketV1,
        normalized_determinant_margin_permille, parse_timestamp, task_text,
    },
    hash_domain,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FRESH_SUBMISSION_SCHEMA: &str = "symthaea.fresh_instance_submission.v1";
pub const FRESH_SCORE_REPORT_SCHEMA: &str = "symthaea.fresh_instance_score_report.v1";
pub const FRESH_EXACT_METRIC_ID: &str = "fresh-exact-integer-pair.v1";
pub const FRESH_MAX_RESPONSE_BYTES: usize = 1_048_576;
pub const FRESH_MAX_TOTAL_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
pub const FRESH_MAX_SUBMISSION_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshResponseRecordV1 {
    pub item_id: String,
    /// Exact raw model output; empty/whitespace-only is an explicit abstention.
    pub response_text: String,
    pub response_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshAnswerSubmissionV1 {
    pub schema: String,
    pub run_id: String,
    pub model_provider: String,
    pub model_name: String,
    pub model_version: String,
    pub model_access_mode: String,
    pub model_checkpoint_sha256: String,
    pub public_packet_sha256: String,
    pub inference_harness_sha256: String,
    pub prompt_and_decoding_sha256: String,
    pub metric_definition_sha256: String,
    /// Self-reported execution receipt digest; not authenticated here.
    pub run_execution_receipt_sha256: String,
    pub model_call_count: u64,
    pub input_token_count: u64,
    pub output_token_count: u64,
    pub wall_clock_millis: u64,
    pub tool_call_count: u64,
    pub human_assistance_minutes: u64,
    pub compute_cost_microusd: u64,
    pub responses: Vec<FreshResponseRecordV1>,
    pub submission_sha256: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FreshItemScoreStatusV1 {
    Correct,
    Incorrect,
    InvalidFormat,
    Abstained,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshItemScoreV1 {
    pub item_id: String,
    pub response_sha256: String,
    pub status: FreshItemScoreStatusV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshScoreReportV1 {
    pub schema: String,
    pub metric_id: String,
    pub run_id: String,
    pub model_provider: String,
    pub model_name: String,
    pub model_version: String,
    pub model_access_mode: String,
    pub model_checkpoint_sha256: String,
    pub batch_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub task_family: String,
    pub public_packet_sha256: String,
    pub sealed_oracle_sha256: String,
    pub manifest_sha256: String,
    pub submission_sha256: String,
    pub inference_harness_sha256: String,
    pub prompt_and_decoding_sha256: String,
    pub metric_definition_sha256: String,
    pub run_execution_receipt_sha256: String,
    pub item_count: u64,
    pub answered_count: u64,
    pub correct_count: u64,
    pub incorrect_count: u64,
    pub invalid_format_count: u64,
    pub abstained_count: u64,
    pub valid_format_response_count: u64,
    pub accuracy_over_all_permille: f64,
    pub coverage_permille: f64,
    pub accuracy_when_answered_permille: Option<f64>,
    pub accuracy_when_valid_format_permille: Option<f64>,
    pub model_call_count: u64,
    pub input_token_count: u64,
    pub output_token_count: u64,
    pub wall_clock_millis: u64,
    pub tool_call_count: u64,
    pub human_assistance_minutes: u64,
    pub compute_cost_microusd: u64,
    pub execution_receipt_verification: String,
    pub independently_verified: bool,
    pub items: Vec<FreshItemScoreV1>,
    pub report_sha256: String,
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

fn determinant(parameters: &LinearSystemParametersV1) -> i64 {
    i64::from(parameters.a11) * i64::from(parameters.a22)
        - i64::from(parameters.a12) * i64::from(parameters.a21)
}

fn compute_rhs(
    parameters: &LinearSystemParametersV1,
    solution: &ExactLinearSolutionV1,
) -> Result<(i64, i64), String> {
    let b1 = i64::from(parameters.a11)
        .checked_mul(i64::from(solution.x))
        .and_then(|value| {
            i64::from(parameters.a12)
                .checked_mul(i64::from(solution.y))
                .and_then(|right| value.checked_add(right))
        })
        .ok_or_else(|| "linear-system rhs b1 overflowed".to_owned())?;
    let b2 = i64::from(parameters.a21)
        .checked_mul(i64::from(solution.x))
        .and_then(|value| {
            i64::from(parameters.a22)
                .checked_mul(i64::from(solution.y))
                .and_then(|right| value.checked_add(right))
        })
        .ok_or_else(|| "linear-system rhs b2 overflowed".to_owned())?;
    Ok((b1, b2))
}

fn public_packet_digest(packet: &FreshPublicChallengePacketV1) -> Result<String, String> {
    let mut clone = packet.clone();
    clone.public_packet_sha256.clear();
    let bytes = serde_json::to_vec(&clone)
        .map_err(|error| format!("cannot serialize public packet: {error}"))?;
    Ok(hash_domain(b"fresh-public-packet.v1", &bytes))
}

fn sealed_oracle_digest(packet: &SealedFreshOraclePacketV1) -> Result<String, String> {
    let mut clone = packet.clone();
    clone.sealed_oracle_sha256.clear();
    let bytes = serde_json::to_vec(&clone)
        .map_err(|error| format!("cannot serialize sealed oracle: {error}"))?;
    Ok(hash_domain(b"fresh-sealed-oracle.v1", &bytes))
}

fn manifest_digest(manifest: &FreshChallengeManifestV1) -> Result<String, String> {
    let mut clone = manifest.clone();
    clone.manifest_sha256.clear();
    let bytes = serde_json::to_vec(&clone)
        .map_err(|error| format!("cannot serialize fresh manifest: {error}"))?;
    Ok(hash_domain(b"fresh-challenge-manifest.v1", &bytes))
}

fn item_set_digest(packet: &FreshPublicChallengePacketV1) -> Result<String, String> {
    let items: Vec<(&str, &str)> = packet
        .tasks
        .iter()
        .map(|task| (task.item_id.as_str(), task.task_text_sha256.as_str()))
        .collect();
    let bytes = serde_json::to_vec(&items)
        .map_err(|error| format!("cannot serialize item set: {error}"))?;
    Ok(hash_domain(b"fresh-instance-item-set.v1", &bytes))
}

fn decode_seed_hex(value: &str) -> Result<Vec<u8>, String> {
    if value.len() < 64 || value.len() % 2 != 0 {
        return Err("sealed generation seed must contain at least 32 bytes of hex".to_owned());
    }
    let mut output = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let digit = |byte: u8| match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        };
        let high =
            digit(pair[0]).ok_or_else(|| "seed must use lowercase hexadecimal".to_owned())?;
        let low = digit(pair[1]).ok_or_else(|| "seed must use lowercase hexadecimal".to_owned())?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

pub(crate) fn validate_bundle(
    bundle: &FreshChallengeBundleV1,
) -> Result<BTreeMap<String, ExactLinearSolutionV1>, String> {
    let public = &bundle.public_packet;
    let oracle = &bundle.sealed_oracle_packet;
    let manifest = &bundle.manifest;
    if public.schema != crate::fresh_instance::FRESH_PUBLIC_SCHEMA
        || oracle.schema != crate::fresh_instance::FRESH_ORACLE_SCHEMA
        || manifest.schema != crate::fresh_instance::FRESH_MANIFEST_SCHEMA
    {
        return Err("fresh bundle contains an unknown schema version".to_owned());
    }
    for (digest, label) in [
        (&public.public_packet_sha256, "public_packet_sha256"),
        (&oracle.sealed_oracle_sha256, "sealed_oracle_sha256"),
        (&manifest.manifest_sha256, "manifest_sha256"),
    ] {
        validate_digest(digest, label)?;
    }
    for (digest, label) in [
        (&public.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&public.generator_source_sha256, "generator_source_sha256"),
        (&public.generator_binary_sha256, "generator_binary_sha256"),
        (&public.runtime_image_sha256, "runtime_image_sha256"),
        (&public.generator_config_sha256, "generator_config_sha256"),
        (
            &manifest.checkpoint_freeze_receipt_sha256,
            "checkpoint_freeze_receipt_sha256",
        ),
        (
            &manifest.model_checkpoint_sha256,
            "manifest.model_checkpoint_sha256",
        ),
        (
            &manifest.generator_source_sha256,
            "manifest.generator_source_sha256",
        ),
        (
            &manifest.generator_binary_sha256,
            "manifest.generator_binary_sha256",
        ),
        (
            &manifest.runtime_image_sha256,
            "manifest.runtime_image_sha256",
        ),
        (
            &manifest.generator_config_sha256,
            "manifest.generator_config_sha256",
        ),
        (
            &manifest.generation_seed_commitment_sha256,
            "generation_seed_commitment_sha256",
        ),
    ] {
        validate_digest(digest, label)?;
    }
    parse_timestamp(
        &manifest.checkpoint_frozen_at_utc,
        "checkpoint_frozen_at_utc",
    )?;
    parse_timestamp(
        &manifest.challenge_generated_at_utc,
        "challenge_generated_at_utc",
    )?;
    crate::fresh_instance::parse_timestamp(
        &manifest.checkpoint_frozen_at_utc,
        "checkpoint_frozen_at_utc",
    )?;
    crate::fresh_instance::parse_timestamp(
        &manifest.challenge_generated_at_utc,
        "challenge_generated_at_utc",
    )?;
    validate_digest(
        &manifest.checkpoint_freeze_receipt_sha256,
        "checkpoint_freeze_receipt_sha256",
    )?;
    if public_packet_digest(public)? != public.public_packet_sha256 {
        return Err("public packet digest does not match contents".to_owned());
    }
    if sealed_oracle_digest(oracle)? != oracle.sealed_oracle_sha256 {
        return Err("sealed oracle digest does not match contents".to_owned());
    }
    if manifest_digest(manifest)? != manifest.manifest_sha256 {
        return Err("fresh manifest digest does not match contents".to_owned());
    }
    if public.batch_id != oracle.batch_id
        || public.batch_id != manifest.batch_id
        || public.benchmark_id != manifest.benchmark_id
        || public.benchmark_version != manifest.benchmark_version
        || public.task_family != manifest.task_family
        || public.model_checkpoint_sha256 != manifest.model_checkpoint_sha256
        || public.public_packet_sha256 != manifest.public_packet_sha256
        || oracle.sealed_oracle_sha256 != manifest.sealed_oracle_sha256
        || public.generator_source_sha256 != manifest.generator_source_sha256
        || public.generator_binary_sha256 != manifest.generator_binary_sha256
        || public.runtime_image_sha256 != manifest.runtime_image_sha256
        || public.generator_config_sha256 != manifest.generator_config_sha256
        || public.checkpoint_frozen_at_utc != manifest.checkpoint_frozen_at_utc
        || public.challenge_generated_at_utc != manifest.challenge_generated_at_utc
        || public.item_count != manifest.item_count
        || oracle.item_count != manifest.item_count
        || public.tasks.len() as u64 != manifest.item_count
        || oracle.entries.len() as u64 != manifest.item_count
        || item_set_digest(public)? != manifest.item_set_sha256
    {
        return Err("public packet, sealed oracle and manifest bindings do not agree".to_owned());
    }
    if manifest.checkpoint_frozen_at_utc >= manifest.challenge_generated_at_utc {
        return Err("checkpoint freeze timestamp must precede fresh generation".to_owned());
    }
    let seed = decode_seed_hex(&oracle.generation_seed_hex)?;
    if hash_domain(b"fresh-instance-seed.v1", &seed) != oracle.generation_seed_commitment_sha256
        || oracle.generation_seed_commitment_sha256 != manifest.generation_seed_commitment_sha256
        || u64::try_from(seed.len())
            .ok()
            .and_then(|n| n.checked_mul(8))
            != Some(manifest.seed_length_bits)
    {
        return Err("sealed oracle seed does not match committed generation seed".to_owned());
    }

    let public_ids: BTreeSet<&str> = public
        .tasks
        .iter()
        .map(|task| task.item_id.as_str())
        .collect();
    let oracle_ids: BTreeSet<&str> = oracle
        .entries
        .iter()
        .map(|entry| entry.item_id.as_str())
        .collect();
    if public_ids.len() != public.tasks.len()
        || oracle_ids.len() != oracle.entries.len()
        || public_ids != oracle_ids
    {
        return Err(
            "public task and sealed oracle item IDs must be unique and identical".to_owned(),
        );
    }
    let mut answers = BTreeMap::new();
    let public_task_by_id: BTreeMap<&str, &str> = public
        .tasks
        .iter()
        .map(|task| (task.item_id.as_str(), task.task_text.as_str()))
        .collect();
    let mut task_text_digests = BTreeSet::new();
    for task in &public.tasks {
        validate_digest(&task.task_text_sha256, "task_text_sha256")?;
        if !task_text_digests.insert(task.task_text_sha256.as_str()) {
            return Err("public packet contains duplicate task text digests".to_owned());
        }
        if hash_domain(b"fresh-task-text.v1", task.task_text.as_bytes()) != task.task_text_sha256 {
            return Err(format!("task {} text digest mismatch", task.item_id));
        }
    }
    for entry in &oracle.entries {
        validate_digest(&entry.parameter_sha256, "parameter_sha256")?;
        validate_digest(&entry.answer_sha256, "answer_sha256")?;
        if determinant(&entry.parameters) == 0 {
            return Err(format!(
                "oracle item {} has a singular matrix",
                entry.item_id
            ));
        }
        if normalized_determinant_margin_permille(&entry.parameters)
            != entry.normalized_determinant_margin_permille
        {
            return Err(format!(
                "oracle stability margin does not match matrix parameters for {}",
                entry.item_id
            ));
        }
        let parameter_bytes = serde_json::to_vec(&entry.parameters)
            .map_err(|error| format!("cannot serialize hidden parameters: {error}"))?;
        if hash_domain(b"fresh-task-parameters.v1", &parameter_bytes) != entry.parameter_sha256 {
            return Err(format!(
                "oracle parameters digest mismatch for {}",
                entry.item_id
            ));
        }
        let answer_bytes = serde_json::to_vec(&entry.solution)
            .map_err(|error| format!("cannot serialize oracle answer: {error}"))?;
        if hash_domain(b"fresh-task-answer.v1", &answer_bytes) != entry.answer_sha256 {
            return Err(format!(
                "oracle answer digest mismatch for {}",
                entry.item_id
            ));
        }
        let (b1, b2) = compute_rhs(&entry.parameters, &entry.solution)?;
        if b1 != entry.parameters.b1 || b2 != entry.parameters.b2 {
            return Err(format!(
                "oracle answer does not satisfy item {}",
                entry.item_id
            ));
        }
        let expected_public_task = task_text(&entry.parameters);
        if public_task_by_id.get(entry.item_id.as_str()).copied()
            != Some(expected_public_task.as_str())
        {
            return Err(format!(
                "public task text does not correspond to sealed oracle parameters for {}",
                entry.item_id
            ));
        }
        answers.insert(entry.item_id.clone(), entry.solution.clone());
    }
    Ok(answers)
}

fn submission_digest(submission: &FreshAnswerSubmissionV1) -> Result<String, String> {
    let mut clone = submission.clone();
    clone.submission_sha256.clear();
    let bytes = serde_json::to_vec(&clone)
        .map_err(|error| format!("cannot serialize answer submission: {error}"))?;
    Ok(hash_domain(b"fresh-answer-submission.v1", &bytes))
}

fn response_digest(response_text: &str) -> String {
    hash_domain(b"fresh-model-response.v1", response_text.as_bytes())
}

pub fn verify_fresh_score_report_integrity(report: &FreshScoreReportV1) -> Result<(), String> {
    if report.schema != FRESH_SCORE_REPORT_SCHEMA || report.metric_id != FRESH_EXACT_METRIC_ID {
        return Err("fresh score report schema or metric identifier is invalid".to_owned());
    }
    validate_digest(&report.report_sha256, "report_sha256")?;
    let mut normalized = report.clone();
    normalized.report_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize normalized score report: {error}"))?;
    if hash_domain(b"fresh-instance-score-report.v1", &bytes) != report.report_sha256 {
        return Err("fresh score report digest does not match contents".to_owned());
    }
    let mut seen_items = BTreeSet::new();
    let mut correct = 0_u64;
    let mut incorrect = 0_u64;
    let mut invalid = 0_u64;
    let mut abstained = 0_u64;
    for item in &report.items {
        if item.item_id.trim().is_empty() || !seen_items.insert(item.item_id.as_str()) {
            return Err("fresh score report item IDs must be unique and nonempty".to_owned());
        }
        validate_digest(&item.response_sha256, "item.response_sha256")?;
        match item.status {
            FreshItemScoreStatusV1::Correct => correct += 1,
            FreshItemScoreStatusV1::Incorrect => incorrect += 1,
            FreshItemScoreStatusV1::InvalidFormat => invalid += 1,
            FreshItemScoreStatusV1::Abstained => abstained += 1,
        }
    }
    let total = correct
        .checked_add(incorrect)
        .and_then(|value| value.checked_add(invalid))
        .and_then(|value| value.checked_add(abstained))
        .ok_or_else(|| "fresh score report count partition overflowed".to_owned())?;
    let answered = total.saturating_sub(abstained);
    let valid_format = correct
        .checked_add(incorrect)
        .ok_or_else(|| "fresh score report valid-format count overflowed".to_owned())?;
    let expected_accuracy = correct as f64 * 1_000.0 / total as f64;
    let expected_coverage = answered as f64 * 1_000.0 / total as f64;
    let rate_matches = |actual: Option<f64>, expected: Option<f64>| match (actual, expected) {
        (None, None) => true,
        (Some(actual), Some(expected)) => actual.is_finite() && (actual - expected).abs() < 1.0e-9,
        _ => false,
    };
    if total == 0
        || total != report.item_count
        || correct != report.correct_count
        || incorrect != report.incorrect_count
        || invalid != report.invalid_format_count
        || abstained != report.abstained_count
        || answered != report.answered_count
        || valid_format != report.valid_format_response_count
        || !report.accuracy_over_all_permille.is_finite()
        || !report.coverage_permille.is_finite()
        || (report.accuracy_over_all_permille - expected_accuracy).abs() >= 1.0e-9
        || (report.coverage_permille - expected_coverage).abs() >= 1.0e-9
        || !rate_matches(
            report.accuracy_when_answered_permille,
            optional_rate(correct, answered),
        )
        || !rate_matches(
            report.accuracy_when_valid_format_permille,
            optional_rate(correct, valid_format),
        )
        || report.independently_verified
        || report.execution_receipt_verification
            != "digest_present_signature_not_cryptographically_verified"
    {
        return Err("fresh score report item statuses, summary counts/rates or trust boundary are inconsistent".to_owned());
    }
    Ok(())
}

fn optional_rate(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 * 1_000.0 / denominator as f64)
    }
}

/// Score a strict model response submission against the evaluator-only sealed
/// oracle. The resulting report contains no hidden parameters or correct answers.
pub fn score_fresh_submission(
    bundle: &FreshChallengeBundleV1,
    submission_bytes: &[u8],
) -> Result<FreshScoreReportV1, String> {
    let answers = validate_bundle(bundle)?;
    if submission_bytes.len() > FRESH_MAX_SUBMISSION_BYTES {
        return Err("answer submission exceeds the 32 MiB safety bound".to_owned());
    }
    let submission: FreshAnswerSubmissionV1 = serde_json::from_slice(submission_bytes)
        .map_err(|error| format!("answer submission does not match strict v1 JSON: {error}"))?;
    if submission.schema != FRESH_SUBMISSION_SCHEMA {
        return Err(format!(
            "submission schema must be {FRESH_SUBMISSION_SCHEMA}"
        ));
    }
    if submission.run_id.trim().is_empty()
        || submission.run_id.len() > 256
        || submission.model_provider.trim().is_empty()
        || submission.model_name.trim().is_empty()
        || submission.model_version.trim().is_empty()
        || submission.model_access_mode.trim().is_empty()
    {
        return Err("run/model identity fields must be nonempty".to_owned());
    }
    for (digest, label) in [
        (
            &submission.model_checkpoint_sha256,
            "model_checkpoint_sha256",
        ),
        (&submission.public_packet_sha256, "public_packet_sha256"),
        (
            &submission.inference_harness_sha256,
            "inference_harness_sha256",
        ),
        (
            &submission.prompt_and_decoding_sha256,
            "prompt_and_decoding_sha256",
        ),
        (
            &submission.metric_definition_sha256,
            "metric_definition_sha256",
        ),
        (
            &submission.run_execution_receipt_sha256,
            "run_execution_receipt_sha256",
        ),
        (&submission.submission_sha256, "submission_sha256"),
    ] {
        validate_digest(digest, label)?;
    }
    if submission.model_checkpoint_sha256 != bundle.public_packet.model_checkpoint_sha256
        || submission.public_packet_sha256 != bundle.public_packet.public_packet_sha256
    {
        return Err("submission model checkpoint/public packet binding mismatch".to_owned());
    }
    if submission.metric_definition_sha256
        != hash_domain(
            b"fresh-metric-definition.v1",
            FRESH_EXACT_METRIC_ID.as_bytes(),
        )
    {
        return Err(
            "submission metric definition digest does not match fresh exact-match scorer"
                .to_owned(),
        );
    }
    if submission.model_call_count == 0 || submission.wall_clock_millis == 0 {
        return Err("model_call_count and wall_clock_millis must be positive".to_owned());
    }
    if submission.responses.len() != bundle.public_packet.tasks.len()
        || submission.responses.len() != answers.len()
    {
        return Err(
            "submission must contain exactly one response record per public item".to_owned(),
        );
    }
    let response_bytes_total = submission
        .responses
        .iter()
        .try_fold(0_usize, |total, record| {
            if record.response_text.len() > FRESH_MAX_RESPONSE_BYTES {
                return Err("an individual response exceeds the 1 MiB safety bound".to_owned());
            }
            total
                .checked_add(record.response_text.len())
                .ok_or_else(|| "total response bytes overflow usize".to_owned())
        })?;
    if response_bytes_total > FRESH_MAX_TOTAL_RESPONSE_BYTES {
        return Err("total response text exceeds the 16 MiB safety bound".to_owned());
    }

    let mut item_ids = BTreeSet::new();
    let mut correct_count = 0_u64;
    let mut incorrect_count = 0_u64;
    let mut invalid_format_count = 0_u64;
    let mut abstained_count = 0_u64;
    let mut items = Vec::with_capacity(submission.responses.len());
    for record in &submission.responses {
        if record.item_id.trim().is_empty() || !item_ids.insert(record.item_id.as_str()) {
            return Err("response item IDs must be unique and nonempty".to_owned());
        }
        let answer = answers
            .get(&record.item_id)
            .ok_or_else(|| format!("submission contains unknown item ID {}", record.item_id))?;
        validate_digest(&record.response_sha256, "response_sha256")?;
        if response_digest(&record.response_text) != record.response_sha256 {
            return Err(format!(
                "response {} digest does not match raw response text",
                record.item_id
            ));
        }
        let status = if record.response_text.trim().is_empty() {
            abstained_count = abstained_count
                .checked_add(1)
                .ok_or_else(|| "abstention count overflowed".to_owned())?;
            FreshItemScoreStatusV1::Abstained
        } else {
            match serde_json::from_str::<ExactLinearSolutionV1>(&record.response_text) {
                Ok(parsed) if &parsed == answer => {
                    correct_count = correct_count
                        .checked_add(1)
                        .ok_or_else(|| "correct count overflowed".to_owned())?;
                    FreshItemScoreStatusV1::Correct
                }
                Ok(_) => {
                    incorrect_count = incorrect_count
                        .checked_add(1)
                        .ok_or_else(|| "incorrect count overflowed".to_owned())?;
                    FreshItemScoreStatusV1::Incorrect
                }
                Err(_) => {
                    invalid_format_count = invalid_format_count
                        .checked_add(1)
                        .ok_or_else(|| "invalid format count overflowed".to_owned())?;
                    FreshItemScoreStatusV1::InvalidFormat
                }
            }
        };
        items.push(FreshItemScoreV1 {
            item_id: record.item_id.clone(),
            response_sha256: record.response_sha256.clone(),
            status,
        });
    }
    if item_ids.len() != answers.len() {
        return Err("submission response item set is incomplete".to_owned());
    }
    if submission_digest(&submission)? != submission.submission_sha256 {
        return Err("submission digest does not match its contents".to_owned());
    }

    let item_count = u64::try_from(answers.len()).map_err(|_| "item count exceeds u64")?;
    let answered_count = item_count
        .checked_sub(abstained_count)
        .ok_or_else(|| "abstention count exceeds item count".to_owned())?;
    let valid_format_response_count = correct_count
        .checked_add(incorrect_count)
        .ok_or_else(|| "valid response count overflowed".to_owned())?;
    let partitioned = correct_count
        .checked_add(incorrect_count)
        .and_then(|value| value.checked_add(invalid_format_count))
        .and_then(|value| value.checked_add(abstained_count))
        .ok_or_else(|| "score outcome counts overflowed".to_owned())?;
    if partitioned != item_count {
        return Err(
            "correct/incorrect/invalid/abstained counts do not partition item set".to_owned(),
        );
    }
    let mut report = FreshScoreReportV1 {
        schema: FRESH_SCORE_REPORT_SCHEMA.to_owned(),
        metric_id: FRESH_EXACT_METRIC_ID.to_owned(),
        run_id: submission.run_id.clone(),
        model_provider: submission.model_provider.clone(),
        model_name: submission.model_name.clone(),
        model_version: submission.model_version.clone(),
        model_access_mode: submission.model_access_mode.clone(),
        model_checkpoint_sha256: submission.model_checkpoint_sha256.clone(),
        batch_id: bundle.public_packet.batch_id.clone(),
        benchmark_id: bundle.public_packet.benchmark_id.clone(),
        benchmark_version: bundle.public_packet.benchmark_version.clone(),
        task_family: bundle.public_packet.task_family.clone(),
        public_packet_sha256: bundle.public_packet.public_packet_sha256.clone(),
        sealed_oracle_sha256: bundle.sealed_oracle_packet.sealed_oracle_sha256.clone(),
        manifest_sha256: bundle.manifest.manifest_sha256.clone(),
        submission_sha256: submission.submission_sha256.clone(),
        inference_harness_sha256: submission.inference_harness_sha256.clone(),
        prompt_and_decoding_sha256: submission.prompt_and_decoding_sha256.clone(),
        metric_definition_sha256: submission.metric_definition_sha256.clone(),
        run_execution_receipt_sha256: submission.run_execution_receipt_sha256.clone(),
        item_count,
        answered_count,
        correct_count,
        incorrect_count,
        invalid_format_count,
        abstained_count,
        valid_format_response_count,
        accuracy_over_all_permille: correct_count as f64 * 1_000.0 / item_count as f64,
        coverage_permille: answered_count as f64 * 1_000.0 / item_count as f64,
        accuracy_when_answered_permille: optional_rate(correct_count, answered_count),
        accuracy_when_valid_format_permille: optional_rate(correct_count, valid_format_response_count),
        model_call_count: submission.model_call_count,
        input_token_count: submission.input_token_count,
        output_token_count: submission.output_token_count,
        wall_clock_millis: submission.wall_clock_millis,
        tool_call_count: submission.tool_call_count,
        human_assistance_minutes: submission.human_assistance_minutes,
        compute_cost_microusd: submission.compute_cost_microusd,
        execution_receipt_verification: "digest_present_signature_not_cryptographically_verified".to_owned(),
        independently_verified: false,
        items,
        report_sha256: String::new(),
        claim_boundary: "exact_json_integer_pair_match_to_sealed_oracle;run_execution_receipt_is_self_reported;no_claim_of_task_recipe_novelty_or_training_data_absence".to_owned(),
    };
    let report_bytes = serde_json::to_vec(&report)
        .map_err(|error| format!("cannot serialize fresh score report: {error}"))?;
    report.report_sha256 = hash_domain(b"fresh-instance-score-report.v1", &report_bytes);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fresh_instance::{
        FRESH_CONFIG_SCHEMA, FreshChallengeConfigV1, generate_fresh_challenge_bundle,
    };
    use serde_json::Value;

    fn digest(label: &str) -> String {
        hash_domain(b"fresh-scoring-test", label.as_bytes())
    }

    fn config() -> FreshChallengeConfigV1 {
        let mut config = FreshChallengeConfigV1 {
            schema: FRESH_CONFIG_SCHEMA.to_owned(),
            batch_id: "fresh-score-test".to_owned(),
            benchmark_id: "fresh-score-benchmark".to_owned(),
            benchmark_version: "v1".to_owned(),
            task_family: "linear-system-2x2".to_owned(),
            requested_item_count: 8,
            matrix_coefficient_min: -9,
            matrix_coefficient_max: 9,
            solution_component_min: -5,
            solution_component_max: 5,
            minimum_absolute_determinant: 1,
            minimum_normalized_determinant_margin_permille: 50,
            model_checkpoint_sha256: digest("checkpoint"),
            checkpoint_freeze_receipt_sha256: digest("freeze"),
            checkpoint_frozen_at_utc: "2026-10-31T18:00:00Z".to_owned(),
            challenge_generated_at_utc: "2026-10-31T18:01:00Z".to_owned(),
            generator_source_sha256: digest("source"),
            generator_binary_sha256: digest("binary"),
            runtime_image_sha256: digest("runtime"),
            generator_config_sha256: String::new(),
        };
        config.generator_config_sha256 =
            crate::fresh_instance::canonical_config_digest(&config).unwrap();
        config
    }

    fn seed() -> Vec<u8> {
        vec![0xA1; 32]
    }

    fn solution_text(solution: &ExactLinearSolutionV1) -> String {
        serde_json::to_string(solution).unwrap()
    }

    fn build_submission(
        bundle: &FreshChallengeBundleV1,
        responses: Vec<FreshResponseRecordV1>,
    ) -> FreshAnswerSubmissionV1 {
        let mut submission = FreshAnswerSubmissionV1 {
            schema: FRESH_SUBMISSION_SCHEMA.to_owned(),
            run_id: "fresh-score-run-001".to_owned(),
            model_provider: "fixture-provider".to_owned(),
            model_name: "fixture-model".to_owned(),
            model_version: "v1".to_owned(),
            model_access_mode: "offline-fixture".to_owned(),
            model_checkpoint_sha256: bundle.public_packet.model_checkpoint_sha256.clone(),
            public_packet_sha256: bundle.public_packet.public_packet_sha256.clone(),
            inference_harness_sha256: digest("harness"),
            prompt_and_decoding_sha256: digest("prompt-decoding"),
            metric_definition_sha256: hash_domain(
                b"fresh-metric-definition.v1",
                FRESH_EXACT_METRIC_ID.as_bytes(),
            ),
            run_execution_receipt_sha256: digest("execution-receipt"),
            model_call_count: bundle.public_packet.item_count,
            input_token_count: 8_000,
            output_token_count: 128,
            wall_clock_millis: 10_000,
            tool_call_count: 0,
            human_assistance_minutes: 0,
            compute_cost_microusd: 0,
            responses,
            submission_sha256: String::new(),
        };
        submission.submission_sha256 = submission_digest(&submission).unwrap();
        submission
    }

    fn exact_responses(bundle: &FreshChallengeBundleV1) -> Vec<FreshResponseRecordV1> {
        bundle
            .sealed_oracle_packet
            .entries
            .iter()
            .map(|entry| {
                let response_text = solution_text(&entry.solution);
                FreshResponseRecordV1 {
                    item_id: entry.item_id.clone(),
                    response_sha256: response_digest(&response_text),
                    response_text,
                }
            })
            .collect()
    }

    fn score(
        bundle: &FreshChallengeBundleV1,
        responses: Vec<FreshResponseRecordV1>,
    ) -> FreshScoreReportV1 {
        let submission = build_submission(bundle, responses);
        score_fresh_submission(bundle, &serde_json::to_vec(&submission).unwrap()).unwrap()
    }

    #[test]
    fn submission_and_score_report_schemas_match_rust_serialization() {
        let submission_schema: Value = serde_json::from_str(include_str!(
            "../schema/fresh-answer-submission.schema.json"
        ))
        .unwrap();
        let report_schema: Value = serde_json::from_str(include_str!(
            "../schema/fresh-instance-score-report.schema.json"
        ))
        .unwrap();
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let submission = build_submission(&bundle, exact_responses(&bundle));
        let report =
            score_fresh_submission(&bundle, &serde_json::to_vec(&submission).unwrap()).unwrap();
        let submission_value = serde_json::to_value(&submission).unwrap();
        let report_value = serde_json::to_value(&report).unwrap();
        for (schema, value, version) in [
            (
                &submission_schema,
                submission_value,
                FRESH_SUBMISSION_SCHEMA,
            ),
            (&report_schema, report_value, FRESH_SCORE_REPORT_SCHEMA),
        ] {
            assert_eq!(
                schema["properties"]["schema"]["const"].as_str(),
                Some(version)
            );
            let required: BTreeSet<String> = schema["required"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item.as_str().unwrap().to_owned())
                .collect();
            let properties: BTreeSet<String> = schema["properties"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect();
            let actual: BTreeSet<String> = value.as_object().unwrap().keys().cloned().collect();
            assert_eq!(required, properties);
            assert_eq!(actual, properties);
        }
        let response_properties: BTreeSet<String> =
            submission_schema["properties"]["responses"]["items"]["properties"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect();
        let actual_response: BTreeSet<String> = serde_json::to_value(&submission.responses[0])
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(response_properties, actual_response);
        let report_item_properties: BTreeSet<String> =
            report_schema["properties"]["items"]["items"]["properties"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect();
        let actual_report_item: BTreeSet<String> = serde_json::to_value(&report.items[0])
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(report_item_properties, actual_report_item);
        let public_report = serde_json::to_string(&report).unwrap();
        assert!(!public_report.contains("\"solution\""));
        assert!(!public_report.contains("\"parameters\""));
        assert!(!public_report.contains(&bundle.sealed_oracle_packet.generation_seed_hex));
        assert!(!report.independently_verified);
    }

    #[test]
    fn perfect_answers_score_full_accuracy_and_coverage() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let report = score(&bundle, exact_responses(&bundle));
        assert_eq!(report.correct_count, bundle.public_packet.item_count);
        assert_eq!(report.accuracy_over_all_permille, 1_000.0);
        assert_eq!(report.coverage_permille, 1_000.0);
        assert_eq!(report.invalid_format_count, 0);
        assert!(!report.independently_verified);
    }

    #[test]
    fn abstentions_reduce_overall_accuracy_and_coverage() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let mut responses = exact_responses(&bundle);
        responses[0].response_text.clear();
        responses[0].response_sha256 = response_digest("");
        let report = score(&bundle, responses);
        assert_eq!(report.abstained_count, 1);
        assert_eq!(report.answered_count, bundle.public_packet.item_count - 1);
        assert_eq!(report.coverage_permille, 875.0);
        assert_eq!(report.accuracy_over_all_permille, 875.0);
        assert_eq!(report.accuracy_when_answered_permille, Some(1_000.0));
    }

    #[test]
    fn malformed_nonempty_responses_are_invalid_not_abstentions() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let mut responses = exact_responses(&bundle);
        responses[0].response_text = "I think x=1 and y=2".to_owned();
        responses[0].response_sha256 = response_digest(&responses[0].response_text);
        let report = score(&bundle, responses);
        assert_eq!(report.invalid_format_count, 1);
        assert_eq!(report.abstained_count, 0);
        assert_eq!(report.correct_count, 7);
    }

    #[test]
    fn extra_fields_floats_and_trailing_prose_are_not_exact_json_pairs() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let cases = [
            r#"{"x":1,"y":2,"explanation":"guess"}"#,
            r#"{"x":1.0,"y":2}"#,
            r#"{"x":1,"y":2} trailing"#,
            r#"{"x":1,"x":2,"y":2}"#,
        ];
        for case in cases {
            let mut responses = exact_responses(&bundle);
            responses[0].response_text = case.to_owned();
            responses[0].response_sha256 = response_digest(case);
            let report = score(&bundle, responses);
            assert_eq!(report.invalid_format_count, 1);
        }
    }

    #[test]
    fn wrong_valid_integer_solution_is_incorrect_not_invalid() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let mut responses = exact_responses(&bundle);
        let first = &bundle.sealed_oracle_packet.entries[0];
        let wrong = ExactLinearSolutionV1 {
            x: first.solution.x.saturating_add(1),
            y: first.solution.y,
        };
        responses[0].response_text = solution_text(&wrong);
        responses[0].response_sha256 = response_digest(&responses[0].response_text);
        let report = score(&bundle, responses);
        assert_eq!(report.incorrect_count, 1);
        assert_eq!(report.invalid_format_count, 0);
        assert_eq!(report.correct_count, 7);
    }

    #[test]
    fn missing_duplicate_and_unknown_item_ids_fail_closed() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let mut responses = exact_responses(&bundle);
        responses.pop();
        assert!(
            score_fresh_submission(
                &bundle,
                &serde_json::to_vec(&build_submission(&bundle, responses)).unwrap()
            )
            .unwrap_err()
            .contains("exactly one response")
        );

        let mut responses = exact_responses(&bundle);
        responses[1].item_id = responses[0].item_id.clone();
        assert!(
            score_fresh_submission(
                &bundle,
                &serde_json::to_vec(&build_submission(&bundle, responses)).unwrap()
            )
            .unwrap_err()
            .contains("unique and nonempty")
        );

        let mut responses = exact_responses(&bundle);
        responses[0].item_id = "unknown-item".to_owned();
        assert!(
            score_fresh_submission(
                &bundle,
                &serde_json::to_vec(&build_submission(&bundle, responses)).unwrap()
            )
            .unwrap_err()
            .contains("unknown item ID")
        );
    }

    #[test]
    fn oversized_submission_payload_fails_before_parsing() {
        let oversized = vec![b' '; FRESH_MAX_SUBMISSION_BYTES + 1];
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        assert!(
            score_fresh_submission(&bundle, &oversized)
                .unwrap_err()
                .contains("32 MiB safety bound")
        );
    }

    #[test]
    fn response_and_submission_digest_mutations_fail_closed() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let mut submission = build_submission(&bundle, exact_responses(&bundle));
        submission.responses[0].response_text.push(' ');
        assert!(
            score_fresh_submission(&bundle, &serde_json::to_vec(&submission).unwrap())
                .unwrap_err()
                .contains("response")
        );
        let mut submission = build_submission(&bundle, exact_responses(&bundle));
        submission.input_token_count += 1;
        assert!(
            score_fresh_submission(&bundle, &serde_json::to_vec(&submission).unwrap())
                .unwrap_err()
                .contains("submission digest")
        );
    }

    #[test]
    fn wrong_checkpoint_or_public_packet_binding_fails() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let mut submission = build_submission(&bundle, exact_responses(&bundle));
        submission.model_checkpoint_sha256 = digest("another-checkpoint");
        assert!(
            score_fresh_submission(&bundle, &serde_json::to_vec(&submission).unwrap())
                .unwrap_err()
                .contains("checkpoint/public packet")
        );
    }

    #[test]
    fn scorer_rejects_rehashed_public_task_that_no_longer_matches_oracle_parameters() {
        let mut bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        bundle.public_packet.tasks[0].task_text.push_str(" + 0");
        bundle.public_packet.tasks[0].task_text_sha256 = hash_domain(
            b"fresh-task-text.v1",
            bundle.public_packet.tasks[0].task_text.as_bytes(),
        );
        bundle.public_packet.public_packet_sha256 =
            public_packet_digest(&bundle.public_packet).unwrap();
        bundle.manifest.public_packet_sha256 = bundle.public_packet.public_packet_sha256.clone();
        bundle.manifest.manifest_sha256 = manifest_digest(&bundle.manifest).unwrap();
        assert!(
            score_fresh_submission(
                &bundle,
                &serde_json::to_vec(&build_submission(&bundle, exact_responses(&bundle))).unwrap()
            )
            .unwrap_err()
            .contains("does not correspond to sealed oracle parameters")
        );
    }

    #[test]
    fn scorer_rejects_rehashed_oracle_with_false_stability_margin() {
        let mut bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        bundle.sealed_oracle_packet.entries[0].normalized_determinant_margin_permille = 1_000;
        bundle.sealed_oracle_packet.sealed_oracle_sha256 =
            sealed_oracle_digest(&bundle.sealed_oracle_packet).unwrap();
        bundle.manifest.sealed_oracle_sha256 =
            bundle.sealed_oracle_packet.sealed_oracle_sha256.clone();
        bundle.manifest.manifest_sha256 = manifest_digest(&bundle.manifest).unwrap();
        assert!(
            score_fresh_submission(
                &bundle,
                &serde_json::to_vec(&build_submission(&bundle, exact_responses(&bundle))).unwrap()
            )
            .unwrap_err()
            .contains("stability margin does not match matrix parameters")
        );
    }

    #[test]
    fn scorer_rejects_malformed_freeze_timestamps_and_metadata_digests() {
        let mut bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        bundle.manifest.checkpoint_frozen_at_utc = "2026-02-30T18:00:00Z".to_owned();
        bundle.manifest.manifest_sha256 = manifest_digest(&bundle.manifest).unwrap();
        assert!(
            score_fresh_submission(
                &bundle,
                &serde_json::to_vec(&build_submission(&bundle, exact_responses(&bundle))).unwrap()
            )
            .unwrap_err()
            .contains("impossible date or clock")
        );

        let mut bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        bundle.public_packet.generator_binary_sha256 = "not-a-digest".to_owned();
        bundle.public_packet.public_packet_sha256 =
            public_packet_digest(&bundle.public_packet).unwrap();
        bundle.manifest.generator_binary_sha256 = "not-a-digest".to_owned();
        bundle.manifest.public_packet_sha256 = bundle.public_packet.public_packet_sha256.clone();
        bundle.manifest.manifest_sha256 = manifest_digest(&bundle.manifest).unwrap();
        assert!(
            score_fresh_submission(
                &bundle,
                &serde_json::to_vec(&build_submission(&bundle, exact_responses(&bundle))).unwrap()
            )
            .unwrap_err()
            .contains("generator_binary_sha256")
        );
    }

    #[test]
    fn scorer_rejects_tampered_oracle_even_if_answers_are_changed() {
        let mut bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        bundle.sealed_oracle_packet.entries[0].solution.x += 1;
        assert!(
            score_fresh_submission(
                &bundle,
                &serde_json::to_vec(&build_submission(&bundle, exact_responses(&bundle))).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn unknown_submission_and_response_fields_are_rejected() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let submission = build_submission(&bundle, exact_responses(&bundle));
        let mut json: Value = serde_json::to_value(&submission).unwrap();
        json["unexpected"] = serde_json::json!("not-allowed");
        assert!(serde_json::from_value::<FreshAnswerSubmissionV1>(json).is_err());

        let mut response_json: Value = serde_json::to_value(&submission.responses[0]).unwrap();
        response_json["expected_answer"] = serde_json::json!({"x": 1, "y": 2});
        assert!(serde_json::from_value::<FreshResponseRecordV1>(response_json).is_err());
    }

    #[test]
    fn report_integrity_verifier_detects_field_mutations() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let mut report = score(&bundle, exact_responses(&bundle));
        assert!(verify_fresh_score_report_integrity(&report).is_ok());
        report.correct_count -= 1;
        assert!(
            verify_fresh_score_report_integrity(&report)
                .unwrap_err()
                .contains("digest does not match")
        );
    }

    #[test]
    fn rehashed_report_with_false_summary_counts_is_rejected() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let mut report = score(&bundle, exact_responses(&bundle));
        report.correct_count -= 1;
        report.incorrect_count += 1;
        report.report_sha256.clear();
        let bytes = serde_json::to_vec(&report).unwrap();
        report.report_sha256 = hash_domain(b"fresh-instance-score-report.v1", &bytes);
        assert!(
            verify_fresh_score_report_integrity(&report)
                .unwrap_err()
                .contains("item statuses, summary counts/rates")
        );
    }

    #[test]
    fn no_answered_items_yields_absent_conditional_accuracy() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let mut responses = exact_responses(&bundle);
        for response in &mut responses {
            response.response_text.clear();
            response.response_sha256 = response_digest("");
        }
        let report = score(&bundle, responses);
        assert_eq!(report.accuracy_over_all_permille, 0.0);
        assert_eq!(report.coverage_permille, 0.0);
        assert_eq!(report.accuracy_when_answered_permille, None);
        assert_eq!(report.accuracy_when_valid_format_permille, None);
    }
}
