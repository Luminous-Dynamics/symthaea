//! CoDeC (Contamination Detection via Context) scoring.
//!
//! Paper-faithful unit of analysis: one dataset. For each target sequence,
//! compare mean target-token log-probability without context against the same
//! target token IDs after prepending examples drawn from the candidate dataset.
//! Average delta over the declared context seeds, mark delta < 0, and aggregate
//! the negative-delta fraction across target items.
//!
//! This is a gray-box, no-generation scorer. It estimates dataset-distribution
//! exposure/specialization; a high score is not proof of exact item membership.
//! Dataset-level AUROC is deliberately computed over dataset observations, not
//! over the target items nested within one dataset.

use crate::hash_domain;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CODEC_REQUEST_SCHEMA: &str = "symthaea.codec_dataset_request.v1";
pub const CODEC_REPORT_SCHEMA: &str = "symthaea.codec_dataset_report.v1";
pub const CODEC_DATASET_AUC_SCHEMA: &str = "symthaea.codec_dataset_level_auc.v1";
pub const CODEC_DATASET_AUC_REQUEST_SCHEMA: &str = "symthaea.codec_dataset_level_auc_request.v1";
pub const MAX_DATASET_ITEMS: usize = 100_000;
pub const MAX_TARGET_TOKENS: usize = 32_768;
pub const MAX_CONTEXT_ITEMS: usize = 8;
pub const MIN_CONTEXT_SEEDS: usize = 1;
pub const MAX_CONTEXT_SEEDS: usize = 20;
pub const MIN_PAPER_TARGET_ITEMS: usize = 100;
pub const MAX_CONTEXT_POLICY_ID: usize = 128;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CoDecSamplingModeV1 {
    /// The ICLR paper's basic, compute-efficient setup: one context draw.
    SingleSeedEfficiency,
    /// Average the per-target delta over five independently sampled contexts.
    FiveSeedPaperRobustness,
    /// Preregistered multi-seed sensitivity extension (2..=20 seeds).
    MultiSeedSensitivity,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoDecContextEvaluationV1 {
    pub context_seed: u64,
    pub context_item_ids: Vec<String>,
    /// SHA-256 over the canonical sorted context item-ID list.
    pub context_item_set_sha256: String,
    pub target_token_sequence_sha256: String,
    /// Average target-token log-probabilities after prepending these contexts.
    /// The target sequence must be byte/token aligned with the baseline.
    pub target_token_logprobs: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoDecTargetInputV1 {
    pub item_id: String,
    pub target_token_count: u64,
    /// Digest of the exact target token-ID sequence, not only its source text.
    pub target_token_sequence_sha256: String,
    pub baseline_target_token_sequence_sha256: String,
    /// One log-probability per target token, without any context examples.
    pub baseline_target_token_logprobs: Vec<f64>,
    pub context_evaluations: Vec<CoDecContextEvaluationV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoDecDatasetRequestV1 {
    pub schema: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
    pub chat_template_sha256: String,
    /// Digest over the full dataset manifest, including content and item IDs.
    pub dataset_manifest_sha256: String,
    pub dataset_item_set_sha256: String,
    pub context_policy_id: String,
    pub context_policy_sha256: String,
    pub inference_backend_id: String,
    pub inference_receipt_sha256: String,
    pub n_context_examples: u32,
    pub sampling_mode: CoDecSamplingModeV1,
    /// Exact dataset candidate universe. Every target item and sampled context
    /// ID must occur here. Context item IDs must exclude their own target ID.
    pub dataset_item_ids: Vec<String>,
    pub targets: Vec<CoDecTargetInputV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CoDecTargetReportV1 {
    pub item_id: String,
    pub target_token_count: u64,
    pub baseline_mean_logprob_nats: f64,
    pub context_mean_logprob_nats: f64,
    pub mean_delta_nats_per_token: f64,
    pub per_seed_delta_nats_per_token: Vec<f64>,
    pub negative_delta: bool,
    pub context_seed_count: u64,
    pub context_item_ids_by_seed: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CoDecWilsonIntervalV1 {
    pub lower_95: f64,
    pub upper_95: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CoDecDatasetReportV1 {
    pub schema: String,
    pub implementation_id: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
    pub chat_template_sha256: String,
    pub dataset_manifest_sha256: String,
    pub dataset_item_set_sha256: String,
    pub target_item_set_sha256: String,
    pub context_policy_id: String,
    pub context_policy_sha256: String,
    pub inference_backend_id: String,
    pub inference_receipt_sha256: String,
    pub n_context_examples: u32,
    pub sampling_mode: CoDecSamplingModeV1,
    pub context_seed_count_per_target: u64,
    pub target_item_count: u64,
    pub target_token_count_total: u64,
    pub negative_delta_count: u64,
    /// 1000 * negative_delta_count / target_item_count.
    pub codec_score_permille: f64,
    /// Descriptive item-level Wilson interval. Shared contexts can induce
    /// dependence, so this is not a dataset-cluster confidence interval.
    pub item_level_wilson_95: CoDecWilsonIntervalV1,
    pub mean_delta_nats_per_token: f64,
    pub standard_deviation_delta_across_items: f64,
    pub median_delta_nats_per_token: f64,
    pub per_item: Vec<CoDecTargetReportV1>,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub item_interval_assumes_independent_targets: bool,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoDecDatasetLabelledScoreV1 {
    pub dataset_id: String,
    pub source_family_id: String,
    /// Ground truth from an independently verified training-data manifest or
    /// randomized training intervention, not a detector-derived label.
    pub known_training_exposure: bool,
    pub codec_score_permille: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoDecDatasetLevelAucRequestV1 {
    pub schema: String,
    pub datasets: Vec<CoDecDatasetLabelledScoreV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CoDecDatasetLevelAucReportV1 {
    pub schema: String,
    pub dataset_observation_count: u64,
    pub source_family_count: u64,
    pub exposed_dataset_count: u64,
    pub unexposed_dataset_count: u64,
    pub dataset_level_auroc: f64,
    pub dataset_level_auprc: f64,
    pub unit_of_analysis: String,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub claim_boundary: String,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn valid_id(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= maximum
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
}

fn canonical_sorted_ids(ids: &[String], label: &str) -> Result<Vec<u8>, String> {
    if ids.is_empty() {
        return Err(format!("{label} cannot be empty"));
    }
    let mut sorted = ids.to_vec();
    if sorted.iter().any(|id| !valid_id(id, 128)) {
        return Err(format!("{label} contains an invalid item ID"));
    }
    sorted.sort();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(format!("{label} contains duplicate item IDs"));
    }
    serde_json::to_vec(&sorted).map_err(|error| format!("cannot serialize {label}: {error}"))
}

/// Public canonical commitment helper for the exact candidate item universe.
pub fn compute_codec_item_set_sha256(ids: &[String]) -> Result<String, String> {
    let bytes = canonical_sorted_ids(ids, "CoDeC item set")?;
    Ok(hash_domain(b"codec-item-set.v1", &bytes))
}

/// Public canonical commitment helper for a target's sampled context items.
pub fn compute_codec_context_item_set_sha256(ids: &[String]) -> Result<String, String> {
    let bytes = canonical_sorted_ids(ids, "CoDeC context item set")?;
    Ok(hash_domain(b"codec-context-item-set.v1", &bytes))
}

fn validate_logprobs(values: &[f64], expected_count: u64, label: &str) -> Result<(), String> {
    if values.is_empty()
        || values.len() > MAX_TARGET_TOKENS
        || values.len() as u64 != expected_count
    {
        return Err(format!(
            "{label} log-probability count must exactly equal target_token_count and be in 1..={MAX_TARGET_TOKENS}"
        ));
    }
    if values.iter().any(|value| !value.is_finite() || *value > 0.0) {
        return Err(format!("{label} log-probabilities must be finite and non-positive"));
    }
    Ok(())
}

fn mean(values: &[f64], label: &str) -> Result<f64, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} cannot be empty or non-finite"));
    }
    let result = values.iter().sum::<f64>() / values.len() as f64;
    if result.is_finite() {
        Ok(result)
    } else {
        Err(format!("{label} mean is non-finite"))
    }
}

fn median(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("CoDeC median requires finite values".to_owned());
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    if ordered.len().is_multiple_of(2) {
        Ok((ordered[middle - 1] + ordered[middle]) / 2.0)
    } else {
        Ok(ordered[middle])
    }
}

fn sample_sd(values: &[f64], average: f64) -> Result<f64, String> {
    if values.len() < 2 {
        return Ok(0.0);
    }
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let sd = variance.sqrt();
    if sd.is_finite() {
        Ok(sd)
    } else {
        Err("CoDeC item delta standard deviation is non-finite".to_owned())
    }
}

fn wilson_interval(successes: u64, total: u64) -> Result<CoDecWilsonIntervalV1, String> {
    if total == 0 || successes > total {
        return Err("CoDeC Wilson interval requires 0 <= successes <= positive total".to_owned());
    }
    let n = total as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    Ok(CoDecWilsonIntervalV1 {
        lower_95: (center - half).max(0.0),
        upper_95: (center + half).min(1.0),
    })
}

fn expected_seed_count(mode: CoDecSamplingModeV1) -> std::ops::RangeInclusive<usize> {
    match mode {
        CoDecSamplingModeV1::SingleSeedEfficiency => 1..=1,
        CoDecSamplingModeV1::FiveSeedPaperRobustness => 5..=5,
        CoDecSamplingModeV1::MultiSeedSensitivity => 2..=MAX_CONTEXT_SEEDS,
    }
}

fn validate_request(request: &CoDecDatasetRequestV1) -> Result<(), String> {
    if request.schema != CODEC_REQUEST_SCHEMA {
        return Err(format!("request schema must be {CODEC_REQUEST_SCHEMA}"));
    }
    for (value, label, maximum) in [
        (&request.dataset_id, "dataset_id", 128),
        (&request.dataset_version, "dataset_version", 64),
        (&request.context_policy_id, "context_policy_id", MAX_CONTEXT_POLICY_ID),
        (&request.inference_backend_id, "inference_backend_id", 128),
    ] {
        if !valid_id(value, maximum) {
            return Err(format!("{label} must be a valid bounded opaque ID"));
        }
    }
    for (value, label) in [
        (&request.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&request.tokenizer_sha256, "tokenizer_sha256"),
        (&request.chat_template_sha256, "chat_template_sha256"),
        (&request.dataset_manifest_sha256, "dataset_manifest_sha256"),
        (&request.dataset_item_set_sha256, "dataset_item_set_sha256"),
        (&request.context_policy_sha256, "context_policy_sha256"),
        (&request.inference_receipt_sha256, "inference_receipt_sha256"),
    ] {
        require_digest(value, label)?;
    }
    if !(1..=MAX_CONTEXT_ITEMS as u32).contains(&request.n_context_examples) {
        return Err(format!("n_context_examples must be in 1..={MAX_CONTEXT_ITEMS}"));
    }
    if !expected_seed_count(request.sampling_mode)
        .contains(&request.targets.first().map_or(0, |target| target.context_evaluations.len()))
    {
        return Err("context-seed count does not match the declared sampling mode".to_owned());
    }
    if request.dataset_item_ids.is_empty()
        || request.dataset_item_ids.len() > MAX_DATASET_ITEMS
        || request.targets.is_empty()
        || request.targets.len() > request.dataset_item_ids.len()
        || request.targets.len() > MAX_DATASET_ITEMS
    {
        return Err("dataset and target item counts are empty or exceed configured bounds".to_owned());
    }
    let expected_dataset_item_set = compute_codec_item_set_sha256(&request.dataset_item_ids)?;
    if expected_dataset_item_set != request.dataset_item_set_sha256 {
        return Err("dataset item-set digest does not match the declared dataset item IDs".to_owned());
    }
    let dataset_ids: BTreeSet<&str> = request.dataset_item_ids.iter().map(String::as_str).collect();
    let mut target_ids = BTreeSet::new();
    for target in &request.targets {
        if !valid_id(&target.item_id, 128)
            || !target_ids.insert(target.item_id.as_str())
            || !dataset_ids.contains(target.item_id.as_str())
        {
            return Err("target item IDs must be unique and contained in the candidate dataset".to_owned());
        }
        require_digest(&target.target_token_sequence_sha256, "target_token_sequence_sha256")?;
        require_digest(
            &target.baseline_target_token_sequence_sha256,
            "baseline_target_token_sequence_sha256",
        )?;
        if target.baseline_target_token_sequence_sha256 != target.target_token_sequence_sha256 {
            return Err("baseline target token sequence digest differs from the target token sequence".to_owned());
        }
        validate_logprobs(
            &target.baseline_target_token_logprobs,
            target.target_token_count,
            "baseline",
        )?;
        if !expected_seed_count(request.sampling_mode)
            .contains(&target.context_evaluations.len())
        {
            return Err("every target must have the same context-seed count required by sampling mode".to_owned());
        }
        let mut seeds = BTreeSet::new();
        for evaluation in &target.context_evaluations {
            if !seeds.insert(evaluation.context_seed) {
                return Err("context sampling seeds must be unique within a target".to_owned());
            }
            if evaluation.context_item_ids.len() != request.n_context_examples as usize
                || evaluation.context_item_ids.is_empty()
            {
                return Err("each context evaluation must include exactly n_context_examples items".to_owned());
            }
            if evaluation.context_item_ids.contains(&target.item_id) {
                return Err("a target item cannot appear in its own in-context examples".to_owned());
            }
            for context_id in &evaluation.context_item_ids {
                if !dataset_ids.contains(context_id.as_str()) {
                    return Err("sampled context item is not in the declared candidate dataset".to_owned());
                }
            }
            let expected_context_digest =
                compute_codec_context_item_set_sha256(&evaluation.context_item_ids)?;
            require_digest(&evaluation.context_item_set_sha256, "context_item_set_sha256")?;
            if evaluation.context_item_set_sha256 != expected_context_digest {
                return Err("context item-set digest mismatch".to_owned());
            }
            require_digest(
                &evaluation.target_token_sequence_sha256,
                "context target_token_sequence_sha256",
            )?;
            if evaluation.target_token_sequence_sha256 != target.target_token_sequence_sha256 {
                return Err("baseline and context evaluations must score the exact same target token sequence".to_owned());
            }
            validate_logprobs(
                &evaluation.target_token_logprobs,
                target.target_token_count,
                "in-context target",
            )?;
        }
    }
    Ok(())
}

/// Score one candidate dataset. For every target, average the baseline and
/// context-conditioned target-token log-probabilities; average per-seed deltas;
/// mark a target negative only when its mean delta is strictly below zero.
/// Dataset CoDeC score is the negative-target fraction.
pub fn score_codec_dataset(
    request: &CoDecDatasetRequestV1,
) -> Result<CoDecDatasetReportV1, String> {
    validate_request(request)?;
    let mut target_reports = Vec::with_capacity(request.targets.len());
    let mut all_deltas = Vec::with_capacity(request.targets.len());
    let mut negative_count = 0_u64;
    let mut target_token_total = 0_u64;

    for target in &request.targets {
        let baseline = mean(
            &target.baseline_target_token_logprobs,
            "baseline target-token log-probabilities",
        )?;
        let mut seed_deltas = Vec::with_capacity(target.context_evaluations.len());
        let mut context_means = Vec::with_capacity(target.context_evaluations.len());
        let mut contexts_by_seed = Vec::with_capacity(target.context_evaluations.len());
        for evaluation in &target.context_evaluations {
            let contextual = mean(
                &evaluation.target_token_logprobs,
                "in-context target-token log-probabilities",
            )?;
            context_means.push(contextual);
            seed_deltas.push(contextual - baseline);
            contexts_by_seed.push(evaluation.context_item_ids.clone());
        }
        let context_mean = mean(&context_means, "per-target in-context means")?;
        let mean_delta = mean(&seed_deltas, "per-target CoDeC deltas")?;
        let negative_delta = mean_delta < 0.0;
        if negative_delta {
            negative_count += 1;
        }
        target_token_total = target_token_total
            .checked_add(target.target_token_count)
            .ok_or_else(|| "CoDeC target-token count overflows".to_owned())?;
        all_deltas.push(mean_delta);
        target_reports.push(CoDecTargetReportV1 {
            item_id: target.item_id.clone(),
            target_token_count: target.target_token_count,
            baseline_mean_logprob_nats: baseline,
            context_mean_logprob_nats: context_mean,
            mean_delta_nats_per_token: mean_delta,
            per_seed_delta_nats_per_token: seed_deltas,
            negative_delta,
            context_seed_count: target.context_evaluations.len() as u64,
            context_item_ids_by_seed: contexts_by_seed,
        });
    }

    let delta_mean = mean(&all_deltas, "dataset target deltas")?;
    let delta_sd = sample_sd(&all_deltas, delta_mean)?;
    let negative_fraction = negative_count as f64 / all_deltas.len() as f64;
    let report_bytes = serde_json::to_vec(request)
        .map_err(|error| format!("cannot serialize CoDeC dataset request: {error}"))?;
    let target_item_set_sha256 = compute_codec_item_set_sha256(
        &request.targets.iter().map(|target| target.item_id.clone()).collect::<Vec<_>>(),
    )?;
    Ok(CoDecDatasetReportV1 {
        schema: CODEC_REPORT_SCHEMA.to_owned(),
        implementation_id: "symthaea.codec.paper_faithful.v1".to_owned(),
        dataset_id: request.dataset_id.clone(),
        dataset_version: request.dataset_version.clone(),
        model_checkpoint_sha256: request.model_checkpoint_sha256.clone(),
        tokenizer_sha256: request.tokenizer_sha256.clone(),
        chat_template_sha256: request.chat_template_sha256.clone(),
        dataset_manifest_sha256: request.dataset_manifest_sha256.clone(),
        dataset_item_set_sha256: request.dataset_item_set_sha256.clone(),
        target_item_set_sha256,
        context_policy_id: request.context_policy_id.clone(),
        context_policy_sha256: request.context_policy_sha256.clone(),
        inference_backend_id: request.inference_backend_id.clone(),
        inference_receipt_sha256: request.inference_receipt_sha256.clone(),
        n_context_examples: request.n_context_examples,
        sampling_mode: request.sampling_mode,
        context_seed_count_per_target: request
            .targets
            .first()
            .map_or(0, |target| target.context_evaluations.len()) as u64,
        target_item_count: request.targets.len() as u64,
        target_token_count_total: target_token_total,
        negative_delta_count: negative_count,
        codec_score_permille: negative_fraction * 1_000.0,
        item_level_wilson_95: wilson_interval(negative_count, request.targets.len() as u64)?,
        mean_delta_nats_per_token: delta_mean,
        standard_deviation_delta_across_items: delta_sd,
        median_delta_nats_per_token: median(&all_deltas)?,
        per_item: target_reports,
        independently_authenticated: false,
        confirmatory_eligible: false,
        item_interval_assumes_independent_targets: true,
        claim_boundary: "CoDeC is a dataset-level gray-box context intervention score. The report aggregates the fraction of target items whose average target-token log-probability decreases when same-dataset context is prepended. The item-level Wilson interval assumes independent target indicators; shared contexts, source clusters and dataset-level dependence can violate that assumption. A high score can reflect exact memorization or closely related/synthetic distribution exposure and is not proof of exact item membership. This crate validates supplied log-probability arrays and declared digests but does not run or authenticate the model/inference backend, prove that the context was actually prepended as declared, identify the training stage of exposure, or establish universal detection performance.".to_owned(),
    })
}

fn dataset_level_metrics(
    observations: &[CoDecDatasetLabelledScoreV1],
) -> Result<(f64, f64), String> {
    let positives = observations
        .iter()
        .filter(|observation| observation.known_training_exposure)
        .count();
    let negatives = observations.len() - positives;
    if positives == 0 || negatives == 0 {
        return Err("dataset-level metrics require exposed and unexposed datasets".to_owned());
    }
    let mut ascending = observations.to_vec();
    ascending.sort_by(|left, right| left.codec_score_permille.total_cmp(&right.codec_score_permille));
    let mut rank_sum = 0.0;
    let mut index = 0;
    while index < ascending.len() {
        let mut end = index + 1;
        while end < ascending.len()
            && ascending[end].codec_score_permille == ascending[index].codec_score_permille
        {
            end += 1;
        }
        let average_rank = ((index + 1 + end) as f64) / 2.0;
        let positive_ties = ascending[index..end]
            .iter()
            .filter(|observation| observation.known_training_exposure)
            .count();
        rank_sum += average_rank * positive_ties as f64;
        index = end;
    }
    let p = positives as f64;
    let n = negatives as f64;
    let auroc = (rank_sum - p * (p + 1.0) / 2.0) / (p * n);

    let mut descending = observations.to_vec();
    descending.sort_by(|left, right| right.codec_score_permille.total_cmp(&left.codec_score_permille));
    let mut true_positives = 0_u64;
    let mut false_positives = 0_u64;
    let mut previous_recall = 0.0;
    let mut auprc = 0.0;
    index = 0;
    while index < descending.len() {
        let mut end = index + 1;
        while end < descending.len()
            && descending[end].codec_score_permille == descending[index].codec_score_permille
        {
            end += 1;
        }
        for observation in &descending[index..end] {
            if observation.known_training_exposure {
                true_positives += 1;
            } else {
                false_positives += 1;
            }
        }
        let recall = true_positives as f64 / p;
        let precision = true_positives as f64 / (true_positives + false_positives) as f64;
        auprc += (recall - previous_recall) * precision;
        previous_recall = recall;
        index = end;
    }
    if !auroc.is_finite()
        || !auprc.is_finite()
        || !(0.0..=1.0).contains(&auroc)
        || !(0.0..=1.0).contains(&auprc)
    {
        return Err("dataset-level CoDeC AUROC/AUPRC is invalid".to_owned());
    }
    Ok((auroc, auprc))
}

/// Compute AUROC across independently labelled datasets. The unit of analysis
/// is each dataset report, never an individual target item.
pub fn codec_dataset_level_auroc(
    request: &CoDecDatasetLevelAucRequestV1,
) -> Result<CoDecDatasetLevelAucReportV1, String> {
    if request.schema != CODEC_DATASET_AUC_REQUEST_SCHEMA {
        return Err(format!("dataset-level AUC request schema must be {CODEC_DATASET_AUC_REQUEST_SCHEMA}"));
    }
    let observations = &request.datasets;
    if observations.len() < 2 || observations.len() > 100_000 {
        return Err("dataset-level AUROC requires 2..=100000 dataset observations".to_owned());
    }
    let mut ids = BTreeSet::new();
    for observation in observations {
        if !valid_id(&observation.dataset_id, 128)
            || !valid_id(&observation.source_family_id, 128)
            || !ids.insert(observation.dataset_id.as_str())
            || !observation.codec_score_permille.is_finite()
            || !(0.0..=1_000.0).contains(&observation.codec_score_permille)
        {
            return Err("dataset-level observations require unique IDs and scores in [0, 1000] permille".to_owned());
        }
    }
    let positives = observations
        .iter()
        .filter(|observation| observation.known_training_exposure)
        .count();
    let negatives = observations.len() - positives;
    if positives == 0 || negatives == 0 {
        return Err("dataset-level AUROC requires exposed and unexposed datasets".to_owned());
    }

    let (auroc, auprc) = dataset_level_metrics(observations)?;
    let source_family_count = observations
        .iter()
        .map(|observation| observation.source_family_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    Ok(CoDecDatasetLevelAucReportV1 {
        schema: CODEC_DATASET_AUC_SCHEMA.to_owned(),
        dataset_observation_count: observations.len() as u64,
        source_family_count: source_family_count as u64,
        exposed_dataset_count: positives as u64,
        unexposed_dataset_count: negatives as u64,
        dataset_level_auroc: auroc,
        dataset_level_auprc: auprc,
        unit_of_analysis: "independent_dataset".to_owned(),
        independently_authenticated: false,
        confirmatory_eligible: false,
        claim_boundary: "This AUROC compares independently supplied dataset-level CoDeC scores against dataset-level exposure labels. It is not item-level AUROC. Dataset independence, source-family dependence, label authenticity and the validity of each CoDeC score remain caller responsibilities.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn target(id: &str, baseline: f64, contextual: f64, seed: u64) -> CoDecTargetInputV1 {
        let target_logprobs = vec![baseline; 3];
        let context_logprobs = vec![contextual; 3];
        let contexts = vec![format!("context-{seed}-a")];
        CoDecTargetInputV1 {
            item_id: id.to_owned(),
            target_token_count: 3,
            target_token_sequence_sha256: digest('a'),
            baseline_target_token_sequence_sha256: digest('a'),
            baseline_target_token_logprobs: target_logprobs,
            context_evaluations: vec![CoDecContextEvaluationV1 {
                context_seed: seed,
                context_item_set_sha256: compute_codec_context_item_set_sha256(&contexts).unwrap(),
                target_token_sequence_sha256: digest('a'),
                context_item_ids: contexts,
                target_token_logprobs: context_logprobs,
            }],
        }
    }

    fn fixture() -> CoDecDatasetRequestV1 {
        let ids = vec![
            "item-1".to_owned(),
            "item-2".to_owned(),
            "context-1-a".to_owned(),
            "context-2-a".to_owned(),
        ];
        CoDecDatasetRequestV1 {
            schema: CODEC_REQUEST_SCHEMA.to_owned(),
            dataset_id: "fixture-dataset".to_owned(),
            dataset_version: "v1".to_owned(),
            model_checkpoint_sha256: digest('b'),
            tokenizer_sha256: digest('c'),
            chat_template_sha256: digest('d'),
            dataset_manifest_sha256: digest('e'),
            dataset_item_set_sha256: compute_codec_item_set_sha256(&ids).unwrap(),
            context_policy_id: "random-single-context".to_owned(),
            context_policy_sha256: digest('f'),
            inference_backend_id: "fixture-backend".to_owned(),
            inference_receipt_sha256: digest('1'),
            n_context_examples: 1,
            sampling_mode: CoDecSamplingModeV1::SingleSeedEfficiency,
            dataset_item_ids: ids,
            targets: vec![
                target("item-1", -1.0, -1.2, 1),
                target("item-2", -1.0, -0.8, 2),
            ],
        }
    }

    #[test]
    fn hand_computable_json_fixture_round_trips_and_scores() {
        let request: CoDecDatasetRequestV1 = serde_json::from_str(include_str!(
            "../fixtures/codec-dataset-request.hand.v1.json"
        ))
        .unwrap();
        let report = score_codec_dataset(&request).unwrap();
        assert_eq!(report.dataset_id, "codec-fixture-dataset");
        assert_eq!(report.target_item_count, 2);
        assert_eq!(report.negative_delta_count, 1);
        assert_eq!(report.codec_score_permille, 500.0);
        assert!(!report.confirmatory_eligible);
    }

    #[test]
    fn paper_score_is_fraction_of_strictly_negative_mean_deltas() {
        let report = score_codec_dataset(&fixture()).unwrap();
        assert_eq!(report.negative_delta_count, 1);
        assert_eq!(report.codec_score_permille, 500.0);
        assert!(!report.confirmatory_eligible);
    }

    #[test]
    fn strict_zero_delta_is_not_counted_as_negative() {
        let mut request = fixture();
        request.targets[0].context_evaluations[0].target_token_logprobs =
            request.targets[0].baseline_target_token_logprobs.clone();
        let report = score_codec_dataset(&request).unwrap();
        assert_eq!(report.negative_delta_count, 0);
        assert_eq!(report.codec_score_permille, 0.0);
    }

    #[test]
    fn baseline_and_context_token_sequence_digests_must_match() {
        let mut request = fixture();
        request.targets[0].context_evaluations[0].target_token_sequence_sha256 = digest('z');
        assert!(score_codec_dataset(&request).is_err());
        request = fixture();
        request.targets[0].baseline_target_token_sequence_sha256 = digest('z');
        assert!(score_codec_dataset(&request).is_err());
    }

    #[test]
    fn context_may_not_contain_its_target() {
        let mut request = fixture();
        request.targets[0].context_evaluations[0].context_item_ids[0] =
            request.targets[0].item_id.clone();
        request.targets[0].context_evaluations[0].context_item_set_sha256 =
            compute_codec_context_item_set_sha256(
                &request.targets[0].context_evaluations[0].context_item_ids,
            )
            .unwrap();
        assert!(score_codec_dataset(&request).is_err());
    }

    #[test]
    fn target_token_alignment_is_exact() {
        let mut request = fixture();
        request.targets[0].context_evaluations[0]
            .target_token_logprobs
            .pop();
        assert!(score_codec_dataset(&request).is_err());
    }

    #[test]
    fn context_ids_must_be_unique_and_in_candidate_dataset() {
        let mut request = fixture();
        request.targets[0].context_evaluations[0].context_item_ids =
            vec!["context-1-a".to_owned(), "context-1-a".to_owned()];
        request.n_context_examples = 2;
        request.targets[0].context_evaluations[0].context_item_set_sha256 =
            compute_codec_context_item_set_sha256(
                &request.targets[0].context_evaluations[0].context_item_ids,
            )
            .unwrap();
        assert!(score_codec_dataset(&request).is_err());
    }

    #[test]
    fn five_seed_mode_requires_five_unique_context_draws() {
        let mut request = fixture();
        request.sampling_mode = CoDecSamplingModeV1::FiveSeedPaperRobustness;
        assert!(score_codec_dataset(&request).is_err());
    }

    #[test]
    fn dataset_auc_is_computed_over_datasets_not_items() {
        let rows = vec![
            CoDecDatasetLabelledScoreV1 {
                dataset_id: "train-a".to_owned(),
                source_family_id: "family-a".to_owned(),
                known_training_exposure: true,
                codec_score_permille: 900.0,
            },
            CoDecDatasetLabelledScoreV1 {
                dataset_id: "train-b".to_owned(),
                source_family_id: "family-b".to_owned(),
                known_training_exposure: true,
                codec_score_permille: 800.0,
            },
            CoDecDatasetLabelledScoreV1 {
                dataset_id: "heldout-a".to_owned(),
                source_family_id: "family-c".to_owned(),
                known_training_exposure: false,
                codec_score_permille: 200.0,
            },
            CoDecDatasetLabelledScoreV1 {
                dataset_id: "heldout-b".to_owned(),
                source_family_id: "family-d".to_owned(),
                known_training_exposure: false,
                codec_score_permille: 300.0,
            },
        ];
        let report = codec_dataset_level_auroc(&CoDecDatasetLevelAucRequestV1 {
            schema: CODEC_DATASET_AUC_REQUEST_SCHEMA.to_owned(),
            datasets: rows,
        })
        .unwrap();
        assert_eq!(report.dataset_level_auroc, 1.0);
        assert_eq!(report.dataset_level_auprc, 1.0);
        assert_eq!(report.source_family_count, 4);
        assert_eq!(report.unit_of_analysis, "independent_dataset");
    }

    #[test]
    fn five_seed_paper_robustness_mode_averages_per_target_deltas() {
        let mut request = fixture();
        request.dataset_item_ids.extend([
            "context-seed-3".to_owned(),
            "context-seed-4".to_owned(),
            "context-seed-5".to_owned(),
        ]);
        request.dataset_item_set_sha256 =
            compute_codec_item_set_sha256(&request.dataset_item_ids).unwrap();
        request.sampling_mode = CoDecSamplingModeV1::FiveSeedPaperRobustness;
        for (index, target) in request.targets.iter_mut().enumerate() {
            let contextual = if index == 0 {
                [-1.1, -1.2, -1.3, -1.4, -1.5]
            } else {
                [-0.5, -0.6, -0.7, -0.8, -0.9]
            };
            target.context_evaluations.clear();
            for (seed_index, value) in contextual.into_iter().enumerate() {
                let context_id = if seed_index < 2 {
                    format!("context-{}-a", seed_index + 1)
                } else {
                    format!("context-seed-{}", seed_index + 1)
                };
                let ids = vec![context_id];
                target.context_evaluations.push(CoDecContextEvaluationV1 {
                    context_seed: seed_index as u64 + 10,
                    context_item_set_sha256: compute_codec_context_item_set_sha256(&ids).unwrap(),
                    context_item_ids: ids,
                    target_token_logprobs: vec![value; 3],
                });
            }
        }
        let report = score_codec_dataset(&request).unwrap();
        assert_eq!(report.context_seed_count_per_target, 5);
        assert_eq!(report.negative_delta_count, 1);
        assert_eq!(report.codec_score_permille, 500.0);
        assert_eq!(report.per_item[0].per_seed_delta_nats_per_token.len(), 5);
    }

    #[test]
    fn duplicate_dataset_labels_are_rejected() {
        let mut rows = vec![
            CoDecDatasetLabelledScoreV1 {
                dataset_id: "same".to_owned(),
                known_training_exposure: true,
                codec_score_permille: 900.0,
            },
            CoDecDatasetLabelledScoreV1 {
                dataset_id: "same".to_owned(),
                known_training_exposure: false,
                codec_score_permille: 100.0,
            },
        ];
        let request = CoDecDatasetLevelAucRequestV1 {
            schema: CODEC_DATASET_AUC_REQUEST_SCHEMA.to_owned(),
            datasets: rows.clone(),
        };
        assert!(codec_dataset_level_auroc(&request).is_err());
        rows[1].dataset_id = "different".to_owned();
        rows[1].source_family_id = "family-b".to_owned();
        rows[1].codec_score_permille = 100.0;
        let request = CoDecDatasetLevelAucRequestV1 {
            schema: CODEC_DATASET_AUC_REQUEST_SCHEMA.to_owned(),
            datasets: rows,
        };
        assert_eq!(codec_dataset_level_auroc(&request).unwrap().dataset_level_auroc, 1.0);
    }
}
