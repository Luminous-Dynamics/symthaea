use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde::Serialize;
use sha2::{Digest, Sha256};

pub const DEFAULT_BOOTSTRAP_SAMPLES: usize = 1_000;
const MIN_WORST_GROUP_ITEMS: usize = 20;

#[derive(Debug, Clone)]
struct MetadataItem {
    id: String,
    target: Option<f64>,
    group: String,
}

#[derive(Debug, Clone)]
struct Prediction {
    id: String,
    score: f64,
}

#[derive(Debug)]
struct ParsedMetadata {
    items: Vec<MetadataItem>,
    id_column: String,
    target_column: Option<String>,
}

#[derive(Debug)]
struct ParsedPredictions {
    items: Vec<Prediction>,
    id_column: String,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Metrics {
    pub count: usize,
    pub srcc: Option<f64>,
    pub lcc: Option<f64>,
    pub kendall_tau_b: Option<f64>,
    pub mse: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricInterval {
    pub lower_95: f64,
    pub upper_95: f64,
    pub valid_replicates: usize,
    pub total_replicates: usize,
    pub seed: u64,
    pub sampling_unit: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GroupSlice {
    pub group: String,
    pub item_count: usize,
    pub metrics: Metrics,
    pub included_in_worst_group: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemReport {
    pub system_name: String,
    pub prediction_sha256: String,
    pub metrics: Metrics,
    pub srcc_interval_95: Option<MetricInterval>,
    pub groups: Vec<GroupSlice>,
    pub worst_group: Option<String>,
    pub worst_group_srcc: Option<f64>,
    pub worst_group_minimum_items: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoreReport {
    pub schema_version: String,
    pub mode: String,
    pub metadata_sha256: String,
    pub metadata_id_column: String,
    pub target_score_column: String,
    pub prediction_id_column: String,
    pub item_count: usize,
    pub metrics: Metrics,
    pub system: SystemReport,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeltaInterval {
    pub lower_95: f64,
    pub upper_95: f64,
    pub valid_replicates: usize,
    pub total_replicates: usize,
    pub seed: u64,
    pub sampling_unit: String,
    pub positive_means_candidate_better_srcc: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonReport {
    pub schema_version: String,
    pub mode: String,
    pub metadata_sha256: String,
    pub metadata_id_column: String,
    pub target_score_column: String,
    pub baseline_prediction_id_column: String,
    pub candidate_prediction_id_column: String,
    pub item_count: usize,
    pub baseline: SystemReport,
    pub candidate: SystemReport,
    pub candidate_minus_baseline_srcc: Option<f64>,
    pub paired_srcc_delta_interval_95: Option<DeltaInterval>,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubmissionReport {
    pub schema_version: String,
    pub mode: String,
    pub evaluation_manifest_sha256: String,
    pub prediction_sha256: String,
    pub evaluation_id_column: String,
    pub prediction_id_column: String,
    pub expected_item_count: usize,
    pub received_item_count: usize,
    pub minimum_prediction: f64,
    pub maximum_prediction: f64,
    pub exact_id_coverage: bool,
    pub output_score_range: String,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct ItemPair {
    target: f64,
    prediction: f64,
}

pub fn score_csv(
    metadata_bytes: &[u8],
    prediction_bytes: &[u8],
    system_name: &str,
    bootstrap_samples: usize,
    seed: u64,
) -> Result<ScoreReport, String> {
    let metadata = parse_metadata(metadata_bytes, true)?;
    let predictions = parse_predictions(prediction_bytes)?;
    let aligned = align_predictions(&metadata.items, &predictions.items)?;
    let pairs: Vec<ItemPair> = aligned
        .iter()
        .map(|(item, prediction)| ItemPair {
            target: item.target.expect("labels required in score mode"),
            prediction: prediction.score,
        })
        .collect();

    let system = build_system_report(
        system_name,
        &metadata.items,
        prediction_bytes,
        &pairs,
        bootstrap_samples,
        seed,
    )?;

    Ok(ScoreReport {
        schema_version: "mel-xacle-audit-report-v1".to_owned(),
        mode: "score".to_owned(),
        metadata_sha256: sha256_hex(metadata_bytes),
        metadata_id_column: metadata.id_column,
        target_score_column: metadata
            .target_column
            .expect("labelled mode requires a target score column"),
        prediction_id_column: predictions.id_column,
        item_count: metadata.items.len(),
        metrics: system.metrics,
        system,
        claim_ceiling: score_claim_ceiling(),
    })
}

pub fn compare_csv(
    metadata_bytes: &[u8],
    baseline_bytes: &[u8],
    baseline_name: &str,
    candidate_bytes: &[u8],
    candidate_name: &str,
    bootstrap_samples: usize,
    seed: u64,
) -> Result<ComparisonReport, String> {
    if bootstrap_samples == 0 {
        return Err("bootstrap_samples must be greater than zero".to_owned());
    }
    let metadata = parse_metadata(metadata_bytes, true)?;
    let baseline_predictions = parse_predictions(baseline_bytes)?;
    let candidate_predictions = parse_predictions(candidate_bytes)?;
    let baseline_aligned = align_predictions(&metadata.items, &baseline_predictions.items)?;
    let candidate_aligned = align_predictions(&metadata.items, &candidate_predictions.items)?;

    let baseline_pairs: Vec<ItemPair> = baseline_aligned
        .iter()
        .map(|(item, prediction)| ItemPair {
            target: item.target.expect("labels required in compare mode"),
            prediction: prediction.score,
        })
        .collect();
    let candidate_pairs: Vec<ItemPair> = candidate_aligned
        .iter()
        .map(|(item, prediction)| ItemPair {
            target: item.target.expect("labels required in compare mode"),
            prediction: prediction.score,
        })
        .collect();

    let baseline = build_system_report(
        baseline_name,
        &metadata.items,
        baseline_bytes,
        &baseline_pairs,
        0,
        seed,
    )?;
    let candidate = build_system_report(
        candidate_name,
        &metadata.items,
        candidate_bytes,
        &candidate_pairs,
        0,
        seed,
    )?;

    let candidate_minus_baseline_srcc = match (baseline.metrics.srcc, candidate.metrics.srcc) {
        (Some(base), Some(candidate)) => Some(candidate - base),
        _ => None,
    };
    let interval =
        paired_srcc_delta_interval(&baseline_pairs, &candidate_pairs, bootstrap_samples, seed);

    Ok(ComparisonReport {
        schema_version: "mel-xacle-audit-report-v1".to_owned(),
        mode: "compare".to_owned(),
        metadata_sha256: sha256_hex(metadata_bytes),
        metadata_id_column: metadata.id_column,
        target_score_column: metadata
            .target_column
            .expect("labelled mode requires a target score column"),
        baseline_prediction_id_column: baseline_predictions.id_column,
        candidate_prediction_id_column: candidate_predictions.id_column,
        item_count: metadata.items.len(),
        baseline,
        candidate,
        candidate_minus_baseline_srcc,
        paired_srcc_delta_interval_95: interval,
        claim_ceiling: score_claim_ceiling(),
    })
}

pub fn validate_submission_csv(
    evaluation_manifest_bytes: &[u8],
    prediction_bytes: &[u8],
    system_name: &str,
) -> Result<SubmissionReport, String> {
    let metadata = parse_metadata(evaluation_manifest_bytes, false)?;
    let predictions = parse_predictions(prediction_bytes)?;
    let _aligned = align_predictions(&metadata.items, &predictions.items)?;

    let minimum_prediction = predictions
        .items
        .iter()
        .map(|prediction| prediction.score)
        .fold(f64::INFINITY, f64::min);
    let maximum_prediction = predictions
        .items
        .iter()
        .map(|prediction| prediction.score)
        .fold(f64::NEG_INFINITY, f64::max);

    Ok(SubmissionReport {
        schema_version: "mel-xacle-audit-report-v1".to_owned(),
        mode: "validate-submission".to_owned(),
        evaluation_manifest_sha256: sha256_hex(evaluation_manifest_bytes),
        prediction_sha256: sha256_hex(prediction_bytes),
        evaluation_id_column: metadata.id_column,
        prediction_id_column: predictions.id_column,
        expected_item_count: metadata.items.len(),
        received_item_count: predictions.items.len(),
        minimum_prediction,
        maximum_prediction,
        exact_id_coverage: true,
        output_score_range: "finite [0, 10]".to_owned(),
        claim_ceiling: vec![
            format!("submission integrity only for {system_name}"),
            "no hidden evaluation labels were accessed or inferred".to_owned(),
            "this report does not predict official challenge performance or rank".to_owned(),
        ],
    })
}

fn parse_metadata(bytes: &[u8], labels_required: bool) -> Result<ParsedMetadata, String> {
    let mut reader = csv::ReaderBuilder::new().flexible(false).from_reader(bytes);
    let headers = reader
        .headers()
        .map_err(|error| format!("invalid metadata CSV header: {error}"))?
        .clone();

    let (id_index, id_column) = header_alias_index(
        &headers,
        &["wave_file_name", "wav_file_name"],
        "metadata item ID",
        true,
    )?
    .expect("required metadata ID column");
    let target = header_alias_index(
        &headers,
        &["average_score", "average_semantic_alignement_score"],
        "target score",
        labels_required,
    )?;
    let (target_index, target_column) = match target {
        Some((index, name)) => (Some(index), Some(name)),
        None => (None, None),
    };
    let model_id_index = header_index(&headers, "model_id", false)?;
    let model_name_index = header_index(&headers, "model_name", false)?;

    let mut seen = BTreeSet::new();
    let mut metadata = Vec::new();
    for (row_index, record) in reader.records().enumerate() {
        let record = record.map_err(|error| {
            format!(
                "invalid metadata CSV record at row {}: {error}",
                row_index + 2
            )
        })?;
        let id = field(&record, id_index, &id_column, row_index + 2)?;
        if id.trim().is_empty() {
            return Err(format!(
                "empty {} at metadata row {}",
                id_column,
                row_index + 2
            ));
        }
        if !seen.insert(id.to_owned()) {
            return Err(format!("duplicate metadata {}: {id}", id_column));
        }

        let target_value = if let Some(index) = target_index {
            let score_column = target_column.as_deref().expect("target column name");
            let value = field(&record, index, score_column, row_index + 2)?;
            Some(parse_score(value, score_column, row_index + 2)?)
        } else {
            None
        };

        let model_id =
            optional_field(&record, model_id_index).filter(|value| !value.trim().is_empty());
        let model_name =
            optional_field(&record, model_name_index).filter(|value| !value.trim().is_empty());
        let group = if let Some(model_id) = model_id {
            format!("model_id:{model_id}")
        } else if let Some(model_name) = model_name {
            format!("model_name:{model_name}")
        } else {
            "generator:unknown".to_owned()
        };

        metadata.push(MetadataItem {
            id: id.to_owned(),
            target: target_value,
            group,
        });
    }

    if metadata.is_empty() {
        return Err("metadata CSV contains no data rows".to_owned());
    }
    Ok(ParsedMetadata {
        items: metadata,
        id_column,
        target_column,
    })
}

fn parse_predictions(bytes: &[u8]) -> Result<ParsedPredictions, String> {
    let mut reader = csv::ReaderBuilder::new().flexible(false).from_reader(bytes);
    let headers = reader
        .headers()
        .map_err(|error| format!("invalid prediction CSV header: {error}"))?
        .clone();
    let (id_index, id_column) = header_alias_index(
        &headers,
        &["wav_file_name", "wave_file_name"],
        "prediction item ID",
        true,
    )?
    .expect("required prediction ID column");
    let score_index =
        header_index(&headers, "pred_score", true)?.expect("required prediction score header");

    let mut seen = BTreeSet::new();
    let mut predictions = Vec::new();
    for (row_index, record) in reader.records().enumerate() {
        let record = record.map_err(|error| {
            format!(
                "invalid prediction CSV record at row {}: {error}",
                row_index + 2
            )
        })?;
        let id = field(&record, id_index, &id_column, row_index + 2)?;
        if id.trim().is_empty() {
            return Err(format!(
                "empty prediction {} at row {}",
                id_column,
                row_index + 2
            ));
        }
        if !seen.insert(id.to_owned()) {
            return Err(format!("duplicate prediction {}: {id}", id_column));
        }
        let raw_score = field(&record, score_index, "pred_score", row_index + 2)?;
        let score = parse_score(raw_score, "pred_score", row_index + 2)?;
        predictions.push(Prediction {
            id: id.to_owned(),
            score,
        });
    }
    if predictions.is_empty() {
        return Err("prediction CSV contains no data rows".to_owned());
    }
    Ok(ParsedPredictions {
        items: predictions,
        id_column,
    })
}

fn align_predictions<'a>(
    metadata: &'a [MetadataItem],
    predictions: &'a [Prediction],
) -> Result<Vec<(&'a MetadataItem, &'a Prediction)>, String> {
    let metadata_ids: BTreeSet<&str> = metadata.iter().map(|item| item.id.as_str()).collect();
    let prediction_by_id: BTreeMap<&str, &Prediction> = predictions
        .iter()
        .map(|prediction| (prediction.id.as_str(), prediction))
        .collect();

    let missing: Vec<&str> = metadata_ids
        .iter()
        .copied()
        .filter(|id| !prediction_by_id.contains_key(id))
        .take(10)
        .collect();
    let extra: Vec<&str> = prediction_by_id
        .keys()
        .copied()
        .filter(|id| !metadata_ids.contains(id))
        .take(10)
        .collect();

    if !missing.is_empty() || !extra.is_empty() || metadata.len() != predictions.len() {
        return Err(format!(
            "prediction coverage mismatch: expected {} unique IDs, received {}; missing (up to 10)={missing:?}; extra (up to 10)={extra:?}",
            metadata.len(),
            predictions.len()
        ));
    }

    metadata
        .iter()
        .map(|item| {
            prediction_by_id
                .get(item.id.as_str())
                .map(|prediction| (item, *prediction))
                .ok_or_else(|| format!("internal alignment failure for {}", item.id))
        })
        .collect()
}

fn build_system_report(
    system_name: &str,
    metadata: &[MetadataItem],
    prediction_bytes: &[u8],
    pairs: &[ItemPair],
    bootstrap_samples: usize,
    seed: u64,
) -> Result<SystemReport, String> {
    if metadata.len() != pairs.len() {
        return Err("system report inputs have inconsistent lengths".to_owned());
    }
    let metrics = calculate_metrics(pairs);
    let srcc_interval = if bootstrap_samples > 0 {
        bootstrap_srcc_interval(pairs, bootstrap_samples, seed)
    } else {
        None
    };

    let mut grouped: BTreeMap<String, Vec<ItemPair>> = BTreeMap::new();
    for (item, pair) in metadata.iter().zip(pairs) {
        grouped.entry(item.group.clone()).or_default().push(*pair);
    }

    let groups: Vec<GroupSlice> = grouped
        .iter()
        .map(|(group, group_pairs)| {
            let group_metrics = calculate_metrics(group_pairs);
            GroupSlice {
                group: group.clone(),
                item_count: group_pairs.len(),
                metrics: group_metrics,
                included_in_worst_group: group_pairs.len() >= MIN_WORST_GROUP_ITEMS,
            }
        })
        .collect();

    let worst = groups
        .iter()
        .filter(|group| group.included_in_worst_group)
        .filter_map(|group| group.metrics.srcc.map(|score| (group.group.clone(), score)))
        .min_by(|left, right| left.1.total_cmp(&right.1));

    Ok(SystemReport {
        system_name: system_name.to_owned(),
        prediction_sha256: sha256_hex(prediction_bytes),
        metrics,
        srcc_interval_95: srcc_interval,
        groups,
        worst_group: worst.as_ref().map(|value| value.0.clone()),
        worst_group_srcc: worst.map(|value| value.1),
        worst_group_minimum_items: MIN_WORST_GROUP_ITEMS,
    })
}

fn calculate_metrics(pairs: &[ItemPair]) -> Metrics {
    if pairs.is_empty() {
        return Metrics {
            count: 0,
            srcc: None,
            lcc: None,
            kendall_tau_b: None,
            mse: None,
        };
    }
    let targets: Vec<f64> = pairs.iter().map(|pair| pair.target).collect();
    let predictions: Vec<f64> = pairs.iter().map(|pair| pair.prediction).collect();
    let squared_error = pairs
        .iter()
        .map(|pair| (pair.prediction - pair.target).powi(2))
        .sum::<f64>();

    Metrics {
        count: pairs.len(),
        srcc: spearman(&targets, &predictions),
        lcc: pearson(&targets, &predictions),
        kendall_tau_b: kendall_tau_b(&targets, &predictions),
        mse: Some(squared_error / pairs.len() as f64),
    }
}

fn spearman(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    pearson(&average_ranks(left), &average_ranks(right))
}

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (left_value, right_value) in left.iter().zip(right) {
        let left_delta = left_value - left_mean;
        let right_delta = right_value - right_mean;
        covariance += left_delta * right_delta;
        left_variance += left_delta.powi(2);
        right_variance += right_delta.powi(2);
    }
    let denominator = (left_variance * right_variance).sqrt();
    if denominator <= f64::EPSILON {
        None
    } else {
        Some(covariance / denominator)
    }
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < indices.len() {
        let mut end = start + 1;
        while end < indices.len() && values[indices[start]] == values[indices[end]] {
            end += 1;
        }
        let average_rank = (start + 1 + end) as f64 / 2.0;
        for index in &indices[start..end] {
            ranks[*index] = average_rank;
        }
        start = end;
    }
    ranks
}

fn kendall_tau_b(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let n = left.len();
    let total_pairs = (n * (n - 1) / 2) as f64;
    let mut concordant = 0.0_f64;
    let mut discordant = 0.0_f64;
    let mut ties_left = 0.0_f64;
    let mut ties_right = 0.0_f64;

    for i in 0..n {
        for j in (i + 1)..n {
            let left_order = left[i]
                .partial_cmp(&left[j])
                .unwrap_or(std::cmp::Ordering::Equal);
            let right_order = right[i]
                .partial_cmp(&right[j])
                .unwrap_or(std::cmp::Ordering::Equal);
            match (left_order, right_order) {
                (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => {
                    ties_left += 1.0;
                    ties_right += 1.0;
                }
                (std::cmp::Ordering::Equal, _) => ties_left += 1.0,
                (_, std::cmp::Ordering::Equal) => ties_right += 1.0,
                (a, b) if a == b => concordant += 1.0,
                _ => discordant += 1.0,
            }
        }
    }

    let left_comparable = total_pairs - ties_left;
    let right_comparable = total_pairs - ties_right;
    let denominator = (left_comparable * right_comparable).sqrt();
    if denominator <= f64::EPSILON {
        None
    } else {
        Some((concordant - discordant) / denominator)
    }
}

fn bootstrap_srcc_interval(
    pairs: &[ItemPair],
    bootstrap_samples: usize,
    seed: u64,
) -> Option<MetricInterval> {
    if pairs.len() < 3 || bootstrap_samples == 0 {
        return None;
    }
    let mut rng = SplitMix64::new(seed);
    let mut values = Vec::with_capacity(bootstrap_samples);
    for _ in 0..bootstrap_samples {
        let sample = sample_pairs(pairs, &mut rng);
        if let Some(srcc) = calculate_metrics_fast_srcc(&sample) {
            values.push(srcc);
        }
    }
    interval_from_values(values, bootstrap_samples, seed, "audio-text item")
}

fn paired_srcc_delta_interval(
    baseline: &[ItemPair],
    candidate: &[ItemPair],
    bootstrap_samples: usize,
    seed: u64,
) -> Option<DeltaInterval> {
    if baseline.len() != candidate.len() || baseline.len() < 3 || bootstrap_samples == 0 {
        return None;
    }
    let mut rng = SplitMix64::new(seed);
    let mut deltas = Vec::with_capacity(bootstrap_samples);
    for _ in 0..bootstrap_samples {
        let indices: Vec<usize> = (0..baseline.len())
            .map(|_| rng.index(baseline.len()))
            .collect();
        let baseline_sample: Vec<ItemPair> = indices.iter().map(|index| baseline[*index]).collect();
        let candidate_sample: Vec<ItemPair> =
            indices.iter().map(|index| candidate[*index]).collect();
        if let (Some(base), Some(candidate)) = (
            calculate_metrics_fast_srcc(&baseline_sample),
            calculate_metrics_fast_srcc(&candidate_sample),
        ) {
            deltas.push(candidate - base);
        }
    }
    let (lower_95, upper_95) = percentile_interval(&mut deltas, bootstrap_samples)?;
    Some(DeltaInterval {
        lower_95,
        upper_95,
        valid_replicates: deltas.len(),
        total_replicates: bootstrap_samples,
        seed,
        sampling_unit: "paired audio-text item; same resampled IDs for both systems".to_owned(),
        positive_means_candidate_better_srcc: true,
    })
}

fn calculate_metrics_fast_srcc(pairs: &[ItemPair]) -> Option<f64> {
    if pairs.len() < 3 {
        return None;
    }
    let targets: Vec<f64> = pairs.iter().map(|pair| pair.target).collect();
    let predictions: Vec<f64> = pairs.iter().map(|pair| pair.prediction).collect();
    spearman(&targets, &predictions)
}

fn sample_pairs(pairs: &[ItemPair], rng: &mut SplitMix64) -> Vec<ItemPair> {
    (0..pairs.len())
        .map(|_| pairs[rng.index(pairs.len())])
        .collect()
}

fn interval_from_values(
    mut values: Vec<f64>,
    total_replicates: usize,
    seed: u64,
    sampling_unit: &str,
) -> Option<MetricInterval> {
    let valid_replicates = values.len();
    let (lower_95, upper_95) = percentile_interval(&mut values, total_replicates)?;
    Some(MetricInterval {
        lower_95,
        upper_95,
        valid_replicates,
        total_replicates,
        seed,
        sampling_unit: sampling_unit.to_owned(),
    })
}

fn percentile_interval(values: &mut [f64], requested_replicates: usize) -> Option<(f64, f64)> {
    if values.len() < 50 || requested_replicates == 0 {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let low_index = ((values.len() - 1) as f64 * 0.025).floor() as usize;
    let high_index = ((values.len() - 1) as f64 * 0.975).ceil() as usize;
    Some((values[low_index], values[high_index]))
}

fn field<'a>(
    record: &'a csv::StringRecord,
    index: usize,
    name: &str,
    row_number: usize,
) -> Result<&'a str, String> {
    record
        .get(index)
        .ok_or_else(|| format!("missing {name} at CSV row {row_number}"))
}

fn optional_field(record: &csv::StringRecord, index: Option<usize>) -> Option<&str> {
    index.and_then(|index| record.get(index))
}

fn header_alias_index(
    headers: &csv::StringRecord,
    aliases: &[&str],
    logical_name: &str,
    required: bool,
) -> Result<Option<(usize, String)>, String> {
    let mut present = Vec::new();
    for alias in aliases {
        if let Some(index) = header_index(headers, alias, false)? {
            present.push((index, (*alias).to_owned()));
        }
    }
    if present.len() > 1 {
        let names: Vec<&str> = present.iter().map(|(_, name)| name.as_str()).collect();
        return Err(format!(
            "CSV contains multiple columns for {logical_name}: {names:?}; keep exactly one"
        ));
    }
    match present.into_iter().next() {
        Some(column) => Ok(Some(column)),
        None if required => Err(format!(
            "CSV is missing {logical_name}; accepted column names: {aliases:?}"
        )),
        None => Ok(None),
    }
}

fn header_index(
    headers: &csv::StringRecord,
    name: &str,
    required: bool,
) -> Result<Option<usize>, String> {
    let mut matches = headers
        .iter()
        .enumerate()
        .filter_map(|(index, header)| (header == name).then_some(index));
    let first = matches.next();
    if matches.next().is_some() {
        return Err(format!("CSV contains duplicate header {name}"));
    }
    if required && first.is_none() {
        return Err(format!("CSV is missing required header {name}"));
    }
    Ok(first)
}

fn parse_score(value: &str, name: &str, row_number: usize) -> Result<f64, String> {
    let score = value
        .parse::<f64>()
        .map_err(|_| format!("invalid {name} at row {row_number}: {value:?}"))?;
    if !score.is_finite() || !(0.0..=10.0).contains(&score) {
        return Err(format!(
            "{name} must be finite and in [0, 10] at row {row_number}; got {score}"
        ));
    }
    Ok(score)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest.iter() {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn score_claim_ceiling() -> Vec<String> {
    vec![
        "metrics apply only to the supplied labelled metadata and exact prediction CSV bytes".to_owned(),
        "the audit does not prove generator-held-out split lineage or hidden-evaluation performance".to_owned(),
        "group metrics depend on correct model_id/model_name metadata".to_owned(),
        "bootstrap resamples audio-text items, not listener identities or independent human cohorts".to_owned(),
        "this tool does not access or infer hidden XACLE evaluation labels".to_owned(),
    ]
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const METADATA: &str = "wave_file_name,text,average_semantic_alignement_score,model_id,model_name\n\
a0.wav,prompt 0,0,m1,Generator One\n\
a1.wav,prompt 1,1,m1,Generator One\n\
a2.wav,prompt 2,2,m1,Generator One\n\
a3.wav,prompt 3,3,m1,Generator One\n\
a4.wav,prompt 4,4,m1,Generator One\n\
a5.wav,prompt 5,5,m1,Generator One\n\
b0.wav,prompt 6,6,m2,Generator Two\n\
b1.wav,prompt 7,7,m2,Generator Two\n\
b2.wav,prompt 8,8,m2,Generator Two\n\
b3.wav,prompt 9,9,m2,Generator Two\n\
b4.wav,prompt 10,10,m2,Generator Two\n\
b5.wav,prompt 11,9,m2,Generator Two\n";

    const BASELINE: &str = "wav_file_name,pred_score\n\
a0.wav,0\n\
a1.wav,1\n\
a2.wav,2\n\
a3.wav,3\n\
a4.wav,4\n\
a5.wav,5\n\
b0.wav,6\n\
b1.wav,7\n\
b2.wav,8\n\
b3.wav,9\n\
b4.wav,10\n\
b5.wav,8\n";

    const CANDIDATE: &str = "wav_file_name,pred_score\n\
a0.wav,0\n\
a1.wav,1\n\
a2.wav,2\n\
a3.wav,3\n\
a4.wav,4\n\
a5.wav,5\n\
b0.wav,6\n\
b1.wav,7\n\
b2.wav,8\n\
b3.wav,9\n\
b4.wav,10\n\
b5.wav,9\n";

    #[test]
    fn score_uses_ids_and_reports_generator_slices() {
        let report = score_csv(
            METADATA.as_bytes(),
            CANDIDATE.as_bytes(),
            "candidate",
            100,
            42,
        )
        .expect("fixture should score");
        assert_eq!(report.item_count, 12);
        assert_eq!(report.metrics.count, 12);
        assert_eq!(report.system.groups.len(), 2);
        assert_eq!(report.system.groups[0].group, "model_id:m1");
        assert_eq!(report.system.groups[0].metrics.srcc, Some(1.0));
    }

    #[test]
    fn compare_uses_paired_item_bootstrap() {
        let report = compare_csv(
            METADATA.as_bytes(),
            BASELINE.as_bytes(),
            "baseline",
            CANDIDATE.as_bytes(),
            "candidate",
            200,
            7,
        )
        .expect("fixture should compare");
        assert_eq!(report.item_count, 12);
        assert!(report.candidate_minus_baseline_srcc.unwrap() > 0.0);
        assert!(report.paired_srcc_delta_interval_95.is_some());
    }

    #[test]
    fn alternate_prediction_identifier_alias_is_accepted_and_reported() {
        let predictions = b"wave_file_name,pred_score\na0.wav,0\na1.wav,1\na2.wav,2\na3.wav,3\na4.wav,4\na5.wav,5\nb0.wav,6\nb1.wav,7\nb2.wav,8\nb3.wav,9\nb4.wav,10\nb5.wav,9\n";
        let report = score_csv(METADATA.as_bytes(), predictions, "alternate-output", 0, 11)
            .expect("wave_file_name prediction alias should be accepted");
        assert_eq!(report.metadata_id_column, "wave_file_name");
        assert_eq!(report.prediction_id_column, "wave_file_name");
        assert_eq!(report.target_score_column, "average_semantic_alignement_score");
        assert_eq!(report.metrics.srcc, Some(1.0));
    }

    #[test]
    fn baseline_preprocessed_metadata_dialect_is_accepted_and_reported() {
        let metadata = b"wav_file_name,text,average_score\na.wav,calm piano,6.0\nb.wav,bright jazz,8.0\n";
        let predictions = b"wav_file_name,pred_score\nb.wav,8.0\na.wav,6.0\n";
        let report = score_csv(metadata, predictions, "baseline-dialect", 0, 9)
            .expect("baseline's preprocessed CSV dialect should be accepted");
        assert_eq!(report.metadata_id_column, "wav_file_name");
        assert_eq!(report.target_score_column, "average_score");
        assert_eq!(report.prediction_id_column, "wav_file_name");
        assert_eq!(report.metrics.count, 2);
        assert_eq!(report.metrics.srcc, Some(1.0));
    }

    #[test]
    fn ambiguous_identifier_aliases_are_rejected() {
        let metadata = b"wave_file_name,wav_file_name,average_score\na.wav,a.wav,6.0\n";
        let error = parse_metadata(metadata, true)
            .expect_err("both identifier aliases must be rejected as ambiguous");
        assert!(error.contains("multiple columns for metadata item ID"));
    }

    #[test]
    fn ambiguous_target_aliases_are_rejected() {
        let metadata = b"wave_file_name,average_score,average_semantic_alignement_score\na.wav,6.0,6.0\n";
        let error = parse_metadata(metadata, true)
            .expect_err("both target aliases must be rejected as ambiguous");
        assert!(error.contains("multiple columns for target score"));
    }

    #[test]
    fn duplicate_metadata_ids_are_rejected() {
        let duplicate = b"wave_file_name,average_score\na.wav,1\na.wav,2\n";
        assert!(
            parse_metadata(duplicate, true)
                .expect_err("duplicate ID must fail")
                .contains("duplicate metadata")
        );
    }

    #[test]
    fn duplicate_prediction_ids_are_rejected() {
        let duplicate = b"wav_file_name,pred_score\na.wav,1\na.wav,2\n";
        assert!(
            parse_predictions(duplicate)
                .expect_err("duplicate prediction ID must fail")
                .contains("duplicate prediction")
        );
    }

    #[test]
    fn missing_or_extra_prediction_ids_are_rejected() {
        let missing = b"wav_file_name,pred_score\na0.wav,1\n";
        assert!(
            score_csv(METADATA.as_bytes(), missing, "x", 0, 42)
                .expect_err("coverage mismatch must fail")
                .contains("coverage mismatch")
        );

        let extra = format!("{BASELINE}extra.wav,5\n");
        assert!(
            score_csv(METADATA.as_bytes(), extra.as_bytes(), "x", 0, 42)
                .expect_err("extra ID must fail")
                .contains("coverage mismatch")
        );
    }

    #[test]
    fn non_finite_and_out_of_range_scores_are_rejected() {
        for score in ["NaN", "inf", "-0.1", "10.1"] {
            let csv = format!("wav_file_name,pred_score\na0.wav,{score}\n");
            let error =
                parse_predictions(csv.as_bytes()).expect_err("invalid output should be rejected");
            assert!(error.contains("finite and in [0, 10]"));
        }
    }

    #[test]
    fn kendall_tau_b_handles_ties() {
        let value = kendall_tau_b(&[1.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 2.0, 3.0])
            .expect("non-degenerate tau-b");
        assert!(value.is_finite());
        assert!((-1.0..=1.0).contains(&value));
    }

    #[test]
    fn blind_submission_validation_does_not_require_labels() {
        let evaluation = b"wave_file_name,text\nx0.wav,ambient piano\nx1.wav,fast jazz\n";
        let predictions = b"wav_file_name,pred_score\nx1.wav,8.2\nx0.wav,6.4\n";
        let report = validate_submission_csv(evaluation, predictions, "test-system")
            .expect("blind submission should validate regardless of row order");
        assert_eq!(report.expected_item_count, 2);
        assert_eq!(report.received_item_count, 2);
        assert!(report.exact_id_coverage);
    }

    #[test]
    fn blind_submission_rejects_missing_rows() {
        let evaluation = b"wave_file_name,text\nx0.wav,ambient piano\nx1.wav,fast jazz\n";
        let predictions = b"wav_file_name,pred_score\nx0.wav,6.4\n";
        assert!(
            validate_submission_csv(evaluation, predictions, "system")
                .expect_err("missing row must fail")
                .contains("coverage mismatch")
        );
    }

    #[test]
    fn bootstrap_interval_is_deterministic() {
        let a = score_csv(METADATA.as_bytes(), CANDIDATE.as_bytes(), "x", 100, 123).expect("score");
        let b = score_csv(METADATA.as_bytes(), CANDIDATE.as_bytes(), "x", 100, 123).expect("score");
        assert_eq!(
            a.system.srcc_interval_95.unwrap().lower_95,
            b.system.srcc_interval_95.unwrap().lower_95
        );
    }
}
