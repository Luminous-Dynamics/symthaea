//! Paper-faithful LaRA metric and SLaRA aggregation kernel.
//!
//! Model-loader agnostic: a model-specific adapter supplies mean-pooled
//! hidden representations for original, blanked, semantic-neighbour and
//! blank-preserving paraphrase inputs. No prompts are generated here.
//!
//! RSM and RSI use sample standard deviations (ddof=1) across semantic
//! neighbours. DC uses the paper's smoothed cosine denominator. SLaRA applies
//! signed-log compression, clean-reference median / (1.4826 * MAD), flips
//! RSI only, then averages aligned metric-layer contributions.
//!
//! This module computes a representation-geometry score, not a membership
//! verdict. It cannot authenticate hidden-state provenance or custody.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const LARA_INPUT_SCHEMA: &str = "symthaea.lara_metric_input.v1";
pub const LARA_REPORT_SCHEMA: &str = "symthaea.slara_report.v1";
pub const LARA_REQUEST_SCHEMA: &str = "symthaea.slara_scoring_request.v1";
pub const LARA_PAPER_IMPLEMENTATION_ID: &str = "symthaea.lara.paper_faithful.v1";
pub const LARA_EPSILON: f64 = 1.0e-8;
pub const MIN_SEMANTIC_NEIGHBORS: usize = 2;
pub const MAX_SEMANTIC_NEIGHBORS: usize = 1_024;
pub const MIN_PARAPHRASE_VARIANTS: usize = 2;
pub const MAX_PARAPHRASE_VARIANTS: usize = 256;
pub const MAX_HIDDEN_DIMENSION: usize = 65_536;
pub const MAX_LAYERS: usize = 512;
pub const MIN_CLEAN_REFERENCE_ITEMS: usize = 2;
pub const MAD_NORMAL_CONSISTENCY_FACTOR: f64 = 1.4826;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LayerProbeInputV1 {
    pub layer_id: String,
    pub original_complete: Vec<f64>,
    pub original_incomplete: Vec<f64>,
    pub neighbor_complete: Vec<Vec<f64>>,
    pub neighbor_incomplete: Vec<Vec<f64>>,
    pub original_paraphrase_representations: Vec<Vec<f64>>,
    pub neighbor_paraphrase_representations: Vec<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaraMetricInputV1 {
    pub schema: String,
    pub sample_id: String,
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
    pub perturbation_protocol_sha256: String,
    pub representation_extractor_sha256: String,
    pub layers: Vec<LayerProbeInputV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LayerMetricsV1 {
    pub layer_id: String,
    pub rsm: f64,
    pub dc: f64,
    pub rsi: f64,
    pub semantic_neighbor_count: u64,
    pub original_paraphrase_count: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LayerContributionV1 {
    pub layer_id: String,
    pub rsm_signed_log: f64,
    pub dc_signed_log: f64,
    pub rsi_signed_log: f64,
    pub rsm_clean_center: f64,
    pub dc_clean_center: f64,
    pub rsi_clean_center: f64,
    pub rsm_clean_scale: f64,
    pub dc_clean_scale: f64,
    pub rsi_clean_scale: f64,
    pub rsm_aligned_z: f64,
    pub dc_aligned_z: f64,
    pub rsi_aligned_z: f64,
    pub layer_mean: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SlaraReportV1 {
    pub schema: String,
    pub sample_id: String,
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
    pub perturbation_protocol_sha256: String,
    pub representation_extractor_sha256: String,
    pub clean_reference_count: u64,
    pub implementation_id: String,
    pub candidate_input_sha256: String,
    pub candidate_metric_vector_sha256: String,
    pub clean_reference_metric_manifest_sha256: String,
    pub layer_count: u64,
    pub metric_layer_contribution_count: u64,
    pub slara_score: f64,
    pub layer_contributions: Vec<LayerContributionV1>,
    pub degenerate_clean_scales: Vec<String>,
    pub paper_faithful_aggregation: bool,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CleanReferenceMetricItemV1 {
    pub sample_id: String,
    pub model_checkpoint_sha256: String,
    pub tokenizer_sha256: String,
    pub perturbation_protocol_sha256: String,
    pub representation_extractor_sha256: String,
    pub layer_metrics: Vec<LayerMetricsV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaraScoringRequestV1 {
    pub schema: String,
    pub candidate: LaraMetricInputV1,
    pub clean_reference: Vec<CleanReferenceMetricItemV1>,
}

#[derive(Debug, Clone, Copy)]
struct VectorPairStats {
    original_norm: f64,
    neighbor_mean_norm: f64,
    neighbor_sample_std: f64,
}

struct VectorPairResult {
    stats: VectorPairStats,
    original_shift: Vec<f64>,
    neighbor_shifts: Vec<Vec<f64>>,
}

fn check_finite(values: &[f64], label: &str) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} contains a non-finite value"));
    }
    Ok(())
}

fn validate_vector(vector: &[f64], label: &str) -> Result<(), String> {
    if vector.is_empty() || vector.len() > MAX_HIDDEN_DIMENSION {
        return Err(format!(
            "{label} dimension must be in 1..={MAX_HIDDEN_DIMENSION}"
        ));
    }
    check_finite(vector, label)
}

fn vector_norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn euclidean_distance(left: &[f64], right: &[f64], label: &str) -> Result<f64, String> {
    if left.len() != right.len() {
        return Err(format!("{label} vectors have different dimensions"));
    }
    validate_vector(left, label)?;
    validate_vector(right, label)?;
    let distance = left
        .iter()
        .zip(right)
        .map(|(a, b)| {
            let difference = a - b;
            difference * difference
        })
        .sum::<f64>()
        .sqrt();
    if distance.is_finite() {
        Ok(distance)
    } else {
        Err(format!("{label} distance is non-finite"))
    }
}

fn subtraction(left: &[f64], right: &[f64], label: &str) -> Result<Vec<f64>, String> {
    if left.len() != right.len() {
        return Err(format!("{label} vectors have different dimensions"));
    }
    validate_vector(left, label)?;
    validate_vector(right, label)?;
    let result: Vec<f64> = left.iter().zip(right).map(|(a, b)| a - b).collect();
    check_finite(&result, label)?;
    Ok(result)
}

fn mean_vector(vectors: &[Vec<f64>], label: &str) -> Result<Vec<f64>, String> {
    let first = vectors
        .first()
        .ok_or_else(|| format!("{label} cannot be empty"))?;
    validate_vector(first, label)?;
    let dimension = first.len();
    let mut mean = vec![0.0; dimension];
    for vector in vectors {
        validate_vector(vector, label)?;
        if vector.len() != dimension {
            return Err(format!("{label} vectors have inconsistent dimensions"));
        }
        for (sum, value) in mean.iter_mut().zip(vector) {
            *sum += value;
        }
    }
    let count = vectors.len() as f64;
    for value in &mut mean {
        *value /= count;
    }
    check_finite(&mean, label)?;
    Ok(mean)
}

fn sample_mean_and_std(values: &[f64], label: &str) -> Result<(f64, f64), String> {
    if values.len() < MIN_SEMANTIC_NEIGHBORS {
        return Err(format!(
            "{label} requires at least {MIN_SEMANTIC_NEIGHBORS} observations"
        ));
    }
    check_finite(values, label)?;
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let std = variance.sqrt();
    if mean.is_finite() && std.is_finite() {
        Ok((mean, std))
    } else {
        Err(format!(
            "{label} mean or sample standard deviation is non-finite"
        ))
    }
}

fn local_representation_stability(variants: &[Vec<f64>], label: &str) -> Result<f64, String> {
    if variants.len() < MIN_PARAPHRASE_VARIANTS || variants.len() > MAX_PARAPHRASE_VARIANTS {
        return Err(format!(
            "{label} requires {MIN_PARAPHRASE_VARIANTS}..={MAX_PARAPHRASE_VARIANTS} variants"
        ));
    }
    let centroid = mean_vector(variants, label)?;
    let mut distances = Vec::with_capacity(variants.len());
    for variant in variants {
        distances.push(euclidean_distance(variant, &centroid, label)?);
    }
    let stability = distances.iter().sum::<f64>() / distances.len() as f64;
    if stability.is_finite() {
        Ok(stability)
    } else {
        Err(format!("{label} stability is non-finite"))
    }
}

fn vector_pair_stats(
    original_complete: &[f64],
    original_incomplete: &[f64],
    neighbor_complete: &[Vec<f64>],
    neighbor_incomplete: &[Vec<f64>],
) -> Result<VectorPairResult, String> {
    if neighbor_complete.len() != neighbor_incomplete.len()
        || !(MIN_SEMANTIC_NEIGHBORS..=MAX_SEMANTIC_NEIGHBORS).contains(&neighbor_complete.len())
    {
        return Err(format!(
            "semantic-neighbour pairs must be matched and in {}..={}",
            MIN_SEMANTIC_NEIGHBORS, MAX_SEMANTIC_NEIGHBORS
        ));
    }
    let original_shift = subtraction(
        original_complete,
        original_incomplete,
        "original complete/incomplete",
    )?;
    let mut neighbor_shifts = Vec::with_capacity(neighbor_complete.len());
    let mut neighbor_norms = Vec::with_capacity(neighbor_complete.len());
    for (complete, incomplete) in neighbor_complete.iter().zip(neighbor_incomplete) {
        let shift = subtraction(complete, incomplete, "neighbour complete/incomplete")?;
        neighbor_norms.push(vector_norm(&shift));
        neighbor_shifts.push(shift);
    }
    check_finite(&neighbor_norms, "neighbour shift magnitudes")?;
    let (neighbor_mean_norm, neighbor_sample_std) =
        sample_mean_and_std(&neighbor_norms, "neighbour shift magnitudes")?;
    let stats = VectorPairStats {
        original_norm: vector_norm(&pair.original_shift),
        neighbor_mean_norm,
        neighbor_sample_std,
    };
    if !stats.original_norm.is_finite() {
        return Err("original shift magnitude is non-finite".to_owned());
    }
    Ok(VectorPairResult { stats, original_shift, neighbor_shifts })
}

/// Compute LaRA's paper-defined RSM, DC and RSI for one transformer layer.
pub fn compute_layer_metrics(input: &LayerProbeInputV1) -> Result<LayerMetricsV1, String> {
    if input.layer_id.trim().is_empty()
        || input.layer_id.len() > 128
        || !input
            .layer_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err("layer_id must be a nonempty opaque ASCII identifier <=128 bytes".to_owned());
    }

    let pair = vector_pair_stats(
        &input.original_complete,
        &input.original_incomplete,
        &input.neighbor_complete,
        &input.neighbor_incomplete,
    )?;
    let rsm = (pair.stats.original_norm - pair.stats.neighbor_mean_norm)
        / (pair.stats.neighbor_sample_std + LARA_EPSILON);

    let mean_neighbor_shift = mean_vector(&pair.neighbor_shifts, "mean semantic-neighbour shift")?;
    if pair.original_shift.len() != mean_neighbor_shift.len() {
        return Err("original and semantic-neighbour hidden dimensions differ".to_owned());
    }
    let dot = pair.original_shift
        .iter()
        .zip(&mean_neighbor_shift)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let dc = dot
        / ((vector_norm(&original_shift) + LARA_EPSILON)
            * (vector_norm(&mean_neighbor_shift) + LARA_EPSILON));

    if input.neighbor_paraphrase_representations.len() != pair.neighbor_shifts.len() {
        return Err(
            "each semantic neighbour must have one matched paraphrase representation group"
                .to_owned(),
        );
    }
    let hidden_dimension = input.original_complete.len();
    if input
        .original_paraphrase_representations
        .iter()
        .chain(input.neighbor_paraphrase_representations.iter().flatten())
        .any(|representation| representation.len() != hidden_dimension)
    {
        return Err("paraphrase hidden dimensions must match the model layer dimension".to_owned());
    }
    if input
        .neighbor_paraphrase_representations
        .iter()
        .any(|variants| variants.len() != input.original_paraphrase_representations.len())
    {
        return Err(
            "all semantic neighbours must use the same paraphrase variant count".to_owned(),
        );
    }
    let original_rsi = local_representation_stability(
        &input.original_paraphrase_representations,
        "original paraphrase representations",
    )?;
    let mut neighbor_rsi = Vec::with_capacity(input.neighbor_paraphrase_representations.len());
    for (index, variants) in input.neighbor_paraphrase_representations.iter().enumerate() {
        neighbor_rsi.push(local_representation_stability(
            variants,
            &format!("semantic neighbour {index} paraphrase representations"),
        )?);
    }
    let (neighbor_rsi_mean, neighbor_rsi_std) =
        sample_mean_and_std(&neighbor_rsi, "semantic-neighbour RSI")?;
    let rsi = (original_rsi - neighbor_rsi_mean) / (neighbor_rsi_std + LARA_EPSILON);

    for (name, value) in [("RSM", rsm), ("DC", dc), ("RSI", rsi)] {
        if !value.is_finite() {
            return Err(format!("{name} metric is non-finite"));
        }
    }

    Ok(LayerMetricsV1 {
        layer_id: input.layer_id.clone(),
        rsm,
        dc,
        rsi,
        semantic_neighbor_count: neighbor_shifts.len() as u64,
        original_paraphrase_count: input.original_paraphrase_representations.len() as u64,
    })
}

fn validate_metric_layers(layers: &[LayerMetricsV1], label: &str) -> Result<(), String> {
    if layers.is_empty() || layers.len() > MAX_LAYERS {
        return Err(format!("{label} layer count must be in 1..={MAX_LAYERS}"));
    }
    let mut ids = BTreeSet::new();
    for layer in layers {
        if layer.layer_id.trim().is_empty()
            || layer.layer_id.len() > 128
            || !ids.insert(layer.layer_id.as_str())
        {
            return Err(format!(
                "{label} has an empty, oversized or duplicate layer ID"
            ));
        }
        if !layer
            .layer_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
        {
            return Err(format!("{label} contains an invalid layer ID"));
        }
        if !(MIN_SEMANTIC_NEIGHBORS as u64..=MAX_SEMANTIC_NEIGHBORS as u64)
            .contains(&layer.semantic_neighbor_count)
            || !(MIN_PARAPHRASE_VARIANTS as u64..=MAX_PARAPHRASE_VARIANTS as u64)
                .contains(&layer.original_paraphrase_count)
        {
            return Err(format!(
                "{label} has invalid semantic-neighbour or paraphrase counts"
            ));
        }
        for (name, value) in [("RSM", layer.rsm), ("DC", layer.dc), ("RSI", layer.rsi)] {
            if !value.is_finite() {
                return Err(format!("{label} {name} contains a non-finite value"));
            }
            if name == "DC" && !(-1.0..=1.0).contains(&value) {
                return Err(format!("{label} DC must be in [-1, 1]"));
            }
        }
    }
    Ok(())
}

fn signed_log(value: f64) -> f64 {
    value.signum() * value.abs().ln_1p()
}

fn median(values: &[f64], label: &str) -> Result<f64, String> {
    if values.is_empty() {
        return Err(format!("{label} cannot be empty"));
    }
    check_finite(values, label)?;
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let middle = sorted.len() / 2;
    let result = if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(format!("{label} median is non-finite"))
    }
}

fn median_absolute_deviation(values: &[f64], center: f64, label: &str) -> Result<f64, String> {
    let deviations: Vec<f64> = values.iter().map(|value| (value - center).abs()).collect();
    median(&deviations, label)
}

fn metric_value(layer: &LayerMetricsV1, metric: MetricKind) -> f64 {
    match metric {
        MetricKind::Rsm => layer.rsm,
        MetricKind::Dc => layer.dc,
        MetricKind::Rsi => layer.rsi,
    }
}

#[derive(Debug, Clone, Copy)]
enum MetricKind {
    Rsm,
    Dc,
    Rsi,
}

impl MetricKind {
    fn label(self) -> &'static str {
        match self {
            Self::Rsm => "rsm",
            Self::Dc => "dc",
            Self::Rsi => "rsi",
        }
    }
}

/// Compute paper-faithful SLaRA from one sample's metrics and a disjoint clean
/// reference cohort. No learned weights, trend rules or test-set fitting occur.
fn aggregate_slara(
    sample_id: &str,
    model_checkpoint_sha256: &str,
    tokenizer_sha256: &str,
    perturbation_protocol_sha256: &str,
    representation_extractor_sha256: &str,
    sample_layers: &[LayerMetricsV1],
    clean_reference: &[Vec<LayerMetricsV1>],
) -> Result<SlaraReportV1, String> {
    if sample_id.trim().is_empty() || sample_id.len() > 256 {
        return Err("sample_id must be nonempty and at most 256 bytes".to_owned());
    }
    for (digest, label) in [
        (model_checkpoint_sha256, "model_checkpoint_sha256"),
        (tokenizer_sha256, "tokenizer_sha256"),
        (perturbation_protocol_sha256, "perturbation_protocol_sha256"),
        (
            representation_extractor_sha256,
            "representation_extractor_sha256",
        ),
    ] {
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!("{label} must be a lowercase SHA-256 digest"));
        }
    }
    validate_metric_layers(sample_layers, "sample")?;
    if clean_reference.len() < MIN_CLEAN_REFERENCE_ITEMS {
        return Err(format!(
            "clean reference requires at least {MIN_CLEAN_REFERENCE_ITEMS} items"
        ));
    }
    if clean_reference.len() > 100_000 {
        return Err("clean reference exceeds 100000 items".to_owned());
    }

    let sample_layer_ids: BTreeSet<&str> = sample_layers
        .iter()
        .map(|layer| layer.layer_id.as_str())
        .collect();
    let mut reference_values = Vec::with_capacity(clean_reference.len());
    for (index, layers) in clean_reference.iter().enumerate() {
        validate_metric_layers(layers, &format!("clean reference item {index}"))?;
        let reference_layer_ids: BTreeSet<&str> =
            layers.iter().map(|layer| layer.layer_id.as_str()).collect();
        if reference_layer_ids != sample_layer_ids {
            return Err(
                "every clean reference item must have exactly the sample's layer-ID set".to_owned(),
            );
        }
        reference_values.push(layers);
    }

    let mut contributions = Vec::with_capacity(sample_layers.len());
    let mut total = 0.0;
    let mut contribution_count = 0_u64;
    let mut degenerate = Vec::new();

    for sample_layer in sample_layers {
        let mut values = [0.0_f64; 3];
        let mut centers = [0.0_f64; 3];
        let mut scales = [0.0_f64; 3];
        let mut aligned = [0.0_f64; 3];
        for (slot, metric) in [MetricKind::Rsm, MetricKind::Dc, MetricKind::Rsi]
            .into_iter()
            .enumerate()
        {
            values[slot] = signed_log(metric_value(sample_layer, metric));
            let clean_transformed: Vec<f64> = reference_values
                .iter()
                .map(|layers| {
                    let layer = layers
                        .iter()
                        .find(|candidate| candidate.layer_id == sample_layer.layer_id)
                        .expect("layer-ID set validated above");
                    signed_log(metric_value(layer, metric))
                })
                .collect();
            let center = median(&clean_transformed, "clean reference metric")?;
            let mad = median_absolute_deviation(
                &clean_transformed,
                center,
                "clean reference absolute deviations",
            )?;
            let scale = MAD_NORMAL_CONSISTENCY_FACTOR * mad;
            centers[slot] = center;
            scales[slot] = scale;
            if scale <= f64::EPSILON {
                degenerate.push(format!("{}:{}", sample_layer.layer_id, metric.label()));
            }
            let z = (values[slot] - center) / (scale + LARA_EPSILON);
            aligned[slot] = if matches!(metric, MetricKind::Rsi) {
                -z
            } else {
                z
            };
            if !aligned[slot].is_finite() {
                return Err(format!(
                    "non-finite robust-standardized contribution for {}:{}",
                    sample_layer.layer_id,
                    metric.label()
                ));
            }
        }
        let layer_mean = (aligned[0] + aligned[1] + aligned[2]) / 3.0;
        total += layer_mean;
        contribution_count = contribution_count
            .checked_add(3)
            .ok_or_else(|| "metric-layer contribution count overflows".to_owned())?;
        contributions.push(LayerContributionV1 {
            layer_id: sample_layer.layer_id.clone(),
            rsm_signed_log: values[0],
            dc_signed_log: values[1],
            rsi_signed_log: values[2],
            rsm_clean_center: centers[0],
            dc_clean_center: centers[1],
            rsi_clean_center: centers[2],
            rsm_clean_scale: scales[0],
            dc_clean_scale: scales[1],
            rsi_clean_scale: scales[2],
            rsm_aligned_z: aligned[0],
            dc_aligned_z: aligned[1],
            rsi_aligned_z: aligned[2],
            layer_mean,
        });
    }
    let slara_score = total / sample_layers.len() as f64;
    if !slara_score.is_finite() {
        return Err("SLaRA aggregate is non-finite".to_owned());
    }

    Ok(SlaraReportV1 {
        schema: LARA_REPORT_SCHEMA.to_owned(),
        sample_id: sample_id.to_owned(),
        model_checkpoint_sha256: model_checkpoint_sha256.to_owned(),
        tokenizer_sha256: tokenizer_sha256.to_owned(),
        perturbation_protocol_sha256: perturbation_protocol_sha256.to_owned(),
        representation_extractor_sha256: representation_extractor_sha256.to_owned(),
        clean_reference_count: clean_reference.len() as u64,
        implementation_id: LARA_PAPER_IMPLEMENTATION_ID.to_owned(),
        candidate_input_sha256: String::new(),
        candidate_metric_vector_sha256: String::new(),
        clean_reference_metric_manifest_sha256: String::new(),
        layer_count: sample_layers.len() as u64,
        metric_layer_contribution_count: contribution_count,
        slara_score,
        layer_contributions: contributions,
        degenerate_clean_scales: degenerate,
        paper_faithful_aggregation: true,
        independently_authenticated: false,
        confirmatory_eligible: false,
        claim_boundary: "This score is a paper-equation representation-geometry anomaly score conditional on caller-supplied hidden states, perturbation protocol, model/tokenizer identities and clean reference. Clean-reference metric rows are caller-supplied summaries: this crate validates identity fields, layer sets, sample counts and numerical domains, but cannot recompute their hidden states or authenticate how they were extracted. It does not prove training membership, prove that exposure caused an answer, authenticate data/model custody, or establish generalization beyond the declared cohort. The clean reference must be disjoint from the candidate evaluation set and independently qualified by the caller.".to_owned(),
    })
}

fn domain_hash(domain: &[u8], payload: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-lara-domain-hash.v1\0");
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((payload.len() as u64).to_be_bytes());
    hasher.update(payload);
    let bytes = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

pub fn compute_lara_metrics(input: &LaraMetricInputV1) -> Result<Vec<LayerMetricsV1>, String> {
    if input.schema != LARA_INPUT_SCHEMA {
        return Err(format!("candidate schema must be {LARA_INPUT_SCHEMA}"));
    }
    if input.sample_id.trim().is_empty() || input.sample_id.len() > 256 {
        return Err("candidate sample_id must be nonempty and <=256 bytes".to_owned());
    }
    for (digest, label) in [
        (&input.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&input.tokenizer_sha256, "tokenizer_sha256"),
        (
            &input.perturbation_protocol_sha256,
            "perturbation_protocol_sha256",
        ),
        (
            &input.representation_extractor_sha256,
            "representation_extractor_sha256",
        ),
    ] {
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!("{label} must be a lowercase SHA-256 digest"));
        }
    }
    if input.layers.is_empty() || input.layers.len() > MAX_LAYERS {
        return Err(format!("candidate layer count must be in 1..={MAX_LAYERS}"));
    }
    let mut output = Vec::with_capacity(input.layers.len());
    let mut ids = BTreeSet::new();
    for layer in &input.layers {
        if !ids.insert(layer.layer_id.as_str()) {
            return Err("candidate contains duplicate layer IDs".to_owned());
        }
        output.push(compute_layer_metrics(layer)?);
    }
    Ok(output)
}

/// Validate model/tokenizer/protocol identity across candidate and clean
/// references, compute the candidate metrics, and produce the paper-faithful
/// SLaRA score. Clean-reference records are sorted by opaque sample ID before
/// their domain-separated manifest digest is calculated.
pub fn score_lara_request(request: &LaraScoringRequestV1) -> Result<SlaraReportV1, String> {
    if request.schema != LARA_REQUEST_SCHEMA {
        return Err(format!("request schema must be {LARA_REQUEST_SCHEMA}"));
    }
    let candidate_layers = compute_lara_metrics(&request.candidate)?;
    if request.clean_reference.len() < MIN_CLEAN_REFERENCE_ITEMS
        || request.clean_reference.len() > 100_000
    {
        return Err(format!(
            "clean-reference item count must be in {MIN_CLEAN_REFERENCE_ITEMS}..=100000"
        ));
    }

    let candidate = &request.candidate;
    let mut reference_records: Vec<&CleanReferenceMetricItemV1> =
        request.clean_reference.iter().collect();
    reference_records.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    let mut reference_ids = BTreeSet::new();
    let mut clean_metric_rows = Vec::with_capacity(reference_records.len());

    for item in &reference_records {
        if item.sample_id.trim().is_empty()
            || item.sample_id.len() > 256
            || item.sample_id == candidate.sample_id
            || !reference_ids.insert(item.sample_id.as_str())
        {
            return Err(
                "clean-reference sample IDs must be unique, nonempty and exclude the candidate"
                    .to_owned(),
            );
        }
        if item.model_checkpoint_sha256 != candidate.model_checkpoint_sha256
            || item.tokenizer_sha256 != candidate.tokenizer_sha256
            || item.perturbation_protocol_sha256 != candidate.perturbation_protocol_sha256
            || item.representation_extractor_sha256 != candidate.representation_extractor_sha256
        {
            return Err("clean reference model/checkpoint/tokenizer/perturbation/extractor identities must exactly match candidate".to_owned());
        }
        validate_metric_layers(&item.layer_metrics, "clean reference")?;
        if item.layer_metrics.len() != candidate_layers.len() {
            return Err("clean reference layer count differs from candidate".to_owned());
        }
        for candidate_layer in &candidate_layers {
            let reference_layer = item
                .layer_metrics
                .iter()
                .find(|layer| layer.layer_id == candidate_layer.layer_id)
                .ok_or_else(|| "clean reference layer IDs differ from candidate".to_owned())?;
            if reference_layer.semantic_neighbor_count != candidate_layer.semantic_neighbor_count
                || reference_layer.original_paraphrase_count
                    != candidate_layer.original_paraphrase_count
            {
                return Err(
                    "clean reference perturbation sample counts differ from candidate".to_owned(),
                );
            }
        }
        clean_metric_rows.push(item.layer_metrics.clone());
    }

    let candidate_input_bytes = serde_json::to_vec(candidate)
        .map_err(|error| format!("cannot serialize candidate metric input: {error}"))?;
    let candidate_metric_bytes = serde_json::to_vec(&candidate_layers)
        .map_err(|error| format!("cannot serialize candidate metric vector: {error}"))?;
    let clean_manifest_bytes = serde_json::to_vec(&reference_records)
        .map_err(|error| format!("cannot serialize clean-reference manifest: {error}"))?;

    let mut report = aggregate_slara(
        &candidate.sample_id,
        &candidate.model_checkpoint_sha256,
        &candidate.tokenizer_sha256,
        &candidate.perturbation_protocol_sha256,
        &candidate.representation_extractor_sha256,
        &candidate_layers,
        &clean_metric_rows,
    )?;
    report.candidate_input_sha256 = domain_hash(b"lara-candidate-input.v1", &candidate_input_bytes);
    report.candidate_metric_vector_sha256 =
        domain_hash(b"lara-candidate-metric-vector.v1", &candidate_metric_bytes);
    report.clean_reference_metric_manifest_sha256 = domain_hash(
        b"lara-clean-reference-metric-manifest.v1",
        &clean_manifest_bytes,
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(values: &[f64]) -> Vec<f64> {
        values.to_vec()
    }

    fn sample_layer() -> LayerProbeInputV1 {
        LayerProbeInputV1 {
            layer_id: "L4".to_owned(),
            original_complete: vector(&[3.0, 0.0]),
            original_incomplete: vector(&[0.0, 0.0]),
            neighbor_complete: vec![vector(&[1.0, 0.0]), vector(&[2.0, 0.0])],
            neighbor_incomplete: vec![vector(&[0.0, 0.0]), vector(&[0.0, 0.0])],
            original_paraphrase_representations: vec![vector(&[0.0, 0.0]), vector(&[2.0, 0.0])],
            neighbor_paraphrase_representations: vec![
                vec![vector(&[0.0, 0.0]), vector(&[1.0, 0.0])],
                vec![vector(&[0.0, 0.0]), vector(&[2.0, 0.0])],
            ],
        }
    }

    fn metrics(layer_id: &str, rsm: f64, dc: f64, rsi: f64) -> LayerMetricsV1 {
        LayerMetricsV1 {
            layer_id: layer_id.to_owned(),
            rsm,
            dc,
            rsi,
            semantic_neighbor_count: 2,
            original_paraphrase_count: 2,
        }
    }

    fn digest(label: &str) -> String {
        format!("{:0<64}", label)
    }

    #[test]
    fn rsm_uses_sample_standard_deviation_of_semantic_neighbours() {
        let result = compute_layer_metrics(&sample_layer()).unwrap();
        let expected_std = (0.5_f64).sqrt();
        let expected = (3.0 - 1.5) / (expected_std + LARA_EPSILON);
        assert!((result.rsm - expected).abs() < 1.0e-10);
    }

    #[test]
    fn dc_matches_smoothed_cosine_of_original_and_mean_neighbour_shifts() {
        let result = compute_layer_metrics(&sample_layer()).unwrap();
        let expected = 4.5 / ((3.0 + LARA_EPSILON) * (1.5 + LARA_EPSILON));
        assert!((result.dc - expected).abs() < 1.0e-10);
    }

    #[test]
    fn rsi_uses_mean_centroid_distance_and_neighbour_sample_standard_deviation() {
        let result = compute_layer_metrics(&sample_layer()).unwrap();
        let original_rsi = 1.0;
        let first_neighbour_rsi = 0.5;
        let second_neighbour_rsi = 1.0;
        let neighbour_mean = (first_neighbour_rsi + second_neighbour_rsi) / 2.0;
        let sample_std = (0.125_f64).sqrt();
        let expected = (original_rsi - neighbour_mean) / (sample_std + LARA_EPSILON);
        assert!((result.rsi - expected).abs() < 1.0e-10);
    }

    #[test]
    fn zero_shift_has_finite_metrics() {
        let mut input = sample_layer();
        input.original_complete = vector(&[0.0, 0.0]);
        input.original_incomplete = vector(&[0.0, 0.0]);
        let result = compute_layer_metrics(&input).unwrap();
        assert!(result.rsm.is_finite());
        assert!(result.dc.is_finite());
    }

    #[test]
    fn mismatched_vector_dimensions_fail_closed() {
        let mut input = sample_layer();
        input.original_incomplete = vector(&[0.0]);
        assert!(compute_layer_metrics(&input).is_err());
    }

    #[test]
    fn one_semantic_neighbour_is_rejected_for_sample_variance() {
        let mut input = sample_layer();
        input.neighbor_complete.pop();
        input.neighbor_incomplete.pop();
        input.neighbor_paraphrase_representations.pop();
        assert!(compute_layer_metrics(&input).is_err());
    }

    #[test]
    fn one_paraphrase_variant_is_rejected() {
        let mut input = sample_layer();
        input.original_paraphrase_representations.pop();
        assert!(compute_layer_metrics(&input).is_err());
    }

    #[test]
    fn non_finite_hidden_values_are_rejected() {
        let mut input = sample_layer();
        input.original_complete[0] = f64::NAN;
        assert!(compute_layer_metrics(&input).is_err());
    }

    #[test]
    fn robust_clean_standardization_flips_rsi_only() {
        let sample = vec![metrics("L0", 4.0, 0.8, -4.0)];
        let clean = vec![
            vec![metrics("L0", -1.0, -0.2, 1.0)],
            vec![metrics("L0", 0.0, 0.0, 0.0)],
            vec![metrics("L0", 1.0, 0.2, -1.0)],
        ];
        let report = aggregate_slara(
            "sample-1",
            &digest("1"),
            &digest("2"),
            &digest("3"),
            &digest("4"),
            &sample,
            &clean,
        )
        .unwrap();
        let layer = &report.layer_contributions[0];
        assert!(layer.rsm_aligned_z > 0.0);
        assert!(layer.dc_aligned_z > 0.0);
        assert!(layer.rsi_aligned_z > 0.0);
        assert!(report.slara_score > 0.0);
        assert!(report.paper_faithful_aggregation);
    }

    #[test]
    fn zero_mad_is_reported_as_degenerate_not_silently_hidden() {
        let sample = vec![metrics("L0", 3.0, 0.5, -2.0)];
        let clean = vec![
            vec![metrics("L0", 1.0, 0.1, 1.0)],
            vec![metrics("L0", 1.0, 0.1, 1.0)],
            vec![metrics("L0", 1.0, 0.1, 1.0)],
        ];
        let report = aggregate_slara(
            "sample-1",
            &digest("1"),
            &digest("2"),
            &digest("3"),
            &digest("4"),
            &sample,
            &clean,
        )
        .unwrap();
        assert_eq!(report.degenerate_clean_scales.len(), 3);
        assert!(report.slara_score.is_finite());
    }

    #[test]
    fn layer_order_does_not_change_average_score() {
        let sample_a = vec![metrics("L0", 4.0, 0.8, -4.0), metrics("L1", 2.0, 0.4, -2.0)];
        let sample_b = vec![sample_a[1].clone(), sample_a[0].clone()];
        let clean = vec![
            vec![
                metrics("L0", -1.0, -0.2, 1.0),
                metrics("L1", -2.0, -0.4, 2.0),
            ],
            vec![metrics("L0", 0.0, 0.0, 0.0), metrics("L1", 0.0, 0.0, 0.0)],
            vec![metrics("L0", 1.0, 0.2, -1.0), metrics("L1", 2.0, 0.4, -2.0)],
        ];
        let a = aggregate_slara(
            "sample-1",
            &digest("1"),
            &digest("2"),
            &digest("3"),
            &digest("4"),
            &sample_a,
            &clean,
        )
        .unwrap();
        let b = aggregate_slara(
            "sample-1",
            &digest("1"),
            &digest("2"),
            &digest("3"),
            &digest("4"),
            &sample_b,
            &clean,
        )
        .unwrap();
        assert!((a.slara_score - b.slara_score).abs() < 1.0e-12);
    }

    fn scoring_request() -> LaraScoringRequestV1 {
        let digest_value = digest("a");
        let candidate = LaraMetricInputV1 {
            schema: LARA_INPUT_SCHEMA.to_owned(),
            sample_id: "candidate-001".to_owned(),
            model_checkpoint_sha256: digest_value.clone(),
            tokenizer_sha256: digest("b"),
            perturbation_protocol_sha256: digest("c"),
            representation_extractor_sha256: digest("d"),
            layers: vec![sample_layer()],
        };
        LaraScoringRequestV1 {
            schema: LARA_REQUEST_SCHEMA.to_owned(),
            candidate,
            clean_reference: vec![
                CleanReferenceMetricItemV1 {
                    sample_id: "clean-001".to_owned(),
                    model_checkpoint_sha256: digest_value.clone(),
                    tokenizer_sha256: digest("b"),
                    perturbation_protocol_sha256: digest("c"),
                    representation_extractor_sha256: digest("d"),
                    layer_metrics: vec![metrics("L4", -1.0, -0.1, 1.0)],
                },
                CleanReferenceMetricItemV1 {
                    sample_id: "clean-002".to_owned(),
                    model_checkpoint_sha256: digest_value,
                    tokenizer_sha256: digest("b"),
                    perturbation_protocol_sha256: digest("c"),
                    representation_extractor_sha256: digest("d"),
                    layer_metrics: vec![metrics("L4", 1.0, 0.1, -1.0)],
                },
            ],
        }
    }

    #[test]
    fn complete_request_produces_paper_faithful_unqualified_report() {
        let report = score_lara_request(&scoring_request()).unwrap();
        assert!(report.paper_faithful_aggregation);
        assert!(!report.independently_authenticated);
        assert!(!report.confirmatory_eligible);
        assert_eq!(report.clean_reference_count, 2);
        assert_eq!(report.metric_layer_contribution_count, 3);
        assert_eq!(report.candidate_input_sha256.len(), 64);
        assert_eq!(report.candidate_metric_vector_sha256.len(), 64);
        assert_eq!(report.implementation_id, LARA_PAPER_IMPLEMENTATION_ID);
        assert_eq!(report.clean_reference_metric_manifest_sha256.len(), 64);
    }

    #[test]
    fn candidate_cannot_appear_in_its_clean_reference() {
        let mut request = scoring_request();
        request.clean_reference[0].sample_id = request.candidate.sample_id.clone();
        assert!(score_lara_request(&request).is_err());
    }

    #[test]
    fn clean_reference_must_use_exact_same_checkpoint_and_protocol() {
        let mut request = scoring_request();
        request.clean_reference[0].model_checkpoint_sha256 = digest("z");
        assert!(score_lara_request(&request).is_err());
    }

    #[test]
    fn hand_computable_json_fixture_round_trips_and_scores() {
        let request: LaraScoringRequestV1 = serde_json::from_str(include_str!(
            "../fixtures/lara-scoring-request.hand.v1.json"
        ))
        .unwrap();
        let report = score_lara_request(&request).unwrap();
        assert_eq!(report.sample_id, "candidate-fixture-001");
        assert_eq!(report.clean_reference_count, 2);
        assert!(report.slara_score.is_finite());
        assert!(!report.confirmatory_eligible);
    }

    #[test]
    fn duplicate_layer_ids_are_rejected() {
        let duplicate = vec![metrics("L0", 1.0, 0.1, 1.0), metrics("L0", 2.0, 0.2, 2.0)];
        assert!(validate_metric_layers(&duplicate, "test").is_err());
    }

    #[test]
    fn clean_reference_with_missing_layer_is_rejected() {
        let sample = vec![metrics("L0", 1.0, 0.1, 1.0), metrics("L1", 2.0, 0.2, 2.0)];
        let clean = vec![
            vec![metrics("L0", 0.0, 0.0, 0.0)],
            vec![metrics("L0", 1.0, 0.1, 1.0)],
        ];
        assert!(
            aggregate_slara(
                "sample-1",
                &digest("1"),
                &digest("2"),
                &digest("3"),
                &digest("4"),
                &sample,
                &clean
            )
            .is_err()
        );
    }
}
