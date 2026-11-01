//! Symthaea-versus-TinyML benchmark foundations.
//! Separates static inference, temporal adaptation, and active sensing tracks.
//! Includes a source-layout HDC-LTC memory payload estimator and deterministic
//! non-stationary event-stream fixtures. This crate does not report candidate wins.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const BENCHMARK_SCHEMA: &str = "symthaea.edge-intelligence-benchmark";
const BENCHMARK_VERSION: &str = "0.1.0";
const STREAM_SCHEMA: &str = "symthaea.temporal-drift-stream";
const STREAM_VERSION: &str = "0.1.0";
const BENCHMARK_DOMAIN: &[u8] = b"symthaea.edge-intelligence-benchmark.v0.1";
const STREAM_DOMAIN: &[u8] = b"symthaea.temporal-drift-stream.v0.1";
const VECTOR_FIELDS_PER_NEURON: usize = 8;
const NETWORK_VECTORS_PER_LAYER: usize = 2;
const MAX_STREAM_EVENTS: usize = 1_000_000;
const MAX_CANDIDATES: usize = 64;
const PPM: u64 = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeBenchmarkError {
    pub field: String,
    pub message: String,
}
impl EdgeBenchmarkError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}
impl fmt::Display for EdgeBenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for EdgeBenchmarkError {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(bytes);
    format!("{:x}", h.finalize())
}

// ============================================================================
// HDC-LTC memory preflight
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HdcLtcMemoryEstimate {
    pub schema: String,
    pub version: String,
    pub dimension: usize,
    pub layer_sizes: Vec<usize>,
    pub neuron_count: usize,
    pub layer_count: usize,
    pub vector_fields_per_neuron: usize,
    pub network_vectors_per_layer: usize,
    pub vector_payload_count: usize,
    /// Network-owned persistent hypervector payload; excludes Vec and allocator overhead.
    pub persistent_vector_payload_bytes: u64,
    /// One caller-owned input hypervector, reported separately.
    pub one_input_vector_bytes: u64,
    /// Persistent network payload plus one input vector, not a full linker-map estimate.
    pub minimum_working_set_payload_bytes: u64,
}

pub fn estimate_hdc_ltc_memory(
    dimension: usize,
    layer_sizes: &[usize],
) -> Result<HdcLtcMemoryEstimate, EdgeBenchmarkError> {
    if dimension == 0 {
        return Err(EdgeBenchmarkError::new("dimension", "must be positive"));
    }
    if layer_sizes.is_empty() || layer_sizes.iter().any(|n| *n == 0) {
        return Err(EdgeBenchmarkError::new("layer_sizes", "must contain non-empty layers"));
    }
    let neuron_count = layer_sizes.iter().try_fold(0usize, |sum, n| {
        sum.checked_add(*n).ok_or_else(|| EdgeBenchmarkError::new("layer_sizes", "neuron count overflow"))
    })?;
    let layer_count = layer_sizes.len();
    let vector_payload_count = neuron_count
        .checked_mul(VECTOR_FIELDS_PER_NEURON)
        .and_then(|n| n.checked_add(layer_count.checked_mul(NETWORK_VECTORS_PER_LAYER)?))
        .ok_or_else(|| EdgeBenchmarkError::new("vector_payload_count", "vector count overflow"))?;
    let vector_bytes = u64::try_from(dimension).ok()
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| EdgeBenchmarkError::new("dimension", "vector byte count overflow"))?;
    let persistent_vector_payload_bytes = u64::try_from(vector_payload_count).ok()
        .and_then(|n| n.checked_mul(vector_bytes))
        .ok_or_else(|| EdgeBenchmarkError::new("persistent_vector_payload_bytes", "byte count overflow"))?;
    let one_input_vector_bytes = vector_bytes;
    let minimum_working_set_payload_bytes = persistent_vector_payload_bytes
        .checked_add(one_input_vector_bytes)
        .ok_or_else(|| EdgeBenchmarkError::new("minimum_working_set_payload_bytes", "byte count overflow"))?;

    Ok(HdcLtcMemoryEstimate {
        schema: "symthaea.hdc-ltc-memory-estimate".to_owned(),
        version: "0.1.0".to_owned(),
        dimension,
        layer_sizes: layer_sizes.to_vec(),
        neuron_count,
        layer_count,
        vector_fields_per_neuron: VECTOR_FIELDS_PER_NEURON,
        network_vectors_per_layer: NETWORK_VECTORS_PER_LAYER,
        vector_payload_count,
        persistent_vector_payload_bytes,
        one_input_vector_bytes,
        minimum_working_set_payload_bytes,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HdcLtcDeploymentAssessment {
    pub estimate: HdcLtcMemoryEstimate,
    pub sram_budget_bytes: u64,
    pub protected_reserve_bytes: u64,
    pub available_sram_bytes: u64,
    pub fits_vector_payload_lower_bound: bool,
    pub excluded_memory_categories: Vec<String>,
}

pub fn assess_hdc_ltc_sram_budget(
    estimate: HdcLtcMemoryEstimate,
    sram_budget_bytes: u64,
    protected_reserve_bytes: u64,
) -> Result<HdcLtcDeploymentAssessment, EdgeBenchmarkError> {
    let available = sram_budget_bytes.checked_sub(protected_reserve_bytes)
        .ok_or_else(|| EdgeBenchmarkError::new("protected_reserve_bytes", "reserve exceeds SRAM budget"))?;
    Ok(HdcLtcDeploymentAssessment {
        fits_vector_payload_lower_bound: estimate.minimum_working_set_payload_bytes <= available,
        estimate,
        sram_budget_bytes,
        protected_reserve_bytes,
        available_sram_bytes: available,
        excluded_memory_categories: vec![
            "Vec headers and allocator metadata".to_owned(),
            "temporary/intermediate hypervectors".to_owned(),
            "sensor staging and output buffers beyond one input vector".to_owned(),
            "firmware, runtime, stack, DMA and interrupt state".to_owned(),
            "alignment, linker placement and memory fragmentation".to_owned(),
        ],
    })
}

// ============================================================================
// Deterministic temporal-drift stream
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TemporalDriftSchedule {
    Stationary,
    Sudden { at_event: usize, to_regime: u8 },
    Gradual { start_event: usize, end_event: usize, to_regime: u8 },
    Recurring { drift_at_event: usize, return_at_event: usize, to_regime: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalStreamConfig {
    pub event_count: usize,
    pub seed: u64,
    pub base_interval_us: u64,
    pub timestamp_jitter_ppm: u32,
    pub feature_dropout_ppm: u32,
    pub out_of_distribution_rate_ppm: u32,
    pub label_delay_us: u64,
    pub drift: TemporalDriftSchedule,
}
impl TemporalStreamConfig {
    pub fn validate(&self) -> Result<(), EdgeBenchmarkError> {
        if self.event_count == 0 || self.event_count > MAX_STREAM_EVENTS {
            return Err(EdgeBenchmarkError::new("event_count", format!("must be in 1..={MAX_STREAM_EVENTS}")));
        }
        if self.base_interval_us == 0 {
            return Err(EdgeBenchmarkError::new("base_interval_us", "must be positive"));
        }
        if self.timestamp_jitter_ppm > 500_000 {
            return Err(EdgeBenchmarkError::new("timestamp_jitter_ppm", "must be <= 500,000 ppm"));
        }
        if self.feature_dropout_ppm > PPM as u32 || self.out_of_distribution_rate_ppm > PPM as u32 {
            return Err(EdgeBenchmarkError::new("rates", "dropout and OOD rates must be in [0, 1,000,000] ppm"));
        }
        let valid_regime = |r: u8| (1..=2).contains(&r);
        match self.drift {
            TemporalDriftSchedule::Stationary => {}
            TemporalDriftSchedule::Sudden { at_event, to_regime } => {
                if at_event == 0 || at_event >= self.event_count || !valid_regime(to_regime) {
                    return Err(EdgeBenchmarkError::new("drift", "sudden drift needs an interior event and target regime 1 or 2"));
                }
            }
            TemporalDriftSchedule::Gradual { start_event, end_event, to_regime } => {
                if start_event == 0 || start_event >= end_event || end_event >= self.event_count || !valid_regime(to_regime) {
                    return Err(EdgeBenchmarkError::new("drift", "gradual drift needs 0 < start < end < event_count and target regime 1 or 2"));
                }
            }
            TemporalDriftSchedule::Recurring { drift_at_event, return_at_event, to_regime } => {
                if drift_at_event == 0 || drift_at_event >= return_at_event
                    || return_at_event >= self.event_count || !valid_regime(to_regime)
                {
                    return Err(EdgeBenchmarkError::new("drift", "recurring drift needs 0 < drift < return < event_count and target regime 1 or 2"));
                }
            }
        }
        Ok(())
    }
}

/// Observation payload exposed to the candidate model. It contains no ground-truth label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalObservation {
    pub event_index: usize,
    pub timestamp_us: u64,
    pub features: [i16; 4],
    /// Bit i set means feature i is missing.
    pub missing_feature_mask: u8,
}

/// Evaluator-only ground truth, released after the configured delay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DelayedGroundTruth {
    pub event_index: usize,
    pub label_available_at_us: u64,
    pub label: u8,
    pub regime_id: u8,
    pub transition_fraction_ppm: u32,
    pub is_out_of_distribution: bool,
}

/// Evaluator-owned dataset container. It intentionally does not implement Serialize,
/// Deserialize or Debug so the combined delayed-label stream cannot be exposed accidentally
/// through a model-facing serialization or debug dump. Candidate code receives only the
/// public observation records; delayed labels remain private to this module.
#[derive(Clone, PartialEq, Eq)]
pub struct TemporalStreamDataset {
    pub schema: String,
    pub version: String,
    config: TemporalStreamConfig,
    pub observations: Vec<TemporalObservation>,
    delayed_labels: Vec<DelayedGroundTruth>,
    pub dataset_sha256: String,
}

#[derive(Debug, Clone)]
struct XorShift64 { state: u64 }
impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed } }
    }
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 { self.next() % n }
}

fn drift_fraction(config: &TemporalStreamConfig, index: usize) -> (u8, u32) {
    match config.drift {
        TemporalDriftSchedule::Stationary => (0, 0),
        TemporalDriftSchedule::Sudden { at_event, to_regime } => {
            if index < at_event { (0, 0) } else { (to_regime, PPM as u32) }
        }
        TemporalDriftSchedule::Gradual { start_event, end_event, to_regime } => {
            if index < start_event { (0, 0) }
            else if index >= end_event { (to_regime, PPM as u32) }
            else {
                let fraction = ((index - start_event) as u64 * PPM
                    / (end_event - start_event) as u64) as u32;
                (to_regime, fraction)
            }
        }
        TemporalDriftSchedule::Recurring { drift_at_event, return_at_event, to_regime } => {
            if index >= drift_at_event && index < return_at_event { (to_regime, PPM as u32) }
            else { (0, 0) }
        }
    }
}

fn regime_score(regime: u8, x: &[i16; 4]) -> i64 {
    match regime {
        0 => x[0] as i64 + x[1] as i64,
        1 => x[2] as i64 + x[3] as i64,
        2 => if (x[0] >= 0) == (x[1] >= 0) { 1024 } else { -1024 },
        _ => 0,
    }
}

fn generate_stream(config: &TemporalStreamConfig) -> Result<TemporalStreamDataset, EdgeBenchmarkError> {
    config.validate()?;
    let mut rng = XorShift64::new(config.seed);
    let mut observations = Vec::with_capacity(config.event_count);
    let mut delayed_labels = Vec::with_capacity(config.event_count);
    let mut timestamp_us = 0u64;

    for index in 0..config.event_count {
        let is_ood = rng.below(PPM) < config.out_of_distribution_rate_ppm as u64;
        let scale = if is_ood { 4i32 } else { 1i32 };
        let latent_features = std::array::from_fn(|_| {
            ((rng.below(2049) as i32 - 1024) * scale) as i16
        });
        let mut missing_feature_mask = 0u8;
        for feature in 0..4 {
            if rng.below(PPM) < config.feature_dropout_ppm as u64 {
                missing_feature_mask |= 1 << feature;
            }
        }
        // Ground truth uses the latent complete signal. Model-facing observations zero-fill
        // dropped features and expose only the missingness mask, never the hidden values.
        let observed_features = std::array::from_fn(|feature| {
            if missing_feature_mask & (1 << feature) != 0 { 0 } else { latent_features[feature] }
        });
        let (target_regime, fraction) = drift_fraction(config, index);
        let score_a = regime_score(0, &latent_features);
        let score_b = regime_score(target_regime, &latent_features);
        let score = score_a * (PPM as i64 - fraction as i64) + score_b * fraction as i64;
        let label = u8::from(score >= 0);
        let regime_id = if fraction >= 500_000 { target_regime } else { 0 };

        if index > 0 {
            let centered = rng.below(2 * PPM + 1) as i128 - PPM as i128;
            let jitter = config.base_interval_us as i128
                * config.timestamp_jitter_ppm as i128
                * centered / (PPM as i128 * PPM as i128);
            let interval = (config.base_interval_us as i128 + jitter).max(1) as u64;
            timestamp_us = timestamp_us.checked_add(interval)
                .ok_or_else(|| EdgeBenchmarkError::new("timestamp_us", "overflow"))?;
        }
        let label_available_at_us = timestamp_us.checked_add(config.label_delay_us)
            .ok_or_else(|| EdgeBenchmarkError::new("label_available_at_us", "overflow"))?;
        observations.push(TemporalObservation {
            event_index: index,
            timestamp_us,
            features: observed_features,
            missing_feature_mask,
        });
        delayed_labels.push(DelayedGroundTruth {
            event_index: index,
            label_available_at_us,
            label,
            regime_id,
            transition_fraction_ppm: fraction,
            is_out_of_distribution: is_ood,
        });
    }

    let mut dataset = TemporalStreamDataset {
        schema: STREAM_SCHEMA.to_owned(),
        version: STREAM_VERSION.to_owned(),
        config: config.clone(),
        observations,
        delayed_labels,
        dataset_sha256: String::new(),
    };
    dataset.dataset_sha256 = compute_temporal_stream_sha256(&dataset)?;
    Ok(dataset)
}

/// Generate a deterministic non-stationary event stream from a frozen seed/config.
/// This is a controlled fixture generator, not a substitute for real sensor datasets.
pub fn generate_temporal_drift_stream(
    config: &TemporalStreamConfig,
) -> Result<TemporalStreamDataset, EdgeBenchmarkError> {
    generate_stream(config)
}
#[derive(Serialize)]
struct TemporalStreamCanonical<'a> {
    schema: &'a str,
    version: &'a str,
    config: &'a TemporalStreamConfig,
    observations: &'a [TemporalObservation],
    delayed_labels: &'a [DelayedGroundTruth],
}

pub fn compute_temporal_stream_sha256(
    dataset: &TemporalStreamDataset,
) -> Result<String, EdgeBenchmarkError> {
    let canonical = TemporalStreamCanonical {
        schema: &dataset.schema,
        version: &dataset.version,
        config: &dataset.config,
        observations: &dataset.observations,
        delayed_labels: &dataset.delayed_labels,
    };
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|e| EdgeBenchmarkError::new("dataset", e.to_string()))?;
    Ok(digest(STREAM_DOMAIN, &bytes))
}
pub fn validate_temporal_stream_dataset(
    dataset: &TemporalStreamDataset,
) -> Result<(), EdgeBenchmarkError> {
    if dataset.schema != STREAM_SCHEMA || dataset.version != STREAM_VERSION {
        return Err(EdgeBenchmarkError::new("schema_version", "unsupported stream schema/version"));
    }
    if generate_stream(&dataset.config)? != *dataset {
        return Err(EdgeBenchmarkError::new("dataset.reproduction", "stream differs from deterministic source replay"));
    }
    Ok(())
}

// ============================================================================
// Causal prequential execution trace
// ============================================================================

const PREQUENTIAL_TRACE_SCHEMA: &str = "symthaea.prequential-execution-trace";
const PREQUENTIAL_RECEIPT_SCHEMA: &str = "symthaea.prequential-audit-receipt";
const PREQUENTIAL_DOMAIN: &[u8] = b"symthaea.prequential-execution-trace.v0.1";
const PREQUENTIAL_RECEIPT_DOMAIN: &[u8] = b"symthaea.prequential-audit-receipt.v0.1";

/// Actions are serialized in actual execution order. Labels are not part of observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PrequentialAction {
    Prediction {
        event_index: usize,
        timestamp_us: u64,
        predicted_class: Option<u8>,
        positive_probability_q15: u16,
        /// Higher means more likely to be an out-of-distribution event.
        novelty_score_q15: u16,
        model_state_sha256: String,
    },
    LabelReleased {
        event_index: usize,
        timestamp_us: u64,
        label: u8,
    },
    ModelUpdate {
        event_index: usize,
        timestamp_us: u64,
        state_before_sha256: String,
        state_after_sha256: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrequentialExecutionTrace {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub candidate_id: String,
    pub candidate_artifact_sha256: String,
    pub dataset_sha256: String,
    pub initial_model_state_sha256: String,
    /// Exact phase-energy and metrology receipt commitments for this candidate/run.
    pub energy_receipt_sha256: String,
    pub metrology_receipt_sha256: String,
    pub actions: Vec<PrequentialAction>,
    pub execution_trace_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalRegimeMetrics {
    pub regime_id: u8,
    pub event_count: usize,
    pub correct_count: usize,
    pub accuracy: f64,
    pub false_positive_count: usize,
    pub false_negative_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrequentialMetrics {
    pub event_count: usize,
    pub correct_count: usize,
    pub true_positive_count: usize,
    pub true_negative_count: usize,
    pub false_positive_count: usize,
    pub false_negative_count: usize,
    pub abstention_count: usize,
    pub accuracy: f64,
    pub balanced_accuracy: Option<f64>,
    pub coverage: f64,
    pub selective_accuracy: Option<f64>,
    pub brier_score: f64,
    pub open_set_auprc: Option<f64>,
    pub per_regime_metrics: Vec<TemporalRegimeMetrics>,
    pub worst_regime_accuracy: Option<f64>,
    /// Mean number of events from a detected regime transition to recovered rolling accuracy.
    pub mean_drift_recovery_samples: Option<f64>,
    /// Maximum percentage-point accuracy drop when a previously seen regime recurs.
    pub maximum_regime_retention_loss_pp: Option<f64>,
    pub false_alarms_per_hour: f64,
    pub missed_events_per_hour: f64,
    pub mean_prediction_latency_us: f64,
    pub label_release_count: usize,
    pub model_update_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrequentialAuditReceipt {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub candidate_id: String,
    pub candidate_artifact_sha256: String,
    pub dataset_sha256: String,
    pub execution_trace_sha256: String,
    pub energy_receipt_sha256: String,
    pub metrology_receipt_sha256: String,
    pub metrics: PrequentialMetrics,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, Copy)]
struct PredictionObservation {
    timestamp_us: u64,
    predicted_class: Option<u8>,
    positive_probability_q15: u16,
    novelty_score_q15: u16,
}

pub fn compute_prequential_execution_trace_sha256(
    trace: &PrequentialExecutionTrace,
) -> Result<String, EdgeBenchmarkError> {
    let mut canonical = trace.clone();
    canonical.execution_trace_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| EdgeBenchmarkError::new("prequential_trace", error.to_string()))?;
    Ok(digest(PREQUENTIAL_DOMAIN, &bytes))
}

fn compute_prequential_receipt_sha256(
    receipt: &PrequentialAuditReceipt,
) -> Result<String, EdgeBenchmarkError> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| EdgeBenchmarkError::new("prequential_receipt", error.to_string()))?;
    Ok(digest(PREQUENTIAL_RECEIPT_DOMAIN, &bytes))
}

/// Validate the event/action chronology and recompute metrics from delayed labels.
///
/// Invariants:
/// - one prediction and one delayed label release per observation;
/// - prediction timestamp is at/after the observation and strictly before label availability;
/// - label release is at/after the frozen availability timestamp and carries the right label;
/// - a model update can occur only after that event's label is released;
/// - predictions bind the current model-state digest, and every update advances the state chain;
/// - action timestamps are globally non-decreasing in execution order.
pub fn audit_prequential_execution(
    dataset: &TemporalStreamDataset,
    trace: &PrequentialExecutionTrace,
) -> Result<PrequentialAuditReceipt, EdgeBenchmarkError> {
    validate_temporal_stream_dataset(dataset)?;
    if trace.schema != PREQUENTIAL_TRACE_SCHEMA || trace.version != BENCHMARK_VERSION {
        return Err(EdgeBenchmarkError::new("prequential_trace.schema_version", "unsupported trace schema/version"));
    }
    if trace.run_id.trim().is_empty() || trace.candidate_id.trim().is_empty() {
        return Err(EdgeBenchmarkError::new("prequential_trace.identity", "run and candidate IDs must be non-empty"));
    }
    for (field, value) in [
        ("candidate_artifact_sha256", trace.candidate_artifact_sha256.as_str()),
        ("dataset_sha256", trace.dataset_sha256.as_str()),
        ("initial_model_state_sha256", trace.initial_model_state_sha256.as_str()),
        ("energy_receipt_sha256", trace.energy_receipt_sha256.as_str()),
        ("metrology_receipt_sha256", trace.metrology_receipt_sha256.as_str()),
        ("execution_trace_sha256", trace.execution_trace_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            return Err(EdgeBenchmarkError::new(
                format!("prequential_trace.{field}"),
                "must be lowercase SHA-256 hex",
            ));
        }
    }
    if trace.dataset_sha256 != dataset.dataset_sha256 {
        return Err(EdgeBenchmarkError::new("prequential_trace.dataset_sha256", "trace is bound to a different dataset"));
    }
    if trace.actions.is_empty() {
        return Err(EdgeBenchmarkError::new("prequential_trace.actions", "execution action list must not be empty"));
    }
    if compute_prequential_execution_trace_sha256(trace)? != trace.execution_trace_sha256 {
        return Err(EdgeBenchmarkError::new("prequential_trace.execution_trace_sha256", "canonical trace digest mismatch"));
    }
    if dataset.observations.len() != dataset.delayed_labels.len() || dataset.observations.is_empty() {
        return Err(EdgeBenchmarkError::new("dataset.inventory", "observation and delayed-label inventories must be equal and non-empty"));
    }

    let count = dataset.observations.len();
    let mut predictions = vec![None::<PredictionObservation>; count];
    let mut label_released = vec![false; count];
    let mut label_release_times = vec![None::<u64>; count];
    let mut updates_seen = vec![false; count];
    let mut current_state_sha256 = trace.initial_model_state_sha256.clone();
    let mut previous_action_time = None::<u64>;
    let mut update_count = 0usize;

    for (action_index, action) in trace.actions.iter().enumerate() {
        let action_time = match action {
            PrequentialAction::Prediction { timestamp_us, .. }
            | PrequentialAction::LabelReleased { timestamp_us, .. }
            | PrequentialAction::ModelUpdate { timestamp_us, .. } => *timestamp_us,
        };
        if previous_action_time.is_some_and(|previous| action_time < previous) {
            return Err(EdgeBenchmarkError::new(
                format!("prequential_trace.actions[{action_index}].timestamp_us"),
                "action timestamps must be non-decreasing in execution order",
            ));
        }
        previous_action_time = Some(action_time);

        match action {
            PrequentialAction::Prediction {
                event_index,
                timestamp_us,
                predicted_class,
                positive_probability_q15,
                model_state_sha256,
            } => {
                if *event_index >= count {
                    return Err(EdgeBenchmarkError::new(format!("prequential_trace.actions[{action_index}].event_index"), "event index is out of range"));
                }
                if predictions[*event_index].is_some() {
                    return Err(EdgeBenchmarkError::new(format!("prequential_trace.actions[{action_index}]"), "duplicate prediction for event"));
                }
                if predicted_class.is_some_and(|label| label > 1) {
                    return Err(EdgeBenchmarkError::new(format!("prequential_trace.actions[{action_index}].predicted_class"), "binary prediction must be 0, 1 or abstain"));
                }
                if *positive_probability_q15 > 32_768 || *novelty_score_q15 > 32_768 {
                    return Err(EdgeBenchmarkError::new(format!("prequential_trace.actions[{action_index}].probability"), "positive-class probability and novelty score must be in Q0.15 range [0,32768]"));
                }
                let observation = &dataset.observations[*event_index];
                let delayed_label = &dataset.delayed_labels[*event_index];
                if observation.event_index != *event_index || delayed_label.event_index != *event_index {
                    return Err(EdgeBenchmarkError::new("dataset.event_identity", "observation and label indices must match their array position"));
                }
                if *timestamp_us < observation.timestamp_us {
                    return Err(EdgeBenchmarkError::new(format!("prequential_trace.actions[{action_index}].timestamp_us"), "prediction precedes observation"));
                }
                if *timestamp_us >= delayed_label.label_available_at_us {
                    return Err(EdgeBenchmarkError::new(
                        format!("prequential_trace.actions[{action_index}].timestamp_us"),
                        "prediction is not strictly before ground-truth label availability",
                    ));
                }
                if *model_state_sha256 != current_state_sha256 || !is_sha256(model_state_sha256) {
                    return Err(EdgeBenchmarkError::new(
                        format!("prequential_trace.actions[{action_index}].model_state_sha256"),
                        "prediction does not bind the current model-state digest",
                    ));
                }
                predictions[*event_index] = Some(PredictionObservation {
                    timestamp_us: *timestamp_us,
                    predicted_class: *predicted_class,
                    positive_probability_q15: *positive_probability_q15,
                });
            }
            PrequentialAction::LabelReleased { event_index, timestamp_us, label } => {
                if *event_index >= count {
                    return Err(EdgeBenchmarkError::new(format!("prequential_trace.actions[{action_index}].event_index"), "event index is out of range"));
                }
                if predictions[*event_index].is_none() {
                    return Err(EdgeBenchmarkError::new(
                        format!("prequential_trace.actions[{action_index}]"),
                        "label release occurs before the event prediction",
                    ));
                }
                if label_released[*event_index] {
                    return Err(EdgeBenchmarkError::new(format!("prequential_trace.actions[{action_index}]"), "duplicate delayed-label release"));
                }
                let expected = &dataset.delayed_labels[*event_index];
                if *label > 1 || *label != expected.label || *timestamp_us < expected.label_available_at_us {
                    return Err(EdgeBenchmarkError::new(
                        format!("prequential_trace.actions[{action_index}]"),
                        "released label differs from ground truth or arrives before its availability time",
                    ));
                }
                label_released[*event_index] = true;
                label_release_times[*event_index] = Some(*timestamp_us);
            }
            PrequentialAction::ModelUpdate {
                event_index,
                timestamp_us,
                state_before_sha256,
                state_after_sha256,
            } => {
                if *event_index >= count {
                    return Err(EdgeBenchmarkError::new(format!("prequential_trace.actions[{action_index}].event_index"), "event index is out of range"));
                }
                if !label_released[*event_index] {
                    return Err(EdgeBenchmarkError::new(
                        format!("prequential_trace.actions[{action_index}]"),
                        "model update occurs before the corresponding label is released",
                    ));
                }
                if updates_seen[*event_index] {
                    return Err(EdgeBenchmarkError::new(format!("prequential_trace.actions[{action_index}]"), "more than one update recorded for one event"));
                }
                let release_time = label_release_times[*event_index].ok_or_else(|| {
                    EdgeBenchmarkError::new(format!("prequential_trace.actions[{action_index}]"), "missing label release timestamp")
                })?;
                if *timestamp_us < release_time {
                    return Err(EdgeBenchmarkError::new(format!("prequential_trace.actions[{action_index}].timestamp_us"), "update occurs before its label release"));
                }
                if !is_sha256(state_before_sha256)
                    || !is_sha256(state_after_sha256)
                    || *state_before_sha256 != current_state_sha256
                {
                    return Err(EdgeBenchmarkError::new(
                        format!("prequential_trace.actions[{action_index}].state_digest"),
                        "update must consume the current model-state digest and produce a valid digest",
                    ));
                }
                current_state_sha256 = state_after_sha256.clone();
                updates_seen[*event_index] = true;
                update_count += 1;
            }
        }
    }

    if predictions.iter().any(Option::is_none) || label_released.iter().any(|released| !released) {
        return Err(EdgeBenchmarkError::new(
            "prequential_trace.actions",
            "every observation requires exactly one prediction and one delayed-label release",
        ));
    }

    let mut true_positive = 0usize;
    let mut true_negative = 0usize;
    let mut false_positive = 0usize;
    let mut false_negative = 0usize;
    let mut correct = 0usize;
    let mut accepted = 0usize;
    let mut abstentions = 0usize;
    let mut brier_sum = 0.0f64;
    let mut prediction_latency_sum = 0u128;

    for index in 0..count {
        let prediction = predictions[index].ok_or_else(|| {
            EdgeBenchmarkError::new("prequential_trace.predictions", "missing prediction after inventory check")
        })?;
        let label = dataset.delayed_labels[index].label;
        let probability = prediction.positive_probability_q15 as f64 / 32_768.0;
        let target = label as f64;
        brier_sum += (probability - target) * (probability - target);
        prediction_latency_sum += (prediction.timestamp_us - dataset.observations[index].timestamp_us) as u128;

        match prediction.predicted_class {
            Some(predicted) => {
                accepted += 1;
                if predicted == label { correct += 1; }
                match (predicted, label) {
                    (1, 1) => true_positive += 1,
                    (0, 0) => true_negative += 1,
                    (1, 0) => false_positive += 1,
                    (0, 1) => false_negative += 1,
                    _ => {}
                }
            }
            None => {
                abstentions += 1;
                if label == 1 { false_negative += 1; }
            }
        }
    }

    let positive_total = true_positive + false_negative;
    let negative_total = true_negative + false_positive;
    let balanced_accuracy = if positive_total > 0 && negative_total > 0 {
        Some(0.5 * (
            true_positive as f64 / positive_total as f64
                + true_negative as f64 / negative_total as f64
        ))
    } else {
        None
    };
    let selective_accuracy = if accepted > 0 { Some(correct as f64 / accepted as f64) } else { None };
    let first_time = dataset.observations.first().expect("non-empty checked").timestamp_us;
    let last_time = dataset.observations.last().expect("non-empty checked").timestamp_us;
    let exposure_us = (last_time.saturating_sub(first_time))
        .saturating_add(dataset.config.base_interval_us)
        .max(1) as f64;
    let exposure_hours = exposure_us / 3_600_000_000.0;

    let metrics = PrequentialMetrics {
        event_count: count,
        correct_count: correct,
        true_positive_count: true_positive,
        true_negative_count: true_negative,
        false_positive_count: false_positive,
        false_negative_count: false_negative,
        abstention_count: abstentions,
        accuracy: correct as f64 / count as f64,
        balanced_accuracy,
        coverage: accepted as f64 / count as f64,
        selective_accuracy,
        brier_score: brier_sum / count as f64,
        false_alarms_per_hour: false_positive as f64 / exposure_hours,
        missed_events_per_hour: false_negative as f64 / exposure_hours,
        mean_prediction_latency_us: prediction_latency_sum as f64 / count as f64,
        label_release_count: count,
        model_update_count: update_count,
    };

    let mut receipt = PrequentialAuditReceipt {
        schema: PREQUENTIAL_RECEIPT_SCHEMA.to_owned(),
        version: BENCHMARK_VERSION.to_owned(),
        run_id: trace.run_id.clone(),
        candidate_id: trace.candidate_id.clone(),
        candidate_artifact_sha256: trace.candidate_artifact_sha256.clone(),
        dataset_sha256: dataset.dataset_sha256.clone(),
        execution_trace_sha256: trace.execution_trace_sha256.clone(),
        energy_receipt_sha256: trace.energy_receipt_sha256.clone(),
        metrology_receipt_sha256: trace.metrology_receipt_sha256.clone(),
        metrics,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = compute_prequential_receipt_sha256(&receipt)?;
    Ok(receipt)
}

// ============================================================================
// Frozen comparison manifest
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeBenchmarkTrack { StaticInference, TemporalAdaptation, ActiveSensing }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeCandidateFamily {
    TfliteMicroCmsisNn,
    ExecuTorchCmsisNn,
    MicroTvm,
    StaticTinyMl,
    AdaptiveTinyMl,
    ThresholdEma,
    HdcOnly,
    CfcOnly,
    HdcLtc,
    HdcLtcActiveSensing,
    ActiveSensingTinyMl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeEnergySemantics { PerInference, PerSuccessfulApplicationEvent }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeBenchmarkTarget {
    pub target_id: String,
    pub board_revision: String,
    pub cpu_core: String,
    pub accelerator_profile: String,
    pub sram_budget_bytes: u64,
    pub protected_sram_reserve_bytes: u64,
    pub flash_budget_bytes: u64,
    pub p95_deadline_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeCandidateDefinition {
    pub candidate_id: String,
    pub family: EdgeCandidateFamily,
    pub runtime: String,
    pub precision: String,
    pub artifact_sha256: String,
    pub configuration_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeBenchmarkManifest {
    pub schema: String,
    pub version: String,
    pub benchmark_id: String,
    pub track: EdgeBenchmarkTrack,
    pub target: EdgeBenchmarkTarget,
    pub dataset_sha256: String,
    pub train_split_sha256: String,
    pub validation_split_sha256: String,
    pub locked_test_split_sha256: String,
    pub quality_metric_id: String,
    pub minimum_quality_score: f64,
    pub maximum_false_alarms_per_hour: f64,
    pub maximum_missed_events_per_hour: f64,
    pub minimum_coverage: f64,
    pub energy_semantics: EdgeEnergySemantics,
    pub prequential_predict_before_learn: bool,
    pub locked_test_labels: bool,
    pub candidate_definitions: Vec<EdgeCandidateDefinition>,
    pub manifest_sha256: String,
}

pub fn compute_edge_benchmark_manifest_sha256(
    manifest: &EdgeBenchmarkManifest,
) -> Result<String, EdgeBenchmarkError> {
    let mut canonical = manifest.clone();
    canonical.manifest_sha256.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|e| EdgeBenchmarkError::new("manifest", e.to_string()))?;
    Ok(digest(BENCHMARK_DOMAIN, &bytes))
}

fn require_families(
    manifest: &EdgeBenchmarkManifest,
    required: &[EdgeCandidateFamily],
) -> Result<(), EdgeBenchmarkError> {
    for family in required {
        if !manifest.candidate_definitions.iter().any(|candidate| candidate.family == *family) {
            return Err(EdgeBenchmarkError::new(
                "candidate_definitions",
                format!("required baseline/candidate family {family:?} is missing"),
            ));
        }
    }
    Ok(())
}

pub fn validate_edge_benchmark_manifest(
    manifest: &EdgeBenchmarkManifest,
) -> Result<(), EdgeBenchmarkError> {
    if manifest.schema != BENCHMARK_SCHEMA || manifest.version != BENCHMARK_VERSION {
        return Err(EdgeBenchmarkError::new("schema_version", "unsupported benchmark schema/version"));
    }
    if manifest.benchmark_id.trim().is_empty()
        || manifest.target.target_id.trim().is_empty()
        || manifest.target.board_revision.trim().is_empty()
        || manifest.target.cpu_core.trim().is_empty()
        || manifest.target.accelerator_profile.trim().is_empty()
        || manifest.quality_metric_id.trim().is_empty()
    {
        return Err(EdgeBenchmarkError::new("identity", "benchmark/target identity and quality metric must be non-empty"));
    }
    for (field, value) in [
        ("dataset_sha256", manifest.dataset_sha256.as_str()),
        ("train_split_sha256", manifest.train_split_sha256.as_str()),
        ("validation_split_sha256", manifest.validation_split_sha256.as_str()),
        ("locked_test_split_sha256", manifest.locked_test_split_sha256.as_str()),
        ("manifest_sha256", manifest.manifest_sha256.as_str()),
    ] {
        if !is_sha256(value) {
            return Err(EdgeBenchmarkError::new(field, "must be lowercase SHA-256 hex"));
        }
    }
    if manifest.train_split_sha256 == manifest.validation_split_sha256
        || manifest.train_split_sha256 == manifest.locked_test_split_sha256
        || manifest.validation_split_sha256 == manifest.locked_test_split_sha256
    {
        return Err(EdgeBenchmarkError::new("splits", "train, validation and locked-test digests must be distinct"));
    }
    if manifest.target.sram_budget_bytes == 0
        || manifest.target.flash_budget_bytes == 0
        || manifest.target.p95_deadline_ms == 0
        || manifest.target.protected_sram_reserve_bytes >= manifest.target.sram_budget_bytes
    {
        return Err(EdgeBenchmarkError::new("target.budgets", "invalid memory budget, protected reserve or deadline"));
    }
    if !manifest.minimum_quality_score.is_finite()
        || !(0.0..=1.0).contains(&manifest.minimum_quality_score)
        || !manifest.maximum_false_alarms_per_hour.is_finite()
        || manifest.maximum_false_alarms_per_hour < 0.0
        || !manifest.maximum_missed_events_per_hour.is_finite()
        || manifest.maximum_missed_events_per_hour < 0.0
        || !manifest.minimum_coverage.is_finite()
        || !(0.0..=1.0).contains(&manifest.minimum_coverage)
    {
        return Err(EdgeBenchmarkError::new("quality_constraints", "quality and minimum coverage must be [0,1]; false/missed event limits must be finite and non-negative"));
    }
    if !manifest.prequential_predict_before_learn || !manifest.locked_test_labels {
        return Err(EdgeBenchmarkError::new("evaluation_policy", "must enforce predict-before-learn and locked test labels"));
    }
    if manifest.candidate_definitions.is_empty() || manifest.candidate_definitions.len() > MAX_CANDIDATES {
        return Err(EdgeBenchmarkError::new("candidate_definitions", format!("must contain 1..={MAX_CANDIDATES} candidates")));
    }
    let mut ids = HashSet::with_capacity(manifest.candidate_definitions.len());
    for (i, candidate) in manifest.candidate_definitions.iter().enumerate() {
        if candidate.candidate_id.trim().is_empty()
            || candidate.runtime.trim().is_empty()
            || candidate.precision.trim().is_empty()
            || !is_sha256(&candidate.artifact_sha256)
            || !is_sha256(&candidate.configuration_sha256)
            || !ids.insert(candidate.candidate_id.as_str())
        {
            return Err(EdgeBenchmarkError::new(format!("candidate_definitions[{i}]"), "invalid or duplicate candidate definition"));
        }
    }
    if manifest.candidate_definitions.windows(2).any(|pair| pair[0].candidate_id >= pair[1].candidate_id) {
        return Err(EdgeBenchmarkError::new("candidate_definitions", "candidates must be sorted by candidate_id"));
    }

    match manifest.track {
        EdgeBenchmarkTrack::StaticInference => {
            if manifest.energy_semantics != EdgeEnergySemantics::PerInference {
                return Err(EdgeBenchmarkError::new("energy_semantics", "static track requires energy per inference"));
            }
            require_families(manifest, &[EdgeCandidateFamily::TfliteMicroCmsisNn, EdgeCandidateFamily::ExecuTorchCmsisNn])?;
        }
        EdgeBenchmarkTrack::TemporalAdaptation => {
            if manifest.energy_semantics != EdgeEnergySemantics::PerSuccessfulApplicationEvent {
                return Err(EdgeBenchmarkError::new("energy_semantics", "temporal track requires energy per successful application event"));
            }
            require_families(manifest, &[
                EdgeCandidateFamily::StaticTinyMl,
                EdgeCandidateFamily::AdaptiveTinyMl,
                EdgeCandidateFamily::ThresholdEma,
                EdgeCandidateFamily::HdcOnly,
                EdgeCandidateFamily::CfcOnly,
                EdgeCandidateFamily::HdcLtc,
            ])?;
        }
        EdgeBenchmarkTrack::ActiveSensing => {
            if manifest.energy_semantics != EdgeEnergySemantics::PerSuccessfulApplicationEvent {
                return Err(EdgeBenchmarkError::new("energy_semantics", "active-sensing track requires energy per successful application event"));
            }
            require_families(manifest, &[
                EdgeCandidateFamily::AdaptiveTinyMl,
                EdgeCandidateFamily::ThresholdEma,
                EdgeCandidateFamily::HdcLtc,
                EdgeCandidateFamily::HdcLtcActiveSensing,
                EdgeCandidateFamily::ActiveSensingTinyMl,
            ])?;
        }
    }
    if compute_edge_benchmark_manifest_sha256(manifest)? != manifest.manifest_sha256 {
        return Err(EdgeBenchmarkError::new("manifest_sha256", "does not match canonical manifest"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn static_manifest() -> EdgeBenchmarkManifest {
        let candidates = vec![
            EdgeCandidateDefinition {
                candidate_id: "executorch-cmsis".into(),
                family: EdgeCandidateFamily::ExecuTorchCmsisNn,
                runtime: "ExecuTorch Cortex-M + CMSIS-NN".into(),
                precision: "int8".into(),
                artifact_sha256: sha('a'),
                configuration_sha256: sha('b'),
            },
            EdgeCandidateDefinition {
                candidate_id: "tflm-cmsis".into(),
                family: EdgeCandidateFamily::TfliteMicroCmsisNn,
                runtime: "TFLite Micro + CMSIS-NN".into(),
                precision: "int8".into(),
                artifact_sha256: sha('c'),
                configuration_sha256: sha('d'),
            },
        ];
        let mut manifest = EdgeBenchmarkManifest {
            schema: BENCHMARK_SCHEMA.into(),
            version: BENCHMARK_VERSION.into(),
            benchmark_id: "symthaea-tinyml-static-v0".into(),
            track: EdgeBenchmarkTrack::StaticInference,
            target: EdgeBenchmarkTarget {
                target_id: "cortex-m55-fixture".into(),
                board_revision: "board-rev-a".into(),
                cpu_core: "cortex-m55".into(),
                accelerator_profile: "cpu-only-npu-disabled".into(),
                sram_budget_bytes: 512 * 1024,
                protected_sram_reserve_bytes: 64 * 1024,
                flash_budget_bytes: 2 * 1024 * 1024,
                p95_deadline_ms: 100,
            },
            dataset_sha256: sha('e'),
            train_split_sha256: sha('f'),
            validation_split_sha256: sha('1'),
            locked_test_split_sha256: sha('2'),
            quality_metric_id: "top1_accuracy".into(),
            minimum_quality_score: 0.85,
            maximum_false_alarms_per_hour: 10.0,
            maximum_missed_events_per_hour: 10.0,
            minimum_coverage: 0.95,
            energy_semantics: EdgeEnergySemantics::PerInference,
            prequential_predict_before_learn: true,
            locked_test_labels: true,
            candidate_definitions: candidates,
            manifest_sha256: String::new(),
        };
        manifest.manifest_sha256 = compute_edge_benchmark_manifest_sha256(&manifest).unwrap();
        manifest
    }

    fn stream_config() -> TemporalStreamConfig {
        TemporalStreamConfig {
            event_count: 2_000,
            seed: 0x51_7EA,
            base_interval_us: 10_000,
            timestamp_jitter_ppm: 150_000,
            feature_dropout_ppm: 25_000,
            out_of_distribution_rate_ppm: 30_000,
            label_delay_us: 20_000,
            drift: TemporalDriftSchedule::Gradual { start_event: 500, end_event: 900, to_regime: 1 },
        }
    }

    fn prequential_trace(
        dataset: &TemporalStreamDataset,
        include_updates: bool,
    ) -> PrequentialExecutionTrace {
        let mut actions = Vec::new();
        let mut state_sha = sha('a');
        let mut next_release = 0usize;

        for observation in &dataset.observations {
            while next_release < dataset.delayed_labels.len()
                && dataset.delayed_labels[next_release].label_available_at_us <= observation.timestamp_us
            {
                let label = &dataset.delayed_labels[next_release];
                actions.push(PrequentialAction::LabelReleased {
                    event_index: label.event_index,
                    timestamp_us: label.label_available_at_us,
                    label: label.label,
                });
                if include_updates {
                    let next_state = digest(
                        b"test-prequential-state",
                        format!("{state_sha}:{}", label.event_index).as_bytes(),
                    );
                    actions.push(PrequentialAction::ModelUpdate {
                        event_index: label.event_index,
                        timestamp_us: label.label_available_at_us,
                        state_before_sha256: state_sha.clone(),
                        state_after_sha256: next_state.clone(),
                    });
                    state_sha = next_state;
                }
                next_release += 1;
            }

            // Test fixture only: a test oracle emits the known label after constructing the
            // observation. The auditor checks chronology; candidate implementations do not
            // receive the delayed-label stream in their input interface.
            let label = dataset.delayed_labels[observation.event_index].label;
            actions.push(PrequentialAction::Prediction {
                event_index: observation.event_index,
                timestamp_us: observation.timestamp_us,
                predicted_class: Some(label),
                positive_probability_q15: if label == 1 { 32_768 } else { 0 },
                model_state_sha256: state_sha.clone(),
            });
        }

        while next_release < dataset.delayed_labels.len() {
            let label = &dataset.delayed_labels[next_release];
            actions.push(PrequentialAction::LabelReleased {
                event_index: label.event_index,
                timestamp_us: label.label_available_at_us,
                label: label.label,
            });
            if include_updates {
                let next_state = digest(
                    b"test-prequential-state",
                    format!("{state_sha}:{}", label.event_index).as_bytes(),
                );
                actions.push(PrequentialAction::ModelUpdate {
                    event_index: label.event_index,
                    timestamp_us: label.label_available_at_us,
                    state_before_sha256: state_sha.clone(),
                    state_after_sha256: next_state.clone(),
                });
                state_sha = next_state;
            }
            next_release += 1;
        }

        let mut trace = PrequentialExecutionTrace {
            schema: PREQUENTIAL_TRACE_SCHEMA.into(),
            version: BENCHMARK_VERSION.into(),
            run_id: "prequential-test-run".into(),
            candidate_id: "hdc-ltc-fixed-fixture".into(),
            candidate_artifact_sha256: sha('b'),
            dataset_sha256: dataset.dataset_sha256.clone(),
            initial_model_state_sha256: sha('a'),
            energy_receipt_sha256: sha('c'),
            metrology_receipt_sha256: sha('d'),
            actions,
            execution_trace_sha256: String::new(),
        };
        trace.execution_trace_sha256 = compute_prequential_execution_trace_sha256(&trace).unwrap();
        trace
    }

    #[test]
    fn static_manifest_requires_modern_optimized_tinyml_baselines_and_digest() {
        let manifest = static_manifest();
        validate_edge_benchmark_manifest(&manifest).unwrap();

        let mut incomplete = manifest.clone();
        incomplete.candidate_definitions.remove(0);
        incomplete.manifest_sha256 = compute_edge_benchmark_manifest_sha256(&incomplete).unwrap();
        assert!(validate_edge_benchmark_manifest(&incomplete).is_err());

        let mut tampered = manifest;
        tampered.locked_test_split_sha256 = sha('3');
        assert!(validate_edge_benchmark_manifest(&tampered).is_err());
    }

    #[test]
    fn prequential_audit_accepts_predict_release_update_state_chain() {
        let dataset = generate_temporal_drift_stream(&stream_config()).unwrap();
        let trace = prequential_trace(&dataset, true);
        let receipt = audit_prequential_execution(&dataset, &trace).unwrap();
        assert_eq!(receipt.metrics.event_count, dataset.observations.len());
        assert_eq!(receipt.metrics.label_release_count, dataset.observations.len());
        assert_eq!(receipt.metrics.model_update_count, dataset.observations.len());
        assert_eq!(receipt.metrics.accuracy, 1.0);
        assert_eq!(receipt.metrics.coverage, 1.0);
        assert_eq!(receipt.energy_receipt_sha256, trace.energy_receipt_sha256);
        assert!(is_sha256(&receipt.receipt_sha256));
    }

    #[test]
    fn prequential_audit_rejects_prediction_at_or_after_label_availability() {
        let dataset = generate_temporal_drift_stream(&stream_config()).unwrap();
        let mut trace = prequential_trace(&dataset, false);
        let prediction = trace.actions.iter_mut().find_map(|action| match action {
            PrequentialAction::Prediction { event_index, timestamp_us, .. } if *event_index == 0 => Some(timestamp_us),
            _ => None,
        }).unwrap();
        *prediction = dataset.delayed_labels[0].label_available_at_us;
        trace.execution_trace_sha256 = compute_prequential_execution_trace_sha256(&trace).unwrap();
        assert!(audit_prequential_execution(&dataset, &trace).unwrap_err().to_string().contains("strictly before"));
    }

    #[test]
    fn prequential_audit_rejects_update_before_label_release() {
        let dataset = generate_temporal_drift_stream(&stream_config()).unwrap();
        let mut trace = prequential_trace(&dataset, true);
        let update_index = trace.actions.iter().position(|action| {
            matches!(action, PrequentialAction::ModelUpdate { event_index: 0, .. })
        }).unwrap();
        let release_index = trace.actions.iter().position(|action| {
            matches!(action, PrequentialAction::LabelReleased { event_index: 0, .. })
        }).unwrap();
        trace.actions.swap(update_index, release_index);
        trace.execution_trace_sha256 = compute_prequential_execution_trace_sha256(&trace).unwrap();
        assert!(audit_prequential_execution(&dataset, &trace).is_err());
    }

    #[test]
    fn prequential_audit_rejects_broken_model_state_chain() {
        let dataset = generate_temporal_drift_stream(&stream_config()).unwrap();
        let mut trace = prequential_trace(&dataset, true);
        let prediction = trace.actions.iter_mut().find_map(|action| match action {
            PrequentialAction::Prediction { event_index, model_state_sha256, .. } if *event_index == 5 => Some(model_state_sha256),
            _ => None,
        }).unwrap();
        *prediction = sha('e');
        trace.execution_trace_sha256 = compute_prequential_execution_trace_sha256(&trace).unwrap();
        assert!(audit_prequential_execution(&dataset, &trace).unwrap_err().to_string().contains("current model-state"));
    }

    #[test]
    fn prequential_abstentions_reduce_coverage_and_are_not_scored_as_correct() {
        let dataset = generate_temporal_drift_stream(&stream_config()).unwrap();
        let mut trace = prequential_trace(&dataset, false);
        let prediction = trace.actions.iter_mut().find_map(|action| match action {
            PrequentialAction::Prediction { event_index: 0, predicted_class, .. } => Some(predicted_class),
            _ => None,
        }).unwrap();
        *prediction = None;
        trace.execution_trace_sha256 = compute_prequential_execution_trace_sha256(&trace).unwrap();
        let receipt = audit_prequential_execution(&dataset, &trace).unwrap();
        assert_eq!(receipt.metrics.abstention_count, 1);
        assert!(receipt.metrics.coverage < 1.0);
        assert!(receipt.metrics.accuracy < 1.0);
    }

    #[test]
    fn prequential_trace_digest_tampering_is_rejected() {
        let dataset = generate_temporal_drift_stream(&stream_config()).unwrap();
        let mut trace = prequential_trace(&dataset, false);
        trace.energy_receipt_sha256 = sha('f');
        assert!(audit_prequential_execution(&dataset, &trace).unwrap_err().to_string().contains("digest mismatch"));
    }

    #[test]
    fn default_hdc_ltc_profile_has_expected_payload_lower_bound() {
        let estimate = estimate_hdc_ltc_memory(16_384, &[4, 8, 4]).unwrap();
        assert_eq!(estimate.vector_payload_count, 134);
        assert_eq!(estimate.persistent_vector_payload_bytes, 8_781_824);
        assert_eq!(estimate.one_input_vector_bytes, 65_536);
        assert_eq!(estimate.minimum_working_set_payload_bytes, 8_847_360);
    }

    #[test]
    fn compact_hdc_ltc_profile_changes_mcu_fit_boundary() {
        let estimate = estimate_hdc_ltc_memory(256, &[4, 8, 4]).unwrap();
        assert_eq!(estimate.persistent_vector_payload_bytes, 137_216);
        assert_eq!(estimate.minimum_working_set_payload_bytes, 138_240);
        let assessment = assess_hdc_ltc_sram_budget(estimate, 256 * 1024, 128 * 1024).unwrap();
        assert!(!assessment.fits_vector_payload_lower_bound);
    }

    #[test]
    fn temporal_stream_is_deterministic_and_replayable() {
        let first = generate_temporal_drift_stream(&stream_config()).unwrap();
        let second = generate_temporal_drift_stream(&stream_config()).unwrap();
        assert!(first == second);
        validate_temporal_stream_dataset(&first).unwrap();
    }

    #[test]
    fn temporal_stream_covers_jitter_dropout_drift_and_ood() {
        let stream = generate_temporal_drift_stream(&stream_config()).unwrap();
        assert!(stream.observations.windows(2).all(|p| p[1].timestamp_us > p[0].timestamp_us));
        assert!(stream.observations.iter().any(|e| e.missing_feature_mask != 0));
        assert!(stream.delayed_labels.iter().any(|e| e.is_out_of_distribution));
        assert_eq!(stream.delayed_labels[499].transition_fraction_ppm, 0);
        assert_eq!(stream.delayed_labels[899].transition_fraction_ppm, 997_500);
        assert_eq!(stream.delayed_labels[900].transition_fraction_ppm, 1_000_000);
        assert!(stream.delayed_labels.iter().zip(&stream.observations)
            .all(|(label, observation)| label.label_available_at_us > observation.timestamp_us));
    }

    #[test]
    fn dropped_features_are_zero_filled_before_model_observation() {
        let stream = generate_temporal_drift_stream(&stream_config()).unwrap();
        let mut dropped = 0usize;
        for observation in &stream.observations {
            for feature in 0..4 {
                if observation.missing_feature_mask & (1 << feature) != 0 {
                    dropped += 1;
                    assert_eq!(observation.features[feature], 0);
                }
            }
        }
        assert!(dropped > 0);
    }

    #[test]
    fn stream_digest_rejects_event_tampering() {
        let mut stream = generate_temporal_drift_stream(&stream_config()).unwrap();
        stream.delayed_labels[0].label ^= 1;
        assert!(validate_temporal_stream_dataset(&stream).is_err());
    }

    #[test]
    fn invalid_stream_schedule_is_rejected() {
        let mut config = stream_config();
        config.drift = TemporalDriftSchedule::Gradual { start_event: 900, end_event: 500, to_regime: 1 };
        assert!(config.validate().is_err());
    }
}
