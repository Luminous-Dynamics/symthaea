//! Label-hidden synthetic benchmark for periodic on-device adaptation.
//!
//! The candidate interface receives only unlabeled fixed-point target features.
//! Labels, source/target prior, domain transformation and random seed remain
//! evaluator-owned. This is an algorithm-track fixture, not a sandbox or a
//! physical measurement of adaptation latency/energy.

use std::error::Error;
use std::fmt;

use super::digest::CanonicalHasher;

const PROBABILITY_SCALE: u32 = 1_000_000;
const MAX_BATCHES: u32 = 1_024;
const MAX_SAMPLES_PER_BATCH: usize = 512;
const SOURCE_CENTRE_Q15: i32 = 16_384;
const ADAPTER_STATE_WRITE_BYTES: u64 = 8;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgeAdaptationError {
    pub field: String,
    pub message: String,
}

impl EdgeAdaptationError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for EdgeAdaptationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for EdgeAdaptationError {}

/// Q15 feature values are presented to the candidate without evaluator labels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureQ15 {
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicTargetSample {
    pub sample_id: u64,
    pub features: FeatureQ15,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicTargetBatch {
    pub batch_index: u32,
    pub samples: Vec<PublicTargetSample>,
}

/// Fixed affine/corruption transformation applied to target samples after the
/// configured shift start. Rotation is a right-angle quadrant, avoiding
/// cross-platform trigonometric drift in the synthetic fixture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EdgeDomainShift {
    pub rotation_quadrants: u8,
    /// 1,000,000 ppm is identity scale. Valid range is 100,000..=2,000,000.
    pub scale_ppm: u32,
    pub bias_x_q15: i32,
    pub bias_y_q15: i32,
    pub noise_amplitude_q15: u32,
}

impl EdgeDomainShift {
    pub const IDENTITY: Self = Self {
        rotation_quadrants: 0,
        scale_ppm: PROBABILITY_SCALE,
        bias_x_q15: 0,
        bias_y_q15: 0,
        noise_amplitude_q15: 0,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EdgeAdaptationScenario {
    /// Evaluator-only seed. Never passed to candidate model methods.
    pub seed: u64,
    pub batch_count: u32,
    pub samples_per_batch: u32,
    /// Target class-0 prior, in ppm. Values 0 and 1,000,000 model absent-class
    /// target exposure; 500,000 is balanced class exposure.
    pub class0_probability_ppm: u32,
    pub shift_start_batch: u32,
    pub shift: EdgeDomainShift,
}

impl EdgeAdaptationScenario {
    pub fn validate(&self) -> Result<(), EdgeAdaptationError> {
        if self.batch_count == 0 || self.batch_count > MAX_BATCHES {
            return Err(EdgeAdaptationError::new(
                "scenario.batch_count",
                format!("batch count must be in 1..={MAX_BATCHES}"),
            ));
        }
        if self.samples_per_batch == 0
            || self.samples_per_batch as usize > MAX_SAMPLES_PER_BATCH
        {
            return Err(EdgeAdaptationError::new(
                "scenario.samples_per_batch",
                format!("samples per batch must be in 1..={MAX_SAMPLES_PER_BATCH}"),
            ));
        }
        if self.class0_probability_ppm > PROBABILITY_SCALE {
            return Err(EdgeAdaptationError::new(
                "scenario.class0_probability_ppm",
                "class prior must be in [0, 1,000,000] ppm",
            ));
        }
        if self.shift_start_batch > self.batch_count {
            return Err(EdgeAdaptationError::new(
                "scenario.shift_start_batch",
                "shift start cannot exceed the episode batch count",
            ));
        }
        if self.shift.rotation_quadrants > 3
            || !(100_000..=2_000_000).contains(&self.shift.scale_ppm)
            || self.shift.bias_x_q15.unsigned_abs() > 100_000
            || self.shift.bias_y_q15.unsigned_abs() > 100_000
            || self.shift.noise_amplitude_q15 > 16_384
        {
            return Err(EdgeAdaptationError::new(
                "scenario.shift",
                "rotation, scale, bias or noise is outside the supported fixture bounds",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptationTrigger {
    Disabled,
    /// Adapt after every Nth completed target batch.
    Periodic { period_batches: u32, minimum_samples: u32 },
    /// Evaluate an unlabeled batch mean every batch and update only when its
    /// Manhattan distance from the current offset reaches the threshold.
    ShiftTriggered { threshold_q15: u32, minimum_samples: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptationUpdateReceipt {
    pub batch_index: u32,
    pub adaptation_considered: bool,
    pub update_applied: bool,
    pub sample_count: u32,
    /// Declared operation estimate, not measured CPU cycles or latency.
    pub compute_operations_estimate: u64,
    /// Persistent model-state bytes declared written by this update.
    pub persistent_state_write_bytes: u64,
    pub previous_state_sha256: [u8; 32],
    pub next_state_sha256: [u8; 32],
}

pub trait UnsupervisedEdgeModel {
    /// Prediction is read-only with respect to model state and receives no label.
    fn predict(&self, sample: &PublicTargetSample) -> u8;

    /// Optional update after the entire current batch has already been scored.
    /// Implementations receive unlabeled features only.
    fn adapt_after_batch(&mut self, batch: &PublicTargetBatch) -> AdaptationUpdateReceipt;

    fn state_sha256(&self) -> [u8; 32];
}

/// Transparent binary linear-control model with an optional unlabeled mean
/// centering update. Its classifier uses x-offset_x; offset_y is retained and
/// committed so non-x shifts still incur the actual update-state cost.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeanCenterEdgeModel {
    offset_x_q15: i32,
    offset_y_q15: i32,
    trigger: AdaptationTrigger,
    alpha_ppm: u32,
}

impl MeanCenterEdgeModel {
    pub fn frozen() -> Self {
        Self {
            offset_x_q15: 0,
            offset_y_q15: 0,
            trigger: AdaptationTrigger::Disabled,
            alpha_ppm: 0,
        }
    }

    pub fn periodic(
        period_batches: u32,
        minimum_samples: u32,
        alpha_ppm: u32,
    ) -> Result<Self, EdgeAdaptationError> {
        if period_batches == 0 || minimum_samples == 0 || alpha_ppm == 0
            || alpha_ppm > PROBABILITY_SCALE
        {
            return Err(EdgeAdaptationError::new(
                "model.periodic",
                "period, minimum samples and alpha must be positive; alpha <= 1,000,000 ppm",
            ));
        }
        Ok(Self {
            offset_x_q15: 0,
            offset_y_q15: 0,
            trigger: AdaptationTrigger::Periodic { period_batches, minimum_samples },
            alpha_ppm,
        })
    }

    pub fn shift_triggered(
        threshold_q15: u32,
        minimum_samples: u32,
        alpha_ppm: u32,
    ) -> Result<Self, EdgeAdaptationError> {
        if threshold_q15 == 0 || minimum_samples == 0 || alpha_ppm == 0
            || alpha_ppm > PROBABILITY_SCALE
        {
            return Err(EdgeAdaptationError::new(
                "model.shift_triggered",
                "threshold, minimum samples and alpha must be positive; alpha <= 1,000,000 ppm",
            ));
        }
        Ok(Self {
            offset_x_q15: 0,
            offset_y_q15: 0,
            trigger: AdaptationTrigger::ShiftTriggered { threshold_q15, minimum_samples },
            alpha_ppm,
        })
    }

    fn state_digest_for(offset_x_q15: i32, offset_y_q15: i32) -> [u8; 32] {
        let mut hash = CanonicalHasher::new(b"symthaea.temporal.edge-adaptation.mean-center-state.v1");
        hash.u32(offset_x_q15 as u32);
        hash.u32(offset_y_q15 as u32);
        hash.finish()
    }

    fn blend(old: i32, observed: i32, alpha_ppm: u32) -> i32 {
        let alpha = i64::from(alpha_ppm);
        let retained = i64::from(PROBABILITY_SCALE - alpha_ppm);
        let numerator = i64::from(old) * retained + i64::from(observed) * alpha;
        let half = i64::from(PROBABILITY_SCALE / 2);
        if numerator >= 0 {
            ((numerator + half) / i64::from(PROBABILITY_SCALE)) as i32
        } else {
            ((numerator - half) / i64::from(PROBABILITY_SCALE)) as i32
        }
    }
}

impl UnsupervisedEdgeModel for MeanCenterEdgeModel {
    fn predict(&self, sample: &PublicTargetSample) -> u8 {
        if sample.features.x.saturating_sub(self.offset_x_q15) < 0 { 0 } else { 1 }
    }

    fn adapt_after_batch(&mut self, batch: &PublicTargetBatch) -> AdaptationUpdateReceipt {
        let previous_state_sha256 = self.state_sha256();
        let sample_count = batch.samples.len().min(u32::MAX as usize) as u32;

        let (adaptation_considered, trigger_met, compute_operations_estimate) = match self.trigger {
            AdaptationTrigger::Disabled => (false, false, 0),
            AdaptationTrigger::Periodic { period_batches, minimum_samples } => {
                let scheduled = (batch.batch_index + 1) % period_batches == 0;
                let enough = sample_count >= minimum_samples;
                let considered = scheduled && enough;
                let ops = if considered {
                    u64::from(sample_count) * 4 + 12
                } else {
                    0
                };
                (considered, considered, ops)
            }
            AdaptationTrigger::ShiftTriggered { threshold_q15, minimum_samples } => {
                let ops = u64::from(sample_count) * 4 + 8;
                if sample_count < minimum_samples || sample_count == 0 {
                    (true, false, ops)
                } else {
                    let (mean_x, mean_y) = batch_mean(batch);
                    let distance = i64::from(mean_x.saturating_sub(self.offset_x_q15)).abs()
                        + i64::from(mean_y.saturating_sub(self.offset_y_q15)).abs();
                    (true, distance >= i64::from(threshold_q15), ops)
                }
            }
        };

        let mut update_applied = false;
        let mut persistent_state_write_bytes = 0_u64;
        if trigger_met && sample_count > 0 {
            let (mean_x, mean_y) = batch_mean(batch);
            let next_x = Self::blend(self.offset_x_q15, mean_x, self.alpha_ppm);
            let next_y = Self::blend(self.offset_y_q15, mean_y, self.alpha_ppm);
            if next_x != self.offset_x_q15 || next_y != self.offset_y_q15 {
                self.offset_x_q15 = next_x;
                self.offset_y_q15 = next_y;
                update_applied = true;
                persistent_state_write_bytes = ADAPTER_STATE_WRITE_BYTES;
            }
        }

        AdaptationUpdateReceipt {
            batch_index: batch.batch_index,
            adaptation_considered,
            update_applied,
            sample_count,
            compute_operations_estimate,
            persistent_state_write_bytes,
            previous_state_sha256,
            next_state_sha256: self.state_sha256(),
        }
    }

    fn state_sha256(&self) -> [u8; 32] {
        Self::state_digest_for(self.offset_x_q15, self.offset_y_q15)
    }
}

fn batch_mean(batch: &PublicTargetBatch) -> (i32, i32) {
    if batch.samples.is_empty() {
        return (0, 0);
    }
    let mut sum_x = 0_i64;
    let mut sum_y = 0_i64;
    for sample in &batch.samples {
        sum_x += i64::from(sample.features.x);
        sum_y += i64::from(sample.features.y);
    }
    let count = batch.samples.len() as i64;
    ((sum_x / count) as i32, (sum_y / count) as i32)
}

#[derive(Clone, Copy, Debug)]
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

    fn below(&mut self, upper_exclusive: u64) -> u64 {
        if upper_exclusive == 0 { 0 } else { self.next_u64() % upper_exclusive }
    }
}

#[derive(Clone, Debug)]
struct HiddenTargetBatch {
    public: PublicTargetBatch,
    labels: Vec<u8>,
}

fn rotate_q15(feature: FeatureQ15, quadrants: u8) -> FeatureQ15 {
    match quadrants {
        0 => feature,
        1 => FeatureQ15 { x: -feature.y, y: feature.x },
        2 => FeatureQ15 { x: -feature.x, y: -feature.y },
        3 => FeatureQ15 { x: feature.y, y: -feature.x },
        _ => feature,
    }
}

fn scale_q15(value: i32, scale_ppm: u32) -> i32 {
    ((i64::from(value) * i64::from(scale_ppm)) / i64::from(PROBABILITY_SCALE))
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn make_hidden_batch(
    scenario: &EdgeAdaptationScenario,
    batch_index: u32,
) -> HiddenTargetBatch {
    let mut samples = Vec::with_capacity(scenario.samples_per_batch as usize);
    let mut labels = Vec::with_capacity(scenario.samples_per_batch as usize);
    let shifted = batch_index >= scenario.shift_start_batch;

    for sample_index in 0..scenario.samples_per_batch {
        let sample_id = u64::from(batch_index) * u64::from(scenario.samples_per_batch)
            + u64::from(sample_index);
        // Randomness is keyed by stable sample identity, not policy call order.
        // A candidate sampling/updating at another cadence cannot alter later
        // counterfactual samples.
        let sample_seed = scenario.seed
            ^ sample_id.wrapping_mul(0x9E3779B97F4A7C15)
            ^ 0x454447455F414450;
        let mut rng = SplitMix64::new(sample_seed);
        let label = if rng.below(u64::from(PROBABILITY_SCALE))
            < u64::from(scenario.class0_probability_ppm)
        {
            0_u8
        } else {
            1_u8
        };
        let centre_x = if label == 0 { -SOURCE_CENTRE_Q15 } else { SOURCE_CENTRE_Q15 };
        let centre = FeatureQ15 { x: centre_x, y: 0 };
        let mut feature = if shifted {
            let rotated = rotate_q15(centre, scenario.shift.rotation_quadrants);
            FeatureQ15 {
                x: scale_q15(rotated.x, scenario.shift.scale_ppm)
                    .saturating_add(scenario.shift.bias_x_q15),
                y: scale_q15(rotated.y, scenario.shift.scale_ppm)
                    .saturating_add(scenario.shift.bias_y_q15),
            }
        } else {
            centre
        };

        if shifted && scenario.shift.noise_amplitude_q15 > 0 {
            let span = u64::from(scenario.shift.noise_amplitude_q15)
                .saturating_mul(2)
                .saturating_add(1);
            let noise_x = rng.below(span) as i64
                - i64::from(scenario.shift.noise_amplitude_q15);
            let noise_y = rng.below(span) as i64
                - i64::from(scenario.shift.noise_amplitude_q15);
            feature.x = feature.x.saturating_add(noise_x as i32);
            feature.y = feature.y.saturating_add(noise_y as i32);
        }

        samples.push(PublicTargetSample { sample_id, features: feature });
        labels.push(label);
    }

    HiddenTargetBatch {
        public: PublicTargetBatch { batch_index, samples },
        labels,
    }
}

fn scenario_digest(scenario: &EdgeAdaptationScenario) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.edge-adaptation-scenario.v1");
    hash.u64(scenario.seed);
    hash.u32(scenario.batch_count);
    hash.u32(scenario.samples_per_batch);
    hash.u32(scenario.class0_probability_ppm);
    hash.u32(scenario.shift_start_batch);
    hash.u8(scenario.shift.rotation_quadrants);
    hash.u32(scenario.shift.scale_ppm);
    hash.u32(scenario.shift.bias_x_q15 as u32);
    hash.u32(scenario.shift.bias_y_q15 as u32);
    hash.u32(scenario.shift.noise_amplitude_q15);
    hash.finish()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgeAdaptationBatchMetrics {
    pub batch_index: u32,
    pub sample_count: u64,
    pub confusion_counts: [u64; 4],
    pub accuracy_ppm: u32,
    pub class0_support: u64,
    pub class1_support: u64,
    pub class0_recall_ppm: Option<u32>,
    pub class1_recall_ppm: Option<u32>,
    /// Mean recall over classes present in this batch; support is reported too.
    pub balanced_accuracy_present_classes_ppm: Option<u32>,
    pub worst_present_class_recall_ppm: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgeAdaptationRunReceipt {
    pub scenario_sha256: [u8; 32],
    pub candidate_sha256: [u8; 32],
    pub initial_state_sha256: [u8; 32],
    pub final_state_sha256: [u8; 32],
    pub trace_sha256: [u8; 32],
    pub batch_metrics: Vec<EdgeAdaptationBatchMetrics>,
    pub update_receipts: Vec<AdaptationUpdateReceipt>,
    pub confusion_counts: [u64; 4],
    pub sample_count: u64,
    pub accuracy_ppm: u32,
    pub class0_support: u64,
    pub class1_support: u64,
    pub class0_recall_ppm: Option<u32>,
    pub class1_recall_ppm: Option<u32>,
    pub balanced_accuracy_present_classes_ppm: Option<u32>,
    pub worst_present_class_recall_ppm: Option<u32>,
    pub adaptation_considered_count: u64,
    pub updates_applied_count: u64,
    pub compute_operations_estimate: u64,
    pub persistent_state_write_bytes: u64,
    pub receipt_sha256: [u8; 32],
}

fn ppm(numerator: u64, denominator: u64) -> Option<u32> {
    if denominator == 0 {
        None
    } else {
        Some(((u128::from(numerator) * u128::from(PROBABILITY_SCALE)
            / u128::from(denominator))
            .min(u128::from(PROBABILITY_SCALE))) as u32)
    }
}

fn metrics_from_confusion(
    batch_index: u32,
    confusion: [u64; 4],
) -> EdgeAdaptationBatchMetrics {
    let class0_support = confusion[0].saturating_add(confusion[1]);
    let class1_support = confusion[2].saturating_add(confusion[3]);
    let total = class0_support.saturating_add(class1_support);
    let class0_recall = ppm(confusion[0], class0_support);
    let class1_recall = ppm(confusion[3], class1_support);
    let mut recalls = Vec::with_capacity(2);
    if let Some(value) = class0_recall { recalls.push(value); }
    if let Some(value) = class1_recall { recalls.push(value); }
    let balanced = if recalls.is_empty() {
        None
    } else {
        Some((recalls.iter().map(|value| u64::from(*value)).sum::<u64>()
            / recalls.len() as u64) as u32)
    };
    let worst = recalls.iter().copied().min();
    EdgeAdaptationBatchMetrics {
        batch_index,
        sample_count: total,
        confusion_counts: confusion,
        accuracy_ppm: ppm(confusion[0].saturating_add(confusion[3]), total).unwrap_or(0),
        class0_support,
        class1_support,
        class0_recall_ppm: class0_recall,
        class1_recall_ppm: class1_recall,
        balanced_accuracy_present_classes_ppm: balanced,
        worst_present_class_recall_ppm: worst,
    }
}

fn receipt_digest(receipt: &EdgeAdaptationRunReceipt) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.edge-adaptation-run-receipt.v1");
    hash.bytes(&receipt.scenario_sha256);
    hash.bytes(&receipt.candidate_sha256);
    hash.bytes(&receipt.initial_state_sha256);
    hash.bytes(&receipt.final_state_sha256);
    hash.bytes(&receipt.trace_sha256);
    hash.usize(receipt.batch_metrics.len());
    for batch in &receipt.batch_metrics {
        hash.u32(batch.batch_index);
        hash.u64(batch.sample_count);
        for count in batch.confusion_counts { hash.u64(count); }
        hash.u32(batch.accuracy_ppm);
        hash.u64(batch.class0_support);
        hash.u64(batch.class1_support);
        hash.u32(batch.class0_recall_ppm.unwrap_or(u32::MAX));
        hash.u32(batch.class1_recall_ppm.unwrap_or(u32::MAX));
        hash.u32(batch.balanced_accuracy_present_classes_ppm.unwrap_or(u32::MAX));
        hash.u32(batch.worst_present_class_recall_ppm.unwrap_or(u32::MAX));
    }
    hash.usize(receipt.update_receipts.len());
    for update in &receipt.update_receipts {
        hash.u32(update.batch_index);
        hash.u8(u8::from(update.adaptation_considered));
        hash.u8(u8::from(update.update_applied));
        hash.u32(update.sample_count);
        hash.u64(update.compute_operations_estimate);
        hash.u64(update.persistent_state_write_bytes);
        hash.bytes(&update.previous_state_sha256);
        hash.bytes(&update.next_state_sha256);
    }
    for count in receipt.confusion_counts { hash.u64(count); }
    hash.u64(receipt.sample_count);
    hash.u32(receipt.accuracy_ppm);
    hash.u64(receipt.class0_support);
    hash.u64(receipt.class1_support);
    hash.u32(receipt.class0_recall_ppm.unwrap_or(u32::MAX));
    hash.u32(receipt.class1_recall_ppm.unwrap_or(u32::MAX));
    hash.u32(receipt.balanced_accuracy_present_classes_ppm.unwrap_or(u32::MAX));
    hash.u32(receipt.worst_present_class_recall_ppm.unwrap_or(u32::MAX));
    hash.u64(receipt.adaptation_considered_count);
    hash.u64(receipt.updates_applied_count);
    hash.u64(receipt.compute_operations_estimate);
    hash.u64(receipt.persistent_state_write_bytes);
    hash.finish()
}

/// Run a label-hidden target episode. For each batch the model predicts every
/// sample first; the evaluator then releases only an unlabeled batch to the
/// candidate's adaptation callback. Any update can affect subsequent batches
/// but never the predictions already scored in the current batch.
pub fn run_edge_adaptation_episode<M: UnsupervisedEdgeModel>(
    scenario: EdgeAdaptationScenario,
    candidate_sha256: [u8; 32],
    model: &mut M,
) -> Result<EdgeAdaptationRunReceipt, EdgeAdaptationError> {
    scenario.validate()?;
    if candidate_sha256 == [0; 32] {
        return Err(EdgeAdaptationError::new(
            "candidate_sha256",
            "candidate artifact/configuration digest must be nonzero",
        ));
    }

    let initial_state_sha256 = model.state_sha256();
    if initial_state_sha256 == [0; 32] {
        return Err(EdgeAdaptationError::new(
            "model.state_sha256",
            "initial model-state digest must be nonzero",
        ));
    }
    let mut previous_state_sha256 = initial_state_sha256;
    let mut trace = CanonicalHasher::new(b"symthaea.temporal.edge-adaptation-public-trace.v1");
    let mut batch_metrics = Vec::with_capacity(scenario.batch_count as usize);
    let mut update_receipts = Vec::with_capacity(scenario.batch_count as usize);
    let mut confusion_counts = [0_u64; 4];
    let mut sample_count = 0_u64;

    for batch_index in 0..scenario.batch_count {
        let hidden = make_hidden_batch(&scenario, batch_index);
        trace.u32(batch_index);
        trace.usize(hidden.public.samples.len());

        for (sample, label) in hidden.public.samples.iter().zip(hidden.labels.iter()) {
            let state_before_prediction = model.state_sha256();
            if state_before_prediction != previous_state_sha256 {
                return Err(EdgeAdaptationError::new(
                    "model.state_sha256",
                    "model state changed outside an audited adaptation boundary",
                ));
            }
            let prediction = model.predict(sample);
            if prediction > 1 {
                return Err(EdgeAdaptationError::new(
                    "model.prediction",
                    "binary classifier prediction must be class 0 or 1",
                ));
            }
            let truth = *label;
            let cell = usize::from(truth) * 2 + usize::from(prediction);
            confusion_counts[cell] = confusion_counts[cell].checked_add(1).ok_or_else(|| {
                EdgeAdaptationError::new("result.confusion_counts", "counter overflow")
            })?;
            sample_count = sample_count.checked_add(1).ok_or_else(|| {
                EdgeAdaptationError::new("result.sample_count", "counter overflow")
            })?;
            trace.u64(sample.sample_id);
            trace.u32(sample.features.x as u32);
            trace.u32(sample.features.y as u32);
            trace.u8(prediction);
        }

        let batch_metrics_for_receipt = {
            let mut local = [0_u64; 4];
            let batch_start = (batch_index * scenario.samples_per_batch) as usize;
            let batch_end = batch_start + hidden.labels.len();
            // Recompute the batch confusion from the candidate predictions
            // already committed in the trace by using a second deterministic pass.
            // The model is read-only during predict, so this is exactly paired.
            for (sample, truth) in hidden.public.samples.iter().zip(hidden.labels.iter()) {
                let prediction = model.predict(sample);
                let cell = usize::from(*truth) * 2 + usize::from(prediction);
                local[cell] = local[cell].saturating_add(1);
            }
            let _ = (batch_start, batch_end);
            local
        };
        let batch_metrics_for_receipt = metrics_from_confusion(batch_index, batch_metrics_for_receipt);
        batch_metrics.push(batch_metrics_for_receipt);

        let update = model.adapt_after_batch(&hidden.public);
        if update.batch_index != batch_index
            || update.sample_count as usize != hidden.public.samples.len()
            || update.previous_state_sha256 != previous_state_sha256
            || update.next_state_sha256 != model.state_sha256()
            || (update.update_applied && !update.adaptation_considered)
            || (update.update_applied && update.next_state_sha256 == update.previous_state_sha256)
            || (!update.update_applied && update.next_state_sha256 != update.previous_state_sha256)
            || (update.update_applied && update.persistent_state_write_bytes == 0)
            || (!update.update_applied && update.persistent_state_write_bytes != 0)
        {
            return Err(EdgeAdaptationError::new(
                format!("update_receipts[{batch_index}]"),
                "adaptation receipt does not match the observed model state transition",
            ));
        }
        trace.u8(u8::from(update.adaptation_considered));
        trace.u8(u8::from(update.update_applied));
        trace.u64(update.compute_operations_estimate);
        trace.u64(update.persistent_state_write_bytes);
        trace.bytes(&update.previous_state_sha256);
        trace.bytes(&update.next_state_sha256);
        previous_state_sha256 = update.next_state_sha256;
        update_receipts.push(update);
    }

    let overall = metrics_from_confusion(scenario.batch_count, confusion_counts);
    let adaptation_considered_count = update_receipts
        .iter()
        .filter(|update| update.adaptation_considered)
        .count() as u64;
    let updates_applied_count = update_receipts
        .iter()
        .filter(|update| update.update_applied)
        .count() as u64;
    let compute_operations_estimate = update_receipts.iter()
        .map(|update| update.compute_operations_estimate)
        .sum::<u64>();
    let persistent_state_write_bytes = update_receipts.iter()
        .map(|update| update.persistent_state_write_bytes)
        .sum::<u64>();

    let mut receipt = EdgeAdaptationRunReceipt {
        scenario_sha256: scenario_digest(&scenario),
        candidate_sha256,
        initial_state_sha256,
        final_state_sha256: previous_state_sha256,
        trace_sha256: trace.finish(),
        batch_metrics,
        update_receipts,
        confusion_counts,
        sample_count,
        accuracy_ppm: overall.accuracy_ppm,
        class0_support: overall.class0_support,
        class1_support: overall.class1_support,
        class0_recall_ppm: overall.class0_recall_ppm,
        class1_recall_ppm: overall.class1_recall_ppm,
        balanced_accuracy_present_classes_ppm: overall.balanced_accuracy_present_classes_ppm,
        worst_present_class_recall_ppm: overall.worst_present_class_recall_ppm,
        adaptation_considered_count,
        updates_applied_count,
        compute_operations_estimate,
        persistent_state_write_bytes,
        receipt_sha256: [0; 32],
    };
    receipt.receipt_sha256 = receipt_digest(&receipt);
    validate_edge_adaptation_receipt(&receipt)?;
    Ok(receipt)
}

pub fn validate_edge_adaptation_receipt(
    receipt: &EdgeAdaptationRunReceipt,
) -> Result<(), EdgeAdaptationError> {
    if receipt.scenario_sha256 == [0; 32]
        || receipt.candidate_sha256 == [0; 32]
        || receipt.initial_state_sha256 == [0; 32]
        || receipt.final_state_sha256 == [0; 32]
    {
        return Err(EdgeAdaptationError::new(
            "receipt.identity",
            "scenario, candidate and state digests must be nonzero",
        ));
    }
    if receipt.batch_metrics.is_empty()
        || receipt.batch_metrics.len() != receipt.update_receipts.len()
    {
        return Err(EdgeAdaptationError::new(
            "receipt.batch_metrics",
            "one batch metric and one update receipt are required per target batch",
        ));
    }

    let mut total_confusion = [0_u64; 4];
    let mut total_samples = 0_u64;
    let mut previous_state = receipt.initial_state_sha256;
    let mut considered = 0_u64;
    let mut applied = 0_u64;
    let mut operations = 0_u64;
    let mut writes = 0_u64;

    for index in 0..receipt.batch_metrics.len() {
        let batch = &receipt.batch_metrics[index];
        let update = &receipt.update_receipts[index];
        if batch.batch_index as usize != index || update.batch_index as usize != index {
            return Err(EdgeAdaptationError::new(
                format!("receipt.batch[{index}]"),
                "batch metrics and update receipts must be contiguous and ordered",
            ));
        }
        let recomputed = metrics_from_confusion(batch.batch_index, batch.confusion_counts);
        if *batch != recomputed {
            return Err(EdgeAdaptationError::new(
                format!("receipt.batch_metrics[{index}]"),
                "batch metrics do not reproduce from the confusion counts",
            ));
        }
        for cell in 0..4 {
            total_confusion[cell] = total_confusion[cell]
                .checked_add(batch.confusion_counts[cell])
                .ok_or_else(|| EdgeAdaptationError::new(
                    "receipt.confusion_counts", "aggregate confusion overflow",
                ))?;
        }
        total_samples = total_samples.checked_add(batch.sample_count).ok_or_else(|| {
            EdgeAdaptationError::new("receipt.sample_count", "aggregate sample overflow")
        })?;

        if update.previous_state_sha256 != previous_state
            || (update.update_applied && !update.adaptation_considered)
            || (update.update_applied && update.next_state_sha256 == update.previous_state_sha256)
            || (!update.update_applied && update.next_state_sha256 != update.previous_state_sha256)
            || (update.update_applied && update.persistent_state_write_bytes == 0)
            || (!update.update_applied && update.persistent_state_write_bytes != 0)
        {
            return Err(EdgeAdaptationError::new(
                format!("receipt.update_receipts[{index}]"),
                "state transition chain or update/write semantics are invalid",
            ));
        }
        previous_state = update.next_state_sha256;
        if update.adaptation_considered { considered += 1; }
        if update.update_applied { applied += 1; }
        operations = operations.checked_add(update.compute_operations_estimate).ok_or_else(|| {
            EdgeAdaptationError::new("receipt.compute_operations_estimate", "overflow")
        })?;
        writes = writes.checked_add(update.persistent_state_write_bytes).ok_or_else(|| {
            EdgeAdaptationError::new("receipt.persistent_state_write_bytes", "overflow")
        })?;
    }

    let overall = metrics_from_confusion(receipt.batch_metrics.len() as u32, total_confusion);
    if previous_state != receipt.final_state_sha256
        || total_confusion != receipt.confusion_counts
        || total_samples != receipt.sample_count
        || receipt.accuracy_ppm != overall.accuracy_ppm
        || receipt.class0_support != overall.class0_support
        || receipt.class1_support != overall.class1_support
        || receipt.class0_recall_ppm != overall.class0_recall_ppm
        || receipt.class1_recall_ppm != overall.class1_recall_ppm
        || receipt.balanced_accuracy_present_classes_ppm != overall.balanced_accuracy_present_classes_ppm
        || receipt.worst_present_class_recall_ppm != overall.worst_present_class_recall_ppm
        || receipt.adaptation_considered_count != considered
        || receipt.updates_applied_count != applied
        || receipt.compute_operations_estimate != operations
        || receipt.persistent_state_write_bytes != writes
    {
        return Err(EdgeAdaptationError::new(
            "receipt.summary",
            "aggregate metrics or update costs do not reproduce from batch receipts",
        ));
    }
    if receipt.receipt_sha256 != receipt_digest(receipt) {
        return Err(EdgeAdaptationError::new(
            "receipt.receipt_sha256",
            "canonical receipt digest mismatch",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario() -> EdgeAdaptationScenario {
        EdgeAdaptationScenario {
            seed: 0xB077A,
            batch_count: 12,
            samples_per_batch: 64,
            class0_probability_ppm: 500_000,
            shift_start_batch: 0,
            shift: EdgeDomainShift {
                rotation_quadrants: 0,
                scale_ppm: PROBABILITY_SCALE,
                bias_x_q15: 20_000,
                bias_y_q15: 0,
                noise_amplitude_q15: 0,
            },
        }
    }

    fn candidate_digest(byte: u8) -> [u8; 32] { [byte; 32] }

    #[test]
    fn periodic_mean_centering_recovers_balanced_translation_after_update() {
        let scenario = scenario();
        let mut frozen = MeanCenterEdgeModel::frozen();
        let mut periodic = MeanCenterEdgeModel::periodic(1, 32, PROBABILITY_SCALE).unwrap();
        let frozen_result = run_edge_adaptation_episode(scenario, candidate_digest(1), &mut frozen).unwrap();
        let periodic_result = run_edge_adaptation_episode(scenario, candidate_digest(2), &mut periodic).unwrap();
        assert!(periodic_result.accuracy_ppm > frozen_result.accuracy_ppm);
        assert!(periodic_result.balanced_accuracy_present_classes_ppm
            > frozen_result.balanced_accuracy_present_classes_ppm);
        assert_eq!(periodic_result.updates_applied_count, 1);
        assert!(periodic_result.persistent_state_write_bytes >= ADAPTER_STATE_WRITE_BYTES);
        validate_edge_adaptation_receipt(&periodic_result).unwrap();
    }

    #[test]
    fn class_composition_shift_can_trigger_a_harmful_mean_update() {
        let mut scenario = scenario();
        scenario.shift = EdgeDomainShift::IDENTITY;
        scenario.class0_probability_ppm = PROBABILITY_SCALE;
        let mut frozen = MeanCenterEdgeModel::frozen();
        let mut periodic = MeanCenterEdgeModel::periodic(1, 32, PROBABILITY_SCALE).unwrap();
        let frozen_result = run_edge_adaptation_episode(scenario, candidate_digest(3), &mut frozen).unwrap();
        let periodic_result = run_edge_adaptation_episode(scenario, candidate_digest(4), &mut periodic).unwrap();
        assert_eq!(frozen_result.accuracy_ppm, PROBABILITY_SCALE);
        assert!(periodic_result.accuracy_ppm < frozen_result.accuracy_ppm);
        assert_eq!(periodic_result.updates_applied_count, 1);
    }

    #[test]
    fn candidate_interface_exposes_unlabeled_features_only() {
        fn candidate_api_has_no_label_argument<M: UnsupervisedEdgeModel>(
            model: &M,
            batch: &PublicTargetBatch,
        ) {
            let _ = model.predict(&batch.samples[0]);
        }
        let mut model = MeanCenterEdgeModel::frozen();
        let hidden = make_hidden_batch(&scenario(), 0);
        candidate_api_has_no_label_argument(&model, &hidden.public);
        assert_eq!(hidden.labels.len(), hidden.public.samples.len());
    }

    #[test]
    fn feature_generation_is_paired_across_candidate_policies() {
        let scenario = scenario();
        let first = make_hidden_batch(&scenario, 7);
        let second = make_hidden_batch(&scenario, 7);
        assert_eq!(first.public, second.public);
        assert_eq!(first.labels, second.labels);
    }

    #[test]
    fn prediction_is_scored_before_the_same_batch_adaptation_update() {
        let mut scenario = scenario();
        scenario.class0_probability_ppm = PROBABILITY_SCALE;
        scenario.shift = EdgeDomainShift::IDENTITY;
        let mut model = MeanCenterEdgeModel::periodic(1, 1, PROBABILITY_SCALE).unwrap();
        let result = run_edge_adaptation_episode(scenario, candidate_digest(5), &mut model).unwrap();
        let first_batch = &result.batch_metrics[0];
        assert_eq!(first_batch.accuracy_ppm, PROBABILITY_SCALE);
        assert_eq!(result.updates_applied_count, 1);
        assert!(result.batch_metrics[1].accuracy_ppm < PROBABILITY_SCALE);
    }

    #[test]
    fn rotation_scale_and_compound_shifts_have_distinct_commitments() {
        let mut base = scenario();
        base.shift = EdgeDomainShift::IDENTITY;
        let mut rotated = base;
        rotated.shift.rotation_quadrants = 1;
        let mut scaled = base;
        scaled.shift.scale_ppm = 1_500_000;
        let mut compound = base;
        compound.shift.rotation_quadrants = 1;
        compound.shift.scale_ppm = 1_500_000;
        compound.shift.bias_y_q15 = 10_000;
        compound.shift.noise_amplitude_q15 = 1_000;
        assert_ne!(scenario_digest(&base), scenario_digest(&rotated));
        assert_ne!(scenario_digest(&base), scenario_digest(&scaled));
        assert_ne!(scenario_digest(&rotated), scenario_digest(&compound));
    }

    #[test]
    fn shift_triggered_adapter_declines_below_threshold() {
        let mut scenario = scenario();
        scenario.shift = EdgeDomainShift::IDENTITY;
        let mut model = MeanCenterEdgeModel::shift_triggered(5_000, 32, PROBABILITY_SCALE).unwrap();
        let result = run_edge_adaptation_episode(scenario, candidate_digest(6), &mut model).unwrap();
        assert_eq!(result.updates_applied_count, 0);
        assert_eq!(result.persistent_state_write_bytes, 0);
    }

    #[test]
    fn update_state_chain_and_result_digest_reject_tampering() {
        let mut model = MeanCenterEdgeModel::periodic(1, 32, PROBABILITY_SCALE).unwrap();
        let mut receipt = run_edge_adaptation_episode(scenario(), candidate_digest(7), &mut model).unwrap();
        validate_edge_adaptation_receipt(&receipt).unwrap();
        receipt.update_receipts[0].next_state_sha256[0] ^= 1;
        assert!(validate_edge_adaptation_receipt(&receipt).is_err());
    }

    #[test]
    fn invalid_scenario_or_model_parameters_fail_closed() {
        let mut invalid = scenario();
        invalid.shift.rotation_quadrants = 4;
        assert!(invalid.validate().is_err());
        assert!(MeanCenterEdgeModel::periodic(0, 32, PROBABILITY_SCALE).is_err());
        assert!(MeanCenterEdgeModel::shift_triggered(0, 32, PROBABILITY_SCALE).is_err());
    }

    #[test]
    fn absent_class_is_reported_as_undefined_not_as_zero_support_recall() {
        let mut scenario = scenario();
        scenario.class0_probability_ppm = 0;
        scenario.shift = EdgeDomainShift::IDENTITY;
        let mut frozen = MeanCenterEdgeModel::frozen();
        let result = run_edge_adaptation_episode(scenario, candidate_digest(8), &mut frozen).unwrap();
        assert_eq!(result.class0_support, 0);
        assert_eq!(result.class0_recall_ppm, None);
        assert_eq!(result.class1_recall_ppm, Some(PROBABILITY_SCALE));
        assert_eq!(result.balanced_accuracy_present_classes_ppm, Some(PROBABILITY_SCALE));
    }

    #[test]
    fn update_costs_are_explicit_and_reproducible() {
        let scenario = scenario();
        let mut frozen = MeanCenterEdgeModel::frozen();
        let mut periodic = MeanCenterEdgeModel::periodic(1, 32, PROBABILITY_SCALE).unwrap();
        let no_adapt = run_edge_adaptation_episode(scenario, candidate_digest(9), &mut frozen).unwrap();
        let adapted = run_edge_adaptation_episode(scenario, candidate_digest(10), &mut periodic).unwrap();
        assert_eq!(no_adapt.compute_operations_estimate, 0);
        assert!(adapted.compute_operations_estimate > 0);
        assert!(adapted.persistent_state_write_bytes > no_adapt.persistent_state_write_bytes);
    }
}
