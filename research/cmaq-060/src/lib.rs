#![forbid(unsafe_code)]

//! Compute-sparse predictable expert selection for direct conditional paired
//! equality. Exactly one expert is selected before current response values,
//! only its feature map is evaluated, and its single positive factor updates
//! one global e-process. The selector is an optimization policy; validity
//! comes from the selected expert's conditional e-factor.

use std::collections::HashSet;

use cmaq_048_fourier_kernel_approachability::{FourierFeatureMap, KernelError};
use cmaq_053_compositional_risk_budget::Rational;
use cmaq_054_evidence_adaptive_kernel_family::KernelExpertConfig;
use cmaq_056_typed_null_manifest::{KernelFamilyManifest, ManifestError, NullCalibration, TypedRiskPlan};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

const POLICY_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-060/SPARSE-PREDICTABLE-EXPERT/V1\0";
const TRACE_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-060/TRACE/V1\0";
const MAX_EXPERTS: usize = 32;
const MAX_FEATURES_PER_EXPERT: usize = 1_024;
const MAX_TOTAL_FEATURES: usize = 8_192;
const MAX_PAIR_IDS: usize = 1_000_000;
const MAX_ID_BYTES: usize = 256;
const NORM_TOLERANCE: f64 = 1e-12;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SparseBetError {
    ReferenceCalibrationUnsupported,
    EmptyExpertInventory,
    TooManyExperts,
    ExpertInventoryMismatch,
    DuplicateExpertId,
    InvalidExpertId,
    InvalidBettingRate,
    InvalidExplorationRate,
    InvalidHorizon,
    HorizonExceeded,
    InvalidPairId,
    PairIdTooLong,
    DuplicatePairId,
    PairIdInventoryFull,
    SequenceMismatch,
    AlreadyQuarantined,
    PreparedStateMismatch,
    InvalidInputDimension,
    NonFiniteValue,
    FeatureDimensionOverflow,
    ExpertFeatureLimitExceeded,
    TotalFeatureBudgetExceeded,
    ObservationNormExceeded,
    InvalidSelectionProbability,
    InvalidFactor,
    ArithmeticOverflow,
    Manifest(ManifestError),
    FeatureMap(KernelError),
}

impl From<ManifestError> for SparseBetError {
    fn from(value: ManifestError) -> Self { Self::Manifest(value) }
}
impl From<KernelError> for SparseBetError {
    fn from(value: KernelError) -> Self { Self::FeatureMap(value) }
}
impl std::fmt::Display for SparseBetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for SparseBetError {}

#[derive(Clone, Debug)]
struct SparseExpert {
    id: String,
    response_map: FourierFeatureMap,
    covariate_map: FourierFeatureMap,
    response_sha256: String,
    covariate_sha256: String,
    feature_sum: Vec<f64>,
    selected_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseSelectionUpdate {
    pub sequence: u64,
    pub pair_id: String,
    pub selected_expert_id: String,
    pub selected_expert_index: usize,
    pub selection_probability: f64,
    pub selection_probabilities: Vec<f64>,
    pub random_draw: f64,
    pub direction_norm: f64,
    pub projected_feature_difference: f64,
    pub betting_rate: f64,
    pub factor: f64,
    pub log_factor: f64,
    pub normalized_bandit_reward: f64,
    pub log_e_value: f64,
    pub log_threshold: f64,
    pub quarantined: bool,
    pub feature_coordinates_evaluated: usize,
    pub policy_sha256: String,
    pub policy_trace_sha256: String,
}

#[derive(Debug)]
pub struct PreparedSparseUpdate {
    update: SparseSelectionUpdate,
    base_sequence: u64,
    base_trace_sha256: String,
    pair_id: String,
    next_feature_sum: Vec<f64>,
    next_selected_count: u64,
    next_log_weights: Vec<f64>,
    next_rng: ChaCha20Rng,
    next_trace_sha256: String,
}
impl PreparedSparseUpdate {
    pub fn update(&self) -> &SparseSelectionUpdate { &self.update }
}

/// One-process bandit-selected conditional e-monitor.
///
/// The selector uses EXP3-style exponential weights with explicit uniform
/// exploration. It learns from the bounded normalized log-factor reward, but
/// the e-process remains valid independently of any regret guarantee. Each
/// expert's direction/history is updated only when that expert is selected.
#[derive(Clone, Debug)]
pub struct SparsePredictableExpertMonitor {
    experts: Vec<SparseExpert>,
    alpha_test: Rational,
    betting_rate: f64,
    exploration_rate: f64,
    learning_rate: f64,
    horizon_cap: u64,
    log_threshold: f64,
    log_weights: Vec<f64>,
    selector_rng: ChaCha20Rng,
    policy_sha256: String,
    policy_trace_sha256: String,
    observations_seen: u64,
    pair_ids: HashSet<String>,
    log_e_value: f64,
    quarantined: bool,
}

impl SparsePredictableExpertMonitor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        risk_plan: &TypedRiskPlan,
        manifest: &KernelFamilyManifest,
        configurations: &[KernelExpertConfig],
        betting_rate: f64,
        exploration_rate: f64,
        horizon_cap: u64,
        selector_seed: [u8; 32],
    ) -> Result<Self, SparseBetError> {
        if configurations.is_empty() { return Err(SparseBetError::EmptyExpertInventory); }
        if configurations.len() > MAX_EXPERTS { return Err(SparseBetError::TooManyExperts); }
        if !betting_rate.is_finite() || betting_rate <= 0.0 || betting_rate >= 1.0 {
            return Err(SparseBetError::InvalidBettingRate);
        }
        if !exploration_rate.is_finite() || exploration_rate <= 0.0 || exploration_rate >= 1.0 {
            return Err(SparseBetError::InvalidExplorationRate);
        }
        if horizon_cap == 0 { return Err(SparseBetError::InvalidHorizon); }
        if risk_plan.reference_failure_budget().is_some()
            || !matches!(manifest.calibration(), NullCalibration::DirectConditionalPairedEquality { .. }) {
            return Err(SparseBetError::ReferenceCalibrationUnsupported);
        }
        risk_plan.verify_manifest(manifest)?;
        risk_plan.verify_expert_inventory(configurations)?;

        let registered = manifest.representation_weights();
        if registered.len() != configurations.len() {
            return Err(SparseBetError::ExpertInventoryMismatch);
        }

        let mut experts = Vec::with_capacity(configurations.len());
        let mut ids = HashSet::with_capacity(configurations.len());
        let mut total_features = 0usize;
        let mut log_weights = Vec::with_capacity(configurations.len());

        for (config, prior) in configurations.iter().zip(registered) {
            if config.expert_id.trim().is_empty() || config.expert_id.len() > MAX_ID_BYTES {
                return Err(SparseBetError::InvalidExpertId);
            }
            if !ids.insert(config.expert_id.clone()) {
                return Err(SparseBetError::DuplicateExpertId);
            }
            if config.expert_id != prior.member_id || config.prior_weight != prior.weight {
                return Err(SparseBetError::ExpertInventoryMismatch);
            }
            let response_dimension = config.response_map.feature_dimension();
            let covariate_dimension = config.covariate_map.feature_dimension();
            let dimension = response_dimension.checked_mul(covariate_dimension)
                .ok_or(SparseBetError::FeatureDimensionOverflow)?;
            if dimension > MAX_FEATURES_PER_EXPERT {
                return Err(SparseBetError::ExpertFeatureLimitExceeded);
            }
            total_features = total_features.checked_add(dimension)
                .ok_or(SparseBetError::FeatureDimensionOverflow)?;
            if total_features > MAX_TOTAL_FEATURES {
                return Err(SparseBetError::TotalFeatureBudgetExceeded);
            }
            log_weights.push(rational_log(prior.weight)?);
            experts.push(SparseExpert {
                id: config.expert_id.clone(),
                response_map: config.response_map.clone(),
                covariate_map: config.covariate_map.clone(),
                response_sha256: config.response_map.map_sha256().to_string(),
                covariate_sha256: config.covariate_map.map_sha256().to_string(),
                feature_sum: vec![0.0; dimension],
                selected_count: 0,
            });
        }

        let alpha_test = risk_plan.alpha_test();
        let log_threshold = rational_log_inverse(alpha_test)?;
        let learning_rate = if experts.len() <= 1 {
            0.0
        } else {
            (2.0 * (experts.len() as f64).ln() / (horizon_cap as f64 * experts.len() as f64)).sqrt()
        };
        if !learning_rate.is_finite() {
            return Err(SparseBetError::ArithmeticOverflow);
        }

        let policy_sha256 = policy_digest(
            &manifest.digests().portfolio_inventory_sha256,
            alpha_test,
            betting_rate,
            exploration_rate,
            horizon_cap,
            &selector_seed,
            &log_weights,
        );
        let policy_trace_sha256 = policy_sha256.clone();

        Ok(Self {
            experts,
            alpha_test,
            betting_rate,
            exploration_rate,
            learning_rate,
            horizon_cap,
            log_threshold,
            log_weights,
            selector_rng: ChaCha20Rng::from_seed(selector_seed),
            policy_sha256,
            policy_trace_sha256,
            observations_seen: 0,
            pair_ids: HashSet::new(),
            log_e_value: 0.0,
            quarantined: false,
        })
    }

    /// Prepares one observation. Selection probabilities and the random draw
    /// are computed before any current X/Y transformation. Only the chosen
    /// expert's maps are evaluated.
    pub fn prepare_pair(
        &self,
        sequence: u64,
        pair_id: &str,
        covariate: &[f64],
        x: &[f64],
        y: &[f64],
    ) -> Result<PreparedSparseUpdate, SparseBetError> {
        if self.quarantined { return Err(SparseBetError::AlreadyQuarantined); }
        if sequence > self.horizon_cap { return Err(SparseBetError::HorizonExceeded); }
        if pair_id.trim().is_empty() { return Err(SparseBetError::InvalidPairId); }
        if pair_id.len() > MAX_ID_BYTES { return Err(SparseBetError::PairIdTooLong); }
        if self.pair_ids.contains(pair_id) { return Err(SparseBetError::DuplicatePairId); }
        if self.pair_ids.len() >= MAX_PAIR_IDS { return Err(SparseBetError::PairIdInventoryFull); }
        let expected = self.observations_seen.checked_add(1).ok_or(SparseBetError::ArithmeticOverflow)?;
        if sequence != expected { return Err(SparseBetError::SequenceMismatch); }

        // Selector randomness and probability vector depend only on the past
        // and the registered selector seed, never on current response values.
        let probabilities = self.selection_probabilities()?;
        let mut next_rng = self.selector_rng.clone();
        let random_draw: f64 = next_rng.random();
        let selected_index = sample_index(&probabilities, random_draw)?;
        let probability = probabilities[selected_index];
        if probability <= 0.0 || !probability.is_finite() {
            return Err(SparseBetError::InvalidSelectionProbability);
        }

        let expert = &self.experts[selected_index];
        if covariate.len() != expert.covariate_map.input_dimension()
            || x.len() != expert.response_map.input_dimension()
            || y.len() != expert.response_map.input_dimension()
            || x.is_empty()
            || covariate.is_empty()
        {
            return Err(SparseBetError::InvalidInputDimension);
        }
        if covariate.iter().chain(x).chain(y).any(|v| !v.is_finite()) {
            return Err(SparseBetError::NonFiniteValue);
        }

        // Direction is formed only from this expert's selected-subsequence history.
        let direction_norm = l2_norm(&expert.feature_sum);
        if !direction_norm.is_finite() { return Err(SparseBetError::ArithmeticOverflow); }
        let direction = if direction_norm > 0.0 {
            expert.feature_sum.iter().map(|v| *v / direction_norm).collect::<Vec<_>>()
        } else {
            vec![0.0; expert.feature_sum.len()]
        };

        // Only the selected expert's response/covariate transforms are evaluated.
        let response_x = expert.response_map.transform(x)?;
        let response_y = expert.response_map.transform(y)?;
        let context_features = expert.covariate_map.transform(covariate)?;
        let response_dimension = expert.response_map.feature_dimension();
        let covariate_dimension = expert.covariate_map.feature_dimension();
        let feature_dimension = response_dimension.checked_mul(covariate_dimension)
            .ok_or(SparseBetError::FeatureDimensionOverflow)?;
        let mut difference = vec![0.0; feature_dimension];

        for (z_index, z_feature) in context_features.iter().enumerate() {
            for (response_index, (y_feature, x_feature)) in response_y.iter().zip(&response_x).enumerate() {
                let index = z_index * response_dimension + response_index;
                difference[index] = 0.5 * z_feature * (y_feature - x_feature);
            }
        }

        let difference_norm = l2_norm(&difference);
        if !difference_norm.is_finite() { return Err(SparseBetError::ArithmeticOverflow); }
        if difference_norm > 1.0 + NORM_TOLERANCE {
            return Err(SparseBetError::ObservationNormExceeded);
        }
        let projected = dot(&direction, &difference)?;
        if !projected.is_finite() || !(-1.0-NORM_TOLERANCE..=1.0+NORM_TOLERANCE).contains(&projected) {
            return Err(SparseBetError::ArithmeticOverflow);
        }

        let factor = 1.0 + self.betting_rate * projected;
        if !factor.is_finite() || factor <= 0.0 { return Err(SparseBetError::InvalidFactor); }
        let log_factor = factor.ln();
        let next_log_e = self.log_e_value + log_factor;
        if !next_log_e.is_finite() { return Err(SparseBetError::ArithmeticOverflow); }

        // Normalize bounded log-factor reward to [0,1] for EXP3. This changes
        // selector optimization only; the evidence process still multiplies
        // the actual factor and spends alpha exactly once.
        let lower = (1.0 - self.betting_rate).ln();
        let upper = (1.0 + self.betting_rate).ln();
        let reward = ((log_factor - lower) / (upper - lower)).clamp(0.0, 1.0);

        let mut next_log_weights = self.log_weights.clone();
        if self.experts.len() > 1 {
            let estimate = reward / probability;
            next_log_weights[selected_index] += self.learning_rate * estimate;
            normalize_log_weights(&mut next_log_weights)?;
        }

        let mut next_feature_sum = expert.feature_sum.clone();
        for (sum, value) in next_feature_sum.iter_mut().zip(&difference) {
            *sum += value;
            if !sum.is_finite() { return Err(SparseBetError::ArithmeticOverflow); }
        }
        let next_selected_count = expert.selected_count.checked_add(1)
            .ok_or(SparseBetError::ArithmeticOverflow)?;
        let quarantined = next_log_e >= self.log_threshold;
        let trace = extend_trace(
            &self.policy_trace_sha256,
            sequence,
            pair_id,
            selected_index,
            &expert.response_sha256,
            &expert.covariate_sha256,
            probability,
            random_draw,
            projected,
            factor,
            reward,
            &next_log_weights,
        );

        let update = SparseSelectionUpdate {
            sequence,
            pair_id: pair_id.to_string(),
            selected_expert_id: expert.id.clone(),
            selected_expert_index: selected_index,
            selection_probability: probability,
            selection_probabilities: probabilities,
            random_draw,
            direction_norm,
            projected_feature_difference: projected,
            betting_rate: self.betting_rate,
            factor,
            log_factor,
            normalized_bandit_reward: reward,
            log_e_value: next_log_e,
            log_threshold: self.log_threshold,
            quarantined,
            feature_coordinates_evaluated: response_x.len() + response_y.len()
                + context_features.len() + difference.len(),
            policy_sha256: self.policy_sha256.clone(),
            policy_trace_sha256: trace.clone(),
        };

        Ok(PreparedSparseUpdate {
            update,
            base_sequence: self.observations_seen,
            base_trace_sha256: self.policy_trace_sha256.clone(),
            pair_id: pair_id.to_string(),
            next_feature_sum,
            next_selected_count,
            next_log_weights,
            next_rng,
            next_trace_sha256: trace,
        })
    }

    /// Commits the selected factor, selected-expert history, selector update
    /// and replay ID in one transaction after the parent is ready.
    pub fn commit_prepared(
        &mut self,
        prepared: PreparedSparseUpdate,
    ) -> Result<SparseSelectionUpdate, SparseBetError> {
        if self.quarantined { return Err(SparseBetError::AlreadyQuarantined); }
        if self.observations_seen != prepared.base_sequence
            || self.policy_trace_sha256 != prepared.base_trace_sha256
            || prepared.update.policy_sha256 != self.policy_sha256
            || prepared.update.pair_id != prepared.pair_id
            || prepared.update.sequence != self.observations_seen.checked_add(1).ok_or(SparseBetError::ArithmeticOverflow)?
        {
            return Err(SparseBetError::PreparedStateMismatch);
        }
        if self.pair_ids.contains(&prepared.pair_id) { return Err(SparseBetError::DuplicatePairId); }
        let index = prepared.update.selected_expert_index;
        if index >= self.experts.len()
            || self.experts[index].id != prepared.update.selected_expert_id
            || prepared.next_feature_sum.len() != self.experts[index].feature_sum.len()
            || prepared.next_log_weights.len() != self.experts.len()
        {
            return Err(SparseBetError::PreparedStateMismatch);
        }

        self.experts[index].feature_sum = prepared.next_feature_sum;
        self.experts[index].selected_count = prepared.next_selected_count;
        self.log_weights = prepared.next_log_weights;
        self.selector_rng = prepared.next_rng;
        self.observations_seen = prepared.update.sequence;
        self.pair_ids.insert(prepared.pair_id);
        self.log_e_value = prepared.update.log_e_value;
        self.quarantined = prepared.update.quarantined;
        self.policy_trace_sha256 = prepared.next_trace_sha256;
        Ok(prepared.update)
    }

    pub fn observe_pair(
        &mut self, sequence: u64, pair_id: &str, covariate: &[f64], x: &[f64], y: &[f64],
    ) -> Result<SparseSelectionUpdate, SparseBetError> {
        let prepared = self.prepare_pair(sequence, pair_id, covariate, x, y)?;
        self.commit_prepared(prepared)
    }

    fn selection_probabilities(&self) -> Result<Vec<f64>, SparseBetError> {
        let count = self.log_weights.len();
        let max = self.log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !max.is_finite() { return Err(SparseBetError::ArithmeticOverflow); }
        let mut weights = self.log_weights.iter().map(|x| (*x - max).exp()).collect::<Vec<_>>();
        let sum = weights.iter().sum::<f64>();
        if !sum.is_finite() || sum <= 0.0 { return Err(SparseBetError::ArithmeticOverflow); }
        for weight in &mut weights { *weight /= sum; }
        let mut probabilities = weights.iter()
            .map(|weight| (1.0 - self.exploration_rate) * weight + self.exploration_rate / count as f64)
            .collect::<Vec<_>>();
        let total = probabilities.iter().sum::<f64>();
        if !total.is_finite() || total <= 0.0 { return Err(SparseBetError::InvalidSelectionProbability); }
        for probability in &mut probabilities { *probability /= total; }
        Ok(probabilities)
    }

    pub fn observations_seen(&self) -> u64 { self.observations_seen }
    pub fn log_e_value(&self) -> f64 { self.log_e_value }
    pub fn log_threshold(&self) -> f64 { self.log_threshold }
    pub fn quarantined(&self) -> bool { self.quarantined }
    pub fn selected_counts(&self) -> Vec<u64> { self.experts.iter().map(|e| e.selected_count).collect() }
    pub fn policy_sha256(&self) -> &str { &self.policy_sha256 }
    pub fn policy_trace_sha256(&self) -> &str { &self.policy_trace_sha256 }
    pub fn alpha_test(&self) -> Rational { self.alpha_test }
}

fn sample_index(probabilities: &[f64], draw: f64) -> Result<usize, SparseBetError> {
    if probabilities.is_empty() || !draw.is_finite() || !(0.0..1.0).contains(&draw) {
        return Err(SparseBetError::InvalidSelectionProbability);
    }
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if draw < cumulative { return Ok(index); }
    }
    Ok(probabilities.len() - 1)
}

fn normalize_log_weights(log_weights: &mut [f64]) -> Result<(), SparseBetError> {
    let max = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() { return Err(SparseBetError::ArithmeticOverflow); }
    let normalizer = max + log_weights.iter().map(|x| (*x - max).exp()).sum::<f64>().ln();
    if !normalizer.is_finite() { return Err(SparseBetError::ArithmeticOverflow); }
    for weight in log_weights {
        *weight -= normalizer;
        if !weight.is_finite() { return Err(SparseBetError::ArithmeticOverflow); }
    }
    Ok(())
}

fn rational_log(value: Rational) -> Result<f64, SparseBetError> {
    if value.numerator() == 0 || value.denominator() == 0 { return Err(SparseBetError::ArithmeticOverflow); }
    let result = (value.numerator() as f64).ln() - (value.denominator() as f64).ln();
    if result.is_finite() { Ok(result) } else { Err(SparseBetError::ArithmeticOverflow) }
}
fn rational_log_inverse(value: Rational) -> Result<f64, SparseBetError> {
    if value.numerator() == 0 || value.denominator() == 0 { return Err(SparseBetError::ArithmeticOverflow); }
    let result = (value.denominator() as f64).ln() - (value.numerator() as f64).ln();
    if result.is_finite() { Ok(result) } else { Err(SparseBetError::ArithmeticOverflow) }
}
fn l2_norm(values: &[f64]) -> f64 { values.iter().map(|v| v * v).sum::<f64>().sqrt() }
fn dot(left: &[f64], right: &[f64]) -> Result<f64, SparseBetError> {
    if left.len() != right.len() { return Err(SparseBetError::InvalidInputDimension); }
    let value = left.iter().zip(right).map(|(a,b)| a*b).sum::<f64>();
    if value.is_finite() { Ok(value) } else { Err(SparseBetError::ArithmeticOverflow) }
}

fn policy_digest(
    manifest: &str, alpha: Rational, lambda: f64, gamma: f64, horizon: u64,
    seed: &[u8; 32], initial_weights: &[f64],
) -> String {
    let mut h = Sha256::new();
    h.update(POLICY_DOMAIN); h.update(manifest.as_bytes());
    h.update(alpha.numerator().to_be_bytes()); h.update(alpha.denominator().to_be_bytes());
    h.update(lambda.to_bits().to_be_bytes()); h.update(gamma.to_bits().to_be_bytes());
    h.update(horizon.to_be_bytes()); h.update(seed);
    for weight in initial_weights { h.update(weight.to_bits().to_be_bytes()); }
    hex(&h.finalize())
}

fn extend_trace(
    prior: &str,
    sequence: u64,
    pair_id: &str,
    selected: usize,
    response_map_sha256: &str,
    covariate_map_sha256: &str,
    probability: f64,
    draw: f64,
    projected: f64,
    factor: f64,
    reward: f64,
    next_log_weights: &[f64],
) -> String {
    let mut h = Sha256::new();
    h.update(TRACE_DOMAIN);
    h.update(prior.as_bytes());
    h.update(sequence.to_be_bytes());
    h.update((pair_id.len() as u64).to_be_bytes());
    h.update(pair_id.as_bytes());
    h.update((selected as u64).to_be_bytes());
    h.update(response_map_sha256.as_bytes());
    h.update(covariate_map_sha256.as_bytes());
    h.update(probability.to_bits().to_be_bytes());
    h.update(draw.to_bits().to_be_bytes());
    h.update(projected.to_bits().to_be_bytes());
    h.update(factor.to_bits().to_be_bytes());
    h.update(reward.to_bits().to_be_bytes());
    for weight in next_log_weights {
        h.update(weight.to_bits().to_be_bytes());
    }
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len()*2);
    for byte in bytes {
        result.push(DIGITS[(byte>>4) as usize] as char);
        result.push(DIGITS[(byte&15) as usize] as char);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmaq_046_convex_target_approachability::BoxTarget;
    use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
    use cmaq_053_compositional_risk_budget::{FleetBudgetPlan, FleetScope};

    const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const HASH_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const HASH_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const HASH_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const HASH_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn config(id: &str, response: f64, map: &str, cov: &str, target: &str) -> KernelExpertConfig {
        KernelExpertConfig {
            expert_id: id.into(),
            response_map: FourierFeatureMap::new(1, vec![vec![response]], map.into()).unwrap(),
            covariate_map: FourierFeatureMap::new(1, vec![vec![0.5]], cov.into()).unwrap(),
            target: BoxTarget::new(vec![-0.01; 4], vec![0.01; 4]).unwrap(),
            prior_weight: Rational::new(1, 2).unwrap(),
            target_sha256: target.into(),
            response_reference_sha256: HASH_E.into(),
            covariate_reference_sha256: HASH_F.into(),
        }
    }

    fn configs() -> Vec<KernelExpertConfig> {
        vec![
            config("low", 0.5, HASH_A, HASH_B, HASH_C),
            config("high", 1.0, HASH_D, HASH_E, HASH_F),
        ]
    }

    fn single_monitor(seed: [u8; 32]) -> SparsePredictableExpertMonitor {
        let mut only = configs().remove(0);
        only.prior_weight = Rational::one();
        let configurations = vec![only];
        let manifest = KernelFamilyManifest::build(&configurations, NullCalibration::direct_v1()).unwrap();
        let fleet = FleetBudgetPlan::new(Rational::probability(1, 20).unwrap(), HASH_A.into(), HASH_B.into()).unwrap();
        let scope = FleetScope {
            stream_id: "sparse-single-test".into(),
            asset_id: "asset".into(),
            evidence_channel_id: "paired".into(),
            asset_index: 1,
            evidence_channel_index: 1,
            deployment_index: 1,
        };
        let local = fleet.local_alpha_for(&scope).unwrap();
        let plan = TypedRiskPlan::new(&fleet, scope, local.divided_by_integer(2).unwrap(), &manifest).unwrap();
        SparsePredictableExpertMonitor::new(&plan, &manifest, &configurations, 0.5, 0.1, 256, seed).unwrap()
    }

    fn monitor(seed: [u8;32]) -> SparsePredictableExpertMonitor {
        let configurations = configs();
        let manifest = KernelFamilyManifest::build(&configurations, NullCalibration::direct_v1()).unwrap();
        let fleet = FleetBudgetPlan::new(Rational::probability(1,20).unwrap(), HASH_A.into(), HASH_B.into()).unwrap();
        let scope = FleetScope {
            stream_id: "sparse-test".into(),
            asset_id: "asset".into(),
            evidence_channel_id: "paired".into(),
            asset_index: 1,
            evidence_channel_index: 1,
            deployment_index: 1,
        };
        let local = fleet.local_alpha_for(&scope).unwrap();
        let plan = TypedRiskPlan::new(&fleet, scope, local.divided_by_integer(2).unwrap(), &manifest).unwrap();
        SparsePredictableExpertMonitor::new(&plan, &manifest, &configurations, 0.5, 0.1, 256, seed).unwrap()
    }

    #[test]
    fn equal_pairs_preserve_unit_evidence_and_selector_runs() {
        let mut m = monitor([7u8;32]);
        for t in 1..=100u64 {
            let u = m.observe_pair(t, &format!("equal-{t}"), &[0.2], &[0.25], &[0.25]).unwrap();
            assert!((u.factor - 1.0).abs() < 1e-12);
            assert!(u.log_e_value.abs() < 1e-10);
            assert!((u.selection_probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(u.selection_probabilities.iter().all(|p| *p >= 0.05 - 1e-12));
            assert_eq!(u.feature_coordinates_evaluated, 10);
        }
        assert_eq!(m.observations_seen(), 100);
        assert_eq!(m.selected_counts().iter().sum::<u64>(), 100);
    }

    #[test]
    fn current_pair_does_not_change_pre_pair_selection_or_direction() {
        let mut baseline = single_monitor([9u8; 32]);
        baseline.observe_pair(1, "history", &[0.1], &[0.0], &[1.0]).unwrap();
        let mut left = baseline.clone();
        let mut right = baseline;
        let a = left.prepare_pair(2, "left", &[0.25], &[0.1], &[0.2]).unwrap();
        let b = right.prepare_pair(2, "right", &[0.25], &[2.0], &[-1.0]).unwrap();
        assert_eq!(a.update.selected_expert_index, b.update.selected_expert_index);
        assert_eq!(a.update.selection_probability, b.update.selection_probability);
        assert_eq!(a.update.random_draw, b.update.random_draw);
        assert!(a.update.direction_norm > 0.0);
        assert_eq!(a.update.direction_norm, b.update.direction_norm);
        // The selected action is invariant to the current pair. Different
        // pairs may or may not project differently onto this one-dimensional
        // direction, so the selection/direction checks are the negative control.
        assert_eq!(left.observations_seen(), 1);
        assert_eq!(right.observations_seen(), 1);
    }

    #[test]
    fn prepare_is_non_mutating_and_stale_transaction_rejected() {
        let mut m = monitor([11u8;32]);
        let stale = m.prepare_pair(1, "stale", &[0.0], &[0.0], &[0.1]).unwrap();
        m.observe_pair(1, "first", &[0.0], &[0.0], &[0.0]).unwrap();
        let before = m.observations_seen();
        assert_eq!(m.commit_prepared(stale), Err(SparseBetError::PreparedStateMismatch));
        assert_eq!(m.observations_seen(), before);
    }

    #[test]
    fn replay_and_out_of_order_pairs_fail_without_partial_commit() {
        let mut m = monitor([13u8;32]);
        assert_eq!(m.observe_pair(2, "late", &[0.0], &[0.0], &[0.0]), Err(SparseBetError::SequenceMismatch));
        m.observe_pair(1, "once", &[0.0], &[0.0], &[0.0]).unwrap();
        assert_eq!(m.observe_pair(2, "once", &[0.0], &[0.0], &[0.0]), Err(SparseBetError::DuplicatePairId));
        assert_eq!(m.observations_seen(), 1);
    }
}
