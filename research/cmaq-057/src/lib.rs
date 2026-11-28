#![forbid(unsafe_code)]

//! Prequential adaptive kernel betting for direct conditional paired equality.
//!
//! Exactly one representation and one betting direction are selected from
//! history before each current pair is transformed. Only after the single
//! e-process increment is computed are all candidate feature summaries
//! updated for use at the next observation. The policy therefore adapts
//! predictably without taking a post-hoc maximum across same-time experts.

use std::collections::HashSet;

use cmaq_048_fourier_kernel_approachability::{FourierFeatureMap, KernelError};
use cmaq_053_compositional_risk_budget::Rational;
use cmaq_054_evidence_adaptive_kernel_family::KernelExpertConfig;
use cmaq_056_typed_null_manifest::{
    KernelFamilyManifest, ManifestError, NullCalibration, TypedRiskPlan,
};
use sha2::{Digest, Sha256};

const POLICY_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-057/PREDICTABLE-ADAPTIVE-BET/V1\0";
const POLICY_VERSION: &[u8] = b"lagged-largest-feature-mean-direction-v1";
const MAX_EXPERTS: usize = 32;
const MAX_EXPERT_FEATURES: usize = 1_024;
const MAX_TOTAL_FEATURES: usize = 8_192;
const MAX_PAIR_ID_BYTES: usize = 256;
const MAX_PAIR_IDS: usize = 1_000_000;
const NORM_TOLERANCE: f64 = 1e-12;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdaptiveBetError {
    ReferenceCalibrationUnsupported,
    EmptyExpertInventory,
    TooManyExperts,
    InvalidExpertId,
    DuplicateExpertId,
    InvalidBettingRate,
    InvalidPairId,
    PairIdTooLong,
    DuplicatePairId,
    PairIdInventoryFull,
    SequenceMismatch,
    AlreadyQuarantined,
    PreparedStateMismatch,
    PreparedExpertInventoryMismatch,
    InputDimensionMismatch,
    FeatureDimensionOverflow,
    ExpertFeatureLimitExceeded,
    TotalFeatureBudgetExceeded,
    ObservationNormExceeded,
    NonFiniteValue,
    ArithmeticOverflow,
    Manifest(ManifestError),
    KernelFeature(KernelError),
}

impl From<ManifestError> for AdaptiveBetError {
    fn from(value: ManifestError) -> Self {
        Self::Manifest(value)
    }
}

impl From<KernelError> for AdaptiveBetError {
    fn from(value: KernelError) -> Self {
        Self::KernelFeature(value)
    }
}

impl std::fmt::Display for AdaptiveBetError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for AdaptiveBetError {}

#[derive(Clone, Debug)]
struct AdaptiveExpert {
    expert_id: String,
    response_map: FourierFeatureMap,
    covariate_map: FourierFeatureMap,
    response_map_sha256: String,
    covariate_map_sha256: String,
    feature_sum: Vec<f64>,
    historical_score: f64,
    log_e_value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertHistoryScore {
    pub expert_id: String,
    pub historical_mean_norm: f64,
    pub feature_dimension: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredictableBetAction {
    pub sequence: u64,
    pub selected_expert_id: String,
    pub response_map_sha256: String,
    pub covariate_map_sha256: String,
    /// Direction derived solely from observations preceding this sequence.
    pub direction: Vec<f64>,
    pub direction_norm: f64,
    pub historical_sum_norm: f64,
    pub historical_score_used: f64,
    pub betting_rate: f64,
    pub projected_feature_difference: f64,
    pub betting_factor: f64,
    pub log_e_value: f64,
    pub log_threshold: f64,
    /// False for a child whose stopping decision belongs to an outer restart
    /// mixture. The log threshold remains recorded for comparison only.
    pub threshold_enforced: bool,
    pub quarantined: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertFactorUpdate {
    pub expert_id: String,
    pub response_map_sha256: String,
    pub covariate_map_sha256: String,
    /// Direction derived only from the expert's history before the pair.
    pub direction: Vec<f64>,
    pub projected_feature_difference: f64,
    pub betting_factor: f64,
    pub log_factor: f64,
    pub log_e_value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredictableAdaptiveUpdate {
    pub action: PredictableBetAction,
    /// One valid predictable factor per registered expert. These are suitable
    /// inputs to a fixed kernel-prior mixture such as CMAQ-058.
    pub expert_factors: Vec<ExpertFactorUpdate>,
    /// Candidate scores after the current pair; these are available only to
    /// choose the representation for the next observation.
    pub next_expert_history: Vec<ExpertHistoryScore>,
    pub observations_seen: u64,
    pub pair_id: String,
    pub policy_sha256: String,
    pub policy_trace_sha256: String,
}

/// Prepared, immutable single-pair policy transaction. The current
/// monitor is unchanged until commit_prepared succeeds. A base sequence and
/// trace digest prevent stale or out-of-order prepared transactions.
#[derive(Debug)]
pub struct PreparedAdaptiveUpdate {
    update: PredictableAdaptiveUpdate,
    base_sequence: u64,
    base_policy_trace_sha256: String,
    pair_id: String,
    staged_sums: Vec<(Vec<f64>, f64)>,
    next_policy_trace_sha256: String,
}

impl PreparedAdaptiveUpdate {
    pub fn update(&self) -> &PredictableAdaptiveUpdate {
        &self.update
    }
}

#[derive(Clone, Debug)]
pub struct PredictableAdaptiveKernelMonitor {
    experts: Vec<AdaptiveExpert>,
    alpha_test: Rational,
    betting_rate: f64,
    log_threshold: f64,
    stop_at_threshold: bool,
    manifest_sha256: String,
    policy_sha256: String,
    policy_trace_sha256: String,
    observations_seen: u64,
    seen_pair_ids: HashSet<String>,
    log_e_value: f64,
    quarantined: bool,
}

impl PredictableAdaptiveKernelMonitor {
    /// Constructs a direct conditional paired-equality monitor.
    ///
    /// The typed plan must be in DirectConditionalPairedEquality mode. The
    /// supplied expert configs are re-canonicalized and checked against the
    /// plan's manifest before the policy is accepted. The prior weights in
    /// those configs bind the inventory but are not used to mixture-average
    /// the e-process: only one predictable expert bet is made per observation.
    pub fn new(
        risk_plan: &TypedRiskPlan,
        manifest: &KernelFamilyManifest,
        configurations: &[KernelExpertConfig],
        betting_rate: f64,
    ) -> Result<Self, AdaptiveBetError> {
        Self::new_with_stopping(
            risk_plan,
            manifest,
            configurations,
            betting_rate,
            true,
        )
    }

    /// Constructs a child factor producer whose stopping decision belongs to
    /// a parent process (for example CMAQ-058's restart mixture). It continues
    /// to emit positive predictable factors even after its own cumulative
    /// wealth crosses 1/alpha_test. The parent must use the returned factors
    /// and make the only quarantine decision.
    pub fn new_unthresholded(
        risk_plan: &TypedRiskPlan,
        manifest: &KernelFamilyManifest,
        configurations: &[KernelExpertConfig],
        betting_rate: f64,
    ) -> Result<Self, AdaptiveBetError> {
        Self::new_with_stopping(
            risk_plan,
            manifest,
            configurations,
            betting_rate,
            false,
        )
    }

    fn new_with_stopping(
        risk_plan: &TypedRiskPlan,
        manifest: &KernelFamilyManifest,
        configurations: &[KernelExpertConfig],
        betting_rate: f64,
        stop_at_threshold: bool,
    ) -> Result<Self, AdaptiveBetError> {
        if risk_plan.reference_failure_budget().is_some()
            || !matches!(
                manifest.calibration(),
                NullCalibration::DirectConditionalPairedEquality { .. }
            )
        {
            return Err(AdaptiveBetError::ReferenceCalibrationUnsupported);
        }
        if !betting_rate.is_finite() || betting_rate <= 0.0 || betting_rate >= 1.0 {
            return Err(AdaptiveBetError::InvalidBettingRate);
        }
        if configurations.is_empty() {
            return Err(AdaptiveBetError::EmptyExpertInventory);
        }
        if configurations.len() > MAX_EXPERTS {
            return Err(AdaptiveBetError::TooManyExperts);
        }

        risk_plan.verify_manifest(manifest)?;
        risk_plan.verify_expert_inventory(configurations)?;

        let mut seen_ids = HashSet::with_capacity(configurations.len());
        let mut experts = Vec::with_capacity(configurations.len());
        let mut total_features = 0usize;

        for configuration in configurations {
            if configuration.expert_id.trim().is_empty()
                || configuration.expert_id.len() > MAX_PAIR_ID_BYTES
            {
                return Err(AdaptiveBetError::InvalidExpertId);
            }
            if !seen_ids.insert(configuration.expert_id.clone()) {
                return Err(AdaptiveBetError::DuplicateExpertId);
            }

            let response_dimension = configuration.response_map.feature_dimension();
            let covariate_dimension = configuration.covariate_map.feature_dimension();
            let feature_dimension = response_dimension
                .checked_mul(covariate_dimension)
                .ok_or(AdaptiveBetError::FeatureDimensionOverflow)?;
            if feature_dimension > MAX_EXPERT_FEATURES {
                return Err(AdaptiveBetError::ExpertFeatureLimitExceeded);
            }
            total_features = total_features
                .checked_add(feature_dimension)
                .ok_or(AdaptiveBetError::FeatureDimensionOverflow)?;
            if total_features > MAX_TOTAL_FEATURES {
                return Err(AdaptiveBetError::TotalFeatureBudgetExceeded);
            }

            experts.push(AdaptiveExpert {
                expert_id: configuration.expert_id.clone(),
                response_map: configuration.response_map.clone(),
                covariate_map: configuration.covariate_map.clone(),
                response_map_sha256: configuration.response_map.map_sha256().to_string(),
                covariate_map_sha256: configuration.covariate_map.map_sha256().to_string(),
                feature_sum: vec![0.0; feature_dimension],
                historical_score: 0.0,
                log_e_value: 0.0,
            });
        }

        let alpha_test = risk_plan.alpha_test();
        let log_threshold = rational_log_inverse(alpha_test)?;
        let manifest_sha256 = manifest.digests().portfolio_inventory_sha256.clone();
        let policy_sha256 =
            policy_digest(&manifest_sha256, alpha_test, betting_rate, stop_at_threshold);
        let policy_trace_sha256 = policy_sha256.clone();

        Ok(Self {
            experts,
            alpha_test,
            betting_rate,
            log_threshold,
            stop_at_threshold,
            manifest_sha256,
            policy_sha256,
            policy_trace_sha256,
            observations_seen: 0,
            seen_pair_ids: HashSet::new(),
            log_e_value: 0.0,
            quarantined: false,
        })
    }

    /// Processes one pair in a single atomic transaction.
    ///
    /// Crucial ordering:
    /// 1. select expert and direction from the policy state before this pair;
    /// 2. evaluate the selected representation and betting factor;
    /// 3. update every candidate's historical feature sum for future use;
    /// 4. commit wealth, policy state, sequence and replay ID together.
    ///
    /// The caller never supplies the selected expert or direction, so it
    /// cannot choose the same-observation maximum through this API.
    pub fn prepare_pair(
        &self,
        sequence: u64,
        pair_id: &str,
        covariate: &[f64],
        x: &[f64],
        y: &[f64],
    ) -> Result<PreparedAdaptiveUpdate, AdaptiveBetError> {
        if self.quarantined {
            return Err(AdaptiveBetError::AlreadyQuarantined);
        }
        if pair_id.trim().is_empty() {
            return Err(AdaptiveBetError::InvalidPairId);
        }
        if pair_id.len() > MAX_PAIR_ID_BYTES {
            return Err(AdaptiveBetError::PairIdTooLong);
        }
        if self.seen_pair_ids.contains(pair_id) {
            return Err(AdaptiveBetError::DuplicatePairId);
        }
        if self.seen_pair_ids.len() >= MAX_PAIR_IDS {
            return Err(AdaptiveBetError::PairIdInventoryFull);
        }

        let expected_sequence = self
            .observations_seen
            .checked_add(1)
            .ok_or(AdaptiveBetError::ArithmeticOverflow)?;
        if sequence != expected_sequence {
            return Err(AdaptiveBetError::SequenceMismatch);
        }

        // The active representation and direction are selected strictly
        // from historical state before any current pair transform is run.
        let selected_index = self.select_expert_from_history();
        let selected = &self.experts[selected_index];
        let historical_score_used = selected.historical_score;
        let direction_norm = l2_norm(&selected.feature_sum);
        if !direction_norm.is_finite() {
            return Err(AdaptiveBetError::ArithmeticOverflow);
        }
        let direction = if direction_norm > 0.0 {
            selected
                .feature_sum
                .iter()
                .map(|value| *value / direction_norm)
                .collect::<Vec<_>>()
        } else {
            vec![0.0; selected.feature_sum.len()]
        };

        // Stage current feature differences for all candidate maps. These
        // values do not influence the already selected map or direction.
        let mut staged_differences = Vec::with_capacity(self.experts.len());
        for expert in &self.experts {
            let response_x = expert.response_map.transform(x)?;
            let response_y = expert.response_map.transform(y)?;
            let covariate_features = expert.covariate_map.transform(covariate)?;
            let response_dimension = expert.response_map.feature_dimension();
            let covariate_dimension = expert.covariate_map.feature_dimension();
            let feature_dimension = response_dimension
                .checked_mul(covariate_dimension)
                .ok_or(AdaptiveBetError::FeatureDimensionOverflow)?;
            let mut difference = vec![0.0; feature_dimension];

            for (covariate_index, covariate_feature) in
                covariate_features.iter().enumerate()
            {
                for (response_index, (y_feature, x_feature)) in
                    response_y.iter().zip(&response_x).enumerate()
                {
                    let index = covariate_index * response_dimension + response_index;
                    difference[index] =
                        0.5 * covariate_feature * (y_feature - x_feature);
                }
            }

            let norm = l2_norm(&difference);
            if !norm.is_finite() {
                return Err(AdaptiveBetError::ArithmeticOverflow);
            }
            if norm > 1.0 + NORM_TOLERANCE {
                return Err(AdaptiveBetError::ObservationNormExceeded);
            }
            staged_differences.push(difference);
        }

        // Also produce one valid prequential factor for every registered
        // expert. Each direction is based only on that expert's historical
        // feature sum. CMAQ-058 can mix these factor streams over kernel and
        // restart priors without instantiating every start-time child.
        let mut expert_factors = Vec::with_capacity(self.experts.len());
        for (expert, difference) in self.experts.iter().zip(&staged_differences) {
            let historical_norm = l2_norm(&expert.feature_sum);
            if !historical_norm.is_finite() {
                return Err(AdaptiveBetError::ArithmeticOverflow);
            }
            let expert_direction = if historical_norm > 0.0 {
                expert
                    .feature_sum
                    .iter()
                    .map(|value| *value / historical_norm)
                    .collect::<Vec<_>>()
            } else {
                vec![0.0; expert.feature_sum.len()]
            };
            let projected_feature_difference = dot(&expert_direction, difference)?;
            if !projected_feature_difference.is_finite()
                || !(-1.0 - NORM_TOLERANCE..=1.0 + NORM_TOLERANCE)
                    .contains(&projected_feature_difference)
            {
                return Err(AdaptiveBetError::ArithmeticOverflow);
            }
            let betting_factor = 1.0 + self.betting_rate * projected_feature_difference;
            if !betting_factor.is_finite() || betting_factor <= 0.0 {
                return Err(AdaptiveBetError::ArithmeticOverflow);
            }
            let log_factor = betting_factor.ln();
            let log_e_value = expert.log_e_value + log_factor;
            if !log_e_value.is_finite() {
                return Err(AdaptiveBetError::ArithmeticOverflow);
            }
            expert_factors.push(ExpertFactorUpdate {
                expert_id: expert.expert_id.clone(),
                response_map_sha256: expert.response_map_sha256.clone(),
                covariate_map_sha256: expert.covariate_map_sha256.clone(),
                direction: expert_direction,
                projected_feature_difference,
                betting_factor,
                log_factor,
                log_e_value,
            });
        }

        let selected_difference = &staged_differences[selected_index];
        let projected = dot(&direction, selected_difference)?;
        if !projected.is_finite() || !(-1.0 - NORM_TOLERANCE..=1.0 + NORM_TOLERANCE).contains(&projected) {
            return Err(AdaptiveBetError::ArithmeticOverflow);
        }

        // The direct paired-equality target is S={0}, so the support value
        // vanishes. A strict lambda<1 keeps 1+lambda*q strictly positive.
        let betting_factor = 1.0 + self.betting_rate * projected;
        if !betting_factor.is_finite() || betting_factor <= 0.0 {
            return Err(AdaptiveBetError::ArithmeticOverflow);
        }
        let next_log_e_value = self.log_e_value + betting_factor.ln();
        if !next_log_e_value.is_finite() {
            return Err(AdaptiveBetError::ArithmeticOverflow);
        }
        let next_quarantined =
            self.stop_at_threshold && next_log_e_value >= self.log_threshold;

        // Stage all expert history for the next observation. Current feature
        // values cannot retroactively change this observation's betting action.
        let mut staged_sums = Vec::with_capacity(self.experts.len());
        let mut next_history = Vec::with_capacity(self.experts.len());
        for (expert, difference) in self.experts.iter().zip(&staged_differences) {
            let mut next_sum = expert.feature_sum.clone();
            for (sum, value) in next_sum.iter_mut().zip(difference) {
                *sum += value;
                if !sum.is_finite() {
                    return Err(AdaptiveBetError::ArithmeticOverflow);
                }
            }
            let score = l2_norm(&next_sum);
            if !score.is_finite() {
                return Err(AdaptiveBetError::ArithmeticOverflow);
            }
            next_history.push(ExpertHistoryScore {
                expert_id: expert.expert_id.clone(),
                historical_mean_norm: score / expected_sequence as f64,
                feature_dimension: next_sum.len(),
            });
            staged_sums.push((next_sum, score));
        }

        let action = PredictableBetAction {
            sequence,
            selected_expert_id: selected.expert_id.clone(),
            response_map_sha256: selected.response_map_sha256.clone(),
            covariate_map_sha256: selected.covariate_map_sha256.clone(),
            direction: direction.clone(),
            direction_norm: l2_norm(&direction),
            historical_sum_norm: direction_norm,
            historical_score_used,
            betting_rate: self.betting_rate,
            projected_feature_difference: projected,
            betting_factor,
            log_e_value: next_log_e_value,
            log_threshold: self.log_threshold,
            threshold_enforced: self.stop_at_threshold,
            quarantined: next_quarantined,
        };

        let next_trace_sha256 = extend_policy_trace(
            &self.policy_trace_sha256,
            pair_id,
            &action,
            &self.experts,
            &staged_differences,
            &expert_factors,
        );

        Ok(PreparedAdaptiveUpdate {
            update: PredictableAdaptiveUpdate {
                action,
                expert_factors,
                next_expert_history: next_history,
                observations_seen: expected_sequence,
                pair_id: pair_id.to_string(),
                policy_sha256: self.policy_sha256.clone(),
                policy_trace_sha256: next_trace_sha256.clone(),
            },
            base_sequence: self.observations_seen,
            base_policy_trace_sha256: self.policy_trace_sha256.clone(),
            pair_id: pair_id.to_string(),
            staged_sums,
            next_policy_trace_sha256: next_trace_sha256,
        })
    }

    /// Commits a previously prepared pair after verifying that no other
    /// observation changed the base policy state. All arithmetic and feature
    /// work has already succeeded, so the commit itself contains no fallible
    /// statistical operations.
    pub fn commit_prepared(
        &mut self,
        prepared: PreparedAdaptiveUpdate,
    ) -> Result<PredictableAdaptiveUpdate, AdaptiveBetError> {
        if self.quarantined {
            return Err(AdaptiveBetError::AlreadyQuarantined);
        }
        if self.observations_seen != prepared.base_sequence
            || self.policy_trace_sha256 != prepared.base_policy_trace_sha256
            || prepared.update.policy_sha256 != self.policy_sha256
            || prepared.update.pair_id != prepared.pair_id
            || prepared.update.observations_seen
                != self.observations_seen.checked_add(1)
                    .ok_or(AdaptiveBetError::ArithmeticOverflow)?
        {
            return Err(AdaptiveBetError::PreparedStateMismatch);
        }
        if self.seen_pair_ids.contains(&prepared.pair_id) {
            return Err(AdaptiveBetError::DuplicatePairId);
        }
        if self.seen_pair_ids.len() >= MAX_PAIR_IDS {
            return Err(AdaptiveBetError::PairIdInventoryFull);
        }
        if prepared.staged_sums.len() != self.experts.len()
            || prepared.update.expert_factors.len() != self.experts.len()
        {
            return Err(AdaptiveBetError::PreparedExpertInventoryMismatch);
        }

        for (index, (expert, (sum, score))) in
            self.experts.iter_mut().zip(prepared.staged_sums).enumerate()
        {
            expert.feature_sum = sum;
            expert.historical_score = score;
            expert.log_e_value = prepared.update.expert_factors[index].log_e_value;
        }
        self.observations_seen = prepared.update.observations_seen;
        self.seen_pair_ids.insert(prepared.pair_id);
        self.log_e_value = prepared.update.action.log_e_value;
        self.quarantined = prepared.update.action.quarantined;
        self.policy_trace_sha256 = prepared.next_policy_trace_sha256;

        Ok(prepared.update)
    }

    /// Standalone convenience API. Parent aggregators should call
    /// prepare_pair, validate their own state from the returned factors, then
    /// commit_prepared once the parent transaction is ready.
    pub fn observe_pair(
        &mut self,
        sequence: u64,
        pair_id: &str,
        covariate: &[f64],
        x: &[f64],
        y: &[f64],
    ) -> Result<PredictableAdaptiveUpdate, AdaptiveBetError> {
        let prepared = self.prepare_pair(sequence, pair_id, covariate, x, y)?;
        self.commit_prepared(prepared)
    }

    fn select_expert_from_history(&self) -> usize {
        let mut selected = 0usize;
        for index in 1..self.experts.len() {
            // Strict comparison retains the earliest inventory member on
            // exact ties. This decision is made before current pair values.
            if self.experts[index]
                .historical_score
                .total_cmp(&self.experts[selected].historical_score)
                .is_gt()
            {
                selected = index;
            }
        }
        selected
    }

    pub fn observations_seen(&self) -> u64 {
        self.observations_seen
    }

    pub fn active_expert_count(&self) -> usize {
        self.experts.len()
    }

    pub fn alpha_test(&self) -> Rational {
        self.alpha_test
    }

    pub fn betting_rate(&self) -> f64 {
        self.betting_rate
    }

    pub fn log_e_value(&self) -> f64 {
        self.log_e_value
    }

    pub fn log_threshold(&self) -> f64 {
        self.log_threshold
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    pub fn policy_sha256(&self) -> &str {
        &self.policy_sha256
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    /// Chained digest binding every selected expert/direction/rate and every
    /// candidate feature difference used to update the next policy state.
    pub fn policy_trace_sha256(&self) -> &str {
        &self.policy_trace_sha256
    }

    pub fn replay_inventory_size(&self) -> usize {
        self.seen_pair_ids.len()
    }

    pub fn history(&self) -> Vec<ExpertHistoryScore> {
        self.experts
            .iter()
            .map(|expert| ExpertHistoryScore {
                expert_id: expert.expert_id.clone(),
                historical_mean_norm: if self.observations_seen == 0 {
                    0.0
                } else {
                    expert.historical_score / self.observations_seen as f64
                },
                feature_dimension: expert.feature_sum.len(),
            })
            .collect()
    }
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, AdaptiveBetError> {
    if left.len() != right.len() {
        return Err(AdaptiveBetError::InputDimensionMismatch);
    }
    let mut result = 0.0;
    for (a, b) in left.iter().zip(right) {
        result += a * b;
    }
    if result.is_finite() {
        Ok(result)
    } else {
        Err(AdaptiveBetError::ArithmeticOverflow)
    }
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |norm, value| norm.hypot(*value))
}

fn rational_log_inverse(value: Rational) -> Result<f64, AdaptiveBetError> {
    let result = (value.denominator() as f64).ln() - (value.numerator() as f64).ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(AdaptiveBetError::ArithmeticOverflow)
    }
}

fn extend_policy_trace(
    previous_sha256: &str,
    pair_id: &str,
    action: &PredictableBetAction,
    experts: &[AdaptiveExpert],
    staged_differences: &[Vec<f64>],
    expert_factors: &[ExpertFactorUpdate],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"LUMINOUS-DYNAMICS/CMAQ-057/POLICY-TRACE/V1\0");
    hasher.update(previous_sha256.as_bytes());
    trace_string(&mut hasher, pair_id);
    hasher.update(action.sequence.to_be_bytes());
    trace_string(&mut hasher, &action.selected_expert_id);
    trace_string(&mut hasher, &action.response_map_sha256);
    trace_string(&mut hasher, &action.covariate_map_sha256);
    hasher.update(canonical_f64_bits(action.betting_rate).to_be_bytes());
    hasher.update(canonical_f64_bits(action.projected_feature_difference).to_be_bytes());
    hasher.update(canonical_f64_bits(action.betting_factor).to_be_bytes());
    hasher.update(canonical_f64_bits(action.log_e_value).to_be_bytes());
    hasher.update(canonical_f64_bits(action.log_threshold).to_be_bytes());
    hasher.update(canonical_f64_bits(action.direction_norm).to_be_bytes());
    hasher.update(canonical_f64_bits(action.historical_sum_norm).to_be_bytes());
    hasher.update(canonical_f64_bits(action.historical_score_used).to_be_bytes());
    hasher.update([u8::from(action.quarantined)]);
    hasher.update((action.direction.len() as u64).to_be_bytes());
    for value in &action.direction {
        hasher.update(canonical_f64_bits(*value).to_be_bytes());
    }
    hasher.update((experts.len() as u64).to_be_bytes());
    for (expert, difference) in experts.iter().zip(staged_differences) {
        trace_string(&mut hasher, &expert.expert_id);
        trace_string(&mut hasher, &expert.response_map_sha256);
        trace_string(&mut hasher, &expert.covariate_map_sha256);
        hasher.update((difference.len() as u64).to_be_bytes());
        for value in difference {
            hasher.update(canonical_f64_bits(*value).to_be_bytes());
        }
    }
    hasher.update((expert_factors.len() as u64).to_be_bytes());
    for factor in expert_factors {
        trace_string(&mut hasher, &factor.expert_id);
        hasher.update(canonical_f64_bits(factor.projected_feature_difference).to_be_bytes());
        hasher.update(canonical_f64_bits(factor.betting_factor).to_be_bytes());
        hasher.update(canonical_f64_bits(factor.log_factor).to_be_bytes());
        hasher.update(canonical_f64_bits(factor.log_e_value).to_be_bytes());
        hasher.update((factor.direction.len() as u64).to_be_bytes());
        for value in &factor.direction {
            hasher.update(canonical_f64_bits(*value).to_be_bytes());
        }
    }
    let digest = hasher.finalize();
    digest_hex(&digest)
}

fn trace_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn canonical_f64_bits(value: f64) -> u64 {
    if value == 0.0 { 0.0f64.to_bits() } else { value.to_bits() }
}

fn digest_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn policy_digest(
    manifest_sha256: &str,
    alpha_test: Rational,
    betting_rate: f64,
    stop_at_threshold: bool,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(POLICY_DOMAIN);
    hasher.update(POLICY_VERSION);
    hasher.update([0]);
    hasher.update(manifest_sha256.as_bytes());
    hasher.update(alpha_test.numerator().to_be_bytes());
    hasher.update(alpha_test.denominator().to_be_bytes());
    hasher.update(canonical_f64_bits(betting_rate).to_be_bytes());
    hasher.update([u8::from(stop_at_threshold)]);
    hasher.update((MAX_EXPERTS as u64).to_be_bytes());
    hasher.update((MAX_EXPERT_FEATURES as u64).to_be_bytes());
    hasher.update((MAX_TOTAL_FEATURES as u64).to_be_bytes());
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmaq_046_convex_target_approachability::BoxTarget;
    use cmaq_048_fourier_kernel_approachability::FourierFeatureMap;
    use cmaq_053_compositional_risk_budget::{
        FleetBudgetPlan, FleetScope, RepresentationWeight,
    };
    use cmaq_056_typed_null_manifest::{NullCalibration, TypedRiskPlan};

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const SHA_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn expert(
        id: &str,
        frequency: f64,
        response_sha: &str,
        covariate_sha: &str,
        target_sha: &str,
        prior: Rational,
    ) -> KernelExpertConfig {
        KernelExpertConfig {
            expert_id: id.into(),
            response_map: FourierFeatureMap::new(
                1,
                vec![vec![frequency]],
                response_sha.into(),
            )
            .unwrap(),
            covariate_map: FourierFeatureMap::new(
                1,
                vec![vec![0.5]],
                covariate_sha.into(),
            )
            .unwrap(),
            target: BoxTarget::new(vec![-0.01; 4], vec![0.01; 4]).unwrap(),
            prior_weight: prior,
            target_sha256: target_sha.into(),
            response_reference_sha256: SHA_E.into(),
            covariate_reference_sha256: SHA_F.into(),
        }
    }

    fn configs() -> Vec<KernelExpertConfig> {
        // The higher-frequency map is first so the first tie resolves to it.
        vec![
            expert("high", 0.5, SHA_A, SHA_B, SHA_C, Rational::new(1, 2).unwrap()),
            expert("low", 1.0, SHA_D, SHA_E, SHA_F, Rational::new(1, 2).unwrap()),
        ]
    }

    fn typed_plan(configurations: &[KernelExpertConfig]) -> (TypedRiskPlan, KernelFamilyManifest) {
        let manifest =
            KernelFamilyManifest::build(configurations, NullCalibration::direct_v1()).unwrap();
        let fleet = FleetBudgetPlan::new(
            Rational::probability(1, 20).unwrap(),
            SHA_A.into(),
            SHA_B.into(),
        )
        .unwrap();
        let scope = FleetScope {
            stream_id: "cmaq-057-test".into(),
            asset_id: "asset-1".into(),
            channel_id: "conditional-pair".into(),
            deployment_id: "deployment-1".into(),
            asset_index: 1,
            channel_index: 1,
            deployment_index: 1,
        };
        let local = fleet.local_alpha_for(&scope).unwrap();
        let alpha_test = local.divided_by_integer(2).unwrap();
        let plan = TypedRiskPlan::new(&fleet, scope, alpha_test, &manifest).unwrap();
        (plan, manifest)
    }

    fn monitor(configurations: &[KernelExpertConfig]) -> PredictableAdaptiveKernelMonitor {
        let (plan, manifest) = typed_plan(configurations);
        PredictableAdaptiveKernelMonitor::new(&plan, &manifest, configurations, 0.5).unwrap()
    }

    #[test]
    fn unthresholded_child_continues_and_emits_factors_after_child_threshold() {
        let configurations = configs();
        let (plan, manifest) = typed_plan(&configurations);
        let mut monitor = PredictableAdaptiveKernelMonitor::new_unthresholded(
            &plan,
            &manifest,
            &configurations,
            0.9,
        )
        .unwrap();

        let mut crossed_child_threshold = false;
        for sequence in 1..=2_000u64 {
            let update = monitor
                .observe_pair(
                    sequence,
                    &format!("unthresholded-{sequence}"),
                    &[0.0],
                    &[0.0],
                    &[std::f64::consts::PI],
                )
                .unwrap();
            assert!(!update.action.threshold_enforced);
            assert!(!update.action.quarantined);
            if update.action.log_e_value >= update.action.log_threshold {
                crossed_child_threshold = true;
            }
            if crossed_child_threshold {
                let next = monitor
                    .observe_pair(
                        sequence + 1,
                        &format!("unthresholded-continue-{}", sequence + 1),
                        &[0.0],
                        &[0.0],
                        &[std::f64::consts::PI],
                    )
                    .unwrap();
                assert!(!next.action.quarantined);
                assert!(next.action.betting_factor > 0.0);
                break;
            }
        }
        assert!(crossed_child_threshold);
        assert!(!monitor.is_quarantined());
    }

    #[test]
    fn prepared_pair_does_not_mutate_until_explicit_commit() {
        let configurations = configs();
        let mut monitor = monitor(&configurations);
        let prepared = monitor
            .prepare_pair(1, "prepared", &[0.0], &[0.0], &[std::f64::consts::PI])
            .unwrap();

        assert_eq!(monitor.observations_seen(), 0);
        assert_eq!(monitor.replay_inventory_size(), 0);
        assert_eq!(monitor.log_e_value(), 0.0);
        assert_eq!(prepared.update().observations_seen, 1);
        assert_eq!(prepared.update().pair_id, "prepared");

        let committed = monitor.commit_prepared(prepared).unwrap();
        assert_eq!(committed.observations_seen, 1);
        assert_eq!(monitor.observations_seen(), 1);
        assert_eq!(monitor.replay_inventory_size(), 1);
        assert_eq!(monitor.policy_trace_sha256(), committed.policy_trace_sha256);
    }

    #[test]
    fn stale_prepared_pair_is_rejected_without_partial_mutation() {
        let configurations = configs();
        let mut monitor = monitor(&configurations);
        let stale = monitor
            .prepare_pair(1, "stale", &[0.0], &[0.0], &[std::f64::consts::PI])
            .unwrap();
        monitor
            .observe_pair(1, "committed-first", &[0.0], &[0.0], &[0.0])
            .unwrap();
        let before_sequence = monitor.observations_seen();
        let before_trace = monitor.policy_trace_sha256().to_string();
        assert_eq!(
            monitor.commit_prepared(stale),
            Err(AdaptiveBetError::PreparedStateMismatch)
        );
        assert_eq!(monitor.observations_seen(), before_sequence);
        assert_eq!(monitor.policy_trace_sha256(), before_trace);
        assert_eq!(monitor.replay_inventory_size(), 1);
    }

    #[test]
    fn exact_pairs_preserve_unit_wealth_with_adaptive_policy() {
        let configurations = configs();
        let mut monitor = monitor(&configurations);
        for sequence in 1..=128u64 {
            let update = monitor
                .observe_pair(
                    sequence,
                    &format!("equal-pair-{sequence}"),
                    &[(sequence as f64 / 11.0).sin()],
                    &[sequence as f64 / 100.0],
                    &[sequence as f64 / 100.0],
                )
                .unwrap();
            assert!(!update.action.quarantined);
            assert!(update.action.log_e_value.abs() < 1e-12);
            assert_eq!(update.observations_seen, sequence);
        }
        assert!(monitor.log_e_value().abs() < 1e-12);
        assert_eq!(monitor.replay_inventory_size(), 128);
    }

    #[test]
    fn current_pair_leak_control_uses_only_the_previous_action() {
        let configurations = configs();
        let mut monitor = monitor(&configurations);

        // At t=1 all historical scores tie, so inventory member "high" is
        // selected before the current pair is transformed. The current pair
        // produces a larger diagnostic score for "low", but cannot change
        // the already selected action at t=1.
        let first = monitor
            .observe_pair(1, "first", &[0.0], &[0.0], &[std::f64::consts::PI])
            .unwrap();
        assert_eq!(first.action.selected_expert_id, "high");
        let after_first = monitor.history();
        assert!(after_first[1].historical_mean_norm > after_first[0].historical_mean_norm);

        // The larger "low" score is now in the past and can select the next
        // bet. This verifies lagged selection rather than same-sample choice.
        let second = monitor
            .observe_pair(2, "second", &[0.0], &[0.0], &[std::f64::consts::PI])
            .unwrap();
        assert_eq!(second.action.selected_expert_id, "low");
        assert!(second.action.direction_norm > 0.0);
        assert!(second.action.projected_feature_difference > 0.0);
        assert!(second.action.log_e_value > first.action.log_e_value);
    }

    #[test]
    fn adaptive_bet_detects_a_persistent_registered_conditional_feature_shift() {
        let configurations = configs();
        let mut monitor = monitor(&configurations);
        let mut quarantined = false;
        for sequence in 1..=64u64 {
            let update = monitor
                .observe_pair(
                    sequence,
                    &format!("shift-{sequence}"),
                    &[0.0],
                    &[0.0],
                    &[std::f64::consts::PI],
                )
                .unwrap();
            if update.action.quarantined {
                quarantined = true;
                break;
            }
        }
        assert!(quarantined);
        assert!(monitor.is_quarantined());
    }

    #[test]
    fn runtime_inventory_mutation_is_rejected_by_manifest_recomputation() {
        let configurations = configs();
        let (plan, manifest) = typed_plan(&configurations);
        let mut changed = configs();
        changed[1].response_map = FourierFeatureMap::new(
            1,
            vec![vec![1.75]],
            SHA_D.into(),
        )
        .unwrap();
        assert_eq!(
            PredictableAdaptiveKernelMonitor::new(&plan, &manifest, &changed, 0.5).err(),
            Some(AdaptiveBetError::Manifest(
                ManifestError::ManifestDigestMismatch("manifest")
            ))
        );
    }

    #[test]
    fn reference_calibrated_mode_is_not_silently_treated_as_direct_pair_equality() {
        let configurations = configs();
        let manifest = KernelFamilyManifest::build(
            &configurations,
            NullCalibration::reference_cmaq052_v1(Rational::new(1, 320).unwrap()),
        )
        .unwrap();
        let fleet = FleetBudgetPlan::new(
            Rational::probability(1, 20).unwrap(),
            SHA_A.into(),
            SHA_B.into(),
        )
        .unwrap();
        let scope = FleetScope {
            stream_id: "cmaq-057-reference-test".into(),
            asset_id: "asset-1".into(),
            channel_id: "conditional-pair".into(),
            deployment_id: "deployment-1".into(),
            asset_index: 1,
            channel_index: 1,
            deployment_index: 1,
        };
        let local = fleet.local_alpha_for(&scope).unwrap();
        let plan = TypedRiskPlan::new(
            &fleet,
            scope,
            local.divided_by_integer(2).unwrap(),
            &manifest,
        )
        .unwrap();
        assert_eq!(
            PredictableAdaptiveKernelMonitor::new(&plan, &manifest, &configurations, 0.5).err(),
            Some(AdaptiveBetError::ReferenceCalibrationUnsupported)
        );
    }

    #[test]
    fn replay_sequence_and_transform_failures_do_not_mutate_state() {
        let configurations = configs();
        let mut monitor = monitor(&configurations);
        assert!(matches!(
            monitor.observe_pair(1, "bad", &[0.0, 1.0], &[0.0], &[1.0]),
            Err(AdaptiveBetError::KernelFeature(KernelError::DimensionMismatch))
        ));
        assert_eq!(monitor.observations_seen(), 0);
        assert_eq!(monitor.replay_inventory_size(), 0);

        monitor
            .observe_pair(1, "retry", &[0.0], &[0.0], &[0.0])
            .unwrap();
        assert_eq!(
            monitor.observe_pair(2, "retry", &[0.0], &[0.0], &[0.0]),
            Err(AdaptiveBetError::DuplicatePairId)
        );
        assert_eq!(
            monitor.observe_pair(3, "next", &[0.0], &[0.0], &[0.0]),
            Err(AdaptiveBetError::SequenceMismatch)
        );
        assert_eq!(monitor.observations_seen(), 1);
    }

    #[test]
    fn policy_trace_chains_actions_and_candidate_feature_history() {
        let configurations = configs();
        let mut first = monitor(&configurations);
        let mut second = monitor(&configurations);

        let first_update = first
            .observe_pair(1, "same-pair", &[0.0], &[0.0], &[std::f64::consts::PI])
            .unwrap();
        let second_update = second
            .observe_pair(1, "same-pair", &[0.0], &[0.0], &[0.0])
            .unwrap();

        assert_ne!(
            first_update.policy_trace_sha256,
            second_update.policy_trace_sha256
        );
        assert_eq!(
            first.policy_trace_sha256(),
            first_update.policy_trace_sha256
        );

        let prior_trace = first.policy_trace_sha256().to_string();
        let next = first
            .observe_pair(2, "next-pair", &[0.0], &[0.0], &[std::f64::consts::PI])
            .unwrap();
        assert_ne!(next.policy_trace_sha256, prior_trace);
    }

    #[test]
    fn invalid_betting_rates_fail_closed() {
        let configurations = configs();
        let (plan, manifest) = typed_plan(&configurations);
        for rate in [0.0, 1.0, -0.1, f64::NAN, f64::INFINITY] {
            assert_eq!(
                PredictableAdaptiveKernelMonitor::new(&plan, &manifest, &configurations, rate).err(),
                Some(AdaptiveBetError::InvalidBettingRate)
            );
        }
    }
}
