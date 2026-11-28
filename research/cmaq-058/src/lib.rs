#![forbid(unsafe_code)]

//! Exact O(K) capital-restart renewal mixture over predictable kernel factors.
//!
//! A single CMAQ-057 policy state produces one prequential factor per kernel
//! expert. For each expert k, every possible start time reuses that same
//! factor stream after its start. The capital is reset at each start, but the
//! policy state is shared. The all-start mixture therefore admits the exact
//! recurrence:
//
//!   R_(k,t) = f_(k,t) * (R_(k,t-1) + nu_t)
//!
//! with total wealth U_t + sum_k w_k R_(k,t), where U_t is the exact mass of
//! not-yet-started components. This does not reset the policy at each start.

use cmaq_053_compositional_risk_budget::Rational;
use cmaq_054_evidence_adaptive_kernel_family::KernelExpertConfig;
use cmaq_055_dyadic_start_time_prior::{StartPriorError, future_mass_after, weight_for_start};
use cmaq_056_typed_null_manifest::{
    KernelFamilyManifest, ManifestError, NullCalibration, TypedRiskPlan,
};
use cmaq_057_predictable_adaptive_kernel_betting::{
    AdaptiveBetError, ExpertFactorUpdate, PredictableAdaptiveKernelMonitor,
};
use sha2::{Digest, Sha256};

const MAX_RENEWAL_HORIZON: u64 = 1_024;
const MAX_EXPERTS: usize = 32;
const MAX_PAIR_ID_BYTES: usize = 256;
const POSTERIOR_TOLERANCE: f64 = 1e-8;
const FACTOR_CONSISTENCY_TOLERANCE: f64 = 1e-10;
const FACTOR_BATCH_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/CMAQ-058/FACTOR-BATCH/V1\0";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenewalError {
    InvalidHorizon,
    HorizonExceeded,
    AlreadyQuarantined,
    SequenceMismatch,
    InvalidPairId,
    PairIdTooLong,
    ReferenceCalibrationUnsupported,
    EmptyExpertInventory,
    TooManyExperts,
    ExpertInventoryMismatch,
    PolicyThresholdMustBeExternal,
    PolicyIdentityMismatch,
    ManifestIdentityMismatch,
    InvalidFactor,
    FactorMismatch,
    ArithmeticOverflow,
    PosteriorNormalizationFailure,
    StartPrior(StartPriorError),
    Manifest(ManifestError),
    AdaptiveBet(AdaptiveBetError),
}

impl From<StartPriorError> for RenewalError {
    fn from(value: StartPriorError) -> Self {
        Self::StartPrior(value)
    }
}

impl From<ManifestError> for RenewalError {
    fn from(value: ManifestError) -> Self {
        Self::Manifest(value)
    }
}

impl From<AdaptiveBetError> for RenewalError {
    fn from(value: AdaptiveBetError) -> Self {
        Self::AdaptiveBet(value)
    }
}

impl std::fmt::Display for RenewalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RenewalError {}

#[derive(Clone, Debug)]
struct KernelPrior {
    expert_id: String,
    response_map_sha256: String,
    covariate_map_sha256: String,
    weight: Rational,
    log_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelRestartPosterior {
    pub expert_id: String,
    pub prior_weight: Rational,
    /// Log of the exact all-start wealth for this expert, before kernel weight.
    pub log_restart_wealth: f64,
    /// Posterior mass of this expert's restarted wealth in the full mixture.
    pub posterior_weight: f64,
    /// Cumulative log wealth of the shared, non-restarted expert factor stream.
    pub log_cumulative_factor_wealth: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenewalUpdate {
    pub sequence: u64,
    pub pair_id: String,
    pub log_mixture_e_value: f64,
    pub log_threshold: f64,
    pub quarantined: bool,
    pub start_prior: Rational,
    pub future_start_mass: Rational,
    pub future_start_posterior: f64,
    pub kernels: Vec<KernelRestartPosterior>,
    pub expert_factors: Vec<ExpertFactorUpdate>,
    pub policy_sha256: String,
    pub policy_trace_sha256: String,
    pub factor_batch_sha256: String,
}

/// Capital-restarted all-start mixture with one shared prequential policy per
/// kernel expert.
///
/// For each expert k, define the same factor stream f_(k,t) for every start s.
/// Its start-s component has capital one immediately before s and then
/// multiplies only by factors from s onward. The policy state which produces
/// f_(k,t) is global/shared and may include observations before s; this remains
/// valid because each factor is predictable under the full filtration.
///
/// This is different from CMAQ-055's full-reset oracle, which restarts both
/// policy and capital. The recurrence is exact for the capital-only design:
///
///   C_(k,t) = C_(k,t-1) + log(f_(k,t))
///   A_(k,t) = A_(k,t-1) + nu_t * exp(-C_(k,t-1))
///   R_(k,t) = exp(C_(k,t)) * A_(k,t)
///
/// The implementation stores log(C) and log(A), and uses log-add-exp.
/// Restart bookkeeping is O(K) per observation and O(K) numeric state. The
/// shared CMAQ-057 policy owns one O(T) replay ledger, not one ledger per start.
#[derive(Clone, Debug)]
pub struct CapitalRestartRenewalMonitor {
    policy: PredictableAdaptiveKernelMonitor,
    kernels: Vec<KernelPrior>,
    log_cumulative_factor_wealth: Vec<f64>,
    log_weighted_inverse_prefix_sum: Vec<f64>,
    log_restart_wealth: Vec<f64>,
    alpha_test: Rational,
    log_threshold: f64,
    horizon_cap: u64,
    observations_seen: u64,
    future_start_mass: Rational,
    log_mixture_e_value: f64,
    quarantined: bool,
    policy_sha256: String,
    manifest_sha256: String,
    last_policy_trace_sha256: String,
}

impl CapitalRestartRenewalMonitor {
    /// Builds the restart mixture and its shared, unthresholded CMAQ-057
    /// factor producer. The outer renewal mixture owns the only quarantine
    /// decision; the child policy must therefore run unthresholded.
    pub fn new(
        risk_plan: &TypedRiskPlan,
        manifest: &KernelFamilyManifest,
        configurations: &[KernelExpertConfig],
        betting_rate: f64,
        horizon_cap: u64,
    ) -> Result<Self, RenewalError> {
        if horizon_cap == 0 || horizon_cap > MAX_RENEWAL_HORIZON {
            return Err(RenewalError::InvalidHorizon);
        }
        if configurations.is_empty() {
            return Err(RenewalError::EmptyExpertInventory);
        }
        if configurations.len() > MAX_EXPERTS {
            return Err(RenewalError::TooManyExperts);
        }
        if risk_plan.reference_failure_budget().is_some()
            || !matches!(
                manifest.calibration(),
                NullCalibration::DirectConditionalPairedEquality { .. }
            )
        {
            return Err(RenewalError::ReferenceCalibrationUnsupported);
        }

        risk_plan.verify_manifest(manifest)?;
        let policy = PredictableAdaptiveKernelMonitor::new_unthresholded(
            risk_plan,
            manifest,
            configurations,
            betting_rate,
        )?;
        if policy.policy_sha256().is_empty()
            || policy.manifest_sha256() != manifest.digests().portfolio_inventory_sha256
        {
            return Err(RenewalError::PolicyIdentityMismatch);
        }

        let weights = manifest.representation_weights();
        if weights.len() != configurations.len() {
            return Err(RenewalError::ExpertInventoryMismatch);
        }

        let mut kernels = Vec::with_capacity(weights.len());
        let mut total_weight = None;
        for (configuration, weight) in configurations.iter().zip(weights) {
            if configuration.expert_id != weight.member_id {
                return Err(RenewalError::ExpertInventoryMismatch);
            }
            let log_weight = rational_log(weight.weight)?;
            total_weight = Some(match total_weight {
                None => weight.weight,
                Some(previous) => previous
                    .checked_add(weight.weight)
                    .map_err(|_| RenewalError::ExpertInventoryMismatch)?,
            });
            kernels.push(KernelPrior {
                expert_id: weight.member_id.clone(),
                response_map_sha256: configuration.response_map.map_sha256().to_string(),
                covariate_map_sha256: configuration.covariate_map.map_sha256().to_string(),
                weight: weight.weight,
                log_weight,
            });
        }
        if total_weight != Some(Rational::one()) {
            return Err(RenewalError::ExpertInventoryMismatch);
        }

        let alpha_test = risk_plan.alpha_test();
        let log_threshold = rational_log_inverse(alpha_test)?;
        let policy_sha256 = policy.policy_sha256().to_string();
        let manifest_sha256 = manifest.digests().portfolio_inventory_sha256.clone();
        let last_policy_trace_sha256 = policy.policy_trace_sha256().to_string();

        Ok(Self {
            policy,
            log_cumulative_factor_wealth: vec![0.0; kernels.len()],
            log_weighted_inverse_prefix_sum: vec![f64::NEG_INFINITY; kernels.len()],
            log_restart_wealth: vec![f64::NEG_INFINITY; kernels.len()],
            kernels,
            alpha_test,
            log_threshold,
            horizon_cap,
            observations_seen: 0,
            future_start_mass: Rational::one(),
            log_mixture_e_value: 0.0,
            quarantined: false,
            policy_sha256,
            manifest_sha256,
            last_policy_trace_sha256,
        })
    }

    /// Processes one pair through the shared CMAQ-057 factor policy and then
    /// applies the exact O(K) renewal recurrence. The child prepares an
    /// immutable update without mutating its replay ledger; the parent stages
    /// all renewal values and posteriors before committing the child and parent.
    pub fn observe_pair(
        &mut self,
        sequence: u64,
        pair_id: &str,
        covariate: &[f64],
        x: &[f64],
        y: &[f64],
    ) -> Result<RenewalUpdate, RenewalError> {
        if self.quarantined {
            return Err(RenewalError::AlreadyQuarantined);
        }
        if sequence > self.horizon_cap {
            return Err(RenewalError::HorizonExceeded);
        }
        if pair_id.trim().is_empty() {
            return Err(RenewalError::InvalidPairId);
        }
        if pair_id.len() > MAX_PAIR_ID_BYTES {
            return Err(RenewalError::PairIdTooLong);
        }
        let expected_sequence = self
            .observations_seen
            .checked_add(1)
            .ok_or(RenewalError::ArithmeticOverflow)?;
        if sequence != expected_sequence {
            return Err(RenewalError::SequenceMismatch);
        }

        // Prepare the shared policy without mutating it or cloning its
        // O(T) replay ledger. The parent inspects the emitted factors and
        // stages every renewal/posterior calculation before committing both.
        let prepared_policy = self
            .policy
            .prepare_pair(sequence, pair_id, covariate, x, y)?;
        let policy_update = prepared_policy.update().clone();
        if policy_update.action.threshold_enforced || policy_update.action.quarantined {
            return Err(RenewalError::PolicyThresholdMustBeExternal);
        }
        if policy_update.policy_sha256 != self.policy_sha256
            || self.policy.policy_sha256() != self.policy_sha256
        {
            return Err(RenewalError::PolicyIdentityMismatch);
        }
        if policy_update.policy_trace_sha256 == self.last_policy_trace_sha256 {
            return Err(RenewalError::PolicyIdentityMismatch);
        }
        if policy_update.expert_factors.len() != self.kernels.len() {
            return Err(RenewalError::ExpertInventoryMismatch);
        }

        let start_prior = weight_for_start(sequence)?;
        let log_start_prior = rational_log(start_prior)?;
        let future_start_mass = future_mass_after(sequence)?;
        let log_future_mass = rational_log(future_start_mass)?;

        let mut next_cumulative = Vec::with_capacity(self.kernels.len());
        let mut next_inverse_prefix = Vec::with_capacity(self.kernels.len());
        let mut next_restart_wealth = Vec::with_capacity(self.kernels.len());

        for (index, (kernel, factor)) in self
            .kernels
            .iter()
            .zip(&policy_update.expert_factors)
            .enumerate()
        {
            if factor.expert_id != kernel.expert_id
                || factor.response_map_sha256 != kernel.response_map_sha256
                || factor.covariate_map_sha256 != kernel.covariate_map_sha256
            {
                return Err(RenewalError::ExpertInventoryMismatch);
            }
            if !factor.betting_factor.is_finite()
                || factor.betting_factor <= 0.0
                || !factor.log_factor.is_finite()
                || !factor.log_e_value.is_finite()
            {
                return Err(RenewalError::InvalidFactor);
            }
            let expected_log_factor = factor.betting_factor.ln();
            if (expected_log_factor - factor.log_factor).abs() > FACTOR_CONSISTENCY_TOLERANCE {
                return Err(RenewalError::FactorMismatch);
            }

            let (cumulative, inverse_prefix, restart_wealth) = advance_kernel_restart(
                self.log_cumulative_factor_wealth[index],
                self.log_weighted_inverse_prefix_sum[index],
                log_start_prior,
                factor.log_factor,
            )?;
            if (cumulative - factor.log_e_value).abs()
                > FACTOR_CONSISTENCY_TOLERANCE * (1.0 + cumulative.abs())
            {
                return Err(RenewalError::FactorMismatch);
            }

            next_cumulative.push(cumulative);
            next_inverse_prefix.push(inverse_prefix);
            next_restart_wealth.push(restart_wealth);
        }

        let next_log_mixture =
            mixture_log_wealth(&self.kernels, &next_restart_wealth, log_future_mass)?;
        let future_posterior = (log_future_mass - next_log_mixture).exp();
        if !future_posterior.is_finite()
            || !(0.0..=1.0 + POSTERIOR_TOLERANCE).contains(&future_posterior)
        {
            return Err(RenewalError::PosteriorNormalizationFailure);
        }

        let mut kernels = Vec::with_capacity(self.kernels.len());
        let mut posterior_sum = future_posterior;
        for (index, kernel) in self.kernels.iter().enumerate() {
            let log_weighted_wealth = kernel.log_weight + next_restart_wealth[index];
            let posterior = (log_weighted_wealth - next_log_mixture).exp();
            if !posterior.is_finite() || !(0.0..=1.0 + POSTERIOR_TOLERANCE).contains(&posterior) {
                return Err(RenewalError::PosteriorNormalizationFailure);
            }
            posterior_sum += posterior;
            kernels.push(KernelRestartPosterior {
                expert_id: kernel.expert_id.clone(),
                prior_weight: kernel.weight,
                log_restart_wealth: next_restart_wealth[index],
                posterior_weight: posterior,
                log_cumulative_factor_wealth: next_cumulative[index],
            });
        }
        if !posterior_sum.is_finite() || (posterior_sum - 1.0).abs() > POSTERIOR_TOLERANCE {
            return Err(RenewalError::PosteriorNormalizationFailure);
        }

        let next_quarantined = next_log_mixture >= self.log_threshold;
        let factor_batch_sha256 = factor_batch_digest(
            &self.manifest_sha256,
            &self.policy_sha256,
            &policy_update.policy_trace_sha256,
            sequence,
            pair_id,
            &policy_update.expert_factors,
        );

        // Commit the child policy and the complete kernel × start mixture only
        // after every factor, renewal state, total wealth and posterior passes.
        self.policy.commit_prepared(prepared_policy)?;
        self.log_cumulative_factor_wealth = next_cumulative;
        self.log_weighted_inverse_prefix_sum = next_inverse_prefix;
        self.log_restart_wealth = next_restart_wealth;
        self.observations_seen = sequence;
        self.future_start_mass = future_start_mass;
        self.log_mixture_e_value = next_log_mixture;
        self.quarantined = next_quarantined;
        self.last_policy_trace_sha256 = policy_update.policy_trace_sha256.clone();

        Ok(RenewalUpdate {
            sequence,
            pair_id: pair_id.to_string(),
            log_mixture_e_value: next_log_mixture,
            log_threshold: self.log_threshold,
            quarantined: next_quarantined,
            start_prior,
            future_start_mass,
            future_start_posterior: future_posterior,
            kernels,
            expert_factors: policy_update.expert_factors,
            policy_sha256: self.policy_sha256.clone(),
            policy_trace_sha256: policy_update.policy_trace_sha256,
            factor_batch_sha256,
        })
    }

    pub fn observations_seen(&self) -> u64 {
        self.observations_seen
    }

    pub fn horizon_cap(&self) -> u64 {
        self.horizon_cap
    }

    pub fn expert_count(&self) -> usize {
        self.kernels.len()
    }

    pub fn alpha_test(&self) -> Rational {
        self.alpha_test
    }

    pub fn log_mixture_e_value(&self) -> f64 {
        self.log_mixture_e_value
    }

    pub fn log_threshold(&self) -> f64 {
        self.log_threshold
    }

    pub fn future_start_mass(&self) -> Rational {
        self.future_start_mass
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

    pub fn policy_trace_sha256(&self) -> &str {
        &self.last_policy_trace_sha256
    }

    /// O(K) numeric state is maintained regardless of the number of active
    /// restart components. The shared policy owns the sole O(T) pair ledger.
    pub fn restart_state_count(&self) -> usize {
        self.log_cumulative_factor_wealth.len()
            + self.log_weighted_inverse_prefix_sum.len()
            + self.log_restart_wealth.len()
    }

    pub fn pair_replay_inventory_size(&self) -> usize {
        self.policy.replay_inventory_size()
    }
}

/// Advances one exact capital-restart component in the log domain.
///
/// C_t = C_(t-1) + log(f_t)
/// A_t = A_(t-1) + nu_t * exp(-C_(t-1))
/// log R_t = C_t + log(A_t)
fn advance_kernel_restart(
    log_cumulative_previous: f64,
    log_inverse_prefix_previous: f64,
    log_start_prior: f64,
    log_factor: f64,
) -> Result<(f64, f64, f64), RenewalError> {
    if !log_cumulative_previous.is_finite()
        || log_start_prior.is_nan()
        || !log_factor.is_finite()
        || log_inverse_prefix_previous.is_nan()
    {
        return Err(RenewalError::ArithmeticOverflow);
    }
    let log_inverse_prefix = log_add_exp(
        log_inverse_prefix_previous,
        log_start_prior - log_cumulative_previous,
    )?;
    let log_cumulative = log_cumulative_previous + log_factor;
    let log_restart_wealth = log_cumulative + log_inverse_prefix;
    if !log_cumulative.is_finite()
        || !log_inverse_prefix.is_finite()
        || !log_restart_wealth.is_finite()
    {
        return Err(RenewalError::ArithmeticOverflow);
    }
    Ok((log_cumulative, log_inverse_prefix, log_restart_wealth))
}

fn mixture_log_wealth(
    kernels: &[KernelPrior],
    log_restart_wealth: &[f64],
    log_future_mass: f64,
) -> Result<f64, RenewalError> {
    if kernels.len() != log_restart_wealth.len() || !log_future_mass.is_finite() {
        return Err(RenewalError::ArithmeticOverflow);
    }
    let mut terms = Vec::with_capacity(kernels.len() + 1);
    terms.push(log_future_mass);
    for (kernel, restart_wealth) in kernels.iter().zip(log_restart_wealth) {
        let term = kernel.log_weight + restart_wealth;
        if !term.is_finite() {
            return Err(RenewalError::ArithmeticOverflow);
        }
        terms.push(term);
    }
    log_sum_exp(&terms)
}

fn log_add_exp(left: f64, right: f64) -> Result<f64, RenewalError> {
    if left.is_nan() || right.is_nan() {
        return Err(RenewalError::ArithmeticOverflow);
    }
    if left == f64::NEG_INFINITY {
        return if right.is_finite() {
            Ok(right)
        } else {
            Err(RenewalError::ArithmeticOverflow)
        };
    }
    if right == f64::NEG_INFINITY {
        return if left.is_finite() {
            Ok(left)
        } else {
            Err(RenewalError::ArithmeticOverflow)
        };
    }
    if !left.is_finite() || !right.is_finite() {
        return Err(RenewalError::ArithmeticOverflow);
    }
    let maximum = left.max(right);
    let result = maximum + ((left - maximum).exp() + (right - maximum).exp()).ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(RenewalError::ArithmeticOverflow)
    }
}

fn log_sum_exp(values: &[f64]) -> Result<f64, RenewalError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(RenewalError::ArithmeticOverflow);
    }
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled_sum = values
        .iter()
        .map(|value| (*value - maximum).exp())
        .sum::<f64>();
    let result = maximum + scaled_sum.ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(RenewalError::ArithmeticOverflow)
    }
}

fn rational_log(value: Rational) -> Result<f64, RenewalError> {
    if value.numerator() == 0 || value.denominator() == 0 {
        return Err(RenewalError::ArithmeticOverflow);
    }
    let result = (value.numerator() as f64).ln() - (value.denominator() as f64).ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(RenewalError::ArithmeticOverflow)
    }
}

fn rational_log_inverse(value: Rational) -> Result<f64, RenewalError> {
    if value.numerator() == 0 || value.denominator() == 0 {
        return Err(RenewalError::ArithmeticOverflow);
    }
    let result = (value.denominator() as f64).ln() - (value.numerator() as f64).ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(RenewalError::ArithmeticOverflow)
    }
}

fn factor_batch_digest(
    manifest_sha256: &str,
    policy_sha256: &str,
    policy_trace_sha256: &str,
    sequence: u64,
    pair_id: &str,
    factors: &[ExpertFactorUpdate],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(FACTOR_BATCH_DOMAIN);
    hasher.update(manifest_sha256.as_bytes());
    hasher.update(policy_sha256.as_bytes());
    hasher.update(policy_trace_sha256.as_bytes());
    hasher.update(sequence.to_be_bytes());
    hasher.update((pair_id.len() as u64).to_be_bytes());
    hasher.update(pair_id.as_bytes());
    hasher.update((factors.len() as u64).to_be_bytes());
    for factor in factors {
        hasher.update((factor.expert_id.len() as u64).to_be_bytes());
        hasher.update(factor.expert_id.as_bytes());
        hasher.update(factor.log_factor.to_bits().to_be_bytes());
        hasher.update(factor.log_e_value.to_bits().to_be_bytes());
    }
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
    use cmaq_053_compositional_risk_budget::{FleetBudgetPlan, FleetScope, RepresentationWeight};

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const SHA_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn configuration(
        id: &str,
        frequency: f64,
        map_sha: &str,
        covariate_sha: &str,
        target_sha: &str,
    ) -> KernelExpertConfig {
        KernelExpertConfig {
            expert_id: id.into(),
            response_map: FourierFeatureMap::new(1, vec![vec![frequency]], map_sha.into()).unwrap(),
            covariate_map: FourierFeatureMap::new(1, vec![vec![0.5]], covariate_sha.into())
                .unwrap(),
            target: BoxTarget::new(vec![-0.01; 4], vec![0.01; 4]).unwrap(),
            prior_weight: Rational::new(1, 2).unwrap(),
            target_sha256: target_sha.into(),
            response_reference_sha256: SHA_E.into(),
            covariate_reference_sha256: SHA_F.into(),
        }
    }

    fn configs() -> Vec<KernelExpertConfig> {
        vec![
            configuration("low", 1.0, SHA_A, SHA_B, SHA_C),
            configuration("high", 0.5, SHA_D, SHA_E, SHA_F),
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
            stream_id: "cmaq-058-test".into(),
            asset_id: "asset-1".into(),
            channel_id: "conditional-paired".into(),
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
        (plan, manifest)
    }

    fn monitor(horizon: u64) -> CapitalRestartRenewalMonitor {
        let configurations = configs();
        let (plan, manifest) = typed_plan(&configurations);
        CapitalRestartRenewalMonitor::new(&plan, &manifest, &configurations, 0.5, horizon).unwrap()
    }

    fn naive_restart_logs(factors_by_kernel: &[Vec<f64>]) -> (Vec<f64>, f64) {
        let horizon = factors_by_kernel[0].len() as u64;
        let future = future_mass_after(horizon).unwrap();
        let mut kernel_logs = Vec::with_capacity(factors_by_kernel.len());
        for factors in factors_by_kernel {
            let mut start_terms = Vec::with_capacity(factors.len());
            for start in 1..=horizon {
                let mut log_product = 0.0;
                for factor in &factors[(start - 1) as usize..] {
                    log_product += factor.ln();
                }
                start_terms
                    .push(rational_log(weight_for_start(start).unwrap()).unwrap() + log_product);
            }
            kernel_logs.push(log_sum_exp(&start_terms).unwrap());
        }
        (kernel_logs, rational_log(future).unwrap())
    }

    #[test]
    fn log_recurrence_matches_naive_all_start_enumerator_for_gains_and_losses() {
        let factors = vec![
            vec![1.1, 0.8, 1.3, 0.95, 1.2, 0.7, 1.05, 1.15],
            vec![0.9, 1.4, 0.7, 1.05, 1.1, 0.85, 1.2, 0.95],
        ];
        let mut cumulative = vec![0.0; 2];
        let mut log_a = vec![f64::NEG_INFINITY; 2];
        let mut log_restart = vec![f64::NEG_INFINITY; 2];

        for time in 1..=8u64 {
            let nu = weight_for_start(time).unwrap();
            let log_nu = rational_log(nu).unwrap();
            for kernel in 0..2 {
                let (next_cumulative, next_log_a, next_log_restart) = advance_kernel_restart(
                    cumulative[kernel],
                    log_a[kernel],
                    log_nu,
                    factors[kernel][(time - 1) as usize].ln(),
                )
                .unwrap();
                cumulative[kernel] = next_cumulative;
                log_a[kernel] = next_log_a;
                log_restart[kernel] = next_log_restart;
            }

            let prefix = factors
                .iter()
                .map(|stream| stream[..time as usize].to_vec())
                .collect::<Vec<_>>();
            let (naive_kernels, naive_future) = naive_restart_logs(&prefix);
            for kernel in 0..2 {
                assert!(
                    (log_restart[kernel] - naive_kernels[kernel]).abs() < 1e-10,
                    "kernel {kernel}, t={time}"
                );
            }

            let manifest =
                KernelFamilyManifest::build(&configs(), NullCalibration::direct_v1()).unwrap();
            let weights = manifest.representation_weights().to_vec();
            let log_total = log_sum_exp(&[
                naive_future,
                rational_log(weights[0].weight).unwrap() + naive_kernels[0],
                rational_log(weights[1].weight).unwrap() + naive_kernels[1],
            ])
            .unwrap();
            let direct_total = log_sum_exp(&[
                rational_log(future_mass_after(time).unwrap()).unwrap(),
                rational_log(weights[0].weight).unwrap() + log_restart[0],
                rational_log(weights[1].weight).unwrap() + log_restart[1],
            ])
            .unwrap();
            assert!((log_total - direct_total).abs() < 1e-10);
        }
    }

    #[test]
    fn renewal_recurrence_matches_naive_enumerator_across_kernel_counts_and_dyadic_horizons() {
        let horizons = [
            1usize, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1_024,
        ];

        for kernel_count in [1usize, 2, 3, 7] {
            for horizon in horizons {
                let factors_by_kernel = (0..kernel_count)
                    .map(|kernel| {
                        (1..=horizon)
                            .map(|time| {
                                let kernel = kernel as u64;
                                let time = time as u64;
                                let residue =
                                    (time * 37 + kernel * 19 + time * (kernel + 1) * 7) % 97;
                                match residue % 5 {
                                    0 => 0.55,
                                    1 => 1.45,
                                    2 => 0.80,
                                    3 => 1.20,
                                    _ => 1.00,
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();

                let mut cumulative = vec![0.0; kernel_count];
                let mut inverse_prefix = vec![f64::NEG_INFINITY; kernel_count];
                let mut restart_wealth = vec![f64::NEG_INFINITY; kernel_count];

                for time in 1..=horizon {
                    let log_start_prior =
                        rational_log(weight_for_start(time as u64).unwrap()).unwrap();
                    for kernel in 0..kernel_count {
                        let (next_cumulative, next_inverse_prefix, next_restart_wealth) =
                            advance_kernel_restart(
                                cumulative[kernel],
                                inverse_prefix[kernel],
                                log_start_prior,
                                factors_by_kernel[kernel][time - 1].ln(),
                            )
                            .unwrap();
                        cumulative[kernel] = next_cumulative;
                        inverse_prefix[kernel] = next_inverse_prefix;
                        restart_wealth[kernel] = next_restart_wealth;
                    }
                }

                let (naive_wealth, naive_future) = naive_restart_logs(&factors_by_kernel);
                for kernel in 0..kernel_count {
                    assert!(
                        (restart_wealth[kernel] - naive_wealth[kernel]).abs() < 1e-9,
                        "kernel_count={kernel_count}, horizon={horizon}, kernel={kernel}"
                    );
                }

                // Equal rational priors are used here to test the aggregation
                // independently of the manifest-construction test fixtures.
                let log_kernel_prior = -(kernel_count as f64).ln();
                let mut fast_terms =
                    vec![rational_log(future_mass_after(horizon as u64).unwrap()).unwrap()];
                let mut naive_terms = vec![naive_future];
                for kernel in 0..kernel_count {
                    fast_terms.push(log_kernel_prior + restart_wealth[kernel]);
                    naive_terms.push(log_kernel_prior + naive_wealth[kernel]);
                }
                let fast_total = log_sum_exp(&fast_terms).unwrap();
                let naive_total = log_sum_exp(&naive_terms).unwrap();
                assert!(
                    (fast_total - naive_total).abs() < 1e-9,
                    "mixture kernel_count={kernel_count}, horizon={horizon}"
                );

                let posterior_sum = fast_terms
                    .iter()
                    .map(|term| (*term - fast_total).exp())
                    .sum::<f64>();
                assert!(
                    (posterior_sum - 1.0).abs() < 1e-10,
                    "posterior kernel_count={kernel_count}, horizon={horizon}"
                );
            }
        }
    }

    #[test]
    fn dyadic_start_prior_and_future_mass_sum_to_one_exactly_through_horizon_cap() {
        let mut started_mass = None;
        for sequence in 1..=MAX_RENEWAL_HORIZON {
            let prior_weight = weight_for_start(sequence).unwrap();
            started_mass = Some(match started_mass {
                Some(previous) => previous.checked_add(prior_weight).unwrap(),
                None => prior_weight,
            });
            let total_mass = started_mass
                .unwrap()
                .checked_add(future_mass_after(sequence).unwrap())
                .unwrap();
            assert_eq!(
                total_mass,
                Rational::one(),
                "start prior mass did not normalize at sequence {sequence}"
            );
        }
    }

    #[test]
    fn unit_factors_preserve_unit_total_wealth_and_exact_future_mass() {
        let mut monitor = monitor(64);
        for sequence in 1..=64u64 {
            let update = monitor
                .observe_pair(
                    sequence,
                    &format!("equal-{sequence}"),
                    &[(sequence as f64 / 9.0).sin()],
                    &[0.25],
                    &[0.25],
                )
                .unwrap();
            assert!(!update.quarantined);
            assert!(update.log_mixture_e_value.abs() < 1e-10);
            assert_eq!(
                update.future_start_mass,
                future_mass_after(sequence).unwrap()
            );
            assert!(
                (update.future_start_posterior
                    - update.future_start_mass.numerator() as f64
                        / update.future_start_mass.denominator() as f64)
                    .abs()
                    < 1e-10
            );
        }
        assert_eq!(monitor.observations_seen(), 64);
        assert_eq!(monitor.restart_state_count(), 6);
        assert_eq!(monitor.pair_replay_inventory_size(), 64);
    }

    #[test]
    fn kernel_and_future_posteriors_normalize() {
        let mut monitor = monitor(16);
        let update = monitor
            .observe_pair(1, "one", &[0.0], &[0.0], &[0.0])
            .unwrap();
        let posterior_sum = update
            .kernels
            .iter()
            .map(|kernel| kernel.posterior_weight)
            .sum::<f64>()
            + update.future_start_posterior;
        assert!((posterior_sum - 1.0).abs() < 1e-10);
        assert_eq!(update.kernels.len(), 2);
        assert!((update.kernels[0].posterior_weight - 0.25).abs() < 1e-10);
        assert!((update.kernels[1].posterior_weight - 0.25).abs() < 1e-10);
        assert!((update.future_start_posterior - 0.5).abs() < 1e-10);
    }

    #[test]
    fn parent_quarantine_uses_one_threshold_after_full_kernel_start_mixture() {
        let mut monitor = monitor(128);
        let mut alarm = None;
        for sequence in 1..=128u64 {
            let (covariate, x, y) = if sequence % 2 == 0 {
                ([std::f64::consts::PI], [std::f64::consts::TAU], [0.0])
            } else {
                ([0.0], [0.0], [std::f64::consts::TAU])
            };
            let update = monitor
                .observe_pair(sequence, &format!("shift-{sequence}"), &covariate, &x, &y)
                .unwrap();
            if update.quarantined {
                alarm = Some(update);
                break;
            }
        }
        let update =
            alarm.expect("registered conditional feature shift should accumulate evidence");
        assert!(update.log_mixture_e_value >= update.log_threshold);
        assert!(!update.expert_factors.is_empty());
        assert!(!update.policy_trace_sha256.is_empty());
        assert!(!update.factor_batch_sha256.is_empty());
    }

    #[test]
    fn failed_pair_does_not_mutate_child_or_renewal_state() {
        let mut monitor = monitor(16);
        assert!(matches!(
            monitor.observe_pair(1, "bad", &[0.0, 1.0], &[0.0], &[1.0]),
            Err(RenewalError::AdaptiveBet(_))
        ));
        assert_eq!(monitor.observations_seen(), 0);
        assert_eq!(monitor.log_mixture_e_value(), 0.0);
        assert_eq!(monitor.pair_replay_inventory_size(), 0);

        monitor
            .observe_pair(1, "retry", &[0.0], &[0.0], &[0.0])
            .unwrap();
        assert_eq!(
            monitor.observe_pair(2, "retry", &[0.0], &[0.0], &[0.0]),
            Err(RenewalError::AdaptiveBet(AdaptiveBetError::DuplicatePairId))
        );
        assert_eq!(monitor.observations_seen(), 1);
    }

    #[test]
    fn horizon_limit_fails_closed() {
        let configurations = configs();
        let (plan, manifest) = typed_plan(&configurations);
        assert_eq!(
            CapitalRestartRenewalMonitor::new(
                &plan,
                &manifest,
                &configurations,
                0.5,
                MAX_RENEWAL_HORIZON + 1,
            )
            .err(),
            Some(RenewalError::InvalidHorizon)
        );
    }
}
