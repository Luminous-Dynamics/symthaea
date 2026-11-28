#![forbid(unsafe_code)]

//! Anytime-valid, fixed-prior portfolio over conditional paired Fourier experts.
//!
//! Every expert observes a bounded product feature difference under the same
//! conditional paired-distribution null. The family wealth is the fixed
//! normalized mixture of the expert e-processes. Evidence-posterior weights
//! change with observed wealth, but prior weights, maps, targets and alpha are
//! immutable. The portfolio is bound to a CMAQ-053 composed risk plan.

use std::collections::HashSet;

use cmaq_046_convex_target_approachability::{
    ApproachabilityError, ApproachabilityUpdate, BoxTarget,
    ConvexTargetApproachabilityMonitor,
};
use cmaq_048_fourier_kernel_approachability::{FourierFeatureMap, KernelError};
use cmaq_053_compositional_risk_budget::{ComposedRiskPlan, Rational};

const MAX_EXPERTS: usize = 32;
const MAX_EXPERT_FEATURES: usize = 1_024;
const MAX_TOTAL_FEATURES: usize = 8_192;
const MAX_PAIR_ID_BYTES: usize = 256;
const MAX_PAIR_IDS: usize = 1_000_000;
const DIGEST_HEX_LEN: usize = 64;
const NORM_TOLERANCE: f64 = 1e-12;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelFamilyError {
    EmptyExpertInventory,
    TooManyExperts,
    InvalidExpertId,
    DuplicateExpertId,
    InvalidDigest(&'static str),
    RiskPlanInventoryMismatch,
    RiskPlanWeightMismatch,
    PriorWeightsDoNotSumToOne,
    AlphaConversionOverflow,
    DimensionOverflow,
    ExpertFeatureLimitExceeded,
    TotalFeatureBudgetExceeded,
    InputDimensionMismatch,
    TargetDimensionMismatch,
    InvalidPairId,
    PairIdTooLong,
    DuplicatePairId,
    PairIdInventoryFull,
    SequenceMismatch,
    AlreadyQuarantined,
    ObservationNormExceeded,
    ArithmeticOverflow,
    Monitor(ApproachabilityError),
    FeatureMap(KernelError),
}

impl From<ApproachabilityError> for KernelFamilyError {
    fn from(value: ApproachabilityError) -> Self {
        Self::Monitor(value)
    }
}

impl From<KernelError> for KernelFamilyError {
    fn from(value: KernelError) -> Self {
        Self::FeatureMap(value)
    }
}

#[derive(Clone, Debug)]
pub struct KernelExpertConfig {
    pub expert_id: String,
    pub response_map: FourierFeatureMap,
    pub covariate_map: FourierFeatureMap,
    pub target: BoxTarget,
    pub prior_weight: Rational,
    pub target_sha256: String,
    pub response_reference_sha256: String,
    pub covariate_reference_sha256: String,
}

#[derive(Clone, Debug)]
struct KernelExpert {
    expert_id: String,
    response_map: FourierFeatureMap,
    covariate_map: FourierFeatureMap,
    prior_weight: Rational,
    log_prior_weight: f64,
    member_alpha: Rational,
    target_sha256: String,
    response_reference_sha256: String,
    covariate_reference_sha256: String,
    monitor: ConvexTargetApproachabilityMonitor,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertPosteriorWeight {
    pub expert_id: String,
    pub prior_weight: Rational,
    pub posterior_weight: f64,
    pub log_e_value: f64,
    pub member_alpha: (u32, u32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelExpertUpdate {
    pub expert_id: String,
    pub posterior_weight: f64,
    pub prior_weight: Rational,
    pub log_e_value: f64,
    pub evidence: ApproachabilityUpdate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelFamilyUpdate {
    pub sequence: u64,
    pub pair_id: String,
    pub log_mixture_e_value: f64,
    pub log_threshold: f64,
    pub experts: Vec<KernelExpertUpdate>,
    pub quarantined: bool,
}

/// Fixed inventory of conditional paired Fourier experts.
///
/// The child alpha is alpha_portfolio * prior_weight. A child therefore
/// cannot cross its own Ville threshold before its weighted contribution
/// alone reaches 1/alpha_portfolio, which necessarily implies that the family
/// mixture has crossed its threshold. The family still checks its own mixture
/// after every atomic update.
#[derive(Clone, Debug)]
pub struct KernelFamilyPortfolio {
    experts: Vec<KernelExpert>,
    alpha_portfolio: Rational,
    log_threshold: f64,
    portfolio_contract_sha256: String,
    fleet_plan_sha256: String,
    deployment_plan_sha256: String,
    observations_seen: u64,
    seen_pair_ids: HashSet<String>,
    log_mixture_e_value: f64,
    quarantined: bool,
}

impl KernelFamilyPortfolio {
    pub fn new(
        risk_plan: &ComposedRiskPlan,
        portfolio_contract_sha256: String,
        configurations: Vec<KernelExpertConfig>,
    ) -> Result<Self, KernelFamilyError> {
        if configurations.is_empty() {
            return Err(KernelFamilyError::EmptyExpertInventory);
        }
        if configurations.len() > MAX_EXPERTS {
            return Err(KernelFamilyError::TooManyExperts);
        }
        validate_digest(&portfolio_contract_sha256, "portfolio_contract_sha256")?;

        let receipt = risk_plan.receipt();
        if receipt.bindings.representation_inventory_sha256 != portfolio_contract_sha256 {
            return Err(KernelFamilyError::RiskPlanInventoryMismatch);
        }
        if receipt.representation_weights.len() != configurations.len() {
            return Err(KernelFamilyError::RiskPlanWeightMismatch);
        }

        let alpha_portfolio = risk_plan.alpha_test();
        let mut prior_sum = None;
        let mut expert_ids = HashSet::with_capacity(configurations.len());
        let mut experts = Vec::with_capacity(configurations.len());
        let mut total_features = 0usize;
        let mut response_input_dimension = None;
        let mut covariate_input_dimension = None;

        for (index, configuration) in configurations.into_iter().enumerate() {
            if configuration.expert_id.trim().is_empty()
                || configuration.expert_id.len() > MAX_PAIR_ID_BYTES
            {
                return Err(KernelFamilyError::InvalidExpertId);
            }
            if !expert_ids.insert(configuration.expert_id.clone()) {
                return Err(KernelFamilyError::DuplicateExpertId);
            }

            let registered_weight = &receipt.representation_weights[index];
            if registered_weight.member_id != configuration.expert_id
                || registered_weight.weight != configuration.prior_weight
            {
                return Err(KernelFamilyError::RiskPlanWeightMismatch);
            }

            prior_sum = Some(match prior_sum {
                None => configuration.prior_weight,
                Some(value) => value
                    .checked_add(configuration.prior_weight)
                    .map_err(|_| KernelFamilyError::PriorWeightsDoNotSumToOne)?,
            });

            validate_digest(&configuration.target_sha256, "target_sha256")?;
            validate_digest(
                &configuration.response_reference_sha256,
                "response_reference_sha256",
            )?;
            validate_digest(
                &configuration.covariate_reference_sha256,
                "covariate_reference_sha256",
            )?;

            let response_dimension = configuration.response_map.feature_dimension();
            let covariate_dimension = configuration.covariate_map.feature_dimension();
            let expert_dimension = response_dimension
                .checked_mul(covariate_dimension)
                .ok_or(KernelFamilyError::DimensionOverflow)?;
            if expert_dimension > MAX_EXPERT_FEATURES {
                return Err(KernelFamilyError::ExpertFeatureLimitExceeded);
            }
            total_features = total_features
                .checked_add(expert_dimension)
                .ok_or(KernelFamilyError::DimensionOverflow)?;
            if total_features > MAX_TOTAL_FEATURES {
                return Err(KernelFamilyError::TotalFeatureBudgetExceeded);
            }
            if configuration.target.dimension() != expert_dimension {
                return Err(KernelFamilyError::TargetDimensionMismatch);
            }

            match response_input_dimension {
                None => response_input_dimension = Some(configuration.response_map.input_dimension()),
                Some(expected) if expected != configuration.response_map.input_dimension() => {
                    return Err(KernelFamilyError::InputDimensionMismatch);
                }
                Some(_) => {}
            }
            match covariate_input_dimension {
                None => covariate_input_dimension = Some(configuration.covariate_map.input_dimension()),
                Some(expected) if expected != configuration.covariate_map.input_dimension() => {
                    return Err(KernelFamilyError::InputDimensionMismatch);
                }
                Some(_) => {}
            }

            let member_alpha = multiply_probabilities(alpha_portfolio, configuration.prior_weight)?;
            let member_alpha_numerator = u32::try_from(member_alpha.numerator())
                .map_err(|_| KernelFamilyError::AlphaConversionOverflow)?;
            let member_alpha_denominator = u32::try_from(member_alpha.denominator())
                .map_err(|_| KernelFamilyError::AlphaConversionOverflow)?;
            let log_prior_weight = rational_log(configuration.prior_weight)?;
            let monitor = ConvexTargetApproachabilityMonitor::new(
                configuration.target,
                1.0,
                member_alpha_numerator,
                member_alpha_denominator,
            )?;

            experts.push(KernelExpert {
                expert_id: configuration.expert_id,
                response_map: configuration.response_map,
                covariate_map: configuration.covariate_map,
                prior_weight: configuration.prior_weight,
                log_prior_weight,
                member_alpha,
                target_sha256: configuration.target_sha256,
                response_reference_sha256: configuration.response_reference_sha256,
                covariate_reference_sha256: configuration.covariate_reference_sha256,
                monitor,
            });
        }

        if prior_sum != Some(Rational::one()) {
            return Err(KernelFamilyError::PriorWeightsDoNotSumToOne);
        }

        let log_threshold = rational_log_inverse(alpha_portfolio)?;
        let initial_log_mixture = mixture_log_e_value(&experts, None)?;
        if !initial_log_mixture.is_finite() || initial_log_mixture.abs() > 1e-10 {
            return Err(KernelFamilyError::ArithmeticOverflow);
        }

        Ok(Self {
            experts,
            alpha_portfolio,
            log_threshold,
            portfolio_contract_sha256,
            fleet_plan_sha256: receipt.fleet_plan_sha256.clone(),
            deployment_plan_sha256: receipt.bindings.deployment_plan_sha256.clone(),
            observations_seen: 0,
            seen_pair_ids: HashSet::new(),
            log_mixture_e_value: initial_log_mixture,
            quarantined: false,
        })
    }

    /// Adds one common-covariate paired response. Every expert is staged on a
    /// cloned compact CMAQ-046 state; pair identity is consumed and states are
    /// committed only after every transform, update and mixture calculation
    /// succeeds.
    pub fn observe_pair(
        &mut self,
        sequence: u64,
        pair_id: &str,
        covariate: &[f64],
        x: &[f64],
        y: &[f64],
    ) -> Result<KernelFamilyUpdate, KernelFamilyError> {
        if self.quarantined {
            return Err(KernelFamilyError::AlreadyQuarantined);
        }
        if pair_id.trim().is_empty() {
            return Err(KernelFamilyError::InvalidPairId);
        }
        if pair_id.len() > MAX_PAIR_ID_BYTES {
            return Err(KernelFamilyError::PairIdTooLong);
        }
        if self.seen_pair_ids.contains(pair_id) {
            return Err(KernelFamilyError::DuplicatePairId);
        }
        if self.seen_pair_ids.len() >= MAX_PAIR_IDS {
            return Err(KernelFamilyError::PairIdInventoryFull);
        }

        let expected_sequence = self
            .observations_seen
            .checked_add(1)
            .ok_or(KernelFamilyError::ArithmeticOverflow)?;
        if sequence != expected_sequence {
            return Err(KernelFamilyError::SequenceMismatch);
        }

        let mut staged_monitors = self
            .experts
            .iter()
            .map(|expert| expert.monitor.clone())
            .collect::<Vec<_>>();
        let mut staged_updates = Vec::with_capacity(self.experts.len());

        for (expert, monitor) in self.experts.iter().zip(staged_monitors.iter_mut()) {
            let response_x = expert.response_map.transform(x)?;
            let response_y = expert.response_map.transform(y)?;
            let covariate_features = expert.covariate_map.transform(covariate)?;
            let response_dimension = expert.response_map.feature_dimension();
            let covariate_dimension = expert.covariate_map.feature_dimension();
            let product_dimension = response_dimension
                .checked_mul(covariate_dimension)
                .ok_or(KernelFamilyError::DimensionOverflow)?;
            let mut observation = vec![0.0; product_dimension];

            for (covariate_index, covariate_feature) in
                covariate_features.iter().enumerate()
            {
                for (response_index, (y_feature, x_feature)) in
                    response_y.iter().zip(&response_x).enumerate()
                {
                    let feature_index = covariate_index * response_dimension + response_index;
                    observation[feature_index] =
                        0.5 * covariate_feature * (y_feature - x_feature);
                }
            }

            let norm = l2_norm(&observation);
            if !norm.is_finite() {
                return Err(KernelFamilyError::ArithmeticOverflow);
            }
            if norm > 1.0 + NORM_TOLERANCE {
                return Err(KernelFamilyError::ObservationNormExceeded);
            }

            let update = monitor.observe(&observation)?;
            staged_updates.push(update);
        }

        let next_log_mixture = mixture_log_e_value(&self.experts, Some(&staged_monitors))?;
        let next_posterior = posterior_weights(
            &self.experts,
            &staged_monitors,
            next_log_mixture,
        )?;
        let next_quarantined = next_log_mixture >= self.log_threshold;

        // If a child has reached its alpha_portfolio * prior threshold,
        // its weighted wealth alone reaches the family threshold.
        if staged_monitors.iter().any(|monitor| monitor.is_quarantined())
            && !next_quarantined
        {
            return Err(KernelFamilyError::ArithmeticOverflow);
        }

        let mut expert_updates = Vec::with_capacity(self.experts.len());
        for (((expert, monitor), evidence), posterior_weight) in self
            .experts
            .iter_mut()
            .zip(staged_monitors.into_iter())
            .zip(staged_updates.into_iter())
            .zip(next_posterior.into_iter())
        {
            monitor_state_replace(&mut expert.monitor, monitor);
            expert_updates.push(KernelExpertUpdate {
                expert_id: expert.expert_id.clone(),
                posterior_weight,
                prior_weight: expert.prior_weight,
                log_e_value: expert.monitor.log_e_value(),
                evidence,
            });
        }

        self.observations_seen = expected_sequence;
        self.seen_pair_ids.insert(pair_id.to_string());
        self.log_mixture_e_value = next_log_mixture;
        self.quarantined = next_quarantined;

        Ok(KernelFamilyUpdate {
            sequence,
            pair_id: pair_id.to_string(),
            log_mixture_e_value: next_log_mixture,
            log_threshold: self.log_threshold,
            experts: expert_updates,
            quarantined: next_quarantined,
        })
    }

    pub fn posterior_weights(&self) -> Result<Vec<ExpertPosteriorWeight>, KernelFamilyError> {
        let monitors = self
            .experts
            .iter()
            .map(|expert| expert.monitor.clone())
            .collect::<Vec<_>>();
        let posterior = posterior_weights(&self.experts, &monitors, self.log_mixture_e_value)?;
        Ok(self
            .experts
            .iter()
            .zip(monitors.iter())
            .zip(posterior)
            .map(|((expert, monitor), posterior_weight)| {
                let numerator = u32::try_from(expert.member_alpha.numerator())
                    .map_err(|_| KernelFamilyError::AlphaConversionOverflow)?;
                let denominator = u32::try_from(expert.member_alpha.denominator())
                    .map_err(|_| KernelFamilyError::AlphaConversionOverflow)?;
                Ok(ExpertPosteriorWeight {
                    expert_id: expert.expert_id.clone(),
                    prior_weight: expert.prior_weight,
                    posterior_weight,
                    log_e_value: monitor.log_e_value(),
                    member_alpha: (numerator, denominator),
                })
            })
            .collect::<Result<Vec<_>, KernelFamilyError>>()?)
    }

    pub fn observations_seen(&self) -> u64 {
        self.observations_seen
    }

    pub fn alpha_portfolio(&self) -> Rational {
        self.alpha_portfolio
    }

    pub fn log_mixture_e_value(&self) -> f64 {
        self.log_mixture_e_value
    }

    pub fn log_threshold(&self) -> f64 {
        self.log_threshold
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    pub fn expert_count(&self) -> usize {
        self.experts.len()
    }

    pub fn total_feature_count(&self) -> usize {
        self.experts
            .iter()
            .map(|expert| {
                expert.response_map.feature_dimension()
                    * expert.covariate_map.feature_dimension()
            })
            .sum()
    }

    pub fn portfolio_contract_sha256(&self) -> &str {
        &self.portfolio_contract_sha256
    }

    pub fn fleet_plan_sha256(&self) -> &str {
        &self.fleet_plan_sha256
    }

    pub fn deployment_plan_sha256(&self) -> &str {
        &self.deployment_plan_sha256
    }

    pub fn map_bindings(&self) -> Vec<(String, String, String, String)> {
        self.experts
            .iter()
            .map(|expert| {
                (
                    expert.expert_id.clone(),
                    expert.response_map.map_sha256().to_string(),
                    expert.covariate_map.map_sha256().to_string(),
                    expert.target_sha256.clone(),
                )
            })
            .collect()
    }
}

/// Replaces only the bounded CMAQ-046 state; immutable maps/bindings remain
/// in the original expert object.
fn monitor_state_replace(
    destination: &mut ConvexTargetApproachabilityMonitor,
    source: ConvexTargetApproachabilityMonitor,
) {
    *destination = source;
}

fn multiply_probabilities(
    left: Rational,
    right: Rational,
) -> Result<Rational, KernelFamilyError> {
    let first_cancel = gcd(left.numerator(), right.denominator());
    let second_cancel = gcd(right.numerator(), left.denominator());
    let numerator = (left.numerator() / first_cancel)
        .checked_mul(right.numerator() / second_cancel)
        .ok_or(KernelFamilyError::AlphaConversionOverflow)?;
    let denominator = (left.denominator() / second_cancel)
        .checked_mul(right.denominator() / first_cancel)
        .ok_or(KernelFamilyError::AlphaConversionOverflow)?;
    Rational::probability(numerator, denominator)
        .map_err(|_| KernelFamilyError::AlphaConversionOverflow)
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn rational_log(value: Rational) -> Result<f64, KernelFamilyError> {
    let result = (value.numerator() as f64).ln() - (value.denominator() as f64).ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(KernelFamilyError::ArithmeticOverflow)
    }
}

fn rational_log_inverse(value: Rational) -> Result<f64, KernelFamilyError> {
    let result = (value.denominator() as f64).ln() - (value.numerator() as f64).ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(KernelFamilyError::ArithmeticOverflow)
    }
}

fn mixture_log_e_value(
    experts: &[KernelExpert],
    staged_monitors: Option<&[ConvexTargetApproachabilityMonitor]>,
) -> Result<f64, KernelFamilyError> {
    if let Some(monitors) = staged_monitors {
        if monitors.len() != experts.len() {
            return Err(KernelFamilyError::RiskPlanWeightMismatch);
        }
    }
    let mut maximum = f64::NEG_INFINITY;
    let mut terms = Vec::with_capacity(experts.len());
    for (index, expert) in experts.iter().enumerate() {
        let log_e = staged_monitors
            .map(|monitors| monitors[index].log_e_value())
            .unwrap_or_else(|| expert.monitor.log_e_value());
        let term = expert.log_prior_weight + log_e;
        if !term.is_finite() {
            return Err(KernelFamilyError::ArithmeticOverflow);
        }
        maximum = maximum.max(term);
        terms.push(term);
    }
    let sum = terms.iter().map(|term| (*term - maximum).exp()).sum::<f64>();
    let result = maximum + sum.ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(KernelFamilyError::ArithmeticOverflow)
    }
}

fn posterior_weights(
    experts: &[KernelExpert],
    staged_monitors: &[ConvexTargetApproachabilityMonitor],
    log_mixture: f64,
) -> Result<Vec<f64>, KernelFamilyError> {
    if experts.len() != staged_monitors.len() || !log_mixture.is_finite() {
        return Err(KernelFamilyError::ArithmeticOverflow);
    }
    experts
        .iter()
        .zip(staged_monitors)
        .map(|(expert, monitor)| {
            let log_weight = expert.log_prior_weight + monitor.log_e_value() - log_mixture;
            let weight = log_weight.exp();
            if weight.is_finite() && (0.0..=1.0 + 1e-10).contains(&weight) {
                Ok(weight)
            } else {
                Err(KernelFamilyError::ArithmeticOverflow)
            }
        })
        .collect()
}

fn validate_digest(value: &str, label: &'static str) -> Result<(), KernelFamilyError> {
    if value.len() == DIGEST_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(KernelFamilyError::InvalidDigest(label))
    }
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |norm, value| norm.hypot(*value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmaq_053_compositional_risk_budget::{
        FleetBudgetPlan, FleetScope, RepresentationWeight, RiskPlanBindings,
    };

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const SHA_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const SHA_1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const SHA_2: &str = "2222222222222222222222222222222222222222222222222222222222222222";

    fn risk_plan(weights: Vec<RepresentationWeight>) -> ComposedRiskPlan {
        let fleet = FleetBudgetPlan::new(
            Rational::probability(1, 20).unwrap(),
            SHA_A.into(),
            SHA_B.into(),
        )
        .unwrap();
        let scope = FleetScope {
            stream_id: "stream-054".into(),
            asset_id: "asset-1".into(),
            channel_id: "channel-1".into(),
            deployment_id: "deployment-1".into(),
            asset_index: 1,
            channel_index: 1,
            deployment_index: 1,
        };
        let local = fleet.local_alpha_for(&scope).unwrap();
        let alpha_test = local.divided_by_integer(2).unwrap();
        let delta = local.divided_by_integer(2).unwrap();
        ComposedRiskPlan::new(
            &fleet,
            scope,
            alpha_test,
            delta,
            weights,
            RiskPlanBindings {
                deployment_plan_sha256: SHA_C.into(),
                reference_manifest_sha256: SHA_D.into(),
                target_envelope_sha256: SHA_E.into(),
                monitor_build_sha256: SHA_F.into(),
                representation_inventory_sha256: SHA_1.into(),
                calibration_method_sha256: SHA_2.into(),
            },
        )
        .unwrap()
    }

    fn weight(numerator: u128, denominator: u128) -> Rational {
        Rational::new(numerator, denominator).unwrap()
    }

    fn response_map(frequency: f64, digest: &str) -> FourierFeatureMap {
        FourierFeatureMap::new(1, vec![vec![frequency]], digest.into()).unwrap()
    }

    fn covariate_map(digest: &str) -> FourierFeatureMap {
        FourierFeatureMap::new(1, vec![vec![0.0]], digest.into()).unwrap()
    }

    fn target() -> BoxTarget {
        BoxTarget::new(vec![-0.005; 4], vec![0.005; 4]).unwrap()
    }

    fn configuration(
        id: &str,
        response_frequency: f64,
        weight_value: Rational,
        response_digest: &str,
        covariate_digest: &str,
        target_digest: &str,
    ) -> KernelExpertConfig {
        KernelExpertConfig {
            expert_id: id.into(),
            response_map: response_map(response_frequency, response_digest),
            covariate_map: covariate_map(covariate_digest),
            target: target(),
            prior_weight: weight_value,
            target_sha256: target_digest.into(),
            response_reference_sha256: SHA_E.into(),
            covariate_reference_sha256: SHA_F.into(),
        }
    }

    fn two_expert_configs() -> Vec<KernelExpertConfig> {
        vec![
            configuration("low-frequency", 1.0, weight(1, 2), SHA_A, SHA_B, SHA_C),
            configuration("high-frequency", 0.5, weight(1, 2), SHA_D, SHA_E, SHA_F),
        ]
    }

    fn portfolio() -> KernelFamilyPortfolio {
        let weights = vec![
            RepresentationWeight {
                member_id: "low-frequency".into(),
                weight: weight(1, 2),
            },
            RepresentationWeight {
                member_id: "high-frequency".into(),
                weight: weight(1, 2),
            },
        ];
        KernelFamilyPortfolio::new(&risk_plan(weights), SHA_1.into(), two_expert_configs()).unwrap()
    }

    #[test]
    fn initial_mixture_is_unit_wealth_and_posterior_equals_prior() {
        let family = portfolio();
        assert_eq!(family.expert_count(), 2);
        assert_eq!(family.observations_seen(), 0);
        assert!(family.log_mixture_e_value().abs() < 1e-12);
        let posterior = family.posterior_weights().unwrap();
        assert_eq!(posterior.len(), 2);
        assert!((posterior[0].posterior_weight - 0.5).abs() < 1e-12);
        assert!((posterior[1].posterior_weight - 0.5).abs() < 1e-12);
        assert_eq!(posterior[0].member_alpha, (1, 640));
        assert_eq!(posterior[1].member_alpha, (1, 640));
    }

    #[test]
    fn scale_selective_shift_reweights_toward_the_resolving_expert() {
        let mut family = portfolio();
        let mut update = None;
        for sequence in 1..=200u64 {
            let result = family
                .observe_pair(sequence, &format!("pair-{sequence}"), &[0.0], &[0.0], &[std::f64::consts::TAU])
                .unwrap();
            if result.quarantined {
                update = Some(result);
                break;
            }
        }
        let result = update.expect("the high-frequency expert should detect the registered shift");
        assert!(result.quarantined);
        assert!(result.log_mixture_e_value >= result.log_threshold);
        let low = result
            .experts
            .iter()
            .find(|expert| expert.expert_id == "low-frequency")
            .unwrap();
        let high = result
            .experts
            .iter()
            .find(|expert| expert.expert_id == "high-frequency")
            .unwrap();
        assert!(high.posterior_weight > 0.99);
        assert!(high.log_e_value > low.log_e_value + 5.0);
    }

    #[test]
    fn equal_pairs_do_not_create_evidence_or_change_posterior() {
        let mut family = portfolio();
        for sequence in 1..=200u64 {
            let update = family
                .observe_pair(sequence, &format!("null-{sequence}"), &[sequence as f64 / 10.0], &[0.25], &[0.25])
                .unwrap();
            assert!(!update.quarantined);
        }
        assert_eq!(family.observations_seen(), 200);
        assert!(family.log_mixture_e_value().abs() < 1e-12);
        let posterior = family.posterior_weights().unwrap();
        assert!((posterior[0].posterior_weight - 0.5).abs() < 1e-12);
        assert!((posterior[1].posterior_weight - 0.5).abs() < 1e-12);
    }

    #[test]
    fn pair_replay_and_sequence_errors_do_not_mutate_family() {
        let mut family = portfolio();
        family
            .observe_pair(1, "pair-1", &[0.0], &[0.25], &[0.25])
            .unwrap();
        let before = family.clone();
        assert_eq!(
            family.observe_pair(2, "pair-1", &[0.0], &[0.25], &[0.25]),
            Err(KernelFamilyError::DuplicatePairId)
        );
        assert_eq!(
            family.observe_pair(3, "pair-3", &[0.0], &[0.25], &[0.25]),
            Err(KernelFamilyError::SequenceMismatch)
        );
        assert_eq!(family.observations_seen(), before.observations_seen());
        assert_eq!(family.log_mixture_e_value(), before.log_mixture_e_value());
        assert_eq!(family.posterior_weights().unwrap(), before.posterior_weights().unwrap());
    }

    #[test]
    fn invalid_later_expert_does_not_commit_earlier_expert_or_consume_pair_id() {
        let risk = risk_plan(vec![
            RepresentationWeight {
                member_id: "safe".into(),
                weight: weight(1, 2),
            },
            RepresentationWeight {
                member_id: "overflowing".into(),
                weight: weight(1, 2),
            },
        ]);
        let configurations = vec![
            configuration("safe", 1.0, weight(1, 2), SHA_A, SHA_B, SHA_C),
            configuration("overflowing", f64::MAX, weight(1, 2), SHA_D, SHA_E, SHA_F),
        ];
        let mut family = KernelFamilyPortfolio::new(&risk, SHA_1.into(), configurations).unwrap();
        assert!(matches!(
            family.observe_pair(1, "retry-me", &[0.0], &[0.0], &[2.0]),
            Err(KernelFamilyError::FeatureMap(KernelError::ArithmeticOverflow))
        ));
        assert_eq!(family.observations_seen(), 0);
        assert!(family.log_mixture_e_value().abs() < 1e-12);
        // The failed transaction did not consume the pair ID; a valid retry works.
        family
            .observe_pair(1, "retry-me", &[0.0], &[0.0], &[1.0])
            .unwrap();
        assert_eq!(family.observations_seen(), 1);
    }

    #[test]
    fn inventory_digest_and_registered_weights_are_mandatory() {
        let risk = risk_plan(vec![
            RepresentationWeight {
                member_id: "low-frequency".into(),
                weight: weight(1, 2),
            },
            RepresentationWeight {
                member_id: "high-frequency".into(),
                weight: weight(1, 2),
            },
        ]);
        assert_eq!(
            KernelFamilyPortfolio::new(&risk, SHA_2.into(), two_expert_configs()).err(),
            Some(KernelFamilyError::RiskPlanInventoryMismatch)
        );
        let mismatched = vec![
            configuration("low-frequency", 1.0, weight(1, 2), SHA_A, SHA_B, SHA_C),
            configuration("high-frequency", 0.5, weight(1, 4), SHA_D, SHA_E, SHA_F),
        ];
        assert_eq!(
            KernelFamilyPortfolio::new(&risk, SHA_1.into(), mismatched).err(),
            Some(KernelFamilyError::RiskPlanWeightMismatch)
        );
    }

    #[test]
    fn family_quarantine_is_permanent() {
        let mut family = portfolio();
        let mut stopped = false;
        for sequence in 1..=200u64 {
            let update = family
                .observe_pair(sequence, &format!("shift-{sequence}"), &[0.0], &[0.0], &[std::f64::consts::TAU])
                .unwrap();
            if update.quarantined {
                stopped = true;
                break;
            }
        }
        assert!(stopped);
        assert!(family.is_quarantined());
        assert_eq!(
            family.observe_pair(999, "late", &[0.0], &[0.0], &[0.0]),
            Err(KernelFamilyError::AlreadyQuarantined)
        );
    }
}
