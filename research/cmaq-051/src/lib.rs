#![forbid(unsafe_code)]

//! Pre-registered representation portfolio for paired conditional testing.
//!
//! Two bounded paired-feature observations are computed from every shared
//! covariate/response pair:
//!
//! 1. one-hot context block times paired response Fourier difference;
//! 2. continuous covariate Fourier feature times paired response difference.
//!
//! Each is monitored by a CMAQ-046 nonnegative test supermartingale under the
//! same conditional paired-distribution null. Their fixed equal-weight
//! mixture is also a nonnegative test supermartingale; no independence
//! assumption between representation experts is required.
//!
//! A child process is stopped at its own threshold and its wealth is held
//! constant (optional stopping). The other child continues. The portfolio
//! itself quarantines at the mixture's Ville threshold.

use std::collections::HashSet;

use cmaq_046_convex_target_approachability::{
    ApproachabilityError, ApproachabilityUpdate, BoxTarget, ConvexTargetApproachabilityMonitor,
};
use cmaq_048_fourier_kernel_approachability::{FourierFeatureMap, KernelError};

const MAX_CONTEXTS: usize = 256;
const MAX_MEMBER_FEATURES: usize = 1_024;
const MAX_PAIR_ID_BYTES: usize = 256;
const MAX_PAIR_IDS: usize = 1_000_000;
const DIGEST_HEX_LEN: usize = 64;
const NORM_TOLERANCE: f64 = 1e-12;
const LOG_HALF: f64 = -std::f64::consts::LN_2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortfolioError {
    EmptyContextInventory,
    TooManyContexts,
    DimensionOverflow,
    MemberFeatureLimitExceeded,
    TargetDimensionMismatch,
    InvalidDigest(&'static str),
    EmptyPairId,
    PairIdTooLong,
    DuplicatePairId,
    PairIdInventoryFull,
    UnknownContext,
    SequenceMismatch,
    AlreadyQuarantined,
    ArithmeticOverflow,
    ObservationNormExceeded,
    Monitor(ApproachabilityError),
    FeatureMap(KernelError),
}

impl From<ApproachabilityError> for PortfolioError {
    fn from(value: ApproachabilityError) -> Self {
        Self::Monitor(value)
    }
}

impl From<KernelError> for PortfolioError {
    fn from(value: KernelError) -> Self {
        Self::FeatureMap(value)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepresentationPortfolioUpdate {
    pub sequence: u64,
    pub pair_id: String,
    pub context_id: usize,
    pub contextual_member_update: Option<ApproachabilityUpdate>,
    pub product_member_update: Option<ApproachabilityUpdate>,
    pub contextual_member_log_e: f64,
    pub product_member_log_e: f64,
    pub mixture_log_e: f64,
    pub log_threshold: f64,
    pub contextual_member_stopped: bool,
    pub product_member_stopped: bool,
    pub quarantined: bool,
}

/// A fixed 50/50 portfolio over two conditional paired-distribution witnesses.
///
/// The representation inventory, prior weights, maps, targets, alpha and
/// digest bindings are immutable. The portfolio does not choose a winning
/// representation after observing outcomes.
#[derive(Clone, Debug, PartialEq)]
pub struct RepresentationEPortfolio {
    response_map: FourierFeatureMap,
    covariate_map: FourierFeatureMap,
    context_count: usize,
    context_inventory_sha256: String,
    portfolio_contract_sha256: String,
    contextual_target_sha256: String,
    product_target_sha256: String,
    response_reference_sha256: String,
    covariate_reference_sha256: String,
    alpha_numerator: u32,
    alpha_denominator: u32,
    observations_seen: u64,
    contextual_monitor: ConvexTargetApproachabilityMonitor,
    product_monitor: ConvexTargetApproachabilityMonitor,
    seen_pair_ids: HashSet<String>,
    log_mixture_e: f64,
    quarantined: bool,
}

impl RepresentationEPortfolio {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        response_map: FourierFeatureMap,
        covariate_map: FourierFeatureMap,
        context_count: usize,
        contextual_target: BoxTarget,
        product_target: BoxTarget,
        alpha_numerator: u32,
        alpha_denominator: u32,
        context_inventory_sha256: String,
        portfolio_contract_sha256: String,
        contextual_target_sha256: String,
        product_target_sha256: String,
        response_reference_sha256: String,
        covariate_reference_sha256: String,
    ) -> Result<Self, PortfolioError> {
        if context_count == 0 {
            return Err(PortfolioError::EmptyContextInventory);
        }
        if context_count > MAX_CONTEXTS {
            return Err(PortfolioError::TooManyContexts);
        }

        let response_dimension = response_map.feature_dimension();
        let covariate_dimension = covariate_map.feature_dimension();
        let contextual_dimension = context_count
            .checked_mul(response_dimension)
            .ok_or(PortfolioError::DimensionOverflow)?;
        let product_dimension = response_dimension
            .checked_mul(covariate_dimension)
            .ok_or(PortfolioError::DimensionOverflow)?;
        if contextual_dimension > MAX_MEMBER_FEATURES || product_dimension > MAX_MEMBER_FEATURES {
            return Err(PortfolioError::MemberFeatureLimitExceeded);
        }
        if contextual_target.dimension() != contextual_dimension
            || product_target.dimension() != product_dimension
        {
            return Err(PortfolioError::TargetDimensionMismatch);
        }
        validate_digest(&context_inventory_sha256, "context_inventory_sha256")?;
        validate_digest(&portfolio_contract_sha256, "portfolio_contract_sha256")?;
        validate_digest(&contextual_target_sha256, "contextual_target_sha256")?;
        validate_digest(&product_target_sha256, "product_target_sha256")?;
        validate_digest(&response_reference_sha256, "response_reference_sha256")?;
        validate_digest(&covariate_reference_sha256, "covariate_reference_sha256")?;

        let contextual_monitor = ConvexTargetApproachabilityMonitor::new(
            contextual_target,
            1.0,
            alpha_numerator,
            alpha_denominator,
        )?;
        let product_monitor = ConvexTargetApproachabilityMonitor::new(
            product_target,
            1.0,
            alpha_numerator,
            alpha_denominator,
        )?;
        let log_mixture_e = log_mixture(
            contextual_monitor.log_e_value(),
            product_monitor.log_e_value(),
        )?;
        let log_threshold = (f64::from(alpha_denominator) / f64::from(alpha_numerator)).ln();
        if !log_threshold.is_finite() {
            return Err(PortfolioError::ArithmeticOverflow);
        }

        Ok(Self {
            response_map,
            covariate_map,
            context_count,
            context_inventory_sha256,
            portfolio_contract_sha256,
            contextual_target_sha256,
            product_target_sha256,
            response_reference_sha256,
            covariate_reference_sha256,
            alpha_numerator,
            alpha_denominator,
            observations_seen: 0,
            contextual_monitor,
            product_monitor,
            seen_pair_ids: HashSet::new(),
            log_mixture_e,
            quarantined: false,
        })
    }

    /// Adds one paired response/covariate record to both fixed members.
    ///
    /// The context ID must be assigned from pre-response covariates only.
    /// The two compact CMAQ-046 states are cloned and staged; pair IDs are
    /// stored only once in this portfolio, avoiding per-round cloning of
    /// unbounded member replay inventories. No member or portfolio state is
    /// committed unless both active-member updates succeed.
    pub fn observe_pair(
        &mut self,
        sequence: u64,
        pair_id: &str,
        context_id: usize,
        covariate: &[f64],
        x: &[f64],
        y: &[f64],
    ) -> Result<RepresentationPortfolioUpdate, PortfolioError> {
        if self.quarantined {
            return Err(PortfolioError::AlreadyQuarantined);
        }
        if pair_id.trim().is_empty() {
            return Err(PortfolioError::EmptyPairId);
        }
        if pair_id.len() > MAX_PAIR_ID_BYTES {
            return Err(PortfolioError::PairIdTooLong);
        }
        if self.seen_pair_ids.contains(pair_id) {
            return Err(PortfolioError::DuplicatePairId);
        }
        if self.seen_pair_ids.len() >= MAX_PAIR_IDS {
            return Err(PortfolioError::PairIdInventoryFull);
        }
        if context_id >= self.context_count {
            return Err(PortfolioError::UnknownContext);
        }
        let expected_sequence = self
            .observations_seen
            .checked_add(1)
            .ok_or(PortfolioError::ArithmeticOverflow)?;
        if sequence != expected_sequence {
            return Err(PortfolioError::SequenceMismatch);
        }

        // All transformations and deterministic norm checks occur before
        // staging either member's e-process state.
        let response_x = self.response_map.transform(x)?;
        let response_y = self.response_map.transform(y)?;
        let covariate_features = self.covariate_map.transform(covariate)?;
        let response_dimension = self.response_map.feature_dimension();
        let covariate_dimension = self.covariate_map.feature_dimension();

        let mut contextual_difference = vec![0.0; self.context_count * response_dimension];
        let context_offset = context_id * response_dimension;
        for (index, (y_feature, x_feature)) in response_y.iter().zip(&response_x).enumerate() {
            contextual_difference[context_offset + index] = 0.5 * (y_feature - x_feature);
        }

        let mut product_difference = vec![0.0; response_dimension * covariate_dimension];
        for (covariate_index, covariate_feature) in covariate_features.iter().enumerate() {
            for (response_index, (y_feature, x_feature)) in
                response_y.iter().zip(&response_x).enumerate()
            {
                let index = covariate_index * response_dimension + response_index;
                product_difference[index] = 0.5 * covariate_feature * (y_feature - x_feature);
            }
        }

        if l2_norm(&contextual_difference) > 1.0 + NORM_TOLERANCE
            || l2_norm(&product_difference) > 1.0 + NORM_TOLERANCE
        {
            return Err(PortfolioError::ObservationNormExceeded);
        }

        // Stage only the compact O(d) CMAQ-046 monitor states. Replay IDs live
        // once in the portfolio and are inserted only after all updates pass.
        let mut next_contextual = self.contextual_monitor.clone();
        let mut next_product = self.product_monitor.clone();
        let mut contextual_update = None;
        let mut product_update = None;

        if !next_contextual.is_quarantined() {
            contextual_update = Some(next_contextual.observe(&contextual_difference)?);
        }
        if !next_product.is_quarantined() {
            product_update = Some(next_product.observe(&product_difference)?);
        }

        let contextual_log_e = next_contextual.log_e_value();
        let product_log_e = next_product.log_e_value();
        let next_log_mixture = log_mixture(contextual_log_e, product_log_e)?;
        let log_threshold =
            (f64::from(self.alpha_denominator) / f64::from(self.alpha_numerator)).ln();
        if !next_log_mixture.is_finite() || !log_threshold.is_finite() {
            return Err(PortfolioError::ArithmeticOverflow);
        }
        let quarantined = next_log_mixture >= log_threshold;

        // Atomic portfolio commit.
        self.contextual_monitor = next_contextual;
        self.product_monitor = next_product;
        self.observations_seen = expected_sequence;
        self.seen_pair_ids.insert(pair_id.to_string());
        self.log_mixture_e = next_log_mixture;
        self.quarantined = quarantined;

        Ok(RepresentationPortfolioUpdate {
            sequence,
            pair_id: pair_id.to_string(),
            context_id,
            contextual_member_stopped: self.contextual_monitor.is_quarantined(),
            product_member_stopped: self.product_monitor.is_quarantined(),
            contextual_member_log_e: contextual_log_e,
            product_member_log_e: product_log_e,
            contextual_member_update: contextual_update,
            product_member_update: product_update,
            mixture_log_e: next_log_mixture,
            log_threshold,
            quarantined,
        })
    }

    pub fn response_map(&self) -> &FourierFeatureMap {
        &self.response_map
    }

    pub fn covariate_map(&self) -> &FourierFeatureMap {
        &self.covariate_map
    }

    pub fn context_count(&self) -> usize {
        self.context_count
    }

    pub fn observations_seen(&self) -> u64 {
        self.observations_seen
    }

    pub fn alpha(&self) -> (u32, u32) {
        (self.alpha_numerator, self.alpha_denominator)
    }

    pub fn contextual_member_log_e(&self) -> f64 {
        self.contextual_monitor.log_e_value()
    }

    pub fn product_member_log_e(&self) -> f64 {
        self.product_monitor.log_e_value()
    }

    pub fn mixture_log_e(&self) -> f64 {
        self.log_mixture_e
    }

    pub fn log_threshold(&self) -> f64 {
        (f64::from(self.alpha_denominator) / f64::from(self.alpha_numerator)).ln()
    }

    pub fn contextual_member_stopped(&self) -> bool {
        self.contextual_monitor.is_quarantined()
    }

    pub fn product_member_stopped(&self) -> bool {
        self.product_monitor.is_quarantined()
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    pub fn context_inventory_sha256(&self) -> &str {
        &self.context_inventory_sha256
    }

    pub fn portfolio_contract_sha256(&self) -> &str {
        &self.portfolio_contract_sha256
    }

    pub fn contextual_target_sha256(&self) -> &str {
        &self.contextual_target_sha256
    }

    pub fn product_target_sha256(&self) -> &str {
        &self.product_target_sha256
    }

    pub fn response_reference_sha256(&self) -> &str {
        &self.response_reference_sha256
    }

    pub fn covariate_reference_sha256(&self) -> &str {
        &self.covariate_reference_sha256
    }
}

fn log_mixture(left: f64, right: f64) -> Result<f64, PortfolioError> {
    if !left.is_finite() || !right.is_finite() {
        return Err(PortfolioError::ArithmeticOverflow);
    }
    let maximum = left.max(right);
    let value = maximum + ((left - maximum).exp() + (right - maximum).exp()).ln() + LOG_HALF;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PortfolioError::ArithmeticOverflow)
    }
}

fn validate_digest(value: &str, label: &'static str) -> Result<(), PortfolioError> {
    if value.len() == DIGEST_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(PortfolioError::InvalidDigest(label))
    }
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |norm, value| norm.hypot(*value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const SHA_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn feature_map(digest: &str) -> FourierFeatureMap {
        FourierFeatureMap::new(1, vec![vec![1.0], vec![3.0]], digest.into()).unwrap()
    }

    fn portfolio(
        contextual_target: BoxTarget,
        product_target: BoxTarget,
    ) -> RepresentationEPortfolio {
        RepresentationEPortfolio::new(
            feature_map(SHA_A),
            feature_map(SHA_B),
            2,
            contextual_target,
            product_target,
            1,
            20,
            SHA_C.into(),
            SHA_D.into(),
            SHA_E.into(),
            SHA_F.into(),
            SHA_A.into(),
            SHA_B.into(),
        )
        .unwrap()
    }

    fn narrow_context_target() -> BoxTarget {
        BoxTarget::new(vec![-0.005; 8], vec![0.005; 8]).unwrap()
    }

    fn narrow_product_target() -> BoxTarget {
        BoxTarget::new(vec![-0.005; 16], vec![0.005; 16]).unwrap()
    }

    #[test]
    fn initial_equal_weight_log_mixture_is_exactly_zero() {
        let portfolio = portfolio(narrow_context_target(), narrow_product_target());
        assert!((portfolio.mixture_log_e()).abs() < 1e-12);
        assert_eq!(portfolio.alpha(), (1, 20));
        assert_eq!(portfolio.observations_seen(), 0);
    }

    #[test]
    fn mixture_log_matches_equal_arithmetic_e_value_average() {
        for (left, right) in [(0.0, 0.0), (2.0, -3.0), (12.0, 9.0), (-8.0, -4.0)] {
            let mixed = log_mixture(left, right).unwrap();
            let maximum = left.max(right);
            let expected =
                maximum + (((left - maximum).exp() + (right - maximum).exp()) / 2.0).ln();
            assert!((mixed - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn both_fixed_representations_receive_each_valid_pair_once() {
        let mut portfolio = portfolio(narrow_context_target(), narrow_product_target());
        let update = portfolio
            .observe_pair(1, "pair-1", 0, &[0.2], &[0.2], &[0.2])
            .unwrap();
        assert!(update.contextual_member_update.is_some());
        assert!(update.product_member_update.is_some());
        assert_eq!(portfolio.observations_seen(), 1);
        assert_eq!(update.mixture_log_e, portfolio.mixture_log_e());
    }

    #[test]
    fn equal_marginal_conditional_mismatch_is_detected_by_fixed_mixture() {
        let mut portfolio = portfolio(narrow_context_target(), narrow_product_target());
        let mut quarantined_at = None;
        for sequence in 1..=300u64 {
            let context = (sequence as usize - 1) % 2;
            let (covariate, x, y) = if context == 0 {
                (0.2, 0.2, 0.8)
            } else {
                (-0.2, 0.8, 0.2)
            };
            let update = portfolio
                .observe_pair(
                    sequence,
                    &format!("pair-{sequence}"),
                    context,
                    &[covariate],
                    &[x],
                    &[y],
                )
                .unwrap();
            if update.quarantined {
                quarantined_at = Some(sequence);
                break;
            }
        }
        assert!(quarantined_at.is_some());
        assert!(portfolio.is_quarantined());
        assert!(portfolio.mixture_log_e() >= portfolio.log_threshold());
    }

    #[test]
    fn stopped_member_freezes_while_other_member_continues() {
        let broad_product_target = BoxTarget::new(vec![-1.0; 16], vec![1.0; 16]).unwrap();
        let mut portfolio = portfolio(narrow_context_target(), broad_product_target);
        let mut context_stopped_at = None;

        for sequence in 1..=300u64 {
            let context = (sequence as usize - 1) % 2;
            let (covariate, x, y) = if context == 0 {
                (0.2, 0.2, 0.8)
            } else {
                (-0.2, 0.8, 0.2)
            };
            let update = portfolio
                .observe_pair(
                    sequence,
                    &format!("pair-{sequence}"),
                    context,
                    &[covariate],
                    &[x],
                    &[y],
                )
                .unwrap();
            if update.contextual_member_stopped && context_stopped_at.is_none() {
                context_stopped_at = Some(sequence);
            }
            if context_stopped_at.is_some() && sequence > context_stopped_at.unwrap() {
                assert!(update.contextual_member_update.is_none());
                assert!(update.product_member_update.is_some());
                break;
            }
        }

        assert!(context_stopped_at.is_some());
        assert!(portfolio.contextual_member_stopped());
        assert!(!portfolio.product_member_stopped());
        assert!(portfolio.observations_seen() > context_stopped_at.unwrap());
    }

    #[test]
    fn duplicate_pair_wrong_sequence_and_unknown_context_do_not_mutate_state() {
        let mut portfolio = portfolio(narrow_context_target(), narrow_product_target());
        portfolio
            .observe_pair(1, "pair-1", 0, &[0.2], &[0.2], &[0.2])
            .unwrap();
        let before = portfolio.clone();
        assert_eq!(
            portfolio.observe_pair(2, "pair-1", 0, &[0.2], &[0.2], &[0.2]),
            Err(PortfolioError::DuplicatePairId)
        );
        assert_eq!(
            portfolio.observe_pair(3, "pair-3", 0, &[0.2], &[0.2], &[0.2]),
            Err(PortfolioError::SequenceMismatch)
        );
        assert_eq!(
            portfolio.observe_pair(2, "pair-2", 2, &[0.2], &[0.2], &[0.2]),
            Err(PortfolioError::UnknownContext)
        );
        assert_eq!(portfolio, before);
    }

    #[test]
    fn invalid_pair_inputs_fail_before_either_member_commits() {
        let mut portfolio = portfolio(narrow_context_target(), narrow_product_target());
        assert!(matches!(
            portfolio.observe_pair(1, "pair-1", 0, &[f64::NAN], &[0.2], &[0.2]),
            Err(PortfolioError::FeatureMap(KernelError::NonFiniteValue))
        ));
        assert_eq!(portfolio.observations_seen(), 0);
        assert_eq!(portfolio.mixture_log_e(), 0.0);
    }

    #[test]
    fn malformed_bindings_and_target_dimensions_fail_closed() {
        assert_eq!(
            RepresentationEPortfolio::new(
                feature_map(SHA_A),
                feature_map(SHA_B),
                2,
                BoxTarget::new(vec![-0.1; 7], vec![0.1; 7]).unwrap(),
                narrow_product_target(),
                1,
                20,
                SHA_C.into(),
                SHA_D.into(),
                SHA_E.into(),
                SHA_F.into(),
                SHA_A.into(),
                SHA_B.into(),
            )
            .err(),
            Some(PortfolioError::TargetDimensionMismatch)
        );
        assert_eq!(
            validate_digest("bad", "portfolio_contract_sha256"),
            Err(PortfolioError::InvalidDigest("portfolio_contract_sha256"))
        );
    }

    #[test]
    fn portfolio_quarantine_is_permanent() {
        let mut portfolio = portfolio(narrow_context_target(), narrow_product_target());
        for sequence in 1..=300u64 {
            let context = (sequence as usize - 1) % 2;
            let (covariate, x, y) = if context == 0 {
                (0.2, 0.2, 0.8)
            } else {
                (-0.2, 0.8, 0.2)
            };
            let update = portfolio
                .observe_pair(
                    sequence,
                    &format!("pair-{sequence}"),
                    context,
                    &[covariate],
                    &[x],
                    &[y],
                )
                .unwrap();
            if update.quarantined {
                break;
            }
        }
        assert!(portfolio.is_quarantined());
        assert_eq!(
            portfolio.observe_pair(999, "late", 0, &[0.2], &[0.2], &[0.2]),
            Err(PortfolioError::AlreadyQuarantined)
        );
    }
}
