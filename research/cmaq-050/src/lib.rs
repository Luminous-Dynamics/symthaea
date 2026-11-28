#![forbid(unsafe_code)]

//! Continuous-covariate product Fourier paired conditional discrepancy.
//!
//! Let phi_X map responses and psi_Z map shared covariates using frozen
//! unit-norm Fourier features. Each pair produces
//!
//!   r_t = 0.5 * psi_Z(Z_t) ⊗ [phi_X(Y_t) - phi_X(X_t)].
//!
//! Since both feature maps have norm one, ||r_t||_2 <= 1. Under the
//! conditional distribution null, E[phi_X(Y_t)-phi_X(X_t) | Z_t,F_(t-1)] = 0.
//! Iterated expectation therefore gives E[r_t | F_(t-1)] = 0. The bounded
//! vector is passed to CMAQ-046's predictable OCO/e-process monitor.
//!
//! This is a finite product-feature approximation, not a reproduction of
//! SCPT's nearest-neighbor conditional MMD estimator.

use std::collections::HashSet;

use cmaq_046_convex_target_approachability::{
    ApproachabilityError, ApproachabilityUpdate, BoxTarget, ConvexTargetApproachabilityMonitor,
};
use cmaq_048_fourier_kernel_approachability::{FourierFeatureMap, KernelError};

const MAX_PRODUCT_FEATURES: usize = 1_024;
const MAX_PAIR_ID_BYTES: usize = 256;
const MAX_PAIR_IDS: usize = 1_000_000;
const NORM_TOLERANCE: f64 = 1e-12;
const DIGEST_HEX_LEN: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProductPairError {
    DimensionOverflow,
    TooManyProductFeatures,
    TargetDimensionMismatch,
    InvalidDigest(&'static str),
    EmptyPairId,
    PairIdTooLong,
    DuplicatePairId,
    PairIdInventoryFull,
    SequenceMismatch,
    ArithmeticOverflow,
    ObservationNormExceeded,
    Monitor(ApproachabilityError),
    FeatureMap(KernelError),
}

impl From<ApproachabilityError> for ProductPairError {
    fn from(value: ApproachabilityError) -> Self {
        Self::Monitor(value)
    }
}

impl From<KernelError> for ProductPairError {
    fn from(value: KernelError) -> Self {
        Self::FeatureMap(value)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProductPairUpdate {
    pub sequence: u64,
    pub pair_id: String,
    /// 0.5 * psi_Z(z) tensor (phi_X(y)-phi_X(x)).
    pub product_feature_difference: Vec<f64>,
    pub evidence: ApproachabilityUpdate,
}

/// Paired conditional discrepancy monitor using a continuous covariate map.
///
/// The response and covariate Fourier maps are immutable. The covariate map
/// supplies smooth pre-response weighting functions in place of CMAQ-049's
/// hard context partition. Both frequency banks, the target and the reference
/// receipt must be frozen before test data is observed.
#[derive(Clone, Debug, PartialEq)]
pub struct ProductFourierConditionalMonitor {
    response_map: FourierFeatureMap,
    covariate_map: FourierFeatureMap,
    target_sha256: String,
    response_reference_sha256: String,
    covariate_reference_sha256: String,
    monitor: ConvexTargetApproachabilityMonitor,
    seen_pair_ids: HashSet<String>,
}

impl ProductFourierConditionalMonitor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        response_map: FourierFeatureMap,
        covariate_map: FourierFeatureMap,
        target: BoxTarget,
        alpha_numerator: u32,
        alpha_denominator: u32,
        target_sha256: String,
        response_reference_sha256: String,
        covariate_reference_sha256: String,
    ) -> Result<Self, ProductPairError> {
        let product_dimension = response_map
            .feature_dimension()
            .checked_mul(covariate_map.feature_dimension())
            .ok_or(ProductPairError::DimensionOverflow)?;
        if product_dimension > MAX_PRODUCT_FEATURES {
            return Err(ProductPairError::TooManyProductFeatures);
        }
        if target.dimension() != product_dimension {
            return Err(ProductPairError::TargetDimensionMismatch);
        }
        validate_digest(&target_sha256, "target_sha256")?;
        validate_digest(
            &response_reference_sha256,
            "response_reference_sha256",
        )?;
        validate_digest(
            &covariate_reference_sha256,
            "covariate_reference_sha256",
        )?;

        let monitor = ConvexTargetApproachabilityMonitor::new(
            target,
            1.0,
            alpha_numerator,
            alpha_denominator,
        )?;
        Ok(Self {
            response_map,
            covariate_map,
            target_sha256,
            response_reference_sha256,
            covariate_reference_sha256,
            monitor,
            seen_pair_ids: HashSet::new(),
        })
    }

    /// Adds one paired observation. The shared covariate may be random at time
    /// t, but must be available before either response is evaluated by the
    /// paired-monitoring protocol. A successful pair ID is consumed once in
    /// this process; use a durable external ledger across process restarts.
    pub fn observe_pair(
        &mut self,
        sequence: u64,
        pair_id: &str,
        covariate: &[f64],
        x: &[f64],
        y: &[f64],
    ) -> Result<ProductPairUpdate, ProductPairError> {
        if pair_id.trim().is_empty() {
            return Err(ProductPairError::EmptyPairId);
        }
        if pair_id.len() > MAX_PAIR_ID_BYTES {
            return Err(ProductPairError::PairIdTooLong);
        }
        if self.seen_pair_ids.contains(pair_id) {
            return Err(ProductPairError::DuplicatePairId);
        }
        if self.seen_pair_ids.len() >= MAX_PAIR_IDS {
            return Err(ProductPairError::PairIdInventoryFull);
        }
        let expected_sequence = self
            .monitor
            .observations_seen()
            .checked_add(1)
            .ok_or(ProductPairError::ArithmeticOverflow)?;
        if sequence != expected_sequence {
            return Err(ProductPairError::SequenceMismatch);
        }

        let covariate_features = self.covariate_map.transform(covariate)?;
        let x_features = self.response_map.transform(x)?;
        let y_features = self.response_map.transform(y)?;
        let response_dimension = self.response_map.feature_dimension();
        let covariate_dimension = self.covariate_map.feature_dimension();
        let mut product_feature_difference =
            vec![0.0; response_dimension * covariate_dimension];

        for covariate_index in 0..covariate_dimension {
            for response_index in 0..response_dimension {
                let product_index =
                    covariate_index * response_dimension + response_index;
                product_feature_difference[product_index] = 0.5
                    * covariate_features[covariate_index]
                    * (y_features[response_index] - x_features[response_index]);
            }
        }
        let norm = l2_norm(&product_feature_difference);
        if !norm.is_finite() {
            return Err(ProductPairError::ArithmeticOverflow);
        }
        if norm > 1.0 + NORM_TOLERANCE {
            return Err(ProductPairError::ObservationNormExceeded);
        }

        // The CMAQ-046 update is transactional. Consume the pair ID only after
        // the evidence update succeeds.
        let evidence = self.monitor.observe(&product_feature_difference)?;
        self.seen_pair_ids.insert(pair_id.to_string());

        Ok(ProductPairUpdate {
            sequence,
            pair_id: pair_id.to_string(),
            product_feature_difference,
            evidence,
        })
    }

    pub fn response_map(&self) -> &FourierFeatureMap {
        &self.response_map
    }

    pub fn covariate_map(&self) -> &FourierFeatureMap {
        &self.covariate_map
    }

    pub fn product_feature_dimension(&self) -> usize {
        self.response_map.feature_dimension() * self.covariate_map.feature_dimension()
    }

    pub fn target_sha256(&self) -> &str {
        &self.target_sha256
    }

    pub fn response_reference_sha256(&self) -> &str {
        &self.response_reference_sha256
    }

    pub fn covariate_reference_sha256(&self) -> &str {
        &self.covariate_reference_sha256
    }

    pub fn observations_seen(&self) -> u64 {
        self.monitor.observations_seen()
    }

    pub fn log_e_value(&self) -> f64 {
        self.monitor.log_e_value()
    }

    pub fn is_quarantined(&self) -> bool {
        self.monitor.is_quarantined()
    }

    pub fn mean_product_feature_difference(
        &self,
    ) -> Result<Vec<f64>, ApproachabilityError> {
        self.monitor.mean_observation()
    }

    pub fn distance_to_null_target(&self) -> Result<f64, ApproachabilityError> {
        self.monitor.distance_to_target()
    }
}

fn validate_digest(value: &str, label: &'static str) -> Result<(), ProductPairError> {
    if value.len() == DIGEST_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(ProductPairError::InvalidDigest(label))
    }
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |norm, value| norm.hypot(*value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESPONSE_MAP_SHA: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const COVARIATE_MAP_SHA: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const TARGET_SHA: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const RESPONSE_REFERENCE_SHA: &str =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const COVARIATE_REFERENCE_SHA: &str =
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn response_map() -> FourierFeatureMap {
        FourierFeatureMap::new(
            1,
            vec![vec![1.0], vec![3.0]],
            RESPONSE_MAP_SHA.into(),
        )
        .unwrap()
    }

    fn covariate_map() -> FourierFeatureMap {
        FourierFeatureMap::new(1, vec![vec![1.0], vec![3.0]], COVARIATE_MAP_SHA.into()).unwrap()
    }

    fn target() -> BoxTarget {
        BoxTarget::new(vec![-0.005; 16], vec![0.005; 16]).unwrap()
    }

    fn monitor() -> ProductFourierConditionalMonitor {
        ProductFourierConditionalMonitor::new(
            response_map(),
            covariate_map(),
            target(),
            1,
            20,
            TARGET_SHA.into(),
            RESPONSE_REFERENCE_SHA.into(),
            COVARIATE_REFERENCE_SHA.into(),
        )
        .unwrap()
    }

    #[test]
    fn product_feature_observations_are_bounded_by_one() {
        let mut monitor = monitor();
        for (sequence, (z, x, y)) in [
            (0.2, 0.2, 0.8),
            (-0.2, 0.8, 0.2),
            (0.9, -0.4, 0.4),
            (-0.9, 0.4, -0.4),
            (0.0, 0.0, 0.0),
        ]
        .into_iter()
        .enumerate()
        {
            let update = monitor
                .observe_pair(sequence as u64 + 1, &format!("pair-{sequence}"), &[z], &[x], &[y])
                .unwrap();
            assert!(l2_norm(&update.product_feature_difference) <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn identical_responses_produce_zero_even_with_changing_covariates() {
        let mut monitor = monitor();
        for (index, covariate) in [0.1, -0.7, 0.9, -0.2].into_iter().enumerate() {
            let update = monitor
                .observe_pair(
                    index as u64 + 1,
                    &format!("pair-{index}"),
                    &[covariate],
                    &[0.2],
                    &[0.2],
                )
                .unwrap();
            assert!(update
                .product_feature_difference
                .iter()
                .all(|value| value.abs() < 1e-12));
        }
        assert_eq!(monitor.log_e_value(), 0.0);
    }

    #[test]
    fn pair_reversal_negates_product_feature_vector() {
        let mut forward = monitor();
        let mut reverse = monitor();
        let a = forward
            .observe_pair(1, "forward", &[0.3], &[0.2], &[0.8])
            .unwrap();
        let b = reverse
            .observe_pair(1, "reverse", &[0.3], &[0.8], &[0.2])
            .unwrap();
        for (left, right) in a
            .product_feature_difference
            .iter()
            .zip(&b.product_feature_difference)
        {
            assert!((left + right).abs() < 1e-12);
        }
    }

    #[test]
    fn conditional_shift_with_equal_marginals_accumulates_evidence() {
        let mut monitor = monitor();
        let mut quarantined_at = None;
        for index in 0..400 {
            let (z, x, y) = if index % 2 == 0 {
                (0.2, 0.2, 0.8)
            } else {
                (-0.2, 0.8, 0.2)
            };
            let update = monitor
                .observe_pair(index as u64 + 1, &format!("pair-{index}"), &[z], &[x], &[y])
                .unwrap();
            if update.evidence.quarantined {
                quarantined_at = Some(index + 1);
                break;
            }
        }
        assert!(quarantined_at.is_some());
        assert!(monitor.is_quarantined());
    }

    #[test]
    fn pair_sequence_and_duplicate_identity_are_fail_closed() {
        let mut monitor = monitor();
        monitor
            .observe_pair(1, "pair-1", &[0.2], &[0.2], &[0.2])
            .unwrap();
        let before = monitor.clone();
        assert_eq!(
            monitor.observe_pair(2, "pair-1", &[0.2], &[0.2], &[0.2]),
            Err(ProductPairError::DuplicatePairId)
        );
        assert_eq!(
            monitor.observe_pair(4, "pair-4", &[0.2], &[0.2], &[0.2]),
            Err(ProductPairError::SequenceMismatch)
        );
        assert_eq!(monitor, before);
    }

    #[test]
    fn malformed_maps_digests_and_dimensions_fail_closed() {
        assert_eq!(
            ProductFourierConditionalMonitor::new(
                response_map(),
                covariate_map(),
                BoxTarget::new(vec![-0.1; 15], vec![0.1; 15]).unwrap(),
                1,
                20,
                TARGET_SHA.into(),
                RESPONSE_REFERENCE_SHA.into(),
                COVARIATE_REFERENCE_SHA.into(),
            )
            .err(),
            Some(ProductPairError::TargetDimensionMismatch)
        );
        assert_eq!(
            ProductFourierConditionalMonitor::new(
                response_map(),
                covariate_map(),
                target(),
                1,
                20,
                TARGET_SHA.into(),
                "bad".into(),
                COVARIATE_REFERENCE_SHA.into(),
            )
            .err(),
            Some(ProductPairError::InvalidDigest("response_reference_sha256"))
        );
    }

    #[test]
    fn product_dimension_limit_is_enforced() {
        let response = FourierFeatureMap::new(
            1,
            (0..16).map(|index| vec![index as f64 + 1.0]).collect(),
            RESPONSE_MAP_SHA.into(),
        )
        .unwrap();
        let covariates = FourierFeatureMap::new(
            1,
            (0..16).map(|index| vec![index as f64 + 1.0]).collect(),
            COVARIATE_MAP_SHA.into(),
        )
        .unwrap();
        let result = ProductFourierConditionalMonitor::new(
            response,
            covariates,
            BoxTarget::new(vec![-0.1; 1024], vec![0.1; 1024]).unwrap(),
            1,
            20,
            TARGET_SHA.into(),
            RESPONSE_REFERENCE_SHA.into(),
            COVARIATE_REFERENCE_SHA.into(),
        );
        assert_eq!(result.err(), Some(ProductPairError::TooManyProductFeatures));
    }
}
