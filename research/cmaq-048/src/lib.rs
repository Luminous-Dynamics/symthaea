#![forbid(unsafe_code)]

//! Fixed Fourier-feature kernel mean embeddings with anytime-valid approachability evidence.
//!
//! A fixed bank of frequencies maps x to
//!
//!   phi(x) = M^(-1/2) [cos(w_1'x), sin(w_1'x), ..., cos(w_M'x), sin(w_M'x)].
//!
//! Every embedding has exact mathematical L2 norm one. The frozen BoxTarget
//! therefore describes an allowed conditional-mean feature-embedding region.
//! CMAQ-046 monitors these bounded vectors. When frequencies are sampled from
//! the Gaussian spectral measure, phi approximates a Gaussian RBF kernel map.
//!
//! The statistical guarantee is exact for the finite registered feature map
//! and the conditional-mean target contract. The map is only a finite
//! approximation to an infinite-dimensional characteristic kernel.

use cmaq_046_convex_target_approachability::{
    ApproachabilityError, ApproachabilityUpdate, BoxTarget, ConvexTargetApproachabilityMonitor,
};

const DIGEST_HEX_LEN: usize = 64;
const MAX_INPUT_DIMENSION: usize = 1_024;
const MAX_FREQUENCIES: usize = 512;
const INPUT_NORM_TOLERANCE: f64 = 1e-12;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelError {
    EmptyFrequencyBank,
    FeatureBankTooLarge,
    DimensionMismatch,
    InvalidDigest(&'static str),
    InvalidInputNormBound,
    NonFiniteValue,
    InputNormExceeded,
    ArithmeticOverflow,
    Monitor(ApproachabilityError),
}

impl From<ApproachabilityError> for KernelError {
    fn from(value: ApproachabilityError) -> Self {
        Self::Monitor(value)
    }
}

/// A fixed, externally generated and digested finite Fourier frequency bank.
///
/// The caller supplies the frequencies directly instead of relying on an
/// implicit PRNG or in-crate Gaussian sampler. For an RBF approximation,
/// independently sample each frequency from N(0, ell^-2 I) using the
/// pre-registered bandwidth ell, then freeze and digest the resulting matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct FourierFeatureMap {
    input_dimension: usize,
    frequencies: Vec<Vec<f64>>,
    map_sha256: String,
    inverse_sqrt_frequency_count: f64,
}

impl FourierFeatureMap {
    pub fn new(
        input_dimension: usize,
        frequencies: Vec<Vec<f64>>,
        map_sha256: String,
    ) -> Result<Self, KernelError> {
        if input_dimension == 0 || input_dimension > MAX_INPUT_DIMENSION {
            return Err(KernelError::DimensionMismatch);
        }
        if frequencies.is_empty() {
            return Err(KernelError::EmptyFrequencyBank);
        }
        if frequencies.len() > MAX_FREQUENCIES {
            return Err(KernelError::FeatureBankTooLarge);
        }
        validate_digest(&map_sha256, "map_sha256")?;

        for frequency in &frequencies {
            if frequency.len() != input_dimension {
                return Err(KernelError::DimensionMismatch);
            }
            if frequency.iter().any(|value| !value.is_finite()) {
                return Err(KernelError::NonFiniteValue);
            }
        }

        let inverse_sqrt_frequency_count = 1.0 / (frequencies.len() as f64).sqrt();
        if !inverse_sqrt_frequency_count.is_finite() || inverse_sqrt_frequency_count <= 0.0 {
            return Err(KernelError::ArithmeticOverflow);
        }

        Ok(Self {
            input_dimension,
            frequencies,
            map_sha256,
            inverse_sqrt_frequency_count,
        })
    }

    pub fn input_dimension(&self) -> usize {
        self.input_dimension
    }

    pub fn frequency_count(&self) -> usize {
        self.frequencies.len()
    }

    pub fn feature_dimension(&self) -> usize {
        self.frequencies.len() * 2
    }

    pub fn map_sha256(&self) -> &str {
        &self.map_sha256
    }

    pub fn frequencies(&self) -> &[Vec<f64>] {
        &self.frequencies
    }

    /// Computes the paired sine/cosine embedding with norm one.
    pub fn transform(&self, input: &[f64]) -> Result<Vec<f64>, KernelError> {
        if input.len() != self.input_dimension {
            return Err(KernelError::DimensionMismatch);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(KernelError::NonFiniteValue);
        }

        let mut features = Vec::with_capacity(self.feature_dimension());
        for frequency in &self.frequencies {
            let phase = dot(frequency, input)?;
            let (sine, cosine) = phase.sin_cos();
            features.push(cosine * self.inverse_sqrt_frequency_count);
            features.push(sine * self.inverse_sqrt_frequency_count);
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err(KernelError::ArithmeticOverflow);
        }
        Ok(features)
    }

    /// Approximate Gaussian-kernel inner product for two inputs under this
    /// fixed frequency bank.
    pub fn approximate_rbf_kernel(&self, left: &[f64], right: &[f64]) -> Result<f64, KernelError> {
        let left_features = self.transform(left)?;
        let right_features = self.transform(right)?;
        dot(&left_features, &right_features)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelApproachabilityUpdate {
    pub raw_input: Vec<f64>,
    pub feature_embedding: Vec<f64>,
    pub evidence: ApproachabilityUpdate,
}

/// Online kernel mean embedding monitor backed by CMAQ-046.
///
/// The feature map and target are immutable. The target must describe an
/// independently justified envelope for E[phi(X_t) | F_(t-1)] under the null.
/// Digests are structurally checked; cryptographic authentication and the
/// truth of the null envelope remain external responsibilities.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelMeanApproachabilityMonitor {
    feature_map: FourierFeatureMap,
    input_norm_bound: f64,
    target_sha256: String,
    null_reference_receipt_sha256: String,
    monitor: ConvexTargetApproachabilityMonitor,
}

impl KernelMeanApproachabilityMonitor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        feature_map: FourierFeatureMap,
        target: BoxTarget,
        input_norm_bound: f64,
        alpha_numerator: u32,
        alpha_denominator: u32,
        target_sha256: String,
        null_reference_receipt_sha256: String,
    ) -> Result<Self, KernelError> {
        if target.dimension() != feature_map.feature_dimension() {
            return Err(KernelError::DimensionMismatch);
        }
        if !input_norm_bound.is_finite() || input_norm_bound <= 0.0 {
            return Err(KernelError::InvalidInputNormBound);
        }
        validate_digest(&target_sha256, "target_sha256")?;
        validate_digest(
            &null_reference_receipt_sha256,
            "null_reference_receipt_sha256",
        )?;
        let monitor = ConvexTargetApproachabilityMonitor::new(
            target,
            1.0,
            alpha_numerator,
            alpha_denominator,
        )?;

        Ok(Self {
            feature_map,
            input_norm_bound,
            target_sha256,
            null_reference_receipt_sha256,
            monitor,
        })
    }

    /// Transforms and observes one raw input. Validation or transform failure
    /// occurs before the CMAQ-046 evidence state is mutated.
    pub fn observe(&mut self, input: &[f64]) -> Result<KernelApproachabilityUpdate, KernelError> {
        if input.len() != self.feature_map.input_dimension() {
            return Err(KernelError::DimensionMismatch);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(KernelError::NonFiniteValue);
        }
        if l2_norm(input) > self.input_norm_bound + INPUT_NORM_TOLERANCE {
            return Err(KernelError::InputNormExceeded);
        }

        let feature_embedding = self.feature_map.transform(input)?;
        let evidence = self.monitor.observe(&feature_embedding)?;
        Ok(KernelApproachabilityUpdate {
            raw_input: input.to_vec(),
            feature_embedding,
            evidence,
        })
    }

    pub fn feature_map(&self) -> &FourierFeatureMap {
        &self.feature_map
    }

    pub fn target_sha256(&self) -> &str {
        &self.target_sha256
    }

    pub fn null_reference_receipt_sha256(&self) -> &str {
        &self.null_reference_receipt_sha256
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

    pub fn mean_feature_embedding(&self) -> Result<Vec<f64>, ApproachabilityError> {
        self.monitor.mean_observation()
    }

    pub fn distance_to_null_embedding_target(&self) -> Result<f64, ApproachabilityError> {
        self.monitor.distance_to_target()
    }

    /// Signed pathwise OCO regret from the inner approachability monitor.
    pub fn realized_regret(&self) -> Result<f64, ApproachabilityError> {
        self.monitor.realized_regret()
    }
}

fn validate_digest(value: &str, label: &'static str) -> Result<(), KernelError> {
    if value.len() == DIGEST_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(KernelError::InvalidDigest(label))
    }
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, KernelError> {
    if left.len() != right.len() {
        return Err(KernelError::DimensionMismatch);
    }
    let mut result = 0.0;
    for (a, b) in left.iter().zip(right) {
        result += a * b;
    }
    if result.is_finite() {
        Ok(result)
    } else {
        Err(KernelError::ArithmeticOverflow)
    }
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |norm, value| norm.hypot(*value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAP_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const TARGET_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const REFERENCE_SHA: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn map() -> FourierFeatureMap {
        FourierFeatureMap::new(1, vec![vec![1.0], vec![3.0]], MAP_SHA.into()).unwrap()
    }

    fn target() -> BoxTarget {
        let scale = 2.0f64.sqrt().recip();
        let means = [(0.2f64).cos() * scale, (0.6f64).cos() * scale];
        let tolerance = 0.01;
        BoxTarget::new(
            vec![
                means[0] - tolerance,
                -tolerance,
                means[1] - tolerance,
                -tolerance,
            ],
            vec![
                means[0] + tolerance,
                tolerance,
                means[1] + tolerance,
                tolerance,
            ],
        )
        .unwrap()
    }

    fn monitor() -> KernelMeanApproachabilityMonitor {
        KernelMeanApproachabilityMonitor::new(
            map(),
            target(),
            1.0,
            1,
            20,
            TARGET_SHA.into(),
            REFERENCE_SHA.into(),
        )
        .unwrap()
    }

    #[test]
    fn embedding_dimension_and_unit_norm_are_exact_up_to_roundoff() {
        let map = map();
        assert_eq!(map.feature_dimension(), 4);
        for input in [[0.0], [0.2], [-0.2], [0.8], [-1.0]] {
            let features = map.transform(&input).unwrap();
            assert_eq!(features.len(), 4);
            assert!((l2_norm(&features) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn symmetric_pair_has_exact_cosine_means_and_zero_sine_mean() {
        let map = map();
        let positive = map.transform(&[0.2]).unwrap();
        let negative = map.transform(&[-0.2]).unwrap();
        for index in [0, 2] {
            assert!((positive[index] - negative[index]).abs() < 1e-12);
        }
        for index in [1, 3] {
            assert!((positive[index] + negative[index]).abs() < 1e-12);
            assert!(positive[index].abs() < 1.0);
        }
        let scale = 2.0f64.sqrt().recip();
        assert!((positive[0] - 0.2f64.cos() * scale).abs() < 1e-12);
        assert!((positive[2] - 0.6f64.cos() * scale).abs() < 1e-12);
    }

    #[test]
    fn scale_change_preserving_raw_mean_changes_the_feature_mean() {
        let map = map();
        let null_positive = map.transform(&[0.2]).unwrap();
        let null_negative = map.transform(&[-0.2]).unwrap();
        let shifted_positive = map.transform(&[0.8]).unwrap();
        let shifted_negative = map.transform(&[-0.8]).unwrap();

        let null_mean = null_positive
            .iter()
            .zip(&null_negative)
            .map(|(left, right)| (left + right) / 2.0)
            .collect::<Vec<_>>();
        let shifted_mean = shifted_positive
            .iter()
            .zip(&shifted_negative)
            .map(|(left, right)| (left + right) / 2.0)
            .collect::<Vec<_>>();

        assert_eq!([0.2f64, -0.2f64].iter().sum::<f64>(), 0.0);
        assert_eq!([0.8f64, -0.8f64].iter().sum::<f64>(), 0.0);
        assert!(
            l2_norm(
                &null_mean
                    .iter()
                    .zip(&shifted_mean)
                    .map(|(left, right)| left - right)
                    .collect::<Vec<_>>()
            ) > 0.5
        );
        assert!(null_mean[1].abs() < 1e-12);
        assert!(shifted_mean[1].abs() < 1e-12);
        assert!(null_mean[3].abs() < 1e-12);
        assert!(shifted_mean[3].abs() < 1e-12);
    }

    #[test]
    fn approximate_rbf_inner_product_matches_registered_cosine_average() {
        let map = map();
        let kernel = map.approximate_rbf_kernel(&[0.2], &[-0.4]).unwrap();
        let expected = (0.6f64.cos() + 1.8f64.cos()) / 2.0;
        assert!((kernel - expected).abs() < 1e-12);
    }

    #[test]
    fn input_and_target_validation_fail_closed() {
        assert_eq!(
            FourierFeatureMap::new(1, Vec::new(), MAP_SHA.into()).err(),
            Some(KernelError::EmptyFrequencyBank)
        );
        assert_eq!(
            FourierFeatureMap::new(1, vec![vec![1.0, 2.0]], MAP_SHA.into()).err(),
            Some(KernelError::DimensionMismatch)
        );
        assert_eq!(
            FourierFeatureMap::new(1, vec![vec![f64::NAN]], MAP_SHA.into()).err(),
            Some(KernelError::NonFiniteValue)
        );
        assert_eq!(
            FourierFeatureMap::new(1, vec![vec![1.0]], "bad".into()).err(),
            Some(KernelError::InvalidDigest("map_sha256"))
        );
    }

    #[test]
    fn monitor_rejects_invalid_inputs_without_advancing_evidence() {
        let mut monitor = monitor();
        let before = monitor.clone();
        assert_eq!(
            monitor.observe(&[f64::NAN]),
            Err(KernelError::NonFiniteValue)
        );
        assert_eq!(
            monitor.observe(&[1.01]),
            Err(KernelError::InputNormExceeded)
        );
        assert_eq!(
            monitor.observe(&[0.0, 0.1]),
            Err(KernelError::DimensionMismatch)
        );
        assert_eq!(monitor, before);
    }

    #[test]
    fn persistent_scale_shift_accumulates_anytime_evidence() {
        let mut monitor = monitor();
        let mut quarantined_at = None;
        for index in 0..600 {
            let sample = if index % 2 == 0 { 0.8 } else { -0.8 };
            let update = monitor.observe(&[sample]).unwrap();
            if update.evidence.quarantined {
                quarantined_at = Some(index + 1);
                break;
            }
        }
        assert!(quarantined_at.is_some());
        assert!(monitor.is_quarantined());
        assert!(monitor.log_e_value() >= (20.0f64).ln());
    }

    #[test]
    fn null_reference_target_and_feature_map_bindings_are_immutable() {
        let monitor = monitor();
        assert_eq!(monitor.feature_map().map_sha256(), MAP_SHA);
        assert_eq!(monitor.target_sha256(), TARGET_SHA);
        assert_eq!(monitor.null_reference_receipt_sha256(), REFERENCE_SHA);
        assert_eq!(monitor.feature_map().frequencies(), &[vec![1.0], vec![3.0]]);
    }

    #[test]
    fn wrong_target_dimension_and_malformed_receipt_fail_closed() {
        let wrong_target = BoxTarget::new(vec![-0.1; 3], vec![0.1; 3]).unwrap();
        assert_eq!(
            KernelMeanApproachabilityMonitor::new(
                map(),
                wrong_target,
                1.0,
                1,
                20,
                TARGET_SHA.into(),
                REFERENCE_SHA.into(),
            )
            .err(),
            Some(KernelError::DimensionMismatch)
        );
        assert_eq!(
            KernelMeanApproachabilityMonitor::new(
                map(),
                target(),
                1.0,
                1,
                20,
                TARGET_SHA.into(),
                "bad".into(),
            )
            .err(),
            Some(KernelError::InvalidDigest("null_reference_receipt_sha256"))
        );
    }

    #[test]
    fn input_norm_bound_is_enforced_before_feature_evaluation() {
        let mut monitor = KernelMeanApproachabilityMonitor::new(
            map(),
            target(),
            0.5,
            1,
            20,
            TARGET_SHA.into(),
            REFERENCE_SHA.into(),
        )
        .unwrap();
        assert_eq!(monitor.observe(&[0.6]), Err(KernelError::InputNormExceeded));
        assert_eq!(monitor.observations_seen(), 0);
    }
}
