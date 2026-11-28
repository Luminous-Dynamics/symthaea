#![forbid(unsafe_code)]

//! Context-blocked paired Fourier-kernel discrepancy with anytime-valid evidence.
//!
//! Each pair shares a pre-registered context/covariate stratum c. With a fixed
//! Fourier feature map phi, the bounded observation is:
//!
//! ```text
//! r_t = 0.5 * e_c tensor (phi(Y_t) - phi(X_t))
//! ```
//!
//! Since ||phi(x)||=1, ||r_t||<=1. Under conditional distribution equality
//! within every context, E[r_t | history, context]=0. One-hot context blocks
//! prevent conditional discrepancies in different strata from cancelling in
//! a pooled marginal feature mean.
//!
//! The guarantee is exact for this finite feature bank and zero/tolerance
//! conditional-mean target. It is not a reproduction of SCPT's conditional
//! nearest-neighbor MMD construction.

use std::collections::HashSet;

use cmaq_046_convex_target_approachability::{
    ApproachabilityError, ApproachabilityUpdate, BoxTarget, ConvexTargetApproachabilityMonitor,
};
use cmaq_048_fourier_kernel_approachability::{FourierFeatureMap, KernelError};

const MAX_CONTEXTS: usize = 1_024;
const MAX_TOTAL_FEATURES: usize = 1_024;
const MAX_PAIR_ID_BYTES: usize = 256;
const MAX_PAIR_IDS: usize = 1_000_000;
const NORM_TOLERANCE: f64 = 1e-12;
const DIGEST_HEX_LEN: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PairedKernelError {
    EmptyContextInventory,
    TooManyContexts,
    TooManyContextFeatures,
    DimensionMismatch,
    InvalidDigest(&'static str),
    EmptyPairId,
    PairIdTooLong,
    DuplicatePairId,
    PairIdInventoryFull,
    UnknownContext,
    SequenceMismatch,
    ArithmeticOverflow,
    EmbeddingNormExceeded,
    Monitor(ApproachabilityError),
    FeatureMap(KernelError),
}

impl From<ApproachabilityError> for PairedKernelError {
    fn from(value: ApproachabilityError) -> Self {
        Self::Monitor(value)
    }
}

impl From<KernelError> for PairedKernelError {
    fn from(value: KernelError) -> Self {
        Self::FeatureMap(value)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContextualPairUpdate {
    pub sequence: u64,
    pub pair_id: String,
    pub context_id: usize,
    /// 0.5 * (phi(Y)-phi(X)), placed in the selected context block.
    pub contextual_feature_difference: Vec<f64>,
    pub evidence: ApproachabilityUpdate,
}

/// Conditional paired Fourier-feature discrepancy monitor.
///
/// Context IDs are a frozen finite partition of shared covariates. A
/// continuous covariate must be mapped to a pre-registered context/feature
/// representation outside this crate. Unknown context IDs fail closed.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextualPairedKernelMonitor {
    feature_map: FourierFeatureMap,
    context_count: usize,
    context_inventory_sha256: String,
    target_sha256: String,
    null_reference_receipt_sha256: String,
    monitor: ConvexTargetApproachabilityMonitor,
    seen_pair_ids: HashSet<String>,
}

impl ContextualPairedKernelMonitor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        feature_map: FourierFeatureMap,
        context_count: usize,
        target: BoxTarget,
        alpha_numerator: u32,
        alpha_denominator: u32,
        context_inventory_sha256: String,
        target_sha256: String,
        null_reference_receipt_sha256: String,
    ) -> Result<Self, PairedKernelError> {
        if context_count == 0 {
            return Err(PairedKernelError::EmptyContextInventory);
        }
        if context_count > MAX_CONTEXTS {
            return Err(PairedKernelError::TooManyContexts);
        }
        let total_dimension = context_count
            .checked_mul(feature_map.feature_dimension())
            .ok_or(PairedKernelError::ArithmeticOverflow)?;
        if total_dimension > MAX_TOTAL_FEATURES {
            return Err(PairedKernelError::TooManyContextFeatures);
        }
        if target.dimension() != total_dimension {
            return Err(PairedKernelError::DimensionMismatch);
        }
        validate_digest(&context_inventory_sha256, "context_inventory_sha256")?;
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
            context_count,
            context_inventory_sha256,
            target_sha256,
            null_reference_receipt_sha256,
            monitor,
            seen_pair_ids: HashSet::new(),
        })
    }

    /// Adds one paired observation. The context ID must have been assigned
    /// using pre-response covariates only; it must not depend on X or Y. A
    /// successful update consumes its sample ID exactly once in this process.
    /// The caller must use a durable external ledger for replay protection
    /// across process restarts.
    pub fn observe_pair(
        &mut self,
        sequence: u64,
        pair_id: &str,
        context_id: usize,
        x: &[f64],
        y: &[f64],
    ) -> Result<ContextualPairUpdate, PairedKernelError> {
        if pair_id.trim().is_empty() {
            return Err(PairedKernelError::EmptyPairId);
        }
        if pair_id.len() > MAX_PAIR_ID_BYTES {
            return Err(PairedKernelError::PairIdTooLong);
        }
        if self.seen_pair_ids.contains(pair_id) {
            return Err(PairedKernelError::DuplicatePairId);
        }
        if self.seen_pair_ids.len() >= MAX_PAIR_IDS {
            return Err(PairedKernelError::PairIdInventoryFull);
        }
        if context_id >= self.context_count {
            return Err(PairedKernelError::UnknownContext);
        }
        let expected_sequence = self
            .monitor
            .observations_seen()
            .checked_add(1)
            .ok_or(PairedKernelError::ArithmeticOverflow)?;
        if sequence != expected_sequence {
            return Err(PairedKernelError::SequenceMismatch);
        }

        let x_features = self.feature_map.transform(x)?;
        let y_features = self.feature_map.transform(y)?;
        let feature_dimension = self.feature_map.feature_dimension();
        let mut contextual_feature_difference = vec![0.0; self.context_count * feature_dimension];
        let offset = context_id * feature_dimension;
        for index in 0..feature_dimension {
            contextual_feature_difference[offset + index] =
                0.5 * (y_features[index] - x_features[index]);
        }
        let norm = l2_norm(&contextual_feature_difference);
        if !norm.is_finite() {
            return Err(PairedKernelError::ArithmeticOverflow);
        }
        if norm > 1.0 + NORM_TOLERANCE {
            return Err(PairedKernelError::EmbeddingNormExceeded);
        }

        // CMAQ-046 commits only after successful validation and arithmetic.
        // Consume the sample ID after that successful evidence update.
        let evidence = self.monitor.observe(&contextual_feature_difference)?;
        self.seen_pair_ids.insert(pair_id.to_string());

        Ok(ContextualPairUpdate {
            sequence,
            pair_id: pair_id.to_string(),
            context_id,
            contextual_feature_difference,
            evidence,
        })
    }

    pub fn feature_map(&self) -> &FourierFeatureMap {
        &self.feature_map
    }

    pub fn context_count(&self) -> usize {
        self.context_count
    }

    pub fn context_inventory_sha256(&self) -> &str {
        &self.context_inventory_sha256
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

    pub fn mean_contextual_feature_difference(&self) -> Result<Vec<f64>, ApproachabilityError> {
        self.monitor.mean_observation()
    }

    pub fn distance_to_null_target(&self) -> Result<f64, ApproachabilityError> {
        self.monitor.distance_to_target()
    }
}

fn validate_digest(value: &str, label: &'static str) -> Result<(), PairedKernelError> {
    if value.len() == DIGEST_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(PairedKernelError::InvalidDigest(label))
    }
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |norm, value| norm.hypot(*value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAP_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CONTEXT_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const TARGET_SHA: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const REFERENCE_SHA: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn map() -> FourierFeatureMap {
        FourierFeatureMap::new(1, vec![vec![1.0], vec![3.0]], MAP_SHA.into()).unwrap()
    }

    fn target(context_count: usize) -> BoxTarget {
        let tolerance = 0.005;
        BoxTarget::new(
            vec![-tolerance; context_count * 4],
            vec![tolerance; context_count * 4],
        )
        .unwrap()
    }

    fn monitor(context_count: usize) -> ContextualPairedKernelMonitor {
        ContextualPairedKernelMonitor::new(
            map(),
            context_count,
            target(context_count),
            1,
            20,
            CONTEXT_SHA.into(),
            TARGET_SHA.into(),
            REFERENCE_SHA.into(),
        )
        .unwrap()
    }

    #[test]
    fn contextual_pair_difference_is_bounded_by_one() {
        let mut monitor = monitor(2);
        let update = monitor
            .observe_pair(1, "pair-1", 0, &[0.2], &[0.8])
            .unwrap();
        assert!(l2_norm(&update.contextual_feature_difference) <= 1.0 + 1e-12);
        assert_eq!(update.contextual_feature_difference.len(), 8);
    }

    #[test]
    fn identical_pair_produces_zero_feature_difference_and_no_evidence_growth() {
        let mut monitor = monitor(2);
        let update = monitor
            .observe_pair(1, "pair-1", 1, &[0.2], &[0.2])
            .unwrap();
        assert!(
            update
                .contextual_feature_difference
                .iter()
                .all(|value| value.abs() < 1e-12)
        );
        assert_eq!(monitor.log_e_value(), 0.0);
    }

    #[test]
    fn pair_reversal_negates_the_contextual_feature_vector() {
        let mut forward = monitor(2);
        let mut reverse = monitor(2);
        let a = forward
            .observe_pair(1, "pair-a", 0, &[0.2], &[0.8])
            .unwrap();
        let b = reverse
            .observe_pair(1, "pair-b", 0, &[0.8], &[0.2])
            .unwrap();
        for (left, right) in a
            .contextual_feature_difference
            .iter()
            .zip(&b.contextual_feature_difference)
        {
            assert!((left + right).abs() < 1e-12);
        }
    }

    #[test]
    fn opposing_context_discrepancies_cannot_cancel_in_context_block_space() {
        let map = map();
        let at_small = map.transform(&[0.2]).unwrap();
        let at_large = map.transform(&[0.8]).unwrap();
        let forward = at_large
            .iter()
            .zip(&at_small)
            .map(|(large, small)| 0.5 * (large - small))
            .collect::<Vec<_>>();
        let reverse = forward.iter().map(|value| -*value).collect::<Vec<_>>();

        let pooled = forward
            .iter()
            .zip(&reverse)
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        assert!(l2_norm(&pooled) < 1e-12);

        let mut contextual = forward.clone();
        contextual.extend(reverse);
        assert!(l2_norm(&contextual) > 0.5);
    }

    #[test]
    fn conditional_mismatch_with_equal_marginal_mixture_accumulates_evidence() {
        let mut monitor = monitor(2);
        let mut quarantined_at = None;
        for index in 0..300 {
            let (context, x, y) = if index % 2 == 0 {
                (0, 0.2, 0.8)
            } else {
                (1, 0.8, 0.2)
            };
            let update = monitor
                .observe_pair(
                    index as u64 + 1,
                    &format!("pair-{index}"),
                    context,
                    &[x],
                    &[y],
                )
                .unwrap();
            if update.evidence.quarantined {
                quarantined_at = Some(index + 1);
                break;
            }
        }
        assert!(quarantined_at.is_some());
        assert!(monitor.is_quarantined());
        assert!(monitor.log_e_value() >= 20.0f64.ln());
    }

    #[test]
    fn duplicate_pair_and_wrong_sequence_do_not_mutate_evidence() {
        let mut monitor = monitor(2);
        monitor
            .observe_pair(1, "pair-1", 0, &[0.2], &[0.2])
            .unwrap();
        let before = monitor.clone();
        assert_eq!(
            monitor.observe_pair(2, "pair-1", 0, &[0.2], &[0.2]),
            Err(PairedKernelError::DuplicatePairId)
        );
        assert_eq!(
            monitor.observe_pair(3, "pair-3", 1, &[0.2], &[0.2]),
            Err(PairedKernelError::SequenceMismatch)
        );
        assert_eq!(monitor, before);
    }

    #[test]
    fn unknown_context_and_invalid_pair_id_fail_closed() {
        let mut monitor = monitor(2);
        assert_eq!(
            monitor.observe_pair(1, "pair-1", 2, &[0.2], &[0.2]),
            Err(PairedKernelError::UnknownContext)
        );
        assert_eq!(
            monitor.observe_pair(1, " ", 0, &[0.2], &[0.2]),
            Err(PairedKernelError::EmptyPairId)
        );
        assert_eq!(monitor.observations_seen(), 0);
    }

    #[test]
    fn frozen_context_map_and_reference_bindings_are_immutable() {
        let monitor = monitor(2);
        assert_eq!(monitor.feature_map().map_sha256(), MAP_SHA);
        assert_eq!(monitor.context_count(), 2);
        assert_eq!(monitor.context_inventory_sha256(), CONTEXT_SHA);
        assert_eq!(monitor.target_sha256(), TARGET_SHA);
        assert_eq!(monitor.null_reference_receipt_sha256(), REFERENCE_SHA);
    }

    #[test]
    fn context_inventory_and_target_dimensions_are_bounded() {
        assert_eq!(
            ContextualPairedKernelMonitor::new(
                map(),
                0,
                target(1),
                1,
                20,
                CONTEXT_SHA.into(),
                TARGET_SHA.into(),
                REFERENCE_SHA.into(),
            )
            .err(),
            Some(PairedKernelError::EmptyContextInventory)
        );
        // A valid 1,024-dimensional target is enough to reach the
        // context product-size guard (257 * 4 = 1,028 features).
        let too_many = BoxTarget::new(vec![-0.1; 1024], vec![0.1; 1024]).unwrap();
        assert_eq!(
            ContextualPairedKernelMonitor::new(
                map(),
                257,
                too_many,
                1,
                20,
                CONTEXT_SHA.into(),
                TARGET_SHA.into(),
                REFERENCE_SHA.into(),
            )
            .err(),
            Some(PairedKernelError::TooManyContextFeatures)
        );
    }
}
