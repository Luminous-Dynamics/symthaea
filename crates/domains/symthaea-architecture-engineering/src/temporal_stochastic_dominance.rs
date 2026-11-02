//! Anytime-valid first-order stochastic-dominance test for finite known exposure support.
//!
//! This adapts Arnold et al. (2026), "Betting on Bets", to paired temporal-exposure
//! comparisons. Let X=-control_exposure and Y=-treatment_exposure (higher reward is
//! better). The null is Y <=_1 X: the treatment reward is stochastically dominated
//! by control, so treatment has no distributional upside. For every finite-support
//! threshold z, the building-block bet is
//!
//!   S_t(lambda,z) = 1 + lambda * (1{X_t <= z} - 1{Y_t <= z}).
//!
//! With predictable lambda in [0,1], this is an e-variable under the marginal FSD
//! null. A fixed uniform mixture over all non-maximum support thresholds is valid.
//! The paper's predictable plug-in GRO bet is used at each threshold. On finite
//! known support, the fixed uniform mixture has positive mass at every threshold,
//! satisfying the paper's condition for its asymptotic power-one result.
//!
//! Statistical assumptions: the paired (X_t,Y_t) sequence is IID across t; within-pair
//! dependence is arbitrary. No cross-pair interference or nonstationarity is covered.
//! Fair-coin assignment labels are still required by the temporal intervention protocol,
//! but randomization alone does not make a non-IID time sequence valid.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::temporal_intervention_family::{
    TemporalInterventionAssignmentScheme,
    TemporalInterventionAxis,
    TemporalInterventionFamily,
    TemporalInterventionLedgerError,
    TemporalInterventionPairOutcome,
    TemporalInterventionEvidenceLedger,
    temporal_intervention_stream_id,
    verify_temporal_intervention_family,
    verify_temporal_intervention_evidence_ledger,
};

const SUPPORT_DOMAIN: &[u8] = b"symthaea.temporal-fsd-support-profile.v0.1\0";
const FAMILY_DOMAIN: &[u8] = b"symthaea.temporal-fsd-family.v0.1\0";
const MAX_SUPPORT_BINS: usize = 512;
pub const MAX_FSD_PAIRS: u64 = 1_000_000;

/// One predeclared finite exposure support for one contrast/outcome axis.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalExposureSupportProfileInput {
    pub contrast_id: String,
    pub axis: TemporalInterventionAxis,
    /// Strictly increasing possible exposure values, in microseconds.
    pub exposure_support_us: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalExposureSupportProfile {
    pub stream_id: String,
    pub exposure_support_us: Vec<u64>,
    pub profile_sha256: String,
}

/// Complete distributional family: the generic ledger owns contrast/axis multiplicity;
/// the outer digest additionally freezes every finite exposure support.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalStochasticDominanceFamily {
    pub schema: String,
    pub version: String,
    pub temporal_family: TemporalInterventionFamily,
    pub support_profiles: Vec<TemporalExposureSupportProfile>,
    pub family_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalStochasticDominanceError {
    pub field: String,
    pub message: String,
}

impl TemporalStochasticDominanceError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }

    fn from_temporal(error: TemporalInterventionLedgerError) -> Self {
        Self::new(error.field, error.message)
    }
}

impl fmt::Display for TemporalStochasticDominanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for TemporalStochasticDominanceError {}

pub fn create_temporal_stochastic_dominance_family(
    temporal_family: TemporalInterventionFamily,
    support_inputs: Vec<TemporalExposureSupportProfileInput>,
) -> Result<TemporalStochasticDominanceFamily, TemporalStochasticDominanceError> {
    verify_temporal_intervention_family(&temporal_family)
        .map_err(TemporalStochasticDominanceError::from_temporal)?;

    let expected_streams: BTreeSet<String> = temporal_family.confidence_family.streams
        .iter().map(|stream| stream.stream_id.clone()).collect();
    if support_inputs.len() != expected_streams.len() {
        return Err(TemporalStochasticDominanceError::new(
            "support_profiles",
            format!(
                "expected exactly {} support profiles (one per preregistered contrast-axis stream)",
                expected_streams.len()
            ),
        ));
    }

    let mut profiles = Vec::with_capacity(support_inputs.len());
    let mut seen = BTreeSet::new();
    for input in support_inputs {
        validate_exposure_support(&input.exposure_support_us)?;
        let stream_id = temporal_intervention_stream_id(
            &temporal_family,
            &input.contrast_id,
            input.axis,
        )
        .map_err(TemporalStochasticDominanceError::from_temporal)?;
        if !expected_streams.contains(&stream_id) {
            return Err(TemporalStochasticDominanceError::new(
                "support_profiles.stream_id",
                "support profile does not map to a declared temporal confidence stream",
            ));
        }
        if !seen.insert(stream_id.clone()) {
            return Err(TemporalStochasticDominanceError::new(
                "support_profiles.stream_id",
                "duplicate contrast/axis support profile",
            ));
        }
        let profile_sha256 = support_profile_digest(&stream_id, &input.exposure_support_us);
        profiles.push(TemporalExposureSupportProfile {
            stream_id,
            exposure_support_us: input.exposure_support_us,
            profile_sha256,
        });
    }
    profiles.sort_by(|left, right| left.stream_id.cmp(&right.stream_id));
    if seen != expected_streams {
        return Err(TemporalStochasticDominanceError::new(
            "support_profiles",
            "support profiles do not cover the complete preregistered stream family",
        ));
    }

    let mut family = TemporalStochasticDominanceFamily {
        schema: "symthaea.temporal-stochastic-dominance-family".to_owned(),
        version: "0.1.0".to_owned(),
        temporal_family,
        support_profiles: profiles,
        family_sha256: String::new(),
    };
    family.family_sha256 = distributional_family_digest(&family)?;
    verify_temporal_stochastic_dominance_family(&family)?;
    Ok(family)
}

pub fn verify_temporal_stochastic_dominance_family(
    family: &TemporalStochasticDominanceFamily,
) -> Result<(), TemporalStochasticDominanceError> {
    if family.schema != "symthaea.temporal-stochastic-dominance-family"
        || family.version != "0.1.0"
    {
        return Err(TemporalStochasticDominanceError::new(
            "schema_version", "unsupported stochastic-dominance family schema",
        ));
    }
    verify_temporal_intervention_family(&family.temporal_family)
        .map_err(TemporalStochasticDominanceError::from_temporal)?;

    let expected: BTreeSet<String> = family.temporal_family.confidence_family.streams
        .iter().map(|stream| stream.stream_id.clone()).collect();
    if family.support_profiles.len() != expected.len()
        || family.support_profiles.windows(2).any(|pair| pair[0].stream_id >= pair[1].stream_id)
    {
        return Err(TemporalStochasticDominanceError::new(
            "support_profiles", "profiles must be complete, unique and strictly sorted by stream ID",
        ));
    }

    let actual: BTreeSet<String> = family.support_profiles.iter()
        .map(|profile| profile.stream_id.clone())
        .collect();
    if actual != expected {
        return Err(TemporalStochasticDominanceError::new(
            "support_profiles", "profile stream IDs do not equal the complete generic family",
        ));
    }
    for profile in &family.support_profiles {
        validate_exposure_support(&profile.exposure_support_us)?;
        let digest = support_profile_digest(&profile.stream_id, &profile.exposure_support_us);
        if digest != profile.profile_sha256 {
            return Err(TemporalStochasticDominanceError::new(
                "support_profiles.profile_sha256", "support profile digest mismatch",
            ));
        }
    }

    let expected_digest = distributional_family_digest(family)?;
    if expected_digest != family.family_sha256 {
        return Err(TemporalStochasticDominanceError::new(
            "family_sha256", "distributional family digest mismatch",
        ));
    }
    Ok(())
}

fn validate_exposure_support(support: &[u64]) -> Result<(), TemporalStochasticDominanceError> {
    if support.len() < 2 || support.len() > MAX_SUPPORT_BINS {
        return Err(TemporalStochasticDominanceError::new(
            "exposure_support_us",
            format!("support size must be in 2..={MAX_SUPPORT_BINS}"),
        ));
    }
    if support.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(TemporalStochasticDominanceError::new(
            "exposure_support_us", "support must be strictly increasing and unique",
        ));
    }
    Ok(())
}

fn support_profile_digest(stream_id: &str, support: &[u64]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(SUPPORT_DOMAIN);
    hash_string(&mut hasher, stream_id);
    hasher.update((support.len() as u64).to_le_bytes());
    for value in support {
        hasher.update(value.to_le_bytes());
    }
    hex_digest(&hasher.finalize())
}

fn distributional_family_digest(
    family: &TemporalStochasticDominanceFamily,
) -> Result<String, TemporalStochasticDominanceError> {
    let mut copy = family.clone();
    copy.family_sha256.clear();
    let bytes = serde_json::to_vec(&copy).map_err(|error| {
        TemporalStochasticDominanceError::new("family", error.to_string())
    })?;
    let mut hasher = Sha256::new();
    hasher.update(FAMILY_DOMAIN);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    Ok(hex_digest(&hasher.finalize()))
}

fn hash_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ThresholdBetCounts {
    pub positive_crossings: u64,
    pub negative_crossings: u64,
    pub ties: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThresholdBetReport {
    pub reward_threshold: i128,
    pub counts: ThresholdBetCounts,
    /// Plug-in growth-rate-optimal bet used for the next pair (predictable).
    pub next_lambda: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FirstOrderDominanceReport {
    pub family_sha256: String,
    pub generic_manifest_sha256: String,
    pub stream_id: String,
    pub contrast_id: String,
    pub axis: TemporalInterventionAxis,
    pub profile_sha256: String,
    /// Present when this report was reconstructed from a verified append-only evidence ledger.
    pub evidence_ledger_sha256: Option<String>,
    pub observed_pairs: u64,
    pub stream_alpha: f64,
    pub log_e_value: f64,
    pub log_boundary: f64,
    pub numerical_guard: f64,
    pub stream_rejected_no_upside: bool,
    pub first_rejection_look: Option<u64>,
    pub thresholds: Vec<ThresholdBetReport>,
}

#[derive(Clone, Debug)]
pub struct FirstOrderDominanceState {
    family_sha256: String,
    generic_manifest_sha256: String,
    stream_id: String,
    contrast_id: String,
    axis: TemporalInterventionAxis,
    profile_sha256: String,
    exposure_support_us: Vec<u64>,
    reward_thresholds: Vec<i128>,
    threshold_counts: Vec<ThresholdBetCounts>,
    clip_c_ppm: u32,
    maximum_pairs: u64,
    observed_pairs: u64,
    log_e_value: f64,
    numerical_guard: f64,
    stream_alpha: f64,
    log_boundary: f64,
    first_rejection_look: Option<u64>,
    pair_ids: BTreeSet<u64>,
    contexts: BTreeSet<[u8; 32]>,
    receipts: BTreeSet<[u8; 32]>,
}

impl FirstOrderDominanceState {
    /// Construct a one-sided test of the null that treatment has no first-order
    /// stochastic upside over control. c_ppm is the GRO plug-in cap parameter c,
    /// expressed in parts per million.
    pub fn new(
        family: &TemporalStochasticDominanceFamily,
        contrast_id: &str,
        axis: TemporalInterventionAxis,
        clip_c_ppm: u32,
        maximum_pairs: u64,
    ) -> Result<Self, TemporalStochasticDominanceError> {
        verify_temporal_stochastic_dominance_family(family)?;
        if clip_c_ppm == 0 || clip_c_ppm > 500_000 {
            return Err(TemporalStochasticDominanceError::new(
                "clip_c_ppm", "c must be in 1..=500000 PPM",
            ));
        }
        if maximum_pairs == 0 || maximum_pairs > MAX_FSD_PAIRS {
            return Err(TemporalStochasticDominanceError::new(
                "maximum_pairs", format!("maximum pair count must be in 1..={MAX_FSD_PAIRS}"),
            ));
        }

        let stream_id = temporal_intervention_stream_id(
            &family.temporal_family, contrast_id, axis,
        ).map_err(TemporalStochasticDominanceError::from_temporal)?;
        let profile = family.support_profiles.iter()
            .find(|profile| profile.stream_id == stream_id)
            .ok_or_else(|| TemporalStochasticDominanceError::new(
                "support_profile", "missing preregistered finite exposure support",
            ))?;
        let manifest = &family.temporal_family.confidence_family;
        let stream_count = manifest.streams.len();
        if stream_count == 0 || manifest.family_alpha <= 0.0 || manifest.family_alpha >= 1.0 {
            return Err(TemporalStochasticDominanceError::new(
                "confidence_family", "invalid generic family alpha or empty stream family",
            ));
        }
        let stream_alpha = manifest.family_alpha / stream_count as f64;
        if !(stream_alpha > 0.0 && stream_alpha < 1.0) {
            return Err(TemporalStochasticDominanceError::new(
                "stream_alpha", "derived stream alpha is outside (0,1)",
            ));
        }

        // Exposure is a cost; negate it so larger reward is better. The maximum
        // reward threshold is omitted because its CDF difference is always zero.
        let reward_support: Vec<i128> = profile.exposure_support_us.iter()
            .rev().map(|value| -i128::from(*value)).collect();
        let reward_thresholds = reward_support[..reward_support.len() - 1].to_vec();

        Ok(Self {
            family_sha256: family.family_sha256.clone(),
            generic_manifest_sha256: manifest.manifest_sha256.clone(),
            stream_id,
            contrast_id: contrast_id.to_owned(),
            axis,
            profile_sha256: profile.profile_sha256.clone(),
            exposure_support_us: profile.exposure_support_us.clone(),
            reward_thresholds,
            threshold_counts: vec![ThresholdBetCounts::default(); reward_support.len() - 1],
            clip_c_ppm,
            maximum_pairs,
            observed_pairs: 0,
            log_e_value: 0.0,
            numerical_guard: 0.0,
            stream_alpha,
            log_boundary: (1.0 / stream_alpha).ln(),
            first_rejection_look: None,
            pair_ids: BTreeSet::new(),
            contexts: BTreeSet::new(),
            receipts: BTreeSet::new(),
        })
    }

    /// Append the next evaluator-validated pair. The bet parameters use only
    /// past pairs; the current pair updates evidence after the bet is selected.
    pub fn observe(
        &mut self,
        sequence: u64,
        pair: &TemporalInterventionPairOutcome,
    ) -> Result<FirstOrderDominanceReport, TemporalStochasticDominanceError> {
        if self.observed_pairs >= self.maximum_pairs {
            return Err(TemporalStochasticDominanceError::new(
                "maximum_pairs", "configured maximum pair count reached",
            ));
        }
        let expected = self.observed_pairs + 1;
        if sequence != expected {
            return Err(TemporalStochasticDominanceError::new(
                "sequence", format!("expected sequential look {expected}, received {sequence}"),
            ));
        }
        validate_pair(self, pair)?;

        let x_reward = -i128::from(pair.control_exposure_us);
        let y_reward = -i128::from(pair.treatment_exposure_us);
        let mut step_sum = 0.0_f64;
        let mut step_guard = 0.0_f64;
        let threshold_count = self.reward_thresholds.len();

        for (index, threshold) in self.reward_thresholds.iter().enumerate() {
            let counts = self.threshold_counts[index];
            let lambda = plugin_gro_lambda(counts, self.clip_c_ppm);
            let x_le = x_reward <= *threshold;
            let y_le = y_reward <= *threshold;
            let difference = i8::from(x_le) - i8::from(y_le);
            let factor = 1.0 + lambda * f64::from(difference);
            if !(factor > 0.0 && factor.is_finite()) {
                return Err(TemporalStochasticDominanceError::new(
                    "bet_factor", "non-positive or non-finite e-variable factor",
                ));
            }

            // Lower the numerical increment by a scale-aware guard before
            // aggregating. This is an engineering guard, not a formal correctly-
            // rounded arithmetic proof; high-assurance replay should use a
            // higher-precision verifier.
            let local_guard = if factor == 1.0 {
                0.0
            } else {
                16.0 * f64::EPSILON * factor.abs().max(1.0)
            };
            let lower_factor = (factor - local_guard).max(f64::MIN_POSITIVE);
            step_sum += lower_factor;
            step_guard += local_guard;
        }

        let mean_increment = step_sum / threshold_count as f64;
        let aggregation_guard = if mean_increment == 1.0 && step_guard == 0.0 {
            0.0
        } else {
            16.0 * f64::EPSILON * mean_increment.abs().max(1.0)
                * (threshold_count as f64 + 2.0)
        };
        let lower_increment = (mean_increment - aggregation_guard).max(f64::MIN_POSITIVE);
        let log_increment = if lower_increment == 1.0 { 0.0 } else { lower_increment.ln() };
        let log_guard = if log_increment == 0.0 && aggregation_guard == 0.0 {
            0.0
        } else {
            16.0 * f64::EPSILON * log_increment.abs().max(1.0)
                + step_guard / threshold_count as f64
        };
        self.numerical_guard += aggregation_guard + log_guard;
        self.log_e_value += log_increment;

        for (index, threshold) in self.reward_thresholds.iter().enumerate() {
            let x_le = x_reward <= *threshold;
            let y_le = y_reward <= *threshold;
            let difference = i8::from(x_le) - i8::from(y_le);
            let counts = &mut self.threshold_counts[index];
            match difference {
                1 => counts.positive_crossings += 1,
                -1 => counts.negative_crossings += 1,
                _ => counts.ties += 1,
            }
        }

        self.observed_pairs += 1;
        let conservative_log_e = self.log_e_value - self.numerical_guard;
        if conservative_log_e >= self.log_boundary && self.first_rejection_look.is_none() {
            self.first_rejection_look = Some(self.observed_pairs);
        }
        Ok(self.report())
    }

    pub fn report(&self) -> FirstOrderDominanceReport {
        FirstOrderDominanceReport {
            family_sha256: self.family_sha256.clone(),
            generic_manifest_sha256: self.generic_manifest_sha256.clone(),
            stream_id: self.stream_id.clone(),
            contrast_id: self.contrast_id.clone(),
            axis: self.axis,
            profile_sha256: self.profile_sha256.clone(),
            evidence_ledger_sha256: None,
            observed_pairs: self.observed_pairs,
            stream_alpha: self.stream_alpha,
            log_e_value: self.log_e_value - self.numerical_guard,
            log_boundary: self.log_boundary,
            numerical_guard: self.numerical_guard,
            stream_rejected_no_upside: self.first_rejection_look.is_some(),
            first_rejection_look: self.first_rejection_look,
            thresholds: self.reward_thresholds.iter().enumerate().map(|(index, threshold)| {
                ThresholdBetReport {
                    reward_threshold: *threshold,
                    counts: self.threshold_counts[index],
                    next_lambda: plugin_gro_lambda(self.threshold_counts[index], self.clip_c_ppm),
                }
            }).collect(),
        }
    }
}


/// Reconstruct an FSD report from the canonical temporal intervention evidence ledger.
///
/// The generic temporal ledger is the source of truth for ordered pair outcomes,
/// assignment receipts, context/source-family bindings and cumulative look history.
/// This function verifies the entire ledger, checks that its contrast family matches
/// the frozen finite-support family, then deterministically replays the selected
/// contrast/axis stream. No second persistent FSD history is needed.
pub fn replay_first_order_dominance_from_evidence_ledger(
    family: &TemporalStochasticDominanceFamily,
    evidence_ledger: &TemporalInterventionEvidenceLedger,
    contrast_id: &str,
    axis: TemporalInterventionAxis,
    clip_c_ppm: u32,
    maximum_pairs: u64,
) -> Result<FirstOrderDominanceReport, TemporalStochasticDominanceError> {
    verify_temporal_stochastic_dominance_family(family)?;
    verify_temporal_intervention_evidence_ledger(evidence_ledger)
        .map_err(TemporalStochasticDominanceError::from_temporal)?;
    if evidence_ledger.family != family.temporal_family {
        return Err(TemporalStochasticDominanceError::new(
            "evidence_ledger.family",
            "temporal evidence ledger family does not match the stochastic-dominance family",
        ));
    }

    let stream_id = temporal_intervention_stream_id(
        &family.temporal_family, contrast_id, axis,
    ).map_err(TemporalStochasticDominanceError::from_temporal)?;

    let mut bindings: Vec<_> = evidence_ledger.pair_bindings.iter()
        .filter(|binding| binding.stream_id == stream_id)
        .collect();
    bindings.sort_by_key(|binding| binding.look_index);
    if bindings.windows(2).any(|pair| pair[0].look_index + 1 != pair[1].look_index) {
        return Err(TemporalStochasticDominanceError::new(
            "pair_bindings.look_index",
            "selected stream look sequence is not contiguous",
        ));
    }

    let mut state = FirstOrderDominanceState::new(
        family, contrast_id, axis, clip_c_ppm, maximum_pairs,
    )?;
    let mut sequence = 0_u64;
    for binding in bindings {
        if binding.contrast_id != contrast_id || binding.axis != axis {
            return Err(TemporalStochasticDominanceError::new(
                "pair_bindings.identity",
                "selected stream binding has a mismatched contrast or axis",
            ));
        }
        for pair in &binding.pair_outcomes {
            sequence = sequence.checked_add(1).ok_or_else(|| {
                TemporalStochasticDominanceError::new("sequence", "pair sequence overflow")
            })?;
            state.observe(sequence, pair)?;
        }
    }
    let mut report = state.report();
    report.evidence_ledger_sha256 = Some(evidence_ledger.evidence_ledger_sha256.clone());
    Ok(report)
}

fn validate_pair(
    state: &FirstOrderDominanceState,
    pair: &TemporalInterventionPairOutcome,
) -> Result<(), TemporalStochasticDominanceError> {
    if pair.pair_id == 0 {
        return Err(TemporalStochasticDominanceError::new("pair_id", "pair ID must be nonzero"));
    }
    if pair.context_sha256 == [0; 32] || pair.source_family_sha256 == [0; 32]
        || pair.assignment_receipt_sha256 == [0; 32]
    {
        return Err(TemporalStochasticDominanceError::new(
            "pair_digests", "context, source-family and assignment digests must be nonzero",
        ));
    }
    if state.pair_ids.contains(&pair.pair_id)
        || state.contexts.contains(&pair.context_sha256)
        || state.receipts.contains(&pair.assignment_receipt_sha256)
    {
        return Err(TemporalStochasticDominanceError::new(
            "pair_identity", "duplicate pair ID, context or assignment receipt",
        ));
    }
    if pair.assignment_scheme != TemporalInterventionAssignmentScheme::IndependentFairCoin
        || pair.treatment_probability_ppm != 500_000
    {
        return Err(TemporalStochasticDominanceError::new(
            "assignment", "requires independent fair-coin treatment orientation at 500000 PPM",
        ));
    }
    if pair.episode_duration_us == 0
        || pair.treatment_exposure_us > pair.episode_duration_us
        || pair.control_exposure_us > pair.episode_duration_us
    {
        return Err(TemporalStochasticDominanceError::new(
            "exposure", "arm exposure must be within a positive episode horizon",
        ));
    }
    if state.exposure_support_us.binary_search(&pair.treatment_exposure_us).is_err()
        || state.exposure_support_us.binary_search(&pair.control_exposure_us).is_err()
    {
        return Err(TemporalStochasticDominanceError::new(
            "exposure_support", "arm exposure is outside the frozen finite support",
        ));
    }
    Ok(())
}

/// Plug-in growth-rate-optimal bet from the empirical paired crossing counts.
/// Positive crossings are X<=z<Y; negative crossings are Y<=z<X.
fn plugin_gro_lambda(counts: ThresholdBetCounts, clip_c_ppm: u32) -> f64 {
    let positive = counts.positive_crossings;
    let negative = counts.negative_crossings;
    let total = positive.saturating_add(negative);
    if total == 0 || positive <= negative {
        return 0.0;
    }
    let empirical = (positive - negative) as f64 / total as f64;
    let cap = 1.0 - f64::from(clip_c_ppm) / 1_000_000.0;
    (empirical.min(cap) - 8.0 * f64::EPSILON).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn pair(id: u64, treatment: u64, control: u64) -> TemporalInterventionPairOutcome {
        TemporalInterventionPairOutcome {
            pair_id: id,
            context_sha256: digest((20 + id) as u8),
            source_family_sha256: digest(9),
            assignment_receipt_sha256: digest((80 + id) as u8),
            assignment_scheme: TemporalInterventionAssignmentScheme::IndependentFairCoin,
            treatment_probability_ppm: 500_000,
            treatment_was_first: id % 2 == 0,
            episode_duration_us: 100,
            treatment_exposure_us: treatment,
            control_exposure_us: control,
        }
    }

    fn base_family() -> TemporalInterventionFamily {
        use crate::temporal_intervention_family::{
            TemporalInterventionContrast, create_temporal_intervention_family,
        };
        let digest_hex = |v: u8| format!("{v:02x}").repeat(32);
        create_temporal_intervention_family(
            "fsd-family",
            0.05,
            vec![
                TemporalInterventionContrast {
                    contrast_id: "pareto-vs-greedy".to_owned(),
                    task_sha256: digest_hex(1),
                    model_sha256: digest_hex(2),
                    evaluator_sha256: digest_hex(3),
                    treatment_policy_sha256: digest_hex(4),
                    control_policy_sha256: digest_hex(5),
                    assignment_manifest_sha256: digest_hex(6),
                }
            ],
        ).unwrap()
    }

    fn fsd_family() -> TemporalStochasticDominanceFamily {
        create_temporal_stochastic_dominance_family(
            base_family(),
            TemporalInterventionAxis::ALL.into_iter().map(|axis| {
                TemporalExposureSupportProfileInput {
                    contrast_id: "pareto-vs-greedy".to_owned(),
                    axis,
                    exposure_support_us: vec![0, 1, 10, 100],
                }
            }).collect(),
        ).unwrap()
    }

    #[test]
    fn fixed_finite_support_profile_covers_all_family_streams_and_is_committed() {
        let family = fsd_family();
        assert_eq!(family.support_profiles.len(), 3);
        assert_eq!(
            family.support_profiles.iter().map(|p| p.exposure_support_us.as_slice())
                .collect::<Vec<_>>(),
            vec![&[0, 1, 10, 100][..]; 3]
        );
        verify_temporal_stochastic_dominance_family(&family).unwrap();

        let mut mutated = family.clone();
        mutated.support_profiles[0].exposure_support_us[1] = 2;
        assert!(verify_temporal_stochastic_dominance_family(&mutated).is_err());
    }

    #[test]
    fn support_profile_must_cover_each_declared_stream_exactly_once() {
        let base = base_family();
        let missing = vec![
            TemporalExposureSupportProfileInput {
                contrast_id: "pareto-vs-greedy".to_owned(),
                axis: TemporalInterventionAxis::WrongBelief,
                exposure_support_us: vec![0, 1],
            }
        ];
        assert!(create_temporal_stochastic_dominance_family(base, missing).is_err());
    }

    #[test]
    fn family_alpha_and_stream_boundary_are_derived_not_caller_supplied() {
        let family = fsd_family();
        let state = FirstOrderDominanceState::new(
            &family, "pareto-vs-greedy", TemporalInterventionAxis::WrongBelief, 10_000, 100,
        ).unwrap();
        assert!((state.stream_alpha - (0.05 / 3.0)).abs() < 1e-14);
        assert!((state.log_boundary - (60.0_f64).ln()).abs() < 1e-14);
        assert_eq!(state.reward_thresholds, vec![-100, -10, -1]);
    }

    #[test]
    fn first_pair_uses_only_past_data_and_has_zero_initial_gro_bet() {
        let family = fsd_family();
        let mut state = FirstOrderDominanceState::new(
            &family, "pareto-vs-greedy", TemporalInterventionAxis::WrongBelief, 10_000, 100,
        ).unwrap();
        let report = state.observe(1, &pair(1, 0, 100)).unwrap();
        assert_eq!(report.thresholds[0].counts.positive_crossings, 1);
        assert!(report.thresholds.iter().all(|threshold| threshold.next_lambda >= 0.0));
        assert_eq!(report.log_e_value, 0.0);
    }

    #[test]
    fn sustained_treatment_lower_exposure_eventually_rejects_no_upside_null() {
        let family = fsd_family();
        let mut state = FirstOrderDominanceState::new(
            &family, "pareto-vs-greedy", TemporalInterventionAxis::WrongBelief, 10_000, 100,
        ).unwrap();
        let mut report = state.report();
        for look in 1..=40 {
            report = state.observe(look, &pair(look, 0, 100)).unwrap();
            if report.stream_rejected_no_upside {
                break;
            }
        }
        assert!(report.stream_rejected_no_upside);
        assert!(report.first_rejection_look.is_some());
    }

    #[test]
    fn a_no_upside_deterministic_null_does_not_create_positive_evidence() {
        let family = fsd_family();
        let mut state = FirstOrderDominanceState::new(
            &family, "pareto-vs-greedy", TemporalInterventionAxis::WrongBelief, 10_000, 100,
        ).unwrap();
        for look in 1..=30 {
            let report = state.observe(look, &pair(look, 0, 0)).unwrap();
            assert!(!report.stream_rejected_no_upside);
            assert_eq!(report.log_e_value, 0.0);
        }
    }

    #[test]
    fn pair_assignment_support_horizon_and_duplicate_fail_closed() {
        let family = fsd_family();
        let mut state = FirstOrderDominanceState::new(
            &family, "pareto-vs-greedy", TemporalInterventionAxis::WrongBelief, 10_000, 100,
        ).unwrap();
        let mut blocked = pair(1, 0, 10);
        blocked.assignment_scheme = TemporalInterventionAssignmentScheme::FixedCountBlocked;
        assert!(state.observe(1, &blocked).is_err());
        assert_eq!(state.observed_pairs, 0);

        let mut outside = pair(1, 0, 10);
        outside.control_exposure_us = 50;
        assert!(state.observe(1, &outside).is_err());

        state.observe(1, &pair(1, 0, 10)).unwrap();
        assert!(state.observe(2, &pair(1, 0, 10)).is_err());
    }

    #[test]
    fn predictable_mixture_is_a_one_step_supermartingale_on_a_stochastic_dominance_null() {
        // Independent paired outcomes with control exposure C~Bernoulli(0.5)
        // and treatment exposure T~Bernoulli(0.75). Lower exposure is better,
        // so treatment is stochastically worse and the null is true.
        // At the active threshold, E[D]=P(C=1,T=0)-P(C=0,T=1)
        // = 0.125-0.375=-0.25. Other thresholds have D=0.
        fn explore(state: FirstOrderDominanceState, depth: u64) {
            if depth >= 4 {
                return;
            }
            let prior_log = state.report().log_e_value;
            let outcomes = [
                (0_u64, 0_u64, 0.125_f64),
                (1_u64, 0_u64, 0.375_f64),
                (0_u64, 1_u64, 0.125_f64),
                (1_u64, 1_u64, 0.375_f64),
            ];
            let mut conditional_mean = 0.0_f64;
            for (treatment, control, probability) in outcomes {
                let mut child = state.clone();
                let report = child.observe(depth + 1, &pair(depth + 1, treatment, control)).unwrap();
                let incremental_wealth = (report.log_e_value - prior_log).exp();
                conditional_mean += probability * incremental_wealth;
                explore(child, depth + 1);
            }
            assert!(conditional_mean <= 1.0 + 1e-10);
        }

        let family = fsd_family();
        let state = FirstOrderDominanceState::new(
            &family,
            "pareto-vs-greedy",
            TemporalInterventionAxis::WrongBelief,
            10_000,
            8,
        ).unwrap();
        explore(state, 0);
    }

    #[test]
    fn non_fair_pair_assignment_probability_is_rejected() {
        let family = fsd_family();
        let mut state = FirstOrderDominanceState::new(
            &family, "pareto-vs-greedy", TemporalInterventionAxis::WrongBelief, 10_000, 100,
        ).unwrap();
        let mut biased = pair(1, 0, 10);
        biased.treatment_probability_ppm = 250_000;
        assert!(state.observe(1, &biased).is_err());
        assert_eq!(state.observed_pairs, 0);
    }

    #[test]
    fn replay_from_canonical_pair_ledger_matches_direct_fsd_state() {
        use crate::temporal_intervention_family::{
            append_temporal_intervention_batch,
            create_temporal_intervention_evidence_ledger,
        };

        let family = fsd_family();
        let temporal_family = family.temporal_family.clone();
        let contrast_id = "pareto-vs-greedy";
        let axis = TemporalInterventionAxis::WrongBelief;
        let mut evidence = create_temporal_intervention_evidence_ledger(temporal_family).unwrap();

        let pairs = [
            pair(1, 0, 100),
            pair(2, 1, 10),
            pair(3, 10, 1),
            pair(4, 0, 100),
        ];
        append_temporal_intervention_batch(
            &mut evidence, contrast_id, axis, &pairs[..2],
        ).unwrap();
        append_temporal_intervention_batch(
            &mut evidence, contrast_id, axis, &pairs[2..],
        ).unwrap();

        let replayed = replay_first_order_dominance_from_evidence_ledger(
            &family, &evidence, contrast_id, axis, 10_000, 100,
        ).unwrap();

        let mut direct = FirstOrderDominanceState::new(
            &family, contrast_id, axis, 10_000, 100,
        ).unwrap();
        for (index, pair) in pairs.iter().enumerate() {
            direct.observe(index as u64 + 1, pair).unwrap();
        }
        let mut expected = direct.report();
        expected.evidence_ledger_sha256 = Some(evidence.evidence_ledger_sha256.clone());
        assert_eq!(replayed, expected);
    }

    #[test]
    fn replay_rejects_tampered_pair_exposure_binding() {
        use crate::temporal_intervention_family::{
            append_temporal_intervention_batch,
            create_temporal_intervention_evidence_ledger,
        };

        let family = fsd_family();
        let temporal_family = family.temporal_family.clone();
        let contrast_id = "pareto-vs-greedy";
        let axis = TemporalInterventionAxis::WrongBelief;
        let mut evidence = create_temporal_intervention_evidence_ledger(temporal_family).unwrap();
        append_temporal_intervention_batch(
            &mut evidence, contrast_id, axis, &[pair(1, 0, 100)],
        ).unwrap();

        evidence.pair_bindings[0].pair_outcomes[0].treatment_exposure_us = 100;
        assert!(replay_first_order_dominance_from_evidence_ledger(
            &family, &evidence, contrast_id, axis, 10_000, 100,
        ).is_err());
    }

    #[test]
    fn sequential_look_gap_does_not_advance_evidence() {
        let family = fsd_family();
        let mut state = FirstOrderDominanceState::new(
            &family, "pareto-vs-greedy", TemporalInterventionAxis::WrongBelief, 10_000, 100,
        ).unwrap();
        assert!(state.observe(2, &pair(1, 0, 10)).is_err());
        assert_eq!(state.observed_pairs, 0);
    }
}
