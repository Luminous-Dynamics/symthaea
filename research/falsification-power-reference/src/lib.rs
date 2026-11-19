//! A small, dependency-free power oracle for declared Gaussian model discrepancies.
//!
//! This crate is intentionally narrower than a model-adequacy test or experiment
//! selector. It computes alternative-specific best-case power for a fixed
//! observation design when the discrepancy direction and iid Gaussian noise
//! variance are known to the evaluator. See the repository research note for
//! assumptions and claim boundaries.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Errors for invalid designs and power-envelope declarations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PowerError {
    EmptyObservationDesign,
    EmptyAlternativeSet,
    EmptyAlternativeId,
    DuplicateAlternativeId(String),
    MismatchedObservationCount { expected: usize, actual: usize },
    NonFiniteResidual { index: usize },
    InvalidSigma,
    InvalidAlpha,
    InvalidTargetPower,
}

impl fmt::Display for PowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyObservationDesign => write!(f, "observation design must not be empty"),
            Self::EmptyAlternativeSet => write!(f, "declared alternative set must not be empty"),
            Self::EmptyAlternativeId => write!(f, "alternative identifier must not be empty"),
            Self::DuplicateAlternativeId(id) => write!(f, "duplicate alternative identifier: {id}"),
            Self::MismatchedObservationCount { expected, actual } => {
                write!(
                    f,
                    "alternative has {actual} observations; expected {expected}"
                )
            }
            Self::NonFiniteResidual { index } => {
                write!(f, "expected residual at index {index} is not finite")
            }
            Self::InvalidSigma => write!(f, "noise sigma must be finite and strictly positive"),
            Self::InvalidAlpha => write!(f, "alpha must be finite and in (1e-12, 1 - 1e-12)"),
            Self::InvalidTargetPower => write!(
                f,
                "target power must be finite and strictly between alpha and 1"
            ),
        }
    }
}

impl Error for PowerError {}

/// One evaluator-declared alternative relative to the working model.
///
/// expected_residuals[i] is alternative prediction minus working-model
/// prediction at the i-th observation in the fixed design. These values are
/// evaluator-side data and must not be exposed to the research agent.
#[derive(Debug, Clone, PartialEq)]
pub struct DeclaredAlternative {
    pub id: String,
    pub expected_residuals: Vec<f64>,
}

impl DeclaredAlternative {
    pub fn new(id: impl Into<String>, expected_residuals: Vec<f64>) -> Self {
        Self {
            id: id.into(),
            expected_residuals,
        }
    }
}

/// Best-case power for one known discrepancy direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerResult {
    pub observations: usize,
    pub noncentrality: f64,
    pub critical_value: f64,
    pub power: f64,
}

/// Whether one alternative reaches the predeclared power target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerDisposition {
    TargetMet,
    Underpowered,
}

/// One row in a power-envelope report.
#[derive(Debug, Clone, PartialEq)]
pub struct AlternativePower {
    pub id: String,
    pub result: PowerResult,
    pub disposition: PowerDisposition,
}

/// Alternative-specific report for one fixed observation design.
///
/// target_met is true only when every declared alternative reaches the target.
/// The minimum is never hidden by averaging across alternatives.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerEnvelopeReport {
    pub alpha: f64,
    pub target_power: f64,
    pub minimum_power: f64,
    pub worst_case_alternative: String,
    pub target_met: bool,
    pub underpowered_alternatives: Vec<String>,
    pub alternatives: Vec<AlternativePower>,
}

/// Compute best-case power for a fixed, evaluator-declared Gaussian shift.
///
/// Assumptions:
/// - observations are independent with common, known standard deviation sigma;
/// - the null mean is the working-model prediction;
/// - the full discrepancy direction is known to the evaluator;
/// - the test is the one-sided likelihood-ratio test aligned with that known direction;
/// - the design, alpha and stopping rule are fixed (no sequential peeking).
///
/// This is an oracle diagnostic, not an operational adequacy test. Estimating
/// nuisance parameters, unknown/correlated noise, adaptive stopping,
/// multiplicity and imperfect discrepancy specification can lower actual
/// power and require separate qualification.
pub fn known_direction_power(
    expected_residuals: &[f64],
    sigma: f64,
    alpha: f64,
) -> Result<PowerResult, PowerError> {
    validate_sigma(sigma)?;
    validate_alpha(alpha)?;
    if expected_residuals.is_empty() {
        return Err(PowerError::EmptyObservationDesign);
    }

    let mut squared_signal_to_noise = 0.0;
    for (index, residual) in expected_residuals.iter().copied().enumerate() {
        if !residual.is_finite() {
            return Err(PowerError::NonFiniteResidual { index });
        }
        let standardized = residual / sigma;
        squared_signal_to_noise += standardized * standardized;
    }

    let noncentrality = squared_signal_to_noise.sqrt();
    let critical_value = normal_quantile(1.0 - alpha);
    let power = one_sided_power(noncentrality, critical_value);

    Ok(PowerResult {
        observations: expected_residuals.len(),
        noncentrality,
        critical_value,
        power,
    })
}

/// Evaluate every declared alternative under the same fixed design size.
///
/// All alternatives must provide one expected residual per observation in the
/// same design order. This checks vector lengths only: callers must bind the
/// same observation IDs, units, sensor basis and acquisition-cost accounting.
pub fn evaluate_power_envelope(
    alternatives: &[DeclaredAlternative],
    sigma: f64,
    alpha: f64,
    target_power: f64,
) -> Result<PowerEnvelopeReport, PowerError> {
    validate_sigma(sigma)?;
    validate_alpha(alpha)?;
    validate_target_power(alpha, target_power)?;
    let Some(first) = alternatives.first() else {
        return Err(PowerError::EmptyAlternativeSet);
    };
    let expected_observations = first.expected_residuals.len();
    if expected_observations == 0 {
        return Err(PowerError::EmptyObservationDesign);
    }

    let mut seen = BTreeSet::new();
    let mut rows = Vec::with_capacity(alternatives.len());
    for alternative in alternatives {
        if alternative.id.trim().is_empty() {
            return Err(PowerError::EmptyAlternativeId);
        }
        if !seen.insert(alternative.id.clone()) {
            return Err(PowerError::DuplicateAlternativeId(alternative.id.clone()));
        }
        if alternative.expected_residuals.len() != expected_observations {
            return Err(PowerError::MismatchedObservationCount {
                expected: expected_observations,
                actual: alternative.expected_residuals.len(),
            });
        }
        let result = known_direction_power(&alternative.expected_residuals, sigma, alpha)?;
        let disposition = if result.power >= target_power {
            PowerDisposition::TargetMet
        } else {
            PowerDisposition::Underpowered
        };
        rows.push(AlternativePower {
            id: alternative.id.clone(),
            result,
            disposition,
        });
    }

    let worst = rows
        .iter()
        .min_by(|left, right| left.result.power.total_cmp(&right.result.power))
        .expect("the non-empty alternative set was checked above");
    let minimum_power = worst.result.power;
    let worst_case_alternative = worst.id.clone();
    let underpowered_alternatives = rows
        .iter()
        .filter(|row| row.disposition == PowerDisposition::Underpowered)
        .map(|row| row.id.clone())
        .collect::<Vec<_>>();

    Ok(PowerEnvelopeReport {
        alpha,
        target_power,
        minimum_power,
        worst_case_alternative,
        target_met: underpowered_alternatives.is_empty(),
        underpowered_alternatives,
        alternatives: rows,
    })
}

/// Return the minimum number of iid replicates needed for a constant,
/// evaluator-known mean discrepancy to reach target_power.
///
/// None means no finite replicate count can reach the target (for example,
/// a zero discrepancy). The calculation uses the same known-variance,
/// one-sided Gaussian oracle as known_direction_power.
pub fn minimum_replicates_for_power(
    expected_residual: f64,
    sigma: f64,
    alpha: f64,
    target_power: f64,
) -> Result<Option<u64>, PowerError> {
    validate_sigma(sigma)?;
    validate_alpha(alpha)?;
    validate_target_power(alpha, target_power)?;
    if !expected_residual.is_finite() {
        return Err(PowerError::NonFiniteResidual { index: 0 });
    }
    if expected_residual == 0.0 {
        return Ok(None);
    }

    let effect_size = expected_residual.abs() / sigma;
    if !effect_size.is_finite() {
        return Ok(Some(1));
    }

    let critical_value = normal_quantile(1.0 - alpha);
    let power_at = |n: u64| one_sided_power(effect_size * (n as f64).sqrt(), critical_value);

    let mut high = 1_u64;
    while power_at(high) < target_power {
        if high > u64::MAX / 2 {
            return Ok(None);
        }
        high *= 2;
    }

    let mut low = 0_u64;
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if power_at(middle) >= target_power {
            high = middle;
        } else {
            low = middle;
        }
    }
    Ok(Some(high))
}

fn validate_sigma(sigma: f64) -> Result<(), PowerError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        Err(PowerError::InvalidSigma)
    } else {
        Ok(())
    }
}

fn validate_alpha(alpha: f64) -> Result<(), PowerError> {
    if !alpha.is_finite() || alpha <= 1.0e-12 || alpha >= 1.0 - 1.0e-12 {
        Err(PowerError::InvalidAlpha)
    } else {
        Ok(())
    }
}

fn validate_target_power(alpha: f64, target: f64) -> Result<(), PowerError> {
    if !target.is_finite() || target <= alpha || target >= 1.0 {
        Err(PowerError::InvalidTargetPower)
    } else {
        Ok(())
    }
}

fn one_sided_power(noncentrality: f64, critical_value: f64) -> f64 {
    if noncentrality.is_infinite() {
        return 1.0;
    }
    (1.0 - normal_cdf(critical_value - noncentrality)).clamp(0.0, 1.0)
}

/// Standard normal CDF using the Abramowitz-Stegun 7.1.26 approximation.
fn normal_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.5;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }

    let absolute = x.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * absolute);
    let density = 0.398_942_280_401_432_7 * (-0.5 * absolute * absolute).exp();
    let polynomial = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let tail = density * polynomial;
    if x > 0.0 { 1.0 - tail } else { tail }
}

/// Acklam's rational approximation to the standard normal quantile.
fn normal_quantile(probability: f64) -> f64 {
    const A: [f64; 6] = [
        -39.696_830_286_653_76,
        220.946_098_424_520_5,
        -275.928_510_446_968_7,
        138.357_751_867_269,
        -30.664_798_066_147_16,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -54.476_098_798_224_06,
        161.585_836_858_040_9,
        -155.698_979_859_886_6,
        66.801_311_887_719_72,
        -13.280_681_552_885_72,
    ];
    const C: [f64; 6] = [
        -0.007_784_894_002_430_293,
        -0.322_396_458_041_136_5,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        0.007_784_695_709_041_462,
        0.322_467_129_070_039_8,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.024_25;
    const HIGH: f64 = 1.0 - LOW;

    if probability < LOW {
        let q = (-2.0 * probability.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if probability <= HIGH {
        let q = probability - 0.5;
        let r = q * q;
        return (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    }

    let q = (-2.0 * (1.0 - probability).ln()).sqrt();
    -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
        / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
}

/// Errors for sequential likelihood-ratio evidence updates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequentialError {
    EmptyIdentifier,
    InvalidAlpha,
    InvalidSigma,
    NonFiniteDiscrepancy,
    NonFiniteObservation,
    PredictionAlreadyPending,
    NoPredictionPending,
    ObservationIdentityMismatch,
    DuplicateObservationId,
    NumericalOverflow,
    EmptyMixture,
    DuplicateAlternativeId(String),
    InvalidMixtureWeight(String),
    MixtureWeightsNotNormalizable,
    MixturePredictionCountMismatch,
    MissingMixturePrediction(String),
    DuplicateMixturePrediction(String),
}

impl fmt::Display for SequentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Error for SequentialError {}

#[derive(Debug, Clone, PartialEq)]
struct PendingGaussianPrediction {
    observation_id: String,
    decision_digest: String,
    expected_discrepancy: f64,
    sigma: f64,
}

/// One sequential update. The running maximum, not the current e-value,
/// determines the anytime p-value and is never reset by adverse evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SequentialUpdate {
    pub observations: u64,
    pub log_e_value: f64,
    pub running_max_log_e_value: f64,
    pub anytime_p_value: f64,
    pub crossed_alpha: bool,
}

/// Predict-before-observe Gaussian likelihood-ratio e-process for one
/// evaluator-declared alternative to a simple conditional Gaussian null.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictableGaussianEProcess {
    alternative_id: String,
    alpha: f64,
    pending: Option<PendingGaussianPrediction>,
    consumed_observation_ids: BTreeSet<String>,
    observations: u64,
    log_e_value: f64,
    running_max_log_e_value: f64,
}

impl PredictableGaussianEProcess {
    pub fn new(alternative_id: impl Into<String>, alpha: f64) -> Result<Self, SequentialError> {
        let alternative_id = alternative_id.into();
        validate_sequential_identifier(&alternative_id)?;
        validate_sequential_alpha(alpha)?;
        Ok(Self {
            alternative_id,
            alpha,
            pending: None,
            consumed_observation_ids: BTreeSet::new(),
            observations: 0,
            log_e_value: 0.0,
            running_max_log_e_value: 0.0,
        })
    }

    pub fn alternative_id(&self) -> &str {
        &self.alternative_id
    }

    pub fn observations(&self) -> u64 {
        self.observations
    }

    pub fn log_e_value(&self) -> f64 {
        self.log_e_value
    }

    pub fn running_max_log_e_value(&self) -> f64 {
        self.running_max_log_e_value
    }

    pub fn commit_prediction(
        &mut self,
        observation_id: impl Into<String>,
        decision_digest: impl Into<String>,
        expected_discrepancy: f64,
        sigma: f64,
    ) -> Result<(), SequentialError> {
        if self.pending.is_some() {
            return Err(SequentialError::PredictionAlreadyPending);
        }
        let observation_id = observation_id.into();
        let decision_digest = decision_digest.into();
        validate_sequential_identifier(&observation_id)?;
        validate_sequential_identifier(&decision_digest)?;
        if self.consumed_observation_ids.contains(&observation_id) {
            return Err(SequentialError::DuplicateObservationId);
        }
        if !expected_discrepancy.is_finite() {
            return Err(SequentialError::NonFiniteDiscrepancy);
        }
        validate_sequential_sigma(sigma)?;

        self.pending = Some(PendingGaussianPrediction {
            observation_id,
            decision_digest,
            expected_discrepancy,
            sigma,
        });
        Ok(())
    }

    pub fn observe(
        &mut self,
        observation_id: &str,
        decision_digest: &str,
        observed_residual: f64,
    ) -> Result<SequentialUpdate, SequentialError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or(SequentialError::NoPredictionPending)?;
        if pending.observation_id != observation_id || pending.decision_digest != decision_digest {
            return Err(SequentialError::ObservationIdentityMismatch);
        }
        if !observed_residual.is_finite() {
            return Err(SequentialError::NonFiniteObservation);
        }
        if self.consumed_observation_ids.contains(observation_id) {
            return Err(SequentialError::DuplicateObservationId);
        }

        let increment = gaussian_log_likelihood_ratio(
            pending.expected_discrepancy,
            observed_residual,
            pending.sigma,
        )?;
        let next_log_e_value = self.log_e_value + increment;
        if !next_log_e_value.is_finite() {
            return Err(SequentialError::NumericalOverflow);
        }
        let next_running_max = self.running_max_log_e_value.max(next_log_e_value);
        let next_observations = self
            .observations
            .checked_add(1)
            .ok_or(SequentialError::NumericalOverflow)?;

        self.log_e_value = next_log_e_value;
        self.running_max_log_e_value = next_running_max;
        self.observations = next_observations;
        self.consumed_observation_ids.insert(observation_id.to_owned());
        self.pending = None;

        Ok(self.snapshot())
    }

    pub fn snapshot(&self) -> SequentialUpdate {
        SequentialUpdate {
            observations: self.observations,
            log_e_value: self.log_e_value,
            running_max_log_e_value: self.running_max_log_e_value,
            anytime_p_value: anytime_p_value(self.running_max_log_e_value),
            crossed_alpha: self.running_max_log_e_value >= -self.alpha.ln(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MixtureAlternative {
    pub id: String,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MixtureDiscrepancy {
    pub alternative_id: String,
    pub expected_discrepancy: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct PendingMixturePrediction {
    observation_id: String,
    decision_digest: String,
    sigma: f64,
    expected_discrepancies: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MixtureSequentialUpdate {
    pub observations: u64,
    pub component_log_e_values: Vec<(String, f64)>,
    pub log_e_value: f64,
    pub running_max_log_e_value: f64,
    pub anytime_p_value: f64,
    pub crossed_alpha: bool,
}

/// Fixed, predeclared mixture over alternatives. Weights are normalized once
/// at construction and cannot be changed after outcomes become observable.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictableGaussianMixture {
    alternatives: Vec<MixtureAlternative>,
    alpha: f64,
    pending: Option<PendingMixturePrediction>,
    consumed_observation_ids: BTreeSet<String>,
    observations: u64,
    component_log_e_values: Vec<f64>,
    log_e_value: f64,
    running_max_log_e_value: f64,
}

impl PredictableGaussianMixture {
    pub fn new(
        alternatives: Vec<MixtureAlternative>,
        alpha: f64,
    ) -> Result<Self, SequentialError> {
        validate_sequential_alpha(alpha)?;
        if alternatives.is_empty() {
            return Err(SequentialError::EmptyMixture);
        }

        let mut seen = BTreeSet::new();
        let mut total_weight = 0.0_f64;
        for alternative in &alternatives {
            validate_sequential_identifier(&alternative.id)?;
            if !seen.insert(alternative.id.clone()) {
                return Err(SequentialError::DuplicateAlternativeId(
                    alternative.id.clone(),
                ));
            }
            if !alternative.weight.is_finite() || alternative.weight <= 0.0 {
                return Err(SequentialError::InvalidMixtureWeight(
                    alternative.id.clone(),
                ));
            }
            total_weight += alternative.weight;
            if !total_weight.is_finite() {
                return Err(SequentialError::MixtureWeightsNotNormalizable);
            }
        }
        if total_weight <= 0.0 {
            return Err(SequentialError::MixtureWeightsNotNormalizable);
        }

        let mut normalized = alternatives
            .into_iter()
            .map(|alternative| MixtureAlternative {
                id: alternative.id,
                weight: alternative.weight / total_weight,
            })
            .collect::<Vec<_>>();
        normalized.sort_by(|left, right| left.id.cmp(&right.id));

        // Make the stored binary64 weights sum to one in the same order used
        // by the audit API. The final weight absorbs normalization round-off.
        if normalized.len() > 1 {
            let prefix_sum = normalized[..normalized.len() - 1]
                .iter()
                .map(|alternative| alternative.weight)
                .sum::<f64>();
            let last = normalized.last_mut().expect("non-empty mixture");
            last.weight = 1.0 - prefix_sum;
        } else {
            normalized[0].weight = 1.0;
        }
        if normalized.iter().any(|alternative| {
            !alternative.weight.is_finite() || alternative.weight <= 0.0
        }) {
            return Err(SequentialError::MixtureWeightsNotNormalizable);
        }

        let component_count = normalized.len();
        Ok(Self {
            alternatives: normalized,
            alpha,
            pending: None,
            consumed_observation_ids: BTreeSet::new(),
            observations: 0,
            component_log_e_values: vec![0.0; component_count],
            log_e_value: 0.0,
            running_max_log_e_value: 0.0,
        })
    }

    pub fn alternatives(&self) -> &[MixtureAlternative] {
        &self.alternatives
    }

    pub fn observations(&self) -> u64 {
        self.observations
    }

    pub fn commit_prediction(
        &mut self,
        observation_id: impl Into<String>,
        decision_digest: impl Into<String>,
        sigma: f64,
        predictions: &[MixtureDiscrepancy],
    ) -> Result<(), SequentialError> {
        if self.pending.is_some() {
            return Err(SequentialError::PredictionAlreadyPending);
        }
        let observation_id = observation_id.into();
        let decision_digest = decision_digest.into();
        validate_sequential_identifier(&observation_id)?;
        validate_sequential_identifier(&decision_digest)?;
        if self.consumed_observation_ids.contains(&observation_id) {
            return Err(SequentialError::DuplicateObservationId);
        }
        validate_sequential_sigma(sigma)?;
        if predictions.len() != self.alternatives.len() {
            return Err(SequentialError::MixturePredictionCountMismatch);
        }

        let mut by_id = std::collections::BTreeMap::new();
        for prediction in predictions {
            validate_sequential_identifier(&prediction.alternative_id)?;
            if !prediction.expected_discrepancy.is_finite() {
                return Err(SequentialError::NonFiniteDiscrepancy);
            }
            if by_id
                .insert(
                    prediction.alternative_id.clone(),
                    prediction.expected_discrepancy,
                )
                .is_some()
            {
                return Err(SequentialError::DuplicateMixturePrediction(
                    prediction.alternative_id.clone(),
                ));
            }
        }
        let mut expected_discrepancies = Vec::with_capacity(self.alternatives.len());
        for alternative in &self.alternatives {
            let discrepancy = by_id
                .remove(&alternative.id)
                .ok_or_else(|| {
                    SequentialError::MissingMixturePrediction(alternative.id.clone())
                })?;
            expected_discrepancies.push(discrepancy);
        }
        if !by_id.is_empty() {
            return Err(SequentialError::MixturePredictionCountMismatch);
        }

        self.pending = Some(PendingMixturePrediction {
            observation_id,
            decision_digest,
            sigma,
            expected_discrepancies,
        });
        Ok(())
    }

    pub fn observe(
        &mut self,
        observation_id: &str,
        decision_digest: &str,
        observed_residual: f64,
    ) -> Result<MixtureSequentialUpdate, SequentialError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or(SequentialError::NoPredictionPending)?;
        if pending.observation_id != observation_id || pending.decision_digest != decision_digest {
            return Err(SequentialError::ObservationIdentityMismatch);
        }
        if !observed_residual.is_finite() {
            return Err(SequentialError::NonFiniteObservation);
        }
        if self.consumed_observation_ids.contains(observation_id) {
            return Err(SequentialError::DuplicateObservationId);
        }

        let mut next_components = Vec::with_capacity(self.alternatives.len());
        for index in 0..self.alternatives.len() {
            let increment = gaussian_log_likelihood_ratio(
                pending.expected_discrepancies[index],
                observed_residual,
                pending.sigma,
            )?;
            let next = self.component_log_e_values[index] + increment;
            if !next.is_finite() {
                return Err(SequentialError::NumericalOverflow);
            }
            next_components.push(next);
        }

        let next_mixture_log_e = mixture_log_e_value(&self.alternatives, &next_components)?;
        let next_running_max = self.running_max_log_e_value.max(next_mixture_log_e);
        let next_observations = self
            .observations
            .checked_add(1)
            .ok_or(SequentialError::NumericalOverflow)?;

        self.component_log_e_values = next_components;
        self.log_e_value = next_mixture_log_e;
        self.running_max_log_e_value = next_running_max;
        self.observations = next_observations;
        self.consumed_observation_ids.insert(observation_id.to_owned());
        self.pending = None;

        Ok(self.snapshot())
    }

    pub fn snapshot(&self) -> MixtureSequentialUpdate {
        MixtureSequentialUpdate {
            observations: self.observations,
            component_log_e_values: self
                .alternatives
                .iter()
                .zip(self.component_log_e_values.iter())
                .map(|(alternative, log_e)| (alternative.id.clone(), *log_e))
                .collect(),
            log_e_value: self.log_e_value,
            running_max_log_e_value: self.running_max_log_e_value,
            anytime_p_value: anytime_p_value(self.running_max_log_e_value),
            crossed_alpha: self.running_max_log_e_value >= -self.alpha.ln(),
        }
    }
}

fn validate_sequential_identifier(value: &str) -> Result<(), SequentialError> {
    if value.trim().is_empty() {
        Err(SequentialError::EmptyIdentifier)
    } else {
        Ok(())
    }
}

fn validate_sequential_alpha(alpha: f64) -> Result<(), SequentialError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        Err(SequentialError::InvalidAlpha)
    } else {
        Ok(())
    }
}

fn validate_sequential_sigma(sigma: f64) -> Result<(), SequentialError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        Err(SequentialError::InvalidSigma)
    } else {
        Ok(())
    }
}

fn gaussian_log_likelihood_ratio(
    expected_discrepancy: f64,
    observed_residual: f64,
    sigma: f64,
) -> Result<f64, SequentialError> {
    let variance = sigma * sigma;
    let product = expected_discrepancy * observed_residual;
    let squared_discrepancy = expected_discrepancy * expected_discrepancy;
    if !variance.is_finite()
        || variance <= 0.0
        || !product.is_finite()
        || !squared_discrepancy.is_finite()
    {
        return Err(SequentialError::NumericalOverflow);
    }
    let increment = product / variance - squared_discrepancy / (2.0 * variance);
    if !increment.is_finite() {
        return Err(SequentialError::NumericalOverflow);
    }
    Ok(increment)
}

fn mixture_log_e_value(
    alternatives: &[MixtureAlternative],
    component_log_e_values: &[f64],
) -> Result<f64, SequentialError> {
    if alternatives.len() != component_log_e_values.len() || alternatives.is_empty() {
        return Err(SequentialError::MixturePredictionCountMismatch);
    }
    let terms = alternatives
        .iter()
        .zip(component_log_e_values.iter())
        .map(|(alternative, component)| alternative.weight.ln() + component)
        .collect::<Vec<_>>();
    let maximum = terms
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return Err(SequentialError::NumericalOverflow);
    }
    let scaled_sum = terms
        .iter()
        .map(|term| (*term - maximum).exp())
        .sum::<f64>();
    let value = maximum + scaled_sum.ln();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SequentialError::NumericalOverflow)
    }
}

fn anytime_p_value(running_max_log_e_value: f64) -> f64 {
    if running_max_log_e_value <= 0.0 {
        return 1.0;
    }
    (-running_max_log_e_value)
        .exp()
        .max(f64::MIN_POSITIVE)
        .min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_gaussian_log_likelihood_ratio_matches_known_answer() {
        let mut process = PredictableGaussianEProcess::new("reflection-break-a", 0.05).unwrap();
        process
            .commit_prediction("obs-1", "decision-1", 2.0, 2.0)
            .unwrap();
        let update = process.observe("obs-1", "decision-1", 1.5).unwrap();
        assert!((update.log_e_value - 0.25).abs() < 1.0e-12);
        assert_eq!(update.running_max_log_e_value, 0.25);
        assert!(!update.crossed_alpha);
    }

    #[test]
    fn sequential_process_requires_predict_before_matching_observation() {
        let mut process = PredictableGaussianEProcess::new("rotation-break", 0.05).unwrap();
        assert_eq!(
            process.observe("obs-1", "decision-1", 0.2),
            Err(SequentialError::NoPredictionPending)
        );
        process
            .commit_prediction("obs-1", "decision-1", 0.3, 0.1)
            .unwrap();
        assert_eq!(
            process.observe("wrong-obs", "decision-1", 0.2),
            Err(SequentialError::ObservationIdentityMismatch)
        );
        assert_eq!(process.observations(), 0);
        assert_eq!(process.log_e_value(), 0.0);
        assert!(process.pending.is_some());
    }

    #[test]
    fn running_maximum_preserves_earlier_boundary_crossing() {
        let mut process = PredictableGaussianEProcess::new("mirror-break", 0.05).unwrap();
        process
            .commit_prediction("obs-1", "decision-1", 1.0, 1.0)
            .unwrap();
        let first = process.observe("obs-1", "decision-1", 8.0).unwrap();
        assert!(first.crossed_alpha);
        let max_before = first.running_max_log_e_value;

        process
            .commit_prediction("obs-2", "decision-2", 1.0, 1.0)
            .unwrap();
        let second = process.observe("obs-2", "decision-2", -8.0).unwrap();
        assert!(second.log_e_value < first.log_e_value);
        assert_eq!(second.running_max_log_e_value, max_before);
        assert_eq!(second.anytime_p_value, first.anytime_p_value);
        assert!(second.crossed_alpha);
    }

    #[test]
    fn invalid_sequential_update_does_not_consume_pending_prediction() {
        let mut process = PredictableGaussianEProcess::new("translation-break", 0.05).unwrap();
        process
            .commit_prediction("obs-1", "decision-1", f64::MAX, f64::MIN_POSITIVE)
            .unwrap();
        assert_eq!(
            process.observe("obs-1", "decision-1", f64::MAX),
            Err(SequentialError::NumericalOverflow)
        );
        assert_eq!(process.observations(), 0);
        assert!(process.pending.is_some());
        assert!(process.consumed_observation_ids.is_empty());
    }

    #[test]
    fn sequential_process_rejects_duplicate_observation_ids() {
        let mut process = PredictableGaussianEProcess::new("glide-break", 0.05).unwrap();
        process
            .commit_prediction("obs-1", "decision-1", 0.1, 1.0)
            .unwrap();
        process.observe("obs-1", "decision-1", 0.0).unwrap();
        assert_eq!(
            process.commit_prediction("obs-1", "decision-2", 0.1, 1.0),
            Err(SequentialError::DuplicateObservationId)
        );
    }

    #[test]
    fn fixed_mixture_normalizes_weights_and_matches_component_sum() {
        let mut mixture = PredictableGaussianMixture::new(
            vec![
                MixtureAlternative { id: "break-x".to_owned(), weight: 2.0 },
                MixtureAlternative { id: "break-y".to_owned(), weight: 1.0 },
            ],
            0.05,
        )
        .unwrap();
        let weights = mixture.alternatives().iter().map(|item| item.weight).sum::<f64>();
        assert!((weights - 1.0).abs() < 1.0e-15);

        let predictions = [
            MixtureDiscrepancy { alternative_id: "break-y".to_owned(), expected_discrepancy: 0.3 },
            MixtureDiscrepancy { alternative_id: "break-x".to_owned(), expected_discrepancy: 0.1 },
        ];
        mixture
            .commit_prediction("obs-1", "decision-1", 0.2, &predictions)
            .unwrap();
        let update = mixture.observe("obs-1", "decision-1", 0.25).unwrap();

        let direct = mixture
            .alternatives()
            .iter()
            .map(|alternative| {
                let component = update
                    .component_log_e_values
                    .iter()
                    .find(|(id, _)| id == &alternative.id)
                    .unwrap()
                    .1;
                alternative.weight * component.exp()
            })
            .sum::<f64>()
            .ln();
        assert!((update.log_e_value - direct).abs() < 1.0e-12);
    }

    #[test]
    fn fixed_mixture_rejects_weights_that_underflow_to_zero_when_normalized() {
        assert_eq!(
            PredictableGaussianMixture::new(
                vec![
                    MixtureAlternative {
                        id: "tiny".to_owned(),
                        weight: f64::MIN_POSITIVE,
                    },
                    MixtureAlternative {
                        id: "large".to_owned(),
                        weight: f64::MAX,
                    },
                ],
                0.05,
            ),
            Err(SequentialError::MixtureWeightsNotNormalizable)
        );
    }

    #[test]
    fn fixed_mixture_rejects_missing_duplicate_and_nonfinite_predictions() {
        let mut mixture = PredictableGaussianMixture::new(
            vec![
                MixtureAlternative { id: "break-a".to_owned(), weight: 0.5 },
                MixtureAlternative { id: "break-b".to_owned(), weight: 0.5 },
            ],
            0.05,
        )
        .unwrap();
        assert_eq!(
            mixture.commit_prediction("obs-1", "decision-1", 1.0, &[
                MixtureDiscrepancy { alternative_id: "break-a".to_owned(), expected_discrepancy: 0.2 }
            ]),
            Err(SequentialError::MixturePredictionCountMismatch)
        );
        assert!(mixture.pending.is_none());
        assert_eq!(
            mixture.commit_prediction("obs-1", "decision-1", 1.0, &[
                MixtureDiscrepancy { alternative_id: "break-a".to_owned(), expected_discrepancy: 0.2 },
                MixtureDiscrepancy { alternative_id: "break-a".to_owned(), expected_discrepancy: 0.3 }
            ]),
            Err(SequentialError::DuplicateMixturePrediction("break-a".to_owned()))
        );
        assert!(mixture.pending.is_none());
        assert_eq!(
            mixture.commit_prediction("obs-1", "decision-1", 1.0, &[
                MixtureDiscrepancy { alternative_id: "break-a".to_owned(), expected_discrepancy: 0.2 },
                MixtureDiscrepancy { alternative_id: "break-b".to_owned(), expected_discrepancy: f64::NAN }
            ]),
            Err(SequentialError::NonFiniteDiscrepancy)
        );
        assert!(mixture.pending.is_none());
    }

    #[test]
    fn fixed_mixture_is_order_independent_and_rejects_posthoc_identity_change() {
        let alternatives = vec![
            MixtureAlternative { id: "break-b".to_owned(), weight: 0.4 },
            MixtureAlternative { id: "break-a".to_owned(), weight: 0.6 },
        ];
        let mut mixture = PredictableGaussianMixture::new(alternatives, 0.05).unwrap();
        assert_eq!(mixture.alternatives()[0].id, "break-a");
        assert_eq!(mixture.alternatives()[1].id, "break-b");
        assert_eq!(
            mixture.commit_prediction("obs-1", "decision-1", 1.0, &[
                MixtureDiscrepancy { alternative_id: "break-a".to_owned(), expected_discrepancy: 0.2 },
                MixtureDiscrepancy { alternative_id: "unregistered".to_owned(), expected_discrepancy: 0.3 }
            ]),
            Err(SequentialError::MissingMixturePrediction("break-b".to_owned()))
        );
        assert!(mixture.pending.is_none());
    }

    #[test]
    fn nominal_null_power_matches_alpha() {
        let result = known_direction_power(&[0.0, 0.0, 0.0], 0.15, 0.05).unwrap();
        assert!((result.power - 0.05).abs() < 2.0e-7);
    }

    #[test]
    fn critical_value_matches_one_sided_five_percent_test() {
        let result = known_direction_power(&[0.1], 0.15, 0.05).unwrap();
        assert!((result.critical_value - 1.644_853_626_95).abs() < 1.0e-7);
    }

    #[test]
    fn known_direction_power_matches_normal_reference_value() {
        let result = known_direction_power(&[0.24, 0.24, 0.24], 0.15, 0.05).unwrap();
        assert!((result.power - 0.870_007_728_878_440_5).abs() < 2.0e-7);
    }

    #[test]
    fn reversing_the_declared_discrepancy_does_not_change_one_sided_power() {
        let positive = known_direction_power(&[0.1, 0.2, 0.3], 0.15, 0.05).unwrap();
        let negative = known_direction_power(&[-0.1, -0.2, -0.3], 0.15, 0.05).unwrap();
        assert!((positive.power - negative.power).abs() < 1.0e-12);
    }

    #[test]
    fn additional_informative_observations_do_not_reduce_oracle_power() {
        let one = known_direction_power(&[0.12], 0.15, 0.05).unwrap();
        let two = known_direction_power(&[0.12, 0.12], 0.15, 0.05).unwrap();
        let three = known_direction_power(&[0.12, 0.12, 0.12], 0.15, 0.05).unwrap();
        assert!(one.power < two.power);
        assert!(two.power < three.power);
    }

    #[test]
    fn minimum_replicates_is_the_first_count_to_reach_target() {
        let count = minimum_replicates_for_power(0.24, 0.15, 0.05, 0.80)
            .unwrap()
            .unwrap();
        assert_eq!(count, 3);
        let two = known_direction_power(&[0.24; 2], 0.15, 0.05).unwrap();
        let three = known_direction_power(&[0.24; 3], 0.15, 0.05).unwrap();
        assert!(two.power < 0.80);
        assert!(three.power >= 0.80);
    }

    #[test]
    fn envelope_preserves_each_alternative_and_uses_the_worst_case() {
        let alternatives = [
            DeclaredAlternative::new("early", vec![0.24, 0.24, 0.24, 0.24]),
            DeclaredAlternative::new("late", vec![0.01, 0.01, 0.01, 0.01]),
        ];
        let report = evaluate_power_envelope(&alternatives, 0.15, 0.05, 0.80).unwrap();
        assert_eq!(report.worst_case_alternative, "late");
        assert_eq!(report.underpowered_alternatives, vec!["late"]);
        assert!(!report.target_met);
        assert_eq!(report.alternatives.len(), 2);
    }

    #[test]
    fn envelope_rejects_unequal_observation_budgets() {
        let alternatives = [
            DeclaredAlternative::new("a", vec![0.1, 0.2]),
            DeclaredAlternative::new("b", vec![0.1]),
        ];
        assert!(matches!(
            evaluate_power_envelope(&alternatives, 0.15, 0.05, 0.80),
            Err(PowerError::MismatchedObservationCount { .. })
        ));
    }

    #[test]
    fn zero_discrepancy_cannot_reach_power_above_alpha() {
        assert_eq!(
            minimum_replicates_for_power(0.0, 0.15, 0.05, 0.80).unwrap(),
            None
        );
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        assert_eq!(
            known_direction_power(&[], 0.15, 0.05),
            Err(PowerError::EmptyObservationDesign)
        );
        assert_eq!(
            known_direction_power(&[0.1], 0.0, 0.05),
            Err(PowerError::InvalidSigma)
        );
        assert_eq!(
            known_direction_power(&[f64::NAN], 0.15, 0.05),
            Err(PowerError::NonFiniteResidual { index: 0 })
        );
        assert_eq!(
            known_direction_power(&[0.1], 0.15, 0.0),
            Err(PowerError::InvalidAlpha)
        );
        assert_eq!(
            minimum_replicates_for_power(0.1, 0.15, 0.05, 0.05),
            Err(PowerError::InvalidTargetPower)
        );
    }

    #[test]
    fn envelope_rejects_duplicate_ids() {
        let alternatives = [
            DeclaredAlternative::new("same", vec![0.1]),
            DeclaredAlternative::new("same", vec![0.2]),
        ];
        assert_eq!(
            evaluate_power_envelope(&alternatives, 0.15, 0.05, 0.80),
            Err(PowerError::DuplicateAlternativeId("same".to_owned()))
        );
    }
}
