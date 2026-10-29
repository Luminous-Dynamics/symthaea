//! First-order component-envelope calculations for photonic architecture co-design.
//!
//! These functions model only a declared primitive-output yield: source availability,
//! independent optical-path survival, detector efficiency, and conditional gate-success
//! factors. They do not model state fidelity, correlated errors, quantum interference,
//! GKP quality, error correction, or a logical operation. Architecture families must
//! supply family-specific models; do not collapse them into one universal score.

use std::error::Error;
use std::fmt;

/// Convert nonnegative insertion loss in dB into power/photon transmission.
pub fn transmission_probability(loss_db: f64) -> Result<f64, PhotonicEnvelopeError> {
    if !loss_db.is_finite() || loss_db < 0.0 {
        return Err(PhotonicEnvelopeError::InvalidOpticalLoss(loss_db));
    }
    Ok(10.0_f64.powf(-loss_db / 10.0))
}

/// Probability range for a component parameter, ordered lower <= nominal <= upper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbabilityInterval {
    /// Conservative lower estimate.
    pub lower: f64,
    /// Nominal estimate used by the central scenario.
    pub nominal: f64,
    /// Conservative upper estimate.
    pub upper: f64,
}

impl ProbabilityInterval {
    /// Construct and validate a probability interval in [0, 1].
    pub fn new(lower: f64, nominal: f64, upper: f64) -> Result<Self, PhotonicEnvelopeError> {
        for value in [lower, nominal, upper] {
            validate_probability(value, "probability interval")?;
        }
        if lower > nominal || nominal > upper {
            return Err(PhotonicEnvelopeError::InvalidProbabilityInterval);
        }
        Ok(Self { lower, nominal, upper })
    }
}

/// Validate a finite probability in [0, 1].
fn validate_probability(value: f64, field: &'static str) -> Result<(), PhotonicEnvelopeError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(PhotonicEnvelopeError::InvalidProbability { field, value })
    }
}

/// One required photon channel through a declared optical path.
///
/// insertion_loss_db includes only the declared path boundary. Detector efficiency is
/// separate and must not be double-counted in insertion loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpticalChannel {
    /// Combined insertion/propagation/coupling loss assigned to this path, in dB.
    pub insertion_loss_db: f64,
    /// Terminal detector efficiency, if this path ends in a required photon detection.
    /// Use None for a photon that must survive the path but is not detected here.
    pub detector_efficiency: Option<f64>,
}

impl OpticalChannel {
    /// Create a required optical channel and validate its declared component values.
    pub fn new(
        insertion_loss_db: f64,
        detector_efficiency: Option<f64>,
    ) -> Result<Self, PhotonicEnvelopeError> {
        transmission_probability(insertion_loss_db)?;
        if let Some(efficiency) = detector_efficiency {
            validate_probability(efficiency, "detector efficiency")?;
        }
        Ok(Self { insertion_loss_db, detector_efficiency })
    }

    /// Survival probability through the declared optical path, excluding detection.
    pub fn transmission_probability(&self) -> Result<f64, PhotonicEnvelopeError> {
        transmission_probability(self.insertion_loss_db)
    }

    /// Probability of surviving the path and, if present, being detected.
    pub fn detected_survival_probability(&self) -> Result<f64, PhotonicEnvelopeError> {
        let transmission = self.transmission_probability()?;
        Ok(transmission * self.detector_efficiency.unwrap_or(1.0))
    }
}

/// A single-shot primitive model with an all-or-nothing acceptance condition.
///
/// Assumption: source validity, listed channel survival/detection, and conditional
/// gate-success events are independent. Every listed channel and gate must succeed
/// in one trial; a failure restarts the whole primitive. This is deliberately not
/// a model of adaptive repeat-until-success (RUS), feedforward or fault tolerance.
#[derive(Debug, Clone, PartialEq)]
pub struct PhotonicPrimitiveEnvelope {
    /// Probability that one source-stage attempt creates the required valid input state.
    pub source_success_probability: f64,
    /// Optical paths whose photons are all required for acceptance.
    pub required_channels: Vec<OpticalChannel>,
    /// Conditional success probability of each gate that must succeed this trial.
    pub conditional_gate_success_probabilities: Vec<f64>,
    /// Aggregate primitive attempts per second for the declared system boundary.
    pub attempts_per_second: f64,
    /// Photons consumed/emitted per whole primitive attempt, including a failed attempt.
    pub photons_per_attempt: u64,
}

impl PhotonicPrimitiveEnvelope {
    /// Validate a one-shot primitive model before it is evaluated.
    pub fn validate(&self) -> Result<(), PhotonicEnvelopeError> {
        validate_probability(self.source_success_probability, "source success")?;
        for probability in &self.conditional_gate_success_probabilities {
            validate_probability(*probability, "conditional gate success")?;
        }
        for channel in &self.required_channels {
            transmission_probability(channel.insertion_loss_db)?;
            if let Some(efficiency) = channel.detector_efficiency {
                validate_probability(efficiency, "detector efficiency")?;
            }
        }
        if !self.attempts_per_second.is_finite() || self.attempts_per_second <= 0.0 {
            return Err(PhotonicEnvelopeError::InvalidAttemptRate(self.attempts_per_second));
        }
        if self.photons_per_attempt == 0 {
            return Err(PhotonicEnvelopeError::ZeroPhotonsPerAttempt);
        }
        Ok(())
    }

    /// Calculate first-order event yield and resource expectation under this model.
    pub fn estimate(&self) -> Result<PrimitiveEnvelopeEstimate, PhotonicEnvelopeError> {
        self.validate()?;
        let mut path_transmission_probability = 1.0;
        let mut detector_yield_probability = 1.0;
        for channel in &self.required_channels {
            path_transmission_probability *= channel.transmission_probability()?;
            detector_yield_probability *= channel.detector_efficiency.unwrap_or(1.0);
        }
        let conditional_gate_success_probability =
            self.conditional_gate_success_probabilities.iter().product::<f64>();
        let accepted_event_probability = self.source_success_probability
            * path_transmission_probability
            * detector_yield_probability
            * conditional_gate_success_probability;

        let (
            expected_attempts_per_accepted_output,
            expected_seconds_per_accepted_output,
            expected_photons_per_accepted_output,
        ) = if accepted_event_probability > 0.0 {
            let expected_attempts = 1.0 / accepted_event_probability;
            (
                Some(expected_attempts),
                Some(expected_attempts / self.attempts_per_second),
                Some(expected_attempts * self.photons_per_attempt as f64),
            )
        } else {
            (None, None, None)
        };

        Ok(PrimitiveEnvelopeEstimate {
            path_transmission_probability,
            detector_yield_probability,
            conditional_gate_success_probability,
            accepted_event_probability,
            expected_attempts_per_accepted_output,
            expected_seconds_per_accepted_output,
            expected_photons_per_accepted_output,
        })
    }

    /// Attempts needed to reach a target probability of at least one accepted output,
    /// assuming independent identical whole-primitive trials.
    ///
    /// Returns None when this model assigns zero probability to acceptance.
    pub fn attempts_for_target_success_probability(
        &self,
        target_probability: f64,
    ) -> Result<Option<u64>, PhotonicEnvelopeError> {
        validate_probability(target_probability, "target success probability")?;
        if target_probability <= 0.0 {
            return Err(PhotonicEnvelopeError::InvalidTargetSuccessProbability(target_probability));
        }
        let probability = self.estimate()?.accepted_event_probability;
        if probability <= 0.0 {
            return Ok(None);
        }
        if probability >= 1.0 {
            return Ok(Some(1));
        }
        if target_probability >= 1.0 {
            return Err(PhotonicEnvelopeError::CertainTargetRequiresInfiniteAttempts);
        }
        let raw = ((1.0 - target_probability).ln() / (1.0 - probability).ln()).ceil();
        if !raw.is_finite() || raw > u64::MAX as f64 {
            return Err(PhotonicEnvelopeError::AttemptCountOverflow);
        }
        Ok(Some((raw as u64).max(1)))
    }
}

/// First-order output from a photonic primitive envelope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrimitiveEnvelopeEstimate {
    /// Joint transmission of all required photon paths, before detector efficiency.
    pub path_transmission_probability: f64,
    /// Product of detector efficiencies for channels with a declared detector.
    pub detector_yield_probability: f64,
    /// Product of conditional gate-success probabilities.
    pub conditional_gate_success_probability: f64,
    /// Approximate accepted-output probability per whole-primitive attempt.
    pub accepted_event_probability: f64,
    /// Expected iid whole-primitive attempts per accepted output; None if impossible.
    pub expected_attempts_per_accepted_output: Option<f64>,
    /// Expected wait at the declared aggregate attempt rate; None if impossible.
    pub expected_seconds_per_accepted_output: Option<f64>,
    /// Expected photons consumed for one accepted output under the restart model.
    pub expected_photons_per_accepted_output: Option<f64>,
}

/// Approximate chance that at least one of parallel_sources independent sources
/// produces a valid output and the selected output survives the switching loss.
///
/// This idealized envelope does not model simultaneous-output collisions, source
/// distinguishability, switching latency, detector conditioning, or multiphoton noise.
pub fn multiplexed_source_success_probability(
    single_source_success_probability: f64,
    parallel_sources: usize,
    switch_loss_db: f64,
) -> Result<f64, PhotonicEnvelopeError> {
    validate_probability(single_source_success_probability, "single source success")?;
    if parallel_sources == 0 {
        return Err(PhotonicEnvelopeError::ZeroParallelSources);
    }
    let switch_transmission = transmission_probability(switch_loss_db)?;
    let at_least_one_source =
        1.0 - (1.0 - single_source_success_probability).powf(parallel_sources as f64);
    Ok((at_least_one_source * switch_transmission).clamp(0.0, 1.0))
}

/// Errors returned by component-envelope calculations.
#[derive(Debug, Clone, PartialEq)]
pub enum PhotonicEnvelopeError {
    /// Invalid probability or probability interval member.
    InvalidProbability { field: &'static str, value: f64 },
    /// Interval endpoints are not ordered.
    InvalidProbabilityInterval,
    /// Invalid or negative optical loss, in dB.
    InvalidOpticalLoss(f64),
    /// Invalid aggregate attempt rate.
    InvalidAttemptRate(f64),
    /// A resource event cannot have zero declared photon cost.
    ZeroPhotonsPerAttempt,
    /// Multiplexing requires at least one source.
    ZeroParallelSources,
    /// Target success probability must lie in (0, 1].
    InvalidTargetSuccessProbability(f64),
    /// Required attempt count exceeds the supported integer range.
    AttemptCountOverflow,
    /// A probability below one cannot reach certainty in a finite number of iid attempts.
    CertainTargetRequiresInfiniteAttempts,
}

impl fmt::Display for PhotonicEnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProbability { field, value } =>
                write!(f, "invalid probability for {field}: {value}"),
            Self::InvalidProbabilityInterval =>
                write!(f, "probability interval must satisfy lower <= nominal <= upper"),
            Self::InvalidOpticalLoss(loss) =>
                write!(f, "optical loss must be finite and nonnegative, got {loss} dB"),
            Self::InvalidAttemptRate(rate) =>
                write!(f, "attempt rate must be finite and positive, got {rate} Hz"),
            Self::ZeroPhotonsPerAttempt =>
                write!(f, "photons per attempt must be positive"),
            Self::ZeroParallelSources =>
                write!(f, "parallel source count must be positive"),
            Self::InvalidTargetSuccessProbability(target) =>
                write!(f, "target success probability must be in (0, 1], got {target}"),
            Self::AttemptCountOverflow =>
                write!(f, "required attempt count overflows u64"),
            Self::CertainTargetRequiresInfiniteAttempts =>
                write!(f, "target success probability of one needs certainty or infinitely many attempts"),
        }
    }
}

impl Error for PhotonicEnvelopeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(left: f64, right: f64) {
        assert!((left - right).abs() <= 1e-10, "left={left}, right={right}");
    }

    fn simple_model(probability: f64) -> PhotonicPrimitiveEnvelope {
        PhotonicPrimitiveEnvelope {
            source_success_probability: probability,
            required_channels: Vec::new(),
            conditional_gate_success_probabilities: Vec::new(),
            attempts_per_second: 1_000.0,
            photons_per_attempt: 2,
        }
    }

    #[test]
    fn decibels_convert_to_expected_transmission() {
        near(transmission_probability(0.0).unwrap(), 1.0);
        near(transmission_probability(3.01029995664).unwrap(), 0.5);
        near(transmission_probability(10.0).unwrap(), 0.1);
        assert!(transmission_probability(-0.1).is_err());
        assert!(transmission_probability(f64::NAN).is_err());
        assert!(transmission_probability(f64::INFINITY).is_err());
    }

    #[test]
    fn component_yield_keeps_loss_detection_and_gate_success_separate() {
        let model = PhotonicPrimitiveEnvelope {
            source_success_probability: 0.9,
            required_channels: vec![OpticalChannel::new(3.01029995664, Some(0.8)).unwrap()],
            conditional_gate_success_probabilities: vec![0.5],
            attempts_per_second: 1_000.0,
            photons_per_attempt: 2,
        };
        let result = model.estimate().unwrap();
        near(result.path_transmission_probability, 0.5);
        near(result.detector_yield_probability, 0.8);
        near(result.conditional_gate_success_probability, 0.5);
        near(result.accepted_event_probability, 0.18);
        near(result.expected_attempts_per_accepted_output.unwrap(), 1.0 / 0.18);
        near(result.expected_seconds_per_accepted_output.unwrap(), 1.0 / 180.0);
        near(result.expected_photons_per_accepted_output.unwrap(), 2.0 / 0.18);
    }

    #[test]
    fn multiplexing_has_expected_idealized_source_yield() {
        near(multiplexed_source_success_probability(0.1, 4, 0.0).unwrap(), 0.3439);
        near(
            multiplexed_source_success_probability(0.1, 4, 10.0).unwrap(),
            0.03439,
        );
        assert!(multiplexed_source_success_probability(0.1, 0, 0.0).is_err());
    }

    #[test]
    fn target_success_probability_uses_geometric_attempt_bound() {
        let model = simple_model(0.5);
        assert_eq!(model.attempts_for_target_success_probability(0.95).unwrap(), Some(5));
        assert!(matches!(
            model.attempts_for_target_success_probability(1.0),
            Err(PhotonicEnvelopeError::CertainTargetRequiresInfiniteAttempts)
        ));
        assert_eq!(
            simple_model(1.0).attempts_for_target_success_probability(1.0).unwrap(),
            Some(1)
        );
        assert!(model.attempts_for_target_success_probability(0.0).is_err());
        assert!(model.attempts_for_target_success_probability(1.1).is_err());
    }

    #[test]
    fn zero_yield_is_unknown_cost_not_infinite_or_zero_cost() {
        let result = simple_model(0.0).estimate().unwrap();
        assert_eq!(result.accepted_event_probability, 0.0);
        assert_eq!(result.expected_attempts_per_accepted_output, None);
        assert_eq!(result.expected_seconds_per_accepted_output, None);
        assert_eq!(result.expected_photons_per_accepted_output, None);
    }

    #[test]
    fn rejects_invalid_component_probabilities_and_rates() {
        assert!(ProbabilityInterval::new(0.2, 0.1, 0.5).is_err());
        assert!(ProbabilityInterval::new(0.1, 0.2, 1.1).is_err());
        assert!(OpticalChannel::new(0.1, Some(1.2)).is_err());

        let mut invalid = simple_model(0.3);
        invalid.attempts_per_second = 0.0;
        assert!(invalid.estimate().is_err());
        invalid = simple_model(0.3);
        invalid.photons_per_attempt = 0;
        assert!(invalid.estimate().is_err());
        invalid = simple_model(0.3);
        invalid.conditional_gate_success_probabilities.push(f64::NEG_INFINITY);
        assert!(invalid.estimate().is_err());
    }

    #[test]
    fn probability_intervals_preserve_uncertainty_without_collapsing_to_mean() {
        let interval = ProbabilityInterval::new(0.7, 0.8, 0.9).unwrap();
        assert_eq!(interval.lower, 0.7);
        assert_eq!(interval.nominal, 0.8);
        assert_eq!(interval.upper, 0.9);
    }
}
