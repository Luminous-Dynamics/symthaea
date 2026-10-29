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

/// Interval for an insertion-loss parameter, in dB.
///
/// The fields are ordered as lower_db <= nominal_db <= upper_db. Since larger
/// loss means lower transmission, the corresponding probability interval reverses
/// these endpoints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossIntervalDb {
    /// Lowest plausible insertion loss.
    pub lower_db: f64,
    /// Nominal insertion loss.
    pub nominal_db: f64,
    /// Highest plausible insertion loss.
    pub upper_db: f64,
}

impl LossIntervalDb {
    /// Construct and validate a nonnegative ordered insertion-loss interval.
    pub fn new(
        lower_db: f64,
        nominal_db: f64,
        upper_db: f64,
    ) -> Result<Self, PhotonicEnvelopeError> {
        for loss in [lower_db, nominal_db, upper_db] {
            transmission_probability(loss)?;
        }
        if lower_db > nominal_db || nominal_db > upper_db {
            return Err(PhotonicEnvelopeError::InvalidLossInterval);
        }
        Ok(Self { lower_db, nominal_db, upper_db })
    }

    /// Convert the loss range to transmission probability, reversing the endpoints.
    pub fn transmission_interval(&self) -> Result<ProbabilityInterval, PhotonicEnvelopeError> {
        ProbabilityInterval::new(
            transmission_probability(self.upper_db)?,
            transmission_probability(self.nominal_db)?,
            transmission_probability(self.lower_db)?,
        )
    }
}

/// Optical path whose loss and optional detector efficiency are uncertain intervals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpticalChannelInterval {
    /// Interval for the optical insertion loss.
    pub insertion_loss_db: LossIntervalDb,
    /// Detector efficiency interval; None means no detector is assigned to this path.
    pub detector_efficiency: Option<ProbabilityInterval>,
}

impl OpticalChannelInterval {
    /// Validate path loss and detector interval semantics.
    pub fn validate(&self) -> Result<(), PhotonicEnvelopeError> {
        LossIntervalDb::new(
            self.insertion_loss_db.lower_db,
            self.insertion_loss_db.nominal_db,
            self.insertion_loss_db.upper_db,
        )?;
        if let Some(efficiency) = self.detector_efficiency {
            ProbabilityInterval::new(efficiency.lower, efficiency.nominal, efficiency.upper)?;
        }
        Ok(())
    }
}

/// Component-envelope model that propagates declared parameter intervals.
///
/// Multiplication uses independent component interval bounds and assumes the same
/// one-shot, restart-on-any-failure semantics as PhotonicPrimitiveEnvelope. These
/// are deterministic parameter ranges, not confidence intervals or a noise model.
#[derive(Debug, Clone, PartialEq)]
pub struct PhotonicPrimitiveUncertaintyEnvelope {
    /// Interval for the probability that the source stage supplies a valid input state.
    pub source_success_probability: ProbabilityInterval,
    /// Required paths and their insertion-loss/detection-efficiency intervals.
    pub required_channels: Vec<OpticalChannelInterval>,
    /// Interval for each conditional gate-success probability.
    pub conditional_gate_success_probabilities: Vec<ProbabilityInterval>,
    /// Positive interval for aggregate primitive attempts per second.
    pub attempts_per_second: ProbabilityInterval,
    /// Photons consumed/emitted per whole primitive attempt.
    pub photons_per_attempt: u64,
}

impl PhotonicPrimitiveUncertaintyEnvelope {
    /// Propagate conservative component ranges into output probability and cost ranges.
    pub fn estimate(&self) -> Result<RobustPrimitiveEnvelopeEstimate, PhotonicEnvelopeError> {
        ProbabilityInterval::new(
            self.source_success_probability.lower,
            self.source_success_probability.nominal,
            self.source_success_probability.upper,
        )?;
        for probability in &self.conditional_gate_success_probabilities {
            ProbabilityInterval::new(probability.lower, probability.nominal, probability.upper)?;
        }
        for channel in &self.required_channels {
            channel.validate()?;
        }
        ProbabilityInterval::new(
            self.attempts_per_second.lower,
            self.attempts_per_second.nominal,
            self.attempts_per_second.upper,
        )?;
        if self.attempts_per_second.lower <= 0.0 {
            return Err(PhotonicEnvelopeError::InvalidAttemptRate(self.attempts_per_second.lower));
        }
        if self.photons_per_attempt == 0 {
            return Err(PhotonicEnvelopeError::ZeroPhotonsPerAttempt);
        }

        let unit = ProbabilityInterval { lower: 1.0, nominal: 1.0, upper: 1.0 };
        let mut probability = self.source_success_probability;
        for channel in &self.required_channels {
            probability = multiply_probability_intervals(
                probability,
                channel.insertion_loss_db.transmission_interval()?,
            )?;
            if let Some(detector_efficiency) = channel.detector_efficiency {
                probability = multiply_probability_intervals(probability, detector_efficiency)?;
            }
        }
        for gate_probability in &self.conditional_gate_success_probabilities {
            probability = multiply_probability_intervals(probability, *gate_probability)?;
        }
        // Keep the interval API explicitly bounded even when there are no channels/gates.
        probability = multiply_probability_intervals(probability, unit)?;

        let rate = self.attempts_per_second;
        let expected_attempts = BoundedCostInterval {
            lower: reciprocal_if_positive(probability.upper),
            nominal: reciprocal_if_positive(probability.nominal),
            upper: reciprocal_if_positive(probability.lower),
        };
        let expected_seconds = BoundedCostInterval {
            lower: reciprocal_if_positive(probability.upper * rate.upper),
            nominal: reciprocal_if_positive(probability.nominal * rate.nominal),
            upper: reciprocal_if_positive(probability.lower * rate.lower),
        };
        let photons = self.photons_per_attempt as f64;
        let expected_photons = BoundedCostInterval {
            lower: expected_attempts.lower.map(|value| value * photons),
            nominal: expected_attempts.nominal.map(|value| value * photons),
            upper: expected_attempts.upper.map(|value| value * photons),
        };

        Ok(RobustPrimitiveEnvelopeEstimate {
            accepted_event_probability: probability,
            expected_attempts,
            expected_seconds,
            expected_photons,
        })
    }
}

fn multiply_probability_intervals(
    left: ProbabilityInterval,
    right: ProbabilityInterval,
) -> Result<ProbabilityInterval, PhotonicEnvelopeError> {
    ProbabilityInterval::new(
        left.lower * right.lower,
        left.nominal * right.nominal,
        left.upper * right.upper,
    )
}

fn reciprocal_if_positive(value: f64) -> Option<f64> {
    (value > 0.0).then(|| 1.0 / value)
}

/// Bounded estimate for an expected cost. A missing upper bound means the declared
/// parameter interval allows zero output probability, so finite worst-case cost is
/// not established; it never means zero cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedCostInterval {
    /// Best-case cost implied by the declared bounds, if finite.
    pub lower: Option<f64>,
    /// Nominal cost implied by nominal parameters, if finite.
    pub nominal: Option<f64>,
    /// Worst-case finite cost implied by the declared bounds, if finite.
    pub upper: Option<f64>,
}

/// Robust, interval-propagated output from the component-envelope model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustPrimitiveEnvelopeEstimate {
    /// Accepted-event probability interval.
    pub accepted_event_probability: ProbabilityInterval,
    /// Expected attempts per accepted output.
    pub expected_attempts: BoundedCostInterval,
    /// Expected seconds per accepted output.
    pub expected_seconds: BoundedCostInterval,
    /// Expected photons per accepted output.
    pub expected_photons: BoundedCostInterval,
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
        if let Some(efficiency) = self.detector_efficiency {
            validate_probability(efficiency, "detector efficiency")?;
            Ok(transmission * efficiency)
        } else {
            Ok(transmission)
        }
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

/// Objective for choosing an optical route through a directed connectivity graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpticalRouteObjective {
    /// Minimize hop count, then use deterministic lexicographic path tie-breaking.
    FewestHops,
    /// Minimize the sum of nominal insertion losses in dB.
    MinimumNominalLoss,
    /// Minimize the sum of upper-bound insertion losses in dB.
    MinimumWorstCaseLoss,
}

/// Directed optical interconnect with a declared insertion-loss interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpticalLink {
    /// Directed source node identifier.
    pub from: usize,
    /// Directed destination node identifier.
    pub to: usize,
    /// Insertion-loss interval for this interconnect, in dB.
    pub loss_db: LossIntervalDb,
}

/// Result of routing one photon path under a declared route objective.
#[derive(Debug, Clone, PartialEq)]
pub struct OpticalRoute {
    /// Node sequence, including source and destination.
    pub nodes: Vec<usize>,
    /// Sum of link losses along the selected route.
    pub total_loss_db: LossIntervalDb,
    /// Transmission interval implied by the route loss interval.
    pub transmission_probability: ProbabilityInterval,
}

#[derive(Debug, Clone)]
struct RouteCandidate {
    nodes: Vec<usize>,
    cost: f64,
    loss: LossIntervalDb,
}

fn route_candidate_is_better(candidate: &RouteCandidate, current: &RouteCandidate) -> bool {
    match candidate.cost.total_cmp(&current.cost) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => {
            candidate.nodes.len() < current.nodes.len()
                || (candidate.nodes.len() == current.nodes.len()
                    && candidate.nodes < current.nodes)
        }
    }
}

fn link_cost(link: &OpticalLink, objective: OpticalRouteObjective) -> f64 {
    match objective {
        OpticalRouteObjective::FewestHops => 1.0,
        OpticalRouteObjective::MinimumNominalLoss => link.loss_db.nominal_db,
        OpticalRouteObjective::MinimumWorstCaseLoss => link.loss_db.upper_db,
    }
}

/// Find a deterministic, loop-free route through a directed optical connectivity graph.
///
/// This is a topology-level first-order router. It minimizes hop count, nominal total
/// loss, or worst-case loss according to the chosen objective; it does not optimize
/// interference, crosstalk, wavelength conflicts, detector noise, fusion schedule,
/// code constraints, or multi-commodity congestion.
pub fn find_optical_route(
    node_count: usize,
    links: &[OpticalLink],
    source: usize,
    destination: usize,
    objective: OpticalRouteObjective,
) -> Result<Option<OpticalRoute>, PhotonicEnvelopeError> {
    if node_count == 0 {
        return Err(PhotonicEnvelopeError::InvalidNodeCount);
    }
    if source >= node_count {
        return Err(PhotonicEnvelopeError::OpticalNodeOutOfRange(source));
    }
    if destination >= node_count {
        return Err(PhotonicEnvelopeError::OpticalNodeOutOfRange(destination));
    }
    for link in links {
        if link.from >= node_count {
            return Err(PhotonicEnvelopeError::OpticalNodeOutOfRange(link.from));
        }
        if link.to >= node_count {
            return Err(PhotonicEnvelopeError::OpticalNodeOutOfRange(link.to));
        }
        if link.from == link.to {
            return Err(PhotonicEnvelopeError::SelfLoopOpticalLink(link.from));
        }
        LossIntervalDb::new(
            link.loss_db.lower_db,
            link.loss_db.nominal_db,
            link.loss_db.upper_db,
        )?;
    }

    if source == destination {
        return Ok(Some(OpticalRoute {
            nodes: vec![source],
            total_loss_db: LossIntervalDb { lower_db: 0.0, nominal_db: 0.0, upper_db: 0.0 },
            transmission_probability: ProbabilityInterval { lower: 1.0, nominal: 1.0, upper: 1.0 },
        }));
    }

    let zero_loss = LossIntervalDb { lower_db: 0.0, nominal_db: 0.0, upper_db: 0.0 };
    let mut best: Vec<Option<RouteCandidate>> = vec![None; node_count];
    let mut settled = vec![false; node_count];
    best[source] = Some(RouteCandidate { nodes: vec![source], cost: 0.0, loss: zero_loss });

    for _ in 0..node_count {
        let mut chosen: Option<usize> = None;
        for node in 0..node_count {
            if settled[node] || best[node].is_none() {
                continue;
            }
            if let Some(current) = chosen {
                let candidate = best[node].as_ref().expect("candidate checked above");
                let incumbent = best[current].as_ref().expect("chosen candidate exists");
                if route_candidate_is_better(candidate, incumbent) {
                    chosen = Some(node);
                }
            } else {
                chosen = Some(node);
            }
        }
        let Some(node) = chosen else { break };
        if node == destination {
            break;
        }
        settled[node] = true;
        let current = best[node].as_ref().expect("chosen node has a route").clone();

        for link in links.iter().filter(|link| link.from == node) {
            if settled[link.to] || current.nodes.contains(&link.to) {
                continue;
            }
            let loss = LossIntervalDb::new(
                current.loss.lower_db + link.loss_db.lower_db,
                current.loss.nominal_db + link.loss_db.nominal_db,
                current.loss.upper_db + link.loss_db.upper_db,
            )?;
            let mut nodes = current.nodes.clone();
            nodes.push(link.to);
            let cost = current.cost + link_cost(link, objective);
            if !cost.is_finite() {
                return Err(PhotonicEnvelopeError::InvalidRouteCost);
            }
            let candidate = RouteCandidate { nodes, cost, loss };
            let should_replace = best[link.to]
                .as_ref()
                .is_none_or(|incumbent| route_candidate_is_better(&candidate, incumbent));
            if should_replace {
                best[link.to] = Some(candidate);
            }
        }
    }

    let Some(route) = best[destination].take() else {
        return Ok(None);
    };
    let transmission_probability = ProbabilityInterval::new(
        transmission_probability(route.loss.upper_db)?,
        transmission_probability(route.loss.nominal_db)?,
        transmission_probability(route.loss.lower_db)?,
    )?;
    Ok(Some(OpticalRoute {
        nodes: route.nodes,
        total_loss_db: route.loss,
        transmission_probability,
    }))
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
    /// Loss interval endpoints are not ordered.
    InvalidLossInterval,
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
            Self::InvalidNodeCount =>
                write!(f, "optical topology must have at least one node"),
            Self::OpticalNodeOutOfRange(node) =>
                write!(f, "optical node {node} is out of range"),
            Self::SelfLoopOpticalLink(node) =>
                write!(f, "optical link self-loop at node {node} is not supported"),
            Self::InvalidRouteCost =>
                write!(f, "optical route cost is not finite"),
            Self::InvalidAttemptRate(rate) =>
                write!(f, "attempt rate must be finite and positive, got {rate} Hz"),
            Self::InvalidLossInterval =>
                write!(f, "loss interval must satisfy lower <= nominal <= upper"),
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
    fn route_objectives_choose_shortest_nominal_loss_or_robust_loss_paths() {
        let links = vec![
            OpticalLink {
                from: 0, to: 1,
                loss_db: LossIntervalDb::new(0.1, 0.2, 3.0).unwrap(),
            },
            OpticalLink {
                from: 1, to: 3,
                loss_db: LossIntervalDb::new(0.1, 0.2, 3.0).unwrap(),
            },
            OpticalLink {
                from: 0, to: 2,
                loss_db: LossIntervalDb::new(0.8, 1.0, 1.2).unwrap(),
            },
            OpticalLink {
                from: 2, to: 3,
                loss_db: LossIntervalDb::new(0.8, 1.0, 1.2).unwrap(),
            },
        ];
        let fewest = find_optical_route(
            4, &links, 0, 3, OpticalRouteObjective::FewestHops,
        ).unwrap().unwrap();
        let nominal = find_optical_route(
            4, &links, 0, 3, OpticalRouteObjective::MinimumNominalLoss,
        ).unwrap().unwrap();
        let robust = find_optical_route(
            4, &links, 0, 3, OpticalRouteObjective::MinimumWorstCaseLoss,
        ).unwrap().unwrap();

        assert_eq!(fewest.nodes, vec![0, 1, 3]);
        assert_eq!(nominal.nodes, vec![0, 1, 3]);
        assert_eq!(robust.nodes, vec![0, 2, 3]);
        near(nominal.total_loss_db.nominal_db, 0.4);
        near(nominal.total_loss_db.upper_db, 6.0);
        near(robust.total_loss_db.upper_db, 2.4);
        assert!(robust.transmission_probability.lower > nominal.transmission_probability.lower);
    }

    #[test]
    fn optical_route_handles_identity_disconnection_and_invalid_topology() {
        assert_eq!(
            find_optical_route(2, &[], 0, 1, OpticalRouteObjective::MinimumNominalLoss)
                .unwrap(),
            None
        );
        let identity = find_optical_route(
            2, &[], 1, 1, OpticalRouteObjective::MinimumNominalLoss,
        ).unwrap().unwrap();
        assert_eq!(identity.nodes, vec![1]);
        near(identity.transmission_probability.lower, 1.0);
        assert!(find_optical_route(0, &[], 0, 0, OpticalRouteObjective::FewestHops).is_err());
        assert!(find_optical_route(2, &[], 2, 0, OpticalRouteObjective::FewestHops).is_err());
        let self_loop = [OpticalLink {
            from: 0, to: 0, loss_db: LossIntervalDb::new(0.0, 0.0, 0.0).unwrap()
        }];
        assert!(matches!(
            find_optical_route(1, &self_loop, 0, 0, OpticalRouteObjective::FewestHops),
            Err(PhotonicEnvelopeError::SelfLoopOpticalLink(0))
        ));
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
    fn direct_channel_method_rejects_invalid_public_field_values() {
        let invalid = OpticalChannel {
            insertion_loss_db: 0.0,
            detector_efficiency: Some(1.1),
        };
        assert!(invalid.detected_survival_probability().is_err());
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
    fn loss_interval_reverses_transmission_bounds() {
        let loss = LossIntervalDb::new(0.0, 3.01029995664, 10.0).unwrap();
        let transmission = loss.transmission_interval().unwrap();
        near(transmission.lower, 0.1);
        near(transmission.nominal, 0.5);
        near(transmission.upper, 1.0);
        assert!(LossIntervalDb::new(2.0, 1.0, 3.0).is_err());
    }

    #[test]
    fn uncertain_primitive_propagates_bounds_and_does_not_call_them_confidence_intervals() {
        let model = PhotonicPrimitiveUncertaintyEnvelope {
            source_success_probability: ProbabilityInterval::new(0.5, 0.6, 0.7).unwrap(),
            required_channels: vec![OpticalChannelInterval {
                insertion_loss_db: LossIntervalDb::new(1.0, 2.0, 3.0).unwrap(),
                detector_efficiency: Some(ProbabilityInterval::new(0.8, 0.85, 0.9).unwrap()),
            }],
            conditional_gate_success_probabilities: vec![
                ProbabilityInterval::new(0.4, 0.5, 0.6).unwrap(),
            ],
            attempts_per_second: ProbabilityInterval::new(100.0, 120.0, 150.0).unwrap(),
            photons_per_attempt: 4,
        };
        let result = model.estimate().unwrap();
        assert!(result.accepted_event_probability.lower
            <= result.accepted_event_probability.nominal);
        assert!(result.accepted_event_probability.nominal
            <= result.accepted_event_probability.upper);
        assert!(result.expected_seconds.lower.unwrap()
            <= result.expected_seconds.nominal.unwrap());
        assert!(result.expected_seconds.nominal.unwrap()
            <= result.expected_seconds.upper.unwrap());
        assert!(result.expected_photons.lower.unwrap()
            <= result.expected_photons.nominal.unwrap());
        assert!(result.expected_photons.nominal.unwrap()
            <= result.expected_photons.upper.unwrap());
    }

    #[test]
    fn uncertain_zero_lower_yield_leaves_worst_case_cost_unbounded() {
        let model = PhotonicPrimitiveUncertaintyEnvelope {
            source_success_probability: ProbabilityInterval::new(0.0, 0.5, 0.8).unwrap(),
            required_channels: Vec::new(),
            conditional_gate_success_probabilities: Vec::new(),
            attempts_per_second: ProbabilityInterval::new(100.0, 100.0, 100.0).unwrap(),
            photons_per_attempt: 1,
        };
        let result = model.estimate().unwrap();
        assert!(result.expected_attempts.lower.is_some());
        assert!(result.expected_attempts.nominal.is_some());
        assert_eq!(result.expected_attempts.upper, None);
        assert_eq!(result.expected_seconds.upper, None);
        assert_eq!(result.expected_photons.upper, None);
    }

    #[test]
    fn probability_intervals_preserve_uncertainty_without_collapsing_to_mean() {
        let interval = ProbabilityInterval::new(0.7, 0.8, 0.9).unwrap();
        assert_eq!(interval.lower, 0.7);
        assert_eq!(interval.nominal, 0.8);
        assert_eq!(interval.upper, 0.9);
    }
}
