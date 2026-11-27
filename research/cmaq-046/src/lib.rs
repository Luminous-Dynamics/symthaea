#![forbid(unsafe_code)]

//! Convex-target approachability residuals converted into anytime-valid evidence.
//!
//! The null is an adaptive conditional-mean constraint:
//!
//!     E[r_t | F_{t-1}] ∈ S,
//!
//! where S is a frozen compact convex box. A predictable OCO normal w_t in the
//! unit ball produces q_t = <w_t, r_t> - h_S(w_t). Under the null,
//! E[q_t | F_{t-1}] <= 0. Since |q_t| is bounded, a fixed nonnegative bet
//! converts q_t into an e-process. This is not a PRM and does not require
//! exchangeable ranks, but it only detects conditional-mean departures from S.
//!
//! The OCO regret diagnostic is kept distinct from the e-process guarantee.

const BETTING_RATE: f64 = 0.5;
const NEG_INFINITY: f64 = f64::NEG_INFINITY;
const MAX_DIMENSION: usize = 1_024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApproachabilityError {
    EmptyDimension,
    DimensionTooLarge,
    DimensionMismatch,
    InvalidBounds,
    InvalidObservationNormBound,
    InvalidAlpha,
    NonFiniteValue,
    ObservationNormExceeded,
    CounterOverflow,
    ArithmeticOverflow,
    NegativeRealizedRegret,
    AlreadyQuarantined,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoxTarget {
    lower: Vec<f64>,
    upper: Vec<f64>,
    maximum_corner_norm: f64,
}

impl BoxTarget {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Result<Self, ApproachabilityError> {
        if lower.len() != upper.len() {
            return Err(ApproachabilityError::DimensionMismatch);
        }
        if lower.is_empty() {
            return Err(ApproachabilityError::EmptyDimension);
        }
        if lower.len() > MAX_DIMENSION {
            return Err(ApproachabilityError::DimensionTooLarge);
        }

        let mut maximum_corner_norm = 0.0f64;
        for (lo, hi) in lower.iter().zip(&upper) {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(ApproachabilityError::NonFiniteValue);
            }
            if lo > hi {
                return Err(ApproachabilityError::InvalidBounds);
            }
            maximum_corner_norm = maximum_corner_norm.hypot(lo.abs().max(hi.abs()));
        }
        if !maximum_corner_norm.is_finite() {
            return Err(ApproachabilityError::ArithmeticOverflow);
        }

        Ok(Self {
            lower,
            upper,
            maximum_corner_norm,
        })
    }

    pub fn dimension(&self) -> usize {
        self.lower.len()
    }

    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    pub fn maximum_corner_norm(&self) -> f64 {
        self.maximum_corner_norm
    }

    /// Support function h_S(w) = sup_{s in S} <w,s>.
    pub fn support(&self, normal: &[f64]) -> Result<f64, ApproachabilityError> {
        self.validate_vector(normal)?;
        let mut result = 0.0;
        for ((weight, lower), upper) in normal.iter().zip(&self.lower).zip(&self.upper) {
            let endpoint = if *weight >= 0.0 { *upper } else { *lower };
            result += weight * endpoint;
        }
        if result.is_finite() {
            Ok(result)
        } else {
            Err(ApproachabilityError::ArithmeticOverflow)
        }
    }

    /// One valid support-function subgradient at normal.
    ///
    /// At a zero coordinate any point in [lower, upper] is valid; clamping zero
    /// into that interval is deterministic and keeps the selection bounded.
    pub fn support_subgradient(&self, normal: &[f64]) -> Result<Vec<f64>, ApproachabilityError> {
        self.validate_vector(normal)?;
        Ok(normal
            .iter()
            .zip(&self.lower)
            .zip(&self.upper)
            .map(|((weight, lower), upper)| {
                if *weight > 0.0 {
                    *upper
                } else if *weight < 0.0 {
                    *lower
                } else {
                    0.0f64.clamp(*lower, *upper)
                }
            })
            .collect())
    }

    /// Euclidean distance from a point to the closed box.
    pub fn distance(&self, point: &[f64]) -> Result<f64, ApproachabilityError> {
        self.validate_vector(point)?;
        let mut distance = 0.0f64;
        for ((value, lower), upper) in point.iter().zip(&self.lower).zip(&self.upper) {
            let outside = if value < lower {
                lower - value
            } else if value > upper {
                value - upper
            } else {
                0.0
            };
            distance = distance.hypot(outside);
        }
        if distance.is_finite() {
            Ok(distance)
        } else {
            Err(ApproachabilityError::ArithmeticOverflow)
        }
    }

    fn validate_vector(&self, vector: &[f64]) -> Result<(), ApproachabilityError> {
        if vector.len() != self.dimension() {
            return Err(ApproachabilityError::DimensionMismatch);
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(ApproachabilityError::NonFiniteValue);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApproachabilityUpdate {
    pub observations_seen: u64,
    /// Direction selected before the current observation was revealed.
    pub normal_used: Vec<f64>,
    /// OGD direction to be used on the next observation.
    pub next_normal: Vec<f64>,
    pub support_value: f64,
    pub support_residual: f64,
    pub normalized_residual: f64,
    pub betting_factor: f64,
    pub log_e_value: f64,
    pub log_threshold: f64,
    pub mean_observation: Vec<f64>,
    pub distance_to_target: f64,
    /// Realized OCO regret: T*distance(mean,S) - sum_t q_t.
    pub realized_regret: f64,
    pub quarantined: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvexTargetApproachabilityMonitor {
    target: BoxTarget,
    observation_norm_bound: f64,
    residual_bound: f64,
    alpha_numerator: u32,
    alpha_denominator: u32,
    observations_seen: u64,
    normal: Vec<f64>,
    observation_sum: Vec<f64>,
    residual_sum: f64,
    log_e_value: f64,
    quarantined: bool,
}

impl ConvexTargetApproachabilityMonitor {
    pub fn new(
        target: BoxTarget,
        observation_norm_bound: f64,
        alpha_numerator: u32,
        alpha_denominator: u32,
    ) -> Result<Self, ApproachabilityError> {
        if !observation_norm_bound.is_finite() || observation_norm_bound <= 0.0 {
            return Err(ApproachabilityError::InvalidObservationNormBound);
        }
        if alpha_numerator == 0 || alpha_denominator == 0 || alpha_numerator >= alpha_denominator {
            return Err(ApproachabilityError::InvalidAlpha);
        }
        let residual_bound = observation_norm_bound + target.maximum_corner_norm();
        if !residual_bound.is_finite() || residual_bound <= 0.0 {
            return Err(ApproachabilityError::ArithmeticOverflow);
        }

        Ok(Self {
            normal: vec![0.0; target.dimension()],
            observation_sum: vec![0.0; target.dimension()],
            target,
            observation_norm_bound,
            residual_bound,
            alpha_numerator,
            alpha_denominator,
            observations_seen: 0,
            residual_sum: 0.0,
            log_e_value: 0.0,
            quarantined: false,
        })
    }

    /// Adds one bounded vector observation.
    ///
    /// The current normal is chosen using only past observations. The next
    /// normal is computed only after this observation's e-process update. OGD
    /// minimizes the convex loss h_S(w)-<w,r_t> over the unit L2 ball with
    /// eta_t = 1/(B*sqrt(t)), where B bounds the support residual magnitude.
    ///
    /// State is committed only after every validation and arithmetic operation
    /// succeeds. The fixed bet is lambda=1/2; no ONS betting-regret claim is
    /// imported from the reference paper.
    pub fn observe(
        &mut self,
        observation: &[f64],
    ) -> Result<ApproachabilityUpdate, ApproachabilityError> {
        if self.quarantined {
            return Err(ApproachabilityError::AlreadyQuarantined);
        }
        self.target.validate_vector(observation)?;

        let norm = l2_norm(observation);
        if !norm.is_finite() {
            return Err(ApproachabilityError::ArithmeticOverflow);
        }
        if norm > self.observation_norm_bound + 1e-12 {
            return Err(ApproachabilityError::ObservationNormExceeded);
        }

        let next_seen = self
            .observations_seen
            .checked_add(1)
            .ok_or(ApproachabilityError::CounterOverflow)?;
        let normal_used = self.normal.clone();
        let support_value = self.target.support(&normal_used)?;
        let inner_product = dot(&normal_used, observation)?;
        let support_residual = inner_product - support_value;
        let normalized_residual = support_residual / self.residual_bound;
        if !support_residual.is_finite()
            || !normalized_residual.is_finite()
            || !(-1.0 - 1e-10..=1.0 + 1e-10).contains(&normalized_residual)
        {
            return Err(ApproachabilityError::ArithmeticOverflow);
        }

        let betting_factor = 1.0 + BETTING_RATE * normalized_residual;
        if !betting_factor.is_finite() || betting_factor < 0.5 - 1e-12 {
            return Err(ApproachabilityError::ArithmeticOverflow);
        }
        let next_log_e_value = self.log_e_value + betting_factor.ln();
        if !next_log_e_value.is_finite() {
            return Err(ApproachabilityError::ArithmeticOverflow);
        }

        let mut next_observation_sum = self.observation_sum.clone();
        for (sum, value) in next_observation_sum.iter_mut().zip(observation) {
            *sum += value;
            if !sum.is_finite() {
                return Err(ApproachabilityError::ArithmeticOverflow);
            }
        }
        let next_residual_sum = self.residual_sum + support_residual;
        if !next_residual_sum.is_finite() {
            return Err(ApproachabilityError::ArithmeticOverflow);
        }

        let next_mean = next_observation_sum
            .iter()
            .map(|sum| *sum / next_seen as f64)
            .collect::<Vec<_>>();
        if next_mean.iter().any(|value| !value.is_finite()) {
            return Err(ApproachabilityError::ArithmeticOverflow);
        }
        let distance_to_target = self.target.distance(&next_mean)?;
        let horizon_distance = distance_to_target * next_seen as f64;
        let raw_regret = horizon_distance - next_residual_sum;
        let regret_tolerance = 1e-9 * horizon_distance.abs().max(next_residual_sum.abs()).max(1.0);
        if !raw_regret.is_finite() {
            return Err(ApproachabilityError::ArithmeticOverflow);
        }
        if raw_regret < -regret_tolerance {
            return Err(ApproachabilityError::NegativeRealizedRegret);
        }
        let realized_regret = raw_regret.max(0.0);

        let subgradient = self.target.support_subgradient(&normal_used)?;
        let gradient_step = 1.0 / (self.residual_bound * (next_seen as f64).sqrt());
        let mut next_normal = Vec::with_capacity(normal_used.len());
        for ((weight, observation_value), support_gradient) in
            normal_used.iter().zip(observation).zip(subgradient.iter())
        {
            next_normal.push(weight + gradient_step * (observation_value - support_gradient));
        }
        project_unit_ball(&mut next_normal)?;
        let log_threshold =
            (f64::from(self.alpha_denominator) / f64::from(self.alpha_numerator)).ln();
        if !log_threshold.is_finite() {
            return Err(ApproachabilityError::ArithmeticOverflow);
        }
        let quarantined = next_log_e_value >= log_threshold;

        // Atomic state commit.
        self.observations_seen = next_seen;
        self.normal = next_normal.clone();
        self.observation_sum = next_observation_sum;
        self.residual_sum = next_residual_sum;
        self.log_e_value = next_log_e_value;
        if quarantined {
            self.quarantined = true;
        }

        Ok(ApproachabilityUpdate {
            observations_seen: next_seen,
            normal_used,
            next_normal,
            support_value,
            support_residual,
            normalized_residual,
            betting_factor,
            log_e_value: next_log_e_value,
            log_threshold,
            mean_observation: next_mean,
            distance_to_target,
            realized_regret,
            quarantined,
        })
    }

    pub fn target(&self) -> &BoxTarget {
        &self.target
    }

    pub fn observations_seen(&self) -> u64 {
        self.observations_seen
    }

    pub fn normal_for_next_observation(&self) -> &[f64] {
        &self.normal
    }

    pub fn mean_observation(&self) -> Result<Vec<f64>, ApproachabilityError> {
        if self.observations_seen == 0 {
            return Ok(vec![0.0; self.target.dimension()]);
        }
        Ok(self
            .observation_sum
            .iter()
            .map(|sum| *sum / self.observations_seen as f64)
            .collect())
    }

    pub fn distance_to_target(&self) -> Result<f64, ApproachabilityError> {
        self.target.distance(&self.mean_observation()?)
    }

    pub fn cumulative_support_residual(&self) -> f64 {
        self.residual_sum
    }

    pub fn realized_regret(&self) -> Result<f64, ApproachabilityError> {
        if self.observations_seen == 0 {
            return Ok(0.0);
        }
        let distance = self.distance_to_target()?;
        let raw = distance * self.observations_seen as f64 - self.residual_sum;
        if !raw.is_finite() {
            return Err(ApproachabilityError::ArithmeticOverflow);
        }
        if raw < -1e-9 * raw.abs().max(1.0) {
            return Err(ApproachabilityError::NegativeRealizedRegret);
        }
        Ok(raw.max(0.0))
    }

    pub fn log_e_value(&self) -> f64 {
        self.log_e_value
    }

    pub fn e_value(&self) -> f64 {
        self.log_e_value.exp()
    }

    pub fn residual_bound(&self) -> f64 {
        self.residual_bound
    }

    pub fn threshold_log_wealth(&self) -> f64 {
        (f64::from(self.alpha_denominator) / f64::from(self.alpha_numerator)).ln()
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantined
    }

    pub fn observation_norm_bound(&self) -> f64 {
        self.observation_norm_bound
    }

    pub fn alpha(&self) -> (u32, u32) {
        (self.alpha_numerator, self.alpha_denominator)
    }
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, ApproachabilityError> {
    if left.len() != right.len() {
        return Err(ApproachabilityError::DimensionMismatch);
    }
    let mut value = 0.0;
    for (a, b) in left.iter().zip(right) {
        value += a * b;
    }
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ApproachabilityError::ArithmeticOverflow)
    }
}

fn l2_norm(vector: &[f64]) -> f64 {
    vector.iter().fold(0.0f64, |norm, value| norm.hypot(*value))
}

fn project_unit_ball(vector: &mut [f64]) -> Result<(), ApproachabilityError> {
    let norm = l2_norm(vector);
    if !norm.is_finite() {
        return Err(ApproachabilityError::ArithmeticOverflow);
    }
    if norm > 1.0 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(ApproachabilityError::ArithmeticOverflow);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> BoxTarget {
        BoxTarget::new(vec![-0.1, -0.2], vec![0.1, 0.2]).unwrap()
    }

    fn monitor() -> ConvexTargetApproachabilityMonitor {
        ConvexTargetApproachabilityMonitor::new(target(), 1.0, 1, 20).unwrap()
    }

    #[test]
    fn box_support_and_distance_are_exact_for_axis_aligned_examples() {
        let target = target();
        assert!((target.support(&[1.0, -1.0]).unwrap() - 0.3).abs() < 1e-12);
        assert!((target.distance(&[0.4, -0.5]).unwrap() - 0.3f64 * 2.0f64.sqrt()).abs() < 1e-12);
        assert_eq!(target.distance(&[0.0, 0.0]).unwrap(), 0.0);
    }

    #[test]
    fn support_subgradient_is_valid_including_zero_coordinates() {
        let target = target();
        for normal in [[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [0.0, -1.0]] {
            let subgradient = target.support_subgradient(&normal).unwrap();
            let value = dot(&normal, &subgradient).unwrap();
            assert!((value - target.support(&normal).unwrap()).abs() < 1e-12);
        }
    }

    #[test]
    fn conditional_mean_in_target_has_nonpositive_expected_support_residual() {
        let target = target();
        let mean = [0.04, -0.1];
        let directions = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [-0.8, 0.6], [-1.0, 0.0]];
        for direction in directions {
            let residual = dot(&direction, &mean).unwrap() - target.support(&direction).unwrap();
            assert!(residual <= 1e-12);
        }
    }

    #[test]
    fn initial_wealth_is_one_and_normal_is_predictable() {
        let mut monitor = monitor();
        assert_eq!(monitor.e_value(), 1.0);
        assert_eq!(monitor.normal_for_next_observation(), &[0.0, 0.0]);

        let first = monitor.observe(&[0.4, 0.0]).unwrap();
        assert_eq!(first.normal_used, vec![0.0, 0.0]);
        assert!(first.next_normal.iter().any(|value| value.abs() > 0.0));
        assert_eq!(monitor.normal_for_next_observation(), first.next_normal);

        let second = monitor.observe(&[0.4, 0.0]).unwrap();
        assert_eq!(second.normal_used, first.next_normal);
    }

    #[test]
    fn pathwise_distance_residual_regret_identity_holds() {
        let mut monitor = monitor();
        let observations = [
            [0.5, 0.0],
            [0.3, -0.1],
            [0.4, 0.2],
            [-0.2, 0.1],
            [0.6, -0.2],
        ];
        let mut last = None;
        for observation in observations {
            last = Some(monitor.observe(&observation).unwrap());
        }
        let report = last.unwrap();
        let left = report.distance_to_target;
        let lhs = left;
        let rhs = (monitor.cumulative_support_residual() + monitor.realized_regret().unwrap())
            / monitor.observations_seen() as f64;
        assert!((lhs - rhs).abs() < 1e-10);
    }

    #[test]
    fn normalized_residual_and_betting_factor_are_bounded() {
        let mut monitor = monitor();
        for observation in [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]] {
            let update = monitor.observe(&observation).unwrap();
            assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&update.normalized_residual));
            assert!(update.betting_factor >= 0.5 - 1e-12);
        }
    }

    #[test]
    fn invalid_observation_does_not_mutate_state() {
        let mut monitor = monitor();
        monitor.observe(&[0.2, 0.3]).unwrap();
        let before = monitor.clone();
        assert_eq!(
            monitor.observe(&[0.2]),
            Err(ApproachabilityError::DimensionMismatch)
        );
        assert_eq!(
            monitor.observe(&[f64::NAN, 0.0]),
            Err(ApproachabilityError::NonFiniteValue)
        );
        assert_eq!(
            monitor.observe(&[1.0, 1.0]),
            Err(ApproachabilityError::ObservationNormExceeded)
        );
        assert_eq!(monitor, before);
    }

    #[test]
    fn separated_mean_sequence_accumulates_evidence_and_quarantines() {
        let box_target = BoxTarget::new(vec![-0.05, -0.05], vec![0.05, 0.05]).unwrap();
        let mut monitor = ConvexTargetApproachabilityMonitor::new(box_target, 1.0, 1, 20).unwrap();
        let mut quarantined_at = None;
        for index in 0..400 {
            let noise = if index % 2 == 0 { 0.1 } else { -0.1 };
            let update = monitor.observe(&[0.5 + noise, 0.1 + noise]).unwrap();
            if update.quarantined {
                quarantined_at = Some(index + 1);
                break;
            }
        }
        assert!(quarantined_at.is_some());
        assert!(monitor.is_quarantined());
        assert!(monitor.log_e_value() >= monitor.threshold_log_wealth());
    }

    #[test]
    fn conditional_null_mean_inside_box_produces_no_positive_residual_from_zero_normal() {
        let mut monitor = monitor();
        // With w_1=0, q_1=0. The following observations are centered around
        // an in-target mean; the test checks path diagnostics, not a random FPR.
        let first = monitor.observe(&[0.0, 0.0]).unwrap();
        assert_eq!(first.support_residual, 0.0);
        for value in [0.05, -0.05, 0.05, -0.05] {
            let _ = monitor.observe(&[value, value]).unwrap();
        }
        assert!(monitor.log_e_value().is_finite());
    }

    #[test]
    fn quarantine_is_permanent() {
        let mut monitor = monitor();
        for index in 0..500 {
            let update = monitor.observe(&[0.8, 0.0]).unwrap();
            if update.quarantined {
                break;
            }
            if index == 499 {
                panic!("separated sequence did not quarantine");
            }
        }
        assert_eq!(
            monitor.observe(&[0.0, 0.0]),
            Err(ApproachabilityError::AlreadyQuarantined)
        );
    }

    #[test]
    fn observation_counter_overflow_fails_before_mutation() {
        let mut monitor = monitor();
        monitor.observations_seen = u64::MAX;
        let before = monitor.clone();
        assert_eq!(
            monitor.observe(&[0.0, 0.0]),
            Err(ApproachabilityError::CounterOverflow)
        );
        assert_eq!(monitor, before);
    }

    #[test]
    fn invalid_target_alpha_and_bounds_are_rejected() {
        assert_eq!(
            BoxTarget::new(vec![0.0], vec![]).err(),
            Some(ApproachabilityError::DimensionMismatch)
        );
        assert_eq!(
            BoxTarget::new(vec![1.0], vec![0.0]).err(),
            Some(ApproachabilityError::InvalidBounds)
        );
        assert_eq!(
            ConvexTargetApproachabilityMonitor::new(target(), 0.0, 1, 20).err(),
            Some(ApproachabilityError::InvalidObservationNormBound)
        );
        assert_eq!(
            ConvexTargetApproachabilityMonitor::new(target(), 1.0, 1, 1).err(),
            Some(ApproachabilityError::InvalidAlpha)
        );
    }
}
