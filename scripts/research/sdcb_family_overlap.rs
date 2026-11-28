//! Exact nuisance-envelope overlap witness for the OU / first-order sensor model.
//!
//! This is a dependency-free algebraic reference, not a general structural
//! identifiability solver. It proves a sufficient exact-equivalence condition:
//! swapping the physical and sensor poles while preserving q produces the
//! same complete stationary covariance kernel for every nonnegative lag.
//!
//! Build and run:
//!   rustc --edition=2024 --test scripts/research/sdcb_family_overlap.rs -o /tmp/sdcb-family-overlap-tests
//!   /tmp/sdcb-family-overlap-tests
//!   rustc --edition=2024 scripts/research/sdcb_family_overlap.rs -o /tmp/sdcb-family-overlap
//!   /tmp/sdcb-family-overlap
//!
//! All rational parameter checks use checked i128 arithmetic. No floating-point
//! tolerance is used to decide whether the algebraic overlap witness is exact.

use std::cmp::Ordering;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticError {
    NonPositiveRational,
    ReversedInterval,
    Overflow,
}

impl fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveRational => write!(f, "rational numerator and denominator must be positive"),
            Self::ReversedInterval => write!(f, "interval lower bound exceeds upper bound"),
            Self::Overflow => write!(f, "checked rational arithmetic overflowed"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    numerator: i128,
    denominator: i128,
}

impl Rational {
    pub fn new(numerator: i128, denominator: i128) -> Result<Self, ArithmeticError> {
        if numerator <= 0 || denominator <= 0 {
            return Err(ArithmeticError::NonPositiveRational);
        }
        let divisor = gcd(numerator as u128, denominator as u128) as i128;
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub fn checked_add(self, other: Self) -> Result<Self, ArithmeticError> {
        let left = self
            .numerator
            .checked_mul(other.denominator)
            .ok_or(ArithmeticError::Overflow)?;
        let right = other
            .numerator
            .checked_mul(self.denominator)
            .ok_or(ArithmeticError::Overflow)?;
        let numerator = left.checked_add(right).ok_or(ArithmeticError::Overflow)?;
        let denominator = self
            .denominator
            .checked_mul(other.denominator)
            .ok_or(ArithmeticError::Overflow)?;
        Self::new(numerator, denominator)
    }

    pub fn checked_mul(self, other: Self) -> Result<Self, ArithmeticError> {
        let numerator = self
            .numerator
            .checked_mul(other.numerator)
            .ok_or(ArithmeticError::Overflow)?;
        let denominator = self
            .denominator
            .checked_mul(other.denominator)
            .ok_or(ArithmeticError::Overflow)?;
        Self::new(numerator, denominator)
    }

    pub fn checked_div(self, other: Self) -> Result<Self, ArithmeticError> {
        if other.numerator <= 0 {
            return Err(ArithmeticError::NonPositiveRational);
        }
        let numerator = self
            .numerator
            .checked_mul(other.denominator)
            .ok_or(ArithmeticError::Overflow)?;
        let denominator = self
            .denominator
            .checked_mul(other.numerator)
            .ok_or(ArithmeticError::Overflow)?;
        Self::new(numerator, denominator)
    }

    pub fn checked_cmp(self, other: Self) -> Result<Ordering, ArithmeticError> {
        let left = self
            .numerator
            .checked_mul(other.denominator)
            .ok_or(ArithmeticError::Overflow)?;
        let right = other
            .numerator
            .checked_mul(self.denominator)
            .ok_or(ArithmeticError::Overflow)?;
        Ok(left.cmp(&right))
    }

    pub fn contains(self, lower: Self, upper: Self) -> Result<bool, ArithmeticError> {
        Ok(lower.checked_cmp(self)? != Ordering::Greater
            && self.checked_cmp(upper)? != Ordering::Greater)
    }

    pub fn to_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OuSensorWorld {
    pub physical_decay: Rational,
    pub sensor_rate: Rational,
    pub process_stationary_variance: Rational,
}

impl OuSensorWorld {
    pub fn output_zero_lag_variance(self) -> Result<Rational, ArithmeticError> {
        let numerator = self
            .sensor_rate
            .checked_mul(self.process_stationary_variance)?;
        let denominator = self.physical_decay.checked_add(self.sensor_rate)?;
        numerator.checked_div(denominator)
    }
}

/// Construct process variance that holds the observed zero-lag variance q fixed.
pub fn variance_preserving_q(
    q: Rational,
    physical_decay: Rational,
    sensor_rate: Rational,
) -> Result<Rational, ArithmeticError> {
    q.checked_mul(physical_decay.checked_add(sensor_rate)?)?
        .checked_div(sensor_rate)
}

/// Sufficient exact witness for the OU + first-order low-pass pole-swap symmetry.
///
/// The output covariance kernel is symmetric in its two rates once q is fixed:
/// C(h)=q exp(-min(lambda,mu)h)
///      * [1 + min(lambda,mu) h * psi(abs(lambda-mu)h)].
/// The minimum and absolute gap are invariant under swapping lambda and mu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactOverlapWitness {
    pub world_a: OuSensorWorld,
    pub world_b: OuSensorWorld,
    pub shared_q: Rational,
}

pub fn exact_pole_swap_witness(
    world_a: OuSensorWorld,
    world_b: OuSensorWorld,
) -> Result<Option<ExactOverlapWitness>, ArithmeticError> {
    if world_a.physical_decay != world_b.sensor_rate
        || world_a.sensor_rate != world_b.physical_decay
    {
        return Ok(None);
    }

    let q_a = world_a.output_zero_lag_variance()?;
    let q_b = world_b.output_zero_lag_variance()?;
    if q_a != q_b {
        return Ok(None);
    }

    Ok(Some(ExactOverlapWitness {
        world_a,
        world_b,
        shared_q: q_a,
    }))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositiveInterval {
    pub lower: Rational,
    pub upper: Rational,
}

impl PositiveInterval {
    pub fn new(lower: Rational, upper: Rational) -> Result<Self, ArithmeticError> {
        if lower.checked_cmp(upper)? == Ordering::Greater {
            return Err(ArithmeticError::ReversedInterval);
        }
        Ok(Self { lower, upper })
    }

    pub fn contains(self, value: Rational) -> Result<bool, ArithmeticError> {
        value.contains(self.lower, self.upper)
    }
}

pub fn sensor_inlet_unit_step(sensor_rate: f64, elapsed: f64) -> Option<f64> {
    if !sensor_rate.is_finite()
        || sensor_rate <= 0.0
        || !elapsed.is_finite()
        || elapsed < 0.0
    {
        return None;
    }
    let response = -(-sensor_rate * elapsed).exp_m1();
    response.is_finite().then_some(response)
}

fn rational(numerator: i128, denominator: i128) -> Rational {
    Rational::new(numerator, denominator).expect("frozen positive rational")
}

fn world_a() -> OuSensorWorld {
    OuSensorWorld {
        physical_decay: rational(1, 4),
        sensor_rate: rational(4, 1),
        process_stationary_variance: rational(9, 1),
    }
}

fn exact_world_b() -> OuSensorWorld {
    OuSensorWorld {
        physical_decay: rational(4, 1),
        sensor_rate: rational(1, 4),
        process_stationary_variance: rational(144, 1),
    }
}

fn near_world_b() -> OuSensorWorld {
    let q = world_a().output_zero_lag_variance().expect("valid q");
    let lambda = rational(101, 25); // 4.04
    let mu = rational(1, 4);
    let variance = variance_preserving_q(q, lambda, mu).expect("valid variance");
    OuSensorWorld {
        physical_decay: lambda,
        sensor_rate: mu,
        process_stationary_variance: variance,
    }
}

fn main() {
    let a = world_a();
    let q = a.output_zero_lag_variance().expect("valid q");
    let envelope = PositiveInterval::new(rational(39_996, 10_000), rational(40_804, 10_000))
        .expect("valid envelope");
    let witness = exact_pole_swap_witness(a, exact_world_b())
        .expect("exact arithmetic")
        .expect("frozen exact pole-swap witness");
    let nominal_is_exact = exact_pole_swap_witness(a, near_world_b())
        .expect("exact arithmetic")
        .is_some();

    println!("schema=sdcb-family-overlap-reference-v1");
    println!("status=FAMILY_INTERSECTION");
    println!("world_a_q={q}");
    println!("witness_world_b_lambda={}", witness.world_b.physical_decay);
    println!("witness_world_b_mu={}", witness.world_b.sensor_rate);
    println!(
        "witness_world_b_process_variance={}",
        witness.world_b.process_stationary_variance
    );
    println!(
        "uncertainty_envelope_contains_alias={}",
        envelope
            .contains(witness.world_b.physical_decay)
            .expect("checked interval membership")
    );
    println!("nominal_near_pair_is_exact_alias={nominal_is_exact}");
    println!("uniform_passive_power_ceiling=alpha");
    println!("claim_scope=synthetic_ou_first_order_sensor_model_only");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_pole_swap_has_equal_q_and_full_rate_swap() {
        let witness = exact_pole_swap_witness(world_a(), exact_world_b())
            .unwrap()
            .unwrap();
        assert_eq!(witness.shared_q, rational(144, 17));
        assert_eq!(witness.world_a.physical_decay, witness.world_b.sensor_rate);
        assert_eq!(witness.world_a.sensor_rate, witness.world_b.physical_decay);
    }

    #[test]
    fn nominal_near_swap_is_not_an_exact_alias() {
        assert!(exact_pole_swap_witness(world_a(), near_world_b())
            .unwrap()
            .is_none());
    }

    #[test]
    fn one_percent_parameter_envelope_contains_exact_alias() {
        let envelope =
            PositiveInterval::new(rational(39_996, 10_000), rational(40_804, 10_000))
                .unwrap();
        assert!(envelope.contains(rational(4, 1)).unwrap());
    }

    #[test]
    fn half_percent_comparator_envelope_excludes_exact_alias() {
        let envelope =
            PositiveInterval::new(rational(40_198, 10_000), rational(40_602, 10_000))
                .unwrap();
        assert!(!envelope.contains(rational(4, 1)).unwrap());
    }

    #[test]
    fn q_preserving_variance_at_swapped_pole_is_exactly_144() {
        let q = world_a().output_zero_lag_variance().unwrap();
        let variance = variance_preserving_q(q, rational(4, 1), rational(1, 4)).unwrap();
        assert_eq!(variance, rational(144, 1));
    }

    #[test]
    fn q_preserving_variance_reproduces_near_swap_fixture() {
        let q = world_a().output_zero_lag_variance().unwrap();
        let variance =
            variance_preserving_q(q, rational(101, 25), rational(1, 4)).unwrap();
        assert_eq!(variance, rational(61_776, 425));
        assert!((variance.to_f64() - 145.35529411764706).abs() < 1e-12);
    }

    #[test]
    fn inlet_step_distinguishes_the_sensor_rates() {
        let a = sensor_inlet_unit_step(4.0, 0.5).unwrap();
        let b = sensor_inlet_unit_step(0.25, 0.5).unwrap();
        assert!((a - 0.8646647167633873).abs() < 2e-15);
        assert!((b - 0.1175030974154046).abs() < 2e-15);
        assert!(a - b > 0.7);
    }

    #[test]
    fn invalid_step_inputs_fail_closed() {
        assert!(sensor_inlet_unit_step(0.0, 0.5).is_none());
        assert!(sensor_inlet_unit_step(4.0, f64::NAN).is_none());
        assert!(sensor_inlet_unit_step(4.0, -0.1).is_none());
    }

    #[test]
    fn rational_interval_rejects_reversed_bounds() {
        assert_eq!(
            PositiveInterval::new(rational(5, 1), rational(4, 1)).unwrap_err(),
            ArithmeticError::ReversedInterval
        );
    }
}
