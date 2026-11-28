//! Exact finite-alphabet TV-ball robust hypothesis-testing fixture.
//!
//! Nominal alternative: Bernoulli(4/5), TV radius 3/20.
//! Nominal null:       Bernoulli(1/5), TV radius 1/20.
//! Alpha: 1/10.
//!
//! For Bernoulli laws, TV distance is |p-p0|, so the sets are
//! A: p in [13/20, 19/20], B: q in [3/20, 1/4].
//!
//! The exact robust randomized test is phi(0)=0, phi(1)=2/5.
//! Its worst-null size and worst-alternative power are 1/10 and 13/50.
//! A matching dual certificate uses A*=(7/20,13/20), lambda=1,
//! B*=(3/4,1/4), w=13/5, and z=(0,0).
//!
//! This is a finite two-outcome known-answer verifier, not a generic
//! TV-polytope vertex enumerator or continuous-density robust detector.

use std::cmp::Ordering;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Q {
    n: i128,
    d: i128,
}

impl Q {
    fn new(n: i128, d: i128) -> Self {
        assert_ne!(d, 0);
        let (n, d) = if d < 0 { (-n, -d) } else { (n, d) };
        let g = gcd(n.abs(), d);
        Self { n: n / g, d: d / g }
    }

    fn zero() -> Self { Self::new(0, 1) }
    fn one() -> Self { Self::new(1, 1) }

    fn add(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.d + rhs.n * self.d, self.d * rhs.d)
    }

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.d - rhs.n * self.d, self.d * rhs.d)
    }

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.n, self.d * rhs.d)
    }

    fn div(self, rhs: Self) -> Self {
        assert_ne!(rhs.n, 0);
        Self::new(self.n * rhs.d, self.d * rhs.n)
    }

    fn cmp(self, rhs: Self) -> Ordering {
        (self.n * rhs.d).cmp(&(rhs.n * self.d))
    }

    fn min(self, rhs: Self) -> Self {
        if self.cmp(rhs) == Ordering::Greater { rhs } else { self }
    }

    fn max(self, rhs: Self) -> Self {
        if self.cmp(rhs) == Ordering::Less { rhs } else { self }
    }

    fn is_le(self, rhs: Self) -> bool { self.cmp(rhs) != Ordering::Greater }
}

impl fmt::Display for Q {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.d == 1 { write!(f, "{}", self.n) }
        else { write!(f, "{}/{}", self.n, self.d) }
    }
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}

const ALPHA: Q = Q { n: 1, d: 10 };
const PHI_ZERO: Q = Q { n: 0, d: 1 };
const PHI_ONE: Q = Q { n: 2, d: 5 };

const ALT_NOMINAL: Q = Q { n: 4, d: 5 };
const ALT_RADIUS: Q = Q { n: 3, d: 20 };
const NULL_NOMINAL: Q = Q { n: 1, d: 5 };
const NULL_RADIUS: Q = Q { n: 1, d: 20 };

const ALT_LOW: Q = Q { n: 13, d: 20 };
const ALT_HIGH: Q = Q { n: 19, d: 20 };
const NULL_LOW: Q = Q { n: 3, d: 20 };
const NULL_HIGH: Q = Q { n: 1, d: 4 };

fn bernoulli_law(p: Q) -> [Q; 2] {
    [Q::one().sub(p), p]
}

fn total_variation(left: [Q; 2], right: [Q; 2]) -> Q {
    let d0 = left[0].sub(right[0]);
    let d1 = left[1].sub(right[1]);
    let abs0 = if d0.n < 0 { Q::zero().sub(d0) } else { d0 };
    let abs1 = if d1.n < 0 { Q::zero().sub(d1) } else { d1 };
    abs0.add(abs1).div(Q::new(2, 1))
}

fn rejection_probability(p: Q) -> Q {
    let law = bernoulli_law(p);
    law[0].mul(PHI_ZERO).add(law[1].mul(PHI_ONE))
}

fn worst_null_size() -> Q {
    rejection_probability(NULL_HIGH)
}

fn worst_alternative_power() -> Q {
    rejection_probability(ALT_LOW)
}

fn dual_objective() -> Q {
    ALPHA.mul(Q::new(13, 5))
}

fn dual_positive_part() -> [Q; 2] {
    let alternative = bernoulli_law(ALT_LOW);
    let null = bernoulli_law(NULL_HIGH);
    let w = Q::new(13, 5);
    let d0 = alternative[0].sub(w.mul(null[0]));
    let d1 = alternative[1].sub(w.mul(null[1]));
    [
        if d0.n > 0 { d0 } else { Q::zero() },
        if d1.n > 0 { d1 } else { Q::zero() },
    ]
}

fn glrt_ratio(outcome_one: bool) -> Q {
    let alt0 = bernoulli_law(ALT_LOW);
    let alt1 = bernoulli_law(ALT_HIGH);
    let null0 = bernoulli_law(NULL_LOW);
    let null1 = bernoulli_law(NULL_HIGH);
    let alt_max = if outcome_one { alt0[1].max(alt1[1]) } else { alt0[0].max(alt1[0]) };
    let null_max = if outcome_one { null0[1].max(null1[1]) } else { null0[0].max(null1[0]) };
    alt_max.div(null_max)
}

/// Worst-null tail probability for the GLRT ordering on the two outcomes.
fn p_sup(outcome_one: bool) -> Q {
    let score_zero = glrt_ratio(false);
    let score_one = glrt_ratio(true);
    if outcome_one {
        // score_one is the unique largest statistic value; tail is {Y=1}.
        NULL_LOW.max(NULL_HIGH)
    } else if score_zero.cmp(score_one) != Ordering::Greater {
        Q::one()
    } else {
        Q::one()
    }
}

fn run_fixture() {
    let alt_low_tv = total_variation(bernoulli_law(ALT_LOW), bernoulli_law(ALT_NOMINAL));
    let alt_high_tv = total_variation(bernoulli_law(ALT_HIGH), bernoulli_law(ALT_NOMINAL));
    let null_low_tv = total_variation(bernoulli_law(NULL_LOW), bernoulli_law(NULL_NOMINAL));
    let null_high_tv = total_variation(bernoulli_law(NULL_HIGH), bernoulli_law(NULL_NOMINAL));

    assert_eq!(alt_low_tv, ALT_RADIUS);
    assert_eq!(alt_high_tv, ALT_RADIUS);
    assert_eq!(null_low_tv, NULL_RADIUS);
    assert_eq!(null_high_tv, NULL_RADIUS);
    assert_eq!(worst_null_size(), ALPHA);
    assert_eq!(worst_alternative_power(), Q::new(13, 50));
    assert_eq!(dual_positive_part(), [Q::zero(), Q::zero()]);
    assert_eq!(dual_objective(), Q::new(13, 50));
    assert_eq!(worst_alternative_power(), dual_objective());
    assert!(p_sup(true).cmp(ALPHA) == Ordering::Greater);
    assert_eq!(p_sup(true), Q::new(1, 4));

    println!("status=EXACT_ASYMMETRIC_TV_BALL_FIXTURE");
    println!("alpha={ALPHA}");
    println!("alternative_tv_radius={ALT_RADIUS}");
    println!("null_tv_radius={NULL_RADIUS}");
    println!("alternative_interval=[{ALT_LOW},{ALT_HIGH}]");
    println!("null_interval=[{NULL_LOW},{NULL_HIGH}]");
    println!("worst_null_size={}", worst_null_size());
    println!("worst_alternative_power={}", worst_alternative_power());
    println!("dual_weight_null={}", Q::new(13, 5));
    println!("dual_positive_part={:?}", dual_positive_part());
    println!("dual_objective={}", dual_objective());
    println!("p_sup_y1={}", p_sup(true));
    println!("p_sup_y0={}", p_sup(false));
}

fn main() { run_fixture(); }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernoulli_tv_is_absolute_parameter_difference() {
        assert_eq!(total_variation(bernoulli_law(ALT_LOW), bernoulli_law(ALT_NOMINAL)), ALT_RADIUS);
        assert_eq!(total_variation(bernoulli_law(ALT_HIGH), bernoulli_law(ALT_NOMINAL)), ALT_RADIUS);
        assert_eq!(total_variation(bernoulli_law(NULL_LOW), bernoulli_law(NULL_NOMINAL)), NULL_RADIUS);
        assert_eq!(total_variation(bernoulli_law(NULL_HIGH), bernoulli_law(NULL_NOMINAL)), NULL_RADIUS);
    }

    #[test]
    fn robust_test_controls_size_at_both_null_endpoints() {
        assert_eq!(rejection_probability(NULL_LOW), Q::new(3, 50));
        assert_eq!(rejection_probability(NULL_HIGH), ALPHA);
        assert!(rejection_probability(NULL_LOW).is_le(ALPHA));
    }

    #[test]
    fn robust_test_has_worst_case_power_at_lower_alternative_endpoint() {
        assert_eq!(rejection_probability(ALT_LOW), Q::new(13, 50));
        assert_eq!(rejection_probability(ALT_HIGH), Q::new(19, 50));
        assert!(rejection_probability(ALT_LOW).is_le(rejection_probability(ALT_HIGH)));
    }

    #[test]
    fn primal_and_dual_certificates_match_exactly() {
        assert_eq!(worst_null_size(), ALPHA);
        assert_eq!(worst_alternative_power(), Q::new(13, 50));
        assert_eq!(dual_positive_part(), [Q::zero(), Q::zero()]);
        assert_eq!(dual_objective(), Q::new(13, 50));
        assert_eq!(worst_alternative_power(), dual_objective());
    }

    #[test]
    fn least_favourable_dual_mixture_is_the_declared_boundary_pair() {
        let a_star = bernoulli_law(ALT_LOW);
        let b_star = bernoulli_law(NULL_HIGH);
        let w = Q::new(13, 5);
        assert!(a_star[0].is_le(w.mul(b_star[0])));
        assert_eq!(a_star[1], w.mul(b_star[1]));
    }

    #[test]
    fn p_sup_glrt_is_valid_but_cannot_reject_at_alpha() {
        assert_eq!(glrt_ratio(true), Q::new(19, 5));
        assert_eq!(glrt_ratio(false), Q::new(7, 17));
        assert_eq!(p_sup(true), Q::new(1, 4));
        assert_eq!(p_sup(false), Q::one());
        assert!(p_sup(true).cmp(ALPHA) == Ordering::Greater);
    }

    #[test]
    fn unequal_robustness_radii_are_not_conflated() {
        assert_eq!(ALT_RADIUS, Q::new(3, 20));
        assert_eq!(NULL_RADIUS, Q::new(1, 20));
        assert!(ALT_RADIUS.cmp(NULL_RADIUS) == Ordering::Greater);
    }

    #[test]
    fn robust_test_is_randomized_at_the_least_favourable_boundary() {
        assert_eq!(PHI_ZERO, Q::zero());
        assert_eq!(PHI_ONE, Q::new(2, 5));
        assert_eq!(NULL_HIGH.mul(PHI_ONE), ALPHA);
        assert_eq!(ALT_LOW.mul(PHI_ONE), Q::new(13, 50));
    }
}
