//! Exact dual-certificate sensitivity fixture for robust experiment design.
//!
//! Design variable u is in [0,1]. Two Bernoulli alternative laws and two
//! Bernoulli null laws vary affinely with u:
//!   A1: p=3/4 + u/10       A2: p=17/20 - u/10
//!   B1: q=1/5 + u/10       B2: q=3/10 - u/10
//!
//! At alpha=1/10, the robust test is phi(0)=0, phi(1)=alpha/q_max(u).
//! Its exact maximin power is alpha*p_min(u)/q_max(u). The least-favourable
//! alternative and null swap at u=1/2, where the optimal dual certificate
//! is non-unique. The value is maximized at the nonsmooth switch.
//!
//! This is a fixed exact-rational sensitivity fixture, not a general
//! differentiable LP solver or continuous global design optimizer.

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

fn r(n: i128, d: i128) -> Q { Q::new(n, d) }

const ALPHA: Q = Q { n: 1, d: 10 };
const HALF: Q = Q { n: 1, d: 2 };

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DesignPoint {
    u: Q,
}

impl DesignPoint {
    fn new(u: Q) -> Self {
        assert!(Q::zero().is_le(u) && u.is_le(Q::one()));
        Self { u }
    }

    fn alternatives(self) -> [Q; 2] {
        let u_over_ten = self.u.div(r(10, 1));
        [r(3, 4).add(u_over_ten), r(17, 20).sub(u_over_ten)]
    }

    fn nulls(self) -> [Q; 2] {
        let u_over_ten = self.u.div(r(10, 1));
        [r(1, 5).add(u_over_ten), r(3, 10).sub(u_over_ten)]
    }

    fn p_min(self) -> Q {
        let p = self.alternatives();
        p[0].min(p[1])
    }

    fn q_max(self) -> Q {
        let q = self.nulls();
        q[0].max(q[1])
    }

    fn test_phi_one(self) -> Q {
        ALPHA.div(self.q_max())
    }

    fn maximin_power(self) -> Q {
        ALPHA.mul(self.p_min()).div(self.q_max())
    }

    fn active_alternatives(self) -> [bool; 2] {
        let p = self.alternatives();
        let min = p[0].min(p[1]);
        [p[0] == min, p[1] == min]
    }

    fn active_nulls(self) -> [bool; 2] {
        let q = self.nulls();
        let max = q[0].max(q[1]);
        [q[0] == max, q[1] == max]
    }

    /// Returns one extreme optimal dual certificate (lambda, w).
    /// At u=1/2, the certificate is not unique; this is a canonical tie-break.
    fn canonical_dual(self) -> ([Q; 2], [Q; 2]) {
        let alt = self.alternatives();
        let null = self.nulls();
        if alt[0].cmp(alt[1]) != Ordering::Greater {
            if null[0].cmp(null[1]) == Ordering::Less {
                ([Q::one(), Q::zero()], [Q::zero(), self.p_min().div(self.q_max())])
            } else {
                ([Q::one(), Q::zero()], [self.p_min().div(self.q_max()), Q::zero()])
            }
        } else if null[0].cmp(null[1]) == Ordering::Greater {
            ([Q::zero(), Q::one()], [self.p_min().div(self.q_max()), Q::zero()])
        } else {
            ([Q::zero(), Q::one()], [Q::zero(), self.p_min().div(self.q_max())])
        }
    }

    /// Exact derivative of alpha*p_min/q_max on the active smooth branch.
    fn smooth_derivative(self) -> Option<Q> {
        if self.u == HALF {
            return None;
        }
        let alt = self.alternatives();
        let null = self.nulls();
        let (p, dp, q, dq) = if self.u.cmp(HALF) == Ordering::Less {
            (alt[0], r(1, 10), null[1], r(-1, 10))
        } else {
            (alt[1], r(-1, 10), null[0], r(1, 10))
        };
        let numerator = dp.mul(q).sub(p.mul(dq));
        Some(ALPHA.mul(numerator).div(q.mul(q)))
    }

    fn one_sided_derivatives_at_switch() -> (Q, Q) {
        let p = r(4, 5);
        let q = r(1, 4);
        let denominator = q.mul(q);
        let left = ALPHA.mul(r(1, 10).mul(q).sub(p.mul(r(-1, 10))))
            .div(denominator);
        let right = ALPHA.mul(r(-1, 10).mul(q).sub(p.mul(r(1, 10))))
            .div(denominator);
        (left, right)
    }

    fn dual_objective(self, lambda: [Q; 2], w: [Q; 2]) -> Q {
        let alternatives = self.alternatives();
        let nulls = self.nulls();
        let alt_mix_p = lambda[0].mul(alternatives[0]).add(lambda[1].mul(alternatives[1]));
        let null_mix_one = w[0].mul(nulls[0]).add(w[1].mul(nulls[1]));
        let null_mix_zero = w[0].mul(Q::one().sub(nulls[0]))
            .add(w[1].mul(Q::one().sub(nulls[1])));
        ALPHA.mul(w[0].add(w[1]))
            .add(positive(alt_mix_p.sub(null_mix_one)))
            .add(positive(Q::one().sub(alt_mix_p).sub(null_mix_zero)))
    }

    fn dual_feasible(self, lambda: [Q; 2], w: [Q; 2]) -> bool {
        if lambda[0].add(lambda[1]) != Q::one() {
            return false;
        }
        if !lambda.iter().all(|x| x.is_le(Q::one())) ||
            !w.iter().all(|x| Q::zero().is_le(*x)) {
            return false;
        }
        let alternatives = self.alternatives();
        let nulls = self.nulls();
        let alt_one = lambda[0].mul(alternatives[0])
            .add(lambda[1].mul(alternatives[1]));
        let null_one = w[0].mul(nulls[0]).add(w[1].mul(nulls[1]));
        let null_zero = w[0].mul(Q::one().sub(nulls[0]))
            .add(w[1].mul(Q::one().sub(nulls[1])));
        alt_one.is_le(null_one)
            && Q::one().sub(alt_one).is_le(null_zero)
    }
}

fn positive(x: Q) -> Q {
    if x.cmp(Q::zero()) == Ordering::Greater { x } else { Q::zero() }
}

fn robust_value(u: Q) -> Q {
    DesignPoint::new(u).maximin_power()
}

fn finite_difference_left(h: Q) -> Q {
    let center = robust_value(HALF);
    let left = robust_value(HALF.sub(h));
    center.sub(left).div(h)
}

fn finite_difference_right(h: Q) -> Q {
    let right = robust_value(HALF.add(h));
    let center = robust_value(HALF);
    right.sub(center).div(h)
}

fn run_fixture() {
    let points = [r(0, 1), r(1, 4), r(1, 2), r(3, 4), r(1, 1)];
    for u in points {
        let point = DesignPoint::new(u);
        let (lambda, w) = point.canonical_dual();
        assert!(point.dual_feasible(lambda, w));
        assert_eq!(point.dual_objective(lambda, w), point.maximin_power());
        println!(
            "u={u};p_min={};q_max={};phi_1={};V={};dV={:?};lambda={:?};w={:?}",
            point.p_min(),
            point.q_max(),
            point.test_phi_one(),
            point.maximin_power(),
            point.smooth_derivative(),
            lambda,
            w
        );
    }
    let (left, right) = DesignPoint::one_sided_derivatives_at_switch();
    println!("switch_left_derivative={left};switch_right_derivative={right}");
    println!("left_difference_quotient_h_1_100={}", finite_difference_left(r(1, 100)));
    println!("right_difference_quotient_h_1_100={}", finite_difference_right(r(1, 100)));
    println!("switch_clarke_interval=[{},{}]", right, left);
}

fn main() { run_fixture(); }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_values_and_tests_are_exact() {
        let expected = [
            (r(0, 1), r(3, 4), r(3, 10), r(1, 3), r(1, 4)),
            (r(1, 4), r(31, 40), r(11, 40), r(4, 11), r(31, 110)),
            (r(1, 2), r(4, 5), r(1, 4), r(2, 5), r(8, 25)),
            (r(3, 4), r(31, 40), r(11, 40), r(4, 11), r(31, 110)),
            (r(1, 1), r(3, 4), r(3, 10), r(1, 3), r(1, 4)),
        ];
        for (u, p, q, phi, value) in expected {
            let point = DesignPoint::new(u);
            assert_eq!(point.p_min(), p);
            assert_eq!(point.q_max(), q);
            assert_eq!(point.test_phi_one(), phi);
            assert_eq!(point.maximin_power(), value);
        }
    }

    #[test]
    fn least_favourable_members_switch_at_one_half() {
        let left = DesignPoint::new(r(1, 4));
        let tie = DesignPoint::new(r(1, 2));
        let right = DesignPoint::new(r(3, 4));
        assert_eq!(left.active_alternatives(), [true, false]);
        assert_eq!(left.active_nulls(), [false, true]);
        assert_eq!(right.active_alternatives(), [false, true]);
        assert_eq!(right.active_nulls(), [true, false]);
        assert_eq!(tie.active_alternatives(), [true, true]);
        assert_eq!(tie.active_nulls(), [true, true]);
    }

    #[test]
    fn canonical_primal_dual_certificates_match_on_all_points() {
        for u in [r(0, 1), r(1, 4), r(1, 2), r(3, 4), r(1, 1)] {
            let point = DesignPoint::new(u);
            let (lambda, w) = point.canonical_dual();
            assert!(point.dual_feasible(lambda, w));
            assert_eq!(point.dual_objective(lambda, w), point.maximin_power());
        }
    }

    #[test]
    fn switch_has_a_non_unique_optimal_dual_certificate() {
        let point = DesignPoint::new(HALF);
        let lambda = [r(1, 3), r(2, 3)];
        let w = [r(8, 5), r(8, 5)];
        assert!(point.dual_feasible(lambda, w));
        assert_eq!(point.dual_objective(lambda, w), r(8, 25));
        let canonical = point.canonical_dual();
        assert_ne!(canonical, (lambda, w));
        assert_eq!(point.active_alternatives(), [true, true]);
        assert_eq!(point.active_nulls(), [true, true]);
    }

    #[test]
    fn smooth_dual_sensitivities_match_exact_envelope_derivatives() {
        assert_eq!(
            DesignPoint::new(r(0, 1)).smooth_derivative(),
            Some(r(7, 60))
        );
        assert_eq!(
            DesignPoint::new(r(1, 4)).smooth_derivative(),
            Some(r(84, 605))
        );
        assert_eq!(
            DesignPoint::new(r(3, 4)).smooth_derivative(),
            Some(r(-84, 605))
        );
        assert_eq!(
            DesignPoint::new(r(1, 1)).smooth_derivative(),
            Some(r(-7, 60))
        );
        assert_eq!(DesignPoint::new(HALF).smooth_derivative(), None);
    }

    #[test]
    fn switch_has_opposite_one_sided_derivatives_and_set_valued_clarke_slope() {
        let (left, right) = DesignPoint::one_sided_derivatives_at_switch();
        assert_eq!(left, r(21, 125));
        assert_eq!(right, r(-21, 125));
        assert_eq!(right, r(-21, 125));
        assert_eq!(left, r(21, 125));
    }

    #[test]
    fn one_sided_difference_quotients_converge_to_switch_derivatives() {
        let h = r(1, 10000);
        let left_fd = finite_difference_left(h);
        let right_fd = finite_difference_right(h);
        let (left, right) = DesignPoint::one_sided_derivatives_at_switch();
        assert!(left_fd.cmp(left) == Ordering::Less);
        assert!(right_fd.cmp(right) == Ordering::Greater);
        assert_eq!(left_fd, r(4200, 25001));
        assert_eq!(right_fd, r(-4200, 25001));
    }

    #[test]
    fn robust_power_is_maximized_at_the_active_set_switch() {
        let center = robust_value(HALF);
        for u in [r(0, 1), r(1, 4), r(3, 4), r(1, 1)] {
            assert!(robust_value(u).cmp(center) == Ordering::Less);
        }
        assert_eq!(center, r(8, 25));
    }
}
