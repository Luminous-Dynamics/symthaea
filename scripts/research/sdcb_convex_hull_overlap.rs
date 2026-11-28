//! Exact rational verifier for the SDCB convex-hull overlap fixture.
//!
//! This is a fixture-specific known-answer verifier, not a general LP solver.
//! It uses exact i128 rational arithmetic on Y={y1,y2} and checks:
//! - pairwise-disjoint null/alternative laws;
//! - an exact convex-mixture identity;
//! - a matching primal randomized test and dual certificate;
//! - the conservative p_sup-GLRT comparator.
//!
//! Run:
//!   rustc --edition=2024 --test scripts/research/sdcb_convex_hull_overlap.rs -o /tmp/sdcb-convex-hull-tests
//!   /tmp/sdcb-convex-hull-tests
//!   rustc --edition=2024 scripts/research/sdcb_convex_hull_overlap.rs -o /tmp/sdcb-convex-hull
//!   /tmp/sdcb-convex-hull

use std::cmp::Ordering;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Rat {
    n: i128,
    d: i128,
}

impl Rat {
    fn new(n: i128, d: i128) -> Self {
        assert_ne!(d, 0, "zero rational denominator");
        let sign = if d < 0 { -1 } else { 1 };
        let n = n * sign;
        let d = d.abs();
        let g = gcd(n.abs(), d);
        Self { n: n / g, d: d / g }
    }

    fn zero() -> Self { Self::new(0, 1) }
    fn one() -> Self { Self::new(1, 1) }

    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.n * rhs.d + rhs.n * self.d,
            self.d * rhs.d,
        )
    }

    fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.n * rhs.d - rhs.n * self.d,
            self.d * rhs.d,
        )
    }

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.n, self.d * rhs.d)
    }

    fn div(self, rhs: Self) -> Self {
        assert_ne!(rhs.n, 0, "division by zero rational");
        Self::new(self.n * rhs.d, self.d * rhs.n)
    }

    fn cmp(self, rhs: Self) -> Ordering {
        (self.n * rhs.d).cmp(&(rhs.n * self.d))
    }

    fn max(self, rhs: Self) -> Self {
        if self.cmp(rhs) == Ordering::Less { rhs } else { self }
    }

    fn is_nonnegative(self) -> bool { self.n >= 0 }
}

impl fmt::Display for Rat {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Law {
    id: &'static str,
    p: [Rat; 2],
}

impl Law {
    fn valid(self) -> bool {
        self.p.iter().all(|x| x.is_nonnegative())
            && self.p[0].add(self.p[1]) == Rat::one()
    }
}

const ALPHA_NUM: i128 = 1;
const ALPHA_DEN: i128 = 10;

fn alpha() -> Rat { Rat::new(ALPHA_NUM, ALPHA_DEN) }

fn null_family() -> [Law; 1] {
    [Law {
        id: "B1",
        p: [Rat::new(1, 3), Rat::new(2, 3)],
    }]
}

fn alternative_family() -> [Law; 2] {
    [
        Law { id: "A1", p: [Rat::one(), Rat::zero()] },
        Law { id: "A2", p: [Rat::zero(), Rat::one()] },
    ]
}

fn mixture(family: &[Law], weights: &[Rat]) -> [Rat; 2] {
    assert_eq!(family.len(), weights.len());
    let mut result = [Rat::zero(), Rat::zero()];
    for (law, weight) in family.iter().zip(weights.iter()) {
        assert!(law.valid());
        assert!(weight.is_nonnegative());
        for (k, slot) in result.iter_mut().enumerate() {
            *slot = slot.add(law.p[k].mul(*weight));
        }
    }
    result
}

fn pairwise_intersection(nulls: &[Law], alternatives: &[Law]) -> bool {
    nulls.iter().any(|b| alternatives.iter().any(|a| a.p == b.p))
}

fn expectation(phi: [Rat; 2], law: Law) -> Rat {
    phi[0].mul(law.p[0]).add(phi[1].mul(law.p[1]))
}

fn max_probability_at(family: &[Law], outcome: usize) -> Rat {
    family.iter().map(|law| law.p[outcome])
        .reduce(Rat::max).expect("non-empty family")
}

fn glrt_score_ratio(
    outcome: usize,
    nulls: &[Law],
    alternatives: &[Law],
) -> Rat {
    max_probability_at(alternatives, outcome)
        .div(max_probability_at(nulls, outcome))
}

/// Exact p_sup for a finite outcome space and the max-density-ratio ordering.
/// The tail includes ties: score(Y) >= score(observed).
fn p_sup(
    observed: usize,
    nulls: &[Law],
    alternatives: &[Law],
) -> Rat {
    let observed_score = glrt_score_ratio(observed, nulls, alternatives);
    nulls.iter().map(|null| {
        (0..2).filter(|&candidate| {
            glrt_score_ratio(candidate, nulls, alternatives)
                .cmp(observed_score) != Ordering::Less
        }).fold(Rat::zero(), |sum, candidate| {
            sum.add(null.p[candidate])
        })
    }).reduce(Rat::max).expect("non-empty null family")
}

fn positive_part_dual_integral(
    alternatives: &[Law],
    alternative_weights: &[Rat],
    nulls: &[Law],
    null_weights: &[Rat],
) -> Rat {
    let qa = mixture(alternatives, alternative_weights);
    let qb = mixture(nulls, null_weights);
    (0..2).fold(Rat::zero(), |sum, k| {
        sum.add(qa[k].sub(qb[k]).max(Rat::zero()))
    })
}

fn dual_objective(
    alpha: Rat,
    alternatives: &[Law],
    alternative_weights: &[Rat],
    nulls: &[Law],
    null_weights: &[Rat],
) -> Rat {
    let null_weight_sum = null_weights.iter()
        .copied().fold(Rat::zero(), Rat::add);
    alpha.mul(null_weight_sum).add(positive_part_dual_integral(
        alternatives, alternative_weights, nulls, null_weights,
    ))
}

fn run_fixture() {
    let nulls = null_family();
    let alternatives = alternative_family();
    assert!(nulls.iter().all(|x| x.valid()));
    assert!(alternatives.iter().all(|x| x.valid()));

    let alpha = alpha();
    let lambda = [Rat::new(1, 3), Rat::new(2, 3)];
    let w = [Rat::one()];

    let q_a = mixture(&alternatives, &lambda);
    let q_b = mixture(&nulls, &w);
    assert_eq!(q_a, q_b, "alternative and null mixtures must coincide");
    assert!(!pairwise_intersection(&nulls, &alternatives));

    let phi = [alpha, alpha];
    let sizes: Vec<Rat> = nulls.iter().map(|law| expectation(phi, *law)).collect();
    let powers: Vec<Rat> = alternatives.iter()
        .map(|law| expectation(phi, *law)).collect();
    assert!(sizes.iter().all(|size| size.cmp(alpha) != Ordering::Greater));
    let primal_lower = powers.iter().copied().reduce(|a, b| {
        if a.cmp(b) == Ordering::Greater { b } else { a }
    }).expect("non-empty alternative family");

    let dual_upper = dual_objective(alpha, &alternatives, &lambda, &nulls, &w);
    let gap = dual_upper.sub(primal_lower);
    assert_eq!(primal_lower, alpha);
    assert_eq!(dual_upper, alpha);
    assert_eq!(gap, Rat::zero());

    let p_values = [p_sup(0, &nulls, &alternatives), p_sup(1, &nulls, &alternatives)];
    let reject = p_values.map(|p| p.cmp(alpha) != Ordering::Greater);
    let p_sup_power: Vec<Rat> = alternatives.iter().map(|law| {
        (0..2).filter(|&k| reject[k]).fold(Rat::zero(), |sum, k| {
            sum.add(law.p[k])
        })
    }).collect();

    println!("status=EXACT_RATIONAL_FINITE_OUTCOME_FIXTURE");
    println!("pairwise_intersection=false");
    println!("convex_hull_intersection=true");
    println!("alpha={alpha}");
    println!("alternative_mixture=({},{})", q_a[0], q_a[1]);
    println!("null_mixture=({},{})", q_b[0], q_b[1]);
    println!("primal_worst_case_power={primal_lower}");
    println!("dual_upper_bound={dual_upper}");
    println!("primal_dual_gap={gap}");
    println!("p_sup_at_y1={}", p_values[0]);
    println!("p_sup_at_y2={}", p_values[1]);
    println!("p_sup_glrt_power_A1={}", p_sup_power[0]);
    println!("p_sup_glrt_power_A2={}", p_sup_power[1]);
}

fn main() { run_fixture(); }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_laws_are_valid_probability_vectors() {
        assert!(null_family().iter().chain(alternative_family().iter()).all(|x| x.valid()));
    }

    #[test]
    fn no_pairwise_family_laws_intersect() {
        assert!(!pairwise_intersection(&null_family(), &alternative_family()));
    }

    #[test]
    fn alternative_mixture_equals_null_law_exactly() {
        let a = alternative_family();
        let b = null_family();
        let lambda = [Rat::new(1, 3), Rat::new(2, 3)];
        assert_eq!(mixture(&a, &lambda), mixture(&b, &[Rat::one()]));
        assert_eq!(mixture(&a, &lambda), [Rat::new(1, 3), Rat::new(2, 3)]);
    }

    #[test]
    fn constant_randomized_test_is_primal_optimum() {
        let a = alternative_family();
        let b = null_family();
        let alpha = alpha();
        let phi = [alpha, alpha];
        for law in b.iter() { assert_eq!(expectation(phi, *law), alpha); }
        for law in a.iter() { assert_eq!(expectation(phi, *law), alpha); }
    }

    #[test]
    fn dual_certificate_matches_primal_exactly() {
        let a = alternative_family();
        let b = null_family();
        let alpha = alpha();
        let lambda = [Rat::new(1, 3), Rat::new(2, 3)];
        let w = [Rat::one()];
        let positive = positive_part_dual_integral(&a, &lambda, &b, &w);
        let upper = dual_objective(alpha, &a, &lambda, &b, &w);
        assert_eq!(positive, Rat::zero());
        assert_eq!(upper, alpha);
        assert_eq!(upper.sub(alpha), Rat::zero());
    }

    #[test]
    fn p_sup_glrt_is_valid_but_has_zero_power_at_alpha_point_one() {
        let a = alternative_family();
        let b = null_family();
        let alpha = alpha();
        let p = [p_sup(0, &b, &a), p_sup(1, &b, &a)];
        assert_eq!(p, [Rat::new(1, 3), Rat::one()]);
        let reject = p.map(|value| value.cmp(alpha) != Ordering::Greater);
        assert_eq!(reject, [false, false]);
        let powers: Vec<Rat> = a.iter().map(|law| {
            (0..2).filter(|&k| reject[k])
                .fold(Rat::zero(), |sum, k| sum.add(law.p[k]))
        }).collect();
        assert_eq!(powers, [Rat::zero(), Rat::zero()]);
    }

    #[test]
    fn adding_a_duplicate_member_does_not_change_mixture_identity() {
        let a = alternative_family();
        let duplicated = [a[0], a[1], a[0]];
        let weights = [Rat::new(1, 6), Rat::new(2, 3), Rat::new(1, 6)];
        assert_eq!(
            mixture(&a, &[Rat::new(1, 3), Rat::new(2, 3)]),
            mixture(&duplicated, &weights)
        );
    }
}
