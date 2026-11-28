//! Exact two-step fixture for composite-null lower-envelope test induction.
//!
//! Under each null member B_j, g_n^j = E_j[phi_N | F_n] is a martingale.
//! The robust process is g_n = min_j g_n^j. For a true member k and any
//! bounded stopping time tau, g_tau <= g_tau^k pointwise, hence
//! E_k[g_tau] <= E_k[g_tau^k] = E_k[phi_N] <= alpha.
//!
//! This exact finite fixture also demonstrates:
//! - terminal matching at N;
//! - every deterministic public stopping rule is valid under both nulls;
//! - using only one member's process under the other null can fail;
//! - max aggregation can fail;
//! - the lower envelope need not be a supermartingale under each null.
//!
//! Scope: exact two-step Bernoulli finite-tree known answer, not a general
//! composite-sequential inference engine.

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

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.n, self.d * rhs.d)
    }

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.d - rhs.n * self.d, self.d * rhs.d)
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

const P_B1: Q = Q { n: 1, d: 5 };
const P_B2: Q = Q { n: 4, d: 5 };
const ALPHA: Q = Q { n: 1, d: 20 };
const TERMINAL_REJECTION: Q = Q { n: 5, d: 32 };

fn prob_x(p: Q, x: bool) -> Q {
    if x { p } else { Q::one().sub(p) }
}

fn terminal_phi(x1: bool, x2: bool) -> Q {
    if x1 != x2 { TERMINAL_REJECTION } else { Q::zero() }
}

/// Conditional expectation at the full filtration equals the measurable test.
fn member_conditional_at_horizon(_p: Q, x1: bool, x2: bool) -> Q {
    terminal_phi(x1, x2)
}

fn terminal_size(p: Q) -> Q {
    let mut total = Q::zero();
    for x1 in [false, true] {
        for x2 in [false, true] {
            total = total.add(prob_x(p, x1).mul(prob_x(p, x2)).mul(terminal_phi(x1, x2)));
        }
    }
    total
}

/// g_1^j(x1) = E_j[phi_N | X1=x1].
fn member_conditional(p: Q, x1: bool) -> Q {
    [false, true].into_iter().fold(Q::zero(), |sum, x2| {
        sum.add(prob_x(p, x2).mul(terminal_phi(x1, x2)))
    })
}

/// Robust lower envelope min_j g_1^j(x1).
fn robust_lower(x1: bool) -> Q {
    member_conditional(P_B1, x1).min(member_conditional(P_B2, x1))
}

/// Upper-envelope negative control max_j g_1^j(x1).
fn max_aggregate(x1: bool) -> Q {
    member_conditional(P_B1, x1).max(member_conditional(P_B2, x1))
}

#[derive(Clone, Copy, Debug)]
enum StopRule {
    StopAtZero,
    StopAtOneEverywhere,
    StopAtOneWhenX1Zero,
    StopAtOneWhenX1One,
    ContinueToHorizon,
}

fn stop_at_one(rule: StopRule, x1: bool) -> bool {
    match rule {
        StopRule::StopAtZero => false,
        StopRule::StopAtOneEverywhere => true,
        StopRule::StopAtOneWhenX1Zero => !x1,
        StopRule::StopAtOneWhenX1One => x1,
        StopRule::ContinueToHorizon => false,
    }
}

/// Expected value of the robust induced level at a deterministic stopping time.
fn expected_robust_level(p_true: Q, rule: StopRule) -> Q {
    if matches!(rule, StopRule::StopAtZero) {
        return terminal_size(P_B1).min(terminal_size(P_B2));
    }
    [false, true].into_iter().fold(Q::zero(), |sum, x1| {
        let stopped_value = if stop_at_one(rule, x1) {
            robust_lower(x1)
        } else {
            member_conditional(p_true, x1)
        };
        sum.add(prob_x(p_true, x1).mul(stopped_value))
    })
}

/// Expected value under p_true of a single member's conditional process.
fn expected_single_member_process(p_true: Q, p_model: Q) -> Q {
    [false, true].into_iter().fold(Q::zero(), |sum, x1| {
        sum.add(prob_x(p_true, x1).mul(member_conditional(p_model, x1)))
    })
}

/// Expected value under p_true of max_j g_1^j.
fn expected_max_process(p_true: Q) -> Q {
    [false, true].into_iter().fold(Q::zero(), |sum, x1| {
        sum.add(prob_x(p_true, x1).mul(max_aggregate(x1)))
    })
}

fn stopping_rules() -> [StopRule; 5] {
    [
        StopRule::StopAtZero,
        StopRule::StopAtOneEverywhere,
        StopRule::StopAtOneWhenX1Zero,
        StopRule::StopAtOneWhenX1One,
        StopRule::ContinueToHorizon,
    ]
}

fn run_fixture() {
    assert_eq!(terminal_size(P_B1), ALPHA);
    assert_eq!(terminal_size(P_B2), ALPHA);
    assert_eq!(robust_lower(false), Q::new(1, 32));
    assert_eq!(robust_lower(true), Q::new(1, 32));
    assert_eq!(terminal_phi(false, false), Q::zero());
    assert_eq!(terminal_phi(false, true), TERMINAL_REJECTION);
    assert_eq!(terminal_phi(true, false), TERMINAL_REJECTION);
    assert_eq!(terminal_phi(true, true), Q::zero());

    println!("status=EXACT_RATIONAL_COMPOSITE_NULL_INDUCTION_FIXTURE");
    println!("alpha={ALPHA}");
    println!("terminal_size_B1={}", terminal_size(P_B1));
    println!("terminal_size_B2={}", terminal_size(P_B2));
    println!(
        "B1_g1_x0={};B1_g1_x1={}",
        member_conditional(P_B1, false),
        member_conditional(P_B1, true)
    );
    println!(
        "B2_g1_x0={};B2_g1_x1={}",
        member_conditional(P_B2, false),
        member_conditional(P_B2, true)
    );
    println!(
        "robust_g1_x0={};robust_g1_x1={}",
        robust_lower(false),
        robust_lower(true)
    );
    println!("single_B1_process_under_B2={}", expected_single_member_process(P_B2, P_B1));
    println!("max_process_under_B1={}", expected_max_process(P_B1));
    println!("max_process_under_B2={}", expected_max_process(P_B2));
    for (index, rule) in stopping_rules().into_iter().enumerate() {
        println!(
            "stop_rule_{index}_B1={};B2={}",
            expected_robust_level(P_B1, rule),
            expected_robust_level(P_B2, rule)
        );
    }
}

fn main() { run_fixture(); }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_test_has_exact_uniform_size() {
        assert_eq!(terminal_size(P_B1), ALPHA);
        assert_eq!(terminal_size(P_B2), ALPHA);
    }

    #[test]
    fn member_specific_conditional_levels_match_frozen_values() {
        assert_eq!(member_conditional(P_B1, true), Q::new(1, 8));
        assert_eq!(member_conditional(P_B1, false), Q::new(1, 32));
        assert_eq!(member_conditional(P_B2, true), Q::new(1, 32));
        assert_eq!(member_conditional(P_B2, false), Q::new(1, 8));
    }

    #[test]
    fn lower_envelope_is_one_thirty_second_at_every_first_history() {
        assert_eq!(robust_lower(false), Q::new(1, 32));
        assert_eq!(robust_lower(true), Q::new(1, 32));
    }

    #[test]
    fn induced_process_matches_terminal_randomized_test() {
        for p in [P_B1, P_B2] {
            for x1 in [false, true] {
                for x2 in [false, true] {
                    assert_eq!(
                        member_conditional_at_horizon(p, x1, x2),
                        terminal_phi(x1, x2)
                    );
                }
            }
        }
    }

    #[test]
    fn every_deterministic_stopping_rule_controls_both_null_members() {
        for rule in stopping_rules() {
            assert!(expected_robust_level(P_B1, rule).is_le(ALPHA));
            assert!(expected_robust_level(P_B2, rule).is_le(ALPHA));
        }
        assert_eq!(expected_robust_level(P_B1, StopRule::StopAtZero), ALPHA);
        assert_eq!(expected_robust_level(P_B1, StopRule::ContinueToHorizon), ALPHA);
        assert_eq!(expected_robust_level(P_B2, StopRule::StopAtZero), ALPHA);
        assert_eq!(expected_robust_level(P_B2, StopRule::ContinueToHorizon), ALPHA);
    }

    #[test]
    fn reusing_only_one_null_members_process_can_violate_robust_size() {
        let crossed = expected_single_member_process(P_B2, P_B1);
        assert_eq!(crossed, Q::new(17, 160));
        assert!(crossed.cmp(ALPHA) == Ordering::Greater);
    }

    #[test]
    fn max_aggregation_can_violate_robust_size() {
        assert_eq!(expected_max_process(P_B1), Q::new(1, 8));
        assert_eq!(expected_max_process(P_B2), Q::new(1, 8));
        assert!(expected_max_process(P_B1).cmp(ALPHA) == Ordering::Greater);
    }

    #[test]
    fn lower_envelope_is_not_generally_a_memberwise_supermartingale() {
        // Under B1 after X1=1, the conditional expectation of terminal phi
        // is 1/8, while the robust lower-envelope value at that history is 1/32.
        assert_eq!(member_conditional(P_B1, true), Q::new(1, 8));
        assert_eq!(robust_lower(true), Q::new(1, 32));
        assert!(member_conditional(P_B1, true).cmp(robust_lower(true)) == Ordering::Greater);
    }
}
