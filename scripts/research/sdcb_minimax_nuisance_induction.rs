//! Exact finite integration of a certified maximin terminal test with
//! composite-null lower-envelope induction over adaptive nuisance trajectories.
//!
//! Observation path: (X1,X2) in {00,01,10,11}.
//! Nuisance paths: p1 in {1/5,4/5}; predictable p2(0),p2(1) in {1/5,4/5}.
//! All 8 deterministic history-adaptive trajectory laws are frozen members.
//!
//! Alternative law A=(0,1/2,1/2,0).
//! alpha=1/10; terminal test phi=(0,1/8,1/8,0).
//! Worst null discordance is 4/5, so worst-null size is 1/10.
//! Alternative power is 1/8.
//!
//! Dual certificate: lambda=1 on A; null mixture is half each of the
//! (p1,p20,p21)=(1/5,4/5,1/5) and (4/5,4/5,1/5) trajectory laws;
//! total unnormalized null weight w=5/4, with per-member weights 5/8.
//! Pointwise A <= w Q0, positive-part slack zero, dual objective 1/8.
//! Therefore the exact finite-family maximin value is 1/8.
//!
//! Memberwise induced tests g_n^j=E_j[phi|F_n]; robust lower envelope g_n=min_j g_n^j.
//! Exact result: g0=1/40; g1(0)=g1(1)=1/40; g2=phi.
//! For every null member and bounded public stopping time tau<=2:
//! E_j[g_tau] <= E_j[phi] <= 1/10.
//!
//! This is an exact finite integration benchmark for a published construction,
//! not a new anytime-valid theorem, continuous-family solver or physical result.

use std::cmp::Ordering;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Q { n: i128, d: i128 }

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
fn one_minus(x: Q) -> Q { Q::one().sub(x) }

const P_LOW: Q = Q { n: 1, d: 5 };
const P_HIGH: Q = Q { n: 4, d: 5 };
const ALPHA: Q = Q { n: 1, d: 10 };
const GAMMA: Q = Q { n: 1, d: 8 };

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Trajectory {
    id: usize,
    p1: Q,
    p2_if_x1_zero: Q,
    p2_if_x1_one: Q,
}

impl Trajectory {
    fn joint(self) -> [Q; 4] {
        let x1_zero = one_minus(self.p1);
        let x1_one = self.p1;
        let p2_zero = self.p2_if_x1_zero;
        let p2_one = self.p2_if_x1_one;
        [
            x1_zero.mul(one_minus(p2_zero)), // 00
            x1_zero.mul(p2_zero),            // 01
            x1_one.mul(one_minus(p2_one)),   // 10
            x1_one.mul(p2_one),              // 11
        ]
    }

    fn discordance_probability(self) -> Q {
        let path = self.joint();
        path[1].add(path[2])
    }

    fn terminal_size(self) -> Q {
        self.discordance_probability().mul(GAMMA)
    }

    /// E[phi | X1=0] = p2(0)*gamma; E[phi | X1=1]=(1-p2(1))*gamma.
    fn conditional_at_one(self) -> [Q; 2] {
        [
            self.p2_if_x1_zero.mul(GAMMA),
            one_minus(self.p2_if_x1_one).mul(GAMMA),
        ]
    }
}

fn all_trajectories() -> Vec<Trajectory> {
    let ps = [P_LOW, P_HIGH];
    let mut out = Vec::new();
    let mut id = 0;
    for p1 in ps {
        for p20 in ps {
            for p21 in ps {
                out.push(Trajectory {
                    id, p1, p2_if_x1_zero: p20, p2_if_x1_one: p21,
                });
                id += 1;
            }
        }
    }
    out
}

fn alternative_law() -> [Q; 4] {
    [Q::zero(), r(1, 2), r(1, 2), Q::zero()]
}

fn terminal_test() -> [Q; 4] {
    [Q::zero(), GAMMA, GAMMA, Q::zero()]
}

fn member_g2_at_history(_t: Trajectory, x1: bool, x2: bool) -> Q {
    if x1 != x2 { GAMMA } else { Q::zero() }
}

fn expectation(law: [Q; 4], test: [Q; 4]) -> Q {
    (0..4).fold(Q::zero(), |sum, k| sum.add(law[k].mul(test[k])))
}

fn robust_g0(trajectories: &[Trajectory]) -> Q {
    trajectories.iter().map(|t| t.terminal_size())
        .reduce(Q::min).expect("nonempty trajectory family")
}

fn member_g1(t: Trajectory) -> [Q; 2] {
    t.conditional_at_one()
}

fn robust_g1(trajectories: &[Trajectory]) -> [Q; 2] {
    let mut values = [Q::one(), Q::one()];
    for t in trajectories {
        let g = member_g1(*t);
        values[0] = values[0].min(g[0]);
        values[1] = values[1].min(g[1]);
    }
    values
}

#[derive(Clone, Copy, Debug)]
enum StopRule {
    StopAtZero,
    StopAtOneOnNeither,
    StopAtOneOnX1Zero,
    StopAtOneOnX1One,
    StopAtOneOnBoth,
}

fn expected_induced_level(t: Trajectory, robust_zero: Q, robust_one: [Q; 2], rule: StopRule) -> Q {
    let x1_zero_prob = one_minus(t.p1);
    let x1_one_prob = t.p1;
    let g_member = member_g1(t);

    match rule {
        StopRule::StopAtZero => robust_zero,
        StopRule::StopAtOneOnNeither => {
            x1_zero_prob.mul(g_member[0]).add(x1_one_prob.mul(g_member[1]))
        }
        StopRule::StopAtOneOnX1Zero => {
            x1_zero_prob.mul(robust_one[0]).add(x1_one_prob.mul(g_member[1]))
        }
        StopRule::StopAtOneOnX1One => {
            x1_zero_prob.mul(g_member[0]).add(x1_one_prob.mul(robust_one[1]))
        }
        StopRule::StopAtOneOnBoth => {
            x1_zero_prob.mul(robust_one[0]).add(x1_one_prob.mul(robust_one[1]))
        }
    }
}

fn all_stopping_rules() -> [StopRule; 5] {
    [
        StopRule::StopAtZero,
        StopRule::StopAtOneOnNeither,
        StopRule::StopAtOneOnX1Zero,
        StopRule::StopAtOneOnX1One,
        StopRule::StopAtOneOnBoth,
    ]
}

fn expected_alt_induced_level(robust_zero: Q, robust_one: [Q; 2], rule: StopRule) -> Q {
    let half = r(1, 2);
    let alt_conditional = r(1, 8);
    match rule {
        StopRule::StopAtZero => robust_zero,
        StopRule::StopAtOneOnNeither => alt_conditional,
        StopRule::StopAtOneOnX1Zero => half.mul(robust_one[0]).add(half.mul(alt_conditional)),
        StopRule::StopAtOneOnX1One => half.mul(alt_conditional).add(half.mul(robust_one[1])),
        StopRule::StopAtOneOnBoth => half.mul(robust_one[0]).add(half.mul(robust_one[1])),
    }
}

/// Exact null-mixture dual witness: equal mixture of trajectory IDs 2 and 6
/// in the nested-loop order (IDs are 0-based); both have worst discordance 4/5.
fn dual_null_mixture(trajectories: &[Trajectory]) -> [Q; 4] {
    let q2 = trajectories[2].joint();
    let q6 = trajectories[6].joint();
    (0..4).map(|k| q2[k].add(q6[k]).div(r(2, 1)))
        .try_into().expect("four outcomes")
}

fn dual_pointwise_check() -> bool {
    let trajectories = all_trajectories();
    let q0 = dual_null_mixture(&trajectories);
    let alt = alternative_law();
    let w = r(5, 4);
    (0..4).all(|k| alt[k].is_le(w.mul(q0[k])))
}

fn dual_objective() -> Q {
    ALPHA.mul(r(5, 4))
}

fn run_fixture() {
    let trajectories = all_trajectories();
    let robust_zero = robust_g0(&trajectories);
    let robust_one = robust_g1(&trajectories);
    let null_sizes: Vec<Q> = trajectories.iter().map(|t| t.terminal_size()).collect();
    let max_size = null_sizes.iter().copied().reduce(Q::max).unwrap();
    let min_size = null_sizes.iter().copied().reduce(Q::min).unwrap();
    let alt = alternative_law();
    let phi = terminal_test();
    let alt_power = expectation(alt, phi);
    let max_stopped = all_stopping_rules().into_iter()
        .flat_map(|rule| trajectories.iter().map(move |t| expected_induced_level(*t, robust_zero, robust_one, rule)))
        .reduce(Q::max).unwrap();

    assert_eq!(trajectories.len(), 8);
    assert_eq!(robust_zero, r(1, 40));
    assert_eq!(robust_one, [r(1, 40), r(1, 40)]);
    assert_eq!(min_size, r(1, 40));
    assert_eq!(max_size, ALPHA);
    assert_eq!(alt_power, r(1, 8));
    assert!(dual_pointwise_check());
    assert_eq!(dual_objective(), alt_power);
    assert_eq!(max_stopped, ALPHA);

    println!("status=EXACT_MINIMAX_NUISANCE_INDUCTION_INTEGRATION");
    println!("null_trajectory_count={}", trajectories.len());
    println!("alpha={ALPHA}");
    println!("primal_test={:?}", phi);
    println!("null_size_min={min_size};null_size_max={max_size}");
    println!("alternative_power={alt_power}");
    println!("dual_lambda=1;dual_null_weights={:?};dual_upper_bound={}", [r(5, 8), r(5, 8)], dual_objective());
    println!("dual_pointwise_feasible={}", dual_pointwise_check());
    println!("robust_g0={robust_zero};robust_g1={:?};terminal_g2={:?}", robust_one, phi);
    println!("max_expected_level_over_all_stops_and_null_members={max_stopped}");
    for (idx, rule) in all_stopping_rules().into_iter().enumerate() {
        let values: Vec<Q> = trajectories.iter()
            .map(|t| expected_induced_level(*t, robust_zero, robust_one, rule))
            .collect();
        let worst = values.iter().copied().reduce(Q::max).unwrap();
        println!("stopping_rule_{idx}_worst_null_level={worst};alternative_induced_level={}",
            expected_alt_induced_level(robust_zero, robust_one, rule));
    }
    for t in &trajectories {
        let g = member_g1(*t);
        println!("trajectory_{}_terminal_size={};g1_x0={};g1_x1={}",
            t.id, t.terminal_size(), g[0], g[1]);
    }
}

fn main() { run_fixture(); }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_eight_complete_trajectory_laws_are_normalized() {
        let ts = all_trajectories();
        assert_eq!(ts.len(), 8);
        for t in ts {
            assert_eq!(t.joint().iter().copied().fold(Q::zero(), Q::add), Q::one());
        }
    }

    #[test]
    fn family_contains_persistent_and_precommitted_subfamilies() {
        let ts = all_trajectories();
        assert!(ts.iter().any(|t| t.p1 == P_LOW && t.p2_if_x1_zero == P_LOW && t.p2_if_x1_one == P_LOW));
        assert!(ts.iter().any(|t| t.p1 == P_HIGH && t.p2_if_x1_zero == P_HIGH && t.p2_if_x1_one == P_HIGH));
        for p1 in [P_LOW, P_HIGH] {
            for p2 in [P_LOW, P_HIGH] {
                assert!(ts.iter().any(|t| t.p1 == p1 && t.p2_if_x1_zero == p2 && t.p2_if_x1_one == p2));
            }
        }
    }

    #[test]
    fn terminal_minimax_test_controls_every_null_member() {
        let ts = all_trajectories();
        let phi = terminal_test();
        let sizes: Vec<Q> = ts.iter().map(|t| expectation(t.joint(), phi)).collect();
        assert!(sizes.iter().all(|x| x.is_le(ALPHA)));
        assert_eq!(sizes.iter().copied().reduce(Q::max).unwrap(), ALPHA);
        assert_eq!(sizes.iter().copied().reduce(Q::min).unwrap(), r(1, 40));
    }

    #[test]
    fn exact_alternative_power_and_dual_certificate_match() {
        let alt = alternative_law();
        let phi = terminal_test();
        let power = expectation(alt, phi);
        let q0 = dual_null_mixture(&all_trajectories());
        let w = r(5, 4);
        assert_eq!(q0, [r(1, 10), r(2, 5), r(2, 5), r(1, 10)]);
        assert!(dual_pointwise_check());
        assert_eq!(dual_objective(), r(1, 8));
        assert_eq!(power, dual_objective());
        for k in 0..4 {
            assert!(alt[k].is_le(w.mul(q0[k])));
        }
    }

    #[test]
    fn all_member_conditional_tests_have_the_expected_values() {
        for t in all_trajectories() {
            let g = member_g1(t);
            assert!(g[0] == r(1, 40) || g[0] == r(1, 10));
            assert!(g[1] == r(1, 40) || g[1] == r(1, 10));
            assert_eq!(t.terminal_size(), expectation(t.joint(), terminal_test()));
        }
    }

    #[test]
    fn robust_lower_envelope_is_exact_and_matches_at_terminal_horizon() {
        let ts = all_trajectories();
        assert_eq!(robust_g0(&ts), r(1, 40));
        assert_eq!(robust_g1(&ts), [r(1, 40), r(1, 40)]);
        assert_eq!(terminal_test(), [Q::zero(), GAMMA, GAMMA, Q::zero()]);
        // At F2 the conditional expectation equals the terminal test itself.
        for phi in terminal_test() {
            assert!(Q::zero().is_le(phi) && phi.is_le(Q::one()));
        }
    }

    #[test]
    fn every_public_deterministic_stopping_rule_is_valid_for_every_null_member() {
        let ts = all_trajectories();
        let g0 = robust_g0(&ts);
        let g1 = robust_g1(&ts);
        for rule in all_stopping_rules() {
            for t in &ts {
                let value = expected_induced_level(*t, g0, g1, rule);
                assert!(value.is_le(ALPHA), "trajectory={} level={}", t.id, value);
            }
        }
    }

    #[test]
    fn selective_stopping_maxima_are_exact() {
        let ts = all_trajectories();
        let g0 = robust_g0(&ts);
        let g1 = robust_g1(&ts);
        let worst_for = |rule| ts.iter()
            .map(|t| expected_induced_level(*t, g0, g1, rule))
            .reduce(Q::max).unwrap();
        assert_eq!(worst_for(StopRule::StopAtZero), r(1, 40));
        assert_eq!(worst_for(StopRule::StopAtOneOnNeither), ALPHA);
        assert_eq!(worst_for(StopRule::StopAtOneOnX1Zero), r(17, 200));
        assert_eq!(worst_for(StopRule::StopAtOneOnX1One), r(17, 200));
        assert_eq!(worst_for(StopRule::StopAtOneOnBoth), r(1, 40));
    }

    #[test]
    fn lower_envelope_is_not_mislabeled_as_memberwise_supermartingale() {
        let t = all_trajectories().into_iter()
            .find(|t| t.p1 == P_LOW && t.p2_if_x1_zero == P_HIGH)
            .unwrap();
        assert_eq!(member_g1(t)[0], r(1, 10));
        assert_eq!(robust_g1(&all_trajectories())[0], r(1, 40));
        assert!(member_g1(t)[0].cmp(robust_g1(&all_trajectories())[0]) == Ordering::Greater);
    }

    #[test]
    fn early_stopping_power_cost_is_separate_from_anytime_size_validity() {
        let alt = alternative_law();
        let phi = terminal_test();
        assert_eq!(expectation(alt, phi), r(1, 8));
        let g0 = robust_g0(&all_trajectories());
        let g1 = robust_g1(&all_trajectories());
        assert_eq!(g0, r(1, 40));
        assert_eq!(g1, [r(1, 40), r(1, 40)]);
        assert_eq!(expected_alt_induced_level(g0, g1, StopRule::StopAtZero), r(1, 40));
        assert_eq!(expected_alt_induced_level(g0, g1, StopRule::StopAtOneOnBoth), r(1, 40));
        assert_eq!(expected_alt_induced_level(g0, g1, StopRule::StopAtOneOnNeither), r(1, 8));
        assert_eq!(expected_alt_induced_level(g0, g1, StopRule::StopAtOneOnX1Zero), r(3, 40));
        assert_eq!(expected_alt_induced_level(g0, g1, StopRule::StopAtOneOnX1One), r(3, 40));
    }

    #[test]
    fn terminal_conditional_expectation_matches_phi_on_every_history() {
        let phi = terminal_test();
        let ts = all_trajectories();
        for t in &ts {
            for (x1, x2, index) in [(false,false,0),(false,true,1),(true,false,2),(true,true,3)] {
                let terminal_conditional = member_g2_at_history(*t, x1, x2);
                assert_eq!(terminal_conditional, phi[index]);
                assert!(terminal_conditional.is_le(Q::one()));
            }
        }
    }
}
