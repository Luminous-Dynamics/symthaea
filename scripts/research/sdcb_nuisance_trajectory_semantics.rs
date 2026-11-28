//! Exact rational comparison of persistent, rectangular and
//! history-adaptive nuisance trajectory families.
//!
//! Two-stage binary observations. The per-stage nuisance set is {1/5,4/5}.
//! Terminal randomized test rejects with probability 5/16 iff X1 != X2.
//! alpha=1/10.
//!
//! PERSISTENT_SHARED: one p is fixed for both stages. Size = 1/10.
//! RECTANGULAR_PRECOMMITTED: p1,p2 can differ but are fixed before samples.
//! Worst size = 17/80.
//! RECTANGULAR_HISTORY_ADAPTIVE: p2 can depend on observed X1.
//! Enumerating all deterministic selectors gives worst size = 1/4.
//!
//! This is an exact two-stage fixture, not a model of any physical sensor.

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
    fn cmp(self, rhs: Self) -> std::cmp::Ordering {
        (self.n * rhs.d).cmp(&(rhs.n * self.d))
    }
    fn max(self, rhs: Self) -> Self {
        if self.cmp(rhs) == std::cmp::Ordering::Less { rhs } else { self }
    }
    fn min(self, rhs: Self) -> Self {
        if self.cmp(rhs) == std::cmp::Ordering::Greater { rhs } else { self }
    }
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

const P_LOW: Q = Q { n: 1, d: 5 };
const P_HIGH: Q = Q { n: 4, d: 5 };
const ALPHA: Q = Q { n: 1, d: 10 };
const TERMINAL_RANDOMIZATION: Q = Q { n: 5, d: 16 };

fn complement(p: Q) -> Q { Q::one().sub(p) }

fn discord_probability_precommitted(p1: Q, p2: Q) -> Q {
    p1.mul(complement(p2)).add(complement(p1).mul(p2))
}

fn rejection_level(discord_probability: Q, terminal_randomization: Q) -> Q {
    discord_probability.mul(terminal_randomization)
}

fn persistent_discord_probability(p: Q) -> Q {
    discord_probability_precommitted(p, p)
}

fn persistent_worst_discord_probability() -> Q {
    persistent_discord_probability(P_LOW).max(persistent_discord_probability(P_HIGH))
}

fn rectangular_precommitted_worst_discord_probability() -> Q {
    let parameters = [P_LOW, P_HIGH];
    let mut worst = Q::zero();
    for p1 in parameters {
        for p2 in parameters {
            worst = worst.max(discord_probability_precommitted(p1, p2));
        }
    }
    worst
}

/// A deterministic non-anticipating policy chooses p2 after observing X1.
/// p2_if_zero is used on X1=0; p2_if_one is used on X1=1.
fn history_adaptive_discord_probability(p1: Q, p2_if_zero: Q, p2_if_one: Q) -> Q {
    complement(p1).mul(p2_if_zero).add(p1.mul(complement(p2_if_one)))
}

fn history_adaptive_worst_discord_probability() -> Q {
    let parameters = [P_LOW, P_HIGH];
    let mut worst = Q::zero();
    for p1 in parameters {
        for p2_if_zero in parameters {
            for p2_if_one in parameters {
                worst = worst.max(history_adaptive_discord_probability(
                    p1, p2_if_zero, p2_if_one,
                ));
            }
        }
    }
    worst
}

fn valid_randomization_cap(max_discord_probability: Q) -> Q {
    ALPHA.div(max_discord_probability)
}

fn run_fixture() {
    let persistent_discord = persistent_worst_discord_probability();
    let rectangular_discord = rectangular_precommitted_worst_discord_probability();
    let adaptive_discord = history_adaptive_worst_discord_probability();
    let persistent_level = rejection_level(persistent_discord, TERMINAL_RANDOMIZATION);
    let rectangular_level = rejection_level(rectangular_discord, TERMINAL_RANDOMIZATION);
    let adaptive_level = rejection_level(adaptive_discord, TERMINAL_RANDOMIZATION);
    let rectangular_cap = valid_randomization_cap(rectangular_discord);
    let adaptive_cap = valid_randomization_cap(adaptive_discord);

    assert_eq!(persistent_discord, Q::new(8, 25));
    assert_eq!(rectangular_discord, Q::new(17, 25));
    assert_eq!(adaptive_discord, Q::new(4, 5));
    assert_eq!(persistent_level, ALPHA);
    assert_eq!(rectangular_level, Q::new(17, 80));
    assert_eq!(adaptive_level, Q::new(1, 4));
    assert_eq!(rectangular_cap, Q::new(5, 34));
    assert_eq!(adaptive_cap, Q::new(1, 8));

    println!("status=EXACT_NUISANCE_TRAJECTORY_SEMANTICS_FIXTURE");
    println!("alpha={ALPHA}");
    println!("terminal_randomization={TERMINAL_RANDOMIZATION}");
    println!("persistent_worst_discord={persistent_discord}");
    println!("persistent_size={persistent_level}");
    println!("rectangular_precommitted_worst_discord={rectangular_discord}");
    println!("rectangular_precommitted_size={rectangular_level}");
    println!("rectangular_precommitted_randomization_cap={rectangular_cap}");
    println!("history_adaptive_worst_discord={adaptive_discord}");
    println!("history_adaptive_size={adaptive_level}");
    println!("history_adaptive_randomization_cap={adaptive_cap}");
}

fn main() { run_fixture(); }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_shared_nuisance_has_exact_size_alpha() {
        assert_eq!(persistent_discord_probability(P_LOW), Q::new(8, 25));
        assert_eq!(persistent_discord_probability(P_HIGH), Q::new(8, 25));
        assert_eq!(
            rejection_level(persistent_worst_discord_probability(), TERMINAL_RANDOMIZATION),
            ALPHA
        );
    }

    #[test]
    fn rectangular_precommitted_family_has_worst_level_seventeen_over_eighty() {
        assert_eq!(rectangular_precommitted_worst_discord_probability(), Q::new(17, 25));
        assert_eq!(
            rejection_level(
                rectangular_precommitted_worst_discord_probability(),
                TERMINAL_RANDOMIZATION
            ),
            Q::new(17, 80)
        );
        assert!(ALPHA.cmp(rejection_level(
            rectangular_precommitted_worst_discord_probability(),
            TERMINAL_RANDOMIZATION
        )) == std::cmp::Ordering::Less);
    }

    #[test]
    fn history_adaptive_family_has_worst_level_one_quarter() {
        let worst = history_adaptive_worst_discord_probability();
        assert_eq!(worst, Q::new(4, 5));
        assert_eq!(rejection_level(worst, TERMINAL_RANDOMIZATION), Q::new(1, 4));
    }

    #[test]
    fn all_eight_predictable_second_stage_selectors_are_enumerated() {
        let parameters = [P_LOW, P_HIGH];
        let mut seen = 0;
        let mut worst = Q::zero();
        for p1 in parameters {
            for p2_if_zero in parameters {
                for p2_if_one in parameters {
                    let probability = history_adaptive_discord_probability(
                        p1, p2_if_zero, p2_if_one,
                    );
                    worst = worst.max(probability);
                    seen += 1;
                }
            }
        }
        assert_eq!(seen, 8);
        assert_eq!(worst, Q::new(4, 5));
    }

    #[test]
    fn exact_randomization_caps_restore_alpha() {
        let rectangular_cap = valid_randomization_cap(
            rectangular_precommitted_worst_discord_probability()
        );
        let adaptive_cap = valid_randomization_cap(
            history_adaptive_worst_discord_probability()
        );
        assert_eq!(rectangular_cap, Q::new(5, 34));
        assert_eq!(adaptive_cap, Q::new(1, 8));
        assert_eq!(
            rejection_level(
                rectangular_precommitted_worst_discord_probability(),
                rectangular_cap
            ),
            ALPHA
        );
        assert_eq!(
            rejection_level(history_adaptive_worst_discord_probability(), adaptive_cap),
            ALPHA
        );
    }

    #[test]
    fn adaptive_selector_is_nonanticipating_and_history_conditioned() {
        let p1 = P_LOW;
        let p2_if_zero = P_HIGH;
        let p2_if_one = P_LOW;
        assert_eq!(
            history_adaptive_discord_probability(p1, p2_if_zero, p2_if_one),
            Q::new(4, 5)
        );
    }

    #[test]
    fn iid_and_rectangular_families_are_not_interchangeable() {
        let persistent = persistent_worst_discord_probability();
        let precommitted = rectangular_precommitted_worst_discord_probability();
        let adaptive = history_adaptive_worst_discord_probability();
        assert!(persistent.cmp(precommitted) == std::cmp::Ordering::Less);
        assert!(precommitted.cmp(adaptive) == std::cmp::Ordering::Less);
    }

    #[test]
    fn invalid_randomization_cap_cannot_exceed_one() {
        assert!(valid_randomization_cap(persistent_worst_discord_probability())
            .cmp(Q::one()) != std::cmp::Ordering::Greater);
        assert!(valid_randomization_cap(rectangular_precommitted_worst_discord_probability())
            .cmp(Q::one()) != std::cmp::Ordering::Greater);
        assert!(valid_randomization_cap(history_adaptive_worst_discord_probability())
            .cmp(Q::one()) != std::cmp::Ordering::Greater);
    }
}
