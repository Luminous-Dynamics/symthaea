//! Exact noisy Bernoulli testing and induced anytime-valid process fixture.
//!
//! This is a finite-outcome M0 benchmark, not a physical sensor model.
//! The Bernoulli probabilities represent a frozen thresholded noisy channel.
//! The conditional-expectation induction is the published construction of
//! Koning & van Meer (2026), implemented here as an exact rational regression.
//!
//! Run:
//!   rustc --edition=2024 --test scripts/research/sdcb_noisy_binomial_induction.rs -o /tmp/sdcb-noisy-binomial-tests
//!   /tmp/sdcb-noisy-binomial-tests
//!   rustc --edition=2024 scripts/research/sdcb_noisy_binomial_induction.rs -o /tmp/sdcb-noisy-binomial
//!   /tmp/sdcb-noisy-binomial

use std::cmp::Ordering;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Error {
    InvalidInput,
    Overflow,
    NoPowerWithinSearchLimit,
}

type Exact<T> = Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Q {
    n: i128,
    d: i128,
}

impl Q {
    fn new(n: i128, d: i128) -> Exact<Self> {
        if d == 0 {
            return Err(Error::InvalidInput);
        }
        let (n, d) = if d < 0 {
            (
                n.checked_neg().ok_or(Error::Overflow)?,
                d.checked_neg().ok_or(Error::Overflow)?,
            )
        } else {
            (n, d)
        };
        let abs_n = n.checked_abs().ok_or(Error::Overflow)?;
        let g = gcd(abs_n, d);
        Ok(Self { n: n / g, d: d / g })
    }

    const fn zero() -> Self { Self { n: 0, d: 1 } }
    const fn one() -> Self { Self { n: 1, d: 1 } }

    fn add(self, rhs: Self) -> Exact<Self> {
        let a = self.n.checked_mul(rhs.d).ok_or(Error::Overflow)?;
        let b = rhs.n.checked_mul(self.d).ok_or(Error::Overflow)?;
        let n = a.checked_add(b).ok_or(Error::Overflow)?;
        let d = self.d.checked_mul(rhs.d).ok_or(Error::Overflow)?;
        Self::new(n, d)
    }

    fn sub(self, rhs: Self) -> Exact<Self> {
        let a = self.n.checked_mul(rhs.d).ok_or(Error::Overflow)?;
        let b = rhs.n.checked_mul(self.d).ok_or(Error::Overflow)?;
        let n = a.checked_sub(b).ok_or(Error::Overflow)?;
        let d = self.d.checked_mul(rhs.d).ok_or(Error::Overflow)?;
        Self::new(n, d)
    }

    fn mul(self, rhs: Self) -> Exact<Self> {
        let n = self.n.checked_mul(rhs.n).ok_or(Error::Overflow)?;
        let d = self.d.checked_mul(rhs.d).ok_or(Error::Overflow)?;
        Self::new(n, d)
    }

    fn div(self, rhs: Self) -> Exact<Self> {
        if rhs.n == 0 {
            return Err(Error::InvalidInput);
        }
        let n = self.n.checked_mul(rhs.d).ok_or(Error::Overflow)?;
        let d = self.d.checked_mul(rhs.n).ok_or(Error::Overflow)?;
        Self::new(n, d)
    }

    fn cmp(self, rhs: Self) -> Exact<Ordering> {
        let a = self.n.checked_mul(rhs.d).ok_or(Error::Overflow)?;
        let b = rhs.n.checked_mul(self.d).ok_or(Error::Overflow)?;
        Ok(a.cmp(&b))
    }

    fn is_nonnegative(self) -> bool { self.n >= 0 }
    fn is_zero(self) -> bool { self.n == 0 }
}

impl fmt::Display for Q {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.d == 1 {
            write!(f, "{}", self.n)
        } else {
            write!(f, "{}/{}", self.n, self.d)
        }
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

fn q(n: i128, d: i128) -> Q {
    Q::new(n, d).expect("frozen fixture rational must be representable")
}

fn qpow(base: Q, exponent: usize) -> Exact<Q> {
    let mut value = Q::one();
    for _ in 0..exponent {
        value = value.mul(base)?;
    }
    Ok(value)
}

fn choose(n: usize, k: usize) -> Exact<i128> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut value = 1_i128;
    for i in 0..k {
        let numerator = (n - i) as i128;
        value = value
            .checked_mul(numerator)
            .ok_or(Error::Overflow)?
            / ((i + 1) as i128);
    }
    Ok(value)
}

fn binomial_pmf(n: usize, k: usize, p: Q) -> Exact<Q> {
    if k > n || !p.is_nonnegative() || p.cmp(Q::one())? == Ordering::Greater {
        return Err(Error::InvalidInput);
    }
    let coefficient = Q::new(choose(n, k)?, 1)?;
    let failure = Q::one().sub(p)?;
    coefficient.mul(qpow(p, k)?)?.mul(qpow(failure, n - k)?)
}

fn binomial_law(n: usize, p: Q) -> Exact<Vec<Q>> {
    (0..=n).map(|k| binomial_pmf(n, k, p)).collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FixedTest {
    n: usize,
    null_endpoint: Q,
    alternative_endpoint: Q,
    alpha: Q,
    rejection_probability_by_success_count: Vec<Q>,
    achieved_size: Q,
    achieved_power: Q,
}

/// Exact Neyman–Pearson test for Bernoulli(p0) versus Bernoulli(p1), p1>p0.
/// The MLR ordering is the success count, so the optimizer fills k=n,...,0
/// and randomizes at most one boundary count.
fn fixed_np_test(n: usize, p0: Q, p1: Q, alpha: Q) -> Exact<FixedTest> {
    if p0.cmp(p1)? != Ordering::Less
        || !alpha.is_nonnegative()
        || alpha.cmp(Q::one())? != Ordering::Less
    {
        return Err(Error::InvalidInput);
    }
    let null_law = binomial_law(n, p0)?;
    let alternative_law = binomial_law(n, p1)?;
    let mut remaining = alpha;
    let mut phi = vec![Q::zero(); n + 1];

    for k in (0..=n).rev() {
        if remaining.is_zero() {
            break;
        }
        if remaining.cmp(null_law[k])? != Ordering::Less {
            phi[k] = Q::one();
            remaining = remaining.sub(null_law[k])?;
        } else {
            phi[k] = remaining.div(null_law[k])?;
            remaining = Q::zero();
        }
    }
    if !remaining.is_zero() {
        return Err(Error::InvalidInput);
    }

    let mut size = Q::zero();
    let mut power = Q::zero();
    for k in 0..=n {
        size = size.add(null_law[k].mul(phi[k])?)?;
        power = power.add(alternative_law[k].mul(phi[k])?)?;
    }

    Ok(FixedTest {
        n,
        null_endpoint: p0,
        alternative_endpoint: p1,
        alpha,
        rejection_probability_by_success_count: phi,
        achieved_size: size,
        achieved_power: power,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FrontierPoint {
    epsilon: Q,
    null_endpoint: Q,
    alternative_endpoint: Q,
    minimum_n: usize,
    achieved_power: Q,
    predecessor_power: Q,
    test: FixedTest,
}

/// TV(Bernoulli(p), Bernoulli(q))=|p-q|. For the ordered interval families,
/// a nondecreasing count test has worst-null endpoint p0+epsilon and
/// worst-alternative endpoint p1-epsilon.
fn minimum_sample_frontier(
    p0_nominal: Q,
    p1_nominal: Q,
    epsilon: Q,
    alpha: Q,
    target_power: Q,
    max_n: usize,
) -> Exact<FrontierPoint> {
    let p0 = p0_nominal.add(epsilon)?;
    let p1 = p1_nominal.sub(epsilon)?;
    if !epsilon.is_nonnegative()
        || p0.cmp(p1)? != Ordering::Less
        || p0.cmp(Q::one())? != Ordering::Less
        || !p1.is_nonnegative()
        || p1.cmp(Q::one())? == Ordering::Greater
    {
        return Err(Error::InvalidInput);
    }

    for n in 1..=max_n {
        let test = fixed_np_test(n, p0, p1, alpha)?;
        if test.achieved_power.cmp(target_power)? != Ordering::Less {
            let predecessor_power = if n == 1 {
                Q::zero()
            } else {
                fixed_np_test(n - 1, p0, p1, alpha)?.achieved_power
            };
            return Ok(FrontierPoint {
                epsilon,
                null_endpoint: p0,
                alternative_endpoint: p1,
                minimum_n: n,
                achieved_power: test.achieved_power,
                predecessor_power,
                test,
            });
        }
    }
    Err(Error::NoPowerWithinSearchLimit)
}

/// g_n(s)=E_{p0}[phi_N(S_N) | S_n=s], the terminal-test induction of
/// Koning & van Meer (2026). Rows are n=0..N; row n has n+1 histories.
fn induced_process(terminal: &FixedTest) -> Exact<Vec<Vec<Q>>> {
    let n_max = terminal.n;
    let p0 = terminal.null_endpoint;
    let mut rows = Vec::with_capacity(n_max + 1);
    for n in 0..=n_max {
        let remaining = n_max - n;
        let future_law = binomial_law(remaining, p0)?;
        let mut row = Vec::with_capacity(n + 1);
        for observed_successes in 0..=n {
            let mut value = Q::zero();
            for future_successes in 0..=remaining {
                let terminal_successes = observed_successes + future_successes;
                value = value.add(
                    future_law[future_successes]
                        .mul(terminal.rejection_probability_by_success_count[terminal_successes])?
                )?;
            }
            row.push(value);
        }
        rows.push(row);
    }
    Ok(rows)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InductionCheck {
    initial_value: Q,
    terminal_values_match: bool,
    martingale_equalities: usize,
    monotone_history_inequalities: usize,
    null_endpoint_supermartingale_equalities: usize,
    lower_null_supermartingale_inequalities: usize,
}

fn verify_induced_process(terminal: &FixedTest) -> Exact<(Vec<Vec<Q>>, InductionCheck)> {
    let rows = induced_process(terminal)?;
    let n_max = terminal.n;
    let p0 = terminal.null_endpoint;
    let p0_complement = Q::one().sub(p0)?;
    let initial_value = rows[0][0];

    if initial_value != terminal.alpha {
        return Err(Error::InvalidInput);
    }

    let mut terminal_values_match = true;
    for s in 0..=n_max {
        terminal_values_match &= rows[n_max][s] == terminal.rejection_probability_by_success_count[s];
        if s > 0
            && terminal.rejection_probability_by_success_count[s]
                .cmp(terminal.rejection_probability_by_success_count[s - 1])? == Ordering::Less
        {
            return Err(Error::InvalidInput);
        }
    }
    if !terminal_values_match {
        return Err(Error::InvalidInput);
    }

    let mut martingale_equalities = 0;
    let mut monotone_history_inequalities = 0;
    let mut null_endpoint_supermartingale_equalities = 0;
    let mut lower_null_supermartingale_inequalities = 0;

    for n in 0..n_max {
        for s in 0..=n {
            let g0 = rows[n][s];
            let g_next_0 = rows[n + 1][s];
            let g_next_1 = rows[n + 1][s + 1];

            // Conditional expectation at the least-favourable null endpoint.
            let endpoint_conditional = p0_complement
                .mul(g_next_0)?
                .add(p0.mul(g_next_1)?)?;
            if endpoint_conditional != g0 {
                return Err(Error::InvalidInput);
            }
            martingale_equalities += 1;

            // The terminal test is increasing in the success count, and so
            // is its conditional expectation at every intermediate history.
            if g_next_1.cmp(g_next_0)? == Ordering::Less {
                return Err(Error::InvalidInput);
            }
            monotone_history_inequalities += 1;

            // Check p=0 as a strict lower-null endpoint. For any p<=p0,
            // monotonicity in the next Bernoulli outcome implies the
            // conditional expectation is no larger than at p0.
            let p_zero_conditional = g_next_0;
            if p_zero_conditional.cmp(g0)? != Ordering::Greater {
                lower_null_supermartingale_inequalities += 1;
            } else {
                return Err(Error::InvalidInput);
            }

            // Boundary equality is also the supermartingale inequality at p0.
            if endpoint_conditional.cmp(g0)? != Ordering::Greater {
                null_endpoint_supermartingale_equalities += 1;
            } else {
                return Err(Error::InvalidInput);
            }
        }
    }

    Ok((rows, InductionCheck {
        initial_value,
        terminal_values_match,
        martingale_equalities,
        monotone_history_inequalities,
        null_endpoint_supermartingale_equalities,
        lower_null_supermartingale_inequalities,
    }))
}

fn overlap_control_power(alpha: Q) -> Q {
    // If P0=P1, every level-alpha test has power <=alpha; phi(y)=alpha attains it.
    alpha
}

fn main() {
    let alpha = q(1, 20);
    let target = q(4, 5);
    let p0_nominal = q(1, 4);
    let p1_nominal = q(3, 4);
    let epsilons = [q(0, 1), q(1, 20), q(1, 12)];

    println!("fixture=sdcb_noisy_binomial_induction_v1");
    println!("alpha={alpha}; target_power={target}");
    println!("claim_scope=synthetic_thresholded_bernoulli_channel;not_physical_sensor_data");
    println!("overlap_control_maximin_power={}", overlap_control_power(alpha));

    for epsilon in epsilons {
        match minimum_sample_frontier(p0_nominal, p1_nominal, epsilon, alpha, target, 20) {
            Ok(point) => {
                let (_, check) = verify_induced_process(&point.test)
                    .expect("the induced process must validate");
                println!(
                    "epsilon={};p0_worst={};p1_worst={};minimum_n={};predecessor_power={};power={};size={};g0={};induced_steps={};terminal_match={}",
                    point.epsilon,
                    point.null_endpoint,
                    point.alternative_endpoint,
                    point.minimum_n,
                    point.predecessor_power,
                    point.achieved_power,
                    point.test.achieved_size,
                    check.initial_value,
                    check.martingale_equalities,
                    check.terminal_values_match
                );
            }
            Err(error) => println!("epsilon={epsilon};status=UNRESOLVED:{error:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALPHA: Q = Q { n: 1, d: 20 };
    const TARGET: Q = Q { n: 4, d: 5 };
    const P0: Q = Q { n: 1, d: 4 };
    const P1: Q = Q { n: 3, d: 4 };

    #[test]
    fn nominal_noisy_channel_reaches_eighty_percent_at_six_samples() {
        let point = minimum_sample_frontier(P0, P1, Q::zero(), ALPHA, TARGET, 20).unwrap();
        assert_eq!(point.minimum_n, 6);
        assert_eq!(point.achieved_power, q(1079, 1280));
        assert_eq!(point.test.achieved_size, ALPHA);
        assert_eq!(point.predecessor_power, q(471, 640));
        assert!(point.predecessor_power.cmp(TARGET).unwrap() == Ordering::Less);
    }

    #[test]
    fn tv_radius_one_twentieth_moves_minimum_sample_size_to_ten() {
        let point = minimum_sample_frontier(P0, P1, q(1, 20), ALPHA, TARGET, 20).unwrap();
        assert_eq!(point.null_endpoint, q(3, 10));
        assert_eq!(point.alternative_endpoint, q(7, 10));
        assert_eq!(point.minimum_n, 10);
        assert_eq!(point.achieved_power, q(21_309_567, 25_000_000));
        assert_eq!(point.test.achieved_size, ALPHA);
        assert_eq!(point.predecessor_power, q(11_809_567, 15_000_000));
    }

    #[test]
    fn tv_radius_one_twelfth_moves_minimum_sample_size_to_fourteen() {
        let point = minimum_sample_frontier(P0, P1, q(1, 12), ALPHA, TARGET, 20).unwrap();
        assert_eq!(point.null_endpoint, q(1, 3));
        assert_eq!(point.alternative_endpoint, q(2, 3));
        assert_eq!(point.minimum_n, 14);
        assert_eq!(point.achieved_power, q(726_367, 885_735));
        assert_eq!(point.test.achieved_size, ALPHA);
        assert_eq!(point.predecessor_power, q(1_398_367, 1_771_470));
    }

    #[test]
    fn induced_process_matches_terminal_test_and_is_a_boundary_martingale() {
        for epsilon in [Q::zero(), q(1, 20), q(1, 12)] {
            let point = minimum_sample_frontier(P0, P1, epsilon, ALPHA, TARGET, 20).unwrap();
            let (rows, check) = verify_induced_process(&point.test).unwrap();
            assert_eq!(check.initial_value, ALPHA);
            assert!(check.terminal_values_match);
            assert_eq!(
                check.martingale_equalities,
                point.minimum_n * (point.minimum_n + 1) / 2
            );
            assert_eq!(
                check.monotone_history_inequalities,
                point.minimum_n * (point.minimum_n + 1) / 2
            );
            assert_eq!(rows[point.minimum_n], point.test.rejection_probability_by_success_count);
        }
    }

    #[test]
    fn lower_null_parameters_make_the_induced_process_a_supermartingale() {
        let point = minimum_sample_frontier(P0, P1, q(1, 20), ALPHA, TARGET, 20).unwrap();
        let (rows, check) = verify_induced_process(&point.test).unwrap();
        assert_eq!(
            check.lower_null_supermartingale_inequalities,
            point.minimum_n * (point.minimum_n + 1) / 2
        );

        // Also check an interior null parameter p=1/5 exactly at every history.
        let p = q(1, 5);
        let complement = Q::one().sub(p).unwrap();
        for n in 0..point.minimum_n {
            for s in 0..=n {
                let conditional = complement.mul(rows[n + 1][s]).unwrap()
                    .add(p.mul(rows[n + 1][s + 1]).unwrap()).unwrap();
                assert_eq!(
                    conditional.cmp(rows[n][s]).unwrap(),
                    Ordering::Less
                );
            }
        }
    }

    #[test]
    fn identical_passive_and_upstream_laws_cannot_exceed_alpha() {
        let common_law = binomial_law(8, q(1, 2)).unwrap();
        let constant_test = vec![ALPHA; common_law.len()];
        let mut size = Q::zero();
        let mut power = Q::zero();
        for (probability, rejection_probability) in common_law.iter().zip(constant_test.iter()) {
            size = size.add(probability.mul(*rejection_probability).unwrap()).unwrap();
            power = power.add(probability.mul(*rejection_probability).unwrap()).unwrap();
        }
        assert_eq!(size, ALPHA);
        assert_eq!(power, ALPHA);
        assert_eq!(overlap_control_power(ALPHA), ALPHA);
    }

    #[test]
    fn malformed_or_nonseparated_inputs_fail_closed() {
        assert_eq!(
            fixed_np_test(6, q(3, 4), q(1, 4), ALPHA),
            Err(Error::InvalidInput)
        );
        assert_eq!(
            minimum_sample_frontier(P0, P1, q(1, 4), ALPHA, TARGET, 20),
            Err(Error::InvalidInput)
        );
    }
}
