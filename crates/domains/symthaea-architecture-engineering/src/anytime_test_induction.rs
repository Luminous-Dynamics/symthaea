//! Fixed-horizon Bernoulli tests induced into an anytime-valid sequence.
//!
//! This is a finite, simple-null reference inspired by test induction: construct a
//! randomized terminal Neyman-Pearson test, then take its conditional expectation
//! under the null at every prefix. The resulting [0,1]-valued process is a bounded
//! null martingale, has the same terminal test at horizon N, and is valid under
//! bounded optional stopping. It is not a general confidence sequence or a composite
//! null guarantee for arbitrary bounded objective observations.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{error::Error, fmt};

const PLAN_SCHEMA: &str = "symthaea.architecture-engineering.induced-bernoulli-test-plan";
const PLAN_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.induced-bernoulli-test-report";
const REPORT_VERSION: &str = "0.1.0";
const MAX_HORIZON: usize = 512;
const PROBABILITY_TOLERANCE: f64 = 1e-10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InducedTestError { pub path: String, pub message: String }
impl InducedTestError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}
impl fmt::Display for InducedTestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}: {}", self.path, self.message) }
}
impl Error for InducedTestError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InducedBernoulliTestPlan {
    pub schema: String,
    pub version: String,
    pub test_id: String,
    pub null_probability: f64,
    pub alternative_probability: f64,
    pub terminal_horizon: usize,
    pub significance_level: f64,
}

impl InducedBernoulliTestPlan {
    pub fn new(
        test_id: impl Into<String>,
        null_probability: f64,
        alternative_probability: f64,
        terminal_horizon: usize,
        significance_level: f64,
    ) -> Result<Self, InducedTestError> {
        let plan = Self {
            schema: PLAN_SCHEMA.into(), version: PLAN_VERSION.into(),
            test_id: test_id.into(), null_probability, alternative_probability,
            terminal_horizon, significance_level,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), InducedTestError> {
        if self.schema != PLAN_SCHEMA || self.version != PLAN_VERSION {
            return Err(InducedTestError::new("plan.schema", "unsupported induced Bernoulli test schema/version"));
        }
        if self.test_id.trim().is_empty() {
            return Err(InducedTestError::new("plan.test_id", "test ID must be non-empty"));
        }
        if !self.null_probability.is_finite() || !(0.0..1.0).contains(&self.null_probability) {
            return Err(InducedTestError::new("plan.null_probability", "simple null probability must lie strictly between zero and one"));
        }
        if !self.alternative_probability.is_finite()
            || !(self.null_probability..=1.0).contains(&self.alternative_probability)
            || self.alternative_probability <= self.null_probability
        {
            return Err(InducedTestError::new("plan.alternative_probability", "one-sided alternative must be strictly greater than the null and at most one"));
        }
        if self.terminal_horizon == 0 || self.terminal_horizon > MAX_HORIZON {
            return Err(InducedTestError::new("plan.terminal_horizon", format!("horizon must be in 1..={MAX_HORIZON}")));
        }
        if !self.significance_level.is_finite() || !(0.0..1.0).contains(&self.significance_level) {
            return Err(InducedTestError::new("plan.significance_level", "alpha must be finite and strictly between zero and one"));
        }
        Ok(())
    }

    pub fn plan_sha256(&self) -> Result<String, InducedTestError> {
        self.validate()?;
        digest(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalRandomizedTest {
    pub horizon: usize,
    /// Reject with probability one when terminal successes exceed this count.
    pub critical_successes: usize,
    /// Randomized rejection probability at exactly critical_successes.
    pub boundary_rejection_probability: f64,
    /// Complete terminal rejection function indexed by total successes 0..=N.
    pub rejection_probability_by_successes: Vec<f64>,
    pub null_size: f64,
    pub alternative_power: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InducedPrefixTestValue {
    pub observations: usize,
    pub successes: usize,
    /// Conditional null expectation of the fixed-horizon terminal test.
    pub induced_test_value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InducedBernoulliTestReport {
    pub schema: String,
    pub version: String,
    pub test_id: String,
    pub plan_sha256: String,
    pub terminal_test: TerminalRandomizedTest,
    pub prefix_values: Vec<InducedPrefixTestValue>,
    pub initial_test_value: f64,
    /// Maximum error in E_0[phi_(n+1) | F_n] = phi_n over every reachable prefix.
    pub max_null_martingale_residual: f64,
    pub terminal_equivalence_max_abs_error: f64,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    test_id: &'a str,
    plan_sha256: &'a str,
    terminal_test: &'a TerminalRandomizedTest,
    prefix_values: &'a [InducedPrefixTestValue],
    initial_test_value: f64,
    max_null_martingale_residual: f64,
    terminal_equivalence_max_abs_error: f64,
}

pub fn analyze_induced_bernoulli_test(
    plan: &InducedBernoulliTestPlan,
) -> Result<InducedBernoulliTestReport, InducedTestError> {
    plan.validate()?;
    let terminal_test = build_terminal_test(plan)?;
    let prefix_values = build_prefix_values(plan, &terminal_test)?;
    let initial_test_value = prefix_value(&prefix_values, 0, 0)?;
    let max_null_martingale_residual = martingale_residual(plan, &prefix_values)?;
    let terminal_equivalence_max_abs_error = terminal_equivalence_error(plan, &terminal_test, &prefix_values)?;
    let plan_sha256 = plan.plan_sha256()?;
    let payload = ReportPayload {
        schema: REPORT_SCHEMA, version: REPORT_VERSION, test_id: &plan.test_id,
        plan_sha256: &plan_sha256, terminal_test: &terminal_test,
        prefix_values: &prefix_values, initial_test_value,
        max_null_martingale_residual, terminal_equivalence_max_abs_error,
    };
    let report_sha256 = digest(&payload)?;
    Ok(InducedBernoulliTestReport {
        schema: REPORT_SCHEMA.into(), version: REPORT_VERSION.into(),
        test_id: plan.test_id.clone(), plan_sha256, terminal_test, prefix_values,
        initial_test_value, max_null_martingale_residual,
        terminal_equivalence_max_abs_error, report_sha256,
    })
}

pub fn verify_induced_bernoulli_test_report(
    plan: &InducedBernoulliTestPlan,
    report: &InducedBernoulliTestReport,
) -> Result<(), InducedTestError> {
    if analyze_induced_bernoulli_test(plan)? != *report {
        return Err(InducedTestError::new("report", "stored induced test differs from exact finite reconstruction"));
    }
    Ok(())
}

fn build_terminal_test(plan: &InducedBernoulliTestPlan) -> Result<TerminalRandomizedTest, InducedTestError> {
    let n = plan.terminal_horizon;
    let null_pmf = binomial_pmf(n, plan.null_probability)?;
    let alternative_pmf = binomial_pmf(n, plan.alternative_probability)?;
    let alpha = plan.significance_level;
    let mut strict_upper_tail = 0.0;
    let mut critical = None;
    let mut boundary_probability = 0.0;

    // Traverse from high to low successes. The likelihood ratio is monotone in
    // the sufficient statistic for p1 > p0, so the NP rejection region is an upper tail.
    for successes in (0..=n).rev() {
        let at_boundary = null_pmf[successes];
        if strict_upper_tail <= alpha + PROBABILITY_TOLERANCE
            && strict_upper_tail + at_boundary >= alpha - PROBABILITY_TOLERANCE
        {
            if at_boundary <= 0.0 {
                return Err(InducedTestError::new("terminal_test", "critical binomial mass underflowed to zero"));
            }
            critical = Some(successes);
            boundary_probability = ((alpha - strict_upper_tail) / at_boundary).clamp(0.0, 1.0);
            break;
        }
        strict_upper_tail += at_boundary;
    }
    let critical_successes = critical.ok_or_else(|| {
        InducedTestError::new("terminal_test", "could not locate a randomized NP critical boundary")
    })?;
    let rejection_probability_by_successes: Vec<f64> = (0..=n)
        .map(|successes| {
            if successes > critical_successes { 1.0 }
            else if successes == critical_successes { boundary_probability }
            else { 0.0 }
        }).collect();
    let null_size = compensated_dot(&null_pmf, &rejection_probability_by_successes);
    let alternative_power = compensated_dot(&alternative_pmf, &rejection_probability_by_successes);
    if !null_size.is_finite() || !alternative_power.is_finite()
        || (null_size - alpha).abs() > 1e-8
        || !(0.0..=1.0).contains(&alternative_power)
    {
        return Err(InducedTestError::new("terminal_test", "terminal test size/power reconstruction failed"));
    }
    Ok(TerminalRandomizedTest {
        horizon: n, critical_successes, boundary_rejection_probability: boundary_probability,
        rejection_probability_by_successes, null_size, alternative_power,
    })
}

fn build_prefix_values(
    plan: &InducedBernoulliTestPlan,
    terminal: &TerminalRandomizedTest,
) -> Result<Vec<InducedPrefixTestValue>, InducedTestError> {
    let horizon = plan.terminal_horizon;
    let mut values = Vec::with_capacity((horizon + 1) * (horizon + 2) / 2);
    for observations in 0..=horizon {
        let remaining = horizon - observations;
        let continuation_pmf = binomial_pmf(remaining, plan.null_probability)?;
        for successes in 0..=observations {
            let mut induced = 0.0;
            let mut compensation = 0.0;
            for (future_successes, probability) in continuation_pmf.iter().enumerate() {
                let terminal_successes = successes + future_successes;
                kahan_add(
                    &mut induced, &mut compensation,
                    probability * terminal.rejection_probability_by_successes[terminal_successes],
                );
            }
            if !induced.is_finite() || !(0.0..=1.0).contains(&induced) {
                return Err(InducedTestError::new(
                    format!("prefix_values[n={observations},s={successes}]"),
                    "induced test value is non-finite or outside [0,1]",
                ));
            }
            values.push(InducedPrefixTestValue {
                observations, successes, induced_test_value: induced,
            });
        }
    }
    Ok(values)
}

fn martingale_residual(
    plan: &InducedBernoulliTestPlan,
    prefix: &[InducedPrefixTestValue],
) -> Result<f64, InducedTestError> {
    let p = plan.null_probability;
    let mut maximum = 0.0_f64;
    for n in 0..plan.terminal_horizon {
        for successes in 0..=n {
            let current = prefix_value(prefix, n, successes)?;
            let next_failure = prefix_value(prefix, n + 1, successes)?;
            let next_success = prefix_value(prefix, n + 1, successes + 1)?;
            let conditional = (1.0 - p) * next_failure + p * next_success;
            maximum = maximum.max((conditional - current).abs());
        }
    }
    Ok(maximum)
}

fn terminal_equivalence_error(
    plan: &InducedBernoulliTestPlan,
    terminal: &TerminalRandomizedTest,
    prefix: &[InducedPrefixTestValue],
) -> Result<f64, InducedTestError> {
    let n = plan.terminal_horizon;
    let mut maximum = 0.0_f64;
    for successes in 0..=n {
        let induced = prefix_value(prefix, n, successes)?;
        maximum = maximum.max((induced - terminal.rejection_probability_by_successes[successes]).abs());
    }
    Ok(maximum)
}

fn prefix_value(
    values: &[InducedPrefixTestValue],
    observations: usize,
    successes: usize,
) -> Result<f64, InducedTestError> {
    if successes > observations {
        return Err(InducedTestError::new("prefix", "success count exceeds observation count"));
    }
    let index = observations.checked_mul(observations + 1).and_then(|v| v.checked_div(2))
        .and_then(|offset| offset.checked_add(successes))
        .ok_or_else(|| InducedTestError::new("prefix", "prefix index overflow"))?;
    values.get(index).map(|value| value.induced_test_value)
        .ok_or_else(|| InducedTestError::new("prefix", "prefix value is missing"))
}

/// Stable binomial probabilities using log probabilities and log-sum-exp normalization.
fn binomial_pmf(n: usize, p: f64) -> Result<Vec<f64>, InducedTestError> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(InducedTestError::new("binomial.p", "p must be finite and lie in [0, 1]"));
    }
    if p == 0.0 {
        let mut values = vec![0.0; n + 1];
        values[0] = 1.0;
        return Ok(values);
    }
    if p == 1.0 {
        let mut values = vec![0.0; n + 1];
        values[n] = 1.0;
        return Ok(values);
    }
    let log_p = p.ln();
    let log_q = (1.0 - p).ln();
    let mut log_values = Vec::with_capacity(n + 1);
    for k in 0..=n {
        let log_choose = log_binomial_coefficient(n, k);
        log_values.push(log_choose + k as f64 * log_p + (n - k) as f64 * log_q);
    }
    let maximum = log_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_values.iter().map(|value| (value - maximum).exp()).collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(InducedTestError::new("binomial.pmf", "log-sum-exp normalization failed"));
    }
    let values: Vec<f64> = weights.iter().map(|weight| weight / total).collect();
    let sum: f64 = values.iter().sum();
    if (sum - 1.0).abs() > 1e-12 {
        return Err(InducedTestError::new("binomial.pmf", "probabilities do not normalize"));
    }
    Ok(values)
}

fn log_binomial_coefficient(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    (1..=k).map(|index| ((n + 1 - index) as f64 / index as f64).ln()).sum()
}

fn compensated_dot(a: &[f64], b: &[f64]) -> f64 {
    let (mut sum, mut compensation) = (0.0, 0.0);
    for (left, right) in a.iter().zip(b) { kahan_add(&mut sum, &mut compensation, left * right); }
    sum
}
fn kahan_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let adjusted = value - *compensation;
    let next = *sum + adjusted;
    *compensation = (next - *sum) - adjusted;
    *sum = next;
}
fn digest<T: Serialize>(value: &T) -> Result<String, InducedTestError> {
    let bytes = serde_json::to_vec(value).map_err(|error| InducedTestError::new("serialization", error.to_string()))?;
    Ok(Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> InducedBernoulliTestPlan {
        InducedBernoulliTestPlan::new("one-sided-quality-threshold", 0.2, 0.5, 12, 0.05).unwrap()
    }

    fn stopping_value(
        report: &InducedBernoulliTestReport,
        null_p: f64,
        stop_rule: impl Fn(usize, usize) -> bool,
    ) -> f64 {
        let n = report.terminal_test.horizon;
        let mut expectation = 0.0;
        for path_mask in 0..(1usize << n) {
            let mut successes_at_stop = 0usize;
            let mut stop_at = n;
            let mut stopped = false;
            let mut full_path_probability = 1.0;
            for time in 1..=n {
                let success = (path_mask >> (time - 1)) & 1 == 1;
                if success {
                    full_path_probability *= null_p;
                    if !stopped { successes_at_stop += 1; }
                } else {
                    full_path_probability *= 1.0 - null_p;
                }
                if !stopped && stop_rule(time, successes_at_stop) {
                    stop_at = time;
                    stopped = true;
                }
            }
            let value = prefix_value(&report.prefix_values, stop_at, successes_at_stop).unwrap();
            expectation += full_path_probability * value;
        }
        expectation
    }

    #[test]
    fn induced_process_is_bounded_martingale_and_matches_terminal_test() {
        let plan = plan();
        let report = analyze_induced_bernoulli_test(&plan).unwrap();
        assert!((report.terminal_test.null_size - plan.significance_level).abs() < 1e-10);
        assert!((report.initial_test_value - plan.significance_level).abs() < 1e-10);
        assert!(report.max_null_martingale_residual < 1e-10);
        assert!(report.terminal_equivalence_max_abs_error < 1e-12);
        assert!(report.terminal_test.alternative_power > report.terminal_test.null_size);
        verify_induced_bernoulli_test_report(&plan, &report).unwrap();
        assert!(report.prefix_values.iter().all(|v| (0.0..=1.0).contains(&v.induced_test_value)));
    }

    #[test]
    fn optional_stopping_threshold_rules_preserve_the_null_level() {
        let plan = plan();
        let report = analyze_induced_bernoulli_test(&plan).unwrap();
        for success_threshold in 1..=plan.terminal_horizon + 1 {
            for minimum_failure_count in 1..=plan.terminal_horizon + 1 {
                let expected = stopping_value(&report, plan.null_probability, |n, successes| {
                    successes >= success_threshold || n - successes >= minimum_failure_count || n == plan.terminal_horizon
                });
                assert!((expected - plan.significance_level).abs() < 1e-9,
                    "optional stopping changed null expectation: {expected}");
            }
        }
    }

    #[test]
    fn terminal_test_is_randomized_at_the_boundary_to_attain_declared_alpha() {
        let report = analyze_induced_bernoulli_test(&plan()).unwrap();
        assert!((0.0..=1.0).contains(&report.terminal_test.boundary_rejection_probability));
        assert!(report.terminal_test.rejection_probability_by_successes.windows(2)
            .all(|pair| pair[0] <= pair[1]));
        assert!((report.terminal_test.null_size - 0.05).abs() < 1e-10);
    }

    #[test]
    fn invalid_simple_null_alternative_horizon_and_alpha_fail_closed() {
        assert!(InducedBernoulliTestPlan::new("bad",0.0,0.5,12,0.05).is_err());
        assert!(InducedBernoulliTestPlan::new("bad",0.4,0.3,12,0.05).is_err());
        assert!(InducedBernoulliTestPlan::new("bad",0.2,0.5,0,0.05).is_err());
        assert!(InducedBernoulliTestPlan::new("bad",0.2,0.5,MAX_HORIZON+1,0.05).is_err());
        assert!(InducedBernoulliTestPlan::new("bad",0.2,0.5,12,1.0).is_err());
    }

    #[test]
    fn degenerate_alternative_probability_one_is_supported() {
        let plan = InducedBernoulliTestPlan::new("certain-success-alternative", 0.2, 1.0, 8, 0.05).unwrap();
        let report = analyze_induced_bernoulli_test(&plan).unwrap();
        assert!(report.terminal_test.alternative_power > 0.99);
        assert!((report.terminal_test.null_size - 0.05).abs() < 1e-10);
        assert!(report.max_null_martingale_residual < 1e-10);
    }

    #[test]
    fn report_and_plan_identity_are_bound() {
        let p = plan();
        let report = analyze_induced_bernoulli_test(&p).unwrap();
        let mut altered = report.clone();
        altered.prefix_values[0].induced_test_value += 0.01;
        assert!(verify_induced_bernoulli_test_report(&p,&altered).is_err());
        let other = InducedBernoulliTestPlan::new("one-sided-quality-threshold",0.2,0.7,12,0.05).unwrap();
        assert_ne!(p.plan_sha256().unwrap(),other.plan_sha256().unwrap());
    }
}
