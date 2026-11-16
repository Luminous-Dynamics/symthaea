//! Fixed-design Gaussian terminal-test induction reference.
//!
//! Given one frozen simple Gaussian null/alternative and a deterministic
//! observation design, this module induces an anytime-valid [0,1]-valued
//! randomized test process from the fixed-horizon Neyman-Pearson terminal
//! test. It is not a p-value process and is not valid for adaptive future
//! designs unless their remaining conditional likelihood is derived anew.

use std::error::Error;
use std::fmt;

use super::{normal_cdf, normal_quantile, one_sided_power};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GaussianInductionError {
    EmptyDesign,
    MismatchedDesignLengths { discrepancies: usize, sigmas: usize },
    InvalidAlpha,
    InvalidSigma { index: usize },
    NonFiniteDiscrepancy { index: usize },
    NonFiniteObservation { index: usize },
    PrefixTooLong { expected: usize, actual: usize },
    ZeroInformation,
    NumericalOverflow,
    InvalidUniformDraw,
}

impl fmt::Display for GaussianInductionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDesign => write!(f, "fixed Gaussian design must not be empty"),
            Self::MismatchedDesignLengths {
                discrepancies,
                sigmas,
            } => write!(
                f,
                "design has {discrepancies} discrepancies and {sigmas} standard deviations"
            ),
            Self::InvalidAlpha => write!(f, "alpha must be finite and in (1e-12, 1 - 1e-12)"),
            Self::InvalidSigma { index } => {
                write!(f, "sigma at design index {index} must be finite and positive")
            }
            Self::NonFiniteDiscrepancy { index } => {
                write!(f, "discrepancy at design index {index} must be finite")
            }
            Self::NonFiniteObservation { index } => {
                write!(f, "observation at prefix index {index} must be finite")
            }
            Self::PrefixTooLong { expected, actual } => {
                write!(f, "prefix has {actual} observations; maximum is {expected}")
            }
            Self::ZeroInformation => write!(
                f,
                "fixed design has zero information and no non-randomized terminal NP test"
            ),
            Self::NumericalOverflow => write!(f, "Gaussian design information or log likelihood overflowed"),
            Self::InvalidUniformDraw => write!(f, "randomization draw must be finite and in [0, 1)"),
        }
    }
}

impl Error for GaussianInductionError {}

/// A fixed, deterministic, evaluator-declared simple-vs-simple Gaussian design.
///
/// Under H0, Y_t ~ Normal(0, sigma_t^2); under H1, Y_t ~ Normal(d_t, sigma_t^2).
/// Both d_t and sigma_t are frozen before the sequence. Future design settings
/// must not depend on future observations.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianTerminalInducedTest {
    alpha: f64,
    expected_discrepancies: Vec<f64>,
    sigmas: Vec<f64>,
    information_by_prefix: Vec<f64>,
    total_information: f64,
    critical_log_likelihood_ratio: f64,
    terminal_power: f64,
}

/// Conditional probability, under H0, that the frozen terminal NP test rejects
/// given the observed prefix. This is the induced test value phi_t, not a
/// p-value. At a stopping time, compare an independent U~Uniform(0,1) to this
/// value to obtain the corresponding randomized terminal-test decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianInducedPrefixValue {
    pub observations: usize,
    pub log_likelihood_ratio: f64,
    pub remaining_information: f64,
    pub critical_log_likelihood_ratio: f64,
    pub conditional_rejection_probability: f64,
}

impl GaussianInducedPrefixValue {
    /// Draw the randomized terminal-test decision using an independent
    /// uniform variate. The randomizer must not be derived from observations.
    pub fn randomized_reject(self, uniform_draw: f64) -> Result<bool, GaussianInductionError> {
        if !uniform_draw.is_finite() || !(0.0..1.0).contains(&uniform_draw) {
            return Err(GaussianInductionError::InvalidUniformDraw);
        }
        Ok(uniform_draw < self.conditional_rejection_probability)
    }
}

impl GaussianTerminalInducedTest {
    /// Construct the induced test from a frozen deterministic design.
    ///
    /// I = sum_t d_t^2 / sigma_t^2 and the terminal one-sided NP critical
    /// log-likelihood ratio is -I/2 + sqrt(I) * Phi^{-1}(1-alpha).
    pub fn new(
        alpha: f64,
        expected_discrepancies: Vec<f64>,
        sigmas: Vec<f64>,
    ) -> Result<Self, GaussianInductionError> {
        if !alpha.is_finite() || alpha <= 1.0e-12 || alpha >= 1.0 - 1.0e-12 {
            return Err(GaussianInductionError::InvalidAlpha);
        }
        if expected_discrepancies.is_empty() {
            return Err(GaussianInductionError::EmptyDesign);
        }
        if expected_discrepancies.len() != sigmas.len() {
            return Err(GaussianInductionError::MismatchedDesignLengths {
                discrepancies: expected_discrepancies.len(),
                sigmas: sigmas.len(),
            });
        }

        let mut information_by_step = Vec::with_capacity(expected_discrepancies.len());
        for (index, (discrepancy, sigma)) in expected_discrepancies
            .iter()
            .copied()
            .zip(sigmas.iter().copied())
            .enumerate()
        {
            if !discrepancy.is_finite() {
                return Err(GaussianInductionError::NonFiniteDiscrepancy { index });
            }
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(GaussianInductionError::InvalidSigma { index });
            }
            let standardized = discrepancy / sigma;
            let information = standardized * standardized;
            if !information.is_finite() {
                return Err(GaussianInductionError::NumericalOverflow);
            }
            information_by_step.push(information);
        }

        let total_information = information_by_step.iter().sum::<f64>();
        if !total_information.is_finite() {
            return Err(GaussianInductionError::NumericalOverflow);
        }
        if total_information <= 0.0 {
            return Err(GaussianInductionError::ZeroInformation);
        }

        // Suffix information at prefix t: sum_{s>t} d_s^2 / sigma_s^2.
        let n = information_by_step.len();
        let mut information_by_prefix = vec![0.0; n + 1];
        for index in (0..n).rev() {
            information_by_prefix[index] =
                information_by_prefix[index + 1] + information_by_step[index];
        }
        if information_by_prefix.iter().any(|value| !value.is_finite()) {
            return Err(GaussianInductionError::NumericalOverflow);
        }

        let z_critical = normal_quantile(1.0 - alpha);
        let critical_log_likelihood_ratio =
            -0.5 * total_information + total_information.sqrt() * z_critical;
        if !critical_log_likelihood_ratio.is_finite() {
            return Err(GaussianInductionError::NumericalOverflow);
        }

        let terminal_power = one_sided_power(total_information.sqrt(), z_critical);
        if !terminal_power.is_finite() {
            return Err(GaussianInductionError::NumericalOverflow);
        }

        Ok(Self {
            alpha,
            expected_discrepancies,
            sigmas,
            information_by_prefix,
            total_information,
            critical_log_likelihood_ratio,
            terminal_power,
        })
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn horizon(&self) -> usize {
        self.expected_discrepancies.len()
    }

    pub fn total_information(&self) -> f64 {
        self.total_information
    }

    pub fn critical_log_likelihood_ratio(&self) -> f64 {
        self.critical_log_likelihood_ratio
    }

    pub fn terminal_power(&self) -> f64 {
        self.terminal_power
    }

    /// Evaluate the induced test at any observed prefix, including the empty
    /// prefix and the fixed terminal horizon.
    ///
    /// Under H0, future log-likelihood increments are independent Gaussian
    /// variables with total mean -I_remaining/2 and variance I_remaining.
    /// The returned value is their conditional probability of crossing the
    /// frozen terminal NP boundary.
    pub fn evaluate_prefix(
        &self,
        observed_residuals: &[f64],
    ) -> Result<GaussianInducedPrefixValue, GaussianInductionError> {
        if observed_residuals.len() > self.horizon() {
            return Err(GaussianInductionError::PrefixTooLong {
                expected: self.horizon(),
                actual: observed_residuals.len(),
            });
        }

        let mut log_likelihood_ratio = 0.0;
        for (index, observed) in observed_residuals.iter().copied().enumerate() {
            if !observed.is_finite() {
                return Err(GaussianInductionError::NonFiniteObservation { index });
            }
            let discrepancy = self.expected_discrepancies[index];
            let sigma = self.sigmas[index];
            let standardized_discrepancy = discrepancy / sigma;
            let standardized_observation = observed / sigma;
            let increment = standardized_discrepancy * standardized_observation
                - 0.5 * standardized_discrepancy * standardized_discrepancy;
            if !increment.is_finite() {
                return Err(GaussianInductionError::NumericalOverflow);
            }
            log_likelihood_ratio += increment;
            if !log_likelihood_ratio.is_finite() {
                return Err(GaussianInductionError::NumericalOverflow);
            }
        }

        let observations = observed_residuals.len();
        let remaining_information = self.information_by_prefix[observations];
        let conditional_rejection_probability = if remaining_information <= 0.0 {
            if log_likelihood_ratio >= self.critical_log_likelihood_ratio {
                1.0
            } else {
                0.0
            }
        } else {
            let standardized_margin = (log_likelihood_ratio
                - self.critical_log_likelihood_ratio
                - 0.5 * remaining_information)
                / remaining_information.sqrt();
            normal_cdf(standardized_margin).clamp(0.0, 1.0)
        };

        Ok(GaussianInducedPrefixValue {
            observations,
            log_likelihood_ratio,
            remaining_information,
            critical_log_likelihood_ratio: self.critical_log_likelihood_ratio,
            conditional_rejection_probability,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_standard_normal(state: &mut u64) -> f64 {
        let mut next = || {
            *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = *state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            ((value >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
        };
        let u1 = next();
        let u2 = next();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    #[test]
    fn empty_prefix_test_value_equals_nominal_alpha() {
        let test =
            GaussianTerminalInducedTest::new(0.05, vec![0.24; 3], vec![0.15; 3]).unwrap();
        let prefix = test.evaluate_prefix(&[]).unwrap();
        assert!((prefix.conditional_rejection_probability - 0.05).abs() < 2.0e-7);
        assert_eq!(prefix.observations, 0);
        assert!((prefix.remaining_information - test.total_information()).abs() < 1.0e-12);
    }

    #[test]
    fn terminal_prefix_matches_the_fixed_n_p_test_exactly() {
        let test =
            GaussianTerminalInducedTest::new(0.05, vec![0.24; 3], vec![0.15; 3]).unwrap();
        let above = test.evaluate_prefix(&[0.2, 0.2, 0.2]).unwrap();
        let below = test.evaluate_prefix(&[-0.2, -0.2, -0.2]).unwrap();
        assert_eq!(
            above.conditional_rejection_probability,
            if above.log_likelihood_ratio >= test.critical_log_likelihood_ratio() {
                1.0
            } else {
                0.0
            }
        );
        assert_eq!(
            below.conditional_rejection_probability,
            if below.log_likelihood_ratio >= test.critical_log_likelihood_ratio() {
                1.0
            } else {
                0.0
            }
        );
    }

    #[test]
    fn terminal_power_matches_the_known_direction_gaussian_reference() {
        let test =
            GaussianTerminalInducedTest::new(0.05, vec![0.24; 3], vec![0.15; 3]).unwrap();
        let power = super::super::known_direction_power(&[0.24; 3], 0.15, 0.05).unwrap();
        assert!((test.terminal_power() - power.power).abs() < 2.0e-7);
    }

    #[test]
    fn fixed_design_conditional_values_satisfy_a_prefix_martingale_check() {
        let test =
            GaussianTerminalInducedTest::new(0.05, vec![0.2; 3], vec![0.5; 3]).unwrap();
        let prefix = test.evaluate_prefix(&[0.15]).unwrap();
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut total = 0.0;
        const SAMPLES: usize = 80_000;
        for _ in 0..SAMPLES {
            let y2 = 0.5 * deterministic_standard_normal(&mut state);
            let y3 = 0.5 * deterministic_standard_normal(&mut state);
            let value = test.evaluate_prefix(&[0.15, y2, y3]).unwrap();
            total += value.conditional_rejection_probability;
        }
        let empirical_conditional_mean = total / SAMPLES as f64;
        assert!(
            (empirical_conditional_mean - prefix.conditional_rejection_probability).abs() < 0.006,
            "conditional mean {empirical_conditional_mean:.5} differs from prefix value {:.5}",
            prefix.conditional_rejection_probability
        );
    }

    #[test]
    fn optional_stopping_randomized_test_retains_null_level_in_smoke_simulation() {
        const PATHS: usize = 40_000;
        const HORIZON: usize = 12;
        let discrepancies = vec![0.22; HORIZON];
        let sigmas = vec![0.8; HORIZON];
        let test = GaussianTerminalInducedTest::new(0.05, discrepancies, sigmas).unwrap();
        let mut observation_state = 0xa5a5_1234_9876_fedc_u64;
        let mut randomizer_state = 0x4f1b_bcdd_6a09_e667_u64;
        let mut rejections = 0_usize;

        for _ in 0..PATHS {
            let mut observations = Vec::with_capacity(HORIZON);
            let mut stopped = None;
            for look in 1..=HORIZON {
                let y = 0.8 * deterministic_standard_normal(&mut observation_state);
                observations.push(y);
                let prefix = test.evaluate_prefix(&observations).unwrap();
                if prefix.conditional_rejection_probability >= 0.90 || look == HORIZON {
                    stopped = Some(prefix);
                    break;
                }
            }
            let prefix = stopped.expect("the fixed terminal horizon always stops");
            let randomizer = {
                // A separately seeded simulator stream supplies the
                // independent terminal-test randomization draw.
                randomizer_state =
                    randomizer_state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut value = randomizer_state;
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                value ^= value >> 31;
                ((value >> 11) as f64) / 9_007_199_254_740_992.0
            };
            if prefix.randomized_reject(randomizer).unwrap() {
                rejections += 1;
            }
        }

        let rate = rejections as f64 / PATHS as f64;
        assert!(
            rate <= 0.058,
            "empirical stopped-test size {rate:.4} exceeded 0.058"
        );
    }

    #[test]
    fn randomized_decision_uses_an_independent_uniform_draw() {
        let test = GaussianTerminalInducedTest::new(0.05, vec![0.2], vec![0.5]).unwrap();
        let prefix = test.evaluate_prefix(&[]).unwrap();
        assert!(!prefix.randomized_reject(0.99).unwrap());
        assert!(prefix.randomized_reject(0.01).unwrap());
        assert_eq!(
            prefix.randomized_reject(1.0),
            Err(GaussianInductionError::InvalidUniformDraw)
        );
    }

    #[test]
    fn malformed_designs_and_prefixes_fail_closed() {
        assert_eq!(
            GaussianTerminalInducedTest::new(0.05, vec![], vec![]).err(),
            Some(GaussianInductionError::EmptyDesign)
        );
        assert_eq!(
            GaussianTerminalInducedTest::new(0.05, vec![0.1], vec![]).err(),
            Some(GaussianInductionError::MismatchedDesignLengths {
                discrepancies: 1,
                sigmas: 0
            })
        );
        assert_eq!(
            GaussianTerminalInducedTest::new(0.05, vec![0.0, 0.0], vec![1.0, 1.0]).err(),
            Some(GaussianInductionError::ZeroInformation)
        );
        assert!(matches!(
            GaussianTerminalInducedTest::new(0.05, vec![0.1], vec![0.0]),
            Err(GaussianInductionError::InvalidSigma { index: 0 })
        ));
        let test = GaussianTerminalInducedTest::new(0.05, vec![0.1], vec![0.2]).unwrap();
        assert!(matches!(
            test.evaluate_prefix(&[0.1, 0.2]),
            Err(GaussianInductionError::PrefixTooLong { .. })
        ));
        assert!(matches!(
            test.evaluate_prefix(&[f64::NAN]),
            Err(GaussianInductionError::NonFiniteObservation { index: 0 })
        ));
    }
}
