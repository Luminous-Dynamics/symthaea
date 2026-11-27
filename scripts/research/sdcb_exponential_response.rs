//! Standalone analytic reference for a first-order instrument response driven by
//! a stationary scalar Ornstein-Uhlenbeck measurand.
//!
//! State ordering is [x, z], where
//!   dx = -lambda * x dt + sqrt(2 * lambda * process_variance) dW
//!   dz = (x - z) / response_time_constant dt.
//!
//! This is a low-dimensional analytic oracle for checking an augmented
//! continuous-time state-space implementation. It is not a production filter,
//! a calibrated sensor model, a transport-delay model, or a validated enclosure.

const MAX_SCALED_TIME: f64 = 700.0;

/// Stable evaluation of (1 - exp(-x)) / x for x >= 0.
/// For subnormal x, the deviation from one is far smaller than half an ulp
/// at one; using one avoids division by a subnormal value.
fn one_minus_exp_neg_over_x(x: f64) -> f64 {
    if x < f64::MIN_POSITIVE {
        1.0
    } else {
        -(-x).exp_m1() / x
    }
}

/// Compute exp(-rate*time) while compensating for the rounded binary64
/// product. FMA returns the residual of rate*time - rounded_product.
fn exp_neg_rate_time(rate: f64, elapsed_time: f64) -> f64 {
    let product = rate * elapsed_time;
    let residual = rate.mul_add(elapsed_time, -product);
    (-product).exp() * (-residual).exp()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseModelError {
    InvalidProcessDecayRate,
    InvalidProcessVariance,
    InvalidResponseTimeConstant,
    InvalidElapsedTime,
    InvalidFrequency,
    NumericalFailure,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FirstOrderResponseOu {
    lambda: f64,
    process_variance: f64,
    mu: f64,
}

impl FirstOrderResponseOu {
    /// Construct the stationary OU + first-order response reference.
    ///
    /// lambda is the latent OU decay rate (1 / time), process_variance is
    /// the stationary variance of x, and response_time_constant is tau_s.
    pub fn new(
        lambda: f64,
        process_variance: f64,
        response_time_constant: f64,
    ) -> Result<Self, ResponseModelError> {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(ResponseModelError::InvalidProcessDecayRate);
        }
        if !process_variance.is_finite() || process_variance < 0.0 {
            return Err(ResponseModelError::InvalidProcessVariance);
        }
        if !response_time_constant.is_finite() || response_time_constant <= 0.0 {
            return Err(ResponseModelError::InvalidResponseTimeConstant);
        }
        let mu = 1.0 / response_time_constant;
        if !mu.is_finite() || mu <= 0.0 {
            return Err(ResponseModelError::InvalidResponseTimeConstant);
        }
        Ok(Self {
            lambda,
            process_variance,
            mu,
        })
    }

    pub fn process_decay_rate(&self) -> f64 {
        self.lambda
    }

    pub fn response_rate(&self) -> f64 {
        self.mu
    }

    pub fn process_variance(&self) -> f64 {
        self.process_variance
    }

    /// Continuous-time linear-SDE matrices for the coupled physical/sensor
    /// state S=[x,z]:
    ///   dS = A S dt + G dW,  Qc = G G^T.
    ///
    /// The sensor has no independent process noise here; acquisition/electronics
    /// noise is supplied separately at the observation layer.
    pub fn continuous_time_matrices(
        &self,
    ) -> Result<([[f64; 2]; 2], [[f64; 2]; 2]), ResponseModelError> {
        let diffusion = (self.lambda * self.process_variance) * 2.0;
        if !diffusion.is_finite() || diffusion < 0.0 {
            return Err(ResponseModelError::NumericalFailure);
        }
        Ok((
            [[-self.lambda, 0.0], [self.mu, -self.mu]],
            [[diffusion, 0.0], [0.0, 0.0]],
        ))
    }

    /// Stationary covariance of [x, z].
    ///
    /// P = [[var(x), cov(x,z)],
    ///      [cov(x,z), var(z)]]
    ///
    /// For the declared OU + first-order filter:
    /// cov(x,z) = var(z) = mu * var(x) / (lambda + mu).
    pub fn stationary_covariance(&self) -> [[f64; 2]; 2] {
        // Compute mu*variance/(lambda+mu) without overflowing the sum or
        // losing a subnormal mu/lambda ratio before the variance restores scale.
        let cross = if self.mu >= self.lambda {
            let ratio = self.lambda / self.mu;
            self.process_variance / (1.0 + ratio)
        } else {
            let ratio = self.mu / self.lambda;
            if ratio >= f64::MIN_POSITIVE {
                (self.process_variance * ratio) / (1.0 + ratio)
            } else {
                // Prefer (variance/lambda)*mu when that quotient is normal;
                // otherwise (variance*mu)/lambda has bounded intermediates
                // in the remaining subnormal-ratio regime.
                let variance_over_lambda = self.process_variance / self.lambda;
                if variance_over_lambda >= f64::MIN_POSITIVE
                    && variance_over_lambda.is_finite()
                {
                    variance_over_lambda * self.mu
                } else {
                    (self.process_variance * self.mu) / self.lambda
                }
            }
        };
        [
            [self.process_variance, cross],
            [cross, cross],
        ]
    }

    /// Exact transition matrix exp(A * elapsed_time), A=[[-lambda,0],[mu,-mu]].
    ///
    /// The off-diagonal is evaluated with exp_m1 and a slow-rate factorization
    /// to avoid cancellation when lambda and mu are close.
    pub fn transition_matrix(
        &self,
        elapsed_time: f64,
    ) -> Result<[[f64; 2]; 2], ResponseModelError> {
        if !elapsed_time.is_finite() || elapsed_time < 0.0 {
            return Err(ResponseModelError::InvalidElapsedTime);
        }

        let l = self.lambda;
        let m = self.mu;
        let t = elapsed_time;
        let scaled_l = l * t;
        let scaled_m = m * t;
        // Keep both transition modes away from silent exponential underflow.
        // The bounded oracle rejects larger dimensionless propagation rather
        // than discarding tiny transition terms that may be magnified by P.
        if !scaled_l.is_finite()
            || !scaled_m.is_finite()
            || scaled_l > MAX_SCALED_TIME
            || scaled_m > MAX_SCALED_TIME
        {
            return Err(ResponseModelError::NumericalFailure);
        }
        let e_l = exp_neg_rate_time(l, t);
        let e_m = exp_neg_rate_time(m, t);
        let delta = m - l;
        let gap_scaled = delta.abs() * t;
        if !gap_scaled.is_finite() {
            return Err(ResponseModelError::NumericalFailure);
        }
        // The transition gain is (mu*t) * (1-exp(-|mu-lambda|t)) /
        // (|mu-lambda|t). This factorization avoids intermediate underflow
        // and preserves accuracy for subnormal rate-gap products.
        let response_gain = scaled_m * one_minus_exp_neg_over_x(gap_scaled);
        let lower_left = if delta > 0.0 {
            e_l * response_gain
        } else if delta < 0.0 {
            e_m * response_gain
        } else {
            e_l * scaled_m
        };

        if !lower_left.is_finite() {
            return Err(ResponseModelError::NumericalFailure);
        }
        Ok([
            [e_l, 0.0],
            [lower_left, e_m],
        ])
    }

    /// Cov(S(t + h), S(t)) for h >= 0, where S=[x,z] is stationary.
    ///
    /// The result is exp(Ah) P. The reverse-time covariance is its transpose.
    pub fn future_past_cross_covariance(
        &self,
        elapsed_time: f64,
    ) -> Result<[[f64; 2]; 2], ResponseModelError> {
        let f = self.transition_matrix(elapsed_time)?;
        let p = self.stationary_covariance();

        let mut out = [[0.0; 2]; 2];
        for row in 0..2 {
            for col in 0..2 {
                out[row][col] = f[row][0] * p[0][col] + f[row][1] * p[1][col];
            }
        }
        Ok(out)
    }

    /// Stationary autocovariance of the filtered sensor output z.
    pub fn sensor_autocovariance(
        &self,
        elapsed_time: f64,
    ) -> Result<f64, ResponseModelError> {
        Ok(self.future_past_cross_covariance(elapsed_time)?[1][1])
    }

    /// Independently reduced convolution formula for Cov(z(t+h), z(t)), h >= 0.
    ///
    /// For h >= 0:
    /// C_zz(h) = q * (mu*exp(-lambda*h) - lambda*exp(-mu*h)) / (mu-lambda)
    /// with q = mu*process_variance/(lambda+mu). The implementation uses
    /// exp_m1-based factorizations and the repeated-root limit to avoid
    /// cancellation when the two rates are close.
    ///
    /// This serves as an independent analytic check on exp(Ah)P.
    pub fn sensor_autocovariance_convolution(
        &self,
        elapsed_time: f64,
    ) -> Result<f64, ResponseModelError> {
        if !elapsed_time.is_finite() || elapsed_time < 0.0 {
            return Err(ResponseModelError::InvalidElapsedTime);
        }

        let q = self.stationary_covariance()[1][1];
        let h = elapsed_time;
        let slow_rate = self.lambda.min(self.mu);
        let slow_scaled = slow_rate * h;
        let gap_scaled = (self.lambda - self.mu).abs() * h;
        if !slow_scaled.is_finite()
            || slow_scaled > MAX_SCALED_TIME
            || !gap_scaled.is_finite()
            || gap_scaled > MAX_SCALED_TIME
        {
            return Err(ResponseModelError::NumericalFailure);
        }
        let factor = exp_neg_rate_time(slow_rate, h)
            * (1.0 + slow_scaled * one_minus_exp_neg_over_x(gap_scaled));
        let covariance = q * factor;

        if covariance.is_finite() && covariance >= 0.0 {
            Ok(covariance)
        } else {
            Err(ResponseModelError::NumericalFailure)
        }
    }

    /// Normalized step response of the first-order instrument state.
    ///
    /// This is the response to a unit step in the latent measurand with the
    /// sensor initially at zero. It is not a step response of a complete
    /// sampling, transport, purification, or reporting chain.
    pub fn normalized_step_response(
        &self,
        elapsed_time: f64,
    ) -> Result<f64, ResponseModelError> {
        if !elapsed_time.is_finite() || elapsed_time < 0.0 {
            return Err(ResponseModelError::InvalidElapsedTime);
        }
        let scaled = self.mu * elapsed_time;
        if !scaled.is_finite() {
            // The exact positive product exceeds binary64 range, so the
            // normalized first-order step response rounds to one.
            return Ok(1.0);
        }
        let residual = self.mu.mul_add(elapsed_time, -scaled);
        let response = -(-scaled).exp_m1()
            + (-scaled).exp() * (-(-residual).exp_m1());
        if response.is_finite() && (0.0..=1.0).contains(&response) {
            Ok(response)
        } else {
            Err(ResponseModelError::NumericalFailure)
        }
    }

    /// Magnitude and phase (radians) of H(i*omega)=mu/(mu+i*omega).
    pub fn frequency_response(
        &self,
        angular_frequency: f64,
    ) -> Result<(f64, f64), ResponseModelError> {
        if !angular_frequency.is_finite() {
            return Err(ResponseModelError::InvalidFrequency);
        }
        let magnitude = self.mu / self.mu.hypot(angular_frequency);
        let phase = -angular_frequency.atan2(self.mu);
        Ok((magnitude, phase))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn stationary_covariance_preserves_scale_when_rate_ratio_underflows() {
        let lambda = 3.2990604078250345e62;
        let mu = 1.2242014100976064e-274;
        let tau = 1.0 / mu;
        let process_variance = 4.672577555410409e143;
        let model = FirstOrderResponseOu::new(lambda, process_variance, tau).unwrap();
        close(
            model.stationary_covariance()[1][1],
            1.7338803553145542e-193,
            2.0e-207,
        );
    }

    #[test]
    fn stationary_covariance_preserves_subnormal_rate_ratio() {
        let lambda = 7.071003610674585e260;
        let mu = 5.036876054794767e-63;
        let tau = 1.0 / mu;
        let variance = 2.545515374320597e50;
        let model = FirstOrderResponseOu::new(lambda, variance, tau).unwrap();
        close(
            model.stationary_covariance()[1][1],
            1.8132426656764452e-273,
            2.0e-287,
        );
    }

    #[test]
    fn stationary_covariance_matches_analytic_ou_filter_identity() {
        let model = FirstOrderResponseOu::new(1.0, 1.0, 1.0).unwrap();
        let p = model.stationary_covariance();
        close(p[0][0], 1.0, 1.0e-15);
        close(p[0][1], 0.5, 1.0e-15);
        close(p[1][0], 0.5, 1.0e-15);
        close(p[1][1], 0.5, 1.0e-15);
    }

    #[test]
    fn equal_process_and_sensor_rates_use_the_repeated_root_limit() {
        let model = FirstOrderResponseOu::new(1.0, 1.0, 1.0).unwrap();
        let actual = model.sensor_autocovariance(1.0).unwrap();
        let expected = (-1.0_f64).exp();
        close(actual, expected, 2.0e-15);
    }

    #[test]
    fn distinct_rates_match_closed_form_cross_covariance() {
        let model = FirstOrderResponseOu::new(1.0, 3.0, 0.5).unwrap();
        let actual = model.future_past_cross_covariance(1.0).unwrap();
        let expected = [
            [1.103638323514327, 0.7357588823428847],
            [1.6659355140822032, 1.200847198212544],
        ];
        for row in 0..2 {
            for col in 0..2 {
                close(actual[row][col], expected[row][col], 3.0e-14);
            }
        }
    }

    #[test]
    fn near_equal_rates_remain_continuous() {
        let model = FirstOrderResponseOu::new(1.0, 1.0, 1.0 / (1.0 + 1.0e-12)).unwrap();
        let actual = model.sensor_autocovariance(1.0).unwrap();
        let repeated_root_limit = (-1.0_f64).exp();
        close(actual, repeated_root_limit, 2.0e-12);
    }

    #[test]
    fn passive_single_channel_cannot_label_process_vs_sensor_rate() {
        // The two models swap lambda and mu. Adjusting the unknown latent
        // process variance preserves q = mu*variance/(lambda+mu), while the
        // normalized autocovariance factor is symmetric in the two rates.
        let physical_slow = FirstOrderResponseOu::new(0.25, 9.0, 0.25).unwrap();
        let sensor_slow = FirstOrderResponseOu::new(4.0, 144.0, 4.0).unwrap();
        for elapsed in [0.0, 0.01, 0.1, 0.5, 1.0, 4.0, 10.0] {
            close(
                physical_slow.sensor_autocovariance_convolution(elapsed).unwrap(),
                sensor_slow.sensor_autocovariance_convolution(elapsed).unwrap(),
                2.0e-14,
            );
            close(
                physical_slow.sensor_autocovariance(elapsed).unwrap(),
                sensor_slow.sensor_autocovariance(elapsed).unwrap(),
                2.0e-14,
            );
        }
    }

    #[test]
    fn sensor_inlet_step_breaks_the_process_sensor_pole_ambiguity() {
        let physical_slow = FirstOrderResponseOu::new(0.25, 9.0, 0.25).unwrap();
        let sensor_slow = FirstOrderResponseOu::new(4.0, 144.0, 4.0).unwrap();
        let elapsed = 0.5;
        let fast_sensor_step = physical_slow.normalized_step_response(elapsed).unwrap();
        let slow_sensor_step = sensor_slow.normalized_step_response(elapsed).unwrap();
        close(fast_sensor_step, 0.8646647167633873, 2.0e-15);
        close(slow_sensor_step, 0.11750309741540454, 2.0e-15);
        assert!((fast_sensor_step - slow_sensor_step).abs() > 0.7);
    }

    #[test]
    fn direct_convolution_matches_state_space_across_rate_orderings() {
        let cases = [
            (1.0, 3.0, 0.5, 0.0),
            (1.0, 3.0, 0.5, 0.25),
            (1.0, 3.0, 0.5, 1.0),
            (0.7, 2.3, 1.4, 2.0),
            (2.0, 4.0, 0.5, 0.25),
            (0.4, 1.7, 2.5, 7.0),
        ];
        for (lambda, variance, tau, elapsed) in cases {
            let model = FirstOrderResponseOu::new(lambda, variance, tau).unwrap();
            let state_space = model.sensor_autocovariance(elapsed).unwrap();
            let convolution = model.sensor_autocovariance_convolution(elapsed).unwrap();
            close(state_space, convolution, 5.0e-14);
        }
    }

    #[test]
    fn direct_convolution_matches_repeated_and_near_repeated_roots() {
        for tau in [1.0, 1.0 / (1.0 + 1.0e-12), 1.0 / (1.0 - 1.0e-12)] {
            let model = FirstOrderResponseOu::new(1.0, 1.0, tau).unwrap();
            for elapsed in [0.0, 0.1, 1.0, 10.0] {
                close(
                    model.sensor_autocovariance_convolution(elapsed).unwrap(),
                    model.sensor_autocovariance(elapsed).unwrap(),
                    3.0e-12,
                );
            }
        }
    }

    #[test]
    fn zero_lag_cross_covariance_is_the_stationary_covariance() {
        let model = FirstOrderResponseOu::new(0.7, 2.3, 1.4).unwrap();
        let p = model.stationary_covariance();
        let cross = model.future_past_cross_covariance(0.0).unwrap();
        for row in 0..2 {
            for col in 0..2 {
                close(cross[row][col], p[row][col], 1.0e-14);
            }
        }
    }

    #[test]
    fn equal_rate_transition_has_finite_repeated_root() {
        let model = FirstOrderResponseOu::new(2.0, 1.0, 0.5).unwrap();
        let f = model.transition_matrix(0.75).unwrap();
        let e = (-1.5_f64).exp();
        close(f[0][0], e, 1.0e-15);
        close(f[1][1], e, 1.0e-15);
        close(f[1][0], 1.5 * e, 1.0e-15);
    }

    #[test]
    fn step_response_reaches_ninety_percent_after_log_ten_tau() {
        let tau = 2.5;
        let model = FirstOrderResponseOu::new(0.8, 1.0, tau).unwrap();
        let t90 = tau * 10.0_f64.ln();
        close(model.normalized_step_response(t90).unwrap(), 0.9, 2.0e-15);
    }

    #[test]
    fn step_response_retains_small_fractional_changes() {
        let model = FirstOrderResponseOu::new(1.0, 1.0, 1.0).unwrap();
        let elapsed = 1.0e-20;
        let actual = model.normalized_step_response(elapsed).unwrap();
        close(actual, 1.0e-20, 1.0e-35);
    }

    #[test]
    fn frequency_response_has_minus_three_db_at_sensor_rate() {
        let model = FirstOrderResponseOu::new(1.0, 1.0, 0.25).unwrap();
        let (magnitude, phase) = model.frequency_response(4.0).unwrap();
        close(magnitude, std::f64::consts::FRAC_1_SQRT_2, 2.0e-15);
        close(phase, -std::f64::consts::FRAC_PI_4, 2.0e-15);
    }

    #[test]
    fn tiny_rate_and_large_elapsed_time_preserve_representable_transition() {
        let lambda = 1.1606977977727926e-247;
        let tau = 1.0 / 1.1606977977727924e-247;
        let model = FirstOrderResponseOu::new(lambda, 1.0, tau).unwrap();
        let actual = model.transition_matrix(5.169304199174938e249).unwrap()[1][0];
        assert!(actual > 0.0);
        close(actual, 1.5902379318025706e-258, 2.0e-270);
    }

    #[test]
    fn tiny_rate_gap_times_tiny_time_does_not_erase_transition() {
        let model = FirstOrderResponseOu::new(1.0, 1.0, 1.0 / (1.0 + f64::EPSILON)).unwrap();
        let t = 1.0e-310;
        let transition = model.transition_matrix(t).unwrap();
        let expected = t * (-t).exp();
        assert!(transition[1][0] > 0.0);
        close(transition[1][0], expected, 1.0e-323);
    }

    #[test]
    fn compensated_rate_time_product_matches_high_precision_transition() {
        let lambda = 7.646017666493531e-32;
        let tau = 1.0 / 7.64601766649353e-32;
        let elapsed = 7.847222255702168e33;
        let model = FirstOrderResponseOu::new(lambda, 1.0, tau).unwrap();
        let actual = model.transition_matrix(elapsed).unwrap()[1][0];
        close(actual, 1.5902379318027118e-258, 1.0e-272);
    }

    #[test]
    fn subnormal_rate_gap_product_uses_repeated_root_limit() {
        let cases = [
            (1.0725948952756432e-199, 1.0725948952756434e-199, 1.1348145351131996e-109),
            (7.277817676318291e-181, 7.27781767631829e-181, 3.0193425571006826e-128),
        ];
        for (lambda, target_mu, elapsed) in cases {
            let tau = 1.0 / target_mu;
            let model = FirstOrderResponseOu::new(lambda, 1.0, tau).unwrap();
            let mu = model.response_rate();
            let actual = model.transition_matrix(elapsed).unwrap()[1][0];
            let repeated_root = (mu * elapsed) * (-(lambda * elapsed)).exp();
            assert!(actual > 0.0);
            close(actual, repeated_root, f64::from_bits(1));
        }
    }

    #[test]
    fn subnormal_rate_gap_avoids_intermediate_product_underflow() {
        let tau = f64::MAX / 2.0;
        let mu = 1.0 / tau;
        for lambda in [mu.next_down(), mu.next_up()] {
            let model = FirstOrderResponseOu::new(lambda, 1.0, tau).unwrap();
            let transition = model.transition_matrix(1.0).unwrap();
            assert!(transition[1][0] > 0.0);
            close(transition[1][0], mu, mu * 1.0e-12);
        }
    }

    #[test]
    fn extreme_elapsed_time_rejects_instead_of_silent_underflow() {
        let model = FirstOrderResponseOu::new(1.0, 1.0, 1.0).unwrap();
        assert_eq!(
            model.transition_matrix(701.0).unwrap_err(),
            ResponseModelError::NumericalFailure
        );
        assert_eq!(
            model.sensor_autocovariance_convolution(701.0).unwrap_err(),
            ResponseModelError::NumericalFailure
        );
    }

    #[test]
    fn invalid_parameters_and_negative_time_reject() {
        assert_eq!(
            FirstOrderResponseOu::new(0.0, 1.0, 1.0).unwrap_err(),
            ResponseModelError::InvalidProcessDecayRate
        );
        assert_eq!(
            FirstOrderResponseOu::new(1.0, -1.0, 1.0).unwrap_err(),
            ResponseModelError::InvalidProcessVariance
        );
        assert_eq!(
            FirstOrderResponseOu::new(1.0, 1.0, 0.0).unwrap_err(),
            ResponseModelError::InvalidResponseTimeConstant
        );
        assert_eq!(
            FirstOrderResponseOu::new(1.0, 1.0, f64::from_bits(1)).unwrap_err(),
            ResponseModelError::InvalidResponseTimeConstant
        );
        let model = FirstOrderResponseOu::new(1.0, 1.0, 1.0).unwrap();
        assert_eq!(
            model.sensor_autocovariance(-0.1).unwrap_err(),
            ResponseModelError::InvalidElapsedTime
        );
    }
}
