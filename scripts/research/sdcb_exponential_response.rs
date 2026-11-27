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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseModelError {
    InvalidProcessDecayRate,
    InvalidProcessVariance,
    InvalidResponseTimeConstant,
    InvalidElapsedTime,
    InvalidFrequency,
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

    /// Stationary covariance of [x, z].
    ///
    /// P = [[var(x), cov(x,z)],
    ///      [cov(x,z), var(z)]]
    ///
    /// For the declared OU + first-order filter:
    /// cov(x,z) = var(z) = mu * var(x) / (lambda + mu).
    pub fn stationary_covariance(&self) -> [[f64; 2]; 2] {
        // Compute mu/(lambda+mu) without overflowing the sum.
        let response_fraction = if self.mu >= self.lambda {
            1.0 / (1.0 + self.lambda / self.mu)
        } else {
            let ratio = self.mu / self.lambda;
            ratio / (1.0 + ratio)
        };
        let cross = response_fraction * self.process_variance;
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
        let e_l = (-l * t).exp();
        let e_m = (-m * t).exp();
        let delta = m - l;

        let lower_left = if delta > 0.0 {
            // m * exp(-l t) * (1 - exp(-delta t)) / delta
            m * e_l * (-(-delta * t).exp_m1()) / delta
        } else if delta < 0.0 {
            // m * exp(-m t) * (1 - exp(-(-delta) t)) / (-delta)
            let gap = -delta;
            m * e_m * (-(-gap * t).exp_m1()) / gap
        } else {
            m * t * e_l
        };

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
        Ok(1.0 - (-self.mu * elapsed_time).exp())
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
    fn frequency_response_has_minus_three_db_at_sensor_rate() {
        let model = FirstOrderResponseOu::new(1.0, 1.0, 0.25).unwrap();
        let (magnitude, phase) = model.frequency_response(4.0).unwrap();
        close(magnitude, std::f64::consts::FRAC_1_SQRT_2, 2.0e-15);
        close(phase, -std::f64::consts::FRAC_PI_4, 2.0e-15);
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
