//! Composition bridge from the OU + first-order sensor response oracle
//! to the existing interval-integrated observation engine.
//!
//! Build from the repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_response_interval_bridge.rs -o /tmp/sdcb-response-interval
//!
//! This is a research integration fixture. It does not add a second interval
//! integrator, planner, or evidence ledger.

#[path = "sdcb_exponential_response.rs"]
mod response;
#[path = "sdcb_interval_observation.rs"]
mod interval_observation;

use interval_observation::{
    build_interval_observation_bundle,
    predict_observation_covariance,
    IntervalObservationBundle,
    MeasurementWindow,
};
use response::{FirstOrderResponseOu, ResponseModelError};

#[derive(Debug, Clone, PartialEq)]
pub enum BridgeError {
    Response(ResponseModelError),
    Interval(interval_observation::Error),
}

impl From<ResponseModelError> for BridgeError {
    fn from(value: ResponseModelError) -> Self {
        Self::Response(value)
    }
}

impl From<interval_observation::Error> for BridgeError {
    fn from(value: interval_observation::Error) -> Self {
        Self::Interval(value)
    }
}

fn to_matrix(input: [[f64; 2]; 2]) -> Vec<Vec<f64>> {
    input.into_iter().map(|row| row.into_iter().collect()).collect()
}

/// Build the existing interval-observation bundle with the measured state
/// loading fixed to z (the instrument-response state), not x (the measurand).
pub fn build_response_interval_bundle(
    model: &FirstOrderResponseOu,
    windows: &[MeasurementWindow],
) -> Result<IntervalObservationBundle, BridgeError> {
    let (generator, diffusion) = model.continuous_time_matrices()?;
    Ok(build_interval_observation_bundle(
        &to_matrix(generator),
        &to_matrix(diffusion),
        windows,
    )?)
}

/// Predict the covariance of interval-averaged sensor outputs from the
/// stationary coupled physical/sensor state. Acquisition noise is an
/// independent, caller-supplied covariance owned by the existing adapter.
pub fn predict_stationary_response_window_covariance(
    model: &FirstOrderResponseOu,
    bundle: &IntervalObservationBundle,
    acquisition_noise_covariance: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, BridgeError> {
    let initial_covariance = to_matrix(model.stationary_covariance());
    Ok(predict_observation_covariance(
        &initial_covariance,
        bundle,
        acquisition_noise_covariance,
    )?)
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

    fn sensor_window(id: &str, start: f64, end: f64) -> MeasurementWindow {
        MeasurementWindow {
            id: id.to_owned(),
            start_seconds: start,
            end_seconds: end,
            loadings: vec![0.0, 1.0],
        }
    }

    fn zero_noise(n: usize) -> Vec<Vec<f64>> {
        vec![vec![0.0; n]; n]
    }

    #[test]
    fn response_dynamics_compose_with_existing_overlapping_window_engine() {
        // dx=-x dt + sqrt(2)dW, dz=2(x-z)dt, stationary var(x)=1.
        // The resulting sensor autocovariance is (2/3)*(2e^-h - e^-2h).
        let model = FirstOrderResponseOu::new(1.0, 1.0, 0.5).unwrap();
        let windows = [
            sensor_window("long", 0.0, 2.0),
            sensor_window("nested_tail", 1.0, 2.0),
        ];
        let bundle = build_response_interval_bundle(&model, &windows).unwrap();
        let covariance =
            predict_stationary_response_window_covariance(&model, &bundle, &zero_noise(2))
                .unwrap();

        // Independent SciPy dblquad references, also checked against the
        // closed-form covariance primitive in the authoring runtime.
        close(covariance[0][0], 0.5053638855836804, 2.0e-10);
        close(covariance[0][1], 0.5053638855836804, 2.0e-10);
        close(covariance[1][0], 0.5053638855836804, 2.0e-10);
        close(covariance[1][1], 0.6025667487115534, 2.0e-10);
    }

    #[test]
    fn adjacent_response_windows_preserve_cross_window_covariance() {
        let model = FirstOrderResponseOu::new(1.0, 1.0, 0.5).unwrap();
        let windows = [
            sensor_window("first_half", 0.0, 1.0),
            sensor_window("second_half", 1.0, 2.0),
        ];
        let bundle = build_response_interval_bundle(&model, &windows).unwrap();
        let covariance =
            predict_stationary_response_window_covariance(&model, &bundle, &zero_noise(2))
                .unwrap();

        close(covariance[0][0], 0.6025667487115534, 2.0e-10);
        close(covariance[0][1], 0.4081610224557192, 2.0e-10);
        close(covariance[1][0], 0.4081610224557192, 2.0e-10);
        close(covariance[1][1], 0.6025667487115534, 2.0e-10);
    }

    #[test]
    fn response_window_model_rejects_duplicate_ids_and_empty_windows() {
        let model = FirstOrderResponseOu::new(1.0, 1.0, 0.5).unwrap();
        let duplicate = [
            sensor_window("same", 0.0, 1.0),
            sensor_window("same", 1.0, 2.0),
        ];
        assert!(matches!(
            build_response_interval_bundle(&model, &duplicate),
            Err(BridgeError::Interval(interval_observation::Error::DuplicateWindowIdentity))
        ));

        let empty = [sensor_window("zero", 1.0, 1.0)];
        assert!(matches!(
            build_response_interval_bundle(&model, &empty),
            Err(BridgeError::Interval(interval_observation::Error::InvalidWindow))
        ));
    }

    #[test]
    fn sensor_acquisition_noise_is_added_after_physical_response_averaging() {
        let model = FirstOrderResponseOu::new(1.0, 1.0, 0.5).unwrap();
        let windows = [
            sensor_window("a", 0.0, 1.0),
            sensor_window("b", 1.0, 2.0),
        ];
        let bundle = build_response_interval_bundle(&model, &windows).unwrap();
        let no_noise =
            predict_stationary_response_window_covariance(&model, &bundle, &zero_noise(2))
                .unwrap();
        let noise = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
        let with_noise =
            predict_stationary_response_window_covariance(&model, &bundle, &noise).unwrap();
        close(with_noise[0][0] - no_noise[0][0], 0.04, 2.0e-14);
        close(with_noise[1][1] - no_noise[1][1], 0.09, 2.0e-14);
        close(with_noise[0][1], no_noise[0][1], 2.0e-14);
    }
}
