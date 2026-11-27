//! Exact interval-averaged observations for vector continuous-time SDEs.
//!
//! Build from repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_interval_observation.rs -o /tmp/sdcb-interval-observation
//! Run:
//!   /tmp/sdcb-interval-observation
//!
//! Reuses the Van Loan state-space oracle in sdcb_vector_state_space.rs.
//! Each observation is a rectangular-window average with its own start/end
//! times and loading row. Piecewise augmented-state propagation handles
//! overlapping, nested, duplicate and disjoint windows exactly.

#[path = "sdcb_vector_state_space.rs"]
mod state_space;

use state_space::{DiscreteTransition, GaussianState, Matrix};

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementWindow {
    pub id: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub loadings: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntervalObservationBundle {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub durations_seconds: Vec<f64>,
    /// Transition from the state at bundle start to the state at bundle end.
    pub state_transition: Matrix,
    /// Process covariance of the state at bundle end, excluding initial-state covariance.
    pub state_process_covariance: Matrix,
    /// Maps the initial state to each interval-averaged observation.
    pub observation_operator: Matrix,
    /// Process-noise covariance among interval averages, excluding initial-state covariance.
    pub observation_process_covariance: Matrix,
    /// Process-noise cross-covariance between the final state and interval averages.
    pub state_observation_process_covariance: Matrix,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    EmptyWindowSet,
    InvalidWindow,
    DuplicateWindowIdentity,
    InvalidLoading,
    TooManyAugmentedStates,
    InvalidDimensions,
    InvalidMeasurementNoise,
    InvalidObservation,
    NonPositiveDefiniteInnovation,
    NumericFailure,
    StateSpace(state_space::Error),
}

impl From<state_space::Error> for Error {
    fn from(value: state_space::Error) -> Self {
        Error::StateSpace(value)
    }
}

fn zeros(rows: usize, columns: usize) -> Matrix {
    vec![vec![0.0; columns]; rows]
}

fn identity(n: usize) -> Matrix {
    let mut result = zeros(n, n);
    for i in 0..n {
        result[i][i] = 1.0;
    }
    result
}

fn transpose(matrix: &[Vec<f64>]) -> Matrix {
    let rows = matrix.len();
    let columns = matrix[0].len();
    let mut result = zeros(columns, rows);
    for i in 0..rows {
        for j in 0..columns {
            result[j][i] = matrix[i][j];
        }
    }
    result
}

fn multiply(left: &[Vec<f64>], right: &[Vec<f64>]) -> Result<Matrix, Error> {
    if left.is_empty() || right.is_empty()
        || left[0].len() != right.len()
        || left.iter().any(|row| row.len() != left[0].len())
        || right.iter().any(|row| row.len() != right[0].len())
    {
        return Err(Error::InvalidDimensions);
    }
    let mut result = zeros(left.len(), right[0].len());
    for i in 0..left.len() {
        for k in 0..right.len() {
            for j in 0..right[0].len() {
                result[i][j] += left[i][k] * right[k][j];
            }
        }
    }
    if result.iter().flatten().all(|value| value.is_finite()) {
        Ok(result)
    } else {
        Err(Error::NumericFailure)
    }
}

fn add(left: &[Vec<f64>], right: &[Vec<f64>]) -> Result<Matrix, Error> {
    if left.len() != right.len()
        || left.iter().zip(right).any(|(a, b)| a.len() != b.len())
    {
        return Err(Error::InvalidDimensions);
    }
    let result: Matrix = left.iter().zip(right).map(|(a, b)| {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }).collect();
    if result.iter().flatten().all(|value| value.is_finite()) {
        Ok(result)
    } else {
        Err(Error::NumericFailure)
    }
}

fn subtract(left: &[Vec<f64>], right: &[Vec<f64>]) -> Result<Matrix, Error> {
    if left.len() != right.len()
        || left.iter().zip(right).any(|(a, b)| a.len() != b.len())
    {
        return Err(Error::InvalidDimensions);
    }
    let result: Matrix = left.iter().zip(right).map(|(a, b)| {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    }).collect();
    if result.iter().flatten().all(|value| value.is_finite()) {
        Ok(result)
    } else {
        Err(Error::NumericFailure)
    }
}

fn matrix_vector(matrix: &[Vec<f64>], vector: &[f64]) -> Result<Vec<f64>, Error> {
    if matrix.is_empty() || matrix.iter().any(|row| row.len() != vector.len()) {
        return Err(Error::InvalidDimensions);
    }
    let result: Vec<f64> = matrix.iter().map(|row| {
        row.iter().zip(vector).map(|(a, b)| a * b).sum()
    }).collect();
    if result.iter().all(|value| value.is_finite()) {
        Ok(result)
    } else {
        Err(Error::NumericFailure)
    }
}

fn cholesky(matrix: &[Vec<f64>]) -> Result<Matrix, Error> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n)
        || matrix.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidDimensions);
    }
    let scale = matrix.iter().flatten().map(|value| value.abs()).fold(0.0_f64, f64::max);
    for i in 0..n {
        for j in 0..i {
            if (matrix[i][j] - matrix[j][i]).abs() > 1.0e-10 * scale.max(f64::MIN_POSITIVE) {
                return Err(Error::NonPositiveDefiniteInnovation);
            }
        }
    }
    let mut lower = zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i][j];
            for k in 0..j {
                value -= lower[i][k] * lower[j][k];
            }
            if i == j {
                if !value.is_finite() || value <= 1.0e-300 {
                    return Err(Error::NonPositiveDefiniteInnovation);
                }
                lower[i][j] = value.sqrt();
            } else {
                lower[i][j] = value / lower[j][j];
            }
        }
    }
    Ok(lower)
}

fn solve_cholesky_vector(lower: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, Error> {
    let n = lower.len();
    if rhs.len() != n {
        return Err(Error::InvalidDimensions);
    }
    let mut forward = vec![0.0; n];
    for i in 0..n {
        let mut value = rhs[i];
        for j in 0..i {
            value -= lower[i][j] * forward[j];
        }
        forward[i] = value / lower[i][i];
    }
    let mut result = vec![0.0; n];
    for i in (0..n).rev() {
        let mut value = forward[i];
        for j in (i + 1)..n {
            value -= lower[j][i] * result[j];
        }
        result[i] = value / lower[i][i];
    }
    if result.iter().all(|value| value.is_finite()) {
        Ok(result)
    } else {
        Err(Error::NumericFailure)
    }
}

fn solve_cholesky_matrix(lower: &[Vec<f64>], rhs: &[Vec<f64>]) -> Result<Matrix, Error> {
    if rhs.len() != lower.len() || rhs.is_empty()
        || rhs.iter().any(|row| row.len() != rhs[0].len())
    {
        return Err(Error::InvalidDimensions);
    }
    let columns = rhs[0].len();
    let mut result = zeros(rhs.len(), columns);
    for column in 0..columns {
        let vector: Vec<f64> = rhs.iter().map(|row| row[column]).collect();
        let solved = solve_cholesky_vector(lower, &vector)?;
        for row in 0..rhs.len() {
            result[row][column] = solved[row];
        }
    }
    Ok(result)
}

/// Build exact first and second moments for interval-averaged observations.
///
/// The initial state is at the earliest window start. Each integrator is
/// activated at its own start time and frozen at its own end time. All active
/// interval accumulators are propagated jointly with the physical state so
/// their process-noise covariance and endpoint cross-covariance are retained.
pub fn build_interval_observation_bundle(
    generator: &[Vec<f64>],
    diffusion_covariance: &[Vec<f64>],
    windows: &[MeasurementWindow],
) -> Result<IntervalObservationBundle, Error> {
    if windows.is_empty() {
        return Err(Error::EmptyWindowSet);
    }
    let n = generator.len();
    if n == 0 || generator.iter().any(|row| row.len() != n)
        || diffusion_covariance.len() != n
        || diffusion_covariance.iter().any(|row| row.len() != n)
    {
        return Err(Error::InvalidDimensions);
    }
    if n + windows.len() > 8 {
        return Err(Error::TooManyAugmentedStates);
    }
    for (index, window) in windows.iter().enumerate() {
        if window.id.trim().is_empty()
            || windows[..index].iter().any(|earlier| earlier.id == window.id)
        {
            return Err(Error::DuplicateWindowIdentity);
        }
        if !window.start_seconds.is_finite()
            || !window.end_seconds.is_finite()
            || window.end_seconds <= window.start_seconds
        {
            return Err(Error::InvalidWindow);
        }
        if window.loadings.len() != n || window.loadings.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidLoading);
        }
    }

    let global_start = windows.iter().map(|window| window.start_seconds)
        .fold(f64::INFINITY, f64::min);
    let global_end = windows.iter().map(|window| window.end_seconds)
        .fold(f64::NEG_INFINITY, f64::max);
    let total_time = global_end - global_start;
    if !total_time.is_finite() || total_time <= 0.0 {
        return Err(Error::InvalidWindow);
    }
    let m = windows.len();
    let augmented_dimension = n + m;

    let relative_starts: Vec<f64> = windows.iter()
        .map(|window| window.start_seconds - global_start).collect();
    let relative_ends: Vec<f64> = windows.iter()
        .map(|window| window.end_seconds - global_start).collect();
    let durations: Vec<f64> = windows.iter()
        .map(|window| window.end_seconds - window.start_seconds).collect();

    let mut boundaries = vec![0.0, total_time];
    boundaries.extend(relative_starts.iter().copied());
    boundaries.extend(relative_ends.iter().copied());
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup_by(|left, right| *left == *right);

    let mut accumulated_transition = identity(augmented_dimension);
    let mut accumulated_covariance = zeros(augmented_dimension, augmented_dimension);

    for pair in boundaries.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let delta = end - start;
        if delta <= 0.0 {
            continue;
        }
        let mut augmented_generator = zeros(augmented_dimension, augmented_dimension);
        let mut augmented_diffusion = zeros(augmented_dimension, augmented_dimension);
        for i in 0..n {
            for j in 0..n {
                augmented_generator[i][j] = generator[i][j];
                augmented_diffusion[i][j] = diffusion_covariance[i][j];
            }
        }
        for observation_index in 0..m {
            let active = relative_starts[observation_index] <= start
                && relative_ends[observation_index] >= end;
            if active {
                for state_index in 0..n {
                    augmented_generator[n + observation_index][state_index] =
                        windows[observation_index].loadings[state_index];
                }
            }
        }

        let segment = state_space::discretize_linear_sde(
            &augmented_generator, &augmented_diffusion, delta
        )?;
        let propagated_covariance = multiply(
            &multiply(&segment.transition, &accumulated_covariance)?,
            &transpose(&segment.transition),
        )?;
        accumulated_covariance = add(&propagated_covariance, &segment.process_covariance)?;
        accumulated_transition = multiply(&segment.transition, &accumulated_transition)?;
    }

    let mut state_transition = zeros(n, n);
    let mut state_process_covariance = zeros(n, n);
    let mut observation_operator = zeros(m, n);
    let mut observation_process_covariance = zeros(m, m);
    let mut state_observation_process_covariance = zeros(n, m);

    for i in 0..n {
        for j in 0..n {
            state_transition[i][j] = accumulated_transition[i][j];
            state_process_covariance[i][j] = accumulated_covariance[i][j];
        }
        for j in 0..m {
            state_observation_process_covariance[i][j] =
                accumulated_covariance[i][n + j] / durations[j];
        }
    }
    for i in 0..m {
        for j in 0..n {
            observation_operator[i][j] =
                accumulated_transition[n + i][j] / durations[i];
        }
        for j in 0..m {
            observation_process_covariance[i][j] =
                accumulated_covariance[n + i][n + j] / (durations[i] * durations[j]);
        }
    }

    Ok(IntervalObservationBundle {
        start_seconds: global_start,
        end_seconds: global_end,
        durations_seconds: durations,
        state_transition,
        state_process_covariance,
        observation_operator,
        observation_process_covariance,
        state_observation_process_covariance,
    })
}

fn validate_psd_covariance(covariance: &[Vec<f64>]) -> Result<(), Error> {
    let n = covariance.len();
    if n == 0 || covariance.iter().any(|row| row.len() != n) {
        return Err(Error::InvalidDimensions);
    }
    let check = state_space::propagate_gaussian(
        &GaussianState { mean: vec![0.0; n], covariance: covariance.to_vec() },
        &DiscreteTransition {
            transition: identity(n),
            process_covariance: zeros(n, n),
            elapsed_seconds: 0.0,
        },
    );
    check.map(|_| ()).map_err(Error::StateSpace)
}

/// Predict the observation covariance for an initial Gaussian state and an
/// independent acquisition-noise covariance. Process and overlapping-window
/// covariance are already included in the bundle.
pub fn predict_observation_covariance(
    initial_covariance: &[Vec<f64>],
    bundle: &IntervalObservationBundle,
    acquisition_noise_covariance: &[Vec<f64>],
) -> Result<Matrix, Error> {
    let n = bundle.state_transition.len();
    let m = bundle.durations_seconds.len();
    if initial_covariance.len() != n
        || initial_covariance.iter().any(|row| row.len() != n)
        || acquisition_noise_covariance.len() != m
        || acquisition_noise_covariance.iter().any(|row| row.len() != m)
    {
        return Err(Error::InvalidDimensions);
    }
    validate_psd_covariance(initial_covariance)?;
    validate_psd_covariance(acquisition_noise_covariance)?;
    let projected = multiply(
        &multiply(&bundle.observation_operator, initial_covariance)?,
        &transpose(&bundle.observation_operator),
    )?;
    add(
        &add(&projected, &bundle.observation_process_covariance)?,
        acquisition_noise_covariance,
    )
}

/// Condition the final state on interval-averaged observations. The update
/// uses Cov(x_end, y), including the process-noise cross-covariance created
/// during the observation windows; treating y as an endpoint sample is wrong.
pub fn condition_endpoint_on_observations(
    initial_state: &GaussianState,
    bundle: &IntervalObservationBundle,
    observations: &[f64],
    acquisition_noise_covariance: &[Vec<f64>],
) -> Result<GaussianState, Error> {
    let n = bundle.state_transition.len();
    let m = bundle.durations_seconds.len();
    if initial_state.mean.len() != n
        || initial_state.covariance.len() != n
        || initial_state.covariance.iter().any(|row| row.len() != n)
        || observations.len() != m
        || observations.iter().any(|value| !value.is_finite())
        || acquisition_noise_covariance.len() != m
        || acquisition_noise_covariance.iter().any(|row| row.len() != m)
    {
        return Err(Error::InvalidDimensions);
    }

    let transition = DiscreteTransition {
        transition: bundle.state_transition.clone(),
        process_covariance: bundle.state_process_covariance.clone(),
        elapsed_seconds: bundle.end_seconds - bundle.start_seconds,
    };
    let predicted_endpoint = state_space::propagate_gaussian(initial_state, &transition)?;

    let predicted_observation_mean = matrix_vector(
        &bundle.observation_operator, &initial_state.mean
    )?;
    let innovation: Vec<f64> = observations.iter()
        .zip(&predicted_observation_mean)
        .map(|(observed, predicted)| observed - predicted)
        .collect();

    let observation_covariance = predict_observation_covariance(
        &initial_state.covariance, bundle, acquisition_noise_covariance
    )?;
    let lower = cholesky(&observation_covariance)?;

    let endpoint_observation_cross = add(
        &multiply(
            &multiply(&bundle.state_transition, &initial_state.covariance)?,
            &transpose(&bundle.observation_operator),
        )?,
        &bundle.state_observation_process_covariance,
    )?;

    let solved_innovation = solve_cholesky_vector(&lower, &innovation)?;
    let mean_update = matrix_vector(&endpoint_observation_cross, &solved_innovation)?;
    let updated_mean: Vec<f64> = predicted_endpoint.mean.iter()
        .zip(mean_update).map(|(mean, update)| mean + update).collect();

    let solved_cross = solve_cholesky_matrix(&lower, &transpose(&endpoint_observation_cross))?;
    let covariance_reduction = multiply(&endpoint_observation_cross, &solved_cross)?;
    let mut updated_covariance = subtract(
        &predicted_endpoint.covariance, &covariance_reduction
    )?;
    for i in 0..n {
        for j in 0..i {
            let symmetric = 0.5 * (updated_covariance[i][j] + updated_covariance[j][i]);
            updated_covariance[i][j] = symmetric;
            updated_covariance[j][i] = symmetric;
        }
    }

    // Reuse the state-space oracle's PSD guard without changing the result.
    let identity_transition = DiscreteTransition {
        transition: identity(n),
        process_covariance: zeros(n, n),
        elapsed_seconds: 0.0,
    };
    let checked = state_space::propagate_gaussian(
        &GaussianState { mean: updated_mean, covariance: updated_covariance },
        &identity_transition,
    )?;
    Ok(checked)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!(
            (a-b).abs() <= tol,
            "actual={a:.15}, expected={b:.15}, tolerance={tol:e}"
        );
    }

    fn ou_windows() -> Vec<MeasurementWindow> {
        vec![
            MeasurementWindow {
                id:"slow-reference".into(), start_seconds:0.0, end_seconds:2.0,
                loadings:vec![1.0]
            },
            MeasurementWindow {
                id:"fast-mobile".into(), start_seconds:1.0, end_seconds:2.0,
                loadings:vec![1.0]
            }
        ]
    }

    #[test]
    fn nested_window_covariance_matches_scalar_ou_reference() {
        let bundle=build_interval_observation_bundle(
            &vec![vec![-1.0]],&vec![vec![2.0]],&ou_windows()
        ).unwrap();
        let predicted=predict_observation_covariance(
            &vec![vec![1.0]],&bundle,&vec![vec![0.0,0.0],vec![0.0,0.0]]
        ).unwrap();
        close(predicted[0][0],0.5676676416183064,2.0e-12);
        close(predicted[0][1],0.5676676416183064,2.0e-12);
        close(predicted[1][0],0.5676676416183064,2.0e-12);
        close(predicted[1][1],0.7357588823428847,2.0e-12);
    }

    #[test]
    fn duplicate_windows_produce_duplicate_covariance_rows() {
        let mut windows=ou_windows();
        windows.push(MeasurementWindow {
            id:"same-as-reference".into(),start_seconds:0.0,end_seconds:2.0,
            loadings:vec![1.0]
        });
        let bundle=build_interval_observation_bundle(
            &vec![vec![-1.0]],&vec![vec![2.0]],&windows
        ).unwrap();
        let predicted=predict_observation_covariance(
            &vec![vec![1.0]],&bundle,&zeros(3,3)
        ).unwrap();
        for column in 0..3 {
            close(predicted[0][column],predicted[2][column],2.0e-12);
        }
    }

    #[test]
    fn disjoint_windows_recover_exponential_gap_covariance() {
        let windows=vec![
            MeasurementWindow{id:"early".into(),start_seconds:0.0,end_seconds:1.0,loadings:vec![1.0]},
            MeasurementWindow{id:"late".into(),start_seconds:2.0,end_seconds:3.0,loadings:vec![1.0]}
        ];
        let bundle=build_interval_observation_bundle(
            &vec![vec![-1.0]],&vec![vec![2.0]],&windows
        ).unwrap();
        let predicted=predict_observation_covariance(
            &vec![vec![1.0]],&bundle,&zeros(2,2)
        ).unwrap();
        let expected=(-1.0_f64).exp()*(1.0-(-1.0_f64).exp()).powi(2);
        close(predicted[0][1],expected,2.0e-12);
    }

    #[test]
    fn endpoint_conditioning_uses_interval_process_cross_covariance() {
        let bundle=build_interval_observation_bundle(
            &vec![vec![-1.0]],&vec![vec![2.0]],&ou_windows()
        ).unwrap();
        let prior=GaussianState{mean:vec![0.0],covariance:vec![vec![1.0]]};
        let posterior=condition_endpoint_on_observations(
            &prior,&bundle,&[1.0,0.7],&vec![vec![0.01,0.0],vec![0.0,0.01]]
        ).unwrap();
        close(posterior.mean[0],0.43660632952719985,2.0e-11);
        close(posterior.covariance[0][0],0.4478182406823277,2.0e-11);
    }

    #[test]
    fn unequal_observation_loadings_preserve_cross_factor_contributions() {
        let a=vec![vec![-1.0,0.2],vec![0.0,-0.5]];
        let qc=vec![vec![0.3,0.1],vec![0.1,0.2]];
        let windows=vec![
            MeasurementWindow{id:"reference-factor".into(),start_seconds:0.0,end_seconds:2.0,loadings:vec![0.0,1.0]},
            MeasurementWindow{id:"mixed-mobile".into(),start_seconds:0.5,end_seconds:1.5,loadings:vec![1.0,0.5]}
        ];
        let bundle=build_interval_observation_bundle(&a,&qc,&windows).unwrap();
        assert!(bundle.observation_operator[0][1].abs()>0.1);
        assert!(bundle.observation_operator[1][0].abs()>0.1);
        assert!(bundle.observation_process_covariance[0][1].abs()>0.0);
    }

    #[test]
    fn acquisition_noise_is_added_separately_from_process_covariance() {
        let bundle=build_interval_observation_bundle(
            &vec![vec![-1.0]],&vec![vec![2.0]],&ou_windows()
        ).unwrap();
        let base=predict_observation_covariance(
            &vec![vec![1.0]],&bundle,&zeros(2,2)
        ).unwrap();
        let noisy=predict_observation_covariance(
            &vec![vec![1.0]],&bundle,&vec![vec![0.01,0.003],vec![0.003,0.02]]
        ).unwrap();
        close(noisy[0][0]-base[0][0],0.01,1.0e-14);
        close(noisy[0][1]-base[0][1],0.003,1.0e-14);
        close(noisy[1][1]-base[1][1],0.02,1.0e-14);
    }

    #[test]
    fn invalid_windows_duplicate_ids_and_excess_dimension_fail_closed() {
        let invalid=vec![
            MeasurementWindow{id:"bad".into(),start_seconds:1.0,end_seconds:1.0,loadings:vec![1.0]}
        ];
        assert_eq!(
            build_interval_observation_bundle(&vec![vec![-1.0]],&vec![vec![1.0]],&invalid),
            Err(Error::InvalidWindow)
        );
        let duplicate=vec![
            MeasurementWindow{id:"x".into(),start_seconds:0.0,end_seconds:1.0,loadings:vec![1.0]},
            MeasurementWindow{id:"x".into(),start_seconds:0.0,end_seconds:2.0,loadings:vec![1.0]}
        ];
        assert_eq!(
            build_interval_observation_bundle(&vec![vec![-1.0]],&vec![vec![1.0]],&duplicate),
            Err(Error::DuplicateWindowIdentity)
        );


    #[test]
    fn invalid_acquisition_covariance_fails_closed() {
        let bundle=build_interval_observation_bundle(
            &vec![vec![-1.0]],&vec![vec![2.0]],&ou_windows()
        ).unwrap();
        assert!(matches!(
            predict_observation_covariance(
                &vec![vec![1.0]],&bundle,&vec![vec![-0.01,0.0],vec![0.0,0.01]]
            ),
            Err(Error::StateSpace(state_space::Error::CovarianceNotPositiveSemidefinite))
        ));
    }
    }
}
