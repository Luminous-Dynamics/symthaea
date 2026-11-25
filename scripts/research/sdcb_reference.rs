//! Standalone, dependency-free analytic reference controls for SDCB v1.
//!
//! This file is intentionally usable without the Symthaea workspace:
//!   rustc --edition=2024 --test scripts/research/sdcb_reference.rs -o /tmp/sdcb-reference-tests
//!   /tmp/sdcb-reference-tests
//!
//! It is a small reference oracle for hand-checkable cases, not a model,
//! benchmark result, Lean checker, or production spectral-analysis library.

/// The discrete-time stability classification for a finite 2x2 real matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscreteStability {
    /// Spectral radius is strictly below one (within the chosen tolerance).
    AsymptoticallyStable,
    /// Spectral radius is within tolerance of one.
    Marginal,
    /// Spectral radius is strictly above one (outside the chosen tolerance).
    Unstable,
}

fn finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// Eigenvalues of the real symmetric matrix [[a, b], [b, d]], in ascending order.
pub fn symmetric_eigenvalues_2x2(a: f64, b: f64, d: f64) -> Result<[f64; 2], &'static str> {
    if !finite(&[a, b, d]) {
        return Err("matrix entries must be finite");
    }
    let center = (a + d) * 0.5;
    let radius = (((a - d) * (a - d) + 4.0 * b * b) * 0.25).sqrt();
    let values = [center - radius, center + radius];
    if finite(&values) {
        Ok(values)
    } else {
        Err("eigenvalue calculation overflowed")
    }
}

/// Spectral radius of a real 2x2 matrix [[a, b], [c, d]].
///
/// For a complex-conjugate eigenvalue pair, both eigenvalues have modulus
/// sqrt(det(A)); for real eigenvalues the larger absolute root is returned.
pub fn spectral_radius_2x2(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
) -> Result<f64, &'static str> {
    if !finite(&[a, b, c, d]) {
        return Err("matrix entries must be finite");
    }
    let trace = a + d;
    let determinant = a * d - b * c;
    let discriminant = trace * trace - 4.0 * determinant;
    if !finite(&[trace, determinant, discriminant]) {
        return Err("spectral calculation overflowed");
    }

    let radius = if discriminant >= 0.0 {
        let root = discriminant.sqrt();
        let lambda_1 = (trace + root) * 0.5;
        let lambda_2 = (trace - root) * 0.5;
        lambda_1.abs().max(lambda_2.abs())
    } else if determinant >= 0.0 {
        determinant.sqrt()
    } else {
        return Err("invalid negative determinant for complex eigenvalue pair");
    };

    if radius.is_finite() {
        Ok(radius)
    } else {
        Err("spectral radius is not finite")
    }
}

/// Classify a discrete-time linear system x[t+1] = A x[t] by spectral radius.
pub fn classify_discrete_stability(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    tolerance: f64,
) -> Result<DiscreteStability, &'static str> {
    if !tolerance.is_finite() || tolerance < 0.0 || tolerance >= 1.0 {
        return Err("tolerance must be finite and in [0, 1)");
    }
    let radius = spectral_radius_2x2(a, b, c, d)?;
    Ok(if radius < 1.0 - tolerance {
        DiscreteStability::AsymptoticallyStable
    } else if radius > 1.0 + tolerance {
        DiscreteStability::Unstable
    } else {
        DiscreteStability::Marginal
    })
}

/// Exact 2x2 continuous-time Hurwitz criterion for x'(t) = A x(t).
///
/// For a real 2x2 matrix this is asymptotically stable iff trace(A) < 0
/// and det(A) > 0. This checks the linear reference case only.
pub fn continuous_time_hurwitz_2x2(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
) -> Result<bool, &'static str> {
    if !finite(&[a, b, c, d]) {
        return Err("matrix entries must be finite");
    }
    let trace = a + d;
    let determinant = a * d - b * c;
    if !finite(&[trace, determinant]) {
        return Err("Hurwitz calculation overflowed");
    }
    Ok(trace < 0.0 && determinant > 0.0)
}

/// Adjacent-gap ratios for a strictly increasing finite sequence.
///
/// For gaps s_i = x_(i+1)-x_i, returns min(s_i,s_(i+1))/max(...).
/// These ratios are invariant to positive affine changes of the levels.
/// They are descriptive statistics, not a proof of a random-matrix ensemble.
pub fn adjacent_gap_ratios(levels: &[f64]) -> Result<Vec<f64>, &'static str> {
    if levels.len() < 3 {
        return Err("at least three spectral levels are required");
    }
    if !levels.iter().all(|value| value.is_finite()) {
        return Err("spectral levels must be finite");
    }
    let mut gaps = Vec::with_capacity(levels.len() - 1);
    for pair in levels.windows(2) {
        let gap = pair[1] - pair[0];
        if !gap.is_finite() || gap <= 0.0 {
            return Err("spectral levels must be strictly increasing");
        }
        gaps.push(gap);
    }

    Ok(gaps
        .windows(2)
        .map(|pair| pair[0].min(pair[1]) / pair[0].max(pair[1]))
        .collect())
}

/// Simple deterministic sieve, returning all primes <= limit.
pub fn prime_sieve(limit: usize) -> Vec<usize> {
    if limit < 2 {
        return Vec::new();
    }
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    is_prime[1] = false;

    let mut p = 2usize;
    while p <= limit / p {
        if is_prime[p] {
            let mut multiple = p * p;
            while multiple <= limit {
                is_prime[multiple] = false;
                if multiple > limit - p {
                    break;
                }
                multiple += p;
            }
        }
        p += 1;
    }

    is_prime
        .into_iter()
        .enumerate()
        .filter_map(|(number, prime)| prime.then_some(number))
        .collect()
}

fn validate_binary_vector(vector: &[u8]) -> Result<(), &'static str> {
    if vector.is_empty() {
        return Err("binary vector must not be empty");
    }
    if vector.iter().any(|bit| *bit > 1) {
        return Err("binary vector entries must be 0 or 1");
    }
    Ok(())
}

/// Bind two binary hypervectors using XOR.
pub fn xor_bind(left: &[u8], right: &[u8]) -> Result<Vec<u8>, &'static str> {
    validate_binary_vector(left)?;
    validate_binary_vector(right)?;
    if left.len() != right.len() {
        return Err("binary vector dimensions must match");
    }
    Ok(left.iter().zip(right).map(|(a, b)| a ^ b).collect())
}

/// Unbind an XOR-bound vector using either original operand.
pub fn xor_unbind(bound: &[u8], known_operand: &[u8]) -> Result<Vec<u8>, &'static str> {
    xor_bind(bound, known_operand)
}

/// Majority bundle of an odd number of equally sized binary hypervectors.
///
/// Odd cardinality avoids ties and an implicit tie-breaking bias.
pub fn majority_bundle(vectors: &[&[u8]]) -> Result<Vec<u8>, &'static str> {
    if vectors.is_empty() || vectors.len() % 2 == 0 {
        return Err("majority bundle requires a nonzero odd number of vectors");
    }
    for vector in vectors {
        validate_binary_vector(vector)?;
    }
    let dimension = vectors[0].len();
    if vectors.iter().any(|vector| vector.len() != dimension) {
        return Err("all bundle vectors must have equal dimensions");
    }

    let threshold = vectors.len() / 2;
    Ok((0..dimension)
        .map(|index| {
            let ones = vectors.iter().filter(|vector| vector[index] == 1).count();
            u8::from(ones > threshold)
        })
        .collect())
}

/// Hamming distance between binary hypervectors.
pub fn hamming_distance(left: &[u8], right: &[u8]) -> Result<usize, &'static str> {
    validate_binary_vector(left)?;
    validate_binary_vector(right)?;
    if left.len() != right.len() {
        return Err("binary vector dimensions must match");
    }
    Ok(left.iter().zip(right).filter(|(a, b)| a != b).count())
}

/// Deterministic, non-cryptographic binary codeword generator for frozen tests.
pub fn deterministic_codeword(seed: u64, dimension: usize) -> Result<Vec<u8>, &'static str> {
    if dimension == 0 {
        return Err("dimension must be nonzero");
    }
    let mut state = if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    };
    let mut output = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let value = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        output.push((value >> 63) as u8);
    }
    Ok(output)
}


/// Rank of a real 2x2 matrix, used by the observability and excitation controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixRank {
    Zero,
    One,
    Two,
}

/// Rank of the 2x2 observability matrix [C; C A] for a linear state-space model.
pub fn observability_rank_2x2(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    sensor_1: f64,
    sensor_2: f64,
    relative_tolerance: f64,
) -> Result<MatrixRank, &'static str> {
    if !finite(&[a, b, c, d, sensor_1, sensor_2]) {
        return Err("system and sensor entries must be finite");
    }
    if !relative_tolerance.is_finite() || !(0.0..1.0).contains(&relative_tolerance) {
        return Err("relative tolerance must be finite and in [0, 1)");
    }

    let row_2_1 = sensor_1 * a + sensor_2 * c;
    let row_2_2 = sensor_1 * b + sensor_2 * d;
    let entries = [sensor_1, sensor_2, row_2_1, row_2_2];
    if !finite(&[row_2_1, row_2_2]) {
        return Err("observability calculation overflowed");
    }
    let scale = entries
        .iter()
        .fold(0.0_f64, |maximum, entry| maximum.max(entry.abs()));
    if scale == 0.0 {
        return Ok(MatrixRank::Zero);
    }

    // Normalize before the determinant so rank is invariant to unit scaling.
    let normalized_determinant =
        (sensor_1 / scale) * (row_2_2 / scale)
            - (sensor_2 / scale) * (row_2_1 / scale);
    if !normalized_determinant.is_finite() {
        return Err("normalized observability calculation overflowed");
    }

    Ok(if normalized_determinant.abs() > relative_tolerance {
        MatrixRank::Two
    } else {
        MatrixRank::One
    })
}

/// Numerical rank of a two-feature input-regressor matrix via its Gram matrix.
///
/// This is a small persistent-excitation control, not a complete theorem about
/// identifiability for arbitrary nonlinear or time-varying systems.
pub fn input_regressor_rank_2d(
    samples: &[[f64; 2]],
    relative_tolerance: f64,
) -> Result<MatrixRank, &'static str> {
    if !relative_tolerance.is_finite() || !(0.0..1.0).contains(&relative_tolerance) {
        return Err("relative tolerance must be finite and in [0, 1)");
    }
    if samples.iter().flatten().any(|value| !value.is_finite()) {
        return Err("input samples must be finite");
    }

    // Normalize input units before forming the Gram matrix to avoid scale
    // changing the numerical rank decision (and to reduce overflow risk).
    let input_scale = samples
        .iter()
        .flatten()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if input_scale == 0.0 {
        return Ok(MatrixRank::Zero);
    }

    let mut g_11 = 0.0;
    let mut g_12 = 0.0;
    let mut g_22 = 0.0;
    for sample in samples {
        let first = sample[0] / input_scale;
        let second = sample[1] / input_scale;
        g_11 += first * first;
        g_12 += first * second;
        g_22 += second * second;
    }
    let gram_scale = g_11.max(g_12.abs()).max(g_22);
    if !gram_scale.is_finite() {
        return Err("input Gram matrix calculation overflowed");
    }
    if gram_scale == 0.0 {
        return Ok(MatrixRank::Zero);
    }
    let normalized_determinant =
        (g_11 / gram_scale) * (g_22 / gram_scale)
            - (g_12 / gram_scale) * (g_12 / gram_scale);
    if !normalized_determinant.is_finite() {
        return Err("normalized input Gram calculation overflowed");
    }

    Ok(if normalized_determinant > relative_tolerance {
        MatrixRank::Two
    } else {
        MatrixRank::One
    })
}

/// Observations of a diagonal two-state linear system over a finite horizon.
///
/// This deliberately tiny simulator makes hidden-mode non-identifiability
/// controls hand-checkable. It is not a general state-space simulator.
pub fn diagonal_observation_trace(
    visible_eigenvalue: f64,
    hidden_eigenvalue: f64,
    initial_visible: f64,
    initial_hidden: f64,
    sensor_visible: f64,
    sensor_hidden: f64,
    steps: usize,
) -> Result<Vec<f64>, &'static str> {
    if !finite(&[
        visible_eigenvalue,
        hidden_eigenvalue,
        initial_visible,
        initial_hidden,
        sensor_visible,
        sensor_hidden,
    ]) {
        return Err("system, state and sensor values must be finite");
    }
    if steps == 0 {
        return Err("observation horizon must be nonzero");
    }

    let mut visible = initial_visible;
    let mut hidden = initial_hidden;
    let mut observations = Vec::with_capacity(steps);
    for _ in 0..steps {
        observations.push(sensor_visible * visible + sensor_hidden * hidden);
        visible *= visible_eigenvalue;
        hidden *= hidden_eigenvalue;
    }
    if observations.iter().all(|value| value.is_finite()) {
        Ok(observations)
    } else {
        Err("observation simulation overflowed")
    }
}


/// Result of selecting a sensor that separates competing hidden-state models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorSelection {
    /// Zero-based index into the caller's candidate-sensor list.
    pub candidate_index: usize,
    /// Minimum pairwise prediction distance across all model hypotheses.
    pub minimum_pairwise_separation: f64,
    /// Sum of pairwise prediction distances across all model hypotheses.
    pub total_pairwise_separation: f64,
}

/// Select a unit-norm sensor that best separates competing diagonal two-state models.
///
/// All candidate sensors are normalized to unit Euclidean norm, so a sensor
/// cannot win merely by multiplying its gain. The objective is lexicographic:
/// maximize the minimum pairwise prediction separation, then total separation.
/// This is a tiny deterministic active-measurement reference control, not a
/// general Bayesian experimental-design algorithm. Equal sensor noise/cost is
/// assumed; real experiments must model both explicitly.
pub fn select_discriminating_sensor(
    visible_eigenvalue: f64,
    hidden_eigenvalues: &[f64],
    initial_state: [f64; 2],
    candidate_sensors: &[[f64; 2]],
    steps: usize,
    minimum_separation: f64,
) -> Result<Option<SensorSelection>, &'static str> {
    if !finite(&[
        visible_eigenvalue,
        initial_state[0],
        initial_state[1],
        minimum_separation,
    ]) || hidden_eigenvalues.iter().any(|value| !value.is_finite())
        || candidate_sensors.iter().flatten().any(|value| !value.is_finite())
    {
        return Err("experiment-design inputs must be finite");
    }
    if hidden_eigenvalues.len() < 2 {
        return Err("at least two competing hidden models are required");
    }
    if candidate_sensors.is_empty() {
        return Err("at least one candidate sensor is required");
    }
    if steps == 0 {
        return Err("experiment horizon must be nonzero");
    }
    if minimum_separation < 0.0 {
        return Err("minimum separation must be nonnegative");
    }

    let mut best: Option<SensorSelection> = None;
    for (candidate_index, candidate) in candidate_sensors.iter().enumerate() {
        let norm = candidate[0].hypot(candidate[1]);
        if !norm.is_finite() || norm == 0.0 {
            return Err("candidate sensor must have a finite nonzero norm");
        }
        let sensor = [candidate[0] / norm, candidate[1] / norm];
        let mut traces = Vec::with_capacity(hidden_eigenvalues.len());
        for hidden_eigenvalue in hidden_eigenvalues {
            traces.push(diagonal_observation_trace(
                visible_eigenvalue,
                *hidden_eigenvalue,
                initial_state[0],
                initial_state[1],
                sensor[0],
                sensor[1],
                steps,
            )?);
        }

        let mut minimum_pairwise_separation = f64::INFINITY;
        let mut total_pairwise_separation = 0.0;
        for left in 0..traces.len() {
            for right in (left + 1)..traces.len() {
                let mut squared_distance = 0.0;
                for (left_value, right_value) in traces[left].iter().zip(&traces[right]) {
                    let difference = left_value - right_value;
                    squared_distance += difference * difference;
                }
                let distance = squared_distance.sqrt();
                if !distance.is_finite() {
                    return Err("prediction separation overflowed");
                }
                minimum_pairwise_separation =
                    minimum_pairwise_separation.min(distance);
                total_pairwise_separation += distance;
            }
        }
        if !total_pairwise_separation.is_finite() {
            return Err("total prediction separation overflowed");
        }

        let candidate_result = SensorSelection {
            candidate_index,
            minimum_pairwise_separation,
            total_pairwise_separation,
        };
        let should_replace = match best {
            None => true,
            Some(current) => {
                candidate_result.minimum_pairwise_separation
                    > current.minimum_pairwise_separation
                    || (candidate_result.minimum_pairwise_separation
                        == current.minimum_pairwise_separation
                        && candidate_result.total_pairwise_separation
                            > current.total_pairwise_separation)
            }
        };
        if should_replace {
            best = Some(candidate_result);
        }
    }

    Ok(best.filter(|selection| {
        selection.minimum_pairwise_separation > minimum_separation
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) {
        assert!(
            (left - right).abs() <= 1e-12,
            "left={left:.16}, right={right:.16}"
        );
    }

    #[test]
    fn symmetric_matrix_has_expected_eigenvalues() {
        let values = symmetric_eigenvalues_2x2(2.0, 1.0, 2.0).unwrap();
        close(values[0], 1.0);
        close(values[1], 3.0);
    }

    #[test]
    fn symmetric_spectrum_rejects_non_finite_entries() {
        assert!(symmetric_eigenvalues_2x2(f64::NAN, 0.0, 1.0).is_err());
    }

    #[test]
    fn diagonal_discrete_system_is_stable() {
        assert_eq!(
            classify_discrete_stability(0.5, 0.0, 0.0, 0.25, 1e-12).unwrap(),
            DiscreteStability::AsymptoticallyStable
        );
    }

    #[test]
    fn discrete_system_with_unit_root_is_marginal() {
        assert_eq!(
            classify_discrete_stability(1.0, 0.0, 0.0, 0.5, 1e-12).unwrap(),
            DiscreteStability::Marginal
        );
    }

    #[test]
    fn discrete_system_above_unit_radius_is_unstable() {
        assert_eq!(
            classify_discrete_stability(1.25, 0.0, 0.0, 0.5, 1e-12).unwrap(),
            DiscreteStability::Unstable
        );
    }

    #[test]
    fn complex_pair_uses_modulus_not_real_part() {
        close(spectral_radius_2x2(0.0, -0.9, 0.9, 0.0).unwrap(), 0.9);
    }

    #[test]
    fn discrete_stability_rejects_invalid_tolerance() {
        assert!(classify_discrete_stability(0.5, 0.0, 0.0, 0.5, -1.0).is_err());
        assert!(classify_discrete_stability(0.5, 0.0, 0.0, 0.5, f64::NAN).is_err());
    }

    #[test]
    fn continuous_time_hurwitz_control_matches_trace_determinant_criterion() {
        assert!(continuous_time_hurwitz_2x2(-1.0, 0.0, 0.0, -2.0).unwrap());
        assert!(!continuous_time_hurwitz_2x2(1.0, 0.0, 0.0, -2.0).unwrap());
        assert!(!continuous_time_hurwitz_2x2(0.0, 1.0, 1.0, 0.0).unwrap());
    }

    #[test]
    fn adjacent_gap_ratios_match_hand_calculation() {
        let ratios = adjacent_gap_ratios(&[0.0, 1.0, 3.0, 6.0]).unwrap();
        close(ratios[0], 0.5);
        close(ratios[1], 2.0 / 3.0);
    }

    #[test]
    fn adjacent_gap_ratios_are_affine_invariant() {
        let first = adjacent_gap_ratios(&[0.0, 1.0, 3.0, 6.0]).unwrap();
        let transformed = adjacent_gap_ratios(&[11.0, 13.0, 17.0, 23.0]).unwrap();
        for (left, right) in first.iter().zip(transformed) {
            close(*left, right);
        }
    }

    #[test]
    fn adjacent_gap_ratios_reject_ties_and_unsorted_values() {
        assert!(adjacent_gap_ratios(&[0.0, 1.0, 1.0]).is_err());
        assert!(adjacent_gap_ratios(&[0.0, 2.0, 1.0]).is_err());
    }

    #[test]
    fn adjacent_gap_ratios_reject_non_finite_values() {
        assert!(adjacent_gap_ratios(&[0.0, 1.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn prime_sieve_matches_frozen_small_reference() {
        assert_eq!(prime_sieve(30), vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
        assert_eq!(prime_sieve(1), Vec::<usize>::new());
        assert_eq!(prime_sieve(2), vec![2]);
    }

    #[test]
    fn xor_binding_is_self_inverse_and_recovers_operand() {
        let left = [1, 0, 1, 1, 0, 0, 1, 0];
        let right = [0, 1, 1, 0, 0, 1, 0, 1];
        let bound = xor_bind(&left, &right).unwrap();
        assert_eq!(xor_unbind(&bound, &right).unwrap(), left.to_vec());
        assert_eq!(xor_unbind(&bound, &left).unwrap(), right.to_vec());
        assert_eq!(xor_unbind(&bound, &right).unwrap(), left);
    }

    #[test]
    fn majority_bundle_matches_frozen_three_vector_reference() {
        let a = [1, 0, 1, 0, 0, 1];
        let b = [1, 1, 0, 0, 1, 1];
        let c = [0, 1, 1, 1, 1, 0];
        assert_eq!(
            majority_bundle(&[&a, &b, &c]).unwrap(),
            vec![1, 1, 1, 0, 1, 1]
        );
    }

    #[test]
    fn majority_bundle_rejects_even_cardinality() {
        let a = [0, 1, 0];
        let b = [1, 0, 1];
        assert!(majority_bundle(&[&a, &b]).is_err());
    }

    #[test]
    fn hamming_distance_matches_reference() {
        assert_eq!(hamming_distance(&[1, 0, 1], &[0, 0, 1]).unwrap(), 1);
    }

    #[test]
    fn codeword_generation_is_reproducible_and_seed_sensitive() {
        let first = deterministic_codeword(42, 256).unwrap();
        let repeated = deterministic_codeword(42, 256).unwrap();
        let other = deterministic_codeword(43, 256).unwrap();
        assert_eq!(first, repeated);
        assert_ne!(first, other);
        assert!(first.iter().all(|bit| *bit <= 1));
    }


    #[test]
    fn observability_distinguishes_visible_hidden_and_unobservable_modes() {
        assert_eq!(
            observability_rank_2x2(0.5, 0.0, 0.0, 0.2, 1.0, 0.0, 1e-12).unwrap(),
            MatrixRank::One
        );
        assert_eq!(
            observability_rank_2x2(0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1e-12).unwrap(),
            MatrixRank::Two
        );
        assert_eq!(
            observability_rank_2x2(0.5, 0.0, 0.0, 0.2, 1.0, 1.0, 1e-12).unwrap(),
            MatrixRank::Two
        );
        assert_eq!(
            observability_rank_2x2(0.5, 0.0, 0.0, 0.2, 0.0, 0.0, 1e-12).unwrap(),
            MatrixRank::Zero
        );
    }

    #[test]
    fn different_hidden_dynamics_can_be_observationally_identical() {
        let first = diagonal_observation_trace(0.5, 0.2, 1.0, 1.0, 1.0, 0.0, 8).unwrap();
        let second = diagonal_observation_trace(0.5, 0.9, 1.0, 1.0, 1.0, 0.0, 8).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            observability_rank_2x2(0.5, 0.0, 0.0, 0.2, 1.0, 0.0, 1e-12).unwrap(),
            MatrixRank::One
        );
        let first_with_hidden_sensor =
            diagonal_observation_trace(0.5, 0.2, 1.0, 1.0, 0.0, 1.0, 8).unwrap();
        let second_with_hidden_sensor =
            diagonal_observation_trace(0.5, 0.9, 1.0, 1.0, 0.0, 1.0, 8).unwrap();
        assert_ne!(first_with_hidden_sensor, second_with_hidden_sensor);
    }

    #[test]
    fn input_regressor_rank_detects_persistent_excitation_controls() {
        assert_eq!(
            input_regressor_rank_2d(&[[1.0, 0.0], [2.0, 0.0]], 1e-12).unwrap(),
            MatrixRank::One
        );
        assert_eq!(
            input_regressor_rank_2d(&[[1.0, 0.0], [0.0, 1.0]], 1e-12).unwrap(),
            MatrixRank::Two
        );
        assert_eq!(input_regressor_rank_2d(&[], 1e-12).unwrap(), MatrixRank::Zero);
    }


    #[test]
    fn observability_and_excitation_ranks_are_invariant_to_unit_scaling() {
        assert_eq!(
            observability_rank_2x2(0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1e-12).unwrap(),
            MatrixRank::Two
        );
        assert_eq!(
            observability_rank_2x2(0.0, 1.0, -1.0, 0.0, 1e-8, 0.0, 1e-12).unwrap(),
            MatrixRank::Two
        );
        assert_eq!(
            input_regressor_rank_2d(&[[1.0, 0.0], [0.0, 1.0]], 1e-12).unwrap(),
            MatrixRank::Two
        );
        assert_eq!(
            input_regressor_rank_2d(&[[1e-8, 0.0], [0.0, 1e-8]], 1e-12).unwrap(),
            MatrixRank::Two
        );
    }

    #[test]
    fn active_sensor_selection_prefers_sensor_that_separates_hidden_models() {
        let selected = select_discriminating_sensor(
            0.5,
            &[0.2, 0.9],
            [1.0, 1.0],
            &[[1.0, 0.0], [0.0, 5.0]],
            8,
            1e-12,
        )
        .unwrap()
        .unwrap();
        assert_eq!(selected.candidate_index, 1);
        assert!(selected.minimum_pairwise_separation > 0.0);
        assert!(selected.total_pairwise_separation > 0.0);
    }

    #[test]
    fn active_sensor_selection_abstains_when_no_candidate_separates_models() {
        let selected = select_discriminating_sensor(
            0.5,
            &[0.2, 0.9],
            [1.0, 1.0],
            &[[1.0, 0.0]],
            8,
            1e-12,
        )
        .unwrap();
        assert!(selected.is_none());
    }

    #[test]
    fn active_sensor_selection_is_invariant_to_sensor_gain() {
        let selected = select_discriminating_sensor(
            0.5,
            &[0.2, 0.9],
            [1.0, 1.0],
            &[[0.0, 1.0], [0.0, 10.0]],
            8,
            1e-12,
        )
        .unwrap()
        .unwrap();
        assert_eq!(selected.candidate_index, 0);
    }

    #[test]
    fn identifiability_controls_reject_invalid_inputs() {
        assert!(observability_rank_2x2(1.0, 0.0, 0.0, 1.0, f64::NAN, 0.0, 1e-12).is_err());
        assert!(input_regressor_rank_2d(&[[1.0, f64::INFINITY]], 1e-12).is_err());
        assert!(input_regressor_rank_2d(&[[1.0, 0.0]], -1.0).is_err());
        assert!(diagonal_observation_trace(0.5, 0.2, 1.0, 1.0, 1.0, 0.0, 0).is_err());
    }

    #[test]
    fn vector_operations_reject_invalid_inputs() {
        assert!(xor_bind(&[], &[]).is_err());
        assert!(xor_bind(&[0, 1], &[0]).is_err());
        assert!(xor_bind(&[0, 2], &[0, 1]).is_err());
        assert!(majority_bundle(&[]).is_err());
        assert!(deterministic_codeword(1, 0).is_err());
    }
}
