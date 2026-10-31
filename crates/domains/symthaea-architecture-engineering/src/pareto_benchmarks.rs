//! Deterministic canonical problem generators for Pareto-search evaluation.
//!
//! v0.1 starts with ZDT1 (two objectives) and DTLZ2 (three objectives). Both return
//! minimization objectives, matching the canonical benchmark definitions. Reference-front
//! samplers are deterministic and intended for indicator evaluation, not optimization.

use std::error::Error;
use std::fmt;

const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParetoBenchmarkError {
    pub path: String,
    pub message: String,
}

impl ParetoBenchmarkError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ParetoBenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for ParetoBenchmarkError {}

/// Canonical ZDT1 two-objective minimization problem.
///
/// The decision vector must contain at least two values in [0, 1].
/// The analytic Pareto front is f2 = 1 - sqrt(f1), with f1 in [0, 1].
pub fn zdt1_objectives(decision: &[f64]) -> Result<[f64; 2], ParetoBenchmarkError> {
    validate_decision(decision, 2, "zdt1.decision")?;
    let first = decision[0];
    let tail_mean = decision[1..].iter().sum::<f64>() / (decision.len() - 1) as f64;
    let g = 1.0 + 9.0 * tail_mean;
    let second = g * (1.0 - (first / g).sqrt());
    if !g.is_finite() || !second.is_finite() {
        return Err(ParetoBenchmarkError::new(
            "zdt1.objectives",
            "objective calculation overflowed",
        ));
    }
    Ok([first, second])
}

/// Deterministic ZDT1 analytic Pareto-front sample, including both endpoints.
pub fn zdt1_reference_front(samples: usize) -> Result<Vec<Vec<f64>>, ParetoBenchmarkError> {
    if samples < 2 {
        return Err(ParetoBenchmarkError::new(
            "zdt1.reference_front.samples",
            "at least two reference samples are required",
        ));
    }
    let denominator = (samples - 1) as f64;
    Ok((0..samples)
        .map(|index| {
            let first = index as f64 / denominator;
            vec![first, 1.0 - first.sqrt()]
        })
        .collect())
}

/// Canonical three-objective DTLZ2 minimization problem.
///
/// The decision vector must contain at least three values in [0, 1]. The returned objectives
/// lie on the unit sphere in the positive octant when g=0.
pub fn dtlz2_three_objectives(
    decision: &[f64],
) -> Result<[f64; 3], ParetoBenchmarkError> {
    validate_decision(decision, 3, "dtlz2.decision")?;
    let g = decision[2..]
        .iter()
        .map(|value| (value - 0.5).powi(2))
        .sum::<f64>();
    let theta = decision[0] * PI * 0.5;
    let phi = decision[1] * PI * 0.5;
    let scale = 1.0 + g;
    let objectives = [
        scale * theta.cos() * phi.cos(),
        scale * theta.cos() * phi.sin(),
        scale * theta.sin(),
    ];
    if !g.is_finite() || objectives.iter().any(|value| !value.is_finite()) {
        return Err(ParetoBenchmarkError::new(
            "dtlz2.objectives",
            "objective calculation overflowed",
        ));
    }
    Ok(objectives)
}

/// Deterministic grid over the analytic three-objective DTLZ2 Pareto front.
///
/// Samples the positive unit sphere using z and azimuth coordinates. The north-pole point
/// is emitted once rather than duplicated for every azimuth. The grid is suitable as a
/// fixed IGD+/epsilon reference set; it is not claimed to be an exactly uniform geodesic mesh.
pub fn dtlz2_three_objective_reference_front(
    steps: usize,
) -> Result<Vec<Vec<f64>>, ParetoBenchmarkError> {
    if steps < 2 {
        return Err(ParetoBenchmarkError::new(
            "dtlz2.reference_front.steps",
            "at least two grid steps are required",
        ));
    }
    let denominator = (steps - 1) as f64;
    let mut front = Vec::with_capacity(steps * (steps - 1) + 1);
    for z_index in 0..steps {
        let z = z_index as f64 / denominator;
        let radius = (1.0 - z * z).sqrt();
        let azimuth_steps = if z_index == steps - 1 { 1 } else { steps };
        for azimuth_index in 0..azimuth_steps {
            let azimuth = (azimuth_index as f64 / denominator) * PI * 0.5;
            front.push(vec![
                radius * azimuth.cos(),
                radius * azimuth.sin(),
                z,
            ]);
        }
    }
    Ok(front)
}

fn validate_decision(
    decision: &[f64],
    minimum_dimensions: usize,
    path: &str,
) -> Result<(), ParetoBenchmarkError> {
    if decision.len() < minimum_dimensions {
        return Err(ParetoBenchmarkError::new(
            format!("{path}.len"),
            format!("decision vector requires at least {minimum_dimensions} values"),
        ));
    }
    for (index, value) in decision.iter().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(value) {
            return Err(ParetoBenchmarkError::new(
                format!("{path}[{index}]"),
                "decision values must be finite and in [0, 1]",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zdt1_known_decision_and_reference_endpoints() {
        let objectives = zdt1_objectives(&[0.5, 0.5]).unwrap();
        let g = 1.0 + 9.0 * 0.5;
        let expected_second = g * (1.0 - (0.5 / g).sqrt());
        assert!((objectives[0] - 0.5).abs() < 1e-12);
        assert!((objectives[1] - expected_second).abs() < 1e-12);

        let reference = zdt1_reference_front(3).unwrap();
        assert_eq!(reference.len(), 3);
        assert_eq!(reference[0], vec![0.0, 1.0]);
        assert_eq!(reference[2], vec![1.0, 0.0]);
    }

    #[test]
    fn dtlz2_three_objective_front_has_unit_sphere_geometry_when_g_is_zero() {
        let objectives = dtlz2_three_objectives(&[0.5, 0.5, 0.5]).unwrap();
        let squared_norm = objectives.iter().map(|value| value * value).sum::<f64>();
        assert!((squared_norm - 1.0).abs() < 1e-12);

        let front = dtlz2_three_objective_reference_front(5).unwrap();
        assert_eq!(front.len(), 21);
        for point in front {
            let norm = point.iter().map(|value| value * value).sum::<f64>();
            assert!((norm - 1.0).abs() < 1e-12);
            assert!(point.iter().all(|value| *value >= 0.0));
        }
    }

    #[test]
    fn invalid_decisions_and_reference_grid_sizes_are_rejected() {
        assert!(zdt1_objectives(&[0.5]).is_err());
        assert!(zdt1_objectives(&[0.5, 1.1]).is_err());
        assert!(dtlz2_three_objectives(&[0.5, 0.5]).is_err());
        assert!(dtlz2_three_objectives(&[0.5, f64::NAN, 0.5]).is_err());
        assert!(zdt1_reference_front(1).is_err());
        assert!(dtlz2_three_objective_reference_front(1).is_err());
    }
}
