// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Distribution and inequality metrics for non-negative observations.

use crate::error::{EconomicsError, Result, ensure_finite, ensure_slice_finite};

fn validated_scaled(values: &[f64], context: &'static str) -> Result<Vec<f64>> {
    if values.is_empty() {
        return Err(EconomicsError::EmptyInput { context });
    }
    ensure_slice_finite(values, context)?;
    if values.iter().any(|value| *value < 0.0) {
        return Err(EconomicsError::InvalidParameter {
            context: "inequality observations must be non-negative",
        });
    }
    let scale = values.iter().copied().fold(0.0_f64, f64::max);
    if scale == 0.0 {
        Ok(vec![0.0; values.len()])
    } else {
        Ok(values.iter().map(|value| value / scale).collect())
    }
}

/// Gini coefficient for a non-negative distribution.
///
/// The finite-sample maximum is `(n-1)/n`; use [`normalized_gini`] when a
/// concentration in one observation should map to exactly `1.0`.
pub fn gini(values: &[f64]) -> Result<f64> {
    let mut sorted = validated_scaled(values, "Gini observations")?;
    let scale = sorted.iter().copied().fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Ok(0.0);
    }
    sorted.sort_by(f64::total_cmp);

    let n = sorted.len() as f64;
    let sum: f64 = sorted.iter().sum();
    let weighted: f64 = sorted
        .iter()
        .enumerate()
        .map(|(index, value)| (index as f64 + 1.0) * value)
        .sum();
    let result = (2.0 * weighted - (n + 1.0) * sum) / (n * sum);
    if result.is_finite() {
        Ok(result.clamp(0.0, 1.0))
    } else {
        Err(EconomicsError::NumericalFailure {
            context: "Gini calculation failed",
        })
    }
}

/// Finite-sample normalized Gini in `[0, 1]`.
pub fn normalized_gini(values: &[f64]) -> Result<f64> {
    if values.len() <= 1 {
        // Still validate the observation rather than accepting NaN or negative.
        validated_scaled(values, "normalized Gini observations")?;
        return Ok(0.0);
    }
    let maximum = (values.len() - 1) as f64 / values.len() as f64;
    Ok((gini(values)? / maximum).clamp(0.0, 1.0))
}

/// One point on a Lorenz curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LorenzPoint {
    pub population_share: f64,
    pub value_share: f64,
}

/// Lorenz curve including `(0,0)` and `(1,1)`.
///
/// An all-zero population is represented by the equality line because every
/// observation is identical and aggregate shares would otherwise be `0/0`.
pub fn lorenz_curve(values: &[f64]) -> Result<Vec<LorenzPoint>> {
    let mut sorted = validated_scaled(values, "Lorenz-curve observations")?;
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let sum: f64 = sorted.iter().sum();
    let mut curve = Vec::with_capacity(n + 1);
    curve.push(LorenzPoint {
        population_share: 0.0,
        value_share: 0.0,
    });
    if sum == 0.0 {
        for index in 1..=n {
            let share = index as f64 / n as f64;
            curve.push(LorenzPoint {
                population_share: share,
                value_share: share,
            });
        }
        return Ok(curve);
    }

    let mut cumulative = 0.0;
    for (index, value) in sorted.into_iter().enumerate() {
        cumulative += value;
        curve.push(LorenzPoint {
            population_share: (index + 1) as f64 / n as f64,
            value_share: (cumulative / sum).clamp(0.0, 1.0),
        });
    }
    if let Some(last) = curve.last_mut() {
        last.population_share = 1.0;
        last.value_share = 1.0;
    }
    Ok(curve)
}

/// Hoover (Robin Hood) index: share of total value requiring redistribution to
/// reach equality.
pub fn hoover_index(values: &[f64]) -> Result<f64> {
    let scaled = validated_scaled(values, "Hoover-index observations")?;
    let sum: f64 = scaled.iter().sum();
    if sum == 0.0 {
        return Ok(0.0);
    }
    let mean = sum / scaled.len() as f64;
    let absolute_deviation: f64 = scaled.iter().map(|value| (value - mean).abs()).sum();
    Ok((0.5 * absolute_deviation / sum).clamp(0.0, 1.0))
}

/// Theil T index. Zero means equality; larger values indicate concentration.
pub fn theil_t(values: &[f64]) -> Result<f64> {
    let scaled = validated_scaled(values, "Theil-index observations")?;
    let sum: f64 = scaled.iter().sum();
    if sum == 0.0 {
        return Ok(0.0);
    }
    let mean = sum / scaled.len() as f64;
    let index = scaled
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| {
            let ratio = value / mean;
            ratio * ratio.ln()
        })
        .sum::<f64>()
        / scaled.len() as f64;
    if index.is_finite() {
        Ok(index.max(0.0))
    } else {
        Err(EconomicsError::NumericalFailure {
            context: "Theil-index calculation failed",
        })
    }
}

/// Atkinson index with inequality-aversion parameter `epsilon >= 0`.
///
/// `epsilon = 0` assigns no inequality penalty. `epsilon = 1` uses the
/// geometric-mean limit. Any zero observation produces maximal inequality when
/// `epsilon >= 1`.
pub fn atkinson_index(values: &[f64], epsilon: f64) -> Result<f64> {
    ensure_finite(epsilon, "Atkinson epsilon")?;
    if epsilon < 0.0 {
        return Err(EconomicsError::InvalidParameter {
            context: "Atkinson epsilon must be non-negative",
        });
    }
    let scaled = validated_scaled(values, "Atkinson-index observations")?;
    let sum: f64 = scaled.iter().sum();
    if sum == 0.0 || epsilon == 0.0 {
        return Ok(0.0);
    }
    let mean = sum / scaled.len() as f64;
    let equally_distributed_equivalent = if (epsilon - 1.0).abs() <= 1e-12 {
        if scaled.contains(&0.0) {
            0.0
        } else {
            (scaled.iter().map(|value| value.ln()).sum::<f64>() / scaled.len() as f64).exp()
        }
    } else if epsilon > 1.0 && scaled.contains(&0.0) {
        0.0
    } else {
        let power = 1.0 - epsilon;
        let moment =
            scaled.iter().map(|value| value.powf(power)).sum::<f64>() / scaled.len() as f64;
        moment.powf(1.0 / power)
    };
    let result = 1.0 - equally_distributed_equivalent / mean;
    if result.is_finite() {
        Ok(result.clamp(0.0, 1.0))
    } else {
        Err(EconomicsError::NumericalFailure {
            context: "Atkinson-index calculation failed",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_equality_is_zero_across_metrics() {
        let values = [1.0, 1.0, 1.0, 1.0];
        assert!(gini(&values).unwrap().abs() < 1e-12);
        assert!(normalized_gini(&values).unwrap().abs() < 1e-12);
        assert!(hoover_index(&values).unwrap().abs() < 1e-12);
        assert!(theil_t(&values).unwrap().abs() < 1e-12);
        assert!(atkinson_index(&values, 1.0).unwrap().abs() < 1e-12);
    }

    #[test]
    fn total_concentration_normalizes_to_one() {
        let values = [0.0, 0.0, 0.0, 1.0];
        assert!((gini(&values).unwrap() - 0.75).abs() < 1e-12);
        assert!((normalized_gini(&values).unwrap() - 1.0).abs() < 1e-12);
        assert!((hoover_index(&values).unwrap() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn lorenz_curve_has_truthful_endpoints() {
        let curve = lorenz_curve(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(curve.first().unwrap().population_share, 0.0);
        assert_eq!(curve.first().unwrap().value_share, 0.0);
        assert_eq!(curve.last().unwrap().population_share, 1.0);
        assert_eq!(curve.last().unwrap().value_share, 1.0);
        assert!(
            curve
                .windows(2)
                .all(|pair| pair[0].value_share <= pair[1].value_share)
        );
    }

    #[test]
    fn progressive_transfer_reduces_major_metrics() {
        let before = [0.0, 0.0, 0.0, 100.0];
        let after = [0.0, 0.0, 10.0, 90.0];
        assert!(gini(&after).unwrap() < gini(&before).unwrap());
        assert!(theil_t(&after).unwrap() < theil_t(&before).unwrap());
        assert!(atkinson_index(&after, 0.5).unwrap() < atkinson_index(&before, 0.5).unwrap());
    }

    #[test]
    fn invalid_observations_are_not_dropped() {
        assert!(gini(&[-10.0, 10.0, 10.0]).is_err());
        assert!(gini(&[f64::NAN, 1.0, 1.0]).is_err());
        assert!(gini(&[]).is_err());
    }

    #[test]
    fn scale_and_order_invariant() {
        let first = [5.0, 1.0, 3.0, 2.0, 4.0];
        let second = [10.0, 20.0, 30.0, 40.0, 50.0];
        assert!((gini(&first).unwrap() - gini(&second).unwrap()).abs() < 1e-12);
        assert!((theil_t(&first).unwrap() - theil_t(&second).unwrap()).abs() < 1e-12);
    }
}
