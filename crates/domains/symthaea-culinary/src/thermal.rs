//! A time–temperature trajectory and the two things Phase 1 asks of it: its peak
//! temperature (for coagulation windows) and the microbial log-reduction it
//! delivers (for pasteurization).
//!
//! Phase 1 treats the trajectory as *given* — "does this curve pasteurize?" — and
//! does not ask whether the curve is physically *achievable* (that heat-transfer
//! question is Phase 2, where `symthaea-thermofluids` becomes load-bearing).

/// A malformed time-temperature trajectory or invalid lethality calculation input.
#[derive(Clone, Debug, PartialEq)]
pub enum ThermalTrajectoryError {
    Empty,
    TooFewPoints { actual: usize },
    NonFiniteTime { index: usize },
    NegativeTime { index: usize, time_minutes: f64 },
    NonIncreasingTime {
        index: usize,
        previous_minutes: f64,
        current_minutes: f64,
    },
    NonFiniteTemperature { index: usize },
    BelowAbsoluteZero { index: usize, temperature_c: f64 },
    InvalidDRef { d_ref: f64 },
    InvalidReferenceTemperature { t_ref: f64 },
    InvalidZ { z: f64 },
    NonFiniteResult,
}

impl std::fmt::Display for ThermalTrajectoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "thermal trajectory is empty"),
            Self::TooFewPoints { actual } => write!(
                f,
                "thermal trajectory needs at least 2 points, got {actual}"
            ),
            Self::NonFiniteTime { index } => {
                write!(f, "thermal trajectory point {index} has non-finite time")
            }
            Self::NegativeTime { index, time_minutes } => write!(
                f,
                "thermal trajectory point {index} has negative time {time_minutes} min"
            ),
            Self::NonIncreasingTime {
                index,
                previous_minutes,
                current_minutes,
            } => write!(
                f,
                "thermal trajectory time must strictly increase at point {index}: {previous_minutes} min then {current_minutes} min"
            ),
            Self::NonFiniteTemperature { index } => write!(
                f,
                "thermal trajectory point {index} has non-finite temperature"
            ),
            Self::BelowAbsoluteZero {
                index,
                temperature_c,
            } => write!(
                f,
                "thermal trajectory point {index} is below absolute zero: {temperature_c} °C"
            ),
            Self::InvalidDRef { d_ref } => {
                write!(f, "D-reference time must be finite and positive, got {d_ref}")
            }
            Self::InvalidReferenceTemperature { t_ref } => write!(
                f,
                "reference temperature must be finite, got {t_ref} °C"
            ),
            Self::InvalidZ { z } => {
                write!(f, "z-value must be finite and positive, got {z} °C")
            }
            Self::NonFiniteResult => {
                write!(f, "thermal lethality calculation produced a non-finite result")
            }
        }
    }
}

impl std::error::Error for ThermalTrajectoryError {}

/// A trajectory of `(time_minutes, core_temperature_celsius)` points, assumed
/// piecewise-linear in temperature between consecutive samples. The validated
/// constructor requires finite samples and strictly increasing, non-negative times.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ThermalTrajectory {
    pub points: Vec<(f64, f64)>,
}

impl ThermalTrajectory {
    /// Construct a trajectory after validating its samples.
    ///
    /// Times are expressed in minutes and temperatures in degrees Celsius.
    /// Source order is preserved: samples are never silently sorted, merged,
    /// clamped or imputed. Duplicate timestamps are rejected because a
    /// piecewise-linear segment would otherwise have zero duration.
    pub fn try_new(
        points: Vec<(f64, f64)>,
    ) -> Result<Self, ThermalTrajectoryError> {
        validate_points(&points)?;
        Ok(Self { points })
    }

    pub fn new(points: Vec<(f64, f64)>) -> Self {
        Self { points }
    }

    /// A flat hold at `temp_c` for `minutes` (two points).
    pub fn hold(temp_c: f64, minutes: f64) -> Self {
        Self {
            points: vec![(0.0, temp_c), (minutes, temp_c)],
        }
    }

    /// Highest temperature reached anywhere on the trajectory.
    pub fn peak_temp(&self) -> f64 {
        self.points
            .iter()
            .map(|&(_, t)| t)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Total decimal (log10) reduction using a fallible, validated calculation.
    ///
    /// Rejects malformed trajectories and invalid D/z inputs. This calculation
    /// is a literature-parameterized model, not a food-safety determination.
    pub fn try_log_reduction(
        &self,
        d_ref: f64,
        t_ref: f64,
        z: f64,
    ) -> Result<f64, ThermalTrajectoryError> {
        validate_points(&self.points)?;
        if !d_ref.is_finite() || d_ref <= 0.0 {
            return Err(ThermalTrajectoryError::InvalidDRef { d_ref });
        }
        if !t_ref.is_finite() {
            return Err(ThermalTrajectoryError::InvalidReferenceTemperature { t_ref });
        }
        if !z.is_finite() || z <= 0.0 {
            return Err(ThermalTrajectoryError::InvalidZ { z });
        }

        let lethality = |temp: f64| -> f64 {
            10f64.powf((temp - t_ref) / z) / d_ref
        };
        let mut total = 0.0;
        const SUBSTEPS: usize = 64;
        for w in self.points.windows(2) {
            let (t0, temp0) = w[0];
            let (t1, temp1) = w[1];
            let h = (t1 - t0) / SUBSTEPS as f64;
            for k in 0..SUBSTEPS {
                let f0 = (k as f64) / SUBSTEPS as f64;
                let f1 = (k as f64 + 1.0) / SUBSTEPS as f64;
                let a = lethality(temp0 + (temp1 - temp0) * f0);
                let b = lethality(temp0 + (temp1 - temp0) * f1);
                total += 0.5 * (a + b) * h;
            }
        }
        if !total.is_finite() {
            return Err(ThermalTrajectoryError::NonFiniteResult);
        }
        Ok(total)
    }

    /// Backwards-compatible legacy API.
    ///
    /// New callers should use try_log_reduction and handle its typed error.
    /// This wrapper preserves the historical zero fallback for invalid inputs
    /// so existing research callers are not source-broken.
    pub fn log_reduction(&self, d_ref: f64, t_ref: f64, z: f64) -> f64 {
        self.try_log_reduction(d_ref, t_ref, z).unwrap_or(0.0)
    }
}


/// Validate trajectory samples without changing their order or contents.
fn validate_points(points: &[(f64, f64)]) -> Result<(), ThermalTrajectoryError> {
    if points.is_empty() {
        return Err(ThermalTrajectoryError::Empty);
    }
    if points.len() < 2 {
        return Err(ThermalTrajectoryError::TooFewPoints {
            actual: points.len(),
        });
    }

    for (index, &(time_minutes, temperature_c)) in points.iter().enumerate() {
        if !time_minutes.is_finite() {
            return Err(ThermalTrajectoryError::NonFiniteTime { index });
        }
        if time_minutes < 0.0 {
            return Err(ThermalTrajectoryError::NegativeTime {
                index,
                time_minutes,
            });
        }
        if !temperature_c.is_finite() {
            return Err(ThermalTrajectoryError::NonFiniteTemperature { index });
        }
        if temperature_c < -273.15 {
            return Err(ThermalTrajectoryError::BelowAbsoluteZero {
                index,
                temperature_c,
            });
        }
        if index > 0 {
            let previous_minutes = points[index - 1].0;
            if time_minutes <= previous_minutes {
                return Err(ThermalTrajectoryError::NonIncreasingTime {
                    index,
                    previous_minutes,
                    current_minutes: time_minutes,
                });
            }
        }
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_of_a_ramp() {
        let t = ThermalTrajectory::new(vec![(0.0, 20.0), (5.0, 72.0), (10.0, 60.0)]);
        assert_eq!(t.peak_temp(), 72.0);
    }

    #[test]
    fn hold_at_reference_gives_expected_decades() {
        // At exactly the reference temperature, D = d_ref, so t minutes of hold
        // delivers t / d_ref decades. 0.396-min D, 3.96-min hold ⇒ 10 log.
        let t = ThermalTrajectory::hold(60.0, 3.96);
        let lr = t.log_reduction(0.396, 60.0, 5.56);
        assert!((lr - 10.0).abs() < 0.05, "expected ~10 log, got {lr}");
    }

    #[test]
    fn lower_temperature_is_far_less_lethal() {
        // 5 °C below reference with z=5.56 ⇒ D is ~10^(5/5.56) ≈ 7.9× longer,
        // so the same hold delivers ~1/7.9 the decades.
        let hot = ThermalTrajectory::hold(60.0, 10.0).log_reduction(0.396, 60.0, 5.56);
        let cool = ThermalTrajectory::hold(55.0, 10.0).log_reduction(0.396, 60.0, 5.56);
        assert!(cool < hot / 5.0, "cool={cool} hot={hot}");
    }

    #[test]
    fn try_new_accepts_finite_strictly_increasing_samples() {
        let trajectory = ThermalTrajectory::try_new(vec![(0.0, 20.0), (2.0, 60.0)]);
        assert!(trajectory.is_ok());
    }

    #[test]
    fn try_new_rejects_empty_and_single_sample_trajectories() {
        assert!(matches!(
            ThermalTrajectory::try_new(vec![]),
            Err(ThermalTrajectoryError::Empty)
        ));
        assert!(matches!(
            ThermalTrajectory::try_new(vec![(0.0, 20.0)]),
            Err(ThermalTrajectoryError::TooFewPoints { actual: 1 })
        ));
    }

    #[test]
    fn try_new_rejects_duplicate_and_decreasing_timestamps() {
        assert!(matches!(
            ThermalTrajectory::try_new(vec![(0.0, 20.0), (0.0, 21.0)]),
            Err(ThermalTrajectoryError::NonIncreasingTime { .. })
        ));
        assert!(matches!(
            ThermalTrajectory::try_new(vec![(1.0, 20.0), (0.5, 21.0)]),
            Err(ThermalTrajectoryError::NonIncreasingTime { .. })
        ));
    }

    #[test]
    fn try_new_rejects_non_finite_and_negative_samples() {
        assert!(matches!(
            ThermalTrajectory::try_new(vec![(0.0, 20.0), (f64::NAN, 21.0)]),
            Err(ThermalTrajectoryError::NonFiniteTime { index: 1 })
        ));
        assert!(matches!(
            ThermalTrajectory::try_new(vec![(0.0, 20.0), (1.0, f64::INFINITY)]),
            Err(ThermalTrajectoryError::NonFiniteTemperature { index: 1 })
        ));
        assert!(matches!(
            ThermalTrajectory::try_new(vec![(0.0, 20.0), (1.0, -300.0)]),
            Err(ThermalTrajectoryError::BelowAbsoluteZero { index: 1, .. })
        ));
        assert!(matches!(
            ThermalTrajectory::try_new(vec![(0.0, 20.0), (-1.0, 21.0)]),
            Err(ThermalTrajectoryError::NegativeTime { index: 1, .. })
        ));
    }

    #[test]
    fn try_log_reduction_rejects_invalid_kinetic_parameters() {
        let trajectory = ThermalTrajectory::hold(60.0, 5.0);
        assert!(matches!(
            trajectory.try_log_reduction(0.0, 60.0, 5.56),
            Err(ThermalTrajectoryError::InvalidDRef { .. })
        ));
        assert!(matches!(
            trajectory.try_log_reduction(0.396, f64::NAN, 5.56),
            Err(ThermalTrajectoryError::InvalidReferenceTemperature { .. })
        ));
        assert!(matches!(
            trajectory.try_log_reduction(0.396, 60.0, f64::INFINITY),
            Err(ThermalTrajectoryError::InvalidZ { .. })
        ));
    }

    #[test]
    fn try_log_reduction_revalidates_unchecked_legacy_values() {
        let malformed = ThermalTrajectory::new(vec![(0.0, 20.0), (0.0, 60.0)]);
        assert!(matches!(
            malformed.try_log_reduction(0.396, 60.0, 5.56),
            Err(ThermalTrajectoryError::NonIncreasingTime { .. })
        ));
    }

    #[test]
    fn try_log_reduction_matches_legacy_result_for_valid_hold() {
        let trajectory = ThermalTrajectory::hold(60.0, 3.96);
        let safe = trajectory.try_log_reduction(0.396, 60.0, 5.56).unwrap();
        let legacy = trajectory.log_reduction(0.396, 60.0, 5.56);
        assert!((safe - legacy).abs() < 1e-12);
        assert!((safe - 10.0).abs() < 0.05);
    }

}
