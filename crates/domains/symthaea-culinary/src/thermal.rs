//! A time–temperature trajectory and the two things Phase 1 asks of it: its peak
//! temperature (for coagulation windows) and the microbial log-reduction it
//! delivers (for pasteurization).
//!
//! Phase 1 treats the trajectory as *given* — "does this curve pasteurize?" — and
//! does not ask whether the curve is physically *achievable* (that heat-transfer
//! question is Phase 2, where `symthaea-thermofluids` becomes load-bearing).

/// A trajectory of `(time_minutes, core_temperature_celsius)` points, assumed
/// piecewise-linear in temperature between consecutive samples. Times must be
/// non-decreasing.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ThermalTrajectory {
    pub points: Vec<(f64, f64)>,
}

impl ThermalTrajectory {
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

    /// Total decimal (log₁₀) reduction of a micro-organism with decimal-reduction
    /// time `d_ref` minutes at `t_ref` °C and z-value `z` °C, delivered by this
    /// trajectory:  ∫ dt / D(T(t)),  where  D(T) = d_ref · 10^((t_ref − T)/z).
    ///
    /// Equivalently the integral of the lethality rate 10^((T − t_ref)/z) / d_ref.
    /// Each linear segment is sub-sampled so the exponential is integrated accurately.
    pub fn log_reduction(&self, d_ref: f64, t_ref: f64, z: f64) -> f64 {
        if d_ref <= 0.0 || z <= 0.0 {
            return 0.0;
        }
        let lethality = |temp: f64| -> f64 { 10f64.powf((temp - t_ref) / z) / d_ref };
        let mut total = 0.0;
        const SUBSTEPS: usize = 64;
        for w in self.points.windows(2) {
            let (t0, temp0) = w[0];
            let (t1, temp1) = w[1];
            let dt = t1 - t0;
            if dt <= 0.0 {
                continue;
            }
            let h = dt / SUBSTEPS as f64;
            for k in 0..SUBSTEPS {
                let f0 = (k as f64) / SUBSTEPS as f64;
                let f1 = (k as f64 + 1.0) / SUBSTEPS as f64;
                let a = lethality(temp0 + (temp1 - temp0) * f0);
                let b = lethality(temp0 + (temp1 - temp0) * f1);
                total += 0.5 * (a + b) * h; // trapezoid
            }
        }
        total
    }
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
}
