use super::*;

/// Maximum integration work accepted by the general conjecture-engine sampler.
const MAX_RK4_STEPS_PER_TRAJECTORY: f64 = 100_000.0;
const MAX_RK4_STATE_DIMENSION: usize = 4_096;
const MAX_RK4_STATE_UPDATES_PER_TRAJECTORY: f64 = 10_000_000.0;

/// Counters emitted by the shared RK4 integrator. Failed runs retain the work
/// already performed, so rejected integrations are not silently counted as free.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Rk4TrajectoryWorkV1 {
    pub(crate) attempted_steps: u64,
    pub(crate) completed_steps: u64,
    pub(crate) rhs_evaluations: u64,
    pub(crate) state_updates: u64,
    pub(crate) completed: bool,
}

fn evaluate_rk4_rhs(
    f: &impl Fn(&[f64], f64) -> Vec<f64>,
    state: &[f64],
    time: f64,
    dimension: usize,
    work: &mut Rk4TrajectoryWorkV1,
) -> Option<Vec<f64>> {
    if state.len() != dimension
        || state.iter().any(|value| !value.is_finite())
        || !time.is_finite()
    {
        return None;
    }
    work.rhs_evaluations += 1;
    let output = f(state, time);
    if output.len() != dimension || output.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(output)
}

/// Historical pair-returning wrapper retained for existing callers.
pub(crate) fn rk4_trajectory(
    f: impl Fn(&[f64], f64) -> Vec<f64>,
    y0: &[f64],
    t_end: f64,
    dt: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let (times, states, _work) = rk4_trajectory_with_work(f, y0, t_end, dt);
    (times, states)
}

/// Fixed-step fourth-order Runge–Kutta trajectory generator (RK4), with exact
/// counts for attempted/completed steps, RHS evaluations, and state updates.
///
/// Invalid inputs, excessive work, malformed/non-finite RHS values, or numerical
/// overflow return an empty trajectory with the work completed before rejection.
pub(crate) fn rk4_trajectory_with_work(
    f: impl Fn(&[f64], f64) -> Vec<f64>,
    y0: &[f64],
    t_end: f64,
    dt: f64,
) -> (Vec<f64>, Vec<Vec<f64>>, Rk4TrajectoryWorkV1) {
    let mut work = Rk4TrajectoryWorkV1::default();
    let dimension = y0.len();
    if dimension == 0
        || dimension > MAX_RK4_STATE_DIMENSION
        || y0.iter().any(|value| !value.is_finite())
        || !t_end.is_finite()
        || t_end < 0.0
        || !dt.is_finite()
        || dt <= 0.0
    {
        return (Vec::new(), Vec::new(), work);
    }
    if t_end == 0.0 {
        work.completed = true;
        return (vec![0.0], vec![y0.to_vec()], work);
    }

    let estimated_steps = (t_end / dt).ceil();
    let estimated_updates = estimated_steps * dimension as f64;
    if !estimated_steps.is_finite()
        || estimated_steps < 1.0
        || estimated_steps > MAX_RK4_STEPS_PER_TRAJECTORY
        || estimated_updates > MAX_RK4_STATE_UPDATES_PER_TRAJECTORY
    {
        return (Vec::new(), Vec::new(), work);
    }
    let max_steps = estimated_steps as usize;
    let mut state = y0.to_vec();
    let mut times = Vec::with_capacity(max_steps.saturating_add(1));
    let mut states = Vec::with_capacity(max_steps.saturating_add(1));
    times.push(0.0);
    states.push(state.clone());

    for step in 0..max_steps {
        let t = (step as f64 * dt).min(t_end);
        if t >= t_end {
            break;
        }
        let next_t = if step + 1 == max_steps {
            t_end
        } else {
            ((step + 1) as f64 * dt).min(t_end)
        };
        let h = next_t - t;
        if !h.is_finite() || h <= 0.0 || t + h <= t {
            return (Vec::new(), Vec::new(), work);
        }
        work.attempted_steps += 1;

        let Some(k1) = evaluate_rk4_rhs(&f, &state, t, dimension, &mut work) else {
            return (Vec::new(), Vec::new(), work);
        };
        let y2: Vec<f64> = (0..dimension).map(|i| state[i] + h * 0.5 * k1[i]).collect();
        let Some(k2) = evaluate_rk4_rhs(&f, &y2, t + 0.5 * h, dimension, &mut work) else {
            return (Vec::new(), Vec::new(), work);
        };
        let y3: Vec<f64> = (0..dimension).map(|i| state[i] + h * 0.5 * k2[i]).collect();
        let Some(k3) = evaluate_rk4_rhs(&f, &y3, t + 0.5 * h, dimension, &mut work) else {
            return (Vec::new(), Vec::new(), work);
        };
        let y4: Vec<f64> = (0..dimension).map(|i| state[i] + h * k3[i]).collect();
        let Some(k4) = evaluate_rk4_rhs(&f, &y4, t + h, dimension, &mut work) else {
            return (Vec::new(), Vec::new(), work);
        };

        for i in 0..dimension {
            state[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        // Count the state arithmetic actually performed, including a step that
        // overflows and is then rejected before it can be marked completed.
        work.state_updates = work.state_updates.saturating_add(dimension as u64);
        if state.iter().any(|value| !value.is_finite()) {
            return (Vec::new(), Vec::new(), work);
        }
        work.completed_steps += 1;
        times.push(next_t);
        states.push(state.clone());
        if next_t == t_end {
            break;
        }
    }

    if times.last().copied() != Some(t_end) {
        return (Vec::new(), Vec::new(), work);
    }
    work.completed = true;
    (times, states, work)
}

pub(crate) fn lorenz_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (x, y, z) = (s[0], s[1], s[2]);
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    vec![sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
}

pub(crate) fn harmonic_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    vec![s[1], -s[0]]
}

pub(crate) fn lotka_volterra_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (x, y) = (s[0], s[1]);
    vec![x * (1.0 - y), y * (x - 1.0)]
}

pub(crate) fn kepler_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (x, y, vx, vy) = (s[0], s[1], s[2], s[3]);
    let r2 = x * x + y * y;
    let r3 = r2 * r2.sqrt();
    if r3 < 1e-15 {
        return vec![vx, vy, 0.0, 0.0];
    }
    vec![vx, vy, -x / r3, -y / r3]
}

pub(crate) fn double_pendulum_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (t1, t2, w1, w2) = (s[0], s[1], s[2], s[3]);
    let g = 9.81;
    let delta = t1 - t2;
    let (sd, cd) = (delta.sin(), delta.cos());
    let det = 2.0 - cd * cd;
    if det.abs() < 1e-15 {
        return vec![w1, w2, 0.0, 0.0];
    }
    let f1 = -2.0 * g * t1.sin() - w2 * w2 * sd - w1 * w1 * sd * cd;
    let f2 = -g * t2.sin() + w1 * w1 * sd + w2 * w2 * sd * cd;
    let a1 = (f1 - cd * f2) / det;
    let a2 = (2.0 * f2 - cd * f1) / det;
    vec![w1, w2, a1, a2]
}

pub(crate) fn double_pendulum_energy(s: &[f64]) -> f64 {
    let (t1, t2, w1, w2) = (s[0], s[1], s[2], s[3]);
    let g = 9.81;
    let kinetic = 0.5 * (2.0 * w1 * w1 + w2 * w2 + 2.0 * w1 * w2 * (t1 - t2).cos());
    let potential = -g * (2.0 * t1.cos() + t2.cos());
    kinetic + potential
}

pub(crate) fn henon_heiles_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (x, y, px, py) = (s[0], s[1], s[2], s[3]);
    vec![px, py, -x - 2.0 * x * y, -y - x * x + y * y]
}

pub(crate) fn henon_heiles_energy(s: &[f64]) -> f64 {
    let (x, y, px, py) = (s[0], s[1], s[2], s[3]);
    0.5 * (px * px + py * py) + 0.5 * (x * x + y * y) + x * x * y - y * y * y / 3.0
}

pub(crate) fn schwarzschild_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (r, _phi, pr, l) = (s[0], s[1], s[2], s[3]);
    if r < 2.5 {
        return vec![0.0; 4];
    }
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r3 * r;
    vec![pr, l / r2, -1.0 / r2 + l * l / r3 - 3.0 * l * l / r4, 0.0]
}

pub(crate) fn newtonian_orbit_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (r, _phi, pr, l) = (s[0], s[1], s[2], s[3]);
    if r < 0.1 {
        return vec![0.0; 4];
    }
    let r2 = r * r;
    let r3 = r2 * r;
    vec![pr, l / r2, -1.0 / r2 + l * l / r3, 0.0]
}

pub(crate) fn schwarzschild_v_eff(r: f64, l: f64) -> f64 {
    -1.0 / r + l * l / (2.0 * r * r) - l * l / (r * r * r)
}

pub(crate) fn newtonian_v_eff(r: f64, l: f64) -> f64 {
    -1.0 / r + l * l / (2.0 * r * r)
}

pub fn observe_gr_correction(l: f64, r_min: f64, r_max: f64, n_points: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (0..n_points)
        .map(|i| {
            let r = r_min + (r_max - r_min) * i as f64 / (n_points - 1) as f64;
            let diff = schwarzschild_v_eff(r, l) - newtonian_v_eff(r, l);
            (r, diff)
        })
        .filter(|(_, v)| v.is_finite())
        .collect();
    ObservedSequence::new("V_GR-V_Newton(r)", MathDomain::Physics, data)
}

pub fn compute_rolling_average(
    states: &[Vec<f64>],
    eval_fn: &dyn Fn(&[f64]) -> f64,
    window_size: usize,
) -> Vec<f64> {
    if states.len() < window_size {
        return Vec::new();
    }
    let values: Vec<f64> = states.iter().map(|s| eval_fn(s)).collect();
    values
        .windows(window_size)
        .map(|w| w.iter().sum::<f64>() / w.len() as f64)
        .collect()
}

pub fn check_virial_theorem(
    states: &[Vec<f64>],
    kinetic_fn: &dyn Fn(&[f64]) -> f64,
    potential_fn: &dyn Fn(&[f64]) -> f64,
    window_size: usize,
) -> (f64, f64) {
    let t_avg = compute_rolling_average(states, kinetic_fn, window_size);
    let v_avg = compute_rolling_average(states, potential_fn, window_size);

    if t_avg.is_empty() || v_avg.is_empty() {
        return (f64::NAN, f64::MAX);
    }

    let n = t_avg.len().min(v_avg.len());
    let ratios: Vec<f64> = (0..n)
        .filter_map(|i| {
            if v_avg[i].abs() > 1e-10 {
                Some(2.0 * t_avg[i] / v_avg[i])
            } else {
                None
            }
        })
        .collect();

    if ratios.is_empty() {
        return (f64::NAN, f64::MAX);
    }
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let var = ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ratios.len() as f64;
    (mean, var)
}


#[cfg(test)]
mod bounded_rk4_trajectory_tests {
    use super::{rk4_trajectory, rk4_trajectory_with_work};

    fn constant_rhs(state: &[f64], _time: f64) -> Vec<f64> {
        vec![1.0; state.len()]
    }

    #[test]
    fn indexed_rk4_reaches_exact_non_binary_endpoints() {
        for (t_end, dt) in [(4.0, 0.01), (1.0, 0.1)] {
            let (times, states) = rk4_trajectory(constant_rhs, &[0.0], t_end, dt);
            assert!(!times.is_empty());
            assert_eq!(times.last().copied(), Some(t_end));
            assert_eq!(states.len(), times.len());
            assert!((states.last().unwrap()[0] - t_end).abs() < 1e-10);
        }
    }

    #[test]
    fn invalid_timing_or_rhs_fails_closed_without_partial_trajectory() {
        let (times, states) = rk4_trajectory(constant_rhs, &[0.0], 1.0, 0.0);
        assert!(times.is_empty());
        assert!(states.is_empty());

        let (times, states) = rk4_trajectory(|_state, _time| vec![f64::NAN], &[0.0], 1.0, 0.1);
        assert!(times.is_empty());
        assert!(states.is_empty());

        let (times, states) = rk4_trajectory(|_state, _time| vec![], &[0.0], 1.0, 0.1);
        assert!(times.is_empty());
        assert!(states.is_empty());
    }

    #[test]
    fn successful_work_counters_match_completed_rk4_steps() {
        let (times, states, work) = rk4_trajectory_with_work(constant_rhs, &[0.0, 1.0], 1.0, 0.1);
        assert!(work.completed);
        assert_eq!(work.attempted_steps, work.completed_steps);
        assert_eq!(work.completed_steps as usize, states.len() - 1);
        assert_eq!(times.len(), states.len());
        assert_eq!(work.rhs_evaluations, work.completed_steps * 4);
        assert_eq!(work.state_updates, work.completed_steps * 2);
    }

    #[test]
    fn failed_work_counters_preserve_rhs_calls_without_partial_trajectory() {
        let (times, states, work) = rk4_trajectory_with_work(
            |_state, time| if time == 0.0 { vec![1.0] } else { vec![f64::NAN] },
            &[0.0],
            1.0,
            0.1,
        );
        assert!(times.is_empty());
        assert!(states.is_empty());
        assert!(!work.completed);
        assert_eq!(work.attempted_steps, 1);
        assert_eq!(work.completed_steps, 0);
        assert_eq!(work.rhs_evaluations, 2);
        assert_eq!(work.state_updates, 0);
    }

    #[test]
    fn overflowed_state_update_work_is_counted_before_rejection() {
        let (_times, states, work) = rk4_trajectory_with_work(
            |state, _time| vec![f64::MAX; state.len()],
            &[0.0],
            1.0,
            0.1,
        );
        assert!(states.is_empty());
        assert!(!work.completed);
        assert_eq!(work.attempted_steps, 1);
        assert_eq!(work.completed_steps, 0);
        assert_eq!(work.rhs_evaluations, 4);
        assert_eq!(work.state_updates, 1);
    }

    #[test]
    fn excessive_step_budget_is_rejected_before_allocation() {
        let (times, states) = rk4_trajectory(constant_rhs, &[0.0], 1.0e9, 1.0e-6);
        assert!(times.is_empty());
        assert!(states.is_empty());
    }
}
