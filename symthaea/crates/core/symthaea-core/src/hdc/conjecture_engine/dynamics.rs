use super::*;

pub(crate) fn rk45_trajectory(
    f: impl Fn(&[f64], f64) -> Vec<f64>,
    y0: &[f64],
    t_end: f64,
    dt: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut t = 0.0;
    let mut y = y0.to_vec();
    let mut times = vec![t];
    let mut states = vec![y.clone()];
    let dim = y0.len();

    while t < t_end {
        let h = dt.min(t_end - t);
        let k1 = f(&y, t);
        let y2: Vec<f64> = (0..dim).map(|i| y[i] + h * 0.5 * k1[i]).collect();
        let k2 = f(&y2, t + 0.5 * h);
        let y3: Vec<f64> = (0..dim).map(|i| y[i] + h * 0.5 * k2[i]).collect();
        let k3 = f(&y3, t + 0.5 * h);
        let y4: Vec<f64> = (0..dim).map(|i| y[i] + h * k3[i]).collect();
        let k4 = f(&y4, t + h);
        for i in 0..dim {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
        times.push(t);
        states.push(y.clone());
    }
    (times, states)
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
