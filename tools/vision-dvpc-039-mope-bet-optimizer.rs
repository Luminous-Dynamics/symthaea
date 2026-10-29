//! VIS-DVPC-039: standalone MOPE quadratic constrained-bet optimizer kernel.
//!
//! Implements the two-dimensional quadratic subproblem from the "Avoiding Grid Search"
//! section of Karampatziakis, Mineiro & Ramdas (ICML 2021):
//!   maximize psi * lambda^T A lambda + b^T lambda
//!   subject to lambda_2 >= 0,
//!              lambda_1 + lambda_2 <= 1/2,
//!              (1-W) * lambda_1 + lambda_2 <= 1/2,
//! where psi = 2 - 4 ln(2) < 0 and A is symmetric positive semidefinite.
//!
//! The optimizer evaluates the feasible unconstrained stationary point when it exists,
//! and the exact one-dimensional optimum on each polygon edge. This handles singular
//! matrices without relying on unstable inverses on a face. W=1 is a valid degeneracy
//! for unit importance ratios; lambda_1 then has no effect and is set to zero.
//!
//! This is only the numeric optimizer kernel, not a full MOPE confidence sequence.
#![deny(unsafe_code)]

use std::env;
use std::fmt::Write as _;

const PSI: f64 = 2.0 - 4.0 * std::f64::consts::LN_2;
const EPS: f64 = 1e-10;
const PSD_TOL: f64 = 1e-10;
const FEAS_TOL: f64 = 1e-9;
const GRID_N: usize = 160;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Vec2 {
    x: f64,
    y: f64,
}

impl Vec2 {
    fn new(x: f64, y: f64) -> Self { Self { x, y } }
    fn dot(self, other: Self) -> f64 { self.x * other.x + self.y * other.y }
    fn add(self, other: Self) -> Self { Self::new(self.x + other.x, self.y + other.y) }
    fn sub(self, other: Self) -> Self { Self::new(self.x - other.x, self.y - other.y) }
    fn scale(self, scalar: f64) -> Self { Self::new(self.x * scalar, self.y * scalar) }
    fn finite(self) -> bool { self.x.is_finite() && self.y.is_finite() }
}

#[derive(Clone, Copy, Debug)]
struct Sym2 {
    a00: f64,
    a01: f64,
    a11: f64,
}

impl Sym2 {
    fn quad(self, value: Vec2) -> f64 {
        self.a00 * value.x * value.x
            + 2.0 * self.a01 * value.x * value.y
            + self.a11 * value.y * value.y
    }

    fn bilinear(self, left: Vec2, right: Vec2) -> f64 {
        self.a00 * left.x * right.x
            + self.a01 * (left.x * right.y + left.y * right.x)
            + self.a11 * left.y * right.y
    }

    fn finite(self) -> bool {
        self.a00.is_finite() && self.a01.is_finite() && self.a11.is_finite()
    }

    fn determinant(self) -> f64 { self.a00 * self.a11 - self.a01 * self.a01 }

    fn is_psd(self) -> bool {
        if !self.finite() || self.a00 < -PSD_TOL || self.a11 < -PSD_TOL {
            return false;
        }
        let scale = 1.0_f64.max((self.a00 * self.a11).abs()).max((self.a01 * self.a01).abs());
        self.determinant() >= -PSD_TOL * scale
    }
}

#[derive(Clone, Debug)]
struct Optimum {
    lambda: Vec2,
    objective: f64,
    active_constraints: Vec<&'static str>,
    slack_lambda2: f64,
    slack_sum: f64,
    slack_weighted: f64,
    stationary_point_considered: bool,
    edge_candidates_considered: usize,
}

fn objective(a: Sym2, b: Vec2, lambda: Vec2) -> f64 {
    PSI * a.quad(lambda) + b.dot(lambda)
}

fn slacks(lambda: Vec2, max_weight: f64) -> (f64, f64, f64) {
    (
        lambda.y,
        0.5 - lambda.x - lambda.y,
        0.5 - (1.0 - max_weight) * lambda.x - lambda.y,
    )
}

fn feasible(lambda: Vec2, max_weight: f64, tolerance: f64) -> bool {
    if !lambda.finite() || !max_weight.is_finite() {
        return false;
    }
    let (s0, s1, s2) = slacks(lambda, max_weight);
    s0 >= -tolerance && s1 >= -tolerance && s2 >= -tolerance
}

fn validate(a: Sym2, b: Vec2, max_weight: f64) -> Result<(), String> {
    if !a.finite() || !b.finite() || !max_weight.is_finite() {
        return Err("A, b and W must be finite".into());
    }
    if max_weight < 1.0 - EPS {
        return Err("W must be at least 1 for importance ratios with E[w]=1".into());
    }
    if !a.is_psd() {
        return Err("A must be symmetric positive semidefinite".into());
    }
    Ok(())
}

fn edge_maximum(a: Sym2, b: Vec2, start: Vec2, end: Vec2) -> Vec2 {
    let direction = end.sub(start);
    // f(start + t*d) = constant + linear*t + quadratic*t^2.
    let quadratic = PSI * a.quad(direction);
    let linear = 2.0 * PSI * a.bilinear(start, direction) + b.dot(direction);
    let t = if quadratic < -EPS {
        (-linear / (2.0 * quadratic)).clamp(0.0, 1.0)
    } else if linear > 0.0 {
        // The edge objective is linear (or numerically flat); select its increasing endpoint.
        1.0
    } else {
        0.0
    };
    start.add(direction.scale(t))
}

fn optimize(a: Sym2, b: Vec2, max_weight: f64) -> Result<Optimum, String> {
    validate(a, b, max_weight)?;

    let mut candidates: Vec<Vec2> = Vec::with_capacity(10);
    let mut stationary_point_considered = false;
    let mut edge_candidates_considered = 0usize;

    if (max_weight - 1.0).abs() <= EPS {
        // For valid importance-ratio samples with W=1, every w_i=1: A_00=A_01=b_0=0.
        // The lambda_1 direction has no effect; require that input invariant before reducing.
        if a.a00.abs() > PSD_TOL || a.a01.abs() > PSD_TOL || b.x.abs() > PSD_TOL {
            return Err("W=1 degeneracy requires A00=A01=b0=0 (unit importance ratios)".into());
        }
        let lambda_y = if a.a11 > PSD_TOL {
            (-b.y / (2.0 * PSI * a.a11)).clamp(0.0, 0.5)
        } else if b.y > 0.0 {
            0.5
        } else {
            0.0
        };
        candidates.push(Vec2::new(0.0, lambda_y));
    } else {
        // The feasible region is a triangle for W>1.
        let v0 = Vec2::new(1.0 / (2.0 * (1.0 - max_weight)), 0.0);
        let v1 = Vec2::new(0.0, 0.5);
        let v2 = Vec2::new(0.5, 0.0);
        candidates.extend([v0, v1, v2]);

        // Unconstrained stationary point lambda = -(2*psi*A)^(-1) b, when A is invertible.
        let det_a = a.determinant();
        let determinant_scale = 1.0_f64.max(a.a00.abs() * a.a11.abs()).max(a.a01 * a.a01);
        if det_a > PSD_TOL * determinant_scale {
            let factor = -1.0 / (2.0 * PSI * det_a);
            let stationary = Vec2::new(
                factor * (a.a11 * b.x - a.a01 * b.y),
                factor * (-a.a01 * b.x + a.a00 * b.y),
            );
            if stationary.finite() && feasible(stationary, max_weight, FEAS_TOL) {
                candidates.push(stationary);
                stationary_point_considered = true;
            }
        }

        // The objective is concave; its maximum on every compact edge is its stationary
        // point on that segment or an endpoint. Explicit segment optimization is robust
        // to singular matrices and parallel/flat edge faces.
        for (start, end) in [(v0, v1), (v1, v2), (v2, v0)] {
            let candidate = edge_maximum(a, b, start, end);
            if !candidate.finite() || !feasible(candidate, max_weight, FEAS_TOL) {
                return Err("edge optimizer produced an infeasible candidate".into());
            }
            candidates.push(candidate);
            edge_candidates_considered += 1;
        }
    }

    let mut best: Option<(Vec2, f64)> = None;
    for candidate in candidates {
        if !feasible(candidate, max_weight, FEAS_TOL) {
            continue;
        }
        let value = objective(a, b, candidate);
        if !value.is_finite() {
            return Err("objective evaluated to a non-finite value".into());
        }
        if best.map(|(_, current)| value > current).unwrap_or(true) {
            best = Some((candidate, value));
        }
    }
    let (lambda, value) = best.ok_or_else(|| "no feasible optimizer candidate".to_owned())?;
    let (slack_lambda2, slack_sum, slack_weighted) = slacks(lambda, max_weight);
    let mut active_constraints = Vec::new();
    if slack_lambda2.abs() <= FEAS_TOL { active_constraints.push("lambda2>=0"); }
    if slack_sum.abs() <= FEAS_TOL { active_constraints.push("lambda1+lambda2<=0.5"); }
    if slack_weighted.abs() <= FEAS_TOL { active_constraints.push("(1-W)lambda1+lambda2<=0.5"); }

    Ok(Optimum {
        lambda,
        objective: value,
        active_constraints,
        slack_lambda2,
        slack_sum,
        slack_weighted,
        stationary_point_considered,
        edge_candidates_considered,
    })
}

fn dense_grid_best(a: Sym2, b: Vec2, max_weight: f64) -> f64 {
    if (max_weight - 1.0).abs() <= EPS {
        let mut best = f64::NEG_INFINITY;
        for i in 0..=GRID_N {
            let y = 0.5 * i as f64 / GRID_N as f64;
            best = best.max(objective(a, b, Vec2::new(0.0, y)));
        }
        return best;
    }
    let v0 = Vec2::new(1.0 / (2.0 * (1.0 - max_weight)), 0.0);
    let v1 = Vec2::new(0.0, 0.5);
    let v2 = Vec2::new(0.5, 0.0);
    let mut best = f64::NEG_INFINITY;
    for i in 0..=GRID_N {
        for j in 0..=(GRID_N - i) {
            let alpha = i as f64 / GRID_N as f64;
            let beta = j as f64 / GRID_N as f64;
            let gamma = 1.0 - alpha - beta;
            let lambda = v0.scale(gamma).add(v1.scale(alpha)).add(v2.scale(beta));
            best = best.max(objective(a, b, lambda));
        }
    }
    best
}

#[derive(Clone, Copy)]
struct Fixture {
    id: &'static str,
    a: Sym2,
    b: Vec2,
    w: f64,
}

const FIXTURES: [Fixture; 6] = [
    Fixture {
        id: "interior_optimum",
        a: Sym2 { a00: 1.0, a01: 0.0, a11: 1.0 },
        b: Vec2 { x: -2.0 * PSI * 0.1, y: -2.0 * PSI * 0.1 },
        w: 50.0,
    },
    Fixture {
        id: "mixed_psd",
        a: Sym2 { a00: 4.0, a01: 1.0, a11: 2.0 },
        b: Vec2 { x: 2.0, y: -0.5 },
        w: 50.0,
    },
    Fixture {
        id: "singular_psd",
        a: Sym2 { a00: 1.0, a01: 1.0, a11: 1.0 },
        b: Vec2 { x: 0.7, y: 0.3 },
        w: 50.0,
    },
    Fixture {
        id: "lambda2_zero_edge",
        a: Sym2 { a00: 1.0, a01: 0.0, a11: 1.0 },
        b: Vec2 { x: 5.0, y: -5.0 },
        w: 50.0,
    },
    Fixture {
        id: "weighted_edge",
        a: Sym2 { a00: 2.0, a01: 0.3, a11: 1.0 },
        b: Vec2 { x: -1.0, y: 5.0 },
        w: 50.0,
    },
    Fixture {
        id: "unit_weight_degeneracy",
        a: Sym2 { a00: 0.0, a01: 0.0, a11: 3.0 },
        b: Vec2 { x: 0.0, y: 0.25 },
        w: 1.0,
    },
];

fn sweep() -> Result<String, String> {
    let mut out = String::from("record_type\tcase_id\tmetric\tvalue\n");
    for fixture in FIXTURES {
        let optimum = optimize(fixture.a, fixture.b, fixture.w)?;
        let grid_best = dense_grid_best(fixture.a, fixture.b, fixture.w);
        if optimum.objective + 1e-8 < grid_best {
            return Err(format!("{}: exact optimizer is worse than feasible dense-grid best", fixture.id));
        }
        if !feasible(optimum.lambda, fixture.w, FEAS_TOL) {
            return Err(format!("{}: result fails feasible-set check", fixture.id));
        }
        for (name, value) in [
            ("W", fixture.w),
            ("A00", fixture.a.a00),
            ("A01", fixture.a.a01),
            ("A11", fixture.a.a11),
            ("b0", fixture.b.x),
            ("b1", fixture.b.y),
            ("lambda1", optimum.lambda.x),
            ("lambda2", optimum.lambda.y),
            ("objective", optimum.objective),
            ("dense_grid_best_diagnostic", grid_best),
            ("slack_lambda2", optimum.slack_lambda2),
            ("slack_sum", optimum.slack_sum),
            ("slack_weighted", optimum.slack_weighted),
        ] {
            let _ = writeln!(out, "METRIC\t{}\t{}\t{value:.12}", fixture.id, name);
        }
        let _ = writeln!(out, "METRIC\t{}\tstationary_point_considered\t{}", fixture.id, optimum.stationary_point_considered);
        let _ = writeln!(out, "METRIC\t{}\tedge_candidates_considered\t{}", fixture.id, optimum.edge_candidates_considered);
        let _ = writeln!(out, "METRIC\t{}\tactive_constraints\t{}", fixture.id, optimum.active_constraints.join(","));
        let _ = writeln!(out, "CASE\t{}\tstatus\tOK", fixture.id);
    }
    Ok(out)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("--sweep") => sweep(),
        _ => Err("usage: vision-dvpc-039-mope-bet-optimizer --sweep".to_owned()),
    };
    match result {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_039_ERROR\t{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!((actual - expected).abs() <= tolerance, "actual={actual:.15}, expected={expected:.15}");
    }

    #[test]
    fn published_constant_and_feasible_vertices_are_exact() {
        close(PSI, 2.0 - 4.0 * std::f64::consts::LN_2, EPS);
        let w = 50.0;
        let v0 = Vec2::new(1.0 / (2.0 * (1.0 - w)), 0.0);
        let v1 = Vec2::new(0.0, 0.5);
        let v2 = Vec2::new(0.5, 0.0);
        assert!(feasible(v0, w, FEAS_TOL));
        assert!(feasible(v1, w, FEAS_TOL));
        assert!(feasible(v2, w, FEAS_TOL));
    }

    #[test]
    fn known_interior_stationary_point_is_recovered() {
        let fixture = FIXTURES[0];
        let result = optimize(fixture.a, fixture.b, fixture.w).unwrap();
        close(result.lambda.x, 0.1, 1e-9);
        close(result.lambda.y, 0.1, 1e-9);
        assert!(result.stationary_point_considered);
        assert!(result.active_constraints.is_empty());
    }

    #[test]
    fn constrained_optimum_dominates_dense_feasible_grid() {
        for fixture in FIXTURES {
            let result = optimize(fixture.a, fixture.b, fixture.w).unwrap();
            let grid_best = dense_grid_best(fixture.a, fixture.b, fixture.w);
            assert!(result.objective + 1e-8 >= grid_best,
                "{}: optimizer={} grid_best={}", fixture.id, result.objective, grid_best);
            assert!(feasible(result.lambda, fixture.w, FEAS_TOL));
        }
    }

    #[test]
    fn singular_positive_semidefinite_matrix_is_supported() {
        let fixture = FIXTURES[2];
        assert!(fixture.a.determinant().abs() <= EPS);
        let result = optimize(fixture.a, fixture.b, fixture.w).unwrap();
        assert!(result.objective.is_finite());
    }

    #[test]
    fn unit_importance_weight_degeneracy_is_reduced_safely() {
        let fixture = FIXTURES[5];
        let result = optimize(fixture.a, fixture.b, fixture.w).unwrap();
        close(result.lambda.x, 0.0, EPS);
        close(result.lambda.y, (-fixture.b.y / (2.0 * PSI * fixture.a.a11)).clamp(0.0, 0.5), EPS);
        assert!(result.active_constraints.is_empty() || result.active_constraints.len() == 1);
        let inconsistent = Sym2 { a00: 1.0, a01: 0.0, a11: 3.0 };
        assert!(optimize(inconsistent, Vec2::new(0.0, 0.25), 1.0).is_err());
    }

    #[test]
    fn invalid_weights_and_non_psd_matrix_fail_closed() {
        assert!(optimize(FIXTURES[0].a, FIXTURES[0].b, 0.8).is_err());
        let invalid = Sym2 { a00: 1.0, a01: 2.0, a11: 1.0 };
        assert!(optimize(invalid, Vec2::new(0.1, 0.2), 50.0).is_err());
        assert!(optimize(FIXTURES[0].a, Vec2::new(f64::NAN, 0.0), 50.0).is_err());
    }

    #[test]
    fn all_fixture_results_have_nonnegative_slacks() {
        for fixture in FIXTURES {
            let result = optimize(fixture.a, fixture.b, fixture.w).unwrap();
            assert!(result.slack_lambda2 >= -FEAS_TOL);
            assert!(result.slack_sum >= -FEAS_TOL);
            assert!(result.slack_weighted >= -FEAS_TOL);
        }
    }
}
