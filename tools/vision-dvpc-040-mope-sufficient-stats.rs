//! VIS-DVPC-040: streaming MOPE sufficient-statistic and polynomial identity audit.
//!
//! This module reuses the optimizer kernel from the parent optimizer PR via a path
//! module. It does NOT produce an end-to-end confidence sequence. It verifies the
//! sufficient-statistic identities used by MOPE's quadratic log-wealth lower bound.
//!
//! For z=w-1, d=w*r, b(v)=[z,d-v], and A(v)=b(v)b(v)^T:
//!   sum A(v) = A0 + v*A1 + v^2*A2
//!   sum b(v) = b0 + v*b1
//! The bet-dependent lower bound is:
//!   P(v)=C-S*v+Q+T*v+U*v^2
//! and must equal sum_i [lambda_i^T b_i(v) + psi lambda_i^T A_i(v) lambda_i].
//!
//! Source equations: Karampatziakis, Mineiro & Ramdas (ICML 2021), eqs. (7)-(14),
//! (18)-(22), and the LCS pseudocode.
#![deny(unsafe_code)]

#[path = "vision-dvpc-039-mope-bet-optimizer.rs"]
mod optimizer;

use std::env;
use std::fmt::Write as _;

const PSI: f64 = 2.0 - 4.0 * std::f64::consts::LN_2;
const EPS: f64 = 1e-10;
const FEAS_TOL: f64 = 1e-9;
const IDENTITY_TOL: f64 = 5e-10;
const LEVEL_ALPHA: f64 = 0.05;
const TEST_VALUES: [f64; 21] = [
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
];

#[derive(Clone, Copy, Debug)]
struct ObservedStats {
    a0: optimizer::Sym2,
    a1: optimizer::Sym2,
    a2: optimizer::Sym2,
    b0: optimizer::Vec2,
    b1: optimizer::Vec2,
    count: u64,
}

impl Default for ObservedStats {
    fn default() -> Self {
        Self {
            a0: optimizer::Sym2 { a00: 0.0, a01: 0.0, a11: 0.0 },
            a1: optimizer::Sym2 { a00: 0.0, a01: 0.0, a11: 0.0 },
            a2: optimizer::Sym2 { a00: 0.0, a01: 0.0, a11: 0.0 },
            b0: optimizer::Vec2::new(0.0, 0.0),
            b1: optimizer::Vec2::new(0.0, 0.0),
            count: 0,
        }
    }
}

impl ObservedStats {
    fn update(&mut self, w: f64, r: f64) {
        let z = w - 1.0;
        let d = w * r;
        self.a0.a00 += z * z;
        self.a0.a01 += z * d;
        self.a0.a11 += d * d;

        self.a1.a00 += 0.0;
        self.a1.a01 += -z;
        self.a1.a11 += -2.0 * d;

        self.a2.a00 += 0.0;
        self.a2.a01 += 0.0;
        self.a2.a11 += 1.0;

        self.b0.x += z;
        self.b0.y += d;
        self.b1.x += 0.0;
        self.b1.y += -1.0;
        self.count += 1;
    }

    fn matrix_at(self, value: f64) -> optimizer::Sym2 {
        optimizer::Sym2 {
            a00: self.a0.a00 + value * self.a1.a00 + value * value * self.a2.a00,
            a01: self.a0.a01 + value * self.a1.a01 + value * value * self.a2.a01,
            a11: self.a0.a11 + value * self.a1.a11 + value * value * self.a2.a11,
        }
    }

    fn vector_at(self, value: f64) -> optimizer::Vec2 {
        optimizer::Vec2::new(
            self.b0.x + value * self.b1.x,
            self.b0.y + value * self.b1.y,
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PolynomialStats {
    c: f64,
    s: f64,
    q: f64,
    t: f64,
    u: f64,
    count: u64,
}

impl PolynomialStats {
    fn update(&mut self, w: f64, r: f64, lambda: optimizer::Vec2) {
        let z = w - 1.0;
        let d = w * r;
        let a0 = optimizer::Sym2 {
            a00: z * z,
            a01: z * d,
            a11: d * d,
        };
        let a1 = optimizer::Sym2 {
            a00: 0.0,
            a01: -z,
            a11: -2.0 * d,
        };
        self.c += lambda.x * z + lambda.y * d;
        self.s += lambda.y;
        self.q += PSI * a0.quad(lambda);
        self.t += PSI * a1.quad(lambda);
        self.u += PSI * lambda.y * lambda.y;
        self.count += 1;
    }

    fn value(self, candidate: f64) -> f64 {
        self.c - self.s * candidate + self.q + self.t * candidate + self.u * candidate * candidate
    }

    fn coefficients_at_level(self, level: f64) -> [f64; 3] {
        [self.u, self.t - self.s, self.c + self.q - level]
    }

    fn roots_at_level(self, level: f64) -> Vec<f64> {
        real_roots(self.coefficients_at_level(level))
    }
}

fn real_roots([a, b, c]: [f64; 3]) -> Vec<f64> {
    if [a, b, c].iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    let scale = a.abs().max(b.abs()).max(c.abs());
    if scale == 0.0 {
        return Vec::new();
    }
    let (a, b, c) = (a / scale, b / scale, c / scale);
    if a.abs() <= EPS {
        if b.abs() <= EPS {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let mut discriminant = b * b - 4.0 * a * c;
    if discriminant < -EPS {
        return Vec::new();
    }
    discriminant = discriminant.max(0.0);
    let root_disc = discriminant.sqrt();
    // Stable quadratic formula; the other root follows from the product c/a.
    let q = -0.5 * (b + if b >= 0.0 { root_disc } else { -root_disc });
    if q.abs() <= EPS {
        return vec![-b / (2.0 * a)];
    }
    let r1 = q / a;
    let r2 = c / q;
    if (r1 - r2).abs() <= EPS {
        vec![r1]
    } else if r1 < r2 {
        vec![r1, r2]
    } else {
        vec![r2, r1]
    }
}

#[derive(Clone, Copy, Debug)]
struct StreamState {
    max_weight: f64,
    alpha: f64,
    observed: ObservedStats,
    polynomial: PolynomialStats,
    current_lambda: optimizer::Vec2,
}

#[derive(Clone, Debug)]
struct StepReport {
    count: u64,
    lambda_used: optimizer::Vec2,
    polynomial: PolynomialStats,
    level_roots: Vec<f64>,
    selected_value_for_next_bet: Option<f64>,
    next_lambda: Option<optimizer::Vec2>,
}

impl StreamState {
    fn new(max_weight: f64, alpha: f64) -> Result<Self, String> {
        if !max_weight.is_finite() || max_weight < 1.0 - EPS {
            return Err("max_weight must be finite and at least 1".to_owned());
        }
        if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
            return Err("alpha must be finite and in (0,1)".to_owned());
        }
        Ok(Self {
            max_weight,
            alpha,
            observed: ObservedStats::default(),
            polynomial: PolynomialStats::default(),
            current_lambda: optimizer::Vec2::new(0.0, 0.0),
        })
    }

    fn observe(&mut self, w: f64, r: f64) -> Result<StepReport, String> {
        if !w.is_finite() || !r.is_finite() || w < 0.0 || w > self.max_weight + FEAS_TOL
            || !(0.0..=1.0).contains(&r)
        {
            return Err("sample requires finite w in [0,W] and reward r in [0,1]".to_owned());
        }
        if (self.max_weight - 1.0).abs() <= EPS && (w - 1.0).abs() > FEAS_TOL {
            return Err("W=1 requires unit importance weights w=1".to_owned());
        }
        if !lambda_feasible(self.current_lambda, self.max_weight) {
            return Err("predictable bet is outside the paper's feasible region".to_owned());
        }

        let lambda_used = self.current_lambda;
        self.polynomial.update(w, r, lambda_used);
        self.observed.update(w, r);

        let level = (2.0 / self.alpha).ln();
        let roots = self.polynomial.roots_at_level(level);
        Ok(StepReport {
            count: self.observed.count,
            lambda_used,
            polynomial: self.polynomial,
            level_roots: roots,
            selected_value_for_next_bet: None,
            next_lambda: None,
        })
    }

    fn choose_next_bet(&mut self, candidate_value: f64) -> Result<optimizer::Vec2, String> {
        if !candidate_value.is_finite() || !(0.0..=1.0).contains(&candidate_value) {
            return Err("candidate_value must lie in [0,1]".to_owned());
        }
        let a = self.observed.matrix_at(candidate_value);
        let b = self.observed.vector_at(candidate_value);
        if !a.is_psd() {
            return Err("assembled sufficient-statistic matrix is not PSD".to_owned());
        }
        let optimum = optimizer::optimize(a, b, self.max_weight)?;
        if !lambda_feasible(optimum.lambda, self.max_weight) {
            return Err("optimizer returned an infeasible bet".to_owned());
        }
        self.current_lambda = optimum.lambda;
        Ok(optimum.lambda)
    }

    fn current_lambda(self) -> optimizer::Vec2 { self.current_lambda }
    fn observed(self) -> ObservedStats { self.observed }
    fn polynomial(self) -> PolynomialStats { self.polynomial }
}

fn lambda_feasible(lambda: optimizer::Vec2, max_weight: f64) -> bool {
    lambda.finite()
        && lambda.y >= -FEAS_TOL
        && lambda.x + lambda.y <= 0.5 + FEAS_TOL
        && (1.0 - max_weight) * lambda.x + lambda.y <= 0.5 + FEAS_TOL
}

fn direct_observed(samples: &[(f64, f64)], value: f64) -> (optimizer::Sym2, optimizer::Vec2) {
    let mut a = optimizer::Sym2 { a00: 0.0, a01: 0.0, a11: 0.0 };
    let mut b = optimizer::Vec2::new(0.0, 0.0);
    for &(w, r) in samples {
        let z = w - 1.0;
        let d = w * r - value;
        a.a00 += z * z;
        a.a01 += z * d;
        a.a11 += d * d;
        b.x += z;
        b.y += d;
    }
    (a, b)
}

fn direct_polynomial(samples: &[((f64, f64), optimizer::Vec2)], value: f64) -> f64 {
    samples.iter().map(|&((w, r), lambda)| {
        let z = w - 1.0;
        let d = w * r - value;
        let b = optimizer::Vec2::new(z, d);
        let a = optimizer::Sym2 { a00: z*z, a01: z*d, a11: d*d };
        (lambda.x * b.x + lambda.y * b.y) + PSI * a.quad(lambda)
    }).sum()
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs()))
}

fn sweep() -> Result<String, String> {
    let sequences: [(&str, f64, Vec<(f64, f64)>); 4] = [
        ("unit_weights", 1.0, vec![(1.0, 0.0), (1.0, 1.0), (1.0, 0.25), (1.0, 0.75)]),
        ("fractional_weights", 3.0, vec![(0.0, 0.5), (0.25, 1.0), (0.5, 0.0), (1.25, 0.8), (3.0, 0.6)]),
        ("high_weight", 100.0, vec![(1.0, 0.0), (100.0, 1.0), (0.0, 1.0), (50.0, 0.25), (2.0, 0.4)]),
        ("zero_reward", 4.0, vec![(0.5, 0.0), (4.0, 0.0), (1.0, 0.0), (2.0, 0.0)]),
    ];
    let mut output=String::from("record_type\tsequence_id\tcount\tvalue\tmetric\tobserved\tdirect\n");
    for (sequence_id, w_max, samples) in sequences {
        let mut stream = StreamState::new(w_max, LEVEL_ALPHA)?;
        let mut direct_samples: Vec<(f64,f64)> = Vec::new();
        let mut direct_bets: Vec<((f64,f64),optimizer::Vec2)> = Vec::new();
        for (index, &(w,r)) in samples.iter().enumerate() {
            let lambda=stream.current_lambda();
            let report=stream.observe(w,r)?;
            direct_samples.push((w,r));
            direct_bets.push(((w,r),lambda));
            for &value in &TEST_VALUES {
                let (a,b)=direct_observed(&direct_samples,value);
                let actual_a=report.count as f64;
                let _=actual_a;
                let stored_a=stream.observed().matrix_at(value);
                let stored_b=stream.observed().vector_at(value);
                if !close(stored_a.a00,a.a00,IDENTITY_TOL) || !close(stored_a.a01,a.a01,IDENTITY_TOL)
                    || !close(stored_a.a11,a.a11,IDENTITY_TOL) || !close(stored_b.x,b.x,IDENTITY_TOL)
                    || !close(stored_b.y,b.y,IDENTITY_TOL)
                {
                    return Err(format!("{sequence_id} prefix {}: sufficient-stat identity mismatch at value {value}",index+1));
                }
                let direct_bound=direct_polynomial(&direct_bets,value);
                let stored_bound=stream.polynomial().value(value);
                if !close(stored_bound,direct_bound,IDENTITY_TOL) {
                    return Err(format!("{sequence_id} prefix {}: polynomial identity mismatch at value {value}: {stored_bound} vs {direct_bound}",index+1));
                }
                let _=writeln!(output,"IDENTITY\t{sequence_id}\t{}\t{value:.2}\tpolynomial_lower_bound\t{stored_bound:.12}\t{direct_bound:.12}",index+1);
            }
            let candidate_value=if index % 3 == 0 {0.25} else if index % 3 == 1 {0.5} else {0.75};
            let next=stream.choose_next_bet(candidate_value)?;
            let _=writeln!(output,"BET\t{sequence_id}\t{}\t{candidate_value:.2}\tlambda1\t{:.12}\t-",index+1,next.x);
            let _=writeln!(output,"BET\t{sequence_id}\t{}\t{candidate_value:.2}\tlambda2\t{:.12}\t-",index+1,next.y);
            let report=StepReport { next_lambda:Some(next),selected_value_for_next_bet:Some(candidate_value),..report };
            if report.count != (index+1) as u64 { return Err("stream count drift".into()); }
        }
        let _=writeln!(output,"SEQUENCE\t{sequence_id}\t{}\t-\tprobability_or_coverage_claim\tNOT_EVALUATED\tNOT_EVALUATED",samples.len());
        let _=writeln!(output,"SEQUENCE\t{sequence_id}\t{}\t-\tstate_count\t{}\tO(1)",samples.len(),std::mem::size_of::<StreamState>());
    }
    Ok(output)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("--sweep") => sweep(),
        _ => Err("usage: vision-dvpc-040-mope-sufficient-stats --sweep".to_owned()),
    };
    match result {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_040_ERROR\t{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficient_expansion_matches_direct_sums_for_all_values_and_prefixes() {
        let sequences: [(f64, &[(f64,f64)]); 3] = [
            (1.0, &[(1.0,0.0),(1.0,1.0),(1.0,0.25)]),
            (4.0, &[(0.0,0.5),(0.25,1.0),(3.0,0.6),(1.0,0.0)]),
            (100.0, &[(1.0,0.0),(100.0,1.0),(50.0,0.25),(0.0,1.0)]),
        ];
        for (w_max,samples) in sequences {
            let mut stream=StreamState::new(w_max,LEVEL_ALPHA).unwrap();
            let mut direct:Vec<(f64,f64)>=Vec::new();
            let mut bet_records:Vec<((f64,f64),optimizer::Vec2)>=Vec::new();
            for &(w,r) in samples {
                let lambda=stream.current_lambda();
                stream.observe(w,r).unwrap();
                direct.push((w,r));
                bet_records.push(((w,r),lambda));
                for &v in &TEST_VALUES {
                    let (a,b)=direct_observed(&direct,v);
                    let got_a=stream.observed().matrix_at(v);
                    let got_b=stream.observed().vector_at(v);
                    assert!(close(got_a.a00,a.a00,IDENTITY_TOL));
                    assert!(close(got_a.a01,a.a01,IDENTITY_TOL));
                    assert!(close(got_a.a11,a.a11,IDENTITY_TOL));
                    assert!(close(got_b.x,b.x,IDENTITY_TOL));
                    assert!(close(got_b.y,b.y,IDENTITY_TOL));
                    assert!(close(stream.polynomial().value(v),direct_polynomial(&bet_records,v),IDENTITY_TOL));
                }
                stream.choose_next_bet(0.5).unwrap();
            }
        }
    }

    #[test]
    fn optimizer_wiring_returns_feasible_predictable_bets() {
        for w_max in [1.0, 3.0, 100.0] {
            let mut stream=StreamState::new(w_max,0.05).unwrap();
            let sample=if w_max==1.0 {(1.0,0.4)} else {(w_max,0.4)};
            stream.observe(sample.0,sample.1).unwrap();
            let bet=stream.choose_next_bet(0.5).unwrap();
            assert!(lambda_feasible(bet,w_max));
        }
    }

    #[test]
    fn invalid_stream_inputs_fail_closed() {
        assert!(StreamState::new(0.9,0.05).is_err());
        assert!(StreamState::new(2.0,1.0).is_err());
        let mut stream=StreamState::new(2.0,0.05).unwrap();
        assert!(stream.observe(f64::NAN,0.5).is_err());
        assert!(stream.observe(1.0,-0.1).is_err());
        assert!(stream.observe(3.0,0.5).is_err());
        let mut unit=StreamState::new(1.0,0.05).unwrap();
        assert!(unit.observe(0.5,0.5).is_err());
    }

    #[test]
    fn polynomial_root_solver_handles_linear_quadratic_and_no_root_cases() {
        let roots=real_roots([1.0,-3.0,2.0]);
        assert_eq!(roots.len(),2);
        assert!(close(roots[0],1.0,EPS));
        assert!(close(roots[1],2.0,EPS));
        let linear=real_roots([0.0,2.0,-1.0]);
        assert_eq!(linear.len(),1);
        assert!(close(linear[0],0.5,EPS));
        assert!(real_roots([1.0,0.0,1.0]).is_empty());
    }
}
