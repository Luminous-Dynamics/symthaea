#![forbid(unsafe_code)]

//! Executable Clarabel 0.11.1 SDP smoke tests for AIRROAD's PSD mapping.
//!
//! This project is a nested Cargo workspace. It compiles the exact checked-in
//! AIRROAD matrix source as a module without resolving the root workspace.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, PSDTriangleConeT, SolverStatus, ZeroConeT,
};
#[path = "../../../crates/domains/symthaea-airroad-iqc/src/lib.rs"]
mod symthaea_airroad_iqc;

use symthaea_airroad_iqc::{Complex64, HermitianMatrix, RealSymmetricMatrix};

fn settings() -> DefaultSettings<f64> {
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.max_iter = 200;
    settings.tol_gap_abs = 1.0e-9;
    settings.tol_gap_rel = 1.0e-9;
    settings.tol_feas = 1.0e-9;
    settings
}

fn diagonal_csc(order: usize, value: f64) -> CscMatrix<f64> {
    let indices: Vec<usize> = (0..order).collect();
    let values = vec![value; order];
    CscMatrix::new_from_triplets(
        order,
        order,
        indices.clone(),
        indices,
        values,
    )
}

fn max_abs_difference(lhs: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn clarabel_solves_analytic_realified_hermitian_psd_problem() {
    // H = [[2, 1+2i], [1-2i, 5]] is Hermitian positive definite.
    // The AIRROAD realification R(H) is 4x4, real symmetric, and positive
    // definite. Solve min trace(X) subject to X - R(H) >= 0.
    let h = HermitianMatrix::try_new(
        2,
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(1.0, -2.0),
            Complex64::new(5.0, 0.0),
        ],
    )
    .expect("Hermitian input");
    let realified = h.realify().expect("realification");
    let order = realified.order();
    assert_eq!(order, 4);

    let expected_x = realified
        .to_clarabel_psd_triangle()
        .expect("finite svec");
    let n = expected_x.len();
    assert_eq!(n, 10);

    // Variables are svec(X). The conic slack is
    // s = -svec(R(H)) - (-I)*svec(X) = svec(X - R(H)).
    let p = CscMatrix::zeros((n, n));
    let a = diagonal_csc(n, -1.0);
    let b: Vec<f64> = expected_x.iter().map(|value| -*value).collect();
    let c = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let cones = vec![PSDTriangleConeT(order)];

    let mut solver =
        DefaultSolver::new(&p, &c, &a, &b, &cones, settings()).expect("valid SDP data");
    solver.solve();

    let solution = &solver.solution;
    assert_eq!(solution.status, SolverStatus::Solved);
    assert_eq!(solution.x.len(), n);
    assert!(
        max_abs_difference(&solution.x, &expected_x) < 2.0e-6,
        "primal svec solution differs from the known optimum"
    );
    assert!((solution.obj_val - 14.0).abs() < 2.0e-6);
    assert!((solution.obj_val_dual - 14.0).abs() < 2.0e-6);
    assert!(solution.r_prim < 2.0e-7);
    assert!(solution.r_dual < 2.0e-7);

    // Decode the solver's primal variable through the inverse AIRROAD map.
    let decoded = RealSymmetricMatrix::from_clarabel_psd_triangle(order, &solution.x)
        .expect("decode solver primal");
    assert!(max_abs_difference(decoded.as_slice(), realified.as_slice()) < 2.0e-6);
}

#[test]
fn clarabel_returns_a_checkable_primal_infeasibility_witness() {
    // Primal:
    //   minimize trace(X)
    //   subject to X - I >= 0 and X_11 = 0.
    //
    // The PSD constraint implies X_11 >= 1, so the equality makes the model
    // infeasible. Clarabel's certificate z must satisfy A^T z = 0,
    // z_PSD >= 0, and b^T z < 0.
    let n_psd = 3;
    let p = CscMatrix::zeros((n_psd, n_psd));
    let a = CscMatrix::new_from_triplets(
        4,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2, 0],
        vec![-1.0, -1.0, -1.0, 1.0],
    );
    let b = vec![-1.0, 0.0, -1.0, 0.0];
    let c = vec![1.0, 0.0, 1.0];
    let cones = vec![PSDTriangleConeT(2), ZeroConeT(1)];

    let mut solver =
        DefaultSolver::new(&p, &c, &a, &b, &cones, settings()).expect("valid infeasible SDP");
    solver.solve();

    let solution = &solver.solution;
    assert_eq!(solution.status, SolverStatus::PrimalInfeasible);
    assert_eq!(solution.z.len(), 4);

    // Decode the PSD portion of the dual ray and verify its 2x2 principal
    // minors. The zero-cone multiplier is unrestricted.
    let psd_dual = RealSymmetricMatrix::from_clarabel_psd_triangle(2, &solution.z[..3])
        .expect("decode dual PSD certificate");
    let z00 = psd_dual.get(0, 0).expect("in range");
    let z01 = psd_dual.get(0, 1).expect("in range");
    let z11 = psd_dual.get(1, 1).expect("in range");
    let tolerance = 2.0e-7;
    assert!(z00 >= -tolerance);
    assert!(z11 >= -tolerance);
    assert!(z00 * z11 - z01 * z01 >= -tolerance);

    // A^T z = 0 for the explicitly declared A, and b^T z is strictly negative.
    assert!((solution.z[3] - solution.z[0]).abs() < tolerance);
    assert!(solution.z[0].abs() > tolerance);
    assert!(solution.z[1].abs() < tolerance);
    assert!(solution.z[2].abs() < tolerance);
    let b_dot_z: f64 = b.iter().zip(&solution.z).map(|(bi, zi)| bi * zi).sum();
    assert!(b_dot_z < -tolerance);
}
