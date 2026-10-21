//! Nonlinear Hamiltonian structure checks for fixed-step integrators.
//!
//! The test uses the canonical Hénon–Heiles Hamiltonian, a separable nonlinear
//! system. A centered finite-difference Jacobian is used only as an independent
//! diagnostic of the implemented discrete map; this is not a formal proof.

use ode_verify_001_harness::dynamical_system::{
    DynamicalSystem, Integrator, Rk4Integrator, VerletIntegrator,
};

const DIM: usize = 4;
const J: [[f64; DIM]; DIM] = [
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
];

struct HenonHeiles;

impl DynamicalSystem for HenonHeiles {
    fn state_dim(&self) -> usize {
        DIM
    }

    fn derivatives(&self, state: &[f64], _t: f64) -> Vec<f64> {
        // H = 1/2 (px^2 + py^2 + x^2 + y^2) + x^2 y - y^3 / 3.
        // State order is [x, y, px, py]; unit masses make p = velocity.
        let x = state[0];
        let y = state[1];
        let px = state[2];
        let py = state[3];

        vec![px, py, -x - 2.0 * x * y, -y - x * x + y * y]
    }

    fn name(&self) -> &str {
        "Hénon–Heiles Hamiltonian"
    }
}

fn step_map(integrator: &dyn Integrator, state: [f64; DIM], dt: f64) -> [f64; DIM] {
    let next = integrator.step(&HenonHeiles, &state, 0.0, dt);
    [next[0], next[1], next[2], next[3]]
}

fn numerical_jacobian(
    integrator: &dyn Integrator,
    state: [f64; DIM],
    dt: f64,
) -> [[f64; DIM]; DIM] {
    const EPSILON: f64 = 1e-5;
    const BASIS: [[f64; DIM]; DIM] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let mut jacobian = [[0.0; DIM]; DIM];

    for (column, direction) in BASIS.iter().enumerate() {
        let mut plus_state = state;
        let mut minus_state = state;

        for ((plus, minus), delta) in plus_state
            .iter_mut()
            .zip(minus_state.iter_mut())
            .zip(direction.iter())
        {
            *plus += EPSILON * delta;
            *minus -= EPSILON * delta;
        }

        let plus = step_map(integrator, plus_state, dt);
        let minus = step_map(integrator, minus_state, dt);

        for ((jacobian_row, plus_value), minus_value) in
            jacobian.iter_mut().zip(plus.iter()).zip(minus.iter())
        {
            jacobian_row[column] = (plus_value - minus_value) / (2.0 * EPSILON);
        }
    }

    jacobian
}

/// Return the max-entry defect of M^T J M - J.
fn symplectic_defect(integrator: &dyn Integrator, state: [f64; DIM], dt: f64) -> f64 {
    let matrix = numerical_jacobian(integrator, state, dt);

    // Multiplication by the canonical J is sparse:
    // J M = [row_2(M); row_3(M); -row_0(M); -row_1(M)].
    let j_times_matrix = [
        matrix[2],
        matrix[3],
        matrix[0].map(|value| -value),
        matrix[1].map(|value| -value),
    ];

    let mut max_defect: f64 = 0.0;
    for (i, row) in matrix.iter().enumerate() {
        for (j, _entry) in row.iter().enumerate() {
            let product = matrix
                .iter()
                .zip(j_times_matrix.iter())
                .map(|(matrix_row, jm_row)| matrix_row[i] * jm_row[j])
                .sum::<f64>();
            max_defect = max_defect.max((product - J[i][j]).abs());
        }
    }
    max_defect
}

#[test]
fn verlet_preserves_nonlinear_hamiltonian_structure_across_frozen_grid() {
    let states = [
        [0.12, 0.20, 0.05, -0.10],
        [0.20, -0.10, 0.12, 0.03],
        [-0.15, 0.08, -0.04, 0.11],
    ];
    let step_sizes = [0.2, 0.4, 0.6];

    for state in states {
        for dt in step_sizes {
            let defect = symplectic_defect(&VerletIntegrator, state, dt);
            assert!(
                defect <= 2e-7,
                "Verlet symplectic defect {defect:.3e} exceeds 2e-7 for state={state:?}, dt={dt}"
            );
        }
    }
}

#[test]
fn rk4_fails_nonlinear_symplecticity_negative_control() {
    let state = [0.12, 0.20, 0.05, -0.10];
    let defect = symplectic_defect(&Rk4Integrator, state, 0.5);
    assert!(
        defect > 1e-4,
        "RK4 negative control did not expose a symplecticity defect: {defect:.3e}"
    );
}
