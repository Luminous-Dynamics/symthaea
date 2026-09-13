// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Experiment-only orthogonal adapter for HYPERSPACE-004.
//!
//! The target planner sees only an external four-coordinate oracle and a
//! conservative axis-aligned sampling envelope. Validity remains delegated to the
//! canonical finite-w shell after inverse rotation. No external axis is privileged
//! as the canonical extra coordinate.

use std::error::Error;

use blake3::Hasher;
use symthaea_core::continuous_reachability::{
    ContinuousReachabilityError, ContinuousValidityOracle, ContinuousValidityOracleProfile,
    OracleVerdict,
};
use symthaea_core::hyperspace_benchmark::{FiniteWShellOracle, HyperspaceDimension};
use symthaea_core::sampling_reachability::BoundedEuclideanSamplingOracle;
use symthaea_core::state_space::{EuclideanSpace, StateSpace};

pub const ROUNDTRIP_TOLERANCE: f64 = 9.094_947_017_729_282e-13; // 2^-40
pub const MATERIAL_CLEARANCE: f64 = 9.536_743_164_062_5e-7; // 2^-20
const C: f64 = 0.6;
const S: f64 = 0.8;
const ORTHOGONALITY_TOLERANCE: f64 = 3.552_713_678_800_501e-15; // 2^-48
const LENGTH_TOLERANCE: f64 = 9.094_947_017_729_282e-13; // 2^-40
const CLEARANCE_OFFSET: f64 = 9.536_743_164_062_5e-7; // 2^-20

pub type Mat4 = [[f64; 4]; 4];

#[derive(Clone, Debug)]
pub struct TransformSpec {
    pub name: &'static str,
    pub matrix: Mat4,
    pub expected_feasible_transform_identity: &'static str,
}

#[derive(Clone, Debug)]
pub struct RotatedSamplingOracle {
    canonical: FiniteWShellOracle,
    matrix: Mat4,
    inverse: Mat4,
    sampling_min: Vec<f64>,
    sampling_max: Vec<f64>,
    transform_identity: [u8; 32],
    profile: ContinuousValidityOracleProfile,
}

impl RotatedSamplingOracle {
    pub fn new(
        canonical: FiniteWShellOracle,
        matrix: Mat4,
    ) -> Result<Self, Box<dyn Error>> {
        if canonical.benchmark_dimension() != HyperspaceDimension::R4 {
            return Err("orthogonal sampling wrapper requires canonical R4 oracle".into());
        }
        if matrix.iter().flatten().any(|value| !value.is_finite()) {
            return Err("rotation matrix must be finite".into());
        }

        let inverse = transpose(matrix);
        let transform_profile = h003_transform_profile(&canonical, matrix)?;
        let transform_identity = transform_profile.identity();
        let (sampling_min, sampling_max) = conservative_envelope(&canonical, &matrix)?;

        let state_space = EuclideanSpace::new(4);
        let mut parameters = Vec::new();
        parameters.extend_from_slice(&canonical.profile().identity());
        parameters.extend_from_slice(&transform_identity);
        for row in matrix {
            for value in row {
                parameters.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        for value in sampling_min.iter().chain(sampling_max.iter()) {
            parameters.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let profile = ContinuousValidityOracleProfile::new(
            state_space.profile().identity(),
            "hyperspace-rotated-sampling-wrapper-v1",
            parameters,
        )?;

        Ok(Self {
            canonical,
            matrix,
            inverse,
            sampling_min,
            sampling_max,
            transform_identity,
            profile,
        })
    }

    pub fn to_external(
        &self,
        canonical: &[f64],
    ) -> Result<Vec<f64>, ContinuousReachabilityError> {
        mat_vec(&self.matrix, canonical)
    }

    pub fn to_canonical(
        &self,
        external: &[f64],
    ) -> Result<Vec<f64>, ContinuousReachabilityError> {
        mat_vec(&self.inverse, external)
    }

    pub fn transform_identity(&self) -> [u8; 32] {
        self.transform_identity
    }

    pub fn matrix(&self) -> &Mat4 {
        &self.matrix
    }

    pub fn envelope_volume_ratio(&self) -> f64 {
        let external_volume = self
            .sampling_min
            .iter()
            .zip(&self.sampling_max)
            .map(|(min, max)| max - min)
            .product::<f64>();
        let canonical_volume = self
            .canonical
            .sampling_min()
            .iter()
            .zip(self.canonical.sampling_max())
            .map(|(min, max)| max - min)
            .product::<f64>();
        external_volume / canonical_volume
    }
}

impl ContinuousValidityOracle for RotatedSamplingOracle {
    fn profile(&self) -> &ContinuousValidityOracleProfile {
        &self.profile
    }

    fn state_verdict(&self, state: &[f64]) -> Result<OracleVerdict, ContinuousReachabilityError> {
        self.canonical.state_verdict(&self.to_canonical(state)?)
    }

    fn segment_verdict(
        &self,
        from: &[f64],
        to: &[f64],
    ) -> Result<OracleVerdict, ContinuousReachabilityError> {
        self.canonical
            .segment_verdict(&self.to_canonical(from)?, &self.to_canonical(to)?)
    }
}

impl BoundedEuclideanSamplingOracle for RotatedSamplingOracle {
    fn sampling_min(&self) -> &[f64] {
        &self.sampling_min
    }

    fn sampling_max(&self) -> &[f64] {
        &self.sampling_max
    }
}

pub fn frozen_transforms() -> Vec<TransformSpec> {
    let rxw = givens(0, 3);
    let ryw = givens(1, 3);
    let rzw = givens(2, 3);
    vec![
        TransformSpec {
            name: "R_xw",
            matrix: rxw,
            expected_feasible_transform_identity:
                "d474e7d71d5485fb81f2e454dc3193b29422e847a77ae941bfb4daad6a5ce306",
        },
        TransformSpec {
            name: "R_yw",
            matrix: ryw,
            expected_feasible_transform_identity:
                "1901563cb777ec6dc29c7ac4e7ce1a30a4dfb35dbe57704d096d0012a9012fcc",
        },
        TransformSpec {
            name: "R_zw",
            matrix: rzw,
            expected_feasible_transform_identity:
                "4ac03922725954f6e3c77bdc55e445f14fecc51913f88c0d90d8a8adae66d411",
        },
        TransformSpec {
            name: "R_yw_R_xw",
            matrix: mat_mul(&ryw, &rxw),
            expected_feasible_transform_identity:
                "f2ff21fae9cdab4f673ddc75de8686c1533f9241101577eaf62df237aad9834f",
        },
        TransformSpec {
            name: "R_zw_R_xw",
            matrix: mat_mul(&rzw, &rxw),
            expected_feasible_transform_identity:
                "3faa3203f60abed024b7cb14d860674100c37f2f5a901cbdae231069ba9e48ba",
        },
        TransformSpec {
            name: "R_zw_R_yw_R_xw",
            matrix: mat_mul(&rzw, &mat_mul(&ryw, &rxw)),
            expected_feasible_transform_identity:
                "4b9e1f13c37d68d35b0b8bd1cf4eaac948e90956a32f786f667f83a0103b0382",
        },
    ]
}

pub fn matrix_bits(matrix: &Mat4) -> Vec<Vec<String>> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| format!("{:016x}", value.to_bits()))
                .collect()
        })
        .collect()
}

pub fn path_roundtrip_error(
    oracle: &RotatedSamplingOracle,
    external_path: &[Vec<f64>],
) -> Result<f64, ContinuousReachabilityError> {
    let mut worst = 0.0_f64;
    for external in external_path {
        let canonical = oracle.to_canonical(external)?;
        let reconstructed = oracle.to_external(&canonical)?;
        for (&a, &b) in external.iter().zip(&reconstructed) {
            let error = (a - b).abs();
            if !error.is_finite() {
                return Err(ContinuousReachabilityError::OracleEvaluation {
                    reason: "orthogonal roundtrip diagnostic became non-finite".to_string(),
                });
            }
            worst = worst.max(error);
        }
    }
    Ok(worst)
}

fn conservative_envelope(
    canonical: &FiniteWShellOracle,
    matrix: &Mat4,
) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let mins = canonical.sampling_min();
    let maxs = canonical.sampling_max();
    if mins.len() != 4 || maxs.len() != 4 {
        return Err("canonical R4 sampling bounds must contain four coordinates".into());
    }

    let mut bounds = [0.0_f64; 4];
    for axis in 0..4 {
        if mins[axis].to_bits() != (-maxs[axis]).to_bits() {
            return Err("HYPERSPACE-004 requires symmetric canonical sampling bounds".into());
        }
        if !maxs[axis].is_finite() || maxs[axis] <= 0.0 {
            return Err("canonical sampling bounds must be finite and positive".into());
        }
    }

    for row in 0..4 {
        let mut bound = 0.0_f64;
        for column in 0..4 {
            bound += matrix[row][column].abs() * maxs[column];
        }
        if !bound.is_finite() || bound <= 0.0 {
            return Err(format!("invalid rotated sampling envelope on axis {row}").into());
        }
        bounds[row] = bound + 0.0;
    }

    Ok((
        bounds.iter().map(|bound| -bound).collect(),
        bounds.to_vec(),
    ))
}

fn h003_transform_profile(
    canonical: &FiniteWShellOracle,
    matrix: Mat4,
) -> Result<ContinuousValidityOracleProfile, ContinuousReachabilityError> {
    let state_space = EuclideanSpace::new(4);
    let mut parameters = Vec::new();
    parameters.extend_from_slice(&canonical.profile().identity());
    parameters.extend_from_slice(&h003_numerical_policy_identity());
    for row in matrix {
        for value in row {
            parameters.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    ContinuousValidityOracleProfile::new(
        state_space.profile().identity(),
        "hyperspace-orthogonal-mixing-v1",
        parameters,
    )
}

fn h003_numerical_policy_identity() -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-hyperspace-003-numerical-policy-v1\0");
    for value in [
        ORTHOGONALITY_TOLERANCE,
        ROUNDTRIP_TOLERANCE,
        LENGTH_TOLERANCE,
        CLEARANCE_OFFSET,
        C,
        S,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn givens(a: usize, b: usize) -> Mat4 {
    let mut matrix = identity();
    matrix[a][a] = C;
    matrix[a][b] = -S;
    matrix[b][a] = S;
    matrix[b][b] = C;
    matrix
}

fn identity() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn transpose(matrix: Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            out[row][column] = matrix[column][row];
        }
    }
    out
}

fn mat_mul(left: &Mat4, right: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            out[row][column] = (0..4).map(|k| left[row][k] * right[k][column]).sum();
        }
    }
    out
}

fn mat_vec(matrix: &Mat4, vector: &[f64]) -> Result<Vec<f64>, ContinuousReachabilityError> {
    if vector.len() != 4 || vector.iter().any(|value| !value.is_finite()) {
        return Err(ContinuousReachabilityError::OracleEvaluation {
            reason: "orthogonal transport requires exactly four finite coordinates".to_string(),
        });
    }
    let mut out = vec![0.0; 4];
    for row in 0..4 {
        out[row] = (0..4)
            .map(|column| matrix[row][column] * vector[column])
            .sum();
        if !out[row].is_finite() {
            return Err(ContinuousReachabilityError::OracleEvaluation {
                reason: "orthogonal transport produced non-finite coordinate".to_string(),
            });
        }
        out[row] += 0.0;
    }
    Ok(out)
}
