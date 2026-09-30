//! Machine-readable scalar/SIMD operator conformance evidence.
//!
//! Research-only evidence schema. The records are generated from deterministic
//! inputs so later trajectory qualification can consume operator evidence
//! without duplicating the scalar-oracle logic.
//!
//! This module does not claim benchmark performance: `logical_bytes` is a
//! declared operation-level accounting quantity, not measured memory traffic.

use serde::{Deserialize, Serialize};

use super::{
    resolution_space::HdcResolution,
    simd_continuous::{bind_simd, bundle_simd, dot_product_simd, norm_simd, similarity_simd},
};

pub const EVIDENCE_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_SEED_A: u64 = 42;
pub const DEFAULT_SEED_B: u64 = 43;

const EXTENDED_DIMS: &[usize] = &[16_384, 32_768, 65_536, 131_072, 262_144];
const TOLERANCE_SCALAR: f64 = 1e-4;
const TOLERANCE_VECTOR: f64 = 1e-5;

/// One deterministic operator qualification observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatorEvidenceRecord {
    pub schema_version: u32,
    pub representation: &'static str,
    pub resolution: usize,
    pub operation: &'static str,
    pub seed_a: u64,
    pub seed_b: u64,
    /// Scalar-oracle value. For vector-valued operations this is a deterministic
    /// checksum (sum of elements), while max_abs_error is the fidelity metric.
    pub scalar_reference: f64,
    /// SIMD value using the same convention as scalar_reference.
    pub simd_result: f64,
    pub abs_error: f64,
    pub relative_error: f64,
    pub max_abs_error: f64,
    pub max_relative_error: f64,
    pub tolerance: f64,
    pub logical_bytes: usize,
    pub qualification_status: &'static str,
}

/// Generate the deterministic operator matrix used by the >64K evidence gate.
///
/// The returned records are suitable for JSON serialization and downstream
/// trajectory qualification. A failed record is retained rather than filtered
/// out, so evidence consumers cannot mistake "missing" for "qualified".
pub fn generate_extended_resolution_evidence() -> Vec<OperatorEvidenceRecord> {
    let mut records = Vec::with_capacity(EXTENDED_DIMS.len() * 5);

    for &dim in EXTENDED_DIMS {
        let a = deterministic_vec(dim, DEFAULT_SEED_A);
        let b = deterministic_vec(dim, DEFAULT_SEED_B);

        let scalar_dot = scalar_dot(&a, &b);
        let simd_dot = dot_product_simd(&a, &b);
        records.push(scalar_record(
            dim,
            "dot",
            scalar_dot,
            simd_dot,
            2 * dim * std::mem::size_of::<f32>(),
            TOLERANCE_SCALAR,
        ));

        let scalar_bind = scalar_bind(&a, &b);
        let simd_bind = bind_simd(&a, &b);
        records.push(vector_record(
            dim,
            "bind",
            &scalar_bind,
            &simd_bind,
            3 * dim * std::mem::size_of::<f32>(),
            TOLERANCE_VECTOR,
        ));

        let refs = vec![a.as_slice(), b.as_slice()];
        let weights = [1.0f32, 2.0];
        let scalar_bundle = scalar_bundle(&a, &b);
        let simd_bundle = bundle_simd(&refs, &weights);
        records.push(vector_record(
            dim,
            "bundle",
            &scalar_bundle,
            &simd_bundle,
            3 * dim * std::mem::size_of::<f32>(),
            TOLERANCE_VECTOR,
        ));

        let scalar_norm = scalar_norm(&a);
        let simd_norm = norm_simd(&a);
        records.push(scalar_record(
            dim,
            "norm",
            scalar_norm,
            simd_norm,
            dim * std::mem::size_of::<f32>(),
            TOLERANCE_SCALAR,
        ));

        let scalar_similarity = scalar_similarity(&a, &b);
        let simd_similarity = similarity_simd(&a, &b);
        records.push(scalar_record(
            dim,
            "similarity",
            scalar_similarity,
            simd_similarity,
            2 * dim * std::mem::size_of::<f32>(),
            TOLERANCE_SCALAR,
        ));
    }

    records
}

/// Serialize the current evidence matrix as deterministic pretty JSON.
pub fn extended_resolution_evidence_json() -> String {
    serde_json::to_string_pretty(&generate_extended_resolution_evidence())
        .expect("operator evidence records are JSON-serializable")
}

fn scalar_record(
    dim: usize,
    operation: &'static str,
    scalar: f32,
    simd: f32,
    logical_bytes: usize,
    tolerance: f64,
) -> OperatorEvidenceRecord {
    let abs_error = f64::from((simd - scalar).abs());
    let relative_error = abs_error / f64::from(scalar.abs().max(1.0));
    let status = if relative_error <= tolerance {
        "qualified"
    } else {
        "failed"
    };

    OperatorEvidenceRecord {
        schema_version: EVIDENCE_SCHEMA_VERSION,
        representation: "continuous_f32",
        resolution: dim,
        operation,
        seed_a: DEFAULT_SEED_A,
        seed_b: DEFAULT_SEED_B,
        scalar_reference: f64::from(scalar),
        simd_result: f64::from(simd),
        abs_error,
        relative_error,
        max_abs_error: abs_error,
        max_relative_error: relative_error,
        tolerance,
        logical_bytes,
        qualification_status: status,
    }
}

fn vector_record(
    dim: usize,
    operation: &'static str,
    scalar: &[f32],
    simd: &[f32],
    logical_bytes: usize,
    tolerance: f64,
) -> OperatorEvidenceRecord {
    assert_eq!(scalar.len(), simd.len(), "evidence oracle length mismatch");

    let mut max_abs_error = 0.0f64;
    let mut max_relative_error = 0.0f64;
    for (&expected, &actual) in scalar.iter().zip(simd.iter()) {
        let abs_error = f64::from((actual - expected).abs());
        let relative_error = abs_error / f64::from(expected.abs().max(1.0));
        max_abs_error = max_abs_error.max(abs_error);
        max_relative_error = max_relative_error.max(relative_error);
    }

    let scalar_checksum = scalar.iter().map(|&x| f64::from(x)).sum();
    let simd_checksum = simd.iter().map(|&x| f64::from(x)).sum();
    let abs_error = (simd_checksum - scalar_checksum).abs();
    let relative_error = abs_error / scalar_checksum.abs().max(1.0);
    let status = if max_abs_error <= tolerance && max_relative_error <= tolerance {
        "qualified"
    } else {
        "failed"
    };

    OperatorEvidenceRecord {
        schema_version: EVIDENCE_SCHEMA_VERSION,
        representation: "continuous_f32",
        resolution: dim,
        operation,
        seed_a: DEFAULT_SEED_A,
        seed_b: DEFAULT_SEED_B,
        scalar_reference: scalar_checksum,
        simd_result: simd_checksum,
        abs_error,
        relative_error,
        max_abs_error,
        max_relative_error,
        tolerance,
        logical_bytes,
        qualification_status: status,
    }
}

fn deterministic_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut values = Vec::with_capacity(dim);

    for _ in 0..dim {
        // SplitMix64-style deterministic stream; conversion uses the high 24
        // bits so the f32 input sequence is stable and inexpensive.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let unit = ((z >> 40) as u32) as f32 / ((1u32 << 24) - 1) as f32;
        values.push(unit * 2.0 - 1.0);
    }

    values
}

fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn scalar_bind(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| x * y).collect()
}

fn scalar_bundle(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x + 2.0 * y) / 3.0)
        .collect()
}

fn scalar_norm(a: &[f32]) -> f32 {
    a.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn scalar_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_ab = scalar_dot(a, b);
    let dot_aa = scalar_dot(a, a);
    let dot_bb = scalar_dot(b, b);
    (dot_ab / (dot_aa * dot_bb).sqrt()).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_matrix_has_stable_shape_and_schema() {
        let records = generate_extended_resolution_evidence();
        assert_eq!(records.len(), EXTENDED_DIMS.len() * 5);
        assert!(records
            .iter()
            .all(|record| record.schema_version == EVIDENCE_SCHEMA_VERSION));
        assert!(records.iter().all(|record| record.representation == "continuous_f32"));
    }

    #[test]
    fn evidence_covers_every_operator_at_every_resolution() {
        for &dim in EXTENDED_DIMS {
            let ops: Vec<_> = generate_extended_resolution_evidence()
                .into_iter()
                .filter(|record| record.resolution == dim)
                .map(|record| record.operation)
                .collect();
            assert_eq!(ops, vec!["dot", "bind", "bundle", "norm", "similarity"]);
        }
    }

    #[test]
    fn evidence_is_machine_readable_and_round_trips() {
        let json = extended_resolution_evidence_json();
        let records: Vec<OperatorEvidenceRecord> =
            serde_json::from_str(&json).expect("evidence JSON should deserialize");
        assert_eq!(records, generate_extended_resolution_evidence());
    }

    #[test]
    fn evidence_reports_all_current_records_as_qualified() {
        let records = generate_extended_resolution_evidence();
        assert!(records
            .iter()
            .all(|record| record.qualification_status == "qualified"));
    }

    #[test]
    fn logical_bytes_match_operation_contract() {
        let records = generate_extended_resolution_evidence();
        for record in records {
            let dim = record.resolution;
            let expected = match record.operation {
                "dot" | "similarity" => 2 * dim * std::mem::size_of::<f32>(),
                "bind" | "bundle" => 3 * dim * std::mem::size_of::<f32>(),
                "norm" => dim * std::mem::size_of::<f32>(),
                _ => unreachable!(),
            };
            assert_eq!(record.logical_bytes, expected);
        }
    }

    #[test]
    fn resolution_metadata_is_consistent_with_evidence_ladder() {
        for &dim in EXTENDED_DIMS {
            let resolution = HdcResolution::new(dim).expect("matrix dimension must be valid");
            assert!(resolution.is_canonical() || resolution.is_exploratory());
        }
    }
}
