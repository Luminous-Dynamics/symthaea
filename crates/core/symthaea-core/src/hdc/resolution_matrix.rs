//! Deterministic resolution-quality matrix for adaptive HDC research.
//!
//! The matrix measures observable geometry and algebraic distortion across the
//! full 1K..64K ladder. It is deliberately independent of production
//! `BinaryHV` so a favorable result cannot silently change the 16K ABI.

use super::adaptive_resolution::{HdcResolution, PackedBinaryHv};
use super::resolution_metrics::hamming_similarity;
use super::resolution_projection::{ProjectionFamily, ProjectionSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolutionMatrixRow {
    pub source: HdcResolution,
    pub target: HdcResolution,
    pub family: ProjectionFamily,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolutionMatrixMeasurement {
    pub row: ResolutionMatrixRow,
    pub pair_similarity_mae: f64,
    pub top1_preservation: f64,
    pub binding_mae: f64,
    pub bundle_mae: f64,
}

fn next_u64(state: &mut u64) -> u64 {
    // SplitMix64: deterministic, fast, and independent of platform RNGs.
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn fixture(resolution: HdcResolution, seed: u64) -> PackedBinaryHv {
    let mut state = seed;
    let bytes = (0..resolution.bytes())
        .map(|_| next_u64(&mut state) as u8)
        .collect();
    PackedBinaryHv::from_bytes(resolution, bytes).expect("fixture length is exact")
}

fn bind(a: &PackedBinaryHv, b: &PackedBinaryHv) -> PackedBinaryHv {
    assert_eq!(a.resolution(), b.resolution());
    PackedBinaryHv::from_bytes(
        a.resolution(),
        a.as_bytes().iter().zip(b.as_bytes()).map(|(x, y)| x ^ y).collect(),
    ).expect("bind preserves length")
}

fn bundle3(a: &PackedBinaryHv, b: &PackedBinaryHv, c: &PackedBinaryHv) -> PackedBinaryHv {
    assert_eq!(a.resolution(), b.resolution());
    assert_eq!(a.resolution(), c.resolution());
    let bytes = a.as_bytes().iter().zip(b.as_bytes()).zip(c.as_bytes())
        .map(|((x, y), z)| {
            // Bitwise majority without per-bit loops.
            (x & y) | (x & z) | (y & z)
        })
        .collect();
    PackedBinaryHv::from_bytes(a.resolution(), bytes).expect("bundle preserves length")
}

fn pair_mae(source: &[PackedBinaryHv], projected: &[PackedBinaryHv]) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for i in 0..source.len() {
        for j in (i + 1)..source.len() {
            let s = hamming_similarity(&source[i], &source[j]).unwrap();
            let p = hamming_similarity(&projected[i], &projected[j]).unwrap();
            total += (s - p).abs();
            count += 1;
        }
    }
    total / count as f64
}

fn top1_preservation(source: &[PackedBinaryHv], projected: &[PackedBinaryHv]) -> f64 {
    let mut preserved = 0usize;
    for q in 0..source.len() {
        let source_best = (0..source.len()).filter(|&i| i != q)
            .max_by(|&a, &b| hamming_similarity(&source[q], &source[a])
                .partial_cmp(&hamming_similarity(&source[q], &source[b])).unwrap())
            .unwrap();
        let projected_best = (0..projected.len()).filter(|&i| i != q)
            .max_by(|&a, &b| hamming_similarity(&projected[q], &projected[a])
                .partial_cmp(&hamming_similarity(&projected[q], &projected[b])).unwrap())
            .unwrap();
        if source_best == projected_best {
            preserved += 1;
        }
    }
    preserved as f64 / source.len() as f64
}

/// Evaluate every strict demotion in the 1K..64K ladder for each projection
/// family. The fixture seed is explicit, making results reproducible.
pub fn evaluate_matrix(seed: u64) -> Vec<ResolutionMatrixMeasurement> {
    let mut out = Vec::new();

    for &source_resolution in HdcResolution::ALL.iter().rev() {
        let source = (0..16)
            .map(|i| fixture(source_resolution, seed.wrapping_add(i as u64)))
            .collect::<Vec<_>>();

        for &target_resolution in HdcResolution::ALL.iter() {
            if target_resolution.bits() >= source_resolution.bits() {
                continue;
            }

            for family in [ProjectionFamily::Prefix, ProjectionFamily::FoldXor] {
                let spec = ProjectionSpec::new(family, source_resolution, target_resolution);
                let projected = source.iter().map(|v| spec.demote(v).unwrap()).collect::<Vec<_>>();

                let source_binding = bind(&source[0], &source[1]);
                let projected_binding = bind(&projected[0], &projected[1]);
                let projected_source_binding = spec.demote(&source_binding).unwrap();
                let binding_mae = 1.0 - hamming_similarity(
                    &projected_binding,
                    &projected_source_binding,
                ).unwrap();

                let source_bundle = bundle3(&source[0], &source[1], &source[2]);
                let projected_bundle = bundle3(&projected[0], &projected[1], &projected[2]);
                let projected_source_bundle = spec.demote(&source_bundle).unwrap();
                let bundle_mae = 1.0 - hamming_similarity(
                    &projected_bundle,
                    &projected_source_bundle,
                ).unwrap();

                out.push(ResolutionMatrixMeasurement {
                    row: ResolutionMatrixRow { source: source_resolution, target: target_resolution, family },
                    pair_similarity_mae: pair_mae(&source, &projected),
                    top1_preservation: top1_preservation(&source, &projected),
                    binding_mae,
                    bundle_mae,
                });
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_covers_every_strict_resolution_pair_and_family() {
        let rows = evaluate_matrix(0x5eed_6531);
        // 7 choose 2 strict source/target pairs × 2 families.
        assert_eq!(rows.len(), 42);
        assert!(rows.iter().all(|r| r.row.source.bits() > r.row.target.bits()));
    }

    #[test]
    fn projection_is_deterministic_for_a_fixed_seed() {
        assert_eq!(evaluate_matrix(1234), evaluate_matrix(1234));
        assert_ne!(evaluate_matrix(1234), evaluate_matrix(5678));
    }

    #[test]
    fn xor_binding_commutes_with_both_research_projections() {
        for row in evaluate_matrix(0x5eed_6531) {
            assert_eq!(row.binding_mae, 0.0, "{:?}", row.row);
        }
    }

    #[test]
    fn bundle_distortion_is_measured_not_assumed_zero() {
        let rows = evaluate_matrix(0x5eed_6531);
        assert!(rows.iter().any(|r| r.bundle_mae > 0.0));
    }

    #[test]
    fn matrix_metrics_are_finite_and_bounded() {
        for row in evaluate_matrix(0x5eed_6531) {
            assert!(row.pair_similarity_mae.is_finite());
            assert!((0.0..=1.0).contains(&row.pair_similarity_mae));
            assert!((0.0..=1.0).contains(&row.top1_preservation));
            assert!((0.0..=1.0).contains(&row.binding_mae));
            assert!((0.0..=1.0).contains(&row.bundle_mae));
        }
    }
}
