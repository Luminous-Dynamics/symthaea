//! Reproducible adaptive-resolution quality matrix.
//!
//! This is intentionally an integration harness: it exercises only the
//! public research boundary and cannot mutate production BinaryHV semantics.

use symthaea_core::hdc::adaptive_resolution::{HdcResolution, PackedBinaryHv};
use symthaea_core::hdc::resolution_metrics::hamming_similarity;
use symthaea_core::hdc::resolution_projection::{ProjectionFamily, ProjectionSpec};

fn next_u64(state: &mut u64) -> u64 {
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
    PackedBinaryHv::from_bytes(resolution, bytes).unwrap()
}

fn bind(a: &PackedBinaryHv, b: &PackedBinaryHv) -> PackedBinaryHv {
    PackedBinaryHv::from_bytes(
        a.resolution(),
        a.as_bytes().iter().zip(b.as_bytes()).map(|(x, y)| x ^ y).collect(),
    ).unwrap()
}

fn bundle3(a: &PackedBinaryHv, b: &PackedBinaryHv, c: &PackedBinaryHv) -> PackedBinaryHv {
    let bytes = a.as_bytes().iter().zip(b.as_bytes()).zip(c.as_bytes())
        .map(|((x, y), z)| (x & y) | (x & z) | (y & z))
        .collect();
    PackedBinaryHv::from_bytes(a.resolution(), bytes).unwrap()
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
        preserved += usize::from(source_best == projected_best);
    }
    preserved as f64 / source.len() as f64
}

#[derive(Debug, Clone, Copy)]
struct Measurement {
    source: HdcResolution,
    target: HdcResolution,
    family: ProjectionFamily,
    pair_similarity_mae: f64,
    top1_preservation: f64,
    binding_mae: f64,
    bundle_mae: f64,
}

fn evaluate(seed: u64) -> Vec<Measurement> {
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

                // XOR binding is expected to commute with both linear research
                // projections. Keep this as a measured invariant.
                let source_binding = bind(&source[0], &source[1]);
                let projected_binding = bind(&projected[0], &projected[1]);
                let projected_source_binding = spec.demote(&source_binding).unwrap();
                let binding_mae = 1.0 - hamming_similarity(
                    &projected_binding, &projected_source_binding,
                ).unwrap();

                // Majority bundling is nonlinear; measure distortion instead
                // of assuming that projection preserves it.
                let source_bundle = bundle3(&source[0], &source[1], &source[2]);
                let projected_bundle = bundle3(&projected[0], &projected[1], &projected[2]);
                let projected_source_bundle = spec.demote(&source_bundle).unwrap();
                let bundle_mae = 1.0 - hamming_similarity(
                    &projected_bundle, &projected_source_bundle,
                ).unwrap();

                out.push(Measurement {
                    source: source_resolution,
                    target: target_resolution,
                    family,
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

#[test]
fn matrix_covers_all_strict_pairs_and_families() {
    let rows = evaluate(0x5eed_6531);
    assert_eq!(rows.len(), 42); // 7 choose 2 × 2 projection families
    assert!(rows.iter().all(|r| r.source.bits() > r.target.bits()));
}

#[test]
fn matrix_is_deterministic_under_fixed_seed() {
    let a = evaluate(1234);
    let b = evaluate(1234);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.source, y.source);
        assert_eq!(x.target, y.target);
        assert_eq!(x.family, y.family);
        assert_eq!(x.pair_similarity_mae.to_bits(), y.pair_similarity_mae.to_bits());
        assert_eq!(x.top1_preservation.to_bits(), y.top1_preservation.to_bits());
        assert_eq!(x.binding_mae.to_bits(), y.binding_mae.to_bits());
        assert_eq!(x.bundle_mae.to_bits(), y.bundle_mae.to_bits());
    }
}

#[test]
fn xor_binding_commutes_exactly_with_research_projections() {
    for row in evaluate(0x5eed_6531) {
        assert_eq!(row.binding_mae, 0.0, "{row:?}");
    }
}

#[test]
fn bundle_distortion_is_measured_not_assumed_zero() {
    let rows = evaluate(0x5eed_6531);
    assert!(rows.iter().any(|r| r.bundle_mae > 0.0));
}

#[test]
fn matrix_metrics_are_finite_and_bounded() {
    for row in evaluate(0x5eed_6531) {
        assert!((0.0..=1.0).contains(&row.pair_similarity_mae));
        assert!((0.0..=1.0).contains(&row.top1_preservation));
        assert!((0.0..=1.0).contains(&row.binding_mae));
        assert!((0.0..=1.0).contains(&row.bundle_mae));
    }
}
