//! Research-only machine-readable HDC-LTC resolution projection matrix.
//!
//! Run with:
//!   cargo run -p symthaea-core --example hdc_ltc_resolution_matrix > matrix.json
//!
//! The output is intentionally data, not a quality verdict. Production adaptive
//! resizing must remain evidence-gated until trajectory-level measurements exist.

use serde::Serialize;
use symthaea_core::hdc::{
    continuous_resolution_projection::{project, ContinuousProjectionFamily},
    ContinuousHV,
};

const DIMS: [usize; 7] = [1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536];
const SEED: u64 = 0x4844_432d_4d415431;

#[derive(Debug, Serialize)]
struct Row {
    schema_version: &'static str,
    source_dim: usize,
    target_dim: usize,
    family: &'static str,
    seed: u64,
    energy_retained: f32,
    binding_error: f32,
    bundle_error: f32,
    permutation_error: f32,
}

fn normalized_l2(a: &ContinuousHV, b: &ContinuousHV) -> f32 {
    a.subtract(b).norm() / a.norm().max(1e-12)
}

fn row(source_dim: usize, target_dim: usize, seed: u64, family: &str) -> Row {
    let a = ContinuousHV::random(source_dim, seed);
    let b = ContinuousHV::random(source_dim, seed.wrapping_add(1));
    let c = ContinuousHV::random(source_dim, seed.wrapping_add(2));

    let (pa, pb, pc, projected_binding, projected_bundle, projected_permutation) =
        if family == "hadamard_truncate_v1" {
            let pa = project(&a, target_dim, ContinuousProjectionFamily::HadamardTruncateV1)
                .expect("strict power-of-two contraction");
            let pb = project(&b, target_dim, ContinuousProjectionFamily::HadamardTruncateV1)
                .expect("strict power-of-two contraction");
            let pc = project(&c, target_dim, ContinuousProjectionFamily::HadamardTruncateV1)
                .expect("strict power-of-two contraction");
            let binding = project(
                &a.bind(&b),
                target_dim,
                ContinuousProjectionFamily::HadamardTruncateV1,
            ).expect("strict power-of-two contraction");
            let bundle = project(
                &ContinuousHV::bundle(&[&a, &b, &c]),
                target_dim,
                ContinuousProjectionFamily::HadamardTruncateV1,
            ).expect("strict power-of-two contraction");
            let permutation = project(
                &a.permute(1),
                target_dim,
                ContinuousProjectionFamily::HadamardTruncateV1,
            ).expect("strict power-of-two contraction");
            (pa, pb, pc, binding, bundle, permutation)
        } else {
            let pa = a.dilate(target_dim);
            let pb = b.dilate(target_dim);
            let pc = c.dilate(target_dim);
            let binding = a.bind(&b).dilate(target_dim);
            let bundle = ContinuousHV::bundle(&[&a, &b, &c]).dilate(target_dim);
            let permutation = a.permute(1).dilate(target_dim);
            (pa, pb, pc, binding, bundle, permutation)
        };

    let energy_before = a.norm().powi(2);
    let energy_after = pa.norm().powi(2);

    Row {
        schema_version: "hdc-ltc-resolution-projection-matrix.v1",
        source_dim,
        target_dim,
        family,
        seed,
        energy_retained: energy_after / energy_before.max(1e-12),
        binding_error: normalized_l2(&projected_binding, &pa.bind(&pb)),
        bundle_error: normalized_l2(
            &projected_bundle,
            &ContinuousHV::bundle(&[&pa, &pb, &pc]),
        ),
        permutation_error: normalized_l2(&projected_permutation, &pa.permute(1)),
    }
}

fn main() {
    let mut rows = Vec::with_capacity(42);

    for (source_index, &source_dim) in DIMS.iter().enumerate().skip(1) {
        for (target_index, &target_dim) in DIMS.iter().enumerate().take(source_index) {
            let seed = SEED
                .wrapping_add((source_index as u64) << 32)
                .wrapping_add(target_index as u64);
            rows.push(row(source_dim, target_dim, seed, "legacy_dilate"));
            rows.push(row(source_dim, target_dim, seed, "hadamard_truncate_v1"));
        }
    }

    let payload = serde_json::json!({
        "schema_version": "hdc-ltc-resolution-projection-matrix.v1",
        "dimensions": DIMS,
        "transition_count": rows.len(),
        "families": ["legacy_dilate", "hadamard_truncate_v1"],
        "strict_contraction_transition_count_per_family": 21,
        "policy": {
            "purpose": "characterization",
            "unknown_is_not_zero": true,
            "no_production_quality_gate": true
        },
        "rows": rows
    });

    println!("{}", serde_json::to_string_pretty(&payload).expect("serialize matrix"));
}
