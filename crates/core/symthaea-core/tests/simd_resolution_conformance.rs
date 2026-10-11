//! Resolution-generic SIMD conformance for the 1K..64K adaptive HDC ladder.
//!
//! This is deliberately a correctness contract, not a performance claim.
//! The scalar implementations are the oracle; SIMD is required to remain
//! finite and numerically close across every production research dimension.
//!
//! Run with:
//!   cargo test -p symthaea-core --test simd_resolution_conformance
//!
//! The test also includes non-power-of-two tail dimensions so that SIMD code
//! cannot accidentally rely on the adaptive ladder being perfectly aligned.

#![cfg(feature = "simd")]

use symthaea_core::hdc::{
    simd_continuous::{
        bind_simd, bundle_simd, dot_product_simd, norm_simd, similarity_simd,
    },
    ContinuousHV, HdcLtcUnifiedNeuron, UnifiedActivation, UnifiedConfig,
};

const DIMS: &[usize] = &[1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536];
const TAIL_DIMS: &[usize] = &[1_003, 1_027, 4_099, 16_387];

fn values(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = seed
                .wrapping_add((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                .rotate_left((i % 63) as u32);
            let unit = (x as f64 / u64::MAX as f64) as f32;
            (unit * 2.0 - 1.0) * (1.0 + ((i % 17) as f32) * 0.01)
        })
        .collect()
}

fn assert_close(a: f32, b: f32, abs_tol: f32, rel_tol: f32, label: &str) {
    let scale = a.abs().max(b.abs()).max(1.0);
    let err = (a - b).abs();
    assert!(
        err <= abs_tol.max(rel_tol * scale),
        "{label}: a={a:?}, b={b:?}, abs_err={err:?}, scale={scale:?}"
    );
}

fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn scalar_norm(a: &[f32]) -> f32 {
    scalar_dot(a, a).sqrt()
}

fn scalar_bind(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| x * y).collect()
}

fn scalar_bundle(hvs: &[&[f32]], weights: &[f32]) -> Vec<f32> {
    let weight_sum: f32 = weights.iter().sum();
    let inv = if weight_sum.abs() > 1e-10 {
        1.0 / weight_sum
    } else {
        0.0
    };
    (0..hvs[0].len())
        .map(|i| {
            hvs.iter()
                .zip(weights)
                .map(|(hv, &w)| hv[i] * w)
                .sum::<f32>()
                * inv
        })
        .collect()
}

fn scalar_similarity(a: &[f32], b: &[f32]) -> f32 {
    let denom = scalar_norm(a) * scalar_norm(b);
    if denom <= 1e-12 {
        0.0
    } else {
        scalar_dot(a, b) / denom
    }
}

#[test]
fn simd_matches_scalar_across_adaptive_resolution_ladder() {
    for &dim in DIMS {
        let a = values(dim, 0xA11CE);
        let b = values(dim, 0xB0B);

        assert_close(
            dot_product_simd(&a, &b),
            scalar_dot(&a, &b),
            2e-3,
            2e-5,
            &format!("dot dim={dim}"),
        );
        assert_close(
            norm_simd(&a),
            scalar_norm(&a),
            2e-3,
            2e-5,
            &format!("norm dim={dim}"),
        );
        assert_close(
            similarity_simd(&a, &b),
            scalar_similarity(&a, &b),
            2e-5,
            2e-5,
            &format!("similarity dim={dim}"),
        );

        let bound = bind_simd(&a, &b);
        let scalar_bound = scalar_bind(&a, &b);
        assert_eq!(bound.len(), dim);
        for (i, (&got, &want)) in bound.iter().zip(&scalar_bound).enumerate() {
            assert_close(got, want, 0.0, 0.0, &format!("bind dim={dim} i={i}"));
        }

        let c = values(dim, 0xC0FFEE);
        let hvs = [&a[..], &b[..], &c[..]];
        let weights = [0.5, 0.3, 0.2];
        let bundled = bundle_simd(&hvs, &weights);
        let scalar_bundled = scalar_bundle(&hvs, &weights);
        assert_eq!(bundled.len(), dim);
        for (i, (&got, &want)) in bundled.iter().zip(&scalar_bundled).enumerate() {
            assert_close(got, want, 2e-5, 2e-5, &format!("bundle dim={dim} i={i}"));
        }
    }
}

#[test]
fn simd_handles_non_ladder_tail_dimensions() {
    for &dim in TAIL_DIMS {
        let a = values(dim, 0x1234);
        let b = values(dim, 0x5678);

        assert_close(
            dot_product_simd(&a, &b),
            scalar_dot(&a, &b),
            2e-3,
            2e-5,
            &format!("tail dot dim={dim}"),
        );
        assert_close(
            norm_simd(&a),
            scalar_norm(&a),
            2e-3,
            2e-5,
            &format!("tail norm dim={dim}"),
        );

        let bound = bind_simd(&a, &b);
        let scalar_bound = scalar_bind(&a, &b);
        assert_eq!(bound, scalar_bound, "tail bind dim={dim}");

        let hvs = [&a[..], &b[..]];
        let weights = [0.7, 0.3];
        let bundled = bundle_simd(&hvs, &weights);
        let scalar_bundled = scalar_bundle(&hvs, &weights);
        for (i, (&got, &want)) in bundled.iter().zip(&scalar_bundled).enumerate() {
            assert_close(
                got,
                want,
                2e-5,
                2e-5,
                &format!("tail bundle dim={dim} i={i}"),
            );
        }
    }
}

fn config(dim: usize) -> UnifiedConfig {
    UnifiedConfig {
        dimension: dim,
        activation: UnifiedActivation::Tanh,
        tau_base: 0.1,
        backbone_tau: 0.5,
        learning_rate: 0.01,
        momentum: 0.9,
        weight_decay: 0.0001,
        gating_steepness: 1.0,
        interp_bias: 0.0,
        fourier_frequencies: Vec::new(),
        fourier_amplitude: 0.1,
    }
}

#[test]
fn fused_liquid_evolution_is_resolution_generic() {
    let dts = [0.001, 0.017, 0.031, 0.007];

    for &dim in DIMS {
        let mut reference = HdcLtcUnifiedNeuron::new(config(dim), 0xCAFE);
        let mut fused = HdcLtcUnifiedNeuron::new(config(dim), 0xCAFE);

        for (step, &dt) in dts.iter().enumerate() {
            let input = ContinuousHV::random(
                dim,
                0xFACE_0000u64.wrapping_add(step as u64 * 7919),
            );
            reference.evolve_closed_form(dt, &input);
            fused.evolve_closed_form_fused(dt, &input);

            let similarity = reference.state().similarity(fused.state());
            let state_error = reference
                .state()
                .subtract(fused.state())
                .norm()
                / reference.state().norm().max(1e-12);

            assert!(similarity.is_finite(), "non-finite similarity dim={dim}");
            assert!(state_error.is_finite(), "non-finite state error dim={dim}");
            assert!(
                state_error < 0.02,
                "fused evolution drift too large at dim={dim}, step={step}: {state_error}"
            );
        }
    }
}
