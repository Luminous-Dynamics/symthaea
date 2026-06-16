// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Criterion benchmarks for the vision manifold pipeline.
//!
//! Validates 50Hz (20ms budget) feasibility for real frame sizes.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use symthaea_vision_manifold::encoder::MotionField;
use symthaea_vision_manifold::{
    MultiScaleEncoder, PatchHdcEncoder, VisionBridge, VisionConfig, VisionManifold,
};

fn gradient_frame(width: u32, height: u32) -> Vec<u8> {
    (0..width * height).map(|i| (i % 256) as u8).collect()
}

fn rgb_frame(width: u32, height: u32) -> Vec<u8> {
    (0..width * height)
        .flat_map(|i| {
            let v = (i % 256) as u8;
            [v, v / 2, 255 - v]
        })
        .collect()
}

fn bench_single_frame_encode(c: &mut Criterion) {
    let cfg = VisionConfig::default();
    let mut encoder = PatchHdcEncoder::new(&cfg, 128, 128);
    let frame = gradient_frame(128, 128);

    c.bench_function("encode_frame_128x128_gray", |b| {
        b.iter(|| {
            let (hv, patches) = encoder.encode_frame(black_box(&frame), 128, 128, 1);
            black_box((hv, patches));
        })
    });
}

fn bench_multiscale_encode(c: &mut Criterion) {
    let cfg = VisionConfig::default();
    let mut encoder = MultiScaleEncoder::new(&cfg, 128, 128);
    let frame = gradient_frame(128, 128);

    c.bench_function("multiscale_encode_128x128_gray", |b| {
        b.iter(|| {
            let result = encoder.encode_frame(black_box(&frame), 128, 128, 1);
            black_box(result);
        })
    });
}

fn bench_cfc_step(c: &mut Criterion) {
    let cfg = VisionConfig::default();
    let mut manifold = VisionManifold::new(cfg, 128, 128);
    let frame = gradient_frame(128, 128);
    // Warm up with one frame
    manifold.observe_frame(&frame, 128, 128, 1, 0.033);

    c.bench_function("cfc_observe_frame_128x128", |b| {
        b.iter(|| {
            let tel = manifold.observe_frame(black_box(&frame), 128, 128, 1, 0.033);
            black_box(tel);
        })
    });
}

fn bench_motion_field(c: &mut Criterion) {
    let cfg = VisionConfig::default();
    let encoder = PatchHdcEncoder::new(&cfg, 128, 128);
    let motion = MotionField::new(cfg.hdc_dim, cfg.seed + 500_000);

    let frame_a = gradient_frame(128, 128);
    let frame_b: Vec<u8> = frame_a.iter().map(|&v| v.wrapping_add(10)).collect();

    let grid = encoder.grid_for(128, 128);
    let lum_a: Vec<f32> = frame_a.iter().map(|&v| v as f32 / 255.0).collect();
    let lum_b: Vec<f32> = frame_b.iter().map(|&v| v as f32 / 255.0).collect();

    // Compute per-patch luminance
    let rows = grid.rows;
    let cols = grid.cols;
    let ps = cfg.patch_size;
    let w = 128usize;
    let n_patches = rows * cols;
    let mut patch_lum_a = vec![0.0f32; n_patches];
    let mut patch_lum_b = vec![0.0f32; n_patches];
    for r in 0..rows {
        for c_idx in 0..cols {
            let mut sum_a = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut count = 0;
            for py in 0..ps {
                for px in 0..ps {
                    let y = r * ps + py;
                    let x = c_idx * ps + px;
                    let idx = y * w + x;
                    if idx < lum_a.len() {
                        sum_a += lum_a[idx];
                        sum_b += lum_b[idx];
                        count += 1;
                    }
                }
            }
            let pidx = r * cols + c_idx;
            if count > 0 {
                patch_lum_a[pidx] = sum_a / count as f32;
                patch_lum_b[pidx] = sum_b / count as f32;
            }
        }
    }

    c.bench_function("motion_field_128x128", |b| {
        b.iter(|| {
            let result = motion.compute(
                black_box(&patch_lum_b),
                black_box(&patch_lum_a),
                rows,
                cols,
                encoder.row_basis(),
                encoder.col_basis(),
            );
            black_box(result);
        })
    });
}

fn bench_full_pipeline_bridge(c: &mut Criterion) {
    let cfg = VisionConfig::default();
    let mut bridge = VisionBridge::new(cfg, 128, 128);
    let frame = gradient_frame(128, 128);
    // Warm up
    bridge.process_frame(&frame, 128, 128, 1, 0.033);

    c.bench_function("bridge_full_pipeline_128x128", |b| {
        b.iter(|| {
            let hv = bridge.process_frame(black_box(&frame), 128, 128, 1, 0.033);
            black_box(hv);
        })
    });
}

fn bench_rgb_full_pipeline(c: &mut Criterion) {
    let cfg = VisionConfig::default();
    let mut bridge = VisionBridge::new(cfg, 128, 128);
    let frame = rgb_frame(128, 128);
    // Warm up
    bridge.process_frame(&frame, 128, 128, 3, 0.033);

    c.bench_function("bridge_full_pipeline_128x128_rgb", |b| {
        b.iter(|| {
            let hv = bridge.process_frame(black_box(&frame), 128, 128, 3, 0.033);
            black_box(hv);
        })
    });
}

criterion_group!(
    benches,
    bench_single_frame_encode,
    bench_multiscale_encode,
    bench_cfc_step,
    bench_motion_field,
    bench_full_pipeline_bridge,
    bench_rgb_full_pipeline,
);
criterion_main!(benches);
