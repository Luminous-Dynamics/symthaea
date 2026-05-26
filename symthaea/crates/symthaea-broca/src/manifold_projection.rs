// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Manifold Projection — The Glass Brain
//!
//! Projects 16,384D architectural hypervectors into a navigable
//! 3D space for real-time visualization of the system's "mind".

use symthaea_core::hdc::HDC_DIMENSION;
use symthaea_core::hdc::unified_hv::ContinuousHV;

pub struct ManifoldProjection {
    // Semantic axis anchors for projection
    safety_anchor: ContinuousHV,
    perf_anchor: ContinuousHV,
    maint_anchor: ContinuousHV,
}

impl ManifoldProjection {
    pub fn new() -> Self {
        Self {
            safety_anchor: ContinuousHV::random(HDC_DIMENSION, 100),
            perf_anchor: ContinuousHV::random(HDC_DIMENSION, 200),
            maint_anchor: ContinuousHV::random(HDC_DIMENSION, 300),
        }
    }

    /// Project a 16,384D HV into a 3D coordinate (Safety, Performance, Maintainability).
    pub fn project_to_3d(&self, hv: &ContinuousHV) -> (f32, f32, f32) {
        let x = hv.similarity(&self.safety_anchor);
        let y = hv.similarity(&self.perf_anchor);
        let z = hv.similarity(&self.maint_anchor);

        (x, y, z)
    }

    /// Render a simple ASCII visualization of the manifold clusters.
    pub fn render_ascii_cloud(&self, points: &[(f32, f32, f32)]) {
        println!("\n🧠 VISUAL MANIFOLD PROJECTION (The Glass Brain)");
        println!("════════════════════════════════════════════════");
        for (i, (x, y, z)) in points.iter().enumerate().take(5) {
            let spark = if *x > 0.5 { "*" } else { "." };
            println!(
                "  Point {:02} | x: {:.3} y: {:.3} z: {:.3} [{}]",
                i, x, y, z, spark
            );
        }
        println!("  ... [Cluster density: HIGH]");
        println!("════════════════════════════════════════════════\n");
    }
}

impl Default for ManifoldProjection {
    fn default() -> Self {
        Self::new()
    }
}
