// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based tests for infill generation invariants.

use proptest::prelude::*;
use symthaea_fabrication_kernel::{
    infill::{InfillConfig, InfillPattern, generate_infill},
    slicer::{Contour, Point2, SliceLayer},
};

fn square_layer(size: f32) -> SliceLayer {
    SliceLayer {
        z: 0.2,
        outer_contours: vec![Contour {
            points: vec![
                Point2::new(0.0, 0.0),
                Point2::new(size, 0.0),
                Point2::new(size, size),
                Point2::new(0.0, size),
            ],
        }],
        inner_contours: vec![],
        infill_lines: vec![],
    }
}

proptest! {
    /// All infill endpoints must lie within the contour bounding box (with tolerance).
    #[test]
    fn infill_endpoints_within_bounds(
        size in 5.0f32..50.0,
        density in 0.05f32..1.0,
        angle in 0.0f32..180.0,
    ) {
        let layer = square_layer(size);
        let config = InfillConfig {
            pattern: InfillPattern::Rectilinear,
            density,
            angle_degrees: angle,
        };
        let lines = generate_infill(&layer, &config, 0.4);
        let tol = 0.5; // generous tolerance for edge effects
        for seg in &lines {
            prop_assert!(seg.start.x >= -tol && seg.start.x <= size + tol,
                "start.x={} out of [0, {}]", seg.start.x, size);
            prop_assert!(seg.start.y >= -tol && seg.start.y <= size + tol,
                "start.y={} out of [0, {}]", seg.start.y, size);
            prop_assert!(seg.end.x >= -tol && seg.end.x <= size + tol,
                "end.x={} out of [0, {}]", seg.end.x, size);
            prop_assert!(seg.end.y >= -tol && seg.end.y <= size + tol,
                "end.y={} out of [0, {}]", seg.end.y, size);
        }
    }

    /// Infill line count should increase monotonically with density.
    #[test]
    fn infill_monotonic_with_density(
        size in 10.0f32..30.0,
        angle in 0.0f32..90.0,
    ) {
        let layer = square_layer(size);
        let low = InfillConfig { pattern: InfillPattern::Rectilinear, density: 0.1, angle_degrees: angle };
        let high = InfillConfig { pattern: InfillPattern::Rectilinear, density: 0.5, angle_degrees: angle };
        let low_count = generate_infill(&layer, &low, 0.4).len();
        let high_count = generate_infill(&layer, &high, 0.4).len();
        prop_assert!(high_count >= low_count,
            "Higher density ({}) should produce >= lines than lower ({}): {} vs {}",
            0.5, 0.1, high_count, low_count);
    }

    /// No infill segment should be zero-length.
    #[test]
    fn infill_no_zero_length(
        size in 5.0f32..50.0,
        density in 0.05f32..1.0,
        pattern_idx in 0u8..3,
    ) {
        let layer = square_layer(size);
        let pattern = match pattern_idx {
            0 => InfillPattern::Rectilinear,
            1 => InfillPattern::Grid,
            _ => InfillPattern::Honeycomb,
        };
        let config = InfillConfig { pattern, density, angle_degrees: 45.0 };
        let lines = generate_infill(&layer, &config, 0.4);
        for seg in &lines {
            let len = seg.length();
            prop_assert!(len > 1e-4, "Zero-length segment: len={}", len);
        }
    }
}
