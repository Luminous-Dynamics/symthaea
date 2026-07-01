// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmark: slice a complex mesh at fine layer height.

use criterion::{Criterion, criterion_group, criterion_main};
use symthaea_fabrication_kernel::{
    csg::{BooleanOp, CSGNode, Primitive, Transform3D},
    infill::{InfillConfig, InfillPattern},
    mesh::resolve_to_mesh,
    slicer::{SliceConfig, slice_mesh},
    toolpath::{ToolpathConfig, generate_gcode},
};

fn complex_design() -> CSGNode {
    let cube = CSGNode::Transform {
        node: Box::new(CSGNode::Primitive(Primitive::Cube)),
        transform: Transform3D {
            scale: [20.0, 20.0, 10.0],
            ..Transform3D::default()
        },
    };
    let hole = CSGNode::Transform {
        node: Box::new(CSGNode::Primitive(Primitive::Cylinder)),
        transform: Transform3D {
            scale: [6.0, 6.0, 12.0],
            ..Transform3D::default()
        },
    };
    CSGNode::Boolean {
        op: BooleanOp::Subtract,
        left: Box::new(cube),
        right: Box::new(hole),
    }
}

fn bench_slice_mesh(c: &mut Criterion) {
    let mesh = resolve_to_mesh(&complex_design());
    let config = SliceConfig {
        layer_height: 0.05,
        nozzle_diameter: 0.4,
        tolerance: 1e-3,
        infill: Some(InfillConfig {
            pattern: InfillPattern::Rectilinear,
            density: 0.2,
            angle_degrees: 45.0,
        }),
    };
    c.bench_function("slice_mesh_0.05mm", |b| {
        b.iter(|| slice_mesh(&mesh, &config))
    });
}

fn bench_gcode_generation(c: &mut Criterion) {
    let mesh = resolve_to_mesh(&complex_design());
    let slice_config = SliceConfig {
        layer_height: 0.1,
        nozzle_diameter: 0.4,
        tolerance: 1e-3,
        infill: Some(InfillConfig {
            pattern: InfillPattern::Rectilinear,
            density: 0.2,
            angle_degrees: 45.0,
        }),
    };
    let layers = slice_mesh(&mesh, &slice_config);
    let toolpath_config = ToolpathConfig::default();
    c.bench_function("gcode_generation", |b| {
        b.iter(|| generate_gcode(&layers, &slice_config, &toolpath_config))
    });
}

criterion_group!(benches, bench_slice_mesh, bench_gcode_generation);
criterion_main!(benches);
