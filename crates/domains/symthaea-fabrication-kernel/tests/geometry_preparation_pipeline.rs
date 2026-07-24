// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fabrication_kernel::{
    CSGNode, FabricationProcessPolicy, FabricationReadyMesh, ProcessPreparedMesh, SliceConfig,
    TessellationPolicy, ToolpathConfig, Transform3D, export_3mf_package, resolve_to_mesh,
    resolve_to_mesh_with_policy, slice_fabrication_ready, try_generate_gcode,
};

#[test]
fn adaptive_geometry_crosses_process_gate_and_exports_3mf() {
    let design = CSGNode::cylinder().with_transform(Transform3D {
        scale: [20.0, 20.0, 10.0],
        translate: [10.0, 10.0, 5.0],
        ..Default::default()
    });
    let mesh = resolve_to_mesh_with_policy(
        &design,
        TessellationPolicy {
            max_chord_error_mm: 0.05,
            min_segments: 16,
            max_segments: 256,
        },
    );
    let prepared = ProcessPreparedMesh::try_new(mesh, FabricationProcessPolicy::default())
        .expect("plate-aligned cylinder should gain process authority");
    assert!(!prepared.report().support_plan.requires_support());

    let package = export_3mf_package(prepared.mesh());
    assert_eq!(&package[0..4], &0x0403_4b50u32.to_le_bytes());

    let layers = slice_fabrication_ready(prepared.geometry(), &SliceConfig::default())
        .expect("process-prepared geometry should remain sliceable");
    let gcode = try_generate_gcode(&layers, &SliceConfig::default(), &ToolpathConfig::default())
        .expect("prepared geometry should generate a non-empty program");
    assert!(gcode.command_count() > 0);
    assert!(gcode.total_extrusion_mm > 0.0);
}

#[test]
fn overlapping_closed_shells_do_not_gain_geometry_authority() {
    let mut left = resolve_to_mesh(&CSGNode::cube());
    let right = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
        translate: [0.25, 0.25, 0.25],
        ..Default::default()
    }));
    left.merge(&right);
    let report = FabricationReadyMesh::try_new(left)
        .err()
        .expect("overlapping shells must fail the self-intersection gate");
    assert!(report.self_intersection_scan_complete);
    assert!(!report.self_intersections.is_empty());
}

#[test]
fn floating_geometry_fails_when_support_authority_is_disabled() {
    let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
        translate: [0.0, 0.0, 2.0],
        ..Default::default()
    }));
    let result = ProcessPreparedMesh::try_new(
        mesh,
        FabricationProcessPolicy {
            allow_sacrificial_supports: false,
            ..FabricationProcessPolicy::default()
        },
    );
    assert!(result.is_err());
}
