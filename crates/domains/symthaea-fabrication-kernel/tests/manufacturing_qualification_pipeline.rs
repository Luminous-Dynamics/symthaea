// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fabrication_kernel::{
    CSGNode, FabricationProcessPolicy, GeometryFingerprintPolicy, MachineProfile, ManifestMismatch,
    ManufacturingReadyMesh, MeshRepairPolicy, MinimumFeaturePolicy, SliceConfig, ToolpathConfig,
    Transform3D, ValidatedGCode, build_fabrication_manifest, export_3mf_package_with_manifest,
    fingerprint_mesh_geometry, repair_mesh, resolve_to_mesh, slice_manufacturing_ready,
    try_generate_gcode, verify_fabrication_manifest,
};

#[test]
fn repaired_geometry_reaches_manifest_bound_3mf_package() {
    let mut imported = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
        translate: [10.0, 10.0, 0.5],
        ..Default::default()
    }));
    imported.indices.push(imported.indices[0]);
    imported.indices.push([0, 0, 1]);
    imported.indices.push([0, 1, u32::MAX]);

    let repaired = repair_mesh(&imported, MeshRepairPolicy::default()).unwrap();
    assert_eq!(repaired.report.removed_duplicate_triangles, 1);
    assert_eq!(repaired.report.removed_degenerate_triangles, 1);
    assert_eq!(repaired.report.removed_out_of_bounds_triangles, 1);

    let ready = ManufacturingReadyMesh::try_new(
        repaired.mesh,
        FabricationProcessPolicy::default(),
        MinimumFeaturePolicy::default(),
    )
    .unwrap();
    let slice_config = SliceConfig::default();
    let toolpath_config = ToolpathConfig::default();
    let machine = MachineProfile::default();
    let layers = slice_manufacturing_ready(&ready, &slice_config).unwrap();
    let program = try_generate_gcode(&layers, &slice_config, &toolpath_config).unwrap();
    let validated = ValidatedGCode::try_new(program, &machine).unwrap();

    let manifest = build_fabrication_manifest(
        &ready,
        &slice_config,
        &toolpath_config,
        &machine,
        &layers,
        &validated,
    )
    .unwrap();
    let verification = verify_fabrication_manifest(
        &manifest,
        &ready,
        &slice_config,
        &toolpath_config,
        &machine,
        &layers,
        &validated,
    )
    .unwrap();
    assert!(verification.matches());

    let package = export_3mf_package_with_manifest(ready.mesh(), &manifest).unwrap();
    assert!(
        package
            .windows("Metadata/fabrication-manifest.json".len())
            .any(|window| window == b"Metadata/fabrication-manifest.json")
    );
    assert!(
        package
            .windows(manifest.schema_version.len())
            .any(|window| window == manifest.schema_version.as_bytes())
    );
}

#[test]
fn manifest_detects_geometry_and_policy_drift() {
    let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
        translate: [10.0, 10.0, 0.5],
        ..Default::default()
    }));
    let original_geometry =
        fingerprint_mesh_geometry(&mesh, GeometryFingerprintPolicy::default()).unwrap();
    let ready = ManufacturingReadyMesh::try_new(
        mesh,
        FabricationProcessPolicy::default(),
        MinimumFeaturePolicy::default(),
    )
    .unwrap();
    let slice_config = SliceConfig::default();
    let toolpath_config = ToolpathConfig::default();
    let machine = MachineProfile::default();
    let layers = slice_manufacturing_ready(&ready, &slice_config).unwrap();
    let program = try_generate_gcode(&layers, &slice_config, &toolpath_config).unwrap();
    let validated = ValidatedGCode::try_new(program, &machine).unwrap();
    let manifest = build_fabrication_manifest(
        &ready,
        &slice_config,
        &toolpath_config,
        &machine,
        &layers,
        &validated,
    )
    .unwrap();
    assert_eq!(manifest.geometry, original_geometry);

    let mut changed_slice = slice_config.clone();
    changed_slice.layer_height = 0.1;
    let report = verify_fabrication_manifest(
        &manifest,
        &ready,
        &changed_slice,
        &toolpath_config,
        &machine,
        &layers,
        &validated,
    )
    .unwrap();
    assert!(report.mismatches.contains(&ManifestMismatch::SliceConfig));
    assert!(report.mismatches.contains(&ManifestMismatch::Pipeline));
}
