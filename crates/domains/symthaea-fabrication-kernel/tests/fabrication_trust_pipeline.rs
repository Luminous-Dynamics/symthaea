// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fabrication_kernel::printer_control::{MockPrinter, PrinterApi};
use symthaea_fabrication_kernel::{
    CSGNode, FabricationReadyMesh, GCodeCommand, GCodeProgram, MachineProfile, SliceConfig,
    ToolpathConfig, Transform3D, ValidatedGCode, compute_signed_volume, resolve_to_mesh,
    slice_fabrication_ready, submit_validated_gcode, try_generate_gcode,
    validate_gcode_for_machine,
};

fn printable_cube() -> CSGNode {
    CSGNode::cube().with_transform(Transform3D {
        scale: [20.0, 20.0, 1.0],
        rotate: [0.0, 0.0, 0.0],
        translate: [10.0, 10.0, 0.5],
    })
}

#[test]
fn staged_pipeline_reaches_mock_submission_without_raw_authority() {
    let mesh = resolve_to_mesh(&printable_cube());
    let ready = FabricationReadyMesh::try_new(mesh).expect("cube should pass mesh gate");
    let layers = slice_fabrication_ready(&ready, &SliceConfig::default())
        .expect("validated mesh should slice");
    assert!(!layers.is_empty());

    let program = try_generate_gcode(&layers, &SliceConfig::default(), &ToolpathConfig::default())
        .expect("strict toolpath inputs should generate G-code");
    let validated = ValidatedGCode::try_new(program, &MachineProfile::default())
        .expect("program should fit the default machine envelope");

    let mut printer = MockPrinter::new();
    printer.connect().unwrap();
    let job_id = submit_validated_gcode(&mut printer, &validated).unwrap();
    assert!(!job_id.is_empty());
}

#[test]
fn translation_preserves_volume_and_uniform_scale_is_cubic() {
    let base = resolve_to_mesh(&CSGNode::cube());
    let translated = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
        translate: [100.0, -50.0, 25.0],
        ..Default::default()
    }));
    let scaled = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
        scale: [2.0, 2.0, 2.0],
        ..Default::default()
    }));

    let base_volume = compute_signed_volume(&base).abs();
    let translated_volume = compute_signed_volume(&translated).abs();
    let scaled_volume = compute_signed_volume(&scaled).abs();
    assert!((translated_volume - base_volume).abs() < 1.0e-4);
    assert!((scaled_volume - base_volume * 8.0).abs() < 1.0e-4);
}

#[test]
fn machine_profile_rejects_negative_build_coordinates() {
    let program = GCodeProgram {
        commands: vec![
            GCodeCommand::G28,
            GCodeCommand::G0 {
                x: Some(-0.01),
                y: Some(1.0),
                z: Some(0.2),
                f: Some(1000.0),
            },
        ],
        total_extrusion_mm: 0.0,
    };
    let report = validate_gcode_for_machine(&program, &MachineProfile::default());
    assert!(!report.is_safe_to_submit());
    assert!(ValidatedGCode::try_new(program, &MachineProfile::default()).is_err());
}
