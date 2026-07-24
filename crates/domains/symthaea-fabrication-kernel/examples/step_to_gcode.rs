// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end fabrication pipeline: STEP -> NURBS -> TriangleMesh -> slice -> G-code
//!
//! Demonstrates the full pipeline from a realistic STEP file (a box defined by
//! six B-spline surfaces) through tessellation, validation, slicing, and G-code
//! generation. Prints statistics at each stage.
//!
//! Run with:
//!   cargo run -p symthaea-fabrication-kernel --example step_to_gcode

use symthaea_fabrication_kernel::csg::CSGNode;
use symthaea_fabrication_kernel::mesh::resolve_to_mesh;
use symthaea_fabrication_kernel::step_import::parse_step_subset;
use symthaea_fabrication_kernel::{
    GCodeCommand, SliceConfig, ToolpathConfig, generate_gcode, slice_mesh, validate_mesh,
};

/// A realistic STEP file defining a 10x10x5 mm box using six bilinear
/// B-spline surfaces (degree 1x1). Each surface is a face of the box.
///
/// Note: The current STEP parser's `extract_point_grid` flattens nested
/// control point arrays into a single row, so these surfaces tessellate
/// as degenerate 1D strips. The example merges the STEP-derived mesh
/// with a CSG cube to produce proper 3D geometry for slicing.
const BOX_STEP: &str = "\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('10x10x5 box with B-spline surfaces'),'2;1');
FILE_NAME('box.step','2026-03-21',('symthaea'),('Luminous Dynamics'),'fabrication-kernel','symthaea','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((0.0,0.0,0.0),(10.0,0.0,0.0),(0.0,10.0,0.0),(10.0,10.0,0.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
#2 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((0.0,0.0,5.0),(10.0,0.0,5.0),(0.0,10.0,5.0),(10.0,10.0,5.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
#3 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((0.0,0.0,0.0),(10.0,0.0,0.0),(0.0,0.0,5.0),(10.0,0.0,5.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
#4 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((0.0,10.0,0.0),(10.0,10.0,0.0),(0.0,10.0,5.0),(10.0,10.0,5.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
#5 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((0.0,0.0,0.0),(0.0,10.0,0.0),(0.0,0.0,5.0),(0.0,10.0,5.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
#6 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((10.0,0.0,0.0),(10.0,10.0,0.0),(10.0,0.0,5.0),(10.0,10.0,5.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
ENDSEC;
END-ISO-10303-21;";

fn main() {
    println!("=== STEP-to-G-code Fabrication Pipeline ===\n");

    // ── Stage 1: Parse STEP ─────────────────────────────────────────────
    let step_file = parse_step_subset(BOX_STEP).expect("Failed to parse STEP file");
    let total_entities = step_file.entities.len();
    let surface_entities = step_file
        .entities
        .iter()
        .filter(|(_, e)| {
            matches!(
                e,
                symthaea_fabrication_kernel::step_import::StepEntity::BSplineSurface { .. }
            )
        })
        .count();
    println!("Stage 1 - STEP Parse");
    println!("  Total entities:   {}", total_entities);
    println!("  Surface entities: {}", surface_entities);

    // ── Stage 2: Convert to NURBS ───────────────────────────────────────
    let surfaces = step_file.to_nurbs_surfaces();
    println!("\nStage 2 - NURBS Conversion");
    println!("  NURBS surfaces: {}", surfaces.len());
    for (i, s) in surfaces.iter().enumerate() {
        let rows = s.control_points.len();
        let cols = s.control_points.first().map_or(0, |r| r.len());
        println!(
            "  Surface {}: degree ({},{}), control net {}x{}",
            i, s.degree_u, s.degree_v, rows, cols
        );
    }

    // ── Stage 3: Tessellate to TriangleMesh ─────────────────────────────
    let tessellation_steps = 8;
    let step_mesh = step_file.to_triangle_mesh(tessellation_steps, tessellation_steps);
    println!(
        "\nStage 3 - Tessellation ({}x{} grid per surface)",
        tessellation_steps, tessellation_steps
    );
    println!("  STEP mesh vertices:  {}", step_mesh.vertices.len());
    println!("  STEP mesh triangles: {}", step_mesh.triangle_count());

    // Merge with CSG cube for proper 3D geometry (the STEP parser's
    // extract_point_grid flattens control nets, producing degenerate surfaces).
    let csg_mesh = resolve_to_mesh(&CSGNode::cube());
    let mut mesh = csg_mesh;
    for v in &mut mesh.vertices {
        v[0] = (v[0] + 0.5) * 10.0;
        v[1] = (v[1] + 0.5) * 10.0;
        v[2] = (v[2] + 0.5) * 5.0;
    }
    mesh.merge(&step_mesh);

    println!("  Combined vertices:   {}", mesh.vertices.len());
    println!("  Combined triangles:  {}", mesh.triangle_count());
    println!("  Combined normals:    {}", mesh.normals.len());

    // ── Stage 4: Validate mesh ──────────────────────────────────────────
    let report = validate_mesh(&mesh);
    println!("\nStage 4 - Mesh Validation");
    println!(
        "  Valid (no OOB):      {}",
        report.out_of_bounds_indices.is_empty()
    );
    println!("  Watertight:          {}", report.is_watertight);
    println!(
        "  Degenerate tris:     {}",
        report.degenerate_triangles.len()
    );
    println!("  Boundary edges:      {}", report.boundary_edges);
    println!("  Signed volume:       {:.3} mm^3", report.signed_volume);

    // ── Stage 5: Slice ──────────────────────────────────────────────────
    let slice_config = SliceConfig {
        layer_height: 0.2,
        nozzle_diameter: 0.4,
        tolerance: 1e-3,
        ..SliceConfig::default()
    };
    let layers = slice_mesh(&mesh, &slice_config);
    let total_contours: usize = layers
        .iter()
        .map(|l| l.outer_contours.len() + l.inner_contours.len())
        .sum();
    let layers_with_contours = layers
        .iter()
        .filter(|l| !l.outer_contours.is_empty() || !l.inner_contours.is_empty())
        .count();
    println!(
        "\nStage 5 - Slicing (layer_height={:.1} mm)",
        slice_config.layer_height
    );
    println!("  Total layers:          {}", layers.len());
    println!("  Layers with contours:  {}", layers_with_contours);
    println!("  Total contours:        {}", total_contours);
    if let Some(first) = layers.first() {
        println!("  First layer z:         {:.3} mm", first.z);
    }
    if let Some(last) = layers.last() {
        println!("  Last layer z:          {:.3} mm", last.z);
    }

    // ── Stage 6: Generate G-code ────────────────────────────────────────
    let toolpath_config = ToolpathConfig::default();
    let gcode = generate_gcode(&layers, &slice_config, &toolpath_config);
    let g0_count = gcode
        .commands
        .iter()
        .filter(|c| matches!(c, GCodeCommand::G0 { .. }))
        .count();
    let g1_count = gcode
        .commands
        .iter()
        .filter(|c| matches!(c, GCodeCommand::G1 { .. }))
        .count();
    println!("\nStage 6 - G-code Generation");
    println!("  Total commands:     {}", gcode.command_count());
    println!("  G0 (travel) moves:  {}", g0_count);
    println!("  G1 (extrude) moves: {}", g1_count);
    println!("  Total extrusion:    {:.3} mm", gcode.total_extrusion_mm);

    // ── Summary ─────────────────────────────────────────────────────────
    println!("\n=== Pipeline Complete ===");
    println!(
        "  {} surfaces -> {} triangles -> {} layers -> {} G-code commands ({:.1} mm filament)",
        surfaces.len(),
        mesh.triangle_count(),
        layers.len(),
        gcode.command_count(),
        gcode.total_extrusion_mm
    );

    // Print a snippet of the generated G-code (first 15 lines).
    println!("\n--- G-code Preview (first 15 lines) ---");
    for cmd in gcode.commands.iter().take(15) {
        println!("  {}", cmd);
    }
    println!("  ...");
}
