//! HMR-0 block-out geometry previews through Symthaea's fabrication kernel.
//!
//! These are coarse collision-envelope models only, not manufacturable designs.
//! Dimensions are millimetres. The generated STL files must not be fabricated
//! or used with a robot without a separately approved design and test process.
//!
//! Run from the repository root:
//!   cargo run -p symthaea-fabrication-kernel --example hmr0_tool_envelope_preview
//!   cargo run -p symthaea-fabrication-kernel --example hmr0_tool_envelope_preview -- target/hmr0-previews
//!
//! Output geometry uses the existing CSG -> mesh -> STL pipeline and is checked
//! for basic mesh validity and STL round-trip integrity.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use symthaea_fabrication_kernel::csg::CSGNode;
use symthaea_fabrication_kernel::export::export_stl;
use symthaea_fabrication_kernel::import::parse_stl;
use symthaea_fabrication_kernel::mesh::resolve_to_mesh;
use symthaea_fabrication_kernel::validate::validate_mesh;

fn cuboid(dimensions_mm: [f64; 3], center_mm: [f64; 3]) -> CSGNode {
    CSGNode::cube()
        .scale(dimensions_mm[0], dimensions_mm[1], dimensions_mm[2])
        .translate(center_mm[0], center_mm[1], center_mm[2])
}

/// Simplified collision envelope, with rigid rectangular proxies for the pads.
fn sealed_tote_gripper_envelope() -> CSGNode {
    // Mount/bridge rail; simplified as a box.
    let rail = cuboid([320.0, 40.0, 20.0], [0.0, 0.0, -50.0]);

    // Parallel jaw envelopes; the 2 mm vertical overlap with the rail is
    // deliberate for a single fused CSG solid.
    let left_jaw = cuboid([20.0, 90.0, 80.0], [-144.0, 0.0, -2.0]);
    let right_jaw = cuboid([20.0, 90.0, 80.0], [144.0, 0.0, -2.0]);

    // Pad envelopes are placeholders: no elastomer, texture, compliance or
    // friction model is implied by these rigid cuboids.
    let left_pad = cuboid([5.0, 70.0, 70.0], [-132.5, 0.0, -2.0]);
    let right_pad = cuboid([5.0, 70.0, 70.0], [132.5, 0.0, -2.0]);

    rail.union(left_jaw)
        .union(right_jaw)
        .union(left_pad)
        .union(right_pad)
}

/// Simplified passive tray docking envelope; rails and end stop are fused.
fn passive_tray_dock_envelope() -> CSGNode {
    let base = cuboid([380.0, 280.0, 8.0], [0.0, 0.0, -24.0]);
    let left_guide = cuboid([8.0, 260.0, 36.0], [-184.0, 0.0, -2.5]);
    let right_guide = cuboid([8.0, 260.0, 36.0], [184.0, 0.0, -2.5]);
    let end_stop = cuboid([380.0, 8.0, 36.0], [0.0, 126.0, -2.5]);

    base.union(left_guide)
        .union(right_guide)
        .union(end_stop)
}

fn export_validated(name: &str, model: &CSGNode, output_dir: &Path) -> Result<(), String> {
    let mesh = resolve_to_mesh(model);
    let initial_report = validate_mesh(&mesh);
    if !initial_report.is_valid() {
        return Err(format!(
            "{name}: generated mesh failed validation (triangles={}, watertight={}, signed_volume={})",
            mesh.triangle_count(),
            initial_report.is_watertight,
            initial_report.signed_volume
        ));
    }

    let stl = export_stl(&mesh);
    let parsed = parse_stl(&stl).map_err(|error| format!("{name}: STL round-trip parse failed: {error:?}"))?;
    let round_trip_report = validate_mesh(&parsed);
    if !round_trip_report.is_valid() {
        return Err(format!("{name}: exported STL failed round-trip mesh validation"));
    }

    let output_path: PathBuf = output_dir.join(format!("{name}.stl"));
    fs::write(&output_path, &stl)
        .map_err(|error| format!("{name}: unable to write {:?}: {error}", output_path))?;

    println!(
        "{name}: {} triangles, {} bytes, mesh valid, STL round-trip valid -> {}",
        parsed.triangle_count(),
        stl.len(),
        output_path.display()
    );
    Ok(())
}

fn main() {
    let output_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/hmr0-tool-envelope-previews"));
    if let Err(error) = fs::create_dir_all(&output_dir) {
        eprintln!("Unable to create preview output directory: {error}");
        std::process::exit(2);
    }

    for (name, model) in [
        ("sealed_tote_gripper_envelope", sealed_tote_gripper_envelope()),
        ("passive_tray_dock_envelope", passive_tray_dock_envelope()),
    ] {
        if let Err(error) = export_validated(name, &model, &output_dir) {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }

    println!("CONCEPT ONLY: no fabrication, robot use, patient contact or clinical function is authorized.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gripper_collision_envelope_is_a_valid_mesh() {
        let mesh = resolve_to_mesh(&sealed_tote_gripper_envelope());
        assert!(validate_mesh(&mesh).is_valid());
    }

    #[test]
    fn passive_tray_collision_envelope_is_a_valid_mesh() {
        let mesh = resolve_to_mesh(&passive_tray_dock_envelope());
        assert!(validate_mesh(&mesh).is_valid());
    }
}
