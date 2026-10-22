//! HMR-0 block-out geometry previews from Symthaea's versioned tool concept catalog.
//!
//! These are coarse collision-envelope models only, not manufacturable designs.
//! Dimensions are millimetres. The generated STL files must not be fabricated
//! or used with a robot without a separately approved design and test process.
//!
//! Run from the repository root:
//!   cargo run -p symthaea-fabrication-kernel --example hmr0_tool_envelope_preview
//!   cargo run -p symthaea-fabrication-kernel --example hmr0_tool_envelope_preview -- target/hmr0-previews
//!
//! Geometry parameters are read from docs/engineering/hmr0-tool-cad-concepts-v0.1.json.
//! Before export, every nominal parameter is checked against its declared range;
//! each STL is then checked for mesh validity and round-trip parse integrity.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use symthaea_fabrication_kernel::csg::CSGNode;
use symthaea_fabrication_kernel::export::export_stl;
use symthaea_fabrication_kernel::import::parse_stl;
use symthaea_fabrication_kernel::mesh::resolve_to_mesh;
use symthaea_fabrication_kernel::validate::validate_mesh;

const CONCEPT_CATALOG: &[u8] =
    include_bytes!("../../../../docs/engineering/hmr0-tool-cad-concepts-v0.1.json");

fn find_concept<'a>(catalog: &'a Value, tool_id: &str) -> Result<&'a Value, String> {
    catalog
        .get("concepts")
        .and_then(Value::as_array)
        .and_then(|concepts| {
            concepts
                .iter()
                .find(|concept| concept.get("tool_id").and_then(Value::as_str) == Some(tool_id))
        })
        .ok_or_else(|| format!("tool concept '{tool_id}' was not found in the versioned catalog"))
}

fn nominal_parameter(catalog: &Value, tool_id: &str, name: &str) -> Result<f64, String> {
    let concept = find_concept(catalog, tool_id)?;
    let parameter = concept
        .get("geometry")
        .and_then(|geometry| geometry.get("parameters"))
        .and_then(Value::as_array)
        .and_then(|parameters| {
            parameters
                .iter()
                .find(|parameter| parameter.get("name").and_then(Value::as_str) == Some(name))
        })
        .ok_or_else(|| format!("{tool_id}: required geometry parameter '{name}' is missing"))?;

    let range = parameter
        .get("candidate_range")
        .and_then(Value::as_array)
        .filter(|range| range.len() == 2)
        .ok_or_else(|| format!("{tool_id}.{name}: candidate_range must have two numbers"))?;
    let minimum = range[0]
        .as_f64()
        .ok_or_else(|| format!("{tool_id}.{name}: range minimum is not numeric"))?;
    let maximum = range[1]
        .as_f64()
        .ok_or_else(|| format!("{tool_id}.{name}: range maximum is not numeric"))?;
    let nominal = parameter
        .get("nominal_candidate")
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{tool_id}.{name}: nominal_candidate is not numeric"))?;

    if !minimum.is_finite()
        || !maximum.is_finite()
        || !nominal.is_finite()
        || minimum > maximum
        || nominal < minimum
        || nominal > maximum
    {
        return Err(format!(
            "{tool_id}.{name}: nominal {nominal} is outside the declared finite range [{minimum}, {maximum}]"
        ));
    }
    Ok(nominal)
}

fn cuboid(dimensions_mm: [f64; 3], center_mm: [f64; 3]) -> CSGNode {
    CSGNode::cube()
        .scale(dimensions_mm[0], dimensions_mm[1], dimensions_mm[2])
        .translate(center_mm[0], center_mm[1], center_mm[2])
}

/// Simplified collision envelope driven by the named parameters in the catalog.
/// Pads are rigid rectangular proxies: no compliance, friction or material model
/// is implied by this geometry.
fn sealed_tote_gripper_envelope(catalog: &Value) -> Result<CSGNode, String> {
    const ID: &str = "SEALED_TOTE_GRIPPER";
    let opening = nominal_parameter(catalog, ID, "jaw_opening_max")?;
    let pad_width = nominal_parameter(catalog, ID, "pad_width")?;
    let pad_height = nominal_parameter(catalog, ID, "pad_contact_height")?;
    let finger_length = nominal_parameter(catalog, ID, "finger_length")?;
    let jaw_thickness = nominal_parameter(catalog, ID, "jaw_thickness")?;
    let pad_thickness = nominal_parameter(catalog, ID, "pad_thickness")?;
    let bridge_depth = nominal_parameter(catalog, ID, "bridge_depth")?;
    let bridge_thickness = nominal_parameter(catalog, ID, "bridge_thickness")?;
    let bridge_margin = nominal_parameter(catalog, ID, "bridge_end_margin")?;
    let vertical_overlap = nominal_parameter(catalog, ID, "bridge_to_jaw_overlap")?;
    let jaw_center_z = nominal_parameter(catalog, ID, "jaw_center_z")?;

    let jaw_height = pad_height
        + nominal_parameter(catalog, ID, "jaw_height_margin")?;
    let pad_center_x = opening / 2.0 + pad_thickness / 2.0;
    let jaw_pad_overlap = nominal_parameter(catalog, ID, "jaw_pad_overlap")?;
    let jaw_center_x =
        opening / 2.0 + pad_thickness + jaw_thickness / 2.0 - jaw_pad_overlap;
    let bridge_length = 2.0 * (jaw_center_x + jaw_thickness / 2.0 + bridge_margin);
    let jaw_bottom = jaw_center_z - jaw_height / 2.0;
    let bridge_center_z = jaw_bottom + vertical_overlap - bridge_thickness / 2.0;

    let rail = cuboid(
        [bridge_length, bridge_depth, bridge_thickness],
        [0.0, 0.0, bridge_center_z],
    );
    let left_jaw = cuboid(
        [jaw_thickness, finger_length, jaw_height],
        [-jaw_center_x, 0.0, jaw_center_z],
    );
    let right_jaw = cuboid(
        [jaw_thickness, finger_length, jaw_height],
        [jaw_center_x, 0.0, jaw_center_z],
    );
    let left_pad = cuboid(
        [pad_thickness, pad_width, pad_height],
        [-pad_center_x, 0.0, jaw_center_z],
    );
    let right_pad = cuboid(
        [pad_thickness, pad_width, pad_height],
        [pad_center_x, 0.0, jaw_center_z],
    );

    Ok(rail
        .union(left_jaw)
        .union(right_jaw)
        .union(left_pad)
        .union(right_pad))
}

/// Simplified passive tray docking envelope driven by catalog nominal parameters.
fn passive_tray_dock_envelope(catalog: &Value) -> Result<CSGNode, String> {
    const ID: &str = "PASSIVE_TRAY_DOCK";
    let tray_width = nominal_parameter(catalog, ID, "tray_width")?;
    let tray_depth = nominal_parameter(catalog, ID, "tray_depth")?;
    let guide_thickness = nominal_parameter(catalog, ID, "guide_thickness")?;
    let guide_height = nominal_parameter(catalog, ID, "guide_height")?;
    let base_thickness = nominal_parameter(catalog, ID, "base_thickness")?;
    let edge_margin = nominal_parameter(catalog, ID, "base_edge_margin")?;
    let front_margin = nominal_parameter(catalog, ID, "base_front_margin")?;
    let rear_margin = nominal_parameter(catalog, ID, "base_rear_margin")?;
    let guide_overlap = nominal_parameter(catalog, ID, "guide_base_overlap")?;
    let stop_thickness = nominal_parameter(catalog, ID, "end_stop_thickness")?;

    let base_width = tray_width + 2.0 * (guide_thickness + edge_margin);
    let base_depth = tray_depth + front_margin + rear_margin;
    let base = cuboid([base_width, base_depth, base_thickness], [0.0, 0.0, 0.0]);
    let guide_z = base_thickness / 2.0 + guide_height / 2.0 - guide_overlap;
    let guide_x = tray_width / 2.0 + guide_thickness / 2.0;
    let left_guide = cuboid(
        [guide_thickness, tray_depth, guide_height],
        [-guide_x, 0.0, guide_z],
    );
    let right_guide = cuboid(
        [guide_thickness, tray_depth, guide_height],
        [guide_x, 0.0, guide_z],
    );
    let end_stop = cuboid(
        [base_width, stop_thickness, guide_height],
        [0.0, tray_depth / 2.0 - stop_thickness / 2.0, guide_z],
    );

    Ok(base
        .union(left_guide)
        .union(right_guide)
        .union(end_stop))
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
    let parsed =
        parse_stl(&stl).map_err(|error| format!("{name}: STL round-trip parse failed: {error:?}"))?;
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

    let catalog: Value = match serde_json::from_slice(CONCEPT_CATALOG) {
        Ok(catalog) => catalog,
        Err(error) => {
            eprintln!("HMR-0 parameter catalog is invalid JSON: {error}");
            std::process::exit(2);
        }
    };
    let gripper = match sealed_tote_gripper_envelope(&catalog) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("HMR-0 gripper parameter binding failed: {error}");
            std::process::exit(2);
        }
    };
    let dock = match passive_tray_dock_envelope(&catalog) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("HMR-0 tray dock parameter binding failed: {error}");
            std::process::exit(2);
        }
    };

    for (name, model) in [
        ("sealed_tote_gripper_envelope", gripper),
        ("passive_tray_dock_envelope", dock),
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

    fn catalog() -> Value {
        serde_json::from_slice(CONCEPT_CATALOG).expect("HMR-0 concept catalog parses")
    }

    #[test]
    fn gripper_collision_envelope_is_generated_from_catalog_parameters() {
        let model = sealed_tote_gripper_envelope(&catalog()).expect("gripper parameters in range");
        assert!(validate_mesh(&resolve_to_mesh(&model)).is_valid());
    }

    #[test]
    fn passive_tray_collision_envelope_is_generated_from_catalog_parameters() {
        let model = passive_tray_dock_envelope(&catalog()).expect("dock parameters in range");
        assert!(validate_mesh(&resolve_to_mesh(&model)).is_valid());
    }

    fn set_nominal(catalog: &mut Value, tool_id: &str, name: &str, value: f64) {
        let concept = catalog["concepts"]
            .as_array_mut()
            .expect("concept list")
            .iter_mut()
            .find(|concept| concept["tool_id"] == tool_id)
            .expect("concept exists");
        let parameters = concept["geometry"]["parameters"]
            .as_array_mut()
            .expect("parameter list");
        let parameter = parameters
            .iter_mut()
            .find(|parameter| parameter["name"] == name)
            .expect("parameter exists");
        parameter["nominal_candidate"] = serde_json::json!(value);
    }

    #[test]
    fn gripper_geometry_remains_valid_across_opening_and_pad_width_sweep() {
        for opening in [220.0, 260.0, 320.0] {
            for pad_width in [45.0, 60.0, 75.0] {
                let mut source = catalog();
                set_nominal(&mut source, "SEALED_TOTE_GRIPPER", "jaw_opening_max", opening);
                set_nominal(&mut source, "SEALED_TOTE_GRIPPER", "pad_width", pad_width);
                let model = sealed_tote_gripper_envelope(&source)
                    .expect("all sweep values remain inside catalog ranges");
                let mesh = resolve_to_mesh(&model);
                let report = validate_mesh(&mesh);
                assert!(
                    report.is_valid(),
                    "gripper opening={opening} pad_width={pad_width}: {:?}",
                    report
                );
            }
        }
    }

    #[test]
    fn tray_dock_geometry_remains_valid_across_width_and_depth_sweep() {
        for tray_width in [250.0, 360.0, 500.0] {
            for tray_depth in [180.0, 260.0, 360.0] {
                let mut source = catalog();
                set_nominal(&mut source, "PASSIVE_TRAY_DOCK", "tray_width", tray_width);
                set_nominal(&mut source, "PASSIVE_TRAY_DOCK", "tray_depth", tray_depth);
                let model = passive_tray_dock_envelope(&source)
                    .expect("all sweep values remain inside catalog ranges");
                let mesh = resolve_to_mesh(&model);
                let report = validate_mesh(&mesh);
                assert!(
                    report.is_valid(),
                    "tray width={tray_width} depth={tray_depth}: {:?}",
                    report
                );
            }
        }
    }

    #[test]
    fn out_of_range_nominal_parameter_fails_closed_before_geometry_generation() {
        let mut source = catalog();
        let parameters = source["concepts"][0]["geometry"]["parameters"]
            .as_array_mut()
            .expect("gripper parameter list");
        let jaw = parameters
            .iter_mut()
            .find(|parameter| parameter["name"] == "jaw_opening_max")
            .expect("jaw_opening_max parameter");
        jaw["candidate_range"] = serde_json::json!([250.0, 255.0]);
        assert!(sealed_tote_gripper_envelope(&source).is_err());
    }
}
