// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! fab_resolve — CLI for resolving CSG trees to STL meshes
//!
//! Usage:
//!   fab_resolve <input.json> <output.stl>
//!   fab_resolve --primitive cube --scale 0.05,0.02,0.01 -o bracket.stl
//!   fab_resolve --validate input.stl
//!
//! Input JSON format:
//! {
//!   "Primitive": "Cube"
//! }
//! or:
//! {
//!   "Transform": {
//!     "node": { "Primitive": "Cube" },
//!     "transform": { "scale": [0.05, 0.02, 0.01], "rotate": [0,0,0], "translate": [0,0,0] }
//!   }
//! }

use std::env;
use std::fs;
use std::process;

use symthaea_fabrication_kernel::csg::{CSGNode, Primitive, Transform3D};
use symthaea_fabrication_kernel::export::export_stl;
use symthaea_fabrication_kernel::import::parse_stl;
use symthaea_fabrication_kernel::mesh::resolve_to_mesh;
use symthaea_fabrication_kernel::validate::validate_mesh;

fn print_usage() {
    eprintln!("fab_resolve — CSG-to-STL mesh resolution service");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  fab_resolve <input.json> <output.stl>    Resolve CSG tree JSON to STL");
    eprintln!("  fab_resolve --primitive <name> -o <out>   Generate primitive STL");
    eprintln!("  fab_resolve --validate <input.stl>        Validate an STL file");
    eprintln!();
    eprintln!("Primitives: cube, cylinder, sphere, cone, torus");
    eprintln!("Options:");
    eprintln!("  --scale x,y,z      Scale factors");
    eprintln!("  --translate x,y,z  Translation offset");
    eprintln!("  --rotate x,y,z     Rotation in degrees");
}

fn parse_vec3(s: &str) -> Result<[f32; 3], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err(format!("Expected x,y,z, got '{}'", s));
    }
    let x: f32 = parts[0].parse().map_err(|e| format!("Bad x: {}", e))?;
    let y: f32 = parts[1].parse().map_err(|e| format!("Bad y: {}", e))?;
    let z: f32 = parts[2].parse().map_err(|e| format!("Bad z: {}", e))?;
    Ok([x, y, z])
}

fn parse_primitive(name: &str) -> Result<Primitive, String> {
    match name.to_lowercase().as_str() {
        "cube" => Ok(Primitive::Cube),
        "cylinder" => Ok(Primitive::Cylinder),
        "sphere" => Ok(Primitive::Sphere),
        "cone" => Ok(Primitive::Cone),
        "torus" => Ok(Primitive::Torus),
        _ => Err(format!(
            "Unknown primitive: '{}'. Use: cube, cylinder, sphere, cone, torus",
            name
        )),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    // Mode: --validate
    if args[1] == "--validate" {
        if args.len() < 3 {
            eprintln!("Error: --validate requires a file path");
            process::exit(1);
        }
        let data = fs::read(&args[2]).unwrap_or_else(|e| {
            eprintln!("Error reading '{}': {}", args[2], e);
            process::exit(1);
        });
        let mesh = parse_stl(&data).unwrap_or_else(|e| {
            eprintln!("Error parsing STL: {:?}", e);
            process::exit(1);
        });
        let report = validate_mesh(&mesh);
        println!("Validation Report:");
        println!("  Triangles: {}", mesh.triangle_count());
        println!("  Vertices: {}", mesh.vertices.len());
        println!("  Watertight: {}", report.is_watertight);
        println!("  Boundary edges: {}", report.boundary_edges);
        println!(
            "  Degenerate triangles: {}",
            report.degenerate_triangles.len()
        );
        println!(
            "  Inconsistent normals: {}",
            report.inconsistent_normals.len()
        );
        println!(
            "  Out-of-bounds indices: {}",
            report.out_of_bounds_indices.len()
        );
        println!("  Signed volume: {:.6}", report.signed_volume);
        println!("  Valid: {}", report.is_valid());
        println!("  Printable: {}", report.is_printable());
        if !report.is_valid() {
            process::exit(2);
        }
        return;
    }

    // Mode: --primitive
    if args[1] == "--primitive" {
        if args.len() < 3 {
            eprintln!("Error: --primitive requires a primitive name");
            process::exit(1);
        }
        let prim = parse_primitive(&args[2]).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            process::exit(1);
        });

        let mut scale = [1.0f32; 3];
        let mut rotate = [0.0f32; 3];
        let mut translate = [0.0f32; 3];
        let mut output_path = String::from("output.stl");

        let mut i = 3;
        while i < args.len() {
            match args[i].as_str() {
                "--scale" => {
                    i += 1;
                    scale = parse_vec3(&args[i]).unwrap_or_else(|e| {
                        eprintln!("{}", e);
                        process::exit(1);
                    });
                }
                "--rotate" => {
                    i += 1;
                    rotate = parse_vec3(&args[i]).unwrap_or_else(|e| {
                        eprintln!("{}", e);
                        process::exit(1);
                    });
                }
                "--translate" => {
                    i += 1;
                    translate = parse_vec3(&args[i]).unwrap_or_else(|e| {
                        eprintln!("{}", e);
                        process::exit(1);
                    });
                }
                "-o" | "--output" => {
                    i += 1;
                    output_path = args[i].clone();
                }
                _ => {
                    eprintln!("Unknown option: {}", args[i]);
                    process::exit(1);
                }
            }
            i += 1;
        }

        let node = CSGNode::Transform {
            node: Box::new(CSGNode::Primitive(prim)),
            transform: Transform3D {
                scale,
                rotate,
                translate,
            },
        };
        let mesh = resolve_to_mesh(&node);
        let stl_data = export_stl(&mesh);
        fs::write(&output_path, &stl_data).unwrap_or_else(|e| {
            eprintln!("Error writing '{}': {}", output_path, e);
            process::exit(1);
        });
        let report = validate_mesh(&mesh);
        println!(
            "Generated: {} ({} triangles, {} bytes, valid: {})",
            output_path,
            mesh.triangle_count(),
            stl_data.len(),
            report.is_valid()
        );
        return;
    }

    // Mode: JSON CSG → STL
    if args.len() < 3 {
        eprintln!("Error: expected <input.json> <output.stl>");
        print_usage();
        process::exit(1);
    }

    let json_str = fs::read_to_string(&args[1]).unwrap_or_else(|e| {
        eprintln!("Error reading '{}': {}", args[1], e);
        process::exit(1);
    });

    let node: CSGNode = serde_json::from_str(&json_str).unwrap_or_else(|e| {
        eprintln!("Error parsing CSG JSON: {}", e);
        process::exit(1);
    });

    let mesh = resolve_to_mesh(&node);
    let stl_data = export_stl(&mesh);
    fs::write(&args[2], &stl_data).unwrap_or_else(|e| {
        eprintln!("Error writing '{}': {}", args[2], e);
        process::exit(1);
    });

    let report = validate_mesh(&mesh);
    println!(
        "Resolved CSG → {} ({} triangles, {} bytes)",
        args[2],
        mesh.triangle_count(),
        stl_data.len()
    );
    println!(
        "  Valid: {}, Printable: {}",
        report.is_valid(),
        report.is_printable()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vec3_valid() {
        assert_eq!(parse_vec3("1.0,2.0,3.0").unwrap(), [1.0, 2.0, 3.0]);
        assert_eq!(parse_vec3("0,0,0").unwrap(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_parse_vec3_invalid() {
        assert!(parse_vec3("1,2").is_err());
        assert!(parse_vec3("abc").is_err());
    }

    #[test]
    fn test_parse_primitive_valid() {
        assert_eq!(parse_primitive("cube").unwrap(), Primitive::Cube);
        assert_eq!(parse_primitive("Sphere").unwrap(), Primitive::Sphere);
        assert_eq!(parse_primitive("CYLINDER").unwrap(), Primitive::Cylinder);
    }

    #[test]
    fn test_parse_primitive_invalid() {
        assert!(parse_primitive("pyramid").is_err());
    }
}
