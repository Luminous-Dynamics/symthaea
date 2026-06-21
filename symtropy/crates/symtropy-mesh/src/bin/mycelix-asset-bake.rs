// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Mycelix Asset Baker (headless virtual geometry asset compiler).
//!
//! Ingests raw scene geometry along with physical attributes, partitions triangles into
//! optimized meshlet clusters (representing the METIS clustering phase), and compiles them into:
//! 1. A binary `.mpvg` (Multi-Physics Virtual Geometry) buffer containing physics strata strides.
//! 2. A visual-only mesh representation (mocked visual output).

use std::env;
use std::fs::File;
use std::io::{self, Write};
use symtropy_mesh::meshlet_physics::{MultiPhysicsMeshletMesh, MultiPhysicsVertex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: mycelix-asset-bake <input_mock_json> <output_mpvg_path>");
        eprintln!("\nFormat of input mock JSON:");
        eprintln!(
            "{{\n  \"vertices\": [[x,y,z], ...],\n  \"normals\": [[x,y,z], ...],\n  \"uvs\": [[u,v], ...],\n  \"indices\": [0, 1, 2, ...],\n  \"mass_density\": 7800.0,\n  \"elastic_modulus\": 200e9,\n  \"thermal_conductivity\": 50.0,\n  \"acoustic_impedance\": 46e6\n}}"
        );
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    println!(
        "Starting sandboxed virtual geometry bake for: {}",
        input_path
    );

    // Read input file or generate mock geometry if input path is a helper tag
    let (vertices, indices) = if input_path == "gen-mock-sphere" {
        println!("Generating mock spherical shell topology...");
        generate_mock_sphere()
    } else {
        println!("Parsing input JSON geometry from {}...", input_path);
        parse_json_geometry(input_path)?
    };

    println!(
        "Ingested {} vertices and {} indices. Compiling meshlet DAG...",
        vertices.len(),
        indices.len()
    );

    // Build the Multi-Physics Virtual Geometry mesh
    let mesh = MultiPhysicsMeshletMesh::build_prototype(vertices, indices);

    // Serialize to binary .mpvg buffer
    let cooked_bytes = mesh.cook_binary()?;
    let mut out_file = File::create(output_path)?;
    out_file.write_all(&cooked_bytes)?;

    println!("Bake completed successfully!");
    println!("Compiled virtual geometry details:");
    println!("- Output path: {}", output_path);
    println!("- Cooked size: {} bytes", cooked_bytes.len());
    println!("- Total meshlet clusters: {}", mesh.meshlets.len());
    if let Some(m0) = mesh.meshlets.first() {
        println!("- Meshlet [0] Total Mass: {:.2} kg", m0.total_mass);
        println!(
            "- Meshlet [0] Center of Mass: {:?}",
            m0.center_of_mass.as_slice()
        );
        println!(
            "- Meshlet [0] Average Elasticity: {:.2} GPa",
            m0.average_elasticity / 1e9
        );
    }

    Ok(())
}

/// Parses a simple JSON mesh format representing raw input from Blender / OpenUSD compilers.
fn parse_json_geometry(
    path: &str,
) -> Result<(Vec<MultiPhysicsVertex>, Vec<u8>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let mock_data: serde_json::Value = serde_json::from_reader(reader)?;

    let vertices_arr = mock_data["vertices"]
        .as_array()
        .ok_or("Missing vertices array")?;
    let normals_arr = mock_data["normals"]
        .as_array()
        .ok_or("Missing normals array")?;
    let uvs_arr = mock_data["uvs"].as_array().ok_or("Missing uvs array")?;
    let indices_arr = mock_data["indices"]
        .as_array()
        .ok_or("Missing indices array")?;

    let mass_density = mock_data["mass_density"].as_f64().unwrap_or(2700.0) as f32;
    let elastic_modulus = mock_data["elastic_modulus"].as_f64().unwrap_or(70e9) as f32;
    let thermal_conductivity = mock_data["thermal_conductivity"].as_f64().unwrap_or(200.0) as f32;
    let acoustic_impedance = mock_data["acoustic_impedance"].as_f64().unwrap_or(17e6) as f32;

    let mut vertices = Vec::new();
    for i in 0..vertices_arr.len() {
        let pos = &vertices_arr[i];
        let norm = &normals_arr[i];
        let uv = &uvs_arr[i];

        vertices.push(MultiPhysicsVertex {
            position: [
                pos[0].as_f64().unwrap() as f32,
                pos[1].as_f64().unwrap() as f32,
                pos[2].as_f64().unwrap() as f32,
            ],
            normal: [
                norm[0].as_f64().unwrap() as f32,
                norm[1].as_f64().unwrap() as f32,
                norm[2].as_f64().unwrap() as f32,
            ],
            uv: [
                uv[0].as_f64().unwrap() as f32,
                uv[1].as_f64().unwrap() as f32,
            ],
            mass_density,
            elastic_modulus,
            thermal_conductivity,
            acoustic_impedance,
        });
    }

    let indices = indices_arr
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();

    Ok((vertices, indices))
}

/// Helper to generate a mock spherical shell for standalone local tests/runs.
fn generate_mock_sphere() -> (Vec<MultiPhysicsVertex>, Vec<u8>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // 8 points forming a simple double-pyramid / octahedron (approximation of sphere)
    let raw_vertices: Vec<[f32; 3]> = vec![
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ];

    for &pos in &raw_vertices {
        let normal = {
            let len = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2])
                .sqrt()
                .max(0.001);
            [pos[0] / len, pos[1] / len, pos[2] / len]
        };

        vertices.push(MultiPhysicsVertex {
            position: pos,
            normal,
            uv: [0.0, 0.0],
            mass_density: 7800.0, // steel
            elastic_modulus: 200e9,
            thermal_conductivity: 50.0,
            acoustic_impedance: 46e6,
        });
    }

    // Indices forming 8 triangles
    indices.extend_from_slice(&[
        0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 5, 2, 1, 5, 3, 2, 5, 4, 3, 5, 1, 4,
    ]);

    (vertices, indices)
}
