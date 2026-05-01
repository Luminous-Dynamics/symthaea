// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! 4D geometry demo: demonstrates N-dimensional math.
//!
//! Creates a tesseract (4D hypercube), rotates it in the XW plane,
//! and shows the 3D cross-section at different W positions.
//!
//! Run: cargo run -p symtropy-math --example four_d_demo

use symtropy_math::{Bivector, ConvexHull, Hyperplane, Point, Rotor};

fn main() {
    println!("=== 4D Geometry Demo ===\n");

    // Create a tesseract (4D unit cube)
    let tesseract = ConvexHull::<4>::unit_cube();
    println!("Tesseract: {} vertices", tesseract.num_vertices());
    println!("(A tesseract has 16 vertices, 32 edges, 24 faces, 8 cells)\n");

    // Show all vertices
    println!("Vertices:");
    for (i, v) in tesseract.vertices.iter().enumerate() {
        println!(
            "  V{:02}: ({:+.0}, {:+.0}, {:+.0}, {:+.0})",
            i, v[0], v[1], v[2], v[3]
        );
    }

    // Rotate in the XW plane (a rotation that only exists in 4D!)
    println!("\n--- Rotating in the XW plane by 45° ---");
    let plane = Bivector::<4>::unit_plane(0, 3); // X-W plane
    let rotation = Rotor::from_plane_angle(&plane, std::f64::consts::FRAC_PI_4);

    println!("Rotated vertices:");
    for (i, v) in tesseract.vertices.iter().enumerate() {
        let p = Point::new([v[0], v[1], v[2], v[3]]);
        let rotated = rotation.rotate_point(&p);
        println!(
            "  V{:02}: ({:+.3}, {:+.3}, {:+.3}, {:+.3})",
            i,
            rotated.coord(0),
            rotated.coord(1),
            rotated.coord(2),
            rotated.coord(3)
        );
    }

    // Cross-section slicing: slice the tesseract at different W values
    println!("\n--- 3D Cross-Sections (slicing at different W) ---");
    println!("(Like shining a flashlight through a 4D object)\n");

    for w_slice in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        let plane = Hyperplane::<4>::new(nalgebra::SVector::from([0.0, 0.0, 0.0, 1.0]), w_slice);

        // Count how many vertices are on each side
        let mut front = 0;
        let mut back = 0;
        let mut on = 0;
        for v in &tesseract.vertices {
            let p = Point::new([v[0], v[1], v[2], v[3]]);
            match plane.classify(&p) {
                symtropy_math::hyperplane::Side::Front => front += 1,
                symtropy_math::hyperplane::Side::Back => back += 1,
                symtropy_math::hyperplane::Side::On => on += 1,
            }
        }

        // The cross-section at w=0 of a unit tesseract [-1,1]^4 is a cube [-1,1]^3
        let section_type = match w_slice as i32 {
            -1 | 1 => "face (cube)",
            0 => "center slice (full cube)",
            _ => "intermediate",
        };

        println!("  W={w_slice:+.1}: {front} front, {back} back, {on} on plane -> {section_type}",);
    }

    // Demonstrate 4D rotation preserves distances
    println!("\n--- Distance Preservation ---");
    let p1 = Point::<4>::new([1.0, 0.0, 0.0, 0.0]);
    let p2 = Point::<4>::new([0.0, 1.0, 0.0, 0.0]);
    let d_before = p1.distance(&p2);

    let r1 = rotation.rotate_point(&p1);
    let r2 = rotation.rotate_point(&p2);
    let d_after = r1.distance(&r2);

    println!("  Distance before rotation: {:.6}", d_before);
    println!("  Distance after rotation:  {:.6}", d_after);
    println!("  Preserved: {}", (d_before - d_after).abs() < 1e-10);

    // Double rotation (unique to 4D: rotate in two planes simultaneously)
    println!("\n--- Double Rotation (4D only) ---");
    let mut double_plane = Bivector::<4>::zero();
    double_plane.set(0, 1, 1.0); // XY plane
    double_plane.set(2, 3, 1.0); // ZW plane (simultaneous!)
    let double_rot = Rotor::from_plane_angle(&double_plane, std::f64::consts::FRAC_PI_4);

    let p = Point::new([1.0, 0.0, 1.0, 0.0]);
    let rotated = double_rot.rotate_point(&p);
    println!(
        "  Original: ({:.3}, {:.3}, {:.3}, {:.3})",
        p.coord(0),
        p.coord(1),
        p.coord(2),
        p.coord(3)
    );
    println!(
        "  Rotated:  ({:.3}, {:.3}, {:.3}, {:.3})",
        rotated.coord(0),
        rotated.coord(1),
        rotated.coord(2),
        rotated.coord(3)
    );
    println!("  (Both XY and ZW planes rotated simultaneously!)");
    println!(
        "  Norm preserved: {}",
        (p.0.norm() - rotated.0.norm()).abs() < 1e-10
    );
    println!("  (Norm: {:.6} → {:.6})", p.0.norm(), rotated.0.norm());
}
