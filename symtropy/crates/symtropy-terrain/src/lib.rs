// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Terrain and heightmap-to-mesh conversion for Symtropy.

use nalgebra::SVector;
use symtropy_mesh::{Triangle, TriangleMesh};

/// A 2D heightmap for terrain generation.
pub struct HeightMap {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
    pub scale: SVector<f64, 3>,
}

impl HeightMap {
    /// Create a new heightmap from raw data.
    pub fn new(data: Vec<f64>, width: usize, height: usize, scale: SVector<f64, 3>) -> Self {
        Self {
            data,
            width,
            height,
            scale,
        }
    }

    /// Convert the heightmap into a triangle mesh collider.
    pub fn to_triangle_mesh(&self) -> TriangleMesh {
        let mut triangles = Vec::new();

        for y in 0..(self.height - 1) {
            for x in 0..(self.width - 1) {
                // Get 4 corners of the quad
                let v00 = self.get_vertex(x, y);
                let v10 = self.get_vertex(x + 1, y);
                let v01 = self.get_vertex(x, y + 1);
                let v11 = self.get_vertex(x + 1, y + 1);

                // Two triangles per quad
                triangles.push(Triangle {
                    vertices: [v00, v10, v01],
                });
                triangles.push(Triangle {
                    vertices: [v10, v11, v01],
                });
            }
        }

        TriangleMesh::new(triangles)
    }

    fn get_vertex(&self, x: usize, y: usize) -> SVector<f64, 3> {
        let h = self.data[y * self.width + x];
        SVector::from([
            (x as f64) * self.scale[0],
            h * self.scale[1],
            (y as f64) * self.scale[2],
        ])
    }
}
