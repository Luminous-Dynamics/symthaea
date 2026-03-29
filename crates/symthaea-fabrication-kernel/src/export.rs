// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! STL and 3MF file export
//!
//! Binary STL: 80-byte header + u32 triangle count + per-triangle data (50 bytes each)
//! 3MF: XML-based model with vertices and triangles

use crate::mesh::TriangleMesh;

/// Export mesh as binary STL
pub fn export_stl(mesh: &TriangleMesh) -> Vec<u8> {
    let tri_count = mesh.indices.len() as u32;
    let mut buf = Vec::with_capacity(84 + tri_count as usize * 50);

    // 80-byte header
    let header = b"symthaea-fabrication-kernel STL output\0";
    buf.extend_from_slice(header);
    buf.extend_from_slice(&[0u8; 80 - 39]); // Pad to 80 bytes

    // Triangle count
    buf.extend_from_slice(&tri_count.to_le_bytes());

    // Per-triangle: normal (3×f32) + 3 vertices (3×f32 each) + attribute (u16)
    for tri in &mesh.indices {
        // Compute face normal from first triangle's vertices
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];

        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        let (nx, ny, nz) = if len > 1e-10 {
            (nx / len, ny / len, nz / len)
        } else {
            (0.0, 0.0, 1.0)
        };

        // Normal
        buf.extend_from_slice(&nx.to_le_bytes());
        buf.extend_from_slice(&ny.to_le_bytes());
        buf.extend_from_slice(&nz.to_le_bytes());

        // Vertices
        for &idx in tri {
            let v = mesh.vertices[idx as usize];
            buf.extend_from_slice(&v[0].to_le_bytes());
            buf.extend_from_slice(&v[1].to_le_bytes());
            buf.extend_from_slice(&v[2].to_le_bytes());
        }

        // Attribute byte count
        buf.extend_from_slice(&0u16.to_le_bytes());
    }

    buf
}

/// Export mesh as 3MF (XML model)
pub fn export_3mf(mesh: &TriangleMesh) -> String {
    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<model unit=\"millimeter\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n");
    xml.push_str("  <resources>\n");
    xml.push_str("    <object id=\"1\" type=\"model\">\n");
    xml.push_str("      <mesh>\n");

    // Vertices
    xml.push_str("        <vertices>\n");
    for v in &mesh.vertices {
        xml.push_str(&format!(
            "          <vertex x=\"{:.6}\" y=\"{:.6}\" z=\"{:.6}\" />\n",
            v[0], v[1], v[2]
        ));
    }
    xml.push_str("        </vertices>\n");

    // Triangles
    xml.push_str("        <triangles>\n");
    for tri in &mesh.indices {
        xml.push_str(&format!(
            "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\" />\n",
            tri[0], tri[1], tri[2]
        ));
    }
    xml.push_str("        </triangles>\n");

    xml.push_str("      </mesh>\n");
    xml.push_str("    </object>\n");
    xml.push_str("  </resources>\n");
    xml.push_str("  <build>\n");
    xml.push_str("    <item objectid=\"1\" />\n");
    xml.push_str("  </build>\n");
    xml.push_str("</model>\n");

    xml
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::TriangleMesh;

    fn simple_triangle() -> TriangleMesh {
        TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            indices: vec![[0, 1, 2]],
        }
    }

    #[test]
    fn test_stl_empty() {
        let mesh = TriangleMesh::empty();
        let stl = export_stl(&mesh);
        assert_eq!(stl.len(), 84); // header + count, no triangles
    }

    #[test]
    fn test_stl_size() {
        let mesh = simple_triangle();
        let stl = export_stl(&mesh);
        assert_eq!(stl.len(), 84 + 50); // 1 triangle
    }

    #[test]
    fn test_stl_triangle_count() {
        let stl = export_stl(&simple_triangle());
        let count = u32::from_le_bytes([stl[80], stl[81], stl[82], stl[83]]);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_3mf_xml() {
        let xml = export_3mf(&simple_triangle());
        assert!(xml.contains("<vertex"));
        assert!(xml.contains("<triangle"));
        assert!(xml.contains("3dmanufacturing"));
    }
}
