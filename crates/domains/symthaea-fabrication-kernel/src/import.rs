// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! STL import: parse binary and ASCII STL files into TriangleMesh

use crate::mesh::TriangleMesh;

/// Bounded STL parser profile for untrusted uploads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StlParseLimits {
    pub max_input_bytes: usize,
    pub max_triangles: usize,
    pub max_ascii_lines: usize,
    pub max_ascii_line_bytes: usize,
}

impl Default for StlParseLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 512 * 1024 * 1024,
            max_triangles: 5_000_000,
            max_ascii_lines: 25_000_000,
            max_ascii_line_bytes: 16 * 1024,
        }
    }
}

/// Parse a binary STL file into a TriangleMesh.
///
/// Binary STL format:
/// - 80 bytes: header (ignored)
/// - 4 bytes: u32 LE triangle count
/// - Per triangle (50 bytes):
///   - 12 bytes: normal (3×f32 LE)
///   - 36 bytes: 3 vertices (9×f32 LE)
///   - 2 bytes: attribute byte count (ignored)
pub fn parse_binary_stl(data: &[u8]) -> Result<TriangleMesh, StlError> {
    parse_binary_stl_with_limits(data, StlParseLimits::default())
}

/// Parse binary STL with explicit allocation and work limits.
pub fn parse_binary_stl_with_limits(
    data: &[u8],
    limits: StlParseLimits,
) -> Result<TriangleMesh, StlError> {
    if data.len() > limits.max_input_bytes {
        return Err(StlError::ResourceLimit {
            resource: "input bytes",
            limit: limits.max_input_bytes,
        });
    }
    if data.len() < 84 {
        return Err(StlError::TooShort(data.len()));
    }

    let tri_count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    if tri_count > limits.max_triangles {
        return Err(StlError::ResourceLimit {
            resource: "triangles",
            limit: limits.max_triangles,
        });
    }
    let triangle_bytes = tri_count.checked_mul(50).ok_or(StlError::ResourceLimit {
        resource: "triangle bytes",
        limit: limits.max_input_bytes,
    })?;
    let expected_len = 84usize
        .checked_add(triangle_bytes)
        .ok_or(StlError::ResourceLimit {
            resource: "input bytes",
            limit: limits.max_input_bytes,
        })?;
    if data.len() < expected_len {
        return Err(StlError::TruncatedData {
            expected: expected_len,
            actual: data.len(),
        });
    }

    let mut vertices = Vec::with_capacity(tri_count * 3);
    let mut normals = Vec::with_capacity(tri_count * 3);
    let mut indices = Vec::with_capacity(tri_count);

    for i in 0..tri_count {
        let base = 84 + i * 50;
        // Skip stored normal — recompute from vertices for consistency
        let v0 = read_vertex(data, base + 12);
        let v1 = read_vertex(data, base + 24);
        let v2 = read_vertex(data, base + 36);

        // Validate vertices
        for v in [v0, v1, v2] {
            for c in v {
                if !c.is_finite() {
                    return Err(StlError::NonFiniteCoordinate(i));
                }
            }
        }

        // Compute face normal from cross product
        let normal = compute_normal(v0, v1, v2);

        let idx = vertices.len() as u32;
        vertices.push(v0);
        vertices.push(v1);
        vertices.push(v2);
        normals.push(normal);
        normals.push(normal);
        normals.push(normal);
        indices.push([idx, idx + 1, idx + 2]);
    }

    Ok(TriangleMesh {
        vertices,
        normals,
        indices,
    })
}

/// Detect whether data looks like ASCII STL (starts with "solid")
pub fn is_ascii_stl(data: &[u8]) -> bool {
    data.len() > 6
        && data[..6].eq_ignore_ascii_case(b"solid ")
        && !data[6..data.len().min(84)].contains(&0)
}

/// Parse ASCII STL text into a TriangleMesh
pub fn parse_ascii_stl(text: &str) -> Result<TriangleMesh, StlError> {
    parse_ascii_stl_with_limits(text, StlParseLimits::default())
}

/// Parse ASCII STL with explicit allocation and work limits.
pub fn parse_ascii_stl_with_limits(
    text: &str,
    limits: StlParseLimits,
) -> Result<TriangleMesh, StlError> {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    if text.len() > limits.max_input_bytes {
        return Err(StlError::ResourceLimit {
            resource: "input bytes",
            limit: limits.max_input_bytes,
        });
    }
    let mut tri_verts: Vec<[f32; 3]> = Vec::new();

    for (line_index, line) in text.lines().enumerate() {
        if line_index >= limits.max_ascii_lines {
            return Err(StlError::ResourceLimit {
                resource: "ASCII lines",
                limit: limits.max_ascii_lines,
            });
        }
        if line.len() > limits.max_ascii_line_bytes {
            return Err(StlError::ResourceLimit {
                resource: "ASCII line bytes",
                limit: limits.max_ascii_line_bytes,
            });
        }
        let trimmed = line.trim();
        if trimmed.starts_with("vertex") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(StlError::ParseError("bad vertex line".into()));
            }
            let x: f32 = parts[1]
                .parse()
                .map_err(|_| StlError::ParseError("bad x".into()))?;
            let y: f32 = parts[2]
                .parse()
                .map_err(|_| StlError::ParseError("bad y".into()))?;
            let z: f32 = parts[3]
                .parse()
                .map_err(|_| StlError::ParseError("bad z".into()))?;
            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                return Err(StlError::NonFiniteCoordinate(indices.len()));
            }
            tri_verts.push([x, y, z]);

            if tri_verts.len() == 3 {
                if indices.len() >= limits.max_triangles {
                    return Err(StlError::ResourceLimit {
                        resource: "triangles",
                        limit: limits.max_triangles,
                    });
                }
                let normal = compute_normal(tri_verts[0], tri_verts[1], tri_verts[2]);
                let idx = vertices.len() as u32;
                vertices.extend_from_slice(&tri_verts);
                normals.extend_from_slice(&[normal; 3]);
                indices.push([idx, idx + 1, idx + 2]);
                tri_verts.clear();
            }
        }
    }

    if vertices.is_empty() {
        return Err(StlError::ParseError("no triangles found".into()));
    }

    Ok(TriangleMesh {
        vertices,
        normals,
        indices,
    })
}

/// Auto-detect format and parse
pub fn parse_stl(data: &[u8]) -> Result<TriangleMesh, StlError> {
    parse_stl_with_limits(data, StlParseLimits::default())
}

/// Auto-detect and parse STL with explicit resource limits.
pub fn parse_stl_with_limits(
    data: &[u8],
    limits: StlParseLimits,
) -> Result<TriangleMesh, StlError> {
    if is_ascii_stl(data) {
        let text =
            std::str::from_utf8(data).map_err(|_| StlError::ParseError("invalid UTF-8".into()))?;
        parse_ascii_stl_with_limits(text, limits)
    } else {
        parse_binary_stl_with_limits(data, limits)
    }
}

fn read_vertex(data: &[u8], offset: usize) -> [f32; 3] {
    [
        f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]),
        f32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]),
        f32::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]),
    ]
}

fn compute_normal(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> [f32; 3] {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let nx = e1[1] * e2[2] - e1[2] * e2[1];
    let ny = e1[2] * e2[0] - e1[0] * e2[2];
    let nz = e1[0] * e2[1] - e1[1] * e2[0];
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-10 {
        [nx / len, ny / len, nz / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[derive(Debug)]
pub enum StlError {
    TooShort(usize),
    TruncatedData {
        expected: usize,
        actual: usize,
    },
    NonFiniteCoordinate(usize),
    ParseError(String),
    ResourceLimit {
        resource: &'static str,
        limit: usize,
    },
}

impl std::fmt::Display for StlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort(n) => write!(f, "STL data too short: {} bytes", n),
            Self::TruncatedData { expected, actual } => write!(
                f,
                "STL truncated: expected {} bytes, got {}",
                expected, actual
            ),
            Self::NonFiniteCoordinate(tri) => {
                write!(f, "non-finite coordinate in triangle {}", tri)
            }
            Self::ParseError(msg) => write!(f, "STL parse error: {}", msg),
            Self::ResourceLimit { resource, limit } => {
                write!(f, "STL resource limit exceeded: {} > {}", resource, limit)
            }
        }
    }
}

impl std::error::Error for StlError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::export_stl;

    fn triangle_mesh() -> TriangleMesh {
        TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        }
    }

    #[test]
    fn binary_triangle_limit_is_enforced_before_allocation() {
        let mut data = vec![0u8; 84];
        data[80..84].copy_from_slice(&2u32.to_le_bytes());
        let result = parse_binary_stl_with_limits(
            &data,
            StlParseLimits {
                max_triangles: 1,
                ..StlParseLimits::default()
            },
        );
        assert!(matches!(
            result,
            Err(StlError::ResourceLimit {
                resource: "triangles",
                ..
            })
        ));
    }

    #[test]
    fn ascii_line_limit_is_enforced() {
        let result = parse_ascii_stl_with_limits(
            "solid x\nendsolid x",
            StlParseLimits {
                max_ascii_lines: 1,
                ..StlParseLimits::default()
            },
        );
        assert!(matches!(
            result,
            Err(StlError::ResourceLimit {
                resource: "ASCII lines",
                ..
            })
        ));
    }

    #[test]
    fn ascii_triangle_limit_is_enforced() {
        let ascii = "solid x\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendsolid x";
        let result = parse_ascii_stl_with_limits(
            ascii,
            StlParseLimits {
                max_triangles: 0,
                ..StlParseLimits::default()
            },
        );
        assert!(matches!(
            result,
            Err(StlError::ResourceLimit {
                resource: "triangles",
                ..
            })
        ));
    }

    #[test]
    fn roundtrip_binary_stl() {
        let original = triangle_mesh();
        let stl_bytes = export_stl(&original);
        let parsed = parse_binary_stl(&stl_bytes).unwrap();
        assert_eq!(parsed.triangle_count(), original.triangle_count());
        assert_eq!(parsed.vertices.len(), original.vertices.len());
    }

    #[test]
    fn roundtrip_cube() {
        let cube = crate::mesh::resolve_to_mesh(&crate::csg::CSGNode::cube());
        let stl = export_stl(&cube);
        let parsed = parse_binary_stl(&stl).unwrap();
        assert_eq!(parsed.triangle_count(), cube.triangle_count());
    }

    #[test]
    fn roundtrip_sphere() {
        let sphere = crate::mesh::resolve_to_mesh(&crate::csg::CSGNode::sphere());
        let stl = export_stl(&sphere);
        let parsed = parse_binary_stl(&stl).unwrap();
        assert_eq!(parsed.triangle_count(), sphere.triangle_count());
    }

    #[test]
    fn reject_too_short() {
        assert!(parse_binary_stl(&[0u8; 10]).is_err());
    }

    #[test]
    fn reject_truncated() {
        // Header says 1 triangle but no triangle data
        let mut data = vec![0u8; 84];
        data[80] = 1; // 1 triangle
        assert!(parse_binary_stl(&data).is_err());
    }

    #[test]
    fn parse_empty_mesh() {
        let empty = TriangleMesh::empty();
        let stl = export_stl(&empty);
        let parsed = parse_binary_stl(&stl).unwrap();
        assert_eq!(parsed.triangle_count(), 0);
    }

    #[test]
    fn ascii_stl_parse() {
        let ascii = r#"solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test"#;
        let mesh = parse_ascii_stl(ascii).unwrap();
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn ascii_detection() {
        assert!(is_ascii_stl(b"solid test\nfacet"));
        assert!(!is_ascii_stl(&[0u8; 84]));
    }

    #[test]
    fn auto_detect_binary() {
        let mesh = triangle_mesh();
        let stl = export_stl(&mesh);
        let parsed = parse_stl(&stl).unwrap();
        assert_eq!(parsed.triangle_count(), 1);
    }

    #[test]
    fn auto_detect_ascii() {
        let ascii = b"solid test\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid test";
        let parsed = parse_stl(ascii).unwrap();
        assert_eq!(parsed.triangle_count(), 1);
    }
}
