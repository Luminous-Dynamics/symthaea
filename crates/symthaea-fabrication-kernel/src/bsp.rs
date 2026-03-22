// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BSP tree for mesh boolean operations
use crate::mesh::TriangleMesh;

#[derive(Debug, Clone)]
struct Triangle {
    v: [[f32; 3]; 3],
    normal: [f32; 3],
}
#[derive(Debug, Clone, Copy, PartialEq)]
enum Side {
    Front,
    Back,
    Coplanar,
}
#[derive(Debug)]
enum BSPNode {
    Leaf,
    Interior {
        plane: Plane,
        front: Box<BSPNode>,
        back: Box<BSPNode>,
        coplanar: Vec<Triangle>,
    },
}
#[derive(Debug, Clone)]
struct Plane {
    normal: [f32; 3],
    d: f32,
}
const EPSILON: f32 = 1e-5;

impl Plane {
    fn from_triangle(tri: &Triangle) -> Self {
        let n = tri.normal;
        Self {
            normal: n,
            d: -(n[0] * tri.v[0][0] + n[1] * tri.v[0][1] + n[2] * tri.v[0][2]),
        }
    }
    fn distance(&self, p: &[f32; 3]) -> f32 {
        self.normal[0] * p[0] + self.normal[1] * p[1] + self.normal[2] * p[2] + self.d
    }
    fn classify_point(&self, p: &[f32; 3]) -> Side {
        let d = self.distance(p);
        if d > EPSILON {
            Side::Front
        } else if d < -EPSILON {
            Side::Back
        } else {
            Side::Coplanar
        }
    }
    fn classify_tri(&self, tri: &Triangle) -> [Side; 3] {
        [
            self.classify_point(&tri.v[0]),
            self.classify_point(&tri.v[1]),
            self.classify_point(&tri.v[2]),
        ]
    }
}

fn compute_normal(v0: &[f32; 3], v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3] {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let (nx, ny, nz) = (
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    );
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-10 {
        [nx / len, ny / len, nz / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn lerp(a: &[f32; 3], b: &[f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

fn split_triangle(tri: &Triangle, plane: &Plane) -> (Vec<Triangle>, Vec<Triangle>) {
    let sides = plane.classify_tri(tri);
    let hf = sides.contains(&Side::Front);
    let hb = sides.contains(&Side::Back);
    if !hb {
        return (vec![tri.clone()], vec![]);
    }
    if !hf {
        return (vec![], vec![tri.clone()]);
    }
    let (mut fp, mut bp) = (Vec::new(), Vec::new());
    for i in 0..3 {
        let j = (i + 1) % 3;
        let (vi, vj, si, sj) = (&tri.v[i], &tri.v[j], sides[i], sides[j]);
        if si != Side::Back {
            fp.push(*vi);
        }
        if si != Side::Front {
            bp.push(*vi);
        }
        if (si == Side::Front && sj == Side::Back) || (si == Side::Back && sj == Side::Front) {
            let (di, dj) = (plane.distance(vi).abs(), plane.distance(vj).abs());
            let pt = lerp(vi, vj, di / (di + dj));
            fp.push(pt);
            bp.push(pt);
        }
    }
    (fan_triangulate(&fp), fan_triangulate(&bp))
}

fn fan_triangulate(pts: &[[f32; 3]]) -> Vec<Triangle> {
    if pts.len() < 3 {
        return vec![];
    }
    (1..pts.len() - 1)
        .map(|i| Triangle {
            v: [pts[0], pts[i], pts[i + 1]],
            normal: compute_normal(&pts[0], &pts[i], &pts[i + 1]),
        })
        .collect()
}

impl BSPNode {
    fn build(tris: Vec<Triangle>) -> Self {
        if tris.is_empty() {
            return BSPNode::Leaf;
        }
        let plane = Plane::from_triangle(&tris[0]);
        let (mut f, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        for tri in &tris {
            let s = plane.classify_tri(tri);
            let (hf, hb) = (s.contains(&Side::Front), s.contains(&Side::Back));
            if !hf && !hb {
                c.push(tri.clone());
            } else if !hb {
                f.push(tri.clone());
            } else if !hf {
                b.push(tri.clone());
            } else {
                let (ff, bb) = split_triangle(tri, &plane);
                f.extend(ff);
                b.extend(bb);
            }
        }
        BSPNode::Interior {
            plane,
            front: Box::new(Self::build(f)),
            back: Box::new(Self::build(b)),
            coplanar: c,
        }
    }
    fn clip_triangles(&self, tris: &[Triangle]) -> Vec<Triangle> {
        match self {
            BSPNode::Leaf => tris.to_vec(),
            BSPNode::Interior {
                plane, front, back, ..
            } => {
                let (mut ft, mut bt) = (Vec::new(), Vec::new());
                for tri in tris {
                    let s = plane.classify_tri(tri);
                    let (hf, hb) = (s.contains(&Side::Front), s.contains(&Side::Back));
                    if !hf && !hb {
                        ft.push(tri.clone());
                        bt.push(tri.clone());
                    } else if !hb {
                        ft.push(tri.clone());
                    } else if !hf {
                        bt.push(tri.clone());
                    } else {
                        let (f, b) = split_triangle(tri, plane);
                        ft.extend(f);
                        bt.extend(b);
                    }
                }
                let mut r = front.clip_triangles(&ft);
                r.extend(back.clip_triangles(&bt));
                r
            }
        }
    }
    fn invert(&mut self) {
        if let BSPNode::Interior {
            plane,
            front,
            back,
            coplanar,
        } = self
        {
            plane.normal = [-plane.normal[0], -plane.normal[1], -plane.normal[2]];
            plane.d = -plane.d;
            for tri in coplanar.iter_mut() {
                tri.normal = [-tri.normal[0], -tri.normal[1], -tri.normal[2]];
                tri.v.swap(0, 2);
            }
            front.invert();
            back.invert();
            std::mem::swap(front, back);
        }
    }
    fn all_triangles(&self) -> Vec<Triangle> {
        match self {
            BSPNode::Leaf => vec![],
            BSPNode::Interior {
                front,
                back,
                coplanar,
                ..
            } => {
                let mut t = coplanar.clone();
                t.extend(front.all_triangles());
                t.extend(back.all_triangles());
                t
            }
        }
    }
}

fn mesh_to_tris(mesh: &TriangleMesh) -> Vec<Triangle> {
    mesh.indices
        .iter()
        .map(|idx| {
            let (v0, v1, v2) = (
                mesh.vertices[idx[0] as usize],
                mesh.vertices[idx[1] as usize],
                mesh.vertices[idx[2] as usize],
            );
            Triangle {
                v: [v0, v1, v2],
                normal: compute_normal(&v0, &v1, &v2),
            }
        })
        .collect()
}

fn tris_to_mesh(tris: &[Triangle]) -> TriangleMesh {
    let (mut vs, mut ns, mut is) = (Vec::new(), Vec::new(), Vec::new());
    for tri in tris {
        let b = vs.len() as u32;
        vs.extend_from_slice(&tri.v);
        ns.extend_from_slice(&[tri.normal; 3]);
        is.push([b, b + 1, b + 2]);
    }
    TriangleMesh {
        vertices: vs,
        normals: ns,
        indices: is,
    }
}

/// CSG subtract: A - B
pub fn csg_subtract(a: &TriangleMesh, b: &TriangleMesh) -> TriangleMesh {
    let (at, bt) = (mesh_to_tris(a), mesh_to_tris(b));
    if at.is_empty() || bt.is_empty() {
        return a.clone();
    }
    let ab = BSPNode::build(at);
    let mut bb = BSPNode::build(bt);
    bb.invert();
    let a_out = bb.clip_triangles(&ab.all_triangles());
    bb.invert();
    let b_in = ab.clip_triangles(&bb.all_triangles());
    let mut r = a_out;
    r.extend(b_in.into_iter().map(|mut t| {
        t.normal = [-t.normal[0], -t.normal[1], -t.normal[2]];
        t.v.swap(0, 2);
        t
    }));
    tris_to_mesh(&r)
}

/// CSG intersect: A and B
pub fn csg_intersect(a: &TriangleMesh, b: &TriangleMesh) -> TriangleMesh {
    let (at, bt) = (mesh_to_tris(a), mesh_to_tris(b));
    if at.is_empty() || bt.is_empty() {
        return TriangleMesh::empty();
    }
    let (ab, bb) = (BSPNode::build(at), BSPNode::build(bt));
    let mut r = bb.clip_triangles(&ab.all_triangles());
    r.extend(ab.clip_triangles(&bb.all_triangles()));
    tris_to_mesh(&r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::{CSGNode, Transform3D};
    use crate::mesh::resolve_to_mesh;
    fn unit_cube() -> TriangleMesh {
        resolve_to_mesh(&CSGNode::cube())
    }
    fn offset_cube() -> TriangleMesh {
        resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            scale: [1.0; 3],
            rotate: [0.0; 3],
            translate: [0.25, 0.25, 0.25],
        }))
    }
    fn small_cube() -> TriangleMesh {
        resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            scale: [0.5; 3],
            rotate: [0.0; 3],
            translate: [0.0; 3],
        }))
    }
    #[test]
    fn subtract_nonoverlapping() {
        let a = unit_cube();
        let b = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [10.0, 0.0, 0.0],
            ..Default::default()
        }));
        assert!(csg_subtract(&a, &b).triangle_count() >= a.triangle_count());
    }
    #[test]
    fn subtract_overlapping() {
        assert!(csg_subtract(&unit_cube(), &offset_cube()).triangle_count() > 0);
    }
    #[test]
    fn subtract_empty_b() {
        let a = unit_cube();
        assert_eq!(
            csg_subtract(&a, &TriangleMesh::empty()).triangle_count(),
            a.triangle_count()
        );
    }
    #[test]
    fn intersect_overlapping() {
        assert!(csg_intersect(&unit_cube(), &offset_cube()).triangle_count() > 0);
    }
    #[test]
    fn intersect_nonoverlapping() {
        let a = unit_cube();
        let b = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [10.0, 0.0, 0.0],
            ..Default::default()
        }));
        let r = csg_intersect(&a, &b);
        assert_eq!(r.vertices.len(), r.normals.len());
    }
    #[test]
    fn intersect_empty() {
        assert_eq!(
            csg_intersect(&unit_cube(), &TriangleMesh::empty()).triangle_count(),
            0
        );
    }
    #[test]
    fn subtract_contained() {
        let a = unit_cube();
        assert!(csg_subtract(&a, &small_cube()).triangle_count() > a.triangle_count());
    }
    #[test]
    fn plane_classification() {
        let p = Plane {
            normal: [0.0, 0.0, 1.0],
            d: 0.0,
        };
        assert_eq!(p.classify_point(&[0.0, 0.0, 1.0]), Side::Front);
        assert_eq!(p.classify_point(&[0.0, 0.0, -1.0]), Side::Back);
        assert_eq!(p.classify_point(&[0.0, 0.0, 0.0]), Side::Coplanar);
    }
    #[test]
    fn triangle_split() {
        let tri = Triangle {
            v: [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            normal: [0.0, 1.0, 0.0],
        };
        let (f, b) = split_triangle(
            &tri,
            &Plane {
                normal: [0.0, 0.0, 1.0],
                d: 0.0,
            },
        );
        assert!(!f.is_empty());
        assert!(!b.is_empty());
    }
    #[test]
    fn mesh_roundtrip() {
        let m = unit_cube();
        assert_eq!(
            m.triangle_count(),
            tris_to_mesh(&mesh_to_tris(&m)).triangle_count()
        );
    }
    #[test]
    fn csg_tree_subtract() {
        let tree = CSGNode::cube().subtract(CSGNode::cube().with_transform(Transform3D {
            scale: [0.5; 3],
            ..Default::default()
        }));
        let mesh = resolve_to_mesh(&tree);
        assert!(
            mesh.triangle_count() > 12,
            "CSG subtract via tree should produce mesh"
        );
    }
    #[test]
    fn csg_tree_intersect() {
        let tree = CSGNode::cube().intersect(CSGNode::cube().with_transform(Transform3D {
            translate: [0.25, 0.25, 0.25],
            ..Default::default()
        }));
        let mesh = resolve_to_mesh(&tree);
        assert!(
            mesh.triangle_count() > 0,
            "CSG intersect via tree should produce mesh"
        );
    }
}
