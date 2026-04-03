// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Broadphase collision detection via LBVH (Linear Bounding Volume Hierarchy).
//!
//! Uses Morton codes (Z-order curves) for spatial sorting, then builds a BVH
//! from the sorted order. The Morton encoding is the only D-dependent code;
//! the hierarchy is dimension-agnostic. Falls back to brute-force for <50 bodies.
//!
//! # Determinism
//! Morton codes use integer quantization, avoiding floating-point heuristics.
//! Pair list sorted by (BodyHandle, BodyHandle) for identical resolution order.

use nalgebra::SVector;
use crate::body::{BodyHandle, BodyType, RigidBody};

/// Pair of body handles that may be colliding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BroadphasePair(pub BodyHandle, pub BodyHandle);

/// Axis-aligned bounding box in D dimensions.
#[derive(Clone, Debug)]
pub struct Aabb<const D: usize> {
    pub min: SVector<f64, D>,
    pub max: SVector<f64, D>,
}

impl<const D: usize> Aabb<D> {
    pub fn empty() -> Self {
        Self { min: SVector::from_element(f64::MAX), max: SVector::from_element(f64::MIN) }
    }
    pub fn union(&self, other: &Self) -> Self {
        let mut min = self.min; let mut max = self.max;
        for i in 0..D {
            if other.min[i] < min[i] { min[i] = other.min[i]; }
            if other.max[i] > max[i] { max[i] = other.max[i]; }
        }
        Self { min, max }
    }
    pub fn overlaps(&self, other: &Self) -> bool {
        for i in 0..D {
            if self.max[i] < other.min[i] || self.min[i] > other.max[i] { return false; }
        }
        true
    }
    pub fn from_sphere(center: &SVector<f64, D>, radius: f64) -> Self {
        Self { min: center - SVector::from_element(radius), max: center + SVector::from_element(radius) }
    }
}

// Morton codes
fn quantize(value: f64, world_min: f64, world_max: f64, bits: usize) -> u32 {
    let range = world_max - world_min;
    if range <= 0.0 { return 0; }
    let max_val = ((1u64 << bits) - 1) as f64;
    let normalized = ((value - world_min) / range).clamp(0.0, 1.0);
    (normalized * max_val) as u32
}

/// Encode a D-dimensional position into a 64-bit Morton code.
pub fn morton_encode<const D: usize>(
    position: &SVector<f64, D>, world_min: &SVector<f64, D>, world_max: &SVector<f64, D>,
) -> u64 {
    let bits = 64 / D;
    let mut code: u64 = 0;
    for bit in 0..bits {
        for axis in 0..D {
            let q = quantize(position[axis], world_min[axis], world_max[axis], bits);
            code |= (((q >> bit) & 1) as u64) << (bit * D + axis);
        }
    }
    code
}

/// Extract top prefix_bits from a Morton code (for spatial authority zones).
pub fn morton_prefix(code: u64, prefix_bits: usize) -> u64 {
    if prefix_bits >= 64 { code } else { code >> (64 - prefix_bits) }
}

// LBVH
const BRUTE_FORCE_THRESHOLD: usize = 50;
const LEAF_BIT: u32 = 0x8000_0000;
fn is_leaf(idx: u32) -> bool { idx & LEAF_BIT != 0 }
fn leaf_index(idx: u32) -> usize { (idx & !LEAF_BIT) as usize }

#[derive(Clone, Debug)]
struct BvhNode<const D: usize> { aabb: Aabb<D>, left: u32, right: u32 }

/// Linear Bounding Volume Hierarchy.
pub struct Lbvh<const D: usize> {
    sorted_indices: Vec<usize>,
    leaf_aabbs: Vec<Aabb<D>>,
    nodes: Vec<BvhNode<D>>,
}

impl<const D: usize> Lbvh<D> {
    pub fn build(bodies: &[RigidBody<D>]) -> Self {
        let n = bodies.len();
        let mut aabbs = Vec::with_capacity(n);
        let mut world_min = SVector::<f64, D>::from_element(f64::MAX);
        let mut world_max = SVector::<f64, D>::from_element(f64::MIN);
        for body in bodies {
            let (c_local, r) = body.collider.bounding_sphere();
            let c_world = body.transform.transform_point(&c_local).0;
            let aabb = Aabb::from_sphere(&c_world, r);
            for i in 0..D {
                if aabb.min[i] < world_min[i] { world_min[i] = aabb.min[i]; }
                if aabb.max[i] > world_max[i] { world_max[i] = aabb.max[i]; }
            }
            aabbs.push(aabb);
        }
        for i in 0..D {
            if world_max[i] - world_min[i] < 1e-6 { world_min[i] -= 1.0; world_max[i] += 1.0; }
        }
        let mut morton_codes: Vec<u64> = aabbs.iter()
            .map(|a| { let c = (a.min + a.max) * 0.5; morton_encode::<D>(&c, &world_min, &world_max) })
            .collect();
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by_key(|&i| morton_codes[i]);
        let sorted_aabbs: Vec<Aabb<D>> = sorted_indices.iter().map(|&i| aabbs[i].clone()).collect();
        let nodes = if n <= 1 { Vec::new() } else { Self::build_nodes(&sorted_indices.iter().map(|&i| morton_codes[i]).collect::<Vec<_>>(), &sorted_aabbs) };
        Self { sorted_indices, leaf_aabbs: sorted_aabbs, nodes }
    }

    fn find_split(codes: &[u64], first: usize, last: usize) -> usize {
        if codes[first] == codes[last] { return (first + last) / 2; }
        let cp = (codes[first] ^ codes[last]).leading_zeros();
        let mut split = first; let mut step = last - first;
        loop {
            step = (step + 1) / 2;
            let ns = split + step;
            if ns < last && (codes[first] ^ codes[ns]).leading_zeros() > cp { split = ns; }
            if step <= 1 { break; }
        }
        split
    }

    fn build_nodes(codes: &[u64], aabbs: &[Aabb<D>]) -> Vec<BvhNode<D>> {
        let mut nodes = Vec::with_capacity(codes.len() - 1);
        Self::build_rec(codes, aabbs, 0, codes.len() - 1, &mut nodes);
        nodes
    }

    fn build_rec(codes: &[u64], aabbs: &[Aabb<D>], first: usize, last: usize, nodes: &mut Vec<BvhNode<D>>) -> u32 {
        if first == last { return LEAF_BIT | (first as u32); }
        let split = Self::find_split(codes, first, last);
        let left = Self::build_rec(codes, aabbs, first, split, nodes);
        let right = Self::build_rec(codes, aabbs, split + 1, last, nodes);
        let la = if is_leaf(left) { aabbs[leaf_index(left)].clone() } else { nodes[left as usize].aabb.clone() };
        let ra = if is_leaf(right) { aabbs[leaf_index(right)].clone() } else { nodes[right as usize].aabb.clone() };
        let idx = nodes.len() as u32;
        nodes.push(BvhNode { aabb: la.union(&ra), left, right });
        idx
    }

    pub fn query_pairs(&self, bodies: &[RigidBody<D>]) -> Vec<BroadphasePair> {
        let n = self.sorted_indices.len();
        if n <= 1 { return Vec::new(); }
        let mut pairs = Vec::new();
        let root = (self.nodes.len() - 1) as u32;
        for li in 0..n {
            let la = &self.leaf_aabbs[li];
            let lt = bodies[self.sorted_indices[li]].body_type;
            let mut stack = arrayvec::ArrayVec::<u32, 64>::new();
            stack.push(root);
            while let Some(ni) = stack.pop() {
                if is_leaf(ni) {
                    let oi = leaf_index(ni);
                    if oi > li {
                        let ot = bodies[self.sorted_indices[oi]].body_type;
                        if lt == BodyType::Static && ot == BodyType::Static { continue; }
                        if la.overlaps(&self.leaf_aabbs[oi]) {
                            let (ha, hb) = (bodies[self.sorted_indices[li]].handle, bodies[self.sorted_indices[oi]].handle);
                            if ha < hb { pairs.push(BroadphasePair(ha, hb)); }
                            else { pairs.push(BroadphasePair(hb, ha)); }
                        }
                    }
                    continue;
                }
                let node = &self.nodes[ni as usize];
                if la.overlaps(&node.aabb) { stack.push(node.left); stack.push(node.right); }
            }
        }
        pairs.sort_unstable_by_key(|p| (p.0, p.1));
        pairs.dedup();
        pairs
    }
}

/// Find all potentially colliding pairs. Uses LBVH for ≥50 bodies, brute-force otherwise.
pub fn find_pairs<const D: usize>(bodies: &[RigidBody<D>]) -> Vec<BroadphasePair> {
    if bodies.len() < BRUTE_FORCE_THRESHOLD { find_pairs_brute(bodies) }
    else { Lbvh::build(bodies).query_pairs(bodies) }
}

fn find_pairs_brute<const D: usize>(bodies: &[RigidBody<D>]) -> Vec<BroadphasePair> {
    let mut pairs = Vec::new();
    for i in 0..bodies.len() {
        let ti = bodies[i].body_type;
        for j in (i+1)..bodies.len() {
            if ti == BodyType::Static && bodies[j].body_type == BodyType::Static { continue; }
            let (ci, ri) = bodies[i].collider.bounding_sphere();
            let (cj, rj) = bodies[j].collider.bounding_sphere();
            let wi = bodies[i].transform.transform_point(&ci);
            let wj = bodies[j].transform.transform_point(&cj);
            if wi.distance(&wj) <= ri + rj {
                pairs.push(BroadphasePair(bodies[i].handle, bodies[j].handle));
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use symtropy_math::{Point, Sphere};

    #[test]
    fn overlapping_pair_detected() {
        let bodies = vec![
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([1.5, 0.0, 0.0]), 1.0, 1.0),
        ];
        assert_eq!(find_pairs(&bodies).len(), 1);
    }

    #[test]
    fn separated_pair_not_detected() {
        let bodies = vec![
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([5.0, 0.0, 0.0]), 1.0, 1.0),
        ];
        assert!(find_pairs(&bodies).is_empty());
    }

    #[test]
    fn static_static_ignored() {
        let bodies = vec![
            RigidBody::<3>::static_body(BodyHandle(0), Point::origin(), Box::new(Sphere::unit())),
            RigidBody::<3>::static_body(BodyHandle(1), Point::new([0.5, 0.0, 0.0]), Box::new(Sphere::unit())),
        ];
        assert!(find_pairs(&bodies).is_empty());
    }

    #[test]
    fn static_dynamic_detected() {
        let bodies = vec![
            RigidBody::<3>::static_body(BodyHandle(0), Point::origin(), Box::new(Sphere::unit())),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([1.5, 0.0, 0.0]), 1.0, 1.0),
        ];
        assert_eq!(find_pairs(&bodies).len(), 1);
    }

    #[test]
    fn multiple_pairs() {
        let bodies = vec![
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([1.5, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(2), Point::new([10.0, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(3), Point::new([1.0, 1.0, 0.0]), 1.0, 1.0),
        ];
        assert!(find_pairs(&bodies).len() >= 2);
    }

    #[test]
    fn morton_encode_2d_origin() {
        let pos = SVector::<f64, 2>::from([0.0, 0.0]);
        let min = SVector::<f64, 2>::from([0.0, 0.0]);
        let max = SVector::<f64, 2>::from([100.0, 100.0]);
        assert_eq!(morton_encode::<2>(&pos, &min, &max), 0);
    }

    #[test]
    fn morton_locality() {
        let min = SVector::<f64, 3>::from([0.0; 3]);
        let max = SVector::<f64, 3>::from([100.0; 3]);
        let a = morton_encode::<3>(&SVector::from([10.0, 10.0, 10.0]), &min, &max);
        let b = morton_encode::<3>(&SVector::from([11.0, 10.0, 10.0]), &min, &max);
        let far = morton_encode::<3>(&SVector::from([90.0, 90.0, 90.0]), &min, &max);
        assert!((a as i64 - b as i64).unsigned_abs() < (a as i64 - far as i64).unsigned_abs());
    }

    #[test]
    fn lbvh_matches_brute_force() {
        let mut bodies = Vec::new();
        for i in 0..(BRUTE_FORCE_THRESHOLD + 10) {
            let x = (i as f64) * 0.5;
            bodies.push(RigidBody::<2>::dynamic_sphere(BodyHandle(i), Point::new([x, 0.0]), 1.0, 1.0));
        }
        let mut brute = find_pairs_brute(&bodies);
        brute.sort_by_key(|p| (p.0, p.1));
        let lbvh = find_pairs(&bodies);
        for pair in &brute {
            assert!(lbvh.contains(pair), "LBVH missing pair {:?}", pair);
        }
    }

    #[test]
    fn lbvh_deterministic() {
        let bodies = vec![
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([1.5, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(2), Point::new([0.5, 0.5, 0.5]), 0.8, 1.0),
        ];
        assert_eq!(find_pairs(&bodies), find_pairs(&bodies));
    }

    #[test]
    fn aabb_overlap() {
        let a = Aabb::<2> { min: SVector::from([0.0, 0.0]), max: SVector::from([2.0, 2.0]) };
        let b = Aabb::<2> { min: SVector::from([1.0, 1.0]), max: SVector::from([3.0, 3.0]) };
        let c = Aabb::<2> { min: SVector::from([5.0, 5.0]), max: SVector::from([6.0, 6.0]) };
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }
}
