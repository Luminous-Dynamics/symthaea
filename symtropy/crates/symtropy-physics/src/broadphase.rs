// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
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

use crate::body::{BodyHandle, BodyType, RigidBody};
use nalgebra::SVector;

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
        Self {
            min: SVector::from_element(f64::MAX),
            max: SVector::from_element(f64::MIN),
        }
    }
    pub fn union(&self, other: &Self) -> Self {
        let mut min = self.min;
        let mut max = self.max;
        for i in 0..D {
            if other.min[i] < min[i] {
                min[i] = other.min[i];
            }
            if other.max[i] > max[i] {
                max[i] = other.max[i];
            }
        }
        Self { min, max }
    }
    pub fn overlaps(&self, other: &Self) -> bool {
        for i in 0..D {
            if self.max[i] < other.min[i] || self.min[i] > other.max[i] {
                return false;
            }
        }
        true
    }
    pub fn from_sphere(center: &SVector<f64, D>, radius: f64) -> Self {
        Self {
            min: center - SVector::from_element(radius),
            max: center + SVector::from_element(radius),
        }
    }
}

// Morton codes
fn quantize(value: f64, world_min: f64, world_max: f64, bits: usize) -> u32 {
    let range = world_max - world_min;
    if range <= 0.0 {
        return 0;
    }
    let max_val = ((1u64 << bits) - 1) as f64;
    let normalized = ((value - world_min) / range).clamp(0.0, 1.0);
    (normalized * max_val) as u32
}

/// Encode a D-dimensional position into a 64-bit Morton code.
pub fn morton_encode<const D: usize>(
    position: &SVector<f64, D>,
    world_min: &SVector<f64, D>,
    world_max: &SVector<f64, D>,
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
    if prefix_bits >= 64 {
        code
    } else {
        code >> (64 - prefix_bits)
    }
}

// LBVH
const BRUTE_FORCE_THRESHOLD: usize = 50;
const LEAF_BIT: u32 = 0x8000_0000;
fn is_leaf(idx: u32) -> bool {
    idx & LEAF_BIT != 0
}
fn leaf_index(idx: u32) -> usize {
    (idx & !LEAF_BIT) as usize
}

#[derive(Clone, Debug)]
struct BvhNode<const D: usize> {
    aabb: Aabb<D>,
    left: u32,
    right: u32,
}

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
                if aabb.min[i] < world_min[i] {
                    world_min[i] = aabb.min[i];
                }
                if aabb.max[i] > world_max[i] {
                    world_max[i] = aabb.max[i];
                }
            }
            aabbs.push(aabb);
        }
        for i in 0..D {
            if world_max[i] - world_min[i] < 1e-6 {
                world_min[i] -= 1.0;
                world_max[i] += 1.0;
            }
        }
        let morton_codes: Vec<u64> = aabbs
            .iter()
            .map(|a| {
                let c = (a.min + a.max) * 0.5;
                morton_encode::<D>(&c, &world_min, &world_max)
            })
            .collect();
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by_key(|&i| morton_codes[i]);
        let sorted_aabbs: Vec<Aabb<D>> = sorted_indices.iter().map(|&i| aabbs[i].clone()).collect();
        let nodes = if n <= 1 {
            Vec::new()
        } else {
            Self::build_nodes(
                &sorted_indices
                    .iter()
                    .map(|&i| morton_codes[i])
                    .collect::<Vec<_>>(),
                &sorted_aabbs,
            )
        };
        Self {
            sorted_indices,
            leaf_aabbs: sorted_aabbs,
            nodes,
        }
    }

    fn find_split(codes: &[u64], first: usize, last: usize) -> usize {
        if codes[first] == codes[last] {
            return (first + last) / 2;
        }
        let cp = (codes[first] ^ codes[last]).leading_zeros();
        let mut split = first;
        let mut step = last - first;
        loop {
            step = step.div_ceil(2);
            let ns = split + step;
            if ns < last && (codes[first] ^ codes[ns]).leading_zeros() > cp {
                split = ns;
            }
            if step <= 1 {
                break;
            }
        }
        split
    }

    fn build_nodes(codes: &[u64], aabbs: &[Aabb<D>]) -> Vec<BvhNode<D>> {
        let mut nodes = Vec::with_capacity(codes.len() - 1);
        Self::build_rec(codes, aabbs, 0, codes.len() - 1, &mut nodes);
        nodes
    }

    fn build_rec(
        codes: &[u64],
        aabbs: &[Aabb<D>],
        first: usize,
        last: usize,
        nodes: &mut Vec<BvhNode<D>>,
    ) -> u32 {
        if first == last {
            return LEAF_BIT | (first as u32);
        }
        let split = Self::find_split(codes, first, last);
        let left = Self::build_rec(codes, aabbs, first, split, nodes);
        let right = Self::build_rec(codes, aabbs, split + 1, last, nodes);
        let la = if is_leaf(left) {
            aabbs[leaf_index(left)].clone()
        } else {
            nodes[left as usize].aabb.clone()
        };
        let ra = if is_leaf(right) {
            aabbs[leaf_index(right)].clone()
        } else {
            nodes[right as usize].aabb.clone()
        };
        let idx = nodes.len() as u32;
        nodes.push(BvhNode {
            aabb: la.union(&ra),
            left,
            right,
        });
        idx
    }

    pub fn query_pairs(&self, bodies: &[RigidBody<D>]) -> Vec<BroadphasePair> {
        let n = self.sorted_indices.len();
        if n <= 1 {
            return Vec::new();
        }
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
                        if lt == BodyType::Static && ot == BodyType::Static {
                            continue;
                        }
                        if !should_collide(
                            &bodies[self.sorted_indices[li]],
                            &bodies[self.sorted_indices[oi]],
                        ) {
                            continue;
                        }
                        if la.overlaps(&self.leaf_aabbs[oi]) {
                            let (ha, hb) = (
                                bodies[self.sorted_indices[li]].handle,
                                bodies[self.sorted_indices[oi]].handle,
                            );
                            if ha < hb {
                                pairs.push(BroadphasePair(ha, hb));
                            } else {
                                pairs.push(BroadphasePair(hb, ha));
                            }
                        }
                    }
                    continue;
                }
                let node = &self.nodes[ni as usize];
                if la.overlaps(&node.aabb) {
                    stack.push(node.left);
                    stack.push(node.right);
                }
            }
        }
        pairs.sort_unstable_by_key(|p| (p.0, p.1));
        pairs.dedup();
        pairs
    }
}

/// Check if two bodies should collide based on collision group/mask filters.
/// Both bodies must accept each other: `(a.group & b.mask != 0) && (b.group & a.mask != 0)`.
#[inline]
fn should_collide<const D: usize>(a: &RigidBody<D>, b: &RigidBody<D>) -> bool {
    (a.collision_group & b.collision_mask != 0) && (b.collision_group & a.collision_mask != 0)
}

/// Find all potentially colliding pairs. Uses LBVH for ≥50 bodies, brute-force otherwise.
pub fn find_pairs<const D: usize>(bodies: &[RigidBody<D>]) -> Vec<BroadphasePair> {
    if bodies.len() < BRUTE_FORCE_THRESHOLD {
        find_pairs_brute(bodies)
    } else {
        Lbvh::build(bodies).query_pairs(bodies)
    }
}

// ---------------------------------------------------------------------------
// Incremental broadphase: cache static/kinematic AABBs across frames
// ---------------------------------------------------------------------------

/// Pre-computed broadphase descriptor for a single body (AABB + sphere + filter).
///
/// Decoupled from `RigidBody<D>` so it can be cached without holding references.
/// Stores both the AABB (for static-cache queries) and the bounding sphere center
/// and radius (for dynamic–dynamic sphere-distance test, matching `find_pairs_brute`).
#[derive(Clone, Debug)]
pub struct BpEntry<const D: usize> {
    pub handle: BodyHandle,
    pub aabb: Aabb<D>,
    /// World-space bounding sphere center.
    pub sphere_center: SVector<f64, D>,
    /// Bounding sphere radius.
    pub sphere_radius: f64,
    pub body_type: BodyType,
    pub collision_group: u32,
    pub collision_mask: u32,
}

impl<const D: usize> BpEntry<D> {
    pub fn from_body(body: &RigidBody<D>) -> Self {
        let (c_local, r) = body.collider.bounding_sphere();
        let c_world = body.transform.transform_point(&c_local).0;
        BpEntry {
            handle: body.handle,
            aabb: Aabb::from_sphere(&c_world, r),
            sphere_center: c_world,
            sphere_radius: r,
            body_type: body.body_type,
            collision_group: body.collision_group,
            collision_mask: body.collision_mask,
        }
    }
}

/// Cached broadphase state for Static and Kinematic bodies.
///
/// Rebuilt only when a static/kinematic body is added or removed (`dirty` flag).
/// This avoids including immovable geometry in the per-frame LBVH rebuild,
/// which is O(n log n) in the total body count.
///
/// For scenes with many static bodies (terrain, walls) and few dynamic bodies,
/// this typically halves or better the broadphase cost.
pub struct StaticBroadphase<const D: usize> {
    entries: Vec<BpEntry<D>>,
}

impl<const D: usize> StaticBroadphase<D> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Rebuild the cache from all bodies in the world.
    ///
    /// Call this when `static_tree_dirty` is set on `PhysicsWorld`.
    pub fn rebuild(&mut self, bodies: &[RigidBody<D>]) {
        self.entries = bodies
            .iter()
            .filter(|b| b.body_type == BodyType::Static || b.body_type == BodyType::Kinematic)
            .map(BpEntry::from_body)
            .collect();
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Enumerate pairs between a dynamic entry and all cached static entries.
    ///
    /// Uses sphere-distance test (same as `find_pairs_brute`) for exact equivalence.
    fn query_against<'a>(
        &'a self,
        dyn_entry: &'a BpEntry<D>,
    ) -> impl Iterator<Item = BroadphasePair> + 'a {
        self.entries.iter().filter_map(move |s| {
            let group_ok = (dyn_entry.collision_group & s.collision_mask != 0)
                && (s.collision_group & dyn_entry.collision_mask != 0);
            if group_ok {
                let dist = (dyn_entry.sphere_center - s.sphere_center).norm();
                if dist <= dyn_entry.sphere_radius + s.sphere_radius {
                    let (ha, hb) = (dyn_entry.handle, s.handle);
                    let pair = if ha < hb {
                        BroadphasePair(ha, hb)
                    } else {
                        BroadphasePair(hb, ha)
                    };
                    return Some(pair);
                }
            }
            None
        })
    }
}

impl<const D: usize> Default for StaticBroadphase<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Find all potentially colliding pairs using the incremental static cache.
///
/// Separates the body list into dynamic and non-dynamic entries:
/// - Dynamic–dynamic: brute-force O(n_dyn²), acceptable for typical games
///   where the number of actively moving bodies is small.
/// - Dynamic–static: linear scan against the prebuilt `StaticBroadphase`.
///   For large static geometry this avoids re-packing static bodies into the
///   LBVH every frame.
///
/// For very large dynamic counts (≥ BRUTE_FORCE_THRESHOLD) the existing
/// `find_pairs` (full LBVH over all bodies) is called instead, since it is
/// superior at scale.
pub fn find_pairs_incremental<const D: usize>(
    bodies: &[RigidBody<D>],
    static_bp: &StaticBroadphase<D>,
) -> Vec<BroadphasePair> {
    // Fast paths
    if bodies.is_empty() {
        return Vec::new();
    }
    if static_bp.is_empty() {
        // No static cache: identical to the standard path.
        return find_pairs(bodies);
    }

    // Compute AABB entries for dynamic bodies only
    let dynamic: Vec<BpEntry<D>> = bodies
        .iter()
        .filter(|b| b.body_type == BodyType::Dynamic)
        .map(BpEntry::from_body)
        .collect();

    // For large dynamic counts fall back to the full LBVH path which is more
    // efficient at scale. The static cache saves rebuilding the full tree.
    if dynamic.len() >= BRUTE_FORCE_THRESHOLD {
        return find_pairs(bodies);
    }

    let mut pairs = Vec::new();

    // Dynamic–dynamic pairs (brute force, sphere distance test matching find_pairs_brute)
    for i in 0..dynamic.len() {
        for j in (i + 1)..dynamic.len() {
            let a = &dynamic[i];
            let b = &dynamic[j];
            let group_ok = (a.collision_group & b.collision_mask != 0)
                && (b.collision_group & a.collision_mask != 0);
            if group_ok {
                let dist = (a.sphere_center - b.sphere_center).norm();
                if dist <= a.sphere_radius + b.sphere_radius {
                    let (ha, hb) = (a.handle, b.handle);
                    if ha < hb {
                        pairs.push(BroadphasePair(ha, hb));
                    } else {
                        pairs.push(BroadphasePair(hb, ha));
                    }
                }
            }
        }
    }

    // Dynamic–static pairs (linear scan against cached static AABB entries)
    for dyn_entry in &dynamic {
        pairs.extend(static_bp.query_against(dyn_entry));
    }

    pairs.sort_unstable_by_key(|p| (p.0, p.1));
    pairs.dedup();
    pairs
}

fn find_pairs_brute<const D: usize>(bodies: &[RigidBody<D>]) -> Vec<BroadphasePair> {
    let mut pairs = Vec::new();
    for i in 0..bodies.len() {
        let ti = bodies[i].body_type;
        for j in (i + 1)..bodies.len() {
            if ti == BodyType::Static && bodies[j].body_type == BodyType::Static {
                continue;
            }
            if !should_collide(&bodies[i], &bodies[j]) {
                continue;
            }
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
            RigidBody::<3>::static_body(
                BodyHandle(1),
                Point::new([0.5, 0.0, 0.0]),
                Box::new(Sphere::unit()),
            ),
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
            bodies.push(RigidBody::<2>::dynamic_sphere(
                BodyHandle(i),
                Point::new([x, 0.0]),
                1.0,
                1.0,
            ));
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
        let a = Aabb::<2> {
            min: SVector::from([0.0, 0.0]),
            max: SVector::from([2.0, 2.0]),
        };
        let b = Aabb::<2> {
            min: SVector::from([1.0, 1.0]),
            max: SVector::from([3.0, 3.0]),
        };
        let c = Aabb::<2> {
            min: SVector::from([5.0, 5.0]),
            max: SVector::from([6.0, 6.0]),
        };
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    // --- Incremental broadphase tests ---

    #[test]
    fn incremental_dynamic_static_detected() {
        let bodies = vec![
            RigidBody::<3>::static_body(BodyHandle(0), Point::origin(), Box::new(Sphere::unit())),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([1.5, 0.0, 0.0]), 1.0, 1.0),
        ];
        let mut static_bp = super::StaticBroadphase::new();
        static_bp.rebuild(&bodies);

        let pairs = super::find_pairs_incremental(&bodies, &static_bp);
        assert_eq!(pairs.len(), 1);
        let p = pairs[0];
        assert!(
            (p.0 == BodyHandle(0) && p.1 == BodyHandle(1))
                || (p.0 == BodyHandle(1) && p.1 == BodyHandle(0))
        );
    }

    #[test]
    fn incremental_static_static_not_produced() {
        let bodies = vec![
            RigidBody::<3>::static_body(BodyHandle(0), Point::origin(), Box::new(Sphere::unit())),
            RigidBody::<3>::static_body(
                BodyHandle(1),
                Point::new([0.5, 0.0, 0.0]),
                Box::new(Sphere::unit()),
            ),
        ];
        let mut static_bp = super::StaticBroadphase::new();
        static_bp.rebuild(&bodies);

        let pairs = super::find_pairs_incremental(&bodies, &static_bp);
        assert!(pairs.is_empty(), "static–static must never collide");
    }

    #[test]
    fn incremental_matches_standard_find_pairs() {
        let bodies = vec![
            RigidBody::<3>::static_body(
                BodyHandle(0),
                Point::new([0.0, 0.0, 0.0]),
                Box::new(Sphere::unit()),
            ),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([1.5, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(2), Point::new([0.0, 1.5, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(3), Point::new([10.0, 0.0, 0.0]), 1.0, 1.0),
        ];
        let mut static_bp = super::StaticBroadphase::new();
        static_bp.rebuild(&bodies);

        let mut incremental = super::find_pairs_incremental(&bodies, &static_bp);
        let mut standard = find_pairs(&bodies);
        incremental.sort_by_key(|p| (p.0, p.1));
        standard.sort_by_key(|p| (p.0, p.1));
        assert_eq!(
            incremental, standard,
            "incremental must match standard broadphase"
        );
    }

    #[test]
    fn incremental_no_static_degrades_to_standard() {
        let bodies = vec![
            RigidBody::<3>::dynamic_sphere(BodyHandle(0), Point::new([0.0, 0.0, 0.0]), 1.0, 1.0),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([1.5, 0.0, 0.0]), 1.0, 1.0),
        ];
        let empty_bp = super::StaticBroadphase::<3>::new(); // no rebuild
        let pairs = super::find_pairs_incremental(&bodies, &empty_bp);
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn static_broadphase_rebuild_updates_cache() {
        let bodies = vec![
            RigidBody::<3>::static_body(BodyHandle(0), Point::origin(), Box::new(Sphere::unit())),
            RigidBody::<3>::dynamic_sphere(BodyHandle(1), Point::new([1.5, 0.0, 0.0]), 1.0, 1.0),
        ];
        let mut static_bp = super::StaticBroadphase::<3>::new();
        assert_eq!(static_bp.len(), 0);

        static_bp.rebuild(&bodies);
        assert_eq!(static_bp.len(), 1, "should cache the one static body");
    }
}
