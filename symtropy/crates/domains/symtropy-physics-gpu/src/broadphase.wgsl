// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

struct GpuAabb {
    min: vec3<f32>,
    max: vec3<f32>,
    body_index: u32,
};

struct CollisionPair {
    body_a: u32,
    body_b: u32,
};

@group(0) @binding(0) var<storage, read> input_aabbs: array<GpuAabb>;
@group(0) @binding(1) var<storage, read_write> output_pairs: array<CollisionPair>;
@group(0) @binding(2) var<storage, read_write> pair_count: atomic<u32>;

// Simple O(N^2) GPU broadphase for initial scaffolding.
// Will be upgraded to spatial hashing (Grid/Morton) in next iteration.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let num_bodies = arrayLength(&input_aabbs);

    if (i >= num_bodies) {
        return;
    }

    let a = input_aabbs[i];

    for (var j = i + 1u; j < num_bodies; j = j + 1u) {
        let b = input_aabbs[j];

        // AABB overlap check
        if (a.max.x < b.min.x || a.min.x > b.max.x) { continue; }
        if (a.max.y < b.min.y || a.min.y > b.max.y) { continue; }
        if (a.max.z < b.min.z || a.min.z > b.max.z) { continue; }

        // Overlap found
        let pair_idx = atomicAdd(&pair_count, 1u);
        if (pair_idx < arrayLength(&output_pairs)) {
            output_pairs[pair_idx] = CollisionPair(a.body_index, b.body_index);
        }
    }
}
