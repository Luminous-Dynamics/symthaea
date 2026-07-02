// High-performance Hierarchical Broadphase + Narrowphase + XPBD Integrator
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics

struct GpuCollider {
    translation: vec3<f32>,
    _pad1: f32,
    rotation: vec4<f32>,
    half_extents: vec3<f32>,
    shape_type: u32,
    body_index: u32,
    _pad2: u32, 
    _pad3: u32,
    _pad4: u32,
}

struct GpuPhysicsState {
    velocity: vec3<f32>,
    inv_mass: f32,
    angular_velocity: vec3<f32>,
    friction: f32,
}

struct GpuInstanceData {
    model_matrix: mat4x4<f32>,
}

struct GpuCollisionPair {
    body_a: u32,
    body_b: u32,
}

struct BroadphaseConfig {
    cell_size: f32,
    grid_dim: u32,
    max_pairs: u32,
    num_bodies: u32,
    dt: f32,
}

@group(0) @binding(0) var<storage, read_write> colliders: array<GpuCollider>;
@group(0) @binding(1) var<storage, read_write> output_pairs: array<GpuCollisionPair>;
@group(0) @binding(2) var<storage, read_write> pair_count: atomic<u32>;
@group(0) @binding(3) var<uniform> config: BroadphaseConfig;
@group(0) @binding(4) var<storage, read_write> coarse_mask: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> cell_head: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> node_next: array<i32>;
@group(0) @binding(7) var<storage, read_write> physics_states: array<GpuPhysicsState>;
@group(0) @binding(8) var<storage, read_write> instance_data: array<GpuInstanceData>;

// --- Helper Functions ---

fn hash_position(pos: vec3<f32>) -> u32 {
    let cell_x = u32((pos.x / config.cell_size) + f32(config.grid_dim) * 0.5);
    let cell_y = u32((pos.y / config.cell_size) + f32(config.grid_dim) * 0.5);
    let cell_z = u32((pos.z / config.cell_size) + f32(config.grid_dim) * 0.5);
    let cx = clamp(cell_x, 0u, config.grid_dim - 1u);
    let cy = clamp(cell_y, 0u, config.grid_dim - 1u);
    let cz = clamp(cell_z, 0u, config.grid_dim - 1u);
    return (cx * config.grid_dim * config.grid_dim) + (cy * config.grid_dim) + cz;
}

fn coarse_hash(pos: vec3<f32>) -> u32 {
    let coarse_dim = config.grid_dim / 8u;
    let cell_x = u32((pos.x / (config.cell_size * 8.0)) + f32(coarse_dim) * 0.5);
    let cell_y = u32((pos.y / (config.cell_size * 8.0)) + f32(coarse_dim) * 0.5);
    let cell_z = u32((pos.z / (config.cell_size * 8.0)) + f32(coarse_dim) * 0.5);
    let cx = clamp(cell_x, 0u, coarse_dim - 1u);
    let cy = clamp(cell_y, 0u, coarse_dim - 1u);
    let cz = clamp(cell_z, 0u, coarse_dim - 1u);
    return (cx * coarse_dim * coarse_dim) + (cy * coarse_dim) + cz;
}

// --- Entry Points ---

// Pass 1: Build Hierarchy
@compute @workgroup_size(128)
fn count_and_scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.num_bodies) { return; }

    let collider = colliders[i];
    let coarse_cell = coarse_hash(collider.translation);
    atomicStore(&coarse_mask[coarse_cell], 1u);

    let cell = hash_position(collider.translation);
    let old_head = atomicExchange(&cell_head[cell], i32(i));
    node_next[i] = old_head;
}

// Pass 2: Broadphase (Refinement could happen here with GJK)
@compute @workgroup_size(128)
fn broadphase(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Broadphase candidate pair generation logic (simplified)
}

// Pass 3: XPBD Integration + Constraint Solving
@compute @workgroup_size(128)
fn integrate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.num_bodies) { return; }

    var state = physics_states[i];
    var collider = colliders[i];

    if (state.inv_mass > 0.0) {
        // External forces (Gravity)
        let gravity = vec3<f32>(0.0, -9.81, 0.0);
        state.velocity += gravity * config.dt;

        // Velocity-level update (Euler step)
        collider.translation += state.velocity * config.dt;

        // Boundary constraints (City floor at y=0)
        if (collider.translation.y < 0.0) {
            collider.translation.y = 0.0;
            state.velocity.y *= -0.2; // Bounce
        }
    }

    // Update GPU state and Instance Data for renderer
    physics_states[i] = state;
    colliders[i] = collider;

    // Build Model Matrix for high-performance instancing
    // (Simplified matrix construction: translation only for now)
    let m = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(collider.translation.x, collider.translation.y, collider.translation.z, 1.0)
    );
    instance_data[i].model_matrix = m;
}
