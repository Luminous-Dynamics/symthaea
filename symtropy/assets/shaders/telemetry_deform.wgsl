// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

#import bevy_pbr::{
    mesh_functions,
    mesh_view_bindings::view,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

#import bevy_pbr::forward_io::Vertex

struct NodeTelemetryGpu {
    position: vec3<f32>,
    variational_free_energy: f32,
    bandwidth_bps: f32,
    latency_ms: f32,
    tunnel_state: u32,
    dht_holding_completeness: f32,
    gossip_frequency_hz: f32,
    validation_failure_count: u32,
    wasm_memory_fraction: f32,
    last_hot_reload_time: f32,
    holographic_coherence: f32,
    thermal_gradient: f32,
    circuit_load: f32,
    _padding: f32,
};

struct TelemetrySettings {
    node_index: u32,
    time: f32,
    deformation_scale: f32,
    noise_frequency: f32,
};

@group(2) @binding(100)
var<uniform> settings: TelemetrySettings;

@group(2) @binding(101)
var<storage, read> nodes: array<NodeTelemetryGpu>;

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;

    // Fetch the telemetry node data
    let node = nodes[settings.node_index];
    let surprise = node.variational_free_energy;

    // Spatial noise/wave displacement function
    let p = vertex.position;
    let wave = sin(p.x * settings.noise_frequency + settings.time * 4.0) *
               cos(p.y * settings.noise_frequency + settings.time * 3.2) *
               sin(p.z * settings.noise_frequency + settings.time * 5.1);

    // Displace vertices along normal vector, scaled by Active Inference surprise spike
    let displacement = vertex.normal * surprise * settings.deformation_scale * wave;
    let modified_pos = p + displacement;

    // Standard Bevy transform calculations
    let model = mesh_functions::get_world_from_local(vertex.instance_index);
    let world_pos = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(modified_pos, 1.0));
    
    out.world_position = world_pos;
    out.world_normal = mesh_functions::mesh_normal_local_to_world(vertex.normal, vertex.instance_index);
    out.uv = vertex.uv;
    out.clip_position = mesh_functions::mesh_position_world_to_clip(world_pos.xyz);

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        model,
        vertex.tangent
    );
#endif

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif

    return out;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // 1. Generate standard PBR input from material
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // 2. Fetch node telemetry data
    let node = nodes[settings.node_index];
    let surprise = node.variational_free_energy;
    let coherence = node.holographic_coherence; // Phi
    let thermal = node.thermal_gradient;        // Delta Kelvin
    let load = node.circuit_load;               // load line
    let failures = node.validation_failure_count;

    // 3. Holographic Coherence (Phi) color mapping
    // Low coherence (unstable) -> Cool/Cyan/Purple
    // High coherence (stable) -> Gold/Amber/Greenish-yellow
    let low_coherence_color = vec3<f32>(0.0, 0.4, 1.0); // Electric Cyan-Blue
    let high_coherence_color = vec3<f32>(0.9, 0.65, 0.1); // Golden Amber
    let base_tint = mix(low_coherence_color, high_coherence_color, coherence);

    // 4. Modulate base color with the active inference telemetry tint
    pbr_input.material.base_color = vec4<f32>(
        mix(pbr_input.material.base_color.rgb, base_tint, 0.7),
        pbr_input.material.base_color.a
    );

    // 5. Thermal Gradient and load -> Emissive Glow
    // If the node is running hot or under heavy load, it starts to glow red/orange
    let thermal_offset = max(thermal - 293.15, 0.0); // Delta above baseline room temp (293.15K)
    let heat_glow = clamp(thermal_offset * 0.05 + load * 0.4, 0.0, 1.0);
    let thermal_color = vec3<f32>(1.0, 0.15, 0.0) * heat_glow * 3.0; // Glow intensity
    pbr_input.material.emissive = pbr_input.material.emissive + vec4<f32>(thermal_color, 0.0);

    // 6. Validation Failure / Cryptographic Slashing -> Static noise/flicker
    if (failures > 0u) {
        let flicker_speed = 30.0;
        let noise = fract(sin(dot(in.world_position.xy, vec2<f32>(12.9898, 78.233)) + settings.time * flicker_speed) * 43758.5453);
        let flicker_threshold = 1.0 - (f32(failures) * 0.15); // more failures -> more flickering
        if (noise > flicker_threshold) {
            // Drop base color brightness or tint red to indicate handshake/crypto compromise
            pbr_input.material.base_color = vec4<f32>(
                pbr_input.material.base_color.rgb * 0.1 + vec3<f32>(0.8, 0.0, 0.0),
                pbr_input.material.base_color.a
            );
            pbr_input.material.emissive = pbr_input.material.emissive + vec4<f32>(1.5, 0.0, 0.0, 0.0);
        }
    }

    // 7. Subtle holographic scanline
    let scanline = sin(in.world_position.y * 12.0 + settings.time * 6.0) * 0.08 + 0.92;
    pbr_input.material.base_color = vec4<f32>(pbr_input.material.base_color.rgb * scanline, pbr_input.material.base_color.a);

    // Standard Bevy output pipeline
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
