// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use web_sys::{WebGl2RenderingContext, WebGlProgram, WebGlShader};

// ─── Shared GLSL Utilities ──────────────────────────────────────

/// 3D Simplex noise for procedural effects (aurora, caustics, organic motion).
/// Prepended to shaders that need it.
const SNOISE_GLSL: &str = r#"
// Simplex noise — adapted from Ashima Arts (MIT License)
vec3 mod289(vec3 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+10.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(
        i.z + vec4(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 105.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
"#;

/// Eight Harmonies color palette as GLSL constant array.
const HARMONY_COLORS_GLSL: &str = r#"
// Eight Harmonies of Infinite Love — ERC palette
const vec3 HARMONY_COLORS[8] = vec3[8](
    vec3(0.40, 0.49, 0.92),  // 1. Resonant Coherence — #667eea (indigo-blue)
    vec3(0.91, 0.12, 0.39),  // 2. Pan-Sentient Flourishing — #E91E63 (heart pink)
    vec3(0.00, 0.54, 0.48),  // 3. Integral Wisdom — #00897B (wisdom green)
    vec3(1.00, 0.84, 0.00),  // 4. Infinite Play — #FFD700 (sacred gold)
    vec3(0.00, 0.87, 1.00),  // 5. Universal Interconnectedness — #00ddff (cyan)
    vec3(0.49, 0.23, 0.93),  // 6. Sacred Reciprocity — #7c3aed (deep purple)
    vec3(0.23, 0.51, 0.96),  // 7. Evolutionary Progression — #3b82f6 (electric blue)
    vec3(0.06, 0.09, 0.16)   // 8. Sacred Stillness — deep indigo (fade)
);

vec3 harmony_at_time(float t) {
    float cycle = mod(t, 8.0) / 8.0;
    float idx_f = cycle * 8.0;
    int idx_a = int(idx_f) % 8;
    int idx_b = (idx_a + 1) % 8;
    float blend = fract(idx_f);
    return mix(HARMONY_COLORS[idx_a], HARMONY_COLORS[idx_b], blend);
}
"#;

// ─── Holographic Earth Shader (Phase 1) ─────────────────────────

pub const EARTH_VERT: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4 u_projection;
uniform mat4 u_model_view;
uniform mat3 u_normal_matrix;
uniform sampler2D u_topo_vertex;  // topology texture for vertex displacement
uniform float u_relief_scale;     // 0.0 = flat, 1.0 = full relief

out vec3 v_normal;
out vec2 v_uv;
out vec3 v_view_position;
out vec3 v_world_normal;
out float v_elevation;

void main() {
    // Sample topology for elevation (vertex shader uses textureLod)
    float elevation = textureLod(u_topo_vertex, a_uv, 0.0).r;

    // Displace vertex outward along normal based on elevation
    // Scale: 0.0 = sea level, 0.03 = Everest-scale peak
    float displacement = elevation * 0.03 * u_relief_scale;
    vec3 displaced = a_position + a_normal * displacement;

    v_normal = normalize(u_normal_matrix * a_normal);
    v_world_normal = normalize(displaced);  // use displaced position for world normal
    v_uv = a_uv;
    v_elevation = elevation;

    vec4 mv_position = u_model_view * vec4(displaced, 1.0);
    v_view_position = -mv_position.xyz;
    gl_Position = u_projection * mv_position;
}
"#;

pub fn earth_frag() -> String {
    let mut s = String::from("#version 300 es\nprecision highp float;\n");
    s.push_str(SNOISE_GLSL);
    s.push_str(HARMONY_COLORS_GLSL);
    s.push_str(r#"
uniform float u_time;
uniform float u_psi;  // consciousness level [0, 1]
uniform sampler2D u_earth_texture;
uniform sampler2D u_boundaries_texture;
uniform sampler2D u_nightlights_texture;
uniform vec4 u_phi_hotspots[12];  // xyz=position, w=intensity
uniform vec3 u_sun_direction;     // direction toward sun (normalized)
uniform float u_show_core;        // 0.0 = normal, 1.0 = cutaway view

in vec3 v_normal;
in vec2 v_uv;
in vec3 v_view_position;
in vec3 v_world_normal;
in float v_elevation;

out vec4 frag_color;

void main() {
    float PI = 3.14159265;

    // ═══════════════════════════════════════════════════════════
    // PLANETARY CORE CUTAWAY — clean hemisphere slice
    // ═══════════════════════════════════════════════════════════
    if (u_show_core > 0.5) {
        // Clean vertical cut: discard the front half (z > 0)
        if (v_world_normal.z > 0.01) {
            // On the cut face: render geological cross-section
            // Use distance from center (y and x) as radial depth
            vec3 n = normalize(v_world_normal);
            float r = length(n.xy); // radial distance on the cut face

            // Layer colors (geological cross-section, inside → outside)
            vec3 layer_color;
            if (r < 0.19) {
                // Inner core: bright golden (solid iron crystal)
                layer_color = vec3(1.0, 0.85, 0.4);
            } else if (r < 0.54) {
                // Outer core: orange (liquid iron)
                float t = (r - 0.19) / 0.35;
                layer_color = mix(vec3(1.0, 0.7, 0.25), vec3(0.85, 0.4, 0.1), t);
                // Animated convection
                float conv = snoise(vec3(n.xy * 5.0, u_time * 0.2)) * 0.1;
                layer_color += vec3(conv, conv * 0.3, 0.0);
            } else if (r < 0.89) {
                // Mantle: dark red to brown
                float t = (r - 0.54) / 0.35;
                layer_color = mix(vec3(0.6, 0.2, 0.05), vec3(0.3, 0.1, 0.03), t);
                // Slow convection cells
                float conv = snoise(vec3(n.xy * 3.0, u_time * 0.1)) * 0.08;
                layer_color += vec3(conv, conv * 0.2, 0.0);
            } else {
                // Crust: thin dark shell
                layer_color = vec3(0.15, 0.1, 0.06);
            }

            // Sharp boundary lines between layers
            float b1 = 1.0 - smoothstep(0.0, 0.008, abs(r - 0.19));
            float b2 = 1.0 - smoothstep(0.0, 0.008, abs(r - 0.54));
            float b3 = 1.0 - smoothstep(0.0, 0.008, abs(r - 0.89));
            layer_color += vec3(1.0, 0.9, 0.5) * (b1 + b2 + b3) * 0.4;

            // Self-luminous: brighter toward center
            float glow = 0.6 + (1.0 - r) * 0.6;
            frag_color = vec4(layer_color * glow, 1.0);
            return;
        }
    }

    vec4 tex_color = texture(u_earth_texture, v_uv);

    // ═══════════════════════════════════════════════════════════
    // FIX 1: Hard binary land/ocean mask (no muddy blending)
    // ═══════════════════════════════════════════════════════════
    float blue_dominance = tex_color.b - ((tex_color.r + tex_color.g) * 0.5);
    float brightness = (tex_color.r + tex_color.g + tex_color.b) / 3.0;

    // Hard mask: land vs ocean
    float is_land = 1.0 - smoothstep(0.04, 0.10, blue_dominance);
    // Dark regions: only classify as ocean if ALSO blue-dominant
    // This preserves dark forests (Congo, Amazon) which are dark but NOT blue
    float dark_ocean = smoothstep(0.12, 0.05, brightness) * smoothstep(0.0, 0.04, blue_dominance);
    is_land *= (1.0 - dark_ocean);
    // Rescue: any region with green > blue is land (forests)
    float green_rescue = step(tex_color.b, tex_color.g + 0.02);
    is_land = max(is_land, green_rescue * (1.0 - smoothstep(0.03, 0.08, blue_dominance)));

    // Coastline detection: narrow band at land/ocean boundary
    float coast_band = smoothstep(0.02, 0.08, blue_dominance) - smoothstep(0.08, 0.18, blue_dominance);

    // ═══════════════════════════════════════════════════════════
    // LAND: Terrain-varied holographic surface from texture data
    // ═══════════════════════════════════════════════════════════
    // Use actual texture RGB to derive terrain type
    float green_dominance = tex_color.g - (tex_color.r + tex_color.b) * 0.5;
    float red_channel = tex_color.r;
    float green_channel = tex_color.g;

    // Dense tropical forest: very high green, dark → deep emerald
    vec3 tropical_color = vec3(0.01, 0.15, 0.06);
    float is_tropical = smoothstep(0.05, 0.12, green_dominance) * (1.0 - smoothstep(0.35, 0.50, brightness));

    // Temperate forest: moderate green → sage green
    vec3 temperate_color = vec3(0.03, 0.18, 0.09);
    float is_temperate = smoothstep(0.02, 0.08, green_dominance) * smoothstep(0.20, 0.35, brightness);

    // Grassland/savanna: yellow-green
    vec3 grass_color = vec3(0.08, 0.14, 0.04);
    float is_grass = smoothstep(0.0, 0.03, green_dominance) * smoothstep(0.25, 0.40, green_channel)
                   * (1.0 - smoothstep(0.05, 0.10, green_dominance));

    // Desert/arid: high red, low green → warm amber-brown
    vec3 desert_color = vec3(0.14, 0.08, 0.03);
    float is_desert = smoothstep(0.22, 0.38, red_channel) * (1.0 - smoothstep(-0.02, 0.04, green_dominance));

    // Ice/snow/glacier: high brightness, low saturation → cool white-blue
    vec3 ice_color = vec3(0.22, 0.26, 0.30);
    float tex_sat = max(max(tex_color.r, tex_color.g), tex_color.b) - min(min(tex_color.r, tex_color.g), tex_color.b);
    float is_ice = smoothstep(0.50, 0.65, brightness) * (1.0 - smoothstep(0.08, 0.18, tex_sat));

    // Urban areas: bright grey spots (high brightness, low saturation, not ice latitude)
    vec3 urban_color = vec3(0.18, 0.16, 0.14);
    float is_urban = smoothstep(0.45, 0.60, brightness) * (1.0 - smoothstep(0.10, 0.20, tex_sat))
                   * (1.0 - smoothstep(0.6, 0.8, abs(v_world_normal.y)));  // not near poles

    // Mountain/highland: moderate brightness, neutral → grey-green
    vec3 mountain_color = vec3(0.05, 0.08, 0.06);
    float is_mountain = smoothstep(0.30, 0.45, brightness) * (1.0 - is_desert) * (1.0 - is_ice) * (1.0 - is_urban);

    // Base land: dark, solid — holographic contrast
    vec3 land_color = vec3(0.03, 0.12, 0.08);

    // Layer terrain types (order matters — later overrides earlier)
    land_color = mix(land_color, grass_color, is_grass * 0.7);
    land_color = mix(land_color, temperate_color, is_temperate * 0.8);
    land_color = mix(land_color, tropical_color, is_tropical * 0.9);
    land_color = mix(land_color, desert_color, is_desert * 0.85);
    land_color = mix(land_color, mountain_color, is_mountain * 0.5);
    land_color = mix(land_color, ice_color, is_ice * 0.7);
    land_color = mix(land_color, urban_color, is_urban * 0.5);

    // ── Elevation from topology texture ──
    float elevation = texture(u_boundaries_texture, v_uv).r;
    // Low elevation (0-0.3): plains/valleys — slightly darker
    // Mid elevation (0.3-0.6): foothills — base color
    // High elevation (0.6-0.8): mountains — grey-brown highlights
    // Very high (0.8+): peaks — snow-capped brightness
    float is_highland = smoothstep(0.45, 0.65, elevation);
    float is_peak = smoothstep(0.70, 0.85, elevation);
    float is_valley = 1.0 - smoothstep(0.15, 0.35, elevation);

    // Highland: add grey-brown rock tones
    land_color = mix(land_color, vec3(0.12, 0.10, 0.08), is_highland * 0.4);
    // Peaks: snow/ice highlights
    land_color = mix(land_color, vec3(0.40, 0.42, 0.45), is_peak * 0.5);
    // Valleys: slightly richer/darker
    land_color *= (1.0 - is_valley * 0.15);

    // Micro-detail from texture luminance
    float micro = (brightness - 0.3) * 0.10;
    land_color += vec3(micro * 0.3, micro * 0.5, micro * 0.2);

    // Blend with actual Blue Marble texture for realism (40% real, 60% procedural)
    vec3 real_land = tex_color.rgb * 0.6;  // darken the real texture
    land_color = mix(land_color, real_land, 0.65);

    // Sacred Gold coastline glow
    land_color += vec3(1.0, 0.84, 0.0) * coast_band * 0.3;

    // ═══════════════════════════════════════════════════════════
    // OCEAN: Depth-varied with coral reef detection
    // ═══════════════════════════════════════════════════════════
    vec3 ocean_color = vec3(0.01, 0.008, 0.04);  // very deep indigo

    // Coral reefs & shallow water: turquoise in texture (high G+B, moderate brightness)
    float shallow = smoothstep(0.12, 0.20, brightness) * smoothstep(0.03, 0.08, blue_dominance);
    float is_coral = shallow * (1.0 - is_land);
    // Coral: vibrant turquoise glow
    ocean_color = mix(ocean_color, vec3(0.0, 0.20, 0.18), is_coral * 0.8);

    // Deep ocean: darker with depth (lower brightness = deeper)
    float depth_factor = 1.0 - smoothstep(0.05, 0.15, brightness);
    ocean_color *= (0.6 + depth_factor * 0.4);

    // Subtle animated caustics (stronger near reefs)
    float caustic = sin(v_uv.x * 50.0 + u_time * 0.6) * sin(v_uv.y * 45.0 - u_time * 0.4);
    caustic = pow(caustic * 0.5 + 0.5, 4.0);
    ocean_color += vec3(0.01, 0.04, 0.08) * caustic * (0.05 + is_coral * 0.15);

    // ═══════════════════════════════════════════════════════════
    // STRICT separation: no muddy mixing
    // ═══════════════════════════════════════════════════════════
    vec3 base = mix(ocean_color, land_color, is_land);

    // ═══════════════════════════════════════════════════════════
    // OBJECT-SPACE sacred geometry grid (no pole pinching)
    // Uses v_world_normal (XYZ on unit sphere) instead of UV
    // ═══════════════════════════════════════════════════════════
    vec3 p = normalize(v_world_normal);

    // 8-fold symmetry: project onto 4 great-circle planes
    // Each plane's distance function creates 2 opposing great circles = 8 total
    float line_width = 0.002;  // ultra-thin

    // 4 vertical great circles at 0°, 45°, 90°, 135° longitude
    float gc1 = abs(p.x);                                          // 0° / 180°
    float gc2 = abs(p.z);                                          // 90° / 270°
    float gc3 = abs(p.x * 0.7071 + p.z * 0.7071);                 // 45° / 225°
    float gc4 = abs(p.x * 0.7071 - p.z * 0.7071);                 // 135° / 315°

    // Horizontal parallels: 4 evenly spaced by Y
    float h1 = abs(p.y);                                           // equator
    float h2 = abs(p.y - 0.5) * abs(p.y + 0.5);                   // ±30°
    float h3 = abs(abs(p.y) - 0.707);                              // ±45°

    // ── Localized Φ warping: grid intensifies near high-consciousness sites ──
    float phi_influence = 0.0;
    for (int i = 0; i < 12; i++) {
        vec3 hotspot = u_phi_hotspots[i].xyz;
        float intensity = u_phi_hotspots[i].w;
        if (intensity > 0.01) {
            float d = distance(p, normalize(hotspot));
            // Gaussian influence: peaks at hotspot, falls off with distance
            phi_influence += intensity * exp(-d * d * 12.0);
        }
    }
    phi_influence = min(phi_influence, 1.0);

    // Grid widens and brightens near high-Φ regions
    float local_width = line_width * (1.0 + phi_influence * 3.0);

    // Convert distances to razor-thin lines (warped by Φ)
    float grid = 0.0;
    grid = max(grid, 1.0 - smoothstep(0.0, local_width, gc1));
    grid = max(grid, 1.0 - smoothstep(0.0, local_width, gc2));
    grid = max(grid, 1.0 - smoothstep(0.0, local_width, gc3));
    grid = max(grid, 1.0 - smoothstep(0.0, local_width, gc4));
    grid = max(grid, 1.0 - smoothstep(0.0, local_width, h1));
    grid = max(grid, 1.0 - smoothstep(0.0, local_width * 1.5, h3));

    // Phi glow: add golden emission near hotspots
    base += vec3(1.0, 0.84, 0.0) * phi_influence * 0.15;

    // Breathing modulation (8-second Sacred Stillness cycle)
    float breath = sin(u_time * 0.7854) * 0.15 + 0.85;
    grid *= breath;

    // Grid color cycles through Eight Harmonies, applied ON TOP
    vec3 grid_color = harmony_at_time(u_time) * 0.6;
    base += grid_color * grid * 0.2;

    // ═══════════════════════════════════════════════════════════
    // FIX 3: Tight Fresnel — edges only, not washing the center
    // ═══════════════════════════════════════════════════════════
    vec3 view_dir = normalize(v_view_position);
    float ndotv = max(dot(v_normal, view_dir), 0.0);
    float rim = 1.0 - ndotv;

    // Very tight Fresnel: only at extreme edges
    float rim_r = pow(rim, 4.5);
    float rim_g = pow(rim, 5.0);
    float rim_b = pow(rim, 5.5);
    vec3 iridescent = vec3(rim_r * 0.5, rim_g * 0.3, rim_b * 0.8);

    // View-angle hue shift — only at edges
    float angle_hue = dot(v_world_normal, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
    vec3 angle_color = harmony_at_time(u_time + angle_hue * 4.0);
    iridescent *= angle_color;

    base += iridescent * 0.25;

    // ── Day/Night solar illumination ──
    float sun_dot = dot(v_world_normal, u_sun_direction);
    float daylight = smoothstep(-0.1, 0.3, sun_dot); // smooth terminator

    // Day side: bright with sun illumination
    base *= 0.4 + daylight * 0.8;

    // Night side: real NASA city lights from night texture
    float night = 1.0 - daylight;
    if (night > 0.1) {
        vec3 night_tex = texture(u_nightlights_texture, v_uv).rgb;
        float light_intensity = (night_tex.r + night_tex.g + night_tex.b) / 3.0;
        // Warm golden city glow
        vec3 city_color = mix(
            vec3(1.0, 0.7, 0.3),  // warm sodium vapor
            vec3(0.9, 0.9, 1.0),  // cool LED white
            smoothstep(0.3, 0.7, light_intensity)
        );
        base += city_color * light_intensity * night * 1.5;
    }

    // Night ocean: faint bioluminescent blue glow
    if (night > 0.3 && is_land < 0.5) {
        base += vec3(0.01, 0.02, 0.05) * night * 0.5;
    }

    // Ocean specular: sun reflection on water (sun glint)
    if (is_land < 0.5 && daylight > 0.3) {
        vec3 half_vec = normalize(u_sun_direction + view_dir);
        float spec = pow(max(dot(v_world_normal, half_vec), 0.0), 64.0);
        base += vec3(1.0, 0.95, 0.85) * spec * 0.6 * daylight;
    }

    // Twilight band: subtle warm glow at terminator
    float terminator = smoothstep(-0.05, 0.05, sun_dot) * (1.0 - smoothstep(0.05, 0.15, sun_dot));
    base += vec3(0.3, 0.12, 0.05) * terminator * 0.15;

    // ── Consciousness modulation ──
    float psi_lum = pow(u_psi, 0.7);
    base *= 0.7 + psi_lum * 0.5;

    // Subtle warmth tint at high consciousness
    vec3 psi_tint = mix(vec3(0.77, 0.58, 0.42), vec3(0.91, 0.77, 0.28), u_psi);
    base += psi_tint * psi_lum * 0.05;

    // Land fully opaque, ocean semi-transparent (holographic)
    // When core view is on, make ocean much more transparent to see interior glow
    float ocean_alpha = u_show_core > 0.5 ? 0.3 : 0.75;
    float alpha = mix(ocean_alpha, 1.0, is_land);

    frag_color = vec4(base, alpha);
}
"#);
    s
}

// ─── Consciousness Atmosphere (Phase 2) ─────────────────────────

pub const ATMOSPHERE_INNER_VERT: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_projection;
uniform mat4 u_model_view;
uniform mat3 u_normal_matrix;

out vec3 v_normal;
out vec3 v_world_normal;

void main() {
    v_normal = normalize(u_normal_matrix * a_normal);
    v_world_normal = a_normal;
    gl_Position = u_projection * u_model_view * vec4(a_position, 1.0);
}
"#;

pub fn atmosphere_inner_frag() -> String {
    let mut s = String::from("#version 300 es\nprecision highp float;\n");
    s.push_str(HARMONY_COLORS_GLSL);
    s.push_str(r#"
uniform float u_time;

in vec3 v_normal;
in vec3 v_world_normal;

out vec4 frag_color;

void main() {
    float intensity = pow(0.65 - dot(v_normal, vec3(0.0, 0.0, 1.0)), 3.0);
    intensity = max(intensity, 0.0);

    // Cycle through Eight Harmonies over 8-second Sacred Stillness breathing
    vec3 harmony_color = harmony_at_time(u_time);
    vec3 atmosphere = harmony_color * intensity;

    // Breathing modulation (8-second cycle)
    float breath = sin(u_time * 0.7854) * 0.1 + 0.9;  // 2π/8 ≈ 0.7854
    atmosphere *= breath;

    frag_color = vec4(atmosphere, intensity * 0.5);
}
"#);
    s
}

pub fn atmosphere_outer_frag() -> String {
    let mut s = String::from("#version 300 es\nprecision highp float;\n");
    s.push_str(SNOISE_GLSL);
    s.push_str(HARMONY_COLORS_GLSL);
    s.push_str(r#"
uniform float u_time;

in vec3 v_normal;
in vec3 v_world_normal;

out vec4 frag_color;

void main() {
    float intensity = pow(0.65 - dot(v_normal, vec3(0.0, 0.0, 1.0)), 3.5);
    intensity = max(intensity, 0.0);

    // Base atmosphere with harmony cycling
    vec3 base_color = harmony_at_time(u_time + 2.0) * 0.5;

    // Aurora at poles — concentrated near Y extremes
    float polar = pow(abs(v_world_normal.y), 3.0);
    float aurora = snoise(vec3(v_world_normal.xz * 3.0, u_time * 0.2));
    aurora = smoothstep(0.2, 0.8, aurora) * polar;

    // Aurora color: vivid green-cyan shifting
    vec3 aurora_color = mix(
        vec3(0.0, 1.0, 0.53),   // Mycelix lime #00ff88
        vec3(0.0, 0.87, 1.0),   // Mycelix cyan #00ddff
        snoise(vec3(v_world_normal.xz * 2.0, u_time * 0.15)) * 0.5 + 0.5
    );

    vec3 atmosphere = base_color * intensity + aurora_color * aurora * 0.4;

    frag_color = vec4(atmosphere, intensity * 0.35 + aurora * 0.5);
}
"#);
    s
}

// ─── Organic Markers (Phase 4) ──────────────────────────────────

pub const MARKER_VERT: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;    // quad vertex
layout(location = 1) in vec3 i_position;    // instance: world position
layout(location = 2) in vec3 i_color;       // instance: marker color
layout(location = 3) in float i_size;       // instance: marker size
layout(location = 4) in float i_type;       // instance: 0=energy, 1=geothermal, 2=vault, 3=terra_lumina

uniform mat4 u_projection;
uniform mat4 u_model_view;
uniform vec3 u_camera_position;  // eye position for horizon culling

out vec3 v_color;
out vec2 v_local;
flat out float v_type;
out float v_visibility;

void main() {
    // ── GPU Horizon Culling ──
    // Hide markers on the far side of the globe
    vec3 marker_normal = normalize(i_position); // normal at marker (sphere surface)
    vec3 to_camera = normalize(u_camera_position - i_position);
    float horizon = dot(marker_normal, to_camera);

    // Markers behind the horizon: move to clip space discard position
    if (horizon < -0.05) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0); // behind near plane
        v_visibility = 0.0;
        v_color = i_color;
        v_local = vec2(0.0);
        v_type = i_type;
        return;
    }

    // ── Edge scaling: reduce marker size at steep viewing angles ──
    float edge_scale = smoothstep(-0.05, 0.3, horizon);

    vec4 center_clip = u_projection * u_model_view * vec4(i_position, 1.0);
    vec2 offset = a_position.xy * i_size * edge_scale;
    center_clip.xy += offset * center_clip.w;

    v_color = i_color;
    v_local = a_position.xy;
    v_type = i_type;
    v_visibility = edge_scale;
    gl_Position = center_clip;
}
"#;

pub const MARKER_FRAG: &str = r#"#version 300 es
precision highp float;

uniform float u_time;

in vec3 v_color;
in vec2 v_local;
in float v_visibility;
flat in float v_type;

out vec4 frag_color;

// SDF for regular octagon
float sdf_octagon(vec2 p, float r) {
    vec2 q = abs(p);
    float d = max(q.x + q.y, max(q.x, q.y) * 1.41421356) - r;
    return d;
}

void main() {
    float dist = length(v_local) * 2.0;
    float pulse = sin(u_time * 1.2) * 0.08 + 0.92;
    // Discard markers behind the horizon
    if (v_visibility < 0.01) discard;

    int marker_type = int(v_type + 0.5);

    vec3 color;
    float alpha;

    if (marker_type == 1) {
        // ── Geothermal: Bioluminescent 5-petal bloom ──
        float angle = atan(v_local.y, v_local.x);
        float petal = 0.35 + 0.12 * sin(angle * 5.0 + u_time * 0.8);
        float r = length(v_local) * 2.0;
        float sdf = r - petal;
        float shape = 1.0 - smoothstep(-0.08, 0.02, sdf);
        float inner = 1.0 - smoothstep(0.0, 0.15, r);
        color = v_color * (shape * 1.5 + inner * 2.0) * pulse;
        alpha = shape + inner * 0.5;

    } else if (marker_type == 2) {
        // ── Resontia Vault: Octagonal sacred geometry ──
        float oct_outer = sdf_octagon(v_local, 0.38);
        float oct_inner = sdf_octagon(v_local, 0.38 / 1.618);  // golden ratio
        float oct_core = sdf_octagon(v_local, 0.38 / (1.618 * 1.618));

        float ring_outer = 1.0 - smoothstep(-0.02, 0.01, oct_outer);
        float ring_inner = (1.0 - smoothstep(-0.015, 0.01, oct_inner)) * 0.6;
        float core = 1.0 - smoothstep(-0.01, 0.01, oct_core);

        float shape = ring_outer + ring_inner + core * 1.5;
        color = v_color * shape * pulse;
        // Subtle rotation glow
        float rot_angle = atan(v_local.y, v_local.x) + u_time * 0.3;
        float spoke = pow(abs(sin(rot_angle * 4.0)), 8.0) * ring_outer * 0.3;
        color += v_color * spoke;
        alpha = clamp(shape + spoke, 0.0, 1.0);

    } else if (marker_type == 3) {
        // ── Terra Lumina: Crystalline hexagonal prism ──
        // Hex SDF
        vec2 q = abs(v_local);
        float hex = max(q.x * 0.866 + q.y * 0.5, q.y) * 2.0;
        float shape = 1.0 - smoothstep(0.6, 0.65, hex);
        // Internal refraction glow
        vec2 shifted = v_local + vec2(sin(u_time * 0.5) * 0.05, cos(u_time * 0.7) * 0.05);
        float inner_hex = max(abs(shifted.x) * 0.866 + abs(shifted.y) * 0.5, abs(shifted.y)) * 2.0;
        float refraction = (1.0 - smoothstep(0.3, 0.35, inner_hex)) * 0.8;
        // Prismatic color shift
        vec3 prism = v_color;
        prism.r *= 1.0 + sin(u_time * 1.5) * 0.2;
        prism.b *= 1.0 + cos(u_time * 1.5) * 0.2;
        color = prism * (shape * 1.2 + refraction * 1.5) * pulse;
        alpha = shape + refraction * 0.5;

    } else {
        // ── Energy: Original core+ring+halo ──
        float core = 1.0 - smoothstep(0.55, 0.65, dist);
        float ring = smoothstep(0.60, 0.70, dist) * (1.0 - smoothstep(0.85, 1.0, dist));
        float halo = (1.0 - smoothstep(0.85, 1.0, dist)) * 0.3;
        color = v_color * 1.3 * pulse * core
              + v_color * 1.8 * pulse * ring
              + v_color * 0.5 * halo;
        alpha = core + ring * 0.8 + halo;
    }

    if (alpha < 0.01) discard;
    frag_color = vec4(color, alpha);
}
"#;

// ─── Mycelial Arc (Phase 3) ─────────────────────────────────────

pub const ARC_VERT: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in float a_progress;

uniform mat4 u_projection;
uniform mat4 u_model_view;

out float v_progress;
out float v_depth;  // camera distance for LOD

void main() {
    v_progress = a_progress;
    vec4 mv_pos = u_model_view * vec4(a_position, 1.0);
    v_depth = -mv_pos.z;  // positive depth = distance from camera
    gl_Position = u_projection * mv_pos;
}
"#;

pub const ARC_FRAG: &str = r#"#version 300 es
precision highp float;

uniform float u_time;
uniform vec3 u_color;
uniform float u_flow_offset;

in float v_progress;
in float v_depth;

out vec4 frag_color;

void main() {
    // ═══════════════════════════════════════════════════════════
    // LOD-based Volumetric Energy Conduit
    // ═══════════════════════════════════════════════════════════

    // Camera distance → LOD level (smooth blend)
    // LOD 0: > 5.0 depth = orbital (thin bright thread)
    // LOD 1: 2.5 - 5.0 = atmospheric (capillary tube with core)
    // LOD 2: < 2.5 = surface (accelerator ring structure)
    float lod_macro = smoothstep(4.0, 6.0, v_depth);     // 1.0 at far
    float lod_micro = 1.0 - smoothstep(2.0, 3.5, v_depth); // 1.0 at close

    // ── Mycelial multi-pulse flow (always present) ──
    float flow1 = exp(-8.0 * pow(fract(v_progress - u_time * 0.30 + u_flow_offset) - 0.5, 2.0));
    float flow2 = exp(-8.0 * pow(fract(v_progress - u_time * 0.20 + u_flow_offset + 0.33) - 0.5, 2.0));
    float flow3 = exp(-8.0 * pow(fract(v_progress - u_time * 0.15 + u_flow_offset + 0.66) - 0.5, 2.0));
    float flow = max(max(flow1, flow2), flow3);

    // ── Accelerator ring structure (LOD 2: close up) ──
    // Mathematical rings: 80% transparent, 20% visible ring segments
    float ring_freq = 40.0;  // number of rings along corridor
    float ring_phase = fract(v_progress * ring_freq);
    float ring = smoothstep(0.0, 0.08, ring_phase) * (1.0 - smoothstep(0.15, 0.23, ring_phase));
    // High-velocity train pulse through ring centers
    float train = exp(-12.0 * pow(fract(v_progress - u_time * 0.5 + u_flow_offset * 0.3) - 0.5, 2.0));
    float ring_structure = ring * 0.8 + train * 1.5;

    // ── Capillary tube glow (LOD 1: mid-range) ──
    float capillary_core = flow * 1.0 + 0.2;  // concentrated inner core
    float capillary_hull = 0.08;  // faint outer hull always visible

    // ── Blend LOD layers ──
    float structure;
    if (lod_micro > 0.5) {
        // Close: show ring structure + train
        structure = mix(capillary_core, ring_structure, lod_micro);
    } else if (lod_macro > 0.5) {
        // Far: intense thin thread
        structure = (flow * 1.5 + 0.3) * (1.0 + lod_macro * 0.5);
    } else {
        // Mid: capillary tube
        structure = capillary_core + capillary_hull;
    }

    // ── Endpoint fade ──
    float start_fade = smoothstep(0.0, 0.08, v_progress);
    float end_fade = smoothstep(1.0, 0.92, v_progress);
    float fade = start_fade * end_fade;

    // ── Color: base with cyan pulse highlights ──
    vec3 pulse_color = mix(u_color, vec3(0.0, 0.87, 1.0), flow * 0.5);
    // Ring structure gets brighter white-cyan
    vec3 ring_color = mix(pulse_color, vec3(0.6, 0.95, 1.0), ring * lod_micro);
    // Train gets intense white
    ring_color += vec3(1.0, 1.0, 1.0) * train * lod_micro * 0.5;

    vec3 final_color = mix(pulse_color, ring_color, lod_micro) * structure;
    float alpha = fade * min(structure, 1.0);

    if (alpha < 0.01) discard;
    frag_color = vec4(final_color, alpha);
}
"#;

// ─── Starfield (point sprites, kept for depth) ──────────────────

pub const STAR_VERT: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;       // spectral color
layout(location = 2) in float a_brightness; // magnitude

uniform mat4 u_projection;
uniform mat4 u_model_view;
uniform float u_time;

out vec3 v_color;
out float v_twinkle;
out float v_brightness;

void main() {
    gl_Position = u_projection * u_model_view * vec4(a_position, 1.0);

    // Atmospheric twinkle (scintillation)
    float phase = dot(a_position, vec3(7.3, 13.7, 19.1));
    float twinkle_speed = 0.5 + fract(phase * 0.1) * 1.5; // varied twinkle rates
    v_twinkle = sin(u_time * twinkle_speed + phase) * 0.25 + 0.75;

    // Point size from magnitude: bright stars are larger
    gl_PointSize = 0.5 + a_brightness * 2.5;

    v_color = a_color;
    v_brightness = a_brightness;
}
"#;

pub const STAR_FRAG: &str = r#"#version 300 es
precision highp float;

in vec3 v_color;
in float v_twinkle;
in float v_brightness;

out vec4 frag_color;

void main() {
    float dist = length(gl_PointCoord - vec2(0.5));

    // Bright stars get a soft glow halo, dim stars are sharp points
    float core = 1.0 - smoothstep(0.0, 0.2, dist);
    float halo = (1.0 - smoothstep(0.1, 0.5, dist)) * v_brightness * 0.3;

    float intensity = (core + halo) * v_twinkle * v_brightness;
    vec3 color = v_color * intensity;

    // Bright stars have a slight diffraction cross (4-point star)
    if (v_brightness > 0.7) {
        vec2 p = gl_PointCoord - vec2(0.5);
        float cross_h = exp(-abs(p.y) * 20.0) * exp(-abs(p.x) * 5.0);
        float cross_v = exp(-abs(p.x) * 20.0) * exp(-abs(p.y) * 5.0);
        float diffraction = (cross_h + cross_v) * 0.15 * v_brightness;
        color += v_color * diffraction;
    }

    float alpha = core + halo;
    if (alpha < 0.01) discard;
    frag_color = vec4(color, min(alpha, 1.0));
}
"#;

// ─── Sacred Geometry Background (fullscreen quad) ────────────────

pub fn sacred_background_frag() -> String {
    let mut s = String::from("#version 300 es\nprecision highp float;\n");
    s.push_str(HARMONY_COLORS_GLSL);
    s.push_str(r#"
uniform float u_time;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    // Map UV from [0,1] to centered coordinates
    vec2 p = (v_uv - 0.5) * 2.0;
    float r = length(p);

    // ── Seed of Life: 7 overlapping circles ──
    float circle_r = 0.35;
    float d0 = abs(length(p) - circle_r);
    float d1 = abs(length(p - vec2(circle_r, 0.0)) - circle_r);
    float d2 = abs(length(p - vec2(-circle_r, 0.0)) - circle_r);
    float d3 = abs(length(p - vec2(circle_r * 0.5, circle_r * 0.866)) - circle_r);
    float d4 = abs(length(p - vec2(-circle_r * 0.5, circle_r * 0.866)) - circle_r);
    float d5 = abs(length(p - vec2(circle_r * 0.5, -circle_r * 0.866)) - circle_r);
    float d6 = abs(length(p - vec2(-circle_r * 0.5, -circle_r * 0.866)) - circle_r);

    float seed = min(min(min(d0, d1), min(d2, d3)), min(min(d4, d5), d6));
    float line = smoothstep(0.008, 0.0, seed);

    // ── 8-spoke mandala rays ──
    float angle = atan(p.y, p.x);
    float spoke = abs(sin(angle * 4.0));  // 8 spokes
    float spoke_line = smoothstep(0.995, 1.0, spoke) * smoothstep(0.08, 0.15, r);
    // Fade spokes at large radius
    spoke_line *= 1.0 - smoothstep(0.6, 0.9, r);

    // ── Outer ring ──
    float outer_ring = abs(r - 0.75);
    float ring_line = smoothstep(0.006, 0.0, outer_ring);

    float pattern = max(max(line, spoke_line * 0.5), ring_line * 0.4);

    // Color: subtle, cycling through harmonies
    vec3 color = harmony_at_time(u_time * 0.5) * 0.15;
    // Deep background
    vec3 bg = vec3(0.008, 0.005, 0.02);

    // ── Milky Way nebular glow ──
    // Tilted band across the screen (matching star concentration)
    float galactic_y = p.x * 0.5 + p.y * 0.866; // ~60° tilt
    float milky_band = exp(-galactic_y * galactic_y * 8.0); // Gaussian band
    // Nebular structure: wispy variation
    float nebula_detail = sin(p.x * 12.0 + p.y * 8.0) * 0.3
                        + sin(p.x * 5.0 - p.y * 15.0) * 0.2 + 0.5;
    vec3 milky_color = vec3(0.06, 0.05, 0.08) * milky_band * nebula_detail;
    // Warm core of Milky Way
    float core_glow = exp(-length(p - vec2(-0.2, -0.15)) * 3.0);
    milky_color += vec3(0.08, 0.05, 0.02) * core_glow;
    bg += milky_color;

    // Radial vignette
    float vignette = 1.0 - smoothstep(0.3, 1.0, r);

    vec3 final_color = bg + color * pattern * vignette;

    frag_color = vec4(final_color, 1.0);
}
"#);
    s
}

// ─── Celestial Bodies (point sprites for planets/sun) ────────────

pub const CELESTIAL_VERT: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;  // pre-scaled position
layout(location = 1) in vec3 a_color;
layout(location = 2) in float a_size;

uniform mat4 u_projection;
uniform mat4 u_model_view;
uniform float u_time;

out vec3 v_color;
out float v_glow;

void main() {
    gl_Position = u_projection * u_model_view * vec4(a_position, 1.0);
    // Subtle flicker
    float phase = dot(a_position, vec3(7.0, 13.0, 19.0));
    v_glow = sin(u_time * 1.5 + phase) * 0.15 + 0.85;
    gl_PointSize = a_size * v_glow;
    v_color = a_color;
}
"#;

pub const CELESTIAL_FRAG: &str = r#"#version 300 es
precision highp float;

in vec3 v_color;
in float v_glow;
out vec4 frag_color;

void main() {
    float dist = length(gl_PointCoord - vec2(0.5));
    // Soft radial glow
    float core = 1.0 - smoothstep(0.0, 0.2, dist);
    float halo = (1.0 - smoothstep(0.15, 0.5, dist)) * 0.4;
    float alpha = core + halo;
    vec3 color = v_color * (core * 0.8 + halo * 0.5) * v_glow;
    if (alpha < 0.01) discard;
    frag_color = vec4(color, alpha);
}
"#;

// ─── Cloud Layer ─────────────────────────────────────────────────

pub const CLOUD_VERT: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4 u_projection;
uniform mat4 u_model_view;
uniform float u_time;

out vec2 v_uv;
out vec3 v_normal;
out vec3 v_view_position;

void main() {
    // Slowly rotate clouds independently of earth
    float cloud_rotation = u_time * 0.002; // very slow drift, barely perceptible
    float c = cos(cloud_rotation);
    float s = sin(cloud_rotation);
    vec3 rotated = vec3(
        a_position.x * c - a_position.z * s,
        a_position.y,
        a_position.x * s + a_position.z * c
    );
    v_uv = a_uv;
    vec4 mv_pos = u_model_view * vec4(rotated, 1.0);
    v_view_position = -mv_pos.xyz;
    v_normal = a_normal;
    gl_Position = u_projection * mv_pos;
}
"#;

pub const CLOUD_FRAG: &str = r#"#version 300 es
precision highp float;

uniform sampler2D u_cloud_texture;

in vec2 v_uv;
in vec3 v_normal;
in vec3 v_view_position;

out vec4 frag_color;

void main() {
    float cloud = texture(u_cloud_texture, v_uv).r;
    // Threshold: only show dense clouds
    float alpha = smoothstep(0.3, 0.7, cloud) * 0.45;
    // Slightly brighter at edges (limb brightening)
    vec3 view_dir = normalize(v_view_position);
    float rim = 1.0 - max(dot(normalize(v_normal), view_dir), 0.0);
    alpha *= (1.0 + rim * 0.3);
    // White clouds with slight blue tint
    vec3 cloud_color = vec3(0.85, 0.88, 0.92);
    if (alpha < 0.01) discard;
    frag_color = vec4(cloud_color, alpha);
}
"#;

// ─── Celestial Body Sphere (Sun/Moon) ────────────────────────────

pub const CELESTIAL_SPHERE_VERT: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4 u_projection;
uniform mat4 u_model_view;
uniform vec3 u_body_position;
uniform float u_body_radius;

out vec3 v_normal;
out vec3 v_view_position;
out vec2 v_uv;

void main() {
    vec3 world_pos = a_position * u_body_radius + u_body_position;
    vec4 mv_pos = u_model_view * vec4(world_pos, 1.0);
    v_view_position = -mv_pos.xyz;
    v_normal = a_normal;
    v_uv = a_uv;
    gl_Position = u_projection * mv_pos;
}
"#;

/// Photorealistic textured body shader — used for Moon and planets
pub const TEXTURED_BODY_FRAG: &str = r#"#version 300 es
precision highp float;

uniform sampler2D u_body_texture;
uniform vec3 u_sun_direction;
uniform float u_ambient;  // minimum light level

in vec3 v_normal;
in vec3 v_view_position;
in vec2 v_uv;

out vec4 frag_color;

void main() {
    vec3 n = normalize(v_normal);
    vec3 tex = texture(u_body_texture, v_uv).rgb;

    // Directional sunlight
    float diffuse = max(dot(n, u_sun_direction), 0.0);
    float light = u_ambient + diffuse * (1.0 - u_ambient);

    // Subtle rim highlight
    vec3 view_dir = normalize(v_view_position);
    float rim = 1.0 - max(dot(n, view_dir), 0.0);
    vec3 rim_color = vec3(0.1, 0.12, 0.15) * pow(rim, 4.0) * 0.2;

    vec3 color = tex * light + rim_color;
    frag_color = vec4(color, 1.0);
}
"#;

pub fn sun_frag() -> String {
    let mut s = String::from("#version 300 es\nprecision highp float;\n");
    s.push_str(SNOISE_GLSL);
    s.push_str(r#"
uniform float u_time;

in vec3 v_normal;
in vec3 v_view_position;

out vec4 frag_color;

void main() {
    vec3 view_dir = normalize(v_view_position);
    float ndotv = max(dot(v_normal, view_dir), 0.0);
    float rim = 1.0 - ndotv;

    // ═══ Solar surface: multi-octave convection cells ═══
    vec3 sp = v_normal;
    // Large convection cells (supergranulation)
    float cell1 = snoise(sp * 2.0 + vec3(u_time * 0.05, 0.0, 0.0));
    // Medium granulation
    float cell2 = snoise(sp * 5.0 + vec3(0.0, u_time * 0.08, u_time * 0.03));
    // Fine granulation
    float cell3 = snoise(sp * 12.0 - vec3(u_time * 0.04, u_time * 0.06, 0.0));
    float surface_detail = cell1 * 0.4 + cell2 * 0.35 + cell3 * 0.15 + 0.5;

    // ═══ Photosphere: blinding white-gold center ═══
    vec3 photosphere = mix(
        vec3(1.0, 0.80, 0.35),  // cooler cell boundary
        vec3(1.0, 0.97, 0.85),  // white-hot cell center
        surface_detail
    );
    // Intense brightness (drives bloom hard)
    photosphere *= 4.0;
    // Limb darkening: center 100%, edge ~40% (real solar physics)
    photosphere *= pow(ndotv, 0.3);

    // ═══ Sunspots: dark patches that drift ═══
    float spot_noise = snoise(sp * 1.5 + vec3(u_time * 0.01));
    float sunspot = smoothstep(0.6, 0.7, spot_noise) * 0.4;
    photosphere *= (1.0 - sunspot);

    // ═══ Chromosphere: thin red-orange layer at limb ═══
    float chromo_band = smoothstep(0.85, 0.95, rim) * (1.0 - smoothstep(0.95, 1.0, rim));
    vec3 chromosphere = vec3(1.0, 0.25, 0.05) * chromo_band * 3.0;

    // ═══ Inner corona: golden-white streamers ═══
    float corona_angle = atan(v_normal.y, v_normal.x);
    // Streamer structure (radial rays)
    float streamers = pow(abs(sin(corona_angle * 4.0 + u_time * 0.1)), 3.0);
    float corona_falloff = pow(rim, 2.0);
    vec3 inner_corona = vec3(1.0, 0.85, 0.5) * corona_falloff * (0.8 + streamers * 1.2) * 2.0;

    // ═══ Outer corona: faint red glow extending far ═══
    vec3 outer_corona = vec3(0.8, 0.2, 0.05) * pow(rim, 4.0) * 1.5;

    // ═══ Prominences: bright eruptions at random limb positions ═══
    float prom1 = pow(rim, 6.0) * pow(sin(corona_angle * 2.0 + u_time * 0.2) * 0.5 + 0.5, 6.0);
    float prom2 = pow(rim, 5.0) * pow(cos(corona_angle * 3.0 - u_time * 0.15) * 0.5 + 0.5, 5.0);
    vec3 prominences = vec3(1.0, 0.5, 0.15) * (prom1 + prom2) * 3.0;

    // ═══ Solar flare pulse ═══
    float flare_cycle = sin(u_time * 0.2) * 0.5 + 0.5;
    float flare_burst = pow(flare_cycle, 8.0) * 0.3;  // rare bright bursts

    // ═══ Combine all layers ═══
    vec3 color = photosphere + chromosphere + inner_corona + outer_corona + prominences;
    color *= (1.0 + flare_burst);

    frag_color = vec4(color, 1.0);
}
"#);
    s
}

pub fn moon_frag() -> String {
    let mut s = String::from("#version 300 es\nprecision highp float;\n");
    s.push_str(SNOISE_GLSL);
    s.push_str(r#"
uniform vec3 u_sun_direction;

in vec3 v_normal;
in vec3 v_view_position;

out vec4 frag_color;

// Crater: returns albedo multiplier (0.85 = slightly darker floor, 1.1 = bright rim)
float crater_mark(vec3 p, vec3 center, float radius) {
    float d = distance(p, center);
    // Floor: very slight darkening (0.85-1.0), never coal-black
    float floor_dark = 1.0 - 0.15 * smoothstep(radius * 0.8, 0.0, d);
    // Rim: bright raised ring
    float rim_bright = 1.0 + 0.2 * exp(-pow((d - radius) / (radius * 0.15), 2.0));
    return floor_dark * rim_bright;
}

void main() {
    vec3 n = normalize(v_normal);
    vec3 view_dir = normalize(v_view_position);

    // ═══ Mare/Highland: smooth distinct regions ═══
    float mare_field = snoise(n * 1.0);
    float nearside = dot(n, vec3(0.0, 0.0, 1.0)) * 0.25;
    float is_mare = smoothstep(0.15, 0.40, mare_field + nearside);

    // Moderate contrast (not extreme)
    vec3 highland = vec3(0.50, 0.48, 0.44);
    vec3 mare_col = vec3(0.28, 0.27, 0.25);
    vec3 surface = mix(highland, mare_col, is_mare);

    // ═══ Craters: subtle albedo marks, never coal-black ═══
    float crater_albedo = 1.0;
    crater_albedo *= crater_mark(n, normalize(vec3(0.2, 0.85, 0.1)), 0.20);
    crater_albedo *= crater_mark(n, normalize(vec3(-0.4, 0.4, 0.7)), 0.16);
    crater_albedo *= crater_mark(n, normalize(vec3(0.6, -0.3, 0.5)), 0.13);
    crater_albedo *= crater_mark(n, normalize(vec3(-0.1, -0.7, 0.5)), 0.11);
    crater_albedo *= crater_mark(n, normalize(vec3(0.3, 0.3, -0.8)), 0.18);
    crater_albedo *= crater_mark(n, normalize(vec3(-0.7, 0.2, -0.4)), 0.09);
    crater_albedo *= crater_mark(n, normalize(vec3(0.5, -0.6, -0.4)), 0.07);
    crater_albedo *= crater_mark(n, normalize(vec3(-0.3, 0.9, 0.0)), 0.06);

    surface *= clamp(crater_albedo, 0.75, 1.3);

    // ═══ Lighting: simple, clean ═══
    float diffuse = max(dot(n, u_sun_direction), 0.0);
    float term = smoothstep(-0.02, 0.03, dot(n, u_sun_direction));
    float light = 0.08 + diffuse * term * 0.92;

    // ═══ Earthshine on dark side ═══
    float earth_face = max(-dot(n, u_sun_direction), 0.0);
    vec3 earthshine = vec3(0.08, 0.11, 0.18) * earth_face * 0.15;

    vec3 color = surface * light + earthshine;

    frag_color = vec4(color, 1.0);
}
"#);
    s
}

// ─── Inner Core Sphere (rendered inside the globe) ───────────────

pub fn core_sphere_frag() -> String {
    let mut s = String::from("#version 300 es\nprecision highp float;\n");
    s.push_str(SNOISE_GLSL);
    s.push_str(r#"
uniform float u_time;

in vec3 v_normal;
in vec3 v_view_position;
in vec2 v_uv;

out vec4 frag_color;

void main() {
    vec3 n = normalize(v_normal);
    vec3 view_dir = normalize(v_view_position);
    float ndotv = max(dot(n, view_dir), 0.0);

    // ── Radial layer from surface position ──
    // Use latitude as proxy for depth visualization
    float lat = asin(n.y) / 1.5708; // -1 to 1
    float lon = atan(n.z, n.x);

    // ── Animated convection currents ──
    float conv1 = snoise(vec3(n.xy * 3.0, u_time * 0.15));
    float conv2 = snoise(vec3(n.yz * 4.0, u_time * 0.12));
    float convection = conv1 * 0.5 + conv2 * 0.3 + 0.5;

    // Flow lines: like lava lamp / magma movement
    float flow = sin(lat * 8.0 + convection * 4.0 + u_time * 0.3) * 0.5 + 0.5;
    float flow2 = sin(lon * 6.0 + convection * 3.0 - u_time * 0.2) * 0.5 + 0.5;

    // ── Color: hot mantle with convection cells ──
    vec3 hot = vec3(1.0, 0.6, 0.1);    // bright orange-yellow
    vec3 warm = vec3(0.7, 0.2, 0.05);  // dark red
    vec3 cool = vec3(0.3, 0.08, 0.02); // deep brown-red

    // Mix based on convection intensity
    vec3 color = mix(cool, warm, convection);
    color = mix(color, hot, flow * flow2 * convection);

    // ── Bright upwelling plumes ──
    float plume1 = exp(-8.0 * pow(length(n.xz - vec2(0.3, 0.5)), 2.0));
    float plume2 = exp(-6.0 * pow(length(n.xz - vec2(-0.4, -0.3)), 2.0));
    float plume3 = exp(-10.0 * pow(length(n.xz - vec2(0.1, -0.6)), 2.0));
    float plumes = plume1 + plume2 + plume3;
    color += vec3(1.0, 0.8, 0.3) * plumes * 0.5;

    // ── Emission glow (core is self-luminous) ──
    float emission = 1.5 + convection * 0.5;
    color *= emission;

    // ── Rim: slight edge glow ──
    float rim = 1.0 - ndotv;
    color += vec3(1.0, 0.5, 0.1) * pow(rim, 3.0) * 0.3;

    frag_color = vec4(color, 0.85);
}
"#);
    s
}

// ─── Solar Wind Particles ────────────────────────────────────────

pub const SOLAR_WIND_VERT: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in float a_angle;    // radial angle around Sun
layout(location = 1) in float a_speed;    // speed variation
layout(location = 2) in float a_offset;   // lifecycle offset

uniform mat4 u_projection;
uniform mat4 u_model_view;
uniform float u_time;
uniform vec3 u_sun_position;

out float v_alpha;
out vec3 v_color;

void main() {
    // Particle lifecycle: spiral outward from Sun
    float t = mod(u_time * a_speed + a_offset, 8.0); // 8-second cycle
    float progress = t / 8.0;

    // Spiral: angle increases with distance (Parker spiral)
    float dist = 1.0 + progress * 12.0; // expands outward from Sun
    float spiral_angle = a_angle + progress * 2.0; // spiral twist

    // Position relative to Sun
    vec3 pos = u_sun_position + vec3(
        cos(spiral_angle) * dist * 0.3,
        sin(a_angle * 3.0) * dist * 0.08,  // slight vertical spread
        sin(spiral_angle) * dist * 0.3
    );

    vec4 mv_pos = u_model_view * vec4(pos, 1.0);
    gl_Position = u_projection * mv_pos;

    // Fade: bright near Sun, fading with distance
    v_alpha = (1.0 - progress) * 0.4;
    gl_PointSize = 1.0 + (1.0 - progress) * 1.5;

    // Color: hot near Sun, cooler as it expands
    v_color = mix(vec3(1.0, 0.8, 0.3), vec3(0.3, 0.5, 0.8), progress);
}
"#;

pub const SOLAR_WIND_FRAG: &str = r#"#version 300 es
precision highp float;

in float v_alpha;
in vec3 v_color;
out vec4 frag_color;

void main() {
    float dist = length(gl_PointCoord - vec2(0.5));
    float soft = 1.0 - smoothstep(0.2, 0.5, dist);
    if (v_alpha * soft < 0.01) discard;
    frag_color = vec4(v_color, v_alpha * soft);
}
"#;

// ─── Sun Corona Glow (large transparent sphere) ─────────────────

pub const CORONA_FRAG: &str = r#"#version 300 es
precision highp float;

uniform float u_time;

in vec3 v_normal;
in vec3 v_view_position;

out vec4 frag_color;

void main() {
    vec3 view_dir = normalize(v_view_position);
    float ndotv = max(dot(v_normal, view_dir), 0.0);

    // Multi-layer volumetric corona
    // Layer 1: Bright core glow (exponential falloff from center)
    float core_glow = exp(-2.0 * (1.0 - ndotv));
    vec3 core_color = vec3(1.0, 0.92, 0.7) * core_glow * 2.0;

    // Layer 2: Extended warm corona
    float mid_glow = exp(-4.0 * (1.0 - ndotv));
    vec3 mid_color = vec3(1.0, 0.65, 0.25) * mid_glow * 0.8;

    // Layer 3: Faint outer halo
    float outer_glow = exp(-8.0 * (1.0 - ndotv));
    vec3 outer_color = vec3(0.7, 0.2, 0.05) * outer_glow * 0.4;

    // Radial streamer rays (god ray approximation)
    float angle = atan(v_normal.y, v_normal.x);
    float streamers = pow(abs(sin(angle * 8.0 + u_time * 0.06)), 6.0);
    float streamer_fade = exp(-3.0 * (1.0 - ndotv));
    vec3 ray_color = vec3(1.0, 0.85, 0.5) * streamers * streamer_fade * 0.5;

    vec3 color = core_color + mid_color + outer_color + ray_color;
    float alpha = core_glow * 0.6 + mid_glow * 0.3 + outer_glow * 0.1;

    if (alpha < 0.003) discard;
    frag_color = vec4(color, min(alpha, 0.95));
}
"#;

// ─── Consciousness Particle Field ────────────────────────────────

/// GPU-driven particles — lifecycle computed entirely in vertex shader.
/// Each particle has a spawn_time attribute; the shader computes age/fade.
pub const PARTICLE_VERT: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;   // position on shell
layout(location = 1) in float a_spawn_time; // offset for lifecycle phase

uniform mat4 u_projection;
uniform mat4 u_model_view;
uniform float u_time;
uniform float u_psi;

out float v_alpha;
out vec3 v_color;

void main() {
    // Lifecycle: 3-second cycle, offset by spawn_time
    float age = mod(u_time - a_spawn_time, 3.0);
    float life = age / 3.0;

    // Fade in (0-0.3), full (0.3-0.7), fade out (0.7-1.0)
    v_alpha = smoothstep(0.0, 0.3, life) * smoothstep(1.0, 0.7, life);
    v_alpha *= u_psi;  // particles dim with low consciousness

    // Drift outward slightly over lifetime
    float drift = 1.0 + life * 0.02;
    vec3 pos = a_position * drift;

    // Orbital drift: slow rotation
    float angle = u_time * 0.1 + a_spawn_time * 2.0;
    float c = cos(angle * 0.05);
    float s = sin(angle * 0.05);
    pos = vec3(pos.x * c - pos.z * s, pos.y, pos.x * s + pos.z * c);

    vec4 mv_pos = u_model_view * vec4(pos, 1.0);
    gl_Position = u_projection * mv_pos;

    // Size: 1-3 pixels based on psi
    gl_PointSize = 1.0 + u_psi * 2.0;

    // Color: gold at high psi, lavender at low
    vec3 gold = vec3(0.91, 0.77, 0.28);
    vec3 lavender = vec3(0.6, 0.5, 0.8);
    v_color = mix(lavender, gold, u_psi);
}
"#;

pub const PARTICLE_FRAG: &str = r#"#version 300 es
precision highp float;

in float v_alpha;
in vec3 v_color;
out vec4 frag_color;

void main() {
    float dist = length(gl_PointCoord - vec2(0.5));
    float soft = 1.0 - smoothstep(0.2, 0.5, dist);
    float alpha = v_alpha * soft;
    if (alpha < 0.01) discard;
    frag_color = vec4(v_color * 1.2, alpha * 0.6);
}
"#;

// ─── Bloom Post-Processing (Phase 6) ────────────────────────────

/// Shared fullscreen quad vertex shader for all post-process passes.
pub const FULLSCREEN_VERT: &str = r#"#version 300 es
precision highp float;

layout(location = 0) in vec2 a_position;

out vec2 v_uv;

void main() {
    v_uv = a_position * 0.5 + 0.5;  // [-1,1] → [0,1]
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"#;

/// Extract bright pixels above threshold for bloom.
pub const BLOOM_EXTRACT_FRAG: &str = r#"#version 300 es
precision highp float;

uniform sampler2D u_scene;
uniform float u_threshold;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 color = texture(u_scene, v_uv).rgb;
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float contribution = max(luminance - u_threshold, 0.0);
    // Soft knee: don't clip harshly
    contribution = contribution / (contribution + 1.0);
    frag_color = vec4(color * contribution, 1.0);
}
"#;

/// 9-tap Gaussian blur (one direction per pass).
pub const BLOOM_BLUR_FRAG: &str = r#"#version 300 es
precision highp float;

uniform sampler2D u_input;
uniform vec2 u_direction;  // (1/width, 0) or (0, 1/height)

in vec2 v_uv;
out vec4 frag_color;

void main() {
    // 9-tap Gaussian weights (sigma ≈ 4)
    float weights[5] = float[5](0.2270270, 0.1945946, 0.1216216, 0.0540541, 0.0162162);

    vec3 result = texture(u_input, v_uv).rgb * weights[0];
    for (int i = 1; i < 5; i++) {
        vec2 offset = u_direction * float(i);
        result += texture(u_input, v_uv + offset).rgb * weights[i];
        result += texture(u_input, v_uv - offset).rgb * weights[i];
    }

    frag_color = vec4(result, 1.0);
}
"#;

/// Composite scene + bloom.
pub const BLOOM_COMPOSITE_FRAG: &str = r#"#version 300 es
precision highp float;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_strength;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 bloom = texture(u_bloom, v_uv).rgb;
    vec3 result = scene + bloom * u_bloom_strength;
    // ACES filmic tone mapping (cinematic contrast)
    vec3 a = result * (result * 2.51 + 0.03);
    vec3 b = result * (result * 2.43 + 0.59) + 0.14;
    result = clamp(a / b, 0.0, 1.0);
    frag_color = vec4(result, 1.0);
}
"#;

// ─── Shader Compilation ─────────────────────────────────────────

pub fn compile_shader(
    gl: &WebGl2RenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = gl
        .create_shader(shader_type)
        .ok_or("Failed to create shader")?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);

    if gl
        .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        let log = gl
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| "Unknown error".to_string());
        gl.delete_shader(Some(&shader));
        Err(format!("Shader compilation failed: {log}"))
    }
}

pub fn compile_program(
    gl: &WebGl2RenderingContext,
    vert_src: &str,
    frag_src: &str,
) -> Result<WebGlProgram, String> {
    let vert = compile_shader(gl, WebGl2RenderingContext::VERTEX_SHADER, vert_src)?;
    let frag = compile_shader(gl, WebGl2RenderingContext::FRAGMENT_SHADER, frag_src)?;

    let program = gl.create_program().ok_or("Failed to create program")?;
    gl.attach_shader(&program, &vert);
    gl.attach_shader(&program, &frag);
    gl.link_program(&program);

    gl.delete_shader(Some(&vert));
    gl.delete_shader(Some(&frag));

    if gl
        .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        let log = gl
            .get_program_info_log(&program)
            .unwrap_or_else(|| "Unknown error".to_string());
        gl.delete_program(Some(&program));
        Err(format!("Program link failed: {log}"))
    }
}
