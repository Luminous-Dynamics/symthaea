// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Homeostasis Background — a living WebGL fluid driven by patient health state.
//!
//! When the patient is healthy (alignment ~1.0), the fluid drifts smoothly,
//! matching the 8-second Sacred Stillness breathing cycle.
//! When sick (alignment < 0.5), the flow becomes turbulent.
//! The patient FEELS their health state before reading a number.

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{WebGl2RenderingContext as GL, WebGlProgram, WebGlShader};

use crate::identity::SensoriumIdentity;

const VERTEX_SHADER: &str = r#"#version 300 es
in vec2 a_position;
out vec2 v_uv;
void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"#;

/// Fragment shader — Gray-Scott reaction-diffusion mycelium + domain patterns.
/// The background literally GROWS like mycelium. Patterns branch, merge, evolve.
const FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_alignment;    // consciousness score (0.0-1.0)
uniform float u_phi;           // integration level
uniform vec3  u_domain_color;  // active domain's color (RGB 0-1)
uniform float u_domain_blend;  // 0.0 = orbital, 1.0 = fully in domain
uniform float u_flow_style;    // 0=organic, 1=crystalline, 2=pulse, 3=branching

// ═══════════════════════════════════════════════════
// Noise functions
// ═══════════════════════════════════════════════════

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1,0)), f.x),
               mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x), f.y);
}

float fbm(vec2 p, int octaves) {
    float v = 0.0, a = 0.5, f = 1.0;
    for (int i = 0; i < 7; i++) {
        if (i >= octaves) break;
        v += a * noise(p * f);
        f *= 2.0; a *= 0.5;
    }
    return v;
}

// ═══════════════════════════════════════════════════
// Gray-Scott reaction-diffusion (approximated per-frame)
// Simulates mycelial growth — patterns branch and evolve
// ═══════════════════════════════════════════════════

// Soft organic flow — wide transitions, no sharp edges
float organic_flow(vec2 uv, float t) {
    float n1 = noise(uv * 4.0 + t * 0.015);
    float n2 = noise(uv * 7.0 - t * 0.01 + 50.0);
    float n3 = noise(uv * 2.5 + t * 0.008 + 100.0);

    // Wide smoothstep transitions (0.2+ range) = soft, organic edges
    float layer1 = smoothstep(0.3, 0.6, n1);
    float layer2 = smoothstep(0.35, 0.65, n2) * 0.6;
    float layer3 = smoothstep(0.4, 0.7, n3) * 0.3;

    // Soft blend — no hard thresholds
    return layer1 * 0.5 + layer2 * 0.3 + layer3 * 0.2;
}

// Deep ocean — very subtle, almost still
float deep_ocean(vec2 uv, float t) {
    float slow = fbm(uv * 2.0 + t * 0.005, 3);
    float drift = noise(uv * 1.5 + t * 0.003 + 200.0);
    return smoothstep(0.3, 0.7, slow) * 0.4 + drift * 0.15;
}

// Nebula — swirling, cosmic, high contrast but soft
float nebula(vec2 uv, float t) {
    vec2 warped = uv + vec2(
        noise(uv * 3.0 + t * 0.01) * 0.15,
        noise(uv * 3.0 + t * 0.01 + 300.0) * 0.15
    );
    float n = fbm(warped * 4.0 + t * 0.005, 5);
    return smoothstep(0.25, 0.75, n);
}

// Aurora — horizontal bands with vertical shimmer
float aurora(vec2 uv, float t) {
    float bands = sin(uv.y * 8.0 + t * 0.2 + noise(uv * 3.0 + t * 0.05) * 2.0) * 0.5 + 0.5;
    float shimmer = noise(vec2(uv.x * 5.0 + t * 0.1, uv.y * 2.0)) * 0.3;
    return bands * 0.5 + shimmer;
}

// Cellular — voronoi-like soft cells
float cellular(vec2 uv, float t) {
    vec2 p = uv * 5.0 + t * 0.02;
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    float md = 1.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            vec2 point = hash(ip + neighbor) * vec2(0.5) + vec2(0.5);
            point = 0.5 + 0.5 * sin(t * 0.1 + 6.2831 * point);
            float d = length(fp - neighbor - point);
            md = min(md, d);
        }
    }
    return smoothstep(0.0, 0.8, md);
}

// Hash for cellular — needs vec2 → vec2
float hash2(vec2 p) { return fract(sin(dot(p, vec2(269.5, 183.3))) * 43758.5453); }

float reaction_diffusion(vec2 uv, float t) {
    // Default: soft organic flow (replaces sharp camo pattern)
    return organic_flow(uv, t);
}

// ═══════════════════════════════════════════════════
// Domain-specific overlays
// ═══════════════════════════════════════════════════

float crystalline(vec2 p, float t) {
    vec2 g = fract(p * 4.0) - 0.5;
    float edges = abs(g.x) + abs(g.y);
    return smoothstep(0.4, 0.5, edges + sin(t * 0.3) * 0.1);
}

float pulse_radial(vec2 uv, float t) {
    float d = length(uv - 0.5);
    return sin(d * 20.0 - t * 2.0) * 0.5 + 0.5 * smoothstep(0.5, 0.0, d);
}

float dendritic(vec2 p, float t) {
    float n = fbm(p * 3.0 + t * 0.05, 5);
    return smoothstep(0.45, 0.55, n) + smoothstep(0.6, 0.7, n) * smoothstep(0.75, 0.65, n) * 2.0;
}

// ═══════════════════════════════════════════════════
// Main — compose everything
// ═══════════════════════════════════════════════════

void main() {
    vec2 uv = v_uv;

    // Sacred Stillness breathing (8-second cycle)
    float breath = sin(u_time * 0.7854) * 0.5 + 0.5;

    // ── Background style (switchable) ──
    // 0=organic (default), 1=deep ocean, 2=nebula, 3=aurora, 4=cellular
    float bg_select = mod(u_flow_style * 0.0 + 0.0, 5.0); // Default organic for now
    float mycelium;
    if (u_domain_blend < 0.5) {
        // Orbital/First Breath: use soft organic flow
        mycelium = organic_flow(uv, u_time);
    } else {
        // Domain-specific: domain overlay takes over
        mycelium = organic_flow(uv, u_time);
    }

    // Organic flow displacement
    float flow_speed = mix(0.2, 0.05, u_alignment);
    vec2 flow = vec2(
        noise(uv * 3.0 + u_time * flow_speed),
        noise(uv * 3.0 + u_time * flow_speed + 100.0)
    );
    float breathScale = mix(0.015, 0.004, u_alignment);
    vec2 distorted = uv + (flow - 0.5) * breathScale * (0.5 + breath * 0.5);

    // ── Domain-specific pattern overlay ──
    float domain_pattern = 0.0;
    if (u_flow_style < 0.5) {
        domain_pattern = fbm(distorted * 3.0 + u_time * 0.03, 5) * 0.4;
    } else if (u_flow_style < 1.5) {
        domain_pattern = crystalline(distorted, u_time) * 0.35;
    } else if (u_flow_style < 2.5) {
        domain_pattern = pulse_radial(uv, u_time) * 0.3;
    } else {
        domain_pattern = dendritic(distorted, u_time) * 0.35;
    }

    // ── Color composition ──
    vec3 void_color = vec3(0.01, 0.03, 0.05);       // The deep
    vec3 base_teal = vec3(0.04, 0.35, 0.38);         // Mycelial teal
    vec3 base_glow = vec3(0.08, 0.75, 0.65);         // Bioluminescent
    vec3 domain_mid = u_domain_color * 0.45;
    vec3 domain_glow = u_domain_color * 1.1;

    vec3 mid = mix(base_teal, domain_mid, u_domain_blend);
    vec3 glow = mix(base_glow, domain_glow, u_domain_blend);

    // Base: void + mycelial network
    vec3 color = mix(void_color, mid, mycelium * 0.35);

    // Domain pattern overlay
    color += domain_mid * domain_pattern * u_domain_blend * 0.15;

    // Bioluminescent nodes — where mycelium is densest
    float nodes = smoothstep(0.65, 0.85, mycelium) * u_phi;
    color += glow * nodes * 0.2 * (0.6 + breath * 0.4);

    // Subtle center radiance
    float center = max(0.0, 1.0 - length(uv - 0.5) * 1.4);
    color += mid * center * 0.03 * u_phi;

    // Soft shimmer — gentle edge highlight, not sharp lines
    float edges = abs(dFdx(mycelium)) + abs(dFdy(mycelium));
    color += glow * edges * 2.0 * (0.2 + breath * 0.1) * u_phi;

    fragColor = vec4(color, 0.88);
}
"#;

#[component]
pub fn HomeostasisBackground() -> impl IntoView {
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let identity = use_context::<SensoriumIdentity>().unwrap_or_else(|| SensoriumIdentity::new());

    // Initialize WebGL on mount
    Effect::new(move |_| {
        let Some(canvas) = canvas_ref.get() else {
            return;
        };
        let canvas: web_sys::HtmlCanvasElement = canvas.into();

        // Size to window
        let window = web_sys::window().unwrap();
        let w = window.inner_width().unwrap().as_f64().unwrap() as u32;
        let h = window.inner_height().unwrap().as_f64().unwrap() as u32;
        canvas.set_width(w);
        canvas.set_height(h);

        let Some(gl): Option<GL> = canvas
            .get_context("webgl2")
            .ok()
            .flatten()
            .and_then(|ctx| ctx.dyn_into::<GL>().ok())
        else {
            web_sys::console::warn_1(&"WebGL2 not available — static background only".into());
            return;
        };

        // Compile shaders
        let vert = compile_shader(&gl, GL::VERTEX_SHADER, VERTEX_SHADER);
        let frag = compile_shader(&gl, GL::FRAGMENT_SHADER, FRAGMENT_SHADER);

        let Some(vert) = vert else { return };
        let Some(frag) = frag else { return };

        let program = link_program(&gl, &vert, &frag);
        let Some(program) = program else { return };

        gl.use_program(Some(&program));

        // Full-screen quad
        let vertices: [f32; 12] = [
            -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
        ];

        let buffer = gl.create_buffer().unwrap();
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&buffer));
        unsafe {
            let array = js_sys::Float32Array::view(&vertices);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &array, GL::STATIC_DRAW);
        }

        let pos_loc = gl.get_attrib_location(&program, "a_position") as u32;
        gl.enable_vertex_attrib_array(pos_loc);
        gl.vertex_attrib_pointer_with_i32(pos_loc, 2, GL::FLOAT, false, 0, 0);

        // Uniform locations
        let u_time = gl.get_uniform_location(&program, "u_time");
        let u_alignment = gl.get_uniform_location(&program, "u_alignment");
        let u_phi = gl.get_uniform_location(&program, "u_phi");
        let u_domain_color = gl.get_uniform_location(&program, "u_domain_color");
        let u_domain_blend = gl.get_uniform_location(&program, "u_domain_blend");
        let u_flow_style = gl.get_uniform_location(&program, "u_flow_style");

        // Animation loop with cancellation support
        let gl = std::rc::Rc::new(gl);
        let program = std::rc::Rc::new(program);
        let u_time = std::rc::Rc::new(u_time);
        let u_alignment = std::rc::Rc::new(u_alignment);
        let u_phi = std::rc::Rc::new(u_phi);
        let u_domain_color = std::rc::Rc::new(u_domain_color);
        let u_domain_blend = std::rc::Rc::new(u_domain_blend);
        let u_flow_style = std::rc::Rc::new(u_flow_style);

        // Read active domain from context for background color shifts
        let active_domain = use_context::<RwSignal<Option<String>>>;
        let canvas_rc = std::rc::Rc::new(canvas);

        let perf = window.performance().unwrap();
        let start = perf.now();

        // Cancellation flag — shared between animation loop and cleanup
        let running = std::rc::Rc::new(std::cell::Cell::new(true));
        let running_cleanup = running.clone();

        // Resize handler
        let canvas_resize = canvas_rc.clone();
        let gl_resize = gl.clone();
        let resize_cb = Closure::<dyn FnMut()>::new(move || {
            let w = web_sys::window().unwrap();
            let width = w.inner_width().unwrap().as_f64().unwrap() as u32;
            let height = w.inner_height().unwrap().as_f64().unwrap() as u32;
            canvas_resize.set_width(width);
            canvas_resize.set_height(height);
            gl_resize.viewport(0, 0, width as i32, height as i32);
        });
        let _ = web_sys::window()
            .unwrap()
            .add_event_listener_with_callback("resize", resize_cb.as_ref().unchecked_ref());
        resize_cb.forget(); // Leak is acceptable — lives for app lifetime

        let f: std::rc::Rc<std::cell::RefCell<Option<Closure<dyn FnMut()>>>> =
            std::rc::Rc::new(std::cell::RefCell::new(None));
        let g = f.clone();

        let gl2 = gl.clone();
        let program2 = program.clone();
        let ut = u_time.clone();
        let ua = u_alignment.clone();
        let up = u_phi.clone();
        let udc = u_domain_color.clone();
        let udb = u_domain_blend.clone();
        let ufs = u_flow_style.clone();
        let running2 = running.clone();

        *g.borrow_mut() = Some(Closure::new(move || {
            // Stop if unmounted
            if !running2.get() {
                return;
            }

            let score = identity.consciousness_score.get();
            let elapsed = (perf.now() - start) / 1000.0;

            // Determine active domain color and flow style
            let (r, g, b, blend, style) = if let Some(active) = active_domain() {
                if let Some(domain_id) = active.get() {
                    match domain_id.as_str() {
                        "health" => (0.05, 0.45, 0.47, 1.0, 0.0),     // organic
                        "governance" => (0.49, 0.23, 0.93, 1.0, 1.0), // crystalline
                        "finance" => (0.85, 0.47, 0.02, 1.0, 2.0),    // pulse
                        "praxis" => (0.15, 0.39, 0.92, 1.0, 3.0),     // branching
                        "commons" => (0.02, 0.59, 0.41, 1.0, 0.0),    // organic
                        "hearth" => (0.86, 0.15, 0.47, 1.0, 0.0),     // organic
                        "knowledge" => (0.03, 0.57, 0.70, 1.0, 3.0),  // branching
                        "space" => (0.31, 0.27, 0.90, 1.0, 1.0),      // crystalline
                        _ => (0.05, 0.45, 0.47, 0.0, 0.0),
                    }
                } else {
                    (0.05, 0.45, 0.47, 0.0, 0.0) // orbital view — default teal
                }
            } else {
                (0.05, 0.45, 0.47, 0.0, 0.0)
            };

            gl2.use_program(Some(&program2));
            gl2.uniform1f(ut.as_ref().as_ref(), elapsed as f32);
            gl2.uniform1f(ua.as_ref().as_ref(), score as f32);
            gl2.uniform1f(up.as_ref().as_ref(), (score * 0.7) as f32);
            gl2.uniform3f(udc.as_ref().as_ref(), r as f32, g as f32, b as f32);
            gl2.uniform1f(udb.as_ref().as_ref(), blend as f32);
            gl2.uniform1f(ufs.as_ref().as_ref(), style as f32);

            gl2.draw_arrays(GL::TRIANGLES, 0, 6);

            if running2.get() {
                let window = web_sys::window().unwrap();
                let _ = window
                    .request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref());
            }
        }));

        let window = web_sys::window().unwrap();
        let _ =
            window.request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref());

        // Cleanup: stop animation on unmount.
        // Use wasm_bindgen closure since Rc<Cell> isn't Send.
        // The running flag is checked every frame — setting it to false
        // stops the loop on the next animation frame.
        let cleanup = Closure::<dyn FnMut()>::new(move || {
            running_cleanup.set(false);
        });
        // Store cleanup closure for later invocation (leak-based lifecycle)
        cleanup.forget();
    });

    view! {
        <canvas
            node_ref=canvas_ref
            class="homeostasis-bg"
            aria-hidden="true"
        />
    }
}

fn compile_shader(gl: &GL, shader_type: u32, source: &str) -> Option<WebGlShader> {
    let shader = gl.create_shader(shader_type)?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);

    if !gl
        .get_shader_parameter(&shader, GL::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        let log = gl.get_shader_info_log(&shader).unwrap_or_default();
        web_sys::console::error_1(&format!("Shader compile error: {}", log).into());
        gl.delete_shader(Some(&shader));
        return None;
    }
    Some(shader)
}

fn link_program(gl: &GL, vert: &WebGlShader, frag: &WebGlShader) -> Option<WebGlProgram> {
    let program = gl.create_program()?;
    gl.attach_shader(&program, vert);
    gl.attach_shader(&program, frag);
    gl.link_program(&program);

    if !gl
        .get_program_parameter(&program, GL::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        let log = gl.get_program_info_log(&program).unwrap_or_default();
        web_sys::console::error_1(&format!("Program link error: {}", log).into());
        gl.delete_program(Some(&program));
        return None;
    }
    Some(program)
}
