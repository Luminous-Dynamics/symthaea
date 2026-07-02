// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! WebGL HDC Holograph — High-dimensional vector visualizer.
//! Renders 16,384-dimensional vectors as holographic patterns to demonstrate 
//! mathematical robustness to noise.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{WebGl2RenderingContext, WebGlProgram, WebGlShader};

#[component]
pub fn HdcHolograph() -> impl IntoView {
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    
    // State: The 16,384-bit vector (128x128 grid)
    let (base_vector, _set_base_vector) = signal(generate_random_16k());
    let (noise_level, set_noise_level) = signal(0.0f32); // 0.0 to 1.0 (50% max entropy)
    
    // Derived: Similarity score (Hamming distance based)
    let similarity = Memo::new(move |_| {
        1.0 - (noise_level.get() * 0.5) // Max noise (1.0) flips 50% of bits, making it orthogonal
    });

    // WebGL Effect
    Effect::new(move |_| {
        if let Some(canvas) = canvas_ref.get() {
            let gl = canvas
                .get_context("webgl2")
                .unwrap()
                .unwrap()
                .dyn_into::<WebGl2RenderingContext>()
                .unwrap();

            let program = link_program(&gl, VERTEX_SHADER, FRAGMENT_SHADER).unwrap();
            gl.use_program(Some(&program));

            // Setup vertices (full screen quad)
            let vertices: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
            let buffer = gl.create_buffer().ok_or("failed to create buffer").unwrap();
            gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&buffer));
            unsafe {
                let view = js_sys::Float32Array::view(&vertices);
                gl.buffer_data_with_array_buffer_view(
                    WebGl2RenderingContext::ARRAY_BUFFER,
                    &view,
                    WebGl2RenderingContext::STATIC_DRAW,
                );
            }

            let vao = gl.create_vertex_array().ok_or("failed to create VAO").unwrap();
            gl.bind_vertex_array(Some(&vao));
            let pos_loc = gl.get_attrib_location(&program, "position") as u32;
            gl.enable_vertex_attrib_array(pos_loc);
            gl.vertex_attrib_pointer_with_i32(pos_loc, 2, WebGl2RenderingContext::FLOAT, false, 0, 0);

            // Upload vector as texture
            let texture = gl.create_texture().unwrap();
            gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
            
            // Texture parameters for sharp pixels
            gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_MIN_FILTER, WebGl2RenderingContext::NEAREST as i32);
            gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_MAG_FILTER, WebGl2RenderingContext::NEAREST as i32);

            // Render Loop (Simplified for Leptos reactive updates)
            let base = base_vector.get();
            let noise = noise_level.get();
            let mut display_data = base.clone();
            
            // Inject noise (flip bits)
            if noise > 0.0 {
                let flip_count = (16384.0 * noise * 0.5) as usize;
                for _ in 0..flip_count {
                    let idx = (rand::random::<f32>() * 16383.0) as usize;
                    display_data[idx] = !display_data[idx];
                }
            }

            // Convert bools to grayscale bytes
            let tex_bytes: Vec<u8> = display_data.iter().map(|&b| if b { 255 } else { 0 }).collect();
            
            gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                WebGl2RenderingContext::LUMINANCE as i32,
                128,
                128,
                0,
                WebGl2RenderingContext::LUMINANCE,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                Some(&tex_bytes),
            ).unwrap();

            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);
            gl.draw_arrays(WebGl2RenderingContext::TRIANGLE_STRIP, 0, 4);
        }
    });

    view! {
        <div class="hdc-holograph">
            <div class="holograph-container">
                <canvas 
                    node_ref=canvas_ref 
                    width="256" 
                    height="256" 
                    style="width: 100%; aspect-ratio: 1; image-rendering: pixelated; border: 1px solid var(--border); border-radius: 8px"
                ></canvas>
                <div class="holograph-overlay">
                    <span class="dimension-tag">"16,384 Dimensions"</span>
                </div>
            </div>

            <div class="holograph-controls">
                <div class="control-row">
                    <label>"Noise Injection (Entropy)"</label>
                    <input 
                        type="range" 
                        min="0" 
                        max="1" 
                        step="0.01" 
                        prop:value=move || noise_level.get()
                        on:input=move |ev| set_noise_level.set(event_target_value(&ev).parse().unwrap_or(0.0))
                    />
                </div>

                <div class="stats-grid">
                    <div class="stat-box">
                        <span class="stat-label">"Cosine Similarity"</span>
                        <span class="stat-value" style=move || format!("color: {}", if similarity.get() > 0.8 { "var(--success)" } else if similarity.get() > 0.5 { "var(--warning)" } else { "var(--error)" })>
                            {move || format!("{:.3}", similarity.get())}
                        </span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">"Hamming Distance"</span>
                        <span class="stat-value">{move || (noise_level.get() * 8192.0) as u32}</span>
                    </div>
                </div>

                <div class="intuition-box">
                    {move || if similarity.get() > 0.7 {
                        view! { <p>"Signal is intact. HDC can still recover the meaning perfectly."</p> }.into_any()
                    } else if similarity.get() > 0.5 {
                        view! { <p>"High noise, but pseudo-orthogonality ensures the signal survives."</p> }.into_any()
                    } else {
                        view! { <p>"Near orthogonality. The vector is now distinct from its original self."</p> }.into_any()
                    }}
                </div>
            </div>
        </div>
    }
}

fn generate_random_16k() -> Vec<bool> {
    let mut bits = Vec::with_capacity(16384);
    for _ in 0..16384 {
        bits.push(rand::random());
    }
    bits
}

const VERTEX_SHADER: &str = r#"#version 300 es
in vec2 position;
out vec2 v_texCoord;
void main() {
    v_texCoord = position * 0.5 + 0.5;
    v_texCoord.y = 1.0 - v_texCoord.y;
    gl_Position = vec4(position, 0.0, 1.0);
}
"#;

const FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;
in vec2 v_texCoord;
uniform sampler2D u_texture;
out vec4 outColor;
void main() {
    float val = texture(u_texture, v_texCoord).r;
    // Map binary states to a cool cyber-teal aesthetic
    vec3 color = mix(vec2(0.02, 0.05).xyx, vec3(0.0, 0.8, 0.8), val);
    outColor = vec4(color, 1.0);
}
"#;

fn compile_shader(gl: &WebGl2RenderingContext, shader_type: u32, source: &str) -> Result<WebGlShader, String> {
    let shader = gl.create_shader(shader_type).ok_or_else(|| String::from("Unable to create shader object"))?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);

    if gl.get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS).as_bool().unwrap_or(false) {
        Ok(shader)
    } else {
        Err(gl.get_shader_info_log(&shader).unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

fn link_program(gl: &WebGl2RenderingContext, vert_source: &str, frag_source: &str) -> Result<WebGlProgram, String> {
    let vert_shader = compile_shader(gl, WebGl2RenderingContext::VERTEX_SHADER, vert_source)?;
    let frag_shader = compile_shader(gl, WebGl2RenderingContext::FRAGMENT_SHADER, frag_source)?;

    let program = gl.create_program().ok_or_else(|| String::from("Unable to create shader object"))?;

    gl.attach_shader(&program, &vert_shader);
    gl.attach_shader(&program, &frag_shader);
    gl.link_program(&program);

    if gl.get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS).as_bool().unwrap_or(false) {
        Ok(program)
    } else {
        Err(gl.get_program_info_log(&program).unwrap_or_else(|| String::from("Unknown error linking program")))
    }
}
