// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{HtmlImageElement, WebGl2RenderingContext, WebGlTexture};

/// Create a 1x1 placeholder texture, then async-load the real image and swap it in.
/// Returns the texture handle immediately (placeholder until image loads).
pub fn load_texture(
    gl: &WebGl2RenderingContext,
    url: &str,
) -> Result<WebGlTexture, String> {
    let texture = gl.create_texture().ok_or("Failed to create texture")?;
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));

    // 1x1 dark blue placeholder
    let placeholder: [u8; 4] = [5, 15, 40, 255];
    gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        WebGl2RenderingContext::RGBA as i32,
        1,
        1,
        0,
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(&placeholder),
    )
    .map_err(|e| format!("tex_image_2d placeholder: {e:?}"))?;

    // Start async image load
    let image = HtmlImageElement::new().map_err(|e| format!("HtmlImageElement: {e:?}"))?;
    image.set_cross_origin(Some("anonymous"));

    let gl_clone = gl.clone();
    let texture_clone = texture.clone();
    let image_clone = image.clone();

    let onload = Closure::wrap(Box::new(move || {
        gl_clone.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture_clone));
        let _ = gl_clone.tex_image_2d_with_u32_and_u32_and_html_image_element(
            WebGl2RenderingContext::TEXTURE_2D,
            0,
            WebGl2RenderingContext::RGBA as i32,
            WebGl2RenderingContext::RGBA,
            WebGl2RenderingContext::UNSIGNED_BYTE,
            &image_clone,
        );
        gl_clone.generate_mipmap(WebGl2RenderingContext::TEXTURE_2D);
        gl_clone.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            WebGl2RenderingContext::LINEAR_MIPMAP_LINEAR as i32,
        );
        gl_clone.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MAG_FILTER,
            WebGl2RenderingContext::LINEAR as i32,
        );
        gl_clone.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_S,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        gl_clone.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_T,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        log::info!("Texture loaded: {}x{}", image_clone.width(), image_clone.height());
    }) as Box<dyn Fn()>);

    image.set_onload(Some(onload.as_ref().unchecked_ref()));
    onload.forget(); // leak closure — lives for app lifetime

    image.set_src(url);

    // Set initial texture params for placeholder
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_MIN_FILTER,
        WebGl2RenderingContext::NEAREST as i32,
    );
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_MAG_FILTER,
        WebGl2RenderingContext::NEAREST as i32,
    );

    Ok(texture)
}
