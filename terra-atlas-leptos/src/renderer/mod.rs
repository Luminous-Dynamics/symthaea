// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
pub mod camera;
pub mod geometry;
pub mod math;
pub mod picking;
pub mod shaders;
pub mod texture;

use wasm_bindgen::JsCast;

use camera::OrbitalCamera;
use math::Vec3;
use shaders::compile_program;
use web_sys::{
    WebGl2RenderingContext as GL, WebGlBuffer, WebGlFramebuffer, WebGlProgram,
    WebGlRenderbuffer, WebGlTexture, WebGlUniformLocation, WebGlVertexArrayObject,
};

/// Per-instance marker data sent to the GPU.
pub struct MarkerInstance {
    pub position: Vec3,
    pub color: Vec3,
    pub size: f32,
    pub marker_type: f32, // 0=energy, 1=geothermal, 2=vault, 3=terra_lumina
}

/// Arc data for a single corridor/trade route.
pub struct ArcData {
    pub vao: WebGlVertexArrayObject,
    pub vertex_count: i32,
    pub color: Vec3,
    pub flow_offset: f32,
}

struct Programs {
    earth: WebGlProgram,
    atmo_inner: WebGlProgram,
    atmo_outer: WebGlProgram,
    marker: WebGlProgram,
    arc: WebGlProgram,
    star: WebGlProgram,
    sacred_bg: WebGlProgram,
    cloud: WebGlProgram,
    textured_body: WebGlProgram,
    celestial: WebGlProgram,
    particle: WebGlProgram,
    core_sphere: WebGlProgram,
    solar_wind: WebGlProgram,
    corona: WebGlProgram,
    bloom_extract: WebGlProgram,
    bloom_blur: WebGlProgram,
    bloom_composite: WebGlProgram,
}

struct CelestialBody {
    texture: WebGlTexture,
    distance: f32,      // orbital distance from origin
    radius: f32,        // visual radius
    orbit_speed: f32,   // radians per second
    orbit_offset: f32,  // initial angle
    y_offset: f32,      // vertical offset
    ambient: f32,       // minimum light level (Sun = 1.0, others < 1.0)
    is_sun: bool,       // Sun uses emissive shader, not lit
}

struct BloomFbo {
    framebuffer: WebGlFramebuffer,
    color_texture: WebGlTexture,
    width: u32,
    height: u32,
}

pub struct GlobeRenderer {
    gl: GL,
    programs: Programs,

    // Geometry VAOs
    earth_vao: WebGlVertexArrayObject,
    earth_index_count: i32,
    atmo_inner_vao: WebGlVertexArrayObject,
    atmo_inner_index_count: i32,
    atmo_outer_vao: WebGlVertexArrayObject,
    atmo_outer_index_count: i32,
    star_vao: WebGlVertexArrayObject,
    star_count: i32,

    // Marker instanced rendering
    marker_quad_vao: Option<WebGlVertexArrayObject>,
    marker_instance_buffer: Option<WebGlBuffer>,
    marker_count: i32,

    // Arc VAOs
    arcs: Vec<ArcData>,

    // Textures
    earth_texture: WebGlTexture,
    bump_texture: WebGlTexture,
    cloud_texture: WebGlTexture,
    nightlights_texture: WebGlTexture,

    // Cloud layer
    cloud_vao: WebGlVertexArrayObject,
    cloud_index_count: i32,

    // Core sphere (inner mantle, radius 0.5)
    core_vao: WebGlVertexArrayObject,
    core_index_count: i32,

    // Camera
    pub camera: OrbitalCamera,

    // Celestial bodies
    celestial_vao: WebGlVertexArrayObject,
    celestial_count: i32,
    body_sphere_vao: WebGlVertexArrayObject,
    body_sphere_index_count: i32,
    bodies: Vec<CelestialBody>,

    // Consciousness particle field
    particle_vao: WebGlVertexArrayObject,
    particle_count: i32,

    // Solar wind particles
    solar_wind_vao: WebGlVertexArrayObject,
    solar_wind_count: i32,

    // Bloom post-processing
    scene_fbo: Option<BloomFbo>,
    bloom_fbo_a: Option<BloomFbo>,
    bloom_fbo_b: Option<BloomFbo>,
    fullscreen_vao: WebGlVertexArrayObject,
    last_width: u32,
    last_height: u32,

    // State
    time: f64,
    psi: f32,           // consciousness level [0, 1]
    psi_heartbeat: bool, // if true, psi pulses automatically
    pub show_core: bool, // cutaway view toggle

    // Localized Φ hotspots (up to 12): [x, y, z, intensity] per hotspot
    phi_hotspots: Vec<[f32; 4]>,
}

impl GlobeRenderer {
    pub fn init(gl: GL) -> Result<Self, String> {
        // Compile shader programs — earth/atmo use dynamic string builders
        let earth_frag_src = shaders::earth_frag();
        let atmo_inner_frag_src = shaders::atmosphere_inner_frag();
        let atmo_outer_frag_src = shaders::atmosphere_outer_frag();

        let sacred_bg_frag_src = shaders::sacred_background_frag();

        let programs = Programs {
            earth: compile_program(&gl, shaders::EARTH_VERT, &earth_frag_src)?,
            atmo_inner: compile_program(&gl, shaders::ATMOSPHERE_INNER_VERT, &atmo_inner_frag_src)?,
            atmo_outer: compile_program(&gl, shaders::ATMOSPHERE_INNER_VERT, &atmo_outer_frag_src)?,
            marker: compile_program(&gl, shaders::MARKER_VERT, shaders::MARKER_FRAG)?,
            arc: compile_program(&gl, shaders::ARC_VERT, shaders::ARC_FRAG)?,
            star: compile_program(&gl, shaders::STAR_VERT, shaders::STAR_FRAG)?,
            sacred_bg: compile_program(&gl, shaders::FULLSCREEN_VERT, &sacred_bg_frag_src)?,
            cloud: compile_program(&gl, shaders::CLOUD_VERT, shaders::CLOUD_FRAG)?,
            core_sphere: compile_program(&gl, shaders::EARTH_VERT, &shaders::core_sphere_frag())?,
            textured_body: compile_program(&gl, shaders::CELESTIAL_SPHERE_VERT, shaders::TEXTURED_BODY_FRAG)?,
            celestial: compile_program(&gl, shaders::CELESTIAL_VERT, shaders::CELESTIAL_FRAG)?,
            particle: compile_program(&gl, shaders::PARTICLE_VERT, shaders::PARTICLE_FRAG)?,
            solar_wind: compile_program(&gl, shaders::SOLAR_WIND_VERT, shaders::SOLAR_WIND_FRAG)?,
            corona: compile_program(&gl, shaders::CELESTIAL_SPHERE_VERT, shaders::CORONA_FRAG)?,
            bloom_extract: compile_program(&gl, shaders::FULLSCREEN_VERT, shaders::BLOOM_EXTRACT_FRAG)?,
            bloom_blur: compile_program(&gl, shaders::FULLSCREEN_VERT, shaders::BLOOM_BLUR_FRAG)?,
            bloom_composite: compile_program(&gl, shaders::FULLSCREEN_VERT, shaders::BLOOM_COMPOSITE_FRAG)?,
        };

        log::info!("All 15 shader programs compiled successfully");

        // Generate and upload geometry
        let (earth_vao, earth_index_count) = Self::create_sphere_vao(&gl, 128, 128, 1.0)?;
        let (atmo_inner_vao, atmo_inner_index_count) = Self::create_sphere_vao(&gl, 64, 64, 1.08)?;
        let (atmo_outer_vao, atmo_outer_index_count) = Self::create_sphere_vao(&gl, 64, 64, 1.12)?;
        let (star_vao, star_count) = Self::create_starfield_vao(&gl, 5000, 50.0)?;

        // Load textures
        let earth_texture = texture::load_texture(&gl, "assets/globe-textures/earth-blue-marble.jpg")?;
        let bump_texture = texture::load_texture(&gl, "assets/globe-textures/earth-topology.png")?;
        let cloud_texture = texture::load_texture(&gl, "assets/globe-textures/earth-clouds.jpg")?;
        let nightlights_texture = texture::load_texture(&gl, "assets/globe-textures/earth-nightlights.jpg")?;

        // Cloud sphere (radius 1.02, slightly above earth surface)
        let (cloud_vao, cloud_index_count) = Self::create_sphere_vao(&gl, 64, 64, 1.02)?;

        // Core sphere (radius 0.6, inside the globe — visible through translucent ocean)
        let (core_vao, core_index_count) = Self::create_sphere_vao(&gl, 32, 32, 0.6)?;

        // Fullscreen quad VAO for bloom/background passes
        let fullscreen_vao = Self::create_fullscreen_quad_vao(&gl)?;

        // Celestial bodies (Sun + 5 planets at logarithmic-scaled distances)
        let (celestial_vao, celestial_count) = Self::create_celestial_vao(&gl)?;

        // Consciousness particle field (GPU-driven lifecycle)
        let (particle_vao, particle_count) = Self::create_particle_vao(&gl, 2000)?;

        // Sphere mesh for celestial bodies (unit sphere, scaled by uniform)
        let (body_sphere_vao, body_sphere_index_count) = Self::create_sphere_vao(&gl, 32, 32, 1.0)?;

        // Load planet textures and create celestial bodies
        let bodies = vec![
            CelestialBody {
                texture: texture::load_texture(&gl, "assets/globe-textures/sun.jpg")?,
                distance: 20.0, radius: 0.6, orbit_speed: 0.02, orbit_offset: 0.0,
                y_offset: 2.0, ambient: 1.0, is_sun: true,
            },
            CelestialBody {
                texture: texture::load_texture(&gl, "assets/globe-textures/moon.jpg")?,
                distance: 3.5, radius: 0.15, orbit_speed: 0.05, orbit_offset: 0.0,
                y_offset: 0.0, ambient: 0.06, is_sun: false,
            },
            CelestialBody {
                texture: texture::load_texture(&gl, "assets/globe-textures/venus.jpg")?,
                distance: 8.0, radius: 0.12, orbit_speed: 0.015, orbit_offset: 1.2,
                y_offset: 0.5, ambient: 0.05, is_sun: false,
            },
            CelestialBody {
                texture: texture::load_texture(&gl, "assets/globe-textures/mars.jpg")?,
                distance: 12.0, radius: 0.10, orbit_speed: 0.01, orbit_offset: 2.5,
                y_offset: -0.8, ambient: 0.04, is_sun: false,
            },
            CelestialBody {
                texture: texture::load_texture(&gl, "assets/globe-textures/jupiter.jpg")?,
                distance: 25.0, radius: 0.35, orbit_speed: 0.005, orbit_offset: 4.0,
                y_offset: -1.5, ambient: 0.03, is_sun: false,
            },
            CelestialBody {
                texture: texture::load_texture(&gl, "assets/globe-textures/saturn.jpg")?,
                distance: 35.0, radius: 0.30, orbit_speed: 0.003, orbit_offset: 5.5,
                y_offset: 1.0, ambient: 0.03, is_sun: false,
            },
        ];

        // Solar wind particles
        let (solar_wind_vao, solar_wind_count) = Self::create_solar_wind_vao(&gl, 500)?;

        // GL state
        gl.enable(GL::DEPTH_TEST);
        gl.enable(GL::CULL_FACE);
        gl.cull_face(GL::BACK);
        gl.clear_color(0.0, 0.0, 0.02, 1.0);

        Ok(Self {
            gl,
            programs,
            earth_vao,
            earth_index_count,
            atmo_inner_vao,
            atmo_inner_index_count,
            atmo_outer_vao,
            atmo_outer_index_count,
            star_vao,
            star_count,
            marker_quad_vao: None,
            marker_instance_buffer: None,
            marker_count: 0,
            arcs: Vec::new(),
            earth_texture,
            bump_texture,
            cloud_texture,
            nightlights_texture,
            cloud_vao,
            cloud_index_count,
            core_vao,
            core_index_count,
            camera: OrbitalCamera::new(),
            celestial_vao,
            celestial_count,
            body_sphere_vao,
            body_sphere_index_count,
            bodies,
            particle_vao,
            particle_count,
            solar_wind_vao,
            solar_wind_count,
            scene_fbo: None,
            bloom_fbo_a: None,
            bloom_fbo_b: None,
            fullscreen_vao,
            last_width: 0,
            last_height: 0,
            time: 0.0,
            psi: 0.5,
            psi_heartbeat: true,
            show_core: false,
            phi_hotspots: Vec::new(),
        })
    }

    pub fn time_secs(&self) -> f64 {
        self.time
    }

    pub fn set_psi(&mut self, psi: f32) {
        self.psi = psi.clamp(0.0, 1.0);
    }

    pub fn set_phi_hotspots(&mut self, hotspots: Vec<[f32; 4]>) {
        self.phi_hotspots = hotspots;
    }

    pub fn frame(&mut self, time_ms: f64, canvas_width: u32, canvas_height: u32) {
        self.time = time_ms / 1000.0;
        let gl = &self.gl;

        // Consciousness heartbeat: organic pulsing between 0.3 and 0.8
        if self.psi_heartbeat {
            let t = self.time as f32;
            // Compound breathing: 8s Sacred Stillness + 2s consciousness pulse
            let slow = (t * std::f32::consts::PI / 4.0).sin(); // 8s cycle
            let fast = (t * std::f32::consts::PI).sin();        // 2s cycle
            self.psi = 0.55 + slow * 0.15 + fast * 0.05;
        }

        self.camera.update(self.time as f64);

        let width = canvas_width as f32;
        let height = canvas_height as f32;
        if width <= 0.0 || height <= 0.0 {
            return;
        }

        // Recreate FBOs if canvas resized
        if canvas_width != self.last_width || canvas_height != self.last_height {
            self.scene_fbo = Self::create_fbo(gl, canvas_width, canvas_height).ok();
            self.bloom_fbo_a = Self::create_fbo(gl, canvas_width / 2, canvas_height / 2).ok();
            self.bloom_fbo_b = Self::create_fbo(gl, canvas_width / 2, canvas_height / 2).ok();
            self.last_width = canvas_width;
            self.last_height = canvas_height;
        }

        let aspect = width / height;
        let projection = self.camera.projection_matrix(aspect);
        let view = self.camera.view_matrix(self.time as f64);
        let model_view = view;
        let proj_arr = projection.as_f32_array();
        let mv_arr = model_view.as_f32_array();
        let normal_mat = model_view.normal_matrix();

        let has_bloom = self.scene_fbo.is_some() && self.bloom_fbo_a.is_some() && self.bloom_fbo_b.is_some();

        // ── Pass 1: Render scene to FBO (or screen if no bloom) ──
        if has_bloom {
            let fbo = self.scene_fbo.as_ref().unwrap();
            gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&fbo.framebuffer));
            gl.viewport(0, 0, canvas_width as i32, canvas_height as i32);
        } else {
            gl.viewport(0, 0, canvas_width as i32, canvas_height as i32);
        }

        gl.clear(GL::COLOR_BUFFER_BIT | GL::DEPTH_BUFFER_BIT);

        // 0. Sacred geometry background (fullscreen quad, no depth)
        self.draw_sacred_background();

        // 1. Starfield + celestial bodies
        self.draw_starfield(&proj_arr, &mv_arr);
        self.draw_bodies(&proj_arr, &mv_arr);
        self.draw_solar_wind(&proj_arr, &mv_arr);
        self.draw_celestials(&proj_arr, &mv_arr);

        // 2. Core (if active, rendered before earth so it shows through translucent ocean)
        if self.show_core {
            self.draw_core(&proj_arr, &mv_arr, &normal_mat);
        }

        // 3. Earth + clouds + atmosphere + particles
        self.draw_earth(&proj_arr, &mv_arr, &normal_mat);
        self.draw_clouds(&proj_arr, &mv_arr);
        self.draw_atmosphere(&proj_arr, &mv_arr, &normal_mat);
        self.draw_particles(&proj_arr, &mv_arr);
        if self.marker_count > 0 {
            self.draw_markers(&proj_arr, &mv_arr);
        }
        self.draw_arcs(&proj_arr, &mv_arr);

        // ── Pass 2-4: Bloom post-processing ──
        if has_bloom {
            self.draw_bloom(canvas_width, canvas_height);
        }
    }

    fn draw_bloom(&self, width: u32, height: u32) {
        let gl = &self.gl;
        let scene_fbo = self.scene_fbo.as_ref().unwrap();
        let bloom_a = self.bloom_fbo_a.as_ref().unwrap();
        let bloom_b = self.bloom_fbo_b.as_ref().unwrap();

        gl.disable(GL::DEPTH_TEST);
        gl.disable(GL::CULL_FACE);

        // ── Extract bright pixels → bloom_a ──
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&bloom_a.framebuffer));
        gl.viewport(0, 0, bloom_a.width as i32, bloom_a.height as i32);
        gl.clear(GL::COLOR_BUFFER_BIT);

        gl.use_program(Some(&self.programs.bloom_extract));
        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, Some(&scene_fbo.color_texture));
        let loc = gl.get_uniform_location(&self.programs.bloom_extract, "u_scene");
        gl.uniform1i(loc.as_ref(), 0);
        let loc = gl.get_uniform_location(&self.programs.bloom_extract, "u_threshold");
        gl.uniform1f(loc.as_ref(), 0.35);

        gl.bind_vertex_array(Some(&self.fullscreen_vao));
        gl.draw_arrays(GL::TRIANGLES, 0, 6);

        // ── Horizontal blur: bloom_a → bloom_b ──
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&bloom_b.framebuffer));
        gl.viewport(0, 0, bloom_b.width as i32, bloom_b.height as i32);
        gl.clear(GL::COLOR_BUFFER_BIT);

        gl.use_program(Some(&self.programs.bloom_blur));
        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, Some(&bloom_a.color_texture));
        let loc = gl.get_uniform_location(&self.programs.bloom_blur, "u_input");
        gl.uniform1i(loc.as_ref(), 0);
        let loc = gl.get_uniform_location(&self.programs.bloom_blur, "u_direction");
        gl.uniform2f(loc.as_ref(), 1.0 / bloom_a.width as f32, 0.0);

        gl.draw_arrays(GL::TRIANGLES, 0, 6);

        // ── Vertical blur: bloom_b → bloom_a ──
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&bloom_a.framebuffer));
        gl.viewport(0, 0, bloom_a.width as i32, bloom_a.height as i32);
        gl.clear(GL::COLOR_BUFFER_BIT);

        gl.bind_texture(GL::TEXTURE_2D, Some(&bloom_b.color_texture));
        let loc = gl.get_uniform_location(&self.programs.bloom_blur, "u_direction");
        gl.uniform2f(loc.as_ref(), 0.0, 1.0 / bloom_b.height as f32);

        gl.draw_arrays(GL::TRIANGLES, 0, 6);

        // ── Composite: scene + bloom → screen ──
        gl.bind_framebuffer(GL::FRAMEBUFFER, None);
        gl.viewport(0, 0, width as i32, height as i32);
        gl.clear(GL::COLOR_BUFFER_BIT);

        gl.use_program(Some(&self.programs.bloom_composite));

        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, Some(&scene_fbo.color_texture));
        let loc = gl.get_uniform_location(&self.programs.bloom_composite, "u_scene");
        gl.uniform1i(loc.as_ref(), 0);

        gl.active_texture(GL::TEXTURE1);
        gl.bind_texture(GL::TEXTURE_2D, Some(&bloom_a.color_texture));
        let loc = gl.get_uniform_location(&self.programs.bloom_composite, "u_bloom");
        gl.uniform1i(loc.as_ref(), 1);

        let loc = gl.get_uniform_location(&self.programs.bloom_composite, "u_bloom_strength");
        gl.uniform1f(loc.as_ref(), 0.8);

        gl.draw_arrays(GL::TRIANGLES, 0, 6);

        // Restore state
        gl.enable(GL::DEPTH_TEST);
        gl.enable(GL::CULL_FACE);
    }

    fn draw_sacred_background(&self) {
        let gl = &self.gl;
        gl.disable(GL::DEPTH_TEST);
        gl.disable(GL::CULL_FACE);
        gl.use_program(Some(&self.programs.sacred_bg));

        let time_loc = gl.get_uniform_location(&self.programs.sacred_bg, "u_time");
        gl.uniform1f(time_loc.as_ref(), self.time as f32);

        gl.bind_vertex_array(Some(&self.fullscreen_vao));
        gl.draw_arrays(GL::TRIANGLES, 0, 6);

        gl.enable(GL::DEPTH_TEST);
        gl.enable(GL::CULL_FACE);
    }

    fn draw_starfield(&self, proj: &[f32; 16], mv: &[f32; 16]) {
        let gl = &self.gl;
        gl.disable(GL::DEPTH_TEST);
        gl.use_program(Some(&self.programs.star));
        self.set_matrix_uniforms(&self.programs.star, proj, mv);
        let time_loc = gl.get_uniform_location(&self.programs.star, "u_time");
        gl.uniform1f(time_loc.as_ref(), self.time as f32);
        gl.bind_vertex_array(Some(&self.star_vao));
        gl.draw_arrays(GL::POINTS, 0, self.star_count);
        gl.enable(GL::DEPTH_TEST);
    }

    fn draw_celestials(&self, proj: &[f32; 16], mv: &[f32; 16]) {
        let gl = &self.gl;
        gl.disable(GL::DEPTH_TEST);
        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE); // additive for glow

        gl.use_program(Some(&self.programs.celestial));
        self.set_matrix_uniforms(&self.programs.celestial, proj, mv);
        let time_loc = gl.get_uniform_location(&self.programs.celestial, "u_time");
        gl.uniform1f(time_loc.as_ref(), self.time as f32);

        gl.bind_vertex_array(Some(&self.celestial_vao));
        gl.draw_arrays(GL::POINTS, 0, self.celestial_count);

        gl.enable(GL::DEPTH_TEST);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);
    }

    fn draw_bodies(&self, proj: &[f32; 16], mv: &[f32; 16]) {
        let gl = &self.gl;
        let t = self.time as f32;

        // Compute Sun position for lighting all bodies
        let sun_body = &self.bodies[0]; // Sun is always index 0
        let sun_angle = t * sun_body.orbit_speed + sun_body.orbit_offset;
        let sun_pos = [
            sun_body.distance * sun_angle.cos(),
            sun_body.y_offset,
            sun_body.distance * sun_angle.sin(),
        ];
        let sun_len = (sun_pos[0]*sun_pos[0] + sun_pos[1]*sun_pos[1] + sun_pos[2]*sun_pos[2]).sqrt();
        let sun_dir = [sun_pos[0]/sun_len, sun_pos[1]/sun_len, sun_pos[2]/sun_len];

        gl.enable(GL::DEPTH_TEST);
        gl.cull_face(GL::BACK);

        // Draw each body with its photorealistic texture
        for body in &self.bodies {
            let angle = t * body.orbit_speed + body.orbit_offset;
            let pos = [
                body.distance * angle.cos(),
                body.y_offset + (angle * 0.7).sin() * body.y_offset.abs() * 0.3,
                body.distance * angle.sin(),
            ];

            if body.is_sun {
                // Sun: use textured body but with full ambient (self-lit)
                gl.use_program(Some(&self.programs.textured_body));
                self.set_matrix_uniforms(&self.programs.textured_body, proj, mv);

                let loc = gl.get_uniform_location(&self.programs.textured_body, "u_body_position");
                gl.uniform3f(loc.as_ref(), pos[0], pos[1], pos[2]);
                let loc = gl.get_uniform_location(&self.programs.textured_body, "u_body_radius");
                gl.uniform1f(loc.as_ref(), body.radius);
                let loc = gl.get_uniform_location(&self.programs.textured_body, "u_sun_direction");
                gl.uniform3f(loc.as_ref(), 0.0, 0.0, 1.0); // self-lit
                let loc = gl.get_uniform_location(&self.programs.textured_body, "u_ambient");
                gl.uniform1f(loc.as_ref(), 1.0); // fully lit

                gl.active_texture(GL::TEXTURE0);
                gl.bind_texture(GL::TEXTURE_2D, Some(&body.texture));
                let loc = gl.get_uniform_location(&self.programs.textured_body, "u_body_texture");
                gl.uniform1i(loc.as_ref(), 0);

                gl.bind_vertex_array(Some(&self.body_sphere_vao));
                gl.draw_elements_with_i32(GL::TRIANGLES, self.body_sphere_index_count, GL::UNSIGNED_INT, 0);

                // Corona glow
                gl.enable(GL::BLEND);
                gl.blend_func(GL::SRC_ALPHA, GL::ONE);
                gl.depth_mask(false);
                gl.cull_face(GL::FRONT);

                gl.use_program(Some(&self.programs.corona));
                self.set_matrix_uniforms(&self.programs.corona, proj, mv);
                let loc = gl.get_uniform_location(&self.programs.corona, "u_body_position");
                gl.uniform3f(loc.as_ref(), pos[0], pos[1], pos[2]);
                let loc = gl.get_uniform_location(&self.programs.corona, "u_body_radius");
                gl.uniform1f(loc.as_ref(), body.radius * 2.5);
                let loc = gl.get_uniform_location(&self.programs.corona, "u_time");
                gl.uniform1f(loc.as_ref(), t);

                gl.draw_elements_with_i32(GL::TRIANGLES, self.body_sphere_index_count, GL::UNSIGNED_INT, 0);

                gl.depth_mask(true);
                gl.cull_face(GL::BACK);
                gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);
            } else {
                // Planet/Moon: textured with sun lighting
                gl.use_program(Some(&self.programs.textured_body));
                self.set_matrix_uniforms(&self.programs.textured_body, proj, mv);

                let loc = gl.get_uniform_location(&self.programs.textured_body, "u_body_position");
                gl.uniform3f(loc.as_ref(), pos[0], pos[1], pos[2]);
                let loc = gl.get_uniform_location(&self.programs.textured_body, "u_body_radius");
                gl.uniform1f(loc.as_ref(), body.radius);
                let loc = gl.get_uniform_location(&self.programs.textured_body, "u_sun_direction");
                gl.uniform3f(loc.as_ref(), sun_dir[0], sun_dir[1], sun_dir[2]);
                let loc = gl.get_uniform_location(&self.programs.textured_body, "u_ambient");
                gl.uniform1f(loc.as_ref(), body.ambient);

                gl.active_texture(GL::TEXTURE0);
                gl.bind_texture(GL::TEXTURE_2D, Some(&body.texture));
                let loc = gl.get_uniform_location(&self.programs.textured_body, "u_body_texture");
                gl.uniform1i(loc.as_ref(), 0);

                gl.bind_vertex_array(Some(&self.body_sphere_vao));
                gl.draw_elements_with_i32(GL::TRIANGLES, self.body_sphere_index_count, GL::UNSIGNED_INT, 0);
            }
        }
    }

    fn draw_solar_wind(&self, proj: &[f32; 16], mv: &[f32; 16]) {
        let gl = &self.gl;
        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE); // additive
        gl.depth_mask(false);

        gl.use_program(Some(&self.programs.solar_wind));
        self.set_matrix_uniforms(&self.programs.solar_wind, proj, mv);

        let t = self.time as f32;
        let time_loc = gl.get_uniform_location(&self.programs.solar_wind, "u_time");
        gl.uniform1f(time_loc.as_ref(), t);

        // Pass Sun position
        let sun_angle = t * 0.02;
        let sun_pos = [20.0 * sun_angle.cos(), 2.0, 20.0 * sun_angle.sin()];
        let loc = gl.get_uniform_location(&self.programs.solar_wind, "u_sun_position");
        gl.uniform3f(loc.as_ref(), sun_pos[0], sun_pos[1], sun_pos[2]);

        gl.bind_vertex_array(Some(&self.solar_wind_vao));
        gl.draw_arrays(GL::POINTS, 0, self.solar_wind_count);

        gl.depth_mask(true);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);
    }

    fn draw_clouds(&self, proj: &[f32; 16], mv: &[f32; 16]) {
        let gl = &self.gl;
        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);
        gl.cull_face(GL::BACK);

        gl.use_program(Some(&self.programs.cloud));
        self.set_matrix_uniforms(&self.programs.cloud, proj, mv);

        let time_loc = gl.get_uniform_location(&self.programs.cloud, "u_time");
        gl.uniform1f(time_loc.as_ref(), self.time as f32);

        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, Some(&self.cloud_texture));
        let tex_loc = gl.get_uniform_location(&self.programs.cloud, "u_cloud_texture");
        gl.uniform1i(tex_loc.as_ref(), 0);

        gl.bind_vertex_array(Some(&self.cloud_vao));
        gl.draw_elements_with_i32(GL::TRIANGLES, self.cloud_index_count, GL::UNSIGNED_INT, 0);
    }

    fn draw_core(&self, proj: &[f32; 16], mv: &[f32; 16], normal: &[f32; 9]) {
        let gl = &self.gl;
        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE); // additive — core glows through
        gl.cull_face(GL::BACK);

        gl.use_program(Some(&self.programs.core_sphere));
        self.set_matrix_uniforms(&self.programs.core_sphere, proj, mv);

        let loc = gl.get_uniform_location(&self.programs.core_sphere, "u_normal_matrix");
        gl.uniform_matrix3fv_with_f32_array(loc.as_ref(), false, normal);
        let loc = gl.get_uniform_location(&self.programs.core_sphere, "u_time");
        gl.uniform1f(loc.as_ref(), self.time as f32);

        gl.bind_vertex_array(Some(&self.core_vao));
        gl.draw_elements_with_i32(GL::TRIANGLES, self.core_index_count, GL::UNSIGNED_INT, 0);

        gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);
    }

    fn draw_particles(&self, proj: &[f32; 16], mv: &[f32; 16]) {
        let gl = &self.gl;
        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE); // additive
        gl.depth_mask(false);

        gl.use_program(Some(&self.programs.particle));
        self.set_matrix_uniforms(&self.programs.particle, proj, mv);

        let time_loc = gl.get_uniform_location(&self.programs.particle, "u_time");
        gl.uniform1f(time_loc.as_ref(), self.time as f32);
        let psi_loc = gl.get_uniform_location(&self.programs.particle, "u_psi");
        gl.uniform1f(psi_loc.as_ref(), self.psi);

        gl.bind_vertex_array(Some(&self.particle_vao));
        gl.draw_arrays(GL::POINTS, 0, self.particle_count);

        gl.depth_mask(true);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);
    }

    fn draw_earth(&self, proj: &[f32; 16], mv: &[f32; 16], normal: &[f32; 9]) {
        let gl = &self.gl;

        // Enable blending for semi-transparent ocean
        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);
        gl.cull_face(GL::BACK);

        gl.use_program(Some(&self.programs.earth));
        self.set_matrix_uniforms(&self.programs.earth, proj, mv);

        let loc = gl.get_uniform_location(&self.programs.earth, "u_normal_matrix");
        gl.uniform_matrix3fv_with_f32_array(loc.as_ref(), false, normal);

        let time_loc = gl.get_uniform_location(&self.programs.earth, "u_time");
        gl.uniform1f(time_loc.as_ref(), self.time as f32);

        let psi_loc = gl.get_uniform_location(&self.programs.earth, "u_psi");
        gl.uniform1f(psi_loc.as_ref(), self.psi);

        let core_loc = gl.get_uniform_location(&self.programs.earth, "u_show_core");
        gl.uniform1f(core_loc.as_ref(), if self.show_core { 1.0 } else { 0.0 });

        // Pass sun direction for day/night shading (from Sun body orbit)
        let sun_angle_e = self.time as f32 * 0.02; // matches bodies[0].orbit_speed
        let sun_d = [20.0 * sun_angle_e.cos(), 2.0, 20.0 * sun_angle_e.sin()];
        let sun_l = (sun_d[0]*sun_d[0] + sun_d[1]*sun_d[1] + sun_d[2]*sun_d[2]).sqrt();
        let loc = gl.get_uniform_location(&self.programs.earth, "u_sun_direction");
        gl.uniform3f(loc.as_ref(), sun_d[0]/sun_l, sun_d[1]/sun_l, sun_d[2]/sun_l);

        // Pass Φ hotspots for localized grid warping
        if !self.phi_hotspots.is_empty() {
            let mut flat: Vec<f32> = Vec::with_capacity(48); // 12 × 4
            for h in &self.phi_hotspots {
                flat.extend_from_slice(h);
            }
            // Pad to exactly 48 floats
            while flat.len() < 48 { flat.push(0.0); }
            let loc = gl.get_uniform_location(&self.programs.earth, "u_phi_hotspots");
            gl.uniform4fv_with_f32_array(loc.as_ref(), &flat);
        }

        // Bind textures (used for land/ocean detection data)
        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, Some(&self.earth_texture));
        let tex_loc = gl.get_uniform_location(&self.programs.earth, "u_earth_texture");
        gl.uniform1i(tex_loc.as_ref(), 0);

        gl.active_texture(GL::TEXTURE1);
        gl.bind_texture(GL::TEXTURE_2D, Some(&self.bump_texture));
        let bump_loc = gl.get_uniform_location(&self.programs.earth, "u_boundaries_texture");
        gl.uniform1i(bump_loc.as_ref(), 1);

        gl.active_texture(GL::TEXTURE2);
        gl.bind_texture(GL::TEXTURE_2D, Some(&self.nightlights_texture));
        let night_loc = gl.get_uniform_location(&self.programs.earth, "u_nightlights_texture");
        gl.uniform1i(night_loc.as_ref(), 2);

        gl.bind_vertex_array(Some(&self.earth_vao));
        gl.draw_elements_with_i32(GL::TRIANGLES, self.earth_index_count, GL::UNSIGNED_INT, 0);

        gl.disable(GL::BLEND);
    }

    fn draw_atmosphere(&self, proj: &[f32; 16], mv: &[f32; 16], normal: &[f32; 9]) {
        let gl = &self.gl;
        gl.enable(GL::BLEND);
        gl.blend_func(GL::ONE, GL::ONE); // additive
        gl.depth_mask(false);
        gl.cull_face(GL::FRONT); // back faces (inside-out)

        // Inner atmosphere — harmony cycling
        gl.use_program(Some(&self.programs.atmo_inner));
        self.set_matrix_uniforms(&self.programs.atmo_inner, proj, mv);
        let loc = gl.get_uniform_location(&self.programs.atmo_inner, "u_normal_matrix");
        gl.uniform_matrix3fv_with_f32_array(loc.as_ref(), false, normal);
        let time_loc = gl.get_uniform_location(&self.programs.atmo_inner, "u_time");
        gl.uniform1f(time_loc.as_ref(), self.time as f32);
        gl.bind_vertex_array(Some(&self.atmo_inner_vao));
        gl.draw_elements_with_i32(GL::TRIANGLES, self.atmo_inner_index_count, GL::UNSIGNED_INT, 0);

        // Outer atmosphere — aurora + harmony
        gl.use_program(Some(&self.programs.atmo_outer));
        self.set_matrix_uniforms(&self.programs.atmo_outer, proj, mv);
        let loc = gl.get_uniform_location(&self.programs.atmo_outer, "u_normal_matrix");
        gl.uniform_matrix3fv_with_f32_array(loc.as_ref(), false, normal);
        let time_loc = gl.get_uniform_location(&self.programs.atmo_outer, "u_time");
        gl.uniform1f(time_loc.as_ref(), self.time as f32);
        gl.bind_vertex_array(Some(&self.atmo_outer_vao));
        gl.draw_elements_with_i32(GL::TRIANGLES, self.atmo_outer_index_count, GL::UNSIGNED_INT, 0);

        // Restore
        gl.depth_mask(true);
        gl.cull_face(GL::BACK);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);
    }

    fn draw_markers(&self, proj: &[f32; 16], mv: &[f32; 16]) {
        let gl = &self.gl;
        let vao = match &self.marker_quad_vao {
            Some(v) => v,
            None => return,
        };

        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);

        gl.use_program(Some(&self.programs.marker));
        self.set_matrix_uniforms(&self.programs.marker, proj, mv);

        let time_loc = gl.get_uniform_location(&self.programs.marker, "u_time");
        gl.uniform1f(time_loc.as_ref(), self.time as f32);

        // Pass camera position for horizon culling
        let eye = self.camera.eye_position(self.time);
        let cam_loc = gl.get_uniform_location(&self.programs.marker, "u_camera_position");
        gl.uniform3f(cam_loc.as_ref(), eye.x, eye.y, eye.z);

        gl.bind_vertex_array(Some(vao));
        gl.draw_arrays_instanced(GL::TRIANGLES, 0, 6, self.marker_count);

        gl.disable(GL::BLEND);
    }

    fn draw_arcs(&self, proj: &[f32; 16], mv: &[f32; 16]) {
        if self.arcs.is_empty() {
            return;
        }
        let gl = &self.gl;
        gl.enable(GL::BLEND);
        gl.blend_func(GL::ONE, GL::ONE); // additive for bioluminescent glow

        gl.use_program(Some(&self.programs.arc));
        self.set_matrix_uniforms(&self.programs.arc, proj, mv);

        let time_loc = gl.get_uniform_location(&self.programs.arc, "u_time");
        gl.uniform1f(time_loc.as_ref(), self.time as f32);

        let color_loc = gl.get_uniform_location(&self.programs.arc, "u_color");
        let flow_loc = gl.get_uniform_location(&self.programs.arc, "u_flow_offset");

        for arc in &self.arcs {
            gl.uniform3f(color_loc.as_ref(), arc.color.x, arc.color.y, arc.color.z);
            gl.uniform1f(flow_loc.as_ref(), arc.flow_offset);
            gl.bind_vertex_array(Some(&arc.vao));
            gl.draw_arrays(GL::LINE_STRIP, 0, arc.vertex_count);
        }

        gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);
    }

    fn set_matrix_uniforms(&self, program: &WebGlProgram, proj: &[f32; 16], mv: &[f32; 16]) {
        let gl = &self.gl;
        let proj_loc = gl.get_uniform_location(program, "u_projection");
        gl.uniform_matrix4fv_with_f32_array(proj_loc.as_ref(), false, proj);
        let mv_loc = gl.get_uniform_location(program, "u_model_view");
        gl.uniform_matrix4fv_with_f32_array(mv_loc.as_ref(), false, mv);
    }

    // ─── Marker Updates ──────────────────────────────────────────

    pub fn update_markers(&mut self, markers: &[MarkerInstance]) {
        let gl = &self.gl;
        self.marker_count = markers.len() as i32;

        if markers.is_empty() {
            self.marker_quad_vao = None;
            return;
        }

        // Instance data: [pos.xyz, color.xyz, size, type] × N = 8 floats per instance
        let mut instance_data = Vec::with_capacity(markers.len() * 8);
        for m in markers {
            instance_data.push(m.position.x);
            instance_data.push(m.position.y);
            instance_data.push(m.position.z);
            instance_data.push(m.color.x);
            instance_data.push(m.color.y);
            instance_data.push(m.color.z);
            instance_data.push(m.size);
            instance_data.push(m.marker_type);
        }

        let vao = gl.create_vertex_array().unwrap();
        gl.bind_vertex_array(Some(&vao));

        // Quad vertices (billboard, 2 triangles) at location 0
        let quad_verts: [f32; 12] = [
            -0.5, -0.5,
             0.5, -0.5,
             0.5,  0.5,
            -0.5, -0.5,
             0.5,  0.5,
            -0.5,  0.5,
        ];
        let quad_buf = gl.create_buffer().unwrap();
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&quad_buf));
        unsafe {
            let view = js_sys::Float32Array::view(&quad_verts);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &view, GL::STATIC_DRAW);
        }
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_with_i32(0, 2, GL::FLOAT, false, 0, 0);

        // Instance buffer at locations 1-4
        let inst_buf = gl.create_buffer().unwrap();
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&inst_buf));
        unsafe {
            let view = js_sys::Float32Array::view(&instance_data);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &view, GL::DYNAMIC_DRAW);
        }

        let stride = 8 * 4; // 8 floats × 4 bytes
        // i_position (location 1)
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_with_i32(1, 3, GL::FLOAT, false, stride, 0);
        gl.vertex_attrib_divisor(1, 1);
        // i_color (location 2)
        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_with_i32(2, 3, GL::FLOAT, false, stride, 3 * 4);
        gl.vertex_attrib_divisor(2, 1);
        // i_size (location 3)
        gl.enable_vertex_attrib_array(3);
        gl.vertex_attrib_pointer_with_i32(3, 1, GL::FLOAT, false, stride, 6 * 4);
        gl.vertex_attrib_divisor(3, 1);
        // i_type (location 4)
        gl.enable_vertex_attrib_array(4);
        gl.vertex_attrib_pointer_with_i32(4, 1, GL::FLOAT, false, stride, 7 * 4);
        gl.vertex_attrib_divisor(4, 1);

        gl.bind_vertex_array(None);

        self.marker_quad_vao = Some(vao);
        self.marker_instance_buffer = Some(inst_buf);
    }

    // ─── Arc Updates ─────────────────────────────────────────────

    pub fn update_arcs(&mut self, arc_datas: Vec<(Vec<f32>, Vec3, f32)>) {
        self.arcs.clear();

        let gl = &self.gl;
        for (vertices, color, flow_offset) in arc_datas {
            let vertex_count = (vertices.len() / 4) as i32;
            if vertex_count == 0 {
                continue;
            }

            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(&vao));

            let buf = gl.create_buffer().unwrap();
            gl.bind_buffer(GL::ARRAY_BUFFER, Some(&buf));
            unsafe {
                let view = js_sys::Float32Array::view(&vertices);
                gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &view, GL::STATIC_DRAW);
            }

            let stride = 4 * 4;
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_with_i32(0, 3, GL::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_with_i32(1, 1, GL::FLOAT, false, stride, 3 * 4);

            gl.bind_vertex_array(None);

            self.arcs.push(ArcData {
                vao,
                vertex_count,
                color,
                flow_offset,
            });
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────

    fn create_sphere_vao(
        gl: &GL,
        lat_segs: u32,
        lon_segs: u32,
        radius: f32,
    ) -> Result<(WebGlVertexArrayObject, i32), String> {
        let (vertices, indices) = geometry::generate_sphere(lat_segs, lon_segs, radius);

        let vao = gl.create_vertex_array().ok_or("create_vertex_array failed")?;
        gl.bind_vertex_array(Some(&vao));

        let vbo = gl.create_buffer().ok_or("create_buffer failed")?;
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&vbo));
        unsafe {
            let view = js_sys::Float32Array::view(&vertices);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &view, GL::STATIC_DRAW);
        }

        let stride = 8 * 4;
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_with_i32(0, 3, GL::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_with_i32(1, 3, GL::FLOAT, false, stride, 3 * 4);
        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_with_i32(2, 2, GL::FLOAT, false, stride, 6 * 4);

        let ebo = gl.create_buffer().ok_or("create_buffer failed")?;
        gl.bind_buffer(GL::ELEMENT_ARRAY_BUFFER, Some(&ebo));
        unsafe {
            let view = js_sys::Uint32Array::view(&indices);
            gl.buffer_data_with_array_buffer_view(GL::ELEMENT_ARRAY_BUFFER, &view, GL::STATIC_DRAW);
        }

        gl.bind_vertex_array(None);

        Ok((vao, indices.len() as i32))
    }

    fn create_starfield_vao(
        gl: &GL,
        count: u32,
        radius: f32,
    ) -> Result<(WebGlVertexArrayObject, i32), String> {
        let data = geometry::generate_starfield(count, radius);

        let vao = gl.create_vertex_array().ok_or("create_vertex_array failed")?;
        gl.bind_vertex_array(Some(&vao));

        let vbo = gl.create_buffer().ok_or("create_buffer failed")?;
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&vbo));
        unsafe {
            let view = js_sys::Float32Array::view(&data);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &view, GL::STATIC_DRAW);
        }

        let stride = 7 * 4; // 7 floats: pos.xyz + color.rgb + brightness
        gl.enable_vertex_attrib_array(0); // position
        gl.vertex_attrib_pointer_with_i32(0, 3, GL::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(1); // color
        gl.vertex_attrib_pointer_with_i32(1, 3, GL::FLOAT, false, stride, 3 * 4);
        gl.enable_vertex_attrib_array(2); // brightness
        gl.vertex_attrib_pointer_with_i32(2, 1, GL::FLOAT, false, stride, 6 * 4);

        gl.bind_vertex_array(None);

        Ok((vao, count as i32))
    }

    fn create_solar_wind_vao(gl: &GL, count: u32) -> Result<(WebGlVertexArrayObject, i32), String> {
        // Each particle: angle, speed, offset
        let mut data = Vec::with_capacity(count as usize * 3);
        let mut seed: u32 = 0x501A_B10D;
        let next_f32 = |s: &mut u32| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 17;
            *s ^= *s << 5;
            (*s as f32) / u32::MAX as f32
        };

        for _ in 0..count {
            data.push(next_f32(&mut seed) * std::f32::consts::PI * 2.0); // angle
            data.push(0.3 + next_f32(&mut seed) * 0.7); // speed
            data.push(next_f32(&mut seed) * 8.0); // lifecycle offset
        }

        let vao = gl.create_vertex_array().ok_or("create_vertex_array failed")?;
        gl.bind_vertex_array(Some(&vao));

        let buf = gl.create_buffer().ok_or("create_buffer failed")?;
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&buf));
        unsafe {
            let view = js_sys::Float32Array::view(&data);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &view, GL::STATIC_DRAW);
        }

        let stride = 3 * 4;
        gl.enable_vertex_attrib_array(0); // angle
        gl.vertex_attrib_pointer_with_i32(0, 1, GL::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(1); // speed
        gl.vertex_attrib_pointer_with_i32(1, 1, GL::FLOAT, false, stride, 4);
        gl.enable_vertex_attrib_array(2); // offset
        gl.vertex_attrib_pointer_with_i32(2, 1, GL::FLOAT, false, stride, 8);

        gl.bind_vertex_array(None);
        Ok((vao, count as i32))
    }

    fn create_particle_vao(gl: &GL, count: u32) -> Result<(WebGlVertexArrayObject, i32), String> {
        // Generate particles on a shell between radius 1.12 and 1.25
        let mut data = Vec::with_capacity(count as usize * 4); // xyz + spawn_time

        let mut seed: u32 = 0xCAFE_BABE;
        let next_f32 = |s: &mut u32| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 17;
            *s ^= *s << 5;
            (*s as f32) / u32::MAX as f32
        };

        for _ in 0..count {
            // Uniform distribution on sphere shell
            let u = next_f32(&mut seed) * 2.0 - 1.0;
            let theta = next_f32(&mut seed) * 2.0 * std::f32::consts::PI;
            let r_shell = 1.12 + next_f32(&mut seed) * 0.13; // 1.12 to 1.25
            let r = (1.0 - u * u).sqrt();

            data.push(r * theta.cos() * r_shell);
            data.push(u * r_shell);
            data.push(r * theta.sin() * r_shell);
            data.push(next_f32(&mut seed) * 3.0); // spawn_time offset [0, 3]
        }

        let vao = gl.create_vertex_array().ok_or("create_vertex_array failed")?;
        gl.bind_vertex_array(Some(&vao));

        let buf = gl.create_buffer().ok_or("create_buffer failed")?;
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&buf));
        unsafe {
            let view = js_sys::Float32Array::view(&data);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &view, GL::STATIC_DRAW);
        }

        let stride = 4 * 4; // 4 floats
        gl.enable_vertex_attrib_array(0); // position
        gl.vertex_attrib_pointer_with_i32(0, 3, GL::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(1); // spawn_time
        gl.vertex_attrib_pointer_with_i32(1, 1, GL::FLOAT, false, stride, 3 * 4);

        gl.bind_vertex_array(None);
        Ok((vao, count as i32))
    }

    fn create_celestial_vao(gl: &GL) -> Result<(WebGlVertexArrayObject, i32), String> {
        // Solar system bodies at logarithmic-scaled distances
        // Format: [x, y, z, r, g, b, point_size] per body
        let bodies: Vec<[f32; 7]> = vec![
            // Sun: bright golden glow, large, placed behind globe
            [  8.0,  3.0,  5.0,   1.0, 0.95, 0.6,   12.0],
            // Venus: warm white
            [  3.5,  0.8,  2.5,   0.9, 0.85, 0.7,    3.0],
            // Mars: reddish
            [ -5.0,  1.5, -4.0,   0.9, 0.4,  0.3,    3.5],
            // Jupiter: amber
            [-10.0, -2.0,  7.0,   0.8, 0.7,  0.5,    5.0],
            // Saturn: golden
            [ 12.0, -3.0, -8.0,   0.85, 0.75, 0.5,   4.5],
            // Moon: cool white, close
            [  1.2,  0.5, -0.8,   0.8, 0.82, 0.85,   2.5],
        ];

        let count = bodies.len() as i32;
        let mut data = Vec::with_capacity(bodies.len() * 7);
        for b in &bodies {
            data.extend_from_slice(b);
        }

        let vao = gl.create_vertex_array().ok_or("create_vertex_array failed")?;
        gl.bind_vertex_array(Some(&vao));

        let buf = gl.create_buffer().ok_or("create_buffer failed")?;
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&buf));
        unsafe {
            let view = js_sys::Float32Array::view(&data);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &view, GL::STATIC_DRAW);
        }

        let stride = 7 * 4;
        gl.enable_vertex_attrib_array(0); // position
        gl.vertex_attrib_pointer_with_i32(0, 3, GL::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(1); // color
        gl.vertex_attrib_pointer_with_i32(1, 3, GL::FLOAT, false, stride, 3 * 4);
        gl.enable_vertex_attrib_array(2); // size
        gl.vertex_attrib_pointer_with_i32(2, 1, GL::FLOAT, false, stride, 6 * 4);

        gl.bind_vertex_array(None);
        Ok((vao, count))
    }

    fn create_fullscreen_quad_vao(gl: &GL) -> Result<WebGlVertexArrayObject, String> {
        let vao = gl.create_vertex_array().ok_or("create_vertex_array failed")?;
        gl.bind_vertex_array(Some(&vao));

        // Two triangles covering [-1, 1] NDC
        let quad: [f32; 12] = [
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
        ];
        let buf = gl.create_buffer().ok_or("create_buffer failed")?;
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&buf));
        unsafe {
            let view = js_sys::Float32Array::view(&quad);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &view, GL::STATIC_DRAW);
        }
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_with_i32(0, 2, GL::FLOAT, false, 0, 0);

        gl.bind_vertex_array(None);
        Ok(vao)
    }

    fn create_fbo(gl: &GL, width: u32, height: u32) -> Result<BloomFbo, String> {
        // Enable float color buffer if available (needed for RGBA16F)
        let has_float = gl.get_extension("EXT_color_buffer_float").ok().flatten().is_some()
            || gl.get_extension("EXT_color_buffer_half_float").ok().flatten().is_some();

        let framebuffer = gl.create_framebuffer().ok_or("create_framebuffer failed")?;
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&framebuffer));

        // Color texture — use RGBA16F if available, RGBA8 fallback
        let color_texture = gl.create_texture().ok_or("create_texture failed")?;
        gl.bind_texture(GL::TEXTURE_2D, Some(&color_texture));

        let (internal_format, format_type) = if has_float {
            (GL::RGBA16F as i32, GL::HALF_FLOAT)
        } else {
            (GL::RGBA as i32, GL::UNSIGNED_BYTE)
        };

        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            GL::TEXTURE_2D,
            0,
            internal_format,
            width as i32,
            height as i32,
            0,
            GL::RGBA,
            format_type,
            None,
        ).map_err(|e| format!("tex_image_2d: {e:?}"))?;
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, GL::LINEAR as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MAG_FILTER, GL::LINEAR as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_S, GL::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_T, GL::CLAMP_TO_EDGE as i32);
        gl.framebuffer_texture_2d(
            GL::FRAMEBUFFER,
            GL::COLOR_ATTACHMENT0,
            GL::TEXTURE_2D,
            Some(&color_texture),
            0,
        );

        // Depth renderbuffer (only needed for scene FBO)
        let depth_rb = gl.create_renderbuffer().ok_or("create_renderbuffer failed")?;
        gl.bind_renderbuffer(GL::RENDERBUFFER, Some(&depth_rb));
        gl.renderbuffer_storage(GL::RENDERBUFFER, GL::DEPTH_COMPONENT16, width as i32, height as i32);
        gl.framebuffer_renderbuffer(
            GL::FRAMEBUFFER,
            GL::DEPTH_ATTACHMENT,
            GL::RENDERBUFFER,
            Some(&depth_rb),
        );

        let status = gl.check_framebuffer_status(GL::FRAMEBUFFER);
        if status != GL::FRAMEBUFFER_COMPLETE {
            return Err(format!("Framebuffer incomplete: status {status}"));
        }

        gl.bind_framebuffer(GL::FRAMEBUFFER, None);

        Ok(BloomFbo {
            framebuffer,
            color_texture,
            width,
            height,
        })
    }
}
