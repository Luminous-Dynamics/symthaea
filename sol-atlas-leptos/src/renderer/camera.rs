// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::math::{Mat4, Vec3};
use std::f32::consts::PI;

const ZOOM_MIN: f32 = 1.8;
const ZOOM_MAX: f32 = 8.0;
const INERTIA_DECAY: f32 = 0.92;
const AUTO_ROTATE_SPEED: f32 = 0.0008; // satellite orbit speed
const DRIFT_AMPLITUDE: f32 = 0.01;
const PHI_CLAMP: f32 = 85.0 * PI / 180.0;

pub struct OrbitalCamera {
    pub theta: f32,    // azimuth
    pub phi: f32,      // elevation
    pub distance: f32,
    vel_theta: f32,
    vel_phi: f32,
    dragging: bool,
    last_x: f32,
    last_y: f32,
    // Look-at target offset (for planet fly-to)
    look_offset: Vec3,
    // Touch state
    pinch_start_dist: Option<f32>,
    pinch_start_zoom: f32,
    // Fly-to animation
    fly_target: Option<FlyTarget>,
}

struct FlyTarget {
    target_theta: f32,
    target_phi: f32,
    target_distance: f32,
    target_offset: Vec3, // look-at offset from origin
    start_theta: f32,
    start_phi: f32,
    start_distance: f32,
    start_offset: Vec3,
    progress: f32, // 0.0 to 1.0
}

impl OrbitalCamera {
    pub fn new() -> Self {
        Self {
            theta: -0.5, // slight initial rotation (matches Three.js earth.rotation.y = -0.5)
            phi: 0.0,
            distance: 4.2,
            vel_theta: 0.0,
            vel_phi: 0.0,
            dragging: false,
            last_x: 0.0,
            last_y: 0.0,
            look_offset: Vec3::ZERO,
            pinch_start_dist: None,
            pinch_start_zoom: 4.2,
            fly_target: None,
        }
    }

    pub fn on_mouse_down(&mut self, x: f32, y: f32) {
        self.dragging = true;
        self.last_x = x;
        self.last_y = y;
    }

    pub fn on_mouse_move(&mut self, x: f32, y: f32) {
        if !self.dragging {
            return;
        }
        let dx = x - self.last_x;
        let dy = y - self.last_y;
        self.vel_theta += dx * 0.002;
        self.vel_phi += dy * 0.002;
        self.last_x = x;
        self.last_y = y;
    }

    pub fn on_mouse_up(&mut self) {
        self.dragging = false;
    }

    pub fn on_wheel(&mut self, delta_y: f32) {
        let zoom_delta = delta_y * 0.001;
        self.distance = (self.distance + zoom_delta).clamp(ZOOM_MIN, ZOOM_MAX);
    }

    pub fn on_touch_start(&mut self, touches: &[(f32, f32)]) {
        match touches.len() {
            1 => {
                self.dragging = true;
                self.last_x = touches[0].0;
                self.last_y = touches[0].1;
                self.pinch_start_dist = None;
            }
            2 => {
                let dx = touches[1].0 - touches[0].0;
                let dy = touches[1].1 - touches[0].1;
                self.pinch_start_dist = Some((dx * dx + dy * dy).sqrt());
                self.pinch_start_zoom = self.distance;
                self.dragging = false;
            }
            _ => {}
        }
    }

    pub fn on_touch_move(&mut self, touches: &[(f32, f32)]) {
        match touches.len() {
            1 if self.dragging => {
                let dx = touches[0].0 - self.last_x;
                let dy = touches[0].1 - self.last_y;
                self.vel_theta += dx * 0.003;
                self.vel_phi += dy * 0.003;
                self.last_x = touches[0].0;
                self.last_y = touches[0].1;
            }
            2 => {
                if let Some(start_dist) = self.pinch_start_dist {
                    let dx = touches[1].0 - touches[0].0;
                    let dy = touches[1].1 - touches[0].1;
                    let cur_dist = (dx * dx + dy * dy).sqrt();
                    let scale = start_dist / cur_dist;
                    self.distance = (self.pinch_start_zoom * scale).clamp(ZOOM_MIN, ZOOM_MAX);
                }
            }
            _ => {}
        }
    }

    pub fn on_touch_end(&mut self) {
        self.dragging = false;
        self.pinch_start_dist = None;
    }

    /// Start a smooth fly-to animation to look at a target position
    pub fn fly_to(&mut self, target_pos: Vec3, view_distance: f32) {
        // Compute theta/phi to look at target
        let dir = target_pos.normalize();
        let target_theta = dir.x.atan2(dir.z);
        let target_phi = dir.y.asin().clamp(-PHI_CLAMP, PHI_CLAMP);

        self.fly_target = Some(FlyTarget {
            target_theta,
            target_phi,
            target_distance: view_distance,
            target_offset: target_pos,
            start_theta: self.theta,
            start_phi: self.phi,
            start_distance: self.distance,
            start_offset: Vec3::ZERO,
            progress: 0.0,
        });
        // Kill velocity during fly-to
        self.vel_theta = 0.0;
        self.vel_phi = 0.0;
    }

    /// Return to Earth overview
    pub fn fly_home(&mut self) {
        self.fly_target = Some(FlyTarget {
            target_theta: self.theta, // keep current angle
            target_phi: 0.0,
            target_distance: 4.2,
            target_offset: Vec3::ZERO,
            start_theta: self.theta,
            start_phi: self.phi,
            start_distance: self.distance,
            start_offset: self.look_offset,
            progress: 0.0,
        });
        self.vel_theta = 0.0;
        self.vel_phi = 0.0;
    }

    pub fn is_flying(&self) -> bool {
        self.fly_target.is_some()
    }

    pub fn update(&mut self, _time: f64) {
        // Handle fly-to animation
        if let Some(ref mut fly) = self.fly_target {
            fly.progress += 0.015; // ~60 frames for full animation
            if fly.progress >= 1.0 {
                self.theta = fly.target_theta;
                self.phi = fly.target_phi;
                self.distance = fly.target_distance;
                self.look_offset = fly.target_offset;
                self.fly_target = None;
            } else {
                // Cubic ease-in-out
                let t = fly.progress;
                let ease = if t < 0.5 { 4.0 * t * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(3) / 2.0 };
                self.theta = fly.start_theta + (fly.target_theta - fly.start_theta) * ease;
                self.phi = fly.start_phi + (fly.target_phi - fly.start_phi) * ease;
                self.distance = fly.start_distance + (fly.target_distance - fly.start_distance) * ease;
                self.look_offset = fly.start_offset.lerp(fly.target_offset, ease);
            }
            return; // skip normal camera controls during fly-to
        }

        // Apply inertial velocity
        self.theta += self.vel_theta;
        self.phi = (self.phi + self.vel_phi).clamp(-PHI_CLAMP, PHI_CLAMP);

        // Decay
        self.vel_theta *= INERTIA_DECAY;
        self.vel_phi *= INERTIA_DECAY;

        // Auto-rotate when nearly idle
        if !self.dragging && self.vel_theta.abs() < 0.001 && self.vel_phi.abs() < 0.001 {
            self.theta += AUTO_ROTATE_SPEED;
        }
    }

    pub fn eye_position(&self, time: f64) -> Vec3 {
        let t = time as f32;
        const PHI: f32 = 1.618033988; // golden ratio

        // Golden-ratio multi-frequency Lissajous drift (never visibly repeats)
        let drift_x = (t * 0.08).sin() * DRIFT_AMPLITUDE
            + (t * 0.08 * PHI).sin() * DRIFT_AMPLITUDE * 0.4;
        let drift_y = (t * 0.10).cos() * DRIFT_AMPLITUDE
            + (t * 0.10 * PHI).cos() * DRIFT_AMPLITUDE * 0.4;
        let drift_z = (t * 0.06).sin() * DRIFT_AMPLITUDE * 0.3;

        // 8-second Sacred Stillness breathing (0.3% scale oscillation)
        let breath = 1.0 + 0.003 * (t * std::f32::consts::PI / 4.0).sin().powi(2);

        let x = self.distance * breath * self.phi.cos() * self.theta.sin() + drift_x;
        let y = self.distance * breath * self.phi.sin() + drift_y;
        let z = self.distance * breath * self.phi.cos() * self.theta.cos() + drift_z;
        Vec3::new(x, y, z)
    }

    pub fn view_matrix(&self, time: f64) -> Mat4 {
        let eye = self.eye_position(time) + self.look_offset;
        Mat4::look_at(eye, self.look_offset, Vec3::UP)
    }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective(45.0 * PI / 180.0, aspect, 0.1, 1000.0)
    }

    pub fn is_dragging(&self) -> bool {
        self.dragging
    }
}
