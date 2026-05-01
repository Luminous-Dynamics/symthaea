// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real-time dimension transitioning: seamlessly shift between 2D, 2.5D, 3D, and 4D.
//!
//! No other game engine does this. The physics always runs in the same dimension;
//! the "dimension mode" controls how the world is projected and experienced:
//!
//! - **2D**: Top-down orthographic. Classic dungeon crawler. (Current default)
//! - **2.5D**: Isometric/oblique perspective. Adds visual depth without changing gameplay.
//! - **3D**: Full 3D camera with free look. Walls become tall. Z axis visible.
//! - **4D**: 3D view + W slider. Move the slider to reveal hidden rooms in the 4th dimension.
//!
//! Transition is smooth — camera angle interpolates, projection blends.
//! Press F1-F4 to switch dimensions, or use the scroll wheel to slide.

use bevy::prelude::*;

/// Current dimension mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Resource)]
pub enum DimensionMode {
    /// Top-down 2D. Camera at (0, 0, 999), looking down.
    D2,
    /// Isometric 2.5D. Camera tilted ~30° from top-down.
    D2Half,
    /// Full 3D. Camera free-look in XYZ.
    D3,
    /// 4D cross-section. 3D view + W slider for 4th dimension.
    D4,
}

impl Default for DimensionMode {
    fn default() -> Self {
        Self::D2
    }
}

impl DimensionMode {
    /// Camera elevation angle (radians from straight down).
    /// 0 = top-down (2D), π/2 = horizontal (3D)
    pub fn camera_pitch(&self) -> f32 {
        match self {
            Self::D2 => 0.0,      // straight down
            Self::D2Half => 0.52, // ~30° tilt (isometric)
            Self::D3 => 1.05,     // ~60° (3D perspective)
            Self::D4 => 1.05,     // same as 3D + W slider
        }
    }

    /// Camera distance multiplier.
    pub fn camera_distance(&self) -> f32 {
        match self {
            Self::D2 => 1.0,
            Self::D2Half => 1.2,
            Self::D3 => 0.8,
            Self::D4 => 0.8,
        }
    }

    /// Whether the W slider is active (4D cross-section).
    pub fn has_w_slider(&self) -> bool {
        matches!(self, Self::D4)
    }

    /// Whether sprite rendering should be used (2D/2.5D) vs mesh (3D/4D).
    pub fn uses_sprites(&self) -> bool {
        matches!(self, Self::D2 | Self::D2Half)
    }

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::D2 => "2D",
            Self::D2Half => "2.5D",
            Self::D3 => "3D",
            Self::D4 => "4D",
        }
    }
}

/// Dimension transition state — handles smooth interpolation between modes.
#[derive(Resource)]
pub struct DimensionTransition {
    pub current: DimensionMode,
    pub target: DimensionMode,
    /// Interpolation progress [0, 1]. 1.0 = fully transitioned.
    pub progress: f32,
    /// Transition speed (progress per second).
    pub speed: f32,
    /// W slider position for 4D mode.
    pub w_position: f32,
    /// W slider range.
    pub w_range: f32,
}

impl Default for DimensionTransition {
    fn default() -> Self {
        Self {
            current: DimensionMode::D2,
            target: DimensionMode::D2,
            progress: 1.0,
            speed: 2.0,
            w_position: 0.0,
            w_range: 100.0,
        }
    }
}

impl DimensionTransition {
    /// Whether a transition is in progress.
    pub fn transitioning(&self) -> bool {
        self.progress < 1.0
    }

    /// Effective camera pitch (blended between current and target).
    pub fn effective_pitch(&self) -> f32 {
        let from = self.current.camera_pitch();
        let to = self.target.camera_pitch();
        from + (to - from) * self.progress
    }

    /// Effective camera distance.
    pub fn effective_distance(&self) -> f32 {
        let from = self.current.camera_distance();
        let to = self.target.camera_distance();
        from + (to - from) * self.progress
    }
}

/// Handle dimension switching input (F1-F4 keys).
pub fn dimension_input_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut transition: ResMut<DimensionTransition>,
) {
    let new_target = if keyboard.just_pressed(KeyCode::F1) {
        Some(DimensionMode::D2)
    } else if keyboard.just_pressed(KeyCode::F2) {
        Some(DimensionMode::D2Half)
    } else if keyboard.just_pressed(KeyCode::F3) {
        Some(DimensionMode::D3)
    } else if keyboard.just_pressed(KeyCode::F4) {
        Some(DimensionMode::D4)
    } else {
        None
    };

    if let Some(target) = new_target {
        if target != transition.target {
            transition.current = transition.target;
            transition.target = target;
            transition.progress = 0.0;
            eprintln!(
                "[dimension] Transitioning: {} → {}",
                transition.current.name(),
                transition.target.name()
            );
        }
    }

    // W slider for 4D mode ([ and ] keys)
    if transition.target == DimensionMode::D4 || transition.current == DimensionMode::D4 {
        if keyboard.pressed(KeyCode::BracketLeft) {
            transition.w_position = (transition.w_position - 1.0).max(-transition.w_range);
        }
        if keyboard.pressed(KeyCode::BracketRight) {
            transition.w_position = (transition.w_position + 1.0).min(transition.w_range);
        }
    }
}

/// Advance the dimension transition animation.
pub fn dimension_transition_system(time: Res<Time>, mut transition: ResMut<DimensionTransition>) {
    if transition.progress < 1.0 {
        transition.progress = (transition.progress + time.delta_secs() * transition.speed).min(1.0);

        if transition.progress >= 1.0 {
            transition.current = transition.target;
            eprintln!(
                "[dimension] Transition complete: now in {} mode",
                transition.current.name()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_2d() {
        let t = DimensionTransition::default();
        assert_eq!(t.current, DimensionMode::D2);
        assert!(!t.transitioning());
    }

    #[test]
    fn pitch_increases_with_dimension() {
        assert!(DimensionMode::D2.camera_pitch() < DimensionMode::D2Half.camera_pitch());
        assert!(DimensionMode::D2Half.camera_pitch() < DimensionMode::D3.camera_pitch());
    }

    #[test]
    fn effective_pitch_interpolates() {
        let mut t = DimensionTransition::default();
        t.target = DimensionMode::D3;
        t.progress = 0.5;
        let pitch = t.effective_pitch();
        assert!(pitch > DimensionMode::D2.camera_pitch());
        assert!(pitch < DimensionMode::D3.camera_pitch());
    }

    #[test]
    fn w_slider_only_in_4d() {
        assert!(!DimensionMode::D2.has_w_slider());
        assert!(!DimensionMode::D3.has_w_slider());
        assert!(DimensionMode::D4.has_w_slider());
    }

    #[test]
    fn sprites_in_2d_meshes_in_3d() {
        assert!(DimensionMode::D2.uses_sprites());
        assert!(DimensionMode::D2Half.uses_sprites());
        assert!(!DimensionMode::D3.uses_sprites());
        assert!(!DimensionMode::D4.uses_sprites());
    }
}
