// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Renderer-neutral, deterministic 3D model for Mycelial Presence.
//!
//! This is the canonical organism geometry. SVG, Bevy/WGPU, and future renderers
//! consume the same cubic filaments and orbital cells; no renderer owns identity,
//! cognitive state, or procedural randomness.

use serde::{Deserialize, Serialize};

use crate::embodiment::{
    EmbodimentActivity, EmbodimentSnapshot, GazeTarget, Observed, SpeechOutputState, VisemeSignal,
};

pub const MYCELIAL_MODEL_VERSION: u16 = 1;
const GOLDEN_ANGLE: f32 = 2.399_963_1;
const ORBITAL_CELL_COUNT: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresenceMode {
    /// Low-distraction, ambient organism with the full filament canopy.
    Ambient,
    /// More visible structure for inspecting available runtime signals.
    Cognitive,
    /// A clearer core and shallower canopy for direct conversation.
    Conversational,
}

/// Renderer-neutral 3D point. Coordinates are organism-local world units.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpatialPoint3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl SpatialPoint3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub(crate) fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    pub(crate) fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    pub(crate) fn scale(self, factor: f32) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        }
    }

    pub(crate) fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub(crate) fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub(crate) fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub(crate) fn normalized(self) -> Self {
        let length = self.length();
        if length.is_finite() && length > 1.0e-6 {
            self.scale(1.0 / length)
        } else {
            Self {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            }
        }
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

/// One cubic Bézier hypha. A renderer can tessellate it into a tube, ribbon,
/// line strip, or projected path without changing its identity or topology.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFilament {
    pub id: u32,
    pub parent_id: Option<u32>,
    pub generation: u8,
    pub terminal: bool,
    pub start: SpatialPoint3,
    pub control_a: SpatialPoint3,
    pub control_b: SpatialPoint3,
    pub end: SpatialPoint3,
    pub radius_start: f32,
    pub radius_end: f32,
}

/// A small cell-like structure orbiting the central nucleus.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OrbitalCell {
    pub id: u16,
    pub position: SpatialPoint3,
    pub radius: f32,
}

/// Complete, portable Mycelial Presence state for any renderer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialModel {
    pub schema_version: u16,
    pub mode: PresenceMode,
    /// Missing values stay missing: renderers must not invent runtime state.
    pub activity: Option<Observed<EmbodimentActivity>>,
    pub speech_output: Option<Observed<SpeechOutputState>>,
    pub viseme: Option<Observed<VisemeSignal>>,
    pub gaze: Option<Observed<GazeTarget>>,
    pub uncertainty: Option<Observed<f32>>,
    pub filaments: Vec<MycelialFilament>,
    pub orbital_cells: Vec<OrbitalCell>,
}

impl MycelialModel {
    pub fn filament_count(&self) -> usize {
        self.filaments.len()
    }

    pub fn terminal_filament_count(&self) -> usize {
        self.filaments
            .iter()
            .filter(|filament| filament.terminal)
            .count()
    }
}

/// Build a reproducible 3D organism from the supplied semantic snapshot.
///
/// Mode changes presentation complexity only. It never changes the meaning of
/// runtime signals, and generated geometry is explicitly renderer-local.
pub fn build_mycelial_model(snapshot: &EmbodimentSnapshot, mode: PresenceMode) -> MycelialModel {
    let (primary_count, branch_depth, canopy_scale) = match mode {
        PresenceMode::Ambient => (8usize, 3u8, 1.0f32),
        PresenceMode::Cognitive => (10usize, 3u8, 0.92f32),
        PresenceMode::Conversational => (8usize, 2u8, 0.82f32),
    };

    let mut filaments = Vec::new();
    let mut next_id = 0u32;

    for index in 0..primary_count {
        let angle = index as f32 * GOLDEN_ANGLE - std::f32::consts::FRAC_PI_2;
        let z_bias = (angle * 1.7).sin() * 0.18;
        let direction = SpatialPoint3 {
            x: angle.cos(),
            y: angle.sin(),
            z: z_bias,
        }
        .normalized();
        let length = (106.0 + ((index * 29) % 23) as f32) * canopy_scale;
        append_filament(
            &mut filaments,
            &mut next_id,
            None,
            SpatialPoint3::ZERO,
            direction,
            direction,
            length,
            branch_depth,
            0,
            (index as u32).wrapping_mul(0x9E37_79B9).wrapping_add(17),
        );
    }

    let orbital_cells = (0..ORBITAL_CELL_COUNT)
        .map(|index| {
            let angle = index as f32 * std::f32::consts::TAU / ORBITAL_CELL_COUNT as f32 + 0.18;
            let radius = if index % 2 == 0 { 3.0 } else { 2.2 };
            OrbitalCell {
                id: index as u16,
                position: SpatialPoint3 {
                    x: angle.cos() * 47.0,
                    y: angle.sin() * 34.0,
                    z: (angle * 2.0).sin() * 14.0,
                },
                radius,
            }
        })
        .collect();

    MycelialModel {
        schema_version: MYCELIAL_MODEL_VERSION,
        mode,
        activity: snapshot.activity.clone(),
        speech_output: snapshot.speech_output.clone(),
        // Viseme animation is valid only while a runtime event confirms playback.
        viseme: if snapshot.current_speech_output() == Some(SpeechOutputState::Playing) {
            snapshot.viseme.clone().and_then(|mut observed| {
                if !observed.value.weight.is_finite() {
                    return None;
                }
                observed.value.weight = finite_unit(observed.value.weight);
                Some(observed)
            })
        } else {
            None
        },
        gaze: snapshot.gaze.clone(),
        uncertainty: snapshot.uncertainty.clone().map(|mut observed| {
            observed.value = finite_unit(observed.value);
            observed
        }),
        filaments,
        orbital_cells,
    }
}

#[allow(clippy::too_many_arguments)]
fn append_filament(
    output: &mut Vec<MycelialFilament>,
    next_id: &mut u32,
    parent_id: Option<u32>,
    start: SpatialPoint3,
    direction: SpatialPoint3,
    incoming_tangent: SpatialPoint3,
    length: f32,
    remaining_depth: u8,
    generation: u8,
    seed: u32,
) {
    let direction = direction.normalized();
    let reference = if direction.z.abs() > 0.82 {
        SpatialPoint3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    } else {
        SpatialPoint3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    };
    let side = direction.cross(reference).normalized();
    let lift = side.cross(direction).normalized();
    let bend_a = signed_noise(seed.wrapping_mul(31).wrapping_add(7));
    let bend_b = signed_noise(seed.wrapping_mul(67).wrapping_add(19));
    let lateral = side
        .scale(bend_a * length * 0.075)
        .add(lift.scale(bend_b * length * 0.052));

    let end = start.add(direction.scale(length)).add(lateral.scale(1.08));
    // Keep child branches tangent-continuous at the fork, then let them
    // diverge along their own deterministic direction.
    let start_tangent = if parent_id.is_some() {
        incoming_tangent.normalized()
    } else {
        direction
    };
    let control_a = start.add(start_tangent.scale(length * 0.28));
    let control_b = start
        .add(direction.scale(length * 0.76))
        .add(lateral.scale(0.88));

    let end_tangent = end.sub(control_b).normalized();
    let id = *next_id;
    *next_id = (*next_id).wrapping_add(1);
    let radius_start = 2.8 * 0.66f32.powi(generation as i32);
    output.push(MycelialFilament {
        id,
        parent_id,
        generation,
        terminal: remaining_depth == 0,
        start,
        control_a,
        control_b,
        end,
        radius_start,
        radius_end: radius_start * 0.74,
    });

    if remaining_depth == 0 {
        return;
    }

    let child_length = length * 0.655;
    let fork = 0.44 + (seed % 5) as f32 * 0.018;
    for (side_sign, child_slot) in [(-1.0f32, 1u32), (1.0f32, 2u32)] {
        let child_direction = direction
            .scale(fork.cos())
            .add(side.scale(side_sign * fork.sin()))
            .add(lift.scale(signed_noise(seed ^ child_slot.wrapping_mul(0xA511_E9B3)) * 0.10))
            .normalized();
        append_filament(
            output,
            next_id,
            Some(id),
            end,
            child_direction,
            end_tangent,
            child_length,
            remaining_depth - 1,
            generation + 1,
            seed.wrapping_mul(1664525)
                .wrapping_add(1013904223 ^ child_slot),
        );
    }
}

fn signed_noise(seed: u32) -> f32 {
    let mut value = seed;
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^= value >> 16;
    (value as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
}

fn finite_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment::{Observed, SignalProvenance, SpeechOutputState};

    fn observed<T>(value: T, sequence: u64) -> Observed<T> {
        Observed {
            value,
            provenance: SignalProvenance::RuntimeEvent,
            sequence,
            monotonic_ns: sequence * 1_000,
        }
    }

    #[test]
    fn model_is_deterministic_and_has_a_true_spatial_canopy() {
        let snapshot = EmbodimentSnapshot::default();
        let first = build_mycelial_model(&snapshot, PresenceMode::Ambient);
        let second = build_mycelial_model(&snapshot, PresenceMode::Ambient);
        assert_eq!(first, second);
        assert_eq!(first.schema_version, MYCELIAL_MODEL_VERSION);
        assert_eq!(first.filament_count(), 8 * 15);
        assert_eq!(first.terminal_filament_count(), 8 * 8);
        assert_eq!(first.orbital_cells.len(), ORBITAL_CELL_COUNT);
        assert!(first.filaments.iter().any(|filament| {
            filament.start.z != 0.0
                || filament.control_a.z != 0.0
                || filament.control_b.z != 0.0
                || filament.end.z != 0.0
        }));
    }

    #[test]
    fn every_curve_is_finite_bounded_and_has_a_valid_parent_order() {
        let model = build_mycelial_model(&EmbodimentSnapshot::default(), PresenceMode::Cognitive);
        let mut seen = std::collections::HashSet::new();
        for filament in &model.filaments {
            assert!(seen.insert(filament.id));
            assert!(filament.start.is_finite());
            assert!(filament.control_a.is_finite());
            assert!(filament.control_b.is_finite());
            assert!(filament.end.is_finite());
            assert!(filament.radius_start.is_finite() && filament.radius_start > 0.0);
            assert!(filament.radius_end.is_finite() && filament.radius_end > 0.0);
            if let Some(parent_id) = filament.parent_id {
                assert!(seen.contains(&parent_id), "parent must precede child");
            }
            for point in [
                filament.start,
                filament.control_a,
                filament.control_b,
                filament.end,
            ] {
                assert!(point.length() < 700.0, "unbounded point: {point:?}");
            }
        }
    }

    #[test]
    fn modes_change_complexity_without_changing_the_event_snapshot() {
        let mut snapshot = EmbodimentSnapshot::default();
        snapshot.speech_output = Some(observed(SpeechOutputState::Playing, 4));
        let ambient = build_mycelial_model(&snapshot, PresenceMode::Ambient);
        let cognitive = build_mycelial_model(&snapshot, PresenceMode::Cognitive);
        let conversational = build_mycelial_model(&snapshot, PresenceMode::Conversational);
        assert!(cognitive.filament_count() > ambient.filament_count());
        assert!(ambient.filament_count() > conversational.filament_count());
        assert_eq!(ambient.speech_output, cognitive.speech_output);
        assert_eq!(ambient.speech_output, conversational.speech_output);
    }

    #[test]
    fn viseme_is_not_forwarded_without_confirmed_playback() {
        use crate::embodiment::{Viseme, VisemeSignal};
        let mut snapshot = EmbodimentSnapshot::default();
        snapshot.viseme = Some(observed(
            VisemeSignal {
                viseme: Viseme::Aa,
                weight: 0.8,
            },
            2,
        ));
        assert!(
            build_mycelial_model(&snapshot, PresenceMode::Conversational)
                .viseme
                .is_none()
        );
        snapshot.speech_output = Some(observed(SpeechOutputState::Playing, 3));
        assert!(
            build_mycelial_model(&snapshot, PresenceMode::Conversational)
                .viseme
                .is_some()
        );
    }
    #[test]
    fn child_filaments_begin_at_parent_endpoint_with_matching_tangent() {
        let model = build_mycelial_model(&EmbodimentSnapshot::default(), PresenceMode::Ambient);
        for child in model
            .filaments
            .iter()
            .filter(|filament| filament.parent_id.is_some())
        {
            let parent_id = child.parent_id.expect("filtered child");
            let parent = model
                .filaments
                .iter()
                .find(|filament| filament.id == parent_id)
                .expect("parent exists");
            let gap = child.start.sub(parent.end).length();
            assert!(gap < 1.0e-5, "child root must equal parent endpoint");
            let parent_tangent = parent.end.sub(parent.control_b).normalized();
            let child_tangent = child.control_a.sub(child.start).normalized();
            assert!(
                parent_tangent.dot(child_tangent) > 0.9999,
                "parent and child tangents should be continuous at fork"
            );
        }
    }

    #[test]
    fn playing_viseme_weight_is_bounded_and_nonfinite_weights_are_dropped() {
        use crate::embodiment::{Viseme, VisemeSignal};
        let mut snapshot = EmbodimentSnapshot::default();
        snapshot.speech_output = Some(observed(SpeechOutputState::Playing, 1));
        snapshot.viseme = Some(observed(
            VisemeSignal {
                viseme: Viseme::Ee,
                weight: 1.7,
            },
            2,
        ));
        let model = build_mycelial_model(&snapshot, PresenceMode::Ambient);
        assert_eq!(
            model.viseme.as_ref().map(|value| value.value.weight),
            Some(1.0)
        );

        snapshot.viseme = Some(observed(
            VisemeSignal {
                viseme: Viseme::Ee,
                weight: f32::NAN,
            },
            3,
        ));
        assert!(
            build_mycelial_model(&snapshot, PresenceMode::Ambient)
                .viseme
                .is_none()
        );
    }
}
