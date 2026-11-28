// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic, form-specific placement for the Soma Bevy renderer.
//!
//! This module changes presentation geometry only. It does not read semantic
//! payloads, alter scene roles, change authorization, or infer confidence.
//! The current SceneElement contract does not carry graph-edge endpoints, so
//! placements are role-aware composition anchors rather than claims that a
//! connection glyph encodes a particular semantic relationship.

use bevy::prelude::{Vec2, Vec3};
use std::f32::consts::{FRAC_PI_2, PI};
use symthaea_soma::embodiment_projection::EmbodimentForm;
use symthaea_soma::embodiment_scene::{SceneElement, SceneRole};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct SceneElementPlacement {
    pub translation: Vec3,
    pub rotation_radians: f32,
}

/// Layout is a pure function of the selected form and immutable scene roles.
/// The same scene frame therefore receives the same placement on every replay.
pub(super) fn layout_scene_elements(
    form: EmbodimentForm,
    elements: &[SceneElement],
) -> Vec<SceneElementPlacement> {
    elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            let (ordinal, count) = role_ordinal(elements, index, element.role);
            let spread = centered_spread(ordinal, count);
            let (position, rotation) = role_placement(form, element.role, spread);
            SceneElementPlacement {
                translation: Vec3::new(position.x, position.y, 0.0),
                rotation_radians: rotation,
            }
        })
        .collect()
}

fn role_ordinal(elements: &[SceneElement], index: usize, role: SceneRole) -> (usize, usize) {
    let ordinal = elements[..index]
        .iter()
        .filter(|element| element.role == role)
        .count();
    let count = elements.iter().filter(|element| element.role == role).count();
    (ordinal, count)
}

fn centered_spread(ordinal: usize, count: usize) -> f32 {
    if count <= 1 {
        0.0
    } else {
        ordinal as f32 - (count.saturating_sub(1) as f32 * 0.5)
    }
}

fn polar(radius: f32, angle: f32) -> Vec2 {
    Vec2::new(angle.cos() * radius, angle.sin() * radius)
}

fn role_placement(
    form: EmbodimentForm,
    role: SceneRole,
    spread: f32,
) -> (Vec2, f32) {
    match role {
        SceneRole::Core => (Vec2::ZERO, 0.0),

        SceneRole::Connection => match form {
            EmbodimentForm::LivingConstellation => {
                (polar(34.0, -0.72 + spread * 0.22), -0.72)
            }
            EmbodimentForm::PuppetTheatre => {
                (Vec2::new(spread * 14.0, -34.0), FRAC_PI_2)
            }
            EmbodimentForm::MycelialBeing => {
                (Vec2::new(25.0 + spread * 12.0, -22.0), -0.68)
            }
            EmbodimentForm::Swarm => {
                (Vec2::new(24.0 + spread * 11.0, -16.0 + spread * 5.0), -0.45)
            }
            EmbodimentForm::LivingInstrument => {
                (Vec2::new(0.0, 30.0 + spread * 10.0), 0.0)
            }
            EmbodimentForm::HearthCompanion => {
                (Vec2::new(27.0, 18.0 + spread * 10.0), 0.52)
            }
            EmbodimentForm::CognitiveMandala => {
                (polar(36.0, -FRAC_PI_2 + spread * 0.24), -FRAC_PI_2)
            }
        },

        SceneRole::TurnState => match form {
            EmbodimentForm::LivingConstellation => {
                (polar(49.0, 0.42 + spread * 0.18), 0.42)
            }
            EmbodimentForm::PuppetTheatre => {
                (Vec2::new(spread * 12.0, 54.0), FRAC_PI_2)
            }
            EmbodimentForm::MycelialBeing => {
                (polar(52.0, -0.08 + spread * 0.18), -0.08)
            }
            EmbodimentForm::Swarm => {
                (polar(35.0, 0.15 + spread * 0.25), 0.15)
            }
            EmbodimentForm::LivingInstrument => {
                (Vec2::new(45.0, 5.0 + spread * 9.0), 0.0)
            }
            EmbodimentForm::HearthCompanion => {
                (Vec2::new(0.0, 46.0 + spread * 9.0), 0.0)
            }
            EmbodimentForm::CognitiveMandala => {
                (polar(50.0, -FRAC_PI_2 + spread * 0.2), -FRAC_PI_2)
            }
        },

        SceneRole::Backchannel => match form {
            EmbodimentForm::LivingConstellation => {
                (polar(43.0, 2.35 + spread * 0.2), 2.35)
            }
            EmbodimentForm::PuppetTheatre => {
                (Vec2::new(-30.0 + spread * 11.0, 30.0), -0.3)
            }
            EmbodimentForm::MycelialBeing => {
                (polar(43.0, 2.75 + spread * 0.2), 2.75)
            }
            EmbodimentForm::Swarm => {
                (Vec2::new(-26.0 + spread * 10.0, 20.0 + spread * 5.0), 0.4)
            }
            EmbodimentForm::LivingInstrument => {
                (Vec2::new(-42.0, -10.0 + spread * 9.0), 0.0)
            }
            EmbodimentForm::HearthCompanion => {
                (Vec2::new(-26.0, 31.0 + spread * 9.0), -0.55)
            }
            EmbodimentForm::CognitiveMandala => {
                (polar(44.0, PI + spread * 0.2), PI)
            }
        },

        SceneRole::Attention => match form {
            EmbodimentForm::LivingConstellation => {
                (polar(58.0, FRAC_PI_2 + spread * 0.18), FRAC_PI_2)
            }
            EmbodimentForm::PuppetTheatre => {
                (Vec2::new(34.0 + spread * 10.0, 58.0), 0.0)
            }
            EmbodimentForm::MycelialBeing => {
                (polar(57.0, 0.2 + spread * 0.18), 0.2)
            }
            EmbodimentForm::Swarm => {
                (polar(42.0, FRAC_PI_2 + spread * 0.22), FRAC_PI_2)
            }
            EmbodimentForm::LivingInstrument => {
                (Vec2::new(0.0, 48.0 + spread * 8.0), 0.0)
            }
            EmbodimentForm::HearthCompanion => {
                (Vec2::new(31.0, 32.0 + spread * 8.0), 0.48)
            }
            EmbodimentForm::CognitiveMandala => {
                (polar(57.0, FRAC_PI_2 + spread * 0.18), FRAC_PI_2)
            }
        },

        SceneRole::Expression => match form {
            EmbodimentForm::LivingConstellation => {
                (polar(45.0, -2.35 + spread * 0.18), -2.35)
            }
            EmbodimentForm::PuppetTheatre => {
                (Vec2::new(0.0, -52.0 + spread * 9.0), FRAC_PI_2)
            }
            EmbodimentForm::MycelialBeing => {
                (polar(45.0, -2.7 + spread * 0.18), -2.7)
            }
            EmbodimentForm::Swarm => {
                (Vec2::new(-25.0 + spread * 10.0, -25.0), -0.6)
            }
            EmbodimentForm::LivingInstrument => {
                (Vec2::new(0.0, -44.0 + spread * 9.0), 0.0)
            }
            EmbodimentForm::HearthCompanion => {
                (Vec2::new(0.0, -31.0 + spread * 8.0), 0.0)
            }
            EmbodimentForm::CognitiveMandala => {
                (polar(44.0, -FRAC_PI_2 + spread * 0.18), -FRAC_PI_2)
            }
        },

        SceneRole::Alternative => {
            let angle = -FRAC_PI_2 + spread * 0.5;
            let radius = match form {
                EmbodimentForm::LivingConstellation => 54.0,
                EmbodimentForm::PuppetTheatre => 57.0,
                EmbodimentForm::MycelialBeing => 50.0,
                EmbodimentForm::Swarm => 40.0,
                EmbodimentForm::LivingInstrument => 52.0,
                EmbodimentForm::HearthCompanion => 48.0,
                EmbodimentForm::CognitiveMandala => 56.0,
            };
            (polar(radius, angle), angle)
        }

        SceneRole::Evidence => match form {
            EmbodimentForm::LivingConstellation => (Vec2::new(-43.0, -43.0), -0.35),
            EmbodimentForm::PuppetTheatre => (Vec2::new(0.0, -66.0), 0.0),
            EmbodimentForm::MycelialBeing => (Vec2::new(-47.0, -31.0), -0.55),
            EmbodimentForm::Swarm => (Vec2::new(-34.0, -30.0), -0.6),
            EmbodimentForm::LivingInstrument => (Vec2::new(-47.0, 0.0), 0.0),
            EmbodimentForm::HearthCompanion => (Vec2::new(-33.0, -25.0), -0.2),
            EmbodimentForm::CognitiveMandala => (Vec2::new(-43.0, -43.0), -FRAC_PI_2),
        },

        SceneRole::UserControl => match form {
            EmbodimentForm::LivingConstellation => (Vec2::new(43.0, -43.0), 0.35),
            EmbodimentForm::PuppetTheatre => (Vec2::new(0.0, -68.0), 0.0),
            EmbodimentForm::MycelialBeing => (Vec2::new(47.0, -31.0), 0.55),
            EmbodimentForm::Swarm => (Vec2::new(34.0, -30.0), 0.6),
            EmbodimentForm::LivingInstrument => (Vec2::new(47.0, -18.0), 0.0),
            EmbodimentForm::HearthCompanion => (Vec2::new(33.0, -25.0), 0.2),
            EmbodimentForm::CognitiveMandala => (Vec2::new(43.0, -43.0), FRAC_PI_2),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_soma::embodiment_scene::{SceneGlyph, SceneMotion};

    fn element(role: SceneRole, glyph: SceneGlyph) -> SceneElement {
        SceneElement {
            role,
            glyph,
            weight: None,
            motion: SceneMotion::Static,
        }
    }

    #[test]
    fn core_anchor_is_stable_at_the_origin_for_all_forms() {
        let elements = [element(SceneRole::Core, SceneGlyph::StarNode)];
        for form in EmbodimentForm::ALL {
            let placements = layout_scene_elements(form, &elements);
            assert_eq!(placements.len(), 1);
            assert_eq!(placements[0].translation, Vec3::ZERO);
        }
    }

    #[test]
    fn all_seven_forms_have_distinct_connection_anchors() {
        let elements = [element(SceneRole::Connection, SceneGlyph::Filament)];
        let anchors: Vec<Vec3> = EmbodimentForm::ALL
            .iter()
            .map(|form| layout_scene_elements(*form, &elements)[0].translation)
            .collect();

        for left in 0..anchors.len() {
            for right in (left + 1)..anchors.len() {
                assert_ne!(anchors[left], anchors[right]);
            }
        }
    }

    #[test]
    fn layout_is_deterministic_and_preserves_element_order() {
        let elements = [
            element(SceneRole::Core, SceneGlyph::StarNode),
            element(SceneRole::Connection, SceneGlyph::Filament),
            element(SceneRole::Alternative, SceneGlyph::Branch),
            element(SceneRole::Alternative, SceneGlyph::Branch),
        ];
        for form in EmbodimentForm::ALL {
            let first = layout_scene_elements(form, &elements);
            let second = layout_scene_elements(form, &elements);
            assert_eq!(first, second);
            assert_eq!(first.len(), elements.len());
        }
    }

    #[test]
    fn alternative_cues_receive_separate_deterministic_positions() {
        let elements = [
            element(SceneRole::Alternative, SceneGlyph::Branch),
            element(SceneRole::Alternative, SceneGlyph::Branch),
            element(SceneRole::Alternative, SceneGlyph::Branch),
        ];
        for form in EmbodimentForm::ALL {
            let placements = layout_scene_elements(form, &elements);
            assert_ne!(placements[0].translation, placements[1].translation);
            assert_ne!(placements[1].translation, placements[2].translation);
            assert_ne!(placements[0].translation, placements[2].translation);
        }
    }

    #[test]
    fn generated_placements_are_finite() {
        let elements = [
            element(SceneRole::Core, SceneGlyph::StarNode),
            element(SceneRole::Connection, SceneGlyph::Filament),
            element(SceneRole::TurnState, SceneGlyph::Orbit),
            element(SceneRole::Backchannel, SceneGlyph::Marker),
            element(SceneRole::Attention, SceneGlyph::Branch),
            element(SceneRole::Expression, SceneGlyph::Marker),
            element(SceneRole::Alternative, SceneGlyph::Branch),
            element(SceneRole::Evidence, SceneGlyph::Marker),
            element(SceneRole::UserControl, SceneGlyph::Marker),
        ];
        for form in EmbodimentForm::ALL {
            for placement in layout_scene_elements(form, &elements) {
                assert!(placement.translation.is_finite());
                assert!(placement.rotation_radians.is_finite());
            }
        }
    }
}
