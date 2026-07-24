// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed multi-contact support for feet, hands, knees, and protective impacts.

use serde::{Deserialize, Serialize};

use crate::contact::{ContactFrame, ContactSource};
use crate::types::HumanoidState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContactSite {
    RightFoot,
    LeftFoot,
    RightHand,
    LeftHand,
    RightKnee,
    LeftKnee,
    RightForearm,
    LeftForearm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialContact {
    pub site: ContactSite,
    pub active: bool,
    pub point_world_m: [f64; 3],
    pub force_world_n: [f64; 3],
    pub torque_world_nm: [f64; 3],
    pub normal_world: [f64; 3],
    pub friction: f64,
    pub confidence: f64,
    pub protective: bool,
}

impl SpatialContact {
    pub fn validate(&self) -> bool {
        self.point_world_m.iter().all(|value| value.is_finite())
            && self.force_world_n.iter().all(|value| value.is_finite())
            && self.torque_world_nm.iter().all(|value| value.is_finite())
            && self.normal_world.iter().all(|value| value.is_finite())
            && norm3(self.normal_world) > 1.0e-9
            && self.friction.is_finite()
            && self.friction >= 0.0
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiContactFrame {
    pub timestamp: f64,
    pub source: ContactSource,
    pub contacts: Vec<SpatialContact>,
}

impl MultiContactFrame {
    pub fn from_feet(frame: &ContactFrame) -> Self {
        let contacts = [
            (ContactSite::RightFoot, frame.right),
            (ContactSite::LeftFoot, frame.left),
        ]
        .into_iter()
        .map(|(site, foot)| SpatialContact {
            site,
            active: foot.in_contact,
            point_world_m: foot.point_world_m,
            force_world_n: foot.force_world_n,
            torque_world_nm: foot.torque_world_nm,
            normal_world: [0.0, 0.0, 1.0],
            friction: 1.0,
            confidence: foot.confidence,
            protective: false,
        })
        .collect();
        Self {
            timestamp: frame.timestamp,
            source: frame.source,
            contacts,
        }
    }

    pub fn validate(&self) -> bool {
        self.timestamp.is_finite()
            && !self.contacts.is_empty()
            && self.contacts.iter().all(SpatialContact::validate)
            && self.contacts.iter().enumerate().all(|(index, contact)| {
                self.contacts[..index]
                    .iter()
                    .all(|earlier| earlier.site != contact.site)
            })
    }

    pub fn active(&self) -> impl Iterator<Item = &SpatialContact> {
        self.contacts.iter().filter(|contact| contact.active)
    }

    pub fn active_count(&self) -> usize {
        self.active().count()
    }

    pub fn has_upper_body_support(&self) -> bool {
        self.active().any(|contact| {
            matches!(
                contact.site,
                ContactSite::RightHand
                    | ContactSite::LeftHand
                    | ContactSite::RightForearm
                    | ContactSite::LeftForearm
            )
        })
    }

    pub fn has_knee_support(&self) -> bool {
        self.active()
            .any(|contact| matches!(contact.site, ContactSite::RightKnee | ContactSite::LeftKnee))
    }

    pub fn total_normal_force_n(&self) -> f64 {
        self.active()
            .map(|contact| dot3(contact.force_world_n, normalize3(contact.normal_world)).max(0.0))
            .sum()
    }

    pub fn minimum_active_confidence(&self) -> f64 {
        let mut confidence = 1.0f64;
        let mut any = false;
        for contact in self.active() {
            any = true;
            confidence = confidence.min(contact.confidence);
        }
        if any { confidence } else { 0.0 }
    }

    /// Convex-hull area of active contact points projected onto the ground.
    pub fn support_polygon_area_m2(&self) -> f64 {
        let mut points = self
            .active()
            .map(|contact| [contact.point_world_m[0], contact.point_world_m[1]])
            .collect::<Vec<_>>();
        if points.len() < 3 {
            return 0.0;
        }
        points.sort_by(|left, right| {
            left[0]
                .partial_cmp(&right[0])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left[1]
                        .partial_cmp(&right[1])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        let mut lower: Vec<[f64; 2]> = Vec::new();
        for point in &points {
            while lower.len() >= 2
                && cross2(lower[lower.len() - 2], lower[lower.len() - 1], *point) <= 0.0
            {
                lower.pop();
            }
            lower.push(*point);
        }
        let mut upper: Vec<[f64; 2]> = Vec::new();
        for point in points.iter().rev() {
            while upper.len() >= 2
                && cross2(upper[upper.len() - 2], upper[upper.len() - 1], *point) <= 0.0
            {
                upper.pop();
            }
            upper.push(*point);
        }
        lower.pop();
        upper.pop();
        lower.extend(upper);
        polygon_area(&lower)
    }

    /// Conservative protective contact candidates inferred from state geometry.
    /// These are planned contacts, not measured contact truth.
    pub fn with_protective_candidates(mut self, state: &HumanoidState) -> Self {
        if state.uprightness() >= 0.72 {
            return self;
        }
        let extremity = |offset: usize| -> [f64; 3] {
            if state.extremities.len() >= offset + 3 {
                [
                    state.extremities[offset],
                    state.extremities[offset + 1],
                    state.extremities[offset + 2],
                ]
            } else {
                state.root_position
            }
        };
        for (site, point) in [
            (ContactSite::RightHand, extremity(0)),
            (ContactSite::LeftHand, extremity(3)),
        ] {
            if self.contacts.iter().all(|contact| contact.site != site) {
                self.contacts.push(SpatialContact {
                    site,
                    active: false,
                    point_world_m: point,
                    force_world_n: [0.0; 3],
                    torque_world_nm: [0.0; 3],
                    normal_world: [0.0, 0.0, 1.0],
                    friction: 0.7,
                    confidence: 0.45,
                    protective: true,
                });
            }
        }
        self
    }
}

fn norm3(vector: [f64; 3]) -> f64 {
    dot3(vector, vector).sqrt()
}

fn normalize3(vector: [f64; 3]) -> [f64; 3] {
    let norm = norm3(vector);
    if norm <= 1.0e-12 {
        [0.0, 0.0, 1.0]
    } else {
        [vector[0] / norm, vector[1] / norm, vector[2] / norm]
    }
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross2(origin: [f64; 2], left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])
}

fn polygon_area(points: &[[f64; 2]]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut twice_area = 0.0;
    for index in 0..points.len() {
        let next = (index + 1) % points.len();
        twice_area += points[index][0] * points[next][1] - points[next][0] * points[index][1];
    }
    0.5 * twice_area.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;

    #[test]
    fn duplicate_sites_are_rejected() {
        let state = HumanoidState::default_for(HumanoidMorphology::Dmc21);
        let feet = ContactFrame::estimated_from_state(&state, 0.05);
        let mut multi = MultiContactFrame::from_feet(&feet);
        multi.contacts.push(multi.contacts[0].clone());
        assert!(!multi.validate());
    }

    #[test]
    fn protective_candidates_are_not_reported_as_active_truth() {
        let mut state = HumanoidState::default_for(HumanoidMorphology::Dmc21);
        state.torso_vertical = [1.0, 0.0, 0.0];
        let feet = ContactFrame::estimated_from_state(&state, 0.05);
        let multi = MultiContactFrame::from_feet(&feet).with_protective_candidates(&state);
        assert!(multi.contacts.iter().any(|contact| contact.protective));
        assert!(
            multi
                .contacts
                .iter()
                .filter(|contact| contact.protective)
                .all(|contact| !contact.active)
        );
    }
}
