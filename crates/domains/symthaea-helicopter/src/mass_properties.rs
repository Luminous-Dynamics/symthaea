// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dynamic mass, center-of-gravity, and diagonal-inertia accounting.
//!
//! Payload changes must not be represented as mass-only scalars: a suspended
//! or forward payload changes trim moments and body inertia as well. This module
//! computes a deterministic diagonal approximation using the parallel-axis
//! theorem. It remains a reduced-order rigid-body model and does not represent
//! fuel slosh, flexible loads, or full products of inertia.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MassElement {
    pub id: String,
    pub mass_kg: f64,
    /// Element center in body coordinates, meters.
    pub position_body_m: [f64; 3],
    /// Intrinsic diagonal inertia about the element center, kg·m².
    pub intrinsic_inertia_kg_m2: [f64; 3],
    /// Whether an aggregate payload-drop declaration may remove this mass.
    pub droppable: bool,
}

impl MassElement {
    fn validate(&self) -> Result<(), MassPropertiesError> {
        if self.id.trim().is_empty() {
            return Err(MassPropertiesError::EmptyElementId);
        }
        if !self.mass_kg.is_finite() || self.mass_kg <= 0.0 {
            return Err(MassPropertiesError::InvalidMass);
        }
        if !self.position_body_m.iter().all(|value| value.is_finite())
            || !self
                .intrinsic_inertia_kg_m2
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        {
            return Err(MassPropertiesError::NonFiniteValue);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MassProperties {
    pub total_mass_kg: f64,
    pub center_of_gravity_body_m: [f64; 3],
    pub diagonal_inertia_about_cg_kg_m2: [f64; 3],
    pub remaining_droppable_mass_kg: f64,
    pub applied_payload_drop_kg: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MassPropertiesError {
    EmptyModel,
    EmptyElementId,
    DuplicateElementId,
    InvalidMass,
    NonFiniteValue,
    NegativePayloadDrop,
    PayloadDropExceedsAvailable,
    DegenerateInertia,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MassPropertiesModel {
    elements: Vec<MassElement>,
}

impl MassPropertiesModel {
    pub fn new(elements: Vec<MassElement>) -> Result<Self, MassPropertiesError> {
        if elements.is_empty() {
            return Err(MassPropertiesError::EmptyModel);
        }
        for element in &elements {
            element.validate()?;
        }
        for index in 0..elements.len() {
            if elements[index + 1..]
                .iter()
                .any(|other| other.id == elements[index].id)
            {
                return Err(MassPropertiesError::DuplicateElementId);
            }
        }
        let model = Self { elements };
        model.properties_with_payload_drop(0.0)?;
        Ok(model)
    }

    /// A nominal 500 kg research helicopter with a 100 kg jettisonable mission load.
    pub fn default_sar() -> Self {
        Self::new(vec![
            MassElement {
                id: "airframe".into(),
                mass_kg: 300.0,
                position_body_m: [0.0, 0.0, 0.0],
                intrinsic_inertia_kg_m2: [1_150.0, 1_200.0, 1_600.0],
                droppable: false,
            },
            MassElement {
                id: "powertrain_and_fuel".into(),
                mass_kg: 100.0,
                position_body_m: [-0.25, 0.0, 0.05],
                intrinsic_inertia_kg_m2: [180.0, 200.0, 220.0],
                droppable: false,
            },
            MassElement {
                id: "mission_payload".into(),
                mass_kg: 100.0,
                position_body_m: [0.25, 0.0, -0.55],
                intrinsic_inertia_kg_m2: [35.0, 40.0, 25.0],
                droppable: true,
            },
        ])
        .expect("default SAR mass model must remain valid")
    }

    pub fn elements(&self) -> &[MassElement] {
        &self.elements
    }

    pub fn droppable_mass_kg(&self) -> f64 {
        self.elements
            .iter()
            .filter(|element| element.droppable)
            .map(|element| element.mass_kg)
            .sum()
    }

    /// Compute properties after removing `payload_drop_kg` from droppable
    /// elements in stable declaration order. Partial removal keeps the same
    /// station, which is conservative for an aggregate legacy drop input.
    pub fn properties_with_payload_drop(
        &self,
        payload_drop_kg: f64,
    ) -> Result<MassProperties, MassPropertiesError> {
        if !payload_drop_kg.is_finite() {
            return Err(MassPropertiesError::NonFiniteValue);
        }
        if payload_drop_kg < 0.0 {
            return Err(MassPropertiesError::NegativePayloadDrop);
        }
        let available = self.droppable_mass_kg();
        if payload_drop_kg > available + 1.0e-9 {
            return Err(MassPropertiesError::PayloadDropExceedsAvailable);
        }

        let mut removal_remaining = payload_drop_kg;
        let mut retained: Vec<(&MassElement, f64)> = Vec::with_capacity(self.elements.len());
        for element in &self.elements {
            let removed = if element.droppable {
                let removed = removal_remaining.min(element.mass_kg);
                removal_remaining -= removed;
                removed
            } else {
                0.0
            };
            let retained_mass = element.mass_kg - removed;
            if retained_mass > 0.0 {
                retained.push((element, retained_mass));
            }
        }

        let total_mass_kg: f64 = retained.iter().map(|(_, mass)| *mass).sum();
        if !total_mass_kg.is_finite() || total_mass_kg <= 0.0 {
            return Err(MassPropertiesError::InvalidMass);
        }
        let mut center_of_gravity_body_m = [0.0; 3];
        for (element, mass) in &retained {
            for axis in 0..3 {
                center_of_gravity_body_m[axis] += element.position_body_m[axis] * mass;
            }
        }
        for value in &mut center_of_gravity_body_m {
            *value /= total_mass_kg;
        }

        let mut inertia = [0.0; 3];
        for (element, retained_mass) in retained {
            let retained_fraction = retained_mass / element.mass_kg;
            let dx = element.position_body_m[0] - center_of_gravity_body_m[0];
            let dy = element.position_body_m[1] - center_of_gravity_body_m[1];
            let dz = element.position_body_m[2] - center_of_gravity_body_m[2];
            inertia[0] += element.intrinsic_inertia_kg_m2[0] * retained_fraction
                + retained_mass * (dy * dy + dz * dz);
            inertia[1] += element.intrinsic_inertia_kg_m2[1] * retained_fraction
                + retained_mass * (dx * dx + dz * dz);
            inertia[2] += element.intrinsic_inertia_kg_m2[2] * retained_fraction
                + retained_mass * (dx * dx + dy * dy);
        }
        if !inertia
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err(MassPropertiesError::DegenerateInertia);
        }

        Ok(MassProperties {
            total_mass_kg,
            center_of_gravity_body_m,
            diagonal_inertia_about_cg_kg_m2: inertia,
            remaining_droppable_mass_kg: available - payload_drop_kg,
            applied_payload_drop_kg: payload_drop_kg,
        })
    }
}

impl Default for MassPropertiesModel {
    fn default() -> Self {
        Self::default_sar()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_is_500_kg_and_has_centered_low_cg() {
        let properties = MassPropertiesModel::default()
            .properties_with_payload_drop(0.0)
            .unwrap();
        assert!((properties.total_mass_kg - 500.0).abs() < 1.0e-9);
        assert!(properties.center_of_gravity_body_m[0].abs() < 1.0e-12);
        assert!(properties.center_of_gravity_body_m[2] < 0.0);
        assert_eq!(properties.remaining_droppable_mass_kg, 100.0);
    }

    #[test]
    fn payload_drop_changes_mass_cg_and_inertia() {
        let model = MassPropertiesModel::default();
        let loaded = model.properties_with_payload_drop(0.0).unwrap();
        let dropped = model.properties_with_payload_drop(75.0).unwrap();
        assert_eq!(dropped.total_mass_kg, 425.0);
        assert!(dropped.center_of_gravity_body_m[0] < loaded.center_of_gravity_body_m[0]);
        assert_ne!(
            dropped.diagonal_inertia_about_cg_kg_m2,
            loaded.diagonal_inertia_about_cg_kg_m2
        );
    }

    #[test]
    fn excessive_drop_fails_instead_of_removing_airframe_mass() {
        assert_eq!(
            MassPropertiesModel::default().properties_with_payload_drop(101.0),
            Err(MassPropertiesError::PayloadDropExceedsAvailable)
        );
    }

    #[test]
    fn parallel_axis_term_increases_offset_element_inertia() {
        let centered = MassPropertiesModel::new(vec![MassElement {
            id: "centered".into(),
            mass_kg: 10.0,
            position_body_m: [0.0; 3],
            intrinsic_inertia_kg_m2: [1.0; 3],
            droppable: false,
        }])
        .unwrap()
        .properties_with_payload_drop(0.0)
        .unwrap();
        let offset = MassPropertiesModel::new(vec![
            MassElement {
                id: "base".into(),
                mass_kg: 10.0,
                position_body_m: [0.0; 3],
                intrinsic_inertia_kg_m2: [1.0; 3],
                droppable: false,
            },
            MassElement {
                id: "offset".into(),
                mass_kg: 1.0,
                position_body_m: [1.0, 0.0, 0.0],
                intrinsic_inertia_kg_m2: [0.0; 3],
                droppable: false,
            },
        ])
        .unwrap()
        .properties_with_payload_drop(0.0)
        .unwrap();
        assert!(
            offset.diagonal_inertia_about_cg_kg_m2[1] > centered.diagonal_inertia_about_cg_kg_m2[1]
        );
        assert!(
            offset.diagonal_inertia_about_cg_kg_m2[2] > centered.diagonal_inertia_about_cg_kg_m2[2]
        );
    }
}
