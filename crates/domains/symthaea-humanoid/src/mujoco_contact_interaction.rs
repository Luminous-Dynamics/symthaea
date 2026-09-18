// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MuJoCo contact-pair interaction evidence for qualified humanoid support patches.
//!
//! This module extracts the *resolved contact pair* limits carried by MuJoCo's
//! live `mjContact` records. Immutable support geometry is owned by
//! `mujoco_contact_patch`; contact-pair/material interaction is a separate,
//! time-bound proposition.

use std::sync::Arc;

use mujoco_rs::prelude::*;
use serde::{Deserialize, Serialize};

use crate::contact_wrench::{ContactInteractionLimitSource, ContactInteractionLimitsV1};
use crate::mujoco_contact_patch::MuJoCoFootPatchSetV1;
use crate::multi_contact::ContactSite;

const DEFAULT_FRICTION_MATCH_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MuJoCoContactDimensionalityV1 {
    NormalOnly,
    Tangential,
    Torsional,
    Rolling,
}

impl MuJoCoContactDimensionalityV1 {
    pub const fn from_dim(dim: i32) -> Option<Self> {
        match dim {
            1 => Some(Self::NormalOnly),
            3 => Some(Self::Tangential),
            4 => Some(Self::Torsional),
            6 => Some(Self::Rolling),
            _ => None,
        }
    }

    pub const fn dim(self) -> i32 {
        match self {
            Self::NormalOnly => 1,
            Self::Tangential => 3,
            Self::Torsional => 4,
            Self::Rolling => 6,
        }
    }

    pub const fn grants_sliding(self) -> bool {
        !matches!(self, Self::NormalOnly)
    }

    pub const fn grants_torsion(self) -> bool {
        matches!(self, Self::Torsional | Self::Rolling)
    }

    pub const fn grants_rolling(self) -> bool {
        matches!(self, Self::Rolling)
    }
}

/// Conservative projection used while the downstream interaction contract owns
/// only one scalar sliding coefficient. Raw anisotropic MuJoCo coefficients are
/// retained separately in the evidence record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlidingFrictionProjectionV1 {
    MinimumResolvedTangential,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MuJoCoContactInteractionRecordV1 {
    pub site: ContactSite,
    pub model_id: String,
    pub model_signature: u64,
    pub sampled_at_s: f64,
    pub foot_geom_id: usize,
    pub foot_geom_name: String,
    pub environment_geom_id: usize,
    pub environment_geom_name: String,
    /// Exact indices into the live `MjData::contact()` slice that contributed.
    pub contact_indices: Vec<usize>,
    pub dimensionality: MuJoCoContactDimensionalityV1,
    /// Resolved MuJoCo contact friction `[tangent1, tangent2, spin, roll1, roll2]`.
    pub resolved_friction: [f64; 5],
    pub sliding_projection: SlidingFrictionProjectionV1,
    pub limits: ContactInteractionLimitsV1,
}

impl MuJoCoContactInteractionRecordV1 {
    pub fn validate(&self) -> bool {
        !self.model_id.trim().is_empty()
            && self.sampled_at_s.is_finite()
            && self.sampled_at_s >= 0.0
            && !self.foot_geom_name.trim().is_empty()
            && !self.environment_geom_name.trim().is_empty()
            && !self.contact_indices.is_empty()
            && self
                .contact_indices
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            && self
                .resolved_friction
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && self.limits.validate()
            && self.limits.site == self.site
            && self.limits.source == ContactInteractionLimitSource::SimulatorContactPair
            && same_sample_time(self.limits.sampled_at_s, self.sampled_at_s)
            && (!self.dimensionality.grants_sliding()
                || self.limits.sliding_friction_coefficient
                    <= self.resolved_friction[0].min(self.resolved_friction[1]) + 1.0e-12)
            && (self.dimensionality.grants_sliding()
                || self.limits.sliding_friction_coefficient == 0.0)
            && (self.dimensionality.grants_torsion()
                || self.limits.torsional_friction_radius_m == 0.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MuJoCoContactInteractionError {
    EmptyModelIdentity,
    ModelMismatch,
    PatchSetMismatch,
    UnsupportedSite,
    MissingFootPatchRecord,
    NoActiveContact,
    InvalidContactGeom,
    MissingEnvironmentName,
    AmbiguousEnvironmentGeom,
    UnsupportedContactDimensionality(i32),
    InvalidResolvedFriction,
    InconsistentContactDimensionality,
    InconsistentInteractionLimits,
    InvalidResult,
}

#[derive(Debug, Clone, Copy)]
struct ContactSample {
    index: usize,
    environment_geom_id: usize,
    dim: i32,
    friction: [f64; 5],
}

/// Extract conservative interaction limits for one mapped foot from MuJoCo's
/// currently active constraint contacts.
///
/// Only contacts whose `exclude == 0` and `efc_address >= 0` are admitted. The
/// foot geometry identity must come from the same exact 001C patch set and the
/// patch-set sample time must match the current MuJoCo data time.
pub fn extract_mujoco_contact_interaction(
    model: &MjModel,
    data: &MjData<Arc<MjModel>>,
    patches: &MuJoCoFootPatchSetV1,
    site: ContactSite,
) -> Result<MuJoCoContactInteractionRecordV1, MuJoCoContactInteractionError> {
    if patches.model_id.trim().is_empty() {
        return Err(MuJoCoContactInteractionError::EmptyModelIdentity);
    }
    if model.signature() != data.model().signature() {
        return Err(MuJoCoContactInteractionError::ModelMismatch);
    }
    if patches.model_signature != model.signature()
        || !same_sample_time(patches.sampled_at_s, data.time())
    {
        return Err(MuJoCoContactInteractionError::PatchSetMismatch);
    }

    if !matches!(site, ContactSite::RightFoot | ContactSite::LeftFoot) {
        return Err(MuJoCoContactInteractionError::UnsupportedSite);
    }
    let patch_record = patches
        .record_for_site(site)
        .ok_or(MuJoCoContactInteractionError::MissingFootPatchRecord)?;
    let foot_geom_id = patch_record.geom_id;

    let mut samples = Vec::new();
    for (index, contact) in data.contact().iter().enumerate() {
        if contact.exclude != 0 || contact.efc_address < 0 {
            continue;
        }
        let foot_is_first = contact.geom1 == foot_geom_id as i32;
        let foot_is_second = contact.geom2 == foot_geom_id as i32;
        if !foot_is_first && !foot_is_second {
            continue;
        }
        let environment_raw = if foot_is_first {
            contact.geom2
        } else {
            contact.geom1
        };
        if environment_raw < 0 {
            return Err(MuJoCoContactInteractionError::InvalidContactGeom);
        }
        samples.push(ContactSample {
            index,
            environment_geom_id: environment_raw as usize,
            dim: contact.dim,
            friction: contact.friction,
        });
    }

    let (environment_geom_id, contact_indices, dimensionality, friction) =
        aggregate_samples(&samples, DEFAULT_FRICTION_MATCH_TOLERANCE)?;
    let environment_geom_name = model
        .id_to_name(MjtObj::mjOBJ_GEOM, environment_geom_id)
        .ok_or(MuJoCoContactInteractionError::MissingEnvironmentName)?
        .to_string();

    let sliding_friction_coefficient = if dimensionality.grants_sliding() {
        friction[0].min(friction[1])
    } else {
        0.0
    };
    let torsional_friction_radius_m = if dimensionality.grants_torsion() {
        friction[2]
    } else {
        0.0
    };
    let sampled_at_s = data.time();
    let interaction_id = format!(
        "{}:sig:{:016x}:site:{site:?}:foot:{}:{}:env:{}:{}:dim:{}:friction:{:.17e}:{:.17e}:{:.17e}:{:.17e}:{:.17e}:projection:min-tangent-v1",
        patches.model_id,
        model.signature(),
        patch_record.physical_geom_name,
        foot_geom_id,
        environment_geom_name,
        environment_geom_id,
        dimensionality.dim(),
        friction[0],
        friction[1],
        friction[2],
        friction[3],
        friction[4],
    );
    let limits = ContactInteractionLimitsV1 {
        site,
        sliding_friction_coefficient,
        torsional_friction_radius_m,
        source: ContactInteractionLimitSource::SimulatorContactPair,
        interaction_id,
        sampled_at_s,
        confidence: 1.0,
    };

    let result = MuJoCoContactInteractionRecordV1 {
        site,
        model_id: patches.model_id.clone(),
        model_signature: model.signature(),
        sampled_at_s,
        foot_geom_id,
        foot_geom_name: patch_record.physical_geom_name.clone(),
        environment_geom_id,
        environment_geom_name,
        contact_indices,
        dimensionality,
        resolved_friction: friction,
        sliding_projection: SlidingFrictionProjectionV1::MinimumResolvedTangential,
        limits,
    };
    if result.validate() {
        Ok(result)
    } else {
        Err(MuJoCoContactInteractionError::InvalidResult)
    }
}

fn aggregate_samples(
    samples: &[ContactSample],
    friction_tolerance: f64,
) -> Result<(usize, Vec<usize>, MuJoCoContactDimensionalityV1, [f64; 5]), MuJoCoContactInteractionError>
{
    let first = samples
        .first()
        .ok_or(MuJoCoContactInteractionError::NoActiveContact)?;
    if !friction_tolerance.is_finite() || friction_tolerance < 0.0 {
        return Err(MuJoCoContactInteractionError::InvalidResolvedFriction);
    }
    let dimensionality = MuJoCoContactDimensionalityV1::from_dim(first.dim)
        .ok_or(MuJoCoContactInteractionError::UnsupportedContactDimensionality(first.dim))?;
    validate_friction(first.friction)?;

    let mut indices = Vec::with_capacity(samples.len());
    for sample in samples {
        if sample.environment_geom_id != first.environment_geom_id {
            return Err(MuJoCoContactInteractionError::AmbiguousEnvironmentGeom);
        }
        if sample.dim != first.dim {
            return Err(MuJoCoContactInteractionError::InconsistentContactDimensionality);
        }
        MuJoCoContactDimensionalityV1::from_dim(sample.dim)
            .ok_or(MuJoCoContactInteractionError::UnsupportedContactDimensionality(sample.dim))?;
        validate_friction(sample.friction)?;
        if !friction_matches(first.friction, sample.friction, friction_tolerance) {
            return Err(MuJoCoContactInteractionError::InconsistentInteractionLimits);
        }
        indices.push(sample.index);
    }
    indices.sort_unstable();
    indices.dedup();

    Ok((
        first.environment_geom_id,
        indices,
        dimensionality,
        first.friction,
    ))
}

fn validate_friction(friction: [f64; 5]) -> Result<(), MuJoCoContactInteractionError> {
    if friction
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    {
        Ok(())
    } else {
        Err(MuJoCoContactInteractionError::InvalidResolvedFriction)
    }
}

fn friction_matches(left: [f64; 5], right: [f64; 5], tolerance: f64) -> bool {
    left.into_iter()
        .zip(right)
        .all(|(left, right)| (left - right).abs() <= tolerance)
}

fn same_sample_time(left: f64, right: f64) -> bool {
    let tolerance = 1.0e-9 * (1.0 + left.abs().max(right.abs()));
    (left - right).abs() <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::mujoco_contact_patch::extract_mujoco_foot_patch_set;
    use crate::simulator::{HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator};
    use crate::types::HumanoidCommand;

    const MODEL_ID: &str = "generated-dmc21-contact-interaction-v1";

    fn sample(index: usize, env: usize, dim: i32, friction: [f64; 5]) -> ContactSample {
        ContactSample {
            index,
            environment_geom_id: env,
            dim,
            friction,
        }
    }

    #[test]
    fn dimensionality_grants_only_declared_authority() {
        assert!(!MuJoCoContactDimensionalityV1::NormalOnly.grants_sliding());
        assert!(MuJoCoContactDimensionalityV1::Tangential.grants_sliding());
        assert!(!MuJoCoContactDimensionalityV1::Tangential.grants_torsion());
        assert!(MuJoCoContactDimensionalityV1::Torsional.grants_torsion());
        assert!(!MuJoCoContactDimensionalityV1::Torsional.grants_rolling());
        assert!(MuJoCoContactDimensionalityV1::Rolling.grants_rolling());
    }

    #[test]
    fn multiple_consistent_contacts_retain_all_indices() {
        let friction = [0.8, 0.7, 0.03, 0.01, 0.01];
        let samples = [sample(2, 7, 4, friction), sample(5, 7, 4, friction)];
        let (_, indices, dim, resolved) = aggregate_samples(&samples, 1.0e-12).unwrap();
        assert_eq!(indices, vec![2, 5]);
        assert_eq!(dim, MuJoCoContactDimensionalityV1::Torsional);
        assert_eq!(resolved, friction);
    }

    #[test]
    fn differing_environment_geoms_fail_closed() {
        let friction = [0.8, 0.8, 0.03, 0.0, 0.0];
        let samples = [sample(0, 7, 4, friction), sample(1, 8, 4, friction)];
        assert_eq!(
            aggregate_samples(&samples, 1.0e-12),
            Err(MuJoCoContactInteractionError::AmbiguousEnvironmentGeom)
        );
    }

    #[test]
    fn inconsistent_resolved_friction_fails_closed() {
        let first = [0.8, 0.8, 0.03, 0.0, 0.0];
        let second = [0.7, 0.7, 0.03, 0.0, 0.0];
        let samples = [sample(0, 7, 4, first), sample(1, 7, 4, second)];
        assert_eq!(
            aggregate_samples(&samples, 1.0e-12),
            Err(MuJoCoContactInteractionError::InconsistentInteractionLimits)
        );
    }

    #[test]
    fn unsupported_contact_dimensionality_fails_closed() {
        let samples = [sample(0, 7, 2, [0.8, 0.8, 0.0, 0.0, 0.0])];
        assert_eq!(
            aggregate_samples(&samples, 1.0e-12),
            Err(MuJoCoContactInteractionError::UnsupportedContactDimensionality(2))
        );
    }

    #[test]
    fn generated_default_contact_is_tangential_and_grants_no_torsion() {
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(HumanoidMorphology::Dmc21).unwrap();
        // Ensure current forward/constraint state is populated before extracting contacts.
        sim.step(&HumanoidCommand::zero(), 0.0);
        let model = Arc::clone(sim.model_arc());
        let patches =
            extract_mujoco_foot_patch_set(model.as_ref(), sim.data_mut(), MODEL_ID).unwrap();

        let mut extracted = Vec::new();
        for site in [ContactSite::RightFoot, ContactSite::LeftFoot] {
            match extract_mujoco_contact_interaction(
                model.as_ref(),
                sim.data_mut(),
                &patches,
                site,
            ) {
                Ok(record) => extracted.push(record),
                Err(MuJoCoContactInteractionError::NoActiveContact) => {}
                Err(error) => panic!("unexpected MuJoCo interaction extraction failure: {error:?}"),
            }
        }
        assert!(!extracted.is_empty(), "generated standing humanoid should have at least one active foot contact");
        for record in extracted {
            assert_eq!(record.environment_geom_name, "floor");
            assert_eq!(record.dimensionality, MuJoCoContactDimensionalityV1::Tangential);
            assert!(record.limits.sliding_friction_coefficient > 0.0);
            assert_eq!(record.limits.torsional_friction_radius_m, 0.0);
            assert!(record.resolved_friction[2] > 0.0);
            assert!(!record.contact_indices.is_empty());
            assert_eq!(
                record.limits.source,
                ContactInteractionLimitSource::SimulatorContactPair
            );
        }
    }

    #[test]
    fn stale_patch_set_time_is_rejected() {
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(HumanoidMorphology::Dmc21).unwrap();
        let model = Arc::clone(sim.model_arc());
        let patches =
            extract_mujoco_foot_patch_set(model.as_ref(), sim.data_mut(), MODEL_ID).unwrap();
        sim.step(&HumanoidCommand::zero(), 0.0);
        assert_eq!(
            extract_mujoco_contact_interaction(
                model.as_ref(),
                sim.data_mut(),
                &patches,
                ContactSite::RightFoot,
            ),
            Err(MuJoCoContactInteractionError::PatchSetMismatch)
        );
    }
}
