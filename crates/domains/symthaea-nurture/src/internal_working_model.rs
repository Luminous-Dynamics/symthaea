// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Internal Working Model (IWM) — Bowlby's representational system.
//!
//! The IWM encodes three core beliefs (self_worthy, other_reliable, world_safe)
//! both as scalar values and as a 16,384D hypervector. Learning rate is modulated
//! by critical period plasticity: high during infancy, declining with age.
//!
//! Science: Bowlby (1969), Bretherton & Munholland (1999).

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::types::{CaregiverAction, InternalWorkingModel};

/// Encode the three IWM dimensions into a 16,384D hypervector.
///
/// Uses three orthogonal basis vectors (seeded deterministically) bound with
/// scalar-scaled copies, then bundled into a single representation.
pub fn encode_working_model(model: &InternalWorkingModel) -> ContinuousHV {
    let self_basis = ContinuousHV::random(HDC_DIMENSION, 1001);
    let other_basis = ContinuousHV::random(HDC_DIMENSION, 1002);
    let world_basis = ContinuousHV::random(HDC_DIMENSION, 1003);

    let self_hv = self_basis.scale(model.self_worthy as f32);
    let other_hv = other_basis.scale(model.other_reliable as f32);
    let world_hv = world_basis.scale(model.world_safe as f32);

    ContinuousHV::bundle(&[&self_hv, &other_hv, &world_hv]).normalize()
}

/// Update the IWM based on a caregiver interaction.
///
/// Learning rate is modulated by `plasticity` (critical period gating):
/// - High plasticity (early infancy): rapid belief updating
/// - Low plasticity (post-critical period): slow, resistant to change
///
/// Each dimension moves toward the observed caregiver quality:
/// - `self_worthy` <- warmth (warm care signals "I am worthy")
/// - `other_reliable` <- responsiveness (consistent response signals "others are reliable")
/// - `world_safe` <- overall quality (good care signals "world is safe")
pub fn update_working_model(
    model: &mut InternalWorkingModel,
    action: &CaregiverAction,
    plasticity: f64,
) {
    let lr = 0.01 * plasticity;

    model.self_worthy += lr * (action.warmth - model.self_worthy);
    model.other_reliable += lr * (action.responsiveness - model.other_reliable);
    model.world_safe += lr * (action.quality() - model.world_safe);

    // Clamp to [0, 1]
    model.self_worthy = model.self_worthy.clamp(0.0, 1.0);
    model.other_reliable = model.other_reliable.clamp(0.0, 1.0);
    model.world_safe = model.world_safe.clamp(0.0, 1.0);

    // Update HDC encoding
    model.model_hv = encode_working_model(model);
}

/// Create a default (neutral) internal working model.
pub fn new_working_model() -> InternalWorkingModel {
    let model = InternalWorkingModel {
        self_worthy: 0.5,
        other_reliable: 0.5,
        world_safe: 0.5,
        model_hv: ContinuousHV::zero(HDC_DIMENSION),
    };
    InternalWorkingModel {
        model_hv: encode_working_model(&model),
        ..model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_working_model_nonzero() {
        let model = new_working_model();
        let norm = model.model_hv.norm();
        assert!(
            norm > 0.5,
            "Encoded model HV should be normalized, got {norm}"
        );
    }

    #[test]
    fn test_encode_different_models_differ() {
        let m1 = InternalWorkingModel {
            self_worthy: 0.9,
            other_reliable: 0.9,
            world_safe: 0.9,
            model_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let m2 = InternalWorkingModel {
            self_worthy: 0.1,
            other_reliable: 0.9,
            world_safe: 0.5,
            model_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let hv1 = encode_working_model(&m1);
        let hv2 = encode_working_model(&m2);
        // Same basis vectors but different scales -> different HVs
        let sim = hv1.similarity(&hv2);
        assert!(
            sim < 0.99,
            "Different IWMs should produce different HVs, sim={sim}"
        );
    }

    #[test]
    fn test_update_toward_warmth() {
        let mut model = new_working_model();
        let warm_action = CaregiverAction::reliable();
        for _ in 0..500 {
            update_working_model(&mut model, &warm_action, 1.0);
        }
        assert!(
            model.self_worthy > 0.8,
            "self_worthy should approach warmth=0.85, got {}",
            model.self_worthy
        );
        assert!(
            model.other_reliable > 0.8,
            "other_reliable should approach responsiveness=0.9, got {}",
            model.other_reliable
        );
    }

    #[test]
    fn test_update_toward_neglect() {
        let mut model = new_working_model();
        let cold_action = CaregiverAction::unresponsive();
        for _ in 0..500 {
            update_working_model(&mut model, &cold_action, 1.0);
        }
        assert!(
            model.self_worthy < 0.3,
            "self_worthy should approach warmth=0.2, got {}",
            model.self_worthy
        );
        assert!(
            model.other_reliable < 0.2,
            "other_reliable should approach responsiveness=0.1, got {}",
            model.other_reliable
        );
    }

    #[test]
    fn test_plasticity_modulates_learning_rate() {
        let mut fast = new_working_model();
        let mut slow = new_working_model();
        let action = CaregiverAction::reliable();

        for _ in 0..100 {
            update_working_model(&mut fast, &action, 1.0);
            update_working_model(&mut slow, &action, 0.1);
        }

        // High plasticity should converge faster
        assert!(
            fast.self_worthy > slow.self_worthy,
            "High plasticity should learn faster: fast={}, slow={}",
            fast.self_worthy,
            slow.self_worthy
        );
    }

    #[test]
    fn test_values_clamped() {
        let mut model = InternalWorkingModel {
            self_worthy: 0.99,
            other_reliable: 0.99,
            world_safe: 0.99,
            model_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let action = CaregiverAction::reliable();
        for _ in 0..10000 {
            update_working_model(&mut model, &action, 1.0);
        }
        assert!(model.self_worthy <= 1.0);
        assert!(model.other_reliable <= 1.0);
        assert!(model.world_safe <= 1.0);
    }

    #[test]
    fn test_iwm_convergence_stable() {
        let mut model = new_working_model();
        let action = CaregiverAction::reliable();
        for _ in 0..2000 {
            update_working_model(&mut model, &action, 1.0);
        }
        let sw_before = model.self_worthy;
        for _ in 0..100 {
            update_working_model(&mut model, &action, 1.0);
        }
        // Should be very close to converged
        assert!(
            (model.self_worthy - sw_before).abs() < 0.01,
            "Should be near convergence after 2000 steps"
        );
    }
}
