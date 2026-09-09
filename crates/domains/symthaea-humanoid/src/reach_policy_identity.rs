// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic local identity for exact Reach evidence policies.
//!
//! Policy IDs are human/operator-facing labels. Qualification evidence must also
//! bind the actual threshold/fallback contents so reusing a label with changed
//! criteria cannot silently mix evidence lineages.

use crate::reach_execution_evidence::HumanoidReachCommandEvidencePolicy;
use crate::reach_outcome_evidence::HumanoidReachOutcomeEvidencePolicy;

pub fn humanoid_reach_command_policy_fingerprint(
    policy: &HumanoidReachCommandEvidencePolicy,
) -> u64 {
    if !valid_id(&policy.policy_id)
        || policy.subject_fingerprint == 0
        || !policy.maximum_full_dynamics_age_s.is_finite()
        || !policy.minimum_jacobian_confidence.is_finite()
        || !policy.minimum_goal_authority_scale.is_finite()
        || !policy.maximum_whole_body_objective_residual.is_finite()
        || !policy.maximum_joint_utilization.is_finite()
        || !policy.maximum_inverse_dynamics_violation.is_finite()
        || !policy.maximum_contact_dynamics_residual_nm.is_finite()
        || !policy.maximum_contact_acceleration_residual.is_finite()
        || !policy.maximum_contact_friction_utilization.is_finite()
        || !policy.maximum_floating_base_dynamics_residual.is_finite()
    {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_bytes(&mut hash, policy.policy_id.as_bytes());
    feed_u64(&mut hash, policy.subject_fingerprint);
    feed_u64(&mut hash, policy.maximum_full_dynamics_age_s.to_bits());
    feed_u64(&mut hash, policy.minimum_jacobian_confidence.to_bits());
    feed_u64(&mut hash, policy.minimum_goal_authority_scale.to_bits() as u64);
    feed_u64(&mut hash, policy.maximum_whole_body_objective_residual.to_bits());
    feed_u64(&mut hash, policy.maximum_joint_utilization.to_bits());
    feed_u64(&mut hash, policy.maximum_inverse_dynamics_violation.to_bits());
    feed_u64(&mut hash, policy.allow_inverse_dynamics_fallback as u64);
    feed_u64(&mut hash, policy.maximum_contact_dynamics_residual_nm.to_bits());
    feed_u64(&mut hash, policy.maximum_contact_acceleration_residual.to_bits());
    feed_u64(&mut hash, policy.maximum_contact_friction_utilization.to_bits());
    feed_u64(&mut hash, policy.allow_contact_dynamics_fallback as u64);
    feed_u64(&mut hash, policy.require_floating_base_model as u64);
    feed_u64(&mut hash, policy.maximum_floating_base_dynamics_residual.to_bits());
    feed_u64(&mut hash, policy.allow_floating_base_fallback as u64);
    feed_u64(&mut hash, policy.maximum_final_safety_interventions as u64);
    if hash == 0 { 1 } else { hash }
}

pub fn humanoid_reach_outcome_policy_fingerprint(
    policy: &HumanoidReachOutcomeEvidencePolicy,
) -> u64 {
    if !valid_id(&policy.policy_id)
        || policy.subject_fingerprint == 0
        || !policy.maximum_observation_age_s.is_finite()
        || !policy.maximum_elapsed_since_preparation_s.is_finite()
        || !policy.maximum_post_command_error_m.is_finite()
        || !policy.minimum_progress_m.is_finite()
        || !policy.minimum_fractional_progress.is_finite()
    {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_bytes(&mut hash, policy.policy_id.as_bytes());
    feed_u64(&mut hash, policy.subject_fingerprint);
    feed_u64(&mut hash, policy.maximum_observation_age_s.to_bits());
    feed_u64(&mut hash, policy.maximum_elapsed_since_preparation_s.to_bits());
    feed_u64(&mut hash, policy.maximum_post_command_error_m.to_bits());
    feed_u64(&mut hash, policy.minimum_progress_m.to_bits());
    feed_u64(&mut hash, policy.minimum_fractional_progress.to_bits());
    feed_u64(&mut hash, policy.allow_already_within_tolerance as u64);
    if hash == 0 { 1 } else { hash }
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

fn feed_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= byte as u64;
        *hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
}

fn feed_bytes(hash: &mut u64, bytes: &[u8]) {
    feed_u64(hash, bytes.len() as u64);
    for byte in bytes {
        *hash ^= *byte as u64;
        *hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command_policy() -> HumanoidReachCommandEvidencePolicy {
        HumanoidReachCommandEvidencePolicy {
            policy_id: "command-policy-v1".into(),
            subject_fingerprint: 42,
            maximum_full_dynamics_age_s: 0.05,
            minimum_jacobian_confidence: 0.8,
            minimum_goal_authority_scale: 0.5,
            maximum_whole_body_objective_residual: 0.2,
            maximum_joint_utilization: 1.0,
            maximum_inverse_dynamics_violation: 0.01,
            allow_inverse_dynamics_fallback: false,
            maximum_contact_dynamics_residual_nm: 0.05,
            maximum_contact_acceleration_residual: 0.05,
            maximum_contact_friction_utilization: 1.0,
            allow_contact_dynamics_fallback: false,
            require_floating_base_model: true,
            maximum_floating_base_dynamics_residual: 0.05,
            allow_floating_base_fallback: false,
            maximum_final_safety_interventions: 0,
        }
    }

    fn outcome_policy() -> HumanoidReachOutcomeEvidencePolicy {
        HumanoidReachOutcomeEvidencePolicy {
            policy_id: "outcome-policy-v1".into(),
            subject_fingerprint: 42,
            maximum_observation_age_s: 0.02,
            maximum_elapsed_since_preparation_s: 0.1,
            maximum_post_command_error_m: 0.05,
            minimum_progress_m: 0.01,
            minimum_fractional_progress: 0.2,
            allow_already_within_tolerance: true,
        }
    }

    #[test]
    fn command_threshold_change_changes_identity_even_when_id_is_reused() {
        let a = command_policy();
        let mut b = command_policy();
        b.maximum_inverse_dynamics_violation = 0.02;
        assert_ne!(
            humanoid_reach_command_policy_fingerprint(&a),
            humanoid_reach_command_policy_fingerprint(&b)
        );
    }

    #[test]
    fn outcome_threshold_change_changes_identity_even_when_id_is_reused() {
        let a = outcome_policy();
        let mut b = outcome_policy();
        b.maximum_post_command_error_m = 0.08;
        assert_ne!(
            humanoid_reach_outcome_policy_fingerprint(&a),
            humanoid_reach_outcome_policy_fingerprint(&b)
        );
    }
}
