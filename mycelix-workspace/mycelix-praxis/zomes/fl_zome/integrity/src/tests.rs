// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unit tests for FL integrity zome validation logic
//!
//! Tests validate entry validation functions for:
//! - FlUpdate
//! - FlRound
//! - PrivacyParams

use super::*;
use praxis_core::{ModelHash, RoundId, RoundState};

// =============================================================================
// Helper functions for creating test data
// =============================================================================

fn create_valid_fl_update() -> FlUpdate {
    FlUpdate {
        round_id: RoundId("test_round_1".to_string()),
        model_id: "model_v1".to_string(),
        parent_model_hash: ModelHash("abc123".to_string()),
        grad_commitment: vec![1, 2, 3, 4, 5],
        clipped_l2_norm: 1.0,
        local_val_loss: 0.5,
        sample_count: 100,
        timestamp: 1000,
    }
}

fn create_valid_fl_round() -> FlRound {
    FlRound {
        round_id: RoundId("round_1".to_string()),
        model_id: "model_v1".to_string(),
        state: RoundState::Discover,
        base_model_hash: ModelHash("base_abc".to_string()),
        min_participants: 5,
        max_participants: 20,
        current_participants: 0,
        aggregation_method: "trimmed_mean".to_string(),
        clip_norm: 1.0,
        started_at: 1000,
        completed_at: None,
        aggregated_model_hash: None,
        privacy_epsilon: Some(1.0),
        privacy_delta: Some(0.00001),
    }
}

fn create_valid_privacy_params() -> PrivacyParams {
    PrivacyParams {
        round_id: RoundId("round_1".to_string()),
        base_epsilon: 2.0,
        min_epsilon: 0.5,
        max_epsilon: 5.0,
        sensitivity_score: 0.5,
    }
}

// =============================================================================
// FlUpdate validation tests
// =============================================================================

#[cfg(test)]
mod fl_update_validation_tests {
    use super::*;

    #[test]
    fn test_valid_fl_update() {
        let update = create_valid_fl_update();
        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_update_negative_l2_norm() {
        let mut update = create_valid_fl_update();
        update.clipped_l2_norm = -1.0;

        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_update_zero_sample_count() {
        let mut update = create_valid_fl_update();
        update.sample_count = 0;

        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_update_negative_val_loss() {
        let mut update = create_valid_fl_update();
        update.local_val_loss = -0.5;

        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_update_infinite_val_loss() {
        let mut update = create_valid_fl_update();
        update.local_val_loss = f32::INFINITY;

        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_update_nan_val_loss() {
        let mut update = create_valid_fl_update();
        update.local_val_loss = f32::NAN;

        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_update_zero_l2_norm() {
        let mut update = create_valid_fl_update();
        update.clipped_l2_norm = 0.0;

        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_update_large_sample_count() {
        let mut update = create_valid_fl_update();
        update.sample_count = 1_000_000;

        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_update_empty_grad_commitment() {
        let mut update = create_valid_fl_update();
        update.grad_commitment = vec![];

        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_update_large_grad_commitment() {
        let mut update = create_valid_fl_update();
        update.grad_commitment = vec![0u8; 10000];

        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }
}

// =============================================================================
// FlRound validation tests
// =============================================================================

#[cfg(test)]
mod fl_round_validation_tests {
    use super::*;

    #[test]
    fn test_valid_fl_round() {
        let round = create_valid_fl_round();
        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_round_min_exceeds_max_participants() {
        let mut round = create_valid_fl_round();
        round.min_participants = 30;
        round.max_participants = 20;

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_zero_min_participants() {
        let mut round = create_valid_fl_round();
        round.min_participants = 0;

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_current_exceeds_max() {
        let mut round = create_valid_fl_round();
        round.current_participants = 25;
        round.max_participants = 20;

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_zero_clip_norm() {
        let mut round = create_valid_fl_round();
        round.clip_norm = 0.0;

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_negative_clip_norm() {
        let mut round = create_valid_fl_round();
        round.clip_norm = -1.0;

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_negative_epsilon() {
        let mut round = create_valid_fl_round();
        round.privacy_epsilon = Some(-1.0);

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_zero_epsilon() {
        let mut round = create_valid_fl_round();
        round.privacy_epsilon = Some(0.0);

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_delta_below_zero() {
        let mut round = create_valid_fl_round();
        round.privacy_delta = Some(-0.1);

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_delta_above_one() {
        let mut round = create_valid_fl_round();
        round.privacy_delta = Some(1.1);

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_delta_at_zero() {
        let mut round = create_valid_fl_round();
        round.privacy_delta = Some(0.0);

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_round_delta_at_one() {
        let mut round = create_valid_fl_round();
        round.privacy_delta = Some(1.0);

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_round_no_privacy_params() {
        let mut round = create_valid_fl_round();
        round.privacy_epsilon = None;
        round.privacy_delta = None;

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_round_equal_min_max_participants() {
        let mut round = create_valid_fl_round();
        round.min_participants = 10;
        round.max_participants = 10;

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_round_completed_with_hash() {
        let mut round = create_valid_fl_round();
        round.state = RoundState::Aggregate;
        round.completed_at = Some(2000);
        round.aggregated_model_hash = Some(ModelHash("aggregated_abc".to_string()));

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_round_large_clip_norm() {
        let mut round = create_valid_fl_round();
        round.clip_norm = 100.0;

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_round_many_participants() {
        let mut round = create_valid_fl_round();
        round.min_participants = 100;
        round.max_participants = 1000;
        round.current_participants = 500;

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }
}

// =============================================================================
// PrivacyParams validation tests
// =============================================================================

#[cfg(test)]
mod privacy_params_validation_tests {
    use super::*;

    #[test]
    fn test_valid_privacy_params() {
        let params = create_valid_privacy_params();
        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_privacy_params_negative_min_epsilon() {
        let mut params = create_valid_privacy_params();
        params.min_epsilon = -0.1;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_zero_min_epsilon() {
        let mut params = create_valid_privacy_params();
        params.min_epsilon = 0.0;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_negative_max_epsilon() {
        let mut params = create_valid_privacy_params();
        params.max_epsilon = -1.0;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_negative_base_epsilon() {
        let mut params = create_valid_privacy_params();
        params.base_epsilon = -0.5;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_min_exceeds_base() {
        let mut params = create_valid_privacy_params();
        params.min_epsilon = 3.0;
        params.base_epsilon = 2.0;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_base_exceeds_max() {
        let mut params = create_valid_privacy_params();
        params.base_epsilon = 6.0;
        params.max_epsilon = 5.0;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_sensitivity_negative() {
        let mut params = create_valid_privacy_params();
        params.sensitivity_score = -0.1;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_sensitivity_above_one() {
        let mut params = create_valid_privacy_params();
        params.sensitivity_score = 1.1;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_sensitivity_at_zero() {
        let mut params = create_valid_privacy_params();
        params.sensitivity_score = 0.0;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_privacy_params_sensitivity_at_one() {
        let mut params = create_valid_privacy_params();
        params.sensitivity_score = 1.0;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_privacy_params_equal_min_base_max() {
        let mut params = create_valid_privacy_params();
        params.min_epsilon = 1.0;
        params.base_epsilon = 1.0;
        params.max_epsilon = 1.0;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_privacy_params_large_epsilon_range() {
        let mut params = create_valid_privacy_params();
        params.min_epsilon = 0.01;
        params.base_epsilon = 5.0;
        params.max_epsilon = 100.0;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_privacy_params_small_epsilon_range() {
        let mut params = create_valid_privacy_params();
        params.min_epsilon = 0.9;
        params.base_epsilon = 1.0;
        params.max_epsilon = 1.1;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }
}

// =============================================================================
// Edge case tests
// =============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_fl_update_very_small_l2_norm() {
        let mut update = create_valid_fl_update();
        update.clipped_l2_norm = 0.000001;

        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_update_very_large_l2_norm() {
        let mut update = create_valid_fl_update();
        update.clipped_l2_norm = 1_000_000.0;

        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_round_very_small_epsilon() {
        let mut round = create_valid_fl_round();
        round.privacy_epsilon = Some(0.00001);

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_round_very_large_epsilon() {
        let mut round = create_valid_fl_round();
        round.privacy_epsilon = Some(100.0);

        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_privacy_params_very_narrow_range() {
        let mut params = create_valid_privacy_params();
        params.min_epsilon = 1.0;
        params.base_epsilon = 1.0001;
        params.max_epsilon = 1.0002;

        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }
}

// =============================================================================
// Security validation tests (is_finite guards)
// =============================================================================

#[cfg(test)]
mod security_validation_tests {
    use super::*;

    // ---- FlUpdate NaN/Inf guards ----

    #[test]
    fn test_fl_update_nan_l2_norm_rejected() {
        let mut update = create_valid_fl_update();
        update.clipped_l2_norm = f32::NAN;
        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_update_inf_l2_norm_rejected() {
        let mut update = create_valid_fl_update();
        update.clipped_l2_norm = f32::INFINITY;
        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_update_neg_inf_l2_norm_rejected() {
        let mut update = create_valid_fl_update();
        update.clipped_l2_norm = f32::NEG_INFINITY;
        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_update_valid_passes() {
        let update = create_valid_fl_update();
        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_fl_update_zero_samples_rejected() {
        let mut update = create_valid_fl_update();
        update.sample_count = 0;
        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_update_nan_val_loss_rejected() {
        let mut update = create_valid_fl_update();
        update.local_val_loss = f32::NAN;
        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_update_inf_val_loss_rejected() {
        let mut update = create_valid_fl_update();
        update.local_val_loss = f32::INFINITY;
        let result = validate_fl_update(update);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ---- FlRound NaN/Inf guards ----

    #[test]
    fn test_fl_round_nan_clip_norm_rejected() {
        let mut round = create_valid_fl_round();
        round.clip_norm = f32::NAN;
        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_inf_clip_norm_rejected() {
        let mut round = create_valid_fl_round();
        round.clip_norm = f32::INFINITY;
        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_nan_epsilon_rejected() {
        let mut round = create_valid_fl_round();
        round.privacy_epsilon = Some(f32::NAN);
        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_inf_epsilon_rejected() {
        let mut round = create_valid_fl_round();
        round.privacy_epsilon = Some(f32::INFINITY);
        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_nan_delta_rejected() {
        let mut round = create_valid_fl_round();
        round.privacy_delta = Some(f32::NAN);
        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_inf_delta_rejected() {
        let mut round = create_valid_fl_round();
        round.privacy_delta = Some(f32::INFINITY);
        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_fl_round_min_gt_max_rejected() {
        let mut round = create_valid_fl_round();
        round.min_participants = 10;
        round.max_participants = 3;
        let result = validate_fl_round(round);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ---- PrivacyParams NaN/Inf guards ----

    #[test]
    fn test_privacy_params_nan_epsilon_rejected() {
        let mut params = create_valid_privacy_params();
        params.base_epsilon = f32::NAN;
        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_inf_epsilon_rejected() {
        let mut params = create_valid_privacy_params();
        params.base_epsilon = f32::INFINITY;
        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_nan_min_epsilon_rejected() {
        let mut params = create_valid_privacy_params();
        params.min_epsilon = f32::NAN;
        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_nan_max_epsilon_rejected() {
        let mut params = create_valid_privacy_params();
        params.max_epsilon = f32::NAN;
        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_nan_sensitivity_rejected() {
        let mut params = create_valid_privacy_params();
        params.sensitivity_score = f32::NAN;
        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_inf_sensitivity_rejected() {
        let mut params = create_valid_privacy_params();
        params.sensitivity_score = f32::INFINITY;
        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_privacy_params_valid_passes() {
        let params = create_valid_privacy_params();
        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_privacy_params_wrong_order_rejected() {
        let mut params = create_valid_privacy_params();
        params.min_epsilon = 2.0; // min > base (base is 2.0, so min = base, but > would fail)
        params.base_epsilon = 1.0; // base < min
        params.max_epsilon = 3.0;
        let result = validate_privacy_params(params);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }
}
