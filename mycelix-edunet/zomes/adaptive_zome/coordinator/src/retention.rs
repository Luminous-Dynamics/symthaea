// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Retention prediction and review scheduling for spaced repetition.
//!
//! Contains: skill retention forecasting, batch predictions, optimal review scheduling.
//!
//! Research: Ebbinghaus (1885), Pimsleur (1967), Wozniak & Gorzelanczyk (1994) - Forgetting Curves
//!
//! Extracted from lib.rs as a pure structural refactor — no logic changes.

use hdk::prelude::*;
use adaptive_integrity::*;
use crate::mastery::{get_skill_mastery_impl, GetSkillMasteryInput};

// ============== Types ==============

/// Input for retention prediction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetentionPredictionInput {
    pub skill_hash: ActionHash,
}

/// Input for batch retention prediction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchRetentionInput {
    pub skill_hashes: Vec<ActionHash>,
}

/// Batch output for retention predictions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchRetentionResult {
    pub predictions: Vec<SkillRetentionPrediction>,
    pub overall_retention_permille: u16,
    pub skills_at_risk: u32,
    pub skills_healthy: u32,
}

/// Individual skill retention with hash
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillRetentionPrediction {
    pub skill_hash: ActionHash,
    pub prediction: RetentionPrediction,
    pub is_at_risk: bool,
}

/// Input for review schedule generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReviewScheduleInput {
    pub skill_hash: ActionHash,
    pub target_retention_permille: u16,
    pub forecast_days: u32,
}

/// Output for review schedule
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReviewScheduleResult {
    pub skill_hash: ActionHash,
    pub current_stability_minutes: u32,
    pub schedule_minutes: Vec<u32>,
    pub total_reviews: u32,
}

// ============== Functions ==============

/// Predict retention for a specific skill
/// Returns detailed retention forecast based on forgetting curve
pub(crate) fn predict_skill_retention_impl(input: RetentionPredictionInput) -> ExternResult<RetentionPrediction> {
    // Get the skill mastery
    let mastery = get_skill_mastery_impl(GetSkillMasteryInput { skill_hash: input.skill_hash })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Skill mastery not found".to_string()
        )))?;

    // Use minutes_since_practice from the mastery record directly
    let last_review_minutes_ago = mastery.minutes_since_practice;

    // Calculate last interval (estimate based on successful reviews)
    // More correct attempts suggest longer intervals have been used
    let last_interval_minutes = if mastery.correct_attempts > 0 {
        // Rough estimate: more correct attempts = longer intervals
        60 * mastery.correct_attempts.min(10)
    } else {
        0
    };

    // Estimate difficulty from accuracy (lower accuracy = higher difficulty)
    let accuracy_permille = if mastery.total_attempts > 0 {
        (mastery.correct_attempts * 1000 / mastery.total_attempts) as u16
    } else {
        500
    };
    let difficulty_permille = 1000 - accuracy_permille;

    Ok(predict_retention(
        mastery.mastery_permille,
        mastery.correct_attempts,
        difficulty_permille,
        last_review_minutes_ago,
        last_interval_minutes,
        mastery.confidence_permille,
    ))
}

/// Predict retention for multiple skills (batch operation)
/// Efficient for dashboards and analytics
pub(crate) fn predict_retention_batch_impl(input: BatchRetentionInput) -> ExternResult<BatchRetentionResult> {
    let mut predictions = Vec::with_capacity(input.skill_hashes.len());
    let mut total_retention: u32 = 0;
    let mut skills_at_risk: u32 = 0;
    let mut skills_healthy: u32 = 0;

    let mut skipped_count: u32 = 0;
    for skill_hash in input.skill_hashes {
        match predict_skill_retention_impl(RetentionPredictionInput { skill_hash: skill_hash.clone() }) {
            Ok(prediction) => {
                let is_at_risk = prediction.current_retention_permille < 800;
                if is_at_risk {
                    skills_at_risk += 1;
                } else {
                    skills_healthy += 1;
                }
                total_retention += prediction.current_retention_permille as u32;

                predictions.push(SkillRetentionPrediction {
                    skill_hash,
                    prediction,
                    is_at_risk,
                });
            }
            Err(e) => {
                // Log prediction failures instead of silently continuing
                // This helps identify skills with missing data or corrupted records
                skipped_count += 1;
                debug!(
                    "Failed to predict retention for skill {:?}: {:?} (skipped {} so far)",
                    skill_hash, e, skipped_count
                );
                continue;
            }
        }
    }

    // Log summary if there were prediction failures
    if skipped_count > 0 {
        warn!(
            "Batch retention prediction completed with {} skipped skills out of {} total",
            skipped_count, predictions.len() + skipped_count as usize
        );
    }

    let overall_retention = if !predictions.is_empty() {
        (total_retention / predictions.len() as u32) as u16
    } else {
        0
    };

    Ok(BatchRetentionResult {
        predictions,
        overall_retention_permille: overall_retention,
        skills_at_risk,
        skills_healthy,
    })
}

/// Generate optimal review schedule to maintain target retention
pub(crate) fn get_optimal_review_schedule_impl(input: ReviewScheduleInput) -> ExternResult<ReviewScheduleResult> {
    let prediction = predict_skill_retention_impl(RetentionPredictionInput {
        skill_hash: input.skill_hash.clone(),
    })?;

    let schedule = optimal_review_schedule(
        prediction.stability_minutes as f64,
        input.target_retention_permille,
        input.forecast_days,
    );

    Ok(ReviewScheduleResult {
        skill_hash: input.skill_hash,
        current_stability_minutes: prediction.stability_minutes,
        schedule_minutes: schedule.clone(),
        total_reviews: schedule.len() as u32,
    })
}
