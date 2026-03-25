// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mastery tracking with Bayesian Knowledge Tracing (BKT).
//!
//! Contains: mastery record creation, practice attempt recording,
//! skill mastery queries, pagination, and review scheduling.
//!
//! Research: Corbett & Anderson (1994) - Knowledge Tracing
//!
//! Extracted from lib.rs as a pure structural refactor — no logic changes.

use hdk::prelude::*;
use adaptive_integrity::*;
use crate::{
    current_time, mastery_anchor,
    DEFAULT_LEARN_RATE, DEFAULT_GUESS_RATE, DEFAULT_SLIP_RATE,
    RETENTION_TARGET, MASTERY_THRESHOLD,
};

// ============== Types ==============

/// Input for creating a skill mastery record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateMasteryInput {
    pub skill_hash: ActionHash,
    /// Optional custom BKT parameters
    pub learn_rate: Option<u16>,
    pub guess_rate: Option<u16>,
    pub slip_rate: Option<u16>,
}

/// Input for recording a practice attempt
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordAttemptInput {
    pub mastery_hash: ActionHash,
    pub correct: bool,
    pub response_time_ms: u32,
}

/// Result of a mastery update
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MasteryUpdateResult {
    pub old_mastery_permille: u16,
    pub new_mastery_permille: u16,
    pub mastery_level: MasteryLevel,
    pub just_mastered: bool,
    pub total_attempts: u32,
    pub accuracy_permille: u16,
    pub next_review_minutes: u32,
}

/// Input for getting a specific skill's mastery
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetSkillMasteryInput {
    pub skill_hash: ActionHash,
}

/// Get paginated masteries with pre-computed stats
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaginatedMasteriesInput {
    /// Number of items to skip
    pub offset: usize,
    /// Maximum items to return
    pub limit: usize,
    /// Filter by minimum mastery
    pub min_mastery: Option<u16>,
    /// Filter by maximum mastery
    pub max_mastery: Option<u16>,
    /// Only return due for review
    pub due_only: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaginatedMasteriesResult {
    /// Masteries for this page
    pub items: Vec<SkillMastery>,
    /// Total count (before pagination)
    pub total: usize,
    /// Has more items after this page
    pub has_more: bool,
    /// Pre-computed quick stats
    pub quick_stats: QuickStats,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuickStats {
    pub total: usize,
    pub due_count: usize,
    pub avg_mastery: u16,
}

// ============== Functions ==============

/// Create or get mastery record for a skill
pub(crate) fn get_or_create_mastery_impl(input: CreateMasteryInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_time()?;
    let anchor = mastery_anchor()?;

    // Check if mastery already exists
    let links = get_links(
        LinkQuery::try_new(anchor.clone(), LinkTypes::LearnerToMasteries)?,
        GetStrategy::Local,
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                let mastery: SkillMastery = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No mastery entry".into())))?;

                if mastery.skill_hash == input.skill_hash {
                    return Ok(action_hash);
                }
            }
        }
    }

    // Create new mastery record
    let mastery = SkillMastery {
        learner: agent,
        skill_hash: input.skill_hash.clone(),
        mastery_permille: 0,
        learn_rate_permille: input.learn_rate.unwrap_or(DEFAULT_LEARN_RATE),
        guess_permille: input.guess_rate.unwrap_or(DEFAULT_GUESS_RATE),
        slip_permille: input.slip_rate.unwrap_or(DEFAULT_SLIP_RATE),
        total_attempts: 0,
        correct_attempts: 0,
        recent_attempts_bits: 0,
        recent_count: 0,
        minutes_since_practice: 0,
        decay_estimate_permille: 0,
        next_optimal_review: now,
        confidence_permille: 0,
        created_at: now,
        modified_at: now,
    };

    let action_hash = create_entry(EntryTypes::SkillMastery(mastery))?;

    // Link from learner anchor
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::LearnerToMasteries,
        (),
    )?;

    // Link from skill (for aggregation)
    create_link(
        input.skill_hash,
        action_hash.clone(),
        LinkTypes::SkillToMasteries,
        (),
    )?;

    Ok(action_hash)
}

/// Record a practice attempt and update mastery
pub(crate) fn record_attempt_impl(input: RecordAttemptInput) -> ExternResult<MasteryUpdateResult> {
    let now = current_time()?;

    // Get current mastery
    let record = get(input.mastery_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Mastery not found".into())))?;

    let mut mastery: SkillMastery = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No mastery entry".into())))?;

    let old_mastery = mastery.mastery_permille;

    // Update mastery using BKT
    let new_mastery = bkt_update(
        mastery.mastery_permille,
        input.correct,
        mastery.learn_rate_permille,
        mastery.guess_permille,
        mastery.slip_permille,
    );

    // Update stats
    mastery.mastery_permille = new_mastery;
    mastery.total_attempts += 1;
    if input.correct {
        mastery.correct_attempts += 1;
    }

    // Update recent attempts bitfield
    mastery.recent_attempts_bits = (mastery.recent_attempts_bits << 1) | (if input.correct { 1 } else { 0 });
    mastery.recent_count = (mastery.recent_count + 1).min(10);

    // Calculate next optimal review time
    let minutes_since = ((now - mastery.modified_at) / 60_000_000) as u32; // micros to minutes
    mastery.next_optimal_review = now + (optimal_review_time(
        new_mastery,
        minutes_since,
        RETENTION_TARGET,
    ) as i64 * 60_000_000); // minutes to micros

    // Update confidence based on attempt count
    mastery.confidence_permille = ((mastery.total_attempts * 50).min(1000)) as u16;
    mastery.modified_at = now;

    // Update the entry
    update_entry(input.mastery_hash, mastery.clone())?;

    let mastery_level = MasteryLevel::from_permille(new_mastery);
    let just_mastered = old_mastery < MASTERY_THRESHOLD && new_mastery >= MASTERY_THRESHOLD;

    let accuracy_permille = if mastery.total_attempts > 0 {
        (mastery.correct_attempts * 1000 / mastery.total_attempts) as u16
    } else {
        0
    };

    let next_review_minutes = ((mastery.next_optimal_review - now) / 60_000_000) as u32;

    Ok(MasteryUpdateResult {
        old_mastery_permille: old_mastery,
        new_mastery_permille: new_mastery,
        mastery_level,
        just_mastered,
        total_attempts: mastery.total_attempts,
        accuracy_permille,
        next_review_minutes,
    })
}

/// Get all mastery records for the learner
pub(crate) fn get_my_masteries_impl(_: ()) -> ExternResult<Vec<SkillMastery>> {
    let anchor = mastery_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::LearnerToMasteries)?,
        GetStrategy::Local,
    )?;

    let mut masteries = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                let mastery: SkillMastery = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No mastery entry".into())))?;
                masteries.push(mastery);
            }
        }
    }

    Ok(masteries)
}

/// Get mastery record for a specific skill
pub(crate) fn get_skill_mastery_impl(input: GetSkillMasteryInput) -> ExternResult<Option<SkillMastery>> {
    // Get all masteries and find the one matching the skill hash
    let masteries = get_my_masteries_impl(())?;

    Ok(masteries.into_iter().find(|m| m.skill_hash == input.skill_hash))
}

/// Get skills due for review
pub(crate) fn get_due_for_review_impl(_: ()) -> ExternResult<Vec<SkillMastery>> {
    let now = current_time()?;
    let masteries = get_my_masteries_impl(())?;

    let mut due: Vec<SkillMastery> = masteries
        .into_iter()
        .filter(|m| m.next_optimal_review <= now && m.mastery_permille > 0)
        .collect();

    // Sort by most urgent first
    due.sort_by(|a, b| a.next_optimal_review.cmp(&b.next_optimal_review));

    Ok(due)
}

/// Get paginated masteries with filtering - optimized for UI lists
pub(crate) fn get_masteries_paginated_impl(input: PaginatedMasteriesInput) -> ExternResult<PaginatedMasteriesResult> {
    let now = current_time()?;
    let all_masteries = get_my_masteries_impl(())?;

    // Apply filters
    let filtered: Vec<&SkillMastery> = all_masteries.iter()
        .filter(|m| {
            let min_ok = input.min_mastery.map(|min| m.mastery_permille >= min).unwrap_or(true);
            let max_ok = input.max_mastery.map(|max| m.mastery_permille <= max).unwrap_or(true);
            let due_ok = input.due_only.map(|due| !due || m.next_optimal_review <= now).unwrap_or(true);
            min_ok && max_ok && due_ok
        })
        .collect();

    let total = filtered.len();
    let items: Vec<SkillMastery> = filtered.into_iter()
        .skip(input.offset)
        .take(input.limit)
        .cloned()
        .collect();
    let has_more = input.offset + items.len() < total;

    // Compute quick stats from full list (not just page)
    let due_count = all_masteries.iter().filter(|m| m.next_optimal_review <= now).count();
    let avg_mastery = if !all_masteries.is_empty() {
        (all_masteries.iter().map(|m| m.mastery_permille as u32).sum::<u32>()
            / all_masteries.len() as u32) as u16
    } else {
        0
    };

    Ok(PaginatedMasteriesResult {
        items,
        total,
        has_more,
        quick_stats: QuickStats {
            total: all_masteries.len(),
            due_count,
            avg_mastery,
        },
    })
}
