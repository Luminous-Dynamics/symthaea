// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Helper functions for the adaptive learning coordinator zome.
//!
//! Extracted from lib.rs as a pure structural refactor — no logic changes.

use hdk::prelude::*;
use adaptive_integrity::*;

/// Convert a Holochain Timestamp to i64 (microseconds)
pub(crate) fn timestamp_to_i64(ts: Timestamp) -> i64 {
    ts.as_micros()
}

/// Get current time as i64
pub(crate) fn current_time() -> ExternResult<i64> {
    Ok(timestamp_to_i64(sys_time()?))
}

/// Get learner anchor hash for profiles
pub(crate) fn profile_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.profile.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToProfile)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for masteries
pub(crate) fn mastery_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.mastery.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToMasteries)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for goals
pub(crate) fn goal_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.goal.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToGoals)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for sessions
pub(crate) fn session_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.session.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToSessions)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for paths
pub(crate) fn path_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.path.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToPaths)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for recommendations
pub(crate) fn recommendation_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.rec.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToRecommendations)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for assessments
pub(crate) fn assessment_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.assessment.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToAssessments)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Helper to truncate text
pub(crate) fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len.saturating_sub(3)])
    }
}
